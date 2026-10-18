"""Capture a bounded, native FlashVSR+ CPU stream reference.

The command is deliberately opt-in (``--run``): loading the released DiT and
LQ projection is a multi-gigabyte operation.  The capture uses the pinned
upstream ``StreamState``, ``LQProjection.stream_forward`` and
``FlashVSRDiT.forward_stream`` directly.  It is a reference for the denoiser
trajectory only; it does not decode a video or qualify final image quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import types
from unittest.mock import patch
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from reference_paths import DIT_CHECKPOINT, LQ_CHECKPOINT, PROMPT_PATH, SOURCE_ROOT


ROOT = Path(__file__).resolve().parent
SOURCE = SOURCE_ROOT.resolve()
MODEL_ROOT = SOURCE.parent
DIT_WEIGHTS = Path(
    os.environ.get(
        "FLASHVSR_PLUS_TURBO_DIT",
        str(DIT_CHECKPOINT),
    )
).resolve()
LQ_WEIGHTS = Path(
    os.environ.get(
        "FLASHVSR_PLUS_TURBO_LQ",
        str(LQ_CHECKPOINT),
    )
).resolve()
PROMPT_WEIGHTS = PROMPT_PATH.resolve()
INPUT_ROOT = Path(
    os.environ.get(
        "FLASHVSR_PLUS_RGB_REFERENCE",
        str(ROOT.parent.parent / "external/rgb-reference"),
    )
).resolve()
INPUT_FRAMES = (INPUT_ROOT / "frames.rgb24").resolve()
INPUT_MANIFEST = (INPUT_ROOT / "manifest.json").resolve()
OUTPUT = Path(
    os.environ.get(
        "FLASHVSR_PLUS_VIDEO_OUTPUT",
        str(ROOT.parent.parent / "external/video-native-cpu"),
    )
).resolve()

SOURCE_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
SOURCE_FRAME_COUNT = 17
SOURCE_HEIGHT = 2160
SOURCE_WIDTH = 3840
TARGET_HEIGHT = 128
TARGET_WIDTH = 128
CHUNK_LATENTS = 4
TEMPORAL_DOWNSAMPLE = 4
LATENT_CHANNELS = 16
SPATIAL_DOWNSAMPLE = 8
TOKEN_SHAPE = (4, 8, 8)
DIT_LAYERS = 30
HEAD_DIM = 128
PROMPT_SEED = 1

if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

# The source package's top-level ``inference`` module eagerly imports its CLI
# pipeline, which in turn imports optional OpenCV video I/O.  This capture only
# needs the source config/preprocessing/state modules; a package stub keeps
# that import boundary CPU-only and does not replace any source implementation.
if "inference" not in sys.modules:
    inference_package = types.ModuleType("inference")
    inference_package.__path__ = [str(SOURCE / "inference")]
    inference_package.__package__ = "inference"
    sys.modules["inference"] = inference_package
try:  # pragma: no cover - depends on the lab environment
    import cv2 as _cv2  # noqa: F401
except ImportError:  # source input.py only needs cv2 when decoding a file
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))

# These are imported only from the pinned source tree.  In particular, the
# capture does not import a CUDA runner or hardware detector.
from inference.config import (  # noqa: E402
    HYBRID_B25_ORIGINAL_LAYERS,
    HYBRID_B25_REPLACEMENT_LAYERS,
)
from inference.presets import stream_config_for_mode  # noqa: E402
from inference.preprocessing import repeat_last_frame  # noqa: E402
from inference.streaming import StreamState  # noqa: E402
from model_io import load_model_checkpoint  # noqa: E402
from models import (  # noqa: E402
    BlockReplacementConfig,
    FlashVSRModelConfig,
    LQProjectionConfig,
    build_model,
)
from models.embeddings import rope_angles_3d, sinusoidal_embedding_1d  # noqa: E402
from inference.loading.model_bundle import FlowMatchInferenceScheduler  # noqa: E402
from plus_projection import authenticate_checkpoint  # noqa: E402
from plus_text import authenticate_prompt  # noqa: E402
from plus_weights import authenticate_dit  # noqa: E402
from plus_modes import resolve_plus_mode  # noqa: E402


EXPECTED_INPUT_STATUS = "pass_common_rgb_only_no_model"
EXPECTED_INPUT_FILE = "frames.rgb24"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    # Hash the owning bytes directly; UHD diagnostic hashing must not create
    # another multi-gigabyte copy of an already contiguous decoded video.
    return hashlib.sha256(np.ascontiguousarray(value)).hexdigest()


def bf16_bits(value: torch.Tensor) -> np.ndarray:
    if value.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 tensor, got {value.dtype}")
    return value.detach().contiguous().cpu().view(torch.uint16).numpy().copy()


def bf16_f32(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(dtype=torch.float32).cpu().numpy().copy()


def _bits_and_f32(value: torch.Tensor, name: str) -> dict[str, np.ndarray]:
    return {f"{name}_bf16": bf16_bits(value), f"{name}_f32": bf16_f32(value)}


def rope_arrays(
    temporal_offset: int,
    *,
    token_shape: tuple[int, int, int] = TOKEN_SHAPE,
) -> dict[str, np.ndarray]:
    """Return the upstream FP32 RoPE factors in token-major archive order."""
    if (not isinstance(token_shape, tuple) or len(token_shape) != 3
            or any(type(value) is not int or value <= 0 for value in token_shape)):
        raise ValueError("token_shape must be a tuple of three positive integers")
    angles = rope_angles_3d(
        token_shape,
        HEAD_DIM,
        torch.device("cpu"),
        temporal_offset=temporal_offset,
    )
    return {
        "rope_cos_f32": angles.double().cos().to(torch.float32).view(
            1, 1, angles.shape[0], -1
        ).permute(0, 2, 1, 3).contiguous().numpy(),
        "rope_sin_f32": angles.double().sin().to(torch.float32).view(
            1, 1, angles.shape[0], -1
        ).permute(0, 2, 1, 3).contiguous().numpy(),
    }


def source_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_clean_source() -> str:
    head = source_head()
    if head != SOURCE_COMMIT:
        raise RuntimeError(f"pinned source commit mismatch: {head} != {SOURCE_COMMIT}")
    result = subprocess.run(
        ["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"pinned source tree is not clean:\n{result.stdout}")
    return head


def load_rgb_reference(
    *,
    frames_path: Path = INPUT_FRAMES,
    manifest_path: Path = INPUT_MANIFEST,
    frame_count: int = SOURCE_FRAME_COUNT,
    target_hw: tuple[int, int] = (TARGET_HEIGHT, TARGET_WIDTH),
) -> tuple[np.ndarray, torch.Tensor, dict[str, Any]]:
    """Read only one 4K frame at a time from the existing RGB24 reference."""
    if (not isinstance(target_hw, tuple) or len(target_hw) != 2
            or any(type(value) is not int or value <= 0 for value in target_hw)):
        raise ValueError("target_hw must be a tuple of two positive integers")
    target_height, target_width = target_hw
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != EXPECTED_INPUT_STATUS:
        raise ValueError(
            f"unexpected RGB24 manifest status: {manifest.get('status')!r}"
        )
    if manifest.get("file") != EXPECTED_INPUT_FILE or frames_path.name != EXPECTED_INPUT_FILE:
        raise ValueError("RGB24 manifest/payload file identity mismatch")
    shape = tuple(manifest.get("shape", ()))
    if shape != (49, SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError(f"unexpected RGB24 manifest shape: {shape}")
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    if manifest.get("dtype") != "uint8" or manifest.get("bytes") != expected_bytes:
        raise ValueError("RGB24 payload does not match its pinned manifest")
    if frames_path.stat().st_size != expected_bytes:
        raise ValueError("RGB24 payload byte size does not match its pinned manifest")
    payload_hash = manifest.get("sha256")
    if not isinstance(payload_hash, str) or len(payload_hash) != 64:
        raise ValueError("RGB24 manifest is missing its payload hash")
    if frame_count <= 0 or frame_count > shape[0]:
        raise ValueError(f"frame_count must be in [1,{shape[0]}], got {frame_count}")

    # A memmap keeps the 1.2 GB source out of resident Python memory.  Each
    # selected frame is resized and released before the next one is touched.
    raw = np.memmap(
        frames_path,
        mode="r",
        dtype=np.uint8,
        shape=shape,
        order="C",
    )
    resized: list[np.ndarray] = []
    raw_hashes: list[str] = []
    expected_frames = manifest.get("frames", [])
    if not isinstance(expected_frames, list) or len(expected_frames) != shape[0]:
        raise ValueError("RGB24 manifest must contain one receipt per source frame")
    for index in range(frame_count):
        frame = np.asarray(raw[index])
        raw_hash = sha256_array(frame)
        expected = expected_frames[index].get("raw_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"RGB24 frame {index} is missing its raw hash")
        if raw_hash != expected:
            raise ValueError(f"RGB24 frame {index} raw hash mismatch")
        raw_hashes.append(raw_hash)
        # Resize the complete source frame without a center crop.  The caller
        # may then apply the model's explicit bottom/right edge padding.
        resized.append(
            np.asarray(
                Image.fromarray(frame, mode="RGB").resize(
                    (target_width, target_height),
                    resample=Image.Resampling.BICUBIC,
                )
            )
        )
    del raw
    rgb = np.ascontiguousarray(np.stack(resized, axis=0), dtype=np.uint8)
    normalized = (
        torch.from_numpy(rgb)
        .permute(0, 3, 1, 2)
        .float()
        .div(127.5)
        .sub(1.0)
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
        .contiguous()
    )
    receipt = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "payload_path": str(frames_path),
        "payload_bytes": int(frames_path.stat().st_size),
        "payload_sha256_expected": manifest.get("sha256"),
        "payload_sha256_verified": False,
        "payload_hash_note": "Full payload is intentionally not read; every consumed source frame is hash-verified.",
        "shape": list(shape),
        "dtype": "uint8",
        "consumed_frame_count": frame_count,
        "consumed_frame_raw_sha256": raw_hashes,
        "resized_rgb_sha256": sha256_array(rgb),
        "normalization": "float32 = uint8 / 127.5 - 1.0; CHW -> [1,3,F,H,W]",
        "geometry": [target_height, target_width],
        "resize": "PIL.Image.resize full frame with BICUBIC; no center crop",
    }
    return rgb, normalized, receipt


def plan_stream(frame_count: int = SOURCE_FRAME_COUNT) -> dict[str, Any]:
    if frame_count != SOURCE_FRAME_COUNT:
        raise ValueError("this bounded reference is intentionally fixed to 17 source frames")
    requested_latents = (frame_count - 1) // TEMPORAL_DOWNSAMPLE + 1
    padded_latents = ((requested_latents + CHUNK_LATENTS - 1) // CHUNK_LATENTS) * CHUNK_LATENTS
    projection_frames = 1 + padded_latents * TEMPORAL_DOWNSAMPLE
    calls = []
    for call in range(padded_latents // CHUNK_LATENTS):
        source_starts = [1 + (call * CHUNK_LATENTS + inner) * TEMPORAL_DOWNSAMPLE for inner in range(CHUNK_LATENTS)]
        source_ends = [start + TEMPORAL_DOWNSAMPLE for start in source_starts]
        calls.append(
            {
                "call": call,
                "latent_indices": list(range(call * CHUNK_LATENTS, (call + 1) * CHUNK_LATENTS)),
                "projection_frame_ranges": [[start, end] for start, end in zip(source_starts, source_ends)],
                "write_cache": call + 1 < padded_latents // CHUNK_LATENTS,
                "temporal_offset": call * CHUNK_LATENTS,
            }
        )
    return {
        "source_frames": frame_count,
        "requested_latents": requested_latents,
        "padded_latents": padded_latents,
        "projection_frames": projection_frames,
        "calls": calls,
    }


def build_native_model(*, config=None, mode: str = "turbo") -> tuple[torch.nn.Module, Any]:
    """Build the selected Plus model with one shared strict DiT/LQ loader."""
    spec = resolve_plus_mode(mode)
    paths = spec.helper_paths(MODEL_ROOT / "models")
    # Preserve existing explicit Turbo-path overrides; Base never falls back
    # to those paths or to Turbo's replacement-layer schedule.
    dit_path = DIT_WEIGHTS if mode == "turbo" else paths["dit"]
    lq_path = LQ_WEIGHTS if mode == "turbo" else paths["lq"]
    if not dit_path.is_file() or not lq_path.is_file():
        raise FileNotFoundError(f"missing {mode} checkpoints: {dit_path}, {lq_path}")
    config = config or stream_config_for_mode(
        mode,
        device="cpu",
        seed=PROMPT_SEED,
        hardware_profile="generic",
        attn_kernel="sdpa",
        decode_every="auto",
        tiny_decoder_mode="serial",
        tiny_decoder_channels_last=False,
        cache_offload_mode="never",
    )
    config.validate()
    if tuple(config.replacement_layers) != spec.replacement_layers:
        raise ValueError(f"replacement layer schedule does not match {mode}")
    model_config = FlashVSRModelConfig(
        attention=config.attention_config(),
        lq_projection=LQProjectionConfig(
            enabled=True,
        # The released Turbo LQ checkpoint contains one projection head.  Its
        # condition is injected into block 0; the remaining blocks receive no
        # separate LQ condition.
        num_injection_layers=1,
        ),
        replacement=BlockReplacementConfig(layers=config.replacement_layers),
    )
    model = build_model(model_config)
    # The released DiT file intentionally excludes the eight LQ-projection
    # tensors; load it with that exact, explicit omission and reject every
    # other missing/skipped/shape-mismatched key.  The projection is loaded
    # strictly in the next step, so no model parameter remains uninitialized.
    dit_report = load_model_checkpoint(model, dit_path, min_coverage=0.0)
    lq_report = load_model_checkpoint(
        model.LQ_proj_in,
        lq_path,
        component_prefix="LQ_proj_in.",
        min_coverage=1.0,
    )
    expected_dit_missing = {
        f"LQ_proj_in.{name}" for name in model.LQ_proj_in.state_dict()
    }
    if set(dit_report.missing) != expected_dit_missing:
        raise RuntimeError(
            "DiT checkpoint schema changed: expected only LQ projection keys "
            f"to be absent, got {dit_report.missing}"
        )
    for label, report in (("DiT", dit_report), ("LQ projection", lq_report)):
        if report.shape_mismatch or report.skipped:
            raise RuntimeError(
                f"{label} checkpoint schema is incomplete: "
                f"missing={report.missing[:4]}, shape_mismatch={report.shape_mismatch[:4]}, "
                f"skipped={report.skipped[:4]}"
            )
    model.to(device="cpu", dtype=torch.bfloat16).eval()
    reports = {
        "mode": mode,
        "dit_loaded": dit_report.loaded_count,
        "dit_missing": len(dit_report.missing),
        "lq_loaded": lq_report.loaded_count,
        "lq_missing": len(lq_report.missing),
    }
    return model, reports


def _context_tensor() -> torch.Tensor:
    value = torch.load(PROMPT_WEIGHTS, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError("prompt checkpoint must contain a tensor")
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"unexpected prompt shape; expected [1,T,C], got {tuple(value.shape)}")
    return value.contiguous()


def _project_conditions(
    model: torch.nn.Module,
    projection_video: torch.Tensor,
    call: int,
) -> list[torch.Tensor]:
    projection = model.LQ_proj_in
    conditions: list[torch.Tensor] | None = None
    if call == 0:
        first = projection_video[:, :, :1].to(dtype=torch.bfloat16)
        current = projection.stream_forward(first)
        if current is not None:
            raise RuntimeError("LQ projection warm-up unexpectedly returned conditions")
    for inner in range(CHUNK_LATENTS):
        latent_index = call * CHUNK_LATENTS + inner
        start = 1 + latent_index * TEMPORAL_DOWNSAMPLE
        clip = projection_video[:, :, start : start + TEMPORAL_DOWNSAMPLE].to(
            dtype=torch.bfloat16
        )
        current = projection.stream_forward(clip)
        if current is None or len(current) != 1:
            raise RuntimeError("LQ projection returned an incomplete layer list")
        if conditions is None:
            conditions = list(current)
        else:
            conditions = [torch.cat((old, new), dim=1) for old, new in zip(conditions, current)]
    if conditions is None:
        raise RuntimeError("LQ projection returned no conditions")
    return conditions


def _layer_observer(model: torch.nn.Module, records: dict[str, torch.Tensor]):
    """Install temporary wrappers around the source block stream methods."""
    import models.attention as attention_module

    active: dict[str, int | None] = {"layer": None}
    original_layout = attention_module.build_stream_sparse_layout

    def observe_layout(*args, **kwargs):
        layout = original_layout(*args, **kwargs)
        layer = active["layer"]
        if layer is not None:
            records[f"layer{layer}_source_mask"] = layout.block_mask.detach().cpu().clone()
            records[f"layer{layer}_cache_indices"] = layout.cache_indices.detach().cpu().clone()
        return layout

    originals: list[Any] = []
    for layer, block in enumerate(model.blocks):
        original = block.forward_stream
        originals.append(original)

        def wrapped(*args, _original=original, _layer=layer, **kwargs):
            if not args:
                raise RuntimeError("unexpected DiT block call without x")
            x = args[0]
            active["layer"] = _layer
            records[f"layer{_layer}_in"] = x.detach().cpu().clone()
            try:
                result = _original(*args, **kwargs)
            finally:
                active["layer"] = None
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("source DiT block stream result changed")
            records[f"layer{_layer}_out"] = result[0].detach().cpu().clone()
            next_cache = result[1]
            records[f"layer{_layer}_cache"] = next_cache
            return result

        block.forward_stream = wrapped

    class _Observer:
        def close(self) -> None:
            for block, original in zip(model.blocks, originals):
                block.forward_stream = original

        def __enter__(self):
            self.patch = torch.no_grad()
            self.patch.__enter__()
            self.layout_patch = patch.object(
                attention_module, "build_stream_sparse_layout", observe_layout
            )
            self.layout_patch.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.layout_patch.__exit__(exc_type, exc, tb)
            self.close()
            self.patch.__exit__(exc_type, exc, tb)

    return _Observer()


def _cache_receipt(cache: Any) -> dict[str, Any] | None:
    if cache is None:
        return None
    if not isinstance(cache, tuple) or len(cache) != 2:
        raise TypeError("unexpected stream cache")
    result = {}
    for name, value in zip(("k", "v"), cache):
        result[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256_array(bf16_bits(value)),
        }
    return result


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_array(value)}
            for name, value in arrays.items()
        },
    }


@torch.inference_mode()
def capture(output: Path = OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"capture output must be a fresh leaf: {output}")
    start_time = time.monotonic()
    require_clean_source()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        print("[capture] authenticating pinned source/checkpoints", flush=True)
        dit_auth = authenticate_dit(DIT_WEIGHTS)
        lq_auth = authenticate_checkpoint(LQ_WEIGHTS)
        prompt_auth = authenticate_prompt(PROMPT_WEIGHTS)
        checkpoint_receipts = {
            "dit": dit_auth,
            "lq_projection": {
                "path": str(LQ_WEIGHTS.resolve()),
                "bytes": int(LQ_WEIGHTS.stat().st_size),
                "sha256": lq_auth,
            },
            "prompt": prompt_auth,
        }
        print("[capture] checkpoint authentication complete", flush=True)
        print("[capture] loading and resizing first 17 source frames", flush=True)
        rgb, source_video, input_receipt = load_rgb_reference()
        plan = plan_stream()
        projection_video = repeat_last_frame(source_video, plan["projection_frames"])
        context_source = _context_tensor()
        print("[capture] building authenticated native BF16 model", flush=True)
        model, load_receipt = build_native_model()
        print("[capture] native model loaded; beginning call 0", flush=True)
        model.LQ_proj_in.clear_cache()
        context = context_source.to(dtype=torch.bfloat16)
        scheduler = FlowMatchInferenceScheduler(shift=5.0)
        timestep = scheduler.timesteps(1, torch.device("cpu"), torch.bfloat16)[0].reshape(1)
        embedded_context = model.text_embedding(context)
        time_embedding = model.time_embedding(
            sinusoidal_embedding_1d(model.freq_dim, timestep).to(dtype=torch.bfloat16)
        )
        time_modulation = model.time_projection(time_embedding).unflatten(1, (6, model.dim))
        rope_by_call = {call: rope_arrays(call * CHUNK_LATENTS) for call in range(2)}

        output.mkdir(parents=True)
        input_arrays = {
            "rgb_uint8": rgb,
            "source_video_f32": source_video.numpy().copy(),
            "projection_video_f32": projection_video.numpy().copy(),
            "prompt_context_bf16": bf16_bits(context_source),
            "prompt_context_f32": bf16_f32(context_source),
            "context_embedded_bf16": bf16_bits(embedded_context),
            "context_embedded_f32": bf16_f32(embedded_context),
            "timestep_bf16": bf16_bits(timestep.reshape(1)),
            "timestep_f32": bf16_f32(timestep.reshape(1)),
            "time_embedding_bf16": bf16_bits(time_embedding.unsqueeze(1)),
            "time_embedding_f32": bf16_f32(time_embedding.unsqueeze(1)),
            "time_modulation_bf16": bf16_bits(time_modulation),
            "time_modulation_f32": bf16_f32(time_modulation),
            "t_mod_bf16": bf16_bits(time_modulation),
            "t_mod_f32": bf16_f32(time_modulation),
            "rope_cos_call0_f32": rope_by_call[0]["rope_cos_f32"],
            "rope_sin_call0_f32": rope_by_call[0]["rope_sin_f32"],
            "rope_cos_call1_f32": rope_by_call[1]["rope_cos_f32"],
            "rope_sin_call1_f32": rope_by_call[1]["rope_sin_f32"],
        }
        input_info = _write_npz(output / "inputs.npz", input_arrays)
        call_infos = []
        generator = torch.Generator(device="cpu").manual_seed(PROMPT_SEED)
        spatial_windows = (TARGET_HEIGHT // 16 // 8) * (TARGET_WIDTH // 16 // 8)
        topk = int(spatial_windows * spatial_windows * 2.5) - 1
        state = StreamState(
            attention_topk=topk,
            attention_cache_blocks=3,
            attention_block_size=2,
            attention_block_m=128,
            release_attention_workspaces=True,
        )
        for call in range(2):
            print(f"[capture] call {call} projection/denoiser start", flush=True)
            conditions = _project_conditions(model, projection_video, call)
            current = torch.randn(
                1,
                LATENT_CHANNELS,
                CHUNK_LATENTS,
                TARGET_HEIGHT // SPATIAL_DOWNSAMPLE,
                TARGET_WIDTH // SPATIAL_DOWNSAMPLE,
                generator=generator,
                device="cpu",
                dtype=torch.bfloat16,
            )
            state.write_cache = call == 0
            records: dict[str, torch.Tensor] = {}
            patch_tokens, token_shape = model.patchify(current)
            if tuple(token_shape) != TOKEN_SHAPE:
                raise RuntimeError(f"unexpected patch token shape: {token_shape}")
            with _layer_observer(model, records):
                prediction = model.forward_stream(
                    current,
                    timestep.expand(current.shape[0]),
                    context,
                    layer_conditions=conditions,
                    state=state,
                )
            denoised = scheduler.step(prediction, timestep, current)
            arrays: dict[str, np.ndarray] = {}
            arrays.update(_bits_and_f32(current, "noise"))
            arrays.update(_bits_and_f32(patch_tokens, "patch_tokens"))
            arrays.update(rope_by_call[call])
            for layer, condition in enumerate(conditions):
                arrays.update(_bits_and_f32(condition, f"lq_condition{layer}"))
                if layer == 0:
                    arrays.update(_bits_and_f32(condition, "lq_condition"))
            for name, value in records.items():
                if name.endswith("_cache"):
                    continue
                if name.endswith("_source_mask") or name.endswith("_cache_indices"):
                    arrays[name] = value.numpy().copy()
                else:
                    arrays.update(_bits_and_f32(value, name))
            arrays.update(_bits_and_f32(prediction, "prediction"))
            arrays.update(_bits_and_f32(denoised, "denoised"))
            call_info = _write_npz(output / f"call{call}.npz", arrays)
            call_info.update(
                {
                    "call": call,
                    "temporal_offset": int(state.chunk_index * CHUNK_LATENTS),
                    "write_cache": bool(state.write_cache),
                    "cache_receipts": {
                        f"layer{layer}": _cache_receipt(records.get(f"layer{layer}_cache"))
                        for layer in range(DIT_LAYERS)
                    },
                    "noise_generator": {"device": "cpu", "seed": PROMPT_SEED, "draw_order": "call-major"},
                    "prediction_finite": bool(torch.isfinite(prediction).all()),
                    "denoised_finite": bool(torch.isfinite(denoised).all()),
                }
            )
            call_infos.append(call_info)
            print(
                f"[capture] call {call} complete; prediction_finite={call_info['prediction_finite']} "
                f"denoised_finite={call_info['denoised_finite']}",
                flush=True,
            )
            state.chunk_index = call + 1
            del conditions, current, patch_tokens, prediction, denoised
        source_files = {
            name: sha256_file(SOURCE / name)
            for name in (
                "inference/config.py",
                "inference/presets.py",
                "inference/preprocessing/input.py",
                "inference/preprocessing/padding.py",
                "inference/streaming/state.py",
                "models/config.py",
                "models/dit.py",
                "models/blocks.py",
                "models/attention.py",
                "models/attention_backends/sparse.py",
                "models/attention_backends/kernels.py",
                "models/conditioning.py",
                "models/embeddings.py",
            )
        }
        manifest = {
            "format": "flashvsr-plus-native-video-cpu-v1",
            "status": "cpu_native_denoiser_capture_complete",
            "quality_scope": "denoiser trajectory only; no decoder or image-quality qualification",
            "source": {
                "root": str(SOURCE),
                "commit": SOURCE_COMMIT,
                "files_sha256": source_files,
            },
            "checkpoints": checkpoint_receipts,
            "dtype": "torch.bfloat16",
            "threads": 2,
            "seed": PROMPT_SEED,
            "preset": {
                "mode": "turbo",
                "attention_kernel": "sdpa",
                "is_causal": True,
                "is_sparse": True,
                "is_hybrid_sliding": True,
                "block_size": 2,
                "global_layers": list(HYBRID_B25_ORIGINAL_LAYERS),
                "replacement_layers": list(HYBRID_B25_REPLACEMENT_LAYERS),
                "global_cache_blocks": 3,
                "local_cache_temporal_windows": 1,
                "sliding_local_range": 3,
                "sparse_local_range": 11,
                "sparse_topk_ratio": 2.5,
                "sparse_window_size": [2, 8, 8],
                "lq_projection_layers": 1,
                "temporal_compression": 4,
            },
            "input": input_receipt,
            "plan": plan,
            "load": load_receipt,
            "archive": {"inputs": input_info, "calls": call_infos},
            "runtime": {
                "wall_seconds": time.monotonic() - start_time,
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        files = sorted(path for path in output.iterdir() if path.is_file())
        lines = [f"{sha256_file(path)}  {path.name}" for path in files]
        (output / "results.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"[capture] complete output={output} wall_seconds={manifest['runtime']['wall_seconds']:.3f} "
            f"peak_rss_kib={manifest['runtime']['peak_rss_kib']}",
            flush=True,
        )
        return manifest
    finally:
        torch.set_num_threads(old_threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly run the multi-gigabyte CPU capture")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("refusing the heavy capture without --run")
    manifest = capture(args.output.resolve())
    print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
