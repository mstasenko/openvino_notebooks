"""Prepare authenticated FlashVSR+ common video inputs without model inference.

The artifact contains only resized/edge-padded source pixels, seeded per-call
BF16 noise chunks, the authenticated prompt, and source RoPE factors.  DiT/LQ
weights and learned timestep conditioning remain the responsibility of the
caller that consumes this ABI.
"""

from __future__ import annotations

import argparse
import json
import resource
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from capture_plus_video_cpu import (
    CHUNK_LATENTS,
    INPUT_FRAMES,
    INPUT_MANIFEST,
    LATENT_CHANNELS,
    PROMPT_SEED,
    SOURCE,
    SOURCE_COMMIT,
    SOURCE_FRAME_COUNT,
    SPATIAL_DOWNSAMPLE,
    bf16_f32,
    load_rgb_reference,
    repeat_last_frame,
    require_clean_source,
    rope_arrays,
    sha256_array,
    sha256_file,
)
from frame_geometry import FrameGeometry
from plus_text import PROMPT_PATH, authenticate_prompt, load_prompt
from plus_stream import build_stream_plan


FORMAT = "flashvsr-plus-common-video-input-v1"
FORMAT_V2 = "flashvsr-plus-common-video-input-v2"
STATUS = "common_input_preparation_complete"
MAX_SOURCE_FRAME_COUNT = 49
TOKEN_HEAD_DIM = 64
ARRAY_NAMES = (
    "projection_video_f32",
    "noise0_f32",
    "noise1_f32",
    "prompt_context_f32",
    "rope_cos_call0_f32",
    "rope_sin_call0_f32",
    "rope_cos_call1_f32",
    "rope_sin_call1_f32",
)
SOURCE_FILES = (
    "inference/config.py",
    "inference/presets.py",
    "inference/preprocessing/padding.py",
    "models/embeddings.py",
)
HOST_PREFLIGHT_HEADROOM_BYTES = 2 << 30


def call_array_names(call: int) -> tuple[str, str, str]:
    """Return the noise/cosine/sine fields owned by one stream call."""

    if type(call) is not int or call < 0:
        raise ValueError("call must be a non-negative integer")
    return (
        f"noise{call}_f32",
        f"rope_cos_call{call}_f32",
        f"rope_sin_call{call}_f32",
    )


def expected_array_names(call_count: int, *, legacy: bool = False) -> tuple[str, ...]:
    """Return the authenticated NPZ order for a fixed number of calls.

    The v1 tuple is intentionally preserved byte-for-byte.  v2 appends later
    call fields to that tuple, so a consumer can read the first two calls with
    the original names while still rejecting an omitted/reordered member.
    """

    if type(call_count) is not int or call_count < 2:
        raise ValueError("call_count must be an integer >= 2")
    if legacy:
        if call_count != 2:
            raise ValueError("legacy common input requires exactly two calls")
        return ARRAY_NAMES
    names = list(ARRAY_NAMES)
    for call in range(2, call_count):
        names.extend(call_array_names(call))
    return tuple(names)


def _validate_source_frame_count(value: Any) -> int:
    if type(value) is not int or not SOURCE_FRAME_COUNT <= value <= MAX_SOURCE_FRAME_COUNT:
        raise ValueError(
            f"source frame count must be an integer in [{SOURCE_FRAME_COUNT}, {MAX_SOURCE_FRAME_COUNT}]"
        )
    return value


def _metadata_equal_exact(actual: Any, expected: Any) -> bool:
    """Compare JSON metadata without Python's bool/int equality loophole."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (tuple(actual) == tuple(expected)
                and all(_metadata_equal_exact(actual[key], expected[key])
                        for key in expected))
    if isinstance(expected, list):
        return (len(actual) == len(expected)
                and all(_metadata_equal_exact(left, right)
                        for left, right in zip(actual, expected)))
    return actual == expected


def _plan_metadata(source_frame_count: int) -> dict[str, Any]:
    """Recompute the bounded upstream stream schedule from source count."""

    source_frame_count = _validate_source_frame_count(source_frame_count)
    plan = build_stream_plan(source_frame_count, fps=1.0)
    calls: list[dict[str, Any]] = []
    for call in plan.iter_calls():
        calls.append({
            "call": call.index,
            "latent_start": call.latent_start,
            "latent_stop": call.latent_stop,
            "frame_offset": call.frame_offset,
            "projection_start": call.projection_start,
            "projection_stop": call.projection_stop,
            "projection_source_indices": list(call.projection_source_indices),
            "decoder_start": call.decoder_start,
            "decoder_stop": call.decoder_stop,
            "decoder_source_indices": list(call.decoder_source_indices),
            "merged_output_start": call.merged_output_start,
            "merged_output_stop": call.merged_output_stop,
            "last": call.last,
        })
    return {
        "source_frames": plan.source_frames,
        "effective_frames": plan.effective_frames,
        "expanded_frames": plan.expanded_frames,
        "requested_latents": plan.requested_latents,
        "padded_latents": plan.padded_latents,
        "call_count": plan.call_count,
        "projection_frames": plan.projection_frames,
        "decoder_frames": plan.decoder_frames,
        "merged_decoder_frames": plan.merged_decoder_frames,
        "output_frames": plan.output_frames,
        "temporal_compression": plan.temporal_compression,
        "chunk_latents": CHUNK_LATENTS,
        "calls": calls,
    }


def input_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    """Authenticate and recompute the temporal plan represented by a manifest.

    v1 manifests predate explicit plan metadata, so their fixed 17-frame plan
    is returned from the same source-of-truth arithmetic.  v2 must carry the
    complete bounded schedule and exact member metadata; no caller may silently
    sort or infer a different schedule from NPZ member names.
    """

    if not isinstance(manifest, dict):
        raise ValueError("common input manifest must be an object")
    format_name = manifest.get("format")
    if (format_name not in (FORMAT, FORMAT_V2)
            or manifest.get("status") != STATUS):
        raise ValueError("common input manifest format changed")
    source_count = _validate_source_frame_count(manifest.get("source_frame_count"))
    expected = _plan_metadata(source_count)
    if format_name == FORMAT:
        if source_count != SOURCE_FRAME_COUNT:
            raise ValueError("legacy common input source frame count changed")
        if "plan" in manifest:
            raise ValueError("legacy common input unexpectedly contains a plan")
    else:
        supplied = manifest.get("plan")
        if not _metadata_equal_exact(supplied, expected):
            raise ValueError("common input temporal plan changed")
    return expected


def _hex(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def parse_frame_size(value: str) -> tuple[int, int]:
    """Parse the CLI's visible ``WIDTHxHEIGHT`` geometry."""

    if not isinstance(value, str) or re.fullmatch(r"[0-9]+x[0-9]+", value) is None:
        raise ValueError("frame size must be WIDTHxHEIGHT")
    width, height = (int(part) for part in value.split("x"))
    geometry = FrameGeometry(height, width)
    return geometry.height, geometry.width


def _pad_rgb(rgb: np.ndarray, geometry: FrameGeometry) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 4 or rgb.shape[1:] != (geometry.height, geometry.width, 3):
        expected = f"(N, {geometry.height}, {geometry.width}, 3)"
        raise ValueError(f"source RGB must be uint8 {expected}, got {rgb.shape} {rgb.dtype}")
    _validate_source_frame_count(int(rgb.shape[0]))
    if rgb.dtype != np.uint8:
        raise ValueError(f"source RGB must be uint8, got {rgb.shape} {rgb.dtype}")
    if not np.isfinite(rgb).all():
        raise ValueError("source RGB contains non-finite values")
    working_h, working_w = geometry.working_hw
    return np.pad(
        rgb,
        ((0, 0), (0, working_h - geometry.height),
         (0, working_w - geometry.width), (0, 0)),
        mode="edge",
    )


def build_arrays(
    rgb: np.ndarray,
    prompt_context_f32: np.ndarray,
    geometry: FrameGeometry,
    *,
    seed: int = PROMPT_SEED,
) -> dict[str, np.ndarray]:
    """Build the common ABI from visible resized RGB and an F32 prompt."""

    padded = _pad_rgb(rgb, geometry)
    source_frame_count = int(padded.shape[0])
    plan = _plan_metadata(source_frame_count)
    prompt = np.asarray(prompt_context_f32)
    if prompt.shape != (1, 512, 4096) or prompt.dtype != np.float32:
        raise ValueError("prompt_context_f32 must be finite FP32 [1,512,4096]")
    if not np.isfinite(prompt).all():
        raise ValueError("prompt_context_f32 contains non-finite values")

    source_video = (
        np.ascontiguousarray(padded.transpose(3, 0, 1, 2), dtype=np.float32)
        / np.float32(127.5)
        - np.float32(1.0)
    )[None, ...]
    source_tensor = torch.from_numpy(source_video)
    projection = repeat_last_frame(source_tensor, plan["projection_frames"]).numpy().copy()
    working_h, working_w = geometry.working_hw
    latent_h, latent_w = working_h // SPATIAL_DOWNSAMPLE, working_w // SPATIAL_DOWNSAMPLE
    token_shape = (CHUNK_LATENTS, working_h // 16, working_w // 16)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    generated: dict[str, np.ndarray] = {
        "projection_video_f32": np.ascontiguousarray(projection, dtype=np.float32),
        "prompt_context_f32": np.ascontiguousarray(prompt, dtype=np.float32),
    }
    for call in range(plan["call_count"]):
        noise = torch.randn(
            1,
            LATENT_CHANNELS,
            CHUNK_LATENTS,
            latent_h,
            latent_w,
            generator=generator,
            device="cpu",
            dtype=torch.bfloat16,
        )
        generated[f"noise{call}_f32"] = bf16_f32(noise)
        factors = rope_arrays(call * CHUNK_LATENTS, token_shape=token_shape)
        generated[f"rope_cos_call{call}_f32"] = np.ascontiguousarray(
            factors["rope_cos_f32"], dtype=np.float32
        )
        generated[f"rope_sin_call{call}_f32"] = np.ascontiguousarray(
            factors["rope_sin_f32"], dtype=np.float32
        )
    names = expected_array_names(plan["call_count"], legacy=source_frame_count == SOURCE_FRAME_COUNT)
    arrays = {name: generated[name] for name in names}
    if tuple(arrays) != names:
        raise AssertionError("common input field order changed")
    for name, value in arrays.items():
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid common input array: {name}")
    return arrays


def _array_info(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": sha256_array(value),
    }


def _source_receipt(source_rgb: dict[str, Any], padded_rgb: np.ndarray) -> dict[str, Any]:
    receipt = dict(source_rgb)
    receipt["working_rgb_sha256"] = sha256_array(padded_rgb)
    receipt["source_policy"] = "original authenticated RGB24 frames; full-frame bicubic resize; bottom/right edge padding; no crop"
    return receipt


def _validate_v2_source_receipt(source_rgb: dict[str, Any], source_frame_count: int) -> None:
    """Check the selected raw-frame hashes against the pinned source manifest."""

    if (source_rgb.get("manifest_sha256") != sha256_file(INPUT_MANIFEST)
            or source_rgb.get("payload_path") != str(INPUT_FRAMES)
            or source_rgb.get("manifest_path") != str(INPUT_MANIFEST)):
        raise ValueError("common input source manifest receipt changed")
    pinned = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    frames = pinned.get("frames")
    if (pinned.get("status") != "pass_common_rgb_only_no_model"
            or pinned.get("file") != INPUT_FRAMES.name
            or pinned.get("dtype") != "uint8"
            or not isinstance(frames, list)
            or len(frames) < source_frame_count):
        raise ValueError("pinned source manifest is invalid")
    selected_frames = frames[:source_frame_count]
    if any(not isinstance(frame, dict) for frame in selected_frames):
        raise ValueError("pinned source manifest frame receipt is invalid")
    expected = [frame.get("raw_sha256") for frame in selected_frames]
    actual = source_rgb.get("consumed_frame_raw_sha256")
    if (not _metadata_equal_exact(actual, expected) or any(
        not isinstance(value, str) or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in actual
    )):
        raise ValueError("common input consumed source hashes changed")
    if source_rgb.get("payload_sha256_expected") != pinned.get("sha256"):
        raise ValueError("common input source payload hash receipt changed")
    expected_selection = {"start_frame": 0, "count": source_frame_count}
    expected_pts = [frame.get("source_pts") for frame in selected_frames]
    if (not _metadata_equal_exact(source_rgb.get("selection"), expected_selection)
            or not _metadata_equal_exact(source_rgb.get("source_pts"), expected_pts)):
        raise ValueError("common input source timeline provenance changed")


def _manifest(
    output: Path,
    arrays: dict[str, np.ndarray],
    geometry: FrameGeometry,
    source_rgb: dict[str, Any],
    prompt: dict[str, Any],
    *,
    started: float,
) -> dict[str, Any]:
    source_frame_count = _validate_source_frame_count(
        source_rgb.get("consumed_frame_count")
    )
    plan = _plan_metadata(source_frame_count)
    legacy = source_frame_count == SOURCE_FRAME_COUNT
    working_h, working_w = geometry.working_hw
    token_count = CHUNK_LATENTS * (working_h // 16) * (working_w // 16)
    source_files = {
        name: sha256_file(SOURCE / name)
        for name in SOURCE_FILES
    }
    input_path = output / "inputs.npz"
    input_info = {
        "path": str(input_path.resolve()),
        "sha256": sha256_file(input_path),
        "arrays": {name: _array_info(arrays[name]) for name in arrays},
    }
    manifest = {
        "format": FORMAT if legacy else FORMAT_V2,
        "status": STATUS,
        "source": {
            "root": str(SOURCE),
            "commit": SOURCE_COMMIT,
            "files_sha256": source_files,
        },
        "source_frame_count": source_frame_count,
        "visible_hw": [geometry.height, geometry.width],
        "working_hw": [working_h, working_w],
        "frame_geometry": {
            "visible_hw": [geometry.height, geometry.width],
            "working_hw": [working_h, working_w],
            "padding": "bottom/right edge",
        },
        "conditioning": {
            "timestep": 1000.0,
            "temporal_downsample": 4,
            "latent_channels": LATENT_CHANNELS,
            "latent_shape_per_call": [1, LATENT_CHANNELS, CHUNK_LATENTS, working_h // 8, working_w // 8],
            "token_shape_per_call": [CHUNK_LATENTS, working_h // 16, working_w // 16],
            "rope_head_dim": TOKEN_HEAD_DIM,
        },
        "noise": {
            "seed": PROMPT_SEED,
            "device": "cpu",
            "dtype_generated": "torch.bfloat16",
            "draw_order": "call-major",
            "call_count": plan["call_count"],
        },
        "source_rgb": source_rgb,
        "prompt": prompt,
        "inputs": input_info,
        "worker_sha256": sha256_file(Path(__file__)),
        "runtime": {
            "threads": 2,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }
    if not legacy:
        manifest["conditioning"].update({
            "chunk_latents": CHUNK_LATENTS,
            "call_count": plan["call_count"],
            "projection_frames": plan["projection_frames"],
            "decoder_frames": plan["decoder_frames"],
        })
        manifest["plan"] = plan
    return manifest


def _expected_shapes(
    working_hw: tuple[int, int],
    *,
    source_frame_count: int = SOURCE_FRAME_COUNT,
) -> dict[str, tuple[int, ...]]:
    source_frame_count = _validate_source_frame_count(source_frame_count)
    plan = _plan_metadata(source_frame_count)
    height, width = working_hw
    token_shape = (CHUNK_LATENTS, height // 16, width // 16)
    token_count = int(np.prod(token_shape))
    expected = {
        "projection_video_f32": (1, 3, plan["projection_frames"], height, width),
        "prompt_context_f32": (1, 512, 4096),
    }
    for call in range(plan["call_count"]):
        noise_name, cos_name, sin_name = call_array_names(call)
        expected[noise_name] = (
            1, LATENT_CHANNELS, CHUNK_LATENTS, height // 8, width // 8
        )
        expected[cos_name] = (1, token_count, 1, TOKEN_HEAD_DIM)
        expected[sin_name] = (1, token_count, 1, TOKEN_HEAD_DIM)
    names = expected_array_names(plan["call_count"], legacy=source_frame_count == SOURCE_FRAME_COUNT)
    return {name: expected[name] for name in names}


def _estimated_peak_bytes(geometry: FrameGeometry, source_frame_count: int) -> int:
    """Conservative host allocation estimate used before reading source frames."""

    source_frame_count = _validate_source_frame_count(source_frame_count)
    plan = _plan_metadata(source_frame_count)
    height, width = geometry.working_hw
    source_rgb = source_frame_count * geometry.height * geometry.width * 3
    source_f32 = source_frame_count * 3 * height * width * 4
    projection = 3 * plan["projection_frames"] * height * width * 4
    prompt = 512 * 4096 * 4
    noise = plan["call_count"] * LATENT_CHANNELS * CHUNK_LATENTS * (height // 8) * (width // 8) * 4
    rope = plan["call_count"] * 2 * plan["chunk_latents"] * (height // 16) * (width // 16) * TOKEN_HEAD_DIM * 4
    # Source tensors and projection are live while the generated fields are
    # built; one extra projection-sized allowance covers conversion temporaries.
    return int(source_rgb + source_f32 + (2 * projection) + prompt + noise + rope)


def _available_host_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[1].isdigit() and fields[2] == "kB":
                    return int(fields[1]) * 1024
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        return None
    return None


def _preflight_host_memory(geometry: FrameGeometry, source_frame_count: int) -> None:
    """Reject a UHD preparation before source/array allocations if RAM is tight."""

    estimate = _estimated_peak_bytes(geometry, source_frame_count)
    # Small fixtures do not need a host-memory policy check.  UHD-sized
    # artifacts do, because a source mmap is followed by several F32 owners.
    if estimate < (1 << 30):
        return
    available = _available_host_bytes()
    if available is not None and available < estimate + HOST_PREFLIGHT_HEADROOM_BYTES:
        raise MemoryError(
            "insufficient host memory for common input preparation: "
            f"estimated {estimate} bytes, available {available} bytes, "
            f"headroom {HOST_PREFLIGHT_HEADROOM_BYTES} bytes"
        )


def write_artifact(
    output: Path,
    arrays: dict[str, np.ndarray],
    geometry: FrameGeometry,
    source_rgb: dict[str, Any],
    prompt: dict[str, Any],
    *,
    started: float = 0.0,
) -> dict[str, Any]:
    """Write one fresh common-input leaf and its authenticated manifest."""

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"common input output must be fresh: {output}")
    source_frame_count = _validate_source_frame_count(
        source_rgb.get("consumed_frame_count")
    )
    plan = _plan_metadata(source_frame_count)
    if source_frame_count != SOURCE_FRAME_COUNT:
        _validate_v2_source_receipt(source_rgb, source_frame_count)
    names = expected_array_names(
        plan["call_count"], legacy=source_frame_count == SOURCE_FRAME_COUNT
    )
    expected = _expected_shapes(
        geometry.working_hw, source_frame_count=source_frame_count
    )
    if tuple(arrays) != names or any(
        not isinstance(arrays[name], np.ndarray)
        or arrays[name].dtype != np.float32
        or arrays[name].shape != expected[name]
        or not np.isfinite(arrays[name]).all()
        for name in names
    ):
        raise ValueError("common input array ABI changed")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "inputs.npz", **arrays)
    manifest = _manifest(output, arrays, geometry, source_rgb, prompt, started=started)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def prepare(
    output: Path,
    *,
    frame_size: str,
    frame_count: int = SOURCE_FRAME_COUNT,
) -> dict[str, Any]:
    _validate_source_frame_count(frame_count)
    height, width = parse_frame_size(frame_size)
    geometry = FrameGeometry(height, width)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"common input output must be fresh: {output}")
    started = time.monotonic()
    require_clean_source()
    _preflight_host_memory(geometry, frame_count)
    prompt_context, prompt_receipt = load_prompt(PROMPT_PATH)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        rgb, _, source_rgb = load_rgb_reference(
            frame_count=frame_count,
            target_hw=(geometry.height, geometry.width),
        )
        padded = _pad_rgb(rgb, geometry)
        arrays = build_arrays(rgb, prompt_context, geometry, seed=PROMPT_SEED)
        del rgb
        source_rgb = _source_receipt(source_rgb, padded)
        if frame_count != SOURCE_FRAME_COUNT:
            source_manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
            source_frames = source_manifest.get("frames")
            if not isinstance(source_frames, list) or len(source_frames) < frame_count:
                raise ValueError("authenticated source manifest frame receipt is incomplete")
            source_rgb["selection"] = {"start_frame": 0, "count": frame_count}
            source_rgb["source_pts"] = [
                frame.get("source_pts") for frame in source_frames[:frame_count]
            ]
            if any(type(value) is not int for value in source_rgb["source_pts"]):
                raise ValueError("authenticated source manifest timestamps are invalid")
        prompt_receipt = {
            **prompt_receipt,
            "array_dtype": "float32",
            "array_shape": list(prompt_context.shape),
            "array_sha256": sha256_array(prompt_context),
        }
        return write_artifact(
            output,
            arrays,
            geometry,
            source_rgb,
            prompt_receipt,
            started=started,
        )
    finally:
        torch.set_num_threads(old_threads)


def load_input(
    root: Path | str,
    manifest_sha256: str,
    fields: tuple[str, ...] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Authenticate a common-input artifact and optionally materialize selected fields."""

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != _hex(manifest_sha256, "manifest_sha256"):
        raise ValueError("common input manifest fingerprint changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") not in (FORMAT, FORMAT_V2)
            or manifest.get("status") != STATUS):
        raise ValueError("common input manifest format/status changed")
    legacy = manifest.get("format") == FORMAT
    source_frame_count = _validate_source_frame_count(manifest.get("source_frame_count"))
    if legacy and source_frame_count != SOURCE_FRAME_COUNT:
        raise ValueError("legacy common input source frame count changed")
    plan = input_plan(manifest)
    expected_names = expected_array_names(
        plan["call_count"], legacy=legacy
    )
    if manifest.get("source", {}).get("commit") != SOURCE_COMMIT:
        raise ValueError("common input source commit changed")
    source = manifest.get("source", {})
    if source.get("root") != str(SOURCE):
        raise ValueError("common input source root changed")
    if require_clean_source() != SOURCE_COMMIT:
        raise ValueError("common input source revision changed")
    source_files = source.get("files_sha256")
    if not isinstance(source_files, dict) or tuple(source_files) != SOURCE_FILES:
        raise ValueError("common input source file receipt changed")
    for name in SOURCE_FILES:
        if sha256_file(SOURCE / name) != source_files[name]:
            raise ValueError(f"common input source file changed: {name}")
    visible = manifest.get("visible_hw")
    working = manifest.get("working_hw")
    if (not isinstance(visible, list) or not isinstance(working, list)
            or len(visible) != 2 or len(working) != 2
            or any(type(value) is not int for value in (*visible, *working))):
        raise ValueError("common input geometry metadata is invalid")
    geometry = FrameGeometry(*visible)
    if list(geometry.working_hw) != working:
        raise ValueError("common input working geometry does not match FrameGeometry")
    input_info = manifest.get("inputs")
    if not isinstance(input_info, dict):
        raise ValueError("common input archive receipt is missing")
    input_path = root / "inputs.npz"
    if Path(input_info.get("path", "")).resolve() != input_path:
        raise ValueError("common input archive path changed")
    if sha256_file(input_path) != input_info.get("sha256"):
        raise ValueError("common input archive fingerprint changed")
    arrays_info = input_info.get("arrays")
    if not isinstance(arrays_info, dict) or tuple(arrays_info) != expected_names:
        raise ValueError("common input field order changed")
    expected = _expected_shapes(
        tuple(working), source_frame_count=source_frame_count
    )
    if fields is None:
        selected_names = frozenset(expected_names)
    else:
        if type(fields) is not tuple or not fields:
            raise ValueError("common input field selection must be a non-empty tuple")
        if any(type(name) is not str for name in fields):
            raise ValueError("common input field selection names are invalid")
        if len(set(fields)) != len(fields):
            raise ValueError("common input field selection contains duplicates")
        if any(name not in expected_names for name in fields):
            raise ValueError("common input field selection contains unknown field")
        selected_names = frozenset(fields)
    for name in expected_names:
        metadata = arrays_info[name]
        if (not isinstance(metadata, dict)
                or metadata.get("shape") != list(expected[name])
                or metadata.get("dtype") != "float32"):
            raise ValueError(f"common input field metadata changed: {name}")
    with np.load(input_path, allow_pickle=False) as archive:
        if tuple(archive.files) != expected_names:
            raise ValueError("common input NPZ member order changed")
        arrays: dict[str, np.ndarray] = {}
        for name in expected_names:
            if name not in selected_names:
                continue
            metadata = arrays_info[name]
            raw = np.require(archive[name], requirements=["C", "O", "W"])
            if raw.dtype != np.float32:
                raise ValueError(f"common input field dtype changed: {name}")
            value = np.asarray(raw, dtype=np.float32)
            if (metadata.get("shape") != list(expected[name])
                    or value.shape != expected[name]
                    or sha256_array(value) != metadata.get("sha256")
                    or not np.isfinite(value).all()):
                raise ValueError(f"common input field changed: {name}")
            arrays[name] = value
    conditioning = manifest.get("conditioning", {})
    if (not isinstance(conditioning, dict)
            or not _metadata_equal_exact(conditioning.get("timestep"), 1000.0)
            or not _metadata_equal_exact(conditioning.get("temporal_downsample"), 4)):
        raise ValueError("common input fixed conditioning metadata changed")
    if not legacy:
        height, width = tuple(working)
        expected_conditioning = {
            "chunk_latents": CHUNK_LATENTS,
            "call_count": plan["call_count"],
            "projection_frames": plan["projection_frames"],
            "decoder_frames": plan["decoder_frames"],
            "latent_shape_per_call": [
                1, LATENT_CHANNELS, CHUNK_LATENTS, height // 8, width // 8
            ],
            "token_shape_per_call": [CHUNK_LATENTS, height // 16, width // 16],
        }
        if any(not _metadata_equal_exact(conditioning.get(name), value)
               for name, value in expected_conditioning.items()):
            raise ValueError("common input temporal conditioning metadata changed")
        noise = manifest.get("noise")
        if (not isinstance(noise, dict)
                or not _metadata_equal_exact(noise.get("call_count"), plan["call_count"])):
            raise ValueError("common input noise call count changed")
    source_rgb = manifest.get("source_rgb", {})
    if (source_rgb.get("payload_path") != str(INPUT_FRAMES)
            or source_rgb.get("manifest_path") != str(INPUT_MANIFEST)
            or source_rgb.get("consumed_frame_count") != source_frame_count
            or len(source_rgb.get("consumed_frame_raw_sha256", [])) != source_frame_count):
        raise ValueError("common input source pixel provenance changed")
    if not legacy:
        _validate_v2_source_receipt(source_rgb, source_frame_count)
        selection = source_rgb.get("selection")
        source_pts = source_rgb.get("source_pts")
        if (not _metadata_equal_exact(
                    selection, {"start_frame": 0, "count": source_frame_count}
                )
                or not isinstance(source_pts, list)
                or len(source_pts) != source_frame_count
                or any(type(value) is not int for value in source_pts)):
            raise ValueError("common input source timeline provenance changed")
    prompt = manifest.get("prompt", {})
    prompt_auth = authenticate_prompt(PROMPT_PATH)
    if (prompt.get("path") != str(PROMPT_PATH.resolve())
            or prompt.get("sha256") != prompt.get("expected_sha256")
            or prompt.get("sha256") != prompt_auth["sha256"]
            or prompt.get("bytes") != prompt_auth["bytes"]
            or prompt.get("shape") != [1, 512, 4096]):
        raise ValueError("common input prompt provenance changed")
    return arrays, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly prepare the input artifact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-size", required=True, help="visible WIDTHxHEIGHT; working geometry is edge-padded")
    parser.add_argument("--frames", type=int, default=SOURCE_FRAME_COUNT)
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("refusing preparation without --run")
    try:
        manifest = prepare(args.output, frame_size=args.frame_size, frame_count=args.frames)
    except (FileExistsError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({"status": manifest["status"], "output": str(Path(args.output).resolve())}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
