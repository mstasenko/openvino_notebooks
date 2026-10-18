"""CPU-only native FlashVSR+ TCDecoder oracle for the captured DiT chunks.

This command intentionally calls the pinned TAEHV ``decode_video`` directly.
It preserves the native decoder's memory between the two calls, records the
exact per-latent packed inputs, and trims the normalized result to the 17
requested source frames.  It is a decoder trajectory reference, not a GPU
qualification.
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from reference_paths import SOURCE_ROOT


ROOT = Path(__file__).resolve().parent
INPUT = Path(
    os.environ.get(
        "FLASHVSR_PLUS_VIDEO_INPUT",
        str(ROOT.parent.parent / "external/video-native-cpu"),
    )
).resolve()
OUTPUT = Path(
    os.environ.get(
        "FLASHVSR_PLUS_VIDEO_DECODER_OUTPUT",
        str(ROOT.parent.parent / "external/video-decoder-native-cpu"),
    )
).resolve()
INPUT_MANIFEST_SHA256 = "865664296e628e07664d02edb36639568839bcb17bc5091c81ab34fa1e2c807f"
PREDICTION_ROOT_DEFAULT = Path(
    os.environ.get(
        "FLASHVSR_PLUS_VIDEO_PREDICTION_ROOT",
        str(INPUT),
    )
).resolve()
DECODER_REFERENCE = Path(
    str(ROOT.parent.parent / "external/video-decoder-native-cpu")
).resolve()
DECODER_REFERENCE_MANIFEST_SHA256 = (
    "58d239b261a45acaae8152cb3dbbcc07df8e63329d1cfed9186776096065c2ec"
)
DECODER_REFERENCE_FINAL_SHA256 = (
    "57438de244fe88260390fa81fc955bd06615c2c249c9c59957271bb05b99d246"
)
SOURCE_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
SOURCE_FRAME_COUNT = 17
DECODER_CONDITION_RANGES = ((0, 13), (13, 29))
NATIVE_XPU_STATUS = "native_xpu_cpu_variation_measured_not_ov_qualification"
NATIVE_XPU_DEVICE = "xpu:0"
NATIVE_XPU_BDF = "0000:03:00.0"
NATIVE_XPU_UUID = "86800be2-0000-0000-0300-000000000000"
SOURCE_ROOT = SOURCE_ROOT.resolve()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plus_decoder import load_decoder  # noqa: E402
from plus_projection import CHECKPOINT_SHA256 as LQ_CHECKPOINT_SHA256  # noqa: E402
from plus_text import PROMPT_BYTES, PROMPT_PATH, PROMPT_SHA256, PROMPT_SHAPE  # noqa: E402
from plus_weights import (  # noqa: E402
    CHECKPOINT as DIT_CHECKPOINT,
    CHECKPOINT_BYTES as DIT_CHECKPOINT_BYTES,
    CHECKPOINT_SHA256 as DIT_CHECKPOINT_SHA256,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def bf16_bits(value: torch.Tensor) -> np.ndarray:
    if value.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 tensor, got {value.dtype}")
    return value.detach().contiguous().cpu().view(torch.uint16).numpy().copy()


def bf16_f32(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(dtype=torch.float32).cpu().numpy().copy()


def bits_to_bf16(value: np.ndarray) -> torch.Tensor:
    bits = np.ascontiguousarray(value, dtype=np.uint16)
    return torch.from_numpy(bits.copy()).view(torch.bfloat16)


def _bits_and_f32(value: torch.Tensor, name: str) -> dict[str, np.ndarray]:
    return {f"{name}_bf16": bf16_bits(value), f"{name}_f32": bf16_f32(value)}


def decoder_plan() -> dict[str, Any]:
    return {
        "source_frames": SOURCE_FRAME_COUNT,
        "calls": [
            {
                "call": 0,
                "latent_count": 4,
                "condition_frame_range": list(DECODER_CONDITION_RANGES[0]),
                "raw_output_frames": 13,
                "initial_trim_frames": 3,
                "final_source_frame_range": [0, 13],
            },
            {
                "call": 1,
                "latent_count": 4,
                "condition_frame_range": list(DECODER_CONDITION_RANGES[1]),
                "raw_output_frames": 16,
                "initial_trim_frames": 0,
                "final_source_frame_range": [13, 29],
            },
        ],
        "final_trim_frames": SOURCE_FRAME_COUNT,
        "condition_timeline": "projection_video[0:13] then projection_video[13:29]",
    }


def source_git_receipt() -> dict[str, Any]:
    source_root = SOURCE_ROOT
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if head != SOURCE_COMMIT:
        raise RuntimeError(f"pinned source commit mismatch: {head} != {SOURCE_COMMIT}")
    status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"pinned source tree is not clean:\n{status}")
    return {"root": str(source_root), "commit": head, "clean": True}


def resolve_prediction_root(value: Path | str | None = None) -> Path:
    return Path(value).resolve() if value is not None else PREDICTION_ROOT_DEFAULT


def _authenticate_native_input() -> tuple[dict[str, Any], dict[str, Any]]:
    if not INPUT.is_dir():
        raise FileNotFoundError(INPUT)
    manifest_path = INPUT / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != INPUT_MANIFEST_SHA256:
        raise ValueError(f"captured video manifest changed: {manifest_hash}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "cpu_native_denoiser_capture_complete":
        raise ValueError("captured video is not a completed native CPU reference")
    if manifest.get("source", {}).get("commit") != SOURCE_COMMIT:
        raise ValueError("captured video source commit is not pinned")
    archive = manifest.get("archive", {})
    input_info = archive.get("inputs", {})
    input_path = INPUT / "inputs.npz"
    if input_info.get("sha256") != sha256_file(input_path):
        raise ValueError("captured inputs.npz hash mismatch")
    call_info = archive.get("calls", [])
    if len(call_info) != 2:
        raise ValueError("captured video must contain exactly two denoiser calls")
    for call in range(2):
        path = INPUT / f"call{call}.npz"
        if call_info[call].get("sha256") != sha256_file(path):
            raise ValueError(f"captured call{call}.npz hash mismatch")
    return manifest, {"manifest_path": str(manifest_path), "manifest_sha256": manifest_hash}


def _load_native_denoised() -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for call in range(2):
        path = INPUT / f"call{call}.npz"
        with np.load(path, allow_pickle=False) as loaded:
            required = {"denoised_bf16"}
            if not required <= set(loaded.files):
                raise ValueError(f"captured call{call} is missing denoised_bf16")
            value = np.array(loaded["denoised_bf16"], dtype=np.uint16, copy=True)
            if value.shape != (1, 16, 4, 16, 16):
                raise ValueError(f"unexpected denoised call{call} shape: {value.shape}")
            arrays.append(value)
    return arrays


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _authenticate_native_xpu_prediction(
    prediction_root: Path,
    native_manifest_sha256: str,
    report: dict[str, Any],
) -> list[np.ndarray]:
    """Authenticate the explicitly separate native-XPU B580 result schema."""

    if report.get("status") != NATIVE_XPU_STATUS:
        raise ValueError("native-XPU prediction status is not terminal")
    if report.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("native-XPU prediction source commit is not pinned")
    if report.get("source_root") != str(SOURCE_ROOT):
        raise ValueError("native-XPU prediction source root changed")
    if report.get("reference_manifest_sha256") != native_manifest_sha256:
        raise ValueError("native-XPU prediction does not reference the authenticated native capture")
    if report.get("device") != NATIVE_XPU_DEVICE or report.get("bdf") != NATIVE_XPU_BDF:
        raise ValueError("native-XPU prediction device/BDF changed")
    properties = report.get("properties")
    if (
        not isinstance(properties, str)
        or "Intel(R) Arc(TM) B580 Graphics" not in properties
        or f"uuid={NATIVE_XPU_UUID}" not in properties
    ):
        raise ValueError("native-XPU prediction B580 UUID/properties changed")
    torch_version = report.get("torch")
    if not isinstance(torch_version, str) or "+xpu" not in torch_version:
        raise ValueError("native-XPU prediction is not a Torch-XPU result")
    _require_sha256(report.get("worker_sha256"), "native-XPU worker SHA256")

    weights = report.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("native-XPU prediction weights receipt is missing")
    dit = weights.get("dit")
    if not isinstance(dit, dict):
        raise ValueError("native-XPU DIT receipt is missing")
    if (
        dit.get("path") != str(DIT_CHECKPOINT.resolve())
        or dit.get("bytes") != DIT_CHECKPOINT_BYTES
        or dit.get("sha256") != DIT_CHECKPOINT_SHA256
        or dit.get("key_count") != 825
        or dit.get("layer_count") != 30
        or dit.get("layer_parameter_count") != 27
        or dit.get("source_dtype") != "F32"
    ):
        raise ValueError("native-XPU DIT receipt changed")
    if weights.get("lq") != LQ_CHECKPOINT_SHA256:
        raise ValueError("native-XPU LQ checkpoint receipt changed")
    prompt = weights.get("prompt")
    if not isinstance(prompt, dict) or (
        prompt.get("path") != str(PROMPT_PATH.resolve())
        or prompt.get("bytes") != PROMPT_BYTES
        or prompt.get("sha256") != PROMPT_SHA256
        or prompt.get("expected_sha256") != PROMPT_SHA256
        or prompt.get("shape") != list(PROMPT_SHAPE)
        or prompt.get("source_dtype") != "BF16"
    ):
        raise ValueError("native-XPU prompt receipt changed")

    calls = report.get("calls")
    if not isinstance(calls, list) or len(calls) != 2:
        raise ValueError("native-XPU prediction must contain exactly two calls")
    arrays: list[np.ndarray] = []
    for call in range(2):
        path = prediction_root / f"call{call}.npz"
        if not path.is_file():
            raise ValueError(f"native-XPU prediction call{call}.npz is missing")
        entry = calls[call]
        if not isinstance(entry, dict) or entry.get("call") != call:
            raise ValueError(f"native-XPU prediction call{call} report entry is malformed")
        call_sha256 = _require_sha256(entry.get("raw_sha256"), f"native-XPU call{call} SHA256")
        if sha256_file(path) != call_sha256:
            raise ValueError(f"native-XPU prediction call{call}.npz fingerprint mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            required = {"condition", "prediction", "denoised"}
            required.update(
                field
                for layer in range(30)
                for field in (f"layer{layer}_out", f"layer{layer}_mask")
            )
            if set(loaded.files) != required:
                raise ValueError(f"native-XPU prediction call{call} schema is incomplete")
            expected_shapes = {
                "condition": (1, 256, 1536),
                "prediction": (1, 16, 4, 16, 16),
                "denoised": (1, 16, 4, 16, 16),
            }
            values = {}
            for name, shape in expected_shapes.items():
                value = loaded[name]
                if value.dtype != np.dtype("float32") or value.shape != shape:
                    raise ValueError(
                        f"native-XPU prediction call{call} {name} ABI changed: "
                        f"{value.shape} {value.dtype}"
                    )
                if not np.isfinite(value).all():
                    raise ValueError(f"native-XPU prediction call{call} {name} is non-finite")
                values[name] = np.array(value, copy=True)
            for layer in range(30):
                layer_output = loaded[f"layer{layer}_out"]
                layer_mask = loaded[f"layer{layer}_mask"]
                if layer_output.dtype != np.dtype("float32") or layer_output.shape != (1, 256, 1536):
                    raise ValueError(f"native-XPU prediction call{call} layer{layer} output ABI changed")
                if not np.isfinite(layer_output).all():
                    raise ValueError(f"native-XPU prediction call{call} layer{layer} output is non-finite")
                if (
                    layer_mask.dtype != np.dtype("bool")
                    or layer_mask.ndim != 3
                    or tuple(layer_mask.shape[:2]) != (12, 2)
                    or layer_mask.shape[2] <= 0
                ):
                    raise ValueError(f"native-XPU prediction call{call} layer{layer} mask ABI changed")
            arrays.append(values["denoised"])
    return arrays


def authenticate_prediction_root(
    prediction_root: Path,
    native_manifest_sha256: str,
    results_sha256: str,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Authenticate a caller-pinned completed denoised-F32 archive."""
    prediction_root = prediction_root.resolve()
    if len(results_sha256) != 64 or any(char not in "0123456789abcdef" for char in results_sha256):
        raise ValueError("prediction results SHA256 must be 64 lowercase hexadecimal characters")
    results_path = prediction_root / "results.json"
    if not results_path.is_file() or sha256_file(results_path) != results_sha256:
        raise ValueError("prediction results fingerprint mismatch")
    report = json.loads(results_path.read_text(encoding="utf-8"))
    status = report.get("status")
    if status == NATIVE_XPU_STATUS:
        return report, _authenticate_native_xpu_prediction(
            prediction_root, native_manifest_sha256, report
        )
    if not isinstance(status, str) or not (status.startswith("pass_") or status.startswith("fail_")):
        raise ValueError("prediction archive is not a completed pass/fail diagnostic result")
    if report.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("prediction source commit is not pinned")
    if report.get("reference_manifest_sha256") != native_manifest_sha256:
        raise ValueError("prediction does not reference the authenticated native capture")
    runtime = report.get("runtime")
    if not isinstance(runtime, str) or runtime.split(".", 2)[:2] != ["2026", "5"]:
        raise ValueError("prediction runtime is not OpenVINO 2026.5")
    calls = report.get("calls")
    if report.get("device") != "GPU.1" or not isinstance(calls, list) or len(calls) != 2:
        raise ValueError("prediction device/call contract changed")
    arrays: list[np.ndarray] = []
    for call in range(2):
        path = prediction_root / f"call{call}.npz"
        if not path.is_file():
            raise ValueError(f"prediction call{call}.npz is missing")
        entry = calls[call]
        if not isinstance(entry, dict):
            raise ValueError(f"prediction call{call} report entry is malformed")
        call_sha256 = entry.get("raw_sha256")
        if entry.get("call") != call or not isinstance(call_sha256, str) or len(call_sha256) != 64:
            raise ValueError(f"prediction call{call} report fingerprint changed")
        if sha256_file(path) != call_sha256:
            raise ValueError(f"prediction call{call}.npz fingerprint mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) < {"denoised"}:
                raise ValueError(f"prediction call{call} is missing denoised F32 output")
            source_value = loaded["denoised"]
            if source_value.dtype != np.dtype("float32"):
                raise ValueError(f"prediction call{call} denoised storage must be F32")
            value = np.array(source_value, copy=True)
        if value.shape != (1, 16, 4, 16, 16) or not np.isfinite(value).all():
            raise ValueError(f"unexpected/non-finite prediction call{call}: {value.shape}")
        arrays.append(value)
    return report, arrays


def authenticate_input(
    prediction_root: Path | str | None = None,
    prediction_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[np.ndarray]]:
    native_manifest, input_receipt = _authenticate_native_input()
    prediction_root = resolve_prediction_root(prediction_root)
    if prediction_root == INPUT:
        if prediction_sha256 is not None:
            raise ValueError("--prediction-sha256 requires --prediction-root")
        return native_manifest, input_receipt, _load_native_denoised()
    if prediction_sha256 is None:
        raise ValueError("--prediction-sha256 is required with --prediction-root")
    prediction_report, arrays = authenticate_prediction_root(
        prediction_root, input_receipt["manifest_sha256"], prediction_sha256
    )
    input_receipt["prediction_root"] = str(prediction_root)
    input_receipt["prediction_results_sha256"] = prediction_sha256
    input_receipt["prediction_status"] = prediction_report.get("status")
    if prediction_report.get("status") == NATIVE_XPU_STATUS:
        input_receipt["prediction_backend"] = "native_xpu_b580"
        input_receipt["prediction_device"] = prediction_report.get("device")
    else:
        input_receipt["prediction_backend"] = "openvino_gpu"
        input_receipt["prediction_device"] = prediction_report.get("device")
    return native_manifest, input_receipt, arrays


def pack_latents(decoder: torch.nn.Module, condition: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Match TAEHV.decode_video's exact PixelShuffle3d + latent concatenation."""
    if condition.dtype != torch.bfloat16 or latents.dtype != torch.bfloat16:
        raise TypeError("decoder pack inputs must be BF16")
    if tuple(condition.shape[-2:]) != (128, 128) or tuple(latents.shape) != (1, 16, 4, 16, 16):
        raise ValueError(f"unexpected decoder pack shapes: {condition.shape}, {latents.shape}")
    packed_condition = decoder.pixel_shuffle(condition)
    if tuple(packed_condition.shape) != (1, 4, 768, 16, 16):
        raise ValueError(f"unexpected PixelShuffle3d condition shape: {packed_condition.shape}")
    return torch.cat((packed_condition, latents.transpose(1, 2)), dim=2)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": sha256_array(value),
            }
            for name, value in arrays.items()
        },
    }


def authenticate_decoder_reference() -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = DECODER_REFERENCE / "manifest.json"
    final_path = DECODER_REFERENCE / "final.npz"
    if not manifest_path.is_file() or sha256_file(manifest_path) != DECODER_REFERENCE_MANIFEST_SHA256:
        raise ValueError("native decoder reference manifest fingerprint changed")
    if not final_path.is_file() or sha256_file(final_path) != DECODER_REFERENCE_FINAL_SHA256:
        raise ValueError("native decoder reference final fingerprint changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "cpu_native_decoder_capture_complete":
        raise ValueError("native decoder reference is not complete")
    if manifest.get("input", {}).get("manifest", {}).get("manifest_sha256") != INPUT_MANIFEST_SHA256:
        raise ValueError("native decoder reference input changed")
    final_info = manifest.get("final", {}).get("file", {})
    if final_info.get("sha256") != DECODER_REFERENCE_FINAL_SHA256:
        raise ValueError("native decoder reference final receipt changed")
    with np.load(final_path, allow_pickle=False) as loaded:
        if "normalized_final_f32" not in loaded.files:
            raise ValueError("native decoder reference is missing normalized final output")
        final = np.array(loaded["normalized_final_f32"], dtype=np.float32, copy=True)
    if final.shape != (1, SOURCE_FRAME_COUNT, 3, 128, 128) or not np.isfinite(final).all():
        raise ValueError(f"unexpected native decoder reference shape: {final.shape}")
    final_array_info = final_info.get("arrays", {}).get("normalized_final_f32", {})
    if final_array_info.get("sha256") != sha256_array(final):
        raise ValueError("native decoder reference final array receipt changed")
    return final, {
        "root": str(DECODER_REFERENCE),
        "manifest_sha256": DECODER_REFERENCE_MANIFEST_SHA256,
        "final_sha256": DECODER_REFERENCE_FINAL_SHA256,
    }


def _quality_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("quality comparison requires finite, equally shaped arrays")

    def one(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
        difference = lhs.astype(np.float64) - rhs.astype(np.float64)
        denominator = max(float(np.linalg.norm(rhs.astype(np.float64))), 1e-30)
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        lhs_rgb = np.clip((lhs.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)
        rhs_rgb = np.clip((rhs.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)
        rgb_mse = float(np.mean(np.square(lhs_rgb - rhs_rgb)))
        return {
            "normalized_l2": float(np.linalg.norm(difference) / denominator),
            "rmse": rmse,
            "clipped_rgb_psnr_db": None if rgb_mse == 0.0 else float(10.0 * np.log10(1.0 / rgb_mse)),
            "clipped_rgb_mse": rgb_mse,
        }

    overall = one(actual, reference)
    return {
        **overall,
        "per_frame": [
            {"frame": frame, **one(actual[:, frame], reference[:, frame])}
            for frame in range(actual.shape[1])
        ],
    }


def _write_comparison_pngs(output: Path, actual: np.ndarray, reference: np.ndarray) -> list[dict[str, Any]]:
    try:
        from PIL import Image
    except ImportError:
        return []
    records = []
    for frame in (0, 8, SOURCE_FRAME_COUNT - 1):
        for name, value in (("actual", actual), ("native", reference)):
            rgb = np.transpose(value[0, frame], (1, 2, 0))
            pixels = np.clip(np.rint((rgb + 1.0) * 127.5), 0.0, 255.0).astype(np.uint8)
            path = output / f"comparison_frame{frame:02d}_{name}.png"
            Image.fromarray(pixels, mode="RGB").save(path)
            records.append({"path": str(path), "sha256": sha256_file(path), "frame": frame, "kind": name})
    return records


@torch.inference_mode()
def decode(
    output: Path = OUTPUT,
    prediction_root: Path | str | None = None,
    prediction_sha256: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"decoder output must be a fresh leaf: {output}")
    started = time.monotonic()
    source_receipt = source_git_receipt()
    print("[decoder] authenticating captured native DiT archive", flush=True)
    prediction_root = resolve_prediction_root(prediction_root)
    input_manifest, input_receipt, denoised_values = authenticate_input(prediction_root, prediction_sha256)
    prediction_backend = input_receipt.get("prediction_backend", "native_cpu")
    native_final, native_final_receipt = authenticate_decoder_reference()
    print("[decoder] loading authenticated TAEHV decoder on CPU", flush=True)
    decoder, state, decoder_receipt = load_decoder()
    del state
    decoder.to(device="cpu", dtype=torch.bfloat16).eval()
    decoder.clean_mem()
    with np.load(INPUT / "inputs.npz", allow_pickle=False) as inputs:
        # The capture stores the authoritative normalized projection in F32;
        # the source runner casts each clip to execution BF16 before projection
        # or decoder use, so make that same single boundary here.
        projection_video = torch.from_numpy(
            np.ascontiguousarray(inputs["projection_video_f32"], dtype=np.float32)
        ).to(dtype=torch.bfloat16)
    if tuple(projection_video.shape) != (1, 3, 33, 128, 128):
        raise ValueError(f"unexpected projection video shape: {projection_video.shape}")
    if not torch.isfinite(projection_video).all():
        raise FloatingPointError("captured projection video contains non-finite values")
    output.mkdir(parents=True)
    plan = decoder_plan()
    call_infos = []
    normalized_outputs = []
    for call, (condition_range, value) in enumerate(zip(DECODER_CONDITION_RANGES, denoised_values)):
        print(f"[decoder] call {call} start condition={condition_range}", flush=True)
        condition = projection_video[:, :, condition_range[0] : condition_range[1]]
        latents = (
            bits_to_bf16(value)
            if value.dtype == np.uint16
            else torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(dtype=torch.bfloat16)
        )
        packed = pack_latents(decoder, condition, latents)
        packed_arrays: dict[str, np.ndarray] = {}
        for latent_index in range(4):
            packed_arrays[f"packed_latent{call * 4 + latent_index}_bf16"] = bf16_bits(
                packed[:, latent_index]
            )
        raw = decoder.decode_video(
            latents.transpose(1, 2),
            parallel=False,
            show_progress_bar=False,
            cond=condition,
        )
        if tuple(raw.shape) != (1, plan["calls"][call]["raw_output_frames"], 3, 128, 128):
            raise RuntimeError(f"unexpected decoder call{call} output shape: {tuple(raw.shape)}")
        if not torch.isfinite(raw).all():
            raise FloatingPointError(f"non-finite raw decoder output in call {call}")
        normalized = raw.mul(2).sub(1)
        if not torch.isfinite(normalized).all():
            raise FloatingPointError(f"non-finite normalized decoder output in call {call}")
        normalized_outputs.append(normalized.detach().clone())
        arrays = dict(packed_arrays)
        arrays.update(_bits_and_f32(raw, "raw"))
        arrays.update(_bits_and_f32(normalized, "normalized"))
        call_file = _write_npz(output / f"call{call}.npz", arrays)
        call_infos.append(
            {
                "call": call,
                "condition_frame_range": list(condition_range),
                "packed_latent_indices": list(range(call * 4, call * 4 + 4)),
                "raw_shape": list(raw.shape),
                "normalized_shape": list(normalized.shape),
                "raw_finite": True,
                "normalized_finite": True,
                "file": call_file,
            }
        )
        print(f"[decoder] call {call} complete raw_frames={raw.shape[1]}", flush=True)
        del condition, latents, packed, raw, normalized
    final = torch.cat(normalized_outputs, dim=1)[:, :SOURCE_FRAME_COUNT]
    final_arrays = _bits_and_f32(final, "normalized_final")
    final_file = _write_npz(output / "final.npz", final_arrays)
    final_f32 = final_arrays["normalized_final_f32"]
    comparison = _quality_metrics(final_f32, native_final)
    comparison.update(
        reference=native_final_receipt,
        prediction_root=str(prediction_root),
        scope=(
            "native XPU denoised chunks plus native BF16 TAEHV CPU decoder; not XPU decoder/full performance"
            if prediction_backend == "native_xpu_b580"
            else "OV denoised chunks plus native CPU decoder; not GPU decoder or full performance"
        ),
        pngs=_write_comparison_pngs(output, final_f32, native_final),
    )
    decoder.clean_mem()
    manifest = {
        "format": "flashvsr-plus-native-video-decoder-cpu-v1",
        "status": (
            "cpu_native_decoder_from_native_xpu_prediction_complete"
            if prediction_backend == "native_xpu_b580"
            else "cpu_native_decoder_from_ov_prediction_complete"
            if prediction_root != INPUT
            else "cpu_native_decoder_capture_complete"
        ),
        "quality_scope": (
            "native XPU denoised chunks plus native BF16 TAEHV CPU decoder only; no XPU decoder/full-performance qualification"
            if prediction_backend == "native_xpu_b580"
            else "OV denoised chunks plus native BF16 TAEHV CPU decoder only; no GPU decoder/full-performance qualification"
            if prediction_root != INPUT
            else "native BF16 TAEHV decoder trajectory only; no GPU qualification"
        ),
        "source": source_receipt,
        "decoder": decoder_receipt,
        "input": {
            "root": str(INPUT),
            "manifest": input_receipt,
            "source_manifest_status": input_manifest.get("status"),
            "source_manifest_input_hashes": input_manifest.get("input", {}).get("consumed_frame_raw_sha256"),
            "prediction_root": str(prediction_root),
        },
        "dtype": "torch.bfloat16",
        "threads": 2,
        "parallel": False,
        "plan": plan,
        "calls": call_infos,
        "final": {
            "source_frame_count": SOURCE_FRAME_COUNT,
            "shape": list(final.shape),
            "finite": bool(torch.isfinite(final).all()),
            "file": final_file,
        },
        "comparison": comparison,
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "results.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )
    print(
        f"[decoder] complete output={output} wall_seconds={manifest['runtime']['wall_seconds']:.3f} "
        f"peak_rss_kib={manifest['runtime']['peak_rss_kib']}",
        flush=True,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly run the CPU decoder oracle")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=None,
        help="authenticated denoised-F32 prediction archive (defaults to the native CPU capture)",
    )
    parser.add_argument(
        "--prediction-sha256",
        default=None,
        help="SHA256 of prediction-root/results.json; required with --prediction-root",
    )
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("refusing the decoder capture without --run")
    if args.prediction_root is None and args.prediction_sha256 is not None:
        parser.error("--prediction-sha256 requires --prediction-root")
    if args.prediction_root is not None and args.prediction_sha256 is None:
        parser.error("--prediction-sha256 is required with --prediction-root")
    manifest = decode(
        args.output.resolve(),
        resolve_prediction_root(args.prediction_root),
        args.prediction_sha256,
    )
    print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
