"""Compare a native Plus source archive with a resident OpenVINO archive.

The comparison is deliberately descriptive.  It authenticates the shared
common-input artifact and both output ABIs, reports latent differences and
decoded PSNR, and preserves measured stage timings.  It does not apply a
quality or performance pass gate and makes no 4K/frame extrapolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from plus_projection import CHECKPOINT_SHA256 as LQ_CHECKPOINT_SHA256
from plus_text import PROMPT_BYTES, PROMPT_SHA256
from plus_weights import (
    CHECKPOINT_BYTES as DIT_CHECKPOINT_BYTES,
    CHECKPOINT_SHA256 as DIT_CHECKPOINT_SHA256,
    GLOBAL_SHAPES,
    LAYER_PARAMETER_COUNT,
    NUM_LAYERS,
    SCHEMA_KEY_COUNT,
)
from prepare_plus_video_input import (
    FORMAT as INPUT_FORMAT,
    FORMAT_V2 as INPUT_FORMAT_V2,
    SOURCE,
    SOURCE_COMMIT,
    STATUS as INPUT_STATUS,
    input_plan,
    load_input,
    sha256_array,
)
from run_plus_native_xpu import preflight_input_geometry
from reference_paths import EVIDENCE_ROOT


NATIVE_STATUS = "native_xpu_b580_diagnostic_complete"
RESIDENT_STATUS = "complete_resident_pilot_not_quality_qualified"
FORMAT = "flashvsr-plus-native-vs-resident-comparison-v1"
CALLS = 2
LAYERS = 30
B580_UUID = "86800be2-0000-0000-0300-000000000000"
CUDA45_DEVICE_UUID = "GPU-376cdd2a-9343-0add-67da-9d230a3638df"
CUDA45_SPEED_REPORT = EVIDENCE_ROOT / "native-cuda-speed/results.json"
CUDA45_ARCHIVE_REPORT = EVIDENCE_ROOT / "native-cuda-archive/results.json"
CUDA45_REFERENCE_LINK = EVIDENCE_ROOT / "native-uhd-reference-link.json"
CUDA45_REFERENCE_LINK_SHA256 = (
    "6e6acb353091bc0f2d9971c30bc70b8e0fc6d0ce47dda3066fb31660faee145a"
)
CUDA45_RAW_SHA256 = "7515af9a2c73fc8515eedd65b5fb9808651a9159a9d97da8dfb8cb768c38075d"
CUDA45_INPUT_PLAN_FIELDS = (
    "backend",
    "device_uuid",
    "source_commit",
    "source_root",
    "input",
    "input_plan",
    "weights",
    "config",
    "stream_policy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _resolve_member(root: Path, path_value: Any, fallback: str, label: str) -> Path:
    path = root / fallback
    if path_value is not None:
        try:
            reported = Path(path_value).resolve()
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} path is malformed") from error
        if reported != path.resolve():
            raise ValueError(f"{label} path does not belong to its archive root")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def _load_npz(
    path: Path,
    label: str,
    *,
    expected_order: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if expected_order is not None and tuple(archive.files) != expected_order:
                raise ValueError(f"{label} NPZ member order changed")
            return {name: np.array(archive[name], copy=True, order="C") for name in archive.files}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{label} is not a valid NPZ archive") from error


def _f32_field(
    values: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    value = values.get(name)
    if value is None or value.dtype != np.dtype("float32") or value.shape != shape:
        raise ValueError(f"{label}/{name} ABI changed: expected float32 {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{label}/{name} contains non-finite values")
    return value


def _bf16_round_bits(value: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.dtype("float32") or not np.isfinite(value).all():
        raise ValueError(f"{label} must be finite float32")
    raw = np.ascontiguousarray(value).view(np.uint32)
    rounded = ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)
    if np.any((rounded & np.uint16(0x7f80)) == np.uint16(0x7f80)):
        raise ValueError(f"{label} overflows finite BF16 storage")
    return rounded


def _bf16_field(
    values: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    label: str,
    reference: np.ndarray,
) -> np.ndarray:
    value = values.get(name)
    if value is None or value.dtype != np.dtype("uint16") or value.shape != shape:
        raise ValueError(f"{label}/{name} ABI changed: expected uint16 {shape}")
    if np.any((value & np.uint16(0x7f80)) == np.uint16(0x7f80)):
        raise ValueError(f"{label}/{name} contains non-finite BF16 storage")
    expected = _bf16_round_bits(reference, f"{label}/{name} reference")
    if not np.array_equal(value, expected):
        raise ValueError(f"{label}/{name} does not match its F32 BF16 representation")
    return value


def _require_fields(values: dict[str, np.ndarray], expected: set[str], label: str) -> None:
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise ValueError(f"{label} archive fields changed: missing={missing}, extra={extra}")


def _metadata_equal_exact(actual: Any, expected: Any) -> bool:
    """Compare JSON metadata without Python's bool/int equality loophole."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return tuple(actual) == tuple(expected) and all(
            _metadata_equal_exact(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _metadata_equal_exact(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _archive_record(path: Path, label: str, claimed: Any = None) -> dict[str, Any]:
    observed = sha256_file(path)
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": observed,
        "receipt_claimed": claimed is not None,
        "receipt_verified": False,
    }
    if claimed is not None:
        if (not isinstance(claimed, str) or len(claimed) != 64
                or any(char not in "0123456789abcdef" for char in claimed)):
            raise ValueError(f"{label} SHA256 receipt is malformed")
        if observed != claimed:
            raise ValueError(f"{label} SHA256 receipt does not match")
        record["claimed_sha256"] = claimed
        record["receipt_verified"] = True
    else:
        record["claimed_sha256"] = None
    return record


def _path_receipt(receipt: Any, label: str) -> tuple[str, int, str]:
    if not isinstance(receipt, dict):
        raise ValueError(f"{label} weight receipt is missing")
    path = receipt.get("path")
    size = receipt.get("bytes")
    digest = receipt.get("sha256")
    if (not isinstance(path, str) or not path
            or not isinstance(size, int) or isinstance(size, bool) or size <= 0
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError(f"{label} weight receipt is malformed")
    return path, size, digest


def _validate_dit_receipt(receipt: Any, label: str) -> dict[str, Any]:
    path, size, digest = _path_receipt(receipt, label)
    if size != DIT_CHECKPOINT_BYTES or digest != DIT_CHECKPOINT_SHA256:
        raise ValueError(f"{label} is not the pinned Turbo DiT checkpoint")
    schema = {
        "key_count": SCHEMA_KEY_COUNT,
        "layer_count": NUM_LAYERS,
        "layer_parameter_count": LAYER_PARAMETER_COUNT,
        "global_key_count": len(GLOBAL_SHAPES),
        "source_dtype": "F32",
    }
    for key, expected in schema.items():
        if receipt.get(key) != expected:
            raise ValueError(f"{label} schema receipt changed: {key}")
    return {"path": path, "bytes": size, "sha256": digest, "schema": schema}


def _validate_lq_receipt(receipt: Any) -> dict[str, Any]:
    path, size, digest = _path_receipt(receipt, "LQ projection")
    if digest != LQ_CHECKPOINT_SHA256:
        raise ValueError("LQ projection receipt is not the pinned checkpoint")
    return {"path": path, "bytes": size, "sha256": digest}


def _validate_prompt_receipt(receipt: Any) -> dict[str, Any]:
    path, size, digest = _path_receipt(receipt, "prompt")
    if (size != PROMPT_BYTES or digest != PROMPT_SHA256
            or receipt.get("expected_sha256") != PROMPT_SHA256
            or receipt.get("shape") != [1, 512, 4096]
            or receipt.get("source_dtype") != "BF16"):
        raise ValueError("prompt receipt is not the pinned BF16 prompt")
    return {"path": path, "bytes": size, "sha256": digest, "shape": [1, 512, 4096]}


def _validate_native_weights(report: dict[str, Any]) -> dict[str, Any]:
    receipts = report.get("weights")
    if not isinstance(receipts, dict) or set(receipts) != {"dit", "lq_projection", "prompt"}:
        raise ValueError("native weight receipts are incomplete")
    return {
        "dit": _validate_dit_receipt(receipts["dit"], "native DiT"),
        "lq_projection": _validate_lq_receipt(receipts["lq_projection"]),
        "prompt": _validate_prompt_receipt(receipts["prompt"]),
    }


def _validate_resident_weights(report: dict[str, Any]) -> dict[str, Any]:
    receipts = report.get("weights")
    if not isinstance(receipts, dict) or set(receipts) != {"path", "bytes", "sha256",
                                                          "key_count", "layer_count",
                                                          "layer_parameter_count",
                                                          "global_key_count", "source_dtype"}:
        raise ValueError("resident DiT weight receipt is incomplete")
    return {
        "dit": _validate_dit_receipt(receipts, "resident DiT"),
        "lq_projection": {"reported": False},
        "prompt": {"reported": False},
    }


_DECODER_IDENTITY_FIELDS = ("source_commit", "source_sha256", "checkpoint_sha256")


def _decoder_checkpoint_pins() -> tuple[str, str]:
    """Load canonical decoder fingerprints without importing model state."""

    # Keep denoiser-only validation independent of the Torch decoder module;
    # this lazy import loads only its constants and never constructs weights.
    from plus_decoder import CHECKPOINT_SHA256, SOURCE_SHA256

    return SOURCE_SHA256, CHECKPOINT_SHA256


def _require_decoder_identity(report: dict[str, Any], label: str) -> dict[str, str]:
    candidates: list[dict[str, Any]] = []
    decode = report.get("decode")
    if isinstance(decode, dict):
        decoder = decode.get("decoder")
        if isinstance(decoder, dict):
            candidates.append(decoder)
        nested_runs = decode.get("runs")
        if isinstance(nested_runs, list):
            candidates.extend(
                item["decoder"]
                for item in nested_runs
                if isinstance(item, dict) and isinstance(item.get("decoder"), dict)
            )
    runs = report.get("runs")
    if isinstance(runs, list):
        candidates.extend(
            item["decode"]["decoder"]
            for item in runs
            if isinstance(item, dict)
            and isinstance(item.get("decode"), dict)
            and isinstance(item["decode"].get("decoder"), dict)
        )
    if not candidates:
        raise ValueError(f"{label} decoder identity is missing")
    identity: dict[str, str] | None = None
    for field in _DECODER_IDENTITY_FIELDS:
        values = [candidate.get(field) for candidate in candidates]
        value = values[0]
        if any(other != value for other in values[1:]):
            raise ValueError(f"{label} decoder identity changed: {field}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} decoder identity is malformed: {field}")
        if identity is None:
            identity = {}
        identity[field] = value
    assert identity is not None
    return identity


def _require_resident_identity(
    native_report: dict[str, Any],
    resident_report: dict[str, Any],
    source: dict[str, Any],
    label: str,
) -> None:
    """Require full-run resident source/decoder/LQ identity to match native pins.

    Denoiser-only validation intentionally stays independent of this check;
    only complete native-vs-resident comparisons need the decoder and LQ
    identity boundary.
    """
    source_commit = source.get("commit")
    source_root = source.get("root")
    if not isinstance(source_commit, str) or not isinstance(source_root, str):
        raise ValueError(f"{label} authenticated source identity is malformed")
    if native_report.get("source_commit") != source_commit or native_report.get("source_root") != source_root:
        raise ValueError(f"{label} native source identity does not match authenticated input")
    native_decoder = _require_decoder_identity(native_report, f"{label} native")
    if native_decoder["source_commit"] != source_commit:
        raise ValueError(f"{label} native decoder source commit does not match authenticated input")
    decoder_source_sha256, decoder_checkpoint_sha256 = _decoder_checkpoint_pins()
    if native_decoder["source_sha256"] != decoder_source_sha256:
        raise ValueError(f"{label} native decoder source fingerprint is not pinned")
    if native_decoder["checkpoint_sha256"] != decoder_checkpoint_sha256:
        raise ValueError(f"{label} native decoder checkpoint is not pinned")

    decoded = resident_report.get("decoded")
    resident_decoder = decoded.get("decoder") if isinstance(decoded, dict) else None
    if not isinstance(resident_decoder, dict):
        raise ValueError(f"{label} resident decoder identity is missing")
    for field, expected in native_decoder.items():
        if resident_decoder.get(field) != expected:
            raise ValueError(f"{label} resident decoder identity changed: {field}")
    if isinstance(decoded, dict) and "phase" in decoded:
        phase = decoded["phase"]
        if not isinstance(phase, dict):
            raise ValueError(f"{label} resident decoder phase identity is malformed")
        for field, expected in native_decoder.items():
            if phase.get(field) != expected:
                raise ValueError(f"{label} resident decoder phase identity changed: {field}")

    native_weights = native_report.get("weights")
    native_lq = native_weights.get("lq_projection") if isinstance(native_weights, dict) else None
    projection = resident_report.get("projection")
    resident_lq = projection.get("weights") if isinstance(projection, dict) else None
    if not isinstance(native_lq, dict) or not isinstance(resident_lq, dict):
        raise ValueError(f"{label} LQ projection identity is missing")
    if native_lq.get("sha256") != LQ_CHECKPOINT_SHA256:
        raise ValueError(f"{label} native LQ checkpoint is not pinned")
    if resident_lq.get("source_commit") != source_commit:
        raise ValueError(f"{label} resident LQ source commit does not match authenticated input")
    if resident_lq.get("sha256") != LQ_CHECKPOINT_SHA256:
        raise ValueError(f"{label} resident LQ checkpoint identity changed")


def _validate_input_manifest(
    manifest: dict[str, Any], *, allow_v2: bool = False
) -> tuple[tuple[int, int], tuple[int, int]]:
    visible = manifest.get("visible_hw")
    working = manifest.get("working_hw")
    if (
        manifest.get("format") not in (
            (INPUT_FORMAT, INPUT_FORMAT_V2) if allow_v2 else (INPUT_FORMAT,)
        )
        or manifest.get("status") != INPUT_STATUS
        or not isinstance(visible, list)
        or not isinstance(working, list)
        or len(visible) != 2
        or len(working) != 2
        or any(type(value) is not int or value <= 0 for value in (*visible, *working))
        or visible[0] > working[0]
        or visible[1] > working[1]
    ):
        raise ValueError("common input visible/working geometry is malformed")
    source = manifest.get("source")
    if (not isinstance(source, dict)
            or source.get("commit") != SOURCE_COMMIT
            or source.get("root") != str(SOURCE)):
        raise ValueError("common input source identity is malformed")
    return (int(visible[0]), int(visible[1])), (int(working[0]), int(working[1]))


def _validate_source_mask(value: np.ndarray, working_hw: tuple[int, int], label: str) -> None:
    if value.dtype != np.dtype("bool") or value.ndim != 4:
        raise ValueError(f"{label} ABI changed: expected bool [batch,heads,q_blocks,k_blocks]")
    spatial_windows = (working_hw[0] // 16) * (working_hw[1] // 16) // 64
    if spatial_windows < 1:
        raise ValueError(f"{label} geometry has no complete spatial attention window")
    query_blocks = 2 * spatial_windows
    if (tuple(value.shape[:3]) != (1, 12, query_blocks)
            or value.shape[3] < query_blocks
            or value.shape[3] % spatial_windows):
        raise ValueError(f"{label} mask geometry changed: {value.shape}")


def _validate_native_report(
    root: Path,
    report: dict[str, Any],
    input_sha256: str,
    visible_hw: tuple[int, int],
    working_hw: tuple[int, int],
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    if (
        report.get("status") != NATIVE_STATUS
        or report.get("backend") != "native_torch_xpu"
        or report.get("device_uuid") != B580_UUID
    ):
        raise ValueError("native report status/backend changed")
    if report.get("input", {}).get("manifest_sha256") != input_sha256:
        raise ValueError("native report input manifest does not match")
    if tuple(report.get("input", {}).get("working_hw", ())) != working_hw:
        raise ValueError("native report geometry does not match common input")
    calls = report.get("calls")
    if not isinstance(calls, list) or len(calls) != CALLS:
        raise ValueError("native report must contain exactly two calls")
    latent_shape = (1, 16, 4, working_hw[0] // 8, working_hw[1] // 8)
    denoised: list[np.ndarray] = []
    archives: list[dict[str, Any]] = []
    for call, entry in enumerate(calls):
        if not isinstance(entry, dict) or entry.get("call") != call:
            raise ValueError(f"native call{call} report entry is malformed")
        archive = entry.get("archive")
        if not isinstance(archive, dict):
            raise ValueError(f"native call{call} archive receipt is missing")
        path = _resolve_member(root, archive.get("path"), f"call{call}.npz", f"native call{call}")
        if archive.get("sha256") is None:
            raise ValueError(f"native call{call} archive SHA256 receipt is missing")
        archives.append(_archive_record(path, f"native call{call}", archive["sha256"]))
        values = _load_npz(path, f"native call{call}")
        _require_fields(
            values,
            {"lq_condition0_f32", "prediction_f32", "denoised_f32"}
            | {f"layer{layer}_source_mask" for layer in range(LAYERS)},
            f"native call{call}",
        )
        condition = _f32_field(values, "lq_condition0_f32", (1, 4 * working_hw[0] // 16 * working_hw[1] // 16, 1536), f"native call{call}")
        del condition
        _f32_field(values, "prediction_f32", latent_shape, f"native call{call}")
        denoised.append(_f32_field(values, "denoised_f32", latent_shape, f"native call{call}"))
        for layer in range(LAYERS):
            mask = values.get(f"layer{layer}_source_mask")
            _validate_source_mask(mask, working_hw, f"native call{call}/layer{layer}_source_mask")
    decode = report.get("decode")
    if not isinstance(decode, dict):
        raise ValueError("native report has no completed decoder receipt")
    final_receipt = decode.get("final")
    if not isinstance(final_receipt, dict):
        raise ValueError("native decoder final receipt is missing")
    final_path = _resolve_member(root, final_receipt.get("path"), "final.npz", "native final")
    if final_receipt.get("sha256") is None:
        raise ValueError("native final archive SHA256 receipt is missing")
    archives.append(_archive_record(final_path, "native final", final_receipt["sha256"]))
    final_values = _load_npz(final_path, "native final")
    _require_fields(final_values, {"normalized_final_f32"}, "native final")
    final_shape = (1, 17, 3, visible_hw[0], visible_hw[1])
    final = _f32_field(final_values, "normalized_final_f32", final_shape, "native final")
    return denoised, final, {"report": report, "final_values": final_values, "archives": archives}


def _resident_plan(report, plan):
    """Keep the legacy ABI fixed; streams require an authenticated schedule."""
    if plan is None:
        return None
    from prepare_plus_video_input import _metadata_equal_exact, _plan_metadata

    if not isinstance(plan, dict) or type(plan.get("source_frames")) is not int:
        raise ValueError("resident stream requires an authenticated input plan")
    expected = _plan_metadata(plan["source_frames"])
    if not _metadata_equal_exact(plan, expected):
        raise ValueError("resident stream input plan changed")
    reported = report.get("input_plan")
    if "input_plan" not in report and expected["source_frames"] == 17:
        return expected  # Legacy 17-frame reports predate explicit plan metadata.
    if not _metadata_equal_exact(reported, expected):
        raise ValueError("resident report input plan does not match")
    return expected


def validate_resident_latents(
    root: Path,
    report: dict[str, Any],
    input_sha256: str,
    visible_hw: tuple[int, int],
    working_hw: tuple[int, int],
    *,
    plan: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Authenticate a resident denoiser report without requiring decoder output.

    The denoiser-only resident worker intentionally has no ``decoded`` receipt.
    Keep every report, weight, archive, shape, and BF16/F32 check here so a
    decoder diagnostic can consume its latents without weakening the complete
    native-vs-resident comparison below.
    """
    if (
        report.get("status") != RESIDENT_STATUS
        or report.get("device") != "GPU.1"
        or not isinstance(report.get("runtime"), str)
        or not report["runtime"].startswith("2026.5.")
    ):
        raise ValueError("resident report status/device changed")
    if report.get("input_sha256") != input_sha256:
        raise ValueError("resident report input manifest does not match")
    plan = _resident_plan(report, plan)
    count = CALLS if plan is None else plan["call_count"]
    weights = _validate_resident_weights(report)
    calls = report.get("calls")
    if not isinstance(calls, list) or len(calls) != count:
        raise ValueError(f"resident report must contain exactly {count} calls")
    latent_shape = (1, 16, 4, working_hw[0] // 8, working_hw[1] // 8)
    denoised: list[np.ndarray] = []
    archives: list[dict[str, Any]] = []
    for call, entry in enumerate(calls):
        if not isinstance(entry, dict) or entry.get("call") != call:
            raise ValueError(f"resident call{call} report entry is malformed")
        if "archive" not in entry:
            path = _resolve_member(root, None, f"call{call}.npz", f"resident call{call}")
            # Older resident workers did not persist per-call receipts.  Hash
            # the bytes here, but keep the result explicitly comparator-only.
            archives.append(_archive_record(path, f"resident call{call}"))
        else:
            archive = entry["archive"]
            if not isinstance(archive, dict) or archive.get("path") is None or archive.get("sha256") is None:
                raise ValueError(f"resident call{call} archive receipt is malformed")
            path = _resolve_member(
                root, archive.get("path"), f"call{call}.npz", f"resident call{call}"
            )
            archives.append(
                _archive_record(path, f"resident call{call}", archive.get("sha256"))
            )
        values = _load_npz(path, f"resident call{call}")
        _require_fields(values, {"denoised_bf16", "denoised_f32"}, f"resident call{call}")
        denoised.append(_f32_field(values, "denoised_f32", latent_shape, f"resident call{call}"))
        _bf16_field(values, "denoised_bf16", latent_shape, f"resident call{call}", denoised[-1])
        _finite_number(entry.get("seconds"), f"resident call{call}.seconds")
        stages = entry.get("stages")
        if not isinstance(stages, dict) or not stages:
            raise ValueError(f"resident call{call} stage timings are missing")
        for name, value in stages.items():
            _finite_number(value, f"resident call{call}.stages.{name}")

    return denoised, {"report": report, "weights": weights, "archives": archives}


def _validate_resident_report(
    root: Path,
    report: dict[str, Any],
    input_sha256: str,
    visible_hw: tuple[int, int],
    working_hw: tuple[int, int],
    *,
    plan: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    plan = _resident_plan(report, plan)
    denoised, resident = validate_resident_latents(
        root, report, input_sha256, visible_hw, working_hw, plan=plan
    )
    archives = resident["archives"]
    decoded = report.get("decoded")
    if not isinstance(decoded, dict):
        raise ValueError("resident report has no completed decoder receipt")
    if decoded.get("sha256") is None:
        raise ValueError("resident final archive SHA256 receipt is missing")
    final_path = _resolve_member(root, None, "decoded_final.npz", "resident final")
    archives.append(_archive_record(final_path, "resident final", decoded["sha256"]))
    final_values = _load_npz(final_path, "resident final")
    _require_fields(final_values, {"normalized_final_f32"}, "resident final")
    frames = 17 if plan is None else plan["output_frames"]
    final_shape = (1, frames, 3, visible_hw[0], visible_hw[1])
    final = _f32_field(final_values, "normalized_final_f32", final_shape, "resident final")
    return denoised, final, {"report": report, "final_values": final_values, "archives": archives}


def _cuda45_plan(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the authenticated V2 plan and the native CUDA call ABI."""
    if manifest.get("format") != INPUT_FORMAT_V2 or manifest.get("source_frame_count") != 45:
        raise ValueError("CUDA45 comparison requires the authenticated 45-frame V2 input")
    plan = input_plan(manifest)
    if (
        plan.get("call_count") != 3
        or plan.get("requested_latents") != 12
        or plan.get("padded_latents") != 12
        or plan.get("projection_frames") != 49
        or plan.get("output_frames") != 45
    ):
        raise ValueError("authenticated V2 input plan is not the canonical 45-frame schedule")
    # The native worker records three derived fields that are not part of the
    # common-input V2 plan.  Recompute those small fields here instead of
    # importing the Torch decoder into this CPU/OpenVINO comparator.
    calls = plan.get("calls")
    if not isinstance(calls, list) or len(calls) != 3:
        raise ValueError("authenticated V2 input calls are not canonical")
    native_plan = {**plan, "calls": []}
    for call_index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError("authenticated V2 input call is malformed")
        latent_start = call_index * 4
        latent_stop = latent_start + 4
        expected_last = call_index == len(calls) - 1
        expected_projection_start = 0 if call_index == 0 else 1 + latent_start * 4
        expected_projection_stop = 1 + latent_stop * 4
        if (
            type(call.get("call")) is not int
            or call.get("call") != call_index
            or call.get("latent_start") != latent_start
            or call.get("latent_stop") != latent_stop
            or call.get("projection_start") != expected_projection_start
            or call.get("projection_stop") != expected_projection_stop
            or type(call.get("last")) is not bool
            or call.get("last") != expected_last
        ):
            raise ValueError("authenticated V2 input call chronology changed")
        enriched = dict(call)
        enriched.update(
            {
                "latent_indices": list(range(latent_start, latent_stop)),
                "write_cache": not expected_last,
                "temporal_offset": latent_start,
            }
        )
        native_plan["calls"].append(enriched)
    return plan, native_plan


def _cuda45_report_path(
    path: Path, receipt: Any, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(receipt, dict):
        raise ValueError(f"CUDA45 {label} report receipt is missing")
    reported_path = receipt.get("path")
    try:
        if Path(reported_path).resolve() != path.resolve():
            raise ValueError(f"CUDA45 {label} report path changed")
    except (TypeError, ValueError) as error:
        raise ValueError(f"CUDA45 {label} report path is malformed") from error
    report = _load_json(path, f"CUDA45 {label} report")
    record = _archive_record(path, f"CUDA45 {label} report", receipt.get("sha256"))
    return report, record


def _cuda45_report_input(
    report: dict[str, Any],
    input_root: Path,
    input_sha256: str,
    manifest: dict[str, Any],
    native_plan: dict[str, Any],
    label: str,
) -> None:
    input_info = report.get("input")
    if not isinstance(input_info, dict):
        raise ValueError(f"CUDA45 {label} input receipt is missing")
    if (
        input_info.get("root") != str(input_root)
        or input_info.get("manifest_sha256") != input_sha256
        or input_info.get("format") != INPUT_FORMAT_V2
        or input_info.get("visible_hw") != manifest.get("visible_hw")
        or input_info.get("working_hw") != manifest.get("working_hw")
    ):
        raise ValueError(f"CUDA45 {label} input identity changed")
    input_receipt = manifest.get("inputs")
    if not isinstance(input_receipt, dict) or not isinstance(input_receipt.get("arrays"), dict):
        raise ValueError(f"CUDA45 {label} input archive receipt is missing")
    expected_shapes = {
        name: metadata.get("shape")
        for name, metadata in input_receipt["arrays"].items()
        if isinstance(metadata, dict)
    }
    if len(expected_shapes) != len(input_receipt["arrays"]):
        raise ValueError(f"CUDA45 {label} input array metadata is malformed")
    if not _metadata_equal_exact(input_info.get("array_shapes"), expected_shapes):
        raise ValueError(f"CUDA45 {label} input array shapes changed")
    if not _metadata_equal_exact(report.get("input_plan"), native_plan):
        raise ValueError(f"CUDA45 {label} input plan changed")


def _cuda45_validate_report(
    report: dict[str, Any],
    label: str,
    input_root: Path,
    input_sha256: str,
    manifest: dict[str, Any],
    native_plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        report.get("status") != "complete_native_plus_cuda"
        or report.get("backend") != "native_torch_cuda"
        or report.get("device") != "cuda:0"
        or report.get("device_uuid") != CUDA45_DEVICE_UUID
        or report.get("device_properties_uuid") != CUDA45_DEVICE_UUID.removeprefix("GPU-")
        or report.get("source_commit") != SOURCE_COMMIT
        or report.get("source_root") != str(SOURCE)
    ):
        raise ValueError(f"CUDA45 {label} report provenance changed")
    if not isinstance(report.get("config"), dict) or not isinstance(report.get("stream_policy"), dict):
        raise ValueError(f"CUDA45 {label} algorithmic configuration is missing")
    worker_sha = report.get("worker_sha256")
    if (
        not isinstance(worker_sha, str)
        or len(worker_sha) != 64
        or any(char not in "0123456789abcdef" for char in worker_sha)
    ):
        raise ValueError(f"CUDA45 {label} worker SHA256 receipt is malformed")
    _cuda45_report_input(report, input_root, input_sha256, manifest, native_plan, label)
    weights = _validate_native_weights(report)
    run_count = report.get("run_count")
    calls_per_run = report.get("calls_per_run")
    if run_count != 1 or calls_per_run != 3:
        raise ValueError(f"CUDA45 {label} report must contain one three-call run")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValueError(f"CUDA45 {label} run receipt is malformed")
    run = runs[0]
    if run.get("run") != 0:
        raise ValueError(f"CUDA45 {label} run index changed")
    for name, expected in (
        ("source_frame_count", 45),
        ("requested_latents", 12),
        ("padded_latents", 12),
        ("visible_frame_count", 45),
    ):
        if run.get(name) != expected:
            raise ValueError(f"CUDA45 {label} {name} changed")
    matched = _finite_number(
        run.get("matched_processing_seconds"),
        f"CUDA45 {label}.matched_processing_seconds",
    )
    total = _finite_number(
        report.get("matched_processing_seconds_total"),
        f"CUDA45 {label}.matched_processing_seconds_total",
    )
    if matched <= 0 or total <= 0 or matched != total:
        raise ValueError(f"CUDA45 {label} matched clock is inconsistent")
    calls = run.get("calls")
    if not isinstance(calls, list) or len(calls) != 3:
        raise ValueError(f"CUDA45 {label} call receipt is malformed")
    latent_shape = (1, 16, 4, 272, 480)
    for index, entry in enumerate(calls):
        expected_call = native_plan["calls"][index]
        if (
            not isinstance(entry, dict)
            or entry.get("call") != index
            or entry.get("latent_indices") != expected_call["latent_indices"]
            or entry.get("write_cache") != expected_call["write_cache"]
            or entry.get("last") != expected_call["last"]
            or entry.get("temporal_offset") != expected_call["temporal_offset"]
            or entry.get("projection_frame_range")
            != [expected_call["projection_start"], expected_call["projection_stop"]]
            or entry.get("condition_shape") != [1, 130560, 1536]
            or entry.get("prediction_shape") != list(latent_shape)
            or entry.get("denoised_shape") != list(latent_shape)
            or entry.get("prediction_finite") is not True
            or entry.get("denoised_finite") is not True
        ):
            raise ValueError(f"CUDA45 {label} call{index} schedule/shape changed")
        _finite_number(entry.get("projection_seconds"), f"CUDA45 {label} call{index}.projection_seconds")
        _finite_number(entry.get("denoiser_seconds"), f"CUDA45 {label} call{index}.denoiser_seconds")
    decode = run.get("decode")
    if not isinstance(decode, dict):
        raise ValueError(f"CUDA45 {label} decoder receipt is missing")
    for name, expected in (
        ("status", "native_bf16_taehv_serial_45_complete"),
        ("dtype", "torch.bfloat16"),
        ("parallel", False),
        ("source_frame_count", 45),
        ("requested_latents", 12),
        ("padded_latents", 12),
        ("latent_slots", 12),
        ("startup_trim_frames", 3),
        ("working_hw", [2176, 3840]),
        ("visible_hw", [2160, 3840]),
    ):
        if decode.get(name) != expected:
            raise ValueError(f"CUDA45 {label} decoder {name} changed")
    if decode.get("decoded_sha256") != CUDA45_RAW_SHA256 or decode.get("repeat_bitexact") is not True:
        raise ValueError(f"CUDA45 {label} decoder raw receipt changed")
    return {"report": report, "weights": weights, "run": run, "decode": decode}, {
        "matched_processing_seconds": matched,
        "worker_sha256": worker_sha,
    }


def _cuda45_validate_identity(speed: dict[str, Any], archive: dict[str, Any]) -> None:
    for field in CUDA45_INPUT_PLAN_FIELDS:
        if not _metadata_equal_exact(speed.get(field), archive.get(field)):
            raise ValueError(f"CUDA45 speed/archive {field} identity changed")


def _cuda45_latents(
    speed_path: Path,
    speed_run: dict[str, Any],
    working_hw: tuple[int, int],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    receipt = speed_run.get("latent_archive")
    if not isinstance(receipt, dict):
        raise ValueError("CUDA45 speed latent archive receipt is missing")
    path = _resolve_member(speed_path.parent, receipt.get("path"), "denoised_latents.npz", "CUDA45 latent archive")
    archive_record = _archive_record(path, "CUDA45 latent archive", receipt.get("sha256"))
    expected_names = ("call0_bf16", "call1_bf16", "call2_bf16")
    values = _load_npz(path, "CUDA45 latent archive", expected_order=expected_names)
    _require_fields(values, set(expected_names), "CUDA45 latent archive")
    arrays = receipt.get("arrays")
    if not isinstance(arrays, dict) or tuple(arrays) != expected_names:
        raise ValueError("CUDA45 latent archive array receipt changed")
    shape = (1, 16, 4, working_hw[0] // 8, working_hw[1] // 8)
    denoised: list[np.ndarray] = []
    for name in expected_names:
        value = values[name]
        metadata = arrays[name]
        if (
            not isinstance(metadata, dict)
            or metadata.get("shape") != list(shape)
            or metadata.get("dtype") != "uint16"
            or metadata.get("sha256") != sha256_array(value)
            or value.dtype != np.dtype("uint16")
            or value.shape != shape
            or np.any((value & np.uint16(0x7f80)) == np.uint16(0x7f80))
        ):
            raise ValueError(f"CUDA45 {name} latent storage changed")
        bits = value.astype(np.uint32, copy=False) << np.uint32(16)
        decoded = bits.view(np.float32)
        if not np.isfinite(decoded).all():
            raise ValueError(f"CUDA45 {name} latent values are non-finite")
        denoised.append(decoded)
    return denoised, {"archive": archive_record, "receipt": receipt}


def _cuda45_final(
    archive_path: Path,
    archive_decode: dict[str, Any],
    visible_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    receipt = archive_decode.get("archive")
    if not isinstance(receipt, dict):
        raise ValueError("CUDA45 decoded archive receipt is missing")
    path = _resolve_member(archive_path.parent, receipt.get("path"), "decoded_final.npz", "CUDA45 decoded archive")
    archive_record = _archive_record(path, "CUDA45 decoded archive", receipt.get("sha256"))
    values = _load_npz(path, "CUDA45 decoded archive", expected_order=("normalized_final_f32",))
    final = _f32_field(values, "normalized_final_f32", (1, 45, 3, *visible_hw), "CUDA45 decoded archive")
    metadata = receipt.get("arrays")
    if (
        not isinstance(metadata, dict)
        or tuple(metadata) != ("normalized_final_f32",)
        or not isinstance(metadata["normalized_final_f32"], dict)
        or metadata["normalized_final_f32"].get("shape") != list(final.shape)
        or metadata["normalized_final_f32"].get("dtype") != "float32"
        or metadata["normalized_final_f32"].get("sha256") != sha256_array(final)
        or sha256_array(final) != CUDA45_RAW_SHA256
    ):
        raise ValueError("CUDA45 decoded array receipt changed")
    return final, {"archive": archive_record, "receipt": receipt}


def compare_cuda45(
    resident_root: Path | str,
    input_root: Path | str,
    input_sha256: str,
    output: Path | str,
    *,
    speed_report: Path | str = CUDA45_SPEED_REPORT,
    archive_report: Path | str = CUDA45_ARCHIVE_REPORT,
    reference_link: Path | str = CUDA45_REFERENCE_LINK,
) -> dict[str, Any]:
    """Compare the resident report with the explicitly pinned CUDA45 reference.

    The speed and decoded-archive receipts are intentionally authenticated as
    separate runs.  The speed clock is never replaced by the archive run's
    slower diagnostic clock.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"comparison output must be fresh: {output}")
    resident_root = Path(resident_root).resolve()
    input_root = Path(input_root).resolve()
    speed_path = Path(speed_report).resolve()
    archive_path = Path(archive_report).resolve()
    link_path = Path(reference_link).resolve()
    if speed_path == archive_path:
        raise ValueError("CUDA45 speed and archive reports must be distinct")
    if sha256_file(link_path) != CUDA45_REFERENCE_LINK_SHA256:
        raise ValueError("CUDA45 reference-link receipt fingerprint changed")
    link = _load_json(link_path, "CUDA45 reference link")
    if (
        link.get("scope") != "Read-only CPU reference audit; no GPU use or model inference"
        or link.get("identity_fields_equal") != list(CUDA45_INPUT_PLAN_FIELDS)
        or not isinstance(link.get("decoded_archive"), dict)
        or link["decoded_archive"].get("raw_hash_matches_both_reports") is not True
    ):
        raise ValueError("CUDA45 reference-link identity receipt changed")
    speed, speed_record = _cuda45_report_path(
        speed_path, link.get("speed_report"), "speed"
    )
    archive, archive_record = _cuda45_report_path(
        archive_path, link.get("archive_report"), "archive"
    )
    preflight_hw = preflight_input_geometry(
        input_root, input_sha256, max_working_side=3840, allow_stream=True
    )
    # Authenticate the V2 manifest and metadata while materializing only the
    # small prompt field; projection/noise/RoPE arrays remain on disk.
    arrays, input_manifest = load_input(
        input_root, input_sha256, fields=("prompt_context_f32",)
    )
    del arrays
    visible_hw, working_hw = _validate_input_manifest(input_manifest, allow_v2=True)
    if working_hw != preflight_hw:
        raise ValueError("common input geometry changed during authentication")
    manifest_plan, native_plan = _cuda45_plan(input_manifest)
    speed_meta, speed_clock = _cuda45_validate_report(
        speed, "speed", input_root, input_sha256, input_manifest, native_plan
    )
    archive_meta, archive_clock = _cuda45_validate_report(
        archive, "archive", input_root, input_sha256, input_manifest, native_plan
    )
    _cuda45_validate_identity(speed, archive)
    if (
        link["speed_report"].get("status") != speed.get("status")
        or link["archive_report"].get("status") != archive.get("status")
        or link["speed_report"].get("matched_processing_seconds_total")
        != speed_clock["matched_processing_seconds"]
        or link["archive_report"].get("matched_processing_seconds_total")
        != archive_clock["matched_processing_seconds"]
    ):
        raise ValueError("CUDA45 reference-link clock/provenance changed")
    speed_latents, speed_latent_extra = _cuda45_latents(
        speed_path, speed_meta["run"], working_hw
    )
    speed_decode = speed_meta["decode"]
    archive_decode = archive_meta["decode"]
    if speed_decode.get("archive") is not None:
        raise ValueError("CUDA45 speed receipt unexpectedly owns a decoded archive")
    if speed_decode.get("decoded_sha256") != archive_decode.get("decoded_sha256"):
        raise ValueError("CUDA45 speed/archive decoded raw hashes differ")
    native_final, archive_final_extra = _cuda45_final(
        archive_path, archive_decode, visible_hw
    )
    decoded_link = link["decoded_archive"]
    if (
        decoded_link.get("path") != archive_final_extra["archive"]["path"]
        or decoded_link.get("sha256") != archive_final_extra["archive"]["sha256"]
        or decoded_link.get("array_sha256") != CUDA45_RAW_SHA256
        or decoded_link.get("shape") != list(native_final.shape)
        or decoded_link.get("dtype") != "<f4"
        or decoded_link.get("array_bytes") != native_final.nbytes
    ):
        raise ValueError("CUDA45 decoded archive link changed")
    resident_report = _load_json(resident_root / "results.json", "resident results.json")
    resident_worker_sha = resident_report.get("worker_sha256")
    if (
        not isinstance(resident_worker_sha, str)
        or len(resident_worker_sha) != 64
        or any(char not in "0123456789abcdef" for char in resident_worker_sha)
    ):
        raise ValueError("resident worker SHA256 receipt is malformed")
    _require_resident_identity(
        archive, resident_report, input_manifest["source"], "CUDA45 comparison"
    )
    resident_weights = _validate_resident_weights(resident_report)
    resident_latents, resident_final, resident_extra = _validate_resident_report(
        resident_root, resident_report, input_sha256, visible_hw, working_hw,
        plan=manifest_plan,
    )
    if len(resident_latents) != len(speed_latents):
        raise ValueError("resident/CUDA45 latent call count changed")
    latent_calls = [
        {"call": call, **_latent_metrics(speed_latents[call], resident_latents[call])}
        for call in range(len(speed_latents))
    ]
    resident_calls = []
    for call, entry in enumerate(resident_report["calls"]):
        resident_calls.append({
            "call": call,
            "seconds": _finite_number(entry["seconds"], f"resident call{call}.seconds"),
            "stages": {
                name: _finite_number(value, f"resident call{call}.stages.{name}")
                for name, value in entry["stages"].items()
            },
        })
    result = {
        "format": "flashvsr-plus-native-cuda45-vs-resident-comparison-v1",
        "status": "comparison_cuda45_complete",
        "scope": "authenticated native CUDA45 reference versus resident OpenVINO on the same V2 45-frame input",
        "quality_gate": "not applied",
        "input": {
            "root": str(input_root),
            "manifest_sha256": input_sha256,
            "visible_hw": list(visible_hw),
            "working_hw": list(working_hw),
            "source_commit": input_manifest["source"]["commit"],
            "source_root": input_manifest["source"]["root"],
            "plan": manifest_plan,
            "source_authenticated_by_load_input": True,
        },
        "native_cuda": {
            "device_uuid": CUDA45_DEVICE_UUID,
            "speed": {
                "report": str(speed_path),
                "report_sha256": speed_record["sha256"],
                "worker_sha256": speed_clock["worker_sha256"],
                "matched_processing_seconds": speed_clock["matched_processing_seconds"],
                "latent_archive": speed_latent_extra["archive"],
            },
            "decoded_archive": {
                "report": str(archive_path),
                "report_sha256": archive_record["sha256"],
                "worker_sha256": archive_clock["worker_sha256"],
                "matched_processing_seconds": archive_clock["matched_processing_seconds"],
                "archive": archive_final_extra["archive"],
                "raw_sha256": CUDA45_RAW_SHA256,
            },
            "reference_link": {
                "path": str(link_path),
                "sha256": CUDA45_REFERENCE_LINK_SHA256,
            },
            "weights": speed_meta["weights"],
        },
        "resident": {
            "root": str(resident_root),
            "status": resident_report["status"],
            "device": resident_report.get("device"),
            "runtime": resident_report.get("runtime"),
            "worker_sha256": resident_worker_sha,
            "weights": resident_weights,
        },
        "latent": {"calls": latent_calls},
        "decoded": {"normalized": _psnr(native_final, resident_final, 2.0)},
        "archives": {
            "native_cuda_speed_report": speed_record,
            "native_cuda_latent": speed_latent_extra["archive"],
            "native_cuda_archive_report": archive_record,
            "native_cuda_decoded": archive_final_extra["archive"],
            "resident": resident_extra["archives"],
        },
        "timings": {
            "native_cuda": {
                "matched_processing_seconds": speed_clock["matched_processing_seconds"],
                "decoded_archive_report_matched_processing_seconds": archive_clock["matched_processing_seconds"],
                "note": "Separate complete speed and decoded-archive runs; the archive clock is not added to the speed clock.",
            },
            "resident": {
                "matched_processing_seconds": _finite_number(
                    resident_report.get("matched_processing_seconds"),
                    "resident.matched_processing_seconds"),
                "processing_clock": resident_report.get("processing_clock"),
                "setup_seconds": _finite_number(resident_report.get("setup_seconds"), "resident.setup_seconds"),
                "calls": resident_calls,
                "decode_seconds": _finite_number(resident_report.get("decode_seconds"), "resident.decode_seconds"),
                "wall_seconds": _finite_number(resident_report.get("wall_seconds"), "resident.wall_seconds"),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


_METRIC_CHUNK_ELEMENTS = 1 << 20


def _metric_chunks(
    native: np.ndarray,
    resident: np.ndarray,
    *,
    chunk_elements: int = _METRIC_CHUNK_ELEMENTS,
):
    """Yield bounded FP64 reference/difference chunks in C order.

    ``nditer`` keeps conversion bounded even when either input is a strided
    view.  Consumers must finish with a yielded pair before requesting the
    next pair; the iterator may reuse its external buffer for the next chunk.
    """
    if native.shape != resident.shape:
        raise ValueError(f"metric shape mismatch: {native.shape} != {resident.shape}")
    if (
        isinstance(chunk_elements, (bool, np.bool_))
        or not isinstance(chunk_elements, (int, np.integer))
        or int(chunk_elements) <= 0
    ):
        raise ValueError("chunk_elements must be a positive integer")
    iterator = np.nditer(
        (native, resident),
        flags=("external_loop", "buffered"),
        op_flags=(("readonly",), ("readonly",)),
        order="C",
        buffersize=int(chunk_elements),
    )
    for native_chunk, resident_chunk in iterator:
        reference = np.asarray(native_chunk, dtype=np.float64)
        difference = np.asarray(resident_chunk, dtype=np.float64) - reference
        yield reference, difference


def _latent_metrics(native: np.ndarray, resident: np.ndarray) -> dict[str, float]:
    if native.shape != resident.shape:
        raise ValueError(f"latent shape mismatch: {native.shape} != {resident.shape}")
    reference_squared = 0.0
    difference_squared = 0.0
    reference_max_abs = 0.0
    difference_max_abs = 0.0
    for reference, difference in _metric_chunks(native, resident):
        reference_squared += float(np.sum(reference * reference, dtype=np.float64))
        difference_squared += float(np.sum(difference * difference, dtype=np.float64))
        reference_max_abs = max(reference_max_abs, float(np.abs(reference).max()))
        difference_max_abs = max(difference_max_abs, float(np.abs(difference).max()))
    reference_l2 = math.sqrt(reference_squared)
    difference_l2 = math.sqrt(difference_squared)
    return {
        "l2": difference_l2,
        "relative_l2": float(difference_l2 / max(reference_l2, 1e-30)),
        "max_abs": difference_max_abs,
        "normalized_max_abs": float(difference_max_abs / max(reference_max_abs, 1e-30)),
        "reference_l2": reference_l2,
        "reference_max_abs": reference_max_abs,
    }


def _psnr(native: np.ndarray, resident: np.ndarray, data_range: float) -> dict[str, Any]:
    if native.shape != resident.shape:
        raise ValueError(f"decoded shape mismatch: {native.shape} != {resident.shape}")
    frames: list[dict[str, Any]] = []
    overall_squared = 0.0
    overall_elements = 0
    for index in range(native.shape[1]):
        frame_squared = 0.0
        frame_elements = 0
        for _, difference in _metric_chunks(native[:, index], resident[:, index]):
            frame_squared += float(np.sum(difference * difference, dtype=np.float64))
            frame_elements += difference.size
        overall_squared += frame_squared
        overall_elements += frame_elements
        mse = float(frame_squared / frame_elements)
        frames.append({
            "frame": index,
            "mse": mse,
            "psnr_db": None if mse == 0 else float(10.0 * math.log10((data_range * data_range) / mse)),
        })
    overall_mse = float(overall_squared / overall_elements)
    finite_psnr = [item["psnr_db"] for item in frames if item["psnr_db"] is not None]
    worst = min(frames, key=lambda item: item["psnr_db"] if item["psnr_db"] is not None else math.inf)
    return {
        "data_range": data_range,
        "mse": overall_mse,
        "psnr_db": None if overall_mse == 0 else float(10.0 * math.log10((data_range * data_range) / overall_mse)),
        "exact": overall_mse == 0,
        "per_frame": frames,
        "worst_frame": worst,
        "finite_frame_count": len(finite_psnr),
    }


def _optional_raw_metrics(
    native_values: dict[str, np.ndarray],
    resident_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    native = native_values.get("raw_final_f32")
    resident = resident_values.get("raw_final_f32")
    if native is None or resident is None:
        if native is None and resident is None:
            reason = "raw final arrays are not present in either current worker ABI"
        elif native is None:
            reason = "raw final array is absent from the native archive"
        else:
            reason = "raw final array is absent from the resident archive"
        return {
            "available": False,
            "reason": reason,
        }
    if native.dtype != np.dtype("float32") or resident.dtype != np.dtype("float32"):
        raise ValueError("raw final arrays must be float32")
    if not np.isfinite(native).all() or not np.isfinite(resident).all():
        raise ValueError("raw final arrays contain non-finite values")
    return {"available": True, **_psnr(native, resident, 1.0)}


def _timings(native_report: dict[str, Any], resident_report: dict[str, Any]) -> dict[str, Any]:
    native_calls = []
    for call, entry in enumerate(native_report["calls"]):
        projection = _finite_number(entry.get("projection_seconds"), f"native call{call}.projection_seconds")
        denoiser = _finite_number(entry.get("denoiser_seconds"), f"native call{call}.denoiser_seconds")
        native_calls.append({
            "call": call,
            "projection_seconds": projection,
            "denoiser_seconds": denoiser,
            "measured_denoiser_pipeline_seconds": projection + denoiser,
        })
    resident_calls = []
    for call, entry in enumerate(resident_report["calls"]):
        stages = {name: _finite_number(value, f"resident call{call}.stages.{name}") for name, value in entry["stages"].items()}
        resident_calls.append({"call": call, "seconds": _finite_number(entry["seconds"], f"resident call{call}.seconds"), "stages": stages})
    result: dict[str, Any] = {
        "native": {
            "model_load_cpu_seconds": _finite_number(native_report.get("model_load_cpu_seconds"), "native.model_load_cpu_seconds"),
            "model_transfer_seconds": _finite_number(native_report.get("model_transfer_seconds"), "native.model_transfer_seconds"),
            "calls": native_calls,
            "decode_seconds": _finite_number(native_report["decode"].get("seconds"), "native.decode.seconds"),
            "wall_seconds": _finite_number(native_report.get("diagnostic_wall_seconds"), "native.diagnostic_wall_seconds"),
        },
        "resident": {
            "setup_seconds": _finite_number(resident_report.get("setup_seconds"), "resident.setup_seconds"),
            "calls": resident_calls,
            "decode_seconds": _finite_number(resident_report.get("decode_seconds"), "resident.decode_seconds"),
            "wall_seconds": _finite_number(resident_report.get("wall_seconds"), "resident.wall_seconds"),
        },
        "note": "Measured bounded 512x256 run timings only; no 4K/frame throughput or quality extrapolation.",
    }
    return result


def compare(
    native_root: Path | str,
    resident_root: Path | str,
    input_root: Path | str,
    input_sha256: str,
    output: Path | str,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"comparison output must be fresh: {output}")
    native_root = Path(native_root).resolve()
    resident_root = Path(resident_root).resolve()
    input_root = Path(input_root).resolve()
    preflight_hw = preflight_input_geometry(input_root, input_sha256)
    _arrays, input_manifest = load_input(input_root, input_sha256)
    visible_hw, working_hw = _validate_input_manifest(input_manifest)
    if working_hw != preflight_hw:
        raise ValueError("common input geometry changed during authentication")
    native_report = _load_json(native_root / "results.json", "native results.json")
    resident_report = _load_json(resident_root / "results.json", "resident results.json")
    source = input_manifest["source"]
    if native_report.get("source_commit") != source["commit"]:
        raise ValueError("native report source commit does not match common input")
    if native_report.get("source_root") != source["root"]:
        raise ValueError("native report source root does not match common input")
    for label, report in (("native", native_report), ("resident", resident_report)):
        digest = report.get("worker_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"{label} worker SHA256 receipt is malformed")
    _require_resident_identity(
        native_report, resident_report, source, "legacy comparison"
    )
    native_weights = _validate_native_weights(native_report)
    resident_weights = _validate_resident_weights(resident_report)
    native_latents, native_final, native_extra = _validate_native_report(
        native_root, native_report, input_sha256, visible_hw, working_hw
    )
    resident_latents, resident_final, resident_extra = _validate_resident_report(
        resident_root, resident_report, input_sha256, visible_hw, working_hw
    )
    latent_calls = [
        {"call": call, **_latent_metrics(native_latents[call], resident_latents[call])}
        for call in range(CALLS)
    ]
    result = {
        "format": FORMAT,
        "status": "comparison_complete",
        "scope": "native source versus resident OpenVINO on the same authenticated bounded common input",
        "quality_gate": "not applied",
        "input": {
            "root": str(input_root),
            "manifest_sha256": input_sha256,
            "visible_hw": list(visible_hw),
            "working_hw": list(working_hw),
            "source_commit": source["commit"],
            "source_root": source["root"],
            "source_authenticated_by_load_input": True,
        },
        "native": {
            "root": str(native_root),
            "status": native_report["status"],
            "device": native_report.get("device"),
            "device_uuid": native_report.get("device_uuid"),
            "worker_sha256": native_report.get("worker_sha256"),
            "weights": native_weights,
        },
        "resident": {
            "root": str(resident_root),
            "status": resident_report["status"],
            "device": resident_report.get("device"),
            "runtime": resident_report.get("runtime"),
            "worker_sha256": resident_report.get("worker_sha256"),
            "weights": resident_weights,
        },
        "latent": {"calls": latent_calls},
        "decoded": {
            "normalized": _psnr(native_final, resident_final, 2.0),
            "raw": _optional_raw_metrics(native_extra["final_values"], resident_extra["final_values"]),
        },
        "archives": {
            "native": native_extra["archives"],
            "resident": resident_extra["archives"],
            "note": (
                "Native call/final hashes are verified against worker receipts. "
                "Resident call hashes are verified against per-call worker receipts "
                "when present; older resident reports remain comparator-only byte "
                "hash evidence. Resident final hash is verified against its report receipt."
            ),
        },
        "timings": _timings(native_report, resident_report),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, help="legacy native source output root")
    parser.add_argument(
        "--native-cuda45",
        action="store_true",
        help="explicitly compare against the pinned authenticated CUDA45 reference",
    )
    parser.add_argument("--cuda45-speed-report", type=Path, default=CUDA45_SPEED_REPORT)
    parser.add_argument("--cuda45-archive-report", type=Path, default=CUDA45_ARCHIVE_REPORT)
    parser.add_argument("--cuda45-reference-link", type=Path, default=CUDA45_REFERENCE_LINK)
    parser.add_argument("--resident", type=Path, required=True, help="resident OpenVINO output root")
    parser.add_argument("--input", type=Path, required=True, help="authenticated common-input root")
    parser.add_argument("--input-sha256", required=True, help="SHA256 of common input manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="fresh machine-readable comparison JSON")
    args = parser.parse_args(argv)
    try:
        if args.native_cuda45:
            if args.native is not None:
                parser.error("--native cannot be combined with --native-cuda45")
            result = compare_cuda45(
                args.resident,
                args.input,
                args.input_sha256,
                args.output,
                speed_report=args.cuda45_speed_report,
                archive_report=args.cuda45_archive_report,
                reference_link=args.cuda45_reference_link,
            )
        else:
            if args.native is None:
                parser.error("--native is required unless --native-cuda45 is selected")
            result = compare(args.native, args.resident, args.input, args.input_sha256, args.output)
    except (FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve())}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
