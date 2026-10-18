#!/usr/bin/env python3
"""Finite GPU.1 validation for the fused SeedVR2 block envelope.

The caller owns host coordination and supplies a fresh marker.  This runner
never discovers or selects another device, and it requires an explicit new
receipt path so CLI inspection cannot overwrite prior evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
DEVICE = "GPU.1"
BLOCKS = (0, 16, 31)
SHAPE = (2, 135, 240)
SOURCE_TOKENS = math.prod(SHAPE)
VID_DIM = 2560
TXT_TOKENS = 58
EMB_DIM = 15360
WINDOW_CAPACITY = 208
MAX_VIDEO_LEN = 405
ATTENTION_LEN = 463
ROPE_WIDTH = 126
MAX_MARKER_AGE_SECONDS = 60.0
TARGET = {
    "bdf": "0000:03:00.0",
    "driver": "xe",
    "render_node": "renderD128",
    "device": "B580",
    "openvino_selector": DEVICE,
}


class ContractError(RuntimeError):
    pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preflight-marker", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    return result


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError("local helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss() -> dict[str, int]:
    kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"peak_rss_kib": kib, "peak_rss_bytes": kib * 1024}


def read_marker(path: Path, now: float | None = None) -> dict[str, Any]:
    stat = path.stat()
    age = (time.time() if now is None else now) - stat.st_mtime
    if not path.is_file() or age < 0 or age > MAX_MARKER_AGE_SECONDS:
        raise ContractError("preflight marker is missing or stale")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("status") not in {"pass", "ready"}:
        raise ContractError("preflight marker is not affirmative")
    target = payload.get("target", payload)
    if not isinstance(target, Mapping):
        raise ContractError("preflight target is invalid")
    normalized = {
        "bdf": target.get("bdf", target.get("pci_bdf")),
        "driver": target.get("driver"),
        "render_node": str(target.get("render_node", "")).rsplit("/", 1)[-1],
        "device": target.get("device", target.get("pci_device")),
        "openvino_selector": target.get("openvino_selector", payload.get("openvino_selector")),
    }
    if "B580" in str(normalized["device"]):
        normalized["device"] = "B580"
    if normalized != TARGET:
        raise ContractError("preflight target does not match GPU.1 B580")
    return {"age_seconds": round(age, 3), "target": normalized}


def artifact_inventory(block: int) -> dict[str, Any]:
    xml = HERE / f"block{block}_fused_fixed_envelope_fp16.xml"
    binary = xml.with_suffix(".bin")
    source_receipt = HERE / f"receipt.fused_fixed_envelope_block{block}.json"
    expected = json.loads(source_receipt.read_text(encoding="utf-8"))
    result = {
        "xml": {"name": xml.name, "bytes": xml.stat().st_size, "sha256": sha256(xml)},
        "bin": {"name": binary.name, "bytes": binary.stat().st_size, "sha256": sha256(binary)},
    }
    saved = expected["saved_ir"]
    if result["xml"]["sha256"] != saved["xml_sha256"] or result["bin"]["sha256"] != saved["bin_sha256"]:
        raise ContractError("fused IR hash differs from its CPU parity receipt")
    return result


def metadata(block: int, np: Any, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    helper = load_module(
        PARENT / f"dynamic_window_reverse_gather_block{block}_export.py",
        f"b580_fused_helper_{block}",
    )
    decomposition = load_module(
        PARENT / "chunked_attention_decomposition_export.py",
        f"b580_fused_decomposition_{block}",
    )
    fused = load_module(HERE / "fused_fixed_envelope_export.py", f"b580_fused_contract_{block}")
    if str(helper.SOURCE) not in sys.path:
        sys.path.insert(0, str(helper.SOURCE))
    from src.models.dit_3b.rope import get_na_rope

    summary = decomposition.window_metadata(helper, SHAPE)
    contract = helper.make_window_contract(
        SHAPE,
        max_video_len=int(summary["max_video_len"]),
        txt_len=TXT_TOKENS,
        rope=get_na_rope("mmrope3d", 128),
        torch=torch,
    )
    padded = fused.pad_window_metadata(contract, torch)
    coverage = fused.validate_padded_metadata(padded, torch)
    arrays = {
        name: value.detach().cpu().numpy()
        for name, value in padded.items()
        if hasattr(value, "detach")
    }
    if arrays["window_idx"].shape != (WINDOW_CAPACITY, MAX_VIDEO_LEN):
        raise ContractError("fixed window envelope shape is invalid")
    return arrays, {
        "window_method": helper.WINDOW_METHOD,
        "actual_window_count": int(padded["actual_window_count"]),
        "dummy_window_count": int(padded["dummy_window_count"]),
        "coverage": coverage,
    }


def random_inputs(seed: int, np: Any) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "vid": rng.standard_normal((SOURCE_TOKENS, VID_DIM), dtype=np.float32) * np.float32(0.02),
        "txt": rng.standard_normal((TXT_TOKENS, VID_DIM), dtype=np.float32) * np.float32(0.02),
        "emb": rng.standard_normal((1, EMB_DIM), dtype=np.float32) * np.float32(0.02),
    }


def run_block(block: int, ov: Any, np: Any, torch: Any) -> dict[str, Any]:
    inventory = artifact_inventory(block)
    arrays, contract = metadata(block, np, torch)
    values = random_inputs(2026091800 + block, np)
    values.update({
        "window_idx": arrays["window_idx"],
        "window_key_mask": arrays["window_key_mask"],
        "window_rope_freqs": arrays["window_rope_freqs"],
        "window_valid": arrays["window_valid"],
        "window_reverse_idx": arrays["window_reverse_idx"],
    })
    core = ov.Core()
    model = core.read_model(str(HERE / inventory["xml"]["name"]))
    expected_inputs = {
        "vid": (SOURCE_TOKENS, VID_DIM),
        "txt": (TXT_TOKENS, VID_DIM),
        "emb": (1, EMB_DIM),
        "window_idx": (WINDOW_CAPACITY, MAX_VIDEO_LEN),
        "window_key_mask": (WINDOW_CAPACITY, ATTENTION_LEN),
        "window_rope_freqs": (WINDOW_CAPACITY, ATTENTION_LEN, ROPE_WIDTH),
        "window_valid": (WINDOW_CAPACITY,),
        "window_reverse_idx": (SOURCE_TOKENS,),
    }
    if {port.get_any_name() for port in model.inputs} != set(expected_inputs):
        raise ContractError("fused IR input names differ from the runner contract")
    model.reshape({name: ov.PartialShape(shape) for name, shape in expected_inputs.items()})
    compile_started = time.perf_counter()
    compiled = core.compile_model(model, DEVICE, {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": ov.Type.f16,
        "NUM_STREAMS": "1",
    })
    compile_seconds = time.perf_counter() - compile_started
    execution_devices = [str(value) for value in compiled.get_property("EXECUTION_DEVICES")]
    if execution_devices != [DEVICE]:
        raise ContractError("fused graph did not compile exclusively for GPU.1")
    request = compiled.create_infer_request()
    infer_started = time.perf_counter()
    result = request.infer(values)
    infer_seconds = time.perf_counter() - infer_started
    outputs = [np.array(result[port], copy=True) for port in compiled.outputs]
    if [tuple(value.shape) for value in outputs] != [(SOURCE_TOKENS, VID_DIM), (TXT_TOKENS, VID_DIM)]:
        raise ContractError("fused graph output shape is invalid")
    if not all(bool(np.isfinite(value).all()) for value in outputs):
        raise ContractError("fused graph output is non-finite")
    count = int(contract["actual_window_count"])
    dummy_values = dict(values)
    dummy_values["window_idx"] = values["window_idx"].copy()
    dummy_values["window_key_mask"] = values["window_key_mask"].copy()
    dummy_values["window_rope_freqs"] = values["window_rope_freqs"].copy()
    dummy_values["window_idx"][count:, :] = SOURCE_TOKENS - 1
    dummy_values["window_key_mask"][count:, :] = False
    dummy_values["window_key_mask"][count:, 0] = True
    dummy_values["window_rope_freqs"][count:, :, 0] = np.float32(0.375)
    dummy_started = time.perf_counter()
    dummy_result = request.infer(dummy_values)
    dummy_seconds = time.perf_counter() - dummy_started
    dummy_outputs = [np.array(dummy_result[port], copy=True) for port in compiled.outputs]
    max_dummy_error = max(float(np.max(np.abs(a - b))) for a, b in zip(outputs, dummy_outputs))
    if max_dummy_error != 0.0:
        raise ContractError("dummy fixed-envelope rows affected fused output")
    checksums = [hashlib.sha256(value.tobytes()).hexdigest() for value in outputs]
    result_receipt = {
        "block": block,
        "status": "pass",
        "device": DEVICE,
        "execution_devices": execution_devices,
        "artifacts": inventory,
        "contract": contract,
        "compile_seconds": compile_seconds,
        "infer_seconds": infer_seconds,
        "dummy_infer_seconds": dummy_seconds,
        "dummy_max_abs_error": max_dummy_error,
        "output_shapes": [list(value.shape) for value in outputs],
        "output_sha256": checksums,
        **peak_rss(),
    }
    del request, compiled, model, core, outputs, dummy_outputs
    gc.collect()
    return result_receipt


def write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.receipt.exists():
        raise ContractError("receipt already exists")
    receipt: dict[str, Any] = {
        "runner": "b580_fused_fixed_envelope_runner",
        "status": "failure",
        "device": DEVICE,
        "shape": list(SHAPE),
        "blocks": list(BLOCKS),
    }
    try:
        receipt["preflight"] = read_marker(args.preflight_marker)
        import numpy as np
        import openvino as ov
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        receipt["block_receipts"] = [run_block(block, ov, np, torch) for block in BLOCKS]
        receipt["status"] = "pass"
    except Exception as exc:
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = "sanitized validation failure"
    receipt.update(peak_rss())
    write_receipt(args.receipt, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": args.receipt.name}, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
