#!/usr/bin/env python3
"""Validate one dynamic-window SeedVR2 block on the exact Arc B580."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch


HERE = Path(__file__).resolve().parent
TRIAL = HERE.parents[1]
DEVICE = "GPU.1"
SHAPE = (2, 135, 240)


def load_helper(path: Path):
    spec = importlib.util.spec_from_file_location("otc_seedvr_dynamic_window", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("block", choices=("block0", "block16"))
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.receipt}")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    suffix = "0" if args.block == "block0" else "16"
    helper = load_helper(HERE / f"dynamic_window_reverse_gather_block{suffix}_export.py")
    model_path = HERE / f"block{suffix}_dynamic_window_reverse_gather_fp16.xml"
    sys.path.insert(0, str(TRIAL / "source"))
    from src.models.dit_3b.rope import get_na_rope
    from src.models.dit_3b.window import get_window_op

    rope = get_na_rope("mmrope3d", helper.HEAD_DIM)
    slices = get_window_op(helper.WINDOW_METHOD)(SHAPE, helper.WINDOW)
    max_video_len = max(
        (time_slice.stop - time_slice.start)
        * (height_slice.stop - height_slice.start)
        * (width_slice.stop - width_slice.start)
        for time_slice, height_slice, width_slice in slices
    )
    contract = helper.make_window_contract(
        SHAPE,
        max_video_len=max_video_len,
        txt_len=helper.TXT_TOKENS,
        rope=rope,
        torch=torch,
    )

    core = ov.Core()
    full_name = core.get_property(DEVICE, "FULL_DEVICE_NAME")
    device_type = str(core.get_property(DEVICE, "DEVICE_TYPE"))
    if "B580" not in full_name or "DISCRETE" not in device_type:
        raise RuntimeError(f"{DEVICE} is not the Arc B580: {full_name} / {device_type}")

    model = core.read_model(model_path)
    # Intel's GPU plugin cannot lower this block with fully dynamic reduction
    # axes. Specialize the same IR to the export's actual token/window sizes;
    # this avoids the former worst-case padding without creating more weights.
    model.reshape({
        "vid": [SHAPE[0] * SHAPE[1] * SHAPE[2], helper.INNER_DIM],
        "txt": [helper.TXT_TOKENS, helper.INNER_DIM],
        "emb": [1, helper.EMB_DIM],
        "window_idx": list(contract["window_idx"].shape),
        "window_key_mask": list(contract["window_key_mask"].shape),
        "window_rope_freqs": list(contract["window_rope_freqs"].shape),
        "window_reverse_idx": list(contract["window_reverse_idx"].shape),
    })
    started = time.perf_counter()
    compiled = core.compile_model(model, DEVICE, {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": ov.Type.f16,
        "NUM_STREAMS": "1",
    })
    compile_seconds = time.perf_counter() - started
    execution_devices = list(compiled.get_property("EXECUTION_DEVICES"))
    if execution_devices != [DEVICE]:
        raise RuntimeError(f"unexpected execution devices: {execution_devices}")

    rng = np.random.default_rng(2026091916 + int(suffix))
    inputs = {
        "vid": rng.standard_normal((SHAPE[0] * SHAPE[1] * SHAPE[2], helper.INNER_DIM), dtype=np.float32) * 0.02,
        "txt": rng.standard_normal((helper.TXT_TOKENS, helper.INNER_DIM), dtype=np.float32) * 0.02,
        "emb": rng.standard_normal((1, helper.EMB_DIM), dtype=np.float32) * 0.02,
        "window_idx": contract["window_idx"].numpy(),
        "window_key_mask": contract["window_key_mask"].numpy(),
        "window_rope_freqs": contract["window_rope_freqs"].numpy(),
        "window_reverse_idx": contract["window_reverse_idx"].numpy(),
    }
    request = compiled.create_infer_request()
    started = time.perf_counter()
    first = request.infer(inputs)
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    second = request.infer(inputs)
    warm_seconds = time.perf_counter() - started

    outputs = []
    for port in compiled.outputs:
        value = np.asarray(first[port], dtype=np.float32)
        repeated = np.asarray(second[port], dtype=np.float32)
        outputs.append({
            "name": port.get_any_name(),
            "shape": list(value.shape),
            "finite": bool(np.isfinite(value).all()),
            "max_abs": float(np.max(np.abs(value))),
            "rms": float(np.sqrt(np.mean(np.square(value)))),
            "repeat_max_abs": float(np.max(np.abs(repeated - value))),
        })
    if not all(item["finite"] for item in outputs):
        raise RuntimeError("non-finite output")

    receipt = {
        "status": "pass",
        "block": args.block,
        "shape": list(SHAPE),
        "device": DEVICE,
        "full_device_name": full_name,
        "device_type": device_type,
        "execution_devices": execution_devices,
        "model": str(model_path),
        "compile_seconds": compile_seconds,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "window_count": int(contract["window_count"]),
        "max_video_len": int(contract["max_video_len"]),
        "max_attention_len": int(contract["max_attention_len"]),
        "outputs": outputs,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "openvino": ov.__version__,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
