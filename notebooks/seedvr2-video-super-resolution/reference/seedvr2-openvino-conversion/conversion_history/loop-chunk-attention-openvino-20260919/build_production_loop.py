#!/usr/bin/env python3
"""Save the production static attention-only Loop graph without compiling it."""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import sys
from typing import Any

import numpy as np
import openvino as ov

from loop_chunk_attention import (
    HERE,
    SHARED_IR,
    build_loop_tail_model,
    model_io,
    sha256_file,
    validate_saved_loop_ir,
)


TRIAL_ROOT = HERE.parents[1]
SOURCE = Path(os.environ.get("SEEDVR_SOURCE", TRIAL_ROOT / "source_snapshot"))
SHAPE = (2, 240, 135)
WINDOW = (4, 3, 3)
WINDOW_METHODS = {
    "even": "720pwin_by_size_bysize",
    "odd": "720pswin_by_size_bysize",
}
CHUNK_SIZE = 16
MAX_VIDEO_LEN = 405
EVEN_XML = HERE / "production_static_t2_h240_w135_even_loop_chunk16_fp16.xml"
ODD_XML = HERE / "production_static_t2_h240_w135_odd_loop_chunk16_fp16.xml"
RECEIPT_PATH = HERE / "production_static_graph_receipt.json"
INVOCATION = (
    "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "
    "PYTHONPATH=<external-runtime> "
    "nice -n 15 taskset -c 0-3 "
    "<external-python> "
    "build_production_loop.py"
)


def _window_geometry(method: str) -> tuple[int, int, np.ndarray]:
    sys.dont_write_bytecode = True
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from src.models.dit_3b.window import get_window_op

    slices = get_window_op(method)(SHAPE, WINDOW)
    grid = np.arange(math.prod(SHAPE), dtype=np.int64).reshape(SHAPE)
    lengths: list[int] = []
    reverse_idx = np.full((math.prod(SHAPE),), -1, dtype=np.int64)
    for window_number, window_slice in enumerate(slices):
        indices = grid[window_slice].reshape(-1)
        lengths.append(int(indices.size))
        positions = window_number * MAX_VIDEO_LEN + np.arange(indices.size, dtype=np.int64)
        if np.any(reverse_idx[indices] >= 0):
            raise AssertionError("source window geometry contains overlapping tokens")
        reverse_idx[indices] = positions
    if np.any(reverse_idx < 0) or np.unique(reverse_idx).size != reverse_idx.size:
        raise AssertionError("source window geometry does not cover every token exactly once")
    return len(slices), max(lengths), reverse_idx


def _build_variant(core: Any, label: str, window_method: str, output_xml: Path) -> dict[str, Any]:
    actual_windows, actual_max_video_len, reverse_idx = _window_geometry(window_method)
    if actual_max_video_len != MAX_VIDEO_LEN:
        raise AssertionError(
            f"{label}: exact shape max window len {actual_max_video_len} != {MAX_VIDEO_LEN}"
        )
    loop_windows = (actual_windows // CHUNK_SIZE) * CHUNK_SIZE
    tail_windows = actual_windows - loop_windows
    graph = build_loop_tail_model(
        core,
        source_tokens=math.prod(SHAPE),
        total_windows=actual_windows,
        max_video_len=MAX_VIDEO_LEN,
        chunk_size=CHUNK_SIZE,
        reverse_idx=reverse_idx,
        qkv_type=ov.Type.f16,
        video_output_type=ov.Type.f16,
        include_text_stats=False,
        name=f"seedvr2_production_static_attention_loop_{label}_t2_h240_w135",
    )

    # This saves the static graph only.  Do not add a compile_model or infer call here.
    ov.save_model(graph["model"], str(output_xml), compress_to_fp16=True)
    saved_structure = validate_saved_loop_ir(output_xml)
    if saved_structure["sdpa_count"] != 2:
        raise AssertionError(f"{label}: expected one Loop-body SDPA and one tail SDPA: {saved_structure}")
    saved = core.read_model(output_xml)

    full_meta_names = ("window_idx", "window_key_mask", "window_rope_freqs")
    expected_io = {
        "vid_q": (f"[{math.prod(SHAPE)},20,128]", "<Type: 'float16'>"),
        "vid_k": (f"[{math.prod(SHAPE)},20,128]", "<Type: 'float16'>"),
        "vid_v": (f"[{math.prod(SHAPE)},20,128]", "<Type: 'float16'>"),
        "txt_q": ("[58,20,128]", "<Type: 'float16'>"),
        "txt_k": ("[58,20,128]", "<Type: 'float16'>"),
        "txt_v": ("[58,20,128]", "<Type: 'float16'>"),
        full_meta_names[0]: (f"[{actual_windows},{MAX_VIDEO_LEN}]", "<Type: 'int64_t'>"),
        full_meta_names[1]: (f"[{actual_windows},{MAX_VIDEO_LEN + 58}]", "<Type: 'char'>"),
        full_meta_names[2]: (f"[{actual_windows},{MAX_VIDEO_LEN + 58},126]", "<Type: 'float32'>"),
    }
    saved_inputs = {item["name"]: item for item in model_io(saved)["inputs"]}
    for input_name, (expected_shape, expected_type) in expected_io.items():
        actual = saved_inputs[input_name]
        if actual["shape"].replace(" ", "") != expected_shape:
            raise AssertionError(f"{input_name} static shape mismatch: {actual}")
        if actual["element_type"] != expected_type:
            raise AssertionError(f"{input_name} element type mismatch: {actual}")
    if len(saved_inputs) != 9:
        raise AssertionError(f"{label}: expected exactly 9 external inputs, got {sorted(saved_inputs)}")
    saved_outputs = {item["name"]: item for item in model_io(saved)["outputs"]}
    expected_video_shape = f"[{math.prod(SHAPE)},2560]"
    if saved_outputs["vid_out"]["shape"].replace(" ", "") != expected_video_shape:
        raise AssertionError(f"gathered video shape mismatch: {saved_outputs['vid_out']}")
    if saved_outputs["vid_out"]["element_type"] != "<Type: 'float16'>":
        raise AssertionError(f"gathered video output is not FP16: {saved_outputs['vid_out']}")
    if saved_outputs["txt_out"]["shape"].replace(" ", "") != "[58,2560]":
        raise AssertionError(f"text output shape mismatch: {saved_outputs['txt_out']}")
    if saved_outputs["txt_out"]["element_type"] != "<Type: 'float32'>":
        raise AssertionError(f"text output is not FP32: {saved_outputs['txt_out']}")

    return {
        "label": label,
        "window_method": window_method,
        "contract": {
            "shape_t_h_w": list(SHAPE),
            "source_video_tokens": math.prod(SHAPE),
            "window_size": list(WINDOW),
            "actual_window_count": actual_windows,
            "max_video_len": actual_max_video_len,
            "full_loop_windows": loop_windows,
            "tail_windows": tail_windows,
            "chunk_size": CHUNK_SIZE,
            "loop_iterations": graph["trip_count"],
            "dummy_window_slots": 0,
            "qkv_inputs_fp16": True,
            "gathered_video_output_fp16": True,
            "text_accumulation_fp32": True,
            "text_output_fp32": True,
            "inverse_gather_positions_unique": bool(np.unique(reverse_idx).size == reverse_idx.size),
            "outer_input_count": len(saved_inputs),
            "full_metadata_parameters": list(full_meta_names),
        },
        "saved_graph": {
            "xml": str(output_xml),
            "bin": str(output_xml.with_suffix(".bin")),
            "xml_bytes": output_xml.stat().st_size,
            "bin_bytes": output_xml.with_suffix(".bin").stat().st_size,
            "total_bytes": output_xml.stat().st_size + output_xml.with_suffix(".bin").stat().st_size,
            "xml_sha256": sha256_file(output_xml),
            "bin_sha256": sha256_file(output_xml.with_suffix(".bin")),
            "inputs": model_io(saved)["inputs"],
            "outputs": model_io(saved)["outputs"],
            "outer_ordered_op_counts": graph["outer_ordered_op_counts"],
            "loop_body_ordered_op_counts": graph["body_ordered_op_counts"],
            "tail_attention_ordered_op_counts": graph["tail_ordered_op_counts"],
            "saved_xml_structure": saved_structure,
        },
    }


def main() -> int:
    core = ov.Core()
    variants = [
        _build_variant(core, "even", WINDOW_METHODS["even"], EVEN_XML),
        _build_variant(core, "odd", WINDOW_METHODS["odd"], ODD_XML),
    ]
    receipt = {
        "status": "saved_static_graphs",
        "invocation": INVOCATION,
        "openvino_version": ov.__version__,
        "execution_limits": {
            "affinity_cpus": sorted(os.sched_getaffinity(0)),
            "nice_value": os.getpriority(os.PRIO_PROCESS, 0),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "model_weights_loaded": False,
        "production_graph_cpu_compile_called": False,
        "production_graph_infer_called": False,
        "saved_shared_attention_ir": {
            "path": str(SHARED_IR),
            "xml_bytes": SHARED_IR.stat().st_size,
            "bin_bytes": SHARED_IR.with_suffix(".bin").stat().st_size,
            "xml_sha256": sha256_file(SHARED_IR),
            "bin_sha256": sha256_file(SHARED_IR.with_suffix(".bin")),
        },
        "variants": variants,
        "intermediate_global_envelope_graph": {
            "xml": str(HERE / "production_static_t2_h240_w135_loop_chunk16_fp16.xml"),
            "superseded_by": str(EVEN_XML),
            "reason": "replaced generic 338x537 bounds with exact shape-specific Loop plus real tail contracts",
        },
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
