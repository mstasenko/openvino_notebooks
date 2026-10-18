#!/usr/bin/env python3
"""CPU parity proof and uncompiled production IRs for a fused SeedVR2 block 0."""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import openvino as ov


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
LOOP_DIR = EVIDENCE / "loop-chunk-attention-openvino-20260919"
DYNAMIC_DIR = EVIDENCE / "dynamic-window-length-openvino-20260918"
sys.path.insert(0, str(LOOP_DIR))

from loop_chunk_attention import (  # noqa: E402
    QKV_NAMES,
    TXT_TOKENS,
    WINDOW_META_NAMES,
    build_loop_tail_model,
    model_io,
    sha256_file,
    xml_loop_structure,
)
from build_production_loop import (  # noqa: E402
    MAX_VIDEO_LEN,
    SHAPE,
    WINDOW_METHODS,
    _window_geometry,
)


PREFIX_XML = DYNAMIC_DIR / "block0_chunked_prefix_fp16.xml"
SUFFIX_XML = DYNAMIC_DIR / "block0_chunked_suffix_fp16.xml"
CHUNK_SIZE = 16
SMALL_TOKENS = 8
SMALL_WINDOWS = 5
SMALL_MAX_VIDEO_LEN = 2
SMALL_CHUNK_SIZE = 2
SEED = 20260919
MAX_ABS_THRESHOLD = 5.0e-3
RMS_REL_THRESHOLD = 1.0e-4
RECEIPT = HERE / "receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def parameters(model: ov.Model) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for parameter in model.get_parameters():
        name = parameter.output(0).get_any_name()
        if name in result:
            raise AssertionError(f"duplicate Parameter name {name}")
        result[name] = parameter
    return result


def result_values(model: ov.Model) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for port, node in zip(model.outputs, model.get_results()):
        name = port.get_any_name()
        if name in result:
            raise AssertionError(f"duplicate Result name {name}")
        result[name] = node.input_value(0)
    return result


def replace_parameter(parameter: Any, value: Any) -> None:
    targets = list(parameter.output(0).get_target_inputs())
    if not targets:
        raise AssertionError(f"Parameter {parameter.output(0).get_any_name()} has no consumers")
    for target in targets:
        target.replace_source_output(value)


def load_split_models(core: ov.Core, source_tokens: int) -> tuple[ov.Model, ov.Model]:
    prefix = core.read_model(PREFIX_XML)
    prefix.reshape(
        {
            "vid": [source_tokens, 2560],
            "txt": [TXT_TOKENS, 2560],
            "emb": [1, 15360],
        }
    )
    suffix = core.read_model(SUFFIX_XML)
    suffix.reshape(
        {
            "vid": [source_tokens, 2560],
            "txt": [TXT_TOKENS, 2560],
            "emb": [1, 15360],
            "attn_vid": [source_tokens, 2560],
            "attn_txt": [TXT_TOKENS, 2560],
        }
    )
    return prefix, suffix


def compose_block(
    core: ov.Core,
    *,
    source_tokens: int,
    total_windows: int,
    max_video_len: int,
    chunk_size: int,
    reverse_idx: np.ndarray,
    name: str,
) -> ov.Model:
    prefix, suffix = load_split_models(core, source_tokens)
    attention = build_loop_tail_model(
        core,
        source_tokens=source_tokens,
        total_windows=total_windows,
        max_video_len=max_video_len,
        chunk_size=chunk_size,
        reverse_idx=reverse_idx,
        qkv_type=ov.Type.f32,
        video_output_type=ov.Type.f32,
        include_text_stats=False,
        name=f"{name}_attention",
    )["model"]

    prefix_inputs = parameters(prefix)
    prefix_outputs = result_values(prefix)
    attention_inputs = parameters(attention)
    attention_outputs = result_values(attention)
    suffix_inputs = parameters(suffix)
    suffix_outputs = result_values(suffix)

    for qkv_name in QKV_NAMES:
        replace_parameter(attention_inputs[qkv_name], prefix_outputs[qkv_name])
    for state_name in ("vid", "txt", "emb"):
        replace_parameter(suffix_inputs[state_name], prefix_inputs[state_name].output(0))
    replace_parameter(suffix_inputs["attn_vid"], attention_outputs["vid_out"])
    replace_parameter(suffix_inputs["attn_txt"], attention_outputs["txt_out"])

    external = [prefix_inputs[name] for name in ("vid", "txt", "emb")]
    external.extend(attention_inputs[name] for name in WINDOW_META_NAMES)
    fused = ov.Model(
        [suffix_outputs["vid_out"], suffix_outputs["txt_out"]],
        external,
        name,
    )
    for port, output_name in zip(fused.outputs, ("vid_out", "txt_out")):
        port.get_tensor().set_names({output_name})
    expected_inputs = {"vid", "txt", "emb", *WINDOW_META_NAMES}
    actual_inputs = {port.get_any_name() for port in fused.inputs}
    if actual_inputs != expected_inputs:
        raise AssertionError(f"fused input contract differs: {sorted(actual_inputs)}")
    if len([node for node in fused.get_ordered_ops() if node.get_type_name() == "Loop"]) != 1:
        raise AssertionError("fused block must contain exactly one Loop")
    return fused


def small_metadata() -> dict[str, np.ndarray]:
    lengths = (2, 2, 2, 1, 1)
    idx = np.zeros((SMALL_WINDOWS, SMALL_MAX_VIDEO_LEN), dtype=np.int64)
    mask = np.zeros((SMALL_WINDOWS, SMALL_MAX_VIDEO_LEN + TXT_TOKENS), dtype=np.bool_)
    rope = np.zeros((SMALL_WINDOWS, SMALL_MAX_VIDEO_LEN + TXT_TOKENS, 126), dtype=np.float32)
    reverse = np.full((SMALL_TOKENS,), -1, dtype=np.int64)
    offset = 0
    phase = np.arange(126, dtype=np.float32) * np.float32(1.0e-6)
    for window, length in enumerate(lengths):
        source = np.arange(offset, offset + length, dtype=np.int64)
        idx[window, :length] = source
        mask[window, :length] = True
        mask[window, SMALL_MAX_VIDEO_LEN:] = True
        reverse[source] = window * SMALL_MAX_VIDEO_LEN + np.arange(length, dtype=np.int64)
        rope[window, :length] = np.float32(window + 1) * np.float32(1.0e-4) + phase
        rope[window, SMALL_MAX_VIDEO_LEN:] = np.float32(window + 1) * np.float32(2.0e-4) + phase
        offset += length
    if offset != SMALL_TOKENS or np.any(reverse < 0):
        raise AssertionError("small metadata does not cover all tokens")
    return {
        "window_idx": idx,
        "window_key_mask": mask,
        "window_rope_freqs": rope,
        "reverse_idx": reverse,
    }


def run(compiled: Any, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    outputs = compiled(values)
    return {
        port.get_any_name(): np.array(outputs[port], copy=True)
        for port in compiled.outputs
    }


def metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    rms_difference = math.sqrt(float(np.mean(np.square(difference))))
    rms_reference = math.sqrt(float(np.mean(np.square(reference.astype(np.float64)))))
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "rms_relative_error": rms_difference / max(rms_reference, 1.0e-12),
    }


def small_cpu_proof(core: ov.Core) -> dict[str, Any]:
    metadata = small_metadata()
    generator = np.random.default_rng(SEED)
    video = generator.standard_normal((SMALL_TOKENS, 2560), dtype=np.float32) * np.float32(0.02)
    text = generator.standard_normal((TXT_TOKENS, 2560), dtype=np.float32) * np.float32(0.02)
    embedding = generator.standard_normal((1, 15360), dtype=np.float32) * np.float32(0.02)
    config = {"INFERENCE_NUM_THREADS": 4, "NUM_STREAMS": 1}

    prefix, suffix = load_split_models(core, SMALL_TOKENS)
    attention = build_loop_tail_model(
        core,
        source_tokens=SMALL_TOKENS,
        total_windows=SMALL_WINDOWS,
        max_video_len=SMALL_MAX_VIDEO_LEN,
        chunk_size=SMALL_CHUNK_SIZE,
        reverse_idx=metadata["reverse_idx"],
        qkv_type=ov.Type.f32,
        video_output_type=ov.Type.f32,
        include_text_stats=False,
        name="small_split_attention",
    )["model"]
    compiled_prefix = core.compile_model(prefix, "CPU", config)
    compiled_attention = core.compile_model(attention, "CPU", config)
    compiled_suffix = core.compile_model(suffix, "CPU", config)
    qkv = run(compiled_prefix, {"vid": video, "txt": text, "emb": embedding})
    attention_feed = {name: qkv[name] for name in QKV_NAMES}
    attention_feed.update({name: metadata[name] for name in WINDOW_META_NAMES})
    attended = run(compiled_attention, attention_feed)
    reference = run(
        compiled_suffix,
        {
            "vid": video,
            "txt": text,
            "emb": embedding,
            "attn_vid": attended["vid_out"],
            "attn_txt": attended["txt_out"],
        },
    )
    del compiled_prefix, compiled_attention, compiled_suffix, prefix, suffix, attention, qkv, attended
    gc.collect()

    fused = compose_block(
        core,
        source_tokens=SMALL_TOKENS,
        total_windows=SMALL_WINDOWS,
        max_video_len=SMALL_MAX_VIDEO_LEN,
        chunk_size=SMALL_CHUNK_SIZE,
        reverse_idx=metadata["reverse_idx"],
        name="seedvr2_block0_small_fused_loop_tail",
    )
    compiled_fused = core.compile_model(fused, "CPU", config)
    candidate = run(
        compiled_fused,
        {
            "vid": video,
            "txt": text,
            "emb": embedding,
            **{name: metadata[name] for name in WINDOW_META_NAMES},
        },
    )
    metrics = {name: metric(reference[name], candidate[name]) for name in ("vid_out", "txt_out")}
    if any(not np.isfinite(candidate[name]).all() for name in candidate):
        raise AssertionError("fused output contains a non-finite value")
    if any(
        value["max_abs_error"] > MAX_ABS_THRESHOLD
        or value["rms_relative_error"] > RMS_REL_THRESHOLD
        for value in metrics.values()
    ):
        raise AssertionError(f"fused block parity failed: {metrics}")
    return {
        "status": "pass",
        "metrics": metrics,
        "input_contract": model_io(fused)["inputs"],
        "output_contract": model_io(fused)["outputs"],
        "ordered_op_counts": dict(
            sorted(Counter(node.get_type_name() for node in fused.get_ordered_ops()).items())
        ),
    }


def save_production_variant(core: ov.Core, label: str) -> dict[str, Any]:
    window_count, max_video_len, reverse_idx = _window_geometry(WINDOW_METHODS[label])
    if max_video_len != MAX_VIDEO_LEN:
        raise AssertionError(f"{label} max window length differs")
    model = compose_block(
        core,
        source_tokens=math.prod(SHAPE),
        total_windows=window_count,
        max_video_len=max_video_len,
        chunk_size=CHUNK_SIZE,
        reverse_idx=reverse_idx,
        name=f"seedvr2_block0_{label}_fused_loop_tail",
    )
    xml = HERE / f"block0_{label}_fused_loop_tail_fp16.xml"
    ov.save_model(model, str(xml), compress_to_fp16=True)
    saved = core.read_model(xml)
    structure = xml_loop_structure(xml)
    inputs = model_io(saved)["inputs"]
    if len(inputs) != 6 or structure["loop_count"] != 1 or structure["sdpa_count"] != 2:
        raise AssertionError(f"{label} production fused structure differs")
    if structure["sdpa_per_loop_body"] != [1]:
        raise AssertionError(f"{label} Loop body must contain one SDPA")
    if {item["name"] for item in inputs} != {"vid", "txt", "emb", *WINDOW_META_NAMES}:
        raise AssertionError(f"{label} production input names differ")
    bin_path = xml.with_suffix(".bin")
    return {
        "label": label,
        "window_count": window_count,
        "max_video_len": max_video_len,
        "loop_iterations": window_count // CHUNK_SIZE,
        "tail_windows": window_count % CHUNK_SIZE,
        "xml": str(xml),
        "bin": str(bin_path),
        "xml_bytes": xml.stat().st_size,
        "bin_bytes": bin_path.stat().st_size,
        "xml_sha256": sha256_file(xml),
        "bin_sha256": sha256_file(bin_path),
        "inputs": inputs,
        "outputs": model_io(saved)["outputs"],
        "structure": structure,
        "production_compiled": False,
        "production_inferred": False,
    }


def main() -> int:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "running",
        "openvino_version": ov.__version__,
        "device": "CPU",
        "gpu_used": False,
        "thread_cap": 4,
        "prefix_bin_bytes": PREFIX_XML.with_suffix(".bin").stat().st_size,
        "suffix_bin_bytes": SUFFIX_XML.with_suffix(".bin").stat().st_size,
    }
    try:
        core = ov.Core()
        result["small_cpu_proof"] = small_cpu_proof(core)
        result["production_variants"] = [
            save_production_variant(core, "even"),
            save_production_variant(core, "odd"),
        ]
        split_bytes = result["prefix_bin_bytes"] + result["suffix_bin_bytes"]
        result["split_bin_total_bytes"] = split_bytes
        result["production_bin_deltas"] = {
            item["label"]: item["bin_bytes"] - split_bytes
            for item in result["production_variants"]
        }
        result["status"] = "pass"
    except Exception as error:
        result["status"] = "fail"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        RECEIPT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
