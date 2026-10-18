#!/usr/bin/env python3
"""CPU-only saved-IR parity for the missing SeedVR2 portrait case.

This evidence-local runner reuses the canonical decomposition helpers and reads
the already-qualified per-block IRs. It evaluates only [T,H,W]=[2,240,135]; it
does not convert, save, or modify model artifacts or canonical receipts.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "portrait_cpu_validation"
PORTRAIT_SHAPE = (2, 240, 135)
THREADS = 4
CHUNK_SIZE = 16
STAGES = {
    "prefix": {
        "stem": "chunked_prefix",
        "inputs": ["vid", "txt", "emb"],
        "outputs": ["vid_q", "vid_k", "vid_v", "txt_q", "txt_k", "txt_v"],
    },
    "chunk": {
        "stem": "chunked_attention",
        "inputs": [
            "vid_q", "vid_k", "vid_v", "txt_q", "txt_k", "txt_v",
            "window_idx", "window_key_mask", "window_rope_freqs",
        ],
        "outputs": ["vid_window_out", "txt_sum", "txt_count"],
    },
    "suffix": {
        "stem": "chunked_suffix",
        "inputs": ["vid", "txt", "emb", "attn_vid", "attn_txt"],
        "outputs": ["vid_out", "txt_out"],
    },
}


def select_case_shapes() -> tuple[tuple[int, int, int], ...]:
    """Return only the missing portrait parity case, with H > W."""

    return (PORTRAIT_SHAPE,)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_shapes(values: Sequence[Any], torch: Any) -> list[list[int]]:
    result = []
    for value in values:
        if not bool(torch.isfinite(value).all().item()):
            raise AssertionError("non-finite saved-IR output")
        result.append([int(item) for item in value.shape])
    return result


def _assert_metric(metric: Mapping[str, Any], max_abs: float, rms_relative: float, label: str) -> None:
    if metric["max_abs_error"] > max_abs or metric["max_rms_relative_error"] > rms_relative:
        raise AssertionError(f"{label} threshold failure: {dict(metric)}")


def _model_paths(block_index: int, stage: str) -> tuple[Path, Path]:
    xml_path = HERE / f"block{block_index}_{STAGES[stage]['stem']}_fp16.xml"
    return xml_path, xml_path.with_suffix(".bin")


def _compile_ir(
    exporter: Any,
    ov: Any,
    core: Any,
    block_index: int,
    stage: str,
) -> tuple[Any, Any, Path, Path, float, list[str]]:
    xml_path, bin_path = _model_paths(block_index, stage)
    if not xml_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(f"missing existing {stage} IR for block {block_index}")
    model = core.read_model(str(xml_path))
    contract = STAGES[stage]
    exporter._named_ports(model, contract["inputs"])
    exporter._name_outputs(model, contract["outputs"])
    exporter._reshape_model(ov, model, contract["inputs"])
    compile_start = time.perf_counter()
    compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    compile_seconds = time.perf_counter() - compile_start
    execution_devices = [str(value) for value in compiled.get_property("EXECUTION_DEVICES")]
    if execution_devices != ["CPU"]:
        raise AssertionError(f"saved IR compiled to unexpected device(s): {execution_devices}")
    return model, compiled, xml_path, bin_path, compile_seconds, execution_devices


def _load_block(
    exporter: Any,
    base: Any,
    trusted_helper: Any,
    block_index: int,
    torch: Any,
) -> tuple[Any, bool, bool, str, Mapping[str, Any]]:
    if block_index == 0:
        state, shard_meta = trusted_helper._load_block_weights(exporter.WEIGHTS)
        source_block = trusted_helper._build_block(exporter.SOURCE, state, torch)
        return source_block, False, False, base.WINDOW_METHOD, shard_meta

    expected_shared = block_index >= 10
    expected_last = block_index == 31
    if bool(base.SHARED_WEIGHTS) != expected_shared:
        raise AssertionError("block shared-weight family differs from the canonical index mapping")
    if bool(base.IS_LAST_LAYER) != expected_last:
        raise AssertionError("block last-layer contract differs from the canonical index mapping")
    state, shard_meta = base.load_block_weights(exporter.WEIGHTS)
    source_block = base.build_source_block(exporter.SOURCE, state, torch)
    return source_block, expected_shared, expected_last, base.WINDOW_METHOD, shard_meta


def run_block(block_index: int) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = OUTPUT_DIR / f"receipt.portrait_chunked_decomposition_block{block_index}.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite {receipt_path.name}")

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(THREADS)
    print(f"PORTRAIT_CPU_RUN pid={os.getpid()} block={block_index}", flush=True)
    start = time.perf_counter()
    receipt: dict[str, Any] = {
        "status": "running",
        "task": "chunked-attention-portrait-cpu-validation-openvino-20260918",
        "block_index": block_index,
        "shape": list(PORTRAIT_SHAPE),
        "orientation_contract": "T=2 and H>W; shape is [T,H,W]",
        "execution": {"device": "CPU", "gpu_inspection": False, "gpu": "not inspected/used", "thread_cap": THREADS},
        "artifacts_reused": True,
        "artifact_writes": False,
        "thresholds": {
            "eager_max_abs_error_le": 1.0e-3,
            "eager_rms_relative_error_le": 1.0e-4,
            "saved_ir_max_abs_error_le": 5.0e-3,
            "saved_ir_rms_relative_error_le": 1.0e-4,
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        import numpy as np
        import torch
        import openvino as ov

        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        if hasattr(torch, "xpu"):
            torch.xpu.is_available = lambda: False
            if hasattr(torch.xpu, "device_count"):
                torch.xpu.device_count = lambda: 0

        exporter = _load_module(HERE / "chunked_attention_decomposition_export.py", "canonical_chunked_exporter")
        base_path = HERE / f"dynamic_window_reverse_gather_block{block_index}_export.py"
        base = exporter._load_module(base_path, f"portrait_reverse_gather_base_{block_index}")
        trusted_helper = exporter._load_module(exporter.TRUSTED_HELPER, f"portrait_trusted_block0_{block_index}")
        prior_path = HERE / f"receipt.chunked_decomposition_block{block_index}.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("status") != "pass":
            raise AssertionError("canonical per-block receipt is not passing")
        trace_graph = prior.get("trace_graph", {})
        forbidden_absent = all(
            not flags.get(key, False)
            for flags in trace_graph.values()
            for key in ("contains_index_add", "contains_scatter", "contains_nonzero")
        )
        if len(trace_graph) != 3 or not forbidden_absent:
            raise AssertionError("canonical prefix/chunk/suffix forbidden-op trace is absent or failing")

        source_block, shared_weights, is_last_layer, window_method, shard_meta = _load_block(
            exporter, base, trusted_helper, block_index, torch
        )
        if (block_index % 2 == 0) != (window_method == "720pwin_by_size_bysize"):
            raise AssertionError("window shift does not match the canonical block-index alternation")
        trusted = exporter._trusted_wrapper(source_block, torch)
        prefix, chunk, suffix = exporter.build_decomposition_modules(torch, source_block)

        summary = exporter.window_metadata(base, PORTRAIT_SHAPE)
        contract = base.make_window_contract(
            PORTRAIT_SHAPE,
            max_video_len=summary["max_video_len"],
            txt_len=exporter.TXT_TOKENS,
            rope=source_block.attn.rope,
            torch=torch,
        )
        coverage = exporter.validate_window_coverage(contract, torch)
        contract["shape"] = PORTRAIT_SHAPE
        contract["max_video_len"] = summary["max_video_len"]
        chunk_count = int(summary["window_count"])
        if chunk_count > exporter.PRODUCTION_MAX_WINDOWS or summary["max_video_len"] > exporter.PRODUCTION_MAX_VIDEO_LEN:
            raise AssertionError("portrait shape exceeds the recorded production window caps")
        if PORTRAIT_SHAPE not in exporter.PRODUCTION_SHAPES:
            raise AssertionError("portrait shape is missing from the canonical production-shape contract")

        vid, txt, emb = exporter.build_inputs(contract, 2026091820 + block_index, torch)
        with torch.inference_mode():
            trusted_start = time.perf_counter()
            trusted_out = trusted(vid, txt, contract["vid_shape"], contract["txt_shape"], emb)
            trusted_seconds = time.perf_counter() - trusted_start
            exporter._assert_finite(trusted_out, torch, "trusted portrait reference")

            prefix_start = time.perf_counter()
            prefix_eager = prefix(vid, txt, emb)
            prefix_eager_seconds = time.perf_counter() - prefix_start
            eager_vid, eager_txt, eager_chunk_timing = exporter.collect_eager_attention(
                chunk, prefix_eager, contract, CHUNK_SIZE, torch
            )
            eager_out = suffix(vid, txt, emb, eager_vid, eager_txt)
            exporter._assert_finite(eager_out, torch, "eager portrait decomposition")
            eager_metric = exporter.metrics(trusted_out, eager_out, np)
            _assert_metric(eager_metric, 1.0e-3, 1.0e-4, "eager portrait decomposition")
            eager_shapes = _finite_shapes(eager_out, torch)
            del prefix_eager, eager_vid, eager_txt, eager_out
            gc.collect()

            core = ov.Core()
            stages: dict[str, Any] = {}
            prefix_model, prefix_compiled, prefix_xml, prefix_bin, prefix_compile_seconds, prefix_devices = _compile_ir(
                exporter, ov, core, block_index, "prefix"
            )
            prefix_start = time.perf_counter()
            prefix_result = prefix_compiled(
                exporter._feed(prefix_compiled, STAGES["prefix"]["inputs"], (vid, txt, emb))
            )
            prefix_saved = exporter._output_tuple(prefix_result, prefix_compiled, torch)
            prefix_saved_shapes = _finite_shapes(prefix_saved, torch)
            prefix_seconds = time.perf_counter() - prefix_start
            stages["prefix"] = {
                "status": "pass",
                "xml": str(prefix_xml),
                "bin": str(prefix_bin),
                "xml_bytes": prefix_xml.stat().st_size,
                "bin_bytes": prefix_bin.stat().st_size,
                "compile_seconds": prefix_compile_seconds,
                "execution_devices": prefix_devices,
                "infer_seconds": prefix_seconds,
                "output_shapes": prefix_saved_shapes,
            }
            del prefix_result, prefix_compiled, prefix_model
            gc.collect()

            chunk_model, chunk_compiled, chunk_xml, chunk_bin, chunk_compile_seconds, chunk_devices = _compile_ir(
                exporter, ov, core, block_index, "chunk"
            )
            source_tokens = int(contract["vid_shape"].prod().item())
            max_len = int(contract["max_video_len"])
            vid_acc = torch.zeros((source_tokens, exporter.INNER_DIM), dtype=torch.float32)
            txt_sum = torch.zeros((exporter.TXT_TOKENS, exporter.INNER_DIM), dtype=torch.float32)
            text_count = 0.0
            chunk_events: list[dict[str, Any]] = []
            global_reverse = contract["window_reverse_idx"]
            for start_index in range(0, chunk_count, CHUNK_SIZE):
                end_index = min(chunk_count, start_index + CHUNK_SIZE)
                part = exporter.split_contract(contract, start_index, end_index)
                values = tuple(prefix_saved) + (
                    part["window_idx"], part["window_key_mask"], part["window_rope_freqs"]
                )
                infer_start = time.perf_counter()
                result = chunk_compiled(
                    exporter._feed(chunk_compiled, STAGES["chunk"]["inputs"], values)
                )
                elapsed = time.perf_counter() - infer_start
                video_chunk, text_chunk_sum, text_chunk_count = exporter._output_tuple(
                    result, chunk_compiled, torch
                )
                _finite_shapes((video_chunk, text_chunk_sum, text_chunk_count), torch)
                local_reverse = global_reverse - start_index * max_len
                source_idx = part["window_idx"][:, :max_len][part["window_key_mask"][:, :max_len]]
                selected_positions = local_reverse.index_select(0, source_idx)
                if int(torch.unique(source_idx).numel()) != int(source_idx.numel()):
                    raise AssertionError("saved chunk inverse-gather selected a duplicate source token")
                vid_acc.index_copy_(
                    0,
                    source_idx,
                    video_chunk.reshape(-1, exporter.INNER_DIM).index_select(0, selected_positions),
                )
                txt_sum = txt_sum + text_chunk_sum
                count = float(text_chunk_count.reshape(-1)[0].item())
                if not math.isfinite(count) or count <= 0:
                    raise AssertionError("saved chunk text count is not finite and positive")
                text_count += count
                chunk_events.append({"start": start_index, "end": end_index, "seconds": elapsed})
                del result, video_chunk, text_chunk_sum, text_chunk_count, part, values
            if abs(text_count - chunk_count) > 1.0e-6:
                raise AssertionError(f"saved chunk text count mismatch {text_count} != {chunk_count}")
            attn_txt = txt_sum / text_count
            _finite_shapes((vid_acc, attn_txt), torch)
            stages["chunk"] = {
                "status": "pass",
                "xml": str(chunk_xml),
                "bin": str(chunk_bin),
                "xml_bytes": chunk_xml.stat().st_size,
                "bin_bytes": chunk_bin.stat().st_size,
                "compile_seconds": chunk_compile_seconds,
                "execution_devices": chunk_devices,
                "chunk_size": CHUNK_SIZE,
                "window_count": chunk_count,
                "chunk_call_count": len(chunk_events),
                "text_count": text_count,
                "chunk_events": chunk_events,
            }
            del chunk_compiled, chunk_model, prefix_saved, txt_sum
            gc.collect()

            suffix_model, suffix_compiled, suffix_xml, suffix_bin, suffix_compile_seconds, suffix_devices = _compile_ir(
                exporter, ov, core, block_index, "suffix"
            )
            suffix_start = time.perf_counter()
            suffix_result = suffix_compiled(
                exporter._feed(
                    suffix_compiled,
                    STAGES["suffix"]["inputs"],
                    (vid, txt, emb, vid_acc, attn_txt),
                )
            )
            saved_out = exporter._output_tuple(suffix_result, suffix_compiled, torch)
            saved_shapes = _finite_shapes(saved_out, torch)
            suffix_seconds = time.perf_counter() - suffix_start
            saved_metric = exporter.metrics(trusted_out, saved_out, np)
            _assert_metric(saved_metric, 5.0e-3, 1.0e-4, "saved portrait IR")
            stages["suffix"] = {
                "status": "pass",
                "xml": str(suffix_xml),
                "bin": str(suffix_bin),
                "xml_bytes": suffix_xml.stat().st_size,
                "bin_bytes": suffix_bin.stat().st_size,
                "compile_seconds": suffix_compile_seconds,
                "execution_devices": suffix_devices,
                "infer_seconds": suffix_seconds,
                "output_shapes": saved_shapes,
            }
            del suffix_result, suffix_compiled, suffix_model, saved_out
            gc.collect()

        receipt.update(
            {
                "status": "pass",
                "artifact_source": "existing dynamic-window-length-openvino-20260918 IRs; read with OpenVINO Core.read_model",
                "canonical_receipt": str(prior_path),
                "canonical_trace_graph": trace_graph,
                "forbidden_ops_absent": forbidden_absent,
                "source_config_method_matches_index_alternation": True,
                "block_family": "unshared" if not shared_weights else "shared",
                "is_last_layer": is_last_layer,
                "window_method": window_method,
                "window_contract": {
                    "shape": list(PORTRAIT_SHAPE),
                    "window_count": chunk_count,
                    "max_video_len": int(summary["max_video_len"]),
                    "production_max_windows": exporter.PRODUCTION_MAX_WINDOWS,
                    "production_max_video_len": exporter.PRODUCTION_MAX_VIDEO_LEN,
                    "coverage": coverage,
                    "window_lengths": summary["window_lengths"],
                },
                "weights": dict(shard_meta),
                "stages": stages,
                "parity": {
                    "trusted_seconds": trusted_seconds,
                    "eager_prefix_seconds": prefix_eager_seconds,
                    "eager_vs_trusted": eager_metric,
                    "eager_output_shapes": eager_shapes,
                    "saved_ir_vs_trusted": saved_metric,
                    "saved_ir_output_shapes": saved_shapes,
                    "text_count_exact": abs(text_count - chunk_count) <= 1.0e-6,
                },
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": time.perf_counter() - start,
                **exporter.peak_rss(),
            }
        )
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "block_index": block_index,
                    "shape": list(PORTRAIT_SHAPE),
                    "elapsed_seconds": receipt["elapsed_seconds"],
                    "peak_rss_bytes": receipt["peak_rss_bytes"],
                    "saved_ir_max_abs_error": saved_metric["max_abs_error"],
                    "saved_ir_max_rms_relative_error": saved_metric["max_rms_relative_error"],
                    "receipt": str(receipt_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return receipt
    except Exception as exc:
        receipt.update(
            {
                "status": "failure",
                "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        try:
            import resource

            kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            receipt.update({"peak_rss_kib": kib, "peak_rss_bytes": kib * 1024})
        except Exception:
            pass
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "failure", "block_index": block_index, "receipt": str(receipt_path)}, sort_keys=True), flush=True)
        raise


def write_aggregate() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "manifest.portrait_chunked_decomposition_cpu_all_blocks.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    canonical_manifest_path = HERE / "manifest.chunked_decomposition_all_blocks.json"
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    if canonical_manifest.get("blocks_expected") != 32 or canonical_manifest.get("blocks_present") != 32:
        raise AssertionError("canonical aggregate does not contain exactly 32 blocks")

    summaries = []
    for block_index in range(32):
        receipt_path = OUTPUT_DIR / f"receipt.portrait_chunked_decomposition_block{block_index}.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"missing portrait receipt for block {block_index}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "pass" or receipt.get("shape") != list(PORTRAIT_SHAPE):
            raise AssertionError(f"portrait receipt did not pass exact shape for block {block_index}")
        if not receipt.get("forbidden_ops_absent"):
            raise AssertionError(f"forbidden-op trace check failed for block {block_index}")
        if not receipt.get("parity", {}).get("text_count_exact"):
            raise AssertionError(f"text-count contract failed for block {block_index}")
        if set(receipt.get("stages", {})) != set(STAGES):
            raise AssertionError(f"prefix/chunk/suffix coverage is incomplete for block {block_index}")
        if any(receipt["stages"][stage].get("status") != "pass" for stage in STAGES):
            raise AssertionError(f"one or more saved-IR stages failed for block {block_index}")
        summaries.append(
            {
                "block_index": block_index,
                "status": receipt["status"],
                "window_method": receipt["window_method"],
                "block_family": receipt["block_family"],
                "window_count": receipt["window_contract"]["window_count"],
                "peak_rss_bytes": receipt.get("peak_rss_bytes"),
                "elapsed_seconds": receipt.get("elapsed_seconds"),
                "forbidden_ops_absent": receipt["forbidden_ops_absent"],
                "eager_max_abs_error": receipt["parity"]["eager_vs_trusted"]["max_abs_error"],
                "eager_max_rms_relative_error": receipt["parity"]["eager_vs_trusted"]["max_rms_relative_error"],
                "saved_ir_max_abs_error": receipt["parity"]["saved_ir_vs_trusted"]["max_abs_error"],
                "saved_ir_max_rms_relative_error": receipt["parity"]["saved_ir_vs_trusted"]["max_rms_relative_error"],
            }
        )
    block16 = summaries[16]
    if block16["window_method"] != "720pwin_by_size_bysize":
        raise AssertionError("block16 must remain unshifted")
    if any((item["block_index"] % 2 == 0) != (item["window_method"] == "720pwin_by_size_bysize") for item in summaries):
        raise AssertionError("one or more blocks violate the canonical unshifted/shifted alternation")

    aggregate = {
        "status": "portrait_case_passed_for_all_32_blocks",
        "task": "chunked-attention-portrait-cpu-validation-openvino-20260918",
        "shape": list(PORTRAIT_SHAPE),
        "orientation_contract": "T=2 and H>W; shape is [T,H,W]",
        "blocks_expected": 32,
        "blocks_present": len(summaries),
        "all_artifact_receipts_pass": len(summaries) == 32,
        "all_measured_matrix_parity_pass": all(
            item["eager_max_abs_error"] <= 1.0e-3
            and item["eager_max_rms_relative_error"] <= 1.0e-4
            and item["saved_ir_max_abs_error"] <= 5.0e-3
            and item["saved_ir_max_rms_relative_error"] <= 1.0e-4
            for item in summaries
        ),
        "all_traced_forbidden_ops_absent": all(item["forbidden_ops_absent"] for item in summaries),
        "all_prefix_chunk_suffix_executed": True,
        "block16_unshifted": block16["window_method"] == "720pwin_by_size_bysize",
        "cpu_only": True,
        "threads_per_process": THREADS,
        "artifact_reuse": {"performed": False, "policy": canonical_manifest.get("artifact_reuse", {}).get("policy")},
        "canonical_artifact_bytes_unchanged": canonical_manifest.get("total_artifact_bytes"),
        "canonical_receipts_preserved": True,
        "receipts": summaries,
    }
    if not aggregate["all_measured_matrix_parity_pass"] or not aggregate["all_traced_forbidden_ops_absent"]:
        raise AssertionError("aggregate contains a portrait parity threshold or forbidden-op failure")
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": aggregate["status"], "aggregate": str(path), "blocks": len(summaries)}, sort_keys=True), flush=True)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-index", type=int, choices=tuple(range(32)))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        write_aggregate()
        return 0
    if args.block_index is None:
        parser.error("--block-index or --aggregate is required")
    run_block(args.block_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
