#!/usr/bin/env python3
"""Deterministic CPU parity test for the shared-attention Loop plus real tail."""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import traceback
import unittest

import numpy as np
import openvino as ov

from loop_chunk_attention import (
    HEADS,
    HEAD_DIM,
    INNER_DIM,
    SHARED_IR,
    TXT_TOKENS,
    WINDOW_META_NAMES,
    QKV_NAMES,
    SAVED_IR_MAX_ABS_THRESHOLD,
    SAVED_IR_RMS_REL_THRESHOLD,
    _load_static_attention,
    build_loop_tail_model,
    model_io,
    validate_saved_loop_ir,
)


HERE = Path(__file__).resolve().parent
PROOF_XML = HERE / "small_exact_loop_tail_proof.xml"
RECEIPT_PATH = HERE / "cpu_loop_proof_receipt_packed-output-f32.json"
PRODUCTION_RECEIPT = HERE / "production_static_graph_receipt.json"
CHUNK_SIZE = 2
SOURCE_TOKENS = 8
WINDOW_COUNT = 5
PADDED_VIDEO_LEN = 2
LOOP_WINDOWS = 4
TAIL_WINDOWS = 1
SEED = 20260919
CPU_THREADS = 4
INVOCATION = (
    "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "
    "PYTHONPATH=<external-runtime> "
    "nice -n 15 taskset -c 0-3 "
    "<external-python> "
    "test_loop_chunk_attention.py"
)


def _contract() -> dict[str, np.ndarray | int | list[int]]:
    lengths = (2, 2, 2, 1, 1)
    idx = np.zeros((WINDOW_COUNT, PADDED_VIDEO_LEN), dtype=np.int64)
    key_mask = np.zeros((WINDOW_COUNT, PADDED_VIDEO_LEN + TXT_TOKENS), dtype=np.bool_)
    rope = np.zeros(
        (WINDOW_COUNT, PADDED_VIDEO_LEN + TXT_TOKENS, 126), dtype=np.float32
    )
    reverse = np.full((SOURCE_TOKENS,), -1, dtype=np.int64)
    offset = 0
    phase = np.arange(126, dtype=np.float32) * np.float32(1.0e-6)
    for window_number, length in enumerate(lengths):
        source_ids = np.arange(offset, offset + length, dtype=np.int64)
        idx[window_number, :length] = source_ids
        key_mask[window_number, :length] = True
        key_mask[window_number, PADDED_VIDEO_LEN:] = True
        reverse[source_ids] = window_number * PADDED_VIDEO_LEN + np.arange(
            length, dtype=np.int64
        )
        rope[window_number, :length, :] = np.float32(window_number + 1) * np.float32(1.0e-4) + phase
        rope[
            window_number,
            PADDED_VIDEO_LEN : PADDED_VIDEO_LEN + TXT_TOKENS,
            :,
        ] = np.float32(window_number + 1) * np.float32(2.0e-4) + phase
        offset += length
    if offset != SOURCE_TOKENS or np.any(reverse < 0):
        raise AssertionError("small metadata contract does not cover all source tokens")
    if np.unique(reverse).size != SOURCE_TOKENS:
        raise AssertionError("small metadata reverse gather is not one-to-one")
    return {
        "window_idx": idx,
        "window_key_mask": key_mask,
        "window_rope_freqs": rope,
        "window_reverse_idx": reverse,
        "window_lengths": list(lengths),
    }


def _inputs() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(SEED)
    values: dict[str, np.ndarray] = {}
    for name in QKV_NAMES:
        shape = (
            (SOURCE_TOKENS, HEADS, HEAD_DIM)
            if name.startswith("vid_")
            else (TXT_TOKENS, HEADS, HEAD_DIM)
        )
        values[name] = generator.standard_normal(shape, dtype=np.float32) * np.float32(0.02)
    return values


def _compiled_outputs(compiled: Any, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = compiled(values)
    return {
        port.get_any_name(): np.array(result[port], copy=True)
        for port in compiled.outputs
    }


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    rms_diff = math.sqrt(float(np.mean(np.square(diff))))
    rms_ref = math.sqrt(float(np.mean(np.square(reference.astype(np.float64)))))
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "rms_relative_error": rms_diff / max(rms_ref, 1.0e-12),
    }


def _structure_counts(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.get_ordered_ops():
        name = node.get_type_name()
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _cpu_proof() -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "status": "running",
        "openvino_version": ov.__version__,
        "device": "CPU",
        "invocation": INVOCATION,
        "cpu_threads_requested": CPU_THREADS,
        "model_weights_loaded": False,
        "source": {
            "shared_attention_ir": str(SHARED_IR),
            "seed": SEED,
            "source_video_tokens": SOURCE_TOKENS,
            "window_count": WINDOW_COUNT,
            "window_lengths": [2, 2, 2, 1, 1],
            "max_video_len": PADDED_VIDEO_LEN,
            "chunk_size": CHUNK_SIZE,
            "full_loop_windows": LOOP_WINDOWS,
            "loop_iterations": LOOP_WINDOWS // CHUNK_SIZE,
            "tail_windows": TAIL_WINDOWS,
            "dummy_windows": 0,
        },
        "tolerances": {
            "saved_ir_max_abs_error": SAVED_IR_MAX_ABS_THRESHOLD,
            "saved_ir_rms_relative_error": SAVED_IR_RMS_REL_THRESHOLD,
        },
        "supersedes_receipt": str(HERE / "cpu_loop_proof_receipt.json"),
    }
    graph: dict[str, Any] | None = None
    try:
        metadata = _contract()
        qkv = _inputs()
        feed: dict[str, np.ndarray] = dict(qkv)
        for meta_name in WINDOW_META_NAMES:
            feed[meta_name] = np.asarray(metadata[meta_name])

        core = ov.Core()
        graph = build_loop_tail_model(
            core,
            source_tokens=SOURCE_TOKENS,
            total_windows=WINDOW_COUNT,
            max_video_len=PADDED_VIDEO_LEN,
            chunk_size=CHUNK_SIZE,
            reverse_idx=np.asarray(metadata["window_reverse_idx"], dtype=np.int64),
            qkv_type=ov.Type.f32,
            video_output_type=ov.Type.f32,
            include_text_stats=True,
            name="small_deterministic_shared_attention_loop_plus_tail",
        )
        ov.save_model(graph["model"], str(PROOF_XML), compress_to_fp16=True)
        saved_structure = validate_saved_loop_ir(PROOF_XML)
        if saved_structure["loop_count"] != 1 or saved_structure["sdpa_per_loop_body"] != [1]:
            raise AssertionError(f"small saved IR structure mismatch: {saved_structure}")
        if saved_structure["sdpa_count"] != 2:
            raise AssertionError(f"small Loop+tail IR must have two total SDPAs: {saved_structure}")
        if len(model_io(core.read_model(PROOF_XML))["inputs"]) != 9:
            raise AssertionError("small proof graph must expose six QKV plus three full metadata inputs")

        config = {"INFERENCE_NUM_THREADS": CPU_THREADS, "NUM_STREAMS": 1}
        compiled_loop = core.compile_model(graph["model"], "CPU", config)
        loop_result = _compiled_outputs(compiled_loop, feed)

        sequential_chunk, _ = _load_static_attention(
            core,
            source_tokens=SOURCE_TOKENS,
            window_count=CHUNK_SIZE,
            max_video_len=PADDED_VIDEO_LEN,
        )
        sequential_tail, _ = _load_static_attention(
            core,
            source_tokens=SOURCE_TOKENS,
            window_count=TAIL_WINDOWS,
            max_video_len=PADDED_VIDEO_LEN,
        )
        compiled_chunk = core.compile_model(sequential_chunk, "CPU", config)
        compiled_tail = core.compile_model(sequential_tail, "CPU", config)

        video_parts: list[np.ndarray] = []
        text_sum = np.zeros((TXT_TOKENS, INNER_DIM), dtype=np.float32)
        text_count = np.zeros((1,), dtype=np.float32)
        sequential_calls: list[dict[str, int]] = []
        for start in range(0, LOOP_WINDOWS, CHUNK_SIZE):
            chunk_feed = dict(qkv)
            for meta_name in WINDOW_META_NAMES:
                chunk_feed[meta_name] = np.asarray(
                    metadata[meta_name][start : start + CHUNK_SIZE]
                )
            result = _compiled_outputs(compiled_chunk, chunk_feed)
            video_parts.append(result["vid_window_out"])
            text_sum += result["txt_sum"]
            text_count += result["txt_count"]
            sequential_calls.append({"start": start, "end": start + CHUNK_SIZE})

        tail_feed = dict(qkv)
        for meta_name in WINDOW_META_NAMES:
            tail_feed[meta_name] = np.asarray(metadata[meta_name][LOOP_WINDOWS:])
        tail_result = _compiled_outputs(compiled_tail, tail_feed)
        video_parts.append(tail_result["vid_window_out"])
        text_sum += tail_result["txt_sum"]
        text_count += tail_result["txt_count"]
        sequential_calls.append({"start": LOOP_WINDOWS, "end": WINDOW_COUNT})

        sequential_windows = np.concatenate(video_parts, axis=0)
        sequential_video = sequential_windows.reshape(-1, INNER_DIM)[
            np.asarray(metadata["window_reverse_idx"], dtype=np.int64)
        ]
        sequential_text = text_sum / text_count
        reference = {"vid_out": sequential_video, "txt_out": sequential_text}
        metrics = {
            "vid_out": _metric(reference["vid_out"], loop_result["vid_out"]),
            "txt_out": _metric(reference["txt_out"], loop_result["txt_out"]),
            "txt_sum": _metric(text_sum, loop_result["txt_sum"]),
            "txt_count": _metric(text_count, loop_result["txt_count"]),
        }
        for name in ("vid_out", "txt_out", "txt_sum", "txt_count"):
            if not np.isfinite(loop_result[name]).all():
                raise AssertionError(f"Loop {name} contains non-finite values")
        if float(text_count[0]) != float(WINDOW_COUNT):
            raise AssertionError(f"sequential text count mismatch: {text_count}")
        if float(loop_result["txt_count"][0]) != float(WINDOW_COUNT):
            raise AssertionError(f"Loop text count mismatch: {loop_result['txt_count']}")
        if any(
            metric["max_abs_error"] > SAVED_IR_MAX_ABS_THRESHOLD
            or metric["rms_relative_error"] > SAVED_IR_RMS_REL_THRESHOLD
            for metric in metrics.values()
        ):
            raise AssertionError(f"Loop vs sequential tolerance failure: {metrics}")

        receipt.update(
            {
                "status": "pass",
                "host_execution": {
                    "affinity_cpus": sorted(os.sched_getaffinity(0)),
                    "nice_value": os.getpriority(os.PRIO_PROCESS, 0),
                },
                "contract": {
                    "window_lengths": metadata["window_lengths"],
                    "coverage_occurrences_min": int(
                        np.bincount(
                            np.asarray(metadata["window_idx"])[
                                np.asarray(metadata["window_key_mask"][:, :PADDED_VIDEO_LEN])
                            ],
                            minlength=SOURCE_TOKENS,
                        ).min()
                    ),
                    "coverage_occurrences_max": int(
                        np.bincount(
                            np.asarray(metadata["window_idx"])[
                                np.asarray(metadata["window_key_mask"][:, :PADDED_VIDEO_LEN])
                            ],
                            minlength=SOURCE_TOKENS,
                        ).max()
                    ),
                    "reverse_positions_unique": bool(
                        np.unique(metadata["window_reverse_idx"]).size == SOURCE_TOKENS
                    ),
                    "text_count": float(loop_result["txt_count"][0]),
                    "sequential_calls": sequential_calls,
                },
                "comparison": {
                    "reference": "saved shared attention IR called sequentially for two full chunks plus the exact one-window tail",
                    "candidate": "one OpenVINO Loop for the two full chunks, then the exact one-window tail subgraph",
                    "metrics": metrics,
                    "thresholds_passed": True,
                },
                "saved_proof_graph": {
                    "xml": str(PROOF_XML),
                    "bin": str(PROOF_XML.with_suffix(".bin")),
                    "xml_bytes": PROOF_XML.stat().st_size,
                    "bin_bytes": PROOF_XML.with_suffix(".bin").stat().st_size,
                    "total_bytes": PROOF_XML.stat().st_size + PROOF_XML.with_suffix(".bin").stat().st_size,
                    "ordered_ops_outer": graph["outer_ordered_op_counts"],
                    "ordered_ops_loop_body": graph["body_ordered_op_counts"],
                    "ordered_ops_tail": graph["tail_ordered_op_counts"],
                    "saved_xml_structure": saved_structure,
                    "io": model_io(core.read_model(PROOF_XML)),
                    "outer_input_count": len(model_io(core.read_model(PROOF_XML))["inputs"]),
                },
                "production_static_graphs_compiled_or_inferred": False,
                "gpu_loop_gate": "future B580 gate; no GPU Loop compile or inference performed",
            }
        )
        return receipt
    except Exception as exc:
        receipt.update(
            {
                "status": "fail",
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        raise
    finally:
        if graph is not None:
            receipt["graph_build"] = {
                "trip_count": graph["trip_count"],
                "loop_body_ordered_op_counts": graph["body_ordered_op_counts"],
                "outer_ordered_op_counts": graph["outer_ordered_op_counts"],
            }
        RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


class LoopChunkAttentionTests(unittest.TestCase):
    def test_exact_loop_plus_tail_cpu_parity(self) -> None:
        receipt = _cpu_proof()
        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(receipt["contract"]["text_count"], WINDOW_COUNT)
        self.assertEqual(receipt["contract"]["sequential_calls"], [{"start": 0, "end": 2}, {"start": 2, "end": 4}, {"start": 4, "end": 5}])
        self.assertEqual(receipt["comparison"]["thresholds_passed"], True)

    def test_production_static_receipts_show_exact_even_odd_tail_shapes(self) -> None:
        receipt = json.loads(PRODUCTION_RECEIPT.read_text())
        self.assertEqual(receipt["status"], "saved_static_graphs")
        by_label = {variant["label"]: variant for variant in receipt["variants"]}
        self.assertEqual(
            (by_label["even"]["contract"]["actual_window_count"],
             by_label["even"]["contract"]["full_loop_windows"],
             by_label["even"]["contract"]["tail_windows"],
             by_label["even"]["contract"]["max_video_len"]),
            (162, 160, 2, 405),
        )
        self.assertEqual(
            (by_label["odd"]["contract"]["actual_window_count"],
             by_label["odd"]["contract"]["full_loop_windows"],
             by_label["odd"]["contract"]["tail_windows"],
             by_label["odd"]["contract"]["max_video_len"]),
            (200, 192, 8, 405),
        )
        for variant in by_label.values():
            self.assertEqual(variant["saved_graph"]["saved_xml_structure"]["loop_count"], 1)
            self.assertEqual(variant["saved_graph"]["saved_xml_structure"]["sdpa_per_loop_body"], [1])
            self.assertEqual(variant["saved_graph"]["saved_xml_structure"]["sdpa_count"], 2)
            self.assertEqual(variant["contract"]["outer_input_count"], 9)
            self.assertEqual(len(variant["saved_graph"]["inputs"]), 9)
        self.assertFalse(receipt["production_graph_cpu_compile_called"])
        self.assertFalse(receipt["production_graph_infer_called"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LoopChunkAttentionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
