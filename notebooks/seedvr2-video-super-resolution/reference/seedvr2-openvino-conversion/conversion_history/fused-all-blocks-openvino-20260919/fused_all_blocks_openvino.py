#!/usr/bin/env python3
"""Save and validate one CPU-built fused SeedVR2 Loop IR for each DiT block."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
TRIAL_ROOT = EVIDENCE.parent
DYNAMIC_DIR = EVIDENCE / "dynamic-window-length-openvino-20260918"
LOOP_DIR = EVIDENCE / "loop-chunk-attention-openvino-20260919"
BLOCK0_DIR = EVIDENCE / "fused-loop-block0-openvino-20260919"
RECEIPT = HERE / "receipt.json"
CHUNK_SIZE = 16
EXPECTED_BLOCKS = tuple(range(32))
EXPECTED_GEOMETRY = {"even": (162, 405), "odd": (200, 405)}
FP32 = "<Type: 'float32'>"
I64 = "<Type: 'int64_t'>"
BOOL = "<Type: 'char'>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="must be this script's directory",
    )
    return parser.parse_args()


def write_receipt(value: dict[str, Any]) -> None:
    pending = RECEIPT.with_suffix(".json.pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(pending, RECEIPT)


def check_io_contract(
    io: dict[str, list[dict[str, Any]]], *, window_count: int, max_video_len: int
) -> None:
    expected_inputs = {
        "vid": ("[64800,2560]", FP32),
        "txt": ("[58,2560]", FP32),
        "emb": ("[1,15360]", FP32),
        "window_idx": (f"[{window_count},{max_video_len}]", I64),
        "window_key_mask": (f"[{window_count},{max_video_len + 58}]", BOOL),
        "window_rope_freqs": (f"[{window_count},{max_video_len + 58},126]", FP32),
    }
    actual_inputs = {
        item["name"]: (item["shape"], item["element_type"])
        for item in io["inputs"]
    }
    if len(io["inputs"]) != 6 or actual_inputs != expected_inputs:
        raise AssertionError(f"six-input fused contract differs: {actual_inputs}")

    expected_outputs = {
        "vid_out": ("[64800,2560]", FP32),
        "txt_out": ("[58,2560]", FP32),
    }
    actual_outputs = {
        item["name"]: (item["shape"], item["element_type"])
        for item in io["outputs"]
    }
    if len(io["outputs"]) != 2 or actual_outputs != expected_outputs:
        raise AssertionError(f"fused output contract differs: {actual_outputs}")


def validate_fp32_computation(xml_path: Path) -> dict[str, int]:
    root = ET.parse(xml_path).getroot()
    compressed_constants = 0
    nonconstant_fp16_outputs: list[tuple[str, str]] = []
    for layer in root.iter("layer"):
        layer_type = layer.get("type", "")
        if layer_type in {"Const", "Constant"}:
            if any(
                node.get("element_type") == "f16"
                for node in layer.findall("./data")
            ):
                compressed_constants += 1
            continue
        for port in layer.findall("./output/port"):
            if port.get("precision") in {"FP16", "BF16"}:
                nonconstant_fp16_outputs.append(
                    (layer.get("name", ""), port.get("precision", ""))
                )
        for data in layer.findall("./data"):
            if data.get("element_type") in {"f16", "bf16"}:
                nonconstant_fp16_outputs.append(
                    (layer.get("name", ""), data.get("element_type", ""))
                )
    if nonconstant_fp16_outputs:
        raise AssertionError(
            "non-constant computation outputs are not FP32: "
            f"{nonconstant_fp16_outputs[:10]}"
        )
    if compressed_constants == 0:
        raise AssertionError("FP16 constant compression was not found in serialized IR")
    return {
        "fp16_compressed_constant_layer_count": compressed_constants,
        "nonconstant_fp16_output_count": 0,
    }


def validate_structure(path: Path, structure: dict[str, Any]) -> None:
    if structure["loop_count"] != 1:
        raise AssertionError(f"serialized graph has {structure['loop_count']} Loops")
    if structure["sdpa_count"] != 2:
        raise AssertionError(
            f"serialized graph has {structure['sdpa_count']} SDPAs, expected 2"
        )
    if structure["sdpa_per_loop_body"] != [1]:
        raise AssertionError(
            f"Loop body SDPA count differs: {structure['sdpa_per_loop_body']}"
        )
    if not path.is_file():
        raise AssertionError(f"serialized XML is missing: {path}")


def main() -> int:
    args = parse_args()
    requested_output = args.output_dir.resolve()
    if requested_output != HERE:
        raise SystemExit(f"refusing output outside script directory: {requested_output}")
    if RECEIPT.exists() or RECEIPT.with_suffix(".json.pending").exists():
        raise SystemExit(f"refusing to overwrite an existing receipt in {HERE}")

    expected_outputs = [
        HERE / f"block{index}_{'even' if index % 2 == 0 else 'odd'}_fused_loop_tail_fp16.{suffix}"
        for index in EXPECTED_BLOCKS
        for suffix in ("xml", "bin")
    ]
    existing_outputs = [str(path) for path in expected_outputs if path.exists()]
    if existing_outputs:
        raise SystemExit(f"refusing to overwrite existing graph artifacts: {existing_outputs}")

    source_pairs: dict[int, dict[str, Path]] = {}
    missing_sources: list[str] = []
    for index in EXPECTED_BLOCKS:
        pair = {
            "prefix_xml": DYNAMIC_DIR / f"block{index}_chunked_prefix_fp16.xml",
            "prefix_bin": DYNAMIC_DIR / f"block{index}_chunked_prefix_fp16.bin",
            "suffix_xml": DYNAMIC_DIR / f"block{index}_chunked_suffix_fp16.xml",
            "suffix_bin": DYNAMIC_DIR / f"block{index}_chunked_suffix_fp16.bin",
        }
        missing_sources.extend(str(path) for path in pair.values() if not path.is_file())
        source_pairs[index] = pair
    if missing_sources:
        raise SystemExit(f"source prefix/suffix IRs are missing: {missing_sources}")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(BLOCK0_DIR))
    import openvino as ov
    import fused_block0_proof as proof

    if proof.CHUNK_SIZE != CHUNK_SIZE:
        raise AssertionError(f"proven block0 chunk size changed: {proof.CHUNK_SIZE}")
    if proof.SHAPE != (2, 240, 135) or proof.MAX_VIDEO_LEN != 405:
        raise AssertionError("proven production shape or max window length changed")

    baseline = json.loads((BLOCK0_DIR / "receipt.json").read_text())
    baseline_deltas = baseline["production_bin_deltas"]
    if set(baseline_deltas) != {"even", "odd"}:
        raise AssertionError(f"block0 baseline parity deltas differ: {baseline_deltas}")

    geometries: dict[str, dict[str, Any]] = {}
    for parity, method in proof.WINDOW_METHODS.items():
        window_count, max_video_len, reverse_idx = proof._window_geometry(method)
        if (window_count, max_video_len) != EXPECTED_GEOMETRY[parity]:
            raise AssertionError(
                f"{parity} geometry differs: {(window_count, max_video_len)}"
            )
        if reverse_idx.shape != (64800,) or len(set(reverse_idx.tolist())) != 64800:
            raise AssertionError(f"{parity} reverse index is not an exact token permutation")
        geometries[parity] = {
            "method": method,
            "window_count": window_count,
            "max_video_len": max_video_len,
            "reverse_idx": reverse_idx,
        }

    source_prefix_total = sum(
        source_pairs[index]["prefix_bin"].stat().st_size for index in EXPECTED_BLOCKS
    )
    source_suffix_total = sum(
        source_pairs[index]["suffix_bin"].stat().st_size for index in EXPECTED_BLOCKS
    )
    result: dict[str, Any] = {
        "status": "running",
        "openvino_version": ov.__version__,
        "device": "CPU graph composition, IR read, and serialization only",
        "gpu_used": False,
        "gpu_queried": False,
        "production_graphs_compiled": False,
        "production_graphs_inferred": False,
        "cpu_affinity": "0-3",
        "nice_level": 19,
        "output_dir": str(HERE),
        "builder": str(BLOCK0_DIR / "fused_block0_proof.py"),
        "loop_tail_builder": str(LOOP_DIR / "loop_chunk_attention.py"),
        "geometry_builder": str(LOOP_DIR / "build_production_loop.py"),
        "source_ir_dir": str(DYNAMIC_DIR),
        "geometry": {
            parity: {
                "method": value["method"],
                "window_count": value["window_count"],
                "max_video_len": value["max_video_len"],
                "chunk_size": CHUNK_SIZE,
                "loop_iterations": value["window_count"] // CHUNK_SIZE,
                "tail_windows": value["window_count"] % CHUNK_SIZE,
            }
            for parity, value in geometries.items()
        },
        "baseline_block0_bin_deltas_bytes": baseline_deltas,
        "source_prefix_bin_total_bytes": source_prefix_total,
        "source_suffix_bin_total_bytes": source_suffix_total,
        "source_prefix_suffix_bin_total_bytes": source_prefix_total + source_suffix_total,
        "graphs": [],
    }
    write_receipt(result)

    core = ov.Core()
    start_time = time.perf_counter()
    for index in EXPECTED_BLOCKS:
        parity = "even" if index % 2 == 0 else "odd"
        geometry = geometries[parity]
        pair = source_pairs[index]
        proof.PREFIX_XML = pair["prefix_xml"]
        proof.SUFFIX_XML = pair["suffix_xml"]

        graph_name = f"seedvr2_block{index}_{parity}_fused_loop_tail"
        model = proof.compose_block(
            core,
            source_tokens=64800,
            total_windows=geometry["window_count"],
            max_video_len=geometry["max_video_len"],
            chunk_size=CHUNK_SIZE,
            reverse_idx=geometry["reverse_idx"],
            name=graph_name,
        )

        xml_path = HERE / f"block{index}_{parity}_fused_loop_tail_fp16.xml"
        bin_path = xml_path.with_suffix(".bin")
        ov.save_model(model, str(xml_path), compress_to_fp16=True)
        del model
        gc.collect()

        structure = proof.xml_loop_structure(xml_path)
        validate_structure(xml_path, structure)
        saved = core.read_model(xml_path)
        io = proof.model_io(saved)
        check_io_contract(
            io,
            window_count=geometry["window_count"],
            max_video_len=geometry["max_video_len"],
        )
        precision = validate_fp32_computation(xml_path)
        del saved
        gc.collect()

        if not bin_path.is_file():
            raise AssertionError(f"serialized BIN is missing: {bin_path}")
        source_prefix_bytes = pair["prefix_bin"].stat().st_size
        source_suffix_bytes = pair["suffix_bin"].stat().st_size
        source_pair_bytes = source_prefix_bytes + source_suffix_bytes
        fused_bin_bytes = bin_path.stat().st_size
        bin_delta = fused_bin_bytes - source_pair_bytes
        if not 0 <= bin_delta <= 1_000_000:
            raise AssertionError(
                f"block {index} fused BIN overhead is outside the attention-constant envelope: "
                f"{bin_delta} bytes"
            )

        graph = {
            "block_index": index,
            "parity": parity,
            "window_method": geometry["method"],
            "window_count": geometry["window_count"],
            "max_video_len": geometry["max_video_len"],
            "chunk_size": CHUNK_SIZE,
            "loop_iterations": geometry["window_count"] // CHUNK_SIZE,
            "tail_windows": geometry["window_count"] % CHUNK_SIZE,
            "source_prefix_xml": str(pair["prefix_xml"].relative_to(TRIAL_ROOT)),
            "source_suffix_xml": str(pair["suffix_xml"].relative_to(TRIAL_ROOT)),
            "source_prefix_bin_bytes": source_prefix_bytes,
            "source_suffix_bin_bytes": source_suffix_bytes,
            "source_prefix_suffix_bin_bytes": source_pair_bytes,
            "xml_path": str(xml_path.relative_to(TRIAL_ROOT)),
            "bin_path": str(bin_path.relative_to(TRIAL_ROOT)),
            "xml_bytes": xml_path.stat().st_size,
            "bin_bytes": fused_bin_bytes,
            "bin_delta_vs_source_pair_bytes": bin_delta,
            "xml_sha256": proof.sha256(xml_path),
            "bin_sha256": proof.sha256(bin_path),
            "inputs": io["inputs"],
            "outputs": io["outputs"],
            "structure": structure,
            "precision": precision,
            "production_compiled": False,
            "production_inferred": False,
        }
        result["graphs"].append(graph)
        result["fused_bin_total_bytes"] = sum(
            item["bin_bytes"] for item in result["graphs"]
        )
        result["completed_blocks"] = len(result["graphs"])
        result["elapsed_seconds"] = time.perf_counter() - start_time
        write_receipt(result)
        print(
            json.dumps(
                {
                    "block": index,
                    "parity": parity,
                    "xml": graph["xml_path"],
                    "bin_bytes": fused_bin_bytes,
                    "bin_delta_bytes": bin_delta,
                    "xml_sha256": graph["xml_sha256"],
                    "bin_sha256": graph["bin_sha256"],
                    "structure": structure,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del geometry, pair, graph
        gc.collect()

    if len(result["graphs"]) != 32:
        raise AssertionError(f"only {len(result['graphs'])} block graphs were saved")
    fused_total = sum(item["bin_bytes"] for item in result["graphs"])
    source_total = source_prefix_total + source_suffix_total
    aggregate_delta = fused_total - source_total
    expected_baseline_delta = sum(
        baseline_deltas["even" if index % 2 == 0 else "odd"]
        for index in EXPECTED_BLOCKS
    )
    result["aggregate"] = {
        "block_count": len(result["graphs"]),
        "fused_bin_total_bytes": fused_total,
        "source_prefix_bin_total_bytes": source_prefix_total,
        "source_suffix_bin_total_bytes": source_suffix_total,
        "source_prefix_suffix_bin_total_bytes": source_total,
        "fused_minus_source_prefix_suffix_bin_bytes": aggregate_delta,
        "expected_total_from_block0_baseline_delta_bytes": expected_baseline_delta,
        "delta_matches_block0_baseline": aggregate_delta == expected_baseline_delta,
        "delta_fraction_of_source_percent": (100.0 * aggregate_delta / source_total),
        "minimum_per_block_bin_delta_bytes": min(
            item["bin_delta_vs_source_pair_bytes"] for item in result["graphs"]
        ),
        "maximum_per_block_bin_delta_bytes": max(
            item["bin_delta_vs_source_pair_bytes"] for item in result["graphs"]
        ),
    }
    result["elapsed_seconds"] = time.perf_counter() - start_time
    result["status"] = "pass"
    result["all_structure_contract_hash_size_checks_pass"] = True
    write_receipt(result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "blocks": len(result["graphs"]),
                "aggregate": result["aggregate"],
                "elapsed_seconds": result["elapsed_seconds"],
                "receipt": str(RECEIPT.relative_to(TRIAL_ROOT)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
