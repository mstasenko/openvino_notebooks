#!/usr/bin/env python3
"""Export the Flash cross-attention K/V graph as structural OpenVINO IR.

This is a build-time graph export.  The selected reference root is trusted
executable input, while the manifest records its exact source closure.  The
five per-layer tensors remain BF16 Parameters; no learned payload is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from flash_export import (
    canonical_port_record,
    checked_reference_root as _checked_reference_root,
    check_port_records,
    fresh_output,
    graph_source_records,
    reference_import_path,
    reject_f16_intermediates,
    require_reference_modules,
    require_runtime_version,
    serialize_structural_ir,
    sha256_file,
    validate_constant_nodes,
    write_structural_manifest,
)


OPENVINO_VERSION = "2026.5.0-23457-7d81aad55a5"
PRECISION = "bf16"
TEXT_SHAPE = [1, 512, 1536]
CROSS_WEIGHTS = {
    "weight__blocks_0_cross_attn_k_weight": [1536, 1536],
    "weight__blocks_0_cross_attn_k_bias": [1536],
    "weight__blocks_0_cross_attn_norm_k_weight": [1536],
    "weight__blocks_0_cross_attn_v_weight": [1536, 1536],
    "weight__blocks_0_cross_attn_v_bias": [1536],
}
INPUT_CONTRACT = {"context": (TEXT_SHAPE, PRECISION), **{
    name: (shape, PRECISION) for name, shape in CROSS_WEIGHTS.items()
}}
OUTPUT_CONTRACT = {"cross_k": (TEXT_SHAPE, PRECISION),
                   "cross_v": (TEXT_SHAPE, PRECISION)}

REQUIRED_REFERENCE_FILES = (
    "openvino/full_runtime/plus_text.py",
    "openvino/full_runtime/plus_storage.py",
    "openvino/full_runtime/plus_weights.py",
    "openvino/full_runtime/plus_edges.py",
    "openvino/full_runtime/plus_modes.py",
    "openvino/parameter_layers/contract.py",
)
REFERENCE_MODULES = {
    "plus_text": "openvino/full_runtime/plus_text.py",
    "plus_storage": "openvino/full_runtime/plus_storage.py",
    "plus_weights": "openvino/full_runtime/plus_weights.py",
    "plus_edges": "openvino/full_runtime/plus_edges.py",
    "plus_modes": "openvino/full_runtime/plus_modes.py",
    "plus_weights_parameter_contract": "openvino/parameter_layers/contract.py",
}


def checked_reference_root(value: Path) -> Path:
    return _checked_reference_root(value, REQUIRED_REFERENCE_FILES)


def _port_records(model: Any, role: str) -> list[dict[str, Any]]:
    return [canonical_port_record(port) for port in getattr(model, role)]


def validate_graph_contract(model: Any, *, ov: Any) -> dict[str, Any]:
    inputs = _port_records(model, "inputs")
    outputs = _port_records(model, "outputs")
    check_port_records(inputs, INPUT_CONTRACT, "cross K/V input")
    check_port_records(outputs, OUTPUT_CONTRACT, "cross K/V output")
    input_names = {record["name"] for record in inputs}
    output_names = {record["name"] for record in outputs}
    if input_names & output_names:
        raise ValueError("cross K/V input/output names overlap")
    if any(port.get_node().get_type_name() != "Parameter" for port in model.inputs):
        raise ValueError("cross K/V inputs must be ordinary Parameters")
    if any(port.get_node().get_type_name() != "Result" for port in model.outputs):
        raise ValueError("cross K/V outputs must be ordinary Result nodes")
    nodes = list(model.get_ordered_ops())
    if any(node.get_type_name() in {"ReadValue", "Assign"} for node in nodes):
        raise ValueError("cross K/V graph must not contain state nodes")
    if list(model.get_variables()) or list(model.get_sinks()):
        raise ValueError("cross K/V graph must not contain Variables or Sinks")
    reject_f16_intermediates(model, "cross K/V")
    constant_report = validate_constant_nodes(model, forbidden_prefixes=("weight__",))
    if str(model.get_friendly_name()) != "flashvsr_plus_cross_kv_bf16":
        raise ValueError("cross K/V graph friendly name changed")
    return {
        "friendly_name": str(model.get_friendly_name()),
        "inputs": sorted(inputs, key=lambda record: record["name"]),
        "outputs": sorted(outputs, key=lambda record: record["name"]),
        "learned_parameter_names": sorted(CROSS_WEIGHTS),
        "context_shape": list(TEXT_SHAPE),
        "cross_shape": list(TEXT_SHAPE),
        "state_variables": [],
        "sinks": [],
        "openvino_types": {"bf16": str(ov.Type.bf16), "f32": str(ov.Type.f32)},
        **constant_report,
    }


def export_cross(reference_root: Path, output: Path) -> dict[str, Any]:
    root = checked_reference_root(reference_root)
    destination = fresh_output(output)
    with reference_import_path(root):
        import openvino as ov
        from plus_text import build_cross_kv

        runtime_version = str(ov.get_version())
        require_runtime_version(runtime_version, OPENVINO_VERSION)
        require_reference_modules(root, REFERENCE_MODULES)
        model = build_cross_kv(None, precision=PRECISION)
        graph = validate_graph_contract(model, ov=ov)
        artifacts = serialize_structural_ir(ov, model, destination, "flash_cross_kv_bf16")
        source_files = graph_source_records(root, REQUIRED_REFERENCE_FILES)
        manifest = {
            "schema": 1,
            "status": "pass",
            "scope": "structural Flash cross K/V IR; learned BF16 Parameters only",
            "trust": {
                "reference_root": "explicit curated executable source; not sandboxed",
                "source_hashes_are_provenance_only": True,
                "learned_payloads_loaded": False,
            },
            "graph_role": "context_to_cross_k_cross_v",
            "precision": PRECISION,
            "context_shape": list(TEXT_SHAPE),
            "cross_shape": list(TEXT_SHAPE),
            "openvino": {
                "required_version": OPENVINO_VERSION,
                "actual_version": runtime_version,
                "compilation_performed": False,
                "inference_performed": False,
            },
            "exporter_sha256": sha256_file(Path(__file__).resolve()),
            "source_files": source_files,
            "artifacts": artifacts,
            "graph": graph,
        }
        return write_structural_manifest(destination, manifest)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="curated Flash reference root containing openvino/")
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh output directory for XML/BIN and manifest")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = export_cross(args.reference_root, args.output)
    print(json.dumps({"status": manifest["status"],
                      "xml_sha256": manifest["artifacts"]["xml"]["sha256"],
                      "bin_sha256": manifest["artifacts"]["bin"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
