#!/usr/bin/env python3
"""Export the fixed Flash text MLP without loading learned weights.

The caller selects trusted executable reference sources. All four learned
tensors remain BF16 Parameters, and the observable text_linear Result keeps
the source rounding boundary before GELU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from flash_export import (
    canonical_port_record, checked_reference_root, check_port_records,
    fresh_output, graph_source_records, reference_import_path,
    reject_f16_intermediates, require_reference_modules, require_runtime_version,
    serialize_structural_ir, sha256_file, validate_constant_nodes,
    write_structural_manifest,
)


OPENVINO_VERSION = "2026.5.0-23457-7d81aad55a5"
PRECISION = "bf16"
PROMPT_SHAPE = [1, 512, 4096]
TEXT_SHAPE = [1, 512, 1536]
TEXT_WEIGHTS = {
    "weight__text_embedding_0_weight": [1536, 4096],
    "weight__text_embedding_0_bias": [1536],
    "weight__text_embedding_2_weight": [1536, 1536],
    "weight__text_embedding_2_bias": [1536],
}
INPUT_CONTRACT = {"prompt": (PROMPT_SHAPE, PRECISION), **{
    name: (shape, PRECISION) for name, shape in TEXT_WEIGHTS.items()
}}
OUTPUT_CONTRACT = {"context": (TEXT_SHAPE, PRECISION),
                   "text_linear": (TEXT_SHAPE, PRECISION)}
REFERENCE_MODULES = {
    "plus_text": "openvino/full_runtime/plus_text.py",
    "plus_storage": "openvino/full_runtime/plus_storage.py",
    "plus_weights": "openvino/full_runtime/plus_weights.py",
    "plus_edges": "openvino/full_runtime/plus_edges.py",
    "plus_modes": "openvino/full_runtime/plus_modes.py",
    "plus_weights_parameter_contract": "openvino/parameter_layers/contract.py",
}
REQUIRED_REFERENCE_FILES = tuple(REFERENCE_MODULES.values())


def validate_graph_contract(model: Any) -> dict[str, Any]:
    inputs = [canonical_port_record(port) for port in model.inputs]
    outputs = [canonical_port_record(port) for port in model.outputs]
    check_port_records(inputs, INPUT_CONTRACT, "text input")
    check_port_records(outputs, OUTPUT_CONTRACT, "text output")
    if any(port.get_node().get_type_name() != "Parameter" for port in model.inputs):
        raise ValueError("text inputs must be ordinary Parameters")
    if list(model.get_variables()) or list(model.get_sinks()):
        raise ValueError("text graph must not contain hidden state")
    if any(port.get_node().get_type_name() != "Result" for port in model.outputs):
        raise ValueError("text graph must have ordinary Result outputs")
    nodes = list(model.get_ordered_ops())
    if any(node.get_type_name() in {"ReadValue", "Assign"} for node in nodes):
        raise ValueError("text graph must not contain state operations")
    reject_f16_intermediates(model, "text")
    constants = validate_constant_nodes(model, forbidden_prefixes=("weight__",))
    gelus = [node for node in nodes if node.get_type_name() == "Gelu"]
    if len(gelus) != 1 or str(gelus[0].get_attributes().get("approximation_mode")).lower() != "tanh":
        raise ValueError("text graph requires the source tanh GELU")
    # The Result must observe the exact edge consumed by GELU, not a later
    # recomputed value with the same shape/name.
    stored = model.output("text_linear").get_node().input_value(0)
    if gelus[0].input_value(0) != stored:
        raise ValueError("text graph lost its pre-GELU BF16 storage boundary")
    if str(model.get_friendly_name()) != "flashvsr_plus_text_bf16":
        raise ValueError("text graph friendly name changed")
    return {"friendly_name": str(model.get_friendly_name()),
            "inputs": sorted(inputs, key=lambda item: item["name"]),
            "outputs": sorted(outputs, key=lambda item: item["name"]),
            "learned_parameter_names": sorted(TEXT_WEIGHTS),
            "storage_boundary": "text_linear before GELU(tanh)", **constants}


def export_text(reference_root: Path, output: Path) -> dict[str, Any]:
    root = checked_reference_root(reference_root, REQUIRED_REFERENCE_FILES)
    destination = fresh_output(output)
    with reference_import_path(root):
        import openvino as ov
        from plus_text import build_text_embedding

        require_runtime_version(str(ov.get_version()), OPENVINO_VERSION)
        require_reference_modules(root, REFERENCE_MODULES)
        model = build_text_embedding(None, PRECISION, preserve_storage=True)
        graph = validate_graph_contract(model)
        artifacts = serialize_structural_ir(ov, model, destination, "flash_text_bf16")
        return write_structural_manifest(destination, {
            "schema": 1, "status": "pass",
            "scope": "structural Flash text IR; learned BF16 Parameters only",
            "trust": {"reference_root": "explicit curated executable source; not sandboxed",
                      "source_hashes_are_provenance_only": True,
                      "learned_payloads_loaded": False},
            "graph_role": "prompt_to_text_context", "precision": PRECISION,
            "openvino": {"required_version": OPENVINO_VERSION,
                         "actual_version": str(ov.get_version()),
                         "compilation_performed": False, "inference_performed": False},
            "exporter_sha256": sha256_file(Path(__file__).resolve()),
            "source_files": graph_source_records(root, REQUIRED_REFERENCE_FILES),
            "artifacts": artifacts, "graph": graph,
        })


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = export_text(args.reference_root, args.output)
    print(json.dumps({"status": manifest["status"], "artifacts": manifest["artifacts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
