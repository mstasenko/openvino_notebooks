#!/usr/bin/env python3
"""Export the Flash LQ projection's external-weight structural IR.

The explicit reference root is trusted executable build input.  This command
only constructs and serializes OpenVINO graphs; it does not load learned
payloads, create a Core, compile, infer, or query a device.  Learned tensors
remain BF16 Parameters in the emitted IR and are supplied later by the native
weight binder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from flash_export import (
    assert_no_private_paths,
    canonical_port_record,
    checked_reference_root as _checked_reference_root,
    check_port_records,
    fresh_output,
    helper_fingerprint,
    loaded_source_paths,
    reference_import_path,
    reject_f16_intermediates,
    require_reference_modules,
    require_runtime_version,
    sha256_file,
    source_record,
    validate_constant_nodes,
)

# Preserve the original exporter-local hooks while sharing their implementation
# with the PRE exporter.
_source_record = source_record
_assert_no_private_paths = assert_no_private_paths


VISIBLE_HW = (2160, 3840)
WORKING_HW = (2176, 3840)
PRECISION = "bf16"
OPENVINO_VERSION = "2026.5.0-23457-7d81aad55a5"
DATA_INPUT_SHAPES = {
    "video_startup": [1, 3, 1, *WORKING_HW],
    "video_steady": [1, 3, 4, *WORKING_HW],
    "cache1": [1, 768, 2, WORKING_HW[0] // 16, WORKING_HW[1] // 16],
    "cache2": [1, 2048, 2, WORKING_HW[0] // 16, WORKING_HW[1] // 16],
    "old_cache2": [1, 2048, 2, WORKING_HW[0] // 16, WORKING_HW[1] // 16],
    "current_cache2": [1, 2048, 2, WORKING_HW[0] // 16, WORKING_HW[1] // 16],
}
TOKEN_COUNT = (WORKING_HW[0] // 16) * (WORKING_HW[1] // 16)
WEIGHT_SHAPES = {
    "weight__conv1_bias": [2048],
    "weight__conv1_weight": [2048, 768, 4, 3, 3],
    "weight__conv2_bias": [3072],
    "weight__conv2_weight": [3072, 2048, 4, 3, 3],
    "weight__linear_layers_0_bias": [1536],
    "weight__linear_layers_0_weight": [1536, 3072],
    "weight__norm1_weight": [2048, 1, 1, 1],
    "weight__norm2_weight": [3072, 1, 1, 1],
}
STARTUP_WEIGHTS = (
    "weight__conv1_bias", "weight__conv1_weight", "weight__norm1_weight",
)
STAGE2_WEIGHTS = (
    "weight__conv2_bias", "weight__conv2_weight",
    "weight__linear_layers_0_bias", "weight__linear_layers_0_weight",
    "weight__norm2_weight",
)
REQUIRED_REFERENCE_FILES = (
    "openvino/full_runtime/plus_projection.py",
    "openvino/full_runtime/plus_storage.py",
    "openvino/full_runtime/plus_modes.py",
)
REFERENCE_MODULES = {
    "plus_projection": "openvino/full_runtime/plus_projection.py",
    "plus_storage": "openvino/full_runtime/plus_storage.py",
    "plus_modes": "openvino/full_runtime/plus_modes.py",
}


def checked_reference_root(value: Path) -> Path:
    return _checked_reference_root(value, REQUIRED_REFERENCE_FILES)


def _loaded_source_paths(root: Path) -> list[Path]:
    return loaded_source_paths(root, under=root / "openvino" / "full_runtime")


def _require_runtime_version(version: str) -> None:
    require_runtime_version(version, OPENVINO_VERSION)


def _port_record(port: Any) -> dict[str, Any]:
    return canonical_port_record(port)


def _check_ports(model: Any, contract: dict[str, tuple[list[int], str]], role: str):
    records = [_port_record(port) for port in getattr(model, role)]
    check_port_records(records, contract, f"projection {role}")
    return records


def _data(shape_key: str) -> tuple[list[int], str]:
    return (DATA_INPUT_SHAPES[shape_key], "bf16")


def _graph_contracts() -> dict[str, dict[str, tuple[list[int], str]]]:
    cache_outputs = {
        "cache1": (DATA_INPUT_SHAPES["cache1"], "bf16"),
        "cache2": (DATA_INPUT_SHAPES["cache2"], "bf16"),
    }
    return {
        "startup": {
            "inputs": {"video": _data("video_startup"),
                       **{name: (shape, "bf16") for name, shape in WEIGHT_SHAPES.items()
                          if name in STARTUP_WEIGHTS}},
            "outputs": cache_outputs,
        },
        "stage1": {
            # The internal graph calls this state ``cache1``.  At the IR
            # boundary it must be distinct from the output cache1 Result;
            # export_projection renames the cloned Parameter to old_cache1.
            "inputs": {"video": _data("video_steady"),
                       "old_cache1": _data("cache1"),
                       **{name: (WEIGHT_SHAPES[name], "bf16") for name in STARTUP_WEIGHTS}},
            "outputs": cache_outputs,
        },
        "stage2": {
            "inputs": {"old_cache2": _data("old_cache2"),
                       "current_cache2": _data("current_cache2"),
                       **{name: (WEIGHT_SHAPES[name], "bf16") for name in STAGE2_WEIGHTS}},
            "outputs": {"projected": ([1, TOKEN_COUNT, 1536], "bf16")},
        },
    }


def validate_graph_contract(model: Any, *, graph_name: str) -> dict[str, Any]:
    contracts = _graph_contracts()
    if graph_name not in contracts:
        raise ValueError(f"unknown projection graph: {graph_name}")
    contract = contracts[graph_name]
    raw_input_names = {canonical_port_record(port)["name"] for port in model.inputs}
    raw_output_names = {canonical_port_record(port)["name"] for port in model.outputs}
    overlap = sorted(raw_input_names & raw_output_names)
    if overlap:
        raise ValueError(
            f"projection input/output names collide at the IR boundary: {overlap}"
        )
    inputs = _check_ports(model, contract["inputs"], "inputs")
    outputs = _check_ports(model, contract["outputs"], "outputs")
    if any(port.get_node().get_type_name() != "Parameter" for port in model.inputs):
        raise ValueError("projection inputs must be ordinary Parameters")
    nodes = list(model.get_ordered_ops())
    if any(port.get_node().get_type_name() != "Result" for port in model.outputs):
        raise ValueError("projection outputs must be ordinary Result nodes")
    if any(node.get_type_name() in {"ReadValue", "Assign"} for node in nodes):
        raise ValueError("projection export must not contain state nodes")
    if list(model.get_variables()) or list(model.get_sinks()):
        raise ValueError("projection export must not contain Variables or Sinks")
    reject_f16_intermediates(model, "projection")
    constant_report = validate_constant_nodes(model, forbidden_prefixes=("weight__",))
    learned_inputs = sorted(record["name"] for record in inputs
                            if record["name"].startswith("weight__"))
    expected_learned = sorted(name for name in contract["inputs"] if name.startswith("weight__"))
    if learned_inputs != expected_learned:
        raise ValueError("learned projection input ABI changed")
    return {
        "graph": graph_name,
        "friendly_name": str(model.get_friendly_name()),
        "inputs": sorted(inputs, key=lambda record: record["name"]),
        "outputs": sorted(outputs, key=lambda record: record["name"]),
        "learned_parameter_names": learned_inputs,
        **constant_report,
        "state_variables": [],
        "sinks": [],
    }


def _normalize_prefix_cache1_parameter(model: Any) -> Any:
    """Give the prefix's old cache input a name distinct from its output."""

    candidates = [
        parameter for parameter in model.parameters
        if sorted(str(name) for name in parameter.output(0).get_names()) == ["cache1"]
    ]
    if len(candidates) != 1:
        raise ValueError("prefix must contain exactly one cache1 input Parameter")
    parameter = candidates[0]
    if str(parameter.get_friendly_name()) != "cache1":
        raise ValueError("prefix cache1 Parameter friendly name changed")
    parameter.set_friendly_name("old_cache1")
    parameter.output(0).get_tensor().set_names({"old_cache1"})
    model.validate_nodes_and_infer_types()
    return model


def export_projection(reference_root: Path, output: Path) -> dict[str, Any]:
    root = checked_reference_root(reference_root)
    destination = fresh_output(output)
    with reference_import_path(root):
        import openvino as ov
        from plus_projection import build_projection_graph, split_projection_graph

        runtime_version = str(ov.get_version())
        _require_runtime_version(runtime_version)
        require_reference_modules(root, REFERENCE_MODULES)
        startup = build_projection_graph(None, True, PRECISION,
                                         height=WORKING_HW[0], width=WORKING_HW[1])
        steady = build_projection_graph(None, False, PRECISION,
                                        height=WORKING_HW[0], width=WORKING_HW[1])
        stage1, stage2 = split_projection_graph(steady)
        stage1 = _normalize_prefix_cache1_parameter(stage1)
        models = {"startup": startup, "stage1": stage1, "stage2": stage2}
        graph_reports = {}
        artifacts = {}
        for name, model in models.items():
            graph_reports[name] = validate_graph_contract(model, graph_name=name)
            xml_path = destination / f"flash_lq_projection_{name}_bf16.xml"
            bin_path = destination / f"flash_lq_projection_{name}_bf16.bin"
            ov.serialize(model, str(xml_path), str(bin_path))
            for artifact in (xml_path, bin_path):
                _assert_no_private_paths(artifact, artifact.read_bytes())
            artifacts[name] = {
                "xml": {"path": xml_path.name, "bytes": xml_path.stat().st_size,
                        "sha256": sha256_file(xml_path)},
                "bin": {"path": bin_path.name, "bytes": bin_path.stat().st_size,
                        "sha256": sha256_file(bin_path)},
            }
        source_files = [_source_record(root, path) for path in _loaded_source_paths(root)]
        expected_paths = {Path(relative).as_posix() for relative in REQUIRED_REFERENCE_FILES}
        actual_paths = {record["path"] for record in source_files}
        if not expected_paths.issubset(actual_paths):
            raise ValueError("projection source provenance is incomplete")
        manifest = {
            "schema": 1,
            "status": "pass",
            "scope": "structural Flash LQ projection IR; learned BF16 Parameters only",
            "trust": {
                "reference_root": "explicit curated executable source; not sandboxed",
                "source_hashes_are_provenance_only": True,
                "learned_payloads_loaded": False,
            },
            "geometry": {
                "visible_hw": list(VISIBLE_HW), "working_hw": list(WORKING_HW),
                "token_hw": [WORKING_HW[0] // 16, WORKING_HW[1] // 16],
                "token_count": TOKEN_COUNT, "precision": PRECISION,
            },
            "openvino": {
                "required_version": OPENVINO_VERSION,
                "actual_version": runtime_version,
                "compilation_performed": False,
                "inference_performed": False,
            },
            "exporter_sha256": sha256_file(Path(__file__).resolve()),
            "export_helper": helper_fingerprint(),
            "source_files": source_files,
            "artifacts": artifacts,
            "graphs": graph_reports,
        }
        encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        _assert_no_private_paths(destination / "manifest.json", encoded)
        (destination / "manifest.json").write_bytes(encoded)
        return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="curated reference root containing openvino/full_runtime")
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh output directory for projection IR and manifest")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = export_projection(args.reference_root, args.output)
    print(json.dumps({
        "status": manifest["status"],
        "graphs": sorted(manifest["graphs"]),
        "artifact_count": sum(len(value) for value in manifest["artifacts"].values()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
