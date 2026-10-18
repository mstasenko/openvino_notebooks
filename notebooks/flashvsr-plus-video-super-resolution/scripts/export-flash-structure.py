#!/usr/bin/env python3
"""Export the reviewed Flash PRE graph as structural OpenVINO IR.

This is a build-time graph export only: the explicit reference root is trusted
executable source, while the manifest hashes that provenance.  No learned
values are loaded, and this script does not create a Core, compile, infer, or
query a device.  The resulting XML/BIN contains ordinary Result ports only.
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
    require_reference_modules,
    require_runtime_version,
    sha256_file,
    source_record,
    validate_constant_nodes,
)

# Keep these narrow private names as compatibility hooks for the existing
# CPU contract tests and for callers that imported the original exporter.
_source_record = source_record
_assert_no_private_paths = assert_no_private_paths


ROWS = 4096
VISIBLE_HW = (2160, 3840)
PRECISION = "bf16"
OPENVINO_VERSION = "2026.5.0-23457-7d81aad55a5"

INPUT_CONTRACT = {
    "x": ([1, ROWS, 1536], "bf16"),
    "t_mod": ([1, 6, 1536], "bf16"),
    "rope_cos": ([1, ROWS, 1, 64], "f32"),
    "rope_sin": ([1, ROWS, 1, 64], "f32"),
    "weight__blocks_0_modulation": ([1, 6, 1536], "bf16"),
    "weight__blocks_0_self_attn_q_weight": ([1536, 1536], "bf16"),
    "weight__blocks_0_self_attn_q_bias": ([1536], "bf16"),
    "weight__blocks_0_self_attn_norm_q_weight": ([1536], "bf16"),
    "weight__blocks_0_self_attn_k_weight": ([1536, 1536], "bf16"),
    "weight__blocks_0_self_attn_k_bias": ([1536], "bf16"),
    "weight__blocks_0_self_attn_norm_k_weight": ([1536], "bf16"),
    "weight__blocks_0_self_attn_v_weight": ([1536, 1536], "bf16"),
    "weight__blocks_0_self_attn_v_bias": ([1536], "bf16"),
}
PRIVATE_RESULTS = (
    "modulation_product", "q_linear_storage", "k_linear_storage",
    "norm1_storage", "q_rms_storage", "k_rms_storage",
)
OUTPUT_CONTRACT = {
    **{name: ([1, ROWS, 1536], "bf16") for name in PRIVATE_RESULTS},
    "q_device_0": ([12, 16, 128, 128], "bf16"),
    "q_device_1": ([12, 16, 128, 128], "bf16"),
    "q_finite": ([], "f16"),
    "k_device": ([32, 128, 12, 128], "bf16"),
    "v_device": ([32, 128, 12, 128], "bf16"),
    "kv_finite": ([], "f16"),
    "q_pooled": ([12, 32, 128], "f32"),
    "k_pooled": ([12, 32, 128], "f32"),
    "gate_msa": ([1, 1, 1536], "bf16"),
    "shift_mlp": ([1, 1, 1536], "bf16"),
    "scale_mlp": ([1, 1, 1536], "bf16"),
    "gate_mlp": ([1, 1, 1536], "bf16"),
}

# These source files are required by the selected graph builder.
# The manifest is still generated from modules actually loaded at runtime.
REQUIRED_REFERENCE_FILES = (
    "openvino/full_runtime/plus_graphs.py",
    "openvino/full_runtime/frame_geometry.py",
    "openvino/full_runtime/prepare_real.py",
    "openvino/full_runtime/real_inputs.py",
    "openvino/full_runtime/source_precision_check.py",
    "openvino/full_runtime/window_pre.py",
    "openvino/full_runtime/physical_modulation.py",
    "openvino/full_runtime/half_storage.py",
    "openvino/full_runtime/device_query_graph.py",
    "openvino/parameter_layers/builders.py",
    "openvino/parameter_layers/contract.py",
    "openvino/full_model/contract.py",
    "openvino/model_edges/convert.py",
    "openvino/block/graph.py",
    "openvino/block/contract.py",
)
REFERENCE_MODULES = {
    "device_query_graph": "openvino/full_runtime/device_query_graph.py",
    "frame_geometry": "openvino/full_runtime/frame_geometry.py",
    "plus_graphs": "openvino/full_runtime/plus_graphs.py",
}


def checked_reference_root(value: Path) -> Path:
    return _checked_reference_root(value, REQUIRED_REFERENCE_FILES)


def _module_source_paths(root: Path) -> list[Path]:
    # The reference root can also contain a pinned runtime/venv sibling.
    # Provenance is limited to the curated graph source tree, not every Python
    # module physically below the reference checkout.
    return loaded_source_paths(root, under=Path(root) / "openvino")


def _require_source_closure(source_files: list[dict[str, Any]]) -> None:
    expected = {Path(relative).as_posix() for relative in REQUIRED_REFERENCE_FILES}
    actual = {record["path"] for record in source_files}
    if not expected.issubset(actual):
        raise ValueError("PRE source provenance is incomplete")


def _port_record(port: Any) -> dict[str, Any]:
    return canonical_port_record(port, include_names=True)


def _check_ports(records: list[dict[str, Any]], contract: dict[str, tuple[list[int], str]], role: str):
    return check_port_records(records, contract, f"PRE {role}")


def _require_runtime_version(version: str) -> None:
    require_runtime_version(version, OPENVINO_VERSION)


def validate_graph_contract(model: Any, *, ov: Any) -> dict[str, Any]:
    inputs = [_port_record(port) for port in model.inputs]
    outputs = [_port_record(port) for port in model.outputs]
    input_by_name = _check_ports(inputs, INPUT_CONTRACT, "input")
    _check_ports(outputs, OUTPUT_CONTRACT, "output")
    if any(port.get_node().get_type_name() != "Parameter" for port in model.inputs):
        raise ValueError("every PRE input must be an ordinary Parameter")
    nodes = list(model.get_ordered_ops())
    if any(port.get_node().get_type_name() != "Result" for port in model.outputs):
        raise ValueError("every PRE output must be an ordinary Result")
    if any(node.get_type_name() in {"ReadValue", "Assign"} for node in nodes):
        raise ValueError("structural PRE export must not contain ReadValue/Assign")
    if list(model.get_variables()) or list(model.get_sinks()):
        raise ValueError("structural PRE export must not contain state Variables/Sinks")
    constant_report = validate_constant_nodes(
        model, forbidden_prefixes=("weight__", "blocks."),
    )
    return {
        "inputs": sorted(inputs, key=lambda record: record["name"]),
        "outputs": sorted(outputs, key=lambda record: record["name"]),
        "private_bf16_results": list(PRIVATE_RESULTS),
        "parameter_input_names": sorted(name for name in input_by_name if name.startswith("weight__")),
        "variables": [], "sinks": [],
        **constant_report,
        "friendly_name": str(model.get_friendly_name()),
        "openvino_type_names": {"bf16": str(ov.Type.bf16), "f32": str(ov.Type.f32)},
    }


def export_structure(reference_root: Path, output: Path) -> dict[str, Any]:
    root = checked_reference_root(reference_root)
    destination = fresh_output(output)
    with reference_import_path(root):
        import openvino as ov

        runtime_version = str(ov.get_version())
        _require_runtime_version(runtime_version)
        from device_query_graph import expose_device_queries
        from frame_geometry import FrameGeometry
        from plus_graphs import build_bounded_pre, preserve_modulation_product
        require_reference_modules(root, REFERENCE_MODULES)

        frame = FrameGeometry(*VISIBLE_HW)
        model, builder_geometry, residual_removed = build_bounded_pre(
            frame, rows=ROWS, precision=PRECISION
        )
        if not residual_removed:
            raise ValueError("bounded PRE builder did not remove its residual Result")
        preserve_modulation_product(
            model, role="pre", linear_stores=True, norm_stores=True, state_sinks=False
        )
        expose_device_queries(model, remote_kv=True, source_type=PRECISION)
        graph = validate_graph_contract(model, ov=ov)

        xml_path = destination / "flash_pre_4096_bf16.xml"
        bin_path = destination / "flash_pre_4096_bf16.bin"
        ov.serialize(model, str(xml_path), str(bin_path))
        for artifact in (xml_path, bin_path):
            _assert_no_private_paths(artifact, artifact.read_bytes())
        source_files = [_source_record(root, path) for path in _module_source_paths(root)]
        _require_source_closure(source_files)
        manifest = {
            "schema": 1, "status": "pass",
            "scope": "structural Flash PRE IR only; no learned constants, compilation, inference, weights, or quality qualification",
            "trust": {
                "reference_root": "explicit curated executable source; not sandboxed",
                "source_hashes_are_provenance_only": True,
            },
            "geometry": {
                "visible_hw": list(VISIBLE_HW), "working_hw": list(frame.working_hw),
                "token_hw": list(frame.token_hw), "rows": ROWS, "precision": PRECISION,
                "builder_geometry": [int(value) for value in builder_geometry],
            },
            "openvino": {
                "required_version": OPENVINO_VERSION, "actual_version": runtime_version,
                "compilation_performed": False, "inference_performed": False,
            },
            "exporter_sha256": sha256_file(Path(__file__).resolve()),
            "export_helper": helper_fingerprint(),
            "source_files": source_files,
            "artifacts": {
                "xml": {"path": xml_path.name, "bytes": xml_path.stat().st_size,
                        "sha256": sha256_file(xml_path)},
                "bin": {"path": bin_path.name, "bytes": bin_path.stat().st_size,
                        "sha256": sha256_file(bin_path)},
            },
            "graph": graph,
        }
        encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        _assert_no_private_paths(destination / "manifest.json", encoded)
        (destination / "manifest.json").write_bytes(encoded)
        return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="curated Flash reference root containing openvino/")
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh output directory for XML/BIN and manifest")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = export_structure(args.reference_root, args.output)
    print(json.dumps({"status": manifest["status"],
                      "xml_sha256": manifest["artifacts"]["xml"]["sha256"],
                      "bin_sha256": manifest["artifacts"]["bin"]["sha256"],
                      "source_files": len(manifest["source_files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
