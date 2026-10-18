#!/usr/bin/env python3
"""Export the FlashVSR+ edge graphs as structural BF16 OpenVINO IR.

This is a build-time export only.  The explicit reference root is trusted
executable source; its hashes are provenance, not a sandbox.  Learned edge
weights remain BF16 Parameters and no checkpoint payload is read.  The prefix
expects caller-prepared BF16 ``t`` and ``t_mod`` values: source-F32/time-weight
conditioning must be authenticated and performed once outside this graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from flash_export import (
    canonical_port_record,
    checked_reference_root,
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
VISIBLE_HW = (2160, 3840)
WORKING_HW = (2176, 3840)
EDGE_WEIGHT_SHAPES = {
    "patch_embedding.weight": [1536, 16, 1, 2, 2],
    "patch_embedding.bias": [1536],
    "head.modulation": [1, 2, 1536],
    "head.proj.weight": [64, 1536],
    "head.proj.bias": [64],
}
WEIGHT_PORTS = {
    key: "weight__" + key.replace(".", "_") for key in EDGE_WEIGHT_SHAPES
}
PREFIX_WEIGHT_KEYS = ("patch_embedding.weight", "patch_embedding.bias")
SUFFIX_WEIGHT_KEYS = ("head.modulation", "head.proj.weight", "head.proj.bias")

REFERENCE_MODULES = {
    "plus_edges": "openvino/full_runtime/plus_edges.py",
    "plus_graphs": "openvino/full_runtime/plus_graphs.py",
    "frame_geometry": "openvino/full_runtime/frame_geometry.py",
    "prepare_real": "openvino/full_runtime/prepare_real.py",
    "real_inputs": "openvino/full_runtime/real_inputs.py",
    "source_precision_check": "openvino/full_runtime/source_precision_check.py",
    "plus_storage": "openvino/full_runtime/plus_storage.py",
    "plus_modes": "openvino/full_runtime/plus_modes.py",
}
REQUIRED_REFERENCE_FILES = tuple(REFERENCE_MODULES.values())


def _contracts(geometry: Any, *, role: str) -> tuple[dict[str, tuple[list[int], str]],
                                                    dict[str, tuple[list[int], str]]]:
    try:
        frames, height, width = geometry.f, geometry.h, geometry.w
    except AttributeError as error:
        raise ValueError("edge contract geometry must expose f, h, and w") from error
    if (type(frames) is not int or type(height) is not int or type(width) is not int
            or frames != 4 or height <= 0 or width <= 0):
        raise ValueError("edge contract geometry is invalid")
    shapes = {
        "latent": [1, 16, frames, height * 2, width * 2],
        "tokens": [1, frames * height * width, 1536],
        "t": [1, 1, 1536],
        "t_mod": [1, 6, 1536],
        "embedding": [1, 256],
        "output": [1, 16, frames, height * 2, width * 2],
    }
    name = str(geometry.name)
    if role == "prefix":
        inputs = {
            f"{name}_x": (list(shapes["latent"]), PRECISION),
            "t": (list(shapes["t"]), PRECISION),
            "t_mod": (list(shapes["t_mod"]), PRECISION),
            WEIGHT_PORTS["patch_embedding.weight"]:
                (list(EDGE_WEIGHT_SHAPES["patch_embedding.weight"]), PRECISION),
            WEIGHT_PORTS["patch_embedding.bias"]:
                (list(EDGE_WEIGHT_SHAPES["patch_embedding.bias"]), PRECISION),
        }
        outputs = {
            f"{name}_tokens_out": (list(shapes["tokens"]), PRECISION),
            f"{name}_t_out": (list(shapes["t"]), PRECISION),
            f"{name}_t_mod_out": (list(shapes["t_mod"]), PRECISION),
            f"{name}_embedding_out": (list(shapes["embedding"]), PRECISION),
        }
        return inputs, outputs
    if role == "suffix":
        inputs = {
            f"{name}_tokens_in": (list(shapes["tokens"]), PRECISION),
            f"{name}_t_in": (list(shapes["t"]), PRECISION),
            WEIGHT_PORTS["head.modulation"]:
                (list(EDGE_WEIGHT_SHAPES["head.modulation"]), PRECISION),
            WEIGHT_PORTS["head.proj.weight"]:
                (list(EDGE_WEIGHT_SHAPES["head.proj.weight"]), PRECISION),
            WEIGHT_PORTS["head.proj.bias"]:
                (list(EDGE_WEIGHT_SHAPES["head.proj.bias"]), PRECISION),
        }
        return inputs, {f"{name}_out": (list(shapes["output"]), PRECISION)}
    raise ValueError(f"unknown edge graph role: {role}")


def validate_graph_contract(model: Any, geometry: Any, *, role: str,
                            ov: Any) -> dict[str, Any]:
    expected_inputs, expected_outputs = _contracts(geometry, role=role)
    inputs = [canonical_port_record(port) for port in model.inputs]
    outputs = [canonical_port_record(port) for port in model.outputs]
    check_port_records(inputs, expected_inputs, f"edge {role} input")
    check_port_records(outputs, expected_outputs, f"edge {role} output")
    input_names = {record["name"] for record in inputs}
    output_names = {record["name"] for record in outputs}
    if input_names & output_names:
        raise ValueError(f"edge {role} input/output names overlap")
    if any(port.get_node().get_type_name() != "Parameter" for port in model.inputs):
        raise ValueError(f"edge {role} inputs must be Parameters")
    if any(port.get_node().get_type_name() != "Result" for port in model.outputs):
        raise ValueError(f"edge {role} outputs must be ordinary Results")
    nodes = list(model.get_ordered_ops())
    if any(node.get_type_name() in {"ReadValue", "Assign"} for node in nodes):
        raise ValueError(f"edge {role} graph must not contain state operations")
    if list(model.get_variables()) or list(model.get_sinks()):
        raise ValueError(f"edge {role} graph must not contain hidden state")
    reject_f16_intermediates(model, f"edge {role}")
    constants = validate_constant_nodes(model, forbidden_prefixes=("weight__",))
    learned = sorted(record["name"] for record in inputs
                     if record["name"].startswith("weight__"))
    expected_learned = sorted(name for name in expected_inputs
                              if name.startswith("weight__"))
    if learned != expected_learned:
        raise ValueError(f"edge {role} learned Parameter ABI changed")
    expected_name = f"flashvsr_plus_{geometry.name}_{role}_{PRECISION}"
    if str(model.get_friendly_name()) != expected_name:
        raise ValueError(f"edge {role} friendly name changed")
    return {
        "role": role,
        "friendly_name": str(model.get_friendly_name()),
        "inputs": sorted(inputs, key=lambda record: record["name"]),
        "outputs": sorted(outputs, key=lambda record: record["name"]),
        "learned_parameter_names": learned,
        "state_variables": [],
        "sinks": [],
        "conditioning": (
            {"timestep_inputs": ["t", "t_mod"],
             "source_f32_preparation": "fixed_conditioning(..., 'bf16') first two results",
             "performed_by_export": False,
             "one_time_preparation_required": True}
            if role == "prefix" else None
        ),
        "openvino_types": {"bf16": str(ov.Type.bf16), "f32": str(ov.Type.f32)},
        **constants,
    }


def export_edges(reference_root: Path, output: Path) -> dict[str, Any]:
    root = checked_reference_root(reference_root, REQUIRED_REFERENCE_FILES)
    destination = fresh_output(output)
    with reference_import_path(root):
        import openvino as ov
        from frame_geometry import FrameGeometry
        from plus_edges import build_prefix, build_suffix
        from plus_graphs import BlockGeometry
        # plus_edges imports storage_tensor lazily while constructing the
        # external BF16 graphs. Load that exact curated module before the
        # closure check so provenance covers the code actually used below.
        import plus_storage

        runtime_version = str(ov.get_version())
        require_runtime_version(runtime_version, OPENVINO_VERSION)
        require_reference_modules(root, REFERENCE_MODULES)
        frame = FrameGeometry(*VISIBLE_HW)
        if tuple(frame.working_hw) != WORKING_HW:
            raise ValueError("reviewed edge working geometry changed")
        geometry = BlockGeometry(frame)
        geometry.validate()
        models = {
            "prefix": build_prefix(None, geometry, storage=PRECISION),
            "suffix": build_suffix(None, geometry, storage=PRECISION),
        }
        reports = {}
        artifacts = {}
        for role, model in models.items():
            reports[role] = validate_graph_contract(model, geometry, role=role, ov=ov)
            artifacts[role] = serialize_structural_ir(
                ov, model, destination, f"flash_edges_{role}_{PRECISION}"
            )
        source_files = graph_source_records(root, REQUIRED_REFERENCE_FILES)
        return write_structural_manifest(destination, {
            "schema": 1,
            "status": "pass",
            "scope": "structural Flash edge IR; learned BF16 Parameters only",
            "trust": {
                "reference_root": "explicit curated executable source; not sandboxed",
                "source_hashes_are_provenance_only": True,
                "learned_payloads_loaded": False,
            },
            "graph_role": "four-latent_prefix_and_suffix",
            "precision": PRECISION,
            "geometry": {
                "visible_hw": list(VISIBLE_HW),
                "working_hw": list(WORKING_HW),
                "token_hw": [int(geometry.h), int(geometry.w)],
                "latent_hw": [int(geometry.h * 2), int(geometry.w * 2)],
                "frames": int(geometry.f),
                "tokens": int(geometry.tokens),
            },
            "conditioning": {
                "prefix_timestep_inputs": ["t", "t_mod"],
                "source_f32_preparation": "fixed_conditioning(authenticated source-F32 weights, 'bf16')",
                "one_time_preparation_required": True,
                "performed_by_export": False,
            },
            "openvino": {
                "required_version": OPENVINO_VERSION,
                "actual_version": runtime_version,
                "compilation_performed": False,
                "inference_performed": False,
            },
            "exporter_sha256": sha256_file(Path(__file__).resolve()),
            "source_files": source_files,
            "artifacts": artifacts,
            "graphs": reports,
        })


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="curated Flash reference root containing openvino/")
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh output directory for prefix/suffix XML and BIN")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = export_edges(args.reference_root, args.output)
    print(json.dumps({"status": manifest["status"],
                      "artifacts": manifest["artifacts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
