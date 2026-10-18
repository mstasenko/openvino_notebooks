#!/usr/bin/env python3
"""Export the complete or phased Flash TCDecoder as structural IR.

This is a build-time graph export only.  The caller supplies a trusted
reference runtime root and the separately pinned ``tiny_model.py`` source.
The source module is constructed on Torch's meta device, so no checkpoint or
learned tensor payload is loaded.  Every learned graph Parameter is mapped by
the identity of the corresponding source Parameter; shape/value matching is
never used.  The default complete explicit-state graph is bounded to the
decoder's HD geometry.  The opt-in phased bundle is the reviewed
16-tile/12-row-halo UHD building block and keeps learned values external.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from flash_export import (
    canonical_port_record,
    checked_reference_file,
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
    source_record,
    validate_constant_nodes,
    write_structural_manifest,
)


OPENVINO_VERSION = "2026.5.0-23457-7d81aad55a5"
PRECISION = "bf16"
STATE_NAMES = (
    "mem_512_0", "mem_512_1", "mem_512_2",
    "mem_256_0", "mem_256_1", "mem_256_2",
    "mem_128_0", "mem_128_1", "mem_128_2",
)
RESIDUAL_NAMES = tuple(f"residual_linear_{index}" for index in range(12))
REQUIRED_REFERENCE_FILES = (
    "openvino/full_runtime/plus_decoder.py",
    "openvino/full_runtime/plus_storage.py",
    "openvino/decoder/convert_decoder.py",
)
REFERENCE_MODULES = {
    "plus_decoder": "openvino/full_runtime/plus_decoder.py",
    "plus_storage": "openvino/full_runtime/plus_storage.py",
}
PHASED_REFERENCE_MODULES = {
    **REFERENCE_MODULES,
    "plus_phased_decoder": "openvino/full_runtime/plus_phased_decoder.py",
    "frame_geometry": "openvino/full_runtime/frame_geometry.py",
}
PHASED_REFERENCE_FILES = (
    *REQUIRED_REFERENCE_FILES,
    "openvino/full_runtime/plus_phased_decoder.py",
    "openvino/full_runtime/frame_geometry.py",
)
PHASED_SOURCE_HW = (2176, 3840)
PHASED_TILE_COUNT = 16
PHASED_HALO_ROWS = 12
PHASED_DYNAMIC_SOURCE_NAMES = {
    "decoder.15.conv.weight",
    "decoder.21.conv.weight",
}
PHASED_STATIC_BLOCKS = {
    "front": frozenset(range(14)),
    "grow_memory": frozenset((14, 16, 17, 18, 19)),
    "remote_frame_child": frozenset((20, 22, 23, 24, 25, 26)),
}


def _shape(value: Any) -> list[int]:
    try:
        result = [int(dim) for dim in value]
    except Exception as error:
        raise ValueError("decoder shape is not a static integer sequence") from error
    if any(dim <= 0 for dim in result):
        raise ValueError(f"decoder shape is invalid: {result}")
    return result


def _source_file(reference_root: Path, decoder_source: Path | None) -> Path:
    root = Path(reference_root).resolve(strict=True)
    candidate = (root / "source/inference/decoders/tiny_model.py"
                 if decoder_source is None else Path(decoder_source))
    if candidate.name != "tiny_model.py":
        raise ValueError("decoder source must be a regular tiny_model.py file")
    # Validate the exact source file through the same no-symlink reader used
    # for the runtime closure.  A source outside the runtime root is allowed,
    # but it remains an explicit caller-selected executable input.
    return checked_reference_file(candidate.parent, candidate.name)


def _load_meta_decoder(plus_decoder: Any, source_file: Path):
    previous_source = plus_decoder.SOURCE
    try:
        plus_decoder.SOURCE = source_file
        source = plus_decoder._load_source_module()
    finally:
        plus_decoder.SOURCE = previous_source
    import torch
    with torch.device("meta"):
        decoder = source.TAEHV(
            checkpoint_path=None,
            channels=list(plus_decoder.CHANNELS),
            latent_channels=plus_decoder.LATENT_CHANNELS,
        )
    decoder = decoder.to(dtype=torch.float32).eval()
    parameters = list(decoder.named_parameters())
    if len(parameters) != 66 or len({name for name, _ in parameters}) != 66:
        raise ValueError("authenticated decoder topology must expose exactly 66 unique Parameters")
    if any(parameter.dtype != torch.float32 for _, parameter in parameters):
        raise ValueError("decoder topology Parameters must be F32 before storage conversion")
    return decoder, parameters


def _make_parameter_factory(ov: Any, named_parameters: list[tuple[str, Any]]):
    by_identity: dict[int, str] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    canonical_names: dict[str, str] = {}
    for name, parameter in named_parameters:
        if id(parameter) in by_identity:
            raise ValueError("source Parameter identity is shared by two canonical names")
        by_identity[id(parameter)] = name
        shapes[name] = tuple(int(dim) for dim in parameter.shape)
        port_name = "weight__" + name.replace(".", "_")
        previous = canonical_names.setdefault(port_name, name)
        if previous != name:
            raise ValueError(f"canonical decoder Parameter name collision: {previous}, {name}")
    ports: dict[str, Any] = {}
    port_shapes: dict[str, tuple[int, ...]] = {}

    def factory(source: Any, graph_shape: tuple[int, ...]):
        name = by_identity.get(id(source))
        if name is None:
            raise ValueError("decoder graph requested an unregistered learned Parameter")
        expected = shapes[name]
        graph_shape = tuple(int(dim) for dim in graph_shape)
        # Biases are reshaped by the converter only after identity lookup;
        # this is the sole permitted ABI shape adaptation.
        if graph_shape != expected:
            if not (len(expected) == 1 and graph_shape == (1, expected[0], 1, 1)):
                raise ValueError(f"learned Parameter shape changed for {name}")
        if name in ports:
            if port_shapes[name] != graph_shape:
                raise ValueError(f"learned Parameter is reused with a different shape: {name}")
            return ports[name]
        port_name = "weight__" + name.replace(".", "_")
        if not port_name.replace("_", "").isalnum() or not port_name.startswith("weight__"):
            raise ValueError(f"invalid canonical decoder Parameter name: {name}")
        parameter = ov.opset1.parameter(list(graph_shape), ov.Type.bf16, name=port_name)
        parameter.set_friendly_name(port_name)
        parameter.output(0).get_tensor().set_names({port_name})
        ports[name] = parameter.output(0)
        port_shapes[name] = graph_shape
        return ports[name]

    return factory, ports


def _learned_shape(parameter: Any) -> list[int]:
    shape = _shape(parameter.shape)
    return [1, shape[0], 1, 1] if len(shape) == 1 else shape


def _structural_guards(model: Any, role: str) -> dict[str, Any]:
    if any(port.get_node().get_type_name() != "Parameter" for port in model.inputs):
        raise ValueError(f"{role} inputs must be ordinary Parameters")
    if any(port.get_node().get_type_name() != "Result" for port in model.outputs):
        raise ValueError(f"{role} outputs must be ordinary Results")
    if any(node.get_type_name() in {"ReadValue", "Assign"} for node in model.get_ordered_ops()):
        raise ValueError(f"{role} structural export must not contain state operations")
    if list(model.get_variables()) or list(model.get_sinks()):
        raise ValueError(f"{role} structural export must not contain Variables or Sinks")
    reject_f16_intermediates(model, role)
    return validate_constant_nodes(model, forbidden_prefixes=("weight__",))


def _check_contract(model: Any, geometry: dict[str, Any],
                    named_parameters: list[tuple[str, Any]]) -> dict[str, Any]:
    source_names = tuple(name for name, _ in named_parameters)
    learned_contract = {
        "weight__" + name.replace(".", "_"): (_learned_shape(parameter), PRECISION)
        for name, parameter in named_parameters
    }
    if len(learned_contract) != 66:
        raise ValueError("decoder must declare exactly 66 unique learned Parameters")
    data_contract = {
        "packed": ([1, 784, *geometry["latent_hw"]], PRECISION),
        **{name: (list(shape), PRECISION)
           for name, shape in zip(STATE_NAMES, geometry["state_shapes"])},
    }
    input_contract = {**data_contract, **learned_contract}
    inputs = [canonical_port_record(port) for port in model.inputs]
    outputs = [canonical_port_record(port) for port in model.outputs]
    check_port_records(inputs, input_contract, "decoder input")
    state_contract = {
        f"{name}_out": (list(shape), PRECISION)
        for name, shape in zip(STATE_NAMES, geometry["state_shapes"])
    }
    state_shapes = geometry["state_shapes"]
    residual_shapes = (
        *(list(state_shapes[0]) for _ in range(3)),
        *(list(state_shapes[3]) for _ in range(3)),
        *(list(state_shapes[6]) for _ in range(6)),
    )
    output_contract = {
        "frames": ([1, 4, 3, *geometry["source_hw"]], PRECISION),
        **state_contract,
    }
    output_contract.update({
        name: (shape, PRECISION)
        for name, shape in zip(RESIDUAL_NAMES, residual_shapes)
    })
    check_port_records(outputs, output_contract, "decoder public output")
    constants = _structural_guards(model, "decoder")
    return {
        "friendly_name": str(model.get_friendly_name()),
        "inputs": sorted(inputs, key=lambda item: item["name"]),
        "outputs": sorted(outputs, key=lambda item: item["name"]),
        "learned_parameter_count": 66,
        "learned_parameter_names": sorted(source_names),
        "learned_input_names": sorted(learned_contract),
        "private_residual_results": list(RESIDUAL_NAMES),
        "state_variables": [],
        "sinks": [],
        **constants,
    }


def _reviewed_phased_tiles(source_hw: tuple[int, int], history_memory_tiles: Any) -> dict[str, Any]:
    """Return the one explicitly reviewed UHD tile plan, fail-closed."""

    if tuple(source_hw) != PHASED_SOURCE_HW:
        raise ValueError(
            "phased bundle currently supports only the reviewed 2176x3840 source geometry"
        )
    try:
        plan = history_memory_tiles(
            tuple(source_hw), tile_count=PHASED_TILE_COUNT, halo_rows=PHASED_HALO_ROWS
        )
    except (TypeError, ValueError, AssertionError) as error:
        raise ValueError("reviewed phased history tile geometry is unavailable") from error
    if (tuple(plan.get("source_hw", ())) != tuple(source_hw) or
            plan.get("tile_count") != PHASED_TILE_COUNT or
            plan.get("halo_rows") != PHASED_HALO_ROWS or
            len(plan.get("tiles", ())) != PHASED_TILE_COUNT):
        raise ValueError("phased history tile helper returned an unexpected plan")
    tile_source_hw = tuple(int(value) for value in plan.get("tile_source_hw", ()))
    if tile_source_hw != (184, 3840):
        raise ValueError("phased tile_source_hw no longer matches the reviewed geometry")
    required_tile_keys = {
        "name", "input_rows", "output_rows", "frame_rows", "compact_rows",
        "source_frame_rows", "source_output_rows",
    }
    for tile in plan["tiles"]:
        if not required_tile_keys.issubset(tile):
            raise ValueError("phased tile descriptor is missing source/frame coordinates")
        for key in ("input_rows", "output_rows", "frame_rows", "compact_rows",
                    "source_frame_rows", "source_output_rows"):
            pair = tile[key]
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
                    any(type(value) is not int for value in pair) or pair[0] < 0 or
                    pair[1] <= pair[0]):
                raise ValueError(f"phased tile {key} coordinates are invalid")
        if any(value % 2 for value in tile["input_rows"]):
            raise ValueError("phased compact tile boundaries must be even")
    return plan


def _static_bindings(
    named_parameters: list[tuple[str, Any]],
    ports: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Authenticate source shapes and partition the expected phase inputs.

    Phase ownership follows source block boundaries, not the graph's own
    input list: a disconnected or misplaced weight must fail the ABI check.
    """

    by_name = dict(named_parameters)
    expected_static = set(by_name) - PHASED_DYNAMIC_SOURCE_NAMES
    if len(by_name) != 66 or len(expected_static) != 64:
        raise ValueError("authenticated decoder must have 64 static phased bindings")
    if set(ports) != expected_static:
        missing = sorted(expected_static - set(ports))
        extra = sorted(set(ports) - expected_static)
        raise ValueError(f"phased static binding coverage changed: missing={missing}, extra={extra}")
    records: list[dict[str, Any]] = []
    phase_ports: dict[str, dict[str, Any]] = {phase: {} for phase in PHASED_STATIC_BLOCKS}
    for source_name in sorted(expected_static):
        source = by_name[source_name]
        port = ports[source_name]
        record = canonical_port_record(port)
        graph_name = "weight__" + source_name.replace(".", "_")
        source_shape = _shape(source.shape)
        graph_shape = _learned_shape(source)
        if (record != {"name": graph_name, "shape": graph_shape, "element_type": PRECISION} or
                port.get_node().get_type_name() != "Parameter"):
            raise ValueError(f"phased static binding ABI changed for {source_name}")
        fields = source_name.split(".", 2)
        if len(fields) != 3 or fields[0] != "decoder" or not fields[1].isdigit():
            raise ValueError(f"unknown phased source Parameter: {source_name}")
        owners = [phase for phase, blocks in PHASED_STATIC_BLOCKS.items()
                  if int(fields[1]) in blocks]
        if len(owners) != 1:
            raise ValueError(f"source Parameter is outside the static phase boundaries: {source_name}")
        phase_ports[owners[0]][graph_name] = port
        records.append({
            "source_parameter_name": source_name,
            "graph_input_name": graph_name,
            "source_shape": source_shape,
            "graph_shape": graph_shape,
            "source_dtype": "f32",
            "storage": PRECISION,
        })
    return records, phase_ports


def _validate_bank_descriptors(
    phase: str,
    input_name: str,
    banks: list[dict[str, Any]],
    source_by_name: Mapping[str, Any],
    source_name: str,
    source_shape: tuple[int, ...],
    slice_shape: tuple[int, ...],
    width: int,
) -> dict[str, Any]:
    """Authenticate two child slices without materializing their learned data."""

    if not isinstance(banks, list) or len(banks) != 2:
        raise ValueError(f"{phase} must expose exactly two source bank slices")
    source = source_by_name.get(source_name)
    if source is None or tuple(int(value) for value in source.shape) != source_shape:
        raise ValueError(f"{phase} source bank Parameter shape changed")
    expected_block = int(source_name.split(".")[1])
    slices = []
    for child, descriptor in enumerate(banks):
        if not isinstance(descriptor, dict):
            raise ValueError(f"{phase} bank descriptor is not a mapping")
        if descriptor.get("source_parameter") is not source:
            raise ValueError(f"{phase} bank descriptor lost source Parameter identity")
        if (descriptor.get("source_parameter_name") != source_name or
                descriptor.get("source_block") != expected_block or
                descriptor.get("child") != child or
                descriptor.get("source_dtype") != "f32" or
                descriptor.get("storage") != PRECISION):
            raise ValueError(f"{phase} bank descriptor provenance changed")
        start = descriptor.get("start")
        stop = descriptor.get("stop")
        shape = tuple(descriptor.get("shape", ()))
        expected_start = child * width
        if (type(start) is not int or type(stop) is not int or
                (start, stop) != (expected_start, expected_start + width) or
                shape != slice_shape):
            raise ValueError(f"{phase} bank slice offset or shape changed")
        slices.append({
            "child": child,
            "axis": 0,
            "start": start,
            "stop": stop,
            "shape": list(slice_shape),
            "source_dtype": "f32",
            "storage": PRECISION,
        })
    return {
        "phase": phase,
        "input_name": input_name,
        "source_parameter_name": source_name,
        "source_shape": list(source_shape),
        "source_dtype": "f32",
        "storage": PRECISION,
        "slices": slices,
    }


def _check_phase_contract(
    model: Any,
    phase: str,
    full_geometry: dict[str, Any],
    tile_geometry: dict[str, Any],
    static_ports: Mapping[str, Any],
) -> dict[str, Any]:
    """Check one phased graph's ordinary external ABI and structural guards."""

    full_h, full_w = full_geometry["source_hw"]
    tile_h, tile_w = tile_geometry["source_hw"]
    full_states = full_geometry["state_shapes"]
    tile_state = [1, 128, tile_h // 2, tile_w // 2]
    if phase == "front":
        base_names = ["packed", *STATE_NAMES[:6]]
        base_contract = {
            "packed": ([1, 784, full_h // 8, full_w // 8], PRECISION),
            **{name: (list(shape), PRECISION)
               for name, shape in zip(STATE_NAMES[:6], full_states[:6])},
        }
        output_names = ["middle", *[f"{name}_out" for name in STATE_NAMES[:6]],
                        *[f"residual_linear_{index}" for index in range(6)]]
        output_contract = {
            "middle": ([1, 256, full_h // 4, full_w // 4], PRECISION),
            **{f"{name}_out": (list(shape), PRECISION)
               for name, shape in zip(STATE_NAMES[:6], full_states[:6])},
            **{f"residual_linear_{index}": (
                list(full_states[0 if index < 3 else 3]), PRECISION)
                for index in range(6)},
        }
    elif phase == "grow_memory":
        base_names = ["middle_in", "grow_weight", *STATE_NAMES[6:]]
        base_contract = {
            "middle_in": ([1, 256, tile_h // 4, tile_w // 4], PRECISION),
            "grow_weight": ([256, 256, 1, 1], PRECISION),
            **{name: (tile_state, PRECISION) for name in STATE_NAMES[6:]},
        }
        output_names = ["S0", "S1", "S2", "middle",
                        "residual_linear_6", "residual_linear_7", "residual_linear_8"]
        output_contract = {
            name: (tile_state, PRECISION) for name in ("S0", "S1", "S2", "middle")
        }
        output_contract.update({
            f"residual_linear_{index}": (tile_state, PRECISION)
            for index in range(6, 9)
        })
    elif phase == "remote_frame_child":
        base_names = ["middle", "frame_grow"]
        base_contract = {
            "middle": (tile_state, PRECISION),
            "frame_grow": ([128, 128, 1, 1], PRECISION),
        }
        output_names = ["frame"]
        output_contract = {"frame": ([1, 3, tile_h, tile_w], PRECISION)}
    else:
        raise ValueError(f"unknown phased graph: {phase}")

    inputs = [canonical_port_record(port) for port in model.inputs]
    outputs = [canonical_port_record(port) for port in model.outputs]
    static_contracts = {
        name: (list(port.get_shape()), PRECISION) for name, port in static_ports.items()
    }
    contracts = {**base_contract,
                 **static_contracts}
    check_port_records(inputs, contracts, f"phased {phase} input")
    if inputs[:len(base_names)] != [
        {"name": name, "shape": contracts[name][0], "element_type": contracts[name][1]}
        for name in base_names
    ]:
        raise ValueError(f"{phase} base input order changed")
    actual_ports = {port.get_any_name(): port for port in model.inputs}
    if any(actual_ports[name].get_node() != port.get_node()
           for name, port in static_ports.items()):
        raise ValueError(f"{phase} learned input lost its authenticated source binding")
    check_port_records(outputs, output_contract, f"phased {phase} output")
    if [record["name"] for record in outputs] != output_names:
        raise ValueError(f"{phase} output order changed")
    constants = _structural_guards(model, f"phased {phase}")
    return {
        "phase": phase,
        "friendly_name": str(model.get_friendly_name()),
        "inputs": sorted(inputs, key=lambda record: record["name"]),
        "outputs": sorted(outputs, key=lambda record: record["name"]),
        "static_parameter_names": sorted(static_ports),
        "dynamic_inputs": [name for name in base_names if name in {"grow_weight", "frame_grow"}],
        "state_variables": [], "sinks": [],
        **constants,
    }


def _phase_source_records(root: Path) -> list[dict[str, Any]]:
    records = graph_source_records(root, tuple(PHASED_REFERENCE_MODULES.values()))
    converter_path = checked_reference_file(root, "openvino/decoder/convert_decoder.py")
    if not any(record["path"] == "openvino/decoder/convert_decoder.py" for record in records):
        records.append(source_record(root, converter_path))
    return records


def export_phased_bundle(reference_root: Path, output: Path,
                         source_hw: tuple[int, int],
                         decoder_source: Path | None = None) -> dict[str, Any]:
    """Export the three external-weight graphs used by the reviewed UHD path."""

    root = checked_reference_root(reference_root, PHASED_REFERENCE_FILES)
    source_file = _source_file(root, decoder_source)
    destination = fresh_output(output)
    with reference_import_path(root):
        import openvino as ov
        import frame_geometry
        import plus_decoder
        import plus_phased_decoder

        require_runtime_version(str(ov.get_version()), OPENVINO_VERSION)
        require_reference_modules(root, PHASED_REFERENCE_MODULES)
        if sha256_file(source_file) != str(plus_decoder.SOURCE_SHA256):
            raise ValueError("decoder source fingerprint does not match the pinned source")
        tile_plan = _reviewed_phased_tiles(tuple(source_hw), frame_geometry.history_memory_tiles)
        full_geometry = plus_phased_decoder.phased_geometry(tuple(source_hw))
        tile_source_hw = tuple(int(value) for value in tile_plan["tile_source_hw"])
        tile_geometry = plus_phased_decoder.phased_geometry(tile_source_hw)
        decoder, named_parameters = _load_meta_decoder(plus_decoder, source_file)
        factory, learned_ports = _make_parameter_factory(ov, named_parameters)
        front, front_metadata = plus_phased_decoder.build_front(
            decoder, tuple(source_hw), learned_parameter_factory=factory
        )
        growth, growth_banks, growth_metadata = plus_phased_decoder.build_growth_history(
            decoder, tile_source_hw, learned_parameter_factory=factory
        )
        frame, frame_banks, frame_metadata = plus_phased_decoder.build_remote_frame_child(
            decoder, tile_source_hw, learned_parameter_factory=factory
        )
        static_records, phase_ports = _static_bindings(named_parameters, learned_ports)
        source_by_name = dict(named_parameters)
        growth_bank_record = _validate_bank_descriptors(
            "grow_memory", "grow_weight", growth_banks, source_by_name,
            "decoder.15.conv.weight", (512, 256, 1, 1), (256, 256, 1, 1), 256,
        )
        frame_bank_record = _validate_bank_descriptors(
            "remote_frame_child", "frame_grow", frame_banks, source_by_name,
            "decoder.21.conv.weight", (256, 128, 1, 1), (128, 128, 1, 1), 128,
        )
        models = {
            "front": (front, front_metadata, full_geometry),
            "grow_memory": (growth, growth_metadata, tile_geometry),
            "remote_frame_child": (frame, frame_metadata, tile_geometry),
        }
        reports: dict[str, Any] = {}
        artifacts: dict[str, Any] = {}
        for phase, (model, metadata, geometry) in models.items():
            reports[phase] = {
                "metadata": metadata,
                "contract": _check_phase_contract(
                    model, phase, full_geometry, geometry, phase_ports[phase]
                ),
            }
            artifacts[phase] = serialize_structural_ir(
                ov, model, destination, f"flash_decoder_phased_{phase}_bf16"
            )
        source_records = _phase_source_records(root)
        source_records.append({
            "path": "decoder-source/tiny_model.py",
            "bytes": source_file.stat().st_size,
            "sha256": sha256_file(source_file),
        })
        manifest = {
            "schema": 1,
            "status": "pass",
            "bundle": "phased",
            "scope": "three-graph phased Flash TCDecoder bundle; reviewed UHD tile geometry",
            "trust": {
                "reference_root": "explicit curated executable source; not sandboxed",
                "source_hashes_are_provenance_only": True,
                "learned_payloads_loaded": False,
                "topology_source": "authenticated tiny_model.py constructed on Torch meta device",
            },
            "geometry": {
                "source_hw": list(source_hw),
                "tile_source_hw": list(tile_source_hw),
                "tile_count": PHASED_TILE_COUNT,
                "halo_rows": PHASED_HALO_ROWS,
                "tile_plan": tile_plan,
                "full_state_shapes": [list(shape) for shape in full_geometry["state_shapes"]],
                "tile_state_shapes": [list(shape) for shape in tile_geometry["state_shapes"]],
            },
            "precision": PRECISION,
            "openvino": {
                "required_version": OPENVINO_VERSION,
                "actual_version": str(ov.get_version()),
                "compilation_performed": False,
                "inference_performed": False,
                "device_queries_performed": False,
            },
            "decoder_source": {
                "path": "decoder-source/tiny_model.py",
                "bytes": source_file.stat().st_size,
                "sha256": sha256_file(source_file),
                "expected_sha256": str(plus_decoder.SOURCE_SHA256),
            },
            "learned_storage": {
                "static_parameter_count": len(static_records),
                "static_bindings": static_records,
                "source_bank_descriptor_count": 2,
                "source_bank_slice_count": 4,
                "source_bank_descriptors": [growth_bank_record, frame_bank_record],
                "mapping": "source Parameter identity to external BF16 graph Parameters; child banks are source F32 slices",
                "materialized_learned_payloads": False,
            },
            "exporter_sha256": sha256_file(Path(__file__).resolve()),
            "source_files": source_records,
            "artifacts": artifacts,
            "graphs": reports,
        }
        return write_structural_manifest(destination, manifest)


def export_decoder(reference_root: Path, output: Path, source_hw: tuple[int, int],
                   decoder_source: Path | None = None) -> dict[str, Any]:
    root = checked_reference_root(reference_root, REQUIRED_REFERENCE_FILES)
    source_file = _source_file(root, decoder_source)
    destination = fresh_output(output)
    with reference_import_path(root):
        import openvino as ov
        import plus_decoder

        require_runtime_version(str(ov.get_version()), OPENVINO_VERSION)
        require_reference_modules(root, REFERENCE_MODULES)
        if sha256_file(source_file) != str(plus_decoder.SOURCE_SHA256):
            raise ValueError("decoder source fingerprint does not match the pinned source")
        native_converter = plus_decoder._load_native_converter()
        geometry = native_converter.decoder_geometry(tuple(source_hw))
        decoder, named_parameters = _load_meta_decoder(plus_decoder, source_file)
        factory, learned_ports = _make_parameter_factory(ov, named_parameters)
        model = plus_decoder.build_decoder_graph(
            decoder,
            source_hw=geometry["source_hw"],
            precision=PRECISION,
            combined_mem=True,
            learned_parameter_factory=factory,
        )
        if set(learned_ports) != {name for name, _ in named_parameters}:
            raise ValueError("decoder must use every authenticated source Parameter")
        # Match phased state Results and keep input/output names unambiguous
        # after portable IR serialization. This changes no source arithmetic.
        for name in STATE_NAMES:
            model.output(name).get_tensor().set_names({f"{name}_out"})
        graph = _check_contract(model, geometry, named_parameters)
        artifacts = serialize_structural_ir(ov, model, destination, "flash_decoder_complete_bf16")
        # The converter is loaded through spec_from_file_location and is
        # intentionally not inserted into sys.modules; record that exact
        # required file explicitly alongside the loaded runtime modules.
        source_records = graph_source_records(root, tuple(REFERENCE_MODULES.values()))
        converter_path = checked_reference_file(root, "openvino/decoder/convert_decoder.py")
        if not any(record["path"] == "openvino/decoder/convert_decoder.py"
                   for record in source_records):
            source_records.append(source_record(root, converter_path))
        source_records.append({
            "path": "decoder-source/tiny_model.py",
            "bytes": source_file.stat().st_size,
            "sha256": sha256_file(source_file),
        })
        manifest = {
            "schema": 1,
            "status": "pass",
            "scope": "complete explicit-state decoder structural IR; HD-bounded building block",
            "trust": {
                "reference_root": "explicit curated executable source; not sandboxed",
                "source_hashes_are_provenance_only": True,
                "learned_payloads_loaded": False,
                "topology_source": "authenticated tiny_model.py constructed on Torch meta device",
            },
            "geometry": {
                "source_hw": list(geometry["source_hw"]),
                "latent_hw": list(geometry["latent_hw"]),
                "state_shapes": [list(shape) for shape in geometry["state_shapes"]],
                "complete_graph_bound": "decoder_geometry HD bound",
                "uhd_exported": False,
                "uhd_requirement": "phased decoder bundle is required for UHD history execution",
            },
            "precision": PRECISION,
            "openvino": {
                "required_version": OPENVINO_VERSION,
                "actual_version": str(ov.get_version()),
                "compilation_performed": False,
                "inference_performed": False,
                "device_queries_performed": False,
            },
            "decoder_source": {
                "path": "decoder-source/tiny_model.py",
                "bytes": source_file.stat().st_size,
                "sha256": sha256_file(source_file),
                "expected_sha256": str(plus_decoder.SOURCE_SHA256),
            },
            "learned_storage": {
                "parameter_count": 66,
                "mapping": "source Parameter identity to one BF16 graph Parameter",
                "bias_reshape": "source one-dimensional bias identity, graph [1,C,1,1]",
                "combined_mem": True,
            },
            "exporter_sha256": sha256_file(Path(__file__).resolve()),
            "source_files": source_records,
            "artifacts": artifacts,
            "graph": graph,
        }
        return write_structural_manifest(destination, manifest)


def _parse_hw(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", 1)
        parsed = (int(height), int(width))
        if min(parsed) <= 0:
            raise ValueError("geometry dimensions must be positive")
        return parsed
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("geometry must be WIDTHxHEIGHT") from error


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", choices=("complete", "phased"), default="complete",
                        help="export the unchanged complete graph or the opt-in phased bundle")
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="curated root containing openvino/ runtime sources")
    parser.add_argument("--decoder-source", type=Path, required=True,
                        help="authenticated tiny_model.py; no checkpoint is loaded")
    parser.add_argument("--source-size", required=True,
                        help="explicit decoder source raster WIDTHxHEIGHT")
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh output directory for structural XML/BIN/manifest")
    args = parser.parse_args(list(argv) if argv is not None else None)
    source_hw = _parse_hw(args.source_size)
    if args.bundle == "phased":
        manifest = export_phased_bundle(
            args.reference_root, args.output, source_hw, args.decoder_source
        )
    else:
        manifest = export_decoder(
            args.reference_root, args.output, source_hw, args.decoder_source
        )
    print(json.dumps({"status": manifest["status"],
                      "artifacts": manifest["artifacts"],
                      "bundle": manifest.get("bundle", "complete"),
                      "learned_parameter_count": manifest["learned_storage"].get(
                          "parameter_count",
                          manifest["learned_storage"].get("static_parameter_count"),
                      )},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
