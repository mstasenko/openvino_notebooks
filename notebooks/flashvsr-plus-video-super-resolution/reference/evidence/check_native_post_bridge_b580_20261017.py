#!/usr/bin/env python3
"""Authenticated finite POST qualification worker for one B580 graph.

This is deliberately a single-graph, three-layer component check.  It does
not run a trajectory or a decoder.  The native CPU fixture is authenticated
before OpenVINO Core is created; ``--run`` is the only path that can create a
Core or compile the selected graph.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from reference_paths import EVIDENCE_ROOT

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve()
BASE = HERE.parents[1]
EVIDENCE = BASE / "evidence"
PINNED_RUNTIME = BASE / "runtime-openvino-2026.5.0.dev20261014"
sys.path[:0] = [str(BASE), str(PINNED_RUNTIME), str(BASE / "openvino"),
                str(BASE / "openvino" / "block"),
                str(BASE / "openvino" / "chunked_attention"),
                str(BASE / "openvino" / "full_runtime"), str(EVIDENCE)]

import openvino as ov  # noqa: E402
import check_bf16_cross_bridge_b580_20261017 as bridge_runtime  # noqa: E402
import check_native_cross_bridge_b580_20261017 as native_reference  # noqa: E402
from plus_graphs import build_post_rows  # noqa: E402
from plus_resident import ResidentGraph  # noqa: E402
from plus_weights import (CHECKPOINT, LAYER_SHAPES, ROLE_KEYS, authenticate_dit,
                          load_layer)  # noqa: E402


DEVICE = "GPU.1"
BDF = "0000:03:00.0"
LAYERS = (0, 23, 29)
VARIANTS = ("legacy-default", "legacy-dynamic", "bridge-dynamic",
            "bridge-ffn2-dynamic", "bridge-precise-dynamic",
            "bridge-precise-ffn2-dynamic")
BRIDGE_VARIANTS = ("bridge-dynamic", "bridge-ffn2-dynamic",
                   "bridge-precise-dynamic", "bridge-precise-ffn2-dynamic")
PRECISE_VARIANTS = ("bridge-precise-dynamic", "bridge-precise-ffn2-dynamic")
FFN2_VARIANTS = ("bridge-ffn2-dynamic", "bridge-precise-ffn2-dynamic")
FFN2_STATE_VARIANT = "bridge-precise-ffn2-dynamic"
FFN2_PRIVATE_OUTPUTS = ("ffn2_linear_storage", "ffn2_gate_storage")
FFN2_STATE_IDS = tuple("plus_post_ffn2_state_" + name
                       for name in FFN2_PRIVATE_OUTPUTS)
FFN2_STATE_BYTES = 25_165_824
FFN2_STATE_CAPTURE_PHASES = ("A1", "B", "A2")
DEFAULT_SPARSE_TYPE = "OtcPlusSparseXmx"
PRECISE_SPARSE_TYPE = "OtcCrossPreciseXmx"
REFERENCE_DEFAULT = EVIDENCE_ROOT / "native-cross-cpu-20261017-attempt1"
SOURCE_ROWS = 2048
ROWS = 4096
CHANNELS = 1536
KV_ROWS = 512
Q_SHAPE = (1, SOURCE_ROWS, CHANNELS)
ROW_SHAPE = (1, ROWS, CHANNELS)
KV_SHAPE = (1, KV_ROWS, CHANNELS)
CONTEXT_SLOT_SHAPE = (12, 16, 128, 128)
NORM_LIMIT = 0.015625
REL_LIMIT = 0.005
WARMUPS = 50
SAMPLES = 100
DEVICE_RESERVE_BYTES = 2 << 30
WORKSPACE_ALLOWANCE_BYTES = 2 << 30
HOST_RSS_LIMIT_BYTES = 4 << 30
BRIDGE_BYTES = 94_374_912
REFERENCE_SOURCE_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
REQUIRED_FIELDS = (
    "feed_x_bits", "feed_self_context_bits", "feed_context_bits",
    "feed_t_mod_bits", "feed_gate_msa_bits", "feed_shift_mlp_bits",
    "feed_scale_mlp_bits", "feed_gate_mlp_bits", "native_cross_k_bits",
    "native_cross_v_bits", "native_x_out_bits",
)
GATE_FIELDS = ("feed_gate_msa_bits", "feed_shift_mlp_bits",
               "feed_scale_mlp_bits", "feed_gate_mlp_bits")


def _validate_ffn2_state_option(variant: str, enabled: bool) -> bool:
    if type(enabled) is not bool:
        raise TypeError("ffn2_state must be a bool")
    if enabled and variant != FFN2_STATE_VARIANT:
        raise ValueError(
            "--ffn2-state is allowed only for bridge-precise-ffn2-dynamic")
    return enabled


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    value = np.asarray(value)
    if not value.flags.c_contiguous:
        raise ValueError("evidence arrays must be contiguous")
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _finite_bits(value: np.ndarray) -> bool:
    value = np.asarray(value)
    return (value.dtype == np.uint16 and value.flags.c_contiguous
            and not np.any((value & np.uint16(0x7F80)) == np.uint16(0x7F80)))


def _reference_receipt(root: Path, expected_sha256: str) -> tuple[Path, dict]:
    # Keep the already-qualified receipt policy in one place.  This worker adds
    # the exact source commit because it is the boundary for the POST fixture.
    receipt_path, receipt = native_reference._reference_receipt(
        root, expected_sha256)
    if receipt.get("source_commit") != REFERENCE_SOURCE_COMMIT:
        raise ValueError("native reference source commit changed")
    return receipt_path, receipt


def _field_shape(name: str) -> tuple[int, ...]:
    if name in ("feed_x_bits", "feed_self_context_bits", "native_x_out_bits"):
        return Q_SHAPE
    if name in ("feed_context_bits", "native_cross_k_bits", "native_cross_v_bits"):
        return KV_SHAPE
    if name == "feed_t_mod_bits":
        return (1, 6, CHANNELS)
    if name in GATE_FIELDS:
        return (1, 1, CHANNELS)
    raise KeyError(name)


def _load_reference_layer(root: Path, record: dict) -> tuple[dict, dict]:
    layer = int(record["layer"])
    archive = record.get("archive", {})
    path = Path(archive.get("path", "")).resolve()
    expected = (root.resolve() / f"layer{layer}-native-cross.npz").resolve()
    if path != expected or not path.is_file():
        raise ValueError(f"layer {layer} archive path is not the authenticated leaf")
    if path.stat().st_size != int(archive.get("bytes", -1)):
        raise ValueError(f"layer {layer} archive byte count changed")
    if _sha_file(path) != archive.get("sha256"):
        raise ValueError(f"layer {layer} archive SHA-256 changed")
    metadata = record.get("arrays", {})
    values, fields = {}, {}
    with np.load(path, allow_pickle=False) as loaded:
        for name in REQUIRED_FIELDS:
            spec = metadata.get(name)
            if not isinstance(spec, dict) or name not in loaded.files:
                raise ValueError(f"layer {layer} missing required field {name}")
            value = np.ascontiguousarray(loaded[name])
            expected_shape = _field_shape(name)
            if (value.dtype != np.uint16 or value.shape != expected_shape
                    or spec.get("dtype") != "uint16"
                    or tuple(spec.get("shape", ())) != expected_shape
                    or _sha_array(value) != spec.get("sha256")
                    or not _finite_bits(value)):
                raise ValueError(f"layer {layer} field authentication failed: {name}")
            values[name] = value
            fields[name] = {"shape": list(value.shape), "dtype": "uint16",
                            "sha256": spec["sha256"]}
    return values, fields


def _authenticate_reference(root: Path, receipt: dict) -> tuple[dict, dict]:
    layers, metadata = {}, {}
    for record in receipt["layers"]:
        layer = int(record["layer"])
        layers[layer], metadata[layer] = _load_reference_layer(root, record)
    return layers, metadata


def _repeat_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[1:] != Q_SHAPE[1:] or value.shape[1] != SOURCE_ROWS:
        raise ValueError("native source row geometry changed")
    return np.ascontiguousarray(np.concatenate((value, value), axis=1))


def pack_context(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert build_post_rows' concat(axis=1)->transpose->reshape wiring."""
    value = np.asarray(value)
    if value.dtype != np.uint16 or value.shape != ROW_SHAPE or not value.flags.c_contiguous:
        raise ValueError("POST context must be contiguous 4096-row BF16 bits")
    ordered = value.reshape(32, 128, 12, 128)
    slots = np.ascontiguousarray(ordered.transpose(2, 0, 1, 3))
    return np.ascontiguousarray(slots[:, :16]), np.ascontiguousarray(slots[:, 16:])


def _unpack_context(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    joined = np.concatenate((first, second), axis=1)
    return np.ascontiguousarray(joined.transpose(1, 2, 0, 3).reshape(ROW_SHAPE))


def _post_model(variant: str, *, ffn2_state: bool = False):
    _validate_ffn2_state_option(variant, ffn2_state)
    if variant not in VARIANTS:
        raise ValueError(f"unknown POST variant: {variant}")
    if variant in BRIDGE_VARIANTS:
        from post_cross_bridge_20261017 import build_post_cross_bridge
        model = build_post_cross_bridge(ffn2_matmul=variant in FFN2_VARIANTS)
        if variant in PRECISE_VARIANTS:
            from cross_precise_20261017 import rewrite_clone
            model = rewrite_clone(model, s1_count=2)
            if ffn2_state:
                # This is the reviewed two-Assign clone; it removes only the
                # two private Result sinks and leaves the public x_out ABI
                # and all arithmetic nodes unchanged.
                from post_ffn2_state_20261017 import replace_ffn2_results_with_assigns
                model, _ = replace_ffn2_results_with_assigns(model)
            return model
        return model
    return build_post_rows(ROWS, precision="bf16", remote_context=True,
                           ffn2_matmul=False)


def _validate_post_model(model, *, ffn2_matmul=False) -> dict:
    if type(ffn2_matmul) is not bool:
        raise TypeError("ffn2_matmul must be a bool")
    inputs = {p.get_any_name(): p for p in model.inputs}
    outputs = {p.get_any_name(): p for p in model.outputs}
    expected_data = {
        "x": ROW_SHAPE,
        "self_context_0": CONTEXT_SLOT_SHAPE,
        "self_context_1": CONTEXT_SLOT_SHAPE,
        "cross_k": KV_SHAPE, "cross_v": KV_SHAPE,
        "gate_msa": (1, 1, CHANNELS), "shift_mlp": (1, 1, CHANNELS),
        "scale_mlp": (1, 1, CHANNELS), "gate_mlp": (1, 1, CHANNELS),
    }
    if set(name for name in inputs if not name.startswith("weight__")) != set(expected_data):
        raise ValueError("POST data input ABI changed")
    for name, shape in expected_data.items():
        if tuple(inputs[name].get_shape()) != shape or inputs[name].get_element_type() != ov.Type.bf16:
            raise ValueError(f"POST input ABI changed: {name}")
    expected_weights = {"weight__" + key.replace(".", "_") for key in ROLE_KEYS["post"]}
    actual_weights = {name for name in inputs if name.startswith("weight__")}
    if actual_weights != expected_weights:
        raise ValueError("POST learned input ABI changed")
    expected_outputs = {"x_out"}
    if ffn2_matmul:
        expected_outputs.update({"ffn2_linear_storage", "ffn2_gate_storage"})
    if set(outputs) != expected_outputs:
        raise ValueError("POST output ABI changed")
    for name, output in outputs.items():
        if (tuple(output.get_shape()) != ROW_SHAPE
                or output.get_element_type() != ov.Type.bf16):
            raise ValueError(f"POST {name} storage changed")
    return {"inputs": {name: {"shape": list(port.get_shape()),
                               "element_type": str(port.get_element_type())}
                        for name, port in inputs.items()},
            "outputs": {name: {"shape": list(port.get_shape()),
                                "element_type": str(port.get_element_type())}
                         for name, port in outputs.items()}}


def _validate_ffn2_state_model(model) -> dict:
    """Validate the opt-in Assign graph without weakening the normal ABI."""
    abi = _validate_post_model(model, ffn2_matmul=False)
    from post_ffn2_state_20261017 import _validate_assign_states
    _validate_assign_states(model)
    if tuple(port.get_any_name() for port in model.outputs) != ("x_out",):
        raise ValueError("FFN2 state graph must expose only public x_out")
    if (len(model.get_variables()) != 2 or len(model.get_sinks()) != 2
            or any(node.get_type_name() == "ReadValue"
                   for node in model.get_ordered_ops())):
        raise ValueError("FFN2 state graph must have exactly two state sinks")
    abi["states"] = [
        {"state_id": state_id, "source_output": output_name,
         "shape": list(ROW_SHAPE), "element_type": "bf16",
         "assign_type": "Assign", "read_value_feedback": False}
        for state_id, output_name in zip(FFN2_STATE_IDS, FFN2_PRIVATE_OUTPUTS)
    ]
    return abi


def _port_bytes(port) -> int:
    shape = tuple(int(v) for v in port.get_shape())
    return int(port.get_element_type().size) * int(np.prod(shape, dtype=np.int64))


def _inventory(model, variant: str, *, ffn2_state: bool = False) -> dict:
    _validate_ffn2_state_option(variant, ffn2_state)
    inputs = {p.get_any_name(): p for p in model.inputs}
    data = {name: port for name, port in inputs.items() if not name.startswith("weight__")}
    data_bytes = sum(_port_bytes(port) for port in data.values())
    alternate_names = {"x", "self_context_0", "self_context_1"}
    alternate_bytes = sum(_port_bytes(data[name]) for name in alternate_names)
    fixed_bytes = data_bytes - alternate_bytes
    compiled_output_bytes = sum(_port_bytes(port) for port in model.outputs)
    # Assign-sink graphs no longer expose the two private Results as model
    # outputs.  Keep charging those original roots and the original request
    # allowance: no alias or dropped-buffer credit is valid until a separate
    # device proof establishes it.
    retained_private_result_bytes = (
        FFN2_STATE_BYTES if ffn2_state else 0)
    output_bytes = compiled_output_bytes + retained_private_result_bytes
    weights_f32 = sum(int(np.prod(shape, dtype=np.int64)) * 4
                      for key, shape in LAYER_SHAPES.items()
                      if key in ROLE_KEYS["post"])
    weights_bf16 = weights_f32 // 2
    request_allowance = data_bytes + output_bytes
    bridge_allowance = 2 * BRIDGE_BYTES if variant in BRIDGE_VARIANTS else 0
    state_bytes = FFN2_STATE_BYTES if ffn2_state else 0
    known = (fixed_bytes + 2 * alternate_bytes + output_bytes + weights_bf16
             + request_allowance + bridge_allowance + state_bytes)
    return {
        "data_port_bytes": int(data_bytes),
        "fixed_bf16_root_bytes": int(fixed_bytes),
        "alternate_ab_bf16_root_bytes": int(2 * alternate_bytes),
        "output_root_bytes": int(output_bytes),
        "compiled_output_port_bytes": int(compiled_output_bytes),
        "retained_private_result_root_bytes": int(retained_private_result_bytes),
        "post_weight_source_f32_bytes": int(weights_f32),
        "post_weight_device_bf16_bytes": int(weights_bf16),
        "request_allowance_bytes": int(request_allowance),
        "bridge_workspace_allowance_bytes": int(bridge_allowance),
        "ffn2_state_bytes": int(state_bytes),
        "ffn2_state_alias_credit_bytes": 0,
        "unknown_plugin_workspace_bytes": WORKSPACE_ALLOWANCE_BYTES,
        "device_reserve_bytes": DEVICE_RESERVE_BYTES,
        "known_plus_workspace_reserve_bytes": int(known + WORKSPACE_ALLOWANCE_BYTES
                                                     + DEVICE_RESERVE_BYTES),
        "alias_credit_bytes": 0,
        "scope": ("one compiled POST graph, fixed roots plus alternate A/B roots; "
                  "original private Results and request allowance retained; "
                  "no alias credit"),
    }


def _host_inventory(reference_layers: dict, model, *, ffn2_state: bool = False) -> dict:
    """Bound host owners while raw references, one layer, and outputs coexist."""
    if type(ffn2_state) is not bool:
        raise TypeError("ffn2_state must be a bool")
    reference_bytes = sum(int(value.nbytes)
                         for layer in reference_layers.values()
                         for value in layer.values())
    # load_layer briefly owns every source tensor, even though POST uploads
    # only its own subset after the unused source owners are released.
    layer_f32 = sum(int(np.prod(shape, dtype=np.int64)) * 4
                    for shape in LAYER_SHAPES.values())
    layer_bf16 = sum(int(np.prod(LAYER_SHAPES[key], dtype=np.int64)) * 2
                     for key in ROLE_KEYS["post"])
    data_bytes = sum(_port_bytes(port) for port in model.inputs
                     if not port.get_any_name().startswith("weight__"))
    output_bytes = sum(_port_bytes(port) for port in model.outputs)
    # Keep all three A/B/A outputs for the final archive and charge one full
    # host staging copy for the current request.  No alias credit is assumed.
    output_archive_bytes = len(LAYERS) * 3 * output_bytes
    state_capture_bytes = (FFN2_STATE_BYTES * len(FFN2_STATE_CAPTURE_PHASES)
                           * len(LAYERS) if ffn2_state else 0)
    required = (reference_bytes + layer_f32 + layer_bf16 + data_bytes
                + output_archive_bytes + state_capture_bytes)
    return {
        "authenticated_reference_raw_bytes": int(reference_bytes),
        "one_layer_source_f32_bytes": int(layer_f32),
        "one_layer_staged_bf16_bytes": int(layer_bf16),
        "one_request_host_staging_bytes": int(data_bytes),
        "raw_output_archive_bytes": int(output_archive_bytes),
        "ffn2_state_capture_bytes": int(state_capture_bytes),
        "ffn2_state_capture_phases": list(FFN2_STATE_CAPTURE_PHASES),
        "alias_credit_bytes": 0,
        "required_host_bytes": int(required),
        "host_rss_limit_bytes": HOST_RSS_LIMIT_BYTES,
        "scope": ("three authenticated references + one layer source/staging + "
                  "A/B/A outputs; optional 2-state A/B/A archive"),
    }


def _xml_port_specs(layer: ET.Element, section_name: str) -> list[dict]:
    section = layer.find(section_name)
    if section is None:
        raise ValueError(f"serialized CustomLayer is missing {section_name} ports")
    aliases = {"F32": "FP32", "F16": "FP16", "BFLOAT16": "BF16"}
    specs = []
    for port in sorted(section.findall("port"),
                       key=lambda item: int(item.attrib["id"])):
        precision = port.attrib.get("precision")
        dims = tuple(int(dim.text) for dim in port.findall("dim"))
        if precision is None or not dims or any(value <= 0 for value in dims):
            raise ValueError("serialized transport port is missing a static ABI")
        specs.append({"id": int(port.attrib["id"]),
                      "precision": aliases.get(precision.upper(), precision.upper()),
                      "shape": dims})
    return specs


def _transport_primitive(value: str) -> str:
    value = value.lower()
    for suffix in ("__f16", "__f32"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    return "".join(char for char in value if char.isalnum())


def _precision_pairs(layer: ET.Element) -> list[tuple[str, str]]:
    inputs = _xml_port_specs(layer, "input")
    outputs = _xml_port_specs(layer, "output")
    if len(inputs) != len(outputs):
        raise ValueError("serialized precision node changed port count")
    return [(source["precision"], target["precision"])
            for source, target in zip(inputs, outputs)]


def _require_raw_source(source, expected, layers, root):
    """Raw BF16 may cross format-only reorders, never numeric operations."""
    seen = set()
    while source.attrib.get("type") in {"Convert", "Reorder"}:
        node_id = source.attrib.get("id")
        if node_id in seen or _precision_pairs(source) != [("FP16", "FP16")]:
            raise ValueError("raw transport route contains a cycle or numeric conversion")
        seen.add(node_id)
        edges = [edge for edge in root.findall("./edges/edge")
                 if edge.attrib.get("to-layer") == node_id]
        if len(edges) != 1 or edges[0].attrib.get("to-port") != "0":
            raise ValueError("raw transport reorder must have one input")
        source = layers.get(edges[0].attrib.get("from-layer"))
        if source is None:
            raise ValueError("raw transport reorder source missing")
    data = source.find("data")
    primitive = "" if data is None else data.attrib.get("primitiveType", "")
    if (source.attrib.get("type") != "CustomGPUPrimitive"
            or _transport_primitive(primitive) != expected):
        raise ValueError(f"raw transport must originate at {expected}")
    return source.attrib["id"]


def _serialized_post_transport_abi(path: Path, *, ids_proven: bool,
                                   sparse_type: str = DEFAULT_SPARSE_TYPE) -> dict:
    """Validate all bridged POST transport ops and their direct runtime edges.

    The pinned plugin may fold compile-time block IDs out of the runtime S1
    port list.  That three-port form is accepted only with the precompile graph
    proof; a four-port form must retain the exact I32 constant ABI.
    """
    if type(sparse_type) is not str or not sparse_type:
        raise TypeError("sparse_type must be a non-empty string")
    if sparse_type not in {DEFAULT_SPARSE_TYPE, PRECISE_SPARSE_TYPE}:
        raise ValueError("unsupported POST sparse primitive")
    sparse_name = _transport_primitive(sparse_type)
    root = ET.parse(path).getroot()
    layers = {layer.attrib.get("id"): layer
              for layer in root.findall("./layers/layer")}
    custom: dict[str, list[ET.Element]] = {}
    for layer in layers.values():
        if layer.attrib.get("type") != "CustomGPUPrimitive":
            continue
        data = layer.find("data")
        primitive = "" if data is None else data.attrib.get("primitiveType", "")
        custom.setdefault(_transport_primitive(primitive), []).append(layer)
    expected_names = {"otccrossqpack", "otccrosskvpack",
                      "otccrosscontextunpack", sparse_name}
    if set(custom) != expected_names:
        raise ValueError(f"full POST transport primitive set changed: {sorted(custom)}")
    if len(custom[sparse_name]) != 2:
        raise ValueError("full POST requires exactly two sparse attention primitives")
    expected = {
        "otccrossqpack": ([ ("FP32", (1, 1, 2048, 1536)) ],
                           [("FP16", (12, 16, 128, 128))], 2),
        "otccrosskvpack": ([ ("FP32", (1, 1, 512, 1536)) ],
                           [("FP16", (4, 128, 12, 128))], (2, 4)),
        "otccrosscontextunpack": ([ ("FP16", (12, 16, 128, 128)) ],
                                   [("FP32", (1, 1, 2048, 1536))], 2),
    }
    if any(len(custom.get(name, ())) not in
           (spec[2] if isinstance(spec[2], tuple) else (spec[2],))
           for name, spec in expected.items()):
        raise ValueError("full POST transport primitive count changed")
    result = {"custom_counts": {name: len(custom[name]) for name in sorted(custom)},
              "transport_ports": {}, "folded_block_id_ports": 0,
              "no_numeric_f16_transport": True}

    def check(name: str, layer: ET.Element, input_specs, output_specs):
        inputs = _xml_port_specs(layer, "input")
        outputs = _xml_port_specs(layer, "output")
        actual_inputs = [(item["precision"], item["shape"]) for item in inputs]
        actual_outputs = [(item["precision"], item["shape"]) for item in outputs]
        # The runtime graph can expose a second identical output port for the
        # second consumer of a CSE-shared pack. The source op still has one output.
        shared_kv_fanout = (name == "otccrosskvpack" and len(custom[name]) == 2
                            and actual_outputs == output_specs * 2)
        if (actual_inputs != input_specs
                or (actual_outputs != output_specs and not shared_kv_fanout)):
            raise ValueError(f"serialized {name} transport ABI changed")
        edges = [edge for edge in root.findall("./edges/edge")
                 if edge.attrib.get("to-layer") == layer.attrib.get("id")]
        if len(edges) != len(input_specs):
            raise ValueError(f"serialized {name} edge count changed")
        direct = []
        for position, edge in enumerate(
                sorted(edges, key=lambda item: int(item.attrib["to-port"]))):
            if int(edge.attrib.get("to-port", "-1")) != position:
                raise ValueError(f"serialized {name} input port order changed")
            source = layers.get(edge.attrib.get("from-layer"))
            if source is None:
                raise ValueError(f"serialized {name} edge source missing")
            source_specs = _xml_port_specs(source, "output")
            source_port = int(edge.attrib.get("from-port", "0"))
            matches = [item for item in source_specs if item["id"] == source_port]
            if len(matches) != 1:
                raise ValueError(f"serialized {name} edge output port changed")
            expected_precision, expected_shape = input_specs[position]
            if (matches[0]["precision"] != expected_precision
                    or matches[0]["shape"] != expected_shape):
                raise ValueError(f"serialized {name} direct source ABI changed")
            raw_source_id = None
            if expected_precision == "FP16":
                expected_source = (sparse_name if name == "otccrosscontextunpack"
                                   else "otccrossqpack" if position == 0
                                   else "otccrosskvpack")
                raw_source_id = _require_raw_source(
                    source, expected_source, layers, root)
            direct.append({"precision": matches[0]["precision"],
                           "shape": list(matches[0]["shape"]),
                           "raw_source_id": raw_source_id})
        return {"input_ports": inputs, "output_ports": outputs,
                "direct_input_ports": direct}

    for name, (input_specs, output_specs, _) in expected.items():
        result["transport_ports"][name] = [
            check(name, layer, input_specs, output_specs)
            for layer in custom[name]]

    sparse_base = [("FP16", (12, 16, 128, 128)),
                   ("FP16", (4, 128, 12, 128)),
                   ("FP16", (4, 128, 12, 128))]
    sparse_ids = sparse_base + [("I32", (12, 16, 1, 4))]
    for layer in custom[sparse_name]:
        inputs = _xml_port_specs(layer, "input")
        if len(inputs) not in (3, 4):
            raise ValueError("serialized S1 input count changed")
        if len(inputs) == 3 and not ids_proven:
            raise ValueError("serialized S1 folded IDs lack source proof")
        if len(inputs) == 3:
            result["folded_block_id_ports"] += 1
        specs = sparse_ids if len(inputs) == 4 else sparse_base
        result["transport_ports"].setdefault(sparse_name, []).append(
            check(sparse_name, layer, specs,
                  [("FP16", (12, 16, 128, 128))]))
    sources = [entry["direct_input_ports"]
               for entry in result["transport_ports"][sparse_name]]
    q_ids = [ports[0]["raw_source_id"] for ports in sources]
    kv_pairs = [tuple(ports[i]["raw_source_id"] for i in (1, 2))
                for ports in sources]
    ids_for = lambda name: {node.attrib["id"] for node in custom[name]}
    if len(set(q_ids)) != 2 or set(q_ids) != ids_for("otccrossqpack"):
        raise ValueError("POST attention branches must consume distinct query packs")
    if (any(pair[0] == pair[1] for pair in kv_pairs)
            or {item for pair in kv_pairs for item in pair} != ids_for("otccrosskvpack")
            or (len(custom["otccrosskvpack"]) == 2 and kv_pairs[0] != kv_pairs[1])):
        raise ValueError("POST K/V pack reuse changed branch ordering or ownership")
    unpack_ids = [entry["direct_input_ports"][0]["raw_source_id"]
                  for entry in result["transport_ports"]["otccrosscontextunpack"]]
    if len(set(unpack_ids)) != 2 or set(unpack_ids) != ids_for(sparse_name):
        raise ValueError("POST output branches must consume distinct attention results")
    # The compiler merges identical K/V packs shared by both query halves.
    # This validates that rewrite; the conservative memory charge is unchanged.
    result["shared_kv_pack_roots"] = len(custom["otccrosskvpack"]) == 2
    return result


def _validate_bridge_ids(model, *, sparse_type: str = DEFAULT_SPARSE_TYPE) -> dict:
    if type(sparse_type) is not str or not sparse_type:
        raise TypeError("sparse_type must be a non-empty string")
    if sparse_type not in {DEFAULT_SPARSE_TYPE, PRECISE_SPARSE_TYPE}:
        raise ValueError("unsupported POST sparse primitive")
    expected = np.broadcast_to(
        np.arange(4, dtype=np.int32).reshape(1, 1, 1, 4),
        (12, 16, 1, 4)).copy()
    sparse_ops = [node for node in model.get_ordered_ops()
                  if node.get_type_name() in {DEFAULT_SPARSE_TYPE, PRECISE_SPARSE_TYPE}]
    if (len(sparse_ops) != 2 or any(node.get_type_name() != sparse_type or
                                  node.get_input_size() != 4 for node in sparse_ops)):
        raise ValueError("POST bridge must contain two four-input S1 nodes")
    proofs = []
    for node in sparse_ops:
        constant = node.input_value(3).get_node()
        if constant.get_type_name() != "Constant":
            raise ValueError("POST S1 block-ID input must be a direct Constant")
        if constant.get_output_element_type(0) != ov.Type.i32:
            raise ValueError("POST S1 block-ID Constant must be I32")
        shape = tuple(int(item) for item in
                     constant.get_output_partial_shape(0).to_shape())
        value = np.ascontiguousarray(np.asarray(constant.get_data()))
        if shape != expected.shape or value.dtype != np.int32 or value.shape != expected.shape:
            raise ValueError("POST S1 block-ID Constant ABI changed")
        if not np.array_equal(value, expected):
            raise ValueError("POST S1 block-ID Constant ordering changed")
        proofs.append({"friendly_name": constant.get_friendly_name(),
                       "shape": list(value.shape), "dtype": str(value.dtype),
                       "sha256": _sha_array(value), "values_exact": True})
    return {"count": len(proofs), "constants": proofs}


def _compile_options(variant: str, plugin_config: Path | None = None, *,
                     profile_operators: bool = False) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown POST variant: {variant}")
    if type(profile_operators) is not bool:
        raise TypeError("profile_operators must be a bool")
    options = {}
    if variant != "legacy-default":
        options["INFERENCE_PRECISION_HINT"] = ov.Type.dynamic
    if plugin_config is not None:
        options["CONFIG_FILE"] = str(plugin_config)
    if profile_operators:
        options[ov.properties.enable_profiling()] = True
    return options


def _policy(compiled, variant: str) -> dict:
    mode_key = "EXECUTION_MODE_HINT"
    actual_mode = str(compiled.get_property(mode_key))
    mode_token = "".join(ch for ch in actual_mode.upper() if ch.isalnum())
    if mode_token not in {"ACCURACY", "EXECUTIONMODEACCURACY"}:
        raise ValueError(f"POST execution mode is not ACCURACY: {actual_mode}")
    try:
        actual_precision = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    except Exception as error:  # pragma: no cover - plugin property availability
        actual_precision = f"unavailable: {type(error).__name__}: {error}"
    requested = "compile_config_default" if variant == "legacy-default" else "dynamic"
    precision_token = "".join(ch for ch in actual_precision.upper() if ch.isalnum())
    if requested == "dynamic" and precision_token not in {"DYNAMIC", "TYPEDYNAMIC"}:
        raise ValueError(f"POST precision policy is not dynamic: {actual_precision}")
    return {"requested_precision": requested, "actual_precision": actual_precision,
            "requested_execution_mode": "ACCURACY",
            "actual_execution_mode": actual_mode}


def _host_tensor(bits: np.ndarray):
    bits = np.asarray(bits)
    if not _finite_bits(bits):
        raise ValueError("finite contiguous BF16 bits required")
    tensor = ov.Tensor(ov.Type.bf16, bits.shape)
    tensor.data.view(np.uint16)[...] = bits
    return tensor


def _copy_output(root) -> np.ndarray:
    host = ov.Tensor(ov.Type.bf16, root.get_shape())
    root.copy_to(host)
    bits = np.ascontiguousarray(host.data.view(np.uint16))
    if not _finite_bits(bits):
        raise FloatingPointError("POST x_out is non-finite")
    return bits.copy()


def _copy_named_output(graph, name: str) -> np.ndarray:
    if name not in graph.buffers:
        raise ValueError(f"POST output owner is missing: {name}")
    return _copy_output(graph.buffers[name])


def _copy_ffn2_private_outputs(graph) -> dict[str, np.ndarray]:
    values = {name: _copy_named_output(graph, name)
              for name in FFN2_PRIVATE_OUTPUTS}
    for name, value in values.items():
        if value.shape != ROW_SHAPE or not _finite_bits(value):
            raise ValueError(f"invalid private FFN2 Result ABI: {name}")
    return values


def _copy_ffn2_states(request) -> dict[str, np.ndarray]:
    """Read exactly two Assign states after a completed request."""
    try:
        states = list(request.query_state())
    except Exception as error:
        raise RuntimeError("FFN2 Assign graph has no queryable state API") from error
    values = {}
    if len(states) != len(FFN2_STATE_IDS):
        raise ValueError("FFN2 request must own exactly two states")
    for state in states:
        state_id = str(state.name)
        tensor = state.state
        shape = tuple(int(value) for value in tensor.shape)
        if (state_id not in FFN2_STATE_IDS or state_id in values
                or tensor.element_type != ov.Type.bf16
                or shape != ROW_SHAPE):
            raise ValueError(f"invalid FFN2 state ABI: {state_id}")
        bits = np.ascontiguousarray(np.array(tensor.data.view(np.uint16), copy=True))
        if not _finite_bits(bits):
            raise FloatingPointError(f"non-finite FFN2 state: {state_id}")
        values[state_id] = bits
    if set(values) != set(FFN2_STATE_IDS):
        raise ValueError(f"FFN2 state IDs changed: {sorted(values)}")
    return values


def _state_output_values(states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    expected = {
        "plus_post_ffn2_state_" + output: output
        for output in FFN2_PRIVATE_OUTPUTS
    }
    if set(states) != set(expected):
        raise ValueError("FFN2 state key set changed")
    return {output: states[state_id] for state_id, output in expected.items()}


def _validate_ffn2_aba(records: dict[str, dict[str, np.ndarray]],
                        label: str = "FFN2 observables") -> dict:
    if set(records) != set(FFN2_STATE_CAPTURE_PHASES):
        raise ValueError(f"{label} must contain A1/B/A2 observations")
    first, changed, restored = records["A1"], records["B"], records["A2"]
    names = set(first)
    if names != set(changed) or names != set(restored) or not names:
        raise ValueError(f"{label} observable key set changed")
    differs = any(not np.array_equal(first[name], changed[name]) for name in names)
    restored_exact = all(np.array_equal(first[name], restored[name]) for name in names)
    if not differs:
        raise AssertionError(f"{label} A/B observations were identical")
    if not restored_exact:
        raise AssertionError(f"{label} A->B->A did not restore")
    return {"a_differs_from_b": True, "a_restored": True}


def _observable_summary(values: dict[str, np.ndarray]) -> dict:
    return {name: {"shape": list(value.shape), "raw_sha256": _sha_array(value)}
            for name, value in values.items()}


def _save_ffn2_observables(output: Path, layer: int, *, private: dict,
                           states: dict) -> dict:
    """Save bounded A/B/A raw bytes before any state/private equality gate."""
    arrays = {}
    for phase in FFN2_STATE_CAPTURE_PHASES:
        if phase in private:
            arrays.update({f"private_{phase}_{name}": value
                           for name, value in private[phase].items()})
        if phase in states:
            arrays.update({f"state_{phase}_{name}": value
                           for name, value in _state_output_values(states[phase]).items()})
    if not arrays:
        raise ValueError("no FFN2 observables were captured")
    path = output / f"layer{layer}_ffn2_observables.npz"
    np.savez(path, **arrays)
    return {"path": str(path), "bytes": int(path.stat().st_size),
            "sha256": _sha_file(path), "members": sorted(arrays)}


def _request(graph) -> float:
    start = time.perf_counter()
    graph.request.start_async()
    graph.request.wait()
    return time.perf_counter() - start


def _profiling_records(request) -> list[dict]:
    """Serialize every plugin record, including zero/failed-status entries."""
    records = []
    for item in request.get_profiling_info():
        records.append({
            "node_name": str(item.node_name),
            "node_type": str(item.node_type),
            "exec_type": str(item.exec_type),
            "status": str(item.status),
            "real_seconds": float(item.real_time.total_seconds()),
            "cpu_seconds": float(item.cpu_time.total_seconds()),
        })
    return records


def _upload_post_bank(graph, values: dict, bank=None):
    from plus_storage import storage_tensor
    names = {"weight__" + key.replace(".", "_"): key for key in ROLE_KEYS["post"]}
    if set(values) != set(names.values()):
        raise ValueError("complete layer source required for POST bank")
    if bank is None:
        bank = {name: graph.upload(name, values[key]) for name, key in names.items()}
        return bank
    if set(bank) != set(names):
        raise ValueError("POST bank keyset changed")
    staged = {}
    try:
        for name, key in names.items():
            graph._validate_tensor(name, bank[name], graph.inputs[name])
            staged[name] = storage_tensor(values[key], "bf16")
        for name in names:
            staged[name].copy_to(bank[name])
    finally:
        staged.clear()
    return bank


def _metric(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected32 = (expected.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    actual32 = (actual.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    delta = actual32 - expected32
    max_abs = float(np.max(np.abs(delta)))
    return {"max_abs": max_abs,
            "normalized_max_abs": max_abs / max(float(np.max(np.abs(expected32))), 1e-30),
            "relative_l2": float(np.linalg.norm(delta) /
                                  max(float(np.linalg.norm(expected32)), 1e-30))}


def _source_feeds(layer: dict) -> tuple[dict, dict, dict]:
    x_a = _repeat_rows(layer["feed_x_bits"])
    context_a = _repeat_rows(layer["feed_self_context_bits"])
    x_b = np.ascontiguousarray(np.roll(x_a, 1, axis=1))
    context_b = np.ascontiguousarray(np.roll(context_a, 1, axis=1))
    c0a, c1a = pack_context(context_a)
    c0b, c1b = pack_context(context_b)
    fixed = {name.removeprefix("feed_").removesuffix("_bits"): layer[name]
             for name in GATE_FIELDS}
    fixed.update(cross_k=layer["native_cross_k_bits"], cross_v=layer["native_cross_v_bits"])
    a = {"x": x_a, "self_context_0": c0a, "self_context_1": c1a}
    b = {"x": x_b, "self_context_0": c0b, "self_context_1": c1b}
    return a, b, fixed


def _set_roots(graph, roots):
    for name, root in roots.items():
        graph.request.set_tensor(name, root)


def _source_hashes(*, ffn2_state: bool = False) -> dict[str, str]:
    if type(ffn2_state) is not bool:
        raise TypeError("ffn2_state must be a bool")
    paths = (HERE, Path(build_post_rows.__code__.co_filename).resolve(),
             BASE / "openvino" / "full_runtime" / "plus_resident.py",
             EVIDENCE / "post_cross_bridge_20261017.py",
             EVIDENCE / "cross_precise_20261017.py",
             EVIDENCE / "bf16_cross_bridge_20261017.xml",
             EVIDENCE / "bf16_cross_bridge_20261017.cl",
             BASE / "openvino" / "chunked_attention" / "plus_sparse_xmx.xml",
             BASE / "openvino" / "chunked_attention" / "plus_sparse_xmx.cl",
             BASE / "openvino" / "full_runtime" / "plus_graphs.py",
             BASE / "openvino" / "full_runtime" / "plus_resident.py",
             BASE / "openvino" / "full_runtime" / "plus_weights.py")
    if ffn2_state:
        paths += (EVIDENCE / "post_ffn2_state_20261017.py",)
    result = {}
    for path in paths:
        path = Path(path)
        if path.is_file():
            result[str(path)] = _sha_file(path)
    return result


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _resource_guard(core, device: str, stage: str) -> dict:
    stats = bridge_runtime._stats(core, device)
    rss = _rss_bytes()
    if int(stats["free_bytes"]) < DEVICE_RESERVE_BYTES:
        raise MemoryError(f"POST device reserve breached at {stage}")
    if rss > HOST_RSS_LIMIT_BYTES:
        raise MemoryError(f"POST host RSS policy breached at {stage}")
    return {"stage": stage, "host_rss_bytes": rss, "gpu": stats}


def _run(args) -> int:
    if not args.run:
        raise ValueError("GPU execution requires explicit --run")
    if args.variant not in VARIANTS:
        raise ValueError("unknown POST variant")
    _validate_ffn2_state_option(args.variant, args.ffn2_state)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("--output must name a fresh directory")
    output.mkdir(parents=False, exist_ok=False)
    receipt_path, receipt = _reference_receipt(args.reference, args.reference_sha256)
    layers, field_metadata = _authenticate_reference(args.reference.resolve(), receipt)
    weights_receipt = authenticate_dit(CHECKPOINT, mode="turbo")
    report = {
        "status": "authenticated",
        "scope": "one full remote-context POST graph; no trajectory/decoder/full-pipeline claim",
        "variant": args.variant, "target": {"device": DEVICE, "bdf": BDF},
        "rows": ROWS,
        "ffn2_matmul": args.variant in FFN2_VARIANTS,
        "remote_context": True,
        "selected_layers": list(LAYERS), "warmups": WARMUPS, "samples": SAMPLES,
        "reference_receipt": str(receipt_path),
        "reference_receipt_sha256": args.reference_sha256.lower(),
        "reference_source_commit": receipt.get("source_commit"),
        "reference_layers": [{"layer": layer, "fields": field_metadata[layer]}
                             for layer in LAYERS],
        "weights_authentication": weights_receipt,
        "source_sha256": _source_hashes(ffn2_state=args.ffn2_state),
        "ffn2_state": {
            "enabled": args.ffn2_state,
            "allowed_variant": FFN2_STATE_VARIANT,
            "state_ids": list(FFN2_STATE_IDS),
            "state_shape": list(ROW_SHAPE),
            "state_element_type": "bf16",
            "state_bytes": FFN2_STATE_BYTES if args.ffn2_state else 0,
            "capture_phases": list(FFN2_STATE_CAPTURE_PHASES),
            "capture_private_results": (
                args.variant == FFN2_STATE_VARIANT and not args.ffn2_state),
            "capture_state_values": args.ffn2_state,
            "alias_credit_bytes": 0,
        },
        "timing_interval": "request.start_async through request.wait; all bind/readback outside",
        "profile_operators": args.profile_operators,
        "operator_profiling": {
            "requested": args.profile_operators,
            "diagnostic_only": True,
            "record_scope": (
                "each layer's final timed A request after 50 warmups and 100 samples; "
                "profiling retrieval and stability readback are outside the timing interval"
            ),
        },
        "limits": {"normalized_max_abs": NORM_LIMIT, "relative_l2": REL_LIMIT},
    }
    if args.variant in PRECISE_VARIANTS:
        helper_path = EVIDENCE / "cross_precise_20261017.py"
        report["precise_helper"] = {
            "path": str(helper_path), "sha256": _sha_file(helper_path)}
    report_path = output / "results.json"

    def persist():
        report["host_rss_peak_bytes"] = _rss_bytes()
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    persist()
    core = context = graph = output_root = None
    bank = fixed = roots_a = roots_b = None
    source = post_values = a_bits = b_bits = fixed_bits = None
    expected_a = expected_b = actual_a = actual_b = actual_a_after_b = None
    target = feeds = root = None
    try:
        if _rss_bytes() > HOST_RSS_LIMIT_BYTES:
            raise MemoryError("host RSS exceeded 4-GiB pre-Core policy")
        model = _post_model(args.variant, ffn2_state=args.ffn2_state)
        ffn2_matmul = args.variant in FFN2_VARIANTS
        report["abi"] = (_validate_ffn2_state_model(model)
                          if args.ffn2_state else
                          _validate_post_model(model, ffn2_matmul=ffn2_matmul))
        report["precompile_inventory"] = _inventory(
            model, args.variant, ffn2_state=args.ffn2_state)
        report["host_inventory"] = _host_inventory(
            layers, model, ffn2_state=args.ffn2_state)
        if report["host_inventory"]["required_host_bytes"] > HOST_RSS_LIMIT_BYTES:
            raise MemoryError("POST host inventory exceeds the 4-GiB policy")
        if args.variant in BRIDGE_VARIANTS:
            sparse_type = (PRECISE_SPARSE_TYPE if args.variant in PRECISE_VARIANTS
                           else DEFAULT_SPARSE_TYPE)
            report["sparse_type"] = sparse_type
            report["bridge_constant_proof"] = _validate_bridge_ids(
                model, sparse_type=sparse_type)
            if args.variant in PRECISE_VARIANTS:
                from cross_precise_20261017 import make_precise_plugin_config
                plugin_config = make_precise_plugin_config(output / "precise-config")
            else:
                plugin_config = bridge_runtime._make_plugin_config(output)
            report["plugin_config"] = {"path": str(plugin_config),
                                        "sha256": _sha_file(plugin_config)}
        else:
            plugin_config = None
        runtime_version = str(ov.get_version())
        if not runtime_version.startswith("2026.5."):
            raise RuntimeError(f"pinned OpenVINO 2026.5 required, got {runtime_version}")
        core = ov.Core()
        from attention_from_cuda import resolve
        device, devices = resolve(core, expected_name="B580")
        if device != DEVICE or not any(item.get("device") == DEVICE
                                       and item.get("bdf") == BDF for item in devices):
            raise RuntimeError(f"exact B580 GPU.1/BDF resolution failed: {devices!r}")
        context = core.get_default_context(device)
        before = bridge_runtime._stats(core, device)
        required = int(report["precompile_inventory"]["known_plus_workspace_reserve_bytes"])
        if required > int(before["free_bytes"]):
            raise MemoryError("POST precompile inventory would consume the device reserve")
        report.update({"runtime": runtime_version, "devices": devices,
                       "before_compile_stats": before})
        options = _compile_options(args.variant, plugin_config,
                                   profile_operators=args.profile_operators)
        graph = ResidentGraph(core, model, device, context=context, config=options)
        report["compile_seconds"] = graph.compile_seconds
        report["compile_policy"] = _policy(graph.compiled, args.variant)
        report["after_compile_stats"] = bridge_runtime._stats(core, device)
        report["runtime_xml"] = str(output / "post.runtime.xml")
        ov.serialize(graph.compiled.get_runtime_model(), report["runtime_xml"])
        report["runtime_xml_sha256"] = _sha_file(Path(report["runtime_xml"]))
        if args.variant in BRIDGE_VARIANTS:
            report["transport_abi"] = _serialized_post_transport_abi(
                Path(report["runtime_xml"]), ids_proven=True,
                sparse_type=(PRECISE_SPARSE_TYPE
                             if args.variant in PRECISE_VARIANTS
                             else DEFAULT_SPARSE_TYPE))
        output_root = graph.buffers["x_out"]
        if not hasattr(output_root, "get_byte_size"):
            raise ValueError("POST output root is not an owning remote tensor")
        output_records, timings = {}, {}
        report["layers"] = {}
        capture_private = report["ffn2_state"]["capture_private_results"]
        for layer in LAYERS:
            from safetensors import safe_open
            with safe_open(str(CHECKPOINT), framework="np", device="cpu") as handle:
                source = load_layer(handle, layer)
            post_values = {key: source[key] for key in ROLE_KEYS["post"]}
            del source
            layer_report = {"memory_samples": []}
            private_records, state_records = {}, {}
            report["layers"][str(layer)] = layer_report
            layer_report["memory_samples"].append(
                _resource_guard(core, device, "layer_before_weight_upload"))
            bank = _upload_post_bank(graph, post_values, bank)
            del post_values
            layer_report["memory_samples"].append(
                _resource_guard(core, device, "layer_weight_upload"))
            a_bits, b_bits, fixed_bits = _source_feeds(layers[layer])
            roots_a = {name: None for name in a_bits}
            roots_b = {name: None for name in b_bits}
            fixed = {name: None for name in fixed_bits}
            for name, bits in fixed_bits.items():
                fixed[name] = graph.allocate(graph.inputs[name])
                _host_tensor(bits).copy_to(fixed[name])
                graph.request.set_tensor(name, fixed[name])
            for target, feeds in ((roots_a, a_bits), (roots_b, b_bits)):
                for name, bits in feeds.items():
                    target[name] = graph.allocate(graph.inputs[name])
                    _host_tensor(bits).copy_to(target[name])
            for name, root in bank.items():
                graph.request.set_tensor(name, root)
            layer_report["memory_samples"].append(
                _resource_guard(core, device, "layer_inputs_bound"))
            _set_roots(graph, roots_a)
            _request(graph)
            actual_a = _copy_output(output_root)
            if capture_private and not args.ffn2_state:
                private_records["A1"] = _copy_ffn2_private_outputs(graph)
            if args.ffn2_state:
                state_records["A1"] = _copy_ffn2_states(graph.request)
            _set_roots(graph, roots_b)
            _request(graph)
            actual_b = _copy_output(output_root)
            if capture_private and not args.ffn2_state:
                private_records["B"] = _copy_ffn2_private_outputs(graph)
            if args.ffn2_state:
                state_records["B"] = _copy_ffn2_states(graph.request)
            _set_roots(graph, roots_a)
            _request(graph)
            actual_a_after_b = _copy_output(output_root)
            if capture_private and not args.ffn2_state:
                private_records["A2"] = _copy_ffn2_private_outputs(graph)
            if args.ffn2_state:
                state_records["A2"] = _copy_ffn2_states(graph.request)
            layer_report["memory_samples"].append(
                _resource_guard(core, device, "layer_aba_complete"))
            expected_a = _repeat_rows(layers[layer]["native_x_out_bits"])
            expected_b = np.ascontiguousarray(np.roll(expected_a, 1, axis=1))
            metrics = {"a": _metric(expected_a, actual_a),
                       "b": _metric(expected_b, actual_b),
                       "a_after_b": _metric(expected_a, actual_a_after_b)}
            numerical_failure = any(
                m["normalized_max_abs"] > NORM_LIMIT or m["relative_l2"] > REL_LIMIT
                for m in metrics.values())
            aba_failure = (np.array_equal(actual_a, actual_b)
                           or not np.array_equal(actual_a, actual_a_after_b))
            output_records[str(layer)] = {"a": actual_a, "b": actual_b,
                                          "a_after_b": actual_a_after_b}
            layer_report["outputs"] = {
                label: {"shape": list(value.shape), "raw_sha256": _sha_array(value)}
                for label, value in output_records[str(layer)].items()}
            layer_report["metrics"] = metrics
            if capture_private:
                layer_report["ffn2_private_results"] = {
                    "phases": {phase: _observable_summary(values)
                               for phase, values in private_records.items()},
                }
            if args.ffn2_state:
                state_summary = {
                    phase: _observable_summary(_state_output_values(values))
                    for phase, values in state_records.items()
                }
                layer_report["ffn2_states"] = {
                    "state_ids": list(FFN2_STATE_IDS), "phases": state_summary}
            if capture_private or args.ffn2_state:
                layer_report["ffn2_observable_archive"] = _save_ffn2_observables(
                    output, layer, private=private_records, states=state_records)
            partial = output / f"layer{layer}_post_outputs.npz"
            np.savez(partial, **{label: value for label, value in
                                 output_records[str(layer)].items()})
            layer_report["partial_archive"] = {
                "path": str(partial), "bytes": partial.stat().st_size,
                "sha256": _sha_file(partial)}
            persist()
            if numerical_failure:
                raise ValueError(f"layer {layer} POST numerical gate failed: {metrics}")
            if capture_private:
                layer_report["ffn2_private_results"]["aba"] = _validate_ffn2_aba(
                    private_records, "FFN2 private Results")
            if args.ffn2_state:
                layer_report["ffn2_states"]["aba"] = _validate_ffn2_aba(
                    {phase: _state_output_values(values)
                     for phase, values in state_records.items()},
                    "FFN2 Assign states")
            if aba_failure:
                raise ValueError(f"layer {layer} POST A-B-A input binding proof failed")
            timings[str(layer)] = {"warmups": WARMUPS,
                                   "warmup_seconds": [], "sample_seconds": []}
            _set_roots(graph, roots_a)
            for _ in range(WARMUPS):
                _request(graph)
            if capture_private and not args.ffn2_state:
                late_private = _copy_ffn2_private_outputs(graph)
                layer_report["ffn2_private_results"]["late_A"] = _observable_summary(
                    late_private)
                if any(not np.array_equal(late_private[name], private_records["A1"][name])
                       for name in FFN2_PRIVATE_OUTPUTS):
                    raise AssertionError("late warm A private Result changed")
                del late_private
            if args.ffn2_state:
                late_states = _copy_ffn2_states(graph.request)
                layer_report["ffn2_states"]["late_A"] = _observable_summary(
                    _state_output_values(late_states))
                if any(not np.array_equal(late_states[state_id], state_records["A1"][state_id])
                       for state_id in FFN2_STATE_IDS):
                    raise AssertionError("late warm A Assign state changed")
                del late_states
            layer_report["memory_samples"].append(
                _resource_guard(core, device, "layer_warmups_complete"))
            for _ in range(SAMPLES):
                timings[str(layer)]["sample_seconds"].append(_request(graph))
            timings[str(layer)]["summary"] = {
                "mean": float(np.mean(timings[str(layer)]["sample_seconds"])),
                "median": float(np.median(timings[str(layer)]["sample_seconds"])),
                "min": float(np.min(timings[str(layer)]["sample_seconds"])),
                "max": float(np.max(timings[str(layer)]["sample_seconds"])),
            }
            layer_report["timing_summary"] = timings[str(layer)]["summary"]
            if args.profile_operators:
                layer_report["operator_profile_records"] = _profiling_records(graph.request)
                late = _copy_output(output_root)
                layer_report["profiled_output_raw_sha256"] = _sha_array(late)
                if not np.array_equal(late, actual_a_after_b):
                    raise ValueError("profiled warm output differs from restored A")
                del late
            layer_report["memory_samples"].append(
                _resource_guard(core, device, "layer_timing_complete"))
            persist()
            del a_bits, b_bits, fixed_bits, expected_a, expected_b
            del actual_a, actual_b, actual_a_after_b
            del private_records, state_records
            roots_a = roots_b = fixed = None
            gc.collect()
        archive = output / "post_outputs.npz"
        np.savez(archive, **{f"layer{layer}_{label}": value
                             for layer, values in output_records.items()
                             for label, value in values.items()})
        report["output_archive"] = {"path": str(archive), "sha256": _sha_file(archive),
                                     "bytes": archive.stat().st_size}
        for layer, values in output_records.items():
            layer_report = report["layers"].setdefault(layer, {})
            layer_report["outputs"] = {
                label: {"shape": list(value.shape), "raw_sha256": _sha_array(value)}
                for label, value in values.items()}
            layer_report["metrics"] = {
                label: _metric(
                    _repeat_rows(layers[int(layer)]["native_x_out_bits"])
                    if label != "b" else np.ascontiguousarray(np.roll(
                        _repeat_rows(layers[int(layer)]["native_x_out_bits"]), 1, axis=1)),
                    value)
                for label, value in values.items()}
        report["timing"] = timings
        report["after_infer_stats"] = bridge_runtime._stats(core, device)
        report["status"] = "passed_native_post_qualification_component"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        raise
    finally:
        roots_a = roots_b = fixed = bank = None
        source = post_values = a_bits = b_bits = fixed_bits = None
        expected_a = expected_b = actual_a = actual_b = actual_a_after_b = None
        target = feeds = root = None
        if graph is not None:
            graph.request = None
            graph.compiled = None
            graph.buffers.clear()
            graph.inputs.clear()
            graph.outputs.clear()
        graph = output_root = context = None
        gc.collect()
        if core is not None:
            try:
                report["final_gpu_memory"] = bridge_runtime._stats(core, DEVICE)
                report["final_allocations_zero"] = report["final_gpu_memory"].get("allocated_bytes") == 0
                if not report["final_allocations_zero"]:
                    report.update(status="fail_allocations_not_released",
                                  error="GPU allocations survived final retirement")
            except BaseException as error:
                report["final_allocations_zero"] = False
                report["final_stats_error"] = f"{type(error).__name__}: {error}"
                report.update(status="fail_final_gpu_stats",
                              error="final GPU statistics unavailable")
        report["final_host_rss_bytes"] = _rss_bytes()
        report["final_host_rss_within_limit"] = (
            report["final_host_rss_bytes"] <= HOST_RSS_LIMIT_BYTES)
        if not report["final_host_rss_within_limit"]:
            report.update(status="fail_host_rss", error="final host RSS exceeded policy")
        core = None
        persist()
    print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)
    return 0 if report["status"].startswith("passed") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--reference", type=Path, default=REFERENCE_DEFAULT)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--ffn2-state", action="store_true",
        help=("replace only the two precise FFN2 private Results with Assign "
              "states; allowed only for bridge-precise-ffn2-dynamic"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-operators", action="store_true",
                        help="enable diagnostic plugin profiling and record per-layer nodes")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("GPU execution requires explicit --run")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
