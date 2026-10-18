#!/usr/bin/env python3
"""Opt-in B580 proof for the lossless BF16 bridge draft.

This worker has two independent checks: all 65,280 finite BF16 words pass a
Q-pack/context-unpack round trip, and a fixed four-key sparse attention graph
using the bridge is raw-bit identical to the existing F16-transport S1 graph
for realistic finite A-B-A inputs.  It is a component proof only; it is not a
full-model, quality, or throughput qualification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

BASE = Path(__file__).resolve().parents[1]
PINNED_RUNTIME = BASE / "runtime-openvino-2026.5.0.dev20261014"
CHUNKED = BASE / "openvino" / "chunked_attention"
EVIDENCE = BASE / "evidence"
sys.path[:0] = [str(PINNED_RUNTIME), str(CHUNKED), str(BASE / "openvino"),
                str(EVIDENCE)]

import openvino as ov  # noqa: E402
import bf16_cross_bridge_20261017 as bridge  # noqa: E402
import plus_sparse_xmx as kernel  # noqa: E402


HEADS, QUERIES, DEPTH, KEYS = 12, 16, 128, 4
Q_SHAPE = (HEADS, QUERIES, DEPTH, DEPTH)
KV_SHAPE = (KEYS, DEPTH, HEADS, DEPTH)
IDS_SHAPE = (HEADS, QUERIES, 1, KEYS)
OUT_SHAPE = Q_SHAPE
DEVICE = "GPU.1"
BDF = "0000:03:00.0"
SEED = 20261017
DEVICE_RESERVE_BYTES = 2 << 30
WORKSPACE_ALLOWANCE_BYTES = 2 << 30
HOST_RSS_LIMIT_BYTES = 4 << 30
CONTROL_XML = CHUNKED / "plus_sparse_xmx.xml"
CONTROL_CL = CHUNKED / "plus_sparse_xmx.cl"
BRIDGE_XML = EVIDENCE / "bf16_cross_bridge_20261017.xml"
BRIDGE_CL = EVIDENCE / "bf16_cross_bridge_20261017.cl"
BRIDGE_PY = EVIDENCE / "bf16_cross_bridge_20261017.py"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    value = np.asarray(value)
    if not value.flags.c_contiguous:
        raise ValueError("evidence arrays must be C-contiguous")
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _finite_bits(value: np.ndarray) -> bool:
    value = np.asarray(value)
    return (value.dtype == np.uint16 and value.flags.c_contiguous
            and not np.any((value & np.uint16(0x7F80)) == np.uint16(0x7F80)))


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
                raise ValueError("Malformed VmRSS")
            return int(fields[1]) * 1024
    raise ValueError("VmRSS unavailable")


def _check_rss(stage: str, report: dict) -> None:
    rss = _rss_bytes()
    report.setdefault("host_rss_samples", []).append({"stage": stage, "bytes": rss})
    if rss > HOST_RSS_LIMIT_BYTES:
        raise MemoryError("host RSS exceeded 4-GiB bridge limit")


def _finite_words() -> np.ndarray:
    words = np.arange(1 << 16, dtype=np.uint16)
    return words[(words & np.uint16(0x7F80)) != np.uint16(0x7F80)]


def _realistic_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    q_a = kernel.float32_to_bf16_bits(
        rng.normal(0.0, 0.2, Q_SHAPE).astype(np.float32))
    q_b = kernel.float32_to_bf16_bits(
        rng.normal(0.0, 0.2, Q_SHAPE).astype(np.float32))
    k = kernel.float32_to_bf16_bits(
        rng.normal(0.0, 0.2, KV_SHAPE).astype(np.float32))
    v = kernel.float32_to_bf16_bits(
        rng.normal(0.0, 0.2, KV_SHAPE).astype(np.float32))
    ids = np.broadcast_to(
        np.arange(KEYS, dtype=np.int32).reshape(1, 1, 1, KEYS), IDS_SHAPE
    ).copy()
    result = {name: np.ascontiguousarray(value) for name, value in (
        ("q_a", q_a), ("q_b", q_b), ("k", k), ("v", v), ("ids", ids))}
    for name in ("q_a", "q_b", "k", "v"):
        if not _finite_bits(result[name]):
            raise ValueError(f"generated {name} is not finite BF16 storage")
    kernel.validate_inputs(result["q_a"], result["k"], result["v"], result["ids"])
    return result


def _source_inputs(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert control-layout raw bits to the bridge's token-major source ABI."""
    result = {}
    for name in ("q_a", "q_b"):
        result[name] = np.ascontiguousarray(
            data[name].transpose(1, 2, 0, 3).reshape(bridge.Q_SHAPE))
    for name in ("k", "v"):
        result[name] = np.ascontiguousarray(data[name].reshape(bridge.KV_SHAPE))
    return result


def _build_roundtrip_model():
    return bridge.build_roundtrip_model()


def _source_context_to_control(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.uint16 or value.shape != bridge.Q_SHAPE or not value.flags.c_contiguous:
        raise ValueError("bridge context output must be contiguous source-layout uint16 bits")
    return np.ascontiguousarray(value.reshape(1, 16, 128, 12, 128)[0]
                                .transpose(2, 0, 1, 3))


def _host_tensor(bits: np.ndarray, kind):
    bits = np.asarray(bits)
    if bits.dtype != np.uint16 or not bits.flags.c_contiguous or not _finite_bits(bits):
        raise ValueError("finite contiguous uint16 storage required")
    tensor = ov.Tensor(kind, bits.shape)
    tensor.data.view(np.uint16)[...] = bits
    return tensor


def _copy_output(output_root, kind, shape) -> np.ndarray:
    host = ov.Tensor(kind, shape)
    output_root.copy_to(host)
    bits = np.ascontiguousarray(host.data.view(np.uint16))
    if bits.shape != tuple(shape) or not _finite_bits(bits):
        raise ValueError("compiled bridge produced invalid finite BF16 transport")
    return np.array(bits, copy=True)


def _upload_host(remote_root, bits: np.ndarray, kind) -> None:
    """Use the pinned runtime's owning-host -> remote copy direction."""
    _host_tensor(bits, kind).copy_to(remote_root)


def _stats(core, device: str) -> dict:
    raw = dict(core.get_property(device, "GPU_MEMORY_STATISTICS"))
    allocated = sum(int(raw.get(key, 0)) for key in
                    ("cl_mem", "usm_device", "usm_host", "usm_shared", "unknown"))
    total = int(core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE"))
    return {"allocated_bytes": allocated, "total_bytes": total,
            "free_bytes": total - allocated, "statistics": raw}


def _inventory(data: dict[str, np.ndarray]) -> dict:
    bridge_bytes = bridge.bridge_inventory_bytes()
    control_roots = sum(int(data[name].nbytes) for name in ("q_a", "q_b", "k", "v"))
    ids = int(data["ids"].nbytes)
    output = int(np.prod(OUT_SHAPE, dtype=np.int64) * 2)
    source_inputs = sum(int(data[name].nbytes) for name in ("q_a", "q_b", "k", "v"))
    # This is intentionally a sum of independent request/root/staging owners;
    # no CustomLayer alias or plugin workspace credit is assumed.
    default_request = 2 * (source_inputs + ids + output)
    named = (bridge_bytes["conservative_bridge_bytes"] + control_roots + ids
             + output + default_request + WORKSPACE_ALLOWANCE_BYTES)
    return {
        "bridge_named_bytes": bridge_bytes,
        "control_device_roots_bytes": control_roots,
        "ordered_ids_bytes": ids,
        "control_output_root_bytes": output,
        "default_request_allowance_bytes": default_request,
        "workspace_allowance_bytes": WORKSPACE_ALLOWANCE_BYTES,
        "device_reserve_bytes": DEVICE_RESERVE_BYTES,
        "named_plus_reserve_bytes": int(named + DEVICE_RESERVE_BYTES),
    }


def _make_plugin_config(output: Path) -> Path:
    """Create a fresh combined registration without modifying source assets."""
    shutil.copy2(BRIDGE_CL, output / BRIDGE_CL.name)
    shutil.copy2(CONTROL_CL, output / CONTROL_CL.name)
    config = output / "bf16_cross_bridge_custom_layers.xml"
    config.write_text(BRIDGE_XML.read_text(encoding="utf-8") + "\n"
                      + CONTROL_XML.read_text(encoding="utf-8"), encoding="utf-8")
    return config


def _compile(core, context, device, model, config: Path, label: str,
             precision, output: Path, *, public_inputs=(), public_output=None):
    options = {
        "CONFIG_FILE": str(config),
        "INFERENCE_PRECISION_HINT": precision,
        "EXECUTION_MODE_HINT": "ACCURACY",
        "PERFORMANCE_HINT": ov.properties.hint.PerformanceMode.LATENCY,
        "NUM_STREAMS": ov.properties.streams.Num(1),
        "CACHE_DIR": "",
    }
    started = time.perf_counter()
    compiled = core.compile_model(model, context, options)
    compile_seconds = time.perf_counter() - started
    if list(compiled.get_property("EXECUTION_DEVICES")) != [device]:
        raise ValueError(f"{label} compiled on an unintended device")
    actual_precision = compiled.get_property("INFERENCE_PRECISION_HINT")
    actual_execution_mode = compiled.get_property("EXECUTION_MODE_HINT")
    public_abi = _compiled_public_abi(compiled, public_inputs, public_output)
    runtime = output / f"runtime-{label}.xml"
    ov.serialize(compiled.get_runtime_model(), runtime)
    return compiled, {
        "compile_seconds": compile_seconds,
        "runtime_xml": runtime.name,
        "runtime_xml_sha256": _sha_file(runtime),
        "requested_precision": str(precision),
        "requested_execution_mode": "ACCURACY",
        "actual_precision": str(actual_precision),
        "actual_execution_mode": str(actual_execution_mode),
        "public_abi": public_abi,
        "execution_devices": list(compiled.get_property("EXECUTION_DEVICES")),
    }


def _compiled_public_abi(compiled, expected_inputs, expected_output):
    """Validate the public model ports after compilation, not only the model."""
    inputs = []
    for name, expected_type, expected_shape in expected_inputs:
        try:
            port = compiled.input(name)
        except Exception as error:
            raise ValueError(f"compiled model is missing public input {name}") from error
        shape = tuple(int(item) for item in port.get_shape())
        if shape != tuple(expected_shape) or port.get_element_type() != expected_type:
            raise ValueError(
                f"compiled public input {name} ABI changed: "
                f"{shape}/{port.get_element_type()}"
            )
        inputs.append({"name": name, "shape": list(shape),
                       "element_type": str(port.get_element_type())})
    output = None
    try:
        output = compiled.output(expected_output)
    except Exception as error:
        raise ValueError(f"compiled model is missing public output {expected_output}") from error
    shape = tuple(int(item) for item in output.get_shape())
    expected_type = (ov.Type.f16 if expected_output == "attention" else ov.Type.bf16)
    expected_shape = OUT_SHAPE if expected_output == "attention" else bridge.Q_SHAPE
    if shape != tuple(expected_shape) or output.get_element_type() != expected_type:
        raise ValueError(
            f"compiled public output {expected_output} ABI changed: "
            f"{shape}/{output.get_element_type()}"
        )
    return {"inputs": inputs, "output": {
        "name": expected_output, "shape": list(shape),
        "element_type": str(output.get_element_type()),
    }}


def _validate_block_constant(model, output: Path | None = None) -> dict:
    """Prove the bridge's compile-time block IDs before accepting a 3-port S1."""
    expected = np.broadcast_to(
        np.arange(KEYS, dtype=np.int32).reshape(1, 1, 1, KEYS), IDS_SHAPE
    ).copy()
    sparse_ops = [node for node in model.get_ordered_ops()
                  if node.get_type_name() == "OtcPlusSparseXmx"]
    if len(sparse_ops) != 1 or sparse_ops[0].get_input_size() != 4:
        raise ValueError("bridge must contain exactly one four-input S1 op")
    sparse = sparse_ops[0]
    node = sparse.input_value(3).get_node()
    if node.get_type_name() != "Constant":
        raise ValueError("S1 block-ID input must be directly sourced by Constant")
    if node.get_output_element_type(0) != ov.Type.i32:
        raise ValueError("S1 block-ID Constant must be int32")
    shape = tuple(int(item) for item in
                 node.get_output_partial_shape(0).to_shape())
    if shape != IDS_SHAPE:
        raise ValueError("S1 block-ID Constant shape changed")
    value = np.ascontiguousarray(np.asarray(node.get_data()))
    if value.dtype != np.int32 or value.shape != IDS_SHAPE:
        raise ValueError("block-ID Constant ABI changed")
    if not np.array_equal(value, expected):
        raise ValueError("block-ID Constant is not the exact ordered 0..3 pattern")
    proof = {"name": node.get_friendly_name(), "dtype": str(value.dtype),
             "shape": list(value.shape), "unique_values": np.unique(value).tolist(),
             "sha256": _sha_array(value), "values_exact": True}
    if output is not None:
        asset = Path(output) / "block_constant.npy"
        np.save(asset, value, allow_pickle=False)
        proof["asset"] = asset.name
        proof["asset_sha256"] = _sha_file(asset)
    return proof


def _port_specs(layer: ET.Element, section_name: str) -> list[dict]:
    section = layer.find(section_name)
    if section is None:
        raise ValueError(f"serialized CustomLayer is missing {section_name} ports")
    specs = []
    aliases = {"F32": "FP32", "F16": "FP16", "BFLOAT16": "BF16"}
    for port in sorted(section.findall("port"), key=lambda item: int(item.attrib["id"])):
        precision = port.attrib.get("precision")
        if precision is None:
            raise ValueError("serialized CustomLayer port has no precision")
        dims = tuple(int(dim.text) for dim in port.findall("dim"))
        if not dims or any(value <= 0 for value in dims):
            raise ValueError("serialized CustomLayer port has no static positive shape")
        specs.append({"id": int(port.attrib["id"]),
                      "precision": aliases.get(precision.upper(), precision.upper()),
                      "shape": list(dims)})
    return specs


def _serialized_bridge_abi(path: Path, *, require_attention: bool = False,
                           constant_ids_proven: bool = False) -> dict:
    """Check serialized CustomLayer ports and forbid numeric F16 Converts."""
    if constant_ids_proven and not require_attention:
        raise ValueError("constant-ID proof is only valid for the S1 attention graph")
    root = ET.parse(path).getroot()
    layers = {layer.attrib.get("id"): layer for layer in root.findall("./layers/layer")}
    custom = {}
    for layer in layers.values():
        if layer.attrib.get("type") == "CustomGPUPrimitive":
            data = layer.find("data")
            primitive = "" if data is None else data.attrib.get("primitiveType", "")
            # The pinned GPU plugin appends the output precision to its
            # CustomLayer entry (for example ``otc_plus_sparse_xmx__f16``).
            # Strip only the two observed transport suffixes; an unknown
            # suffix must remain an unexpected primitive rather than silently
            # broadening the accepted ABI.
            primitive = primitive.lower()
            for suffix in ("__f16", "__f32"):
                if primitive.endswith(suffix):
                    primitive = primitive[:-len(suffix)]
                    break
            normalized = "".join(ch for ch in primitive if ch.isalnum())
            custom.setdefault(normalized, []).append(layer)
    expected = {
        "otccrossqpack": (("FP32", [1, 1, 2048, 1536]),
                          ("FP16", [12, 16, 128, 128]), 1),
        "otccrosscontextunpack": (("FP16", [12, 16, 128, 128]),
                                   ("FP32", [1, 1, 2048, 1536]), 1),
    }
    if require_attention:
        expected["otccrosskvpack"] = (("FP32", [1, 1, 512, 1536]),
                                       ("FP16", [4, 128, 12, 128]), 2)
        sparse_inputs = (("FP16", [12, 16, 128, 128]),
                         ("FP16", [4, 128, 12, 128]),
                         ("FP16", [4, 128, 12, 128]))
        if not constant_ids_proven:
            sparse_inputs += (("I32", [12, 16, 1, 4]),)
        expected["otcplussparsexmx"] = (
            sparse_inputs,
            (("FP16", [12, 16, 128, 128]),), 1)
    allowed = set(expected)
    actual = set(custom)
    missing = sorted(allowed - actual)
    extra = sorted(actual - allowed)
    if missing or extra:
        raise ValueError(f"serialized CustomLayers changed; missing={missing}, extra={extra}")

    def check_layer_abi(primitive, layer, input_specs, output_specs):
        inputs = _port_specs(layer, "input")
        outputs = _port_specs(layer, "output")
        actual_inputs = [(item["precision"], tuple(item["shape"])) for item in inputs]
        actual_outputs = [(item["precision"], tuple(item["shape"])) for item in outputs]
        expected_inputs = [(precision, tuple(shape)) for precision, shape in input_specs]
        expected_outputs = [(precision, tuple(shape)) for precision, shape in output_specs]
        if actual_inputs != expected_inputs:
            raise ValueError(f"serialized {primitive} input ABI changed: {inputs}")
        if actual_outputs != expected_outputs:
            raise ValueError(f"serialized {primitive} output ABI changed: {outputs}")
        incoming = [edge for edge in root.findall("./edges/edge")
                    if edge.attrib.get("to-layer") == layer.attrib.get("id")]
        if len(incoming) != len(input_specs):
            raise ValueError(f"serialized {primitive} input edge count changed")
        source_precisions = []
        for edge in sorted(incoming, key=lambda item: int(item.attrib["to-port"])):
            source = layers.get(edge.attrib.get("from-layer"))
            if source is None:
                raise ValueError(f"serialized {primitive} input edge has no source")
            source_ports = _port_specs(source, "output")
            source_port = int(edge.attrib.get("from-port", "0"))
            matching_ports = [item for item in source_ports if item["id"] == source_port]
            if len(matching_ports) != 1:
                raise ValueError(f"serialized {primitive} input edge port is out of range")
            source_precisions.append(matching_ports[0]["precision"])
        expected_precisions = [spec[0] for spec in expected_inputs]
        if source_precisions != expected_precisions:
            raise ValueError(f"serialized {primitive} direct input ABI changed")
        return {"input_ports": inputs, "output_ports": outputs,
                "direct_input_precisions": source_precisions}

    port_report = {}
    for primitive, (input_spec, output_spec, expected_count) in expected.items():
        if len(custom[primitive]) != expected_count:
            raise ValueError(f"serialized {primitive} count changed")
        rows = []
        for layer in custom[primitive]:
            if primitive == "otcplussparsexmx":
                input_specs = input_spec
                output_specs = output_spec
            else:
                input_specs = (input_spec,)
                output_specs = (output_spec,)
            rows.append(check_layer_abi(primitive, layer, input_specs, output_specs))
        port_report[primitive] = rows
    convert_edges = []
    reorder_edges = []
    aliases = {"F32": "FP32", "F16": "FP16"}
    for layer in layers.values():
        layer_type = layer.attrib.get("type")
        if layer_type not in {"Convert", "Reorder"}:
            continue
        inputs = _port_specs(layer, "input")
        outputs = _port_specs(layer, "output")
        in_precisions = [item["precision"] for item in inputs]
        out_precisions = [item["precision"] for item in outputs]
        if len(in_precisions) != len(out_precisions):
            raise ValueError(f"serialized {layer_type} port count changed")
        pairs = list(zip(in_precisions, out_precisions))
        if layer_type == "Convert":
            if any("FP16" in pair for pair in pairs):
                raise ValueError("serialized bridge contains an implicit numeric F16 Convert")
            convert_edges.extend(pairs)
        else:
            # Same-type reorders are layout-only.  A plugin may represent the
            # explicit BF16<->F32 carrier as a Reorder, but an F16 precision
            # change would be an unqualified numeric half conversion.
            allowed = (("BF16", "FP32"), ("FP32", "BF16"))
            if any(source != target and (source, target) not in allowed
                   for source, target in pairs):
                raise ValueError("serialized bridge contains an unsafe precision-changing Reorder")
            reorder_edges.extend(pairs)
    if convert_edges and not all(pair in (("BF16", "FP32"), ("FP32", "BF16"))
                                 for pair in convert_edges):
        raise ValueError(f"unexpected bridge Convert precision pairs: {convert_edges}")
    return {"custom_primitives": sorted(custom), "custom_port_abi": port_report,
            "convert_pairs": convert_edges, "reorder_pairs": reorder_edges,
            "no_numeric_f16_convert": True, "no_unsafe_precision_changing_reorder": True}


def _run_roundtrip(compiled, context, words: np.ndarray) -> np.ndarray:
    root = output_root = request = None
    try:
        root = context.create_tensor(ov.Type.bf16, ov.Shape(words.shape), {})
        _upload_host(root, words, ov.Type.bf16)
        output_root = context.create_tensor(ov.Type.bf16, ov.Shape(words.shape), {})
        request = compiled.create_infer_request()
        request.set_tensor("q_bf16", root)
        # The public model output remains the BF16 context name; the internal
        # bridge ports are rank-four F32/transport shapes only.
        request.set_tensor(compiled.output(0).get_any_name(), output_root)
        # Keep the qualified asynchronous request lifecycle.  Synchronous
        # infer() eagerly maps remote outputs on this pinned plugin.
        request.start_async()
        request.wait()
        return _copy_output(output_root, ov.Type.bf16, words.shape)
    finally:
        request = output_root = root = None
        gc.collect()


def _run_attention(compiled, context, device, q_values, k, v, ids,
                   *, source_layout: bool) -> dict[str, np.ndarray]:
    kind = ov.Type.bf16 if source_layout else ov.Type.f16
    roots = request = output_root = id_tensor = None
    try:
        roots = {}
        for name, value in (("k", k), ("v", v), ("q_a", q_values["a"]),
                            ("q_b", q_values["b"])):
            roots[name] = context.create_tensor(kind, ov.Shape(value.shape), {})
            _upload_host(roots[name], value, kind)
        id_tensor = ov.Tensor(ov.Type.i32, ids.shape)
        id_tensor.data[...] = ids
        out_shape = bridge.Q_SHAPE if source_layout else OUT_SHAPE
        output_root = context.create_tensor(kind, ov.Shape(out_shape), {})
        request = compiled.create_infer_request()
        if source_layout:
            fixed = (("k_bf16", roots["k"]), ("v_bf16", roots["v"]))
            q_name = "q_bf16"
            out_name = "context_bf16"
        else:
            fixed = (("K", roots["k"]), ("V", roots["v"]), ("blocks", id_tensor))
            q_name = "Q"
            out_name = "attention"
        for name, tensor in fixed:
            request.set_tensor(name, tensor)
        result = {}
        for label, q in (("a", roots["q_a"]), ("b", roots["q_b"]),
                         ("a_after_b", roots["q_a"])):
            request.set_tensor(q_name, q)
            request.set_tensor(out_name, output_root)
            request.start_async()
            request.wait()
            result[label] = _copy_output(output_root, kind, out_shape)
        if np.array_equal(result["a"], result["b"]):
            raise ValueError("attention ignored changed B input")
        if not np.array_equal(result["a"], result["a_after_b"]):
            raise ValueError("attention failed raw A-B-A restoration")
        return result
    finally:
        request = output_root = id_tensor = None
        roots.clear()
        gc.collect()


def _status_exit_code(status: str) -> int:
    return 0 if status == "pass_bf16_cross_bridge_bitexact" else 1


def _record_final_stats(report: dict, stats: dict) -> None:
    """Persist final allocation evidence and turn a leak into a hard failure."""
    report["final_gpu_memory"] = stats
    report["final_allocations_zero"] = int(stats.get("allocated_bytes", -1)) == 0
    if not report["final_allocations_zero"]:
        prior = report.get("error")
        report["status"] = "fail_allocations_not_released"
        report["error"] = "GPU allocations remained after teardown"
        if prior:
            report["prior_error"] = prior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="explicitly enable pinned B580 execution")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("GPU execution requires explicit --run")
    if args.output.exists():
        parser.error("output directory must be fresh")
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = (Path(__file__), BRIDGE_PY, BRIDGE_CL, BRIDGE_XML,
                    CHUNKED / "plus_sparse_xmx.py", CONTROL_CL, CONTROL_XML,
                    BASE / "openvino" / "attention_from_cuda.py")
    report = {
        "status": "initializing",
        "scope": "lossless BF16 bridge component; no full-model quality/throughput claim",
        "quality_qualified": False,
        "target_bdf": BDF,
        "requested_gpu_model": "B580",
        "geometry": {"bridge_q": list(bridge.Q_SHAPE),
                      "bridge_kv": list(bridge.KV_SHAPE),
                      "packed_q": list(bridge.Q_PACKED_SHAPE),
                      "packed_kv": list(bridge.KV_PACKED_SHAPE),
                      "control_q": list(Q_SHAPE), "control_kv": list(KV_SHAPE),
                      "ids": list(IDS_SHAPE)},
        "source_sha256": {str(path): _sha_file(path) for path in source_files},
    }

    def persist():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    persist()
    core = context = compiled = model = None
    try:
        data = _realistic_inputs()
        source = _source_inputs(data)
        finite = _finite_words()
        roundtrip_input = np.resize(finite, int(np.prod(bridge.Q_SHAPE))).reshape(bridge.Q_SHAPE).copy()
        report["finite_word_coverage"] = {"finite_word_count": int(finite.size),
                                           "all_finite_words_present": True,
                                           "input_sha256": _sha_array(roundtrip_input)}
        report["realistic_input_sha256"] = {name: _sha_array(value) for name, value in data.items()}
        report["bridge_source_input_sha256"] = {name: _sha_array(value) for name, value in source.items()}
        np.savez(args.output / "inputs.npz", roundtrip_q=roundtrip_input,
                 q_a=data["q_a"], q_b=data["q_b"], k=data["k"], v=data["v"], ids=data["ids"])
        report["inputs_archive_sha256"] = _sha_file(args.output / "inputs.npz")
        report["inventory"] = _inventory(data)
        _check_rss("after_inputs", report)
        if not ov.get_version().startswith("2026.5."):
            raise ValueError(f"pinned OpenVINO 2026.5 required, got {ov.get_version()}")
        core = ov.Core()
        from attention_from_cuda import resolve
        device, devices = resolve(core, expected_name="B580")
        if device != DEVICE:
            raise ValueError(f"expected B580 at {BDF} as GPU.1, got {device}")
        report["devices"] = devices
        report["selected_device"] = device
        context = core.get_default_context(device)
        stats = _stats(core, device)
        if report["inventory"]["named_plus_reserve_bytes"] > stats["free_bytes"]:
            raise MemoryError("bridge preflight would consume the device reserve")
        report["preflight_gpu_memory"] = stats
        config = _make_plugin_config(args.output)
        report["plugin_config"] = {"name": config.name, "sha256": _sha_file(config),
                                    "bridge_xml": BRIDGE_XML.name, "control_xml": CONTROL_XML.name}

        # First compile and run the non-attention all-finite-word graph.  This
        # proves both custom bridge endpoints without any extreme-word XMX math.
        model = _build_roundtrip_model()
        _check_rss("before_compile_roundtrip", report)
        compiled, compile_report = _compile(core, context, device, model, config,
                                            "roundtrip", ov.Type.dynamic, args.output,
                                            public_inputs=(("q_bf16", bridge.BF16,
                                                            bridge.Q_SHAPE),),
                                            public_output="context_bf16")
        compile_report["serialized_bridge_abi"] = _serialized_bridge_abi(
            args.output / compile_report["runtime_xml"])
        report["roundtrip_compile"] = compile_report
        roundtrip = _run_roundtrip(compiled, context, roundtrip_input)
        _check_rss("after_roundtrip", report)
        if not np.array_equal(roundtrip, roundtrip_input):
            raise ValueError("all-finite BF16 Q-pack/context-unpack roundtrip drifted")
        report["roundtrip"] = {"bitexact": True, "output_sha256": _sha_array(roundtrip),
                               "shape": list(roundtrip.shape)}
        np.savez(args.output / "roundtrip.npz", input=roundtrip_input, output=roundtrip)
        report["roundtrip_archive_sha256"] = _sha_file(args.output / "roundtrip.npz")
        compiled = model = None
        gc.collect()
        report["gpu_memory_after_roundtrip"] = _stats(core, device)

        # Compile the complete fixed bridge graph and the unchanged S1 control
        # sequentially, so their plugin owners do not coexist accidentally.
        model = bridge.build_bridge_model()
        block_proof = _validate_block_constant(model, args.output)
        report["block_constant_proof"] = block_proof
        _check_rss("before_compile_bridge", report)
        compiled, bridge_compile = _compile(core, context, device, model, config,
                                            "bridge", ov.Type.dynamic, args.output,
                                            public_inputs=(("q_bf16", bridge.BF16,
                                                            bridge.Q_SHAPE),
                                                           ("k_bf16", bridge.BF16,
                                                            bridge.KV_SHAPE),
                                                           ("v_bf16", bridge.BF16,
                                                            bridge.KV_SHAPE)),
                                            public_output="context_bf16")
        bridge_compile["serialized_bridge_abi"] = _serialized_bridge_abi(
            args.output / bridge_compile["runtime_xml"], require_attention=True,
            constant_ids_proven=True)
        report["bridge_compile"] = bridge_compile
        bridge_q = {"a": source["q_a"], "b": source["q_b"]}
        bridge_outputs = _run_attention(compiled, context, device, bridge_q,
                                        source["k"], source["v"], data["ids"],
                                        source_layout=True)
        _check_rss("after_bridge", report)
        bridge_control_layout = {
            key: _source_context_to_control(value)
            for key, value in bridge_outputs.items()
        }
        report["bridge_outputs"] = {key: _sha_array(value)
                                    for key, value in bridge_control_layout.items()}
        np.savez(args.output / "bridge_outputs.npz", **bridge_control_layout)
        report["bridge_archive_sha256"] = _sha_file(args.output / "bridge_outputs.npz")
        compiled = model = None
        gc.collect()
        report["gpu_memory_after_bridge"] = _stats(core, device)

        model = kernel.build(HEADS, QUERIES, KEYS, KEYS, canonical_kv=True)
        _check_rss("before_compile_control", report)
        compiled, control_compile = _compile(core, context, device, model,
                                             CONTROL_XML, "control", ov.Type.f16, args.output,
                                             public_inputs=(("Q", ov.Type.f16, Q_SHAPE),
                                                            ("K", ov.Type.f16, KV_SHAPE),
                                                            ("V", ov.Type.f16, KV_SHAPE),
                                                            ("blocks", ov.Type.i32, IDS_SHAPE)),
                                             public_output="attention")
        report["control_compile"] = control_compile
        control_outputs = _run_attention(
            compiled, context, device, {"a": data["q_a"], "b": data["q_b"]},
                                         data["k"], data["v"], data["ids"],
                                         source_layout=False)
        _check_rss("after_control", report)
        report["control_outputs"] = {key: _sha_array(value)
                                     for key, value in control_outputs.items()}
        np.savez(args.output / "control_outputs.npz", **control_outputs)
        report["control_archive_sha256"] = _sha_file(args.output / "control_outputs.npz")
        comparison = {
            key: bool(np.array_equal(control_outputs[key], bridge_control_layout[key]))
            for key in ("a", "b", "a_after_b")
        }
        report["bridge_control_raw_bitexact"] = comparison
        if not all(comparison.values()):
            raise ValueError(f"bridge/control raw output mismatch: {comparison}")
        report["status"] = "pass_bf16_cross_bridge_bitexact"
        report["quality_qualified"] = False
        report["note"] = "Sequential component proof only; no full-model claim"
    except Exception as error:
        report["status"] = "fail"
        report["quality_qualified"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        compiled = model = context = None
        gc.collect()
        if core is not None and report.get("selected_device"):
            try:
                _record_final_stats(report, _stats(core, report["selected_device"]))
            except Exception as error:
                report["status"] = "fail_final_gpu_statistics"
                report["error"] = f"{type(error).__name__}: {error}"
        try:
            report["host_rss_final_bytes"] = _rss_bytes()
            if report["host_rss_final_bytes"] > HOST_RSS_LIMIT_BYTES:
                report["status"] = "fail_host_rss_limit"
                report["error"] = "host RSS exceeded 4-GiB bridge limit after teardown"
        except Exception as error:
            report["status"] = "fail_host_rss_readback"
            report["error"] = f"{type(error).__name__}: {error}"
        report["cleanup"] = "compiled/request/context references released"
        persist()
    if _status_exit_code(report["status"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

