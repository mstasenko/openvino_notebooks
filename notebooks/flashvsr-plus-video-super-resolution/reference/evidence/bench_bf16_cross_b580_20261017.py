#!/usr/bin/env python3
"""Component timing comparison for the BF16 bridge and legacy F32 cross graph.

Both graphs consume the same fixed 2K-query/512-key BF16 roots.  Inputs and
outputs stay in one GPU context during warmups/timing; output readback is only
for the untimed A-B-A and numerical component check.  This is not a native
full-model qualification or throughput claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

BASE = Path(__file__).resolve().parents[1]
PINNED_RUNTIME = BASE / "runtime-openvino-2026.5.0.dev20261014"
EVIDENCE = BASE / "evidence"
sys.path[:0] = [
    str(PINNED_RUNTIME), str(BASE / "openvino"),
    str(BASE / "openvino" / "block"),
    str(BASE / "openvino" / "full_runtime"), str(EVIDENCE),
]

import openvino as ov  # noqa: E402
from openvino import opset13 as ops  # noqa: E402

import block.graph as block_graph  # noqa: E402
import check_bf16_cross_bridge_b580_20261017 as bridge_worker  # noqa: E402
import plus_graphs  # noqa: E402
from plus_pipeline import compile_config  # noqa: E402


DEVICE = "GPU.1"
BDF = "0000:03:00.0"
WARMUPS = 50
SAMPLES = 100
NORM_LIMIT = 0.015625
REL_LIMIT = 0.005
DEVICE_RESERVE_BYTES = 2 << 30
UNKNOWN_PLUGIN_WORKSPACE_ALLOWANCE_BYTES = 2 << 30
HOST_RSS_LIMIT_BYTES = 4 << 30
BRIDGE_XML = EVIDENCE / "bf16_cross_bridge_20261017.xml"
BRIDGE_CL = EVIDENCE / "bf16_cross_bridge_20261017.cl"
CONTROL_XML = BASE / "openvino" / "chunked_attention" / "plus_sparse_xmx.xml"
CONTROL_CL = BASE / "openvino" / "chunked_attention" / "plus_sparse_xmx.cl"


def _build_legacy_model():
    """Build the existing F32 cross-attention math with BF16 storage edges."""
    q = ops.parameter([1, 2048, 1536], ov.Type.f16, name="q_bf16")
    k = ops.parameter([1, 512, 1536], ov.Type.f16, name="k_bf16")
    v = ops.parameter([1, 512, 1536], ov.Type.f16, name="v_bf16")
    context = block_graph._cross_attention(ops, q, k, v, 2048)
    context.output(0).get_tensor().set_names({"legacy_context_bf16"})
    model = ov.Model([context], [q, k, v], "legacy_f32_cross")
    model.validate_nodes_and_infer_types()
    model = plus_graphs.with_bf16_storage(model)
    model.output(0).get_tensor().set_names({"legacy_context_bf16"})
    model.validate_nodes_and_infer_types()
    return model


def _legacy_compile_options(device):
    options = dict(compile_config(ov, device))
    # Match the bridge's explicit dynamic/ACCURACY policy while retaining the
    # existing legacy compile_config stream/performance settings.
    options["INFERENCE_PRECISION_HINT"] = ov.Type.dynamic
    return options


def _compile_legacy(core, context, device, model, output: Path):
    options = _legacy_compile_options(device)
    started = time.perf_counter()
    compiled = core.compile_model(model, context, options)
    seconds = time.perf_counter() - started
    if list(compiled.get_property("EXECUTION_DEVICES")) != [device]:
        raise ValueError("legacy graph compiled on an unintended device")
    runtime = output / "runtime-legacy.xml"
    ov.serialize(compiled.get_runtime_model(), runtime)
    public = bridge_worker._compiled_public_abi(
        compiled,
        (("q_bf16", ov.Type.bf16, (1, 2048, 1536)),
         ("k_bf16", ov.Type.bf16, (1, 512, 1536)),
         ("v_bf16", ov.Type.bf16, (1, 512, 1536))),
        "legacy_context_bf16",
    )
    policy = _require_compile_policy(compiled, "legacy")
    return compiled, {
        "compile_seconds": seconds,
        "runtime_xml": runtime.name,
        "runtime_xml_sha256": bridge_worker._sha_file(runtime),
        "compile_options": {str(key): str(value) for key, value in options.items()},
        "requested_precision": str(options["INFERENCE_PRECISION_HINT"]),
        "requested_execution_mode": str(options["EXECUTION_MODE_HINT"]),
        "actual_precision": str(policy["actual_precision"]),
        "actual_execution_mode": str(policy["actual_execution_mode"]),
        "matched_compile_policy": policy,
        "execution_devices": list(compiled.get_property("EXECUTION_DEVICES")),
        "public_abi": public,
    }


def _inventory(data: dict[str, np.ndarray]) -> dict:
    base = bridge_worker._inventory(data)
    input_bytes = sum(int(data[name].nbytes)
                      for name in ("q_a", "q_b", "k", "v"))
    output_bytes = int(np.prod(bridge_worker.bridge.Q_SHAPE, dtype=np.int64) * 2)
    two_requests = 2 * (input_bytes + output_bytes)
    q_f32 = int(np.prod((1, 2048, 1536), dtype=np.int64) * 4)
    kv_f32_each = int(np.prod((1, 512, 1536), dtype=np.int64) * 4)
    score_f32_each = int(12 * 2048 * 512 * 4)
    legacy_f32_tensors = q_f32 + 2 * kv_f32_each
    legacy_f32_attention = 3 * score_f32_each + 2 * q_f32
    legacy_f32_workspace = legacy_f32_tensors + legacy_f32_attention
    bridge_allowance = int(base["bridge_named_bytes"]["conservative_bridge_bytes"])
    named = (input_bytes + 2 * output_bytes + two_requests
             + bridge_allowance + legacy_f32_workspace
             + UNKNOWN_PLUGIN_WORKSPACE_ALLOWANCE_BYTES)
    return {
        "shared_bf16_input_roots_bytes": input_bytes,
        "q_root_count": 2,
        "two_output_roots_bytes": 2 * output_bytes,
        "two_request_allowance_bytes": two_requests,
        "bridge_compiled_graph_allowance_bytes": bridge_allowance,
        "legacy_f32_tensor_bytes": legacy_f32_tensors,
        "legacy_f32_score_context_output_bytes": legacy_f32_attention,
        "legacy_f32_workspace_allowance_bytes": legacy_f32_workspace,
        "unknown_plugin_workspace_allowance_bytes": UNKNOWN_PLUGIN_WORKSPACE_ALLOWANCE_BYTES,
        "unknown_plugin_workspace_allowance_role": (
            "single conservative 2GiB allowance for unclassified plugin/workspace storage"),
        "device_reserve_bytes": DEVICE_RESERVE_BYTES,
        "named_plus_reserve_bytes": int(named + DEVICE_RESERVE_BYTES),
        "bridge_worker_reference": base,
    }


def _property_matches(actual, expected) -> bool:
    return actual == expected or str(actual) == str(expected)


def _require_compile_policy(compiled, label: str) -> dict:
    actual_precision = compiled.get_property("INFERENCE_PRECISION_HINT")
    actual_execution_mode = compiled.get_property("EXECUTION_MODE_HINT")
    if not _property_matches(actual_precision, ov.Type.dynamic):
        raise ValueError(f"{label} compiled with non-dynamic precision: {actual_precision}")
    expected_mode = ov.properties.hint.ExecutionMode.ACCURACY
    if not _property_matches(actual_execution_mode, expected_mode):
        raise ValueError(
            f"{label} compiled with non-ACCURACY execution mode: {actual_execution_mode}")
    return {"expected_precision": str(ov.Type.dynamic),
            "expected_execution_mode": str(expected_mode),
            "actual_precision": str(actual_precision),
            "actual_execution_mode": str(actual_execution_mode)}


def _record_stage_sample(report: dict, label: str, stats: dict, rss_bytes: int) -> dict:
    """Record point-in-time safety samples; these are not allocator peaks."""
    sample = {"measurement_scope": "point-in-time sample, not a proven peak",
              "gpu_memory": stats, "host_rss_bytes": int(rss_bytes)}
    report.setdefault("stage_samples", {})[label] = sample
    if not 0 <= rss_bytes <= HOST_RSS_LIMIT_BYTES:
        raise MemoryError(f"{label} exceeds the 4GiB host RSS limit: {rss_bytes}")
    free_bytes = int(stats["free_bytes"])
    if free_bytes < DEVICE_RESERVE_BYTES:
        raise MemoryError(
            f"{label} leaves less than the 2GiB device reserve: {free_bytes}")
    return sample


def _sample_stage(core, device: str, report: dict, label: str) -> dict:
    return _record_stage_sample(report, label, bridge_worker._stats(core, device),
                                bridge_worker._rss_bytes())


def _bind(compiled, context, roots, output_name, output_shape):
    request = compiled.create_infer_request()
    for name, tensor in roots.items():
        request.set_tensor(name, tensor)
    output = context.create_tensor(ov.Type.bf16, ov.Shape(output_shape), {})
    request.set_tensor(output_name, output)
    return {"request": request, "output": output, "q_name": "q_bf16",
            "output_name": output_name, "output_shape": tuple(output_shape)}


def _infer(state, q_root) -> float:
    state["request"].set_tensor(state["q_name"], q_root)
    started = time.perf_counter()
    state["request"].start_async()
    state["request"].wait()
    return time.perf_counter() - started


def _read(state) -> np.ndarray:
    return bridge_worker._copy_output(
        state["output"], ov.Type.bf16, state["output_shape"])


def _run_aba(state, q_roots) -> dict[str, np.ndarray]:
    outputs = {}
    for label, root in (("a", q_roots["a"]), ("b", q_roots["b"]),
                        ("a_after_b", q_roots["a"])):
        _infer(state, root)
        outputs[label] = _read(state)
    if np.array_equal(outputs["a"], outputs["b"]):
        raise ValueError("graph ignored changed B input")
    if not np.array_equal(outputs["a"], outputs["a_after_b"]):
        raise ValueError("graph failed raw A-B-A restoration")
    return outputs


def _metric(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected64 = bridge_worker.bridge.bf16_bits_to_f32(
        expected, bridge_worker.bridge.Q_SHAPE).astype(np.float64)
    actual64 = bridge_worker.bridge.bf16_bits_to_f32(
        actual, bridge_worker.bridge.Q_SHAPE).astype(np.float64)
    delta = actual64 - expected64
    max_abs = float(np.max(np.abs(delta)))
    return {
        "max_abs": max_abs,
        "normalized_max_error": max_abs / max(float(np.max(np.abs(expected64))), 1e-30),
        "relative_l2": float(np.linalg.norm(delta)
                              / max(float(np.linalg.norm(expected64)), 1e-30)),
    }


def _run_timing(states, q_roots) -> dict:
    for _ in range(WARMUPS):
        _infer(states["bridge"], q_roots["a"])
        _infer(states["legacy"], q_roots["a"])
    values = {"bridge": [], "legacy": []}
    orders = []
    for index in range(SAMPLES):
        order = ("bridge", "legacy") if index % 2 == 0 else ("legacy", "bridge")
        orders.append(list(order))
        for name in order:
            values[name].append(_infer(states[name], q_roots["a"]))
    return {
        "warmups_per_variant": WARMUPS,
        "paired_samples": SAMPLES,
        "orders": orders,
        "interval": "request.start_async through request.wait; readback excluded",
        "seconds": values,
        "summary": {
            name: {"mean": float(np.mean(rows)), "median": float(np.median(rows)),
                   "min": float(np.min(rows)), "max": float(np.max(rows))}
            for name, rows in values.items()
        },
    }


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
    source_files = (
        Path(__file__), Path(bridge_worker.__file__), bridge_worker.BRIDGE_PY,
        bridge_worker.BRIDGE_CL, bridge_worker.BRIDGE_XML,
        BASE / "openvino" / "block" / "graph.py",
        BASE / "openvino" / "full_runtime" / "plus_graphs.py",
        BASE / "openvino" / "full_runtime" / "plus_pipeline.py",
        BASE / "openvino" / "chunked_attention" / "plus_sparse_xmx.py",
        CONTROL_CL, CONTROL_XML, BASE / "openvino" / "attention_from_cuda.py",
    )
    report = {
        "status": "initializing",
        "scope": "BF16 bridge versus legacy F32 cross component timing; no full-model qualification",
        "quality_qualified": False,
        "target_bdf": BDF,
        "requested_gpu_model": "B580",
        "timing_contract": "50 warmups each, then 100 paired alternating bridge/legacy orders; GPU-resident inputs/output, readback excluded",
        "thresholds": {"normalized_max_error": NORM_LIMIT, "relative_l2": REL_LIMIT},
        "stage_samples": {},
        "source_sha256": {str(path): bridge_worker._sha_file(path) for path in source_files},
    }

    def persist():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    persist()
    core = context = None
    models = {}
    compiled = {}
    states = {}
    roots = q_roots = None
    try:
        data = bridge_worker._realistic_inputs()
        source = bridge_worker._source_inputs(data)
        report["input_sha256"] = {name: bridge_worker._sha_array(value)
                                  for name, value in source.items()}
        np.savez(args.output / "inputs.npz", q_a=source["q_a"], q_b=source["q_b"],
                 k=source["k"], v=source["v"], ids=data["ids"])
        report["inputs_archive_sha256"] = bridge_worker._sha_file(args.output / "inputs.npz")
        report["inventory"] = _inventory(data)
        bridge_worker._check_rss("after_inputs", report)
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
        preflight = bridge_worker._stats(core, device)
        report["preflight_gpu_memory"] = preflight
        if report["inventory"]["named_plus_reserve_bytes"] > preflight["free_bytes"]:
            raise MemoryError("component timing inventory would consume the device reserve")
        config = bridge_worker._make_plugin_config(args.output)
        report["plugin_config"] = {"name": config.name,
                                    "sha256": bridge_worker._sha_file(config)}

        models["bridge"] = bridge_worker.bridge.build_bridge_model()
        bridge_proof = bridge_worker._validate_block_constant(models["bridge"], args.output)
        report["block_constant_proof"] = bridge_proof
        compiled["bridge"], bridge_compile = bridge_worker._compile(
            core, context, device, models["bridge"], config, "bridge",
            ov.Type.dynamic, args.output,
            public_inputs=(("q_bf16", bridge_worker.bridge.BF16,
                            bridge_worker.bridge.Q_SHAPE),
                           ("k_bf16", bridge_worker.bridge.BF16,
                            bridge_worker.bridge.KV_SHAPE),
                           ("v_bf16", bridge_worker.bridge.BF16,
                            bridge_worker.bridge.KV_SHAPE)),
            public_output="context_bf16")
        bridge_compile["serialized_bridge_abi"] = bridge_worker._serialized_bridge_abi(
            args.output / bridge_compile["runtime_xml"], require_attention=True,
            constant_ids_proven=True)
        bridge_compile["matched_compile_policy"] = _require_compile_policy(
            compiled["bridge"], "bridge")
        report["bridge_compile"] = bridge_compile
        _sample_stage(core, device, report, "post_bridge_compile")
        persist()

        models["legacy"] = _build_legacy_model()
        compiled["legacy"], report["legacy_compile"] = _compile_legacy(
            core, context, device, models["legacy"], args.output)
        _sample_stage(core, device, report, "post_legacy_compile")
        persist()
        models.clear()
        gc.collect()

        roots = {}
        for name, value in (("q_a", source["q_a"]), ("q_b", source["q_b"]),
                            ("k", source["k"]), ("v", source["v"])):
            root = context.create_tensor(ov.Type.bf16, ov.Shape(value.shape), {})
            bridge_worker._upload_host(root, value, ov.Type.bf16)
            roots[name] = root
        q_roots = {"a": roots["q_a"], "b": roots["q_b"]}
        states["bridge"] = _bind(
            compiled["bridge"], context,
            {"q_bf16": roots["q_a"], "k_bf16": roots["k"], "v_bf16": roots["v"]},
            "context_bf16", bridge_worker.bridge.Q_SHAPE)
        states["legacy"] = _bind(
            compiled["legacy"], context,
            {"q_bf16": roots["q_a"], "k_bf16": roots["k"], "v_bf16": roots["v"]},
            "legacy_context_bf16", bridge_worker.bridge.Q_SHAPE)
        _sample_stage(core, device, report, "bound")
        persist()
        bridge_outputs = _run_aba(states["bridge"], q_roots)
        legacy_outputs = _run_aba(states["legacy"], q_roots)
        report["aba"] = {"bridge_sha256": {k: bridge_worker._sha_array(v)
                                            for k, v in bridge_outputs.items()},
                          "legacy_sha256": {k: bridge_worker._sha_array(v)
                                            for k, v in legacy_outputs.items()},
                          "bridge_bitwise_restored": True,
                          "legacy_bitwise_restored": True}
        report["bridge_vs_legacy"] = {
            label: _metric(legacy_outputs[label], bridge_outputs[label])
            for label in ("a", "b", "a_after_b")
        }
        report["numerical_within_component_limits"] = all(
            row["normalized_max_error"] <= NORM_LIMIT
            and row["relative_l2"] <= REL_LIMIT
            for row in report["bridge_vs_legacy"].values())
        np.savez(args.output / "aba_outputs.npz",
                 **{f"bridge_{k}": v for k, v in bridge_outputs.items()},
                 **{f"legacy_{k}": v for k, v in legacy_outputs.items()})
        report["aba_archive_sha256"] = bridge_worker._sha_file(args.output / "aba_outputs.npz")
        _sample_stage(core, device, report, "post_aba")
        persist()
        _sample_stage(core, device, report, "pre_timing")
        report["timing"] = _run_timing(states, q_roots)
        _sample_stage(core, device, report, "post_timing")
        persist()
        report["status"] = ("pass_component_bridge_legacy"
                             if report["numerical_within_component_limits"]
                             else "fail_component_numeric_drift")
    except Exception as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        states.clear()
        compiled.clear()
        models.clear()
        if isinstance(q_roots, dict):
            q_roots.clear()
        if isinstance(roots, dict):
            roots.clear()
        q_roots = roots = None
        root = None
        roots = None
        context = None
        gc.collect()
        if core is not None and report.get("selected_device"):
            try:
                report["final_gpu_memory"] = bridge_worker._stats(
                    core, report["selected_device"])
                report["final_allocations_zero"] = (
                    report["final_gpu_memory"]["allocated_bytes"] == 0)
                if not report["final_allocations_zero"]:
                    report["status"] = "fail_allocations_not_released"
                    report["error"] = "GPU allocations remained after teardown"
            except Exception as error:
                report["status"] = "fail_final_gpu_statistics"
                report["error"] = f"{type(error).__name__}: {error}"
        report["host_rss_final_bytes"] = bridge_worker._rss_bytes()
        if report["host_rss_final_bytes"] > HOST_RSS_LIMIT_BYTES:
            report["status"] = "fail_host_rss_limit"
            report["error"] = "host RSS exceeded component limit after teardown"
        persist()
    if report["status"] != "pass_component_bridge_legacy":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
