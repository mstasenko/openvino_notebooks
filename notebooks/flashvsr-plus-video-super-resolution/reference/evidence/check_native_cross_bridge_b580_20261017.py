#!/usr/bin/env python3
"""Compare the bridge and legacy cross graphs with authenticated native CPU data.

This is a context-only component check.  It is not a full Plus block or model
quality qualification.  The native receipt is hashed before any NPZ member is
loaded; all three bounded layer fixtures are authenticated before Core
creation, then consumed while the same two compiled graphs are reused.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

BASE = Path(__file__).resolve().parents[1]
EVIDENCE = BASE / "evidence"
sys.path[:0] = [str(BASE / "runtime-openvino-2026.5.0.dev20261014"),
                str(BASE / "openvino"), str(BASE / "openvino" / "block"),
                str(BASE / "openvino" / "full_runtime"), str(EVIDENCE)]

import openvino as ov  # noqa: E402
import bench_bf16_cross_b580_20261017 as component  # noqa: E402


DEVICE = "GPU.1"
BDF = "0000:03:00.0"
LAYERS = (0, 23, 29)
Q_SHAPE = (1, 2048, 1536)
KV_SHAPE = (1, 512, 1536)
NORM_LIMIT = 0.015625
REL_LIMIT = 0.005
DEVICE_RESERVE_BYTES = 2 << 30
HOST_RSS_LIMIT_BYTES = 4 << 30
REQUIRED_FIELDS = ("native_q_rms_bits", "native_cross_k_bits",
                   "native_cross_v_bits", "native_cross_context_bits")


def _reference_receipt(root: Path, expected_sha256: str) -> tuple[Path, dict]:
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF"
                                        for c in expected_sha256):
        raise ValueError("--reference-sha256 must be a 64-digit hexadecimal digest")
    root = root.resolve()
    receipt_path = (root / "results.json").resolve()
    actual = component.bridge_worker._sha_file(receipt_path)
    if actual != expected_sha256.lower():
        raise ValueError("native reference receipt SHA-256 does not match the supplied digest")
    with receipt_path.open("r", encoding="utf-8") as stream:
        receipt = json.load(stream)
    if receipt.get("status") != "pass":
        raise ValueError("native reference receipt is not a passing CPU receipt")
    if receipt.get("selected_layers") != list(LAYERS):
        raise ValueError("native reference selected layer set changed")
    if (receipt.get("device") != "cpu"
            or receipt.get("dtype") != "BF16 raw storage / F32 diagnostic references"
            or receipt.get("base_q_rows") != 256
            or receipt.get("native_executed_q_rows") != 256
            or receipt.get("consumer_q_rows") != 2048
            or receipt.get("repeat_factor") != 8
            or receipt.get("context_rows") != 512
            or receipt.get("heads") != 12
            or receipt.get("head_dim") != 128
            or receipt.get("consumer_rows_are_tiled_only") is not True
            or receipt.get("limits") != {
                "normalized_max_abs": 0.015625, "relative_l2": 0.005}):
        raise ValueError("native reference geometry or policy fields changed")
    layers = receipt.get("layers")
    if not isinstance(layers, list) or [row.get("layer") for row in layers] != list(LAYERS):
        raise ValueError("native reference layer records are incomplete or reordered")
    return receipt_path, receipt


def _load_layer(root: Path, record: dict) -> tuple[dict, dict]:
    layer = record["layer"]
    archive_record = record.get("archive", {})
    archive_path = Path(archive_record.get("path", "")).resolve()
    expected_path = (root.resolve() / f"layer{layer}-native-cross.npz").resolve()
    if archive_path != expected_path or archive_path.name != expected_path.name:
        raise ValueError(f"layer {layer} archive path is not the authenticated leaf")
    if archive_path.stat().st_size != int(archive_record.get("bytes", -1)):
        raise ValueError(f"layer {layer} archive byte count changed")
    if component.bridge_worker._sha_file(archive_path) != archive_record.get("sha256"):
        raise ValueError(f"layer {layer} archive SHA-256 changed")
    metadata = record.get("arrays", {})
    values = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for name in REQUIRED_FIELDS:
            spec = metadata.get(name)
            if not isinstance(spec, dict) or name not in archive.files:
                raise ValueError(f"layer {layer} missing authenticated field {name}")
            value = np.ascontiguousarray(archive[name])
            shape = tuple(int(item) for item in spec.get("shape", ()))
            if (value.dtype != np.uint16 or value.shape != shape
                    or spec.get("dtype") != "uint16"
                    or component.bridge_worker._sha_array(value) != spec.get("sha256")
                    or not component.bridge_worker._finite_bits(value)):
                raise ValueError(f"layer {layer} raw field {name} failed authentication")
            values[name] = value
    if values["native_q_rms_bits"].shape != Q_SHAPE:
        raise ValueError("native q_rms shape is not the 2048-row source ABI")
    if any(values[name].shape != KV_SHAPE
           for name in ("native_cross_k_bits", "native_cross_v_bits")):
        raise ValueError("native cross K/V shape is not the 512-row source ABI")
    if values["native_cross_context_bits"].shape != Q_SHAPE:
        raise ValueError("native cross context shape is not the 2048-row source ABI")
    return values, {name: metadata[name] for name in REQUIRED_FIELDS}


def _authenticate_reference_layers(root: Path, receipt: dict):
    """Load and authenticate all bounded native leaves before device setup."""
    layers, field_metadata = {}, {}
    for record in receipt["layers"]:
        layer = int(record["layer"])
        layers[layer], field_metadata[layer] = _load_layer(root, record)
    first = layers[LAYERS[0]]
    inventory_data = {
        "q_a": first["native_q_rms_bits"],
        "q_b": np.ascontiguousarray(np.roll(first["native_q_rms_bits"], 1, axis=1)),
        "k": first["native_cross_k_bits"],
        "v": first["native_cross_v_bits"],
        "ids": np.broadcast_to(
            np.arange(component.bridge_worker.bridge.BLOCK_SHAPE[-1], dtype=np.int32)
            .reshape(1, 1, 1, -1), component.bridge_worker.bridge.BLOCK_SHAPE).copy(),
    }
    return layers, field_metadata, component._inventory(inventory_data)


def _require_runtime_version(version) -> str:
    version = str(version)
    if not version.startswith("2026.5."):
        raise ValueError(f"pinned OpenVINO 2026.5 required, got {version}")
    return version


def _require_device_admission(inventory: dict, stats: dict) -> None:
    if int(inventory["named_plus_reserve_bytes"]) > int(stats["free_bytes"]):
        raise MemoryError("native cross inventory would consume the device reserve")


def _record_outputs(report: dict, layer: int, expected_a: np.ndarray,
                    expected_b: np.ndarray, bridge_outputs: dict,
                    legacy_outputs: dict) -> dict:
    metrics = {}
    for name, outputs in (("bridge", bridge_outputs), ("legacy", legacy_outputs)):
        for label, expected in (("a", expected_a), ("b", expected_b),
                                ("a_after_b", expected_a)):
            metrics[f"{name}_{label}"] = component._metric(expected, outputs[label])
    report.setdefault("layers", []).append({
        "layer": layer,
        "metrics": metrics,
        "bridge_bitwise_restored": np.array_equal(
            bridge_outputs["a"], bridge_outputs["a_after_b"]),
        "legacy_bitwise_restored": np.array_equal(
            legacy_outputs["a"], legacy_outputs["a_after_b"]),
    })
    return metrics


def _within(metrics: dict) -> bool:
    return all(row["normalized_max_error"] <= NORM_LIMIT
               and row["relative_l2"] <= REL_LIMIT
               for row in metrics.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="explicitly enable pinned B580 execution")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("GPU execution requires explicit --run")
    if args.output.exists():
        parser.error("output directory must be fresh")
    args.output.mkdir(parents=True, exist_ok=False)

    receipt_path, receipt = _reference_receipt(args.reference, args.reference_sha256)
    source_files = (Path(__file__), Path(component.__file__),
                    Path(component.bridge_worker.__file__),
                    Path(component.bridge_worker.bridge.__file__), component.BRIDGE_CL,
                    component.BRIDGE_XML, component.CONTROL_CL, component.CONTROL_XML,
                    component.BASE / "openvino" / "block" / "graph.py",
                    component.BASE / "openvino" / "full_runtime" / "plus_graphs.py",
                    component.BASE / "openvino" / "full_runtime" / "plus_pipeline.py",
                    component.BASE / "openvino" / "attention_from_cuda.py")
    report = {
        "status": "initializing",
        "scope": "authenticated native CPU cross context versus bridge/legacy; context-only component",
        "quality_qualified": False,
        "target_bdf": BDF,
        "requested_gpu_model": "B580",
        "reference_receipt": str(receipt_path),
        "reference_receipt_sha256": args.reference_sha256.lower(),
        "reference_source_commit": receipt.get("source_commit"),
        "reference_checkpoint": receipt.get("checkpoint"),
        "selected_layers": list(LAYERS),
        "known_raw_fields": list(REQUIRED_FIELDS),
        "limits": {"normalized_max_abs": NORM_LIMIT, "relative_l2": REL_LIMIT},
        "source_sha256": {str(path): component.bridge_worker._sha_file(path)
                           for path in source_files},
        "stage_samples": {},
    }

    def persist():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    persist()
    core = context = None
    models, compiled, states = {}, {}, {}
    roots = q_roots = None
    reference_layers, reference_field_metadata = {}, {}
    all_outputs = {}
    try:
        (reference_layers, reference_field_metadata,
         report["inventory"]) = _authenticate_reference_layers(
            args.reference.resolve(), receipt)
        component.bridge_worker._check_rss("after_reference_auth", report)
        report["reference_layers"] = [
            {"layer": int(record["layer"]), "archive": record["archive"],
             "fields": reference_field_metadata[int(record["layer"])]}
            for record in receipt["layers"]]
        persist()

        report["openvino_version"] = _require_runtime_version(ov.get_version())
        core = ov.Core()
        from attention_from_cuda import resolve
        device, devices = resolve(core, expected_name="B580")
        if device != DEVICE:
            raise ValueError(f"expected B580 at {BDF} as GPU.1, got {device}")
        report["devices"] = devices
        report["selected_device"] = device
        context = core.get_default_context(device)
        preflight = component.bridge_worker._stats(core, device)
        report["preflight_gpu_memory"] = preflight
        _require_device_admission(report["inventory"], preflight)

        config = component.bridge_worker._make_plugin_config(args.output)
        report["plugin_config"] = {"name": config.name,
                                    "sha256": component.bridge_worker._sha_file(config)}
        models["bridge"] = component.bridge_worker.bridge.build_bridge_model()
        report["block_constant_proof"] = component.bridge_worker._validate_block_constant(
            models["bridge"], args.output)
        compiled["bridge"], bridge_compile = component.bridge_worker._compile(
            core, context, device, models["bridge"], config, "bridge", ov.Type.dynamic,
            args.output,
            public_inputs=(("q_bf16", component.bridge_worker.bridge.BF16, Q_SHAPE),
                           ("k_bf16", component.bridge_worker.bridge.BF16, KV_SHAPE),
                           ("v_bf16", component.bridge_worker.bridge.BF16, KV_SHAPE)),
            public_output="context_bf16")
        bridge_compile["serialized_bridge_abi"] = component.bridge_worker._serialized_bridge_abi(
            args.output / bridge_compile["runtime_xml"], require_attention=True,
            constant_ids_proven=True)
        bridge_compile["matched_compile_policy"] = component._require_compile_policy(
            compiled["bridge"], "bridge")
        report["bridge_compile"] = bridge_compile
        component._sample_stage(core, device, report, "post_bridge_compile")
        persist()

        models["legacy"] = component._build_legacy_model()
        compiled["legacy"], report["legacy_compile"] = component._compile_legacy(
            core, context, device, models["legacy"], args.output)
        component._sample_stage(core, device, report, "post_legacy_compile")
        persist()
        models.clear()
        gc.collect()

        first = reference_layers[LAYERS[0]]
        roots = {}
        for name, value in (("q_a", first["native_q_rms_bits"]),
                            ("q_b", np.ascontiguousarray(
                                np.roll(first["native_q_rms_bits"], 1, axis=1))),
                            ("k", first["native_cross_k_bits"]),
                            ("v", first["native_cross_v_bits"])):
            roots[name] = context.create_tensor(ov.Type.bf16, ov.Shape(value.shape), {})
            component.bridge_worker._upload_host(roots[name], value, ov.Type.bf16)
        q_roots = {"a": roots["q_a"], "b": roots["q_b"]}
        states["bridge"] = component._bind(
            compiled["bridge"], context,
            {"q_bf16": roots["q_a"], "k_bf16": roots["k"], "v_bf16": roots["v"]},
            "context_bf16", Q_SHAPE)
        states["legacy"] = component._bind(
            compiled["legacy"], context,
            {"q_bf16": roots["q_a"], "k_bf16": roots["k"], "v_bf16": roots["v"]},
            "legacy_context_bf16", Q_SHAPE)
        component._sample_stage(core, device, report, "bound")
        persist()

        for record in receipt["layers"]:
            layer = int(record["layer"])
            values = reference_layers[layer]
            field_metadata = reference_field_metadata[layer]
            q_a = values["native_q_rms_bits"]
            q_b = np.ascontiguousarray(np.roll(q_a, 1, axis=1))
            expected_a = values["native_cross_context_bits"]
            expected_b = np.ascontiguousarray(np.roll(expected_a, 1, axis=1))
            for name, value in (("q_a", q_a), ("q_b", q_b),
                                ("k", values["native_cross_k_bits"]),
                                ("v", values["native_cross_v_bits"])):
                component.bridge_worker._upload_host(roots[name], value, ov.Type.bf16)
            bridge_outputs = component._run_aba(states["bridge"], q_roots)
            legacy_outputs = component._run_aba(states["legacy"], q_roots)
            metrics = _record_outputs(report, layer, expected_a, expected_b,
                                      bridge_outputs, legacy_outputs)
            for name, value in (("native_a", expected_a), ("native_b", expected_b)):
                all_outputs[f"layer{layer}_{name}"] = np.array(value, copy=True)
            for graph_name, outputs in (("bridge", bridge_outputs),
                                        ("legacy", legacy_outputs)):
                for label, value in outputs.items():
                    all_outputs[f"layer{layer}_{graph_name}_{label}"] = value
            report["layers"][-1]["input_field_sha256"] = {
                name: field_metadata[name]["sha256"] for name in REQUIRED_FIELDS}
            report["layers"][-1]["output_sha256"] = {
                name: component.bridge_worker._sha_array(value)
                for name, value in all_outputs.items() if name.startswith(f"layer{layer}_")}
            report["layers"][-1]["within_component_limits"] = _within(metrics)
            component._sample_stage(core, device, report, f"layer{layer}")
            persist()

        np.savez(args.output / "outputs.npz", **all_outputs)
        report["outputs_archive_sha256"] = component.bridge_worker._sha_file(
            args.output / "outputs.npz")
        report["numeric_within_component_limits"] = all(
            row["within_component_limits"] for row in report["layers"])
        report["status"] = ("pass_native_cross_component"
                             if report["numeric_within_component_limits"]
                             else "fail_native_cross_component_drift")
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
        reference_layers.clear()
        reference_field_metadata.clear()
        all_outputs.clear()
        context = None
        gc.collect()
        if core is not None and report.get("selected_device"):
            try:
                report["final_gpu_memory"] = component.bridge_worker._stats(
                    core, report["selected_device"])
                report["final_allocations_zero"] = (
                    report["final_gpu_memory"]["allocated_bytes"] == 0)
                if not report["final_allocations_zero"]:
                    report["status"] = "fail_allocations_not_released"
                    report["error"] = "GPU allocations remained after teardown"
            except Exception as error:
                report["status"] = "fail_final_gpu_statistics"
                report["error"] = f"{type(error).__name__}: {error}"
        report["host_rss_final_bytes"] = component.bridge_worker._rss_bytes()
        if report["host_rss_final_bytes"] > HOST_RSS_LIMIT_BYTES:
            report["status"] = "fail_host_rss_limit"
            report["error"] = "host RSS exceeded component limit after teardown"
        persist()
    if report["status"] != "pass_native_cross_component":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
