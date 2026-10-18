"""Finite real-video Plus trajectory check, not a UHD throughput benchmark.

The native capture supplies identical source pixels/noise and a per-layer
oracle. All learned operations run on the exact B580; small host copies keep
the first integration failure localizable before adopting resident storage.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from reference_paths import SOURCE_ROOT

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parents[1]
SOURCE = SOURCE_ROOT
sys.path.insert(0, str(TRIAL / "openvino"))
sys.path.insert(0, str(TRIAL / "openvino/chunked_attention"))

from frame_geometry import FrameGeometry
from plus_edges import SOURCE_COMMIT, load_edge_weights, build_prefix, build_suffix
from plus_graphs import BlockGeometry, build_block, preserve_modulation_product
from plus_masks import build_stream_sparse_layout
from plus_pipeline import Graph, SparseXmx, join_history, unpartition_context, pack_decoder_inputs
from plus_projection import load_weights, build_projection_graph
from plus_storage import storage_tensor, tensor_array
from plus_stream import build_stream_plan
from plus_text import load_prompt, load_text_weights, build_text_embedding, build_cross_kv
from plus_weights import CHECKPOINT, authenticate_dit, load_layer, ROLE_KEYS, EXTRA_LAYER_KEYS
from prepare_real import parameter_modules


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_reference(root, manifest_sha256):
    root = Path(root).resolve()
    manifest = root / "manifest.json"
    if sha(manifest) != manifest_sha256:
        raise ValueError("Native capture manifest changed")
    receipt = json.loads(manifest.read_text())
    if (receipt.get("format") != "flashvsr-plus-native-video-cpu-v1"
            or receipt.get("status") != "cpu_native_denoiser_capture_complete"
            or receipt.get("source", {}).get("commit") != SOURCE_COMMIT
            or receipt.get("dtype") != "torch.bfloat16"
            or receipt.get("input", {}).get("geometry") != [128, 128]
            or receipt.get("input", {}).get("consumed_frame_count") != 17):
        raise ValueError("Native capture provenance/geometry changed")
    records = [receipt["archive"]["inputs"], *receipt["archive"]["calls"]]
    if len(records) != 3:
        raise ValueError("Expected inputs plus exactly two native calls")
    loaded = []
    for name, record in zip(("inputs.npz", "call0.npz", "call1.npz"), records):
        path = root / name
        if Path(record["path"]).resolve() != path or sha(path) != record["sha256"]:
            raise ValueError("Native archive changed: " + name)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(record["arrays"]):
                raise ValueError("Native archive field names changed")
            values = {}
            for field, metadata in record["arrays"].items():
                value = np.array(archive[field], copy=True)
                digest = hashlib.sha256(value.tobytes()).hexdigest()
                if (list(value.shape) != metadata["shape"] or str(value.dtype) != metadata["dtype"]
                        or digest != metadata["sha256"] or not np.isfinite(value).all()):
                    raise ValueError("Native field changed: " + name + "/" + field)
                values[field] = value
            loaded.append(values)
    if loaded[0]["projection_video_f32"].shape != (1, 3, 33, 128, 128):
        raise ValueError("Expected bounded native source video")
    for values in loaded[1:]:
        for name, shape in (("noise_f32", (1, 16, 4, 16, 16)),
                            ("lq_condition0_f32", (1, 256, 1536)),
                            ("patch_tokens_f32", (1, 256, 1536)),
                            ("rope_cos_f32", (1, 256, 1, 64)),
                            ("rope_sin_f32", (1, 256, 1, 64))):
            if values[name].shape != shape or values[name].dtype != np.float32:
                raise ValueError("Native stream ABI changed: " + name)
        for layer in range(30):
            if values[f"layer{layer}_out_f32"].shape != (1, 256, 1536):
                raise ValueError("Native layer output ABI changed")
    return loaded, receipt


def metric(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if (expected.shape != actual.shape or not np.isfinite(expected).all()
            or not np.isfinite(actual).all()):
        raise ValueError("Non-finite or differently shaped Plus output")
    difference = actual.astype(np.float64) - expected
    normalized = float(np.abs(difference).max() / max(float(np.abs(expected).max()), 1e-30))
    relative = float(np.linalg.norm(difference) / max(float(np.linalg.norm(expected.astype(np.float64))), 1e-30))
    return {"normalized_max_abs": normalized, "relative_l2": relative,
            "pass": normalized <= .015625 and relative <= .005}


def mask_inputs(layer, prepared, merged, actual, native):
    """Snapshot the first branch divergence, independently of numeric gates.

    Earlier numeric failures can precede a mask change. Keep the exact pooled
    inputs (including history) so CPU source replay can distinguish a changed
    model state from a mask-adapter arithmetic bug.
    """
    return {"first_mask_" + name: np.array(value, copy=True) for name, value in {
        "layer": np.int32(layer), "q_pooled": prepared["q_pooled"],
        "k_pooled": merged["pooled"], "q_windows": prepared["q_windows"],
        "k_windows": merged["k"], "v_windows": merged["v"],
        "actual": actual, "native": native,
    }.items()}


def observe_text_storage(model):
    """Diagnostic source Linear/GELU boundaries; not extra text processing."""
    gelu, = [node for node in model.get_ordered_ops() if node.get_type_name() == "Gelu"]
    linear = gelu.input_value(0)
    for name, value in (("text_linear", linear), ("text_gelu", gelu.output(0))):
        value.get_tensor().set_names({name})
        if name not in {port.get_any_name() for port in model.outputs}:
            model.add_outputs([value])
    model.validate_nodes_and_infer_types()
    return model


def decode_stream(core, device, packed_calls, output):
    """Complete the tiny actual stream using the qualified stateful decoder."""
    from plus_decoder import load_decoder, build_decoder_graph, STATE_NAMES
    from decode_plus_video_cpu import authenticate_decoder_reference, _quality_metrics
    native_final, reference_receipt = authenticate_decoder_reference()
    decoder, state, decoder_receipt = load_decoder()
    del state
    model = build_decoder_graph(decoder)
    graph = Graph(core, model, device)
    memories = {name: np.zeros(tuple(model.input(name).shape), np.float32) for name in STATE_NAMES}
    chunks = []
    for call, packed in enumerate(packed_calls):
        frames = []
        for inner in range(4):
            values = graph({"packed": packed[:, inner], **memories})
            memories = {name: values[name] for name in STATE_NAMES}
            frames.append(values["frames"])
        raw = np.concatenate(frames, axis=1)
        if call == 0:
            raw = raw[:, 3:]
        chunks.append(raw)
        np.savez_compressed(output / f"decoded_call{call}.npz", frames=raw, **memories)
    raw = np.concatenate(chunks, axis=1)[:, :17]
    rounded = lambda x: tensor_array(storage_tensor(x, "bf16"))
    normalized = rounded(rounded(raw * np.float32(2)) - np.float32(1))
    final_path = output / "decoded_final.npz"
    np.savez_compressed(final_path, normalized_final_f32=normalized)
    quality = _quality_metrics(normalized, native_final)
    return {"scope": "tiny actual OV denoiser+decoder; CPU reference comparison is not UHD quality qualification",
            "decoder": decoder_receipt, "reference": reference_receipt,
            "normalized_shape": list(normalized.shape), "final_sha256": sha(final_path),
            "cpu_reference_comparison": quality}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-inputs", action="store_true",
                        help="diagnostic only: reset each layer to its native input/conditioning")
    parser.add_argument("--reference-masks", action="store_true",
                        help="diagnostic only: use native sparse choices to isolate branch amplification")
    parser.add_argument("--capture-layer", type=int, choices=range(30),
                        help="save one layer's exact pre/post inputs and outputs for CPU localization")
    parser.add_argument("--pre-boundaries", action="store_true",
                        help="diagnostic only: expose named source pre boundaries for localization")
    parser.add_argument("--text-boundaries", action="store_true",
                        help="diagnostic only: observe source text Linear and GELU storage")
    parser.add_argument("--decode", action="store_true",
                        help="finish the tiny actual stream with stateful B580 decoding")
    parser.add_argument("--pre-product-store", action="store_true",
                        help="preserve source pre modulation multiply before shift addition")
    parser.add_argument("--pre-linear-stores", action="store_true",
                        help="preserve source Q/K Linear storage before RMS normalization")
    parser.add_argument("--post-product-store", action="store_true",
                        help="preserve the same source multiplication before FFN modulation shift")
    parser.add_argument("--pre-norm-stores", action="store_true",
                        help="preserve source norm1 and Q/K RMS outputs before following arithmetic")
    args = parser.parse_args()
    if not args.run or args.output.exists():
        parser.error("Explicit --run and a fresh output leaf required")
    if args.pre_linear_stores and not args.pre_product_store:
        parser.error("Linear storage control requires source modulation product storage")
    if args.pre_norm_stores and not args.pre_linear_stores:
        parser.error("Normalization storage control requires source Q/K Linear storage")
    if args.decode and (args.reference_inputs or args.reference_masks):
        parser.error("Full decoding requires actual model inputs and generated masks")
    if subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip() != SOURCE_COMMIT:
        raise ValueError("Official Plus source revision changed")
    if subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True).strip():
        raise ValueError("Official Plus source has local changes")
    references, native_receipt = load_reference(args.reference, args.reference_sha256)
    inputs, *calls = references
    weight_receipt = authenticate_dit()
    prompt, prompt_receipt = load_prompt()
    if not np.array_equal(prompt, inputs["prompt_context_f32"]):
        raise ValueError("Native capture prompt differs from authenticated prompt")
    import openvino as ov
    from safetensors import safe_open
    from attention_from_cuda import resolve
    if not ov.get_version().startswith("2026.5."):
        raise ValueError("Qualified isolated OpenVINO runtime required")
    core = ov.Core()
    device, inventory = resolve(core)
    if device != "GPU.1":
        raise ValueError("Exact B580 resolution changed")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "worker-source.py")
    report = {"status": "initializing", "scope": __doc__, "runtime": ov.get_version(),
              "device": device, "inventory": inventory, "source_commit": SOURCE_COMMIT,
              "reference_manifest_sha256": args.reference_sha256,
              "reference_checkpoints": native_receipt["checkpoints"],
              "weights": weight_receipt, "prompt": prompt_receipt,
              "limits": {"normalized_max_abs": .015625, "relative_l2": .005},
              "worker_sha256": sha(__file__), "calls": []}
    report["diagnostic_controls"] = {"reference_inputs": args.reference_inputs,
                                     "reference_masks": args.reference_masks,
                                     "capture_layer": args.capture_layer,
                                     "pre_boundaries": args.pre_boundaries}
    report["diagnostic_controls"]["text_boundaries"] = args.text_boundaries
    report["decode_requested"] = args.decode
    report["diagnostic_controls"]["pre_product_store"] = args.pre_product_store
    report["diagnostic_controls"]["pre_linear_stores"] = args.pre_linear_stores
    report["diagnostic_controls"]["post_product_store"] = args.post_product_store
    report["diagnostic_controls"]["pre_norm_stores"] = args.pre_norm_stores
    def persist():
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def rounded(value):
        return tensor_array(storage_tensor(value, "bf16"))
    geometry = BlockGeometry(FrameGeometry(128, 128))
    contract, _, _ = parameter_modules()
    plan = build_stream_plan(17, 30)
    histories = [None] * 30
    packed_calls = []
    start = time.perf_counter()
    try:
        edges, _ = load_edge_weights(verify_hash=False)  # authenticated above
        prefix = Graph(core, build_prefix(edges, geometry), device)
        suffix = Graph(core, build_suffix(edges, geometry), device)
        pre_model = build_block("pre", geometry.frame)
        if args.pre_product_store:
            preserve_modulation_product(pre_model, linear_stores=args.pre_linear_stores,
                                        norm_stores=args.pre_norm_stores)
        if args.pre_boundaries:
            from check_plus_block_cpu import _diagnostic_nodes
            _diagnostic_nodes(pre_model, "pre", geometry.token_shape, contract)
        pre = Graph(core, pre_model, device)
        post_model = build_block("post", geometry.frame)
        if args.post_product_store:
            preserve_modulation_product(post_model, "post")
        post = Graph(core, post_model, device)
        cross = Graph(core, build_cross_kv(), device)
        with safe_open(str(CHECKPOINT), framework="np", device="cpu") as handle:
            text_model = build_text_embedding(load_text_weights(handle))
            if args.text_boundaries:
                observe_text_storage(text_model)
            text = Graph(core, text_model, device)
            text_values = text({"prompt": prompt})
            context = text_values["context"]
            np.savez_compressed(args.output / "text.npz", **text_values)
            report["text_raw_sha256"] = sha(args.output / "text.npz")
            report["text_context"] = metric(inputs["context_embedded_f32"], context)
            if args.reference_inputs:
                context = inputs["context_embedded_f32"]
        # The projection keeps its own two causal states. No model inference
        # in this check uses the native reference's conditioning as a shortcut.
        projection_weights = load_weights()
        startup_keys = ("conv1.bias", "conv1.weight", "norm1.weight")
        startup = Graph(core, build_projection_graph({key: projection_weights[key] for key in startup_keys}, True), device)
        projection = Graph(core, build_projection_graph(projection_weights, False), device)
        del projection_weights
        gc.collect()
        video = inputs["projection_video_f32"]
        startup_values = startup({"video": video[:, :, :1]})
        projection_state = {name: startup_values[name] for name in ("cache1", "cache2")}
        del startup
        attention = SparseXmx(core, device)
        with safe_open(str(CHECKPOINT), framework="np", device="cpu") as handle:
            for call, reference in enumerate(calls):
                call_start = time.perf_counter()
                chunks = []
                for inner in range(4):
                    frame_start = 1 + (4 * call + inner) * 4
                    values = projection({"video": video[:, :, frame_start:frame_start + 4], **projection_state})
                    chunks.append(values.pop("projected"))
                    projection_state = {name: values[name] for name in ("cache1", "cache2")}
                condition = np.concatenate(chunks, axis=1)
                boundary = prefix({geometry.name + "_x": reference["noise_f32"]})
                x = boundary[geometry.name + "_tokens_out"]
                t = boundary[geometry.name + "_t_out"]
                modulation = boundary[geometry.name + "_t_mod_out"]
                scores = {"projection": metric(reference["lq_condition0_f32"], condition),
                          "patch": metric(reference["patch_tokens_f32"], x),
                          "time_embedding": metric(inputs["time_embedding_f32"], t),
                          "time_modulation": metric(inputs["time_modulation_f32"], modulation)}
                x = rounded(x + condition)
                raw = {"condition": condition, "patch": boundary[geometry.name + "_tokens_out"],
                       "t": t, "t_mod": modulation}
                layers = []
                first_failure = None
                first_mask_failure = None
                if args.reference_inputs:
                    t, modulation = inputs["time_embedding_f32"], inputs["time_modulation_f32"]
                for layer in range(30):
                    if args.reference_inputs:
                        x = reference[f"layer{layer}_in_f32"]
                    weights = load_layer(handle, layer)
                    pre.bind({contract.parameter_port(key): weights[key] for key in ROLE_KEYS["pre"]})
                    post.bind({contract.parameter_port(key): weights[key] for key in ROLE_KEYS["post"]})
                    cross.bind({"weight__" + key.replace(".", "_"): weights[key] for key in EXTRA_LAYER_KEYS})
                    cross_values = cross({"context": context})
                    prepared = pre({"x": x, "t_mod": modulation,
                                    "rope_cos": reference["rope_cos_f32"], "rope_sin": reference["rope_sin_f32"]})
                    global_layer = layer % 6 == 0
                    retain = 3 if global_layer else 1
                    current = {"k": prepared["k_windows"], "v": prepared["v_windows"], "pooled": prepared["k_pooled"]}
                    merged, histories[layer] = join_history(current, histories[layer], spatial_windows=1,
                                                           retain=retain, write=not plan.call(call).last)
                    layout = build_stream_sparse_layout(prepared["q_pooled"], merged["pooled"],
                        temporal_windows=2, spatial_grid=(1, 1), cache_temporal_groups=retain,
                        local_range=11 if global_layer else 3, is_sparse=True)
                    native_mask = reference[f"layer{layer}_source_mask"][0]
                    used_mask = native_mask if args.reference_masks else layout.block_mask
                    attended = attention(prepared["q_windows"], merged["k"], merged["v"], used_mask)
                    self_context = unpartition_context(attended, (4, 8, 8))
                    feeds = {"x": x, "self_context": self_context, **cross_values,
                             **{name: prepared[name] for name in ("gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")}}
                    x = post(feeds)["x_out"]
                    layer_score = metric(reference[f"layer{layer}_out_f32"], x)
                    mask_same = np.array_equal(native_mask, layout.block_mask)
                    cache_same = np.array_equal(reference[f"layer{layer}_cache_indices"][0], layout.cache_indices)
                    if first_mask_failure is None and not mask_same:
                        first_mask_failure = layer
                        raw.update(mask_inputs(layer, prepared, merged, layout.block_mask, native_mask))
                    layers.append({"layer": layer, **layer_score, "mask_exact": mask_same,
                                   "cache_indices_exact": cache_same})
                    if first_failure is None and not (layer_score["pass"] and mask_same and cache_same):
                        first_failure = layer
                        raw.update({"first_failure_" + name: value for name, value in feeds.items()})
                        raw.update({"first_failure_pre_" + name: value for name, value in prepared.items()})
                        raw.update(first_failure_merged_k=merged["k"], first_failure_merged_v=merged["v"])
                    raw[f"layer{layer}_out"] = x
                    raw[f"layer{layer}_mask"] = layout.block_mask
                    if layer == args.capture_layer:
                        raw.update({"captured_" + name: value for name, value in feeds.items()})
                        raw.update({"captured_pre_" + name: value for name, value in prepared.items()})
                        raw.update(captured_merged_k=merged["k"], captured_merged_v=merged["v"])
                    print(json.dumps({"call": call, "layer": layer, **layers[-1]}), flush=True)
                prediction = suffix({geometry.name + "_tokens_in": x, geometry.name + "_t_in": t})[geometry.name + "_out"]
                denoised = rounded(reference["noise_f32"] - prediction)
                if args.decode:
                    start_frame, stop_frame = ((0, 13), (13, 29))[call]
                    packed_calls.append(pack_decoder_inputs(video[:, :, start_frame:stop_frame], denoised))
                scores.update(prediction=metric(reference["prediction_f32"], prediction),
                              denoised=metric(reference["denoised_f32"], denoised))
                raw.update(prediction=prediction, denoised=denoised)
                artifact = args.output / f"call{call}.npz"
                np.savez_compressed(artifact, **raw)
                report["calls"].append({"call": call, "metrics": scores, "layers": layers,
                    "first_failing_layer": first_failure,
                    "first_mask_failure": first_mask_failure,
                    "diagnostic_seconds": time.perf_counter() - call_start, "raw_sha256": sha(artifact)})
                persist()
        report["terminal_histories_released"] = all(value is None for value in histories)
        if args.decode:
            report["decoded_stream"] = decode_stream(core, device, packed_calls, args.output)
        passed = (report["text_context"]["pass"] and report["terminal_histories_released"]
                  and all(all(score["pass"] for score in row["metrics"].values())
                          and all(layer["pass"] and (layer["mask_exact"] or args.reference_masks) and layer["cache_indices_exact"]
                                  for layer in row["layers"]) for row in report["calls"]))
        report["status"] = "pass_real_video_denoiser_only" if passed else "fail_numerical_trajectory"
        if args.reference_inputs or args.reference_masks:
            report["status"] = "pass_diagnostic_control_only" if passed else "fail_diagnostic_control"
            report["qualification_warning"] = "Native-input/mask controls do not qualify the actual end-to-end backend"
    except BaseException as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        report["diagnostic_wall_seconds"] = time.perf_counter() - start
        persist()
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
