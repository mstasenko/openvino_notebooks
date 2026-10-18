#!/usr/bin/env python3
"""CPU qualification of native-F16 text/cross-cache preparation, not F32 caches."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[2]
TEXT = {"text_embedding.0.weight": (1536, 4096), "text_embedding.0.bias": (1536,),
        "text_embedding.2.weight": (1536, 1536), "text_embedding.2.bias": (1536,)}
KV = {"cross_attn.k.weight": (1536, 1536), "cross_attn.k.bias": (1536,),
      "cross_attn.norm_k.weight": (1536,), "cross_attn.v.weight": (1536, 1536),
      "cross_attn.v.bias": (1536,)}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def port(key):
    return key.replace(".", "_")


def build(role):
    import openvino as ov
    from openvino import opset13 as ops
    widths = TEXT if role == "text" else KV
    x = ops.parameter([1, 512, 4096 if role == "text" else 1536], ov.Type.f16, name="x")
    params = {key: ops.parameter(shape, ov.Type.f16, name=port(key)) for key, shape in widths.items()}
    def linear(value, name):
        # Native Linear rounds only after the bias, not between matmul and bias.
        return ops.add(ops.matmul(value, params[name + ".weight"], False, True), params[name + ".bias"])
    if role == "text":
        outputs = [linear(ops.gelu(linear(x, "text_embedding.0"), "tanh"), "text_embedding.2")]
        names = ["context"]
    else:
        k = linear(x, "cross_attn.k")
        k32 = ops.convert(k, ov.Type.f32)
        square_mean = ops.reduce_mean(ops.multiply(k32, k32), ops.constant(np.array([-1], np.int64)), True)
        normalized = ops.divide(k32, ops.sqrt(ops.add(square_mean, ops.constant(np.float32(1e-6)))))
        k = ops.multiply(ops.convert(normalized, ov.Type.f16), params["cross_attn.norm_k.weight"])
        outputs = [k, linear(x, "cross_attn.v")]
        names = ["cross_k", "cross_v"]
    for node, name in zip(outputs, names):
        node.output(0).get_tensor().set_names({name})
    for key, node in params.items():
        node.output(0).get_tensor().set_names({port(key)})
    x.output(0).get_tensor().set_names({"x"})
    return ov.Model(outputs, [x, *params.values()], f"source_f16_{role}")


def infer(model, feeds):
    request = model.create_infer_request()
    request.infer(feeds, share_inputs=False, share_outputs=False)
    return {out.get_any_name(): np.array(request.get_output_tensor(i).data, copy=True) for i, out in enumerate(model.outputs)}


def metric(reference, actual):
    r, a = np.asarray(reference, np.float64), np.asarray(actual, np.float64)
    if r.shape != a.shape or not np.isfinite(a).all():
        raise ValueError("Invalid cache shape/finiteness")
    delta = a-r
    rel = float(np.linalg.norm(delta)/max(np.linalg.norm(r), 1e-30))
    nmax = float(abs(delta).max()/max(abs(r).max(), 1e-30))
    return {"relative_l2": rel, "normalized_max_abs": nmax, "passed": rel <= .005 and nmax <= .015625}


def load_weights(handle, schema, prefix=""):
    values = {}
    for key, shape in schema.items():
        view = handle.get_slice(prefix+key)
        if tuple(view.get_shape()) != shape or view.get_dtype() != "F32":
            raise ValueError(f"Source parameter ABI changed: {prefix+key}")
        value = np.ascontiguousarray(handle.get_tensor(prefix+key).astype(np.float16))
        if not np.isfinite(value).all():
            raise ValueError("Non-finite F16 parameter")
        values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("CPU","B580"), default="CPU")
    parser.add_argument("--cpu-evidence", type=Path)
    args = parser.parse_args()
    if not args.run:
        parser.error("Explicit --run required for CPU-only qualification")
    if args.device == "B580" and args.cpu_evidence is None:
        parser.error("B580 requires matching CPU evidence and fresh external exact-device preflight")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import openvino as ov
    from safetensors import safe_open
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = BASE / "evidence/fp16-denoiser-stages-attempt1"
    receipt = json.loads((source / "results.json").read_text())
    capture_path = source / "capture_layers.npz"
    if receipt["status"] != "pass" or sha(capture_path) != receipt["capture_layers"]["sha256"]:
        raise ValueError("Authenticated native-F16 capture required")
    with np.load(capture_path, allow_pickle=False) as archive:
        native = {name: archive[name].copy() for name in archive.files if name.startswith("cross_cache")}
    checkpoint = BASE / "models/diffusion_pytorch_model_streaming_dmd.safetensors"
    if sha(checkpoint) != "bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be":
        raise ValueError("Checkpoint changed")
    prompt_path = BASE / "source/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth"
    if sha(prompt_path) != "4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef":
        raise ValueError("Prompt changed")
    prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)
    if prompt.shape != (1,512,4096) or prompt.dtype != torch.bfloat16:
        raise ValueError("Prompt ABI changed")
    prompt = prompt.half()
    report = {"status": "initializing", "scope": "native F16 cache preparation, layers0/1 only",
              "requested_device": args.device,
              "script_sha256": sha(Path(__file__)), "runtime": ov.get_version(), "graphs": [], "cases": [],
              "checkpoint_sha256": sha(checkpoint), "capture_sha256": sha(capture_path)}
    core = ov.Core()
    device = "CPU"
    cpu = None
    if args.device == "B580":
        sys.path.insert(0,str(BASE / "openvino"))
        from attention_from_cuda import resolve
        device, inventory = resolve(core)
        report["device_inventory"] = inventory
        cpu = json.loads((args.cpu_evidence / "results.json").read_text())
        if cpu["status"] != "pass" or len(cpu["graphs"]) != 2:
            raise ValueError("Passing two-graph CPU qualification required")
    report["device"] = device
    config = {"INFERENCE_PRECISION_HINT": ov.Type.f16, "NUM_STREAMS": ov.properties.streams.Num(1),
              "PERFORMANCE_HINT": ov.properties.hint.PerformanceMode.LATENCY, "INFERENCE_NUM_THREADS": 2, "CACHE_DIR": ""}
    if device != "CPU":
        del config["INFERENCE_NUM_THREADS"]
    compiled = {}
    try:
        for role in ("text", "kv"):
            if cpu is None:
                model = build(role)
            else:
                original, = [row for row in cpu["graphs"] if row["role"] == role]
                qualified_path = args.cpu_evidence / f"{role}.xml"
                if sha(qualified_path) != original["xml_sha256"] or sha(qualified_path.with_suffix(".bin")) != original["bin_sha256"]:
                    raise ValueError("CPU-qualified native cache artifact changed")
                # Reuse the actual qualified IR. Rebuilding after compilation
                # changes auto-generated node IDs differently on CPU/GPU.
                model = core.read_model(qualified_path)
            path = args.output / f"{role}.xml"
            ov.serialize(model, str(path), str(path.with_suffix(".bin")))
            compiled[role] = core.compile_model(model, device, config)
            if list(compiled[role].get_property("EXECUTION_DEVICES")) != [device]:
                raise ValueError("Unexpected cache preparation execution device")
            report["graphs"].append({"role": role, "xml": str(path), "xml_sha256": sha(path), "bin_sha256": sha(path.with_suffix(".bin"))})
        with safe_open(str(checkpoint), framework="np", device="cpu") as handle:
            text = load_weights(handle, TEXT)
            context = infer(compiled["text"], {"x": prompt.numpy(), **{port(k): v for k,v in text.items()}})["context"]
            def tensor(key, values):
                return torch.from_numpy(values[key])
            with torch.inference_mode():
                expected_context = torch.nn.functional.linear(torch.nn.functional.gelu(
                    torch.nn.functional.linear(prompt, tensor("text_embedding.0.weight",text), tensor("text_embedding.0.bias",text)), approximate="tanh"),
                    tensor("text_embedding.2.weight",text), tensor("text_embedding.2.bias",text))
            report["context_cpu_source"] = metric(expected_context.numpy(), context)
            for layer in (0,1):
                values = load_weights(handle, KV, f"blocks.{layer}.")
                outputs = infer(compiled["kv"], {"x": context, **{port(k): v for k,v in values.items()}})
                with torch.inference_mode():
                    k = torch.nn.functional.linear(expected_context, tensor("cross_attn.k.weight",values), tensor("cross_attn.k.bias",values))
                    k_norm = k.float()*torch.rsqrt(k.float().square().mean(-1,keepdim=True)+1e-6)
                    expected_k = k_norm.half()*tensor("cross_attn.norm_k.weight",values)
                    expected_v = torch.nn.functional.linear(expected_context, tensor("cross_attn.v.weight",values), tensor("cross_attn.v.bias",values))
                raw_path = args.output / f"layer{layer:02d}.npz"
                np.savez(raw_path, **outputs)
                report["cases"].append({"layer": layer, "raw": str(raw_path), "sha256": sha(raw_path),
                    "cpu_source_k": metric(expected_k.numpy(),outputs["cross_k"]), "cpu_source_v": metric(expected_v.numpy(),outputs["cross_v"]),
                    "cuda_source_k": metric(native[f"cross_cache_layer{layer:02d}_k"],outputs["cross_k"]),
                    "cuda_source_v": metric(native[f"cross_cache_layer{layer:02d}_v"],outputs["cross_v"])})
        report["status"] = "pass" if report["context_cpu_source"]["passed"] and all(v["passed"] for row in report["cases"] for k,v in row.items() if k.startswith(("cpu_source", "cuda_source"))) else "fail"
    except BaseException as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        compiled.clear()
        core = None
        gc.collect()
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "cases": report["cases"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
