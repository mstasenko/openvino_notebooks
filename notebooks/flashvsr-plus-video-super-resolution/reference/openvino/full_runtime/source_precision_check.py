#!/usr/bin/env python3
"""Same-feed CPU FP16 source pre-block check against saved native CUDA tensors."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    F = torch.nn.functional
    qualifier = load("fp16_pre_parameter_helper", BASE / "openvino/parameter_layers/qualify_cpu.py")
    contract = load("fp16_pre_parameter_contract", BASE / "openvino/parameter_layers/contract.py")
    block = qualifier._load_block_qualify()
    dit = block.import_pinned_dit()
    source = qualifier._load_source_oracle()
    reference = BASE / "evidence/fp16-denoiser-stages-attempt1"
    record = json.loads((reference / "results.json").read_text())
    capture = reference / "capture_layers.npz"
    mask_path = reference / "mask_trace.npz"
    if qualifier.sha256(capture) != record["capture_layers"]["sha256"] or qualifier.sha256(mask_path) != record["mask_trace"]["sha256"]:
        raise ValueError("Native capture changed")
    report = {"scope": "CPU native FP16 pre-block, identical CUDA inputs; no GPU execution", "cases": [],
              "source_sha256": qualifier.sha256(BASE / "source/FlashVSR/diffsynth/models/wan_video_dit.py"),
              "capture_sha256": qualifier.sha256(capture), "script_sha256": qualifier.sha256(Path(__file__))}
    with np.load(capture, allow_pickle=False) as saved, np.load(mask_path, allow_pickle=False) as masks, safe_open(str(qualifier.CHECKPOINT), framework="np", device="cpu") as handle, torch.inference_mode():
        for layer in (0, 1):
            values = qualifier._load_layer(handle, contract, layer)
            weights = {key: torch.from_numpy(value) for key,value in values.items()}
            def linear(x, prefix):
                return F.linear(x, weights[prefix+".weight"], weights[prefix+".bias"])
            # The source RMSNorm method needs eps from its instance; keep the
            # expression exactly as source (F32 reduction then cast-before-scale).
            def rms(x, prefix):
                return (x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)).to(x.dtype)*weights[prefix+".weight"]
            call = 0
            prefix = f"reset0_call{call}_layer{layer:02d}"
            x = torch.from_numpy(saved[prefix+"_input_residual"].copy())
            t_mod = torch.from_numpy(saved["t_mod"].copy())
            shifts = (weights["blocks.0.modulation"]+t_mod).chunk(6,dim=1)
            modulated = dit.modulate(F.layer_norm(x, (1536,), eps=1e-6), shifts[0], shifts[1])
            freqs = source._source_freqs(dit, torch, 0)
            q = dit.rope_apply(rms(linear(modulated,"blocks.0.self_attn.q"),"blocks.0.self_attn.norm_q"),freqs,12)
            k = dit.rope_apply(rms(linear(modulated,"blocks.0.self_attn.k"),"blocks.0.self_attn.norm_k"),freqs,12)
            q = dit.WindowPartition3D.partition(q.reshape(1,6,8,8,1536),(2,8,8))
            k = dit.WindowPartition3D.partition(k.reshape(1,6,8,8,1536),(2,8,8))
            actual = dit.generate_draft_block_mask(1,12,3,q,k,topk=1,local_attn_mask=block.local_mask(dit,torch)).numpy()
            expected = masks[prefix]
            raw_path = args.output / f"layer{layer:02d}.npz"
            np.savez(raw_path, q=q.numpy(), k=k.numpy(), mask=actual, native_mask=expected)
            report["cases"].append({"layer": layer, "q": qualifier.metric(saved[prefix+"_mask_q_w"],q.numpy()),
                "k": qualifier.metric(saved[prefix+"_mask_k_w"],k.numpy()),
                "mask_exact": bool(np.array_equal(actual,expected)), "different_mask_bits": int(np.count_nonzero(actual!=expected)),
                "raw_sha256": qualifier.sha256(raw_path)})
    (args.output / "results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
