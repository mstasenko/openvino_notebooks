#!/usr/bin/env python3
"""Private same-fixture true T5 BF16 XPU quality reference; each phase exits naturally."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.dont_write_bytecode = True
TRIAL = Path(__file__).resolve().parents[3]
SOURCE = TRIAL / "source"
HEAD = "8d4f3c4bd854695bfcbd25e775b9ea5852ca5f2c"
FIXTURE_HASH = "a807955d5a65a444a929605d659e3068124603a2921996508b86109fc82b604f"
PINS = {
    "openvino-vae-probe.py": "0dc3f2fdb05e88ee064381bb0cca7c2a32caf54ab180b0d60f7ce2872ddae541",
    "evidence/vae-backend-20260918/torch_vae_backend_probe.py": "5cb25714ed14f80962414100b0e0afb2f2348cdb3719a3440b6275f04e266b2d",
}
WEIGHTS = {
    "dit": ("seedvr2_ema_3b_fp16.safetensors", 6783018808, "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"),
    "vae": ("ema_vae_fp16.safetensors", 501324814, "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_helper(relative, name):
    path = TRIAL / relative
    if digest(path) != PINS[relative]:
        raise RuntimeError("helper hash changed: " + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config():
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    value = OmegaConf.to_container(OmegaConf.load(SOURCE / "configs_3b/main.yaml").dit.model, resolve=True)
    value.pop("__object__")
    value["attention_mode"] = "sdpa"
    assert (value["num_layers"], value["vid_dim"], value["emb_dim"]) == (32, 2560, 15360)
    return value


def fixture(path):
    import numpy as np
    if digest(path / "receipt.json") != FIXTURE_HASH:
        raise RuntimeError("frozen fixture hash changed")
    pins = json.loads((path / "receipt.json").read_text())["files"]
    spec = {"rgb": ("frames.rgb24", (5, 768, 960, 3)),
            "shared_normalized": ("normalized.bcthw.f32", (1, 3, 5, 768, 960)),
            "fixed_noise": ("noise.thwc.f32", (2, 96, 120, 16)),
            "text_positive": ("text_positive.f32", (58, 5120))}
    values = {}
    for key, (name, shape) in spec.items():
        file = path / name
        if file.stat().st_size != pins[name]["bytes"] or digest(file) != pins[name]["sha256"]:
            raise RuntimeError("frozen fixture file changed: " + name)
        dtype = np.uint8 if key == "rgb" else np.dtype("<f4")
        values[key] = np.fromfile(file, dtype=dtype).reshape(shape)
        assert np.isfinite(values[key]).all()
    values["timestep"] = np.array([1000], np.float32)
    normalized = (values["rgb"].astype(np.float32) / np.float32(127.5) - np.float32(1)).transpose(3, 0, 1, 2)[None]
    assert np.array_equal(normalized, values["shared_normalized"])
    return values


def vae_runtime(model):
    # The qualified loader constructs the base class, not the runtime wrapper.
    # These attributes belong to the pinned architecture, not checkpoint keys.
    import yaml
    cfg = yaml.safe_load((SOURCE / "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml").read_text())
    assert 2 ** (len(cfg["block_out_channels"]) - 1) == cfg["spatial_downsample_factor"] == 8
    assert 2 ** cfg["temporal_scale_num"] == cfg["temporal_downsample_factor"] == 4
    model.spatial_downsample_factor = 8
    model.temporal_downsample_factor = 4
    model.debug = None
    assert model.slicing_sample_min_size == cfg["slicing_sample_min_size"] == 4
    assert model.slicing_latent_min_size == 1 and not model.training
    assert not model.use_slicing
    return {"spatial_downsample_factor": 8, "temporal_downsample_factor": 4,
            "tile_pixels": [512, 512], "overlap_pixels": [64, 64],
            "temporal_memory": "T5 encode and T2 decode unsliced MemoryState.DISABLED",
            "blend_precision": "upstream BF16 ramps/counts/accumulation; native F32"}


def source_check():
    git = ["git", "-c", "safe.directory=" + str(SOURCE.resolve()), "-C", str(SOURCE)]
    actual = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if actual != HEAD:
        raise RuntimeError("source revision changed")
    subprocess.run([*git, "diff", "--quiet", "HEAD", "--", "src", "configs_3b"], check=True)
    for relative, expected in PINS.items():
        if digest(TRIAL / relative) != expected:
            raise RuntimeError("helper changed: " + relative)


def meta_dit(torch):
    from src.models.dit_3b.nadit import NaDiT
    with torch.device("meta"):
        return NaDiT(**config())


def runtime_buffers(model, torch, device):
    """Restore only upstream zero-initialized, nonpersistent rotary caches."""
    restored = []
    for module_name, module in model.named_modules():
        for name in module._non_persistent_buffers_set:
            value = module._buffers.get(name)
            if value is None or not value.is_meta:
                continue
            if type(module).__name__ != "RotaryEmbedding" or name not in ("cached_freqs", "dummy"):
                raise RuntimeError("unknown nonpersistent meta buffer: " + module_name + "." + name)
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            module._buffers[name] = torch.zeros(value.shape, device=device, dtype=dtype)
            restored.append(module_name + "." + name)
    return restored


def check_meta():
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    # Project compatibility imports ask availability APIs. Synthetic callbacks
    # and blocked optional accelerator extensions keep this audit CPU-only.
    for backend in (torch.cuda, torch.xpu):
        backend.is_available = lambda: False
        backend.device_count = lambda: 0
    torch.backends.mps.is_available = lambda: False
    for name in ("omni_xpu_kernel", "flash_attn", "flash_attn_interface", "sageattention", "sageattn3"):
        sys.modules[name] = None
    sys.path.insert(0, str(SOURCE))
    model = meta_dit(torch)
    expected = model.state_dict()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    nonpersistent = sorted(set(buffers) - set(expected))
    report = {"source_head": HEAD, "fixture_sha256": FIXTURE_HASH,
              "parameters": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "meta": v.is_meta} for k, v in parameters.items()},
              "buffers": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "meta": v.is_meta,
                                  "persistent": k in expected} for k, v in buffers.items()},
              "nonpersistent_meta_buffers": nonpersistent,
              "parameter_elements": sum(v.numel() for v in parameters.values()),
              "buffer_elements": sum(v.numel() for v in buffers.values()), "checkpoint_header": {}}
    path = TRIAL / "models" / WEIGHTS["dit"][0]
    with safe_open(str(path), framework="pt", device="cpu") as archive:
        assert set(archive.keys()) == set(expected), "checkpoint keys differ"
        for name in archive.keys():
            metadata = archive.get_slice(name)
            shape = metadata.get_shape()
            assert tuple(shape) == tuple(expected[name].shape), "checkpoint shape differs: " + name
            report["checkpoint_header"][name] = {"shape": shape, "dtype": metadata.get_dtype()}
    # Allocate only ~44 MB of fresh CPU rotary cache buffers, never parameters.
    report["runtime_buffers_reconstructed"] = runtime_buffers(model, torch, torch.device("cpu"))
    assert all(not v.is_meta for k, v in model.named_buffers() if k in nonpersistent)
    report.update(status="pass", gpu_queried=False, checkpoint_tensors_materialized=False,
                  parameters_materialized=False, host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    target = Path(__file__).resolve().parent / "check-meta.json"
    with target.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "pass", "parameter_keys": len(parameters), "buffer_keys": len(buffers),
                      "nonpersistent_buffer_keys": len(nonpersistent), "parameter_elements": report["parameter_elements"],
                      "gpu_queried": False, "checkpoint_tensors_materialized": False, "report": str(target)}))


def streamed_dit(torch, device):
    from safetensors import safe_open
    # Meta construction avoids a 13.6 GB CPU model plus a second checkpoint copy.
    model = meta_dit(torch)
    expected = model.state_dict()
    state = {}
    with safe_open(str(TRIAL / "models" / WEIGHTS["dit"][0]), framework="pt", device="cpu") as archive:
        if set(archive.keys()) != set(expected):
            raise RuntimeError("DiT checkpoint keys differ")
        for name in archive.keys():
            cpu = archive.get_tensor(name)
            if cpu.shape != expected[name].shape:
                raise RuntimeError("DiT checkpoint shape differs: " + name)
            # BF16 applies to parameters and checkpoint buffers, including RoPE.
            dtype = torch.bfloat16 if cpu.is_floating_point() else cpu.dtype
            state[name] = cpu.to(device=device, dtype=dtype)
            del cpu
    model.load_state_dict(state, strict=True, assign=True)
    del state, expected
    runtime_buffers(model, torch, device)
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise RuntimeError("DiT retains unmaterialized buffers or parameters")
    if any(value.device != device for value in (*model.parameters(), *model.buffers())):
        raise RuntimeError("DiT contains tensors on another device")
    gc.collect()
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "check-meta", "encode", "dit", "decode"), required=True)
    parser.add_argument("--bdf", default="0000:03:00.0")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.bdf != "0000:03:00.0":
        parser.error("this isolated reference is restricted to the reviewed B580 BDF")
    expected_output = Path(__file__).resolve().parent / "xpu-t5-results"
    if args.output_dir.resolve() != expected_output.resolve():
        parser.error("output-dir must be the assigned isolated reference directory")
    source_check()
    values = fixture(args.fixture_dir)
    cfg = config()
    if args.phase == "check":
        print(json.dumps({"status": "pass", "source_head": HEAD, "fixture_sha256": FIXTURE_HASH,
                          "layers": cfg["num_layers"], "gpu_queried": False, "weights_loaded": False}))
        return
    if args.phase == "check-meta":
        check_meta()
        return
    import numpy as np
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / (args.phase + ".json")
    if receipt_path.exists():
        raise RuntimeError("refusing existing phase receipt")
    receipt = {"phase": args.phase, "status": "running", "fixture_sha256": FIXTURE_HASH,
               "source_head": HEAD, "helper_pins": PINS, "script_sha256": digest(Path(__file__)),
               "outputs": {}, "no_resize": True, "input_frames": 5, "latent_frames": 2,
               "noise_contract": "frozen THWC2 noise_a, true T5; not five independent T1 calls",
               "no_wavelet": True, "precision": "BF16 weights/state/compute; F32 host scaling",
               "scope": "independent same-fixture visual-quality reference, not strict parity"}
    started = time.monotonic()

    def write_receipt():
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    def save(name, array, shape):
        path = args.output_dir / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        finite = np.isfinite(array)
        receipt["outputs"][name] = {"sha256": digest(path), "shape": list(array.shape),
            "dtype": str(array.dtype), "finite_count": int(finite.sum()), "elements": int(array.size),
            "min_finite": float(array[finite].min()) if finite.any() else None,
            "max_finite": float(array[finite].max()) if finite.any() else None}
        write_receipt()  # Preserve raw output and statistics before enforcing guards.
        if array.shape != shape or not finite.all():
            raise RuntimeError("output shape/finite guard failed: " + name)

    def prior(phase, name, shape):
        evidence = json.loads((args.output_dir / (phase + ".json")).read_text())
        path = args.output_dir / (name + ".npy")
        assert evidence["status"] == "completed" and evidence["fixture_sha256"] == FIXTURE_HASH
        assert digest(path) == evidence["outputs"][name]["sha256"]
        array = np.load(path, allow_pickle=False)
        assert array.shape == shape and array.dtype == np.float32 and np.isfinite(array).all()
        return array

    write_receipt()
    try:
        weight_kind = "dit" if args.phase == "dit" else "vae"
        name, size, expected_hash = WEIGHTS[weight_kind]
        weight = TRIAL / "models" / name
        if weight.stat().st_size != size or digest(weight) != expected_hash:
            raise RuntimeError("checkpoint hash/size mismatch")
        receipt["weights"] = {"name": name, "bytes": size, "sha256": expected_hash}
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        resolver = load_helper("evidence/vae-backend-20260918/torch_vae_backend_probe.py", "xpu_identity_reference")
        device, identity = resolver._resolve_xpu_identity(torch, args.bdf)
        torch.xpu.set_device(device)
        torch.xpu.reset_peak_memory_stats(device)
        receipt["device_identity"] = identity
        receipt["torch_version"] = torch.__version__
        write_receipt()
        sys.path.insert(0, str(SOURCE))

        def readback(tensor):
            result = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
            torch.xpu.synchronize(device)
            return result

        def gpu(array):
            return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.bfloat16)

        with torch.inference_mode():
            if args.phase == "dit":
                mean = prior("encode", "mean", (1, 16, 2, 96, 120))
                noise = values["fixed_noise"]
                condition = mean[0].transpose(1, 2, 3, 0) * np.float32(0.9152)
                video = np.concatenate((noise, condition, np.ones((2, 96, 120, 1), np.float32)), axis=-1).reshape(-1, 33)
                load_started = time.monotonic()
                model = streamed_dit(torch, device)
                torch.xpu.synchronize(device)
                receipt["model_load_seconds"] = time.monotonic() - load_started
                video_tensor, text_tensor, timestep_tensor = gpu(video), gpu(values["text_positive"]), gpu(values["timestep"])
                vid_shape = torch.tensor([[2, 96, 120]], device=device, dtype=torch.long)
                txt_shape = torch.tensor([[58]], device=device, dtype=torch.long)
                torch.xpu.synchronize(device)
                infer_started = time.monotonic()
                prediction = model(video_tensor, text_tensor, vid_shape, txt_shape, timestep_tensor,
                    disable_cache=False).vid_sample
                torch.xpu.synchronize(device)
                receipt["inference_seconds"] = time.monotonic() - infer_started
                prediction = readback(prediction)
                save("prediction_flat", prediction, (23040, 16))
                prediction = prediction.reshape(2, 96, 120, 16)
                save("prediction", prediction, (2, 96, 120, 16))
                restored = ((noise - prediction) / np.float32(0.9152)).transpose(3, 0, 1, 2)[None]
                save("restored_latent", np.ascontiguousarray(restored), (1, 16, 2, 96, 120))
            else:
                helper = load_helper("openvino-vae-probe.py", "xpu_vae_reference")
                load_started = time.monotonic()
                model, _ = helper.load_reduced_vae(SOURCE, weight)
                model = model.to(device=device, dtype=torch.bfloat16).eval()
                receipt["vae_runtime"] = vae_runtime(model)
                torch.xpu.synchronize(device)
                receipt["model_load_seconds"] = time.monotonic() - load_started
                if args.phase == "encode":
                    value = gpu(values["shared_normalized"])
                    torch.xpu.synchronize(device)
                    infer_started = time.monotonic()
                    encoded = model.tiled_encode(value, tile_size=(512, 512), tile_overlap=(64, 64))
                    torch.xpu.synchronize(device)
                    receipt["inference_seconds"] = time.monotonic() - infer_started
                    moments = readback(encoded)
                    save("encoder_moments", moments, (1, 32, 2, 96, 120))
                    save("mean", np.ascontiguousarray(moments[:, :16]), (1, 16, 2, 96, 120))
                else:
                    restored = prior("dit", "restored_latent", (1, 16, 2, 96, 120))
                    value = gpu(restored)
                    torch.xpu.synchronize(device)
                    infer_started = time.monotonic()
                    decoded = model.tiled_decode(value, tile_size=(512, 512), tile_overlap=(64, 64))
                    torch.xpu.synchronize(device)
                    receipt["inference_seconds"] = time.monotonic() - infer_started
                    decoded = readback(decoded)
                    save("decoded_pre_wavelet", decoded, (1, 3, 5, 768, 960))
                    rgb = np.floor((np.clip(decoded[0].transpose(1, 2, 3, 0), -1, 1) + np.float32(1)) * np.float32(127.5) + np.float32(0.5)).astype(np.uint8)
                    with (args.output_dir / "reference.rgb24").open("xb") as stream:
                        stream.write(rgb.tobytes())
                    receipt["rgb_sha256"] = digest(args.output_dir / "reference.rgb24")
        torch.xpu.synchronize(device)
        receipt["gpu_memory"] = resolver._memory_snapshot(torch, device, "completed")
        receipt["status"] = "completed"
        write_receipt()
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
        write_receipt()
        raise


if __name__ == "__main__":
    main()
