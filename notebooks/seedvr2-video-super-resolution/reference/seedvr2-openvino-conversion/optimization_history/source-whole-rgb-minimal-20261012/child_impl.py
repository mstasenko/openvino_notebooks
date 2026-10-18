"""Fixed BF16 child derivative: only compact archival/layout and separated clocks."""
import json
import sys
import time
from pathlib import Path
from artifacts import CALLS, SHAPES, LATENT_HEIGHT, LATENT_WIDTH, array, save, mean_shape, prediction_shape, validate_raws

def child(root, phase, old):
    seal = old.seal
    inputs = old.inputs
    physical_admission = old.physical_admission
    # Child-only imports/device work. The supervisor never imports Torch.
    import numpy as np
    ref, uhd, values = inputs()
    directory = root / phase
    directory.mkdir()
    diagnostics = seal.authority.load(seal.authority.DIAGNOSTICS, "phase_diagnostics")
    diag = diagnostics.StageDiagnostics(directory / "stages.jsonl")
    started = time.monotonic()
    try:
        import torch
        if torch.__version__ != "2.9.1+xpu" or torch.version.cuda is not None or torch.version.hip is not None or hasattr(torch._C, "_cuda_getDeviceCount"):
            raise ValueError("requires pinned XPU-only Torch")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(ref.SOURCE))
        resolver = ref.load_helper("evidence/vae-backend-20260918/torch_vae_backend_probe.py", "phase_identity")
        device, identity = resolver._resolve_xpu_identity(torch, "0000:03:00.0")
        torch.xpu.set_device(device)
        from src.optimization.memory_manager import clear_memory, clear_rope_lru_caches, _standard_model_movement
        def mark(stage):
            torch.xpu.synchronize(device)
            diag.phase(stage, time.monotonic() - started, lambda: diagnostics.allocator_stats(torch, device))
        mark("identity_ready")
        if phase == "dit":
            # UHD-specific observation after initialized runtime identity, before upload.
            def preserve_admission(observation):
                with (directory / "admission.json").open("x") as stream:
                    json.dump(observation, stream)
                if "raw" in observation:
                    with (directory / "vram_mm.txt").open("x") as stream:
                        stream.write(observation["raw"])
            physical_admission(Path("/sys/kernel/debug/dri/0000:03:00.0/tile0/vram_mm"),
                               diag.last_stage["gpu_allocator"]["memory_allocated"],
                               preserve_admission)
        kind = "dit" if phase == "dit" else "vae"
        name, size, digest = ref.WEIGHTS[kind]
        weight = ref.TRIAL / "models" / name
        if weight.stat().st_size != size or ref.digest(weight) != digest:
            raise ValueError("checkpoint identity changed")
        load_start = time.monotonic()
        if phase == "dit":
            model = uhd.streamed_dit(torch, device)
        else:
            loader = ref.load_helper("openvino-vae-probe.py", "phase_vae_loader")
            model, _ = loader.load_reduced_vae(ref.SOURCE, weight)
            model = model.to(device=device, dtype=torch.bfloat16).eval()
            uhd.vae_runtime(model)
        mark("model_ready")
        load_seconds = time.monotonic() - load_start
        durations = []
        archive_seconds = []
        handoff_seconds = []
        finite_outputs = []
        gpu = lambda value: torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            for call in CALLS:
                diag.frame = call
                loaded = None
                reading = time.monotonic()
                if phase in ("dit", "decode"):
                    predecessor = "encode" if phase == "dit" else "dit"
                    loaded = array(root / predecessor / (call + ".f32"), SHAPES[predecessor])
                handoff_seconds.append(time.monotonic() - reading)
                beginning = time.monotonic()
                if phase == "encode":
                    rgb = values["frames"]["02" if call == "warmup02" else call]
                    normalized = np.ascontiguousarray((rgb.astype(np.float32) / np.float32(127.5) - np.float32(1)).transpose(2, 0, 1)[None, :, None])
                    value = gpu(normalized)
                    output = model.tiled_encode(value, tile_size=(512, 512), tile_overlap=(64, 64))
                    output = output[:, :16]  # Same posterior mean; do not transfer unused variance.
                elif phase == "dit":
                    moments = loaded
                    condition = moments[0, :, 0].transpose(1, 2, 0)[None] * np.float32(0.9152)
                    video = np.concatenate((values["fixed_noise"], condition, np.ones((1, LATENT_HEIGHT, LATENT_WIDTH, 1), np.float32)), axis=-1).reshape(-1, 33)
                    value, text, timestep = gpu(video), gpu(values["text_positive"]), gpu(values["timestep"])
                    vs = torch.tensor([[1, LATENT_HEIGHT, LATENT_WIDTH]], device=device, dtype=torch.long)
                    ts = torch.tensor([[58]], device=device, dtype=torch.long)
                    output = model(value, text, vs, ts, timestep, disable_cache=False).vid_sample
                else:
                    prediction = np.ascontiguousarray(loaded[0].transpose(1, 2, 3, 0))
                    restored = ((values["fixed_noise"] - prediction) / np.float32(0.9152)).transpose(3, 0, 1, 2)[None]
                    value = gpu(restored)
                    output = model.tiled_decode(value, tile_size=(512, 512), tile_overlap=(64, 64))
                raw = output.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
                torch.xpu.synchronize(device)
                finite_outputs.append(bool(np.isfinite(raw).all()))
                if phase == "encode":
                    raw = mean_shape(raw)
                elif phase == "dit":
                    raw = prediction_shape(raw)
                del output, value
                if phase == "dit":
                    del text, timestep, vs, ts
                    clear_rope_lru_caches(model)
                if phase == "decode":
                    raw = np.floor((np.clip(raw[0].transpose(1, 2, 0), -1, 1) + np.float32(1)) * np.float32(127.5) + np.float32(0.5)).astype(np.uint8)
                    raw = np.ascontiguousarray(raw)
                # Complete owned CPU output and normal per-call cleanup, no archive/hash in this clock.
                durations.append(time.monotonic() - beginning)
                writing = time.monotonic()
                save(directory / (call + (".rgb24" if phase == "decode" else ".f32")), raw)
                archive_seconds.append(time.monotonic() - writing)
                mark("cpu_readback_saved")
        cleanup_start = time.monotonic()
        _standard_model_movement(model, device, torch.device("cpu"), "cpu", "DiT" if phase == "dit" else "VAE")
        del model
        clear_memory(deep=True, force=True)
        mark("normal_cleanup_complete")
        cleanup_seconds = time.monotonic() - cleanup_start
        receipt = {"status": "PASS", "phase": phase, "calls": list(CALLS), "identity": identity,
            "seconds": time.monotonic() - started, "model_load_seconds": load_seconds,
            "call_seconds": durations, "raw_write_seconds": archive_seconds,
            "handoff_read_seconds": handoff_seconds, "normal_cleanup_seconds": cleanup_seconds,
            "call_boundary": "owned CPU input through complete CPU output; archive/hash excluded",
            "handoff_reads_charged_phase_wall": True, "final_allocator": diag.last_stage["gpu_allocator"]}
        validate_raws(root, phase)
        if not all(finite_outputs):
            raise ValueError("nonfinite completed source output; raws preserved")
        if receipt["final_allocator"]["memory_allocated"] != 0:
            raise ValueError("live Torch allocation at normal teardown")
        with (directory / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, allow_nan=False)
    except BaseException as error:
        diag.fault(error)  # No GPU operation, telemetry or cleanup after an exception.
        raise
