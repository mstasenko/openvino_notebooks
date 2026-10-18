#!/usr/bin/env python3
"""Private process-isolated source benchmark; UHD distinct-fixture phase-major candidate; no automatic fit approval."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import seal
from admission import physical_admission

HERE = Path(__file__).resolve().parent
CALLS = ("warmup02", "01", "02", "03")
PHASES = ("encode", "dit", "decode")
LABEL = "process-isolated UHD phase-major; all startup/load/handoff/teardown charged"
HEIGHT, WIDTH = 2160, 3840
LATENT_HEIGHT, LATENT_WIDTH = 270, 480
SHAPES = {"encode": (1, 32, LATENT_HEIGHT, LATENT_WIDTH),
          "dit": (1, LATENT_HEIGHT, LATENT_WIDTH, 16),
          "decode": (1, 3, HEIGHT, WIDTH)}


def prediction_shape(raw):
    return raw.reshape(SHAPES["dit"])


def array(path, shape):
    import numpy as np
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid() or path.stat().st_nlink != 1:
        raise ValueError("handoff must be an owned regular file")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError("unexpected handoff format")
        actual_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if actual_shape != shape or fortran or dtype != np.float32 or path.stat().st_size != stream.tell() + math.prod(shape) * 4:
            raise ValueError("handoff header/size mismatch")
    value = np.load(path, allow_pickle=False)
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("malformed handoff: " + str(path))
    return value


def save(path, value):
    import numpy as np
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def inputs():
    ref = seal.authority.load(seal.authority.SMALL, "phase_uhd_source")
    ref.source_check()
    uhd = seal.authority.load(seal.authority.UHD, "phase_uhd_loader")
    producer = seal.load_fixture_producer()
    values, frames = producer.fixtures(uhd)
    values["frames"] = frames
    return ref, uhd, values


def validate_raws(root, phase):
    shape = SHAPES[phase]
    for call in CALLS:
        array(root / phase / (call + ".npy"), shape)
        if phase == "decode":
            path = root / phase / (call + ".rgb24")
            if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid() or path.stat().st_nlink != 1 or path.stat().st_size != HEIGHT * WIDTH * 3:
                raise ValueError("malformed RGB output")


def validate_outputs(root, phase):
    validate_raws(root, phase)
    receipt = json.loads((root / phase / "receipt.json").read_text())
    if receipt["phase"] != phase or receipt["calls"] != list(CALLS) or receipt["status"] != "PASS":
        raise ValueError("child receipt mismatch")
    elapsed = [receipt["seconds"], receipt["model_load_seconds"], *receipt["call_seconds"]]
    if len(receipt["call_seconds"]) != 4 or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in elapsed):
        raise ValueError("invalid child clock")


def sequence(root, launch, validate=validate_outputs, clock=time.monotonic):
    started = clock()
    children = []
    for phase in PHASES:
        before = clock()
        result = launch(phase)
        children.append({"phase": phase, "exit_code": result, "wall_seconds": clock() - before})
        if result != 0:
            raise RuntimeError("child failed; no next phase: " + phase)
        validate(root, phase)  # Charged handoff validation before any next launch.
    return {"label": LABEL, "whole_video_seconds": clock() - started,
            "children": children, "warmup_calls": 1, "measured_calls": 3,
            "input_scope": "warmup02 then three distinct real UHD frames01/02/03; same fixed noise/text",
            "steady_stage_sum_is_whole_video": False, "quality_pass": False}


def child(root, phase):
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
        gpu = lambda value: torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            for call in CALLS:
                diag.frame = call
                beginning = time.monotonic()
                if phase == "encode":
                    rgb = values["frames"]["02" if call == "warmup02" else call]
                    normalized = np.ascontiguousarray((rgb.astype(np.float32) / np.float32(127.5) - np.float32(1)).transpose(2, 0, 1)[None, :, None])
                    value = gpu(normalized)
                    output = model.tiled_encode(value, tile_size=(512, 512), tile_overlap=(64, 64))
                elif phase == "dit":
                    moments = array(root / "encode" / (call + ".npy"), SHAPES["encode"])
                    condition = moments[0, :16].transpose(1, 2, 0)[None] * np.float32(0.9152)
                    video = np.concatenate((values["fixed_noise"], condition, np.ones((1, LATENT_HEIGHT, LATENT_WIDTH, 1), np.float32)), axis=-1).reshape(-1, 33)
                    value, text, timestep = gpu(video), gpu(values["text_positive"]), gpu(values["timestep"])
                    vs = torch.tensor([[1, LATENT_HEIGHT, LATENT_WIDTH]], device=device, dtype=torch.long)
                    ts = torch.tensor([[58]], device=device, dtype=torch.long)
                    output = model(value, text, vs, ts, timestep, disable_cache=False).vid_sample
                else:
                    prediction = array(root / "dit" / (call + ".npy"), SHAPES["dit"])
                    restored = ((values["fixed_noise"] - prediction) / np.float32(0.9152)).transpose(3, 0, 1, 2)[None]
                    value = gpu(restored)
                    output = model.tiled_decode(value, tile_size=(512, 512), tile_overlap=(64, 64))
                raw = output.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
                torch.xpu.synchronize(device)
                if phase == "dit":
                    raw = prediction_shape(raw)
                del output, value
                if phase == "dit":
                    del text, timestep, vs, ts
                    clear_rope_lru_caches(model)
                # Save every completed result before numerical guards. Disk is charged.
                save(directory / (call + ".npy"), raw)
                if phase == "decode":
                    rgb = np.floor((np.clip(raw[0].transpose(1, 2, 0), -1, 1) + np.float32(1)) * np.float32(127.5) + np.float32(0.5)).astype(np.uint8)
                    with (directory / (call + ".rgb24")).open("xb") as stream:
                        stream.write(rgb.tobytes())
                mark("cpu_readback_saved")
                durations.append(time.monotonic() - beginning)
        _standard_model_movement(model, device, torch.device("cpu"), "cpu", "DiT" if phase == "dit" else "VAE")
        del model
        clear_memory(deep=True, force=True)
        mark("normal_cleanup_complete")
        receipt = {"status": "PASS", "phase": phase, "calls": list(CALLS), "identity": identity,
            "seconds": time.monotonic() - started, "model_load_seconds": load_seconds,
            "call_seconds": durations, "final_allocator": diag.last_stage["gpu_allocator"]}
        validate_raws(root, phase)
        if receipt["final_allocator"]["memory_allocated"] != 0:
            raise ValueError("live Torch allocation at normal teardown")
        with (directory / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, allow_nan=False)
    except BaseException as error:
        diag.fault(error)  # No GPU operation, telemetry or cleanup after an exception.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=PHASES)
    args = parser.parse_args()
    if args.check:
        inputs()
        print("PASS UHD distinct four-call contract; no Torch import/model/GPU")
        return
    if args.output is None or args.output.parent.resolve() != HERE or args.output.is_symlink():
        parser.error("output must be an exclusive direct child")
    if args.child:
        if not args.output.is_dir():
            parser.error("parent output missing")
        seal.verify_local()
        child(args.output, args.child)
        return
    seal.verify()  # Full existing closure outside benchmark clock, CPU only.
    args.output.mkdir()
    def launch(phase):
        with (args.output / (phase + ".log")).open("xb") as stream:
            return subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                "--child", phase, "--output", str(args.output.resolve())], stdout=stream, stderr=subprocess.STDOUT).returncode
    try:
        receipt = sequence(args.output, launch)
        with (args.output / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
    except BaseException as error:
        with (args.output / "error.json").open("x") as stream:
            json.dump({"error": str(error), "label": LABEL, "no_next_child": True}, stream)
        raise


if __name__ == "__main__":
    main()
