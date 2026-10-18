"""Finite small true-T5 source DiT N0/N16 controls; root launches only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import time
from adapter import configure_prefix

HERE = Path(__file__).resolve().parent
E = HERE.parent
T = E.parents[1]
PARENT = E / "temporal-native-prototype"
PARENT_SHA = "df7bc49d5f30c99ae3ccbae7f0f7057f68e471c544dae4b27f3861da5fbbbcad"
MEAN_SHA = "31f2071bac08df078d1734ea603349dc15d58701b34ddd6ad8922733254e70f6"
ENCODE_SHA = "e0c59f0120e04f90b70e28e0b201d3d25ec4da7de20b974509d7a6d4a7f2e8a8"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authority():
    if digest(PARENT / "xpu-pins.json") != PARENT_SHA:
        raise ValueError("Parent runtime/source authority changed")
    parent = json.loads((PARENT / "xpu-pins.json").read_text())
    selected = json.loads((HERE / "SOURCE-PINS.json").read_text())
    for path, row in selected["files"].items():
        path = Path(path)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Selected source/input changed: " + str(path))
        if str(path) in parent["files"] and parent["files"][str(path)] != row:
            raise ValueError("Selected source differs from frozen parent authority")
    reference = PARENT / "xpu_t5_reference.py"
    spec = importlib.util.spec_from_file_location("small_t5_reference", reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)  # Parent top-level imports neither Torch nor models.
    ref.source_check()
    return ref, parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, choices=(0, 16), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ref, parent = authority()
    import numpy as np
    values = ref.fixture(PARENT / "fixture")
    encode = json.loads((PARENT / "xpu-t5-results/encode.json").read_text())
    mean_path = PARENT / "xpu-t5-results/mean.npy"
    if digest(PARENT / "xpu-t5-results/encode.json") != ENCODE_SHA or digest(mean_path) != MEAN_SHA:
        raise ValueError("Completed mean authority changed")
    mean = np.load(mean_path, allow_pickle=False)
    if (encode["status"] != "completed" or encode["fixture_sha256"] != ref.FIXTURE_HASH or
            mean.shape != (1, 16, 2, 96, 120) or mean.dtype != np.float32 or not np.isfinite(mean).all()):
        raise ValueError("Completed mean schema differs")
    if args.check:
        print(json.dumps({"status": "PASS_INPUTS_ONLY", "blocks": args.blocks, "Torch_imports": 0, "GPU_calls": 0}))
        return
    if args.output is None or args.output.parent.resolve() != HERE or args.output.exists() or args.output.is_symlink():
        raise ValueError("Require absent direct task-owned output directory")
    args.output.mkdir(mode=0o700)
    receipt = {"status": "running", "blocks": args.blocks, "outputs": {}, "calls": [],
               "mean_sha256": MEAN_SHA, "fixture_sha256": ref.FIXTURE_HASH,
               "parent_authority_sha256": PARENT_SHA, "no_retry": True}
    started = time.monotonic()
    def record():
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    record()
    try:
        import torch
        if torch.__version__ != "2.9.1+xpu" or torch.version.cuda is not None or torch.version.hip is not None or hasattr(torch._C, "_cuda_getDeviceCount"):
            raise ValueError("Requires pinned XPU-only Torch")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        resolver = ref.load_helper("evidence/vae-backend-20260918/torch_vae_backend_probe.py", "small_swap_identity")
        device, identity = resolver._resolve_xpu_identity(torch, "0000:03:00.0")
        if identity["torch_uuid_raw"].replace("-", "").lower() != "86800be2000000000300000000000000":
            raise ValueError("B580 UUID differs")
        torch.xpu.set_device(device)
        sys.path.insert(0, str(ref.SOURCE))
        from src.utils.debug import Debug
        from src.optimization.memory_manager import clear_rope_lru_caches
        debug = Debug(enabled=False)  # Retained through every wrapped forward.
        load_started = time.monotonic()
        weight = ref.TRIAL / "models" / ref.WEIGHTS["dit"][0]
        # Full checkpoint SHA belongs to root's inherited prelaunch closure verify.
        if weight.stat().st_size != ref.WEIGHTS["dit"][1] or parent["files"][str(weight)]["sha256"] != ref.WEIGHTS["dit"][2]:
            raise ValueError("Checkpoint authority/size differs")
        model = ref.streamed_dit(torch, device)
        placement = configure_prefix(model, args.blocks, device, torch.device("cpu"), debug)
        torch.xpu.synchronize(device)
        receipt.update(identity=identity, placement=placement, model_load_and_swap_setup_seconds=time.monotonic() - load_started)
        with torch.inference_mode():
            for call in ("warmup", "replay"):
                call_started = time.monotonic()
                noise = values["fixed_noise"]
                condition = mean[0].transpose(1, 2, 3, 0) * np.float32(0.9152)
                video = np.concatenate((noise, condition, np.ones((2, 96, 120, 1), np.float32)), axis=-1).reshape(-1, 33)
                gpu = lambda value: torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.bfloat16)
                vid, txt, timestep = gpu(video), gpu(values["text_positive"]), gpu(values["timestep"])
                vs = torch.tensor([[2, 96, 120]], device=device, dtype=torch.long)
                ts = torch.tensor([[58]], device=device, dtype=torch.long)
                output = model(vid, txt, vs, ts, timestep, disable_cache=False).vid_sample
                prediction = output.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
                torch.xpu.synchronize(device)
                # Preserve completed prediction even if restoration cannot reshape it.
                def save(role, array):
                    path = args.output / (call + "-" + role + ".npy")
                    with path.open("xb") as stream:
                        np.save(stream, array, allow_pickle=False)
                    receipt["outputs"][path.name] = {"sha256": digest(path), "shape": list(array.shape), "dtype": str(array.dtype)}
                save("prediction", prediction)
                record()
                restored = ((noise - prediction.reshape(2, 96, 120, 16)) / np.float32(0.9152)).transpose(3, 0, 1, 2)[None]
                save("restored", np.ascontiguousarray(restored))
                record()
                if prediction.shape != (23040, 16) or restored.shape != (1, 16, 2, 96, 120) or not np.isfinite(prediction).all() or not np.isfinite(restored).all():
                    raise ValueError("Completed output shape/finite guard failed")
                del output, vid, txt, timestep, vs, ts
                clear_rope_lru_caches(model)  # Same healthy replay boundary for N0/N16.
                receipt["calls"].append({"call": call, "seconds": time.monotonic() - call_started})
                record()
                print(json.dumps({"call_complete": call, "seconds": receipt["calls"][-1]["seconds"]}), flush=True)
        model.to(torch.device("cpu"))  # Healthy normal teardown only; wrappers are not retried.
        del model, debug
        receipt["status"] = "completed"
        receipt["replay_byte_exact"] = all(receipt["outputs"]["warmup-" + role + ".npy"]["sha256"] == receipt["outputs"]["replay-" + role + ".npy"]["sha256"] for role in ("prediction", "restored"))
        receipt["quality_award"] = False
        receipt["wall_seconds"] = time.monotonic() - started
        record()
    except BaseException as error:
        receipt.update(status="failed", error=str(error), GPU_calls_in_handler=0, no_retry=True)
        record()
        raise


if __name__ == "__main__":
    main()
