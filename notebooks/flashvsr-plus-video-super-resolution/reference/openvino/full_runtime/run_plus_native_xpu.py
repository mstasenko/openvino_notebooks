"""Finite native FlashVSR+ B580 trajectory diagnostic.

This worker consumes the authenticated common-input ABI and runs the pinned
Turbo BF16 source model on exactly one B580.  It is deliberately a diagnostic
reference, not an OpenVINO backend or a throughput benchmark: only the LQ
condition, prediction, denoised chunk, and compact source sparse masks are
saved for each of the two calls.

No device API is touched unless ``--run`` is supplied.  The optional decoder
uses the same native TAEHV source and condition ranges as the CPU decoder
oracle and is likewise a finite quality diagnostic.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import resource
import time
from typing import Any, Iterator

import numpy as np
import torch

from capture_plus_video_cpu import (
    CHUNK_LATENTS,
    DIT_WEIGHTS,
    LATENT_CHANNELS,
    LQ_WEIGHTS,
    PROMPT_WEIGHTS,
    SOURCE,
    SOURCE_COMMIT,
    TEMPORAL_DOWNSAMPLE,
    FlowMatchInferenceScheduler,
    StreamState,
    _project_conditions,
    _write_npz,
    authenticate_checkpoint,
    authenticate_dit,
    authenticate_prompt,
    build_native_model,
    require_clean_source,
    sha256_file,
)
from prepare_plus_video_input import FORMAT as INPUT_FORMAT, FORMAT_V2
from prepare_plus_video_input import load_input


B580_UUID = "86800be2-0000-0000-0300-000000000000"
B580_BDF = "0000:03:00.0"
MAX_WORKING_SIDE = 512
STATUS = "native_xpu_b580_diagnostic_complete"
ERROR_STATUS = "error"


def select_b580() -> tuple[torch.device, Any]:
    """Select and authenticate the one explicitly permitted native device."""

    names = [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]
    selected = [
        i for i, name in enumerate(names)
        if name == "Intel(R) Arc(TM) B580 Graphics"
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one B580: {names}")
    index, = selected
    properties = torch.xpu.get_device_properties(index)
    if str(properties.uuid) != B580_UUID:
        raise ValueError("Native XPU B580 UUID changed")
    torch.xpu.set_device(index)
    return torch.device("xpu", index), properties


def validate_input_geometry(manifest: dict[str, Any], *, max_working_side=MAX_WORKING_SIDE,
                            allow_stream=False) -> tuple[int, int]:
    """Reject the larger-than-piloted geometry before any device allocation."""

    if (type(max_working_side) is not int or max_working_side < 128
            or max_working_side > 3840 or max_working_side % 128):
        raise ValueError("Explicit bounded working-side limit required")
    if type(allow_stream) is not bool:
        raise ValueError("allow_stream must be boolean")
    # The native diagnostic still consumes only its original two-call ABI.
    # Only a controller that validates/executes input_plan may opt into v2.
    formats = (INPUT_FORMAT, FORMAT_V2) if allow_stream else (INPUT_FORMAT,)
    if manifest.get("format") not in formats:
        raise ValueError("common input format changed")
    working = manifest.get("working_hw")
    if (
        not isinstance(working, list)
        or len(working) != 2
        or any(type(value) is not int or value <= 0 for value in working)
    ):
        raise ValueError("common input working geometry is invalid")
    if any(value > max_working_side for value in working):
        raise ValueError(
            f"B580 pilot is limited to working sides <= {max_working_side}; "
            f"got {working}"
        )
    if any(value % 16 for value in working):
        raise ValueError("working geometry must be divisible by the source patch size")
    return int(working[0]), int(working[1])


def preflight_input_geometry(root: Path | str, manifest_sha256: str, *, max_working_side=MAX_WORKING_SIDE,
                             allow_stream=False) -> tuple[int, int]:
    """Read only the authenticated manifest before opening a large NPZ."""

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if (
        not manifest_path.is_file()
        or not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in manifest_sha256)
        or sha256_file(manifest_path) != manifest_sha256
    ):
        raise ValueError("common input manifest fingerprint changed")
    return validate_input_geometry(json.loads(manifest_path.read_text(encoding="utf-8")),
                                   max_working_side=max_working_side, allow_stream=allow_stream)


@contextmanager
def limited_mask_observer() -> Iterator[list[np.ndarray]]:
    """Capture only source sparse masks, never block activations or caches."""

    import models.attention as attention_module

    masks: list[np.ndarray] = []
    original = attention_module.build_stream_sparse_layout

    def observe_layout(*args: Any, **kwargs: Any):
        layout = original(*args, **kwargs)
        masks.append(layout.block_mask.detach().cpu().numpy().copy())
        return layout

    attention_module.build_stream_sparse_layout = observe_layout
    try:
        yield masks
    finally:
        attention_module.build_stream_sparse_layout = original


def _f32(value: torch.Tensor, name: str) -> np.ndarray:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"non-finite native output: {name}")
    return value.detach().to(dtype=torch.float32).cpu().numpy().copy()


def _auth_receipts() -> dict[str, Any]:
    """Authenticate all learned inputs before loading the source model."""

    dit = authenticate_dit(DIT_WEIGHTS)
    lq_hash = authenticate_checkpoint(LQ_WEIGHTS)
    prompt = authenticate_prompt(PROMPT_WEIGHTS)
    return {
        "dit": dit,
        "lq_projection": {
            "path": str(LQ_WEIGHTS.resolve()),
            "bytes": int(LQ_WEIGHTS.stat().st_size),
            "sha256": lq_hash,
        },
        "prompt": prompt,
    }


def _save_pngs(output: Path, normalized: np.ndarray) -> list[str]:
    from PIL import Image

    if normalized.ndim != 5 or normalized.shape[0] != 1 or normalized.shape[1] < 17 or normalized.shape[2] != 3:
        raise ValueError(f"unexpected decoded output shape: {normalized.shape}")
    names: list[str] = []
    for frame in (0, 8, 16):
        rgb = np.clip(
            np.rint((normalized[0, frame].transpose(1, 2, 0) + 1.0) * 127.5),
            0,
            255,
        ).astype(np.uint8)
        path = output / f"frame{frame:02d}.png"
        Image.fromarray(rgb, mode="RGB").save(path)
        names.append(path.name)
    return names


def _decode_native(
    output: Path,
    device: torch.device,
    projection_video: torch.Tensor,
    denoised_cpu: list[torch.Tensor],
    visible_hw: tuple[int, int],
) -> dict[str, Any]:
    """Run the authenticated native BF16 TAEHV decoder for the two chunks."""

    from plus_decoder import load_decoder

    decoder, _state, decoder_receipt = load_decoder()
    del _state
    decoder = decoder.to(device=device, dtype=torch.bfloat16).eval()
    decoder.clean_mem()
    index = device.index
    normalized_outputs: list[torch.Tensor] = []
    calls: list[dict[str, Any]] = []
    started = time.perf_counter()
    for call, (start, stop) in enumerate(((0, 13), (13, 29))):
        condition = projection_video[:, :, start:stop].to(dtype=torch.bfloat16)
        latents = denoised_cpu[call].to(device=device, dtype=torch.bfloat16)
        torch.xpu.synchronize(index)
        call_start = time.perf_counter()
        raw = decoder.decode_video(
            latents.transpose(1, 2),
            parallel=False,
            show_progress_bar=False,
            cond=condition,
        )
        torch.xpu.synchronize(index)
        if tuple(raw.shape[2:]) != (3, projection_video.shape[-2], projection_video.shape[-1]):
            raise RuntimeError(f"unexpected native decoder call{call} shape: {tuple(raw.shape)}")
        expected_frames = (13, 16)[call]
        if raw.shape[1] != expected_frames:
            raise RuntimeError(
                f"unexpected native decoder call{call} frame count: "
                f"{raw.shape[1]} != {expected_frames}"
            )
        if not torch.isfinite(raw).all().item():
            raise FloatingPointError(f"non-finite native decoder call{call}")
        normalized = raw.mul(2).sub(1)
        if not torch.isfinite(normalized).all().item():
            raise FloatingPointError(f"non-finite normalized decoder call{call}")
        normalized_outputs.append(normalized.detach().cpu())
        calls.append(
            {
                "call": call,
                "condition_frame_range": [start, stop],
                "raw_shape": list(raw.shape),
                "normalized_shape": list(normalized.shape),
                "seconds": time.perf_counter() - call_start,
            }
        )
        del condition, latents, raw, normalized
    visible_height, visible_width = visible_hw
    if visible_height > projection_video.shape[-2] or visible_width > projection_video.shape[-1]:
        raise ValueError(f"visible geometry exceeds working decoder geometry: {visible_hw}")
    final = torch.cat(normalized_outputs, dim=1)[:, :17, :, :visible_height, :visible_width]
    final_array = _f32(final, "decoded_final")
    final_file = _write_npz(output / "final.npz", {"normalized_final_f32": final_array})
    pngs = _save_pngs(output, final_array)
    decoder.clean_mem()
    del decoder, final
    return {
        "status": "native_bf16_taehv_complete",
        "decoder": decoder_receipt,
        "calls": calls,
        "final_shape": list(final_array.shape),
        "visible_hw": [visible_height, visible_width],
        "final": final_file,
        "pngs": pngs,
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly allow B580 execution")
    parser.add_argument("--input", type=Path, required=True, help="common-input artifact root")
    parser.add_argument("--input-sha256", required=True, help="SHA256 of input manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="fresh diagnostic output leaf")
    parser.add_argument("--decode", action="store_true", help="also run the native BF16 TAEHV decoder")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("refusing native XPU execution without --run")
    if args.output.exists():
        parser.error("output must be a fresh leaf")
    args.input = args.input.resolve()
    args.output = args.output.resolve()

    started = time.perf_counter()
    require_clean_source()
    auth_started = time.perf_counter()
    weights = _auth_receipts()
    weights_authentication_seconds = time.perf_counter() - auth_started
    input_auth_started = time.perf_counter()
    preflight_hw = preflight_input_geometry(args.input, args.input_sha256)
    inputs, input_manifest = load_input(args.input, args.input_sha256)
    input_authentication_seconds = time.perf_counter() - input_auth_started
    working_hw = validate_input_geometry(input_manifest)
    if working_hw != preflight_hw:
        raise ValueError("common input geometry changed during authentication")
    device, properties = select_b580()
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "initializing",
        "scope": (
            "native pinned Turbo BF16 source trajectory on one B580 using the "
            "authenticated common-input artifact; not OpenVINO qualification or performance"
        ),
        "backend": "native_torch_xpu",
        "torch": torch.__version__,
        "device": str(device),
        "device_uuid": str(properties.uuid),
        "properties": str(properties),
        "bdf": B580_BDF,
        "source_commit": SOURCE_COMMIT,
        "source_root": str(SOURCE),
        "worker_sha256": sha256_file(Path(__file__)),
        "input": {
            "root": str(Path(args.input).resolve()),
            "manifest_sha256": args.input_sha256,
            "format": input_manifest.get("format"),
            "working_hw": list(working_hw),
            "visible_hw": input_manifest.get("visible_hw"),
            "array_shapes": {name: list(value.shape) for name, value in inputs.items()},
        },
        "weights": weights,
        "weights_authentication_seconds": weights_authentication_seconds,
        "input_authentication_seconds": input_authentication_seconds,
        "calls": [],
        "decode_requested": bool(args.decode),
    }
    model = state = scheduler = video = context = None
    denoised_cpu: list[torch.Tensor] = []
    index = device.index
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # An embedding process may already have started its inter-op pool;
            # this does not alter the native model or device contract.
            pass
        load_started = time.perf_counter()
        print("[native-xpu] loading authenticated BF16 source model", flush=True)
        model, load_receipt = build_native_model()
        report["model_load_cpu_seconds"] = time.perf_counter() - load_started
        transfer_started = time.perf_counter()
        model = model.to(device=device)
        model.LQ_proj_in.clear_cache()
        report["model_transfer_seconds"] = time.perf_counter() - transfer_started
        report["model_load"] = load_receipt
        video = torch.from_numpy(inputs["projection_video_f32"]).to(
            device=device, dtype=torch.bfloat16
        )
        context = torch.from_numpy(inputs["prompt_context_f32"]).to(
            device=device, dtype=torch.bfloat16
        )
        scheduler = FlowMatchInferenceScheduler(shift=5.0)
        timestep = scheduler.timesteps(1, device, torch.bfloat16)[0].reshape(1)
        if float(timestep.detach().cpu().item()) != 1000.0:
            raise RuntimeError(f"pinned Turbo timestep changed: {timestep}")
        spatial_windows = (working_hw[0] // 16 // 8) * (working_hw[1] // 16 // 8)
        topk = int(spatial_windows * spatial_windows * 2.5) - 1
        state = StreamState(
            attention_topk=topk,
            attention_cache_blocks=3,
            attention_block_size=2,
            attention_block_m=128,
            release_attention_workspaces=True,
        )
        report["stream"] = {
            "chunk_latents": CHUNK_LATENTS,
            "temporal_downsample": TEMPORAL_DOWNSAMPLE,
            "latent_channels": LATENT_CHANNELS,
            "spatial_windows": spatial_windows,
            "attention_topk": topk,
            "attention_cache_blocks": 3,
            "attention_block_size": 2,
            "attention_block_m": 128,
            "attention_kernel": "sdpa",
        }
        for call in range(2):
            print(f"[native-xpu] call {call} projection/denoiser start", flush=True)
            torch.xpu.synchronize(index)
            projection_started = time.perf_counter()
            conditions = _project_conditions(model, video, call)
            torch.xpu.synchronize(index)
            projection_seconds = time.perf_counter() - projection_started
            if len(conditions) != 1:
                raise RuntimeError(f"Turbo source condition count changed: {len(conditions)}")
            current = torch.from_numpy(inputs[f"noise{call}_f32"]).to(
                device=device, dtype=torch.bfloat16
            )
            state.write_cache = call == 0
            torch.xpu.synchronize(index)
            denoise_started = time.perf_counter()
            with limited_mask_observer() as masks:
                prediction = model.forward_stream(
                    current,
                    timestep,
                    context,
                    layer_conditions=conditions,
                    state=state,
                )
            denoised = scheduler.step(prediction, timestep, current)
            torch.xpu.synchronize(index)
            denoise_seconds = time.perf_counter() - denoise_started
            arrays: dict[str, np.ndarray] = {
                "lq_condition0_f32": _f32(conditions[0], "lq_condition0"),
                "prediction_f32": _f32(prediction, "prediction"),
                "denoised_f32": _f32(denoised, "denoised"),
            }
            for layer, mask in enumerate(masks):
                if mask.dtype != np.bool_:
                    raise TypeError(f"source mask {layer} is not bool: {mask.dtype}")
                arrays[f"layer{layer}_source_mask"] = np.array(mask, copy=True, order="C")
            archive = _write_npz(args.output / f"call{call}.npz", arrays)
            denoised_cpu.append(denoised.detach().to(device="cpu"))
            call_info = {
                "call": call,
                "temporal_offset": call * CHUNK_LATENTS,
                "write_cache": bool(state.write_cache),
                "archive": archive,
                "mask_count": len(masks),
                "prediction_finite": True,
                "denoised_finite": True,
                "projection_seconds": projection_seconds,
                "denoiser_seconds": denoise_seconds,
                "condition_shape": list(arrays["lq_condition0_f32"].shape),
                "prediction_shape": list(arrays["prediction_f32"].shape),
                "denoised_shape": list(arrays["denoised_f32"].shape),
            }
            report["calls"].append(call_info)
            print(json.dumps(call_info, sort_keys=True), flush=True)
            state.chunk_index = call + 1
            del conditions, current, prediction, denoised
        if args.decode:
            # Decoder has no dependency on DiT/LQ parameters or their caches.
            model = state = scheduler = context = None
            gc.collect()
            torch.xpu.empty_cache()
            report["decode"] = _decode_native(
                args.output,
                device,
                video,
                denoised_cpu,
                tuple(input_manifest["visible_hw"]),
            )
        report["status"] = STATUS
        report["diagnostic_wall_seconds"] = time.perf_counter() - started
    except BaseException as error:
        report.update(status=ERROR_STATUS, error=repr(error))
        raise
    finally:
        try:
            torch.xpu.synchronize(index)
        except BaseException:
            pass
        del model, state, scheduler, video, context
        denoised_cpu.clear()
        gc.collect()
        try:
            torch.xpu.empty_cache()
        except BaseException:
            pass
        torch.set_num_threads(old_threads)
        report.setdefault("diagnostic_wall_seconds", time.perf_counter() - started)
        report["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
