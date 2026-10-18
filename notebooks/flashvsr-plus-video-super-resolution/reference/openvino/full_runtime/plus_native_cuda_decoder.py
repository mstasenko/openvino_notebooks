"""Serial native BF16 FlashVSR+ TAEHV decoder for an authenticated CUDA call.

The source decoder's temporal upsamplers turn one latent into four output
frames.  The source-derived plan selects one startup frame and then four-frame
condition windows for the requested timeline.  Keeping the conditioning video
on CPU and transferring only one selected group is intentional: retaining the
full UHD condition tensor on CUDA needlessly consumes multiple GiB in BF16.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Sequence

import numpy as np
import torch


SOURCE_FRAMES = 17
LATENT_CALLS = 2
LATENTS_PER_CALL = 4
OUTPUT_FRAMES = 17
LATENT_CHANNELS = 16
TEMPORAL_DOWNSAMPLE = 4


def _default_plan(source_frames: int = SOURCE_FRAMES) -> dict[str, Any]:
    """Return the source-derived padded latent plan used by the default ABI."""

    if type(source_frames) is not int or source_frames < 1:
        raise ValueError("source frame count must be a positive integer")
    requested = 1 + (source_frames - 1) // TEMPORAL_DOWNSAMPLE
    padded = ((requested + LATENTS_PER_CALL - 1) // LATENTS_PER_CALL) * LATENTS_PER_CALL
    calls = []
    for call in range(padded // LATENTS_PER_CALL):
        indices = list(range(call * LATENTS_PER_CALL, (call + 1) * LATENTS_PER_CALL))
        calls.append(
            {
                "call": call,
                "latent_indices": indices,
                "projection_frame_ranges": [
                    [1 + index * TEMPORAL_DOWNSAMPLE,
                     1 + (index + 1) * TEMPORAL_DOWNSAMPLE]
                    for index in indices
                ],
                "write_cache": call + 1 < padded // LATENTS_PER_CALL,
                "temporal_offset": call * LATENTS_PER_CALL,
            }
        )
    return {
        "source_frames": source_frames,
        "requested_latents": requested,
        "padded_latents": padded,
        "projection_frames": 1 + padded * TEMPORAL_DOWNSAMPLE,
        "calls": calls,
    }


def _normalize_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the small source plan without trusting archive-derived counts."""

    if plan is None:
        plan = _default_plan()
    if not isinstance(plan, dict):
        raise TypeError("input_plan must be a dictionary")
    required = ("source_frames", "requested_latents", "padded_latents", "calls")
    if any(key not in plan for key in required):
        raise ValueError("input_plan is missing source-derived fields")
    source_frames = plan["source_frames"]
    requested = plan["requested_latents"]
    padded = plan["padded_latents"]
    calls = plan["calls"]
    if (
        type(source_frames) is not int or source_frames < 1
        or type(requested) is not int or requested < 1
        or type(padded) is not int or padded < requested
        or padded % LATENTS_PER_CALL
        or not isinstance(calls, list)
        or len(calls) != padded // LATENTS_PER_CALL
    ):
        raise ValueError("input_plan latent counts are invalid")
    expected_requested = 1 + (source_frames - 1) // TEMPORAL_DOWNSAMPLE
    if requested != expected_requested:
        raise ValueError("input_plan requested latent count is not causal")
    expected_padded = (
        (requested + LATENTS_PER_CALL - 1) // LATENTS_PER_CALL
    ) * LATENTS_PER_CALL
    if padded != expected_padded:
        raise ValueError("input_plan padded latent count is not source-derived")
    normalized_calls = []
    for call_index, item in enumerate(calls):
        if not isinstance(item, dict):
            raise ValueError("input_plan call entry is invalid")
        expected = list(range(call_index * LATENTS_PER_CALL,
                              (call_index + 1) * LATENTS_PER_CALL))
        indices = item.get("latent_indices")
        if indices is None:
            latent_start = item.get("latent_start")
            latent_stop = item.get("latent_stop")
            if latent_start != expected[0] or latent_stop != expected[-1] + 1:
                raise ValueError("input_plan latent order changed")
            indices = expected
        if indices != expected:
            raise ValueError("input_plan latent order changed")
        expected_projection_start = (
            0 if call_index == 0
            else 1 + expected[0] * TEMPORAL_DOWNSAMPLE
        )
        expected_projection_stop = 1 + (expected[-1] + 1) * TEMPORAL_DOWNSAMPLE
        if "projection_start" in item or "projection_stop" in item:
            if (
                item.get("projection_start") != expected_projection_start
                or item.get("projection_stop") != expected_projection_stop
            ):
                raise ValueError("input_plan projection chronology changed")
        ranges = item.get("projection_frame_ranges")
        if ranges is not None:
            if not isinstance(ranges, list) or len(ranges) != LATENTS_PER_CALL:
                raise ValueError("input_plan projection ranges are invalid")
            for index, pair in zip(indices, ranges):
                if (
                    not isinstance(pair, list) or len(pair) != 2
                    or pair != [1 + index * TEMPORAL_DOWNSAMPLE,
                                1 + (index + 1) * TEMPORAL_DOWNSAMPLE]
                ):
                    raise ValueError("input_plan projection chronology changed")
        write_cache = item.get("write_cache")
        if write_cache is None:
            if type(item.get("last")) is not bool:
                raise ValueError("input_plan cache policy is invalid")
            write_cache = not item["last"]
        if type(write_cache) is not bool:
            raise ValueError("input_plan cache policy is invalid")
        if "last" in item and type(item["last"]) is not bool:
            raise ValueError("input_plan terminal marker is invalid")
        expected_last = call_index + 1 == len(calls)
        if write_cache != (not expected_last):
            raise ValueError("input_plan cache policy changed")
        if "last" in item and item["last"] != expected_last:
            raise ValueError("input_plan terminal marker changed")
        if "last" in item and item["last"] == write_cache:
            raise ValueError("input_plan cache policy is inconsistent")
        normalized_calls.append(
            {
                **item,
                "call": call_index,
                "latent_indices": indices,
                "latent_start": expected[0],
                "latent_stop": expected[-1] + 1,
                "projection_start": expected_projection_start,
                "projection_stop": expected_projection_stop,
                "write_cache": write_cache,
                "last": not write_cache,
                "temporal_offset": item.get("temporal_offset", expected[0]),
            }
        )
    expected_projection = 1 + padded * TEMPORAL_DOWNSAMPLE
    if "projection_frames" in plan and plan["projection_frames"] != expected_projection:
        raise ValueError("input_plan projection length changed")
    return {
        **plan,
        "source_frames": source_frames,
        "requested_latents": requested,
        "padded_latents": padded,
        "projection_frames": expected_projection,
        "calls": normalized_calls,
    }


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"non-finite {name}")


def _validate_inputs(
    projection_video: torch.Tensor,
    denoised_cpu: Sequence[torch.Tensor],
    visible_hw: tuple[int, int],
    plan: dict[str, Any],
) -> tuple[int, int]:
    if not isinstance(projection_video, torch.Tensor):
        raise TypeError("projection_video must be a torch.Tensor")
    if projection_video.device.type != "cpu" or projection_video.dtype != torch.bfloat16:
        raise TypeError("projection_video must be CPU BF16 so the full UHD video is not retained on CUDA")
    if projection_video.ndim != 5 or projection_video.shape[:2] != (1, 3):
        raise ValueError(f"unexpected projection_video shape: {tuple(projection_video.shape)}")
    if (
        projection_video.shape[2] < plan["source_frames"]
        or any(int(v) <= 0 for v in projection_video.shape[3:])
        or any(int(v) % 8 for v in projection_video.shape[3:])
    ):
        raise ValueError(
            "projection_video must contain at least "
            f"{plan['source_frames']} frames with 8-pixel decoder geometry"
        )
    if (
        not isinstance(denoised_cpu, Sequence)
        or len(denoised_cpu) != len(plan["calls"])
        or any(not isinstance(value, torch.Tensor) for value in denoised_cpu)
    ):
        raise TypeError(
            "denoised_cpu must contain exactly "
            f"{len(plan['calls'])} tensors"
        )
    spatial = tuple(int(v) for v in projection_video.shape[3:])
    for call, value in enumerate(denoised_cpu):
        if value.device.type != "cpu" or value.dtype != torch.bfloat16:
            raise TypeError(f"denoised_cpu[{call}] must be CPU BF16")
        expected = (1, LATENT_CHANNELS, LATENTS_PER_CALL, spatial[0] // 8, spatial[1] // 8)
        if value.ndim != 5 or tuple(value.shape) != expected:
            raise ValueError(f"unexpected denoised_cpu[{call}] shape: {tuple(value.shape)} != {expected}")
        _finite(value, f"denoised_cpu[{call}]")
    height, width = visible_hw
    if (
        type(height) is not int
        or type(width) is not int
        or height <= 0
        or width <= 0
        or height > spatial[0]
        or width > spatial[1]
    ):
        raise ValueError(f"visible geometry exceeds decoder geometry: {visible_hw} > {spatial}")
    _finite(projection_video[:, :, :plan["source_frames"]], "projection_video source frames")
    return spatial


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _decode_specs(plan: dict[str, Any], grouped_latents: bool):
    """Yield either the legacy one-latent or source four-latent schedule."""

    requested = plan["requested_latents"]
    if grouped_latents:
        starts = range(0, requested, LATENTS_PER_CALL)
    else:
        starts = range(requested)
    for latent_start in starts:
        count = (
            min(LATENTS_PER_CALL, requested - latent_start)
            if grouped_latents else 1
        )
        first = latent_start == 0
        if grouped_latents:
            condition_start = 0 if first else TEMPORAL_DOWNSAMPLE * latent_start - 3
            condition_count = (
                TEMPORAL_DOWNSAMPLE * count - 3 if first
                else TEMPORAL_DOWNSAMPLE * count
            )
        else:
            condition_start = 0 if first else 1 + TEMPORAL_DOWNSAMPLE * (latent_start - 1)
            condition_count = 1 if first else TEMPORAL_DOWNSAMPLE
        yield {
            "slot": latent_start,
            "latent_call": latent_start // LATENTS_PER_CALL,
            "latent_index": latent_start % LATENTS_PER_CALL,
            "latent_start": latent_start,
            "latent_stop": latent_start + count,
            "latent_count": count,
            "condition_start": condition_start,
            "condition_stop": condition_start + condition_count,
            "output_start": 0 if first else 1 + TEMPORAL_DOWNSAMPLE * (latent_start - 1),
            "expected_frames": condition_count,
        }


def _configure_decoder_layout(
    decoder: torch.nn.Module,
    device: torch.device,
    *,
    channels_last: bool,
) -> bool:
    """Match the source TinyDecoder layout opt-in without wrapping the module."""

    if not hasattr(decoder, "use_channels_last"):
        return False
    parameter = next(decoder.parameters(), None)
    supported = (
        device.type == "cuda"
        and parameter is not None
        and parameter.dtype in (torch.float16, torch.bfloat16)
    )
    if supported:
        decoder.to(
            memory_format=(
                torch.channels_last if channels_last else torch.contiguous_format
            )
        )
        decoder.use_channels_last = bool(channels_last)
        return bool(channels_last)
    decoder.use_channels_last = False
    return False


@torch.inference_mode()
def decode_native_cuda(
    projection_video: torch.Tensor,
    denoised_cpu: Sequence[torch.Tensor],
    visible_hw: tuple[int, int],
    device: torch.device | str,
    *,
    decoder: torch.nn.Module | None = None,
    decoder_receipt: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    grouped_latents: bool = False,
    channels_last: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode the source-derived frame count with the pinned native BF16 TAEHV.

    ``projection_video`` and ``denoised_cpu`` must remain CPU-owned inputs;
    the default uploads one four-frame condition slice and one latent per slot.
    ``grouped_latents=True`` uses the source four-latent decoder schedule and
    writes each group directly into the one owning CPU output buffer.
    Returned pixels are normalized F32 ``[1,source_frames,3,H,W]`` NumPy data.
    File/PNG writes intentionally belong outside the processing clock.
    """

    plan = _normalize_plan(plan)
    spatial = _validate_inputs(projection_video, denoised_cpu, visible_hw, plan)
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"native CUDA decoder requires cpu test or cuda device, got {device}")
    owned_decoder = decoder is None
    if decoder is None:
        from plus_decoder import load_decoder

        decoder, _, decoder_receipt = load_decoder()
    if not isinstance(decoder, torch.nn.Module) or not hasattr(decoder, "decode_video"):
        raise TypeError("decoder must be the authenticated native TAEHV module")
    if not hasattr(decoder, "clean_mem"):
        raise TypeError("decoder must expose clean_mem()")

    decoder = decoder.to(device=device, dtype=torch.bfloat16).eval()
    effective_channels_last = _configure_decoder_layout(
        decoder, device, channels_last=channels_last
    )
    decoder.clean_mem()
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    # Keep one owning host destination and copy each slot into its final range.
    # A list followed by cat() would briefly require another full decoded UHD
    # tensor, which is avoidable at the 45/49-frame matched-input sizes.
    final_cpu = torch.empty(
        (1, plan["source_frames"], 3, visible_hw[0], visible_hw[1]),
        dtype=torch.float32,
        device="cpu",
    )
    try:
        for spec in _decode_specs(plan, grouped_latents):
            condition_start = spec["condition_start"]
            condition_stop = spec["condition_stop"]
            latent = denoised_cpu[spec["latent_call"]][
                :, :, spec["latent_index"] : spec["latent_index"] + spec["latent_count"]
            ]
            condition = projection_video[:, :, condition_start:condition_stop]
            _finite(condition, f"condition slot {spec['slot']}")
            _finite(latent, f"latent slot {spec['slot']}")
            condition_device = condition.to(device=device, dtype=torch.bfloat16)
            latent_device = latent.to(device=device, dtype=torch.bfloat16).transpose(1, 2)
            _sync(device)
            call_started = time.perf_counter()
            raw = decoder.decode_video(
                latent_device,
                parallel=False,
                show_progress_bar=False,
                cond=condition_device,
            )
            _sync(device)
            compute_seconds = time.perf_counter() - call_started
            expected_frames = spec["expected_frames"]
            expected_shape = (1, expected_frames, 3, spatial[0], spatial[1])
            if tuple(raw.shape) != expected_shape:
                raise RuntimeError(
                    f"unexpected native decoder group {spec['slot']} shape: "
                    f"{tuple(raw.shape)} != {expected_shape}"
                )
            if raw.dtype != torch.bfloat16:
                raise TypeError(
                    f"native decoder group {spec['slot']} returned {raw.dtype}, "
                    "expected BF16"
                )
            _finite(raw, f"decoder group {spec['slot']}")
            output_start = spec["output_start"]
            output_stop = output_start + expected_frames
            if grouped_latents:
                # The source TinyDecoder copies BF16 chunks to CPU, then
                # applies mul(2).sub(1) in BF16. Normalize in-place here to
                # preserve that rounding without a second full output buffer.
                readback_started = time.perf_counter()
                output_cpu = raw.detach().to(
                    device="cpu", dtype=torch.bfloat16
                ).contiguous()
                _sync(device)
                readback_seconds = time.perf_counter() - readback_started
                normalization_started = time.perf_counter()
                output_cpu.mul_(2).sub_(1)
                normalization_seconds = time.perf_counter() - normalization_started
                _finite(output_cpu, f"normalized group {spec['slot']}")
                final_cpu[:, output_start:output_stop].copy_(
                    output_cpu[:, :, :, : visible_hw[0], : visible_hw[1]]
                )
            else:
                _sync(device)
                normalization_started = time.perf_counter()
                normalized = raw.mul(2).sub(1)
                _sync(device)
                normalization_seconds = time.perf_counter() - normalization_started
                _finite(normalized, f"normalized group {spec['slot']}")
                readback_started = time.perf_counter()
                output_cpu = normalized.detach().to(
                    device="cpu", dtype=torch.float32
                ).contiguous()
                final_cpu[:, output_start:output_stop].copy_(
                    output_cpu[:, :, :, : visible_hw[0], : visible_hw[1]]
                )
                _sync(device)
                readback_seconds = time.perf_counter() - readback_started
            calls.append(
                {
                    "slot": spec["slot"],
                    "latent_call": spec["latent_call"],
                    "latent_index": spec["latent_index"],
                    "latent_count": spec["latent_count"],
                    "latent_indices": list(
                        range(spec["latent_start"], spec["latent_stop"])
                    ),
                    "condition_frame_range": [condition_start, condition_stop],
                    "output_frame_range": [output_start, output_stop],
                    "initial_trim_frames": 3 if spec["slot"] == 0 else 0,
                    "raw_shape": list(raw.shape),
                    "compute_seconds": compute_seconds,
                    "normalization_seconds": normalization_seconds,
                    "readback_seconds": readback_seconds,
                }
            )
            del condition_device, latent_device, raw, output_cpu
            if not grouped_latents:
                del normalized

        final = final_cpu
        if tuple(final.shape) != (1, plan["source_frames"], 3, visible_hw[0], visible_hw[1]):
            raise RuntimeError(f"unexpected native decoder final shape: {tuple(final.shape)}")
        # .numpy() retains final_cpu through ndarray.base, so the returned
        # array remains owning/safe without a second full-size copy.
        result = final.numpy()
        _finite(torch.from_numpy(result), "decoded final")
        receipt = {
            "status": f"native_bf16_taehv_serial_{plan['source_frames']}_complete",
            "dtype": "torch.bfloat16",
            "parallel": False,
            "grouped_latents": bool(grouped_latents),
            "channels_last_requested": bool(channels_last),
            "channels_last_effective": effective_channels_last,
            "source_frame_count": plan["source_frames"],
            "requested_latents": plan["requested_latents"],
            "padded_latents": plan["padded_latents"],
            "latent_slots": plan["requested_latents"],
            "decoder_calls": len(calls),
            "startup_trim_frames": 3,
            "condition_timeline": "frame0 padded by PixelShuffle3d, then source-derived 4-frame causal ranges",
            "working_hw": list(spatial),
            "visible_hw": list(visible_hw),
            "calls": calls,
            "decoder": decoder_receipt or {},
            "decode_seconds": time.perf_counter() - started,
            "compute_seconds": sum(item["compute_seconds"] for item in calls),
            "normalization_seconds": sum(item["normalization_seconds"] for item in calls),
            "readback_seconds": sum(item["readback_seconds"] for item in calls),
        }
        return result, receipt
    finally:
        decoder.clean_mem()
        if owned_decoder:
            del decoder
            gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

