"""Bounded B580 phased BF16 decoder qualification over completed latents.

Interleaved graphs remain limited to HD; retired-stage sweeps admit padded
UHD with explicit memory allowances. The worker decodes authenticated reference
latents; it does not rerun the denoiser or monkey-patch another CLI.  It is a
qualification run, not a UHD admission or throughput claim.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from check_plus_video_b580 import sha
from compare_plus_resident import (
    _cuda45_latents, _cuda45_plan, _cuda45_validate_report, _metadata_equal_exact,
    _latent_metrics,
    _psnr,
    _validate_resident_report,
    validate_resident_latents,
)
from frame_geometry import FrameGeometry, frame_pair_tiles, history_memory_tiles
from plus_pipeline import pack_decoder_input_bits, pack_decoder_inputs, pack_rgb_decoder_input_bits
from plus_parent_store import BF16ParentStore
from plus_resident import ResidentGraph, bits_tensor, tensor_bits
from plus_storage import is_finite_bf16, storage_tensor, tensor_array
from prepare_plus_video_input import input_plan, load_input
from run_plus_native_xpu import preflight_input_geometry


LIMITS = {"normalized_max_abs": 0.015625, "relative_l2": 0.005}
MAX_WORKING_SIDE = 1280
MAX_WORKING_AREA = 1280 * 768
MAX_STAGED_SIDE = 3840
MAX_STAGED_AREA = 3840 * 2176


def _check_geometry(working_hw, *, sweep_phases=False):
    side, area = ((MAX_STAGED_SIDE, MAX_STAGED_AREA) if sweep_phases
                  else (MAX_WORKING_SIDE, MAX_WORKING_AREA))
    if max(working_hw) > side or np.prod(working_hw) > area:
        raise ValueError("decoder geometry exceeds its qualified staging limit")


def _load_native_latent_reference(root, input_root, input_sha256, input_manifest,
                                  stream_plan, working_hw, report_sha256):
    """Reuse the authenticated native CUDA reference, without relabeling it."""
    root = Path(root).resolve()
    report_path = root / "results.json"
    if sha(report_path) != report_sha256:
        raise ValueError("native CUDA reference report changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_plan, native_plan = _cuda45_plan(input_manifest)
    if not _metadata_equal_exact(stream_plan, manifest_plan):
        raise ValueError("native CUDA reference requires the canonical stream plan")
    metadata, _ = _cuda45_validate_report(
        report, "latent-reference", Path(input_root).resolve(), input_sha256,
        input_manifest, native_plan)
    values, archive = _cuda45_latents(report_path, metadata["run"], tuple(working_hw))
    bits = []
    for value in values:
        packed = tensor_bits(storage_tensor(value, "bf16"))
        # The canonical loader expands raw BF16 exactly; enforce that this
        # handoff does not introduce another rounding or conversion boundary.
        expanded = (packed.astype(np.uint32) << np.uint32(16)).view(np.float32)
        if not np.array_equal(expanded.view(np.uint32), value.view(np.uint32)):
            raise ValueError("native CUDA BF16 latent round-trip changed bits")
        bits.append(packed)
    return bits, {"origin": "native_cuda_latents", "report_sha256": report_sha256,
                  **archive}


def _state_bits(shapes):
    return {name: np.zeros(shape, dtype=np.uint16) for name, shape in shapes}


def _bind_public_outputs(graphs, state_names, ov):
    """Bind decoder state/frame outputs to owning host BF16 tensors once."""

    for phase in ("front", "history0", "history1"):
        if phase not in graphs:
            continue
        names = state_names[:6] if phase == "front" else state_names[6:]
        graph = graphs[phase]
        for name in names:
            output_name = f"{name}_out"
            port = graph.outputs[output_name]
            host = ov.Tensor(port.get_element_type(), port.get_shape())
            if host.element_type != ov.Type.bf16:
                raise TypeError(f"{phase}/{output_name} must remain BF16")
            graph.bind_output(output_name, host)
    for phase, names in (("frame0", ("frame",)), ("frame1", ("frame",)),
                         ("frame_pair", ("frame0", "frame1"))):
        if phase not in graphs:
            continue
        graph = graphs[phase]
        for name in names:
            port = graph.outputs[name]
            host = ov.Tensor(port.get_element_type(), port.get_shape())
            if host.element_type != ov.Type.bf16:
                raise TypeError(f"{phase}/{name} must remain BF16")
            graph.bind_output(name, host)
    if "memory17_19" in graphs:
        graph = graphs["memory17_19"]
        for name in state_names[7:]:
            output_name = f"{name}_out"
            port = graph.outputs[output_name]
            host = ov.Tensor(port.get_element_type(), port.get_shape())
            if host.element_type != ov.Type.bf16:
                raise TypeError("fused history state must remain BF16")
            graph.bind_output(output_name, host)


def _run_states(graph, names, feeds):
    values = graph.infer(
        feeds, read=tuple(f"{name}_out" for name in names)
    )
    return values, graph.buffers["middle"]


def _read_frames(graphs, middle):
    return [
        tensor_array(bits_tensor(graphs[name].infer(
            {"middle": middle}, read=("frame",)
        )["frame"]))
        for name in ("frame0", "frame1")
    ]


def _copy_middle_bits(graph, ov):
    port = graph.outputs["middle"]
    host = ov.Tensor(port.get_element_type(), port.get_shape())
    if host.element_type != ov.Type.bf16:
        raise TypeError("phased middle must remain BF16")
    graph.buffers["middle"].copy_to(host)
    result = tensor_bits(host)
    bits_tensor(result, share=True)  # validate without cloning this private owner
    return result


def _gpu_stats(core, device):
    return (core.get_property(device, "GPU_MEMORY_STATISTICS")
            if str(device).startswith("GPU") else {})


def _timed_states(graph, names, feeds):
    started = time.perf_counter()
    values, _ = _run_states(graph, names, feeds)
    return values, time.perf_counter() - started


def _timed_middle(graph, ov):
    started = time.perf_counter()
    return _copy_middle_bits(graph, ov), time.perf_counter() - started


def _timed_frame(graph, parent):
    started = time.perf_counter()
    value = graph.infer({"middle": bits_tensor(parent)}, read=("frame",))["frame"]
    infer_seconds = time.perf_counter() - started
    started = time.perf_counter()
    value = tensor_array(bits_tensor(value))
    return value, infer_seconds, time.perf_counter() - started


def _sweep_packed_bits(video, denoised):
    for call, (start, stop) in enumerate(((0, 13), (13, 29))):
        condition = video[:, :, start:stop]
        values = np.ascontiguousarray(denoised[call])
        # The second call's final three latent windows are outside the 17-frame
        # visible output; omitting them is the causal 20-raw-frame schedule.
        for inner in range(4 if call == 0 else 1):
            yield pack_decoder_input_bits(condition, values, inner)


def _validated_disk_plan(manifest):
    """Return the exact source-derived plan admitted by disk-parent mode."""

    if not isinstance(manifest, dict):
        raise ValueError("common input manifest must be an object")
    plan = input_plan(manifest)
    if plan["source_frames"] not in (17, 45, 49):
        raise ValueError("disk-parent decoder supports source streams 17, 45, or 49")
    if plan["requested_latents"] not in (5, 12, 13):
        raise ValueError("disk-parent decoder latent count is outside the reviewed set")
    if plan["call_count"] not in (2, 3, 4):
        raise ValueError("disk-parent decoder call count is outside the reviewed set")
    return plan


def _sweep_packed_bits_plan(video, denoised, plan, *, rgb_source=None):
    """Yield only active four-latent packets from an authenticated plan."""

    if not isinstance(denoised, (list, tuple)) or len(denoised) != plan["call_count"]:
        raise ValueError("denoised chunks do not match the authenticated stream plan")
    requested = plan["requested_latents"]
    if rgb_source is not None:
        from plus_stream import build_stream_plan
        rgb_plan = build_stream_plan(plan["source_frames"], fps=1.0)
    for call in plan["calls"]:
        index = int(call["call"])
        active = max(0, min(4, requested - int(call["latent_start"])))
        condition = (video[:, :, int(call["decoder_start"]):int(call["decoder_stop"])]
                     if rgb_source is None else None)
        values = np.ascontiguousarray(denoised[index])
        for inner in range(active):
            if rgb_source is None:
                packed = pack_decoder_input_bits(condition, values, inner)
            else:
                rgb4 = rgb_source.read_decoder_inner(rgb_plan, index, inner)
                packed = pack_rgb_decoder_input_bits(rgb4, values, inner=inner)
                del rgb4
            yield index, inner, packed
            # Do not retain the previous packet while preparing the next one.
            del packed


def _valid_parent_bits(value, shape, name):
    """Validate a raw finite BF16 parent without a numeric F16 conversion."""

    value = np.asarray(value)
    shape = tuple(shape)
    if value.dtype != np.uint16 or value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous uint16 BF16 storage with shape {shape}")
    if not is_finite_bf16(value):
        raise ValueError(f"{name} contains non-finite BF16 storage")
    return value


def _disk_parent_sweep(
    front_parents, run_grow, run_memory, run_frame, *, latent_count,
    output_frames, parent_shape, frame_shape, scratch_root,
    store_factory=None, frame_to_f32=None,
    normalize=None, stage_start=None, stage_end=None, visible_hw=None,
    fuse_memories=False, normalize_raw=None, run_frame_pair=None,
    run_memory_sweep=None, run_growth_sweep=None,
):
    """Run grow, three MemBlocks, and frame children with bounded disk parents.

    The callbacks keep this storage/lifetime contract testable without loading
    the decoder.  ``run_memory`` receives the current input and the preceding
    input; both are reused only after its synchronous return.  A destination
    store is complete before the source store is closed and handed off.
    Fused memory receives three preceding inputs and returns the two new
    intermediate inputs; its first next state is the current owner itself.
    """

    if type(fuse_memories) is not bool:
        raise ValueError("fuse_memories must be boolean")
    if run_frame_pair is not None and not callable(run_frame_pair):
        raise ValueError("run_frame_pair must be callable")
    if run_memory_sweep is not None and not callable(run_memory_sweep):
        raise ValueError("run_memory_sweep must be callable")
    if run_memory_sweep is not None and not fuse_memories:
        raise ValueError("run_memory_sweep requires fuse_memories=True")
    if run_growth_sweep is not None and (not callable(run_growth_sweep)
            or not fuse_memories or run_memory_sweep is not None):
        raise ValueError("run_growth_sweep requires fused memory without a separate memory sweep")
    if normalize_raw is not None and not callable(normalize_raw):
        raise ValueError("normalize_raw must be callable")
    if normalize_raw is not None and normalize is not None:
        raise ValueError("choose normalize_raw or normalize, not both")
    if type(latent_count) is not int or latent_count not in (5, 12, 13):
        raise ValueError("latent_count must be one of 5, 12, or 13")
    if type(output_frames) is not int or output_frames not in (17, 45, 49):
        raise ValueError("output_frames must be one of 17, 45, or 49")
    if output_frames != 4 * latent_count - 3:
        raise ValueError("output_frames does not match the causal latent count")
    if len(front_parents) != latent_count:
        raise ValueError("front parent count does not match the active stream plan")
    parent_shape = tuple(parent_shape)
    frame_shape = tuple(frame_shape)
    if frame_shape[:2] != (1, 3) or len(frame_shape) != 4:
        raise ValueError("frame callback must return [1,3,H,W] BF16 storage")
    if visible_hw is None:
        visible_hw = frame_shape[-2:]
    visible_hw = tuple(visible_hw)
    if (len(visible_hw) != 2 or any(type(value) is not int or value <= 0
                                     for value in visible_hw)
            or visible_hw[0] > frame_shape[-2] or visible_hw[1] > frame_shape[-1]):
        raise ValueError("visible geometry exceeds the frame callback raster")
    if frame_to_f32 is None:
        frame_to_f32 = lambda value: tensor_array(bits_tensor(value))
    if normalize is None:
        from plus_decoder import _normalize_decoder_chunk
        normalize = _normalize_decoder_chunk

    if store_factory is None:
        store_factory = BF16ParentStore
    history_count = 2 * latent_count
    store = None
    active_store = None
    memory_sweep_receipts = {}
    growth_sweep_receipt = None
    started = time.perf_counter()
    try:
        with ExitStack() as stores:
            def make_store():
                return stores.enter_context(
                    store_factory(parent_shape, history_count, scratch_root)
                )

            store = make_store()
            growth_stage = "grow_memory" if run_growth_sweep is not None else "grow"
            if stage_start is not None:
                stage_start(growth_stage)
            if run_growth_sweep is not None:
                growth_sweep_receipt = run_growth_sweep(front_parents, store, history_count)
                if not isinstance(growth_sweep_receipt, dict):
                    raise ValueError("run_growth_sweep must return a JSON object receipt")
                try:
                    json.dumps(growth_sweep_receipt, allow_nan=False)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("run_growth_sweep returned a non-JSON-safe receipt") from exc
            else:
                for index, parent in enumerate(front_parents):
                    for child in (0, 1):
                        result = _valid_parent_bits(
                            run_grow(child, parent), parent_shape,
                            f"grow{child}[{index}]")
                        if np.shares_memory(result, parent):
                            raise ValueError("grow result aliases its parent input")
                        store.write(2 * index + child, result)
                        del result
                del parent
            store.require_complete()
            # The front graph has retired; its L host middles are no longer
            # needed once grow has durably materialized all 2L history slots.
            front_parents.clear()
            if stage_end is not None:
                stage_end(growth_stage)
            active_store = store

            stages = ("memory17_19",) if fuse_memories else ("memory17", "memory18", "memory19")
            if run_growth_sweep is not None:
                stages = ()
            for stage in stages:
                destination = make_store()
                if stage_start is not None:
                    stage_start(stage)
                if run_memory_sweep is not None:
                    if destination is active_store:
                        raise ValueError(
                            "source and destination stores must be distinct")
                    sweep_receipt = run_memory_sweep(
                        active_store, destination, history_count)
                    if not isinstance(sweep_receipt, dict):
                        raise ValueError(
                            "run_memory_sweep must return a JSON object receipt")
                    try:
                        json.dumps(sweep_receipt, allow_nan=False)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError(
                            "run_memory_sweep returned a non-JSON-safe receipt"
                        ) from exc
                    destination.require_complete()
                    memory_sweep_receipts[stage] = dict(sweep_receipt)
                    active_store.close()
                    active_store = destination
                    if stage_end is not None:
                        stage_end(stage)
                    continue
                current = np.empty(parent_shape, dtype=np.uint16)
                if fuse_memories:
                    from plus_decoder import STATE_NAMES
                    state_names = STATE_NAMES[6:]
                    past = {name: np.zeros(parent_shape, dtype=np.uint16) for name in state_names}
                else:
                    past = np.zeros(parent_shape, dtype=np.uint16)
                for index in range(history_count):
                    active_store.read_into(index, current)
                    values = run_memory(stage, current, past)
                    if fuse_memories:
                        expected = {"middle", *(f"{name}_out" for name in state_names[1:])}
                        if not isinstance(values, dict) or set(values) != expected:
                            raise ValueError("fused memory output ABI changed")
                        values = {name: _valid_parent_bits(value, parent_shape, name)
                                  for name, value in values.items()}
                        if any(np.shares_memory(value, source)
                               for value in values.values()
                               for source in (current, *past.values())):
                            raise ValueError("fused memory result aliases an input buffer")
                        owners = tuple(values.values())
                        if any(np.shares_memory(left, right)
                               for index, left in enumerate(owners)
                               for right in owners[index + 1:]):
                            raise ValueError("fused memory outputs alias each other")
                        del owners
                        result = values["middle"]
                    else:
                        result = values
                    if not fuse_memories:
                        result = _valid_parent_bits(result, parent_shape, f"{stage}[{index}]")
                    if not fuse_memories and (np.shares_memory(result, current)
                                               or np.shares_memory(result, past)):
                        raise ValueError(f"{stage} result aliases an input buffer")
                    destination.write(index, result)
                    if fuse_memories:
                        # The first history is the preceding input owner; the
                        # other two are the fused graph's intermediate inputs.
                        reusable = past[state_names[0]]
                        past = {state_names[0]: current,
                                **{name: values[f"{name}_out"] for name in state_names[1:]}}
                        current = reusable
                        del reusable
                    else:
                        past, current = current, past
                    del result, values
                destination.require_complete()
                active_store.close()
                active_store = destination
                if stage_end is not None:
                    stage_end(stage)
                del current, past

            final = np.empty((1, output_frames, 3, visible_hw[0], visible_hw[1]),
                             dtype=np.float32)
            seen = np.zeros(output_frames, dtype=bool)

            def emit(raw, visible):
                if normalize_raw is None:
                    decoded = np.asarray(frame_to_f32(raw), dtype=np.float32)
                    if decoded.shape != frame_shape or not np.isfinite(decoded).all():
                        raise ValueError("frame callback returned invalid decoded frame shape/value")
                    normalized = np.asarray(normalize(decoded), dtype=np.float32)
                else:
                    normalized = np.asarray(normalize_raw(raw))
                    if normalized.dtype != np.float32:
                        raise ValueError("raw decoder normalization must return float32")
                if normalized.shape != frame_shape:
                    raise ValueError("decoder normalization changed frame shape")
                if not np.isfinite(normalized).all():
                    raise ValueError("decoder normalization produced non-finite output")
                if seen[visible]:
                    raise ValueError(f"visible frame {visible} was written twice")
                final[:, visible] = normalized[:, :, :visible_hw[0], :visible_hw[1]]
                seen[visible] = True

            if run_frame_pair is not None:
                if stage_start is not None:
                    stage_start("frame_pair")
                # Parent zero has no visible child. Parent one has one startup
                # child, still computed/validated by the shared pair graph.
                for index in range(1, history_count):
                    values = run_frame_pair(index, active_store)
                    if (not isinstance(values, dict) or set(values) != {0, 1}
                            or any(type(key) is not int for key in values)):
                        raise ValueError("frame-pair output keys must be integers 0 and 1")
                    for child in (0, 1):
                        raw = values[child]
                        if not isinstance(raw, np.ndarray) or not raw.flags.owndata:
                            raise ValueError("frame-pair outputs must own their BF16 arrays")
                        _valid_parent_bits(raw, frame_shape, f"frame_pair{child}[{index}]")
                    if np.shares_memory(values[0], values[1]):
                        raise ValueError("frame-pair outputs alias each other")
                    for child in (0, 1):
                        visible = 2 * index + child - 3
                        if 0 <= visible < output_frames:
                            emit(values[child], visible)
                    del values, raw
                if stage_end is not None:
                    stage_end("frame_pair")
            else:
                parent = np.empty(parent_shape, dtype=np.uint16)
                for child in (0, 1):
                    stage = f"frame{child}"
                    if stage_start is not None:
                        stage_start(stage)
                    for index in range(history_count):
                        visible = 2 * index + child - 3
                        if not 0 <= visible < output_frames:
                            continue
                        active_store.read_into(index, parent)
                        raw = _valid_parent_bits(
                            run_frame(child, parent), frame_shape, f"{stage}[{index}]")
                        emit(raw, visible)
                        del raw
                    if stage_end is not None:
                        stage_end(stage)
                del parent
            if not np.all(seen):
                missing = np.flatnonzero(~seen).tolist()
                raise ValueError(f"decoder visible frame coverage is incomplete: {missing}")
            receipt = {
                "latent_count": latent_count,
                "history_parent_count": history_count,
                "output_frames": output_frames,
                "visible_indices": np.flatnonzero(seen).tolist(),
                "frame_inferences": int(np.count_nonzero(seen)),
                "skipped_frame_inferences": 2 * history_count - int(np.count_nonzero(seen)),
                "processing_seconds": time.perf_counter() - started,
                "parent_storage": "two alternating BF16ParentStore files",
            }
            if run_memory_sweep is not None:
                receipt["memory_sweep_receipts"] = memory_sweep_receipts
            if run_growth_sweep is not None:
                receipt["growth_sweep_receipt"] = growth_sweep_receipt
            if run_frame_pair is not None:
                receipt.update({
                    "pair_inferences": history_count - 1,
                    "tile_inferences": 2 * (history_count - 1),
                    "frame_inferences": 2 * (history_count - 1),
                    "skipped_frame_inferences": 2,
                    "discarded_frame_outputs": 1,
                })
            return final, receipt
    finally:
        # ExitStack closes the current store on every path.  Clear these names
        # so callbacks cannot accidentally retain a closed parent owner.
        store = active_store = None


def _phase_decode_sweep_disk(
    core, device, context, geometry, input_root, input_sha, denoised, output,
    models, phase_geometry, phase_receipt, decoder_receipt, stream_plan,
    scratch_root, setup_seconds=0.0, *, fuse_memories=False, frame_pair=False,
    remote_history=False, fuse_growth=False, growth_weights=None,
    remote_frames=False, frame_weights=None, rgb_source_manifest=None,
):
    """Run the dynamic source stream with disk-backed chronological parents."""

    from plus_decoder import (STATE_NAMES, _normalize_bf16_decoder_bits,
                              _validated_decoder_plan)
    from prepare_plus_video_input import _metadata_equal_exact
    import openvino as ov

    if type(fuse_memories) is not bool:
        raise ValueError("fuse_memories must be boolean")
    if type(frame_pair) is not bool:
        raise ValueError("frame_pair must be boolean")
    if type(remote_history) is not bool or (remote_history and not fuse_memories):
        raise ValueError("remote_history requires fuse_memories=True")
    if type(fuse_growth) is not bool or (fuse_growth and not remote_history):
        raise ValueError("fuse_growth requires remote_history=True")
    if fuse_growth != (growth_weights is not None):
        raise ValueError("fused growth requires its two child banks")
    if (type(remote_frames) is not bool
            or (remote_frames and (not fuse_growth or frame_pair))
            or remote_frames != (frame_weights is not None)):
        raise ValueError("remote frames require fused growth and their child banks, without frame pair")
    # Validate the caller's typed source chronology before opening the
    # authenticated archive.  The loaded manifest is checked against this
    # canonical plan below, but cannot be allowed to establish the plan.
    validated_plan = _validated_decoder_plan(stream_plan)
    if (validated_plan["source_frames"] not in (17, 45, 49)
            or validated_plan["requested_latents"] not in (5, 12, 13)
            or validated_plan["call_count"] not in (2, 3, 4)):
        raise ValueError("disk-parent decoder plan is outside the admitted streams")
    outer_started = time.perf_counter()
    source_started = time.perf_counter()
    rgb_source = None
    if rgb_source_manifest is None:
        selected, loaded_manifest = load_input(
            input_root, input_sha, fields=("projection_video_f32",))
        video = selected["projection_video_f32"]
        expected_video = (1, 3, int(validated_plan["projection_frames"]),
                          *geometry.working_hw)
        if video.dtype != np.float32 or tuple(video.shape) != expected_video:
            raise ValueError("selected projection video does not match the stream plan")
        if not np.isfinite(video).all():
            raise ValueError("selected projection video contains non-finite values")
    else:
        # The initial full load already authenticated every common-input field.
        # Recheck the small manifest, not the unused multi-GB NPZ source member.
        raw_manifest = (Path(input_root).resolve() / "manifest.json").read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != input_sha:
            raise ValueError("common input manifest changed before RGB decoder handoff")
        loaded_manifest = json.loads(raw_manifest)
        if not _metadata_equal_exact(loaded_manifest, rgb_source_manifest):
            raise ValueError("authenticated RGB decoder receipt changed")
        source = loaded_manifest["source_rgb"]
        video = selected = None
    source_reload_seconds = time.perf_counter() - source_started
    loaded_plan = _validated_disk_plan(loaded_manifest)
    if not _metadata_equal_exact(loaded_plan, validated_plan):
        raise ValueError("authenticated decoder plan changed during selective load")
    stream_plan = validated_plan

    memories = _state_bits(zip(STATE_NAMES[:6], tuple(phase_geometry["state_shapes"][:6])))
    groups = []
    active = {}
    stage_data = {}

    def start_stage(name):
        names = ("grow0", "grow1") if name == "grow" else (name,)
        if name == "grow_frames":
            names = ("grow_memory", "memory_finite", "remote_frame_child")
        if name == "grow_memory" or (name == "memory17_19" and remote_history):
            names = (name, "memory_finite")
        before = _gpu_stats(core, device)
        graphs, compile_seconds = _compile_sweep_graphs(
            core, device, context, models, names, STATE_NAMES, ov,
            **({"remote_history": True} if remote_history else {}),
        )
        stage_data[name] = {
            "before": before, "graphs": graphs, "compiled": names,
            "compile_seconds": compile_seconds, "infer_seconds": 0.0,
            "read_seconds": 0.0, "started": time.perf_counter(),
        }
        active[name] = graphs

    def end_stage(name):
        data = stage_data[name]
        report = _retire_sweep_group(
            core, device, data["graphs"], name, data["compiled"],
            2 * int(stream_plan["requested_latents"]),
            data["compile_seconds"], data["infer_seconds"],
            data["read_seconds"], data["before"],
        )
        elapsed = time.perf_counter() - data["started"]
        report["processing_seconds"] = elapsed
        report["disk_parent_io_included"] = True
        groups.append(report)
        active.pop(name, None)

    before = _gpu_stats(core, device)
    front_graphs, compile_seconds = _compile_sweep_graphs(
        core, device, context, models, ("front",), STATE_NAMES, ov
    )
    front_started = time.perf_counter()
    front_infer = front_read = 0.0
    front_parents = []
    try:
        if rgb_source_manifest is not None:
            from plus_rgb_source import RGBSourceWindowReader
            source_started = time.perf_counter()
            rgb_source = RGBSourceWindowReader(
                frames_path=source["payload_path"], manifest_path=source["manifest_path"],
                input_receipt=source, target_hw=(geometry.height, geometry.width))
            source_reload_seconds += time.perf_counter() - source_started
        for _, _, packed in _sweep_packed_bits_plan(
                video, denoised, stream_plan, rgb_source=rgb_source):
            # These owners remain live until the synchronous inference returns;
            # wrapping their BF16 bytes avoids another full host-side copy.
            front, seconds = _timed_states(
                front_graphs["front"], STATE_NAMES[:6],
                {"packed": bits_tensor(packed, share=True),
                 **{name: bits_tensor(memories[name], share=True) for name in STATE_NAMES[:6]}},
            )
            front_infer += seconds
            memories.update({name: front[f"{name}_out"] for name in STATE_NAMES[:6]})
            parent, seconds = _timed_middle(front_graphs["front"], ov)
            front_parents.append(parent)
            front_read += seconds
            del packed
    except Exception:
        front_graphs.clear()
        raise
    finally:
        if rgb_source is not None:
            rgb_source.close()
    del front, parent
    groups.append(_retire_sweep_group(
        core, device, front_graphs, "front", ("front",),
        int(stream_plan["requested_latents"]), compile_seconds,
        front_infer, front_read, before,
        int(sum(value.nbytes for value in front_parents)),
    ))
    groups[-1]["processing_seconds"] = time.perf_counter() - front_started
    groups[-1]["disk_parent_io_included"] = False
    del memories, selected, video
    gc.collect()

    parent_shape = tuple(phase_geometry["state_shapes"][6])
    frame_shape = (1, 3, int(geometry.working_hw[0]), int(geometry.working_hw[1]))
    pair_tiles = frame_pair_tiles(geometry.working_hw) if frame_pair else None
    tile_input = None

    def run_grow(child, parent):
        stamp = time.perf_counter()
        value = active["grow"][f"grow{child}"].infer(
            {"middle": bits_tensor(parent, share=True)}, read=("middle",)
        )["middle"]
        stage_data["grow"]["infer_seconds"] += time.perf_counter() - stamp
        return value

    def run_memory(stage, current, past):
        stamp = time.perf_counter()
        feeds = ({name: bits_tensor(value, share=True) for name, value in past.items()}
                 if fuse_memories else {"past": bits_tensor(past, share=True)})
        read = ("middle", *(f"{name}_out" for name in STATE_NAMES[7:])) if fuse_memories else ("middle",)
        value = active[stage][stage].infer(
            {"current": bits_tensor(current, share=True), **feeds}, read=read)
        stage_data[stage]["infer_seconds"] += time.perf_counter() - stamp
        return value if fuse_memories else value["middle"]

    def run_frame(child, parent):
        stamp = time.perf_counter()
        value = active[f"frame{child}"][f"frame{child}"].infer(
            {"middle": bits_tensor(parent, share=True)}, read=("frame",)
        )["frame"]
        stage_data[f"frame{child}"]["infer_seconds"] += time.perf_counter() - stamp
        return value

    def run_memory_sweep(source, destination, count):
        from plus_history_ring import sweep_history
        graphs = active["memory17_19"]
        receipt = sweep_history(
            core, device, graphs["memory17_19"], graphs["memory_finite"],
            geometry.working_hw, source, destination, count)
        stage_data["memory17_19"]["infer_seconds"] += receipt["infer_seconds"]
        return receipt

    def run_growth_sweep(source, destination, count):
        from plus_history_ring import sweep_history
        graphs = active["grow_memory"]
        receipt = sweep_history(
            core, device, graphs["grow_memory"], graphs["memory_finite"],
            geometry.working_hw, source, destination, count,
            growth_weights=growth_weights)
        stage_data["grow_memory"]["infer_seconds"] += receipt["infer_seconds"]
        return receipt

    def run_frame_pair(index, store):
        nonlocal tile_input
        if tile_input is None:
            tile_input = np.empty(pair_tiles["middle_shape"], dtype=np.uint16)
        frames = {child: np.empty(frame_shape, dtype=np.uint16) for child in (0, 1)}
        graph = active["frame_pair"]["frame_pair"]
        for tile in pair_tiles["tiles"]:
            store.read_rows_into(index, *tile["input_rows"], tile_input)
            stamp = time.perf_counter()
            graph.infer({"middle": bits_tensor(tile_input, share=True)}, read=())
            stage_data["frame_pair"]["infer_seconds"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            start, stop = tile["output_rows"]
            target_start, target_stop = tile["frame_rows"]
            for child in (0, 1):
                # Host-bound outputs are overwritten by the next tile. Copy
                # their valid interior to independent full-frame owners now.
                raw = tensor_bits(graph.buffers[f"frame{child}"], copy=False)
                frames[child][..., target_start:target_stop, :] = raw[..., start:stop, :]
            stage_data["frame_pair"]["read_seconds"] += time.perf_counter() - stamp
        return frames

    try:
        if remote_frames:
            from plus_history_ring import sweep_growth_frames
            start_stage("grow_frames")
            graphs = active["grow_frames"]
            final, parent_receipt = sweep_growth_frames(
                core, device, graphs["grow_memory"], graphs["memory_finite"],
                graphs["remote_frame_child"], geometry.working_hw, front_parents,
                growth_weights, frame_weights,
                output_frames=int(stream_plan["output_frames"]),
                visible_hw=(int(geometry.height), int(geometry.width)))
            stage_data["grow_frames"]["infer_seconds"] = sum(
                parent_receipt["stages"][name] for name in ("growth", "finite", "frame"))
            front_parents.clear()
            end_stage("grow_frames")
            groups[-1]["disk_parent_io_included"] = False
        else:
            final, parent_receipt = _disk_parent_sweep(
                front_parents, run_grow, run_memory, run_frame,
                latent_count=int(stream_plan["requested_latents"]),
                output_frames=int(stream_plan["output_frames"]),
                parent_shape=parent_shape, frame_shape=frame_shape,
                scratch_root=scratch_root,
                normalize_raw=_normalize_bf16_decoder_bits,
                stage_start=start_stage, stage_end=end_stage,
                visible_hw=(int(geometry.height), int(geometry.width)),
                fuse_memories=fuse_memories,
                run_frame_pair=run_frame_pair if frame_pair else None,
                run_memory_sweep=run_memory_sweep if remote_history and not fuse_growth else None,
                run_growth_sweep=run_growth_sweep if fuse_growth else None,
            )
    finally:
        for data in stage_data.values():
            data["graphs"].clear()
        active.clear()
        gc.collect()
    del front_parents
    compile_seconds = sum(float(item.get("compile_seconds", 0.0)) for item in groups)
    outer_seconds = time.perf_counter() - outer_started
    processing = outer_seconds - compile_seconds
    if (not np.isfinite(outer_seconds) or not np.isfinite(compile_seconds)
            or not np.isfinite(processing) or outer_seconds < 0.0
            or processing < 0.0 or processing > outer_seconds):
        raise RuntimeError("invalid phased decoder processing/compile timing")
    report = {
        "scope": "dynamic source-ordered phased decoder with disk BF16 parents",
        "decoder": decoder_receipt, "phase": phase_receipt,
        "phase_names": [item["name"] for item in groups],
        "sweep_latents": int(stream_plan["requested_latents"]),
        "raw_frames_before_trim": 4 * int(stream_plan["requested_latents"]),
        "output_frames": int(stream_plan["output_frames"]),
        "input_plan": stream_plan,
        "normalization_callback": "_normalize_bf16_decoder_bits",
        "fuse_memories": fuse_memories,
        "frame_pair": frame_pair,
        "frame_pair_tiles": pair_tiles,
        "remote_history": remote_history,
        "fuse_growth": fuse_growth,
        "remote_frames": remote_frames,
        "state_storage": ("six front host BF16 states and fifty-one tiled GPU history roots"
                          if remote_frames else
                          "six front host BF16 states and twenty-seven tiled GPU history roots"
                          if fuse_growth else
                          "six front host BF16 states and fifteen tiled GPU history roots"
                          if remote_history else
                          "six front and three fused-history host BF16 states"
                          if fuse_memories else "six front host BF16 states")
                         + ("; expanded parents remain on GPU" if remote_frames else
                            "; chronological parents use two alternating disk stores"),
        "state_output_buffers": {
            "front": [f"{name}_out" for name in STATE_NAMES[:6]],
            **({"remote_frame_child": ["frame"]} if remote_frames else
               {"frame_pair": ["frame0", "frame1"]} if frame_pair
               else {"frame0": ["frame"], "frame1": ["frame"]}),
            **({"grow_memory": ["S0", "S1", "S2"]} if fuse_growth else
               {"memory17_19": [f"{name}_out" for name in STATE_NAMES[7:]]}
               if fuse_memories else {}),
        },
        "middle_handoff": ("same-context BF16 remote growth/history -> frame child; no parent store"
                           if remote_frames else
                           "growth and history write one final BF16ParentStore"
                           if fuse_growth else
                           "grow interleaves 2L parents into one BF16ParentStore; MemBlocks ping-pong stores"),
        "private_outputs_gpu_only": [f"residual_linear_{i}" for i in range(9)],
        "host_bound_outputs": [*[f"{name}_out" for name in STATE_NAMES[:6]],
                               *([f"{name}_out" for name in STATE_NAMES[7:]]
                                 if fuse_memories and not remote_history else []),
                               *([] if remote_frames else
                                 ["frame0", "frame1"] if frame_pair else ["frame"])],
        "frames_readback": ("sixteen full-halo tiles; normalize/scatter valid cores directly"
                            if remote_frames else
                            "two row tiles per parent; two independent raw BF16 frames"
                            if frame_pair else
                            "one parent slot and one frame child at a time; direct visible scatter"),
        "phase_groups": groups,
        "parent_storage": parent_receipt,
        "shape": list(final.shape),
        "source_reload_seconds": source_reload_seconds,
        "source_transport": "bounded_rgb4" if rgb_source_manifest is not None else "normalized_f32_npz",
        "graph_setup_seconds": setup_seconds,
        "setup_seconds": float(setup_seconds) + compile_seconds,
        "outer_seconds_before_archive": outer_seconds,
        "compile_seconds_excluded": compile_seconds,
        "stage_seconds_sum": sum(float(item.get("processing_seconds", 0.0))
                               for item in groups),
        "seconds": processing,
        "matched_processing_seconds": processing,
        "gpu_statistics": _gpu_stats(core, device),
    }
    # The caller owns archive/hash persistence so a continuous controller can
    # keep this returned array in its single processing clock without a giant
    # NPZ write or reread inside the decoder core.
    return final, report


def _compile_sweep_graphs(core, device, context, models, names, state_names, ov,
                          *, remote_history=False):
    print(json.dumps({"stage": "compile_decoder", "graphs": list(names)}), flush=True)
    started = time.perf_counter()
    graphs = {}
    for name in names:
        model = models.pop(name)
        config = ({"INFERENCE_PRECISION_HINT": ov.Type.f32}
                  if name == "memory_finite" else None)
        graphs[name] = ResidentGraph(core, model, device, context=context,
                                    **({"config": config} if config else {}))
        if not (remote_history and name == "memory17_19"):
            _bind_public_outputs({name: graphs[name]}, state_names, ov)
        del model
    return graphs, time.perf_counter() - started


def _retire_sweep_group(core, device, graphs, name, compiled, parents,
                        compile_seconds, infer_seconds, read_seconds, before,
                        retained_host_middle_bytes=0):
    after_infer = _gpu_stats(core, device)
    graphs.clear()
    gc.collect()
    report = {
        "name": name, "compiled_graphs": list(compiled),
        "latent_parents": parents, "compile_seconds": compile_seconds,
        "infer_seconds": infer_seconds, "middle_read_seconds": read_seconds,
        "gpu_statistics_before": before, "gpu_statistics_after_infer": after_infer,
        "gpu_statistics_after_retirement": _gpu_stats(core, device),
    }
    if any(int(value) for value in report["gpu_statistics_after_retirement"].values()):
        raise RuntimeError(f"Decoder group {name} retained GPU allocations after retirement")
    if retained_host_middle_bytes:
        report["retained_host_middle_bytes"] = retained_host_middle_bytes
    print(json.dumps({"stage": "retired_decoder", "group": name,
                      "infer_seconds": infer_seconds}), flush=True)
    return report


def _phase_decode_sweep(
    core, device, context, geometry, video, denoised, output,
    models, phase_geometry, phase_receipt, decoder_receipt,
):
    """Run front/history/frame groups with host BF16 parent retention."""

    from plus_decoder import STATE_NAMES
    import openvino as ov

    memories = _state_bits(zip(STATE_NAMES, tuple(phase_geometry["state_shapes"])))
    groups, front_parents = [], []
    before = _gpu_stats(core, device)
    front_graphs, compile_seconds = _compile_sweep_graphs(
        core, device, context, models, ("front",), STATE_NAMES, ov
    )
    infer_seconds = read_seconds = 0.0
    for value in _sweep_packed_bits(video, denoised):
        front, seconds = _timed_states(
            front_graphs["front"], STATE_NAMES[:6],
            {"packed": bits_tensor(value),
             **{name: bits_tensor(memories[name]) for name in STATE_NAMES[:6]}},
        )
        infer_seconds += seconds
        memories.update({name: front[f"{name}_out"] for name in STATE_NAMES[:6]})
        parent, seconds = _timed_middle(front_graphs["front"], ov)
        front_parents.append(parent)
        read_seconds += seconds
    del value, front
    groups.append(_retire_sweep_group(
        core, device, front_graphs, "front", ("front",), 5,
        compile_seconds, infer_seconds, read_seconds, before,
        int(sum(value.nbytes for value in front_parents)),
    ))
    del memories

    before = _gpu_stats(core, device)
    grow_graphs, compile_seconds = _compile_sweep_graphs(
        core, device, context, models, ("grow0", "grow1"), STATE_NAMES, ov
    )
    infer_seconds = read_seconds = 0.0
    history_parents = []
    for parent in front_parents:
        for name in ("grow0", "grow1"):
            stamp = time.perf_counter()
            history_parents.append(grow_graphs[name].infer(
                {"middle": bits_tensor(parent)}, read=("middle",))["middle"])
            infer_seconds += time.perf_counter() - stamp
    groups.append(_retire_sweep_group(
        core, device, grow_graphs, "grow", ("grow0", "grow1"), 5,
        compile_seconds, infer_seconds, read_seconds, before,
        int(sum(value.nbytes for value in history_parents)),
    ))
    del front_parents, parent

    # Each MemBlock needs the preceding INPUT at that same block, not the
    # preceding output. Sweep source blocks in order, retaining their complete
    # temporal sequence in compact host BF16, with just one GPU block live.
    for name in ("memory17", "memory18", "memory19"):
        before = _gpu_stats(core, device)
        memory_graphs, compile_seconds = _compile_sweep_graphs(
            core, device, context, models, (name,), STATE_NAMES, ov)
        following, past = [], np.zeros_like(history_parents[0])
        infer_seconds = 0.0
        for current in history_parents:
            stamp = time.perf_counter()
            following.append(memory_graphs[name].infer(
                {"current": bits_tensor(current), "past": bits_tensor(past)},
                read=("middle",))["middle"])
            infer_seconds += time.perf_counter() - stamp
            past = current
        history_parents = following
        del current, past
        groups.append(_retire_sweep_group(
            core, device, memory_graphs, name, (name,), 10,
            compile_seconds, infer_seconds, 0.0, before,
            int(sum(value.nbytes for value in history_parents))))

    # Frame children have no temporal state. Sweep them separately instead of
    # retaining two full-resolution convolution workspaces on the GPU.
    final = np.empty((1, 17, 3, geometry.height, geometry.width), np.float32)
    for child, name in enumerate(("frame0", "frame1")):
        before = _gpu_stats(core, device)
        frame_graphs, compile_seconds = _compile_sweep_graphs(
            core, device, context, models, (name,), STATE_NAMES, ov)
        infer_seconds = read_seconds = 0.0
        for index, parent in enumerate(history_parents):
            value, infer, read = _timed_frame(frame_graphs[name], parent)
            visible = 2 * index + child - 3
            if 0 <= visible < 17:
                final[:, visible] = value[:, :, :geometry.height, :geometry.width]
            infer_seconds += infer
            read_seconds += read
        groups.append(_retire_sweep_group(
            core, device, frame_graphs, name, (name,), 10,
            compile_seconds, infer_seconds, read_seconds, before))
    normalized = tensor_array(storage_tensor(
        tensor_array(storage_tensor(final * np.float32(2), "bf16")) - np.float32(1), "bf16"
    ))
    final_path = output / "decoded_final.npz"
    np.savez_compressed(final_path, normalized_final_f32=normalized)
    return {
        "scope": "up to padded UHD decoder sweep; named buffers only, no allocator peak claim",
        "decoder": decoder_receipt, "phase": phase_receipt,
        "phase_names": [item["name"] for item in groups], "sweep_latents": 5,
        "raw_frames_before_trim": 20,
        "state_storage": "front six host BF16 histories; each memory stage carries its preceding input in chronological order",
        "state_output_buffers": {
            "front": [f"{name}_out" for name in STATE_NAMES[:6]],
            "frame0": ["frame"], "frame1": ["frame"],
        },
        "middle_handoff": "five front plus ten history parent middles copied to host BF16",
        "private_outputs_gpu_only": [f"residual_linear_{i}" for i in range(9)],
        "host_bound_outputs": [*[f"{name}_out" for name in STATE_NAMES[:6]], "frame"],
        "frames_readback": "one frame child resident at a time; public host BF16 readback",
        "phase_groups": groups, "shape": list(normalized.shape),
        "sha256": sha(final_path), "seconds": sum(
            item["compile_seconds"] + item["infer_seconds"] + item["middle_read_seconds"]
            for item in groups
        ), "gpu_statistics": _gpu_stats(core, device),
    }


def _phase_decode(core, device, context, geometry, video, denoised, output,
                  *, sweep_phases=False, disk_parents=False,
                  input_root=None, input_sha=None, stream_plan=None,
                  scratch_root=None, fuse_memories=False, frame_pair=False,
                  remote_history=False, fuse_growth=False, remote_frames=False,
                  rgb_source_manifest=None):
    """Decode legacy resident chunks or an authenticated staged stream."""

    from plus_decoder import STATE_NAMES, load_decoder
    from plus_phased_decoder import (
        build_frame_pair, build_phased_decoder, build_history_stages,
        build_growth_history, build_remote_frame_child, phased_geometry,
    )

    _check_geometry(geometry.working_hw, sweep_phases=sweep_phases)
    if disk_parents and not sweep_phases:
        raise ValueError("disk-parent decoding requires phased sweeps")
    if rgb_source_manifest is not None and not disk_parents:
        raise ValueError("bounded RGB source requires disk-parent decoding")
    if rgb_source_manifest is not None and not isinstance(rgb_source_manifest, dict):
        raise ValueError("bounded RGB source requires the authenticated common manifest")
    if type(fuse_memories) is not bool or (fuse_memories and not disk_parents):
        raise ValueError("fused memory requires disk-parent decoding")
    if type(frame_pair) is not bool or (frame_pair and not disk_parents):
        raise ValueError("frame pair requires disk-parent decoding")
    if type(remote_history) is not bool or (remote_history and not (disk_parents and fuse_memories)):
        raise ValueError("remote history requires fused disk-parent decoding")
    if type(fuse_growth) is not bool or (fuse_growth and not remote_history):
        raise ValueError("fused growth requires remote history")
    if type(remote_frames) is not bool or (remote_frames and (not fuse_growth or frame_pair)):
        raise ValueError("remote frames require fused growth without frame pair")
    setup_started = time.perf_counter()
    decoder, state, decoder_receipt = load_decoder()
    del state
    models, phase_receipt = build_phased_decoder(decoder, geometry.working_hw)
    phase_geometry = phased_geometry(geometry.working_hw)
    if sweep_phases:
        del models["history0"], models["history1"]
        history_tiles = (history_memory_tiles(
            geometry.working_hw, **({"tile_count": 16, "halo_rows": 12} if remote_frames else
                                    {"tile_count": 8} if fuse_growth else {}))
            if remote_history else None)
        growth_weights = None
        frame_weights = None
        if fuse_growth:
            model, growth_weights, history_receipt = build_growth_history(
                decoder, tuple(history_tiles["tile_source_hw"]))
            history_models = {"grow_memory": model}
            del model
        else:
            history_models, history_receipt = build_history_stages(
                decoder, geometry.working_hw, fuse_memories=fuse_memories,
                **({"memory_source_hw": tuple(history_tiles["tile_source_hw"])}
                   if remote_history else {}))
        models.update(history_models)
        if remote_history:
            from plus_history_ring import build_state_finite
            models["memory_finite"] = build_state_finite(history_tiles["tile_middle_shape"])
        phase_receipt = {**phase_receipt, "history_stages": history_receipt}
        if remote_frames:
            del models["frame0"], models["frame1"]
            models["remote_frame_child"], frame_weights, frame_receipt = build_remote_frame_child(
                decoder, tuple(history_tiles["tile_source_hw"]))
            phase_receipt = {**phase_receipt, "remote_frame_child": frame_receipt}
        if frame_pair:
            del models["frame0"], models["frame1"]
            models["frame_pair"], pair_receipt = build_frame_pair(decoder, geometry.working_hw)
            phase_receipt = {**phase_receipt, "frame_pair": pair_receipt}
        if disk_parents:
            if input_root is None or input_sha is None or stream_plan is None:
                raise ValueError("disk-parent mode requires authenticated input plan")
            result = _phase_decode_sweep_disk(
                core, device, context, geometry, input_root, input_sha,
                denoised, output, models, phase_geometry, phase_receipt,
                decoder_receipt, stream_plan, scratch_root,
                setup_seconds=time.perf_counter() - setup_started,
                fuse_memories=fuse_memories,
                frame_pair=frame_pair,
                remote_history=remote_history,
                fuse_growth=fuse_growth, growth_weights=growth_weights,
                remote_frames=remote_frames, frame_weights=frame_weights,
                rgb_source_manifest=rgb_source_manifest,
            )
        else:
            result = _phase_decode_sweep(
                core, device, context, geometry, video, denoised, output,
                models, phase_geometry, phase_receipt, decoder_receipt,
            )
        del decoder
        return result
    graphs = {
        name: ResidentGraph(core, models[name], device, context=context)
        for name in ("front", "history0", "history1", "frame0", "frame1")
    }
    import openvino as ov
    _bind_public_outputs(graphs, STATE_NAMES, ov)
    del models, decoder

    state_shapes = tuple(phase_geometry["state_shapes"])
    memories = _state_bits(zip(STATE_NAMES, state_shapes))
    state_receipts = []
    frames = []
    started = time.perf_counter()
    for call, (start, stop) in enumerate(((0, 13), (13, 29))):
        values = tensor_array(bits_tensor(denoised[call]))
        packed = pack_decoder_inputs(video[:, :, start:stop], values)
        call_frames = []
        for inner in range(4):
            front, front_middle = _run_states(graphs["front"], STATE_NAMES[:6],
                {"packed": storage_tensor(packed[:, inner], "bf16"),
                 **{name: bits_tensor(memories[name]) for name in STATE_NAMES[:6]}},
            )
            for name in STATE_NAMES[:6]:
                memories[name] = front[f"{name}_out"]

            history0, history0_middle = _run_states(graphs["history0"], STATE_NAMES[6:],
                {"middle": front_middle,
                 **{name: bits_tensor(memories[name]) for name in STATE_NAMES[6:]}},
            )
            first_frames = _read_frames(graphs, history0_middle)

            history1, history1_middle = _run_states(graphs["history1"], STATE_NAMES[6:],
                {"middle": front_middle,
                 **{name: bits_tensor(history0[f"{name}_out"])
                    for name in STATE_NAMES[6:]}},
            )
            second_frames = _read_frames(graphs, history1_middle)
            for name in STATE_NAMES[6:]:
                memories[name] = history1[f"{name}_out"]
            for name in STATE_NAMES:
                bits_tensor(memories[name])  # reject non-finite GPU state at its owner boundary
            call_frames.append(np.stack([*first_frames, *second_frames], axis=1))
            state_receipts.append({
                "call": call, "inner": inner,
                "state_sha256": {
                    name: hashlib.sha256(np.ascontiguousarray(memories[name]).tobytes()).hexdigest()
                    for name in STATE_NAMES
                },
            })
        call_output = np.concatenate(call_frames, axis=1)
        frames.append(call_output[:, 3:] if call == 0 else call_output)

    final = np.concatenate(frames, axis=1)[:, :17, :, :geometry.height, :geometry.width]
    normalized = tensor_array(
        storage_tensor(
            tensor_array(storage_tensor(final * np.float32(2), "bf16")) - np.float32(1),
            "bf16",
        )
    )
    final_path = output / "decoded_final.npz"
    np.savez_compressed(final_path, normalized_final_f32=normalized)
    private = sorted(
        name for graph in graphs.values() for name in graph.buffers
        if name.startswith("residual_linear_")
    )
    return {
        "scope": "padded HD resident phased decoder qualification; no UHD admission claim",
        "decoder": decoder_receipt,
        "phase": phase_receipt,
        "phase_names": ["front", "history0", "history1", "frame0", "frame1"],
        "state_storage": "nine host BF16 state arrays; front six carried once per latent; history three history0->history1",
        "middle_handoff": "same-context remote buffers; no host middle readback",
        "private_outputs_gpu_only": private,
        "host_bound_outputs": [
            *[f"{name}_out" for name in STATE_NAMES], "frame"
        ],
        "frames_readback": "frame0/frame1 public outputs read to host BF16 for final crop",
        "state_receipts": state_receipts,
        "shape": list(normalized.shape),
        "sha256": sha(final_path),
        "seconds": time.perf_counter() - started,
        "gpu_statistics": (core.get_property(device, "GPU_MEMORY_STATISTICS")
                           if str(device).startswith("GPU") else {}),
    }


def _compare(actual, expected, output, reference_sha, input_sha, phase_report):
    """Compare only decoder output; the input latent bits are shared exactly."""

    normalized = _psnr(expected, actual, 2.0)
    final_metrics = _latent_metrics(expected, actual)
    within = (final_metrics["normalized_max_abs"] <= LIMITS["normalized_max_abs"]
              and final_metrics["relative_l2"] <= LIMITS["relative_l2"])
    comparison = {
        "format": "flashvsr-plus-phased-decoder-b580-comparison-v1",
        "status": "pass" if within else "mismatch",
        "scope": "padded HD source-ordered phased BF16 decoder over shared reference latents",
        "quality_gate": "legacy resident decoder limits; no exact equality requirement",
        "limits": LIMITS,
        "reference_sha256": reference_sha,
        "input_sha256": input_sha,
        "decoded": {"normalized": normalized, "metrics": final_metrics},
        "phase_decoder": phase_report,
    }
    path = output / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return comparison


def _disk_memory_preflight(core, device, geometry, stream_plan, *,
                           fuse_memories=False, frame_pair=False, remote_history=False,
                           fuse_growth=False, remote_frames=False, rgb_source_shape=None):
    """Check named device/host/disk budgets before compiling any phase graph."""

    from dataclasses import asdict
    from plus_memory import staged_decoder_memory
    from plus_parent_store import DISK_RESERVE_BYTES, _scratch_root
    from run_plus_resident import _host_memory_budget

    plans = staged_decoder_memory(
        geometry,
        latent_count=int(stream_plan["requested_latents"]),
        parent_storage="disk",
        source_frames=int(stream_plan["projection_frames"]),
        output_frames=int(stream_plan["output_frames"]),
        extra_device_bytes=4 << 30,
        extra_host_bytes=6 << 30,
        fuse_memories=fuse_memories,
        frame_pair=frame_pair,
        remote_history=remote_history,
        fuse_growth=fuse_growth,
        remote_frames=remote_frames,
        rgb_source_shape=rgb_source_shape,
    )
    device_total = int(core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE"))
    initial_stats = core.get_property(device, "GPU_MEMORY_STATISTICS")
    allocated_device = sum(
        int(initial_stats.get(key, 0))
        for key in ("cl_mem", "usm_device", "usm_shared", "unknown")
    )
    device_budget = device_total - allocated_device - (2 << 30)
    host_total = int(_host_memory_budget())
    host_budget = host_total - (2 << 30)
    if device_budget <= 0 or host_budget <= 0:
        raise MemoryError("host/device budget is below the required reserve")
    if any(plan.required_device_bytes > device_budget
           or plan.required_host_bytes > host_budget for plan in plans):
        raise MemoryError("staged decoder named buffers plus workspace allowance exceed available budget")

    scratch = _scratch_root()
    disk_values = {int(plan.disk_storage_bytes) for plan in plans}
    reserve_values = {int(plan.disk_reserve_bytes) for plan in plans}
    if len(disk_values) != 1 or len(reserve_values) != 1:
        raise ValueError("staged disk ledger changed between decoder groups")
    disk_storage = disk_values.pop()
    disk_reserve = reserve_values.pop()
    if disk_reserve < DISK_RESERVE_BYTES:
        raise MemoryError("staged parent disk reserve is below 16 GiB")
    disk_required = disk_storage + disk_reserve
    disk_free = int(shutil.disk_usage(scratch).free)
    if disk_free < disk_required:
        raise MemoryError(
            f"staged parent scratch space is insufficient: need {disk_required} bytes"
        )

    records = []
    for plan in plans:
        record = asdict(plan)
        record.update({
            "required_device_bytes": int(plan.required_device_bytes),
            "required_host_bytes": int(plan.required_host_bytes),
            "disk_storage_bytes": int(plan.disk_storage_bytes),
            "disk_reserve_bytes": int(plan.disk_reserve_bytes),
        })
        records.append(record)
    return {
        "groups": records,
        "device_budget": device_budget,
        "device_total_bytes": device_total,
        "device_allocated_bytes_before": allocated_device,
        "host_budget": host_budget,
        "scope": "named buffers plus explicit allowances, not an allocator peak proof",
        "scratch_root": str(scratch),
        "disk_storage_bytes": disk_storage,
        "disk_reserve_bytes": disk_reserve,
        "disk_required_bytes": disk_required,
        "disk_free_bytes_before": disk_free,
        "stream_plan": stream_plan,
    }, scratch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweep-phases", action="store_true",
                        help="run five-latent front/history/frame compile-retire sweep")
    parser.add_argument("--disk-parents", action="store_true",
                        help="use authenticated dynamic stream and disk BF16 parents")
    parser.add_argument("--fuse-memories", action="store_true",
                        help="keep the three consecutive history blocks in one graph")
    parser.add_argument("--frame-pair", action="store_true",
                        help="decode both frame children together using bounded row tiles")
    parser.add_argument("--remote-history", action="store_true",
                        help="retain fused decoder history in bounded GPU row tiles")
    parser.add_argument("--fuse-growth", action="store_true",
                        help="combine tiled growth/history without intermediate disk parents")
    parser.add_argument("--remote-frames", action="store_true",
                        help="consume GPU history tiles directly without expanded disk parents")
    parser.add_argument("--denoiser-only-reference", action="store_true",
                        help="decode saved latents without claiming a decoder quality comparison")
    parser.add_argument("--native-latent-reference", action="store_true",
                        help="reference contains authenticated native CUDA latents; diagnostic only")
    args = parser.parse_args(argv)
    if args.native_latent_reference:
        if not (args.sweep_phases and args.disk_parents):
            parser.error("Native latent reference requires staged disk decoding")
        args.denoiser_only_reference = True
    if not args.run or args.output.exists():
        parser.error("Explicit --run and a fresh output leaf required")
    if args.denoiser_only_reference and not args.sweep_phases:
        parser.error("Denoiser-only reference requires the bounded staged decoder")
    if args.disk_parents and not args.sweep_phases:
        parser.error("Disk-parent mode requires the bounded staged decoder")
    if args.disk_parents and not args.denoiser_only_reference:
        parser.error("Disk-parent mode initially requires a denoiser-only reference")
    if args.fuse_memories and not args.disk_parents:
        parser.error("Fused memory requires disk-parent mode")
    if args.frame_pair and not args.disk_parents:
        parser.error("Frame pair requires disk-parent mode")
    if args.remote_history and not (args.disk_parents and args.fuse_memories):
        parser.error("Remote history requires fused disk-parent mode")
    if args.fuse_growth and not args.remote_history:
        parser.error("Fused growth requires remote history")
    if args.remote_frames and (not args.fuse_growth or args.frame_pair):
        parser.error("Remote frames require fused growth without frame pair")
    working_hw = preflight_input_geometry(
        args.input, args.input_sha256,
        max_working_side=MAX_STAGED_SIDE if args.sweep_phases else MAX_WORKING_SIDE,
        allow_stream=args.disk_parents,
    )
    _check_geometry(working_hw, sweep_phases=args.sweep_phases)
    if args.disk_parents:
        manifest_path = Path(args.input) / "manifest.json"
        input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stream_plan = _validated_disk_plan(input_manifest)
        arrays = None
    else:
        arrays, input_manifest = load_input(args.input, args.input_sha256)
        stream_plan = None
    geometry = FrameGeometry(*input_manifest["visible_hw"])
    if tuple(geometry.working_hw) != tuple(working_hw):
        raise ValueError("common input geometry changed during authentication")
    native_receipt = None
    if args.native_latent_reference:
        bits, native_receipt = _load_native_latent_reference(
            args.reference, args.input, args.input_sha256, input_manifest,
            stream_plan, working_hw, args.reference_sha256)
        expected = None
    else:
        if sha(args.reference / "results.json") != args.reference_sha256:
            raise ValueError("Completed resident reference report changed")
        reference = json.loads((args.reference / "results.json").read_text(encoding="utf-8"))
        if args.denoiser_only_reference:
            latents, _ = validate_resident_latents(args.reference, reference,
                args.input_sha256, (geometry.height, geometry.width), working_hw,
                plan=stream_plan)
            expected = None
        else:
            latents, expected, _ = _validate_resident_report(
                args.reference, reference, args.input_sha256,
                (geometry.height, geometry.width), working_hw, plan=stream_plan)
        bits = [tensor_bits(storage_tensor(value, "bf16")) for value in latents]
        del latents

    import openvino as ov
    from attention_from_cuda import resolve
    core = ov.Core()
    device, inventory = resolve(core)
    if device != "GPU.1" or not ov.get_version().startswith("2026.5."):
        raise ValueError("Qualified exact B580/runtime required")
    memory = None
    scratch_root = None
    if args.sweep_phases:
        if args.disk_parents:
            memory, scratch_root = _disk_memory_preflight(
                core, device, geometry, stream_plan, fuse_memories=args.fuse_memories,
                frame_pair=args.frame_pair,
                remote_history=args.remote_history,
                fuse_growth=args.fuse_growth,
                remote_frames=args.remote_frames,
            )
        else:
            from dataclasses import asdict
            from plus_memory import staged_decoder_memory
            plans = staged_decoder_memory(geometry, extra_device_bytes=4 << 30,
                                          extra_host_bytes=6 << 30)
            device_budget = int(core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE")) - (2 << 30)
            meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            host_budget = int(meminfo["MemAvailable"].split()[0]) * 1024 - (2 << 30)
            if any(plan.required_device_bytes > device_budget or plan.required_host_bytes > host_budget
                   for plan in plans):
                raise MemoryError("Staged decoder named buffers plus workspace allowance exceed available budget")
            memory = {"groups": [asdict(plan) for plan in plans],
                      "device_budget": device_budget, "host_budget": host_budget,
                      "scope": "named buffers plus explicit allowances, not an allocator peak proof"}
    context = core.get_default_context(device)
    args.output.mkdir(parents=True, exist_ok=False)
    phase_result = _phase_decode(
        core, device, context, geometry,
        None if args.disk_parents else arrays["projection_video_f32"], bits, args.output,
        disk_parents=args.disk_parents, input_root=args.input,
        input_sha=args.input_sha256, stream_plan=stream_plan,
        scratch_root=scratch_root,
        sweep_phases=args.sweep_phases,
        fuse_memories=args.fuse_memories,
        frame_pair=args.frame_pair,
        remote_history=args.remote_history,
        fuse_growth=args.fuse_growth,
        remote_frames=args.remote_frames,
    )
    if args.disk_parents:
        actual, phase_report = phase_result
        # Disk-parent mode returns its owning output directly.  Archive and
        # hash only after the measured processing interval has ended.
        final_path = args.output / "decoded_final.npz"
        np.savez_compressed(final_path, normalized_final_f32=actual)
        phase_report["sha256"] = sha(final_path)
    else:
        phase_report = phase_result
        with np.load(args.output / "decoded_final.npz", allow_pickle=False) as archive:
            actual = np.array(archive["normalized_final_f32"], copy=True)
    if expected is None:
        comparison = {"status": "not_compared", "reason": "reference contains denoiser latents only",
                      "input_sha256": args.input_sha256, "reference_sha256": args.reference_sha256}
        if native_receipt is not None:
            comparison.update(reference_origin="native_cuda_latents",
                              native_reference=native_receipt)
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        from run_plus_native_xpu import _save_pngs
        phase_report["pngs"] = _save_pngs(args.output, actual)
    else:
        comparison = _compare(actual, expected, args.output, args.reference_sha256,
                              args.input_sha256, phase_report)
    report = {
        "status": "complete_phased_decoder_b580",
        "device": device,
        "runtime": ov.get_version(),
        "inventory": inventory,
        "input_sha256": args.input_sha256,
        "reference_sha256": args.reference_sha256,
        "worker_sha256": sha(Path(__file__)),
        "decoded": phase_report,
        "comparison": comparison,
        "memory_inventory": memory,
    }
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": comparison["status"],
                      "phase_seconds": phase_report["seconds"],
                      "comparison": str((args.output / "comparison.json").resolve())}),
          flush=True)
    if comparison["status"] == "mismatch":
        raise AssertionError("Phased decoder exceeded the legacy resident component limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
