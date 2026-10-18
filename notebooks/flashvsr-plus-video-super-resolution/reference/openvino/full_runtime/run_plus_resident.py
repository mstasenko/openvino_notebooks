"""Finite larger-video Plus pilot with resident weights and bounded token work.

This is the measured integration path, not an installed OTC backend.
The legacy --decode path stays HD-only. --decode --disk-decoder uses retired
decoder stages with bounded disk-backed BF16 parents for padded UHD streams.
"""
import argparse
import faulthandler
import gc
import importlib.util
import json
import shutil
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "chunked_attention"))
sys.path.insert(0, str(HERE.parents[1]))
from experiment_memory import process_snapshot
from prepare_plus_video_input import call_array_names, input_plan, load_input
from frame_geometry import FrameGeometry
from plus_pipeline import Graph, join_history
from plus_resident import ResidentGraph, ResidentAttention, bits_tensor, tensor_bits
from plus_storage import storage_tensor, tensor_array
from plus_graphs import BlockGeometry, build_bounded_pre, build_post_rows, preserve_modulation_product
from plus_bounded import infer_pre, infer_post
from plus_edges import load_edge_weights, build_prefix, build_suffix
from plus_projection import project_video_conditions
from plus_text import build_text_embedding, build_cross_kv, load_text_weights
from plus_weights import CHECKPOINT, authenticate_dit, load_layer
from plus_modes import resolve_plus_mode
from plus_masks import build_stream_sparse_layout
from check_plus_video_b580 import sha


def port_name(key):
    return "weight__" + key.replace(".", "_")


def _check_geometry(working_hw, *, decode=False, disk_decoder=False,
                    fuse_memories=False, frame_pair=False, remote_history=False,
                    fuse_growth=False, remote_frames=False):
    if type(frame_pair) is not bool or (frame_pair and not disk_decoder):
        raise ValueError("Frame pair requires --disk-decoder")
    if type(remote_history) is not bool or (remote_history and not (disk_decoder and fuse_memories)):
        raise ValueError("Remote history requires --disk-decoder and --fuse-memories")
    if type(fuse_growth) is not bool or (fuse_growth and not remote_history):
        raise ValueError("Fused growth requires --remote-history")
    if type(remote_frames) is not bool or (remote_frames and (not fuse_growth or frame_pair)):
        raise ValueError("Remote frames require --fuse-growth without --frame-pair")
    if fuse_memories and not disk_decoder:
        raise ValueError("Fused memory requires --disk-decoder")
    if disk_decoder and not decode:
        raise ValueError("Disk decoder requires --decode")
    if max(working_hw) > 3840 or int(np.prod(working_hw)) > 2176 * 3840:
        raise ValueError("Resident pilot exceeds reviewed padded-UHD geometry")
    if decode and not disk_decoder and max(working_hw) > 1280:
        raise ValueError("The unphased decoder remains limited to padded HD")


def _check_decode_plan(plan, *, decode=False):
    """Validate the shared decoder chronology before opening a device."""

    if not isinstance(plan, dict) or type(plan.get("source_frames")) is not int:
        raise ValueError("authenticated input plan is missing source_frames")
    if decode:
        from plus_decoder import _validated_decoder_plan
        _validated_decoder_plan(plan)


def _host_memory_budget(
    *,
    meminfo_path=Path("/proc/meminfo"),
    cgroup_file=Path("/proc/self/cgroup"),
    cgroup_root=Path("/sys/fs/cgroup"),
):
    """Return total host budget bounded by the current unified cgroup."""

    try:
        available = None
        for line in Path(meminfo_path).read_text(encoding="ascii").splitlines():
            fields = line.split()
            if fields[:1] == ["MemAvailable:"]:
                if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
                    raise RuntimeError("MemAvailable is malformed")
                available = int(fields[1]) * 1024
                break
        if available is None or available <= 0:
            raise RuntimeError("MemAvailable is missing")

        unified = []
        for line in Path(cgroup_file).read_text(encoding="ascii").splitlines():
            fields = line.split(":", 2)
            if len(fields) == 3 and fields[1] == "":
                unified.append(fields[2])
        if len(unified) != 1:
            raise RuntimeError("unified cgroup path is missing or ambiguous")
        relative = unified[0].strip()
        if not relative.startswith("/") or "\x00" in relative:
            raise RuntimeError("unified cgroup path is malformed")
        root = Path(cgroup_root).resolve(strict=True)
        relative_path = Path(relative.lstrip("/"))
        if any(part in ("", ".", "..") for part in relative_path.parts):
            raise RuntimeError("unified cgroup path traversal")
        group = (root / relative_path).resolve(strict=True)
        try:
            group.relative_to(root)
        except ValueError as error:
            raise RuntimeError("unified cgroup path escapes cgroup root") from error
        maximum = (group / "memory.max").read_text(encoding="ascii").strip()
        if maximum == "max":
            return available
        if not maximum.isdigit() or int(maximum) <= 0:
            raise RuntimeError("memory.max is malformed")
        return min(available, int(maximum))
    except (OSError, UnicodeError) as error:
        raise RuntimeError("unable to read unified host memory budget") from error


def typed(graph, arrays):
    import openvino as ov
    # These feeds are consumed before ResidentGraph.infer returns. Keep their
    # NumPy owners via typed Tensor views instead of allocating a second copy.
    result = {}
    for name, value in arrays.items():
        if isinstance(value, (ov.Tensor, ov.RemoteTensor)):
            result[name] = value  # ResidentGraph validates shape/type/context.
        elif graph.inputs[name].get_element_type() == ov.Type.bf16:
            result[name] = bits_tensor(np.require(value, requirements=["C", "W"]), share=True)
        else:
            result[name] = ov.Tensor(np.ascontiguousarray(value), shared_memory=False)
    return result


def _check_remote_context(remote_context, remote_pre, rows):
    if type(remote_context) is not bool or type(remote_pre) is not bool:
        raise ValueError("Remote boundary options must be boolean")
    if remote_context and (not remote_pre or type(rows) is not int or rows != 4096):
        raise ValueError("Remote context requires --remote-pre and rows=4096")


def _validate_layout_options(rope_rank4, cross_query_chunk_tokens, *, remote_pre, rows,
                              ffn2_matmul=False, cross_precise=False, remote_context=False,
                              remote_tokens=False, ffn2_state=False):
    """Validate graph-layout selectors before input, weight, or device access."""

    if type(rope_rank4) is not bool:
        raise ValueError("--rope-rank4 must be boolean")
    if type(remote_pre) is not bool or type(remote_context) is not bool:
        raise ValueError("--remote-pre and --remote-context must be boolean")
    if type(ffn2_matmul) is not bool:
        raise ValueError("--ffn2-matmul must be boolean")
    if type(remote_tokens) is not bool:
        raise ValueError("--remote-tokens must be boolean")
    if type(ffn2_state) is not bool:
        raise ValueError("--ffn2-state must be boolean")
    if ffn2_state and (
            not cross_precise or not ffn2_matmul or not remote_pre or
            not remote_context or not remote_tokens or rows != 4096):
        raise ValueError(
            "FFN2 state requires --cross-precise --ffn2-matmul --remote-pre "
            "--remote-context --remote-tokens and rows=4096/native BF16")
    if type(cross_precise) is not bool or (cross_precise and
            (not remote_pre or not remote_context or rows != 4096 or cross_query_chunk_tokens != 128)):
        raise ValueError("Precise cross requires C4096 remote PRE/context and the default query chunk")
    if type(cross_query_chunk_tokens) is not int:
        raise ValueError("cross-query chunk tokens must be an integer")
    if cross_query_chunk_tokens not in (128, 256, 512):
        raise ValueError("cross-query chunk tokens must be one of 128, 256, or 512")
    if rope_rank4 and (not remote_pre or rows != 4096):
        raise ValueError("Rank-4 RoPE requires --remote-pre and --rows 4096")


def _rank4_rope_model(model):
    """Apply the authenticated research rewrite only when explicitly selected."""

    helper = HERE.parents[1] / "evidence" / "rope_rank4_pre_20261017.py"
    if not helper.is_file():
        raise FileNotFoundError(f"rank-4 RoPE helper is missing: {helper}")
    spec = importlib.util.spec_from_file_location("_otc_research_rope_rank4", helper)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load rank-4 RoPE helper: {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rank4_rope(model)


def _build_resident_models(geometry, rows, *, remote_pre, remote_context,
                           rope_rank4, cross_query_chunk_tokens, ffn2_matmul=False,
                           state_barriers=False, cross_precise=False,
                           remote_tokens=False, ffn2_state=False):
    """Build PRE/POST with the selected, ABI-preserving graph variants."""

    if type(state_barriers) is not bool or (state_barriers and not remote_pre):
        raise ValueError("State barriers require a boolean opt-in and --remote-pre")
    _validate_layout_options(rope_rank4, cross_query_chunk_tokens,
                             remote_pre=remote_pre, rows=rows, ffn2_matmul=ffn2_matmul,
                             cross_precise=cross_precise, remote_context=remote_context,
                             remote_tokens=remote_tokens, ffn2_state=ffn2_state)
    model, token_geometry, residual = build_bounded_pre(geometry, rows=rows)
    preserve_modulation_product(model, linear_stores=True, norm_stores=True,
                                state_sinks=state_barriers)
    if remote_pre:
        from device_query_graph import expose_device_queries
        expose_device_queries(model, remote_kv=True, source_type="bf16")
    if rope_rank4:
        model = _rank4_rope_model(model)
    post_model = build_post_rows(
        rows, remote_context=remote_context,
        cross_query_chunk_tokens=cross_query_chunk_tokens,
        ffn2_matmul=ffn2_matmul,
    )
    if cross_precise:
        sys.path.insert(0, str(HERE.parents[1] / "evidence"))
        from post_cross_bridge_20261017 import rewrite_post_cross_bridge
        from cross_precise_20261017 import rewrite_clone
        post_model = rewrite_clone(rewrite_post_cross_bridge(
            post_model, ffn2_matmul=ffn2_matmul), s1_count=2)
    if ffn2_state:
        sys.path.insert(0, str(HERE.parents[1] / "evidence"))
        from post_ffn2_state_20261017 import replace_ffn2_results_with_assigns
        # This must remain after the precise clone: the state sinks observe the
        # exact graph that the selected cross-bridge variant will compile.
        post_model, _ = replace_ffn2_results_with_assigns(post_model)
    return model, token_geometry, residual, post_model


def _ffn2_state_metadata(model):
    """Validate and serialize the opt-in POST Assign ABI without readback."""

    helper = HERE.parents[1] / "evidence" / "post_ffn2_state_20261017.py"
    sys.path.insert(0, str(helper.parent))
    from post_ffn2_state_20261017 import FFN2_PRIVATE_OUTPUTS, _validate_assign_states
    import openvino as ov

    _validate_assign_states(model)
    if (len(model.get_variables()) != 2 or len(model.get_sinks()) != 2
            or tuple(output.get_any_name() for output in model.outputs) != ("x_out",)
            or any(node.get_type_name() == "ReadValue" for node in model.get_ordered_ops())):
        raise ValueError("FFN2 state requires only x_out and two overwriting Assign sinks")
    variables = {item.info.variable_id: item for item in model.get_variables()}
    sinks = {item.get_variable_id(): item for item in model.get_sinks()}
    states = []
    for state_id in sorted(variables):
        output = sinks[state_id].output(0)
        states.append({
            "state_id": state_id,
            "source_output": output.get_any_name(),
            "shape": [int(value) for value in output.get_shape()],
            "element_type": ("bf16" if output.get_element_type() == ov.Type.bf16
                             else str(output.get_element_type())),
            "assign_type": sinks[state_id].get_type_name(),
            "read_value_feedback": False,
        })
    return {
        "helper_sha256": sha(helper),
        "public_outputs": [output.get_any_name() for output in model.outputs],
        "private_outputs": list(FFN2_PRIVATE_OUTPUTS),
        "state_ids": sorted(variables),
        "states": states,
        "no_read_value": not any(
            node.get_type_name() == "ReadValue" for node in model.get_ordered_ops()),
        "memory_credit_bytes": 0,
    }


def _history_prefix_bytes(histories, layer):
    if layer >= len(histories) or histories[layer] is None:
        return None
    previous = histories[layer].view()
    if previous is None:
        return None
    return tuple(previous[name].nbytes for name in ("k", "v"))


def _prefetch_history(store, histories, layer):
    """Read only an existing valid prefix; never page in unused capacity."""
    sizes = _history_prefix_bytes(histories, layer)
    if sizes is None:
        return 0.0
    started = time.perf_counter()
    store.prefetch(layer, active_bytes=sizes)
    return time.perf_counter() - started


def _flush_prefetch_history(store, layer, next_sizes):
    """Disk-only worker: immutable coordinates, no GPU or mutable editor state."""
    store.flush(layer)
    if next_sizes is None:
        return 0.0
    started = time.perf_counter()
    store.prefetch(layer + 1, active_bytes=next_sizes)
    return time.perf_counter() - started


def _wait_history_io(pending, report):
    if pending is not None:
        started = time.perf_counter()
        prefetched = pending.result()  # Propagate failures before reusing maps.
        report["history_io_wait_seconds"] += time.perf_counter() - started
        report["history_prefetch_seconds"] += prefetched
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--mode", choices=("turbo", "base"), default="turbo")
    parser.add_argument("--disk-history", action="store_true",
                        help="retain exact BF16 history on disk with one active mapped layer")
    parser.add_argument("--prefetch-history", action="store_true",
                        help="read the next valid disk-history prefix ahead of GPU work")
    parser.add_argument("--overlap-history", action="store_true",
                        help="overlap one bounded history flush with current GPU work")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--disk-decoder", action="store_true",
                        help="retire decoder stages and keep large BF16 parents on disk")
    parser.add_argument("--rgb-decoder", action="store_true",
                        help="read bounded authenticated RGB windows instead of reloading F32 source")
    parser.add_argument("--rgb-input", action="store_true",
                        help="include RGB source resize/normalization in the processing clock")
    parser.add_argument("--fuse-memories", action="store_true",
                        help="process consecutive decoder history blocks together")
    parser.add_argument("--frame-pair", action="store_true",
                        help="share the final frame-child prefix with bounded row tiles")
    parser.add_argument("--remote-history", action="store_true",
                        help="retain fused decoder history in bounded GPU row tiles")
    parser.add_argument("--fuse-growth", action="store_true",
                        help="combine tiled growth/history without intermediate disk parents")
    parser.add_argument("--remote-frames", action="store_true",
                        help="consume GPU history tiles directly without expanded disk parents")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--remote-pre", action="store_true",
                        help="keep Q and fresh K/V on the owned GPU between pre and attention")
    parser.add_argument("--rope-rank4", action="store_true",
                        help="use the authenticated rank-4 RoPE PRE rewrite (C4096 remote-pre only)")
    parser.add_argument("--cross-query-chunk-tokens", type=int, choices=(128, 256, 512), default=128,
                        help="POST cross-attention query chunk size; default 128")
    parser.add_argument("--ffn2-matmul", action="store_true",
                        help="use token-major FFN2 with source BF16 storage results")
    parser.add_argument("--cross-precise", action="store_true",
                        help="qualify accurate BF16 cross attention with two fewer resident weight banks")
    parser.add_argument("--remote-context", action="store_true",
                        help="reuse consumed GPU query slots for the attention-to-post handoff")
    parser.add_argument("--remote-tokens", action="store_true",
                        help="keep x on GPU; stream layers26..29 through one weight bank")
    parser.add_argument("--compact-kv", action="store_true",
                        help="read only retained local K/V history slabs back to the host")
    parser.add_argument("--retained-kv-roi", action="store_true",
                        help="copy retained GPU K/V directly into existing host history")
    parser.add_argument("--split-key", action="store_true",
                        help="qualify four-way FP32 split-key attention (changes BF16 rounding)")
    parser.add_argument("--binary-prefix", action="store_true",
                        help="find the existing packed-ID prefix with an exact binary search")
    parser.add_argument("--gpu-model", choices=("B580", "A770"), default="B580",
                        help="exact-BDF Intel GPU model; B580 remains the default")
    parser.add_argument("--state-barriers", action="store_true",
                        help="keep exact PRE BF16 rounding in six GPU-only Assign stores")
    parser.add_argument("--ffn2-state", action="store_true",
                        help="replace the two precise POST FFN2 Results with GPU-only Assign states")
    parser.add_argument("--bounded-prefix", action="store_true",
                        help="combine raw prefix/condition bits in bounded token upload slabs")
    parser.add_argument("--profile-operators", action="store_true",
                        help="diagnostic cumulative-average pre/post operator timings")
    args = parser.parse_args()
    if not args.run or args.output.exists() or args.rows not in range(128, 8193, 128):
        parser.error("Explicit --run, fresh output and128..8192 whole-window rows required")
    try:
        if args.binary_prefix and not args.split_key:
            raise ValueError("Binary prefix requires --split-key")
        if args.state_barriers and not args.remote_pre:
            raise ValueError("State barriers require --remote-pre")
        if args.overlap_history and not args.prefetch_history:
            raise ValueError("History I/O overlap requires --prefetch-history")
        if args.prefetch_history and not args.disk_history:
            raise ValueError("History read-ahead requires --disk-history")
        if args.disk_history and not (args.remote_pre and args.compact_kv and args.retained_kv_roi):
            raise ValueError("Disk history requires the compact retained-ROI remote-pre lifecycle")
        _check_remote_context(args.remote_context, args.remote_pre, args.rows)
        if args.rgb_decoder and not (args.decode and args.disk_decoder):
            raise ValueError("RGB decoder requires --decode and --disk-decoder")
        if args.rgb_input and not args.rgb_decoder:
            raise ValueError("RGB input requires --rgb-decoder")
        _validate_layout_options(args.rope_rank4, args.cross_query_chunk_tokens,
                                 remote_pre=args.remote_pre, rows=args.rows,
                                 ffn2_matmul=args.ffn2_matmul, cross_precise=args.cross_precise,
                                 remote_context=args.remote_context,
                                 remote_tokens=args.remote_tokens, ffn2_state=args.ffn2_state)
        if args.cross_precise and not args.remote_tokens:
            raise ValueError("Precise cross requires --remote-tokens")
        if args.remote_tokens and not args.remote_context:
            raise ValueError("Remote tokens require --remote-context")
        if args.bounded_prefix and not args.remote_tokens:
            raise ValueError("Bounded prefix requires --remote-tokens")
        if args.compact_kv and not args.remote_pre:
            raise ValueError("Compact K/V requires --remote-pre")
        if args.retained_kv_roi and not args.compact_kv:
            raise ValueError("Retained K/V ROI requires --compact-kv and --remote-pre")
        if args.split_key and (not args.remote_pre or (args.decode and not args.disk_decoder)):
            raise ValueError("Split-key requires --remote-pre and denoiser-only or --disk-decoder")
        if args.compact_kv and args.decode and not args.disk_decoder:
            raise ValueError("Compact K/V requires denoiser-only or --disk-decoder")
        if args.remote_frames and (not args.fuse_growth or args.frame_pair):
            raise ValueError("Remote frames require --fuse-growth without --frame-pair")
    except ValueError as error:
        parser.error(str(error))
    faulthandler.enable()
    from run_plus_native_xpu import preflight_input_geometry
    working_hw = preflight_input_geometry(args.input, args.input_sha256,
                                         max_working_side=3840, allow_stream=True)
    try:
        _check_geometry(working_hw, decode=args.decode, disk_decoder=args.disk_decoder,
                        fuse_memories=args.fuse_memories, frame_pair=args.frame_pair,
                        remote_history=args.remote_history, fuse_growth=args.fuse_growth,
                        remote_frames=args.remote_frames)
    except ValueError as error:
        parser.error(str(error))
    try:
        # The manifest hash and geometry were authenticated by the read-only
        # preflight above; validate the source-derived schedule before opening
        # the large input archive or probing a device.
        manifest = json.loads((args.input.resolve() / "manifest.json").read_text(
            encoding="utf-8"
        ))
        stream_plan = input_plan(manifest)
        _check_decode_plan(stream_plan, decode=args.decode)
        if args.disk_decoder:
            from check_plus_phased_decoder_b580 import _validated_disk_plan
            _validated_disk_plan(manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    fields = tuple(name for name in manifest["inputs"]["arrays"]
                   if name != "projection_video_f32") if args.rgb_input else None
    arrays, input_receipt = load_input(args.input, args.input_sha256, fields=fields)
    loaded_plan = input_plan(input_receipt)
    if loaded_plan != stream_plan:
        raise ValueError("common input temporal plan changed during authentication")
    stream_plan = loaded_plan
    geometry = FrameGeometry(*input_receipt["visible_hw"])
    if args.remote_pre and (args.rows != 4096 or BlockGeometry(geometry).tokens < 4096):
        parser.error("Remote pre requires the reviewed4096-row/C16 layout")
    mode = resolve_plus_mode(args.mode)
    checkpoint = (CHECKPOINT if args.mode == "turbo" else
                  mode.helper_paths(CHECKPOINT.parent.parent)["dit"])
    weights_receipt = authenticate_dit(checkpoint, mode=args.mode)
    history_store = None
    history_pool = pending_history_io = None
    histories = []
    history_shapes = {}
    history_disk_bytes = 0
    if args.disk_history:
        sys.path.insert(0, str(HERE.parents[1] / "matched_benchmark"))
        from disk_kv import DiskKv, DISK_RESERVE_BYTES, _scratch_root
        for layer in range(30):
            shape = (mode.retain_groups(layer) * geometry.spatial_windows, 128, 1536)
            history_shapes[layer] = (shape, shape)
            history_disk_bytes += 2 * int(np.prod(shape)) * 2
        # Refuse insufficient scratch before allocating any model/device data.
        if shutil.disk_usage(_scratch_root()).free < history_disk_bytes + DISK_RESERVE_BYTES:
            raise MemoryError("Insufficient scratch for exact Plus history and its 16GiB reserve")
    import openvino as ov
    from safetensors import safe_open
    from attention_from_cuda import resolve
    core = ov.Core()
    device, inventory = resolve(core, expected_name=args.gpu_model)
    if ((args.gpu_model == "B580" and device != "GPU.1")
            or not ov.get_version().startswith("2026.5.")):
        raise ValueError("Exact selected Intel GPU/runtime required")
    from plus_memory import admit_plus_memory, POST_CROSS_BRIDGE_RESIDENT_BANKS
    from dataclasses import asdict
    initial_stats = core.get_property(device, "GPU_MEMORY_STATISTICS")
    # These are plugin allocations, not a system-wide free-memory oracle.
    # The gate separately rejects other owners before this process starts.
    allocated = sum(int(initial_stats.get(key, 0)) for key in ("cl_mem", "usm_device", "usm_shared", "unknown"))
    total = int(core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE"))
    memory = admit_plus_memory(geometry,
        mode=args.mode, disk_history=args.disk_history,
        prefetched_history=args.prefetch_history,
        available_device_bytes=max(0, total - allocated),
        available_host_bytes=_host_memory_budget(),
        resident_weight_layers=30, workspace_rows=min(args.rows, BlockGeometry(geometry).tokens),
        stream_call_count=stream_plan["call_count"],
        remote_pre=args.remote_pre,
        remote_tokens=args.remote_tokens,
        compact_kv=args.compact_kv,
        retained_kv_roi=args.retained_kv_roi,
        ffn2_matmul=args.ffn2_matmul,
        post_ffn2_state=args.ffn2_state,
        post_cross_bridge=args.cross_precise,
        split_key=args.split_key,
        reserve_device_bytes=2 << 30, reserve_host_bytes=2 << 30,
        # The 1440p pilot exposed plugin-owned buffers beyond named tensors.
        # Keep that measured headroom until the request inventory is reduced.
        # Keep the old conservative workspace allowance. The optional context
        # path reuses Q roots, but budget the finite-check workspace and both
        # bounded post-layout intermediates without assuming compiler fusion.
        extra_device_bytes=(4 << 30) + ((36 << 20) if args.remote_context else 0),
        extra_host_bytes=4 << 30, extra_buffers_complete=True,
        phase_grouping="all_disjoint", phase_lifetimes_disjoint=True,
        host_phase_grouping="resident" if args.remote_pre and
            (not args.decode or args.disk_decoder) else None,
        decoder_phase=args.decode and not args.disk_decoder,
        decoder_weight_bytes=90_676_742 if args.decode and not args.disk_decoder else 0,
        decoder_output_frames=stream_plan["output_frames"]
            if args.decode and not args.disk_decoder else None)
    decoder_memory = scratch_root = None
    if args.disk_decoder:
        from check_plus_phased_decoder_b580 import _disk_memory_preflight
        decoder_memory, scratch_root = _disk_memory_preflight(
            core, device, geometry, stream_plan, fuse_memories=args.fuse_memories,
            frame_pair=args.frame_pair, remote_history=args.remote_history,
            fuse_growth=args.fuse_growth, remote_frames=args.remote_frames,
            rgb_source_shape=tuple(input_receipt["source_rgb"]["shape"])
                if args.rgb_decoder else None)
    context = core.get_default_context(device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "initializing", "scope": __doc__, "runtime": ov.get_version(),
              "device": device, "inventory": inventory, "input_sha256": args.input_sha256,
              "weights": weights_receipt, "worker_sha256": sha(__file__), "calls": [],
              "mode": args.mode, "disk_history": args.disk_history,
              "gpu_model": args.gpu_model, "state_barriers": args.state_barriers,
              "prefetch_history": args.prefetch_history,
              "overlap_history": args.overlap_history,
              "history_io_wait_seconds": 0.0,
              "history_io_clock": "flush/prefetch may overlap GPU work; residual wait and whole processing span are measured",
              "history_prefetch_seconds": 0.0,
              "history_disk_bytes": history_disk_bytes,
              "input_plan": stream_plan,
              "rope_layout": "rank4" if args.rope_rank4 else "original_rank5",
              "cross_query_chunk_tokens": args.cross_query_chunk_tokens,
              "cross_precise": args.cross_precise,
              "ffn2_linear": "matmul_stores" if args.ffn2_matmul else "convolution",
              "ffn2_private_outputs_gpu_only": bool(args.ffn2_matmul and not args.ffn2_state),
              "ffn2_state": {"enabled": bool(args.ffn2_state)}}
    report["prompt"] = input_receipt["prompt"]
    report["memory_inventory"] = asdict(memory)
    if args.state_barriers:
        report["memory_inventory"]["assumptions"] += (
            "six same-sized BF16 Assign states replace the accounted private PRE Results; no memory credit",)
    if decoder_memory is not None:
        report["decoder_memory_inventory"] = decoder_memory
    report["gpu_memory"] = []
    def sample_memory(stage):
        report["gpu_memory"].append({"stage": stage,
            "statistics": core.get_property(device, "GPU_MEMORY_STATISTICS"),
            "host_memory": process_snapshot()})
    begin = time.perf_counter()
    try:
        if args.disk_history:
            history_store = DiskKv(layers=30, prefix="otc-ov-kv-", storage_dtype="bf16",
                                  max_shapes=history_shapes, active_axes=dict.fromkeys(range(30), 0),
                                  required_bytes=history_disk_bytes)
        if args.overlap_history:
            from concurrent.futures import ThreadPoolExecutor
            history_pool = ThreadPoolExecutor(max_workers=1)
        sample_memory("before_setup")
        report["gpu_memory"][-1].update(total_bytes=total, plugin_allocated_bytes=allocated)
        source_prepare_started = time.perf_counter()
        if args.rgb_input:
            from plus_rgb_source import RGBSourceWindowReader
            from plus_stream import build_stream_plan
            source = input_receipt["source_rgb"]
            with RGBSourceWindowReader(
                    frames_path=source["payload_path"], manifest_path=source["manifest_path"],
                    input_receipt=source, target_hw=(geometry.height, geometry.width)) as reader:
                video = reader.normalized_projection(build_stream_plan(
                    stream_plan["source_frames"], fps=1.0))
        else:
            video = arrays.pop("projection_video_f32")
        report["source_prepare_seconds"] = time.perf_counter() - source_prepare_started
        report["source_input_hw"] = input_receipt["source_rgb"]["shape"][1:3]
        report["source_preparation_included"] = args.rgb_input
        stamp = time.perf_counter()
        projection_conditions, report["projection"] = project_video_conditions(
            core, device, context, geometry, video, input_plan=stream_plan, mode=args.mode)
        report["projection_seconds"] = time.perf_counter() - stamp
        sample_memory("projection_retired_before_denoiser")
        retired = report["gpu_memory"][-1]["statistics"]
        if any(int(retired.get(key, 0)) for key in
               ("cl_mem", "usm_device", "usm_host", "usm_shared", "unknown")):
            raise RuntimeError("Projection GPU storage survived phase retirement")
        # The denoiser needs compact conditions, not pixels. The decoder
        # reloads F32 or reads bounded RGB only after the large histories retire.
        del video
        sample_memory("source_pixels_released_before_denoiser")
        denoiser_setup_started = time.perf_counter()
        block_geometry = BlockGeometry(geometry)
        rows = min(args.rows, block_geometry.tokens)
        model, token_geometry, _, post_model = _build_resident_models(
            geometry, rows, remote_pre=args.remote_pre,
            remote_context=args.remote_context, rope_rank4=args.rope_rank4,
            cross_query_chunk_tokens=args.cross_query_chunk_tokens,
            ffn2_matmul=args.ffn2_matmul, state_barriers=args.state_barriers,
            cross_precise=args.cross_precise, remote_tokens=args.remote_tokens,
            ffn2_state=args.ffn2_state)
        if args.ffn2_state:
            report["ffn2_state"].update(_ffn2_state_metadata(post_model))
        profile = {ov.properties.enable_profiling(): True} if args.profile_operators else {}
        pre = ResidentGraph(core, model, device, context=context, config=profile)
        post_profile = dict(profile)
        if args.cross_precise:
            from cross_precise_20261017 import make_precise_plugin_config, PRECISE_TYPE
            from check_native_post_bridge_b580_20261017 import (
                _validate_bridge_ids, _serialized_post_transport_abi, _policy)
            _validate_bridge_ids(post_model, sparse_type=PRECISE_TYPE)
            config_path = make_precise_plugin_config(args.output / "precise-config")
            post_profile.update(CONFIG_FILE=str(config_path), INFERENCE_PRECISION_HINT=ov.Type.dynamic)
            report["cross_precise_config_sha256"] = sha(config_path)
            report["cross_precise_source_sha256"] = {name: sha(HERE.parents[1] / "evidence" / name)
                for name in ("cross_precise_20261017.py", "post_cross_bridge_20261017.py")}
        post = ResidentGraph(core, post_model, device, context=context, config=post_profile)
        if args.cross_precise:
            report["cross_precise_compile_policy"] = _policy(post.compiled,
                "bridge-precise-ffn2-dynamic" if args.ffn2_matmul else "bridge-precise-dynamic")
            runtime_path = args.output / "post.runtime.xml"
            ov.serialize(post.compiled.get_runtime_model(), str(runtime_path))
            report["cross_precise_transport"] = _serialized_post_transport_abi(
                runtime_path, ids_proven=True, sparse_type=PRECISE_TYPE)
            report["cross_precise_runtime_sha256"] = sha(runtime_path)
        del post_model
        sample_memory("pre_post_requests")
        pre_public = ("q_windows", "k_windows", "v_windows", "q_pooled", "k_pooled",
                      "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
        if args.remote_pre:
            pre_public = ("k_device", "v_device", "q_finite", "kv_finite", "q_pooled", "k_pooled",
                          "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
        report["remote_pre"] = args.remote_pre
        report["remote_context"] = args.remote_context
        report["remote_tokens"] = args.remote_tokens
        report["compact_kv"] = args.compact_kv
        report["retained_kv_roi"] = args.retained_kv_roi
        report["split_key"] = args.split_key
        report["binary_prefix"] = args.binary_prefix
        report["bounded_prefix"] = args.bounded_prefix
        report["remote_pre_kv_readback_bytes"] = 0
        report["remote_pre_kv_readback_copies"] = 0
        report["retained_kv_readback_seconds"] = 0.
        report["public_output_transport"] = (
            "GPU_QKV_handoff; retained_KV_ROI_direct_to_owner; no_slab_KV_host_outputs"
            if args.retained_kv_roi else
            "GPU_QKV_handoff; fresh_KV_read_only_for_nonterminal_history" if args.remote_pre
            else "remote_explicit_readback")
        report["shared_input_buffers"] = True
        cross = ResidentGraph(core, build_cross_kv(), device, context=context)
        edges, _ = load_edge_weights(checkpoint, verify_hash=False, mode=args.mode)
        prefix_model = build_prefix(edges, block_geometry)
        prefix = (ResidentGraph(core, prefix_model, device, context=context)
                  if args.bounded_prefix else Graph(core, prefix_model, device))
        del prefix_model
        suffix = Graph(core, build_suffix(edges, block_geometry), device)
        del edges
        sample_memory("prefix_suffix_cross_requests")
        with safe_open(str(checkpoint), framework="np", device="cpu") as handle:
            text = Graph(core, build_text_embedding(load_text_weights(handle)), device)
            embedded = tensor_bits(storage_tensor(text({"prompt": arrays["prompt_context_f32"]})["context"], "bf16"))
            embedded_tensor = bits_tensor(embedded)
            del text
            sample_memory("text_embedding_retired")
            banks, cross_banks = [], []
            # Explicitly free two weight-bank allocations to fund the named
            # no-alias bridge workspace; do not consume the safety reserve.
            shared_bank_start = (POST_CROSS_BRIDGE_RESIDENT_BANKS - 1
                                 if args.cross_precise else 26)
            report["resident_pre_post_bank_count"] = shared_bank_start + 1 if args.remote_tokens else 30
            report["shared_weight_bank_start"] = shared_bank_start if args.remote_tokens else None
            cross_weights = None
            upload_bytes = 0
            for layer in range(30):
                values = load_layer(handle, layer)
                if args.remote_tokens:
                    from plus_weight_banks import weight_bank
                    bank = weight_bank((pre, post), values,
                                       None if layer <= shared_bank_start else banks[shared_bank_start])
                    cross_weights = weight_bank((cross,), values, cross_weights)
                    cross_feeds = cross_weights
                    upload_bytes += sum(int(value.size * 2) for value in values.values())
                else:
                    bank = {}
                    for key, value in values.items():
                        name = port_name(key)
                        owner = next(graph for graph in (pre, post, cross) if name in graph.inputs)
                        bank[name] = owner.upload(name, value)
                        upload_bytes += int(value.size * 2)
                    cross_feeds = bank
                banks.append(bank)
                feeds = {name: cross_feeds[name] for name in cross.inputs if name.startswith("weight__")}
                pair = {name: post.allocate(post.inputs[name]) for name in ("cross_k", "cross_v")}
                for name, target in pair.items():
                    cross.bind_output(name, target)
                cross.infer({"context": embedded_tensor, **feeds}, read=())
                cross_banks.append(pair)
                print(json.dumps({"stage": "resident_weights", "layer": layer}), flush=True)
        # Drop the reusable cross INPUT bank before allocating x; cached cross
        # OUTPUT pairs stay live. Do not leave the last feeds dict holding it.
        del values, embedded_tensor, cross, pair, target, cross_weights, cross_feeds
        if not args.remote_tokens:
            del owner
        feeds = {}
        sample_memory("block_weight_banks")
        query_batch = next(size for size in (1, 2, 3, 4, 8, 16)
                           if size >= min(16, block_geometry.query_windows))
        attention = ResidentAttention(core, device, context, geometry,
                                      query_windows=query_batch, split_key=args.split_key,
                                      binary_prefix=args.binary_prefix)
        tokens = None
        if args.remote_tokens:
            from plus_tokens import RemoteTokenState
            tokens = RemoteTokenState(post, token_geometry, rows)
        sample_memory("attention_roots")
        if args.remote_pre:
            from plus_remote_history import CompactHistory
        # Histories are constructed lazily, after call 0 consumes its condition.
        # Setup must not overlap all condition owners with full history capacity.
        histories = [None] * 30
        report["host_history_transport"] = (
            "compact_in_place; retained_KV_ROI_direct_to_owner; pooled_only_merge"
            if args.retained_kv_roi else
            "compact_in_place; nonterminal_KV_host_staged; pooled_only_merge" if args.remote_pre
            else "source_order_concatenated_KV")
        denoised_chunks = []
        resident_bytes = sum(owner.get_byte_size() for owner in
                             {id(owner): owner for bank in banks for owner in bank.values()}.values())
        report.update(resident_block_weight_bytes=resident_bytes,
                      block_weight_upload_bytes=upload_bytes, rows=rows,
                      denoiser_setup_seconds=time.perf_counter() - denoiser_setup_started,
                      setup_seconds=time.perf_counter() - begin)
        sample_memory("resident_setup")
        if len(projection_conditions) != stream_plan["call_count"]:
            raise ValueError("projection condition call count changed")
        for call_meta in stream_plan["calls"]:
            call = call_meta["call"]
            if call != len(report["calls"]):
                raise ValueError("input plan call chronology changed")
            # A zip iterator can retain its last yielded condition internally.
            # Index directly so deleting this call's owner actually releases it.
            condition_bits = projection_conditions[call]
            noise_name, rope_cos_name, rope_sin_name = call_array_names(call)
            started = time.perf_counter()
            if args.bounded_prefix:
                # Preserve the existing BF16 patch boundary. Only the small
                # upload slab expands to F32 for source-exact addition/RNE.
                boundary = prefix.infer(
                    {block_geometry.name + "_x": storage_tensor(arrays[noise_name], "bf16")},
                    read=(block_geometry.name + "_tokens_out", block_geometry.name + "_t_out",
                          block_geometry.name + "_t_mod_out"))
                tokens.load(boundary[block_geometry.name + "_tokens_out"], condition_bits)
                x = tokens
                modulation = boundary[block_geometry.name + "_t_mod_out"]
                timestep = tensor_array(bits_tensor(boundary[block_geometry.name + "_t_out"]))
            else:
                condition = tensor_array(bits_tensor(condition_bits))
                boundary = prefix({block_geometry.name + "_x": arrays[noise_name]})
                x = tensor_bits(storage_tensor(boundary[block_geometry.name + "_tokens_out"] + condition, "bf16"))
                if tokens is not None:
                    tokens.load(x)
                    x = tokens
                modulation = tensor_bits(storage_tensor(boundary[block_geometry.name + "_t_mod_out"], "bf16"))
                timestep = boundary[block_geometry.name + "_t_out"]
                del condition
            projection_conditions[call] = None
            del boundary, condition_bits
            report.setdefault("remaining_conditions", []).append(
                sum(value is not None for value in projection_conditions))
            stage_seconds = {"prefix": time.perf_counter() - started, "weights": 0.,
                             "pre": 0., "attention": 0., "post": 0.}
            pre_seconds = {"history_prepare": 0., "bounded_pre": 0., "history_update": 0.,
                           "history_validate": 0., "history_buffers": 0., "history_upload": 0.}
            pre_request_seconds = {"bind": 0., "execute": 0., "read": 0.}
            attention_seconds = {"layout": 0.}
            for layer in range(30):
                # One pending task only. It must finish before any next-layer
                # state changes; the two-prefix memory ledger remains bounded.
                pending_history_io = _wait_history_io(pending_history_io, report)
                layer_start = time.perf_counter()
                retain = mode.retain_groups(layer)
                bank = banks[layer]
                if tokens is not None and layer >= shared_bank_start:
                    # Setup left the shared bank holding layer29, so refill
                    # its first layer too, on every call. All prior requests waited.
                    # Refill the same owners only
                    # after reading one authenticated layer; count this warm work.
                    with safe_open(str(checkpoint), framework="np", device="cpu") as handle:
                        values = load_layer(handle, layer)
                    weight_bank((pre, post), values, bank)
                    report["block_weight_upload_bytes"] += sum(
                        value.get_byte_size() for value in bank.values())
                    del values
                stage_seconds["weights"] += time.perf_counter() - layer_start
                layer_start = time.perf_counter()
                if args.remote_pre:
                    if args.prefetch_history and layer == 0:
                        report["history_prefetch_seconds"] += _prefetch_history(
                            history_store, histories, layer)
                    history = histories[layer]
                    if history is None:
                        history = CompactHistory(geometry.spatial_windows, retain,
                            backing=history_store.allocate_arrays(layer)
                            if history_store is not None else None)
                        histories[layer] = history
                    # copy_to completes the old-prefix upload before update()
                    # is allowed to reuse that owner's host storage.
                    attention.prepare_remote(history.view())
                    for name, seconds in attention.last_prepare_seconds.items():
                        pre_seconds["history_" + name] += seconds
                    if layer == 0:
                        print(json.dumps({"stage": "remote_pre_ready", "call": call}), flush=True)
                pre_seconds["history_prepare"] += time.perf_counter() - layer_start
                # Final-call K/V are consumed from GPU roots, never retained
                # for another call. Keep pooled scores for this call's mask.
                read_kv = not args.retained_kv_roi and not (args.remote_pre and call_meta["last"])
                kv_start = (geometry.spatial_windows
                            if args.compact_kv and read_kv and retain == 1 else 0)
                def pre_infer(slab, start=None, count=None):
                    slab_kv = read_kv and (not kv_start or start + count > kv_start)
                    pre_read = tuple(name for name in pre_public
                                     if slab_kv or name not in ("k_device", "v_device"))
                    feeds = {name: bank[name] for name in pre.inputs if name.startswith("weight__")}
                    if args.remote_pre:
                        for name, target in attention.pre_targets(start, count).items():
                            pre.bind_output(name, target)
                    if args.remote_pre and layer == 0 and start == 0:
                        print(json.dumps({"stage": "remote_pre_first_infer", "call": call}), flush=True)
                    values = pre.infer({**typed(pre, slab), **feeds}, read=pre_read)
                    for name, seconds in pre.last_seconds.items():
                        pre_request_seconds[name] += seconds
                    if not args.remote_pre:
                        return values
                    for name in ("q_finite", "kv_finite"):
                        valid = values.pop(name)
                        # ResidentGraph returns raw storage bits for F16 scalar
                        # outputs. 0x3c00 is the exact finite-check result 1.0.
                        if valid.shape != () or valid.dtype != np.uint16 or valid.item() != 0x3c00:
                            raise FloatingPointError("Remote pre produced non-finite Q/K/V: " + name)
                    attention.complete_pre(start, count)
                    if slab_kv:
                        for source, destination in (("k_device", "k_windows"), ("v_device", "v_windows")):
                            raw = values.pop(source)
                            report["remote_pre_kv_readback_bytes"] += int(raw.nbytes)
                            report["remote_pre_kv_readback_copies"] += 1
                            values[destination] = raw.reshape(rows // 128, 128, 1536)
                    return values
                pre_started = time.perf_counter()
                prepared = infer_pre(pre_infer, {"x": x, "t_mod": modulation,
                    "rope_cos": arrays[rope_cos_name],
                    "rope_sin": arrays[rope_sin_name]}, token_geometry, rows,
                    include_residual=False, remote=args.remote_pre, read_kv=read_kv,
                    kv_start=kv_start)
                pre_seconds["bounded_pre"] += time.perf_counter() - pre_started
                history_started = time.perf_counter()
                current_history = {"pooled": prepared["k_pooled"]}
                if read_kv:
                    current_history.update(k=prepared["k_windows"], v=prepared["v_windows"])
                if args.remote_pre:
                    merged = None  # Attention consumes the GPU roots, not a host K/V merge.
                    def read_retained(k_destination, v_destination, source_start):
                        stamp = time.perf_counter()
                        attention.read_retained_into(k_destination, v_destination, source_start)
                        report["retained_kv_readback_seconds"] += time.perf_counter() - stamp
                        report["remote_pre_kv_readback_bytes"] += (
                            k_destination.nbytes + v_destination.nbytes)
                        report["remote_pre_kv_readback_copies"] += 2
                    merged_pooled = history.update(current_history, terminal=call_meta["last"],
                        kv_start=kv_start, read_kv=read_retained
                        if args.retained_kv_roi and not call_meta["last"] else None)
                    if history.released:
                        histories[layer] = None
                        if history_store is not None:
                            history_store.retire(layer)
                    elif history_store is not None:
                        if history_pool is not None:
                            pending_history_io = history_pool.submit(
                                _flush_prefetch_history, history_store, layer,
                                _history_prefix_bytes(histories, layer + 1))
                        else:
                            history_store.flush(layer)
                    if args.prefetch_history and pending_history_io is None:
                        # The current mapping has been evicted/retired. Read
                        # next while current attention and POST use GPU roots.
                        report["history_prefetch_seconds"] += _prefetch_history(
                            history_store, histories, layer + 1)
                else:
                    merged, histories[layer] = join_history(current_history, histories[layer],
                        spatial_windows=geometry.spatial_windows, retain=retain,
                        write=not call_meta["last"], raw_bf16=True)
                    merged_pooled = merged["pooled"]
                del current_history
                pre_seconds["history_update"] += time.perf_counter() - history_started
                stage_seconds["pre"] += time.perf_counter() - layer_start
                if layer == 0:
                    sample_memory(f"call{call}_first_pre")
                stamp = time.perf_counter()
                layout = build_stream_sparse_layout(prepared["q_pooled"], merged_pooled, temporal_windows=2,
                    spatial_grid=(geometry.token_hw[0] // 8, geometry.token_hw[1] // 8),
                    cache_temporal_groups=retain, local_range=mode.local_range(layer))
                attention_seconds["layout"] += time.perf_counter() - stamp
                if args.remote_pre:
                    self_context = attention.infer_remote(
                        layout.block_mask, retain_context=args.remote_context)
                else:
                    self_context = attention.infer(
                        prepared["q_windows"], merged["k"], merged["v"], layout.block_mask)
                for name, seconds in attention.last_seconds.items():
                    attention_seconds[name] = attention_seconds.get(name, 0.) + seconds
                stage_seconds["attention"] += time.perf_counter() - stamp
                if layer in (0, 1):
                    sample_memory(f"call{call}_attention_layer{layer}_keys{merged_pooled.shape[1]}")
                stamp = time.perf_counter()
                gates = {name: prepared[name] for name in ("gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")}
                def post_infer(slab, start=None, count=None):
                    feeds = {name: bank[name] for name in post.inputs if name.startswith("weight__")}
                    remote = (attention.context_targets(start, count)
                              if args.remote_context else {})
                    return post.infer({**typed(post, {key: value for key, value in slab.items()
                        if key not in ("cross_k", "cross_v")}),
                        **remote, **cross_banks[layer], **feeds},
                        read=() if tokens is not None else ("x_out",))
                post_arrays = {"x": x, **gates, **cross_banks[layer]}
                if args.remote_context:
                    x = infer_post(post_infer, post_arrays, rows,
                                   out=None if tokens is not None else x,
                                   remote_context=True, context_geometry=token_geometry)["x_out"]
                else:
                    x = infer_post(post_infer, {**post_arrays, "self_context": self_context},
                                   rows, out=x)["x_out"]
                del post_arrays
                stage_seconds["post"] += time.perf_counter() - stamp
                if layer == 0:
                    sample_memory(f"call{call}_first_post")
                print(json.dumps({"call": call, "layer": layer}), flush=True)
                # Do not keep the previous layer's full current K/V while the
                # next infer_pre allocates its replacement.
                del prepared, merged, merged_pooled, self_context, layout, gates
            # Layer 29 must also drain before suffix/next-call work.
            pending_history_io = _wait_history_io(pending_history_io, report)
            canonical = tokens.read() if tokens is not None else x
            prediction = suffix({block_geometry.name + "_tokens_in": tensor_array(bits_tensor(canonical)),
                block_geometry.name + "_t_in": timestep})[block_geometry.name + "_out"]
            del canonical
            denoised_chunks.append(tensor_bits(storage_tensor(arrays[noise_name] - prediction, "bf16")))
            report["calls"].append({"call": call, "seconds": time.perf_counter() - started,
                "stages": stage_seconds, "attention_stages": attention_seconds,
                # Request phases are nested inside bounded_pre, not extra work
                # to add to the three sequential PRE phases above.
                "pre_stages": pre_seconds, "pre_request_stages": pre_request_seconds})
            if args.profile_operators:
                # GPU plugin counters average all submissions since compilation.
                report["calls"][-1]["operator_aggregation"] = "cumulative_average_since_compile"
                report["calls"][-1]["cumulative_average_operators"] = {
                    role: [{"name": item.node_name, "type": item.node_type, "execution": item.exec_type,
                            "status": str(item.status), "real_seconds": item.real_time.total_seconds(),
                            "cpu_seconds": item.cpu_time.total_seconds()}
                           for item in graph.request.get_profiling_info()]
                    for role, graph in (("pre", pre), ("post", post))}
            sample_memory(f"call{call}")
        report["terminal_histories_released"] = all(value is None for value in histories)
        if history_pool is not None:
            history_pool.shutdown(wait=True)
            history_pool = None
        if history_store is not None:
            report["history_disk_flush_seconds"] = history_store.seconds
            history_store.close()
        # Release all denoiser GPU allocations before constructing the decoder.
        retirement_started = time.perf_counter()
        del pre, post, prefix, suffix, attention, banks, cross_banks, bank, feeds
        del arrays, projection_conditions, x, prediction, model, tokens
        gc.collect()
        sample_memory("before_decoder")
        report["denoiser_retire_seconds"] = time.perf_counter() - retirement_started
        normalized = None
        if args.decode:
            started = time.perf_counter()
            if args.disk_decoder:
                from check_plus_phased_decoder_b580 import _phase_decode
                if any(int(value) for value in
                       report["gpu_memory"][-1]["statistics"].values()):
                    raise RuntimeError("Denoiser GPU storage survived phase retirement")
                normalized, report["decoded"] = _phase_decode(
                    core, device, context, geometry, None, denoised_chunks, args.output,
                    sweep_phases=True, disk_parents=True, input_root=args.input,
                    input_sha=args.input_sha256, stream_plan=stream_plan,
                    scratch_root=scratch_root, fuse_memories=args.fuse_memories,
                    frame_pair=args.frame_pair, remote_history=args.remote_history,
                    fuse_growth=args.fuse_growth, remote_frames=args.remote_frames,
                    rgb_source_manifest=input_receipt if args.rgb_decoder else None)
                report["source_reload_seconds"] = report["decoded"]["source_reload_seconds"]
            else:
                from plus_decoder import decode_resident_video
                decoder_arrays, _ = load_input(args.input, args.input_sha256,
                                               fields=("projection_video_f32",))
                video = decoder_arrays["projection_video_f32"]
                del decoder_arrays
                report["source_reload_seconds"] = time.perf_counter() - started
                normalized, report["decoded"] = decode_resident_video(
                    core, device, context, geometry, video, denoised_chunks, stream_plan)
                del video
            report["decode_seconds"] = time.perf_counter() - started
            gc.collect()
            sample_memory("after_decoder")
        # One serial CPU-input -> CPU-output span includes transfers and phase
        # retirement. Subtract only disjoint one-off load/compile intervals;
        # never subtract the old setup_seconds, which includes LQ inference.
        elapsed = time.perf_counter() - begin
        excluded_setup = {
            "projection": report["projection"]["setup_seconds"],
            "denoiser": report["denoiser_setup_seconds"],
            "attention_compile": sum(call["attention_stages"].get("compile", 0.)
                                     for call in report["calls"]),
            "decoder": report["decoded"]["setup_seconds"] if args.decode else 0.,
        }
        processing = elapsed - sum(excluded_setup.values())
        if not np.isfinite(processing) or not 0 <= processing <= elapsed:
            raise RuntimeError("Processing clock has overlapping or invalid setup intervals")
        report["processing_clock"] = {
            "scope": ("RGB CPU source, including resize/normalization, to normalized FP32 CPU frames"
                      if args.rgb_input else "normalized CPU input to normalized FP32 CPU frames")
                     if args.decode else "projection and denoising only; no restored frames",
            "elapsed_before_diagnostics_seconds": elapsed,
            "excluded_one_off_setup_seconds": excluded_setup,
            "processing_seconds": processing,
        }
        if args.decode:
            report["matched_processing_seconds"] = processing
        print(json.dumps({"stage": "processing_complete", **report["processing_clock"]}),
              flush=True)
        # Diagnostic compression and hashing are outside all inference clocks.
        for call, bits in enumerate(denoised_chunks):
            path = args.output / f"call{call}.npz"
            np.savez_compressed(path, denoised_bf16=bits,
                                denoised_f32=tensor_array(bits_tensor(bits)))
            report["calls"][call]["archive"] = {
                "path": str(path.resolve()), "sha256": sha(path)}
        if normalized is not None:
            path = args.output / "decoded_final.npz"
            np.savez_compressed(path, normalized_final_f32=normalized)
            report["decoded"]["sha256"] = sha(path)
        report["status"] = "complete_resident_pilot_not_quality_qualified"
    except BaseException as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        if history_pool is not None:
            history_pool.shutdown(wait=True)
            if pending_history_io is not None:
                try:
                    pending_history_io.result()
                except BaseException as error:
                    # Preserve an inference failure while retaining I/O evidence.
                    report["history_io_error"] = repr(error)
        if history_store is not None:
            for history in histories:
                if history is not None:
                    history.release()
            try:
                history_store.close()
            except (BufferError, OSError) as error:
                # Exception tracebacks can still hold a borrowed view. Never
                # invalidate it; preserve the exact scratch path for teardown.
                report["history_cleanup_error"] = repr(error)
                report["history_scratch"] = str(history_store.root)
        report["wall_seconds"] = time.perf_counter() - begin
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "device", "setup_seconds", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
