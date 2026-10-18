"""Bounded native-BF16 state retention for the existing disk decoder sweep."""

import time

import numpy as np


def build_state_finite(shape):
    """Check every BF16 value, including subnormals, with one tiny host flag."""
    import openvino as ov
    from openvino import opset13 as ops

    if (not isinstance(shape, (tuple, list)) or len(shape) != 4
            or tuple(shape[:2]) != (1, 128)
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise ValueError("history finite input must be a static [1,128,H,W] shape")
    value = ops.parameter(list(shape), ov.Type.bf16, name="history_state")
    # IsFinite in the pinned CPU plugin rejects valid subnormals. This
    # comparison instead rejects exactly Inf/NaN without changing state bits.
    finite = ops.less(ops.convert(ops.absolute(value), ov.Type.f32),
                      ops.constant(np.float32(np.inf), ov.Type.f32))
    flag = ops.reduce_min(ops.convert(finite, ov.Type.f16),
                          ops.constant([0, 1, 2, 3], ov.Type.i32), False)
    flag.output(0).get_tensor().set_names({"finite_flag"})
    return ov.Model([flag], [value], "history_state_finite")


def sweep_history(core, device, graph, checker, source_hw, source, destination, count,
                  *, growth_weights=None):
    """Write a chronological history pass; only middle results leave the GPU."""
    from frame_geometry import history_memory_tiles
    from plus_decoder import STATE_NAMES
    from plus_memory import DECODER_STAGED_DEVICE_RESERVE_BYTES
    from plus_resident import bits_tensor
    from plus_storage import is_finite_bf16

    growth = growth_weights is not None
    descriptor = history_memory_tiles(source_hw, **({"tile_count": 8} if growth else {}))
    full_shape, tile_shape = (tuple(descriptor[name]) for name in
                              ("source_middle_shape", "tile_middle_shape"))
    names = STATE_NAMES[6:]
    if (source is destination or type(count) is not int or count <= 0
            or count != destination.count or tuple(destination.shape) != full_shape):
        raise ValueError("remote history stores do not match their geometry/count")
    if growth:
        compact_shape = tuple(descriptor["source_compact_shape"])
        if not isinstance(source, list) or 2 * len(source) != count:
            raise ValueError("growth history requires two children per compact parent")
        if (not isinstance(growth_weights, (tuple, list)) or len(growth_weights) != 2
                or any(not isinstance(value, np.ndarray) or value.dtype != np.uint16
                       or value.shape != (256, 256, 1, 1)
                       or not value.flags.c_contiguous or not is_finite_bf16(value)
                       for value in growth_weights)):
            raise ValueError("growth history requires two finite BF16 child banks")
        state_outputs = ("S0", "S1", "S2")
    else:
        if count != source.count or tuple(source.shape) != full_shape:
            raise ValueError("remote history source does not match its geometry/count")
        state_outputs = tuple(f"{name}_out" for name in names[1:])
    expected = {"middle", *state_outputs,
                "residual_linear_6", "residual_linear_7", "residual_linear_8"}
    ov, context = graph.ov, graph.context
    if context is None or set(graph.outputs) != expected:
        raise ValueError("remote history requires the complete fused GPU graph")
    for name in ("middle", *state_outputs):
        root = graph.buffers[name]
        if not isinstance(root, ov.RemoteTensor) or tuple(root.get_shape()) != tile_shape:
            raise ValueError("history outputs must be remote BF16 tiles")
        # The normal binding checks enforce the compiled type/byte extent
        # and the owned context, including for the two reused output roots.
        graph.bind_output(name, root)
    flag_buffer = checker.buffers["finite_flag"]
    if (isinstance(flag_buffer, ov.RemoteTensor)
            or flag_buffer.element_type != ov.Type.f16
            or tuple(flag_buffer.get_shape()) != ()):
        raise ValueError("history finite flag must be host-owned scalar F16")

    started = time.perf_counter()
    device_budget = (int(core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE"))
                     - DECODER_STAGED_DEVICE_RESERVE_BYTES)
    if device_budget <= 0:
        raise MemoryError("remote history requires the device reserve")
    peak_device_bytes, peak_statistics = 0, {}
    zero_bits = np.zeros(tile_shape, dtype=np.uint16)
    zero = bits_tensor(zero_bits, share=True)

    def new_root():
        value = context.create_tensor(ov.Type.bf16, ov.Shape(tile_shape), {})
        zero.copy_to(value)
        return value

    # Reuse state-output roots as free slots. Every output is fully written
    # and checked before rotation; no transient second set is allocated.
    free = ([graph.buffers[name] for name in state_outputs] if growth else
            [new_root(), *(graph.buffers[name] for name in state_outputs)])
    past = [[new_root() for _ in names] for _ in descriptor["tiles"]]
    ring_roots = 3 * (len(descriptor["tiles"]) + 1)
    if len({id(value) for lane in [*past, free] for value in lane}) != ring_roots:
        raise ValueError("history state roots must be distinct")
    if any(value is graph.buffers["middle"] for lane in [*past, free] for value in lane):
        raise ValueError("history state roots must not alias the middle output")
    del zero, zero_bits
    current = None if growth else np.empty(full_shape, dtype=np.uint16)
    result = np.empty(full_shape, dtype=np.uint16)
    input_shape = tuple(descriptor["tile_compact_shape"]) if growth else tile_shape
    tile_input = np.empty(input_shape, dtype=np.uint16)
    compact_root = context.create_tensor(ov.Type.bf16, ov.Shape(input_shape), {}) if growth else None
    banks = []
    if growth:
        for value in growth_weights:
            root = context.create_tensor(ov.Type.bf16, ov.Shape(value.shape), {})
            bits_tensor(value, share=True).copy_to(root)
            banks.append(root)
    infer_seconds = io_seconds = 0.0
    for index in range(count):
        stamp = time.perf_counter()
        if growth:
            current = source[index // 2]
            if (not isinstance(current, np.ndarray) or current.dtype != np.uint16
                    or current.shape != compact_shape or not current.flags.c_contiguous):
                raise ValueError("growth history compact parent ABI changed")
        else:
            source.read_into(index, current)
        if not is_finite_bf16(current):
            raise FloatingPointError("remote history input contains non-finite BF16")
        io_seconds += time.perf_counter() - stamp
        for tile_index, tile in enumerate(descriptor["tiles"]):
            if len({id(value) for value in (*free, *past[tile_index])}) != 6:
                raise ValueError("history inputs and next-state targets alias")
            stamp = time.perf_counter()
            begin, end = tile["compact_rows" if growth else "input_rows"]
            tile_input[...] = current[..., begin:end, :]
            bits_tensor(tile_input, share=True).copy_to(compact_root if growth else free[0])
            for name, value in zip(state_outputs, free if growth else free[1:]):
                graph.bind_output(name, value)
            feeds = {**({"middle": compact_root, "grow_weight": banks[index % 2]}
                        if growth else {"current": free[0]}),
                     **dict(zip(names, past[tile_index]))}
            middle = graph.infer(feeds, read=("middle",))["middle"]
            if (middle.dtype != np.uint16 or middle.shape != tile_shape
                    or not middle.flags.c_contiguous):
                raise ValueError("remote history middle BF16 ABI changed")
            for value in (free if growth else free[1:]):
                flag_buffer.data.view(np.uint16).reshape(-1).fill(0)
                flag = checker.infer({"history_state": value},
                                     read=("finite_flag",))["finite_flag"]
                if (flag.shape != () or flag.dtype != np.uint16
                        or int(flag.item()) != 0x3c00):
                    raise FloatingPointError("remote history state is not finite BF16")
            # Sample while every ring owner is live, not after the function
            # returns and only the last request's bindings remain. This is an
            # observed live allocation, not a claim about allocator peaks.
            statistics = dict(core.get_property(device, "GPU_MEMORY_STATISTICS"))
            allocated = sum(int(statistics.get(key, 0)) for key in
                            ("cl_mem", "usm_device", "usm_shared", "unknown"))
            if allocated > device_budget:
                raise MemoryError("remote history consumed the required device reserve")
            if allocated > peak_device_bytes:
                peak_device_bytes, peak_statistics = allocated, statistics
            begin, end = tile["output_rows"]
            target_begin, target_end = tile["frame_rows"]
            result[..., target_begin:target_end, :] = middle[..., begin:end, :]
            # All writes and finite checks finish before any slot changes role.
            past[tile_index], free = free, past[tile_index]
            del middle, feeds
            infer_seconds += time.perf_counter() - stamp
        stamp = time.perf_counter()
        if not is_finite_bf16(result):
            raise FloatingPointError("remote history result contains non-finite BF16")
        destination.write(index, result)
        io_seconds += time.perf_counter() - stamp
    return {
        "parents": count, "tiles": len(descriptor["tiles"]),
        "tile_inferences": count * len(descriptor["tiles"]),
        "state_finite_checks": (3 if growth else 2) * count * len(descriptor["tiles"]),
        "state_readbacks": 0, "ring_roots": ring_roots,
        "growth_fused": growth,
        "peak_observed_device_bytes": peak_device_bytes,
        "gpu_statistics_with_ring": peak_statistics,
        "source_hw": list(source_hw), "tile_shape": list(tile_shape),
        **({"tile_geometry": descriptor["tiles"],
            "compact_shape": list(input_shape)} if growth else {}),
        "infer_seconds": infer_seconds, "io_seconds": io_seconds,
        "seconds": time.perf_counter() - started,
    }


def sweep_growth_frames(core, device, graph, checker, frame_graph, source_hw,
                        parents, growth_weights, frame_weights, *, output_frames,
                        visible_hw):
    """Keep each history tile on GPU until both stateless frame children finish.

    Unlike the staged sweep, this consumes compact front parents directly and
    returns normalized frames. No expanded middle or temporal state is copied
    to the host or written to a parent store.
    """
    from frame_geometry import history_memory_tiles
    from plus_decoder import STATE_NAMES, _normalize_bf16_decoder_bits
    from plus_memory import DECODER_STAGED_DEVICE_RESERVE_BYTES
    from plus_resident import bits_tensor
    from plus_storage import is_finite_bf16

    descriptor = history_memory_tiles(source_hw, tile_count=16, halo_rows=12)
    compact_shape = tuple(descriptor["source_compact_shape"])
    tile_shape = tuple(descriptor["tile_middle_shape"])
    tile_compact_shape = tuple(descriptor["tile_compact_shape"])
    frame_shape = (1, 3, *descriptor["tile_source_hw"])
    if (not isinstance(parents, list) or len(parents) not in (5, 12, 13)
            or type(output_frames) is not int
            or output_frames != 4 * len(parents) - 3):
        raise ValueError("remote frames require the complete causal front-parent stream")
    if (not isinstance(visible_hw, (tuple, list)) or len(visible_hw) != 2
            or any(type(value) is not int or value <= 0 for value in visible_hw)
            or any(value > bound for value, bound in zip(visible_hw, source_hw))):
        raise ValueError("visible frame exceeds the source raster")
    for weights, shape in ((growth_weights, (256, 256, 1, 1)),
                           (frame_weights, (128, 128, 1, 1))):
        if (not isinstance(weights, (tuple, list)) or len(weights) != 2
                or any(not isinstance(value, np.ndarray) or value.dtype != np.uint16
                       or value.shape != shape or not value.flags.c_contiguous
                       or not is_finite_bf16(value) for value in weights)):
            raise ValueError("remote frame child banks must be finite BF16 source halves")
    context, ov = graph.context, graph.ov
    if context is None or frame_graph.context is not context or checker.context is not context:
        raise ValueError("remote decoder graphs require one owned context")
    expected = {"S0", "S1", "S2", "middle",
                "residual_linear_6", "residual_linear_7", "residual_linear_8"}
    if set(graph.outputs) != expected or set(frame_graph.outputs) != {"frame"}:
        raise ValueError("remote decoder graph output ABI changed")
    middle = graph.buffers["middle"]
    if not isinstance(middle, ov.RemoteTensor) or tuple(middle.get_shape()) != tile_shape:
        raise ValueError("remote frame middle must stay in GPU BF16 storage")
    for name in ("S0", "S1", "S2"):
        value = graph.buffers[name]
        if not isinstance(value, ov.RemoteTensor) or tuple(value.get_shape()) != tile_shape:
            raise ValueError("remote frame states must stay in GPU BF16 storage")
    frame_output = frame_graph.buffers["frame"]
    if (not isinstance(frame_output, ov.RemoteTensor)
            or tuple(frame_output.get_shape()) != frame_shape):
        raise ValueError("remote frame output must stay in GPU BF16 storage")
    flag_buffer = checker.buffers["finite_flag"]
    if (isinstance(flag_buffer, ov.RemoteTensor) or flag_buffer.element_type != ov.Type.f16
            or tuple(flag_buffer.get_shape()) != ()):
        raise ValueError("remote decoder finite flag must be host scalar F16")

    started = time.perf_counter()
    device_budget = (int(core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE"))
                     - DECODER_STAGED_DEVICE_RESERVE_BYTES)
    if device_budget <= 0:
        raise MemoryError("remote frames require the device reserve")
    zero_bits = np.zeros(tile_shape, dtype=np.uint16)
    zero_host = bits_tensor(zero_bits, share=True)

    def upload(bits):
        root = context.create_tensor(ov.Type.bf16, ov.Shape(bits.shape), {})
        bits_tensor(bits, share=True).copy_to(root)
        return root

    def new_state():
        root = context.create_tensor(ov.Type.bf16, ov.Shape(tile_shape), {})
        zero_host.copy_to(root)
        return root

    free = [graph.buffers[name] for name in ("S0", "S1", "S2")]
    past = [[new_state() for _ in range(3)] for _ in descriptor["tiles"]]
    growth_banks = [upload(value) for value in growth_weights]
    frame_banks = [upload(value) for value in frame_weights]
    tile_input = np.empty(tile_compact_shape, dtype=np.uint16)
    compact_root = context.create_tensor(ov.Type.bf16, ov.Shape(tile_compact_shape), {})
    owners = [middle, frame_graph.buffers["frame"], compact_root,
              *growth_banks, *frame_banks, *free,
              *(root for lane in past for root in lane)]
    if len({id(root) for root in owners}) != len(owners):
        raise ValueError("remote decoder roots must not alias")
    del owners, zero_bits, zero_host
    final = np.empty((1, output_frames, 3, *visible_hw), dtype=np.float32)
    seen = np.zeros((output_frames, visible_hw[0]), dtype=bool)
    timings = {name: 0.0 for name in
               ("compact_upload", "growth", "finite", "frame", "normalize_scatter")}
    compact_uploads = 0
    peak_device_bytes, peak_statistics = 0, {}
    root = None
    try:
        for parent_index, parent in enumerate(parents):
            if (not isinstance(parent, np.ndarray) or parent.dtype != np.uint16
                    or parent.shape != compact_shape or not parent.flags.c_contiguous
                    or not is_finite_bf16(parent)):
                raise ValueError("remote frame compact parent BF16 ABI changed")
            for lane, tile in enumerate(descriptor["tiles"]):
                stamp = time.perf_counter()
                begin, end = tile["compact_rows"]
                tile_input[...] = parent[..., begin:end, :]
                bits_tensor(tile_input, share=True).copy_to(compact_root)
                compact_uploads += 1
                timings["compact_upload"] += time.perf_counter() - stamp
                # Tile histories are independent. Finish both causal children
                # of this lane before reusing its unchanged compact input.
                for child in (0, 1):
                    index = 2 * parent_index + child
                    if len({id(root) for root in (*free, *past[lane])}) != 6:
                        raise ValueError("remote decoder previous/next states alias")
                    for name, root in zip(("S0", "S1", "S2"), free):
                        graph.bind_output(name, root)
                    stamp = time.perf_counter()
                    graph.infer({"middle": compact_root, "grow_weight": growth_banks[child],
                                 **dict(zip(STATE_NAMES[6:], past[lane]))}, read=())
                    timings["growth"] += time.perf_counter() - stamp
                    stamp = time.perf_counter()
                    for root in free:
                        flag_buffer.data.view(np.uint16).fill(0)
                        flag = checker.infer({"history_state": root}, read=("finite_flag",))["finite_flag"]
                        if (flag.shape != () or flag.dtype != np.uint16
                                or int(flag.item()) != 0x3c00):
                            raise FloatingPointError("remote decoder history is not finite")
                    timings["finite"] += time.perf_counter() - stamp
                    for frame_child in (0, 1):
                        stamp = time.perf_counter()
                        raw = frame_graph.infer(
                            {"middle": middle, "weight__frame_grow": frame_banks[frame_child]},
                            read=("frame",))["frame"]
                        timings["frame"] += time.perf_counter() - stamp
                        if (raw.dtype != np.uint16 or raw.shape != frame_shape
                                or not raw.flags.c_contiguous or not is_finite_bf16(raw)):
                            raise ValueError("remote decoder returned invalid BF16 frame tile")
                        visible = 2 * index + frame_child - 3
                        if visible >= 0:
                            stamp = time.perf_counter()
                            start, stop = tile["source_output_rows"]
                            target_start, target_stop = tile["source_frame_rows"]
                            target_stop = min(target_stop, visible_hw[0])
                            if target_start < target_stop:
                                stop = start + target_stop - target_start
                                if np.any(seen[visible, target_start:target_stop]):
                                    raise ValueError("remote decoder wrote a frame row twice")
                                # Normalize only the valid tile core; no full
                                # raw frame or expanded middle needs a host owner.
                                core_bits = np.ascontiguousarray(raw[..., start:stop, :visible_hw[1]])
                                final[:, visible, :, target_start:target_stop, :] = _normalize_bf16_decoder_bits(core_bits)
                                seen[visible, target_start:target_stop] = True
                            timings["normalize_scatter"] += time.perf_counter() - stamp
                    # Every state is checked and both stateless consumers have
                    # finished before previous/next state slots change role.
                    past[lane], free = free, past[lane]
                    statistics = dict(core.get_property(device, "GPU_MEMORY_STATISTICS"))
                    allocated = sum(int(statistics.get(key, 0)) for key in
                                    ("cl_mem", "usm_device", "usm_shared", "unknown"))
                    if allocated > device_budget:
                        raise MemoryError("remote frames consumed the device reserve")
                    if allocated > peak_device_bytes:
                        peak_device_bytes, peak_statistics = allocated, statistics
        if not np.all(seen):
            raise ValueError("remote decoder frame row coverage is incomplete")
        return final, {
            "parents": 2 * len(parents), "tiles": 16, "ring_roots": 51,
            "compact_tile_uploads": compact_uploads,
            "tile_inferences": 2 * len(parents) * 16,
            "frame_tile_inferences": 4 * len(parents) * 16,
            "state_finite_checks": 6 * len(parents) * 16,
            "raw_frame_outputs": 4 * len(parents), "discarded_frame_outputs": 3,
            "output_frames": output_frames, "state_readbacks": 0, "middle_readbacks": 0,
            "parent_store_io_bytes": 0, "tile_geometry": descriptor["tiles"],
            "peak_observed_device_bytes": peak_device_bytes,
            "gpu_statistics_with_ring": peak_statistics,
            "stages": timings, "seconds": time.perf_counter() - started,
        }
    finally:
        # Exceptions retain this frame in their traceback; release loop-local
        # remote owners as well as the ring containers before group retirement.
        root = middle = compact_root = frame_output = value = None
        free.clear()
        past.clear()
        growth_banks.clear()
        frame_banks.clear()
