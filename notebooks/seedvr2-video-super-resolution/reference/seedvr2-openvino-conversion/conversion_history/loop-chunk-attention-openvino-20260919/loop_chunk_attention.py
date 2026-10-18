#!/usr/bin/env python3
"""Wrap the saved shared SeedVR2 window-attention IR in an OpenVINO Loop.

This file intentionally contains no model loading or weight access.  The Loop
body reuses the existing attention-only IR, keeps QKV invariant, slices the
three window metadata arrays, concatenates per-window video values, and merges
text sums/counts across iterations.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import openvino as ov


HERE = Path(__file__).resolve().parent
TRIAL_ROOT = HERE.parents[1]
SHARED_IR = (
    HERE.parent
    / "dynamic-window-length-openvino-20260918"
    / "block0_chunked_attention_fp16.xml"
)
CHUNK_BIN = SHARED_IR.with_suffix(".bin")
WINDOW_META_NAMES = ("window_idx", "window_key_mask", "window_rope_freqs")
QKV_NAMES = ("vid_q", "vid_k", "vid_v", "txt_q", "txt_k", "txt_v")
HEADS = 20
HEAD_DIM = 128
INNER_DIM = HEADS * HEAD_DIM
TXT_TOKENS = 58
ROPE_DIM = 126
SAVED_IR_MAX_ABS_THRESHOLD = 5.0e-3
SAVED_IR_RMS_REL_THRESHOLD = 1.0e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            part = handle.read(1024 * 1024)
            if not part:
                break
            digest.update(part)
    return digest.hexdigest()


def _names(output: Any) -> set[str]:
    return set(output.get_names())


def _ordered_counts(model: Any) -> dict[str, int]:
    return dict(sorted(Counter(node.get_type_name() for node in model.get_ordered_ops()).items()))


def model_io(model: Any) -> dict[str, list[dict[str, Any]]]:
    return {
        "inputs": [
            {
                "name": port.get_any_name(),
                "shape": str(port.get_partial_shape()),
                "element_type": str(port.get_element_type()),
            }
            for port in model.inputs
        ],
        "outputs": [
            {
                "name": port.get_any_name(),
                "shape": str(port.get_partial_shape()),
                "element_type": str(port.get_element_type()),
            }
            for port in model.outputs
        ],
    }


def xml_loop_structure(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    layers = [node for node in root.iter("layer")]
    loops = [node for node in layers if node.get("type") == "Loop"]
    sdpa = [node for node in layers if node.get("type") == "ScaledDotProductAttention"]
    return {
        "all_layer_count": len(layers),
        "loop_count": len(loops),
        "sdpa_count": len(sdpa),
        "sdpa_per_loop_body": [
            sum(1 for node in loop.iter("layer") if node.get("type") == "ScaledDotProductAttention")
            for loop in loops
        ],
    }


def _set_tensor_name(node: Any, name: str) -> None:
    node.set_friendly_name(name)
    node.output(0).get_tensor().set_names({name})


def _parameter(shape: Sequence[int], element_type: Any, name: str) -> Any:
    node = ov.opset16.parameter(list(shape), element_type, name=name)
    node.output(0).get_tensor().set_names({name})
    return node


def _parameter_name(parameter: Any) -> str:
    names = _names(parameter.output(0))
    if len(names) == 1:
        return next(iter(names))
    if not names:
        return parameter.get_friendly_name()
    raise AssertionError(f"ambiguous Parameter tensor names: {sorted(names)}")


def build_loop_model(
    core: Any,
    *,
    source_tokens: int,
    padded_windows: int,
    max_video_len: int,
    chunk_size: int,
    reverse_idx: np.ndarray,
    qkv_type: Any = ov.Type.f32,
    video_output_type: Any = ov.Type.f32,
    include_text_stats: bool = False,
    name: str = "seedvr2_chunk_attention_loop",
) -> dict[str, Any]:
    """Build a static-shape Loop graph whose body contains the saved SDPA once."""

    if not SHARED_IR.is_file() or not CHUNK_BIN.is_file():
        raise FileNotFoundError(f"saved shared attention IR not found: {SHARED_IR}")
    if source_tokens <= 0 or padded_windows <= 0 or max_video_len <= 0 or chunk_size <= 0:
        raise ValueError("all dimensions and chunk_size must be positive")
    if padded_windows % chunk_size:
        raise ValueError("padded window count must be divisible by chunk_size")
    reverse_idx = np.asarray(reverse_idx, dtype=np.int64)
    if reverse_idx.shape != (source_tokens,):
        raise ValueError(f"reverse_idx shape must be ({source_tokens},), got {reverse_idx.shape}")
    if reverse_idx.size and (reverse_idx.min() < 0 or reverse_idx.max() >= padded_windows * max_video_len):
        raise ValueError("reverse_idx selects outside the padded window tensor")
    if np.unique(reverse_idx).size != source_tokens:
        raise ValueError("reverse_idx must contain one distinct position per video token")

    chunk = core.read_model(SHARED_IR)
    reshape_by_name: dict[Any, Any] = {}
    expected_shapes = {
        "vid_q": [source_tokens, HEADS, HEAD_DIM],
        "vid_k": [source_tokens, HEADS, HEAD_DIM],
        "vid_v": [source_tokens, HEADS, HEAD_DIM],
        "txt_q": [TXT_TOKENS, HEADS, HEAD_DIM],
        "txt_k": [TXT_TOKENS, HEADS, HEAD_DIM],
        "txt_v": [TXT_TOKENS, HEADS, HEAD_DIM],
        "window_idx": [chunk_size, max_video_len],
        "window_key_mask": [chunk_size, max_video_len + TXT_TOKENS],
        "window_rope_freqs": [chunk_size, max_video_len + TXT_TOKENS, ROPE_DIM],
    }
    for port in chunk.inputs:
        input_name = port.get_any_name()
        if input_name not in expected_shapes:
            raise AssertionError(f"unexpected saved attention input {input_name!r}")
        reshape_by_name[port] = ov.PartialShape(expected_shapes[input_name])
    if set(port.get_any_name() for port in chunk.inputs) != set(expected_shapes):
        raise AssertionError("saved attention inputs differ from the expected nine-port contract")
    chunk.reshape(reshape_by_name)

    chunk_parameters = {_parameter_name(node): node for node in chunk.get_parameters()}
    if set(chunk_parameters) != set(expected_shapes):
        raise AssertionError(f"reshaped body Parameters differ: {sorted(chunk_parameters)}")
    chunk_values = [result.input_value(0) for result in chunk.get_results()]
    if len(chunk_values) != 3:
        raise AssertionError("saved attention IR must expose video, text sum, and text count")

    text_sum_state = _parameter([TXT_TOKENS, INNER_DIM], ov.Type.f32, "text_sum_state")
    text_count_state = _parameter([1], ov.Type.f32, "text_count_state")
    body_condition = ov.opset16.constant(np.asarray(True, dtype=np.bool_))
    packed_video = ov.opset16.reshape(
        chunk_values[0],
        ops.constant(np.asarray([1, chunk_size * max_video_len, INNER_DIM], dtype=np.int64)),
        False,
        name="pack_loop_video_scan_slice",
    )
    text_sum_next = ov.opset16.add(chunk_values[1], text_sum_state, name="text_sum_accumulate")
    text_count_next = ov.opset16.add(chunk_values[2], text_count_state, name="text_count_accumulate")
    body_parameters = list(chunk.get_parameters()) + [text_sum_state, text_count_state]
    body = ov.Model(
        [
            body_condition.output(0),
            packed_video.output(0),
            text_sum_next.output(0),
            text_count_next.output(0),
        ],
        body_parameters,
        "single_shared_attention_loop_body",
    )

    outer_inputs: list[Any] = []
    outer_parameters: dict[str, Any] = {}
    for input_name in QKV_NAMES:
        shape = (
            [source_tokens, HEADS, HEAD_DIM]
            if input_name.startswith("vid_")
            else [TXT_TOKENS, HEADS, HEAD_DIM]
        )
        parameter = _parameter(shape, qkv_type, input_name)
        outer_inputs.append(parameter)
        outer_parameters[input_name] = parameter
    metadata_shapes = {
        "window_idx": [padded_windows, max_video_len],
        "window_key_mask": [padded_windows, max_video_len + TXT_TOKENS],
        "window_rope_freqs": [padded_windows, max_video_len + TXT_TOKENS, ROPE_DIM],
    }
    metadata_types = {
        "window_idx": ov.Type.i64,
        "window_key_mask": ov.Type.boolean,
        "window_rope_freqs": ov.Type.f32,
    }
    for input_name in WINDOW_META_NAMES:
        parameter = _parameter(metadata_shapes[input_name], metadata_types[input_name], input_name)
        outer_inputs.append(parameter)
        outer_parameters[input_name] = parameter

    ops = ov.opset16
    trip_count_value = padded_windows // chunk_size
    trip_count = ops.constant(np.asarray(trip_count_value, dtype=np.int64))
    execution_condition = ops.constant(np.asarray(True, dtype=np.bool_))
    loop = ov.opset16.loop(trip_count, execution_condition)
    loop.set_friendly_name("chunk_attention_loop")
    loop.set_function(body)
    loop.set_special_body_ports([-1, 0])

    for input_name in QKV_NAMES:
        value = outer_parameters[input_name].output(0)
        if qkv_type != ov.Type.f32:
            value = ops.convert(value, ov.Type.f32, name=f"{input_name}_to_f32").output(0)
        loop.set_invariant_input(chunk_parameters[input_name], value)
    for input_name in WINDOW_META_NAMES:
        loop.set_sliced_input(
            chunk_parameters[input_name],
            outer_parameters[input_name].output(0),
            0,
            chunk_size,
            chunk_size,
            -1,
            0,
        )

    initial_sum = ops.constant(np.zeros((TXT_TOKENS, INNER_DIM), dtype=np.float32))
    initial_count = ops.constant(np.zeros((1,), dtype=np.float32))
    loop.set_merged_input(
        text_sum_state, initial_sum.output(0), text_sum_next.output(0)
    )
    loop.set_merged_input(
        text_count_state, initial_count.output(0), text_count_next.output(0)
    )

    packed_video_scan = loop.get_concatenated_slices(
        packed_video.output(0), 0, 1, 1, -1, 0
    )
    video_windows = ops.reshape(
        packed_video_scan,
        ops.constant(np.asarray([padded_windows, max_video_len, INNER_DIM], dtype=np.int64)),
        False,
        name="reshape_concatenated_loop_video_windows",
    )
    text_sum = loop.get_iter_value(text_sum_next.output(0), -1)
    text_count = loop.get_iter_value(text_count_next.output(0), -1)
    flattened_video = ops.reshape(
        video_windows,
        ops.constant(np.asarray([padded_windows * max_video_len, INNER_DIM], dtype=np.int64)),
        False,
        name="flatten_concatenated_video_windows",
    )
    reverse_constant = ops.constant(reverse_idx)
    gathered_video = ops.gather(
        flattened_video,
        reverse_constant,
        ops.constant(np.asarray(0, dtype=np.int64)),
        0,
        name="exact_source_order_video_gather",
    )
    video_output = ops.convert(
        gathered_video, video_output_type, name="gathered_video_output"
    )
    text_output = ops.divide(text_sum, text_count, name="averaged_text_output")
    outputs = [video_output.output(0), text_output.output(0)]
    output_names = ["vid_out", "txt_out"]
    if include_text_stats:
        outputs.extend([text_sum, text_count])
        output_names.extend(["txt_sum", "txt_count"])
    model = ov.Model(outputs, outer_inputs, name)
    for port, output_name in zip(model.outputs, output_names):
        port.get_tensor().set_names({output_name})

    loop_nodes = [node for node in model.get_ordered_ops() if node.get_type_name() == "Loop"]
    if len(loop_nodes) != 1:
        raise AssertionError(f"outer graph must contain one Loop, found {len(loop_nodes)}")
    loop_node = loop_nodes[0]
    attached_body = loop_node.get_function()
    body_counts = _ordered_counts(attached_body)
    if body_counts.get("ScaledDotProductAttention", 0) != 1:
        raise AssertionError(f"Loop body must contain one saved SDPA: {body_counts}")
    return {
        "model": model,
        "shared_chunk": chunk,
        "loop": loop_node,
        "body": attached_body,
        "trip_count": trip_count_value,
        "padded_windows": padded_windows,
        "chunk_size": chunk_size,
        "source_tokens": source_tokens,
        "max_video_len": max_video_len,
        "include_text_stats": include_text_stats,
        "outer_ordered_op_counts": _ordered_counts(model),
        "body_ordered_op_counts": body_counts,
    }


def _load_static_attention(
    core: Any, *, source_tokens: int, window_count: int, max_video_len: int
) -> tuple[Any, dict[str, Any]]:
    chunk = core.read_model(SHARED_IR)
    expected_shapes = {
        "vid_q": [source_tokens, HEADS, HEAD_DIM],
        "vid_k": [source_tokens, HEADS, HEAD_DIM],
        "vid_v": [source_tokens, HEADS, HEAD_DIM],
        "txt_q": [TXT_TOKENS, HEADS, HEAD_DIM],
        "txt_k": [TXT_TOKENS, HEADS, HEAD_DIM],
        "txt_v": [TXT_TOKENS, HEADS, HEAD_DIM],
        "window_idx": [window_count, max_video_len],
        "window_key_mask": [window_count, max_video_len + TXT_TOKENS],
        "window_rope_freqs": [window_count, max_video_len + TXT_TOKENS, ROPE_DIM],
    }
    reshape_by_name = {}
    for port in chunk.inputs:
        input_name = port.get_any_name()
        if input_name not in expected_shapes:
            raise AssertionError(f"unexpected saved attention input {input_name!r}")
        reshape_by_name[port] = ov.PartialShape(expected_shapes[input_name])
    if set(port.get_any_name() for port in chunk.inputs) != set(expected_shapes):
        raise AssertionError("saved attention inputs differ from the expected nine-port contract")
    chunk.reshape(reshape_by_name)
    parameters = {_parameter_name(node): node for node in chunk.get_parameters()}
    if set(parameters) != set(expected_shapes):
        raise AssertionError(f"reshaped attention Parameters differ: {sorted(parameters)}")
    return chunk, parameters


def build_loop_tail_model(
    core: Any,
    *,
    source_tokens: int,
    total_windows: int,
    max_video_len: int,
    chunk_size: int,
    reverse_idx: np.ndarray,
    qkv_type: Any = ov.Type.f16,
    video_output_type: Any = ov.Type.f16,
    include_text_stats: bool = False,
    name: str = "seedvr2_chunk_attention_loop_with_tail",
) -> dict[str, Any]:
    """Loop over full chunks, run one exact-shape tail body, then gather/average."""

    if not SHARED_IR.is_file() or not CHUNK_BIN.is_file():
        raise FileNotFoundError(f"saved shared attention IR not found: {SHARED_IR}")
    if source_tokens <= 0 or total_windows <= chunk_size or max_video_len <= 0 or chunk_size <= 0:
        raise ValueError("need positive dimensions and at least one full chunk plus a tail")
    loop_windows = (total_windows // chunk_size) * chunk_size
    tail_windows = total_windows - loop_windows
    loop_iterations = loop_windows // chunk_size
    if loop_iterations <= 0 or tail_windows <= 0:
        raise ValueError("exact Loop+tail topology needs at least one full Loop iteration and a non-empty tail")
    reverse_idx = np.asarray(reverse_idx, dtype=np.int64)
    if reverse_idx.shape != (source_tokens,):
        raise ValueError(f"reverse_idx shape must be ({source_tokens},), got {reverse_idx.shape}")
    if reverse_idx.size and (reverse_idx.min() < 0 or reverse_idx.max() >= total_windows * max_video_len):
        raise ValueError("reverse_idx selects outside the exact source-window tensor")
    if np.unique(reverse_idx).size != source_tokens:
        raise ValueError("reverse_idx must contain one distinct position per video token")

    body_chunk, body_parameters = _load_static_attention(
        core, source_tokens=source_tokens, window_count=chunk_size, max_video_len=max_video_len
    )
    body_values = [result.input_value(0) for result in body_chunk.get_results()]
    if len(body_values) != 3:
        raise AssertionError("saved attention IR must expose video, text sum, and text count")
    sum_state = _parameter([TXT_TOKENS, INNER_DIM], ov.Type.f32, "text_sum_state")
    count_state = _parameter([1], ov.Type.f32, "text_count_state")
    body_condition = ov.opset16.constant(np.asarray(True, dtype=np.bool_))
    packed_video = ov.opset16.reshape(
        body_values[0],
        ov.opset16.constant(
            np.asarray([1, chunk_size * max_video_len, INNER_DIM], dtype=np.int64)
        ),
        False,
        name="pack_full_loop_video_scan_slice",
    )
    sum_next = ov.opset16.add(body_values[1], sum_state, name="text_sum_accumulate")
    count_next = ov.opset16.add(body_values[2], count_state, name="text_count_accumulate")
    body = ov.Model(
        [body_condition.output(0), packed_video.output(0), sum_next.output(0), count_next.output(0)],
        list(body_chunk.get_parameters()) + [sum_state, count_state],
        "single_shared_attention_loop_body",
    )

    tail_chunk, tail_parameters = _load_static_attention(
        core, source_tokens=source_tokens, window_count=tail_windows, max_video_len=max_video_len
    )
    tail_values = [result.input_value(0) for result in tail_chunk.get_results()]
    if len(tail_values) != 3:
        raise AssertionError("tail attention IR must expose video, text sum, and text count")

    qkv_parameters: dict[str, Any] = {}
    for input_name in QKV_NAMES:
        shape = (
            [source_tokens, HEADS, HEAD_DIM]
            if input_name.startswith("vid_")
            else [TXT_TOKENS, HEADS, HEAD_DIM]
        )
        external = _parameter(shape, qkv_type, input_name)
        qkv_parameters[input_name] = external
        if qkv_type == ov.Type.f32:
            replacement = external.output(0)
        else:
            replacement = ov.opset16.convert(
                external.output(0), ov.Type.f32, name=f"{input_name}_to_f32"
            ).output(0)
        original_parameter = tail_parameters[input_name]
        targets = list(original_parameter.output(0).get_target_inputs())
        if not targets:
            raise AssertionError(f"tail QKV Parameter {input_name} has no consumers")
        for target in targets:
            target.replace_source_output(replacement)

    metadata_inputs: dict[str, Any] = {}
    metadata_shapes = {
        "window_idx": [total_windows, max_video_len],
        "window_key_mask": [total_windows, max_video_len + TXT_TOKENS],
        "window_rope_freqs": [total_windows, max_video_len + TXT_TOKENS, ROPE_DIM],
    }
    loop_meta_types = {
        "window_idx": ov.Type.i64,
        "window_key_mask": ov.Type.boolean,
        "window_rope_freqs": ov.Type.f32,
    }
    loop_meta_inputs: dict[str, Any] = {}
    tail_meta_values: dict[str, Any] = {}
    slice_start = ov.opset16.constant(np.asarray([0], dtype=np.int64))
    slice_loop_stop = ov.opset16.constant(np.asarray([loop_windows], dtype=np.int64))
    slice_tail_start = ov.opset16.constant(np.asarray([loop_windows], dtype=np.int64))
    slice_total_stop = ov.opset16.constant(np.asarray([total_windows], dtype=np.int64))
    slice_step = ov.opset16.constant(np.asarray([1], dtype=np.int64))
    slice_axis = ov.opset16.constant(np.asarray([0], dtype=np.int64))
    for input_name in WINDOW_META_NAMES:
        full_parameter = _parameter(
            metadata_shapes[input_name], loop_meta_types[input_name], input_name
        )
        metadata_inputs[input_name] = full_parameter
        loop_meta_inputs[input_name] = ov.opset16.slice(
            full_parameter.output(0),
            slice_start,
            slice_loop_stop,
            slice_step,
            slice_axis,
            name=f"slice_full_loop_{input_name}",
        )
        tail_meta_values[input_name] = ov.opset16.slice(
            full_parameter.output(0),
            slice_tail_start,
            slice_total_stop,
            slice_step,
            slice_axis,
            name=f"slice_exact_tail_{input_name}",
        )
        original_parameter = tail_parameters[input_name]
        targets = list(original_parameter.output(0).get_target_inputs())
        if not targets:
            raise AssertionError(f"tail metadata Parameter {input_name} has no consumers")
        for target in targets:
            target.replace_source_output(tail_meta_values[input_name].output(0))

    trip_count = ov.opset16.constant(np.asarray(loop_iterations, dtype=np.int64))
    initial_condition = ov.opset16.constant(np.asarray(True, dtype=np.bool_))
    loop = ov.opset16.loop(trip_count, initial_condition)
    loop.set_friendly_name("full_chunk_attention_loop")
    loop.set_function(body)
    loop.set_special_body_ports([-1, 0])
    for input_name in QKV_NAMES:
        external_value = qkv_parameters[input_name].output(0)
        if qkv_type != ov.Type.f32:
            external_value = ov.opset16.convert(
                external_value, ov.Type.f32, name=f"{input_name}_loop_to_f32"
            ).output(0)
        loop.set_invariant_input(body_parameters[input_name], external_value)
    for input_name in WINDOW_META_NAMES:
        loop.set_sliced_input(
            body_parameters[input_name],
            loop_meta_inputs[input_name].output(0),
            0,
            chunk_size,
            chunk_size,
            -1,
            0,
        )
    sum_initial = ov.opset16.constant(np.zeros((TXT_TOKENS, INNER_DIM), dtype=np.float32))
    count_initial = ov.opset16.constant(np.zeros((1,), dtype=np.float32))
    loop.set_merged_input(sum_state, sum_initial.output(0), sum_next.output(0))
    loop.set_merged_input(count_state, count_initial.output(0), count_next.output(0))
    packed_video_scan = loop.get_concatenated_slices(
        packed_video.output(0), 0, 1, 1, -1, 0
    )
    video_full = ov.opset16.reshape(
        packed_video_scan,
        ov.opset16.constant(
            np.asarray([loop_windows, max_video_len, INNER_DIM], dtype=np.int64)
        ),
        False,
        name="reshape_full_loop_video_windows",
    )
    text_sum_full = loop.get_iter_value(sum_next.output(0), -1)
    text_count_full = loop.get_iter_value(count_next.output(0), -1)

    tail_video = tail_values[0]
    tail_text_sum = tail_values[1]
    tail_text_count = tail_values[2]
    video_windows = ov.opset16.concat([video_full, tail_video], 0, name="loop_plus_tail_video_windows")
    text_sum = ov.opset16.add(text_sum_full, tail_text_sum, name="add_tail_text_sum")
    text_count = ov.opset16.add(text_count_full, tail_text_count, name="add_tail_text_count")
    flattened_video = ov.opset16.reshape(
        video_windows,
        ov.opset16.constant(np.asarray([total_windows * max_video_len, INNER_DIM], dtype=np.int64)),
        False,
        name="flatten_loop_plus_tail_video_windows",
    )
    gathered_video = ov.opset16.gather(
        flattened_video,
        ov.opset16.constant(reverse_idx),
        ov.opset16.constant(np.asarray(0, dtype=np.int64)),
        0,
        name="exact_source_order_video_gather",
    )
    video_output = ov.opset16.convert(
        gathered_video, video_output_type, name="gathered_video_output"
    )
    text_output = ov.opset16.divide(text_sum, text_count, name="averaged_text_output")
    outputs = [video_output.output(0), text_output.output(0)]
    if include_text_stats:
        outputs.extend([text_sum.output(0), text_count.output(0)])

    outer_parameters = list(qkv_parameters.values()) + list(metadata_inputs.values())
    model = ov.Model(outputs, outer_parameters, name)
    output_names = ["vid_out", "txt_out"] + (["txt_sum", "txt_count"] if include_text_stats else [])
    for port, output_name in zip(model.outputs, output_names):
        port.get_tensor().set_names({output_name})
    loop_nodes = [node for node in model.get_ordered_ops() if node.get_type_name() == "Loop"]
    if len(loop_nodes) != 1:
        raise AssertionError(f"outer graph must contain one Loop, found {len(loop_nodes)}")
    loop_node = loop_nodes[0]
    attached_body = loop_node.get_function()
    body_counts = _ordered_counts(attached_body)
    if body_counts.get("ScaledDotProductAttention", 0) != 1:
        raise AssertionError(f"Loop body must contain one saved SDPA: {body_counts}")
    return {
        "model": model,
        "loop_chunk": body_chunk,
        "tail_chunk": tail_chunk,
        "loop": loop_node,
        "body": attached_body,
        "total_windows": total_windows,
        "loop_windows": loop_windows,
        "tail_windows": tail_windows,
        "trip_count": loop_iterations,
        "chunk_size": chunk_size,
        "source_tokens": source_tokens,
        "max_video_len": max_video_len,
        "include_text_stats": include_text_stats,
        "outer_ordered_op_counts": _ordered_counts(model),
        "body_ordered_op_counts": body_counts,
        "tail_ordered_op_counts": _ordered_counts(tail_chunk),
    }


def validate_saved_loop_ir(path: Path) -> dict[str, Any]:
    structure = xml_loop_structure(path)
    if structure["loop_count"] != 1:
        raise AssertionError(f"saved IR must contain one Loop: {structure}")
    if structure["sdpa_per_loop_body"] != [1]:
        raise AssertionError(f"saved Loop IR must contain one body SDPA: {structure}")
    return structure
