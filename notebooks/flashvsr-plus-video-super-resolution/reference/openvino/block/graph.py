#!/usr/bin/env python3
"""Native-shaped OpenVINO f16 graph builders for one Wan DiTBlock.

The builders consume the bounded NumPy dictionary returned by
``weights.load_selected``.  They are intentionally static per geometry: this
keeps the public state shapes auditable and avoids an accidental dynamic graph
that allocates a full UHD/30-layer workload.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

try:
    from .contract import (
        CONTEXT_TOKENS,
        DIM,
        HEADS,
        HEAD_DIM,
        WINDOW,
        WINDOW_TOKENS,
        Geometry,
    )
except ImportError:  # direct import from this directory
    from contract import CONTEXT_TOKENS, DIM, HEADS, HEAD_DIM, WINDOW, WINDOW_TOKENS, Geometry


def _ov():
    import openvino as ov
    from openvino import opset13 as ops

    return ov, ops


def _const(ops, value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return ops.constant(array)


def _shape(ops, values):
    return _const(ops, np.asarray(values, dtype=np.int64))


def _weight(ops, weights: Mapping[str, np.ndarray], key: str):
    value = np.ascontiguousarray(weights[key])
    if value.dtype != np.float16 or not np.isfinite(value).all():
        raise ValueError(f"graph weight {key} must be finite f16")
    return _const(ops, value)


def _linear(ops, x, weights, weight_key: str, bias_key: str):
    weight = _weight(ops, weights, weight_key)
    bias = _weight(ops, weights, bias_key)
    product = ops.matmul(x, weight, False, True)
    bias = ops.reshape(bias, _shape(ops, [1, 1, -1]), False)
    return ops.add(product, bias)


def _reshape(ops, value, shape):
    return ops.reshape(value, _shape(ops, shape), False)


def _layer_norm(ops, x, *, weight=None, bias=None, eps: float = 1e-6):
    """LayerNorm with all reduction/scale arithmetic explicitly in f32."""

    f32 = __import__("openvino").Type.f32
    x32 = ops.convert(x, f32)
    axes = _const(ops, np.array([-1], dtype=np.int64))
    mean = ops.reduce_mean(x32, axes, True)
    centered = ops.subtract(x32, mean)
    variance = ops.reduce_mean(ops.multiply(centered, centered), axes, True)
    normalized = ops.divide(centered, ops.sqrt(ops.add(variance, _const(ops, np.float32(eps)))))
    if weight is not None:
        normalized = ops.multiply(normalized, ops.reshape(_const(ops, np.asarray(weight, dtype=np.float32)), _shape(ops, [1, 1, -1]), False))
    if bias is not None:
        normalized = ops.add(normalized, ops.reshape(_const(ops, np.asarray(bias, dtype=np.float32)), _shape(ops, [1, 1, -1]), False))
    return ops.convert(normalized, __import__("openvino").Type.f16)


def _rms_norm(ops, x, weights, scale_key: str, eps: float = 1e-6):
    """RMSNorm matching the pinned source's f32 reduction boundary."""

    ov = __import__("openvino")
    x32 = ops.convert(x, ov.Type.f32)
    axes = _const(ops, np.array([-1], dtype=np.int64))
    mean_square = ops.reduce_mean(ops.multiply(x32, x32), axes, True)
    denominator = ops.sqrt(ops.add(mean_square, _const(ops, np.float32(eps))))
    # The pinned RMSNorm is ``self.norm(x.float()).to(dtype) * weight``:
    # cast back before the learned weight multiply, not after it.
    normalized = ops.convert(ops.divide(x32, denominator), ov.Type.f16)
    scale = ops.reshape(_weight(ops, weights, scale_key), _shape(ops, [1, 1, -1]), False)
    return ops.multiply(normalized, scale)


def _chunk(ops, value, index: int, width: int = 1):
    starts = _const(ops, np.array([0, index, 0], dtype=np.int64))
    ends = _const(ops, np.array([1, index + width, DIM], dtype=np.int64))
    steps = _const(ops, np.ones(3, dtype=np.int64))
    return ops.strided_slice(value, starts, ends, steps, [0, 0, 0], [0, 0, 0])


def _rope(ops, x, cos, sin, tokens: int):
    """Real-F32 pair rotation for the source's 64 complex RoPE pairs."""

    ov = __import__("openvino")
    x32 = ops.convert(x, ov.Type.f32)
    value = _reshape(ops, x32, [1, tokens, HEADS, HEAD_DIM // 2, 2])
    starts_even = _const(ops, np.zeros(5, dtype=np.int64))
    ends_even = _const(ops, np.array([1, tokens, HEADS, HEAD_DIM // 2, 1], dtype=np.int64))
    steps = _const(ops, np.ones(5, dtype=np.int64))
    even = ops.strided_slice(value, starts_even, ends_even, steps, [0] * 5, [0] * 5)
    starts_odd = _const(ops, np.array([0, 0, 0, 0, 1], dtype=np.int64))
    ends_odd = _const(ops, np.array([1, tokens, HEADS, HEAD_DIM // 2, 2], dtype=np.int64))
    odd = ops.strided_slice(value, starts_odd, ends_odd, steps, [0] * 5, [0] * 5)
    pair_shape = _shape(ops, [1, tokens, 1, HEAD_DIM // 2, 1])
    cos = ops.reshape(cos, pair_shape, False)
    sin = ops.reshape(sin, pair_shape, False)
    first = ops.subtract(ops.multiply(even, cos), ops.multiply(odd, sin))
    second = ops.add(ops.multiply(even, sin), ops.multiply(odd, cos))
    rotated = ops.concat((first, second), 4)
    return ops.convert(_reshape(ops, rotated, [1, tokens, DIM]), ov.Type.f16)


def _partition(ops, x, geometry: Geometry):
    f, h, w = geometry.f, geometry.h, geometry.w
    wf, wh, ww = WINDOW
    value = _reshape(ops, x, [1, f, h, w, DIM])
    value = _reshape(ops, value, [1, f // wf, wf, h // wh, wh, w // ww, ww, DIM])
    value = ops.transpose(value, _const(ops, np.array([0, 1, 3, 5, 2, 4, 6, 7], dtype=np.int64)))
    return _reshape(ops, value, [geometry.query_windows, WINDOW_TOKENS, DIM])


def _reverse(ops, windows, geometry: Geometry):
    f, h, w = geometry.f, geometry.h, geometry.w
    wf, wh, ww = WINDOW
    value = _reshape(ops, windows, [1, f // wf, h // wh, w // ww, wf, wh, ww, DIM])
    value = ops.transpose(value, _const(ops, np.array([0, 1, 4, 2, 5, 3, 6, 7], dtype=np.int64)))
    return _reshape(ops, value, [1, geometry.tokens, DIM])


def build_pre_graph(weights: Mapping[str, np.ndarray], geometry: Geometry, *, precision: str = "f16"):
    """Build pre-QKV/modulation/RoPE/window graph for one static geometry."""

    if precision != "f16":
        raise ValueError("block graph only exposes the explicit f16 storage variant")
    geometry.validate()
    ov, ops = _ov()
    f16, f32 = ov.Type.f16, ov.Type.f32
    x = ops.parameter(list(geometry.token_shape), f16, name="x")
    t_mod = ops.parameter([1, 6, DIM], f16, name="t_mod")
    rope_cos = ops.parameter([1, geometry.tokens, 1, HEAD_DIM // 2], f32, name="rope_cos")
    rope_sin = ops.parameter([1, geometry.tokens, 1, HEAD_DIM // 2], f32, name="rope_sin")

    normalized = _layer_norm(ops, x)
    modulation = ops.add(_weight(ops, weights, "blocks.0.modulation"), t_mod)
    shift_msa, scale_msa = _chunk(ops, modulation, 0), _chunk(ops, modulation, 1)
    gate_msa, shift_mlp = _chunk(ops, modulation, 2), _chunk(ops, modulation, 3)
    scale_mlp, gate_mlp = _chunk(ops, modulation, 4), _chunk(ops, modulation, 5)
    input_x = ops.add(ops.multiply(normalized, ops.add(_const(ops, np.float16(1.0)), scale_msa)), shift_msa)
    q = _linear(ops, input_x, weights, "blocks.0.self_attn.q.weight", "blocks.0.self_attn.q.bias")
    k = _linear(ops, input_x, weights, "blocks.0.self_attn.k.weight", "blocks.0.self_attn.k.bias")
    v = _linear(ops, input_x, weights, "blocks.0.self_attn.v.weight", "blocks.0.self_attn.v.bias")
    q = _rope(ops, _rms_norm(ops, q, weights, "blocks.0.self_attn.norm_q.weight"), rope_cos, rope_sin, geometry.tokens)
    k = _rope(ops, _rms_norm(ops, k, weights, "blocks.0.self_attn.norm_k.weight"), rope_cos, rope_sin, geometry.tokens)
    q_windows = _partition(ops, q, geometry)
    k_windows = _partition(ops, k, geometry)
    v_windows = _partition(ops, v, geometry)
    # Keep the public input name ``x`` distinct from the residual output name.
    # A Parameter reused directly as a Result would otherwise make the CPU
    # request expose ``residual`` as its input tensor name.
    residual = ops.convert(x, f16)
    outputs = [residual, q_windows, k_windows, v_windows, gate_msa, shift_mlp, scale_mlp, gate_mlp]
    names = ("residual", "q_windows", "k_windows", "v_windows", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
    for node, name in zip(outputs, names):
        node.output(0).get_tensor().set_names({name})
    model = ov.Model(outputs, [x, t_mod, rope_cos, rope_sin], f"wan_layer0_pre_qkv_{geometry.name}_f16")
    model.validate_nodes_and_infer_types()
    return model


def _layout_probe_module():
    """Load the existing read-only GatherND+SDPA builder, once per process."""

    probe_dir = Path(__file__).resolve().parents[2].parent / "flashvsr-audit/probe"
    if str(probe_dir) not in sys.path:
        sys.path.insert(0, str(probe_dir))
    path = probe_dir / "cpu_layout_diagnostic.py"
    spec = importlib.util.spec_from_file_location("flashvsr_layout_probe_for_block", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load selected-key builder: {path}")
    module = importlib.util.module_from_spec(spec)
    # ``cpu_layout_diagnostic.py`` declares dataclasses; register the module
    # before execution so dataclasses can resolve its ``__module__`` normally.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_attention_graph(geometry: Geometry, key_windows: int, *, precision: str = "f16", query_chunk_windows: int = 1):
    """Wrap the qualified GatherND+SDPA layout-C builder with f16 state IO.

    ``selectedKVblock`` and ``validSlot`` are packed from the pinned source
    block mask outside this graph.  GatherND selects only those key blocks;
    SDPA performs one global softmax over each selected row.  Empty rows are
    safely dummy-gathered and explicitly zeroed by the shared builder.
    """

    if precision != "f16" or query_chunk_windows != 1:
        raise ValueError("the finite block graph only exposes f16, one-window selected-key chunks")
    geometry.validate()
    expected_keys = geometry.attention_key_windows
    if key_windows != expected_keys:
        raise ValueError(f"source streaming geometry {geometry.name} requires {expected_keys} old+new key windows")
    ov, ops = _ov()
    f16, f32 = ov.Type.f16, ov.Type.f32
    helper = _layout_probe_module()
    probe = sys.modules.get("probe")
    if probe is None or not hasattr(probe, "Case"):
        raise ImportError("selected-key builder did not load its Case definition")
    heads, blocks, block_tokens, depth = HEADS, geometry.query_windows, WINDOW_TOKENS, HEAD_DIM
    block_mask = np.ones((heads, blocks, key_windows), dtype=np.bool_)
    selected_ids = np.broadcast_to(np.arange(key_windows, dtype=np.int32), (heads, blocks, key_windows)).copy()
    valid_slots = np.ones_like(selected_ids, dtype=np.bool_)
    case = probe.Case(f"layer0_{geometry.name}", block_mask, selected_ids, valid_slots, block_tokens, depth)
    layout_model, _ = helper.build_layout_c_graph(case)
    parameters = {node.get_friendly_name(): node for node in layout_model.get_parameters()}

    def state_layout(name: str, windows: int):
        state = ops.parameter([windows, block_tokens, DIM], f16, name=name)
        value = _reshape(ops, state, [windows, block_tokens, HEADS, HEAD_DIM])
        value = ops.transpose(value, _const(ops, np.array([2, 0, 1, 3], dtype=np.int64)))
        return state, ops.convert(value, f32)

    q_state, q_layout = state_layout("q_windows", blocks)
    k_state, k_layout = state_layout("k_windows", key_windows)
    v_state, v_layout = state_layout("v_windows", key_windows)
    for name, value in (("Q", q_layout), ("K", k_layout), ("V", v_layout)):
        old = parameters.get(name)
        if old is None:
            raise AssertionError(f"shared selected-key graph has no {name} parameter")
        old.output(0).replace(value.output(0))
    selected = parameters.get("selectedKVblock")
    valid = parameters.get("validSlot")
    if selected is None or valid is None:
        raise AssertionError("shared selected-key graph parameter names changed")
    # ``layout_model.output(0)`` is the helper's Result port.  Follow its
    # source output before embedding the shared graph; retaining that Result
    # as an interior node makes the CPU plugin reject the new model because
    # the old Result is not in its result list.
    helper_result = layout_model.output(0).get_node()
    attended = ops.convert(helper_result.input_value(0), f16)
    attended = ops.transpose(attended, _const(ops, np.array([1, 2, 0, 3], dtype=np.int64)))
    attended = _reshape(ops, attended, [blocks, block_tokens, DIM])
    output = _reverse(ops, attended, geometry)
    output.output(0).get_tensor().set_names({"self_context"})
    model = ov.Model(
        [output], [q_state, k_state, v_state, selected, valid],
        f"wan_layer0_attention_{geometry.name}_gathernd_sdpa_f16",
    )
    model.validate_nodes_and_infer_types()
    return model


_CROSS_QUERY_CHUNK_TOKENS = (128, 256, 512)


def validate_cross_query_chunk_tokens(value: int) -> int:
    """Validate the independent cross-attention query-row batch size."""

    if type(value) is not int or value not in _CROSS_QUERY_CHUNK_TOKENS:
        raise ValueError("cross_query_chunk_tokens must be exactly one of 128, 256, or 512")
    return value


def _cross_attention(
    ops, query, cache_k, cache_v, query_tokens: int, *, cross_query_chunk_tokens: int = 128,
):
    cross_query_chunk_tokens = validate_cross_query_chunk_tokens(cross_query_chunk_tokens)
    ov = __import__("openvino")
    k = _reshape(ops, cache_k, [1, CONTEXT_TOKENS, HEADS, HEAD_DIM])
    k = ops.transpose(k, _const(ops, np.array([0, 2, 1, 3], dtype=np.int64)))
    v = _reshape(ops, cache_v, [1, CONTEXT_TOKENS, HEADS, HEAD_DIM])
    v = ops.transpose(v, _const(ops, np.array([0, 2, 1, 3], dtype=np.int64)))
    k, v = ops.convert(k, ov.Type.f32), ops.convert(v, ov.Type.f32)
    chunks = []
    for query_start in range(0, query_tokens, cross_query_chunk_tokens):
        query_count = min(cross_query_chunk_tokens, query_tokens - query_start)
        q = ops.strided_slice(
            query,
            _const(ops, np.array([0, query_start, 0], dtype=np.int64)),
            _const(ops, np.array([1, query_start + query_count, DIM], dtype=np.int64)),
            _const(ops, np.ones(3, dtype=np.int64)),
            [0, 0, 0], [0, 0, 0],
        )
        q = _reshape(ops, q, [1, query_count, HEADS, HEAD_DIM])
        q = ops.transpose(q, _const(ops, np.array([0, 2, 1, 3], dtype=np.int64)))
        q = ops.convert(q, ov.Type.f32)
        scores = ops.multiply(ops.matmul(q, k, False, True), _const(ops, np.float32(1.0 / np.sqrt(HEAD_DIM))))
        probabilities = ops.softmax(scores, 3)
        context = ops.matmul(probabilities, v, False, False)
        context = ops.transpose(context, _const(ops, np.array([0, 2, 1, 3], dtype=np.int64)))
        chunks.append(ops.convert(_reshape(ops, context, [1, query_count, DIM]), ov.Type.f16))
    return ops.concat(chunks, 1)


def build_post_graph(weights: Mapping[str, np.ndarray], geometry: Geometry, *, precision: str = "f16"):
    """Build post-self-attention gate, fixed cross-attention, and FFN."""

    if precision != "f16":
        raise ValueError("block graph only exposes the explicit f16 storage variant")
    geometry.validate()
    ov, ops = _ov()
    f16 = ov.Type.f16
    x_input = ops.parameter(list(geometry.token_shape), f16, name="x")
    self_context = ops.parameter(list(geometry.token_shape), f16, name="self_context")
    gate_msa = ops.parameter([1, 1, DIM], f16, name="gate_msa")
    shift_mlp = ops.parameter([1, 1, DIM], f16, name="shift_mlp")
    scale_mlp = ops.parameter([1, 1, DIM], f16, name="scale_mlp")
    gate_mlp = ops.parameter([1, 1, DIM], f16, name="gate_mlp")
    cross_k = ops.parameter([1, CONTEXT_TOKENS, DIM], f16, name="cross_k")
    cross_v = ops.parameter([1, CONTEXT_TOKENS, DIM], f16, name="cross_v")

    self_output = _linear(ops, self_context, weights, "blocks.0.self_attn.o.weight", "blocks.0.self_attn.o.bias")
    x = ops.add(x_input, ops.multiply(gate_msa, self_output))
    cross_input = _layer_norm(
        ops,
        x,
        weight=weights["blocks.0.norm3.weight"],
        bias=weights["blocks.0.norm3.bias"],
    )
    cross_q = _linear(ops, cross_input, weights, "blocks.0.cross_attn.q.weight", "blocks.0.cross_attn.q.bias")
    cross_q = _rms_norm(ops, cross_q, weights, "blocks.0.cross_attn.norm_q.weight")
    cross_output = _cross_attention(ops, cross_q, cross_k, cross_v, geometry.tokens)
    cross_output = _linear(ops, cross_output, weights, "blocks.0.cross_attn.o.weight", "blocks.0.cross_attn.o.bias")
    x = ops.add(x, cross_output)
    input_x = ops.add(ops.multiply(_layer_norm(ops, x), ops.add(_const(ops, np.float16(1.0)), scale_mlp)), shift_mlp)
    hidden = _linear(ops, input_x, weights, "blocks.0.ffn.0.weight", "blocks.0.ffn.0.bias")
    hidden = ops.gelu(hidden, "tanh")
    ffn_output = _linear(ops, hidden, weights, "blocks.0.ffn.2.weight", "blocks.0.ffn.2.bias")
    output = ops.add(x, ops.multiply(gate_mlp, ffn_output))
    output.output(0).get_tensor().set_names({"x_out"})
    model = ov.Model(
        [output],
        [x_input, self_context, gate_msa, shift_mlp, scale_mlp, gate_mlp, cross_k, cross_v],
        f"wan_layer0_post_block_{geometry.name}_f16",
    )
    model.validate_nodes_and_infer_types()
    return model


def compile_properties(ov):
    """Typed CPU configuration for later finite review execution."""

    return {
        "INFERENCE_PRECISION_HINT": ov.Type.f16,
        "PERFORMANCE_HINT": ov.properties.hint.PerformanceMode.LATENCY,
        "NUM_STREAMS": ov.properties.streams.Num(1),
        "INFERENCE_NUM_THREADS": 2,
        "CACHE_DIR": "",
    }

