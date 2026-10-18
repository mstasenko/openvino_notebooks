#!/usr/bin/env python3
"""Parameterized native pre/post graphs for one 128x128 DiT layer.

The frozen block builders embed constants through ``_weight``.  This module
keeps their reviewed layout/normalization/RoPE helpers, but supplies the
learned values through explicit Parameter nodes.  The selected post builder's
reviewed ``_conv_linear`` is called with a small facade whose only injected
operation is the allow-listed parameter lookup; no exported-graph constant
rewriting or global monkeypatch is used.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from contract import (
    CONTEXT_TOKENS,
    DIM,
    EPS,
    FFN_DIM,
    HEAD_DIM,
    PARAM_SHAPES,
    POST_KEYS,
    PRE_KEYS,
    Geometry,
    parameter_port,
)


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
BLOCK_GRAPH = BASE / "openvino/block/graph.py"
POST_VARIANTS = BASE / "openvino/post_diagnostic/first_boundary_variants.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reviewed module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reviewed_graph():
    """Load block/graph.py with its historical contract import isolated."""

    name = "parameter_layers_reviewed_block_graph"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    contract_path = BLOCK_GRAPH.with_name("contract.py")
    block_contract = _load_module("parameter_layers_block_contract", contract_path)
    prior_contract = sys.modules.get("contract")
    sys.modules["contract"] = block_contract
    try:
        graph = _load_module(name, BLOCK_GRAPH)
    finally:
        if prior_contract is None:
            sys.modules.pop("contract", None)
        else:
            sys.modules["contract"] = prior_contract
    return graph


def _reviewed_conv_builder():
    # first_boundary_variants has no ambient graph import at module load.  Its
    # _conv_linear receives the graph facade explicitly below.
    return _load_module("parameter_layers_selected_conv_builder", POST_VARIANTS)


class _Shape:
    """Shape-only value accepted by the reviewed conv helper."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class _ParameterGraphFacade:
    """Reviewed graph primitives plus one explicit parameter lookup."""

    def __init__(self, graph, params: dict[str, Any]):
        self._graph = graph
        self._params = params

    def _weight(self, _ops, _weights, key: str):
        if key not in self._params:
            raise KeyError(f"unallow-listed learned parameter requested: {key}")
        return self._params[key]

    def _const(self, *args, **kwargs):
        return self._graph._const(*args, **kwargs)

    def _shape(self, *args, **kwargs):
        return self._graph._shape(*args, **kwargs)

    def _reshape(self, *args, **kwargs):
        return self._graph._reshape(*args, **kwargs)


def _ov():
    import openvino as ov
    from openvino import opset13 as ops

    return ov, ops


def _parameters(ops, ov, keys: tuple[str, ...]) -> tuple[dict[str, Any], list[Any]]:
    params: dict[str, Any] = {}
    inputs: list[Any] = []
    for key in keys:
        port = parameter_port(key)
        parameter = ops.parameter(list(PARAM_SHAPES[key]), ov.Type.f16, name=port)
        parameter.output(0).get_tensor().set_names({port})
        params[key] = parameter
        inputs.append(parameter)
    return params, inputs


def _reshape(graph, ops, value, shape, _special_zero=False):
    return graph._reshape(ops, value, shape)


def _linear(graph, ops, x, params, weight_key: str, bias_key: str):
    product = ops.matmul(x, params[weight_key], False, True)
    bias = _reshape(graph, ops, params[bias_key], [1, 1, -1], False)
    return ops.add(product, bias)


def _rms_norm(graph, ops, ov, x, params, scale_key: str):
    """Match graph._rms_norm while taking the learned scale as a Parameter."""

    x32 = ops.convert(x, ov.Type.f32)
    axes = graph._const(ops, np.array([-1], dtype=np.int64))
    mean_square = ops.reduce_mean(ops.multiply(x32, x32), axes, True)
    denominator = ops.sqrt(ops.add(mean_square, graph._const(ops, np.float32(EPS))))
    normalized = ops.convert(ops.divide(x32, denominator), ov.Type.f16)
    scale = _reshape(graph, ops, params[scale_key], [1, 1, -1], False)
    return ops.multiply(normalized, scale)


def _layer_norm(graph, ops, ov, x, params=None, weight_key=None, bias_key=None):
    """F32 reduction and F32 learned affine scales, then F16 state output."""

    x32 = ops.convert(x, ov.Type.f32)
    axes = graph._const(ops, np.array([-1], dtype=np.int64))
    mean = ops.reduce_mean(x32, axes, True)
    centered = ops.subtract(x32, mean)
    variance = ops.reduce_mean(ops.multiply(centered, centered), axes, True)
    normalized = ops.divide(
        centered,
        ops.sqrt(ops.add(variance, graph._const(ops, np.float32(EPS)))),
    )
    if params is not None:
        scale = ops.convert(params[weight_key], ov.Type.f32)
        bias = ops.convert(params[bias_key], ov.Type.f32)
        normalized = ops.multiply(
            normalized,
            _reshape(graph, ops, scale, [1, 1, -1], False),
        )
        normalized = ops.add(
            normalized,
            _reshape(graph, ops, bias, [1, 1, -1], False),
        )
    return ops.convert(normalized, ov.Type.f16)


def _data_inputs(ops, ov, geometry: Geometry, role: str) -> tuple[dict[str, Any], list[Any]]:
    token_shape = list(geometry.token_shape)
    if role == "pre":
        specs = (
            ("x", token_shape, ov.Type.f16),
            ("t_mod", [1, 6, DIM], ov.Type.f16),
            ("rope_cos", [1, geometry.tokens, 1, HEAD_DIM // 2], ov.Type.f32),
            ("rope_sin", [1, geometry.tokens, 1, HEAD_DIM // 2], ov.Type.f32),
        )
    elif role == "post":
        specs = (
            ("x", token_shape, ov.Type.f16),
            ("self_context", token_shape, ov.Type.f16),
            ("gate_msa", [1, 1, DIM], ov.Type.f16),
            ("shift_mlp", [1, 1, DIM], ov.Type.f16),
            ("scale_mlp", [1, 1, DIM], ov.Type.f16),
            ("gate_mlp", [1, 1, DIM], ov.Type.f16),
            ("cross_k", [1, CONTEXT_TOKENS, DIM], ov.Type.f16),
            ("cross_v", [1, CONTEXT_TOKENS, DIM], ov.Type.f16),
        )
    else:
        raise ValueError(f"role must be pre or post, got {role!r}")
    values: dict[str, Any] = {}
    inputs: list[Any] = []
    for name, shape, dtype in specs:
        parameter = ops.parameter(shape, dtype, name=name)
        parameter.output(0).get_tensor().set_names({name})
        values[name] = parameter
        inputs.append(parameter)
    return values, inputs


def build_parameterized_pre(geometry: Geometry):
    """Build one reusable static pre graph with F16 learned-value inputs."""

    geometry.validate()
    ov, ops = _ov()
    graph = _reviewed_graph()
    learned, learned_inputs = _parameters(ops, ov, PRE_KEYS)
    data, data_inputs = _data_inputs(ops, ov, geometry, "pre")

    normalized = graph._layer_norm(ops, data["x"])
    modulation = ops.add(learned["blocks.0.modulation"], data["t_mod"])
    shift_msa = graph._chunk(ops, modulation, 0)
    scale_msa = graph._chunk(ops, modulation, 1)
    gate_msa = graph._chunk(ops, modulation, 2)
    shift_mlp = graph._chunk(ops, modulation, 3)
    scale_mlp = graph._chunk(ops, modulation, 4)
    gate_mlp = graph._chunk(ops, modulation, 5)
    input_x = ops.add(
        ops.multiply(normalized, ops.add(graph._const(ops, np.float16(1.0)), scale_msa)),
        shift_msa,
    )
    q = _linear(graph, ops, input_x, learned, "blocks.0.self_attn.q.weight", "blocks.0.self_attn.q.bias")
    k = _linear(graph, ops, input_x, learned, "blocks.0.self_attn.k.weight", "blocks.0.self_attn.k.bias")
    v = _linear(graph, ops, input_x, learned, "blocks.0.self_attn.v.weight", "blocks.0.self_attn.v.bias")
    q = graph._rope(
        ops,
        _rms_norm(graph, ops, ov, q, learned, "blocks.0.self_attn.norm_q.weight"),
        data["rope_cos"], data["rope_sin"], geometry.tokens,
    )
    k = graph._rope(
        ops,
        _rms_norm(graph, ops, ov, k, learned, "blocks.0.self_attn.norm_k.weight"),
        data["rope_cos"], data["rope_sin"], geometry.tokens,
    )
    q_windows = graph._partition(ops, q, geometry)
    k_windows = graph._partition(ops, k, geometry)
    v_windows = graph._partition(ops, v, geometry)
    outputs = [
        ops.convert(data["x"], ov.Type.f16), q_windows, k_windows, v_windows,
        gate_msa, shift_mlp, scale_mlp, gate_mlp,
    ]
    names = ("residual", "q_windows", "k_windows", "v_windows", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
    for value, name in zip(outputs, names):
        value.output(0).get_tensor().set_names({name})
    model = ov.Model(outputs, data_inputs + learned_inputs, f"wan_parameterized_pre_{geometry.name}_f16")
    model.validate_nodes_and_infer_types()
    return model


def build_parameterized_post(geometry: Geometry, *, cross_query_chunk_tokens: int = 128,
                             ffn2_matmul: bool = False):
    """Build one reusable static post graph with selected conv linears."""

    if type(ffn2_matmul) is not bool:
        raise ValueError("ffn2_matmul must be boolean")
    geometry.validate()
    ov, ops = _ov()
    graph = _reviewed_graph()
    cross_query_chunk_tokens = graph.validate_cross_query_chunk_tokens(cross_query_chunk_tokens)
    conv_builder = _reviewed_conv_builder()
    learned, learned_inputs = _parameters(ops, ov, POST_KEYS)
    data, data_inputs = _data_inputs(ops, ov, geometry, "post")
    facade = _ParameterGraphFacade(graph, learned)
    shape_map = {key: _Shape(PARAM_SHAPES[key]) for key in POST_KEYS}

    self_o = conv_builder._conv_linear(
        ops, facade, data["self_context"], shape_map,
        "blocks.0.self_attn.o", geometry.tokens,
    )
    first_residual = ops.add(data["x"], ops.multiply(data["gate_msa"], self_o))
    cross_input = _layer_norm(
        graph, ops, ov, first_residual, learned,
        "blocks.0.norm3.weight", "blocks.0.norm3.bias",
    )
    cross_q = _rms_norm(
        graph, ops, ov,
        _linear(graph, ops, cross_input, learned, "blocks.0.cross_attn.q.weight", "blocks.0.cross_attn.q.bias"),
        learned, "blocks.0.cross_attn.norm_q.weight",
    )
    cross_context = graph._cross_attention(
        ops, cross_q, data["cross_k"], data["cross_v"], geometry.tokens,
        cross_query_chunk_tokens=cross_query_chunk_tokens,
    )
    cross_o = _linear(
        graph, ops, cross_context, learned,
        "blocks.0.cross_attn.o.weight", "blocks.0.cross_attn.o.bias",
    )
    second_residual = ops.add(first_residual, cross_o)
    modulated = ops.add(
        ops.multiply(
            graph._layer_norm(ops, second_residual),
            ops.add(graph._const(ops, np.float16(1.0)), data["scale_mlp"]),
        ),
        data["shift_mlp"],
    )
    hidden = _linear(
        graph, ops, modulated, learned,
        "blocks.0.ffn.0.weight", "blocks.0.ffn.0.bias",
    )
    hidden = ops.gelu(hidden, "tanh")
    if ffn2_matmul:
        ffn_output = _linear(graph, ops, hidden, learned,
                             "blocks.0.ffn.2.weight", "blocks.0.ffn.2.bias")
    else:
        ffn_output = conv_builder._conv_linear(
            ops, facade, hidden, shape_map, "blocks.0.ffn.2", geometry.tokens,
        )
    gate_product = ops.multiply(data["gate_mlp"], ffn_output)
    output = ops.add(second_residual, gate_product)
    output.output(0).get_tensor().set_names({"x_out"})
    outputs = [output]
    if ffn2_matmul:
        # Observable storage prevents FC post-ops from skipping the source's
        # linear and gate-product rounding boundaries before the residual add.
        for value, name in ((ffn_output, "ffn2_linear_storage"),
                            (gate_product, "ffn2_gate_storage")):
            value.output(0).get_tensor().set_names({name})
            outputs.append(value)
    model_name = f"wan_parameterized_post_{geometry.name}_f16_conv_linears"
    if cross_query_chunk_tokens != 128:
        model_name += f"_crossq{cross_query_chunk_tokens}"
    if ffn2_matmul:
        model_name += "_ffn2_matmul_stores"
    model = ov.Model(outputs, data_inputs + learned_inputs, model_name)
    model.validate_nodes_and_infer_types()
    return model


def build_parameterized(role: str, geometry: Geometry):
    if role == "pre":
        return build_parameterized_pre(geometry)
    if role == "post":
        return build_parameterized_post(geometry)
    raise ValueError(f"role must be pre or post, got {role!r}")
