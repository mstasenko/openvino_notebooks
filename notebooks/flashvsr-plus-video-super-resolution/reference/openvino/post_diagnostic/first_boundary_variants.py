#!/usr/bin/env python3
"""Minimal final-only post graphs with first self.o fusion barriers.

All graph arithmetic is reused from ``openvino/block/graph.py``.  The only
variant is the boundary between the F16 self.o linear result and the first
gate/residual expression; no intermediate Results are added to these models.
"""

from __future__ import annotations

import numpy as np


VARIANTS = ("self_o_f32_gate", "self_o_f16_product", "self_o_gate_reshape", "flat_gated_linears", "expanded_gates", "conv_linears")
DIM, HEADS, HEAD_DIM, CONTEXT, WINDOW_TOKENS = 1536, 12, 128, 512, 128


def _flat_gated_linear(ops, graph, residual, gate, activation, weights, prefix, tokens):
    # A 2-D FC makes its feature axis unambiguous to GPU post-op broadcasting.
    width = int(weights[prefix + ".weight"].shape[1])
    activation = graph._reshape(ops, activation, [tokens, width])
    linear = ops.add(ops.matmul(activation, graph._weight(ops, weights, prefix + ".weight"), False, True),
                     graph._weight(ops, weights, prefix + ".bias"))
    value = ops.add(graph._reshape(ops, residual, [tokens, DIM]),
                    ops.multiply(graph._reshape(ops, gate, [1, DIM]), linear))
    return graph._reshape(ops, value, [1, tokens, DIM])


def _conv_linear(ops, graph, activation, weights, prefix, tokens):
    # A 1x1 convolution implements the same dense projection without the
    # failing FC post-op path; the token positions remain independent.
    out_width, in_width = weights[prefix + ".weight"].shape
    activation = ops.transpose(activation, graph._shape(ops, [0, 2, 1]))
    activation = graph._reshape(ops, activation, [1, in_width, 1, tokens])
    weight = graph._reshape(ops, graph._weight(ops, weights, prefix + ".weight"), [out_width, in_width, 1, 1])
    projected = ops.convolution(activation, weight, [1, 1], [0, 0], [0, 0], [1, 1])
    projected = ops.add(projected, graph._reshape(ops, graph._weight(ops, weights, prefix + ".bias"), [1, out_width, 1, 1]))
    projected = graph._reshape(ops, projected, [1, out_width, tokens])
    return ops.transpose(projected, graph._shape(ops, [0, 2, 1]))


def _first_residual(ov, ops, graph, x, gate, self_o, variant):
    f16, f32 = ov.Type.f16, ov.Type.f32
    if variant == "self_o_f32_gate":
        # Keep the linear's public F16 result, then make gate/residual
        # arithmetic explicit F32 and publish the original F16 state boundary.
        product = ops.multiply(ops.convert(gate, f32), ops.convert(self_o, f32))
        return ops.convert(ops.add(ops.convert(x, f32), product), f16)
    if variant == "self_o_f16_product":
        # Preserve the half-product rounding before the residual add.
        product = ops.convert(ops.multiply(ops.convert(gate, f32), ops.convert(self_o, f32)), f16)
        return ops.convert(ops.add(ops.convert(x, f32), ops.convert(product, f32)), f16)
    if variant == "self_o_gate_reshape":
        reshaped_gate = graph._reshape(ops, gate, [1, 1, DIM])
        return ops.add(x, ops.multiply(reshaped_gate, self_o))
    if variant == "expanded_gates":
        expanded = ops.broadcast(gate, graph._shape(ops, list(x.get_output_shape(0))))
        return ops.add(x, ops.multiply(expanded, self_o))
    if variant == "conv_linears":
        return ops.add(x, ops.multiply(gate, self_o))
    raise ValueError(f"unknown first-boundary variant: {variant}")


def build_post_variant(weights, geometry, variant: str):
    """Build one final-only post graph with a narrowly scoped first barrier."""
    import openvino as ov
    from openvino import opset13 as ops
    import graph

    f16, f32 = ov.Type.f16, ov.Type.f32
    x = ops.parameter(list(geometry.token_shape), f16, name="x")
    self_context = ops.parameter(list(geometry.token_shape), f16, name="self_context")
    gate_msa = ops.parameter([1, 1, DIM], f16, name="gate_msa")
    shift_mlp = ops.parameter([1, 1, DIM], f16, name="shift_mlp")
    scale_mlp = ops.parameter([1, 1, DIM], f16, name="scale_mlp")
    gate_mlp = ops.parameter([1, 1, DIM], f16, name="gate_mlp")
    cross_k = ops.parameter([1, CONTEXT, DIM], f16, name="cross_k")
    cross_v = ops.parameter([1, CONTEXT, DIM], f16, name="cross_v")

    if variant == "flat_gated_linears":
        first_residual = _flat_gated_linear(ops, graph, x, gate_msa, self_context, weights, "blocks.0.self_attn.o", geometry.tokens)
    else:
        if variant == "conv_linears":
            self_o = _conv_linear(ops, graph, self_context, weights, "blocks.0.self_attn.o", geometry.tokens)
        else:
            self_o = graph._linear(ops, self_context, weights, "blocks.0.self_attn.o.weight", "blocks.0.self_attn.o.bias")
        first_residual = _first_residual(ov, ops, graph, x, gate_msa, self_o, variant)
    cross_input = graph._layer_norm(ops, first_residual, weight=weights["blocks.0.norm3.weight"], bias=weights["blocks.0.norm3.bias"])
    cross_q = graph._rms_norm(
        ops,
        graph._linear(ops, cross_input, weights, "blocks.0.cross_attn.q.weight", "blocks.0.cross_attn.q.bias"),
        weights,
        "blocks.0.cross_attn.norm_q.weight",
    )

    reshape = lambda value, shape: graph._reshape(ops, value, shape)
    transpose_axes = graph._const(ops, [0, 2, 1, 3], np.int64)
    k = ops.convert(ops.transpose(reshape(cross_k, [1, CONTEXT, HEADS, HEAD_DIM]), transpose_axes), f32)
    v = ops.convert(ops.transpose(reshape(cross_v, [1, CONTEXT, HEADS, HEAD_DIM]), transpose_axes), f32)
    chunks = []
    for start in range(0, geometry.tokens, WINDOW_TOKENS):
        count = min(WINDOW_TOKENS, geometry.tokens - start)
        q = ops.strided_slice(
            cross_q,
            graph._const(ops, [0, start, 0], np.int64),
            graph._const(ops, [1, start + count, DIM], np.int64),
            graph._const(ops, [1, 1, 1], np.int64),
            [0, 0, 0], [0, 0, 0],
        )
        q = ops.convert(ops.transpose(reshape(q, [1, count, HEADS, HEAD_DIM]), transpose_axes), f32)
        scores = ops.multiply(
            ops.matmul(q, k, False, True),
            graph._const(ops, np.float32(1.0 / np.sqrt(HEAD_DIM))),
        )
        probabilities = ops.softmax(scores, 3)
        context = ops.transpose(ops.matmul(probabilities, v, False, False), transpose_axes)
        chunks.append(ops.convert(reshape(context, [1, count, DIM]), f16))
    cross_context = ops.concat(chunks, 1)
    cross_o = graph._linear(ops, cross_context, weights, "blocks.0.cross_attn.o.weight", "blocks.0.cross_attn.o.bias")
    second_residual = ops.add(first_residual, cross_o)
    modulated = ops.add(
        ops.multiply(graph._layer_norm(ops, second_residual), ops.add(graph._const(ops, np.float16(1.0)), scale_mlp)),
        shift_mlp,
    )
    hidden = graph._linear(ops, modulated, weights, "blocks.0.ffn.0.weight", "blocks.0.ffn.0.bias")
    hidden = ops.gelu(hidden, "tanh")
    if variant == "flat_gated_linears":
        output = _flat_gated_linear(ops, graph, second_residual, gate_mlp, hidden, weights, "blocks.0.ffn.2", geometry.tokens)
    else:
        if variant == "conv_linears":
            ffn_output = _conv_linear(ops, graph, hidden, weights, "blocks.0.ffn.2", geometry.tokens)
        else:
            ffn_output = graph._linear(ops, hidden, weights, "blocks.0.ffn.2.weight", "blocks.0.ffn.2.bias")
        if variant == "expanded_gates":
            gate_mlp_value = ops.broadcast(gate_mlp, graph._shape(ops, [1, geometry.tokens, DIM]))
        else:
            gate_mlp_value = gate_mlp
        output = ops.add(second_residual, ops.multiply(gate_mlp_value, ffn_output))
    output.output(0).get_tensor().set_names({"x_out"})
    model = ov.Model(
        [output],
        [x, self_context, gate_msa, shift_mlp, scale_mlp, gate_mlp, cross_k, cross_v],
        f"wan_layer0_post_{geometry.name}_f16_{variant}",
    )
    model.validate_nodes_and_infer_types()
    return model
