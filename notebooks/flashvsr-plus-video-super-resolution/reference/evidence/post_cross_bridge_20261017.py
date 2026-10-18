"""CPU-only fixed-shape POST cross-attention bridge prototype."""

from __future__ import annotations

import openvino as ov
from openvino import opset13 as ops

import bf16_cross_bridge_20261017 as bridge
from plus_graphs import build_post_rows


ROWS = 4096
QUERY_ROWS = 2048
CHANNELS = 1536
KV_ROWS = 512
BRIDGE_BYTES = 94_374_912
BRIDGE_TOTAL_BYTES = 2 * BRIDGE_BYTES
O_WEIGHT = "weight__blocks_0_cross_attn_o_weight"
Q_NORM_WEIGHT = "weight__blocks_0_cross_attn_norm_q_weight"


def _shape(value):
    shape = value.get_partial_shape()
    if not shape.is_static: raise ValueError("bridge requires static shapes")
    return tuple(shape.to_shape())


def _port_signature(port):
    return tuple((tuple(sorted(port.get_names())), str(port.get_element_type()), _shape(port)))


def _target(value, node_type, name):
    targets = [use.get_node() for use in value.output(0).get_target_inputs()]
    if len(targets) != 1 or targets[0].get_type_name() != node_type:
        raise ValueError(f"{name} must have one {node_type} consumer")
    return targets[0]


def _require_post_shape(model, ffn2_matmul):
    if type(ffn2_matmul) is not bool:
        raise TypeError("ffn2_matmul must be a bool")
    expected_outputs = ("x_out",) if not ffn2_matmul else ("x_out", "ffn2_linear_storage", "ffn2_gate_storage")
    expected_inputs = {"x": (1, ROWS, CHANNELS), "gate_msa": (1, 1, CHANNELS), "shift_mlp": (1, 1, CHANNELS), "scale_mlp": (1, 1, CHANNELS), "gate_mlp": (1, 1, CHANNELS), "cross_k": (1, KV_ROWS, CHANNELS), "cross_v": (1, KV_ROWS, CHANNELS), O_WEIGHT: (CHANNELS, CHANNELS), Q_NORM_WEIGHT: (CHANNELS,)}
    for name, shape in expected_inputs.items():
        port = model.input(name)
        if port.get_element_type() != ov.Type.bf16 or _shape(port) != shape:
            raise ValueError(f"POST input {name} must be BF16 {shape}")
    output_names = tuple(port.get_any_name() for port in model.outputs)
    if output_names != expected_outputs: raise ValueError(f"POST output ABI changed: {output_names!r}")
    if model.get_sinks() or model.get_variables():
        raise ValueError("bridge requires a sink-free stateless POST graph")
    return (tuple(_port_signature(port) for port in model.inputs),
            tuple(_port_signature(port) for port in model.outputs))


def _find_cross_q(model):
    weight = model.input(Q_NORM_WEIGHT).get_node()
    reshape = _target(weight, "Reshape", Q_NORM_WEIGHT)
    if _shape(reshape.output(0)) != (1, 1, CHANNELS):
        raise ValueError("cross-Q RMS scale reshape changed")
    multiply = _target(reshape, "Multiply", "cross-Q RMS scale")
    if (multiply.get_output_element_type(0) != ov.Type.bf16 or _shape(multiply.output(0)) != (1, ROWS, CHANNELS)):
        raise ValueError("cross-Q RMS multiply ABI changed")
    uses = list(multiply.output(0).get_target_inputs())
    if len(uses) != ROWS // 128 or any(use.get_index() != 0 or use.get_node().get_type_name() != "StridedSlice" or _shape(use.get_node().output(0)) != (1, 128, CHANNELS) for use in uses):
        raise ValueError("cross-Q RMS multiply is not the canonical 128-row fanout")
    return multiply.output(0)


def _find_cross_context(model):
    weight = model.input(O_WEIGHT).get_node()
    matmul = _target(weight, "MatMul", O_WEIGHT)
    if matmul.get_input_size() != 2 or matmul.input_value(1).get_node() is not weight:
        raise ValueError("cross-O MatMul weight input changed")
    context = matmul.input_value(0)
    node = context.get_node()
    if (node.get_type_name() != "Concat" or context.get_element_type() != ov.Type.bf16 or
            _shape(context) != (1, ROWS, CHANNELS) or node.get_axis() != 1 or
            node.get_input_size() != ROWS // 128 or any(
                _shape(node.input_value(i)) != (1, 128, CHANNELS) or
                node.input_value(i).get_element_type() != ov.Type.bf16
                for i in range(node.get_input_size()))):
        raise ValueError("cross-O does not consume the canonical BF16 Concat context")
    return node


def _slice(value, start):
    return ops.strided_slice(value, ops.constant([0, start, 0], ov.Type.i64), ops.constant([1, start + QUERY_ROWS, CHANNELS], ov.Type.i64), ops.constant([1, 1, 1], ov.Type.i64), [0, 0, 0], [0, 0, 0]).output(0)


def _bridge_context(q, key, value):
    model = bridge.build_bridge_model()
    model.input("q_bf16").get_node().output(0).replace(q)
    model.input("k_bf16").get_node().output(0).replace(key)
    model.input("v_bf16").get_node().output(0).replace(value)
    result = model.output("context_bf16").get_node()
    if result.get_type_name() != "Result" or result.get_input_size() != 1:
        raise ValueError("bridge output Result ABI changed")
    output = result.input_value(0)
    if output.get_element_type() != ov.Type.bf16 or _shape(output) != (1, QUERY_ROWS, CHANNELS):
        raise ValueError("bridge context output ABI changed")
    return output


def rewrite_post_cross_bridge(model, *, ffn2_matmul=False, role="post"):
    if type(role) is not str or role != "post":
        raise ValueError("only role='post' is supported")
    source = model.clone()
    input_abi, output_abi = _require_post_shape(source, ffn2_matmul)
    input_names = {port.get_any_name(): set(port.get_names()) for port in source.inputs}
    query = _find_cross_q(source)
    old_context = _find_cross_context(source)
    key, value = source.input("cross_k"), source.input("cross_v")
    replacement = ops.concat([
        _bridge_context(_slice(query, 0), key, value),
        _bridge_context(_slice(query, QUERY_ROWS), key, value),
    ], 1)
    old_context.output(0).replace(replacement.output(0))
    # Replacing a bridge Parameter unions its name into the model input.
    for port in source.inputs:
        port.get_tensor().set_names(input_names[port.get_any_name()])
    source.validate_nodes_and_infer_types()
    if (tuple(_port_signature(port) for port in source.inputs) != input_abi or
            tuple(_port_signature(port) for port in source.outputs) != output_abi):
        raise ValueError("bridge changed the POST input/output ABI")
    return source


def build_post_cross_bridge(*, ffn2_matmul=False):
    if type(ffn2_matmul) is not bool:
        raise TypeError("ffn2_matmul must be a bool")
    source = build_post_rows(ROWS, precision="bf16", remote_context=True, ffn2_matmul=ffn2_matmul)
    return rewrite_post_cross_bridge(source, ffn2_matmul=ffn2_matmul)


def bridge_metadata():
    per = bridge.bridge_inventory_bytes()["conservative_bridge_bytes"]
    if per != BRIDGE_BYTES:
        raise AssertionError(f"reviewed bridge inventory changed: {per}")
    return {"query_rows_per_bridge": QUERY_ROWS, "bridge_count": 2,
            "bridge_workspace_bytes_per_branch": per,
            "bridge_workspace_bytes_total": 2 * per,
            "alias_credit_bytes": 0, "memory_credit_bytes": 0}


__all__ = ["ROWS", "QUERY_ROWS", "CHANNELS", "KV_ROWS", "BRIDGE_BYTES",
           "BRIDGE_TOTAL_BYTES", "rewrite_post_cross_bridge",
           "build_post_cross_bridge", "bridge_metadata"]
