#!/usr/bin/env python3
"""Opt-in token-batch Conv1x1 rewrite for the reviewed post IR.

The source graph uses one NCHW image whose width is the token count.  For a
selected learned projection this helper reuses its runtime Parameter views and
replaces the final transpose's consumers with:

    [1,N,in] -> reshape [N,in,1,1] -> Conv1x1 + bias -> reshape [1,N,out].

The caller owns immutability and must pass ``model.clone()``.  This module does
not create learned Constants, query a device, or compile/infer a model.
"""

from __future__ import annotations

from collections import namedtuple

_PREFIX = {"self": "blocks.0.self_attn.o", "ffn": "blocks.0.ffn.2"}
_AXES = (0, 2, 1)

def _bad(message: str):
    raise ValueError(f"token-batch post rewrite rejected: {message}")

def _shape(port, label: str) -> tuple[int, ...]:
    partial = port.get_partial_shape()
    if not partial.is_static:
        _bad(f"{label} is dynamic: {partial}")
    return tuple(int(value) for value in partial.to_shape())

def _src(node, index: int):
    return node.input(index).get_source_output()

def _one(output, index: int, label: str):
    targets = list(output.get_target_inputs())
    if len(targets) != 1 or targets[0].get_index() != index:
        _bad(f"{label} must have one consumer on input {index}")
    return targets[0].get_node()

def _node(node, kind: str, label: str):
    if node.get_type_name() != kind:
        _bad(f"{label} must be {kind}, got {node.get_type_name()}")

def _vector(node, ov, expected, label: str):
    if node.get_type_name() not in ("Constant", "Const"):
        _bad(f"{label} must be an i64 Constant")
    if node.output(0).get_element_type() != ov.Type.i64:
        _bad(f"{label} must be i64")
    try:
        actual = tuple(int(value) for value in node.get_vector())
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        _bad(f"{label} is unreadable: {exc}")
    if actual != tuple(expected):
        _bad(f"{label} must be {tuple(expected)!r}, got {actual!r}")

def _attrs(node, label: str, **expected):
    actual = node.get_attributes()
    for key, wanted in expected.items():
        value = actual.get(key)
        if isinstance(wanted, tuple):
            try:
                value = tuple(int(item) for item in value)
            except (TypeError, ValueError):
                pass
        elif key == "auto_pad":
            value, wanted = str(value).lower(), str(wanted).lower()
        if value != wanted:
            _bad(f"{label}.{key} must be {wanted!r}, got {value!r}")

def _name(output) -> str:
    names = output.get_names()
    if len(names) != 1:
        _bad(f"runtime Parameter must have exactly one tensor name, got {sorted(names)!r}")
    return next(iter(names))

def _parameters(model):
    result = {}
    for parameter in model.get_parameters():
        name = _name(parameter.output(0))
        if name in result:
            _bad(f"duplicate runtime Parameter {name!r}")
        result[name] = parameter
    return result

_Match = namedtuple("_Match", "projection source weight bias weight_view bias_view final_transpose tokens input_width output_width")

def _authenticate(ov, projection: str, parameters) -> _Match:
    prefix = _PREFIX[projection]
    weight_name = "weight__" + prefix.replace(".", "_") + "_weight"
    bias_name = "weight__" + prefix.replace(".", "_") + "_bias"
    if weight_name not in parameters or bias_name not in parameters:
        _bad(f"missing exact Parameters {weight_name!r}/{bias_name!r}")
    weight, bias = parameters[weight_name], parameters[bias_name]
    if weight.output(0).get_element_type() != ov.Type.f16 or bias.output(0).get_element_type() != ov.Type.f16:
        _bad(f"{projection} learned Parameters must be F16")
    weight_shape, bias_shape = _shape(weight.output(0), weight_name), _shape(bias.output(0), bias_name)
    if len(weight_shape) != 2 or len(bias_shape) != 1 or bias_shape != (weight_shape[0],):
        _bad(f"{projection} learned dimensions are not [out,in]/[out]")
    output_width, input_width = weight_shape
    if output_width <= 0 or input_width <= 0:
        _bad(f"{projection} learned dimensions must be positive")
    if projection == "self" and output_width != input_width:
        _bad("self.o must retain its square F16 projection dimensions")

    weight_view = _one(weight.output(0), 0, f"{projection} weight")
    bias_view = _one(bias.output(0), 0, f"{projection} bias")
    _node(weight_view, "Reshape", f"{projection} weight view")
    _node(bias_view, "Reshape", f"{projection} bias view")
    if _src(weight_view, 0).get_node() != weight or _src(bias_view, 0).get_node() != bias:
        _bad(f"{projection} reshape views do not consume exact runtime Parameters")
    _vector(_src(weight_view, 1).get_node(), ov, (output_width, input_width, 1, 1),
            f"{projection} weight reshape shape")
    _vector(_src(bias_view, 1).get_node(), ov, (1, output_width, 1, 1),
            f"{projection} bias reshape shape")
    _attrs(weight_view, f"{projection} weight reshape", special_zero=False)
    _attrs(bias_view, f"{projection} bias reshape", special_zero=False)
    # The direct F16 Parameter inputs plus these exact reshape Constants make
    # the view element types/shapes unambiguous; model validation is final.

    convolution = _one(weight_view.output(0), 1, f"{projection} weight view")
    _node(convolution, "Convolution", f"{projection} convolution")
    if _src(convolution, 1).get_node() != weight_view:
        _bad(f"{projection} convolution does not use its weight view")
    _attrs(convolution, f"{projection} convolution", strides=(1, 1), dilations=(1, 1),
           pads_begin=(0, 0), pads_end=(0, 0), auto_pad="explicit")

    added = _one(convolution.output(0), 0, f"{projection} convolution")
    _node(added, "Add", f"{projection} bias add")
    if _src(added, 0).get_node() != convolution or _src(added, 1).get_node() != bias_view:
        _bad(f"{projection} Add inputs must be [convolution,bias-view]")
    _attrs(added, f"{projection} bias add", auto_broadcast="numpy")

    output_view = _one(added.output(0), 0, f"{projection} bias add")
    _node(output_view, "Reshape", f"{projection} output reshape")
    if _src(output_view, 0).get_node() != added:
        _bad(f"{projection} output reshape does not consume the bias add")
    _attrs(output_view, f"{projection} output reshape", special_zero=False)
    old_conv_shape = _shape(convolution.output(0), f"{projection} convolution")
    if len(old_conv_shape) != 4 or old_conv_shape[:3] != (1, output_width, 1) or old_conv_shape[3] <= 0:
        _bad(f"{projection} old convolution shape must be [1,out,1,N]")
    tokens = old_conv_shape[3]
    _vector(_src(output_view, 1).get_node(), ov, (1, output_width, tokens),
            f"{projection} output reshape shape")

    final_transpose = _one(output_view.output(0), 0, f"{projection} output reshape")
    _node(final_transpose, "Transpose", f"{projection} final transpose")
    if _src(final_transpose, 0).get_node() != output_view:
        _bad(f"{projection} final transpose does not consume the output reshape")
    _vector(_src(final_transpose, 1).get_node(), ov, _AXES, f"{projection} final transpose axes")
    if _shape(final_transpose.output(0), f"{projection} final transpose") != (1, tokens, output_width):
        _bad(f"{projection} final transpose has the wrong output shape")
    if final_transpose.output(0).get_element_type() != ov.Type.f16:
        _bad(f"{projection} final transpose must be F16")

    input_view = _src(convolution, 0).get_node()
    _node(input_view, "Reshape", f"{projection} input reshape")
    _attrs(input_view, f"{projection} input reshape", special_zero=False)
    _vector(_src(input_view, 1).get_node(), ov, (1, input_width, 1, tokens),
            f"{projection} input reshape shape")
    input_transpose = _src(input_view, 0).get_node()
    _node(input_transpose, "Transpose", f"{projection} input transpose")
    _vector(_src(input_transpose, 1).get_node(), ov, _AXES, f"{projection} input transpose axes")
    source = _src(input_transpose, 0)
    if source.get_element_type() != ov.Type.f16 or _shape(source, f"{projection} source") != (1, tokens, input_width):
        _bad(f"{projection} source must be F16 [1,N,in]")
    if _one(input_transpose.output(0), 0, f"{projection} input transpose") != input_view:
        _bad(f"{projection} input transpose has an unexpected consumer")
    return _Match(projection, source, weight, bias, weight_view, bias_view,
                  final_transpose, tokens, input_width, output_width)


def rewrite_post(model, projections=("self", "ffn")) -> dict[str, object]:
    """Rewrite selected projections in a caller-supplied cloned OpenVINO Model."""
    if isinstance(projections, (str, bytes)):
        _bad("projections must be an iterable of 'self'/'ffn', not a string")
    try:
        selected = tuple(projections)
    except TypeError as exc:
        _bad(f"projections is not iterable: {exc}")
    if len(selected) != len(set(selected)):
        _bad(f"duplicate projection selector: {selected!r}")
    if any(projection not in _PREFIX for projection in selected):
        _bad(f"unsupported projection selector(s): {selected!r}")
    import openvino as ov
    from openvino import opset13 as ops

    parameters = _parameters(model)
    matches = [_authenticate(ov, projection, parameters) for projection in selected]
    if len({item.final_transpose.get_instance_id() for item in matches}) != len(matches):
        _bad("requested projections share one final transpose")

    branches = []
    for item in matches:
        input_shape, output_shape = [item.tokens, item.input_width, 1, 1], [1, item.tokens, item.output_width]
        input_shape_const = ops.constant(input_shape, ov.Type.i64)
        input_4d = ops.reshape(item.source, input_shape_const, False,
                               name=f"token_batch_{item.projection}_input_reshape")
        convolution = ops.convolution(input_4d, item.weight_view.output(0), [1, 1], [0, 0], [0, 0], [1, 1],
                                      auto_pad="EXPLICIT", name=f"token_batch_{item.projection}_conv1x1")
        added = ops.add(convolution, item.bias_view.output(0), name=f"token_batch_{item.projection}_bias_add")
        output_shape_const = ops.constant(output_shape, ov.Type.i64)
        output = ops.reshape(added, output_shape_const, False,
                             name=f"token_batch_{item.projection}_output_reshape")
        if item.final_transpose.output(0).get_names():
            output.output(0).get_tensor().set_names(item.final_transpose.output(0).get_names())
        if _shape(convolution.output(0), f"{item.projection} token-batch convolution") != (item.tokens, item.output_width, 1, 1):
            _bad(f"{item.projection} token-batch convolution shape inference failed")
        if _shape(output.output(0), f"{item.projection} token-batch output") != tuple(output_shape):
            _bad(f"{item.projection} token-batch output shape inference failed")
        branches.append((item, output))

    # Every requested pattern was authenticated before this first mutation.
    for item, output in branches:
        item.final_transpose.output(0).replace(output.output(0))
    model.validate_nodes_and_infer_types()
    return {
        "variant": "token_batch_conv1x1",
        "projections": list(selected),
        "rewritten": [
            {
                "projection": item.projection,
                "old_final_transpose": str(item.final_transpose.get_friendly_name()),
                "input_shape": [item.tokens, item.input_width, 1, 1],
                "weight_shape": [item.output_width, item.input_width, 1, 1],
                "bias_shape": [1, item.output_width, 1, 1],
                "output_shape": [1, item.tokens, item.output_width],
                "runtime_weight": _name(item.weight.output(0)),
                "runtime_bias": _name(item.bias.output(0)),
            }
            for item, _output in branches
        ],
    }

