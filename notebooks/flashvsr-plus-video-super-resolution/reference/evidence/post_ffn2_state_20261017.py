"""Small CPU-only FFN2 Assign-sink prototype; not a runtime integration."""

from __future__ import annotations

FFN2_PRIVATE_OUTPUTS = ("ffn2_linear_storage", "ffn2_gate_storage")
OUTPUTS = ("x_out",) + FFN2_PRIVATE_OUTPUTS
SHAPE = (1, 4096, 1536)
PRIVATE_STATE_BYTES = 25_165_824


def _shape(port):
    partial = port.get_partial_shape()
    if not partial.is_static:
        raise ValueError("FFN2 state output must have a static shape")
    return tuple(int(value) for value in port.get_shape())


def _validate_source(model, *, role="post"):
    if role != "post":
        raise ValueError("FFN2 state sinks are post-only")
    if tuple(port.get_any_name() for port in model.outputs) != OUTPUTS:
        raise ValueError("expected x_out, ffn2_linear_storage, ffn2_gate_storage outputs")
    if model.get_variables() or model.get_sinks():
        raise ValueError("source FFN2 graph must not already contain variables or sinks")
    import openvino as ov

    for port in model.outputs:
        if (port.get_node().get_type_name() != "Result"
                or _shape(port) != SHAPE
                or port.get_element_type() != ov.Type.bf16
                or port.get_node().input_value(0).get_index() != 0):
            raise ValueError(f"invalid FFN2 output ABI: {port.get_any_name()}")
    linear = model.output("ffn2_linear_storage").get_node().input_value(0).get_node()
    gate = model.output("ffn2_gate_storage").get_node().input_value(0).get_node()
    final = model.output("x_out").get_node().input_value(0).get_node()
    if (linear.get_type_name(), gate.get_type_name(), final.get_type_name()) != (
            "Add", "Multiply", "Add"):
        raise ValueError("FFN2 source must be Add(linear)->Multiply(gate)->Add(x_out)")
    if sum(linear.input_value(i).get_node().get_type_name() == "MatMul"
           for i in range(linear.get_input_size())) != 1:
        raise ValueError("FFN2 linear Add must consume exactly one MatMul")
    if sum(gate.input_value(i).get_node() is linear for i in range(gate.get_input_size())) != 1:
        raise ValueError("FFN2 gate must consume the linear Add")
    if sum(final.input_value(i).get_node() is gate for i in range(final.get_input_size())) != 1:
        raise ValueError("x_out must consume the gate Multiply")


def _validate_assign_states(model):
    """Require each private sink to own the exact observable BF16 state ABI."""
    import openvino as ov

    expected = {"plus_post_ffn2_state_" + name: name for name in FFN2_PRIVATE_OUTPUTS}
    variables = {variable.info.variable_id: variable for variable in model.get_variables()}
    sinks = {sink.get_variable_id(): sink for sink in model.get_sinks()
             if sink.get_type_name() == "Assign"}
    if set(variables) != set(expected) or set(sinks) != set(expected):
        raise ValueError("FFN2 state IDs are not exactly the two expected Assign states")
    for state_id, output_name in expected.items():
        sink, variable = sinks[state_id], variables[state_id]
        output = sink.output(0)
        if (tuple(int(value) for value in output.get_shape()) != SHAPE
                or output.get_element_type() != ov.Type.bf16
                or output.get_any_name() != output_name):
            raise ValueError(f"invalid Assign output ABI: {output_name}")
        info = variable.info
        if (info.variable_id != state_id or info.data_type != ov.Type.bf16
                or tuple(int(value) for value in info.data_shape.to_shape()) != SHAPE):
            raise ValueError(f"invalid Assign variable ABI: {state_id}")


def replace_ffn2_results_with_assigns(model, *, role="post"):
    """Clone the fixed post graph and replace its two private Results."""
    _validate_source(model, role=role)
    converted = model.clone()
    import openvino as ov
    from openvino import opset13 as ops
    from openvino.op.util import Variable, VariableInfo

    for name in FFN2_PRIVATE_OUTPUTS:
        result = converted.output(name).get_node()
        info = VariableInfo()
        info.data_shape = ov.PartialShape(list(SHAPE))
        info.data_type = ov.Type.bf16
        info.variable_id = "plus_post_ffn2_state_" + name
        assign = ops.assign(
            result.input_value(0).get_node(),
            Variable(info),
            "assign_plus_post_ffn2_state_" + name,
        )
        assign.set_friendly_name("assign_plus_post_ffn2_state_" + name)
        assign.output(0).get_tensor().set_names({name})
        converted.remove_result(result)
        converted.add_sinks([assign])
    converted.validate_nodes_and_infer_types()
    if tuple(port.get_any_name() for port in converted.outputs) != ("x_out",):
        raise ValueError("Assign rewrite changed the public x_out ABI")
    if _shape(converted.output("x_out")) != SHAPE:
        raise ValueError("Assign rewrite changed x_out shape")
    if converted.output("x_out").get_element_type() != ov.Type.bf16:
        raise ValueError("Assign rewrite changed x_out type")
    if (len(converted.get_variables()) != 2
            or len(converted.get_sinks()) != 2
            or any(node.get_type_name() == "ReadValue"
                   for node in converted.get_ordered_ops())):
        raise ValueError("FFN2 rewrite must have two Assign sinks and no ReadValue")
    _validate_assign_states(converted)
    descriptor = {
        "public_outputs": ["x_out"],
        "private_outputs": list(FFN2_PRIVATE_OUTPUTS),
        "private_state_bytes": PRIVATE_STATE_BYTES,
        "memory_credit_bytes": 0,
        "states": [
            {"source_output": name, "state_id": "plus_post_ffn2_state_" + name,
             "shape": list(SHAPE), "element_type": "bf16", "assign_type": "Assign",
             "read_value_feedback": False}
            for name in FFN2_PRIVATE_OUTPUTS
        ],
    }
    return converted, descriptor


def build_ffn2_state_sink_probe():
    """Build the reviewed fixed 4096-row BF16 remote post graph."""
    from plus_graphs import build_post_rows

    model = build_post_rows(
        rows=4096, precision="bf16", remote_context=True, ffn2_matmul=True
    )
    return replace_ffn2_results_with_assigns(model)
