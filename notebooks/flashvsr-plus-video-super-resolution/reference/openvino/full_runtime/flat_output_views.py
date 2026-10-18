"""Layout-only result views for singleton-batch GPU stage boundaries."""


def flatten_results(graphs, internal_only=False):
    """Leave arithmetic/inputs untouched; flatten only [1,rows,width] Results.

    The caller must bind these outputs as 2D storage and share a canonical
    3D view with downstream stages. Pre callers restrict this to internal
    boundaries: a public one-window Q/K/V lane is also rank3 with axis0=1.
    Returned shapes define that binding;
    they are not a change to the model's source tensor order or precision.
    """
    import openvino as ov
    from openvino import opset13 as ops
    layouts = {}
    for graph in graphs:
        for port in list(graph.outputs):
            name = port.get_any_name()
            if internal_only and not name.startswith('modulation_'):
                continue
            shape = tuple(port.get_shape())
            if len(shape) != 3 or shape[0] != 1 or shape[1] <= 1:
                continue
            if name in layouts or port.get_element_type() not in (ov.Type.f16, ov.Type.f32):
                raise ValueError('Unique floating-point stage boundary required')
            layouts[name] = shape
            result = port.get_node()
            flat = ops.reshape(result.input_value(0), ops.constant(list(shape[1:]), ov.Type.i64), False)
            flat.output(0).get_tensor().set_names({name})
            result.input(0).replace_source_output(flat.output(0))
        graph.validate_nodes_and_infer_types()
    return layouts
