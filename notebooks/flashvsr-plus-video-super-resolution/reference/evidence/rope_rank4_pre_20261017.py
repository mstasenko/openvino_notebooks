"""Research-only RoPE layout rewrite; preserve every BF16 rounding boundary."""

HEADS, HEAD_DIM, DIM = 12, 128, 1536
PRIVATE_RESULTS = (
    "modulation_product", "q_linear_storage", "k_linear_storage",
    "norm1_storage", "q_rms_storage", "k_rms_storage",
)


def _abi(model):
    return tuple((port.get_any_name(), tuple(port.get_shape()), str(port.get_element_type()))
                 for port in (*model.inputs, *model.outputs))


def rank4_layout_shapes(model):
    return tuple(tuple(node.output(0).get_shape()) for node in model.get_ordered_ops()
                 if node.get_type_name() == "Reshape")


def _literal(node, index, expected):
    import openvino as ov
    value = node.input_value(index).get_node()
    if (value.get_type_name() != "Constant" or value.output(0).get_element_type() != ov.Type.i64
            or list(value.get_vector()) != expected):
        raise ValueError("Original RoPE shape/slice literal changed")


def _node(value, kind, inputs):
    import openvino as ov
    node = value.get_node()
    if (node.get_type_name() != kind or node.get_input_size() != inputs
            or value.get_element_type() != ov.Type.f32):
        raise ValueError("Original F32 RoPE operation changed")
    return node


def _match(final, sources, cos, sin, rows):
    """Authenticate the complete ordered expression, not just its ancestors."""
    flat = _node(final.input_value(0), "Reshape", 2)
    _literal(flat, 1, [1, rows, DIM])
    concat = _node(flat.input_value(0), "Concat", 2)
    if concat.get_attributes()["axis"] != 4:
        raise ValueError("Original RoPE pair concatenation changed")
    first = _node(concat.input_value(0), "Subtract", 2)
    second = _node(concat.input_value(1), "Add", 2)
    a, b = (_node(first.input_value(i), "Multiply", 2) for i in (0, 1))
    c, d = (_node(second.input_value(i), "Multiply", 2) for i in (0, 1))
    even, odd, cos5, sin5 = a.input_value(0), b.input_value(0), a.input_value(1), b.input_value(1)
    if ((c.input_value(0), c.input_value(1), d.input_value(0), d.input_value(1))
            != (even, sin5, odd, cos5)):
        raise ValueError("Original RoPE rotation operands changed")
    even_node, odd_node = _node(even, "StridedSlice", 4), _node(odd, "StridedSlice", 4)
    for node, begin, end in ((even_node, [0, 0, 0, 0, 0], [1, rows, 12, 64, 1]),
                             (odd_node, [0, 0, 0, 0, 1], [1, rows, 12, 64, 2])):
        _literal(node, 1, begin); _literal(node, 2, end); _literal(node, 3, [1] * 5)
        if any(any(values) for values in node.get_attributes().values()):
            raise ValueError("Original RoPE slice masks changed")
    if even_node.input_value(0) != odd_node.input_value(0):
        raise ValueError("Original RoPE pair slices have different sources")
    view = _node(even_node.input_value(0), "Reshape", 2)
    _literal(view, 1, [1, rows, HEADS, 64, 2])
    x32 = view.input_value(0)
    converted = _node(x32, "Convert", 1)
    lanes = [lane for lane, source in sources.items() if converted.input_value(0) == source]
    if len(lanes) != 1:
        raise ValueError("Original RoPE must read the private Q/K RMS storage")
    for value, parameter in ((cos5, cos), (sin5, sin)):
        reshape = _node(value, "Reshape", 2)
        _literal(reshape, 1, [1, rows, 1, 64, 1])
        if reshape.input_value(0) != parameter:
            raise ValueError("Original RoPE cosine/sine source changed")
    return lanes[0], x32


def _rank4_rope_f32(ops, x32, cos, sin, rows):
    import openvino as ov
    value = ops.reshape(x32, ops.constant([rows, HEADS, 64, 2], ov.Type.i64), False)
    even = ops.strided_slice(value, [0, 0, 0, 0], [rows, HEADS, 64, 1], [1] * 4, [0] * 4, [0] * 4)
    odd = ops.strided_slice(value, [0, 0, 0, 1], [rows, HEADS, 64, 2], [1] * 4, [0] * 4, [0] * 4)
    cos4 = ops.reshape(cos, [rows, 1, 64, 1], False)
    sin4 = ops.reshape(sin, [rows, 1, 64, 1], False)
    first = ops.subtract(ops.multiply(even, cos4), ops.multiply(odd, sin4))
    second = ops.add(ops.multiply(even, sin4), ops.multiply(odd, cos4))
    return ops.reshape(ops.concat((first, second), 3), [1, rows, DIM], False)


def rank4_rope(pre_model):
    """Clone the full C4096 PRE; remove only RoPE's leading singleton axis."""
    import openvino as ov
    from openvino import opset13 as ops
    if not isinstance(pre_model, ov.Model):
        raise TypeError("A full BF16 Plus PRE Model is required")
    rows = 4096
    if (tuple(pre_model.input("x").get_shape()) != (1, rows, DIM)
            or pre_model.input("x").get_element_type() != ov.Type.bf16):
        raise ValueError("Full C4096 BF16 PRE is required")
    for name in ("rope_cos", "rope_sin"):
        if (tuple(pre_model.input(name).get_shape()) != (1, rows, 1, 64)
                or pre_model.input(name).get_element_type() != ov.Type.f32):
            raise ValueError("External F32 RoPE feed ABI changed")
    for name in PRIVATE_RESULTS:
        port = pre_model.output(name)
        if tuple(port.get_shape()) != (1, rows, DIM) or port.get_element_type() != ov.Type.bf16:
            raise ValueError("Private BF16 PRE storage changed")
    clone = pre_model.clone()
    sources = {lane: clone.output(lane + "_rms_storage").get_node().input_value(0)
               for lane in ("q", "k")}
    matches = []
    for final in clone.get_ordered_ops():
        if final.get_type_name() != "Convert" or final.output(0).get_element_type() != ov.Type.bf16:
            continue
        flat = final.input_value(0).get_node()
        if (flat.get_type_name() == "Reshape"
                and flat.input_value(0).get_node().get_type_name() == "Concat"):
            lane, x32 = _match(final, sources, clone.input("rope_cos"), clone.input("rope_sin"), rows)
            matches.append((lane, final, x32))
    if len(matches) != 2 or {lane for lane, _, _ in matches} != {"q", "k"}:
        raise ValueError("Expected exactly one original Q and K RoPE")
    # Keep the original final Convert and all its consumers: rounding, names,
    # finite/pooled results and device-output wiring cannot be bypassed.
    for _, final, x32 in matches:
        rotated = _rank4_rope_f32(ops, x32, clone.input("rope_cos"), clone.input("rope_sin"), rows)
        final.input(0).replace_source_output(rotated.output(0))
    clone.validate_nodes_and_infer_types()
    if _abi(clone) != _abi(pre_model):
        raise ValueError("RoPE rewrite changed the external PRE ABI")
    return clone
