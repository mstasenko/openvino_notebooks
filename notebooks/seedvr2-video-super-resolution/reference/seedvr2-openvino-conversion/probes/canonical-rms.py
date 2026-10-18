"""Prospective, exact RMS canonicalization for the SeedVR2 DiT trial.

The function deliberately receives the OpenVINO opset and NumPy modules from
the caller.  Importing this file therefore has no runtime/model side effects.
"""


def _type_name(node):
    return str(node.get_type_name())


def _input(node, index):
    return node.input_value(index)


def _same_output(left, right):
    """Compare OpenVINO outputs by node instance and output index."""
    left_node = left.get_node()
    right_node = right.get_node()
    return (left_node.get_instance_id() == right_node.get_instance_id() and
            left.get_index() == right.get_index())


def _static_shape(output, label):
    partial = output.get_partial_shape()
    if not partial.is_static:
        raise ValueError(f"{label} must have a static shape, got {partial}")
    return tuple(int(value) for value in partial.to_shape())


def _require_f32(output, label):
    element_type = output.get_element_type().get_type_name()
    if element_type != "f32":
        raise ValueError(f"{label} must be f32, got {element_type}")


def _constant_values(node, np, label):
    if _type_name(node) != "Constant":
        raise ValueError(f"{label} must be Constant, got {_type_name(node)}")
    return np.asarray(node.get_vector())


def _scalar_constant(node, np, expected, label, require_f32=True):
    values = _constant_values(node, np, label)
    if values.size != 1:
        raise ValueError(f"{label} must be scalar, got shape {values.shape}")
    if require_f32 and node.get_output_element_type(0).get_type_name() != "f32":
        raise ValueError(f"{label} must be f32, got {node.get_output_element_type(0)}")
    # Compare in f32 so the exact representable 1e-5 value is accepted.
    actual = np.asarray(values.reshape(-1)[0], dtype=np.float32)
    wanted = np.asarray(expected, dtype=np.float32)
    if bool(actual != wanted):
        raise ValueError(f"{label} must equal {expected!r}, got {values.reshape(-1)[0]!r}")


def _axes_last(node, np, label):
    values = _constant_values(node, np, label)
    if values.size != 1 or int(values.reshape(-1)[0]) != -1:
        raise ValueError(f"{label} must be the one-element axes constant [-1], got {values}")


def _validate_target(divide, np):
    """Validate the exact SeedVR2 rank-3 attention RMS contract."""
    x = _input(divide, 0)
    sqrt = _input(divide, 1)
    x_shape = _static_shape(x, "RMS numerator")
    if len(x_shape) != 3 or x_shape[1:] != (20, 128):
        raise ValueError(f"RMS numerator must have static shape [*,20,128], got {x_shape}")
    _require_f32(x, "RMS numerator")
    _require_f32(sqrt, "RMS denominator")

    sqrt_node = sqrt.get_node()
    if _type_name(sqrt_node) != "Sqrt":
        raise ValueError(f"RMS denominator must be Sqrt, got {_type_name(sqrt_node)}")
    add = _input(sqrt_node, 0)
    add_node = add.get_node()
    if _type_name(add_node) != "Add":
        raise ValueError(f"Sqrt input must be Add, got {_type_name(add_node)}")

    mean = _input(add_node, 0)
    mean_node = mean.get_node()
    epsilon = _input(add_node, 1).get_node()
    if _type_name(mean_node) != "ReduceMean":
        raise ValueError(f"RMS Add input 0 must be ReduceMean, got {_type_name(mean_node)}")
    if not bool(mean_node.get_keep_dims()):
        raise ValueError("RMS ReduceMean must keep dimensions")
    _axes_last(_input(mean_node, 1).get_node(), np, "RMS reduction axes")
    _scalar_constant(epsilon, np, 1e-5, "RMS epsilon")
    power = _input(mean_node, 0)
    power_node = power.get_node()
    if _type_name(power_node) != "Power":
        raise ValueError(f"RMS ReduceMean input 0 must be Power, got {_type_name(power_node)}")
    _scalar_constant(_input(power_node, 1).get_node(), np, 2.0, "RMS exponent")
    if not _same_output(x, power_node.input_value(0)):
        raise ValueError("RMS Divide numerator must be the same x output used by Power")
    _require_f32(mean, "RMS mean")
    mean_shape = _static_shape(mean, "RMS mean")
    if mean_shape != (x_shape[0], x_shape[1], 1):
        raise ValueError(f"RMS mean shape must be {(x_shape[0], x_shape[1], 1)}, got {mean_shape}")
    sqrt_shape = _static_shape(sqrt, "RMS denominator")
    if sqrt_shape != mean_shape:
        raise ValueError(f"RMS Sqrt shape must be {mean_shape}, got {sqrt_shape}")
    _require_f32(divide.output(0), "RMS Divide output")
    if _static_shape(divide.output(0), "RMS Divide output") != x_shape:
        raise ValueError(f"RMS Divide output must have shape {x_shape}")
    return x, sqrt


def canonicalize_attention_rms(model, ops, np):
    """Rewrite only exact named attention `x / rms` nodes.

    Returns the rewritten Divide friendly names.  A targeted Divide whose
    graph does not satisfy the exact contract raises ``ValueError`` rather
    than being silently ignored.
    """
    rewritten = []
    # Snapshot first: replacement nodes must not affect traversal.
    for node in list(model.get_ordered_ops()):
        if _type_name(node) != "Divide":
            continue
        name = str(node.get_friendly_name())
        if ".attn.norm_q." not in name and ".attn.norm_k." not in name:
            continue

        x, rms = _validate_target(node, np)
        rms_shape = _static_shape(rms, "RMS denominator")
        ones = ops.constant(np.ones((1,) * len(rms_shape), dtype=np.float32))
        reciprocal = ops.divide(ones, rms)
        replacement = ops.multiply(x, reciprocal)
        replacement.set_friendly_name(name)
        replacement.output(0).set_names(set(str(v) for v in node.output(0).get_names()))
        node.output(0).replace(replacement.output(0))
        rewritten.append(name)
    return rewritten
