"""FlashVSR+ four-latent blocks using the existing OpenVINO graph builders.

These are component graphs, not a qualified end-to-end Plus backend. BF16 is
the upstream default; FP16 remains explicit for comparing the existing kernel.
"""
from dataclasses import dataclass

import numpy as np

from frame_geometry import FrameGeometry
from prepare_real import parameter_modules


@dataclass(frozen=True)
class BlockGeometry:
    frame: FrameGeometry
    f: int = 4

    def validate(self):
        if not isinstance(self.frame, FrameGeometry) or type(self.f) is not int or self.f != 4:
            raise ValueError("FlashVSR+ requires four latent frames and explicit visible geometry")

    @property
    def h(self):
        return self.frame.token_hw[0]

    @property
    def w(self):
        return self.frame.token_hw[1]

    @property
    def name(self):
        height, width = self.frame.working_hw
        return f"plus_f4_{width}x{height}"

    @property
    def tokens(self):
        return self.f * self.h * self.w

    @property
    def query_windows(self):
        return self.tokens // 128

    @property
    def token_shape(self):
        return (1, self.tokens, 1536)


def with_bf16_storage(model):
    """Change explicit half storage boundaries, preserving all F32 reductions.

    Rewrite a clone and validate after replacing all explicit storage types.
    No quantization or F32-to-BF16 blanket conversion is performed.
    """
    import openvino as ov
    from openvino import opset13 as ops

    converted = model.clone()
    for node in converted.get_ordered_ops():
        kind = node.get_type_name()
        if kind == "Parameter" and node.output(0).get_element_type() == ov.Type.f16:
            node.set_element_type(ov.Type.bf16)
        elif kind == "Constant" and node.output(0).get_element_type() == ov.Type.f16:
            value = ops.constant(np.asarray(node.get_data(), np.float32), ov.Type.bf16)
            value.set_friendly_name(node.get_friendly_name())
            value.output(0).get_tensor().set_names(node.output(0).get_names())
            node.output(0).replace(value.output(0))
        elif kind == "Convert" and node.output(0).get_element_type() == ov.Type.f16:
            node.set_attribute("destination_type", "bf16")
    converted.set_friendly_name(model.get_friendly_name() + "_bf16")
    converted.validate_nodes_and_infer_types()
    return converted


def _add_pool_results(model):
    import openvino as ov
    from openvino import opset13 as ops
    windows = int(model.output("q_windows").get_shape()[0])
    for lane in ("q", "k"):
        value = model.output(lane + "_windows").get_node().input_value(0)
        heads = ops.reshape(ops.convert(value, ov.Type.f32), [windows, 128, 12, 128], False)
        # Plus pools in F32, unlike the old port's F16 pooled boundary.
        pooled = ops.transpose(ops.reduce_mean(heads, [1], False), [1, 0, 2])
        pooled.output(0).get_tensor().set_names({lane + "_pooled"})
        model.add_results([ops.result(pooled)])


def _storage(model, precision):
    if precision not in ("bf16", "f16"):
        raise ValueError("Explicit BF16 or FP16 storage required")
    return with_bf16_storage(model) if precision == "bf16" else model


def build_block(role, frame, *, precision="bf16"):
    """Reuse the same pre/post math, with Plus geometry and pooled FP32 Q/K."""

    if precision not in ("bf16", "f16"):
        raise ValueError("Explicit BF16 or FP16 storage required")
    geometry = BlockGeometry(frame)
    geometry.validate()
    _, builders, _ = parameter_modules()
    model = builders.build_parameterized(role, geometry)
    model = _storage(model, precision)
    if role == "pre":
        _add_pool_results(model)
    model.set_friendly_name(f"flashvsr_plus_{role}_{geometry.name}_{precision}")
    model.validate_nodes_and_infer_types()
    return model


def preserve_modulation_product(model, role="pre", *, linear_stores=False,
                               norm_stores=False, state_sinks=False):
    """Keep the source modulation multiply rounded before its shift Add.

    A GPU-only Result makes this storage boundary observable to the compiler;
    otherwise a fused multiply-add can change BF16 keys before attention.
    The resident controller must bind this private result on the GPU.  The
    opt-in ``state_sinks`` form replaces only the six PRE private Results with
    BF16 Assign sinks; it does not add ReadValue feedback or alter the public
    output ABI.
    """
    if (role not in ("pre", "post") or ((linear_stores or norm_stores) and role != "pre")
            or (norm_stores and not linear_stores)):
        raise ValueError("Expected pre/post modulation and pre-only Q/K stores")
    if type(state_sinks) is not bool:
        raise ValueError("state_sinks must be boolean")
    if state_sinks and (role != "pre" or not linear_stores or not norm_stores):
        raise ValueError("PRE state sinks require both linear_stores and norm_stores")
    if "modulation_product" in {port.get_any_name() for port in model.outputs}:
        raise ValueError("Modulation product storage is already observed")
    weight_name = "self_attn_q" if role == "pre" else "ffn_0"
    parameter = model.input(f"weight__blocks_0_{weight_name}_weight").get_node()
    projections = [use.get_node() for use in parameter.output(0).get_target_inputs()
                   if use.get_index() == 1]
    if len(projections) != 1 or projections[0].get_type_name() != "MatMul":
        raise ValueError("Source modulation projection topology changed")
    shifted = projections[0].input_value(0).get_node()
    if shifted.get_type_name() != "Add":
        raise ValueError("Source pre modulation shift changed")
    product = shifted.input_value(0)
    if (product.get_node().get_type_name() != "Multiply"
            or product.get_element_type() != model.input("x").get_element_type()
            or product.get_partial_shape() != model.input("x").get_partial_shape()):
        raise ValueError("Source pre modulation product storage changed")
    observations = [("modulation_product", product)]
    if linear_stores:
        # Linear rounds before the source RMSNorm converts back to F32.
        # Without an observable store, GPU fusion can bypass that rounding.
        for lane in ("q", "k"):
            weight = model.input(f"weight__blocks_0_self_attn_{lane}_weight").get_node()
            matmul, = [use.get_node() for use in weight.output(0).get_target_inputs()
                       if use.get_index() == 1]
            bias_add, = [use.get_node() for use in matmul.output(0).get_target_inputs()]
            if (matmul.get_type_name() != "MatMul" or bias_add.get_type_name() != "Add"
                    or bias_add.output(0).get_element_type() != product.get_element_type()
                    or bias_add.output(0).get_partial_shape() != product.get_partial_shape()):
                raise ValueError("Source pre Linear storage changed")
            observations.append((lane + "_linear_storage", bias_add.output(0)))
    if norm_stores:
        normalized = product.get_node().input_value(0)
        if normalized.get_node().get_type_name() != "Convert":
            raise ValueError("Source norm1 storage changed")
        observed = [("norm1_storage", normalized)]
        for lane in ("q", "k"):
            scale = model.input(f"weight__blocks_0_self_attn_norm_{lane}_weight").get_node()
            reshape, = [use.get_node() for use in scale.output(0).get_target_inputs()]
            weighted, = [use.get_node() for use in reshape.output(0).get_target_inputs()]
            if reshape.get_type_name() != "Reshape" or weighted.get_type_name() != "Multiply":
                raise ValueError("Source RMS scale topology changed")
            observed.append((lane + "_rms_storage", weighted.output(0)))
        for name, value in observed:
            if (value.get_element_type() != product.get_element_type()
                    or value.get_partial_shape() != product.get_partial_shape()):
                raise ValueError("Source normalization storage changed")
            observations.append((name, value))

    if state_sinks:
        import openvino as ov
        from openvino import opset13 as ops
        from openvino.op.util import Variable, VariableInfo

        if model.input("x").get_element_type() != ov.Type.bf16:
            raise ValueError("PRE state sinks require BF16 storage")
        shape = model.input("x").get_partial_shape()
        if not shape.is_static:
            raise ValueError("PRE state sinks require a static BF16 shape")
        state_names = tuple(name for name, _ in observations)
        state_ids = {"plus_pre_state_" + name for name in state_names}
        existing_ids = {
            variable.get_info().variable_id for variable in model.get_variables()
        }
        existing_assign_names = {
            sink.get_friendly_name() for sink in model.get_sinks()
            if sink.get_type_name() == "Assign"
        }
        if existing_ids & state_ids or existing_assign_names & {
                "assign_plus_pre_state_" + name for name in state_names
        }:
            raise ValueError("PRE state sinks are already applied")
        for name, value in observations:
            if (value.get_element_type() != ov.Type.bf16
                    or value.get_partial_shape() != shape or value.get_index() != 0):
                raise ValueError(f"PRE state sink {name} is not the canonical BF16 shape")

    def observe(name, value):
        value.get_tensor().set_names({name})
        if not state_sinks:
            model.add_outputs([value])
            return
        info = VariableInfo()
        info.data_shape = ov.PartialShape(list(shape))
        info.data_type = ov.Type.bf16
        info.variable_id = "plus_pre_state_" + name
        variable = Variable(info)
        assign = ops.assign(
            value.get_node(), variable,
            "assign_plus_pre_state_" + name,
        )
        assign.set_friendly_name("assign_plus_pre_state_" + name)
        assigned = assign.output(0)
        assigned.get_tensor().set_names({name})
        model.add_sinks([assign])

    for name, value in observations:
        observe(name, value)
    model.validate_nodes_and_infer_types()
    return model


def build_bounded_pre(frame, *, rows=4096, precision="bf16"):
    """Schedule pointwise pre math in whole windows, preserving complete context."""
    from window_pre import bounded_model
    source = build_block("pre", frame, precision="f16")
    for lane in ("q", "k"):
        source.remove_result(source.output(lane + "_pooled").get_node())
    bounded, geometry, residual = bounded_model(source, rows)
    bounded = _storage(bounded, precision)
    _add_pool_results(bounded)
    bounded.validate_nodes_and_infer_types()
    return bounded, geometry, residual


@dataclass(frozen=True)
class _PostRows:
    # Cross-attention reads all512 text tokens; only independent query rows
    # are bounded. This geometry is intentionally independent of image shape.
    tokens: int

    def validate(self):
        if type(self.tokens) is not int or not 128 <= self.tokens <= 8192 or self.tokens % 128:
            raise ValueError("Post requires at most8192 rows in whole128-token windows")

    @property
    def token_shape(self):
        return (1, self.tokens, 1536)

    @property
    def name(self):
        return f"plus_post_rows{self.tokens}"


def build_post_rows(rows=3840, *, precision="bf16", remote_context=False,
                    cross_query_chunk_tokens=128, ffn2_matmul=False):
    """Bound FFN/cross-attention workspace without reducing attended context.

    ``remote_context`` keeps the canonical self-attention context as two
    native BF16 window-port inputs.  The reshape is only graph wiring: it does
    not add a host-side full-context owner.
    """
    if type(remote_context) is not bool:
        raise ValueError("remote_context must be boolean")
    if type(ffn2_matmul) is not bool:
        raise ValueError("ffn2_matmul must be boolean")
    if remote_context and (rows != 4096 or precision != "bf16"):
        raise ValueError("Remote post context requires rows=4096 and BF16 storage")
    _, builders, _ = parameter_modules()
    model = _storage(
        builders.build_parameterized_post(
            _PostRows(rows), cross_query_chunk_tokens=cross_query_chunk_tokens,
            ffn2_matmul=ffn2_matmul,
        ),
        precision,
    )
    if not remote_context:
        return model

    import openvino as ov
    from openvino import opset13 as ops

    original = model.input("self_context").get_node()
    slots = [ops.parameter([12, 16, 128, 128], ov.Type.bf16,
                           name=f"self_context_{index}") for index in (0, 1)]
    for slot in slots:
        slot.output(0).get_tensor().set_names({slot.get_friendly_name()})
    joined = ops.concat(slots, 1)
    ordered = ops.transpose(
        joined, ops.constant(np.array([1, 2, 0, 3], dtype=np.int64))
    )
    context = ops.reshape(
        ordered, ops.constant(np.array([1, 4096, 1536], dtype=np.int64)), False
    )
    original.output(0).replace(context.output(0))
    model.remove_parameter(original)
    model.add_parameters(slots)
    model.validate_nodes_and_infer_types()
    return model
