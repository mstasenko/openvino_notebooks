"""CPU-only design proof for a lossless BF16-to-qualified-XMX bridge.

The pinned CustomLayer runtime does not admit native BF16 ports.  This draft
keeps the model's BF16 parameters/outputs outside CustomLayer and uses F32
only as an exact numerical carrier: BF16 -> F32 -> raw ushort transport ->
F32 -> BF16.  The three bridge ports never perform ordinary FP16 arithmetic;
the existing Plus sparse layer remains the only attention operation.

This module is an evidence prototype, not a production runner.  Its pure
packing functions are used by the CPU tests; ``build_bridge_model`` only
constructs a static OpenVINO graph and never creates a Core or device.
"""

from __future__ import annotations

from math import prod

import numpy as np

import openvino as ov
from openvino import opset13 as ops

from plus_sparse_xmx import OtcPlusSparseXmx


BF16 = ov.Type.bf16
F32 = ov.Type.f32
I32 = ov.Type.i32
F16_TRANSPORT = ov.Type.f16
Q_SHAPE = (1, 2048, 1536)
KV_SHAPE = (1, 512, 1536)
Q_F32_SHAPE = (1, 1, 2048, 1536)
KV_F32_SHAPE = (1, 1, 512, 1536)
Q_PACKED_SHAPE = (12, 16, 128, 128)
KV_PACKED_SHAPE = (4, 128, 12, 128)
BLOCK_SHAPE = (12, 16, 1, 4)
_FINITE_EXP_MASK = np.uint16(0x7F80)
_FINITE_EXP_ALL_ONES = np.uint16(0x7F80)


def _require_bf16_bits(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.uint16:
        raise TypeError(f"{name} must contain uint16 BF16 bits")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous uint16 BF16 bits with shape {shape}")
    if np.any((value & _FINITE_EXP_MASK) == _FINITE_EXP_ALL_ONES):
        raise ValueError(f"{name} contains non-finite BF16 words")
    return value


def _require_f32_bf16_decode(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Require an F32 array that is an exact decode of finite BF16 words."""
    value = np.asarray(value)
    if value.dtype != np.float32 or value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous F32 with shape {shape}")
    words = value.view(np.uint32)
    if np.any((words & np.uint32(0xFFFF)) != 0):
        raise ValueError(f"{name} is not an exact BF16-to-F32 decode")
    if np.any((words >> np.uint32(16) & np.uint32(0x7F80)) == np.uint32(0x7F80)):
        raise ValueError(f"{name} contains non-finite BF16 words")
    return value


def bf16_bits_to_f32(value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Decode raw BF16 bits without an intermediate numeric F16 conversion."""
    value = np.asarray(value)
    if value.dtype != np.uint16 or not value.flags.c_contiguous:
        raise TypeError("contiguous uint16 BF16 bits required")
    if shape is not None:
        _require_bf16_bits(value, shape, "value")
    return (value.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_bf16_decode_to_bits(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return exact BF16 words from a validated F32 BF16 decode."""
    value = _require_f32_bf16_decode(value, shape, name)
    return np.ascontiguousarray((value.view(np.uint32) >> np.uint32(16)).astype(np.uint16))


def pack_q_bits(value: np.ndarray) -> np.ndarray:
    """Map [1,2048,1536] BF16 bits to head-major [12,16,128,128]."""
    value = _require_bf16_bits(value, Q_SHAPE, "q")
    return np.ascontiguousarray(value.reshape(1, 16, 128, 12, 128)[0]
                                .transpose(2, 0, 1, 3))


def pack_kv_bits(value: np.ndarray) -> np.ndarray:
    """Map [1,512,1536] BF16 bits to canonical [4,128,12,128]."""
    value = _require_bf16_bits(value, KV_SHAPE, "kv")
    return np.ascontiguousarray(value.reshape(1, 4, 128, 12, 128)
                                .reshape(KV_PACKED_SHAPE))


def unpack_context_bits(value: np.ndarray) -> np.ndarray:
    """Map the qualified raw-BF16 attention result back to [1,2048,1536]."""
    value = _require_bf16_bits(value, Q_PACKED_SHAPE, "attention")
    return np.ascontiguousarray(value.transpose(1, 2, 0, 3)
                                .reshape(Q_SHAPE))


def bridge_roundtrip(value: np.ndarray) -> np.ndarray:
    """Pure CPU proof path for the BF16 -> F32 -> raw -> F32 -> BF16 route."""
    bits = _require_bf16_bits(value, Q_SHAPE, "q")
    # The two packing operations model raw ushort transport; they never cast
    # a BF16 value through numeric FP16.
    packed = pack_q_bits(bits)
    unpacked = unpack_context_bits(packed)
    return f32_bf16_decode_to_bits(
        bf16_bits_to_f32(unpacked, Q_SHAPE), Q_SHAPE, "unpacked"
    )


def _static_input(node, expected_type, expected_shape, name):
    if node.get_input_size() != 1:
        raise ValueError(f"{name} requires exactly one input")
    shape = node.get_input_partial_shape(0)
    if not shape.is_static or tuple(shape.to_shape()) != expected_shape:
        raise ValueError(f"{name} requires static shape {expected_shape}")
    if node.get_input_element_type(0) != expected_type:
        raise TypeError(f"{name} input type must be {expected_type}")


class OtcCrossQPack(ov.Op):
    """F32 exact-BF16 decode -> raw BF16 bits exposed as F16 transport."""

    def __init__(self, inputs):
        super().__init__(self, inputs)
        self.validate_and_infer_types()

    def validate_and_infer_types(self):
        _static_input(self, F32, Q_F32_SHAPE, "OtcCrossQPack")
        self.set_output_type(0, F16_TRANSPORT, ov.PartialShape(Q_PACKED_SHAPE))

    def clone_with_new_inputs(self, inputs):
        return OtcCrossQPack(inputs)

    def visit_attributes(self, visitor):
        return True


class OtcCrossKVPack(ov.Op):
    """F32 exact-BF16 decode -> canonical raw BF16 K/V transport."""

    def __init__(self, inputs):
        super().__init__(self, inputs)
        self.validate_and_infer_types()

    def validate_and_infer_types(self):
        _static_input(self, F32, KV_F32_SHAPE, "OtcCrossKVPack")
        self.set_output_type(0, F16_TRANSPORT, ov.PartialShape(KV_PACKED_SHAPE))

    def clone_with_new_inputs(self, inputs):
        return OtcCrossKVPack(inputs)

    def visit_attributes(self, visitor):
        return True


class OtcCrossContextUnpack(ov.Op):
    """Raw BF16 F16 transport -> numeric F32, with no FP16 arithmetic."""

    def __init__(self, inputs):
        super().__init__(self, inputs)
        self.validate_and_infer_types()

    def validate_and_infer_types(self):
        _static_input(self, F16_TRANSPORT, Q_PACKED_SHAPE, "OtcCrossContextUnpack")
        self.set_output_type(0, F32, ov.PartialShape(Q_F32_SHAPE))

    def clone_with_new_inputs(self, inputs):
        return OtcCrossContextUnpack(inputs)

    def visit_attributes(self, visitor):
        return True


def build_bridge_model():
    """Construct the fixed 2K-query/4-window proof graph without a Core."""
    q = ops.parameter(Q_SHAPE, BF16, name="q_bf16")
    k = ops.parameter(KV_SHAPE, BF16, name="k_bf16")
    v = ops.parameter(KV_SHAPE, BF16, name="v_bf16")
    qf = ops.convert(q, F32)
    kf = ops.convert(k, F32)
    vf = ops.convert(v, F32)
    qf.output(0).get_tensor().set_names({"q_f32_exact_bf16"})
    kf.output(0).get_tensor().set_names({"k_f32_exact_bf16"})
    vf.output(0).get_tensor().set_names({"v_f32_exact_bf16"})
    # CustomLayer JIT dimensions are BFYX, not necessarily the original IR
    # rank-three shape. Explicit F32 reshapes make its four pitches unambiguous.
    qp = OtcCrossQPack([ops.reshape(qf, Q_F32_SHAPE, False).output(0)]).output(0)
    kp = OtcCrossKVPack([ops.reshape(kf, KV_F32_SHAPE, False).output(0)]).output(0)
    vp = OtcCrossKVPack([ops.reshape(vf, KV_F32_SHAPE, False).output(0)]).output(0)
    for output, name in ((qp, "q_raw_bf16_f16_transport"),
                         (kp, "k_raw_bf16_f16_transport"),
                         (vp, "v_raw_bf16_f16_transport")):
        output.get_tensor().set_names({name})
    blocks = ops.constant(np.broadcast_to(np.arange(4, dtype=np.int32), BLOCK_SHAPE).copy(),
                          dtype=I32)
    sparse = OtcPlusSparseXmx([qp, kp, vp, blocks.output(0)]).output(0)
    sparse.get_tensor().set_names({"attention_raw_bf16_f16_transport"})
    unpack = OtcCrossContextUnpack([sparse]).output(0)
    unpack.get_tensor().set_names({"attention_f32_exact_bf16"})
    output = ops.convert(ops.reshape(unpack, Q_SHAPE, False), BF16).output(0)
    output.get_tensor().set_names({"context_bf16"})
    return ov.Model([output], [q, k, v], "bf16_cross_bridge")


def build_roundtrip_model():
    """Exercise every finite BF16 word without attention overflow obscuring it."""
    q = ops.parameter(Q_SHAPE, BF16, name="q_bf16")
    decoded = ops.reshape(ops.convert(q, F32), Q_F32_SHAPE, False)
    packed = OtcCrossQPack([decoded.output(0)])
    unpacked = OtcCrossContextUnpack([packed.output(0)])
    output = ops.convert(ops.reshape(unpacked, Q_SHAPE, False), BF16).output(0)
    output.get_tensor().set_names({"context_bf16"})
    return ov.Model([output], [q], "bf16_bridge_roundtrip")


def bridge_inventory_bytes() -> dict[str, int]:
    """Conservative fixed-shape bridge buffers; no alias credit is assumed."""
    q = prod(Q_SHAPE) * 2
    kv = prod(KV_SHAPE) * 2
    packed_q = prod(Q_PACKED_SHAPE) * 2
    packed_kv = prod(KV_PACKED_SHAPE) * 2
    unpack_f32 = prod(Q_SHAPE) * 4
    inventory = {
        "bf16_input_q_bytes": q,
        "bf16_input_kv_each_bytes": kv,
        "f32_converted_inputs_bytes": 2 * (q + 2 * kv),
        "f32_input_reshape_allowance_bytes": 2 * (q + 2 * kv),
        "raw_f16_transport_q_bytes": packed_q,
        "raw_f16_transport_kv_each_bytes": packed_kv,
        "raw_attention_output_bytes": packed_q,
        "f32_unpack_output_bytes": unpack_f32,
        "f32_output_reshape_allowance_bytes": unpack_f32,
        "bf16_final_output_bytes": q,
        "block_ids_constant_bytes": prod(BLOCK_SHAPE) * 4,
    }
    # K and V each have the listed size; no reshape/allocator alias credit.
    inventory["conservative_bridge_bytes"] = sum(inventory.values()) + kv + packed_kv
    return inventory


__all__ = [
    "BF16", "F32", "F16_TRANSPORT", "I32", "Q_SHAPE", "KV_SHAPE", "Q_F32_SHAPE", "KV_F32_SHAPE",
    "Q_PACKED_SHAPE", "KV_PACKED_SHAPE", "BLOCK_SHAPE", "OtcCrossQPack",
    "OtcCrossKVPack", "OtcCrossContextUnpack", "bf16_bits_to_f32",
    "f32_bf16_decode_to_bits", "pack_q_bits", "pack_kv_bits",
    "unpack_context_bits", "bridge_roundtrip", "build_bridge_model", "build_roundtrip_model",
    "bridge_inventory_bytes",
]

