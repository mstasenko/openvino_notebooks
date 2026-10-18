"""BF16 direct block-list XMX attention component.

The port keeps the original 128-token/128-depth ABI.  BF16 tensors are
represented on the host by their exact 16-bit storage bits (``uint16``).
CustomGPUPrimitive cannot admit BF16 layouts in the pinned runtime. Its ports
use F16 views solely as two-byte transport; the kernel interprets every operand
as BF16 and performs BF16 XMX operations. Do not numerically cast these views.
"""

from math import prod
from pathlib import Path

import numpy as np
import openvino as ov

from sparse_xmx import pack


HERE = Path(__file__).resolve().parent
BF16 = ov.Type.bf16
STORAGE = ov.Type.f16
I32 = ov.Type.i32
_BF16_TYPES = [STORAGE, STORAGE, STORAGE, I32]


def float32_to_bf16_bits(value: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 storage bits using round-to-nearest-even.

    The attention ABI admits finite BF16 values only; NaN payload behavior is
    therefore outside the input/output contract (and is rejected by
    :func:`validate_inputs`).
    """
    value = np.asarray(value)
    if value.dtype != np.float32:
        raise TypeError("float32 input required for BF16 conversion")
    bits = np.ascontiguousarray(value).view(np.uint32)
    upper = bits >> 16
    lower = bits & np.uint32(0xFFFF)
    increment = (lower > 0x8000) | (
        (lower == 0x8000) & ((upper & 1) != 0)
    )
    return (upper + increment.astype(np.uint32)).astype(np.uint16)


def bf16_bits_to_float32(value: np.ndarray) -> np.ndarray:
    """Decode exact BF16 storage bits without changing NaN/sign payload bits."""
    value = np.asarray(value)
    if value.dtype != np.uint16:
        raise TypeError("uint16 BF16 storage bits required")
    return (np.ascontiguousarray(value).astype(np.uint32) << 16).view(np.float32)


def admit_shapes(shapes, types):
    """Validate the original fixed block-list ABI with BF16 operands."""
    if len(shapes) != 4 or len(types) != 4:
        raise ValueError("Q/K/V and ordered block IDs required")
    q, k, v, ids = shapes
    canonical = len(k) == 4 and len(q) == 4 and k[2] == q[0]
    keys = k[0] if canonical else k[1] if len(k) > 1 else 0
    if (
        any(
            len(s) != 4 or any(type(n) is not int or n <= 0 for n in s)
            for s in shapes
        )
        or not 1 <= q[0] <= 12
        or q[1] not in (*range(1, 5), 8, 16)
        or q[2:] != [128, 128]
        or k != v
        or k != ([keys, 128, q[0], 128] if canonical else [q[0], keys, 128, 128])
        or ids[:3] != [q[0], q[1], 1]
        or ids[3] > keys
        or list(types) != _BF16_TYPES
    ):
        raise ValueError(
            "Fixed 128-token/128-depth BF16 states and bounded I32 block lists required"
        )
    if sum(prod(s) * size for s, size in zip(shapes, (2, 2, 2, 4))) + prod(q) * 2 > 6 << 30:
        raise MemoryError("Original six-GiB tensor inventory exceeded")


class OtcPlusSparseXmx(ov.Op):
    def __init__(self, inputs):
        super().__init__(self, inputs)
        self.validate_and_infer_types()

    def validate_and_infer_types(self):
        shapes = [self.get_input_partial_shape(i) for i in range(self.get_input_size())]
        if any(not s.is_static for s in shapes):
            raise ValueError("Only bounded static attention shapes are admitted")
        shapes = [list(s.to_shape()) for s in shapes]
        admit_shapes(shapes, [self.get_input_element_type(i) for i in range(len(shapes))])
        self.set_output_type(0, STORAGE, ov.PartialShape(shapes[0]))

    def clone_with_new_inputs(self, inputs):
        return OtcPlusSparseXmx(inputs)

    def visit_attributes(self, visitor):
        return True


def _admit_split_shapes(shapes, types):
    admit_shapes(shapes, types)
    q, k, _, ids = shapes
    if (q != [12, 16, 128, 128] or k != [k[0], 128, 12, 128]
            or k[0] > 2550 or ids[3] > 2550):
        raise ValueError("Canonical C16 split-key ABI with at most 2550 keys/slots required")


class OtcPlusSplitKeyPartial(ov.Op):
    """Four-way FP32 partial softmax accumulators for the split-key kernel."""

    def __init__(self, inputs):
        super().__init__(self, inputs)
        self.validate_and_infer_types()

    def validate_and_infer_types(self):
        if self.get_input_size() != 4:
            raise ValueError("Split-key partial requires Q/K/V and ordered IDs")
        shapes = [self.get_input_partial_shape(i) for i in range(4)]
        if any(not shape.is_static for shape in shapes):
            raise ValueError("Only bounded static split-key shapes are admitted")
        _admit_split_shapes([list(shape.to_shape()) for shape in shapes],
                            [self.get_input_element_type(i) for i in range(4)])
        self.set_output_type(0, ov.Type.f32, ov.PartialShape([48, 16, 128, 130]))

    def clone_with_new_inputs(self, inputs):
        return OtcPlusSplitKeyPartial(inputs)

    def visit_attributes(self, visitor):
        return True


class OtcPlusSplitKeyMerge(ov.Op):
    """Merge FP32 partial softmax accumulators into the original F16 ABI."""

    def __init__(self, inputs):
        super().__init__(self, inputs)
        self.validate_and_infer_types()

    def validate_and_infer_types(self):
        if self.get_input_size() != 1:
            raise ValueError("Split-key merge requires one partial input")
        partial = self.get_input_partial_shape(0)
        if (
            self.get_input_element_type(0) != ov.Type.f32
            or not partial.is_static
            or list(partial.to_shape()) != [48, 16, 128, 130]
        ):
            raise ValueError("Split-key partial accumulator ABI required")
        self.set_output_type(0, STORAGE, ov.PartialShape([12, 16, 128, 128]))

    def clone_with_new_inputs(self, inputs):
        return OtcPlusSplitKeyMerge(inputs)

    def visit_attributes(self, visitor):
        return True


def build(heads, queries, keys, slots, *, canonical_kv=False, split_key=False):
    from openvino import opset13 as ops

    if type(canonical_kv) is not bool:
        raise ValueError("Explicit canonical K/V flag required")
    if type(split_key) is not bool:
        raise ValueError("Explicit split-key flag required")
    if split_key and not canonical_kv:
        raise ValueError("Split-key mode requires canonical K/V layout")
    cache = [keys, 128, heads, 128] if canonical_kv else [heads, keys, 128, 128]
    shapes = [[heads, queries, 128, 128], cache, cache, [heads, queries, 1, slots]]
    (_admit_split_shapes if split_key else admit_shapes)(shapes, _BF16_TYPES)
    parameters = [
        ops.parameter(shape, dtype, name=name)
        for shape, dtype, name in zip(shapes, _BF16_TYPES, ("Q", "K", "V", "blocks"))
    ]
    for node in parameters:
        node.output(0).get_tensor().set_names({node.get_friendly_name()})
    if split_key:
        partial = OtcPlusSplitKeyPartial([p.output(0) for p in parameters]).output(0)
        partial.get_tensor().set_names({"split_partial"})
        output = OtcPlusSplitKeyMerge([partial])
    else:
        output = OtcPlusSparseXmx([p.output(0) for p in parameters])
        output.set_friendly_name("direct_plus_sparse_xmx")
    output.output(0).get_tensor().set_names({"attention"})
    return ov.Model([output], parameters,
                    "split_key_pipeline" if split_key else "direct_plus_sparse_xmx")


def runtime_graph(spec, *, canonical_kv=False):
    """Use the existing replay controller's Q/K/V/block port names."""
    model = build(
        spec.heads,
        spec.query_chunk_windows,
        spec.key_windows,
        spec.key_windows,
        canonical_kv=canonical_kv,
    )
    for old, new in (("Q", "Q_chunk"), ("K", "K_cache"), ("V", "V_cache")):
        port = model.input(old)
        port.get_node().set_friendly_name(new)
        port.get_tensor().set_names({new})
    return model


def validate_inputs(q, k, v, ids):
    """Validate raw BF16 bits and the unchanged ordered-ID port ABI."""
    arrays = (q, k, v, ids)
    if any(not isinstance(a, np.ndarray) or not a.flags.c_contiguous for a in arrays):
        raise ValueError("Contiguous input arrays required")
    if any(a.dtype != np.uint16 for a in (q, k, v)) or ids.dtype != np.int32:
        raise TypeError("Q/K/V must be uint16 BF16 bits and IDs must be int32")
    admit_shapes([list(a.shape) for a in arrays], _BF16_TYPES)
    if any(not np.isfinite(bf16_bits_to_float32(a)).all() for a in (q, k, v)):
        raise ValueError("Finite BF16 attention inputs required")
    keys = k.shape[0] if k.shape[2] == q.shape[0] else k.shape[1]
    for row in ids.reshape(-1, ids.shape[-1]):
        count = int(np.count_nonzero(row >= 0))
        if (
            np.any(row[count:] != -1)
            or np.any(row[:count] >= keys)
            or np.any(row[:count] < 0)
            or np.any(np.diff(row[:count]) <= 0)
        ):
            raise ValueError("IDs must increase in source order, followed only by -1 padding")


__all__ = [
    "BF16",
    "STORAGE",
    "HERE",
    "OtcPlusSparseXmx",
    "admit_shapes",
    "bf16_bits_to_float32",
    "build",
    "float32_to_bf16_bits",
    "pack",
    "runtime_graph",
    "validate_inputs",
]
