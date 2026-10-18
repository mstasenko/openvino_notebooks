"""Bounded direct block-list XMX attention prototype, not an app backend yet."""
from pathlib import Path
from math import prod
import numpy as np
import openvino as ov

HERE = Path(__file__).resolve().parent


def admit_shapes(shapes, types):
    if len(shapes) != 4 or len(types) != 4:
        raise ValueError('Q/K/V and ordered block IDs required')
    q, k, v, ids = shapes
    canonical = len(k) == 4 and len(q) == 4 and k[2] == q[0]
    keys = k[0] if canonical else k[1] if len(k) > 1 else 0
    if (any(len(s) != 4 or any(type(n) != int or n <= 0 for n in s) for s in shapes)
            or not 1 <= q[0] <= 12 or q[1] not in (*range(1, 5), 8, 16) or q[2:] != [128, 128]
            or k != v or k != ([keys,128,q[0],128] if canonical else [q[0],keys,128,128])
            or ids[:3] != [q[0], q[1], 1] or ids[3] > keys
            or types != [ov.Type.f16] * 3 + [ov.Type.i32]):
        raise ValueError('Fixed 128-token/128-depth FP16 states and bounded I32 block lists required')
    if sum(prod(s) * size for s, size in zip(shapes, (2, 2, 2, 4))) + prod(q) * 2 > 6 << 30:
        raise MemoryError('Original six-GiB tensor inventory exceeded')


class OtcSparseXmx(ov.Op):
    def __init__(self, inputs):
        super().__init__(self, inputs)
        self.validate_and_infer_types()

    def validate_and_infer_types(self):
        shapes = [self.get_input_partial_shape(i) for i in range(self.get_input_size())]
        if any(not s.is_static for s in shapes):
            raise ValueError('Only bounded static attention shapes are admitted')
        shapes = [list(s.to_shape()) for s in shapes]
        admit_shapes(shapes, [self.get_input_element_type(i) for i in range(len(shapes))])
        self.set_output_type(0, ov.Type.f16, ov.PartialShape(shapes[0]))

    def clone_with_new_inputs(self, inputs):
        return OtcSparseXmx(inputs)

    def visit_attributes(self, visitor):
        return True


def build(heads, queries, keys, slots, *, canonical_kv=False):
    from openvino import opset13 as ops
    if type(canonical_kv) is not bool:raise ValueError('Explicit canonical K/V flag required')
    cache = [keys,128,heads,128] if canonical_kv else [heads,keys,128,128]
    shapes = [[heads, queries, 128, 128], cache, cache, [heads, queries, 1, slots]]
    types = [ov.Type.f16] * 3 + [ov.Type.i32]
    admit_shapes(shapes, types)
    parameters = [ops.parameter(shape, dtype, name=name) for shape, dtype, name in
                  zip(shapes, types, ('Q', 'K', 'V', 'blocks'))]
    for node in parameters:
        node.output(0).get_tensor().set_names({node.get_friendly_name()})
    output = OtcSparseXmx([p.output(0) for p in parameters])
    output.set_friendly_name('direct_sparse_xmx')
    output.output(0).get_tensor().set_names({'attention'})
    return ov.Model([output], parameters, 'direct_sparse_xmx')


def runtime_graph(spec, *, canonical_kv=False):
    """Use the existing serial replay controller's Q/K/V port names."""
    model = build(spec.heads, spec.query_chunk_windows, spec.key_windows, spec.key_windows,canonical_kv=canonical_kv)
    for old, new in (('Q', 'Q_chunk'), ('K', 'K_cache'), ('V', 'V_cache')):
        port = model.input(old)
        port.get_node().set_friendly_name(new)
        port.get_tensor().set_names({new})
    return model


def pack(mask, capacity=None):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 3 or not all(mask.shape):
        raise ValueError('Boolean [head,query,key] source mask required')
    # Candidate-only masks repeat across heads. Compare values, not cached
    # identity, so score-dependent masks retain their per-head selections.
    if mask.shape[0] > 1 and np.array_equal(
            mask[1:], np.broadcast_to(mask[:1], mask[1:].shape)):
        packed = pack(mask[:1], capacity)
        return np.broadcast_to(packed, (mask.shape[0], *packed.shape[1:])).copy(order='C')
    slots = max(1, int(mask.sum(-1).max()))
    if capacity is not None:
        if type(capacity) != int or not slots <= capacity <= mask.shape[-1]:
            raise ValueError('Capacity must retain every selected key and not exceed the cache')
        slots = capacity
    result = np.full((*mask.shape[:2], 1, slots), -1, np.int32)
    for head in range(mask.shape[0]):
        for query in range(mask.shape[1]):
            selected = np.flatnonzero(mask[head, query])
            result[head, query, 0, :len(selected)] = selected
    return result


def validate_inputs(q, k, v, ids):
    arrays = (q, k, v, ids)
    if any(not isinstance(a, np.ndarray) or not a.flags.c_contiguous for a in arrays):
        raise ValueError('Contiguous input arrays required')
    admit_shapes([list(a.shape) for a in arrays], [ov.Type(a.dtype) for a in arrays])
    keys = k.shape[0] if k.shape[2] == q.shape[0] else k.shape[1]
    if any(not np.isfinite(a).all() for a in (q, k, v)):
        raise ValueError('Finite attention inputs required')
    for row in ids.reshape(-1, ids.shape[-1]):
        count = int(np.count_nonzero(row >= 0))
        if (np.any(row[count:] != -1) or np.any(row[:count] >= keys)
                or np.any(row[:count] < 0) or np.any(np.diff(row[:count]) <= 0)):
            raise ValueError('IDs must increase in source order, followed only by -1 padding')


def fixtures():
    rng = np.random.default_rng(431)
    q = rng.normal(0, .2, (2, 3, 128, 128)).astype(np.float16)
    k = rng.normal(0, .2, (2, 5, 128, 128)).astype(np.float16)
    v = rng.normal(0, .2, k.shape).astype(np.float16)
    mask = np.array([[[0, 0, 0, 0, 0], [0, 1, 0, 0, 1], [1, 1, 0, 1, 0]],
                     [[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 1, 1, 0, 1]]], np.bool_)
    ids = pack(mask)
    yield 'A', q, k, v, ids, mask
    yield 'B', q, -k, np.ascontiguousarray(np.roll(v, 1, axis=1)), ids, mask
    yield 'A_after_B', q, k, v, ids, mask
    q = np.ones((1, 1, 128, 128), np.float16)
    k = np.ones((1, 2, 128, 128), np.float16); k[:, 0] *= -1
    v = np.zeros_like(k); v[:, 1] = 1
    mask = np.ones((1, 1, 2), np.bool_)
    yield 'global_softmax', q, k, v, pack(mask), mask
