#!/usr/bin/env python3
"""Bounded static-shape selected-key SDPA graphs.

The graph deliberately accepts one bounded query batch at a time.  K/V stay
in the graph as the complete static cache, while ``selectedKVblock`` chooses
the key blocks for each head/query row.  This keeps GatherND, SDPA scores, and
the validity mask proportional to ``query_chunk_windows * key_slots`` rather
than to all query windows times all key windows.

This is a small prototype for the already-reviewed block-mask ABI; it is not
an alternate runtime protocol or a replacement for the production builder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


DEFAULT_HEADS = 12
DEFAULT_DEPTH = 128
DEFAULT_BLOCK_TOKENS = 128
MAX_QUERY_CHUNK_WINDOWS = 4
MASKED_LOGIT_LIMIT = 4096


@dataclass(frozen=True)
class ChunkSpec:
    """Static bucket dimensions for one bounded query-chunk graph."""

    heads: int = DEFAULT_HEADS
    query_windows: int = 1
    key_windows: int = 1
    block_tokens: int = DEFAULT_BLOCK_TOKENS
    depth: int = DEFAULT_DEPTH
    query_chunk_windows: int = 1
    selected_slot_capacity: int = 1

    def validate(self) -> None:
        values = {
            "heads": self.heads,
            "query_windows": self.query_windows,
            "key_windows": self.key_windows,
            "block_tokens": self.block_tokens,
            "depth": self.depth,
            "query_chunk_windows": self.query_chunk_windows,
            "selected_slot_capacity": self.selected_slot_capacity,
        }
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError(f"all chunk dimensions must be positive: {values}")
        if self.query_chunk_windows > self.query_windows:
            raise ValueError("query_chunk_windows cannot exceed query_windows")
        if self.query_chunk_windows > MAX_QUERY_CHUNK_WINDOWS:
            raise ValueError(
                f"query_chunk_windows exceeds bounded prototype limit {MAX_QUERY_CHUNK_WINDOWS}"
            )

    @property
    def query_shape(self) -> tuple[int, int, int, int]:
        return (self.heads, self.query_chunk_windows, self.block_tokens, self.depth)

    @property
    def kv_shape(self) -> tuple[int, int, int, int]:
        return (self.heads, self.key_windows, self.block_tokens, self.depth)

    @property
    def selection_shape(self) -> tuple[int, int, int]:
        return (self.heads, self.query_chunk_windows, self.selected_slot_capacity)


def pack_selection(
    block_mask: np.ndarray,
    *,
    slot_capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack every allowed key block, padding only with invalid slots.

    The order is the source block order.  A row with no allowed blocks keeps a
    dummy ID of zero and all validity bits false; the graph makes that row
    safe for SDPA and zeros its output after the softmax.
    """

    mask = np.asarray(block_mask, dtype=np.bool_)
    if mask.ndim != 3:
        raise ValueError("block_mask must have shape [heads, query_windows, key_windows]")
    heads, query_windows, key_windows = mask.shape
    needed = max(1, int(mask.sum(axis=-1).max(initial=0)))
    slots = needed if slot_capacity is None else int(slot_capacity)
    if slots < needed or slots <= 0:
        raise ValueError(f"slot_capacity={slots} cannot represent max row count {needed}")
    selected = np.zeros((heads, query_windows, slots), dtype=np.int32)
    valid = np.zeros((heads, query_windows, slots), dtype=np.bool_)
    for head in range(heads):
        for query in range(query_windows):
            allowed = np.flatnonzero(mask[head, query])
            count = int(allowed.size)
            if count:
                selected[head, query, :count] = allowed.astype(np.int32)
                valid[head, query, :count] = True
    return selected, valid


def validate_selection(
    block_mask: np.ndarray,
    selected: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Check that selected IDs and validity bits losslessly encode the mask."""

    mask = np.asarray(block_mask, dtype=np.bool_)
    selected = np.asarray(selected)
    valid = np.asarray(valid, dtype=np.bool_)
    if selected.shape != valid.shape or selected.ndim != 3:
        raise ValueError("selected and valid must have the same rank-3 shape")
    if selected.shape[:2] != mask.shape[:2]:
        raise ValueError("selection query shape differs from mask")
    if np.any(selected < 0) or np.any(selected >= mask.shape[2]):
        raise ValueError("selected key block is outside the K/V cache")
    for head in range(mask.shape[0]):
        for query in range(mask.shape[1]):
            expected = np.flatnonzero(mask[head, query])
            actual = selected[head, query, valid[head, query]]
            if not np.array_equal(actual, expected):
                raise ValueError(
                    f"selection changed head={head}, query={query}: "
                    f"expected {expected.tolist()}, got {actual.tolist()}"
                )


def structural_memory_bound(spec: ChunkSpec, *, dtype_bytes: int = 4, gather_layout: str = "nd") -> dict[str, int | float]:
    """Return an upper-bound inventory without allocating or executing.

    The bound counts graph-visible tensors and the largest SDPA score/mask
    tensors.  It intentionally includes both K and V cache inputs once, but
    only one bounded query chunk and its selected slots.
    """

    spec.validate()
    if dtype_bytes not in (2, 4):
        raise ValueError("dtype_bytes must be 2 or 4")
    if gather_layout not in ("nd","flat","masked"):
        raise ValueError("Unknown cache gather layout")
    if gather_layout == "masked" and (dtype_bytes != 2 or spec.query_chunk_windows != 1):
        raise ValueError("Full-cache memory admission requires native F16 C1")
    h, c, b, d = spec.heads, spec.query_chunk_windows, spec.block_tokens, spec.depth
    k, slots = spec.key_windows, spec.selected_slot_capacity
    query_elements = h * c * b * d
    kv_elements = h * k * b * d
    selected_elements = 0 if gather_layout == "masked" else h * c * slots * b * d
    score_elements = h * c * b * (k if gather_layout == "masked" else slots) * b
    selection_elements = h * c * slots
    values: dict[str, int | float] = {
        "query_chunk_elements": query_elements,
        "kv_cache_elements": 2 * kv_elements,
        "gathered_kv_elements": 2 * selected_elements,
        "sdpa_score_elements": score_elements,
        "sdpa_bool_mask_elements": score_elements,
        "sdpa_additive_mask_elements": score_elements,
        "output_elements": query_elements,
        "selection_id_elements": selection_elements,
        "selection_valid_elements": selection_elements,
    }
    values.update(
        {
            "query_chunk_bytes": query_elements * dtype_bytes,
            "kv_cache_bytes": 2 * kv_elements * dtype_bytes,
            "gathered_kv_bytes": 2 * selected_elements * dtype_bytes,
            "sdpa_score_bytes": score_elements * 4,
            "sdpa_bool_mask_bytes": score_elements,
            "sdpa_additive_mask_bytes": score_elements * 4,
            "output_bytes": query_elements * dtype_bytes,
            "selection_id_bytes": selection_elements * 4,
            "selection_valid_bytes": selection_elements,
        }
    )
    counted = (
        "query_chunk_bytes",
        "kv_cache_bytes",
        "gathered_kv_bytes",
        "sdpa_score_bytes",
        "sdpa_bool_mask_bytes",
        "sdpa_additive_mask_bytes",
        "output_bytes",
        "selection_id_bytes",
        "selection_valid_bytes",
    )
    values["bounded_total_bytes"] = sum(int(values[key]) for key in counted)
    if gather_layout == "masked":
        # Equal and AND can each materialize [H,C,S,K] boolean tensors.
        values["selection_to_key_mask_bytes"] = 2*h*c*slots*k + h*c*k
        values["bounded_total_bytes"] += values["selection_to_key_mask_bytes"]
    # Reference only: the forbidden all-query/all-key GatherND shape for the
    # same cache.  This is a number, not an allocation.
    legacy_one_tensor = (
        spec.heads
        * spec.query_windows
        * spec.key_windows
        * spec.block_tokens
        * spec.depth
        * dtype_bytes
    )
    values["legacy_all_query_gathered_k_bytes"] = legacy_one_tensor
    values["legacy_all_query_gathered_v_bytes"] = legacy_one_tensor
    values["legacy_all_query_gathered_kv_bytes"] = 2 * legacy_one_tensor
    values["bounded_to_legacy_gather_ratio"] = (
        float(values["gathered_kv_bytes"])
        / float(values["legacy_all_query_gathered_kv_bytes"])
    )
    return values


def _constant(ops: Any, value: Any, dtype: Any = None) -> Any:
    return ops.constant(np.asarray(value, dtype=dtype))


def build_chunked_graph(spec: ChunkSpec, state_dtype: str = "f32", gather_layout: str = "nd") -> Any:
    """Build one static CPU SDPA graph for ``spec``.

    Inputs are ``Q_chunk`` [H,C,B,D], whole ``K_cache``/``V_cache``
    [H,K,B,D], and per-batch ``selectedKVblock``/``validSlot`` [H,C,S].
    GatherND is the only K/V selection; SDPA sees one global key sequence of
    S*B tokens per head/query row.  ``state_dtype`` controls only the Q/K/V,
    SDPA mask, scale, and output state type; the default f32 graph is kept
    unchanged for existing callers.
    """

    spec.validate()
    if state_dtype not in {"f32", "f16"}:
        raise ValueError(f"state_dtype must be 'f32' or 'f16', got {state_dtype!r}")
    if gather_layout not in {"nd","flat","masked"}:
        raise ValueError("Unknown cache gather layout")
    if gather_layout == "masked" and (state_dtype != "f16" or spec.query_chunk_windows != 1):
        raise ValueError("Full-cache masking is qualified only for native F16 C1")
    import openvino as ov
    from openvino import opset13 as ops

    h, c, b, d = spec.heads, spec.query_chunk_windows, spec.block_tokens, spec.depth
    k_blocks, slots = spec.key_windows, spec.selected_slot_capacity
    state_type = ov.Type.f16 if state_dtype == "f16" else ov.Type.f32
    state_numpy_dtype = np.float16 if state_dtype == "f16" else np.float32
    q = ops.parameter(spec.query_shape, state_type, name="Q_chunk")
    k = ops.parameter(spec.kv_shape, state_type, name="K_cache")
    v = ops.parameter(spec.kv_shape, state_type, name="V_cache")
    selected = ops.parameter(spec.selection_shape, ov.Type.i32, name="selectedKVblock")
    valid = ops.parameter(spec.selection_shape, ov.Type.boolean, name="validSlot")
    mask_valid = valid

    if gather_layout == "masked":
        if k_blocks > np.iinfo(np.int32).max:
            raise ValueError("Cache block ID would overflow int32")
        # OR packed slots into the original key order. In particular, an
        # invalid padding ID must not overwrite an allowed ID (as scatter
        # assignment would). This avoids copying each selected K/V block.
        axis = _constant(ops, [-1], np.int64)
        ids = _constant(ops, np.arange(k_blocks, dtype=np.int32).reshape(1,1,1,k_blocks))
        matches = ops.equal(ops.unsqueeze(selected,axis),ids)
        allowed = ops.logical_and(matches,ops.unsqueeze(valid,axis))
        mask_valid = ops.reduce_logical_or(allowed,_constant(ops,[2],np.int64),False)
        gathered_k, gathered_v = k, v
        slots = k_blocks
    elif gather_layout=="flat":
        if h*k_blocks > np.iinfo(np.int32).max:
            raise ValueError("Flattened cache index would overflow int32")
        # A head's block ID becomes a global row ID. Each row is one contiguous
        # B*D block, preserving per-head selection and the global softmax.
        offsets = (np.arange(h,dtype=np.int32)*k_blocks).reshape(h,1,1)
        indices = ops.add(selected,_constant(ops,offsets))
        axis = _constant(ops,np.int64(0))
        k_rows = ops.reshape(k,_constant(ops,[h*k_blocks,b*d],np.int64),False)
        v_rows = ops.reshape(v,_constant(ops,[h*k_blocks,b*d],np.int64),False)
        gathered_k = ops.gather(k_rows,indices,axis,name="Gather_flat_K_cache")
        gathered_v = ops.gather(v_rows,indices,axis,name="Gather_flat_V_cache")
    else:
        selected_indices = ops.reshape(
            selected,
            _constant(ops, [h, c, slots, 1], np.int64),
            False,
            name="selectedKVblock_indices",
        )
        gathered_k = ops.gather_nd(k, selected_indices, batch_dims=1, name="GatherND_K_cache")
        gathered_v = ops.gather_nd(v, selected_indices, batch_dims=1, name="GatherND_V_cache")
    # Keep head/query in the head axis (qualified layout C); the GPU plugin's
    # alternate batch-axis SDPA layout previously returned the wrong shape.
    q_flat = ops.reshape(q, _constant(ops, [1, h * c, b, d], np.int64), False, name="Q_chunk_batch")
    k_flat = ops.reshape(
        gathered_k,
        _constant(ops, [1, h * c, slots * b, d], np.int64),
        False,
        name="K_selected_chunk_batch",
    )
    v_flat = ops.reshape(
        gathered_v,
        _constant(ops, [1, h * c, slots * b, d], np.int64),
        False,
        name="V_selected_chunk_batch",
    )

    nonempty = ops.reduce_logical_or(
        mask_valid,
        _constant(ops, [-1], np.int64),
        True,
        name="row_has_valid_slot",
    )
    first_slot = np.zeros((1, 1, slots), dtype=np.bool_)
    first_slot[..., 0] = True
    first_slot = ops.broadcast(
        ops.constant(first_slot), _constant(ops, [h, c, slots], np.int64), name="empty_row_dummy_slot"
    )
    safe_valid = ops.logical_or(
        mask_valid,
        ops.logical_and(ops.logical_not(nonempty), first_slot),
        name="safe_valid_slot",
    )
    safe_valid_tokens = ops.broadcast(
        ops.reshape(safe_valid, _constant(ops, [h, c, slots, 1], np.int64), False),
        _constant(ops, [h, c, slots, b], np.int64),
        name="safe_valid_tokens",
    )
    safe_valid_tokens = ops.reshape(
        safe_valid_tokens,
        _constant(ops, [1, h * c, 1, slots * b], np.int64),
        False,
        name="SDPA_valid_tokens",
    )
    additive_mask = ops.select(
        safe_valid_tokens,
        _constant(ops, np.asarray(0.0, dtype=state_numpy_dtype)),
        # A leading all--inf tile produces NaNs in the GPU streaming softmax.
        # With caller-admitted |logit|<=4096, this finite F16 penalty cannot
        # overflow and forbidden probabilities underflow exactly to zero.
        _constant(ops, np.asarray(-32752.0 if gather_layout == "masked" else -np.inf, dtype=state_numpy_dtype)),
        name="SDPA_additive_mask",
    )
    attended = ops.scaled_dot_product_attention(
        q_flat,
        k_flat,
        v_flat,
        additive_mask,
        scale=np.asarray(1.0 / math.sqrt(d), dtype=state_numpy_dtype),
        causal=False,
        name="SDPA_selected_global_softmax",
    )
    output = ops.reshape(
        attended,
        _constant(ops, [h, c, b, d], np.int64),
        False,
        name="attention_chunk_raw",
    )
    nonempty_output = ops.broadcast(
        ops.reshape(nonempty, _constant(ops, [h, c, 1, 1], np.int64), False),
        _constant(ops, [h, c, b, d], np.int64),
        name="row_has_valid_token",
    )
    output = ops.multiply(
        output,
        ops.convert(nonempty_output, state_type),
        name="zero_empty_rows",
    )
    output.output(0).get_tensor().set_names({"attention_chunk"})
    model = ov.Model(
        [output],
        [q, k, v, selected, valid],
        f"chunked_selected_sdpa_h{h}_c{c}_k{k_blocks}_s{slots}_b{b}_d{d}",
    )
    model.validate_nodes_and_infer_types()
    return model

