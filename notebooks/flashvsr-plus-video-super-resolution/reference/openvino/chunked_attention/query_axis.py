"""Conventional full-cache query-axis SDPA prototype.

This module is intentionally separate from ``builder.py``.  It exposes the
source-shaped Q/K/V tensors and a block mask, then puts only the SDPA sequence
axes into the conventional ``[1, H, sequence, D]`` arrangement.  C1 uses a
compact key-only mask broadcast over its block-token query axis; C4 retains
the reviewed dense repeated-query mask.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from builder import ChunkSpec, MASKED_LOGIT_LIMIT


FINITE_MASK_PENALTY = -32752.0
MAX_GRAPH_BYTES = 6 * 2**30


def memory_bound(spec: ChunkSpec, *, dtype_bytes: int = 2, batch_queries: bool = False) -> dict[str, int | float]:
    """Return the conservative no-Gather graph-visible inventory.

    The score and additive-mask terms are counted as F32, even though the
    public state and mask are F16, to cover a backend promotion/copy.
    """

    spec.validate()
    if dtype_bytes != 2:
        raise ValueError("query-axis production inventory is native F16")
    h, c, b, d, s = (
        spec.heads,
        spec.query_chunk_windows,
        spec.block_tokens,
        spec.depth,
        spec.key_windows,
    )
    query = h * c * b * d
    cache = 2 * h * s * b * d
    score = h * c * b * s * b
    mask = h * c * s * b if batch_queries else h * s * b if c == 1 else score
    values = {
        "query_chunk_bytes": query * dtype_bytes,
        "kv_cache_bytes": cache * dtype_bytes,
        "sdpa_score_bytes": score * 4,
        "sdpa_bool_mask_bytes": mask,
        "sdpa_additive_mask_bytes": mask * 4,
        "output_bytes": query * dtype_bytes,
    }
    values["bounded_total_bytes"] = sum(values.values())
    values["bounded_total_gib"] = values["bounded_total_bytes"] / float(2**30)
    return values


def validate_actual_logit_bound(actual_logit_bound: float, depth: int) -> float:
    """Validate the external Cauchy/logit admission used by finite masking."""

    bound = float(actual_logit_bound)
    if not math.isfinite(bound) or bound < 0 or bound > MASKED_LOGIT_LIMIT:
        raise ValueError(f"actual logit bound must be in [0,{MASKED_LOGIT_LIMIT}]")
    unscaled = bound * math.sqrt(depth)
    if not math.isfinite(unscaled) or unscaled >= np.finfo(np.float16).max:
        raise ValueError("admitted unscaled dot product is not safely finite in F16")
    penalty = np.float16(FINITE_MASK_PENALTY)
    extrema = np.asarray([float(penalty) - bound, float(penalty) + bound], dtype=np.float64)
    if not np.isfinite(extrema.astype(np.float16)).all():
        raise ValueError("finite F16 mask addition is not representable")
    gap = np.float32(float(penalty) + 2 * MASKED_LOGIT_LIMIT)
    if np.exp(gap) != np.float32(0.0):
        raise ValueError("finite forbidden-mask probability does not underflow")
    return bound


def _constant(ops: Any, value: Any, dtype: Any = None) -> Any:
    return ops.constant(np.asarray(value, dtype=dtype))


def build_query_axis_graph(
    spec: ChunkSpec,
    *,
    actual_logit_bound: float,
    state_dtype: str = "f16",
    batch_queries: bool = False,
) -> Any:
    """Build one conventional full-cache SDPA graph.

    Public parameters are Q_chunk [H,C,B,D], K_cache/V_cache [H,S,B,D], and
    blockMask [H,C,S] boolean.  The external logit admission is mandatory;
    graph construction rejects the conservative six-GiB inventory by default.
    """

    spec.validate()
    if state_dtype != "f16":
        raise ValueError("query-axis graph is qualified only for native F16")
    validate_actual_logit_bound(actual_logit_bound, spec.depth)
    inventory = memory_bound(spec,batch_queries=batch_queries)
    if int(inventory["bounded_total_bytes"]) > MAX_GRAPH_BYTES:
        raise MemoryError(
            f"query-axis graph inventory exceeds {MAX_GRAPH_BYTES} bytes: {inventory['bounded_total_bytes']}"
        )

    import openvino as ov
    from openvino import opset13 as ops

    h, c, b, d, s = (
        spec.heads,
        spec.query_chunk_windows,
        spec.block_tokens,
        spec.depth,
        spec.key_windows,
    )
    state_type = ov.Type.f16
    q = ops.parameter(spec.query_shape, state_type, name="Q_chunk")
    k = ops.parameter(spec.kv_shape, state_type, name="K_cache")
    v = ops.parameter(spec.kv_shape, state_type, name="V_cache")
    block_mask = ops.parameter((h, c, s), ov.Type.boolean, name="blockMask")

    q_flat = q if batch_queries else ops.reshape(q, _constant(ops, [1, h, c * b, d], np.int64), False, name="Q_query_axis")
    kv_shape=[h,1,s*b,d] if batch_queries else [1,h,s*b,d]
    k_flat = ops.reshape(k, _constant(ops, kv_shape, np.int64), False, name="K_full_cache_axis")
    v_flat = ops.reshape(v, _constant(ops, kv_shape, np.int64), False, name="V_full_cache_axis")

    nonempty = ops.reduce_logical_or(
        block_mask, _constant(ops, [-1], np.int64), True, name="row_has_valid_block"
    )
    first_block = np.zeros((1, 1, s), dtype=np.bool_)
    first_block[..., 0] = True
    first_block = ops.broadcast(
        _constant(ops, first_block), _constant(ops, [h, c, s], np.int64), name="empty_row_first_block"
    )
    safe_block_mask = ops.logical_or(
        block_mask,
        ops.logical_and(ops.logical_not(nonempty), first_block),
        name="safe_block_mask",
    )
    if batch_queries:
        # Original heads become independent SDPA batches; query windows become
        # heads sharing that batch's one K/V head. Each window keeps its own
        # key-only broadcast mask without repeating it over B query tokens.
        token_mask=ops.reshape(ops.broadcast(
            ops.reshape(safe_block_mask,_constant(ops,[h,c,s,1],np.int64),False),
            _constant(ops,[h,c,s,b],np.int64)),_constant(ops,[h,c,1,s*b],np.int64),False)
    elif c == 1:
        # SDPA broadcasts this compact key-axis mask over all B query tokens.
        # C1 therefore avoids materializing the reviewed C4 [B,S*B] mask.
        token_mask = ops.reshape(
            ops.broadcast(
                ops.reshape(safe_block_mask, _constant(ops, [1, h, 1, s, 1], np.int64), False),
                _constant(ops, [1, h, 1, s, b], np.int64),
                name="repeat_block_mask_key_tokens_c1",
            ),
            _constant(ops, [1, h, 1, s * b], np.int64),
            False,
            name="token_mask_c1_compact",
        )
    else:
        token_mask = ops.reshape(
            ops.broadcast(
                ops.reshape(safe_block_mask, _constant(ops, [1, h, c, 1, s, 1], np.int64), False),
                _constant(ops, [1, h, c, b, s, b], np.int64),
                name="repeat_block_mask_tokens",
            ),
            _constant(ops, [1, h, c * b, s * b], np.int64),
            False,
            name="token_mask",
        )
    additive_mask = ops.select(
        token_mask,
        _constant(ops, np.float16(0.0)),
        _constant(ops, np.float16(FINITE_MASK_PENALTY)),
        name="SDPA_finite_additive_mask",
    )
    attended = ops.scaled_dot_product_attention(
        q_flat,
        k_flat,
        v_flat,
        additive_mask,
        scale=np.asarray(1.0 / math.sqrt(d), dtype=np.float16),
        causal=False,
        name="SDPA_query_axis_full_cache",
    )
    raw = ops.reshape(
        attended, _constant(ops, [h, c, b, d], np.int64), False, name="attention_chunk_raw"
    )
    row_valid = ops.broadcast(
        ops.reshape(nonempty, _constant(ops, [h, c, 1, 1], np.int64), False),
        _constant(ops, [h, c, b, d], np.int64),
        name="row_has_valid_token",
    )
    zeros = ops.broadcast(
        _constant(ops, np.float16(0.0)), _constant(ops, [h, c, b, d], np.int64), name="empty_row_positive_zero"
    )
    output = ops.select(row_valid, raw, zeros, name="attention_chunk")
    output.output(0).get_tensor().set_names({"attention_chunk"})
    model = ov.Model(
        [output], [q, k, v, block_mask], f"query_axis_full_cache_h{h}_c{c}_s{s}_b{b}_d{d}"
    )
    model.validate_nodes_and_infer_types()
    return model


__all__ = [
    "FINITE_MASK_PENALTY",
    "MAX_GRAPH_BYTES",
    "build_query_axis_graph",
    "memory_bound",
    "validate_actual_logit_bound",
]
