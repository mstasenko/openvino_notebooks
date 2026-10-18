"""NumPy adapter for the pinned FlashVSR+ streamed block layout.

The Plus stream always supplies the current four-frame chunk (two partitioned
temporal windows) and any retained, suffix-ordered KV windows.  The mask is a
spatial candidate relation followed by the source's grouped, strict top-k
selection; it is deliberately not a token-level causal mask.
"""

from dataclasses import dataclass
import math

import numpy as np


BLOCK_TOKENS = 2 * 8 * 8
HEAD_DIM = 128
DEFAULT_TOPK_RATIO = 2.5
DEFAULT_TEMPORAL_BLOCK_SIZE = 2


@dataclass(frozen=True)
class StreamLayout:
    """Batch-one layout in the shape consumed by the OpenVINO helpers."""

    block_mask: np.ndarray  # [heads, query_blocks, key_blocks]
    cache_indices: np.ndarray  # [heads, retained_key_blocks], source order
    block_token_size: int = BLOCK_TOKENS


def _pooled(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"{name} supports batch one only")
        value = value[0]
    if value.ndim != 3 or value.shape[2] != HEAD_DIM:
        raise ValueError(f"{name} must have shape [heads, blocks, {HEAD_DIM}]")
    if value.shape[0] <= 0 or value.shape[1] <= 0 or value.dtype != np.float32:
        raise ValueError(f"{name} must contain finite FP32 pooled heads and blocks")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite pooled values")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _candidate_mask(
    query_blocks: int,
    key_blocks: int,
    rows: int,
    cols: int,
    local_range: int | None,
) -> np.ndarray:
    spatial = rows * cols
    if local_range is None:
        return np.ones((query_blocks, key_blocks), dtype=np.bool_)
    q_spatial = np.arange(query_blocks, dtype=np.int64) % spatial
    k_spatial = np.arange(key_blocks, dtype=np.int64) % spatial
    q_row, q_col = divmod(q_spatial, cols)
    k_row, k_col = divmod(k_spatial, cols)
    half = local_range // 2
    row_start = q_row - half
    col_start = q_col - half
    return (
        (k_row[None, :] >= row_start[:, None])
        & (k_row[None, :] <= row_start[:, None] + local_range - 1)
        & (k_col[None, :] >= col_start[:, None])
        & (k_col[None, :] <= col_start[:, None] + local_range - 1)
    )


def _strict_grouped_topk(
    scores: np.ndarray,
    candidates: np.ndarray,
    temporal_windows: int,
    temporal_block_size: int,
    topk: int,
) -> np.ndarray:
    """Apply Plus's strict threshold per temporal block group."""
    heads, query_blocks, key_blocks = scores.shape
    grouped_temporal = temporal_windows // temporal_block_size
    spatial = query_blocks // grouped_temporal
    masked = np.where(candidates[None, :, :], scores, np.float32(-np.inf))
    grouped = masked.reshape(heads, grouped_temporal, spatial * key_blocks)
    grouped_candidates = np.broadcast_to(
        candidates.reshape(grouped_temporal, spatial * key_blocks), grouped.shape
    )
    count = min(grouped.shape[-1] - 1, max(int(topk), 0))
    kth = grouped.shape[-1] - count - 1
    thresholds = np.partition(grouped, kth, axis=-1)[..., kth, None]
    selected = (grouped > thresholds) & grouped_candidates
    return selected.reshape(heads, query_blocks, key_blocks)


def _fill_empty_rows(mask: np.ndarray, candidates: np.ndarray) -> None:
    empty = ~mask.any(axis=-1)
    if not empty.any():
        return
    first = candidates.argmax(axis=-1)
    head, query = np.nonzero(empty)
    mask[head, query, first[query]] = True


def _score_bound(q: np.ndarray, k: np.ndarray) -> float:
    """Return a finite-input FP32 dot-product bound in Python float arithmetic."""
    if q.size == 0 or k.size == 0:
        return math.inf
    return (
        float(np.max(np.abs(q)))
        * float(np.max(np.abs(k)))
        * float(HEAD_DIM)
    )


def _score_bound_safe(q: np.ndarray, k: np.ndarray) -> bool:
    """Conservative bound used only to prove the all-candidates shortcut."""
    score_bound = _score_bound(q, k)
    return bool(np.isfinite(score_bound) and
                score_bound <= float(np.finfo(np.float32).max) / 4.0)


def _all_candidates_shortcut_safe(
    q: np.ndarray,
    k: np.ndarray,
    candidates: np.ndarray,
    temporal_windows: int,
    temporal_block_size: int,
    topk: int,
    *,
    score_bound_safe: bool | None = None,
) -> bool:
    """Return whether strict top-k must select every finite candidate.

    The grouped selector ranks ``spatial * key_blocks`` entries and uses the
    (count + 1)-th largest value as a strict threshold.  If every group has
    fewer candidates than ``count + 1``, the selector's second candidate mask
    replaces all non-candidates with ``-inf`` before thresholding.  Therefore
    even a finite candidate whose softmax probability underflowed to zero is
    still selected, and the exact result is the candidate mask.  The FP32
    score-overflow guard remains necessary because non-finite candidate logits
    would not satisfy ``value > -inf``.
    """
    grouped_temporal = temporal_windows // temporal_block_size
    query_blocks, key_blocks = candidates.shape
    if grouped_temporal <= 0 or query_blocks % grouped_temporal:
        return False
    group_rows = query_blocks // grouped_temporal
    group_width = group_rows * key_blocks
    count = min(group_width - 1, max(int(topk), 0))
    if group_width <= 0 or count < 0:
        return False
    candidate_counts = [
        int(candidates[start : start + group_rows].sum())
        for start in range(0, query_blocks, group_rows)
    ]
    if not all(candidate_count < group_width and count >= candidate_count for candidate_count in candidate_counts):
        return False
    if score_bound_safe is None:
        score_bound_safe = _score_bound_safe(q, k)
    if not score_bound_safe:
        return False
    if (q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]
            or q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM
            or not np.isfinite(q).all() or not np.isfinite(k).all()):
        return False
    return True


def build_stream_sparse_layout(
    q_pooled: np.ndarray,
    k_pooled: np.ndarray,
    *,
    temporal_windows: int,
    spatial_grid: tuple[int, int],
    cache_temporal_groups: int,
    local_range: int | None,
    topk: int | None = None,
    topk_ratio: float = DEFAULT_TOPK_RATIO,
    temporal_block_size: int = DEFAULT_TEMPORAL_BLOCK_SIZE,
    is_sparse: bool = True,
) -> StreamLayout:
    """Build a FlashVSR+ stream mask and suffix KV-retention indices.

    ``q_pooled``/``k_pooled`` are batch-one FP32 pooled arrays in source order
    (temporal-window-major, then raster spatial order).  Pooling is intentionally
    outside this function so remote Q/K need not be read back.  ``topk`` is
    the source count; when omitted it is ``int(S*S*topk_ratio)-1``.  The
    returned arrays omit the source's leading batch dimension for direct use
    by the NumPy/OpenVINO runtime.
    """
    q = _pooled(q_pooled, "q_pooled")
    k = _pooled(k_pooled, "k_pooled")
    if q.shape[0] != k.shape[0]:
        raise ValueError("stream q/k head counts must match")
    if not isinstance(spatial_grid, (tuple, list)) or len(spatial_grid) != 2:
        raise ValueError("spatial_grid must be (rows, cols)")
    rows = _positive_int(spatial_grid[0], "spatial rows")
    cols = _positive_int(spatial_grid[1], "spatial cols")
    temporal_windows = _positive_int(temporal_windows, "temporal_windows")
    temporal_block_size = _positive_int(
        temporal_block_size, "temporal_block_size"
    )
    cache_temporal_groups = (
        0
        if isinstance(cache_temporal_groups, (int, np.integer))
        and not isinstance(cache_temporal_groups, bool)
        and int(cache_temporal_groups) == 0
        else _positive_int(cache_temporal_groups, "cache_temporal_groups")
    )
    if temporal_windows % temporal_block_size:
        raise ValueError("temporal block size must divide temporal windows")
    spatial_blocks = rows * cols
    query_blocks, key_blocks = q.shape[1], k.shape[1]
    if query_blocks % temporal_windows or query_blocks // temporal_windows != spatial_blocks:
        raise ValueError("query layout does not match temporal windows and spatial grid")
    if key_blocks % spatial_blocks:
        raise ValueError("key cache must contain complete spatial grids")
    if local_range is not None:
        local_range = _positive_int(local_range, "local_range")
    if topk is None:
        if not isinstance(topk_ratio, (int, float, np.integer, np.floating)):
            raise ValueError("topk_ratio must be numeric")
        topk_ratio = float(topk_ratio)
        if not math.isfinite(topk_ratio) or topk_ratio < 0:
            raise ValueError("topk_ratio must be finite and nonnegative")
        topk = max(int(spatial_blocks * spatial_blocks * topk_ratio) - 1, 0)
    elif isinstance(topk, bool) or not isinstance(topk, (int, np.integer)) or int(topk) < 0:
        raise ValueError("topk must be a nonnegative integer")
    else:
        topk = int(topk)

    candidates = _candidate_mask(
        query_blocks, key_blocks, rows, cols, local_range
    )
    if is_sparse:
        score_bound = _score_bound(q, k)
        shortcut_score_bound_safe = bool(
            np.isfinite(score_bound)
            and score_bound <= float(np.finfo(np.float32).max) / 4.0
        )
        if _all_candidates_shortcut_safe(
            q, k, candidates, temporal_windows, temporal_block_size, topk,
            score_bound_safe=shortcut_score_bound_safe,
        ):
            block_mask = np.broadcast_to(
                candidates[None, :, :], (q.shape[0], query_blocks, key_blocks)
            ).copy()
        else:
            # The fallback performs the source FP32 dot products.  Permit
            # finite bounds up to half the FP32 maximum: this admits the
            # source-valid 1e18 witness while leaving a factor-two margin
            # against accumulator overflow.  The stricter quarter bound
            # above is only needed by the shortcut's non-scoring proof.
            if (not np.isfinite(score_bound)
                    or score_bound > float(np.finfo(np.float32).max) / 2.0):
                raise ValueError(
                    "finite pooled q/k values exceed the safe FP32 score bound"
                )
            scores = np.einsum("hqd,hkd->hqk", q, k)
            scores = scores / np.float32(math.sqrt(HEAD_DIM))
            scores = np.where(
                candidates[None, :, :], scores, np.float32(-np.inf)
            )
            maxima = np.max(scores, axis=-1, keepdims=True)
            shifted = scores - maxima
            logsumexp = maxima + np.log(
                np.exp(shifted).sum(axis=-1, keepdims=True, dtype=np.float32)
            )
            weights = np.exp(scores - logsumexp)
            block_mask = _strict_grouped_topk(
                weights,
                candidates,
                temporal_windows,
                temporal_block_size,
                topk,
            )
    else:
        block_mask = np.broadcast_to(
            candidates[None, :, :], (q.shape[0], query_blocks, key_blocks)
        ).copy()
    _fill_empty_rows(block_mask, candidates)

    keep = min(max(cache_temporal_groups, 1) * spatial_blocks, key_blocks)
    suffix = np.arange(key_blocks - keep, key_blocks, dtype=np.int32)
    cache_indices = np.broadcast_to(suffix, (q.shape[0], keep)).copy()
    return StreamLayout(
        np.ascontiguousarray(block_mask, dtype=np.bool_),
        cache_indices,
    )


__all__ = [
    "BLOCK_TOKENS",
    "DEFAULT_TEMPORAL_BLOCK_SIZE",
    "DEFAULT_TOPK_RATIO",
    "HEAD_DIM",
    "StreamLayout",
    "build_stream_sparse_layout",
]
