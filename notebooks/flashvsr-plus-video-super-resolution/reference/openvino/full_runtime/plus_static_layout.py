"""Narrow, source-compatible static-mask candidate for FlashVSR+ research."""

from __future__ import annotations

import math
from numbers import Integral

import torch

from models.attention_backends.sparse import (
    StreamSparseAttentionLayout,
    _stream_candidate_mask,
    build_stream_sparse_layout as _source_build_stream_sparse_layout,
)


# Only geometries audited on CPU; other valid source geometries use the source.
_REVIEWED_GRIDS = frozenset(((6, 10), (10, 6), (17, 30), (30, 17)))
MAX_POOLED_LOGIT_BOUND = 32.0


def _positive(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0


def _extent_sum(size: int, local_range: int) -> int:
    half = local_range // 2
    return sum(
        max(0, min(size - 1, index - half + local_range - 1) - max(0, index - half) + 1)
        for index in range(size)
    )


def _candidate_count(rows: int, cols: int, local_range: int, group_rows: int, spatial: int, key_groups: int) -> int | None:
    if group_rows % spatial:
        return None
    return (group_rows // spatial) * key_groups * _extent_sum(rows, local_range) * _extent_sum(cols, local_range)


def _static_safe(q: torch.Tensor, k: torch.Tensor, *, temporal_windows: int, spatial_grid: tuple[int, int] | list[int], topk: int,
                 temporal_block_size: int, local_range: int | None, is_sparse: bool) -> bool:
    """Prove threshold ``-inf`` per group before checking tensor values."""
    if (not is_sparse or local_range is None
            or not isinstance(spatial_grid, (tuple, list))
            or len(spatial_grid) != 2):
        return False
    rows, cols = spatial_grid
    if not (_positive(rows) and _positive(cols)):
        return False
    rows, cols = int(rows), int(cols)
    if ((rows, cols) not in _REVIEWED_GRIDS or not _positive(local_range)
            or not _positive(temporal_windows)
            or not _positive(temporal_block_size)):
        return False
    if (temporal_windows % temporal_block_size
            or not isinstance(topk, Integral) or isinstance(topk, bool)
            or int(topk) < 0):
        return False
    if (q.ndim != 5 or k.ndim != 5 or q.device != k.device or q.shape[:2] != k.shape[:2]
            or q.shape[3:] != k.shape[3:]):
        return False
    if any(int(size) <= 0 for size in (*q.shape, *k.shape)):
        return False
    spatial = rows * cols
    query_blocks, key_blocks = q.shape[2], k.shape[2]
    if query_blocks != temporal_windows * spatial or key_blocks % spatial:
        return False
    grouped = temporal_windows // temporal_block_size
    if grouped <= 0 or query_blocks % grouped:
        return False
    group_rows = query_blocks // grouped
    group_width = group_rows * key_blocks
    effective_topk = min(group_width - 1, int(topk))
    candidates = _candidate_count(rows, cols, int(local_range), group_rows,
                                  spatial, key_blocks // spatial)
    return (candidates is not None and candidates < group_width
            and candidates <= effective_topk)


def _bounded_values(q: torch.Tensor, k: torch.Tensor) -> bool:
    # Geometric saturation is insufficient for NaNs, overflowing FP32 draft
    # scores, or softmax underflow. Compute the same FP32 block pooling as the
    # source, then a conservative per-head Cauchy bound in FP64. The final
    # scalar synchronization preserves the exact source fallback for unsafe
    # values without copying Q/K to the host.
    if not q.is_floating_point() or not k.is_floating_point():
        return False
    try:
        q_min, q_max = torch.aminmax(q)
        k_min, k_max = torch.aminmax(k)
        q_pooled = q.mean(dim=3, dtype=torch.float32)
        k_pooled = k.mean(dim=3, dtype=torch.float32)
        q_norm = torch.sqrt(torch.sum(q_pooled.to(torch.float64) ** 2, dim=-1))
        k_norm = torch.sqrt(torch.sum(k_pooled.to(torch.float64) ** 2, dim=-1))
        # Max over batch/blocks per head, then pair corresponding heads as in
        # the source's per-head pooled-score bound.
        q_head = q_norm.amax(dim=(0, 2))
        k_head = k_norm.amax(dim=(0, 2))
        pooled_bound = torch.max(q_head * k_head) / math.sqrt(q.shape[-1])
        safe = ((q_min >= -8192) & (q_max <= 8192)
                & (k_min >= -8192) & (k_max <= 8192)
                & torch.isfinite(pooled_bound)
                & (pooled_bound <= MAX_POOLED_LOGIT_BOUND))
        return bool(safe.item())
    except (RuntimeError, TypeError, ValueError):
        # Unsupported device/dtype reductions must use the pinned source path.
        return False


def build_stream_sparse_layout(
    q_windows: torch.Tensor,
    k_windows: torch.Tensor,
    *,
    temporal_windows: int,
    spatial_grid: tuple[int, int] | list[int],
    topk: int,
    cache_temporal_groups: int,
    temporal_block_size: int,
    local_range: int | None,
    is_sparse: bool = True,
    static_fastpath: bool = False,
) -> StreamSparseAttentionLayout:
    """Call the pinned source builder, or use its proven candidate mask."""
    args = dict(
        temporal_windows=temporal_windows, spatial_grid=spatial_grid, topk=topk,
        cache_temporal_groups=cache_temporal_groups,
        temporal_block_size=temporal_block_size, local_range=local_range,
        is_sparse=is_sparse,
    )
    if not static_fastpath or not _static_safe(
        q_windows, k_windows, temporal_windows=temporal_windows,
        spatial_grid=spatial_grid, topk=topk,
        temporal_block_size=temporal_block_size, local_range=local_range,
        is_sparse=is_sparse,
    ) or not _bounded_values(q_windows, k_windows):
        return _source_build_stream_sparse_layout(q_windows, k_windows, **args)

    rows, cols = (int(value) for value in spatial_grid)
    spatial = rows * cols
    batch, heads, query_blocks, block_tokens, _ = q_windows.shape
    key_blocks = k_windows.shape[2]
    device_index = q_windows.device.index if q_windows.device.index is not None else 0
    candidates = _stream_candidate_mask(
        query_blocks, key_blocks, rows, cols, int(local_range),
        q_windows.device.type, device_index,
    )
    block_mask = candidates.view(1, 1, query_blocks, key_blocks).expand(batch, heads, -1, -1).contiguous()
    keep = min(max(int(cache_temporal_groups), 1) * spatial, key_blocks)
    cache_indices = torch.arange(key_blocks - keep, key_blocks, device=k_windows.device).view(1, 1, keep).expand(batch, heads, keep)
    return StreamSparseAttentionLayout(block_mask, cache_indices, block_tokens)
