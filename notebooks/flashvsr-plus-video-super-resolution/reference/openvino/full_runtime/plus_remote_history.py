"""Exact raw-BF16 compact history for the resident remote attention path.

The caller must consume :meth:`view` synchronously (the remote prefix upload)
before calling :meth:`update`; update mutates the backing arrays in place.
Returned views are therefore invalid after the next update or ``release``.
Only the small pooled merge is allocated; K/V are never concatenated.
"""

from __future__ import annotations

import numpy as np

from plus_storage import is_finite_bf16

_BLOCK = 128
_CHANNELS = 1536
_HEADS = 12


class CompactHistory:
    """One layer's fixed-capacity raw-BF16 K/V and FP32 pooled history.

    ``backing=(k, v)`` is an optional caller-owned pair of exact
    ``(retain * spatial_windows, 128, 1536)`` C-contiguous writable uint16
    arrays.  It lets a mapped cache own the K/V bytes without introducing a
    second in-memory history copy; ``release`` only drops this object's views.
    """

    def __init__(self, spatial_windows: int, retain: int, *, backing=None):
        if type(spatial_windows) is not int or spatial_windows <= 0:
            raise ValueError("spatial_windows must be a positive integer")
        if type(retain) is not int or retain not in (1, 3):
            raise ValueError("retain must be the Plus local/global value 1 or 3")
        self.spatial_windows = spatial_windows
        self.retain = retain
        self.capacity = retain * spatial_windows
        self.valid_count = 0
        expected_shape = (self.capacity, _BLOCK, _CHANNELS)
        if backing is None:
            self._k = np.empty(expected_shape, np.uint16)
            self._v = np.empty_like(self._k)
        else:
            if (type(backing) is not tuple or len(backing) != 2
                    or any(not isinstance(value, np.ndarray) for value in backing)):
                raise ValueError("backing must be a (k, v) NumPy array tuple")
            self._k, self._v = backing
            if any(value.dtype != np.uint16
                   or value.shape != expected_shape
                   or not value.flags.c_contiguous
                   or not value.flags.writeable
                   for value in (self._k, self._v)):
                raise ValueError(
                    "backing K/V must be exact writable C-contiguous uint16 arrays"
                )
            if np.shares_memory(self._k, self._v):
                raise ValueError("backing K/V arrays must not alias")
        self._pooled = np.empty((
            _HEADS, self.capacity, _BLOCK
        ), np.float32)
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    @property
    def k_storage(self) -> np.ndarray | None:
        return self._k

    @property
    def v_storage(self) -> np.ndarray | None:
        return self._v

    @property
    def pooled_storage(self) -> np.ndarray | None:
        return self._pooled

    def view(self) -> dict[str, np.ndarray] | None:
        """Return the existing valid-prefix ABI, or ``None`` before startup."""
        if self._released:
            return None
        if not self.valid_count:
            return None
        return {
            "k": self._k[:self.valid_count],
            "v": self._v[:self.valid_count],
            "pooled": self._pooled[:, :self.valid_count, :],
        }

    def _validate_current(
        self,
        current: dict[str, np.ndarray],
        *,
        terminal: bool = False,
        kv_start: int = 0,
        read_kv=None,
    ):
        if self._released:
            raise ValueError("history has been released")
        if type(kv_start) is not int:
            raise ValueError("kv_start must be an integer")
        spatial = self.spatial_windows
        if kv_start not in (0, spatial):
            raise ValueError("kv_start must be 0 or one spatial suffix")
        if kv_start and self.retain != 1:
            raise ValueError("compact K/V suffix requires retain=1")
        keys = set(current) if isinstance(current, dict) else set()
        if terminal and kv_start:
            raise ValueError("terminal history cannot use a partial K/V suffix")
        if read_kv is not None:
            if terminal or kv_start or keys != {"pooled"}:
                raise ValueError("read_kv requires nonterminal pooled-only current history")
        elif keys != {"k", "v", "pooled"} and not (terminal and keys == {"pooled"}):
            raise ValueError("current history must contain exactly k, v, pooled")
        count = 2 * self.spatial_windows
        kv_count = self.spatial_windows if kv_start else count
        expected = {"k": (kv_count, _BLOCK, _CHANNELS),
                    "v": (kv_count, _BLOCK, _CHANNELS),
                    "pooled": (_HEADS, count, _BLOCK)}
        for name, value in current.items():
            if not isinstance(value, np.ndarray) or value.shape != expected[name]:
                raise ValueError("current history geometry changed: " + name)
            if not value.flags.c_contiguous:
                raise ValueError("current history must be contiguous: " + name)
            if name != "pooled" and (value.dtype != np.uint16 or not is_finite_bf16(value)):
                raise ValueError("current history requires finite raw BF16: " + name)
            if name == "pooled" and (value.dtype != np.float32 or not np.isfinite(value).all()):
                raise ValueError("current pooled history requires finite FP32")
        return current.get("k"), current.get("v"), current["pooled"]

    def update(
        self,
        current: dict[str, np.ndarray],
        *,
        terminal: bool = False,
        kv_start: int = 0,
        read_kv=None,
    ) -> np.ndarray:
        """Merge pooled scores and retain the exact K/V suffix in place.

        ``view()`` must have been consumed by the remote upload before this
        call. ``kv_start=spatial_windows`` is the compact local suffix ABI:
        it supplies only the second current temporal window and is valid only
        for ``retain=1``. The pooled input always remains both current windows.
        On ``terminal=True`` the pooled merge is returned, then this owner
        releases its backing arrays and retains no history. Only that terminal
        call may omit K/V, and terminal calls cannot use a partial suffix.
        With ``read_kv``, a nonterminal call supplies only ``pooled`` and the
        callback fills the existing retained K/V suffix as
        ``read_kv(k_destination, v_destination, source_start)``.  The callback
        must return ``None`` and leave finite raw BF16 bits; failure releases
        this owner so no half-updated history can be consumed.
        """
        if type(terminal) is not bool:
            raise ValueError("terminal must be boolean")
        if read_kv is not None and not callable(read_kv):
            raise ValueError("read_kv must be callable")
        k, v, pooled = self._validate_current(
            current, terminal=terminal, kv_start=kv_start, read_kv=read_kv
        )
        old = self.valid_count
        if old:
            merged_pooled = np.concatenate((self._pooled[:, :old, :], pooled), axis=1)
        else:
            merged_pooled = pooled
        if terminal:
            self.release()
            return merged_pooled

        current_count = 2 * self.spatial_windows
        keep_current = min(current_count, self.capacity)
        keep_old = min(old, self.capacity - keep_current)
        if keep_old:
            old_start = old - keep_old
            # old_start >= keep_old for the only retaining geometries, so the
            # source and destination ranges are disjoint before appending.
            np.copyto(self._k[:keep_old], self._k[old_start:old])
            np.copyto(self._v[:keep_old], self._v[old_start:old])
            np.copyto(self._pooled[:, :keep_old, :],
                      self._pooled[:, old_start:old, :])
        # A compact local K/V update already starts at the retained suffix.
        # Pooled values always retain the last windows of the full two-window
        # input, because the pooled ABI is never compacted.
        current_kv_start = 0 if kv_start else current_count - keep_current
        pooled_start = current_count - keep_current
        destination = keep_old
        if read_kv is None:
            np.copyto(self._k[destination:destination + keep_current],
                      k[current_kv_start:])
            np.copyto(self._v[destination:destination + keep_current],
                      v[current_kv_start:])
        else:
            k_destination = self._k[destination:destination + keep_current]
            v_destination = self._v[destination:destination + keep_current]
            # Mark the existing owner invalid before the callback so a partial
            # or no-op fill cannot accidentally leave a valid-looking state.
            k_destination.fill(np.uint16(0x7F80))
            v_destination.fill(np.uint16(0x7F80))
            try:
                if read_kv(k_destination, v_destination,
                           current_count - keep_current) is not None:
                    raise ValueError("read_kv must return None")
                if not is_finite_bf16(k_destination) or not is_finite_bf16(v_destination):
                    raise ValueError("read_kv produced non-finite raw BF16")
            except Exception:
                self.release()
                raise
        np.copyto(self._pooled[:, destination:destination + keep_current, :],
                  pooled[:, pooled_start:, :])
        self.valid_count = keep_old + keep_current
        return merged_pooled

    def release(self) -> None:
        """Drop the owner references after the final attention consumer."""
        self._k = self._v = self._pooled = None
        self.valid_count = 0
        self._released = True


__all__ = ["CompactHistory"]
