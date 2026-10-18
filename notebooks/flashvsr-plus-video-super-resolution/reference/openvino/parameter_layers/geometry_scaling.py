#!/usr/bin/env python3
"""Geometry-only adapter for the parameterized layer prototype.

The active 128x128 ABI remains owned by ``contract.py``.  This adapter keeps
that ABI untouched while widening only the static spatial geometry accepted by
the existing parameterized builders.  It intentionally has a finite safety
ceiling: the structural qualification target is 512x512 (h=w=32), not a UHD
inference or admission claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from contract import DIM, WINDOW_TOKENS, Geometry


WINDOW_F, WINDOW_H, WINDOW_W = 2, 8, 8
MAX_SAFE_H = 32
MAX_SAFE_W = 32
MAX_SAFE_F = 6
MAX_SAFE_TOKENS = MAX_SAFE_F * MAX_SAFE_H * MAX_SAFE_W


@dataclass(frozen=True)
class GeometryAdapter(Geometry):
    """Parameter-builder geometry with source-compatible static attributes."""

    streaming: bool = True

    @property
    def query_windows(self) -> int:
        return (self.f // WINDOW_F) * (self.h // WINDOW_H) * (self.w // WINDOW_W)

    @property
    def window_shape(self) -> tuple[int, int, int]:
        return (self.query_windows, WINDOW_TOKENS, DIM)

    @property
    def attention_key_windows(self) -> int:
        """Source-compatible old+new capacity; unused by pre/post builders."""

        spatial_windows = (self.h // WINDOW_H) * (self.w // WINDOW_W)
        return self.query_windows if self.f == 6 else 3 * spatial_windows + self.query_windows

    def validate(self) -> None:
        if self.f not in (2, 6):
            raise ValueError(f"f must be 2 or 6 for the source schedule, got {self.f}")
        if self.h <= 0 or self.h % WINDOW_H:
            raise ValueError(f"h must be a positive multiple of {WINDOW_H}, got {self.h}")
        if self.w <= 0 or self.w % WINDOW_W:
            raise ValueError(f"w must be a positive multiple of {WINDOW_W}, got {self.w}")
        if self.h > MAX_SAFE_H or self.w > MAX_SAFE_W:
            raise ValueError(
                f"geometry exceeds the finite structural safety ceiling {MAX_SAFE_H}x{MAX_SAFE_W}: "
                f"{self.h}x{self.w}"
            )
        if self.tokens > MAX_SAFE_TOKENS:
            raise ValueError(
                f"geometry exceeds the finite token safety ceiling {MAX_SAFE_TOKENS}: {self.tokens}"
            )
        if self.query_windows <= 0:
            raise ValueError(f"geometry has no query windows: {self}")


def scaled_geometry(name: str, f: int, h: int, w: int) -> GeometryAdapter:
    """Create and validate one finite static geometry for existing builders."""

    geometry = GeometryAdapter(name, f, h, w)
    geometry.validate()
    return geometry


STARTUP_512 = scaled_geometry("startup_f6_512x512", 6, 32, 32)
STEADY_512 = scaled_geometry("steady_f2_512x512", 2, 32, 32)


def safety_contract() -> dict[str, object]:
    """Return the explicit structural ceiling used by the CPU-only receipt."""

    return {
        "window": [WINDOW_F, WINDOW_H, WINDOW_W],
        "allowed_f": [2, 6],
        "max_safe_h": MAX_SAFE_H,
        "max_safe_w": MAX_SAFE_W,
        "max_safe_tokens": MAX_SAFE_TOKENS,
        "inference_admitted": False,
        "quality_admitted": False,
    }
