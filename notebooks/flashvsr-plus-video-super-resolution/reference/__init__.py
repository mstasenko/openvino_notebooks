"""Portable, research-only FlashVSR+ reference extraction.

The package contains the authenticated Python/OpenVINO closure used by the
qualified local experiments.  It intentionally contains no model weights,
media, generated archives, or native plugin binaries.
"""

from .reference_paths import ROOT

__all__ = ["ROOT"]
