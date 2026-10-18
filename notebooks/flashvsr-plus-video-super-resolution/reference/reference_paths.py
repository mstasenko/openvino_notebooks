"""Explicit paths for the portable FlashVSR+ research extraction.

The upstream qualification scripts were intentionally tied to one workstation
and its private cache.  This module is the only path policy in the extracted
tree: callers may set the environment variables below or pass paths directly
to the public loader functions.  No model, source checkout, media, or scratch
directory is downloaded by this module. The source-only model closure is
bundled separately under ``source/``; learned checkpoints remain external.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# The source closure is bundled under ``reference/source``.  It can still be
# replaced explicitly when reproducing a different authenticated checkout.
SOURCE_ROOT = _path("FLASHVSR_PLUS_SOURCE", ROOT / "source")
MODEL_ROOT = _path("FLASHVSR_PLUS_MODELS", ROOT / "external/models")
SCRATCH_ROOT = _path("FLASHVSR_PLUS_SCRATCH", ROOT / "external/scratch")
EVIDENCE_ROOT = _path("FLASHVSR_PLUS_EVIDENCE", ROOT / "external/evidence")
PROMPT_PATH = _path(
    "FLASHVSR_PLUS_PROMPT", SOURCE_ROOT / "prompt_tensor/posi_prompt.pth"
)
TCDECODER_CHECKPOINT = _path("FLASHVSR_PLUS_TCDECODER", MODEL_ROOT / "TCDecoder.ckpt")
DIT_CHECKPOINT = _path(
    "FLASHVSR_PLUS_DIT", MODEL_ROOT / "turbo/dit.safetensors"
)
LQ_CHECKPOINT = _path(
    "FLASHVSR_PLUS_LQ", MODEL_ROOT / "turbo/lq_proj_in.safetensors"
)


def require_file(path: Path, role: str) -> Path:
    """Return an explicit input path, failing with a useful portable error."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"{role} is not available at {path}; set the corresponding "
            "FLASHVSR_PLUS_* environment variable or pass the loader path"
        )
    return path


__all__ = [
    "ROOT", "SOURCE_ROOT", "MODEL_ROOT", "SCRATCH_ROOT", "EVIDENCE_ROOT", "PROMPT_PATH",
    "TCDECODER_CHECKPOINT", "DIT_CHECKPOINT", "LQ_CHECKPOINT", "require_file",
]
