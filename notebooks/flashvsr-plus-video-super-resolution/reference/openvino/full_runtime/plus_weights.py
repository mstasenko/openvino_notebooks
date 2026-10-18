"""Authenticated, one-layer-at-a-time access to the FlashVSR+ Turbo DiT.

The checkpoint is deliberately kept behind a ``safe_open`` handle supplied by
the caller.  ``load_layer`` copies only the requested block and normalizes its
names to the layer-0 parameter ABI used by the reusable block builders.  It
does not import Torch or OpenVINO and never creates an F16 intermediate.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from plus_edges import EXPECTED_SHAPES as EDGE_SHAPES
from plus_edges import SELECTED_KEYS as EDGE_KEYS
from plus_edges import CHECKPOINT, CHECKPOINT_BYTES, CHECKPOINT_SHA256
from plus_modes import resolve_plus_mode


NUM_LAYERS = 30
LAYER_PARAMETER_COUNT = 27
SCHEMA_KEY_COUNT = 825

_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "parameter_layers" / "contract.py"


def _load_parameter_contract():
    """Load the pure parameter ABI without importing a runtime or graph builder."""

    spec = importlib.util.spec_from_file_location("plus_weights_parameter_contract", _CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load parameter contract: {_CONTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    # A unique name avoids disturbing the historical ``contract`` import used
    # by the graph-builder modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CONTRACT = _load_parameter_contract()
ROLE_KEYS = {role: tuple(keys) for role, keys in _CONTRACT.ROLE_KEYS.items()}
PARAM_SHAPES = {key: tuple(shape) for key, shape in _CONTRACT.PARAM_SHAPES.items()}

# The reusable pre/post ABI intentionally omits cross-attention K/V.  They are
# still per-layer source parameters in the Plus checkpoint and are needed by
# the full 30-layer integration.
EXTRA_LAYER_KEYS = (
    "blocks.0.cross_attn.k.weight",
    "blocks.0.cross_attn.k.bias",
    "blocks.0.cross_attn.norm_k.weight",
    "blocks.0.cross_attn.v.weight",
    "blocks.0.cross_attn.v.bias",
)
EXTRA_LAYER_SHAPES = {
    "blocks.0.cross_attn.k.weight": (1536, 1536),
    "blocks.0.cross_attn.k.bias": (1536,),
    "blocks.0.cross_attn.norm_k.weight": (1536,),
    "blocks.0.cross_attn.v.weight": (1536, 1536),
    "blocks.0.cross_attn.v.bias": (1536,),
}

LAYER_KEYS = tuple(dict.fromkeys((*ROLE_KEYS["pre"], *ROLE_KEYS["post"], *EXTRA_LAYER_KEYS)))
if len(LAYER_KEYS) != LAYER_PARAMETER_COUNT:
    raise AssertionError(f"unexpected Plus layer ABI size: {len(LAYER_KEYS)}")
LAYER_SHAPES = {key: PARAM_SHAPES[key] for key in (*ROLE_KEYS["pre"], *ROLE_KEYS["post"])}
LAYER_SHAPES.update(EXTRA_LAYER_SHAPES)

TEXT_KEYS = (
    "text_embedding.0.weight",
    "text_embedding.0.bias",
    "text_embedding.2.weight",
    "text_embedding.2.bias",
)
TEXT_SHAPES = {
    "text_embedding.0.weight": (1536, 4096),
    "text_embedding.0.bias": (1536,),
    "text_embedding.2.weight": (1536, 1536),
    "text_embedding.2.bias": (1536,),
}
GLOBAL_SHAPES = {**EDGE_SHAPES, **TEXT_SHAPES}
if set(EDGE_KEYS) != set(EDGE_SHAPES):
    raise AssertionError("Plus edge key and shape allow-lists differ")


def _layer_key(canonical_key: str, layer: int) -> str:
    if type(layer) is not int or not 0 <= layer < NUM_LAYERS:
        raise ValueError(f"layer must be an integer in [0,{NUM_LAYERS}), got {layer!r}")
    if not canonical_key.startswith("blocks.0."):
        raise KeyError(f"not a canonical layer key: {canonical_key!r}")
    return canonical_key.replace("blocks.0.", f"blocks.{layer}.", 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_schema() -> dict[str, tuple[int, ...]]:
    expected = dict(GLOBAL_SHAPES)
    for layer in range(NUM_LAYERS):
        expected.update({_layer_key(key, layer): shape for key, shape in LAYER_SHAPES.items()})
    return expected


EXPECTED_SCHEMA = _expected_schema()
if len(EXPECTED_SCHEMA) != SCHEMA_KEY_COUNT:
    raise AssertionError(f"unexpected Plus checkpoint schema size: {len(EXPECTED_SCHEMA)}")


def _slice_metadata(handle: Any, key: str) -> tuple[tuple[int, ...], str]:
    try:
        view = handle.get_slice(key)
        shape = tuple(view.get_shape())
        dtype = str(view.get_dtype())
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read checkpoint metadata for {key!r}") from exc
    return shape, dtype


def validate_schema(handle: Any) -> dict[str, Any]:
    """Validate all 825 headers without materializing any tensor payload."""

    try:
        keys = list(handle.keys())
    except (AttributeError, TypeError) as exc:
        raise TypeError("checkpoint handle must provide keys() and get_slice()") from exc
    try:
        key_set = set(keys)
    except TypeError as exc:
        raise ValueError("Turbo checkpoint keys must be hashable strings") from exc
    if any(not isinstance(key, str) for key in keys):
        raise ValueError("Turbo checkpoint keys must be strings")
    if len(keys) != len(key_set) or key_set != set(EXPECTED_SCHEMA):
        missing = sorted(set(EXPECTED_SCHEMA) - key_set)
        extra = sorted(key_set - set(EXPECTED_SCHEMA))
        raise ValueError(f"Turbo checkpoint key schema mismatch: missing={missing}, extra={extra}")
    for key, expected_shape in EXPECTED_SCHEMA.items():
        shape, dtype = _slice_metadata(handle, key)
        if shape != expected_shape or dtype != "F32":
            raise ValueError(
                f"unexpected checkpoint header {key}: shape={shape} dtype={dtype}; "
                f"expected {expected_shape} F32"
            )
    return {
        "key_count": len(keys),
        "layer_count": NUM_LAYERS,
        "layer_parameter_count": LAYER_PARAMETER_COUNT,
        "global_key_count": len(GLOBAL_SHAPES),
        "source_dtype": "F32",
    }


def authenticate_dit(path: Path = CHECKPOINT, *, mode: str = "turbo") -> dict[str, Any]:
    """Authenticate the selected pinned bytes and exact header-only schema."""

    spec = resolve_plus_mode(mode)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != spec.dit_bytes:
        raise ValueError(f"unexpected {mode} checkpoint size: {size} != {spec.dit_bytes}")
    digest = _sha256(path)
    if digest != spec.dit_sha256:
        raise ValueError(f"unexpected {mode} checkpoint hash: {digest}")

    from safetensors import safe_open

    with safe_open(str(path), framework="np", device="cpu") as handle:
        schema = validate_schema(handle)
    return {
        "path": str(path.resolve()),
        "bytes": int(size),
        "sha256": digest,
        "mode": mode,
        **schema,
    }


def load_layer(handle: Any, index: int) -> dict[str, np.ndarray]:
    """Copy one source block into the canonical layer-0 key ABI.

    The caller owns the ``safe_open`` context.  Every returned array is a
    detached C-contiguous F32 copy, so it remains valid after that context is
    closed and no stale safetensors view can escape.
    """

    if type(index) is not int or not 0 <= index < NUM_LAYERS:
        raise ValueError(f"layer must be an integer in [0,{NUM_LAYERS}), got {index!r}")
    validate_schema(handle)
    values: dict[str, np.ndarray] = {}
    for canonical_key in LAYER_KEYS:
        source_key = _layer_key(canonical_key, index)
        expected_shape = LAYER_SHAPES[canonical_key]
        source = np.asarray(handle.get_tensor(source_key))
        if source.shape != expected_shape or source.dtype != np.dtype(np.float32):
            raise ValueError(
                f"unexpected source tensor {source_key}: shape={source.shape} dtype={source.dtype}; "
                f"expected {expected_shape} F32"
            )
        if not np.isfinite(source).all():
            raise ValueError(f"non-finite source tensor: {source_key}")
        values[canonical_key] = np.array(source, dtype=np.float32, order="C", copy=True)
    return values
