"""Immutable identities and layer policy for the released Plus modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SOURCE_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
MODEL_REPO_REVISION = "46b79520946ac67dec94d1566490690953c7debc"

_DIT_BYTES = 5_676_070_392
_LQ_BYTES = 1_151_384_272
_TURBO_DIT_SHA256 = "7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b"
_TURBO_LQ_SHA256 = "e77b378ae12342eefbb58f924256f362a151b93ce73dd32808fcdaca9c051496"
_BASE_DIT_SHA256 = "58d64c1f1b3a559bafb69bb014f3ab2745e800c89435afdc154966eb96bc7459"
_BASE_LQ_SHA256 = "a1fe997f19060171db8dd4d58c30debc78ddfcc357b63481798792a38a356d8d"
_TURBO_LAYERS = tuple(
    layer for layer in range(30) if layer % 6 not in (0,)
)


def _valid_sha(value: str) -> bool:
    if type(value) is not str or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _layer(layer: int) -> int:
    if type(layer) is not int or not 0 <= layer < 30:
        raise ValueError("layer must be an integer in [0, 29]")
    return layer


@dataclass(frozen=True)
class PlusModeSpec:
    """Authenticated checkpoint identities and the mode's layer schedule."""

    name: str
    dit_sha256: str
    lq_sha256: str
    replacement_layers: tuple[int, ...]
    dit_bytes: int = _DIT_BYTES
    lq_bytes: int = _LQ_BYTES

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in ("turbo", "base"):
            raise ValueError("mode name must be exactly 'turbo' or 'base'")
        if not _valid_sha(self.dit_sha256) or not _valid_sha(self.lq_sha256):
            raise ValueError("checkpoint identities must be lowercase SHA-256 hex")
        expected_sha = (
            (_TURBO_DIT_SHA256, _TURBO_LQ_SHA256)
            if self.name == "turbo"
            else (_BASE_DIT_SHA256, _BASE_LQ_SHA256)
        )
        if (self.dit_sha256, self.lq_sha256) != expected_sha:
            raise ValueError("checkpoint identities do not match the pinned mode")
        if type(self.dit_bytes) is not int or self.dit_bytes != _DIT_BYTES:
            raise ValueError("unexpected DIT checkpoint size")
        if type(self.lq_bytes) is not int or self.lq_bytes != _LQ_BYTES:
            raise ValueError("unexpected LQ checkpoint size")
        if type(self.replacement_layers) is not tuple or any(
            type(layer) is not int or not 0 <= layer < 30
            for layer in self.replacement_layers
        ):
            raise ValueError("replacement_layers must contain layers in [0, 29]")
        if len(set(self.replacement_layers)) != len(self.replacement_layers):
            raise ValueError("replacement_layers must not contain duplicates")
        expected = _TURBO_LAYERS if self.name == "turbo" else ()
        if self.replacement_layers != expected:
            raise ValueError("replacement layer schedule does not match the mode")

    def policy_for_layer(self, layer: int) -> tuple[int, int]:
        layer = _layer(layer)
        return (1, 3) if layer in self.replacement_layers else (3, 11)

    def retain_groups(self, layer: int) -> int:
        return self.policy_for_layer(layer)[0]

    def local_range(self, layer: int) -> int:
        return self.policy_for_layer(layer)[1]

    def helper_paths(self, model_root: str | Path) -> dict[str, Path]:
        """Return only explicit mode paths, rejecting resolved escapes."""

        try:
            root = Path(model_root).expanduser().resolve(strict=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("model_root must be a filesystem path") from exc
        mode_root = root / self.name
        result = {
            "dit": (mode_root / "dit.safetensors").resolve(strict=False),
            "lq": (mode_root / "lq_proj_in.safetensors").resolve(strict=False),
        }
        for path in result.values():
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("checkpoint path escapes model_root") from exc
        return result


_TURBO = PlusModeSpec("turbo", _TURBO_DIT_SHA256, _TURBO_LQ_SHA256, _TURBO_LAYERS)
_BASE = PlusModeSpec("base", _BASE_DIT_SHA256, _BASE_LQ_SHA256, ())


def resolve_plus_mode(name: str = "turbo") -> PlusModeSpec:
    """Resolve exactly ``turbo`` or ``base``; there is no fallback mode."""

    if type(name) is not str:
        raise ValueError("mode name must be exactly 'turbo' or 'base'")
    if name == "turbo":
        return _TURBO
    if name == "base":
        return _BASE
    raise ValueError("unsupported Plus mode; expected exactly 'turbo' or 'base'")


__all__ = ["MODEL_REPO_REVISION", "PlusModeSpec", "SOURCE_COMMIT", "resolve_plus_mode"]
