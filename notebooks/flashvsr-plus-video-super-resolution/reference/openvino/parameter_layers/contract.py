#!/usr/bin/env python3
"""Explicit ABI for parameterized one-block pre/post graphs."""

from __future__ import annotations

from dataclasses import dataclass


DIM = 1536
HEADS = 12
HEAD_DIM = 128
FFN_DIM = 8960
CONTEXT_TOKENS = 512
WINDOW_TOKENS = 128
EPS = 1e-6

# These are the only learned values consumed by the two graph roles.  The
# names remain the canonical layer-0 checkpoint names; the runtime maps
# blocks.{layer}. to this ABI before feeding one batch of F16 values.
PRE_KEYS = (
    "blocks.0.modulation",
    "blocks.0.self_attn.q.weight",
    "blocks.0.self_attn.q.bias",
    "blocks.0.self_attn.norm_q.weight",
    "blocks.0.self_attn.k.weight",
    "blocks.0.self_attn.k.bias",
    "blocks.0.self_attn.norm_k.weight",
    "blocks.0.self_attn.v.weight",
    "blocks.0.self_attn.v.bias",
)

POST_KEYS = (
    "blocks.0.self_attn.o.weight",
    "blocks.0.self_attn.o.bias",
    "blocks.0.norm3.weight",
    "blocks.0.norm3.bias",
    "blocks.0.cross_attn.q.weight",
    "blocks.0.cross_attn.q.bias",
    "blocks.0.cross_attn.norm_q.weight",
    "blocks.0.cross_attn.o.weight",
    "blocks.0.cross_attn.o.bias",
    "blocks.0.ffn.0.weight",
    "blocks.0.ffn.0.bias",
    "blocks.0.ffn.2.weight",
    "blocks.0.ffn.2.bias",
)

ROLE_KEYS = {"pre": PRE_KEYS, "post": POST_KEYS}

PARAM_SHAPES = {
    "blocks.0.modulation": (1, 6, DIM),
    "blocks.0.self_attn.q.weight": (DIM, DIM),
    "blocks.0.self_attn.q.bias": (DIM,),
    "blocks.0.self_attn.norm_q.weight": (DIM,),
    "blocks.0.self_attn.k.weight": (DIM, DIM),
    "blocks.0.self_attn.k.bias": (DIM,),
    "blocks.0.self_attn.norm_k.weight": (DIM,),
    "blocks.0.self_attn.v.weight": (DIM, DIM),
    "blocks.0.self_attn.v.bias": (DIM,),
    "blocks.0.self_attn.o.weight": (DIM, DIM),
    "blocks.0.self_attn.o.bias": (DIM,),
    "blocks.0.norm3.weight": (DIM,),
    "blocks.0.norm3.bias": (DIM,),
    "blocks.0.cross_attn.q.weight": (DIM, DIM),
    "blocks.0.cross_attn.q.bias": (DIM,),
    "blocks.0.cross_attn.norm_q.weight": (DIM,),
    "blocks.0.cross_attn.o.weight": (DIM, DIM),
    "blocks.0.cross_attn.o.bias": (DIM,),
    "blocks.0.ffn.0.weight": (FFN_DIM, DIM),
    "blocks.0.ffn.0.bias": (FFN_DIM,),
    "blocks.0.ffn.2.weight": (DIM, FFN_DIM),
    "blocks.0.ffn.2.bias": (DIM,),
}

for _role, _keys in ROLE_KEYS.items():
    if len(_keys) != len(set(_keys)):
        raise AssertionError(f"duplicate {_role} parameter key")
    if any(_key not in PARAM_SHAPES for _key in _keys):
        raise AssertionError(f"missing shape for {_role} parameter")


def parameter_port(key: str) -> str:
    """Stable input name derived directly from the canonical checkpoint key."""

    if key not in PARAM_SHAPES:
        raise KeyError(f"parameter key is not in the explicit allow-list: {key}")
    return "weight__" + key.replace(".", "_")


def checkpoint_key(key: str, layer: int) -> str:
    if not 0 <= layer < 30:
        raise ValueError(f"layer must be in [0,30), got {layer}")
    if not key.startswith("blocks.0."):
        raise KeyError(key)
    return key.replace("blocks.0.", f"blocks.{layer}.", 1)


@dataclass(frozen=True)
class Geometry:
    name: str
    f: int
    h: int
    w: int

    @property
    def tokens(self) -> int:
        return self.f * self.h * self.w

    @property
    def query_windows(self) -> int:
        return (self.f // 2) * (self.h // 8) * (self.w // 8)

    @property
    def token_shape(self) -> tuple[int, int, int]:
        return (1, self.tokens, DIM)

    def validate(self) -> None:
        if self.f not in (2, 6) or self.h != 8 or self.w != 8:
            raise ValueError(f"parameter layer prototype is pinned to f in (2, 6), h=8, w=8: {self}")


STARTUP = Geometry("startup_f6_128x128", 6, 8, 8)
STEADY = Geometry("steady_f2_128x128", 2, 8, 8)


def parameter_schema(role: str) -> tuple[dict[str, object], ...]:
    if role not in ROLE_KEYS:
        raise ValueError(f"role must be pre or post, got {role!r}")
    return tuple({
        "key": key,
        "port": parameter_port(key),
        "shape": list(PARAM_SHAPES[key]),
        "storage_dtype": "F16",
        "norm_scale_arithmetic": "F32" if key in ("blocks.0.norm3.weight", "blocks.0.norm3.bias") else "native graph boundary",
    } for key in ROLE_KEYS[role])
