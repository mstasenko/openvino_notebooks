#!/usr/bin/env python3
"""Layer-0 Wan DiTBlock shapes/state for the finite CPU prototype.

The graphs split source norm/modulation/RoPE/window qkv, bounded global-softmax
attention, and post-attention gate/cross-attention/FFN.  Public state is f16;
normalization, RoPE, score, and softmax reductions are f32.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple


BASE = Path(__file__).resolve().parents[2]
WEIGHTS = BASE / "models/diffusion_pytorch_model_streaming_dmd.safetensors"
PROMPT = BASE / "source/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth"

EXPECTED_CHECKPOINT_SHA256 = (
    "bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be"
)
EXPECTED_CONFIG = {"has_image_input": False, "patch_size": [1, 2, 2], "in_dim": 16,
                   "dim": 1536, "ffn_dim": 8960, "freq_dim": 256, "text_dim": 4096,
                   "out_dim": 16, "num_heads": 12, "num_layers": 30, "eps": 1e-6}

DIM = EXPECTED_CONFIG["dim"]
FFN_DIM = EXPECTED_CONFIG["ffn_dim"]
TEXT_DIM = EXPECTED_CONFIG["text_dim"]
HEADS = EXPECTED_CONFIG["num_heads"]
HEAD_DIM = DIM // HEADS
CONTEXT_TOKENS = 512
WINDOW = (2, 8, 8)
WINDOW_TOKENS = 128
KV_LEN = 3


# norm1/norm2 are affine=False in the source and have no checkpoint keys.
SELF_KEYS = tuple("blocks.0.self_attn.q.weight blocks.0.self_attn.q.bias blocks.0.self_attn.norm_q.weight "
                  "blocks.0.self_attn.k.weight blocks.0.self_attn.k.bias blocks.0.self_attn.norm_k.weight "
                  "blocks.0.self_attn.v.weight blocks.0.self_attn.v.bias blocks.0.self_attn.o.weight blocks.0.self_attn.o.bias".split())
CROSS_KEYS = tuple("blocks.0.cross_attn.q.weight blocks.0.cross_attn.q.bias blocks.0.cross_attn.norm_q.weight "
                   "blocks.0.cross_attn.k.weight blocks.0.cross_attn.k.bias blocks.0.cross_attn.norm_k.weight "
                   "blocks.0.cross_attn.v.weight blocks.0.cross_attn.v.bias blocks.0.cross_attn.o.weight blocks.0.cross_attn.o.bias".split())
FFN_KEYS = tuple("blocks.0.ffn.0.weight blocks.0.ffn.0.bias blocks.0.ffn.2.weight blocks.0.ffn.2.bias".split())
BLOCK_EXTRA_KEYS = tuple("blocks.0.modulation blocks.0.norm3.weight blocks.0.norm3.bias".split())
TEXT_EMBED_KEYS = tuple("text_embedding.0.weight text_embedding.0.bias text_embedding.2.weight text_embedding.2.bias".split())
BLOCK_KEYS = SELF_KEYS + CROSS_KEYS + FFN_KEYS + BLOCK_EXTRA_KEYS
CACHE_PREP_KEYS = TEXT_EMBED_KEYS + (
    "blocks.0.cross_attn.k.weight",
    "blocks.0.cross_attn.k.bias",
    "blocks.0.cross_attn.norm_k.weight",
    "blocks.0.cross_attn.v.weight",
    "blocks.0.cross_attn.v.bias",
)
ALL_SELECTED_KEYS = BLOCK_KEYS + TEXT_EMBED_KEYS


_MATRIX_KEYS = tuple("blocks.0.self_attn.q.weight blocks.0.self_attn.k.weight blocks.0.self_attn.v.weight blocks.0.self_attn.o.weight "
                     "blocks.0.cross_attn.q.weight blocks.0.cross_attn.k.weight blocks.0.cross_attn.v.weight blocks.0.cross_attn.o.weight".split())
_VECTOR_KEYS = tuple("blocks.0.self_attn.q.bias blocks.0.self_attn.k.bias blocks.0.self_attn.v.bias blocks.0.self_attn.o.bias "
                    "blocks.0.self_attn.norm_q.weight blocks.0.self_attn.norm_k.weight "
                    "blocks.0.cross_attn.q.bias blocks.0.cross_attn.k.bias blocks.0.cross_attn.v.bias blocks.0.cross_attn.o.bias "
                    "blocks.0.cross_attn.norm_q.weight blocks.0.cross_attn.norm_k.weight blocks.0.norm3.weight blocks.0.norm3.bias".split())
EXPECTED_SHAPES = {
    **{key: [DIM, DIM] for key in _MATRIX_KEYS}, **{key: [DIM] for key in _VECTOR_KEYS},
    "blocks.0.ffn.0.weight": [FFN_DIM, DIM],
    "blocks.0.ffn.0.bias": [FFN_DIM],
    "blocks.0.ffn.2.weight": [DIM, FFN_DIM],
    "blocks.0.ffn.2.bias": [DIM],
    "blocks.0.modulation": [1, 6, DIM],
    "text_embedding.0.weight": [DIM, TEXT_DIM],
    "text_embedding.0.bias": [DIM],
    "text_embedding.2.weight": [DIM, DIM],
    "text_embedding.2.bias": [DIM],
}
@dataclass(frozen=True)
class Geometry:
    """Static token geometry for one block invocation."""

    name: str
    f: int
    h: int
    w: int
    streaming: bool

    @property
    def tokens(self) -> int:
        return self.f * self.h * self.w

    @property
    def query_windows(self) -> int:
        return (self.f // WINDOW[0]) * (self.h // WINDOW[1]) * (self.w // WINDOW[2])

    @property
    def token_shape(self) -> Tuple[int, int, int]:
        return (1, self.tokens, DIM)

    @property
    def window_shape(self) -> Tuple[int, int, int]:
        return (self.query_windows, WINDOW_TOKENS, DIM)

    @property
    def attention_key_windows(self) -> int:
        """Old+new key-window capacity for the pinned streaming call."""
        return self.query_windows if self.f == 6 else KV_LEN + self.query_windows

    def validate(self) -> None:
        for value, divisor, label in (
            (self.f, WINDOW[0], "f"),
            (self.h, WINDOW[1], "h"),
            (self.w, WINDOW[2], "w"),
        ):
            if value <= 0 or value % divisor:
                raise ValueError(f"{label}={value} must be positive and divisible by {divisor}")
        if self.streaming and self.f not in (2, 6):
            raise ValueError("the pinned streaming source permits only f=6 startup or f=2 steady")


STARTUP = Geometry("startup_f6_128x128", 6, 8, 8, True)
STEADY = Geometry("steady_f2_128x128", 2, 8, 8, True)


def state_contract(geometry: Geometry) -> Dict[str, object]:
    """Return the small public state ABI for one real streaming geometry."""
    geometry.validate()
    n = geometry.query_windows
    token = list(geometry.token_shape)
    window = list(geometry.window_shape)
    mod = [1, 1, DIM]
    old_windows = 0 if geometry.f == 6 else KV_LEN
    return {
        "geometry": {"name": geometry.name, "f": geometry.f, "h": geometry.h, "w": geometry.w,
                     "tokens": geometry.tokens, "query_windows": n, "window": list(WINDOW)},
        "pre_qkv": {
            "inputs": {"x": [token, "f16"], "t_mod": [[1, 6, DIM], "f16"],
                       "rope_cos/sin": [[1, geometry.tokens, 1, HEAD_DIM // 2], "f32"]},
            "outputs": {name: [shape, "f16"] for name, shape in (
                ("residual", token), ("q_windows", window), ("k_windows", window),
                ("v_windows", window), ("gate_msa", mod), ("shift_mlp", mod),
                ("scale_mlp", mod), ("gate_mlp", mod))},
        },
        "attention": {
            "inputs": {"q_windows": [window, "f16"],
                       "k/v_windows": [[geometry.attention_key_windows, WINDOW_TOKENS, DIM], "f16(old+new)"],
                       "selectedKVblock": [[HEADS, n, geometry.attention_key_windows], "i32"],
                       "validSlot": [[HEADS, n, geometry.attention_key_windows], "boolean"]},
            "outputs": {"self_context": [token, "f16"]},
            "query_chunk_windows": 1,
            "softmax": "all selected old+new key tokens globally per row",
            "selection": "lossless source block-mask packing; invalid slots are masked and empty rows zeroed",
        },
        "post_block": {
            "inputs": {name: [shape, "f16"] for name, shape in (
                ("x", token), ("self_context", token), ("gate_msa", mod),
                ("shift_mlp", mod), ("scale_mlp", mod), ("gate_mlp", mod),
                ("cross_k", [1, CONTEXT_TOKENS, DIM]), ("cross_v", [1, CONTEXT_TOKENS, DIM]))},
            "outputs": {"x_out": [token, "f16"]},
        },
        "cache": {
            "old_kv": [[old_windows, WINDOW_TOKENS, DIM], "f16, oldest-to-newest"],
            "new_kv": [[n, WINDOW_TOKENS, DIM], "f16, current call"],
            "attended_kv": [[geometry.attention_key_windows, WINDOW_TOKENS, DIM], "f16, old then new"],
            "retained": [[KV_LEN, WINDOW_TOKENS, DIM], "f16"],
            "policy": "cat(old,new), attend, then evict oldest until kv_len=3",
        },
    }
def stream_schedule() -> Tuple[Dict[str, object], ...]:
    """Return the source chronological cache schedule for startup and steady."""

    return (
        {
            "step": "startup",
            "geometry": STARTUP.name,
            "old_windows": 0,
            "new_windows": STARTUP.query_windows,
            "attention_key_windows": STARTUP.query_windows,
            "post_attention_cache_windows": KV_LEN,
            "evict_after_attention": 0,
            "source_assertion": "is_stream and pre_cache is None requires f=6",
        },
        {
            "step": "steady",
            "geometry": STEADY.name,
            "old_windows": KV_LEN,
            "new_windows": STEADY.query_windows,
            "attention_key_windows": KV_LEN + STEADY.query_windows,
            "post_attention_cache_windows": KV_LEN,
            "evict_after_attention": STEADY.query_windows,
            "source_assertion": "is_stream and pre_cache is not None requires f=2",
        },
    )


def validate_key_shapes(metadata: Mapping[str, Mapping[str, object]], keys: Iterable[str] = ALL_SELECTED_KEYS) -> None:
    """Validate only metadata entries for the explicitly selected keys."""

    for key in keys:
        if key not in metadata:
            raise KeyError(f"missing selected checkpoint key: {key}")
        shape = list(metadata[key]["shape"])
        dtype = str(metadata[key]["dtype"])
        if shape != EXPECTED_SHAPES[key] or dtype != "F32":
            raise ValueError(
                f"unexpected metadata for {key}: shape={shape}, dtype={dtype}; "
                f"expected shape={EXPECTED_SHAPES[key]}, dtype=F32"
            )


def source_summary() -> Dict[str, object]:
    return {
        "checkpoint": str(WEIGHTS),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "config": EXPECTED_CONFIG,
        "prompt": str(PROMPT),
        "prompt_shape": [1, CONTEXT_TOKENS, TEXT_DIM],
        "prompt_source_dtype": "torch.bfloat16",
        "selected_key_counts": {"block": len(BLOCK_KEYS), "cache_prep": len(CACHE_PREP_KEYS), "with_text": len(ALL_SELECTED_KEYS)},
        "all_checkpoint_key_count": 825,
        "window": list(WINDOW), "context_tokens": CONTEXT_TOKENS, "cross_cache_hidden": DIM,
        "normalization_eps": EXPECTED_CONFIG["eps"],
        "rope_source_precision": "source complex128; graph explicit real-F32 cos/sin",
    }

