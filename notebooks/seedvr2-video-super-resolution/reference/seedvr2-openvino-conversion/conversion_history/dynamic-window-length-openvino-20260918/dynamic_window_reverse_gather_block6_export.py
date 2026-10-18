#!/usr/bin/env python3
"""Bounded CPU proof of a dynamic-window-length SeedVR2 block-16 IR.

The only model weights materialized by this proof are the exact block-16
entries selected from the pinned safetensors file.  The upstream
window operation and block math remain the source of truth.  Window count and
per-window padded length remain runtime dimensions; each case supplies only its
own actual maximum window length. Video reconstruction uses an exact
inverse-gather map: every source-order token maps once to a valid padded
window position, so the lowered graph has no ScatterElementsUpdate/index_add.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TRIAL_ROOT = HERE.parents[1]
SOURCE = Path(os.environ.get("SEEDVR_SOURCE", TRIAL_ROOT / "source_snapshot"))
WEIGHTS = Path(os.environ.get("SEEDVR_WEIGHTS", TRIAL_ROOT / "seedvr2_ema_3b_fp16.safetensors"))
OV_ROOT = Path(os.environ.get("SEEDVR_OPENVINO_ROOT", TRIAL_ROOT / "runtime-openvino"))
TRUSTED_HELPER = HERE.parent / "block-shard-openvino-20260918" / "block0_export.py"
BLOCK_INDEX = 6
BLOCK_PREFIX = f"blocks.{BLOCK_INDEX}."
EXPECTED_TENSOR_COUNT = 29
EXPECTED_PAYLOAD_BYTES = 317_266_986
OUT_XML = HERE / "block6_dynamic_window_reverse_gather_fp16.xml"
OUT_BIN = HERE / "block6_dynamic_window_reverse_gather_fp16.bin"
RECEIPT = HERE / "receipt.reverse_gather_block6.json"
RAW_LOG = HERE / "run.reverse_gather_block6.log"

VID_DIM = 2560
TXT_DIM = 2560
EMB_DIM = 15360
HEADS = 20
HEAD_DIM = 128
INNER_DIM = HEADS * HEAD_DIM
TXT_TOKENS = 58
WINDOW = (4, 3, 3)
WINDOW_METHOD = "720pwin_by_size_bysize"
SHARED_WEIGHTS = False
IS_LAST_LAYER = False
CASE_SHAPES = (
    (1, 48, 60),
    (1, 60, 60),
    (2, 48, 60),
    (1, 60, 90),
    (1, 90, 60),
)
PRODUCTION_MAX_WINDOWS = 338
PRODUCTION_MAX_VIDEO_LEN = 537
MAX_ABS_THRESHOLD = 1.0e-3
RMS_REL_THRESHOLD = 1.0e-4
SAVED_IR_MAX_ABS_THRESHOLD = 5.0e-3
SAVED_IR_RMS_REL_THRESHOLD = 1.0e-4
DISTINCT_THRESHOLD = 1.0e-7
INPUT_NAMES = (
    "vid",
    "txt",
    "emb",
    "window_idx",
    "window_key_mask",
    "window_rope_freqs",
    "window_reverse_idx",
)
OUTPUT_NAMES = ("vid_out", "txt_out")


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def filter_block_keys(keys: Iterable[str]) -> dict[str, str]:
    selected = sorted(key for key in keys if key.startswith(BLOCK_PREFIX))
    if len(selected) != EXPECTED_TENSOR_COUNT:
        raise AssertionError(
            f"block-{BLOCK_INDEX} tensor count mismatch: {len(selected)} != {EXPECTED_TENSOR_COUNT}"
        )
    mapping = {key: key[len(BLOCK_PREFIX) :] for key in selected}
    if any(not value for value in mapping.values()):
        raise AssertionError("block prefix stripping produced an invalid key")
    return mapping


def load_block_weights(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assert the shard contract before materializing any selected tensor."""

    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        mapping = filter_block_keys(handle.keys())
        payload_bytes = 0
        tensor_meta: list[dict[str, Any]] = []
        for source_key in mapping:
            view = handle.get_slice(source_key)
            shape = tuple(int(value) for value in view.get_shape())
            dtype = str(view.get_dtype())
            if dtype != "F16":
                raise AssertionError(f"{source_key} has unexpected dtype {dtype}")
            payload = math.prod(shape) * 2
            payload_bytes += payload
            tensor_meta.append(
                {
                    "source_key": source_key,
                    "stripped_key": mapping[source_key],
                    "shape": list(shape),
                    "dtype": dtype,
                    "payload_bytes": payload,
                }
            )
        if payload_bytes != EXPECTED_PAYLOAD_BYTES:
            raise AssertionError(
                f"block-{BLOCK_INDEX} payload mismatch: {payload_bytes} != {EXPECTED_PAYLOAD_BYTES}"
            )
        state = {mapping[key]: handle.get_tensor(key) for key in mapping}
    return state, {
        "block_index": BLOCK_INDEX,
        "prefix": BLOCK_PREFIX,
        "selected_tensor_count": len(mapping),
        "selected_payload_bytes": payload_bytes,
        "tensors": tensor_meta,
    }


def build_source_block(source: Path, state: Mapping[str, Any], torch: Any) -> Any:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from src.models.dit_3b.modulation import get_ada_layer
    from src.models.dit_3b.nablocks.mmsr_block import NaMMSRTransformerBlock
    from src.models.dit_3b.normalization import get_norm_layer

    norm = get_norm_layer("fusedrms")
    qk_norm = get_norm_layer("fusedrms")
    block = NaMMSRTransformerBlock(
        vid_dim=VID_DIM,
        txt_dim=TXT_DIM,
        emb_dim=EMB_DIM,
        heads=HEADS,
        head_dim=HEAD_DIM,
        expand_ratio=4,
        norm=norm,
        norm_eps=1.0e-5,
        ada=get_ada_layer("single"),
        qk_bias=False,
        qk_norm=qk_norm,
        mlp_type="swiglu",
        shared_weights=SHARED_WEIGHTS,
        rope_type="mmrope3d",
        rope_dim=128,
        is_last_layer=IS_LAST_LAYER,
        attention_mode="sdpa",
        window=WINDOW,
        window_method=WINDOW_METHOD,
    ).to(device="cpu", dtype=torch.float32).eval()
    expected_keys = set(block.state_dict())
    if set(state) != expected_keys:
        raise AssertionError(
            f"block state mismatch: missing={sorted(expected_keys - set(state))}, "
            f"unexpected={sorted(set(state) - expected_keys)}"
        )
    block.load_state_dict(dict(state), strict=True)
    return block


def peak_rss() -> dict[str, int]:
    kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"peak_rss_kib": kib, "peak_rss_bytes": kib * 1024}


def metrics(reference: Sequence[Any], candidate: Sequence[Any], np: Any) -> dict[str, Any]:
    per_output: list[dict[str, float]] = []
    for ref, got in zip(reference, candidate):
        ref_array = ref.detach().cpu().numpy()
        got_array = got.detach().cpu().numpy()
        diff = got_array.astype("float64") - ref_array.astype("float64")
        rms = float(np.sqrt(np.mean(np.square(diff))))
        ref_rms = float(np.sqrt(np.mean(np.square(ref_array))))
        per_output.append(
            {
                "max_abs_error": float(np.max(np.abs(diff))),
                "rms_relative_error": rms / max(ref_rms, 1.0e-12),
            }
        )
    return {
        "per_output": per_output,
        "max_abs_error": max(item["max_abs_error"] for item in per_output),
        "max_rms_relative_error": max(item["rms_relative_error"] for item in per_output),
    }


def _assert_finite(values: Sequence[Any], torch: Any, label: str) -> None:
    for value in values:
        if not bool(torch.isfinite(value).all().item()):
            raise AssertionError(f"{label}: non-finite output")


def attention_storage_elements(
    window_count: int, max_window_len: int, heads: int = HEADS
) -> int:
    """Conceptual score-storage elements for padded window attention."""

    if window_count <= 0 or max_window_len <= 0 or heads <= 0:
        raise ValueError("window_count, max_window_len, and heads must be positive")
    return int(window_count) * int(heads) * int(max_window_len) ** 2


def make_window_contract(
    shape: tuple[int, int, int],
    *,
    max_video_len: int,
    txt_len: int,
    rope: Any,
    torch: Any,
) -> dict[str, Any]:
    """Build source-order indices/masks/frequencies without model graph work."""

    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from src.models.dit_3b.window import get_window_op

    t, h, w = shape
    op = get_window_op(WINDOW_METHOD)
    slices = op(shape, WINDOW)
    index_grid = torch.arange(t * h * w, dtype=torch.long).reshape(t, h, w)
    indices = [index_grid[st, sh, sw].reshape(-1) for st, sh, sw in slices]
    lengths = [int(value.numel()) for value in indices]
    if max(lengths) > max_video_len:
        raise AssertionError(f"{shape}: {max(lengths)} exceeds {max_video_len}")

    vid_shape = torch.tensor([shape], dtype=torch.long)
    txt_shape = torch.tensor([[txt_len]], dtype=torch.long)
    # Upstream NaSwinAttention applies mmrope after partition and passes one
    # local [t,h,w] row per window.  In particular, video frequencies restart
    # at each window's local origin; indexing a full-frame table is not the
    # same operation.  Ask the trusted rope object for the complete repeated
    # local table, then place each row into its padded slot.
    local_shapes = torch.tensor(
        [
            [st.stop - st.start, sh.stop - sh.start, sw.stop - sw.start]
            for st, sh, sw in slices
        ],
        dtype=torch.long,
    )
    local_txt_shapes = torch.full((len(slices), 1), txt_len, dtype=torch.long)
    local_vid_freqs, local_txt_freqs = rope.get_freqs(local_shapes, local_txt_shapes)
    if tuple(local_vid_freqs.shape) != (sum(lengths), 126):
        raise AssertionError(f"unexpected local video RoPE shape: {tuple(local_vid_freqs.shape)}")
    if tuple(local_txt_freqs.shape) != (len(slices) * txt_len, 126):
        raise AssertionError(f"unexpected repeated text RoPE shape: {tuple(local_txt_freqs.shape)}")

    window_idx = torch.zeros((len(indices), max_video_len), dtype=torch.long)
    window_reverse_idx = torch.full((t * h * w,), -1, dtype=torch.long)
    key_mask = torch.zeros((len(indices), max_video_len + txt_len), dtype=torch.bool)
    rope_freqs = torch.zeros(
        (len(indices), max_video_len + txt_len, 126), dtype=torch.float32
    )
    video_freq_offset = 0
    text_freq_offset = 0
    for window_number, index in enumerate(indices):
        length = int(index.numel())
        window_idx[window_number, :length] = index
        padded_positions = (
            torch.arange(length, dtype=torch.long)
            + window_number * max_video_len
        )
        prior = window_reverse_idx.index_select(0, index)
        if bool((prior >= 0).any().item()):
            raise AssertionError(f"{shape}: duplicate source token in window reverse map")
        window_reverse_idx.index_copy_(0, index, padded_positions)
        key_mask[window_number, :length] = True
        key_mask[window_number, max_video_len : max_video_len + txt_len] = True
        rope_freqs[window_number, :length] = local_vid_freqs[
            video_freq_offset : video_freq_offset + length
        ]
        rope_freqs[window_number, max_video_len : max_video_len + txt_len] = local_txt_freqs[
            text_freq_offset : text_freq_offset + txt_len
        ]
        video_freq_offset += length
        text_freq_offset += txt_len

    if bool((window_reverse_idx < 0).any().item()):
        raise AssertionError(f"{shape}: missing source token in window reverse map")
    if int(torch.unique(window_reverse_idx).numel()) != t * h * w:
        raise AssertionError(f"{shape}: duplicate padded position in window reverse map")

    return {
        "vid_shape": vid_shape,
        "txt_shape": txt_shape,
        "window_idx": window_idx,
        "window_reverse_idx": window_reverse_idx,
        "window_key_mask": key_mask,
        "window_rope_freqs": rope_freqs,
        "window_count": len(indices),
        "window_lengths": lengths,
        "max_video_len": max_video_len,
        "max_attention_len": max_video_len + txt_len,
        "shape": list(shape),
    }


def _apply_batched_rope(x: Any, freqs: Any, torch: Any) -> Any:
    """Exact rotary_embedding_torch rotate-half operation over [N,L,H,D]."""

    dtype = x.dtype
    middle = x[..., : freqs.shape[-1]].float()
    pair = middle.reshape(*middle.shape[:-1], middle.shape[-1] // 2, 2)
    x1, x2 = pair.unbind(dim=-1)
    rotated = torch.stack((-x2, x1), dim=-1).reshape_as(middle)
    angles = freqs.float().unsqueeze(-2)
    transformed = middle * angles.cos() + rotated * angles.sin()
    return torch.cat((transformed, x[..., freqs.shape[-1] :]), dim=-1).to(dtype)


class DynamicWindowAttention:  # replaced below after torch is imported
    pass


def build_dynamic_classes(torch: Any) -> tuple[Any, Any, Any]:
    import torch.nn as nn
    import torch.nn.functional as F

    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from src.common.cache import Cache

    class _DynamicWindowAttention(nn.Module):
        def __init__(self, source_attention: Any):
            super().__init__()
            self.proj_qkv = source_attention.proj_qkv
            self.proj_out = source_attention.proj_out
            self.norm_q = source_attention.norm_q
            self.norm_k = source_attention.norm_k
            # Retain the trusted buffer in the module/state contract.  The
            # proof supplies the exact per-window frequencies externally.
            self.rope = source_attention.rope
            self.heads = HEADS
            self.head_dim = HEAD_DIM
            self.inner_dim = INNER_DIM

        def forward(
            self,
            vid: Any,
            txt: Any,
            window_idx: Any,
            window_key_mask: Any,
            window_rope_freqs: Any,
            window_reverse_idx: Any,
        ) -> tuple[Any, Any]:
            idx = window_idx
            key_mask = window_key_mask
            freqs = window_rope_freqs
            window_count = idx.shape[0]
            max_video_len = idx.shape[1]

            vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
            vid_qkv = vid_qkv.reshape(-1, 3, self.heads, self.head_dim)
            txt_qkv = txt_qkv.reshape(-1, 3, self.heads, self.head_dim)
            vid_q, vid_k, vid_v = vid_qkv.unbind(1)
            txt_q, txt_k, txt_v = txt_qkv.unbind(1)

            vid_q = vid_q.index_select(0, idx.reshape(-1)).reshape(
                window_count, max_video_len, self.heads, self.head_dim
            )
            vid_k = vid_k.index_select(0, idx.reshape(-1)).reshape(
                window_count, max_video_len, self.heads, self.head_dim
            )
            vid_v = vid_v.index_select(0, idx.reshape(-1)).reshape(
                window_count, max_video_len, self.heads, self.head_dim
            )

            vid_q, txt_q = self.norm_q(vid_q.reshape(-1, self.heads, self.head_dim), txt_q)
            vid_k, txt_k = self.norm_k(vid_k.reshape(-1, self.heads, self.head_dim), txt_k)
            vid_q = vid_q.reshape(window_count, max_video_len, self.heads, self.head_dim)
            vid_k = vid_k.reshape(window_count, max_video_len, self.heads, self.head_dim)
            txt_q = txt_q.unsqueeze(0).expand(window_count, -1, -1, -1)
            txt_k = txt_k.unsqueeze(0).expand(window_count, -1, -1, -1)
            txt_v = txt_v.unsqueeze(0).expand(window_count, -1, -1, -1)

            q = torch.cat((vid_q, txt_q), dim=1)
            k = torch.cat((vid_k, txt_k), dim=1)
            v = torch.cat((vid_v, txt_v), dim=1)
            q = _apply_batched_rope(q, freqs, torch)
            k = _apply_batched_rope(k, freqs, torch)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attn_mask = key_mask.unsqueeze(1).unsqueeze(1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
            out = out.permute(0, 2, 1, 3)

            vid_out = out[:, :max_video_len]
            # Every source token occurs exactly once in the valid, non-overlap
            # windows. Gather source-order outputs directly from the padded
            # window tensor; invalid padding rows are never selected.
            vid_out_flat = vid_out.reshape(-1, self.inner_dim).index_select(
                0, window_reverse_idx
            )

            txt_out = out[:, max_video_len : max_video_len + TXT_TOKENS]
            txt_out = txt_out.sum(dim=0) / window_count
            txt_out = txt_out.reshape(TXT_TOKENS, self.inner_dim)
            return self.proj_out(vid_out_flat, txt_out)

    class _DynamicBlock(nn.Module):
        def __init__(self, source_block: Any):
            super().__init__()
            self.attn_norm = source_block.attn_norm
            self.attn = _DynamicWindowAttention(source_block.attn)
            self.mlp_norm = source_block.mlp_norm
            self.mlp = source_block.mlp
            self.ada = source_block.ada
            self.is_last_layer = source_block.is_last_layer

        @staticmethod
        def _ada_branch(hid: Any, emb: Any, ada_module: Any, layer_index: int, mode: int) -> Any:
            # AdaSingle's upstream repeat_interleave is a Tile lowering that
            # leaves the CPU plug-in with a dynamic-rank operation.  This is
            # the same B=1 expansion expressed as a rank-static Broadcast:
            # the selected [1,C,3] embedding row is expanded to the actual
            # flattened token count.  No model parameter or numerical branch
            # is changed.
            emb_values = emb.reshape(1, ada_module.dim, 2, 3)[:, :, layer_index, :]
            emb_values = emb_values.expand(hid.shape[0], ada_module.dim, 3)
            shift_a, scale_a, gate_a = emb_values.unbind(-1)
            layer_name = "attn" if layer_index == 0 else "mlp"
            shift_b = getattr(ada_module, f"{layer_name}_shift")
            scale_b = getattr(ada_module, f"{layer_name}_scale")
            gate_b = getattr(ada_module, f"{layer_name}_gate")
            if mode == 0:
                return hid * (scale_a + scale_b) + (shift_a + shift_b)
            return hid * (gate_a + gate_b)

        def _ada_pair(self, vid: Any, txt: Any, emb: Any, layer_index: int, mode: int):
            if self.ada.shared_weights:
                if self.ada.vid_only:
                    return (
                        self._ada_branch(vid, emb, self.ada.all, layer_index, mode),
                        txt,
                    )
                return (
                    self._ada_branch(vid, emb, self.ada.all, layer_index, mode),
                    self._ada_branch(txt, emb, self.ada.all, layer_index, mode),
                )
            if self.ada.vid_only:
                return (
                    self._ada_branch(vid, emb, self.ada.vid, layer_index, mode),
                    txt,
                )
            return (
                self._ada_branch(vid, emb, self.ada.vid, layer_index, mode),
                self._ada_branch(txt, emb, self.ada.txt, layer_index, mode),
            )

        def forward(
            self,
            vid: Any,
            txt: Any,
            emb: Any,
            window_idx: Any,
            window_key_mask: Any,
            window_rope_freqs: Any,
            window_reverse_idx: Any,
        ) -> tuple[Any, Any]:
            vid_attn, txt_attn = self.attn_norm(vid, txt)
            vid_attn, txt_attn = self._ada_pair(vid_attn, txt_attn, emb, 0, 0)
            vid_attn, txt_attn = self.attn(
                vid_attn,
                txt_attn,
                window_idx,
                window_key_mask,
                window_rope_freqs,
                window_reverse_idx,
            )
            vid_attn, txt_attn = self._ada_pair(vid_attn, txt_attn, emb, 0, 1)
            vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)
            vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
            if vid_mlp.dtype != vid_attn.dtype:
                vid_mlp = vid_mlp.to(vid_attn.dtype)
            if txt_mlp.dtype != txt_attn.dtype:
                txt_mlp = txt_mlp.to(txt_attn.dtype)
            vid_mlp, txt_mlp = self._ada_pair(vid_mlp, txt_mlp, emb, 1, 0)
            vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
            vid_mlp, txt_mlp = self._ada_pair(vid_mlp, txt_mlp, emb, 1, 1)
            vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)
            return vid_mlp, txt_mlp

    class _TrustedWrapper(nn.Module):
        def __init__(self, source_block: Any):
            super().__init__()
            self.block = source_block

        def forward(self, vid: Any, txt: Any, vid_shape: Any, txt_shape: Any, emb: Any):
            return self.block(vid, txt, vid_shape, txt_shape, emb, Cache(disable=False))[:2]

    return _DynamicWindowAttention, _DynamicBlock, _TrustedWrapper


def build_inputs(contract: Mapping[str, Any], seed: int, torch: Any) -> tuple[Any, ...]:
    shape = tuple(int(value) for value in contract["shape"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video_tokens = math.prod(shape)
    vid = torch.randn((video_tokens, VID_DIM), generator=generator, dtype=torch.float32) * 0.02
    txt = torch.randn((TXT_TOKENS, TXT_DIM), generator=generator, dtype=torch.float32) * 0.02
    emb = torch.randn((1, EMB_DIM), generator=generator, dtype=torch.float32) * 0.02
    return (
        vid,
        txt,
        emb,
        contract["window_idx"],
        contract["window_key_mask"],
        contract["window_rope_freqs"],
        contract["window_reverse_idx"],
    )


def input_contract(values: Sequence[Any]) -> list[dict[str, Any]]:
    names = (
        "vid",
        "txt",
        "emb",
        "window_idx",
        "window_key_mask",
        "window_rope_freqs",
        "window_reverse_idx",
    )
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in zip(names, values)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--max-threads", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.max_threads))

    receipt: dict[str, Any] = {
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "dynamic-window-reverse-gather-block6-openvino-20260918",
        "execution": {
            "device": "CPU",
            "gpu_inspection": False,
            "advisory_lock": "not inspected/acquired per explicit CPU-only session instruction",
            "thread_cap": args.max_threads,
        },
        "host_coordination": {
            "load_sample_command": "vmstat 1 3",
            "nproc": 16,
            "sample_idle_percent": [89, 80, 80],
            "lock": "not inspected/acquired per explicit CPU-only instruction",
            "gpu": "not inspected/used",
        },
        "commands": {
            "proof": f"OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 {sys.executable} {Path(__file__).name}",
            "focused_tests": f"{sys.executable} test_dynamic_window_length_block6_contract.py -v",
            "openvino_runtime": str(OV_ROOT),
        },
        "thresholds": {
            "max_abs_error": MAX_ABS_THRESHOLD,
            "rms_relative_error": RMS_REL_THRESHOLD,
            "saved_ir_max_abs_error": SAVED_IR_MAX_ABS_THRESHOLD,
            "saved_ir_rms_relative_error": SAVED_IR_RMS_REL_THRESHOLD,
            "distinct_output_delta": DISTINCT_THRESHOLD,
        },
        "source": {"path": str(SOURCE), "trusted_helper": str(TRUSTED_HELPER)},
        "weights": {"path": str(WEIGHTS), "sha256": sha256_file(WEIGHTS)},
        "model_args": {
            "vid_dim": VID_DIM,
            "txt_dim": TXT_DIM,
            "emb_dim": EMB_DIM,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "expand_ratio": 4,
            "norm": "fusedrms",
            "norm_eps": 1.0e-5,
            "ada": "single",
            "qk_bias": False,
            "qk_norm": "fusedrms",
            "mlp_type": "swiglu",
            "rope_type": "mmrope3d",
            "rope_dim": 128,
            "attention_mode": "sdpa",
            "window": list(WINDOW),
            "window_method": WINDOW_METHOD,
            "shared_weights": SHARED_WEIGHTS,
            "is_last_layer": IS_LAST_LAYER,
            "block_index": BLOCK_INDEX,
            "shifted": False,
        },
        "lowering": {
            "attention": "batched padded windows with external index/key-mask/local-mmrope inputs",
            "video_reverse": "exact inverse gather from source-order token to unique valid padded-window position",
            "text_pooling": "mean over dynamic window-count dimension",
            "padding": "per-window video padding only; text occupies fixed tail slots",
            "ada": "batch-1 rank-static broadcast equivalent of upstream repeat_interleave",
            "window_count_axis": "dynamic input dimension; no reshape/recompile between cases",
            "window_length_axis": "dynamic input dimension; per-case actual maximum padding only",
            "reverse_index_axis": "dynamic source-token length; one valid occurrence per source token",
            "text_output": "preserved source block text output",
        },
        "cases": {},
        "events": [],
    }
    start_wall = time.perf_counter()
    try:
        import numpy as np
        import torch

        torch.set_num_threads(args.max_threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        if hasattr(torch, "xpu"):
            torch.xpu.is_available = lambda: False
            if hasattr(torch.xpu, "device_count"):
                torch.xpu.device_count = lambda: 0
        receipt["runtime"] = {
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        }
        helper = _load_module(TRUSTED_HELPER, "trusted_block_helper_for_dynamic")
        receipt["source"]["trusted_helper_sha256"] = sha256_file(TRUSTED_HELPER)
        receipt["source"]["relevant_source_sha256"] = helper._source_hash(SOURCE)[0]
        state, shard_meta = load_block_weights(WEIGHTS)
        receipt["weights"].update(shard_meta)

        # Build one trusted block and one lowered block from the same selected
        # state.  The state contract is checked before either construction.
        trusted_block = build_source_block(SOURCE, state, torch)
        dynamic_source_block = build_source_block(SOURCE, state, torch)
        _, DynamicBlock, TrustedWrapper = build_dynamic_classes(torch)
        trusted = TrustedWrapper(trusted_block).eval()
        lowered = DynamicBlock(dynamic_source_block).eval()
        if set(trusted_block.state_dict()) != set(lowered.state_dict()):
            raise AssertionError("dynamic wrapper changed the trusted state-key contract")
        receipt["state_contract"] = {
            "trusted_state_keys": len(trusted_block.state_dict()),
            "dynamic_state_keys": len(lowered.state_dict()),
            "matching": set(trusted_block.state_dict()) == set(lowered.state_dict()),
        }

        # Build all contracts after the trusted rope object exists.  The saved
        # graph keeps the window-count and per-window length dimensions
        # dynamic; the production cap below is an audit bound only.
        preliminary = []
        case_max_lengths: dict[tuple[int, int, int], int] = {}
        max_windows = PRODUCTION_MAX_WINDOWS
        max_video_len = PRODUCTION_MAX_VIDEO_LEN
        audit_shapes = (
            (2, 240, 240),
            (2, 135, 240),
            (2, 240, 135),
            (1, 1, 240),
            (1, 240, 1),
            (2, 1, 240),
            (2, 240, 1),
        )
        audit = []
        for shape in tuple(CASE_SHAPES) + tuple(audit_shapes):
            if str(SOURCE) not in sys.path:
                sys.path.insert(0, str(SOURCE))
            from src.models.dit_3b.window import get_window_op

            slices = get_window_op(WINDOW_METHOD)(shape, WINDOW)
            lengths = [
                (st.stop - st.start) * (sh.stop - sh.start) * (sw.stop - sw.start)
                for st, sh, sw in slices
            ]
            if len(slices) > max_windows or max(lengths) > max_video_len:
                raise AssertionError(
                    f"production window cap exceeded for {shape}: "
                    f"count={len(slices)} max_len={max(lengths)} "
                    f"caps={max_windows}/{max_video_len}"
                )
            item = {"shape": list(shape), "window_count": len(slices), "max_video_len": max(lengths)}
            audit.append(item)
            if shape in CASE_SHAPES:
                preliminary.append((shape, len(slices), max(lengths)))
                case_max_lengths[shape] = max(lengths)
        receipt["window_contract"] = {
            "window": list(WINDOW),
            "window_method": WINDOW_METHOD,
            "runtime_window_count_axis": "dynamic",
            "runtime_video_window_length_axis": "dynamic",
            "production_audit_max_windows": max_windows,
            "production_audit_max_video_len": max_video_len,
            "score_storage_elements_per_head_at_max": max_windows
            * (max_video_len + TXT_TOKENS) ** 2,
            "score_storage_elements_at_max": attention_storage_elements(
                max_windows, max_video_len + TXT_TOKENS
            ),
            "total_packed_tokens_at_max": max_windows * (max_video_len + TXT_TOKENS),
            "total_packed_square_elements_at_max": (
                max_windows * (max_video_len + TXT_TOKENS)
            )
            ** 2,
            "preliminary": [{"shape": list(s), "window_count": n, "max_video_len": m} for s, n, m in preliminary],
            "case_max_video_lengths": {str(k): v for k, v in case_max_lengths.items()},
            "production_audit": audit,
        }
        contracts: dict[tuple[int, int, int], dict[str, Any]] = {}
        for shape in CASE_SHAPES:
            contracts[shape] = make_window_contract(
                shape,
                max_video_len=case_max_lengths[shape],
                txt_len=TXT_TOKENS,
                rope=dynamic_source_block.attn.rope,
                torch=torch,
            )
            c = contracts[shape]
            c["storage_elements"] = attention_storage_elements(
                c["window_count"], c["max_attention_len"]
            )
            c["packed_tokens"] = sum(c["window_lengths"]) + c["window_count"] * TXT_TOKENS
            c["packed_square_elements"] = c["packed_tokens"] ** 2
            receipt["cases"][str(shape)] = {
                "shape": list(shape),
                "metadata_shapes": {
                    "vid_shape": list(shape),
                    "txt_shape": [1, TXT_TOKENS],
                },
                "window_count": c["window_count"],
                "window_lengths": c["window_lengths"],
                "window_length_max": max(c["window_lengths"]),
                "input_contract": None,
                "eager": {},
                "openvino": {},
                "storage": {
                    "window_score_elements_per_head": c["window_count"]
                    * c["max_attention_len"] ** 2,
                    "window_score_elements": c["storage_elements"],
                    "packed_tokens": c["packed_tokens"],
                    "packed_square_elements": c["packed_square_elements"],
                },
            }

        # Exact trusted-vs-lowered eager comparison at each requested metadata
        # shape.  This occurs before any OpenVINO conversion.
        seeds = {
            shape: 2026091801 + index
            for index, shape in enumerate(CASE_SHAPES)
        }
        eager_outputs: dict[tuple[int, int, int], tuple[tuple[Any, Any], tuple[Any, ...]]] = {}
        with torch.inference_mode():
            for shape in CASE_SHAPES:
                contract = contracts[shape]
                inputs = build_inputs(contract, seeds[shape], torch)
                receipt["cases"][str(shape)]["input_contract"] = input_contract(inputs)
                t0 = time.perf_counter()
                trusted_out = trusted(
                    inputs[0],
                    inputs[1],
                    contract["vid_shape"],
                    contract["txt_shape"],
                    inputs[2],
                )
                trusted_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                lowered_out = lowered(*inputs)
                lowered_s = time.perf_counter() - t0
                _assert_finite(trusted_out, torch, f"trusted {shape}")
                _assert_finite(lowered_out, torch, f"lowered {shape}")
                eager_metric = metrics(trusted_out, lowered_out, np)
                if (
                    eager_metric["max_abs_error"] > MAX_ABS_THRESHOLD
                    or eager_metric["max_rms_relative_error"] > RMS_REL_THRESHOLD
                ):
                    raise AssertionError(f"trusted-vs-lowered eager threshold failure {shape}: {eager_metric}")
                receipt["cases"][str(shape)]["eager"] = {
                    "trusted_seconds": trusted_s,
                    "lowered_seconds": lowered_s,
                    "metrics": eager_metric,
                    "finite": True,
                }
                eager_outputs[shape] = (lowered_out, inputs)
                del trusted_out, lowered_out
                gc.collect()

        # Trace the lowered wrapper once.  The window-count dimension is
        # reshaped dynamic below; only max per-window padding is fixed.
        example_shape = CASE_SHAPES[0]
        example_inputs = eager_outputs[example_shape][1]
        trace_start = time.perf_counter()
        traced = torch.jit.trace(lowered, example_inputs, strict=False, check_trace=False)
        frozen = torch.jit.freeze(traced.eval())
        receipt["trace_seconds"] = time.perf_counter() - trace_start
        receipt["trace_graph"] = {
            "graph_chars": len(str(frozen.graph)),
            "contains_scaled_dot_product_attention": "scaled_dot_product_attention" in str(frozen.graph),
            "contains_index_add": "index_add" in str(frozen.graph),
            "contains_index_select": "index_select" in str(frozen.graph),
            "contains_nonzero": "nonzero" in str(frozen.graph),
        }
        if args.skip_convert:
            receipt["status"] = "skipped_conversion"
            receipt["finished_seconds"] = time.perf_counter() - start_wall
            RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            return 0

        import openvino as ov

        convert_start = time.perf_counter()
        ov_model = ov.convert_model(frozen, example_input=example_inputs)
        receipt["convert_seconds"] = time.perf_counter() - convert_start
        # Give every port a stable semantic name before saving.  Inference
        # below feeds dictionaries by these names, never by incidental port
        # order.
        if len(ov_model.inputs) != len(INPUT_NAMES):
            raise AssertionError(
                f"converted input count mismatch: {len(ov_model.inputs)} != {len(INPUT_NAMES)}"
            )
        for port, name in zip(ov_model.inputs, INPUT_NAMES):
            port.get_tensor().set_names({name})
        if len(ov_model.outputs) != len(OUTPUT_NAMES):
            raise AssertionError(
                f"converted output count mismatch: {len(ov_model.outputs)} != {len(OUTPUT_NAMES)}"
            )
        for port, name in zip(ov_model.outputs, OUTPUT_NAMES):
            port.get_tensor().set_names({name})
        reshape_map = {}
        for port in ov_model.inputs:
            name = port.get_any_name()
            if name == "vid":
                reshape_map[port] = ov.PartialShape([-1, VID_DIM])
            elif name == "txt":
                reshape_map[port] = ov.PartialShape([TXT_TOKENS, TXT_DIM])
            elif name == "emb":
                reshape_map[port] = ov.PartialShape([1, EMB_DIM])
            elif name == "window_idx":
                reshape_map[port] = ov.PartialShape([-1, -1])
            elif name == "window_key_mask":
                reshape_map[port] = ov.PartialShape([-1, -1])
            elif name == "window_rope_freqs":
                reshape_map[port] = ov.PartialShape([-1, -1, 126])
            elif name == "window_reverse_idx":
                reshape_map[port] = ov.PartialShape([-1])
        if set(port.get_any_name() for port in ov_model.inputs) != set(INPUT_NAMES):
            raise AssertionError("converted input names do not match the declared contract")
        ov_model.reshape(reshape_map)
        ov.save_model(ov_model, str(OUT_XML), compress_to_fp16=True)
        receipt["artifact"] = {
            "xml": str(OUT_XML),
            "bin": str(OUT_BIN),
            "precision": "FP16-compressed-OpenVINO-IR",
            "compress_to_fp16": True,
            "xml_bytes": OUT_XML.stat().st_size,
            "bin_bytes": OUT_BIN.stat().st_size,
            "total_bytes": OUT_XML.stat().st_size + OUT_BIN.stat().st_size,
            "bin_below_400mb": OUT_BIN.stat().st_size < 400 * 1024 * 1024,
        }
        receipt["openvino_io"] = [
            {
                "name": sorted(port.get_names())[0] if port.get_names() else port.get_any_name(),
                "shape": str(port.get_partial_shape()),
                "element_type": str(port.get_element_type()),
            }
            for port in ov_model.inputs
        ]
        receipt["openvino_outputs"] = [
            {
                "name": sorted(port.get_names())[0] if port.get_names() else port.get_any_name(),
                "shape": str(port.get_partial_shape()),
                "element_type": str(port.get_element_type()),
            }
            for port in ov_model.outputs
        ]

        core = ov.Core()
        compile_start = time.perf_counter()
        compiled = core.compile_model(ov_model, "CPU", {"INFERENCE_NUM_THREADS": args.max_threads})
        receipt["compile_seconds"] = time.perf_counter() - compile_start

        # Validate all three metadata contracts on the same compiled model.
        # The first and last are also used for the distinct-output check.
        ir_cases = CASE_SHAPES
        for case_number, shape in enumerate(ir_cases):
            seed = seeds[shape] + 100
            contract = contracts[shape]
            inputs = build_inputs(contract, seed, torch)
            with torch.inference_mode():
                lowered_ref = lowered(*inputs)
            _assert_finite(lowered_ref, torch, f"lowered IR reference {shape}")
            infer_inputs = {
                name: value.detach().cpu().numpy()
                for name, value in zip(INPUT_NAMES, inputs)
            }
            expected_by_name = {
                name: tuple(int(d) for d in value.shape)
                for name, value in zip(INPUT_NAMES, inputs)
            }
            for name, array in infer_inputs.items():
                if tuple(int(d) for d in array.shape) != expected_by_name[name]:
                    raise AssertionError(
                        f"runtime input shape mismatch for {name}: {array.shape} vs {expected_by_name[name]}"
                    )
            for port in compiled.inputs:
                name = port.any_name
                if name not in expected_by_name:
                    raise AssertionError(f"compiled input was not named by contract: {name}")
                partial = port.partial_shape
                if partial.rank.is_static:
                    if int(partial.rank.get_length()) != len(expected_by_name[name]):
                        raise AssertionError(
                            f"compiled input rank mismatch for {name}: {partial} vs {expected_by_name[name]}"
                        )
                    for dimension, expected_dimension in zip(partial, expected_by_name[name]):
                        if dimension.is_static and int(dimension.get_length()) != expected_dimension:
                            raise AssertionError(
                                f"compiled input contract mismatch for {name}: {partial} vs {expected_by_name[name]}"
                            )
            t0 = time.perf_counter()
            result = compiled(infer_inputs)
            cold_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            result_warm = compiled(infer_inputs)
            warm_seconds = time.perf_counter() - t0
            got = tuple(result[compiled.outputs[index]] for index in range(len(compiled.outputs)))
            got_warm = tuple(
                result_warm[compiled.outputs[index]] for index in range(len(compiled.outputs))
            )
            got_torch = tuple(torch.from_numpy(value) for value in got)
            got_warm_torch = tuple(torch.from_numpy(value) for value in got_warm)
            _assert_finite(got_torch, torch, f"OpenVINO {shape}")
            metric = metrics(lowered_ref, got_torch, np)
            warm_metric = metrics(got_torch, got_warm_torch, np)
            if (
                metric["max_abs_error"] > SAVED_IR_MAX_ABS_THRESHOLD
                or metric["max_rms_relative_error"] > SAVED_IR_RMS_REL_THRESHOLD
            ):
                raise AssertionError(f"saved IR threshold failure {shape}: {metric}")
            if warm_metric["max_abs_error"] > SAVED_IR_MAX_ABS_THRESHOLD:
                raise AssertionError(f"saved IR cold/warm instability {shape}: {warm_metric}")
            receipt["cases"][str(shape)]["openvino"] = {
                "seed": seed,
                "cold_seconds": cold_seconds,
                "warm_seconds": warm_seconds,
                "metrics_vs_lowered_eager": metric,
                "warm_repeat_metrics": warm_metric,
                "finite": True,
                "output_stats": [
                    {
                        "mean": float(value.mean().item()),
                        "rms": float(torch.sqrt(torch.mean(value * value)).item()),
                        "max_abs": float(value.abs().max().item()),
                    }
                    for value in got_torch
                ],
            }
            del lowered_ref, result, result_warm, got, got_warm, got_torch, got_warm_torch
            gc.collect()

        distinct_cases = (ir_cases[0], ir_cases[-1])
        signatures = []
        for shape in distinct_cases:
            output_stats = receipt["cases"][str(shape)]["openvino"]["output_stats"]
            signatures.append(
                [field for stats in output_stats for field in (stats["mean"], stats["rms"])]
            )
        distinct_delta = max(
            abs(left - right)
            for left, right in zip(signatures[0], signatures[1])
        )
        receipt["openvino_distinct_check"] = {
            "cases": [list(distinct_cases[0]), list(distinct_cases[1])],
            "signature_delta": distinct_delta,
            "threshold": DISTINCT_THRESHOLD,
            "finite": True,
            "distinct": distinct_delta > DISTINCT_THRESHOLD,
        }
        if distinct_delta <= DISTINCT_THRESHOLD:
            raise AssertionError(
                f"saved IR deterministic cases are not distinct: delta={distinct_delta}"
            )

        receipt["status"] = "pass"
        receipt["finished_seconds"] = time.perf_counter() - start_wall
        receipt.update(peak_rss())
        RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception as exc:
        receipt["status"] = "failure"
        receipt["finished_seconds"] = time.perf_counter() - start_wall
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        receipt.update(peak_rss())
        RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
