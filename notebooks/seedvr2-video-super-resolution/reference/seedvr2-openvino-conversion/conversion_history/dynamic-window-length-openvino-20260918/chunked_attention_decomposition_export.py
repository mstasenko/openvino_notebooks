#!/usr/bin/env python3
"""CPU proof of an exact chunked SeedVR2 window-attention decomposition.

The learned block is split into three graphs:

* prefix: attention norm, Ada attention-in, QKV projection, and Q/K norm;
* chunk: weightless padded-window attention for a bounded number of windows,
  returning per-window video values plus a text sum/count;
* suffix: text mean, output projection, Ada attention-out, residual, and the
  exact MLP path.

The caller processes arbitrary chunks of the non-overlapping source windows,
inverse-gathers each video token exactly once, and averages text exactly once.
All graph ports are named and the saved IRs use FP16-compressed constants with
FP32 public I/O where applicable.
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
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TRIAL_ROOT = HERE.parents[1]
SOURCE = Path(os.environ.get("SEEDVR_SOURCE", TRIAL_ROOT / "source_snapshot"))
WEIGHTS = Path(os.environ.get("SEEDVR_WEIGHTS", TRIAL_ROOT / "seedvr2_ema_3b_fp16.safetensors"))
OV_ROOT = Path(os.environ.get("SEEDVR_OPENVINO_ROOT", TRIAL_ROOT / "runtime-openvino"))
TRUSTED_HELPER = HERE.parent / "block-shard-openvino-20260918" / "block0_export.py"

VID_DIM = 2560
TXT_DIM = 2560
EMB_DIM = 15360
HEADS = 20
HEAD_DIM = 128
INNER_DIM = HEADS * HEAD_DIM
TXT_TOKENS = 58
WINDOW = (4, 3, 3)
CASE_SHAPES = (
    (1, 48, 60),
    (1, 60, 60),
    (2, 48, 60),
    (1, 60, 90),
    (1, 90, 60),
    (2, 135, 240),
)
PRODUCTION_SHAPES = (
    (2, 240, 240),
    (2, 135, 240),
    (2, 240, 135),
    (1, 1, 240),
    (1, 240, 1),
    (2, 1, 240),
    (2, 240, 1),
)
PRODUCTION_MAX_WINDOWS = 338
PRODUCTION_MAX_VIDEO_LEN = 537
CHUNK_SIZE_PROOF = 16
MAX_ABS_THRESHOLD = 1.0e-3
RMS_REL_THRESHOLD = 1.0e-4
SAVED_IR_MAX_ABS_THRESHOLD = 5.0e-3
SAVED_IR_RMS_REL_THRESHOLD = 1.0e-4
DISTINCT_THRESHOLD = 1.0e-7


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


def sample_host_cpu_load() -> dict[str, Any]:
    result = subprocess.run(
        ["vmstat", "1", "3"], check=True, capture_output=True, text=True
    )
    rows = [line.split() for line in result.stdout.splitlines() if line.split()]
    header = next(
        (row for row in rows if "id" in row and not row[0].isdigit()), None
    )
    if header is None:
        raise RuntimeError("vmstat CPU header was unavailable")
    idle_index = header.index("id")
    samples = [
        row
        for row in rows
        if row[0].isdigit()
        and len(row) > idle_index
        and row[idle_index].isdigit()
    ]
    if len(samples) < 3:
        raise RuntimeError(f"vmstat returned {len(samples)} CPU rows; expected 3")
    return {
        "sample_command": "vmstat 1 3",
        "nproc": os.cpu_count() or 0,
        "sample_idle_percent": [int(row[idle_index]) for row in samples[-3:]],
        "sampled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


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


def attention_score_elements(
    window_count: int,
    max_video_len: int,
    *,
    chunk_windows: int | None = None,
    heads: int = HEADS,
    txt_tokens: int = TXT_TOKENS,
) -> int:
    """Peak attention score elements for padded windows, not packed-token²."""

    if window_count <= 0 or max_video_len <= 0 or heads <= 0 or txt_tokens < 0:
        raise ValueError("invalid attention storage dimensions")
    active = window_count if chunk_windows is None else min(window_count, chunk_windows)
    return int(active) * int(heads) * (int(max_video_len) + int(txt_tokens)) ** 2


def window_metadata(base: Any, shape: tuple[int, int, int]) -> dict[str, Any]:
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from src.models.dit_3b.window import get_window_op

    slices = get_window_op(base.WINDOW_METHOD)(shape, base.WINDOW)
    lengths = [
        (st.stop - st.start) * (sh.stop - sh.start) * (sw.stop - sw.start)
        for st, sh, sw in slices
    ]
    return {
        "shape": list(shape),
        "window_count": len(slices),
        "max_video_len": max(lengths),
        "window_lengths": lengths,
    }


def split_contract(contract: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    """Return one chunk's metadata while retaining the full per-case padding L."""

    return {
        "window_idx": contract["window_idx"][start:end],
        "window_key_mask": contract["window_key_mask"][start:end],
        "window_rope_freqs": contract["window_rope_freqs"][start:end],
        "start": start,
        "end": end,
        "window_count": end - start,
        "max_video_len": int(contract["max_video_len"]),
    }


def validate_window_coverage(contract: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    """Require exactly one valid occurrence of every source-order video token."""

    idx = contract["window_idx"]
    mask = contract["window_key_mask"][:, : int(contract["max_video_len"])]
    source_tokens = int(contract["vid_shape"].prod().item())
    valid_source = idx[:, : idx.shape[1]][mask]
    occurrences = torch.bincount(valid_source, minlength=source_tokens)
    if int(occurrences.numel()) != source_tokens:
        raise AssertionError("window coverage bincount length mismatch")
    if not bool(torch.equal(occurrences, torch.ones_like(occurrences))):
        raise AssertionError("window coverage has a duplicate or missing source token")
    reverse = contract["window_reverse_idx"]
    if tuple(reverse.shape) != (source_tokens,):
        raise AssertionError("window reverse index has the wrong source-token length")
    if int(torch.unique(reverse).numel()) != source_tokens:
        raise AssertionError("window reverse index has duplicate padded positions")
    flat_valid = mask.reshape(-1)
    if not bool(flat_valid.index_select(0, reverse).all().item()):
        raise AssertionError("window reverse index selects padding")
    return {
        "source_tokens": source_tokens,
        "valid_occurrence_min": int(occurrences.min().item()),
        "valid_occurrence_max": int(occurrences.max().item()),
        "reverse_positions_unique": True,
    }


def _apply_batched_rope(x: Any, freqs: Any, torch: Any) -> Any:
    dtype = x.dtype
    middle = x[..., : freqs.shape[-1]].float()
    pair = middle.reshape(*middle.shape[:-1], middle.shape[-1] // 2, 2)
    x1, x2 = pair.unbind(dim=-1)
    rotated = torch.stack((-x2, x1), dim=-1).reshape_as(middle)
    angles = freqs.float().unsqueeze(-2)
    transformed = middle * angles.cos() + rotated * angles.sin()
    return torch.cat((transformed, x[..., freqs.shape[-1] :]), dim=-1).to(dtype)


def _ada_branch(hid: Any, emb: Any, ada_module: Any, branch_name: str, layer_index: int, mode: int) -> Any:
    if branch_name == "txt" and bool(getattr(ada_module, "vid_only", False)):
        return hid
    if ada_module.shared_weights:
        branch = ada_module.all
    else:
        branch = getattr(ada_module, branch_name)
    if branch is None:
        return hid
    emb_values = emb.reshape(1, branch.dim, 2, 3)[:, :, layer_index, :]
    emb_values = emb_values.expand(hid.shape[0], branch.dim, 3)
    shift_a, scale_a, gate_a = emb_values.unbind(-1)
    shift_b = getattr(branch, "attn_shift" if layer_index == 0 else "mlp_shift", None)
    scale_b = getattr(branch, "attn_scale" if layer_index == 0 else "mlp_scale", None)
    gate_b = getattr(branch, "attn_gate" if layer_index == 0 else "mlp_gate", None)
    if mode == 0:
        return hid * (scale_a + scale_b) + (shift_a + shift_b)
    return hid * (gate_a + gate_b) if gate_b is not None else hid * gate_a


def _ada_pair(vid: Any, txt: Any, emb: Any, ada_module: Any, layer_index: int, mode: int) -> tuple[Any, Any]:
    return (
        _ada_branch(vid, emb, ada_module, "vid", layer_index, mode),
        _ada_branch(txt, emb, ada_module, "txt", layer_index, mode),
    )


def build_chunk_module(torch: Any) -> Any:
    """Build the weightless chunk graph for focused contract tests."""

    import torch.nn as nn
    import torch.nn.functional as F

    class Chunk(nn.Module):
        def forward(
            self,
            vid_q: Any,
            vid_k: Any,
            vid_v: Any,
            txt_q: Any,
            txt_k: Any,
            txt_v: Any,
            window_idx: Any,
            window_key_mask: Any,
            window_rope_freqs: Any,
        ) -> tuple[Any, Any, Any]:
            window_count = window_idx.shape[0]
            max_video_len = window_idx.shape[1]
            flat_idx = window_idx.reshape(-1)
            vid_q = vid_q.index_select(0, flat_idx).reshape(window_count, max_video_len, HEADS, HEAD_DIM)
            vid_k = vid_k.index_select(0, flat_idx).reshape(window_count, max_video_len, HEADS, HEAD_DIM)
            vid_v = vid_v.index_select(0, flat_idx).reshape(window_count, max_video_len, HEADS, HEAD_DIM)
            txt_q = txt_q.unsqueeze(0).expand(window_count, -1, -1, -1)
            txt_k = txt_k.unsqueeze(0).expand(window_count, -1, -1, -1)
            txt_v = txt_v.unsqueeze(0).expand(window_count, -1, -1, -1)
            q = torch.cat((vid_q, txt_q), dim=1)
            k = torch.cat((vid_k, txt_k), dim=1)
            v = torch.cat((vid_v, txt_v), dim=1)
            q = _apply_batched_rope(q, window_rope_freqs, torch)
            k = _apply_batched_rope(k, window_rope_freqs, torch)
            attn_mask = window_key_mask.unsqueeze(1).unsqueeze(1)
            out = F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            ).permute(0, 2, 1, 3)
            vid_out = out[:, :max_video_len].reshape(window_count, max_video_len, INNER_DIM)
            txt_out = out[:, max_video_len : max_video_len + TXT_TOKENS].reshape(
                window_count, TXT_TOKENS, INNER_DIM
            )
            txt_sum = txt_out.sum(dim=0)
            txt_count = window_key_mask[:, -TXT_TOKENS].to(txt_sum.dtype).sum().reshape(1)
            return vid_out, txt_sum, txt_count

    return Chunk().eval()


def build_decomposition_modules(torch: Any, source_block: Any) -> tuple[Any, Any, Any]:
    import torch.nn as nn
    import torch.nn.functional as F

    class Prefix(nn.Module):
        def __init__(self, block: Any):
            super().__init__()
            self.attn_norm = block.attn_norm
            self.attn = block.attn
            self.ada = block.ada

        def forward(self, vid: Any, txt: Any, emb: Any) -> tuple[Any, ...]:
            vid_h, txt_h = self.attn_norm(vid, txt)
            vid_h, txt_h = _ada_pair(vid_h, txt_h, emb, self.ada, 0, 0)
            vid_qkv, txt_qkv = self.attn.proj_qkv(vid_h, txt_h)
            vid_qkv = vid_qkv.reshape(-1, 3, HEADS, HEAD_DIM)
            txt_qkv = txt_qkv.reshape(-1, 3, HEADS, HEAD_DIM)
            vid_q, vid_k, vid_v = vid_qkv.unbind(1)
            txt_q, txt_k, txt_v = txt_qkv.unbind(1)
            vid_q, txt_q = self.attn.norm_q(vid_q, txt_q)
            vid_k, txt_k = self.attn.norm_k(vid_k, txt_k)
            return vid_q, vid_k, vid_v, txt_q, txt_k, txt_v

    class Chunk(nn.Module):
        def forward(
            self,
            vid_q: Any,
            vid_k: Any,
            vid_v: Any,
            txt_q: Any,
            txt_k: Any,
            txt_v: Any,
            window_idx: Any,
            window_key_mask: Any,
            window_rope_freqs: Any,
        ) -> tuple[Any, Any, Any]:
            window_count = window_idx.shape[0]
            max_video_len = window_idx.shape[1]
            flat_idx = window_idx.reshape(-1)
            vid_q = vid_q.index_select(0, flat_idx).reshape(
                window_count, max_video_len, HEADS, HEAD_DIM
            )
            vid_k = vid_k.index_select(0, flat_idx).reshape(
                window_count, max_video_len, HEADS, HEAD_DIM
            )
            vid_v = vid_v.index_select(0, flat_idx).reshape(
                window_count, max_video_len, HEADS, HEAD_DIM
            )
            txt_q = txt_q.unsqueeze(0).expand(window_count, -1, -1, -1)
            txt_k = txt_k.unsqueeze(0).expand(window_count, -1, -1, -1)
            txt_v = txt_v.unsqueeze(0).expand(window_count, -1, -1, -1)
            q = torch.cat((vid_q, txt_q), dim=1)
            k = torch.cat((vid_k, txt_k), dim=1)
            v = torch.cat((vid_v, txt_v), dim=1)
            q = _apply_batched_rope(q, window_rope_freqs, torch)
            k = _apply_batched_rope(k, window_rope_freqs, torch)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attn_mask = window_key_mask.unsqueeze(1).unsqueeze(1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            ).permute(0, 2, 1, 3)
            vid_out = out[:, :max_video_len].reshape(window_count, max_video_len, INNER_DIM)
            txt_out = out[:, max_video_len : max_video_len + TXT_TOKENS].reshape(
                window_count, TXT_TOKENS, INNER_DIM
            )
            txt_sum = txt_out.sum(dim=0)
            txt_count = (
                window_key_mask[:, -TXT_TOKENS]
                .to(txt_sum.dtype)
                .sum()
                .reshape(1)
            )
            return vid_out, txt_sum, txt_count

    class Suffix(nn.Module):
        def __init__(self, block: Any):
            super().__init__()
            self.proj_out = block.attn.proj_out
            self.ada = block.ada
            self.mlp_norm = block.mlp_norm
            self.mlp = block.mlp

        def forward(
            self,
            vid: Any,
            txt: Any,
            emb: Any,
            attn_vid: Any,
            attn_txt: Any,
        ) -> tuple[Any, Any]:
            vid_attn, txt_attn = self.proj_out(attn_vid, attn_txt)
            vid_attn, txt_attn = _ada_pair(vid_attn, txt_attn, emb, self.ada, 0, 1)
            vid_attn, txt_attn = vid_attn + vid, txt_attn + txt
            vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
            if vid_mlp.dtype != vid_attn.dtype:
                vid_mlp = vid_mlp.to(vid_attn.dtype)
            if txt_mlp.dtype != txt_attn.dtype:
                txt_mlp = txt_mlp.to(txt_attn.dtype)
            vid_mlp, txt_mlp = _ada_pair(vid_mlp, txt_mlp, emb, self.ada, 1, 0)
            vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
            vid_mlp, txt_mlp = _ada_pair(vid_mlp, txt_mlp, emb, self.ada, 1, 1)
            return vid_mlp + vid_attn, txt_mlp + txt_attn

    return Prefix(source_block).eval(), Chunk().eval(), Suffix(source_block).eval()


def _trusted_wrapper(source_block: Any, torch: Any) -> Any:
    import torch.nn as nn
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from src.common.cache import Cache

    class Trusted(nn.Module):
        def __init__(self, block: Any):
            super().__init__()
            self.block = block

        def forward(self, vid: Any, txt: Any, vid_shape: Any, txt_shape: Any, emb: Any) -> tuple[Any, Any]:
            return self.block(vid, txt, vid_shape, txt_shape, emb, Cache(disable=False))[:2]

    return Trusted(source_block).eval()


def build_inputs(contract: Mapping[str, Any], seed: int, torch: Any) -> tuple[Any, Any, Any]:
    shape = tuple(int(value) for value in contract["shape"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video_tokens = math.prod(shape)
    vid = torch.randn((video_tokens, VID_DIM), generator=generator, dtype=torch.float32) * 0.02
    txt = torch.randn((TXT_TOKENS, TXT_DIM), generator=generator, dtype=torch.float32) * 0.02
    emb = torch.randn((1, EMB_DIM), generator=generator, dtype=torch.float32) * 0.02
    return vid, txt, emb


def collect_eager_attention(
    chunk: Any,
    prefix_outputs: tuple[Any, ...],
    contract: Mapping[str, Any],
    chunk_size: int,
    torch: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    source_tokens = int(contract["vid_shape"].prod().item())
    max_len = int(contract["max_video_len"])
    vid_acc = torch.zeros((source_tokens, INNER_DIM), dtype=prefix_outputs[0].dtype)
    txt_sum = torch.zeros((TXT_TOKENS, INNER_DIM), dtype=prefix_outputs[3].dtype)
    total_count = 0.0
    calls = []
    window_count = int(contract["window_count"])
    for start in range(0, window_count, chunk_size):
        end = min(window_count, start + chunk_size)
        chunk_contract = split_contract(contract, start, end)
        t0 = time.perf_counter()
        vid_chunk, txt_chunk_sum, txt_chunk_count = chunk(
            *prefix_outputs,
            chunk_contract["window_idx"],
            chunk_contract["window_key_mask"],
            chunk_contract["window_rope_freqs"],
        )
        elapsed = time.perf_counter() - t0
        global_reverse = contract["window_reverse_idx"]
        local_reverse = global_reverse - start * max_len
        source_idx = chunk_contract["window_idx"][:, :max_len][
            chunk_contract["window_key_mask"][:, :max_len]
        ]
        selected_positions = local_reverse.index_select(0, source_idx)
        selected_values = vid_chunk.reshape(-1, INNER_DIM).index_select(0, selected_positions)
        if int(torch.unique(source_idx).numel()) != int(source_idx.numel()):
            raise AssertionError("chunk inverse-gather saw a duplicate source token")
        vid_acc.index_copy_(0, source_idx, selected_values)
        txt_sum = txt_sum + txt_chunk_sum
        count = float(txt_chunk_count.item())
        total_count += count
        calls.append({"start": start, "end": end, "windows": end - start, "seconds": elapsed})
    if abs(total_count - window_count) > 1.0e-6:
        raise AssertionError(f"text count mismatch: {total_count} != {window_count}")
    return vid_acc, txt_sum / total_count, {
        "chunk_size": chunk_size,
        "calls": calls,
        "window_count": window_count,
        "text_count": total_count,
    }


def _named_ports(model: Any, names: Sequence[str]) -> None:
    if len(model.inputs) != len(names):
        raise AssertionError(f"input count mismatch: {len(model.inputs)} != {len(names)}")
    for port, name in zip(model.inputs, names):
        port.get_tensor().set_names({name})


def _name_outputs(model: Any, names: Sequence[str]) -> None:
    if len(model.outputs) != len(names):
        raise AssertionError(f"output count mismatch: {len(model.outputs)} != {len(names)}")
    for port, name in zip(model.outputs, names):
        port.get_tensor().set_names({name})


def _reshape_model(ov: Any, model: Any, input_names: Sequence[str]) -> None:
    reshape_map = {}
    for port in model.inputs:
        name = port.get_any_name()
        if name == "vid":
            reshape_map[port] = ov.PartialShape([-1, VID_DIM])
        elif name == "txt":
            reshape_map[port] = ov.PartialShape([TXT_TOKENS, TXT_DIM])
        elif name == "emb":
            reshape_map[port] = ov.PartialShape([1, EMB_DIM])
        elif name in {"attn_vid", "attn_txt"}:
            reshape_map[port] = ov.PartialShape(
                [-1, INNER_DIM] if name == "attn_vid" else [TXT_TOKENS, INNER_DIM]
            )
        elif name in {"vid_q", "vid_k", "vid_v"}:
            reshape_map[port] = ov.PartialShape([-1, HEADS, HEAD_DIM])
        elif name in {"txt_q", "txt_k", "txt_v"}:
            reshape_map[port] = ov.PartialShape([TXT_TOKENS, HEADS, HEAD_DIM])
        elif name == "window_idx":
            reshape_map[port] = ov.PartialShape([-1, -1])
        elif name == "window_key_mask":
            reshape_map[port] = ov.PartialShape([-1, -1])
        elif name == "window_rope_freqs":
            reshape_map[port] = ov.PartialShape([-1, -1, 126])
    if set(port.get_any_name() for port in model.inputs) != set(input_names):
        raise AssertionError("converted input names do not match declared contract")
    model.reshape(reshape_map)


def _output_tuple(result: Mapping[Any, Any], compiled: Any, torch: Any) -> tuple[Any, ...]:
    return tuple(torch.from_numpy(result[compiled.outputs[i]]) for i in range(len(compiled.outputs)))


def _feed(compiled: Any, names: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    expected = dict(zip(names, values))
    actual = {port.any_name for port in compiled.inputs}
    if actual != set(names):
        raise AssertionError(f"compiled input names mismatch: {sorted(actual)}")
    return {name: value.detach().cpu().numpy() for name, value in expected.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block-index",
        type=int,
        choices=tuple(range(32)),
        required=True,
        help="SeedVR2 transformer block index (all 32 checkpoint blocks are supported)",
    )
    parser.add_argument("--max-threads", type=int, default=4)
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()
    if args.max_threads > 4:
        raise ValueError("this proof is capped at four CPU threads")
    block_index = args.block_index
    base_path = HERE / f"dynamic_window_reverse_gather_block{block_index}_export.py"
    base = _load_module(base_path, f"reverse_gather_base_{block_index}")
    prefix_xml = HERE / f"block{block_index}_chunked_prefix_fp16.xml"
    chunk_xml = HERE / f"block{block_index}_chunked_attention_fp16.xml"
    suffix_xml = HERE / f"block{block_index}_chunked_suffix_fp16.xml"
    prefix_bin = prefix_xml.with_suffix(".bin")
    chunk_bin = chunk_xml.with_suffix(".bin")
    suffix_bin = suffix_xml.with_suffix(".bin")
    receipt_path = HERE / f"receipt.chunked_decomposition_block{block_index}.json"
    start_wall = time.perf_counter()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.max_threads))
    receipt: dict[str, Any] = {
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "chunked-attention-decomposition-openvino-20260918",
        "block_index": block_index,
        "execution": {
            "device": "CPU",
            "gpu_inspection": False,
            "gpu": "not inspected/used",
            "advisory_lock": "not inspected/acquired per explicit CPU-only session instruction",
            "thread_cap": args.max_threads,
        },
        "host_coordination": {
            "preflight": None,
            "pre_conversion": None,
            "thread_cap": args.max_threads,
            "lock": "not inspected/acquired per explicit CPU-only session instruction",
        },
        "commands": {
            "proof": f"OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 {sys.executable} {Path(__file__).name} --block-index {block_index} --max-threads 4",
            "runtime": str(OV_ROOT),
        },
        "thresholds": {
            "eager_max_abs_error": MAX_ABS_THRESHOLD,
            "eager_rms_relative_error": RMS_REL_THRESHOLD,
            "saved_ir_max_abs_error": SAVED_IR_MAX_ABS_THRESHOLD,
            "saved_ir_rms_relative_error": SAVED_IR_RMS_REL_THRESHOLD,
            "distinct_output_delta": DISTINCT_THRESHOLD,
        },
        "decomposition": {
            "prefix": "attn_norm + Ada attention-in + proj_qkv + q/k norm; exposes vid/txt Q/K/V",
            "chunk": "weightless local-RoPE padded-window SDPA; returns per-window video values, text sum, text count",
            "suffix": "text mean + proj_out + Ada attention-out + residual + exact MLP norm/Ada/MLP/Ada/residual",
            "caller": "sequential chunks, exact inverse-gather of each video token, exact text sum/count",
            "max_chunk_windows_proof": CHUNK_SIZE_PROOF,
        },
        "cases": {},
        "events": [],
    }
    try:
        receipt["host_coordination"]["preflight"] = sample_host_cpu_load()
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
        trusted_helper = _load_module(base.TRUSTED_HELPER, f"trusted_block0_for_chunked_{block_index}")
        receipt["source"] = {
            "path": str(SOURCE),
            "base_helper": str(base_path),
            "base_helper_sha256": sha256_file(base_path),
            "trusted_helper": str(base.TRUSTED_HELPER),
            "trusted_helper_sha256": sha256_file(base.TRUSTED_HELPER),
            "relevant_source_sha256": trusted_helper._source_hash(SOURCE)[0],
        }
        receipt["weights"] = {"path": str(WEIGHTS), "sha256": sha256_file(WEIGHTS)}
        if block_index == 0:
            state, shard_meta = trusted_helper._load_block_weights(WEIGHTS)
            source_block = trusted_helper._build_block(SOURCE, state, torch)
            shared_weights = False
            is_last_layer = False
            window_method = base.WINDOW_METHOD
        else:
            expected_shared_weights = block_index >= 10
            expected_last_layer = block_index == 31
            if bool(base.SHARED_WEIGHTS) != expected_shared_weights:
                raise AssertionError(
                    f"block-{block_index} shared-weight family mismatch: "
                    f"{base.SHARED_WEIGHTS!r} != {expected_shared_weights!r}"
                )
            if bool(base.IS_LAST_LAYER) != expected_last_layer:
                raise AssertionError(
                    f"block-{block_index} last-layer contract mismatch: "
                    f"{base.IS_LAST_LAYER!r} != {expected_last_layer!r}"
                )
            state, shard_meta = base.load_block_weights(WEIGHTS)
            source_block = base.build_source_block(SOURCE, state, torch)
            shared_weights = bool(base.SHARED_WEIGHTS)
            is_last_layer = bool(base.IS_LAST_LAYER)
            window_method = base.WINDOW_METHOD
        receipt["weights"].update(shard_meta)
        receipt["model_args"] = {
            "block_index": block_index,
            "num_layers": 32,
            "mm_layers": 10,
            "block_family": "unshared" if not shared_weights else "shared",
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
            "shared_weights": shared_weights,
            "rope_type": "mmrope3d",
            "rope_dim": 128,
            "is_last_layer": is_last_layer,
            "attention_mode": "sdpa",
            "window": list(WINDOW),
            "window_method": window_method,
        }
        trusted = _trusted_wrapper(source_block, torch)
        prefix, chunk, suffix = build_decomposition_modules(torch, source_block)

        contracts: dict[tuple[int, int, int], dict[str, Any]] = {}
        seeds = {shape: 2026091801 + i for i, shape in enumerate(CASE_SHAPES)}
        for shape in CASE_SHAPES:
            summary = window_metadata(base, shape)
            contract = base.make_window_contract(
                shape,
                max_video_len=summary["max_video_len"],
                txt_len=TXT_TOKENS,
                rope=source_block.attn.rope,
                torch=torch,
            )
            coverage = validate_window_coverage(contract, torch)
            contract["storage_elements"] = attention_score_elements(
                summary["window_count"], summary["max_video_len"]
            )
            contract["shape"] = shape
            contracts[shape] = contract
            chunk_sizes = sorted(set([1, 4, CHUNK_SIZE_PROOF, summary["window_count"]]))
            receipt["cases"][str(shape)] = {
                "shape": list(shape),
                "window_count": summary["window_count"],
                "window_length_max": summary["max_video_len"],
                "window_lengths": summary["window_lengths"],
                "chunk_sizes": chunk_sizes,
                "coverage": coverage,
                "input_contract": {
                    "vid": [math.prod(shape), VID_DIM],
                    "txt": [TXT_TOKENS, TXT_DIM],
                    "emb": [1, EMB_DIM],
                    "window_idx": [summary["window_count"], summary["max_video_len"]],
                    "window_key_mask": [summary["window_count"], summary["max_video_len"] + TXT_TOKENS],
                    "window_rope_freqs": [summary["window_count"], summary["max_video_len"] + TXT_TOKENS, 126],
                },
                "attention_score_elements": {
                    "whole_case": attention_score_elements(summary["window_count"], summary["max_video_len"]),
                    "peak_with_16_window_chunks": attention_score_elements(
                        summary["window_count"], summary["max_video_len"], chunk_windows=CHUNK_SIZE_PROOF
                    ),
                },
                "eager": {},
                "saved_ir": {},
            }
        audit = []
        for shape in PRODUCTION_SHAPES:
            item = window_metadata(base, shape)
            if item["window_count"] > PRODUCTION_MAX_WINDOWS or item["max_video_len"] > PRODUCTION_MAX_VIDEO_LEN:
                raise AssertionError(f"production cap exceeded for {shape}: {item}")
            item["peak_score_elements_with_16_window_chunks"] = attention_score_elements(
                item["window_count"], item["max_video_len"], chunk_windows=CHUNK_SIZE_PROOF
            )
            audit.append(item)
        receipt["window_contract"] = {
            "window": list(WINDOW),
            "window_method": window_method,
            "production_caps": {"max_windows": PRODUCTION_MAX_WINDOWS, "max_video_len": PRODUCTION_MAX_VIDEO_LEN},
            "production_audit": audit,
        }

        eager_context: dict[tuple[int, int, int], dict[str, Any]] = {}
        with torch.inference_mode():
            for shape in CASE_SHAPES:
                contract = contracts[shape]
                vid, txt, emb = build_inputs(contract, seeds[shape], torch)
                vid_shape = contract["vid_shape"]
                txt_shape = contract["txt_shape"]
                t0 = time.perf_counter()
                trusted_out = trusted(vid, txt, vid_shape, txt_shape, emb)
                trusted_seconds = time.perf_counter() - t0
                prefix_start = time.perf_counter()
                prefix_out = prefix(vid, txt, emb)
                prefix_seconds = time.perf_counter() - prefix_start
                _assert_finite(trusted_out, torch, f"trusted {shape}")
                eager_case: dict[str, Any] = {
                    "trusted_seconds": trusted_seconds,
                    "prefix_seconds": prefix_seconds,
                    "trusted_vs_decomposition": {},
                    "finite": True,
                }
                decomposed_for_trace = None
                for chunk_size in receipt["cases"][str(shape)]["chunk_sizes"]:
                    t0 = time.perf_counter()
                    attn_vid, attn_txt, chunk_timing = collect_eager_attention(
                        chunk, prefix_out, contract, chunk_size, torch
                    )
                    final_out = suffix(vid, txt, emb, attn_vid, attn_txt)
                    elapsed = time.perf_counter() - t0
                    _assert_finite(final_out, torch, f"decomposition {shape} chunk={chunk_size}")
                    metric = metrics(trusted_out, final_out, np)
                    if metric["max_abs_error"] > MAX_ABS_THRESHOLD or metric["max_rms_relative_error"] > RMS_REL_THRESHOLD:
                        raise AssertionError(f"eager decomposition threshold failure {shape} chunk={chunk_size}: {metric}")
                    eager_case["trusted_vs_decomposition"][str(chunk_size)] = {
                        "metrics": metric,
                        "seconds": elapsed,
                        "chunk_timing": chunk_timing,
                        "text_average_exact": abs(chunk_timing["text_count"] - contract["window_count"]) < 1.0e-6,
                    }
                    if chunk_size == receipt["cases"][str(shape)]["chunk_sizes"][-1]:
                        decomposed_for_trace = (vid, txt, emb, prefix_out, attn_vid, attn_txt)
                    del attn_vid, attn_txt, final_out
                receipt["cases"][str(shape)]["eager"] = eager_case
                eager_context[shape] = {
                    "vid": vid,
                    "txt": txt,
                    "emb": emb,
                    "prefix_out": prefix_out,
                    "suffix_example": decomposed_for_trace,
                    "trusted_out": trusted_out,
                }
                del trusted_out
                gc.collect()

        example_shape = CASE_SHAPES[0]
        example = eager_context[example_shape]
        example_contract = contracts[example_shape]
        # Eager validation intentionally runs under inference_mode. Clone the
        # fixture tensors back to ordinary tensors before TorchScript tracing;
        # this changes no values or graph math and avoids autograd's inference
        # tensor guard while tracing Linear/MLP suffix operators.
        trace_vid = example["vid"].clone()
        trace_txt = example["txt"].clone()
        trace_emb = example["emb"].clone()
        trace_prefix_out = tuple(value.clone() for value in example["prefix_out"])
        prefix_trace = torch.jit.freeze(
            torch.jit.trace(prefix, (trace_vid, trace_txt, trace_emb), strict=False, check_trace=False).eval()
        )
        example_chunk = split_contract(example_contract, 0, min(CHUNK_SIZE_PROOF, example_contract["window_count"]))
        chunk_trace = torch.jit.freeze(
            torch.jit.trace(
                chunk,
                trace_prefix_out
                + (
                    example_chunk["window_idx"],
                    example_chunk["window_key_mask"],
                    example_chunk["window_rope_freqs"],
                ),
                strict=False,
                check_trace=False,
            ).eval()
        )
        suffix_example = example["suffix_example"]
        suffix_trace = torch.jit.freeze(
            torch.jit.trace(
                suffix,
                (trace_vid, trace_txt, trace_emb, suffix_example[4].clone(), suffix_example[5].clone()),
                strict=False,
                check_trace=False,
            ).eval()
        )
        graphs = {"prefix": str(prefix_trace.graph), "chunk": str(chunk_trace.graph), "suffix": str(suffix_trace.graph)}
        receipt["trace_graph"] = {
            name: {
                "graph_chars": len(graph),
                "contains_index_add": "index_add" in graph,
                "contains_scatter": "scatter" in graph.lower(),
                "contains_nonzero": "nonzero" in graph.lower(),
                "contains_scaled_dot_product_attention": "scaled_dot_product_attention" in graph,
            }
            for name, graph in graphs.items()
        }
        if any(item["contains_index_add"] or item["contains_scatter"] or item["contains_nonzero"] for item in receipt["trace_graph"].values()):
            raise AssertionError("forbidden scatter/index_add/nonzero operation in decomposition graph")
        if args.skip_convert:
            receipt["status"] = "skipped_conversion"
            receipt["finished_seconds"] = time.perf_counter() - start_wall
            receipt.update(peak_rss())
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            return 0

        import openvino as ov

        models = {
            "prefix": (prefix_trace, (example["vid"], example["txt"], example["emb"]), ["vid", "txt", "emb"], ["vid_q", "vid_k", "vid_v", "txt_q", "txt_k", "txt_v"], prefix_xml, prefix_bin),
            "chunk": (chunk_trace, tuple(example["prefix_out"]) + (example_chunk["window_idx"], example_chunk["window_key_mask"], example_chunk["window_rope_freqs"]), ["vid_q", "vid_k", "vid_v", "txt_q", "txt_k", "txt_v", "window_idx", "window_key_mask", "window_rope_freqs"], ["vid_window_out", "txt_sum", "txt_count"], chunk_xml, chunk_bin),
            "suffix": (suffix_trace, (suffix_example[0], suffix_example[1], suffix_example[2], suffix_example[4], suffix_example[5]), ["vid", "txt", "emb", "attn_vid", "attn_txt"], ["vid_out", "txt_out"], suffix_xml, suffix_bin),
        }
        ov_models: dict[str, Any] = {}
        receipt["artifacts"] = {}
        convert_start = time.perf_counter()
        receipt["host_coordination"]["pre_conversion"] = sample_host_cpu_load()
        for name, (graph, example_input, input_names, output_names, xml_path, bin_path) in models.items():
            model = ov.convert_model(graph, example_input=example_input)
            _named_ports(model, input_names)
            _name_outputs(model, output_names)
            _reshape_model(ov, model, input_names)
            ov.save_model(model, str(xml_path), compress_to_fp16=True)
            ov_models[name] = model
            receipt["artifacts"][name] = {
                "xml": str(xml_path),
                "bin": str(bin_path),
                "xml_bytes": xml_path.stat().st_size,
                "bin_bytes": bin_path.stat().st_size if bin_path.exists() else 0,
                "total_bytes": xml_path.stat().st_size + (bin_path.stat().st_size if bin_path.exists() else 0),
                "compress_to_fp16": True,
            }
        receipt["convert_seconds"] = time.perf_counter() - convert_start
        receipt["openvino_io"] = {
            name: [
                {"name": p.get_any_name(), "shape": str(p.get_partial_shape()), "element_type": str(p.get_element_type())}
                for p in model.inputs
            ]
            for name, model in ov_models.items()
        }
        receipt["openvino_outputs"] = {
            name: [
                {"name": p.get_any_name(), "shape": str(p.get_partial_shape()), "element_type": str(p.get_element_type())}
                for p in model.outputs
            ]
            for name, model in ov_models.items()
        }
        core = ov.Core()
        compile_start = time.perf_counter()
        compiled = {
            name: core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": args.max_threads})
            for name, model in ov_models.items()
        }
        receipt["compile_seconds"] = time.perf_counter() - compile_start

        with torch.inference_mode():
            for shape in CASE_SHAPES:
                context = eager_context[shape]
                contract = contracts[shape]
                vid, txt, emb = context["vid"], context["txt"], context["emb"]
                prefix_values = (vid, txt, emb)
                prefix_result = compiled["prefix"](_feed(compiled["prefix"], ["vid", "txt", "emb"], prefix_values))
                prefix_saved = _output_tuple(prefix_result, compiled["prefix"], torch)
                saved_case: dict[str, Any] = {}
                for chunk_size in receipt["cases"][str(shape)]["chunk_sizes"]:
                    max_len = int(contract["max_video_len"])
                    source_tokens = int(contract["vid_shape"].prod().item())
                    vid_acc = torch.zeros((source_tokens, INNER_DIM), dtype=torch.float32)
                    txt_sum = torch.zeros((TXT_TOKENS, INNER_DIM), dtype=torch.float32)
                    chunk_events = []
                    text_count = 0.0
                    window_count = int(contract["window_count"])
                    for start in range(0, window_count, chunk_size):
                        end = min(window_count, start + chunk_size)
                        c = split_contract(contract, start, end)
                        c_values = tuple(prefix_saved) + (c["window_idx"], c["window_key_mask"], c["window_rope_freqs"])
                        t0 = time.perf_counter()
                        result = compiled["chunk"](_feed(compiled["chunk"], models["chunk"][2], c_values))
                        elapsed = time.perf_counter() - t0
                        vid_chunk, txt_chunk_sum, txt_chunk_count = _output_tuple(result, compiled["chunk"], torch)
                        global_reverse = contract["window_reverse_idx"]
                        local_reverse = global_reverse - start * max_len
                        source_idx = c["window_idx"][:, :max_len][c["window_key_mask"][:, :max_len]]
                        selected_positions = local_reverse.index_select(0, source_idx)
                        vid_acc.index_copy_(0, source_idx, vid_chunk.reshape(-1, INNER_DIM).index_select(0, selected_positions))
                        txt_sum = txt_sum + txt_chunk_sum
                        text_count += float(txt_chunk_count.item())
                        chunk_events.append({"start": start, "end": end, "windows": end - start, "seconds": elapsed})
                    if abs(text_count - window_count) > 1.0e-6:
                        raise AssertionError(f"saved text count mismatch {shape} chunk={chunk_size}: {text_count} != {window_count}")
                    attn_txt = txt_sum / text_count
                    t0 = time.perf_counter()
                    suffix_result = compiled["suffix"](_feed(compiled["suffix"], models["suffix"][2], (vid, txt, emb, vid_acc, attn_txt)))
                    suffix_seconds = time.perf_counter() - t0
                    saved_out = _output_tuple(suffix_result, compiled["suffix"], torch)
                    _assert_finite(saved_out, torch, f"saved decomposition {shape} chunk={chunk_size}")
                    metric = metrics(context["trusted_out"], saved_out, np)
                    if metric["max_abs_error"] > SAVED_IR_MAX_ABS_THRESHOLD or metric["max_rms_relative_error"] > SAVED_IR_RMS_REL_THRESHOLD:
                        raise AssertionError(f"saved decomposition threshold failure {shape} chunk={chunk_size}: {metric}")
                    saved_case[str(chunk_size)] = {
                        "metrics_vs_trusted": metric,
                        "finite": True,
                        "text_count": text_count,
                        "prefix_seconds": None,
                        "chunk_events": chunk_events,
                        "suffix_seconds": suffix_seconds,
                    }
                receipt["cases"][str(shape)]["saved_ir"] = saved_case
                del prefix_result, prefix_saved
                gc.collect()

        receipt["status"] = "pass"
        receipt["finished_seconds"] = time.perf_counter() - start_wall
        receipt.update(peak_rss())
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception as exc:
        receipt["status"] = "failure"
        receipt["finished_seconds"] = time.perf_counter() - start_wall
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        receipt.update(peak_rss())
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
