#!/usr/bin/env python3
"""CPU proof of one fixed-envelope fused SeedVR2 block graph.

The graph has one public call per learned block.  Its prefix computes Q/K/V
once, thirteen independent weightless attention branches consume sixteen
windows each, and the suffix consumes one inverse-gathered video tensor plus
one exact text sum/count.  The production [2, 135, 240] contract is padded to
208 rows (13 * 16) and every dummy row has one safe key so softmax never sees
an all-masked row.  ``window_valid`` removes those rows from text pooling and
the inverse map never selects their video output.

This proof deliberately leaves the attention branches independent.  The
``stack``/``cat`` reductions combine their results only after all branches
have been formed; no artificial branch-to-branch dependency is introduced.
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
EVIDENCE = HERE.parent
TRIAL_ROOT = EVIDENCE.parents[1]
SOURCE = Path(os.environ.get("SEEDVR_SOURCE", TRIAL_ROOT / "source_snapshot"))
WEIGHTS = Path(os.environ.get("SEEDVR_WEIGHTS", TRIAL_ROOT / "seedvr2_ema_3b_fp16.safetensors"))
DECOMPOSITION_HELPER = EVIDENCE / "chunked_attention_decomposition_export.py"

VID_DIM = 2560
TXT_DIM = 2560
EMB_DIM = 15360
HEADS = 20
HEAD_DIM = 128
INNER_DIM = HEADS * HEAD_DIM
TXT_TOKENS = 58
ROPE_WIDTH = 126
SHAPE = (2, 135, 240)
SOURCE_TOKENS = math.prod(SHAPE)
ACTUAL_MAX_VIDEO_LEN = 405
WINDOW_CAPACITY = 208
CHUNK_SIZE = 16
CHUNK_COUNT = WINDOW_CAPACITY // CHUNK_SIZE
ATTENTION_LEN = ACTUAL_MAX_VIDEO_LEN + TXT_TOKENS
BLOCKS = (0, 16, 31)
EAGER_MAX_ABS_THRESHOLD = 1.0e-3
EAGER_RMS_REL_THRESHOLD = 1.0e-4
SAVED_IR_MAX_ABS_THRESHOLD = 5.0e-3
SAVED_IR_RMS_REL_THRESHOLD = 1.0e-4
DUMMY_MAX_ABS_THRESHOLD = 1.0e-7
DISTINCT_THRESHOLD = 1.0e-7
INPUT_NAMES = (
    "vid",
    "txt",
    "emb",
    "window_idx",
    "window_key_mask",
    "window_rope_freqs",
    "window_valid",
    "window_reverse_idx",
)
OUTPUT_NAMES = ("vid_out", "txt_out")

# Each receipt captures fresh preflight and pre-conversion host samples.
HOST_COORDINATION = {
    "sample_command": "vmstat 1 3",
    "sample_record": "inline preflight and pre-conversion samples in this receipt",
    "competing_work": "proof capped at four threads and run sequentially",
    "timing_validity": "correctness evidence; CPU timings are not performance claims",
}


def load_module(path: Path, name: str) -> Any:
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


def sha256_tensor_sample(value: Any) -> str:
    sample = value.detach().cpu().reshape(-1)[:4096].contiguous().numpy()
    return hashlib.sha256(sample.tobytes()).hexdigest()


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


def assert_finite(values: Sequence[Any], torch: Any, label: str) -> None:
    for value in values:
        if not bool(torch.isfinite(value).all().item()):
            raise AssertionError(f"{label}: non-finite output")


def chunk_ranges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (index * CHUNK_SIZE, (index + 1) * CHUNK_SIZE)
        for index in range(CHUNK_COUNT)
    )


def pad_window_metadata(contract: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    """Pad a proven production contract to the fixed 208 x 405 envelope."""

    count = int(contract["window_count"])
    max_video_len = int(contract["max_video_len"])
    if count <= 0 or count > WINDOW_CAPACITY:
        raise ValueError(f"window count {count} is outside the fixed envelope")
    if max_video_len != ACTUAL_MAX_VIDEO_LEN:
        raise ValueError(
            f"production max video length {max_video_len} != {ACTUAL_MAX_VIDEO_LEN}"
        )
    idx = torch.zeros((WINDOW_CAPACITY, ACTUAL_MAX_VIDEO_LEN), dtype=torch.long)
    key_mask = torch.zeros(
        (WINDOW_CAPACITY, ATTENTION_LEN), dtype=torch.bool
    )
    rope = torch.zeros(
        (WINDOW_CAPACITY, ATTENTION_LEN, ROPE_WIDTH), dtype=torch.float32
    )
    valid = torch.zeros((WINDOW_CAPACITY,), dtype=torch.bool)
    idx[:count] = contract["window_idx"]
    key_mask[:count] = contract["window_key_mask"]
    rope[:count] = contract["window_rope_freqs"]
    valid[:count] = True
    # A dummy row is never selected by window_reverse_idx and is weighted out
    # of text pooling, but it still needs one legal key for finite softmax.
    key_mask[count:, 0] = True
    reverse = contract["window_reverse_idx"].clone()
    if tuple(reverse.shape) != (SOURCE_TOKENS,):
        raise AssertionError(f"unexpected reverse map shape: {tuple(reverse.shape)}")
    return {
        "window_idx": idx,
        "window_key_mask": key_mask,
        "window_rope_freqs": rope,
        "window_valid": valid,
        "window_reverse_idx": reverse,
        "actual_window_count": count,
        "dummy_window_count": WINDOW_CAPACITY - count,
    }


def mutate_dummy_metadata(metadata: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    """Change only ignored dummy rows to prove they cannot affect valid output."""

    result = {
        key: value.clone() if hasattr(value, "clone") else value
        for key, value in metadata.items()
    }
    count = int(metadata["actual_window_count"])
    if count >= WINDOW_CAPACITY:
        raise AssertionError("dummy mutation requires at least one dummy row")
    result["window_idx"][count:, :] = SOURCE_TOKENS - 1
    result["window_key_mask"][count:, :] = False
    result["window_key_mask"][count:, 0] = True
    result["window_rope_freqs"][count:, :, 0] = 0.375
    return result


def validate_padded_metadata(metadata: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    idx = metadata["window_idx"]
    mask = metadata["window_key_mask"]
    rope = metadata["window_rope_freqs"]
    valid = metadata["window_valid"]
    reverse = metadata["window_reverse_idx"]
    if tuple(idx.shape) != (WINDOW_CAPACITY, ACTUAL_MAX_VIDEO_LEN):
        raise AssertionError(f"unexpected fixed index shape: {tuple(idx.shape)}")
    if tuple(mask.shape) != (WINDOW_CAPACITY, ATTENTION_LEN):
        raise AssertionError(f"unexpected fixed mask shape: {tuple(mask.shape)}")
    if tuple(rope.shape) != (WINDOW_CAPACITY, ATTENTION_LEN, ROPE_WIDTH):
        raise AssertionError(f"unexpected fixed RoPE shape: {tuple(rope.shape)}")
    if tuple(valid.shape) != (WINDOW_CAPACITY,):
        raise AssertionError(f"unexpected window_valid shape: {tuple(valid.shape)}")
    if tuple(reverse.shape) != (SOURCE_TOKENS,):
        raise AssertionError(f"unexpected fixed reverse shape: {tuple(reverse.shape)}")
    count = int(metadata["actual_window_count"])
    if int(valid[:count].sum().item()) != count or bool(valid[count:].any().item()):
        raise AssertionError("window_valid does not match actual/dummy rows")
    if bool(mask[count:, 1:].any().item()) or not bool(mask[count:, 0].all().item()):
        raise AssertionError("dummy rows do not have exactly one safe key")
    if not bool(torch.equal(reverse, reverse.clamp_min(0))):
        raise AssertionError("reverse map contains an invalid negative position")
    valid_video = mask[:count, :ACTUAL_MAX_VIDEO_LEN]
    if int(valid_video.sum().item()) != SOURCE_TOKENS:
        raise AssertionError("valid window masks do not cover source tokens")
    return {
        "actual_window_count": count,
        "dummy_window_count": WINDOW_CAPACITY - count,
        "source_tokens": SOURCE_TOKENS,
        "window_capacity": WINDOW_CAPACITY,
        "max_video_len": ACTUAL_MAX_VIDEO_LEN,
        "attention_len": ATTENTION_LEN,
        "chunk_count": CHUNK_COUNT,
        "chunk_size": CHUNK_SIZE,
    }


def build_fused_module(torch: Any, source_block: Any) -> Any:
    """Build one prefix -> 13 independent chunks -> gather -> suffix module."""

    import torch.nn as nn
    import torch.nn.functional as F

    decomp = load_module(DECOMPOSITION_HELPER, "decomposition_for_fused_envelope")
    prefix, _, suffix = decomp.build_decomposition_modules(torch, source_block)

    class PerWindowChunk(nn.Module):
        """The trusted chunk math with text retained per window for valid weighting."""

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
        ) -> tuple[Any, Any]:
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
            q = decomp._apply_batched_rope(q, window_rope_freqs, torch)
            k = decomp._apply_batched_rope(k, window_rope_freqs, torch)
            out = F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                attn_mask=window_key_mask.unsqueeze(1).unsqueeze(1),
                dropout_p=0.0,
                is_causal=False,
            ).permute(0, 2, 1, 3)
            vid_out = out[:, :max_video_len].reshape(
                window_count, max_video_len, INNER_DIM
            )
            txt_out = out[:, max_video_len : max_video_len + TXT_TOKENS].reshape(
                window_count, TXT_TOKENS, INNER_DIM
            )
            return vid_out, txt_out

    chunk = PerWindowChunk().eval()

    class FusedFixedEnvelope(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prefix = prefix
            self.chunk = chunk
            self.suffix = suffix

        def forward(
            self,
            vid: Any,
            txt: Any,
            emb: Any,
            window_idx: Any,
            window_key_mask: Any,
            window_rope_freqs: Any,
            window_valid: Any,
            window_reverse_idx: Any,
        ) -> tuple[Any, Any]:
            prefix_outputs = self.prefix(vid, txt, emb)
            video_chunks = []
            text_chunks = []
            for start, end in chunk_ranges():
                video_chunk, text_windows = self.chunk(
                    *prefix_outputs,
                    window_idx[start:end],
                    window_key_mask[start:end],
                    window_rope_freqs[start:end],
                )
                video_chunks.append(video_chunk)
                valid = window_valid[start:end].to(text_windows.dtype).reshape(-1, 1, 1)
                text_chunks.append((text_windows * valid).sum(dim=0))
            all_video_windows = torch.cat(video_chunks, dim=0)
            gathered_video = all_video_windows.reshape(-1, INNER_DIM).index_select(
                0, window_reverse_idx
            )
            text_sum = torch.stack(text_chunks, dim=0).sum(dim=0)
            text_count = window_valid.to(text_sum.dtype).sum().reshape(1)
            pooled_text = text_sum / text_count
            return self.suffix(vid, txt, emb, gathered_video, pooled_text)

    return FusedFixedEnvelope().eval()


def build_inputs(seed: int, torch: Any) -> tuple[Any, Any, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vid = torch.randn((SOURCE_TOKENS, VID_DIM), generator=generator, dtype=torch.float32) * 0.02
    txt = torch.randn((TXT_TOKENS, TXT_DIM), generator=generator, dtype=torch.float32) * 0.02
    emb = torch.randn((1, EMB_DIM), generator=generator, dtype=torch.float32) * 0.02
    return vid, txt, emb


def load_block(base: Any, block_index: int, decomp: Any, torch: Any) -> tuple[Any, dict[str, Any], bool, bool]:
    trusted_helper = load_module(
        base.TRUSTED_HELPER,
        f"trusted_block_helper_fused_fixed_{block_index}",
    )
    if block_index == 0:
        state, shard_meta = trusted_helper._load_block_weights(decomp.WEIGHTS)
        source_block = trusted_helper._build_block(decomp.SOURCE, state, torch)
        return source_block, shard_meta, False, False
    state, shard_meta = base.load_block_weights(decomp.WEIGHTS)
    source_block = base.build_source_block(decomp.SOURCE, state, torch)
    return source_block, shard_meta, bool(base.SHARED_WEIGHTS), bool(base.IS_LAST_LAYER)


def named_ports(model: Any, names: Sequence[str]) -> None:
    if len(model.inputs) != len(names):
        raise AssertionError(f"input count mismatch: {len(model.inputs)} != {len(names)}")
    for port, name in zip(model.inputs, names):
        port.get_tensor().set_names({name})


def named_outputs(model: Any, names: Sequence[str]) -> None:
    if len(model.outputs) != len(names):
        raise AssertionError(f"output count mismatch: {len(model.outputs)} != {len(names)}")
    for port, name in zip(model.outputs, names):
        port.get_tensor().set_names({name})


def element_type_name(port: Any) -> str:
    """Return a stable OpenVINO element-type spelling across runtime builds."""
    text = str(port.get_element_type()).lower()
    aliases = {
        "float32": "f32",
        "float16": "f16",
        "float64": "f64",
        "int64": "i64",
        "int32": "i32",
        "int16": "i16",
        "int8": "i8",
        "uint64": "u64",
        "uint32": "u32",
        "uint16": "u16",
        "uint8": "u8",
        "boolean": "boolean",
        "bool": "boolean",
    }
    for spelling, canonical in aliases.items():
        if spelling in text:
            return canonical
    return text


def output_tuple(result: Mapping[Any, Any], compiled: Any, torch: Any) -> tuple[Any, ...]:
    return tuple(torch.from_numpy(result[compiled.outputs[index]]) for index in range(len(compiled.outputs)))


def feed(compiled: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    actual = {port.any_name for port in compiled.inputs}
    if actual != set(values):
        raise AssertionError(f"compiled input names mismatch: {sorted(actual)}")
    return {name: value.detach().cpu().numpy() for name, value in values.items()}


def io_summary(model: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": port.get_any_name(),
            "shape": str(port.get_partial_shape()),
            "element_type": element_type_name(port),
        }
        for port in model.inputs
    ]


def graph_summary(graph: Any) -> dict[str, Any]:
    text = str(graph)
    lowered = text.lower()
    return {
        "graph_chars": len(text),
        "chunk_count": text.count("scaled_dot_product_attention"),
        "index_select_count": text.count("aten::index_select"),
        "cat_count": text.count("aten::cat"),
        "stack_count": text.count("aten::stack"),
        "contains_index_add": "index_add" in lowered,
        "contains_scatter": "scatter" in lowered,
        "contains_nonzero": "nonzero" in lowered,
    }


def build_contract(base: Any, source_block: Any, decomp: Any, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = decomp.window_metadata(base, SHAPE)
    if summary["max_video_len"] != ACTUAL_MAX_VIDEO_LEN:
        raise AssertionError(f"unexpected production max length: {summary}")
    if summary["window_count"] not in (162, 200):
        raise AssertionError(f"unexpected production window count: {summary}")
    contract = base.make_window_contract(
        SHAPE,
        max_video_len=summary["max_video_len"],
        txt_len=TXT_TOKENS,
        rope=source_block.attn.rope,
        torch=torch,
    )
    coverage = decomp.validate_window_coverage(contract, torch)
    metadata = pad_window_metadata(contract, torch)
    fixed_summary = validate_padded_metadata(metadata, torch)
    fixed_summary["coverage"] = coverage
    return contract, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-index", type=int, choices=BLOCKS, required=True)
    parser.add_argument("--max-threads", type=int, default=4)
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()
    if args.max_threads <= 0 or args.max_threads > 4:
        raise ValueError("this proof is capped at one to four CPU threads")

    block_index = args.block_index
    base_path = EVIDENCE / f"dynamic_window_reverse_gather_block{block_index}_export.py"
    base = load_module(base_path, f"fixed_envelope_base_{block_index}")
    decomp = load_module(DECOMPOSITION_HELPER, f"fixed_envelope_decomp_{block_index}")
    xml_path = HERE / f"block{block_index}_fused_fixed_envelope_fp16.xml"
    bin_path = xml_path.with_suffix(".bin")
    receipt_path = HERE / f"receipt.fused_fixed_envelope_block{block_index}.json"
    start_wall = time.perf_counter()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.max_threads)
    receipt: dict[str, Any] = {
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "fused-fixed-envelope-openvino-20260918",
        "block_index": block_index,
        "execution": {
            "device": "CPU",
            "gpu_inspection": False,
            "gpu": "not inspected/used",
            "advisory_lock": "not inspected/acquired per explicit CPU-only instruction",
            "thread_cap": args.max_threads,
            "one_host_device_boundary": True,
            "full_qkv_input_repeats": 0,
        },
        "host_coordination": {
            **HOST_COORDINATION,
            "preflight": None,
            "pre_conversion": None,
        },
        "commands": {
            "proof": f"OMP_NUM_THREADS={args.max_threads} MKL_NUM_THREADS={args.max_threads} OPENBLAS_NUM_THREADS={args.max_threads} {sys.executable} {Path(__file__).name} --block-index {block_index} --max-threads {args.max_threads}",
            "runtime": str(decomp.OV_ROOT),
        },
        "contract": {
            "shape": list(SHAPE),
            "source_tokens": SOURCE_TOKENS,
            "window_capacity": WINDOW_CAPACITY,
            "chunk_size": CHUNK_SIZE,
            "chunk_count": CHUNK_COUNT,
            "max_video_len": ACTUAL_MAX_VIDEO_LEN,
            "attention_len": ATTENTION_LEN,
            "txt_tokens": TXT_TOKENS,
            "rope_width": ROPE_WIDTH,
            "chunk_ranges": [list(pair) for pair in chunk_ranges()],
            "padding": "dummy rows have window_idx=0, one safe key, window_valid=0; inverse map selects valid rows only",
        },
        "thresholds": {
            "eager_max_abs_error": EAGER_MAX_ABS_THRESHOLD,
            "eager_rms_relative_error": EAGER_RMS_REL_THRESHOLD,
            "saved_ir_max_abs_error": SAVED_IR_MAX_ABS_THRESHOLD,
            "saved_ir_rms_relative_error": SAVED_IR_RMS_REL_THRESHOLD,
            "dummy_max_abs_error": DUMMY_MAX_ABS_THRESHOLD,
            "distinct_output_delta": DISTINCT_THRESHOLD,
        },
        "graph_design": {
            "prefix": "learned attention norm + Ada attention-in + QKV + Q/K norm, once",
            "chunks": "13 independent unrolled weightless padded-window attention branches, 16 rows each",
            "video_reconstruction": "one concatenate then one exact inverse Gather into 64800 source tokens",
            "text_pooling": "window_valid-weighted text sum divided by exact valid count",
            "suffix": "learned projection + Ada attention-out + residual + exact MLP path",
            "dependencies": "chunks independent until concatenation and text reduction; no artificial chunk chain",
        },
        "source": {
            "path": str(SOURCE),
            "base_helper": str(base_path),
            "base_helper_sha256": sha256_file(base_path),
            "decomposition_helper": str(DECOMPOSITION_HELPER),
            "decomposition_helper_sha256": sha256_file(DECOMPOSITION_HELPER),
        },
        "weights": {"path": str(WEIGHTS), "sha256": sha256_file(WEIGHTS)},
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
        source_block, shard_meta, shared_weights, is_last_layer = load_block(
            base, block_index, decomp, torch
        )
        receipt["weights"].update(shard_meta)
        receipt["model_args"] = {
            "block_index": block_index,
            "block_family": "unshared" if not shared_weights else "shared",
            "shared_weights": shared_weights,
            "is_last_layer": is_last_layer,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "vid_dim": VID_DIM,
            "txt_dim": TXT_DIM,
            "emb_dim": EMB_DIM,
            "rope_type": "mmrope3d",
            "rope_dim": 128,
            "window": [4, 3, 3],
            "window_method": base.WINDOW_METHOD,
        }
        contract, metadata = build_contract(base, source_block, decomp, torch)
        receipt["contract"].update(
            {
                "actual_window_count": metadata["actual_window_count"],
                "dummy_window_count": metadata["dummy_window_count"],
                "window_method": base.WINDOW_METHOD,
                "coverage": validate_padded_metadata(metadata, torch),
            }
        )
        inputs = build_inputs(2026091824, torch)
        vid, txt, emb = inputs
        vid_shape = contract["vid_shape"]
        txt_shape = contract["txt_shape"]
        fused = build_fused_module(torch, source_block)
        trusted = decomp._trusted_wrapper(source_block, torch)
        metadata_values = (
            metadata["window_idx"],
            metadata["window_key_mask"],
            metadata["window_rope_freqs"],
            metadata["window_valid"],
            metadata["window_reverse_idx"],
        )
        mutated_metadata = mutate_dummy_metadata(metadata, torch)
        mutated_values = (
            mutated_metadata["window_idx"],
            mutated_metadata["window_key_mask"],
            mutated_metadata["window_rope_freqs"],
            mutated_metadata["window_valid"],
            mutated_metadata["window_reverse_idx"],
        )

        with torch.inference_mode():
            trusted_start = time.perf_counter()
            trusted_out = trusted(vid, txt, vid_shape, txt_shape, emb)
            trusted_seconds = time.perf_counter() - trusted_start
            fused_start = time.perf_counter()
            fused_out = fused(*inputs, *metadata_values)
            fused_seconds = time.perf_counter() - fused_start
            mutated_start = time.perf_counter()
            mutated_out = fused(*inputs, *mutated_values)
            mutated_seconds = time.perf_counter() - mutated_start
        assert_finite(trusted_out, torch, f"trusted block {block_index}")
        assert_finite(fused_out, torch, f"fused block {block_index}")
        assert_finite(mutated_out, torch, f"mutated dummy block {block_index}")
        eager_metric = metrics(trusted_out, fused_out, np)
        dummy_metric = metrics(fused_out, mutated_out, np)
        if (
            eager_metric["max_abs_error"] > EAGER_MAX_ABS_THRESHOLD
            or eager_metric["max_rms_relative_error"] > EAGER_RMS_REL_THRESHOLD
        ):
            raise AssertionError(f"eager fused threshold failure: {eager_metric}")
        if dummy_metric["max_abs_error"] > DUMMY_MAX_ABS_THRESHOLD:
            raise AssertionError(f"dummy padding changed valid output: {dummy_metric}")
        receipt["cases"]["production"] = {
            "shape": list(SHAPE),
            "input_contract": {
                "vid": [SOURCE_TOKENS, VID_DIM],
                "txt": [TXT_TOKENS, TXT_DIM],
                "emb": [1, EMB_DIM],
                "window_idx": [WINDOW_CAPACITY, ACTUAL_MAX_VIDEO_LEN],
                "window_key_mask": [WINDOW_CAPACITY, ATTENTION_LEN],
                "window_rope_freqs": [WINDOW_CAPACITY, ATTENTION_LEN, ROPE_WIDTH],
                "window_valid": [WINDOW_CAPACITY],
                "window_reverse_idx": [SOURCE_TOKENS],
            },
            "trusted_seconds": trusted_seconds,
            "fused_eager_seconds": fused_seconds,
            "dummy_variant_seconds": mutated_seconds,
            "eager_trusted_vs_fused": eager_metric,
            "dummy_padding_invariance": dummy_metric,
            "finite": True,
        }
        output_sample_sha256 = sha256_tensor_sample(fused_out[0])
        prior_distinct: list[dict[str, Any]] = []
        for prior_index in BLOCKS:
            prior_path = HERE / f"receipt.fused_fixed_envelope_block{prior_index}.json"
            if not prior_path.exists() or prior_index == block_index:
                continue
            prior = json.loads(prior_path.read_text())
            prior_hash = (
                prior.get("cases", {})
                .get("production", {})
                .get("output_sample_sha256")
            )
            if prior.get("status") == "passed" and prior_hash:
                if prior_hash == output_sample_sha256:
                    raise AssertionError(
                        f"block {block_index} output sample equals block {prior_index}"
                    )
                prior_distinct.append(
                    {"block_index": prior_index, "sample_delta": "nonzero_sha256_delta"}
                )
        receipt["cases"]["production"]["output_sample_sha256"] = output_sample_sha256
        receipt["cases"]["production"]["distinct_from_prior_blocks"] = prior_distinct
        del trusted_out, fused_out, mutated_out

        trace_values = tuple(value.clone() for value in inputs + metadata_values)
        trace_start = time.perf_counter()
        fused_trace = torch.jit.freeze(
            torch.jit.trace(
                fused,
                trace_values,
                strict=False,
                check_trace=False,
            ).eval()
        )
        trace_seconds = time.perf_counter() - trace_start
        graph_info = graph_summary(fused_trace.graph)
        graph_info["trace_seconds"] = trace_seconds
        receipt["trace_graph"] = graph_info
        if graph_info["chunk_count"] != CHUNK_COUNT:
            raise AssertionError(f"expected {CHUNK_COUNT} attention branches: {graph_info}")
        if graph_info["contains_index_add"] or graph_info["contains_scatter"] or graph_info["contains_nonzero"]:
            raise AssertionError(f"forbidden graph op: {graph_info}")

        if args.skip_convert:
            receipt["status"] = "skipped_conversion"
            receipt["finished_seconds"] = time.perf_counter() - start_wall
            receipt.update(peak_rss())
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            return 0

        import openvino as ov

        example_values = tuple(value.detach().cpu().numpy() for value in inputs + metadata_values)
        convert_start = time.perf_counter()
        receipt["host_coordination"]["pre_conversion"] = sample_host_cpu_load()
        ov_model = ov.convert_model(fused_trace, example_input=example_values)
        named_ports(ov_model, INPUT_NAMES)
        named_outputs(ov_model, OUTPUT_NAMES)
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=True)
        convert_seconds = time.perf_counter() - convert_start
        if not xml_path.exists() or not bin_path.exists():
            raise AssertionError("OpenVINO save did not create XML and BIN artifacts")
        receipt["artifacts"] = {
            "xml": str(xml_path),
            "bin": str(bin_path),
            "xml_bytes": xml_path.stat().st_size,
            "bin_bytes": bin_path.stat().st_size,
            "total_bytes": xml_path.stat().st_size + bin_path.stat().st_size,
            "compress_to_fp16": True,
            "convert_seconds": convert_seconds,
        }
        receipt["openvino_io"] = io_summary(ov_model)
        receipt["openvino_outputs"] = [
            {
                "name": port.get_any_name(),
                "shape": str(port.get_partial_shape()),
                "element_type": element_type_name(port),
            }
            for port in ov_model.outputs
        ]
        required_public_f32 = {"vid", "txt", "emb", "window_rope_freqs"}
        for item in receipt["openvino_io"]:
            if item["name"] in required_public_f32 and item["element_type"] != "f32":
                raise AssertionError(f"public FP32 input changed type: {item}")
        if any(item["element_type"] != "f32" for item in receipt["openvino_outputs"]):
            raise AssertionError(f"public output is not FP32: {receipt['openvino_outputs']}")

        core = ov.Core()
        saved_model = core.read_model(str(xml_path))
        compile_start = time.perf_counter()
        compiled = core.compile_model(
            saved_model,
            "CPU",
            {"INFERENCE_NUM_THREADS": args.max_threads},
        )
        compile_seconds = time.perf_counter() - compile_start
        receipt["compile_seconds"] = compile_seconds
        baseline_values = {
            "vid": vid,
            "txt": txt,
            "emb": emb,
            "window_idx": metadata["window_idx"],
            "window_key_mask": metadata["window_key_mask"],
            "window_rope_freqs": metadata["window_rope_freqs"],
            "window_valid": metadata["window_valid"],
            "window_reverse_idx": metadata["window_reverse_idx"],
        }
        mutated_values_by_name = {
            "vid": vid,
            "txt": txt,
            "emb": emb,
            "window_idx": mutated_metadata["window_idx"],
            "window_key_mask": mutated_metadata["window_key_mask"],
            "window_rope_freqs": mutated_metadata["window_rope_freqs"],
            "window_valid": mutated_metadata["window_valid"],
            "window_reverse_idx": mutated_metadata["window_reverse_idx"],
        }
        saved_start = time.perf_counter()
        saved_result = compiled(feed(compiled, baseline_values))
        saved_out = output_tuple(saved_result, compiled, torch)
        saved_seconds = time.perf_counter() - saved_start
        saved_mutated_start = time.perf_counter()
        saved_mutated_result = compiled(feed(compiled, mutated_values_by_name))
        saved_mutated_out = output_tuple(saved_mutated_result, compiled, torch)
        saved_mutated_seconds = time.perf_counter() - saved_mutated_start
        assert_finite(saved_out, torch, f"saved IR block {block_index}")
        assert_finite(saved_mutated_out, torch, f"saved IR dummy block {block_index}")
        # Compare against the eager/trusted outputs captured before tracing;
        # fused_out itself was intentionally released, so use a fresh eager
        # call only for the saved-IR comparison and keep the proof explicit.
        with torch.inference_mode():
            eager_for_saved = fused(*inputs, *metadata_values)
        saved_metric = metrics(eager_for_saved, saved_out, np)
        saved_dummy_metric = metrics(saved_out, saved_mutated_out, np)
        if (
            saved_metric["max_abs_error"] > SAVED_IR_MAX_ABS_THRESHOLD
            or saved_metric["max_rms_relative_error"] > SAVED_IR_RMS_REL_THRESHOLD
        ):
            raise AssertionError(f"saved IR threshold failure: {saved_metric}")
        if saved_dummy_metric["max_abs_error"] > DUMMY_MAX_ABS_THRESHOLD:
            raise AssertionError(f"saved IR dummy padding changed output: {saved_dummy_metric}")
        receipt["saved_ir"] = {
            "reloaded_from_saved_xml": True,
            "inference_seconds": saved_seconds,
            "dummy_variant_seconds": saved_mutated_seconds,
            "fused_eager_vs_saved_ir": saved_metric,
            "dummy_padding_invariance": saved_dummy_metric,
            "finite": True,
            "xml_sha256": sha256_file(xml_path),
            "bin_sha256": sha256_file(bin_path),
        }
        receipt["ir_trace"] = {
            "layer_count": sum(1 for line in xml_path.read_text().splitlines() if '<layer ' in line),
            "contains_scatter": "scatter" in xml_path.read_text().lower(),
            "contains_index_add": "index_add" in xml_path.read_text().lower(),
            "contains_nonzero": "nonzero" in xml_path.read_text().lower(),
        }
        if any(receipt["ir_trace"][key] for key in ("contains_scatter", "contains_index_add", "contains_nonzero")):
            raise AssertionError(f"forbidden IR operation: {receipt['ir_trace']}")
        receipt["status"] = "passed"
        receipt["finished_seconds"] = time.perf_counter() - start_wall
        receipt.update(peak_rss())
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        receipt["finished_seconds"] = time.perf_counter() - start_wall
        receipt.update(peak_rss())
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
