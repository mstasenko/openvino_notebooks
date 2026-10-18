#!/usr/bin/env python3
"""CPU-only SeedVR2 3B prelude/postlude OpenVINO storage proof.

The graph boundaries are taken directly from the pinned ``NaDiT`` source:
``vid_in.proj`` plus ``txt_in`` and ``emb_in`` form the prelude; ``vid_out_norm``
plus ``vid_out_ada`` and ``vid_out.proj`` form the postlude.  Only those exact
parameter prefixes are materialized from the official FP16 safetensors file.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


WEIGHT_SHA256 = "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"
WEIGHT_BYTES = 6_783_018_808
WEIGHT_PREFIXES = (
    "emb_in.",
    "txt_in.",
    "vid_in.",
    "vid_out.",
    "vid_out_ada.",
    "vid_out_norm.",
)
SOURCE_FILES = (
    "configs_3b/main.yaml",
    "src/models/dit_3b/nadit.py",
    "src/models/dit_3b/patch/patch_v1.py",
    "src/models/dit_3b/embedding.py",
    "src/models/dit_3b/modulation.py",
    "src/models/dit_3b/normalization.py",
    "src/models/dit_3b/na.py",
    "src/common/cache.py",
    "src/common/distributed/ops.py",
)

# Chosen before running results.  The direct FP32 export threshold follows
# the existing structural probe.  Saved-FP16 storage gets the reviewed 5e-3
# absolute bound while retaining the same relative bound.
EAGER_MAX_ABS = 1.0e-3
EAGER_RMS_REL = 1.0e-4
SAVED_MAX_ABS = 5.0e-3
SAVED_RMS_REL = 1.0e-4
DISTINCT_MAX_ABS = 1.0e-7

CASES = (
    ("t1_landscape", 1, 48, 60),
    ("t1_portrait", 1, 60, 48),
    ("t2_landscape", 2, 48, 60),
    ("t2_portrait", 2, 60, 48),
)
TEXT_TOKENS = 58
VID_IN_CHANNELS = 33
VID_DIM = 2560
TXT_IN_DIM = 5120
EMB_DIM = 15360
VID_OUT_CHANNELS = 16
PATCH = (1, 2, 2)

# NaDiT's output Ada call uses the same cache key as the first video Ada
# modulation in block 0.  The full successful DiT IR under
# evidence/video-crop-probe/ir/dit.xml shows the output branch consuming that
# cached [tokens, 2560, 3] value.  It is the first (attn) 3*2560 group of the
# 15360-wide time embedding, not the output call's own one-layer reshape.
POST_CACHE_KEY = "emb_repeat_0_vid"
POST_CACHE_LAYER_INDEX = 0


def exception_record(exc: BaseException) -> Dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_source(source: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    files: List[Dict[str, Any]] = []
    for relative in SOURCE_FILES:
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                file_digest.update(chunk)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_digest.hexdigest()})
    return {"files": files, "aggregate_sha256": digest.hexdigest()}


def peak_rss() -> Dict[str, int]:
    kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"peak_rss_kib": kib, "peak_rss_bytes": kib * 1024}


def load_openvino(runtime_root: Path) -> Any:
    root = str(runtime_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import openvino as ov

    version = str(ov.__version__)
    if not version.startswith("2026.3.0"):
        raise RuntimeError(f"OpenVINO runtime mismatch: expected 2026.3.0, loaded {version}")
    return ov


def patch_cpu_only(torch_module: Any) -> Dict[str, Any]:
    """Prevent source import-time backend probes from touching real devices."""

    torch_module.cuda.is_available = lambda: False
    torch_module.cuda.device_count = lambda: 0
    if hasattr(torch_module, "xpu"):
        torch_module.xpu.is_available = lambda: False
        if hasattr(torch_module.xpu, "device_count"):
            torch_module.xpu.device_count = lambda: 0
    if hasattr(torch_module.backends, "mps"):
        torch_module.backends.mps.is_available = lambda: False
    return {
        "cuda_available": False,
        "cuda_device_count": 0,
        "xpu_available": False,
        "mps_available": False,
        "physical_accelerator_enumeration": "not called; source checks patched",
    }


def make_model(source: Path, weights_path: Path, torch_module: Any) -> Tuple[Any, Dict[str, Any]]:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from src.models.dit_3b.nadit import NaDiT

    model = NaDiT(
        vid_in_channels=VID_IN_CHANNELS,
        vid_out_channels=VID_OUT_CHANNELS,
        vid_dim=VID_DIM,
        txt_in_dim=TXT_IN_DIM,
        txt_dim=VID_DIM,
        emb_dim=EMB_DIM,
        heads=20,
        head_dim=128,
        expand_ratio=4,
        norm="fusedrms",
        norm_eps=1.0e-5,
        ada="single",
        qk_bias=False,
        qk_norm="fusedrms",
        patch_size=PATCH,
        num_layers=0,
        block_type=[],
        mm_layers=0,
        mlp_type="swiglu",
        patch_type="v1",
        rope_type="mmrope3d",
        rope_dim=128,
        window=None,
        window_method=None,
        msa_type=None,
        mca_type=None,
        txt_in_norm="fusedln",
        txt_in_norm_scale_factor=0.01,
        txt_proj_type="linear",
        vid_out_norm="fusedrms",
        attention_mode="sdpa",
    ).float().eval()

    from safetensors import safe_open

    expected = set(model.state_dict().keys())
    selected: Dict[str, Any] = {}
    metadata: List[Dict[str, Any]] = []
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = sorted(key for key in handle.keys() if key.startswith(WEIGHT_PREFIXES))
        if set(keys) != expected:
            raise AssertionError(
                f"selected parameter keys differ: selected={len(keys)} model={len(expected)}"
            )
        for key in keys:
            value = handle.get_tensor(key)
            selected[key] = value.to(dtype=torch_module.float32)
            metadata.append(
                {
                    "key": key,
                    "shape": list(value.shape),
                    "stored_dtype": str(value.dtype),
                    "elements": int(value.numel()),
                }
            )
    load_result = model.load_state_dict(selected, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise AssertionError(f"strict component load mismatch: {load_result}")
    payload_elements = sum(item["elements"] for item in metadata)
    return model, {
        "class": "src.models.dit_3b.nadit.NaDiT",
        "config": {
            "vid_in_channels": VID_IN_CHANNELS,
            "vid_out_channels": VID_OUT_CHANNELS,
            "vid_dim": VID_DIM,
            "txt_in_dim": TXT_IN_DIM,
            "txt_dim": VID_DIM,
            "emb_dim": EMB_DIM,
            "patch_size": list(PATCH),
            "vid_out_norm": "fusedrms",
            "num_layers": 0,
        },
        "loaded_prefixes": list(WEIGHT_PREFIXES),
        "tensor_count": len(metadata),
        "payload_elements": payload_elements,
        "tensors": metadata,
        "weights_cast_for_eager": "official F16 tensors loaded and cast to FP32; saved IR compresses constants to FP16",
    }


class PreludeWrapper(torch.nn.Module):
    def __init__(self, model: Any, torch_module: Any) -> None:
        super().__init__()
        super().__init__()
        self.torch = torch_module
        self.vid_proj = model.vid_in.proj
        self.txt_in = model.txt_in
        self.emb_in = model.emb_in

    def eval(self) -> "PreludeWrapper":
        self.vid_proj.eval()
        self.txt_in.eval()
        self.emb_in.eval()
        return self

    def forward(self, vid: Any, txt: Any, timestep: Any) -> Tuple[Any, Any, Any]:
        # Exact source PatchIn ordering for patch=(1,2,2), with batch=1.
        from einops import rearrange

        patched = rearrange(
            vid,
            "b c (T t) (H h) (W w) -> b T H W (t h w c)",
            t=PATCH[0],
            h=PATCH[1],
            w=PATCH[2],
        )
        vid_tokens = self.vid_proj(patched)
        txt_tokens = self.txt_in(txt)
        emb = self.emb_in(timestep, device=vid_tokens.device, dtype=vid_tokens.dtype)
        return vid_tokens, txt_tokens, emb


class PostludeWrapper(torch.nn.Module):
    def __init__(self, model: Any, torch_module: Any) -> None:
        super().__init__()
        super().__init__()
        self.torch = torch_module
        self.vid_out_norm = model.vid_out_norm
        self.vid_out_ada = model.vid_out_ada
        self.vid_proj = model.vid_out.proj

    def eval(self) -> "PostludeWrapper":
        self.vid_out_norm.eval()
        self.vid_out_ada.eval()
        self.vid_proj.eval()
        return self

    def forward(self, hidden: Any, emb: Any) -> Any:
        b, temporal, height, width, dim = hidden.shape
        flat = hidden.reshape(-1, dim)
        flat = self.vid_out_norm(flat)
        # The full NaDiT body first caches block-0 video/attn modulation under
        # ``emb_repeat_0_vid``.  NaDiT's output Ada call has idx=0 and the same
        # branch tag, so Cache returns that [tokens, D, 3] value after the
        # output call's initial (otherwise incompatible) reshape.  Recreate
        # that exact cache state at this isolated boundary.
        from src.common.cache import Cache

        first_layer = emb.reshape(1, VID_DIM, 2, 3)[:, :, POST_CACHE_LAYER_INDEX, :]
        cached = first_layer.expand(flat.shape[0], -1, -1)
        cache = Cache(disable=False, cache={POST_CACHE_KEY: cached})
        # The value is already cached; this only selects the source branch.
        hid_len = self.torch.ones((1,), dtype=self.torch.long, device=flat.device)
        flat = self.vid_out_ada(
            flat,
            emb=emb,
            layer="out",
            mode="in",
            hid_len=hid_len,
            cache=cache,
            branch_tag="vid",
        )
        flat = self.vid_proj(flat)
        flat = flat.reshape(b, temporal, height, width, PATCH[0], PATCH[1], PATCH[2], VID_OUT_CHANNELS)
        flat = flat.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return flat.reshape(b, VID_OUT_CHANNELS, temporal * PATCH[0], height * PATCH[1], width * PATCH[2])


def trusted_prelude(model: Any, vid: Any, txt: Any, timestep: Any, torch_module: Any) -> Tuple[Any, Any, Any]:
    """Call the actual NaDiT source boundary, then reshape only for comparison."""

    from src.models.dit_3b import na
    from src.common.cache import Cache

    batch = []
    for index in range(vid.shape[0]):
        batch.append(vid[index].permute(1, 2, 3, 0))
    flat_vid, shape = na.flatten(batch)
    tokens, patched_shape = model.vid_in(flat_vid, shape, Cache(disable=True))
    video = tokens.reshape(1, int(patched_shape[0, 0]), int(patched_shape[0, 1]), int(patched_shape[0, 2]), VID_DIM)
    text = model.txt_in(txt)
    emb = model.emb_in(timestep, device=vid.device, dtype=vid.dtype)
    return video, text, emb


def trusted_post(model: Any, hidden: Any, emb: Any, torch_module: Any) -> Any:
    """Call the actual NaDiT output boundary and reshape to public video layout."""

    from src.common.cache import Cache

    b, temporal, height, width, dim = hidden.shape
    if b != 1:
        raise ValueError("bounded proof uses batch=1")
    flat = hidden.reshape(-1, dim)
    patch_shape = torch_module.tensor([[temporal, height, width]], dtype=torch_module.long)
    original_shape = torch_module.tensor(
        [[temporal * PATCH[0], height * PATCH[1], width * PATCH[2]]], dtype=torch_module.long
    )
    cache = Cache(disable=False)
    cache.cache["patch.vid_shape_before_patchify"] = original_shape
    first_layer = emb.reshape(1, VID_DIM, 2, 3)[:, :, POST_CACHE_LAYER_INDEX, :]
    cache.cache[POST_CACHE_KEY] = torch_module.repeat_interleave(
        first_layer, patch_shape.prod(-1), dim=0
    )
    flat = model.vid_out_norm(flat)
    flat = model.vid_out_ada(
        flat,
        emb=emb,
        layer="out",
        mode="in",
        hid_len=patch_shape.prod(-1),
        cache=cache,
        branch_tag="vid",
    )
    flat, _ = model.vid_out(flat, patch_shape, cache)
    return flat.reshape(1, temporal, height * PATCH[1], width * PATCH[2], VID_OUT_CHANNELS).permute(
        0, 4, 1, 2, 3
    )


def tensor_summary(value: Any, torch_module: Any) -> Dict[str, Any]:
    value = value.detach().cpu().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch_module.isfinite(value).all().item()),
        "min": float(value.min().item()) if value.numel() else None,
        "max": float(value.max().item()) if value.numel() else None,
    }


def compare(reference: Any, candidate: Any, torch_module: Any) -> Dict[str, Any]:
    reference = reference.detach().cpu().float()
    candidate = candidate.detach().cpu().float()
    if reference.shape != candidate.shape:
        return {
            "shape_compatible": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "max_abs_error": None,
            "rms_relative_error": None,
        }
    difference = candidate - reference
    max_abs = float(difference.abs().max().item()) if difference.numel() else 0.0
    rms = float(torch_module.sqrt(torch_module.mean(difference.square())).item())
    ref_rms = float(torch_module.sqrt(torch_module.mean(reference.square())).item())
    return {
        "shape_compatible": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "max_abs_error": max_abs,
        "rms_relative_error": rms / max(ref_rms, 1.0e-12),
    }


def ports(model: Any) -> Dict[str, Any]:
    def one(port: Any) -> Dict[str, Any]:
        shape = port.get_partial_shape()
        return {
            "names": sorted(str(name) for name in port.get_names()),
            "element_type": str(port.get_element_type()),
            "partial_shape": str(shape),
            "shape": [None if dim.is_dynamic else int(dim.get_length()) for dim in shape],
        }

    return {"inputs": [one(port) for port in model.inputs], "outputs": [one(port) for port in model.outputs]}


def set_partial_shape(model: Any, ov: Any, shapes: Sequence[Sequence[int | None]]) -> None:
    mapping: Dict[Any, Any] = {}
    for index, shape in enumerate(shapes):
        mapping[model.input(index)] = ov.PartialShape(
            [ov.Dimension.dynamic() if value is None else int(value) for value in shape]
        )
    model.reshape(mapping)


def conversion(wrapper: Any, inputs: Sequence[Any], ov: Any, torch_module: Any) -> Tuple[Any, Dict[str, Any]]:
    started = time.perf_counter()
    try:
        with torch_module.inference_mode():
            model = ov.convert_model(wrapper, example_input=tuple(inputs))
        return model, {
            "status": "ok",
            "mode": "eager",
            "elapsed_seconds": time.perf_counter() - started,
            "attempts": [],
        }
    except Exception as eager_error:
        return None, {
            "status": "error",
            "mode": "eager",
            "elapsed_seconds": time.perf_counter() - started,
            "exception": exception_record(eager_error),
        }


def compile_and_infer(
    model: Any,
    cases: Sequence[Tuple[str, Sequence[Any], Sequence[Any]]],
    ov: Any,
    torch_module: Any,
    properties: Dict[str, Any],
    wrapper: Any,
    reference_fn: Any,
    saved_thresholds: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    core = ov.Core()
    compiled = core.compile_model(model, "CPU", properties)
    compile_seconds = time.perf_counter() - started
    request = compiled.create_infer_request()
    result: Dict[str, Any] = {
        "compile_seconds": compile_seconds,
        "ports": ports(compiled),
        "cases": [],
        "thresholds": {
            "max_abs_error_le": SAVED_MAX_ABS if saved_thresholds else EAGER_MAX_ABS,
            "rms_relative_error_le": SAVED_RMS_REL if saved_thresholds else EAGER_RMS_REL,
        },
    }
    for name, inputs, expected in cases:
        with torch_module.inference_mode():
            reference = reference_fn(*inputs)
        started = time.perf_counter()
        raw = request.infer({port: value.detach().cpu().numpy() for port, value in zip(compiled.inputs, inputs)})
        infer_seconds = time.perf_counter() - started
        candidate = [torch_module.from_numpy(raw[port]).clone() for port in compiled.outputs]
        reference_values = list(reference) if isinstance(reference, tuple) else [reference]
        comparisons = [
            compare(ref, got, torch_module)
            for ref, got in zip(reference_values, candidate)
        ]
        output_summaries = [tensor_summary(value, torch_module) for value in candidate]
        case_result = {
            "name": name,
            "input_shapes": [list(value.shape) for value in inputs],
            "expected_output_shapes": [list(value) for value in expected],
            "outputs": output_summaries,
            "infer_seconds": infer_seconds,
            "comparisons": comparisons,
            "finite": all(item["finite"] for item in output_summaries),
            "shape_contract": all(
                item["shape_compatible"] and item["candidate_shape"] == list(shape)
                for item, shape in zip(comparisons, expected)
            ),
        }
        case_result["within_threshold"] = all(
            item["shape_compatible"]
            and float(item["max_abs_error"]) <= result["thresholds"]["max_abs_error_le"]
            and float(item["rms_relative_error"]) <= result["thresholds"]["rms_relative_error_le"]
            for item in comparisons
        )
        result["cases"].append(case_result)
    result["status"] = "passed" if all(
        case["finite"] and case["shape_contract"] and case["within_threshold"]
        for case in result["cases"]
    ) else "failed"
    return result


def save_fp16(ov: Any, model: Any, path: Path) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ov.save_model(model, str(path), compress_to_fp16=True)
    files = {
        "xml": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "bin": {
            "path": str(path.with_suffix(".bin")),
            "bytes": path.with_suffix(".bin").stat().st_size,
            "sha256": sha256_file(path.with_suffix(".bin")),
        },
    }
    return {
        "compress_to_fp16": True,
        "elapsed_seconds": time.perf_counter() - started,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }


def constant_storage(model: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Constant":
            continue
        key = str(node.get_output_element_type(0))
        counts[key] = counts.get(key, 0) + 1
    return counts


def make_inputs(torch_module: Any, generator: Any) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for name, temporal, height, width in CASES:
        video = torch_module.randn(
            (1, VID_IN_CHANNELS, temporal, height, width), generator=generator, dtype=torch_module.float32
        )
        text = torch_module.randn((TEXT_TOKENS, TXT_IN_DIM), generator=generator, dtype=torch_module.float32)
        timestep = torch_module.tensor([0.37 + 0.11 * temporal], dtype=torch_module.float32)
        hidden = torch_module.randn(
            (1, temporal, height // 2, width // 2, VID_DIM), generator=generator, dtype=torch_module.float32
        )
        emb = torch_module.randn((1, EMB_DIM), generator=generator, dtype=torch_module.float32)
        result[name] = {
            "pre": (video, text, timestep),
            "post": (hidden, emb),
            "pre_expected": ((1, temporal, height // 2, width // 2, VID_DIM), (TEXT_TOKENS, VID_DIM), (1, EMB_DIM)),
            "post_expected": ((1, VID_OUT_CHANNELS, temporal, height, width),),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ir-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt: Dict[str, Any] = {
        "probe": "seedvr2-seed-prepost-openvino-fp16-storage",
        "status": "not_run",
        "cpu_only": True,
        "advisory_lock": "not inspected or used (explicit task requirement)",
        "command_argv": [str(value) for value in sys.argv],
        "timing_wrapper": "/usr/bin/time -v",
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "thresholds_selected_before_results": {
            "eager_openvino_fp32": {"max_abs_error_le": EAGER_MAX_ABS, "rms_relative_error_le": EAGER_RMS_REL},
            "saved_ir_fp16_storage": {"max_abs_error_le": SAVED_MAX_ABS, "rms_relative_error_le": SAVED_RMS_REL},
            "distinct_max_abs_gt": DISTINCT_MAX_ABS,
        },
        "requested_contract": {
            "cases": [
                {"name": name, "temporal": temporal, "height": height, "width": width}
                for name, temporal, height, width in CASES
            ],
            "text_tokens": TEXT_TOKENS,
            "video_input": [1, VID_IN_CHANNELS, "T", "H", "W"],
            "prelude_outputs": [
                [1, "T", "H/2", "W/2", VID_DIM],
                [TEXT_TOKENS, VID_DIM],
                [1, EMB_DIM],
            ],
            "postlude_inputs": [[1, "T", "H/2", "W/2", VID_DIM], [1, EMB_DIM]],
            "postlude_output": [1, VID_OUT_CHANNELS, "T", "H", "W"],
            "public_io": "FP32; saved graph constants compressed to FP16",
        },
        "stages": {},
    }
    run_started = time.perf_counter()
    try:
        import torch

        receipt["cpu_guard"] = patch_cpu_only(torch)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20260918)
        source = args.source.resolve()
        weights = args.weights.resolve()
        runtime = args.runtime.resolve()
        if weights.stat().st_size != WEIGHT_BYTES:
            raise AssertionError(f"weight byte size mismatch: {weights.stat().st_size}")
        weight_hash = sha256_file(weights)
        if weight_hash != WEIGHT_SHA256:
            raise AssertionError(f"weight SHA256 mismatch: {weight_hash}")
        receipt["provenance"] = {
            "source_root": str(source),
            "source_files": hash_source(source),
            "weights": {"path": str(weights), "bytes": weights.stat().st_size, "sha256": weight_hash},
            "runtime_root": str(runtime),
        }
        ov = load_openvino(runtime)
        receipt["runtime"] = {
            "openvino": str(ov.__version__),
            "module": str(getattr(ov, "__file__", "<unknown>")),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        }
        model, structure = make_model(source, weights, torch)
        receipt["structure"] = structure
        receipt["postlude_cache_contract"] = {
            "cache_key": POST_CACHE_KEY,
            "layer_index": POST_CACHE_LAYER_INDEX,
            "embedding_reshape": [1, VID_DIM, 2, 3],
            "cached_shape": ["tokens", VID_DIM, 3],
            "selected_embedding_columns": [0, 3 * VID_DIM],
            "source_refs": [
                "src/models/dit_3b/nablocks/mmsr_block.py:81,98-113",
                "src/models/dit_3b/modulation.py:75-86",
                "src/models/dit_3b/nadit.py:233-244",
                "src/common/cache.py:26-36",
            ],
            "prior_successful_graph": {
                "path": "evidence/video-crop-probe/ir/dit.xml",
                "block0_embedding_reshape_layer": "52",
                "block0_cached_modulation_layer": "73",
                "block0_cached_modulation_shape": [5760, 2560, 3],
                "output_ada_split_layer": "19888",
                "output_ada_input_edge": {"from_layer": "73", "from_port": "3", "to_layer": "19888", "to_port": "0"},
                "interpretation": "successful full DiT graph wires block-0 video/attn cached modulation directly into vid_out_ada",
            },
        }
        prelude = PreludeWrapper(model, torch).eval()
        postlude = PostludeWrapper(model, torch).eval()
        generator = torch.Generator(device="cpu").manual_seed(20260919)
        inputs = make_inputs(torch, generator)
        pre_example = inputs["t1_landscape"]["pre"]
        post_example = inputs["t1_landscape"]["post"]
        try:
            with torch.inference_mode():
                trusted_post(model, *post_example, torch)
            receipt["trusted_eager_postlude"] = {"status": "ok"}
        except BaseException as error:
            receipt["trusted_eager_postlude"] = {
                "status": "error",
                "exception": exception_record(error),
            }
        pre_model, pre_conversion = conversion(prelude, pre_example, ov, torch)
        post_model, post_conversion = conversion(postlude, post_example, ov, torch)
        receipt["conversion"] = {"prelude": pre_conversion, "postlude": post_conversion}
        if pre_model is None:
            receipt["status"] = "failed"
            return 1
        set_partial_shape(
            pre_model,
            ov,
            [[1, VID_IN_CHANNELS, None, None, None], [TEXT_TOKENS, TXT_IN_DIM], [1]],
        )
        receipt["dynamic_reshape"] = {"prelude": ports(pre_model)}
        properties = {
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": "2",
            "INFERENCE_PRECISION_HINT": ov.Type.f32,
        }

        def pre_reference(vid: Any, txt: Any, timestep: Any) -> Tuple[Any, Any, Any]:
            return trusted_prelude(model, vid, txt, timestep, torch)

        def post_reference(hidden: Any, emb: Any) -> Any:
            return trusted_post(model, hidden, emb, torch)

        pre_cases = [
            (name, inputs[name]["pre"], inputs[name]["pre_expected"])
            for name, _, _, _ in CASES
        ]
        post_cases = [
            (name, inputs[name]["post"], inputs[name]["post_expected"])
            for name, _, _, _ in CASES
        ]
        # The in-memory dynamic graphs are compiled once each and each request
        # is reused over all four temporal/orientation cases.
        receipt["stages"]["prelude_fp32_dynamic"] = compile_and_infer(
            pre_model, pre_cases, ov, torch, properties, prelude, pre_reference, False
        )
        ir_dir = args.ir_dir.resolve()
        saved: Dict[str, Any] = {}
        pre_xml = ir_dir / "prelude_dynamic_fp16.xml"
        pre_saved = save_fp16(ov, pre_model, pre_xml)
        pre_reloaded = ov.Core().read_model(str(pre_xml))
        pre_saved["reloaded_ports"] = ports(pre_reloaded)
        pre_saved["constant_storage_types"] = constant_storage(pre_reloaded)
        pre_saved["readback"] = compile_and_infer(
            pre_reloaded, pre_cases, ov, torch, properties, prelude, pre_reference, True
        )
        saved["prelude_dynamic_fp16"] = pre_saved

        if post_model is None:
            receipt["stages"]["postlude_trusted_source"] = {
                "status": "failed_at_conversion",
                "failure": post_conversion,
                "note": "The source-equivalent block-0 cache contract was applied at the isolated postlude boundary, but conversion failed.",
            }
            receipt["saved_ir"] = saved
            receipt["status"] = "failed"
            return 1

        set_partial_shape(post_model, ov, [[1, None, None, None, VID_DIM], [1, EMB_DIM]])
        receipt["dynamic_reshape"]["postlude"] = ports(post_model)
        receipt["stages"]["postlude_fp32_dynamic"] = compile_and_infer(
            post_model, post_cases, ov, torch, properties, postlude, post_reference, False
        )
        post_xml = ir_dir / "postlude_dynamic_fp16.xml"
        post_saved = save_fp16(ov, post_model, post_xml)
        post_reloaded = ov.Core().read_model(str(post_xml))
        post_saved["reloaded_ports"] = ports(post_reloaded)
        post_saved["constant_storage_types"] = constant_storage(post_reloaded)
        post_saved["readback"] = compile_and_infer(
            post_reloaded, post_cases, ov, torch, properties, postlude, post_reference, True
        )
        saved["postlude_dynamic_fp16"] = post_saved
        receipt["saved_ir"] = saved
        required = [
            receipt["stages"]["prelude_fp32_dynamic"]["status"] == "passed",
            receipt["stages"]["postlude_fp32_dynamic"]["status"] == "passed",
            saved["prelude_dynamic_fp16"]["readback"]["status"] == "passed",
            saved["postlude_dynamic_fp16"]["readback"]["status"] == "passed",
        ]
        receipt["status"] = "passed" if all(required) else "failed"
    except BaseException as error:
        receipt["status"] = "error"
        receipt["exception"] = exception_record(error)
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - run_started
        receipt["peak_rss"] = peak_rss()
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
