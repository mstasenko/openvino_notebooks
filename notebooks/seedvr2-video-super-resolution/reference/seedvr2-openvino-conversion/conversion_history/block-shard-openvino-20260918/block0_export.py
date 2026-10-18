#!/usr/bin/env python3
"""Export and validate one trusted-upstream SeedVR2 3B transformer block.

This helper intentionally constructs only ``blocks.0``.  It loads only the
29 safetensors entries under ``blocks.0.``, strips that prefix, and runs a
fixed, real-shape block contract on CPU.  The window-operation specialization
is imported from the trial's reviewed ``openvino-probe.py`` helper; the source
window math is not reimplemented here.
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


TRIAL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(os.environ.get("SEEDVR_SOURCE", TRIAL_ROOT / "source_snapshot"))
DEFAULT_WEIGHTS = Path(os.environ.get("SEEDVR_WEIGHTS", TRIAL_ROOT / "seedvr2_ema_3b_fp16.safetensors"))
DEFAULT_OV_ROOT = Path(os.environ.get("SEEDVR_OPENVINO_ROOT", TRIAL_ROOT / "runtime-openvino"))
DEFAULT_OUT = Path(__file__).resolve().parent / "block0_fp16.xml"
REVIEWED_HELPER = TRIAL_ROOT / "probes" / "openvino-probe.py"

BLOCK_PREFIX = "blocks.0."
EXPECTED_TENSOR_COUNT = 29
EXPECTED_PAYLOAD_BYTES = 317_266_986
EXPECTED_VID_SHAPE = (1, 48, 60)
EXPECTED_TXT_SHAPE = (1, 58)
EXPECTED_VID_TOKENS = 2_880
EXPECTED_TXT_TOKENS = 58
EXPECTED_DIM = 2_560
EXPECTED_EMB_DIM = 15_360
EXPECTED_OUTPUTS = ((EXPECTED_VID_TOKENS, EXPECTED_DIM), (EXPECTED_TXT_TOKENS, EXPECTED_DIM))

# Keep these equal to the trial's reviewed OpenVINO probe contract.  A failed
# comparison is evidence, not a reason to relax the values.
MAX_ABS_ERROR_THRESHOLD = 1.0e-3
RMS_REL_ERROR_THRESHOLD = 1.0e-4
DISTINCT_OUTPUT_THRESHOLD = 1.0e-7


def _short(value: Any, limit: int = 16_000) -> str:
    text = str(value).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def filter_block0_keys(keys: Iterable[str]) -> dict[str, str]:
    """Return exact block-0 source-to-stripped key mapping.

    This pure helper is tested separately so a broad/full-checkpoint loader
    cannot silently replace the bounded shard loader.
    """

    selected = sorted(key for key in keys if key.startswith(BLOCK_PREFIX))
    if len(selected) != EXPECTED_TENSOR_COUNT:
        raise AssertionError(
            f"block-0 tensor count mismatch: {len(selected)} != {EXPECTED_TENSOR_COUNT}"
        )
    stripped = {key: key[len(BLOCK_PREFIX) :] for key in selected}
    if len(stripped) != len(selected) or any(not value for value in stripped.values()):
        raise AssertionError("block-0 prefix stripping produced an invalid mapping")
    return stripped


def validate_contract_shapes(
    vid_shape: Sequence[int],
    txt_shape: Sequence[int],
    emb_shape: Sequence[int],
) -> None:
    if tuple(vid_shape) != (EXPECTED_VID_TOKENS, EXPECTED_DIM):
        raise AssertionError(f"vid input shape mismatch: {tuple(vid_shape)}")
    if tuple(txt_shape) != (EXPECTED_TXT_TOKENS, EXPECTED_DIM):
        raise AssertionError(f"txt input shape mismatch: {tuple(txt_shape)}")
    if tuple(emb_shape) != (1, EXPECTED_EMB_DIM):
        raise AssertionError(f"emb input shape mismatch: {tuple(emb_shape)}")


def validate_embedded_shapes(vid_shape: Sequence[int], txt_shape: Sequence[int]) -> None:
    if tuple(vid_shape) != EXPECTED_VID_SHAPE:
        raise AssertionError(f"embedded vid_shape mismatch: {tuple(vid_shape)}")
    if tuple(txt_shape) != EXPECTED_TXT_SHAPE:
        raise AssertionError(f"embedded txt_shape mismatch: {tuple(txt_shape)}")
    if math.prod(vid_shape) != EXPECTED_VID_TOKENS:
        raise AssertionError("embedded vid_shape token product mismatch")
    if math.prod(txt_shape) != EXPECTED_TXT_TOKENS:
        raise AssertionError("embedded txt_shape token product mismatch")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(source: Path) -> tuple[str, int]:
    """Hash source files relevant to direct block construction and imports."""

    roots = [source / "configs_3b", source / "src" / "models" / "dit_3b"]
    roots += [
        source / "src" / "common" / "cache.py",
        source / "src" / "common" / "half_precision_fixes.py",
        source / "src" / "common" / "distributed" / "ops.py",
        source / "src" / "common" / "distributed" / "advanced.py",
        source / "src" / "common" / "distributed" / "basic.py",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*.py") if ".git" not in path.parts)
            files.extend(path for path in root.rglob("*.yaml") if ".git" not in path.parts)
    files = sorted(set(files))
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def _peak_rss() -> dict[str, int]:
    kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"peak_rss_kib": kib, "peak_rss_bytes": kib * 1024}


def _cpu_only_guard(torch: Any) -> dict[str, Any]:
    # Import-time project guards: no CUDA/XPU/MPS query or use is permitted in
    # this bounded proof.  The actual computation remains CPU tensors.
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    if hasattr(torch, "xpu"):
        torch.xpu.is_available = lambda: False
        if hasattr(torch.xpu, "device_count"):
            torch.xpu.device_count = lambda: 0
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False
    return {
        "cuda_available": False,
        "cuda_device_count": 0,
        "xpu_available": False,
        "mps_available": False,
        "mode": "synthetic_cpu_only_guards",
    }


def _load_block_weights(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate then materialize only the block-0 safetensors entries."""

    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        mapping = filter_block0_keys(handle.keys())
        payload_bytes = 0
        tensor_meta: list[dict[str, Any]] = []
        for source_key in mapping:
            view = handle.get_slice(source_key)
            shape = tuple(int(value) for value in view.get_shape())
            dtype = str(view.get_dtype())
            if dtype != "F16":
                raise AssertionError(f"{source_key} has unexpected safetensors dtype {dtype}")
            elements = math.prod(shape)
            payload_bytes += elements * 2
            tensor_meta.append(
                {
                    "source_key": source_key,
                    "stripped_key": mapping[source_key],
                    "shape": list(shape),
                    "dtype": dtype,
                    "payload_bytes": elements * 2,
                }
            )
        if payload_bytes != EXPECTED_PAYLOAD_BYTES:
            raise AssertionError(
                f"block-0 payload bytes mismatch: {payload_bytes} != {EXPECTED_PAYLOAD_BYTES}"
            )
        # The assertions above occur before any tensor materialization or
        # model construction.  Each selected entry is then loaded explicitly.
        state = {mapping[key]: handle.get_tensor(key) for key in mapping}
    return state, {
        "selected_tensor_count": len(mapping),
        "selected_payload_bytes": payload_bytes,
        "tensors": tensor_meta,
    }


def _build_block(source: Path, state: Mapping[str, Any], torch: Any) -> Any:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from src.models.dit_3b.modulation import get_ada_layer
    from src.models.dit_3b.nablocks.mmsr_block import NaMMSRTransformerBlock
    from src.models.dit_3b.normalization import get_norm_layer

    norm = get_norm_layer("fusedrms")
    qk_norm = get_norm_layer("fusedrms")
    block = NaMMSRTransformerBlock(
        vid_dim=2560,
        txt_dim=2560,
        emb_dim=15360,
        heads=20,
        head_dim=128,
        expand_ratio=4,
        norm=norm,
        norm_eps=1.0e-5,
        ada=get_ada_layer("single"),
        qk_bias=False,
        qk_norm=qk_norm,
        mlp_type="swiglu",
        shared_weights=False,
        rope_type="mmrope3d",
        rope_dim=128,
        is_last_layer=False,
        attention_mode="sdpa",
        window=(4, 3, 3),
        window_method="720pwin_by_size_bysize",
    ).to(device="cpu", dtype=torch.float32).eval()
    expected_keys = set(block.state_dict().keys())
    if set(state) != expected_keys:
        missing = sorted(expected_keys - set(state))
        unexpected = sorted(set(state) - expected_keys)
        raise AssertionError(f"block state key mismatch: missing={missing}, unexpected={unexpected}")
    block.load_state_dict(dict(state), strict=True)
    return block


def _reviewed_specialize(model: Any) -> dict[str, Any]:
    helper_hash = _sha256_file(REVIEWED_HELPER)
    spec = importlib.util.spec_from_file_location("seedvr_reviewed_openvino_probe", REVIEWED_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reviewed helper: {REVIEWED_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count = int(module._specialize_fixed_window_ops(model))
    return {
        "path": str(REVIEWED_HELPER),
        "sha256": helper_hash,
        "function": "_specialize_fixed_window_ops",
        "wrapped_attention_modules": count,
        "math": "reviewed upstream helper; fixed only for this static contract",
    }


def _stats(torch: Any, values: Sequence[Any]) -> list[dict[str, Any]]:
    result = []
    for value in values:
        result.append(
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(torch.isfinite(value).all().item()),
                "min": float(value.min().item()),
                "max": float(value.max().item()),
                "rms": float(torch.sqrt(torch.mean(value * value)).item()),
            }
        )
    return result


def _numpy_metrics(reference: Sequence[Any], candidate: Sequence[Any], np: Any) -> dict[str, Any]:
    per_output: list[dict[str, float]] = []
    for ref, got in zip(reference, candidate):
        ref_array = ref.detach().cpu().numpy() if hasattr(ref, "detach") else np.asarray(ref)
        got_array = got.detach().cpu().numpy() if hasattr(got, "detach") else np.asarray(got)
        diff = got_array.astype(np.float64) - ref_array.astype(np.float64)
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


def _assert_pair(
    label: str,
    reference_a: Sequence[Any],
    reference_b: Sequence[Any],
    candidate_a: Sequence[Any],
    candidate_b: Sequence[Any],
    np: Any,
) -> dict[str, Any]:
    for case, values in (("a", candidate_a), ("b", candidate_b)):
        shapes = tuple(tuple(int(dimension) for dimension in value.shape) for value in values)
        if shapes != EXPECTED_OUTPUTS:
            raise AssertionError(f"{label} {case} output shape mismatch: {shapes}")
        arrays = [value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value) for value in values]
        if any(not np.isfinite(array).all() for array in arrays):
            raise AssertionError(f"{label} {case} output contains non-finite values")
    metrics_a = _numpy_metrics(reference_a, candidate_a, np)
    metrics_b = _numpy_metrics(reference_b, candidate_b, np)
    candidate_a_arrays = [value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value) for value in candidate_a]
    candidate_b_arrays = [value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value) for value in candidate_b]
    delta = max(float(np.max(np.abs(a - b))) for a, b in zip(candidate_a_arrays, candidate_b_arrays))
    all_metrics = {
        "a": metrics_a,
        "b": metrics_b,
        "output_delta_max_abs": delta,
        "thresholds": {
            "max_abs_error": MAX_ABS_ERROR_THRESHOLD,
            "rms_relative_error": RMS_REL_ERROR_THRESHOLD,
            "distinct_output_max_abs": DISTINCT_OUTPUT_THRESHOLD,
        },
    }
    if metrics_a["max_abs_error"] > MAX_ABS_ERROR_THRESHOLD or metrics_b["max_abs_error"] > MAX_ABS_ERROR_THRESHOLD:
        raise AssertionError(f"{label} max absolute error exceeds fixed threshold: {all_metrics}")
    if metrics_a["max_rms_relative_error"] > RMS_REL_ERROR_THRESHOLD or metrics_b["max_rms_relative_error"] > RMS_REL_ERROR_THRESHOLD:
        raise AssertionError(f"{label} relative RMS error exceeds fixed threshold: {all_metrics}")
    if delta <= DISTINCT_OUTPUT_THRESHOLD:
        raise AssertionError(f"{label} outputs A/B are indistinguishable: {all_metrics}")
    return all_metrics


def _port_info(port: Any, index: int) -> dict[str, Any]:
    partial_shape = port.get_partial_shape()
    shape = [int(value) for value in partial_shape.to_shape()] if partial_shape.is_static else None
    try:
        names = sorted(str(name) for name in port.get_names())
    except Exception:
        names = []
    try:
        any_name = port.get_any_name()
    except Exception:
        any_name = None
    return {
        "index": index,
        "any_name": any_name,
        "names": names,
        "shape": shape,
        "partial_shape": str(partial_shape),
        "element_type": str(port.get_element_type()),
    }


def _run_openvino(compiled: Any, values: Sequence[Any], np: Any) -> tuple[Any, ...]:
    request = compiled.create_infer_request()
    arrays = [value.detach().cpu().numpy() for value in values]
    result = request.infer({port: array for port, array in zip(compiled.inputs, arrays)})
    return tuple(np.array(result[port], copy=True) for port in compiled.outputs)


def _json_error(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": _short(exc),
        "traceback": _short("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))),
    }


def _make_inputs(torch: Any, seed: int) -> tuple[Any, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn((EXPECTED_VID_TOKENS, EXPECTED_DIM), generator=generator, dtype=torch.float32),
        torch.randn((EXPECTED_TXT_TOKENS, EXPECTED_DIM), generator=generator, dtype=torch.float32),
        torch.randn((1, EXPECTED_EMB_DIM), generator=generator, dtype=torch.float32),
    )


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    source = Path(args.source).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    ov_root = Path(args.openvino_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    receipt_path = Path(args.receipt).expanduser().resolve()
    payload: dict[str, Any] = {
        "schema": 1,
        "status": "not_started",
        "scope": "one trusted-upstream FP16 SeedVR2 3B transformer block, block 0 only",
        "source": {"path": str(source)},
        "weights": {
            "path": str(weights),
            "mode": "selected safetensors block-0 keys only",
            "prefix": BLOCK_PREFIX,
            "expected_tensor_count": EXPECTED_TENSOR_COUNT,
            "expected_payload_bytes": EXPECTED_PAYLOAD_BYTES,
        },
        "model_args": {
            "vid_dim": 2560,
            "txt_dim": 2560,
            "emb_dim": 15360,
            "heads": 20,
            "head_dim": 128,
            "expand_ratio": 4,
            "norm": "fusedrms",
            "norm_eps": 1.0e-5,
            "ada": "single",
            "qk_bias": False,
            "qk_norm": "fusedrms",
            "mlp_type": "swiglu",
            "shared_weights": False,
            "rope_type": "mmrope3d",
            "rope_dim": 128,
            "window": [4, 3, 3],
            "window_method": "720pwin_by_size_bysize",
            "attention_mode": "sdpa",
            "is_last_layer": False,
        },
        "contract": {
            "input_shapes": {
                "vid": [EXPECTED_VID_TOKENS, EXPECTED_DIM],
                "txt": [EXPECTED_TXT_TOKENS, EXPECTED_DIM],
                "emb": [1, EXPECTED_EMB_DIM],
            },
            "input_dtypes": {"vid": "torch.float32", "txt": "torch.float32", "emb": "torch.float32"},
            "embedded_constants": {
                "vid_shape": [list(EXPECTED_VID_SHAPE)],
                "txt_shape": [list(EXPECTED_TXT_SHAPE)],
                "cache": "fresh Cache(disable=False) per wrapper call",
            },
            "output_shapes": [list(shape) for shape in EXPECTED_OUTPUTS],
        },
        "cpu": {
            "threads_requested": 4,
            "device": "CPU",
            "gpu_use": False,
            "openvino_root": str(ov_root),
        },
        "coordination": {
            # The archive recorded a host-local coordination path.  It is
            # intentionally not portable or shipped in this source snapshot.
            "advisory_lock_path": None,
            "mode": os.environ.get("SEEDVR_COORDINATION_MODE", "unspecified"),
            "explicit_user_waiver": os.environ.get("SEEDVR_EXPLICIT_LOCK_WAIVER", "false").lower() == "true",
            "load_sample": None,
        },
        "hashes": {},
        "timings_seconds": {},
        "memory": {"rss_unit": "bytes; Linux ru_maxrss source is KiB"},
        "errors": [],
    }
    start = time.perf_counter()
    try:
        load_sample = os.environ.get("SEEDVR_LOAD_SAMPLE_JSON")
        if load_sample:
            payload["coordination"]["load_sample"] = json.loads(load_sample)
        import numpy as np
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        payload["cpu"]["guard"] = _cpu_only_guard(torch)
        payload["hashes"] = {
            "weights_sha256": _sha256_file(weights),
            "reviewed_helper_sha256": _sha256_file(REVIEWED_HELPER),
            "source_tree_sha256": _source_hash(source)[0],
            "source_tree_file_count": _source_hash(source)[1],
        }
        payload["memory"]["before_shard_contract"] = _peak_rss()
        contract_start = time.perf_counter()
        state, shard_meta = _load_block_weights(weights)
        payload["timings_seconds"]["shard_contract_and_load"] = time.perf_counter() - contract_start
        payload["weights"].update(shard_meta)
        payload["memory"]["after_shard_load"] = _peak_rss()

        block = _build_block(source, state, torch)
        del state
        gc.collect()
        payload["memory"]["after_block_construction"] = _peak_rss()
        payload["weights"]["model_state_tensor_count"] = len(block.state_dict())
        payload["weights"]["model_state_payload_bytes_fp16"] = sum(
            int(value.numel()) * 2 for value in block.state_dict().values() if value.dtype in (torch.float16, torch.float32)
        )

        from src.common.cache import Cache

        class BlockWrapper(torch.nn.Module):
            def __init__(self, inner: Any):
                super().__init__()
                self.inner = inner
                self.register_buffer("vid_shape_const", torch.tensor([[1, 48, 60]], dtype=torch.long))
                self.register_buffer("txt_shape_const", torch.tensor([[58]], dtype=torch.long))

            def forward(self, vid: Any, txt: Any, emb: Any) -> tuple[Any, Any]:
                cache = Cache(disable=False)
                vid_out, txt_out, _, _ = self.inner(
                    vid,
                    txt,
                    self.vid_shape_const,
                    self.txt_shape_const,
                    emb,
                    cache,
                )
                return vid_out, txt_out

        wrapper = BlockWrapper(block).eval()
        inputs_a = _make_inputs(torch, 2026091801)
        inputs_b = _make_inputs(torch, 2026091802)
        validate_contract_shapes(*(value.shape for value in inputs_a))
        validate_contract_shapes(*(value.shape for value in inputs_b))
        validate_embedded_shapes(EXPECTED_VID_SHAPE, EXPECTED_TXT_SHAPE)

        eager_start = time.perf_counter()
        with torch.inference_mode():
            eager_a = tuple(wrapper(*inputs_a))
            eager_b = tuple(wrapper(*inputs_b))
        payload["timings_seconds"]["eager_reference"] = time.perf_counter() - eager_start
        for label, values in (("a", eager_a), ("b", eager_b)):
            if any(not bool(torch.isfinite(value).all().item()) for value in values):
                raise AssertionError(f"eager reference {label} contains non-finite values")
        eager_delta = max(float((a - b).abs().max().item()) for a, b in zip(eager_a, eager_b))
        if eager_delta <= DISTINCT_OUTPUT_THRESHOLD:
            raise AssertionError(f"eager reference inputs are indistinguishable: delta={eager_delta}")
        payload["reference"] = {
            "input_seeds": [2026091801, 2026091802],
            "a": _stats(torch, eager_a),
            "b": _stats(torch, eager_b),
            "output_delta_max_abs": eager_delta,
        }
        payload["memory"]["after_eager_reference"] = _peak_rss()

        specialization_start = time.perf_counter()
        specialization = _reviewed_specialize(block)
        payload["specialization"] = specialization
        payload["timings_seconds"]["reviewed_window_specialization"] = time.perf_counter() - specialization_start

        trace_start = time.perf_counter()
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, inputs_a, strict=False, check_trace=False).eval()
            frozen = torch.jit.freeze(traced)
            trace_a = tuple(frozen(*inputs_a))
            trace_b = tuple(frozen(*inputs_b))
        payload["timings_seconds"]["torchscript_trace_freeze"] = time.perf_counter() - trace_start
        payload["torchscript"] = {
            "status": "ok",
            "frozen": True,
            "outputs": [list(value.shape) for value in trace_a],
            "metrics": _assert_pair("TorchScript", eager_a, eager_b, trace_a, trace_b, np),
        }
        payload["memory"]["after_torchscript"] = _peak_rss()

        sys.path.insert(0, str(ov_root))
        import openvino as ov

        payload["openvino"] = {
            "version": str(getattr(ov, "__version__", "unknown")),
            "runtime_root": str(ov_root),
        }
        convert_start = time.perf_counter()
        ov_model = ov.convert_model(frozen, example_input=inputs_a)
        payload["timings_seconds"]["openvino_conversion"] = time.perf_counter() - convert_start
        if len(ov_model.inputs) != 3 or len(ov_model.outputs) != 2:
            raise AssertionError(
                f"OpenVINO graph IO mismatch: inputs={len(ov_model.inputs)}, outputs={len(ov_model.outputs)}"
            )
        ov_model.reshape({port: list(value.shape) for port, value in zip(ov_model.inputs, inputs_a)})
        payload["openvino"]["graph"] = {
            "inputs": [_port_info(port, index) for index, port in enumerate(ov_model.inputs)],
            "outputs": [_port_info(port, index) for index, port in enumerate(ov_model.outputs)],
        }
        payload["memory"]["after_openvino_conversion"] = _peak_rss()

        output.parent.mkdir(parents=True, exist_ok=True)
        save_start = time.perf_counter()
        ov.save_model(ov_model, str(output), compress_to_fp16=True)
        payload["timings_seconds"]["openvino_ir_save_fp16"] = time.perf_counter() - save_start
        bin_path = output.with_suffix(".bin")
        payload["openvino"]["ir"] = {
            "status": "saved",
            "xml": str(output),
            "bin": str(bin_path),
            "compress_to_fp16": True,
            "xml_bytes": output.stat().st_size,
            "bin_bytes": bin_path.stat().st_size if bin_path.is_file() else None,
            "bin_below_400mb": bin_path.is_file() and bin_path.stat().st_size < 400_000_000,
        }

        compile_start = time.perf_counter()
        core = ov.Core()
        properties = {"NUM_STREAMS": 1, "INFERENCE_NUM_THREADS": 4}
        compiled = core.compile_model(ov_model, "CPU", properties)
        payload["timings_seconds"]["openvino_cpu_compile"] = time.perf_counter() - compile_start
        infer_start = time.perf_counter()
        ov_a = _run_openvino(compiled, inputs_a, np)
        ov_b = _run_openvino(compiled, inputs_b, np)
        payload["timings_seconds"]["openvino_cpu_two_case_inference"] = time.perf_counter() - infer_start
        payload["openvino"]["compile"] = {
            "status": "ok",
            "device": "CPU",
            "properties": {key: str(value) for key, value in properties.items()},
            "metrics": _assert_pair("OpenVINO", eager_a, eager_b, ov_a, ov_b, np),
        }
        readback_start = time.perf_counter()
        saved_model = core.read_model(str(output))
        saved_compiled = core.compile_model(saved_model, "CPU", properties)
        saved_a = _run_openvino(saved_compiled, inputs_a, np)
        saved_b = _run_openvino(saved_compiled, inputs_b, np)
        payload["timings_seconds"]["saved_ir_readback_compile_and_two_case_inference"] = time.perf_counter() - readback_start
        try:
            saved_metrics = _assert_pair("Saved OpenVINO IR", eager_a, eager_b, saved_a, saved_b, np)
        except Exception as exc:
            # _assert_pair performs finite/shape/distinct checks before the
            # fixed numerical thresholds.  Preserve the exact raw failure
            # and the measured values in the receipt without relaxing them.
            payload["openvino"]["ir"]["readback_validation"] = {
                "status": "failed",
                "finite_shape_and_distinct_checks_reached_threshold": True,
                "error": _json_error(exc),
            }
            raise
        else:
            payload["openvino"]["ir"]["readback_validation"] = {
                "status": "ok",
                "device": "CPU",
                "metrics": saved_metrics,
            }
        payload["memory"]["after_openvino_cpu_validation"] = _peak_rss()
        payload["status"] = "pass"
        return 0, payload
    except Exception as exc:
        payload["status"] = "failed"
        payload["errors"].append(_json_error(exc))
        return 1, payload
    finally:
        payload["elapsed_seconds"] = time.perf_counter() - start
        payload["memory"]["peak_rss"] = _peak_rss()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if payload["errors"]:
            raw_error_path = receipt_path.with_name("failure-raw.txt")
            raw_error_path.write_text(
                "\n\n".join(item.get("traceback", item.get("message", "")) for item in payload["errors"])
                + "\n",
                encoding="utf-8",
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--openvino-root", default=str(DEFAULT_OV_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--receipt", default=str(Path(__file__).resolve().parent / "receipt.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    code, payload = run(parse_args(argv))
    print(json.dumps({"status": payload.get("status"), "receipt": parse_args(argv).receipt}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
