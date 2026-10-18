#!/usr/bin/env python3
"""Bounded CPU-only OpenVINO probe for the SeedVR2 DiT graph.

This is deliberately a structural probe, not a quality test.  Without
``--weights`` it builds a reduced-width instance of the real 3B NaDiT class;
with a supplied ``--weights`` safetensors checkpoint it keeps the original
32-layer 3B dimensions and loads that checkpoint strictly into an FP32 CPU
model.  Both paths use fixed tiny metadata, then trace and convert the graph
for OpenVINO CPU.

Before project imports, the process installs synthetic false/zero CUDA, XPU,
and MPS availability/count guards.  It does not query those backends or
enumerate OpenVINO devices, and only asks OpenVINO to compile the CPU target.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import traceback
import warnings
from typing import Any, Dict, List, Sequence, Tuple


MAX_ISSUE_TEXT = 1200
MAX_EXCEPTION_TEXT = 16000
MAX_TRACEBACK_TEXT = 16000
MAX_ISSUES = 64


def _short(text: Any, limit: int = MAX_ISSUE_TEXT) -> str:
    value = str(text).strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _issue(exc: BaseException, *, include_traceback: bool = True) -> Dict[str, str]:
    item = {
        "type": type(exc).__name__,
        "message": _short(exc, MAX_EXCEPTION_TEXT),
    }
    if include_traceback:
        item["traceback"] = _short(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            MAX_TRACEBACK_TEXT,
        )
    return item


def _capture_warnings(record: List[Dict[str, str]]):
    """Record warnings without allowing a noisy tracer to grow the artifact."""

    class _Recorder:
        def __enter__(self):
            self._context = warnings.catch_warnings(record=True)
            self._items = self._context.__enter__()
            warnings.simplefilter("always")
            return self

        def __exit__(self, exc_type, exc_value, exc_tb):
            for warning in self._items:
                if len(record) >= MAX_ISSUES:
                    break
                record.append(
                    {
                        "category": warning.category.__name__,
                        "message": _short(warning.message),
                        "filename": str(warning.filename),
                        "lineno": str(warning.lineno),
                    }
                )
            return self._context.__exit__(exc_type, exc_value, exc_tb)

    return _Recorder()


def _git_revision(source: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _cpu_guard(torch: Any) -> Dict[str, Any]:
    """Install synthetic CPU-only backend guards before project imports.

    This probe must not discover or query host devices.  SeedVR2's import path
    asks these APIs, so replace them before importing any project module.
    """

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


def _peak_rss() -> Dict[str, int]:
    """Return Linux ru_maxrss in explicit units for the evidence artifact."""

    # Linux reports ru_maxrss in KiB.  This trial is Linux-only; keep the
    # conversion explicit so the artifact is not mistaken for bytes.
    kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "peak_rss_kib": kib,
        "peak_rss_bytes": kib * 1024,
    }


def _load_nadit(source: Path, torch: Any, weights_path: Path | None) -> Any:
    """Load the exact 3B config in reduced mode or full mode with safetensors."""

    from omegaconf import OmegaConf

    # The checked-in config uses the ``eval`` resolver for repeated block and
    # window entries.  Register it locally instead of importing the full CLI.
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except Exception as exc:
        if "already registered" not in str(exc):
            raise

    config_path = source / "configs_3b" / "main.yaml"
    config = OmegaConf.load(str(config_path))
    dit_cfg = OmegaConf.to_container(config.dit.model, resolve=True)
    if not isinstance(dit_cfg, dict):
        raise TypeError("configs_3b/main.yaml dit.model did not resolve to a mapping")

    dit_cfg.pop("__object__", None)
    if weights_path is None:
        dit_cfg.update(
            {
                "vid_dim": 128,
                "txt_dim": 128,
                "emb_dim": 768,
                "heads": 1,
                "head_dim": 128,
                "rope_dim": 128,
                "num_layers": 2,
                "mm_layers": 1,
                "block_type": ["mmdit_sr", "mmdit_sr"],
                "window": [(4, 3, 3), (4, 3, 3)],
                "window_method": [
                    "720pwin_by_size_bysize",
                    "720pswin_by_size_bysize",
                ],
                "attention_mode": "sdpa",
            }
        )
    else:
        # Full mode intentionally leaves every model dimension/block count
        # from configs_3b/main.yaml intact; only the portable attention path
        # is selected for this CPU reference/export probe.
        dit_cfg["attention_mode"] = "sdpa"

    # Import after the CPU guard.  This is the exact NaDiT implementation from
    # the supplied source tree, not a rewritten/export-specific model.
    from src.models.dit_3b.nadit import NaDiT

    torch.manual_seed(20260913)
    model = NaDiT(**dit_cfg).to(device="cpu", dtype=torch.float32).eval()
    if weights_path is not None:
        if not weights_path.is_file():
            raise FileNotFoundError(f"weights file does not exist: {weights_path}")
        from safetensors.torch import load_file

        # load_file materializes one CPU state mapping.  Release it immediately
        # after strict loading so the full FP32 model is the only retained
        # weight copy before tracing/conversion.
        state = load_file(str(weights_path), device="cpu")
        try:
            model.load_state_dict(state, strict=True)
        finally:
            del state
            gc.collect()
    return model


class FixedNaDiT:
    """Tensor-only wrapper with fixed input metadata for tracing/conversion."""

    def __init__(self, model: Any):
        import torch
        import torch.nn as nn

        class _Wrapper(nn.Module):
            def __init__(self, inner: Any):
                super().__init__()
                self.inner = inner
                self.register_buffer(
                    "vid_shape_const",
                    torch.tensor([[2, 8, 8]], dtype=torch.long),
                )
                self.register_buffer(
                    "txt_shape_const",
                    torch.tensor([[32]], dtype=torch.long),
                )

            def forward(
                self,
                vid: Any,
                txt: Any,
                timestep: Any,
            ) -> Any:
                output = self.inner(
                    vid,
                    txt,
                    self.vid_shape_const,
                    self.txt_shape_const,
                    timestep,
                    disable_cache=False,
                )
                return output.vid_sample

        self.module = _Wrapper(model).eval()


def _make_inputs(torch: Any, seed: int) -> Tuple[Any, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn((128, 33), generator=generator, dtype=torch.float32),
        torch.randn((32, 5120), generator=generator, dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
    )


def _validate_input_contract(inputs: Sequence[Any]) -> None:
    expected = ((128, 33), (32, 5120), (1,))
    actual = tuple(tuple(item.shape) for item in inputs)
    if actual != expected:
        raise RuntimeError(f"probe input contract mismatch: actual={actual}, expected={expected}")


def _specialize_fixed_window_ops(model: Any) -> int:
    """Make only this fixed-shape probe's window dimensions traceable.

    The upstream window math is unchanged.  This deliberately specializes the
    temporal/spatial dimensions to the probe's fixed metadata and must not be
    reused for another resolution or frame count.
    """

    wrapped = 0
    for module in model.modules():
        if module.__class__.__name__ != "NaSwinAttention":
            continue
        original_op = module.window_op

        def fixed_window_op(size, num_windows, original_op=original_op):
            return original_op(tuple(int(d) for d in size), num_windows)

        module.window_op = fixed_window_op
        wrapped += 1
    return wrapped


def _numpy_inputs(inputs: Sequence[Any]) -> List[Any]:
    return [item.detach().cpu().numpy() for item in inputs]


def _stats(torch: Any, output: Any) -> Dict[str, Any]:
    finite = bool(torch.isfinite(output).all().item())
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "min": float(output.min().item()),
        "max": float(output.max().item()),
    }


def _error_metrics(reference: Any, candidate: Any, np: Any) -> Dict[str, float]:
    reference_array = reference.detach().cpu().numpy() if hasattr(reference, "detach") else np.asarray(reference)
    candidate_array = candidate.detach().cpu().numpy() if hasattr(candidate, "detach") else np.asarray(candidate)
    diff = candidate_array.astype(np.float64) - reference_array.astype(np.float64)
    rms = float(np.sqrt(np.mean(np.square(diff))))
    reference_rms = float(np.sqrt(np.mean(np.square(reference_array))))
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "rms_relative_error": rms / max(reference_rms, 1.0e-12),
    }


def _validate_pair(
    label: str,
    output_a: Any,
    output_b: Any,
    reference_a: Any,
    reference_b: Any,
    np: Any,
) -> Dict[str, Any]:
    expected_shape = (128, 16)
    shapes = (tuple(output_a.shape), tuple(output_b.shape))
    if shapes != (expected_shape, expected_shape):
        raise RuntimeError(f"{label} output shape mismatch: A={shapes[0]}, B={shapes[1]}")
    arrays = (
        output_a.detach().cpu().numpy() if hasattr(output_a, "detach") else np.asarray(output_a),
        output_b.detach().cpu().numpy() if hasattr(output_b, "detach") else np.asarray(output_b),
    )
    if not np.isfinite(arrays[0]).all() or not np.isfinite(arrays[1]).all():
        raise RuntimeError(f"{label} output contains non-finite values")
    metrics = {
        "a": _error_metrics(reference_a, output_a, np),
        "b": _error_metrics(reference_b, output_b, np),
        "output_delta_max_abs": float(np.max(np.abs(arrays[0] - arrays[1]))),
    }
    for case in ("a", "b"):
        if metrics[case]["max_abs_error"] > 1.0e-3 or metrics[case]["rms_relative_error"] > 1.0e-4:
            raise RuntimeError(f"{label} numerical error exceeds threshold: {metrics}")
    if metrics["output_delta_max_abs"] <= 1.0e-7:
        raise RuntimeError(f"{label} outputs A and B are indistinguishable: {metrics}")
    return metrics


def _run_openvino(compiled: Any, inputs: Sequence[Any], np: Any) -> Any:
    arrays = _numpy_inputs(inputs)
    request = compiled.create_infer_request()
    feed = {port: array for port, array in zip(compiled.inputs, arrays)}
    result = request.infer(feed)
    if len(compiled.outputs) != 1:
        raise RuntimeError(f"expected one OpenVINO output, got {len(compiled.outputs)}")
    # Copy before this request/result is reused for the next distinct input.
    return np.array(result[compiled.outputs[0]], copy=True)


def _port_info(port: Any, index: int) -> Dict[str, Any]:
    """Record port metadata without using the dynamic-port ``.shape`` property."""

    partial_shape = port.get_partial_shape()
    if partial_shape.is_static:
        shape: Any = [int(dimension) for dimension in partial_shape.to_shape()]
    else:
        shape = None
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


def _error_payload(exc: BaseException) -> Dict[str, str]:
    return _issue(exc, include_traceback=True)


def run(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    source = Path(args.source).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve() if args.weights else None
    full_weights = weights_path is not None
    payload: Dict[str, Any] = {
        "schema": 1,
        "status": "not_started",
        "source": str(source),
        "source_revision": _git_revision(source),
        "cpu_guard": None,
        "contract": {
            "model": (
                "configs_3b/main.yaml -> NaDiT (full 3B checkpoint structural probe)"
                if full_weights
                else "configs_3b/main.yaml -> NaDiT (reduced structural probe)"
            ),
            "video_frames": 5,
            "latent_shape": [2, 8, 8],
            "inputs": {
                "vid": [128, 33],
                "txt": [32, 5120],
                "timestep": [1],
            },
            "captured_metadata": {
                "vid_shape": {"shape": [1, 3], "value": [[2, 8, 8]]},
                "txt_shape": {"shape": [1, 1], "value": [[32]]},
            },
            "fixed_window_dimensions": True,
            "expected_output_shape": [128, 16],
            "model_config": (
                {
                    "vid_dim": 2560,
                    "txt_dim": 2560,
                    "emb_dim": 15360,
                    "heads": 20,
                    "head_dim": 128,
                    "rope_dim": 128,
                    "num_layers": 32,
                    "mm_layers": 10,
                    "attention_mode": "sdpa",
                }
                if full_weights
                else {
                    "vid_dim": 128,
                    "txt_dim": 128,
                    "emb_dim": 768,
                    "heads": 1,
                    "head_dim": 128,
                    "rope_dim": 128,
                    "num_layers": 2,
                    "mm_layers": 1,
                    "attention_mode": "sdpa",
                }
            ),
        },
        "weights": {
            "mode": "full_safetensors_fp32" if full_weights else "random_reduced_fp32",
            "path": str(weights_path) if weights_path is not None else None,
            "load_state_dict_strict": True if full_weights else None,
            "expected_full_parameter_count": 3391476448 if full_weights else None,
        },
        "memory": {
            "rss_unit": "bytes; Linux ru_maxrss source is KiB",
            "peak_rss": {},
        },
        "warnings": [],
        "errors": [],
        "reference": {},
        "trace": {},
        "openvino": {},
    }

    try:
        import numpy as np
        import torch

        # Configure CPU execution before importing any project module.
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        payload["cpu_guard"] = _cpu_guard(torch)

        if not source.is_dir():
            raise FileNotFoundError(f"source directory does not exist: {source}")
        if not (source / "configs_3b" / "main.yaml").is_file():
            raise FileNotFoundError(f"missing config: {source / 'configs_3b' / 'main.yaml'}")

        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        if weights_path is not None and not weights_path.is_file():
            raise FileNotFoundError(f"weights file does not exist: {weights_path}")
        payload["memory"]["peak_rss"]["before_model_load"] = _peak_rss()
        model = _load_nadit(source, torch, weights_path)
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
        payload["weights"].update(
            {
                "parameter_count": parameter_count,
                "parameter_dtype": str(next(model.parameters()).dtype),
                "parameter_count_matches_expected": (
                    parameter_count == payload["weights"]["expected_full_parameter_count"]
                    if full_weights
                    else None
                ),
            }
        )
        payload["memory"]["peak_rss"]["after_model_load"] = _peak_rss()
        wrapper = FixedNaDiT(model).module
        inputs_a = _make_inputs(torch, 1001)
        inputs_b = _make_inputs(torch, 1002)

        _validate_input_contract(inputs_a)
        _validate_input_contract(inputs_b)
        with torch.inference_mode():
            reference_a = wrapper(*inputs_a)
            reference_b = wrapper(*inputs_b)
        if tuple(reference_a.shape) != (128, 16) or tuple(reference_b.shape) != (128, 16):
            raise RuntimeError(
                f"reference output shape mismatch: A={tuple(reference_a.shape)}, "
                f"B={tuple(reference_b.shape)}"
            )
        if not bool(torch.isfinite(reference_a).all().item()):
            raise RuntimeError("reference output A contains non-finite values")
        if not bool(torch.isfinite(reference_b).all().item()):
            raise RuntimeError("reference output B contains non-finite values")
        reference_delta = float((reference_a - reference_b).abs().max().item())
        if reference_delta <= 1.0e-7:
            raise RuntimeError(
                f"distinct random inputs produced nearly identical reference output (delta={reference_delta})"
            )
        payload["reference"] = {
            "input_delta_max_abs": float((inputs_a[0] - inputs_b[0]).abs().max().item()),
            "output_delta_max_abs": reference_delta,
            "a": _stats(torch, reference_a),
            "b": _stats(torch, reference_b),
        }

        # The eager reference above intentionally used the unmodified upstream
        # window operation. This adapter is only for tracing the fixed probe
        # shape; it must not be generalized to another resolution/frame count.
        fixed_window_wrappers = _specialize_fixed_window_ops(model)

        # Trace with fixed tensors first.  This deliberately avoids torch.export/
        # torch.compile, because the real RoPE implementation has an explicit
        # @torch._dynamo.disable boundary.
        trace_warnings: List[Dict[str, str]] = []
        try:
            with torch.inference_mode():
                with _capture_warnings(trace_warnings):
                    traced = torch.jit.trace(
                        wrapper,
                        inputs_a,
                        strict=False,
                        check_trace=False,
                    ).eval()
                frozen = torch.jit.freeze(traced)
                traced_a = frozen(*inputs_a)
                traced_b = frozen(*inputs_b)
            trace_metrics = _validate_pair(
                "TorchScript", traced_a, traced_b, reference_a, reference_b, np
            )
            payload["trace"] = {
                "status": "ok",
                "warnings": trace_warnings,
                "frozen": True,
                "a_shape": list(traced_a.shape),
                "b_shape": list(traced_b.shape),
                "fixed_window_wrappers": fixed_window_wrappers,
                "metrics": trace_metrics,
            }
        except Exception as exc:
            payload["trace"] = {
                "status": "failed",
                "warnings": trace_warnings,
                "fixed_window_wrappers": fixed_window_wrappers,
                "error": _error_payload(exc),
            }
            raise

        try:
            import openvino as ov
        except Exception as exc:
            payload["openvino"] = {
                "status": "dependency_missing",
                "error": _error_payload(exc),
            }
            payload["status"] = "openvino_dependency_missing"
            return 2, payload

        convert_warnings: List[Dict[str, str]] = []
        with torch.inference_mode():
            with _capture_warnings(convert_warnings):
                # Use a fixed TorchScript graph and fixed example tensors. No
                # OpenVINO device enumeration or non-CPU compilation occurs here.
                ov_model = ov.convert_model(frozen, example_input=inputs_a)
        if len(ov_model.inputs) != 3:
            raise RuntimeError(
                f"metadata was not captured as buffers; OpenVINO graph has {len(ov_model.inputs)} inputs"
            )
        payload["openvino"]["conversion"] = {
            "status": "ok",
            "warnings": convert_warnings,
            "inputs": len(ov_model.inputs),
            "outputs": len(ov_model.outputs),
        }

        # The fixed-window adapter already specializes this graph.  Make that
        # specialization explicit before CPU compilation and IR serialization;
        # use tensors' known shapes rather than the dynamic-port `.shape`
        # property, which is not available on all OpenVINO versions.
        ov_model.reshape(
            {
                port: list(tensor.shape)
                for port, tensor in zip(ov_model.inputs, inputs_a)
            }
        )
        payload["openvino"]["graph"] = {
            "inputs": [_port_info(port, index) for index, port in enumerate(ov_model.inputs)],
            "outputs": [_port_info(port, index) for index, port in enumerate(ov_model.outputs)],
        }
        payload["memory"]["peak_rss"]["after_conversion_and_reshape"] = _peak_rss()

        # The frozen graph retains the parameters needed by conversion and
        # later serialization. Release the original eager model, wrapper, and
        # unfrozen trace/results before compiling the full graph.
        del model, wrapper, traced, traced_a, traced_b
        gc.collect()
        payload["memory"]["peak_rss"]["after_original_model_release"] = _peak_rss()

        compile_properties = {
            "NUM_STREAMS": 1,
            "INFERENCE_NUM_THREADS": 2,
            "INFERENCE_PRECISION_HINT": getattr(ov.Type, "f32", "f32"),
        }
        core = ov.Core()
        compiled = core.compile_model(ov_model, "CPU", compile_properties)
        ov_a = _run_openvino(compiled, inputs_a, np)
        ov_b = _run_openvino(compiled, inputs_b, np)
        ov_metrics = _validate_pair(
            "OpenVINO", ov_a, ov_b, reference_a, reference_b, np
        )
        payload["openvino"]["compile"] = {
            "status": "ok",
            "device": "CPU",
            "properties": {key: str(value) for key, value in compile_properties.items()},
            "metrics": ov_metrics,
        }
        payload["memory"]["peak_rss"]["after_cpu_compile_and_compare"] = _peak_rss()

        # The compiled request/core and eager reference tensors are no longer
        # needed. Keep ``frozen`` alive through optional IR serialization.
        del compiled, core, ov_a, ov_b, reference_a, reference_b, inputs_a, inputs_b
        gc.collect()

        if args.ir:
            ir_path = Path(args.ir).expanduser().resolve()
            ir_path.parent.mkdir(parents=True, exist_ok=True)
            ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
            ir_files = [str(ir_path)]
            bin_path = ir_path.with_suffix(".bin")
            if bin_path.is_file():
                ir_files.append(str(bin_path))
            payload["openvino"]["ir"] = {
                "status": "saved",
                "path": str(ir_path),
                "compress_to_fp16": False,
                "files": ir_files,
            }
            payload["memory"]["peak_rss"]["after_ir_save"] = _peak_rss()
        del frozen
        gc.collect()
        payload["status"] = "passed_structural_probe"
        return 0, payload
    except Exception as exc:
        payload["errors"].append(_error_payload(exc))
        payload["status"] = "failed"
        return 1, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Exact SeedVR2 checkout containing configs_3b/main.yaml",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path for the bounded probe evidence JSON",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Optional verified .safetensors checkpoint. Selects the original "
            "full 3B config and loads it strictly on CPU in FP32."
        ),
    )
    parser.add_argument(
        "--ir",
        default=None,
        help=(
            "Optional XML output path. After successful CPU comparison, save "
            "the OpenVINO graph with compress_to_fp16=False."
        ),
    )
    args = parser.parse_args()

    status, payload = run(args)
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output_json": str(output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
