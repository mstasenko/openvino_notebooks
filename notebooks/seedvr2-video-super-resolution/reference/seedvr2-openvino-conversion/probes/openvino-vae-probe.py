#!/usr/bin/env python3
"""CPU-only structural OpenVINO probe for the SeedVR2 VAE.

Without ``--weights`` this constructs a small, randomly initialized VAE.  With
``--weights`` it restores the original YAML channel widths and strictly loads a verified
Safetensors VAE checkpoint.  It answers only whether the encoder and decoder graph shapes
can be converted and executed through OpenVINO CPU; it is not a quality or performance
test.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


INPUT_SHAPE = (1, 3, 5, 64, 64)
LATENT_SHAPE = (1, 16, 2, 8, 8)
OUTPUT_SHAPE = INPUT_SHAPE
REDUCED_BLOCK_CHANNELS = (32, 64, 64, 64)


class ConversionError(RuntimeError):
    """Conversion failed after retaining every attempted conversion traceback."""

    def __init__(self, attempts: List[Dict[str, Any]]) -> None:
        super().__init__("OpenVINO conversion failed in eager and JIT modes")
        self.attempts = attempts


def exception_record(exc: BaseException) -> Dict[str, str]:
    """Return a serializable exception with its complete traceback."""

    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def force_cpu_only(torch_module: Any) -> List[str]:
    """Prevent upstream imports from probing accelerator availability.

    The functions are replaced before importing the SeedVR2 modules.  We intentionally
    do not call any backend query and do not inspect OpenVINO's available devices.
    """

    def always_false(*_args: Any, **_kwargs: Any) -> bool:
        return False

    def always_zero(*_args: Any, **_kwargs: Any) -> int:
        return 0

    patched: List[str] = []

    for backend_name in ("cuda", "xpu"):
        backend = getattr(torch_module, backend_name, None)
        if backend is None:
            continue
        if hasattr(backend, "is_available"):
            setattr(backend, "is_available", always_false)
            patched.append(f"torch.{backend_name}.is_available")
        if hasattr(backend, "device_count"):
            setattr(backend, "device_count", always_zero)
            patched.append(f"torch.{backend_name}.device_count")

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None:
        if hasattr(mps, "is_available"):
            setattr(mps, "is_available", always_false)
            patched.append("torch.backends.mps.is_available")
        if hasattr(mps, "device_count"):
            setattr(mps, "device_count", always_zero)
            patched.append("torch.backends.mps.device_count")

    return patched


def tensor_value(value: Any, torch_module: Any) -> Any:
    """Normalize a framework tensor/array to a detached CPU torch tensor."""

    if not torch_module.is_tensor(value):
        value = torch_module.as_tensor(value)
    return value.detach().to(device="cpu", dtype=torch_module.float32)


def tensor_summary(value: Any, expected_shape: Sequence[int], torch_module: Any) -> Dict[str, Any]:
    tensor = tensor_value(value, torch_module)
    finite = bool(torch_module.isfinite(tensor).all().item())
    return {
        "shape": list(tensor.shape),
        "expected_shape": list(expected_shape),
        "shape_matches": list(tensor.shape) == list(expected_shape),
        "finite": finite,
        "min": float(tensor.min().item()) if tensor.numel() else None,
        "max": float(tensor.max().item()) if tensor.numel() else None,
    }


def comparison_summary(reference: Any, candidate: Any, torch_module: Any) -> Dict[str, Any]:
    reference_tensor = tensor_value(reference, torch_module)
    candidate_tensor = tensor_value(candidate, torch_module)
    if reference_tensor.shape != candidate_tensor.shape:
        return {
            "shape_compatible": False,
            "reference_shape": list(reference_tensor.shape),
            "candidate_shape": list(candidate_tensor.shape),
            "max_abs": None,
            "rms_relative_error": None,
        }

    difference = (candidate_tensor - reference_tensor).abs()
    reference_rms = torch_module.sqrt(torch_module.mean(reference_tensor.square()))
    difference_rms = torch_module.sqrt(torch_module.mean(difference.square()))
    relative = difference_rms / reference_rms.clamp_min(torch_module.finfo(torch_module.float32).eps)
    return {
        "shape_compatible": True,
        "reference_shape": list(reference_tensor.shape),
        "candidate_shape": list(candidate_tensor.shape),
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "rms_relative_error": float(relative.item()),
    }


def output_variation(first: Any, second: Any, torch_module: Any) -> Dict[str, Any]:
    first_tensor = tensor_value(first, torch_module)
    second_tensor = tensor_value(second, torch_module)
    if first_tensor.shape != second_tensor.shape:
        return {
            "shape_compatible": False,
            "max_abs": None,
            "different": False,
        }
    difference = (second_tensor - first_tensor).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    return {
        "shape_compatible": True,
        "max_abs": max_abs,
        "different": math.isfinite(max_abs) and max_abs > 1e-7,
    }


def output_is_valid(summary: Dict[str, Any]) -> bool:
    return bool(summary.get("shape_matches")) and bool(summary.get("finite"))


def comparison_is_within_tolerance(summary: Dict[str, Any]) -> bool:
    max_abs = summary.get("max_abs")
    rms_relative = summary.get("rms_relative_error")
    return (
        summary.get("shape_compatible") is True
        and isinstance(max_abs, (int, float))
        and isinstance(rms_relative, (int, float))
        and math.isfinite(float(max_abs))
        and math.isfinite(float(rms_relative))
        and float(max_abs) <= 1e-3
        and float(rms_relative) <= 1e-4
    )


def load_reduced_vae(
    source_root: Path, weights_path: Path | None
) -> Tuple[Any, Dict[str, Any]]:
    """Build the toy or full-width VAE using the upstream class and config structure."""

    import yaml

    config_path = source_root / "src" / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    original_block_channels = tuple(config["block_out_channels"])
    config["block_out_channels"] = list(
        original_block_channels if weights_path is not None else REDUCED_BLOCK_CHANNELS
    )
    config["norm_num_groups"] = 32
    config["layers_per_block"] = 2
    config["latent_channels"] = 16
    config["temporal_scale_num"] = 2
    config["inflation_mode"] = "pad"
    config["use_quant_conv"] = False
    config["use_post_quant_conv"] = False

    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKL

    vae = VideoAutoencoderKL(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        down_block_types=tuple(config["down_block_types"]),
        up_block_types=tuple(config["up_block_types"]),
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=int(config["layers_per_block"]),
        act_fn=str(config["act_fn"]),
        latent_channels=int(config["latent_channels"]),
        norm_num_groups=int(config["norm_num_groups"]),
        temporal_scale_num=int(config["temporal_scale_num"]),
        inflation_mode=str(config["inflation_mode"]),
        slicing_sample_min_size=int(config["slicing_sample_min_size"]),
        use_quant_conv=bool(config["use_quant_conv"]),
        use_post_quant_conv=bool(config["use_post_quant_conv"]),
    )
    vae.eval()
    if weights_path is not None:
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
        vae.load_state_dict(state_dict, strict=True)

    structure = {
        "class": "src.models.video_vae_v3.modules.attn_video_vae.VideoAutoencoderKL",
        "config_path": str(config_path),
        "block_out_channels": list(config["block_out_channels"]),
        "norm_num_groups": 32,
        "layers_per_block": 2,
        "latent_channels": 16,
        "temporal_scale_num": 2,
        "inflation_mode": "pad",
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "weights": "real_safetensors" if weights_path is not None else "random_initialization",
        "weights_path": str(weights_path) if weights_path is not None else None,
        "parameter_count": sum(parameter.numel() for parameter in vae.parameters()),
    }
    return vae, structure


def make_wrappers(vae: Any, torch_module: Any) -> Tuple[Any, Any]:
    from src.models.video_vae_v3.modules.types import MemoryState

    class EncoderTensorWrapper(torch_module.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.vae = model

        def forward(self, value: Any) -> Any:
            encoded = self.vae.encoder(value, memory_state=MemoryState.DISABLED)
            return encoded[:, :16]

    class DecoderTensorWrapper(torch_module.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.vae = model

        def forward(self, value: Any) -> Any:
            return self.vae.decoder(value, memory_state=MemoryState.DISABLED)

    encoder = EncoderTensorWrapper(vae).eval()
    decoder = DecoderTensorWrapper(vae).eval()
    return encoder, decoder


def convert_openvino(
    ov_module: Any, wrapper: Any, example_input: Any, torch_module: Any
) -> Tuple[Any, str, List[Dict[str, Any]]]:
    """Try eager conversion, then a fixed-input TorchScript conversion."""

    attempts: List[Dict[str, Any]] = []
    try:
        with torch_module.inference_mode():
            converted = ov_module.convert_model(wrapper, example_input=example_input)
        return converted, "eager", attempts
    except Exception as eager_error:
        attempts.append({"mode": "eager", "exception": exception_record(eager_error)})

    try:
        with torch_module.inference_mode():
            traced = torch_module.jit.trace(
                wrapper, (example_input,), strict=False, check_trace=False
            )
            converted = ov_module.convert_model(traced, example_input=example_input)
        return converted, "jit", attempts
    except Exception as jit_error:
        attempts.append({"mode": "jit", "exception": exception_record(jit_error)})
        raise ConversionError(attempts) from jit_error


def port_shape(port: Any) -> List[Any]:
    dimensions: List[Any] = []
    for dimension in port.get_partial_shape():
        dimensions.append(None if dimension.is_dynamic else int(dimension.get_length()))
    return dimensions


def port_info(port: Any) -> Dict[str, Any]:
    return {
        "names": sorted(str(name) for name in port.get_names()),
        "shape": port_shape(port),
    }


def run_openvino(
    ov_module: Any, ov_model: Any, inputs: Iterable[Any], torch_module: Any
) -> Tuple[List[Any], Dict[str, Any]]:
    ov_model.reshape({ov_model.input(0): list(inputs[0].shape)})
    core = ov_module.Core()
    compiled = core.compile_model(
        ov_model,
        "CPU",
        {
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": "2",
            "INFERENCE_PRECISION_HINT": ov_module.Type.f32,
        },
    )
    request = compiled.create_infer_request()
    input_port = compiled.input(0)
    model_io = {
        "inputs": [port_info(port) for port in compiled.inputs],
        "outputs": [port_info(port) for port in compiled.outputs],
    }
    outputs: List[Any] = []
    for value in inputs:
        result = request.infer({input_port: value.detach().cpu().numpy()})
        outputs.append(torch_module.from_numpy(next(iter(result.values()))).clone())
    return outputs, model_io


def probe_wrapper(
    wrapper: Any,
    inputs: Sequence[Any],
    expected_shape: Sequence[int],
    torch_module: Any,
    ov_module: Any,
) -> Tuple[Dict[str, Any], Any | None]:
    stage: Dict[str, Any] = {
        "input_shape": list(inputs[0].shape),
        "expected_output_shape": list(expected_shape),
        "input_count": len(inputs),
        "distinct_inputs": not bool(torch_module.equal(inputs[0], inputs[1])),
        "status": "failed",
    }

    try:
        with torch_module.inference_mode():
            eager_outputs = [wrapper(value) for value in inputs]
        stage["eager"] = {
            "status": "ok",
            "outputs": [
                tensor_summary(value, expected_shape, torch_module) for value in eager_outputs
            ],
        }
    except Exception as eager_error:
        stage["eager"] = {"status": "error", "exception": exception_record(eager_error)}
        return stage, None

    try:
        ov_model, conversion_mode, attempts = convert_openvino(
            ov_module, wrapper, inputs[0], torch_module
        )
        stage["conversion"] = {"status": "ok", "mode": conversion_mode, "attempts": attempts}
    except Exception as conversion_error:
        stage["conversion"] = {
            "status": "error",
            "attempts": getattr(conversion_error, "attempts", []),
            "exception": exception_record(conversion_error),
        }
        return stage, None

    try:
        ov_outputs, model_io = run_openvino(ov_module, ov_model, inputs, torch_module)
        stage["openvino"] = {
            "status": "ok",
            **model_io,
            "outputs": [
                tensor_summary(value, expected_shape, torch_module) for value in ov_outputs
            ],
        }
        stage["comparisons"] = [
            comparison_summary(reference, candidate, torch_module)
            for reference, candidate in zip(eager_outputs, ov_outputs)
        ]
        stage["output_variation"] = {
            "eager": output_variation(eager_outputs[0], eager_outputs[1], torch_module),
            "openvino": output_variation(ov_outputs[0], ov_outputs[1], torch_module),
        }
        eager_valid = all(
            output_is_valid(summary) for summary in stage["eager"]["outputs"]
        )
        openvino_valid = all(
            output_is_valid(summary) for summary in stage["openvino"]["outputs"]
        )
        comparisons_valid = all(
            comparison_is_within_tolerance(summary) for summary in stage["comparisons"]
        )
        eager_distinct = stage["output_variation"]["eager"]["different"]
        openvino_distinct = stage["output_variation"]["openvino"]["different"]
        stage["criteria"] = {
            "eager_outputs_shape_and_finite": eager_valid,
            "openvino_outputs_shape_and_finite": openvino_valid,
            "comparisons_max_abs_le_1e-3_and_rms_relative_le_1e-4": comparisons_valid,
            "eager_outputs_differ_by_more_than_1e-7": eager_distinct,
            "openvino_outputs_differ_by_more_than_1e-7": openvino_distinct,
        }
        stage["status"] = all(stage["criteria"].values()) and stage["distinct_inputs"]
        stage["status"] = "passed" if stage["status"] else "failed"
    except Exception as openvino_error:
        stage["openvino"] = {"status": "error", "exception": exception_record(openvino_error)}
    return stage, ov_model if stage["status"] == "passed" else None


def warning_record(item: warnings.WarningMessage) -> Dict[str, Any]:
    return {
        "category": item.category.__name__,
        "message": str(item.message),
        "filename": item.filename,
        "lineno": item.lineno,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="SeedVR2 source root containing src/models/video_vae_v3/.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Path for the structured probe result JSON.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Optional verified VAE safetensors checkpoint; enables the original full-width model.",
    )
    parser.add_argument(
        "--ir-dir",
        type=Path,
        help="Optional directory for validated FP32 encoder/decoder OpenVINO IR files.",
    )
    return parser.parse_args(argv)


def save_validated_ir(
    ov_module: Any, ir_dir: Path, models: Dict[str, Any]
) -> Dict[str, Any]:
    ir_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Any] = {}
    for name, model in models.items():
        xml_path = ir_dir / f"vae-{name}.xml"
        ov_module.save_model(model, str(xml_path), compress_to_fp16=False)
        saved[name] = {
            "xml": str(xml_path),
            "bin": str(xml_path.with_suffix(".bin")),
            "compress_to_fp16": False,
        }
    return saved


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.source.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve() if args.weights is not None else None
    ir_dir = args.ir_dir.expanduser().resolve() if args.ir_dir is not None else None
    config_path = source_root / "src" / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml"
    result: Dict[str, Any] = {
        "probe": "seedvr2-vae-openvino-structural",
        "status": "not_run",
        "source_root": str(source_root),
        "config_path": str(config_path),
        "weights": {
            "mode": "real_safetensors" if weights_path is not None else "toy_random_initialization",
            "path": str(weights_path) if weights_path is not None else None,
        },
        "ir_dir": str(ir_dir) if ir_dir is not None else None,
        "cpu_only": True,
        "accelerator_queries": "not called",
        "expected": {
            "input": list(INPUT_SHAPE),
            "latent": list(LATENT_SHAPE),
            "output": list(OUTPUT_SHAPE),
        },
        "thread_settings": {"torch_threads": 2, "torch_interop_threads": 1},
        "warnings": [],
        "stages": {},
    }

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            import torch

            result["patched_backends"] = force_cpu_only(torch)
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.manual_seed(20260913)

            if str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))

            import openvino as ov

            if weights_path is not None:
                if not weights_path.is_file():
                    raise FileNotFoundError(f"VAE weights file does not exist: {weights_path}")
                if weights_path.suffix != ".safetensors":
                    raise ValueError(f"VAE weights must be a .safetensors file: {weights_path}")
            vae, structure = load_reduced_vae(source_root, weights_path)
            encoder, decoder = make_wrappers(vae, torch)
            result["structure"] = structure
            result["parameter_count"] = structure["parameter_count"]

            generator = torch.Generator(device="cpu").manual_seed(20260914)
            encoder_inputs = [
                torch.randn(INPUT_SHAPE, generator=generator, dtype=torch.float32),
                torch.randn(INPUT_SHAPE, generator=generator, dtype=torch.float32),
            ]
            decoder_inputs = [
                torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32),
                torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32),
            ]
            encoder_stage, encoder_model = probe_wrapper(
                encoder, encoder_inputs, LATENT_SHAPE, torch, ov
            )
            decoder_stage, decoder_model = probe_wrapper(
                decoder, decoder_inputs, OUTPUT_SHAPE, torch, ov
            )
            result["stages"]["encoder"] = encoder_stage
            result["stages"]["decoder"] = decoder_stage
            if all(stage.get("status") == "passed" for stage in result["stages"].values()):
                result["status"] = "passed"
            else:
                result["status"] = "failed"
            if result["status"] == "passed" and ir_dir is not None:
                try:
                    result["ir"] = {
                        "status": "ok",
                        "files": save_validated_ir(
                            ov,
                            ir_dir,
                            {"encoder": encoder_model, "decoder": decoder_model},
                        ),
                    }
                except Exception as ir_error:
                    result["ir"] = {"status": "error", "exception": exception_record(ir_error)}
                    result["status"] = "failed"
        except BaseException as error:
            result["status"] = "error"
            result["exception"] = exception_record(error)
        finally:
            result["warnings"] = [warning_record(item) for item in caught_warnings]
            result["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
