#!/usr/bin/env python3
"""Prepare/run a bounded CPU OpenVINO SeedVR2 crop qualifier.

The qualifier uses five existing 1280x720 RGB frames, crops the same 320x256
road-sign/person region from each frame, and applies the upstream CPU
``prepare_video_transforms(resolution=768)`` path.  It then runs three isolated
stages: full VAE encoder, full 3B DiT one-step SR, and full VAE decoder.  Only
one Torch model stage is resident at a time.  This is a representative-detail
crop check, not a full-4K quality or throughput qualification.

The script is intentionally not imported by the application.  Accelerator
guards are installed before any SeedVR2 source module is imported.  It never
starts ComfyUI, downloads weights, enumerates devices, or uses a GPU.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback
import warnings
from typing import Any, Dict, List, Sequence, Tuple


FRAME_NAMES = tuple(f"frame-{index:02d}.png" for index in range(1, 6))
CROP_X = 700
CROP_Y = 192
CROP_WIDTH = 320
CROP_HEIGHT = 256
TARGET_WIDTH = 960
TARGET_HEIGHT = 768
VAE_SCALE = 0.9152
INPUT_SHAPE = (1, 3, 5, TARGET_HEIGHT, TARGET_WIDTH)
RAW_LATENT_SHAPE = (1, 16, 2, 96, 120)
CHANNELS_LAST_LATENT_SHAPE = (2, 96, 120, 16)
DIT_VIDEO_SHAPE = (23040, 33)
DIT_TEXT_SHAPE = (58, 5120)
DIT_TIMESTEP_SHAPE = (1,)
DIT_OUTPUT_SHAPE = (23040, 16)
FINAL_SHAPE = (5, TARGET_HEIGHT, TARGET_WIDTH, 3)
SEED = 20260913


class StageFailure(RuntimeError):
    """A stage failed its fail-closed shape/finite/numerical contract."""

    def __init__(self, stage: str, payload: Dict[str, Any]):
        self.stage = stage
        self.payload = payload
        super().__init__(f"{stage} failed its qualifier contract")


def exception_record(exc: BaseException) -> Dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def warning_record(item: warnings.WarningMessage) -> Dict[str, Any]:
    return {
        "category": item.category.__name__,
        "message": str(item.message),
        "filename": str(item.filename),
        "lineno": int(item.lineno),
    }


def progress(message: str) -> None:
    """Emit a bounded phase marker without exposing model/input contents."""

    print(f"[seedvr2-video-probe] {message}", file=sys.stderr, flush=True)


def load_helper(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peak_rss() -> Dict[str, int]:
    # Linux ru_maxrss is KiB.  Keep both units explicit in evidence.
    kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"peak_rss_kib": kib, "peak_rss_bytes": kib * 1024}


def tensor_cpu_float(value: Any, torch_module: Any) -> Any:
    return value.detach().to(device="cpu", dtype=torch_module.float32)


def finite_shape_summary(value: Any, expected_shape: Sequence[int], torch_module: Any) -> Dict[str, Any]:
    tensor = tensor_cpu_float(value, torch_module)
    finite = bool(torch_module.isfinite(tensor).all().item())
    return {
        "shape": list(tensor.shape),
        "expected_shape": list(expected_shape),
        "shape_matches": list(tensor.shape) == list(expected_shape),
        "finite": finite,
        "min": float(tensor.min().item()) if tensor.numel() else None,
        "max": float(tensor.max().item()) if tensor.numel() else None,
    }


def error_metrics(reference: Any, candidate: Any, torch_module: Any) -> Dict[str, Any]:
    reference_tensor = tensor_cpu_float(reference, torch_module)
    candidate_tensor = tensor_cpu_float(candidate, torch_module)
    if reference_tensor.shape != candidate_tensor.shape:
        return {
            "shape_compatible": False,
            "reference_shape": list(reference_tensor.shape),
            "candidate_shape": list(candidate_tensor.shape),
            "max_abs": None,
            "rms_relative_error": None,
        }
    difference = candidate_tensor - reference_tensor
    difference_abs = difference.abs()
    reference_rms = torch_module.sqrt(torch_module.mean(reference_tensor.square()))
    difference_rms = torch_module.sqrt(torch_module.mean(difference.square()))
    relative = difference_rms / reference_rms.clamp_min(torch_module.finfo(torch_module.float32).eps)
    return {
        "shape_compatible": True,
        "reference_shape": list(reference_tensor.shape),
        "candidate_shape": list(candidate_tensor.shape),
        "max_abs": float(difference_abs.max().item()) if difference_abs.numel() else 0.0,
        "rms_relative_error": float(relative.item()),
    }


def output_variation(first: Any, second: Any, torch_module: Any) -> Dict[str, Any]:
    first_tensor = tensor_cpu_float(first, torch_module)
    second_tensor = tensor_cpu_float(second, torch_module)
    if first_tensor.shape != second_tensor.shape:
        return {"shape_compatible": False, "max_abs": None, "different": False}
    delta = (second_tensor - first_tensor).abs()
    maximum = float(delta.max().item()) if delta.numel() else 0.0
    return {
        "shape_compatible": True,
        "max_abs": maximum,
        "different": math.isfinite(maximum) and maximum > 1.0e-7,
    }


def vector_summary(value: Any, torch_module: Any, count: int = 8) -> Dict[str, Any]:
    tensor = tensor_cpu_float(value, torch_module)
    shape = list(tensor.shape)
    flat = tensor.reshape(-1)
    first_values = flat[:count].tolist()
    last_values = flat[-count:].tolist()
    return {
        "shape": shape,
        "dtype": str(torch_module.float32),
        "first_values": [float(item) for item in first_values],
        "last_values": [float(item) for item in last_values],
    }


def contract_ok(stage: Dict[str, Any]) -> bool:
    criteria = stage.get("criteria", {})
    return bool(stage.get("distinct_inputs")) and all(bool(value) for value in criteria.values())


def stage_contract(
    eager_outputs: Sequence[Any],
    openvino_outputs: Sequence[Any],
    expected_shape: Sequence[int],
    torch_module: Any,
) -> Tuple[Dict[str, Any], bool]:
    eager_summaries = [
        finite_shape_summary(value, expected_shape, torch_module) for value in eager_outputs
    ]
    openvino_summaries = [
        finite_shape_summary(value, expected_shape, torch_module) for value in openvino_outputs
    ]
    comparisons = [
        error_metrics(reference, candidate, torch_module)
        for reference, candidate in zip(eager_outputs, openvino_outputs)
    ]
    eager_valid = all(item["shape_matches"] and item["finite"] for item in eager_summaries)
    openvino_valid = all(item["shape_matches"] and item["finite"] for item in openvino_summaries)
    numerical = all(
        item.get("shape_compatible") is True
        and isinstance(item.get("max_abs"), (int, float))
        and isinstance(item.get("rms_relative_error"), (int, float))
        and math.isfinite(float(item["max_abs"]))
        and math.isfinite(float(item["rms_relative_error"]))
        and float(item["max_abs"]) <= 1.0e-3
        and float(item["rms_relative_error"]) <= 1.0e-4
        for item in comparisons
    )
    eager_different = output_variation(eager_outputs[0], eager_outputs[1], torch_module)
    openvino_different = output_variation(openvino_outputs[0], openvino_outputs[1], torch_module)
    payload = {
        "eager": {"outputs": eager_summaries},
        "openvino": {"outputs": openvino_summaries},
        "comparisons": comparisons,
        "output_variation": {
            "eager": eager_different,
            "openvino": openvino_different,
        },
        "criteria": {
            "eager_outputs_shape_and_finite": eager_valid,
            "openvino_outputs_shape_and_finite": openvino_valid,
            "comparisons_max_abs_le_1e-3_and_rms_relative_le_1e-4": numerical,
            "eager_outputs_differ_by_more_than_1e-7": eager_different["different"],
            "openvino_outputs_differ_by_more_than_1e-7": openvino_different["different"],
        },
    }
    return payload, eager_valid and openvino_valid and numerical and eager_different["different"] and openvino_different["different"]


def save_ir(ov_module: Any, model: Any, path: Path) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ov_module.save_model(model, str(path), compress_to_fp16=False)
    return {
        "xml": str(path),
        "bin": str(path.with_suffix(".bin")),
        "compress_to_fp16": False,
    }


def run_vae_stage(
    name: str,
    wrapper: Any,
    inputs: Sequence[Any],
    expected_shape: Sequence[int],
    torch_module: Any,
    ov_module: Any,
    helper: Any,
    ir_path: Path,
    threads: int,
    warnings_out: List[Dict[str, Any]],
    additional_inputs: Sequence[Any] | None = None,
) -> Tuple[Dict[str, Any], List[Any], List[Any], List[Any]]:
    stage: Dict[str, Any] = {
        "name": name,
        "input_shapes": [list(item.shape) for item in inputs],
        "expected_output_shape": list(expected_shape),
        "distinct_inputs": not bool(torch_module.equal(inputs[0], inputs[1])),
        "status": "failed",
        "timings_seconds": {},
    }
    timings = stage["timings_seconds"]
    progress(f"{name}: eager reference start")
    started = time.perf_counter()
    try:
        with torch_module.inference_mode():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                eager_outputs = [wrapper(value) for value in inputs]
            warnings_out.extend(warning_record(item) for item in caught)
        stage["reference_vectors"] = [
            vector_summary(value, torch_module) for value in eager_outputs
        ]
        timings["reference_seconds"] = time.perf_counter() - started
    except BaseException as exc:
        stage["reference_error"] = exception_record(exc)
        raise StageFailure(name, stage) from exc

    progress(f"{name}: OpenVINO conversion start")
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ov_model, conversion_mode, attempts = helper.convert_openvino(
                ov_module, wrapper, inputs[0], torch_module
            )
        warnings_out.extend(warning_record(item) for item in caught)
        stage["conversion"] = {
            "status": "ok",
            "mode": conversion_mode,
            "attempts": attempts,
        }
        timings["conversion_seconds"] = time.perf_counter() - started
    except BaseException as exc:
        stage["conversion"] = {
            "status": "error",
            "exception": exception_record(exc),
        }
        raise StageFailure(name, stage) from exc

    progress(f"{name}: OpenVINO compile start")
    compile_started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ov_model.reshape({ov_model.input(0): list(inputs[0].shape)})
            core = ov_module.Core()
            compiled = core.compile_model(
                ov_model,
                "CPU",
                {
                    "NUM_STREAMS": "1",
                    "INFERENCE_NUM_THREADS": str(threads),
                    "INFERENCE_PRECISION_HINT": ov_module.Type.f32,
                },
            )
            timings["compile_seconds"] = time.perf_counter() - compile_started
            request = compiled.create_infer_request()
            model_io = {
                "inputs": [
                    {"index": index, **helper.port_info(port)}
                    for index, port in enumerate(compiled.inputs)
                ],
                "outputs": [
                    {"index": index, **helper.port_info(port)}
                    for index, port in enumerate(compiled.outputs)
                ],
            }
            progress(f"{name}: OpenVINO inference start")
            infer_started = time.perf_counter()
            openvino_outputs = []
            for value in inputs:
                output = request.infer({compiled.input(0): value.detach().cpu().numpy()})
                openvino_outputs.append(
                    torch_module.from_numpy(next(iter(output.values()))).clone()
                )
            timings["inference_seconds"] = time.perf_counter() - infer_started
        warnings_out.extend(warning_record(item) for item in caught)
        stage["openvino_io"] = model_io
        contract, passed = stage_contract(
            eager_outputs, openvino_outputs, expected_shape, torch_module
        )
        stage.update(contract)
        stage["status"] = "passed" if passed and stage["distinct_inputs"] else "failed"
        stage["openvino_vectors"] = [
            vector_summary(value, torch_module) for value in openvino_outputs
        ]
        if stage["status"] != "passed":
            raise StageFailure(name, stage)
        additional_outputs: List[Any] = []
        if additional_inputs:
            stage["additional_input_shapes"] = [list(item.shape) for item in additional_inputs]
            stage["additional_input_vectors"] = [
                vector_summary(value, torch_module) for value in additional_inputs
            ]
            additional_started = time.perf_counter()
            for value in additional_inputs:
                output = request.infer({compiled.input(0): value.detach().cpu().numpy()})
                additional_outputs.append(
                    torch_module.from_numpy(next(iter(output.values()))).clone()
                )
            timings["additional_inference_seconds"] = time.perf_counter() - additional_started
            additional_summaries = [
                finite_shape_summary(value, expected_shape, torch_module)
                for value in additional_outputs
            ]
            stage["additional_openvino"] = {"outputs": additional_summaries}
            if not all(item["shape_matches"] and item["finite"] for item in additional_summaries):
                raise StageFailure(name, stage)
        progress(f"{name}: IR serialization start")
        serialize_started = time.perf_counter()
        stage["ir"] = save_ir(ov_module, ov_model, ir_path)
        timings["serialization_seconds"] = time.perf_counter() - serialize_started
        return stage, eager_outputs, openvino_outputs, additional_outputs
    except StageFailure:
        raise
    except BaseException as exc:
        stage["openvino_error"] = exception_record(exc)
        raise StageFailure(name, stage) from exc


def fixed_dit_wrapper(model: Any, torch_module: Any) -> Any:
    class Wrapper(torch_module.nn.Module):
        def __init__(self, inner: Any):
            super().__init__()
            self.inner = inner
            self.register_buffer(
                "vid_shape_const",
                torch_module.tensor([[2, 96, 120]], dtype=torch_module.long),
            )
            self.register_buffer(
                "txt_shape_const",
                torch_module.tensor([[58]], dtype=torch_module.long),
            )

        def forward(self, vid: Any, txt: Any, timestep: Any) -> Any:
            result = self.inner(
                vid,
                txt,
                self.vid_shape_const,
                self.txt_shape_const,
                timestep,
                disable_cache=False,
            )
            return result.vid_sample

    return Wrapper(model).eval()


def prepare_inputs(
    input_dir: Path,
    output_dir: Path,
    source_root: Path,
    torch_module: Any,
    numpy_module: Any,
    warnings_out: List[Dict[str, Any]],
) -> Tuple[Any, Any]:
    from PIL import Image

    frames: List[Any] = []
    crops: List[Any] = []
    for name in FRAME_NAMES:
        path = input_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing required input frame: {path}")
        with Image.open(path) as image:
            if image.size != (1280, 720):
                raise ValueError(f"unexpected input dimensions for {path}: {image.size}")
            rgb = image.convert("RGB")
            crop = rgb.crop(
                (CROP_X, CROP_Y, CROP_X + CROP_WIDTH, CROP_Y + CROP_HEIGHT)
            )
            array = numpy_module.asarray(crop, dtype=numpy_module.uint8)
            crops.append(array.copy())
            frames.append(
                torch_module.from_numpy(array.copy()).permute(2, 0, 1).to(dtype=torch_module.float32) / 255.0
            )

    stacked = torch_module.stack(frames, dim=0)
    if tuple(stacked.shape) != (5, 3, CROP_HEIGHT, CROP_WIDTH):
        raise RuntimeError(f"crop tensor contract mismatch: {tuple(stacked.shape)}")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from src.core.generation_utils import prepare_video_transforms

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transform = prepare_video_transforms(768, 0, None)
        transformed = transform(stacked)
    warnings_out.extend(warning_record(item) for item in caught)
    if tuple(transformed.shape) != (3, 5, TARGET_HEIGHT, TARGET_WIDTH):
        raise RuntimeError(f"prepared video contract mismatch: {tuple(transformed.shape)}")
    transformed_batched = transformed.unsqueeze(0)
    if tuple(transformed_batched.shape) != INPUT_SHAPE:
        raise RuntimeError(f"batched prepared video contract mismatch: {tuple(transformed_batched.shape)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate = output_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    progress("prepare: intermediate serialization start")
    numpy_module.savez_compressed(
        intermediate / "preprocessed.npz",
        crops=numpy_module.stack(crops, axis=0),
        transformed=transformed_batched.detach().cpu().numpy(),
    )
    return transformed_batched, numpy_module.stack(crops, axis=0)


def to_output_frames(decoded: Any, torch_module: Any, correction: str, transformed: Any) -> Any:
    # decoded: [1,C,T,H,W] -> [T,C,H,W]
    sample = tensor_cpu_float(decoded, torch_module).squeeze(0).permute(1, 0, 2, 3).contiguous()
    style = transformed.squeeze(0).permute(1, 0, 2, 3).contiguous()
    if correction == "wavelet":
        from src.utils.color_fix import wavelet_reconstruction

        sample = wavelet_reconstruction(sample.clone(), style.clone(), None)
    sample = sample.clamp(-1.0, 1.0).mul(0.5).add(0.5)
    return sample.permute(0, 2, 3, 1).contiguous()


def save_images(
    output_dir: Path,
    crops: Any,
    torch_frames: Any,
    openvino_frames: Any,
    numpy_module: Any,
) -> Dict[str, Any]:
    from PIL import Image

    paths: Dict[str, List[str]] = {"bicubic": [], "torch": [], "openvino": []}
    baseline_dir = output_dir / "images" / "bicubic"
    torch_dir = output_dir / "images" / "torch"
    openvino_dir = output_dir / "images" / "openvino"
    for directory in (baseline_dir, torch_dir, openvino_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for index, crop in enumerate(crops):
        baseline = Image.fromarray(crop, mode="RGB").resize(
            (TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BICUBIC
        )
        baseline_path = baseline_dir / f"frame-{index + 1:02d}.png"
        baseline.save(baseline_path)
        paths["bicubic"].append(str(baseline_path))

        for label, frames, directory in (
            ("torch", torch_frames, torch_dir),
            ("openvino", openvino_frames, openvino_dir),
        ):
            values = numpy_module.asarray(frames[index].detach().cpu() if hasattr(frames[index], "detach") else frames[index])
            values = numpy_module.clip(values * 255.0 + 0.5, 0.0, 255.0).astype(numpy_module.uint8)
            path = directory / f"frame-{index + 1:02d}.png"
            Image.fromarray(values, mode="RGB").save(path)
            paths[label].append(str(path))
    return paths


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def load_preprocessed(path: Path, numpy_module: Any, torch_module: Any) -> Tuple[Any, Any]:
    require_file(path, "preprocessed intermediate")
    with numpy_module.load(path, allow_pickle=False) as stored:
        crops = numpy_module.array(stored["crops"], copy=True)
        transformed = torch_module.from_numpy(
            numpy_module.array(stored["transformed"], copy=True)
        ).to(dtype=torch_module.float32)
    if tuple(transformed.shape) != INPUT_SHAPE:
        raise RuntimeError(f"preprocessed intermediate shape mismatch: {tuple(transformed.shape)}")
    if tuple(crops.shape) != (5, CROP_HEIGHT, CROP_WIDTH, 3):
        raise RuntimeError(f"preprocessed crop shape mismatch: {tuple(crops.shape)}")
    return transformed, crops


def run_prepare_phase(
    input_dir: Path,
    output_dir: Path,
    source_root: Path,
    torch_module: Any,
    numpy_module: Any,
    result: Dict[str, Any],
    warnings_out: List[Dict[str, Any]],
) -> None:
    progress("prepare: crop/transform start")
    started = time.perf_counter()
    transformed, crops = prepare_inputs(
        input_dir, output_dir, source_root, torch_module, numpy_module, warnings_out
    )
    result["stages"]["preprocess"] = {
        "name": "preprocess",
        "status": "passed",
        "output": str(output_dir / "intermediate" / "preprocessed.npz"),
        "input_shape": list(transformed.shape),
        "crop_shape": list(crops.shape),
        "timings_seconds": {"prepare_and_serialize_seconds": time.perf_counter() - started},
    }
    result.setdefault("artifacts", {})["preprocessed"] = str(
        output_dir / "intermediate" / "preprocessed.npz"
    )
    result["memory"]["after_preprocess"] = peak_rss()
    del transformed, crops
    gc.collect()


def run_encode_phase(
    source_root: Path,
    vae_weights: Path,
    output_dir: Path,
    torch_module: Any,
    numpy_module: Any,
    ov_module: Any,
    vae_helper: Any,
    threads: int,
    result: Dict[str, Any],
    warnings_out: List[Dict[str, Any]],
) -> None:
    intermediate_dir = output_dir / "intermediate"
    transformed, _crops = load_preprocessed(
        intermediate_dir / "preprocessed.npz", numpy_module, torch_module
    )
    del _crops
    progress("vae_encoder: model load start")
    model_load_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vae, vae_structure = vae_helper.load_reduced_vae(source_root, vae_weights)
    warnings_out.extend(warning_record(item) for item in caught)
    vae = vae.to(device="cpu", dtype=torch_module.float32).eval()
    result["stages"]["vae_encoder"] = {
        "weights": vae_structure,
        "parameter_count": int(sum(parameter.numel() for parameter in vae.parameters())),
        "model_load_seconds": time.perf_counter() - model_load_started,
    }
    encoder, decoder_unused = vae_helper.make_wrappers(vae, torch_module)
    del decoder_unused
    encoded_input_b = transformed + torch_module.randn(
        transformed.shape,
        generator=torch_module.Generator(device="cpu").manual_seed(SEED + 1),
        dtype=torch_module.float32,
    ) * 0.001
    encoder_stage, encoder_outputs, encoder_openvino_outputs, _unused_encoder_additional = run_vae_stage(
        "vae_encoder",
        encoder,
        [transformed, encoded_input_b],
        RAW_LATENT_SHAPE,
        torch_module,
        ov_module,
        vae_helper,
        intermediate_dir.parent / "ir" / "vae-encoder.xml",
        threads,
        warnings_out,
    )
    result["stages"]["vae_encoder"].update(encoder_stage)
    raw_latent = tensor_cpu_float(encoder_outputs[0], torch_module)
    raw_latent_openvino = tensor_cpu_float(encoder_openvino_outputs[0], torch_module)
    scaled_latent = raw_latent.mul(VAE_SCALE)
    scaled_latent_openvino = raw_latent_openvino.mul(VAE_SCALE)
    channels_last_latent = scaled_latent.squeeze(0).permute(1, 2, 3, 0).contiguous()
    channels_last_latent_openvino = (
        scaled_latent_openvino.squeeze(0).permute(1, 2, 3, 0).contiguous()
    )
    for label, value in (
        ("scaled Torch", channels_last_latent),
        ("scaled OpenVINO", channels_last_latent_openvino),
    ):
        if tuple(value.shape) != CHANNELS_LAST_LATENT_SHAPE:
            raise RuntimeError(f"{label} channel-last latent mismatch: {tuple(value.shape)}")
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    progress("vae_encoder: intermediate serialization start")
    intermediate_serialize_started = time.perf_counter()
    numpy_module.savez_compressed(
        intermediate_dir / "encoded.npz",
        transformed=transformed.detach().cpu().numpy(),
        latent_raw=raw_latent.detach().cpu().numpy(),
        latent_raw_openvino=raw_latent_openvino.detach().cpu().numpy(),
        latent_scaled=scaled_latent.detach().cpu().numpy(),
        latent_scaled_openvino=scaled_latent_openvino.detach().cpu().numpy(),
        latent_channels_last=channels_last_latent.detach().cpu().numpy(),
        latent_channels_last_openvino=channels_last_latent_openvino.detach().cpu().numpy(),
    )
    result["stages"]["vae_encoder"]["intermediate_serialization_seconds"] = (
        time.perf_counter() - intermediate_serialize_started
    )
    result.setdefault("artifacts", {})["encoded"] = str(intermediate_dir / "encoded.npz")
    result["memory"]["after_vae_encoder"] = peak_rss()
    del (
        encoder,
        vae,
        encoder_outputs,
        encoder_openvino_outputs,
        raw_latent,
        raw_latent_openvino,
        scaled_latent,
        scaled_latent_openvino,
        channels_last_latent,
        channels_last_latent_openvino,
        encoded_input_b,
        transformed,
    )
    gc.collect()
    result["memory"]["after_vae_encoder_release"] = peak_rss()


def run_dit_prepare_phase(
    source_root: Path,
    dit_weights: Path,
    output_dir: Path,
    torch_module: Any,
    numpy_module: Any,
    ov_module: Any,
    dit_helper: Any,
    result: Dict[str, Any],
    warnings_out: List[Dict[str, Any]],
) -> None:
    """Build and serialize the fixed DiT candidate without compiling it.

    This phase deliberately ends before any OpenVINO CPU compilation or inference.  The
    Torch model, wrapper, trace, and frozen graph are released before candidate IR
    serialization so the next validation process can start with only the IR and saved
    tensors resident.
    """

    intermediate_dir = output_dir / "intermediate"
    encoded_path = require_file(intermediate_dir / "encoded.npz", "encoded intermediate")
    candidate_xml = output_dir / "ir" / "dit-candidate.xml"
    candidate_bin = candidate_xml.with_suffix(".bin")
    prepare_path = intermediate_dir / "dit-prepare.npz"
    if candidate_xml.exists() or candidate_bin.exists() or prepare_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing DiT prepare artifact: "
            f"{candidate_xml}, {candidate_bin}, or {prepare_path}"
        )

    with numpy_module.load(encoded_path, allow_pickle=False) as encoded:
        latent_channels_last = torch_module.from_numpy(
            numpy_module.array(encoded["latent_channels_last"], copy=True)
        ).to(dtype=torch_module.float32)
        latent_channels_last_openvino = torch_module.from_numpy(
            numpy_module.array(encoded["latent_channels_last_openvino"], copy=True)
        ).to(dtype=torch_module.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text_positive = torch_module.load(
            source_root / "pos_emb.pt", map_location="cpu", weights_only=True
        ).to(dtype=torch_module.float32)
    warnings_out.extend(warning_record(item) for item in caught)
    if tuple(text_positive.shape) != DIT_TEXT_SHAPE:
        raise RuntimeError(f"positive embedding shape mismatch: {tuple(text_positive.shape)}")

    noise_a = torch_module.randn(
        CHANNELS_LAST_LATENT_SHAPE,
        generator=torch_module.Generator(device="cpu").manual_seed(SEED),
        dtype=torch_module.float32,
    )
    noise_b = noise_a + torch_module.randn(
        CHANNELS_LAST_LATENT_SHAPE,
        generator=torch_module.Generator(device="cpu").manual_seed(SEED + 2),
        dtype=torch_module.float32,
    ) * 0.001
    ones = torch_module.ones(
        (*CHANNELS_LAST_LATENT_SHAPE[:-1], 1), dtype=torch_module.float32
    )
    condition = torch_module.cat([latent_channels_last, ones], dim=-1)
    condition_openvino = torch_module.cat([latent_channels_last_openvino, ones], dim=-1)
    video_a = torch_module.cat([noise_a, condition], dim=-1).flatten(0, -2)
    video_b = torch_module.cat([noise_b, condition], dim=-1).flatten(0, -2)
    video_openvino_chain = torch_module.cat(
        [noise_a, condition_openvino], dim=-1
    ).flatten(0, -2)
    timestep = torch_module.tensor([1000.0], dtype=torch_module.float32)
    dit_inputs_a = (video_a, text_positive, timestep)
    dit_inputs_b = (video_b, text_positive, timestep)
    dit_chain_inputs = (video_openvino_chain, text_positive, timestep)
    for label, tensor, expected in (
        ("video", video_a, DIT_VIDEO_SHAPE),
        ("video_openvino_chain", video_openvino_chain, DIT_VIDEO_SHAPE),
        ("text", text_positive, DIT_TEXT_SHAPE),
        ("timestep", timestep, DIT_TIMESTEP_SHAPE),
    ):
        if tuple(tensor.shape) != expected:
            raise RuntimeError(f"DiT {label} shape mismatch: {tuple(tensor.shape)} != {expected}")

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    dit_inputs_path = intermediate_dir / "dit-inputs.npz"
    if not dit_inputs_path.exists():
        progress("dit-prepare: input serialization start")
        numpy_module.savez_compressed(
            dit_inputs_path,
            video_a=video_a.detach().cpu().numpy(),
            video_b=video_b.detach().cpu().numpy(),
            video_openvino_chain=video_openvino_chain.detach().cpu().numpy(),
            text_positive=text_positive.detach().cpu().numpy(),
            timestep=timestep.detach().cpu().numpy(),
            noise_a=noise_a.detach().cpu().numpy(),
            noise_b=noise_b.detach().cpu().numpy(),
            condition=condition.detach().cpu().numpy(),
            condition_openvino=condition_openvino.detach().cpu().numpy(),
        )
    else:
        progress("dit-prepare: preserving existing dit-inputs.npz")

    progress("dit-prepare: model load start")
    model_load_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dit_model = dit_helper._load_nadit(source_root, torch_module, dit_weights)
    warnings_out.extend(warning_record(item) for item in caught)
    stage: Dict[str, Any] = {
        "name": "dit-prepare",
        "status": "failed",
        "parameter_count": int(sum(parameter.numel() for parameter in dit_model.parameters())),
        "parameter_dtype": str(next(dit_model.parameters()).dtype),
        "model_load_seconds": time.perf_counter() - model_load_started,
        "model_config": {
            "vid_in_channels": 33,
            "vid_out_channels": 16,
            "vid_shape": [[2, 96, 120]],
            "txt_shape": [[58]],
            "timestep": 1000.0,
            "cfg_scale": 1.0,
            "one_step": True,
        },
        "input_shapes": [
            [list(item.shape) for item in dit_inputs_a],
            [list(item.shape) for item in dit_inputs_b],
        ],
        "chain_input_shapes": [list(item.shape) for item in dit_chain_inputs],
        "expected_output_shape": list(DIT_OUTPUT_SHAPE),
        "distinct_inputs": not bool(torch_module.equal(video_a, video_b)),
        "timings_seconds": {},
    }
    timings = stage["timings_seconds"]
    wrapper = fixed_dit_wrapper(dit_model, torch_module)
    progress("dit-prepare: eager reference start")
    started = time.perf_counter()
    with torch_module.inference_mode():
        eager_outputs = [wrapper(*dit_inputs_a), wrapper(*dit_inputs_b)]
    timings["reference_seconds"] = time.perf_counter() - started
    eager_outputs_cpu = [tensor_cpu_float(value, torch_module).clone() for value in eager_outputs]
    stage["reference"] = {
        "outputs": [
            finite_shape_summary(value, DIT_OUTPUT_SHAPE, torch_module)
            for value in eager_outputs_cpu
        ],
        "vectors": [vector_summary(value, torch_module) for value in eager_outputs_cpu],
    }
    if not all(item["shape_matches"] and item["finite"] for item in stage["reference"]["outputs"]):
        stage["reference_error"] = {"message": "eager DiT output shape/finite contract failed"}
        raise StageFailure("dit-prepare", stage)
    reference_delta = output_variation(eager_outputs_cpu[0], eager_outputs_cpu[1], torch_module)
    stage["reference_variation"] = reference_delta
    if not reference_delta["different"]:
        raise StageFailure("dit-prepare", stage)

    wrapped_windows = dit_helper._specialize_fixed_window_ops(dit_model)
    stage["fixed_window_wrappers"] = wrapped_windows
    progress("dit-prepare: TorchScript trace/freeze start")
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch_module.inference_mode():
            traced = torch_module.jit.trace(
                wrapper,
                tuple(dit_inputs_a),
                strict=False,
                check_trace=False,
            ).eval()
            frozen = torch_module.jit.freeze(traced)
            traced_outputs = [frozen(*dit_inputs_a), frozen(*dit_inputs_b)]
    warnings_out.extend(warning_record(item) for item in caught)
    timings["trace_freeze_seconds"] = time.perf_counter() - started
    trace_contract, trace_passed = stage_contract(
        eager_outputs_cpu,
        traced_outputs,
        DIT_OUTPUT_SHAPE,
        torch_module,
    )
    stage["trace"] = trace_contract
    if not trace_passed:
        raise StageFailure("dit-prepare", stage)

    progress("dit-prepare: OpenVINO conversion start")
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch_module.inference_mode():
                ov_model = ov_module.convert_model(
                    frozen,
                    example_input=tuple(dit_inputs_a),
                )
        warnings_out.extend(warning_record(item) for item in caught)
        timings["conversion_seconds"] = time.perf_counter() - started
        if len(ov_model.inputs) != 3 or len(ov_model.outputs) != 1:
            raise RuntimeError(
                f"unexpected DiT graph IO: inputs={len(ov_model.inputs)} outputs={len(ov_model.outputs)}"
            )
        ov_model.reshape(
            {
                port: list(tensor.shape)
                for port, tensor in zip(ov_model.inputs, dit_inputs_a)
            }
        )
        stage["candidate_io"] = {
            "inputs": [
                dit_helper._port_info(port, index)
                for index, port in enumerate(ov_model.inputs)
            ],
            "outputs": [
                dit_helper._port_info(port, index)
                for index, port in enumerate(ov_model.outputs)
            ],
        }
    except BaseException as exc:
        stage["conversion_error"] = exception_record(exc)
        raise StageFailure("dit-prepare", stage) from exc

    # Keep only CPU tensors needed by the validation process before writing the candidate
    # graph.  In particular, do not retain Torch weights/traces alongside the graph.
    del dit_model, wrapper, traced, frozen, traced_outputs, eager_outputs
    gc.collect()
    progress("dit-prepare: candidate IR serialization start")
    started = time.perf_counter()
    stage["candidate_ir"] = save_ir(ov_module, ov_model, candidate_xml)
    timings["candidate_serialization_seconds"] = time.perf_counter() - started
    del ov_model
    gc.collect()

    progress("dit-prepare: prepared tensor serialization start")
    started = time.perf_counter()
    numpy_module.savez_compressed(
        prepare_path,
        video_a=video_a.detach().cpu().numpy(),
        video_b=video_b.detach().cpu().numpy(),
        video_openvino_chain=video_openvino_chain.detach().cpu().numpy(),
        text_positive=text_positive.detach().cpu().numpy(),
        timestep=timestep.detach().cpu().numpy(),
        noise_a=noise_a.detach().cpu().numpy(),
        noise_b=noise_b.detach().cpu().numpy(),
        condition=condition.detach().cpu().numpy(),
        condition_openvino=condition_openvino.detach().cpu().numpy(),
        eager_output_a=eager_outputs_cpu[0].detach().cpu().numpy(),
        eager_output_b=eager_outputs_cpu[1].detach().cpu().numpy(),
    )
    timings["prepared_tensor_serialization_seconds"] = time.perf_counter() - started
    stage["status"] = "prepared_unvalidated"
    stage["candidate_validation_status"] = "UNVALIDATED"
    stage["reference_validation_status"] = "UNVALIDATED"
    stage["validation_required"] = True
    result["stages"]["dit-prepare"] = stage
    result.setdefault("artifacts", {})["dit_inputs"] = str(dit_inputs_path)
    result["artifacts"]["dit_prepare"] = str(prepare_path)
    result["artifacts"]["dit_candidate_ir"] = str(candidate_xml)
    result["memory"]["after_dit_prepare"] = peak_rss()
    del (
        latent_channels_last,
        latent_channels_last_openvino,
        text_positive,
        noise_a,
        noise_b,
        condition,
        condition_openvino,
        video_a,
        video_b,
        video_openvino_chain,
        timestep,
        dit_inputs_a,
        dit_inputs_b,
        dit_chain_inputs,
        eager_outputs_cpu,
    )
    gc.collect()
    result["memory"]["after_dit_prepare_release"] = peak_rss()


def run_dit_validate_phase(
    output_dir: Path,
    torch_module: Any,
    numpy_module: Any,
    ov_module: Any,
    dit_helper: Any,
    threads: int,
    result: Dict[str, Any],
) -> None:
    """Compile/infer the candidate IR without loading Torch DiT weights or traces."""

    intermediate_dir = output_dir / "intermediate"
    prepare_path = require_file(intermediate_dir / "dit-prepare.npz", "DiT prepare intermediate")
    candidate_xml = require_file(output_dir / "ir" / "dit-candidate.xml", "DiT candidate IR")
    candidate_bin = require_file(candidate_xml.with_suffix(".bin"), "DiT candidate IR weights")
    standard_xml = output_dir / "ir" / "dit.xml"
    standard_bin = output_dir / "ir" / "dit.bin"
    if standard_xml.exists() or standard_bin.exists():
        raise FileExistsError(
            f"refusing to overwrite promoted DiT IR: {standard_xml} or {standard_bin}"
        )

    with numpy_module.load(prepare_path, allow_pickle=False) as prepared:
        video_a = torch_module.from_numpy(numpy_module.array(prepared["video_a"], copy=True)).to(dtype=torch_module.float32)
        video_b = torch_module.from_numpy(numpy_module.array(prepared["video_b"], copy=True)).to(dtype=torch_module.float32)
        video_openvino_chain = torch_module.from_numpy(
            numpy_module.array(prepared["video_openvino_chain"], copy=True)
        ).to(dtype=torch_module.float32)
        text_positive = torch_module.from_numpy(
            numpy_module.array(prepared["text_positive"], copy=True)
        ).to(dtype=torch_module.float32)
        timestep = torch_module.from_numpy(numpy_module.array(prepared["timestep"], copy=True)).to(dtype=torch_module.float32)
        noise_a = torch_module.from_numpy(numpy_module.array(prepared["noise_a"], copy=True)).to(dtype=torch_module.float32)
        eager_outputs = [
            torch_module.from_numpy(numpy_module.array(prepared["eager_output_a"], copy=True)).to(dtype=torch_module.float32),
            torch_module.from_numpy(numpy_module.array(prepared["eager_output_b"], copy=True)).to(dtype=torch_module.float32),
        ]
    dit_inputs_a = (video_a, text_positive, timestep)
    dit_inputs_b = (video_b, text_positive, timestep)
    dit_chain_inputs = (video_openvino_chain, text_positive, timestep)
    stage: Dict[str, Any] = {
        "name": "dit-validate",
        "status": "failed",
        "input_shapes": [
            [list(item.shape) for item in dit_inputs_a],
            [list(item.shape) for item in dit_inputs_b],
        ],
        "chain_input_shapes": [list(item.shape) for item in dit_chain_inputs],
        "expected_output_shape": list(DIT_OUTPUT_SHAPE),
        "distinct_inputs": not bool(torch_module.equal(video_a, video_b)),
        "candidate_ir": str(candidate_xml),
        "candidate_bin": str(candidate_bin),
        "timings_seconds": {},
    }
    timings = stage["timings_seconds"]
    core = ov_module.Core()
    progress("dit-validate: candidate IR load start")
    started = time.perf_counter()
    ov_model = core.read_model(str(candidate_xml))
    timings["candidate_load_seconds"] = time.perf_counter() - started
    if len(ov_model.inputs) != 3 or len(ov_model.outputs) != 1:
        raise RuntimeError(
            f"unexpected candidate DiT graph IO: inputs={len(ov_model.inputs)} outputs={len(ov_model.outputs)}"
        )
    ov_model.reshape(
        {
            port: list(tensor.shape)
            for port, tensor in zip(ov_model.inputs, dit_inputs_a)
        }
    )
    stage["openvino_io"] = {
        "inputs": [
            dit_helper._port_info(port, index)
            for index, port in enumerate(ov_model.inputs)
        ],
        "outputs": [
            dit_helper._port_info(port, index)
            for index, port in enumerate(ov_model.outputs)
        ],
    }
    compile_properties = {
        "NUM_STREAMS": "1",
        "INFERENCE_NUM_THREADS": str(threads),
        "INFERENCE_PRECISION_HINT": ov_module.Type.f32,
    }
    progress("dit-validate: OpenVINO compile start")
    started = time.perf_counter()
    compiled = core.compile_model(ov_model, "CPU", compile_properties)
    timings["compile_seconds"] = time.perf_counter() - started
    stage["compile_properties"] = {
        key: str(value) for key, value in compile_properties.items()
    }
    # The uncompiled graph is no longer needed once the CPU executable is built.
    del ov_model, core
    gc.collect()

    request = compiled.create_infer_request()
    progress("dit-validate: OpenVINO two-input inference start")
    started = time.perf_counter()
    openvino_outputs = []
    for inputs in (dit_inputs_a, dit_inputs_b):
        inferred = request.infer(
            {
                port: tensor.detach().cpu().numpy()
                for port, tensor in zip(compiled.inputs, inputs)
            }
        )
        openvino_outputs.append(
            torch_module.from_numpy(next(iter(inferred.values()))).clone()
        )
    timings["inference_seconds"] = time.perf_counter() - started
    contract, passed = stage_contract(
        eager_outputs, openvino_outputs, DIT_OUTPUT_SHAPE, torch_module
    )
    stage.update(contract)
    stage["status"] = "passed" if passed and stage["distinct_inputs"] else "failed"
    if stage["status"] != "passed":
        raise StageFailure("dit-validate", stage)

    progress("dit-validate: OpenVINO chain inference start")
    started = time.perf_counter()
    chain_inferred = request.infer(
        {
            port: tensor.detach().cpu().numpy()
            for port, tensor in zip(compiled.inputs, dit_chain_inputs)
        }
    )
    chain_output = torch_module.from_numpy(next(iter(chain_inferred.values()))).clone()
    timings["chain_inference_seconds"] = time.perf_counter() - started
    chain_summary = finite_shape_summary(chain_output, DIT_OUTPUT_SHAPE, torch_module)
    stage["openvino_chain"] = {
        "output": chain_summary,
        "vector": vector_summary(chain_output, torch_module),
    }
    if not chain_summary["shape_matches"] or not chain_summary["finite"]:
        raise StageFailure("dit-validate", stage)

    restored_latent = (
        noise_a.flatten(0, -2) - eager_outputs[0]
    ).unflatten(0, [2, 96, 120])
    restored_latent_openvino = (
        noise_a.flatten(0, -2) - chain_output
    ).unflatten(0, [2, 96, 120])
    restored_latent_openvino_same_input = (
        noise_a.flatten(0, -2) - openvino_outputs[0]
    ).unflatten(0, [2, 96, 120])
    progress("dit-validate: output serialization start")
    started = time.perf_counter()
    numpy_module.savez_compressed(
        intermediate_dir / "dit-outputs.npz",
        prediction_torch=eager_outputs[0].detach().cpu().numpy(),
        prediction_openvino_same_input=openvino_outputs[0].detach().cpu().numpy(),
        prediction_openvino_chain=chain_output.detach().cpu().numpy(),
        restored_latent_torch=restored_latent.detach().cpu().numpy(),
        restored_latent_openvino=restored_latent_openvino.detach().cpu().numpy(),
        restored_latent_openvino_same_input=restored_latent_openvino_same_input.detach().cpu().numpy(),
    )
    timings["output_serialization_seconds"] = time.perf_counter() - started

    progress("dit-validate: candidate IR promotion start")
    started = time.perf_counter()
    candidate_xml.replace(standard_xml)
    candidate_bin.replace(standard_bin)
    timings["promotion_seconds"] = time.perf_counter() - started
    stage["status"] = "passed"
    stage["ir"] = {
        "promoted_from_xml": str(candidate_xml),
        "promoted_from_bin": str(candidate_bin),
        "xml": str(standard_xml),
        "bin": str(standard_bin),
        "compress_to_fp16": False,
    }
    result["stages"]["dit-validate"] = stage
    result.setdefault("artifacts", {})["dit_inputs"] = str(intermediate_dir / "dit-inputs.npz")
    result["artifacts"]["dit_prepare"] = str(prepare_path)
    result["artifacts"]["dit_outputs"] = str(intermediate_dir / "dit-outputs.npz")
    result["artifacts"]["dit_ir"] = str(standard_xml)
    result["memory"]["after_dit_validate"] = peak_rss()
    del (
        compiled,
        request,
        openvino_outputs,
        chain_output,
        restored_latent,
        restored_latent_openvino,
        restored_latent_openvino_same_input,
        video_a,
        video_b,
        video_openvino_chain,
        text_positive,
        timestep,
        noise_a,
        eager_outputs,
        dit_inputs_a,
        dit_inputs_b,
        dit_chain_inputs,
    )
    gc.collect()
    result["memory"]["after_dit_validate_release"] = peak_rss()


def run_decode_phase(
    vae_weights: Path,
    output_dir: Path,
    torch_module: Any,
    numpy_module: Any,
    ov_module: Any,
    vae_helper: Any,
    source_root: Path,
    correction: str,
    threads: int,
    result: Dict[str, Any],
    warnings_out: List[Dict[str, Any]],
) -> None:
    intermediate_dir = output_dir / "intermediate"
    encoded_path = require_file(intermediate_dir / "encoded.npz", "encoded intermediate")
    dit_outputs_path = require_file(intermediate_dir / "dit-outputs.npz", "DiT outputs intermediate")
    with numpy_module.load(encoded_path, allow_pickle=False) as encoded, numpy_module.load(
        dit_outputs_path, allow_pickle=False
    ) as dit_outputs, numpy_module.load(
        intermediate_dir / "preprocessed.npz", allow_pickle=False
    ) as preprocessed:
        transformed = torch_module.from_numpy(
            numpy_module.array(encoded["transformed"], copy=True)
        ).to(dtype=torch_module.float32)
        crops = numpy_module.array(preprocessed["crops"], copy=True)
        restored_scaled = torch_module.from_numpy(
            numpy_module.array(dit_outputs["restored_latent_torch"], copy=True)
        ).to(dtype=torch_module.float32)
        restored_scaled_openvino = torch_module.from_numpy(
            numpy_module.array(dit_outputs["restored_latent_openvino"], copy=True)
        ).to(dtype=torch_module.float32)
    restored_raw = (
        restored_scaled.div(VAE_SCALE).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    )
    restored_raw_openvino = (
        restored_scaled_openvino.div(VAE_SCALE)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .contiguous()
    )
    decoder_input_b = restored_raw + torch_module.randn(
        restored_raw.shape,
        generator=torch_module.Generator(device="cpu").manual_seed(SEED + 3),
        dtype=torch_module.float32,
    ) * 0.001
    progress("vae_decoder: model load start")
    model_load_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vae, vae_structure = vae_helper.load_reduced_vae(source_root, vae_weights)
    warnings_out.extend(warning_record(item) for item in caught)
    vae = vae.to(device="cpu", dtype=torch_module.float32).eval()
    result["stages"]["vae_decoder"] = {
        "weights": vae_structure,
        "parameter_count": int(sum(parameter.numel() for parameter in vae.parameters())),
        "model_load_seconds": time.perf_counter() - model_load_started,
    }
    encoder_unused, decoder = vae_helper.make_wrappers(vae, torch_module)
    del encoder_unused
    decoder_stage, decoder_outputs, decoder_openvino_outputs, decoder_chain_outputs = run_vae_stage(
        "vae_decoder",
        decoder,
        [restored_raw, decoder_input_b],
        INPUT_SHAPE,
        torch_module,
        ov_module,
        vae_helper,
        intermediate_dir.parent / "ir" / "vae-decoder.xml",
        threads,
        warnings_out,
        additional_inputs=[restored_raw_openvino],
    )
    result["stages"]["vae_decoder"].update(decoder_stage)
    torch_frames = to_output_frames(decoder_outputs[0], torch_module, correction, transformed)
    openvino_same_input_frames = to_output_frames(
        decoder_openvino_outputs[0], torch_module, correction, transformed
    )
    openvino_frames = to_output_frames(
        decoder_chain_outputs[0], torch_module, correction, transformed
    )
    torch_summary = finite_shape_summary(torch_frames, FINAL_SHAPE, torch_module)
    openvino_summary = finite_shape_summary(openvino_frames, FINAL_SHAPE, torch_module)
    same_input_summary = finite_shape_summary(
        openvino_same_input_frames, FINAL_SHAPE, torch_module
    )
    if not torch_summary["shape_matches"] or not torch_summary["finite"]:
        raise RuntimeError(f"Torch final output shape/finite contract failed: {torch_summary}")
    if not openvino_summary["shape_matches"] or not openvino_summary["finite"]:
        raise RuntimeError(f"OpenVINO final output shape/finite contract failed: {openvino_summary}")
    if not same_input_summary["shape_matches"] or not same_input_summary["finite"]:
        raise RuntimeError(
            f"OpenVINO same-input final output shape/finite contract failed: {same_input_summary}"
        )
    same_input_metrics = error_metrics(torch_frames, openvino_same_input_frames, torch_module)
    final_metrics = error_metrics(torch_frames, openvino_frames, torch_module)
    for label, metrics in (("same-input decoder", same_input_metrics), ("full chain", final_metrics)):
        if (
            metrics.get("max_abs") is None
            or metrics.get("rms_relative_error") is None
            or metrics["max_abs"] > 1.0e-3
            or metrics["rms_relative_error"] > 1.0e-4
        ):
            raise RuntimeError(f"{label} Torch/OpenVINO restoration exceeds tolerance: {metrics}")
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    progress("vae_decoder: input serialization start")
    input_serialize_started = time.perf_counter()
    numpy_module.savez_compressed(
        intermediate_dir / "decode-inputs.npz",
        decoder_input_torch=restored_raw.detach().cpu().numpy(),
        decoder_input_openvino=restored_raw_openvino.detach().cpu().numpy(),
        decoder_input_perturbed=decoder_input_b.detach().cpu().numpy(),
        transformed=transformed.detach().cpu().numpy(),
    )
    input_serialize_seconds = time.perf_counter() - input_serialize_started
    progress("vae_decoder: output serialization start")
    output_serialize_started = time.perf_counter()
    numpy_module.savez_compressed(
        intermediate_dir / "decode-outputs.npz",
        decoded_torch=decoder_outputs[0].detach().cpu().numpy(),
        decoded_openvino_same_input=decoder_openvino_outputs[0].detach().cpu().numpy(),
        decoded_openvino_chain=decoder_chain_outputs[0].detach().cpu().numpy(),
        final_torch=torch_frames.detach().cpu().numpy(),
        final_openvino_same_input=openvino_same_input_frames.detach().cpu().numpy(),
        final_openvino_chain=openvino_frames.detach().cpu().numpy(),
    )
    result["stages"]["vae_decoder"]["input_serialization_seconds"] = input_serialize_seconds
    result["stages"]["vae_decoder"]["output_serialization_seconds"] = (
        time.perf_counter() - output_serialize_started
    )
    result["final"] = {
        "torch": torch_summary,
        "openvino_chain": openvino_summary,
        "openvino_same_input": same_input_summary,
        "comparison_same_input_decoder": same_input_metrics,
        "comparison_full_chain": final_metrics,
        "vectors": {
            "torch": vector_summary(torch_frames, torch_module),
            "openvino_chain": vector_summary(openvino_frames, torch_module),
            "openvino_same_input": vector_summary(openvino_same_input_frames, torch_module),
        },
    }
    result.setdefault("artifacts", {})["decode_inputs"] = str(intermediate_dir / "decode-inputs.npz")
    result["artifacts"]["decode_outputs"] = str(intermediate_dir / "decode-outputs.npz")
    result["artifacts"]["images"] = save_images(
        output_dir, crops, torch_frames, openvino_frames, numpy_module
    )
    result["memory"]["after_vae_decoder"] = peak_rss()
    del (
        vae,
        decoder,
        decoder_outputs,
        decoder_openvino_outputs,
        decoder_chain_outputs,
        restored_raw,
        restored_raw_openvino,
        restored_scaled,
        restored_scaled_openvino,
        decoder_input_b,
        transformed,
        crops,
        torch_frames,
        openvino_same_input_frames,
        openvino_frames,
    )
    gc.collect()
    result["memory"]["after_vae_decoder_release"] = peak_rss()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--inputs-dir", required=True, type=Path)
    parser.add_argument("--vae-weights", required=True, type=Path)
    parser.add_argument("--dit-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=("prepare", "encode", "dit-prepare", "dit-validate", "decode"),
        default="prepare",
        help="run one explicit resumable phase",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Torch/OpenVINO CPU inference threads (default: 2)",
    )
    parser.add_argument("--color-correction", choices=("wavelet", "none"), default="wavelet")
    return parser.parse_args(argv)


def initial_result(
    args: argparse.Namespace,
    source_root: Path,
    input_dir: Path,
    vae_weights: Path,
    dit_weights: Path,
) -> Dict[str, Any]:
    return {
        "probe": "seedvr2-video-openvino-crop",
        "status": "not_started",
        "phase": args.phase,
        "source_root": str(source_root),
        "input_dir": str(input_dir),
        "weights": {
            "vae": str(vae_weights),
            "dit": str(dit_weights),
            "strict": True,
            "dtype": "torch.float32 CPU reference",
        },
        "cpu_only": True,
        "accelerator_queries": "not called; synthetic guards installed before source imports",
        "threads": {
            "torch": args.threads,
            "torch_interop": 1,
            "openvino_streams": 1,
            "openvino_threads": args.threads,
        },
        "crop": {
            "x": CROP_X,
            "y": CROP_Y,
            "width": CROP_WIDTH,
            "height": CROP_HEIGHT,
            "scale": 3,
            "output_width": TARGET_WIDTH,
            "output_height": TARGET_HEIGHT,
            "frames": list(FRAME_NAMES),
        },
        "expected": {
            "preprocessed": list(INPUT_SHAPE),
            "vae_raw_latent": list(RAW_LATENT_SHAPE),
            "channels_last_latent": list(CHANNELS_LAST_LATENT_SHAPE),
            "dit_vid": list(DIT_VIDEO_SHAPE),
            "dit_txt": list(DIT_TEXT_SHAPE),
            "dit_timestep": list(DIT_TIMESTEP_SHAPE),
            "dit_output": list(DIT_OUTPUT_SHAPE),
            "decoded": list(INPUT_SHAPE),
            "final_frames": list(FINAL_SHAPE),
        },
        "seed": SEED,
        "vae_scaling_factor": VAE_SCALE,
        "color_correction": args.color_correction,
        "warnings": [],
        "stages": {},
        "memory": {},
        "artifacts": {},
    }


def restore_prior_evidence(result: Dict[str, Any], evidence_path: Path) -> List[Dict[str, Any]]:
    if not evidence_path.is_file():
        return []
    try:
        prior = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if prior.get("probe") != result["probe"]:
        return []
    for key in ("stages", "memory", "artifacts"):
        if isinstance(prior.get(key), dict):
            result[key].update(prior[key])
    result["resumed_from"] = str(evidence_path)
    prior_warnings = prior.get("warnings", [])
    return prior_warnings if isinstance(prior_warnings, list) else []


def validate_paths(
    phase: str,
    source_root: Path,
    input_dir: Path,
    vae_weights: Path,
    dit_weights: Path,
) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_root}")
    if not (source_root / "configs_3b" / "main.yaml").is_file():
        raise FileNotFoundError(f"missing 3B config under source: {source_root}")
    if phase == "prepare" and not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if phase in ("encode", "decode"):
        if not vae_weights.is_file() or vae_weights.suffix != ".safetensors":
            raise FileNotFoundError(f"missing VAE safetensors: {vae_weights}")
    if phase == "dit-prepare":
        if not dit_weights.is_file() or dit_weights.suffix != ".safetensors":
            raise FileNotFoundError(f"missing DiT safetensors: {dit_weights}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be a positive integer")
    source_root = args.source.expanduser().resolve()
    input_dir = args.inputs_dir.expanduser().resolve()
    vae_weights = args.vae_weights.expanduser().resolve()
    dit_weights = args.dit_weights.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result = initial_result(args, source_root, input_dir, vae_weights, dit_weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_warnings = restore_prior_evidence(result, output_dir / "evidence.json")
    warnings_out: List[Dict[str, Any]] = []
    try:
        import torch

        # These helper files contain only stdlib imports until their phase functions
        # import SeedVR2.  Install guards before any source module is imported.
        vae_helper = load_helper(
            Path(__file__).with_name("openvino-vae-probe.py"),
            "seedvr2_vae_probe_helper",
        )
        result["patched_backends"] = vae_helper.force_cpu_only(torch)
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        validate_paths(args.phase, source_root, input_dir, vae_weights, dit_weights)
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        import numpy as np

        if args.phase == "prepare":
            run_prepare_phase(input_dir, output_dir, source_root, torch, np, result, warnings_out)
        else:
            import openvino as ov

            dit_helper = None
            if args.phase in ("dit-prepare", "dit-validate"):
                dit_helper = load_helper(
                    Path(__file__).with_name("openvino-probe.py"),
                    "seedvr2_dit_probe_helper",
                )
            if args.phase == "encode":
                run_encode_phase(
                    source_root,
                    vae_weights,
                    output_dir,
                    torch,
                    np,
                    ov,
                    vae_helper,
                    args.threads,
                    result,
                    warnings_out,
                )
            elif args.phase == "dit-prepare":
                run_dit_prepare_phase(
                    source_root,
                    dit_weights,
                    output_dir,
                    torch,
                    np,
                    ov,
                    dit_helper,
                    result,
                    warnings_out,
                )
            elif args.phase == "dit-validate":
                run_dit_validate_phase(
                    output_dir,
                    torch,
                    np,
                    ov,
                    dit_helper,
                    args.threads,
                    result,
                )
            elif args.phase == "decode":
                run_decode_phase(
                    vae_weights,
                    output_dir,
                    torch,
                    np,
                    ov,
                    vae_helper,
                    source_root,
                    args.color_correction,
                    args.threads,
                    result,
                    warnings_out,
                )
        if args.phase == "dit-prepare":
            result["status"] = "prepared_unvalidated"
            result["validation_required"] = True
        else:
            result["status"] = "passed"
    except StageFailure as exc:
        result["status"] = "failed"
        result["failed_stage"] = exc.stage
        result["stages"].setdefault(exc.stage, {}).update(exc.payload)
        result["exception"] = exception_record(exc)
    except BaseException as exc:
        result["status"] = "error"
        result["exception"] = exception_record(exc)
    finally:
        result["warnings"] = prior_warnings + warnings_out
        result["memory"]["final"] = peak_rss()
        evidence = json.dumps(result, indent=2, sort_keys=True) + "\n"
        (output_dir / "evidence.json").write_text(evidence, encoding="utf-8")
        (output_dir / f"evidence-{args.phase}.json").write_text(evidence, encoding="utf-8")

    print(json.dumps({"status": result["status"], "phase": args.phase, "evidence": str(output_dir / "evidence.json")}))
    return 0 if result["status"] in ("passed", "prepared_unvalidated") else 1


if __name__ == "__main__":
    raise SystemExit(main())
