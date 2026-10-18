#!/usr/bin/env python3
"""Raw CPU OpenVINO VAE-decoder roundtrip for artifact localization.

This reads the saved unscaled OpenVINO latent and matching transformed input,
decodes once through the existing CPU VAE decoder IR, and saves raw frames
without wavelet color correction.  It is not a quality or parity pass.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import resource
import time
from typing import Any, Dict, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
ENCODED_NPZ = ROOT / "evidence" / "video-crop-probe" / "intermediate" / "encoded.npz"
DECODER_IR = ROOT / "evidence" / "video-crop-probe" / "ir" / "vae-decoder.xml"
OUTPUT_DIR = ROOT / "evidence" / "vae-roundtrip-cpu-v1"
LATENT_SHAPE = (1, 16, 2, 96, 120)
VIDEO_SHAPE = (1, 3, 5, 768, 960)
FRAME_SHAPE = (5, 768, 960, 3)
THREADS = 8


class ContractError(RuntimeError):
    pass


def _rss() -> Dict[str, int]:
    kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"peak_rss_kib": kib, "peak_rss_bytes": kib * 1024}


def _array(np: Any, value: Any, label: str, shape: Sequence[int]) -> Any:
    if getattr(value, "dtype", None) != np.float32 or tuple(value.shape) != tuple(shape):
        raise ContractError(f"{label} must be float32 {tuple(shape)}, got {getattr(value, 'dtype', None)} {getattr(value, 'shape', None)}")
    if not bool(np.isfinite(value).all()):
        raise ContractError(f"{label} contains non-finite values")
    return np.array(value, dtype=np.float32, copy=True)


def load_inputs(np: Any) -> Tuple[Any, Any]:
    with np.load(str(ENCODED_NPZ), allow_pickle=False) as archive:
        for key in ("latent_raw_openvino", "transformed"):
            if key not in archive.files:
                raise ContractError(f"encoded.npz lacks {key}")
        latent = _array(np, archive["latent_raw_openvino"], "latent_raw_openvino", LATENT_SHAPE)
        transformed = _array(np, archive["transformed"], "transformed", VIDEO_SHAPE)
    return latent, transformed


def _type_name(port: Any) -> str:
    return str(port.get_element_type().get_type_name()).lower()


def _static_shape(port: Any) -> Tuple[int, ...]:
    partial = port.get_partial_shape()
    if not partial.is_static:
        raise ContractError(f"dynamic port shape is unsupported: {partial}")
    return tuple(int(value) for value in partial.to_shape())


def _compile_properties(ov: Any) -> Dict[str, Any]:
    return {
        "INFERENCE_PRECISION_HINT": ov.Type.f32,
        "INFERENCE_NUM_THREADS": str(THREADS),
        "NUM_STREAMS": "1",
        "CACHE_DIR": "",
    }


def decode_once(np: Any, ov: Any, latent: Any) -> Tuple[Any, Dict[str, Any]]:
    properties = _compile_properties(ov)
    core = model = compiled = request = None
    try:
        core = ov.Core()
        model = core.read_model(str(DECODER_IR))
        if len(model.inputs) != 1 or len(model.outputs) != 1:
            raise ContractError("decoder must have one input and one output")
        input_port, output_port = model.inputs[0], model.outputs[0]
        if _static_shape(input_port) != LATENT_SHAPE or _static_shape(output_port) != VIDEO_SHAPE:
            raise ContractError("decoder IO shape contract mismatch")
        if _type_name(input_port) not in {"f32", "float32"} or _type_name(output_port) not in {"f32", "float32"}:
            raise ContractError("decoder IO must be f32")
        compiled = core.compile_model(model, "CPU", properties)
        request = compiled.create_infer_request()
        inferred = request.infer({compiled.inputs[0]: np.array(latent, dtype=np.float32, copy=True)})
        raw = np.array(inferred[compiled.outputs[0]], dtype=np.float32, copy=True)
        return raw, {"input_type": _type_name(input_port), "output_type": _type_name(output_port), "compile_properties": {key: str(value) for key, value in properties.items()}}
    finally:
        request = compiled = model = core = None
        gc.collect()


def _raw_frames(np: Any, raw: Any) -> Any:
    checked = _array(np, raw, "decoder output", VIDEO_SHAPE)
    return np.transpose(checked[0], (1, 2, 3, 0))


def _rgb_frames(np: Any, frames: Any) -> Any:
    return np.rint(np.clip((frames + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)


def _new_output_dir() -> Path:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    return OUTPUT_DIR


def _save_pngs(np: Any, frames: Any, output_dir: Path) -> list[str]:
    from PIL import Image

    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=False)
    paths = []
    for index, frame in enumerate(_rgb_frames(np, frames), 1):
        path = image_dir / f"frame-{index:02d}.png"
        Image.fromarray(frame, mode="RGB").save(path)
        paths.append(str(path))
    return paths


def run(np: Any, ov: Any) -> Tuple[int, Dict[str, Any]]:
    output_dir = _new_output_dir()
    payload: Dict[str, Any] = {
        "schema": 1,
        "phase": "raw_vae_roundtrip",
        "status": "running",
        "validation_status": "visual_validation_required",
        "inputs": {"encoded_npz": str(ENCODED_NPZ), "latent_key": "latent_raw_openvino", "transformed_key": "transformed", "latent_shape": list(LATENT_SHAPE), "transformed_shape": list(VIDEO_SHAPE)},
        "decoder_ir": str(DECODER_IR),
        "timings_seconds": {},
        "memory": {"before": _rss()},
        "whole_ov_graph": False,
    }
    try:
        started = time.perf_counter()
        latent, transformed = load_inputs(np)
        payload["timings_seconds"]["load_inputs"] = time.perf_counter() - started
        started = time.perf_counter()
        raw, io_info = decode_once(np, ov, latent)
        payload["timings_seconds"]["compile_and_infer"] = time.perf_counter() - started
        raw_frames = _raw_frames(np, raw)
        transformed_frames = np.transpose(transformed[0], (1, 2, 3, 0))
        delta = np.abs(raw_frames.astype(np.float64) - transformed_frames.astype(np.float64))
        payload["decoder"] = {"raw_shape": list(raw.shape), "raw_finite": bool(np.isfinite(raw).all()), "raw_min": float(raw.min()), "raw_max": float(raw.max()), **io_info}
        payload["transformed_reference"] = {"shape": list(transformed.shape), "finite": bool(np.isfinite(transformed).all()), "min": float(transformed.min()), "max": float(transformed.max())}
        payload["per_frame_differences"] = [{"frame": index + 1, "mean_abs": float(delta[index].mean()), "max_abs": float(delta[index].max())} for index in range(5)]
        started = time.perf_counter()
        payload["artifacts"] = {"images": _save_pngs(np, raw_frames, output_dir)}
        payload["timings_seconds"]["postprocess_and_save"] = time.perf_counter() - started
        payload["status"] = "roundtrip_produced"
        return_code = 0
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return_code = 1
    payload["memory"]["final"] = _rss()
    (output_dir / "evidence.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return return_code, payload


def main() -> int:
    import numpy as np
    import openvino as ov

    code, payload = run(np, ov)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
