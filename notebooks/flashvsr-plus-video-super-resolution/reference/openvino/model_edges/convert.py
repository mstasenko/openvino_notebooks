#!/usr/bin/env python3
"""Small native-f16 OpenVINO prefix/suffix graphs for the pinned Wan source.

Only the non-block checkpoint tensors used by patch/time/head are read.  The
prefix is consumed by the transformer (tokens, ``t``, and ``t_mod``); the
suffix consumes the transformer tokens and the source head time state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
CHECKPOINT = BASE / "models/diffusion_pytorch_model_streaming_dmd.safetensors"
DIM, FREQ, IN_DIM, OUT_DIM = 1536, 256, 16, 16
FIXED_TIMESTEP = np.float64(1000.0)
PATCH = (1, 2, 2)
EPS = 1e-6
SELECTED_KEYS = (
    "patch_embedding.weight", "patch_embedding.bias",
    "time_embedding.0.weight", "time_embedding.0.bias",
    "time_embedding.2.weight", "time_embedding.2.bias",
    "time_projection.1.weight", "time_projection.1.bias",
    "head.modulation", "head.head.weight", "head.head.bias",
)
EXPECTED_SHAPES = {
    "patch_embedding.weight": [1536, 16, 1, 2, 2],
    "patch_embedding.bias": [1536],
    "time_embedding.0.weight": [1536, 256], "time_embedding.0.bias": [1536],
    "time_embedding.2.weight": [1536, 1536], "time_embedding.2.bias": [1536],
    "time_projection.1.weight": [9216, 1536], "time_projection.1.bias": [9216],
    "head.modulation": [1, 2, 1536],
    "head.head.weight": [64, 1536], "head.head.bias": [64],
}
LAYOUTS = {
    "patch_embedding.weight": "OI DHW (Conv3d)", "patch_embedding.bias": "O",
    "time_embedding.0.weight": "OI", "time_embedding.0.bias": "O",
    "time_embedding.2.weight": "OI", "time_embedding.2.bias": "O",
    "time_projection.1.weight": "OI", "time_projection.1.bias": "O",
    "head.modulation": "B,2,D", "head.head.weight": "OI", "head.head.bias": "O",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_weights(path: Path = CHECKPOINT):
    """Read all headers, then values only for the explicit edge allow-list."""
    from safetensors import safe_open

    values, tensors = {}, {}
    with safe_open(str(path), framework="np", device="cpu") as handle:
        all_keys = list(handle.keys())
        missing = sorted(set(SELECTED_KEYS) - set(all_keys))
        if missing:
            raise KeyError(f"missing edge checkpoint keys: {missing}")
        for key in SELECTED_KEYS:
            view = handle.get_slice(key)
            shape, dtype = list(view.get_shape()), view.get_dtype()
            if shape != EXPECTED_SHAPES[key] or dtype != "F32":
                raise ValueError(f"unexpected {key}: shape={shape} dtype={dtype}")
            source = np.ascontiguousarray(handle.get_tensor(key))
            stored = np.ascontiguousarray(source.astype(np.float16))
            if not np.isfinite(source).all() or not np.isfinite(stored).all():
                raise ValueError(f"non-finite edge tensor after f16 rounding: {key}")
            values[key] = stored
            tensors[key] = {
                "shape": shape, "source_dtype": dtype, "storage_dtype": "F16",
                "layout": LAYOUTS[key], "contiguous_order": "C",
                "source_nbytes": int(source.nbytes), "storage_nbytes": int(stored.nbytes),
                "source_raw_sha256": raw_sha(source), "storage_raw_sha256": raw_sha(stored),
                "source_max_abs": float(np.abs(source).max()),
                "storage_max_abs": float(np.abs(stored).max()),
            }
    return values, {
        "checkpoint_sha256": sha(path), "all_key_count": len(all_keys),
        "selected_keys": list(SELECTED_KEYS), "selected_metadata": tensors,
        "selected_source_bytes": sum(x["source_nbytes"] for x in tensors.values()),
        "selected_storage_bytes": sum(x["storage_nbytes"] for x in tensors.values()),
        "unselected_key_count": len(all_keys) - len(SELECTED_KEYS),
        "value_load_mode": "safe_open explicit non-block allow-list",
    }


def fixed_embedding_f16():
    """Pinned 1000-step source embedding: F64 phase/trig, then F16 storage."""
    inv = np.power(10000.0, -np.arange(FREQ // 2, dtype=np.float64) / (FREQ // 2))
    phase = np.outer(np.asarray([FIXED_TIMESTEP], dtype=np.float64), inv)
    embedding = np.concatenate((np.cos(phase), np.sin(phase)), axis=1).astype(np.float16)
    if not np.isfinite(embedding).all():
        raise ValueError("non-finite fixed sinusoidal embedding")
    return np.ascontiguousarray(embedding)


def _const(ops, value, dtype=None):
    return ops.constant(np.asarray(value, dtype=dtype))


def _reshape(ops, value, shape):
    return ops.reshape(value, _const(ops, shape, np.int64), False)


def _linear(ops, value, weights, wkey, bkey, bias_shape):
    result = ops.matmul(value, _const(ops, weights[wkey]), False, True)
    return ops.add(result, _reshape(ops, _const(ops, weights[bkey]), bias_shape))


def _silu(ops, value):
    return ops.multiply(value, ops.sigmoid(value))


def _layer_norm_f16(ov, ops, value):
    # Source eps is 1e-6; reductions are deliberately f32, with an explicit
    # f16 storage boundary before the f16 learned head projection.
    x = ops.convert(value, ov.Type.f32)
    axes = _const(ops, [-1], np.int64)
    mean = ops.reduce_mean(x, axes, True)
    centered = ops.subtract(x, mean)
    variance = ops.reduce_mean(ops.multiply(centered, centered), axes, True)
    normalized = ops.divide(centered, ops.sqrt(ops.add(variance, _const(ops, EPS, np.float32))))
    return ops.convert(normalized, ov.Type.f16)


def _slice(ops, value, begin, end):
    n = len(begin)
    return ops.strided_slice(value, _const(ops, begin, np.int64), _const(ops, end, np.int64),
                             _const(ops, [1] * n, np.int64), [0] * n, [0] * n)


def build_prefix(weights, frames: int, name: str, embedding_f16=None):
    import openvino as ov
    from openvino import opset13 as ops

    x = ops.parameter([1, IN_DIM, frames, 16, 16], ov.Type.f16, name=f"{name}_x")
    embedding_f16 = fixed_embedding_f16() if embedding_f16 is None else np.asarray(embedding_f16, dtype=np.float16)
    embedding = _const(ops, embedding_f16)
    patch = ops.convolution(x, _const(ops, weights["patch_embedding.weight"]), [1, 2, 2],
                            [0, 0, 0], [0, 0, 0], [1, 1, 1])
    patch = ops.add(patch, _reshape(ops, _const(ops, weights["patch_embedding.bias"]), [1, -1, 1, 1, 1]))
    tokens = _reshape(ops, ops.transpose(patch, _const(ops, [0, 2, 3, 4, 1], np.int64)),
                      [1, frames * 8 * 8, DIM])
    t = _linear(ops, _silu(ops, _linear(ops, embedding, weights,
                                         "time_embedding.0.weight", "time_embedding.0.bias", [1, -1])),
                weights, "time_embedding.2.weight", "time_embedding.2.bias", [1, -1])
    t = _reshape(ops, t, [1, 1, DIM])
    t_mod = _reshape(ops, _linear(ops, _silu(ops, _reshape(ops, t, [1, DIM])),
                                  weights, "time_projection.1.weight", "time_projection.1.bias", [1, -1]),
                     [1, 6, DIM])
    for value, output_name in ((tokens, f"{name}_tokens_out"), (t, f"{name}_t_out"),
                               (t_mod, f"{name}_t_mod_out"), (embedding, f"{name}_embedding_out")):
        value.output(0).get_tensor().set_names({output_name})
    return ov.Model([tokens, t, t_mod, embedding], [x], f"wan_{name}_prefix_f16")


def build_suffix(weights, frames: int, name: str):
    import openvino as ov
    from openvino import opset13 as ops

    tokens = ops.parameter([1, frames * 8 * 8, DIM], ov.Type.f16, name=f"{name}_tokens_in")
    t = ops.parameter([1, 1, DIM], ov.Type.f16, name=f"{name}_t_in")
    modulation = ops.add(_const(ops, weights["head.modulation"]), t)
    shift = _slice(ops, modulation, [0, 0, 0], [1, 1, DIM])
    scale = _slice(ops, modulation, [0, 1, 0], [1, 2, DIM])
    normalized = _layer_norm_f16(ov, ops, tokens)
    modulated = ops.add(ops.multiply(normalized, ops.add(_const(ops, np.float16(1.0)), scale)), shift)
    projected = _linear(ops, modulated, weights, "head.head.weight", "head.head.bias", [1, 1, -1])
    grid = _reshape(ops, projected, [1, frames, 8, 8, 1, 2, 2, OUT_DIM])
    output = _reshape(ops, ops.transpose(grid, _const(ops, [0, 7, 1, 4, 2, 5, 3, 6], np.int64)),
                      [1, OUT_DIM, frames, 16, 16])
    output.output(0).get_tensor().set_names({f"{name}_out"})
    return ov.Model([output], [tokens, t], f"wan_{name}_suffix_f16")


def build_all(weights, embedding_f16=None):
    return {
        f"{name}_{part}": graph
        for name, frames in (("startup_f6_128x128", 6), ("steady_f2_128x128", 2))
        for part, graph in (("prefix", build_prefix(weights, frames, name, embedding_f16)),
                            ("suffix", build_suffix(weights, frames, name)))
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import openvino as ov
    args.output.mkdir(parents=True, exist_ok=False)
    values, report = load_weights()
    for key, model in build_all(values).items():
        ov.save_model(model, args.output / f"{key}.xml", compress_to_fp16=False)
    (args.output / "selection.json").write_text(__import__("json").dumps(report, indent=2) + "\n")
