"""FlashVSR+ f4 prefix and suffix edge graphs.

The edge graphs contain only the source patch/time/head operations.  Their
learned tensors stay in source F32 until OpenVINO creates the requested
storage constants; in particular BF16 is never quantized through FP16.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from plus_modes import SOURCE_COMMIT, resolve_plus_mode
from reference_paths import DIT_CHECKPOINT

CHECKPOINT = DIT_CHECKPOINT
CHECKPOINT_BYTES = resolve_plus_mode().dit_bytes
CHECKPOINT_SHA256 = resolve_plus_mode().dit_sha256

DIM = 1536
FREQ_DIM = 256
IN_CHANNELS = 16
OUT_CHANNELS = 16
PATCH = (1, 2, 2)
EPS = 1e-6
FIXED_TIMESTEP = np.float64(1000.0)

SELECTED_KEYS = (
    "patch_embedding.weight",
    "patch_embedding.bias",
    "time_embedding.0.weight",
    "time_embedding.0.bias",
    "time_embedding.2.weight",
    "time_embedding.2.bias",
    "time_projection.1.weight",
    "time_projection.1.bias",
    "head.modulation",
    "head.proj.weight",
    "head.proj.bias",
)
WEIGHT_PORTS = {key: "weight__" + key.replace(".", "_") for key in SELECTED_KEYS}
PREFIX_WEIGHT_KEYS = ("patch_embedding.weight", "patch_embedding.bias")
SUFFIX_WEIGHT_KEYS = ("head.modulation", "head.proj.weight", "head.proj.bias")
EXPECTED_SHAPES = {
    "patch_embedding.weight": (1536, 16, 1, 2, 2),
    "patch_embedding.bias": (1536,),
    "time_embedding.0.weight": (1536, 256),
    "time_embedding.0.bias": (1536,),
    "time_embedding.2.weight": (1536, 1536),
    "time_embedding.2.bias": (1536,),
    "time_projection.1.weight": (9216, 1536),
    "time_projection.1.bias": (9216,),
    "head.modulation": (1, 2, 1536),
    "head.proj.weight": (64, 1536),
    "head.proj.bias": (64,),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def fixed_embedding_f32() -> np.ndarray:
    """Return the source F64 sinusoid rounded once to source F32."""

    inverse = np.power(
        10000.0,
        -np.arange(FREQ_DIM // 2, dtype=np.float64) / (FREQ_DIM // 2),
    )
    phase = np.outer(np.asarray([FIXED_TIMESTEP], dtype=np.float64), inverse)
    embedding = np.concatenate((np.cos(phase), np.sin(phase)), axis=1).astype(np.float32)
    if embedding.shape != (1, FREQ_DIM) or not np.isfinite(embedding).all():
        raise ValueError("fixed timestep embedding is invalid")
    return np.ascontiguousarray(embedding)


def validate_edge_weights(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate the exact non-block source edge allow-list without casting it."""

    if not isinstance(weights, dict):
        raise ValueError("edge weights must be a mapping")
    if set(weights) != set(SELECTED_KEYS):
        missing = sorted(set(SELECTED_KEYS) - set(weights))
        extra = sorted(set(weights) - set(SELECTED_KEYS))
        raise ValueError(f"edge key set mismatch: missing={missing}, extra={extra}")
    result = {}
    for key in SELECTED_KEYS:
        value = np.asarray(weights[key])
        if value.shape != EXPECTED_SHAPES[key] or value.dtype != np.dtype(np.float32):
            raise ValueError(
                f"unexpected {key}: shape={value.shape} dtype={value.dtype}; source F32 required"
            )
        value = np.ascontiguousarray(value)
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite source edge tensor: {key}")
        result[key] = value
    return result


def load_edge_weights(
    path: Path = CHECKPOINT, *, verify_hash: bool = True, mode: str = "turbo"
) -> tuple[dict[str, np.ndarray], dict]:
    """Read only the 11 edge tensors from the explicitly selected checkpoint."""

    spec = resolve_plus_mode(mode)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != spec.dit_bytes:
        raise ValueError(f"unexpected {mode} checkpoint size: {size} != {spec.dit_bytes}")
    checkpoint_sha = _sha256(path) if verify_hash else None
    if verify_hash and checkpoint_sha != spec.dit_sha256:
        raise ValueError(f"unexpected {mode} checkpoint hash: {checkpoint_sha}")

    from safetensors import safe_open

    values: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    with safe_open(str(path), framework="np", device="cpu") as handle:
        all_keys = list(handle.keys())
        missing = sorted(set(SELECTED_KEYS) - set(all_keys))
        if missing:
            raise KeyError(f"missing edge checkpoint keys: {missing}")
        for key in SELECTED_KEYS:
            view = handle.get_slice(key)
            shape, dtype = tuple(view.get_shape()), view.get_dtype()
            if shape != EXPECTED_SHAPES[key] or dtype != "F32":
                raise ValueError(f"unexpected {key}: shape={shape} dtype={dtype}")
            value = np.ascontiguousarray(handle.get_tensor(key))
            if value.shape != EXPECTED_SHAPES[key] or value.dtype != np.dtype(np.float32):
                raise ValueError(f"unexpected materialized {key}: {value.shape} {value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite source edge tensor: {key}")
            values[key] = value
            metadata[key] = {
                "shape": list(value.shape),
                "source_dtype": "F32",
                "source_nbytes": int(value.nbytes),
                "source_raw_sha256": _raw_sha256(value),
            }
    # The per-tensor checks above intentionally avoid materializing the other
    # 814 tensors.  Validate the completed allow-list once the loop is done.
    values = validate_edge_weights(values)
    return values, {
        "source_commit": SOURCE_COMMIT,
        "checkpoint": str(path),
        "checkpoint_bytes": int(size),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha256_expected": spec.dit_sha256,
        "mode": mode,
        "all_key_count": len(all_keys),
        "selected_keys": list(SELECTED_KEYS),
        "selected_metadata": metadata,
        "value_load_mode": "safe_open explicit non-block edge allow-list",
    }


def _geometry(geometry):
    try:
        frames, height, width = geometry.f, geometry.h, geometry.w
    except AttributeError as exc:
        raise TypeError("FlashVSR+ edge geometry must expose f, h, and w") from exc
    if type(frames) is not int or frames != 4:
        raise ValueError("FlashVSR+ edges require exactly four latent frames")
    if any(type(value) is not int or value <= 0 for value in (height, width)):
        raise ValueError("FlashVSR+ edge token geometry must be positive integers")
    if height % 8 or width % 8:
        raise ValueError("FlashVSR+ edge geometry must contain complete 8x8 windows")
    return frames, height, width


def edge_shapes(geometry) -> dict[str, tuple[int, ...]]:
    frames, height, width = _geometry(geometry)
    tokens = frames * height * width
    latent = (1, IN_CHANNELS, frames, height * 2, width * 2)
    return {
        "latent": latent,
        "tokens": (1, tokens, DIM),
        "t": (1, 1, DIM),
        "t_mod": (1, 6, DIM),
        "embedding": (1, FREQ_DIM),
        "output": latent,
    }


def _storage(storage):
    import openvino as ov

    if storage == "bf16":
        return storage, ov.Type.bf16
    if storage == "f16":
        return storage, ov.Type.f16
    raise ValueError("explicit FlashVSR+ edge storage must be 'bf16' or 'f16'")


def _const(ops, value, dtype=None):
    return ops.constant(np.asarray(value, dtype=dtype))


def _storage_const(ops, value, dtype):
    import openvino as ov
    from plus_storage import storage_tensor
    # ops.constant(F32, BF16) truncates in this runtime. The source loader
    # rounds to nearest even; create exact storage bits before the Constant.
    return ops.constant(storage_tensor(value, "bf16" if dtype == ov.Type.bf16 else "f16"))


def _learned(ops, weights, parameters, key, storage_type):
    """Return one historical Constant or one external BF16 Parameter."""

    if weights is None:
        return parameters[key]
    return _storage_const(ops, weights[key], storage_type)


def _external_parameters(ops, keys, storage_type):
    import openvino as ov

    if storage_type != ov.Type.bf16:
        raise ValueError("external edge weights require BF16 Parameters")
    parameters = {}
    for key in keys:
        port_name = WEIGHT_PORTS[key]
        parameter = ops.parameter(EXPECTED_SHAPES[key], storage_type, name=port_name)
        parameter.output(0).get_tensor().set_names({port_name})
        parameters[key] = parameter.output(0)
    return parameters


def _external_embedding(ops, embedding_f32, storage_type):
    if embedding_f32 is None:
        embedding_f32 = fixed_embedding_f32()
    embedding_f32 = np.asarray(embedding_f32)
    if embedding_f32.shape != (1, FREQ_DIM) or embedding_f32.dtype != np.dtype(np.float32):
        raise ValueError("embedding_f32 must be a [1,256] source-F32 array")
    if not np.isfinite(embedding_f32).all():
        raise ValueError("embedding_f32 must be finite")
    return _storage_const(ops, embedding_f32, storage_type)


def _reshape(ops, value, shape):
    return ops.reshape(value, _const(ops, shape, np.int64), False)


def _linear(ops, value, weight, bias, storage_type, bias_shape):
    result = ops.matmul(
        value,
        weight,
        False,
        True,
    )
    return ops.add(result, _reshape(ops, bias, bias_shape))


def fixed_conditioning(weights, storage, embedding_f32=None):
    """Compute fixed timestep1000 once, honoring fused Linear/SiLU rounding.

    Separate BF16 MatMul/Add or Sigmoid/Multiply graph nodes round an extra
    time compared with the source fused operators. These tiny constant paths
    use FP32 accumulation and round once at each source operator boundary.
    """
    from plus_storage import storage_tensor, tensor_array
    embedding_f32 = fixed_embedding_f32() if embedding_f32 is None else np.asarray(embedding_f32)
    if embedding_f32.shape != (1, FREQ_DIM) or embedding_f32.dtype != np.dtype(np.float32):
        raise ValueError("embedding_f32 must be a [1,256] source-F32 array")
    def cast(value):
        return tensor_array(storage_tensor(value, storage))
    def linear(value, prefix):
        return cast(value @ cast(weights[prefix + ".weight"]).T + cast(weights[prefix + ".bias"]))
    def silu(value):
        with np.errstate(over="ignore"):
            return cast(value / (1 + np.exp(-value)))
    embedding = cast(embedding_f32)
    t = linear(silu(linear(embedding, "time_embedding.0")), "time_embedding.2")
    modulation = linear(silu(t), "time_projection.1")
    return t.reshape(1, 1, DIM), modulation.reshape(1, 6, DIM), embedding


def _slice(ops, value, begin, end):
    count = len(begin)
    return ops.strided_slice(
        value,
        _const(ops, begin, np.int64),
        _const(ops, end, np.int64),
        _const(ops, [1] * count, np.int64),
        [0] * count,
        [0] * count,
    )


def _layer_norm_f32(ov, ops, value, storage_type):
    normalized_input = ops.convert(value, ov.Type.f32)
    axes = _const(ops, [-1], np.int64)
    mean = ops.reduce_mean(normalized_input, axes, True)
    centered = ops.subtract(normalized_input, mean)
    variance = ops.reduce_mean(ops.multiply(centered, centered), axes, True)
    normalized = ops.divide(
        centered,
        ops.sqrt(ops.add(variance, _const(ops, EPS, np.float32))),
    )
    return ops.convert(normalized, storage_type)


def build_prefix(
    weights: dict[str, np.ndarray] | None, geometry, *, storage: str = "bf16", embedding_f32=None
):
    """Build patchify/time embedding for a four-latent Plus block.

    ``weights=None`` is an explicit build-only external-weight mode.  It is
    BF16-only and exposes reachable patch weights as canonical
    ``weight__...`` Parameters.  The fixed timestep ``t`` and ``t_mod`` are
    BF16 inputs because their source F32/time-weight preparation is performed
    once outside this graph (call :func:`fixed_conditioning` with the
    authenticated source-F32 edge weights and bind its first two results); the
    deterministic sinusoid remains a graph constant.  The checked mapping path
    keeps its historical constants and NumPy fused-boundary conditioning
    unchanged.
    """

    import openvino as ov
    from openvino import opset13 as ops

    if weights is not None:
        weights = validate_edge_weights(weights)
    frames, height, width = _geometry(geometry)
    _, storage_type = _storage(storage)
    shapes = edge_shapes(geometry)
    name = getattr(geometry, "name", f"plus_f4_{width}x{height}")

    x = ops.parameter(shapes["latent"], storage_type, name=f"{name}_x")
    parameters = [x]
    learned_parameters = _external_parameters(ops, PREFIX_WEIGHT_KEYS, storage_type) \
        if weights is None else {}
    if weights is None:
        parameters.extend(parameter.get_node() for parameter in learned_parameters.values())
    patch = ops.convolution(
        x,
        _learned(ops, weights, learned_parameters, "patch_embedding.weight", storage_type),
        [1, 2, 2],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
    )
    patch = ops.add(
        patch,
        _reshape(
            ops,
            _learned(ops, weights, learned_parameters, "patch_embedding.bias", storage_type),
            [1, -1, 1, 1, 1],
        ),
    )
    tokens = _reshape(
        ops,
        ops.transpose(patch, _const(ops, [0, 2, 3, 4, 1], np.int64)),
        [1, frames * height * width, DIM],
    )

    if weights is None:
        t_input = ops.parameter(shapes["t"], storage_type, name="t")
        t_mod_input = ops.parameter(shapes["t_mod"], storage_type, name="t_mod")
        parameters.extend([t_input, t_mod_input])
        # Keep stable input names while exposing the same-type values under the
        # geometry-qualified output ABI.  Reshape is a view here; it adds no
        # numeric conversion or learned payload.
        t = _reshape(ops, t_input, shapes["t"])
        t_mod = _reshape(ops, t_mod_input, shapes["t_mod"])
        embedding = _external_embedding(ops, embedding_f32, storage_type)
    else:
        time_value, modulation, embedding_value = fixed_conditioning(weights, storage, embedding_f32)
        t, t_mod, embedding = (
            _storage_const(ops, value, storage_type)
            for value in (time_value, modulation, embedding_value)
        )
    for value, output_name in (
        (tokens, f"{name}_tokens_out"),
        (t, f"{name}_t_out"),
        (t_mod, f"{name}_t_mod_out"),
        (embedding, f"{name}_embedding_out"),
    ):
        value.output(0).get_tensor().set_names({output_name})
    model = ov.Model([tokens, t, t_mod, embedding], parameters,
                     f"flashvsr_plus_{name}_prefix_{storage}")
    model.validate_nodes_and_infer_types()
    return model


def build_suffix(weights: dict[str, np.ndarray] | None, geometry, *, storage: str = "bf16"):
    """Build the source output head and f4 unpatchify operation.

    ``weights=None`` exposes only the three reachable head weights as BF16
    Parameters; the input token/timestep ABI is otherwise unchanged.  This
    path carries no learned payload and rejects F16 because native external
    edge binding is intentionally BF16-only.
    """

    import openvino as ov
    from openvino import opset13 as ops

    if weights is not None:
        weights = validate_edge_weights(weights)
    frames, height, width = _geometry(geometry)
    _, storage_type = _storage(storage)
    shapes = edge_shapes(geometry)
    name = getattr(geometry, "name", f"plus_f4_{width}x{height}")

    tokens = ops.parameter(shapes["tokens"], storage_type, name=f"{name}_tokens_in")
    t = ops.parameter(shapes["t"], storage_type, name=f"{name}_t_in")
    parameters = [tokens, t]
    learned_parameters = _external_parameters(ops, SUFFIX_WEIGHT_KEYS, storage_type) \
        if weights is None else {}
    if weights is None:
        parameters.extend(parameter.get_node() for parameter in learned_parameters.values())
    modulation = ops.add(
        _learned(ops, weights, learned_parameters, "head.modulation", storage_type), t
    )
    shift = _slice(ops, modulation, [0, 0, 0], [1, 1, DIM])
    scale = _slice(ops, modulation, [0, 1, 0], [1, 2, DIM])
    normalized = _layer_norm_f32(ov, ops, tokens, storage_type)
    modulated = ops.add(
        ops.multiply(
            normalized,
            ops.add(_storage_const(ops, np.asarray(1.0, np.float32), storage_type), scale),
        ),
        shift,
    )
    projected = _linear(
        ops,
        modulated,
        _learned(ops, weights, learned_parameters, "head.proj.weight", storage_type),
        _learned(ops, weights, learned_parameters, "head.proj.bias", storage_type),
        storage_type,
        [1, 1, -1],
    )
    grid = _reshape(ops, projected, [1, frames, height, width, 1, 2, 2, OUT_CHANNELS])
    output = _reshape(
        ops,
        ops.transpose(grid, _const(ops, [0, 7, 1, 4, 2, 5, 3, 6], np.int64)),
        list(shapes["output"]),
    )
    output.output(0).get_tensor().set_names({f"{name}_out"})
    model = ov.Model([output], parameters, f"flashvsr_plus_{name}_suffix_{storage}")
    model.validate_nodes_and_infer_types()
    return model


__all__ = [
    "CHECKPOINT",
    "CHECKPOINT_BYTES",
    "CHECKPOINT_SHA256",
    "EXPECTED_SHAPES",
    "FIXED_TIMESTEP",
    "SELECTED_KEYS",
    "PREFIX_WEIGHT_KEYS",
    "SUFFIX_WEIGHT_KEYS",
    "WEIGHT_PORTS",
    "SOURCE_COMMIT",
    "build_prefix",
    "build_suffix",
    "edge_shapes",
    "fixed_embedding_f32",
    "load_edge_weights",
    "validate_edge_weights",
]
