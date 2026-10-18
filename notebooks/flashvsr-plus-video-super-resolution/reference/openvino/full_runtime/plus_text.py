"""FlashVSR+ text embedding and per-layer cross-cache graphs.

The pinned source uses ``Linear -> GELU(approximate='tanh') -> Linear`` for
the context embedding.  Cross K/V uses two native linears; K then follows the
source RMSNorm boundary (F32 reduction, conversion back to the requested
storage type, and learned scale).  Learned tensors remain source F32 until
``plus_storage.storage_tensor`` performs the one requested BF16/F16 rounding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import plus_weights
from plus_storage import storage_tensor, tensor_array
from reference_paths import PROMPT_PATH


PROMPT_BYTES = 4_195_504
PROMPT_SHA256 = "4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef"
PROMPT_SHAPE = (1, 512, 4096)
TEXT_SHAPE = (1, 512, 1536)
EPS = np.float32(1e-6)
STORAGE_PRECISIONS = ("bf16", "f16")

_CROSS_KEYS = plus_weights.EXTRA_LAYER_KEYS
_CROSS_SHAPES = {key: plus_weights.LAYER_SHAPES[key] for key in _CROSS_KEYS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_prompt(path: Path = PROMPT_PATH) -> dict[str, Any]:
    """Authenticate the official prompt container without importing Torch."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != PROMPT_BYTES:
        raise ValueError(f"unexpected Plus prompt size: {size} != {PROMPT_BYTES}")
    digest = _sha256(path)
    if digest != PROMPT_SHA256:
        raise ValueError(f"unexpected Plus prompt hash: {digest}")
    return {
        "path": str(path.resolve()),
        "bytes": int(size),
        "sha256": digest,
        "expected_sha256": PROMPT_SHA256,
        "shape": list(PROMPT_SHAPE),
        "source_dtype": "BF16",
    }


def load_prompt(path: Path = PROMPT_PATH) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an owning F32 prompt array and its authenticated receipt.

    The official prompt is a BF16 Torch tensor inside a small ``.pth``
    container.  This narrowly scoped converter is the only Torch dependency in
    this module; it performs BF16 -> F32 directly and never routes through F16.
    """

    receipt = authenticate_prompt(path)
    import torch

    value = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != PROMPT_SHAPE:
        raise ValueError(f"Plus prompt ABI changed: expected Tensor {PROMPT_SHAPE}")
    if value.dtype != torch.bfloat16 or value.device.type != "cpu":
        raise ValueError(f"Plus prompt ABI changed: expected CPU BF16, got {value.dtype} {value.device}")
    array = np.array(value.detach().float().numpy(), dtype=np.float32, order="C", copy=True)
    if array.shape != PROMPT_SHAPE or not np.isfinite(array).all():
        raise ValueError("Plus prompt conversion produced invalid F32 values")
    receipt = {
        **receipt,
        "array_dtype": "float32",
        "conversion": "authenticated Torch BF16 -> owning NumPy F32; no F16 intermediate",
    }
    return array, receipt


def _precision_type(precision: str):
    import openvino as ov

    if precision not in STORAGE_PRECISIONS:
        raise ValueError("Plus text/cache storage must be 'bf16' or 'f16'")
    return ov.Type.bf16 if precision == "bf16" else ov.Type.f16


def _validate_values(values: Mapping[str, Any], expected: Mapping[str, tuple[int, ...]], role: str) -> dict[str, np.ndarray]:
    if set(values) != set(expected):
        missing = sorted(set(expected) - set(values))
        extra = sorted(set(values) - set(expected))
        raise ValueError(f"{role} weight key mismatch: missing={missing}, extra={extra}")
    checked: dict[str, np.ndarray] = {}
    for key, shape in expected.items():
        value = np.asarray(values[key])
        if value.shape != shape or value.dtype != np.dtype(np.float32):
            raise ValueError(f"unexpected {role} weight {key}: shape={value.shape} dtype={value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite {role} weight: {key}")
        checked[key] = np.array(value, dtype=np.float32, order="C", copy=True)
    return checked


def load_text_weights(handle: Any) -> dict[str, np.ndarray]:
    """Read only the four text MLP tensors from an authenticated handle."""

    plus_weights.validate_schema(handle)
    values: dict[str, np.ndarray] = {}
    for key in plus_weights.TEXT_KEYS:
        shape = plus_weights.TEXT_SHAPES[key]
        view = handle.get_slice(key)
        if tuple(view.get_shape()) != shape or str(view.get_dtype()) != "F32":
            raise ValueError(f"unexpected Plus text header {key}")
        value = np.asarray(handle.get_tensor(key))
        if value.shape != shape or value.dtype != np.dtype(np.float32):
            raise ValueError(f"unexpected Plus text tensor {key}: {value.shape} {value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite Plus text tensor: {key}")
        values[key] = np.array(value, dtype=np.float32, order="C", copy=True)
    return values


def _port(key: str) -> str:
    return "weight__" + key.replace(".", "_")


def _constant(value: np.ndarray, precision: str):
    import openvino as ov
    from openvino import opset13 as ops

    return ops.constant(storage_tensor(value, precision))


def _linear(value, weights: Mapping[str, Any], prefix: str, precision: str, ops):
    weight = weights[f"{prefix}.weight"]
    bias = weights[f"{prefix}.bias"]
    # External-weight graphs pass OpenVINO nodes; standalone graphs pass
    # validated source arrays.  Only arrays cross the storage conversion.
    if isinstance(weight, np.ndarray):
        weight = _constant(weight, precision)
    if isinstance(bias, np.ndarray):
        bias = _constant(bias, precision)
    product = ops.matmul(
        value,
        weight,
        False,
        True,
    )
    return ops.add(product, bias)


def build_text_embedding(
    weights: Mapping[str, Any] | None,
    precision: str = "bf16",
    *,
    preserve_storage: bool = True,
):
    """Build ``prompt -> context`` with fixed [1,512] Plus geometry.

    ``weights=None`` is the build-only native ABI: the four learned text MLP
    tensors become BF16 Parameters named ``weight__...`` instead of embedded
    Constants.  The historical mapping path remains unchanged for standalone
    source comparisons, including its selectable BF16/F16 storage precision.
    """

    checked: dict[str, Any] = {}
    if weights is None:
        if precision != "bf16":
            raise ValueError("external Plus text weights require BF16 Parameters")
    else:
        checked = _validate_values(weights, plus_weights.TEXT_SHAPES, "text")
    storage = _precision_type(precision)
    import openvino as ov
    from openvino import opset13 as ops

    prompt = ops.parameter(list(PROMPT_SHAPE), storage, name="prompt")
    prompt.output(0).get_tensor().set_names({"prompt"})
    inputs = [prompt]
    if weights is None:
        for key, shape in plus_weights.TEXT_SHAPES.items():
            port = _port(key)
            parameter = ops.parameter(list(shape), storage, name=port)
            parameter.output(0).get_tensor().set_names({port})
            checked[key] = parameter
            inputs.append(parameter)
    hidden = _linear(prompt, checked, "text_embedding.0", precision, ops)
    activation = ops.gelu(hidden, "tanh")
    context = _linear(activation, checked, "text_embedding.2", precision, ops)
    context.output(0).get_tensor().set_names({"context"})
    outputs = [context]
    if preserve_storage:
        # Source Linear rounds before GELU. An observable store prevents the
        # GPU/CPU plugins from fusing across that BF16/F16 boundary.
        hidden.output(0).get_tensor().set_names({"text_linear"})
        outputs.append(hidden)
    model = ov.Model(outputs, inputs, f"flashvsr_plus_text_{precision}")
    model.validate_nodes_and_infer_types()
    return model


def _cross_values(weights: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if set(weights) == set(plus_weights.LAYER_KEYS):
        weights = {key: weights[key] for key in _CROSS_KEYS}
    return _validate_values(weights, _CROSS_SHAPES, "cross")


def build_cross_kv(weights: Mapping[str, Any] | None = None, precision: str = "bf16"):
    """Build one reusable context->cross-K/V graph.

    With ``weights=None`` (the integration path), the five per-layer learned
    tensors are Parameters named by the canonical layer-0 ABI, so one compiled
    graph can serve all 30 layers.  Supplying a mapping embeds validated
    constants and is useful for a small standalone CPU comparison.
    """

    storage = _precision_type(precision)
    import openvino as ov
    from openvino import opset13 as ops

    context = ops.parameter(list(TEXT_SHAPE), storage, name="context")
    context.output(0).get_tensor().set_names({"context"})
    params: dict[str, Any] = {}
    inputs = [context]
    if weights is None:
        for key, shape in _CROSS_SHAPES.items():
            parameter = ops.parameter(list(shape), storage, name=_port(key))
            parameter.output(0).get_tensor().set_names({_port(key)})
            params[key] = parameter
            inputs.append(parameter)
    else:
        params = _cross_values(weights)

    k = _linear(context, params, "blocks.0.cross_attn.k", precision, ops)
    k32 = ops.convert(k, ov.Type.f32)
    axes = ops.constant(np.asarray([-1], dtype=np.int64))
    mean_square = ops.reduce_mean(ops.multiply(k32, k32), axes, True)
    denominator = ops.sqrt(ops.add(mean_square, ops.constant(EPS)))
    normalized = ops.convert(ops.divide(k32, denominator), storage)
    if weights is None:
        norm_k = params["blocks.0.cross_attn.norm_k.weight"]
    else:
        norm_k = _constant(params["blocks.0.cross_attn.norm_k.weight"], precision)
    cross_k = ops.multiply(normalized, norm_k)
    v = _linear(context, params, "blocks.0.cross_attn.v", precision, ops)
    cross_k.output(0).get_tensor().set_names({"cross_k"})
    v.output(0).get_tensor().set_names({"cross_v"})
    model = ov.Model([cross_k, v], inputs, f"flashvsr_plus_cross_kv_{precision}")
    model.validate_nodes_and_infer_types()
    return model


def storage_array(tensor) -> np.ndarray:
    """Decode an OpenVINO output tensor to an owning F32 diagnostic array."""

    return np.ascontiguousarray(tensor_array(tensor), dtype=np.float32)
