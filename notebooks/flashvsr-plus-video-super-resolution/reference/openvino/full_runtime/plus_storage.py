"""Single source-F32 conversion to Plus storage, with BF16 RNE bit copies."""
import numpy as np


def is_finite_bf16(bits):
    """Check raw BF16 words in bounded chunks without decoding them to F32."""
    if (not isinstance(bits, np.ndarray) or bits.dtype != np.uint16
            or not bits.flags.c_contiguous):
        raise ValueError("Contiguous uint16 BF16 storage required")
    flat = bits.reshape(-1)
    for start in range(0, flat.size, 1 << 20):
        # Removing the sign makes every Inf/NaN >= 0x7f80. A max reduction
        # avoids allocating both a masked array and a boolean comparison.
        if np.max(flat[start:start + (1 << 20)] & np.uint16(0x7fff)) >= 0x7f80:
            return False
    return True


def storage_tensor(value, precision):
    import openvino as ov
    if precision not in ("bf16", "f16"):
        raise ValueError("Explicit BF16 or F16 storage required")
    value = np.asarray(value, dtype=np.float32, order="C")
    if precision == "f16" and not np.isfinite(value).all():
        raise ValueError("Finite source values required")
    kind = ov.Type.bf16 if precision == "bf16" else ov.Type.f16
    tensor = ov.Tensor(kind, value.shape)
    if precision == "bf16":
        source = value.reshape(-1)
        # Keep rounding scratch bounded even for a full UHD token array.
        # This owning Tensor is published only after every chunk validates.
        destination = tensor.data.view(np.uint16).reshape(-1)
        for start in range(0, source.size, 1 << 20):
            part = source[start:start + (1 << 20)]
            if not np.isfinite(part).all():
                raise ValueError("Finite source values required")
            bits = part.view(np.uint32)
            rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            if not is_finite_bf16(rounded):
                raise ValueError("BF16 storage overflow")
            # Tensor.data exposes BF16 as F16: copy bits, never F16 values.
            destination[start:start + part.size] = rounded
    else:
        with np.errstate(over="ignore"):
            rounded = value.astype(np.float16)
        if not np.isfinite(rounded).all():
            raise ValueError("F16 storage overflow")
        tensor.data[...] = rounded
    return tensor


def tensor_array(tensor):
    """Decode an owning F32 array without a Torch runtime dependency."""
    import openvino as ov
    if tensor.element_type == ov.Type.bf16:
        words = tensor.data.view(np.uint16).astype(np.uint32)
        words <<= np.uint32(16)
        return words.view(np.float32)
    return np.array(tensor.data, dtype=np.float32, copy=True)
