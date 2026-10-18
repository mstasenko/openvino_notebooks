"""Typed diagnostic execution and stream layouts for the Plus integration.

This initial full-trajectory check deliberately reads back small tensors to
localize discrepancies. It is not the resident UHD throughput implementation.
"""
from pathlib import Path
import time

import numpy as np

from plus_storage import storage_tensor, tensor_array
from window_pre import window_indices


def compile_config(ov, device):
    config = {"NUM_STREAMS": ov.properties.streams.Num(1), "CACHE_DIR": "",
              "EXECUTION_MODE_HINT": ov.properties.hint.ExecutionMode.ACCURACY}
    if device == "CPU":
        config["INFERENCE_NUM_THREADS"] = 2
    else:
        config["PERFORMANCE_HINT"] = ov.properties.hint.PerformanceMode.LATENCY
    return config


class Graph:
    """One compiled request with owning, explicitly typed input/output copies."""
    def __init__(self, core, model, device):
        import openvino as ov
        start = time.perf_counter()
        self.compiled = core.compile_model(model, device, compile_config(ov, device))
        self.compile_seconds = time.perf_counter() - start
        if list(self.compiled.get_property("EXECUTION_DEVICES")) != [device]:
            raise ValueError("Plus graph executed on an unintended device")
        self.request = self.compiled.create_infer_request()
        self.fixed = {}

    def bind(self, values):
        self.fixed = {name: self._tensor(self.compiled.input(name), value)
                      for name, value in values.items()}
        for name, tensor in self.fixed.items():
            self.request.set_tensor(name, tensor)

    @staticmethod
    def _tensor(port, value):
        import openvino as ov
        value = np.asarray(value)
        if tuple(value.shape) != tuple(port.get_shape()) or not np.isfinite(value).all():
            raise ValueError("Invalid graph feed: " + port.get_any_name())
        kind = port.get_element_type()
        if kind in (ov.Type.bf16, ov.Type.f16, ov.Type.f32, ov.Type.f64) and value.dtype.kind != "f":
            raise ValueError("Logical float values required; raw storage bits are not numeric feeds")
        if kind in (ov.Type.bf16, ov.Type.f16):
            return storage_tensor(value, "bf16" if kind == ov.Type.bf16 else "f16")
        return ov.Tensor(np.asarray(value, dtype=kind.to_dtype()), shared_memory=False)

    def __call__(self, values):
        names = {port.get_any_name() for port in self.compiled.inputs}
        if set(values) & set(self.fixed) or set(values) | set(self.fixed) != names:
            raise ValueError("Plus graph feeds do not match the complete input ABI")
        tensors = {name: self._tensor(self.compiled.input(name), value)
                   for name, value in values.items()}
        for name, tensor in tensors.items():
            self.request.set_tensor(name, tensor)
        self.request.infer()
        # Own every array: the next request reuses its output allocation.
        return {port.get_any_name(): tensor_array(self.request.get_output_tensor(index))
                for index, port in enumerate(self.compiled.outputs)}


def join_history(current, previous, *, spatial_windows, retain, write, raw_bf16=False):
    """Append current windows in source order; keep a contiguous suffix only."""
    if (type(spatial_windows) is not int or spatial_windows < 1
            or type(retain) is not int or retain not in (1, 3)):
        raise ValueError("Expected Plus spatial windows and local/global retention")
    if type(write) is not bool or set(current) != {"k", "v", "pooled"}:
        raise ValueError("Invalid Plus cache update")
    if type(raw_bf16) is not bool:
        raise ValueError("Explicit BF16 storage flag required")
    def valid(value, name):
        dtype = np.uint16 if raw_bf16 and name != "pooled" else np.float32
        return (value.dtype == dtype and (not np.any((value & 0x7f80) == 0x7f80)
                if dtype == np.uint16 else np.isfinite(value).all()))
    count = spatial_windows * 2
    for name, shape in (("k", (count, 128, 1536)), ("v", (count, 128, 1536)),
                        ("pooled", (12, count, 128))):
        if (not isinstance(current[name], np.ndarray) or current[name].shape != shape
                or not valid(current[name], name)):
            raise ValueError("Invalid current Plus history: " + name)
    merged = {}
    for name, value in current.items():
        axis = 1 if name == "pooled" else 0
        if previous is None:
            merged[name] = value
        else:
            old = np.asarray(previous[name])
            shape = list(value.shape)
            old_count = old.shape[axis] if old.ndim == value.ndim else 0
            shape[axis] = old_count
            if (old.shape != tuple(shape) or old_count < spatial_windows
                    or old_count > retain * spatial_windows or old_count % spatial_windows
                    or not valid(old, name)):
                raise ValueError("Invalid retained Plus history: " + name)
            merged[name] = np.concatenate((old, value), axis=axis)
    if previous is not None and (merged["k"].shape[0] != merged["v"].shape[0]
                                 or merged["k"].shape[0] != merged["pooled"].shape[1]):
        raise ValueError("Plus history lane lengths differ")
    if not write:
        return merged, None
    next_cache = {}
    for name, value in merged.items():
        axis = 1 if name == "pooled" else 0
        indices = [slice(None)] * value.ndim
        indices[axis] = slice(-retain * spatial_windows, None)
        next_cache[name] = np.array(value[tuple(indices)], copy=True)
    return merged, next_cache


def unpartition_context(value, geometry):
    """Head-major attention output -> original f4 raster token sequence."""
    f, h, w = geometry
    if f != 4 or min(h, w) < 8 or h % 8 or w % 8:
        raise ValueError("Complete Plus f4 windows required")
    windows = f * h * w // 128
    if np.shape(value) != (12, windows, 128, 128):
        raise ValueError("Unexpected Plus head-major attention output")
    partitioned = value.transpose(1, 2, 0, 3).reshape(-1, 1536)
    result = np.empty((1, f * h * w, 1536), dtype=value.dtype)
    result[0, window_indices(geometry, 0, windows)] = partitioned
    return result


def pack_decoder_inputs(condition, latents):
    """Source pixel-shuffle conditioning plus one f4 denoised chunk.

    The first decoder chunk has13 source frames; repeating its first frame
    three times matches upstream temporal padding. Later chunks have16.
    There is no image resize or latent approximation at this handoff.
    """
    condition, latents = np.asarray(condition), np.asarray(latents)
    if (condition.ndim != 5 or condition.shape[:2] != (1, 3)
            or condition.shape[2] not in (13, 16) or condition.dtype != np.float32
            or latents.ndim != 5 or latents.shape[:3] != (1, 16, 4)
            or latents.dtype != np.float32
            or not np.isfinite(condition).all() or not np.isfinite(latents).all()):
        raise ValueError("Expected finite FP32 source video and f4 latents")
    height, width = condition.shape[-2:]
    if min(height, width) < 8 or height % 8 or width % 8 or latents.shape[-2:] != (height // 8, width // 8):
        raise ValueError("Decoder conditioning and latent geometry differ")
    condition = tensor_array(storage_tensor(condition, "bf16"))
    latents = tensor_array(storage_tensor(latents, "bf16"))
    if condition.shape[2] == 13:
        condition = np.concatenate((np.repeat(condition[:, :, :1], 3, axis=2), condition), axis=2)
    shuffled = condition.reshape(1, 3, 4, 4, height // 8, 8, width // 8, 8)
    shuffled = shuffled.transpose(0, 2, 1, 3, 5, 7, 4, 6).reshape(1, 4, 768, height // 8, width // 8)
    return np.concatenate((shuffled, latents.transpose(0, 2, 1, 3, 4)), axis=2)


def pack_decoder_input_bits(condition, latents, inner):
    """Pack one decoder chunk directly from raw BF16 latent storage bits.

    Only the four requested source frames are rounded and only the requested
    latent chunk is transposed; the returned array owns its 784-channel bits.
    """
    condition, latents = np.asarray(condition), np.asarray(latents)
    if type(inner) is not int or not 0 <= inner < 4:
        raise ValueError("Decoder inner chunk must be an integer in [0, 4)")
    if (condition.ndim != 5 or condition.shape[:2] != (1, 3)
            or condition.shape[2] not in (13, 16) or condition.dtype != np.float32
            or latents.ndim != 5 or latents.shape[:3] != (1, 16, 4)
            or latents.dtype != np.uint16 or not latents.flags.c_contiguous
            or not np.isfinite(condition).all()):
        raise ValueError("Expected finite FP32 source video and contiguous BF16 bits")
    if np.any((latents & np.uint16(0x7f80)) == np.uint16(0x7f80)):
        raise ValueError("Finite uint16 BF16 latent bits required")
    height, width = condition.shape[-2:]
    if min(height, width) < 8 or height % 8 or width % 8 \
            or latents.shape[-2:] != (height // 8, width // 8):
        raise ValueError("Decoder conditioning and latent geometry differ")
    if condition.shape[2] == 13 and inner == 0:
        selected_source = np.broadcast_to(
            condition[:, :, :1], (1, 3, 4, height, width)
        )
    else:
        source_start = 4 * inner - 3 if condition.shape[2] == 13 else 4 * inner
        selected_source = condition[:, :, source_start:source_start + 4]
    selected = storage_tensor(selected_source, "bf16")
    # The owning typed tensor stays alive until packed has copied its pixels.
    selected_bits = selected.data.view(np.uint16)
    h8, w8 = height // 8, width // 8
    packed = np.empty((1, 784, h8, w8), dtype=np.uint16)
    shuffled = selected_bits.reshape(1, 3, 1, 4, h8, 8, w8, 8)
    shuffled = shuffled.transpose(0, 2, 1, 3, 5, 7, 4, 6).reshape(1, 1, 768, h8, w8)
    packed[:, :768] = shuffled[:, 0]
    packed[:, 768:] = latents[:, :, inner]
    return packed


# Use the source expression and BF16 round-to-nearest-even once for all 256
# colors. Multiplication by a reciprocal instead changes halfway cases.
_rgb_normalized = np.arange(256, dtype=np.float32) / np.float32(127.5) - np.float32(1)
_rgb_words = _rgb_normalized.view(np.uint32)
RGB_TO_BF16 = ((_rgb_words + np.uint32(0x7FFF) + ((_rgb_words >> 16) & 1)) >> 16).astype(np.uint16)
RGB_TO_BF16.setflags(write=False)
del _rgb_normalized, _rgb_words


def pack_rgb_decoder_input_bits(rgb4, latents, *, inner):
    """Pack four authenticated RGB frames and raw BF16 latents, without F32 pixels.

    The source reader supplies the exact four frames, including startup/tail
    repeats. Reorder uint8 indices before lookup so the BF16 reshape is a view,
    not another full pixel allocation.
    """
    if type(inner) is not int or not 0 <= inner < 4:
        raise ValueError("Decoder inner chunk must be an integer in [0, 4)")
    if (not isinstance(rgb4, np.ndarray) or rgb4.dtype != np.uint8
            or rgb4.ndim != 5 or rgb4.shape[:3] != (1, 3, 4)
            or not rgb4.flags.c_contiguous):
        raise ValueError("Expected contiguous uint8 RGB frames [1,3,4,H,W]")
    height, width = rgb4.shape[-2:]
    if min(height, width) < 8 or height % 8 or width % 8:
        raise ValueError("Decoder RGB geometry must be padded to multiples of eight")
    h8, w8 = height // 8, width // 8
    if (not isinstance(latents, np.ndarray) or latents.dtype != np.uint16
            or latents.shape != (1, 16, 4, h8, w8)
            or not latents.flags.c_contiguous
            or np.any((latents & np.uint16(0x7F80)) == np.uint16(0x7F80))):
        raise ValueError("Expected finite contiguous BF16 latent bits [1,16,4,H/8,W/8]")
    indices = np.ascontiguousarray(rgb4.reshape(1, 3, 4, h8, 8, w8, 8).transpose(
        0, 1, 2, 4, 6, 3, 5))
    shuffled = RGB_TO_BF16[indices].reshape(1, 768, h8, w8)
    packed = np.empty((1, 784, h8, w8), dtype=np.uint16)
    packed[:, :768] = shuffled
    packed[:, 768:] = latents[:, :, inner]
    return packed


class SparseXmx:
    """Tiny trajectory diagnostic using the qualified BF16 custom kernel.

    UHD uses the bounded resident controller, not these host-copy requests.
    Limit this diagnostic's graph/tensor inventory even on malformed input.
    """
    def __init__(self, core, device):
        self.core, self.device = core, device
        self.engines = {}

    def __call__(self, q, k, v, mask):
        import openvino as ov
        import plus_sparse_xmx as kernel
        if not 1 <= q.shape[0] <= 4 or k.shape[0] > 10:
            raise ValueError("Host-copy trajectory attention is restricted to tiny geometry")
        q = np.ascontiguousarray(q.reshape(q.shape[0], 128, 12, 128).transpose(2, 0, 1, 3))
        k, v = (np.ascontiguousarray(x.reshape(x.shape[0], 128, 12, 128)) for x in (k, v))
        ids = kernel.pack(mask, capacity=k.shape[0])
        bits = [kernel.float32_to_bf16_bits(x) for x in (q, k, v)]
        kernel.validate_inputs(*bits, ids)
        shape = q.shape[0], q.shape[1], k.shape[0], ids.shape[-1]
        if shape not in self.engines:
            if len(self.engines) == 2:
                # Retire the oldest shape/request and its bound tensors.
                del self.engines[next(iter(self.engines))]
            model = kernel.build(*shape, canonical_kv=True)
            config = {"CONFIG_FILE": str(Path(kernel.__file__).with_suffix(".xml")),
                      "NUM_STREAMS": ov.properties.streams.Num(1),
                      "INFERENCE_PRECISION_HINT": kernel.STORAGE,
                      "PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": ""}
            engine = self.core.compile_model(model, self.device, config)
            if list(engine.get_property("EXECUTION_DEVICES")) != [self.device]:
                raise ValueError("Plus attention executed on an unintended device")
            self.engines[shape] = engine, engine.create_infer_request()
        engine, request = self.engines[shape]
        tensors = []
        for name, data in zip(("Q", "K", "V", "blocks"), (*bits, ids)):
            tensor = ov.Tensor(engine.input(name).get_element_type(), data.shape)
            if name == "blocks":
                tensor.data[...] = data
            else:
                tensor.data.view(np.uint16)[...] = data
            request.set_tensor(name, tensor)
            tensors.append(tensor)
        request.infer()
        result_bits = np.array(request.get_output_tensor().data.view(np.uint16), copy=True)
        result = kernel.bf16_bits_to_float32(result_bits)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite Plus attention output")
        return result
