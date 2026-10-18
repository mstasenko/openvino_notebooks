"""FlashVSR+ four-frame TCDecoder graph with explicit storage boundaries.

The Plus decoder has the same module sequence and state ABI as the pinned
native converter.  This module deliberately reuses that traversal rather than
copying its architecture: only the parameter/constant boundary is replaced so
source F32 weights are rounded directly to BF16 (or F16) with
``plus_storage.storage_tensor``.  In particular, BF16 never passes through an
intermediate F16 tensor.
"""

from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from plus_storage import storage_tensor, tensor_array
from reference_paths import ROOT, SOURCE_ROOT, TCDECODER_CHECKPOINT


HERE = Path(__file__).resolve()
LAB_ROOT = HERE.parents[2]
SOURCE_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
SOURCE = SOURCE_ROOT / "inference/decoders/tiny_model.py"
CHECKPOINT = TCDECODER_CHECKPOINT
SOURCE_SHA256 = "85369a0fcd41073a2acc1050cfeb37a85d123e84c12cbee6339bb4308dc04ac6"
CHECKPOINT_SHA256 = "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161"
PARAMETER_MANIFEST_SHA256 = "b7edfda7d33cca8bbb35979e4580f3bfd07d6e759d3e22f73c78291ab4042b8d"

CHANNELS = (512, 256, 128, 128)
LATENT_CHANNELS = 784
SOURCE_HW = (128, 128)
STORAGE_PRECISIONS = {"bf16", "f16"}

# Keep these names and shapes identical to the native converter.  They are
# part of the stream cache ABI, not an implementation detail of this wrapper.
STATE_NAMES = (
    "mem_512_0",
    "mem_512_1",
    "mem_512_2",
    "mem_256_0",
    "mem_256_1",
    "mem_256_2",
    "mem_128_0",
    "mem_128_1",
    "mem_128_2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_manifest(state: dict[str, torch.Tensor]) -> tuple[dict[str, str], str]:
    hashes = {
        key: hashlib.sha256(
            np.ascontiguousarray(value.detach().cpu().numpy()).tobytes(order="C")
        ).hexdigest()
        for key, value in state.items()
    }
    text = "\n".join(f"{key}:{hashes[key]}" for key in sorted(hashes)) + "\n"
    return hashes, hashlib.sha256(text.encode("ascii")).hexdigest()


def _load_source_module():
    spec = importlib.util.spec_from_file_location("flashvsr_plus_tcdecoder", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned Plus decoder source: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_native_converter():
    """Load the shared native traversal without depending on process PYTHONPATH."""

    path = ROOT / "openvino/decoder/convert_decoder.py"
    spec = importlib.util.spec_from_file_location("flashvsr_native_convert_decoder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import native decoder traversal: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_decoder() -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, Any]]:
    """Load and authenticate the official Plus TCDecoder source/checkpoint."""

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    source_hash = sha256(SOURCE)
    checkpoint_hash = sha256(CHECKPOINT)
    if source_hash != SOURCE_SHA256:
        raise ValueError(f"Pinned Plus decoder source fingerprint changed: {source_hash}")
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError(
            f"Pinned TCDecoder checkpoint fingerprint changed: {checkpoint_hash}"
        )

    source = _load_source_module()
    decoder = source.TAEHV(
        checkpoint_path=None,
        channels=list(CHANNELS),
        latent_channels=LATENT_CHANNELS,
    ).to(device="cpu", dtype=torch.float32).eval()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("TCDecoder checkpoint is not a tensor mapping")
    expected = decoder.state_dict()
    if set(state) != set(expected):
        raise ValueError("TCDecoder checkpoint keys do not exactly match Plus source")
    if len(state) != 66 or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected exactly 66 Plus TCDecoder tensor parameters")
    for key, value in state.items():
        if value.dtype != torch.float32 or tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(f"Unexpected tensor metadata for {key}: {value.dtype} {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite Plus TCDecoder tensor: {key}")
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    parameter_hashes, parameter_manifest = _parameter_manifest(state)
    if parameter_manifest != PARAMETER_MANIFEST_SHA256:
        raise ValueError(f"Plus TCDecoder parameter manifest changed: {parameter_manifest}")
    metadata = {
        "source_commit": SOURCE_COMMIT,
        "source": str(SOURCE),
        "source_sha256": source_hash,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "parameter_count": len(state),
        "parameter_numel": int(sum(value.numel() for value in state.values())),
        "parameter_bytes_f32": int(sum(value.numel() * value.element_size() for value in state.values())),
        "parameter_hashes": parameter_hashes,
        "parameter_manifest_sha256": parameter_manifest,
        "channels": list(CHANNELS),
        "latent_channels": LATENT_CHANNELS,
        "storage_conversion": "source F32 -> plus_storage.storage_tensor directly",
    }
    return decoder, state, metadata


class _StorageOpset:
    """Proxy only the two creation operations used by the native converter."""

    def __init__(self, opset, precision: str):
        self._opset = opset
        self._precision = precision

    def __getattr__(self, name: str):
        if name == "constant":
            def constant(value, *args, **kwargs):
                array = (
                    value.detach().cpu().contiguous().numpy()
                    if isinstance(value, torch.Tensor)
                    else np.asarray(value)
                )
                # Shape/axis constants remain integer control tensors.  All
                # floating constants emitted by the converter are either
                # learned values or source scalar constants and use the same
                # direct storage boundary.
                if array.dtype.kind == "f":
                    value = storage_tensor(array, self._precision)
                    # OpenVINO's opset1 helper does not accept ov.Tensor on
                    # this runtime; opset13 does and creates the same Constant.
                    import openvino as ov

                    return ov.opset13.constant(value, *args, **kwargs)
                return getattr(self._opset, name)(value, *args, **kwargs)

            return constant
        if name == "parameter":
            def parameter(shape, element_type, *args, **kwargs):
                import openvino as ov

                if element_type == ov.Type.f32:
                    element_type = ov.Type.bf16 if self._precision == "bf16" else ov.Type.f16
                return getattr(self._opset, name)(shape, element_type, *args, **kwargs)

            return parameter
        return getattr(self._opset, name)


class _StorageOpenVINO:
    """Small root proxy consumed by ``convert_decoder.build_native_model``."""

    def __init__(self, precision: str):
        if precision not in STORAGE_PRECISIONS:
            raise ValueError("Plus decoder storage must be 'bf16' or 'f16'")
        import openvino as ov

        self.Type = ov.Type
        self.opset1 = _StorageOpset(ov.opset1, precision)
        self.opset8 = _StorageOpset(ov.opset8, precision)
        self.opset11 = _StorageOpset(ov.opset11, precision)

    def __getattr__(self, name: str):
        import openvino as ov

        return getattr(ov, name)


def build_decoder_graph(
    decoder: torch.nn.Module,
    source_hw: tuple[int, int] = SOURCE_HW,
    *,
    precision: str = "bf16",
    combined_mem: bool = True,
    clamp_storage: bool = False,
    residual_storage: bool = True,
    learned_parameter_factory=None,
):
    """Build source-precision decoding: ten public results plus private stores.

    Residual-linear Results preserve rounding before skip addition. A GPU
    controller must bind them to remote buffers, not read them to CPU. The
    switches retain diagnostic controls, not user-facing model modes.
    """

    if precision not in STORAGE_PRECISIONS:
        raise ValueError("Plus decoder storage must be 'bf16' or 'f16'")
    if learned_parameter_factory is not None and not combined_mem:
        raise ValueError("identity-mapped learned parameters require combined_mem=True")
    if not isinstance(decoder, torch.nn.Module) or not hasattr(decoder, "decoder"):
        raise TypeError("decoder must be the authenticated Plus TAEHV module")
    if any(parameter.dtype != torch.float32 for parameter in decoder.parameters()):
        raise ValueError("Plus source decoder must remain F32 until graph storage conversion")

    # This import is intentionally the old native traversal: it owns the
    # source module ordering, MemBlock history contract, and branch chronology.
    native_converter = _load_native_converter()
    model = native_converter.build_native_model(
        _StorageOpenVINO(precision), decoder, source_hw=tuple(source_hw),
        combined_mem=combined_mem, learned_parameter_factory=learned_parameter_factory,
    )
    preserve_decoder_storage(model, precision=precision, clamp_storage=clamp_storage,
                             residual_storage=residual_storage)
    model.set_rt_info("flashvsr-plus-tcdecoder-four-frame-explicit-state-v1",
                      ["conversion", "identity"])
    model.set_rt_info(precision, ["conversion", "storage"])
    return model


def preserve_decoder_storage(model, *, precision="bf16", clamp_storage=False,
                             residual_storage=True, expected_clamps=1, expected_residuals=12):
    """Keep the same source rounding in a complete decoder or exact phase."""
    import openvino as ov
    if precision not in STORAGE_PRECISIONS:
        raise ValueError("Explicit decoder storage precision required")
    ops = ov.opset13
    clamps = [node for node in model.get_ordered_ops() if node.get_type_name() == "Tanh"]
    if len(clamps) != expected_clamps:
        raise ValueError("Source decoder Clamp count changed")
    storage = ov.Type.bf16 if precision == "bf16" else ov.Type.f16
    for old_tanh in clamps:
        old_divide = old_tanh.input_value(0).get_node()
        old_multiply, = [use.get_node() for use in old_tanh.output(0).get_target_inputs()]
        if old_divide.get_type_name() != "Divide" or old_multiply.get_type_name() != "Multiply":
            raise ValueError("Source decoder Clamp topology changed")
        # Source scalar /3 and *3 use FP32 arithmetic then round to storage.
        # A storage-typed Divide can instead fold a low-precision reciprocal.
        scale = ops.constant(np.float32(3))
        divided = ops.convert(ops.divide(ops.convert(old_divide.input_value(0), ov.Type.f32), scale), storage)
        tanh = ops.tanh(divided)
        multiplied = ops.convert(ops.multiply(ops.convert(tanh, ov.Type.f32), scale), storage)
        old_multiply.output(0).replace(multiplied.output(0))
        model.validate_nodes_and_infer_types()
        if clamp_storage:
            for name, node in (("divided", divided), ("tanh", tanh), ("multiplied", multiplied)):
                result = ops.result(node)
                result.output(0).get_tensor().set_names({"clamp_" + name})
                model.add_results([result])
            model.validate_nodes_and_infer_types()
    if residual_storage:
        # Diagnostic: source Conv rounds before the residual Add. GPU post-op
        # fusion can otherwise keep that value in its accumulator precision.
        boundaries = []
        for node in model.get_ordered_ops():
            if node.get_type_name() != "Add":
                continue
            left, right = (node.input_value(i).get_node() for i in (0, 1))
            if (left.get_type_name() == "Add" and right.get_type_name() != "Constant"
                    and left.input_value(0).get_node().get_type_name() == "Convolution"
                    and left.input_value(1).get_node().get_type_name() in {"Constant", "Parameter"}):
                boundaries.append(left)
        if len(boundaries) != expected_residuals:
            raise ValueError("Source decoder residual boundary count changed")
        for index, node in enumerate(boundaries):
            result = ops.result(node)
            result.output(0).get_tensor().set_names({f"residual_linear_{index}"})
            model.add_results([result])
        model.validate_nodes_and_infer_types()
    return model


def _validated_decoder_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Require the exact authenticated source-derived decoder chronology.

    ``run_plus_resident`` authenticates the manifest before opening a device,
    but the decoder is also callable independently.  Recomputing the plan
    here prevents a caller from changing a decoder range while retaining the
    source receipt.  ``_plan_metadata`` is the one source-of-truth adapter
    used by the input loader; no decoder-specific schedule is duplicated.
    """

    if not isinstance(plan, dict) or type(plan.get("source_frames")) is not int:
        raise ValueError("authenticated decoder plan is required")
    from prepare_plus_video_input import _metadata_equal_exact, _plan_metadata

    expected = _plan_metadata(plan["source_frames"])
    if not _metadata_equal_exact(plan, expected):
        raise ValueError("authenticated decoder plan chronology changed")
    if plan["call_count"] not in (2, 3, 4):
        raise ValueError("resident decoder supports two to four source calls")
    return expected


def _normalize_decoder_chunk(value: np.ndarray) -> np.ndarray:
    """Apply the two source BF16 RNE boundaries to one decoded chunk."""

    value = np.asarray(value, dtype=np.float32)
    rounded = tensor_array(storage_tensor(value * np.float32(2), "bf16"))
    return tensor_array(storage_tensor(rounded - np.float32(1), "bf16"))


def _normalize_bf16_decoder_chunk(value: np.ndarray) -> np.ndarray:
    """Normalize a losslessly expanded BF16 frame with one storage round."""

    value = np.asarray(value)
    if value.dtype != np.float32 or np.any(value.view(np.uint32) & np.uint32(0xffff)):
        raise ValueError("Expected losslessly expanded BF16 float32 values")
    # Doubling BF16 is exact unless it overflows; storage_tensor rejects that
    # overflow just like the generic two-round path. Arbitrary F32 is not safe.
    return tensor_array(storage_tensor(value * np.float32(2) - np.float32(1), "bf16"))


_DECODER_NORMALIZE_LUT_SIZE = 1 << 16
_DECODER_NORMALIZE_INVALID_ABS_BITS = np.uint16(0x7F00)


@lru_cache(maxsize=1)
def _decoder_normalization_lut() -> np.ndarray:
    """Build one immutable raw-BF16 -> normalized-F32 table from the source helper."""
    words = np.arange(_DECODER_NORMALIZE_LUT_SIZE, dtype=np.uint16)
    # The last finite exponent overflows on doubling. Leave it and all
    # Inf/NaN codes as NaN sentinels, rejected by the lookup's finite check.
    accepted = ((words & np.uint16(0x7FFF))
                < _DECODER_NORMALIZE_INVALID_ABS_BITS)
    values = (words[accepted].astype(np.uint32) << np.uint32(16)).view(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = _normalize_bf16_decoder_chunk(values)
    if normalized.dtype != np.float32 or normalized.shape != values.shape:
        raise ValueError("decoder normalization LUT oracle returned an invalid shape")
    if not np.isfinite(normalized).all():
        raise ValueError("decoder normalization LUT oracle returned nonfinite values")
    table = np.full(_DECODER_NORMALIZE_LUT_SIZE, np.nan, dtype=np.float32)
    table[accepted] = normalized
    table.setflags(write=False)
    return table


def _normalize_bf16_decoder_bits(value: np.ndarray) -> np.ndarray:
    """Normalize raw contiguous BF16 storage with the cached source-exact table."""
    value = np.asarray(value)
    if (value.dtype != np.uint16 or value.ndim < 1 or not value.size
            or not value.flags.c_contiguous):
        raise ValueError("contiguous nonempty uint16 BF16 storage is required")
    table = _decoder_normalization_lut()
    normalized = np.asarray(table[value])
    if normalized.dtype != np.float32 or normalized.shape != value.shape:
        raise ValueError("raw BF16 decoder normalization returned invalid output")
    if not np.isfinite(normalized).all():
        raise ValueError("raw BF16 decoder normalization input is invalid")
    return normalized


def decode_resident_video(core, device, context, geometry, video, denoised, plan):
    """Decode a source-authenticated stream with one recurrent GPU request.

    The graph still emits four causal frames per latent.  The first emission
    contains three temporal padding frames and is cropped before writing to a
    single visible FP32 owner.  Later calls use the authenticated decoder
    ranges and retain the BF16 ping-pong histories; no full-frame list or
    diagnostic archive is created here.
    """

    plan = _validated_decoder_plan(plan)
    working_hw = tuple(int(value) for value in geometry.working_hw)
    if len(working_hw) != 2 or min(working_hw) < 8 or any(value % 8 for value in working_hw):
        raise ValueError("decoder geometry must be a positive 8-aligned working raster")
    if max(working_hw) > 1280:
        raise ValueError("The unphased decoder remains limited to padded HD")
    height, width = int(geometry.height), int(geometry.width)
    if not (0 < height <= working_hw[0] and 0 < width <= working_hw[1]):
        raise ValueError("decoder visible geometry exceeds its working raster")

    video = np.asarray(video)
    if (video.dtype != np.float32 or video.ndim != 5 or tuple(video.shape[:2]) != (1, 3)
            or video.shape[2] < plan["projection_frames"]
            or tuple(video.shape[-2:]) != working_hw or not np.isfinite(video).all()):
        raise ValueError("decoder video must be finite FP32 [1,3,projection,working_h,working_w]")
    if not isinstance(denoised, (list, tuple)) or len(denoised) != plan["call_count"]:
        raise ValueError("decoder BF16 chunks do not match the authenticated call count")
    latent_shape = (1, 16, 4, working_hw[0] // 8, working_hw[1] // 8)
    for index, value in enumerate(denoised):
        value = np.asarray(value)
        if (value.dtype != np.uint16 or tuple(value.shape) != latent_shape
                or not value.flags.c_contiguous
                or np.any((value & np.uint16(0x7f80)) == np.uint16(0x7f80))):
            raise ValueError(f"decoder chunk {index} must be finite contiguous BF16 bits")

    # Device/runtime imports are deliberately delayed until a valid source
    # plan and finite host inputs have passed the cheap checks above.
    import openvino as ov
    from plus_pipeline import pack_decoder_input_bits
    from plus_resident import ResidentGraph, bits_tensor

    setup_started = time.perf_counter()
    decoder, state, decoder_receipt = load_decoder()
    del state
    graph = ResidentGraph(
        core,
        build_decoder_graph(decoder, source_hw=working_hw),
        device,
        context=context,
    )
    del decoder
    graph.bind_output("frames", ov.Tensor(ov.Type.bf16, graph.outputs["frames"].shape))
    setup_seconds = time.perf_counter() - setup_started
    memories = {
        name: ov.Tensor(ov.Type.bf16, graph.inputs[name].shape) for name in STATE_NAMES
    }
    for value in memories.values():
        value.data.view(np.uint16).fill(0)
    spare = {
        name: ov.Tensor(ov.Type.bf16, graph.inputs[name].shape) for name in STATE_NAMES
    }

    output_frames = int(plan["output_frames"])
    normalized = np.empty(
        (1, output_frames, 3, height, width), dtype=np.float32
    )
    written = 0
    decoded_latents = 0
    call_receipts = []
    for call_meta in plan["calls"]:
        if written >= output_frames:
            break
        call = int(call_meta["call"])
        start, stop = int(call_meta["decoder_start"]), int(call_meta["decoder_stop"])
        call_receipt = {
            "call": call,
            "decoder_start": start,
            "decoder_stop": stop,
            "output_start": written,
            "decoded_latents": 0,
        }
        for inner in range(4):
            if written >= output_frames:
                break
            packed = pack_decoder_input_bits(video[:, :, start:stop], denoised[call], inner)
            for name, target in spare.items():
                graph.bind_output(name, target)
            result = graph.infer(
                {"packed": bits_tensor(packed), **memories}, read=("frames",)
            )["frames"]
            memories, spare = spare, memories
            result = tensor_array(bits_tensor(result))
            expected_shape = (1, 4, 3, working_hw[0], working_hw[1])
            if tuple(result.shape) != expected_shape or not np.isfinite(result).all():
                raise ValueError("resident decoder returned an invalid four-frame tensor")
            if call == 0 and inner == 0:
                result = result[:, 3:]
            remaining = output_frames - written
            count = min(int(result.shape[1]), remaining)
            chunk = _normalize_decoder_chunk(result[:, :count, :, :height, :width])
            normalized[:, written:written + count] = chunk
            written += count
            decoded_latents += 1
            call_receipt["decoded_latents"] += 1
        call_receipt["output_stop"] = written
        call_receipts.append(call_receipt)

    if written != output_frames:
        raise RuntimeError(f"resident decoder stopped at {written} of {output_frames} frames")
    receipt = {
        "decoder": decoder_receipt,
        "setup_seconds": setup_seconds,
        "shape": list(normalized.shape),
        "decoded_latents": decoded_latents,
        "packing": "one BF16 latent at a time",
        "output_frames": output_frames,
        "calls": call_receipts,
        "plan_call_count": plan["call_count"],
    }
    return normalized, receipt
