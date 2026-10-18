#!/usr/bin/env python3
"""Convert the pinned TCDecoder into one explicit-state, four-frame IR.

This is a developer-only converter.  It intentionally unrolls exactly one
conditioned latent through the original decoder order: the two temporal
``TGrow`` branches are expanded depth-first, yielding four RGB frames.  The
nine MemBlock histories are ordinary graph inputs/outputs; no OpenVINO
stateful-variable or generic scheduling framework is introduced here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from numbers import Integral
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "source/FlashVSR"
TC_SOURCE = SOURCE_ROOT / "examples/WanVSR/utils/TCDecoder.py"
CHECKPOINT = ROOT / "models/TCDecoder.ckpt"
CHECKPOINT_SHA256 = "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161"
SOURCE_SHA256 = "33f060b36a7650455745cdd73285c6e946d3bae23de32c1dfab3098a38acfd32"
CHANNELS = (512, 256, 128, 128)
LATENT_CHANNELS = 784
SOURCE_HW = (128, 128)
LATENT_HW = (16, 16)
MAX_SOURCE_HW = (768, 1280)
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
STATE_SHAPES = (
    (1, 512, 16, 16),
    (1, 512, 16, 16),
    (1, 512, 16, 16),
    (1, 256, 32, 32),
    (1, 256, 32, 32),
    (1, 256, 32, 32),
    (1, 128, 64, 64),
    (1, 128, 64, 64),
    (1, 128, 64, 64),
)


def decoder_geometry(source_hw: tuple[int, int] = SOURCE_HW) -> dict[str, Any]:
    """Validate a decoder source grid and derive every spatial port shape.

    The decoder has three fixed 2x spatial growth stages, so source dimensions
    must be positive multiples of eight.  The explicit preparation target is
    bounded to 1280x768 (height x width); larger grids are intentionally not
    admitted by this component converter.
    """
    if (not isinstance(source_hw, (tuple, list)) or len(source_hw) != 2 or
            any(isinstance(value, bool) or not isinstance(value, Integral)
                for value in source_hw)):
        raise ValueError("source geometry must be a (height, width) pair of positive integers")
    height, width = (int(value) for value in source_hw)
    if height <= 0 or width <= 0:
        raise ValueError("source geometry must have positive dimensions")
    if height % 8 or width % 8:
        raise ValueError("decoder source dimensions must be positive multiples of 8")
    if height > MAX_SOURCE_HW[0] or width > MAX_SOURCE_HW[1]:
        raise ValueError(
            f"decoder geometry exceeds the fixed HD bound {MAX_SOURCE_HW[1]}x{MAX_SOURCE_HW[0]}"
        )
    latent_hw = (height // 8, width // 8)
    state_shapes = (
        (1, 512, *latent_hw),
        (1, 512, *latent_hw),
        (1, 512, *latent_hw),
        (1, 256, height // 4, width // 4),
        (1, 256, height // 4, width // 4),
        (1, 256, height // 4, width // 4),
        (1, 128, height // 2, width // 2),
        (1, 128, height // 2, width // 2),
        (1, 128, height // 2, width // 2),
    )
    return {
        "source_hw": (height, width),
        "latent_hw": latent_hw,
        "state_shapes": state_shapes,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_module():
    spec = importlib.util.spec_from_file_location("pinned_tcdecoder_converter", TC_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned decoder source: {TC_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_decoder() -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, Any]]:
    if sha256(TC_SOURCE) != SOURCE_SHA256:
        raise ValueError("Pinned TCDecoder source fingerprint changed")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Pinned TCDecoder checkpoint fingerprint changed")
    source = load_source_module()
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
        raise ValueError("TCDecoder checkpoint keys do not exactly match the source model")
    if len(state) != 66 or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected exactly 66 tensor parameters")
    for key, value in state.items():
        if value.dtype != torch.float32 or tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(f"Unexpected tensor metadata for {key}: {value.dtype} {tuple(value.shape)}")
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    metadata = {
        "source_sha256": SOURCE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "parameter_count": len(state),
        "parameter_numel": int(sum(value.numel() for value in state.values())),
        "parameter_bytes_f32": int(sum(value.numel() * value.element_size() for value in state.values())),
        "channels": list(CHANNELS),
        "latent_channels": LATENT_CHANNELS,
        "source_geometry": list(SOURCE_HW),
        "latent_geometry": list(LATENT_HW),
    }
    return decoder, state, metadata


class FourFrameStatefulDecoder(torch.nn.Module):
    """Static one-latent unroll of the original module sequence.

    The input is one already-packed ``[1, 784, 16, 16]`` conditioned latent.
    The output is ``[1, 4, 3, 128, 128]``.  State inputs/outputs are the
    last pre-MemBlock tensors in source index order.  Branches are traversed
    child 0 then child 1 at each TGrow, matching the source queue's reversed
    insertion and depth-first consumption.
    """

    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder.decoder

    @staticmethod
    def _mem(block: torch.nn.Module, x: torch.Tensor, past: torch.Tensor):
        # The source executor stores the input tensor before each MemBlock and
        # supplies zeros only for the first history.  Returning x as the new
        # state preserves that ownership boundary without an in-place update.
        return block(x, past), x

    def forward(
        self,
        packed: torch.Tensor,
        mem_512_0: torch.Tensor,
        mem_512_1: torch.Tensor,
        mem_512_2: torch.Tensor,
        mem_256_0: torch.Tensor,
        mem_256_1: torch.Tensor,
        mem_256_2: torch.Tensor,
        mem_128_0: torch.Tensor,
        mem_128_1: torch.Tensor,
        mem_128_2: torch.Tensor,
    ):
        state = [mem_512_0, mem_512_1, mem_512_2, mem_256_0, mem_256_1,
                 mem_256_2, mem_128_0, mem_128_1, mem_128_2]
        block = self.decoder

        x = block[0](packed)
        x = block[1](x)
        x = block[2](x)
        x = block[3](x)
        x = block[4](x)
        for offset, index in enumerate((5, 6, 7)):
            x, state[offset] = self._mem(block[index], x, state[offset])
        x = block[8](x)
        x = block[9](x)
        x = block[10](x)
        for offset, index in enumerate((11, 12, 13), start=3):
            x, state[offset] = self._mem(block[index], x, state[offset])

        x = block[14](x)
        # TGrow returns [stride, C, H, W].  The source queue views that as
        # [N, stride*C, H, W] before splitting channels, preserving each
        # leading-batch child as a normal [N, C, H, W] tensor.
        grown = block[15](x).reshape(1, 512, *x.shape[-2:])
        stage2_children = (grown[:, :256], grown[:, 256:])
        outputs = []
        for stage2_child in stage2_children:
            x = block[16](stage2_child)
            for offset, index in enumerate((17, 18, 19), start=6):
                x, state[offset] = self._mem(block[index], x, state[offset])
            x = block[20](x)
            grown = block[21](x).reshape(1, 256, *x.shape[-2:])
            stage3_children = (grown[:, :128], grown[:, 128:])
            for stage3_child in stage3_children:
                x = block[22](stage3_child)
                x = block[23](x)
                x = block[24](x)
                x = block[25](x)
                outputs.append(block[26](x))

        frames = torch.stack(outputs, dim=1)
        return (frames, *state)


def make_example_inputs(
    source_hw: tuple[int, int] = SOURCE_HW,
) -> tuple[torch.Tensor, ...]:
    geometry = decoder_geometry(source_hw)
    packed = torch.zeros((1, LATENT_CHANNELS, *geometry["latent_hw"]), dtype=torch.float32)
    states = tuple(torch.zeros(shape, dtype=torch.float32)
                   for shape in geometry["state_shapes"])
    return (packed, *states)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    """Return the two spatial values used by the pinned Conv2d modules."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-D Conv2d value, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _native_constant(ov, cache: dict[tuple[Any, ...], Any], value: Any,
                     key: tuple[Any, ...] | None = None, *,
                     learned_parameter_factory=None, learned_source=None,
                     learned_shape=None):
    """Create one F32 constant, or an explicit identity-mapped weight port.

    The factory is deliberately opt-in and receives the original module
    Parameter, never a shape/value lookup.  Ordinary scalar/control constants
    therefore remain exactly on the historical path.
    """
    if learned_parameter_factory is not None and learned_source is not None:
        shape = learned_shape if learned_shape is not None else getattr(value, "shape", None)
        if shape is None:
            raise ValueError("learned parameter shape is required")
        parameter = learned_parameter_factory(learned_source, tuple(int(dim) for dim in shape))
        if parameter is not None:
            parameter_node = (parameter.get_node()
                              if hasattr(parameter, "get_node") else parameter)
            cache.setdefault(("__learned_parameter_node__", id(parameter_node)), parameter_node)
            return parameter
    array = (value.detach().cpu().contiguous().numpy().copy()
             if isinstance(value, torch.Tensor) else np.asarray(value))
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    array = np.ascontiguousarray(array)
    if key is None:
        key = ("value", array.dtype.str, tuple(array.shape), array.tobytes())
    if key not in cache:
        cache[key] = ov.opset1.constant(array)
    return cache[key]


def _native_convolution(ov, x, module: torch.nn.Conv2d, weight: torch.Tensor,
                        cache: dict[tuple[Any, ...], Any], key: tuple[Any, ...],
                        *, add_bias: bool = True, learned_parameter_factory=None):
    stride = _pair(module.stride)
    padding = _pair(module.padding)
    dilation = _pair(module.dilation)
    result = ov.opset1.convolution(
        x,
        _native_constant(
            ov, cache, weight, key,
            learned_parameter_factory=learned_parameter_factory,
            learned_source=weight,
            learned_shape=tuple(int(dim) for dim in weight.shape),
        ),
        list(stride), list(padding), list(padding), list(dilation),
    )
    if add_bias and module.bias is not None:
        bias = module.bias.detach().reshape(1, -1, 1, 1)
        result = ov.opset1.add(
            result,
            _native_constant(
                ov, cache, bias, ("bias", id(module.bias)),
                learned_parameter_factory=learned_parameter_factory,
                learned_source=module.bias,
                learned_shape=tuple(int(dim) for dim in bias.shape),
            ),
        )
    return result


def _native_module(ov, x, module: torch.nn.Module,
                   cache: dict[tuple[Any, ...], Any], *,
                   learned_parameter_factory=None):
    """Lower one non-stateful source module without Torch frontend rewrites."""
    if isinstance(module, torch.nn.Conv2d):
        return _native_convolution(ov, x, module, module.weight, cache,
                                    ("weight", id(module.weight)),
                                    learned_parameter_factory=learned_parameter_factory)
    if isinstance(module, torch.nn.ReLU):
        return ov.opset1.relu(x)
    if isinstance(module, torch.nn.Upsample):
        if module.scale_factor != 2 or module.mode != "nearest":
            raise ValueError(f"Unsupported source upsample: {module}")
        # This is the static 2x nearest operation used by nn.Upsample in the
        # pinned source; asymmetric/floor matches PyTorch's integer coordinates.
        return ov.opset11.interpolate(
            x,
            np.array([2.0, 2.0], dtype=np.float32),
            "nearest", "scales",
            axes=np.array([2, 3], dtype=np.int64),
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
    if type(module).__name__ == "Clamp":
        scale = _native_constant(ov, cache, np.array(3.0, dtype=np.float32))
        return ov.opset1.multiply(ov.opset1.tanh(ov.opset1.divide(x, scale)), scale)
    if type(module).__name__ == "TGrow":
        return _native_convolution(ov, x, module.conv, module.conv.weight,
                                   cache, ("weight", id(module.conv.weight)),
                                   learned_parameter_factory=learned_parameter_factory)
    raise TypeError(f"Unsupported source module: {type(module).__name__}")


def _native_memblock(ov, x, past, block: torch.nn.Module,
                     cache: dict[tuple[Any, ...], Any], *, combined_mem: bool = False,
                     learned_parameter_factory=None):
    """Lower MemBlock while retaining explicit source-history ownership.

    The first source convolution consumes ``cat([x, past], 1)``.  Splitting
    its input channels into two convolutions and adding the results is
    algebraically identical in FP32, and avoids a CPU-plugin branch/Concat
    lowering failure for the second TGrow child.  The frozen CPU gate below
    checks the resulting arithmetic against the pinned source.
    """
    first = block.conv[0]
    if combined_mem:
        # Preserve the source's single convolution/storage boundary. Flatten
        # before Concat to avoid the known GPU Conv/Concat matcher failure.
        flat = [ov.opset1.reshape(v, np.array([-1], np.int64), False) for v in (x, past)]
        shape = list(x.get_output_shape(0))
        shape[1] *= 2
        joined = ov.opset1.reshape(ov.opset1.concat(flat, 0), np.array(shape, np.int64), False)
        value = _native_convolution(ov, joined, first, first.weight, cache,
                                    ("mem_weight", id(first.weight), "combined"),
                                    learned_parameter_factory=learned_parameter_factory)
    else:
        n_in = int(first.in_channels // 2)
        left_weight = first.weight[:, :n_in].contiguous()
        right_weight = first.weight[:, n_in:].contiguous()
        left = _native_convolution(
            ov, x, first, left_weight,
            cache, ("mem_weight", id(first.weight), "left"), add_bias=False,
            learned_parameter_factory=learned_parameter_factory,
        )
        right = _native_convolution(
            ov, past, first, right_weight,
            cache, ("mem_weight", id(first.weight), "right"), add_bias=False,
            learned_parameter_factory=learned_parameter_factory,
        )
        # Bias belongs to the original combined convolution and is added once.
        if first.bias is not None:
            bias = first.bias.detach().reshape(1, -1, 1, 1)
            bias_node = _native_constant(
                ov, cache, bias, ("bias", id(first.bias)),
                learned_parameter_factory=learned_parameter_factory,
                learned_source=first.bias,
                learned_shape=tuple(int(dim) for dim in bias.shape),
            )
            value = ov.opset1.add(ov.opset1.add(left, right), bias_node)
        else:
            value = ov.opset1.add(left, right)
    for layer in list(block.conv)[1:]:
        value = _native_module(
            ov, value, layer, cache,
            learned_parameter_factory=learned_parameter_factory,
        )
    skip = x if isinstance(block.skip, torch.nn.Identity) else _native_module(
        ov, x, block.skip, cache,
        learned_parameter_factory=learned_parameter_factory,
    )
    return ov.opset1.relu(ov.opset1.add(value, skip)), x


def _native_slice_channels(ov, x, start: int, stop: int):
    return ov.opset8.slice(
        x,
        np.array([start], dtype=np.int64),
        np.array([stop], dtype=np.int64),
        np.array([1], dtype=np.int64),
        axes=np.array([1], dtype=np.int64),
    )


def build_native_model(
    ov,
    decoder: torch.nn.Module,
    source_hw: tuple[int, int] = SOURCE_HW,
    *, combined_mem: bool = False, learned_parameter_factory=None,
):
    """Build one fixed-geometry latent graph using the native OV operations."""
    if learned_parameter_factory is not None and not combined_mem:
        raise ValueError("identity-mapped learned parameters require combined_mem=True")
    geometry = decoder_geometry(source_hw)
    source_hw = geometry["source_hw"]
    latent_hw = geometry["latent_hw"]
    state_shapes = geometry["state_shapes"]
    block = decoder.decoder
    cache: dict[tuple[Any, ...], Any] = {}

    def module(value, index: int):
        return _native_module(
            ov, value, block[index], cache,
            learned_parameter_factory=learned_parameter_factory,
        )

    def mem(value, past, index: int):
        return _native_memblock(
            ov, value, past, block[index], cache, combined_mem=combined_mem,
            learned_parameter_factory=learned_parameter_factory,
        )

    input_names = ("packed", *STATE_NAMES)
    parameters = [
        ov.opset1.parameter([1, LATENT_CHANNELS, *latent_hw], ov.Type.f32,
                             name=input_names[0]),
    ]
    parameters.extend(
        ov.opset1.parameter(list(shape), ov.Type.f32, name=name)
        for name, shape in zip(STATE_NAMES, state_shapes)
    )

    packed = module(parameters[0], 0)
    for index in range(1, 5):
        packed = module(packed, index)
    states = list(parameters[1:])
    for offset, index in enumerate((5, 6, 7)):
        packed, states[offset] = mem(packed, states[offset], index)
    for index in (8, 9, 10):
        packed = module(packed, index)
    for offset, index in enumerate((11, 12, 13), start=3):
        packed, states[offset] = mem(packed, states[offset], index)
    packed = module(packed, 14)
    grown = module(packed, 15)

    frames = []
    # Each TGrow is split in channel order after its source per-frame 4-D
    # convolution.  This is the source queue's child-0, child-1 depth-first
    # order (the executor inserts reversed children into its front queue).
    for stage2_child in (_native_slice_channels(ov, grown, 0, 256),
                         _native_slice_channels(ov, grown, 256, 512)):
        value = module(stage2_child, 16)
        for offset, index in enumerate((17, 18, 19), start=6):
            value, states[offset] = mem(value, states[offset], index)
        value = module(value, 20)
        grown_stage3 = module(value, 21)
        for stage3_child in (_native_slice_channels(ov, grown_stage3, 0, 128),
                             _native_slice_channels(ov, grown_stage3, 128, 256)):
            value = stage3_child
            for index in (22, 23, 24, 25, 26):
                value = module(value, index)
            frames.append(value)

    frame_tensor = ov.opset1.reshape(
        ov.opset1.concat(frames, 1),
        np.array([1, 4, 3, *source_hw], dtype=np.int64),
        False,
    )
    results = [ov.opset1.result(frame_tensor)]
    results.extend(ov.opset1.result(value) for value in states)
    if learned_parameter_factory is not None:
        # The callback returns an Output, while the Model constructor needs
        # each unique Parameter node declared explicitly.  Registration in the
        # existing traversal cache avoids a second callback protocol and is
        # ignored entirely by the historical constant path.
        parameter_nodes = [
            value for key, value in cache.items()
            if isinstance(key, tuple) and key
            and key[0] == "__learned_parameter_node__"
        ]
        parameters.extend(parameter_nodes)
    model = ov.Model(results, parameters, "tcdecoder_native_explicit_state")
    model.validate_nodes_and_infer_types()
    model.set_rt_info("flashvsr-tcdecoder-four-frame-explicit-state-v2",
                      ["conversion", "identity"])
    for output_port, name in zip(model.outputs, ("frames", *STATE_NAMES)):
        output_port.get_node().set_friendly_name(name)
        output_port.set_names({name})
    for input_port, name in zip(model.inputs, input_names):
        input_port.get_node().set_friendly_name(name)
        input_port.set_names({name})
    model.validate_nodes_and_infer_types()
    return model


def convert(
    output: Path,
    source_hw: tuple[int, int] = SOURCE_HW,
) -> dict[str, Any]:
    import openvino as ov

    decoder, state, metadata = load_decoder()
    geometry = decoder_geometry(source_hw)
    model = build_native_model(ov, decoder, source_hw=geometry["source_hw"])
    input_names = ("packed", *STATE_NAMES)
    output.mkdir(parents=True, exist_ok=False)
    xml = output / "decoder.xml"
    ov.save_model(model, xml, compress_to_fp16=False)
    metadata.update({
        "source_geometry": list(geometry["source_hw"]),
        "latent_geometry": list(geometry["latent_hw"]),
        "ir_xml": str(xml),
        "ir_bin": str(xml.with_suffix(".bin")),
        "ir_xml_bytes": xml.stat().st_size,
        "ir_bin_bytes": xml.with_suffix(".bin").stat().st_size,
        "ir_xml_sha256": sha256(xml),
        "ir_bin_sha256": sha256(xml.with_suffix(".bin")),
        "input_names": list(input_names),
        "output_names": ["frames", *STATE_NAMES],
        "input_shapes": [[1, LATENT_CHANNELS, *geometry["latent_hw"]],
                         *[list(shape) for shape in geometry["state_shapes"]]],
        "output_shapes": [[1, 4, 3, *geometry["source_hw"]],
                          *[list(shape) for shape in geometry["state_shapes"]]],
        "torch": torch.__version__,
        "openvino": ov.get_version(),
        "converter_backend": "native_openvino_ops",
        "memblock_first_conv": "split_x_plus_past_linear_decomposition",
        "branch_order": ["stage2_child0/stage3_child0",
                         "stage2_child0/stage3_child1",
                         "stage2_child1/stage3_child0",
                         "stage2_child1/stage3_child1"],
    })
    (output / "conversion.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-size", "--working-size", "--target-size", default="128x128",
        help="decoder source size WIDTHxHEIGHT (default: 128x128)",
    )
    args = parser.parse_args()
    try:
        width_text, height_text = args.source_size.lower().split("x", 1)
        source_hw = decoder_geometry((int(height_text), int(width_text)))["source_hw"]
    except (AttributeError, TypeError, ValueError) as exc:
        parser.error(f"invalid --source-size {args.source_size!r}: {exc}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    metadata = convert(args.output, source_hw=source_hw)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
