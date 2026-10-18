"""BF16, source-ordered phases for the authenticated FlashVSR+ TCDecoder.

The phase boundary is deliberately the source queue boundary: ``front`` ends
before block 14, each history phase owns one output-channel half of block 15,
and each frame phase owns one output-channel half of block 21.  Thus no
upsampled temporal parent is retained across a phase.  All learned values are
converted directly from source F32 to BF16 through ``_StorageOpenVINO``.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import torch

import plus_decoder
from frame_geometry import frame_pair_tiles


PHASE_NAMES = ("front", "history0", "history1", "frame0", "frame1")
HISTORY_STAGE_NAMES = ("grow0", "grow1", "memory17", "memory18", "memory19")
FUSED_MEMORY_STAGE_NAME = "memory17_19"
STATE_NAMES = plus_decoder.STATE_NAMES
CHANNELS = plus_decoder.CHANNELS
LATENT_CHANNELS = plus_decoder.LATENT_CHANNELS
STORAGE_PRECISION = "bf16"
MAX_SOURCE_DIM = 3840
MAX_SOURCE_HW = (MAX_SOURCE_DIM, MAX_SOURCE_DIM)
MAX_SOURCE_AREA = 2176 * 3840


def phased_geometry(source_hw: tuple[int, int] = plus_decoder.SOURCE_HW) -> dict[str, Any]:
    """Validate reviewed padded-UHD geometry and derive every phase port."""

    if (not isinstance(source_hw, (tuple, list)) or len(source_hw) != 2 or
            any(isinstance(value, bool) or not isinstance(value, Integral)
                for value in source_hw)):
        raise ValueError("source geometry must be a (height, width) pair of integers")
    height, width = (int(value) for value in source_hw)
    if height <= 0 or width <= 0 or height % 8 or width % 8:
        raise ValueError("decoder source dimensions must be positive multiples of 8")
    if (max(height, width) > MAX_SOURCE_DIM or height * width > MAX_SOURCE_AREA):
        raise ValueError("decoder geometry exceeds the reviewed padded-UHD bound")
    latent = (height // 8, width // 8)
    state_shapes = (
        (1, 512, *latent), (1, 512, *latent), (1, 512, *latent),
        (1, 256, height // 4, width // 4),
        (1, 256, height // 4, width // 4),
        (1, 256, height // 4, width // 4),
        (1, 128, height // 2, width // 2),
        (1, 128, height // 2, width // 2),
        (1, 128, height // 2, width // 2),
    )
    return {
        "source_hw": (height, width),
        "latent_hw": latent,
        "state_shapes": state_shapes,
    }


def _parameter(ov, name: str, shape: tuple[int, ...]):
    parameter = ov.opset1.parameter(list(shape), ov.Type.bf16, name=name)
    parameter.output(0).get_tensor().set_names({name})
    return parameter


def _checked_learned_factory(factory):
    """Normalize a source-owned BF16 Parameter node or output to its output.

    The caller owns source-identity/name uniqueness; the structural exporter
    checks those bindings against the authenticated source inventory.
    """

    if factory is None:
        return None
    if not callable(factory):
        raise TypeError("learned_parameter_factory must be callable")
    import openvino as ov

    def create(source, shape):
        parameter = factory(source, tuple(int(dim) for dim in shape))
        if parameter is None:
            raise TypeError("learned_parameter_factory must return a Parameter")
        node = parameter.get_node() if hasattr(parameter, "get_node") else parameter
        if node.get_type_name() != "Parameter":
            raise TypeError("learned_parameter_factory must return a Parameter output")
        if tuple(node.get_output_shape(0)) != tuple(int(dim) for dim in shape):
            raise ValueError("learned Parameter shape does not match the source slice")
        if node.get_output_element_type(0) != ov.Type.bf16:
            raise TypeError("learned Parameters must use BF16 storage")
        if not node.output(0).get_tensor().get_names():
            raise ValueError("learned Parameters require a non-empty tensor name")
        return node.output(0)

    return create


def _learned_parameter_nodes(cache: dict[tuple[Any, ...], Any]) -> list[Any]:
    """Return the unique ordinary Parameters registered by the callback."""

    nodes = []
    seen = set()
    for key, value in cache.items():
        if not (isinstance(key, tuple) and key and
                key[0] == "__learned_parameter_node__"):
            continue
        if value.get_type_name() != "Parameter":
            raise TypeError("learned callback cache contains a non-Parameter node")
        identity = id(value)
        if identity not in seen:
            seen.add(identity)
            nodes.append(value)
    return nodes


def _child_bank_descriptors(parameter, block: int, width: int,
                            shape: tuple[int, ...]) -> list[dict[str, Any]]:
    """Describe source weight slices without materializing BF16 bank arrays."""

    return [
        {
            "source_parameter": parameter,
            "source_parameter_name": f"decoder.{block}.conv.weight",
            "source_block": block,
            "child": child,
            "start": child * width,
            "stop": (child + 1) * width,
            "shape": tuple(int(dim) for dim in shape),
            "source_dtype": "f32",
            "storage": STORAGE_PRECISION,
        }
        for child in range(2)
    ]


def _descriptor_metadata(descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-safe descriptor fields while retaining object descriptors."""

    return [
        {key: value for key, value in descriptor.items()
         if key != "source_parameter"}
        for descriptor in descriptors
    ]


def _decorate(model, name: str, residual_start: int) -> int:
    import openvino as ov

    expected_residuals = {"front": 6, "history0": 3, "history1": 3,
                          "frame0": 0, "frame1": 0, "frame_pair": 0,
                          "grow0": 0, "grow1": 0,
                          "memory17": 1, "memory18": 1, "memory19": 1,
                          FUSED_MEMORY_STAGE_NAME: 3}[name]
    # The complete Plus graph and every phase use the same narrow source
    # storage helper.  It owns the F32 /3, BF16, Tanh, F32 *3, BF16 Clamp
    # sequence and the Conv+bias private stores; this phase only namespaces
    # those stores across phase graphs.
    plus_decoder.preserve_decoder_storage(
        model,
        precision=STORAGE_PRECISION,
        clamp_storage=False,
        residual_storage=True,
        expected_clamps=1 if name == "front" else 0,
        expected_residuals=expected_residuals,
    )
    for local in range(expected_residuals):
        port = model.output(f"residual_linear_{local}")
        port.set_names({f"residual_linear_{residual_start + local}"})
    model.validate_nodes_and_infer_types()
    model.set_rt_info("flashvsr-plus-tcdecoder-phased-explicit-state-v1",
                      ["conversion", "identity"])
    model.set_rt_info(STORAGE_PRECISION, ["conversion", "storage"])
    model.set_rt_info(name, ["conversion", "phase"])
    if any(port.get_element_type() != ov.Type.bf16
           for port in (*model.inputs, *model.outputs)):
        raise TypeError(f"{name} phase ports must remain BF16")
    return expected_residuals


def _growth_child(ov, native, proxy, block, value, child: int, cache, *, index: int = 15):
    """Lower one TGrow child without changing source arithmetic."""

    conv = block[index].conv
    if conv.out_channels % 2 or conv.groups != 1:
        raise ValueError("source temporal growth no longer has two channel halves")
    width_per_child = conv.out_channels // 2
    start = child * width_per_child
    stop = start + width_per_child
    result = native._native_convolution(
        proxy,
        value,
        conv,
        conv.weight[start:stop].contiguous(),
        cache,
        ("grow_weight", index, child),
        add_bias=False,
    )
    if conv.bias is not None:
        bias = conv.bias.detach()[start:stop].reshape(1, -1, 1, 1)
        result = ov.opset1.add(
            result,
            native._native_constant(proxy, cache, bias, ("grow_bias", index, child)),
        )
    return result


def _build_front_phase(
    decoder: torch.nn.Module,
    source_hw: tuple[int, int] | list[int],
    *,
    learned_parameter_factory=None,
    native=None,
    proxy=None,
    cache=None,
) -> tuple[Any, int, dict[str, Any]]:
    """Build only source blocks 0--13 and their six residual stores."""

    if not isinstance(decoder, torch.nn.Module) or not hasattr(decoder, "decoder"):
        raise TypeError("decoder must be the authenticated Plus TAEHV module")
    if any(parameter.dtype != torch.float32 for parameter in decoder.parameters()):
        raise ValueError("Plus source decoder must remain F32 until graph storage conversion")

    import openvino as ov

    geometry = phased_geometry(source_hw)
    shapes = geometry["state_shapes"]
    if native is None:
        native = plus_decoder._load_native_converter()
    if proxy is None:
        proxy = plus_decoder._StorageOpenVINO(STORAGE_PRECISION)
    if cache is None:
        cache = {}
    factory = _checked_learned_factory(learned_parameter_factory)
    block = decoder.decoder

    def module(value, index: int):
        return native._native_module(
            proxy, value, block[index], cache,
            learned_parameter_factory=factory,
        )

    def mem(value, past, index: int):
        return native._native_memblock(
            proxy, value, past, block[index], cache, combined_mem=True,
            learned_parameter_factory=factory,
        )

    packed = _parameter(ov, "packed", (1, LATENT_CHANNELS, *geometry["latent_hw"]))
    front_inputs = [_parameter(ov, name, tuple(shape))
                    for name, shape in zip(STATE_NAMES[:6], shapes[:6])]
    value = packed
    for index in range(5):
        value = module(value, index)
    front_states = []
    for index, past in zip((5, 6, 7), front_inputs[:3]):
        value, state = mem(value, past, index)
        front_states.append(state)
    for index in (8, 9, 10):
        value = module(value, index)
    for index, past in zip((11, 12, 13), front_inputs[3:]):
        value, state = mem(value, past, index)
        front_states.append(state)
    front_outputs = [("middle", value)] + [
        (f"{name}_out", state)
        for name, state in zip(STATE_NAMES[:6], front_states)
    ]
    results = []
    for output_name, output_value in front_outputs:
        result = ov.opset13.result(output_value)
        result.output(0).get_tensor().set_names({output_name})
        results.append(result)
    learned_nodes = _learned_parameter_nodes(cache)
    model = ov.Model(
        results,
        [packed, *front_inputs, *learned_nodes],
        "tcdecoder_phase_front",
    )
    model.validate_nodes_and_infer_types()
    residual_count = _decorate(model, "front", 0)
    input_names = [port.get_any_name() for port in model.inputs]
    output_names = [port.get_any_name() for port in model.outputs]
    expected_inputs = ["packed", *STATE_NAMES[:6]]
    expected_outputs = [
        "middle", *[f"{name}_out" for name in STATE_NAMES[:6]],
        *[f"residual_linear_{index}" for index in range(residual_count)],
    ]
    if factory is None:
        if input_names != expected_inputs:
            raise ValueError("front graph ABI changed")
    elif (input_names[:len(expected_inputs)] != expected_inputs or
          len(input_names) != len(expected_inputs) + len(learned_nodes)):
        raise ValueError("external front graph ABI changed")
    if output_names != expected_outputs:
        raise ValueError("front graph output ABI changed")
    if any(port.get_element_type() != ov.Type.bf16
           for port in (*model.inputs, *model.outputs)):
        raise TypeError("front phase ports must remain BF16")
    metadata = {
        "source_commit": plus_decoder.SOURCE_COMMIT,
        "source_sha256": plus_decoder.SOURCE_SHA256,
        "checkpoint_sha256": plus_decoder.CHECKPOINT_SHA256,
        "phase": "front",
        "source_hw": list(geometry["source_hw"]),
        "latent_hw": list(geometry["latent_hw"]),
        "storage": STORAGE_PRECISION,
        "source_dtype": "f32",
        "input_names": input_names,
        "output_names": output_names,
        "residual_count": residual_count,
        "residual_results": [f"residual_linear_{index}" for index in range(6)],
        "parameter_nodes": len(expected_inputs) + len(learned_nodes),
        "learned_parameter_nodes": len(learned_nodes),
        "learned_parameters_external": factory is not None,
    }
    return model, residual_count, metadata


def build_front(
    decoder: torch.nn.Module,
    source_hw: tuple[int, int] = plus_decoder.SOURCE_HW,
    *,
    learned_parameter_factory=None,
) -> tuple[Any, dict[str, Any]]:
    """Build only the front phase, optionally exposing learned BF16 weights."""

    model, _, metadata = _build_front_phase(
        decoder, source_hw, learned_parameter_factory=learned_parameter_factory,
    )
    return model, metadata


def build_growth_history(
    decoder: torch.nn.Module,
    tile_source_hw: tuple[int, int] | list[int],
    *,
    learned_parameter_factory=None,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Build one BF16 growth+history graph for one complete spatial tile.

    The only runtime learned input is the selected BF16 child half of source
    block 15 (``grow_weight``).  With ``learned_parameter_factory`` supplied,
    static source weights are ordinary BF16 Parameters and the returned child
    banks are source slice descriptors; no learned weight is materialized.
    The graph keeps the source block order and three MemBlock storage
    boundaries, while returning S0/S1/S2 explicitly for a controller's
    per-tile temporal lanes.  The default phased builders remain unchanged.
    """

    if not isinstance(decoder, torch.nn.Module) or not hasattr(decoder, "decoder"):
        raise TypeError("decoder must be the authenticated Plus TAEHV module")
    if any(parameter.dtype != torch.float32 for parameter in decoder.parameters()):
        raise ValueError("Plus source decoder must remain F32 until graph storage conversion")
    try:
        tile_geometry = phased_geometry(tile_source_hw)
    except (TypeError, ValueError) as exc:
        raise ValueError("tile_source_hw is not a valid decoder tile geometry") from exc
    tile_height, tile_width = tile_geometry["source_hw"]

    block = decoder.decoder
    if len(block) <= 19 or not hasattr(block[15], "conv"):
        raise ValueError("source decoder growth/history blocks changed")
    conv = block[15].conv
    shape = tuple(int(value) for value in conv.weight.shape)
    if (shape != (512, 256, 1, 1) or conv.bias is not None or conv.groups != 1
            or tuple(conv.kernel_size) != (1, 1)
            or tuple(conv.stride) != (1, 1)
            or tuple(conv.padding) != (0, 0)
            or tuple(conv.dilation) != (1, 1)
            or conv.weight.dtype != torch.float32):
        raise ValueError("source block15 TGrow ABI changed")

    import openvino as ov
    factory = _checked_learned_factory(learned_parameter_factory)
    if factory is None:
        # The historical return is intentionally retained byte-for-byte: the
        # controller owns these BF16 runtime banks and its source-auth checks
        # depend on the arrays being materialized here.
        from plus_storage import is_finite_bf16, storage_tensor

        source_weight = np.ascontiguousarray(conv.weight.detach().cpu().numpy())
        banks = []
        for child in range(2):
            start, stop = child * 256, (child + 1) * 256
            tensor = storage_tensor(source_weight[start:stop], STORAGE_PRECISION)
            bits = np.array(tensor.data.view(np.uint16), copy=True, order="C")
            if bits.shape != (256, 256, 1, 1) or not is_finite_bf16(bits):
                raise ValueError("source block15 child bank is not finite BF16 storage")
            banks.append(bits)
    else:
        # Exporters authenticate the source Parameter and perform conversion at
        # their own boundary.  Never materialize learned weights on this path.
        banks = _child_bank_descriptors(
            conv.weight, 15, 256, (256, 256, 1, 1)
        )

    native = plus_decoder._load_native_converter()
    proxy = plus_decoder._StorageOpenVINO(STORAGE_PRECISION)
    cache: dict[tuple[Any, ...], Any] = {}
    # Serialized input/output names must be distinct or the IR reader silently
    # uniquifies one. Preserve the historical in-memory ABI on the default path.
    compact_name = "middle_in" if factory is not None else "middle"
    compact = _parameter(
        ov, compact_name, (1, 256, tile_height // 4, tile_width // 4)
    )
    growth_weight = _parameter(ov, "grow_weight", (256, 256, 1, 1))
    value = native._native_module(
        proxy, compact, block[14], cache,
        learned_parameter_factory=factory,
    )
    value = ov.opset1.convolution(
        value, growth_weight,
        list(conv.stride), list(conv.padding), list(conv.padding), list(conv.dilation),
    )
    value = native._native_module(
        proxy, value, block[16], cache,
        learned_parameter_factory=factory,
    )
    states = []
    pasts = [
        _parameter(ov, name, (1, 128, tile_height // 2, tile_width // 2))
        for name in STATE_NAMES[6:]
    ]
    s0 = value
    for index, past in zip((17, 18, 19), pasts):
        value, state = native._native_memblock(
            proxy, value, past, block[index], cache, combined_mem=True,
            learned_parameter_factory=factory,
        )
        states.append(state)
    outputs = []
    for name, port_value in (
        ("S0", s0), ("S1", states[1]), ("S2", states[2]), ("middle", value)
    ):
        result = ov.opset13.result(port_value)
        result.output(0).get_tensor().set_names({name})
        outputs.append(result)
    learned_nodes = _learned_parameter_nodes(cache)
    model = ov.Model(
        outputs, [compact, growth_weight, *pasts, *learned_nodes],
        "tcdecoder_growth_history_tile",
    )
    model.validate_nodes_and_infer_types()
    _decorate(model, FUSED_MEMORY_STAGE_NAME, 6)
    model.set_rt_info("grow_memory", ["conversion", "phase"])

    input_names = [port.get_any_name() for port in model.inputs]
    output_names = [port.get_any_name() for port in model.outputs]
    expected_inputs = [compact_name, "grow_weight", *STATE_NAMES[6:]]
    expected_outputs = [
        "S0", "S1", "S2", "middle",
        "residual_linear_6", "residual_linear_7", "residual_linear_8",
    ]
    base_inputs_match = input_names[:len(expected_inputs)] == expected_inputs
    if ((factory is None and input_names != expected_inputs) or
            (factory is not None and
             (not base_inputs_match or
              len(input_names) != len(expected_inputs) + len(learned_nodes))) or
            output_names != expected_outputs):
        raise ValueError("growth/history graph ABI changed")
    if any(port.get_element_type() != ov.Type.bf16
           for port in (*model.inputs, *model.outputs)):
        raise ValueError("growth/history graph ports must remain BF16")
    if len([node for node in model.get_ordered_ops()
            if node.get_type_name() == "Parameter"]) != 5 + len(learned_nodes):
        raise ValueError("growth/history graph Parameter count changed")
    targets = growth_weight.output(0).get_target_inputs()
    if len(targets) != 1:
        raise ValueError("growth weight must feed exactly one convolution")
    target = next(iter(targets))
    if target.get_index() != 1 or target.get_node().get_type_name() != "Convolution":
        raise ValueError("growth weight is not the convolution filter input")

    metadata = {
        "source_commit": plus_decoder.SOURCE_COMMIT,
        "source_sha256": plus_decoder.SOURCE_SHA256,
        "checkpoint_sha256": plus_decoder.CHECKPOINT_SHA256,
        "phase": "grow_memory",
        "tile_source_hw": [tile_height, tile_width],
        "source_hw": list((tile_height, tile_width)),
        "compact_shape": [1, 256, tile_height // 4, tile_width // 4],
        "state_shape": [1, 128, tile_height // 2, tile_width // 2],
        "storage": STORAGE_PRECISION,
        "source_dtype": "f32",
        "input_names": input_names,
        "output_names": expected_outputs,
        "parameter_nodes": 5 + len(learned_nodes),
        "learned_parameter_nodes": len(learned_nodes),
        "learned_parameters_external": factory is not None,
        "growth_weight_input": "grow_weight",
        "growth_source_block": 15,
        "history_blocks": [17, 18, 19],
        "child_bank_count": 2,
        "child_bank_shape": [256, 256, 1, 1],
        "child_bank_dtype": "uint16_bf16_storage",
        "child_bank_mode": ("bf16_arrays" if factory is None
                             else "source_parameter_slices"),
        "child_bank_descriptors": ([] if factory is None
                                   else _descriptor_metadata(banks)),
        "child_bank_bytes_total": (
            int(sum(bank.nbytes for bank in banks)) if factory is None
            else 2 * 256 * 256 * 2
        ),
        "combined_mem": True,
        "residual_count": 3,
        "residual_results": ["residual_linear_6", "residual_linear_7",
                             "residual_linear_8"],
        "private_results": ["residual_linear_6", "residual_linear_7",
                             "residual_linear_8"],
    }
    return model, banks, metadata


def build_remote_frame_child(
    decoder: torch.nn.Module,
    tile_source_hw: tuple[int, int] | list[int],
    *,
    learned_parameter_factory=None,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Build one BF16 frame child with a runtime block-21 bank.

    This is the qualified sequential-child boundary: block 20 and blocks
    22--26 remain source-owned graph operations, while one selected half of
    block 21 is supplied as a BF16 storage-bit input.  The default returned
    banks are direct source-F32-to-BF16 storage conversions; with an external
    factory they are source slice descriptors, so no numeric FP16 path or
    learned-weight materialization is introduced.
    """

    if (not isinstance(decoder, torch.nn.Module) or
            not hasattr(decoder, "decoder")):
        raise TypeError("decoder must be the authenticated Plus TAEHV module")
    if any(parameter.dtype != torch.float32 for parameter in decoder.parameters()):
        raise ValueError("Plus source decoder must remain F32 until graph storage conversion")
    geometry = phased_geometry(tile_source_hw)
    height, width = geometry["source_hw"]
    block = decoder.decoder
    if len(block) <= 26 or not hasattr(block[21], "conv"):
        raise ValueError("source frame blocks changed")
    conv = block[21].conv
    if (tuple(int(value) for value in conv.weight.shape) != (256, 128, 1, 1)
            or conv.bias is not None or conv.groups != 1
            or tuple(conv.kernel_size) != (1, 1)
            or tuple(conv.stride) != (1, 1)
            or tuple(conv.padding) != (0, 0)
            or tuple(conv.dilation) != (1, 1)
            or conv.weight.dtype != torch.float32):
        raise ValueError("source block-21 child-bank ABI changed")

    import openvino as ov
    factory = _checked_learned_factory(learned_parameter_factory)
    if factory is None:
        from plus_storage import is_finite_bf16, storage_tensor

        source_weight = np.ascontiguousarray(conv.weight.detach().cpu().numpy())
        banks = []
        for child in (0, 1):
            start, stop = child * 128, (child + 1) * 128
            bits = np.array(
                storage_tensor(source_weight[start:stop], STORAGE_PRECISION)
                .data.view(np.uint16), copy=True, order="C"
            )
            if bits.shape != (128, 128, 1, 1) or not is_finite_bf16(bits):
                raise ValueError("source block-21 child bank is not finite BF16 storage")
            banks.append(bits)
    else:
        banks = _child_bank_descriptors(
            conv.weight, 21, 128, (128, 128, 1, 1)
        )

    native = plus_decoder._load_native_converter()
    proxy = plus_decoder._StorageOpenVINO(STORAGE_PRECISION)
    cache: dict[tuple[Any, ...], Any] = {}
    middle = _parameter(ov, "middle", (1, 128, height // 2, width // 2))
    frame_input_name = "frame_grow" if factory is not None else "weight__frame_grow"
    frame_weight = _parameter(ov, frame_input_name, (128, 128, 1, 1))
    value = native._native_module(
        proxy, middle, block[20], cache,
        learned_parameter_factory=factory,
    )
    value = ov.opset1.convolution(
        value, frame_weight,
        list(conv.stride), list(conv.padding), list(conv.padding),
        list(conv.dilation),
    )
    for index in (22, 23, 24, 25, 26):
        value = native._native_module(
            proxy, value, block[index], cache,
            learned_parameter_factory=factory,
        )
    result = ov.opset13.result(value)
    result.output(0).get_tensor().set_names({"frame"})
    learned_nodes = _learned_parameter_nodes(cache)
    model = ov.Model(
        [result], [middle, frame_weight, *learned_nodes],
        "tcdecoder_remote_frame_child",
    )
    model.validate_nodes_and_infer_types()
    # Keep the qualified proof's storage/decorator sequence exactly: the
    # logical phase is remote_frame_child, while the source storage decorator
    # uses the existing zero-residual frame0 contract.
    _decorate(model, "frame0", 0)

    expected_shape = (1, 3, height, width)
    input_names = [port.get_any_name() for port in model.inputs]
    expected_base_inputs = ["middle", frame_input_name]
    if ((factory is None and input_names != expected_base_inputs) or
            (factory is not None and
             (input_names[:len(expected_base_inputs)] != expected_base_inputs or
              len(input_names) != len(expected_base_inputs) + len(learned_nodes))) or
            [port.get_any_name() for port in model.outputs] != ["frame"] or
            tuple(model.output("frame").get_shape()) != expected_shape):
        raise TypeError("remote frame graph ABI changed")
    if any(port.get_element_type() != ov.Type.bf16
           for port in (*model.inputs, *model.outputs)):
        raise TypeError("remote frame graph ports must remain BF16")
    target = frame_weight.output(0).get_target_inputs()
    if (len(target) != 1 or
            next(iter(target)).get_node().get_type_name() != "Convolution" or
            next(iter(target)).get_index() != 1):
        raise TypeError("frame child bank is not the convolution filter input")
    if len([node for node in model.get_ordered_ops()
            if node.get_type_name() == "Interpolate"]) != 1:
        raise TypeError("remote frame graph must contain exactly block-20 upsample")

    metadata = {
        "source_commit": plus_decoder.SOURCE_COMMIT,
        "source_sha256": plus_decoder.SOURCE_SHA256,
        "checkpoint_sha256": plus_decoder.CHECKPOINT_SHA256,
        "phase": "remote_frame_child",
        "decorator_phase": "frame0",
        "tile_source_hw": [height, width],
        "middle_shape": [1, 128, height // 2, width // 2],
        "frame_shape": list(expected_shape),
        "input_names": input_names,
        "output_names": ["frame"],
        "source_blocks": [20, 21, 22, 23, 24, 25, 26],
        "dynamic_block": 21,
        "dynamic_bank_shape": [128, 128, 1, 1],
        "dynamic_bank_count": 2,
        "dynamic_bank_dtype": "uint16_bf16_storage",
        "dynamic_bank_bytes_total": (
            int(sum(bank.nbytes for bank in banks)) if factory is None
            else 2 * 128 * 128 * 2
        ),
        "dynamic_bank_mode": ("bf16_arrays" if factory is None
                               else "source_parameter_slices"),
        "child_bank_mode": ("bf16_arrays" if factory is None
                             else "source_parameter_slices"),
        "child_bank_descriptors": ([] if factory is None
                                   else _descriptor_metadata(banks)),
        "learned_parameter_nodes": len(learned_nodes),
        "learned_parameters_external": factory is not None,
        "storage": STORAGE_PRECISION,
        "source_dtype": "f32",
        "storage_boundary": "BF16 middle -> BF16 frame input",
        "interpolate_nodes": 1,
        "residual_count": 0,
        "residual_results": [],
        "private_results": 0,
    }
    return model, banks, metadata


def build_frame_pair(
    decoder: torch.nn.Module,
    source_hw: tuple[int, int] = plus_decoder.SOURCE_HW,
) -> tuple[Any, dict[str, Any]]:
    """Build one shared-block-20 BF16 graph with two frame results."""

    if not isinstance(decoder, torch.nn.Module) or not hasattr(decoder, "decoder"):
        raise TypeError("decoder must be the authenticated Plus TAEHV module")
    if any(parameter.dtype != torch.float32 for parameter in decoder.parameters()):
        raise ValueError("Plus source decoder must remain F32 until graph storage conversion")
    import openvino as ov

    tiles = frame_pair_tiles(source_hw)
    output_shape = (1, 3, *tiles["tile_hw"])
    native = plus_decoder._load_native_converter()
    proxy = plus_decoder._StorageOpenVINO(STORAGE_PRECISION)
    block = decoder.decoder
    cache: dict[tuple[Any, ...], Any] = {}
    middle = _parameter(ov, "middle", tuple(tiles["middle_shape"]))
    value = native._native_module(proxy, middle, block[20], cache)
    outputs = []
    for child in (0, 1):
        branch = _growth_child(ov, native, proxy, block, value, child, cache, index=21)
        for index in (22, 23, 24, 25, 26):
            branch = native._native_module(proxy, branch, block[index], cache)
        outputs.append(branch)
    results = []
    for name, branch in zip(("frame0", "frame1"), outputs):
        result = ov.opset13.result(branch)
        result.output(0).get_tensor().set_names({name})
        results.append(result)
    model = ov.Model(results, [middle], "tcdecoder_phase_frame_pair")
    model.validate_nodes_and_infer_types()
    _decorate(model, "frame_pair", 0)
    if ([port.get_any_name() for port in model.inputs] != ["middle"] or
            [port.get_any_name() for port in model.outputs] != ["frame0", "frame1"] or
            tuple(model.input("middle").get_shape()) != tuple(tiles["middle_shape"]) or
            any(tuple(model.output(name).get_shape()) != output_shape
                for name in ("frame0", "frame1")) or
            any(port.get_element_type() != ov.Type.bf16
                for port in (*model.inputs, *model.outputs))):
        raise TypeError("frame-pair graph ABI is not the authenticated BF16 contract")
    if (len([node for node in model.get_ordered_ops()
             if node.get_type_name() == "Interpolate"]) != 1 or
            len([node for node in model.get_ordered_ops()
                 if node.get_type_name() == "Result"]) != 2):
        raise ValueError("frame-pair graph topology changed")
    metadata = {
        "source_commit": plus_decoder.SOURCE_COMMIT,
        "source_sha256": plus_decoder.SOURCE_SHA256,
        "checkpoint_sha256": plus_decoder.CHECKPOINT_SHA256,
        "source_dtype": "f32", "storage": STORAGE_PRECISION,
        "source_hw": tiles["source_hw"], "tile_hw": tiles["tile_hw"],
        "middle_shape": list(tiles["middle_shape"]), "output_shape": list(output_shape),
        "input_names": ["middle"], "output_names": ["frame0", "frame1"],
        "shared_source_block": 20, "split_source_block": 21,
        "branch_source_blocks": [22, 23, 24, 25, 26],
        "interpolate_nodes": 1, "result_nodes": 2, "private_results": 0,
    }
    return model, metadata


def build_phased_decoder(
    decoder: torch.nn.Module,
    source_hw: tuple[int, int] = plus_decoder.SOURCE_HW,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build exactly ``front/history0/history1/frame0/frame1`` in BF16.

    ``decoder`` must be the authenticated source-F32 module returned by
    :func:`plus_decoder.load_decoder`.  This builder intentionally has one
    storage mode: direct BF16.
    """

    if not isinstance(decoder, torch.nn.Module) or not hasattr(decoder, "decoder"):
        raise TypeError("decoder must be the authenticated Plus TAEHV module")
    if any(parameter.dtype != torch.float32 for parameter in decoder.parameters()):
        raise ValueError("Plus source decoder must remain F32 until graph storage conversion")

    import openvino as ov

    geometry = phased_geometry(source_hw)
    height, width = geometry["source_hw"]
    shapes = geometry["state_shapes"]
    native = plus_decoder._load_native_converter()
    proxy = plus_decoder._StorageOpenVINO(STORAGE_PRECISION)
    block = decoder.decoder
    cache: dict[tuple[Any, ...], Any] = {}

    def module(value, index: int):
        return native._native_module(proxy, value, block[index], cache)

    def mem(value, past, index: int):
        # Combined mode preserves the source one-convolution boundary and
        # avoids the split-conv accumulation path that is not source-exact.
        return native._native_memblock(
            proxy, value, past, block[index], cache, combined_mem=True
        )

    def make_model(name: str, outputs, parameters, residual_start: int):
        results = []
        for output_name, value in outputs:
            result = ov.opset13.result(value)
            result.output(0).get_tensor().set_names({output_name})
            results.append(result)
        model = ov.Model(results, parameters, f"tcdecoder_phase_{name}")
        model.validate_nodes_and_infer_types()
        count = _decorate(model, name, residual_start)
        return model, count

    # Front construction is shared with build_front so an external exporter
    # can request only this phase without instantiating historical branches.
    front, front_residuals, _ = _build_front_phase(
        decoder, source_hw, native=native, proxy=proxy, cache=cache,
    )

    models: dict[str, Any] = {"front": front}
    residual_count = front_residuals

    # The two history phases share the same compact parent but receive the
    # updated 128-channel history in sequence: history0 then history1.
    for child in range(2):
        middle = _parameter(ov, "middle", (1, 256, height // 4, width // 4))
        value = module(middle, 14)
        value = _growth_child(ov, native, proxy, block, value, child, cache)
        value = module(value, 16)
        history_inputs = [
            _parameter(ov, name, tuple(shape))
            for name, shape in zip(STATE_NAMES[6:], shapes[6:])
        ]
        history_states = []
        for index, past in zip((17, 18, 19), history_inputs):
            value, state = mem(value, past, index)
            history_states.append(state)
        history_outputs = [("middle", value)] + [
            (f"{name}_out", state)
            for name, state in zip(STATE_NAMES[6:], history_states)
        ]
        name = f"history{child}"
        history, count = make_model(
            name, history_outputs, [middle, *history_inputs], residual_start=residual_count
        )
        models[name] = history
        residual_count += count

    # Each frame phase consumes one compact history result, owns block 20 and
    # one channel half of block 21, then emits the exact source frame.
    for child in range(2):
        middle = _parameter(ov, "middle", (1, 128, height // 2, width // 2))
        value = module(middle, 20)
        value = _growth_child(ov, native, proxy, block, value, child, cache, index=21)
        for index in (22, 23, 24, 25, 26):
            value = module(value, index)
        name = f"frame{child}"
        frame, count = make_model(
            name, [("frame", value)], [middle], residual_start=residual_count
        )
        models[name] = frame
        residual_count += count

    if set(models) != set(PHASE_NAMES):
        raise AssertionError("phased decoder model set changed")
    if residual_count != 12:
        raise ValueError(f"Expected 12 source Conv residual boundaries, got {residual_count}")

    metadata = {
        "source_commit": plus_decoder.SOURCE_COMMIT,
        "source_sha256": plus_decoder.SOURCE_SHA256,
        "checkpoint_sha256": plus_decoder.CHECKPOINT_SHA256,
        "source_hw": list(geometry["source_hw"]),
        "latent_hw": list(geometry["latent_hw"]),
        "state_names": list(STATE_NAMES),
        "state_shapes": [list(shape) for shape in shapes],
        "phase_names": list(PHASE_NAMES),
        "storage": STORAGE_PRECISION,
        "combined_mem": True,
        "split_growth": True,
        "fused_history": True,
        "fused_frames": True,
        "front_stops_after_block": 13,
        "history_starts_block": 14,
        "history_stops_after_block": 19,
        "frame_starts_block": 20,
        "residual_output_names": [f"residual_linear_{i}" for i in range(12)],
        "branch_order": [
            "history0/frame0", "history0/frame1",
            "history1/frame0", "history1/frame1",
        ],
    }
    return models, metadata


def build_history_stages(
    decoder: torch.nn.Module,
    source_hw: tuple[int, int] = plus_decoder.SOURCE_HW,
    *,
    fuse_memories: bool = False,
    memory_source_hw: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the opt-in low-memory growth and MemBlock stages.

    ``grow0``/``grow1`` perform source blocks 14, 15 (one TGrow child), and
    16.  ``memory17``/``memory18``/``memory19`` each perform exactly one
    source MemBlock.  A memory graph returns only its computed middle and the
    preserved source Conv result; its ``current`` input is the history value
    to retain for the next temporal call, so no duplicate state Result is
    emitted.  With ``fuse_memories=True`` the three MemBlocks are one graph:
    ``mem_128_0`` is the current input and therefore needs no duplicate output;
    ``mem_128_1_out`` and ``mem_128_2_out`` are the next two previous-input
    states. ``memory_source_hw`` bounds only the fused history graph; growth
    still produces full parents. The default remains the three-graph ABI above.
    """

    if type(fuse_memories) is not bool:
        raise ValueError("fuse_memories must be a boolean")
    if memory_source_hw is not None and not fuse_memories:
        raise ValueError("memory_source_hw requires fuse_memories=True")
    if not isinstance(decoder, torch.nn.Module) or not hasattr(decoder, "decoder"):
        raise TypeError("decoder must be the authenticated Plus TAEHV module")
    if any(parameter.dtype != torch.float32 for parameter in decoder.parameters()):
        raise ValueError("Plus source decoder must remain F32 until graph storage conversion")

    import openvino as ov

    geometry = phased_geometry(source_hw)
    height, width = geometry["source_hw"]
    middle_shape = (1, 128, height // 2, width // 2)
    memory_geometry = phased_geometry(memory_source_hw) if memory_source_hw is not None else geometry
    memory_shape = (1, 128, memory_geometry["source_hw"][0] // 2,
                    memory_geometry["source_hw"][1] // 2)
    native = plus_decoder._load_native_converter()
    proxy = plus_decoder._StorageOpenVINO(STORAGE_PRECISION)
    block = decoder.decoder
    cache: dict[tuple[Any, ...], Any] = {}

    def module(value, index: int):
        return native._native_module(proxy, value, block[index], cache)

    def make_model(name: str, outputs, parameters, residual_start: int):
        results = []
        for output_name, value in outputs:
            result = ov.opset13.result(value)
            result.output(0).get_tensor().set_names({output_name})
            results.append(result)
        model = ov.Model(results, parameters, f"tcdecoder_history_stage_{name}")
        model.validate_nodes_and_infer_types()
        _decorate(model, name, residual_start)
        return model

    models: dict[str, Any] = {}
    for child in (0, 1):
        middle = _parameter(ov, "middle", (1, 256, height // 4, width // 4))
        value = module(middle, 14)
        value = _growth_child(ov, native, proxy, block, value, child, cache)
        value = module(value, 16)
        models[f"grow{child}"] = make_model(
            f"grow{child}", [("middle", value)], [middle], residual_start=0
        )

    if fuse_memories:
        current = _parameter(ov, "current", memory_shape)
        pasts = [_parameter(ov, name, memory_shape) for name in STATE_NAMES[6:]]
        value = current
        states = []
        for index, past in zip((17, 18, 19), pasts):
            value, state = native._native_memblock(
                proxy, value, past, block[index], cache, combined_mem=True
            )
            states.append(state)
        fused_outputs = [("middle", value),
                         (f"{STATE_NAMES[7]}_out", states[1]),
                         (f"{STATE_NAMES[8]}_out", states[2])]
        models[FUSED_MEMORY_STAGE_NAME] = make_model(
            FUSED_MEMORY_STAGE_NAME, fused_outputs, [current, *pasts],
            residual_start=6,
        )
        stage_names = ("grow0", "grow1", FUSED_MEMORY_STAGE_NAME)
    else:
        for index in (17, 18, 19):
            current = _parameter(ov, "current", middle_shape)
            past = _parameter(ov, "past", middle_shape)
            value, _ = native._native_memblock(
                proxy, current, past, block[index], cache, combined_mem=True
            )
            models[f"memory{index}"] = make_model(
                f"memory{index}", [("middle", value)], [current, past],
                # Reuse the existing history0 private-result ABI.  The three
                # models are source blocks 17/18/19, but the controller already
                # consumes their names as residual_linear_6/7/8.
                residual_start=index - 11,
            )
        stage_names = HISTORY_STAGE_NAMES

    if set(models) != set(stage_names):
        raise AssertionError("history stage model set changed")
    if fuse_memories:
        memory_outputs = {
            FUSED_MEMORY_STAGE_NAME: [
                "middle", f"{STATE_NAMES[7]}_out", f"{STATE_NAMES[8]}_out",
                "residual_linear_6", "residual_linear_7", "residual_linear_8",
            ]
        }
        history_contract = (
            "fused memory17/18/19; mem_128_0 is current input, outputs retain "
            "the previous-input states mem_128_1 and mem_128_2"
        )
    else:
        memory_outputs = {
            f"memory{index}": ["middle", f"residual_linear_{index - 11}"]
            for index in (17, 18, 19)
        }
        history_contract = "retain each memory current input as its next past; no state Result"
    metadata = {
        "source_commit": plus_decoder.SOURCE_COMMIT,
        "source_sha256": plus_decoder.SOURCE_SHA256,
        "checkpoint_sha256": plus_decoder.CHECKPOINT_SHA256,
        "source_hw": list(geometry["source_hw"]),
        "latent_hw": list(geometry["latent_hw"]),
        "stage_names": list(stage_names),
        "storage": STORAGE_PRECISION,
        "source_dtype": "f32",
        "combined_mem": True,
        "growth_blocks": [14, 15, 16],
        "memory_blocks": [17, 18, 19],
        "growth_output_shape": list(middle_shape),
        "memory_input_shape": list(memory_shape),
        "memory_outputs": memory_outputs,
        "history_contract": history_contract,
        "growth_order": ["grow0", "grow1"],
    }
    if fuse_memories:
        metadata["fuse_memories"] = True
    if memory_source_hw is not None:
        metadata["memory_source_hw"] = list(memory_geometry["source_hw"])
    return models, metadata


def load_phased_decoder(
    source_hw: tuple[int, int] = plus_decoder.SOURCE_HW,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the official F32 decoder, then build its BF16 phases."""

    decoder, state, metadata = plus_decoder.load_decoder()
    models, phase_metadata = build_phased_decoder(decoder, source_hw)
    phase_metadata = {**phase_metadata, "decoder_metadata": metadata,
                      "parameter_count": len(state)}
    return models, phase_metadata


__all__ = [
    "CHANNELS", "LATENT_CHANNELS", "MAX_SOURCE_AREA", "MAX_SOURCE_HW",
    "FUSED_MEMORY_STAGE_NAME", "HISTORY_STAGE_NAMES", "PHASE_NAMES",
    "STATE_NAMES", "STORAGE_PRECISION",
    "build_frame_pair", "build_front", "build_growth_history",
    "build_remote_frame_child",
    "build_history_stages",
    "build_phased_decoder",
    "load_phased_decoder", "phased_geometry",
]
