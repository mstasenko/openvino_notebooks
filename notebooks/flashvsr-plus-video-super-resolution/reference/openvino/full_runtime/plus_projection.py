"""OpenVINO FlashVSR+ LQ projection graphs.

The official Plus checkpoint is stored as F32 safetensors and the official
loader casts the module once to the selected execution dtype.  This helper
keeps that boundary explicit: constants are made directly from the raw F32
values as F16 or BF16, while ChannelRMSNorm preserves the source's intermediate
storage rounding. Startup and steady graphs expose the same causal cache
ABI as ``models.conditioning.LQProjection.stream_forward``.
"""

from __future__ import annotations

import gc
from pathlib import Path
import hashlib
from numbers import Integral
import time

import numpy as np

from plus_storage import storage_tensor, tensor_array
from plus_modes import SOURCE_COMMIT, resolve_plus_mode
from reference_paths import LQ_CHECKPOINT, SOURCE_ROOT


HERE = Path(__file__).resolve()
BASE = HERE.parents[2]
PLUS_SOURCE = SOURCE_ROOT
CHECKPOINT = LQ_CHECKPOINT
CHECKPOINT_SHA256 = resolve_plus_mode().lq_sha256

WEIGHT_KEYS = (
    "conv1.bias",
    "conv1.weight",
    "conv2.bias",
    "conv2.weight",
    "linear_layers.0.bias",
    "linear_layers.0.weight",
    "norm1.weight",
    "norm2.weight",
)
WEIGHT_SHAPES = {
    "conv1.bias": (2048,),
    "conv1.weight": (2048, 768, 4, 3, 3),
    "conv2.bias": (3072,),
    "conv2.weight": (3072, 2048, 4, 3, 3),
    "linear_layers.0.bias": (1536,),
    "linear_layers.0.weight": (1536, 3072),
    "norm1.weight": (2048, 1, 1, 1),
    "norm2.weight": (3072, 1, 1, 1),
}
STORAGE_PRECISIONS = {"bf16", "f16"}

# The native learned-port ABI is deliberately derived from the safetensors
# keys.  Keeping this mapping here makes the build-time external-weight graph
# use the same names as the native Flash stage without changing the checked
# constant-weight path.
WEIGHT_PORTS = {key: "weight__" + key.replace(".", "_") for key in WEIGHT_KEYS}
STARTUP_WEIGHT_KEYS = ("conv1.bias", "conv1.weight", "norm1.weight")
STEADY_WEIGHT_KEYS = WEIGHT_KEYS
STAGE2_WEIGHT_KEYS = (
    "conv2.bias", "conv2.weight", "linear_layers.0.bias",
    "linear_layers.0.weight", "norm2.weight",
)


def sha256_path(path: Path = CHECKPOINT) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_checkpoint(path: Path = CHECKPOINT, *, mode: str = "turbo") -> str:
    """Require the exact released LQ bytes for the explicitly selected mode."""

    spec = resolve_plus_mode(mode)
    if Path(path).stat().st_size != spec.lq_bytes:
        raise ValueError(f"Plus {mode} LQ checkpoint size changed")
    observed = sha256_path(path)
    if observed != spec.lq_sha256:
        raise ValueError(f"Plus LQ checkpoint identity changed: {observed}")
    return observed


def load_weights(
    path: Path = CHECKPOINT,
    *,
    keys: tuple[str, ...] | None = None,
    authenticate: bool = True,
    mode: str = "turbo",
) -> dict[str, np.ndarray]:
    """Read authenticated raw F32 weights without an intermediate storage cast.

    ``keys`` is useful for startup-only graph construction, avoiding a second
    copy of the 1.1 GB checkpoint.  The complete key set and every selected
    shape/dtype are still checked against the official LQProjection ABI.
    """

    path = Path(path)
    if authenticate:
        authenticate_checkpoint(path, mode=mode)
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - environment contract
        raise ImportError("safetensors is required for the Plus LQ checkpoint") from error

    requested = WEIGHT_KEYS if keys is None else tuple(keys)
    if not requested or any(key not in WEIGHT_SHAPES for key in requested):
        raise ValueError("requested LQ weight keys are not allow-listed")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate LQ weight key")
    with safe_open(str(path), framework="np", device="cpu") as handle:
        actual = tuple(sorted(handle.keys()))
        if set(actual) != set(WEIGHT_KEYS):
            raise ValueError(f"Plus LQ checkpoint keys changed: {actual}")
        result: dict[str, np.ndarray] = {}
        for key in requested:
            value = np.asarray(handle.get_tensor(key))
            if tuple(value.shape) != WEIGHT_SHAPES[key] or value.dtype != np.float32:
                raise ValueError(
                    f"invalid Plus LQ tensor {key}: shape={value.shape}, dtype={value.dtype}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite Plus LQ tensor: {key}")
            # Detach from safetensors' mapped storage before closing the file.
            result[key] = np.array(value, dtype=np.float32, copy=True)
    return result


def projection_geometry(height: int = 128, width: int = 128) -> dict[str, int]:
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, Integral)
        or not isinstance(width, Integral)
    ):
        raise ValueError("height and width must be positive integer dimensions")
    height, width = int(height), int(width)
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("height and width must be positive multiples of 16")
    return {
        "height": height,
        "width": width,
        "token_h": height // 16,
        "token_w": width // 16,
        "token_count": (height // 16) * (width // 16),
    }


def _i64(ops, value):
    return ops.constant(np.asarray(value, dtype=np.int64))


def build_projection_graph(
    weights: dict[str, np.ndarray] | None,
    startup: bool,
    precision: str = "bf16",
    *,
    height: int = 128,
    width: int = 128,
):
    """Build one static startup/steady graph for official LQProjection math.

    A mapping contains the historical checked Constants.  ``weights=None`` is
    an explicit build-time mode that exposes the reachable learned weights as
    BF16 Parameters using the native ``weight__...`` ABI; it never reads or
    embeds learned payloads.
    """

    import openvino as ov
    from openvino import opset13 as ops

    if precision not in STORAGE_PRECISIONS:
        raise ValueError("Plus LQ graph precision must be 'bf16' or 'f16'")
    geometry = projection_geometry(height, width)
    needed = STARTUP_WEIGHT_KEYS if startup else STEADY_WEIGHT_KEYS
    if weights is not None:
        if not isinstance(weights, dict) or set(weights) != set(needed):
            scope = "startup" if startup else "steady"
            raise ValueError(f"Plus LQ {scope} graph received the wrong weight set")
        for key in needed:
            shape = WEIGHT_SHAPES[key]
            value = np.asarray(weights[key])
            if tuple(value.shape) != shape or value.dtype != np.float32:
                raise ValueError(f"invalid graph weight {key}: {value.shape}, {value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite graph weight {key}")
    elif precision != "bf16":
        raise ValueError("external Plus LQ weights require BF16 Parameters")

    storage = ov.Type.bf16 if precision == "bf16" else ov.Type.f16
    f32 = ov.Type.f32

    def constant(value):
        return ops.constant(storage_tensor(value, precision))

    def raw_constant(value):
        return ops.constant(np.asarray(value, dtype=np.float32))

    weight_parameters = {}
    if weights is None:
        for key in needed:
            port_name = WEIGHT_PORTS[key]
            parameter = ops.parameter(WEIGHT_SHAPES[key], storage, name=port_name)
            parameter.output(0).get_tensor().set_names({port_name})
            weight_parameters[key] = parameter.output(0)

    def weight(name):
        if weights is None:
            return weight_parameters[name]
        return constant(weights[name])

    def reshape(value, shape):
        return ops.reshape(value, _i64(ops, shape), False)

    def channel_norm(value, gamma_name: str, channels: int):
        # F.normalize receives storage-typed input: its norm and division
        # each round before scalar scaling and learned gamma multiplication.
        denominator = ops.reduce_l2(value, _i64(ops, [1]), True)
        denominator = ops.maximum(denominator, constant(np.float32(1e-12)))
        normalized = ops.divide(value, denominator)
        scaled = ops.convert(ops.multiply(ops.convert(normalized, f32),
                                         raw_constant(np.float32(channels ** 0.5))), storage)
        gamma = ops.convert(weight(gamma_name), f32)
        normalized_gamma = ops.convert(ops.multiply(ops.convert(scaled, f32), gamma), storage)
        return normalized_gamma

    def convolution(value, prefix: str, temporal_pad: int):
        padded = ops.pad(
            value,
            _i64(ops, [0, 0, temporal_pad, 1, 1]),
            _i64(ops, [0, 0, 0, 1, 1]),
            "edge",
            constant(np.float32(0.0)),
        )
        result = ops.convolution(
            padded,
            weight(prefix + ".weight"),
            [2, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        )
        return ops.add(result, reshape(weight(prefix + ".bias"), [1, -1, 1, 1, 1]))

    frames = 1 if startup else 4
    video = ops.parameter(
        [1, 3, frames, geometry["height"], geometry["width"]], storage, name="video"
    )
    if startup:
        first = ops.tile(video, _i64(ops, [1, 1, 3, 1, 1]))
        frames_value = ops.concat([first, video], 2)
    else:
        frames_value = video

    # This is the official PixelUnshuffle3D permutation:
    # [B,C,F,H/16,16,W/16,16] -> [B,C,16,16,F,H/16,W/16].
    x = ops.reshape(
        frames_value,
        _i64(ops, [1, 3, 4, geometry["token_h"], 16, geometry["token_w"], 16]),
        False,
    )
    x = ops.transpose(x, _i64(ops, [0, 1, 4, 6, 2, 3, 5]))
    x = ops.reshape(
        x,
        _i64(ops, [1, 768, 4, geometry["token_h"], geometry["token_w"]]),
        False,
    )
    next_cache1 = ops.strided_slice(
        x,
        _i64(ops, [0, 0, 2, 0, 0]),
        _i64(ops, [1, 768, 4, geometry["token_h"], geometry["token_w"]]),
        _i64(ops, [1, 1, 1, 1, 1]),
        [0] * 5,
        [0] * 5,
    )
    parameters = [video]
    if weights is None:
        parameters.extend(weight.get_node() for weight in weight_parameters.values())
    if not startup:
        cache1 = ops.parameter(
            [1, 768, 2, geometry["token_h"], geometry["token_w"]], storage, name="cache1"
        )
        cache2 = ops.parameter(
            [1, 2048, 2, geometry["token_h"], geometry["token_w"]], storage, name="cache2"
        )
        parameters.extend([cache1, cache2])
        x = ops.concat([cache1, x], 2)
    x = channel_norm(convolution(x, "conv1", 2 if startup else 0), "norm1.weight", 2048)
    x = ops.multiply(x, ops.sigmoid(x))
    next_cache2 = x

    if startup:
        outputs = [next_cache1, next_cache2]
        names = ("cache1", "cache2")
    else:
        x = ops.concat([cache2, x], 2)
        x = channel_norm(convolution(x, "conv2", 0), "norm2.weight", 3072)
        x = ops.multiply(x, ops.sigmoid(x))
        x = ops.transpose(x, _i64(ops, [0, 2, 3, 4, 1]))
        x = ops.reshape(x, _i64(ops, [1, geometry["token_count"], 3072]), False)
        x = ops.matmul(x, weight("linear_layers.0.weight"), False, True)
        x = ops.add(x, weight("linear_layers.0.bias"))
        outputs = [x, next_cache1, next_cache2]
        names = ("projected", "cache1", "cache2")

    for output, name in zip(outputs, names):
        output.output(0).get_tensor().set_names({name})
    model = ov.Model(
        outputs,
        parameters,
        f"plus_lq_projection_{precision}_{'startup' if startup else 'steady'}",
    )
    model.validate_nodes_and_infer_types()
    return model


def split_projection_graph(model):
    """Cut a steady BF16 graph at the current conv1/cache2 boundary.

    The two returned models retain the original graph nodes and constants.  The
    first owns the video/old-cache1 path and the second owns conv2 onward;
    replacing the conv2 concat inputs makes the boundary unambiguous to graph
    lifetime analysis.
    """

    import openvino as ov
    from openvino import opset13 as ops

    def parameter_name(parameter):
        names = sorted(str(name) for name in parameter.output(0).get_names())
        if len(names) != 1:
            raise ValueError("projection Parameters require one canonical tensor name")
        return names[0]

    def parameter_contract(candidate, data_names, learned_keys):
        expected_data = set(data_names)
        expected_learned = {WEIGHT_PORTS[key] for key in learned_keys}
        by_name = {}
        for parameter in candidate.parameters:
            name = parameter_name(parameter)
            if name in by_name:
                raise ValueError(f"duplicate projection Parameter name: {name}")
            by_name[name] = parameter
        expected = expected_data | expected_learned
        if set(by_name) != expected:
            raise ValueError("projection Parameter ABI changed")
        for name, parameter in by_name.items():
            if parameter.output(0).get_element_type() != ov.Type.bf16:
                raise ValueError(f"projection Parameter is not BF16: {name}")
        for key in learned_keys:
            parameter = by_name[WEIGHT_PORTS[key]]
            if list(parameter.output(0).get_partial_shape().to_shape()) != list(WEIGHT_SHAPES[key]):
                raise ValueError(f"projection learned shape changed: {key}")
        return by_name

    def reachable_parameters(outputs, candidates):
        candidate_names = {
            parameter_name(parameter): parameter for parameter in candidates
        }
        found = set()
        visited = set()

        def visit(value):
            node = value.get_node()
            # OpenVINO may materialize a fresh Python wrapper for the same C++
            # node on each get_node() call, so object identity is not stable.
            node_name = None
            if node.get_type_name() == "Parameter":
                names = sorted(str(name) for name in node.output(0).get_names())
                node_name = names[0] if len(names) == 1 else None
                if node_name in candidate_names:
                    found.add(node_name)
                return
            node_id = (node.get_type_name(), node.get_friendly_name())
            if node_id in visited:
                return
            visited.add(node_id)
            for input_value in node.input_values():
                visit(input_value)

        for output in outputs:
            visit(output)
        return [parameter for parameter in candidates
                if parameter_name(parameter) in found]

    if model.get_friendly_name() != "plus_lq_projection_bf16_steady":
        raise ValueError("expected the authenticated steady BF16 projection graph")
    parameter_names = {parameter_name(p): p for p in model.parameters}
    outputs = {p.get_any_name(): p for p in model.outputs}
    if (set(parameter_names) not in (
            {"video", "cache1", "cache2"},
            {"video", "cache1", "cache2"} | {WEIGHT_PORTS[key] for key in WEIGHT_KEYS},
        )
            or set(outputs) != {"projected", "cache1", "cache2"}
            or any(p.output(0).get_element_type() != ov.Type.bf16 for p in parameter_names.values())
            or any(p.get_element_type() != ov.Type.bf16 for p in outputs.values())):
        raise ValueError("steady projection ABI must be the BF16 Plus contract")
    external_weights = any(name.startswith("weight__") for name in parameter_names)
    learned_keys = STEADY_WEIGHT_KEYS if external_weights else ()
    parameter_contract(model, ("video", "cache1", "cache2"), learned_keys)
    video_shape = list(parameter_names["video"].output(0).get_shape())
    if len(video_shape) != 5 or video_shape[:3] != [1, 3, 4] \
            or video_shape[3] % 16 or video_shape[4] % 16:
        raise ValueError("steady projection video geometry changed")
    spatial = [video_shape[3] // 16, video_shape[4] // 16]
    cache2_shape = [1, 2048, 2, *spatial]
    expected = {
        "cache1": [1, 768, 2, *spatial],
        "cache2": cache2_shape,
    }
    if any(list(parameter_names[name].output(0).get_shape()) != shape
           for name, shape in expected.items()) \
            or list(outputs["cache2"].get_partial_shape().to_shape()) != cache2_shape \
            or list(outputs["projected"].get_partial_shape().to_shape()) \
            != [1, spatial[0] * spatial[1], 1536]:
        raise ValueError("steady projection cache/output ABI changed")

    def body(clone, name):
        value = next(p for p in clone.outputs if p.get_any_name() == name).get_node().input_value(0)
        value.get_tensor().set_names({name})
        return value

    prefix = model.clone()
    prefix_outputs = [body(prefix, "cache1"), body(prefix, "cache2")]
    prefix_parameters = reachable_parameters(prefix_outputs, list(prefix.parameters))
    stage1 = ov.Model(prefix_outputs, prefix_parameters,
                      "plus_lq_projection_bf16_steady_stage1")

    suffix = model.clone()
    current = body(suffix, "cache2")
    projected = body(suffix, "projected")
    old = next(p for p in suffix.parameters if parameter_name(p) == "cache2")
    consumers = [target.get_node() for target in current.get_target_inputs()
                 if target.get_node().get_type_name() == "Concat"]
    if len(consumers) != 1:
        raise ValueError("steady projection conv2 concat ABI changed")
    concat = consumers[0]
    if concat.get_input_size() != 2 or concat.get_axis() != 2:
        raise ValueError("steady projection conv2 concat ABI changed")
    if concat.input(0).get_source_output().get_node() is not old:
        raise ValueError("steady projection old cache2 moved")
    if concat.input(1).get_source_output().get_node() is not current.get_node():
        raise ValueError("steady projection current cache2 moved")
    old_cache2 = ops.parameter(old.output(0).get_partial_shape(), ov.Type.bf16, name="old_cache2")
    current_cache2 = ops.parameter(current.get_partial_shape(), ov.Type.bf16, name="current_cache2")
    concat.input(0).replace_source_output(old_cache2.output(0))
    concat.input(1).replace_source_output(current_cache2.output(0))
    suffix_parameters = reachable_parameters([projected], list(suffix.parameters))
    stage2 = ov.Model([projected], [old_cache2, current_cache2, *suffix_parameters],
                      "plus_lq_projection_bf16_steady_stage2")
    stage1.validate_nodes_and_infer_types()
    stage2.validate_nodes_and_infer_types()
    parameter_contract(stage1, ("video", "cache1"),
                       () if not external_weights else STARTUP_WEIGHT_KEYS)
    parameter_contract(stage2, ("old_cache2", "current_cache2"),
                       () if not external_weights else STAGE2_WEIGHT_KEYS)
    return stage1, stage2


def _metadata_equal_exact(actual, expected):
    """Compare JSON-like plan metadata without bool/int coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (tuple(actual) == tuple(expected)
                and all(_metadata_equal_exact(actual[key], expected[key])
                        for key in expected))
    if isinstance(expected, list):
        return (len(actual) == len(expected)
                and all(_metadata_equal_exact(left, right)
                        for left, right in zip(actual, expected)))
    return actual == expected


def _source_projection_plan(source_frames: int) -> dict:
    """Rebuild the authenticated temporal plan from the pinned stream source."""

    from plus_stream import build_stream_plan

    if type(source_frames) is not int or not 17 <= source_frames <= 49:
        raise ValueError("input_plan source_frames must be an integer in [17, 49]")
    stream = build_stream_plan(source_frames, fps=1.0)
    calls = []
    for call in stream.iter_calls():
        calls.append({
            "call": call.index,
            "latent_start": call.latent_start,
            "latent_stop": call.latent_stop,
            "frame_offset": call.frame_offset,
            "projection_start": call.projection_start,
            "projection_stop": call.projection_stop,
            "projection_source_indices": list(call.projection_source_indices),
            "decoder_start": call.decoder_start,
            "decoder_stop": call.decoder_stop,
            "decoder_source_indices": list(call.decoder_source_indices),
            "merged_output_start": call.merged_output_start,
            "merged_output_stop": call.merged_output_stop,
            "last": call.last,
        })
    return {
        "source_frames": stream.source_frames,
        "effective_frames": stream.effective_frames,
        "expanded_frames": stream.expanded_frames,
        "requested_latents": stream.requested_latents,
        "padded_latents": stream.padded_latents,
        "call_count": stream.call_count,
        "projection_frames": stream.projection_frames,
        "decoder_frames": stream.decoder_frames,
        "merged_decoder_frames": stream.merged_decoder_frames,
        "output_frames": stream.output_frames,
        "temporal_compression": stream.temporal_compression,
        "chunk_latents": 4,
        "calls": calls,
    }


def _validate_input_plan(value):
    """Validate a plan returned by ``prepare_plus_video_input.input_plan``.

    The caller authenticates the manifest before obtaining this dictionary;
    this second check prevents a stale or hand-edited schedule from reaching a
    device request.  Returning fresh source-derived metadata also prevents
    later code from depending on mutable caller-owned dictionaries.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("input_plan must be the authenticated plan dictionary")
    expected = _source_projection_plan(value.get("source_frames"))
    if not _metadata_equal_exact(value, expected):
        raise ValueError("input_plan temporal schedule changed")
    return expected


def _projection_group_starts(plan):
    """Return the four-frame causal projection groups for each stream call."""

    if plan is None:
        # Keep the original 17-frame arithmetic and ordering byte-for-byte.
        return ((1, 5, 9, 13), (17, 21, 25, 29))
    return tuple(
        tuple(range(
            1 if call["call"] == 0 else call["projection_start"],
            call["projection_stop"],
            4,
        ))
        for call in plan["calls"]
    )


def project_video_conditions(core, device, context, geometry, video, *, input_plan=None,
                             mode: str = "turbo"):
    """Run the causal LQ projection before denoiser graphs are allocated.

    The returned arrays are owning BF16 storage-bit buffers.  All projection
    requests, cache state, weights, and compiled graphs are retired before
    this function returns so the caller can verify a zero projection footprint
    before constructing the denoiser.
    """
    # Validate all caller-controlled chronology and host ABI before importing
    # or querying any device-backed graph.  ``input_plan`` is the authenticated
    # result of prepare_plus_video_input.input_plan(manifest).
    spec = resolve_plus_mode(mode)
    checkpoint = (CHECKPOINT if mode == "turbo" else
                  spec.helper_paths(CHECKPOINT.parent.parent)["lq"])
    plan = _validate_input_plan(input_plan)
    height, width = tuple(geometry.working_hw)
    value = np.asarray(video)
    expected = (
        1, 3, 33 if plan is None else plan["projection_frames"], height, width
    )
    if value.dtype != np.dtype("float32") or value.shape != expected:
        raise ValueError(f"projection video ABI changed: expected float32 {expected}")
    if not np.isfinite(value).all():
        raise ValueError("projection video contains non-finite values")
    from plus_resident import ResidentGraph

    token_count = (height // 16) * (width // 16)
    group_starts = _projection_group_starts(plan)
    memory = []
    receipt = {
        "status": "projection_conditions_complete",
        "geometry": {"height": height, "width": width, "token_count": token_count},
        "groups": sum(len(starts) for starts in group_starts),
        "calls": len(group_starts),
        "storage": "BF16 uint16 bits",
        "setup_seconds": 0.0,
        "gpu_memory": memory,
    }
    if plan is not None:
        receipt["input_plan"] = plan

    def sample(stage):
        memory.append({
            "stage": stage,
            "statistics": dict(core.get_property(device, "GPU_MEMORY_STATISTICS")),
        })

    weights = startup = projection_models = projection = projection_tail = None
    projection_state = projection_spare = None
    conditions = []
    setup_seconds = 0.0
    try:
        sample("before_startup")
        setup_started = time.perf_counter()
        weights = load_weights(checkpoint, mode=mode)
        receipt["weights"] = {"path": str(checkpoint), "sha256": spec.lq_sha256,
                              "mode": mode,
                              "source_commit": SOURCE_COMMIT, "key_count": len(WEIGHT_KEYS)}
        startup_keys = ("conv1.bias", "conv1.weight", "norm1.weight")
        startup = ResidentGraph(
            core,
            build_projection_graph(
                {key: weights[key] for key in startup_keys},
                True,
                height=height,
                width=width,
            ),
            device,
            context=context,
        )
        setup_seconds += time.perf_counter() - setup_started
        receipt["setup_seconds"] = setup_seconds
        startup.infer({"video": storage_tensor(value[:, :, :1], "bf16")}, read=())
        projection_state = dict(startup.buffers)
        startup = None
        gc.collect()
        sample("after_startup")

        setup_started = time.perf_counter()
        projection_models = split_projection_graph(
            build_projection_graph(weights, False, height=height, width=width)
        )
        projection = ResidentGraph(core, projection_models[0], device, context=context)
        projection_tail = ResidentGraph(core, projection_models[1], device, context=context)
        projection_spare = {
            name: projection.buffers[name] for name in ("cache1", "cache2")
        }
        projection_models = None
        gc.collect()
        setup_seconds += time.perf_counter() - setup_started
        receipt["setup_seconds"] = setup_seconds
        sample("before_steady")

        for call, starts in enumerate(group_starts):
            condition = np.empty(
                (1, len(starts) * token_count, 1536), dtype=np.uint16
            )
            for inner, start in enumerate(starts):
                for name, target in projection_spare.items():
                    projection.bind_output(name, target)
                projection.infer(
                    {
                        "video": storage_tensor(value[:, :, start:start + 4], "bf16"),
                        "cache1": projection_state["cache1"],
                    },
                    read=(),
                )
                projected = projection_tail.infer(
                    {
                        "old_cache2": projection_state["cache2"],
                        "current_cache2": projection_spare["cache2"],
                    },
                    read=("projected",),
                )["projected"]
                projected = np.asarray(projected)
                expected_projected = (1, token_count, 1536)
                if projected.dtype != np.dtype("uint16") or projected.shape != expected_projected:
                    raise ValueError(
                        f"projection output ABI changed: expected uint16 {expected_projected}"
                    )
                if np.any((projected & np.uint16(0x7f80)) == np.uint16(0x7f80)):
                    raise ValueError("projection output contains non-finite BF16 storage")
                condition[:, inner * token_count:(inner + 1) * token_count, :] = projected
                del projected
                projection_state, projection_spare = projection_spare, projection_state
            conditions.append(condition)
        sample("after_steady")
        receipt["condition_shapes"] = [list(item.shape) for item in conditions]
        return conditions, receipt
    finally:
        weights = startup = projection_models = projection = projection_tail = None
        projection_state = projection_spare = None
        target = None  # The last ping-pong destination also owns GPU storage.
        gc.collect()
        sample("after_retire")


__all__ = [
    "BASE",
    "CHECKPOINT",
    "CHECKPOINT_SHA256",
    "PLUS_SOURCE",
    "SOURCE_COMMIT",
    "WEIGHT_KEYS",
    "WEIGHT_PORTS",
    "WEIGHT_SHAPES",
    "authenticate_checkpoint",
    "build_projection_graph",
    "split_projection_graph",
    "load_weights",
    "project_video_conditions",
    "projection_geometry",
    "sha256_path",
    "storage_tensor",
    "tensor_array",
]
