#!/usr/bin/env python3
"""Static contract for the native-F16 full Wan component converter.

This module contains no OpenVINO graph construction and does not load tensor
values.  It makes the source chronology, component layer ABI, and bounded
checkpoint plan explicit for the 30-layer export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Tuple


BASE = Path(__file__).resolve().parents[2]
FULL_MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE / "models/diffusion_pytorch_model_streaming_dmd.safetensors"
WAN_SOURCE = BASE / "source/FlashVSR/diffsynth/models/wan_video_dit.py"
PIPELINE_SOURCE = BASE / "source/FlashVSR/diffsynth/pipelines/flashvsr_tiny_long.py"
TINY_LONG_INFER_SOURCE = BASE / "source/FlashVSR/examples/WanVSR/infer_flashvsr_v1.1_tiny_long_video.py"
LQ_SOURCE = BASE / "source/FlashVSR/examples/WanVSR/utils/utils.py"
LQ_CHECKPOINT = BASE / "models/LQ_proj_in.ckpt"
PROMPT = BASE / "source/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth"
BLOCK_GRAPH = BASE / "openvino/block/graph.py"
BLOCK_CONTRACT = BASE / "openvino/block/contract.py"
BLOCK_WEIGHTS = BASE / "openvino/block/weights.py"
MODEL_EDGES_CONVERT = BASE / "openvino/model_edges/convert.py"
POST_BUILDER = BASE / "openvino/post_diagnostic/first_boundary_variants.py"

CHECKPOINT_SHA256 = "bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be"
WAN_SOURCE_SHA256 = "ce3827b879f1f8406f1f643630e695b13ec3573ab771dd7a89075975bd5406f5"
PIPELINE_SOURCE_SHA256 = "64f0378eb3adf0ea02f190e5793c0ed66ca65cf7b52b9786679e26e7ecf7f113"
TINY_LONG_INFER_SOURCE_SHA256 = "3e5a32c2b6832888290efd180a0d127e232366670664239cfec9d1b92d17504f"
LQ_SOURCE_SHA256 = "dfe156fd24b35a31810cb82ee7c635952c30818bf8169871e0dca6b4df4b617d"
LQ_CHECKPOINT_SHA256 = "d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2"
PROMPT_SHA256 = "4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef"
BLOCK_GRAPH_SHA256 = "7b3b7fc479d73ba8f171bf1721cb45a1b9c06d1b0a8779bf49fcb91dceb40ad0"
BLOCK_CONTRACT_SHA256 = "df3c111cf3cf7708f5c76d2386c10793a51d4e300d26a9568ca5a225e894535c"
BLOCK_WEIGHTS_SHA256 = "37bafab2709137954ecb6319548451186838db4327e5d308c704817ac95acd99"
MODEL_EDGES_CONVERT_SHA256 = "d0b47cffe15bd1b1fc06a2ed49236c38f939263807a3a19b4ea5bcd58535d359"
POST_BUILDER_SHA256 = "c38aa9e0aef90360af5b445d2b51ea34ba9cad33e5246bbc792308aac0b80bbe"
SOURCE_COMMIT = "cf910c61a60733e610e9c6e8b607f80c3a6c202b"
STREAM_MODE = "T1/temporalFlash"

NUM_LAYERS = 30
WINDOW = (2, 8, 8)
WINDOW_TOKENS = 128
KV_TIME_GROUPS = 3
TOPK_RATIO = 2.0
LOCAL_RANGE = 9
FIXED_TIMESTEP = 1000
PATCH = (1, 2, 2)
DIM = 1536
HEADS = 12
HEAD_DIM = 128
FFN_DIM = 8960
TEXT_DIM = 4096
CONTEXT_TOKENS = 512
EPS = 1e-6
PROMPT_SHAPE = (1, CONTEXT_TOKENS, TEXT_DIM)
LQ_LAYER_COUNT = 1
LQ_KEYS = (
    "conv1.weight", "conv1.bias", "norm1.gamma",
    "conv2.weight", "conv2.bias", "norm2.gamma",
    "linear_layers.0.weight", "linear_layers.0.bias",
)
LQ_SHAPES = {
    "conv1.weight": [2048, 768, 4, 3, 3],
    "conv1.bias": [2048],
    "norm1.gamma": [2048, 1, 1, 1],
    "conv2.weight": [3072, 2048, 4, 3, 3],
    "conv2.bias": [3072],
    "norm2.gamma": [3072, 1, 1, 1],
    "linear_layers.0.weight": [1536, 3072],
    "linear_layers.0.bias": [1536],
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reviewed module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import reviewed contracts by path rather than relying on a caller's
# PYTHONPATH.  In particular, this avoids confusing full_model.contract with
# the block graph's historical direct-import ``contract`` module.
_BLOCK = _load_module("full_model_reviewed_block_contract", BASE / "openvino/block/contract.py")
_EDGES = _load_module("full_model_reviewed_model_edges_convert", BASE / "openvino/model_edges/convert.py")

BLOCK_KEYS = tuple(_BLOCK.BLOCK_KEYS)
TEXT_KEYS = tuple(_BLOCK.TEXT_EMBED_KEYS)
GLOBAL_KEYS = tuple(_EDGES.SELECTED_KEYS)
BLOCK_SHAPES = dict(_BLOCK.EXPECTED_SHAPES)
GLOBAL_SHAPES = dict(_EDGES.EXPECTED_SHAPES)


def _layer_key(key: str, layer_id: int) -> str:
    if not 0 <= layer_id < NUM_LAYERS:
        raise ValueError(f"layer_id must be in [0,{NUM_LAYERS}), got {layer_id}")
    if key.startswith("blocks.0."):
        return key.replace("blocks.0.", f"blocks.{layer_id}.", 1)
    return key


def layer_keys(layer_id: int) -> Tuple[str, ...]:
    """Return the exact checkpoint keys needed for one DiT block."""

    _layer_key("blocks.0.modulation", layer_id)  # validate layer_id
    return tuple(_layer_key(key, layer_id) for key in BLOCK_KEYS)


def layer_shapes(layer_id: int) -> Dict[str, list[int]]:
    """Return metadata expectations with ``blocks.0`` expanded to a layer."""

    _layer_key("blocks.0.modulation", layer_id)  # validate layer_id
    return {_layer_key(key, layer_id): list(shape) for key, shape in BLOCK_SHAPES.items()
            if key.startswith("blocks.0.")}


def normalize_layer_key(key: str, layer_id: int) -> str:
    """Map a checkpoint layer name to the reviewed layer-0 graph ABI."""

    prefix = f"blocks.{layer_id}."
    if not key.startswith(prefix):
        raise KeyError(f"{key!r} is not a layer-{layer_id} checkpoint key")
    return "blocks.0." + key[len(prefix):]


def normalize_layer_mapping(values: Mapping[str, Any], layer_id: int) -> Dict[str, Any]:
    """Adapt one layer's values without renaming unrelated edge/cache keys."""

    output: Dict[str, Any] = {}
    for key, value in values.items():
        output[normalize_layer_key(key, layer_id) if key.startswith(f"blocks.{layer_id}.") else key] = value
    return output


@dataclass(frozen=True)
class GeometrySpec:
    """Token and cache geometry for one source ``model_fn_wan_video`` call."""

    name: str
    f: int
    h: int
    w: int
    streaming: bool = True

    @property
    def tokens(self) -> int:
        return self.f * self.h * self.w

    @property
    def spatial_windows(self) -> int:
        return (self.h // WINDOW[1]) * (self.w // WINDOW[2])

    @property
    def time_groups(self) -> int:
        return self.f // WINDOW[0]

    @property
    def query_windows(self) -> int:
        return self.time_groups * self.spatial_windows

    @property
    def old_cache_windows(self) -> int:
        return 0 if self.f == 6 else KV_TIME_GROUPS * self.spatial_windows

    @property
    def attended_windows(self) -> int:
        return self.old_cache_windows + self.query_windows

    @property
    def retained_cache_windows(self) -> int:
        return KV_TIME_GROUPS * self.spatial_windows

    @property
    def evict_windows(self) -> int:
        return max(0, self.attended_windows - self.retained_cache_windows)

    @property
    def latent_shape(self) -> Tuple[int, int, int, int, int]:
        return (1, 16, self.f, self.h * PATCH[1], self.w * PATCH[2])

    @property
    def token_shape(self) -> Tuple[int, int, int]:
        return (1, self.tokens, DIM)

    @property
    def window_shape(self) -> Tuple[int, int, int]:
        return (self.query_windows, WINDOW_TOKENS, DIM)

    @property
    def cache_shape(self) -> Tuple[int, int, int]:
        return (self.retained_cache_windows, WINDOW_TOKENS, DIM)

    def validate(self) -> None:
        for value, divisor, label in (
            (self.f, WINDOW[0], "f"), (self.h, WINDOW[1], "h"), (self.w, WINDOW[2], "w")
        ):
            if value <= 0 or value % divisor:
                raise ValueError(f"{label}={value} must be positive and divisible by {divisor}")
        if self.streaming and self.f not in (2, 6):
            raise ValueError("the pinned streaming source permits f=6 startup or f=2 steady")


STARTUP = GeometrySpec("startup_f6_128x128", 6, 8, 8)
STEADY = GeometrySpec("steady_f2_128x128", 2, 8, 8)


@dataclass(frozen=True)
class StreamCall:
    """Source call arguments that affect shape, RoPE, and cache chronology."""

    index: int
    geometry: GeometrySpec

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("stream index must be non-negative")
        self.geometry.validate()
        if (self.index == 0) != (self.geometry.f == 6):
            raise ValueError("index 0 is startup f=6; later calls are steady f=2")

    @property
    def startup(self) -> bool:
        return self.index == 0

    @property
    def cur_process_idx(self) -> int:
        return self.index

    @property
    def rope_time_offset(self) -> int:
        return 0 if self.startup else 4 + self.index * 2

    @property
    def local_num(self) -> int:
        return self.geometry.time_groups

    @property
    def topk(self) -> int:
        return int(self.geometry.spatial_windows ** 2 * TOPK_RATIO) - 1

    def as_dict(self) -> Dict[str, object]:
        g = self.geometry
        return {
            "index": self.index,
            "cur_process_idx": self.cur_process_idx,
            "name": g.name,
            "f": g.f,
            "h": g.h,
            "w": g.w,
            "tokens": g.tokens,
            "spatial_windows": g.spatial_windows,
            "local_num": self.local_num,
            "topk_ratio": TOPK_RATIO,
            "topk": self.topk,
            "kv_len": KV_TIME_GROUPS,
            "local_range": LOCAL_RANGE,
            "is_stream": True,
            "rope_time_offset": self.rope_time_offset,
            "old_cache_windows": g.old_cache_windows,
            "new_cache_windows": g.query_windows,
            "attended_windows": g.attended_windows,
            "evict_windows": g.evict_windows,
            "retained_cache_windows": g.retained_cache_windows,
        }


STARTUP_CALL = StreamCall(0, STARTUP)
STEADY_CALL = StreamCall(1, STEADY)


def stream_calls(h: int = 8, w: int = 8) -> Tuple[StreamCall, StreamCall]:
    """Return startup and first-steady contracts for any supported spatial grid."""

    return StreamCall(0, GeometrySpec(f"startup_f6_{h * 16}x{w * 16}", 6, h, w)), StreamCall(
        1, GeometrySpec(f"steady_f2_{h * 16}x{w * 16}", 2, h, w)
    )


@dataclass(frozen=True)
class WeightBatchPlan:
    """Bounded loading order; values never need to coexist for all 30 layers."""

    global_keys: Tuple[str, ...] = GLOBAL_KEYS
    text_keys: Tuple[str, ...] = TEXT_KEYS
    layer_count: int = NUM_LAYERS
    storage_dtype: str = "F16"

    @property
    def layer_key_count(self) -> int:
        return len(BLOCK_KEYS)

    @property
    def total_expected_keys(self) -> int:
        return len(self.global_keys) + len(self.text_keys) + self.layer_count * self.layer_key_count

    def load_order(self) -> Tuple[str, ...]:
        """Describe one safe-open pass; no whole-model ``load_file`` operation."""

        return ("checkpoint_sha256_once", "headers_all_keys", "global_edge_values_once",
                "text_embedding_values_once", "layer_values_one_at_a_time",
                "cross_kv_cache_one_layer_at_a_time", "release_layer_values")

    def bytes_for(self, shape: Tuple[int, ...], itemsize: int = 2) -> int:
        size = 1
        for dim in shape:
            size *= int(dim)
        return size * itemsize

    def layer_storage_bytes(self) -> int:
        return sum(self.bytes_for(tuple(BLOCK_SHAPES[key])) for key in BLOCK_KEYS)

    def layer_source_bytes(self) -> int:
        return sum(self.bytes_for(tuple(BLOCK_SHAPES[key]), 4) for key in BLOCK_KEYS)

    def manifest(self) -> Dict[str, object]:
        return {
            "global_key_count": len(self.global_keys),
            "text_key_count": len(self.text_keys),
            "layer_key_count": self.layer_key_count,
            "layer_count": self.layer_count,
            "total_expected_keys": self.total_expected_keys,
            "storage_dtype": self.storage_dtype,
            "layer_source_bytes": self.layer_source_bytes(),
            "layer_storage_bytes": self.layer_storage_bytes(),
            "load_order": list(self.load_order()),
            "bounded": True,
            "whole_model_state_dict": False,
            "cross_cache": {
                "context_embedding_once": True,
                "per_layer_kv": True,
                "storage_shape": [1, CONTEXT_TOKENS, DIM],
                "prompt_shape": list(PROMPT_SHAPE),
                "gelu": "approximate=tanh, F32 arithmetic",
            },
            "lq_projection": {
                "class": "Causal_LQ4x_Proj",
                "layer_count": LQ_LAYER_COUNT,
                "key_count": len(LQ_KEYS),
                "keys": list(LQ_KEYS),
                "storage_dtype": "F16",
                "source_dtype": "BF16",
                "injection": "only block_id=0, immediately before block 0",
            },
        }


@dataclass(frozen=True)
class FullModelPlan:
    """Review manifest for the prefix → LQ(block 0) → 30 blocks → head path."""

    layer_count: int = NUM_LAYERS
    post_policy: str = "selected_conv_linears"
    no_seed: bool = True
    no_gpu: bool = True

    def manifest(self) -> Dict[str, object]:
        if self.layer_count != NUM_LAYERS:
            raise ValueError(f"full model requires exactly {NUM_LAYERS} layers")
        if self.post_policy not in ("selected_conv_linears", "unresolved", "reviewed_block_graph", "external_selected"):
            raise ValueError(f"unsupported post policy: {self.post_policy!r}")
        return {
            "schema": "flashvsr.full_model.plan.v2",
            "source_commit": SOURCE_COMMIT,
            "source_hashes": {
                "wan_video_dit.py": WAN_SOURCE_SHA256,
                "flashvsr_tiny_long.py": PIPELINE_SOURCE_SHA256,
                "infer_flashvsr_v1.1_tiny_long_video.py": TINY_LONG_INFER_SOURCE_SHA256,
                "WanVSR/utils/utils.py": LQ_SOURCE_SHA256,
                "LQ_proj_in.ckpt": LQ_CHECKPOINT_SHA256,
                "posi_prompt.pth": PROMPT_SHA256,
                "openvino/block/graph.py": BLOCK_GRAPH_SHA256,
                "openvino/block/contract.py": BLOCK_CONTRACT_SHA256,
                "openvino/block/weights.py": BLOCK_WEIGHTS_SHA256,
                "openvino/model_edges/convert.py": MODEL_EDGES_CONVERT_SHA256,
                "openvino/post_diagnostic/first_boundary_variants.py": POST_BUILDER_SHA256,
                "diffusion_pytorch_model_streaming_dmd.safetensors": CHECKPOINT_SHA256,
            },
            "config": {
                "mode": STREAM_MODE,
                "layers": self.layer_count,
                "dim": DIM,
                "ffn_dim": FFN_DIM,
                "heads": HEADS,
                "head_dim": HEAD_DIM,
                "text_dim": TEXT_DIM,
                "context_tokens": CONTEXT_TOKENS,
                "patch": list(PATCH),
                "window": list(WINDOW),
                "kv_time_groups": KV_TIME_GROUPS,
                "timestep": FIXED_TIMESTEP,
                "eps": EPS,
            },
            "calls": [STARTUP_CALL.as_dict(), STEADY_CALL.as_dict()],
            "lq": {
                "model_layer_count": self.layer_count,
                "projection_class": "Causal_LQ4x_Proj",
                "projection_layer_count": LQ_LAYER_COUNT,
                "projection_checkpoint_keys": list(LQ_KEYS),
                "injection": "x = x + LQ_latents[0] immediately before block 0; no LQ add for blocks 1..29",
                "stream_reset": "clear_cache once at sequence start",
                "startup_projection_chunks": 7,
                "steady_projection_chunks": 2,
                "output_list_is_layer_indexed": True,
                "output_count": LQ_LAYER_COUNT,
                "output_block_ids": [0],
                "projection_checkpoint": str(LQ_CHECKPOINT),
            },
            "edges": {
                "prefix": "model_edges prefix: latent F16 -> tokens,t,t_mod",
                "block": "30 repeated layer ABI instances; pre/attention/post split",
                "cross_kv": "text embedding once, per-layer K/V constants or inputs",
                "suffix": "model_edges suffix: tokens,t -> source-layout latent F16",
            },
            "post_policy": self.post_policy,
            "no_seed": self.no_seed,
            "no_gpu": self.no_gpu,
            "ir_export": False,
            "all30_export": False,
        }


def source_summary() -> Dict[str, object]:
    """Stable source pins used by the metadata audit and review manifest."""

    return {
        "source_commit": SOURCE_COMMIT,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "wan_source": str(WAN_SOURCE),
        "wan_source_sha256": WAN_SOURCE_SHA256,
        "pipeline_source": str(PIPELINE_SOURCE),
        "pipeline_source_sha256": PIPELINE_SOURCE_SHA256,
        "tiny_long_infer_source": str(TINY_LONG_INFER_SOURCE),
        "tiny_long_infer_source_sha256": TINY_LONG_INFER_SOURCE_SHA256,
        "lq_source": str(LQ_SOURCE),
        "lq_source_sha256": LQ_SOURCE_SHA256,
        "lq_checkpoint": str(LQ_CHECKPOINT),
        "lq_checkpoint_sha256": LQ_CHECKPOINT_SHA256,
        "lq_projection": {
            "class": "Causal_LQ4x_Proj",
            "layer_count": LQ_LAYER_COUNT,
            "keys": list(LQ_KEYS),
            "shapes": {key: list(shape) for key, shape in LQ_SHAPES.items()},
            "source_dtype": "BF16",
            "storage_dtype": "F16",
            "injection": "LQ_latents[0] before block 0 only",
        },
        "prompt": str(PROMPT),
        "prompt_sha256": PROMPT_SHA256,
        "reviewed_helpers": {
            "block_graph": {"path": str(BLOCK_GRAPH), "sha256": BLOCK_GRAPH_SHA256},
            "block_contract": {"path": str(BLOCK_CONTRACT), "sha256": BLOCK_CONTRACT_SHA256},
            "block_weights": {"path": str(BLOCK_WEIGHTS), "sha256": BLOCK_WEIGHTS_SHA256},
            "model_edges_convert": {"path": str(MODEL_EDGES_CONVERT), "sha256": MODEL_EDGES_CONVERT_SHA256},
            "post_builder": {"path": str(POST_BUILDER), "sha256": POST_BUILDER_SHA256,
                             "variant": "conv_linears"},
        },
        "global_keys": list(GLOBAL_KEYS),
        "text_keys": list(TEXT_KEYS),
        "block_key_count": len(BLOCK_KEYS),
        "all_checkpoint_key_count": len(GLOBAL_KEYS) + len(TEXT_KEYS) + NUM_LAYERS * len(BLOCK_KEYS),
    }


def jsonable(value: object) -> object:
    """Convert the small dataclass contract to ordinary JSON-safe values."""

    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)  # type: ignore[arg-type]
    return value

