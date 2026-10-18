#!/usr/bin/env python3
"""Native graph adapters for the full-model component converter.

The reviewed block graphs remain the implementation of pre-QKV and selected
attention.  The separately reviewed ``conv_linears`` post builder supplies
the first self.o fusion boundary.  This module adapts a full-model layer name
to their layer-0 ABI; the converter owns the layer loop and IR output.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Callable, Mapping, Optional

import numpy as np

try:
    from .contract import (
        BLOCK_KEYS,
        DIM,
        GeometrySpec,
        HEAD_DIM,
        normalize_layer_mapping,
    )
except ImportError:  # direct execution from this directory
    # A caller may already have ``openvino/block`` on PYTHONPATH, where the
    # historical module is also named ``contract``.  Load our sibling by path
    # instead of accepting that accidental ABI collision.
    _contract_spec = importlib.util.spec_from_file_location(
        "full_model_contract_for_builders", Path(__file__).with_name("contract.py")
    )
    if _contract_spec is None or _contract_spec.loader is None:
        raise ImportError("cannot load full_model/contract.py")
    _contract = importlib.util.module_from_spec(_contract_spec)
    sys.modules[_contract_spec.name] = _contract
    _contract_spec.loader.exec_module(_contract)
    BLOCK_KEYS = _contract.BLOCK_KEYS
    DIM = _contract.DIM
    GeometrySpec = _contract.GeometrySpec
    HEAD_DIM = _contract.HEAD_DIM
    normalize_layer_mapping = _contract.normalize_layer_mapping


BASE = Path(__file__).resolve().parents[2]
BLOCK_DIR = BASE / "openvino/block"
POST_POLICIES = ("selected_conv_linears", "reviewed_block_graph", "external_selected", "unresolved")
SELECTED_POST_VARIANT = "conv_linears"
POST_BUILDER = BASE / "openvino/post_diagnostic/first_boundary_variants.py"
MODEL_EDGES_CONVERT = BASE / "openvino/model_edges/convert.py"


def _load_reviewed_graph():
    """Load block/graph.py with its historical direct ``contract`` import."""

    module_name = "full_model_reviewed_block_graph"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    contract_name = "contract"
    prior_contract = sys.modules.get(contract_name)
    contract_path = BLOCK_DIR / "contract.py"
    contract_spec = importlib.util.spec_from_file_location("full_model_block_contract_for_graph", contract_path)
    if contract_spec is None or contract_spec.loader is None:
        raise ImportError(f"cannot load reviewed block contract: {contract_path}")
    block_contract = importlib.util.module_from_spec(contract_spec)
    sys.modules[contract_spec.name] = block_contract
    sys.modules[contract_name] = block_contract
    contract_spec.loader.exec_module(block_contract)
    graph_path = BLOCK_DIR / "graph.py"
    graph_spec = importlib.util.spec_from_file_location(module_name, graph_path)
    if graph_spec is None or graph_spec.loader is None:
        raise ImportError(f"cannot load reviewed block graph: {graph_path}")
    graph = importlib.util.module_from_spec(graph_spec)
    sys.modules[module_name] = graph
    try:
        graph_spec.loader.exec_module(graph)
    finally:
        if prior_contract is None:
            sys.modules.pop(contract_name, None)
        else:
            sys.modules[contract_name] = prior_contract
    return graph


def _load_selected_post_builder():
    """Load the separately reviewed first-boundary builder without global imports."""

    module_name = "full_model_selected_conv_linears_builder"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    graph = _load_reviewed_graph()
    prior_graph = sys.modules.get("graph")
    sys.modules["graph"] = graph
    spec = importlib.util.spec_from_file_location(module_name, POST_BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load selected post builder: {POST_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # ``build_post_variant`` performs its historical ``import graph`` at call
    # time, so keep the reviewed alias alive for the converter process.  It is
    # the same object used to load the graph above, never an ambient module.
    module.graph = graph
    if prior_graph is not None and prior_graph is not graph:
        raise ImportError("a conflicting graph module is already loaded")
    return module


def _load_model_edges():
    """Load the pinned prefix/suffix math for generalized static geometry."""

    module_name = "full_model_reviewed_model_edges_convert"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    return _load_module(module_name, MODEL_EDGES_CONVERT)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reviewed module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_prefix(weights: Mapping[str, np.ndarray], geometry: GeometrySpec, embedding_f16=None):
    """Build the pinned prefix for any valid static token grid."""

    geometry.validate()
    edge = _load_model_edges()
    import openvino as ov
    from openvino import opset13 as ops

    f16 = ov.Type.f16
    x = ops.parameter(list(geometry.latent_shape), f16, name=f"{geometry.name}_x")
    embedding_f16 = edge.fixed_embedding_f16() if embedding_f16 is None else np.asarray(embedding_f16, dtype=np.float16)
    embedding = edge._const(ops, embedding_f16)
    patch = ops.convolution(
        x, edge._const(ops, weights["patch_embedding.weight"]), [1, 2, 2],
        [0, 0, 0], [0, 0, 0], [1, 1, 1],
    )
    patch = ops.add(
        patch, edge._reshape(ops, edge._const(ops, weights["patch_embedding.bias"]), [1, -1, 1, 1, 1])
    )
    tokens = edge._reshape(
        ops, ops.transpose(patch, edge._const(ops, [0, 2, 3, 4, 1], np.int64)),
        [1, geometry.tokens, DIM],
    )
    t = edge._linear(
        ops,
        edge._silu(ops, edge._linear(ops, embedding, weights,
                                     "time_embedding.0.weight", "time_embedding.0.bias", [1, -1])),
        weights, "time_embedding.2.weight", "time_embedding.2.bias", [1, -1],
    )
    t = edge._reshape(ops, t, [1, 1, DIM])
    t_mod = edge._reshape(
        ops,
        edge._linear(ops, edge._silu(ops, edge._reshape(ops, t, [1, DIM])), weights,
                     "time_projection.1.weight", "time_projection.1.bias", [1, -1]),
        [1, 6, DIM],
    )
    for value, output_name in (
        (tokens, f"{geometry.name}_tokens_out"), (t, f"{geometry.name}_t_out"),
        (t_mod, f"{geometry.name}_t_mod_out"), (embedding, f"{geometry.name}_embedding_out"),
    ):
        value.output(0).get_tensor().set_names({output_name})
    model = ov.Model(
        [tokens, t, t_mod, embedding], [x], f"wan_{geometry.name}_prefix_f16"
    )
    model.validate_nodes_and_infer_types()
    return model


def build_suffix(weights: Mapping[str, np.ndarray], geometry: GeometrySpec):
    """Build the pinned suffix for any valid static token grid."""

    geometry.validate()
    edge = _load_model_edges()
    import openvino as ov
    from openvino import opset13 as ops

    f16 = ov.Type.f16
    tokens = ops.parameter(list(geometry.token_shape), f16, name=f"{geometry.name}_tokens_in")
    t = ops.parameter([1, 1, DIM], f16, name=f"{geometry.name}_t_in")
    modulation = ops.add(edge._const(ops, weights["head.modulation"]), t)
    shift = edge._slice(ops, modulation, [0, 0, 0], [1, 1, DIM])
    scale = edge._slice(ops, modulation, [0, 1, 0], [1, 2, DIM])
    normalized = edge._layer_norm_f16(ov, ops, tokens)
    modulated = ops.add(ops.multiply(normalized, ops.add(edge._const(ops, np.float16(1.0)), scale)), shift)
    projected = edge._linear(ops, modulated, weights, "head.head.weight", "head.head.bias", [1, 1, -1])
    grid = edge._reshape(ops, projected, [1, geometry.f, geometry.h, geometry.w, 1, 2, 2, edge.OUT_DIM])
    output = edge._reshape(
        ops, ops.transpose(grid, edge._const(ops, [0, 7, 1, 4, 2, 5, 3, 6], np.int64)),
        [1, edge.OUT_DIM, geometry.f, geometry.h * 2, geometry.w * 2],
    )
    output.output(0).get_tensor().set_names({f"{geometry.name}_out"})
    model = ov.Model(
        [output], [tokens, t], f"wan_{geometry.name}_suffix_f16"
    )
    model.validate_nodes_and_infer_types()
    return model


def _reviewed_geometry(geometry: GeometrySpec):
    """Create the reviewed graph's geometry with generalized spatial cache counts."""

    geometry.validate()
    # graph.Geometry comes from block/contract.py.  Its inherited token/window
    # properties are correct; this subclass only corrects old+new cache window
    # capacity for S>1 spatial windows.
    base_geometry = _load_reviewed_graph().Geometry

    class GeneralizedGeometry(base_geometry):
        @property
        def spatial_windows(self):
            return (self.h // 8) * (self.w // 8)

        @property
        def attention_key_windows(self):
            old = 0 if self.f == 6 else 3 * self.spatial_windows
            return old + self.query_windows

    return GeneralizedGeometry(geometry.name, geometry.f, geometry.h, geometry.w, geometry.streaming)


def normalize_reviewed_weights(values: Mapping[str, np.ndarray], layer_id: int) -> dict[str, np.ndarray]:
    """Map one full-model layer's values to reviewed ``blocks.0.*`` names."""

    if not 0 <= layer_id < 30:
        raise ValueError(f"layer_id must be in [0,30), got {layer_id}")
    normalized = normalize_layer_mapping(values, layer_id)
    missing = sorted(set(BLOCK_KEYS) - set(normalized))
    if missing:
        raise KeyError(f"layer {layer_id} missing reviewed block keys: {missing}")
    return {key: normalized[key] for key in BLOCK_KEYS}


def build_layer_pre(values: Mapping[str, np.ndarray], geometry: GeometrySpec, *, layer_id: int = 0):
    """Build reviewed pre-QKV for one layer after the layer-name adapter."""

    graph = _load_reviewed_graph()
    return graph.build_pre_graph(normalize_reviewed_weights(values, layer_id), _reviewed_geometry(geometry))


def build_layer0_pre(values: Mapping[str, np.ndarray], geometry: GeometrySpec):
    """Build the reviewed pre-QKV graph for the layer-0 smoke gate."""

    return build_layer_pre(values, geometry, layer_id=0)


def build_layer0_attention(geometry: GeometrySpec):
    """Build the reviewed selected-key attention graph for one geometry."""

    reviewed_geometry = _reviewed_geometry(geometry)
    graph = _load_reviewed_graph()
    return graph.build_attention_graph(reviewed_geometry, reviewed_geometry.attention_key_windows)


def build_layer_post(
    values: Mapping[str, np.ndarray],
    geometry: GeometrySpec,
    *,
    layer_id: int = 0,
    post_policy: str,
    external_builder: Optional[Callable[[Mapping[str, np.ndarray], object], object]] = None,
):
    """Build a post graph only under an explicit, review-visible policy.

    ``reviewed_block_graph`` is the existing CPU-qualified arithmetic path.  A
    future self-output barrier or confirmed repair must be supplied through
    ``external_selected``; this adapter never silently selects the rejected
    rank-2 candidate or any GPU-only variant.  ``unresolved`` intentionally
    raises so an exporter cannot claim a resolved post path by omission.
    """

    if post_policy not in POST_POLICIES:
        raise ValueError(f"post_policy must be one of {POST_POLICIES}, got {post_policy!r}")
    if post_policy == "unresolved":
        raise RuntimeError("post graph policy is unresolved; choose an explicitly reviewed variant")
    reviewed_geometry = _reviewed_geometry(geometry)
    if post_policy == "selected_conv_linears":
        selected = _load_selected_post_builder()
        return selected.build_post_variant(
            normalize_reviewed_weights(values, layer_id), reviewed_geometry, SELECTED_POST_VARIANT
        )
    if post_policy == "external_selected":
        if external_builder is None:
            raise ValueError("external_selected requires external_builder")
        return external_builder(normalize_reviewed_weights(values, layer_id), reviewed_geometry)
    graph = _load_reviewed_graph()
    return graph.build_post_graph(normalize_reviewed_weights(values, layer_id), reviewed_geometry)


def build_layer0_post(
    values: Mapping[str, np.ndarray],
    geometry: GeometrySpec,
    *,
    post_policy: str,
    external_builder: Optional[Callable[[Mapping[str, np.ndarray], object], object]] = None,
):
    """Build the explicitly selected layer-0 post graph for the smoke gate."""

    return build_layer_post(
        values, geometry, layer_id=0, post_policy=post_policy, external_builder=external_builder
    )


def build_selected_post(
    values: Mapping[str, np.ndarray], geometry: GeometrySpec, *, layer_id: int = 0
):
    """Build the CPU-qualified ``conv_linears`` post graph for one layer."""

    model = build_layer_post(values, geometry, layer_id=layer_id, post_policy="selected_conv_linears")
    model.validate_nodes_and_infer_types()
    return model


def build_lq_injection_edge(geometry: GeometrySpec):
    """Build the exact source ``x + LQ_latents[0]`` boundary for block 0.

    The pinned TinyLong inference script constructs ``Causal_LQ4x_Proj`` with
    ``layer_num=1``.  Keeping this Add separate makes the single projected
    tensor explicit; blocks 1..29 receive no LQ add.
    """

    geometry.validate()
    import openvino as ov
    from openvino import opset13 as ops

    x = ops.parameter(list(geometry.token_shape), ov.Type.f16, name="x_before_lq")
    lq = ops.parameter(list(geometry.token_shape), ov.Type.f16, name="lq_block0")
    output = ops.add(x, lq)
    output.output(0).get_tensor().set_names({"x_with_lq_block0"})
    model = ov.Model([output], [x, lq], f"wan_lq_injection_{geometry.name}_f16")
    model.validate_nodes_and_infer_types()
    return model


def layer0_abi(geometry: GeometrySpec) -> dict[str, object]:
    """Return names/shapes needed to connect prefix, LQ, and one block."""

    geometry.validate()
    return {
        "geometry": {
            "name": geometry.name,
            "f": geometry.f,
            "h": geometry.h,
            "w": geometry.w,
            "tokens": geometry.tokens,
            "latent_shape": list(geometry.latent_shape),
        },
        "prefix_outputs": {
            "tokens": [list(geometry.token_shape), "F16"],
            "t": [[1, 1, DIM], "F16"],
            "t_mod": [[1, 6, DIM], "F16"],
        },
        "lq": {"input": [list(geometry.token_shape), "F16"], "output": [list(geometry.token_shape), "F16"]},
        "block": {
            "pre": "reviewed block/graph.py build_pre_graph",
            "attention": "reviewed GatherND+SDPA selected-key graph",
            "post": "selected conv_linears first self.o boundary",
            "self_cache": [list(geometry.cache_shape), "F16"],
            "cross_kv": [[1, 512, DIM], "F16 per layer"],
            "rope": [[1, geometry.tokens, 1, HEAD_DIM // 2], "F32"],
        },
        "source_order": ["prefix", "lq_add_for_block0_only", "pre_i", "attention_i", "post_i"]
        + ["cache_update_i"],
    }

