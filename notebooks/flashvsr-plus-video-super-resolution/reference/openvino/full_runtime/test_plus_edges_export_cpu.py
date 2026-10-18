"""Build-only checks for the external BF16 Plus edge graph ABI.

These tests construct OpenVINO IR on the host but never create a Core,
compile a model, or execute inference.  Learned values must remain external
BF16 Parameters; the historical source-F32 mapping is covered separately by
``test_plus_edges_cpu.py``.
"""

from __future__ import annotations

import unittest

import numpy as np
import openvino as ov

from frame_geometry import FrameGeometry
from plus_edges import (
    EXPECTED_SHAPES,
    PREFIX_WEIGHT_KEYS,
    SUFFIX_WEIGHT_KEYS,
    WEIGHT_PORTS,
    build_prefix,
    build_suffix,
    edge_shapes,
)
from plus_graphs import BlockGeometry


GEOMETRY = BlockGeometry(FrameGeometry(128, 128))


def _parameter_names(model):
    return {parameter.output(0).get_any_name() for parameter in model.parameters}


def _output_names(model):
    return {output.get_any_name() for output in model.outputs}


def _learned_constants(model):
    return [
        node
        for node in model.get_ordered_ops()
        if node.get_type_name() == "Constant"
        and any(str(name).startswith("weight__") for name in node.output(0).get_names())
    ]


class PlusEdgesExternalWeightsCpuTests(unittest.TestCase):
    def test_external_prefix_has_only_reachable_bf16_parameters(self):
        model = build_prefix(None, GEOMETRY, storage="bf16")
        name = GEOMETRY.name
        expected_parameters = {
            f"{name}_x",
            "t",
            "t_mod",
            *(WEIGHT_PORTS[key] for key in PREFIX_WEIGHT_KEYS),
        }
        self.assertEqual(_parameter_names(model), expected_parameters)
        self.assertEqual(
            _output_names(model),
            {f"{name}_tokens_out", f"{name}_t_out", f"{name}_t_mod_out",
             f"{name}_embedding_out"},
        )
        self.assertEqual(
            [list(output.get_shape()) for output in model.outputs],
            [list(edge_shapes(GEOMETRY)[key]) for key in ("tokens", "t", "t_mod", "embedding")],
        )
        self.assertTrue(
            all(parameter.output(0).get_element_type() == ov.Type.bf16
                for parameter in model.parameters)
        )
        self.assertFalse(_learned_constants(model))
        self.assertFalse(
            any(output.get_element_type() == ov.Type.f16
                for node in model.get_ordered_ops() for output in node.outputs())
        )

    def test_external_suffix_has_only_reachable_bf16_parameters(self):
        model = build_suffix(None, GEOMETRY, storage="bf16")
        name = GEOMETRY.name
        expected_parameters = {
            f"{name}_tokens_in",
            f"{name}_t_in",
            *(WEIGHT_PORTS[key] for key in SUFFIX_WEIGHT_KEYS),
        }
        self.assertEqual(_parameter_names(model), expected_parameters)
        self.assertEqual(_output_names(model), {f"{name}_out"})
        self.assertEqual(list(model.output(0).get_shape()), list(edge_shapes(GEOMETRY)["output"]))
        self.assertEqual(model.output(0).get_element_type(), ov.Type.bf16)
        self.assertFalse(_learned_constants(model))
        self.assertFalse(
            any(output.get_element_type() == ov.Type.f16
                for node in model.get_ordered_ops() for output in node.outputs())
        )

    def test_external_prefix_exposes_fixed_conditioning_inputs(self):
        model = build_prefix(None, GEOMETRY, storage="bf16")
        self.assertIn("t", _parameter_names(model))
        self.assertIn("t_mod", _parameter_names(model))
        self.assertNotIn("time_embedding_0_weight", _parameter_names(model))
        self.assertNotIn("time_projection_1_weight", _parameter_names(model))
        embedding = model.output(f"{GEOMETRY.name}_embedding_out")
        self.assertEqual(embedding.get_element_type(), ov.Type.bf16)
        self.assertNotIn(f"{GEOMETRY.name}_embedding_out", _parameter_names(model))

    def test_external_mode_rejects_non_bf16_and_bad_embedding(self):
        with self.assertRaises(ValueError):
            build_prefix(None, GEOMETRY, storage="f16")
        with self.assertRaises(ValueError):
            build_suffix(None, GEOMETRY, storage="f16")
        with self.assertRaises(ValueError):
            build_prefix(None, GEOMETRY, storage="f32")
        with self.assertRaises(ValueError):
            build_prefix(None, GEOMETRY,
                         embedding_f32=np.zeros((1, 255), dtype=np.float32))
        with self.assertRaises(ValueError):
            build_prefix(None, GEOMETRY,
                         embedding_f32=np.zeros((1, 256), dtype=np.float64))
        nonfinite = np.zeros((1, 256), dtype=np.float32)
        nonfinite[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_prefix(None, GEOMETRY, embedding_f32=nonfinite)

    def test_malformed_geometry_and_historical_weight_mapping_fail_closed(self):
        with self.assertRaises(ValueError):
            build_prefix(None, BlockGeometry(FrameGeometry(128, 128), 2), storage="bf16")
        with self.assertRaises(ValueError):
            build_prefix(None, type("BadGeometry", (), {"f": 4, "h": 7, "w": 8})())
        with self.assertRaises(ValueError):
            build_prefix({}, GEOMETRY, storage="bf16")
        with self.assertRaises(ValueError):
            build_suffix({}, GEOMETRY, storage="bf16")
        with self.assertRaises(ValueError):
            build_prefix([], GEOMETRY, storage="bf16")

    def test_parameter_shapes_are_the_declared_external_contract(self):
        prefix = build_prefix(None, GEOMETRY, storage="bf16")
        suffix = build_suffix(None, GEOMETRY, storage="bf16")
        for model, keys in ((prefix, PREFIX_WEIGHT_KEYS), (suffix, SUFFIX_WEIGHT_KEYS)):
            for key in keys:
                parameter = next(
                    item for item in model.parameters
                    if item.output(0).get_any_name() == WEIGHT_PORTS[key]
                )
                self.assertEqual(tuple(parameter.output(0).get_shape()), EXPECTED_SHAPES[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
