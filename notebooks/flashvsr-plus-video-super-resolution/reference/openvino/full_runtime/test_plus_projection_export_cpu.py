"""CPU-only structural tests for the external-weight LQ projection ABI."""

from __future__ import annotations

import unittest

from plus_projection import (
    STAGE2_WEIGHT_KEYS,
    STARTUP_WEIGHT_KEYS,
    WEIGHT_KEYS,
    WEIGHT_PORTS,
    WEIGHT_SHAPES,
    build_projection_graph,
    split_projection_graph,
)


def _names(model):
    return {parameter.output(0).get_any_name() for parameter in model.parameters}


class PlusProjectionExternalWeightsCpuTests(unittest.TestCase):
    def test_parameterized_startup_and_steady_have_exact_bf16_abi(self):
        import openvino as ov

        startup = build_projection_graph(None, True, "bf16", height=16, width=16)
        steady = build_projection_graph(None, False, "bf16", height=16, width=16)
        self.assertEqual(
            _names(startup), {"video", *(WEIGHT_PORTS[key] for key in STARTUP_WEIGHT_KEYS)}
        )
        self.assertEqual(
            _names(steady), {"video", "cache1", "cache2",
                              *(WEIGHT_PORTS[key] for key in WEIGHT_KEYS)}
        )
        for model in (startup, steady):
            for parameter in model.parameters:
                self.assertEqual(parameter.output(0).get_element_type(), ov.Type.bf16)
            learned_constants = [
                node for node in model.get_ordered_ops()
                if node.get_type_name() == "Constant"
                and any(str(name).startswith("weight__")
                        for name in node.output(0).get_names())
            ]
            self.assertFalse(learned_constants)

    def test_split_carries_only_reachable_learned_roles(self):
        steady = build_projection_graph(None, False, "bf16", height=16, width=16)
        stage1, stage2 = split_projection_graph(steady)
        self.assertEqual(
            _names(stage1), {"video", "cache1",
                              *(WEIGHT_PORTS[key] for key in STARTUP_WEIGHT_KEYS)}
        )
        self.assertEqual(
            _names(stage2), {"old_cache2", "current_cache2",
                              *(WEIGHT_PORTS[key] for key in STAGE2_WEIGHT_KEYS)}
        )
        self.assertEqual({port.get_any_name() for port in stage1.outputs}, {"cache1", "cache2"})
        self.assertEqual({port.get_any_name() for port in stage2.outputs}, {"projected"})

    def test_constant_path_argument_validation_and_external_precision_guard(self):
        with self.assertRaises(ValueError):
            build_projection_graph({}, True, "bf16", height=16, width=16)
        with self.assertRaises(ValueError):
            build_projection_graph({"weight__unexpected": object()}, True, "bf16",
                                   height=16, width=16)
        with self.assertRaises(ValueError):
            build_projection_graph(None, True, "f16", height=16, width=16)
        with self.assertRaises(ValueError):
            build_projection_graph(None, False, "bf16", height=15, width=16)

    def test_split_rejects_renamed_learned_parameter(self):
        model = build_projection_graph(None, False, "bf16", height=16, width=16)
        renamed = model.clone()
        parameter = next(
            item for item in renamed.parameters
            if item.output(0).get_any_name() == WEIGHT_PORTS["norm2.weight"]
        )
        parameter.output(0).get_tensor().set_names({"weight__unexpected"})
        with self.assertRaises(ValueError):
            split_projection_graph(renamed)

    def test_weight_shapes_are_the_external_contract(self):
        self.assertEqual(set(WEIGHT_SHAPES), set(WEIGHT_KEYS))
        self.assertEqual(set(STARTUP_WEIGHT_KEYS) | set(STAGE2_WEIGHT_KEYS), set(WEIGHT_KEYS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
