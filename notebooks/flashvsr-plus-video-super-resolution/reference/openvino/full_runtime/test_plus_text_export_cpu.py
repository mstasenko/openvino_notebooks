#!/usr/bin/env python3
"""Build-only ABI checks for external BF16 FlashVSR+ text weights.

These tests construct OpenVINO graph objects only.  They deliberately do not
load learned arrays, create ``ov.Core``, compile, or execute an inference.
"""

from __future__ import annotations

import unittest

import plus_text
import plus_weights


class PlusTextExternalAbiCpuTests(unittest.TestCase):
    def test_external_text_parameters_have_fixed_bf16_abi(self) -> None:
        import openvino as ov

        model = plus_text.build_text_embedding(None, "bf16")
        expected = [
            "prompt",
            *("weight__" + key.replace(".", "_") for key in plus_weights.TEXT_KEYS),
        ]
        self.assertEqual([port.get_any_name() for port in model.inputs], expected)
        parameters = {
            parameter.output(0).get_any_name(): parameter
            for parameter in model.parameters
        }
        self.assertEqual(set(parameters), set(expected))
        self.assertEqual(tuple(parameters["prompt"].shape), plus_text.PROMPT_SHAPE)
        self.assertEqual(parameters["prompt"].output(0).get_element_type(), ov.Type.bf16)
        for key, shape in plus_weights.TEXT_SHAPES.items():
            port = "weight__" + key.replace(".", "_")
            self.assertEqual(tuple(parameters[port].shape), shape)
            self.assertEqual(parameters[port].output(0).get_element_type(), ov.Type.bf16)

        self.assertEqual(
            {output.get_any_name() for output in model.outputs},
            {"context", "text_linear"},
        )
        for output in model.outputs:
            self.assertEqual(output.get_element_type(), ov.Type.bf16)
            self.assertEqual(tuple(output.shape), plus_text.TEXT_SHAPE)
        gelus = [node for node in model.get_ordered_ops() if node.get_type_name() == "Gelu"]
        self.assertEqual(len(gelus), 1)
        self.assertEqual(gelus[0].get_attributes().get("approximation_mode").lower(), "tanh")
        learned_constants = [
            node for node in model.get_ordered_ops()
            if node.get_type_name() == "Constant"
            and any(name.startswith("weight__") for name in node.output(0).get_names())
        ]
        self.assertFalse(learned_constants)

    def test_external_storage_boundary_can_be_hidden_without_changing_abi(self) -> None:
        model = plus_text.build_text_embedding(None, "bf16", preserve_storage=False)
        self.assertEqual([output.get_any_name() for output in model.outputs], ["context"])
        self.assertEqual(
            [port.get_any_name() for port in model.inputs],
            [
                "prompt",
                *("weight__" + key.replace(".", "_") for key in plus_weights.TEXT_KEYS),
            ],
        )

    def test_external_mode_rejects_non_bf16_and_malformed_constant_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "external Plus text weights require BF16"):
            plus_text.build_text_embedding(None, "f16")
        with self.assertRaisesRegex(ValueError, "external Plus text weights require BF16"):
            plus_text.build_text_embedding(None, "f32")
        with self.assertRaisesRegex(ValueError, "weight key mismatch"):
            plus_text.build_text_embedding({"wrong": object()}, "bf16")
        with self.assertRaisesRegex(ValueError, "weight key mismatch"):
            plus_text.build_text_embedding({}, "bf16")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
