"""Graph-only checks for the optional external decoder-weight boundary.

The test deliberately constructs the authenticated source module without its
checkpoint and never creates ``ov.Core``.  It proves graph topology, ordinary
BF16 Parameter registration, and the two runtime child-bank contracts; it is
not a numerical or device qualification.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import openvino as ov
from openvino.frontend import FrontEndManager
import torch

import plus_decoder
from plus_phased_decoder import (
    STATE_NAMES,
    build_front,
    build_growth_history,
    build_phased_decoder,
    build_remote_frame_child,
)


def _names(ports):
    return [port.get_any_name() for port in ports]


def _bf16_ports(model):
    return (*model.inputs, *model.outputs)


class ExternalDecoderBuilderCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        source = plus_decoder._load_source_module()
        cls.decoder = source.TAEHV(
            checkpoint_path=None,
            channels=list(plus_decoder.CHANNELS),
            latent_channels=plus_decoder.LATENT_CHANNELS,
        ).to(device="cpu", dtype=torch.float32).eval()
        cls.front_default, cls.front_default_meta = build_front(cls.decoder, (16, 16))
        cls.models_default, cls.phases_default = build_phased_decoder(cls.decoder, (16, 16))
        cls.growth_default, cls.growth_banks, cls.growth_meta = build_growth_history(
            cls.decoder, (16, 16)
        )
        cls.frame_default, cls.frame_banks, cls.frame_meta = build_remote_frame_child(
            cls.decoder, (16, 16)
        )

        cls.factory_calls = []

        def factory(source_parameter, shape):
            index = len(cls.factory_calls)
            name = f"external_decoder_{index}"
            parameter = ov.opset1.parameter(list(shape), ov.Type.bf16, name=name)
            parameter.output(0).get_tensor().set_names({name})
            cls.factory_calls.append((source_parameter, tuple(shape), parameter))
            return parameter.output(0)

        cls.factory = factory
        cls.front_external, cls.front_external_meta = build_front(
            cls.decoder, (16, 16), learned_parameter_factory=factory
        )
        cls.growth_external, cls.growth_external_banks, cls.growth_external_meta = (
            build_growth_history(
                cls.decoder, (16, 16), learned_parameter_factory=factory
            )
        )
        cls.frame_external, cls.frame_external_banks, cls.frame_external_meta = (
            build_remote_frame_child(
                cls.decoder, (16, 16), learned_parameter_factory=factory
            )
        )

    def test_default_phased_abi_and_front_only_contract(self):
        self.assertEqual(
            set(self.models_default), {"front", "history0", "history1", "frame0", "frame1"}
        )
        self.assertEqual(
            _names(self.front_default.inputs), ["packed", *STATE_NAMES[:6]]
        )
        self.assertEqual(
            _names(self.front_default.outputs),
            ["middle", *[f"{name}_out" for name in STATE_NAMES[:6]],
             *[f"residual_linear_{index}" for index in range(6)]],
        )
        self.assertEqual(self.front_default_meta["residual_count"], 6)
        self.assertEqual(self.front_default_meta["learned_parameter_nodes"], 0)
        self.assertEqual(self.phases_default["phase_names"],
                         ["front", "history0", "history1", "frame0", "frame1"])
        self.assertEqual(self.phases_default["residual_output_names"],
                         [f"residual_linear_{index}" for index in range(12)])
        for model in self.models_default.values():
            self.assertTrue(all(port.get_element_type() == ov.Type.bf16
                                for port in _bf16_ports(model)))
        self.assertEqual(_names(self.models_default["history0"].inputs),
                         ["middle", *STATE_NAMES[6:]])
        self.assertEqual(_names(self.models_default["history1"].inputs),
                         ["middle", *STATE_NAMES[6:]])
        for name, start in (("history0", 6), ("history1", 9)):
            self.assertEqual(
                _names(self.models_default[name].outputs),
                ["middle", *[f"{state}_out" for state in STATE_NAMES[6:]],
                 *[f"residual_linear_{index}" for index in range(start, start + 3)]],
            )
        self.assertEqual(_names(self.models_default["frame0"].inputs), ["middle"])
        self.assertEqual(_names(self.models_default["frame1"].inputs), ["middle"])
        self.assertEqual(_names(self.models_default["frame0"].outputs), ["frame"])
        self.assertEqual(_names(self.models_default["frame1"].outputs), ["frame"])

    def test_factory_accepts_parameter_nodes_as_well_as_outputs(self):
        nodes = []

        def factory(_source, shape):
            name = f"node_weight_{len(nodes)}"
            node = ov.opset1.parameter(list(shape), ov.Type.bf16, name=name)
            node.output(0).get_tensor().set_names({name})
            nodes.append(node)
            return node

        model, _, metadata = build_remote_frame_child(
            self.decoder, (16, 16), learned_parameter_factory=factory
        )
        self.assertEqual(metadata["learned_parameter_nodes"], 4)
        self.assertEqual(_names(model.inputs)[2:], [f"node_weight_{i}" for i in range(4)])
        self.assertTrue(all(port.get_node().get_type_name() == "Parameter" for port in model.inputs))

    def test_default_growth_and_frame_banks_are_owned_arrays(self):
        self.assertEqual(_names(self.growth_default.inputs),
                         ["middle", "grow_weight", *STATE_NAMES[6:]])
        self.assertEqual(_names(self.growth_default.outputs),
                         ["S0", "S1", "S2", "middle",
                          "residual_linear_6", "residual_linear_7",
                          "residual_linear_8"])
        self.assertEqual(_names(self.frame_default.inputs),
                         ["middle", "weight__frame_grow"])
        self.assertEqual(_names(self.frame_default.outputs), ["frame"])
        self.assertEqual(self.growth_meta["child_bank_mode"], "bf16_arrays")
        self.assertEqual(self.frame_meta["child_bank_mode"], "bf16_arrays")
        for banks, shape in ((self.growth_banks, (256, 256, 1, 1)),
                             (self.frame_banks, (128, 128, 1, 1))):
            self.assertEqual(len(banks), 2)
            for bank in banks:
                self.assertIsInstance(bank, np.ndarray)
                self.assertEqual(bank.dtype, np.uint16)
                self.assertEqual(bank.shape, shape)
                self.assertTrue(bank.flags.c_contiguous)
        _, growth_again, _ = build_growth_history(self.decoder, (16, 16))
        _, frame_again, _ = build_remote_frame_child(self.decoder, (16, 16))
        for old, new in zip(self.growth_banks, growth_again):
            np.testing.assert_array_equal(old, new)
        for old, new in zip(self.frame_banks, frame_again):
            np.testing.assert_array_equal(old, new)

    def test_external_front_registers_only_ordinary_bf16_parameters(self):
        base = ["packed", *STATE_NAMES[:6]]
        names = _names(self.front_external.inputs)
        self.assertEqual(names[:len(base)], base)
        learned = names[len(base):]
        self.assertEqual(len(learned), self.front_external_meta["learned_parameter_nodes"])
        self.assertTrue(learned)
        self.assertTrue(all(name.startswith("external_decoder_") for name in learned))
        self.assertEqual(
            _names(self.front_external.outputs),
            ["middle", *[f"{name}_out" for name in STATE_NAMES[:6]],
             *[f"residual_linear_{index}" for index in range(6)]],
        )
        self.assertTrue(all(port.get_element_type() == ov.Type.bf16
                            for port in _bf16_ports(self.front_external)))
        self.assertTrue(all(source.__class__.__name__ == "Parameter"
                            for source, _, _ in self.factory_calls))
        self.assertTrue(all(tuple(shape) and node.get_type_name() == "Parameter"
                            for _, shape, node in self.factory_calls))
        self.assertFalse(any(output.get_element_type() == ov.Type.f16
                             for node in self.front_external.get_ordered_ops()
                             for output in node.outputs()))

    def test_external_growth_and_frame_keep_runtime_bank_inputs(self):
        growth_names = _names(self.growth_external.inputs)
        growth_base = ["middle_in", "grow_weight", *STATE_NAMES[6:]]
        self.assertEqual(growth_names[:len(growth_base)], growth_base)
        self.assertNotIn("weight__frame_grow", growth_names)
        self.assertEqual(self.growth_external_meta["growth_weight_input"], "grow_weight")
        self.assertEqual(self.growth_external_meta["child_bank_mode"],
                         "source_parameter_slices")
        self.assertEqual(self.growth_external_meta["learned_parameter_nodes"],
                         len(growth_names) - len(growth_base))
        self.assertEqual(_names(self.growth_external.outputs),
                         ["S0", "S1", "S2", "middle",
                          "residual_linear_6", "residual_linear_7",
                          "residual_linear_8"])
        self.assertTrue(all(isinstance(bank, dict) for bank in self.growth_external_banks))
        self.assertEqual(
            [(bank["source_block"], bank["child"], bank["start"], bank["stop"])
             for bank in self.growth_external_banks],
            [(15, 0, 0, 256), (15, 1, 256, 512)],
        )
        self.assertTrue(all(
            bank["source_parameter"] is self.decoder.decoder[15].conv.weight
            and bank["source_parameter_name"] == "decoder.15.conv.weight"
            for bank in self.growth_external_banks
        ))
        self.assertEqual([bank["shape"] for bank in self.growth_external_banks],
                         [(256, 256, 1, 1)] * 2)
        frame_names = _names(self.frame_external.inputs)
        self.assertEqual(frame_names[:2], ["middle", "frame_grow"])
        self.assertNotIn("weight__frame_grow", frame_names)
        self.assertEqual(self.frame_external_meta["input_names"], frame_names)
        self.assertEqual(self.frame_external_meta["child_bank_mode"],
                         "source_parameter_slices")
        self.assertEqual(
            [(bank["source_block"], bank["child"], bank["start"], bank["stop"])
             for bank in self.frame_external_banks],
            [(21, 0, 0, 128), (21, 1, 128, 256)],
        )
        self.assertTrue(all(
            bank["source_parameter"] is self.decoder.decoder[21].conv.weight
            and bank["source_parameter_name"] == "decoder.21.conv.weight"
            for bank in self.frame_external_banks
        ))
        self.assertEqual([bank["shape"] for bank in self.frame_external_banks],
                         [(128, 128, 1, 1)] * 2)

    def test_external_graph_ports_survive_ir_readback_without_renaming(self):
        frontend = FrontEndManager().load_by_framework("ir")
        self.assertIsNotNone(frontend)

        def abi(ports):
            return [(set(port.get_names()), str(port.get_element_type()), tuple(port.get_shape()))
                    for port in ports]

        with tempfile.TemporaryDirectory(prefix="flash-decoder-ir-") as directory:
            for name, model in (("front", self.front_external),
                                ("grow_memory", self.growth_external),
                                ("remote_frame_child", self.frame_external)):
                with self.subTest(phase=name):
                    path = Path(directory) / f"{name}.xml"
                    ov.serialize(model, str(path))
                    loaded = frontend.convert(frontend.load(str(path)))
                    self.assertEqual(abi(loaded.inputs), abi(model.inputs))
                    self.assertEqual(abi(loaded.outputs), abi(model.outputs))

    def test_factory_is_fail_closed(self):
        with self.assertRaises(TypeError):
            build_front(self.decoder, (16, 16), learned_parameter_factory=object())

        def none_factory(source, shape):
            return None

        with self.assertRaises(TypeError):
            build_front(self.decoder, (16, 16), learned_parameter_factory=none_factory)

        def f32_factory(source, shape):
            return ov.opset1.parameter(list(shape), ov.Type.f32, name="wrong_f32").output(0)

        with self.assertRaises(TypeError):
            build_growth_history(self.decoder, (16, 16), learned_parameter_factory=f32_factory)

        def unnamed_factory(source, shape):
            return ov.opset1.parameter(list(shape), ov.Type.bf16).output(0)

        with self.assertRaises(ValueError):
            build_remote_frame_child(self.decoder, (16, 16),
                                     learned_parameter_factory=unnamed_factory)


if __name__ == "__main__":
    unittest.main()
