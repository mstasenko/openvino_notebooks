#!/usr/bin/env python3
"""CPU-only contract tests for export-flash-decoder.py.

These tests exercise the exporter contract without an OpenVINO Core, device,
compile, or inference request.  The authenticated graph build itself is run
by the bounded structural-export command, while these fakes cover the
fail-closed ABI checks and the identity-only learned-Parameter seam.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "export-flash-decoder.py"
SPEC = importlib.util.spec_from_file_location("export_flash_decoder_tested", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeNode:
    def __init__(self, type_name="Parameter", element_type="bf16", name="node"):
        self._type_name = type_name
        self._element_type = element_type
        self._name = name

    def get_type_name(self):
        return self._type_name

    def get_friendly_name(self):
        return self._name

    def outputs(self):
        return [_FakePort(self._name, (), self._element_type, self)]

    def get_output_shape(self, _index):
        return ()


class _FakePort:
    def __init__(self, name, shape, element_type, node=None, names=None):
        self._name = name
        self._shape = tuple(shape)
        self._element_type = element_type
        self._node = node or _FakeNode(name=name)
        self._names = {name} if names is None else set(names)

    def get_names(self):
        return self._names

    def get_any_name(self):
        return self._name

    def get_shape(self):
        return self._shape

    def get_element_type(self):
        return self._element_type

    def get_node(self):
        return self._node

    def get_tensor(self):
        return self

    def set_names(self, _names):
        return None


class _FakeModel:
    def __init__(self, *, bad_f16=False, hidden=False, learned_constant=False,
                 missing_learned=False, extra_learned=False, renamed_learned=False,
                 bad_state_shape=False, bad_state_type=False, bad_residual_shape=False,
                 wrong_result=False, variable=False, sink=False, input_alias=False,
                 non_parameter_input=False):
        state_shapes = (
            (1, 512, 1, 1), (1, 512, 1, 1), (1, 512, 1, 1),
            (1, 256, 2, 2), (1, 256, 2, 2), (1, 256, 2, 2),
            (1, 128, 4, 4), (1, 128, 4, 4), (1, 128, 4, 4),
        )
        self.inputs = []
        inputs = [("packed", (1, 784, 1, 1))]
        inputs.extend(zip(MODULE.STATE_NAMES, state_shapes))
        inputs.extend((f"weight__p{index}", (1, 1, 1, 1)) for index in range(66))
        if missing_learned:
            inputs.pop()
        if extra_learned:
            inputs.append(("weight__extra", (1, 1, 1, 1)))
        if renamed_learned:
            inputs[-1] = ("weight__renamed", inputs[-1][1])
        if bad_state_shape:
            inputs[1] = (MODULE.STATE_NAMES[0], (1, 512, 2, 1))
        for name, shape in inputs:
            node_type = "Result" if non_parameter_input and name == "packed" else "Parameter"
            element_type = "f16" if bad_state_type and name == MODULE.STATE_NAMES[0] else "bf16"
            names = {name, "alias"} if input_alias and name == "packed" else None
            node = _FakeNode(node_type, element_type, name)
            self.inputs.append(_FakePort(name, shape, element_type, node, names=names))
        self.outputs = []
        outputs = [("frames", (1, 4, 3, 8, 8))]
        outputs.extend(zip((f"{name}_out" for name in MODULE.STATE_NAMES), state_shapes))
        residual_shapes = [state_shapes[0]] * 3 + [state_shapes[3]] * 3 + [state_shapes[6]] * 6
        outputs.extend(zip(MODULE.RESIDUAL_NAMES, residual_shapes))
        for name, shape in outputs:
            element_type = "bf16"
            if bad_residual_shape and name == MODULE.RESIDUAL_NAMES[0]:
                shape = (1, 512, 2, 1)
            node_type = "Parameter" if wrong_result and name == "frames" else "Result"
            node = _FakeNode(node_type, element_type, name)
            self.outputs.append(_FakePort(name, shape, "bf16", node))
        self._bad_f16 = bad_f16
        self._hidden = hidden
        self._learned_constant = learned_constant
        self._variable = variable
        self._sink = sink

    def get_ordered_ops(self):
        nodes = []
        if self._bad_f16:
            nodes.append(_FakeNode("Convert", "f16", "bad_f16"))
        if self._hidden:
            nodes.append(_FakeNode("ReadValue", "bf16", "state"))
        if self._learned_constant:
            nodes.append(_FakeNode("Constant", "bf16", "weight__p0"))
        return nodes

    def get_variables(self):
        return [object()] if self._hidden or self._variable else []

    def get_sinks(self):
        return [object()] if self._sink else []

    def get_friendly_name(self):
        return "fake_decoder"


class _FakeParameter:
    def __init__(self, shape):
        self.shape = tuple(shape)


def _phase_tile_plan():
    return {
        "source_hw": (2176, 3840),
        "tile_source_hw": (184, 3840),
        "tile_count": 16,
        "halo_rows": 12,
        "tiles": tuple({
            "name": f"tile{index}",
            "input_rows": (0, 92), "output_rows": (12, 80),
            "frame_rows": (index * 68, (index + 1) * 68),
            "compact_rows": (0, 46),
            "source_frame_rows": (index * 136, (index + 1) * 136),
            "source_output_rows": (24, 160),
        } for index in range(16)),
    }


class _FakeCreatedParameter:
    def __init__(self, shape, name):
        self._shape = tuple(shape)
        self._name = name
        self._output = _FakePort(name, shape, "bf16", _FakeNode("Parameter", "bf16", name))

    def set_friendly_name(self, _name):
        return None

    def output(self, _index):
        return self._output


class _FakeOpset:
    @staticmethod
    def parameter(shape, _element_type, name):
        return _FakeCreatedParameter(shape, name)


class _FakeOV:
    opset1 = _FakeOpset()
    Type = types.SimpleNamespace(bf16="bf16")


class ExportFlashDecoderCpuTests(unittest.TestCase):
    def test_identity_factory_is_not_shape_or_value_matching(self):
        first = _FakeParameter((4, 4))
        second = _FakeParameter((4, 4))
        factory, ports = MODULE._make_parameter_factory(
            _FakeOV(), [("decoder.0.weight", first), ("decoder.1.weight", second)]
        )
        node = factory(first, (4, 4))
        self.assertEqual(node.get_any_name(), "weight__decoder_0_weight")
        self.assertEqual(set(ports), {"decoder.0.weight"})
        with self.assertRaises(ValueError):
            factory(_FakeParameter((4, 4)), (4, 4))
        with self.assertRaises(ValueError):
            factory(first, (2, 8))
        with self.assertRaises(ValueError):
            factory(first, (4, 4, 1, 1))

    def test_contract_accepts_exact_complete_state_graph(self):
        named = [(f"p{index}", _FakeParameter((1,))) for index in range(66)]
        geometry = {"source_hw": (8, 8), "latent_hw": (1, 1),
                    "state_shapes": ((1, 512, 1, 1),) * 3 +
                    ((1, 256, 2, 2),) * 3 + ((1, 128, 4, 4),) * 3}
        report = MODULE._check_contract(_FakeModel(), geometry, named)
        self.assertEqual(report["learned_parameter_count"], 66)
        self.assertEqual(len(report["private_residual_results"]), 12)
        self.assertEqual(report["constant_node_count"], 0)

    def test_contract_rejects_f16_hidden_state_and_learned_constant(self):
        named = [(f"p{index}", _FakeParameter((1,))) for index in range(66)]
        geometry = {"source_hw": (8, 8), "latent_hw": (1, 1),
                    "state_shapes": ((1, 512, 1, 1),) * 3 +
                    ((1, 256, 2, 2),) * 3 + ((1, 128, 4, 4),) * 3}
        for kwargs in ({"bad_f16": True}, {"hidden": True}, {"learned_constant": True}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MODULE._check_contract(_FakeModel(**kwargs), geometry, named)

    def test_contract_rejects_boundary_mutations(self):
        named = [(f"p{index}", _FakeParameter((1,))) for index in range(66)]
        geometry = {"source_hw": (8, 8), "latent_hw": (1, 1),
                    "state_shapes": ((1, 512, 1, 1),) * 3 +
                    ((1, 256, 2, 2),) * 3 + ((1, 128, 4, 4),) * 3}
        mutations = (
            {"missing_learned": True}, {"extra_learned": True},
            {"renamed_learned": True}, {"bad_state_shape": True},
            {"bad_state_type": True}, {"bad_residual_shape": True},
            {"wrong_result": True}, {"variable": True}, {"sink": True},
            {"input_alias": True}, {"non_parameter_input": True},
        )
        for mutation in mutations:
            with self.subTest(**mutation), self.assertRaises(ValueError):
                MODULE._check_contract(_FakeModel(**mutation), geometry, named)

    def test_geometry_and_explicit_arguments_are_fail_closed(self):
        self.assertEqual(MODULE._parse_hw("8x16"), (16, 8))
        for value in ("16", "8x", "x8", "8x-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE._parse_hw(value)
        signature = inspect.signature(MODULE.export_decoder)
        self.assertIsNone(signature.parameters["decoder_source"].default)
        self.assertTrue(signature.parameters["source_hw"].default is inspect.Parameter.empty)

    def test_phased_geometry_is_explicit_and_uses_history_helper(self):
        calls = []

        def helper(source_hw, *, tile_count, halo_rows):
            calls.append((source_hw, tile_count, halo_rows))
            return _phase_tile_plan()

        plan = MODULE._reviewed_phased_tiles((2176, 3840), helper)
        self.assertEqual(plan["tile_source_hw"], (184, 3840))
        self.assertEqual(calls, [((2176, 3840), 16, 12)])
        with self.assertRaises(ValueError):
            MODULE._reviewed_phased_tiles((2160, 3840), helper)
        bad = dict(plan, tile_source_hw=(192, 3840))
        with self.assertRaises(ValueError):
            MODULE._reviewed_phased_tiles((2176, 3840), lambda *args, **kwargs: bad)

    def test_phased_static_bindings_and_child_descriptor_identity(self):
        names = [f"decoder.{block}.p{index}" for block, count in ((0, 41), (14, 19), (20, 4))
                 for index in range(count)]
        names += ["decoder.15.conv.weight", "decoder.21.conv.weight"]
        source = {name: _FakeParameter((4, 4)) for name in names[:-2]}
        source.update({name: _FakeParameter((4, 4, 1, 1)) for name in names[-2:]})
        ports = {
            name: _FakePort("weight__" + name.replace(".", "_"), (4, 4), "bf16")
            for name in names[:-2]
        }
        records, phase_ports = MODULE._static_bindings(list(source.items()), ports)
        self.assertEqual(len(records), 64)
        self.assertEqual({phase: len(values) for phase, values in phase_ports.items()},
                         {"front": 41, "grow_memory": 19, "remote_frame_child": 4})
        first_name = "weight__" + names[0].replace(".", "_")
        for bad_port in (
            _FakePort("weight__wrong_identity", (4, 4), "bf16"),
            _FakePort(first_name, (2, 8), "bf16"),
            _FakePort(first_name, (4, 4), "f16"),
            _FakePort(first_name, (4, 4), "bf16", _FakeNode("Constant")),
        ):
            with self.subTest(port=bad_port), self.assertRaises(ValueError):
                MODULE._static_bindings(list(source.items()), {**ports, names[0]: bad_port})
        unknown_source = dict(source)
        unknown_source["decoder.27.weight"] = unknown_source.pop(names[0])
        unknown_ports = dict(ports)
        unknown_ports.pop(names[0])
        unknown_ports["decoder.27.weight"] = _FakePort("weight__decoder_27_weight", (4, 4), "bf16")
        with self.assertRaises(ValueError):
            MODULE._static_bindings(list(unknown_source.items()), unknown_ports)
        banks = [
            {"source_parameter": source["decoder.15.conv.weight"],
             "source_parameter_name": "decoder.15.conv.weight", "source_block": 15,
             "child": child, "start": child * 2, "stop": (child + 1) * 2,
             "shape": (2, 4, 1, 1), "source_dtype": "f32", "storage": "bf16"}
            for child in range(2)
        ]
        record = MODULE._validate_bank_descriptors(
            "grow_memory", "grow_weight", banks, source,
            "decoder.15.conv.weight", (4, 4, 1, 1), (2, 4, 1, 1), 2,
        )
        self.assertEqual(record["slices"][1]["start"], 2)
        wrong_identity = [dict(banks[0]), dict(banks[1])]
        wrong_identity[0]["source_parameter"] = _FakeParameter((4, 4))
        with self.assertRaises(ValueError):
            MODULE._validate_bank_descriptors(
                "grow_memory", "grow_weight", wrong_identity, source,
                "decoder.15.conv.weight", (4, 4, 1, 1), (2, 4, 1, 1), 2,
            )
        wrong_identity = [dict(banks[0]), dict(banks[1])]
        wrong_identity[1]["stop"] = 99
        with self.assertRaises(ValueError):
            MODULE._validate_bank_descriptors(
                "grow_memory", "grow_weight", wrong_identity, source,
                "decoder.15.conv.weight", (4, 4, 1, 1), (2, 4, 1, 1), 2,
            )

    def test_phased_contract_requires_exact_source_bound_inputs(self):
        geometry = {"source_hw": (8, 8), "state_shapes": ()}
        static = {f"weight__decoder_20_p{index}": _FakePort(
            f"weight__decoder_20_p{index}", (4, 4), "bf16") for index in range(4)}

        def model():
            result = _FakeModel()
            result.inputs = [
                _FakePort("middle", (1, 128, 4, 4), "bf16"),
                _FakePort("frame_grow", (128, 128, 1, 1), "bf16"),
                *static.values(),
            ]
            result.outputs = [_FakePort("frame", (1, 3, 8, 8), "bf16", _FakeNode("Result"))]
            return result

        def check(value):
            return MODULE._check_phase_contract(
                value, "remote_frame_child", geometry, geometry, static)

        self.assertEqual(len(check(model())["static_parameter_names"]), 4)
        # A source weight requested by the factory but then left disconnected
        # must not disappear from the expected contract.
        missing = model()
        missing.inputs.pop()
        with self.assertRaises(ValueError):
            check(missing)
        foreign = model()
        foreign.inputs.append(_FakePort("weight__decoder_0_p0", (4, 4), "bf16"))
        with self.assertRaises(ValueError):
            check(foreign)
        copied = model()
        original = copied.inputs[-1]
        copied.inputs[-1] = _FakePort(original.get_any_name(), (4, 4), "bf16")
        with self.assertRaisesRegex(ValueError, "source binding"):
            check(copied)
        reordered = model()
        reordered.inputs[0], reordered.inputs[1] = reordered.inputs[1], reordered.inputs[0]
        with self.assertRaises(ValueError):
            check(reordered)

    def test_phased_cli_help_exposes_opt_in_without_changing_required_inputs(self):
        environment = {**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1",
                       "PYTHONPATH": str(HERE)}
        result = subprocess.run(
            [sys.executable, str(SOURCE), "--help"], check=False,
            capture_output=True, text=True, env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--bundle", result.stdout)
        self.assertNotIn("--phased-bundle", result.stdout)

    def test_source_override_is_restored_on_loader_failure(self):
        class _FakePlusDecoder:
            SOURCE = Path("original-tiny-model.py")

            @staticmethod
            def _load_source_module():
                raise RuntimeError("fixture failure")

        with self.assertRaises(RuntimeError):
            MODULE._load_meta_decoder(_FakePlusDecoder, Path("replacement-tiny_model.py"))
        self.assertEqual(_FakePlusDecoder.SOURCE, Path("original-tiny-model.py"))

    def test_cli_help_does_not_import_openvino_or_accept_missing_contract(self):
        environment = {**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, str(SOURCE), "--help"], check=False,
            capture_output=True, text=True, env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("decoder-source", result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(SOURCE), "--reference-root", directory,
                 "--decoder-source", str(Path(directory) / "tiny_model.py"),
                 "--source-size", "8x8", "--output", str(Path(directory) / "out")],
                check=False, capture_output=True, text=True, env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reference", result.stderr.lower())

    def test_exporter_uses_explicit_factory_and_has_no_runtime_execution(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("learned_parameter_factory", source)
        for forbidden in ("compile_model", ".infer(", "safe_open", "safetensors", "Core("):
            self.assertNotIn(forbidden, source)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr in {"get_property", "query"}]
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
