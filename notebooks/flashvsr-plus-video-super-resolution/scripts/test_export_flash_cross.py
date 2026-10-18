#!/usr/bin/env python3
"""CPU-only contract tests for the Flash cross-K/V structural exporter."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from flash_export import assert_no_private_paths


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "export-flash-cross.py"
SPEC = importlib.util.spec_from_file_location("export_flash_cross_tested", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeNode:
    def __init__(self, type_name, friendly_name, outputs=()):
        self._type_name = type_name
        self._friendly_name = friendly_name
        self._outputs = tuple(outputs)

    def get_type_name(self):
        return self._type_name

    def get_friendly_name(self):
        return self._friendly_name

    def outputs(self):
        return list(self._outputs)

    def get_output_size(self):
        return len(self._outputs)

    def output(self, index):
        return self._outputs[index]


class _FakePort:
    def __init__(self, name, shape, element_type, *, names=None, node_type="Parameter"):
        self._name = name
        self._shape = tuple(shape)
        self._element_type = element_type
        self._names = {name} if names is None else set(names)
        self._node = _FakeNode(node_type, name)

    def get_any_name(self):
        return self._name

    def get_names(self):
        return self._names

    def get_shape(self):
        return self._shape

    def get_element_type(self):
        return self._element_type

    def get_node(self):
        return self._node


class _FakeModel:
    def __init__(self, *, extra=False, alias=False, state=False,
                 learned_constant=False, f16_internal=False, bad_output=False,
                 input_node_type="Parameter"):
        self.inputs = [
            _FakePort(name, shape, kind,
                      names={name, "alias"} if alias and name == "context" else None,
                      node_type=input_node_type)
            for name, (shape, kind) in MODULE.INPUT_CONTRACT.items()
        ]
        if extra:
            self.inputs.append(_FakePort("unexpected", (1,), "bf16"))
        self.outputs = [
            _FakePort(name, shape, kind,
                      node_type="ReadValue" if bad_output and name == "cross_k" else "Result")
            for name, (shape, kind) in MODULE.OUTPUT_CONTRACT.items()
        ]
        self._state = state
        self._learned_constant = learned_constant
        self._f16_internal = f16_internal

    def get_friendly_name(self):
        return "flashvsr_plus_cross_kv_bf16"

    def get_variables(self):
        return [object()] if self._state else []

    def get_sinks(self):
        return [object()] if self._state else []

    def get_ordered_ops(self):
        if self._learned_constant:
            return [_FakeNode("Constant", "weight__embedded")]
        if self._f16_internal:
            return [_FakeNode("Convert", "bad_f16", (
                _FakePort("bad_f16", (1,), "f16", node_type="Convert"),))]
        return []


class ExportFlashCrossCpuTests(unittest.TestCase):
    def test_contract_and_pinned_policy(self):
        self.assertEqual(MODULE.OPENVINO_VERSION, "2026.5.0-23457-7d81aad55a5")
        self.assertEqual(MODULE.PRECISION, "bf16")
        self.assertEqual(MODULE.TEXT_SHAPE, [1, 512, 1536])
        self.assertEqual(len(MODULE.CROSS_WEIGHTS), 5)
        report = MODULE.validate_graph_contract(
            _FakeModel(), ov=type("OV", (), {"Type": type("Type", (), {
                "bf16": "bf16", "f32": "f32",
            })})
        )
        self.assertEqual(report["learned_parameter_names"], sorted(MODULE.CROSS_WEIGHTS))
        self.assertEqual(report["state_variables"], [])

    def test_contract_rejects_extra_alias_state_constants_and_f16(self):
        fake_ov = type("OV", (), {"Type": type("Type", (), {
            "bf16": "bf16", "f32": "f32",
        })})
        for model in (
            _FakeModel(extra=True), _FakeModel(alias=True), _FakeModel(state=True),
            _FakeModel(learned_constant=True), _FakeModel(f16_internal=True),
            _FakeModel(bad_output=True), _FakeModel(input_node_type="Result"),
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_graph_contract(model, ov=fake_ov)

    def test_fresh_reference_and_private_path_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            for relative in MODULE.REQUIRED_REFERENCE_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# pinned\n", encoding="utf-8")
            self.assertEqual(MODULE.checked_reference_root(root), root.resolve())
            output = Path(directory) / "out"
            self.assertTrue(MODULE.fresh_output(output).is_dir())
            with self.assertRaises(ValueError):
                MODULE.fresh_output(output)
            linked = Path(directory) / "linked"
            linked.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.checked_reference_root(linked)
            with self.assertRaises(ValueError):
                assert_no_private_paths(output / "x", b"/" + b"cache/private")

    def test_source_is_build_only_and_no_learned_loading(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_from = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("numpy", imports | imported_from)
        source = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("compile_model", ".infer(", "safe_open", "Core("):
            self.assertNotIn(forbidden, source)

    def test_cli_help_does_not_need_openvino(self):
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, str(SOURCE), "--help"], check=False,
            capture_output=True, text=True, env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reference-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
