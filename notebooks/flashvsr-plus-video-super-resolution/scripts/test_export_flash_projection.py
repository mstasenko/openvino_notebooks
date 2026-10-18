#!/usr/bin/env python3
"""CPU-only contract tests for export-flash-projection.py."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "export-flash-projection.py"
SPEC = importlib.util.spec_from_file_location("export_flash_projection_tested", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeNode:
    def __init__(self, type_name, friendly_name, *, element_type="bf16"):
        self._type_name = type_name
        self._friendly_name = friendly_name
        self._element_type = element_type

    def get_type_name(self):
        return self._type_name

    def get_friendly_name(self):
        return self._friendly_name

    def get_output_size(self):
        return 1

    def output(self, _index):
        return _FakePort(self._friendly_name, (), self._element_type, node_type=self._type_name)

    def outputs(self):
        return [self.output(index) for index in range(self.get_output_size())]


class _FakePort:
    def __init__(self, name, shape, element_type, *, node_type="Parameter", names=None):
        self._name = name
        self._shape = tuple(shape)
        self._element_type = element_type
        self._node = _FakeNode(node_type, name)
        self._names = {name} if names is None else set(names)

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
    def __init__(self, graph_name, *, alias=False, extra=False, bad_output=False,
                 state=False, learned_constant=False, read_value=False,
                 bad_shape=False, internal_f16=False, input_node_type="Parameter"):
        contract = MODULE._graph_contracts()[graph_name]
        self.inputs = [
            _FakePort(name, (1, 2048, 2, 1, 1) if bad_shape and name == "old_cache2" else shape,
                      element_type,
                      node_type=input_node_type,
                      names={name, "alias"} if alias and name == "video" else None)
            for name, (shape, element_type) in contract["inputs"].items()
        ]
        if extra:
            self.inputs.append(_FakePort("unexpected", (1,), "bf16"))
        self.outputs = [
            _FakePort(name, shape, element_type,
                      node_type="ReadValue" if bad_output and name == "projected" else "Result")
            for name, (shape, element_type) in contract["outputs"].items()
        ]
        self._state = state
        self._learned_constant = learned_constant
        self._read_value = read_value
        self._internal_f16 = internal_f16

    def get_ordered_ops(self):
        if self._internal_f16:
            return [_FakeNode("Convert", "internal_f16", element_type="f16")]
        if self._read_value:
            return [_FakeNode("ReadValue", "state")]
        if self._learned_constant:
            return [_FakeNode("Constant", "weight__unexpected")]
        return []

    def get_variables(self):
        return [object()] if self._state else []

    def get_sinks(self):
        return []

    def get_friendly_name(self):
        return "fake_projection"


class ExportFlashProjectionCpuTests(unittest.TestCase):
    def test_pinned_geometry_and_contracts(self):
        self.assertEqual(MODULE.VISIBLE_HW, (2160, 3840))
        self.assertEqual(MODULE.WORKING_HW, (2176, 3840))
        self.assertEqual(MODULE.PRECISION, "bf16")
        self.assertEqual(MODULE.OPENVINO_VERSION, "2026.5.0-23457-7d81aad55a5")
        self.assertEqual(set(MODULE._graph_contracts()), {"startup", "stage1", "stage2"})
        self.assertIn("old_cache1", MODULE._graph_contracts()["stage1"]["inputs"])
        self.assertNotIn("cache1", MODULE._graph_contracts()["stage1"]["inputs"])

    def test_each_graph_contract_accepts_exact_ports(self):
        for graph_name in MODULE._graph_contracts():
            report = MODULE.validate_graph_contract(
                _FakeModel(graph_name), graph_name=graph_name
            )
            self.assertEqual(report["graph"], graph_name)
            self.assertEqual(report["state_variables"], [])
            self.assertEqual(report["sinks"], [])

    def test_contract_rejects_extra_alias_state_or_learned_constant(self):
        duplicate = _FakeModel("stage2")
        duplicate.outputs.append(duplicate.outputs[0])
        cross_direction = _FakeModel("stage1")
        old_cache1_index = next(
            index for index, port in enumerate(cross_direction.inputs)
            if port.get_any_name() == "old_cache1"
        )
        cross_direction.inputs[old_cache1_index] = _FakePort(
            "cache1", MODULE.DATA_INPUT_SHAPES["cache1"], "bf16"
        )
        bad_models = [
            ("stage2", _FakeModel("stage2", extra=True)),
            ("startup", _FakeModel("startup", alias=True)),
            ("stage2", _FakeModel("stage2", state=True)),
            ("stage2", _FakeModel("stage2", learned_constant=True)),
            ("stage2", _FakeModel("stage2", read_value=True)),
            ("stage2", _FakeModel("stage2", bad_output=True)),
            ("stage2", _FakeModel("stage2", bad_shape=True)),
            ("stage2", _FakeModel("stage2", input_node_type="Result")),
            ("stage2", duplicate),
            ("stage1", cross_direction),
        ]
        for graph_name, model in bad_models:
            with self.assertRaises(ValueError):
                MODULE.validate_graph_contract(model, graph_name=graph_name)

    def test_contract_rejects_internal_f16_like_native_validator(self):
        with self.assertRaises(ValueError):
            MODULE.validate_graph_contract(
                _FakeModel("stage2", internal_f16=True), graph_name="stage2"
            )

    def test_shared_helper_fingerprint_and_constant_bound(self):
        record = MODULE.helper_fingerprint()
        self.assertEqual(record["name"], "flash_export.py")
        self.assertEqual(len(record["sha256"]), 64)

        class _LargeConstant:
            def get_type_name(self):
                return "Constant"

            def get_friendly_name(self):
                return "structural_large"

            def get_output_size(self):
                return 1

            def output(self, _index):
                return _FakePort("structural_large", (600_000,), "bf16",
                                 node_type="Constant")

        model = type("_ConstantModel", (), {
            "get_ordered_ops": lambda self: [_LargeConstant()],
        })()
        with self.assertRaises(ValueError):
            MODULE.validate_constant_nodes(model)

        class _OpaqueConstant:
            def get_type_name(self):
                return "Constant"

            def get_friendly_name(self):
                return "opaque"

        opaque_model = type("_OpaqueModel", (), {
            "get_ordered_ops": lambda self: [_OpaqueConstant()],
        })()
        with self.assertRaises(ValueError):
            MODULE.validate_constant_nodes(opaque_model)

        class _NegativeConstant(_LargeConstant):
            def output(self, _index):
                return _FakePort("negative", (-1,), "bf16", node_type="Constant")

        negative_model = type("_NegativeModel", (), {
            "get_ordered_ops": lambda self: [_NegativeConstant()],
        })()
        with self.assertRaises(ValueError):
            MODULE.validate_constant_nodes(negative_model)

    def test_source_is_structural_only(self):
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
        for forbidden in ("compile_model", ".infer(", "safe_open", "safetensors", "Core("):
            self.assertNotIn(forbidden, source)

    def test_reference_root_and_fresh_output_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            for relative in MODULE.REQUIRED_REFERENCE_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# reference\n", encoding="utf-8")
            self.assertEqual(MODULE.checked_reference_root(root), root.resolve())
            output = Path(directory) / "out"
            self.assertTrue(MODULE.fresh_output(output).is_dir())
            with self.assertRaises(ValueError):
                MODULE.fresh_output(output)
            linked = Path(directory) / "linked"
            linked.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.checked_reference_root(linked)

    def test_cli_help_and_missing_root_do_not_import_openvino(self):
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        help_result = subprocess.run(
            [sys.executable, str(SOURCE), "--help"], check=False,
            capture_output=True, text=True, env=environment,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(SOURCE), "--reference-root", directory,
                 "--output", str(Path(directory) / "out")], check=False,
                capture_output=True, text=True, env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reference", result.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
