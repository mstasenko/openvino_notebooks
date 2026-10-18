#!/usr/bin/env python3
"""CPU-only tests for the structural Flash PRE exporter."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "export-flash-structure.py"
SPEC = importlib.util.spec_from_file_location("export_flash_structure_tested", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeNode:
    def __init__(self, type_name, friendly_name):
        self._type_name = type_name
        self._friendly_name = friendly_name

    def get_type_name(self):
        return self._type_name

    def get_friendly_name(self):
        return self._friendly_name


class _FakePort:
    def __init__(self, name, shape, element_type, *, names=None, node_type="Result"):
        self._name = name
        self._names = {name} if names is None else set(names)
        self._shape = tuple(shape)
        self._element_type = element_type
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
    def __init__(self, *, state=False, learned_constant=False, extra_input=False,
                 alias=False, private_node_type="Result", private_type=None,
                 read_value=False, input_node_type="Parameter"):
        self.inputs = [
            _FakePort(name, shape, element_type,
                      names={name, "alias"} if alias and name == "x" else None,
                      node_type=input_node_type)
            for name, (shape, element_type) in MODULE.INPUT_CONTRACT.items()
        ]
        if extra_input:
            self.inputs.append(_FakePort("unexpected", (1,), "bf16"))
        self.outputs = []
        for index, (name, (shape, element_type)) in enumerate(MODULE.OUTPUT_CONTRACT.items()):
            self.outputs.append(_FakePort(
                name, shape, private_type if index == 0 and private_type else element_type,
                node_type=private_node_type if index == 0 else "Result",
            ))
        self._state = state
        self._learned_constant = learned_constant
        self._read_value = read_value

    def get_variables(self):
        return [object()] if self._state else []

    def get_sinks(self):
        return [object()] if self._state else []

    def get_ordered_ops(self):
        if self._read_value:
            return [_FakeNode("ReadValue", "state")]
        if self._learned_constant:
            return [_FakeNode("Constant", "weight__bad")]
        return []

    def get_friendly_name(self):
        return "fake_flash_pre"


class ExportFlashStructureCpuTests(unittest.TestCase):
    def test_contract_constants_and_scope(self):
        self.assertEqual(MODULE.ROWS, 4096)
        self.assertEqual(MODULE.VISIBLE_HW, (2160, 3840))
        self.assertEqual(MODULE.PRECISION, "bf16")
        self.assertEqual(MODULE.OPENVINO_VERSION, "2026.5.0-23457-7d81aad55a5")
        self.assertEqual(len(MODULE.INPUT_CONTRACT), 13)
        self.assertEqual(len(MODULE.OUTPUT_CONTRACT), 18)
        self.assertEqual(len(MODULE.PRIVATE_RESULTS), 6)

    def test_source_has_no_runtime_or_weight_execution(self):
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

    def test_fake_graph_contract_and_state_guards(self):
        fake_ov = type("FakeOpenVINO", (), {"Type": type("Type", (), {
            "bf16": "bf16", "f32": "f32",
        })})
        report = MODULE.validate_graph_contract(_FakeModel(), ov=fake_ov)
        self.assertEqual(report["private_bf16_results"], list(MODULE.PRIVATE_RESULTS))
        self.assertEqual(report["parameter_input_names"], sorted(
            name for name in MODULE.INPUT_CONTRACT if name.startswith("weight__")
        ))
        for bad in (
            _FakeModel(state=True), _FakeModel(learned_constant=True),
            _FakeModel(read_value=True), _FakeModel(extra_input=True),
            _FakeModel(alias=True), _FakeModel(private_node_type="ReadValue"),
            _FakeModel(private_type="f32"), _FakeModel(input_node_type="Result"),
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_graph_contract(bad, ov=fake_ov)

    def test_runtime_version_guard(self):
        MODULE._require_runtime_version(MODULE.OPENVINO_VERSION)
        with self.assertRaises(ValueError):
            MODULE._require_runtime_version("2026.3.0-other")

    def test_shared_helper_is_pinned_without_private_path(self):
        record = MODULE.helper_fingerprint()
        self.assertEqual(record["name"], "flash_export.py")
        self.assertEqual(len(record["sha256"]), 64)

    def test_reference_closure_and_output_are_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            root.mkdir()
            with self.assertRaises(ValueError):
                MODULE.checked_reference_root(root)
            output = Path(directory) / "out"
            created = MODULE.fresh_output(output)
            self.assertTrue(created.is_dir())
            with self.assertRaises(ValueError):
                MODULE.fresh_output(output)
            link = Path(directory) / "link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.checked_reference_root(link)

    def test_relative_source_record_and_private_path_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            source = root / "openvino" / "full_runtime" / "plus_graphs.py"
            source.parent.mkdir(parents=True)
            root.mkdir(exist_ok=True)
            source.write_text("structural", encoding="utf-8")
            record = MODULE._source_record(root, source)
            self.assertEqual(record["path"], "openvino/full_runtime/plus_graphs.py")
            self.assertEqual(record["bytes"], len("structural"))
            with self.assertRaises(ValueError):
                MODULE._assert_no_private_paths(source, b"source=/home/private/reference")

    def test_source_paths_exclude_runtime_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            curated = root / "openvino" / "full_runtime" / "curated.py"
            sibling = root / "runtime-openvino" / "openvino" / "runtime.py"
            curated.parent.mkdir(parents=True)
            sibling.parent.mkdir(parents=True)
            curated.write_text("curated", encoding="utf-8")
            sibling.write_text("runtime", encoding="utf-8")
            curated_module = types.ModuleType("_flash_curated_test")
            curated_module.__file__ = str(curated)
            sibling_module = types.ModuleType("_flash_runtime_sibling_test")
            sibling_module.__file__ = str(sibling)
            sys.modules[curated_module.__name__] = curated_module
            sys.modules[sibling_module.__name__] = sibling_module
            try:
                paths = MODULE._module_source_paths(root)
            finally:
                sys.modules.pop(curated_module.__name__, None)
                sys.modules.pop(sibling_module.__name__, None)
            self.assertIn(curated.resolve(), paths)
            self.assertNotIn(sibling.resolve(), paths)

    def test_reference_modules_are_exact_and_source_closure_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            foreign = Path(directory) / "foreign"
            for base in (root, foreign):
                for relative in MODULE.REQUIRED_REFERENCE_FILES:
                    path = base / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("# reference\n", encoding="utf-8")
            root_shadow = root / "plus_graphs.py"
            root_shadow.write_text("# shadow\n", encoding="utf-8")
            saved = {}
            try:
                for name, relative in MODULE.REFERENCE_MODULES.items():
                    saved[name] = sys.modules.get(name)
                    module = types.ModuleType(name)
                    module.__file__ = str(root / relative)
                    sys.modules[name] = module
                MODULE.require_reference_modules(root, MODULE.REFERENCE_MODULES)
                sys.modules["plus_graphs"].__file__ = str(foreign / MODULE.REFERENCE_MODULES["plus_graphs"])
                with self.assertRaises(ValueError):
                    MODULE.require_reference_modules(root, MODULE.REFERENCE_MODULES)
                sys.modules["plus_graphs"].__file__ = str(root_shadow)
                with self.assertRaises(ValueError):
                    MODULE.require_reference_modules(root, MODULE.REFERENCE_MODULES)
            finally:
                for name, previous in saved.items():
                    if previous is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = previous

            complete = [{"path": relative} for relative in MODULE.REQUIRED_REFERENCE_FILES]
            MODULE._require_source_closure(complete)
            with self.assertRaises(ValueError):
                MODULE._require_source_closure(complete[:-1])

    def test_cli_help_and_missing_reference_fail_without_openvino(self):
        clean_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        help_result = subprocess.run(
            [sys.executable, str(SOURCE), "--help"], check=False,
            capture_output=True, text=True, env=clean_env,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(SOURCE), "--reference-root", directory,
                 "--output", str(Path(directory) / "output")], check=False,
                capture_output=True, text=True, env=clean_env,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reference", result.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
