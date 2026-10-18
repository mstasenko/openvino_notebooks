"""Dependency-free contract tests for the structural edge exporter."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import flash_export
from test_export_flash_cross import _FakeNode, _FakePort


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "export-flash-edges.py"
SPEC = importlib.util.spec_from_file_location("export_flash_edges_tested", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeModel:
    def __init__(self, role, *, extra=False, alias=False, state=False,
                 f16_internal=False, bad_output=False, overlap=False):
        self._role = role
        geometry = type("Geometry", (), {
            "name": "plus_f4_3840x2176", "h": 136, "w": 240, "f": 4,
            "tokens": 130560,
        })()
        self._geometry_name = geometry.name
        inputs, outputs = MODULE._contracts(geometry, role=role)
        self.inputs = [
            _FakePort(
                name,
                shape,
                kind,
                names={name, "alias"} if alias and name == next(iter(inputs)) else None,
            )
            for name, (shape, kind) in inputs.items()
        ]
        if extra:
            self.inputs.append(_FakePort("unexpected", (1,), "bf16"))
        self.outputs = [
            _FakePort(
                name,
                shape,
                kind,
                node_type="ReadValue" if bad_output and name == next(iter(outputs)) else "Result",
            )
            for name, (shape, kind) in outputs.items()
        ]
        if overlap:
            self.outputs[0]._name = self.inputs[0].get_any_name()
            self.outputs[0]._names = {self.outputs[0]._name}
        self._state = state
        self._f16_internal = f16_internal

    def get_friendly_name(self):
        return f"flashvsr_plus_{self._geometry_name}_{self._role}_bf16"

    def get_variables(self):
        return [object()] if self._state else []

    def get_sinks(self):
        return [object()] if self._state else []

    def get_ordered_ops(self):
        if self._f16_internal:
            return [_FakeNode("Convert", "bad_f16", (
                _FakePort("bad_f16", (1,), "f16", node_type="Convert"),
            ))]
        return []


class ExportFlashEdgesCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = type("Geometry", (), {
            "name": "plus_f4_3840x2176", "h": 136, "w": 240, "f": 4,
            "tokens": 130560,
        })()
        cls.fake_ov = type("OV", (), {"Type": type("Type", (), {
            "bf16": "bf16", "f32": "f32",
        })})

    def test_fixed_geometry_and_contracts(self):
        self.assertEqual(MODULE.VISIBLE_HW, (2160, 3840))
        self.assertEqual(MODULE.WORKING_HW, (2176, 3840))
        self.assertEqual(MODULE.PRECISION, "bf16")
        for role in ("prefix", "suffix"):
            report = MODULE.validate_graph_contract(
                _FakeModel(role), self.geometry, role=role, ov=self.fake_ov
            )
            self.assertEqual(report["role"], role)
            self.assertEqual(report["constant_node_count"], 0)
            if role == "prefix":
                self.assertTrue(report["conditioning"]["one_time_preparation_required"])
            else:
                self.assertIsNone(report["conditioning"])
        suffix = MODULE.validate_graph_contract(
            _FakeModel("suffix"), self.geometry, role="suffix", ov=self.fake_ov
        )
        self.assertIsNone(suffix["conditioning"])

    def test_contract_rejects_alias_extra_state_overlap_bad_output_and_f16(self):
        for kwargs in (
            {"extra": True}, {"alias": True}, {"state": True},
            {"overlap": True}, {"bad_output": True}, {"f16_internal": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MODULE.validate_graph_contract(
                    _FakeModel("prefix", **kwargs), self.geometry,
                    role="prefix", ov=self.fake_ov,
                )

    def test_contract_rejects_unknown_role(self):
        with self.assertRaises(ValueError):
            MODULE._contracts(self.geometry, role="middle")

    def test_exporter_is_build_only_and_has_no_payload_reader(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported |= {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("numpy", imported)
        source = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("compile_model", ".infer(", "safe_open", "Core(", "safetensors"):
            self.assertNotIn(forbidden, source)

    def test_cli_help_does_not_need_openvino(self):
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, str(SOURCE), "--help"], check=False,
            capture_output=True, text=True, env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reference-root", result.stdout)
        self.assertIn("output", result.stdout)

    def test_shared_fresh_output_and_private_path_guards_remain_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reference"
            for relative in MODULE.REQUIRED_REFERENCE_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# pinned\n", encoding="utf-8")
            self.assertEqual(MODULE.checked_reference_root(root, MODULE.REQUIRED_REFERENCE_FILES),
                             root.resolve())
            output = Path(temporary) / "out"
            self.assertTrue(MODULE.fresh_output(output).is_dir())
            with self.assertRaises(ValueError):
                MODULE.fresh_output(output)
            with self.assertRaises(ValueError):
                flash_export.assert_no_private_paths(output / "x", b"/" + b"cache/private")


if __name__ == "__main__":
    unittest.main(verbosity=2)
