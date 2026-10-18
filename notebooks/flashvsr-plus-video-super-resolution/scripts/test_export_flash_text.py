"""Dependency-free ABI and shared-writer regressions for the text exporter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import flash_export
from test_export_flash_cross import _FakeModel, _FakeNode, _FakePort


SOURCE = Path(__file__).with_name("export-flash-text.py")
SPEC = importlib.util.spec_from_file_location("export_flash_text_tested", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Gelu(_FakeNode):
    def __init__(self, value, mode):
        super().__init__("Gelu", "gelu")
        self.value, self.mode = value, mode

    def get_attributes(self):
        return {"approximation_mode": self.mode}

    def input_value(self, index):
        return self.value


class _TextModel(_FakeModel):
    def __init__(self, *, mode="tanh", wrong_boundary=False, missing_boundary=False,
                 input_node_type="Parameter", **kwargs):
        super().__init__(input_node_type=input_node_type, **kwargs)
        self.inputs = [_FakePort(name, shape, kind, node_type=input_node_type)
                       for name, (shape, kind) in MODULE.INPUT_CONTRACT.items()]
        self.outputs = [_FakePort(name, shape, kind, node_type="Result")
                        for name, (shape, kind) in MODULE.OUTPUT_CONTRACT.items()
                        if not (missing_boundary and name == "text_linear")]
        self.linear = object()
        self.gelu = _Gelu(object() if wrong_boundary else self.linear, mode)
        for port in self.outputs:
            port.get_node().input_value = lambda index: self.linear

    def get_friendly_name(self):
        return "flashvsr_plus_text_bf16"

    def output(self, name):
        return next(port for port in self.outputs if port.get_any_name() == name)

    def get_ordered_ops(self):
        return [*super().get_ordered_ops(), self.gelu]


class ExportFlashTextCpuTests(unittest.TestCase):
    def test_fixed_contract_and_bf16_storage_boundary(self):
        self.assertEqual(MODULE.OPENVINO_VERSION, "2026.5.0-23457-7d81aad55a5")
        self.assertEqual(len(MODULE.TEXT_WEIGHTS), 4)
        report = MODULE.validate_graph_contract(_TextModel())
        self.assertEqual(report["learned_parameter_names"], sorted(MODULE.TEXT_WEIGHTS))
        self.assertEqual(report["storage_boundary"], "text_linear before GELU(tanh)")
        MODULE.validate_graph_contract(_TextModel(mode="TANH"))

    def test_rejects_lost_boundary_wrong_gelu_state_constants_and_f16(self):
        for options in ({"mode": "erf"}, {"wrong_boundary": True},
                        {"missing_boundary": True}, {"state": True},
                        {"learned_constant": True}, {"f16_internal": True},
                        {"input_node_type": "Result"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MODULE.validate_graph_contract(_TextModel(**options))

    def test_source_closure_must_be_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "openvino" / "builder.py"
            source.parent.mkdir()
            source.write_text("# fixture\n")
            with patch.object(flash_export, "loaded_source_paths", return_value=[source]):
                records = flash_export.graph_source_records(root, ["openvino/builder.py"])
                self.assertEqual(records[0]["path"], "openvino/builder.py")
                with self.assertRaises(ValueError):
                    flash_export.graph_source_records(root, ["openvino/missing.py"])

    def test_shared_writers_pin_relative_artifacts_and_reject_private_paths(self):
        model = _TextModel()

        class Reader:
            reloaded = model

            def load_by_framework(self, framework):
                if framework != "ir":
                    raise ValueError("expected IR frontend")
                return self

            def load(self, path):
                return self.reloaded

            def convert(self, model):
                return model

        class Serializer:
            frontend = SimpleNamespace(FrontEndManager=Reader)

            @staticmethod
            def serialize(model, xml, binary):
                Path(xml).write_text("<net/>\n")
                Path(binary).write_bytes(b"\x00\x01")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = flash_export.serialize_structural_ir(Serializer, model, output, "fixture")
            self.assertEqual(records["bin"]["path"], "fixture.bin")
            self.assertEqual(records["bin"]["bytes"], 2)
            for stem in ("../escape", "/escape", "file.xml", "", "UPPER"):
                with self.subTest(stem=stem), self.assertRaises(ValueError):
                    flash_export.serialize_structural_ir(Serializer, model, output, stem)
            changed = _TextModel()
            changed.outputs[-1] = _FakePort("text_linear_1", MODULE.TEXT_SHAPE, "bf16", node_type="Result")
            Reader.reloaded = changed
            with self.assertRaisesRegex(ValueError, "ABI changed during IR readback"):
                flash_export.serialize_structural_ir(Serializer, model, output, "renamed")
            manifest = flash_export.write_structural_manifest(output, {"artifacts": records})
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
            self.assertEqual(manifest["export_helper"], flash_export.helper_fingerprint())
            original = (output / "manifest.json").read_bytes()
            with self.assertRaises(ValueError):
                flash_export.write_structural_manifest(output, {"private": "/" + "cache/private"})
            self.assertEqual((output / "manifest.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
