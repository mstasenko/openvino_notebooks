"""CPU-only tests for the fixed-conditioning asset writer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:  # Numeric packing tests are optional in stdlib CI.
    np = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "export_flash_conditioning_tested", SCRIPT_DIR / "export-flash-conditioning.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExportFlashConditioningCpuTests(unittest.TestCase):
    def test_exact_layout_and_raw_bf16_pack(self):
        if np is None:
            self.skipTest("numpy unavailable")
        t = np.zeros(MODULE.T_SHAPE, dtype=np.uint16)
        t_mod = np.full(MODULE.T_MOD_SHAPE, 0x3F80, dtype=np.uint16)
        payload = MODULE.pack_conditioning_words(t, t_mod)
        self.assertEqual(len(payload), MODULE.CONDITIONING_BYTES)
        self.assertEqual(payload[:MODULE.T_BYTES], b"\0" * MODULE.T_BYTES)
        self.assertEqual(
            payload[MODULE.T_BYTES:MODULE.T_BYTES + 2], b"\x80\x3f"
        )
        record = MODULE._mode_asset_record("turbo", payload)
        self.assertEqual(record["t"]["offset"], 0)
        self.assertEqual(record["t_mod"]["offset"], MODULE.T_BYTES)
        self.assertEqual(record["t_mod"]["shape"], [1, 6, 1536])
        self.assertTrue(record["derived"])

    def test_rejects_bad_shapes_dtype_layout_and_nonfinite_words(self):
        if np is None:
            self.skipTest("numpy unavailable")
        good_t = np.zeros(MODULE.T_SHAPE, dtype=np.uint16)
        good_mod = np.zeros(MODULE.T_MOD_SHAPE, dtype=np.uint16)
        cases = [
            (good_t.reshape(1536), good_mod),
            (good_t.astype(np.int16), good_mod),
            (good_t[:, :, ::2], good_mod),
            (good_t, good_mod.copy()),
        ]
        cases[-1][1].reshape(-1)[7] = np.uint16(0x7F80)
        for t, t_mod in cases:
            with self.subTest(t=t.shape, dtype=t.dtype, mod=t_mod.reshape(-1)[7]):
                with self.assertRaises(ValueError):
                    MODULE.pack_conditioning_words(t, t_mod)
        with self.assertRaises(ValueError):
            MODULE._mode_asset_record("invalid", b"x")

    def test_header_and_provenance_are_relative_and_payload_free(self):
        payload = b"\0" * MODULE.CONDITIONING_BYTES
        record = MODULE._mode_asset_record("turbo", payload)
        source = {
            "name": "dit.safetensors", "bytes": 10,
            "sha256": "11" * 32, "source_commit": "source",
        }
        header = MODULE._render_header({"turbo": record}, ("turbo",))
        self.assertIn("flash_fixed_conditioning_bf16_turbo.bin", header)
        self.assertNotIn("FlashFixedDitSource", header)
        self.assertNotIn("kFlashFixedDitSources", header)
        self.assertIn('"flash_prompt_bf16.bin"', header)
        self.assertIn("kFlashFixedTShape", header)
        self.assertNotIn("/home/", header)
        self.assertNotIn("/cache/", header)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            output.mkdir()
            artifacts = MODULE._write_outputs(
                output, {"turbo": record}, {"turbo": source}, ("turbo",),
                [], MODULE.OPENVINO_VERSION, "22" * 32,
            )
            self.assertEqual(artifacts["header"]["path"], "flash_fixed_assets.hpp")
            provenance = json.loads((output / "flash_fixed_provenance.json").read_text())
            self.assertTrue(provenance["trust"]["derived_arrays_are_not_source_spans"])
            self.assertFalse(provenance["openvino"]["core_created"])
            self.assertEqual(provenance["source_parity"]["embedding_bits_different"], 0)
            self.assertEqual(
                provenance["source_parity"]["receipt_sha256"],
                "e243a9d359f3e7f39d5f2fa212af682fce56b2034db9962d9d7a6d4059de81fd",
            )
            self.assertEqual(provenance["prompt"]["raw_span"]["name"], "flash_prompt_bf16.bin")
            self.assertEqual(
                provenance["prompt"]["raw_span"]["archive_member"], "posi_prompt/data/0"
            )

    def test_reference_mode_runtime_and_fresh_output_guards(self):
        with self.assertRaises(ValueError):
            MODULE.export_conditioning(Path("missing"), Path("missing"), Path("out"), ("bad",))
        with self.assertRaises(ValueError):
            MODULE.require_runtime_version("wrong", MODULE.OPENVINO_VERSION)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reference"
            outside = Path(temporary) / "outside.safetensors"
            (root / "turbo").mkdir(parents=True)
            outside.write_bytes(b"not the pinned checkpoint")
            (root / "turbo" / "dit.safetensors").symlink_to(outside)
            with self.assertRaises(ValueError):
                MODULE.checked_reference_root(root, ("turbo/dit.safetensors",))
            output = Path(temporary) / "existing"
            output.mkdir()
            with self.assertRaises(ValueError):
                MODULE.fresh_output(output)

    def _patched_export_modules(self, temporary, *, runtime_version,
                                load_edge_weights):
        root = Path(temporary) / "reference"
        model_root = Path(temporary) / "models"
        output = Path(temporary) / "output"
        dit = model_root / "turbo" / "dit.safetensors"
        dit.parent.mkdir(parents=True)
        dit.write_bytes(b"dit")
        fake_ov = SimpleNamespace(get_version=lambda: runtime_version)
        fake_edges = SimpleNamespace(
            load_edge_weights=load_edge_weights,
            fixed_conditioning=lambda *_args: (_ for _ in ()).throw(
                AssertionError("fixed_conditioning must not run")),
        )
        fake_modes = SimpleNamespace(
            MODEL_REPO_REVISION="rev",
            resolve_plus_mode=lambda _mode: SimpleNamespace(dit_bytes=3),
        )
        fake_storage = SimpleNamespace()
        patches = [
            patch.dict(sys.modules, {
                "openvino": fake_ov,
                "plus_edges": fake_edges,
                "plus_storage": fake_storage,
                "plus_modes": fake_modes,
            }, clear=False),
            patch.object(MODULE, "checked_reference_root", return_value=root),
            patch.object(MODULE, "checked_reference_file", return_value=dit),
            patch.object(MODULE, "require_reference_modules"),
            patch.object(MODULE, "graph_source_records", return_value=[]),
        ]
        return root, model_root, output, patches

    def test_checkpoint_rejected_before_derivation_or_payload_write(self):
        def reject(*_args, **_kwargs):
            raise ValueError("bad checkpoint hash")

        with tempfile.TemporaryDirectory() as temporary:
            root, model_root, output, patches = self._patched_export_modules(
                temporary, runtime_version=MODULE.OPENVINO_VERSION,
                load_edge_weights=reject,
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(ValueError, "bad checkpoint hash"):
                    MODULE.export_conditioning(root, model_root, output, ("turbo",))
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.glob("*.bin")), [])

    def test_runtime_rejected_before_derivation_or_payload_write(self):
        called = False

        def should_not_load(*_args, **_kwargs):
            nonlocal called
            called = True
            raise AssertionError("checkpoint load must not run")

        with tempfile.TemporaryDirectory() as temporary:
            root, model_root, output, patches = self._patched_export_modules(
                temporary, runtime_version="wrong", load_edge_weights=should_not_load,
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(ValueError, "pinned OpenVINO"):
                    MODULE.export_conditioning(root, model_root, output, ("turbo",))
            self.assertFalse(called)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.glob("*.bin")), [])


if __name__ == "__main__":
    unittest.main()
