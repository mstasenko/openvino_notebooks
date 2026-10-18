#!/usr/bin/env python3
"""Self-contained CPU contract tests for the authenticated decoder pin generator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/prepare-flash-decoder.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("prepare_flash_decoder_test_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(root: Path) -> tuple[Path, dict, dict[str, SimpleNamespace]]:
    """Build a tiny complete manifest without depending on cache artifacts."""

    static_names = [f"decoder.static_{index}" for index in range(64)]
    dynamic_shapes = {
        "decoder.15.conv.weight": (512, 256, 1, 1),
        "decoder.21.conv.weight": (256, 128, 1, 1),
    }
    state = {name: SimpleNamespace(shape=(1,)) for name in static_names}
    state.update({name: SimpleNamespace(shape=shape) for name, shape in dynamic_shapes.items()})
    static = []
    for name in static_names:
        static.append({
            "source_parameter_name": name,
            "source_shape": [1],
            "source_dtype": "f32",
            "storage": "bf16",
            "graph_input_name": "weight__" + name.replace(".", "_"),
            "graph_shape": [1, 1, 1, 1],
        })

    phase_groups = {
        "front": static_names[:41],
        "grow_memory": static_names[41:60],
        "remote_frame_child": static_names[60:],
    }
    graphs = {
        phase: {
            "contract": {
                "static_parameter_names": ["weight__" + name.replace(".", "_") for name in names],
                "dynamic_inputs": [] if phase == "front" else [MODULE.DYNAMIC[phase][0]],
                "inputs": ([] if phase == "front" else [{
                    "name": MODULE.DYNAMIC[phase][0],
                    "element_type": "bf16",
                    "shape": [MODULE.DYNAMIC[phase][2], MODULE.DYNAMIC[phase][2], 1, 1],
                }]),
            },
            "metadata": {"checkpoint_sha256": MODULE.CHECKPOINT_SHA256},
        }
        for phase, names in phase_groups.items()
    }
    banks = []
    for source_name, width in (
        ("decoder.15.conv.weight", 256),
        ("decoder.21.conv.weight", 128),
    ):
        shape = list(dynamic_shapes[source_name])
        banks.append({
            "source_parameter_name": source_name,
            "source_shape": shape,
            "source_dtype": "f32",
            "storage": "bf16",
            "slices": [
                {"child": child, "axis": 0, "start": child * width,
                 "stop": (child + 1) * width, "shape": [width, *shape[1:]],
                 "source_dtype": "f32", "storage": "bf16"}
                for child in range(2)
            ],
        })

    artifacts = {}
    for phase in MODULE.PHASES:
        records = {}
        for kind, suffix, payload in (
            ("xml", "xml", b"<xml/>"),
            ("bin", "bin", b"\x00"),
        ):
            filename = f"{phase}.{suffix}"
            (root / filename).write_bytes(payload)
            records[kind] = {"path": filename, "bytes": len(payload), "sha256": _sha256(payload)}
        artifacts[phase] = records

    manifest = {
        "schema": 1,
        "status": "pass",
        "bundle": "phased",
        "precision": "bf16",
        "geometry": {"source_hw": [2176, 3840], "tile_count": 16, "halo_rows": 12},
        "trust": {"learned_payloads_loaded": False},
        "decoder_source": {"sha256": MODULE.SOURCE_SHA256, "expected_sha256": MODULE.SOURCE_SHA256},
        "learned_storage": {
            "materialized_learned_payloads": False,
            "static_bindings": static,
            "source_bank_slice_count": 4,
            "source_bank_descriptors": banks,
        },
        "graphs": graphs,
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, manifest, state


class DecoderPreparationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest_path, self.manifest, self.state = _fixture(Path(self.temp.name))

    def test_authenticated_manifest_contract_accepts_self_contained_bundle(self):
        validated, artifacts = MODULE._validate_bundle(self.manifest, self.manifest_path, self.state)
        self.assertEqual({phase: len(names) for phase, names in validated["phases"].items()},
                         {"front": 41, "grow_memory": 19, "remote_frame_child": 4})
        self.assertEqual(len(validated["static"]), 64)
        self.assertEqual(len(validated["dynamic"]), 2)
        self.assertEqual(set(artifacts), set(MODULE.PHASES))

    def test_static_count_and_identity_are_fail_closed(self):
        malformed = copy.deepcopy(self.manifest)
        malformed["learned_storage"]["static_bindings"].pop()
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["learned_storage"]["static_bindings"][0]["source_parameter_name"] = "decoder.15.conv.weight"
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["learned_storage"]["static_bindings"][0]["graph_input_name"] = "weight__wrong"
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

    def test_dynamic_slice_and_nested_record_shapes_are_fail_closed(self):
        malformed = copy.deepcopy(self.manifest)
        malformed["learned_storage"]["source_bank_descriptors"][0]["slices"][1]["stop"] += 1
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["graphs"]["grow_memory"]["contract"]["dynamic_inputs"] = []
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["graphs"]["grow_memory"]["contract"]["inputs"][0]["name"] = "wrong_weight"
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["graphs"]["remote_frame_child"]["contract"]["inputs"][0]["shape"][0] = 256
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["graphs"]["remote_frame_child"]["contract"]["inputs"][0]["element_type"] = "f16"
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["graphs"]["remote_frame_child"]["contract"]["inputs"].append({
            "name": "grow_weight", "element_type": "bf16", "shape": [256, 256, 1, 1],
        })
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["learned_storage"]["source_bank_descriptors"][0]["source_parameter_name"] = "decoder.unknown"
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["trust"] = []
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

    def test_phase_coverage_and_artifact_fingerprints_are_fail_closed(self):
        malformed = copy.deepcopy(self.manifest)
        malformed["graphs"]["front"]["contract"]["static_parameter_names"].append("weight__decoder_static_1")
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["artifacts"]["front"]["xml"]["path"] = "nested/front.xml"
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["artifacts"]["front"]["bin"]["bytes"] = 1.0
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["artifacts"]["front"]["xml"]["bytes"] = 0
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["artifacts"]["front"]["bin"]["sha256"] = "bad-sha"
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["trust"].pop("learned_payloads_loaded")
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["learned_storage"].pop("materialized_learned_payloads")
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["artifacts"]["front"]["bin"]["path"] = "../manifest.json"
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

        malformed = copy.deepcopy(self.manifest)
        malformed["artifacts"]["front"]["bin"]["bytes"] += 1
        with self.assertRaises(ValueError):
            MODULE._validate_bundle(malformed, self.manifest_path, self.state)

    def test_symlink_and_child_traversal_are_rejected(self):
        root = Path(self.temp.name)
        target = root / "target.bin"
        target.write_bytes(b"x")
        link = root / "link.bin"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            MODULE._regular(link, "test file")
        with self.assertRaises(ValueError):
            MODULE._safe_child(root, "../target.bin", "asset")
        for value in ("nested/target.bin", r"nested\target.bin", "target:bin", "target\x00.bin"):
            with self.assertRaises(ValueError):
                MODULE._safe_child(root, value, "asset")

    def test_binding_and_segment_serialization_are_bounded(self):
        span = {"offset": 8, "bytes": 16, "elements": 8}
        self.assertEqual(MODULE._span_cpp(span), "{8ULL, 16ULL, 8ULL}")
        pin = MODULE._pin("TCDecoder.ckpt", 16, "00" * 32)
        self.assertIn('"TCDecoder.ckpt"', pin)
        self.assertIn("16ULL", pin)

    def test_child_span_header_uses_compilable_fixed_array_abi(self):
        digest = "00" * 32
        artifacts = {
            phase: {
                "xml": {"path": Path(f"{phase}.xml"), "bytes": 1, "sha256": digest},
                "bin": {"path": Path(f"{phase}.bin"), "bytes": 1, "sha256": digest},
            }
            for phase in MODULE.PHASES
        }
        phase_records = {
            phase: {"bindings": [], "child_spans": [
                {"offset": 0, "bytes": 2, "elements": 1},
                {"offset": 2, "bytes": 2, "elements": 1},
            ], "segments": [], "bin": {
                "name": f"{phase}.prepared.bin", "bytes": 0, "sha256": digest,
            }}
            for phase in MODULE.PHASES
        }
        rendered = MODULE._render_header(
            "TCDecoder.ckpt", 16, digest, digest, artifacts, phase_records
        )
        self.assertIn("kFlashDecoderGrowChildSpans[2] = {", rendered)
        self.assertIn("kFlashDecoderFrameChildSpans[2] = {", rendered)
        self.assertNotIn("std::array<SeedVrWeightSpan, 2>", rendered)
        self.assertNotIn("{{{{", rendered)

    def test_checkpoint_constants_are_not_changed_by_test_inputs(self):
        self.assertEqual(MODULE.CHECKPOINT_BYTES, 189018333)
        self.assertEqual(len(MODULE.CHECKPOINT_SHA256), 64)
        self.assertEqual(MODULE.PHASE_EXPECTED_COUNTS,
                         {"front": 41, "grow_memory": 19, "remote_frame_child": 4})

    def test_fresh_output_rejects_symlink_parent_and_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = MODULE.fresh_output(root / "new" / "decoder")
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)

            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.fresh_output(linked_parent / "created" / "decoder")
            self.assertFalse((real_parent / "created").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
