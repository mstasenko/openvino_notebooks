"""Offline structural checks for the extracted reference tree.

This test intentionally has no OpenVINO, Torch, safetensors, model, media, or
device dependency.  It catches accidental workstation paths and syntax drift
without claiming that a model run is qualified.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch


REFERENCE_ROOT = Path(__file__).resolve().parent
APP_ROOT = REFERENCE_ROOT.parent
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))


def _checked_app_path(root: Path, relative: str | Path) -> Path:
    """Resolve an app-relative path without permitting traversal or symlinks."""

    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise AssertionError(f"manifest path is not app-relative: {relative}")
    if root.is_symlink():
        raise AssertionError(f"app root must not be a symlink: {root}")
    root = root.resolve(strict=True)
    candidate = root / relative
    cursor = root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise AssertionError(f"app path contains a symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AssertionError(f"path escapes app root: {relative}") from error
    return resolved


def _app_path(relative: str | Path) -> Path:
    return _checked_app_path(APP_ROOT, relative)


class PortableClosureTest(unittest.TestCase):
    def test_resource_roots_and_explicit_overrides(self) -> None:
        # Loading only this stdlib module must never discover a device or
        # touch checkpoints. Root options must affect all derived defaults.
        spec = importlib.util.spec_from_file_location("isolated_paths", REFERENCE_ROOT / "reference_paths.py")
        with patch.dict(os.environ, {"FLASHVSR_PLUS_SOURCE": "/data/source",
                                     "FLASHVSR_PLUS_MODELS": "/data/models"}, clear=True):
            paths = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(paths)
            self.assertEqual(paths.PROMPT_PATH, Path("/data/source/prompt_tensor/posi_prompt.pth"))
            self.assertEqual(paths.DIT_CHECKPOINT, Path("/data/models/turbo/dit.safetensors"))
            self.assertEqual(paths.LQ_CHECKPOINT, Path("/data/models/turbo/lq_proj_in.safetensors"))
            self.assertEqual(paths.TCDECODER_CHECKPOINT, Path("/data/models/TCDecoder.ckpt"))
            with patch.dict(os.environ, {"FLASHVSR_PLUS_DIT": "/data/explicit.safetensors"}):
                spec.loader.exec_module(paths)
                self.assertEqual(paths.DIT_CHECKPOINT, Path("/data/explicit.safetensors"))

    def test_flash_post_generator_uses_public_asset_root(self) -> None:
        script = APP_ROOT / "scripts/prepare-flash-post.py"
        spec = importlib.util.spec_from_file_location("isolated_flash_post", script)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(script.parent), *sys.path]):
            assert spec.loader is not None
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(spec.name, None)
        self.assertEqual(module.ROOT, APP_ROOT)
        self.assertEqual(module.ASSET_ROOT, APP_ROOT / "assets/flash-assets")
        self.assertTrue(module.ASSET_ROOT.is_dir())

    def test_paths_are_external_and_provenance_is_well_formed(self) -> None:
        from reference_paths import ROOT as package_root

        self.assertEqual(package_root, REFERENCE_ROOT)
        provenance = json.loads((REFERENCE_ROOT / "provenance.json").read_text(encoding="utf-8"))
        self.assertFalse(provenance["extraction"]["weights_included"])
        self.assertFalse(provenance["extraction"]["media_included"])
        self.assertEqual(provenance["source"]["openvino_revision"], "7d81aad55a5")

    def test_no_private_workstation_defaults(self) -> None:
        # A guard's literal prefix (e.g. b"/cache/") is not a workstation
        # default. Reject concrete paths beneath those prefixes.
        forbidden = re.compile(r"(?<![A-Za-z0-9_/])/(?:home/sap|cache|root)/[A-Za-z0-9_.-]")
        for path in APP_ROOT.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md"}:
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(forbidden.search(text), str(path))

    def test_no_symlinks_or_path_escapes(self) -> None:
        for path in APP_ROOT.rglob("*"):
            self.assertFalse(path.is_symlink(), path)
        self.assertEqual(_app_path("."), APP_ROOT.resolve())

    def test_app_path_rejects_traversal_and_in_or_out_of_root_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "app"
            root.mkdir()
            inside = root / "inside.txt"
            inside.write_text("inside", encoding="utf-8")
            outside = Path(directory) / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            self.assertEqual(_checked_app_path(root, "inside.txt"), inside.resolve())
            with self.assertRaises(AssertionError):
                _checked_app_path(root, outside)
            with self.assertRaises(AssertionError):
                _checked_app_path(root, "../outside.txt")
            (root / "inside-link").symlink_to(inside)
            (root / "outside-link").symlink_to(outside)
            with self.assertRaises(AssertionError):
                _checked_app_path(root, "inside-link")
            with self.assertRaises(AssertionError):
                _checked_app_path(root, "outside-link")

    def test_python_sources_parse_without_importing_dependencies(self) -> None:
        sources = sorted(APP_ROOT.rglob("*.py"))
        self.assertGreater(len(sources), 40)
        for path in sources:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_expected_runtime_closure_is_present(self) -> None:
        expected = (
            "scripts/flashvsr_plus_openvino.py",
            "reference/openvino/full_runtime/run_plus_resident.py",
            "reference/openvino/full_runtime/plus_modes.py",
            "reference/openvino/full_runtime/plus_graphs.py",
            "reference/openvino/full_runtime/plus_memory.py",
            "reference/openvino/full_runtime/plus_storage.py",
            "reference/openvino/full_runtime/test_plus_text_export_cpu.py",
            "reference/openvino/full_runtime/test_plus_edges_export_cpu.py",
            "reference/openvino/chunked_attention/sparse_xmx.py",
            "reference/source/inference/decoders/tiny_model.py",
        )
        for relative in expected:
            self.assertTrue(_app_path(relative).is_file(), relative)

    def test_copied_source_fingerprints(self) -> None:
        manifest = json.loads((REFERENCE_ROOT / "provenance_sources.json").read_text())
        digest = hashlib.sha256()
        paths = set()
        for record in sorted(manifest["files"], key=lambda item: item["path"]):
            relative = Path(record["path"])
            self.assertFalse(relative.is_absolute())
            self.assertNotIn("..", relative.parts)
            self.assertNotIn(record["path"], paths)
            paths.add(record["path"])
            data = _app_path(relative).read_bytes()
            self.assertEqual(len(data), record["bytes"], record["path"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"],
                             record["path"])
            digest.update(f'{record["path"]}\0{record["sha256"]}\n'.encode())
        provenance = json.loads((REFERENCE_ROOT / "provenance.json").read_text())
        self.assertEqual(digest.hexdigest(), manifest["tree_sha256"])
        self.assertEqual(digest.hexdigest(),
                         provenance["extraction"]["source_manifest"]["tree_sha256"])

    def test_no_model_or_media_payloads_are_bundled(self) -> None:
        forbidden_suffixes = {".safetensors", ".ckpt", ".npz", ".npy", ".mp4", ".mkv"}
        payloads = [path for path in APP_ROOT.rglob("*") if path.is_file()
                    and path.suffix.lower() in forbidden_suffixes]
        self.assertEqual(payloads, [])

    def test_native_subset_is_structural_and_pinned(self) -> None:
        native = APP_ROOT / "native"
        assets = APP_ROOT / "assets/flash-assets"
        for relative in (
            "flash_post_session.cpp",
            "flash_post_session.hpp",
            "flash_model.cpp",
            "flash_model.hpp",
            "flash_model_tests.cpp",
            "flash_cross_session.cpp",
            "flash_cross_session.hpp",
            "flash_edges_model.cpp",
            "flash_edges_model.hpp",
            "flash_edges_model_tests.cpp",
            "flash_mask.cpp",
            "flash_mask.hpp",
            "flash_mask_tests.cpp",
            "flash_attention_session.cpp",
            "flash_attention_session.hpp",
            "flash_denoiser.cpp",
            "flash_denoiser.hpp",
            "flash_history.cpp",
            "flash_history.hpp",
            "flash_history_tests.cpp",
            "flash_token_session.cpp",
            "flash_token_session.hpp",
            "flash_tokens.cpp",
            "flash_tokens.hpp",
            "flash_tokens_tests.cpp",
            "flash_rope.cpp",
            "flash_rope.hpp",
            "flash_rope_tests.cpp",
            "flash_text_model.cpp",
            "flash_text_model.hpp",
            "flash_text_model_tests.cpp",
        ):
            self.assertTrue((native / relative).is_file(), relative)
        for relative in (
            "flash_post_4096_bf16.xml",
            "flash_post_4096_bf16.bin",
            "flash_post_assets.hpp",
            "flash_edges_prefix_bf16.xml",
            "flash_edges_prefix_bf16.bin",
            "flash_edges_suffix_bf16.xml",
            "flash_edges_suffix_bf16.bin",
            "flash_edges_provenance.json",
            "flash_attention_assets.hpp",
            "flash_attention_provenance.json",
            "plus_sparse_xmx.cl",
            "plus_split_key.xml",
            "flash_text_assets.hpp",
            "flash_text_bf16.xml",
            "flash_text_bf16.bin",
            "flash_text_provenance.txt",
            "flash_fixed_assets.hpp",
            "flash_fixed_provenance.json",
        ):
            self.assertTrue((assets / relative).is_file(), relative)
        stale_state = list(native.rglob("*_state.xml")) + list(native.rglob("*_state.bin"))
        self.assertEqual(stale_state, [])
        self.assertFalse((native / "flash_spool.cpp").exists())
        self.assertFalse((native / "flash_spool.hpp").exists())
        manifest = json.loads((REFERENCE_ROOT / "provenance_sources.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["file_count"], len(manifest["files"]))
        for record in manifest["files"]:
            self.assertEqual(record["sha256"], record["sanitized_sha256"],
                             record["path"])
            self.assertRegex(record["sanitized_sha256"], r"^[0-9a-f]{64}$")
            if record["origin_sha256"] is not None:
                self.assertRegex(record["origin_sha256"], r"^[0-9a-f]{64}$")
        native_records = [record for record in manifest["files"]
                          if record["path"].startswith("native/")]
        asset_records = [record for record in manifest["files"]
                         if record["path"].startswith("assets/flash-assets/")]
        script_records = [record for record in manifest["files"]
                          if record["path"].startswith("scripts/")]
        reference_records = [record for record in manifest["files"]
                             if record["path"].startswith("reference/")]
        self.assertEqual(len(native_records), 95)
        self.assertEqual(len(asset_records), 48)
        self.assertEqual(len(script_records), 21)
        self.assertEqual(len(reference_records), 153)
        self.assertTrue(all(record["origin"] in {"otc_native_wip",
                                                 "otc_build_support",
                                                 "extraction_build_support"}
                            for record in native_records + asset_records + script_records))
        for relative in (
            "scripts/prepare-flash-post.py",
            "scripts/test_bf16_convert.py",
            "scripts/export-flash-decoder.py",
            "scripts/test_export_flash_decoder.py",
            "scripts/export-flash-edges.py",
            "scripts/test_export_flash_edges.py",
            "scripts/export-flash-text.py",
            "scripts/test_export_flash_text.py",
            "scripts/test_prepare_flash_post.py",
            "scripts/export-flash-conditioning.py",
            "scripts/test_export_flash_conditioning.py",
        ):
            self.assertTrue(_app_path(relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
