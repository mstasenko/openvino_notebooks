#!/usr/bin/env python3
"""CPU-only tests for the native Flash POST table generator."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import stat
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SOURCE = HERE / "prepare-flash-post.py"
SPEC = importlib.util.spec_from_file_location("prepare_flash_post_tested", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareFlashPostCpuTests(unittest.TestCase):
    @staticmethod
    def _fake_stage_inputs(prefix=b"prefix"):
        plan = {
            "bin": {
                "name": "flash_stage_turbo_layer00.bin",
                "bytes": 8,
                "sha256": "1" * 64,
            },
            "segments": [{
                "offset": 0,
                "bytes": len(prefix),
                "source_offset": 0,
                "source": 2,
                "learned": False,
                "sha256": MODULE.ZERO_SHA,
                "encoding": "copy",
            }],
        }
        records = {
            "xml": {"name": "stage.xml", "bytes": 10, "sha256": "2" * 64},
            "bin": {"name": "stage.bin", "bytes": 11, "sha256": "3" * 64},
        }
        source = {
            "dit": {"name": "dit.safetensors", "bytes": 12, "sha256": "4" * 64,
                    "header_bytes": 8},
            "lq": {"name": "lq_proj_in.safetensors", "bytes": 13, "sha256": "5" * 64,
                   "header_bytes": 9},
            "prefix": prefix,
        }
        return records, {"turbo": source}, {"turbo": [plan] * MODULE.LAYERS}

    @staticmethod
    def _fake_lq_inputs():
        roles = {}
        for index, role in enumerate(MODULE.LQ_ROLE_ORDER):
            digest = str(index + 1) * 64
            roles[role] = {
                "xml": {"name": f"flash_lq_{role}.xml", "bytes": 10 + index,
                        "sha256": digest},
                "bin": {"name": f"flash_lq_{role}.bin", "bytes": 100 + index,
                        "sha256": digest},
            }
        mode_records = {
            "turbo": {
                "dit": {"name": "dit.safetensors", "bytes": 12,
                        "sha256": "4" * 64, "header_bytes": 8},
                "lq": {"name": "lq_proj_in.safetensors", "bytes": 13,
                       "sha256": "5" * 64, "header_bytes": 9},
                "roles": roles,
            },
        }
        mode_plans = {"turbo": {}}
        for role in MODULE.LQ_ROLE_ORDER:
            prefix = roles[role]["bin"]["bytes"]
            segments = [{
                "offset": 0, "bytes": prefix, "source_offset": 0,
                "source": 2, "learned": False, "sha256": MODULE.ZERO_SHA,
                "encoding": "copy",
            }]
            offset = prefix
            for key, shape in MODULE.LQ_ROLE_SCHEMAS[role]:
                bytes_ = MODULE._checked_product(shape) * 2
                segments.append({
                    "offset": offset, "bytes": bytes_, "source_offset": 16,
                    "source": 1, "learned": True, "sha256": "6" * 64,
                    "encoding": "f32_to_bf16_rne",
                })
                offset += bytes_
            mode_plans["turbo"][role] = {
                "bin": {"name": f"out-{role}.bin", "bytes": offset,
                        "sha256": "7" * 64},
                "complement": roles[role]["bin"],
                "segments": segments,
            }
        return roles, mode_records, mode_plans

    def test_source_has_no_research_runtime_or_device_import(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_from = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertFalse({"torch", "openvino", "dist"} & (imports | imported_from))
        source = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("compile_model", "available_devices", "Core", "safetensors"):
            if forbidden == "safetensors":
                # Header parsing is intentionally implemented for this file
                # format; importing the safetensors package is forbidden.
                self.assertNotIn("from safetensors", source)
            else:
                self.assertNotIn(forbidden, source)

    def test_fixed_contract_is_portable_graph_and_nine_call_inputs_are_documented(self):
        self.assertEqual(len(MODULE.POST_SCHEMA), 13)
        self.assertEqual(MODULE.ASSET_PINS["xml"][0], "flash_post_4096_bf16.xml")
        self.assertEqual(MODULE.ASSET_PINS["bin"][0], "flash_post_4096_bf16.bin")
        self.assertIn("other nine Parameters", SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(set(MODULE.MODE_PINS), {"turbo", "base"})

    def test_checked_in_asset_root_is_authenticated_and_not_a_cli_override(self):
        # Both OTC and the standalone extraction resolve their own checked-in
        # assets. Authenticate those files instead of hardcoding either layout.
        for pins in (MODULE.ASSET_PINS, MODULE.PRE_ASSET_PINS,
                     MODULE.CROSS_ASSET_PINS, MODULE.TEXT_ASSET_PINS,
                     MODULE.EDGE_PREFIX_ASSET_PINS, MODULE.EDGE_SUFFIX_ASSET_PINS):
            self.assertEqual(set(MODULE._asset_records_for(pins)), set(pins))
        self.assertEqual(tuple(MODULE._lq_asset_records()), MODULE.LQ_ROLE_ORDER)
        source = SOURCE.read_text(encoding="utf-8")
        for override in ('"--asset-root"', "raw_asset_root", "global ASSET_ROOT"):
            self.assertNotIn(override, source)

    def test_pre_contract_pins_schema_and_source_mapping(self):
        self.assertEqual(len(MODULE.PRE_SCHEMA), 9)
        self.assertEqual(
            MODULE.PRE_ASSET_PINS["xml"],
            (
                "flash_pre_4096_bf16.xml", 87_093,
                "a70c2bbb5fb0f65add9e93da461b9dee324a2fb5216c649ccf69edf84876087d",
            ),
        )
        self.assertEqual(
            MODULE.PRE_ASSET_PINS["bin"],
            (
                "flash_pre_4096_bf16.bin", 768,
                "b4ce6a73d4445949f43264f8334d57fcedf5b29504bc2222c110be3e7c60646e",
            ),
        )
        prefix = b"prefix"
        plan = {
            "bin": {
                "name": "flash_pre_4096_bf16_turbo_layer00.bin",
                "bytes": 8, "sha256": "1" * 64,
            },
            "segments": [{
                "offset": 0, "bytes": len(prefix), "source_offset": 0,
                "source": 2, "learned": False, "sha256": MODULE.ZERO_SHA,
                "encoding": "copy",
            }],
        }
        fake = {
            "dit": {"name": "dit.safetensors", "bytes": 10, "sha256": "2" * 64,
                    "header_bytes": 8},
            "lq": {"name": "lq_proj_in.safetensors", "bytes": 11, "sha256": "3" * 64,
                   "header_bytes": 9},
            "prefix": prefix,
        }
        text = MODULE._render_pre_header(
            {"turbo": fake}, {"turbo": [plan] * MODULE.LAYERS},
        )
        self.assertIn("FlashPreWeightBinding", text)
        self.assertIn("std::array<FlashPreWeightBinding, 9>", text)
        self.assertIn("kFlashPreXml", text)
        self.assertIn("kFlashPreStructuralBin", text)
        self.assertIn("kFlashPreTurboBins", text)
        self.assertIn("kFlashPreTurboPlans", text)
        self.assertIn("kFlashPreTurboSources", text)
        self.assertIn("kFlashPreTurboHeaderBytes", text)
        self.assertNotIn("kFlashPost", text)
        for canonical, _shape in MODULE.PRE_SCHEMA:
            self.assertIn(canonical, text)
            self.assertIn("weight__" + canonical.replace(".", "_"), text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("/cache/", text)

    def test_cross_contract_pins_schema_and_ordinary_result_header(self):
        self.assertEqual(len(MODULE.CROSS_SCHEMA), 5)
        self.assertEqual(
            MODULE.CROSS_SCHEMA,
            (
                ("blocks.0.cross_attn.k.weight", (1536, 1536)),
                ("blocks.0.cross_attn.k.bias", (1536,)),
                ("blocks.0.cross_attn.norm_k.weight", (1536,)),
                ("blocks.0.cross_attn.v.weight", (1536, 1536)),
                ("blocks.0.cross_attn.v.bias", (1536,)),
            ),
        )
        self.assertEqual(
            MODULE.CROSS_ASSET_PINS,
            {
                "xml": (
                    "flash_cross_kv_bf16.xml", 9_493,
                    "e57aee54756cd34d775371b8137358fff7ddfb4680ee680dd1c93dc1f299a82d",
                ),
                "bin": (
                    "flash_cross_kv_bf16.bin", 12,
                    "862ca08291ff0bf5f96f630f2f8e3a42384ddc458dde33057ed473fe26612aa5",
                ),
            },
        )
        _records, mode_records, mode_plans = self._fake_stage_inputs()
        text = MODULE._render_stage_header(mode_records, mode_plans, "CROSS")
        self.assertIn("FlashCrossWeightBinding", text)
        self.assertIn("std::array<FlashCrossWeightBinding, 5>", text)
        self.assertIn("kFlashCrossRows = 512U", text)
        self.assertIn("kFlashCrossLayers = 30U", text)
        self.assertIn("kFlashCrossTurboSources[2]", text)
        self.assertIn("kFlashCrossTurboHeaderBytes[2]", text)
        self.assertIn("lq_proj_in.safetensors", text)
        self.assertIn('kFlashCrossGraphVariant = "result_ir_ordinary_results"', text)
        self.assertIn("kFlashCrossTurboPlans", text)
        self.assertNotIn("kFlashCrossConfig", text)
        self.assertNotIn("kFlashCrossSparseSource", text)
        for canonical, _shape in MODULE.CROSS_SCHEMA:
            self.assertIn(canonical, text)
            self.assertIn("weight__" + canonical.replace(".", "_"), text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("/cache/", text)

    def test_cross_provenance_records_common_pair_and_source_zero_usage(self):
        records = {
            "xml": {"name": "flash_cross_kv_bf16.xml", "bytes": 9493,
                    "sha256": "a" * 64},
            "bin": {"name": "flash_cross_kv_bf16.bin", "bytes": 12,
                    "sha256": "b" * 64},
        }
        plan = {"bin": {"name": "cross.bin", "bytes": 100,
                         "sha256": "c" * 64}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cross.txt"
            MODULE._write_stage_provenance(
                path, ("turbo",), records, {"turbo": [plan] * MODULE.LAYERS},
                Path(directory) / "weights", Path(directory) / "converter", "CROSS",
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("Flash CROSS K/V native build-time pins", text)
        self.assertIn("graph_variant=result_ir_ordinary_results", text)
        self.assertIn("rows=512", text)
        self.assertIn("learned_schema=5", text)
        self.assertIn("authenticated_sources=DIT+LQ; cross learned spans use source 0", text)

    def test_text_contract_has_four_global_weights_and_empty_structural_bin(self):
        self.assertEqual(
            MODULE.TEXT_SCHEMA,
            (
                ("text_embedding.0.weight", (1536, 4096)),
                ("text_embedding.0.bias", (1536,)),
                ("text_embedding.2.weight", (1536, 1536)),
                ("text_embedding.2.bias", (1536,)),
            ),
        )
        self.assertEqual(MODULE.TEXT_ASSET_PINS["bin"][1], 0)
        self.assertEqual(
            MODULE.TEXT_ASSET_PINS["bin"][2],
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        records, mode_records, mode_plans = self._fake_stage_inputs()
        text = MODULE._render_stage_header(
            mode_records, {"turbo": [mode_plans["turbo"][0]]}, "TEXT"
        )
        self.assertIn("FlashTextWeightBinding", text)
        self.assertIn("std::array<FlashTextWeightBinding, 4>", text)
        self.assertIn("kFlashTextRows = 512U", text)
        self.assertIn("kFlashTextLayers = 1U", text)
        self.assertIn('kFlashTextGraphVariant = "result_ir_ordinary_results"', text)
        self.assertIn("text_embedding.0.weight", text)
        self.assertIn("text_embedding.2.bias", text)
        self.assertIn("kFlashTextTurboPlans", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("/cache/", text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.txt"
            MODULE._write_stage_provenance(
                path, ("turbo",), records,
                {"turbo": [mode_plans["turbo"][0]]},
                Path(directory) / "weights", Path(directory) / "converter", "TEXT",
            )
            provenance = path.read_text(encoding="utf-8")
        self.assertIn("Flash TEXT native build-time pins", provenance)
        self.assertIn("learned_schema=4 global text-embedding tensors", provenance)
        self.assertIn("fixed_bin=", provenance)

    def test_edge_contracts_pins_global_source_mapping_and_one_plan_headers(self):
        self.assertEqual(
            MODULE.EDGE_PREFIX_SCHEMA,
            (
                ("patch_embedding.weight", (1536, 16, 1, 2, 2)),
                ("patch_embedding.bias", (1536,)),
            ),
        )
        self.assertEqual(
            MODULE.EDGE_SUFFIX_SCHEMA,
            (
                ("head.modulation", (1, 2, 1536)),
                ("head.proj.weight", (64, 1536)),
                ("head.proj.bias", (64,)),
            ),
        )
        self.assertEqual(MODULE.EDGE_PREFIX_ASSET_PINS["bin"][1], 664)
        self.assertEqual(MODULE.EDGE_SUFFIX_ASSET_PINS["bin"][1], 328)
        records, mode_records, mode_plans = self._fake_stage_inputs()
        edge_plans = {"turbo": [mode_plans["turbo"][0]]}
        prefix_text = MODULE._render_stage_header(
            mode_records, edge_plans, "EDGES_PREFIX"
        )
        suffix_text = MODULE._render_stage_header(
            mode_records, edge_plans, "EDGES_SUFFIX"
        )
        for text, family, count in (
            (prefix_text, "FlashEdgesPrefix", 2),
            (suffix_text, "FlashEdgesSuffix", 3),
        ):
            self.assertIn(f"{family}WeightBinding", text)
            self.assertIn("std::array<std::size_t, 5>", text)
            self.assertIn(f"k{family}Layers = 1U", text)
            self.assertIn(f"std::array<SeedVrAssetPin, 1> k{family}TurboBins", text)
            self.assertIn(f"std::array<prepare_detail::Plan, 1> k{family}TurboPlans", text)
            self.assertIn(f"std::array<{family}WeightBinding, {count}>", text)
            self.assertIn("lq_proj_in.safetensors", text)
            self.assertNotIn("/home/", text)
            self.assertNotIn("/cache/", text)

        source_records = {
            key: {"shape": shape, "bytes": MODULE._checked_product(shape) * 4,
                  "source_offset": 0}
            for key, shape in MODULE.EDGE_PREFIX_SCHEMA
        }
        mapped = MODULE._source_layer(
            Path("unused"), 7, source_records,
            schema=MODULE.EDGE_PREFIX_SCHEMA, stage="EDGES_PREFIX",
        )
        self.assertEqual(
            [item["source_key"] for item in mapped],
            [key for key, _shape in MODULE.EDGE_PREFIX_SCHEMA],
        )

    def test_edge_provenance_and_dispatch_are_explicit_and_fail_closed(self):
        records, mode_records, mode_plans = self._fake_stage_inputs()
        one_plan = {"turbo": [mode_plans["turbo"][0]]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.txt"
            MODULE._write_stage_provenance(
                path, ("turbo",), records, one_plan,
                Path(directory) / "weights", Path(directory) / "converter",
                "EDGES_PREFIX",
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("Flash EDGES PREFIX native build-time pins", text)
        self.assertIn("edge_role=prefix", text)
        self.assertIn("learned_schema=2 global edge tensors", text)
        self.assertIn("authenticated_sources=DIT+LQ; edge learned spans use source 0", text)
        self.assertIn("conditioning=caller prepares fixed t/t_mod", text)
        self.assertIn("layer_00_bin=", text)
        self.assertNotIn("layer_01_bin=", text)
        with self.assertRaises(ValueError):
            MODULE._render_stage_header(mode_records, mode_plans, "EDGES_PREFIX")

    def test_stage_defaults_to_post_and_pre_uses_shared_dispatch(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"edges-prefix"', source)
        self.assertIn('"edges-suffix"', source)
        self.assertIn('"text"', source)
        self.assertIn("return _build_post(args)", source)
        self.assertIn("return _build_pre(args)", source)
        self.assertIn("return _build_cross(args)", source)
        self.assertIn("return _build_edges_prefix(args)", source)
        self.assertIn("return _build_edges_suffix(args)", source)
        self.assertIn("return _build_text(args)", source)
        self.assertIn("return _build_lq(args)", source)

    def test_lq_roles_have_exact_key_sets_and_rank_five_binding_contract(self):
        self.assertEqual(MODULE.LQ_ROLE_ORDER, ("startup", "prefix", "suffix"))
        self.assertEqual(tuple(len(MODULE.LQ_ROLE_SCHEMAS[role]) for role in MODULE.LQ_ROLE_ORDER),
                         (3, 3, 5))
        self.assertEqual(MODULE.LQ_SCHEMA_KEYS, {
            "conv1.bias", "conv1.weight", "norm1.weight", "conv2.bias",
            "conv2.weight", "linear_layers.0.bias", "linear_layers.0.weight",
            "norm2.weight",
        })
        roles, mode_records, mode_plans = self._fake_lq_inputs()
        text = MODULE._render_lq_header(mode_records, mode_plans)
        self.assertIn("std::array<std::size_t, 5>", text)
        self.assertIn("std::array<FlashLqWeightBinding, 3> kFlashLqStartupWeights", text)
        self.assertIn("std::array<FlashLqWeightBinding, 3> kFlashLqPrefixWeights", text)
        self.assertIn("std::array<FlashLqWeightBinding, 5> kFlashLqSuffixWeights", text)
        self.assertIn("kFlashLqTurboSources[2]", text)
        self.assertIn("kFlashLqTurboPlans", text)
        self.assertIn(", 1u, true,", text)
        self.assertNotIn("enum class FlashLqRole", text)
        self.assertNotIn("kFlashPost", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("/cache/", text)
        mismatched_roles = {
            **roles,
            "startup": {
                **roles["startup"],
                "xml": {**roles["startup"]["xml"], "name": "other-startup.xml"},
            },
        }
        mismatched_records = {
            "turbo": mode_records["turbo"],
            "base": {**mode_records["turbo"], "roles": mismatched_roles},
        }
        with self.assertRaises(ValueError):
            MODULE._render_lq_header(
                mismatched_records,
                {"turbo": mode_plans["turbo"], "base": mode_plans["turbo"]},
            )

    def test_lq_role_byte_counts_and_rank_limits_are_exact(self):
        role_bytes = {
            role: sum(MODULE._checked_product(shape) * 2
                      for _key, shape in schema)
            for role, schema in MODULE.LQ_ROLE_SCHEMAS.items()
        }
        self.assertEqual(role_bytes["startup"], 113_254_400)
        self.assertEqual(role_bytes["prefix"], 113_254_400)
        self.assertEqual(role_bytes["suffix"], 462_437_376)
        self.assertEqual(
            sum(MODULE._checked_product(shape) * 2
                for _key, shape in MODULE.LQ_ROLE_SCHEMAS["startup"]),
            MODULE._checked_product((2048, 768, 4, 3, 3)) * 2
            + 2 * 2048 * 2,
        )
        with self.assertRaises(ValueError):
            MODULE._checked_product((-1, 2))

    def test_lq_source_schema_rejects_missing_extra_and_wrong_shape(self):
        tensors = {
            key: {"shape": next(shape for schema in MODULE.LQ_ROLE_SCHEMAS.values()
                                 for candidate, shape in schema if candidate == key),
                  "bytes": next(MODULE._checked_product(shape) * 4
                                 for schema in MODULE.LQ_ROLE_SCHEMAS.values()
                                 for candidate, shape in schema if candidate == key),
                  "source_offset": 16}
            for key in MODULE.LQ_SCHEMA_KEYS
        }
        self.assertEqual(len(MODULE._lq_source_items(tensors, "startup")), 3)
        missing = dict(tensors)
        missing.pop("conv1.bias")
        with self.assertRaises(ValueError):
            MODULE._lq_source_items(missing, "startup")
        extra = dict(tensors)
        extra["unexpected"] = {"shape": (1,), "bytes": 4, "source_offset": 16}
        with self.assertRaises(ValueError):
            MODULE._lq_source_items(extra, "suffix")
        wrong = dict(tensors)
        wrong["conv2.weight"] = {**wrong["conv2.weight"], "shape": (1,)}
        with self.assertRaises(ValueError):
            MODULE._lq_source_items(wrong, "suffix")
        wrong_span = dict(tensors)
        wrong_span["conv1.bias"] = {**wrong_span["conv1.bias"], "bytes": 2}
        with self.assertRaises(ValueError):
            MODULE._lq_source_items(wrong_span, "startup")
        negative_span = dict(tensors)
        negative_span["conv1.bias"] = {**negative_span["conv1.bias"], "source_offset": -4}
        with self.assertRaises(ValueError):
            MODULE._lq_source_items(negative_span, "startup")

    def test_plan_can_encode_lq_learned_segments_as_second_authenticated_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(bytes(range(8)))
            converter = root / "converter"
            converter.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "raw=sys.stdin.buffer.read()\n"
                "sys.stdout.buffer.write(raw[::2])\n",
                encoding="utf-8",
            )
            converter.chmod(0o700)
            segments, total, digest = MODULE._plan_from_items(
                source,
                [{"source_offset": 0, "bytes": 8}],
                b"AB",
                converter,
                learned_source=1,
            )
            self.assertEqual(total, 6)
            self.assertEqual(segments[0]["source"], 2)
            self.assertEqual(segments[1]["source"], 1)
            self.assertTrue(segments[1]["learned"])
            expected = hashlib.sha256(b"AB" + bytes([0, 2, 4, 6])).hexdigest()
            self.assertEqual(digest, expected)
            self.assertEqual(
                segments[1]["sha256"], hashlib.sha256(bytes([0, 2, 4, 6])).hexdigest()
            )
            empty_segments, empty_total, empty_digest = MODULE._plan_from_items(
                source, [{"source_offset": 0, "bytes": 8}], b"", converter,
            )
            self.assertEqual(empty_total, 4)
            self.assertEqual(len(empty_segments), 1)
            self.assertEqual(empty_segments[0]["offset"], 0)
            self.assertTrue(empty_segments[0]["learned"])
            self.assertEqual(empty_digest, hashlib.sha256(bytes([0, 2, 4, 6])).hexdigest())
            with self.assertRaises(ValueError):
                MODULE._plan_from_items(
                    source, [{"source_offset": 0, "bytes": 8}], b"AB", converter,
                    learned_source=2,
                )
            with self.assertRaises(ValueError):
                MODULE._plan_from_items(
                    source, [{"source_offset": -1, "bytes": 8}], b"AB", converter,
                )
            with self.assertRaises(ValueError):
                MODULE._plan_from_items(
                    source, [{"source_offset": 0, "bytes": 3}], b"AB", converter,
                )

    def test_lq_structural_pins_are_installed_but_incomplete_mapping_fails_closed(self):
        self.assertEqual(tuple(MODULE.LQ_ASSET_PINS), MODULE.LQ_ROLE_ORDER)
        fake_records = {
            "xml": {"name": "x.xml", "bytes": 1, "sha256": "0" * 64},
            "bin": {"name": "x.bin", "bytes": 1, "sha256": "1" * 64},
        }
        with patch.object(MODULE, "_asset_records_for", return_value=fake_records) as mocked:
            records = MODULE._lq_asset_records()
        self.assertEqual(tuple(records), MODULE.LQ_ROLE_ORDER)
        self.assertEqual(mocked.call_count, len(MODULE.LQ_ROLE_ORDER))
        with patch.object(MODULE, "LQ_ASSET_PINS", {}):
            with self.assertRaises(ValueError):
                MODULE._lq_asset_records()
        incomplete = dict(MODULE.LQ_ASSET_PINS)
        incomplete.pop("suffix")
        with patch.object(MODULE, "LQ_ASSET_PINS", incomplete):
            with self.assertRaises(ValueError):
                MODULE._lq_asset_records()

    def test_lq_build_writes_distinct_role_tables_and_provenance(self):
        roles, mode_records, mode_plans = self._fake_lq_inputs()
        collected = (("turbo",), Path("converter"), roles, mode_records,
                     mode_plans, Path("weights"))
        args = SimpleNamespace(output=None)
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "lq"
            with patch.object(MODULE, "_collect_lq", return_value=collected):
                output = MODULE._build_lq(args)
            self.assertTrue((output / "flash_lq_assets.hpp").is_file())
            provenance = (output / "flash_lq_provenance.txt").read_text(encoding="utf-8")
            self.assertIn("startup_bin=out-startup.bin", provenance)
            self.assertIn("prefix_bin=out-prefix.bin", provenance)
            self.assertIn("suffix_bin=out-suffix.bin", provenance)
            self.assertIn(
                f"weights_root={MODULE.EXTERNAL_WEIGHTS_LABEL}", provenance
            )
            self.assertNotIn(str(Path(directory)), provenance)

    def test_shared_stage_renderer_and_provenance_preserve_wrappers(self):
        records, mode_records, mode_plans = self._fake_stage_inputs()
        post_wrapper = MODULE._render_header(mode_records, mode_plans)
        pre_wrapper = MODULE._render_pre_header(mode_records, mode_plans)
        self.assertEqual(post_wrapper, MODULE._render_stage_header(mode_records, mode_plans, "POST"))
        self.assertEqual(pre_wrapper, MODULE._render_stage_header(mode_records, mode_plans, "PRE"))
        with self.assertRaises(ValueError):
            MODULE._render_stage_header(mode_records, mode_plans, "other")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            converter = root / "converter"
            weights_root = root / "weights"
            weights_root.mkdir()
            for stage, wrapper, writer in (
                ("POST", MODULE._write_provenance, MODULE._write_stage_provenance),
                ("PRE", MODULE._write_pre_provenance, MODULE._write_stage_provenance),
            ):
                wrapped_path = root / f"{stage.lower()}-wrapped.txt"
                shared_path = root / f"{stage.lower()}-shared.txt"
                wrapper(wrapped_path, ("turbo",), records, mode_plans, weights_root, converter)
                writer(shared_path, ("turbo",), records, mode_plans, weights_root, converter, stage)
                self.assertEqual(wrapped_path.read_bytes(), shared_path.read_bytes())
                text = wrapped_path.read_text(encoding="utf-8")
                self.assertIn(
                    f"weights_root={MODULE.EXTERNAL_WEIGHTS_LABEL}", text
                )
                self.assertNotIn(str(weights_root), text)

    def test_shared_build_path_keeps_stage_artifact_names(self):
        records, mode_records, mode_plans = self._fake_stage_inputs()
        collected = (("turbo",), Path("converter"), records, mode_records, mode_plans, Path("weights"))
        args = SimpleNamespace(output=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args.output = root / "post"
            with patch.object(MODULE, "_collect_stage", return_value=collected):
                post = MODULE._build_post(args)
            self.assertEqual(post, root / "post")
            self.assertTrue((post / "flash_post_assets.hpp").is_file())
            self.assertTrue((post / "flash_post_provenance.txt").is_file())

            args.output = root / "pre"
            with patch.object(MODULE, "_collect_stage", return_value=collected):
                pre = MODULE._build_pre(args)
            self.assertEqual(pre, root / "pre")
            self.assertTrue((pre / "flash_pre_assets.hpp").is_file())
            self.assertTrue((pre / "flash_pre_provenance.txt").is_file())

            args.output = root / "cross"
            with patch.object(MODULE, "_collect_stage", return_value=collected):
                cross = MODULE._build_cross(args)
            self.assertEqual(cross, root / "cross")
            self.assertTrue((cross / "flash_cross_assets.hpp").is_file())
            self.assertTrue((cross / "flash_cross_provenance.txt").is_file())

            edge_collected = (
                ("turbo",), Path("converter"), records, mode_records,
                {"turbo": [mode_plans["turbo"][0]]}, Path("weights"),
            )
            args.output = root / "edges-prefix"
            with patch.object(MODULE, "_collect_stage", return_value=edge_collected):
                prefix = MODULE._build_edges_prefix(args)
            self.assertTrue((prefix / "flash_edges_prefix_assets.hpp").is_file())
            self.assertTrue((prefix / "flash_edges_prefix_provenance.txt").is_file())

            args.output = root / "edges-suffix"
            with patch.object(MODULE, "_collect_stage", return_value=edge_collected):
                suffix = MODULE._build_edges_suffix(args)
            self.assertTrue((suffix / "flash_edges_suffix_assets.hpp").is_file())
            self.assertTrue((suffix / "flash_edges_suffix_provenance.txt").is_file())

    def test_checked_product_and_cpp_digest_guards(self):
        self.assertEqual(MODULE._checked_product((1536, 8960)), 1536 * 8960)
        with self.assertRaises(ValueError):
            MODULE._checked_product((True, 2))
        with self.assertRaises(ValueError):
            MODULE._bytes_literal("z" * 64)
        with self.assertRaises(ValueError):
            MODULE._bytes_literal("0" * 63)

    def test_parse_header_checks_offsets_and_shape_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            payload = struct.pack("<ff", 1.0, -0.0)
            header = b'{"tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            header_bytes, tensors = MODULE._parse_header(path)
            self.assertEqual(header_bytes, len(header))
            self.assertEqual(tensors["tensor"]["source_offset"], 8 + len(header))
            self.assertEqual(tensors["tensor"]["bytes"], 8)

            bad = Path(directory) / "bad.safetensors"
            bad_header = b'{"tensor":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}'
            bad.write_bytes(struct.pack("<Q", len(bad_header)) + bad_header + payload)
            with self.assertRaises(ValueError):
                MODULE._parse_header(bad)

    def test_authentication_rejects_escape_symlink_and_wrong_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            source = root / "weights.safetensors"
            source.write_bytes(b"fixed")
            digest = MODULE.sha256_file(source)
            record = MODULE.authenticate_file(
                source, expected_bytes=5, expected_sha256=digest,
                root=root, label="fixture",
            )
            self.assertEqual(record["sha256"], digest)
            outside = root.parent / (root.name + "-outside")
            outside.write_bytes(b"fixed")
            with self.assertRaises(ValueError):
                MODULE.authenticate_file(
                    outside, expected_bytes=5, expected_sha256=digest,
                    root=root, label="escape",
                )
            link = root / "link"
            link.symlink_to(source)
            with self.assertRaises(ValueError):
                MODULE.authenticate_file(
                    link, expected_bytes=5, expected_sha256=digest,
                    root=root, label="symlink",
                )
            with self.assertRaises(ValueError):
                MODULE.authenticate_file(
                    source, expected_bytes=5, expected_sha256="0" * 64,
                    root=root, label="tamper",
                )
            root_link = root.parent / (root.name + "-root-link")
            root_link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE._inside(root_link / source.name, root_link, "root-link")

    def test_converter_output_is_bounded_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            converter = root / "converter"
            converter.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "raw=sys.stdin.buffer.read()\n"
                "sys.stdout.buffer.write(b''.join(raw[i:i+2] for i in range(0,len(raw),4)))\n",
                encoding="utf-8",
            )
            converter.chmod(0o700)
            source = root / "source.bin"
            source.write_bytes(bytes(range(8)))
            self.assertEqual(MODULE._convert(converter, source, 0, 8), bytes([0, 1, 4, 5]))
            with self.assertRaises(ValueError):
                MODULE._convert(converter, source, 0, 6)

    def test_header_names_are_relative_and_mode_specific(self):
        prefix = b"prefix"
        plan = {
            "bin": {"name": "flash_post_state_turbo_layer00.bin", "bytes": 8,
                    "sha256": "1" * 64},
            "segments": [{
                "offset": 0, "bytes": len(prefix), "source_offset": 0,
                "source": 2, "learned": False, "sha256": MODULE.ZERO_SHA,
                "encoding": "copy",
            }],
        }
        fake = {
            "dit": {"name": "dit.safetensors", "bytes": 10, "sha256": "2" * 64,
                    "header_bytes": 8},
            "lq": {"name": "lq_proj_in.safetensors", "bytes": 11, "sha256": "3" * 64,
                   "header_bytes": 9},
            "prefix": prefix,
        }
        text = MODULE._render_header(
            {"turbo": {**fake, "prefix": prefix}},
            {"turbo": [plan] * 30},
        )
        self.assertIn("kFlashPostXml", text)
        self.assertIn("kFlashPostTurboBins", text)
        self.assertIn("kFlashPostTurboPlans", text)
        self.assertIn("kFlashPostWeights", text)
        self.assertIn("port_name", text)
        self.assertNotIn("kFlashPre", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("/cache/", text)

    def test_fresh_output_rejects_existing_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(ValueError):
                MODULE.fresh_output(existing)
            link = root / "link"
            link.symlink_to(existing, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.fresh_output(link)

    def test_fresh_output_rejects_symlink_parent_and_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = MODULE.fresh_output(root / "new" / "stage")
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)

            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.fresh_output(linked_parent / "created" / "stage")
            self.assertFalse((real_parent / "created").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
