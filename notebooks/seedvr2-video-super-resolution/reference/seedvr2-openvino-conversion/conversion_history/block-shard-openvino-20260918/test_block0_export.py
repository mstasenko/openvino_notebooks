#!/usr/bin/env python3
"""Focused pure-contract tests for the bounded block-0 exporter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("block0_export_under_test", HERE / "block0_export.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Block0ExportContractTests(unittest.TestCase):
    def test_filter_requires_exact_block0_shard(self) -> None:
        keys = [
            "blocks.1.unrelated",
            "blocks.0.z",
            "blocks.0.a",
            "embedding.weight",
        ]
        with self.assertRaises(AssertionError):
            MODULE.filter_block0_keys(keys)

    def test_filter_strips_prefix_and_preserves_exact_count(self) -> None:
        keys = [MODULE.BLOCK_PREFIX + f"tensor_{index:02d}" for index in range(MODULE.EXPECTED_TENSOR_COUNT)]
        result = MODULE.filter_block0_keys(keys)
        self.assertEqual(len(result), MODULE.EXPECTED_TENSOR_COUNT)
        self.assertEqual(result[MODULE.BLOCK_PREFIX + "tensor_00"], "tensor_00")

    def test_real_checkpoint_shard_contract(self) -> None:
        from safetensors import safe_open

        weights = MODULE.DEFAULT_WEIGHTS
        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            mapping = MODULE.filter_block0_keys(handle.keys())
            payload_bytes = 0
            for key in mapping:
                view = handle.get_slice(key)
                elements = 1
                for dimension in view.get_shape():
                    elements *= int(dimension)
                self.assertEqual(str(view.get_dtype()), "F16")
                payload_bytes += elements * 2
        self.assertEqual(len(mapping), MODULE.EXPECTED_TENSOR_COUNT)
        self.assertEqual(payload_bytes, MODULE.EXPECTED_PAYLOAD_BYTES)

    def test_input_and_embedded_shape_contracts(self) -> None:
        MODULE.validate_contract_shapes((2880, 2560), (58, 2560), (1, 15360))
        MODULE.validate_embedded_shapes((1, 48, 60), (1, 58))
        with self.assertRaises(AssertionError):
            MODULE.validate_contract_shapes((2880, 2560), (58, 2560), (1, 2560))
        with self.assertRaises(AssertionError):
            MODULE.validate_embedded_shapes((1, 48, 59), (1, 58))


if __name__ == "__main__":
    unittest.main()
