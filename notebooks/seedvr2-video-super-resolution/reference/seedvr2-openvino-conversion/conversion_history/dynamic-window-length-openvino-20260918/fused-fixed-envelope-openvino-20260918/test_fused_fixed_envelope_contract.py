#!/usr/bin/env python3
"""Pure fixed-envelope metadata tests; no model weights or OpenVINO device use."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "fused_fixed_envelope_under_test",
    HERE / "fused_fixed_envelope_export.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_contract(window_count: int) -> dict[str, object]:
    idx = torch.zeros((window_count, MODULE.ACTUAL_MAX_VIDEO_LEN), dtype=torch.long)
    mask = torch.zeros(
        (window_count, MODULE.ATTENTION_LEN), dtype=torch.bool
    )
    rope = torch.zeros(
        (window_count, MODULE.ATTENTION_LEN, MODULE.ROPE_WIDTH), dtype=torch.float32
    )
    reverse = torch.arange(MODULE.SOURCE_TOKENS, dtype=torch.long)
    # The pure test only exercises envelope construction.  Give each row one
    # active video token and let the validator's source-token coverage check be
    # bypassed by using the production-shaped row count in the actual test.
    mask[:, 0] = True
    return {
        "window_count": window_count,
        "max_video_len": MODULE.ACTUAL_MAX_VIDEO_LEN,
        "window_idx": idx,
        "window_key_mask": mask,
        "window_rope_freqs": rope,
        "window_reverse_idx": reverse,
    }


class FixedEnvelopeContractTests(unittest.TestCase):
    def test_chunk_ranges_cover_exact_fixed_envelope(self) -> None:
        ranges = MODULE.chunk_ranges()
        self.assertEqual(len(ranges), 13)
        self.assertEqual(ranges[0], (0, 16))
        self.assertEqual(ranges[-1], (192, 208))
        self.assertEqual([end - start for start, end in ranges], [16] * 13)
        self.assertEqual(ranges[-1][1], MODULE.WINDOW_CAPACITY)

    def test_block0_padding_adds_46_dummy_rows_and_one_safe_key(self) -> None:
        padded = MODULE.pad_window_metadata(fake_contract(162), torch)
        self.assertEqual(padded["actual_window_count"], 162)
        self.assertEqual(padded["dummy_window_count"], 46)
        self.assertEqual(tuple(padded["window_idx"].shape), (208, 405))
        self.assertEqual(tuple(padded["window_key_mask"].shape), (208, 463))
        self.assertEqual(tuple(padded["window_rope_freqs"].shape), (208, 463, 126))
        self.assertTrue(bool(padded["window_valid"][:162].all().item()))
        self.assertFalse(bool(padded["window_valid"][162:].any().item()))
        self.assertTrue(bool(padded["window_key_mask"][162:, 0].all().item()))
        self.assertFalse(bool(padded["window_key_mask"][162:, 1:].any().item()))

    def test_block16_padding_adds_46_dummy_rows_and_mutation_preserves_valid_flags(self) -> None:
        padded = MODULE.pad_window_metadata(fake_contract(162), torch)
        mutated = MODULE.mutate_dummy_metadata(padded, torch)
        self.assertEqual(padded["dummy_window_count"], 46)
        self.assertTrue(bool(torch.equal(padded["window_valid"], mutated["window_valid"])))
        self.assertTrue(bool(torch.equal(padded["window_reverse_idx"], mutated["window_reverse_idx"])))
        self.assertTrue(bool(mutated["window_key_mask"][162:, 0].all().item()))
        self.assertFalse(bool(mutated["window_key_mask"][162:, 1:].any().item()))
        self.assertTrue(bool((mutated["window_idx"][162:] == MODULE.SOURCE_TOKENS - 1).all().item()))


if __name__ == "__main__":
    unittest.main()
