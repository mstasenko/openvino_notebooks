#!/usr/bin/env python3
"""Pure metadata, reverse-gather, mask, and storage-contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "dynamic_window_reverse_gather_block16_under_test",
    HERE / "dynamic_window_reverse_gather_block16_export.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _ZeroRope:
    def get_freqs(self, vid_shape, txt_shape):
        video_tokens = int(vid_shape.prod(dim=1).sum().item())
        text_tokens = int(txt_shape.prod(dim=1).sum().item())
        return torch.zeros((video_tokens, 126)), torch.zeros((text_tokens, 126))


class DynamicContractTests(unittest.TestCase):
    def test_block16_uses_the_unshifted_even_layer_method(self):
        self.assertEqual(MODULE.BLOCK_INDEX, 16)
        self.assertEqual(MODULE.WINDOW_METHOD, "720pwin_by_size_bysize")
        self.assertFalse(MODULE.IS_LAST_LAYER)

    def test_t2_portrait_is_in_the_eager_and_saved_ir_matrix(self):
        self.assertIn((2, 135, 240), MODULE.CASE_SHAPES)

    def _contract(self, shape):
        import sys
        sys.path.insert(0, str(MODULE.SOURCE))
        from src.models.dit_3b.window import get_window_op
        slices = get_window_op(MODULE.WINDOW_METHOD)(shape, MODULE.WINDOW)
        max_len = max(
            (st.stop - st.start) * (sh.stop - sh.start) * (sw.stop - sw.start)
            for st, sh, sw in slices
        )
        return MODULE.make_window_contract(
            shape,
            max_video_len=max_len,
            txt_len=58,
            rope=_ZeroRope(),
            torch=torch,
        )

    def test_active_masks_cover_only_window_video_and_text(self):
        for shape in MODULE.CASE_SHAPES:
            contract = self._contract(shape)
            count = contract["window_count"]
            lengths = contract["window_lengths"]
            self.assertEqual(int(contract["window_idx"].shape[0]), count)
            for index, length in enumerate(lengths):
                mask = contract["window_key_mask"][index]
                self.assertEqual(int(mask[:length].sum().item()), length)
                split = contract["max_video_len"]
                self.assertEqual(int(mask[split : split + 58].sum().item()), 58)
                self.assertEqual(int(mask[split + 58 :].sum().item()), 0)

    def test_storage_is_window_count_times_max_window_square(self):
        for shape in MODULE.CASE_SHAPES:
            contract = self._contract(shape)
            count = contract["window_count"]
            padded_len = contract["max_attention_len"]
            packed_tokens = sum(contract["window_lengths"]) + count * 58
            per_head = count * padded_len * padded_len
            self.assertEqual(
                MODULE.attention_storage_elements(count, padded_len, heads=1), per_head
            )
            self.assertLess(per_head, packed_tokens * packed_tokens)
            self.assertEqual(int(contract["window_idx"].shape[0]), count)
            self.assertEqual(
                int(contract["window_rope_freqs"].shape[-1]), 126
            )

    def test_distinct_temporal_and_spatial_contracts(self):
        spatial = self._contract((1, 48, 60))
        temporal = self._contract((2, 48, 60))
        self.assertNotEqual(spatial["window_count"], temporal["window_count"])
        self.assertEqual(int(temporal["vid_shape"].prod().item()), 5760)
        self.assertEqual(int(temporal["window_idx"].shape[0]), 18)
        self.assertTrue(MODULE.SHARED_WEIGHTS)
        self.assertFalse(MODULE.IS_LAST_LAYER)

    def test_runtime_padding_length_is_case_local_and_dynamic(self):
        short = self._contract((1, 60, 60))
        base = self._contract((1, 48, 60))
        wide = self._contract((1, 60, 90))
        self.assertEqual(short["window_idx"].shape[1], short["max_video_len"])
        self.assertEqual(base["window_idx"].shape[1], base["max_video_len"])
        self.assertEqual(wide["window_idx"].shape[1], wide["max_video_len"])
        self.assertNotEqual(short["max_video_len"], base["max_video_len"])
        self.assertNotEqual(base["max_video_len"], wide["max_video_len"])

    def test_production_cap_shapes_fit_without_padding_rows(self):
        for shape in (
            (2, 240, 240),
            (2, 135, 240),
            (2, 240, 135),
            (1, 1, 240),
            (1, 240, 1),
            (2, 1, 240),
            (2, 240, 1),
        ):
            contract = self._contract(shape)
            self.assertLessEqual(contract["window_count"], MODULE.PRODUCTION_MAX_WINDOWS)
            self.assertLessEqual(max(contract["window_lengths"]), MODULE.PRODUCTION_MAX_VIDEO_LEN)
            self.assertEqual(
                tuple(contract["window_idx"].shape[1:]),
                (contract["max_video_len"],),
            )

    def test_reverse_gather_has_exactly_one_valid_position_per_source_token(self):
        shapes = tuple(self._contract(shape) for shape in MODULE.CASE_SHAPES) + tuple(
            self._contract(shape)
            for shape in (
                (2, 240, 240),
                (2, 135, 240),
                (2, 240, 135),
            )
        )
        for contract in shapes:
            source_tokens = int(contract["vid_shape"].prod().item())
            reverse = contract["window_reverse_idx"]
            self.assertEqual(tuple(reverse.shape), (source_tokens,))
            self.assertEqual(int(torch.unique(reverse).numel()), source_tokens)
            window_count = int(contract["window_idx"].shape[0])
            padded_len = int(contract["max_video_len"])
            self.assertTrue(bool((reverse >= 0).all().item()))
            self.assertTrue(bool((reverse < window_count * padded_len).all().item()))
            valid_positions = contract["window_key_mask"][:, :padded_len].reshape(-1)
            self.assertTrue(bool(valid_positions.index_select(0, reverse).all().item()))
            source_occurrences = torch.bincount(
                contract["window_idx"][:, :padded_len][
                    contract["window_key_mask"][:, :padded_len]
                ],
                minlength=source_tokens,
            )
            self.assertTrue(bool(torch.equal(source_occurrences, torch.ones_like(source_occurrences))))


if __name__ == "__main__":
    unittest.main()
