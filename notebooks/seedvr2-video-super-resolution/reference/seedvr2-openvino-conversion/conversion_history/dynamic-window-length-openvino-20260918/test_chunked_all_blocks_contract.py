#!/usr/bin/env python3
"""Focused pure-CPU contract checks for the all-block chunked proof."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


HERE = Path(__file__).resolve().parent
HELPER_SPEC = importlib.util.spec_from_file_location(
    "chunked_all_blocks_helper", HERE / "chunked_attention_decomposition_export.py"
)
assert HELPER_SPEC is not None and HELPER_SPEC.loader is not None
HELPER = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPER)


class _ZeroRope:
    def get_freqs(self, vid_shape, txt_shape):
        video_tokens = int(vid_shape.prod(dim=1).sum().item())
        text_tokens = int(txt_shape.prod(dim=1).sum().item())
        return torch.zeros((video_tokens, 126)), torch.zeros((text_tokens, 126))


def _load(index: int):
    path = HERE / f"dynamic_window_reverse_gather_block{index}_export.py"
    spec = importlib.util.spec_from_file_location(f"all_blocks_base_{index}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AllBlockContractTests(unittest.TestCase):
    def test_all_indices_have_expected_weight_and_last_layer_family(self):
        for index in range(1, 32):
            base = _load(index)
            self.assertEqual(base.BLOCK_INDEX, index)
            shared = index >= 10
            self.assertEqual(base.SHARED_WEIGHTS, shared)
            self.assertEqual(base.IS_LAST_LAYER, index == 31)
            self.assertEqual(base.EXPECTED_TENSOR_COUNT, 15 if shared else 29)
            self.assertEqual(
                base.EXPECTED_PAYLOAD_BYTES,
                158_633_514 if shared else 317_266_986,
            )

    def test_all_indices_follow_even_unshifted_odd_shifted_method_alternation(self):
        for index in range(32):
            base = _load(index)
            expected = (
                "720pwin_by_size_bysize"
                if index % 2 == 0
                else "720pswin_by_size_bysize"
            )
            self.assertEqual(base.WINDOW_METHOD, expected, f"block {index}")

    def test_all_indices_build_exact_reverse_gather_contracts(self):
        for index in range(32):
            base = _load(index)
            import sys

            sys.path.insert(0, str(HELPER.SOURCE))
            from src.models.dit_3b.window import get_window_op

            for shape in ((1, 48, 60), (2, 48, 60), (2, 135, 240)):
                slices = get_window_op(base.WINDOW_METHOD)(shape, base.WINDOW)
                max_len = max(
                    (st.stop - st.start) * (sh.stop - sh.start) * (sw.stop - sw.start)
                    for st, sh, sw in slices
                )
                contract = base.make_window_contract(
                    shape,
                    max_video_len=max_len,
                    txt_len=58,
                    rope=_ZeroRope(),
                    torch=torch,
                )
                coverage = HELPER.validate_window_coverage(contract, torch)
                self.assertEqual(coverage["valid_occurrence_min"], 1)
                self.assertEqual(coverage["valid_occurrence_max"], 1)
                self.assertTrue(coverage["reverse_positions_unique"])


if __name__ == "__main__":
    unittest.main()
