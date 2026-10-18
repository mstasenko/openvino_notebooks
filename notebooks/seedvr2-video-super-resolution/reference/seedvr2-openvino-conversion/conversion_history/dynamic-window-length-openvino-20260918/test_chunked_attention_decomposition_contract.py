#!/usr/bin/env python3
"""Pure contract tests for the chunked-attention decomposition."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "chunked_attention_decomposition_under_test",
    HERE / "chunked_attention_decomposition_export.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _ZeroRope:
    def get_freqs(self, vid_shape, txt_shape):
        video_tokens = int(vid_shape.prod(dim=1).sum().item())
        text_tokens = int(txt_shape.prod(dim=1).sum().item())
        return torch.zeros((video_tokens, 126)), torch.zeros((text_tokens, 126))


def _base(block_index: int):
    path = HERE / f"dynamic_window_reverse_gather_block{block_index}_export.py"
    spec = importlib.util.spec_from_file_location(f"reverse_base_{block_index}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChunkedDecompositionContractTests(unittest.TestCase):
    def _contract(self, block_index: int, shape: tuple[int, int, int]):
        base = _base(block_index)
        import sys

        sys.path.insert(0, str(MODULE.SOURCE))
        from src.models.dit_3b.window import get_window_op

        slices = get_window_op(base.WINDOW_METHOD)(shape, base.WINDOW)
        max_len = max(
            (st.stop - st.start) * (sh.stop - sh.start) * (sw.stop - sw.start)
            for st, sh, sw in slices
        )
        return base, base.make_window_contract(
            shape,
            max_video_len=max_len,
            txt_len=58,
            rope=_ZeroRope(),
            torch=torch,
        )

    def test_exact_one_hit_video_coverage_for_both_methods(self):
        shapes = (
            (1, 48, 60),
            (1, 60, 60),
            (2, 48, 60),
            (1, 60, 90),
            (1, 90, 60),
            (2, 240, 240),
            (2, 135, 240),
            (2, 240, 135),
        )
        for block_index in (0, 16, 31):
            for shape in shapes:
                _, contract = self._contract(block_index, shape)
                result = MODULE.validate_window_coverage(contract, torch)
                self.assertEqual(result["valid_occurrence_min"], 1)
                self.assertEqual(result["valid_occurrence_max"], 1)
                self.assertTrue(result["reverse_positions_unique"])

    def test_bounded_score_storage_for_full_temporal_portrait_case(self):
        expected = {}
        for block_index in (0, 16, 31):
            base, contract = self._contract(block_index, (2, 135, 240))
            summary = MODULE.window_metadata(base, (2, 135, 240))
            peak = MODULE.attention_score_elements(
                summary["window_count"], summary["max_video_len"], chunk_windows=16
            )
            whole = MODULE.attention_score_elements(
                summary["window_count"], summary["max_video_len"]
            )
            self.assertLess(peak, whole)
            self.assertEqual(summary["max_video_len"], 405)
            self.assertLessEqual(summary["window_count"], MODULE.PRODUCTION_MAX_WINDOWS)
            self.assertLessEqual(summary["max_video_len"], MODULE.PRODUCTION_MAX_VIDEO_LEN)
            expected[block_index] = (summary["window_count"], peak)
            MODULE.validate_window_coverage(contract, torch)
        self.assertEqual(expected[0][0], 162)
        self.assertEqual(expected[16][0], 162)
        self.assertEqual(expected[31][0], 200)

    def test_chunk_size_invariance_and_exact_text_count(self):
        chunk = MODULE.build_chunk_module(torch)
        generator = torch.Generator(device="cpu").manual_seed(20260918)
        windows = 4
        video_len = 6
        video_tokens = windows * video_len
        vid_q = torch.randn((video_tokens, MODULE.HEADS, MODULE.HEAD_DIM), generator=generator)
        vid_k = torch.randn((video_tokens, MODULE.HEADS, MODULE.HEAD_DIM), generator=generator)
        vid_v = torch.randn((video_tokens, MODULE.HEADS, MODULE.HEAD_DIM), generator=generator)
        txt_q = torch.randn((MODULE.TXT_TOKENS, MODULE.HEADS, MODULE.HEAD_DIM), generator=generator)
        txt_k = torch.randn((MODULE.TXT_TOKENS, MODULE.HEADS, MODULE.HEAD_DIM), generator=generator)
        txt_v = torch.randn((MODULE.TXT_TOKENS, MODULE.HEADS, MODULE.HEAD_DIM), generator=generator)
        idx = torch.arange(video_tokens, dtype=torch.long).reshape(windows, video_len)
        mask = torch.ones((windows, video_len + MODULE.TXT_TOKENS), dtype=torch.bool)
        rope = torch.zeros((windows, video_len + MODULE.TXT_TOKENS, 126), dtype=torch.float32)
        full_vid, full_txt_sum, full_count = chunk(
            vid_q, vid_k, vid_v, txt_q, txt_k, txt_v, idx, mask, rope
        )
        self.assertEqual(float(full_count.item()), float(windows))
        for chunk_size in (1, 4, 16, windows):
            vid_parts = []
            txt_sum = torch.zeros_like(full_txt_sum)
            count = 0.0
            for start in range(0, windows, chunk_size):
                end = min(windows, start + chunk_size)
                out_vid, out_txt, out_count = chunk(
                    vid_q,
                    vid_k,
                    vid_v,
                    txt_q,
                    txt_k,
                    txt_v,
                    idx[start:end],
                    mask[start:end],
                    rope[start:end],
                )
                vid_parts.append(out_vid)
                txt_sum += out_txt
                count += float(out_count.item())
            self.assertTrue(torch.equal(torch.cat(vid_parts, dim=0), full_vid))
            self.assertTrue(torch.allclose(txt_sum, full_txt_sum, atol=1.0e-6, rtol=0.0))
            self.assertEqual(count, float(windows))
            self.assertTrue(torch.allclose(txt_sum / count, full_txt_sum / full_count, atol=1.0e-6, rtol=0.0))

    def test_chunk_contract_has_no_scatter_or_index_add(self):
        source = inspect.getsource(MODULE.build_chunk_module)
        self.assertNotIn("index_add", source)
        self.assertNotIn("scatter", source.lower())
        self.assertNotIn("nonzero", source.lower())
        self.assertIn("scaled_dot_product_attention", source)

    def test_block31_is_the_video_only_last_layer(self):
        base = _base(31)
        self.assertTrue(base.SHARED_WEIGHTS)
        self.assertTrue(base.IS_LAST_LAYER)
        self.assertEqual(base.BLOCK_INDEX, 31)


if __name__ == "__main__":
    unittest.main()
