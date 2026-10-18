"""CPU-only contract tests for the RemoteTensor evidence runner."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import inspect
import json
import time
import unittest
import uuid

import b580_split_ir_remote_tensor_runner as runner


def marker(*, age_seconds: float = 0.0, selector: str = runner.DEVICE) -> str:
    checked = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return json.dumps({
        "status": "pass",
        "checked_at_utc": checked.isoformat(),
        "openvino_selector": selector,
        "target": runner.TARGET,
    })


class RunnerContracts(unittest.TestCase):
    def test_help_exits_without_creating_requested_receipt(self) -> None:
        receipt = runner.HERE / f"receipt-help-{uuid.uuid4().hex}.json"
        self.assertFalse(receipt.exists())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            runner.main([
                "--help", "--preflight-marker", "missing.json", "--receipt", str(receipt)
            ])
        self.assertEqual(raised.exception.code, 0)
        self.assertFalse(receipt.exists())

    def test_preflight_requires_fresh_affirmative_gpu1_marker(self) -> None:
        record = runner.parse_preflight_marker(marker(), now=time.time())
        self.assertEqual(record["openvino_selector"], "GPU.1")
        with self.assertRaises(runner.RunnerInputError):
            runner.parse_preflight_marker(marker(selector="GPU"), now=time.time())
        with self.assertRaises(runner.RunnerInputError):
            runner.parse_preflight_marker(marker(age_seconds=61), now=time.time())

    def test_exact_chunk_coverage_and_counts(self) -> None:
        expected = {0: 162, 16: 162, 31: 200}
        self.assertEqual(tuple(expected), runner.BLOCKS)
        for block, windows in expected.items():
            ranges = runner.chunk_ranges(windows)
            flattened = [i for start, end in ranges for i in range(start, end)]
            self.assertEqual(flattened, list(range(windows)))
            self.assertTrue(all(end - start <= 16 for start, end in ranges))
            self.assertEqual(len(ranges), 13 if block == 31 else 11)

    def test_current_receipts_prove_one_shared_chunk_ir(self) -> None:
        proof = runner.validate_current_artifacts()["proof"]
        self.assertEqual(proof["artifact_files_checked"], 76)
        self.assertEqual(proof["artifact_hashes_match"], 76)
        self.assertEqual(proof["shared_chunk"]["blocks_checked"], 32)
        self.assertEqual(proof["shared_chunk"]["unique_binary_hashes"], 1)
        self.assertEqual(proof["shared_chunk"]["normalized_graph_variants"], 1)
        self.assertEqual(proof["shared_chunk"]["port_contract_variants"], 1)
        self.assertEqual(proof["shared_chunk"]["xml_hash_variants"], 4)
        self.assertTrue(proof["block16_unshifted"])
        self.assertEqual(
            {int(k): v["chunk_calls"] for k, v in proof["target_blocks"].items()},
            {0: 11, 16: 11, 31: 13},
        )

    def test_remote_qkv_bound_once(self) -> None:
        class Remote:
            def get_device_name(self) -> str:
                return "GPU.1"

        class Request:
            def __init__(self) -> None:
                self.bound: dict[str, Remote] = {}

            def set_tensor(self, name: str, tensor: Remote) -> None:
                self.bound[name] = tensor

        tensors = {name: Remote() for name in runner.QKV_NAMES}
        request = Request()
        self.assertEqual(runner.bind_remote_qkv(request, tensors, Remote), 6)
        self.assertEqual(request.bound, tensors)

    def test_remote_outputs_use_supported_gpu1_usm_path(self) -> None:
        source = inspect.getsource(runner)
        self.assertIn('ov.Shape(shape)', source)
        self.assertIn('"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"', source)
        self.assertIn('prefix_request.start_async()', source)
        self.assertIn('prefix_request.wait()', source)
        self.assertIn('getattr(helper, "BLOCK_INDEX", 0)', source)

    def test_inverse_gather_is_exact_with_stdlib_lists(self) -> None:
        reverse = [0, 4, 2, 6, 1, 5, 3, 7]
        window_idx = [[0, 4], [2, 6], [1, 5], [3, 7]]
        accumulator: list[int | None] = [None] * 8
        counts = [0] * 8
        for start in (0, 2):
            rows = window_idx[start : start + 2]
            source_ids = [source for row in rows for source in row]
            indices, positions = runner.inverse_gather_positions(
                source_ids,
                reverse,
                start=start,
                max_video_len=2,
                chunk_windows=2,
                source_token_count=8,
            )
            self.assertEqual(indices, source_ids)
            self.assertEqual(positions, [0, 1, 2, 3])
            flat_windows = [source + 100 for row in rows for source in row]
            for source, position in zip(indices, positions):
                accumulator[source] = flat_windows[position]
                counts[source] += 1
        self.assertEqual(accumulator, list(range(100, 108)))
        self.assertEqual(counts, [1] * 8)
        with self.assertRaises(runner.ContractError):
            runner.inverse_gather_positions(
                [0, 0], reverse, start=0, max_video_len=2,
                chunk_windows=2, source_token_count=8,
            )

if __name__ == "__main__":
    unittest.main()
