#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import time
import unittest

import b580_fused_fixed_envelope_runner as runner


class RunnerContractTest(unittest.TestCase):
    def marker(self, root: Path) -> Path:
        path = root / "marker.json"
        path.write_text(json.dumps({"status": "pass", "target": runner.TARGET}))
        return path

    def test_marker_accepts_only_fresh_exact_b580(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed = runner.read_marker(self.marker(root))
            self.assertEqual(parsed["target"], runner.TARGET)
            old = self.marker(root)
            self.assertRaises(runner.ContractError, runner.read_marker, old, time.time() + 61)

    def test_help_does_not_require_or_write_a_receipt(self):
        with self.assertRaises(SystemExit) as result:
            runner.parser().parse_args(["--help"])
        self.assertEqual(result.exception.code, 0)

    def test_contract_constants_cover_full_envelope(self):
        self.assertEqual(runner.SOURCE_TOKENS, 64800)
        self.assertEqual(runner.WINDOW_CAPACITY, 13 * 16)
        self.assertEqual(runner.ATTENTION_LEN, runner.MAX_VIDEO_LEN + runner.TXT_TOKENS)


if __name__ == "__main__":
    unittest.main()
