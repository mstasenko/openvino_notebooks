#!/usr/bin/env python3
"""CPU-only pure tests for the B580 runner admission and shape contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

from b580_chunked_window_validation_runner import (
    CHUNK_SIZE,
    PRODUCTION_SHAPE,
    RunnerInputError,
    chunk_ranges,
    parse_preflight_marker,
    production_token_count,
    validate_preflight_marker,
    validate_shape_chunk_accounting,
)


class PreflightContractTests(unittest.TestCase):
    def test_json_marker_requires_exact_target(self) -> None:
        marker = json.dumps(
            {
                "status": "pass",
                "bdf": "0000:03:00.0",
                "driver": "xe",
                "render_node": "renderD128",
                "device": "B580",
            }
        )
        self.assertEqual(
            parse_preflight_marker(marker),
            {
                "bdf": "0000:03:00.0",
                "driver": "xe",
                "render_node": "renderD128",
                "device": "B580",
            },
        )

    def test_nested_marker_and_stale_marker_fail_closed(self) -> None:
        marker = json.dumps(
            {
                "status": "ready",
                "target": {
                    "bdf": "0000:03:00.0",
                    "driver": "xe",
                    "render_node": "renderD128",
                    "device": "B580",
                },
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker.json"
            path.write_text(marker, encoding="utf-8")
            now = path.stat().st_mtime + 61.0
            with self.assertRaises(RunnerInputError):
                validate_preflight_marker(path, now=now)

    def test_preflight_receipt_aliases_are_normalized(self) -> None:
        marker = json.dumps(
            {
                "status": "pass",
                "target": {
                    "pci_bdf": "0000:03:00.0",
                    "driver": "xe",
                    "render_node": "/dev/dri/renderD128",
                    "pci_device": "Intel Arc B580 / Battlemage G21",
                    "openvino_selector": "GPU.1",
                },
            }
        )
        self.assertEqual(
            parse_preflight_marker(marker),
            {
                "bdf": "0000:03:00.0",
                "driver": "xe",
                "render_node": "renderD128",
                "device": "B580",
            },
        )

    def test_wrong_target_is_rejected(self) -> None:
        with self.assertRaises(RunnerInputError):
            parse_preflight_marker(
                json.dumps(
                    {
                        "bdf": "0000:04:00.0",
                        "driver": "xe",
                        "render_node": "renderD128",
                        "device": "B580",
                    }
                )
            )


class ShapeChunkContractTests(unittest.TestCase):
    def test_production_shape_and_token_count(self) -> None:
        self.assertEqual(PRODUCTION_SHAPE, (2, 135, 240))
        self.assertEqual(production_token_count(), 2 * 135 * 240)

    def test_chunk_ranges_cover_windows_once(self) -> None:
        ranges = chunk_ranges(31, CHUNK_SIZE)
        self.assertEqual(ranges, ((0, 16), (16, 31)))
        self.assertEqual(
            validate_shape_chunk_accounting(
                shape=PRODUCTION_SHAPE,
                window_count=31,
                max_video_len=537,
            )["chunk_count"],
            2,
        )
        flattened = [value for start, end in ranges for value in range(start, end)]
        self.assertEqual(flattened, list(range(31)))

    def test_chunk_accounting_rejects_nonproduction_shape(self) -> None:
        with self.assertRaises(Exception):
            validate_shape_chunk_accounting(
                shape=(2, 240, 240),
                window_count=31,
                max_video_len=537,
            )


if __name__ == "__main__":
    unittest.main()
