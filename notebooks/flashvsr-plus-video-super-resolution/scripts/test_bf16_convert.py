#!/usr/bin/env python3
"""Small black-box tests for the native raw-F32 -> raw-BF16 converter."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest


def _converter_path() -> Path:
    # CTest passes the target path as the first positional argument.  The
    # environment form is useful when running this file by hand.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        value = sys.argv.pop(1)
    else:
        value = os.environ.get("OTC_BF16_CONVERTER")
    if not value:
        raise SystemExit(
            "set OTC_BF16_CONVERTER or pass the converter executable path"
        )
    path = Path(value)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"converter is not an executable file: {path}")
    return path


CONVERTER = (_converter_path() if __name__ == "__main__" else
             Path(os.environ["OTC_BF16_CONVERTER"]) if "OTC_BF16_CONVERTER" in os.environ else None)


def _raw_f32(words: list[int] | tuple[int, ...]) -> bytes:
    return b"".join(struct.pack("<I", word) for word in words)


def _words(raw: bytes) -> tuple[int, ...]:
    if len(raw) % 2:
        raise AssertionError("converter returned an odd number of bytes")
    return struct.unpack(f"<{len(raw) // 2}H", raw)


def _run(raw: bytes, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(CONVERTER), *args],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@unittest.skipIf(CONVERTER is None, "native converter tests run through CTest after building")
class Bf16ConverterTests(unittest.TestCase):
    def test_round_to_even_signed_zero_subnormal_and_boundary(self) -> None:
        words = (
            0x00000000,
            0x80000000,
            0x3F800000,
            0x3F808000,  # halfway, lower BF16 mantissa even
            0x3F818000,  # halfway, lower BF16 mantissa odd
            0xBF808000,
            0xBF818000,
            0x00000001,
            0x007FFFFF,
            0x7F7F0000,
        )
        expected = (
            0x0000,
            0x8000,
            0x3F80,
            0x3F80,
            0x3F82,
            0xBF80,
            0xBF82,
            0x0000,
            0x0080,
            0x7F7F,
        )
        result = _run(_raw_f32(words))
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(_words(result.stdout), expected)

    def test_multichunk_stream_is_exact(self) -> None:
        # 1 MiB plus three words forces a full chunk and a short final chunk.
        count = (1 << 20) // 4 + 3
        result = _run(_raw_f32((0x3F800000,) * count))
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(_words(result.stdout), (0x3F80,) * count)

    def test_nonfinite_and_bf16_overflow_are_rejected(self) -> None:
        for word in (0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC00001, 0x7F7F8000):
            with self.subTest(word=f"0x{word:08x}"):
                result = _run(_raw_f32((word,)))
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)

    def test_partial_or_unaligned_input_is_rejected(self) -> None:
        for raw in (b"\x00", b"\x00\x00\x00", b"\x00" + _raw_f32((0x3F800000,))):
            with self.subTest(size=len(raw)):
                result = _run(raw)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)

    def test_bad_argument_is_rejected(self) -> None:
        result = _run(b"", "unexpected")
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"usage:", result.stderr)

    def test_empty_input_is_empty_success(self) -> None:
        result = _run(b"")
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(result.stdout, b"")

    @unittest.skipUnless(Path("/dev/full").exists(), "requires the Linux failing-write device")
    def test_failed_final_flush_is_failure(self) -> None:
        with open("/dev/full", "wb") as sink:
            result = subprocess.run([str(CONVERTER)], input=_raw_f32((0x3F800000,)),
                                    stdout=sink, stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"stdout", result.stderr)

    def test_partial_word_after_full_chunk_is_failure(self) -> None:
        result = _run(_raw_f32((0x3F800000,) * ((1 << 20) // 4)) + b"\0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"partial", result.stderr)


if __name__ == "__main__":
    unittest.main()
