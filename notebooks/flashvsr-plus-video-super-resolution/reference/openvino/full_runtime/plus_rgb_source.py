"""Authenticated bounded RGB24 source windows for FlashVSR+.

This reader owns one read-only memmap, one current resized/padded window, and
no normalized-F32 source owner. A caller supplies the authenticated
``source_rgb`` receipt emitted by the existing preparer and a canonical
``StreamPlan``. The plan is the only authority that may introduce repeated
tail frames.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve()
DIST = HERE.parents[2]
DEFAULT_INPUT_ROOT = Path(
    os.environ.get(
        "FLASHVSR_PLUS_RGB_REFERENCE",
        str(DIST / "external" / "rgb-reference"),
    )
).resolve()
DEFAULT_FRAMES = (DEFAULT_INPUT_ROOT / "frames.rgb24").resolve()
DEFAULT_MANIFEST = (DEFAULT_INPUT_ROOT / "manifest.json").resolve()
EXPECTED_STATUS = "pass_common_rgb_only_no_model"
EXPECTED_FILE = "frames.rgb24"
MAX_WINDOW_FRAMES = 16
TEMPORAL_COMPRESSION = 4
# Keep the raw source bound aligned with the preparer's admitted 49-frame
# stream and FrameGeometry's reviewed padded-UHD area. Check it before the
# read-only memmap is created so malformed metadata cannot expose an
# unbounded source mapping to a caller.
MAX_SOURCE_FRAMES = 49
MAX_SOURCE_SIDE = 3840
MAX_SOURCE_AREA = 2176 * MAX_SOURCE_SIDE


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_frame(frame: np.ndarray) -> str:
    if not frame.flags.c_contiguous:
        raise ValueError("RGB source frame must be C-contiguous")
    return hashlib.sha256(memoryview(frame).cast("B")).hexdigest()


def _hex(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _same_json(left: Any, right: Any) -> bool:
    """Exact JSON equality; bool must not compare equal to integer metadata."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (set(left) == set(right)
                and all(_same_json(left[key], right[key]) for key in left))
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right)
        )
    return left == right


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


class RGBSourceWindowReader:
    """Read authenticated resized/edge-padded RGB windows without caching."""

    def __init__(
        self,
        *,
        frames_path: Path | str = DEFAULT_FRAMES,
        manifest_path: Path | str = DEFAULT_MANIFEST,
        input_receipt: dict[str, Any],
        target_hw: tuple[int, int],
    ) -> None:
        self.frames_path = Path(frames_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self._closed = False
        self._raw = None
        self._manifest, self._source_count, self._frame_receipts = (
            self._authenticate_source(input_receipt)
        )
        if (not isinstance(target_hw, tuple) or len(target_hw) != 2
                or any(type(value) is not int for value in target_hw)):
            raise ValueError("target_hw must be a tuple of two integers")
        try:
            from frame_geometry import FrameGeometry
            geometry = FrameGeometry(*target_hw)
        except (ImportError, TypeError, ValueError) as error:
            raise ValueError(f"invalid target geometry: {target_hw!r}") from error
        self.visible_hw = (geometry.height, geometry.width)
        self.working_hw = tuple(geometry.working_hw)
        expected_policy = {
            "geometry": [geometry.height, geometry.width],
            "resize": "PIL.Image.resize full frame with BICUBIC; no center crop",
            "normalization": "float32 = uint8 / 127.5 - 1.0; CHW -> [1,3,F,H,W]",
            "source_policy": (
                "original authenticated RGB24 frames; full-frame bicubic resize; "
                "bottom/right edge padding; no crop"
            ),
        }
        if any(not _same_json(input_receipt.get(name), value)
               for name, value in expected_policy.items()):
            raise ValueError("source resize/geometry/normalization policy changed")
        shape = tuple(self._manifest["shape"])
        self._raw = np.memmap(
            self.frames_path,
            mode="r",
            dtype=np.uint8,
            shape=shape,
            order="C",
        )

    def _authenticate_source(
        self, input_receipt: dict[str, Any]
    ) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
        if not isinstance(input_receipt, dict):
            raise ValueError("authenticated source_rgb receipt is required")
        if (self.frames_path.name != EXPECTED_FILE
                or self.manifest_path.name != "manifest.json"):
            raise ValueError("source payload/manifest filenames are not pinned")
        if not self.manifest_path.is_file() or not self.frames_path.is_file():
            raise FileNotFoundError("authenticated RGB payload or manifest is missing")
        manifest_hash = _sha256_file(self.manifest_path)
        if input_receipt.get("manifest_sha256") != manifest_hash:
            raise ValueError("source manifest fingerprint changed")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("RGB24 source manifest must be an object")
        if (manifest.get("status") != EXPECTED_STATUS
                or manifest.get("file") != EXPECTED_FILE
                or manifest.get("dtype") != "uint8"):
            raise ValueError("RGB24 source manifest status/file/dtype changed")
        shape_value = manifest.get("shape")
        if (not isinstance(shape_value, list) or len(shape_value) != 4
                or any(type(value) is not int or value <= 0 for value in shape_value)
                or shape_value[3] != 3):
            raise ValueError("RGB24 source shape changed")
        shape = tuple(shape_value)
        if (shape[0] > MAX_SOURCE_FRAMES
                or max(shape[1], shape[2]) > MAX_SOURCE_SIDE
                or shape[1] * shape[2] > MAX_SOURCE_AREA):
            raise ValueError("RGB24 source shape exceeds admitted geometry")
        expected_bytes = math.prod(shape)
        if (type(manifest.get("bytes")) is not int
                or manifest.get("bytes") != expected_bytes
                or self.frames_path.stat().st_size != expected_bytes):
            raise ValueError("RGB24 source byte size changed")
        _hex(manifest.get("sha256"), "source payload hash")
        if (type(input_receipt.get("payload_bytes")) is not int
                or input_receipt.get("manifest_path") != str(self.manifest_path)
                or input_receipt.get("payload_path") != str(self.frames_path)
                or input_receipt.get("payload_bytes") != expected_bytes
                or input_receipt.get("payload_sha256_expected") != manifest["sha256"]):
            raise ValueError("source payload/manifest paths or hash receipt changed")
        if (not _same_json(input_receipt.get("shape"), shape_value)
                or input_receipt.get("dtype") != "uint8"):
            raise ValueError("source geometry receipt changed")

        frames = manifest.get("frames")
        if not isinstance(frames, list) or len(frames) != shape[0]:
            raise ValueError("RGB24 frame receipts are incomplete")
        for index, frame in enumerate(frames):
            if (not isinstance(frame, dict) or type(frame.get("index")) is not int
                    or frame.get("index") != index
                    or type(frame.get("source_pts")) is not int):
                raise ValueError("RGB24 frame index/PTS receipt changed")
            _hex(frame.get("raw_sha256"), f"source frame {index} raw hash")
        selection = input_receipt.get("selection")
        count = _positive_int(input_receipt.get("consumed_frame_count"), "consumed frame count")
        if count > shape[0] or not _same_json(selection, {"start_frame": 0, "count": count}):
            if not (count == 17 and "selection" not in input_receipt
                    and "source_pts" not in input_receipt):
                raise ValueError("source selection receipt changed")
        expected_hashes = [frame["raw_sha256"] for frame in frames[:count]]
        expected_pts = [frame["source_pts"] for frame in frames[:count]]
        if not _same_json(input_receipt.get("consumed_frame_raw_sha256"), expected_hashes):
            raise ValueError("source frame hash/PTS receipt changed")
        if "selection" in input_receipt or "source_pts" in input_receipt:
            if (not _same_json(selection, {"start_frame": 0, "count": count})
                    or not _same_json(input_receipt.get("source_pts"), expected_pts)):
                raise ValueError("source frame hash/PTS receipt changed")
        if len(input_receipt["consumed_frame_raw_sha256"]) != count:
            raise ValueError("source frame receipt length changed")
        return manifest, count, frames

    @property
    def source_frames(self) -> int:
        return self._source_count

    def close(self) -> None:
        if not self._closed:
            raw, self._raw = self._raw, None
            del raw
            self._closed = True

    def __enter__(self) -> "RGBSourceWindowReader":
        if self._closed:
            raise ValueError("RGB source reader is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _validate_plan(self, plan: Any) -> None:
        try:
            from plus_stream import StreamPlan, build_stream_plan
        except ImportError as error:  # pragma: no cover - packaging error
            raise RuntimeError("Plus stream plan helper is unavailable") from error
        if type(plan) is not StreamPlan:
            raise ValueError("a canonical StreamPlan is required")
        if (type(plan.source_frames) is not int
                or type(plan.temporal_compression) is not int):
            raise ValueError("stream plan scalar types changed")
        if plan.source_frames != self._source_count:
            raise ValueError("stream plan source count differs from authenticated receipt")
        if type(plan.fps) is not float:
            raise ValueError("stream plan FPS type changed")
        if plan.temporal_compression != TEMPORAL_COMPRESSION:
            raise ValueError("stream plan temporal compression changed")
        canonical = build_stream_plan(
            self._source_count,
            plan.fps,
            temporal_compression=TEMPORAL_COMPRESSION,
        )
        if not _same_json(asdict(canonical), asdict(plan)):
            raise ValueError("stream plan is not canonical for the authenticated source")

    def _read_indices(self, indices: tuple[int, ...]) -> np.ndarray:
        if self._closed or self._raw is None:
            raise ValueError("RGB source reader is closed")
        if not 1 <= len(indices) <= MAX_WINDOW_FRAMES:
            raise ValueError("RGB source window must contain 1..16 frames")
        if any(type(index) is not int or not 0 <= index < self._source_count for index in indices):
            raise ValueError("RGB source window contains an unauthenticated frame index")
        target_height, target_width = self.visible_hw
        working_height, working_width = self.working_hw
        output = np.empty(
            (1, 3, len(indices), working_height, working_width), dtype=np.uint8
        )
        for offset, index in enumerate(indices):
            # Snapshot before authentication and decode: the underlying
            # read-only mapping may otherwise change between those operations.
            raw = np.array(self._raw[index], dtype=np.uint8, copy=True, order="C")
            if _sha256_frame(raw) != self._frame_receipts[index]["raw_sha256"]:
                raise ValueError(f"RGB source frame {index} raw hash mismatch")
            visible = np.asarray(
                Image.fromarray(raw, mode="RGB").resize(
                    (target_width, target_height), resample=Image.Resampling.BICUBIC
                ),
                dtype=np.uint8,
            )
            if visible.shape != (target_height, target_width, 3):
                raise ValueError("resized RGB frame geometry changed")
            padded = np.pad(
                visible,
                ((0, working_height - target_height),
                 (0, working_width - target_width), (0, 0)),
                mode="edge",
            )
            output[0, :, offset] = padded.transpose(2, 0, 1)
        # The preallocated output owns the current window; no output
        # references the memmap and no previous window is retained.
        return output

    def read_call(self, plan: Any, call_index: int, *, stream: str = "decoder") -> np.ndarray:
        """Read one bounded decoder call from the canonical plan."""
        if stream != "decoder":
            raise ValueError("RGB source reader is decoder-only; projection is unsupported")
        self._validate_plan(plan)
        call = plan.call(call_index)
        indices = call.decoder_source_indices
        if len(indices) > MAX_WINDOW_FRAMES:
            raise ValueError("decoder call exceeds the bounded 16-frame window")
        return self._read_indices(indices)

    def read_decoder_inner(self, plan: Any, call_index: int, inner: int) -> np.ndarray:
        """Read exactly one four-frame decoder pack window."""
        self._validate_plan(plan)
        if type(inner) is not int or not 0 <= inner < 4:
            raise ValueError("decoder inner must be an integer in [0,4)")
        call = plan.call(call_index)
        source = call.decoder_source_indices
        if len(source) == 13:
            indices = (source[0],) * 4 if inner == 0 else source[1 + 4 * (inner - 1):1 + 4 * inner]
        else:
            indices = source[4 * inner:4 * (inner + 1)]
        if len(indices) != 4:
            raise ValueError("stream plan did not provide a complete decoder inner window")
        return self._read_indices(tuple(indices))

    def normalized_projection(self, plan: Any) -> np.ndarray:
        """Build the canonical normalized projection timeline from decoder windows.

        Only the bounded 13/16-frame decoder windows are materialized.  Each
        authenticated source frame is normalized directly into the single
        returned F32 owner, then the plan's final source frame is repeated to
        the padded projection length.
        """
        self._validate_plan(plan)
        if not 17 <= plan.source_frames <= MAX_SOURCE_FRAMES:
            raise ValueError("normalized projection requires 17..49 source frames")
        height, width = self.working_hw
        projection = np.empty(
            (1, 3, plan.projection_frames, height, width), dtype=np.float32
        )
        next_source = 0
        for call_index in range(plan.call_count):
            call = plan.call(call_index)
            window = self.read_call(plan, call_index)
            if window.shape[2] != len(call.decoder_source_indices):
                raise ValueError("decoder source window shape changed")
            for offset, source_index in enumerate(call.decoder_source_indices):
                if source_index == next_source:
                    destination = projection[0, :, next_source]
                    np.copyto(destination, window[0, :, offset], casting="safe")
                    np.divide(destination, np.float32(127.5), out=destination)
                    np.subtract(destination, np.float32(1.0), out=destination)
                    next_source += 1
                elif source_index != next_source - 1:
                    raise ValueError("decoder windows are not contiguous source chronology")
            del window
        if next_source != plan.source_frames:
            raise ValueError("decoder windows did not cover the source timeline")
        if plan.projection_frames > next_source:
            projection[:, :, next_source:] = projection[:, :, next_source - 1:next_source]
        return projection


__all__ = ["RGBSourceWindowReader", "DEFAULT_FRAMES", "DEFAULT_MANIFEST"]
