"""Dependency-free FlashVSR+ source/latent stream chronology.

The official runner fixes every denoiser call at four latent frames.  Call 0
projects model frames ``0:17`` (the unique first frame plus four 4-frame LQ
clips); later calls project ``17:33``, ``33:49``, and so on.  TinyDecoder
decoder ranges are ``0:13``, then ``13:29``, ``29:45``, ... .  This module
keeps those ranges arithmetic and derives one bounded call on demand; it does
not retain a video-length source-index table.

Spatial preparation remains the application's existing ``FrameGeometry``
contract: resize without a source crop, edge-pad to the 128 grid, and trim
only the bottom/right padding after decode.  This adapter deliberately does
not duplicate that spatial helper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


TEMPORAL_DOWNSAMPLE = 4
STREAM_CHUNK_LATENTS = 4


def _require_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _target_frame_count(frames: int, temporal_compression: int) -> int:
    """The pinned upstream ``target_frame_count`` formula."""
    return ((frames - 1 + temporal_compression - 1) // temporal_compression) * temporal_compression + 1


def _source_index_for_model_frame(
    model_frame: int,
    *,
    source_frames: int,
    expanded_frames: int,
    repeat_factor: int,
) -> int:
    """Map one expanded/padded model frame to an original source frame."""
    if type(model_frame) is not int or model_frame < 0:
        raise ValueError("model_frame must be a non-negative integer")
    if model_frame == 0:
        return 0
    if model_frame >= expanded_frames:
        return source_frames - 1
    source_position = 1 + (model_frame - 1) // repeat_factor
    return min(source_position, source_frames - 1)


@dataclass(frozen=True, slots=True)
class StreamCall:
    """One fixed four-latent call, with only its bounded ranges materialized."""

    index: int
    latent_start: int
    latent_stop: int
    frame_offset: int
    projection_start: int
    projection_stop: int
    projection_source_indices: tuple[int, ...]
    decoder_start: int
    decoder_stop: int
    decoder_source_indices: tuple[int, ...]
    merged_output_start: int
    merged_output_stop: int
    last: bool


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """Scalar timeline plan; call ranges are generated lazily by ``call``."""

    source_frames: int
    fps: float
    temporal_compression: int

    @property
    def repeat_factor(self) -> int:
        return TEMPORAL_DOWNSAMPLE // self.temporal_compression

    @property
    def effective_frames(self) -> int:
        return _target_frame_count(self.source_frames, self.temporal_compression)

    @property
    def expanded_frames(self) -> int:
        return 1 + (self.effective_frames - 1) * self.repeat_factor

    @property
    def requested_latents(self) -> int:
        return (self.expanded_frames - 1) // TEMPORAL_DOWNSAMPLE + 1

    @property
    def padded_latents(self) -> int:
        return (
            (self.requested_latents + STREAM_CHUNK_LATENTS - 1) // STREAM_CHUNK_LATENTS
        ) * STREAM_CHUNK_LATENTS

    @property
    def projection_frames(self) -> int:
        return 1 + self.padded_latents * TEMPORAL_DOWNSAMPLE

    @property
    def decoder_frames(self) -> int:
        return self.padded_latents * TEMPORAL_DOWNSAMPLE - (TEMPORAL_DOWNSAMPLE - 1)

    @property
    def merged_decoder_frames(self) -> int:
        return 1 + (self.decoder_frames - 1) // self.repeat_factor

    @property
    def output_frames(self) -> int:
        return self.source_frames

    @property
    def call_count(self) -> int:
        return self.padded_latents // STREAM_CHUNK_LATENTS

    def source_index(self, model_frame: int) -> int:
        """Return the original source index for one model/padded frame."""
        if model_frame >= self.projection_frames:
            raise ValueError("model_frame exceeds the planned projection timeline")
        return _source_index_for_model_frame(
            model_frame,
            source_frames=self.source_frames,
            expanded_frames=self.expanded_frames,
            repeat_factor=self.repeat_factor,
        )

    def call(self, index: int) -> StreamCall:
        """Derive one upstream call without materializing other calls."""
        if type(index) is not int or not 0 <= index < self.call_count:
            raise ValueError(f"call index must be in [0, {self.call_count})")
        latent_start = index * STREAM_CHUNK_LATENTS
        latent_stop = latent_start + STREAM_CHUNK_LATENTS
        projection_start = 0 if index == 0 else 1 + latent_start * TEMPORAL_DOWNSAMPLE
        projection_stop = 1 + latent_stop * TEMPORAL_DOWNSAMPLE
        decoder_start = max(0, latent_start * TEMPORAL_DOWNSAMPLE - (TEMPORAL_DOWNSAMPLE - 1))
        decoder_stop = latent_stop * TEMPORAL_DOWNSAMPLE - (TEMPORAL_DOWNSAMPLE - 1)

        first_merged_count = 1 + (
            STREAM_CHUNK_LATENTS * TEMPORAL_DOWNSAMPLE - TEMPORAL_DOWNSAMPLE
        ) // self.repeat_factor
        steady_merged_count = STREAM_CHUNK_LATENTS * TEMPORAL_DOWNSAMPLE // self.repeat_factor
        merged_start = 0 if index == 0 else first_merged_count + (index - 1) * steady_merged_count
        merged_stop = merged_start + (first_merged_count if index == 0 else steady_merged_count)

        return StreamCall(
            index=index,
            latent_start=latent_start,
            latent_stop=latent_stop,
            frame_offset=projection_start,
            projection_start=projection_start,
            projection_stop=projection_stop,
            projection_source_indices=tuple(
                self.source_index(frame) for frame in range(projection_start, projection_stop)
            ),
            decoder_start=decoder_start,
            decoder_stop=decoder_stop,
            decoder_source_indices=tuple(
                self.source_index(frame) for frame in range(decoder_start, decoder_stop)
            ),
            merged_output_start=merged_start,
            merged_output_stop=merged_stop,
            last=index + 1 == self.call_count,
        )

    def iter_calls(self):
        """Yield calls lazily, retaining at most one bounded call at a time."""
        for index in range(self.call_count):
            yield self.call(index)

    def trim_output(self, value, *, frame_axis: int = 2):
        """Hide synthetic decoder tail while preserving source FPS/count."""
        shape = getattr(value, "shape", None)
        if shape is None or not -len(shape) <= frame_axis < len(shape):
            raise ValueError("decoded value has no requested frame axis")
        axis = frame_axis % len(shape)
        if shape[axis] < self.source_frames:
            raise ValueError("decoded value is shorter than the source timeline")
        index = [slice(None)] * len(shape)
        index[axis] = slice(0, self.source_frames)
        return value[tuple(index)]


def build_stream_plan(
    source_frames: int,
    fps: float,
    *,
    temporal_compression: int = 4,
) -> StreamPlan:
    """Build the official fixed-four-latent chronology without model data."""
    _require_int(source_frames, "source_frames")
    if type(temporal_compression) is not int or temporal_compression not in (1, 2, 4):
        raise ValueError("temporal_compression must be one of 1, 2, or 4")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")

    return StreamPlan(
        source_frames=source_frames,
        fps=fps,
        temporal_compression=temporal_compression,
    )


__all__ = [
    "StreamCall",
    "StreamPlan",
    "STREAM_CHUNK_LATENTS",
    "TEMPORAL_DOWNSAMPLE",
    "build_stream_plan",
]
