#pragma once

#include "flash_geometry.hpp"

#include <cstddef>

namespace otc::ai {

// FlashVSR+ uses four latent frames per denoiser call and the source Wan
// MM-RoPE table has 22 temporal, 21 height, and 21 width complex pairs.
inline constexpr std::size_t kFlashRopeFrames = 4U;
inline constexpr std::size_t kFlashRopePairs = 64U;
inline constexpr std::size_t kFlashRopeTemporalPairs = 22U;
inline constexpr std::size_t kFlashRopeHeightPairs = 21U;
inline constexpr std::size_t kFlashRopeWidthPairs = 21U;
inline constexpr std::size_t kFlashRopeSpatialTableLength = 1024U;

// Fill caller-owned, contiguous F32 arrays with the source layout
// [1, 4*tokenHeight*tokenWidth, 1, 64].  The flat order is frame, row,
// column, then temporal/height/width pairs.  The implementation may allocate
// bounded per-call axis tables, but retains no video-length state; cos and sin
// must each contain the exact element count returned by flashRopeElements().
// Temporal positions are global: each call starts at call*4 and is evaluated
// with the source's F64 frequency/trigonometric formula. The implementation
// bounds call against flash::StreamPlan(kMaxSourceFrames), rather than using
// the source's old 1024-position cache as a whole-video limit; it never wraps
// or resets temporal positions.
std::size_t flashRopeElements(const FlashGeometry& geometry);

void fillFlashRopeInputs(const FlashGeometry& geometry, std::size_t call,
                         float* cos, std::size_t cosElements,
                         float* sin, std::size_t sinElements);

}  // namespace otc::ai
