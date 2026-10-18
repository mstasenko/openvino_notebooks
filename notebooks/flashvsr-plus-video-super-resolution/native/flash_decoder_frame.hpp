#pragma once

#include "flash_geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace otc::ai {

// These are the fixed, source-derived dimensions of the reviewed UHD decoder
// frame path.  They are deliberately constants rather than a configurable
// tiling API: the native session must match the 16-lane graph assets.
inline constexpr std::size_t kFlashDecoderFrameLanes = 16;
inline constexpr std::size_t kFlashDecoderFrameWorkingHeight = 2176;
inline constexpr std::size_t kFlashDecoderFrameWorkingWidth = 3840;
inline constexpr std::size_t kFlashDecoderFrameChannels = 3;
inline constexpr std::size_t kFlashDecoderFrameCompactRows = 544;
inline constexpr std::size_t kFlashDecoderFrameCompactWidth = 960;
inline constexpr std::size_t kFlashDecoderFrameTileCompactRows = 46;
inline constexpr std::size_t kFlashDecoderFrameTileRows = 184;
inline constexpr std::size_t kFlashDecoderFrameTileCoreRows = 136;

struct FlashDecoderFrameTile {
  // Rows in the [256,544,960] compact parent supplied to this lane.
  std::size_t compact_row_start;
  std::size_t compact_row_count;
  // Rows in the lane's raw [3,184,3840] frame output where the valid core
  // begins, and rows in the full working frame where that core begins.
  std::size_t local_frame_output_start;
  std::size_t global_frame_start;
  std::size_t core_rows;
};

// Return the source-derived descriptor for one lane.  The first and last
// lanes are clamped at the middle-raster boundary; interior lanes retain a
// 12-row middle-raster halo.
const FlashDecoderFrameTile& flashDecoderFrameTile(std::size_t lane);

// Normalize one startup or four steady RGB24 frames into planar BF16
// [1,3,T,H,W] for LQ projection. Shares the decoder conditioning's source
// operation RGB/127.5 - 1 and bottom/right edge padding; never uses [0,1].
// Only the first frame_count pointers are read. All extents and disjointness
// checks finish before the first output write.
void packFlashLqRgb(const std::array<const std::uint8_t*, 4>& rgb,
                    std::size_t frame_count, std::size_t rgb_frame_bytes,
                    const FlashGeometry& geometry, std::uint16_t* packed,
                    std::size_t packed_elements);

// Pack four borrowed, interleaved RGB24 visible frames and one temporal slice
// of a finite BF16 [1,16,4,H/8,W/8] latent chunk into [1,784,H/8,W/8].
// Geometry is canonical edge-padding from makeFlashGeometry; padded pixels
// repeat the last visible row/column. Repeated source pointers are permitted
// for startup/tail padding. The caller owns the one output buffer; no full
// pixel/latent copy is allocated here. Output may not overlap any input.
// All extents and latent words are checked before the first output write.
void packFlashDecoderParentRgb(const std::array<const std::uint8_t*, 4>& rgb,
                               std::size_t rgb_frame_bytes,
                               const FlashGeometry& geometry,
                               const std::uint16_t* latent_chunk,
                               std::size_t latent_elements,
                               std::size_t inner,
                               std::uint16_t* packed,
                               std::size_t packed_elements);

// Validate one raw BF16 [3,184,3840] tile, normalize its valid core with the
// source operation BF16-RNE(2*x - 1), and scatter only the top-left visible
// crop into one caller-owned planar [3, visible_height, visible_width] F32
// frame. Input and output must be disjoint. The complete raw tile is validated
// before the first destination write.
void normalizeAndScatterFlashDecoderTile(const std::uint16_t* raw_bf16,
                                          std::size_t raw_elements,
                                          std::size_t lane,
                                          std::size_t visible_height,
                                          std::size_t visible_width,
                                          float* destination,
                                          std::size_t destination_elements);

}  // namespace otc::ai
