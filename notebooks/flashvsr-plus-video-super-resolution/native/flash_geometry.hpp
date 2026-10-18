#pragma once

#include <cstddef>

namespace otc::ai {

// The Flash spatial graphs consume a 128-pixel grid.  A visible frame is
// represented by a working frame with only bottom/right edge padding: callers
// repeat the last visible row/column into the padded extent, never zero-pad or
// crop the source.  This class owns geometry metadata only; it does not own
// or copy image storage and it does not make fixed IR graphs dynamic.
struct FlashGeometry final {
  std::size_t visibleHeight;
  std::size_t visibleWidth;
  std::size_t workingHeight;
  std::size_t workingWidth;
  std::size_t tokenHeight;
  std::size_t tokenWidth;
  std::size_t latentHeight;
  std::size_t latentWidth;
  std::size_t spatialTokens;
  std::size_t spatialWindows;

  std::size_t bottomPadding() const noexcept {
    return workingHeight - visibleHeight;
  }

  std::size_t rightPadding() const noexcept {
    return workingWidth - visibleWidth;
  }

  // The decoder must trim this visible rectangle after working-frame output.
  // The top-left origin is unchanged by the padding policy.
  std::size_t trimmedHeight() const noexcept {
    return visibleHeight;
  }

  std::size_t trimmedWidth() const noexcept {
    return visibleWidth;
  }
};

// This mirrors src/ai-enhancement.ts: dimensions below the target long edge
// are scaled to a 3840-pixel long edge with independently rounded even axes;
// dimensions already at/above the target are preserved.  Odd source axes are
// accepted here because the app performs this policy before the strict visible
// Flash-frame contract.  The returned plan is deliberately separate from
// FlashGeometry because the preserved branch can exceed the reviewed Flash
// padded-UHD bounds (or remain odd) and must then be rejected by
// makeFlashGeometry (or handled by a future geometry-specific asset set).
struct FlashOutputGeometry final {
  std::size_t sourceHeight;
  std::size_t sourceWidth;
  std::size_t outputHeight;
  std::size_t outputWidth;
  bool spatial;
};

FlashGeometry makeFlashGeometry(std::size_t visibleHeight, std::size_t visibleWidth);
FlashOutputGeometry makeFlashOutputGeometry(std::size_t sourceHeight,
                                            std::size_t sourceWidth);

}  // namespace otc::ai
