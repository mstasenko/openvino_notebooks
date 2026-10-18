#include "flash_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace otc::ai {
namespace {

constexpr std::size_t kGrid = 128U;
constexpr std::size_t kTokenStride = 16U;
constexpr std::size_t kLatentStride = 8U;
constexpr std::size_t kWindowTokens = 64U;
constexpr std::size_t kMaxWorkingSide = 3840U;
constexpr std::size_t kMaxWorkingArea = 2176U * 3840U;
constexpr std::size_t kMaxInputSide = 100'000U;
constexpr std::size_t kOutputLongEdge = 3840U;

[[noreturn]] void reject(const char* message) {
  throw std::invalid_argument(std::string("FlashVSR+ geometry: ") + message);
}

void require(bool condition, const char* message) {
  if (!condition) reject(message);
}

std::size_t product(std::size_t left, std::size_t right, const char* message) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
    reject(message);
  return left * right;
}

void validateCanvasDimension(std::size_t value, const char* label) {
  if (value < 2U || value > kMaxInputSide)
    throw std::invalid_argument(std::string("FlashVSR+ geometry: ") + label +
                                " must be an integer in [2,100000]");
}

void validateInputDimension(std::size_t value, const char* label) {
  validateCanvasDimension(value, label);
  if (value % 2U != 0U)
    throw std::invalid_argument(std::string("FlashVSR+ geometry: ") + label +
                                " must be even for a visible Flash frame");
}

std::size_t roundVisibleDimension(std::size_t value) {
  // The bound is checked before adding the grid remainder, so this cannot
  // overflow even on a 32-bit size_t implementation.
  require(value <= kMaxWorkingSide, "visible dimension exceeds 3840");
  return ((value + kGrid - 1U) / kGrid) * kGrid;
}

std::size_t scaleToOutput(std::size_t value, std::size_t longSide) {
  // Match the app's floating-point operation order, not exact rational
  // rounding: at odd-axis half ties (e.g. 11x2816) those differ by two pixels.
  const double scale = static_cast<double>(kOutputLongEdge) / longSide;
  return std::max<std::size_t>(2U,
      static_cast<std::size_t>(std::floor(value * scale / 2.0 + 0.5)) * 2U);
}

}  // namespace

FlashGeometry makeFlashGeometry(std::size_t visibleHeight, std::size_t visibleWidth) {
  validateInputDimension(visibleHeight, "visible height");
  validateInputDimension(visibleWidth, "visible width");
  const auto workingHeight = roundVisibleDimension(visibleHeight);
  const auto workingWidth = roundVisibleDimension(visibleWidth);
  require(workingHeight <= kMaxWorkingSide && workingWidth <= kMaxWorkingSide,
          "working dimension exceeds 3840");
  require(product(workingHeight, workingWidth, "working area overflows") <= kMaxWorkingArea,
          "working area exceeds padded-UHD bound");

  const auto tokenHeight = workingHeight / kTokenStride;
  const auto tokenWidth = workingWidth / kTokenStride;
  const auto latentHeight = workingHeight / kLatentStride;
  const auto latentWidth = workingWidth / kLatentStride;
  const auto spatialTokens = product(tokenHeight, tokenWidth, "token shape overflows");
  require(spatialTokens != 0U && spatialTokens % kWindowTokens == 0U,
          "token shape is not a whole 64-token window grid");

  return {visibleHeight, visibleWidth, workingHeight, workingWidth,
          tokenHeight, tokenWidth, latentHeight, latentWidth,
          spatialTokens, spatialTokens / kWindowTokens};
}

FlashOutputGeometry makeFlashOutputGeometry(std::size_t sourceHeight,
                                            std::size_t sourceWidth) {
  // The app's canvas validator permits odd source dimensions.  Preserve that
  // input policy: the scaled branch rounds each output axis to even, while a
  // preserved >=3840 branch is later rejected by makeFlashGeometry if it is
  // not a valid visible Flash frame.
  validateCanvasDimension(sourceHeight, "source height");
  validateCanvasDimension(sourceWidth, "source width");
  const auto longSide = sourceHeight > sourceWidth ? sourceHeight : sourceWidth;
  if (longSide >= kOutputLongEdge)
    return {sourceHeight, sourceWidth, sourceHeight, sourceWidth, false};

  return {sourceHeight, sourceWidth, scaleToOutput(sourceHeight, longSide),
          scaleToOutput(sourceWidth, longSide), true};
}

}  // namespace otc::ai
