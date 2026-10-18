#include "flash_rope.hpp"

#include "flash_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace otc::ai {
namespace {

constexpr std::size_t kAxisTemporalDim = 44U;
constexpr std::size_t kAxisSpatialDim = 42U;
constexpr std::size_t kTheta = 10000U;

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

std::size_t checkedMultiply(std::size_t left, std::size_t right,
                            const char* message) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::invalid_argument(message);
  return left * right;
}

std::size_t checkedAdd(std::size_t left, std::size_t right,
                       const char* message) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::invalid_argument(message);
  return left + right;
}

void requireCanonicalGeometry(const FlashGeometry& geometry) {
  const auto canonical = makeFlashGeometry(geometry.visibleHeight, geometry.visibleWidth);
  require(geometry.workingHeight == canonical.workingHeight &&
              geometry.workingWidth == canonical.workingWidth &&
              geometry.tokenHeight == canonical.tokenHeight &&
              geometry.tokenWidth == canonical.tokenWidth &&
              geometry.latentHeight == canonical.latentHeight &&
              geometry.latentWidth == canonical.latentWidth &&
              geometry.spatialTokens == canonical.spatialTokens &&
              geometry.spatialWindows == canonical.spatialWindows,
          "Flash RoPE geometry is not canonical");
  require(geometry.tokenHeight <= kFlashRopeSpatialTableLength &&
              geometry.tokenWidth <= kFlashRopeSpatialTableLength,
          "Flash RoPE spatial position exceeds the source table");
}

void requireDisjoint(const float* first, std::size_t firstElements,
                     const float* second, std::size_t secondElements) {
  const auto firstAddress = reinterpret_cast<std::uintptr_t>(first);
  const auto secondAddress = reinterpret_cast<std::uintptr_t>(second);
  const auto firstBytes = checkedMultiply(firstElements, sizeof(float),
                                          "Flash RoPE output extent overflows");
  const auto secondBytes = checkedMultiply(secondElements, sizeof(float),
                                           "Flash RoPE output extent overflows");
  const auto maximum = std::numeric_limits<std::uintptr_t>::max();
  require(firstAddress <= maximum - firstBytes && secondAddress <= maximum - secondBytes,
          "Flash RoPE output address extent overflows");
  require(firstAddress + firstBytes <= secondAddress ||
              secondAddress + secondBytes <= firstAddress,
          "Flash RoPE cos and sin storage overlaps");
}

double sourceFrequency(std::size_t pair, std::size_t axisDimension) {
  // Match precompute_freqs_cis: torch computes the exponent and power in
  // float64, then performs the reciprocal before forming the outer product.
  const double exponent = static_cast<double>(2U * pair) /
                          static_cast<double>(axisDimension);
  return 1.0 / std::pow(static_cast<double>(kTheta), exponent);
}

struct AxisTable final {
  std::vector<float> cos;
  std::vector<float> sin;
};

AxisTable makeAxisTable(std::size_t firstPosition, std::size_t positions,
                        std::size_t pairCount, std::size_t axisDimension) {
  const auto elements = checkedMultiply(positions, pairCount,
                                        "Flash RoPE axis table overflows");
  AxisTable table{std::vector<float>(elements), std::vector<float>(elements)};
  for (std::size_t position = 0U; position < positions; ++position) {
    const auto base = checkedMultiply(position, pairCount,
                                      "Flash RoPE axis row overflows");
    const auto absolutePosition = checkedAdd(
        firstPosition, position, "Flash RoPE position overflows");
    for (std::size_t pair = 0U; pair < pairCount; ++pair) {
      const double phase = static_cast<double>(absolutePosition) *
                           sourceFrequency(pair, axisDimension);
      // The source's torch.polar(..., angle) emits real then imaginary parts;
      // preserve the same double-precision phase/trig operations before the
      // individual F32 stores.
      table.cos[base + pair] = static_cast<float>(std::cos(phase));
      table.sin[base + pair] = static_cast<float>(std::sin(phase));
    }
  }
  return table;
}

}  // namespace

std::size_t flashRopeElements(const FlashGeometry& geometry) {
  requireCanonicalGeometry(geometry);
  const auto tokens = checkedMultiply(
      checkedMultiply(kFlashRopeFrames, geometry.spatialTokens,
                      "Flash RoPE token count overflows"),
      kFlashRopePairs, "Flash RoPE element count overflows");
  return tokens;
}

void fillFlashRopeInputs(const FlashGeometry& geometry, std::size_t call,
                         float* cos, std::size_t cosElements,
                         float* sin, std::size_t sinElements) {
  const auto elements = flashRopeElements(geometry);
  // StreamPlan is the single source of truth for the admitted whole-video
  // timeline. Only four temporal rows are materialized below, so extending
  // the position does not create a video-length RoPE allocation.
  static const auto maximumCall =
      flash::StreamPlan(flash::kMaxSourceFrames).callCount() - 1U;
  require(call <= maximumCall,
          "Flash RoPE call index exceeds the admitted source timeline");
  require(cos != nullptr && sin != nullptr && cosElements == elements &&
              sinElements == elements,
          "Flash RoPE caller storage has the wrong shape or extent");
  requireDisjoint(cos, cosElements, sin, sinElements);

  const auto temporalOffset = checkedMultiply(
      call, kFlashRopeFrames, "Flash RoPE temporal position overflows");
  const auto temporal = makeAxisTable(temporalOffset, kFlashRopeFrames,
                                      kFlashRopeTemporalPairs, kAxisTemporalDim);
  const auto height = makeAxisTable(0U, geometry.tokenHeight,
                                    kFlashRopeHeightPairs, kAxisSpatialDim);
  const auto width = makeAxisTable(0U, geometry.tokenWidth,
                                   kFlashRopeWidthPairs, kAxisSpatialDim);
  const auto rowTokens = checkedMultiply(geometry.tokenHeight, geometry.tokenWidth,
                                         "Flash RoPE spatial token count overflows");
  std::size_t token = 0U;
  for (std::size_t frame = 0U; frame < kFlashRopeFrames; ++frame) {
    for (std::size_t row = 0U; row < geometry.tokenHeight; ++row) {
      for (std::size_t column = 0U; column < geometry.tokenWidth; ++column, ++token) {
        const auto base = checkedMultiply(token, kFlashRopePairs,
                                          "Flash RoPE row offset overflows");
        const auto temporalBase = frame * kFlashRopeTemporalPairs;
        const auto heightBase = row * kFlashRopeHeightPairs;
        const auto widthBase = column * kFlashRopeWidthPairs;
        std::copy_n(temporal.cos.data() + temporalBase, kFlashRopeTemporalPairs,
                    cos + base);
        std::copy_n(height.cos.data() + heightBase, kFlashRopeHeightPairs,
                    cos + base + kFlashRopeTemporalPairs);
        std::copy_n(width.cos.data() + widthBase, kFlashRopeWidthPairs,
                    cos + base + kFlashRopeTemporalPairs + kFlashRopeHeightPairs);
        std::copy_n(temporal.sin.data() + temporalBase, kFlashRopeTemporalPairs,
                    sin + base);
        std::copy_n(height.sin.data() + heightBase, kFlashRopeHeightPairs,
                    sin + base + kFlashRopeTemporalPairs);
        std::copy_n(width.sin.data() + widthBase, kFlashRopeWidthPairs,
                    sin + base + kFlashRopeTemporalPairs + kFlashRopeHeightPairs);
      }
    }
  }
  require(token == checkedMultiply(kFlashRopeFrames, rowTokens,
                                  "Flash RoPE token count changed"),
          "Flash RoPE token traversal changed");
}

}  // namespace otc::ai
