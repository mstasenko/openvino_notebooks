#include "ai_enhance_core.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

template <typename F>
void expectVaeFailure(F action) {
  bool rejected = false;
  try { action(); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
}

void testVaeOverlap() {
  using namespace otc::ai;
  const auto landscape = makeVaeOverlapPlan(480, 270);
  assert(landscape.tiles.size() == 45U && landscape.tileSize == 64 && landscape.overlap == 8);
  assert(landscape.tiles.front().x == 0 && landscape.tiles.front().y == 0);
  assert(landscape.tiles[1].x == 56 && landscape.tiles[1].y == 0);
  assert(landscape.tiles[9].x == 0 && landscape.tiles[9].y == 56);
  const auto& last = landscape.tiles.back();
  assert(last.x == 448 && last.y == 224 && last.width == 32 && last.height == 46);
  VaeSpatialAccumulator fullLatent(landscape, 2U, 1);
  for (const auto& tile : landscape.tiles) {
    const auto spatial = static_cast<std::size_t>(tile.width) * tile.height;
    std::vector<float> values(spatial * 2U, 1.0F);
    std::fill(values.begin() + spatial, values.end(), -2.0F);
    fullLatent.add(values);
  }
  const auto fullOutput = fullLatent.finish();
  const auto fullSpatial = static_cast<std::size_t>(landscape.width) * landscape.height;
  assert(fullOutput.size() == 2U * fullSpatial);
  for (std::size_t pixel = 0; pixel < fullSpatial; ++pixel) {
    assert(std::abs(fullOutput[pixel] - 1.0F) < 0.000001F);
    assert(std::abs(fullOutput[fullSpatial + pixel] + 2.0F) < 0.000001F);
  }
  const auto portrait = makeVaeOverlapPlan(270, 480);
  assert(portrait.tiles.size() == 45U);
  for (const auto& tile : landscape.tiles) {
    assert(std::any_of(portrait.tiles.begin(), portrait.tiles.end(), [&](const VaeOverlapTile& t) {
      return t.x == tile.y && t.y == tile.x && t.width == tile.height && t.height == tile.width;
    }));
  }
  assert(makeVaeOverlapPlan(64, 64).tiles.size() == 1U);
  const auto skippedTail = makeVaeOverlapPlan(120, 64);
  assert(skippedTail.tiles.size() == 2U && skippedTail.tiles.back().x == 56);
  const auto retainedTail = makeVaeOverlapPlan(121, 64);
  assert(retainedTail.tiles.size() == 3U && retainedTail.tiles.back().width == 9);
  expectVaeFailure([] { (void)makeVaeOverlapPlan(0, 1); });
  expectVaeFailure([] { (void)makeVaeOverlapPlan(std::numeric_limits<int>::max(), 1); });
  expectVaeFailure([] { (void)makeVaeOverlapPlan(1, 1, 512, 512); });
  expectVaeFailure([] { (void)makeVaeOverlapPlan(1, 1, 511, 64); });
  expectVaeFailure([&] { VaeSpatialAccumulator a(landscape, 0U, 1); });
  expectVaeFailure([&] { VaeSpatialAccumulator a(landscape, 1U, 2); });
  expectVaeFailure([&] { VaeSpatialAccumulator a(landscape, std::numeric_limits<std::size_t>::max(), 8); });
  auto malformed = landscape;
  std::swap(malformed.tiles[0], malformed.tiles[1]);
  expectVaeFailure([&] { VaeSpatialAccumulator a(malformed, 1U, 1); });

  // Multiple planes represent channels and times independently. Small tiles
  // exercise the same plan/ramp mapping at encoder scale 1 and decoder scale 8.
  const auto small = makeVaeOverlapPlan(15, 13, 64, 16);
  for (const int scale : {1, 8}) {
    VaeSpatialAccumulator accumulator(small, 6U, scale);
    expectVaeFailure([&] { (void)accumulator.finish(); });
    expectVaeFailure([&] { accumulator.add({}); });
    for (const auto& tile : small.tiles) {
      const auto spatial = static_cast<std::size_t>(tile.width * scale) * (tile.height * scale);
      std::vector<float> values(spatial * 6U);
      for (std::size_t plane = 0; plane < 6U; ++plane) {
        std::fill_n(values.begin() + plane * spatial, spatial, static_cast<float>(plane + 1U));
      }
      accumulator.add(values);
    }
    const auto output = accumulator.finish();
    const auto spatial = static_cast<std::size_t>(small.width * scale) * (small.height * scale);
    assert(output.size() == spatial * 6U);
    for (std::size_t plane = 0; plane < 6U; ++plane) {
      for (std::size_t pixel = 0; pixel < spatial; ++pixel) {
        assert(std::abs(output[plane * spatial + pixel] - static_cast<float>(plane + 1U)) < 0.00001F);
      }
    }
    expectVaeFailure([&] { (void)accumulator.finish(); });
    expectVaeFailure([&] { accumulator.add({}); });
  }

  // Four constant but distinct tiles provide an independent overlap oracle:
  // with a two-sample ramp, the overlap corners select specific contributors.
  const auto cornerPlan = makeVaeOverlapPlan(6, 6, 32, 16);
  assert(cornerPlan.tiles.size() == 4U);
  VaeSpatialAccumulator corners(cornerPlan, 1U, 1);
  for (std::size_t i = 0; i < cornerPlan.tiles.size(); ++i) {
    corners.add(std::vector<float>(16U, static_cast<float>(i + 1U)));
  }
  const auto blended = corners.finish();
  assert(blended[0] == 1.0F && blended[5] == 2.0F);
  assert(blended[30] == 3.0F && blended[35] == 4.0F);
  assert(blended[2U * 6U + 2U] == 1.0F);
  assert(blended[2U * 6U + 3U] == 2.0F);
  assert(blended[3U * 6U + 2U] == 3.0F);
  assert(blended[3U * 6U + 3U] == 4.0F);
  // The interior of a four-sample raised cosine has weights .25 and .75.
  const auto cosinePlan = makeVaeOverlapPlan(12, 8, 64, 32);
  VaeSpatialAccumulator cosine(cosinePlan, 1U, 1);
  cosine.add(std::vector<float>(64U, 0.0F));
  cosine.add(std::vector<float>(64U, 1.0F));
  const auto transition = cosine.finish();
  assert(std::abs(transition[5] - 0.25F) < 0.000001F);
  assert(std::abs(transition[6] - 0.75F) < 0.000001F);

  VaeSpatialAccumulator nonfinite(makeVaeOverlapPlan(1, 1), 1U, 1);
  expectVaeFailure([&] { nonfinite.add({std::numeric_limits<float>::infinity()}); });
  nonfinite.add({2.0F});
  assert(nonfinite.finish()[0] == 2.0F);
  // Heavy overlaps can overflow a finite FP32 sum. Failure must poison
  // the accumulator so a partially modified result cannot be published.
  const auto densePlan = makeVaeOverlapPlan(10, 1, 48, 32);
  VaeSpatialAccumulator overflow(densePlan, 1U, 1);
  overflow.add(std::vector<float>(6U, std::numeric_limits<float>::max()));
  expectVaeFailure([&] { overflow.add(std::vector<float>(6U, std::numeric_limits<float>::max())); });
  expectVaeFailure([&] { (void)overflow.finish(); });
}

void assertWindowMetadataContract(const otc::ai::SeedVrWindowMetadata& metadata) {
  const std::size_t expectedWindowIndices = metadata.windowCount * metadata.maxVideoLen;
  const std::size_t expectedMask = metadata.windowCount * metadata.attentionLen;
  const std::size_t expectedRope =
      expectedMask * otc::ai::kSeedVrRopeFrequencyWidth;
  assert(metadata.sourceTokenCount ==
         static_cast<std::size_t>(metadata.frames) * static_cast<std::size_t>(metadata.height) *
             static_cast<std::size_t>(metadata.width));
  assert(metadata.windowLengths.size() == metadata.windowCount);
  assert(metadata.windowIdx.size() == expectedWindowIndices);
  assert(metadata.windowKeyMask.size() == expectedMask);
  assert(metadata.windowReverseIdx.size() == metadata.sourceTokenCount);
  assert(metadata.windowRopeFreqs.size() == expectedRope);

  std::vector<std::size_t> occurrences(metadata.sourceTokenCount, 0U);
  std::vector<std::uint8_t> reversePositions(expectedWindowIndices, 0U);
  const auto rope = [&](std::size_t window, std::size_t position, std::size_t component) {
    return metadata.windowRopeFreqs[
        (window * metadata.attentionLen + position) * otc::ai::kSeedVrRopeFrequencyWidth +
        component];
  };
  for (std::size_t window = 0U; window < metadata.windowCount; ++window) {
    std::size_t active = 0U;
    for (std::size_t position = 0U; position < metadata.maxVideoLen; ++position) {
      const bool isActive = metadata.windowKeyMask[window * metadata.attentionLen + position] != 0U;
      if (isActive) {
        ++active;
        const std::int64_t source = metadata.windowIdx[window * metadata.maxVideoLen + position];
        assert(source >= 0 && static_cast<std::size_t>(source) < metadata.sourceTokenCount);
        ++occurrences[static_cast<std::size_t>(source)];
        assert(metadata.windowReverseIdx[static_cast<std::size_t>(source)] ==
               static_cast<std::int64_t>(window * metadata.maxVideoLen + position));
      } else {
        assert(metadata.windowIdx[window * metadata.maxVideoLen + position] == 0);
        for (std::size_t component = 0U; component < otc::ai::kSeedVrRopeFrequencyWidth;
             ++component) {
          assert(rope(window, position, component) == 0.0F);
        }
      }
    }
    assert(active == metadata.windowLengths[window]);
    for (std::size_t text = 0U; text < otc::ai::kSeedVrTextTokens; ++text) {
      assert(metadata.windowKeyMask[window * metadata.attentionLen + metadata.maxVideoLen + text] ==
             1U);
    }
  }
  for (const std::size_t count : occurrences) assert(count == 1U);
  std::size_t reversePositionCount = 0U;
  for (const std::int64_t reverse : metadata.windowReverseIdx) {
    assert(reverse >= 0 && static_cast<std::size_t>(reverse) < expectedWindowIndices);
    auto& seen = reversePositions[static_cast<std::size_t>(reverse)];
    assert(seen == 0U);
    seen = 1U;
    ++reversePositionCount;
  }
  assert(reversePositionCount == metadata.sourceTokenCount);
  for (const std::uint8_t seen : reversePositions) assert(seen <= 1U);

  // The first window starts with local (t,h,w)=(0,0,0).  Upstream MM-RoPE3D
  // starts video time at the 58 text-token offset and repeats text frequencies
  // across the three 42-value axis groups.
  assert(metadata.windowRopeFreqs.size() / otc::ai::kSeedVrRopeFrequencyWidth ==
         metadata.windowCount * metadata.attentionLen);
  assert(std::abs(rope(0U, 0U, 0U) - 58.0F) < 0.000001F);
  assert(std::abs(rope(0U, 0U, 1U) - 58.0F) < 0.000001F);
  assert(std::abs(rope(0U, 0U, 2U) - 37.40690994262695F) < 0.000001F);
  assert(rope(0U, 0U, 42U) == 0.0F);
  assert(rope(0U, 0U, 84U) == 0.0F);
  const std::size_t textOne = metadata.maxVideoLen + 1U;
  for (std::size_t component = 0U; component < 42U; ++component) {
    assert(rope(0U, textOne, component) == rope(0U, textOne, 42U + component));
    assert(rope(0U, textOne, component) == rope(0U, textOne, 84U + component));
  }
  assert(std::abs(rope(0U, textOne, 0U) - 1.0F) < 0.000001F);
}

}  // namespace

int main() {
  testVaeOverlap();
  using otc::ai::makeSeedVrWindows;
  using otc::ai::makeSeedVrWindowMetadata;
  using otc::ai::makeSrTiles;
  using otc::ai::outputFrameCount;

  assert(outputFrameCount(1.0, 60.0) == 60U);
  assert(outputFrameCount(1.0, 30.0) == 30U);
  assert(outputFrameCount(0.1, 30.0) == 3U);
  assert(outputFrameCount(0.1, 60.0) == 6U);
  assert(outputFrameCount(0.3, 30.0) == 9U);
  assert(outputFrameCount(0.1, 29.97) == 3U);
  assert(outputFrameCount(0.1, 59.94) == 6U);
  assert(outputFrameCount(1.0 / 60.0, 60.0) == 1U);
  assert(outputFrameCount(0.1001, 30.0) == 4U);

  const int width = 513;
  const int height = 300;
  const auto tiles = makeSrTiles(width, height);
  assert(tiles.size() == 6U);
  std::vector<std::uint8_t> covered(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0U);
  for (const auto& tile : tiles) {
    assert(tile.coreX >= 0 && tile.coreY >= 0);
    assert(tile.coreX + tile.coreWidth <= width);
    assert(tile.coreY + tile.coreHeight <= height);
    assert(tile.inputX >= 0 && tile.inputY >= 0);
    assert(tile.inputX + tile.inputWidth <= width);
    assert(tile.inputY + tile.inputHeight <= height);
    assert(tile.inputWidth <= 320 && tile.inputHeight <= 320);
    for (int y = tile.coreY; y < tile.coreY + tile.coreHeight; ++y) {
      for (int x = tile.coreX; x < tile.coreX + tile.coreWidth; ++x) {
        auto& cell = covered[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                             static_cast<std::size_t>(x)];
        assert(cell == 0U);
        cell = 1U;
      }
    }
  }
  for (const auto cell : covered) assert(cell == 1U);

  const auto landscapeWindows = makeSeedVrWindows(1, 48, 60, false);
  assert(landscapeWindows.size() == 9U);
  assert(landscapeWindows.front().frame == 0 && landscapeWindows.front().frameCount == 1);
  assert(landscapeWindows.front().y == 0 && landscapeWindows.front().height == 18);
  assert(landscapeWindows.front().x == 0 && landscapeWindows.front().width == 23);
  assert(landscapeWindows.back().y == 36 && landscapeWindows.back().height == 12);
  assert(landscapeWindows.back().x == 46 && landscapeWindows.back().width == 14);
  std::size_t landscapeTokens = 0U;
  for (const auto& window : landscapeWindows) {
    assert(window.frame == 0 && window.frameCount == 1);
    landscapeTokens += static_cast<std::size_t>(window.frameCount) *
                       static_cast<std::size_t>(window.height) *
                       static_cast<std::size_t>(window.width);
  }
  assert(landscapeTokens == 48U * 60U);

  const auto temporalWindows = makeSeedVrWindows(2, 48, 60, false);
  assert(temporalWindows.size() == 18U);
  std::size_t temporalTokens = 0U;
  for (const auto& window : temporalWindows) {
    temporalTokens += static_cast<std::size_t>(window.frameCount) *
                      static_cast<std::size_t>(window.height) *
                      static_cast<std::size_t>(window.width);
  }
  assert(temporalTokens == 2U * 48U * 60U);

  const auto shiftedWindows = makeSeedVrWindows(2, 48, 60, true);
  assert(shiftedWindows.size() == 32U);
  assert(shiftedWindows.front().frame == 0 && shiftedWindows.front().frameCount == 1);
  assert(shiftedWindows.front().y == 0 && shiftedWindows.front().height == 9);
  assert(shiftedWindows.front().x == 0 && shiftedWindows.front().width == 11);
  assert(shiftedWindows.back().frame == 1 && shiftedWindows.back().frameCount == 1);
  assert(shiftedWindows.back().y == 45 && shiftedWindows.back().height == 3);
  assert(shiftedWindows.back().x == 57 && shiftedWindows.back().width == 3);
  std::size_t shiftedTokens = 0U;
  for (const auto& window : shiftedWindows) {
    shiftedTokens += static_cast<std::size_t>(window.frameCount) *
                     static_cast<std::size_t>(window.height) *
                     static_cast<std::size_t>(window.width);
  }
  assert(shiftedTokens == 2U * 48U * 60U);

  const auto square4kWindows = makeSeedVrWindows(2, 240, 240, false);
  const auto shiftedSquare4kWindows = makeSeedVrWindows(2, 240, 240, true);
  assert(square4kWindows.size() == 288U);
  assert(shiftedSquare4kWindows.size() == 338U);
  const auto thin4kWindows = makeSeedVrWindows(1, 3, 240, false);
  const auto largestThinWindow = std::max_element(
      thin4kWindows.begin(), thin4kWindows.end(), [](const auto& left, const auto& right) {
        return left.frameCount * left.height * left.width <
               right.frameCount * right.height * right.width;
      });
  assert(largestThinWindow != thin4kWindows.end());
  assert(largestThinWindow->frameCount * largestThinWindow->height * largestThinWindow->width ==
         537);

  bool rejectedInvalidWindowShape = false;
  try {
    (void)makeSeedVrWindows(0, 48, 60, false);
  } catch (const std::invalid_argument&) {
    rejectedInvalidWindowShape = true;
  }
  assert(rejectedInvalidWindowShape);

  const auto oneTokenMetadata = makeSeedVrWindowMetadata(1, 1, 1, false);
  assert(oneTokenMetadata.windowCount == 1U);
  assert(oneTokenMetadata.maxVideoLen == 1U);
  assert(oneTokenMetadata.attentionLen == otc::ai::kSeedVrTextTokens + 1U);
  assertWindowMetadataContract(oneTokenMetadata);

  const auto unshiftedMetadata = makeSeedVrWindowMetadata(1, 48, 60, false);
  assert(unshiftedMetadata.windowCount == 9U);
  assert(unshiftedMetadata.maxVideoLen == 414U);
  assertWindowMetadataContract(unshiftedMetadata);

  const auto shiftedMetadata = makeSeedVrWindowMetadata(2, 48, 60, true);
  assert(shiftedMetadata.windowCount == 32U);
  assert(shiftedMetadata.maxVideoLen == 414U);
  assertWindowMetadataContract(shiftedMetadata);
  const auto unshiftedTemporalMetadata = makeSeedVrWindowMetadata(2, 48, 60, false);
  assert(unshiftedTemporalMetadata.windowCount == 18U);
  assert(unshiftedTemporalMetadata.maxVideoLen == 414U);
  assert(unshiftedTemporalMetadata.windowIdx != shiftedMetadata.windowIdx);
  assert(unshiftedTemporalMetadata.windowReverseIdx != shiftedMetadata.windowReverseIdx);

  {
    constexpr std::size_t innerDimension = 2U;
    constexpr std::size_t chunkSize = 7U;
    std::vector<float> gathered(unshiftedTemporalMetadata.sourceTokenCount * innerDimension, 0.0F);
    std::vector<std::uint8_t> coverage(unshiftedTemporalMetadata.sourceTokenCount, 0U);
    std::size_t totalGathered = 0U;
    for (std::size_t start = 0U; start < unshiftedTemporalMetadata.windowCount; start += chunkSize) {
      const std::size_t end = std::min(unshiftedTemporalMetadata.windowCount, start + chunkSize);
      std::vector<float> chunk((end - start) * unshiftedTemporalMetadata.maxVideoLen * innerDimension,
                               -1.0F);
      for (std::size_t row = 0U; row < end - start; ++row) {
        const std::size_t window = start + row;
        for (std::size_t position = 0U; position < unshiftedTemporalMetadata.maxVideoLen; ++position) {
          const std::size_t metadataPosition = window * unshiftedTemporalMetadata.maxVideoLen + position;
          if (unshiftedTemporalMetadata.windowKeyMask[
                  window * unshiftedTemporalMetadata.attentionLen + position] == 0U) {
            continue;
          }
          const std::size_t source = static_cast<std::size_t>(
              unshiftedTemporalMetadata.windowIdx[metadataPosition]);
          const std::size_t output = (row * unshiftedTemporalMetadata.maxVideoLen + position) *
                                     innerDimension;
          chunk[output] = static_cast<float>(source);
          chunk[output + 1U] = -static_cast<float>(source);
        }
      }
      totalGathered += otc::ai::gatherSeedVrChunk(
          unshiftedTemporalMetadata, start, end, innerDimension, chunk, gathered, coverage);
    }
    assert(totalGathered == unshiftedTemporalMetadata.sourceTokenCount);
    for (std::size_t source = 0U; source < unshiftedTemporalMetadata.sourceTokenCount; ++source) {
      assert(coverage[source] == 1U);
      assert(gathered[source * innerDimension] == static_cast<float>(source));
      assert(gathered[source * innerDimension + 1U] == -static_cast<float>(source));
    }

    bool rejectedDuplicateGather = false;
    try {
      std::vector<float> duplicateChunk(unshiftedTemporalMetadata.maxVideoLen * innerDimension, 0.0F);
      (void)otc::ai::gatherSeedVrChunk(unshiftedTemporalMetadata, 0U, 1U, innerDimension,
                                       duplicateChunk, gathered, coverage);
    } catch (const std::logic_error&) {
      rejectedDuplicateGather = true;
    }
    assert(rejectedDuplicateGather);
  }

  // Production even block 16 remains unshifted; odd block 31 is shifted.
  // Keep the production [T,H,W]=[2,135,240] caps explicit in the native test.
  {
    const auto productionBlock0 = makeSeedVrWindowMetadata(2, 135, 240, false);
    assert(productionBlock0.windowCount == 162U);
    assert(productionBlock0.maxVideoLen == 405U);
    assertWindowMetadataContract(productionBlock0);
  }
  {
    const auto productionBlock16 = makeSeedVrWindowMetadata(2, 135, 240, false);
    assert(!productionBlock16.shifted);
    assert(productionBlock16.windowCount == 162U);
    assert(productionBlock16.maxVideoLen == 405U);
    assertWindowMetadataContract(productionBlock16);
  }
  {
    const auto productionBlock31 = makeSeedVrWindowMetadata(2, 135, 240, true);
    assert(productionBlock31.shifted);
    assert(productionBlock31.windowCount == 200U);
    assert(productionBlock31.maxVideoLen == 405U);
    assertWindowMetadataContract(productionBlock31);
  }

  bool rejectedUnrepresentableRopeShape = false;
  try {
    (void)makeSeedVrWindowMetadata(1, 1, 100'000, false);
  } catch (const std::invalid_argument&) {
    rejectedUnrepresentableRopeShape = true;
  }
  assert(rejectedUnrepresentableRopeShape);

  std::vector<std::uint8_t> still(4U * 4U * 3U, 0U);
  std::vector<std::uint8_t> moving = still;
  moving[0] = 255U;
  assert(!otc::ai::exactRgbEqual(4, 4, still, moving));
  assert(otc::ai::exactRgbEqual(4, 4, still, still));

  constexpr std::size_t tensorWidth = 8U;
  constexpr std::size_t tensorHeight = 8U;
  constexpr int actualWidth = 4;
  constexpr int actualHeight = 3;
  std::vector<float> planar(3U * tensorWidth * tensorHeight, 0.0F);
  for (std::size_t channel = 0U; channel < 3U; ++channel) {
    for (std::size_t y = 0U; y < tensorHeight; ++y) {
      for (std::size_t x = 0U; x < tensorWidth; ++x) {
        planar[channel * tensorWidth * tensorHeight + y * tensorWidth + x] =
            static_cast<float>(channel * 1000U + y * 100U + x);
      }
    }
  }
  const auto interior = otc::ai::samplePlanarRgb(planar.data(), tensorHeight, tensorWidth,
                                                  actualWidth, actualHeight, 1.5, 1.5);
  assert(std::abs(interior[0] - 151.5F) < 0.001F);
  assert(std::abs(interior[1] - 1151.5F) < 0.001F);
  assert(std::abs(interior[2] - 2151.5F) < 0.001F);
  const auto clipped = otc::ai::samplePlanarRgb(planar.data(), tensorHeight, tensorWidth,
                                                 actualWidth, actualHeight, 99.0, 99.0);
  assert(std::abs(clipped[0] - 203.0F) < 0.001F);
  assert(std::abs(clipped[1] - 1203.0F) < 0.001F);
  assert(std::abs(clipped[2] - 2203.0F) < 0.001F);
  std::cout << "seedvr2-openvino-core-tests: ok\n";
  return 0;
}
