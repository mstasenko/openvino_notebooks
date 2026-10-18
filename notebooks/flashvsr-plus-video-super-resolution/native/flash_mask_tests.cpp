#include "flash_mask.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using otc::ai::FlashMaskConfig;
using otc::ai::FlashMaskLayout;
using otc::ai::packFlashOrderedIds;

void expectFailure(const std::function<void()>& action) {
  bool failed = false;
  try {
    action();
  } catch (const std::exception&) {
    failed = true;
  }
  assert(failed);
}

FlashMaskConfig baseConfig() {
  FlashMaskConfig config;
  config.temporalWindows = 2U;
  config.spatialRows = 2U;
  config.spatialCols = 2U;
  config.cacheTemporalGroups = 0U;
  config.topk = 1U;
  config.temporalBlockSize = 2U;
  return config;
}

std::vector<float> pooled(std::size_t heads,
                          std::size_t blocks,
                          float value = 0.0F) {
  return std::vector<float>(heads * blocks * 128U, value);
}

void testNonSparseCandidateAndRetention() {
  auto q = pooled(1U, 8U);
  auto k = pooled(1U, 8U);
  auto config = baseConfig();
  config.isSparse = false;
  config.localRange = 1U;
  const FlashMaskLayout result =
      otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  assert(result.blockMask.size() == 64U);
  for (const auto value : result.blockMask) assert(value == 1U || value == 0U);
  for (std::size_t query = 0U; query < 8U; ++query) {
    const std::size_t spatial = query % 4U;
    for (std::size_t key = 0U; key < 8U; ++key)
      assert(result.blockMask[query * 8U + key] ==
             static_cast<std::uint8_t>(key % 4U == spatial));
  }
  assert(result.retainedKeyBlocks == 4U);
  assert((result.cacheIndices == std::vector<std::int32_t>{4, 5, 6, 7}));
}

void testStrictTiesAndEmptyRepair() {
  auto q = pooled(1U, 8U);
  auto k = pooled(1U, 8U);
  auto config = baseConfig();
  config.localRange = 1U;
  config.topk = 1U;
  const auto result =
      otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  // All candidate weights tie at the threshold. Strict '>' selects none and
  // the source's empty-row repair keeps the first candidate only.
  for (std::size_t query = 0U; query < 8U; ++query) {
    const std::size_t first = query % 4U;
    for (std::size_t key = 0U; key < 8U; ++key)
      assert(result.blockMask[query * 8U + key] ==
             static_cast<std::uint8_t>(key == first));
  }
}

void testShortcutWithExtremeFiniteValues() {
  auto q = pooled(1U, 8U, 1.0e10F);
  auto k = pooled(1U, 8U, -1.0e10F);
  auto config = baseConfig();
  config.localRange = 1U;
  config.topk = 16U;
  const auto result =
      otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  // Candidate count is two per row and topk reaches it; the finite score-bound
  // shortcut must select the complete candidate relation despite underflow.
  for (std::size_t query = 0U; query < 8U; ++query) {
    for (std::size_t key = 0U; key < 8U; ++key)
      assert(result.blockMask[query * 8U + key] ==
             static_cast<std::uint8_t>(key % 4U == query % 4U));
  }
}

void testRetentionAndHeadBroadcast() {
  auto q = pooled(2U, 4U, 0.25F);
  auto k = pooled(2U, 6U, -0.5F);
  auto config = baseConfig();
  config.temporalWindows = 2U;
  config.spatialRows = 1U;
  config.spatialCols = 2U;
  config.cacheTemporalGroups = 3U;
  config.localRange.reset();
  config.topk.reset();
  config.topkRatio = 0.0;
  const auto result =
      otc::ai::buildFlashMaskLayout(q.data(), 2U, 4U, k.data(), 2U, 6U, config);
  assert(result.retainedKeyBlocks == 6U);
  assert((result.cacheIndices == std::vector<std::int32_t>{0, 1, 2, 3, 4, 5,
                                                            0, 1, 2, 3, 4, 5}));
  assert(result.blockMask.size() == 2U * 4U * 6U);
}

void testInvalidInputs() {
  auto q = pooled(1U, 8U);
  auto k = pooled(1U, 8U);
  auto config = baseConfig();
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(nullptr, 1U, 8U, k.data(), 1U, 8U, config);
  });
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(
        q.data(), 1U, 8U, nullptr, 1U,
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) + 1U,
        config);
  });
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 2U, 8U, config);
  });
  q[11] = std::numeric_limits<float>::quiet_NaN();
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  });
  q[11] = 0.0F;
  config.localRange = 0U;
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  });
  config.localRange = std::numeric_limits<std::size_t>::max();
  config.isSparse = false;
  const auto hugeRange =
      otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  for (const auto value : hugeRange.blockMask) assert(value == 1U);
  config.isSparse = true;
  config.localRange.reset();
  config.topk.reset();
  config.topkRatio = -1.0;
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  });
  config.topkRatio = std::numeric_limits<double>::quiet_NaN();
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  });
  config.topkRatio = 2.5;
  config.temporalBlockSize = 3U;
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  });
  config.temporalBlockSize = 2U;
  config.spatialRows = 3U;
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  });
  q = pooled(1U, 8U, std::numeric_limits<float>::max());
  k = pooled(1U, 8U, std::numeric_limits<float>::max());
  config.spatialRows = 2U;
  expectFailure([&] {
    (void)otc::ai::buildFlashMaskLayout(q.data(), 1U, 8U, k.data(), 1U, 8U, config);
  });
}

void testOrderedIds() {
  const std::vector<std::uint8_t> mask{
      0, 1, 0, 0, 1, 0,
      1, 0, 1, 0, 0, 1,
      0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0,
      1, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 0, 1,
  };
  const auto packed = packFlashOrderedIds(mask, 2U, 3U, 6U, 6U);
  assert((packed == std::vector<std::int32_t>{
      1, 4, -1, -1, -1, -1,
      0, 2, 5, -1, -1, -1,
      3, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      0, 3, -1, -1, -1, -1,
      1, 5, -1, -1, -1, -1,
  }));

  // Logical rows and their zero-padded physical equivalents produce the
  // identical ordered prefix/-1 suffix, including empty and full prefixes.
  for (const std::size_t capacity : {7U, 2550U}) {
    std::vector<std::uint8_t> padded(6U * capacity, 0U);
    for (std::size_t row = 0; row < 6U; ++row)
      std::copy_n(mask.data() + row * 6U, 6U, padded.data() + row * capacity);
    assert(packFlashOrderedIds(mask, 2U, 3U, 6U, capacity) ==
           packFlashOrderedIds(padded, 2U, 3U, capacity, capacity));
  }
  assert((packFlashOrderedIds({1, 1, 1, 0, 0, 0}, 1U, 2U, 3U, 4U) ==
          std::vector<std::int32_t>{0, 1, 2, -1, -1, -1, -1, -1}));

  auto expectOrderedFailure = [&](const std::vector<std::uint8_t>& badMask,
                                  std::size_t badHeads,
                                  std::size_t badQueries,
                                  std::size_t badKeys,
                                  std::size_t badCapacity) {
    expectFailure([&] {
      (void)packFlashOrderedIds(badMask, badHeads, badQueries, badKeys,
                                badCapacity);
    });
  };
  expectOrderedFailure(mask, 2U, 3U, 6U, 2U);
  auto badValue = mask;
  badValue[0] = 2U;
  expectOrderedFailure(badValue, 2U, 3U, 6U, 6U);
  expectOrderedFailure(std::vector<std::uint8_t>(mask.size() - 1U),
                       2U, 3U, 6U, 6U);
  expectOrderedFailure(mask, 2U, 3U, 6U, 2551U);
  expectOrderedFailure(mask, 2U, 3U, 6U, 0U);
  expectOrderedFailure(mask, 0U, 3U, 6U, 6U);
  expectOrderedFailure(mask, 2U, 3U, 2551U, 6U);
}

}  // namespace

int main() {
  testNonSparseCandidateAndRetention();
  testStrictTiesAndEmptyRepair();
  testShortcutWithExtremeFiniteValues();
  testRetentionAndHeadBroadcast();
  testInvalidInputs();
  testOrderedIds();
  return 0;
}
