#include "flash_tokens.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using otc::ai::FlashBf16Buffer;
using otc::ai::FlashBf16ConstBuffer;
using otc::ai::FlashTokenGeometry;

void rejects(const std::function<void()>& action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

std::size_t sourceRow(const FlashTokenGeometry& geometry,
                      std::size_t window,
                      std::size_t frame,
                      std::size_t row,
                      std::size_t column) {
  const auto across = geometry.tokenWidth / 8U;
  const auto spatial = window % ((geometry.tokenHeight / 8U) * across);
  const auto pair = window / ((geometry.tokenHeight / 8U) * across);
  const auto blockRow = spatial / across;
  const auto blockColumn = spatial % across;
  return (((pair * 2U + frame) * geometry.tokenHeight + blockRow * 8U + row) *
              geometry.tokenWidth + blockColumn * 8U + column);
}

std::uint16_t pattern(std::size_t index) {
  return static_cast<std::uint16_t>(0x0100U + (index % 0x0100U));
}

void testGeometryAndSlabRules() {
  const auto geometry = otc::ai::makeFlashTokenGeometry(4U, 8U, 160U);
  assert(geometry.tokenCount() == 5120U);
  assert(geometry.windowCount() == 40U);
  assert(otc::ai::flashTokenSlabWords(32U) == 32U * 128U * 1536U);
  assert(otc::ai::flashTokenSlabWords(8U) == 8U * 128U * 1536U);
  assert(otc::ai::flashTokenSlabWords(geometry, 0U, 32U) ==
         32U * 128U * 1536U);
  assert(otc::ai::flashTokenSlabWords(geometry, 32U, 8U) ==
         8U * 128U * 1536U);
  rejects([] { (void)otc::ai::makeFlashTokenGeometry(2U, 8U, 8U); });
  rejects([] { (void)otc::ai::makeFlashTokenGeometry(4U, 7U, 8U); });
  rejects([] { (void)otc::ai::makeFlashTokenGeometry(4U, 8U, 241U); });
  rejects([] { (void)otc::ai::flashTokenSlabWords(0U); });
  rejects([] { (void)otc::ai::flashTokenSlabWords(33U); });
  rejects([&] { (void)otc::ai::flashTokenSlabWords(geometry, 0U, 8U); });
  rejects([&] { (void)otc::ai::flashTokenSlabWords(geometry, 32U, 7U); });
}

void testGatherScatterAndTerminalSlab() {
  const auto geometry = otc::ai::makeFlashTokenGeometry(4U, 8U, 160U);
  const auto canonicalWords = geometry.tokenCount() * 1536U;
  std::vector<std::uint16_t> canonical(canonicalWords);
  for (std::size_t index = 0U; index < canonical.size(); ++index)
    canonical[index] = pattern(index);

  std::vector<std::uint16_t> first(otc::ai::flashTokenSlabWords(32U), 0x55aaU);
  otc::ai::flashGatherTokenSlab(geometry, {canonical.data(), canonical.size()}, 0U,
                                32U, {first.data(), first.size()});
  for (std::size_t window = 0U; window < 32U; ++window) {
    for (std::size_t frame = 0U; frame < 2U; ++frame) {
      for (std::size_t row = 0U; row < 8U; ++row) {
        for (std::size_t column = 0U; column < 8U; ++column) {
          for (std::size_t channel = 0U; channel < 1536U; ++channel) {
            const auto source = sourceRow(geometry, window, frame, row, column) *
                                1536U + channel;
            const auto destination =
                ((window * 128U) + frame * 64U + row * 8U + column) * 1536U +
                channel;
            assert(first[destination] == canonical[source]);
          }
        }
      }
    }
  }

  std::vector<std::uint16_t> tail(otc::ai::flashTokenSlabWords(8U), 0x33ccU);
  otc::ai::flashGatherTokenSlab(geometry, {canonical.data(), canonical.size()}, 32U,
                                8U, {tail.data(), tail.size()});
  std::vector<std::uint16_t> restored(canonicalWords, 0x7777U);
  otc::ai::flashScatterTokenSlab(geometry, {first.data(), first.size()}, 0U, 32U,
                                 {restored.data(), restored.size()});
  otc::ai::flashScatterTokenSlab(geometry, {tail.data(), tail.size()}, 32U, 8U,
                                 {restored.data(), restored.size()});
  assert(restored == canonical);

  first[0U] = 0x7f80U;
  std::fill(restored.begin(), restored.end(), 0x7777U);
  rejects([&] {
    otc::ai::flashScatterTokenSlab(geometry, {first.data(), first.size()}, 0U,
                                   32U, {restored.data(), restored.size()});
  });
  for (const auto value : restored) assert(value == 0x7777U);
  first[0U] = canonical[0U];

  rejects([&] {
    otc::ai::flashGatherTokenSlab(geometry, {canonical.data(), canonical.size()},
                                  0U, 8U, {first.data(), first.size()});
  });
  rejects([&] {
    otc::ai::flashGatherTokenSlab(geometry, {canonical.data(), canonical.size()},
                                  32U, 7U, {tail.data(), tail.size()});
  });
}

void testSlabIndicesMatchSourceWindowPartition() {
  // These are the source's tiny, landscape, portrait, and padded-UHD token
  // geometries.  Check every scheduler slab, including each short terminal
  // slab, against the scalar form of window_pre.window_indices.
  for (const auto [frames, height, width] : {
           std::tuple<std::size_t, std::size_t, std::size_t>{4U, 8U, 8U},
           {4U, 8U, 16U}, {4U, 16U, 8U}, {4U, 16U, 16U},
           {4U, 136U, 240U}, {4U, 240U, 136U}}) {
    const auto geometry = otc::ai::makeFlashTokenGeometry(frames, height, width);
    for (std::size_t start = 0U; start < geometry.windowCount(); start += 32U) {
      const auto count = std::min<std::size_t>(
          32U, geometry.windowCount() - start);
      const auto indices = otc::ai::flashTokenSlabIndices(geometry, start, count);
      assert(indices.size() == count * 128U);
      std::size_t cursor = 0U;
      for (std::size_t local = 0U; local < count; ++local) {
        const auto window = start + local;
        for (std::size_t frame = 0U; frame < 2U; ++frame) {
          for (std::size_t row = 0U; row < 8U; ++row) {
            for (std::size_t column = 0U; column < 8U; ++column) {
              assert(indices[cursor++] ==
                     sourceRow(geometry, window, frame, row, column));
            }
          }
        }
      }
      assert(cursor == indices.size());
    }
  }
}

void testCombineRneAndFailureAtomicity() {
  const auto geometry = otc::ai::makeFlashTokenGeometry(4U, 8U, 8U);
  const auto words = geometry.tokenCount() * 1536U;
  std::vector<std::uint16_t> canonical(words, 0x3f80U);  // 1.0
  std::vector<std::uint16_t> condition(words, 0x3b80U);  // 2^-8; midpoint
  std::vector<std::uint16_t> slab(otc::ai::flashTokenSlabWords(2U), 0x55aaU);
  otc::ai::flashCombineTokenSlab(geometry, {canonical.data(), canonical.size()},
                                 {condition.data(), condition.size()}, 0U, 2U,
                                 {slab.data(), slab.size()});
  // 1 + 2^-8 is exactly halfway between BF16 1.0 and 1+2^-7;
  // BF16 RNE chooses the even lower mantissa.
  for (const auto value : slab) assert(value == 0x3f80U);

  std::fill(condition.begin(), condition.end(), 0x7f80U);
  const auto before = slab;
  rejects([&] {
    otc::ai::flashCombineTokenSlab(geometry, {canonical.data(), canonical.size()},
                                   {condition.data(), condition.size()}, 0U, 2U,
                                   {slab.data(), slab.size()});
  });
  assert(slab == before);

  std::fill(condition.begin(), condition.end(), 0x7f7fU);
  std::fill(canonical.begin(), canonical.end(), 0x7f7fU);
  rejects([&] {
    otc::ai::flashCombineTokenSlab(geometry, {canonical.data(), canonical.size()},
                                   {condition.data(), condition.size()}, 0U, 2U,
                                   {slab.data(), slab.size()});
  });
  assert(slab == before);
}

void testAliasAndNonfiniteRejectionBeforeMutation() {
  const auto geometry = otc::ai::makeFlashTokenGeometry(4U, 8U, 8U);
  const auto words = geometry.tokenCount() * 1536U;
  std::vector<std::uint16_t> canonical(words, 0x3f80U);
  std::vector<std::uint16_t> condition(words, 0x3f80U);
  rejects([&] {
    otc::ai::flashGatherTokenSlab(geometry, {canonical.data(), canonical.size()},
                                  0U, 2U, {canonical.data(), canonical.size()});
  });
  rejects([&] {
    otc::ai::flashCombineTokenSlab(geometry, {canonical.data(), canonical.size()},
                                   {canonical.data(), canonical.size()}, 0U, 2U,
                                   {condition.data(), condition.size()});
  });
  canonical[17U] = 0x7fc1U;
  std::vector<std::uint16_t> slab(otc::ai::flashTokenSlabWords(2U), 0x44bbU);
  const auto before = slab;
  rejects([&] {
    otc::ai::flashGatherTokenSlab(geometry, {canonical.data(), canonical.size()},
                                  0U, 2U, {slab.data(), slab.size()});
  });
  assert(slab == before);
}

}  // namespace

int main() {
  try {
    testGeometryAndSlabRules();
    testGatherScatterAndTerminalSlab();
    testSlabIndicesMatchSourceWindowPartition();
    testCombineRneAndFailureAtomicity();
    testAliasAndNonfiniteRejectionBeforeMutation();
    std::cout << "flash tokens: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
