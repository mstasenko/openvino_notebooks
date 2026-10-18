#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace otc::ai {

// CPU-side block-layout contract used by the streamed Flash path.  The native
// helper intentionally returns only the two arrays consumed by the graph;
// it owns no OpenVINO tensors, device state, or input storage.
struct FlashMaskConfig final {
  std::size_t temporalWindows = 0U;
  std::size_t spatialRows = 0U;
  std::size_t spatialCols = 0U;
  std::size_t cacheTemporalGroups = 0U;
  std::optional<std::size_t> localRange;
  std::optional<std::size_t> topk;
  double topkRatio = 2.5;
  std::size_t temporalBlockSize = 2U;
  bool isSparse = true;
};

struct FlashMaskLayout final {
  std::size_t heads = 0U;
  std::size_t queryBlocks = 0U;
  std::size_t keyBlocks = 0U;
  std::size_t retainedKeyBlocks = 0U;
  // Flattened [heads, queryBlocks, keyBlocks], with one byte per bool.
  std::vector<std::uint8_t> blockMask;
  // Flattened [heads, retainedKeyBlocks], in source suffix order.
  std::vector<std::int32_t> cacheIndices;
};

FlashMaskLayout buildFlashMaskLayout(const float* q,
                                     std::size_t qHeads,
                                     std::size_t qBlocks,
                                     const float* k,
                                     std::size_t kHeads,
                                     std::size_t kBlocks,
                                     const FlashMaskConfig& config);

// Pack one validated boolean block mask into the split-key ABI's flattened
// [heads, queryBlocks, 1, capacity] I32 storage.  Each row contains selected
// key IDs in source order followed only by -1.  The helper owns the returned
// vector and performs no OpenVINO or device work. Capacity may exceed the
// logical key count so a fixed device buffer needs no padded boolean scan.
std::vector<std::int32_t> packFlashOrderedIds(
    const std::vector<std::uint8_t>& blockMask,
    std::size_t heads,
    std::size_t queryBlocks,
    std::size_t keyBlocks,
    std::size_t capacity);

}  // namespace otc::ai
