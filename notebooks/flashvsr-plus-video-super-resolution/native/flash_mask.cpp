#include "flash_mask.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace otc::ai {
namespace {

constexpr std::size_t kHeadDim = 128U;
[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("Flash mask: " + message);
}

void require(bool condition, std::string_view message) {
  if (!condition) fail(std::string(message));
}

std::size_t checkedMultiply(std::size_t left,
                            std::size_t right,
                            const char* label) {
  require(left == 0U || right <= std::numeric_limits<std::size_t>::max() / left,
          std::string(label) + " size overflow");
  return left * right;
}

void requirePositive(std::size_t value, const std::string& label) {
  require(value != 0U, label + " must be positive");
}

std::size_t checkedTensorElements(std::size_t heads,
                                  std::size_t blocks,
                                  const char* label) {
  return checkedMultiply(checkedMultiply(heads, blocks, label), kHeadDim, label);
}

void validatePooled(const float* values,
                    std::size_t heads,
                    std::size_t blocks,
                    const char* label) {
  requirePositive(heads, std::string(label) + " heads");
  requirePositive(blocks, std::string(label) + " blocks");
  const auto elements = checkedTensorElements(heads, blocks, label);
  require(values != nullptr, std::string(label) + " pointer is null");
  for (std::size_t index = 0U; index < elements; ++index) {
    if (!std::isfinite(values[index]))
      fail(std::string(label) + " contains non-finite values");
  }
}

bool inWindow(std::size_t coordinate,
              std::size_t extent,
              std::size_t center,
              std::size_t localRange) {
  // This is the source's signed interval expressed without unsigned
  // subtraction.  In particular, a large local range cannot wrap at zero.
  const std::size_t half = localRange / 2U;
  if (half > center) {
    const std::size_t delta = half - center;
    if (delta >= localRange) return false;
    const std::size_t upper = localRange - 1U - delta;
    return coordinate <= std::min(upper, extent - 1U);
  }
  const std::size_t start = center - half;
  const std::size_t width = localRange - 1U;
  const std::size_t available = (extent - 1U) - start;
  const std::size_t upper = width >= available ? extent - 1U : start + width;
  return coordinate >= start && coordinate <= upper;
}

std::vector<std::uint8_t> candidateMask(std::size_t queryBlocks,
                                        std::size_t keyBlocks,
                                        std::size_t rows,
                                        std::size_t cols,
                                        const std::optional<std::size_t>& localRange) {
  const std::size_t spatialBlocks = checkedMultiply(rows, cols, "spatial grid");
  const std::size_t size = checkedMultiply(queryBlocks, keyBlocks, "candidate mask");
  std::vector<std::uint8_t> candidates(size, 0U);
  for (std::size_t query = 0U; query < queryBlocks; ++query) {
    const std::size_t querySpatial = query % spatialBlocks;
    const std::size_t queryRow = querySpatial / cols;
    const std::size_t queryCol = querySpatial % cols;
    for (std::size_t key = 0U; key < keyBlocks; ++key) {
      bool selected = true;
      if (localRange.has_value()) {
        const std::size_t keySpatial = key % spatialBlocks;
        selected = inWindow(keySpatial / cols, rows, queryRow, *localRange) &&
                   inWindow(keySpatial % cols, cols, queryCol, *localRange);
      }
      candidates[query * keyBlocks + key] = selected ? 1U : 0U;
    }
  }
  return candidates;
}

double scoreBound(const float* q,
                  std::size_t qElements,
                  const float* k,
                  std::size_t kElements) {
  float qMaximum = 0.0F;
  float kMaximum = 0.0F;
  for (std::size_t index = 0U; index < qElements; ++index)
    qMaximum = std::max(qMaximum, std::fabs(q[index]));
  for (std::size_t index = 0U; index < kElements; ++index)
    kMaximum = std::max(kMaximum, std::fabs(k[index]));
  return static_cast<double>(qMaximum) * static_cast<double>(kMaximum) *
         static_cast<double>(kHeadDim);
}

bool shortcutSafe(const std::vector<std::uint8_t>& candidates,
                  std::size_t queryBlocks,
                  std::size_t keyBlocks,
                  std::size_t groupedTemporal,
                  std::size_t topk,
                  double bound) {
  require(groupedTemporal != 0U, "temporal group count must be positive");
  require(queryBlocks % groupedTemporal == 0U,
          "query blocks do not divide temporal groups");
  const std::size_t groupRows = queryBlocks / groupedTemporal;
  const std::size_t groupWidth = checkedMultiply(groupRows, keyBlocks, "group");
  const std::size_t count = std::min(groupWidth - 1U, topk);
  if (!(std::isfinite(bound) &&
        bound <= static_cast<double>(std::numeric_limits<float>::max()) / 4.0))
    return false;
  for (std::size_t group = 0U; group < groupedTemporal; ++group) {
    std::size_t candidateCount = 0U;
    const std::size_t firstRow = group * groupRows;
    for (std::size_t row = 0U; row < groupRows; ++row) {
      for (std::size_t key = 0U; key < keyBlocks; ++key)
        candidateCount += candidates[(firstRow + row) * keyBlocks + key] != 0U;
    }
    if (!(candidateCount < groupWidth && count >= candidateCount)) return false;
  }
  return true;
}

std::size_t resolvedTopk(const FlashMaskConfig& config,
                         std::size_t spatialBlocks) {
  if (config.topk.has_value()) return *config.topk;
  require(std::isfinite(config.topkRatio) && config.topkRatio >= 0.0,
          "topk ratio must be finite and nonnegative");
  const double spatial = static_cast<double>(spatialBlocks);
  const double raw = spatial * spatial * config.topkRatio;
  require(std::isfinite(raw), "derived topk is not finite");
  const double maximum = static_cast<double>(std::numeric_limits<std::size_t>::max());
  require(raw < maximum, "derived topk exceeds size capacity");
  const auto base = static_cast<std::size_t>(raw);
  return base == 0U ? 0U : base - 1U;
}

float dotProduct(const float* left, const float* right) {
  float accumulator = 0.0F;
  for (std::size_t channel = 0U; channel < kHeadDim; ++channel) {
    const float product = left[channel] * right[channel];
    accumulator = static_cast<float>(accumulator + product);
  }
  return accumulator;
}

void fillEmptyRows(std::vector<std::uint8_t>& mask,
                   std::size_t heads,
                   std::size_t queryBlocks,
                   std::size_t keyBlocks,
                   const std::vector<std::uint8_t>& candidates) {
  for (std::size_t head = 0U; head < heads; ++head) {
    for (std::size_t query = 0U; query < queryBlocks; ++query) {
      const std::size_t row = query * keyBlocks;
      std::size_t first = keyBlocks;
      bool any = false;
      for (std::size_t key = 0U; key < keyBlocks; ++key) {
        if (mask[(head * queryBlocks + query) * keyBlocks + key] != 0U) {
          any = true;
          break;
        }
        if (first == keyBlocks && candidates[row + key] != 0U) first = key;
      }
      if (!any) {
        require(first != keyBlocks, "candidate row is empty");
        mask[(head * queryBlocks + query) * keyBlocks + first] = 1U;
      }
    }
  }
}

void sparseWeights(std::vector<float>& weights,
                   const float* q,
                   const float* k,
                   std::size_t heads,
                   std::size_t queryBlocks,
                   std::size_t keyBlocks,
                   const std::vector<std::uint8_t>& candidates) {
  const float scale = static_cast<float>(std::sqrt(128.0));
  for (std::size_t head = 0U; head < heads; ++head) {
    const float* qHead = q + head * queryBlocks * kHeadDim;
    const float* kHead = k + head * keyBlocks * kHeadDim;
    for (std::size_t query = 0U; query < queryBlocks; ++query) {
      const std::size_t weightRow = (head * queryBlocks + query) * keyBlocks;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t key = 0U; key < keyBlocks; ++key) {
        if (candidates[query * keyBlocks + key] == 0U) {
          weights[weightRow + key] = 0.0F;
          continue;
        }
        const float score = dotProduct(qHead + query * kHeadDim,
                                       kHead + key * kHeadDim) /
                            scale;
        require(std::isfinite(score), "FP32 score overflow");
        weights[weightRow + key] = score;
        maximum = std::max(maximum, score);
      }
      require(std::isfinite(maximum), "candidate row has no finite score");
      float sum = 0.0F;
      for (std::size_t key = 0U; key < keyBlocks; ++key) {
        if (candidates[query * keyBlocks + key] == 0U) continue;
        const float shifted = weights[weightRow + key] - maximum;
        const float value = static_cast<float>(std::exp(shifted));
        sum = static_cast<float>(sum + value);
      }
      require(std::isfinite(sum) && sum > 0.0F, "softmax normalization failed");
      const float logSum = maximum + static_cast<float>(std::log(sum));
      require(std::isfinite(logSum), "softmax log-sum-exp is non-finite");
      for (std::size_t key = 0U; key < keyBlocks; ++key) {
        if (candidates[query * keyBlocks + key] == 0U) continue;
        weights[weightRow + key] = static_cast<float>(
            std::exp(weights[weightRow + key] - logSum));
      }
    }
  }
}

void strictGroupedTopk(std::vector<std::uint8_t>& mask,
                       const std::vector<float>& weights,
                       const std::vector<std::uint8_t>& candidates,
                       std::size_t heads,
                       std::size_t queryBlocks,
                       std::size_t keyBlocks,
                       std::size_t groupedTemporal,
                       std::size_t topk) {
  const std::size_t groupRows = queryBlocks / groupedTemporal;
  const std::size_t groupWidth = checkedMultiply(groupRows, keyBlocks, "group");
  const std::size_t count = std::min(groupWidth - 1U, topk);
  const std::size_t kth = groupWidth - count - 1U;
  std::vector<float> grouped(groupWidth);
  for (std::size_t head = 0U; head < heads; ++head) {
    for (std::size_t group = 0U; group < groupedTemporal; ++group) {
      const std::size_t firstRow = group * groupRows;
      std::size_t position = 0U;
      for (std::size_t row = 0U; row < groupRows; ++row) {
        for (std::size_t key = 0U; key < keyBlocks; ++key, ++position) {
          const std::size_t source =
              (head * queryBlocks + firstRow + row) * keyBlocks + key;
          grouped[position] = candidates[(firstRow + row) * keyBlocks + key] != 0U
                                   ? weights[source]
                                   : -std::numeric_limits<float>::infinity();
        }
      }
      std::nth_element(grouped.begin(), grouped.begin() + kth, grouped.end());
      const float threshold = grouped[kth];
      position = 0U;
      for (std::size_t row = 0U; row < groupRows; ++row) {
        for (std::size_t key = 0U; key < keyBlocks; ++key, ++position) {
          const std::size_t source =
              (head * queryBlocks + firstRow + row) * keyBlocks + key;
          mask[source] = candidates[(firstRow + row) * keyBlocks + key] != 0U &&
                                 weights[source] > threshold
                             ? 1U
                             : 0U;
        }
      }
    }
  }
}

}  // namespace

FlashMaskLayout buildFlashMaskLayout(const float* q,
                                     std::size_t qHeads,
                                     std::size_t qBlocks,
                                     const float* k,
                                     std::size_t kHeads,
                                     std::size_t kBlocks,
                                     const FlashMaskConfig& config) {
  require(qHeads == kHeads, "Q/K head counts differ");
  require(kBlocks <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
          "key block count exceeds int32 index capacity");
  validatePooled(q, qHeads, qBlocks, "Q pooled input");
  validatePooled(k, kHeads, kBlocks, "K pooled input");
  requirePositive(config.spatialRows, "spatial rows");
  requirePositive(config.spatialCols, "spatial columns");
  requirePositive(config.temporalWindows, "temporal windows");
  requirePositive(config.temporalBlockSize, "temporal block size");
  require(config.temporalWindows % config.temporalBlockSize == 0U,
          "temporal block size must divide temporal windows");
  const std::size_t spatialBlocks =
      checkedMultiply(config.spatialRows, config.spatialCols, "spatial grid");
  require(qBlocks % config.temporalWindows == 0U &&
              qBlocks / config.temporalWindows == spatialBlocks,
          "query layout does not match temporal windows and spatial grid");
  require(kBlocks % spatialBlocks == 0U,
          "key cache must contain complete spatial grids");
  if (config.localRange.has_value())
    require(*config.localRange != 0U, "local range must be positive");
  const std::size_t topk = resolvedTopk(config, spatialBlocks);
  const std::size_t groupedTemporal =
      config.temporalWindows / config.temporalBlockSize;
  const std::size_t groupRows = qBlocks / groupedTemporal;
  const std::size_t groupWidth = checkedMultiply(groupRows, kBlocks, "group");
  require(groupWidth > 0U, "group width must be positive");

  const auto candidates = candidateMask(
      qBlocks, kBlocks, config.spatialRows, config.spatialCols,
      config.localRange);
  const std::size_t maskSize = checkedMultiply(
      checkedMultiply(qHeads, qBlocks, "mask"), kBlocks, "mask");
  FlashMaskLayout result;
  result.heads = qHeads;
  result.queryBlocks = qBlocks;
  result.keyBlocks = kBlocks;
  result.blockMask.assign(maskSize, 0U);

  if (!config.isSparse) {
    for (std::size_t head = 0U; head < qHeads; ++head)
      std::copy(candidates.begin(), candidates.end(),
                result.blockMask.begin() + head * qBlocks * kBlocks);
  } else {
    const auto qElements = checkedTensorElements(qHeads, qBlocks, "Q pooled input");
    const auto kElements = checkedTensorElements(kHeads, kBlocks, "K pooled input");
    const double bound = scoreBound(q, qElements, k, kElements);
    const bool shortcut = shortcutSafe(
        candidates, qBlocks, kBlocks, groupedTemporal, topk, bound);
    if (shortcut) {
      for (std::size_t head = 0U; head < qHeads; ++head)
        std::copy(candidates.begin(), candidates.end(),
                  result.blockMask.begin() + head * qBlocks * kBlocks);
    } else {
      require(std::isfinite(bound) &&
                  bound <= static_cast<double>(std::numeric_limits<float>::max()) / 2.0,
              "finite pooled Q/K values exceed the safe FP32 score bound");
      const std::size_t weightSize = checkedMultiply(
          checkedMultiply(qHeads, qBlocks, "weights"), kBlocks, "weights");
      std::vector<float> weights(weightSize, 0.0F);
      sparseWeights(weights, q, k, qHeads, qBlocks, kBlocks, candidates);
      strictGroupedTopk(result.blockMask, weights, candidates, qHeads, qBlocks,
                        kBlocks, groupedTemporal, topk);
    }
  }
  fillEmptyRows(result.blockMask, qHeads, qBlocks, kBlocks, candidates);

  std::size_t keep = spatialBlocks;
  if (config.cacheTemporalGroups != 0U) {
    if (config.cacheTemporalGroups > kBlocks / spatialBlocks)
      keep = kBlocks;
    else
      keep = config.cacheTemporalGroups * spatialBlocks;
  }
  keep = std::min(keep, kBlocks);
  result.retainedKeyBlocks = keep;
  const std::size_t indexSize = checkedMultiply(qHeads, keep, "cache indices");
  result.cacheIndices.resize(indexSize);
  for (std::size_t head = 0U; head < qHeads; ++head) {
    for (std::size_t offset = 0U; offset < keep; ++offset) {
      const std::size_t index = kBlocks - keep + offset;
      require(index <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
              "cache index exceeds int32 capacity");
      result.cacheIndices[head * keep + offset] = static_cast<std::int32_t>(index);
    }
  }
  return result;
}

std::vector<std::int32_t> packFlashOrderedIds(
    const std::vector<std::uint8_t>& blockMask,
    std::size_t heads,
    std::size_t queryBlocks,
    std::size_t keyBlocks,
    std::size_t capacity) {
  requirePositive(heads, "ordered-ID heads");
  requirePositive(queryBlocks, "ordered-ID query blocks");
  requirePositive(keyBlocks, "ordered-ID key blocks");
  requirePositive(capacity, "ordered-ID capacity");
  require(keyBlocks <= 2550U, "ordered-ID key blocks exceed split-key limit");
  require(capacity <= 2550U, "ordered-ID capacity exceeds split-key limit");

  const auto rowCount = checkedMultiply(heads, queryBlocks, "ordered-ID rows");
  const auto expectedMask = checkedMultiply(rowCount, keyBlocks, "ordered-ID mask");
  require(blockMask.size() == expectedMask, "ordered-ID mask size changed");
  const auto outputSize = checkedMultiply(rowCount, capacity, "ordered-ID output");
  std::vector<std::int32_t> packed(outputSize, -1);

  for (std::size_t row = 0U; row < rowCount; ++row) {
    std::size_t selected = 0U;
    for (std::size_t key = 0U; key < keyBlocks; ++key) {
      const auto value = blockMask[row * keyBlocks + key];
      require(value <= 1U, "ordered-ID mask must contain only 0/1 values");
      if (value == 0U) continue;
      require(selected < capacity, "ordered-ID row exceeds capacity");
      packed[row * capacity + selected] = static_cast<std::int32_t>(key);
      ++selected;
    }
  }
  return packed;
}

}  // namespace otc::ai
