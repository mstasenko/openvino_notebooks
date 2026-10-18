#include "flash_tokens.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace otc::ai {
namespace {

constexpr std::size_t kSpatialBlock = 8U;
constexpr std::uint16_t kExponentMask = 0x7f80U;

[[noreturn]] void fail(std::string_view message) {
  throw std::invalid_argument("Flash tokens: " + std::string(message));
}

void require(bool condition, std::string_view message) {
  if (!condition) fail(message);
}

std::size_t checkedMultiply(std::size_t left,
                            std::size_t right,
                            std::string_view label) {
  require(left == 0U || right <= std::numeric_limits<std::size_t>::max() / left,
          label);
  return left * right;
}

std::size_t checkedWords(std::size_t elements, std::string_view label) {
  return checkedMultiply(elements, kFlashTokenChannels, label);
}

std::size_t checkedBytes(std::size_t words, std::string_view label) {
  return checkedMultiply(words, sizeof(std::uint16_t), label);
}

void requirePointer(const void* pointer, std::size_t words, std::string_view label) {
  require(words == 0U || pointer != nullptr, label);
  if (words != 0U) (void)checkedBytes(words, label);
}

struct AddressRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
};

AddressRange addressRange(const void* pointer,
                          std::size_t words,
                          std::string_view label) {
  requirePointer(pointer, words, label);
  if (words == 0U) return {};
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  const auto bytes = checkedBytes(words, label);
  require(begin <= std::numeric_limits<std::uintptr_t>::max() - bytes,
          label);
  return {begin, begin + bytes};
}

bool overlaps(AddressRange left, AddressRange right) {
  return left.begin < right.end && right.begin < left.end;
}

void requireDistinct(AddressRange left,
                     AddressRange right,
                     std::string_view message) {
  require(!overlaps(left, right), message);
}

bool finiteBf16(std::uint16_t value) noexcept {
  return (value & kExponentMask) != kExponentMask;
}

float decodeBf16(std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float decoded = 0.0F;
  std::memcpy(&decoded, &bits, sizeof(decoded));
  return decoded;
}

std::uint16_t encodeBf16(float value) {
  require(std::isfinite(value), "BF16 combine produced a non-finite F32 value");
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const auto rounded = static_cast<std::uint16_t>(
      (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
  require(finiteBf16(rounded), "BF16 combine overflowed native storage");
  return rounded;
}

std::size_t geometryElements(const FlashTokenGeometry& geometry) {
  require(geometry.frames == 4U, "only four-frame token geometry is supported");
  require(geometry.tokenHeight >= kSpatialBlock &&
              geometry.tokenWidth >= kSpatialBlock,
          "token geometry is smaller than one spatial block");
  require(geometry.tokenHeight % kSpatialBlock == 0U &&
              geometry.tokenWidth % kSpatialBlock == 0U,
          "token geometry must contain complete 8x8 blocks");
  require(geometry.tokenHeight <= 240U && geometry.tokenWidth <= 240U,
          "token geometry exceeds reviewed spatial bounds");
  const auto spatial = checkedMultiply(geometry.tokenHeight,
                                       geometry.tokenWidth,
                                       "token geometry overflows");
  require(spatial <= 136U * 240U, "token geometry exceeds reviewed area");
  const auto tokens = checkedMultiply(geometry.frames, spatial,
                                      "token geometry overflows");
  require(tokens != 0U && tokens % kFlashTokenWindowTokens == 0U,
          "token geometry is not a whole 128-row window grid");
  (void)checkedWords(tokens, "canonical token storage overflows");
  return tokens;
}

std::size_t slabWords(std::size_t windowCount) {
  require(windowCount != 0U && windowCount <= kFlashTokenSlabWindows,
          "slab window count is outside the 4096-row bound");
  return checkedWords(checkedMultiply(windowCount, kFlashTokenWindowTokens,
                                      "slab rows overflow"),
                     "slab storage overflows");
}

std::size_t validateRange(const FlashTokenGeometry& geometry,
                          std::size_t windowStart,
                          std::size_t windowCount) {
  const auto windows = geometry.windowCount();
  (void)slabWords(windowCount);
  require(windowStart % kFlashTokenSlabWindows == 0U,
          "slab start must be aligned to 32 windows");
  require(windowStart <= windows && windowCount <= windows - windowStart,
          "slab range is outside token geometry");
  require(windowStart + windowCount == windows ||
              windowCount == kFlashTokenSlabWindows,
          "only the terminal token slab may be short");
  return slabWords(windowCount);
}

void validateCanonical(FlashBf16ConstBuffer buffer,
                       std::size_t words,
                       std::string_view label) {
  require(buffer.words == words, label);
  requirePointer(buffer.data, buffer.words, label);
}

void validateSlab(FlashBf16ConstBuffer slab,
                  std::size_t words,
                  std::string_view label) {
  require(slab.words >= words, label);
  requirePointer(slab.data, slab.words, label);
}

void validateFiniteRange(const std::uint16_t* data,
                        std::size_t words,
                        std::string_view label) {
  for (std::size_t index = 0U; index < words; ++index) {
    if (!finiteBf16(data[index])) fail(label);
  }
}

template <typename Visitor>
void visitWindowWords(const FlashTokenGeometry& geometry,
                      std::size_t windowStart,
                      std::size_t windowCount,
                      Visitor&& visitor) {
  const auto blocksAcross = geometry.tokenWidth / kSpatialBlock;
  const auto spatialWindows = (geometry.tokenHeight / kSpatialBlock) * blocksAcross;
  for (std::size_t local = 0U; local < windowCount; ++local) {
    const auto window = windowStart + local;
    const auto pair = window / spatialWindows;
    const auto spatial = window % spatialWindows;
    const auto blockRow = spatial / blocksAcross;
    const auto blockColumn = spatial % blocksAcross;
    for (std::size_t frame = 0U; frame < 2U; ++frame) {
      for (std::size_t row = 0U; row < kSpatialBlock; ++row) {
        for (std::size_t column = 0U; column < kSpatialBlock; ++column) {
          const auto sourceRow =
              (((pair * 2U + frame) * geometry.tokenHeight +
                blockRow * kSpatialBlock + row) * geometry.tokenWidth +
               blockColumn * kSpatialBlock + column);
          const auto sourceWord = sourceRow * kFlashTokenChannels;
          const auto slabRow =
              (local * kFlashTokenWindowTokens) +
              (frame * kSpatialBlock * kSpatialBlock) +
              row * kSpatialBlock + column;
          const auto slabWord = slabRow * kFlashTokenChannels;
          visitor(slabWord, sourceWord);
        }
      }
    }
  }
}

}  // namespace

FlashTokenGeometry makeFlashTokenGeometry(std::size_t frames,
                                          std::size_t tokenHeight,
                                          std::size_t tokenWidth) {
  FlashTokenGeometry geometry{frames, tokenHeight, tokenWidth};
  (void)geometryElements(geometry);
  return geometry;
}

std::size_t flashTokenSlabWords(std::size_t windowCount) {
  return slabWords(windowCount);
}

std::vector<std::size_t> flashTokenSlabIndices(const FlashTokenGeometry& geometry,
                                             std::size_t windowStart,
                                             std::size_t windowCount) {
  (void)geometryElements(geometry);
  (void)validateRange(geometry, windowStart, windowCount);
  std::vector<std::size_t> indices;
  indices.reserve(windowCount * kFlashTokenWindowTokens);
  visitWindowWords(geometry, windowStart, windowCount,
                   [&](std::size_t, std::size_t canonicalWord) {
                     indices.push_back(canonicalWord / kFlashTokenChannels);
                   });
  return indices;
}

std::size_t flashTokenSlabWords(const FlashTokenGeometry& geometry,
                                std::size_t windowStart,
                                std::size_t windowCount) {
  (void)geometryElements(geometry);
  return validateRange(geometry, windowStart, windowCount);
}

void flashGatherTokenSlab(const FlashTokenGeometry& geometry,
                          FlashBf16ConstBuffer canonical,
                          std::size_t windowStart,
                          std::size_t windowCount,
                          FlashBf16Buffer slab) {
  const auto tokens = geometryElements(geometry);
  const auto required = validateRange(geometry, windowStart, windowCount);
  validateCanonical(canonical, checkedWords(tokens, "canonical token storage"),
                    "canonical token storage has the wrong extent");
  validateSlab({slab.data, slab.words}, required,
               "token slab storage is too small");
  const auto canonicalRange = addressRange(canonical.data, canonical.words,
                                           "canonical token storage");
  const auto slabRange = addressRange(slab.data, slab.words, "token slab storage");
  requireDistinct(canonicalRange, slabRange,
                  "canonical and slab storage must not alias");

  visitWindowWords(geometry, windowStart, windowCount,
                   [&](std::size_t, std::size_t sourceWord) {
                     validateFiniteRange(canonical.data + sourceWord,
                                         kFlashTokenChannels,
                                         "canonical token storage is non-finite");
                   });
  visitWindowWords(geometry, windowStart, windowCount,
                   [&](std::size_t slabWord, std::size_t sourceWord) {
                     std::memcpy(slab.data + slabWord, canonical.data + sourceWord,
                                 kFlashTokenChannels * sizeof(std::uint16_t));
                   });
}

void flashScatterTokenSlab(const FlashTokenGeometry& geometry,
                           FlashBf16ConstBuffer slab,
                           std::size_t windowStart,
                           std::size_t windowCount,
                           FlashBf16Buffer canonical) {
  const auto tokens = geometryElements(geometry);
  const auto required = validateRange(geometry, windowStart, windowCount);
  validateSlab(slab, required, "token slab storage has the wrong extent");
  validateCanonical({canonical.data, canonical.words},
                    checkedWords(tokens, "canonical token storage"),
                    "canonical destination has the wrong extent");
  const auto slabRange = addressRange(slab.data, slab.words, "token slab storage");
  const auto canonicalRange = addressRange(canonical.data, canonical.words,
                                           "canonical destination");
  requireDistinct(slabRange, canonicalRange,
                  "canonical and slab storage must not alias");
  validateFiniteRange(slab.data, required, "token slab storage is non-finite");

  visitWindowWords(geometry, windowStart, windowCount,
                   [&](std::size_t slabWord, std::size_t sourceWord) {
                     std::memcpy(canonical.data + sourceWord, slab.data + slabWord,
                                 kFlashTokenChannels * sizeof(std::uint16_t));
                   });
}

void flashCombineTokenSlab(const FlashTokenGeometry& geometry,
                           FlashBf16ConstBuffer canonical,
                           FlashBf16ConstBuffer condition,
                           std::size_t windowStart,
                           std::size_t windowCount,
                           FlashBf16Buffer slab) {
  const auto tokens = geometryElements(geometry);
  const auto required = validateRange(geometry, windowStart, windowCount);
  const auto canonicalWords = checkedWords(tokens, "canonical token storage");
  validateCanonical(canonical, canonicalWords,
                    "canonical token storage has the wrong extent");
  validateCanonical(condition, canonicalWords,
                    "condition token storage has the wrong extent");
  validateSlab({slab.data, slab.words}, required,
               "token slab storage is too small");
  const auto canonicalRange = addressRange(canonical.data, canonical.words,
                                           "canonical token storage");
  const auto conditionRange = addressRange(condition.data, condition.words,
                                           "condition token storage");
  const auto slabRange = addressRange(slab.data, slab.words, "token slab storage");
  requireDistinct(canonicalRange, conditionRange,
                  "canonical and condition storage must not alias");
  requireDistinct(canonicalRange, slabRange,
                  "canonical and slab storage must not alias");
  requireDistinct(conditionRange, slabRange,
                  "condition and slab storage must not alias");

  // Two passes make the operation transactional with respect to the caller's
  // slab: a later overflow or non-finite source can never leave a partial
  // prefix/LQ result behind.
  visitWindowWords(geometry, windowStart, windowCount,
                   [&](std::size_t, std::size_t sourceWord) {
                     for (std::size_t channel = 0U;
                          channel < kFlashTokenChannels; ++channel) {
                       const auto left = canonical.data[sourceWord + channel];
                       const auto right = condition.data[sourceWord + channel];
                       if (!finiteBf16(left) || !finiteBf16(right))
                         fail("prefix/LQ token storage is non-finite");
                       (void)encodeBf16(decodeBf16(left) + decodeBf16(right));
                     }
                   });
  visitWindowWords(geometry, windowStart, windowCount,
                   [&](std::size_t slabWord, std::size_t sourceWord) {
                     for (std::size_t channel = 0U;
                          channel < kFlashTokenChannels; ++channel) {
                       slab.data[slabWord + channel] = encodeBf16(
                           decodeBf16(canonical.data[sourceWord + channel]) +
                           decodeBf16(condition.data[sourceWord + channel]));
                     }
                   });
}

}  // namespace otc::ai
