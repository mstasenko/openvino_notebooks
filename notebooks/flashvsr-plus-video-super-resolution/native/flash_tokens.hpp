#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace otc::ai {

inline constexpr std::size_t kFlashTokenChannels = 1536U;
inline constexpr std::size_t kFlashTokenWindowTokens = 128U;
inline constexpr std::size_t kFlashTokenSlabRows = 4096U;
inline constexpr std::size_t kFlashTokenSlabWindows =
    kFlashTokenSlabRows / kFlashTokenWindowTokens;

// The post graph's canonical order is [frame, token-row, token-column].  A
// window contains two adjacent frames and an 8x8 spatial tile, for 128 rows.
// This is deliberately the fixed f4 contract used by RemoteTokenState; it is
// not a general video-layout abstraction.
struct FlashTokenGeometry final {
  std::size_t frames = 0U;
  std::size_t tokenHeight = 0U;
  std::size_t tokenWidth = 0U;

  std::size_t tokenCount() const noexcept {
    return frames * tokenHeight * tokenWidth;
  }

  std::size_t windowCount() const noexcept {
    return tokenCount() / kFlashTokenWindowTokens;
  }
};

struct FlashBf16ConstBuffer final {
  const std::uint16_t* data = nullptr;
  std::size_t words = 0U;
};

struct FlashBf16Buffer final {
  std::uint16_t* data = nullptr;
  std::size_t words = 0U;
};

FlashTokenGeometry makeFlashTokenGeometry(std::size_t frames,
                                          std::size_t tokenHeight,
                                          std::size_t tokenWidth);

// Return the valid prefix size required in a caller-owned slab.  A slab may
// be larger (the reviewed graph uses 4096 rows); unused words are untouched.
std::size_t flashTokenSlabWords(std::size_t windowCount);

// Canonical row indices in this same window/slab order. RoPE has 64 channels,
// not 1536 token channels; sharing indices avoids a second window convention.
std::vector<std::size_t> flashTokenSlabIndices(const FlashTokenGeometry& geometry,
                                             std::size_t windowStart,
                                             std::size_t windowCount);

// Validate one scheduler range and return its valid BF16 word count.  This is
// shared by the session wrapper so PRE reads and POST commits cannot drift
// from the Python 32-window scheduler contract.
std::size_t flashTokenSlabWords(const FlashTokenGeometry& geometry,
                                std::size_t windowStart,
                                std::size_t windowCount);

// Gather/scatter only one validated, source-order window range.  Ranges must
// match the Python scheduler: starts are 32-window aligned, nonterminal calls
// contain 32 windows, and only the terminal call may be shorter.
void flashGatherTokenSlab(const FlashTokenGeometry& geometry,
                          FlashBf16ConstBuffer canonical,
                          std::size_t windowStart,
                          std::size_t windowCount,
                          FlashBf16Buffer slab);

void flashScatterTokenSlab(const FlashTokenGeometry& geometry,
                           FlashBf16ConstBuffer slab,
                           std::size_t windowStart,
                           std::size_t windowCount,
                           FlashBf16Buffer canonical);

// Combine the BF16 prefix output and BF16 LQ condition in F32, then round
// once to native BF16 storage bits in window-major order.  No numeric F16
// conversion is involved.  All validation happens before the first write.
void flashCombineTokenSlab(const FlashTokenGeometry& geometry,
                           FlashBf16ConstBuffer canonical,
                           FlashBf16ConstBuffer condition,
                           std::size_t windowStart,
                           std::size_t windowCount,
                           FlashBf16Buffer slab);

}  // namespace otc::ai
