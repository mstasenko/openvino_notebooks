#pragma once
#include <cstddef>
#include "seedvr_assets.hpp"
#include "seedvr_prepare.hpp"

namespace otc::ai::prepare_detail {
// One bounded internal preparer operation may publish at most this many
// compiled outputs.  The value is a private compiled-code limit, not an IPC
// or caller-provided capacity.
inline constexpr std::size_t kMaxPreparedPlanCount = 96U;

// Private compiled data/test seam. Never construct these records from IPC/JSON.
enum class SegmentEncoding : std::uint8_t {
  copy = 0,
  f32_to_bf16_rne = 1,
};
struct Segment {
  std::uint64_t offset, bytes, sourceOffset;
  unsigned source;  // 0/1=authenticated learned sources,
                    // 2=structural complement.
  bool learned;
  std::array<std::uint8_t, 32> sha256;
  // Existing generated tables omit this trailing field and remain byte-copy
  // plans.  The Flash extension uses the single fixed conversion below.
  SegmentEncoding encoding = SegmentEncoding::copy;
};
struct Plan {
  SeedVrAssetPin bin, complement;
  const Segment* segments;
  std::size_t count;
};
// Shared by preparation, build-time pinning and decoder normalization. Keep
// this small pure helper independent of the optional filesystem preparer.
// Converts little-endian F32 words in place; only inputBytes/2 bytes survive.
inline void f32ToBf16Rne(std::uint8_t* buffer, std::size_t inputBytes) {
  if (inputBytes % 4 != 0 || (inputBytes != 0 && buffer == nullptr))
    throw std::runtime_error("F32 conversion input is not word aligned or is null");
  for (std::size_t input = 0, output = 0; input < inputBytes;
       input += 4, output += 2) {
    const std::uint32_t bits = std::uint32_t(buffer[input]) |
        (std::uint32_t(buffer[input + 1]) << 8) |
        (std::uint32_t(buffer[input + 2]) << 16) |
        (std::uint32_t(buffer[input + 3]) << 24);
    if ((bits & 0x7f800000U) == 0x7f800000U)
      throw std::runtime_error("F32 source contains NaN or infinity");
    const std::uint32_t rounded = bits + 0x7fffU + ((bits >> 16) & 1U);
    const std::uint16_t result = static_cast<std::uint16_t>(rounded >> 16);
    if ((result & 0x7f80U) == 0x7f80U)
      throw std::runtime_error("finite F32 overflows BF16 during conversion");
    buffer[output] = static_cast<std::uint8_t>(result);
    buffer[output + 1] = static_cast<std::uint8_t>(result >> 8);
  }
}
SeedVrAssetPin pin(const char* name, std::uint64_t bytes, const char* sha);
void prepare(const std::filesystem::path& sourceDirectory,
             const std::filesystem::path& complementDirectory,
             const std::filesystem::path& destination,
             const SeedVrAssetPin (&sources)[2],
             const std::uint64_t (&headerBytes)[2], const Plan* plans,
             std::size_t count, const SeedVrPrepareCallback& progress);
void prepare(const std::filesystem::path& sourceDirectory,
             const std::filesystem::path& complementDirectory,
             const std::filesystem::path& destination,
             const SeedVrAssetPin& source, std::uint64_t headerBytes,
             const Plan* plans, std::size_t count,
             const SeedVrPrepareCallback& progress);
}  // namespace otc::ai::prepare_detail
