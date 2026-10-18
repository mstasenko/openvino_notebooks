#pragma once
#include "seedvr_assets.hpp"
#include "seedvr_prepare.hpp"

namespace otc::ai::prepare_detail {
// Private compiled data/test seam. Never construct these records from IPC/JSON.
struct Segment {
  std::uint64_t offset, bytes, sourceOffset;
  unsigned source;  // 0=DiT, 1=VAE, 2=structural complement.
  bool learned;
  std::array<std::uint8_t, 32> sha256;
};
struct Plan {
  SeedVrAssetPin bin, complement;
  const Segment* segments;
  std::size_t count;
};
SeedVrAssetPin pin(const char* name, std::uint64_t bytes, const char* sha);
void prepare(const std::filesystem::path& sourceDirectory,
             const std::filesystem::path& complementDirectory,
             const std::filesystem::path& destination,
             const SeedVrAssetPin (&sources)[2],
             const std::uint64_t (&headerBytes)[2], const Plan* plans,
             std::size_t count, const SeedVrPrepareCallback& progress);
}  // namespace otc::ai::prepare_detail
