#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace otc::ai {

// Expected pins originate in reviewed native tables, never caller/IPC manifests.
struct SeedVrAssetPin {
  std::string name;
  std::uint64_t bytes;
  std::array<std::uint8_t, 32> sha256;
};
struct SeedVrWeightSpan {
  std::uint64_t offset;
  std::uint64_t bytes;
  std::uint64_t elements; // Two-byte learned storage; checked against the model's type/shape.
};

class SeedVrOwnedIr final {
 public:
  SeedVrOwnedIr(SeedVrOwnedIr&&) noexcept = default;
  SeedVrOwnedIr& operator=(SeedVrOwnedIr&&) noexcept = default;
  SeedVrOwnedIr(const SeedVrOwnedIr&) = delete;
  SeedVrOwnedIr& operator=(const SeedVrOwnedIr&) = delete;

  const std::string& xml() const noexcept { return xml_; }
  const std::uint8_t* binData() const noexcept;
  std::size_t binBytes() const noexcept { return binBytes_; }
  const std::uint8_t* weightData(const SeedVrWeightSpan& span) const;

 private:
  SeedVrOwnedIr() = default;
  friend SeedVrOwnedIr readSeedVrIr(const std::filesystem::path&, const SeedVrAssetPin&,
                                   const SeedVrAssetPin&, const std::vector<SeedVrWeightSpan>&);
  std::string xml_;
  std::vector<std::uint16_t> bin_;
  std::size_t binBytes_ = 0;
  std::size_t prefix_ = 0;
};

// Same API serves retained snapshots and bounded per-block fallback. Parse only
// these owned bytes; OpenVINO models sharing them must not outlive this owner.
// Learned spans are read-only views and may overlap when reusing one weight.
std::string readSeedVrXml(const std::filesystem::path& directory, const SeedVrAssetPin& pin);
void verifySeedVrAsset(const std::filesystem::path& directory, const SeedVrAssetPin& pin);
// Small fixed prompt/conditioning tensors only (at most 8 MiB). Authenticate
// the retained bytes and reject nonfinite BF16 without numeric FP16 conversion.
std::vector<std::uint16_t> readBf16Asset(const std::filesystem::path& directory,
                                       const SeedVrAssetPin& pin);
SeedVrOwnedIr readSeedVrIr(const std::filesystem::path& directory,
                          const SeedVrAssetPin& xml, const SeedVrAssetPin& bin,
                          const std::vector<SeedVrWeightSpan>& spans = {});

} // namespace otc::ai
