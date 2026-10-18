#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
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
  std::uint64_t elements; // FP16; caller derives this using checked shape products.
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
std::string readSeedVrXml(const std::filesystem::path& directory, const SeedVrAssetPin& pin);
void verifySeedVrAsset(const std::filesystem::path& directory, const SeedVrAssetPin& pin);
std::vector<float> readSeedVrPositiveText(const std::filesystem::path& directory);
SeedVrOwnedIr readSeedVrIr(const std::filesystem::path& directory,
                          const SeedVrAssetPin& xml, const SeedVrAssetPin& bin,
                          const std::vector<SeedVrWeightSpan>& spans = {});

struct SeedVrObservedHostMemory {
  std::optional<std::uint64_t> available;
  std::optional<std::uint64_t> cgroupHeadroom;
};
// Unknown or inaccessible host/cgroup limits deliberately disable retention.
SeedVrObservedHostMemory seedVrHostMemory();
// Observations must be obtained by the native consumer, not accepted from IPC.
// Provisional T1 GPU policy, not an atomic reservation or proof of 4K/VAE fit.
bool seedVrCanRetainAssets(std::uint64_t binBytes, std::uint64_t xmlBytes,
                          const SeedVrObservedHostMemory& memory) noexcept;

} // namespace otc::ai
