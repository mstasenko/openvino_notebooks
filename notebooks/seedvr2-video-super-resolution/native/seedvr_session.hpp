#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace otc::ai {

inline constexpr std::size_t kSeedVrDitBlockCount = 32U;

class SeedVrSession final {
 public:
  SeedVrSession(std::filesystem::path modelDirectory, std::string device, int latentFrames,
                int latentHeight, int latentWidth);
  ~SeedVrSession();

  SeedVrSession(const SeedVrSession&) = delete;
  SeedVrSession& operator=(const SeedVrSession&) = delete;
  SeedVrSession(SeedVrSession&&) noexcept;
  SeedVrSession& operator=(SeedVrSession&&) noexcept;

  // Runs all 32 DiT blocks in place. GPU video/text state stays remote between
  // blocks and is copied back only after the final block. Qualified T1/T2 GPU
  // sessions may retain verified host assets/blobs, never idle compiled graphs.
  void run(std::vector<float>& videoTokens, std::vector<float>& textTokens,
           const std::vector<float>& embedding);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace otc::ai
