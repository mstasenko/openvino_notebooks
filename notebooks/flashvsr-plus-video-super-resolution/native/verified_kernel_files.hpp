#pragma once

#include "seedvr_assets.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

namespace otc::ai {

// A compiled-in table supplies these pins.  Each source reference count is
// fixed by the reviewed custom-layer config.
struct VerifiedKernelSourceSpec {
  SeedVrAssetPin asset;
  std::size_t references = 0;
};
struct VerifiedKernelSpec {
  SeedVrAssetPin config;
  std::array<VerifiedKernelSourceSpec, 2> sources;
  std::size_t sourceCount = 0;
};

// Linux-only owner: private canonical config, sealed /proc/self/fd CL sources.
// All returned paths remain valid until this object is destroyed. A root
// worker stages the config only under root-owned /run, never caller TMPDIR.
// No caller-supplied path or XML rewrite rule is accepted.
class VerifiedKernelFiles final {
 public:
  static VerifiedKernelFiles open(const std::filesystem::path& directory,
                                  const VerifiedKernelSpec& spec);

  ~VerifiedKernelFiles();
  VerifiedKernelFiles(VerifiedKernelFiles&&) noexcept;
  VerifiedKernelFiles& operator=(VerifiedKernelFiles&&) noexcept;
  VerifiedKernelFiles(const VerifiedKernelFiles&) = delete;
  VerifiedKernelFiles& operator=(const VerifiedKernelFiles&) = delete;

  const std::string& configPath() const noexcept;
  std::size_t sourceCount() const noexcept;
  const std::string& sourcePath(std::size_t index) const noexcept;

 private:
  class Impl;
  explicit VerifiedKernelFiles(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace otc::ai
