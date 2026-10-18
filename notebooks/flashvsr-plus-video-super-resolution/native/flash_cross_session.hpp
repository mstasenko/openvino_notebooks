#pragma once

#include "flash_stage.hpp"

#include <filesystem>
#include <memory>

namespace otc::ai {

// Build each layer's static text K/V once, into caller-owned remote roots.
// The caller owns device resolution/admission and the lifetime of these roots.
// BF16/refill behavior matches the resident Parameter-weight reference for
// layers 0/23/29 in both modes on the pinned B580 build. This is not yet a
// complete-controller or other-device/backend qualification.
class FlashCrossSession final {
 public:
  FlashCrossSession(ov::Core& core, ov::RemoteContext context,
                    std::filesystem::path assets, FlashMode mode);
  ~FlashCrossSession();
  FlashCrossSession(FlashCrossSession&&) noexcept;
  FlashCrossSession& operator=(FlashCrossSession&&) noexcept;

  void bindLayer(std::size_t layer);
  void run(const ov::Tensor& context, const ov::Tensor& crossK,
           const ov::Tensor& crossV);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace otc::ai
