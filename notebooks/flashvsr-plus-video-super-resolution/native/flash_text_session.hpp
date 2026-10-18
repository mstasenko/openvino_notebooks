#pragma once

#include "flash_stage.hpp"

#include <filesystem>
#include <memory>

namespace otc::ai {

struct FlashTextInputs final {
  ov::Tensor prompt;
};

struct FlashTextOutputs final {
  ov::Tensor context;
  ov::Tensor textLinear;
};

// One authenticated fixed-512-token text stage.  The caller owns admission
// and the remote context; this wrapper owns the structural bytes and one
// learned BF16 bank for the selected reviewed mode.
class FlashTextSession final {
 public:
  FlashTextSession(ov::Core& core, ov::RemoteContext context,
                   std::filesystem::path assets, FlashMode mode);
  ~FlashTextSession();
  FlashTextSession(FlashTextSession&&) noexcept;
  FlashTextSession& operator=(FlashTextSession&&) noexcept;
  FlashTextSession(const FlashTextSession&) = delete;
  FlashTextSession& operator=(const FlashTextSession&) = delete;

  void run(const FlashTextInputs& inputs, const FlashTextOutputs& outputs);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace otc::ai
