#pragma once

#include "flash_stage.hpp"

#include <filesystem>
#include <memory>

namespace otc::ai {

// One causal projection stream using the fixed UHD structural assets. The
// controller owns admission and source-frame mapping; this class owns only
// learned banks and two ping-pong cache pairs in their shared GPU context.
// Startup is retired before the two steady stages are compiled. Admission
// must still cover both cache pairs and the active stages' banks/workspaces.
// This component is not yet qualified as a video-export backend.
class FlashLqSession final {
 public:
  FlashLqSession(ov::Core& core, ov::RemoteContext context,
                 std::filesystem::path assets, FlashMode mode);
  ~FlashLqSession();
  FlashLqSession(FlashLqSession&&) noexcept;
  FlashLqSession& operator=(FlashLqSession&&) noexcept;
  FlashLqSession(const FlashLqSession&) = delete;
  FlashLqSession& operator=(const FlashLqSession&) = delete;

  // Startup consumes frame 0 once. Each project consumes the next four model
  // frames; StreamPlan supplies repetition/padding, not this cache owner.
  void start(const ov::Tensor& firstFrame); // BF16 [1,3,1,2176,3840]
  // The caller retains or consumes this complete remote root before reuse.
  void project(const ov::Tensor& fourFrames, const ov::Tensor& projected);
  // The phase controller calls this only between completed requests. It drops
  // compiled requests/checkers while retaining causal cache roots and state.
  void suspend() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace otc::ai
