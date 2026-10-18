#pragma once

#include "flash_stage.hpp"

#include <filesystem>
#include <memory>

namespace otc::ai {

// Internal, fixed 4096-row stage ABI. These are complete, caller-owned BF16
// finite roots from the same admitted GPU context, not renderer-supplied handles.
struct FlashPostInputs {
  ov::Tensor x;
  ov::Tensor selfContext0, selfContext1;
  ov::Tensor crossK, crossV;
  ov::Tensor gateMsa, shiftMlp, scaleMlp, gateMlp;
};

// Structural validation only: does not query, compile, or execute a device.
void validateFlashPostModel(const ov::Model& model);
// Read the portable three-Result IR, then replace only its two private BF16
// rounding barriers with owning Assign sinks. Stock IR readers require paired
// ReadValues when deserializing Assign; this graph deliberately has no feedback.
void prepareFlashPostModel(ov::Model& model);

class FlashPostSession final {
 public:
  // The pipeline owns admission and the Core/context shared with PRE and
  // attention. Keep them alive until this stage has been destroyed.
  FlashPostSession(ov::Core& core, ov::RemoteContext context,
                   std::filesystem::path assets, FlashMode mode);
  ~FlashPostSession();
  FlashPostSession(FlashPostSession&&) noexcept;
  FlashPostSession& operator=(FlashPostSession&&) noexcept;
  FlashPostSession(const FlashPostSession&) = delete;
  FlashPostSession& operator=(const FlashPostSession&) = delete;

  // Authenticates an already prepared BIN and synchronously uploads its 13
  // weights. A failed bind invalidates the bank; stale weights cannot run.
  void bindLayer(std::size_t layer);
  // Single-threaded synchronous GPU-only stage. Output must be a distinct complete BF16 root
  // [1,4096,1536]. No activation readback or numeric F16 conversion is added.
  void run(const FlashPostInputs& inputs, const ov::Tensor& output);
  // The phase controller calls this only between completed requests.
  void suspend() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace otc::ai
