#pragma once

#include "flash_stage.hpp"

#include <filesystem>
#include <memory>

namespace otc::ai {

// Internal fixed 4096-row ABI; complete roots in the admitted shared context.
struct FlashPreInputs {
  ov::Tensor x, tMod, ropeCos, ropeSin;
};
struct FlashPreOutputs {
  ov::Tensor gateMsa, shiftMlp, scaleMlp, gateMlp;
  ov::Tensor qPooled, kPooled, qDevice0, qDevice1, kDevice, vDevice;
};

class FlashPreSession final {
 public:
  // Core/context admission belongs to the pipeline, just as for POST.
  FlashPreSession(ov::Core& core, ov::RemoteContext context,
                  std::filesystem::path assets, FlashMode mode);
  ~FlashPreSession();
  FlashPreSession(FlashPreSession&&) noexcept;
  FlashPreSession& operator=(FlashPreSession&&) noexcept;
  FlashPreSession(const FlashPreSession&) = delete;
  FlashPreSession& operator=(const FlashPreSession&) = delete;

  // Authenticated nine-weight upload. A failed bind leaves the bank invalid.
  void bindLayer(std::size_t layer);
  // Checks the two host-side finite flags before returning GPU activations.
  void run(const FlashPreInputs& inputs, const FlashPreOutputs& outputs);
  // The phase controller calls this only between completed requests.
  void suspend() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace otc::ai
