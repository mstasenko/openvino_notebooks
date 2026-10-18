#pragma once

#include "flash_stage.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>

namespace otc::ai {

// One bounded compression-4 causal decoder stream. A parent produces four raw frames; the
// first three startup frames and the final padded tail are not published.
// Large front histories/middle use host USM and the tiled tail stays on the
// GPU by default. The opt-in host-tile policy moves the causal past/free ring
// to host USM without changing graph math. These fixed roots and at most four
// visible F32 frames are independent of clip length. This component is not
// yet a qualified export backend; device admission belongs to the controller.
class FlashDecoderSession final {
 public:
  // The consumer must finish using the borrowed planar [3,H,W] F32 frame
  // before returning. Throwing from either callback permanently retires this
  // stream, just like a failed GPU request (including cancellation).
  using FrameConsumer = std::function<void(std::size_t index, const float*, std::size_t elements)>;
  using Progress = std::function<void(std::size_t completedTiles, std::size_t totalTiles)>;

  FlashDecoderSession(ov::Core& core, ov::RemoteContext context,
                      const std::filesystem::path& assets,
                      std::size_t visibleHeight, std::size_t visibleWidth,
                      std::size_t outputFrames,
                      bool useHostTileHistory = false);
  ~FlashDecoderSession();
  FlashDecoderSession(FlashDecoderSession&&) noexcept;
  FlashDecoderSession& operator=(FlashDecoderSession&&) noexcept;
  FlashDecoderSession(const FlashDecoderSession&) = delete;
  FlashDecoderSession& operator=(const FlashDecoderSession&) = delete;

  // Exactly ceil((outputFrames+3)/4) sequential packed BF16 parents are
  // consumed. No end-of-stream flush, hidden padding, or per-clip vector.
  // Input is a complete [1,784,272,480] USM root in the admitted context;
  // the controller's packer must validate finite BF16 bits before upload.
  // Progress runs between synchronous requests, never during a GPU wait.
  void run(const ov::Tensor& packed, const FrameConsumer& consume,
           const Progress& progress = {});
  // The phase controller calls this only between completed requests. It drops
  // compiled requests/checkers while retaining decoder roots and causal state.
  void suspend() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace otc::ai
