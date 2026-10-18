#pragma once

#include "flash_mode.hpp"
#include "flash_tokens.hpp"

#include <openvino/openvino.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace otc::ai {

// One source-rate stream of four-latent Flash calls. This owns the bounded
// denoiser phase, not source pixels, LQ projection, decoding, RIFE, or files
// containing whole-video outputs. Suspend between calls to share the device
// with the causal LQ/decoder owners without discarding temporal history.
class FlashDenoiserSession final {
 public:
  // Fraction of this denoiser phase; callbacks may throw to cancel between
  // synchronous requests. Failure permanently retires the stream.
  using Progress = std::function<void(double)>;

  FlashDenoiserSession(ov::Core& core, ov::RemoteContext context,
                       const std::filesystem::path& assets,
                       const std::filesystem::path& scratch,
                       FlashMode mode, std::size_t sourceFrames);
  ~FlashDenoiserSession();
  FlashDenoiserSession(FlashDenoiserSession&&) noexcept;
  FlashDenoiserSession& operator=(FlashDenoiserSession&&) noexcept;
  FlashDenoiserSession(const FlashDenoiserSession&) = delete;
  FlashDenoiserSession& operator=(const FlashDenoiserSession&) = delete;

  // Sequential calls consume one finite BF16 noise [1,16,4,272,480] and LQ
  // condition [1,130560,1536]. The returned compact latent has the noise shape;
  // the caller can pass it to the decoder and release the consumed condition.
  std::vector<std::uint16_t> run(FlashBf16ConstBuffer noise,
                                 FlashBf16ConstBuffer condition,
                                 const Progress& progress = {});

  // Only between completed calls: drop activations and compiled workspaces,
  // retaining authenticated models, learned banks, and disk-backed history.
  // The next run recreates these bounded, per-call resources lazily.
  void suspend() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace otc::ai
