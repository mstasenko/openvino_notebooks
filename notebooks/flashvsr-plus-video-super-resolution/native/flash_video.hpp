#pragma once

#include "flash_mode.hpp"
#include "flash_source.hpp"

#include <openvino/openvino.hpp>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>

namespace otc::ai {

// The bounded source-rate Flash controller.  It owns only the current source
// window, one four-latent denoiser result and at most four converted RGB
// outputs; the caller remains responsible for any output-clock/RIFE consumer.
// This is an internal fixed-UHD compression-4 seam, not a general video API.
class FlashVideoController final {
 public:
  using Reader = FlashSourceWindow::Reader;
  // Called once before each new LQ/denoiser call and again only when a later
  // decoder parent follows a returned frame.  The caller can use this to
  // hand the device back from an output-clock consumer; no callback is made
  // between the LQ and denoiser phases of one call.
  using BeforeGpu = std::function<void()>;
  using Progress = std::function<void(double)>;

  FlashVideoController(ov::Core& core, ov::RemoteContext context,
                       std::filesystem::path assets,
                       std::filesystem::path scratch, FlashMode mode,
                       std::size_t sourceFrames, Reader reader,
                       BeforeGpu beforeGpu = {}, Progress progress = {},
                       bool useHostTileHistory = false);
  ~FlashVideoController();
  FlashVideoController(FlashVideoController&&) noexcept;
  FlashVideoController& operator=(FlashVideoController&&) noexcept;
  FlashVideoController(const FlashVideoController&) = delete;
  FlashVideoController& operator=(const FlashVideoController&) = delete;

  // Return the next source-rate RGB24 frame.  The result owns its pixels and
  // preserves the source index/cut flag.  Nullopt is the only normal EOS;
  // reader, callback, arithmetic, and GPU failures permanently retire this
  // controller and are rethrown on the failing/next call.
  std::optional<SourceFrame> next();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace otc::ai
