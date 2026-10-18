#pragma once

#include "ai_enhance_core.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ov { class Model; }

namespace otc::ai {

struct SeedVrFrameShape {
  int width;
  int height;
  int paddedWidth;
  int paddedHeight;
  std::size_t latentPixels;
  std::size_t tokens;
  int inputFrames;
  int latentFrames;
};

SeedVrFrameShape seedVrFrameShape(int width, int height, int inputFrames = 1);
void seedVrValidateDevice(const std::string& device);
// Preserve the affected VAE residual boundary before GPU FP16 compilation.
// The original decoded output remains port 0; callers ignore the extra output.
void seedVrPreserveDecoderResidual(ov::Model& model);
// RGB24 must already have the selected output geometry. Resize and optional
// source color correction belong to the caller and must match its reference.
std::vector<float> seedVrNormalizeRgb(const std::vector<std::uint8_t>& rgb,
                                    const SeedVrFrameShape& shape);
std::vector<std::uint8_t> seedVrDecodedRgb(const std::vector<float>& decoded,
                                        const SeedVrFrameShape& shape);

// Extract contiguous C×T BCTHW planes using a latent-coordinate overlap tile.
// Encoder input is RGB at 8x scale; decoder input is 16-channel latent at 1x.
std::vector<float> seedVrVaeTileInput(const std::vector<float>& values,
                                    const SeedVrFrameShape& shape,
                                    const VaeOverlapTile& tile, bool encode);

// Optional synchronous progress: stage name, one-based tile index, tile count.
// Called before compilation/inference and around DiT; exceptions cancel the
// frame. Non-tiled stages report index/count 1/1.
using SeedVrFrameProgress = std::function<void(const char*, std::size_t, std::size_t)>;

inline constexpr const char* kSeedVrNoiseVersion = "splitmix64-boxmuller-f32-hwc-v1";
inline constexpr const char* kSeedVrTemporalNoiseVersion = "splitmix64-boxmuller-f32-thwc-t2-v1";
// One fixed HWC (T1) or explicit time-major THWC (T2) latent noise field. This generator does
// not reproduce torch.randn; supply the reference noise for parity comparisons.
// libm rounding may differ between platforms; the saved field is the contract.
std::vector<float> seedVrFixedNoise(const SeedVrFrameShape& shape, std::uint64_t seed);
// Pure condition packing: BCTHW mean plus frozen THWC noise -> BCTHW 33 channels.
void seedVrLatentCondition(const std::vector<float>& mean, const std::vector<float>& noise,
                          const SeedVrFrameShape& shape, std::vector<float>& condition);

struct SeedVrFrameModels {
  std::filesystem::path encoder;
  std::filesystem::path decoder;
  std::filesystem::path prelude;
  std::filesystem::path postlude;
  std::filesystem::path fusedBlocks;
};

class SeedVrFrameSession final {
 public:
  // positiveText is FP32 [58,5120]; noise is THWC [latentT,H/8,W/8,16].
  // Accepts inputT=1 or 5, latentT=1 or 2. T2 DiT reuse is limited to 960x768.
  // Reviewed T1/T2 metadata guards reshape the same original fused templates.
  // VAE inference overlaps retained DiT state buffers. T1 UHD completed on B580;
  // other devices and UHD T5 remain unqualified. VAE uses 512px tiles/64px
  // overlap; spatial tiling changes context and need not equal untiled output.
  // The GPU decoder uses FP16 with its faulty residual fusion separated;
  // encoder and prelude/postlude retain FP32 arithmetic. Public ports are FP32.
  // Embedding boundaries round to FP16 for coherent block/postlude modulation;
  // this is not an assertion of full FP32 PyTorch parity.
  SeedVrFrameSession(SeedVrFrameModels models, std::string device, int width, int height,
                     std::vector<float> positiveText, std::vector<float> fixedNoise,
                     SeedVrFrameProgress progress = {}, int inputFrames = 1);
  ~SeedVrFrameSession();
  SeedVrFrameSession(const SeedVrFrameSession&) = delete;
  SeedVrFrameSession& operator=(const SeedVrFrameSession&) = delete;
  SeedVrFrameSession(SeedVrFrameSession&&) noexcept;
  SeedVrFrameSession& operator=(SeedVrFrameSession&&) noexcept;

  // Input is chronological concatenated RGB24 for explicit T1 or T5.
  // T5 uses temporal compression, posterior mode, no cross-batch causal cache.
  // Returns cropped RGB24; no full-video frames or intermediates are retained.
  std::vector<std::uint8_t> run(const std::vector<std::uint8_t>& resizedRgb);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace otc::ai
