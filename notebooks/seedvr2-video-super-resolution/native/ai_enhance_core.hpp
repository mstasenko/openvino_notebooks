#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace otc::ai {

struct SrTile {
  int coreX = 0;
  int coreY = 0;
  int coreWidth = 0;
  int coreHeight = 0;
  int inputX = 0;
  int inputY = 0;
  int inputWidth = 0;
  int inputHeight = 0;
};

// VAE crops overlap in latent coordinates; unlike SrTile, they have no halo
// or independently owned core. Encoder placement uses scale 1, decoder scale 8.
struct VaeOverlapTile {
  int x;
  int y;
  int width;
  int height;
};

struct VaeOverlapPlan {
  int width;
  int height;
  int tileSize;
  int overlap;
  std::vector<VaeOverlapTile> tiles;
};

VaeOverlapPlan makeVaeOverlapPlan(int latentWidth, int latentHeight,
                                 int tilePixels = 512, int overlapPixels = 64);

class VaeSpatialAccumulator final {
 public:
  // Each input is contiguous [planes,H,W], where planes may flatten C and T.
  // add() consumes the next planned tile, preserving upstream row-major order.
  VaeSpatialAccumulator(VaeOverlapPlan plan, std::size_t planes, int outputScale);
  void add(const std::vector<float>& tileValues);
  std::vector<float> finish();

 private:
  VaeOverlapPlan plan_;
  std::size_t planes_;
  int scale_;
  std::size_t width_;
  std::size_t spatialSize_;
  std::size_t nextTile_ = 0U;
  bool finished_ = false;
  bool failed_ = false;
  std::vector<float> ramp_;
  std::vector<float> values_;
  std::vector<float> counts_;
};

struct SeedVrWindow {
  int frame = 0;
  int frameCount = 0;
  int y = 0;
  int height = 0;
  int x = 0;
  int width = 0;
};

// SeedVR2's NaSwinAttention metadata uses 58 text tokens and 126 MM-RoPE3D
// frequency values per token (three axes, 42 values per axis).  The flat
// vectors below are row-major: window, padded video/text position, then the
// frequency component where applicable.  windowReverseIdx maps each source
// order video token to its unique padded video position.
inline constexpr std::size_t kSeedVrTextTokens = 58U;
inline constexpr std::size_t kSeedVrRopeFrequencyWidth = 126U;

struct SeedVrWindowMetadata {
  int frames = 0;
  int height = 0;
  int width = 0;
  bool shifted = false;
  std::size_t sourceTokenCount = 0U;
  std::size_t windowCount = 0U;
  std::size_t maxVideoLen = 0U;
  std::size_t attentionLen = 0U;
  std::vector<std::size_t> windowLengths;
  std::vector<std::int64_t> windowIdx;
  std::vector<std::uint8_t> windowKeyMask;
  std::vector<std::int64_t> windowReverseIdx;
  std::vector<float> windowRopeFreqs;
};

std::uint64_t outputFrameCount(double durationSeconds, double outputFps);
std::vector<SrTile> makeSrTiles(int width, int height, int coreSize = 256, int halo = 32);
std::vector<SeedVrWindow> makeSeedVrWindows(int frames, int height, int width, bool shifted);
SeedVrWindowMetadata makeSeedVrWindowMetadata(int frames, int height, int width, bool shifted);
std::size_t gatherSeedVrChunk(const SeedVrWindowMetadata& metadata, std::size_t windowBegin,
                              std::size_t windowEnd, std::size_t innerDimension,
                              const std::vector<float>& chunkVideoOutput,
                              std::vector<float>& videoAccumulator,
                              std::vector<std::uint8_t>& coverage);
bool exactRgbEqual(int width, int height, const std::vector<std::uint8_t>& first,
                  const std::vector<std::uint8_t>& second);
// Hot-path helper. The caller must provide non-null contiguous planar FP32 RGB
// storage with tensorHeight/tensorWidth and actualWidth/actualHeight already
// validated, and finite x/y coordinates. The tensor row stride is tensorWidth.
std::array<float, 3> samplePlanarRgb(const float* data, std::size_t tensorHeight,
                                     std::size_t tensorWidth, int actualWidth, int actualHeight,
                                     double x, double y);

}  // namespace otc::ai
