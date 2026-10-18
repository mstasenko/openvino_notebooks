#include "ai_enhance_core.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>
#include <utility>

namespace otc::ai {

namespace {

constexpr std::size_t kSeedVrRopeAxisWidth = kSeedVrRopeFrequencyWidth / 3U;
constexpr std::size_t kSeedVrRopeAxes = 3U;
// The production proof's largest frequency tensor is below 26 million
// floats.  Keep malformed dimensions from turning this metadata helper into
// an unbounded allocation while leaving room for the proven production cap.
constexpr std::size_t kMaxSeedVrRopeElements = 64U * 1024U * 1024U;
constexpr std::size_t kMaxSeedVrSourceTokens = 50'000'000U;

static_assert(kSeedVrRopeAxisWidth * kSeedVrRopeAxes == kSeedVrRopeFrequencyWidth,
              "MM-RoPE3D axis width must divide the public frequency width");

int divideRoundUp(int value, int divisor) {
  return value / divisor + (value % divisor == 0 ? 0 : 1);
}

int roundHalfToEven(double value) {
  const double lower = std::floor(value);
  const double fraction = value - lower;
  if (fraction < 0.5) return static_cast<int>(lower);
  if (fraction > 0.5) return static_cast<int>(lower + 1.0);
  const int integer = static_cast<int>(lower);
  return integer % 2 == 0 ? integer : integer + 1;
}

std::size_t checkedAdd(std::size_t first, std::size_t second, const char* label) {
  if (second > std::numeric_limits<std::size_t>::max() - first) {
    throw std::invalid_argument(std::string("SeedVR2 ") + label + " size overflows");
  }
  return first + second;
}

std::size_t checkedMultiply(std::size_t first, std::size_t second, const char* label) {
  if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
    throw std::invalid_argument(std::string("SeedVR2 ") + label + " size overflows");
  }
  return first * second;
}

std::size_t checkedProduct3(std::size_t first, std::size_t second, std::size_t third,
                            const char* label) {
  return checkedMultiply(checkedMultiply(first, second, label), third, label);
}

float ropeBaseFrequency(std::size_t component) {
  // This mirrors rotary_embedding_torch's float32 computation:
  // 1 / (10000 ** (float(2*i) / 42)).  Keeping the intermediates float is
  // important; computing this in double changes several entries by an ulp.
  static const std::array<float, kSeedVrRopeAxisWidth / 2U> frequencies = [] {
    std::array<float, kSeedVrRopeAxisWidth / 2U> result{};
    for (std::size_t index = 0U; index < result.size(); ++index) {
      const float exponent = static_cast<float>(2U * index) / 42.0F;
      const float denominator = std::pow(10000.0F, exponent);
      result[index] = 1.0F / denominator;
    }
    return result;
  }();
  assert(component < frequencies.size());
  return frequencies[component];
}

void fillRopeAxis(float* row, std::size_t offset, int position) {
  const float coordinate = static_cast<float>(position);
  for (std::size_t component = 0U; component < kSeedVrRopeAxisWidth / 2U; ++component) {
    const float value = coordinate * ropeBaseFrequency(component);
    row[offset + component * 2U] = value;
    row[offset + component * 2U + 1U] = value;
  }
}

void fillVideoRopeRow(float* row, int localFrame, int localY, int localX) {
  // NaMMRotaryEmbedding3d receives each window's local shape.  Its upstream
  // get_freqs() slices the temporal table at the 58 text-token offset, while
  // height and width restart at zero for every window.
  fillRopeAxis(row, 0U, static_cast<int>(kSeedVrTextTokens) + localFrame);
  fillRopeAxis(row, kSeedVrRopeAxisWidth, localY);
  fillRopeAxis(row, 2U * kSeedVrRopeAxisWidth, localX);
}

void fillTextRopeRow(float* row, int position) {
  fillRopeAxis(row, 0U, position);
  std::copy_n(row, kSeedVrRopeAxisWidth, row + kSeedVrRopeAxisWidth);
  std::copy_n(row, kSeedVrRopeAxisWidth, row + 2U * kSeedVrRopeAxisWidth);
}

}  // namespace

VaeOverlapPlan makeVaeOverlapPlan(int latentWidth, int latentHeight,
                                 int tilePixels, int overlapPixels) {
  if (latentWidth <= 0 || latentHeight <= 0 || latentWidth > 2048 || latentHeight > 2048 ||
      tilePixels < 8 || tilePixels > 16384 || tilePixels % 8 != 0 || overlapPixels < 0 ||
      overlapPixels >= tilePixels || overlapPixels % 8 != 0) {
    throw std::invalid_argument("VAE tiling dimensions must be bounded multiples of eight");
  }
  VaeOverlapPlan plan{latentWidth, latentHeight, tilePixels / 8, overlapPixels / 8, {}};
  const int stride = plan.tileSize - plan.overlap;
  for (int y = 0; y < latentHeight; y += stride) {
    const int height = std::min(plan.tileSize, latentHeight - y);
    for (int x = 0; x < latentWidth; x += stride) {
      const int width = std::min(plan.tileSize, latentWidth - x);
      // The source omits tails fully covered by an earlier tile's overlap.
      if ((y > 0 && height <= plan.overlap) || (x > 0 && width <= plan.overlap)) continue;
      if (plan.tiles.size() >= 4096U) throw std::invalid_argument("VAE tile count exceeds bound");
      plan.tiles.push_back({x, y, width, height});
    }
  }
  return plan;
}

VaeSpatialAccumulator::VaeSpatialAccumulator(VaeOverlapPlan plan, std::size_t planes,
                                             int outputScale)
    : plan_(std::move(plan)), planes_(planes), scale_(outputScale), width_(0U), spatialSize_(0U) {
  if (plan_.tileSize < 1 || plan_.tileSize > 2048 || plan_.overlap < 0 ||
      plan_.overlap >= plan_.tileSize || planes_ == 0U || (scale_ != 1 && scale_ != 8)) {
    throw std::invalid_argument("VAE accumulator configuration is invalid");
  }
  const auto expected = makeVaeOverlapPlan(plan_.width, plan_.height,
                                           plan_.tileSize * 8, plan_.overlap * 8);
  if (expected.tiles.size() != plan_.tiles.size()) {
    throw std::invalid_argument("VAE tile plan is incomplete");
  }
  for (std::size_t i = 0; i < expected.tiles.size(); ++i) {
    const auto& actual = plan_.tiles[i];
    const auto& wanted = expected.tiles[i];
    if (actual.x != wanted.x || actual.y != wanted.y || actual.width != wanted.width ||
        actual.height != wanted.height) throw std::invalid_argument("VAE tile plan order differs");
  }
  width_ = checkedMultiply(static_cast<std::size_t>(plan_.width), scale_, "VAE width");
  spatialSize_ = checkedProduct3(width_, static_cast<std::size_t>(plan_.height), scale_, "VAE spatial");
  const auto elements = checkedMultiply(spatialSize_, planes_, "VAE accumulator");
  if (checkedAdd(elements, spatialSize_, "VAE accumulator and counts") > 256U * 1024U * 1024U) {
    throw std::invalid_argument("VAE accumulator exceeds the one-GiB FP32 bound");
  }
  values_.assign(elements, 0.0F);
  counts_.assign(spatialSize_, 0.0F);
  const auto rampSize = static_cast<std::size_t>(plan_.overlap * scale_);
  ramp_.resize(rampSize);
  for (std::size_t i = 0; i < rampSize; ++i) {
    const float t = rampSize == 1U ? 0.0F : static_cast<float>(i) / static_cast<float>(rampSize - 1U);
    ramp_[i] = 0.5F - 0.5F * std::cos(t * std::acos(-1.0F));
  }
}

void VaeSpatialAccumulator::add(const std::vector<float>& tileValues) {
  if (finished_ || failed_ || nextTile_ >= plan_.tiles.size()) {
    throw std::logic_error("VAE accumulator is complete or failed");
  }
  const auto& tile = plan_.tiles[nextTile_];
  const auto height = static_cast<std::size_t>(tile.height * scale_);
  const auto width = static_cast<std::size_t>(tile.width * scale_);
  const auto tileSpatial = checkedMultiply(height, width, "VAE tile spatial");
  if (tileValues.size() != checkedMultiply(tileSpatial, planes_, "VAE tile values") ||
      !std::all_of(tileValues.begin(), tileValues.end(), [](float v) { return std::isfinite(v); })) {
    throw std::invalid_argument("VAE tile values must match shape and be finite");
  }
  const auto weights = [&](std::size_t length, bool leading, bool trailing) {
    std::vector<float> result(length, 1.0F);
    const auto overlap = std::min(ramp_.size(), length - 1U);
    // Match upstream slice assignments: trailing fades overwrite leading ones
    // when configured overlaps intersect within an unusually short tile.
    if (leading) for (std::size_t i = 0; i < overlap; ++i) result[i] = ramp_[i];
    if (trailing) for (std::size_t i = 0; i < overlap; ++i) result[length - overlap + i] = 1.0F - ramp_[i];
    return result;
  };
  const auto wy = weights(height, tile.y > 0, tile.y + tile.height < plan_.height);
  const auto wx = weights(width, tile.x > 0, tile.x + tile.width < plan_.width);
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      const auto source = y * width + x;
      const auto target = (static_cast<std::size_t>(tile.y * scale_) + y) * width_ +
                          static_cast<std::size_t>(tile.x * scale_) + x;
      const float count = counts_[target] + wy[y] * wx[x];
      if (!std::isfinite(count)) {
        failed_ = true;
        throw std::overflow_error("VAE overlap count overflowed");
      }
      counts_[target] = count;
      for (std::size_t plane = 0; plane < planes_; ++plane) {
        const float weighted = (tileValues[plane * tileSpatial + source] * wy[y]) * wx[x];
        const float sum = values_[plane * spatialSize_ + target] + weighted;
        if (!std::isfinite(sum)) {
          failed_ = true;
          throw std::overflow_error("VAE overlap value overflowed");
        }
        values_[plane * spatialSize_ + target] = sum;
      }
    }
  }
  ++nextTile_;
}

std::vector<float> VaeSpatialAccumulator::finish() {
  if (finished_ || failed_ || nextTile_ != plan_.tiles.size()) {
    throw std::logic_error("VAE accumulator requires exactly one value for every planned tile");
  }
  if (!std::all_of(counts_.begin(), counts_.end(), [](float c) { return std::isfinite(c) && c > 0.0F; })) {
    failed_ = true;
    throw std::logic_error("VAE tile blend left uncovered pixels");
  }
  for (std::size_t plane = 0; plane < planes_; ++plane) {
    for (std::size_t pixel = 0; pixel < spatialSize_; ++pixel) {
      auto& value = values_[plane * spatialSize_ + pixel];
      value /= std::max(counts_[pixel], 1.0e-6F);
      if (!std::isfinite(value)) {
        failed_ = true;
        throw std::overflow_error("VAE normalized value overflowed");
      }
    }
  }
  finished_ = true;
  return std::move(values_);
}

std::uint64_t outputFrameCount(double durationSeconds, double outputFps) {
  if (!std::isfinite(durationSeconds) || !std::isfinite(outputFps) || durationSeconds <= 0.0 ||
      outputFps <= 0.0) {
    throw std::invalid_argument("duration and output fps must be finite and positive");
  }
  const long double product = static_cast<long double>(durationSeconds) *
                              static_cast<long double>(outputFps);
  if (!std::isfinite(product) || product > static_cast<long double>(10'000'000ULL)) {
    throw std::invalid_argument("requested output frame count exceeds the bounded limit");
  }
  // Decimal durations such as 0.1 are represented by binary doubles.  Allow
  // only the corresponding floating-point roundoff at an integer boundary;
  // a meaningful fractional frame remains rounded up normally.
  const long double roundoff =
      8.0L * static_cast<long double>(std::numeric_limits<double>::epsilon()) *
      std::max(1.0L, std::abs(product));
  const long double rounded = std::ceil(product - roundoff);
  if (rounded < 1.0L || rounded > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    throw std::invalid_argument("requested output frame count is out of range");
  }
  return static_cast<std::uint64_t>(rounded);
}

std::vector<SrTile> makeSrTiles(int width, int height, int coreSize, int halo) {
  if (width <= 0 || height <= 0 || coreSize <= 0 || halo < 0) {
    throw std::invalid_argument("SR tile dimensions must be positive");
  }
  std::vector<SrTile> tiles;
  const std::size_t columns = (static_cast<std::size_t>(width) + static_cast<std::size_t>(coreSize) - 1U) /
                              static_cast<std::size_t>(coreSize);
  const std::size_t rows = (static_cast<std::size_t>(height) + static_cast<std::size_t>(coreSize) - 1U) /
                           static_cast<std::size_t>(coreSize);
  tiles.reserve(columns * rows);
  for (int y = 0; y < height; y += coreSize) {
    for (int x = 0; x < width; x += coreSize) {
      const int coreWidth = std::min(coreSize, width - x);
      const int coreHeight = std::min(coreSize, height - y);
      const int inputX = std::max(0, x - halo);
      const int inputY = std::max(0, y - halo);
      const int inputRight = std::min(width, x + coreWidth + halo);
      const int inputBottom = std::min(height, y + coreHeight + halo);
      tiles.push_back({x, y, coreWidth, coreHeight, inputX, inputY,
                       inputRight - inputX, inputBottom - inputY});
    }
  }
  return tiles;
}

std::vector<SeedVrWindow> makeSeedVrWindows(int frames, int height, int width, bool shifted) {
  if (frames <= 0 || height <= 0 || width <= 0 || frames > 10'000 || height > 100'000 ||
      width > 100'000) {
    throw std::invalid_argument("SeedVR2 dimensions are outside their bounded range");
  }

  // SeedVR2 requests 4x3x3 windows at a 45x80 latent reference and maps
  // that layout to the actual latent shape.
  const double scale = std::sqrt(3600.0 / (static_cast<double>(height) * width));
  const int resizedHeight = roundHalfToEven(static_cast<double>(height) * scale);
  const int resizedWidth = roundHalfToEven(static_cast<double>(width) * scale);
  const int windowFrames = divideRoundUp(std::min(frames, 30), 4);
  const int windowHeight = divideRoundUp(resizedHeight, 3);
  const int windowWidth = divideRoundUp(resizedWidth, 3);
  if (windowFrames <= 0 || windowHeight <= 0 || windowWidth <= 0) {
    throw std::invalid_argument("SeedVR2 window dimensions are invalid");
  }

  const double frameShift = shifted && windowFrames < frames ? 0.5 : 0.0;
  const double heightShift = shifted && windowHeight < height ? 0.5 : 0.0;
  const double widthShift = shifted && windowWidth < width ? 0.5 : 0.0;
  const auto windowCount = [](int length, int window, double shift) {
    return shift > 0.0 ? static_cast<int>(std::ceil((length - shift) / window)) + 1 :
                         divideRoundUp(length, window);
  };
  const int frameWindows = windowCount(frames, windowFrames, frameShift);
  const int heightWindows = windowCount(height, windowHeight, heightShift);
  const int widthWindows = windowCount(width, windowWidth, widthShift);
  const std::uint64_t count = static_cast<std::uint64_t>(frameWindows) *
                              static_cast<std::uint64_t>(heightWindows) *
                              static_cast<std::uint64_t>(widthWindows);
  if (count > 1'000'000U) throw std::invalid_argument("SeedVR2 window count is too large");

  std::vector<SeedVrWindow> windows;
  windows.reserve(static_cast<std::size_t>(count));
  const auto bounds = [](int index, int window, double shift, int limit) {
    const int start = std::max(static_cast<int>((index - shift) * window), 0);
    const int end = std::min(static_cast<int>((index - shift + 1.0) * window), limit);
    return std::array<int, 2>{start, end};
  };
  // Preserve the upstream width, height, time traversal order.
  for (int xIndex = 0; xIndex < widthWindows; ++xIndex) {
    const auto x = bounds(xIndex, windowWidth, widthShift, width);
    if (x[1] <= x[0]) continue;
    for (int yIndex = 0; yIndex < heightWindows; ++yIndex) {
      const auto y = bounds(yIndex, windowHeight, heightShift, height);
      if (y[1] <= y[0]) continue;
      for (int frameIndex = 0; frameIndex < frameWindows; ++frameIndex) {
        const auto frame = bounds(frameIndex, windowFrames, frameShift, frames);
        if (frame[1] <= frame[0]) continue;
        windows.push_back({frame[0], frame[1] - frame[0], y[0], y[1] - y[0], x[0], x[1] - x[0]});
      }
    }
  }
  return windows;
}

SeedVrWindowMetadata makeSeedVrWindowMetadata(int frames, int height, int width, bool shifted) {
  const auto windows = makeSeedVrWindows(frames, height, width, shifted);
  if (windows.empty()) {
    throw std::invalid_argument("SeedVR2 window metadata requires at least one window");
  }

  const std::size_t sourceTokenCount = checkedProduct3(
      static_cast<std::size_t>(frames), static_cast<std::size_t>(height),
      static_cast<std::size_t>(width), "source token");
  if (sourceTokenCount == 0U || sourceTokenCount > kMaxSeedVrSourceTokens) {
    throw std::invalid_argument("SeedVR2 source token count is outside the bounded range");
  }

  std::vector<std::size_t> lengths;
  lengths.reserve(windows.size());
  std::size_t maxVideoLen = 0U;
  for (const SeedVrWindow& window : windows) {
    if (window.frame < 0 || window.frameCount <= 0 || window.y < 0 || window.height <= 0 ||
        window.x < 0 || window.width <= 0 || window.frame > frames - window.frameCount ||
        window.y > height - window.height || window.x > width - window.width) {
      throw std::logic_error("SeedVR2 window bounds are invalid");
    }
    const std::size_t length = checkedProduct3(
        static_cast<std::size_t>(window.frameCount), static_cast<std::size_t>(window.height),
        static_cast<std::size_t>(window.width), "window token");
    if (length == 0U) throw std::logic_error("SeedVR2 window has no tokens");
    // The trusted Python RoPE implementation builds a temporal table capped
    // at 1024 and spatial tables capped at 128.  Reject shapes that cannot
    // be represented by that exact upstream contract instead of fabricating
    // extrapolated frequencies.
    if (window.frameCount > 1024 - static_cast<int>(kSeedVrTextTokens) ||
        window.height > 128 || window.width > 128) {
      throw std::invalid_argument("SeedVR2 window shape exceeds the MM-RoPE3D contract");
    }
    lengths.push_back(length);
    maxVideoLen = std::max(maxVideoLen, length);
  }

  const std::size_t windowCount = windows.size();
  const std::size_t attentionLen =
      checkedAdd(maxVideoLen, kSeedVrTextTokens, "attention");
  const std::size_t windowTokenCount =
      checkedMultiply(windowCount, maxVideoLen, "window index");
  const std::size_t maskElementCount =
      checkedMultiply(windowCount, attentionLen, "window mask");
  const std::size_t ropeElementCount = checkedProduct3(
      windowCount, attentionLen, kSeedVrRopeFrequencyWidth, "window RoPE");
  if (ropeElementCount > kMaxSeedVrRopeElements) {
    throw std::invalid_argument("SeedVR2 window RoPE metadata exceeds the bounded limit");
  }

  SeedVrWindowMetadata metadata;
  metadata.frames = frames;
  metadata.height = height;
  metadata.width = width;
  metadata.shifted = shifted;
  metadata.sourceTokenCount = sourceTokenCount;
  metadata.windowCount = windowCount;
  metadata.maxVideoLen = maxVideoLen;
  metadata.attentionLen = attentionLen;
  metadata.windowLengths = lengths;
  metadata.windowIdx.assign(windowTokenCount, 0);
  metadata.windowKeyMask.assign(maskElementCount, 0U);
  metadata.windowReverseIdx.assign(sourceTokenCount, -1);
  metadata.windowRopeFreqs.assign(ropeElementCount, 0.0F);

  for (std::size_t windowNumber = 0U; windowNumber < windowCount; ++windowNumber) {
    const SeedVrWindow& window = windows[windowNumber];
    const std::size_t indexRow = checkedMultiply(windowNumber, maxVideoLen, "window index row");
    const std::size_t maskRow = checkedMultiply(windowNumber, attentionLen, "window mask row");
    const std::size_t ropeRow = checkedMultiply(
        checkedMultiply(windowNumber, attentionLen, "window RoPE row"),
        kSeedVrRopeFrequencyWidth, "window RoPE row");
    std::size_t localToken = 0U;
    for (int localFrame = 0; localFrame < window.frameCount; ++localFrame) {
      const std::size_t sourceFrame =
          static_cast<std::size_t>(window.frame + localFrame);
      for (int localY = 0; localY < window.height; ++localY) {
        const std::size_t sourceY = static_cast<std::size_t>(window.y + localY);
        for (int localX = 0; localX < window.width; ++localX) {
          const std::size_t sourceX = static_cast<std::size_t>(window.x + localX);
          const std::size_t sourceRow = checkedAdd(
              checkedMultiply(sourceFrame, static_cast<std::size_t>(height), "source row"),
              sourceY, "source row");
          const std::size_t sourceIndex = checkedAdd(
              checkedMultiply(sourceRow, static_cast<std::size_t>(width), "source index"),
              sourceX, "source index");
          const std::size_t paddedPosition = checkedAdd(indexRow, localToken, "padded position");
          if (metadata.windowReverseIdx[sourceIndex] >= 0) {
            throw std::logic_error("SeedVR2 window reverse map contains a duplicate token");
          }
          metadata.windowIdx[paddedPosition] = static_cast<std::int64_t>(sourceIndex);
          metadata.windowReverseIdx[sourceIndex] = static_cast<std::int64_t>(paddedPosition);
          metadata.windowKeyMask[maskRow + localToken] = 1U;
          const std::size_t ropePosition = checkedAdd(
              ropeRow, checkedMultiply(localToken, kSeedVrRopeFrequencyWidth, "window RoPE position"),
              "window RoPE position");
          fillVideoRopeRow(metadata.windowRopeFreqs.data() + ropePosition, localFrame, localY,
                           localX);
          ++localToken;
        }
      }
    }
    if (localToken != lengths[windowNumber] || localToken > maxVideoLen) {
      throw std::logic_error("SeedVR2 window metadata length mismatch");
    }
    for (std::size_t textToken = 0U; textToken < kSeedVrTextTokens; ++textToken) {
      metadata.windowKeyMask[maskRow + maxVideoLen + textToken] = 1U;
      const std::size_t ropePosition = checkedAdd(
          ropeRow,
          checkedMultiply(maxVideoLen + textToken, kSeedVrRopeFrequencyWidth,
                          "text RoPE position"),
          "text RoPE position");
      fillTextRopeRow(metadata.windowRopeFreqs.data() + ropePosition,
                      static_cast<int>(textToken));
    }
  }

  for (const std::int64_t reversePosition : metadata.windowReverseIdx) {
    if (reversePosition < 0) {
      throw std::logic_error("SeedVR2 window reverse map does not cover every token");
    }
  }
  return metadata;
}

std::size_t gatherSeedVrChunk(const SeedVrWindowMetadata& metadata, std::size_t windowBegin,
                              std::size_t windowEnd, std::size_t innerDimension,
                              const std::vector<float>& chunkVideoOutput,
                              std::vector<float>& videoAccumulator,
                              std::vector<std::uint8_t>& coverage) {
  if (windowBegin >= windowEnd || windowEnd > metadata.windowCount || innerDimension == 0U ||
      metadata.maxVideoLen == 0U || metadata.attentionLen < metadata.maxVideoLen) {
    throw std::invalid_argument("SeedVR2 gather chunk bounds or dimensions are invalid");
  }
  const std::size_t indexElements =
      checkedMultiply(metadata.windowCount, metadata.maxVideoLen, "gather window index");
  const std::size_t maskElements =
      checkedMultiply(metadata.windowCount, metadata.attentionLen, "gather window mask");
  if (metadata.windowIdx.size() != indexElements || metadata.windowKeyMask.size() != maskElements ||
      metadata.windowReverseIdx.size() != metadata.sourceTokenCount ||
      metadata.windowLengths.size() != metadata.windowCount ||
      coverage.size() != metadata.sourceTokenCount) {
    throw std::invalid_argument("SeedVR2 gather metadata or coverage shape is invalid");
  }
  const std::size_t chunkWindows = windowEnd - windowBegin;
  const std::size_t chunkElements = checkedProduct3(
      chunkWindows, metadata.maxVideoLen, innerDimension, "gather chunk output");
  const std::size_t accumulatorElements =
      checkedMultiply(metadata.sourceTokenCount, innerDimension, "gather accumulator");
  if (chunkVideoOutput.size() != chunkElements || videoAccumulator.size() != accumulatorElements) {
    throw std::invalid_argument("SeedVR2 gather tensor shape is invalid");
  }

  std::size_t gathered = 0U;
  for (std::size_t window = windowBegin; window < windowEnd; ++window) {
    const std::size_t activeIndex = checkedMultiply(window, metadata.maxVideoLen, "gather index row");
    const std::size_t activeMask = checkedMultiply(window, metadata.attentionLen, "gather mask row");
    std::size_t activeCount = 0U;
    for (std::size_t position = 0U; position < metadata.maxVideoLen; ++position) {
      const std::int64_t sourceValue = metadata.windowIdx[activeIndex + position];
      if (metadata.windowKeyMask[activeMask + position] == 0U) {
        if (sourceValue != 0) {
          throw std::logic_error("SeedVR2 padded gather index is not zero");
        }
        continue;
      }
      if (sourceValue < 0 || static_cast<std::size_t>(sourceValue) >= metadata.sourceTokenCount) {
        throw std::logic_error("SeedVR2 gather source index is out of bounds");
      }
      const std::size_t source = static_cast<std::size_t>(sourceValue);
      const std::int64_t reverseValue = metadata.windowReverseIdx[source];
      const std::size_t expectedReverse =
          checkedAdd(activeIndex, position, "gather reverse position");
      if (reverseValue < 0 || static_cast<std::uint64_t>(reverseValue) != expectedReverse) {
        throw std::logic_error("SeedVR2 gather differs from the inverse window map");
      }
      if (coverage[source] != 0U) {
        throw std::logic_error("SeedVR2 gather covered a source token more than once");
      }

      const std::size_t outputRow = checkedMultiply(window - windowBegin, metadata.maxVideoLen,
                                                    "gather chunk row");
      const std::size_t outputOffset = checkedMultiply(
          checkedAdd(outputRow, position, "gather chunk position"), innerDimension,
          "gather chunk offset");
      const std::size_t sourceOffset = checkedMultiply(source, innerDimension, "gather source offset");
      std::copy_n(chunkVideoOutput.data() + outputOffset, innerDimension,
                  videoAccumulator.data() + sourceOffset);
      coverage[source] = 1U;
      ++activeCount;
      ++gathered;
    }
    if (activeCount != metadata.windowLengths[window]) {
      throw std::logic_error("SeedVR2 gather window coverage differs from its metadata");
    }
  }
  return gathered;
}

bool exactRgbEqual(int width, int height, const std::vector<std::uint8_t>& first,
                   const std::vector<std::uint8_t>& second) {
  if (width <= 0 || height <= 0) return false;
  const std::size_t expected = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
  return first.size() == expected && second.size() == expected && first == second;
}

std::array<float, 3> samplePlanarRgb(const float* data, std::size_t tensorHeight,
                                     std::size_t tensorWidth, int actualWidth, int actualHeight,
                                     double x, double y) {
  assert(data != nullptr);
  assert(tensorHeight > 0U && tensorWidth > 0U);
  assert(actualWidth > 0 && actualHeight > 0);
  assert(static_cast<std::size_t>(actualWidth) <= tensorWidth);
  assert(static_cast<std::size_t>(actualHeight) <= tensorHeight);
  assert(std::isfinite(x) && std::isfinite(y));
  assert(tensorHeight <= std::numeric_limits<std::size_t>::max() / tensorWidth);
  assert(tensorHeight * tensorWidth <= std::numeric_limits<std::size_t>::max() / 3U);
  const double clampedX = std::clamp(x, 0.0, static_cast<double>(actualWidth - 1));
  const double clampedY = std::clamp(y, 0.0, static_cast<double>(actualHeight - 1));
  const int x0 = std::clamp(static_cast<int>(std::floor(clampedX)), 0, actualWidth - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(clampedY)), 0, actualHeight - 1);
  const int x1 = std::clamp(x0 + 1, 0, actualWidth - 1);
  const int y1 = std::clamp(y0 + 1, 0, actualHeight - 1);
  const double fx = clampedX - static_cast<double>(x0);
  const double fy = clampedY - static_cast<double>(y0);
  const std::size_t plane = tensorHeight * tensorWidth;
  const auto pixel = [tensorWidth](int row, int column) {
    return static_cast<std::size_t>(row) * tensorWidth + static_cast<std::size_t>(column);
  };
  const std::size_t a = pixel(y0, x0);
  const std::size_t b = pixel(y0, x1);
  const std::size_t c = pixel(y1, x0);
  const std::size_t d = pixel(y1, x1);
  std::array<float, 3> result{};
  for (std::size_t channel = 0U; channel < result.size(); ++channel) {
    const float* channelData = data + channel * plane;
    const double top = static_cast<double>(channelData[a]) * (1.0 - fx) +
                       static_cast<double>(channelData[b]) * fx;
    const double bottom = static_cast<double>(channelData[c]) * (1.0 - fx) +
                          static_cast<double>(channelData[d]) * fx;
    result[channel] = static_cast<float>(top * (1.0 - fy) + bottom * fy);
  }
  return result;
}

}  // namespace otc::ai
