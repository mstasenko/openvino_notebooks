#include "flash_decoder_frame.hpp"

#include "seedvr_prepare_detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace otc::ai {
namespace {

constexpr std::size_t kMiddleRows = kFlashDecoderFrameWorkingHeight / 2;
constexpr std::size_t kMiddleCoreRows = kMiddleRows / kFlashDecoderFrameLanes;
constexpr std::size_t kMiddleHaloRows = 12;
constexpr std::size_t kTileMiddleRows = kMiddleCoreRows + 2 * kMiddleHaloRows;
constexpr std::size_t kRawTileElements =
    kFlashDecoderFrameChannels * kFlashDecoderFrameTileRows *
    kFlashDecoderFrameWorkingWidth;

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

std::array<FlashDecoderFrameTile, kFlashDecoderFrameLanes> makeTiles() {
  std::array<FlashDecoderFrameTile, kFlashDecoderFrameLanes> result{};
  for (std::size_t lane = 0; lane < result.size(); ++lane) {
    const auto global_middle_start = lane * kMiddleCoreRows;
    const auto input_middle_start = std::min(
        std::max<std::size_t>(global_middle_start, kMiddleHaloRows) -
            kMiddleHaloRows,
        kMiddleRows - kTileMiddleRows);
    const auto local_middle_start = global_middle_start - input_middle_start;
    result[lane] = {
        input_middle_start / 2,
        kFlashDecoderFrameTileCompactRows,
        2 * local_middle_start,
        2 * global_middle_start,
        kFlashDecoderFrameTileCoreRows,
    };
  }
  return result;
}

const std::array<FlashDecoderFrameTile, kFlashDecoderFrameLanes>& tiles() {
  static const auto value = makeTiles();
  return value;
}

bool validRawBf16(std::uint16_t bits) {
  // 0x7f00 is the first BF16 magnitude whose doubled F32 value overflows;
  // it is therefore rejected along with Inf/NaN, matching plus_decoder.py.
  return (bits & 0x7fffU) < 0x7f00U;
}

float bitsToFloat(std::uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint16_t roundBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  std::uint8_t bytes[sizeof(bits)] = {
      static_cast<std::uint8_t>(bits), static_cast<std::uint8_t>(bits >> 8),
      static_cast<std::uint8_t>(bits >> 16), static_cast<std::uint8_t>(bits >> 24),
  };
  // Reuse the authenticated native RNE implementation; this is not a numeric
  // FP16 conversion.
  prepare_detail::f32ToBf16Rne(bytes, sizeof(bytes));
  return static_cast<std::uint16_t>(bytes[0] | (std::uint16_t(bytes[1]) << 8));
}

float normalizeWord(std::uint16_t bits) {
  require(validRawBf16(bits), "decoder tile contains non-finite or overflowing BF16");
  const float source = bitsToFloat(static_cast<std::uint32_t>(bits) << 16);
  // Preserve both source operations, without a compiler FMA changing rounding.
  volatile float doubled = source * 2.0f;
  volatile float shifted = doubled - 1.0f;
  const float result = bitsToFloat(static_cast<std::uint32_t>(roundBf16(shifted)) << 16);
  require(std::isfinite(result), "decoder normalization produced non-finite F32");
  return result;
}

const std::array<std::uint16_t, 256>& rgbLut() {
  static const auto value = [] {
    std::array<std::uint16_t, 256> result{};
    for (std::size_t color = 0; color < result.size(); ++color) {
      // Match source F32 division/subtraction, not reciprocal multiplication.
      volatile float divided = static_cast<float>(color) / 127.5f;
      volatile float shifted = divided - 1.0f;
      result[color] = roundBf16(shifted);
    }
    return result;
  }();
  return value;
}

void requireDisjoint(const void* input, std::size_t input_bytes,
                     const void* output, std::size_t output_bytes) {
  const auto first = reinterpret_cast<std::uintptr_t>(input);
  const auto second = reinterpret_cast<std::uintptr_t>(output);
  constexpr auto maximum = std::numeric_limits<std::uintptr_t>::max();
  require(first <= maximum - input_bytes && second <= maximum - output_bytes,
          "decoder packing address extent overflows");
  require(first + input_bytes <= second || second + output_bytes <= first,
          "decoder packed output overlaps an input");
}

const std::array<float, 1U << 16>& normalizationLut() {
  static const auto value = [] {
    std::array<float, 1U << 16> result{};
    result.fill(std::numeric_limits<float>::quiet_NaN());
    for (std::size_t bits = 0; bits < result.size(); ++bits) {
      const auto word = static_cast<std::uint16_t>(bits);
      if (validRawBf16(word)) result[bits] = normalizeWord(word);
    }
    return result;
  }();
  return value;
}

}  // namespace

const FlashDecoderFrameTile& flashDecoderFrameTile(std::size_t lane) {
  if (lane >= kFlashDecoderFrameLanes)
    throw std::out_of_range("decoder frame lane is outside the fixed 16-lane ABI");
  return tiles()[lane];
}

void packFlashLqRgb(const std::array<const std::uint8_t*, 4>& rgb,
                    std::size_t frame_count, std::size_t rgb_frame_bytes,
                    const FlashGeometry& geometry, std::uint16_t* packed,
                    std::size_t packed_elements) {
  const auto canonical = makeFlashGeometry(geometry.visibleHeight, geometry.visibleWidth);
  require(geometry.workingHeight == canonical.workingHeight &&
              geometry.workingWidth == canonical.workingWidth,
          "LQ packing geometry is not canonical");
  // Canonical geometry bounds these products on a supported size_t.
  require(frame_count == 1U || frame_count == 4U, "LQ packing requires one or four frames");
  const auto plane = canonical.workingHeight * canonical.workingWidth;
  require(rgb_frame_bytes == 3U * canonical.visibleHeight * canonical.visibleWidth &&
              packed_elements == 3U * frame_count * plane && packed,
          "LQ packing buffer extent differs");
  for (std::size_t frame = 0; frame < frame_count; ++frame) {
    require(rgb[frame], "LQ RGB frame is null");
    requireDisjoint(rgb[frame], rgb_frame_bytes, packed, packed_elements * sizeof(*packed));
  }
  const auto& lut = rgbLut();
  for (std::size_t channel = 0; channel < 3U; ++channel)
    for (std::size_t frame = 0; frame < frame_count; ++frame)
      for (std::size_t y = 0; y < canonical.workingHeight; ++y) {
        const auto row = std::min(y, canonical.visibleHeight - 1U);
        const auto* source = rgb[frame] + row * canonical.visibleWidth * 3U;
        auto* target = packed + (channel * frame_count + frame) * plane + y * canonical.workingWidth;
        for (std::size_t x = 0; x < canonical.workingWidth; ++x)
          target[x] = lut[source[3U * std::min(x, canonical.visibleWidth - 1U) + channel]];
      }
}

void packFlashDecoderParentRgb(const std::array<const std::uint8_t*, 4>& rgb,
                               std::size_t rgb_frame_bytes,
                               const FlashGeometry& geometry,
                               const std::uint16_t* latent_chunk,
                               std::size_t latent_elements,
                               std::size_t inner,
                               std::uint16_t* packed,
                               std::size_t packed_elements) {
  const auto canonical = makeFlashGeometry(geometry.visibleHeight, geometry.visibleWidth);
  require(geometry.workingHeight == canonical.workingHeight &&
              geometry.workingWidth == canonical.workingWidth &&
              geometry.latentHeight == canonical.latentHeight &&
              geometry.latentWidth == canonical.latentWidth &&
              geometry.tokenHeight == canonical.tokenHeight &&
              geometry.tokenWidth == canonical.tokenWidth &&
              geometry.spatialTokens == canonical.spatialTokens &&
              geometry.spatialWindows == canonical.spatialWindows,
          "decoder packing geometry is not canonical");
  // Canonical geometry bounds all products, including on a 32-bit size_t.
  const auto plane = canonical.latentHeight * canonical.latentWidth;
  require(rgb_frame_bytes == 3 * canonical.visibleHeight * canonical.visibleWidth &&
              latent_elements == 16 * 4 * plane && packed_elements == 784 * plane,
          "decoder packing buffer extent differs");
  require(inner < 4 && packed && latent_chunk, "decoder packing inputs are invalid");
  for (const auto* frame : rgb) {
    require(frame, "decoder RGB frame is null");
    requireDisjoint(frame, rgb_frame_bytes, packed, packed_elements * sizeof(*packed));
  }
  requireDisjoint(latent_chunk, latent_elements * sizeof(*latent_chunk),
                  packed, packed_elements * sizeof(*packed));
  for (std::size_t index = 0; index < latent_elements; ++index)
    require((latent_chunk[index] & 0x7f80U) != 0x7f80U,
            "decoder latent chunk contains non-finite BF16");

  const auto& lut = rgbLut();
  for (std::size_t channel = 0; channel < 3; ++channel)
    for (std::size_t frame = 0; frame < rgb.size(); ++frame)
      for (std::size_t dy = 0; dy < 8; ++dy)
        for (std::size_t dx = 0; dx < 8; ++dx) {
          const auto packed_channel = ((channel * 4 + frame) * 8 + dy) * 8 + dx;
          auto* target = packed + packed_channel * plane;
          for (std::size_t y = 0; y < canonical.latentHeight; ++y) {
            const auto row = std::min(8 * y + dy, canonical.visibleHeight - 1);
            const auto* source = rgb[frame] + row * canonical.visibleWidth * 3;
            for (std::size_t x = 0; x < canonical.latentWidth; ++x) {
              const auto column = std::min(8 * x + dx, canonical.visibleWidth - 1);
              target[y * canonical.latentWidth + x] = lut[source[3 * column + channel]];
            }
          }
        }
  for (std::size_t channel = 0; channel < 16; ++channel)
    std::copy_n(latent_chunk + (channel * 4 + inner) * plane, plane,
                packed + (768 + channel) * plane);
}

void normalizeAndScatterFlashDecoderTile(const std::uint16_t* raw_bf16,
                                          std::size_t raw_elements,
                                          std::size_t lane,
                                          std::size_t visible_height,
                                          std::size_t visible_width,
                                          float* destination,
                                          std::size_t destination_elements) {
  const auto& tile = flashDecoderFrameTile(lane);
  require(visible_height > 0 && visible_height <= kFlashDecoderFrameWorkingHeight &&
              visible_width > 0 && visible_width <= kFlashDecoderFrameWorkingWidth,
          "decoder visible crop exceeds the fixed working raster");
  require(raw_elements == kRawTileElements, "decoder raw tile element count differs");
  // The crop bounds above cap this product at 25,067,520 elements.
  const auto expectedDestination = kFlashDecoderFrameChannels * visible_height * visible_width;
  require(destination_elements == expectedDestination,
          "decoder destination element count differs");
  require(raw_bf16 != nullptr && destination != nullptr,
          "decoder tile and destination pointers are required");
  requireDisjoint(raw_bf16, raw_elements * sizeof(*raw_bf16),
                  destination, destination_elements * sizeof(*destination));
  require(tile.compact_row_start + tile.compact_row_count <=
              kFlashDecoderFrameCompactRows,
          "decoder compact tile exceeds its parent");
  require(tile.local_frame_output_start + tile.core_rows <=
              kFlashDecoderFrameTileRows &&
              tile.global_frame_start + tile.core_rows <=
                  kFlashDecoderFrameWorkingHeight,
          "decoder tile core exceeds its raw or global raster");

  // Validate the entire raw result, including halo/discarded rows, before
  // touching the caller's frame, so a nonfinite tile cannot partly update it.
  for (std::size_t index = 0; index < raw_elements; ++index)
    require(validRawBf16(raw_bf16[index]),
            "decoder tile contains non-finite or overflowing BF16");

  const auto& lut = normalizationLut();
  const auto targetStop = std::min(tile.global_frame_start + tile.core_rows,
                                   visible_height);
  if (targetStop <= tile.global_frame_start) return;
  const auto plane = visible_height * visible_width;
  for (std::size_t channel = 0; channel < kFlashDecoderFrameChannels; ++channel) {
    const auto rawChannel = channel * kFlashDecoderFrameTileRows *
                            kFlashDecoderFrameWorkingWidth;
    const auto destinationChannel = channel * plane;
    for (std::size_t globalRow = tile.global_frame_start;
         globalRow < targetStop; ++globalRow) {
      const auto localRow = tile.local_frame_output_start +
                            (globalRow - tile.global_frame_start);
      const auto rawRow = rawChannel + localRow * kFlashDecoderFrameWorkingWidth;
      const auto destinationRow = destinationChannel + globalRow * visible_width;
      for (std::size_t column = 0; column < visible_width; ++column)
        destination[destinationRow + column] = lut[raw_bf16[rawRow + column]];
    }
  }
}

}  // namespace otc::ai
