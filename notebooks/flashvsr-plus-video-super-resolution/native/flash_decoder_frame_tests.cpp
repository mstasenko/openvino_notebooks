#include "flash_decoder_frame.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Action>
void expectRejected(Action&& action, const char* message) {
  try {
    action();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

std::uint32_t floatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bitsFloat(std::uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint16_t oracleBf16Rne(float value) {
  const auto bits = floatBits(value);
  require((bits & 0x7f800000U) != 0x7f800000U,
          "oracle received non-finite F32");
  const auto rounded = bits + 0x7fffU + ((bits >> 16) & 1U);
  const auto result = static_cast<std::uint16_t>(rounded >> 16);
  require((result & 0x7f80U) != 0x7f80U,
          "oracle F32-to-BF16 conversion overflowed");
  return result;
}

float oracleNormalize(std::uint16_t bits) {
  require((bits & 0x7fffU) < 0x7f00U,
          "oracle received invalid raw BF16");
  const auto source = bitsFloat(static_cast<std::uint32_t>(bits) << 16);
  volatile float doubled = source * 2.0f;
  volatile float shifted = doubled - 1.0f;
  const auto normalized = oracleBf16Rne(shifted);
  return bitsFloat(static_cast<std::uint32_t>(normalized) << 16);
}

std::size_t rawIndex(std::size_t channel, std::size_t row, std::size_t column) {
  return (channel * otc::ai::kFlashDecoderFrameTileRows + row) *
         otc::ai::kFlashDecoderFrameWorkingWidth + column;
}

std::size_t outputIndex(std::size_t channel, std::size_t height,
                        std::size_t width, std::size_t row,
                        std::size_t column) {
  return channel * height * width + row * width + column;
}

void checkTileDescriptors() {
  const std::array<std::size_t, 16> compactStarts{
      0, 28, 62, 96, 130, 164, 198, 232,
      266, 300, 334, 368, 402, 436, 470, 498};
  for (std::size_t lane = 0; lane < compactStarts.size(); ++lane) {
    const auto& tile = otc::ai::flashDecoderFrameTile(lane);
    require(tile.compact_row_start == compactStarts[lane],
            "compact tile start differs");
    require(tile.compact_row_count == 46, "compact tile row count differs");
    require(tile.local_frame_output_start ==
                (lane == 0 ? 0 : (lane == 15 ? 48 : 24)),
            "local frame output start differs");
    require(tile.global_frame_start == lane * 136,
            "global frame start differs");
    require(tile.core_rows == 136, "raw core row count differs");
  }
  expectRejected([] { (void)otc::ai::flashDecoderFrameTile(16); },
                 "invalid lane was accepted");
}

void checkAllFiniteWords() {
  constexpr auto rawElements = otc::ai::kFlashDecoderFrameChannels *
                               otc::ai::kFlashDecoderFrameTileRows *
                               otc::ai::kFlashDecoderFrameWorkingWidth;
  constexpr std::size_t visibleHeight = 2176;
  constexpr std::size_t visibleWidth = 3840;
  std::vector<std::uint16_t> raw(rawElements, 0);
  std::size_t count = 0;
  for (std::size_t bits = 0; bits <= 0xffff; ++bits) {
    if ((bits & 0x7fffU) < 0x7f00U) raw[count++] = static_cast<std::uint16_t>(bits);
  }
  std::vector<float> output(3 * visibleHeight * visibleWidth, -17.0f);
  otc::ai::normalizeAndScatterFlashDecoderTile(
      raw.data(), raw.size(), 0, visibleHeight, visibleWidth,
      output.data(), output.size());
  require(count == 65024, "finite BF16 word coverage differs");
  for (std::size_t index = 0; index < count; ++index) {
    const auto row = index / visibleWidth;
    const auto column = index % visibleWidth;
    require(floatBits(output[outputIndex(0, visibleHeight, visibleWidth, row, column)]) ==
                floatBits(oracleNormalize(raw[index])),
            "raw BF16 normalization differs from independent oracle");
  }
}

void checkOnlyCoreAndBottomCrop() {
  constexpr auto rawElements = otc::ai::kFlashDecoderFrameChannels *
                               otc::ai::kFlashDecoderFrameTileRows *
                               otc::ai::kFlashDecoderFrameWorkingWidth;
  std::vector<std::uint16_t> raw(rawElements, 0);
  std::vector<float> output(3 * 2160 * 3840, -9.0f);
  const auto lane = std::size_t{15};
  raw[rawIndex(0, 0, 0)] = 0x4000;   // halo: must not be scattered
  raw[rawIndex(0, 48, 0)] = 0x3f80;  // global row 2040 -> normalized 1
  raw[rawIndex(0, 167, 0)] = 0x4000; // global row 2159 -> normalized 3
  raw[rawIndex(0, 168, 0)] = 0x4080; // global row 2160 -> cropped
  otc::ai::normalizeAndScatterFlashDecoderTile(
      raw.data(), raw.size(), lane, 2160, 3840, output.data(), output.size());
  require(output[outputIndex(0, 2160, 3840, 2039, 0)] == -9.0f,
          "halo row was scattered");
  require(output[outputIndex(0, 2160, 3840, 2040, 0)] == 1.0f,
          "first valid bottom-core row differs");
  require(output[outputIndex(0, 2160, 3840, 2159, 0)] == 3.0f,
          "last visible bottom-core row differs");
}

void checkMalformedInputs() {
  constexpr auto rawElements = otc::ai::kFlashDecoderFrameChannels *
                               otc::ai::kFlashDecoderFrameTileRows *
                               otc::ai::kFlashDecoderFrameWorkingWidth;
  std::vector<std::uint16_t> raw(rawElements, 0);
  std::vector<float> output(3, 42.0f);
  expectRejected([&] {
    otc::ai::normalizeAndScatterFlashDecoderTile(raw.data(), raw.size(), 0, 1, 1,
        reinterpret_cast<float*>(raw.data()), 3);
  }, "overlapping decoder raw/output storage was accepted");
  auto invoke = [&](std::size_t rawCount, std::size_t lane,
                    std::size_t height, std::size_t width,
                    float* destination, std::size_t destinationCount) {
    otc::ai::normalizeAndScatterFlashDecoderTile(
        raw.data(), rawCount, lane, height, width, destination, destinationCount);
  };
  expectRejected([&] { invoke(raw.size() - 1, 0, 1, 1, output.data(), output.size()); },
                 "short raw tile was accepted");
  expectRejected([&] { invoke(raw.size(), 0, 1, 1, output.data(), 2); },
                 "short destination was accepted");
  expectRejected([&] { invoke(raw.size(), 0, 2177, 1, output.data(), 3); },
                 "oversized visible height was accepted");
  expectRejected([&] { invoke(raw.size(), 0, 1, 3841, output.data(), 3); },
                 "oversized visible width was accepted");
  expectRejected([&] { invoke(raw.size(), 0, 1, 1, nullptr, output.size()); },
                 "null destination was accepted");
  expectRejected([&] {
    otc::ai::normalizeAndScatterFlashDecoderTile(
        nullptr, raw.size(), 0, 1, 1, output.data(), output.size());
  }, "null raw tile was accepted");
  expectRejected([&] { invoke(raw.size(), 0, 0, 1, output.data(), output.size()); },
                 "empty visible crop was accepted");
  for (const auto bad : {std::uint16_t{0x7f00}, std::uint16_t{0x7f80},
                         std::uint16_t{0xff80}, std::uint16_t{0x7fc1},
                         std::uint16_t{0xffc1}}) {
    raw[0] = bad;
    std::fill(output.begin(), output.end(), 42.0f);
    expectRejected([&] { invoke(raw.size(), 0, 1, 1, output.data(), output.size()); },
                   "non-finite/overflow BF16 boundary was accepted");
    require(output[0] == 42.0f && output[1] == 42.0f && output[2] == 42.0f,
            "invalid tile wrote before validation completed");
  }
  raw[0] = 0;
  raw.back() = 0x7fc1;
  expectRejected([&] { invoke(raw.size(), 0, 1, 1, output.data(), output.size()); },
                 "nonfinite discarded halo was accepted");
  require(std::all_of(output.begin(), output.end(), [](float value) { return value == 42.0f; }),
          "invalid final halo element allowed a partial frame write");
  expectRejected([&] {
    invoke(raw.size(), 0, std::numeric_limits<std::size_t>::max(), 1,
           output.data(), output.size());
  }, "overflowing visible extent was accepted");
}

std::uint16_t normalizedRgb(std::uint8_t color) {
  volatile float divided = static_cast<float>(color) / 127.5f;
  volatile float shifted = divided - 1.0f;
  return oracleBf16Rne(shifted);
}

void checkLqRgbPacking() {
  const auto geometry = otc::ai::makeFlashGeometry(16U, 24U);
  const auto bytes = 3U * geometry.visibleHeight * geometry.visibleWidth;
  const auto plane = geometry.workingHeight * geometry.workingWidth;
  std::array<std::vector<std::uint8_t>, 4> frames;
  std::array<const std::uint8_t*, 4> rgb{};
  for (std::size_t frame = 0; frame < frames.size(); ++frame) {
    frames[frame].resize(bytes);
    for (std::size_t i = 0; i < bytes; ++i)
      frames[frame][i] = static_cast<std::uint8_t>((17U * i + 11U * frame) % 256U);
    rgb[frame] = frames[frame].data();
  }
  for (const auto count : {1U, 4U}) {
    std::vector<std::uint16_t> packed(3U * count * plane, 42U);
    otc::ai::packFlashLqRgb(rgb, count, bytes, geometry, packed.data(), packed.size());
    for (std::size_t channel = 0; channel < 3U; ++channel)
      for (std::size_t frame = 0; frame < count; ++frame)
        for (std::size_t y = 0; y < geometry.workingHeight; ++y)
          for (std::size_t x = 0; x < geometry.workingWidth; ++x) {
            const auto input = (std::min(y, geometry.visibleHeight - 1U) * geometry.visibleWidth +
                                std::min(x, geometry.visibleWidth - 1U)) * 3U + channel;
            const auto output = (channel * count + frame) * plane + y * geometry.workingWidth + x;
            require(packed[output] == normalizedRgb(frames[frame][input]),
                    "LQ layout, edge padding or source RGB normalization differs");
          }
    const auto rejected = [&](auto action) {
      std::fill(packed.begin(), packed.end(), 42U);
      expectRejected(action, "invalid LQ input was accepted");
      require(std::all_of(packed.begin(), packed.end(), [](auto value) { return value == 42U; }),
              "invalid LQ input wrote before validation");
    };
    rejected([&] { otc::ai::packFlashLqRgb(rgb, 2U, bytes, geometry, packed.data(), packed.size()); });
    rejected([&] { otc::ai::packFlashLqRgb(rgb, count, bytes - 1U, geometry, packed.data(), packed.size()); });
    rejected([&] { otc::ai::packFlashLqRgb(rgb, count, bytes, geometry, packed.data(), packed.size() - 1U); });
    rejected([&] { otc::ai::packFlashLqRgb(rgb, count, bytes, geometry, nullptr, packed.size()); });
    auto bad = rgb;
    bad[count - 1U] = nullptr;
    rejected([&] { otc::ai::packFlashLqRgb(bad, count, bytes, geometry, packed.data(), packed.size()); });
    bad = rgb;
    bad[0] = reinterpret_cast<const std::uint8_t*>(packed.data());
    rejected([&] { otc::ai::packFlashLqRgb(bad, count, bytes, geometry, packed.data(), packed.size()); });
    auto badGeometry = geometry;
    ++badGeometry.workingHeight;
    rejected([&] { otc::ai::packFlashLqRgb(rgb, count, bytes, badGeometry, packed.data(), packed.size()); });
  }
}

void checkRgbPacking() {
  for (const auto size : {std::array<std::size_t, 2>{16, 24}, {24, 16}, {2, 2}}) {
    const auto geometry = otc::ai::makeFlashGeometry(size[0], size[1]);
    const auto bytes = 3 * geometry.visibleHeight * geometry.visibleWidth;
    const auto plane = geometry.latentHeight * geometry.latentWidth;
    std::array<std::vector<std::uint8_t>, 4> frames;
    std::array<const std::uint8_t*, 4> pointers;
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
      frames[frame].resize(bytes);
      for (std::size_t i = 0; i < bytes; ++i)
        frames[frame][i] = static_cast<std::uint8_t>((17 * i + frame * 11) % 256);
      pointers[frame] = frames[frame].data();
    }
    std::vector<std::uint16_t> latents(16 * 4 * plane);
    for (std::size_t i = 0; i < latents.size(); ++i)
      // Distinguish temporal planes as well as pixels; a period dividing the
      // plane size would fail to catch selecting the wrong inner slice.
      latents[i] = static_cast<std::uint16_t>(0x3e00 + (7 * i + 47 * (i / plane)) % 384);
    std::vector<std::uint16_t> packed(784 * plane);
    for (std::size_t inner = 0; inner < 4; ++inner) {
      otc::ai::packFlashDecoderParentRgb(pointers, bytes, geometry,
          latents.data(), latents.size(), inner, packed.data(), packed.size());
      // Invert channel packing independently: RGB channel, frame, dy, dx.
      for (std::size_t channel = 0; channel < 768; ++channel)
        for (std::size_t y = 0; y < geometry.latentHeight; ++y)
          for (std::size_t x = 0; x < geometry.latentWidth; ++x) {
            const auto row = std::min(8 * y + (channel / 8) % 8, size[0] - 1);
            const auto column = std::min(8 * x + channel % 8, size[1] - 1);
            const auto rgb = frames[(channel / 64) % 4][(row * size[1] + column) * 3 + channel / 256];
            require(packed[channel * plane + y * geometry.latentWidth + x] == normalizedRgb(rgb),
                    "RGB shuffle, padding, or BF16 rounding differs");
          }
      for (std::size_t channel = 0; channel < 16; ++channel)
        for (std::size_t i = 0; i < plane; ++i)
          require(packed[(768 + channel) * plane + i] == latents[(channel * 4 + inner) * plane + i],
                  "decoder latent slice differs");
    }
    pointers.fill(frames[0].data());
    otc::ai::packFlashDecoderParentRgb(pointers, bytes, geometry,
        latents.data(), latents.size(), 0, packed.data(), packed.size());
    for (std::size_t c = 0; c < 3; ++c)
      for (std::size_t frame = 1; frame < 4; ++frame)
        require(std::equal(packed.begin() + c * 256 * plane,
                           packed.begin() + (c * 256 + 64) * plane,
                           packed.begin() + (c * 256 + frame * 64) * plane),
                "startup repeated RGB source differs");
  }
}

void checkRgbPackingRejectsBeforeWrite() {
  const auto geometry = otc::ai::makeFlashGeometry(16, 24);
  const auto plane = geometry.latentHeight * geometry.latentWidth;
  std::vector<std::uint8_t> frame(16 * 24 * 3);
  std::array<const std::uint8_t*, 4> rgb;
  rgb.fill(frame.data());
  std::vector<std::uint16_t> latents(16 * 4 * plane, 0x3f80);
  std::vector<std::uint16_t> packed(784 * plane, 0xabcd);
  auto invoke = [&](const auto& sources, std::size_t bytes, const auto& geo,
                    const std::uint16_t* latent, std::size_t count,
                    std::size_t inner, std::uint16_t* output, std::size_t output_count) {
    expectRejected([&] {
      otc::ai::packFlashDecoderParentRgb(sources, bytes, geo, latent, count, inner, output, output_count);
    }, "malformed RGB pack was accepted");
    require(std::all_of(packed.begin(), packed.end(), [](auto word) { return word == 0xabcd; }),
            "malformed RGB pack partly wrote output");
  };
  invoke(rgb, frame.size() - 1, geometry, latents.data(), latents.size(), 0, packed.data(), packed.size());
  invoke(rgb, frame.size(), geometry, latents.data(), latents.size() - 1, 0, packed.data(), packed.size());
  invoke(rgb, frame.size(), geometry, latents.data(), latents.size(), 4, packed.data(), packed.size());
  invoke(rgb, frame.size(), geometry, nullptr, latents.size(), 0, packed.data(), packed.size());
  invoke(rgb, frame.size(), geometry, latents.data(), latents.size(), 0, nullptr, packed.size());
  invoke(rgb, frame.size(), geometry, latents.data(), latents.size(), 0, packed.data(), packed.size() - 1);
  latents.back() = 0x7fc1; // Invalid even though inner=0 does not consume this word.
  invoke(rgb, frame.size(), geometry, latents.data(), latents.size(), 0, packed.data(), packed.size());
  latents.back() = 0x3f80;
  auto invalid = rgb;
  invalid[3] = nullptr;
  invoke(invalid, frame.size(), geometry, latents.data(), latents.size(), 0, packed.data(), packed.size());
  invalid[3] = reinterpret_cast<const std::uint8_t*>(packed.data() + 1);
  invoke(invalid, frame.size(), geometry, latents.data(), latents.size(), 0, packed.data(), packed.size());
  invoke(rgb, frame.size(), geometry, packed.data(), latents.size(), 0, packed.data(), packed.size());
  auto bad = geometry;
  ++bad.workingWidth;
  invoke(rgb, frame.size(), bad, latents.data(), latents.size(), 0, packed.data(), packed.size());
  bad = geometry;
  bad.latentHeight = 0;
  invoke(rgb, frame.size(), bad, latents.data(), latents.size(), 0, packed.data(), packed.size());
  for (const auto member : {&otc::ai::FlashGeometry::tokenHeight, &otc::ai::FlashGeometry::tokenWidth,
                           &otc::ai::FlashGeometry::spatialTokens, &otc::ai::FlashGeometry::spatialWindows}) {
    bad = geometry;
    ++(bad.*member);
    invoke(rgb, frame.size(), bad, latents.data(), latents.size(), 0, packed.data(), packed.size());
  }
  bad.visibleHeight = std::numeric_limits<std::size_t>::max();
  invoke(rgb, frame.size(), bad, latents.data(), latents.size(), 0, packed.data(), packed.size());
}

}  // namespace

int main() {
  try {
    checkTileDescriptors();
    checkAllFiniteWords();
    checkOnlyCoreAndBottomCrop();
    checkMalformedInputs();
    checkLqRgbPacking();
    checkRgbPacking();
    checkRgbPackingRejectsBeforeWrite();
    std::cout << "flash_decoder_frame=pass descriptors=16 all_finite_bf16_words=65024 "
                 "core_only=pass bottom_crop=pass malformed=pass rgb_pack=pass core_calls=0 "
                 "device_calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "flash_decoder_frame=fail: " << error.what() << '\n';
    return 1;
  }
}
