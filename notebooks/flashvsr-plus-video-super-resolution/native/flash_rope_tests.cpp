#include "flash_rope.hpp"

#include "flash_stream.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using otc::ai::FlashGeometry;

template <typename Function>
void rejects(const Function& function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  assert(rejected);
}

void testCanonicalShapesAndAxes() {
  const auto landscape = otc::ai::makeFlashGeometry(2160U, 3840U);
  const auto portrait = otc::ai::makeFlashGeometry(3840U, 2160U);
  const auto tiny = otc::ai::makeFlashGeometry(16U, 32U);
  assert(landscape.tokenHeight == 136U && landscape.tokenWidth == 240U);
  assert(portrait.tokenHeight == 240U && portrait.tokenWidth == 136U);
  assert(otc::ai::flashRopeElements(landscape) == 4U * 136U * 240U * 64U);
  assert(otc::ai::flashRopeElements(portrait) == 4U * 240U * 136U * 64U);
  assert(otc::ai::flashRopeElements(tiny) == 4U * 8U * 8U * 64U);

  const auto elements = otc::ai::flashRopeElements(tiny);
  std::vector<float> cos(elements, -2.0F), sin(elements, -3.0F);
  otc::ai::fillFlashRopeInputs(tiny, 0U, cos.data(), cos.size(), sin.data(), sin.size());
  // The first source token is position (frame,row,column)=(0,0,0): all
  // angles are zero, so every cosine is one and every sine is zero.
  for (std::size_t pair = 0U; pair < 64U; ++pair) {
    assert(cos[pair] == 1.0F);
    assert(sin[pair] == 0.0F);
  }
  for (float value : cos) assert(std::isfinite(value));
  for (float value : sin) assert(std::isfinite(value));
}

void testCallOffsetsAndCallerContracts() {
  const auto geometry = otc::ai::makeFlashGeometry(16U, 32U);
  const auto elements = otc::ai::flashRopeElements(geometry);
  std::vector<float> cos(elements), sin(elements);
  otc::ai::fillFlashRopeInputs(geometry, 0U, cos.data(), cos.size(), sin.data(), sin.size());
  const auto first = cos;
  otc::ai::fillFlashRopeInputs(geometry, 1U, cos.data(), cos.size(), sin.data(), sin.size());
  assert(cos != first);
  // The old 1024-position cache admitted calls only through 255. Calls past
  // that boundary must retain the same global position formula instead of
  // wrapping or restarting the temporal axis.
  otc::ai::fillFlashRopeInputs(geometry, 255U, cos.data(), cos.size(),
                               sin.data(), sin.size());
  const auto call255 = cos;
  otc::ai::fillFlashRopeInputs(geometry, 256U, cos.data(), cos.size(),
                               sin.data(), sin.size());
  assert(cos != call255);
  const otc::ai::flash::StreamPlan maximumPlan(
      otc::ai::flash::kMaxSourceFrames);
  const auto lastCall = maximumPlan.callCount() - 1U;
  otc::ai::fillFlashRopeInputs(geometry, lastCall, cos.data(), cos.size(),
                               sin.data(), sin.size());
  for (float value : cos) assert(std::isfinite(value));
  for (float value : sin) assert(std::isfinite(value));
  const auto sentinel = cos;
  rejects([&] { otc::ai::fillFlashRopeInputs(geometry, lastCall + 1U,
                                             cos.data(), cos.size(), sin.data(),
                                             sin.size()); });
  assert(cos == sentinel);
  rejects([&] { otc::ai::fillFlashRopeInputs(
      geometry, std::numeric_limits<std::size_t>::max(), cos.data(), cos.size(),
      sin.data(), sin.size()); });
  assert(cos == sentinel);
  rejects([&] { otc::ai::fillFlashRopeInputs(geometry, 0U, cos.data(), cos.size() - 1U,
                                             sin.data(), sin.size()); });
  rejects([&] { otc::ai::fillFlashRopeInputs(geometry, 0U, cos.data(), cos.size(),
                                             cos.data(), cos.size()); });
  rejects([&] { otc::ai::fillFlashRopeInputs(geometry, 0U, nullptr, cos.size(),
                                             sin.data(), sin.size()); });

  auto malformed = geometry;
  ++malformed.tokenHeight;
  const auto malformedSentinel = cos;
  rejects([&] { otc::ai::fillFlashRopeInputs(malformed, 0U, cos.data(), cos.size(),
                                             sin.data(), sin.size()); });
  assert(cos == malformedSentinel);
}

float referenceRopeValue(std::size_t position, std::size_t pair, bool sine) {
  const double exponent = static_cast<double>(2U * pair) / 44.0;
  const double frequency = 1.0 / std::pow(10000.0, exponent);
  const double phase = static_cast<double>(position) * frequency;
  return static_cast<float>(sine ? std::sin(phase) : std::cos(phase));
}

void testLongVideoPositionsUseTheIndependentSourceFormula() {
  const auto geometry = otc::ai::makeFlashGeometry(16U, 32U);
  const auto elements = otc::ai::flashRopeElements(geometry);
  const otc::ai::flash::StreamPlan maximumPlan(
      otc::ai::flash::kMaxSourceFrames);
  const auto lastCall = maximumPlan.callCount() - 1U;
  for (const auto call : {std::size_t{256U}, std::size_t{1024U}, lastCall}) {
    std::vector<float> cos(elements), sin(elements);
    otc::ai::fillFlashRopeInputs(geometry, call, cos.data(), cos.size(),
                                 sin.data(), sin.size());
    for (std::size_t frame = 0U; frame < otc::ai::kFlashRopeFrames; ++frame) {
      const auto token = frame * geometry.spatialTokens;
      const auto base = token * otc::ai::kFlashRopePairs;
      const auto position = call * otc::ai::kFlashRopeFrames + frame;
      for (std::size_t pair = 0U; pair < otc::ai::kFlashRopeTemporalPairs;
           ++pair) {
        assert(cos[base + pair] == referenceRopeValue(position, pair, false));
        assert(sin[base + pair] == referenceRopeValue(position, pair, true));
      }
    }
  }
}

}  // namespace

int main() {
  testCanonicalShapesAndAxes();
  testCallOffsetsAndCallerContracts();
  testLongVideoPositionsUseTheIndependentSourceFormula();
  return 0;
}
