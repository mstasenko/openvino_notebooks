#include "flash_geometry.hpp"

#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void rejects(const std::function<void()>& action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void testCanonicalUhd() {
  const auto visible = otc::ai::makeFlashGeometry(2160U, 3840U);
  assert(visible.workingHeight == 2176U && visible.workingWidth == 3840U);
  assert(visible.bottomPadding() == 16U && visible.rightPadding() == 0U);
  assert(visible.trimmedHeight() == 2160U && visible.trimmedWidth() == 3840U);
  const auto geometry = otc::ai::makeFlashGeometry(2176U, 3840U);
  assert(geometry.visibleHeight == 2176U && geometry.visibleWidth == 3840U);
  assert(geometry.workingHeight == 2176U && geometry.workingWidth == 3840U);
  assert(geometry.tokenHeight == 136U && geometry.tokenWidth == 240U);
  assert(geometry.latentHeight == 272U && geometry.latentWidth == 480U);
  assert(geometry.spatialTokens == 32640U && geometry.spatialWindows == 510U);
  assert(geometry.bottomPadding() == 0U && geometry.rightPadding() == 0U);
  assert(geometry.trimmedHeight() == 2176U && geometry.trimmedWidth() == 3840U);
}

void testPortraitAndLandscapePadding() {
  const auto tall = otc::ai::makeFlashGeometry(3840U, 2160U);
  assert(tall.workingHeight == 3840U && tall.workingWidth == 2176U);
  assert(tall.bottomPadding() == 0U && tall.rightPadding() == 16U);
  assert(tall.tokenHeight == 240U && tall.tokenWidth == 136U);
  assert(tall.spatialTokens == 32640U && tall.spatialWindows == 510U);
  const auto portrait = otc::ai::makeFlashGeometry(1080U, 1920U);
  assert(portrait.workingHeight == 1152U && portrait.workingWidth == 1920U);
  assert(portrait.tokenHeight == 72U && portrait.tokenWidth == 120U);
  assert(portrait.latentHeight == 144U && portrait.latentWidth == 240U);
  assert(portrait.spatialTokens == 8640U && portrait.spatialWindows == 135U);
  assert(portrait.bottomPadding() == 72U && portrait.rightPadding() == 0U);
  assert(portrait.trimmedHeight() == 1080U && portrait.trimmedWidth() == 1920U);

  const auto landscape = otc::ai::makeFlashGeometry(720U, 1280U);
  assert(landscape.workingHeight == 768U && landscape.workingWidth == 1280U);
  assert(landscape.tokenHeight == 48U && landscape.tokenWidth == 80U);
  assert(landscape.latentHeight == 96U && landscape.latentWidth == 160U);
  assert(landscape.spatialTokens == 3840U && landscape.spatialWindows == 60U);
  assert(landscape.bottomPadding() == 48U && landscape.rightPadding() == 0U);

  const auto tiny = otc::ai::makeFlashGeometry(2U, 2U);
  assert(tiny.workingHeight == 128U && tiny.workingWidth == 128U);
  assert(tiny.tokenHeight == 8U && tiny.tokenWidth == 8U);
  assert(tiny.latentHeight == 16U && tiny.latentWidth == 16U);
  assert(tiny.spatialTokens == 64U && tiny.spatialWindows == 1U);
  assert(tiny.bottomPadding() == 126U && tiny.rightPadding() == 126U);
}

void testOutputPolicy() {
  const auto portrait = otc::ai::makeFlashOutputGeometry(1920U, 1080U);
  assert(portrait.spatial && portrait.outputHeight == 3840U && portrait.outputWidth == 2160U);

  const auto landscape = otc::ai::makeFlashOutputGeometry(720U, 1280U);
  assert(landscape.spatial && landscape.outputHeight == 2160U && landscape.outputWidth == 3840U);

  const auto rounded = otc::ai::makeFlashOutputGeometry(1000U, 1600U);
  assert(rounded.spatial && rounded.outputHeight == 2400U && rounded.outputWidth == 3840U);

  const auto oddSource = otc::ai::makeFlashOutputGeometry(719U, 1280U);
  assert(oddSource.spatial && oddSource.outputHeight == 2158U && oddSource.outputWidth == 3840U);

  const auto halfTie = otc::ai::makeFlashOutputGeometry(11U, 2816U);
  assert(halfTie.outputHeight == 14U && halfTie.outputWidth == 3840U);
  const auto tallHalfTie = otc::ai::makeFlashOutputGeometry(2816U, 11U);
  assert(tallHalfTie.outputHeight == 3840U && tallHalfTie.outputWidth == 14U);

  const auto exact = otc::ai::makeFlashOutputGeometry(2160U, 3840U);
  assert(!exact.spatial && exact.outputHeight == 2160U && exact.outputWidth == 3840U);

  // The application preserves dimensions already above the target.  The
  // separate FlashGeometry contract must still reject such fixed-IR work.
  const auto larger = otc::ai::makeFlashOutputGeometry(4000U, 2000U);
  assert(!larger.spatial && larger.outputHeight == 4000U && larger.outputWidth == 2000U);
  rejects([] { (void)otc::ai::makeFlashGeometry(4000U, 2000U); });
}

void testMalformedGeometry() {
  rejects([] { (void)otc::ai::makeFlashGeometry(0U, 128U); });
  rejects([] { (void)otc::ai::makeFlashGeometry(719U, 1280U); });
  rejects([] { (void)otc::ai::makeFlashGeometry(2176U, 3840U + 2U); });
  rejects([] { (void)otc::ai::makeFlashGeometry(3840U, 3840U); });
  rejects([] { (void)otc::ai::makeFlashGeometry(std::numeric_limits<std::size_t>::max(), 128U); });
}

}  // namespace

int main() {
  try {
    testCanonicalUhd();
    testPortraitAndLandscapePadding();
    testOutputPolicy();
    rejects([] { (void)otc::ai::makeFlashOutputGeometry(100001U, 1280U); });
    testMalformedGeometry();
    std::cout << "flash geometry: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
