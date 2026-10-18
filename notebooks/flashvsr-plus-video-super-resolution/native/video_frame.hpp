#pragma once

#include <cstdint>
#include <vector>

namespace otc::ai {

struct RgbFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;
};

struct SourceFrame {
  std::uint64_t index = 0;
  RgbFrame frame;
  bool cutBefore = false;
};

}  // namespace otc::ai
