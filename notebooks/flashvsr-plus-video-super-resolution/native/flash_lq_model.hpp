#pragma once

#include <openvino/openvino.hpp>

#include <cstddef>

namespace otc::ai {

enum class FlashLqRole { startup, prefix, suffix };

// Validate an authenticated, external-weight LQ graph before compilation.
// Dimensions describe the padded working frame, not the visible video crop.
// This function neither compiles a model nor queries a device.
void validateFlashLqModel(const ov::Model& model, FlashLqRole role,
                         std::size_t height, std::size_t width);

}  // namespace otc::ai
