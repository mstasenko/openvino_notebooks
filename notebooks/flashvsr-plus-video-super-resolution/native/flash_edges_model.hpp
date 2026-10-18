#pragma once

#include <openvino/openvino.hpp>

#include <cstddef>

namespace otc::ai {

enum class FlashEdgesRole { prefix, suffix };

// Validate one fixed-UHD external-weight edge graph before compilation.  The
// dimensions are the padded working frame (2176x3840); this is structural
// validation only and does not compile, infer, or query a device.
void validateFlashEdgesModel(const ov::Model& model, FlashEdgesRole role,
                             std::size_t height, std::size_t width);

}  // namespace otc::ai
