#pragma once

#include <openvino/openvino.hpp>

#include <cstddef>

namespace otc::ai {

enum class FlashDecoderRole { front, grow, frame };

// The fixed UHD decoder uses a full-frame front, tiled growth-history stage,
// and tiled frame-child stage.
// Validate the authenticated external-weight ABI before compiling any request;
// this seam does not admit arbitrary geometry or query a device.
void validateFlashDecoderModel(const ov::Model& model, FlashDecoderRole role,
                               std::size_t height, std::size_t width);

}  // namespace otc::ai
