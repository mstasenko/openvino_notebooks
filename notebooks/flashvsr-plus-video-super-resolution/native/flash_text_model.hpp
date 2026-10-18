#pragma once

#include <openvino/openvino.hpp>

namespace otc::ai {

// Validate the fixed BF16 prompt-to-context graph before a caller binds the
// four external text MLP weights.  This is structural validation only: it
// neither compiles the graph nor queries a device.
void validateFlashTextModel(const ov::Model& model);

}  // namespace otc::ai
