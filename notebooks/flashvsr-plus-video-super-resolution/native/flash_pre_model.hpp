#pragma once

#include <openvino/openvino.hpp>

namespace otc::ai {

// Structural validation only: no compilation, device query, or inference.
// The prepared model has the twelve public PRE outputs and six owning BF16
// Assign sinks described by the fixed 4096-row ABI.
void validateFlashPreModel(const ov::Model& model);

// Validate the ordinary 13-input/18-Result IR and replace only its six
// private BF16 Result barriers with owning Assign sinks.  No ReadValue
// feedback is introduced and all public graph math remains unchanged.
void prepareFlashPreModel(ov::Model& model);

}  // namespace otc::ai
