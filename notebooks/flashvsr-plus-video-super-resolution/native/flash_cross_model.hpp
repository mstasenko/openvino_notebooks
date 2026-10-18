#pragma once

#include <openvino/openvino.hpp>

namespace otc::ai {

// Validate the fixed external-weight Cross/K/V graph before a caller binds
// learned data.  This is a structural check only: it does not compile, infer,
// query a device, or add a finite-state sidecar.
void validateFlashCrossModel(const ov::Model& model);

}  // namespace otc::ai
