#pragma once

#include "flash_stage.hpp"

#include <openvino/openvino.hpp>

#include <memory>

namespace otc::ai {

// Shared fixed hygiene for the stateless external-weight graphs. Raw F16
// storage is confined to the registered bit-preserving attention transport;
// numeric F16 operations are rejected. Role
// validators construct their exact contracts and retain any role-specific
// friendly-name/topology checks around this common guard.
void validateFlashStructuralModel(const ov::Model& model,
                                  const FlashPorts& expectedInputs,
                                  const FlashPorts& expectedOutputs);

// Check completed BF16 output roots in a separate, weight-free request. The
// data model is never mutated: an inline predicate fanout prevents the pinned
// GPU plugin from publishing the LQ cache Results into their bound buffers.
std::shared_ptr<ov::Model> makeFlashBf16OutputCheck(const ov::Model& dataModel);

// Build the same one-input checker for a single caller-owned BF16 data root.
// This overload is used to keep large output checks independent and bounded.
std::shared_ptr<ov::Model> makeFlashBf16OutputCheck(const ov::Shape& shape);

}  // namespace otc::ai
