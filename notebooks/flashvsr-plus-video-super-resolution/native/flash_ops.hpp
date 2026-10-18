#pragma once

#include <openvino/openvino.hpp>

#include <cstddef>
#include <memory>

namespace otc::ai {

// Registers the fixed Flash attention/bridge custom-operation ABI.  Registration does not
// query a device, compile a model, or provide a numerical implementation.
void registerFlashOps(ov::Core& core);

// Build the qualified C16/four-way split-key self-attention graph.  The model
// contains only static parameters and custom operations; it performs no Core,
// device, compilation, or inference work.  K/V and the ordered-ID capacity
// are the same bounded value, with raw BF16 bits carried in F16 ports.
std::shared_ptr<ov::Model> makeFlashSplitKeyAttentionModel(
    std::size_t maxKeys = 2550U);

}  // namespace otc::ai
