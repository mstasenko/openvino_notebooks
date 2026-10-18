#pragma once

#include "flash_mode.hpp"
#include "seedvr_assets.hpp"

#include <openvino/openvino.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace otc::ai {

enum class FlashOutputCheck { none, bf16, bf16Readback };
// F16 is a two-byte transport view only for the qualified split-key kernel;
// it must never narrow BF16 activations through numeric FP16 arithmetic.
enum class FlashPrecision { pluginDefault, source, rawBf16Transport };
using FlashPorts = std::map<std::string, std::pair<ov::element::Type, ov::Shape>>;
using FlashTensors = std::map<std::string, ov::Tensor>;

// Inspect storage bits without narrowing BF16 values to FP16.
void validateFlashBf16Output(const std::uint16_t* words, std::size_t elements);

// Validate the BF16 byte extent shared by generated bindings and deserialized
// stage contracts.
void validateFlashWeightExtent(const ov::Shape& shape,
                               const SeedVrWeightSpan& span);

// Keep activations and learned banks in the same device storage as oneDNN.
// Mixed USM/OpenCL-buffer direct copies are broken in the pinned runtime.
// Reusing a USM_DEVICE_BUFFER parameter map creates fresh storage, not a
// typed alias of its MEM_HANDLE. Use explicit copies for controller handoffs.
ov::Tensor flashDeviceTensor(ov::RemoteContext& context,
                             ov::element::Type type, const ov::Shape& shape);

template<class Ports>
void validateFlashPorts(const Ports& ports, const FlashPorts& expected,
                        bool allowCompilerAliases = false) {
  if (ports.size() != expected.size()) throw std::runtime_error("FlashVSR+ port count differs");
  std::set<std::string> seen;
  for (const auto& port : ports) {
    std::string name;
    for (const auto& candidate : port.get_names()) {
      if (!expected.count(candidate)) continue;
      if (!name.empty()) throw std::runtime_error("FlashVSR+ port has ambiguous public names");
      name = candidate;
    }
    // Source IR has exactly one canonical name. The compiler may add private
    // aliases (e.g. mem_256_0_out_1), but must retain exactly one public name.
    if (name.empty() || (!allowCompilerAliases && port.get_names().size() != 1)) {
      std::string names;
      for (const auto& name : port.get_names()) names += " " + name;
      throw std::runtime_error("FlashVSR+ port name is not canonical:" + names);
    }
    const auto found = expected.find(name);
    if (found == expected.end() || !seen.insert(name).second ||
        port.get_element_type() != found->second.first || !port.get_partial_shape().is_static() ||
        port.get_shape() != found->second.second)
      throw std::runtime_error("FlashVSR+ port type/shape differs: " + name);
  }
}

template<class Binding>
ov::Shape flashWeightShape(const Binding& weight) {
  if (!weight.rank || weight.rank > weight.dimensions.size())
    throw std::runtime_error("FlashVSR+ weight rank differs");
  ov::Shape shape(weight.dimensions.begin(), weight.dimensions.begin() + weight.rank);
  validateFlashWeightExtent(shape, weight.span);
  return shape;
}

struct FlashStageWeight {
  std::string name;
  SeedVrWeightSpan span;
  ov::Shape shape;
};

// This seam is also checked without creating a Core or touching a device.
void validateFlashStageWeights(const FlashPorts& inputs,
                               const std::vector<FlashStageWeight>& weights);

template<class Bindings>
std::vector<FlashStageWeight> flashStageWeights(const Bindings& bindings) {
  std::vector<FlashStageWeight> result;
  for (const auto& weight : bindings)
    result.push_back({weight.port_name, weight.span, flashWeightShape(weight)});
  return result;
}

template<class Bindings>
std::vector<SeedVrWeightSpan> flashWeightSpans(const Bindings& bindings) {
  std::vector<SeedVrWeightSpan> result;
  for (const auto& weight : bindings) {
    flashWeightShape(weight);
    result.push_back(weight.span);
  }
  return result;
}

// Shared Flash request mechanics only. The role-specific wrapper authenticates
// IR/BIN bytes, validates its graph, and keeps the structural owner alive. The
// caller owns admission and the resolved context; none of this API is IPC-facing.
class FlashGpuStage final {
 public:
  FlashGpuStage(ov::Core& core, ov::RemoteContext context,
                const std::shared_ptr<ov::Model>& model,
                std::vector<FlashStageWeight> weights,
                FlashPrecision precision,
                const std::string& kernelConfig = {},
                const std::vector<std::string>& finiteFlags = {},
                FlashOutputCheck outputCheck = FlashOutputCheck::none);

  void invalidate() noexcept {
    ready_ = false;
    for (auto& [shape, check] : outputChecks_) check->invalidate();
  }
  // Weighted graphs require bind before run. Weight-free graphs are ready
  // immediately after construction; failed requests still invalidate them.
  void bind(const SeedVrOwnedIr& owner);
  // Release the compiled graph, request, and recursive finite-check workspaces
  // while retaining the authenticated model, context, and learned device
  // banks. The next bind/run lazily rebuilds the exact same compiled policy.
  void suspend() noexcept;
  // Non-flag tensors are complete USM device or host roots. Outputs must be
  // distinct from every input and from each other; read-only inputs may alias.
  // Scalar finite flags stay host-side. bf16Readback trades a reusable host
  // buffer for the large GPU reduction workspace, without moving state roots.
  void run(const FlashTensors& inputs, const FlashTensors& outputs);

 private:
  void ensureCompiled();
  void* validateTensor(const ov::Tensor& tensor,
                       const std::pair<ov::element::Type, ov::Shape>& expected) const;

  ov::Core core_;
  ov::RemoteContext context_;
  void* contextMarker_ = nullptr;
  std::shared_ptr<ov::Model> model_;
  FlashPorts inputs_, outputs_;
  std::vector<FlashStageWeight> bindings_;
  FlashPrecision precision_ = FlashPrecision::pluginDefault;
  ov::AnyMap compileConfig_;
  FlashTensors weights_, flags_;
  ov::CompiledModel compiled_;
  ov::InferRequest request_;
  // Reuse one single-input checker per shape. A many-input UHD checker keeps
  // all masks/reduction scratch resident at once and can exhaust the GPU.
  std::map<ov::Shape, std::unique_ptr<FlashGpuStage>> outputChecks_;
  FlashOutputCheck outputCheck_;
  std::vector<std::uint16_t> readback_;
  bool ready_ = false;
};

} // namespace otc::ai
