#include "flash_stage.hpp"
#include "flash_model.hpp"

#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

#include <algorithm>
#include <limits>

namespace otc::ai {
namespace {

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error("FlashVSR+ GPU stage: " + message);
}

void* marker(const ov::AnyMap& parameters, const char* name) {
  const auto found = parameters.find(name);
  require(found != parameters.end(), std::string("missing remote marker ") + name);
  void* value = found->second.as<void*>();
  require(value != nullptr, std::string("empty remote marker ") + name);
  return value;
}

template<class Ports>
FlashPorts describe(const Ports& ports) {
  FlashPorts result;
  for (const auto& port : ports) {
    require(port.get_names().size() == 1 && port.get_partial_shape().is_static(),
            "static canonical ports required");
    require(result.emplace(port.get_any_name(),
                           std::make_pair(port.get_element_type(), port.get_shape())).second,
            "duplicate port");
  }
  return result;
}

} // namespace

void validateFlashBf16Output(const std::uint16_t* words, std::size_t elements) {
  require(words && elements, "empty BF16 output");
  // Accumulate instead of branching per word so the compiler can vectorize
  // the scan. Every NaN/Inf bit pattern is still rejected before publication.
  std::uint16_t nonFinite = 0;
  for (std::size_t i = 0; i < elements; ++i)
    nonFinite |= (words[i] & 0x7f80U) == 0x7f80U;
  if (nonFinite)
    throw std::runtime_error("FlashVSR+ GPU stage: non-finite BF16 output");
}

ov::Tensor flashDeviceTensor(ov::RemoteContext& context,
                             ov::element::Type type, const ov::Shape& shape) {
  return context.create_tensor(type, shape,
      {ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER)});
}

void validateFlashWeightExtent(const ov::Shape& shape,
                               const SeedVrWeightSpan& span) {
  require(!shape.empty(), "weight must not be scalar");
  std::uint64_t elements = 1;
  for (const auto dimension : shape) {
    require(dimension != 0 &&
                dimension <= std::numeric_limits<std::uint64_t>::max() / elements,
            "weight shape overflows");
    elements *= dimension;
  }
  require(elements == span.elements &&
              elements <= std::numeric_limits<std::uint64_t>::max() / 2U &&
              elements * 2U == span.bytes,
          "weight span differs");
}

void validateFlashStageWeights(const FlashPorts& inputs,
                               const std::vector<FlashStageWeight>& weights) {
  std::set<std::string> weightNames;
  for (const auto& weight : weights) {
    validateFlashWeightExtent(weight.shape, weight.span);
    const auto found = inputs.find(weight.name);
    require(found != inputs.end() &&
                found->second == std::make_pair(ov::element::bf16, weight.shape) &&
                weight.name.rfind("weight__", 0) == 0 && weightNames.insert(weight.name).second,
            "weight binding differs");
  }
  for (const auto& [name, contract] : inputs)
    require(name.rfind("weight__", 0) != 0 || weightNames.count(name), "unbound learned input " + name);
}

FlashGpuStage::FlashGpuStage(ov::Core& core, ov::RemoteContext context,
                             const std::shared_ptr<ov::Model>& model,
                             std::vector<FlashStageWeight> weights,
                             FlashPrecision precision,
                             const std::string& kernelConfig,
                             const std::vector<std::string>& finiteFlags,
                             FlashOutputCheck outputCheck)
    : core_(core),
      context_(std::move(context)),
      model_(model),
      bindings_(std::move(weights)),
      precision_(precision),
      outputCheck_(outputCheck) {
  require(model != nullptr, "missing model");
  require(precision == FlashPrecision::pluginDefault || precision == FlashPrecision::source ||
              precision == FlashPrecision::rawBf16Transport,
          "unknown precision policy");
  require(outputCheck == FlashOutputCheck::none || outputCheck == FlashOutputCheck::bf16 ||
              outputCheck == FlashOutputCheck::bf16Readback,
          "unknown output validation policy");
  require(outputCheck == FlashOutputCheck::none || finiteFlags.empty(),
          "separate output validation cannot also use inline flags");
  inputs_ = describe(model->inputs());
  outputs_ = describe(model->outputs());
  if (outputCheck != FlashOutputCheck::none) {
    require(!outputs_.empty(), "finite check requires data outputs");
    for (const auto& [name, contract] : outputs_)
      require(name != "finite" && contract.first == ov::element::bf16,
              "finite check requires BF16 data with non-reserved names");
  }
  validateFlashStageWeights(inputs_, bindings_);
  std::set<std::string> flagNames;
  for (const auto& name : finiteFlags) {
    const auto found = outputs_.find(name);
    require(found != outputs_.end() && found->second == std::make_pair(ov::element::f16, ov::Shape{}) &&
                flagNames.insert(name).second, "finite flag must be a unique scalar F16 output");
  }
  const auto device = context_.get_device_name();
  require(device.rfind("GPU.", 0) == 0 && device.size() > 4 &&
              std::all_of(device.begin() + 4, device.end(), [](char c) { return c >= '0' && c <= '9'; }),
          "an explicitly resolved GPU context is required");
  require(std::string(ov::get_openvino_version().buildNumber) ==
              "2026.5.0-23457-7d81aad55a5", "unqualified OpenVINO build");
  contextMarker_ = marker(context_.get_params(), "OCL_CONTEXT");
  compileConfig_ = ov::AnyMap{ov::num_streams(ov::streams::Num(1)), {"CACHE_DIR", std::string()},
      // Request the Intel driver's low-power wait policy, not queue priority.
      ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::LOW),
      {"PERFORMANCE_HINT", ov::hint::PerformanceMode::LATENCY},
      {"EXECUTION_MODE_HINT", ov::hint::ExecutionMode::ACCURACY}};
  // POST disables implicit precision changes; PRE uses the plugin default.
  // Split-key attention uses F16-shaped ports containing raw BF16 words.
  const auto requestedPrecision = precision_ == FlashPrecision::rawBf16Transport
      ? ov::element::f16 : ov::element::dynamic;
  if (precision_ != FlashPrecision::pluginDefault)
    compileConfig_["INFERENCE_PRECISION_HINT"] = requestedPrecision;
  if (!kernelConfig.empty()) compileConfig_["CONFIG_FILE"] = kernelConfig;
  for (const auto& weight : bindings_)
    weights_.emplace(weight.name, flashDeviceTensor(context_, ov::element::bf16, weight.shape));
  for (const auto& name : finiteFlags)
    flags_.emplace(name, ov::Tensor(ov::element::f16, {}));
  // Build each output checker once. Its compiled request is independently
  // suspended/resumed with this stage; only its GPU workspace is discarded.
  if (outputCheck_ == FlashOutputCheck::bf16) {
    for (const auto& [name, contract] : outputs_) {
      if (outputChecks_.count(contract.second)) continue;
      auto check = std::make_unique<FlashGpuStage>(
          core_, context_, makeFlashBf16OutputCheck(contract.second),
          std::vector<FlashStageWeight>{}, FlashPrecision::pluginDefault,
          std::string{}, std::vector<std::string>{"finite"});
      outputChecks_.emplace(contract.second, std::move(check));
    }
  }
  // Authenticated weighted stages become ready only after bind(); finite-check
  // and other weight-free stages are valid immediately after construction.
  ready_ = bindings_.empty();
  ensureCompiled();
}

void FlashGpuStage::ensureCompiled() {
  if (compiled_) return;
  const auto device = context_.get_device_name();
  const auto requestedPrecision = precision_ == FlashPrecision::rawBf16Transport
      ? ov::element::f16 : ov::element::dynamic;
  try {
    compiled_ = core_.compile_model(model_, context_, compileConfig_);
    require(compiled_.get_property("EXECUTION_DEVICES").as<std::vector<std::string>>() ==
                std::vector<std::string>{device}, "compiled execution device differs");
    require(static_cast<std::int32_t>(compiled_.get_property(ov::num_streams)) == 1 &&
                compiled_.get_property("EXECUTION_MODE_HINT").as<ov::hint::ExecutionMode>() ==
                    ov::hint::ExecutionMode::ACCURACY &&
                (precision_ == FlashPrecision::pluginDefault ||
                 compiled_.get_property("INFERENCE_PRECISION_HINT").as<ov::element::Type>() ==
                     requestedPrecision), "compiled stream/precision policy differs");
    validateFlashPorts(compiled_.inputs(), inputs_, true);
    validateFlashPorts(compiled_.outputs(), outputs_, true);
    request_ = compiled_.create_infer_request();
    for (const auto& weight : bindings_) {
      request_.set_tensor(weight.name, weights_.at(weight.name));
    }
    for (const auto& [name, tensor] : flags_)
      request_.set_tensor(name, tensor);
    for (auto& [shape, check] : outputChecks_) check->ensureCompiled();
  } catch (...) {
    ready_ = false;
    request_ = ov::InferRequest{};
    compiled_ = ov::CompiledModel{};
    for (auto& [shape, check] : outputChecks_) check->suspend();
    throw;
  }
}

void FlashGpuStage::suspend() noexcept {
  request_ = ov::InferRequest{};
  compiled_ = ov::CompiledModel{};
  std::vector<std::uint16_t>().swap(readback_);
  for (auto& [shape, check] : outputChecks_) check->suspend();
}

void FlashGpuStage::bind(const SeedVrOwnedIr& owner) {
  ensureCompiled();
  invalidate();
  for (const auto& binding : bindings_) {
    const ov::Tensor host(ov::element::bf16, binding.shape, owner.weightData(binding.span));
    auto destination = weights_.at(binding.name).as<ov::RemoteTensor>();
    destination.copy_from(host);
  }
  for (auto& [shape, check] : outputChecks_) check->bind(owner); // No learned banks.
  ready_ = true;
}

void* FlashGpuStage::validateTensor(
    const ov::Tensor& tensor, const std::pair<ov::element::Type, ov::Shape>& expected) const {
  require(tensor && tensor.is<ov::RemoteTensor>() && tensor.get_element_type() == expected.first &&
              tensor.get_shape() == expected.second, "complete typed remote root required");
  std::size_t bytes = expected.first.size();
  require(bytes > 0, "remote storage type has no byte extent");
  for (auto dimension : expected.second) {
    require(dimension && dimension <= std::numeric_limits<std::size_t>::max() / bytes,
            "remote byte extent overflows");
    bytes *= dimension;
  }
  require(tensor.get_byte_size() == bytes, "remote byte extent differs");
  const auto remote = tensor.as<ov::RemoteTensor>();
  require(remote.get_device_name() == context_.get_device_name(), "remote device differs");
  const auto params = remote.get_params();
  require(marker(params, "OCL_CONTEXT") == contextMarker_, "remote context differs");
  const auto storage = params.find(ov::intel_gpu::shared_mem_type.name());
  require(storage != params.end() &&
              (storage->second.as<ov::intel_gpu::SharedMemType>() == ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER ||
               storage->second.as<ov::intel_gpu::SharedMemType>() == ov::intel_gpu::SharedMemType::USM_HOST_BUFFER),
          "USM storage required");
  return marker(params, "MEM_HANDLE");
}

void FlashGpuStage::run(const FlashTensors& inputs, const FlashTensors& outputs) {
  require(ready_, "no completely bound layer");
  require(inputs.size() == inputs_.size() - weights_.size() &&
              outputs.size() == outputs_.size() - flags_.size(), "feed/output count differs");
  std::set<void*> inputHandles, outputHandles;
  for (const auto& [name, tensor] : inputs) {
    const auto found = inputs_.find(name);
    require(found != inputs_.end() && !weights_.count(name), "unexpected data input");
    inputHandles.insert(validateTensor(tensor, found->second));
  }
  for (const auto& [name, tensor] : outputs) {
    const auto found = outputs_.find(name);
    require(found != outputs_.end() && !flags_.count(name), "unexpected data output");
    const auto handle = validateTensor(tensor, found->second);
    require(!inputHandles.count(handle) && outputHandles.insert(handle).second,
            "output aliases an input or another output root");
  }
  // All caller-owned shape, extent, context, and alias checks happen before
  // any lazy compile can allocate a new graph workspace.
  ensureCompiled();
  // start_async/wait avoids eager readback; the selected validator below owns
  // any explicit host copy. Compute and causal roots use the admitted context.
  // Every successful or failed call is synchronous from the controller's view.
  try {
    // A flag must be freshly written by this request, never accepted from a
    // previous successful call if execution fails to publish its scalar.
    for (auto& [name, tensor] : flags_) tensor.data<ov::float16>()[0] = ov::float16(0.0f);
    for (const auto& [name, tensor] : inputs) request_.set_tensor(name, tensor);
    for (const auto& [name, tensor] : outputs) request_.set_tensor(name, tensor);
    request_.start_async();
    request_.wait();
    if (!outputChecks_.empty())
      for (const auto& [name, tensor] : outputs)
        outputChecks_.at(tensor.get_shape())->run({{"data", tensor}}, {});
    if (outputCheck_ == FlashOutputCheck::bf16Readback) {
      for (const auto& [name, tensor] : outputs) {
        const auto params = tensor.as<ov::RemoteTensor>().get_params();
        if (params.at("SHARED_MEM_TYPE").as<ov::intel_gpu::SharedMemType>() ==
            ov::intel_gpu::SharedMemType::USM_HOST_BUFFER) {
          // wait() above completed all GPU writes. Host USM is already CPU-
          // accessible; copying these large causal roots again adds no safety.
          validateFlashBf16Output(
              static_cast<const std::uint16_t*>(marker(params, "MEM_HANDLE")), tensor.get_size());
          continue;
        }
        readback_.resize(tensor.get_size());
        ov::Tensor host(ov::element::bf16, tensor.get_shape(), readback_.data());
        tensor.copy_to(host);
        validateFlashBf16Output(readback_.data(), tensor.get_size());
      }
    }
    for (const auto& [name, tensor] : flags_)
      require(static_cast<float>(tensor.data<const ov::float16>()[0]) == 1.0f,
              "non-finite output reported by " + name);
  } catch (...) {
    invalidate();
    throw;
  }
}

} // namespace otc::ai
