#include "flash_text_model.hpp"
#include "flash_model.hpp"

#include <openvino/op/gelu.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace otc::ai {
namespace {

constexpr std::size_t kTokens = 512;
constexpr std::size_t kPromptChannels = 4096;
constexpr std::size_t kContextChannels = 1536;
const ov::Shape kPrompt{1, kTokens, kPromptChannels};
const ov::Shape kContext{1, kTokens, kContextChannels};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("FlashVSR+ text: " + message);
}

void add(FlashPorts& ports, const char* name, ov::Shape shape) {
  require(ports.emplace(name, std::make_pair(ov::element::bf16, std::move(shape))).second,
          "duplicate contract port");
}

}  // namespace

void validateFlashTextModel(const ov::Model& model) {
  FlashPorts inputs, outputs;
  add(inputs, "prompt", kPrompt);
  add(inputs, "weight__text_embedding_0_weight", {kContextChannels, kPromptChannels});
  add(inputs, "weight__text_embedding_0_bias", {kContextChannels});
  add(inputs, "weight__text_embedding_2_weight", {kContextChannels, kContextChannels});
  add(inputs, "weight__text_embedding_2_bias", {kContextChannels});
  add(outputs, "context", kContext);
  add(outputs, "text_linear", kContext);
  validateFlashStructuralModel(model, inputs, outputs);
  require(model.get_friendly_name() == "flashvsr_plus_text_bf16",
          "friendly name differs");

  std::shared_ptr<ov::op::v7::Gelu> gelu;
  for (const auto& node : model.get_ordered_ops()) {
    if (auto candidate = std::dynamic_pointer_cast<ov::op::v7::Gelu>(node)) {
      require(!gelu, "text graph has multiple GELU nodes");
      gelu = std::move(candidate);
    }
  }
  require(gelu && gelu->get_approximation_mode() == ov::op::GeluApproximationMode::TANH,
          "text graph requires one tanh GELU");

  const auto stored = model.output("text_linear").get_node()->input_value(0);
  require(gelu->input_value(0) == stored,
          "text_linear must preserve the pre-GELU BF16 storage boundary");
}

}  // namespace otc::ai
