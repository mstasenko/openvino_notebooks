#include "flash_cross_model.hpp"
#include "flash_model.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace otc::ai {

namespace {

constexpr std::size_t kRows = 512;
constexpr std::size_t kChannels = 1536;

void add(FlashPorts& ports, const char* name, ov::Shape shape) {
  if (!ports.emplace(name, std::make_pair(ov::element::bf16, std::move(shape))).second)
    throw std::runtime_error("FlashVSR+ Cross: duplicate contract port");
}

}  // namespace

void validateFlashCrossModel(const ov::Model& model) {
  FlashPorts inputs, outputs;
  add(inputs, "context", {1, kRows, kChannels});
  add(inputs, "weight__blocks_0_cross_attn_k_weight", {kChannels, kChannels});
  add(inputs, "weight__blocks_0_cross_attn_k_bias", {kChannels});
  add(inputs, "weight__blocks_0_cross_attn_norm_k_weight", {kChannels});
  add(inputs, "weight__blocks_0_cross_attn_v_weight", {kChannels, kChannels});
  add(inputs, "weight__blocks_0_cross_attn_v_bias", {kChannels});
  add(outputs, "cross_k", {1, kRows, kChannels});
  add(outputs, "cross_v", {1, kRows, kChannels});

  validateFlashStructuralModel(model, inputs, outputs);
}

}  // namespace otc::ai
