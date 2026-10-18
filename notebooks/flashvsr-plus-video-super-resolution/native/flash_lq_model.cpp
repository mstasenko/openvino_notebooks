#include "flash_lq_model.hpp"
#include "flash_model.hpp"

#include <stdexcept>
#include <utility>

namespace otc::ai {
namespace {

void add(FlashPorts& ports, const char* name, ov::Shape shape) {
  if (!ports.emplace(name, std::make_pair(ov::element::bf16, std::move(shape))).second)
    throw std::runtime_error("FlashVSR+ LQ: duplicate contract port");
}

}  // namespace

void validateFlashLqModel(const ov::Model& model, FlashLqRole role,
                         std::size_t height, std::size_t width) {
  // The complete Flash path partitions the padded frame into 128-pixel
  // windows. Bound dimensions before deriving tensor extents or allocating.
  if (!(height && width && height <= 4096 && width <= 4096 &&
        height % 128 == 0 && width % 128 == 0 &&
        height * width <= 2176U * 3840U))
    throw std::invalid_argument("FlashVSR+ LQ: working geometry is not admitted");
  const auto h = height / 16U;
  const auto w = width / 16U;
  const ov::Shape cache1{1, 768, 2, h, w};
  const ov::Shape cache2{1, 2048, 2, h, w};
  FlashPorts inputs, outputs;
  switch (role) {
    case FlashLqRole::startup:
    case FlashLqRole::prefix:
      add(inputs, "video", {1, 3, role == FlashLqRole::startup ? 1U : 4U, height, width});
      // Portable IR needs distinct names for old and next cache tensors;
      // otherwise the IR reader silently uniquifies the input name.
      if (role == FlashLqRole::prefix) add(inputs, "old_cache1", cache1);
      add(inputs, "weight__conv1_bias", {2048});
      add(inputs, "weight__conv1_weight", {2048, 768, 4, 3, 3});
      add(inputs, "weight__norm1_weight", {2048, 1, 1, 1});
      add(outputs, "cache1", cache1);
      add(outputs, "cache2", cache2);
      break;
    case FlashLqRole::suffix:
      add(inputs, "old_cache2", cache2);
      add(inputs, "current_cache2", cache2);
      add(inputs, "weight__conv2_bias", {3072});
      add(inputs, "weight__conv2_weight", {3072, 2048, 4, 3, 3});
      add(inputs, "weight__linear_layers_0_bias", {1536});
      add(inputs, "weight__linear_layers_0_weight", {1536, 3072});
      add(inputs, "weight__norm2_weight", {3072, 1, 1, 1});
      add(outputs, "projected", {1, h * w, 1536});
      break;
    default:
      throw std::invalid_argument("FlashVSR+ LQ role differs");
  }
  validateFlashStructuralModel(model, inputs, outputs);
}

}  // namespace otc::ai
