#include "flash_decoder_model.hpp"

#include "flash-assets/flash_decoder_assets.hpp"
#include "flash_model.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace otc::ai {
namespace {

void add(FlashPorts& ports, const std::string& name, ov::Shape shape) {
  if (!ports.emplace(name, std::make_pair(ov::element::bf16, std::move(shape))).second)
    throw std::runtime_error("FlashVSR+ decoder: duplicate contract port");
}

template<class Bindings>
void addWeights(FlashPorts& inputs, const Bindings& bindings) {
  for (const auto& weight : bindings)
    add(inputs, weight.port_name, flashWeightShape(weight));
}

// Histories are explicit Parameters/Results, not hidden request state. Front
// state names differ at the two boundaries so IR round-tripping cannot alias
// the old and new roots. Growth uses its measured S0/S1/S2 output names.
void addHistories(FlashPorts& inputs, FlashPorts& outputs,
                  std::size_t channels, std::size_t height, std::size_t width,
                  std::size_t firstResidual, bool front) {
  const ov::Shape shape{1, channels, height, width};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto name = "mem_" + std::to_string(channels) + "_" + std::to_string(i);
    add(inputs, name, shape);
    add(outputs, front ? name + "_out" : "S" + std::to_string(i), shape);
    add(outputs, "residual_linear_" + std::to_string(firstResidual + i), shape);
  }
}

}  // namespace

void validateFlashDecoderModel(const ov::Model& model, FlashDecoderRole role,
                               std::size_t height, std::size_t width) {
  if (height != 2176 || width != 3840)
    throw std::invalid_argument("FlashVSR+ decoder: only working 2176x3840 geometry is admitted");

  FlashPorts inputs, outputs;
  const char* friendlyName = nullptr;
  switch (role) {
    case FlashDecoderRole::front:
      friendlyName = "tcdecoder_phase_front";
      add(inputs, "packed", {1, 784, 272, 480});
      addHistories(inputs, outputs, 512, 272, 480, 0, true);
      addHistories(inputs, outputs, 256, 544, 960, 3, true);
      addWeights(inputs, kFlashDecoderFrontWeights);
      add(outputs, "middle", {1, 256, 544, 960});
      break;
    case FlashDecoderRole::grow:
      friendlyName = "tcdecoder_growth_history_tile";
      add(inputs, "middle_in", {1, 256, 46, 960});
      // Child banks are caller inputs, not persistent weight__ bindings.
      add(inputs, "grow_weight", {256, 256, 1, 1});
      addHistories(inputs, outputs, 128, 92, 1920, 6, false);
      addWeights(inputs, kFlashDecoderGrowWeights);
      add(outputs, "middle", {1, 128, 92, 1920});
      break;
    case FlashDecoderRole::frame:
      friendlyName = "tcdecoder_remote_frame_child";
      add(inputs, "middle", {1, 128, 92, 1920});
      add(inputs, "frame_grow", {128, 128, 1, 1});
      addWeights(inputs, kFlashDecoderFrameWeights);
      add(outputs, "frame", {1, 3, 184, 3840});
      break;
    default:
      throw std::invalid_argument("FlashVSR+ decoder role differs");
  }
  validateFlashStructuralModel(model, inputs, outputs);
  if (model.get_friendly_name() != friendlyName)
    throw std::runtime_error("FlashVSR+ decoder friendly name differs");
}

}  // namespace otc::ai
