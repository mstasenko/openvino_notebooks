#include "flash_edges_model.hpp"
#include "flash_model.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace otc::ai {
namespace {

constexpr std::size_t kWorkingHeight = 2176;
constexpr std::size_t kWorkingWidth = 3840;
constexpr std::size_t kFrames = 4;
constexpr std::size_t kChannels = 1536;
constexpr std::size_t kEmbedding = 256;
constexpr const char* kGeometryName = "plus_f4_3840x2176";

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("FlashVSR+ edges: " + message);
}

void add(FlashPorts& ports, const char* name, ov::Shape shape) {
  require(ports.emplace(name, std::make_pair(ov::element::bf16, std::move(shape))).second,
          "duplicate contract port");
}

void contract(FlashEdgesRole role, FlashPorts& inputs, FlashPorts& outputs) {
  const ov::Shape latent{1, 16, kFrames, kWorkingHeight / 8, kWorkingWidth / 8};
  const ov::Shape tokens{1, kFrames * (kWorkingHeight / 16) * (kWorkingWidth / 16),
                         kChannels};
  const ov::Shape timestep{1, 1, kChannels};
  const ov::Shape modulation{1, 6, kChannels};
  const ov::Shape embedding{1, kEmbedding};

  switch (role) {
    case FlashEdgesRole::prefix:
      add(inputs, "plus_f4_3840x2176_x", latent);
      add(inputs, "t", timestep);
      add(inputs, "t_mod", modulation);
      add(inputs, "weight__patch_embedding_weight", {1536, 16, 1, 2, 2});
      add(inputs, "weight__patch_embedding_bias", {1536});
      add(outputs, "plus_f4_3840x2176_tokens_out", tokens);
      add(outputs, "plus_f4_3840x2176_t_out", timestep);
      add(outputs, "plus_f4_3840x2176_t_mod_out", modulation);
      add(outputs, "plus_f4_3840x2176_embedding_out", embedding);
      break;
    case FlashEdgesRole::suffix:
      add(inputs, "plus_f4_3840x2176_tokens_in", tokens);
      add(inputs, "plus_f4_3840x2176_t_in", timestep);
      add(inputs, "weight__head_modulation", {1, 2, 1536});
      add(inputs, "weight__head_proj_weight", {64, 1536});
      add(inputs, "weight__head_proj_bias", {64});
      add(outputs, "plus_f4_3840x2176_out", latent);
      break;
    default:
      throw std::invalid_argument("FlashVSR+ edge role differs");
  }
}

void validateGeometry(std::size_t height, std::size_t width) {
  // This native seam intentionally authenticates the one exported UHD edge
  // shape.  It must not silently accept a different graph geometry.
  require(height == kWorkingHeight && width == kWorkingWidth,
          "only visible 2160x3840 / working 2176x3840 geometry is admitted");
}

}  // namespace

void validateFlashEdgesModel(const ov::Model& model, FlashEdgesRole role,
                             std::size_t height, std::size_t width) {
  validateGeometry(height, width);

  FlashPorts inputs, outputs;
  contract(role, inputs, outputs);
  validateFlashStructuralModel(model, inputs, outputs);

  const auto expectedName = std::string("flashvsr_plus_") + kGeometryName + "_" +
                            (role == FlashEdgesRole::prefix ? "prefix" : "suffix") + "_bf16";
  require(model.get_friendly_name() == expectedName, "edge model friendly name differs");
}

}  // namespace otc::ai
