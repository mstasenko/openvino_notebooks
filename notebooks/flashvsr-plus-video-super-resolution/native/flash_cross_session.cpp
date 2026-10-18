#include "flash_cross_session.hpp"

#include "flash-assets/flash_cross_assets.hpp"
#include "flash_cross_model.hpp"

namespace otc::ai {
namespace {

const std::array<SeedVrAssetPin, 30>& layerPins(FlashMode mode) {
  switch (mode) {
    case FlashMode::turbo: return kFlashCrossTurboBins;
    case FlashMode::base: return kFlashCrossBaseBins;
  }
  throw std::invalid_argument("FlashVSR+ cross mode differs");
}

} // namespace

class FlashCrossSession::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context,
       std::filesystem::path assets, FlashMode mode)
      : assets_(std::move(assets)), mode_(mode),
        structural_(readSeedVrIr(assets_, kFlashCrossXml, layerPins(mode_)[0],
                                 flashWeightSpans(kFlashCrossWeights))) {
    const ov::Tensor bin(ov::element::u8, {structural_.binBytes()}, structural_.binData());
    auto model = core.read_model(structural_.xml(), bin);
    validateFlashCrossModel(*model);
    stage_ = std::make_unique<FlashGpuStage>(core, std::move(context), model,
        flashStageWeights(kFlashCrossWeights), FlashPrecision::pluginDefault, std::string{},
        std::vector<std::string>{}, FlashOutputCheck::bf16);
    stage_->bind(structural_);
  }

  void bindLayer(std::size_t layer) {
    stage_->invalidate();
    const auto& pins = layerPins(mode_);
    if (layer >= pins.size()) throw std::out_of_range("FlashVSR+ cross layer is outside [0,30)");
    // The first owner backs the structural Constants for the model lifetime;
    // later owners can retire as soon as their synchronous upload completes.
    const auto owner = readSeedVrIr(assets_, kFlashCrossXml, pins[layer],
                                    flashWeightSpans(kFlashCrossWeights));
    stage_->bind(owner);
  }

  void run(const ov::Tensor& context, const ov::Tensor& crossK,
           const ov::Tensor& crossV) {
    stage_->run({{"context", context}}, {{"cross_k", crossK}, {"cross_v", crossV}});
  }

 private:
  std::filesystem::path assets_;
  FlashMode mode_;
  SeedVrOwnedIr structural_;
  std::unique_ptr<FlashGpuStage> stage_;
};

FlashCrossSession::FlashCrossSession(ov::Core& core, ov::RemoteContext context,
                                   std::filesystem::path assets, FlashMode mode)
    : impl_(std::make_unique<Impl>(core, std::move(context), std::move(assets), mode)) {}
FlashCrossSession::~FlashCrossSession() = default;
FlashCrossSession::FlashCrossSession(FlashCrossSession&&) noexcept = default;
FlashCrossSession& FlashCrossSession::operator=(FlashCrossSession&&) noexcept = default;
void FlashCrossSession::bindLayer(std::size_t layer) { impl_->bindLayer(layer); }
void FlashCrossSession::run(const ov::Tensor& context, const ov::Tensor& crossK,
                           const ov::Tensor& crossV) {
  impl_->run(context, crossK, crossV);
}

} // namespace otc::ai
