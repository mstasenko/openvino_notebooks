#include "flash_pre_session.hpp"

#include "flash-assets/flash_pre_assets.hpp"
#include "flash_pre_model.hpp"

namespace otc::ai {
namespace {

const std::array<SeedVrAssetPin, 30>& layerPins(FlashMode mode) {
  switch (mode) {
    case FlashMode::turbo: return kFlashPreTurboBins;
    case FlashMode::base: return kFlashPreBaseBins;
  }
  throw std::invalid_argument("FlashVSR+ mode differs");
}

} // namespace

class FlashPreSession::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context, std::filesystem::path assets, FlashMode mode)
      : assets_(std::move(assets)), mode_(mode),
        structural_(readSeedVrIr(assets_, kFlashPreXml, layerPins(mode_)[0], flashWeightSpans(kFlashPreWeights))) {
    const ov::Tensor bin(ov::element::u8, {structural_.binBytes()}, structural_.binData());
    auto model = core.read_model(structural_.xml(), bin);
    prepareFlashPreModel(*model);
    stage_ = std::make_unique<FlashGpuStage>(core, std::move(context), model,
        flashStageWeights(kFlashPreWeights), FlashPrecision::pluginDefault, std::string{},
        std::vector<std::string>{"q_finite", "kv_finite"});
    stage_->bind(structural_);
  }

  void bindLayer(std::size_t layer) {
    stage_->invalidate();
    const auto& pins = layerPins(mode_);
    if (layer >= pins.size()) throw std::out_of_range("FlashVSR+ layer index is outside [0,30)");
    // The initial owner backs IR Constants; later owners retire after upload.
    auto owner = readSeedVrIr(assets_, kFlashPreXml, pins[layer], flashWeightSpans(kFlashPreWeights));
    stage_->bind(owner);
  }

  void run(const FlashPreInputs& in, const FlashPreOutputs& out) {
    stage_->run({{"x", in.x}, {"t_mod", in.tMod}, {"rope_cos", in.ropeCos}, {"rope_sin", in.ropeSin}},
        {{"gate_msa", out.gateMsa}, {"shift_mlp", out.shiftMlp}, {"scale_mlp", out.scaleMlp},
         {"gate_mlp", out.gateMlp}, {"q_pooled", out.qPooled}, {"k_pooled", out.kPooled},
         {"q_device_0", out.qDevice0}, {"q_device_1", out.qDevice1},
         {"k_device", out.kDevice}, {"v_device", out.vDevice}});
  }

  void suspend() noexcept { stage_->suspend(); }

 private:
  std::filesystem::path assets_;
  FlashMode mode_;
  SeedVrOwnedIr structural_;
  std::unique_ptr<FlashGpuStage> stage_;
};

FlashPreSession::FlashPreSession(ov::Core& core, ov::RemoteContext context,
                               std::filesystem::path assets, FlashMode mode)
    : impl_(std::make_unique<Impl>(core, std::move(context), std::move(assets), mode)) {}
FlashPreSession::~FlashPreSession() = default;
FlashPreSession::FlashPreSession(FlashPreSession&&) noexcept = default;
FlashPreSession& FlashPreSession::operator=(FlashPreSession&&) noexcept = default;
void FlashPreSession::bindLayer(std::size_t layer) { impl_->bindLayer(layer); }
void FlashPreSession::run(const FlashPreInputs& inputs, const FlashPreOutputs& outputs) {
  impl_->run(inputs, outputs);
}
void FlashPreSession::suspend() noexcept { impl_->suspend(); }

} // namespace otc::ai
