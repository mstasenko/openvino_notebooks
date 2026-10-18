#include "flash_text_session.hpp"

#include "flash-assets/flash_text_assets.hpp"
#include "flash_text_model.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace otc::ai {
namespace {

const SeedVrAssetPin& modeBin(FlashMode mode) {
  switch (mode) {
    case FlashMode::turbo:
      return kFlashTextTurboBins.at(0);
    case FlashMode::base:
      return kFlashTextBaseBins.at(0);
  }
  throw std::invalid_argument("FlashVSR+ text mode differs");
}

}  // namespace

class FlashTextSession::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context,
       std::filesystem::path assets, FlashMode mode)
      : owner_(readSeedVrIr(assets, kFlashTextXml, modeBin(mode),
                            flashWeightSpans(kFlashTextWeights))) {
    const ov::Tensor bin(ov::element::u8, {owner_.binBytes()}, owner_.binData());
    auto model = core.read_model(owner_.xml(), bin);
    validateFlashTextModel(*model);
    stage_ = std::make_unique<FlashGpuStage>(
        core, std::move(context), model, flashStageWeights(kFlashTextWeights),
        FlashPrecision::pluginDefault, std::string{}, std::vector<std::string>{},
        FlashOutputCheck::bf16);
    stage_->bind(owner_);
  }

  void run(const FlashTextInputs& inputs, const FlashTextOutputs& outputs) {
    stage_->run({{"prompt", inputs.prompt}},
                {{"context", outputs.context}, {"text_linear", outputs.textLinear}});
  }

 private:
  SeedVrOwnedIr owner_;
  std::unique_ptr<FlashGpuStage> stage_;
};

FlashTextSession::FlashTextSession(ov::Core& core, ov::RemoteContext context,
                                   std::filesystem::path assets, FlashMode mode)
    : impl_(std::make_unique<Impl>(core, std::move(context), std::move(assets), mode)) {}

FlashTextSession::~FlashTextSession() = default;
FlashTextSession::FlashTextSession(FlashTextSession&&) noexcept = default;
FlashTextSession& FlashTextSession::operator=(FlashTextSession&&) noexcept = default;

void FlashTextSession::run(const FlashTextInputs& inputs,
                           const FlashTextOutputs& outputs) {
  impl_->run(inputs, outputs);
}

}  // namespace otc::ai
