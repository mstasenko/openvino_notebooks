#include "flash_edges_session.hpp"

#include "flash-assets/flash_edges_prefix_assets.hpp"
#include "flash-assets/flash_edges_suffix_assets.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace otc::ai {
namespace {

constexpr std::size_t kWorkingHeight = 2176U;
constexpr std::size_t kWorkingWidth = 3840U;

const std::array<SeedVrAssetPin, 1>& prefixBins(FlashMode mode) {
  switch (mode) {
    case FlashMode::turbo:
      return kFlashEdgesPrefixTurboBins;
    case FlashMode::base:
      return kFlashEdgesPrefixBaseBins;
  }
  throw std::invalid_argument("FlashVSR+ prefix mode differs");
}

const std::array<SeedVrAssetPin, 1>& suffixBins(FlashMode mode) {
  switch (mode) {
    case FlashMode::turbo:
      return kFlashEdgesSuffixTurboBins;
    case FlashMode::base:
      return kFlashEdgesSuffixBaseBins;
  }
  throw std::invalid_argument("FlashVSR+ suffix mode differs");
}

SeedVrOwnedIr loadOwner(const std::filesystem::path& assets, FlashMode mode,
                        FlashEdgesRole role) {
  switch (role) {
    case FlashEdgesRole::prefix:
      return readSeedVrIr(assets, kFlashEdgesPrefixXml, prefixBins(mode).at(0),
                          flashWeightSpans(kFlashEdgesPrefixWeights));
    case FlashEdgesRole::suffix:
      return readSeedVrIr(assets, kFlashEdgesSuffixXml, suffixBins(mode).at(0),
                          flashWeightSpans(kFlashEdgesSuffixWeights));
  }
  throw std::invalid_argument("FlashVSR+ edge role differs");
}

}  // namespace

class FlashEdgesSession::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context,
       std::filesystem::path assets, FlashMode mode, FlashEdgesRole role)
      : role_(role), owner_(loadOwner(assets, mode, role)) {
    init(core, std::move(context));
  }

  FlashEdgesRole role() const noexcept { return role_; }

  void run(const FlashEdgesPrefixInputs& inputs,
           const FlashEdgesPrefixOutputs& outputs) {
    if (role_ != FlashEdgesRole::prefix)
      throw std::logic_error("prefix inputs used with a suffix edge session");
    stage_->run(
        {{"plus_f4_3840x2176_x", inputs.x}, {"t", inputs.t}, {"t_mod", inputs.tMod}},
        {{"plus_f4_3840x2176_tokens_out", outputs.tokensOut},
         {"plus_f4_3840x2176_t_out", outputs.tOut},
         {"plus_f4_3840x2176_t_mod_out", outputs.tModOut},
         {"plus_f4_3840x2176_embedding_out", outputs.embeddingOut}});
  }

  void run(const FlashEdgesSuffixInputs& inputs,
           const FlashEdgesSuffixOutputs& outputs) {
    if (role_ != FlashEdgesRole::suffix)
      throw std::logic_error("suffix inputs used with a prefix edge session");
    stage_->run(
        {{"plus_f4_3840x2176_tokens_in", inputs.tokensIn},
         {"plus_f4_3840x2176_t_in", inputs.tIn}},
        {{"plus_f4_3840x2176_out", outputs.out}});
  }

  void suspend() noexcept { stage_->suspend(); }

 private:
  void init(ov::Core& core, ov::RemoteContext context) {
    const ov::Tensor bin(ov::element::u8, {owner_.binBytes()}, owner_.binData());
    auto model = core.read_model(owner_.xml(), bin);
    validateFlashEdgesModel(*model, role_, kWorkingHeight, kWorkingWidth);
    std::vector<FlashStageWeight> weights;
    switch (role_) {
      case FlashEdgesRole::prefix:
        weights = flashStageWeights(kFlashEdgesPrefixWeights);
        break;
      case FlashEdgesRole::suffix:
        weights = flashStageWeights(kFlashEdgesSuffixWeights);
        break;
      default:
        throw std::invalid_argument("FlashVSR+ edge role differs");
    }
    stage_ = std::make_unique<FlashGpuStage>(
        core, std::move(context), model, std::move(weights), FlashPrecision::pluginDefault,
        std::string{}, std::vector<std::string>{}, FlashOutputCheck::bf16);
    stage_->bind(owner_);
  }

  FlashEdgesRole role_;
  SeedVrOwnedIr owner_;
  std::unique_ptr<FlashGpuStage> stage_;
};

FlashEdgesSession::FlashEdgesSession(ov::Core& core, ov::RemoteContext context,
                                     std::filesystem::path assets, FlashMode mode,
                                     FlashEdgesRole role)
    : impl_(std::make_unique<Impl>(core, std::move(context), std::move(assets), mode, role)) {}

FlashEdgesSession::~FlashEdgesSession() = default;
FlashEdgesSession::FlashEdgesSession(FlashEdgesSession&&) noexcept = default;
FlashEdgesSession& FlashEdgesSession::operator=(FlashEdgesSession&&) noexcept = default;

FlashEdgesRole FlashEdgesSession::role() const noexcept { return impl_->role(); }

void FlashEdgesSession::run(const FlashEdgesPrefixInputs& inputs,
                            const FlashEdgesPrefixOutputs& outputs) {
  impl_->run(inputs, outputs);
}

void FlashEdgesSession::run(const FlashEdgesSuffixInputs& inputs,
                            const FlashEdgesSuffixOutputs& outputs) {
  impl_->run(inputs, outputs);
}
void FlashEdgesSession::suspend() noexcept { impl_->suspend(); }

}  // namespace otc::ai
