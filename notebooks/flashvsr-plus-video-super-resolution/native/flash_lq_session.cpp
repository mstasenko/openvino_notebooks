#include "flash_lq_session.hpp"

#include "flash-assets/flash_lq_assets.hpp"
#include "flash_lq_model.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace otc::ai {
namespace {

constexpr std::size_t kHeight = 2176;
constexpr std::size_t kWidth = 3840;

const std::array<SeedVrAssetPin, 3>& modeBins(FlashMode mode) {
  switch (mode) {
    case FlashMode::turbo: return kFlashLqTurboBins;
    case FlashMode::base: return kFlashLqBaseBins;
  }
  throw std::invalid_argument("FlashVSR+ LQ mode differs");
}

// The structural constants borrow this owner. Destroy the request/model before
// its backing bytes, just as PRE/POST do; learned uploads are synchronous.
struct ProjectionStage final {
  SeedVrOwnedIr owner;
  std::unique_ptr<FlashGpuStage> gpu;

  template<class Bindings>
  ProjectionStage(ov::Core& core, const ov::RemoteContext& context,
                  const std::filesystem::path& assets,
                  const SeedVrAssetPin& xmlPin, const SeedVrAssetPin& binPin,
                  const Bindings& bindings, FlashLqRole role)
      : owner(readSeedVrIr(assets, xmlPin, binPin, flashWeightSpans(bindings))) {
    const ov::Tensor bin(ov::element::u8, {owner.binBytes()}, owner.binData());
    auto model = core.read_model(owner.xml(), bin);
    validateFlashLqModel(*model, role, kHeight, kWidth);
    gpu = std::make_unique<FlashGpuStage>(core, context, model,
        flashStageWeights(bindings), FlashPrecision::pluginDefault, std::string{},
        std::vector<std::string>{}, FlashOutputCheck::bf16);
    gpu->bind(owner);
  }
};

} // namespace

class FlashLqSession::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context,
       const std::filesystem::path& assets, FlashMode mode)
      : core_(core), context_(std::move(context)), assets_(assets), mode_(mode) {
    const auto& bins = modeBins(mode);
    startup_ = std::make_unique<ProjectionStage>(core, context_, assets,
        kFlashLqStartupXml, bins[0], kFlashLqStartupWeights, FlashLqRole::startup);
    for (std::size_t i = 0; i < cache1_.size(); ++i) {
      cache1_[i] = flashDeviceTensor(context_, ov::element::bf16, {1, 768, 2, kHeight / 16, kWidth / 16});
      cache2_[i] = flashDeviceTensor(context_, ov::element::bf16, {1, 2048, 2, kHeight / 16, kWidth / 16});
    }
  }

  void start(const ov::Tensor& frame) {
    if (failed_ || started_) throw std::logic_error("FlashVSR+ LQ startup is not available");
    try {
      startup_->gpu->run({{"video", frame}}, {{"cache1", cache1_[0]}, {"cache2", cache2_[0]}});
      // Startup never runs again. Retire its bank/workspace before compiling
      // steady stages, so admission need not cover all three stages at once.
      startup_.reset();
      const auto& bins = modeBins(mode_);
      prefix_ = std::make_unique<ProjectionStage>(core_, context_, assets_,
          kFlashLqPrefixXml, bins[1], kFlashLqPrefixWeights, FlashLqRole::prefix);
      suffix_ = std::make_unique<ProjectionStage>(core_, context_, assets_,
          kFlashLqSuffixXml, bins[2], kFlashLqSuffixWeights, FlashLqRole::suffix);
      started_ = true;
    } catch (...) {
      invalidate();
      throw;
    }
  }

  void project(const ov::Tensor& frames, const ov::Tensor& output) {
    if (failed_ || !started_) throw std::logic_error("FlashVSR+ LQ stream is not ready");
    const auto next = 1U - current_;
    try {
      prefix_->gpu->run({{"video", frames}, {"old_cache1", cache1_[current_]}},
          {{"cache1", cache1_[next]}, {"cache2", cache2_[next]}});
      suffix_->gpu->run({{"old_cache2", cache2_[current_]}, {"current_cache2", cache2_[next]}},
          {{"projected", output}});
      // Do not publish a partially advanced cache pair if either stage fails.
      current_ = next;
    } catch (...) {
      invalidate();
      throw;
    }
  }

  void suspend() noexcept {
    if (startup_) startup_->gpu->suspend();
    if (prefix_) prefix_->gpu->suspend();
    if (suffix_) suffix_->gpu->suspend();
  }

 private:
  void invalidate() noexcept {
    failed_ = true;
    if (startup_) startup_->gpu->invalidate();
    if (prefix_) prefix_->gpu->invalidate();
    if (suffix_) suffix_->gpu->invalidate();
  }

  ov::Core core_; // Share the original Core implementation for deferred compilation.
  ov::RemoteContext context_;
  std::filesystem::path assets_;
  FlashMode mode_;
  std::array<ov::Tensor, 2> cache1_, cache2_;
  std::unique_ptr<ProjectionStage> startup_, prefix_, suffix_;
  std::size_t current_ = 0;
  bool started_ = false;
  bool failed_ = false;
};

FlashLqSession::FlashLqSession(ov::Core& core, ov::RemoteContext context,
                             std::filesystem::path assets, FlashMode mode)
    : impl_(std::make_unique<Impl>(core, std::move(context), assets, mode)) {}
FlashLqSession::~FlashLqSession() = default;
FlashLqSession::FlashLqSession(FlashLqSession&&) noexcept = default;
FlashLqSession& FlashLqSession::operator=(FlashLqSession&&) noexcept = default;
void FlashLqSession::start(const ov::Tensor& frame) { impl_->start(frame); }
void FlashLqSession::project(const ov::Tensor& frames, const ov::Tensor& output) {
  impl_->project(frames, output);
}
void FlashLqSession::suspend() noexcept { impl_->suspend(); }

} // namespace otc::ai
