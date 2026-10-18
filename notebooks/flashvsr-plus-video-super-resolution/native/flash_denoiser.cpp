#include "flash_denoiser.hpp"

#include "flash-assets/flash_fixed_assets.hpp"
#include "flash_attention_session.hpp"
#include "flash_cross_session.hpp"
#include "flash_edges_session.hpp"
#include "flash_geometry.hpp"
#include "flash_history.hpp"
#include "flash_mask.hpp"
#include "flash_post_session.hpp"
#include "flash_pre_session.hpp"
#include "flash_rope.hpp"
#include "flash_stream.hpp"
#include "flash_text_session.hpp"
#include "flash_token_session.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace otc::ai {
namespace {

constexpr std::size_t kLayers = 30U, kHeads = 12U, kHeadDim = 128U;
const ov::Shape kLatent{1U, 16U, 4U, 272U, 480U};
const ov::Shape kTokens{1U, 130560U, 1536U};
const ov::Shape kRows{1U, 4096U, 1536U};
const ov::Shape kPooled{12U, 32U, 128U};
const ov::Shape kRopeSlab{1U, 4096U, 1U, 64U};

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(std::string("FlashVSR+ denoiser: ") + message);
}

void upload(ov::Tensor& remote, const void* data) {
  auto destination = remote.as<ov::RemoteTensor>();
  destination.copy_from(ov::Tensor(remote.get_element_type(), remote.get_shape(), data));
}

void read(const ov::Tensor& remote, void* data) {
  ov::Tensor destination(remote.get_element_type(), remote.get_shape(), data);
  remote.copy_to(destination);
}

float decode(std::uint16_t value) noexcept {
  const auto bits = static_cast<std::uint32_t>(value) << 16U;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint16_t rounded(float value) {
  require(std::isfinite(value), "non-finite predicted latent");
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const auto result = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
  require((result & 0x7f80U) != 0x7f80U, "predicted latent overflows BF16");
  return result;
}

// Early admission only: the native worker still verifies observed phase
// retirement and measured peaks. Do not mistake named roots for all plugin
// workspaces; the LQ/decoder owners must suspend their compiled workspaces.
void checkRoom(ov::Core& core, const ov::RemoteContext& context) {
  // The qualified UHD call added about 6.84 GB above suspended ownership.
  // Admit 7 GiB free rather than a smaller named-root-only estimate.
  constexpr std::uint64_t required = 7ULL << 30U;
  const auto device = context.get_device_name();
  const auto total = core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE").as<std::uint64_t>();
  const auto stats = core.get_property(device, "GPU_MEMORY_STATISTICS").as<std::map<std::string, std::uint64_t>>();
  std::uint64_t used = 0U;
  for (const auto* kind : {"usm_device", "usm_shared", "cl_mem", "unknown"}) {
    const auto found = stats.find(kind);
    require(found != stats.end() && found->second <= std::numeric_limits<std::uint64_t>::max() - used,
            "GPU memory statistics are unavailable");
    used += found->second;
  }
  require(total >= required && used <= total - required, "not enough GPU memory for the denoiser phase");
}

} // namespace

class FlashDenoiserSession::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context, const std::filesystem::path& assets,
       const std::filesystem::path& scratch, FlashMode mode, std::size_t sourceFrames)
      : plan_(sourceFrames), mode_(mode), scratch_(scratch), assets_(assets), core_(core),
        geometry_(makeFlashGeometry(2160U, 3840U)),
        tokenGeometry_(makeFlashTokenGeometry(4U, geometry_.tokenHeight, geometry_.tokenWidth)),
        context_(std::move(context)) {
    // Validate the policy before any model or device work.
    (void)flashHistoryPolicy(mode_, 0U);
    // Reserve the complete bounded history before expensive GPU work. These
    // mappings have no resident K/V pages until a layer writes its valid
    // prefix; hard allocation prevents a later disk-full mapping fault.
    if (plan_.callCount() > 1U)
      for (std::size_t layer = 0U; layer < kLayers; ++layer)
        histories_[layer] = std::make_unique<FlashHistoryOwner>(
            scratch_, flashHistoryPolicy(mode_, layer), geometry_.spatialWindows);
    checkRoom(core, context_);
    prepareCross(core, assets);
    // POST registers the shared custom-operation ABI before attention compiles.
    post_ = std::make_unique<FlashPostSession>(core, context_, assets, mode_);
    pre_ = std::make_unique<FlashPreSession>(core, context_, assets, mode_);
    prefix_ = std::make_unique<FlashEdgesSession>(core, context_, assets, mode_, FlashEdgesRole::prefix);
    suffix_ = std::make_unique<FlashEdgesSession>(core, context_, assets, mode_, FlashEdgesRole::suffix);
    t_ = allocate({1U, 1U, 1536U});
    tMod_ = allocate({1U, 6U, 1536U});
    const auto fixed = readBf16Asset(assets, kFlashFixedAssets.at(mode_ == FlashMode::turbo ? 0U : 1U).bin);
    upload(t_, fixed.data() + kFlashFixedTOffset / sizeof(std::uint16_t));
    upload(tMod_, fixed.data() + kFlashFixedTModOffset / sizeof(std::uint16_t));
    suspend();
  }

  std::vector<std::uint16_t> run(FlashBf16ConstBuffer noise, FlashBf16ConstBuffer condition,
                                const Progress& progress) {
    require(!failed_ && call_ < plan_.callCount(), "stream is failed or complete");
    require(noise.data && noise.words == ov::shape_size(kLatent) &&
                condition.data && condition.words == ov::shape_size(kTokens), "call input extent differs");
    validateFlashBf16Output(noise.data, noise.words);
    validateFlashBf16Output(condition.data, condition.words);
    try {
      prepareCallScratch();
      const auto call = plan_.call(call_);
      const auto tick = [&](std::size_t layer, double fraction) {
        if (progress)
          progress((static_cast<double>(call_) +
                    (static_cast<double>(layer) + fraction) / static_cast<double>(kLayers + 1U)) /
                   static_cast<double>(plan_.callCount()));
      };
      tick(0U, 0.0);
      prepareRope();
      upload(noise_, noise.data);
      prefix_->run({noise_, t_, tMod_}, {canonical_, tOut_, tModOut_, embedding_});
      read(canonical_, canonicalHost_.data());
      tokens_->load({canonicalHost_.data(), canonicalHost_.size()}, condition);
      for (std::size_t layer = 0U; layer < kLayers; ++layer) {
        tick(layer, 0.0);
        pre_->bindLayer(layer);
        post_->bindLayer(layer);
        const auto policy = flashHistoryPolicy(mode_, layer);
        auto& history = histories_.at(layer);
        require(history || call.last, "missing active layer history");
        if (history) {
          const auto old = history->view();
          attention_->beginLayer(old.k(), old.v(), old.windows(), old.windows() * kHeadDim * kFlashTokenChannels);
        } else {
          // A one-call clip has no future consumer: do not allocate disk K/V
          // capacity merely to retire it at the end of this same layer.
          attention_->beginLayer(nullptr, nullptr, 0U, 0U);
        } // The synchronous upload has consumed this mapping before update.
        const auto windows = tokenGeometry_.windowCount();
        for (std::size_t start = 0U; start < windows; start += kFlashTokenSlabWindows) {
          const auto count = std::min(kFlashTokenSlabWindows, windows - start);
          tick(layer, 0.25 * static_cast<double>(start) / static_cast<double>(windows));
          const auto targets = attention_->preTargets(start, count);
          const auto& rope = rope_.at(start / kFlashTokenSlabWindows);
          pre_->run({tokens_->inputSlab(start, count), tModOut_, rope.first, rope.second},
              {gateMsa_, shiftMlp_, scaleMlp_, gateMlp_, qPooled_, kPooled_,
               targets.q0, targets.q1, targets.k, targets.v});
          attention_->completePre(start, count); // Only after PRE's finite flags passed.
          gatherPooled(qPooled_, qHost_, start, count);
          gatherPooled(kPooled_, kHost_, start, count);
        }
        FlashHistoryUpdate update;
        update.current_windows = windows;
        update.current_pooled = kHost_.data();
        update.terminal = call.last;
        if (!call.last) update.read_retained = [&](auto* k, auto* v, auto start, auto count) {
          attention_->readRetained(k, v, start, count, count * kHeadDim * kFlashTokenChannels);
        };
        auto merged = history ? history->update(update) : kHost_;
        if (call.last) history.reset();
        const auto keys = merged.size() / (kHeads * kHeadDim);
        FlashMaskConfig config;
        config.temporalWindows = 2U;
        config.spatialRows = geometry_.tokenHeight / 8U;
        config.spatialCols = geometry_.tokenWidth / 8U;
        config.cacheTemporalGroups = policy.retainedGroups;
        config.localRange = policy.localRange;
        auto mask = buildFlashMaskLayout(qHost_.data(), kHeads, windows,
                                         merged.data(), kHeads, keys, config);
        attention_->infer(mask.blockMask, [&](auto done, auto total) {
          tick(layer, 0.25 + 0.5 * static_cast<double>(done) / static_cast<double>(total));
        });
        for (std::size_t start = 0U; start < windows; start += kFlashTokenSlabWindows) {
          const auto count = std::min(kFlashTokenSlabWindows, windows - start);
          tick(layer, 0.75 + 0.25 * static_cast<double>(start) / static_cast<double>(windows));
          const auto contexts = attention_->postTargets(start, count);
          post_->run({tokens_->inputSlab(start, count), contexts.first, contexts.second,
              cross_.at(layer).first, cross_.at(layer).second,
              gateMsa_, shiftMlp_, scaleMlp_, gateMlp_}, postOutput_);
          tokens_->commit(start, count, postOutput_);
        }
        tick(layer, 1.0);
      }
      tokens_->read({canonicalHost_.data(), canonicalHost_.size()});
      upload(canonical_, canonicalHost_.data());
      suffix_->run(FlashEdgesSuffixInputs{canonical_, tOut_}, FlashEdgesSuffixOutputs{predicted_});
      std::vector<std::uint16_t> result(noise.words);
      read(predicted_, result.data());
      validateFlashBf16Output(result.data(), result.size());
      for (std::size_t i = 0U; i < result.size(); ++i)
        result[i] = rounded(decode(noise.data[i]) - decode(result[i]));
      ++call_;
      if (progress) progress(static_cast<double>(call_) / static_cast<double>(plan_.callCount()));
      return result;
    } catch (...) { failed_ = true; throw; }
  }

  void suspend() noexcept {
    pre_->suspend();
    post_->suspend();
    prefix_->suspend();
    suffix_->suspend();
    attention_.reset();
    tokens_.reset();
    for (auto* root : {&noise_, &predicted_, &canonical_, &tOut_, &tModOut_, &embedding_,
                       &gateMsa_, &shiftMlp_, &scaleMlp_, &gateMlp_, &qPooled_, &kPooled_, &postOutput_})
      *root = ov::Tensor{};
    decltype(rope_)().swap(rope_);
    decltype(indices_)().swap(indices_);
    decltype(canonicalHost_)().swap(canonicalHost_);
    decltype(qHost_)().swap(qHost_);
    decltype(kHost_)().swap(kHost_);
  }

 private:
  ov::Tensor allocate(const ov::Shape& shape) { return flashDeviceTensor(context_, ov::element::bf16, shape); }

  void prepareCallScratch() {
    if (attention_) return;
    checkRoom(core_, context_);
    attention_ = std::make_unique<FlashAttentionSession>(core_, context_, assets_, geometry_.spatialWindows);
    tokens_ = std::make_unique<FlashTokenSession>(context_, tokenGeometry_);
    noise_ = allocate(kLatent);
    predicted_ = allocate(kLatent);
    canonical_ = allocate(kTokens); // Prefix output / final suffix input.
    tOut_ = allocate({1U, 1U, 1536U});
    tModOut_ = allocate({1U, 6U, 1536U});
    embedding_ = allocate({1U, 256U});
    gateMsa_ = allocate({1U, 1U, 1536U});
    shiftMlp_ = allocate({1U, 1U, 1536U});
    scaleMlp_ = allocate({1U, 1U, 1536U});
    gateMlp_ = allocate({1U, 1U, 1536U});
    qPooled_ = flashDeviceTensor(context_, ov::element::f32, kPooled);
    kPooled_ = flashDeviceTensor(context_, ov::element::f32, kPooled);
    postOutput_ = allocate(kRows);
    canonicalHost_.resize(ov::shape_size(kTokens));
    const auto windows = tokenGeometry_.windowCount();
    qHost_.resize(kHeads * windows * kHeadDim);
    kHost_.resize(qHost_.size());
    for (std::size_t start = 0U; start < windows; start += kFlashTokenSlabWindows) {
      const auto count = std::min(kFlashTokenSlabWindows, windows - start);
      indices_.push_back(flashTokenSlabIndices(tokenGeometry_, start, count));
      rope_.emplace_back(flashDeviceTensor(context_, ov::element::f32, kRopeSlab),
                         flashDeviceTensor(context_, ov::element::f32, kRopeSlab));
    }
  }

  void prepareCross(ov::Core& core, const std::filesystem::path& assets) {
    auto prompt = readBf16Asset(assets, kFlashFixedPrompt.raw);
    auto promptRoot = allocate({1U, 512U, 4096U});
    auto context = allocate({1U, 512U, 1536U});
    auto textLinear = allocate({1U, 512U, 1536U});
    upload(promptRoot, prompt.data());
    {
      FlashTextSession text(core, context_, assets, mode_);
      text.run({promptRoot}, {context, textLinear});
    }
    FlashCrossSession cross(core, context_, assets, mode_);
    for (std::size_t layer = 0U; layer < kLayers; ++layer) {
      cross.bindLayer(layer);
      cross_.at(layer) = {allocate({1U, 512U, 1536U}), allocate({1U, 512U, 1536U})};
      cross.run(context, cross_.at(layer).first, cross_.at(layer).second);
    }
  }

  void prepareRope() {
    const auto elements = flashRopeElements(geometry_);
    std::vector<float> cos(elements), sin(elements);
    fillFlashRopeInputs(geometry_, call_, cos.data(), cos.size(), sin.data(), sin.size());
    std::vector<float> slab(ov::shape_size(kRopeSlab));
    for (std::size_t i = 0U; i < rope_.size(); ++i) {
      const auto fill = [&](const std::vector<float>& source, ov::Tensor& target) {
        std::fill(slab.begin(), slab.end(), 0.0f);
        const auto& indices = indices_.at(i);
        for (std::size_t row = 0U; row < indices.size(); ++row)
          std::copy_n(source.data() + indices[row] * kFlashRopePairs, kFlashRopePairs,
                      slab.data() + row * kFlashRopePairs);
        upload(target, slab.data());
      };
      fill(cos, rope_[i].first);
      fill(sin, rope_[i].second);
    }
  }

  void gatherPooled(const ov::Tensor& source, std::vector<float>& destination,
                     std::size_t start, std::size_t count) {
    std::array<float, kHeads * kFlashTokenSlabWindows * kHeadDim> slab{};
    read(source, slab.data());
    for (std::size_t h = 0U; h < kHeads; ++h)
      std::copy_n(slab.data() + h * kFlashTokenSlabWindows * kHeadDim, count * kHeadDim,
                  destination.data() + (h * tokenGeometry_.windowCount() + start) * kHeadDim);
  }

  flash::StreamPlan plan_;
  FlashMode mode_;
  std::filesystem::path scratch_, assets_;
  ov::Core core_;
  FlashGeometry geometry_;
  FlashTokenGeometry tokenGeometry_;
  ov::RemoteContext context_;
  std::array<std::pair<ov::Tensor, ov::Tensor>, kLayers> cross_;
  std::unique_ptr<FlashPostSession> post_;
  std::unique_ptr<FlashPreSession> pre_;
  std::unique_ptr<FlashEdgesSession> prefix_, suffix_;
  std::unique_ptr<FlashAttentionSession> attention_;
  std::unique_ptr<FlashTokenSession> tokens_;
  std::array<std::unique_ptr<FlashHistoryOwner>, kLayers> histories_;
  ov::Tensor noise_, predicted_, canonical_, t_, tMod_, tOut_, tModOut_, embedding_;
  ov::Tensor gateMsa_, shiftMlp_, scaleMlp_, gateMlp_, qPooled_, kPooled_, postOutput_;
  std::vector<std::uint16_t> canonicalHost_;
  std::vector<float> qHost_, kHost_;
  std::vector<std::vector<std::size_t>> indices_;
  std::vector<std::pair<ov::Tensor, ov::Tensor>> rope_;
  std::size_t call_ = 0U;
  bool failed_ = false;
};

FlashDenoiserSession::FlashDenoiserSession(ov::Core& core, ov::RemoteContext context,
    const std::filesystem::path& assets, const std::filesystem::path& scratch,
    FlashMode mode, std::size_t sourceFrames)
    : impl_(std::make_unique<Impl>(core, std::move(context), assets, scratch, mode, sourceFrames)) {}
FlashDenoiserSession::~FlashDenoiserSession() = default;
FlashDenoiserSession::FlashDenoiserSession(FlashDenoiserSession&&) noexcept = default;
FlashDenoiserSession& FlashDenoiserSession::operator=(FlashDenoiserSession&&) noexcept = default;
std::vector<std::uint16_t> FlashDenoiserSession::run(FlashBf16ConstBuffer noise,
    FlashBf16ConstBuffer condition, const Progress& progress) { return impl_->run(noise, condition, progress); }
void FlashDenoiserSession::suspend() noexcept { impl_->suspend(); }

} // namespace otc::ai
