#include "flash_video.hpp"

#include "flash_decoder_frame.hpp"
#include "flash_decoder_session.hpp"
#include "flash_denoiser.hpp"
#include "flash_geometry.hpp"
#include "flash_lq_session.hpp"
#include "flash_stage.hpp"
#include "flash_stream.hpp"

#include <openvino/runtime/properties.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace otc::ai {
namespace {

constexpr std::size_t kVisibleHeight = 2160U;
constexpr std::size_t kVisibleWidth = 3840U;
constexpr std::size_t kWorkingHeight = kFlashDecoderFrameWorkingHeight;
constexpr std::size_t kWorkingWidth = kFlashDecoderFrameWorkingWidth;
constexpr std::size_t kRgbChannels = 3U;
constexpr std::size_t kRgbFrameBytes = kVisibleHeight * kVisibleWidth * kRgbChannels;
constexpr std::size_t kLqPlaneWords = kWorkingHeight * kWorkingWidth;
constexpr std::size_t kLqStartupWords = kRgbChannels * kLqPlaneWords;
constexpr std::size_t kLqGroupWords = kRgbChannels * 4U * kLqPlaneWords;
constexpr std::size_t kProjectedWords = 32640U * 1536U;
constexpr std::size_t kConditionWords = 4U * kProjectedWords;
constexpr std::size_t kLatentWords = 16U * 4U * 272U * 480U;
constexpr std::size_t kDecoderPackedWords = 784U * 272U * 480U;
constexpr std::size_t kMaxReadyFrames = 4U;
constexpr std::uint64_t kNoiseSeed = UINT64_C(0);

static_assert(kRgbFrameBytes == 24883200U, "fixed RGB frame extent changed");
static_assert(kLqStartupWords < std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t));
static_assert(kConditionWords < std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t));

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(std::string("FlashVSR+ video: ") + message);
}

void validateMode(FlashMode mode) {
  switch (mode) {
    case FlashMode::turbo:
    case FlashMode::base:
      return;
  }
  throw std::invalid_argument("FlashVSR+ video mode is not recognized");
}

void validateDiscreteGpu(ov::Core& core, const ov::RemoteContext& context) {
  const auto device = context.get_device_name();
  require(device.size() > 4U && device.compare(0U, 4U, "GPU.") == 0 &&
              std::all_of(device.begin() + 4U, device.end(),
                          [](char value) { return value >= '0' && value <= '9'; }),
          "an explicitly resolved GPU context is required");
  ov::device::Type type;
  try {
    type = core.get_property(device, ov::device::type);
  } catch (const ov::Exception&) {
    throw std::invalid_argument("FlashVSR+ video cannot resolve GPU device type");
  }
  require(type == ov::device::Type::DISCRETE,
          "FlashVSR+ video requires a discrete GPU context");
}

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
  std::uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

double uniformOpen(std::uint64_t& state) noexcept {
  // Midpoints exclude log(0); calls consume one continuous Gaussian stream.
  return (static_cast<double>(splitmix64(state) >> 12U) + 0.5) * 0x1p-52;
}

std::uint16_t roundBf16(float value) {
  if (!std::isfinite(value)) throw std::runtime_error("FlashVSR+ video noise is non-finite");
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const auto result = static_cast<std::uint16_t>(
      (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
  if ((result & 0x7f80U) == 0x7f80U)
    throw std::runtime_error("FlashVSR+ video noise overflows BF16");
  return result;
}

void copyHostToRemote(ov::Tensor& remote, void* data) {
  require(data != nullptr, "remote upload has no host data");
  const ov::Tensor host(remote.get_element_type(), remote.get_shape(), data);
  auto destination = remote.as<ov::RemoteTensor>();
  destination.copy_from(host);
}

}  // namespace

class FlashVideoController::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context,
       std::filesystem::path assets, std::filesystem::path scratch,
       FlashMode mode, std::size_t sourceFrames, Reader reader,
       BeforeGpu beforeGpu, Progress progress, bool useHostTileHistory)
      : core_(core), context_(std::move(context)), assets_(std::move(assets)),
        scratch_(std::move(scratch)), mode_(mode), useHostTileHistory_(useHostTileHistory),
        geometry_(makeFlashGeometry(kVisibleHeight, kVisibleWidth)),
        plan_(sourceFrames, 4), source_(plan_, std::move(reader)),
        beforeGpu_(std::move(beforeGpu)), progress_(std::move(progress)) {
    validateMode(mode_);
    validateDiscreteGpu(core_, context_);
  }

  ~Impl() = default;

  std::optional<SourceFrame> next() {
    if (failed_) throw std::logic_error("FlashVSR+ video controller has failed");
    if (!ready_.empty()) return popReady();
    try {
      while (true) {
        if (!active_) {
          if (source_.complete()) {
            require(publishedCount_ == plan_.outputFrames(),
                    "source reached EOS before all decoder frames were published");
            return std::nullopt;
          }
          const auto batch = source_.next();
          validateBatch(batch);
          // Device/model work starts only after geometry, reader output, and
          // all retained source frames in this first window are validated.
          callBeforeGpu();
          report(batch.callIndex, 0.0);
          ensureSessions(batch);
          prepareCall(batch);
        }

        if (nextParent_ < std::min(active_->latentStop, plan_.requestedLatents())) {
          runDecoderParent(nextParent_++);
          if (!ready_.empty()) return popReady();
          continue;
        }

        active_.reset();
        std::vector<std::uint16_t>().swap(latent_);
      }
    } catch (...) {
      fail();
      throw;
    }
  }

 private:
  void callBeforeGpu() {
    if (beforeGpu_) beforeGpu_();
  }

  void report(std::size_t callIndex, double fraction) {
    require(callIndex < plan_.callCount(), "progress call index exceeds the stream");
    const auto bounded = std::clamp(fraction, 0.0, 1.0);
    const auto whole = std::clamp(
        (static_cast<double>(callIndex) + bounded) /
            static_cast<double>(plan_.callCount()),
        0.0, 1.0);
    const auto monotonic = std::max(lastProgress_, whole);
    if (progress_ && (!progressStarted_ || monotonic > lastProgress_))
      progress_(monotonic);
    lastProgress_ = monotonic;
    progressStarted_ = true;
  }

  void validateBatch(const FlashSourceBatch& batch) const {
    require(batch.modelStop > batch.modelStart, "source window is empty");
    require(batch.modelStop - batch.modelStart <= 20U, "source window exceeds its 20-frame bound");
    for (const auto& source : batch.frames) {
      require(source.frame.width == static_cast<int>(kVisibleWidth) &&
                  source.frame.height == static_cast<int>(kVisibleHeight),
              "reader frame geometry differs from the fixed UHD contract");
      require(source.frame.pixels.size() == kRgbFrameBytes,
              "reader frame is not contiguous RGB24 at the fixed geometry");
    }
  }

  void ensureSessions(const FlashSourceBatch& firstBatch) {
    if (denoiser_) return;
    require(firstBatch.callIndex == 0U && firstBatch.modelStart == 0U,
            "the first Flash call does not start at model frame zero");

    // Keep only the decoder's persistent roots while the other phases are
    // admitted. Its compiled requests/workspaces are not needed during LQ or
    // denoising and are explicitly retired before the next owner is built.
    decoder_ = std::make_unique<FlashDecoderSession>(
        core_, context_, assets_, kVisibleHeight, kVisibleWidth, plan_.sourceFrames(),
        useHostTileHistory_);
    decoder_->suspend();

    lq_ = std::make_unique<FlashLqSession>(core_, context_, assets_, mode_);
    {
      std::vector<std::uint16_t> startupPacked(kLqStartupWords);
      auto startup = flashDeviceTensor(context_, ov::element::bf16,
                                       {1U, kRgbChannels, 1U, kWorkingHeight, kWorkingWidth});
      packStartup(0U, startup, startupPacked);
      lq_->start(startup);
      lq_->suspend();
    }
    // Denoiser construction reserves its bounded history and then suspends
    // its per-call stages. It is deliberately last: LQ startup roots above
    // are already gone and only the causal LQ/decoder owners remain.
    denoiser_ = std::make_unique<FlashDenoiserSession>(
        core_, context_, assets_, scratch_, mode_, plan_.sourceFrames());
  }

  const std::uint8_t* sourcePixels(std::size_t modelFrame) const {
    const auto& source = source_.frame(modelFrame);
    require(source.frame.pixels.size() == kRgbFrameBytes, "source pixels changed during packing");
    return source.frame.pixels.data();
  }

  void packStartup(std::size_t modelFrame, ov::Tensor& startup,
                   std::vector<std::uint16_t>& packed) {
    std::array<const std::uint8_t*, 4> rgb{};
    rgb[0] = sourcePixels(modelFrame);
    packFlashLqRgb(rgb, 1U, kRgbFrameBytes, geometry_, packed.data(), kLqStartupWords);
    copyHostToRemote(startup, packed.data());
  }

  void packLqGroup(std::size_t firstModelFrame, ov::Tensor& group,
                   std::vector<std::uint16_t>& packed) {
    std::array<const std::uint8_t*, 4> rgb{};
    for (std::size_t slot = 0U; slot < 4U; ++slot)
      rgb[slot] = sourcePixels(firstModelFrame + slot);
    packFlashLqRgb(rgb, 4U, kRgbFrameBytes, geometry_, packed.data(), kLqGroupWords);
    copyHostToRemote(group, packed.data());
  }

  void prepareCall(const FlashSourceBatch& batch) {
    const auto call = plan_.call(batch.callIndex);
    const auto expectedModelStart = call.index == 0U
        ? call.projection.start : call.projection.start - 4U;
    require(expectedModelStart == batch.modelStart &&
                call.projection.stop == batch.modelStop,
            "source window differs from the planned call");
    require(batch.last == call.last, "source window terminal flag differs from the plan");

    std::vector<std::uint16_t> condition(kConditionWords, 0U);
    const auto firstProjection = batch.callIndex == 0U ? 1U : call.projection.start;
    {
      auto lqGroup = flashDeviceTensor(context_, ov::element::bf16,
                                       {1U, kRgbChannels, 4U, kWorkingHeight, kWorkingWidth});
      auto lqProjected = flashDeviceTensor(context_, ov::element::bf16,
                                           {1U, 32640U, 1536U});
      std::vector<std::uint16_t> lqPacked(kLqGroupWords);
      for (std::size_t group = 0U; group < 4U; ++group) {
        packLqGroup(firstProjection + group * 4U, lqGroup, lqPacked);
        lq_->project(lqGroup, lqProjected);
        ov::Tensor conditionView(ov::element::bf16, {1U, 32640U, 1536U},
                                 condition.data() + group * kProjectedWords);
        lqProjected.copy_to(conditionView);
        report(batch.callIndex, 0.25 * static_cast<double>(group + 1U) / 4.0);
      }
      lq_->suspend();
    }

    auto noise = fillNoise();
    latent_ = denoiser_->run({noise.data(), noise.size()},
                              {condition.data(), condition.size()},
                              [&](double globalFraction) {
                                const auto localFraction = std::clamp(
                                    globalFraction * static_cast<double>(plan_.callCount()) -
                                        static_cast<double>(batch.callIndex),
                                    0.0, 1.0);
                                report(batch.callIndex, 0.25 + 0.5 * localFraction);
                              });
    denoiser_->suspend();
    require(latent_.size() == kLatentWords, "denoiser latent extent differs from decoder contract");

    const auto decoderStop = std::min(call.latentStop, plan_.requestedLatents());
    require(decoderStop > call.latentStart, "planned call has no requested decoder parent");
    active_ = call;
    nextParent_ = call.latentStart;
  }

  std::vector<std::uint16_t> fillNoise() {
    std::vector<std::uint16_t> noise(kLatentWords);
    for (std::size_t index = 0U; index < noise.size(); index += 2U) {
      const double radius = std::sqrt(-2.0 * std::log(uniformOpen(noiseSeed_)));
      const double angle = 6.283185307179586476925286766559 * uniformOpen(noiseSeed_);
      noise[index] = roundBf16(static_cast<float>(radius * std::cos(angle)));
      noise[index + 1U] = roundBf16(static_cast<float>(radius * std::sin(angle)));
    }
    return noise;
  }

  void runDecoderParent(std::size_t parent) {
    require(active_.has_value(), "decoder parent has no active source call");
    // The first parent follows LQ/denoising in this same call. Later parents
    // follow a returned frame, so only those handoffs need the callback.
    if (parent != active_->latentStart) callBeforeGpu();
    std::array<const std::uint8_t*, 4> rgb{};
    for (std::size_t slot = 0U; slot < 4U; ++slot) {
      const auto raw = 4U * parent + slot;
      const auto modelFrame = raw < 3U ? 0U : raw - 3U;
      rgb[slot] = sourcePixels(modelFrame);
    }
    std::vector<std::uint16_t> decoderPackedHost(kDecoderPackedWords);
    auto decoderPacked = flashDeviceTensor(context_, ov::element::bf16,
                                           {1U, 784U, 272U, 480U});
    packFlashDecoderParentRgb(rgb, kRgbFrameBytes, geometry_, latent_.data(),
                              latent_.size(), parent - active_->latentStart,
                              decoderPackedHost.data(), decoderPackedHost.size());
    copyHostToRemote(decoderPacked, decoderPackedHost.data());

    const auto localParent = parent - active_->latentStart;
    const auto parentCount = std::min(active_->latentStop, plan_.requestedLatents()) - active_->latentStart;
    decoder_->run(decoderPacked,
                  [&](std::size_t index, const float* planar, std::size_t elements) {
                    publish(index, planar, elements);
                  },
                  [&](std::size_t completed, std::size_t total) {
                    const double fraction = total == 0U ? 1.0 :
                        (static_cast<double>(localParent) +
                         static_cast<double>(completed) / static_cast<double>(total)) /
                        static_cast<double>(parentCount);
                    report(active_->index, 0.75 + 0.25 * fraction);
                  });
    decoder_->suspend();
  }

  void publish(std::size_t index, const float* planar, std::size_t elements) {
    require(planar != nullptr && elements == kRgbFrameBytes,
            "decoder frame extent differs from the visible RGB contract");
    require(active_.has_value() && index == publishedCount_,
            "decoder published frames out of order");
    require(ready_.size() < kMaxReadyFrames, "decoder output queue exceeded four frames");

    const auto& source = source_.frame(index);
    require(source.index == publishedCount_,
            "decoder published source frames out of order");
    SourceFrame output;
    output.index = source.index;
    output.cutBefore = source.cutBefore;
    output.frame.width = static_cast<int>(kVisibleWidth);
    output.frame.height = static_cast<int>(kVisibleHeight);
    output.frame.pixels.resize(kRgbFrameBytes);
    const auto plane = kVisibleHeight * kVisibleWidth;
    for (std::size_t pixel = 0U; pixel < plane; ++pixel) {
      for (std::size_t channel = 0U; channel < kRgbChannels; ++channel) {
        const float value = planar[channel * plane + pixel];
        require(std::isfinite(value), "decoder produced a non-finite frame");
        const float clipped = std::clamp(value, -1.0F, 1.0F);
        output.frame.pixels[pixel * kRgbChannels + channel] =
            // Preserve the original F32 scaling, then round the bounded
            // nonnegative value without a libm call for each UHD channel.
            static_cast<std::uint8_t>(static_cast<double>((clipped + 1.0F) * 127.5F) + 0.5);
      }
    }
    ready_.push_back(std::move(output));
    ++publishedCount_;
  }

  SourceFrame popReady() {
    SourceFrame result = std::move(ready_.front());
    ready_.pop_front();
    return result;
  }

  void fail() noexcept {
    failed_ = true;
    if (lq_) lq_->suspend();
    if (denoiser_) denoiser_->suspend();
    if (decoder_) decoder_->suspend();
  }

  ov::Core core_;
  ov::RemoteContext context_;
  std::filesystem::path assets_, scratch_;
  FlashMode mode_;
  const bool useHostTileHistory_;
  FlashGeometry geometry_;
  flash::StreamPlan plan_;
  FlashSourceWindow source_;
  BeforeGpu beforeGpu_;
  Progress progress_;

  std::unique_ptr<FlashLqSession> lq_;
  std::unique_ptr<FlashDenoiserSession> denoiser_;
  std::unique_ptr<FlashDecoderSession> decoder_;
  std::vector<std::uint16_t> latent_;
  std::deque<SourceFrame> ready_;
  std::optional<flash::StreamCall> active_;
  std::size_t nextParent_ = 0U;
  std::uint64_t noiseSeed_ = kNoiseSeed;
  std::size_t publishedCount_ = 0U;
  double lastProgress_ = 0.0;
  bool progressStarted_ = false;
  bool failed_ = false;
};

FlashVideoController::FlashVideoController(
    ov::Core& core, ov::RemoteContext context, std::filesystem::path assets,
    std::filesystem::path scratch, FlashMode mode, std::size_t sourceFrames,
    Reader reader, BeforeGpu beforeGpu, Progress progress, bool useHostTileHistory)
    : impl_(std::make_unique<Impl>(core, std::move(context), std::move(assets),
                                   std::move(scratch), mode, sourceFrames,
                                   std::move(reader), std::move(beforeGpu),
                                   std::move(progress), useHostTileHistory)) {}
FlashVideoController::~FlashVideoController() = default;
FlashVideoController::FlashVideoController(FlashVideoController&&) noexcept = default;
FlashVideoController& FlashVideoController::operator=(FlashVideoController&&) noexcept = default;
std::optional<SourceFrame> FlashVideoController::next() { return impl_->next(); }

}  // namespace otc::ai
