#include "flash_decoder_session.hpp"

#include "flash-assets/flash_decoder_assets.hpp"
#include "flash_decoder_frame.hpp"
#include "flash_decoder_model.hpp"
#include "flash_stream.hpp"

#include <openvino/runtime/intel_gpu/remote_properties.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace otc::ai {
namespace {

using Histories = std::array<ov::Tensor, 3>;

struct DecoderStage final {
  // Models borrow the authenticated structural bytes until their requests
  // retire. Learned banks (including child spans) are uploaded synchronously.
  SeedVrOwnedIr owner;
  std::unique_ptr<FlashGpuStage> gpu;

  template<class Bindings>
  DecoderStage(ov::Core& core, const ov::RemoteContext& context,
               const std::filesystem::path& assets,
               const SeedVrAssetPin& xml, const SeedVrAssetPin& bin,
               const Bindings& bindings, FlashDecoderRole role,
               FlashOutputCheck check)
      : owner(readSeedVrIr(assets, xml, bin, flashWeightSpans(bindings))) {
    const ov::Tensor bytes(ov::element::u8, {owner.binBytes()}, owner.binData());
    auto model = core.read_model(owner.xml(), bytes);
    validateFlashDecoderModel(*model, role, 2176, 3840);
    gpu = std::make_unique<FlashGpuStage>(core, context, model,
        flashStageWeights(bindings), FlashPrecision::pluginDefault, std::string{},
        std::vector<std::string>{}, check);
    gpu->bind(owner);
  }
};

const std::array<std::string, 6> kFrontStates{
    "mem_512_0", "mem_512_1", "mem_512_2", "mem_256_0", "mem_256_1", "mem_256_2"};
const ov::Shape kGrowthShape{1, 128, 92, 1920};
const ov::Shape kRawFrameShape{1, 3, 184, 3840};

// The large front histories/residuals and middle use host USM; their GPU
// kernels bind that storage directly, without a duplicate device I/O buffer.
// By default the 51-root tiled history ring, tile scratch, and child banks
// stay device USM. The opt-in policy moves only the alternating causal ring
// (past_ and free_) to host USM; compiled workspace and the caller's packed
// input are counted separately from fresh device statistics, before any of
// these large roots are created.
constexpr std::uint64_t kFrontHistoryBytes =
    3ULL * (512 * 272 * 480 + 256 * 544 * 960) * 2;
constexpr std::uint64_t kTileHistoryBytes = 128ULL * 92 * 1920 * 2;
// The alternating causal inputs contain one three-state history per lane,
// while free_ contains the three destinations for the next child. These are
// the only tiled roots moved to host USM by the opt-in policy; growth
// residuals, tile compute, and learned banks remain device-resident.
constexpr std::uint64_t kSwappedTileRootCount =
    3ULL * kFlashDecoderFrameLanes + 3ULL;
constexpr std::uint64_t kHostRootBytes = 3 * kFrontHistoryBytes + 256ULL * 544 * 960 * 2;
constexpr std::uint64_t kDeviceRootBytes = 256ULL * 46 * 960 * 2 +
    (kSwappedTileRootCount + 4ULL) * kTileHistoryBytes + 3ULL * 184 * 3840 * 2 +
    2ULL * (256 * 256 + 128 * 128) * 2;
static_assert(kHostRootBytes == 3'877'109'760ULL);
static_assert(kDeviceRootBytes == 2'514'268'160ULL);
static_assert(kSwappedTileRootCount == 51ULL);

} // namespace

class FlashDecoderSession::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context,
       const std::filesystem::path& assets,
       std::size_t height, std::size_t width, std::size_t frames,
       bool useHostTileHistory)
      : context_(std::move(context)), plan_(frames), height_(height), width_(width),
        useHostTileHistory_(useHostTileHistory) {
    if (!height || height > kFlashDecoderFrameWorkingHeight ||
        !width || width > kFlashDecoderFrameWorkingWidth)
      throw std::invalid_argument("FlashVSR+ decoder visible crop exceeds the fixed working raster");

    // Device counters below describe the default engine, not an arbitrary
    // imported/shared context. Admission must observe our own allocations.
    const auto marker = context_.get_params().at("OCL_CONTEXT").as<void*>();
    if (!marker || marker != core.get_default_context(context_.get_device_name())
                                 .get_params().at("OCL_CONTEXT").as<void*>())
      throw std::invalid_argument("FlashVSR+ decoder requires the admitted default GPU context");

    checkRoom(core, "before compilation");
    front_ = std::make_unique<DecoderStage>(core, context_, assets,
        kFlashDecoderFrontXml, kFlashDecoderFrontBin, kFlashDecoderFrontWeights,
        // Avoid retaining large GPU finite-reduction workspaces. Validation
        // inspects raw BF16; the GPU kernels still compute every front block.
        FlashDecoderRole::front, FlashOutputCheck::bf16Readback);
    checkRoom(core, "after front compilation");
    grow_ = std::make_unique<DecoderStage>(core, context_, assets,
        kFlashDecoderGrowXml, kFlashDecoderGrowBin, kFlashDecoderGrowWeights,
        FlashDecoderRole::grow, FlashOutputCheck::bf16);
    checkRoom(core, "after growth compilation");
    // The only frame readback is validated in its original BF16 domain by
    // normalizeAndScatterFlashDecoderTile, so no redundant GPU checker.
    frame_ = std::make_unique<DecoderStage>(core, context_, assets,
        kFlashDecoderFrameXml, kFlashDecoderFrameBin, kFlashDecoderFrameWeights,
        FlashDecoderRole::frame, FlashOutputCheck::none);
    checkRoom(core, "after frame compilation");

    for (std::size_t group = 0; group < 2; ++group) {
      const ov::Shape shape = group == 0 ? ov::Shape{1, 512, 272, 480} :
                                          ov::Shape{1, 256, 544, 960};
      auto zero = zeroHost(shape);
      for (std::size_t i = 3 * group; i < 3 * group + 3; ++i) {
        frontStates_[0][i] = allocateFront(shape);
        zero.copy_to(frontStates_[0][i]);
        frontStates_[1][i] = allocateFront(shape);
        frontResiduals_[i] = allocateFront(shape);
      }
    }
    middle_ = allocateFront({1, 256, 544, 960});
    compact_ = allocate({1, 256, 46, 960});
    growthMiddle_ = allocate(kGrowthShape);
    rawFrame_ = allocate(kRawFrameShape);
    hostRaw_ = ov::Tensor(ov::element::bf16, kRawFrameShape);
    for (std::size_t i = 0; i < 3; ++i) {
      free_[i] = allocateTileHistory();
      growthResiduals_[i] = allocate(kGrowthShape);
    }
    const auto zero = zeroHost(kGrowthShape);
    for (auto& lane : past_)
      for (auto& state : lane) {
        state = allocateTileHistory();
        zero.copy_to(state);
      }
    for (std::size_t i = 0; i < 2; ++i) {
      growthBanks_[i] = upload(grow_->owner, kFlashDecoderGrowChildSpans[i], {256, 256, 1, 1});
      frameBanks_[i] = upload(frame_->owner, kFlashDecoderFrameChildSpans[i], {128, 128, 1, 1});
    }
    for (auto& frame : frames_) frame.resize(3 * height_ * width_);
  }

  void run(const ov::Tensor& packed, const FrameConsumer& consume, const Progress& progress) {
    if (failed_) throw std::logic_error("FlashVSR+ decoder stream has failed");
    try {
      if (!consume || parent_ >= plan_.requestedLatents())
        throw std::logic_error("FlashVSR+ decoder consumer or parent count differs");
      if (progress) progress(0, past_.size());
      runFront(packed);
      for (std::size_t lane = 0; lane < past_.size(); ++lane) {
        const auto& tile = flashDecoderFrameTile(lane);
        // The source ROI is strided across channels. copy_to compacts its
        // rows into a complete same-context device USM root.
        const ov::Tensor roi(middle_, {0, 0, tile.compact_row_start, 0},
            {1, 256, tile.compact_row_start + tile.compact_row_count, 960});
        roi.copy_to(compact_);
        for (std::size_t child = 0; child < 2; ++child) {
          runGrowth(lane, child);
          for (std::size_t frameChild = 0; frameChild < 2; ++frameChild) {
            frame_->gpu->run({{"middle", growthMiddle_}, {"frame_grow", frameBanks_[frameChild]}},
                              {{"frame", rawFrame_}});
            rawFrame_.copy_to(hostRaw_);
            const auto slot = 2 * child + frameChild;
            // Check discarded startup/tail frames too before committing their
            // causal histories. Publication is trimmed only after all lanes.
            normalizeAndScatterFlashDecoderTile(
                static_cast<const std::uint16_t*>(hostRaw_.data()), hostRaw_.get_size(),
                lane, height_, width_, frames_[slot].data(), frames_[slot].size());
          }
          // S0/S1/S2 are inputs to source blocks17/18/19, not final block
          // outputs. Rotate only after their finite checks and both consumers.
          std::swap(past_[lane], free_);
        }
        if (progress) progress(lane + 1, past_.size());
      }
      for (std::size_t slot = 0; slot < frames_.size(); ++slot) {
        const auto rawIndex = 4 * parent_ + slot;
        if (rawIndex >= 3 && rawIndex - 3 < plan_.outputFrames())
          consume(rawIndex - 3, frames_[slot].data(), frames_[slot].size());
      }
      ++parent_;
    } catch (...) {
      failed_ = true;
      front_->gpu->invalidate();
      grow_->gpu->invalidate();
      frame_->gpu->invalidate();
      throw;
    }
  }

  void suspend() noexcept {
    front_->gpu->suspend();
    grow_->gpu->suspend();
    frame_->gpu->suspend();
  }

 private:
  void checkRoom(ov::Core& core, const char* phase) const {
    // Early rejection, not an atomic reservation or a first-inference peak
    // guarantee. Host-only USM is outside this device-capacity check.
    constexpr std::uint64_t reserve = 2ULL << 30;
    const auto device = context_.get_device_name();
    const auto total = core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE").as<std::uint64_t>();
    const auto stats = core.get_property(device, "GPU_MEMORY_STATISTICS").as<std::map<std::string, std::uint64_t>>();
    std::uint64_t used = 0;
    for (const auto* kind : {"usm_device", "usm_shared", "cl_mem", "unknown"}) {
      const auto found = stats.find(kind);
      if (found == stats.end() || found->second > std::numeric_limits<std::uint64_t>::max() - used)
        throw std::runtime_error("FlashVSR+ decoder memory statistics are unavailable");
      used += found->second;
    }
    const auto deviceRoots = useHostTileHistory_
        ? kDeviceRootBytes - kSwappedTileRootCount * kTileHistoryBytes
        : kDeviceRootBytes;
    if (total <= reserve || used > total - reserve || deviceRoots > total - reserve - used)
      throw std::runtime_error(std::string("Not enough GPU memory for the FlashVSR+ decoder ") +
          phase + ": current=" + std::to_string(used) +
          ", roots=" + std::to_string(deviceRoots) + ", total=" + std::to_string(total));
  }

  ov::Tensor allocate(const ov::Shape& shape) {
    return flashDeviceTensor(context_, ov::element::bf16, shape);
  }

  ov::Tensor allocateTileHistory() {
    if (useHostTileHistory_)
      return allocateFront(kGrowthShape);
    return allocate(kGrowthShape);
  }

  ov::Tensor allocateFront(const ov::Shape& shape) {
    return context_.create_tensor(ov::element::bf16, shape,
        {ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER)});
  }

  static ov::Tensor zeroHost(const ov::Shape& shape) {
    ov::Tensor value(ov::element::bf16, shape);
    std::memset(value.data(), 0, value.get_byte_size());
    return value;
  }

  ov::Tensor upload(const SeedVrOwnedIr& owner, const SeedVrWeightSpan& span,
                    const ov::Shape& shape) {
    validateFlashWeightExtent(shape, span);
    auto root = allocate(shape);
    const ov::Tensor host(ov::element::bf16, shape, owner.weightData(span));
    host.copy_to(root);
    return root;
  }

  void runFront(const ov::Tensor& packed) {
    FlashTensors inputs{{"packed", packed}}, outputs{{"middle", middle_}};
    const auto next = 1U - frontCurrent_;
    for (std::size_t i = 0; i < kFrontStates.size(); ++i) {
      inputs.emplace(kFrontStates[i], frontStates_[frontCurrent_][i]);
      outputs.emplace(kFrontStates[i] + "_out", frontStates_[next][i]);
      outputs.emplace("residual_linear_" + std::to_string(i), frontResiduals_[i]);
    }
    front_->gpu->run(inputs, outputs);
    frontCurrent_ = next;
  }

  void runGrowth(std::size_t lane, std::size_t child) {
    FlashTensors inputs{{"middle_in", compact_}, {"grow_weight", growthBanks_[child]}},
                 outputs{{"middle", growthMiddle_}};
    for (std::size_t i = 0; i < 3; ++i) {
      inputs.emplace("mem_128_" + std::to_string(i), past_[lane][i]);
      outputs.emplace("S" + std::to_string(i), free_[i]);
      outputs.emplace("residual_linear_" + std::to_string(6 + i), growthResiduals_[i]);
    }
    grow_->gpu->run(inputs, outputs);
  }

  ov::RemoteContext context_;
  flash::StreamPlan plan_;
  std::size_t height_, width_, parent_ = 0, frontCurrent_ = 0;
  const bool useHostTileHistory_;
  bool failed_ = false;
  std::array<std::array<ov::Tensor, 6>, 2> frontStates_;
  std::array<ov::Tensor, 6> frontResiduals_;
  ov::Tensor middle_, compact_, growthMiddle_, rawFrame_, hostRaw_;
  Histories free_, growthResiduals_;
  std::array<Histories, kFlashDecoderFrameLanes> past_;
  std::array<ov::Tensor, 2> growthBanks_, frameBanks_;
  std::array<std::vector<float>, 4> frames_;
  // Destroy requests (and all aliases they retain) before the root owners.
  std::unique_ptr<DecoderStage> front_, grow_, frame_;
};

FlashDecoderSession::FlashDecoderSession(ov::Core& core, ov::RemoteContext context,
    const std::filesystem::path& assets, std::size_t height, std::size_t width,
    std::size_t frames, bool useHostTileHistory)
    : impl_(std::make_unique<Impl>(core, std::move(context), assets, height, width,
                                   frames, useHostTileHistory)) {}
FlashDecoderSession::~FlashDecoderSession() = default;
FlashDecoderSession::FlashDecoderSession(FlashDecoderSession&&) noexcept = default;
FlashDecoderSession& FlashDecoderSession::operator=(FlashDecoderSession&&) noexcept = default;
void FlashDecoderSession::run(const ov::Tensor& packed, const FrameConsumer& consume,
                              const Progress& progress) {
  impl_->run(packed, consume, progress);
}
void FlashDecoderSession::suspend() noexcept { impl_->suspend(); }

} // namespace otc::ai
