#include "flash_attention_session.hpp"

#include "flash-assets/flash_attention_assets.hpp"
#include "flash_mask.hpp"
#include "flash_model.hpp"
#include "flash_ops.hpp"
#include "verified_kernel_files.hpp"

#include <openvino/runtime/intel_gpu/remote_properties.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace otc::ai {
namespace {

constexpr std::size_t kHeads = 12U, kWindow = 128U, kChannels = 1536U;
constexpr std::size_t kQueries = 16U, kSlab = 32U;
const ov::Shape kQueryShape{kHeads, kQueries, kWindow, kWindow};
const ov::Shape kKvSlab{kSlab, kWindow, kHeads, kWindow};

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(std::string("FlashVSR+ attention: ") + message);
}

std::size_t checkedWindows(std::size_t spatial) {
  // The qualified split kernel has 2550 key slots: three retained temporal
  // groups plus two current groups. This also bounds all host arithmetic.
  require(spatial > 0U && spatial <= 510U, "spatial window count is outside the qualified bound");
  return spatial;
}

void requireSlab(std::size_t start, std::size_t count, std::size_t total) {
  require(count > 0U && count <= kSlab && start < total && start % kSlab == 0U &&
              count <= total - start && (count == kSlab || start + count == total),
          "slabs must be aligned 32-window ranges with only a short terminal tail");
}

ov::RemoteTensor windowRoi(const ov::Tensor& root, std::size_t start, std::size_t count) {
  const auto shape = root.get_shape();
  require(shape.size() == 4U && start < shape[0] && count > 0U && count <= shape[0] - start,
          "K/V copy range differs");
  return ov::RemoteTensor(root.as<ov::RemoteTensor>(), {start, 0U, 0U, 0U},
                         {start + count, shape[1], shape[2], shape[3]});
}

// Import the exact pointer, NOT the allocation parameter map. Reusing a
// USM_DEVICE_BUFFER map allocates unrelated storage in the pinned plugin.
// This temporary non-owning view never enters FlashGpuStage. The complete
// owning root remains alive through the synchronous copy; no numeric F16
// conversion, detached alias, or untrusted pointer crosses this boundary.
ov::RemoteTensor rawView(ov::RemoteContext& context, const ov::Tensor& root,
                         ov::element::Type target) {
  require(root && root.is<ov::RemoteTensor>() &&
              (root.get_element_type() == ov::element::bf16 || root.get_element_type() == ov::element::f16) &&
              (target == ov::element::bf16 || target == ov::element::f16),
          "raw copy requires complete two-byte roots");
  const auto remote = root.as<ov::RemoteTensor>();
  const auto params = remote.get_params();
  require(remote.get_device_name() == context.get_device_name() &&
              params.at("OCL_CONTEXT").as<void*>() == context.get_params().at("OCL_CONTEXT").as<void*>() &&
              params.at(ov::intel_gpu::shared_mem_type.name()).as<ov::intel_gpu::SharedMemType>() ==
                  ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER,
          "raw source owner differs");
  void* const handle = params.at("MEM_HANDLE").as<void*>();
  require(handle != nullptr, "raw source has no storage");
  auto alias = context.create_tensor(target, root.get_shape(),
      {ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_USER_BUFFER),
       {"MEM_HANDLE", handle}});
  require(alias.get_params().at("MEM_HANDLE").as<void*>() == handle &&
              alias.get_shape() == root.get_shape() && alias.get_byte_size() == root.get_byte_size() &&
              alias.get_element_type() == target,
          "raw view did not retain the exact owning allocation");
  return alias;
}

void requireDisjoint(const std::uint16_t* k, const std::uint16_t* v, std::size_t elements) {
  require(k && v && elements > 0U && elements <= std::numeric_limits<std::size_t>::max() / sizeof(*k),
          "empty or oversized host history");
  const auto bytes = elements * sizeof(*k);
  const auto a = reinterpret_cast<std::uintptr_t>(k), b = reinterpret_cast<std::uintptr_t>(v);
  require(bytes <= std::numeric_limits<std::uintptr_t>::max() - a &&
              bytes <= std::numeric_limits<std::uintptr_t>::max() - b &&
              (a + bytes <= b || b + bytes <= a), "host K/V overlap or overflow");
}

} // namespace

class FlashAttentionSession::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context, const std::filesystem::path& assets,
       std::size_t spatial)
      : spatial_(checkedWindows(spatial)), windows_(2U * spatial_), capacity_(5U * spatial_),
        context_(std::move(context)),
        kernels_(VerifiedKernelFiles::open(assets, {kFlashSplitKeyConfig,
            {VerifiedKernelSourceSpec{kFlashSplitKeySource, 2U}, {}}, 1U})),
        attention_(core, context_, makeFlashSplitKeyAttentionModel(capacity_), {},
                   FlashPrecision::rawBf16Transport, kernels_.configPath()),
        finite_(core, context_, makeFlashBf16OutputCheck(kQueryShape), {},
                FlashPrecision::pluginDefault, {}, {"finite"}) {
    const ov::Shape kvShape{capacity_, kWindow, kHeads, kWindow};
    k_ = flashDeviceTensor(context_, ov::element::f16, kvShape);
    v_ = flashDeviceTensor(context_, ov::element::f16, kvShape);
    q_ = flashDeviceTensor(context_, ov::element::f16, kQueryShape);
    result_ = flashDeviceTensor(context_, ov::element::f16, kQueryShape);
    ids_ = flashDeviceTensor(context_, ov::element::i32, {kHeads, kQueries, 1U, capacity_});
    kSlab_ = flashDeviceTensor(context_, ov::element::bf16, kKvSlab);
    vSlab_ = flashDeviceTensor(context_, ov::element::bf16, kKvSlab);
    const auto slots = ((windows_ + kSlab - 1U) / kSlab) * 2U;
    queries_.reserve(slots);
    for (std::size_t i = 0U; i < slots; ++i)
      queries_.push_back(flashDeviceTensor(context_, ov::element::bf16, kQueryShape));
  }

  void beginLayer(const std::uint16_t* k, const std::uint16_t* v,
                  std::size_t keys, std::size_t elements) {
    require(!failed_ && (!begun_ || nextPost_ == windows_), "previous layer is incomplete or failed");
    require(keys <= capacity_ - windows_ && keys % spatial_ == 0U &&
                elements == keys * kWindow * kChannels,
            "history prefix extent differs");
    if (keys == 0U) require(!k && !v, "empty history must have null pointers");
    else {
      requireDisjoint(k, v, elements);
      validateFlashBf16Output(k, elements);
      validateFlashBf16Output(v, elements);
    }
    try {
      if (keys) {
        const ov::Shape shape{keys, kWindow, kHeads, kWindow};
        auto dstK = windowRoi(k_, 0U, keys), dstV = windowRoi(v_, 0U, keys);
        dstK.copy_from(ov::Tensor(ov::element::f16, shape, k));
        dstV.copy_from(ov::Tensor(ov::element::f16, shape, v));
      }
      prefix_ = keys;
      nextPre_ = nextPost_ = 0U;
      pending_ = 0U;
      consumed_ = false;
      begun_ = true;
    } catch (...) { failed_ = true; throw; }
  }

  FlashAttentionPreTargets preTargets(std::size_t start, std::size_t count) {
    require(begun_ && !failed_ && !consumed_ && !pending_ && start == nextPre_,
            "PRE target chronology differs");
    requireSlab(start, count, windows_);
    pending_ = count;
    return {queries_.at(start / kQueries), queries_.at(start / kQueries + 1U), kSlab_, vSlab_};
  }

  void completePre(std::size_t start, std::size_t count) {
    require(begun_ && !failed_ && !consumed_ && count > 0U && pending_ == count && start == nextPre_,
            "PRE completion differs from its pending slab");
    try {
      auto kView = rawView(context_, kSlab_, ov::element::f16);
      auto vView = rawView(context_, vSlab_, ov::element::f16);
      auto srcK = windowRoi(kView, 0U, count), srcV = windowRoi(vView, 0U, count);
      auto dstK = windowRoi(k_, prefix_ + start, count), dstV = windowRoi(v_, prefix_ + start, count);
      srcK.copy_to(dstK);
      srcV.copy_to(dstV);
      nextPre_ += count;
      pending_ = 0U;
    } catch (...) { failed_ = true; throw; }
  }

  void readRetained(std::uint16_t* k, std::uint16_t* v, std::size_t sourceStart,
                    std::size_t count, std::size_t elements) {
    require(begun_ && !failed_ && !pending_ && !consumed_ && nextPre_ == windows_,
            "retained K/V requires complete unconsumed PRE");
    require((sourceStart == 0U || sourceStart == spatial_) && count == windows_ - sourceStart &&
                elements == count * kWindow * kChannels, "retained K/V extent differs");
    requireDisjoint(k, v, elements);
    try {
      const ov::Shape shape{count, kWindow, kHeads, kWindow};
      auto srcK = windowRoi(k_, prefix_ + sourceStart, count);
      auto srcV = windowRoi(v_, prefix_ + sourceStart, count);
      ov::Tensor dstK(ov::element::f16, shape, k), dstV(ov::element::f16, shape, v);
      srcK.copy_to(dstK);
      srcV.copy_to(dstV);
    } catch (...) { failed_ = true; throw; }
  }

  void infer(const std::vector<std::uint8_t>& mask,
             const std::function<void(std::size_t, std::size_t)>& progress) {
    require(begun_ && !failed_ && !pending_ && !consumed_ && nextPre_ == windows_,
            "attention requires complete unconsumed PRE");
    const auto keys = prefix_ + windows_;
    require(mask.size() == kHeads * windows_ * keys &&
                std::all_of(mask.begin(), mask.end(), [](auto bit) { return bit <= 1U; }),
            "mask shape or boolean storage differs");
    // Scan only live keys; the packer adds the fixed device ABI's -1 padding.
    std::vector<std::uint8_t> selected(kHeads * kQueries * keys);
    try {
      const auto batches = (windows_ + kQueries - 1U) / kQueries;
      if (progress) progress(0U, batches);
      for (std::size_t start = 0U; start < windows_; start += kQueries) {
        const auto count = std::min(kQueries, windows_ - start);
        std::fill(selected.begin(), selected.end(), 0U);
        for (std::size_t h = 0U; h < kHeads; ++h) {
          for (std::size_t q = 0U; q < count; ++q)
            std::copy_n(mask.data() + (h * windows_ + start + q) * keys, keys,
                        selected.data() + (h * kQueries + q) * keys);
          // Padded queries are discarded by POST, but cannot select stale K/V.
          for (std::size_t q = count; q < kQueries; ++q)
            selected[(h * kQueries + q) * keys] = 1U;
        }
        auto packed = packFlashOrderedIds(selected, kHeads, kQueries, keys, capacity_);
        auto qView = rawView(context_, queries_.at(start / kQueries), ov::element::f16);
        qView.copy_to(q_);
        auto ids = ids_.as<ov::RemoteTensor>();
        ids.copy_from(ov::Tensor(ov::element::i32, ids_.get_shape(), packed.data()));
        attention_.run({{"Q", q_}, {"K", k_}, {"V", v_}, {"blocks", ids_}}, {{"attention", result_}});
        auto resultView = rawView(context_, result_, ov::element::bf16);
        auto& destination = queries_.at(start / kQueries);
        resultView.copy_to(destination);
        finite_.run({{"data", destination}}, {});
        if (progress) progress(start / kQueries + 1U, batches);
      }
      consumed_ = true;
    } catch (...) { failed_ = true; throw; }
  }

  std::pair<ov::Tensor, ov::Tensor> postTargets(std::size_t start, std::size_t count) {
    require(begun_ && !failed_ && consumed_ && start == nextPost_, "POST target chronology differs");
    requireSlab(start, count, windows_);
    auto result = std::make_pair(queries_.at(start / kQueries), queries_.at(start / kQueries + 1U));
    nextPost_ += count;
    return result;
  }

 private:
  std::size_t spatial_, windows_, capacity_;
  ov::RemoteContext context_;
  VerifiedKernelFiles kernels_;
  FlashGpuStage attention_, finite_;
  ov::Tensor k_, v_, q_, result_, ids_, kSlab_, vSlab_;
  std::vector<ov::Tensor> queries_;
  std::size_t prefix_ = 0U, nextPre_ = 0U, nextPost_ = 0U, pending_ = 0U;
  bool begun_ = false, consumed_ = false, failed_ = false;
};

FlashAttentionSession::FlashAttentionSession(ov::Core& core, ov::RemoteContext context,
    const std::filesystem::path& assets, std::size_t spatialWindows)
    : impl_(std::make_unique<Impl>(core, std::move(context), assets, spatialWindows)) {}
FlashAttentionSession::~FlashAttentionSession() = default;
FlashAttentionSession::FlashAttentionSession(FlashAttentionSession&&) noexcept = default;
FlashAttentionSession& FlashAttentionSession::operator=(FlashAttentionSession&&) noexcept = default;
void FlashAttentionSession::beginLayer(const std::uint16_t* k, const std::uint16_t* v,
    std::size_t keys, std::size_t elements) { impl_->beginLayer(k, v, keys, elements); }
FlashAttentionPreTargets FlashAttentionSession::preTargets(std::size_t start, std::size_t count) {
  return impl_->preTargets(start, count);
}
void FlashAttentionSession::completePre(std::size_t start, std::size_t count) { impl_->completePre(start, count); }
void FlashAttentionSession::readRetained(std::uint16_t* k, std::uint16_t* v,
    std::size_t start, std::size_t count, std::size_t elements) { impl_->readRetained(k, v, start, count, elements); }
void FlashAttentionSession::infer(const std::vector<std::uint8_t>& mask,
    const std::function<void(std::size_t, std::size_t)>& progress) { impl_->infer(mask, progress); }
std::pair<ov::Tensor, ov::Tensor> FlashAttentionSession::postTargets(std::size_t start, std::size_t count) {
  return impl_->postTargets(start, count);
}

} // namespace otc::ai
