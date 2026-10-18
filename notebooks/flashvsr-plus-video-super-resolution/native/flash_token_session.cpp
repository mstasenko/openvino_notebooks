#include "flash_token_session.hpp"

#include "flash_stage.hpp"

#include <openvino/runtime/intel_gpu/remote_properties.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace otc::ai {
namespace {

void require(bool value, const std::string& message) {
  if (!value) throw std::invalid_argument("Flash token session: " + message);
}

std::size_t checkedMultiply(std::size_t left, std::size_t right,
                            const char* label) {
  require(left == 0U || right <= std::numeric_limits<std::size_t>::max() / left,
          label);
  return left * right;
}

std::size_t bytesFor(const ov::Shape& shape) {
  std::size_t elements = 1U;
  for (const auto dimension : shape)
    elements = checkedMultiply(elements, dimension, "tensor shape overflows");
  return checkedMultiply(elements, sizeof(std::uint16_t),
                         "tensor byte extent overflows");
}

void* marker(const ov::AnyMap& parameters, const char* name) {
  const auto found = parameters.find(name);
  require(found != parameters.end(), std::string("missing remote marker ") + name);
  void* value = found->second.as<void*>();
  require(value != nullptr, std::string("empty remote marker ") + name);
  return value;
}

}  // namespace

FlashTokenSession::FlashTokenSession(ov::RemoteContext context,
                                     FlashTokenGeometry geometry)
    : geometry_(makeFlashTokenGeometry(geometry.frames, geometry.tokenHeight,
                                       geometry.tokenWidth)),
      context_(std::move(context)) {
  // The factory is deliberately the first operation: malformed geometry must
  // fail before touching the remote context or allocating device storage.
  tokens_ = geometry_.tokenCount();
  windows_ = geometry_.windowCount();
  paddedWindows_ = ((windows_ + kFlashTokenSlabWindows - 1U) /
                    kFlashTokenSlabWindows) * kFlashTokenSlabWindows;
  rootShape_ = {1U, paddedWindows_ * kFlashTokenWindowTokens,
                kFlashTokenChannels};
  slabShape_ = {1U, kFlashTokenSlabRows, kFlashTokenChannels};

  root_ = flashDeviceTensor(context_, ov::element::bf16, rootShape_);
  inputSlab_ = flashDeviceTensor(context_, ov::element::bf16, slabShape_);
  staging_.resize(kFlashTokenSlabRows * kFlashTokenChannels);
  rootHandle_ = validateRemoteRoot(root_, rootShape_, "window-major root");
  inputSlabHandle_ = validateRemoteRoot(inputSlab_, slabShape_, "input slab");
  require(rootHandle_ != inputSlabHandle_,
          "window-major root and input slab must not alias");
}

void FlashTokenSession::validateLoaded(const char* operation) const {
  require(loaded_, std::string(operation) + " requires a loaded root");
}

void* FlashTokenSession::validateRemoteRoot(
    const ov::Tensor& tensor, const ov::Shape& shape, const char* label) const {
  require(tensor && tensor.is<ov::RemoteTensor>(),
          std::string(label) + " must be a remote tensor");
  require(tensor.get_element_type() == ov::element::bf16 &&
              tensor.get_shape() == shape && tensor.get_byte_size() == bytesFor(shape),
          std::string(label) + " must be the expected BF16 shape");
  const auto remote = tensor.as<ov::RemoteTensor>();
  require(remote.get_device_name() == context_.get_device_name(),
          std::string(label) + " belongs to another device");
  require(marker(remote.get_params(), "OCL_CONTEXT") ==
              marker(context_.get_params(), "OCL_CONTEXT"),
          std::string(label) + " belongs to another context");
  const auto params = remote.get_params();
  const auto storage = params.find(ov::intel_gpu::shared_mem_type.name());
  require(storage != params.end(), std::string(label) + " has no GPU storage marker");
  const auto storageType = storage->second.as<ov::intel_gpu::SharedMemType>();
  require(storageType == ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER,
          std::string(label) + " must use USM_DEVICE_BUFFER");
  return marker(params, "MEM_HANDLE");
}

ov::Tensor FlashTokenSession::roi(const ov::Tensor& root,
                                  std::size_t rowStart,
                                  std::size_t rowCount) const {
  require(rowCount != 0U && rowStart <= root.get_shape().at(1) &&
              rowCount <= root.get_shape().at(1) - rowStart,
          "ROI exceeds root shape");
  return ov::RemoteTensor(root, ov::Coordinate{0U, rowStart, 0U},
                          ov::Coordinate{1U, rowStart + rowCount,
                                         kFlashTokenChannels});
}

std::size_t FlashTokenSession::validWords(std::size_t windowStart,
                                          std::size_t windowCount) const {
  return flashTokenSlabWords(geometry_, windowStart, windowCount);
}

void FlashTokenSession::load(FlashBf16ConstBuffer canonical,
                             FlashBf16ConstBuffer condition) {
  require(!loaded_ || passComplete_,
          "load requires an idle or completed pass");
  // Once a new pass is accepted, discard the previous pass state before any
  // source validation or remote copy.  A malformed source or failed upload
  // must never leave an apparently reusable completed root behind.
  loaded_ = false;
  passComplete_ = false;
  nextCommit_ = 0U;
  const auto canonicalWords = checkedMultiply(
      tokens_, kFlashTokenChannels, "canonical words overflow");
  require(canonical.words == canonicalWords && canonical.data != nullptr,
          "canonical storage has the wrong extent");
  require(condition.words == canonicalWords && condition.data != nullptr,
          "condition storage has the wrong extent");
  try {
    for (std::size_t start = 0U; start < windows_;
         start += kFlashTokenSlabWindows) {
      const auto count = std::min(kFlashTokenSlabWindows, windows_ - start);
      (void)validWords(start, count);
      // The terminal range is shorter than a full graph slab.  Uploading the
      // entire zeroed staging slab makes the padded root deterministic, so a
      // later full-slab inputSlab() read cannot expose a previous layer tail.
      std::fill(staging_.begin(), staging_.end(), std::uint16_t{0});
      flashCombineTokenSlab(geometry_, canonical, condition, start, count,
                            {staging_.data(), staging_.size()});
      ov::Tensor host(ov::element::bf16,
                      slabShape_, staging_.data());
      host.copy_to(roi(root_, start * kFlashTokenWindowTokens,
                       kFlashTokenSlabRows));
    }
    loaded_ = true;
  } catch (...) {
    loaded_ = false;
    passComplete_ = false;
    nextCommit_ = 0U;
    throw;
  }
}

const ov::Tensor& FlashTokenSession::inputSlab(std::size_t windowStart,
                                               std::size_t windowCount) {
  validateLoaded("inputSlab");
  (void)validWords(windowStart, windowCount);
  try {
    roi(root_, windowStart * kFlashTokenWindowTokens,
        kFlashTokenSlabRows).copy_to(inputSlab_);
    return inputSlab_;
  } catch (...) {
    loaded_ = false;
    passComplete_ = false;
    throw;
  }
}

void FlashTokenSession::commit(std::size_t windowStart,
                               std::size_t windowCount,
                               const ov::Tensor& postOutputRoot) {
  validateLoaded("commit");
  require(windowStart == nextCommit_, "post commits must be sequential");
  const auto valid = validWords(windowStart, windowCount) /
                     kFlashTokenChannels;
  const auto outputHandle = validateRemoteRoot(
      postOutputRoot, slabShape_, "post output root");
  require(outputHandle != rootHandle_ && outputHandle != inputSlabHandle_,
          "post output root must not alias session roots");
  try {
    roi(postOutputRoot, 0U, valid).copy_to(
        roi(root_, windowStart * kFlashTokenWindowTokens, valid));
    nextCommit_ = windowStart + windowCount;
    if (nextCommit_ == windows_) {
      nextCommit_ = 0U;
      passComplete_ = true;
    } else {
      passComplete_ = false;
    }
  } catch (...) {
    loaded_ = false;
    passComplete_ = false;
    nextCommit_ = 0U;
    throw;
  }
}

void FlashTokenSession::read(FlashBf16Buffer canonical) {
  validateLoaded("read");
  require(passComplete_, "read requires a complete POST pass");
  const auto canonicalWords = tokens_ * kFlashTokenChannels;
  require(canonical.data != nullptr && canonical.words == canonicalWords,
          "canonical destination has the wrong extent");
  try {
    for (std::size_t start = 0U; start < windows_;
         start += kFlashTokenSlabWindows) {
      const auto count = std::min(kFlashTokenSlabWindows, windows_ - start);
      const auto valid = count * kFlashTokenWindowTokens;
      ov::Tensor host(ov::element::bf16,
                      {1U, valid, kFlashTokenChannels}, staging_.data());
      roi(root_, start * kFlashTokenWindowTokens, valid).copy_to(host);
      flashScatterTokenSlab(geometry_, {staging_.data(), staging_.size()}, start,
                            count, canonical);
    }
  } catch (...) {
    loaded_ = false;
    passComplete_ = false;
    nextCommit_ = 0U;
    throw;
  }
}

}  // namespace otc::ai
