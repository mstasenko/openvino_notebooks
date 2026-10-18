#pragma once

#include "flash_tokens.hpp"

#include <openvino/openvino.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace otc::ai {

// Owns one padded window-major BF16 root and one complete 4096-row BF16 graph
// input slab.  The caller supplies/consumes GPU tensors; this class owns no
// model, request, or host copy of the full token stream.
class FlashTokenSession final {
 public:
  FlashTokenSession(ov::RemoteContext context, FlashTokenGeometry geometry);
  ~FlashTokenSession() = default;
  FlashTokenSession(FlashTokenSession&&) noexcept = default;
  FlashTokenSession& operator=(FlashTokenSession&&) noexcept = default;
  FlashTokenSession(const FlashTokenSession&) = delete;
  FlashTokenSession& operator=(const FlashTokenSession&) = delete;

  // Start or restart a pass.  Canonical and condition are raw finite BF16
  // storage bits in canonical [frame,row,column,1536] order.
  void load(FlashBf16ConstBuffer canonical,
            FlashBf16ConstBuffer condition);

  // Read any validated range into the complete [1,4096,1536] input tensor.
  // The source ROI is always a full slab; terminal padding was zeroed by load.
  const ov::Tensor& inputSlab(std::size_t windowStart,
                              std::size_t windowCount);

  // Sequentially commit a POST slab.  postOutputRoot is a distinct same-
  // context USM_DEVICE BF16 [1,4096,1536] output root owned by the caller.
  void commit(std::size_t windowStart, std::size_t windowCount,
              const ov::Tensor& postOutputRoot);

  // Make the canonical root readable after a complete POST pass.  Repeated
  // reads are allowed, and the next commit pass starts again at slab zero.
  void read(FlashBf16Buffer canonical);

  const FlashTokenGeometry& geometry() const noexcept { return geometry_; }
  std::size_t windowCount() const noexcept { return windows_; }

 private:
  void validateLoaded(const char* operation) const;
  void* validateRemoteRoot(const ov::Tensor& tensor,
                           const ov::Shape& shape,
                          const char* label) const;
  ov::Tensor roi(const ov::Tensor& root, std::size_t rowStart,
                 std::size_t rowCount) const;
  std::size_t validWords(std::size_t windowStart,
                         std::size_t windowCount) const;

  FlashTokenGeometry geometry_;
  ov::RemoteContext context_;
  std::size_t tokens_ = 0U;
  std::size_t windows_ = 0U;
  std::size_t paddedWindows_ = 0U;
  ov::Shape rootShape_;
  ov::Shape slabShape_;
  ov::Tensor root_;
  ov::Tensor inputSlab_;
  std::vector<std::uint16_t> staging_;
  void* rootHandle_ = nullptr;
  void* inputSlabHandle_ = nullptr;
  bool loaded_ = false;
  bool passComplete_ = false;
  std::size_t nextCommit_ = 0U;
};

}  // namespace otc::ai
