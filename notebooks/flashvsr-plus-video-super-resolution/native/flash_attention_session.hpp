#pragma once

#include "flash_stage.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace otc::ai {

struct FlashAttentionPreTargets final {
  ov::Tensor q0, q1, k, v;
};

// One active layer's split-key attention. Learned PRE/POST banks belong to
// their existing sessions; this owner keeps only the current Q/context slots
// and one bounded K/V device pair. Historical layers remain in the caller's
// disk-backed store. All methods complete synchronously in the admitted context.
class FlashAttentionSession final {
 public:
  FlashAttentionSession(ov::Core& core, ov::RemoteContext context,
                        const std::filesystem::path& assets,
                        std::size_t spatialWindows = 510U);
  ~FlashAttentionSession();
  FlashAttentionSession(FlashAttentionSession&&) noexcept;
  FlashAttentionSession& operator=(FlashAttentionSession&&) noexcept;
  FlashAttentionSession(const FlashAttentionSession&) = delete;
  FlashAttentionSession& operator=(const FlashAttentionSession&) = delete;

  // Upload only the valid history prefix. Both pointers are null iff keys=0;
  // otherwise each contains keys*128*1536 finite BF16 storage words.
  void beginLayer(const std::uint16_t* k, const std::uint16_t* v,
                  std::size_t keys, std::size_t elements);
  FlashAttentionPreTargets preTargets(std::size_t start, std::size_t count);
  void completePre(std::size_t start, std::size_t count);

  // Read only a completed PRE suffix, before infer consumes its Q slots.
  // Destinations are disjoint, exactly sized host owners supplied by history.
  void readRetained(std::uint16_t* k, std::uint16_t* v,
                    std::size_t sourceStart, std::size_t count,
                    std::size_t elements);
  // mask is [12,2*spatialWindows,historyKeys+2*spatialWindows]. No unselected
  // or stale K/V capacity may enter the kernel. Completed context overwrites
  // consumed Q slots only after the distinct output request has finished.
  // Empty active rows produce zero, as in the qualified split-key graph.
  void infer(const std::vector<std::uint8_t>& mask,
             const std::function<void(std::size_t, std::size_t)>& progress = {});
  // The controller must abort the operation on any PRE/POST failure. A target
  // handoff is not an assertion that the caller's subsequent request succeeded.
  std::pair<ov::Tensor, ov::Tensor> postTargets(std::size_t start,
                                             std::size_t count);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace otc::ai
