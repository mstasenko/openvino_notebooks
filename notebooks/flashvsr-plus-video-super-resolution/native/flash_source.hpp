#pragma once

#include "flash_stream.hpp"
#include "video_frame.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace otc::ai {

// A view over the one mutable source window.  The references remain valid
// until the next next() call; the caller must finish packing/consuming the
// view before advancing the window.  This avoids a second owned 20-frame
// batch while keeping the source chronology explicit.
struct FlashSourceBatch final {
  std::size_t callIndex = 0U;
  std::size_t modelStart = 0U;
  std::size_t modelStop = 0U;
  bool last = false;
  const std::vector<SourceFrame>& frames;
};

// Bounded source ownership for the fixed compression-4 Flash timeline.
// Call zero retains source frames 0..16; subsequent calls retain the four
// overlap frames and the next sixteen model frames, at most twenty owners.
// The reader is called only in source-index order and every cutBefore flag is
// carried unchanged in the retained SourceFrame.
class FlashSourceWindow final {
 public:
  using Reader = std::function<std::optional<SourceFrame>()>;

  FlashSourceWindow(flash::StreamPlan plan, Reader reader);

  // Advance exactly one planned four-latent call.  The returned view is
  // invalidated by the next call or by an exception.
  FlashSourceBatch next();

  std::size_t retainedCount() const noexcept { return retained_.size(); }
  std::size_t nextSourceIndex() const noexcept { return nextSourceIndex_; }
  std::size_t callIndex() const noexcept { return nextCall_; }
  bool complete() const noexcept { return nextCall_ == plan_.callCount(); }

  // Resolve one model/decoder frame in the current batch to its retained
  // source owner.  Only the active [modelStart, modelStop) range is accepted;
  // callers must finish using the returned owner before next() advances.
  const SourceFrame& frame(std::size_t modelFrame) const;

 private:
  void readThrough(std::size_t sourceIndex);

  flash::StreamPlan plan_;
  Reader reader_;
  std::vector<SourceFrame> retained_;
  std::size_t retainedStart_ = 0U;
  std::size_t nextSourceIndex_ = 0U;
  std::size_t nextCall_ = 0U;
  std::size_t activeModelStart_ = 0U;
  std::size_t activeModelStop_ = 0U;
  bool failed_ = false;
};

}  // namespace otc::ai
