#include "flash_source.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace otc::ai {
namespace {

constexpr std::size_t kOverlapModelFrames = 4U;
constexpr std::size_t kMaxRetainedFrames = 20U;

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

}  // namespace

FlashSourceWindow::FlashSourceWindow(flash::StreamPlan plan, Reader reader)
    : plan_(std::move(plan)), reader_(std::move(reader)) {
  require(static_cast<bool>(reader_), "Flash source reader is required");
  if (plan_.temporalCompression() != 4)
    throw std::invalid_argument("Flash source window requires compression 4");
  retained_.reserve(kMaxRetainedFrames);
}

void FlashSourceWindow::readThrough(std::size_t sourceIndex) {
  const auto lastSource = plan_.sourceFrames() - 1U;
  sourceIndex = std::min(sourceIndex, lastSource);
  while (nextSourceIndex_ <= sourceIndex) {
    if (retained_.size() == kMaxRetainedFrames)
      throw std::length_error("Flash source window exceeds its 20-frame bound");
    auto frame = reader_();
    if (!frame) throw std::runtime_error("source ended before the Flash window was filled");
    if (frame->index != static_cast<std::uint64_t>(nextSourceIndex_))
      throw std::runtime_error("source chronology differs from the Flash window");
    retained_.push_back(std::move(*frame));
    ++nextSourceIndex_;
  }
}

FlashSourceBatch FlashSourceWindow::next() {
  if (failed_) throw std::logic_error("Flash source window has failed");
  if (complete()) throw std::out_of_range("Flash source window is complete");

  try {
    const auto call = plan_.call(nextCall_);
    const std::size_t modelStart = call.index == 0U
        ? call.projection.start
        : call.projection.start - kOverlapModelFrames;
    const std::size_t modelStop = call.projection.stop;
    const auto requiredStart = plan_.sourceIndex(modelStart);
    const auto requiredStop = plan_.sourceIndex(modelStop - 1U);

    if (retained_.empty()) {
      retainedStart_ = requiredStart;
    } else {
      if (requiredStart < retainedStart_ || requiredStart > nextSourceIndex_)
        throw std::logic_error("Flash source overlap is not contiguous");
      const auto drop = requiredStart - retainedStart_;
      if (drop > retained_.size())
        throw std::logic_error("Flash source overlap exceeds retained owners");
      retained_.erase(retained_.begin(), retained_.begin() + static_cast<std::ptrdiff_t>(drop));
      retainedStart_ = requiredStart;
    }
    readThrough(requiredStop);
    if (retained_.empty() ||
        retained_.front().index != static_cast<std::uint64_t>(retainedStart_))
      throw std::logic_error("Flash source window lost its retained start");

    if (retained_.size() > kMaxRetainedFrames || modelStop - modelStart > kMaxRetainedFrames)
      throw std::length_error("Flash source window exceeds its 20-frame bound");

    activeModelStart_ = modelStart;
    activeModelStop_ = modelStop;
    ++nextCall_;
    return {call.index, modelStart, modelStop, call.last,
            retained_};
  } catch (...) {
    failed_ = true;
    throw;
  }
}

const SourceFrame& FlashSourceWindow::frame(std::size_t modelFrame) const {
  if (failed_) throw std::logic_error("Flash source window has failed");
  if (modelFrame < activeModelStart_ || modelFrame >= activeModelStop_)
    throw std::out_of_range("Flash model frame is outside the active source window");
  const auto source = plan_.sourceIndex(modelFrame);
  if (source < retainedStart_ || source - retainedStart_ >= retained_.size())
    throw std::logic_error("Flash model frame is outside the retained source owners");
  return retained_[source - retainedStart_];
}

}  // namespace otc::ai
