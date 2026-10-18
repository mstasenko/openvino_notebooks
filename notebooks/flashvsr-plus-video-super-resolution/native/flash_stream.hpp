#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace otc::ai::flash {

// This is the model-time schedule only. It owns no video, decoded pixels, or
// RIFE/output-clock state; the caller can keep one bounded buffer per call and
// trim the padded model tail before handing source frames to RIFE.
inline constexpr std::size_t kTemporalDownsample = 4U;
inline constexpr std::size_t kStreamChunkLatents = 4U;
inline constexpr std::size_t kMaxSourceFrames = 10'000'000U;

// The largest derived timeline is bounded by roughly source_frames * 4 plus
// the fixed first-frame/chunk padding. Keep that invariant explicit so the
// unchecked scalar arithmetic below cannot wrap on a supported size_t.
static_assert(kMaxSourceFrames <=
                  (std::numeric_limits<std::size_t>::max() - 32U) /
                      kTemporalDownsample,
              "Flash stream bound must fit derived timeline arithmetic");

struct StreamRange final {
  std::size_t start = 0U;
  std::size_t stop = 0U;

  std::size_t count() const noexcept { return stop - start; }
};

// Plain scalar data for one fixed four-latent call. Source mapping remains on
// StreamPlan so this object never carries per-frame vectors or hidden owners.
struct StreamCall final {
  std::size_t index = 0U;
  std::size_t latentStart = 0U;
  std::size_t latentStop = 0U;
  std::size_t frameOffset = 0U;
  StreamRange projection;
  StreamRange decoder;
  StreamRange mergedOutput;
  bool last = false;
};

class StreamPlan final {
 public:
  explicit StreamPlan(std::size_t sourceFrames, int temporalCompression = 4)
      : sourceFrames_(sourceFrames), temporalCompression_(temporalCompression) {
    if (sourceFrames_ == 0U || sourceFrames_ > kMaxSourceFrames) {
      throw std::invalid_argument("source frame count is outside the admitted bound");
    }
    if (temporalCompression_ != 1 && temporalCompression_ != 2 && temporalCompression_ != 4) {
      throw std::invalid_argument("temporal compression must be 1, 2, or 4");
    }
    paddedLatents_ = paddedLatentCount();
  }

  std::size_t sourceFrames() const noexcept { return sourceFrames_; }
  int temporalCompression() const noexcept { return temporalCompression_; }
  std::size_t repeatFactor() const noexcept {
    return kTemporalDownsample / static_cast<std::size_t>(temporalCompression_);
  }
  std::size_t effectiveFrames() const noexcept {
    const std::size_t beforeFirst = sourceFrames_ - 1U;
    const std::size_t compression = static_cast<std::size_t>(temporalCompression_);
    return ((beforeFirst + compression - 1U) / compression) * compression + 1U;
  }
  std::size_t expandedFrames() const noexcept {
    return 1U + (effectiveFrames() - 1U) * repeatFactor();
  }
  std::size_t requestedLatents() const noexcept {
    return (expandedFrames() - 1U) / kTemporalDownsample + 1U;
  }
  std::size_t paddedLatents() const noexcept { return paddedLatents_; }
  std::size_t projectionFrames() const noexcept {
    return 1U + paddedLatents_ * kTemporalDownsample;
  }
  std::size_t decoderFrames() const noexcept {
    return paddedLatents_ * kTemporalDownsample - (kTemporalDownsample - 1U);
  }
  std::size_t mergedDecoderFrames() const noexcept {
    // Python discards model padding before merging repeated-frame groups.
    // The effective source tail is trimmed separately after that merge.
    return effectiveFrames();
  }
  std::size_t outputFrames() const noexcept { return sourceFrames_; }
  std::size_t callCount() const noexcept {
    return paddedLatents_ / kStreamChunkLatents;
  }

  // Map one expanded or padded model frame to the original source frame.
  // Frames after the finite temporal expansion are the repeated tail used to
  // fill the final fixed-size latent call.
  std::size_t sourceIndex(std::size_t modelFrame) const {
    if (modelFrame >= projectionFrames()) {
      throw std::invalid_argument("model frame exceeds the planned projection timeline");
    }
    if (modelFrame == 0U) return 0U;
    if (modelFrame >= expandedFrames()) return sourceFrames_ - 1U;
    return std::min(
        1U + (modelFrame - 1U) / repeatFactor(), sourceFrames_ - 1U);
  }

  // Derive one call on demand. No call list or source-index table is retained.
  StreamCall call(std::size_t index) const {
    if (index >= callCount()) {
      throw std::invalid_argument("call index exceeds the planned call timeline");
    }
    const std::size_t latentStart = index * kStreamChunkLatents;
    const std::size_t latentStop = latentStart + kStreamChunkLatents;
    const std::size_t projectionStart = index == 0U
        ? 0U : 1U + latentStart * kTemporalDownsample;
    const std::size_t projectionStop = 1U + latentStop * kTemporalDownsample;
    const std::size_t modelStart = latentStart * kTemporalDownsample;
    const std::size_t decoderStart = modelStart < kTemporalDownsample - 1U
        ? 0U : modelStart - (kTemporalDownsample - 1U);
    const std::size_t decoderStop = latentStop * kTemporalDownsample -
                                    (kTemporalDownsample - 1U);
    const std::size_t firstMergedCount =
        1U + (kStreamChunkLatents * kTemporalDownsample - kTemporalDownsample) /
                 repeatFactor();
    const std::size_t steadyMergedCount =
        kStreamChunkLatents * kTemporalDownsample / repeatFactor();
    const std::size_t mergedStart = index == 0U
        ? 0U : firstMergedCount + (index - 1U) * steadyMergedCount;
    const std::size_t mergedStop = mergedStart +
                                   (index == 0U ? firstMergedCount : steadyMergedCount);
    return {index, latentStart, latentStop, projectionStart,
            {projectionStart, projectionStop}, {decoderStart, decoderStop},
            {std::min(mergedStart, mergedDecoderFrames()),
             std::min(mergedStop, mergedDecoderFrames())}, index + 1U == callCount()};
  }

  // Apply only AFTER discarding model padding and merging repeated frames.
  // This trims the effective source tail; it does not perform temporal merging.
  // RIFE's output clock remains separate.
  StreamRange trimmedOutputRange(std::size_t availableFrames) const {
    if (availableFrames < sourceFrames_) {
      throw std::invalid_argument("decoded output is shorter than the source timeline");
    }
    return {0U, sourceFrames_};
  }

 private:
  std::size_t paddedLatentCount() const noexcept {
    const std::size_t requested = requestedLatents();
    return ((requested + kStreamChunkLatents - 1U) / kStreamChunkLatents) *
           kStreamChunkLatents;
  }

  std::size_t sourceFrames_;
  int temporalCompression_;
  std::size_t paddedLatents_;
};

}  // namespace otc::ai::flash
