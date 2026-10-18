#include "flash_stream.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using otc::ai::flash::StreamPlan;

template <typename F>
void expectFailure(F&& action) {
  bool failed = false;
  try {
    action();
  } catch (const std::exception&) {
    failed = true;
  }
  assert(failed);
}

// Independent scalar oracle. It materializes only these small test fixtures;
// production StreamPlan keeps exactly the arithmetic needed for one call.
struct ReferencePlan {
  std::size_t effective;
  std::size_t expanded;
  std::size_t requested;
  std::size_t padded;
  std::size_t projection;
  std::size_t decoder;
  std::size_t merged;
  std::size_t calls;
  std::size_t repeat;
  std::vector<std::size_t> source;
};

ReferencePlan referencePlan(std::size_t sourceFrames, int compression) {
  const std::size_t c = static_cast<std::size_t>(compression);
  const std::size_t effective = ((sourceFrames - 1U + c - 1U) / c) * c + 1U;
  const std::size_t repeat = 4U / c;
  const std::size_t expanded = 1U + (effective - 1U) * repeat;
  const std::size_t requested = (expanded - 1U) / 4U + 1U;
  const std::size_t padded = ((requested + 3U) / 4U) * 4U;
  const std::size_t projection = 1U + padded * 4U;
  const std::size_t decoder = padded * 4U - 3U;
  // Materialize the finite expanded timeline, not padded decoder groups.
  const std::size_t merged = 1U + (expanded - 1U) / repeat;
  std::vector<std::size_t> source;
  source.reserve(expanded);
  source.push_back(0U);
  for (std::size_t index = 1U; index < effective; ++index) {
    for (std::size_t copy = 0U; copy < repeat; ++copy) {
      source.push_back(std::min(index, sourceFrames - 1U));
    }
  }
  return {effective, expanded, requested, padded, projection, decoder, merged,
          padded / 4U, repeat, std::move(source)};
}

void checkPlanAgainstReference(std::size_t sourceFrames, int compression) {
  const auto expected = referencePlan(sourceFrames, compression);
  const StreamPlan plan(sourceFrames, compression);
  assert(plan.sourceFrames() == sourceFrames);
  assert(plan.temporalCompression() == compression);
  assert(plan.repeatFactor() == expected.repeat);
  assert(plan.effectiveFrames() == expected.effective);
  assert(plan.expandedFrames() == expected.expanded);
  assert(plan.requestedLatents() == expected.requested);
  assert(plan.paddedLatents() == expected.padded);
  assert(plan.projectionFrames() == expected.projection);
  assert(plan.decoderFrames() == expected.decoder);
  assert(plan.mergedDecoderFrames() == expected.merged);
  assert(plan.outputFrames() == sourceFrames);
  assert(plan.callCount() == expected.calls);
  assert(plan.trimmedOutputRange(plan.mergedDecoderFrames()).stop == sourceFrames);

  for (std::size_t modelFrame = 0U; modelFrame < expected.projection; ++modelFrame) {
    const std::size_t expectedSource = modelFrame < expected.expanded
        ? expected.source[modelFrame] : sourceFrames - 1U;
    assert(plan.sourceIndex(modelFrame) == expectedSource);
  }
  for (std::size_t index = 0U; index < expected.calls; ++index) {
    const auto call = plan.call(index);
    const std::size_t latentStart = index * 4U;
    const std::size_t latentStop = latentStart + 4U;
    const std::size_t projectionStart = index == 0U ? 0U : 1U + latentStart * 4U;
    const std::size_t projectionStop = 1U + latentStop * 4U;
    const std::size_t modelStart = latentStart * 4U;
    const std::size_t decoderStart = modelStart < 3U ? 0U : modelStart - 3U;
    const std::size_t decoderStop = latentStop * 4U - 3U;
    const std::size_t firstMerged = 1U + 12U / expected.repeat;
    const std::size_t steadyMerged = 16U / expected.repeat;
    const std::size_t mergedStart = index == 0U
        ? 0U : firstMerged + (index - 1U) * steadyMerged;
    const std::size_t mergedStop =
        mergedStart + (index == 0U ? firstMerged : steadyMerged);
    assert(call.index == index);
    assert(call.latentStart == latentStart && call.latentStop == latentStop);
    assert(call.frameOffset == projectionStart);
    assert(call.projection.start == projectionStart &&
           call.projection.stop == projectionStop);
    assert(call.decoder.start == decoderStart && call.decoder.stop == decoderStop);
    assert(call.mergedOutput.start == std::min(mergedStart, expected.merged) &&
           call.mergedOutput.stop == std::min(mergedStop, expected.merged));
    assert(call.last == (index + 1U == expected.calls));
  }
}

void testAllSmallCountsAndCompressionModes() {
  for (std::size_t sourceFrames = 1U; sourceFrames <= 128U; ++sourceFrames) {
    for (const int compression : {1, 2, 4}) {
      checkPlanAgainstReference(sourceFrames, compression);
    }
  }
}

void testSourceTailAndStartupBoundary() {
  const StreamPlan plan(17U);
  const auto first = plan.call(0U);
  const auto second = plan.call(1U);
  assert(first.projection.count() == 17U && first.decoder.count() == 13U);
  assert(second.projection.count() == 16U && second.decoder.count() == 16U);
  assert(first.mergedOutput.start == 0U && first.mergedOutput.stop == 13U);
  assert(second.mergedOutput.start == 13U && second.mergedOutput.stop == 17U);
  assert(!first.last && second.last);
  for (std::size_t frame = 17U; frame < plan.projectionFrames(); ++frame) {
    assert(plan.sourceIndex(frame) == 16U);
  }
  assert(plan.trimmedOutputRange(plan.mergedDecoderFrames()).stop == 17U);
}

void testPaddedDecoderTailIsNotPublished() {
  // Independent explicit source-clock expectations from the finite expansion,
  // including a rounded-up effective tail that must be trimmed AFTER merging.
  for (const auto [frames, compression, expanded, merged] : {
           std::array<std::size_t, 4>{49U, 4U, 49U, 49U},
           std::array<std::size_t, 4>{9U, 2U, 17U, 9U},
           std::array<std::size_t, 4>{6U, 2U, 13U, 7U},
           std::array<std::size_t, 4>{6U, 1U, 21U, 6U},
           std::array<std::size_t, 4>{1U, 1U, 1U, 1U}}) {
    const StreamPlan plan(frames, static_cast<int>(compression));
    assert(plan.expandedFrames() == expanded && plan.mergedDecoderFrames() == merged);
    assert(plan.decoderFrames() >= expanded);
    std::size_t next = 0;
    for (std::size_t i = 0; i < plan.callCount(); ++i) {
      const auto range = plan.call(i).mergedOutput;
      assert(range.start == next && range.stop >= range.start && range.stop <= merged);
      next = range.stop;
    }
    assert(next == merged && plan.trimmedOutputRange(next).count() == frames);
  }
}

void testLargePlanUsesOnlyScalarState() {
  const StreamPlan plan(otc::ai::flash::kMaxSourceFrames, 1);
  assert(plan.callCount() > 500'000U);
  assert(sizeof(plan) <= 32U);
  const std::size_t last = plan.callCount() - 1U;
  const auto check = [&](std::size_t index) {
    const auto call = plan.call(index);
    assert(call.projection.count() == (index == 0U ? 17U : 16U));
    assert(call.decoder.count() == (index == 0U ? 13U : 16U));
    assert(call.frameOffset == (index == 0U ? 0U : 1U + index * 16U));
    assert(call.last == (index == last));
    assert(plan.sourceIndex(call.projection.start) <= plan.outputFrames() - 1U);
  };
  for (const std::size_t index : {
           std::size_t{0U}, std::size_t{1U}, last / 3U, last / 2U, last - 1U, last}) {
    check(index);
  }
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (int round = 0; round < 32; ++round) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    check(static_cast<std::size_t>(state % plan.callCount()));
  }
}

void testInvalidAndBoundaryInputs() {
  expectFailure([] { StreamPlan(0U); });
  expectFailure([] { StreamPlan(otc::ai::flash::kMaxSourceFrames + 1U); });
  expectFailure([] { StreamPlan(std::numeric_limits<std::size_t>::max()); });
  for (const int compression : {0, 3, 5, -1}) {
    expectFailure([compression] { StreamPlan(1U, compression); });
  }
  const StreamPlan plan(17U);
  expectFailure([&] { (void)plan.call(plan.callCount()); });
  expectFailure([&] { (void)plan.sourceIndex(plan.projectionFrames()); });
  expectFailure([&] { (void)plan.trimmedOutputRange(plan.outputFrames() - 1U); });
}

}  // namespace

int main() {
  testAllSmallCountsAndCompressionModes();
  testSourceTailAndStartupBoundary();
  testPaddedDecoderTailIsNotPublished();
  testLargePlanUsesOnlyScalarState();
  testInvalidAndBoundaryInputs();
  return 0;
}
