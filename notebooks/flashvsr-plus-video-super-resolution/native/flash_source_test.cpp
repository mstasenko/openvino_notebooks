#include "flash_source.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using otc::ai::FlashSourceBatch;
using otc::ai::FlashSourceWindow;
using otc::ai::SourceFrame;
using otc::ai::RgbFrame;
using otc::ai::flash::StreamPlan;

namespace {

void require(bool condition, const char* expression) {
  if (!condition) throw std::runtime_error(std::string("flash source test failed: ") + expression);
}

#define REQUIRE(expression) require((expression), #expression)

std::vector<SourceFrame> makeSource(std::size_t count, const std::set<std::size_t>& cuts = {}) {
  std::vector<SourceFrame> result;
  result.reserve(count);
  for (std::size_t index = 0U; index < count; ++index)
    result.push_back({index, RgbFrame{1, 1, {static_cast<std::uint8_t>(index), 0U, 0U}},
                      cuts.find(index) != cuts.end()});
  return result;
}

template <typename F>
void expectFailure(F&& action) {
  bool failed = false;
  try { action(); } catch (const std::exception&) { failed = true; }
  REQUIRE(failed);
}

void checkBatch(const FlashSourceBatch& batch) {
  REQUIRE(batch.modelStop > batch.modelStart);
  REQUIRE(batch.frames.size() <= 20U);
  REQUIRE(!batch.frames.empty());
  for (std::size_t i = 1U; i < batch.frames.size(); ++i)
    REQUIRE(batch.frames[i].index == batch.frames.front().index + i);
}

void checkDecoderParents(FlashSourceWindow& window, const FlashSourceBatch& batch,
                         const StreamPlan& plan, const std::set<std::size_t>& cuts) {
  // FlashDecoderSession publishes raw decoder slot (4*parent + slot - 3),
  // while startup slots clamp to model frame zero.  Every such frame for this
  // four-latent call must resolve through the same source owner ring.
  for (std::size_t parent = batch.callIndex * 4U;
       parent < batch.callIndex * 4U + 4U; ++parent) {
    for (std::size_t slot = 0U; slot < 4U; ++slot) {
      const auto raw = 4U * parent + slot;
      const auto modelFrame = raw < 3U ? 0U : raw - 3U;
      REQUIRE(modelFrame >= batch.modelStart && modelFrame < batch.modelStop);
      const auto& frame = window.frame(modelFrame);
      REQUIRE(frame.index == plan.sourceIndex(modelFrame));
      REQUIRE(frame.cutBefore == (cuts.find(frame.index) != cuts.end()));
    }
  }
  for (std::size_t modelFrame = batch.modelStart; modelFrame < batch.modelStop; ++modelFrame) {
    const auto& frame = window.frame(modelFrame);
    REQUIRE(frame.index == plan.sourceIndex(modelFrame));
    REQUIRE(frame.cutBefore == (cuts.find(frame.index) != cuts.end()));
  }
}

void checkCount(std::size_t sourceCount, const std::set<std::size_t>& cuts) {
  const StreamPlan plan(sourceCount);
  auto source = makeSource(sourceCount, cuts);
  std::size_t cursor = 0U;
  FlashSourceWindow window(plan, [&]() -> std::optional<SourceFrame> {
    if (cursor == source.size()) return std::nullopt;
    return source[cursor++];
  });
  std::size_t calls = 0U;
  while (!window.complete()) {
    const auto batch = window.next();
    checkBatch(batch);
    checkDecoderParents(window, batch, plan, cuts);
    REQUIRE(batch.callIndex == calls++);
    for (const auto& frame : batch.frames) {
      REQUIRE(frame.index >= batch.frames.front().index);
      REQUIRE(frame.cutBefore == (cuts.find(frame.index) != cuts.end()));
    }
    if (batch.callIndex == 0U) {
      REQUIRE(batch.modelStart == 0U && batch.modelStop == 17U);
      REQUIRE(batch.frames.front().index == 0U);
    } else {
      REQUIRE(batch.modelStart + 4U == plan.call(batch.callIndex).projection.start);
      REQUIRE(batch.frames.front().index == plan.sourceIndex(batch.modelStart));
      expectFailure([&] { (void)window.frame(batch.modelStart - 1U); });
    }
  }
  REQUIRE(calls == plan.callCount());
  REQUIRE(window.retainedCount() <= 20U);
  REQUIRE(cursor == sourceCount);
  expectFailure([&] { (void)window.next(); });
}

void testEvictionAndStaleView() {
  const StreamPlan plan(49U);
  const auto source = makeSource(49U);
  std::size_t cursor = 0U;
  FlashSourceWindow window(plan, [&]() -> std::optional<SourceFrame> {
    if (cursor == source.size()) return std::nullopt;
    return source[cursor++];
  });
  const auto first = window.next();
  REQUIRE(window.frame(0U).index == 0U);
  REQUIRE(first.frames.front().index == 0U);
  const auto second = window.next();
  REQUIRE(second.frames.front().index == 13U);
  expectFailure([&] { (void)window.frame(0U); });
  REQUIRE(window.frame(second.modelStart).index == 13U);
  expectFailure([&] { (void)window.frame(second.modelStop); });
}

void testExpectedCountsAndCuts() {
  for (const std::size_t count : {1U, 9U, 13U, 16U, 17U, 49U})
    checkCount(count, {0U, count > 5U ? 5U : 0U, count > 13U ? 13U : 0U});
}

void testShortEofAndChronology() {
  const StreamPlan plan(17U);
  std::size_t cursor = 0U;
  FlashSourceWindow shortSource(plan, [&]() -> std::optional<SourceFrame> {
    if (cursor == 16U) return std::nullopt;
    return SourceFrame{cursor++, RgbFrame{1, 1, {0U, 0U, 0U}}, false};
  });
  expectFailure([&] { (void)shortSource.next(); });
  expectFailure([&] { (void)shortSource.next(); });

  FlashSourceWindow badOrder(plan, []() -> std::optional<SourceFrame> {
    return SourceFrame{1U, RgbFrame{1, 1, {0U, 0U, 0U}}, false};
  });
  expectFailure([&] { (void)badOrder.next(); });
}

void testInvalidReadersAndCompression() {
  expectFailure([&] { FlashSourceWindow(StreamPlan(1U), {}); });
  expectFailure([&] {
    FlashSourceWindow window(StreamPlan(1U, 2), [] {
      return std::optional<SourceFrame>{SourceFrame{0U, RgbFrame{1, 1, {0U, 0U, 0U}}, false}};
    });
  });
  expectFailure([&] {
    FlashSourceWindow window(StreamPlan(1U), [] { return std::optional<SourceFrame>{}; });
    (void)window.frame(0U);
  });
}

void testBoundedOverlap() {
  const StreamPlan plan(49U);
  const auto source = makeSource(49U);
  std::size_t cursor = 0U;
  FlashSourceWindow window(plan, [&]() -> std::optional<SourceFrame> {
    if (cursor == source.size()) return std::nullopt;
    return source[cursor++];
  });
  const auto first = window.next();
  REQUIRE(first.frames.size() == 17U);
  const auto second = window.next();
  REQUIRE(second.frames.size() == 20U && second.frames.front().index == 13U);
  const auto third = window.next();
  REQUIRE(third.frames.size() == 20U && third.frames.front().index == 29U);
  const auto fourth = window.next();
  REQUIRE(fourth.frames.size() == 4U && fourth.frames.front().index == 45U);
}

}  // namespace

int main() {
  testExpectedCountsAndCuts();
  testShortEofAndChronology();
  testInvalidReadersAndCompression();
  testBoundedOverlap();
  testEvictionAndStaleView();
  return 0;
}

#undef REQUIRE
