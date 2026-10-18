#include "seedvr_stream.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

using namespace otc::ai;

void checkBatch(std::size_t count, bool temporal, std::vector<std::size_t> cuts,
                const std::vector<std::size_t>& expectedBatches) {
  std::size_t reads = 0;
  std::vector<std::vector<RgbFrame>> batches;
  const auto read = [&]() -> std::optional<SourceFrame> {
    if (reads == count + 2) return std::nullopt;
    const auto index = reads++;
    const bool cut = std::find(cuts.begin(), cuts.end(), index) != cuts.end();
    return SourceFrame{index, {1, 1, {static_cast<std::uint8_t>(index), 0, 0}}, cut};
  };
  const auto render = [&](const std::vector<RgbFrame>& batch) {
    batches.push_back(batch);
    return batch;
  };
  SeedVrSourceProvider provider(count, temporal, read, render);
  std::size_t outputs = 0;
  while (const auto output = provider.next()) {
    assert(output->index == outputs);
    assert(output->frame.pixels[0] == outputs);
    assert(output->cutBefore == (std::find(cuts.begin(), cuts.end(), outputs) != cuts.end()));
    ++outputs;
  }
  assert(outputs == count && batches.size() == expectedBatches.size());
  for (std::size_t i = 0; i < batches.size(); ++i) {
    assert(batches[i].size() == expectedBatches[i]);
    for (const auto& frame : batches[i]) {
      if (frame.pixels[0] != batches[i][0].pixels[0]) {
        assert(std::find(cuts.begin(), cuts.end(), frame.pixels[0]) == cuts.end());
      }
    }
    for (std::size_t j = 1; j < batches[i].size(); ++j) {
      const auto previous = batches[i][j - 1].pixels[0];
      const auto current = batches[i][j].pixels[0];
      assert(current == previous || current == previous + 1);
    }
  }
  provider.drain();
  assert(reads == count + 2); // extra decoder frames drained, never rendered
}

int main() {
  for (std::size_t count : {1U, 2U, 3U, 4U, 5U}) {
    checkBatch(count, true, {}, {count == 1 ? 1U : 5U});
  }
  checkBatch(6, true, {}, {5, 1});
  checkBatch(7, true, {2, 5}, {5, 5, 5});
  checkBatch(4, false, {}, {1, 1, 1, 1});

  std::size_t rendered = 0;
  const auto render = [&](const std::vector<RgbFrame>& frames) {
    ++rendered;
    return frames;
  };
  SeedVrSourceProvider empty(1, true, [] { return std::optional<SourceFrame>{}; }, render);
  bool rejected = false;
  try { empty.next(); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected && rendered == 0);

  // Actual retry helper preserves the same padded batch through replacement.
  std::vector<RgbFrame> retryInputs(5, {1, 1, {17, 18, 19}});
  int attempts = 0;
  int replacements = 0;
  const auto retried = renderSourceBatchWithRetry(retryInputs,
      [&](const std::vector<RgbFrame>& inputs) {
        assert(&inputs == &retryInputs);
        assert(inputs[4].pixels == retryInputs[4].pixels);
        if (++attempts == 1) throw std::runtime_error("failed device");
        return inputs;
      },
      [&](const std::exception&) {
        ++replacements;
        return true;
      });
  assert(attempts == 2 && replacements == 1 && retried.size() == 5);
  assert(retried[0].pixels == retryInputs[0].pixels);

  std::size_t reads = 0;
  SeedVrSourceProvider truncated(5, true, [&]() -> std::optional<SourceFrame> {
    if (reads == 2) return std::nullopt;
    return SourceFrame{reads++, {1, 1, {0, 0, 0}}};
  }, render);
  rejected = false;
  try { truncated.next(); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected && rendered == 0); // failed batch cannot expose partial output

  SeedVrSourceProvider partial(1, true, []() -> std::optional<SourceFrame> {
    throw std::runtime_error("partial RGB24 frame");
  }, render);
  rejected = false;
  try { partial.next(); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected && rendered == 0);
}
