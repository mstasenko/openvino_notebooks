#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace otc::ai {

struct RgbFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;
};

struct SourceFrame {
  std::uint64_t index = 0;
  RgbFrame frame;
  bool cutBefore = false;
};

template <typename Render, typename Retry>
std::vector<RgbFrame> renderSourceBatchWithRetry(const std::vector<RgbFrame>& originals,
                                                Render render, Retry retry) {
  while (true) {
    try {
      return render(originals);
    } catch (const std::exception& error) {
      if (!retry(error)) throw;
    }
  }
}

// Enhances only original source frames, never interpolated output-clock frames.
// The reader supplies source-based scene flags; the renderer returns a complete
// batch before any member becomes visible to the output stream.
class SeedVrSourceProvider {
 public:
  using Reader = std::function<std::optional<SourceFrame>()>;
  using Renderer = std::function<std::vector<RgbFrame>(const std::vector<RgbFrame>&)>;

  SeedVrSourceProvider(std::uint64_t required, bool temporal, Reader read, Renderer render)
      : required_(required), temporal_(temporal), read_(std::move(read)),
        render_(std::move(render)) {
    if (!required_) throw std::invalid_argument("source frame count must be positive");
  }

  std::optional<SourceFrame> next() {
    if (cursor_ < ready_.size()) return std::move(ready_[cursor_++]);
    ready_.clear();
    cursor_ = 0;
    auto first = takeOriginal();
    if (!first) return std::nullopt;
    std::vector<SourceFrame> originals;
    originals.push_back(std::move(*first));
    while (temporal_ && originals.size() < 5) {
      auto next = takeOriginal();
      if (!next) break;
      if (next->cutBefore) {
        pending_ = std::move(next);
        break;
      }
      originals.push_back(std::move(*next));
    }
    std::vector<RgbFrame> inputs;
    for (const auto& original : originals) inputs.push_back(original.frame);
    // A true lone frame uses T1; tails of two to four use T5 then trim.
    if (temporal_ && inputs.size() > 1) {
      while (inputs.size() < 5) inputs.push_back(inputs.back());
    }
    auto rendered = render_(inputs);
    if (rendered.size() != inputs.size()) {
      throw std::runtime_error("renderer returned an invalid batch count");
    }
    for (std::size_t i = 0; i < originals.size(); ++i) {
      ready_.push_back({originals[i].index, std::move(rendered[i]), originals[i].cutBefore});
    }
    return std::move(ready_[cursor_++]);
  }

  void drain() {
    while (auto extra = read_()) {
      if (extra->index != readCount_++) throw std::runtime_error("source chronology differs");
    }
    if (readCount_ < required_) throw std::runtime_error("source ended before requested duration");
  }

 private:
  std::optional<SourceFrame> takeOriginal() {
    if (pending_) {
      auto result = std::move(pending_);
      pending_.reset();
      return result;
    }
    if (readCount_ >= required_) return std::nullopt;
    auto result = read_();
    if (!result) throw std::runtime_error("source ended before requested duration");
    if (result->index != readCount_++) throw std::runtime_error("source chronology differs");
    return result;
  }

  std::uint64_t required_, readCount_ = 0;
  bool temporal_;
  Reader read_;
  Renderer render_;
  std::optional<SourceFrame> pending_;
  std::vector<SourceFrame> ready_;
  std::size_t cursor_ = 0;
};

}  // namespace otc::ai
