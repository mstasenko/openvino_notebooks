#pragma once

#include "seedvr_assets.hpp"

#include <array>
#include <cstdint>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace otc::ai::detail {
constexpr std::uint64_t MiB = 1024 * 1024;
constexpr std::uint64_t kRetainedLimit = 2048 * MiB;
constexpr std::uint64_t kTransientReserve = 2048 * MiB;
constexpr std::uint64_t kMargin = 512 * MiB;
constexpr std::uint64_t kGlobalSpare = 2048 * MiB;
constexpr std::uint64_t kOneBlobLimit = 768 * MiB;

enum class ComponentStage { Encoder, Decoder, Prelude, Postlude };
struct ComponentKey {
  ComponentStage stage;
  std::size_t height, width;
  bool operator==(const ComponentKey& other) const {
    return stage == other.stage && height == other.height && width == other.width;
  }
};

// The only keys are fixed-session component geometry. No eviction or disk cache.
class ComponentCache {
 public:
  explicit ComponentCache(std::uint64_t limit = kRetainedLimit) : limit_(limit) {}

  bool admit(bool gpu, const SeedVrObservedHostMemory& memory) {
    if (!gpu || !memory.available || !memory.cgroupHeadroom || retained_ > limit_) {
      disable();
      return false;
    }
    const auto remaining = limit_ - retained_;
    const auto extra = kTransientReserve + kMargin;
    enabled_ = remaining <= UINT64_MAX - extra - kGlobalSpare &&
               *memory.cgroupHeadroom >= remaining + extra &&
               *memory.available >= remaining + extra + kGlobalSpare;
    if (!enabled_) disable();
    return enabled_;
  }

  const std::vector<char>* find(const ComponentKey& key) const {
    if (!enabled_) return nullptr;
    for (const auto& entry : entries_) {
      if (entry && entry->key == key) return &entry->bytes;
    }
    return nullptr;
  }

  bool wants(const ComponentKey& key) const {
    if (!enabled_ || key.height == 0 || key.width == 0) return false;
    if ((key.stage == ComponentStage::Encoder || key.stage == ComponentStage::Decoder) &&
        (key.height > 64 || key.width > 64)) return false;
    const auto range = slots(key.stage);
    for (std::size_t i = range.first; i < range.second; ++i) {
      if (!entries_[i] || entries_[i]->key == key) return true;
    }
    return false;
  }

  bool store(const ComponentKey& key, const std::string& exported,
             const SeedVrObservedHostMemory& memory) {
    if (!wants(key) || exported.empty() || exported.size() > kOneBlobLimit) return false;
    // Reject before the compact copy: observed exporter capacity plus new storage
    // must fit the separate transient bound even for oversized allocator growth.
    if (exported.capacity() > kTransientReserve ||
        exported.size() > kTransientReserve - exported.capacity() ||
        exported.size() > limit_ - retained_ || !memory.available || !memory.cgroupHeadroom ||
        *memory.cgroupHeadroom < exported.size() + kMargin ||
        *memory.available < exported.size() + kMargin + kGlobalSpare) return false;
    const auto range = slots(key.stage);
    for (std::size_t i = range.first; i < range.second; ++i) {
      if (entries_[i] && entries_[i]->key == key) return false;
      if (entries_[i]) continue;
      // Preserve bytes exactly while shedding observed exporter over-allocation.
      std::vector<char> compact;
      try {
        compact.assign(exported.begin(), exported.end());
      } catch (const std::bad_alloc&) {
        disable();
        return false;
      }
      const auto capacity = static_cast<std::uint64_t>(compact.capacity());
      if (retained_ > limit_ || capacity > limit_ - retained_) return false;
      entries_[i] = Entry{key, std::move(compact)};
      retained_ += capacity;
      return true;
    }
    return false;
  }

  std::uint64_t retainedCapacity() const { return retained_; }
  bool enabled() const { return enabled_; }
  void drop() { disable(); }

 private:
  struct Entry { ComponentKey key; std::vector<char> bytes; };
  void disable() {
    for (auto& entry : entries_) entry.reset();
    retained_ = 0;
    enabled_ = false;
  }
  static std::pair<std::size_t, std::size_t> slots(ComponentStage stage) {
    switch (stage) {
      case ComponentStage::Encoder: return {0, 2};
      case ComponentStage::Decoder: return {2, 4};
      case ComponentStage::Prelude: return {4, 5};
      case ComponentStage::Postlude: return {5, 6};
    }
    throw std::invalid_argument("unknown VAE blob stage");
  }
  std::array<std::optional<Entry>, 6> entries_;
  std::uint64_t limit_, retained_ = 0;
  bool enabled_ = false;
};

}  // namespace otc::ai::detail
