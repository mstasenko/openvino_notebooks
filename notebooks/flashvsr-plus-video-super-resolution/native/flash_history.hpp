#pragma once

#include "flash_mode.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace otc::ai {

inline constexpr std::size_t kFlashHistoryCurrentWindows = 2U;
inline constexpr std::size_t kFlashHistoryWindowRows = 128U;
inline constexpr std::size_t kFlashHistoryChannels = 1536U;
inline constexpr std::size_t kFlashHistoryPooledHeads = 12U;
inline constexpr std::size_t kFlashHistoryElementBytes = sizeof(std::uint16_t);
inline constexpr std::size_t kFlashHistoryMaxSpatialWindows = 510U;
inline constexpr std::uint64_t kFlashHistoryDiskReserveBytes = 16ULL << 30U;

struct FlashHistoryPolicy final {
  FlashMode mode;
  std::size_t layer;
  std::size_t retainedGroups;
  std::size_t localRange;
};

// Turbo keeps the reviewed local suffix for replacement layers and the full
// global history for layers 0, 6, ...; Base keeps the full history everywhere.
FlashHistoryPolicy flashHistoryPolicy(FlashMode mode, std::size_t layer);

struct FlashHistoryMetadata final {
  FlashHistoryPolicy policy;
  std::size_t spatialWindows;
  std::size_t capacityWindows;
  std::size_t validWindows;
  bool released;
};

using FlashHistoryReadRetained = std::function<void(
    std::uint16_t* k_destination, std::uint16_t* v_destination,
    std::size_t source_start_windows, std::size_t window_count)>;

struct FlashHistoryUpdate final {
  // Exactly two current temporal windows are required. Either current_k/v
  // are supplied directly, or read_retained fills the retained suffix in the
  // owner synchronously; the two input forms cannot be mixed.
  const std::uint16_t* current_k = nullptr;
  const std::uint16_t* current_v = nullptr;
  std::size_t current_windows = 0U;
  // Pooled F32 has shape [12,current_windows,128].  The owner keeps the
  // retained prefix and merges the current suffix alongside K/V, so callers
  // do not need a second temporal-history implementation.
  const float* current_pooled = nullptr;
  FlashHistoryReadRetained read_retained;
  // A terminal update returns the complete old-plus-current pooled mask and
  // releases the owner without synchronizing unused K/V pages.  It accepts no
  // K/V pointers or retained-read callback.
  bool terminal = false;
};

// A synchronous read-only view of the valid K/V prefix.  The mapped bytes are
// retained by this value, so the pointers remain valid until the view dies.
class FlashHistoryView final {
 public:
  FlashHistoryView() = default;

  const std::uint16_t* k() const noexcept { return k_; }
  const std::uint16_t* v() const noexcept { return v_; }
  std::size_t windows() const noexcept { return windows_; }
  // Pooled values are indexed as head * pooled_head_stride_elements() +
  // window * 128 + channel.  The logical prefix has pooled_elements() values;
  // the stride is the owner's fixed capacity, not the current prefix length.
  const float* pooled() const noexcept { return pooled_; }
  std::size_t pooled_elements() const noexcept { return pooled_elements_; }
  std::size_t pooled_head_stride_elements() const noexcept {
    return pooled_head_stride_elements_;
  }
  bool valid() const noexcept { return keep_alive_ != nullptr; }

 private:
  friend class FlashHistoryOwner;
  FlashHistoryView(const std::uint16_t* k, const std::uint16_t* v,
                   std::size_t windows, const float* pooled,
                   std::size_t pooled_elements,
                   std::size_t pooled_head_stride_elements,
                   std::shared_ptr<const void> keep_alive)
      : k_(k), v_(v), windows_(windows), pooled_(pooled),
        pooled_elements_(pooled_elements),
        pooled_head_stride_elements_(pooled_head_stride_elements),
        keep_alive_(std::move(keep_alive)) {}

  const std::uint16_t* k_ = nullptr;
  const std::uint16_t* v_ = nullptr;
  std::size_t windows_ = 0U;
  const float* pooled_ = nullptr;
  std::size_t pooled_elements_ = 0U;
  std::size_t pooled_head_stride_elements_ = 0U;
  std::shared_ptr<const void> keep_alive_;
};

// One layer's bounded disk-backed BF16 K/V owner.  It deliberately does not
// own a 30-layer map: the controller keeps one persistent owner per layer
// across calls and touches only the currently active pair.  flush() writes
// and advises the active mapping away while retaining the owner; retire() is a
// terminal operation after the final call, not a per-layer-boundary operation.
// Current updates are exactly two spatial windows; only the mode-selected
// retained suffix is copied into the owned files and pooled prefix.
class FlashHistoryOwner final {
 public:
  FlashHistoryOwner(const std::filesystem::path& scratch_parent,
                    FlashHistoryPolicy policy, std::size_t spatial_windows,
                    std::uint64_t disk_reserve_bytes = kFlashHistoryDiskReserveBytes);
  FlashHistoryOwner(const FlashHistoryOwner&) = delete;
  FlashHistoryOwner& operator=(const FlashHistoryOwner&) = delete;
  FlashHistoryOwner(FlashHistoryOwner&&) = delete;
  FlashHistoryOwner& operator=(FlashHistoryOwner&&) = delete;
  ~FlashHistoryOwner() noexcept;

  // Copy the finite current pair, retain its exact temporal suffix, and
  // synchronously persist both owned files before making the new view visible.
  // Pooled F32 has shape [12,current_windows,128].
  std::vector<float> update(const std::uint16_t* current_k,
                            const std::uint16_t* current_v,
                            std::size_t current_windows,
                            const float* current_pooled,
                            bool terminal = false);
  std::vector<float> update(const FlashHistoryUpdate& update);

  // A view must be destroyed before the next update or retirement.  This
  // prevents a GPU upload from observing a file while its slots are rotated.
  FlashHistoryView view() const;

  // Re-sync an already committed pair.  This is useful before a phase switch
  // or asynchronous prefetch, and rejects live views to keep lifetime explicit.
  void flush();
  void retire();

  FlashHistoryMetadata metadata() const noexcept;
  const std::filesystem::path& directory() const noexcept { return directory_; }
  const std::filesystem::path& k_path() const noexcept { return k_path_; }
  const std::filesystem::path& v_path() const noexcept { return v_path_; }

 private:
  struct Mapping;

  void open_directory(const std::filesystem::path& scratch_parent);
  std::shared_ptr<Mapping> create_mapping(const char* name, std::size_t bytes);

  void validate_files() const;
  void validate_directory() const;
  void synchronize();
  bool has_live_view() const noexcept;
  void ensure_open() const;
  void remove_owned_entries_noexcept() noexcept;
  void cleanup_noexcept() noexcept;

  FlashHistoryPolicy policy_;
  std::size_t spatial_windows_;
  std::size_t capacity_windows_;
  std::size_t valid_windows_ = 0U;
  std::size_t window_bytes_ = 0U;
  std::size_t mapping_bytes_ = 0U;
  std::size_t pooled_capacity_elements_ = 0U;
  std::size_t pooled_head_stride_elements_ = 0U;
  std::uint64_t disk_reserve_bytes_;
  // Ownership is captured from the opened parent/worker at construction.  A
  // root worker may give only the private leaf back to the parent owner so a
  // normal user can remove an orphaned leaf; the mapped K/V files remain
  // worker-owned and private.
  std::uintmax_t parent_owner_uid_ = 0U;
  std::uintmax_t parent_owner_gid_ = 0U;
  std::uintmax_t leaf_owner_uid_ = 0U;
  std::uintmax_t leaf_owner_gid_ = 0U;
  std::uintmax_t file_owner_uid_ = 0U;
  std::uintmax_t file_owner_gid_ = 0U;
  int parent_fd_ = -1;
  int leaf_fd_ = -1;
  std::string leaf_name_;
  std::uintmax_t leaf_device_ = 0U;
  std::uintmax_t leaf_inode_ = 0U;
  std::filesystem::path directory_;
  std::filesystem::path k_path_;
  std::filesystem::path v_path_;
  std::shared_ptr<Mapping> k_mapping_;
  std::shared_ptr<Mapping> v_mapping_;
  std::shared_ptr<std::vector<float>> pooled_;
  bool released_ = false;
};

}  // namespace otc::ai
