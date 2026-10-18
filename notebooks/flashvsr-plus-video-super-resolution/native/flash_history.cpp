#include "flash_history.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

namespace otc::ai {
namespace {

constexpr std::size_t kLayerCount = 30U;
constexpr std::uint16_t kBf16ExponentMask = 0x7F80U;

[[noreturn]] void throw_errno(const char* action, const std::filesystem::path& path) {
  throw std::system_error(errno, std::generic_category(),
                          std::string(action) + ": " + path.string());
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* name) {
  if (right != 0U && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::overflow_error(std::string(name) + " overflows size_t");
  return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* name) {
  if (left > std::numeric_limits<std::size_t>::max() - right)
    throw std::overflow_error(std::string(name) + " overflows size_t");
  return left + right;
}

void validate_bf16(const std::uint16_t* words, std::size_t count,
                   const char* name) {
  if (count != 0U && words == nullptr)
    throw std::invalid_argument(std::string(name) + " is null");
  for (std::size_t index = 0U; index < count; ++index) {
    if ((words[index] & kBf16ExponentMask) == kBf16ExponentMask)
      throw std::invalid_argument(std::string(name) + " contains NaN or infinity");
  }
}

void validate_f32(const float* values, std::size_t count, const char* name) {
  if (count != 0U && values == nullptr)
    throw std::invalid_argument(std::string(name) + " is null");
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(values[index]))
      throw std::invalid_argument(std::string(name) + " contains NaN or infinity");
  }
}

bool ranges_overlap_bytes(const void* left, std::size_t left_bytes,
                          const void* right, std::size_t right_bytes) {
  if (left_bytes == 0U || right_bytes == 0U || left == nullptr || right == nullptr)
    return false;
  const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
  const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
  const auto left_end = left_begin + left_bytes;
  const auto right_end = right_begin + right_bytes;
  if (left_end < left_begin || right_end < right_begin)
    throw std::overflow_error("history pointer range overflows uintptr_t");
  return left_begin < right_end && right_begin < left_end;
}

bool ranges_overlap(const std::uint16_t* left, std::size_t left_count,
                    const std::uint16_t* right, std::size_t right_count) {
  return ranges_overlap_bytes(left, checked_mul(left_count, sizeof(std::uint16_t),
                                                "left range"),
                              right, checked_mul(right_count, sizeof(std::uint16_t),
                                                 "right range"));
}

bool same_identity(const struct stat& info, std::uintmax_t device,
                  std::uintmax_t inode) {
  return static_cast<std::uintmax_t>(info.st_dev) == device
      && static_cast<std::uintmax_t>(info.st_ino) == inode;
}

void require_directory(const struct stat& info, const char* name,
                       bool require_owner, std::uintmax_t expected_uid = 0U,
                       std::uintmax_t expected_gid = 0U) {
  if (!S_ISDIR(info.st_mode)
      || (require_owner
          && (static_cast<std::uintmax_t>(info.st_uid) != expected_uid
              || static_cast<std::uintmax_t>(info.st_gid) != expected_gid))
      || (info.st_mode & 0022) != 0)
    throw std::invalid_argument(std::string(name)
                                + " is not a private directory");
}

std::uint64_t available_bytes(int fd) {
  struct statvfs space {};
  if (::fstatvfs(fd, &space) != 0)
    throw_errno("fstatvfs history scratch", "<parent fd>");
  const auto max = std::numeric_limits<std::uint64_t>::max();
  if (space.f_frsize == 0U || space.f_bavail > max / space.f_frsize)
    throw std::overflow_error("history free-space calculation overflows");
  return static_cast<std::uint64_t>(space.f_bavail) * space.f_frsize;
}

}  // namespace

struct FlashHistoryOwner::Mapping final {
  int fd = -1;
  void* data = MAP_FAILED;
  std::size_t bytes = 0U;
  std::string name;
  std::uintmax_t device = 0U;
  std::uintmax_t inode = 0U;

  ~Mapping() noexcept {
    if (data != MAP_FAILED) ::munmap(data, bytes);
    if (fd >= 0) ::close(fd);
  }
};

FlashHistoryPolicy flashHistoryPolicy(FlashMode mode, std::size_t layer) {
  if (layer >= kLayerCount)
    throw std::out_of_range("Flash history layer must be in [0,29]");
  switch (mode) {
    case FlashMode::turbo:
      if (layer % 6U == 0U) return {mode, layer, 3U, 11U};
      return {mode, layer, 1U, 3U};
    case FlashMode::base:
      return {mode, layer, 3U, 11U};
  }
  throw std::invalid_argument("unsupported Flash history mode");
}

void FlashHistoryOwner::open_directory(
    const std::filesystem::path& scratch_parent) {
  if (!scratch_parent.is_absolute())
    throw std::invalid_argument("Flash history scratch parent must be absolute");
  parent_fd_ = ::open(scratch_parent.c_str(),
                      O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (parent_fd_ < 0)
    throw_errno("open history scratch parent", scratch_parent);
  struct stat parent_info {};
  if (::fstat(parent_fd_, &parent_info) != 0)
    throw_errno("fstat history scratch parent", scratch_parent);
  parent_owner_uid_ = static_cast<std::uintmax_t>(parent_info.st_uid);
  parent_owner_gid_ = static_cast<std::uintmax_t>(parent_info.st_gid);
  file_owner_uid_ = static_cast<std::uintmax_t>(::geteuid());
  file_owner_gid_ = static_cast<std::uintmax_t>(::getegid());
  leaf_owner_uid_ = file_owner_uid_;
  leaf_owner_gid_ = file_owner_gid_;
  // The parent may belong to the requesting user while the worker is root.
  // Give that user only the private leaf so an orphan can be removed; the
  // mapped K/V files remain owned by the worker and mode 0600.
  if (file_owner_uid_ == 0U
      && (parent_owner_uid_ != file_owner_uid_
          || parent_owner_gid_ != file_owner_gid_)) {
    leaf_owner_uid_ = parent_owner_uid_;
    leaf_owner_gid_ = parent_owner_gid_;
  }
  require_directory(parent_info, "Flash history scratch parent", false);
  std::string pattern = "/proc/self/fd/" + std::to_string(parent_fd_)
                        + "/otc-flash-history-XXXXXX";
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char* result = ::mkdtemp(writable.data());
  if (result == nullptr) throw_errno("mkdtemp", scratch_parent);
  leaf_name_ = std::filesystem::path(result).filename().string();
  if (leaf_name_.empty() || leaf_name_ == "." || leaf_name_ == "..")
    throw std::runtime_error("invalid Flash history private leaf");
  directory_ = scratch_parent / leaf_name_;
  k_path_ = directory_ / "k.bf16.raw";
  v_path_ = directory_ / "v.bf16.raw";
  struct stat leaf_info {};
  if (::fstatat(parent_fd_, leaf_name_.c_str(), &leaf_info,
                AT_SYMLINK_NOFOLLOW) != 0)
    throw_errno("fstat Flash history private leaf", directory_);
  // Validate the mkdtemp object before opening it, then apply the captured
  // leaf owner on the stable descriptor and validate ownership again.
  require_directory(leaf_info, "Flash history private leaf", false);
  if (static_cast<std::uintmax_t>(leaf_info.st_uid) != file_owner_uid_)
    throw std::runtime_error("Flash history private leaf owner changed");
  leaf_device_ = static_cast<std::uintmax_t>(leaf_info.st_dev);
  leaf_inode_ = static_cast<std::uintmax_t>(leaf_info.st_ino);
  leaf_fd_ = ::openat(parent_fd_, leaf_name_.c_str(),
                      O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (leaf_fd_ < 0)
    throw_errno("open Flash history private leaf", directory_);
  struct stat opened_leaf {};
  if (::fstat(leaf_fd_, &opened_leaf) != 0
      || !S_ISDIR(opened_leaf.st_mode)
      || static_cast<std::uintmax_t>(opened_leaf.st_uid) != file_owner_uid_
      || !same_identity(opened_leaf, leaf_device_, leaf_inode_))
    throw std::runtime_error("Flash history private leaf identity changed");
  // Authenticate the opened object before mutating it. Strip an inherited
  // setgid bit so new K/V files keep the worker's group, not the parent group.
  if (::fchmod(leaf_fd_, S_IRWXU) != 0)
    throw_errno("fchmod Flash history private leaf", directory_);
  // A non-root mkdtemp may inherit a different parent group. Keeping that
  // group on the directory is harmless; its setgid bit was removed above.
  if (file_owner_uid_ != 0U)
    leaf_owner_gid_ = static_cast<std::uintmax_t>(opened_leaf.st_gid);
  if (file_owner_uid_ == 0U &&
      (leaf_owner_uid_ != file_owner_uid_ || leaf_owner_gid_ != file_owner_gid_)) {
    int result = 0;
    do {
      result = ::fchown(leaf_fd_, static_cast<uid_t>(leaf_owner_uid_),
                        static_cast<gid_t>(leaf_owner_gid_));
    } while (result != 0 && errno == EINTR);
    if (result != 0)
      throw_errno("fchown Flash history private leaf", directory_);
  }
  if (::fstat(leaf_fd_, &opened_leaf) != 0
      || !S_ISDIR(opened_leaf.st_mode)
      || !same_identity(opened_leaf, leaf_device_, leaf_inode_))
    throw std::runtime_error("Flash history private leaf identity changed");
  require_directory(opened_leaf, "Flash history private leaf", true,
                    leaf_owner_uid_, leaf_owner_gid_);
  struct stat visible_leaf {};
  if (::fstatat(parent_fd_, leaf_name_.c_str(), &visible_leaf,
                AT_SYMLINK_NOFOLLOW) != 0
      || !same_identity(visible_leaf, leaf_device_, leaf_inode_))
    throw std::runtime_error("Flash history private leaf identity changed");
  require_directory(visible_leaf, "Flash history private leaf", true,
                    leaf_owner_uid_, leaf_owner_gid_);
}

std::shared_ptr<FlashHistoryOwner::Mapping> FlashHistoryOwner::create_mapping(
    const char* name, std::size_t bytes) {
  if (leaf_fd_ < 0 || name == nullptr || *name == '\0')
    throw std::logic_error("Flash history private leaf is not open");
  const int fd = ::openat(leaf_fd_, name,
                          O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW,
                          S_IRUSR | S_IWUSR);
  if (fd < 0) throw_errno("open history file", directory_ / name);
  struct stat owned_info {};
  bool have_identity = ::fstat(fd, &owned_info) == 0;
  const auto owned_device = have_identity
      ? static_cast<std::uintmax_t>(owned_info.st_dev) : 0U;
  const auto owned_inode = have_identity
      ? static_cast<std::uintmax_t>(owned_info.st_ino) : 0U;
  auto fail = [&]() noexcept {
    if (have_identity) {
      struct stat at {};
      if (::fstatat(leaf_fd_, name, &at, AT_SYMLINK_NOFOLLOW) == 0
          && S_ISREG(at.st_mode) && same_identity(at, owned_device, owned_inode))
        ::unlinkat(leaf_fd_, name, 0);
    }
    ::close(fd);
  };
  int allocation_error = 0;
  do {
    allocation_error = ::posix_fallocate(fd, 0, static_cast<off_t>(bytes));
  } while (allocation_error == EINTR);
  if (allocation_error != 0) {
    fail();
    errno = allocation_error;
    throw_errno("posix_fallocate history file", directory_ / name);
  }
  if (::fstat(fd, &owned_info) != 0 || !S_ISREG(owned_info.st_mode)
      || owned_info.st_size < 0
      || static_cast<std::uintmax_t>(owned_info.st_size) != bytes
      || static_cast<std::uintmax_t>(owned_info.st_uid) != file_owner_uid_
      || static_cast<std::uintmax_t>(owned_info.st_gid) != file_owner_gid_
      || (owned_info.st_mode & 0077) != 0
      || owned_info.st_nlink == 0) {
    fail();
    throw std::runtime_error("history file extent or type changed during creation");
  }
  have_identity = true;
  const auto device = static_cast<std::uintmax_t>(owned_info.st_dev);
  const auto inode = static_cast<std::uintmax_t>(owned_info.st_ino);
  void* data = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (data == MAP_FAILED) {
    const int error = errno;
    fail();
    errno = error;
    throw_errno("mmap history file", directory_ / name);
  }
  auto mapping = std::make_shared<Mapping>();
  mapping->fd = fd;
  mapping->data = data;
  mapping->bytes = bytes;
  mapping->name = name;
  mapping->device = device;
  mapping->inode = inode;
  return mapping;
}

FlashHistoryOwner::FlashHistoryOwner(const std::filesystem::path& scratch_parent,
                                     FlashHistoryPolicy policy,
                                     std::size_t spatial_windows,
                                     std::uint64_t disk_reserve_bytes)
    : policy_(policy), spatial_windows_(spatial_windows),
      capacity_windows_(checked_mul(policy.retainedGroups, spatial_windows,
                                    "history capacity")),
      window_bytes_(checked_mul(
          checked_mul(kFlashHistoryWindowRows, kFlashHistoryChannels,
                      "history window elements"), kFlashHistoryElementBytes,
          "history window bytes")),
      mapping_bytes_(checked_mul(capacity_windows_, window_bytes_,
                                "history mapping bytes")),
      disk_reserve_bytes_(disk_reserve_bytes) {
  const auto expected = flashHistoryPolicy(policy.mode, policy.layer);
  if (policy.retainedGroups != expected.retainedGroups
      || policy.localRange != expected.localRange)
    throw std::invalid_argument("Flash history policy does not match its mode/layer");
  if (policy.layer >= kLayerCount || policy.retainedGroups == 0U
      || policy.retainedGroups > 3U || spatial_windows == 0U
      || spatial_windows > kFlashHistoryMaxSpatialWindows)
    throw std::invalid_argument("invalid Flash history policy or spatial window count");
  if (policy.localRange == 0U)
    throw std::invalid_argument("Flash history local range must be positive");
  if (mapping_bytes_ > static_cast<std::size_t>(std::numeric_limits<off_t>::max()))
    throw std::overflow_error("history file exceeds off_t");
  pooled_head_stride_elements_ = checked_mul(
      capacity_windows_, kFlashHistoryWindowRows, "pooled history head stride");
  pooled_capacity_elements_ = checked_mul(
      kFlashHistoryPooledHeads, pooled_head_stride_elements_,
      "pooled history capacity");
  try {
    open_directory(scratch_parent);
    const auto pair_bytes = checked_mul(mapping_bytes_, 2U, "history K/V bytes");
    const auto available = available_bytes(parent_fd_);
    const auto pair_u = static_cast<std::uint64_t>(pair_bytes);
    if (pair_u > available || disk_reserve_bytes_ > available - pair_u)
      throw std::runtime_error("insufficient scratch space for Flash history");
    pooled_ = std::make_shared<std::vector<float>>(pooled_capacity_elements_, 0.0F);
    k_mapping_ = create_mapping("k.bf16.raw", mapping_bytes_);
    v_mapping_ = create_mapping("v.bf16.raw", mapping_bytes_);
  } catch (...) {
    cleanup_noexcept();
    throw;
  }
}

FlashHistoryOwner::~FlashHistoryOwner() noexcept { cleanup_noexcept(); }

void FlashHistoryOwner::ensure_open() const {
  if (released_ || parent_fd_ < 0 || leaf_fd_ < 0 || !k_mapping_
      || !v_mapping_ || !pooled_)
    throw std::runtime_error("Flash history owner is released");
}

void FlashHistoryOwner::validate_directory() const {
  ensure_open();
  struct stat parent_info {};
  if (::fstat(parent_fd_, &parent_info) != 0 || !S_ISDIR(parent_info.st_mode)
      || static_cast<std::uintmax_t>(parent_info.st_uid) != parent_owner_uid_
      || static_cast<std::uintmax_t>(parent_info.st_gid) != parent_owner_gid_
      || (parent_info.st_mode & 0022) != 0)
    throw std::runtime_error("Flash history scratch parent changed");
  struct stat leaf_at {};
  struct stat leaf_opened {};
  if (::fstatat(parent_fd_, leaf_name_.c_str(), &leaf_at,
                AT_SYMLINK_NOFOLLOW) != 0
      || !S_ISDIR(leaf_at.st_mode)
      || static_cast<std::uintmax_t>(leaf_at.st_uid) != leaf_owner_uid_
      || static_cast<std::uintmax_t>(leaf_at.st_gid) != leaf_owner_gid_
      || (leaf_at.st_mode & 0022) != 0
      || !same_identity(leaf_at, leaf_device_, leaf_inode_)
      || ::fstat(leaf_fd_, &leaf_opened) != 0
      || !S_ISDIR(leaf_opened.st_mode)
      || static_cast<std::uintmax_t>(leaf_opened.st_uid) != leaf_owner_uid_
      || static_cast<std::uintmax_t>(leaf_opened.st_gid) != leaf_owner_gid_
      || (leaf_opened.st_mode & 0022) != 0
      || !same_identity(leaf_opened, leaf_device_, leaf_inode_))
    throw std::runtime_error("Flash history private leaf identity changed");
}

void FlashHistoryOwner::validate_files() const {
  validate_directory();
  for (const auto& mapping : {k_mapping_, v_mapping_}) {
    struct stat at {};
    struct stat opened {};
    if (::fstatat(leaf_fd_, mapping->name.c_str(), &at,
                  AT_SYMLINK_NOFOLLOW) != 0 || !S_ISREG(at.st_mode)
        || static_cast<std::uintmax_t>(at.st_uid) != file_owner_uid_
        || static_cast<std::uintmax_t>(at.st_gid) != file_owner_gid_
        || (at.st_mode & 0077) != 0
        || at.st_nlink == 0 || !same_identity(at, mapping->device, mapping->inode)
        || static_cast<std::uintmax_t>(at.st_size) != mapping_bytes_
        || ::fstat(mapping->fd, &opened) != 0 || !S_ISREG(opened.st_mode)
        || static_cast<std::uintmax_t>(opened.st_uid) != file_owner_uid_
        || static_cast<std::uintmax_t>(opened.st_gid) != file_owner_gid_
        || (opened.st_mode & 0077) != 0
        || opened.st_nlink == 0 || !same_identity(opened, mapping->device,
                                                   mapping->inode)
        || static_cast<std::uintmax_t>(opened.st_size) != mapping_bytes_)
      throw std::runtime_error("Flash history file extent or type changed");
  }
}

bool FlashHistoryOwner::has_live_view() const noexcept {
  return k_mapping_.use_count() > 1U || v_mapping_.use_count() > 1U;
}

void FlashHistoryOwner::synchronize() {
  validate_files();
  if (::msync(k_mapping_->data, mapping_bytes_, MS_SYNC) != 0)
    throw_errno("msync K history", k_path_);
  if (::msync(v_mapping_->data, mapping_bytes_, MS_SYNC) != 0)
    throw_errno("msync V history", v_path_);
  if (::fsync(k_mapping_->fd) != 0) throw_errno("fsync K history", k_path_);
  if (::fsync(v_mapping_->fd) != 0) throw_errno("fsync V history", v_path_);
#if defined(__linux__)
  if (::posix_fadvise(k_mapping_->fd, 0, static_cast<off_t>(mapping_bytes_),
                      POSIX_FADV_DONTNEED) != 0)
    throw std::runtime_error("posix_fadvise K history failed");
  if (::posix_fadvise(v_mapping_->fd, 0, static_cast<off_t>(mapping_bytes_),
                      POSIX_FADV_DONTNEED) != 0)
    throw std::runtime_error("posix_fadvise V history failed");
  if (::madvise(k_mapping_->data, mapping_bytes_, MADV_DONTNEED) != 0)
    throw_errno("madvise K history", k_path_);
  if (::madvise(v_mapping_->data, mapping_bytes_, MADV_DONTNEED) != 0)
    throw_errno("madvise V history", v_path_);
#endif
}

std::vector<float> FlashHistoryOwner::update(const std::uint16_t* current_k,
                                             const std::uint16_t* current_v,
                                             std::size_t current_windows,
                                             const float* current_pooled,
                                             bool terminal) {
  return update(FlashHistoryUpdate{current_k, current_v, current_windows,
                                   current_pooled, {}, terminal});
}

std::vector<float> FlashHistoryOwner::update(const FlashHistoryUpdate& current) {
  ensure_open();
  if (has_live_view())
    throw std::logic_error("release the current Flash history view before update");
  const std::size_t expected_windows = checked_mul(
      kFlashHistoryCurrentWindows, spatial_windows_, "current history windows");
  if (current.current_windows != expected_windows)
    throw std::invalid_argument("Flash history update must contain two current windows");
  if ((current.current_k == nullptr) != (current.current_v == nullptr))
    throw std::invalid_argument("current K/V history must be supplied as a pair");
  if (current.terminal) {
    if (current.current_k != nullptr || current.current_v != nullptr
        || current.read_retained)
      throw std::invalid_argument("terminal history cannot include K/V or a callback");
  } else {
    if (current.read_retained
        && (current.current_k != nullptr || current.current_v != nullptr))
      throw std::invalid_argument("retained-read callback cannot be mixed with current K/V");
    if (!current.read_retained
        && (current.current_k == nullptr || current.current_v == nullptr))
      throw std::invalid_argument(
          "current K/V history or a retained-read callback is required");
  }
  const std::size_t current_values = checked_mul(current.current_windows,
                                                  kFlashHistoryWindowRows *
                                                      kFlashHistoryChannels,
                                                  "current history values");
  if (current.current_k != nullptr) {
    validate_bf16(current.current_k, current_values, "current K history");
    validate_bf16(current.current_v, current_values, "current V history");
  }
  const std::size_t pooled_expected = checked_mul(
      checked_mul(kFlashHistoryPooledHeads, current.current_windows,
                  "pooled history windows"),
      kFlashHistoryWindowRows, "pooled history elements");
  if (current.current_pooled == nullptr)
    throw std::invalid_argument("current pooled history is required");
  validate_f32(current.current_pooled, pooled_expected, "current pooled history");
  const std::size_t merged_windows = checked_add(
      valid_windows_, current.current_windows, "merged pooled windows");
  const std::size_t merged_head_stride = checked_mul(
      merged_windows, kFlashHistoryWindowRows, "merged pooled head stride");
  const std::size_t merged_elements = checked_mul(
      kFlashHistoryPooledHeads, merged_head_stride, "merged pooled elements");
  const std::size_t old_pooled_bytes = checked_mul(
      checked_mul(valid_windows_, kFlashHistoryWindowRows, "old pooled values"),
      sizeof(float), "old pooled bytes");
  const std::size_t current_pooled_bytes = checked_mul(
      checked_mul(current.current_windows, kFlashHistoryWindowRows,
                  "current pooled values"),
      sizeof(float), "current pooled bytes");
  const std::size_t current_head_stride = checked_mul(
      current.current_windows, kFlashHistoryWindowRows,
      "current pooled head stride");
  std::vector<float> merged(merged_elements);
  for (std::size_t head = 0U; head < kFlashHistoryPooledHeads; ++head) {
    const std::size_t merged_head = checked_mul(
        head, merged_head_stride, "merged pooled head");
    const std::size_t current_head = checked_mul(
        head, current_head_stride, "current pooled head");
    if (old_pooled_bytes != 0U)
      std::memcpy(merged.data() + merged_head, pooled_->data()
                      + checked_mul(head, pooled_head_stride_elements_,
                                    "owned pooled head"),
                  old_pooled_bytes);
    std::memcpy(merged.data() + merged_head
                    + checked_mul(valid_windows_, kFlashHistoryWindowRows,
                                  "merged current offset"),
                current.current_pooled + current_head, current_pooled_bytes);
  }
  const std::size_t mapped_values = checked_mul(
      capacity_windows_, kFlashHistoryWindowRows * kFlashHistoryChannels,
      "mapped history values");
  if (current.current_k != nullptr && (ranges_overlap(current.current_k, current_values,
                     static_cast<const std::uint16_t*>(k_mapping_->data),
                     mapped_values)
      || ranges_overlap(current.current_k, current_values,
                        static_cast<const std::uint16_t*>(v_mapping_->data),
                        mapped_values)
      || ranges_overlap(current.current_v, current_values,
                        static_cast<const std::uint16_t*>(v_mapping_->data),
                        mapped_values)
      || ranges_overlap(current.current_v, current_values,
                        static_cast<const std::uint16_t*>(k_mapping_->data),
                        mapped_values)))
    throw std::invalid_argument("current history must not alias the owned mapping");
  if (current.current_k != nullptr && ranges_overlap(current.current_k, current_values,
                                                     current.current_v, current_values))
    throw std::invalid_argument("current K/V history must not alias each other");
  const std::size_t pooled_bytes = checked_mul(
      pooled_expected, sizeof(float), "current pooled history bytes");
  const std::size_t pooled_capacity_bytes = checked_mul(
      pooled_capacity_elements_, sizeof(float), "pooled history capacity bytes");
  if (ranges_overlap_bytes(current.current_pooled, pooled_bytes, pooled_->data(),
                           pooled_capacity_bytes)
      || ranges_overlap_bytes(current.current_pooled, pooled_bytes,
                              k_mapping_->data, mapping_bytes_)
      || ranges_overlap_bytes(current.current_pooled, pooled_bytes,
                              v_mapping_->data, mapping_bytes_))
    throw std::invalid_argument("current pooled history must not alias owned storage");
  if (current.current_k != nullptr
      && (ranges_overlap_bytes(current.current_pooled, pooled_bytes,
                               current.current_k,
                               checked_mul(current_values, sizeof(std::uint16_t),
                                           "current K history bytes"))
          || ranges_overlap_bytes(current.current_pooled, pooled_bytes,
                                  current.current_v,
                                  checked_mul(current_values, sizeof(std::uint16_t),
                                              "current V history bytes"))))
    throw std::invalid_argument("current pooled history must not alias current K/V");
  if (current.terminal) {
    // The complete pooled merge was materialized before release. K/V pages
    // are intentionally not synchronized on a terminal mask-only update.
    cleanup_noexcept();
    return merged;
  }
  validate_files();
  const std::size_t keep_current = std::min(current.current_windows, capacity_windows_);
  const std::size_t keep_old = std::min(valid_windows_, capacity_windows_ - keep_current);
  const std::size_t old_start = valid_windows_ - keep_old;
  const std::size_t source_start = current.current_windows - keep_current;
  const std::size_t keep_values = checked_mul(keep_current,
                                              kFlashHistoryWindowRows *
                                                  kFlashHistoryChannels,
                                              "retained history values");
  const std::size_t old_values = checked_mul(old_start,
                                             kFlashHistoryWindowRows *
                                                 kFlashHistoryChannels,
                                             "old history values");
  const std::size_t source_values = checked_mul(source_start,
                                                kFlashHistoryWindowRows *
                                                    kFlashHistoryChannels,
                                                "source history values");
  const std::size_t pooled_old_offset = checked_mul(
      old_start, kFlashHistoryWindowRows, "old pooled offset");
  const std::size_t pooled_source_offset = checked_mul(
      source_start, kFlashHistoryWindowRows, "source pooled offset");
  const std::size_t pooled_destination_offset = checked_mul(
      keep_old, kFlashHistoryWindowRows, "destination pooled offset");
  const std::size_t pooled_copy_bytes = checked_mul(
      keep_current, checked_mul(kFlashHistoryWindowRows, sizeof(float),
                                "pooled window bytes"),
      "pooled copy bytes");
  try {
    auto* k = static_cast<std::uint16_t*>(k_mapping_->data);
    auto* v = static_cast<std::uint16_t*>(v_mapping_->data);
    auto* pooled = pooled_->data();
    if (keep_old != 0U) {
      std::memmove(k, k + old_values, keep_old *
                   kFlashHistoryWindowRows * kFlashHistoryChannels *
                   kFlashHistoryElementBytes);
      std::memmove(v, v + old_values, keep_old *
                   kFlashHistoryWindowRows * kFlashHistoryChannels *
                   kFlashHistoryElementBytes);
      const std::size_t pooled_move_bytes = checked_mul(
          keep_old, checked_mul(kFlashHistoryWindowRows, sizeof(float),
                                "pooled window bytes"),
          "pooled move bytes");
      for (std::size_t head = 0U; head < kFlashHistoryPooledHeads; ++head) {
        const std::size_t head_offset = checked_mul(
            head, pooled_head_stride_elements_, "pooled head offset");
        std::memmove(pooled + head_offset,
                     pooled + head_offset + pooled_old_offset,
                     pooled_move_bytes);
      }
    }
    const std::size_t destination_values = checked_mul(
        keep_old, kFlashHistoryWindowRows * kFlashHistoryChannels,
        "destination history values");
    if (current.read_retained) {
      std::fill(k + destination_values, k + destination_values + keep_values,
                static_cast<std::uint16_t>(0x7F80U));
      std::fill(v + destination_values, v + destination_values + keep_values,
                static_cast<std::uint16_t>(0x7F80U));
      current.read_retained(k + destination_values, v + destination_values,
                            source_start, keep_current);
      validate_bf16(k + destination_values, keep_values,
                    "retained K history callback");
      validate_bf16(v + destination_values, keep_values,
                    "retained V history callback");
    } else {
      std::memcpy(k + destination_values, current.current_k + source_values,
                  checked_mul(keep_values, kFlashHistoryElementBytes,
                              "current K history bytes"));
      std::memcpy(v + destination_values, current.current_v + source_values,
                  checked_mul(keep_values, kFlashHistoryElementBytes,
                              "current V history bytes"));
    }
    for (std::size_t head = 0U; head < kFlashHistoryPooledHeads; ++head) {
      const std::size_t destination_head = checked_mul(
          head, pooled_head_stride_elements_, "pooled destination head");
      const std::size_t source_head = checked_mul(
          head, checked_mul(current.current_windows, kFlashHistoryWindowRows,
                            "current pooled head stride"),
          "pooled source head");
      std::memcpy(pooled + destination_head + pooled_destination_offset,
                  current.current_pooled + source_head + pooled_source_offset,
                  pooled_copy_bytes);
    }
    valid_windows_ = keep_old + keep_current;
    synchronize();
    return merged;
  } catch (...) {
    cleanup_noexcept();
    throw;
  }
}

FlashHistoryView FlashHistoryOwner::view() const {
  ensure_open();
  validate_files();
  if (valid_windows_ == 0U)
    return {};
  struct ViewKeepAlive final {
    std::shared_ptr<Mapping> k;
    std::shared_ptr<Mapping> v;
    std::shared_ptr<std::vector<float>> pooled;
  };
  auto keep = std::make_shared<ViewKeepAlive>(
      ViewKeepAlive{k_mapping_, v_mapping_, pooled_});
  std::shared_ptr<const void> opaque = std::move(keep);
  return FlashHistoryView(static_cast<const std::uint16_t*>(k_mapping_->data),
                          static_cast<const std::uint16_t*>(v_mapping_->data),
                          valid_windows_, pooled_->data(),
                          checked_mul(checked_mul(kFlashHistoryPooledHeads,
                                                  valid_windows_,
                                                  "pooled view windows"),
                                      kFlashHistoryWindowRows,
                                      "pooled view elements"),
                          pooled_head_stride_elements_, std::move(opaque));
}

void FlashHistoryOwner::flush() {
  ensure_open();
  if (has_live_view())
    throw std::logic_error("release the current Flash history view before flush");
  synchronize();
}

void FlashHistoryOwner::retire() {
  ensure_open();
  if (has_live_view())
    throw std::logic_error("release the current Flash history view before retire");
  synchronize();
  cleanup_noexcept();
}

FlashHistoryMetadata FlashHistoryOwner::metadata() const noexcept {
  return {policy_, spatial_windows_, capacity_windows_, valid_windows_, released_};
}

void FlashHistoryOwner::remove_owned_entries_noexcept() noexcept {
  if (leaf_fd_ >= 0) {
    for (const auto& mapping : {k_mapping_, v_mapping_}) {
      if (!mapping || mapping->name.empty()) continue;
      struct stat at {};
      if (::fstatat(leaf_fd_, mapping->name.c_str(), &at,
                    AT_SYMLINK_NOFOLLOW) == 0
          && S_ISREG(at.st_mode)
          && same_identity(at, mapping->device, mapping->inode))
        ::unlinkat(leaf_fd_, mapping->name.c_str(), 0);
    }
  }
  if (parent_fd_ >= 0 && !leaf_name_.empty() && leaf_inode_ != 0U) {
    struct stat leaf_at {};
    if (::fstatat(parent_fd_, leaf_name_.c_str(), &leaf_at,
                  AT_SYMLINK_NOFOLLOW) == 0
        && S_ISDIR(leaf_at.st_mode)
        && same_identity(leaf_at, leaf_device_, leaf_inode_))
      ::unlinkat(parent_fd_, leaf_name_.c_str(), AT_REMOVEDIR);
  }
}

void FlashHistoryOwner::cleanup_noexcept() noexcept {
  if (released_ && parent_fd_ < 0 && leaf_fd_ < 0) return;
  // Keep the Mapping objects alive through identity-checked unlinkat calls;
  // their descriptors also keep mmaps valid while a view owns them.
  remove_owned_entries_noexcept();
  k_mapping_.reset();
  v_mapping_.reset();
  pooled_.reset();
  if (leaf_fd_ >= 0) ::close(leaf_fd_);
  if (parent_fd_ >= 0) ::close(parent_fd_);
  leaf_fd_ = -1;
  parent_fd_ = -1;
  valid_windows_ = 0U;
  released_ = true;
}

}  // namespace otc::ai
