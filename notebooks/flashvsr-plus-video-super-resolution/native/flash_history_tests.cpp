#include "flash_history.hpp"

#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

using otc::ai::FlashHistoryOwner;
using otc::ai::FlashHistoryPolicy;
using otc::ai::FlashHistoryUpdate;
using otc::ai::FlashMode;

constexpr std::size_t kValuesPerWindow = 128U * 1536U;
constexpr std::size_t kPooledHeads = 12U;
constexpr std::size_t kPooledChannels = 128U;

std::filesystem::path scratch(const std::filesystem::path& root) {
  if (!std::filesystem::is_directory(root))
    throw std::invalid_argument("test fixture root is not a directory");
  std::string pattern = (root / "flash-history-test-XXXXXX").string();
  std::vector<char> bytes(pattern.begin(), pattern.end());
  bytes.push_back('\0');
  if (::mkdtemp(bytes.data()) == nullptr)
    throw std::runtime_error("mkdtemp test fixture failed");
  return std::filesystem::path(bytes.data());
}

std::vector<std::uint16_t> current(std::size_t windows, std::uint16_t seed) {
  std::vector<std::uint16_t> result(windows * kValuesPerWindow);
  for (std::size_t index = 0U; index < result.size(); ++index)
    result[index] = static_cast<std::uint16_t>(seed + index % 31U);
  return result;
}

std::vector<float> pooled(std::size_t windows, float seed) {
  std::vector<float> result(kPooledHeads * windows * kPooledChannels);
  for (std::size_t index = 0U; index < result.size(); ++index)
    result[index] = seed + static_cast<float>(index % 17U) * 0.001F;
  return result;
}

template <typename Action>
void rejects(Action&& action) {
  bool failed = false;
  try {
    action();
  } catch (const std::exception&) {
    failed = true;
  }
  assert(failed);
}

void remove_fixture(const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::remove_all(path, error);
  assert(!error);
}

void assert_owner(const std::filesystem::path& path, std::uintmax_t uid,
                  std::uintmax_t gid) {
  struct stat info {};
  assert(::lstat(path.c_str(), &info) == 0);
  assert(static_cast<std::uintmax_t>(info.st_uid) == uid);
  assert(static_cast<std::uintmax_t>(info.st_gid) == gid);
}

void write_sentinel(const std::filesystem::path& path, const std::string& value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  assert(output.good());
  output << value;
  output.close();
  assert(output.good());
}

void assert_allocated(const std::filesystem::path& path) {
  struct stat info {};
  assert(::stat(path.c_str(), &info) == 0);
  assert(info.st_size > 0 && info.st_blocks >= 0);
  const auto bytes = static_cast<std::uintmax_t>(info.st_size);
  const auto blocks = static_cast<std::uintmax_t>(info.st_blocks);
  const auto required_blocks = bytes / 512U + (bytes % 512U != 0U ? 1U : 0U);
  assert(blocks >= required_blocks);
}

std::string read_sentinel(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  assert(input.good());
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void test_policies(const std::filesystem::path& root) {
  for (std::size_t layer = 0U; layer < 30U; ++layer) {
    const auto turbo = otc::ai::flashHistoryPolicy(FlashMode::turbo, layer);
    if (layer % 6U == 0U)
      assert(turbo.retainedGroups == 3U && turbo.localRange == 11U);
    else
      assert(turbo.retainedGroups == 1U && turbo.localRange == 3U);
    const auto base = otc::ai::flashHistoryPolicy(FlashMode::base, layer);
    assert(base.retainedGroups == 3U && base.localRange == 11U);
  }
  rejects([] { otc::ai::flashHistoryPolicy(FlashMode::turbo, 30U); });
  const auto parent = scratch(root);
  rejects([&] {
    FlashHistoryOwner owner(parent, otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                            otc::ai::kFlashHistoryMaxSpatialWindows + 1U, 0U);
  });
  remove_fixture(parent);
  const auto insecure_parent = scratch(root);
  assert(::chmod(insecure_parent.c_str(), 0770) == 0);
  rejects([&] {
    FlashHistoryOwner owner(insecure_parent,
                            otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                            1U, 0U);
  });
  remove_fixture(insecure_parent);
}

void test_root_worker_leaf_owner_cleanup(const std::filesystem::path& root) {
  struct stat fixture_info {};
  assert(::stat(root.c_str(), &fixture_info) == 0);
  if (::geteuid() != 0 || fixture_info.st_uid == 0) {
    std::cout << "root-worker leaf ownership test skipped (requires a non-root "
                 "fixture owner)\n";
    return;
  }

  const auto parent = scratch(root);
  assert(::chown(parent.c_str(), fixture_info.st_uid, fixture_info.st_gid) == 0);
  // Inherited setgid must not change the root worker's K/V group.
  assert(::chmod(parent.c_str(), 02700) == 0);
  const auto policy = otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U);

  // Construct in a child and exit without running the destructor: this leaves
  // the exact private leaf/files as a hard-crash simulation.
  const pid_t worker = ::fork();
  assert(worker >= 0);
  if (worker == 0) {
    try {
      FlashHistoryOwner owner(parent, policy, 1U, 0U);
      if (!std::filesystem::is_directory(owner.directory())) ::_exit(3);
      ::_exit(0);
    } catch (...) {
      ::_exit(4);
    }
  }
  int status = 0;
  assert(::waitpid(worker, &status, 0) == worker);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);

  std::filesystem::path leaf;
  for (const auto& entry : std::filesystem::directory_iterator(parent)) {
    if (entry.is_directory()) {
      assert(leaf.empty());
      leaf = entry.path();
    }
  }
  assert(!leaf.empty());
  const auto target_uid = static_cast<std::uintmax_t>(fixture_info.st_uid);
  const auto target_gid = static_cast<std::uintmax_t>(fixture_info.st_gid);
  assert_owner(leaf, target_uid, target_gid);
  assert_owner(leaf / "k.bf16.raw", static_cast<std::uintmax_t>(::geteuid()),
               static_cast<std::uintmax_t>(::getegid()));
  assert_owner(leaf / "v.bf16.raw", static_cast<std::uintmax_t>(::geteuid()),
               static_cast<std::uintmax_t>(::getegid()));

  // The captured parent owner can remove an orphaned user-owned leaf even
  // though the mapped K/V files remain private to the root worker.
  const pid_t cleaner = ::fork();
  assert(cleaner >= 0);
  if (cleaner == 0) {
    if (::setgid(static_cast<gid_t>(fixture_info.st_gid)) != 0
        || ::setuid(static_cast<uid_t>(fixture_info.st_uid)) != 0)
      ::_exit(5);
    std::error_code error;
    const auto removed = std::filesystem::remove_all(parent, error);
    if (error || removed == 0U || std::filesystem::exists(parent)) ::_exit(6);
    ::_exit(0);
  }
  status = 0;
  assert(::waitpid(cleaner, &status, 0) == cleaner);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  assert(!std::filesystem::exists(parent));
}

void test_retention_and_pooled_merge(const std::filesystem::path& root) {
  const auto parent = scratch(root);
  {
    const FlashHistoryPolicy policy = otc::ai::flashHistoryPolicy(FlashMode::base, 1U);
    FlashHistoryOwner owner(parent, policy, 2U, 0U);
    auto k0 = current(4U, 0x3F80U);
    auto v0 = current(4U, 0x4000U);
    auto p0 = pooled(4U, 0.25F);
    const auto merged0 = owner.update(k0.data(), v0.data(), 4U, p0.data());
    assert(merged0.size() == 12U * 4U * 128U);
    assert(merged0[0] == p0[0]);
    {
      const auto view = owner.view();
      assert(view.valid() && view.windows() == 4U);
      assert(view.k()[0] == k0[0] && view.v()[0] == v0[0]);
      assert(view.pooled_elements() == 12U * 4U * 128U);
      assert(view.pooled()[0] == p0[0]);
    }
    auto k1 = current(4U, 0x4100U);
    auto v1 = current(4U, 0x4200U);
    auto p1 = pooled(4U, 0.75F);
    const auto merged1 = owner.update(k1.data(), v1.data(), 4U, p1.data());
    assert(merged1.size() == 12U * 8U * 128U);
    assert(merged1[2U * kPooledChannels] == p0[2U * kPooledChannels]);
    assert(merged1[4U * kPooledChannels] == p1[0]);
    {
      const auto view = owner.view();
      const std::size_t stride = view.pooled_head_stride_elements();
      assert(view.windows() == 6U);
      assert(view.k()[0] == k0[2U * kValuesPerWindow]);
      assert(view.k()[4U * kValuesPerWindow] == k1[2U * kValuesPerWindow]);
      assert(view.pooled()[0] == p0[2U * kPooledChannels]);
      assert(view.pooled()[4U * kPooledChannels] == p1[2U * kPooledChannels]);
      assert(view.pooled()[stride + 4U * kPooledChannels]
             == p1[4U * kPooledChannels + 2U * kPooledChannels]);
      // A live view prevents both flush and retirement, and update cannot
      // rotate a mapping still visible to the caller.
      rejects([&] { owner.flush(); });
      rejects([&] { owner.retire(); });
      rejects([&] { owner.update(k1.data(), v1.data(), 4U, p1.data()); });
    }
  }
  remove_fixture(parent);

  const auto terminal_parent = scratch(root);
  {
    FlashHistoryOwner owner(terminal_parent,
                            otc::ai::flashHistoryPolicy(FlashMode::base, 1U), 2U, 0U);
    auto k = current(4U, 0x3F80U);
    auto v = current(4U, 0x4000U);
    auto p = pooled(4U, 0.25F);
    owner.update(k.data(), v.data(), 4U, p.data());
    auto terminal = pooled(4U, 0.75F);
    const auto merged = owner.update(
        FlashHistoryUpdate{nullptr, nullptr, 4U, terminal.data(), {}, true});
    assert(merged.size() == 12U * 8U * 128U);
    assert(merged[0] == p[0]);
    assert(merged[4U * kPooledChannels] == terminal[0]);
    assert(owner.metadata().released);
    rejects([&] { (void)owner.view(); });
  }
  remove_fixture(terminal_parent);
}

void test_callback_failure_is_permanent(const std::filesystem::path& root) {
  const auto parent = scratch(root);
  {
    FlashHistoryOwner owner(parent, otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                            2U, 0U);
    auto k = current(4U, 0x3F80U);
    auto v = current(4U, 0x4000U);
    auto p = pooled(4U, 0.5F);
    owner.update(k.data(), v.data(), 4U, p.data());
    rejects([&] {
      owner.update(FlashHistoryUpdate{nullptr, nullptr, 4U, p.data(),
                                      [](std::uint16_t*, std::uint16_t*, std::size_t,
                                         std::size_t) {
                                        throw std::runtime_error("read failed");
                                      }});
    });
    assert(owner.metadata().released);
    rejects([&] { (void)owner.view(); });
  }
  const auto poison_parent = scratch(root);
  {
    FlashHistoryOwner owner(poison_parent,
                            otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U), 2U, 0U);
    auto k = current(4U, 0x3F80U);
    auto v = current(4U, 0x4000U);
    auto p = pooled(4U, 0.5F);
    rejects([&] {
      owner.update(FlashHistoryUpdate{nullptr, nullptr, 4U, p.data(),
                                      [](std::uint16_t*, std::uint16_t*, std::size_t,
                                         std::size_t) {
                                        // The owner poisons the destination before
                                        // this callback, so a no-op cannot pass.
                                      }});
    });
    assert(owner.metadata().released);
  }
  remove_fixture(parent);
  remove_fixture(poison_parent);
}

void test_finite_stale_and_alias_guards(const std::filesystem::path& root) {
  const auto parent = scratch(root);
  {
    FlashHistoryOwner owner(parent, otc::ai::flashHistoryPolicy(FlashMode::base, 1U),
                            1U, 0U);
    auto k = current(2U, 0x3F80U);
    auto v = current(2U, 0x4000U);
    auto p = pooled(2U, 0.5F);
    rejects([&] { owner.update(k.data(), v.data(), 2U, nullptr); });
    p[0] = std::numeric_limits<float>::infinity();
    rejects([&] { owner.update(k.data(), v.data(), 2U, p.data()); });
    p[0] = 0.5F;
    k[7] = 0x7F80U;
    rejects([&] { owner.update(k.data(), v.data(), 2U, p.data()); });
    k[7] = 0x3F80U;
    owner.update(k.data(), v.data(), 2U, p.data());
    const std::uint16_t* alias = nullptr;
    {
      const auto view = owner.view();
      alias = view.k();
    }
    // The view is destroyed before the call, but the owner still owns the
    // mapping. This exercises the explicit source-to-owned alias guard.
    rejects([&] { owner.update(alias, alias, 2U, p.data()); });
    std::filesystem::resize_file(owner.k_path(), 1U);
    rejects([&] { owner.flush(); });
  }
  remove_fixture(parent);
}

std::uint64_t rss_kib() {
  std::ifstream input("/proc/self/status");
  std::string key;
  std::uint64_t value = 0U;
  while (input >> key) {
    if (key != "VmRSS:") {
      std::string ignored;
      std::getline(input, ignored);
      continue;
    }
    input >> value;
    return value;
  }
  return 0U;
}

void test_one_active_owner_rss_and_reserve_overflow(const std::filesystem::path& root) {
  const auto parent = scratch(root);
  {
    FlashHistoryOwner owner(parent, otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                            otc::ai::kFlashHistoryMaxSpatialWindows, 0U);
    assert(owner.metadata().capacityWindows == otc::ai::kFlashHistoryMaxSpatialWindows);
    assert_allocated(owner.k_path());
    assert_allocated(owner.v_path());
    const std::uint64_t rss = rss_kib();
    assert(rss == 0U || rss < 512U * 1024U);
  }
  rejects([&] {
    FlashHistoryOwner owner(parent, otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                            1U, std::numeric_limits<std::uint64_t>::max());
  });
  remove_fixture(parent);
}

bool directory_empty(const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::directory_iterator entries(path, error);
  return !error && entries == std::filesystem::directory_iterator();
}

void test_allocation_failure_is_atomic(const std::filesystem::path& root) {
  // The parent is deliberately created before fork: only the child changes
  // its file-size limit and signal disposition, while the parent can inspect
  // the exact private root after the constructor's failure cleanup.
  const auto parent = scratch(root);
  const pid_t child = ::fork();
  assert(child >= 0);
  if (child == 0) {
    if (::signal(SIGXFSZ, SIG_IGN) == SIG_ERR)
      ::_exit(2);
    const struct rlimit limit{0U, 0U};
    if (::setrlimit(RLIMIT_FSIZE, &limit) != 0)
      ::_exit(3);

    int result = 0;
    try {
      {
        FlashHistoryOwner owner(parent,
                                otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                                1U, 0U);
        result = 4;  // posix_fallocate unexpectedly succeeded.
      }
    } catch (const std::system_error& error) {
      result = error.code().value() == EFBIG ? 0 : 5;
    } catch (...) {
      result = 6;
    }
    if (result != 0)
      ::_exit(result);
    if (!directory_empty(parent))
      ::_exit(7);
    ::_exit(0);
  }

  int status = 0;
  pid_t waited;
  do {
    waited = ::waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  assert(waited == child);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  assert(directory_empty(parent));
  remove_fixture(parent);
}

void test_parent_rename_keeps_cleanup_on_original_identity(
    const std::filesystem::path& root) {
  const auto parent = scratch(root);
  const auto moved = parent.string() + "-moved";
  const auto moved_path = std::filesystem::path(moved);
  const auto old_sentinel = parent / "outside-old.txt";
  write_sentinel(old_sentinel, "old");
  {
    FlashHistoryOwner owner(parent,
                            otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                            1U, 0U);
    std::filesystem::rename(parent, moved_path);
    std::filesystem::create_directory(parent);
    assert(::chmod(parent.c_str(), 0700) == 0);
    const auto replacement_sentinel = parent / "outside-new.txt";
    write_sentinel(replacement_sentinel, "new");
    // The stable parent descriptor still names the renamed original root;
    // the replacement at the old pathname must be irrelevant.
    owner.flush();
    owner.retire();
    assert(read_sentinel(moved_path / "outside-old.txt") == "old");
    assert(read_sentinel(replacement_sentinel) == "new");
  }
  remove_fixture(parent);
  remove_fixture(moved_path);
}

void test_leaf_rename_does_not_remove_replacement(
    const std::filesystem::path& root) {
  const auto parent = scratch(root);
  const auto replacement_sentinel = parent / "leaf-replacement-sentinel.txt";
  {
    FlashHistoryOwner owner(parent,
                            otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                            1U, 0U);
    const auto original_leaf = owner.directory();
    const auto moved_leaf = parent / "leaf-moved";
    std::filesystem::rename(original_leaf, moved_leaf);
    std::filesystem::create_directory(original_leaf);
    assert(::chmod(original_leaf.c_str(), 0700) == 0);
    write_sentinel(replacement_sentinel, "replacement");
    rejects([&] { owner.flush(); });
    assert(read_sentinel(replacement_sentinel) == "replacement");
  }
  assert(read_sentinel(replacement_sentinel) == "replacement");
  remove_fixture(parent);
}

void test_file_replacement_does_not_remove_target_or_sentinel(
    const std::filesystem::path& root) {
  const auto parent = scratch(root);
  const auto outside = parent / "outside-file.txt";
  write_sentinel(outside, "protected");
  {
    FlashHistoryOwner owner(parent,
                            otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                            1U, 0U);
    const auto saved = owner.k_path().string() + ".saved";
    std::filesystem::rename(owner.k_path(), saved);
    std::filesystem::create_symlink(outside, owner.k_path());
    rejects([&] { owner.flush(); });
    assert(std::filesystem::is_symlink(owner.k_path()));
    assert(read_sentinel(outside) == "protected");
  }
  // The owner must not unlink the replacement symlink or follow it to the
  // outside sentinel.  The exact leaf is discovered before fixture cleanup.
  bool found_sentinel = false;
  for (const auto& entry : std::filesystem::directory_iterator(parent)) {
    if (!entry.is_directory()) continue;
    const auto link = entry.path() / "k.bf16.raw";
    if (std::filesystem::is_symlink(link)) {
      found_sentinel = true;
      assert(read_sentinel(outside) == "protected");
    }
  }
  assert(found_sentinel);
  remove_fixture(parent);
}

void test_view_keeps_mapping_alive_after_owner_destruction(
    const std::filesystem::path& root) {
  const auto parent = scratch(root);
  otc::ai::FlashHistoryView retained;
  std::uint16_t expected_k = 0U;
  std::uint16_t expected_v = 0U;
  {
    FlashHistoryOwner owner(parent,
                            otc::ai::flashHistoryPolicy(FlashMode::turbo, 1U),
                            1U, 0U);
    auto k = current(2U, 0x3F80U);
    auto v = current(2U, 0x4000U);
    auto p = pooled(2U, 0.5F);
    expected_k = k[kValuesPerWindow];
    expected_v = v[kValuesPerWindow];
    owner.update(k.data(), v.data(), 2U, p.data());
    retained = owner.view();
    assert(retained.valid() && retained.k()[0] == k[kValuesPerWindow]
           && retained.v()[0] == v[kValuesPerWindow]);
  }
  assert(retained.valid() && retained.windows() == 1U
         && retained.k()[0] == expected_k && retained.v()[0] == expected_v);
  retained = {};
  remove_fixture(parent);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: flash_history_tests FIXTURE_ROOT\n";
    return 2;
  }
  const std::filesystem::path root = argv[1];
  test_policies(root);
  test_root_worker_leaf_owner_cleanup(root);
  test_retention_and_pooled_merge(root);
  test_callback_failure_is_permanent(root);
  test_finite_stale_and_alias_guards(root);
  test_one_active_owner_rss_and_reserve_overflow(root);
  test_allocation_failure_is_atomic(root);
  test_parent_rename_keeps_cleanup_on_original_identity(root);
  test_leaf_rename_does_not_remove_replacement(root);
  test_file_replacement_does_not_remove_target_or_sentinel(root);
  test_view_keeps_mapping_alive_after_owner_destruction(root);
  std::cout << "flash_history CPU contract passed\n";
}
