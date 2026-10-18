#include "seedvr_assets.hpp"
#include "seedvr_component_assets.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <openssl/evp.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <linux/magic.h>
#include <sys/vfs.h>
#endif

namespace otc::ai {
namespace {
constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
constexpr std::uint64_t kXmlLimit = 8ULL * 1024 * 1024;
constexpr std::uint64_t kBinLimit = kGiB;

void validatePin(const SeedVrAssetPin& pin, std::uint64_t limit) {
  if (pin.name.empty() || pin.name == "." || pin.name == ".." ||
      pin.name.find_first_of("/\\:") != std::string::npos ||
      pin.name.find('\0') != std::string::npos || pin.bytes == 0 || pin.bytes > limit ||
      pin.bytes > std::numeric_limits<std::size_t>::max() - 2)
    throw std::invalid_argument("SeedVR2 trusted asset pin is invalid");
}

void validateSpan(const SeedVrWeightSpan& span, std::uint64_t size) {
  if (span.bytes == 0 || span.offset > size || span.bytes > size - span.offset ||
      span.elements > UINT64_MAX / 2 || span.bytes != span.elements * 2)
    throw std::invalid_argument("SeedVR2 learned span is outside its owned BIN");
}

struct File final {
#ifdef _WIN32
  HANDLE handle = INVALID_HANDLE_VALUE;
  ~File() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#else
  int handle = -1;
  int directory = -1;
  ~File() { if (handle >= 0) close(handle); if (directory >= 0) close(directory); }
#endif
};

void readVerified(const std::filesystem::path& directory, const SeedVrAssetPin& pin,
                  void* destination) {
  File file;
#ifdef _WIN32
  file.handle = CreateFileW((directory / pin.name).c_str(), GENERIC_READ, FILE_SHARE_READ,
      nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  BY_HANDLE_FILE_INFORMATION info{};
  if (file.handle == INVALID_HANDLE_VALUE || GetFileType(file.handle) != FILE_TYPE_DISK ||
      !GetFileInformationByHandle(file.handle, &info) ||
      (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) ||
      ((static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow) != pin.bytes)
    throw std::runtime_error("SeedVR2 asset is not an exact regular file");
#else
  file.directory = open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (file.directory >= 0)
    file.handle = openat(file.directory, pin.name.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
  struct stat info{};
  if (file.handle < 0 || fstat(file.handle, &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_size < 0 || static_cast<std::uint64_t>(info.st_size) != pin.bytes)
    throw std::runtime_error("SeedVR2 asset is not an exact regular file");
#endif
  auto* data = static_cast<std::uint8_t*>(destination);
  std::size_t done = 0;
  while (done < pin.bytes) {
    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(pin.bytes - done, 1024 * 1024));
#ifdef _WIN32
    DWORD received = 0;
    if (!ReadFile(file.handle, data + done, static_cast<DWORD>(count), &received, nullptr) || received == 0)
      throw std::runtime_error("SeedVR2 asset read failed");
#else
    const auto received = read(file.handle, data + done, count);
    if (received < 0 && errno == EINTR) continue;
    if (received <= 0) throw std::runtime_error("SeedVR2 asset read failed");
#endif
    done += static_cast<std::size_t>(received);
  }
  // Hash the actual retained storage, not a preceding pathname read/stat memo.
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned length = 0;
  if (EVP_Digest(data, done, digest.data(), &length, EVP_sha256(), nullptr) != 1 ||
      length != pin.sha256.size() || !std::equal(pin.sha256.begin(), pin.sha256.end(), digest.begin()))
    throw std::runtime_error("SeedVR2 owned asset SHA256 differs");
}
} // namespace

const std::uint8_t* SeedVrOwnedIr::binData() const noexcept {
  return bin_.empty() ? nullptr : reinterpret_cast<const std::uint8_t*>(bin_.data()) + prefix_;
}

const std::uint8_t* SeedVrOwnedIr::weightData(const SeedVrWeightSpan& span) const {
  validateSpan(span, binBytes_);
  if (!binData() || (span.offset + prefix_) % alignof(std::uint16_t) != 0)
    throw std::invalid_argument("SeedVR2 learned view is not FP16 aligned");
  return binData() + static_cast<std::size_t>(span.offset);
}

std::string readSeedVrXml(const std::filesystem::path& directory, const SeedVrAssetPin& pin) {
  validatePin(pin, kXmlLimit);
  std::string xml(static_cast<std::size_t>(pin.bytes), '\0');
  readVerified(directory, pin, xml.data());
  if (xml.find('\0') != std::string::npos)
    throw std::runtime_error("SeedVR2 XML contains a null byte");
  return xml;
}

void verifySeedVrAsset(const std::filesystem::path& directory, const SeedVrAssetPin& pin) {
  validatePin(pin, kBinLimit);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(pin.bytes));
  readVerified(directory, pin, bytes.data());
}

std::vector<float> readSeedVrPositiveText(const std::filesystem::path& directory) {
  const std::uint16_t endian = 1;
  if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 ||
      *reinterpret_cast<const std::uint8_t*>(&endian) != 1)
    throw std::runtime_error("SeedVR2 requires little-endian IEEE FP32");
  std::vector<float> text(static_cast<std::size_t>(kSeedVrPositiveText.bytes) / sizeof(float));
  readVerified(directory, kSeedVrPositiveText, text.data());
  if (!std::all_of(text.begin(), text.end(), [](float value) { return std::isfinite(value); }))
    throw std::runtime_error("SeedVR2 positive text contains nonfinite values");
  return text;
}

SeedVrOwnedIr readSeedVrIr(const std::filesystem::path& directory,
                          const SeedVrAssetPin& xml, const SeedVrAssetPin& bin,
                          const std::vector<SeedVrWeightSpan>& spans) {
  validatePin(xml, kXmlLimit);
  validatePin(bin, kBinLimit);
  if (xml.name == bin.name) throw std::invalid_argument("SeedVR2 XML/BIN pins alias");
  SeedVrOwnedIr result;
  result.prefix_ = spans.empty() ? 0 : static_cast<std::size_t>(spans.front().offset % alignof(std::uint16_t));
  for (const auto& span : spans) {
    validateSpan(span, bin.bytes);
    if ((span.offset + result.prefix_) % alignof(std::uint16_t) != 0)
      throw std::invalid_argument("SeedVR2 learned offsets have incompatible alignment");
  }
  result.xml_ = readSeedVrXml(directory, xml);
  result.binBytes_ = static_cast<std::size_t>(bin.bytes);
  result.bin_.resize((result.binBytes_ + result.prefix_ + sizeof(std::uint16_t) - 1) / sizeof(std::uint16_t));
  readVerified(directory, bin, reinterpret_cast<std::uint8_t*>(result.bin_.data()) + result.prefix_);
  return result;
}

SeedVrObservedHostMemory seedVrHostMemory() {
  SeedVrObservedHostMemory result;
#ifdef __linux__
  std::ifstream info("/proc/meminfo");
  std::string line;
  while (std::getline(info, line)) {
    if (line.rfind("MemAvailable:", 0) != 0) continue;
    std::istringstream fields(line);
    std::string label, unit, extra;
    std::uint64_t kib = 0;
    if (fields >> label >> kib >> unit && unit == "kB" && !(fields >> extra) &&
        kib <= UINT64_MAX / 1024) result.available = kib * 1024;
    break;
  }
  std::ifstream membership("/proc/self/cgroup");
  std::filesystem::path group;
  while (std::getline(membership, line)) {
    if (line.rfind("0::/", 0) != 0) continue;
    if (line.size() > 4096) return result;
    group = line.substr(3);
    for (const auto& part : group) if (part == ".." || part == ".") return result;
    break;
  }
  if (group.empty()) return result; // cgroup v1 and unknown namespaces use bounded fallback.
  const std::filesystem::path root("/sys/fs/cgroup");
  struct statfs filesystem{};
  std::error_code error;
  if (statfs(root.c_str(), &filesystem) != 0 || filesystem.f_type != CGROUP2_SUPER_MAGIC ||
      std::filesystem::exists(root / "memory.max", error) || error) return result;
  // A non-root memory-enabled cgroup has memory.max. Requiring memory at a
  // root without that file excludes private subtrees with hidden tighter ancestors.
  std::ifstream controllers(root / "cgroup.controllers");
  std::string controller;
  bool hasMemory = false;
  while (controllers >> controller) hasMemory = hasMemory || controller == "memory";
  if (!hasMemory) return result;
  const auto leaf = group == "/" ? root : root / group.relative_path();
  auto current = leaf;
  std::uint64_t headroom = UINT64_MAX;
  const auto count = [](const std::filesystem::path& path, bool limit) -> std::optional<std::uint64_t> {
    std::ifstream file(path);
    std::string value;
    if (!std::getline(file, value) || value.size() > 20) return std::nullopt;
    if (limit && value == "max") return UINT64_MAX;
    std::uint64_t parsed = 0;
    const auto end = value.data() + value.size();
    const auto converted = std::from_chars(value.data(), end, parsed);
    if (converted.ec != std::errc() || converted.ptr != end) return std::nullopt;
    return parsed;
  };
  // A tighter ancestor applies even when the worker's own cgroup is unlimited.
  while (current != root) {
    const auto limit = count(current / "memory.max", true);
    const auto used = count(current / "memory.current", false);
    if (!limit || !used) return result;
    headroom = std::min(headroom, *used < *limit ? *limit - *used : 0);
    current = current.parent_path();
  }
  // Namespace-relative /proc paths must actually resolve to this process in
  // the visible hierarchy. Also reject migration while reading the limits.
  std::ifstream processes(leaf / "cgroup.procs");
  long pid = 0;
  bool ownsLeaf = false;
  while (processes >> pid) if (pid == getpid()) { ownsLeaf = true; break; }
  membership.clear();
  membership.seekg(0);
  bool unchanged = false;
  while (std::getline(membership, line)) unchanged = unchanged || line == "0::" + group.string();
  if (!ownsLeaf || !unchanged) return result;
  result.cgroupHeadroom = headroom;
#endif
  return result;
}

bool seedVrCanRetainAssets(std::uint64_t binBytes, std::uint64_t xmlBytes,
                          const SeedVrObservedHostMemory& memory) noexcept {
  if (!memory.available || !memory.cgroupHeadroom || binBytes == 0 || binBytes > 7 * kGiB ||
      xmlBytes == 0 || xmlBytes > 6 * kXmlLimit) return false;
  // 32 alignment prefixes, bounded blobs+stream copies, provisional other-host reserve.
  constexpr std::uint64_t overhead = 32 + 768ULL * 1024 * 1024 + 4 * kGiB;
  if (binBytes > UINT64_MAX - xmlBytes || binBytes + xmlBytes > UINT64_MAX - overhead - 2 * kGiB)
    return false;
  const auto required = binBytes + xmlBytes + overhead;
  return *memory.cgroupHeadroom >= required && *memory.available >= required + 2 * kGiB;
}
} // namespace otc::ai
