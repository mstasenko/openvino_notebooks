#include "seedvr_assets.hpp"

#include <algorithm>
#include <limits>
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

namespace otc::ai {
namespace {
constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
constexpr std::uint64_t kXmlLimit = 8ULL * 1024 * 1024;
constexpr std::uint64_t kBinLimit = kGiB;

void validatePin(const SeedVrAssetPin& pin, std::uint64_t limit,
                bool allowEmpty = false) {
  if (pin.name.empty() || pin.name == "." || pin.name == ".." ||
      pin.name.find_first_of("/\\:") != std::string::npos ||
      pin.name.find('\0') != std::string::npos ||
      (!allowEmpty && pin.bytes == 0) || pin.bytes > limit ||
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

std::vector<std::uint16_t> readBf16Asset(const std::filesystem::path& directory,
                                       const SeedVrAssetPin& pin) {
  constexpr std::uint64_t limit = 8ULL * 1024 * 1024;
  validatePin(pin, limit);
  const std::uint16_t endian = 1;
  if (pin.bytes % sizeof(std::uint16_t) != 0 ||
      *reinterpret_cast<const std::uint8_t*>(&endian) != 1)
    throw std::invalid_argument("BF16 asset requires word-aligned little-endian storage");
  std::vector<std::uint16_t> words(static_cast<std::size_t>(pin.bytes / sizeof(std::uint16_t)));
  readVerified(directory, pin, words.data());
  if (!std::all_of(words.begin(), words.end(), [](std::uint16_t word) {
        return (word & 0x7f80U) != 0x7f80U;
      }))
    throw std::runtime_error("BF16 asset contains NaN or infinity");
  return words;
}

SeedVrOwnedIr readSeedVrIr(const std::filesystem::path& directory,
                          const SeedVrAssetPin& xml, const SeedVrAssetPin& bin,
                          const std::vector<SeedVrWeightSpan>& spans) {
  validatePin(xml, kXmlLimit);
  // A textual IR may legitimately have no Constant payload.  This is the
  // structural-BIN call site; generic asset verification remains positive-only.
  validatePin(bin, kBinLimit, true);
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
  if (result.binBytes_ != 0) {
    result.bin_.resize((result.binBytes_ + result.prefix_ + sizeof(std::uint16_t) - 1) /
                       sizeof(std::uint16_t));
    auto* storage = reinterpret_cast<std::uint8_t*>(result.bin_.data());
    readVerified(directory, bin, storage + result.prefix_);
  } else {
    // Keep a real address for the zero-length authenticated read; pointer
    // arithmetic on an empty vector's null data pointer is not valid C++.
    std::uint8_t empty = 0;
    readVerified(directory, bin, &empty);
  }
  return result;
}

} // namespace otc::ai
