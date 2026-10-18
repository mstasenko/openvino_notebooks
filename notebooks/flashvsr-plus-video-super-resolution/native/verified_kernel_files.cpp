#include "verified_kernel_files.hpp"

#ifndef __linux__

#include <stdexcept>
#include <utility>

namespace otc::ai {
class VerifiedKernelFiles::Impl {};

VerifiedKernelFiles VerifiedKernelFiles::open(const std::filesystem::path&,
                                              const VerifiedKernelSpec&) {
  throw std::runtime_error("verified kernel snapshots require Linux memfd seals");
}

VerifiedKernelFiles::VerifiedKernelFiles(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
VerifiedKernelFiles::~VerifiedKernelFiles() = default;
VerifiedKernelFiles::VerifiedKernelFiles(VerifiedKernelFiles&&) noexcept = default;
VerifiedKernelFiles& VerifiedKernelFiles::operator=(VerifiedKernelFiles&&) noexcept = default;
const std::string& VerifiedKernelFiles::configPath() const noexcept {
  static const std::string empty;
  return empty;
}
std::size_t VerifiedKernelFiles::sourceCount() const noexcept { return 0; }
const std::string& VerifiedKernelFiles::sourcePath(std::size_t) const noexcept {
  static const std::string empty;
  return empty;
}
}  // namespace otc::ai

#else

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <linux/memfd.h>
#include <stdexcept>
#include <string_view>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace otc::ai {
namespace {

constexpr std::uint64_t kMaxConfigBytes = 8ULL * 1024 * 1024;
constexpr std::size_t kMaxSources = 2;

[[noreturn]] void fail(const char* message) { throw std::runtime_error(message); }

void require(bool condition, const char* message) {
  if (!condition) fail(message);
}

void validateName(const std::string& name) {
  require(!name.empty() && name != "." && name != ".." &&
              name.find_first_of("/\\:\"'<>&") == std::string::npos &&
              name.find('\0') == std::string::npos,
          "kernel asset name is not a private basename");
}

bool samePin(const SeedVrAssetPin& left, const SeedVrAssetPin& right) {
  return left.name == right.name ||
         (left.bytes == right.bytes && left.sha256 == right.sha256);
}

struct Fd final {
  int value = -1;
  Fd() = default;
  explicit Fd(int value_) : value(value_) {}
  ~Fd() {
    if (value >= 0) close(value);
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : value(std::exchange(other.value, -1)) {}
  Fd& operator=(Fd&& other) noexcept {
    if (this != &other) {
      if (value >= 0) close(value);
      value = std::exchange(other.value, -1);
    }
    return *this;
  }
};

Fd sealedSnapshot(const char* label, const std::string& bytes) {
  const int raw = static_cast<int>(syscall(
      SYS_memfd_create, label, MFD_CLOEXEC | MFD_ALLOW_SEALING));
  require(raw >= 0, "memfd_create failed for kernel snapshot");
  Fd fd(raw);
  std::size_t done = 0;
  while (done < bytes.size()) {
    const auto written = write(fd.value, bytes.data() + done, bytes.size() - done);
    if (written < 0 && errno == EINTR) continue;
    require(written > 0, "kernel snapshot write failed");
    done += static_cast<std::size_t>(written);
  }
  require(ftruncate(fd.value, static_cast<off_t>(bytes.size())) == 0 &&
              fsync(fd.value) == 0 && lseek(fd.value, 0, SEEK_SET) == 0,
          "kernel snapshot finalization failed");
  constexpr int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  require(fcntl(fd.value, F_ADD_SEALS, seals) == 0 &&
              (fcntl(fd.value, F_GET_SEALS) & seals) == seals,
          "kernel snapshot sealing failed");
  return fd;
}

std::string fdPath(int fd) {
  require(access("/proc/self/fd", R_OK | X_OK) == 0,
          "/proc/self/fd is unavailable for kernel snapshot");
  return "/proc/self/fd/" + std::to_string(fd);
}

struct PrivateConfig final {
  std::string directory, path;
  ~PrivateConfig() {
    if (!path.empty()) unlink(path.c_str());
    if (!directory.empty()) rmdir(directory.c_str());
  }

  void createDirectory() {
    // The GPU plugin realpath()s CONFIG_FILE, so a deleted memfd cannot be the
    // config itself. A privileged worker must never stage it below a directory
    // controlled by its unprivileged caller. CL bytes still use sealed memfds.
    const char* base = geteuid() == 0 ? "/run" : "/tmp";
    Fd parent(open(base, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    struct stat status{};
    require(parent.value >= 0 && fstat(parent.value, &status) == 0 &&
                S_ISDIR(status.st_mode) &&
                ((status.st_uid == geteuid() && (status.st_mode & 0022) == 0) ||
                 (geteuid() != 0 && status.st_uid == 0 && (status.st_mode & S_ISVTX))),
            "kernel config parent is not protected");
    std::string pattern = std::string(base) + "/otc-flash-kernels-XXXXXX";
    require(mkdtemp(pattern.data()) != nullptr, "private kernel config directory failed");
    directory = pattern;
    path = directory + "/custom_layers.xml";
  }

  void write(const std::string& bytes) {
    Fd file(open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    require(file.value >= 0, "private kernel config creation failed");
    std::size_t done = 0;
    while (done < bytes.size()) {
      const auto count = ::write(file.value, bytes.data() + done, bytes.size() - done);
      if (count < 0 && errno == EINTR) continue;
      require(count > 0, "private kernel config write failed");
      done += static_cast<std::size_t>(count);
    }
    require(fchmod(file.value, 0400) == 0 && fsync(file.value) == 0,
            "private kernel config finalization failed");
  }
};

struct RewriteSource {
  std::string name;
  std::string path;
  std::size_t expected = 0;
  std::size_t found = 0;
};

std::string rewriteConfig(std::string config,
                          std::array<RewriteSource, kMaxSources>& sources,
                          std::size_t sourceCount,
                          const std::string& configDirectory) {
  std::vector<std::pair<std::size_t, std::size_t>> replacements;
  constexpr std::string_view attribute = "filename=";
  std::size_t cursor = 0;
  while (true) {
    const auto at = config.find(attribute, cursor);
    if (at == std::string::npos) break;
    require(at >= 8 && config.compare(at - 8, 8, "<Source ") == 0,
            "kernel config has an unapproved filename field");
    const auto quote = at + attribute.size();
    require(quote < config.size() && config[quote] == '"',
            "kernel config has malformed filename attribute");
    const auto end = config.find('"', quote + 1);
    require(end != std::string::npos && end + 2 < config.size() &&
                config.compare(end + 1, 2, "/>") == 0,
            "kernel config has malformed Source element");
    const auto name = config.substr(quote + 1, end - quote - 1);
    std::size_t match = sourceCount;
    for (std::size_t index = 0; index < sourceCount; ++index) {
      if (sources[index].name == name) {
        match = index;
        break;
      }
    }
    require(match < sourceCount, "kernel config contains an unapproved source path");
    ++sources[match].found;
    replacements.emplace_back(quote + 1, match);
    cursor = end + 1;
  }
  for (std::size_t index = 0; index < sourceCount; ++index)
    require(sources[index].found == sources[index].expected && sources[index].expected > 0,
            "kernel config source reference count differs from trusted table");
  for (auto it = replacements.rbegin(); it != replacements.rend(); ++it) {
    const auto& source = sources[it->second];
    // Reverse order keeps later offsets valid while replacement paths grow.
    // CustomLayer unconditionally prefixes the canonical config directory.
    // A relative route reaches the retained sealed fd without depending on
    // either an absolute-path exception or a mutable CL pathname.
    const auto relative = std::filesystem::path(source.path).lexically_relative(configDirectory);
    require(!relative.empty() && !relative.is_absolute(), "kernel source route failed");
    config.replace(it->first, source.name.size(), relative.generic_string());
  }
  return config;
}

}  // namespace

class VerifiedKernelFiles::Impl final {
 public:
  std::array<Fd, kMaxSources> sources;
  PrivateConfig config;
  std::array<std::string, kMaxSources> sourcePaths;
  std::size_t sourceCount = 0;
  std::string configPath;
};

VerifiedKernelFiles VerifiedKernelFiles::open(const std::filesystem::path& directory,
                                              const VerifiedKernelSpec& spec) {
  validateName(spec.config.name);
  require(spec.sourceCount > 0 && spec.sourceCount <= kMaxSources,
          "kernel source count is outside the fixed bound");
  require(spec.config.bytes > 0 && spec.config.bytes <= kMaxConfigBytes,
          "kernel config exceeds bounded size");
  for (std::size_t index = 0; index < spec.sourceCount; ++index) {
    const auto& source = spec.sources[index];
    validateName(source.asset.name);
    require(source.references > 0 && source.references <= 64,
            "kernel source reference count is outside the fixed bound");
    require(!samePin(spec.config, source.asset), "kernel config/source pins alias");
    for (std::size_t previous = 0; previous < index; ++previous)
      require(!samePin(spec.sources[previous].asset, source.asset),
              "kernel source pins duplicate");
  }

  // readSeedVrXml is the existing authenticated no-follow reader.  CL source
  // is intentionally treated as textual bytes too, so embedded NUL is rejected
  // before either snapshot is created.
  const auto config = readSeedVrXml(directory, spec.config);
  std::array<RewriteSource, kMaxSources> sources;
  for (std::size_t index = 0; index < spec.sourceCount; ++index)
    sources[index] = {spec.sources[index].asset.name, {},
                      spec.sources[index].references, 0};

  auto impl = std::make_unique<Impl>();
  impl->sourceCount = spec.sourceCount;
  std::array<std::string, kMaxSources> sourceBytes;
  for (std::size_t index = 0; index < spec.sourceCount; ++index) {
    sourceBytes[index] = readSeedVrXml(directory, spec.sources[index].asset);
    impl->sources[index] = sealedSnapshot("otc-flash-kernel-source", sourceBytes[index]);
    sources[index].path = fdPath(impl->sources[index].value);
  }
  impl->config.createDirectory();
  const auto rewritten = rewriteConfig(config, sources, spec.sourceCount, impl->config.directory);
  impl->config.write(rewritten);
  impl->configPath = impl->config.path;
  for (std::size_t index = 0; index < spec.sourceCount; ++index)
    impl->sourcePaths[index] = sources[index].path;
  return VerifiedKernelFiles(std::move(impl));
}

VerifiedKernelFiles::VerifiedKernelFiles(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
VerifiedKernelFiles::~VerifiedKernelFiles() = default;
VerifiedKernelFiles::VerifiedKernelFiles(VerifiedKernelFiles&&) noexcept = default;
VerifiedKernelFiles& VerifiedKernelFiles::operator=(VerifiedKernelFiles&&) noexcept = default;
const std::string& VerifiedKernelFiles::configPath() const noexcept {
  static const std::string empty;
  return impl_ ? impl_->configPath : empty;
}
std::size_t VerifiedKernelFiles::sourceCount() const noexcept {
  return impl_ ? impl_->sourceCount : 0;
}
const std::string& VerifiedKernelFiles::sourcePath(std::size_t index) const noexcept {
  static const std::string empty;
  return impl_ && index < impl_->sourceCount ? impl_->sourcePaths[index] : empty;
}

}  // namespace otc::ai

#endif
