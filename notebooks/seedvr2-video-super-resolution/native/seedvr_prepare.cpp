#include "seedvr_prepare.hpp"
#include "seedvr_prepare_detail.hpp"

#include <cstring>

namespace otc::ai::prepare_detail {
SeedVrAssetPin pin(const char* name, std::uint64_t bytes, const char* sha) {
  if (std::strlen(sha) != 64)
    throw std::runtime_error("invalid trusted SHA length");
  SeedVrAssetPin result{name, bytes, {}};
  const auto nibble = [](char value) -> unsigned {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::runtime_error("invalid SHA hex");
  };
  for (std::size_t i = 0; i < result.sha256.size(); ++i)
    result.sha256[i] = nibble(sha[2 * i]) * 16 + nibble(sha[2 * i + 1]);
  return result;
}
}  // namespace otc::ai::prepare_detail

#ifdef __linux__
#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <memory>

#include "seedvr_prepare_data.hpp"

namespace otc::ai::prepare_detail {
namespace {
constexpr std::uint64_t MiB = 1024 * 1024;
constexpr std::uint64_t GiB = 1024 * MiB;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void validateName(const std::string& name) {
  require(!name.empty() && name != "." && name != ".." &&
              name.find_first_of("/\\:") == std::string::npos &&
              name.find('\0') == std::string::npos,
          "invalid trusted basename");
}

void notify(const SeedVrPrepareCallback& callback, SeedVrPrepareStage stage,
            const std::string& name, std::uint64_t done, std::uint64_t total) {
  if (callback && !callback({stage, name, done, total}))
    throw SeedVrPrepareCancelled();
}

class Hash {
 public:
  Hash() {
    require(context_ &&
                EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) == 1,
            "SHA init failed");
  }
  void add(const void* data, std::size_t bytes) {
    require(EVP_DigestUpdate(context_.get(), data, bytes) == 1,
            "SHA update failed");
  }
  std::array<std::uint8_t, 32> finish() {
    std::array<std::uint8_t, 32> result{};
    unsigned bytes = 0;
    require(EVP_DigestFinal_ex(context_.get(), result.data(), &bytes) == 1 &&
                bytes == result.size(),
            "SHA finish failed");
    return result;
  }

 private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_{
      EVP_MD_CTX_new(), EVP_MD_CTX_free};
};

struct Fd {
  int value = -1;
  ~Fd() {
    if (value >= 0) close(value);
  }
  Fd() = default;
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
};

class VerifiedFile {
 public:
  VerifiedFile(int directory, const SeedVrAssetPin& pin,
               std::uint64_t headerBytes,
               const SeedVrPrepareCallback& progress) {
    validateName(pin.name);
    require(pin.bytes > 0 && pin.bytes <= 8 * GiB, "invalid source size");
    file_.value = openat(directory, pin.name.c_str(),
                         O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC);
    require(file_.value >= 0 && fstat(file_.value, &initial_) == 0 &&
                S_ISREG(initial_.st_mode) && initial_.st_size >= 0 &&
                static_cast<std::uint64_t>(initial_.st_size) == pin.bytes,
            "source is not exact regular file");
    size_ = pin.bytes;
    auto buffer = std::make_unique<std::uint8_t[]>(MiB);
    Hash hash;
    for (std::uint64_t offset = 0; offset < size_;) {
      notify(progress, SeedVrPrepareStage::authenticate, pin.name, offset,
             size_);
      const auto bytes =
          static_cast<std::size_t>(std::min(MiB, size_ - offset));
      read(offset, buffer.get(), bytes);
      hash.add(buffer.get(), bytes);
      offset += bytes;
    }
    require(hash.finish() == pin.sha256, "source full SHA differs");
    if (headerBytes) {
      require(headerBytes <= MiB && size_ >= 8 && headerBytes <= size_ - 8,
              "trusted header size invalid");
      std::uint8_t encoded[8];
      read(0, encoded, sizeof(encoded));
      std::uint64_t length = 0;
      for (unsigned i = 0; i < 8; ++i)
        length |= std::uint64_t(encoded[i]) << (i * 8);
      require(length == headerBytes,
              "authenticated safetensors header size differs");
      payloadStart_ = 8 + length;
    }
    unchanged();
    notify(progress, SeedVrPrepareStage::authenticate, pin.name, size_, size_);
  }
  void read(std::uint64_t offset, void* destination, std::size_t bytes) const {
    require(offset <= size_ && bytes <= size_ - offset,
            "source range outside file");
    auto* cursor = static_cast<char*>(destination);
    while (bytes) {
      const auto received =
          pread(file_.value, cursor, bytes, static_cast<off_t>(offset));
      if (received < 0 && errno == EINTR) continue;
      require(received > 0, "source truncated/read failed");
      offset += received;
      bytes -= received;
      cursor += received;
    }
  }
  void unchanged() const {
    struct stat now{};
    require(fstat(file_.value, &now) == 0 && now.st_dev == initial_.st_dev &&
                now.st_ino == initial_.st_ino &&
                now.st_size == initial_.st_size &&
                now.st_mtim.tv_sec == initial_.st_mtim.tv_sec &&
                now.st_mtim.tv_nsec == initial_.st_mtim.tv_nsec &&
                now.st_ctim.tv_sec == initial_.st_ctim.tv_sec &&
                now.st_ctim.tv_nsec == initial_.st_ctim.tv_nsec,
            "source changed after authentication");
  }
  std::uint64_t size() const { return size_; }
  std::uint64_t payloadStart() const { return payloadStart_; }

 private:
  Fd file_;
  struct stat initial_{};
  std::uint64_t size_ = 0;
  std::uint64_t payloadStart_ = 0;
};

class Output {
 public:
  explicit Output(int directory) : directory_(directory) {
    // No replaceable temporary pathname. Unsupported filesystems fail closed.
    file_.value = openat(directory_, ".", O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
    require(file_.value >= 0, "unnamed temporary creation unsupported/failed");
  }
  void write(const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const char*>(data);
    while (bytes) {
      const auto written = ::write(file_.value, cursor, bytes);
      if (written < 0 && errno == EINTR) continue;
      require(written > 0, "temporary write failed");
      bytes -= written;
      cursor += written;
    }
  }
  void publish(const std::string& name) {
    require(fsync(file_.value) == 0, "temporary fsync failed");
    // Link the verified open descriptor, never a pathname an attacker can swap.
    const auto descriptor = "/proc/self/fd/" + std::to_string(file_.value);
    require(linkat(AT_FDCWD, descriptor.c_str(), directory_, name.c_str(),
                   AT_SYMLINK_FOLLOW) == 0,
            "atomic publish refused existing target or failed");
    // If this fails, the valid linked BIN remains, but preparation is not
    // ready.
    require(fsync(directory_) == 0, "output directory fsync failed");
  }

 private:
  int directory_;
  Fd file_;
};

void reconstruct(const Plan& plan, const VerifiedFile& dit,
                 const VerifiedFile& vae, const VerifiedFile& complement,
                 int destination, const SeedVrPrepareCallback& progress) {
  validateName(plan.bin.name);
  validateName(plan.complement.name);
  require(plan.bin.bytes > 0 && plan.bin.bytes <= GiB &&
              plan.complement.bytes > 0 && plan.complement.bytes <= 11 * MiB &&
              plan.complement.bytes == complement.size() && plan.segments &&
              plan.count > 0 && plan.count <= 4096,
          "invalid reconstruction plan");
  const VerifiedFile* sources[] = {&dit, &vae, &complement};
  std::uint64_t covered = 0;
  std::uint64_t payload = 0;
  // Aliases were collapsed only after exact byte comparison by the pinned
  // proof. Validate the union, not a sum of potentially aliased learned weight
  // spans.
  for (std::size_t i = 0; i < plan.count; ++i) {
    const auto& segment = plan.segments[i];
    require(segment.source < 3 && segment.bytes > 0 &&
                segment.offset == covered &&
                segment.bytes <= plan.bin.bytes - covered,
            "output coverage/overlap invalid");
    const auto& source = *sources[segment.source];
    require(segment.sourceOffset <= source.size() &&
                segment.bytes <= source.size() - segment.sourceOffset,
            "source segment outside file");
    require(segment.learned == (segment.source < 2),
            "segment dtype/source differs");
    if (segment.learned) {
      require(segment.bytes % 2 == 0 &&
                  segment.sourceOffset >= source.payloadStart() &&
                  (segment.sourceOffset - source.payloadStart()) % 2 == 0,
              "F16 source contract differs");
    } else {
      require(segment.sourceOffset == payload, "complement coverage invalid");
      payload += segment.bytes;
    }
    covered += segment.bytes;
  }
  require(covered == plan.bin.bytes && payload == plan.complement.bytes,
          "incomplete BIN/complement coverage");
  dit.unchanged();
  vae.unchanged();
  complement.unchanged();
  Output output(destination);
  auto buffer = std::make_unique<std::uint8_t[]>(MiB);
  Hash whole;
  for (std::size_t i = 0; i < plan.count; ++i) {
    const auto& segment = plan.segments[i];
    Hash slice;
    for (std::uint64_t done = 0; done < segment.bytes;) {
      notify(progress, SeedVrPrepareStage::reconstruct, plan.bin.name,
             segment.offset + done, plan.bin.bytes);
      const auto bytes =
          static_cast<std::size_t>(std::min(MiB, segment.bytes - done));
      sources[segment.source]->read(segment.sourceOffset + done, buffer.get(),
                                    bytes);
      whole.add(buffer.get(), bytes);
      if (segment.learned) slice.add(buffer.get(), bytes);
      output.write(buffer.get(), bytes);
      done += bytes;
    }
    if (segment.learned)
      require(slice.finish() == segment.sha256,
              "learned slice changed/differs");
  }
  require(whole.finish() == plan.bin.sha256,
          "reconstructed full BIN SHA differs");
  dit.unchanged();
  vae.unchanged();
  complement.unchanged();
  notify(progress, SeedVrPrepareStage::reconstruct, plan.bin.name,
         plan.bin.bytes, plan.bin.bytes);
  // A callback may request cancellation or race a source; recheck after it.
  dit.unchanged();
  vae.unchanged();
  complement.unchanged();
  output.publish(plan.bin.name);
  notify(progress, SeedVrPrepareStage::published, plan.bin.name, plan.bin.bytes,
         plan.bin.bytes);
}

void openDirectory(Fd& fd, const std::filesystem::path& path) {
  fd.value =
      open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  require(fd.value >= 0, "managed directory unavailable or symlink");
}
}  // namespace

void prepare(const std::filesystem::path& sourceDirectory,
             const std::filesystem::path& complementDirectory,
             const std::filesystem::path& destination,
             const SeedVrAssetPin (&sourcePins)[2],
             const std::uint64_t (&headers)[2], const Plan* selectedPlans,
             std::size_t count, const SeedVrPrepareCallback& progress) {
  require(selectedPlans && count > 0 && count <= 36,
          "invalid trusted plan count");
  // Reject duplicate destinations before IO; the public table has 36 unique
  // BINs.
  for (std::size_t i = 0; i < count; ++i) {
    validateName(selectedPlans[i].bin.name);
    for (std::size_t j = 0; j < i; ++j)
      require(selectedPlans[i].bin.name != selectedPlans[j].bin.name,
              "duplicate trusted output");
  }
  Fd sourcesDirectory, payloadDirectory, outputDirectory;
  openDirectory(sourcesDirectory, sourceDirectory);
  openDirectory(payloadDirectory, complementDirectory);
  openDirectory(outputDirectory, destination);
  struct stat destinationInfo{};
  require(fstat(outputDirectory.value, &destinationInfo) == 0 &&
              destinationInfo.st_uid == geteuid() &&
              (destinationInfo.st_mode & 0077) == 0,
          "destination must be private to the preparing user");
  // Keep directory descriptors and authenticated sources stable for all 36
  // BINs.
  VerifiedFile dit(sourcesDirectory.value, sourcePins[0], headers[0], progress);
  VerifiedFile vae(sourcesDirectory.value, sourcePins[1], headers[1], progress);
  for (std::size_t i = 0; i < count; ++i) {
    const auto& plan = selectedPlans[i];
    VerifiedFile complement(payloadDirectory.value, plan.complement, 0,
                            progress);
    reconstruct(plan, dit, vae, complement, outputDirectory.value, progress);
  }
}
}  // namespace otc::ai::prepare_detail

namespace otc::ai {
void prepareSeedVrIr(const std::filesystem::path& sourceDirectory,
                     const std::filesystem::path& complementDirectory,
                     const std::filesystem::path& managedDestination,
                     const SeedVrPrepareCallback& progress) {
  const SeedVrAssetPin sources[] = {prepare_detail::source0,
                                    prepare_detail::source1};
  prepare_detail::prepare(sourceDirectory, complementDirectory,
                          managedDestination, sources,
                          prepare_detail::headerBytes, prepare_detail::plans,
                          std::size(prepare_detail::plans), progress);
}
}  // namespace otc::ai
#else
namespace otc::ai {
void prepareSeedVrIr(const std::filesystem::path&, const std::filesystem::path&,
                     const std::filesystem::path&,
                     const SeedVrPrepareCallback&) {
  throw std::runtime_error(
      "SeedVR2 preparation requires Linux O_TMPFILE and /proc/self/fd");
}
}  // namespace otc::ai
#endif
