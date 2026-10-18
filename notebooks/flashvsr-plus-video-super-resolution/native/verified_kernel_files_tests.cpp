#include "verified_kernel_files.hpp"

#include <openssl/evp.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct TempDir final {
  std::filesystem::path path;
  explicit TempDir(std::filesystem::path value) : path(std::move(value)) {
    require(std::filesystem::create_directory(path), "fixture directory must be new");
  }
  ~TempDir() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
};

void writeFile(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(output), "fixture write failed");
}

otc::ai::SeedVrAssetPin pin(const char* name, const std::string& bytes) {
  otc::ai::SeedVrAssetPin result{name, bytes.size(), {}};
  unsigned length = 0;
  require(EVP_Digest(bytes.data(), bytes.size(), result.sha256.data(), &length,
                     EVP_sha256(), nullptr) == 1 &&
              length == result.sha256.size(),
          "fixture digest failed");
  return result;
}

std::string readFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "snapshot read failed");
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void rejects(const std::function<void()>& callback) {
  try {
    callback();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("malformed kernel fixture was accepted");
}

void checkSealed(const std::string& path) {
  const int readFd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  require(readFd >= 0, "sealed snapshot cannot be reopened");
  constexpr int expected = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  require((fcntl(readFd, F_GET_SEALS) & expected) == expected,
          "snapshot is missing a required seal");
  close(readFd);
  const int writeFd = open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (writeFd >= 0) {
    require(write(writeFd, "x", 1) < 0, "sealed snapshot accepted a write");
    close(writeFd);
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: verified-kernel-files-tests NEW_FIXTURE_DIRECTORY");
    TempDir fixture(std::filesystem::absolute(argv[1]));
    const std::string sourceA = "__kernel void split() {}\n";
    const std::string sourceB = "__kernel void sparse() {}\n";
    const std::string config =
        "<CustomLayer><Kernel><Source filename=\"kernel.cl\"/>"
        "<Source filename=\"sparse.cl\"/></Kernel></CustomLayer>\n";
    const auto restore = [&] {
      writeFile(fixture.path / "kernel.cl", sourceA);
      writeFile(fixture.path / "sparse.cl", sourceB);
      writeFile(fixture.path / "kernel.xml", config);
    };
    restore();
    const otc::ai::VerifiedKernelSpec spec{
        pin("kernel.xml", config),
        {otc::ai::VerifiedKernelSourceSpec{pin("kernel.cl", sourceA), 1},
         otc::ai::VerifiedKernelSourceSpec{pin("sparse.cl", sourceB), 1}},
        2};

    std::string releasedConfig;
    std::array<std::string, 2> releasedSources;
    {
      auto files = otc::ai::VerifiedKernelFiles::open(fixture.path, spec);
      require(files.sourceCount() == 2, "source count changed");
      releasedConfig = files.configPath();
      releasedSources = {files.sourcePath(0), files.sourcePath(1)};
      const auto rewritten = readFile(releasedConfig);
      // Match the GPU plugin's realpath(config)->parent + '/' + Source rule,
      // including the fact that it does NOT recognize absolute source paths.
      const auto configDirectory = std::filesystem::canonical(releasedConfig).parent_path();
      for (std::size_t i = 0; i < releasedSources.size(); ++i) {
        const auto relative = std::filesystem::path(releasedSources[i]).lexically_relative(configDirectory);
        require(rewritten.find("filename=\"" + relative.generic_string() + "\"") !=
                    std::string::npos, "config source fields were not rewritten");
        require(readFile((configDirectory / relative).string()) == (i == 0 ? sourceA : sourceB),
                "canonical config-relative source cannot be reopened");
      }
      require(rewritten.find("kernel.cl") == std::string::npos &&
                  rewritten.find("sparse.cl") == std::string::npos,
              "mutable source names remain in rewritten config");
      require(readFile(releasedSources[0]) == sourceA &&
                  readFile(releasedSources[1]) == sourceB,
              "source snapshot differs");
      struct stat configStat{}, directoryStat{};
      require(lstat(releasedConfig.c_str(), &configStat) == 0 && S_ISREG(configStat.st_mode) &&
                  configStat.st_uid == geteuid() && (configStat.st_mode & 0777) == 0400 &&
                  stat(configDirectory.c_str(), &directoryStat) == 0 &&
                  directoryStat.st_uid == geteuid() && (directoryStat.st_mode & 0777) == 0700,
              "private config permissions differ");
      checkSealed(releasedSources[0]);
      checkSealed(releasedSources[1]);

      // The owner is independent of later path mutation/deletion.
      writeFile(fixture.path / "kernel.cl", "mutated source\n");
      std::filesystem::remove(fixture.path / "sparse.cl");
      std::filesystem::remove(fixture.path / "kernel.xml");
      require(readFile(releasedSources[0]) == sourceA &&
                  readFile(releasedSources[1]) == sourceB,
              "source snapshot changed after path mutation");
      require(readFile(releasedConfig).find("/proc/self/fd/") != std::string::npos,
              "config snapshot changed after path deletion");
    }
    require(!std::filesystem::exists(releasedConfig) &&
                !std::filesystem::exists(releasedSources[0]) &&
                !std::filesystem::exists(releasedSources[1]),
            "snapshot descriptors survived owner destruction");

    // Exercise both move operations while the original files are present.
    restore();
    std::string movedPath;
    {
      auto first = otc::ai::VerifiedKernelFiles::open(fixture.path, spec);
      movedPath = first.sourcePath(0);
      auto moved(std::move(first));
      auto assigned = otc::ai::VerifiedKernelFiles::open(fixture.path, spec);
      assigned = std::move(moved);
      require(readFile(assigned.sourcePath(0)) == sourceA, "move assignment lost source owner");
    }
    require(!std::filesystem::exists(movedPath), "moved snapshot was not released");

    // Missing files, wrong pins, and malformed embedded bytes fail before snapshots.
    restore();
    auto badHash = spec;
    badHash.config.sha256[0] ^= 1;
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, badHash); });
    auto badSize = spec;
    ++badSize.sources[0].asset.bytes;
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, badSize); });
    auto duplicate = spec;
    duplicate.sources[1] = duplicate.sources[0];
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, duplicate); });
    auto alias = spec;
    alias.config.name = "kernel.cl";
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, alias); });
    auto tooMany = spec;
    tooMany.sourceCount = 3;
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, tooMany); });

    std::filesystem::remove(fixture.path / "sparse.cl");
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, spec); });
    restore();

    std::filesystem::remove(fixture.path / "kernel.cl");
    std::filesystem::create_symlink("sparse.cl", fixture.path / "kernel.cl");
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, spec); });
    std::filesystem::remove(fixture.path / "kernel.cl");
    require(mkfifo((fixture.path / "kernel.cl").c_str(), 0600) == 0,
            "FIFO fixture creation failed");
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, spec); });
    std::filesystem::remove(fixture.path / "kernel.cl");
    restore();

    const std::string missingSource =
        "<CustomLayer><Source filename=\"kernel.cl\"/></CustomLayer>\n";
    writeFile(fixture.path / "kernel.xml", missingSource);
    auto missingSpec = spec;
    missingSpec.config = pin("kernel.xml", missingSource);
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, missingSpec); });
    auto oneSource = spec;
    oneSource.sourceCount = 1;
    const std::string duplicateReference =
        "<CustomLayer><Source filename=\"kernel.cl\"/>"
        "<Source filename=\"kernel.cl\"/></CustomLayer>\n";
    writeFile(fixture.path / "kernel.xml", duplicateReference);
    auto duplicateReferenceSpec = oneSource;
    duplicateReferenceSpec.config = pin("kernel.xml", duplicateReference);
    rejects([&] {
      otc::ai::VerifiedKernelFiles::open(fixture.path, duplicateReferenceSpec);
    });
    const std::string badQuote =
        "<CustomLayer><Source filename='kernel.cl'/></CustomLayer>\n";
    writeFile(fixture.path / "kernel.xml", badQuote);
    auto badQuoteSpec = oneSource;
    badQuoteSpec.config = pin("kernel.xml", badQuote);
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, badQuoteSpec); });
    const std::string badPath =
        "<CustomLayer><Source filename=\"../kernel.cl\"/></CustomLayer>\n";
    writeFile(fixture.path / "kernel.xml", badPath);
    auto badPathSpec = oneSource;
    badPathSpec.config = pin("kernel.xml", badPath);
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, badPathSpec); });
    std::string nulConfig = "<Source filename=\"kernel.cl\"/>\n";
    nulConfig.push_back('\0');
    writeFile(fixture.path / "kernel.xml", nulConfig);
    auto nulSpec = oneSource;
    nulSpec.config = pin("kernel.xml", nulConfig);
    rejects([&] { otc::ai::VerifiedKernelFiles::open(fixture.path, nulSpec); });

    std::cout << "verified_kernel_files_tests=pass gpu_calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "verified_kernel_files_tests: " << error.what() << '\n';
    return 1;
  }
}
