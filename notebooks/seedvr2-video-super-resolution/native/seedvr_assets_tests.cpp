#include "seedvr_assets.hpp"
#include <openssl/evp.h>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace {
void require(bool condition) { if (!condition) throw std::runtime_error("asset test failed"); }
void rejects(const std::function<void()>& call) {
  bool rejected = false;
  try { call(); } catch (const std::exception&) { rejected = true; }
  require(rejected);
}
void write(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("fixture write failed");
}
otc::ai::SeedVrAssetPin pin(const char* name, const std::string& bytes) {
  otc::ai::SeedVrAssetPin result{name, bytes.size(), {}};
  unsigned length = 0;
  require(EVP_Digest(bytes.data(), bytes.size(), result.sha256.data(), &length, EVP_sha256(), nullptr) == 1);
  require(length == result.sha256.size());
  return result;
}
} // namespace

int main(int argc, char** argv) {
  try {
    using namespace otc::ai;
    if (argc != 2) throw std::runtime_error("usage: assets-tests NEW_FIXTURE_DIRECTORY");
    const std::filesystem::path directory = std::filesystem::absolute(argv[1]);
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("fixture directory must be new");
    const std::string xml = "<net/>";
    const std::string bin = "abcdefghijk";
    const auto xmlPin = pin("model.xml", xml), binPin = pin("model.bin", bin);
    const std::vector<SeedVrWeightSpan> odd{{1, 4, 2}, {7, 4, 2}};
    const auto restore = [&] { write(directory / xmlPin.name, xml); write(directory / binPin.name, bin); };
    restore();
    require(readSeedVrXml(directory, xmlPin) == xml);
    verifySeedVrAsset(directory, binPin);
    rejects([&] { readSeedVrPositiveText(directory); });
    write(directory / "text_positive.f32", std::string(1187840, '\0'));
    rejects([&] { readSeedVrPositiveText(directory); });
    auto owned = readSeedVrIr(directory, xmlPin, binPin, odd);
    require(owned.xml() == xml && owned.binBytes() == bin.size());
    require(std::memcmp(owned.binData(), bin.data(), bin.size()) == 0);
    for (const auto& span : odd) {
      require(reinterpret_cast<std::uintptr_t>(owned.weightData(span)) % alignof(std::uint16_t) == 0);
      require(std::memcmp(owned.weightData(span), bin.data() + span.offset, span.bytes) == 0);
    }
    const auto* beforeMove = owned.binData();
    auto moved = std::move(owned);
    require(moved.binData() == beforeMove);
    rejects([&] { owned.weightData(odd.front()); });
    // A mutation after authentication cannot alter retained XML/BIN or views.
    write(directory / xmlPin.name, "<bad/>");
    write(directory / binPin.name, "ABCDEFGHIJK");
    require(moved.xml() == xml && std::memcmp(moved.binData(), bin.data(), bin.size()) == 0);
    rejects([&] { readSeedVrIr(directory, xmlPin, binPin, odd); });
    restore();
    auto fallback = readSeedVrIr(directory, xmlPin, binPin);
    require(std::memcmp(fallback.binData(), bin.data(), bin.size()) == 0);
    require(reinterpret_cast<std::uintptr_t>(fallback.weightData({0, 4, 2})) % alignof(std::uint16_t) == 0);
    rejects([&] { fallback.weightData({1, 4, 2}); });
    rejects([&] { moved.weightData({UINT64_MAX, 4, 2}); });
    rejects([&] { moved.weightData({1, UINT64_MAX - 1, UINT64_MAX}); });
    rejects([&] { moved.weightData({1, 4, 3}); });
    rejects([&] { moved.weightData({1, 0, 0}); });
    rejects([&] { readSeedVrIr(directory, xmlPin, binPin, {{1, 4, 2}, {2, 4, 2}}); });
    auto badHash = binPin;
    badHash.sha256[0] ^= 1;
    rejects([&] { readSeedVrIr(directory, xmlPin, badHash); });
    rejects([&] { verifySeedVrAsset(directory, badHash); });
    write(directory / binPin.name, bin + "x");
    rejects([&] { readSeedVrIr(directory, xmlPin, binPin); });
    write(directory / binPin.name, bin.substr(0, bin.size() - 1));
    rejects([&] { readSeedVrIr(directory, xmlPin, binPin); });
    restore();
    auto malformed = binPin;
    malformed.bytes = UINT64_MAX;
    rejects([&] { readSeedVrIr(directory, xmlPin, malformed); });
    rejects([&] { verifySeedVrAsset(directory, malformed); });
    malformed = binPin; malformed.name = "../model.bin";
    rejects([&] { readSeedVrIr(directory, xmlPin, malformed); });
    malformed = binPin; malformed.name = std::string("model.bin\0other", 15);
    rejects([&] { readSeedVrIr(directory, xmlPin, malformed); });
    rejects([&] { readSeedVrIr(directory, xmlPin, xmlPin); });
    const std::string nullXml("<net/>\0", 7);
    write(directory / "null.xml", nullXml);
    rejects([&] { readSeedVrXml(directory, pin("null.xml", nullXml)); });
    rejects([&] { readSeedVrIr(directory, pin("null.xml", nullXml), binPin); });
    std::filesystem::create_directory(directory / "not-file.bin");
    malformed = binPin; malformed.name = "not-file.bin";
    rejects([&] { readSeedVrIr(directory, xmlPin, malformed); });
#ifndef _WIN32
    std::filesystem::create_symlink(directory / binPin.name, directory / "link.bin");
    malformed = binPin; malformed.name = "link.bin";
    rejects([&] { readSeedVrIr(directory, xmlPin, malformed); });
    rejects([&] { verifySeedVrAsset(directory, malformed); });
    std::filesystem::create_directory_symlink(directory, directory / "link-dir");
    rejects([&] { readSeedVrIr(directory / "link-dir", xmlPin, binPin); });
    require(mkfifo((directory / "fifo.bin").c_str(), 0600) == 0);
    malformed = binPin; malformed.name = "fifo.bin";
    rejects([&] { readSeedVrIr(directory, xmlPin, malformed); });
    rejects([&] { verifySeedVrAsset(directory, malformed); });
#endif
    constexpr std::uint64_t bins = 6672856292ULL, templates = 1801795ULL;
    require(!seedVrCanRetainAssets(bins, templates, {}));
    require(!seedVrCanRetainAssets(bins, templates, {19ULL << 30, std::nullopt}));
    require(!seedVrCanRetainAssets(bins, templates, {19ULL << 30, 8ULL << 30}));
    require(!seedVrCanRetainAssets(bins, templates, {8ULL << 30, 12ULL << 30}));
    require(seedVrCanRetainAssets(bins, templates, {19ULL << 30, 12ULL << 30}));
    require(!seedVrCanRetainAssets(UINT64_MAX, templates, {UINT64_MAX, UINT64_MAX}));
    require(!seedVrCanRetainAssets(bins, UINT64_MAX, {UINT64_MAX, UINT64_MAX}));
    require(!seedVrCanRetainAssets(0, templates, {UINT64_MAX, UINT64_MAX}));
    const auto observed = seedVrHostMemory();
    if (observed.available) require(*observed.available > 0);
    // Report observations for resource-capped integration checks; never supply
    // them to the worker from app/IPC or treat the snapshot as a reservation.
    std::cout << "observed_available_bytes=" << observed.available.value_or(0)
              << " observed_cgroup_headroom_bytes=" << observed.cgroupHeadroom.value_or(0) << '\n';
    std::cout << "seedvr_assets_tests=pass gpu_calls=0 models_loaded=0\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
