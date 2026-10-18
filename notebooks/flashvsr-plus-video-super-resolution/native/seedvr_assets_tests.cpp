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
    // All finite raw BF16 words survive byte-for-byte, including values beyond
    // numeric FP16's range. This path owns and authenticates the returned data.
    std::string finiteWords;
    std::vector<std::uint16_t> expectedWords;
    for (unsigned bits = 0; bits < 65536; ++bits) {
      if ((bits & 0x7f80U) == 0x7f80U) continue;
      finiteWords.push_back(static_cast<char>(bits & 0xff));
      finiteWords.push_back(static_cast<char>(bits >> 8));
      expectedWords.push_back(static_cast<std::uint16_t>(bits));
    }
    const auto tensorPin = pin("fixed.bf16", finiteWords);
    write(directory / tensorPin.name, finiteWords);
    const auto retainedWords = readBf16Asset(directory, tensorPin);
    require(retainedWords == expectedWords && retainedWords.size() == 65280);
    write(directory / tensorPin.name, std::string(finiteWords.size(), '\0'));
    require(retainedWords == expectedWords);
    rejects([&] { readBf16Asset(directory, tensorPin); });
    write(directory / tensorPin.name, finiteWords.substr(0, finiteWords.size() - 2));
    rejects([&] { readBf16Asset(directory, tensorPin); });
    rejects([&] { readBf16Asset(directory, binPin); }); // Odd byte count.
    auto oversizedTensor = tensorPin;
    oversizedTensor.bytes = 8ULL * 1024 * 1024 + 2;
    rejects([&] { readBf16Asset(directory, oversizedTensor); });
    for (const auto bits : {0x7f80U, 0xff80U, 0x7fc1U, 0xffc1U}) {
      const std::string nonfinite{static_cast<char>(bits & 0xff), static_cast<char>(bits >> 8)};
      write(directory / "nonfinite.bf16", nonfinite);
      rejects([&] { readBf16Asset(directory, pin("nonfinite.bf16", nonfinite)); });
    }
    require(readSeedVrXml(directory, xmlPin) == xml);
    verifySeedVrAsset(directory, binPin);
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
    auto aliases = readSeedVrIr(directory, xmlPin, binPin, {{0, 4, 2}, {2, 4, 2}});
    require(aliases.weightData({0, 4, 2}) + 2 == aliases.weightData({2, 4, 2}));
    // Textual IRs may have no Constant payload, but the empty structural BIN
    // is still opened, size-checked, and SHA-authenticated.
    const auto emptyBinPin = pin("empty.bin", "");
    write(directory / emptyBinPin.name, "");
    auto empty = readSeedVrIr(directory, xmlPin, emptyBinPin);
    require(empty.binBytes() == 0 && empty.binData() == nullptr);
    rejects([&] { empty.weightData({0, 2, 1}); });
    rejects([&] { verifySeedVrAsset(directory, emptyBinPin); });
    rejects([&] { readBf16Asset(directory, emptyBinPin); });
    auto wrongEmptyHash = emptyBinPin;
    wrongEmptyHash.sha256[0] ^= 1;
    rejects([&] { readSeedVrIr(directory, xmlPin, wrongEmptyHash); });
    write(directory / emptyBinPin.name, "x");
    rejects([&] { readSeedVrIr(directory, xmlPin, emptyBinPin); });
    std::filesystem::remove(directory / emptyBinPin.name);
    rejects([&] { readSeedVrIr(directory, xmlPin, emptyBinPin); });
    write(directory / emptyBinPin.name, "");
    const auto emptyXmlPin = pin("empty.xml", "");
    write(directory / emptyXmlPin.name, "");
    rejects([&] { readSeedVrIr(directory, emptyXmlPin, emptyBinPin); });
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
    rejects([&] { readBf16Asset(directory, malformed); });
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
    std::cout << "seedvr_assets_tests=pass gpu_calls=0 models_loaded=0\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
