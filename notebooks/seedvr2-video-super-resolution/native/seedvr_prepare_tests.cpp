#include <openssl/evp.h>
#include <sys/stat.h>

#include <cassert>
#include <fstream>
#include <iostream>

#include "seedvr_prepare_detail.hpp"

using namespace otc::ai;
using namespace otc::ai::prepare_detail;
namespace {
SeedVrAssetPin bytesPin(const char* name, const std::string& bytes) {
  SeedVrAssetPin result{name, bytes.size(), {}};
  unsigned count = 0;
  assert(EVP_Digest(bytes.data(), bytes.size(), result.sha256.data(), &count,
                    EVP_sha256(), nullptr) == 1 &&
         count == 32);
  return result;
}
void write(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(bytes.data(), bytes.size());
  file.close();
  assert(file);
}
unsigned rejected = 0;
template <class Function>
void reject(const char* name, Function test) {
  bool caught = false;
  try {
    test();
  } catch (const std::exception&) {
    caught = true;
  }
  assert(caught);
  ++rejected;
  std::cout << "reject " << name << " PASS\n";
}
}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path directory = argv[1];
  assert(std::filesystem::create_directory(directory));
  std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
  write(directory / "source", "abcd");
  write(directory / "payload", "XY");
  SeedVrAssetPin sources[] = {bytesPin("source", "abcd"),
                              bytesPin("source", "abcd")};
  std::uint64_t headers[] = {0, 0};
  const auto payload = bytesPin("payload", "XY");
  Segment segments[] = {{0, 4, 0, 0, true, sources[0].sha256},
                        {4, 2, 0, 2, false, {}}};
  Plan plan{bytesPin("result.bin", "abcdXY"), payload, segments, 2};
  auto run = [&](const Plan& p, const SeedVrPrepareCallback& progress = {}) {
    prepare(directory, directory, directory, sources, headers, &p, 1, progress);
  };
  auto absent = [&](const char* name) {
    assert(!std::filesystem::exists(directory / name));
  };
  std::filesystem::permissions(
      directory,
      std::filesystem::perms::owner_all | std::filesystem::perms::others_read);
  reject("nonprivate destination", [&] { run(plan); });
  std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
  const auto directoryLink = directory / "directory-link";
  std::filesystem::create_directory_symlink(directory, directoryLink);
  reject("destination directory symlink", [&] {
    prepare(directory, directory, directoryLink, sources, headers, &plan, 1,
            {});
  });
  Plan candidate = plan;
  candidate.bin.name = "cancel-auth.bin";
  reject("authentication cancellation",
         [&] { run(candidate, [](const auto&) { return false; }); });
  absent("cancel-auth.bin");
  candidate.bin.name = "cancel-stream.bin";
  bool typedCancellation = false;
  try {
    run(candidate, [](const auto& p) {
      return p.stage != SeedVrPrepareStage::reconstruct;
    });
  } catch (const SeedVrPrepareCancelled&) {
    typedCancellation = true;
  }
  assert(typedCancellation);
  absent("cancel-stream.bin");
  candidate.bin.name = "cancel-final.bin";
  reject("cancellation before atomic publish", [&] {
    run(candidate, [](const auto& p) {
      return !(p.stage == SeedVrPrepareStage::reconstruct && p.done == p.total);
    });
  });
  absent("cancel-final.bin");
  unsigned events = 0;
  run(plan, [&](const auto& p) {
    assert(p.done <= p.total);
    ++events;
    return true;
  });
  assert(events > 0);
  std::ifstream file(directory / "result.bin", std::ios::binary);
  assert(std::string(std::istreambuf_iterator<char>(file), {}) == "abcdXY");
  reject("existing authenticated output not overwritten", [&] { run(plan); });
  auto invalid = plan;
  invalid.bin.name = "../escape";
  reject("target traversal", [&] { run(invalid); });
  invalid = plan;
  invalid.bin.name = "wrong.bin";
  invalid.bin.sha256[0] ^= 1;
  reject("wrong target hash", [&] { run(invalid); });
  absent("wrong.bin");
  invalid = plan;
  invalid.bin.name = "negative.bin";
  auto first = segments[0];
  segments[0].bytes = UINT64_MAX;
  reject("overflow", [&] { run(invalid); });
  segments[0] = first;
  segments[0].sourceOffset = 4;
  reject("truncated span", [&] { run(invalid); });
  segments[0] = first;
  segments[0].bytes = 3;
  reject("odd F16", [&] { run(invalid); });
  segments[0] = first;
  segments[0].source = 3;
  reject("unknown source", [&] { run(invalid); });
  segments[0] = first;
  segments[0].sha256[0] ^= 1;
  reject("wrong learned slice", [&] { run(invalid); });
  segments[0] = first;
  auto tail = segments[1];
  segments[1].offset = 3;
  reject("overlap alias", [&] { run(invalid); });
  segments[1] = tail;
  segments[1].offset = 5;
  reject("coverage gap", [&] { run(invalid); });
  segments[1] = tail;
  absent("negative.bin");
  const auto original = sources[0];
  sources[0].bytes = 3;
  reject("wrong source size", [&] { run(invalid); });
  sources[0] = original;
  sources[0].sha256[0] ^= 1;
  reject("wrong source hash", [&] { run(invalid); });
  sources[0] = original;
  std::filesystem::create_symlink("source", directory / "link");
  sources[0].name = "link";
  reject("symlink source", [&] { run(invalid); });
  sources[0] = original;
  assert(mkfifo((directory / "fifo").c_str(), 0600) == 0);
  sources[0].name = "fifo";
  reject("nonregular source", [&] { run(invalid); });
  sources[0] = original;
  headers[0] = 1;
  reject("malformed safetensor header", [&] { run(invalid); });
  headers[0] = 0;
  std::filesystem::create_symlink("source", directory / "target.bin");
  invalid.bin.name = "target.bin";
  reject("target symlink no overwrite", [&] { run(invalid); });
  invalid.bin.name = "mutated.bin";
  reject("source mutation after authentication", [&] {
    run(invalid, [&](const auto& p) {
      if (p.stage == SeedVrPrepareStage::reconstruct && p.done == p.total)
        write(directory / "source", "zzzz");
      return true;
    });
  });
  absent("mutated.bin");
  write(directory / "source", "abcd");
  invalid.bin.name = "truncated.bin";
  reject("source truncation during reconstruction", [&] {
    run(invalid, [&](const auto& p) {
      if (p.stage == SeedVrPrepareStage::reconstruct && p.done == 0)
        write(directory / "source", "a");
      return true;
    });
  });
  absent("truncated.bin");
  write(directory / "source", "abcd");
  Plan duplicate[] = {plan, plan};
  reject("duplicate plans", [&] {
    prepare(directory, directory, directory, sources, headers, duplicate, 2,
            {});
  });
  invalid = plan;
  invalid.bin.name = "partial.bin";
  bool published = false;
  reject("cancellation after publish leaves verified file", [&] {
    run(invalid, [&](const auto& p) {
      if (p.stage == SeedVrPrepareStage::published) {
        published = true;
        return false;
      }
      return true;
    });
  });
  assert(published &&
         std::filesystem::file_size(directory / "partial.bin") == 6);
  // Cancellation is polled between bounded chunks, not just between BINs.
  const std::string large(2 * 1024 * 1024, 'x');
  write(directory / "large", large);
  sources[0] = bytesPin("large", large);
  unsigned authPolls = 0;
  reject("cancel between MiB chunks", [&] {
    run(invalid, [&](const auto& p) {
      if (p.stage == SeedVrPrepareStage::authenticate && p.name == "large")
        return ++authPolls < 2;
      return true;
    });
  });
  assert(authPolls == 2);
  std::cout << "PASS verified publication, progress, typed cancellation; "
            << rejected << " negative cases\n";
}
