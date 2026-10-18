#include <openssl/evp.h>
#include <sys/stat.h>

#include <cassert>
#include <array>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>

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
std::string f32Bytes(std::initializer_list<std::uint32_t> words) {
  std::string result;
  result.reserve(words.size() * 4);
  for (const auto word : words)
    for (unsigned byte = 0; byte < 4; ++byte)
      result.push_back(static_cast<char>(word >> (byte * 8)));
  return result;
}
std::string bf16Bytes(std::initializer_list<std::uint16_t> words) {
  std::string result;
  result.reserve(words.size() * 2);
  for (const auto word : words) {
    result.push_back(static_cast<char>(word));
    result.push_back(static_cast<char>(word >> 8));
  }
  return result;
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
  static_assert(kMaxPreparedPlanCount == 96U);
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
  const auto invalidDestination = directory / "invalid-destination";
  assert(std::filesystem::create_directory(invalidDestination));
  std::filesystem::permissions(
      invalidDestination,
      std::filesystem::perms::owner_all | std::filesystem::perms::others_read);
  unsigned invalidDestinationCallbacks = 0;
  bool destinationError = false;
  try {
    prepare(directory / "missing-source", directory / "missing-payload",
            invalidDestination, sources, headers, &plan, 1,
            [&](const auto&) {
              ++invalidDestinationCallbacks;
              return true;
            });
  } catch (const std::runtime_error& error) {
    destinationError =
        std::string(error.what()).find("destination must be private") !=
        std::string::npos;
  }
  assert(destinationError && invalidDestinationCallbacks == 0);
  std::filesystem::remove(invalidDestination);

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
  // An IR with no Constants uses an authenticated empty structural complement;
  // the positive learned source and output BIN still remain mandatory.
  write(directory / "empty-complement", "");
  const auto emptyComplement = bytesPin("empty-complement", "");
  Segment emptySegments[] = {{0, 4, 0, 0, true, sources[0].sha256}};
  Plan emptyPlan{bytesPin("empty-complement.bin", "abcd"), emptyComplement,
                 emptySegments, 1};
  run(emptyPlan);
  std::ifstream emptyOutput(directory / "empty-complement.bin",
                            std::ios::binary);
  assert(std::string(std::istreambuf_iterator<char>(emptyOutput), {}) ==
         "abcd");
  auto badEmptyHash = emptyPlan;
  badEmptyHash.bin.name = "empty-wrong-hash.bin";
  badEmptyHash.complement.sha256[0] ^= 1;
  reject("empty complement wrong hash", [&] { run(badEmptyHash); });
  absent("empty-wrong-hash.bin");
  write(directory / "empty-complement", "x");
  auto nonemptyComplement = emptyPlan;
  nonemptyComplement.bin.name = "empty-nonempty.bin";
  reject("empty complement size mismatch", [&] { run(nonemptyComplement); });
  absent("empty-nonempty.bin");
  write(directory / "empty-complement", "");
  std::filesystem::remove(directory / "empty-complement");
  auto missingComplement = emptyPlan;
  missingComplement.bin.name = "empty-missing.bin";
  reject("empty complement missing", [&] { run(missingComplement); });
  absent("empty-missing.bin");
  write(directory / "empty-complement", "");
  auto zeroOutput = emptyPlan;
  zeroOutput.bin = bytesPin("zero-output.bin", "");
  reject("zero output BIN", [&] { run(zeroOutput); });
  absent("zero-output.bin");
  Segment zeroLearnedSegments[] = {{0, 0, 0, 0, true, {}}};
  auto zeroLearnedPlan = emptyPlan;
  zeroLearnedPlan.bin = bytesPin("zero-learned.bin", "abcd");
  zeroLearnedPlan.segments = zeroLearnedSegments;
  reject("zero learned span", [&] { run(zeroLearnedPlan); });
  absent("zero-learned.bin");
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

  // The Flash learned table is intentionally one bounded operation: all 96
  // compiled plans share the two authenticated source descriptors, and the
  // preparer must reject a 97th plan before opening any directory/source.
  const auto plans96Directory = directory / "plans96";
  assert(std::filesystem::create_directory(plans96Directory));
  std::filesystem::permissions(plans96Directory,
                               std::filesystem::perms::owner_all);
  write(plans96Directory / "source96-0", "abcd");
  write(plans96Directory / "source96-1", "efgh");
  write(plans96Directory / "empty96", "");
  const SeedVrAssetPin sources96[] = {
      bytesPin("source96-0", "abcd"), bytesPin("source96-1", "efgh")};
  const std::uint64_t headers96[] = {0, 0};
  const auto empty96 = bytesPin("empty96", "");
  Segment segments96[] = {
      {0, 4, 0, 0, true, sources96[0].sha256},
  };
  std::array<Plan, kMaxPreparedPlanCount> plans96{};
  for (std::size_t i = 0; i < plans96.size(); ++i) {
    const auto name = "flash-plan-" + std::to_string(i) + ".bin";
    plans96[i] = {bytesPin(name.c_str(), "abcd"), empty96, segments96, 1};
  }
  unsigned source96Completions[2] = {0, 0};
  prepare(plans96Directory, plans96Directory, plans96Directory, sources96,
          headers96, plans96.data(), plans96.size(), [&](const auto& event) {
            for (unsigned source = 0; source < 2; ++source) {
              if (event.stage == SeedVrPrepareStage::authenticate &&
                  event.name == sources96[source].name &&
                  event.done == event.total)
                ++source96Completions[source];
            }
            return true;
          });
  assert(source96Completions[0] == 1 && source96Completions[1] == 1);
  for (const auto& plan96 : plans96) {
    assert(std::filesystem::file_size(plans96Directory / plan96.bin.name) ==
           plan96.bin.bytes);
  }

  const auto cancelled96Directory = directory / "plans96-cancel";
  assert(std::filesystem::create_directory(cancelled96Directory));
  std::filesystem::permissions(cancelled96Directory,
                               std::filesystem::perms::owner_all);
  write(cancelled96Directory / "source96-0", "abcd");
  write(cancelled96Directory / "source96-1", "efgh");
  write(cancelled96Directory / "empty96", "");
  bool cancelled96 = false;
  try {
    prepare(cancelled96Directory, cancelled96Directory,
            cancelled96Directory, sources96, headers96, plans96.data(),
            plans96.size(), [](const auto& event) {
              return !(event.stage == SeedVrPrepareStage::authenticate &&
                       event.done == 0);
            });
  } catch (const SeedVrPrepareCancelled&) {
    cancelled96 = true;
  }
  assert(cancelled96);
  assert(!std::filesystem::exists(cancelled96Directory / "flash-plan-0.bin"));

  std::array<Plan, kMaxPreparedPlanCount + 1> plans97{};
  std::copy(plans96.begin(), plans96.end(), plans97.begin());
  plans97.back() = plans96.back();
  plans97.back().bin.name = "flash-plan-96.bin";
  unsigned rejected97AuthEvents = 0;
  reject("97-plan bound before authentication", [&] {
    prepare(plans96Directory, plans96Directory, plans96Directory,
            sources96, headers96, plans97.data(), plans97.size(),
            [&](const auto&) {
              ++rejected97AuthEvents;
              return true;
            });
  });
  assert(rejected97AuthEvents == 0);
  Plan duplicate96[] = {plans96[0], plans96[0]};
  unsigned duplicate96AuthEvents = 0;
  reject("duplicate 96-table plan before authentication", [&] {
    prepare(plans96Directory, plans96Directory, plans96Directory,
            sources96, headers96, duplicate96, 2, [&](const auto&) {
              ++duplicate96AuthEvents;
              return true;
            });
  });
  assert(duplicate96AuthEvents == 0);

  // The private Flash bridge may convert only learned DiT/VAE F32 spans. The
  // source is padded so alignment and out-of-range checks exercise distinct
  // failure paths while the successful span remains exactly nine F32 words.
  const auto f32 = f32Bytes({0x3f808000U, 0x3f818000U, 0xbf808000U,
                             0xbf818000U, 0x00000000U, 0x80000000U,
                             0x00000001U, 0x007fffffU, 0x7f7f0000U});
  const auto f32Source = f32 + std::string(4, '\0');
  const auto converted = bf16Bytes({0x3f80U, 0x3f82U, 0xbf80U, 0xbf82U,
                                    0x0000U, 0x8000U, 0x0000U, 0x0080U,
                                    0x7f7fU});
  write(directory / "f32source", f32Source);
  const auto convertedSpan = bytesPin("converted-span", converted);
  SeedVrAssetPin f32Sources[] = {bytesPin("f32source", f32Source),
                                 bytesPin("f32source", f32Source)};
  std::uint64_t f32Headers[] = {0, 0};
  Segment f32Segments[] = {
      {0, converted.size(), 0, 0, true, convertedSpan.sha256,
       SegmentEncoding::f32_to_bf16_rne},
      {converted.size(), 2, 0, 2, false, {}}};
  const auto convertedOutput = converted + "XY";
  Plan f32Plan{bytesPin("converted.bin", convertedOutput), payload,
               f32Segments, 2};
  auto runF32 = [&](const Plan& p,
                    const SeedVrPrepareCallback& progress) {
    prepare(directory, directory, directory, f32Sources, f32Headers, &p, 1,
            progress);
  };
  runF32(f32Plan, {});
  std::ifstream convertedFile(directory / "converted.bin", std::ios::binary);
  assert(std::string(std::istreambuf_iterator<char>(convertedFile), {}) ==
         convertedOutput);

  const SeedVrAssetPin f32SingleSource = f32Sources[0];
  auto runSingle = [&](const Plan& p,
                       const SeedVrPrepareCallback& progress = {}) {
    prepare(directory, directory, directory, f32SingleSource, 0, &p, 1,
            progress);
  };
  assert(!std::filesystem::exists(directory / "source1"));
  auto singlePlan = f32Plan;
  singlePlan.bin.name = "single-converted.bin";
  unsigned singleSourceAuthEvents = 0;
  runSingle(singlePlan, [&](const auto& event) {
    if (event.stage == SeedVrPrepareStage::authenticate &&
        event.name == f32SingleSource.name) {
      ++singleSourceAuthEvents;
    }
    return true;
  });
  assert(singleSourceAuthEvents == 2);
  std::ifstream singleFile(directory / "single-converted.bin",
                           std::ios::binary);
  assert(std::string(std::istreambuf_iterator<char>(singleFile), {}) ==
         convertedOutput);

  Segment unavailableSourceSegments[] = {
      {0, 2, 0, 1, true, bytesPin("unavailable-source-span", converted.substr(0, 2)).sha256},
      {2, 2, 0, 2, false, {}}};
  Plan unavailableSourcePlan{
      bytesPin("single-source1.bin", converted.substr(0, 2) + "XY"),
      payload, unavailableSourceSegments, 2};
  reject("single-source source1 unavailable", [&] {
    runSingle(unavailableSourcePlan);
  });
  absent("single-source1.bin");

  Segment complementConversionSegments[] = {
      f32Segments[0],
      {converted.size(), 2, 0, 2, false, {},
       SegmentEncoding::f32_to_bf16_rne}};
  Plan complementConversionPlan{
      bytesPin("single-complement-conversion.bin", convertedOutput), payload,
      complementConversionSegments, 2};
  reject("single-source complement conversion", [&] {
    runSingle(complementConversionPlan);
  });
  absent("single-complement-conversion.bin");

  Segment singleMalformedSegments[2] = {f32Segments[0], f32Segments[1]};
  auto rejectSinglePlan = [&](const char* label, const char* outputName) {
    auto malformedSingle = f32Plan;
    malformedSingle.bin.name = outputName;
    malformedSingle.segments = singleMalformedSegments;
    reject(label, [&] { runSingle(malformedSingle); });
    absent(outputName);
  };
  singleMalformedSegments[0].sourceOffset = 2;
  rejectSinglePlan("single-source F32 alignment", "single-align.bin");
  singleMalformedSegments[0] = f32Segments[0];
  singleMalformedSegments[0].sourceOffset = 8;
  rejectSinglePlan("single-source F32 short slice", "single-short.bin");
  singleMalformedSegments[0] = f32Segments[0];

  auto wrongSingleSource = f32SingleSource;
  wrongSingleSource.sha256[0] ^= 1;
  auto wrongSinglePlan = singlePlan;
  wrongSinglePlan.bin.name = "single-wrong-source.bin";
  reject("single-source wrong hash", [&] {
    prepare(directory, directory, directory, wrongSingleSource, 0,
            &wrongSinglePlan, 1, {});
  });
  absent("single-wrong-source.bin");

  std::filesystem::create_symlink("f32source", directory / "single-link");
  auto symlinkSingleSource = f32SingleSource;
  symlinkSingleSource.name = "single-link";
  auto symlinkSinglePlan = singlePlan;
  symlinkSinglePlan.bin.name = "single-symlink.bin";
  reject("single-source symlink", [&] {
    prepare(directory, directory, directory, symlinkSingleSource, 0,
            &symlinkSinglePlan, 1, {});
  });
  absent("single-symlink.bin");

  auto cancelledSinglePlan = singlePlan;
  cancelledSinglePlan.bin.name = "single-cancel.bin";
  reject("single-source cancellation", [&] {
    runSingle(cancelledSinglePlan, [](const auto& event) {
      return event.stage != SeedVrPrepareStage::reconstruct;
    });
  });
  absent("single-cancel.bin");

  auto mutatedSinglePlan = singlePlan;
  mutatedSinglePlan.bin.name = "single-mutated.bin";
  reject("single-source mutation", [&] {
    runSingle(mutatedSinglePlan, [&](const auto& event) {
      if (event.stage == SeedVrPrepareStage::reconstruct && event.done == 0)
        write(directory / "f32source", f32Source + "x");
      return true;
    });
  });
  absent("single-mutated.bin");
  write(directory / "f32source", f32Source);
  reject("single-source existing output not overwritten", [&] {
    runSingle(singlePlan);
  });

  auto malformed = f32Plan;
  Segment malformedSegments[2];
  auto rejectF32Plan = [&](const char* label, const char* outputName) {
    malformed.bin.name = outputName;
    malformed.segments = malformedSegments;
    reject(label, [&] { runF32(malformed, {}); });
    absent(outputName);
  };
  malformedSegments[0] = f32Segments[0];
  malformedSegments[1] = f32Segments[1];
  malformedSegments[0].bytes -= 1;
  malformedSegments[1].offset = malformedSegments[0].bytes;
  rejectF32Plan("F32 odd output", "f32-odd.bin");
  malformedSegments[0] = f32Segments[0];
  malformedSegments[1] = f32Segments[1];
  malformedSegments[0].sourceOffset = 2;
  rejectF32Plan("F32 source alignment", "f32-align.bin");
  malformedSegments[0].sourceOffset = 8;
  rejectF32Plan("F32 source range", "f32-range.bin");
  malformedSegments[0].sourceOffset = 0;
  malformedSegments[0].encoding = static_cast<SegmentEncoding>(99);
  rejectF32Plan("unknown encoding", "f32-encoding.bin");
  malformedSegments[0] = f32Segments[0];
  malformedSegments[0].learned = false;
  rejectF32Plan("nonlearned conversion", "f32-unlearned.bin");
  malformedSegments[0] = f32Segments[0];
  malformedSegments[0].bytes = UINT64_MAX;
  rejectF32Plan("conversion output overflow", "f32-overflow.bin");

  write(directory / "oddsource", "abc");
  const SeedVrAssetPin oddSources[] = {bytesPin("oddsource", "abc"),
                                       bytesPin("oddsource", "abc")};
  Segment oddSegments[] = {{0, 3, 0, 0, true, oddSources[0].sha256},
                           {3, 2, 0, 2, false, {}}};
  Plan oddPlan{bytesPin("odd-copy.bin", "abcXY"), payload, oddSegments, 2};
  reject("odd learned copy span", [&] {
    prepare(directory, directory, directory, oddSources, headers, &oddPlan, 1,
            {});
  });
  absent("odd-copy.bin");

  const auto oneConverted = bf16Bytes({0x3f80U});
  const auto oneConvertedPin = bytesPin("one-converted", oneConverted);
  write(directory / "odd-payload", "Q");
  const auto oddPayload = bytesPin("odd-payload", "Q");
  Segment oddOffsetSegments[] = {
      {0, 1, 0, 2, false, {}},
      {1, oneConverted.size(), 0, 0, true, oneConvertedPin.sha256,
       SegmentEncoding::f32_to_bf16_rne}};
  Plan oddOffsetPlan{bytesPin("f32-odd-offset.bin", "Q" + oneConverted),
                     oddPayload, oddOffsetSegments, 2};
  reject("F32 odd output offset", [&] { runF32(oddOffsetPlan, {}); });
  absent("f32-odd-offset.bin");

  auto badTarget = f32Plan;
  badTarget.bin.name = "f32-target-hash.bin";
  badTarget.bin.sha256[0] ^= 1;
  reject("F32 target hash", [&] { runF32(badTarget, {}); });
  absent("f32-target-hash.bin");
  auto cancelledF32 = f32Plan;
  cancelledF32.bin.name = "converted-cancel.bin";
  reject("F32 reconstruction cancellation", [&] {
    runF32(cancelledF32, [](const auto& event) {
      return event.stage != SeedVrPrepareStage::reconstruct;
    });
  });
  absent("converted-cancel.bin");

  auto rejectF32Word = [&](const char* label, const char* sourceName,
                           const char* outputName, std::uint32_t word,
                           std::uint16_t expectedWord) {
    const auto sourceBytes = f32Bytes({word});
    const auto convertedWord = bf16Bytes({expectedWord});
    write(directory / sourceName, sourceBytes);
    const SeedVrAssetPin badSources[] = {bytesPin(sourceName, sourceBytes),
                                         bytesPin(sourceName, sourceBytes)};
    Segment badSegments[] = {
        {0, 2, 0, 0, true, bytesPin("bad-span", convertedWord).sha256,
         SegmentEncoding::f32_to_bf16_rne},
        {2, 2, 0, 2, false, {}}};
    Plan badPlan{bytesPin(outputName, convertedWord + "XY"), payload,
                 badSegments, 2};
    reject(label, [&] {
      prepare(directory, directory, directory, badSources, f32Headers,
              &badPlan, 1, {});
    });
    absent(outputName);
  };
  rejectF32Word("F32 NaN", "f32-nan", "f32-nan.bin", 0x7fc00000U,
                0x7fc0U);
  rejectF32Word("F32 infinity", "f32-inf", "f32-inf.bin", 0x7f800000U,
                0x7f80U);
  rejectF32Word("finite F32 BF16 overflow", "f32-bf16-overflow",
                "f32-bf16-overflow.bin", 0x7f7f8000U, 0x7f80U);

  const std::size_t longInputBytes = 2 * 1024 * 1024;
  std::string longF32(longInputBytes, '\0');
  std::string longConverted(longInputBytes / 2, '\0');
  for (std::size_t i = 0; i < longF32.size(); i += 4) {
    longF32[i + 2] = static_cast<char>(0x80);
    longF32[i + 3] = static_cast<char>(0x3f);
  }
  for (std::size_t i = 0; i < longConverted.size(); i += 2) {
    longConverted[i] = static_cast<char>(0x80);
    longConverted[i + 1] = static_cast<char>(0x3f);
  }
  write(directory / "f32-long", longF32);
  const SeedVrAssetPin longSources[] = {bytesPin("f32-long", longF32),
                                        bytesPin("f32-long", longF32)};
  const auto longSpan = bytesPin("long-span", longConverted);
  Segment longSegments[] = {
      {0, longConverted.size(), 0, 0, true, longSpan.sha256,
       SegmentEncoding::f32_to_bf16_rne},
      {longConverted.size(), 2, 0, 2, false, {}}};
  Plan longPlan{bytesPin("f32-long.bin", longConverted + "XY"), payload,
                longSegments, 2};
  auto runLong = [&](const Plan& p, const SeedVrPrepareCallback& progress) {
    prepare(directory, directory, directory, longSources, f32Headers, &p, 1,
            progress);
  };
  auto cancelledLong = longPlan;
  cancelledLong.bin.name = "f32-long-cancel.bin";
  unsigned conversionPolls = 0;
  std::uint64_t secondConversionDone = 0;
  reject("F32 cancellation after first conversion chunk", [&] {
    runLong(cancelledLong, [&](const auto& event) {
      if (event.stage == SeedVrPrepareStage::reconstruct) {
        ++conversionPolls;
        if (conversionPolls == 2) secondConversionDone = event.done;
        return conversionPolls < 2;
      }
      return true;
    });
  });
  assert(conversionPolls == 2 &&
         secondConversionDone == longConverted.size() / 2);
  absent("f32-long-cancel.bin");
  runLong(longPlan, {});
  std::ifstream longFile(directory / "f32-long.bin", std::ios::binary);
  assert(std::string(std::istreambuf_iterator<char>(longFile), {}) ==
         longConverted + "XY");
  auto mutatedF32 = f32Plan;
  mutatedF32.bin.name = "converted-mutated.bin";
  reject("F32 source mutation", [&] {
    runF32(mutatedF32, [&](const auto& event) {
      if (event.stage == SeedVrPrepareStage::reconstruct && event.done == 0)
        write(directory / "f32source", f32Source + "x");
      return true;
    });
  });
  absent("converted-mutated.bin");
  write(directory / "f32source", f32Source);
  std::cout << "PASS verified publication, progress, typed cancellation; "
            << rejected << " negative cases\n";
}
