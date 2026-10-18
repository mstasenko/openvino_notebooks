#include "flash_prepare.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "flash-assets/flash_cross_assets.hpp"
#include "flash-assets/flash_decoder_assets.hpp"
#include "flash-assets/flash_edges_prefix_assets.hpp"
#include "flash-assets/flash_edges_suffix_assets.hpp"
#include "flash-assets/flash_attention_assets.hpp"
#include "flash-assets/flash_fixed_assets.hpp"
#include "flash-assets/flash_lq_assets.hpp"
#include "flash-assets/flash_post_assets.hpp"
#include "flash-assets/flash_pre_assets.hpp"
#include "flash-assets/flash_text_assets.hpp"
#include "seedvr_prepare_detail.hpp"

namespace otc::ai {
namespace {

struct RoleTables {
  const SeedVrAssetPin* sources;
  const std::uint64_t* headers;
  const prepare_detail::Plan* plans;
  std::size_t planCount;
  const char* sourceCommit;
  const char* modelRevision;
};

template <std::size_t Count>
RoleTables role(const SeedVrAssetPin (&sources)[2],
               const std::uint64_t (&headers)[2],
               const std::array<prepare_detail::Plan, Count>& plans,
               const char* sourceCommit, const char* modelRevision) {
  return {sources, headers, plans.data(), Count, sourceCommit, modelRevision};
}

template <std::size_t Count>
RoleTables role(const SeedVrAssetPin (&sources)[2],
               const std::uint64_t (&headers)[2],
               const prepare_detail::Plan (&plans)[Count],
               const char* sourceCommit, const char* modelRevision) {
  return {sources, headers, plans, Count, sourceCommit, modelRevision};
}

using RoleSet = std::array<RoleTables, 7>;

RoleSet rolesFor(FlashMode mode) {
  // Keep each table explicit.  The same basenames are intentionally pinned
  // differently for Turbo and Base, so selecting a mode must happen before
  // any source-directory access.
  switch (mode) {
    case FlashMode::turbo:
      return {
          role(kFlashPreTurboSources, kFlashPreTurboHeaderBytes,
               kFlashPreTurboPlans, kFlashPreTurboSourceCommit,
               kFlashPreTurboModelRevision),
          role(kFlashPostTurboSources, kFlashPostTurboHeaderBytes,
               kFlashPostTurboPlans, kFlashPostTurboSourceCommit,
               kFlashPostTurboModelRevision),
          role(kFlashCrossTurboSources, kFlashCrossTurboHeaderBytes,
               kFlashCrossTurboPlans, kFlashCrossTurboSourceCommit,
               kFlashCrossTurboModelRevision),
          role(kFlashLqTurboSources, kFlashLqTurboHeaderBytes,
               kFlashLqTurboPlans, kFlashLqTurboSourceCommit,
               kFlashLqTurboModelRevision),
          role(kFlashEdgesPrefixTurboSources,
               kFlashEdgesPrefixTurboHeaderBytes, kFlashEdgesPrefixTurboPlans,
               kFlashEdgesPrefixTurboSourceCommit,
               kFlashEdgesPrefixTurboModelRevision),
          role(kFlashEdgesSuffixTurboSources,
               kFlashEdgesSuffixTurboHeaderBytes, kFlashEdgesSuffixTurboPlans,
               kFlashEdgesSuffixTurboSourceCommit,
               kFlashEdgesSuffixTurboModelRevision),
          role(kFlashTextTurboSources, kFlashTextTurboHeaderBytes,
               kFlashTextTurboPlans, kFlashTextTurboSourceCommit,
               kFlashTextTurboModelRevision),
      };
    case FlashMode::base:
      return {
          role(kFlashPreBaseSources, kFlashPreBaseHeaderBytes,
               kFlashPreBasePlans, kFlashPreBaseSourceCommit,
               kFlashPreBaseModelRevision),
          role(kFlashPostBaseSources, kFlashPostBaseHeaderBytes,
               kFlashPostBasePlans, kFlashPostBaseSourceCommit,
               kFlashPostBaseModelRevision),
          role(kFlashCrossBaseSources, kFlashCrossBaseHeaderBytes,
               kFlashCrossBasePlans, kFlashCrossBaseSourceCommit,
               kFlashCrossBaseModelRevision),
          role(kFlashLqBaseSources, kFlashLqBaseHeaderBytes,
               kFlashLqBasePlans, kFlashLqBaseSourceCommit,
               kFlashLqBaseModelRevision),
          role(kFlashEdgesPrefixBaseSources,
               kFlashEdgesPrefixBaseHeaderBytes, kFlashEdgesPrefixBasePlans,
               kFlashEdgesPrefixBaseSourceCommit,
               kFlashEdgesPrefixBaseModelRevision),
          role(kFlashEdgesSuffixBaseSources,
               kFlashEdgesSuffixBaseHeaderBytes, kFlashEdgesSuffixBasePlans,
               kFlashEdgesSuffixBaseSourceCommit,
               kFlashEdgesSuffixBaseModelRevision),
          role(kFlashTextBaseSources, kFlashTextBaseHeaderBytes,
               kFlashTextBasePlans, kFlashTextBaseSourceCommit,
               kFlashTextBaseModelRevision),
      };
  }
  throw std::invalid_argument("unsupported Flash mode");
}

bool samePin(const SeedVrAssetPin& left, const SeedVrAssetPin& right) {
  return left.name == right.name && left.bytes == right.bytes &&
         left.sha256 == right.sha256;
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void validateRoles(const RoleSet& roles) {
  constexpr std::array<std::size_t, 7> expected = {30, 30, 30, 3, 1, 1, 1};
  std::size_t total = 0;
  const auto& reference = roles.front();
  for (std::size_t roleIndex = 0; roleIndex < roles.size(); ++roleIndex) {
    const auto& current = roles[roleIndex];
    require(current.sources && current.headers && current.plans,
            "compiled Flash role is incomplete");
    require(current.planCount == expected[roleIndex],
            "compiled Flash role plan count differs");
    require(std::string(current.sourceCommit) == reference.sourceCommit &&
                std::string(current.modelRevision) == reference.modelRevision,
            "compiled Flash source identity differs between roles");
    for (std::size_t source = 0; source < 2; ++source) {
      require(samePin(current.sources[source], reference.sources[source]) &&
                  current.headers[source] == reference.headers[source],
              "compiled Flash source/header pin differs between roles");
    }
    for (std::size_t i = 0; i < current.planCount; ++i) {
      require(!current.plans[i].bin.name.empty(),
              "compiled Flash BIN has no name");
      for (std::size_t j = 0; j < i; ++j)
        require(current.plans[i].bin.name != current.plans[j].bin.name,
                "duplicate compiled Flash BIN name");
    }
    total += current.planCount;
  }
  require(total == kFlashLearnedPlanCount,
          "compiled Flash plan table total differs");
  // The nested comparison catches collisions between role arrays before the
  // preparer opens either source or destination directory.
  for (std::size_t i = 0; i < roles.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      for (std::size_t left = 0; left < roles[i].planCount; ++left)
        for (std::size_t right = 0; right < roles[j].planCount; ++right)
          require(roles[i].plans[left].bin.name !=
                      roles[j].plans[right].bin.name,
                  "duplicate compiled Flash BIN name across roles");
}

std::array<prepare_detail::Plan, kFlashLearnedPlanCount> flatten(
    const RoleSet& roles) {
  std::array<prepare_detail::Plan, kFlashLearnedPlanCount> result{};
  std::size_t offset = 0;
  for (const auto& current : roles) {
    std::copy(current.plans, current.plans + current.planCount,
              result.begin() + offset);
    offset += current.planCount;
  }
  require(offset == result.size(), "compiled Flash plan flatten overflow");
  return result;
}

const SeedVrAssetPin kFlashEmptyComplement = {
    "empty-complement", 0ULL,
    {0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4,
     0xc8, 0x99, 0x6f, 0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b,
     0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55}};

const prepare_detail::Segment kFlashPromptSegments[] = {
    {0ULL, kFlashFixedPrompt.raw.bytes, kFlashFixedPrompt.rawOffset, 0U,
     true, kFlashFixedPrompt.raw.sha256},
};

const prepare_detail::Plan kFlashPromptPlan = {
    kFlashFixedPrompt.raw, kFlashEmptyComplement, kFlashPromptSegments, 1U};

void addChecked(std::uint64_t& total, std::uint64_t bytes,
                const char* message) {
  require(bytes <= std::numeric_limits<std::uint64_t>::max() - total, message);
  total += bytes;
}

void appendUnique(std::vector<SeedVrAssetPin>& pins, const SeedVrAssetPin& pin) {
  for (const auto& existing : pins)
    if (samePin(existing, pin)) return;
  pins.push_back(pin);
}

std::vector<SeedVrAssetPin> structuralXmls() {
  return {kFlashPreXml,
          kFlashPostXml,
          kFlashCrossXml,
          kFlashLqStartupXml,
          kFlashLqPrefixXml,
          kFlashLqSuffixXml,
          kFlashEdgesPrefixXml,
          kFlashEdgesSuffixXml,
          kFlashTextXml,
          kFlashDecoderFrontXml,
          kFlashDecoderGrowXml,
          kFlashDecoderFrameXml};
}

std::vector<SeedVrAssetPin> structuralBins() {
  return {kFlashPreStructuralBin,
          kFlashPostStructuralBin,
          kFlashCrossStructuralBin,
          kFlashLqStartupStructuralBin,
          kFlashLqPrefixStructuralBin,
          kFlashLqSuffixStructuralBin,
          kFlashEdgesPrefixStructuralBin,
          kFlashEdgesSuffixStructuralBin,
          kFlashTextStructuralBin,
          kFlashDecoderFrontStructuralBin,
          kFlashDecoderGrowStructuralBin,
          kFlashDecoderFrameStructuralBin};
}

std::vector<SeedVrAssetPin> structuralComplements() {
  auto result = structuralBins();
  appendUnique(result, kFlashEmptyComplement);
  return result;
}

std::vector<SeedVrAssetPin> fixedStructuralAssets() {
  return {kFlashSplitKeyConfig,
          kFlashPostConfig,
          kFlashPostBridgeSource,
          kFlashPostSparseSource};
}

std::vector<SeedVrAssetPin> modePlans(const RoleSet& roles) {
  std::vector<SeedVrAssetPin> result;
  for (const auto& current : roles)
    for (std::size_t i = 0; i < current.planCount; ++i)
      appendUnique(result, current.plans[i].bin);
  return result;
}

std::vector<SeedVrAssetPin> preparedPlans(const RoleSet& roles) {
  auto result = modePlans(roles);
  for (const auto& plan : kFlashDecoderPlans) appendUnique(result, plan.bin);
  appendUnique(result, kFlashPromptPlan.bin);
  return result;
}

std::uint64_t sumPins(const std::vector<SeedVrAssetPin>& pins) {
  std::uint64_t total = 0;
  for (const auto& pin : pins) addChecked(total, pin.bytes, "Flash verification total overflow");
  return total;
}

std::uint64_t sumPlans(const prepare_detail::Plan* plans, std::size_t count,
                       bool includeComplements) {
  std::uint64_t total = 0;
  require(plans && count > 0, "Flash plan inventory is empty");
  for (std::size_t i = 0; i < count; ++i) {
    addChecked(total, plans[i].bin.bytes, "Flash preparation total overflow");
    if (includeComplements)
      addChecked(total, plans[i].complement.bytes,
                 "Flash preparation total overflow");
  }
  return total;
}

#ifdef __linux__
void verifyEmptyFile(const std::filesystem::path& directory,
                     const SeedVrAssetPin& pin) {
  const int parent = open(directory.c_str(), O_RDONLY | O_DIRECTORY |
                                             O_NOFOLLOW | O_CLOEXEC);
  require(parent >= 0, "Flash empty complement directory unavailable");
  const int file = openat(parent, pin.name.c_str(), O_RDONLY | O_NOFOLLOW |
                                                   O_CLOEXEC);
  struct stat status{};
  const bool valid = file >= 0 && fstat(file, &status) == 0 &&
                     S_ISREG(status.st_mode) && status.st_size == 0;
  if (file >= 0) close(file);
  close(parent);
  require(valid, "Flash fixed empty complement is not an exact regular file");
}
#else
void verifyEmptyFile(const std::filesystem::path&, const SeedVrAssetPin&) {
  throw std::runtime_error("Flash preparation verification requires Linux file authentication");
}
#endif

void notifyVerification(const SeedVrPrepareCallback& callback,
                        const SeedVrAssetPin& pin, std::uint64_t done,
                        std::uint64_t total) {
  if (callback && !callback({SeedVrPrepareStage::authenticate, pin.name, done,
                             total}))
    throw SeedVrPrepareCancelled();
}

void verifyPinned(const std::filesystem::path& directory,
                  const SeedVrAssetPin& pin, std::uint64_t& done,
                  const SeedVrPrepareCallback& callback,
                  bool finiteBf16 = false) {
  std::uint64_t pinDone = 0;
  notifyVerification(callback, pin, pinDone, pin.bytes);
  if (pin.bytes == 0) {
    // Only the two compiled zero-byte pins are accepted here.  In particular,
    // this is not a general relaxation of the positive asset verifier.
    require(pin.name == kFlashTextStructuralBin.name ||
                pin.name == kFlashEmptyComplement.name,
            "unexpected zero-byte Flash asset pin");
    verifyEmptyFile(directory, pin);
  } else if (finiteBf16) {
    (void)readBf16Asset(directory, pin);
  } else if (pin.name.size() >= 4 &&
             pin.name.compare(pin.name.size() - 4, 4, ".xml") == 0) {
    (void)readSeedVrXml(directory, pin);
  } else {
    verifySeedVrAsset(directory, pin);
  }
  addChecked(pinDone, pin.bytes, "Flash verification progress overflow");
  addChecked(done, pin.bytes, "Flash verification progress overflow");
  notifyVerification(callback, pin, pinDone, pin.bytes);
}

void verifyList(const std::filesystem::path& directory,
                const std::vector<SeedVrAssetPin>& pins, std::uint64_t& done,
                const SeedVrPrepareCallback& callback) {
  for (const auto& pin : pins) verifyPinned(directory, pin, done, callback);
}

void verifyInstalled(const std::filesystem::path& models, FlashMode mode,
                     bool prepared, std::uint64_t& done,
                     const SeedVrPrepareCallback& callback) {
  verifyList(models, structuralXmls(), done, callback);
  verifyList(models, structuralBins(), done, callback);
  verifyList(models, fixedStructuralAssets(), done, callback);
  const auto index = mode == FlashMode::turbo ? 0U : 1U;
  verifyPinned(models, kFlashFixedAssets.at(index).bin, done, callback,
               true);
  if (prepared) {
    const auto roles = rolesFor(mode);
    validateRoles(roles);
    verifyList(models, preparedPlans(roles), done, callback);
  }
}

}  // namespace

#ifdef OTC_FLASH_PREPARE_TESTING
bool testFlashPrepareVerificationProgress(
    const std::filesystem::path& fixtureDirectory) {
  const std::vector<SeedVrAssetPin> pins = {kFlashTextXml, kFlashCrossXml};
  std::uint64_t done = 0;
  std::vector<SeedVrPrepareProgress> records;
  verifyList(fixtureDirectory, pins, done,
             [&records](const SeedVrPrepareProgress& progress) {
               records.push_back(progress);
               return true;
             });
  if (done != pins[0].bytes + pins[1].bytes || records.empty()) return false;
  std::size_t pin = 0;
  std::uint64_t previous = 0;
  for (const auto& record : records) {
    if (pin >= pins.size() || record.name != pins[pin].name) return false;
    if (record.total != pins[pin].bytes || record.done < previous ||
        record.done > record.total)
      return false;
    if (record.done == 0 && previous != 0) return false;
    if (record.done == record.total) {
      if (pin + 1 == pins.size()) {
        previous = record.done;
      } else {
        ++pin;
        previous = 0;
      }
    } else {
      previous = record.done;
    }
  }
  return pin + 1 == pins.size() && previous == pins.back().bytes;
}
#endif

void prepareFlashMode(const std::filesystem::path& sourceDirectory,
                      const std::filesystem::path& complementDirectory,
                      const std::filesystem::path& managedDestination,
                      FlashMode mode,
                      const SeedVrPrepareCallback& progress) {
  // rolesFor() and validation intentionally precede every filesystem call.
  const auto roles = rolesFor(mode);
  validateRoles(roles);
  const auto plans = flatten(roles);
  const SeedVrAssetPin sources[2] = {roles.front().sources[0],
                                     roles.front().sources[1]};
  const std::uint64_t headers[2] = {roles.front().headers[0],
                                   roles.front().headers[1]};
  prepare_detail::prepare(sourceDirectory, complementDirectory,
                          managedDestination, sources, headers, plans.data(),
                          plans.size(), progress);
}

void prepareFlashDecoder(const std::filesystem::path& sourceDirectory,
                         const std::filesystem::path& complementDirectory,
                         const std::filesystem::path& managedDestination,
                         const SeedVrPrepareCallback& progress) {
  prepare_detail::prepare(
      sourceDirectory, complementDirectory, managedDestination,
      kFlashDecoderSource, kFlashDecoderHeaderBytes, kFlashDecoderPlans,
      std::size(kFlashDecoderPlans), progress);
}

void prepareFlashPrompt(const std::filesystem::path& sourceDirectory,
                        const std::filesystem::path& complementDirectory,
                        const std::filesystem::path& managedDestination,
                        const SeedVrPrepareCallback& progress) {
  prepare_detail::prepare(sourceDirectory, complementDirectory,
                          managedDestination, kFlashFixedPrompt.outer, 0ULL,
                          &kFlashPromptPlan, 1U, progress);
}

FlashPrepareTotals flashPrepareTotals(FlashMode mode) {
  const auto roles = rolesFor(mode);
  validateRoles(roles);
  const auto prepared = preparedPlans(roles);
  FlashPrepareTotals result;
  result.structural = sumPins(structuralXmls());
  addChecked(result.structural, sumPins(structuralBins()),
             "Flash structural total overflow");
  addChecked(result.structural, sumPins(fixedStructuralAssets()),
             "Flash structural total overflow");
  addChecked(result.structural,
             kFlashFixedAssets.at(mode == FlashMode::turbo ? 0U : 1U).bin.bytes,
             "Flash structural total overflow");
  addChecked(result.structural, sumPins(structuralComplements()),
             "Flash structural total overflow");
  addChecked(result.mode, roles.front().sources[0].bytes,
             "Flash mode total overflow");
  addChecked(result.mode, roles.front().sources[1].bytes,
             "Flash mode total overflow");
  addChecked(result.mode, sumPlans(roles.front().plans, roles.front().planCount, true),
             "Flash mode total overflow");
  for (std::size_t i = 1; i < roles.size(); ++i)
    addChecked(result.mode, sumPlans(roles[i].plans, roles[i].planCount, true),
               "Flash mode total overflow");
  addChecked(result.decoder, kFlashDecoderSource.bytes,
             "Flash decoder total overflow");
  addChecked(result.decoder,
             sumPlans(kFlashDecoderPlans, std::size(kFlashDecoderPlans), true),
             "Flash decoder total overflow");
  addChecked(result.prompt, kFlashFixedPrompt.outer.bytes,
             "Flash prompt total overflow");
  addChecked(result.prompt, kFlashPromptPlan.bin.bytes,
             "Flash prompt total overflow");
  result.prepared = sumPins(structuralXmls());
  addChecked(result.prepared, sumPins(structuralBins()),
             "Flash prepared total overflow");
  addChecked(result.prepared, sumPins(fixedStructuralAssets()),
             "Flash prepared total overflow");
  addChecked(result.prepared,
             kFlashFixedAssets.at(mode == FlashMode::turbo ? 0U : 1U).bin.bytes,
             "Flash prepared total overflow");
  addChecked(result.prepared, sumPins(prepared), "Flash prepared total overflow");
  return result;
}

void verifyFlashStructuralAssets(const std::filesystem::path& models,
                                 const std::filesystem::path& complements,
                                 FlashMode mode,
                                 const SeedVrPrepareCallback& progress) {
  const auto totals = flashPrepareTotals(mode);
  std::uint64_t done = 0;
  verifyInstalled(models, mode, false, done, progress);
  verifyList(complements, structuralComplements(), done, progress);
  require(done == totals.structural, "Flash structural verification total differs");
}

void verifyFlashPreparedAssets(const std::filesystem::path& models,
                               FlashMode mode,
                               const SeedVrPrepareCallback& progress) {
  const auto totals = flashPrepareTotals(mode);
  const auto modelTotal = totals.prepared;
  std::uint64_t done = 0;
  verifyInstalled(models, mode, true, done, progress);
  require(done == modelTotal, "Flash prepared verification total differs");
}

}  // namespace otc::ai
