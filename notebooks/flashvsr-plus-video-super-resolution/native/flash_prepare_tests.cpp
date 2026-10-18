#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "flash-assets/flash_cross_assets.hpp"
#include "flash-assets/flash_decoder_assets.hpp"
#include "flash-assets/flash_edges_prefix_assets.hpp"
#include "flash-assets/flash_edges_suffix_assets.hpp"
#include "flash-assets/flash_fixed_assets.hpp"
#include "flash-assets/flash_lq_assets.hpp"
#include "flash-assets/flash_post_assets.hpp"
#include "flash-assets/flash_pre_assets.hpp"
#include "flash-assets/flash_text_assets.hpp"
#include "flash_prepare.hpp"

using namespace otc::ai;
using otc::ai::prepare_detail::Plan;

namespace {
struct Role {
  const SeedVrAssetPin* sources;
  const std::uint64_t* headers;
  const Plan* plans;
  std::size_t count;
  const char* commit;
  const char* revision;
};

bool samePin(const SeedVrAssetPin& a, const SeedVrAssetPin& b) {
  return a.name == b.name && a.bytes == b.bytes && a.sha256 == b.sha256;
}

void checkMode(const Role (&roles)[7]) {
  constexpr std::size_t counts[] = {30, 30, 30, 3, 1, 1, 1};
  constexpr std::uint64_t expectedBytes = 3493228156ULL;
  std::uint64_t bytes = 0;
  for (std::size_t role = 0; role < 7; ++role) {
    assert(roles[role].count == counts[role]);
    assert(std::string(roles[role].commit) == roles[0].commit);
    assert(std::string(roles[role].revision) == roles[0].revision);
    for (unsigned source = 0; source < 2; ++source) {
      assert(samePin(roles[role].sources[source], roles[0].sources[source]));
      assert(roles[role].headers[source] == roles[0].headers[source]);
    }
    for (std::size_t i = 0; i < roles[role].count; ++i) {
      assert(!roles[role].plans[i].bin.name.empty());
      bytes += roles[role].plans[i].bin.bytes;
      for (std::size_t j = 0; j < i; ++j)
        assert(roles[role].plans[i].bin.name !=
               roles[role].plans[j].bin.name);
      for (std::size_t other = 0; other < role; ++other)
        for (std::size_t j = 0; j < roles[other].count; ++j)
          assert(roles[role].plans[i].bin.name !=
                 roles[other].plans[j].bin.name);
    }
  }
  assert(bytes == expectedBytes);
}
}  // namespace

int main() {
  const Role turbo[] = {
      {kFlashPreTurboSources, kFlashPreTurboHeaderBytes,
       kFlashPreTurboPlans.data(), kFlashPreTurboPlans.size(),
       kFlashPreTurboSourceCommit, kFlashPreTurboModelRevision},
      {kFlashPostTurboSources, kFlashPostTurboHeaderBytes,
       kFlashPostTurboPlans.data(), kFlashPostTurboPlans.size(),
       kFlashPostTurboSourceCommit, kFlashPostTurboModelRevision},
      {kFlashCrossTurboSources, kFlashCrossTurboHeaderBytes,
       kFlashCrossTurboPlans.data(), kFlashCrossTurboPlans.size(),
       kFlashCrossTurboSourceCommit, kFlashCrossTurboModelRevision},
      {kFlashLqTurboSources, kFlashLqTurboHeaderBytes,
       kFlashLqTurboPlans.data(), kFlashLqTurboPlans.size(),
       kFlashLqTurboSourceCommit, kFlashLqTurboModelRevision},
      {kFlashEdgesPrefixTurboSources, kFlashEdgesPrefixTurboHeaderBytes,
       kFlashEdgesPrefixTurboPlans.data(), kFlashEdgesPrefixTurboPlans.size(),
       kFlashEdgesPrefixTurboSourceCommit,
       kFlashEdgesPrefixTurboModelRevision},
      {kFlashEdgesSuffixTurboSources, kFlashEdgesSuffixTurboHeaderBytes,
       kFlashEdgesSuffixTurboPlans.data(), kFlashEdgesSuffixTurboPlans.size(),
       kFlashEdgesSuffixTurboSourceCommit,
       kFlashEdgesSuffixTurboModelRevision},
      {kFlashTextTurboSources, kFlashTextTurboHeaderBytes,
       kFlashTextTurboPlans.data(), kFlashTextTurboPlans.size(),
       kFlashTextTurboSourceCommit, kFlashTextTurboModelRevision},
  };
  const Role base[] = {
      {kFlashPreBaseSources, kFlashPreBaseHeaderBytes,
       kFlashPreBasePlans.data(), kFlashPreBasePlans.size(),
       kFlashPreBaseSourceCommit, kFlashPreBaseModelRevision},
      {kFlashPostBaseSources, kFlashPostBaseHeaderBytes,
       kFlashPostBasePlans.data(), kFlashPostBasePlans.size(),
       kFlashPostBaseSourceCommit, kFlashPostBaseModelRevision},
      {kFlashCrossBaseSources, kFlashCrossBaseHeaderBytes,
       kFlashCrossBasePlans.data(), kFlashCrossBasePlans.size(),
       kFlashCrossBaseSourceCommit, kFlashCrossBaseModelRevision},
      {kFlashLqBaseSources, kFlashLqBaseHeaderBytes,
       kFlashLqBasePlans.data(), kFlashLqBasePlans.size(),
       kFlashLqBaseSourceCommit, kFlashLqBaseModelRevision},
      {kFlashEdgesPrefixBaseSources, kFlashEdgesPrefixBaseHeaderBytes,
       kFlashEdgesPrefixBasePlans.data(), kFlashEdgesPrefixBasePlans.size(),
       kFlashEdgesPrefixBaseSourceCommit,
       kFlashEdgesPrefixBaseModelRevision},
      {kFlashEdgesSuffixBaseSources, kFlashEdgesSuffixBaseHeaderBytes,
       kFlashEdgesSuffixBasePlans.data(), kFlashEdgesSuffixBasePlans.size(),
       kFlashEdgesSuffixBaseSourceCommit,
       kFlashEdgesSuffixBaseModelRevision},
      {kFlashTextBaseSources, kFlashTextBaseHeaderBytes,
       kFlashTextBasePlans.data(), kFlashTextBasePlans.size(),
       kFlashTextBaseSourceCommit, kFlashTextBaseModelRevision},
  };
  checkMode(turbo);
  checkMode(base);
  // Turbo and Base intentionally share basenames but never their learned
  // source hashes; selecting a mode must not accidentally reuse the other.
  assert(!samePin(kFlashPreTurboSources[0], kFlashPreBaseSources[0]));
  assert(!samePin(kFlashPreTurboSources[1], kFlashPreBaseSources[1]));
  assert(kFlashLearnedPlanCount == 96U);
  assert(kFlashDecoderPlans[0].count == 42U &&
         kFlashDecoderPlans[1].count == 22U &&
         kFlashDecoderPlans[2].count == 7U);
  assert(kFlashFixedPrompt.rawOffset == 448ULL &&
         kFlashFixedPrompt.raw.bytes == 4194304ULL);

  bool rejected = false;
  try {
    prepareFlashMode("/does-not-exist", "/does-not-exist", "/does-not-exist",
                     static_cast<FlashMode>(99));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  std::cout << "PASS 96 Flash plans, source/header identity, decoder/prompt pins, "
               "invalid mode before I/O\n";
}
