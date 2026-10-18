#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "flash_mode.hpp"
#include "seedvr_prepare_detail.hpp"

namespace otc::ai {

inline constexpr std::size_t kFlashLearnedPlanCount = 96U;
static_assert(kFlashLearnedPlanCount <= prepare_detail::kMaxPreparedPlanCount,
              "Flash role table exceeds the bounded native preparer");

struct FlashPrepareTotals {
  std::uint64_t structural = 0;
  std::uint64_t mode = 0;
  std::uint64_t decoder = 0;
  std::uint64_t prompt = 0;
  std::uint64_t prepared = 0;
};

// Return byte-work totals for the fixed compiled inventory.  This is used by
// the CLI only to keep one monotonic progress counter across the three
// preparation calls; it does not inspect caller paths.
FlashPrepareTotals flashPrepareTotals(FlashMode mode);

// Prepare the learned PRE/POST/Cross/LQ/edge/Text BINs for one mode.  The
// compiled role tables are authenticated as a single source pair and then
// passed to the existing atomic preparer in one bounded operation.  XML,
// kernels, fixed conditioning, and other small assets are installed and
// checked by their owning callers; this function only prepares learned BINs.
void prepareFlashMode(const std::filesystem::path& sourceDirectory,
                      const std::filesystem::path& complementDirectory,
                      const std::filesystem::path& managedDestination,
                      FlashMode mode,
                      const SeedVrPrepareCallback& progress = {});

// The decoder and prompt use the existing one-source preparation overload.
// The prompt's authenticated empty complement is the fixed basename
// "empty-complement" in complementDirectory.
void prepareFlashDecoder(const std::filesystem::path& sourceDirectory,
                         const std::filesystem::path& complementDirectory,
                         const std::filesystem::path& managedDestination,
                         const SeedVrPrepareCallback& progress = {});
void prepareFlashPrompt(const std::filesystem::path& sourceDirectory,
                        const std::filesystem::path& complementDirectory,
                        const std::filesystem::path& managedDestination,
                        const SeedVrPrepareCallback& progress = {});

// Verify the installed structural pack and its authenticated complements.
// No learned source is opened.  The ordinary installed verification also
// checks the 96 mode BINs, the three decoder BINs, and the prompt BIN.
void verifyFlashStructuralAssets(const std::filesystem::path& models,
                                 const std::filesystem::path& complements,
                                 FlashMode mode,
                                 const SeedVrPrepareCallback& progress = {});
void verifyFlashPreparedAssets(const std::filesystem::path& models,
                               FlashMode mode,
                               const SeedVrPrepareCallback& progress = {});

}  // namespace otc::ai
