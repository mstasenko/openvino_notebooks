#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>

namespace otc::ai {
enum class SeedVrPrepareStage { authenticate, reconstruct, published };
struct SeedVrPrepareProgress {
  SeedVrPrepareStage stage;
  std::string name;
  std::uint64_t done;
  std::uint64_t total;
};
// Called synchronously, at most one MiB of IO between cancellation polls.
// Return false to cancel. The callback must not modify source/destination
// files.
using SeedVrPrepareCallback = std::function<bool(const SeedVrPrepareProgress&)>;
class SeedVrPrepareCancelled final : public std::runtime_error {
 public:
  SeedVrPrepareCancelled()
      : std::runtime_error("SeedVR2 preparation cancelled") {}
};
// CPU-only. Authority comes exclusively from compiled pins, not caller
// manifests. Directories must be managed by the caller, destination private and
// already created. Each BIN is verified before atomic no-overwrite publication.
// This is not a whole-install transaction: failures leave earlier verified BINs
// intact. The caller installs/authenticates structural XML/text separately and
// declares readiness only after every required asset has succeeded. No source
// is removed.
void prepareSeedVrIr(const std::filesystem::path& sourceDirectory,
                     const std::filesystem::path& complementDirectory,
                     const std::filesystem::path& managedDestination,
                     const SeedVrPrepareCallback& progress = {});
}  // namespace otc::ai
