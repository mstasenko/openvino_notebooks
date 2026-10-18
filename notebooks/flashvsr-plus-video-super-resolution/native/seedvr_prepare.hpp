#pragma once
#include <cstdint>
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
}  // namespace otc::ai
