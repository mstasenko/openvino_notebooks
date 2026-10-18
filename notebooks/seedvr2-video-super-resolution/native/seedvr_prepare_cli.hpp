#pragma once
#include <csignal>
#include <iosfwd>
#include <optional>

namespace otc::ai {
// Dispatch before renderer options, Core construction or device enumeration.
// nullopt means an ordinary export invocation; action exits are 0/1/130.
std::optional<int> runSeedVrPrepareCli(
    int argc, char** argv, volatile std::sig_atomic_t& stopRequested,
    std::ostream& progress, std::ostream& errors);
}  // namespace otc::ai
