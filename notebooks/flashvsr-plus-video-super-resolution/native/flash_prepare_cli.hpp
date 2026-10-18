#pragma once

#include <csignal>
#include <iosfwd>
#include <optional>

namespace otc::ai {

// Dispatch before OpenVINO/Core construction or device discovery.  A nullopt
// result means the argv is not a Flash preparation command; action commands
// return 0, 1, or 130 just like the SeedVR preparation seam.
std::optional<int> runFlashPrepareCli(
    int argc, char** argv, volatile std::sig_atomic_t& stopRequested,
    std::ostream& progress, std::ostream& errors);

}  // namespace otc::ai
