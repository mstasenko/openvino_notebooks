#include <filesystem>
#include <sstream>
#include <vector>

#define main workerMain
#include "main.cpp"
#undef main
#include <cassert>

using namespace otc::ai;

struct RunResult {
  int exit;
  std::string rgb;
  std::string error;
};

RunResult runWorker(const std::string& raw, const char* mode = "temporal") {
  std::vector<std::string> args{"seedvr2-openvino", "--width", "2", "--height", "1",
      "--fps", "3", "--output-width", "2", "--output-height", "1",
      "--output-fps", "3", "--duration", "1", "--device", "CPU", "--mode", mode};
  std::vector<char*> argv;
  for (auto& argument : args) argv.push_back(argument.data());
  std::istringstream input(raw);
  std::ostringstream output;
  std::ostringstream error;
  auto* oldInput = std::cin.rdbuf(input.rdbuf());
  auto* oldOutput = std::cout.rdbuf(output.rdbuf());
  auto* oldError = std::cerr.rdbuf(error.rdbuf());
  std::cin.clear();
  stopRequested = 0;
  const int exit = workerMain(static_cast<int>(argv.size()), argv.data());
  std::cin.rdbuf(oldInput);
  std::cout.rdbuf(oldOutput);
  std::cerr.rdbuf(oldError);
  std::cin.clear();
  return {exit, output.str(), error.str()};
}

int main() {
  assert(canonicalDiscoveredDevice("GPU") == "GPU.0");
  for (const char* device : {"CPU", "GPU.0", "GPU.1", "GPU.12"}) {
    assert(canonicalDiscoveredDevice(device) == device);
  }
  for (const char* device : {"", "AUTO", "MULTI:CPU,GPU", "GPU.", "GPU.x", "GPU.-1",
                             "GPU.0.1", "GPU.0,CPU", "GPU.0 ", "gpu"}) {
    bool rejected = false;
    try { (void)canonicalDiscoveredDevice(device); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  bool aliasRejectedAtFrameBoundary = false;
  try { seedVrValidateDevice("GPU"); }
  catch (const std::invalid_argument&) { aliasRejectedAtFrameBoundary = true; }
  assert(aliasRejectedAtFrameBoundary);

  // Avoid changing stream buffers on workerMain's first sync_with_stdio call.
  std::ios::sync_with_stdio(false);
  const std::string raw("abcdefghijklmnopqr", 18);
  for (const auto* mode : {"temporal", "per-frame"}) {
    const auto result = runWorker(raw + "stuvwx", mode);
    assert(result.exit == 0 && result.rgb == raw);
    assert(result.error.find("device=none") != std::string::npos);
  }
  auto result = runWorker("");
  assert(result.exit == 1 && result.rgb.empty());
  result = runWorker(raw.substr(0, 12));
  assert(result.exit == 1 && result.error.find("source ended") != std::string::npos);
  result = runWorker(raw.substr(0, 14));
  assert(result.exit == 1 && result.error.find("partial RGB24") != std::string::npos);
  result = runWorker(raw, "single-frame");
  assert(result.exit == 1 && result.rgb.empty()); // TS spelling is per-frame

  // Actual source window receives enhanced originals only; no interpolated
  // frame is sent back through the source provider's renderer.
  Options options;
  options.fps = 3;
  int reads = 0;
  int renders = 0;
  SeedVrSourceProvider provider(3, false,
      [&]() -> std::optional<SourceFrame> {
        if (reads == 3) return std::nullopt;
        const auto index = reads++;
        return SourceFrame{static_cast<std::uint64_t>(index), {2, 1, {1, 2, 3, 4, 5, 6}}};
      },
      [&](const std::vector<RgbFrame>& inputs) {
        ++renders;
        return inputs;
      });
  SourceWindow window(options, provider);
  assert(window.initialize());
  for (int index = 0; index < 6; ++index) {
    window.ensureRight(index / 6.0);
    assert(window.left);
    if (window.right) {
      (void)processBaseFrame(window.left->frame, window.right->frame, 0.25,
                            false, nullptr, options);
    }
  }
  assert(reads == 3 && renders == 3);

  for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
    bool rejected = false;
    try { (void)floatToByte(invalid); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
  assert(floatToByte(0.0F) == 0 && floatToByte(1.0F) == 255);

  // Both RIFE and batch catches use this actual policy before activateNext.
  options.device = "AUTO";
  stopRequested = 0;
  assert(mayRetryDevice(options));
  stopRequested = 1;
  assert(!mayRetryDevice(options));
  int replacements = 0;
  bool rejected = false;
  const std::vector<RgbFrame> batch{{2, 1, {1, 2, 3, 4, 5, 6}}};
  try {
    (void)renderSourceBatchWithRetry(batch,
        [](const std::vector<RgbFrame>&) -> std::vector<RgbFrame> {
          throw std::runtime_error("cancelled inference");
        },
        [&](const std::exception&) {
          if (!mayRetryDevice(options)) return false;
          ++replacements;
          return true;
        });
  } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected && replacements == 0);
  stopRequested = 0;
  options.device = "CPU";
  assert(!mayRetryDevice(options));
}
