#include "flash_prepare_cli.hpp"

#include "flash_prepare.hpp"

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace otc::ai {
namespace {

struct Options {
  bool prepare = false;
  bool structuralOnly = false;
  FlashMode mode = FlashMode::turbo;
  std::filesystem::path models;
  std::filesystem::path sources;
  std::filesystem::path complements;
};

FlashMode parseMode(std::string_view value) {
  if (value == "turbo") return FlashMode::turbo;
  if (value == "base") return FlashMode::base;
  throw std::invalid_argument("--mode must be turbo or base");
}

const char* modeName(FlashMode mode) {
  return mode == FlashMode::turbo ? "turbo" : "base";
}

Options parse(int argc, char** argv) {
  if (argc < 2 || !argv || !argv[1]) throw std::invalid_argument("missing Flash action");
  const std::string_view action(argv[1]);
  if (action != "--prepare-models" && action != "--verify-models")
    throw std::invalid_argument("unknown Flash preparation action");

  Options result;
  result.prepare = action == "--prepare-models";
  std::set<std::string> seen;
  bool haveMode = false;
  for (int index = 2; index < argc; ++index) {
    if (!argv[index]) throw std::invalid_argument("missing Flash preparation option");
    const std::string option(argv[index]);
    if (!seen.insert(option).second)
      throw std::invalid_argument("duplicate Flash preparation option");
    if (option == "--structural-only") {
      if (result.prepare)
        throw std::invalid_argument("--structural-only is verify-only");
      result.structuralOnly = true;
      continue;
    }
    if (option != "--mode" && option != "--models" && option != "--sources" &&
        option != "--complements")
      throw std::invalid_argument("unknown Flash preparation option");
    if (++index >= argc || !argv[index] || !argv[index][0] ||
        std::string_view(argv[index]).substr(0, 2) == "--")
      throw std::invalid_argument("missing Flash preparation option value");
    const std::string value(argv[index]);
    if (option == "--mode") {
      result.mode = parseMode(value);
      haveMode = true;
    } else if (option == "--models") {
      result.models = value;
    } else if (option == "--sources") {
      result.sources = value;
    } else {
      result.complements = value;
    }
  }
  if (!haveMode || result.models.empty() ||
      (result.prepare && (result.sources.empty() || result.complements.empty())) ||
      (!result.prepare && !result.sources.empty()) ||
      (result.structuralOnly && result.complements.empty()) ||
      (!result.prepare && !result.structuralOnly && !result.complements.empty()))
    throw std::invalid_argument("incomplete or incompatible Flash preparation options");
  return result;
}

void emit(std::ostream& output, std::string_view phase, const std::string& name,
          std::uint64_t done, std::uint64_t total) {
  output << "{\"phase\":" << std::quoted(std::string(phase))
         << ",\"name\":" << std::quoted(name) << ",\"done\":" << done
         << ",\"total\":" << total << "}\n" << std::flush;
  if (!output) throw std::runtime_error("Flash preparation progress output failed");
}

class AggregateProgress final {
 public:
  AggregateProgress(std::ostream& output, volatile std::sig_atomic_t& stop,
                    std::string phase, std::uint64_t total)
      : output_(output), stop_(stop), phase_(std::move(phase)), total_(total) {}

  void begin(std::uint64_t operationTotal) {
    if (stop_) throw SeedVrPrepareCancelled();
    if (base_ > total_ || operationTotal > total_ - base_)
      throw std::runtime_error("Flash preparation progress total overflow");
    operationTotal_ = operationTotal;
    operationDone_ = 0;
    currentName_.clear();
    currentStage_ = SeedVrPrepareStage::published;
    currentDone_ = 0;
  }

  bool accept(const SeedVrPrepareProgress& value) {
    if (stop_) return false;
    const auto key = std::to_string(static_cast<int>(value.stage)) + '\n' + value.name;
    if (key != currentName_) {
      flushCurrent();
      currentName_ = key;
      currentStage_ = value.stage;
      currentDone_ = 0;
    }
    if (value.stage == SeedVrPrepareStage::published) {
      currentDone_ = 0;
      emit(output_, phase_, value.name, base_ + operationDone_, total_);
      return true;
    }
    if (value.done < currentDone_ || value.done > value.total)
      throw std::runtime_error("Flash preparation progress regressed");
    currentDone_ = value.done;
    emit(output_, phase_, value.name, base_ + operationDone_ + currentDone_, total_);
    return true;
  }

  void finish() {
    if (stop_) throw SeedVrPrepareCancelled();
    flushCurrent();
    if (operationDone_ != operationTotal_)
      throw std::runtime_error("Flash preparation progress total differs");
    base_ += operationTotal_;
  }

 private:
  void flushCurrent() {
    if (currentName_.empty() || currentStage_ == SeedVrPrepareStage::published) return;
    if (operationDone_ > operationTotal_ ||
        currentDone_ > operationTotal_ - operationDone_)
      throw std::runtime_error("Flash preparation progress overflow");
    operationDone_ += currentDone_;
    currentDone_ = 0;
  }

  std::ostream& output_;
  volatile std::sig_atomic_t& stop_;
  std::string phase_;
  std::uint64_t total_ = 0;
  std::uint64_t base_ = 0;
  std::uint64_t operationTotal_ = 0;
  std::uint64_t operationDone_ = 0;
  std::string currentName_;
  SeedVrPrepareStage currentStage_ = SeedVrPrepareStage::published;
  std::uint64_t currentDone_ = 0;
};

void stopIfRequested(volatile std::sig_atomic_t& stop) {
  if (stop) throw SeedVrPrepareCancelled();
}

std::uint64_t preparationTotal(const FlashPrepareTotals& totals) {
  std::uint64_t result = 0;
  for (const auto value : {totals.structural, totals.mode, totals.decoder,
                           totals.prompt, totals.prepared}) {
    if (value > std::numeric_limits<std::uint64_t>::max() - result)
      throw std::runtime_error("Flash preparation progress total overflow");
    result += value;
  }
  return result;
}

template <typename Function>
void runOperation(AggregateProgress& aggregate, std::uint64_t total,
                  Function&& function) {
  aggregate.begin(total);
  function([&aggregate](const SeedVrPrepareProgress& progress) {
    return aggregate.accept(progress);
  });
  aggregate.finish();
}

}  // namespace

std::optional<int> runFlashPrepareCli(int argc, char** argv,
                                      volatile std::sig_atomic_t& stop,
                                      std::ostream& progress,
                                      std::ostream& errors) {
  if (argc < 2 || !argv || !argv[1]) return std::nullopt;
  const std::string_view action(argv[1]);
  if (action != "--prepare-models" && action != "--verify-models")
    return std::nullopt;
  try {
    const auto options = parse(argc, argv);
    stopIfRequested(stop);
    const auto totals = flashPrepareTotals(options.mode);
    if (!options.prepare) {
      AggregateProgress aggregate(progress, stop, "verifying",
                                  options.structuralOnly ? totals.structural
                                                         : totals.prepared);
      if (options.structuralOnly) {
        runOperation(aggregate, totals.structural, [&](const auto& callback) {
          verifyFlashStructuralAssets(options.models, options.complements,
                                      options.mode, callback);
        });
      } else {
        runOperation(aggregate, totals.prepared, [&](const auto& callback) {
          verifyFlashPreparedAssets(options.models, options.mode, callback);
        });
      }
      return 0;
    }

    AggregateProgress aggregate(
        progress, stop, "preparing", preparationTotal(totals));
    runOperation(aggregate, totals.structural, [&](const auto& callback) {
      verifyFlashStructuralAssets(options.models, options.complements,
                                  options.mode, callback);
    });
    const auto modeSource = options.sources / modeName(options.mode);
    const auto sharedSource = options.sources / "shared";
    runOperation(aggregate, totals.mode, [&](const auto& callback) {
      prepareFlashMode(modeSource, options.complements, options.models,
                       options.mode, callback);
    });
    runOperation(aggregate, totals.decoder, [&](const auto& callback) {
      prepareFlashDecoder(sharedSource, options.complements, options.models, callback);
    });
    runOperation(aggregate, totals.prompt, [&](const auto& callback) {
      prepareFlashPrompt(sharedSource, options.complements, options.models, callback);
    });
    runOperation(aggregate, totals.prepared, [&](const auto& callback) {
      verifyFlashPreparedAssets(options.models, options.mode, callback);
    });
    return 0;
  } catch (const SeedVrPrepareCancelled&) {
    errors << "Flash preparation cancelled\n" << std::flush;
    return 130;
  } catch (const std::exception& error) {
    errors << "Flash asset preparation: " << error.what() << '\n' << std::flush;
    return 1;
  }
}

}  // namespace otc::ai
