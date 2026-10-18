#include "seedvr_prepare_cli.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include "seedvr_assets.hpp"
#include "seedvr_component_assets.hpp"
#include "seedvr_model_assets.hpp"
#include "seedvr_prepare.hpp"
#include "seedvr_prepare_data.hpp"

namespace otc::ai {
namespace {
struct Options {
  bool prepare = false;
  bool structuralOnly = false;
  std::filesystem::path models, sources, complements;
};
Options parse(int argc, char** argv) {
  Options result;
  result.prepare = std::string_view(argv[1]) == "--prepare-models";
  std::set<std::string> seen;
  for (int i = 2; i < argc; ++i) {
    const std::string option = argv[i];
    if (!seen.insert(option).second)
      throw std::invalid_argument("duplicate preparation option");
    if (option == "--structural-only" && !result.prepare) {
      result.structuralOnly = true;
      continue;
    }
    if (option != "--models" && option != "--sources" &&
        option != "--complements")
      throw std::invalid_argument("unknown preparation option");
    if (++i >= argc || !argv[i][0] ||
        std::string_view(argv[i]).substr(0, 2) == "--")
      throw std::invalid_argument("missing preparation directory");
    const std::filesystem::path path = argv[i];
    if (option == "--models")
      result.models = path;
    else if (option == "--sources")
      result.sources = path;
    else
      result.complements = path;
  }
  if (result.models.empty() ||
      (result.prepare &&
       (result.sources.empty() || result.complements.empty())) ||
      (!result.prepare && !result.sources.empty()) ||
      (result.structuralOnly && result.complements.empty()) ||
      (!result.prepare && !result.structuralOnly &&
       !result.complements.empty()))
    throw std::invalid_argument(
        "incomplete or incompatible preparation options");
  return result;
}
void stopped(volatile std::sig_atomic_t& stop) {
  if (stop) throw SeedVrPrepareCancelled();
}
void emit(std::ostream& output, const char* phase, const std::string& name,
          std::uint64_t done, std::uint64_t total) {
  output << "{\"phase\":\"" << phase << "\",\"name\":" << std::quoted(name)
         << ",\"done\":" << done << ",\"total\":" << total << "}\n"
         << std::flush;
  if (!output) throw std::runtime_error("preparation progress output failed");
}
bool isXml(const SeedVrAssetPin& pin) {
  return pin.name.size() >= 4 && pin.name.substr(pin.name.size() - 4) == ".xml";
}
void verify(const Options& options, volatile std::sig_atomic_t& stop,
            std::ostream& output) {
  std::uint64_t total = kSeedVrPositiveText.bytes;
  for (const auto& pin : kSeedVrModelAssetPins)
    if (!options.structuralOnly || isXml(pin)) total += pin.bytes;
  for (const auto& component : kSeedVrComponentAssets)
    total += component.xml.bytes +
             (options.structuralOnly ? 0 : component.bin.bytes);
  if (options.structuralOnly)
    for (const auto& plan : prepare_detail::plans)
      total += plan.complement.bytes;
  std::uint64_t done = 0;
  const auto check = [&](const std::filesystem::path& directory,
                         const SeedVrAssetPin& pin) {
    stopped(stop);
    emit(output, "verifying", pin.name, done, total);
    if (isXml(pin))
      (void)readSeedVrXml(directory, pin);
    else
      verifySeedVrAsset(directory, pin);
    stopped(stop);
    done += pin.bytes;
    emit(output, "verifying", pin.name, done, total);
  };
  for (const auto& pin : kSeedVrModelAssetPins)
    if (!options.structuralOnly || isXml(pin)) check(options.models, pin);
  for (const auto& component : kSeedVrComponentAssets) {
    check(options.models, component.xml);
    if (!options.structuralOnly) check(options.models, component.bin);
  }
  stopped(stop);
  (void)readSeedVrPositiveText(options.models);
  done += kSeedVrPositiveText.bytes;
  emit(output, "verifying", kSeedVrPositiveText.name, done, total);
  if (options.structuralOnly)
    for (const auto& plan : prepare_detail::plans)
      check(options.complements, plan.complement);
  stopped(stop);
}
}  // namespace

std::optional<int> runSeedVrPrepareCli(int argc, char** argv,
                                       volatile std::sig_atomic_t& stop,
                                       std::ostream& output,
                                       std::ostream& errors) {
  if (argc < 2 || (std::string_view(argv[1]) != "--prepare-models" &&
                   std::string_view(argv[1]) != "--verify-models"))
    return std::nullopt;
  try {
    const auto options = parse(argc, argv);
    stopped(stop);
    if (!options.prepare) {
      verify(options, stop, output);
      return 0;
    }
    Options structural = options;
    structural.prepare = false;
    structural.structuralOnly = true;
    verify(structural, stop, output);
    std::uint64_t total =
        prepare_detail::source0.bytes + prepare_detail::source1.bytes;
    for (const auto& plan : prepare_detail::plans)
      total += plan.complement.bytes + plan.bin.bytes;
    std::string previous;
    std::uint64_t base = 0, last = 0;
    auto emitted = std::chrono::steady_clock::time_point{};
    prepareSeedVrIr(
        options.sources, options.complements, options.models,
        [&](const auto& progress) {
          if (stop) return false;
          if (progress.stage == SeedVrPrepareStage::published) return true;
          const auto key =
              std::to_string(static_cast<int>(progress.stage)) + progress.name;
          if (key != previous) {
            base += last;
            last = 0;
            previous = key;
          }
          last = progress.done;
          const auto now = std::chrono::steady_clock::now();
          if (progress.done == 0 || progress.done == progress.total ||
              now - emitted >= std::chrono::milliseconds(200)) {
            emit(output, "preparing", progress.name, base + last, total);
            emitted = now;
          }
          return true;
        });
    Options prepared;
    prepared.models = options.models;
    verify(prepared, stop, output);
    return 0;
  } catch (const SeedVrPrepareCancelled&) {
    errors << "SeedVR2 preparation cancelled\n" << std::flush;
    return 130;
  } catch (const std::exception& error) {
    errors << "SeedVR2 asset preparation: " << error.what() << '\n'
           << std::flush;
    return 1;
  }
}
}  // namespace otc::ai
