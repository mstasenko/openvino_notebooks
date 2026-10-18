#include "flash_prepare_cli.hpp"

#include "flash_prepare.hpp"

#include <cassert>
#include <csignal>
#include <sstream>
#include <string>
#include <vector>

#ifdef OTC_FLASH_PREPARE_TESTING
namespace otc::ai {
bool testFlashPrepareVerificationProgress(
    const std::filesystem::path& fixtureDirectory);
}  // namespace otc::ai
#endif

namespace {

std::optional<int> invoke(const std::vector<std::string>& arguments,
                          volatile std::sig_atomic_t& stop,
                          std::string& errors) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (const auto& argument : arguments)
    argv.push_back(const_cast<char*>(argument.c_str()));
  std::ostringstream progress;
  std::ostringstream failure;
  const auto result = otc::ai::runFlashPrepareCli(
      static_cast<int>(argv.size()), argv.data(), stop, progress, failure);
  errors = failure.str();
  return result;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef OTC_FLASH_PREPARE_TESTING
  (void)argc;
  (void)argv;
#endif
  volatile std::sig_atomic_t stop = 0;
  std::string errors;

  assert(!invoke({"test", "--ordinary-export"}, stop, errors).has_value());
  assert(invoke({"test", "--verify-models", "--models", "/missing"}, stop,
                errors) == 1);
  assert(!errors.empty());
  assert(invoke({"test", "--verify-models", "--mode", "invalid", "--models",
                 "/missing"}, stop, errors) == 1);
  assert(invoke({"test", "--verify-models", "--mode", "turbo", "--mode", "base",
                 "--models", "/missing"}, stop, errors) == 1);
  assert(invoke({"test", "--prepare-models", "--mode", "turbo",
                 "--models", "/missing", "--sources", "/missing",
                 "--complements", "/missing", "--structural-only"}, stop,
                errors) == 1);

  stop = 1;
  assert(invoke({"test", "--verify-models", "--mode", "turbo", "--models",
                 "/missing"}, stop, errors) == 130);
  stop = 0;

  const auto totals = otc::ai::flashPrepareTotals(otc::ai::FlashMode::turbo);
  assert(totals.structural > 0 && totals.mode > totals.decoder &&
         totals.prepared > totals.decoder);
#ifdef OTC_FLASH_PREPARE_TESTING
  assert(argc == 2);
  const std::filesystem::path fixture = argv[1];
  assert(std::filesystem::is_directory(fixture));
  assert(otc::ai::testFlashPrepareVerificationProgress(fixture));
#endif
  return 0;
}
