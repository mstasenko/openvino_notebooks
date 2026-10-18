#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "seedvr_prepare_cli.hpp"

using otc::ai::runSeedVrPrepareCli;
int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture = argv[1];
  assert(std::filesystem::create_directory(fixture));
  std::ofstream(fixture / "block0_even_fused_loop_tail_fp16.xml") << "<net/>";
  auto call = [&](std::vector<std::string> arguments,
                  std::sig_atomic_t requested = 0) {
    std::vector<char*> raw;
    for (auto& argument : arguments) raw.push_back(argument.data());
    volatile std::sig_atomic_t stop = requested;
    std::ostringstream progress, errors;
    return runSeedVrPrepareCli(raw.size(), raw.data(), stop, progress, errors);
  };
  assert(!call({"worker", "--models", "export"}));
  assert(!call({"worker"}));
  assert(call({"worker", "--prepare-models"}) == 1);
  assert(call({"worker", "--verify-models"}) == 1);
  assert(call({"worker", "--verify-models", "--models", fixture.string()}, 1) ==
         130);
  assert(call({"worker", "--verify-models", "--models", fixture.string()}) ==
         1);
  assert(call({"worker", "--verify-models", "--models", fixture.string(),
               "--models", fixture.string()}) == 1);
  assert(call({"worker", "--verify-models", "--models", "--device", "GPU"}) ==
         1);
  assert(call({"worker", "--verify-models", "--models", fixture.string(),
               "--device", "CPU"}) == 1);
  assert(call({"worker", "--verify-models", "--models", fixture.string(),
               "--sources", "source"}) == 1);
  assert(call({"worker", "--verify-models", "--models", fixture.string(),
               "--complements", "data"}) == 1);
  assert(call({"worker", "--verify-models", "--models", fixture.string(),
               "--structural-only"}) == 1);
  assert(call({"worker", "--verify-models", "--models", fixture.string(),
               "--complements", "data", "--structural-only",
               "--structural-only"}) == 1);
  assert(call({"worker", "--prepare-models", "--models", fixture.string(),
               "--sources", "source", "--complements", "data",
               "--structural-only"}) == 1);
  std::filesystem::create_directory_symlink(fixture, fixture / "link");
  assert(call({"worker", "--verify-models", "--models",
               (fixture / "link").string()}) == 1);
  std::cout << "PASS CPU-only CLI dispatch/cancellation; malformed "
               "flags/paths/tinywrongasset rejected\n";
}
