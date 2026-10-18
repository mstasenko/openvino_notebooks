#include "flash_stage.hpp"

#include <openvino/op/parameter.hpp>

#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>

using namespace otc::ai;

int main() {
  const FlashPorts ports{{"x", {ov::element::bf16, {1, 2}}},
                         {"weight__w", {ov::element::bf16, {2, 3}}}};
  const FlashStageWeight valid{"weight__w", {8, 12, 6}, {2, 3}};
  const auto rejects = [](const std::function<void()>& call) {
    bool threw = false;
    try { call(); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
  };
  std::vector<std::uint16_t> finite;
  for (unsigned word = 0; word < 65536; ++word)
    if ((word & 0x7f80U) != 0x7f80U) finite.push_back(static_cast<std::uint16_t>(word));
  validateFlashBf16Output(finite.data(), finite.size());
  rejects([&] { validateFlashBf16Output(nullptr, finite.size()); });
  rejects([&] { validateFlashBf16Output(finite.data(), 0); });
  for (unsigned word = 0; word < 65536; ++word) {
    if ((word & 0x7f80U) != 0x7f80U) continue;
    const auto bad = static_cast<std::uint16_t>(word);
    rejects([&] { validateFlashBf16Output(&bad, 1); });
    const std::array<std::uint16_t, 3> late{{0x7f7f, 0xff7f, bad}};
    rejects([&] { validateFlashBf16Output(late.data(), late.size()); });
  }
  // Exercise every lane and scalar tail of the vectorizable scan.
  std::array<std::uint16_t, 67> lanes{};
  lanes.fill(0x7f7f);
  validateFlashBf16Output(lanes.data(), lanes.size());
  for (std::size_t index = 0; index < lanes.size(); ++index) {
    lanes[index] = 0xff81; // Negative NaN must not be hidden by finite neighbors.
    rejects([&] { validateFlashBf16Output(lanes.data(), lanes.size()); });
    lanes[index] = 0x7f7f;
  }
  const FlashPorts publicPorts{{"x", {ov::element::bf16, {1, 2}}},
                                {"y", {ov::element::bf16, {1, 2}}}};
  auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::Shape{1, 2});
  auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::Shape{1, 2});
  const ov::OutputVector aliases{x, y};
  x->output(0).get_tensor().set_names({"x", "x_1"});
  y->output(0).get_tensor().set_names({"y"});
  validateFlashPorts(aliases, publicPorts, true);
  rejects([&] { validateFlashPorts(aliases, publicPorts); }); // Source remains strict.
  x->output(0).get_tensor().set_names({"x_1"});
  rejects([&] { validateFlashPorts(aliases, publicPorts, true); }); // Canonical name lost.
  x->output(0).get_tensor().set_names({"x", "y"});
  rejects([&] { validateFlashPorts(aliases, publicPorts, true); }); // Public alias collision.
  x->output(0).get_tensor().set_names({"x"});
  y->output(0).get_tensor().set_names({"x", "private_alias"});
  rejects([&] { validateFlashPorts(aliases, publicPorts, true); }); // Duplicate public port.
  validateFlashStageWeights(ports, {valid});
  struct BindingFixture {
    std::array<std::size_t, 3> dimensions;
    std::size_t rank;
    SeedVrWeightSpan span;
  };
  const BindingFixture validBinding{{2, 3, 1}, 2, valid.span};
  assert((flashWeightShape(validBinding) == ov::Shape{2, 3}));
  validateFlashWeightExtent({2, 3}, valid.span);
  rejects([&] { validateFlashWeightExtent({}, {0, 0, 0}); });
  rejects([&] { validateFlashWeightExtent({0, 3}, valid.span); });
  rejects([&] { validateFlashWeightExtent({std::numeric_limits<std::size_t>::max(), 2}, valid.span); });
  rejects([&] { validateFlashWeightExtent({2, 3}, {8, 10, 6}); });
  rejects([&] { flashWeightShape(BindingFixture{{2, 3, 1}, 0, valid.span}); });
  rejects([&] { flashWeightShape(BindingFixture{{2, 3, 1}, 4, valid.span}); });
  rejects([&] { validateFlashStageWeights(ports, {}); }); // Cannot feed a missing weight as data.
  rejects([&] { validateFlashStageWeights(ports, {valid, valid}); });
  for (int mutation = 0; mutation < 8; ++mutation) {
    auto bad = valid;
    switch (mutation) {
      case 0: bad.name = "x"; break;
      case 1: bad.name = "weight__unknown"; break;
      case 2: bad.shape = {}; break;
      case 3: bad.shape = {0, 3}; break;
      case 4: bad.shape = {2, 4}; break;
      case 5: bad.shape = {std::numeric_limits<std::size_t>::max(), 2}; break;
      case 6: --bad.span.bytes; break;
      case 7: --bad.span.elements; break;
    }
    rejects([&] { validateFlashStageWeights(ports, {bad}); });
  }
  auto wrongType = ports;
  wrongType.at("weight__w").first = ov::element::f16;
  rejects([&] { validateFlashStageWeights(wrongType, {valid}); });
  auto extra = ports;
  extra.emplace("weight__missing", std::make_pair(ov::element::bf16, ov::Shape{1}));
  rejects([&] { validateFlashStageWeights(extra, {valid}); });
  std::cout << "flash_stage_binding_validation=pass device_calls=0 compile_calls=0 infer_calls=0\n";
}
