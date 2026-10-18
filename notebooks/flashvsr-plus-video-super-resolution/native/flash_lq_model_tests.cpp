#include "flash_lq_model.hpp"
#include "flash_stage.hpp"
#include "flash-assets/flash_lq_assets.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/util/variable.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using otc::ai::FlashLqRole;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

template<class Action>
void expectRejected(Action action) {
  try { action(); }
  catch (const std::exception&) { return; }
  throw std::runtime_error("malformed LQ model was accepted");
}

template<class Mutator>
void rejectMutation(const ov::Model& source, FlashLqRole role, Mutator mutate) {
  auto model = source.clone();
  // A broken test mutator must fail the test, not masquerade as rejection.
  mutate(*model);
  expectRejected([&] { otc::ai::validateFlashLqModel(*model, role, 2176, 3840); });
}

void check(const ov::Model& model, FlashLqRole role) {
  otc::ai::validateFlashLqModel(model, role, 2176, 3840);
  otc::ai::FlashPorts ports;
  for (const auto& port : model.inputs())
    ports.emplace(port.get_any_name(), std::make_pair(port.get_element_type(), port.get_shape()));
  // Instantiate the generated rank-five binding ABI against the actual IR,
  // without loading checkpoint payloads or compiling an execution device.
  switch (role) {
    case FlashLqRole::startup:
      otc::ai::validateFlashStageWeights(ports, otc::ai::flashStageWeights(otc::ai::kFlashLqStartupWeights));
      break;
    case FlashLqRole::prefix:
      otc::ai::validateFlashStageWeights(ports, otc::ai::flashStageWeights(otc::ai::kFlashLqPrefixWeights));
      break;
    case FlashLqRole::suffix:
      otc::ai::validateFlashStageWeights(ports, otc::ai::flashStageWeights(otc::ai::kFlashLqSuffixWeights));
      break;
  }
  for (const auto& hw : {std::pair{0U, 3840U}, std::pair{2175U, 3840U},
                        std::pair{4096U, 4096U}, std::pair{128U, 128U}})
    expectRejected([&] { otc::ai::validateFlashLqModel(model, role, hw.first, hw.second); });
  expectRejected([&] {
    otc::ai::validateFlashLqModel(model, static_cast<FlashLqRole>(99), 2176, 3840);
  });
  const std::string weight = role == FlashLqRole::suffix ?
      "weight__conv2_weight" : "weight__conv1_weight";
  rejectMutation(model, role, [&](ov::Model& bad) {
    bad.input(weight).get_tensor().set_names({"weight__unexpected"});
  });
  rejectMutation(model, role, [&](ov::Model& bad) {
    auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(
        bad.input(weight).get_node_shared_ptr());
    require(parameter != nullptr, "weight port is not a Parameter");
    parameter->set_element_type(ov::element::f16);
    parameter->validate_and_infer_types();
  });
  rejectMutation(model, role, [&](ov::Model& bad) {
    auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(
        bad.input(weight).get_node_shared_ptr());
    bad.remove_parameter(parameter);
  });
  rejectMutation(model, role, [](ov::Model& bad) {
    auto extra = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::Shape{1});
    extra->output(0).get_tensor().set_names({"weight__extra"});
    bad.add_parameters({extra});
  });
  rejectMutation(model, role, [&](ov::Model& bad) {
    const std::string name = role == FlashLqRole::startup ? "video" :
        role == FlashLqRole::prefix ? "old_cache1" : "old_cache2";
    auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(
        bad.input(name).get_node_shared_ptr());
    auto shape = parameter->get_shape();
    shape[2] += 1;
    parameter->set_partial_shape(shape);
    parameter->validate_and_infer_types();
  });
  rejectMutation(model, role, [](ov::Model& bad) {
    bad.output(0).get_tensor().set_names({"unexpected_output"});
  });
  rejectMutation(model, role, [](ov::Model& bad) {
    bad.add_variables({std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
        ov::PartialShape{1}, ov::element::bf16, "hidden_cache"})});
  });
  rejectMutation(model, role, [](ov::Model& bad) {
    auto result = bad.output(0).get_node_shared_ptr();
    const auto names = result->output(0).get_names();
    auto half = std::make_shared<ov::op::v0::Convert>(result->input_value(0), ov::element::f16);
    auto back = std::make_shared<ov::op::v0::Convert>(half, ov::element::bf16);
    back->output(0).get_tensor().set_names(names);
    result->input(0).replace_source_output(back);
    result->output(0).get_tensor().set_names(names);
  });
  rejectMutation(model, role, [](ov::Model& bad) {
    auto result = bad.output(0).get_node_shared_ptr();
    const auto names = result->output(0).get_names();
    auto oversized = ov::op::v0::Constant::create(ov::element::bf16, {4097}, {1.0f});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto sum = std::make_shared<ov::op::v1::ReduceSum>(oversized, axis, false);
    auto add = std::make_shared<ov::op::v1::Add>(result->input_value(0), sum);
    add->output(0).get_tensor().set_names(names);
    result->input(0).replace_source_output(add);
    result->output(0).get_tensor().set_names(names);
  });

}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: flash-lq-model-tests STRUCTURAL_ASSETS");
    // Core is used only by the IR reader: never query or compile a device.
    ov::Core core;
    const std::filesystem::path root(argv[1]);
    using namespace otc::ai;
    struct RolePins {
      FlashLqRole role;
      const SeedVrAssetPin& xml;
      const SeedVrAssetPin& bin;
    };
    const std::array<RolePins, 3> roles{{
        {FlashLqRole::startup, kFlashLqStartupXml, kFlashLqStartupStructuralBin},
        {FlashLqRole::prefix, kFlashLqPrefixXml, kFlashLqPrefixStructuralBin},
        {FlashLqRole::suffix, kFlashLqSuffixXml, kFlashLqSuffixStructuralBin}}};
    for (const auto& [role, xmlPin, binPin] : roles) {
      const auto owner = readSeedVrIr(root, xmlPin, binPin);
      const ov::Tensor bin(ov::element::u8, {owner.binBytes()}, owner.binData());
      const auto model = core.read_model(owner.xml(), bin);
      check(*model, role);
    }
    std::cout << "flash_lq_model_validation=pass "
                 "device_queries=0 compile_calls=0 infer_calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "flash_lq_model_validation=fail: " << error.what() << '\n';
    return 1;
  }
}
