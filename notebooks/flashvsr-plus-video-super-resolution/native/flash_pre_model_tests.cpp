#include "flash_pre_model.hpp"

#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/util/variable.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct LoadedModel {
  std::shared_ptr<ov::Model> model;
};

LoadedModel load(const std::filesystem::path& assets) {
  ov::Core core;
  auto model = core.read_model((assets / "flash_pre_4096_bf16.xml").string(),
                               (assets / "flash_pre_4096_bf16.bin").string());
  return {std::move(model)};
}

template <class Mutator>
void expectRejected(const ov::Model& source, const std::string& label, Mutator mutator) {
  auto malformed = source.clone();
  mutator(*malformed);
  try {
    otc::ai::prepareFlashPreModel(*malformed);
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(label + ": malformed PRE model was accepted");
}

std::shared_ptr<ov::op::v0::Parameter> parameter(ov::Model& model, const std::string& name) {
  return std::dynamic_pointer_cast<ov::op::v0::Parameter>(
      model.input(name).get_node_shared_ptr());
}

void checkPrepared(const ov::Model& model) {
  const std::set<std::string> publicNames{
      "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp", "q_pooled", "k_pooled",
      "q_device_0", "q_device_1", "q_finite", "k_device", "v_device", "kv_finite"};
  const std::set<std::string> stateIds{
      "plus_pre_state_modulation_product", "plus_pre_state_q_linear_storage",
      "plus_pre_state_k_linear_storage", "plus_pre_state_norm1_storage",
      "plus_pre_state_q_rms_storage", "plus_pre_state_k_rms_storage"};
  std::set<std::string> outputs;
  for (const auto& output : model.outputs()) {
    require(output.get_names().size() == 1, "prepared output has multiple names");
    outputs.insert(output.get_any_name());
  }
  require(outputs == publicNames, "prepared PRE public ABI differs");
  std::set<std::string> variables;
  for (const auto& variable : model.get_variables())
    variables.insert(variable->get_info().variable_id);
  require(variables == stateIds, "prepared PRE state IDs differ");
  for (const auto& sink : model.get_sinks()) {
    const auto assign = std::dynamic_pointer_cast<ov::op::v6::Assign>(sink);
    require(assign != nullptr, "prepared PRE sink is not Assign");
    require(stateIds.count(assign->get_variable_id()) == 1,
            "prepared PRE sink ID differs");
  }
}

void checkSuccessAndFailures(const std::filesystem::path& assets) {
  auto loaded = load(assets);
  require(loaded.model->inputs().size() == 13 && loaded.model->outputs().size() == 18,
          "ordinary PRE IR ABI differs before preparation");
  require(loaded.model->get_variables().empty() && loaded.model->get_sinks().empty(),
          "ordinary PRE IR unexpectedly contains state");

  otc::ai::prepareFlashPreModel(*loaded.model);
  otc::ai::validateFlashPreModel(*loaded.model);
  checkPrepared(*loaded.model);
  expectRejected(*loaded.model, "double prepare", [](ov::Model&) {});

  const auto source = load(assets).model;
  expectRejected(*source, "renamed input", [](ov::Model& model) {
    parameter(model, "x")->output(0).set_names({"x_tampered"});
  });
  expectRejected(*source, "wrong input type", [](ov::Model& model) {
    auto x = parameter(model, "x");
    x->set_element_type(ov::element::f16);
    x->validate_and_infer_types();
  });
  expectRejected(*source, "wrong input shape", [](ov::Model& model) {
    auto x = parameter(model, "x");
    x->set_partial_shape({1, 4096, 1535});
    x->validate_and_infer_types();
  });
  expectRejected(*source, "missing public output", [](ov::Model& model) {
    model.remove_result(std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output("k_device").get_node_shared_ptr()));
  });
  expectRejected(*source, "missing private barrier", [](ov::Model& model) {
    model.remove_result(std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output("q_linear_storage").get_node_shared_ptr()));
  });
  expectRejected(*source, "preexisting state", [](ov::Model& model) {
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
        ov::PartialShape{1, 4096, 1536}, ov::element::bf16, "preexisting"});
    model.add_variables({variable});
  });
  expectRejected(*source, "changed private chain", [](ov::Model& model) {
    auto sourceNode = model.output("modulation_product").get_node()->input_value(0).get_node_shared_ptr();
    sourceNode->input(0).replace_source_output(model.input("x"));
  });
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto assets = argc == 2 ? std::filesystem::path(argv[1])
                                 : std::filesystem::path("assets/flash-assets");
    checkSuccessAndFailures(assets);
    std::cout << "flash_pre_model_validation=pass device_calls=0 compile_calls=0 infer_calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "flash_pre_model_validation=fail: " << error.what() << '\n';
    return 1;
  }
}
