#include "flash_cross_model.hpp"

#include "flash-assets/flash_cross_assets.hpp"
#include "flash_stage.hpp"
#include "seedvr_assets.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/util/variable.hpp>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using otc::ai::SeedVrOwnedIr;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct LoadedModel {
  // The model may retain pointers into this authenticated XML/BIN owner.
  SeedVrOwnedIr ir;
  std::shared_ptr<ov::Model> model;
};

template <class Action>
void expectRejected(Action action, const std::string& label) {
  try {
    action();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(label + ": malformed Cross model was accepted");
}

LoadedModel load(const std::filesystem::path& assets) {
  // The fixed Cross BIN contains only structural constants.  Learned spans
  // point into the separately authenticated safetensors sources and are
  // checked against the model ports below, never against this 12-byte BIN.
  auto ir = otc::ai::readSeedVrIr(assets, otc::ai::kFlashCrossXml,
                                  otc::ai::kFlashCrossStructuralBin);
  ov::Core core;
  const ov::Tensor structural(ov::element::u8, ov::Shape{ir.binBytes()}, ir.binData());
  auto model = core.read_model(ir.xml(), structural);
  return {std::move(ir), std::move(model)};
}

template <class Mutator>
void rejectMutation(const ov::Model& source, const std::string& label, Mutator mutate) {
  auto malformed = source.clone();
  // A test mutator must itself construct a graph; only validator rejection is
  // accepted as the expected outcome.
  mutate(*malformed);
  try {
    otc::ai::validateFlashCrossModel(*malformed);
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(label + ": malformed Cross model was accepted");
}

std::shared_ptr<ov::op::v0::Parameter> parameter(ov::Model& model, const std::string& name) {
  return std::dynamic_pointer_cast<ov::op::v0::Parameter>(model.input(name).get_node_shared_ptr());
}

void checkAuthenticatedAbi(const LoadedModel& loaded) {
  const auto& model = *loaded.model;
  otc::ai::validateFlashCrossModel(model);

  otc::ai::FlashPorts ports;
  for (const auto& input : model.inputs()) {
    require(ports.emplace(input.get_any_name(),
                          std::make_pair(input.get_element_type(), input.get_shape()))
                .second,
            "duplicate input port while checking generated weights");
  }
  otc::ai::validateFlashStageWeights(
      ports, otc::ai::flashStageWeights(otc::ai::kFlashCrossWeights));
  const auto spans = otc::ai::flashWeightSpans(otc::ai::kFlashCrossWeights);
  require(spans.size() == 5 && spans.front().offset == 12,
          "generated Cross learned spans differ");

  require(otc::ai::kFlashCrossRows == 512 &&
              otc::ai::kFlashCrossStructuralBin.bytes == 12,
          "generated Cross structural pins differ");
  require(model.inputs().size() == 6 && model.outputs().size() == 2,
          "fixed Cross port count differs");
  std::set<std::string> outputNames;
  for (const auto& output : model.outputs()) outputNames.insert(output.get_any_name());
  require(outputNames == std::set<std::string>{"cross_k", "cross_v"},
          "fixed Cross output names differ");
}

void checkMalformed(const ov::Model& source) {
  rejectMutation(source, "missing context", [](ov::Model& model) {
    model.remove_parameter(parameter(model, "context"));
  });
  rejectMutation(source, "extra input", [](ov::Model& model) {
    auto extra = std::make_shared<ov::op::v0::Parameter>(
        ov::element::bf16, ov::Shape{1, 512, 1536});
    extra->output(0).get_tensor().set_names({"unexpected_input"});
    model.add_parameters({extra});
  });
  rejectMutation(source, "renamed input", [](ov::Model& model) {
    parameter(model, "context")->output(0).get_tensor().set_names({"context_tampered"});
  });
  rejectMutation(source, "wrong input type", [](ov::Model& model) {
    auto context = parameter(model, "context");
    context->set_element_type(ov::element::f16);
    context->validate_and_infer_types();
  });
  rejectMutation(source, "wrong input shape", [](ov::Model& model) {
    auto context = parameter(model, "context");
    context->set_partial_shape({1, 512, 1535});
    context->validate_and_infer_types();
  });
  rejectMutation(source, "renamed learned port", [](ov::Model& model) {
    parameter(model, "weight__blocks_0_cross_attn_k_weight")
        ->output(0).get_tensor().set_names({"weight__tampered"});
  });
  rejectMutation(source, "wrong learned shape", [](ov::Model& model) {
    auto weight = parameter(model, "weight__blocks_0_cross_attn_k_weight");
    weight->set_partial_shape({1536, 1535});
    weight->validate_and_infer_types();
  });
  rejectMutation(source, "missing output", [](ov::Model& model) {
    model.remove_result(std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output("cross_k").get_node_shared_ptr()));
  });
  rejectMutation(source, "extra output", [](ov::Model& model) {
    model.add_results({std::make_shared<ov::op::v0::Result>(model.output("cross_k"))});
  });
  rejectMutation(source, "hidden state", [](ov::Model& model) {
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
        ov::PartialShape{1, 512, 1536}, ov::element::bf16, "unexpected_cross_state"});
    auto assign = std::make_shared<ov::op::v6::Assign>(model.input("context"), variable);
    model.add_variables({variable});
    model.add_sinks({assign});
  });
  rejectMutation(source, "ReadValue state", [](ov::Model& model) {
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
        ov::PartialShape{1, 512, 1536}, ov::element::bf16, "unexpected_read_state"});
    auto read = std::make_shared<ov::op::v6::ReadValue>(variable);
    auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output("cross_k").get_node_shared_ptr());
    auto add = std::make_shared<ov::op::v1::Add>(result->input_value(0), read);
    result->input(0).replace_source_output(add);
    model.add_variables({variable});
    model.validate_nodes_and_infer_types();
  });
  rejectMutation(source, "FP16 intermediate", [](ov::Model& model) {
    auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output("cross_k").get_node_shared_ptr());
    auto toF16 = std::make_shared<ov::op::v0::Convert>(result->input_value(0), ov::element::f16);
    auto back = std::make_shared<ov::op::v0::Convert>(toF16, ov::element::bf16);
    result->input(0).replace_source_output(back);
    model.validate_nodes_and_infer_types();
  });
  rejectMutation(source, "oversized structural Constant", [](ov::Model& model) {
    // Just over the per-Constant limit, not a learned array.
    std::vector<float> values(2049, 0.0f);
    auto oversized = ov::op::v0::Constant::create(ov::element::bf16, {2049}, values);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto scalar = std::make_shared<ov::op::v1::ReduceSum>(oversized, axes, false);
    auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output("cross_k").get_node_shared_ptr());
    auto add = std::make_shared<ov::op::v1::Add>(result->input_value(0), scalar);
    result->input(0).replace_source_output(add);
    model.validate_nodes_and_infer_types();
  });
  rejectMutation(source, "total structural Constant budget", [](ov::Model& model) {
    auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output("cross_k").get_node_shared_ptr());
    auto value = result->input_value(0);
    const std::vector<float> zeros(2048, 0.0f);
    const auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    // Each 4-KiB Constant is allowed alone, but their combined size is not.
    for (int i = 0; i < 17; ++i) {
      const auto block = ov::op::v0::Constant::create(ov::element::bf16, {2048}, zeros);
      const auto scalar = std::make_shared<ov::op::v1::ReduceSum>(block, axes, false);
      value = std::make_shared<ov::op::v1::Add>(value, scalar)->output(0);
    }
    result->input(0).replace_source_output(value);
    model.validate_nodes_and_infer_types();
  });
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: flash-cross-model-tests STRUCTURAL_ASSETS");
    const auto loaded = load(std::filesystem::path(argv[1]));
    checkAuthenticatedAbi(loaded);
    checkMalformed(*loaded.model);
    std::cout << "flash_cross_model_validation=pass device_calls=0 compile_calls=0 infer_calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "flash_cross_model_validation=fail: " << error.what() << '\n';
    return 1;
  }
}
