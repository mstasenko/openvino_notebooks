#include "flash_text_model.hpp"

#include "flash-assets/flash_text_assets.hpp"
#include "flash_model.hpp"
#include "flash_stage.hpp"
#include "seedvr_assets.hpp"

#include <openvino/op/convert.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/result.hpp>

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

template <class Action>
void expectRejected(Action action, const std::string& label) {
  try {
    action();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(label + ": malformed text model was accepted");
}

struct LoadedModel {
  otc::ai::SeedVrOwnedIr ir;
  std::shared_ptr<ov::Model> model;
};

LoadedModel load(const std::filesystem::path& assets) {
  auto ir = otc::ai::readSeedVrIr(assets, otc::ai::kFlashTextXml,
                                  otc::ai::kFlashTextStructuralBin);
  // The authenticated text graph has no Constant nodes and therefore no BIN
  // payload.  Reading the XML-only IR still exercises the exact empty-BIN
  // authentication path without constructing a null-backed zero Tensor.
  ov::Core core;
  const ov::Tensor structural(ov::element::u8, ov::Shape{0});
  auto model = core.read_model(ir.xml(), structural);
  return {std::move(ir), std::move(model)};
}

std::shared_ptr<ov::op::v0::Parameter> parameter(ov::Model& model,
                                                   const std::string& name) {
  return std::dynamic_pointer_cast<ov::op::v0::Parameter>(
      model.input(name).get_node_shared_ptr());
}

template <class Mutator>
void rejectMutation(const ov::Model& source, const std::string& label, Mutator mutate) {
  auto malformed = source.clone();
  mutate(*malformed);
  expectRejected([&] { otc::ai::validateFlashTextModel(*malformed); }, label);
}

void checkAbi(const LoadedModel& loaded) {
  const auto& model = *loaded.model;
  otc::ai::validateFlashTextModel(model);
  require(model.inputs().size() == 5 && model.outputs().size() == 2,
          "text port count differs");
  otc::ai::FlashPorts ports;
  for (const auto& input : model.inputs())
    ports.emplace(input.get_any_name(),
                  std::make_pair(input.get_element_type(), input.get_shape()));
  otc::ai::validateFlashStageWeights(
      ports, otc::ai::flashStageWeights(otc::ai::kFlashTextWeights));
  const auto spans = otc::ai::flashWeightSpans(otc::ai::kFlashTextWeights);
  require(spans.size() == 4 && spans.front().offset == 0,
          "text learned spans differ");
  require(otc::ai::kFlashTextRows == 512 &&
              otc::ai::kFlashTextStructuralBin.bytes == 0,
          "text structural pins differ");
  std::set<std::string> outputs;
  for (const auto& output : model.outputs()) outputs.insert(output.get_any_name());
  require(outputs == std::set<std::string>{"context", "text_linear"},
          "text output names differ");
}

void checkMalformed(const ov::Model& source) {
  rejectMutation(source, "renamed prompt", [](ov::Model& model) {
    parameter(model, "prompt")->output(0).get_tensor().set_names({"prompt_tampered"});
  });
  rejectMutation(source, "wrong learned type", [](ov::Model& model) {
    auto weight = parameter(model, "weight__text_embedding_0_weight");
    weight->set_element_type(ov::element::f16);
    weight->validate_and_infer_types();
  });
  rejectMutation(source, "wrong prompt shape", [](ov::Model& model) {
    auto prompt = parameter(model, "prompt");
    prompt->set_partial_shape({1, 512, 4095});
    prompt->validate_and_infer_types();
  });
  rejectMutation(source, "extra input", [](ov::Model& model) {
    auto extra = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::Shape{1});
    extra->output(0).get_tensor().set_names({"unexpected_input"});
    model.add_parameters({extra});
  });
  rejectMutation(source, "renamed output", [](ov::Model& model) {
    model.output("context").get_tensor().set_names({"context_tampered"});
  });
  rejectMutation(source, "wrong GELU approximation", [](ov::Model& model) {
    for (const auto& node : model.get_ordered_ops()) {
      if (auto gelu = std::dynamic_pointer_cast<ov::op::v7::Gelu>(node))
        gelu->set_approximation_mode(ov::op::GeluApproximationMode::ERF);
    }
    model.validate_nodes_and_infer_types();
  });
  rejectMutation(source, "numeric FP16 intermediate", [](ov::Model& model) {
    auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output("context").get_node_shared_ptr());
    auto convert = std::make_shared<ov::op::v0::Convert>(result->input_value(0),
                                                          ov::element::f16);
    result->input(0).replace_source_output(convert);
    model.validate_nodes_and_infer_types();
  });
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: flash-text-model-tests STRUCTURAL_ASSETS");
    const auto loaded = load(std::filesystem::path(argv[1]));
    checkAbi(loaded);
    checkMalformed(*loaded.model);
    std::cout << "flash_text_model_validation=pass device_calls=0 compile_calls=0 "
                 "infer_calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "flash_text_model_validation=fail: " << error.what() << '\n';
    return 1;
  }
}
