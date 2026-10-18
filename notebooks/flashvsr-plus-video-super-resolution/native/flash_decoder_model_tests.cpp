#include "flash_decoder_model.hpp"

#include "flash-assets/flash_decoder_assets.hpp"
#include "flash_stage.hpp"
#include "seedvr_assets.hpp"

#include <openvino/op/assign.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/util/variable.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using otc::ai::FlashDecoderRole;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template<class Action>
void rejected(const char* label, Action action) {
  try {
    action();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string(label) + ": malformed decoder was accepted");
}

struct LoadedModel {
  // OpenVINO can retain structural Constant views into this authenticated owner.
  otc::ai::SeedVrOwnedIr ir;
  std::shared_ptr<ov::Model> model;
};

LoadedModel load(const std::filesystem::path& assets, FlashDecoderRole role) {
  const auto& xml = role == FlashDecoderRole::front ? otc::ai::kFlashDecoderFrontXml :
      role == FlashDecoderRole::grow ? otc::ai::kFlashDecoderGrowXml : otc::ai::kFlashDecoderFrameXml;
  const auto& bin = role == FlashDecoderRole::front ? otc::ai::kFlashDecoderFrontStructuralBin :
      role == FlashDecoderRole::grow ? otc::ai::kFlashDecoderGrowStructuralBin : otc::ai::kFlashDecoderFrameStructuralBin;
  auto ir = otc::ai::readSeedVrIr(assets, xml, bin);
  ov::Core core;
  const ov::Tensor structural(ov::element::u8, ov::Shape{ir.binBytes()}, ir.binData());
  auto model = core.read_model(ir.xml(), structural);
  return {std::move(ir), std::move(model)};
}

template<class Mutator>
void rejectMutation(const ov::Model& source, FlashDecoderRole role,
                    const char* label, Mutator mutate) {
  auto model = source.clone();
  // An exception in the mutator is a test failure, not validator rejection.
  mutate(*model);
  rejected(label, [&] { otc::ai::validateFlashDecoderModel(*model, role, 2176, 3840); });
}

std::shared_ptr<ov::op::v0::Parameter> parameter(ov::Model& model, const std::string& name) {
  const auto value = std::dynamic_pointer_cast<ov::op::v0::Parameter>(model.input(name).get_node_shared_ptr());
  require(value != nullptr, "decoder input is not a Parameter");
  return value;
}

template<class Bindings>
void checkWeights(const ov::Model& model, const Bindings& weights, std::size_t expected) {
  otc::ai::FlashPorts inputs;
  for (const auto& input : model.inputs())
    require(inputs.emplace(input.get_any_name(), std::make_pair(input.get_element_type(), input.get_shape())).second,
            "decoder input names are not unique");
  const auto bindings = otc::ai::flashStageWeights(weights);
  require(bindings.size() == expected, "decoder static weight count differs");
  otc::ai::validateFlashStageWeights(inputs, bindings);
  for (const auto& binding : bindings)
    require(binding.name != "grow_weight" && binding.name != "frame_grow",
            "dynamic child bank was treated as a static learned binding");
}

void checkMalformed(const ov::Model& source, FlashDecoderRole role) {
  const auto input = role == FlashDecoderRole::front ? "packed" :
      role == FlashDecoderRole::grow ? "middle_in" : "middle";
  const auto output = role == FlashDecoderRole::frame ? "frame" : "middle";
  rejectMutation(source, role, "input alias", [&](ov::Model& model) {
    parameter(model, input)->output(0).get_tensor().set_names({input, "alias"});
  });
  rejectMutation(source, role, "input shape", [&](ov::Model& model) {
    auto value = parameter(model, input);
    auto shape = value->get_shape();
    --shape.back();
    value->set_partial_shape(shape);
    value->validate_and_infer_types();
  });
  rejectMutation(source, role, "missing output", [&](ov::Model& model) {
    model.remove_result(std::dynamic_pointer_cast<ov::op::v0::Result>(model.output(output).get_node_shared_ptr()));
  });
  rejectMutation(source, role, "output alias", [&](ov::Model& model) {
    model.output(output).get_tensor().set_names({output, "alias"});
  });
  rejectMutation(source, role, "model role", [](ov::Model& model) {
    model.set_friendly_name("other_decoder_role");
  });
  rejectMutation(source, role, "FP16 intermediate", [&](ov::Model& model) {
    const auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(model.output(output).get_node_shared_ptr());
    const auto half = std::make_shared<ov::op::v0::Convert>(result->input_value(0), ov::element::f16);
    const auto back = std::make_shared<ov::op::v0::Convert>(half, ov::element::bf16);
    result->input(0).replace_source_output(back);
    model.validate_nodes_and_infer_types();
  });
  rejectMutation(source, role, "hidden state", [&](ov::Model& model) {
    const auto value = model.output(output).get_node()->input_value(0);
    const auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
        value.get_partial_shape(), ov::element::bf16, "decoder_hidden_state"});
    model.add_variables({variable});
    model.add_sinks({std::make_shared<ov::op::v6::Assign>(value, variable)});
  });
  if (role != FlashDecoderRole::front) {
    const auto child = role == FlashDecoderRole::grow ? "grow_weight" : "frame_grow";
    rejectMutation(source, role, "dynamic bank renamed as static", [&](ov::Model& model) {
      parameter(model, child)->output(0).get_tensor().set_names({"weight__child"});
    });
    rejectMutation(source, role, "dynamic bank shape", [&](ov::Model& model) {
      auto value = parameter(model, child);
      auto shape = value->get_shape();
      --shape.front();
      value->set_partial_shape(shape);
      value->validate_and_infer_types();
    });
  }
}

void check(const LoadedModel& loaded, FlashDecoderRole role) {
  const auto& model = *loaded.model;
  otc::ai::validateFlashDecoderModel(model, role, 2176, 3840);
  if (role == FlashDecoderRole::front) {
    checkWeights(model, otc::ai::kFlashDecoderFrontWeights, 41);
    require(model.inputs().size() == 48 && model.outputs().size() == 13, "front ABI differs");
  } else if (role == FlashDecoderRole::grow) {
    checkWeights(model, otc::ai::kFlashDecoderGrowWeights, 19);
    require(model.inputs().size() == 24 && model.outputs().size() == 7, "growth ABI differs");
  } else {
    checkWeights(model, otc::ai::kFlashDecoderFrameWeights, 4);
    require(model.inputs().size() == 6 && model.outputs().size() == 1, "frame ABI differs");
  }
  rejected("unadmitted geometry", [&] { otc::ai::validateFlashDecoderModel(model, role, 2160, 3840); });
  rejected("unknown role", [&] {
    otc::ai::validateFlashDecoderModel(model, static_cast<FlashDecoderRole>(99), 2176, 3840);
  });
  checkMalformed(model, role);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: flash-decoder-model-tests STRUCTURAL_ASSETS");
    for (const auto role : {FlashDecoderRole::front, FlashDecoderRole::grow, FlashDecoderRole::frame}) {
      const auto loaded = load(argv[1], role);
      check(loaded, role);
    }
    std::cout << "flash_decoder_model_validation=pass device_queries=0 compile_calls=0 infer_calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "flash_decoder_model_validation=fail: " << error.what() << '\n';
    return 1;
  }
}
