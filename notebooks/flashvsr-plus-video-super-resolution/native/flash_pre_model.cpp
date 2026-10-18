#include "flash_pre_model.hpp"
#include "flash_stage.hpp"

#include <openvino/op/assign.hpp>
#include <openvino/op/result.hpp>

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace otc::ai {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("FlashVSR+ PRE: " + message);
}

const ov::Shape kRows{1, 4096, 1536};
const ov::Shape kModulation{1, 6, 1536};
const ov::Shape kRope{1, 4096, 1, 64};
const ov::Shape kGate{1, 1, 1536};
const ov::Shape kPooled{12, 32, 128};
const ov::Shape kQueryDevice{12, 16, 128, 128};
const ov::Shape kKvDevice{32, 128, 12, 128};

const std::array<const char*, 6> kPrivateNames{
    "modulation_product", "q_linear_storage", "k_linear_storage",
    "norm1_storage", "q_rms_storage", "k_rms_storage"};

const std::array<const char*, 6> kStateIds{
    "plus_pre_state_modulation_product",
    "plus_pre_state_q_linear_storage",
    "plus_pre_state_k_linear_storage",
    "plus_pre_state_norm1_storage",
    "plus_pre_state_q_rms_storage",
    "plus_pre_state_k_rms_storage"};

struct PortSpec {
  const char* name;
  ov::element::Type type;
  ov::Shape shape;
};

struct PrivateSpec {
  const char* name;
  const char* sourceType;
  const char* downstreamType;
  std::array<const char*, 2> inputTypes;
  std::array<ov::element::Type, 2> inputElementTypes;
  std::array<ov::Shape, 2> inputShapes;
  std::size_t inputCount;
};

const std::array<PortSpec, 13> kInputs{{
    {"x", ov::element::bf16, kRows},
    {"t_mod", ov::element::bf16, kModulation},
    {"rope_cos", ov::element::f32, kRope},
    {"rope_sin", ov::element::f32, kRope},
    {"weight__blocks_0_modulation", ov::element::bf16, kModulation},
    {"weight__blocks_0_self_attn_q_weight", ov::element::bf16, {1536, 1536}},
    {"weight__blocks_0_self_attn_q_bias", ov::element::bf16, {1536}},
    {"weight__blocks_0_self_attn_norm_q_weight", ov::element::bf16, {1536}},
    {"weight__blocks_0_self_attn_k_weight", ov::element::bf16, {1536, 1536}},
    {"weight__blocks_0_self_attn_k_bias", ov::element::bf16, {1536}},
    {"weight__blocks_0_self_attn_norm_k_weight", ov::element::bf16, {1536}},
    {"weight__blocks_0_self_attn_v_weight", ov::element::bf16, {1536, 1536}},
    {"weight__blocks_0_self_attn_v_bias", ov::element::bf16, {1536}},
}};

const std::array<PortSpec, 12> kPublicOutputs{{
    {"gate_msa", ov::element::bf16, kGate},
    {"shift_mlp", ov::element::bf16, kGate},
    {"scale_mlp", ov::element::bf16, kGate},
    {"gate_mlp", ov::element::bf16, kGate},
    {"q_pooled", ov::element::f32, kPooled},
    {"k_pooled", ov::element::f32, kPooled},
    {"q_device_0", ov::element::bf16, kQueryDevice},
    {"q_device_1", ov::element::bf16, kQueryDevice},
    {"q_finite", ov::element::f16, {}},
    {"k_device", ov::element::bf16, kKvDevice},
    {"v_device", ov::element::bf16, kKvDevice},
    {"kv_finite", ov::element::f16, {}},
}};

const std::array<PortSpec, 6> kPrivateOutputs{{
    {"modulation_product", ov::element::bf16, kRows},
    {"q_linear_storage", ov::element::bf16, kRows},
    {"k_linear_storage", ov::element::bf16, kRows},
    {"norm1_storage", ov::element::bf16, kRows},
    {"q_rms_storage", ov::element::bf16, kRows},
    {"k_rms_storage", ov::element::bf16, kRows},
}};

const std::array<PrivateSpec, 6> kPrivateSpecs{{
    {"modulation_product", "Multiply", "Add", {"Convert", "Add"},
     {ov::element::bf16, ov::element::bf16}, {kRows, kGate}, 2},
    {"q_linear_storage", "Add", "Convert", {"MatMul", "Reshape"},
     {ov::element::bf16, ov::element::bf16}, {kRows, kGate}, 2},
    {"k_linear_storage", "Add", "Convert", {"MatMul", "Reshape"},
     {ov::element::bf16, ov::element::bf16}, {kRows, kGate}, 2},
    {"norm1_storage", "Convert", "Multiply", {"Divide", nullptr},
     {ov::element::f32, ov::element::dynamic}, {kRows, {}}, 1},
    {"q_rms_storage", "Multiply", "Convert", {"Convert", "Reshape"},
     {ov::element::bf16, ov::element::bf16}, {kRows, kGate}, 2},
    {"k_rms_storage", "Multiply", "Convert", {"Convert", "Reshape"},
     {ov::element::bf16, ov::element::bf16}, {kRows, kGate}, 2},
}};

template <class Ports, std::size_t N>
void validatePorts(const Ports& ports, const std::array<PortSpec, N>& expected,
                  const char* role) {
  FlashPorts typed;
  for (const auto& spec : expected) typed.emplace(spec.name, std::make_pair(spec.type, spec.shape));
  try { validateFlashPorts(ports, typed); }
  catch (const std::exception& error) { throw std::runtime_error(std::string(role) + ": " + error.what()); }
}

std::array<PortSpec, 18> sourceOutputs() {
  std::array<PortSpec, 18> result{};
  std::copy(kPublicOutputs.begin(), kPublicOutputs.end(), result.begin());
  std::copy(kPrivateOutputs.begin(), kPrivateOutputs.end(), result.begin() + kPublicOutputs.size());
  return result;
}

const PrivateSpec& privateSpec(const std::string& name) {
  const auto found = std::find_if(kPrivateSpecs.begin(), kPrivateSpecs.end(),
                                  [&](const PrivateSpec& spec) { return spec.name == name; });
  require(found != kPrivateSpecs.end(), "unknown private PRE barrier " + name);
  return *found;
}

std::shared_ptr<ov::op::v0::Result> resultFor(ov::Model& model, const char* name) {
  const auto output = model.output(name);
  auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(output.get_node_shared_ptr());
  require(result != nullptr, std::string("PRE output is not a Result: ") + name);
  require(result->input_value(0).get_index() == 0,
          std::string("PRE Result output index differs: ") + name);
  return result;
}

void validatePrivateNode(const std::shared_ptr<const ov::Node>& source,
                         const PrivateSpec& spec, const char* storageTarget) {
  require(source && std::string(source->get_type_name()) == spec.sourceType,
          std::string("private source type differs: ") + spec.name);
  require(source->get_output_element_type(0) == ov::element::bf16 &&
              source->get_output_partial_shape(0).is_static() &&
              source->get_output_shape(0) == kRows,
          std::string("private source shape/type differs: ") + spec.name);
  require(source->get_input_size() == spec.inputCount,
          std::string("private source input count differs: ") + spec.name);
  for (std::size_t i = 0; i < spec.inputCount; ++i) {
    const auto value = source->input_value(i);
    const auto input = value.get_node();
    require(std::string(input->get_type_name()) == spec.inputTypes[i] &&
                value.get_element_type() == spec.inputElementTypes[i] &&
                value.get_partial_shape().is_static() && value.get_shape() == spec.inputShapes[i],
            std::string("private source edge differs: ") + spec.name);
  }

  std::size_t storageCount = 0;
  std::size_t downstreamCount = 0;
  for (const auto& target : source->output(0).get_target_inputs()) {
    const auto type = std::string(target.get_node()->get_type_name());
    if (type == storageTarget) {
      ++storageCount;
    } else if (type == spec.downstreamType) {
      ++downstreamCount;
    } else {
      throw std::runtime_error("FlashVSR+ PRE: private source target differs: " +
                               std::string(spec.name) + " (" + type + ")");
    }
  }
  require(storageCount == 1 && downstreamCount == 1,
          std::string("private source target edges differ: ") + spec.name);
}

void validateNoReadValue(const ov::Model& model) {
  for (const auto& node : model.get_ordered_ops())
    require(std::string(node->get_type_name()) != "ReadValue", "state feedback is forbidden");
}

void validatePortableSource(const ov::Model& model) {
  validatePorts(model.inputs(), kInputs, "PRE input");
  const auto outputs = sourceOutputs();
  validatePorts(model.outputs(), outputs, "portable PRE output");
  require(model.get_variables().empty() && model.get_sinks().empty(),
          "portable PRE source must not contain state");
  validateNoReadValue(model);
  for (const auto& spec : kPrivateSpecs) {
    const auto result = model.output(spec.name).get_node_shared_ptr();
    const auto typed = std::dynamic_pointer_cast<const ov::op::v0::Result>(result);
    require(typed != nullptr,
            std::string("private PRE barrier is not a Result: ") + spec.name);
    validatePrivateNode(typed->input_value(0).get_node_shared_ptr(), spec, "Result");
  }
}

void validatePreparedPrivateSources(const ov::Model& model) {
  std::map<std::string, std::shared_ptr<ov::Node>> sources;
  std::set<std::string> ids;
  for (const auto& sink : model.get_sinks()) {
    const auto assign = std::dynamic_pointer_cast<ov::op::v6::Assign>(sink);
    require(assign != nullptr, "PRE sink is not Assign");
    const auto id = assign->get_variable_id();
    require(ids.insert(id).second, "duplicate PRE Assign state");
    require(assign->output(0).get_names().size() == 1,
            "PRE Assign output must have one name");
    const auto name = assign->output(0).get_any_name();
    require(std::find(kPrivateNames.begin(), kPrivateNames.end(), name) != kPrivateNames.end(),
            "unexpected PRE Assign output name");
    require(sources.emplace(name, assign->input_value(0).get_node_shared_ptr()).second,
            "duplicate PRE Assign output name");
    const auto spec = privateSpec(name);
    require(id == "plus_pre_state_" + name, "PRE Assign state ID differs");
    require(assign->get_input_element_type(0) == ov::element::bf16 &&
                assign->get_input_partial_shape(0).is_static() &&
                assign->get_input_shape(0) == kRows,
            "PRE Assign input ABI differs");
  }
  require(ids.size() == kStateIds.size() && sources.size() == kPrivateNames.size(),
          "PRE state count differs");
  for (const auto& spec : kPrivateSpecs)
    validatePrivateNode(sources.at(spec.name), spec, "Assign");
}

}  // namespace

void prepareFlashPreModel(ov::Model& model) {
  validatePortableSource(model);

  std::array<std::shared_ptr<ov::op::v0::Result>, kPrivateSpecs.size()> results{};
  for (std::size_t i = 0; i < kPrivateSpecs.size(); ++i)
    results[i] = resultFor(model, kPrivateSpecs[i].name);

  for (std::size_t i = 0; i < kPrivateSpecs.size(); ++i) {
    const auto& spec = kPrivateSpecs[i];
    ov::op::util::VariableInfo info;
    info.data_shape = ov::PartialShape(kRows);
    info.data_type = ov::element::bf16;
    info.variable_id = kStateIds[i];
    auto variable = std::make_shared<ov::op::util::Variable>(info);
    auto assign = std::make_shared<ov::op::v6::Assign>(results[i]->input_value(0), variable);
    assign->set_friendly_name("assign_" + std::string(kStateIds[i]));
    assign->output(0).set_names({spec.name});
    model.remove_result(results[i]);
    model.add_variables({variable});
    model.add_sinks({assign});
  }

  // ``remove_result`` removes the result from the model, while the shared
  // handles above would otherwise keep its old consumer edge alive during
  // topology validation.
  results.fill(nullptr);
  model.validate_nodes_and_infer_types();
  validateFlashPreModel(model);
}

void validateFlashPreModel(const ov::Model& model) {
  validatePorts(model.inputs(), kInputs, "PRE input");
  validatePorts(model.outputs(), kPublicOutputs, "PRE public output");
  require(model.get_variables().size() == kStateIds.size() &&
              model.get_sinks().size() == kStateIds.size(),
          "PRE requires six owning Assign states");

  std::set<std::string> variableIds;
  for (const auto& variable : model.get_variables()) {
    const auto info = variable->get_info();
    require(variableIds.insert(info.variable_id).second &&
                std::find(kStateIds.begin(), kStateIds.end(), info.variable_id) != kStateIds.end() &&
                info.data_type == ov::element::bf16 && info.data_shape.is_static() &&
                info.data_shape.to_shape() == kRows,
            "PRE variable ABI differs");
  }
  require(variableIds.size() == kStateIds.size(), "PRE variable IDs differ");
  validateNoReadValue(model);
  validatePreparedPrivateSources(model);
}

}  // namespace otc::ai
