#include "flash_post_session.hpp"

#include "flash_model.hpp"
#include "flash_ops.hpp"
#include "flash-assets/flash_post_assets.hpp"
#include "verified_kernel_files.hpp"

#include <openvino/op/assign.hpp>
#include <openvino/op/result.hpp>

#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace otc::ai {
namespace {

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error("FlashVSR+ POST: " + message);
}

const ov::Shape kRows{1, 4096, 1536};
const ov::Shape kContext{12, 16, 128, 128};
const ov::Shape kCross{1, 512, 1536};
const ov::Shape kGate{1, 1, 1536};

struct StateContract {
  const char* outputName;
  const char* variableId;
};

const std::array<StateContract, 2> kStateContracts{{
    {"ffn2_linear_storage", "plus_post_ffn2_state_ffn2_linear_storage"},
    {"ffn2_gate_storage", "plus_post_ffn2_state_ffn2_gate_storage"}}};

FlashPorts inputPorts() {
  FlashPorts ports{{"x", {ov::element::bf16, kRows}},
          {"self_context_0", {ov::element::bf16, kContext}},
          {"self_context_1", {ov::element::bf16, kContext}},
          {"cross_k", {ov::element::bf16, kCross}},
          {"cross_v", {ov::element::bf16, kCross}},
          {"gate_msa", {ov::element::bf16, kGate}},
          {"shift_mlp", {ov::element::bf16, kGate}},
          {"scale_mlp", {ov::element::bf16, kGate}},
          {"gate_mlp", {ov::element::bf16, kGate}}};
  for (const auto& weight : kFlashPostWeights)
    require(ports.emplace(weight.port_name,
                           std::make_pair(ov::element::bf16, flashWeightShape(weight))).second,
            "duplicate weight name");
  return ports;
}

FlashPorts portableOutputPorts() {
  FlashPorts ports{{"x_out", {ov::element::bf16, kRows}}};
  for (const auto& state : kStateContracts)
    ports.emplace(state.outputName, std::make_pair(ov::element::bf16, kRows));
  return ports;
}

const std::array<SeedVrAssetPin, 30>& layerPins(FlashMode mode) {
  switch (mode) {
    case FlashMode::turbo: return kFlashPostTurboBins;
    case FlashMode::base: return kFlashPostBaseBins;
  }
  throw std::invalid_argument("FlashVSR+ mode differs");
}

VerifiedKernelSpec kernelSpec() {
  return {kFlashPostConfig,
          {VerifiedKernelSourceSpec{kFlashPostBridgeSource, 3},
           VerifiedKernelSourceSpec{kFlashPostSparseSource, 1}}, 2};
}

} // namespace

void prepareFlashPostModel(ov::Model& model) {
  // The source is stateless at this point, so use the same structural guard as
  // the other fixed graphs before mutating it into the stateful POST form.
  validateFlashStructuralModel(model, inputPorts(), portableOutputPorts());
  const auto value = [&](const char* name) {
    const auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output(name).get_node_shared_ptr());
    require(result && result->input_value(0).get_index() == 0,
            "portable source must expose canonical Result nodes");
    return result->input_value(0).get_node_shared_ptr();
  };
  const auto linear = value("ffn2_linear_storage");
  const auto gate = value("ffn2_gate_storage");
  const auto output = value("x_out");
  const auto consumes = [](const std::shared_ptr<ov::Node>& node,
                           const std::shared_ptr<ov::Node>& source) {
    std::size_t count = 0;
    for (const auto& input : node->input_values())
      count += input.get_node_shared_ptr() == source;
    return count;
  };
  require(std::string(linear->get_type_name()) == "Add" &&
              std::string(gate->get_type_name()) == "Multiply" &&
              std::string(output->get_type_name()) == "Add" &&
              consumes(gate, linear) == 1 && consumes(output, gate) == 1,
          "portable FFN2 chain differs");
  std::size_t matmuls = 0;
  for (const auto& input : linear->input_values())
    matmuls += std::string(input.get_node()->get_type_name()) == "MatMul";
  require(matmuls == 1, "portable FFN2 linear Add must consume one MatMul");
  for (const auto& state : kStateContracts) {
    const auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output(state.outputName).get_node_shared_ptr());
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
        ov::PartialShape(kRows), ov::element::bf16, state.variableId});
    auto assign = std::make_shared<ov::op::v6::Assign>(result->input_value(0), variable);
    assign->set_friendly_name("assign_" + std::string(state.variableId));
    assign->output(0).set_names({state.outputName});
    model.remove_result(result);
    model.add_variables({variable});
    model.add_sinks({assign});
  }
  // The computed source nodes and their BF16 types are unchanged. Only storage
  // ownership changes; both states are overwritten on every complete call.
  model.validate_nodes_and_infer_types();
  validateFlashPostModel(model);
}

void validateFlashPostModel(const ov::Model& model) {
  validateFlashPorts(model.inputs(), inputPorts());
  validateFlashPorts(model.outputs(), FlashPorts{{"x_out", {ov::element::bf16, kRows}}});
  std::set<std::string> expected;
  for (const auto& state : kStateContracts) expected.insert(state.variableId);
  std::set<std::string> variables, assigns;
  require(model.get_variables().size() == 2 && model.get_sinks().size() == 2,
          "exactly two owning Assign states required");
  for (const auto& variable : model.get_variables()) {
    const auto info = variable->get_info();
    require(variables.insert(info.variable_id).second && info.data_type == ov::element::bf16 &&
                info.data_shape.is_static() && info.data_shape.to_shape() == kRows,
            "state variable ABI differs");
  }
  for (const auto& sink : model.get_sinks()) {
    const auto assign = std::dynamic_pointer_cast<ov::op::v6::Assign>(sink);
    require(assign && assigns.insert(assign->get_variable_id()).second &&
                assign->get_input_element_type(0) == ov::element::bf16 &&
                assign->get_input_partial_shape(0).is_static() &&
                assign->get_input_shape(0) == kRows, "Assign ABI differs");
  }
  require(variables == expected && assigns == expected, "state IDs differ");
  std::map<std::string, std::size_t> custom;
  for (const auto& node : model.get_ordered_ops()) {
    const std::string kind = node->get_type_name();
    require(kind != "ReadValue", "state feedback is forbidden");
    if (kind.rfind("Otc", 0) == 0) ++custom[kind];
  }
  require(custom == std::map<std::string, std::size_t>{{"OtcCrossQPack", 2},
              {"OtcCrossKVPack", 4}, {"OtcCrossContextUnpack", 2}, {"OtcCrossPreciseXmx", 2}},
          "precise bridge custom-operation set differs");
}

class FlashPostSession::Impl final {
 public:
  Impl(ov::Core& core, ov::RemoteContext context, std::filesystem::path assets, FlashMode mode)
      : assets_(std::move(assets)), mode_(mode),
        kernels_(VerifiedKernelFiles::open(assets_, kernelSpec())),
        structural_(readSeedVrIr(assets_, kFlashPostXml, layerPins(mode_)[0], flashWeightSpans(kFlashPostWeights))) {
    registerFlashOps(core);
    const ov::Tensor bin(ov::element::u8, {structural_.binBytes()}, structural_.binData());
    auto model = core.read_model(structural_.xml(), bin);
    prepareFlashPostModel(*model);
    stage_ = std::make_unique<FlashGpuStage>(core, std::move(context), model,
        flashStageWeights(kFlashPostWeights), FlashPrecision::source, kernels_.configPath());
    stage_->bind(structural_);
  }

  void bindLayer(std::size_t layer) {
    stage_->invalidate();
    const auto& pins = layerPins(mode_);
    require(layer < pins.size(), "layer index is outside [0,30)");
    // The initial owner remains alive for IR Constants; later layer owners
    // exist only through their synchronous upload, not as a second host cache.
    auto owner = readSeedVrIr(assets_, kFlashPostXml, pins[layer], flashWeightSpans(kFlashPostWeights));
    stage_->bind(owner);
  }

  void run(const FlashPostInputs& inputs, const ov::Tensor& output) {
    stage_->run({{"x", inputs.x}, {"self_context_0", inputs.selfContext0},
        {"self_context_1", inputs.selfContext1}, {"cross_k", inputs.crossK},
        {"cross_v", inputs.crossV}, {"gate_msa", inputs.gateMsa},
        {"shift_mlp", inputs.shiftMlp}, {"scale_mlp", inputs.scaleMlp},
        {"gate_mlp", inputs.gateMlp}}, {{"x_out", output}});
  }

  void suspend() noexcept { stage_->suspend(); }

 private:
  std::filesystem::path assets_;
  FlashMode mode_;
  VerifiedKernelFiles kernels_;
  SeedVrOwnedIr structural_;
  std::unique_ptr<FlashGpuStage> stage_;
};

FlashPostSession::FlashPostSession(ov::Core& core, ov::RemoteContext context,
                                 std::filesystem::path assets, FlashMode mode)
    : impl_(std::make_unique<Impl>(core, std::move(context), std::move(assets), mode)) {}
FlashPostSession::~FlashPostSession() = default;
FlashPostSession::FlashPostSession(FlashPostSession&&) noexcept = default;
FlashPostSession& FlashPostSession::operator=(FlashPostSession&&) noexcept = default;
void FlashPostSession::bindLayer(std::size_t layer) { impl_->bindLayer(layer); }
void FlashPostSession::run(const FlashPostInputs& inputs, const ov::Tensor& output) {
  impl_->run(inputs, output);
}
void FlashPostSession::suspend() noexcept { impl_->suspend(); }

} // namespace otc::ai
