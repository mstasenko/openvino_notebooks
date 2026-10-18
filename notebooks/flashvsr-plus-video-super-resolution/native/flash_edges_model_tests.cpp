#include "flash_edges_model.hpp"

#include "seedvr_assets.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/assign.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/util/variable.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using otc::ai::FlashEdgesRole;

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
  throw std::runtime_error(label + ": malformed edge model was accepted");
}

unsigned hexDigit(char value) {
  if (value >= '0' && value <= '9') return static_cast<unsigned>(value - '0');
  if (value >= 'a' && value <= 'f') return static_cast<unsigned>(value - 'a' + 10);
  if (value >= 'A' && value <= 'F') return static_cast<unsigned>(value - 'A' + 10);
  throw std::runtime_error("invalid asset SHA256 literal");
}

otc::ai::SeedVrAssetPin pin(const char* name, std::uint64_t bytes, const char* sha) {
  require(std::string(sha).size() == 64, "asset SHA256 literal length differs");
  otc::ai::SeedVrAssetPin result{name, bytes, {}};
  for (std::size_t i = 0; i < result.sha256.size(); ++i)
    result.sha256[i] = static_cast<std::uint8_t>((hexDigit(sha[2 * i]) << 4) |
                                                 hexDigit(sha[2 * i + 1]));
  return result;
}

struct LoadedModel {
  otc::ai::SeedVrOwnedIr ir;
  std::shared_ptr<ov::Model> model;
};

LoadedModel load(const std::filesystem::path& root, FlashEdgesRole role) {
  const auto prefixXml = pin(
      "flash_edges_prefix_bf16.xml", 8874,
      "a1ab4509d9c4a1b67a1ec54e1d555ee1ceec30e2930c7600326dda9c396c1004");
  const auto prefixBin = pin(
      "flash_edges_prefix_bf16.bin", 664,
      "69bf6dce98cf98395030464b350ca048a648b7617ca864ced452b1db3d209bd1");
  const auto suffixXml = pin(
      "flash_edges_suffix_bf16.xml", 17778,
      "20900cf95b1b32ab3b21d879a1f56210a4918b755acc34fb96b85e0c68f87fe5");
  const auto suffixBin = pin(
      "flash_edges_suffix_bf16.bin", 328,
      "9bec50f4ae9fa668adee56efac658baba7e6f2a73301bc24b6fb613238cb39bb");
  const auto& xml = role == FlashEdgesRole::prefix ? prefixXml : suffixXml;
  const auto& bin = role == FlashEdgesRole::prefix ? prefixBin : suffixBin;
  auto ir = otc::ai::readSeedVrIr(root, xml, bin);
  ov::Core core;
  const ov::Tensor structural(ov::element::u8, ov::Shape{ir.binBytes()}, ir.binData());
  auto model = core.read_model(ir.xml(), structural);
  return {std::move(ir), std::move(model)};
}

std::vector<std::string> dataNames(FlashEdgesRole role) {
  if (role == FlashEdgesRole::prefix)
    return {"plus_f4_3840x2176_tokens_out", "plus_f4_3840x2176_t_out",
            "plus_f4_3840x2176_t_mod_out", "plus_f4_3840x2176_embedding_out"};
  if (role == FlashEdgesRole::suffix) return {"plus_f4_3840x2176_out"};
  throw std::invalid_argument("invalid edge role");
}

template <class Mutator>
void rejectMutation(const ov::Model& source, FlashEdgesRole role,
                    const std::string& label, Mutator mutate) {
  auto malformed = source.clone();
  // Mutator failures are test failures; only validator rejection is expected.
  mutate(*malformed);
  expectRejected(
      [&] { otc::ai::validateFlashEdgesModel(*malformed, role, 2176, 3840); }, label);
}

std::shared_ptr<ov::op::v0::Parameter> parameter(ov::Model& model, const std::string& name) {
  return std::dynamic_pointer_cast<ov::op::v0::Parameter>(model.input(name).get_node_shared_ptr());
}

void checkMalformed(const ov::Model& source, FlashEdgesRole role) {
  const auto firstInput = role == FlashEdgesRole::prefix ?
      "plus_f4_3840x2176_x" : "plus_f4_3840x2176_tokens_in";
  const auto firstOutput = dataNames(role).front();
  const auto weight = role == FlashEdgesRole::prefix ?
      "weight__patch_embedding_weight" : "weight__head_proj_weight";

  rejectMutation(source, role, "renamed input", [&](ov::Model& model) {
    parameter(model, firstInput)->output(0).get_tensor().set_names({"wrong_input"});
  });
  rejectMutation(source, role, "wrong input shape", [&](ov::Model& model) {
    auto input = parameter(model, firstInput);
    auto shape = input->get_shape();
    shape.back() -= 1;
    input->set_partial_shape(shape);
    input->validate_and_infer_types();
  });
  rejectMutation(source, role, "wrong learned type", [&](ov::Model& model) {
    auto input = parameter(model, weight);
    input->set_element_type(ov::element::f16);
    input->validate_and_infer_types();
  });
  rejectMutation(source, role, "extra input", [](ov::Model& model) {
    auto extra = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::Shape{1});
    extra->output(0).get_tensor().set_names({"unexpected_input"});
    model.add_parameters({extra});
  });
  rejectMutation(source, role, "renamed output", [&](ov::Model& model) {
    model.output(firstOutput).get_tensor().set_names({"wrong_output"});
  });
  rejectMutation(source, role, "hidden state", [&](ov::Model& model) {
    const auto sourceValue = model.output(firstOutput).get_node()->input_value(0);
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
        ov::PartialShape(sourceValue.get_shape()), sourceValue.get_element_type(),
        "unexpected_edge_state"});
    auto assign = std::make_shared<ov::op::v6::Assign>(sourceValue, variable);
    model.add_variables({variable});
    model.add_sinks({assign});
  });
  rejectMutation(source, role, "FP16 intermediate", [&](ov::Model& model) {
    auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output(firstOutput).get_node_shared_ptr());
    require(result != nullptr, "edge output is not a Result");
    auto toF16 = std::make_shared<ov::op::v0::Convert>(result->input_value(0), ov::element::f16);
    auto back = std::make_shared<ov::op::v0::Convert>(toF16, ov::element::bf16);
    result->input(0).replace_source_output(back);
    model.validate_nodes_and_infer_types();
  });
  rejectMutation(source, role, "oversized Constant", [&](ov::Model& model) {
    const auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
        model.output(firstOutput).get_node_shared_ptr());
    std::vector<float> values(2049, 0.0f);
    const auto constant = ov::op::v0::Constant::create(ov::element::bf16, {2049}, values);
    const auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    const auto sum = std::make_shared<ov::op::v1::ReduceSum>(constant, axes, false);
    const auto add = std::make_shared<ov::op::v1::Add>(result->input_value(0), sum);
    result->input(0).replace_source_output(add);
    model.validate_nodes_and_infer_types();
  });

}

void check(const LoadedModel& loaded, FlashEdgesRole role) {
  const auto& model = *loaded.model;
  otc::ai::validateFlashEdgesModel(model, role, 2176, 3840);
  require(model.get_parameters().size() == 5U,
          "edge Parameter count differs");
  for (const auto& input : model.inputs()) {
    if (input.get_any_name().rfind("weight__", 0) == 0)
      require(input.get_element_type() == ov::element::bf16,
              "edge learned input is not BF16");
  }
  for (const auto [height, width] : {std::pair{2160U, 3840U}, std::pair{2175U, 3840U},
                                      std::pair{2176U, 3839U}}) {
    expectRejected([&] {
      otc::ai::validateFlashEdgesModel(model, role, height, width);
    }, "unadmitted edge geometry");
  }
  expectRejected([&] {
    otc::ai::validateFlashEdgesModel(model, static_cast<FlashEdgesRole>(99), 2176, 3840);
  }, "unknown edge role");
  checkMalformed(model, role);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: flash-edges-model-tests STRUCTURAL_ASSETS");
    const std::filesystem::path root(argv[1]);
    const auto prefix = load(root, FlashEdgesRole::prefix);
    const auto suffix = load(root, FlashEdgesRole::suffix);
    check(prefix, FlashEdgesRole::prefix);
    check(suffix, FlashEdgesRole::suffix);
    std::cout << "flash_edges_model_validation=pass device_queries=0 compile_calls=0 "
                 "infer_calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "flash_edges_model_validation=fail: " << error.what() << '\n';
    return 1;
  }
}
