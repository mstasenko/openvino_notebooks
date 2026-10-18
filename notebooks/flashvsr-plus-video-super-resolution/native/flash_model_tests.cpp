#include "flash_model.hpp"

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class NonParameterInput final : public ov::op::v0::Parameter {
 public:
  OPENVINO_OP("NotParameter", "opset1");

  NonParameterInput(const ov::element::Type& type, const ov::PartialShape& shape)
      : ov::op::v0::Parameter(type, shape) {}

  std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector&) const override {
    return std::make_shared<NonParameterInput>(get_element_type(), get_partial_shape());
  }
};

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
  throw std::runtime_error(label + ": malformed finite-check input was accepted");
}

std::shared_ptr<ov::op::v0::Parameter> parameter(const ov::element::Type& type,
                                                  const ov::PartialShape& shape,
                                                  const std::string& name) {
  auto result = std::make_shared<ov::op::v0::Parameter>(type, shape);
  result->set_friendly_name(name);
  result->output(0).get_tensor().set_names({name});
  return result;
}

std::shared_ptr<ov::op::v0::Result> result(const std::shared_ptr<ov::Node>& source,
                                           const std::string& name) {
  auto result = std::make_shared<ov::op::v0::Result>(source);
  result->set_friendly_name(name);
  result->output(0).get_tensor().set_names({name});
  return result;
}

std::shared_ptr<ov::Model> ordinaryModel() {
  const auto first = parameter(ov::element::bf16, ov::Shape{2, 4}, "data0");
  const auto second = parameter(ov::element::bf16, ov::Shape{2, 4}, "data1");
  return std::make_shared<ov::Model>(
      ov::ResultVector{result(first, "data0"), result(second, "data1")},
      ov::ParameterVector{first, second}, "ordinary_bf16_data");
}

std::shared_ptr<ov::Model> oneOutputModel(const ov::element::Type& type,
                                          const ov::PartialShape& shape,
                                          const std::string& name = "data0") {
  const auto input = parameter(type, shape, name);
  return std::make_shared<ov::Model>(ov::ResultVector{result(input, name)},
                                     ov::ParameterVector{input}, "one_data_result");
}

std::shared_ptr<ov::Model> aliasedModel() {
  auto model = oneOutputModel(ov::element::bf16, ov::Shape{2, 4});
  model->output(0).get_tensor().set_names({"data0", "alias"});
  return model;
}

std::size_t countType(const ov::Model& model, const char* typeName) {
  std::size_t count = 0;
  for (const auto& node : model.get_ordered_ops())
    count += std::string(node->get_type_name()) == typeName;
  return count;
}

void checkOriginalUnchanged(const ov::Model& original,
                            const std::vector<std::shared_ptr<ov::Node>>& results,
                            const std::vector<std::shared_ptr<ov::Node>>& sources) {
  require(original.outputs().size() == 2, "finite helper changed original output count");
  require(original.get_parameters().size() == 2,
          "finite helper changed original parameter count");
  for (std::size_t index = 0; index < original.outputs().size(); ++index) {
    const auto output = original.output(index);
    require(output.get_node_shared_ptr() == results[index],
            "finite helper replaced an original Result");
    require(output.get_node()->input_value(0).get_node_shared_ptr() == sources[index],
            "finite helper changed an original Result edge");
    require(output.get_element_type() == ov::element::bf16 &&
                output.get_shape() == ov::Shape{2, 4} && output.get_names().size() == 1,
            "finite helper changed an original output ABI");
  }
}

void checkGraphAbi(const ov::Model& check) {
  require(check.get_parameters().size() == 2, "finite checker parameter count differs");
  const std::set<std::string> expectedNames{"data0", "data1"};
  std::set<std::string> actualNames;
  for (const auto& input : check.inputs()) {
    require(input.get_element_type() == ov::element::bf16 &&
                input.get_partial_shape().is_static() && input.get_shape() == ov::Shape{2, 4},
            "finite checker input ABI differs");
    require(input.get_names().size() == 1, "finite checker input is not canonical");
    actualNames.insert(input.get_any_name());
  }
  require(actualNames == expectedNames, "finite checker input names differ");
  require(check.outputs().size() == 1, "finite checker output count differs");
  const auto output = check.output(0);
  require(output.get_element_type() == ov::element::f16 && output.get_shape().empty() &&
              output.get_names().size() == 1 && output.get_any_name() == "finite",
          "finite checker output ABI differs");
  require(std::dynamic_pointer_cast<const ov::op::v0::Result>(output.get_node_shared_ptr()) != nullptr,
          "finite checker output is not a Result");

  require(countType(check, "Abs") == 2, "finite checker does not inspect each BF16 input");
  require(countType(check, "LessEqual") == 2,
          "finite checker does not compare each BF16 input to BF16 max");
  require(countType(check, "ReduceMin") == 2,
          "finite checker does not reduce each input independently");
  require(countType(check, "Minimum") == 1,
          "finite checker does not combine scalar predicates with Minimum");
  require(countType(check, "Convert") == 2,
          "finite checker has an unexpected activation conversion");
  for (const auto& node : check.get_ordered_ops()) {
    if (std::string(node->get_type_name()) != "Convert") continue;
    require(node->input_value(0).get_element_type() == ov::element::boolean &&
                node->get_output_element_type(0) == ov::element::f16,
            "finite checker converts a BF16 activation instead of a Boolean mask");
  }
  const auto outputResult = std::dynamic_pointer_cast<const ov::op::v0::Result>(
      output.get_node_shared_ptr());
  const auto minimum = outputResult->input_value(0).get_node_shared_ptr();
  require(std::string(minimum->get_type_name()) == "Minimum" &&
              minimum->get_output_shape(0).empty() &&
              minimum->get_output_element_type(0) == ov::element::f16,
          "finite checker result is not the scalar F16 Minimum");
}

ov::Tensor makeInput(const std::array<std::uint16_t, 8>& bits) {
  ov::Tensor tensor(ov::element::bf16, ov::Shape{2, 4});
  auto* values = tensor.data<ov::bfloat16>();
  for (std::size_t index = 0; index < bits.size(); ++index)
    values[index] = ov::bfloat16::from_bits(bits[index]);
  return tensor;
}

std::uint16_t evaluate(const ov::Model& check,
                       const std::array<std::uint16_t, 8>& first,
                       const std::array<std::uint16_t, 8>& second) {
  ov::TensorVector outputs{ov::Tensor(ov::element::f16, ov::Shape{})};
  const ov::TensorVector inputs{makeInput(first), makeInput(second)};
  require(check.evaluate(outputs, inputs), "finite checker model.evaluate returned false");
  return outputs.front().data<const ov::float16>()[0].to_bits();
}

std::uint16_t evaluateSingle(const ov::Model& check,
                             const std::array<std::uint16_t, 8>& bits) {
  ov::TensorVector outputs{ov::Tensor(ov::element::f16, ov::Shape{})};
  const ov::TensorVector inputs{makeInput(bits)};
  require(check.evaluate(outputs, inputs),
          "single-input finite checker model.evaluate returned false");
  return outputs.front().data<const ov::float16>()[0].to_bits();
}

void checkSingleShapeOverload() {
  const auto check = otc::ai::makeFlashBf16OutputCheck(ov::Shape{2, 4});
  require(check->inputs().size() == 1 && check->input(0).get_any_name() == "data" &&
              check->input(0).get_element_type() == ov::element::bf16 &&
              check->input(0).get_shape() == ov::Shape{2, 4},
          "single-input finite checker input ABI differs");
  require(check->outputs().size() == 1 && check->output(0).get_any_name() == "finite" &&
              check->output(0).get_element_type() == ov::element::f16 &&
              check->output(0).get_shape().empty(),
          "single-input finite checker output ABI differs");
  require(countType(*check, "Abs") == 1 && countType(*check, "LessEqual") == 1 &&
              countType(*check, "ReduceMin") == 1 && countType(*check, "Minimum") == 0,
          "single-input finite checker graph is not one reduction");

  const std::array<std::uint16_t, 8> finite{
      0x0000, 0x8000, 0x0001, 0x8001, 0x4780, 0xc780, 0x7f7f, 0xff7f};
  require(evaluateSingle(*check, finite) == 0x3c00,
          "single-input checker rejected finite/subnormal/large BF16 values");
  for (const auto bad : {std::uint16_t{0x7f80}, std::uint16_t{0xff80},
                         std::uint16_t{0x7fc1}, std::uint16_t{0xffc1}}) {
    auto input = finite;
    input[3] = bad;
    require(evaluateSingle(*check, input) == 0,
            "single-input checker accepted infinity or NaN");
  }
}

void checkSingleShapeRejectsMalformed() {
  expectRejected([] { (void)otc::ai::makeFlashBf16OutputCheck(ov::Shape{}); },
                 "scalar checker shape");
  expectRejected([] { (void)otc::ai::makeFlashBf16OutputCheck(ov::Shape{0, 4}); },
                 "zero checker shape");
  expectRejected(
      [] {
        (void)otc::ai::makeFlashBf16OutputCheck(
            ov::Shape{std::numeric_limits<std::size_t>::max(), 2});
      },
      "overflow checker shape");
  expectRejected(
      [] { (void)otc::ai::makeFlashBf16OutputCheck(
          ov::Shape{std::numeric_limits<std::size_t>::max() / 2 + 1}); },
      "overflow checker byte extent");
}

void checkEvaluation(const ov::Model& check) {
  const std::array<std::uint16_t, 8> finite{
      0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x4780, 0xc780};
  require(evaluate(check, finite, finite) == 0x3c00,
          "finite checker rejected signed-zero/finite BF16 values");

  for (const auto bad : {std::uint16_t{0x7f80}, std::uint16_t{0xff80},
                         std::uint16_t{0x7fc1}, std::uint16_t{0xffc1},
                         std::uint16_t{0x7f81}, std::uint16_t{0xff81}}) {
    auto first = finite;
    first[3] = bad;
    require(evaluate(check, first, finite) == 0,
            "finite checker accepted an infinity or NaN in first input");
    auto second = finite;
    second[4] = bad;
    require(evaluate(check, finite, second) == 0,
            "finite checker accepted an infinity or NaN in second input");
  }
}

void checkRejectsMalformedModels() {
  expectRejected(
      [] {
        auto empty = std::make_shared<ov::Model>(ov::ResultVector{}, ov::ParameterVector{},
                                                 "empty");
        (void)otc::ai::makeFlashBf16OutputCheck(*empty);
      },
      "empty result list");
  expectRejected(
      [] { (void)otc::ai::makeFlashBf16OutputCheck(
          *oneOutputModel(ov::element::f16, ov::Shape{2, 4})); },
      "non-BF16 result");
  expectRejected(
      [] { (void)otc::ai::makeFlashBf16OutputCheck(
          *oneOutputModel(ov::element::bf16, ov::PartialShape{1, ov::Dimension::dynamic()})); },
      "dynamic result");
  expectRejected(
      [] { (void)otc::ai::makeFlashBf16OutputCheck(
          *oneOutputModel(ov::element::bf16, ov::Shape{0, 4})); },
      "zero-extent result");
  expectRejected([] { (void)otc::ai::makeFlashBf16OutputCheck(*aliasedModel()); },
                  "ambiguous output names");
  expectRejected(
      [] {
        const auto first = parameter(ov::element::bf16, ov::Shape{2, 4}, "data0");
        const auto second = parameter(ov::element::bf16, ov::Shape{2, 4}, "data1");
        auto r0 = result(first, "finite");
        auto r1 = result(second, "data1");
        auto model = std::make_shared<ov::Model>(ov::ResultVector{r0, r1},
                                                  ov::ParameterVector{first, second}, "bad");
        (void)otc::ai::makeFlashBf16OutputCheck(*model);
      },
      "reserved finite output name");
}

void checkStructuralRejectsNonParameterInput() {
  const auto first = std::make_shared<NonParameterInput>(ov::element::bf16, ov::Shape{2, 4});
  first->output(0).get_tensor().set_names({"data0"});
  const auto second = parameter(ov::element::bf16, ov::Shape{2, 4}, "data1");
  const auto model = std::make_shared<ov::Model>(
      ov::ResultVector{result(first, "data0"), result(second, "data1")},
      ov::ParameterVector{first, second}, "non_parameter_input");
  const otc::ai::FlashPorts inputs{
      {"data0", {ov::element::bf16, ov::Shape{2, 4}}},
      {"data1", {ov::element::bf16, ov::Shape{2, 4}}}};
  const otc::ai::FlashPorts outputs = inputs;
  expectRejected(
      [&] { otc::ai::validateFlashStructuralModel(*model, inputs, outputs); },
      "non-Parameter input");
}

}  // namespace

int main() {
  try {
    const auto original = ordinaryModel();
    const std::vector<std::shared_ptr<ov::Node>> results{
        original->output("data0").get_node_shared_ptr(),
        original->output("data1").get_node_shared_ptr()};
    const std::vector<std::shared_ptr<ov::Node>> sources{
        original->output("data0").get_node()->input_value(0).get_node_shared_ptr(),
        original->output("data1").get_node()->input_value(0).get_node_shared_ptr()};

    const auto check = otc::ai::makeFlashBf16OutputCheck(*original);
    checkOriginalUnchanged(*original, results, sources);
    checkGraphAbi(*check);
    checkEvaluation(*check);
    checkSingleShapeOverload();
    checkSingleShapeRejectsMalformed();
    checkRejectsMalformedModels();
    checkStructuralRejectsNonParameterInput();
    std::cout << "flash_model_finite_checker=pass model_evaluate=pass core_calls=0 "
                 "device_calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "flash_model_finite_checker=fail: " << error.what() << '\n';
    return 1;
  }
}
