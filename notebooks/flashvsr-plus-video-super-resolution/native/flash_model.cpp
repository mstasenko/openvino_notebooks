#include "flash_model.hpp"

#include <openvino/op/abs.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/less_eq.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_min.hpp>
#include <openvino/op/result.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace otc::ai {

namespace {

void requireStructural(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(std::string("FlashVSR+ structural model: ") + message);
}

void validateFiniteShape(const ov::Shape& shape) {
  if (shape.empty()) throw std::invalid_argument("FlashVSR+ finite check requires non-scalar data");
  std::size_t elements = 1;
  for (const auto dimension : shape) {
    if (!dimension) throw std::invalid_argument("FlashVSR+ finite check requires nonempty data");
    if (elements > std::numeric_limits<std::size_t>::max() / dimension)
      throw std::invalid_argument("FlashVSR+ finite check shape element count overflows");
    elements *= dimension;
  }
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))
    throw std::invalid_argument("FlashVSR+ finite check BF16 byte extent overflows");
}

std::shared_ptr<ov::op::v0::Result> makeFiniteResult(const ov::ParameterVector& parameters) {
  // BF16 data stays in its original domain, including values beyond FP16's
  // range. Only the exact Boolean 0/1 mask becomes F16. ReduceMin uses the
  // plugin's staged scalar reduction; ReduceLogicalAnd scans a full UHD cache
  // monolithically and is dramatically slower on this pinned GPU build.
  const std::array<std::uint16_t, 1> maximum{{0x7f7f}};
  const auto bound = std::make_shared<ov::op::v0::Constant>(ov::element::bf16, ov::Shape{}, maximum.data());
  std::shared_ptr<ov::Node> allFinite;
  for (const auto& parameter : parameters) {
    const auto absolute = std::make_shared<ov::op::v0::Abs>(parameter);
    const auto finite = std::make_shared<ov::op::v1::LessEqual>(absolute, bound);
    const auto mask = std::make_shared<ov::op::v0::Convert>(finite, ov::element::f16);
    std::vector<std::int64_t> axes;
    for (std::size_t axis = 0; axis < parameter->get_shape().size(); ++axis)
      axes.push_back(static_cast<std::int64_t>(axis));
    const auto indices = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
    const auto reduced = std::make_shared<ov::op::v1::ReduceMin>(mask, indices, false);
    if (allFinite) allFinite = std::make_shared<ov::op::v1::Minimum>(allFinite, reduced);
    else allFinite = reduced;
  }
  if (!allFinite) throw std::invalid_argument("FlashVSR+ finite check requires data Results");
  const auto result = std::make_shared<ov::op::v0::Result>(allFinite);
  result->set_friendly_name("finite");
  result->output(0).get_tensor().set_names({"finite"});
  return result;
}

std::shared_ptr<ov::Model> makeFiniteCheckModel(ov::ParameterVector parameters) {
  auto result = makeFiniteResult(parameters);
  return std::make_shared<ov::Model>(ov::ResultVector{std::move(result)},
                                      std::move(parameters), "flash_bf16_output_check");
}

}  // namespace

void validateFlashStructuralModel(const ov::Model& model,
                                  const FlashPorts& expectedInputs,
                                  const FlashPorts& expectedOutputs) {
  validateFlashPorts(model.inputs(), expectedInputs);
  validateFlashPorts(model.outputs(), expectedOutputs);

  for (const auto& input : model.inputs())
    requireStructural(input.get_node()->get_type_name() == std::string("Parameter"),
                      "all inputs must be Parameters");
  for (const auto& output : model.outputs())
    requireStructural(output.get_node()->get_type_name() == std::string("Result"),
                      "all outputs must be ordinary Results");
  requireStructural(model.get_variables().empty() && model.get_sinks().empty(),
                    "hidden state is not allowed");

  std::size_t constantBytes = 0;
  for (const auto& node : model.get_ordered_ops()) {
    const auto type = std::string(node->get_type_name());
    requireStructural(type != "ReadValue" && type != "Assign",
                      "state operations are not allowed");
    for (const auto& output : node->outputs()) {
      if (output.get_element_type() != ov::element::f16) continue;
      // These custom operations carry BF16 bits in F16-shaped storage, never
      // numeric FP16 values. Keep both ends inside that transport boundary.
      requireStructural(type == "OtcCrossQPack" || type == "OtcCrossKVPack" ||
                            type == "OtcCrossPreciseXmx",
                        "numeric FP16 intermediates are not allowed");
      for (const auto& input : output.get_target_inputs()) {
        const std::string consumer = input.get_node()->get_type_name();
        requireStructural(consumer == "OtcCrossPreciseXmx" || consumer == "OtcCrossContextUnpack",
                          "raw BF16 transport escapes into numeric FP16 operations");
      }
    }
    if (const auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
      const auto bytes = constant->get_byte_size();
      requireStructural(bytes <= 4U * 1024U &&
                            constantBytes <= 64U * 1024U - bytes,
                        "structural Constant exceeds the fixed bound");
      constantBytes += bytes;
    }
  }
}

std::shared_ptr<ov::Model> makeFlashBf16OutputCheck(const ov::Model& dataModel) {
  ov::ParameterVector parameters;
  std::set<std::string> names;
  for (const auto& output : dataModel.outputs()) {
    if (output.get_names().size() != 1 || output.get_any_name() == "finite" ||
        !names.insert(output.get_any_name()).second ||
        !std::dynamic_pointer_cast<const ov::op::v0::Result>(output.get_node_shared_ptr()) ||
        output.get_element_type() != ov::element::bf16 ||
        !output.get_partial_shape().is_static())
      throw std::invalid_argument("FlashVSR+ finite check requires canonical static BF16 Results");
    const auto shape = output.get_shape();
    validateFiniteShape(shape);
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, shape);
    parameter->set_friendly_name(output.get_any_name());
    parameter->output(0).get_tensor().set_names(output.get_names());
    parameters.push_back(std::move(parameter));
  }
  return makeFiniteCheckModel(std::move(parameters));
}

std::shared_ptr<ov::Model> makeFlashBf16OutputCheck(const ov::Shape& shape) {
  validateFiniteShape(shape);
  auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, shape);
  data->set_friendly_name("data");
  data->output(0).get_tensor().set_names({"data"});
  return makeFiniteCheckModel(ov::ParameterVector{std::move(data)});
}

} // namespace otc::ai
