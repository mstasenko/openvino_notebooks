#include "seedvr_blob.hpp"
#include <openvino/openvino.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <cassert>
#include <iostream>

int main() {
  // Tiny synthetic CPU-only model: prove wrapped exporter failure does not
  // destroy the already compiled model. No SeedVR weights or GPU plugin query.
  auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0F});
  auto add = std::make_shared<ov::op::v1::Add>(parameter, constant);
  auto result = std::make_shared<ov::op::v0::Result>(add);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
  ov::Core core;
  auto compiled = core.compile_model(model, "CPU", {{"INFERENCE_NUM_THREADS", 1},
      {"NUM_STREAMS", std::string("1")}, {"INFERENCE_PRECISION_HINT", ov::element::f32}});
  otc::ai::detail::BoundedBlobBuffer buffer(2);
  std::ostream stream(&buffer);
  stream.exceptions(std::ios::badbit | std::ios::failbit);
  bool rejected = false;
  try {
    compiled.export_model(stream);
  } catch (const std::exception&) {
    // This is the same narrow expected-buffer flag used by the private helper.
    if (!buffer.failed()) throw;
    rejected = true;
  }
  assert(rejected);
  float input = 3.0F;
  auto request = compiled.create_infer_request();
  request.set_input_tensor(ov::Tensor(ov::element::f32, {1}, &input));
  request.infer();
  assert(request.get_output_tensor().data<const float>()[0] == 5.0F);
  std::cout << "PASS: wrapped bounded export rejected; original compiled model remains usable; tiny CPU-only fixture\n";
}
