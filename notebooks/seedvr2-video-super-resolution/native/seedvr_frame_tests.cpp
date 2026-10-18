#include "seedvr_frame.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace otc::ai;

template <typename F>
void rejects(F action) {
  bool rejected = false;
  try { action(); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

void temporalRegressions() {
  const auto shape = seedVrFrameShape(17, 19, 5);
  assert(shape.inputFrames == 5 && shape.latentFrames == 2 &&
         shape.tokens == 8 && shape.latentPixels == 16);
  for (int frames : {0, 2, 3, 4, 6, std::numeric_limits<int>::max()}) {
    rejects([&] { seedVrFrameShape(17, 19, frames); });
  }
  rejects([] { seedVrFrameShape(std::numeric_limits<int>::max(), 19, 5); });
  rejects([] { seedVrFrameShape(16384, 16384, 5); });
  assert(seedVrFrameShape(3840, 2160, 5).tokens == 64800);

  std::vector<std::uint8_t> rgb(5U * 17U * 19U * 3U);
  for (std::size_t time = 0; time < 5; ++time) {
    for (std::size_t y = 0; y < 19; ++y) {
      for (std::size_t x = 0; x < 17; ++x) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
          rgb[((time * 19 + y) * 17 + x) * 3 + channel] =
              static_cast<std::uint8_t>((time * 37 + channel * 23 + y + x) % 256);
        }
      }
    }
  }
  const auto normalized = seedVrNormalizeRgb(rgb, shape);
  assert(normalized.size() == 3U * 5U * 32U * 32U);
  for (std::size_t channel = 0; channel < 3; ++channel) {
    for (std::size_t time = 0; time < 5; ++time) {
      for (std::size_t y = 0; y < 32; ++y) {
        for (std::size_t x = 0; x < 32; ++x) {
          const float expected = y < 19 && x < 17
              ? static_cast<float>(rgb[((time * 19 + y) * 17 + x) * 3 + channel]) /
                    127.5F - 1.0F
              : -1.0F;
          assert(normalized[(channel * 5 + time) * 1024 + y * 32 + x] == expected);
        }
      }
    }
  }
  assert(seedVrDecodedRgb(normalized, shape) == rgb);
  rejects([&] { seedVrNormalizeRgb(std::vector<std::uint8_t>(rgb.size() - 1), shape); });
  rejects([&] { seedVrDecodedRgb(std::vector<float>(normalized.size() - 1), shape); });
  auto badShape = shape;
  badShape.latentFrames = 1;
  rejects([&] { seedVrNormalizeRgb(rgb, badShape); });
  auto nonfinite = normalized;
  nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { seedVrDecodedRgb(nonfinite, shape); });

  const auto tileShape = seedVrFrameShape(80, 64, 5);
  const VaeOverlapTile tile{3, 2, 4, 3};
  for (bool encode : {false, true}) {
    const std::size_t frames = encode ? 5 : 2;
    const std::size_t channels = encode ? 3 : 16;
    const std::size_t scale = encode ? 8 : 1;
    const std::size_t width = 10 * scale;
    const std::size_t height = 8 * scale;
    std::vector<float> values(channels * frames * width * height);
    for (std::size_t channel = 0; channel < channels; ++channel) {
      for (std::size_t time = 0; time < frames; ++time) {
        for (std::size_t y = 0; y < height; ++y) {
          for (std::size_t x = 0; x < width; ++x) {
            values[((channel * frames + time) * height + y) * width + x] =
                static_cast<float>(channel * 100000 + time * 10000 + y * 100 + x);
          }
        }
      }
    }
    const auto cropped = seedVrVaeTileInput(values, tileShape, tile, encode);
    assert(cropped.size() == channels * frames * 4 * scale * 3 * scale);
    for (std::size_t channel = 0; channel < channels; ++channel) {
      for (std::size_t time = 0; time < frames; ++time) {
        for (std::size_t y = 0; y < 3 * scale; ++y) {
          for (std::size_t x = 0; x < 4 * scale; ++x) {
            assert(cropped[((channel * frames + time) * 3 * scale + y) * 4 * scale + x] ==
                   static_cast<float>(channel * 100000 + time * 10000 +
                                      (2 * scale + y) * 100 + 3 * scale + x));
          }
        }
      }
    }
    rejects([&] { seedVrVaeTileInput({}, tileShape, tile, encode); });
    rejects([&] {
      seedVrVaeTileInput(values, tileShape,
                        {std::numeric_limits<int>::max(), 2, 4, 3}, encode);
    });
  }

  const auto pixels = shape.latentPixels * 2U;
  std::vector<float> mean(16 * pixels);
  std::vector<float> noise(16 * pixels);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    for (std::size_t channel = 0; channel < 16; ++channel) {
      mean[channel * pixels + pixel] = static_cast<float>(channel * 100 + pixel);
      noise[pixel * 16 + channel] = static_cast<float>(pixel * 100 + channel);
    }
  }
  std::vector<float> condition(pixels * 33U);
  seedVrLatentCondition(mean, noise, shape, condition);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    for (std::size_t channel = 0; channel < 16; ++channel) {
      assert(condition[channel * pixels + pixel] == noise[pixel * 16 + channel]);
      assert(condition[(channel + 16) * pixels + pixel] ==
             mean[channel * pixels + pixel] * 0.9152F);
    }
    assert(condition[32 * pixels + pixel] == 1.0F);
  }
  rejects([&] { seedVrLatentCondition({}, noise, shape, condition); });
  std::vector<float> wrongCondition;
  rejects([&] { seedVrLatentCondition(mean, noise, shape, wrongCondition); });
  noise[0] = std::numeric_limits<float>::infinity();
  rejects([&] { seedVrLatentCondition(mean, noise, shape, condition); });
  const auto singleNoise = seedVrFixedNoise(seedVrFrameShape(17, 19), 42);
  const auto temporalNoise = seedVrFixedNoise(shape, 42);
  assert(temporalNoise.size() == 2 * singleNoise.size() &&
         std::equal(singleNoise.begin(), singleNoise.end(), temporalNoise.begin()));
  assert(!std::equal(singleNoise.begin(), singleNoise.end(),
                     temporalNoise.begin() + singleNoise.size()));
  assert(temporalNoise == seedVrFixedNoise(shape, 42));

  // C×T spatial accumulator retains independent planes, no time folding.
  const auto plan = makeVaeOverlapPlan(10, 8, 32, 8);
  for (std::size_t planes : {32U, 15U}) {
    VaeSpatialAccumulator accumulator(plan, planes, 1);
    for (const auto& part : plan.tiles) {
      std::vector<float> values(planes * part.width * part.height);
      for (std::size_t plane = 0; plane < planes; ++plane) {
        std::fill_n(values.begin() + plane * part.width * part.height,
                    part.width * part.height, static_cast<float>(plane));
      }
      accumulator.add(values);
    }
    const auto result = accumulator.finish();
    for (std::size_t plane = 0; plane < planes; ++plane) {
      for (std::size_t pixel = 0; pixel < 80; ++pixel) {
        assert(std::abs(result[plane * 80 + pixel] - static_cast<float>(plane)) < 1e-5F);
      }
    }
  }

  const auto graph = [](std::size_t time) {
    auto input = std::make_shared<ov::opset13::Parameter>(
        ov::element::f32, ov::Shape{1, 512, time, 2, 3});
    auto bias = ov::opset13::Constant::create(ov::element::f32, ov::Shape{}, {0.25F});
    auto boundary = std::make_shared<ov::opset13::Add>(input, bias);
    boundary->set_friendly_name(
        "__module.vae.decoder.mid_block.resnets.0.conv2/aten::_convolution/Add");
    auto residual = std::make_shared<ov::opset13::Add>(boundary, input);
    return std::make_shared<ov::Model>(ov::OutputVector{residual}, ov::ParameterVector{input});
  };
  for (std::size_t time : {1U, 2U}) {
    auto model = graph(time);
    const auto original = model->output(0);
    seedVrPreserveDecoderResidual(*model);
    assert(model->outputs().size() == 2 && model->output(0) == original);
  }
  rejects([&] { seedVrPreserveDecoderResidual(*graph(3)); });

  // Maximum extra output: [1,512,2,64,64] FP32 = exactly 16 MiB.
  auto maximum = graph(2);
  maximum->get_parameters()[0]->set_partial_shape({1, 512, 2, 64, 64});
  maximum->validate_nodes_and_infer_types();
  const auto original = maximum->output(0);
  seedVrPreserveDecoderResidual(*maximum);
  assert(maximum->output(0) == original);
  assert(maximum->output(1).get_shape() == (ov::Shape{1, 512, 2, 64, 64}));
  assert(ov::shape_size(maximum->output(1).get_shape()) * sizeof(float) ==
         16U * 1024U * 1024U);
  maximum = graph(2);
  maximum->get_parameters()[0]->set_partial_shape({1, 512, 2, 65, 64});
  maximum->validate_nodes_and_infer_types();
  rejects([&] { seedVrPreserveDecoderResidual(*maximum); });

  // T1 condition stays exactly equal to the original scalar loop.
  const auto single = seedVrFrameShape(17, 19);
  const auto t1Noise = seedVrFixedNoise(single, 42);
  std::vector<float> singleMean(single.latentPixels * 16U);
  std::vector<float> actual(single.latentPixels * 33U);
  std::vector<float> expected(single.latentPixels * 33U);
  for (std::size_t i = 0; i < singleMean.size(); ++i) {
    singleMean[i] = static_cast<float>(i % 71) / 8.0F;
  }
  seedVrLatentCondition(singleMean, t1Noise, single, actual);
  for (std::size_t pixel = 0; pixel < single.latentPixels; ++pixel) {
    for (std::size_t channel = 0; channel < 16; ++channel) {
      expected[channel * single.latentPixels + pixel] = t1Noise[pixel * 16 + channel];
      expected[(channel + 16) * single.latentPixels + pixel] =
          singleMean[channel * single.latentPixels + pixel] * 0.9152F;
    }
    expected[32 * single.latentPixels + pixel] = 1.0F;
  }
  assert(actual == expected);
  // No Core, FrameSession, checkpoint, inference, or accelerator API is used.
}

int main() {
  temporalRegressions();
  // A GPU-free graph regression for the FP16 decoder fusion workaround. It
  // must retain decoded output 0 and weights, and materialize only conv+bias.
  constexpr const char* boundaryName =
      "__module.vae.decoder.mid_block.resnets.0.conv2/aten::_convolution/Add";
  const auto decoderGraph = [&](std::size_t channels = 512U, bool add = true) {
    auto input = std::make_shared<ov::opset13::Parameter>(ov::element::f32,
                                                       ov::Shape{1U, channels, 1U, 2U, 3U});
    auto bias = ov::opset13::Constant::create(ov::element::f32, ov::Shape{}, {0.25F});
    std::shared_ptr<ov::Node> boundary;
    if (add) boundary = std::make_shared<ov::opset13::Add>(input, bias);
    else boundary = std::make_shared<ov::opset13::Multiply>(input, bias);
    boundary->set_friendly_name(boundaryName);
    auto residual = std::make_shared<ov::opset13::Add>(boundary, input);
    return std::make_shared<ov::Model>(ov::OutputVector{residual}, ov::ParameterVector{input});
  };
  auto decoder = decoderGraph();
  const auto decoded = decoder->get_results()[0]->input_value(0);
  const auto biased = decoded.get_node()->input_value(0);
  const auto bias = biased.get_node()->input_value(1).get_node_shared_ptr();
  const auto parameters = decoder->get_parameters();
  seedVrPreserveDecoderResidual(*decoder);
  assert(decoder->outputs().size() == 2U);
  assert(decoder->get_results()[0]->input_value(0) == decoded);
  assert(decoder->get_results()[1]->input_value(0) == biased);
  assert(decoder->get_parameters() == parameters);
  assert(biased.get_node()->input_value(1).get_node_shared_ptr() == bias);
  assert(ov::as_type_ptr<ov::opset13::Constant>(bias)->cast_vector<float>() ==
         std::vector<float>{0.25F});
  rejects([&] { seedVrPreserveDecoderResidual(*decoder); });
  rejects([&] { seedVrPreserveDecoderResidual(*decoderGraph(511U)); });
  rejects([&] { seedVrPreserveDecoderResidual(*decoderGraph(512U, false)); });
  auto missing = decoderGraph();
  const auto missingBoundary = missing->get_results()[0]->input_value(0).get_node()->input_value(0);
  missingBoundary.get_node()->set_friendly_name("different-model");
  rejects([&] { seedVrPreserveDecoderResidual(*missing); });
  auto ambiguous = decoderGraph();
  ambiguous->get_results()[0]->input_value(0).get_node()->set_friendly_name(boundaryName);
  rejects([&] { seedVrPreserveDecoderResidual(*ambiguous); });
  auto oversized = decoderGraph();
  oversized->get_parameters()[0]->set_partial_shape({1, 512, 1, 65, 2});
  oversized->validate_nodes_and_infer_types();
  rejects([&] { seedVrPreserveDecoderResidual(*oversized); });

  for (const auto& device : {"CPU", "GPU.0", "GPU.12"}) seedVrValidateDevice(device);
  for (const auto& device : {"", "AUTO", "GPU", "GPU.", "GPU.-1", "GPU.1x", "GPU.0.0",
                             "MULTI:CPU,GPU.0", "HETERO:GPU.0,CPU", "CPU.0", "gpu.0"}) {
    rejects([&] { seedVrValidateDevice(device); });
  }
  const auto shape = seedVrFrameShape(17, 19);
  assert(shape.paddedWidth == 32 && shape.paddedHeight == 32);
  assert(shape.tokens == 4U && shape.latentPixels == 16U);
  assert(seedVrFrameShape(3840, 2160).tokens == 32400U);
  rejects([] { (void)seedVrFrameShape(0, 16); });
  rejects([] { (void)seedVrFrameShape(std::numeric_limits<int>::max(), 16); });
  rejects([] { (void)seedVrFrameShape(16384, 16384); });
  std::vector<std::uint8_t> input(17U * 19U * 3U);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<std::uint8_t>(i % 256U);
  const auto normalized = seedVrNormalizeRgb(input, shape);
  const auto plane = 32U * 32U;
  assert(normalized.size() == 3U * plane);
  assert(normalized[17U] == -1.0F && normalized[19U * 32U] == -1.0F);
  assert(normalized[plane + 17U] == -1.0F && normalized[2U * plane + 17U] == -1.0F);
  assert(seedVrDecodedRgb(normalized, shape) == input);
  rejects([&] { (void)seedVrNormalizeRgb({}, shape); });
  auto malformed = shape;
  malformed.latentPixels += 1U;
  rejects([&] { (void)seedVrNormalizeRgb(input, malformed); });
  auto nonfinite = normalized;
  nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { (void)seedVrDecodedRgb(nonfinite, shape); });
  auto clipped = normalized;
  clipped[0] = -2.0F;
  clipped[plane] = 2.0F;
  const auto clippedRgb = seedVrDecodedRgb(clipped, shape);
  assert(clippedRgb[0] == 0U && clippedRgb[1] == 255U);
  const auto noise = seedVrFixedNoise(shape, 42U);
  assert(noise.size() == shape.latentPixels * 16U);
  assert(noise == seedVrFixedNoise(shape, 42U));
  assert(noise != seedVrFixedNoise(shape, 43U));
  assert(std::all_of(noise.begin(), noise.end(), [](float v) { return std::isfinite(v); }));

  // The T=1 crop preserves channel planes and scales latent coordinates only
  // for RGB input. Values encode global plane/row/column, not local position.
  const auto tileShape = seedVrFrameShape(80, 64);
  const VaeOverlapTile tile{3, 2, 4, 3};
  for (bool encode : {false, true}) {
    const std::size_t scale = encode ? 8U : 1U;
    const std::size_t planes = encode ? 3U : 16U;
    const std::size_t width = 10U * scale;
    const std::size_t height = 8U * scale;
    std::vector<float> values(width * height * planes);
    for (std::size_t c = 0U; c < planes; ++c) {
      for (std::size_t y = 0U; y < height; ++y) {
        for (std::size_t x = 0U; x < width; ++x) {
          values[(c * height + y) * width + x] = static_cast<float>(c * 10000U + y * 100U + x);
        }
      }
    }
    const auto cropped = seedVrVaeTileInput(values, tileShape, tile, encode);
    const auto tileWidth = 4U * scale;
    const auto tileHeight = 3U * scale;
    assert(cropped.size() == planes * tileWidth * tileHeight);
    for (std::size_t c = 0U; c < planes; ++c) {
      for (std::size_t y = 0U; y < tileHeight; ++y) {
        for (std::size_t x = 0U; x < tileWidth; ++x) {
          assert(cropped[(c * tileHeight + y) * tileWidth + x] ==
                 static_cast<float>(c * 10000U + (2U * scale + y) * 100U + 3U * scale + x));
        }
      }
    }
    rejects([&] { (void)seedVrVaeTileInput({}, tileShape, tile, encode); });
    for (const auto& bad : {VaeOverlapTile{-1, 0, 1, 1}, VaeOverlapTile{0, 0, 0, 1},
                           VaeOverlapTile{9, 0, 2, 1}, VaeOverlapTile{0, 7, 1, 2},
                           VaeOverlapTile{std::numeric_limits<int>::max(), 0, 1, 1}}) {
      rejects([&] { (void)seedVrVaeTileInput(values, tileShape, bad, encode); });
    }
    values[(2U * scale) * width + 3U * scale] = std::numeric_limits<float>::infinity();
    rejects([&] { (void)seedVrVaeTileInput(values, tileShape, tile, encode); });
  }
  const auto plan4k = makeVaeOverlapPlan(480, 270);
  assert(plan4k.tileSize == 64 && plan4k.overlap == 8 && plan4k.tiles.size() == 45U);
  assert(plan4k.tiles.back().x == 448 && plan4k.tiles.back().y == 224);
  assert(plan4k.tiles.back().width == 32 && plan4k.tiles.back().height == 46);
  assert(makeVaeOverlapPlan(64, 64).tiles.size() == 1U);
}
