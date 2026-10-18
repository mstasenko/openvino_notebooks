#include "seedvr_frame.hpp"
#include "seedvr_component_assets.hpp"
#include "seedvr_component_cache.hpp"
#include "seedvr_blob.hpp"

#include "seedvr_session.hpp"

#include <openvino/openvino.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace otc::ai {
namespace {
constexpr float kVaeScale = 0.9152F;
constexpr std::size_t kTextElements = 58U * 5120U;
constexpr std::size_t kHiddenWidth = 2560U;
constexpr std::size_t kEmbeddingWidth = 15360U;
constexpr std::size_t kMaxTokens = 64800U;

void validateShape(const SeedVrFrameShape& shape) {
  const auto expected = seedVrFrameShape(shape.width, shape.height, shape.inputFrames);
  if (shape.paddedWidth != expected.paddedWidth || shape.paddedHeight != expected.paddedHeight ||
      shape.latentPixels != expected.latentPixels || shape.tokens != expected.tokens ||
      shape.latentFrames != expected.latentFrames) {
    throw std::invalid_argument("SeedVR2 frame shape is inconsistent");
  }
}

void requireFinite(const std::vector<float>& values) {
  if (!std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); })) {
    throw std::invalid_argument("SeedVR2 frame tensor contains non-finite values");
  }
}

std::uint64_t splitmix64(std::uint64_t& state) {
  std::uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

double uniformOpen(std::uint64_t& state) {
  // A 52-bit midpoint keeps values strictly inside (0,1).
  return (static_cast<double>(splitmix64(state) >> 12U) + 0.5) * 0x1p-52;
}

struct PortContract {
  std::string name;
  ov::Shape shape;
};

void validatePort(const ov::Output<const ov::Node>& port, const ov::Shape& expected,
                  bool concrete) {
  if (port.get_element_type() != ov::element::f32 || port.get_partial_shape().rank().is_dynamic() ||
      port.get_partial_shape().rank().get_length() != static_cast<std::int64_t>(expected.size())) {
    throw std::runtime_error("SeedVR2 frame IR has incompatible public port type or rank");
  }
  const auto actual = port.get_partial_shape();
  for (std::size_t axis = 0U; axis < expected.size(); ++axis) {
    if (!actual[axis].compatible(ov::Dimension(static_cast<std::int64_t>(expected[axis]))) ||
        (concrete && !actual[axis].is_static())) {
      throw std::runtime_error("SeedVR2 frame IR has incompatible public port dimensions");
    }
  }
}

ov::Tensor floatView(const ov::Shape& shape, std::vector<float>& values) {
  std::size_t count = 1U;
  for (std::size_t dimension : shape) {
    if (dimension == 0U || count > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::invalid_argument("SeedVR2 frame tensor size overflows");
    }
    count *= dimension;
  }
  if (values.size() != count) throw std::invalid_argument("SeedVR2 frame tensor size differs");
  return ov::Tensor(ov::element::f32, shape, values.data());
}

std::vector<float> copiedOutput(ov::InferRequest& request, std::size_t index,
                                const ov::Shape& shape) {
  const ov::Tensor tensor = request.get_output_tensor(index);
  if (tensor.get_element_type() != ov::element::f32 || tensor.get_shape() != shape) {
    throw std::runtime_error("SeedVR2 frame inference output contract differs");
  }
  const float* data = tensor.data<const float>();
  std::vector<float> result(data, data + tensor.get_size());
  requireFinite(result);
  return result;
}
}  // namespace

SeedVrFrameShape seedVrFrameShape(int width, int height, int inputFrames) {
  // Validate before adding the padding or multiplying dimensions.
  if ((inputFrames != 1 && inputFrames != 5) || width <= 0 || height <= 0 || width > 16384 || height > 16384) {
    throw std::invalid_argument("SeedVR2 frame dimensions are outside the bounded range");
  }
  const int paddedWidth = (width + 15) / 16 * 16;
  const int paddedHeight = (height + 15) / 16 * 16;
  const int latentFrames = inputFrames == 5 ? 2 : 1;
  const auto spatialTokens = static_cast<std::size_t>(paddedWidth / 16) *
                             static_cast<std::size_t>(paddedHeight / 16);
  const auto tokens = spatialTokens * static_cast<std::size_t>(latentFrames);
  if (tokens > kMaxTokens) throw std::invalid_argument("SeedVR2 frame exceeds the token bound");
  return {width, height, paddedWidth, paddedHeight, spatialTokens * 4U, tokens, inputFrames, latentFrames};
}

void seedVrValidateDevice(const std::string& device) {
  const bool gpu = device.size() > 4U && device.compare(0U, 4U, "GPU.") == 0 &&
                   std::all_of(device.begin() + 4, device.end(),
                               [](char c) { return c >= '0' && c <= '9'; });
  if (device != "CPU" && !gpu) {
    throw std::invalid_argument("SeedVR2 frame device must be exactly CPU or GPU.<digits>");
  }
}

void seedVrPreserveDecoderResidual(ov::Model& model) {
  constexpr const char* name =
      "__module.vae.decoder.mid_block.resnets.0.conv2/aten::_convolution/Add";
  std::shared_ptr<ov::Node> boundary;
  for (const auto& node : model.get_ordered_ops()) {
    if (node->get_friendly_name() != name) continue;
    if (boundary) throw std::invalid_argument("SeedVR2 decoder residual boundary is ambiguous");
    boundary = node;
  }
  if (!boundary || model.outputs().size() != 1U ||
      std::string(boundary->get_type_name()) != "Add" ||
      boundary->get_output_element_type(0) != ov::element::f32 ||
      !boundary->get_output_partial_shape(0).is_static()) {
    throw std::invalid_argument("SeedVR2 decoder residual boundary is incompatible");
  }
  const auto shape = boundary->get_output_shape(0);
  if (shape.size() != 5U || shape[0] != 1U || shape[1] != 512U || (shape[2] != 1U && shape[2] != 2U) ||
      shape[3] == 0U || shape[4] == 0U || shape[3] > 64U || shape[4] > 64U ||
      model.get_results()[0]->input_value(0) == boundary->output(0)) {
    throw std::invalid_argument("SeedVR2 decoder residual boundary shape is incompatible");
  }
  // OpenVINO 2026.3 on B580 corrupts this FP16 conv+bias+residual fusion.
  // Keeping conv+bias materialized separates only the residual Add, preserving
  // fast FP16 kernels through the rest of the decoder. At most 16 MiB for T2;
  // Source runs this mid-block before temporal upsampling, preserving latent T.
  // use a public graph output rather than private plugin flags or FP32 fallback.
  model.add_output(boundary->output(0));
}

std::vector<float> seedVrNormalizeRgb(const std::vector<std::uint8_t>& rgb,
                                    const SeedVrFrameShape& shape) {
  validateShape(shape);
  const auto frames = static_cast<std::size_t>(shape.inputFrames);
  if (rgb.size() != static_cast<std::size_t>(shape.width) * shape.height * 3U * frames) {
    throw std::invalid_argument("SeedVR2 RGB24 frame size differs");
  }
  const auto plane = static_cast<std::size_t>(shape.paddedWidth) * shape.paddedHeight;
  // The source pads black in [0,1] BEFORE normalization: padding becomes -1.
  std::vector<float> result(plane * 3U * frames, -1.0F);
  for (std::size_t t = 0U; t < frames; ++t) {
    for (int y = 0; y < shape.height; ++y) {
      for (int x = 0; x < shape.width; ++x) {
        const auto source = ((t * shape.height + y) * shape.width + x) * 3U;
        const auto target = static_cast<std::size_t>(y) * shape.paddedWidth + x;
        for (std::size_t channel = 0; channel < 3U; ++channel) {
          result[(channel * frames + t) * plane + target] = static_cast<float>(rgb[source + channel]) / 127.5F - 1.0F;
        }
      }
    }
  }
  return result;
}

std::vector<std::uint8_t> seedVrDecodedRgb(const std::vector<float>& decoded,
                                        const SeedVrFrameShape& shape) {
  validateShape(shape);
  const auto frames = static_cast<std::size_t>(shape.inputFrames);
  const auto plane = static_cast<std::size_t>(shape.paddedWidth) * shape.paddedHeight;
  if (decoded.size() != plane * 3U * frames) throw std::invalid_argument("SeedVR2 decoded frame size differs");
  requireFinite(decoded);
  std::vector<std::uint8_t> result(static_cast<std::size_t>(shape.width) * shape.height * 3U * frames);
  for (std::size_t t = 0U; t < frames; ++t) {
    for (int y = 0; y < shape.height; ++y) {
      for (int x = 0; x < shape.width; ++x) {
        const auto source = static_cast<std::size_t>(y) * shape.paddedWidth + x;
        const auto target = ((t * shape.height + y) * shape.width + x) * 3U;
        for (std::size_t channel = 0; channel < 3U; ++channel) {
          const float value = std::clamp(decoded[(channel * frames + t) * plane + source], -1.0F, 1.0F);
          result[target + channel] = static_cast<std::uint8_t>(std::lround((value + 1.0F) * 127.5F));
        }
      }
    }
  }
  return result;
}

std::vector<float> seedVrFixedNoise(const SeedVrFrameShape& shape, std::uint64_t seed) {
  validateShape(shape);
  std::vector<float> result(shape.latentPixels * static_cast<std::size_t>(shape.latentFrames) * 16U);
  for (std::size_t index = 0; index < result.size(); index += 2U) {
    const double radius = std::sqrt(-2.0 * std::log(uniformOpen(seed)));
    const double angle = 6.283185307179586476925286766559 * uniformOpen(seed);
    result[index] = static_cast<float>(radius * std::cos(angle));
    result[index + 1U] = static_cast<float>(radius * std::sin(angle));
  }
  return result;
}

std::vector<float> seedVrVaeTileInput(const std::vector<float>& values,
                                    const SeedVrFrameShape& shape,
                                    const VaeOverlapTile& tile, bool encode) {
  validateShape(shape);
  const int latentWidth = shape.paddedWidth / 8;
  const int latentHeight = shape.paddedHeight / 8;
  // Validate in latent coordinates before multiplying potentially hostile ints.
  if (tile.x < 0 || tile.y < 0 || tile.width <= 0 || tile.height <= 0 ||
      tile.width > latentWidth || tile.height > latentHeight ||
      tile.x > latentWidth - tile.width || tile.y > latentHeight - tile.height) {
    throw std::invalid_argument("SeedVR2 VAE tile is outside the frame");
  }
  const std::size_t scale = encode ? 8U : 1U;
  const std::size_t planes = encode ? 3U * static_cast<std::size_t>(shape.inputFrames)
                                   : 16U * static_cast<std::size_t>(shape.latentFrames);
  const std::size_t width = static_cast<std::size_t>(latentWidth) * scale;
  const std::size_t height = static_cast<std::size_t>(latentHeight) * scale;
  const std::size_t tileWidth = static_cast<std::size_t>(tile.width) * scale;
  const std::size_t tileHeight = static_cast<std::size_t>(tile.height) * scale;
  if (values.size() != planes * width * height) {
    throw std::invalid_argument("SeedVR2 VAE input planes differ");
  }
  std::vector<float> result(planes * tileWidth * tileHeight);
  for (std::size_t plane = 0U; plane < planes; ++plane) {
    for (std::size_t y = 0U; y < tileHeight; ++y) {
      const auto source = plane * width * height +
                          (static_cast<std::size_t>(tile.y) * scale + y) * width +
                          static_cast<std::size_t>(tile.x) * scale;
      const auto target = (plane * tileHeight + y) * tileWidth;
      std::copy_n(values.begin() + source, tileWidth, result.begin() + target);
    }
  }
  requireFinite(result);
  return result;
}

void seedVrLatentCondition(const std::vector<float>& mean, const std::vector<float>& noise,
                          const SeedVrFrameShape& shape, std::vector<float>& condition) {
  validateShape(shape);
  const auto pixels = shape.latentPixels * static_cast<std::size_t>(shape.latentFrames);
  if (mean.size() != pixels * 16U || noise.size() != pixels * 16U || condition.size() != pixels * 33U)
    throw std::invalid_argument("SeedVR2 latent condition size differs");
  requireFinite(mean);
  requireFinite(noise);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    for (std::size_t channel = 0; channel < 16U; ++channel) {
      condition[channel * pixels + pixel] = noise[pixel * 16U + channel];
      condition[(channel + 16U) * pixels + pixel] = mean[channel * pixels + pixel] * kVaeScale;
    }
    condition[32U * pixels + pixel] = 1.0F;
  }
}

class SeedVrFrameSession::Impl final {
 public:
  Impl(SeedVrFrameModels models, std::string device, int width, int height,
       std::vector<float> text, std::vector<float> noise, SeedVrFrameProgress progress, int inputFrames)
      : models_(std::move(models)), device_(std::move(device)), shape_(seedVrFrameShape(width, height, inputFrames)),
        text_(std::move(text)), noise_(std::move(noise)), progress_(std::move(progress)) {
    seedVrValidateDevice(device_);
    if (text_.size() != kTextElements || noise_.size() != shape_.latentPixels * static_cast<std::size_t>(shape_.latentFrames) * 16U) {
      throw std::invalid_argument("SeedVR2 positive text or fixed noise shape differs");
    }
    requireFinite(text_);
    requireFinite(noise_);
    for (const auto& path : {models_.encoder, models_.decoder, models_.prelude, models_.postlude}) {
      auto bin = path;
      bin.replace_extension(".bin");
      if (path.extension() != ".xml" || !std::filesystem::is_regular_file(path) ||
          !std::filesystem::is_regular_file(bin)) {
        throw std::invalid_argument("SeedVR2 frame IR XML/BIN pair is unavailable");
      }
    }
    // This also validates the exact shape-specific fused graph before inference.
    dit_ = std::make_unique<SeedVrSession>(models_.fusedBlocks, device_, shape_.latentFrames,
                                         shape_.paddedHeight / 16, shape_.paddedWidth / 16);
    const auto memory = seedVrHostMemory();
    cache_.admit(cacheGeometryQualified(), memory);
  }

  std::vector<std::uint8_t> run(const std::vector<std::uint8_t>& rgb) {
    const std::size_t h = static_cast<std::size_t>(shape_.paddedHeight);
    const std::size_t w = static_cast<std::size_t>(shape_.paddedWidth);
    const auto latentFrames = static_cast<std::size_t>(shape_.latentFrames);
    const auto latentPixels = shape_.latentPixels * latentFrames;
    const ov::Shape latentShape{1U, 16U, latentFrames, h / 8U, w / 8U};
    const ov::Shape videoShape{1U, 33U, latentFrames, h / 8U, w / 8U};
    const ov::Shape hiddenShape{1U, latentFrames, h / 16U, w / 16U, kHiddenWidth};
    std::vector<float> condition(latentPixels * 33U);
    {
      auto normalized = seedVrNormalizeRgb(rgb, shape_);
      const auto mean = vaeSpatial(normalized, true);
      seedVrLatentCondition(mean, noise_, shape_, condition);
    }  // Release VAE requests, activations and compiled graph before the next stage.
    std::vector<float> video;
    std::vector<float> text;
    std::vector<float> embedding;
    {
      if (progress_) progress_("prelude compile", 1U, 1U);
      auto prelude = compile(models_.prelude,
                             {{"vid", videoShape}, {"txt", {58U, 5120U}}, {"timestep", {1U}}},
                             {hiddenShape, {58U, kHiddenWidth}, {1U, kEmbeddingWidth}});
      auto request = prelude.create_infer_request();
      std::vector<float> timestep{1000.0F};
      request.set_tensor("vid", floatView(videoShape, condition));
      request.set_tensor("txt", floatView({58U, 5120U}, text_));
      request.set_tensor("timestep", floatView({1U}, timestep));
      if (progress_) progress_("prelude infer", 1U, 1U);
      request.infer();
      video = copiedOutput(request, 0U, hiddenShape);
      text = copiedOutput(request, 1U, {58U, kHiddenWidth});
      embedding = copiedOutput(request, 2U, {1U, kEmbeddingWidth});
      request = ov::InferRequest();
      prelude = ov::CompiledModel();
      activeOwner_.reset();
    }
    std::vector<float>().swap(condition);
    // Fused blocks consume FP16 embedding boundaries; reuse that effective
    // embedding when reconstructing their cached block-0 modulation in postlude.
    for (float& value : embedding) value = static_cast<float>(ov::float16(value));
    requireFinite(embedding);
    if (progress_) progress_("dit start", 1U, 1U);
    dit_->run(video, text, embedding);
    if (progress_) progress_("dit end", 1U, 1U);
    std::vector<float> restored;
    {
      if (progress_) progress_("postlude compile", 1U, 1U);
      auto postlude = compile(models_.postlude, {{"hidden", hiddenShape}, {"emb", {1U, kEmbeddingWidth}}},
                              {latentShape});
      auto request = postlude.create_infer_request();
      request.set_tensor("hidden", floatView(hiddenShape, video));
      request.set_tensor("emb", floatView({1U, kEmbeddingWidth}, embedding));
      if (progress_) progress_("postlude infer", 1U, 1U);
      request.infer();
      restored = copiedOutput(request, 0U, latentShape);
      // The audited postlude graph preserves the required block-0 Ada cache.
      for (std::size_t pixel = 0; pixel < latentPixels; ++pixel) {
        for (std::size_t channel = 0; channel < 16U; ++channel) {
          const auto index = channel * latentPixels + pixel;
          restored[index] = (noise_[pixel * 16U + channel] - restored[index]) / kVaeScale;
        }
      }
      requireFinite(restored);
      request = ov::InferRequest();
      postlude = ov::CompiledModel();
      activeOwner_.reset();
    }
    std::vector<float>().swap(video);
    std::vector<float>().swap(text);
    std::vector<float>().swap(embedding);
    return seedVrDecodedRgb(vaeSpatial(restored, false), shape_);
  }

 private:
  std::vector<float> vaeSpatial(const std::vector<float>& input, bool encode) {
    const auto plan = makeVaeOverlapPlan(shape_.paddedWidth / 8, shape_.paddedHeight / 8);
    // BCTHW is C×T contiguous spatial planes; tiles preserve all time.
    const auto inputFrames = static_cast<std::size_t>(shape_.inputFrames);
    const auto latentFrames = static_cast<std::size_t>(shape_.latentFrames);
    VaeSpatialAccumulator accumulator(plan, encode ? 16U * latentFrames : 3U * inputFrames, encode ? 1 : 8);
    ov::CompiledModel compiled;
    ov::InferRequest request;
    int previousHeight = 0;
    int previousWidth = 0;
    for (std::size_t index = 0U; index < plan.tiles.size(); ++index) {
      const auto& tile = plan.tiles[index];
      const auto h = static_cast<std::size_t>(tile.height);
      const auto w = static_cast<std::size_t>(tile.width);
      const ov::Shape inputShape = encode ? ov::Shape{1U, 3U, inputFrames, h * 8U, w * 8U}
                                         : ov::Shape{1U, 16U, latentFrames, h, w};
      const ov::Shape outputShape = encode ? ov::Shape{1U, 16U, latentFrames, h, w}
                                          : ov::Shape{1U, 3U, inputFrames, h * 8U, w * 8U};
      if (tile.height != previousHeight || tile.width != previousWidth) {
        // Release the old request's activations before compiling an edge shape.
        // A single retained model/request bounds plugin scratch as well as weights.
        request = ov::InferRequest();
        compiled = ov::CompiledModel();
        if (progress_) progress_(encode ? "encoder compile" : "decoder compile", index + 1U, plan.tiles.size());
        compiled = compile(encode ? models_.encoder : models_.decoder,
                           {{"value", inputShape}}, {outputShape});
        request = compiled.create_infer_request();
        previousHeight = tile.height;
        previousWidth = tile.width;
      }
      auto cropped = seedVrVaeTileInput(input, shape_, tile, encode);
      request.set_tensor("value", floatView(inputShape, cropped));
      if (progress_) progress_(encode ? "encoder infer" : "decoder infer", index + 1U, plan.tiles.size());
      request.infer();
      accumulator.add(copiedOutput(request, 0U, outputShape));
    }
    request = ov::InferRequest();
    compiled = ov::CompiledModel();
    activeOwner_.reset();
    return accumulator.finish();
  }

  using CacheStage = detail::ComponentStage;

  bool cacheGeometryQualified() const {
    // Only these T1 geometries have qualified component-cache reuse.
    return device_ != "CPU" && shape_.inputFrames == 1 &&
           ((shape_.width == 960 && shape_.height == 768) ||
            (shape_.width == 3840 && shape_.height == 2160));
  }

  CacheStage stageFor(const std::filesystem::path& path) const {
    if (path == models_.encoder) return CacheStage::Encoder;
    if (path == models_.decoder) return CacheStage::Decoder;
    if (path == models_.prelude) return CacheStage::Prelude;
    if (path == models_.postlude) return CacheStage::Postlude;
    throw std::invalid_argument("unknown frame component");
  }

  SeedVrOwnedIr readComponent(const std::filesystem::path& path, CacheStage stage) const {
    const auto& selected = kSeedVrComponentAssets.at(static_cast<std::size_t>(stage));
    if (path.filename() != selected.xml.name) {
      throw std::invalid_argument("frame component trusted basename differs");
    }
    return readSeedVrIr(path.parent_path(), selected.xml, selected.bin);
  }

  void validateCompiled(const ov::CompiledModel& compiled, const std::vector<PortContract>& inputs,
                        const std::vector<ov::Shape>& outputs, bool decoder) const {
    const std::size_t auxiliary = decoder && device_ != "CPU" ? 1U : 0U;
    if (compiled.inputs().size() != inputs.size() || compiled.outputs().size() != outputs.size() + auxiliary) {
      throw std::runtime_error("cached frame component port count differs");
    }
    for (const auto& input : inputs) validatePort(compiled.input(input.name), input.shape, true);
    for (std::size_t i = 0; i < outputs.size(); ++i) validatePort(compiled.output(i), outputs[i], true);
    if (auxiliary) {
      const auto& shape = inputs.at(0).shape;
      validatePort(compiled.output(outputs.size()), {1U, 512U, shape.at(2), shape.at(3), shape.at(4)}, true);
    }
    const auto hint = decoder && device_ != "CPU" ? ov::element::f16 : ov::element::f32;
    const auto devices = compiled.get_property("EXECUTION_DEVICES").as<std::vector<std::string>>();
    const auto actualHint = compiled.get_property(ov::hint::inference_precision);
    const auto streams = static_cast<std::int32_t>(compiled.get_property(ov::num_streams));
    const auto performance = compiled.get_property(ov::hint::performance_mode);
    if (devices != std::vector<std::string>{device_} || actualHint != hint ||
        streams != 1 || performance != ov::hint::PerformanceMode::LATENCY) {
      throw std::runtime_error("cached frame component execution/precision differs");
    }
  }

  ov::AnyMap componentProperties(bool decoder) const {
    const bool fp16Decoder = device_ != "CPU" && decoder;
    ov::AnyMap properties{{"PERFORMANCE_HINT", std::string("LATENCY")},
                          {"NUM_STREAMS", std::string("1")},
                          {"INFERENCE_PRECISION_HINT", fp16Decoder ? ov::element::f16 : ov::element::f32}};
    if (device_ == "CPU") properties.emplace("INFERENCE_NUM_THREADS", 4);
    return properties;
  }

  ov::CompiledModel compile(const std::filesystem::path& path,
                            const std::vector<PortContract>& inputs,
                            const std::vector<ov::Shape>& outputs) {
    const auto stage = stageFor(path);
    const bool decoder = stage == CacheStage::Decoder;
    const auto& shape = inputs.at(0).shape;
    const std::size_t scale = stage == CacheStage::Encoder ? 8U : 1U;
    const bool vae = stage == CacheStage::Encoder || decoder;
    const detail::ComponentKey key{stage,
        vae ? shape.at(3) / scale : static_cast<std::size_t>(shape_.paddedHeight / 8),
        vae ? shape.at(4) / scale : static_cast<std::size_t>(shape_.paddedWidth / 8)};
    // Caller has already released the previous request and compiled model.
    activeOwner_.reset();
    cache_.admit(cacheGeometryQualified(), seedVrHostMemory());
    if (const auto* blob = cache_.find(key)) {
      detail::BlobInputBuffer buffer(blob->data(), blob->size());
      std::istream stream(&buffer);
      auto compiled = core_.import_model(stream, device_, componentProperties(decoder));
      validateCompiled(compiled, inputs, outputs, decoder);
      return compiled;
    }

    activeOwner_.emplace(readComponent(path, stage));
    auto model = core_.read_model(activeOwner_->xml(), ov::Tensor(ov::element::u8,
        {activeOwner_->binBytes()}, activeOwner_->binData()));
    if (model->inputs().size() != inputs.size() || model->outputs().size() != outputs.size()) {
      throw std::runtime_error("frame component public port count differs");
    }
    std::map<std::string, ov::PartialShape> shapes;
    for (const auto& input : inputs) {
      validatePort(model->input(input.name), input.shape, false);
      shapes.emplace(input.name, ov::PartialShape(input.shape));
    }
    model->reshape(shapes);
    for (const auto& input : inputs) validatePort(model->input(input.name), input.shape, true);
    for (std::size_t i = 0; i < outputs.size(); ++i) validatePort(model->output(i), outputs[i], true);
    const bool fp16Decoder = device_ != "CPU" && decoder;
    if (fp16Decoder) seedVrPreserveDecoderResidual(*model);
    const auto properties = componentProperties(decoder);
    auto compiled = core_.compile_model(model, device_, properties);
    validateCompiled(compiled, inputs, outputs, decoder);
    if (cache_.wants(key)) retainCompiled(compiled, key);
    return compiled;
  }

  void retainCompiled(ov::CompiledModel& compiled, const detail::ComponentKey& key) {
    try {
      detail::BoundedBlobBuffer buffer(detail::kOneBlobLimit);
      std::ostream stream(&buffer);
      stream.exceptions(std::ios::badbit | std::ios::failbit);
      try {
        compiled.export_model(stream);
      } catch (const std::exception&) {
        // OpenVINO/ostream can wrap a buffer exception. Only the buffer's
        // explicit expected allocation/bound failure permits uncached fallback.
        if (!buffer.failed()) throw;
        cache_.drop();
        return;
      }
      auto bytes = buffer.take();
      const auto memory = seedVrHostMemory();
      cache_.store(key, bytes, memory);
    } catch (const std::bad_alloc&) {
      // Optional cache allocation cannot invalidate the checked compiled model.
      // Model/device/contract errors still propagate unchanged.
      cache_.drop();
    }
  }

  SeedVrFrameModels models_;
  std::string device_;
  SeedVrFrameShape shape_;
  std::vector<float> text_;
  std::vector<float> noise_;
  SeedVrFrameProgress progress_;
  detail::ComponentCache cache_;
  std::optional<SeedVrOwnedIr> activeOwner_;
  ov::Core core_;
  std::unique_ptr<SeedVrSession> dit_;
};

SeedVrFrameSession::SeedVrFrameSession(SeedVrFrameModels models, std::string device,
                                     int width, int height, std::vector<float> positiveText,
                                     std::vector<float> fixedNoise, SeedVrFrameProgress progress, int inputFrames)
    : impl_(std::make_unique<Impl>(std::move(models), std::move(device), width, height,
                                  std::move(positiveText), std::move(fixedNoise), std::move(progress), inputFrames)) {}
SeedVrFrameSession::~SeedVrFrameSession() = default;
SeedVrFrameSession::SeedVrFrameSession(SeedVrFrameSession&&) noexcept = default;
SeedVrFrameSession& SeedVrFrameSession::operator=(SeedVrFrameSession&&) noexcept = default;
std::vector<std::uint8_t> SeedVrFrameSession::run(const std::vector<std::uint8_t>& resizedRgb) {
  if (!impl_) throw std::logic_error("SeedVR2 frame session was moved from");
  return impl_->run(resizedRgb);
}
}  // namespace otc::ai
