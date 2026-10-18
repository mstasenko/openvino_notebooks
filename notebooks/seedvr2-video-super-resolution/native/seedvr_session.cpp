#include "seedvr_session.hpp"
#include "seedvr_blob.hpp"

#include "ai_enhance_core.hpp"
#include "seedvr_geometry.hpp"
#include "seedvr_model_assets.hpp"

#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/loop.hpp>
#include <openvino/op/util/multi_subgraph_base.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace otc::ai {

namespace {

constexpr std::size_t kVideoWidth = 2560U;
constexpr std::size_t kEmbeddingWidth = 15360U;
constexpr std::size_t kRopeWidth = kSeedVrRopeFrequencyWidth;
constexpr std::size_t kMaxSessionSourceTokens = 64'800U;
constexpr std::size_t kMaxRopeElements = 64U * 1024U * 1024U;
constexpr float kDistinctThreshold = 1.0e-7F;

struct PortSpec {
  std::string name;
  ov::element::Type type;
  ov::PartialShape shape;
};

using PortSpecs = std::vector<PortSpec>;

struct WindowTensors {
  ov::Tensor windowIdx;
  ov::Tensor windowKeyMask;
  ov::Tensor windowRopeFreqs;
};

ov::Dimension staticDimension(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("SeedVR2 tensor dimension is too large");
  }
  return ov::Dimension(static_cast<std::int64_t>(value));
}

ov::PartialShape partialShape(std::initializer_list<ov::Dimension> dimensions) {
  return ov::PartialShape(std::vector<ov::Dimension>(dimensions));
}

std::size_t checkedMultiply(std::size_t left, std::size_t right, const char* label) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string("SeedVR2 ") + label + " size overflows");
  }
  return left * right;
}

std::size_t checkedProduct3(std::size_t first, std::size_t second, std::size_t third,
                            const char* label) {
  return checkedMultiply(checkedMultiply(first, second, label), third, label);
}

PortSpecs fusedInputSpecs(std::size_t videoTokens, std::size_t maxVideoLen,
                          std::size_t windowCount,
                          ov::element::Type stateType = ov::element::f32,
                          ov::element::Type maskType = ov::element::boolean) {
  const std::size_t attentionLen = maxVideoLen + kSeedVrTextTokens;
  return {
      {"vid", stateType,
       partialShape({staticDimension(videoTokens), staticDimension(kVideoWidth)})},
      {"txt", stateType,
       partialShape({staticDimension(kSeedVrTextTokens), staticDimension(kVideoWidth)})},
      {"emb", stateType,
       partialShape({staticDimension(1U), staticDimension(kEmbeddingWidth)})},
      {"window_idx", ov::element::i64,
       partialShape({staticDimension(windowCount), staticDimension(maxVideoLen)})},
      {"window_key_mask", maskType,
       partialShape({staticDimension(windowCount), staticDimension(attentionLen)})},
      {"window_rope_freqs", ov::element::f32,
       partialShape({staticDimension(windowCount), staticDimension(attentionLen),
                     staticDimension(kRopeWidth)})},
  };
}

PortSpecs fusedOutputSpecs(std::size_t videoTokens,
                           ov::element::Type type = ov::element::f32) {
  return {
      {"vid_out", type,
       partialShape({staticDimension(videoTokens), staticDimension(kVideoWidth)})},
      {"txt_out", type,
       partialShape({staticDimension(kSeedVrTextTokens), staticDimension(kVideoWidth)})},
  };
}

std::map<std::string, const PortSpec*> specMap(const PortSpecs& specs) {
  std::map<std::string, const PortSpec*> result;
  for (const PortSpec& spec : specs) {
    if (!result.emplace(spec.name, &spec).second) {
      throw std::logic_error("SeedVR2 model contract contains a duplicate port name");
    }
  }
  return result;
}

void setBoundaryTypes(ov::preprocess::PrePostProcessor& prePostProcessor,
                      const PortSpecs& modelSpecs, const PortSpecs& boundarySpecs,
                      bool inputs) {
  const auto modelPorts = specMap(modelSpecs);
  if (modelPorts.size() != boundarySpecs.size()) {
    throw std::logic_error("SeedVR2 model and boundary port sets differ");
  }
  for (const PortSpec& boundary : boundarySpecs) {
    const auto modelPort = modelPorts.find(boundary.name);
    if (modelPort == modelPorts.end() || modelPort->second->shape != boundary.shape) {
      throw std::logic_error("SeedVR2 model and boundary contract differs for port: " +
                             boundary.name);
    }
    if (modelPort->second->type == boundary.type) continue;
    if (inputs) {
      prePostProcessor.input(boundary.name).tensor().set_element_type(boundary.type);
    } else {
      prePostProcessor.output(boundary.name).tensor().set_element_type(boundary.type);
    }
  }
}

template<class Model>
void validatePortSet(const Model& model, const PortSpecs& specs, bool inputs,
                     bool afterReshape) {
  const auto& ports = inputs ? model.inputs() : model.outputs();
  const auto required = specMap(specs);
  std::set<std::string> found;
  for (const ov::Output<const ov::Node>& port : ports) {
    const std::string name = port.get_any_name();
    const auto expected = required.find(name);
    if (expected == required.end() || !found.insert(name).second) {
      throw std::runtime_error("SeedVR2 IR has an unexpected or duplicate named port: " + name);
    }
    const PortSpec& spec = *expected->second;
    if (port.get_element_type() != spec.type) {
      throw std::runtime_error("SeedVR2 IR element type differs for port: " + name);
    }
    const ov::PartialShape& actualShape = port.get_partial_shape();
    if (actualShape.rank().is_dynamic() ||
        actualShape.rank().get_length() != spec.shape.rank().get_length()) {
      throw std::runtime_error("SeedVR2 IR rank differs for port: " + name);
    }
    for (std::size_t axis = 0U; axis < spec.shape.size(); ++axis) {
      const ov::Dimension& actual = actualShape[axis];
      const ov::Dimension& requested = spec.shape[axis];
      if (requested.is_dynamic()) {
        if (!actual.is_dynamic()) {
          throw std::runtime_error("SeedVR2 IR requires a dynamic dimension for port: " + name);
        }
      } else if (actual.is_static()) {
        if (actual.get_length() != requested.get_length()) {
          throw std::runtime_error("SeedVR2 IR dimension differs for port: " + name);
        }
      } else if (afterReshape) {
        throw std::runtime_error("SeedVR2 reshaped IR dimension remains dynamic for port: " + name);
      }
    }
  }
  if (found.size() != required.size()) {
    throw std::runtime_error("SeedVR2 IR named-port set is incomplete");
  }
}

ov::AnyMap compileProperties(bool gpu) {
  ov::AnyMap properties = {{"PERFORMANCE_HINT", std::string("LATENCY")},
                           {"INFERENCE_PRECISION_HINT", gpu ? ov::element::f16 : ov::element::f32},
                           {"NUM_STREAMS", std::string("1")}};
  if (gpu) properties.emplace("GPU_ENABLE_LOOP_UNROLLING", false);
  return properties;
}

ov::Shape weightShape(const SeedVrWeightBinding& weight) {
  if (weight.rank == 0 || weight.rank > weight.dimensions.size())
    throw std::invalid_argument("SeedVR2 weight rank differs");
  const ov::Shape shape(weight.dimensions.begin(), weight.dimensions.begin() + weight.rank);
  std::size_t elements = 1;
  for (auto dimension : shape) {
    if (dimension == 0) throw std::invalid_argument("SeedVR2 weight dimension is zero");
    elements = checkedMultiply(elements, dimension, "weight shape");
  }
  if (elements != weight.span.elements) throw std::invalid_argument("SeedVR2 weight span differs");
  return shape;
}

std::vector<SeedVrWeightSpan> weightSpans(std::size_t block) {
  const auto& schema = kSeedVrWeightSchemas.at(kSeedVrBlockAssets.at(block).schema);
  std::vector<SeedVrWeightSpan> spans;
  for (std::size_t i = 0; i < schema.count; ++i) {
    weightShape(schema.weights[i]);
    spans.push_back(schema.weights[i].span);
  }
  return spans;
}

using detail::BoundedBlobBuffer;

ov::Tensor inputView(ov::element::Type type, const ov::Shape& shape, const void* data) {
  if (data == nullptr) throw std::invalid_argument("SeedVR2 host input buffer is null");
  return ov::Tensor(type, shape, data);
}

ov::Tensor outputView(ov::element::Type type, const ov::Shape& shape, void* data) {
  if (data == nullptr) throw std::invalid_argument("SeedVR2 host output buffer is null");
  return ov::Tensor(type, shape, data);
}

template <typename T>
ov::Tensor inputVectorView(ov::element::Type type, const ov::Shape& shape,
                           const std::vector<T>& values) {
  return inputView(type, shape, static_cast<const void*>(values.data()));
}

template <typename T>
ov::Tensor outputVectorView(ov::element::Type type, const ov::Shape& shape,
                            std::vector<T>& values) {
  return outputView(type, shape, static_cast<void*>(values.data()));
}

void requireFinite(const std::vector<float>& values, const char* label) {
  if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error(std::string("SeedVR2 ") + label + " contains a non-finite value");
  }
}

std::vector<ov::float16> toFloat16(const std::vector<float>& values) {
  std::vector<ov::float16> result;
  result.reserve(values.size());
  for (float value : values) result.push_back(static_cast<ov::float16>(value));
  return result;
}

void copyToFloat32(const std::vector<ov::float16>& source, std::vector<float>& destination) {
  if (source.size() != destination.size()) {
    throw std::invalid_argument("SeedVR2 output size changed");
  }
  for (std::size_t index = 0U; index < source.size(); ++index) {
    destination[index] = static_cast<float>(source[index]);
  }
}

float maxAbsDifference(const std::vector<float>& left, const std::vector<float>& right) {
  if (left.size() != right.size()) throw std::invalid_argument("SeedVR2 output size changed");
  float maximum = 0.0F;
  for (std::size_t index = 0U; index < left.size(); ++index) {
    maximum = std::max(maximum, std::abs(left[index] - right[index]));
  }
  return maximum;
}

void runSync(ov::InferRequest& request) {
  request.start_async();
  request.wait();
}

void validateSessionWindowMetadata(const SeedVrWindowMetadata& metadata,
                                   std::size_t expectedSourceTokens,
                                   std::size_t expectedMaxVideoLen,
                                   std::size_t expectedWindowCount) {
  const std::size_t expectedAttentionLen = expectedMaxVideoLen + kSeedVrTextTokens;
  if (metadata.sourceTokenCount != expectedSourceTokens ||
      metadata.maxVideoLen != expectedMaxVideoLen ||
      metadata.windowCount != expectedWindowCount ||
      metadata.attentionLen != expectedAttentionLen) {
    throw std::invalid_argument("SeedVR2 fused window metadata shape differs from its contract");
  }
  const std::size_t expectedIndexElements =
      checkedMultiply(expectedWindowCount, expectedMaxVideoLen, "window index");
  const std::size_t expectedMaskElements =
      checkedMultiply(expectedWindowCount, expectedAttentionLen, "window key mask");
  const std::size_t expectedRopeElements =
      checkedProduct3(expectedWindowCount, expectedAttentionLen, kRopeWidth, "window RoPE");
  if (metadata.windowIdx.size() != expectedIndexElements ||
      metadata.windowKeyMask.size() != expectedMaskElements ||
      metadata.windowRopeFreqs.size() != expectedRopeElements) {
    throw std::invalid_argument("SeedVR2 fused window metadata allocation differs from its shape");
  }
}

}  // namespace

class SeedVrSession::Impl final {
 public:
  Impl(std::filesystem::path modelDirectory, std::string device, int latentFrames,
       int latentHeight, int latentWidth)
      : modelDirectory_(std::move(modelDirectory)),
        device_(std::move(device)),
        latentFrames_(latentFrames),
        latentHeight_(latentHeight),
        latentWidth_(latentWidth),
        gpu_(device_.rfind("GPU.", 0) == 0) {
    // The caller resolves the physical adapter; OpenVINO enumeration can change
    // after GPU passthrough or a driver rebind. Never accept AUTO or generic GPU.
    const bool explicitGpu = gpu_ && device_.size() > 4U &&
        std::all_of(device_.begin() + 4, device_.end(),
                    [](char c) { return c >= '0' && c <= '9'; });
    if (!explicitGpu && device_ != "CPU") {
      throw std::invalid_argument("SeedVR2 requires an explicit GPU index or CPU");
    }
    if (!std::filesystem::is_directory(modelDirectory_)) {
      throw std::invalid_argument("SeedVR2 model directory is unavailable");
    }
    if ((latentFrames_ != 1 && latentFrames_ != 2) || latentHeight_ <= 0 || latentWidth_ <= 0) {
      throw std::invalid_argument("SeedVR2 session requires positive T1/T2 geometry");
    }
    sourceTokens_ = checkedProduct3(static_cast<std::size_t>(latentFrames_),
                                    static_cast<std::size_t>(latentHeight_),
                                    static_cast<std::size_t>(latentWidth_), "source token");
    if (sourceTokens_ > kMaxSessionSourceTokens) {
      throw std::invalid_argument("SeedVR2 shape exceeds the qualified token count");
    }

    metadataBuffers_[0] =
        makeSeedVrWindowMetadata(latentFrames_, latentHeight_, latentWidth_, false);
    metadataBuffers_[1] =
        makeSeedVrWindowMetadata(latentFrames_, latentHeight_, latentWidth_, true);
    // Shifted edge windows can have a different padded length at small or
    // non-reference resolutions. Each parity has its own graph contract.
    for (const auto& metadata : metadataBuffers_) {
      validateSessionWindowMetadata(metadata, sourceTokens_, metadata.maxVideoLen,
                                    metadata.windowCount);
      if (checkedProduct3(metadata.windowCount, metadata.attentionLen, kRopeWidth,
                          "window RoPE") > kMaxRopeElements) {
        throw std::invalid_argument("SeedVR2 window RoPE exceeds the qualified allocation bound");
      }
    }
    const std::size_t videoStateElements =
        checkedMultiply(sourceTokens_, kVideoWidth, "video state");
    const std::size_t textStateElements =
        checkedMultiply(kSeedVrTextTokens, kVideoWidth, "text state");

    validateArtifactFiles();
    prepareAssets();
    core_ = std::make_unique<ov::Core>();
    if (gpu_) {
      remoteContext_ = core_->get_default_context(device_);
      if (remoteContext_.get_device_name() != device_) {
        throw std::runtime_error("SeedVR2 default context differs from " + device_);
      }
    }

    allocatePersistentTensors(videoStateElements, textStateElements);
  }

  void run(std::vector<float>& videoTokens, std::vector<float>& textTokens,
           const std::vector<float>& embedding) {
    const std::size_t expectedVideo = checkedMultiply(sourceTokens_, kVideoWidth, "video input");
    const std::size_t expectedText = kSeedVrTextTokens * kVideoWidth;
    if (videoTokens.size() != expectedVideo || textTokens.size() != expectedText ||
        embedding.size() != kEmbeddingWidth) {
      throw std::invalid_argument("SeedVR2 host activation shape is invalid");
    }
    requireFinite(videoTokens, "video input");
    requireFinite(textTokens, "text input");
    requireFinite(embedding, "embedding input");

    const std::vector<ov::float16> videoInputHalf = toFloat16(videoTokens);
    const std::vector<ov::float16> textInputHalf = toFloat16(textTokens);
    const std::vector<ov::float16> embeddingHalf = toFloat16(embedding);
    ov::Tensor embeddingInput =
        inputVectorView(ov::element::f16, {1U, kEmbeddingWidth}, embeddingHalf);
    if (gpu_) {
      auto destination = embeddingRemote_.as<ov::RemoteTensor>();
      destination.copy_from(embeddingInput);
      embeddingInput = embeddingRemote_;
    }
    requireTensor(embeddingInput, ov::element::f16, {1U, kEmbeddingWidth}, gpu_, "embedding");

    const auto inferBlock = [&](std::size_t block, ov::InferRequest& request) {
      const std::size_t parity = block % metadataBuffers_.size();
      const SeedVrWindowMetadata& metadata = metadataBuffers_[parity];
      const WindowTensors& metadataTensors = metadataTensors_[parity];
      const ov::Shape indexShape{metadata.windowCount, metadata.maxVideoLen};
      const ov::Shape maskShape{metadata.windowCount, metadata.attentionLen};
      const ov::Shape ropeShape{metadata.windowCount, metadata.attentionLen, kRopeWidth};
      const ov::Tensor videoInput = stateInput(block, true, videoInputHalf);
      const ov::Tensor textInput = stateInput(block, false, textInputHalf);
      const std::size_t outputSlot = block % states_.size();

      requireTensor(videoInput, ov::element::f16, {sourceTokens_, kVideoWidth},
                    block != 0U && gpu_, "video input");
      requireTensor(textInput, ov::element::f16, {kSeedVrTextTokens, kVideoWidth},
                    block != 0U && gpu_, "text input");
      requireTensor(embeddingInput, ov::element::f16, {1U, kEmbeddingWidth}, gpu_, "embedding");
      requireTensor(metadataTensors.windowIdx, ov::element::i64, indexShape, gpu_,
                    "window indices");
      requireTensor(metadataTensors.windowKeyMask, gpu_ ? ov::element::u8 : ov::element::boolean, maskShape, gpu_,
                    "window key mask");
      requireTensor(metadataTensors.windowRopeFreqs, ov::element::f32, ropeShape, gpu_,
                    "window RoPE frequencies");

      request.set_tensor("vid", videoInput);
      request.set_tensor("txt", textInput);
      request.set_tensor("emb", embeddingInput);
      request.set_tensor("window_idx", metadataTensors.windowIdx);
      request.set_tensor("window_key_mask", metadataTensors.windowKeyMask);
      request.set_tensor("window_rope_freqs", metadataTensors.windowRopeFreqs);
      request.set_tensor("vid_out", states_[outputSlot].video);
      request.set_tensor("txt_out", states_[outputSlot].text);
      runSync(request);
      requireTensor(request.get_tensor("vid_out"), ov::element::f16,
                    {sourceTokens_, kVideoWidth}, gpu_, "video output");
      requireTensor(request.get_tensor("txt_out"), ov::element::f16,
                    {kSeedVrTextTokens, kVideoWidth}, gpu_, "text output");
    };
    if (retainedIr_.empty()) {
      for (std::size_t block = 0; block < kSeedVrDitBlockCount; ++block) {
        auto active = constantModel(block);
        inferBlock(block, active.request);
      }
    } else {
      for (const auto phase : std::array<std::array<std::size_t,2>,4>{{{0,1},{1,10},{10,31},{31,32}}}) {
        const auto& schema = kSeedVrWeightSchemas.at(kSeedVrBlockAssets.at(phase[0]).schema);
        std::map<std::string,ov::Tensor> bank;
        for (std::size_t i = 0; i < schema.count; ++i)
          bank.emplace(schema.weights[i].name, makeTensor(ov::element::f16, weightShape(schema.weights[i])));
        // Reverse destruction releases requests/models before their shared bank.
        std::map<std::size_t,Active> active;
        for (std::size_t block = phase[0]; block < phase[1]; ++block) {
          const auto& assets = kSeedVrBlockAssets.at(block);
          // Large qualified geometries must release the old request/model before
          // compiling/importing the next parity. The shared weight bank survives.
          const bool singleActive = latentHeight_ != 48 || latentWidth_ != 60;
          if (singleActive && !active.count(assets.templateIndex)) active.clear();
          if (!active.count(assets.templateIndex))
            active.emplace(assets.templateIndex, cachedModel(assets.templateIndex));
          auto& request = active.at(assets.templateIndex).request;
          const auto& weights = kSeedVrWeightSchemas.at(assets.schema);
          for (std::size_t i = 0; i < weights.count; ++i) {
            const auto& weight = weights.weights[i];
            const auto shape = weightShape(weight);
            auto destination = bank.at(weight.name).as<ov::RemoteTensor>();
            requireTensor(destination, ov::element::f16, shape, true, "learned weight");
            destination.copy_from(inputView(ov::element::f16, shape, retainedIr_.at(block).weightData(weight.span)));
            request.set_tensor(weight.name, destination);
          }
          inferBlock(block, request);
        }
      }
    }

    const std::size_t finalSlot = (kSeedVrDitBlockCount - 1U) % states_.size();
    std::vector<float> finalVideo(expectedVideo);
    std::vector<float> finalText(expectedText);
    std::vector<ov::float16> finalVideoHalf(expectedVideo);
    std::vector<ov::float16> finalTextHalf(expectedText);
    requireTensor(states_[finalSlot].video, ov::element::f16,
                  {sourceTokens_, kVideoWidth}, gpu_, "final video state");
    requireTensor(states_[finalSlot].text, ov::element::f16,
                  {kSeedVrTextTokens, kVideoWidth}, gpu_, "final text state");
    if (gpu_) {
      ov::Tensor videoHost =
          outputVectorView(ov::element::f16, {sourceTokens_, kVideoWidth}, finalVideoHalf);
      ov::Tensor textHost =
          outputVectorView(ov::element::f16, {kSeedVrTextTokens, kVideoWidth}, finalTextHalf);
      states_[finalSlot].video.as<ov::RemoteTensor>().copy_to(videoHost);
      states_[finalSlot].text.as<ov::RemoteTensor>().copy_to(textHost);
    } else {
      std::copy_n(states_[finalSlot].video.data<const ov::float16>(), expectedVideo,
                  finalVideoHalf.data());
      std::copy_n(states_[finalSlot].text.data<const ov::float16>(), expectedText,
                  finalTextHalf.data());
    }
    copyToFloat32(finalVideoHalf, finalVideo);
    copyToFloat32(finalTextHalf, finalText);
    requireFinite(finalVideo, "final video output");
    requireFinite(finalText, "final text output");
    if (maxAbsDifference(finalVideo, videoTokens) <= kDistinctThreshold ||
        maxAbsDifference(finalText, textTokens) <= kDistinctThreshold) {
      throw std::runtime_error("SeedVR2 final outputs do not differ from their inputs");
    }
    videoTokens.swap(finalVideo);
    textTokens.swap(finalText);
  }

 private:
  struct StateBuffer {
    ov::Tensor video;
    ov::Tensor text;
  };

  struct Active {
    std::optional<SeedVrOwnedIr> owner;
    ov::CompiledModel compiled;
    ov::InferRequest request;
  };

  void prepareAssets() {
    bool templatesPresent = true;
    for (const auto& asset : kSeedVrTemplateAssets) {
      const auto status = std::filesystem::symlink_status(modelDirectory_ / kSeedVrModelAssetPins.at(asset.xmlPin).name);
      if (status.type() == std::filesystem::file_type::not_found) templatesPresent = false;
    }
    const bool retainedGeometry =
        (latentHeight_ == 48 && latentWidth_ == 60) ||
        (latentFrames_ == 1 && ((latentHeight_ == 90 && latentWidth_ == 160) ||
                               (latentHeight_ == 135 && latentWidth_ == 240)));
    if (gpu_ && retainedGeometry && templatesPresent &&
        seedVrCanRetainAssets(kSeedVrSnapshotBinBytes, kSeedVrSnapshotXmlBytes, seedVrHostMemory())) {
      try {
        std::vector<SeedVrOwnedIr> owners;
        std::array<std::string,6> templates;
        owners.reserve(kSeedVrDitBlockCount);
        for (std::size_t block = 0; block < kSeedVrDitBlockCount; ++block) {
          const auto& asset = kSeedVrBlockAssets.at(block);
          owners.push_back(readSeedVrIr(modelDirectory_, kSeedVrModelAssetPins.at(asset.xmlPin),
                                       kSeedVrModelAssetPins.at(asset.binPin), weightSpans(block)));
        }
        for (std::size_t i = 0; i < templates.size(); ++i)
          templates[i] = readSeedVrXml(modelDirectory_, kSeedVrModelAssetPins.at(kSeedVrTemplateAssets[i].xmlPin));
        retainedIr_.swap(owners);
        templateXml_.swap(templates);
        return;
      } catch (const std::bad_alloc&) {
        // No published owner or device work exists; partial snapshots unwind.
      }
    }
    // Missing templates select constant fallback, but a present malformed asset
    // is never disguised as memory pressure or a legitimate legacy installation.
    for (const auto& asset : kSeedVrTemplateAssets) {
      const auto& pin = kSeedVrModelAssetPins.at(asset.xmlPin);
      if (std::filesystem::symlink_status(modelDirectory_ / pin.name).type() != std::filesystem::file_type::not_found)
        readSeedVrXml(modelDirectory_, pin);
    }
  }

  void validateArtifactFiles() const {
    const auto requirePair = [this](const std::string& name) {
      const std::filesystem::path xml = modelDirectory_ / name;
      std::filesystem::path bin = xml;
      bin.replace_extension(".bin");
      if (!std::filesystem::is_regular_file(xml) || !std::filesystem::is_regular_file(bin)) {
        throw std::invalid_argument("SeedVR2 IR XML/BIN pair is unavailable: " + name);
      }
    };
    for (std::size_t block = 0U; block < kSeedVrDitBlockCount; ++block) {
      const char* parity = block % 2U == 0U ? "even" : "odd";
      requirePair("block" + std::to_string(block) + "_" + parity +
                  "_fused_loop_tail_fp16.xml");
    }
  }

  void allocatePersistentTensors(std::size_t videoStateElements, std::size_t textStateElements) {
    if (videoStateElements != checkedMultiply(sourceTokens_, kVideoWidth, "video state") ||
        textStateElements != checkedMultiply(kSeedVrTextTokens, kVideoWidth, "text state")) {
      throw std::logic_error("SeedVR2 state tensor element count differs from its shape");
    }
    const ov::Shape videoStateShape{sourceTokens_, kVideoWidth};
    const ov::Shape textStateShape{kSeedVrTextTokens, kVideoWidth};
    embeddingRemote_ = gpu_ ? makeTensor(ov::element::f16, {1U, kEmbeddingWidth}) : ov::Tensor();
    if (gpu_) {
      requireTensor(embeddingRemote_, ov::element::f16, {1U, kEmbeddingWidth}, true, "embedding");
    }
    for (StateBuffer& state : states_) {
      state.video = makeTensor(ov::element::f16, videoStateShape);
      state.text = makeTensor(ov::element::f16, textStateShape);
      requireTensor(state.video, ov::element::f16, videoStateShape, gpu_, "video state");
      requireTensor(state.text, ov::element::f16, textStateShape, gpu_, "text state");
    }

    for (std::size_t parity = 0U; parity < metadataBuffers_.size(); ++parity) {
      const SeedVrWindowMetadata& metadata = metadataBuffers_[parity];
      const ov::Shape indexShape{metadata.windowCount, metadata.maxVideoLen};
      const ov::Shape maskShape{metadata.windowCount, metadata.attentionLen};
      const ov::Shape ropeShape{metadata.windowCount, metadata.attentionLen, kRopeWidth};
      WindowTensors& tensors = metadataTensors_[parity];
      tensors.windowIdx = makeMetadataTensor(ov::element::i64, indexShape,
                                             metadata.windowIdx, "window indices");
      // Intel GPU remote buffers use u8 for boolean storage. Convert the 0/1
      // boundary back to boolean in preprocessing; keep the saved IR unchanged.
      tensors.windowKeyMask = makeMetadataTensor(gpu_ ? ov::element::u8 : ov::element::boolean, maskShape,
                                                 metadata.windowKeyMask, "window key mask");
      tensors.windowRopeFreqs = makeMetadataTensor(ov::element::f32, ropeShape,
                                                   metadata.windowRopeFreqs,
                                                   "window RoPE frequencies");
      if (gpu_) {
        // Geometry patching still needs the small reverse map and window lengths.
        SeedVrWindowMetadata& uploaded = metadataBuffers_[parity];
        std::vector<std::int64_t>().swap(uploaded.windowIdx);
        std::vector<std::uint8_t>().swap(uploaded.windowKeyMask);
        std::vector<float>().swap(uploaded.windowRopeFreqs);
      }
    }
  }

  ov::Tensor makeTensor(ov::element::Type type, const ov::Shape& shape) {
    if (!gpu_) return ov::Tensor(type, shape);
    return remoteContext_.create_tensor(
        type, shape, {{"SHARED_MEM_TYPE", std::string("USM_DEVICE_BUFFER")}});
  }

  void requireTensor(const ov::Tensor& tensor, ov::element::Type type,
                     const ov::Shape& shape, bool remote, const char* label) const {
    if (!tensor || tensor.get_element_type() != type || tensor.get_shape() != shape) {
      throw std::runtime_error(std::string("SeedVR2 tensor type or shape differs for ") + label);
    }
    const bool isRemote = tensor.is<ov::RemoteTensor>();
    if (isRemote != remote) {
      throw std::runtime_error(std::string("SeedVR2 tensor memory domain differs for ") + label);
    }
    if (isRemote && tensor.as<ov::RemoteTensor>().get_device_name() != device_) {
      throw std::runtime_error(std::string("SeedVR2 remote tensor device differs for ") + label);
    }
  }

  template <typename T>
  ov::Tensor makeMetadataTensor(ov::element::Type type, const ov::Shape& shape,
                                const std::vector<T>& values, const char* label) {
    const ov::Tensor host = inputVectorView(type, shape, values);
    if (!gpu_) {
      requireTensor(host, type, shape, false, label);
      return host;
    }
    ov::Tensor remote = makeTensor(type, shape);
    requireTensor(remote, type, shape, true, label);
    auto destination = remote.as<ov::RemoteTensor>();
    destination.copy_from(host);
    return remote;
  }

  ov::CompiledModel compileIr(std::shared_ptr<ov::Model> model,
                              const SeedVrWindowMetadata& metadata,
                              const PortSpecs& modelInputs, const PortSpecs& modelOutputs,
                              const PortSpecs& boundaryInputs, const PortSpecs& boundaryOutputs,
                              bool finalBlock) {
    patchSeedVrGeometry(*model, metadata);
    if (finalBlock) {
      // Upstream consumes only final video. Its video-only residual doubles
      // dead text, which can overflow FP16. Prune that branch, not finite checks.
      const auto video = model->output("vid_out").get_node()->input_value(0);
      const auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0F});
      const auto text = std::make_shared<ov::op::v1::Add>(model->input("txt"), zero);
      text->output(0).get_tensor().set_names({"txt_out"});
      std::set<ov::Node*> reachable;
      const auto visit = [&](const auto& self, const std::shared_ptr<ov::Node>& node) -> void {
        if (!reachable.insert(node.get()).second) return;
        for (const auto& input : node->input_values()) self(self, input.get_node_shared_ptr());
      };
      visit(visit, video.get_node_shared_ptr());
      visit(visit, text);
      auto parameters = model->get_parameters();
      parameters.erase(std::remove_if(parameters.begin(), parameters.end(), [&](const auto& p) {
        return p->output(0).get_names().count("weight_008") && !reachable.count(p.get());
      }), parameters.end());
      model = std::make_shared<ov::Model>(ov::OutputVector{video, text->output(0)},
                                        parameters, model->get_friendly_name());
      model->output(0).get_tensor().set_names({"vid_out"});
    }
    validatePortSet(*model, modelInputs, true, true);
    validatePortSet(*model, modelOutputs, false, true);
    ov::preprocess::PrePostProcessor prePostProcessor(model);
    setBoundaryTypes(prePostProcessor, modelInputs, boundaryInputs, true);
    setBoundaryTypes(prePostProcessor, modelOutputs, boundaryOutputs, false);
    auto boundaryModel = prePostProcessor.build();
    validatePortSet(*boundaryModel, boundaryInputs, true, true);
    validatePortSet(*boundaryModel, boundaryOutputs, false, true);
    ov::AnyMap properties = compileProperties(gpu_);
    if (gpu_) {
      // Unrolling is global: qualify only flat, short Loops, never nested or
      // unsupported control flow whose expansion this bound does not cover.
      const auto operations = boundaryModel->get_ordered_ops();
      bool hasShortLoop = false;
      const bool flatShortLoops = std::all_of(
          operations.begin(), operations.end(), [&hasShortLoop](const auto& operation) {
            const auto subgraph = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(operation);
            if (!subgraph) return true;
            const auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(operation);
            if (!loop || loop->get_num_iterations() <= 0 || loop->get_num_iterations() >= 16)
              return false;
            hasShortLoop = true;
            for (const auto& body : subgraph->get_functions()) {
              const auto bodyOperations = body->get_ordered_ops();
              if (std::any_of(bodyOperations.begin(), bodyOperations.end(), [](const auto& bodyOperation) {
                    return static_cast<bool>(std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(bodyOperation));
                  })) return false;
            }
            return true;
          });
      properties["GPU_ENABLE_LOOP_UNROLLING"] = hasShortLoop && flatShortLoops;
    }
    ov::CompiledModel compiled = gpu_ ? core_->compile_model(boundaryModel, remoteContext_, properties)
                                      : core_->compile_model(boundaryModel, device_, properties);
    const std::vector<std::string> executionDevices =
        compiled.get_property("EXECUTION_DEVICES").as<std::vector<std::string>>();
    if (executionDevices != std::vector<std::string>{device_}) {
      throw std::runtime_error("SeedVR2 compiled execution device is not exactly " + device_);
    }
    return compiled;
  }

  PortSpecs blockInputs(std::size_t block, bool boundary, bool weights) const {
    const std::size_t parity = block % metadataBuffers_.size();
    const SeedVrWindowMetadata& metadata = metadataBuffers_[parity];
    auto specs = fusedInputSpecs(sourceTokens_, metadata.maxVideoLen, metadata.windowCount,
                                boundary ? ov::element::f16 : ov::element::f32,
                                boundary && gpu_ ? ov::element::u8 : ov::element::boolean);
    if (weights) {
      const auto& schema = kSeedVrWeightSchemas.at(kSeedVrBlockAssets.at(block).schema);
      for (std::size_t i = 0; i < schema.count; ++i)
        specs.push_back({schema.weights[i].name, ov::element::f16, weightShape(schema.weights[i])});
    }
    return specs;
  }

  std::shared_ptr<ov::Model> readModel(const std::string& xml, const SeedVrOwnedIr& owner) {
    return core_->read_model(xml, inputView(ov::element::u8, {owner.binBytes()}, owner.binData()));
  }

  ov::CompiledModel compileBlock(std::size_t block, std::shared_ptr<ov::Model> model, bool weights) {
    return compileIr(std::move(model), metadataBuffers_[block % 2],
                     blockInputs(block, false, weights), fusedOutputSpecs(sourceTokens_),
                     blockInputs(block, true, weights), fusedOutputSpecs(sourceTokens_, ov::element::f16),
                     block + 1 == kSeedVrDitBlockCount);
  }

  Active constantModel(std::size_t block) {
    const auto& asset = kSeedVrBlockAssets.at(block);
    auto owner = readSeedVrIr(modelDirectory_, kSeedVrModelAssetPins.at(asset.xmlPin),
                             kSeedVrModelAssetPins.at(asset.binPin), weightSpans(block));
    auto compiled = compileBlock(block, readModel(owner.xml(), owner), false);
    auto request = compiled.create_infer_request();
    return {std::move(owner), std::move(compiled), std::move(request)};
  }

  void cachedContract(const ov::CompiledModel& compiled, std::size_t block) const {
    validatePortSet(compiled, blockInputs(block, true, true), true, true);
    validatePortSet(compiled, fusedOutputSpecs(sourceTokens_, ov::element::f16), false, true);
    if (compiled.get_property("EXECUTION_DEVICES").as<std::vector<std::string>>() != std::vector<std::string>{device_} ||
        compiled.get_context().get_device_name() != device_)
      throw std::runtime_error("SeedVR2 cached model device/context differs");
    for (const auto& property : std::array<std::pair<const char*,const char*>,3>{{
        {"INFERENCE_PRECISION_HINT","f16"},{"NUM_STREAMS","1"},{"PERFORMANCE_HINT","LATENCY"}}}) {
      std::ostringstream value;
      compiled.get_property(property.first).print(value);
      if (value.str() != property.second) throw std::runtime_error("SeedVR2 cached inference policy differs");
    }
  }

  Active cachedModel(std::size_t index) {
    const auto& asset = kSeedVrTemplateAssets.at(index);
    ov::CompiledModel compiled;
    std::string pendingBlob;
    if (blobs_.at(index).empty()) {
      compiled = compileBlock(asset.representative,
          readModel(templateXml_.at(index), retainedIr_.at(asset.representative)), true);
      constexpr std::size_t perBlob = 128U * 1024U * 1024U, allBlobs = 512U * 1024U * 1024U;
      std::size_t used = 0;
      for (const auto& blob : blobs_) used += blob.size();
      if (used > allBlobs) throw std::length_error("SeedVR2 retained blobs exceed their bound");
      BoundedBlobBuffer buffer(std::min(perBlob, allBlobs - used));
      std::ostream stream(&buffer);
      stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);
      compiled.export_model(stream);
      pendingBlob = buffer.take();
      if (pendingBlob.empty()) throw std::runtime_error("SeedVR2 compiled blob is empty");
    } else {
      detail::BlobInputBuffer buffer(blobs_[index].data(), blobs_[index].size());
      std::istream stream(&buffer);
      compiled = core_->import_model(stream, remoteContext_, ov::AnyMap{});
    }
    cachedContract(compiled, asset.representative);
    if (!pendingBlob.empty()) blobs_[index] = std::move(pendingBlob);
    auto request = compiled.create_infer_request();
    return {std::nullopt, std::move(compiled), std::move(request)};
  }

  ov::Tensor stateInput(std::size_t block, bool video,
                        const std::vector<ov::float16>& hostInput) const {
    if (block == 0U) {
      const ov::Shape shape = video ? ov::Shape{sourceTokens_, kVideoWidth}
                                    : ov::Shape{kSeedVrTextTokens, kVideoWidth};
      return inputVectorView(ov::element::f16, shape, hostInput);
    }
    const StateBuffer& previous = states_[(block - 1U) % states_.size()];
    return video ? previous.video : previous.text;
  }

  std::filesystem::path modelDirectory_;
  std::string device_;
  int latentFrames_;
  int latentHeight_;
  int latentWidth_;
  bool gpu_;
  std::size_t sourceTokens_ = 0U;
  std::vector<SeedVrOwnedIr> retainedIr_;
  std::array<std::string,6> templateXml_, blobs_;
  std::unique_ptr<ov::Core> core_;
  ov::RemoteContext remoteContext_;
  ov::Tensor embeddingRemote_;
  std::array<StateBuffer, 2> states_;
  std::array<SeedVrWindowMetadata, 2> metadataBuffers_;
  std::array<WindowTensors, 2> metadataTensors_;
};

SeedVrSession::SeedVrSession(std::filesystem::path modelDirectory, std::string device,
                             int latentFrames, int latentHeight, int latentWidth)
    : impl_(std::make_unique<Impl>(std::move(modelDirectory), std::move(device), latentFrames,
                                   latentHeight, latentWidth)) {}

SeedVrSession::~SeedVrSession() = default;
SeedVrSession::SeedVrSession(SeedVrSession&&) noexcept = default;
SeedVrSession& SeedVrSession::operator=(SeedVrSession&&) noexcept = default;

void SeedVrSession::run(std::vector<float>& videoTokens, std::vector<float>& textTokens,
                        const std::vector<float>& embedding) {
  if (!impl_) throw std::logic_error("SeedVR2 session is moved from");
  impl_->run(videoTokens, textTokens, embedding);
}

}  // namespace otc::ai
