#include "ai_enhance_core.hpp"
#include "seedvr_frame.hpp"
#include "seedvr_assets.hpp"
#include "seedvr_stream.hpp"
#include "seedvr_prepare_cli.hpp"

#include <openvino/openvino.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace otc::ai {

namespace {

constexpr std::size_t maxFrameBytes = 512U * 1024U * 1024U;
constexpr std::uint64_t maxOutputFrames = 10'000'000ULL;
constexpr double maxDurationSeconds = 24.0 * 60.0 * 60.0;
constexpr double timeEpsilon = 1.0e-9;
constexpr double sceneCutThreshold = 0.30;

volatile std::sig_atomic_t stopRequested = 0;

struct Options {
  std::filesystem::path models;
  bool temporal = true;
  std::filesystem::path rifeModel;
  int width = 0;
  int height = 0;
  double fps = 0.0;
  int outputWidth = 0;
  int outputHeight = 0;
  double outputFps = 0.0;
  double duration = 0.0;
  std::string device;
};

[[noreturn]] void usageError(const std::string& message) {
  throw std::runtime_error(
      message +
      "\nusage: seedvr2-openvino --models DIR --mode temporal|per-frame --rife MODEL.onnx|MODEL.xml "
      "--width N --height N --fps N --output-width N --output-height N "
      "--output-fps N --duration SECONDS --device AUTO|CPU");
}

void requestStop(int) noexcept { stopRequested = 1; }

void installSignalHandlers() {
  struct sigaction action {};
  action.sa_handler = requestStop;
  action.sa_flags = 0;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGINT, &action, nullptr) != 0 || sigaction(SIGTERM, &action, nullptr) != 0) {
    throw std::runtime_error("could not install AI worker signal handlers");
  }
  if (std::signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
    throw std::runtime_error("could not ignore SIGPIPE");
  }
}

std::string requireValue(int argc, char** argv, int& index, const char* option) {
  if (index + 1 >= argc) usageError(std::string("missing value for ") + option);
  ++index;
  return argv[index];
}

int parseInt(const std::string& value, const char* option) {
  std::size_t consumed = 0U;
  try {
    const int result = std::stoi(value, &consumed);
    if (consumed != value.size()) usageError(std::string("invalid ") + option);
    return result;
  } catch (const std::exception&) {
    usageError(std::string("invalid ") + option);
  }
}

double parseDouble(const std::string& value, const char* option) {
  std::size_t consumed = 0U;
  try {
    const double result = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result)) usageError(std::string("invalid ") + option);
    return result;
  } catch (const std::exception&) {
    usageError(std::string("invalid ") + option);
  }
}

std::size_t checkedFrameBytes(int width, int height, const char* label) {
  if (width < 1 || height < 1 || width > 100'000 || height > 100'000) {
    usageError(std::string(label) + " exceeds the bounded 100000x100000 limit");
  }
  const std::uint64_t pixels = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
  const std::uint64_t bytes = pixels * 3U;
  if (pixels == 0U || bytes > maxFrameBytes) {
    usageError(std::string(label) + " exceeds the bounded frame-memory limit");
  }
  return static_cast<std::size_t>(bytes);
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << "usage: seedvr2-openvino --models DIR --mode temporal|per-frame --rife MODEL.onnx|MODEL.xml "
                    "--width N --height N --fps N --output-width N --output-height N "
                    "--output-fps N --duration SECONDS --device AUTO|CPU\n";
      std::exit(0);
    }
    if (argument == "--models") options.models = requireValue(argc, argv, index, "--models");
    else if (argument == "--mode") {
      const auto mode = requireValue(argc, argv, index, "--mode");
      if (mode != "temporal" && mode != "per-frame") usageError("--mode must be temporal or per-frame");
      options.temporal = mode == "temporal";
    }
    else if (argument == "--rife") options.rifeModel = requireValue(argc, argv, index, "--rife");
    else if (argument == "--width") options.width = parseInt(requireValue(argc, argv, index, "--width"), "--width");
    else if (argument == "--height") options.height = parseInt(requireValue(argc, argv, index, "--height"), "--height");
    else if (argument == "--fps") options.fps = parseDouble(requireValue(argc, argv, index, "--fps"), "--fps");
    else if (argument == "--output-width") {
      options.outputWidth = parseInt(requireValue(argc, argv, index, "--output-width"), "--output-width");
    } else if (argument == "--output-height") {
      options.outputHeight = parseInt(requireValue(argc, argv, index, "--output-height"), "--output-height");
    } else if (argument == "--output-fps") {
      options.outputFps = parseDouble(requireValue(argc, argv, index, "--output-fps"), "--output-fps");
    } else if (argument == "--duration") {
      options.duration = parseDouble(requireValue(argc, argv, index, "--duration"), "--duration");
    } else if (argument == "--device") {
      options.device = requireValue(argc, argv, index, "--device");
    } else {
      usageError("unknown option " + std::string(argument));
    }
  }

  if (options.device != "CPU" && options.device != "AUTO") usageError("--device must be CPU or AUTO");
  checkedFrameBytes(options.width, options.height, "source dimensions");
  checkedFrameBytes(options.outputWidth, options.outputHeight, "output dimensions");
  if (!std::isfinite(options.fps) || options.fps <= 0.0 || options.fps > 1000.0) {
    usageError("--fps must be in (0,1000]");
  }
  if (!std::isfinite(options.outputFps) || options.outputFps <= 0.0 || options.outputFps > 1000.0) {
    usageError("--output-fps must be in (0,1000]");
  }
  if (!std::isfinite(options.duration) || options.duration <= 0.0 || options.duration > maxDurationSeconds) {
    usageError("--duration must be in (0,86400]");
  }
  try {
    const std::uint64_t count = outputFrameCount(options.duration, options.outputFps);
    if (count > maxOutputFrames) usageError("requested output frame count exceeds the bounded limit");
    (void)outputFrameCount(options.duration, options.fps);
  } catch (const std::exception& error) {
    usageError(std::string("requested timeline exceeds the bounded limit: ") + error.what());
  }

  const bool needSr = options.outputWidth > options.width || options.outputHeight > options.height;
  const bool needRife = options.outputFps > options.fps + timeEpsilon;
  if (needSr && !std::filesystem::is_directory(options.models)) {
    usageError("--models directory is required for an upscaled output");
  }
  if (needRife && !std::filesystem::is_regular_file(options.rifeModel)) {
    usageError("--rife model is required for a higher output fps and must be a regular file");
  }
  if (needSr) (void)seedVrFrameShape(options.outputWidth, options.outputHeight,
                                   options.temporal ? 5 : 1);
  return options;
}

bool readFrame(std::istream& stream, std::vector<std::uint8_t>& frame) {
  stream.read(reinterpret_cast<char*>(frame.data()), static_cast<std::streamsize>(frame.size()));
  const std::streamsize received = stream.gcount();
  if (stopRequested != 0) return false;
  if (received == 0 && stream.eof()) return false;
  if (received != static_cast<std::streamsize>(frame.size())) {
    throw std::runtime_error("input ended with a partial RGB24 frame");
  }
  return true;
}

RgbFrame copyFrame(const RgbFrame& frame) { return {frame.width, frame.height, frame.pixels}; }

std::uint8_t floatToByte(float value) {
  if (!std::isfinite(value)) throw std::runtime_error("RIFE output contains a non-finite value");
  return static_cast<std::uint8_t>(std::clamp(std::lround(static_cast<double>(value) * 255.0), 0L, 255L));
}

RgbFrame resizeRgb(const RgbFrame& source, int outputWidth, int outputHeight) {
  if (source.width == outputWidth && source.height == outputHeight) return copyFrame(source);
  RgbFrame result{outputWidth, outputHeight,
                  std::vector<std::uint8_t>(static_cast<std::size_t>(outputWidth) *
                                             static_cast<std::size_t>(outputHeight) * 3U)};
  for (int y = 0; y < outputHeight; ++y) {
    const double sourceY = (static_cast<double>(y) + 0.5) * static_cast<double>(source.height) /
                               static_cast<double>(outputHeight) - 0.5;
    const int y0 = std::clamp(static_cast<int>(std::floor(sourceY)), 0, source.height - 1);
    const int y1 = std::clamp(y0 + 1, 0, source.height - 1);
    const double fy = std::clamp(sourceY - static_cast<double>(y0), 0.0, 1.0);
    for (int x = 0; x < outputWidth; ++x) {
      const double sourceX = (static_cast<double>(x) + 0.5) * static_cast<double>(source.width) /
                                 static_cast<double>(outputWidth) - 0.5;
      const int x0 = std::clamp(static_cast<int>(std::floor(sourceX)), 0, source.width - 1);
      const int x1 = std::clamp(x0 + 1, 0, source.width - 1);
      const double fx = std::clamp(sourceX - static_cast<double>(x0), 0.0, 1.0);
      const std::size_t destination =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(outputWidth) +
           static_cast<std::size_t>(x)) * 3U;
      const std::size_t a =
          (static_cast<std::size_t>(y0) * static_cast<std::size_t>(source.width) +
           static_cast<std::size_t>(x0)) * 3U;
      const std::size_t b =
          (static_cast<std::size_t>(y0) * static_cast<std::size_t>(source.width) +
           static_cast<std::size_t>(x1)) * 3U;
      const std::size_t c =
          (static_cast<std::size_t>(y1) * static_cast<std::size_t>(source.width) +
           static_cast<std::size_t>(x0)) * 3U;
      const std::size_t d =
          (static_cast<std::size_t>(y1) * static_cast<std::size_t>(source.width) +
           static_cast<std::size_t>(x1)) * 3U;
      for (int channel = 0; channel < 3; ++channel) {
        const double top = static_cast<double>(source.pixels[a + static_cast<std::size_t>(channel)]) * (1.0 - fx) +
                           static_cast<double>(source.pixels[b + static_cast<std::size_t>(channel)]) * fx;
        const double bottom = static_cast<double>(source.pixels[c + static_cast<std::size_t>(channel)]) * (1.0 - fx) +
                              static_cast<double>(source.pixels[d + static_cast<std::size_t>(channel)]) * fx;
        result.pixels[destination + static_cast<std::size_t>(channel)] =
            static_cast<std::uint8_t>(std::clamp(std::lround(top * (1.0 - fy) + bottom * fy), 0L, 255L));
      }
    }
  }
  return result;
}

std::vector<std::uint8_t> sceneSample(const RgbFrame& frame) {
  constexpr int sampleWidth = 16;
  constexpr int sampleHeight = 16;
  std::vector<std::uint8_t> sample(static_cast<std::size_t>(sampleWidth * sampleHeight));
  for (int y = 0; y < sampleHeight; ++y) {
    const int sourceY = std::min(frame.height - 1, (y * frame.height) / sampleHeight);
    for (int x = 0; x < sampleWidth; ++x) {
      const int sourceX = std::min(frame.width - 1, (x * frame.width) / sampleWidth);
      const std::size_t offset =
          (static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(frame.width) +
           static_cast<std::size_t>(sourceX)) * 3U;
      const int luma = (77 * static_cast<int>(frame.pixels[offset]) +
                        150 * static_cast<int>(frame.pixels[offset + 1U]) +
                        29 * static_cast<int>(frame.pixels[offset + 2U])) >> 8;
      sample[static_cast<std::size_t>(y * sampleWidth + x)] = static_cast<std::uint8_t>(luma);
    }
  }
  return sample;
}


bool identicalFrames(const RgbFrame& first, const RgbFrame& second) {
  return exactRgbEqual(first.width, first.height, first.pixels, second.pixels);
}

int staticDimension(const ov::PartialShape& shape, std::size_t index) {
  if (shape.rank().is_dynamic() || index >= static_cast<std::size_t>(shape.rank().get_length()) ||
      shape[index].is_dynamic()) {
    return -1;
  }
  const auto length = shape[index].get_length();
  if (length < 1 || length > static_cast<std::int64_t>(std::numeric_limits<int>::max())) return -1;
  return static_cast<int>(length);
}

void requireRgbOutputShape(const ov::PartialShape& shape, const char* label) {
  if (shape.rank().is_dynamic() || shape.rank().get_length() != 4 || staticDimension(shape, 0) == 0 ||
      (staticDimension(shape, 0) > 0 && staticDimension(shape, 0) != 1) ||
      (staticDimension(shape, 1) > 0 && staticDimension(shape, 1) != 3)) {
    throw std::runtime_error(std::string(label) + " must be rank-4 NCHW RGB output");
  }
}

void fillImageChannels(const RgbFrame& frame, int tensorWidth, int tensorHeight, float* destination) {
  for (int y = 0; y < tensorHeight; ++y) {
    const int sourceY = std::min(y, frame.height - 1);
    for (int x = 0; x < tensorWidth; ++x) {
      const int sourceX = std::min(x, frame.width - 1);
      const std::size_t source =
          (static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(frame.width) +
           static_cast<std::size_t>(sourceX)) * 3U;
      const std::size_t pixel = static_cast<std::size_t>(y) * static_cast<std::size_t>(tensorWidth) +
                                static_cast<std::size_t>(x);
      for (int channel = 0; channel < 3; ++channel) {
        destination[static_cast<std::size_t>(channel) * static_cast<std::size_t>(tensorWidth) *
                        static_cast<std::size_t>(tensorHeight) +
                    pixel] = static_cast<float>(frame.pixels[source + static_cast<std::size_t>(channel)]) / 255.0F;
      }
    }
  }
}

std::string oneLineError(const std::exception& error) {
  std::string message = error.what();
  for (char& character : message) {
    if (character == '\n' || character == '\r' || character == '\t') character = ' ';
  }
  if (message.size() > 512U) message.resize(512U);
  return message;
}

bool mayRetryDevice(const Options& options) {
  return stopRequested == 0 && options.device == "AUTO";
}

class RifeSession {
 public:
  RifeSession(ov::Core& core, const std::filesystem::path& path, const std::string& device,
              int sourceWidth, int sourceHeight) {
    const std::shared_ptr<ov::Model> model = core.read_model(path.string());
    if (model->inputs().size() != 1U || model->outputs().empty()) {
      throw std::runtime_error("RIFE v4.25 v2 model must have one input and one output");
    }
    inputShape_ = model->input().get_partial_shape();
    if (inputShape_.rank().is_dynamic() || inputShape_.rank().get_length() != 4 ||
        staticDimension(inputShape_, 0) > 1 || staticDimension(inputShape_, 0) == 0 ||
        staticDimension(inputShape_, 1) != 7) {
      throw std::runtime_error("RIFE v4.25 v2 input must be packed FP32 [1,7,H,W]");
    }
    if (model->input().get_element_type() != ov::element::f32) {
      throw std::runtime_error("RIFE v4.25 v2 input must use f32");
    }
    if (sourceWidth <= 0 || sourceHeight <= 0) {
      throw std::runtime_error("RIFE source dimensions must be positive");
    }
    model->reshape(ov::PartialShape{1U, 7U, static_cast<std::int64_t>(sourceHeight),
                                    static_cast<std::int64_t>(sourceWidth)});
    inputShape_ = model->input().get_partial_shape();
    outputShape_ = model->output().get_partial_shape();
    requireRgbOutputShape(outputShape_, "RIFE output");
    if (model->output().get_element_type() != ov::element::f32) {
      throw std::runtime_error("RIFE output must use f32");
    }
    compiled_.emplace(core.compile_model(model, device));
    request_.emplace(compiled_->create_infer_request());
  }

  RgbFrame infer(const RgbFrame& first, const RgbFrame& second, float timestep) {
    if (first.width != second.width || first.height != second.height) {
      throw std::runtime_error("RIFE input dimensions differ");
    }
    const int fixedHeight = staticDimension(inputShape_, 2);
    const int fixedWidth = staticDimension(inputShape_, 3);
    const int tensorHeight = fixedHeight > 0 ? fixedHeight : first.height;
    const int tensorWidth = fixedWidth > 0 ? fixedWidth : first.width;
    if (first.width > tensorWidth || first.height > tensorHeight) {
      throw std::runtime_error("RIFE packed input has a static shape smaller than the source frame");
    }
    ov::Tensor input(ov::element::f32,
                     ov::Shape{1U, 7U, static_cast<std::size_t>(tensorHeight), static_cast<std::size_t>(tensorWidth)});
    float* destination = input.data<float>();
    fillImageChannels(first, tensorWidth, tensorHeight, destination);
    fillImageChannels(second, tensorWidth, tensorHeight,
                      destination + 3U * static_cast<std::size_t>(tensorWidth) *
                                        static_cast<std::size_t>(tensorHeight));
    const std::size_t plane = static_cast<std::size_t>(tensorWidth) * static_cast<std::size_t>(tensorHeight);
    std::fill(destination + 6U * plane, destination + 7U * plane, timestep);
    request_->set_input_tensor(0U, input);

    request_->infer();
    ov::Tensor output = request_->get_output_tensor(0U);
    const ov::Shape shape = output.get_shape();
    if (shape.size() != 4U || shape[0] != 1U || shape[1] != 3U ||
        shape[2] < static_cast<std::size_t>(first.height) ||
        shape[3] < static_cast<std::size_t>(first.width)) {
      throw std::runtime_error("RIFE output does not preserve the source RGB dimensions");
    }
    RgbFrame result{first.width, first.height,
                    std::vector<std::uint8_t>(static_cast<std::size_t>(first.width) *
                                               static_cast<std::size_t>(first.height) * 3U)};
    const float* source = output.data<const float>();
    const std::size_t outputPlane = shape[2] * shape[3];
    for (int y = 0; y < first.height; ++y) {
      for (int x = 0; x < first.width; ++x) {
        const std::size_t pixel = static_cast<std::size_t>(y) * static_cast<std::size_t>(shape[3]) +
                                  static_cast<std::size_t>(x);
        const std::size_t destination =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(first.width) +
             static_cast<std::size_t>(x)) * 3U;
        for (int channel = 0; channel < 3; ++channel) {
          result.pixels[destination + static_cast<std::size_t>(channel)] =
              floatToByte(source[static_cast<std::size_t>(channel) * outputPlane + pixel]);
        }
      }
    }
    return result;
  }

 private:
  ov::PartialShape inputShape_;
  ov::PartialShape outputShape_;
  std::optional<ov::CompiledModel> compiled_;
  std::optional<ov::InferRequest> request_;
};

struct Runtime {
  explicit Runtime(std::string selectedDevice) : device(std::move(selectedDevice)) {}
  ov::Core core;
  std::string device;
  std::unique_ptr<SeedVrFrameSession> seed;
  int seedFrames = 0;
  std::unique_ptr<RifeSession> rife;
};

bool looksDiscreteGpu(const std::string& fullName) {
  std::string lowerName = fullName;
  std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return lowerName.find("arc") != std::string::npos || lowerName.find("discrete") != std::string::npos ||
         lowerName.find("dg1") != std::string::npos || lowerName.find("dg2") != std::string::npos ||
         lowerName.find("flex") != std::string::npos;
}

std::string hardwareLabel(ov::Core& core, const std::string& device) {
  if (device == "CPU") return "CPU";
  try {
    const ov::device::Type type = core.get_property(device, ov::device::type);
    if (type == ov::device::Type::DISCRETE) return "dGPU";
    if (type == ov::device::Type::INTEGRATED) return "iGPU";
  } catch (const ov::Exception&) {
  }
  return {};
}

std::string canonicalDiscoveredDevice(const std::string& discovered) {
  // OpenVINO reports "GPU" for a lone device; its documented alias is GPU.0.
  // Normalize discovery only, so the frame/session boundaries remain strict.
  const std::string device = discovered == "GPU" ? "GPU.0" : discovered;
  seedVrValidateDevice(device);
  return device;
}

std::vector<std::string> deviceCandidates(const std::string& requested) {
  if (requested == "CPU") return {"CPU"};
  ov::Core core;
  std::vector<std::string> discrete;
  std::vector<std::string> integrated;
  std::vector<std::string> available;
  try {
    available = core.get_available_devices();
  } catch (const ov::Exception& error) {
    std::cerr << "seedvr2-openvino: device-discovery failed reason=" << oneLineError(error) << "\n";
    return {"CPU"};
  }
  for (const std::string& discovered : available) {
    if (discovered.rfind("GPU", 0U) != 0U) continue;
    std::string device;
    try {
      device = canonicalDiscoveredDevice(discovered);
    } catch (const std::invalid_argument&) {
      continue; // Unsupported discovery forms must not reach strict sessions.
    }
    bool discreteGpu = false;
    bool typeKnown = false;
    try {
      discreteGpu = core.get_property(device, ov::device::type) == ov::device::Type::DISCRETE;
      typeKnown = true;
    } catch (const ov::Exception&) {
    }
    std::string fullName = device;
    try {
      fullName = core.get_property(device, ov::device::full_name);
    } catch (const ov::Exception&) {
    }
    if ((typeKnown && discreteGpu) || (!typeKnown && looksDiscreteGpu(fullName))) {
      discrete.push_back(device);
    } else {
      integrated.push_back(device);
    }
  }
  std::vector<std::string> result;
  result.insert(result.end(), discrete.begin(), discrete.end());
  result.insert(result.end(), integrated.begin(), integrated.end());
  result.push_back("CPU");
  return result;
}

std::unique_ptr<Runtime> compileRuntime(const std::string& device) {
  auto runtime = std::make_unique<Runtime>(device);
  runtime->core.set_property(device, ov::num_streams(1));
  if (device == "CPU") runtime->core.set_property("CPU", ov::inference_num_threads(4));
  return runtime;
}

std::vector<RgbFrame> enhanceSourceBatch(const std::vector<RgbFrame>& originals,
                                        const Options& options, Runtime* runtime,
                                        const std::vector<float>& text, bool needSr) {
  if (originals.size() != 1 && originals.size() != 5) {
    throw std::invalid_argument("SeedVR2 requires a T1 or T5 batch");
  }
  std::vector<RgbFrame> resized;
  for (const auto& original : originals) {
    resized.push_back(resizeRgb(original, options.outputWidth, options.outputHeight));
  }
  if (!needSr) return resized;
  if (!runtime) throw std::runtime_error("SeedVR2 runtime is unavailable");

  // No RIFE request/model overlaps an active SeedVR compiled stage.
  runtime->rife.reset();
  const int frames = static_cast<int>(originals.size());
  if (runtime->seedFrames != frames) {
    runtime->seed.reset();
    runtime->seedFrames = 0;
    const auto& directory = options.models;
    SeedVrFrameModels models{
        directory / "encoder-dynamic-t-hw.xml", directory / "decoder-dynamic-t-hw.xml",
        directory / "prelude_dynamic_fp16.xml", directory / "postlude_dynamic_fp16.xml",
        directory};
    const auto shape = seedVrFrameShape(options.outputWidth, options.outputHeight, frames);
    auto noise = seedVrFixedNoise(shape, 20260913);
    const auto progress = [](const char* stage, std::size_t tile, std::size_t count) {
      if (stopRequested) throw std::runtime_error("AI worker cancelled");
      std::cerr << "seedvr2-openvino: stage=" << stage << " tile=" << tile << '/' << count << '\n';
    };
    runtime->seed = std::make_unique<SeedVrFrameSession>(
        models, runtime->device, options.outputWidth, options.outputHeight,
        text, std::move(noise), progress, frames);
    runtime->seedFrames = frames;
  }
  const auto frameBytes = checkedFrameBytes(options.outputWidth, options.outputHeight, "output dimensions");
  std::vector<std::uint8_t> packed;
  packed.reserve(frameBytes * originals.size());
  for (const auto& frame : resized) {
    packed.insert(packed.end(), frame.pixels.begin(), frame.pixels.end());
  }
  const auto rendered = runtime->seed->run(packed);
  if (rendered.size() != frameBytes * originals.size()) {
    throw std::runtime_error("SeedVR2 output batch size differs");
  }
  for (std::size_t i = 0; i < resized.size(); ++i) {
    resized[i].pixels.assign(rendered.begin() + i * frameBytes,
                            rendered.begin() + (i + 1) * frameBytes);
  }
  return resized;
}

struct SourceWindow {
  const Options& options;
  SeedVrSourceProvider& provider;
  std::optional<SourceFrame> left;
  std::optional<SourceFrame> right;
  std::uint64_t nextIndex = 0U;
  bool eof = false;

  SourceWindow(const Options& optionsIn, SeedVrSourceProvider& providerIn)
      : options(optionsIn), provider(providerIn) {}

  bool readOne() {
    if (eof) return false;
    auto next = provider.next();
    if (!next) {
      eof = true;
      return false;
    }
    auto frame = std::move(*next);
    ++nextIndex;
    if (!left) left = std::move(frame);
    else right = std::move(frame);
    return true;
  }

  bool initialize() { return readOne(); }

  void ensureRight(double timeSeconds) {
    while (!right || static_cast<double>(right->index) / options.fps < timeSeconds - timeEpsilon) {
      if (right) {
        left = std::move(right);
        right.reset();
      }
      if (!readOne()) break;
    }
  }

};

RgbFrame processBaseFrame(const RgbFrame& first, const RgbFrame& second, double alpha,
                          bool allowRife, Runtime* runtime, const Options& options) {
  if (alpha <= timeEpsilon || alpha >= 1.0 - timeEpsilon || !allowRife) {
    return copyFrame(alpha < 0.5 ? first : second);
  }
  if (!runtime) throw std::runtime_error("RIFE runtime is unavailable");
  if (identicalFrames(first, second)) return copyFrame(first);
  if (!runtime->rife) {
    runtime->rife = std::make_unique<RifeSession>(
        runtime->core, options.rifeModel, runtime->device, first.width, first.height);
  }
  return runtime->rife->infer(first, second, static_cast<float>(alpha));
}

}  // namespace

}  // namespace otc::ai

int main(int argc, char** argv) {
  using namespace otc::ai;
  try {
    installSignalHandlers();
    if (const auto result = runSeedVrPrepareCli(argc, argv, stopRequested,
                                               std::cerr, std::cerr))
      return *result;
    const Options options = parseOptions(argc, argv);
    const bool needSr = options.outputWidth > options.width || options.outputHeight > options.height;
    const bool needRife = options.outputFps > options.fps + timeEpsilon;
    std::cerr << "seedvr2-openvino: starting\n";
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Authenticate the owned positive text before any OpenVINO Core/device work.
    const auto text = needSr ? readSeedVrPositiveText(options.models) : std::vector<float>{};
    std::vector<std::string> candidates;
    if (needSr || needRife) candidates = deviceCandidates(options.device);
    std::unique_ptr<Runtime> runtime;
    std::size_t candidateIndex = 0U;
    auto activateNext = [&]() -> bool {
      // Do not retain a failed device's models while allocating its fallback.
      runtime.reset();
      while (stopRequested == 0 && candidateIndex < candidates.size()) {
        const std::string candidate = candidates[candidateIndex++];
        try {
          runtime = compileRuntime(candidate);
          std::cerr << "seedvr2-openvino: device=" << candidate;
          const std::string hardware = hardwareLabel(runtime->core, candidate);
          if (!hardware.empty()) std::cerr << " hardware=" << hardware;
          std::cerr << "\n";
          return true;
        } catch (const std::exception& error) {
          runtime.reset();
          std::cerr << "seedvr2-openvino: device=" << candidate << " failed reason=" << oneLineError(error) << "\n";
          if (options.device == "CPU") break;
        }
      }
      return false;
    };

    if ((needSr || needRife) && !activateNext()) {
      throw std::runtime_error("could not compile AI models for any requested device");
    }
    if (!needSr && !needRife) std::cerr << "seedvr2-openvino: device=none hardware=none\n";

    const std::uint64_t frameCount = outputFrameCount(options.duration, options.outputFps);
    const std::uint64_t requiredSourceFrames = outputFrameCount(options.duration, options.fps);
    std::uint64_t inputIndex = 0;
    std::vector<std::uint8_t> previousSample;
    const auto readOriginal = [&]() -> std::optional<SourceFrame> {
      RgbFrame frame{options.width, options.height,
                     std::vector<std::uint8_t>(checkedFrameBytes(options.width, options.height, "source dimensions"))};
      if (!readFrame(std::cin, frame.pixels)) return std::nullopt;
      const auto sample = sceneSample(frame);
      double difference = 0;
      for (std::size_t i = 0; i < previousSample.size(); ++i) {
        difference += std::abs(static_cast<int>(sample[i]) - static_cast<int>(previousSample[i])) / 255.0;
      }
      const bool cut = !previousSample.empty() && difference / sample.size() >= sceneCutThreshold;
      previousSample = sample;
      return SourceFrame{inputIndex++, std::move(frame), cut};
    };
    const auto renderBatch = [&](const std::vector<RgbFrame>& originals) {
      return renderSourceBatchWithRetry(originals,
          [&](const std::vector<RgbFrame>& batch) {
            return enhanceSourceBatch(batch, options, runtime.get(), text, needSr);
          },
          [&](const std::exception& error) {
            if (!mayRetryDevice(options)) return false;
            std::cerr << "seedvr2-openvino: batch-retry reason=" << oneLineError(error) << '\n';
            return activateNext();
          });
    };
    SeedVrSourceProvider provider(requiredSourceFrames, needSr && options.temporal,
                                  readOriginal, renderBatch);
    SourceWindow window(options, provider);
    if (!window.initialize()) {
      if (stopRequested) return 0;
      throw std::runtime_error("source ended before the requested duration");
    }
    for (std::uint64_t outputIndex = 0U; outputIndex < frameCount && stopRequested == 0; ++outputIndex) {
      const double timestamp = static_cast<double>(outputIndex) / options.outputFps;
      window.ensureRight(timestamp);
      if (!window.left) throw std::runtime_error("source frame window is empty");

      RgbFrame base;
      if (!window.right) {
        base = copyFrame(window.left->frame);
      } else {
        const double leftTime = static_cast<double>(window.left->index) / options.fps;
        const double rightTime = static_cast<double>(window.right->index) / options.fps;
        const double interval = rightTime - leftTime;
        const double alpha = interval > 0.0 ? std::clamp((timestamp - leftTime) / interval, 0.0, 1.0) : 0.0;
        while (true) {
          try {
            base = processBaseFrame(window.left->frame, window.right->frame, alpha, needRife && !window.right->cutBefore,
                                    runtime.get(), options);
            break;
          } catch (const std::exception& error) {
            if (!mayRetryDevice(options)) throw;
            const std::string failedDevice = runtime ? runtime->device : "unknown";
            std::cerr << "seedvr2-openvino: device=" << failedDevice << " failed reason=" << oneLineError(error) << "\n";
            if (!activateNext()) throw;
            std::cerr << "seedvr2-openvino: retry-after-inference-failure reason=" << oneLineError(error) << "\n";
          }
        }
      }

      const auto& output = base;
      std::cout.write(reinterpret_cast<const char*>(output.pixels.data()),
                      static_cast<std::streamsize>(output.pixels.size()));
      if (!std::cout) throw std::runtime_error("output pipe closed");
    }
    if (stopRequested == 0) {
      // Drain frames beyond the requested timeline so a finite decoder pipe can exit
      // cleanly.  The count check rejects a genuinely truncated source instead of
      // silently extending it with a long held final frame.
      provider.drain();
    }
    return 0;
  } catch (const std::exception& error) {
    if (stopRequested) return 0;
    std::cerr << "seedvr2-openvino: error=" << error.what() << "\n";
    return 1;
  }
}
