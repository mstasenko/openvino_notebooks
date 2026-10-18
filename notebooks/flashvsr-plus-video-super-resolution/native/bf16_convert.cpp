#include "seedvr_prepare_detail.hpp"

#include <cstdio>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
constexpr std::size_t kChunkBytes = 1U << 20;

void setBinaryStreams() {
#ifdef _WIN32
  if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
      _setmode(_fileno(stdout), _O_BINARY) == -1)
    throw std::runtime_error("cannot set binary standard streams");
#endif
}

int run() {
  std::vector<std::uint8_t> buffer(kChunkBytes);
  for (;;) {
    std::cin.read(reinterpret_cast<char*>(buffer.data()),
                  static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = std::cin.gcount();
    if (std::cin.bad() || (std::cin.fail() && !std::cin.eof()))
      throw std::runtime_error("stdin read failed");
    const auto bytes = static_cast<std::size_t>(got);
    if ((bytes % sizeof(std::uint32_t)) != 0)
      throw std::runtime_error("stdin has a partial F32 word");
    otc::ai::prepare_detail::f32ToBf16Rne(buffer.data(), bytes);
    const auto outputBytes = bytes / 2;
    std::cout.write(reinterpret_cast<const char*>(buffer.data()),
                    static_cast<std::streamsize>(outputBytes));
    if (!std::cout) throw std::runtime_error("stdout write failed");
    if (std::cin.eof()) break;
  }
  // A short final write may still be buffered: report a failed flush as failure.
  std::cout.flush();
  if (!std::cout) throw std::runtime_error("stdout flush failed");
  return 0;
}
}  // namespace

int main(int argc, char**) {
  if (argc != 1) {
    std::cerr << "usage: otc-ai-enhance-bf16 < raw-f32 > raw-bf16\n";
    return 2;
  }
  try {
    setBinaryStreams();
    return run();
  } catch (const std::exception& error) {
    std::cerr << "otc-ai-enhance-bf16: " << error.what() << '\n';
    return 1;
  }
}
