#pragma once

#include "flash_edges_model.hpp"
#include "flash_stage.hpp"

#include <filesystem>
#include <memory>

namespace otc::ai {

struct FlashEdgesPrefixInputs final {
  ov::Tensor x;
  ov::Tensor t;
  ov::Tensor tMod;
};

struct FlashEdgesPrefixOutputs final {
  ov::Tensor tokensOut;
  ov::Tensor tOut;
  ov::Tensor tModOut;
  ov::Tensor embeddingOut;
};

struct FlashEdgesSuffixInputs final {
  ov::Tensor tokensIn;
  ov::Tensor tIn;
};

struct FlashEdgesSuffixOutputs final {
  ov::Tensor out;
};

// One authenticated fixed-UHD prefix or suffix edge stage.  Prefix and suffix
// have separate typed calls so a caller cannot accidentally bind the wrong
// graph role or port set.
class FlashEdgesSession final {
 public:
  FlashEdgesSession(ov::Core& core, ov::RemoteContext context,
                    std::filesystem::path assets, FlashMode mode,
                    FlashEdgesRole role);
  ~FlashEdgesSession();
  FlashEdgesSession(FlashEdgesSession&&) noexcept;
  FlashEdgesSession& operator=(FlashEdgesSession&&) noexcept;
  FlashEdgesSession(const FlashEdgesSession&) = delete;
  FlashEdgesSession& operator=(const FlashEdgesSession&) = delete;

  FlashEdgesRole role() const noexcept;
  void run(const FlashEdgesPrefixInputs& inputs,
           const FlashEdgesPrefixOutputs& outputs);
  void run(const FlashEdgesSuffixInputs& inputs,
           const FlashEdgesSuffixOutputs& outputs);
  // The phase controller calls this only between completed requests.
  void suspend() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace otc::ai
