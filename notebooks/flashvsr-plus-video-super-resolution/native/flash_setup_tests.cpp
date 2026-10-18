#include "flash_edges_session.hpp"
#include "flash_text_session.hpp"
#include "flash_attention_session.hpp"
#include "flash_token_session.hpp"
#include "flash_denoiser.hpp"

#include <cassert>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <stdexcept>

namespace {

constexpr const char* kMissingAssets =
    "/__otc_flash_text_edges_constructor_test_assets__/";

template <class Action>
void expectInvalidArgument(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

template <class Action>
void expectMissingAsset(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

void testInvalidModesAndRolesRejectBeforeAssetIO() {
  ov::Core core;
  ov::RemoteContext context;
  const auto invalidMode = static_cast<otc::ai::FlashMode>(255);
  const auto invalidRole = static_cast<otc::ai::FlashEdgesRole>(255);

  expectInvalidArgument([&] {
    otc::ai::FlashTextSession session(core, context, kMissingAssets, invalidMode);
  });
  expectInvalidArgument([&] {
    otc::ai::FlashEdgesSession session(core, context, kMissingAssets, invalidMode,
                                       otc::ai::FlashEdgesRole::prefix);
  });
  expectInvalidArgument([&] {
    otc::ai::FlashEdgesSession session(core, context, kMissingAssets, invalidMode,
                                       otc::ai::FlashEdgesRole::suffix);
  });
  expectInvalidArgument([&] {
    otc::ai::FlashEdgesSession session(core, context, kMissingAssets,
                                       otc::ai::FlashMode::turbo, invalidRole);
  });
}

void testValidSelectionReachesAuthenticatedAssetRead() {
  ov::Core core;
  ov::RemoteContext context;

  // These selections pass mode/role validation and fail while authenticating
  // the missing asset directory. Therefore no read_model/compile/infer path
  // is reached by this CPU-only ordering test.
  expectMissingAsset([&] {
    otc::ai::FlashTextSession session(core, context, kMissingAssets,
                                      otc::ai::FlashMode::turbo);
  });
  expectMissingAsset([&] {
    otc::ai::FlashEdgesSession session(core, context, kMissingAssets,
                                       otc::ai::FlashMode::turbo,
                                       otc::ai::FlashEdgesRole::prefix);
  });
  expectMissingAsset([&] {
    otc::ai::FlashEdgesSession session(core, context, kMissingAssets,
                                       otc::ai::FlashMode::base,
                                       otc::ai::FlashEdgesRole::suffix);
  });
}

void testAttentionAdmissionPrecedesDeviceWork() {
  ov::Core core;
  ov::RemoteContext context;
  for (const auto windows : {0U, 511U}) {
    bool rejected = false;
    try {
      otc::ai::FlashAttentionSession session(core, context, kMissingAssets, windows);
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("spatial window count") != std::string::npos;
    }
    assert(rejected); // No assets or context have been dereferenced.
  }
  expectMissingAsset([&] {
    otc::ai::FlashAttentionSession session(core, context, kMissingAssets);
  });
}

void testTokenGeometryPrecedesContextWork() {
  for (const otc::ai::FlashTokenGeometry geometry : {
           otc::ai::FlashTokenGeometry{0U, 8U, 8U},
           otc::ai::FlashTokenGeometry{4U, 7U, 8U},
           otc::ai::FlashTokenGeometry{4U, 240U, 240U}}) {
    expectInvalidArgument([&] {
      otc::ai::FlashTokenSession session(ov::RemoteContext{}, geometry);
    });
  }
}

#if defined(__linux__)
void testDenoiserPlanPrecedesDeviceWork() {
  ov::Core core;
  ov::RemoteContext context;
  expectInvalidArgument([&] {
    otc::ai::FlashDenoiserSession session(core, context, kMissingAssets,
        kMissingAssets, static_cast<otc::ai::FlashMode>(255), 1U);
  });
  expectInvalidArgument([&] {
    otc::ai::FlashDenoiserSession session(core, context, kMissingAssets,
        kMissingAssets, otc::ai::FlashMode::turbo, 0U);
  });
}
#endif

}  // namespace

int main() {
  testInvalidModesAndRolesRejectBeforeAssetIO();
  testValidSelectionReachesAuthenticatedAssetRead();
  testAttentionAdmissionPrecedesDeviceWork();
  testTokenGeometryPrecedesContextWork();
#if defined(__linux__)
  testDenoiserPlanPrecedesDeviceWork();
#endif
  return 0;
}
