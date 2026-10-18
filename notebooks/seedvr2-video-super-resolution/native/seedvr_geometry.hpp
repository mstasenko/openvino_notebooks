#pragma once

namespace ov { class Model; }

namespace otc::ai {
struct SeedVrWindowMetadata;

// Patches a fresh, verified fused template in memory; never reads/copies weights
// or selects a device. Initial support is T1 only. Extra weight Parameters stay
// unchanged. The loader owns artifact integrity and template/session lifetime.
void patchSeedVrGeometry(ov::Model& model, const SeedVrWindowMetadata& metadata);
}
