#include "flash_ops.hpp"

#include <openvino/op/op.hpp>
#include <openvino/op/parameter.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace otc::ai {
namespace {

using Shape = ov::Shape;

constexpr std::uint64_t kSixGiB = 6ULL << 30;
constexpr std::size_t kSplitHeads = 12U;
constexpr std::size_t kSplitQueries = 16U;
constexpr std::size_t kSplitHeadDim = 128U;
constexpr std::size_t kSplitCount = 4U;
constexpr std::size_t kSplitMaxKeys = 2550U;

enum class SparseKind { direct, preciseCross, splitKey };

// The direct and split-key kernels share one authenticated source. SIMD8
// (A770) and SIMD16 (B580) use the same raw-BF16 tensor/partial-buffer ABI;
// only their per-lane matrix operands and memory access differ.

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition)
        fail(message);
}

std::uint64_t checked_product(const Shape& shape, const std::string& label) {
    std::uint64_t product = 1;
    for (const auto value : shape) {
        if (value != 0 && product > std::numeric_limits<std::uint64_t>::max() / value)
            fail(label + ": element-count overflow");
        product *= static_cast<std::uint64_t>(value);
    }
    return product;
}

void require_static_port(const ov::Node& node,
                         std::size_t index,
                         const ov::element::Type& type,
                         const Shape& shape,
                         const std::string& label) {
    require(index < node.get_input_size(), label + ": input index out of range");
    const auto partial = node.get_input_partial_shape(index);
    require(partial.is_static(), label + ": dynamic input shape is not admitted");
    require(partial.to_shape() == shape, label + ": input shape changed");
    require(node.get_input_element_type(index) == type, label + ": input type changed");
}

void require_transport_bytes(const Shape& q,
                             const Shape& k,
                             const Shape& ids,
                             const std::string& label) {
    std::uint64_t bytes = 0;
    for (const auto& shape : {q, k, k, q}) {
        const auto elements = checked_product(shape, label);
        if (elements > (std::numeric_limits<std::uint64_t>::max() - bytes) / 2U)
            fail(label + ": tensor-byte overflow");
        bytes += elements * 2U;
    }
    const auto id_elements = checked_product(ids, label);
    if (id_elements > (std::numeric_limits<std::uint64_t>::max() - bytes) / 4U)
        fail(label + ": ID-byte overflow");
    bytes += id_elements * 4U;
    require(bytes <= kSixGiB, label + ": six-GiB tensor inventory exceeded");
}

void validate_sparse_inputs(ov::Node& node, SparseKind kind, const std::string& label) {
    require(node.get_input_size() == 4, label + ": four inputs required");
    const Shape q{12, 16, 128, 128};
    require_static_port(node, 0, ov::element::f16, q, label + " q");

    const auto k_partial = node.get_input_partial_shape(1);
    const auto v_partial = node.get_input_partial_shape(2);
    const auto ids_partial = node.get_input_partial_shape(3);
    require(k_partial.is_static() && v_partial.is_static() && ids_partial.is_static(),
            label + ": dynamic K/V/ID shape is not admitted");
    const Shape k = k_partial.to_shape();
    const Shape v = v_partial.to_shape();
    const Shape ids = ids_partial.to_shape();
    require(k == v, label + ": K/V shapes must match");
    const auto maxKeys = kind == SparseKind::splitKey ? kSplitMaxKeys : 5100U;
    require(k.size() == 4 && k[0] >= 1 && k[0] <= maxKeys && k[1] == 128 &&
                k[2] == 12 && k[3] == 128,
            label + ": canonical K/V shape is not admitted");
    require(node.get_input_element_type(1) == ov::element::f16 &&
                node.get_input_element_type(2) == ov::element::f16,
            label + ": K/V must use raw F16 transport");
    if (kind == SparseKind::splitKey) {
        require(ids == Shape{12, 16, 1, k[0]},
                label + ": ordered ID-slot shape is not admitted");
    } else {
        require(ids.size() == 4 && ids[0] == 12 && ids[1] == 16 && ids[2] == 1 &&
                    ids[3] >= 1 && ids[3] <= k[0],
                label + ": ordered ID-slot shape is not admitted");
    }
    require(node.get_input_element_type(3) == ov::element::i32,
            label + ": IDs must be I32");
    if (kind == SparseKind::preciseCross) {
        require(k[0] == 4 && ids[3] == 4,
                label + ": fixed bridge requires four canonical key blocks and slots");
    }
    require_transport_bytes(q, k, ids, label);
    if (kind != SparseKind::splitKey)
        node.set_output_type(0, ov::element::f16, ov::PartialShape(q));
}

class OtcPlusSplitKeyPartial final : public ov::op::Op {
public:
    OPENVINO_OP("OtcPlusSplitKeyPartial", "extension");
    OtcPlusSplitKeyPartial() = default;
    explicit OtcPlusSplitKeyPartial(const ov::OutputVector& args) : ov::op::Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        validate_sparse_inputs(*this, SparseKind::splitKey, "OtcPlusSplitKeyPartial");
        set_output_type(0, ov::element::f32,
                        ov::PartialShape{kSplitCount * kSplitHeads,
                                          kSplitQueries, kSplitHeadDim, 130U});
    }

    bool visit_attributes(ov::AttributeVisitor&) override { return true; }

    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& args) const override {
        return std::make_shared<OtcPlusSplitKeyPartial>(args);
    }
};

class OtcPlusSplitKeyMerge final : public ov::op::Op {
public:
    OPENVINO_OP("OtcPlusSplitKeyMerge", "extension");
    OtcPlusSplitKeyMerge() = default;
    explicit OtcPlusSplitKeyMerge(const ov::OutputVector& args) : ov::op::Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        require(get_input_size() == 1,
                "OtcPlusSplitKeyMerge: one partial input required");
        const auto partial = get_input_partial_shape(0);
        require(partial.is_static() &&
                    partial.to_shape() == Shape{kSplitCount * kSplitHeads,
                                                kSplitQueries, kSplitHeadDim, 130U},
                "OtcPlusSplitKeyMerge: partial accumulator shape changed");
        require(get_input_element_type(0) == ov::element::f32,
                "OtcPlusSplitKeyMerge: partial accumulator must be FP32");
        set_output_type(0, ov::element::f16,
                        ov::PartialShape{kSplitHeads, kSplitQueries,
                                          kSplitHeadDim, kSplitHeadDim});
    }

    bool visit_attributes(ov::AttributeVisitor&) override { return true; }

    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& args) const override {
        return std::make_shared<OtcPlusSplitKeyMerge>(args);
    }
};

class OtcCrossQPack final : public ov::op::Op {
public:
    OPENVINO_OP("OtcCrossQPack", "extension");
    OtcCrossQPack() = default;
    explicit OtcCrossQPack(const ov::OutputVector& args) : ov::op::Op(args) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        require(get_input_size() == 1, "OtcCrossQPack: exactly one input required");
        require_static_port(*this, 0, ov::element::f32, Shape{1, 1, 2048, 1536},
                            "OtcCrossQPack");
        set_output_type(0, ov::element::f16, ov::PartialShape{12, 16, 128, 128});
    }
    bool visit_attributes(ov::AttributeVisitor&) override { return true; }
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        return std::make_shared<OtcCrossQPack>(args);
    }
};

class OtcCrossKVPack final : public ov::op::Op {
public:
    OPENVINO_OP("OtcCrossKVPack", "extension");
    OtcCrossKVPack() = default;
    explicit OtcCrossKVPack(const ov::OutputVector& args) : ov::op::Op(args) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        require(get_input_size() == 1, "OtcCrossKVPack: exactly one input required");
        require_static_port(*this, 0, ov::element::f32, Shape{1, 1, 512, 1536},
                            "OtcCrossKVPack");
        set_output_type(0, ov::element::f16, ov::PartialShape{4, 128, 12, 128});
    }
    bool visit_attributes(ov::AttributeVisitor&) override { return true; }
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        return std::make_shared<OtcCrossKVPack>(args);
    }
};

class OtcCrossContextUnpack final : public ov::op::Op {
public:
    OPENVINO_OP("OtcCrossContextUnpack", "extension");
    OtcCrossContextUnpack() = default;
    explicit OtcCrossContextUnpack(const ov::OutputVector& args) : ov::op::Op(args) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        require(get_input_size() == 1, "OtcCrossContextUnpack: exactly one input required");
        require_static_port(*this, 0, ov::element::f16, Shape{12, 16, 128, 128},
                            "OtcCrossContextUnpack");
        set_output_type(0, ov::element::f32, ov::PartialShape{1, 1, 2048, 1536});
    }
    bool visit_attributes(ov::AttributeVisitor&) override { return true; }
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        return std::make_shared<OtcCrossContextUnpack>(args);
    }
};

class OtcCrossPreciseXmx final : public ov::op::Op {
public:
    OPENVINO_OP("OtcCrossPreciseXmx", "extension");
    OtcCrossPreciseXmx() = default;
    explicit OtcCrossPreciseXmx(const ov::OutputVector& args) : ov::op::Op(args) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        validate_sparse_inputs(*this, SparseKind::preciseCross, "OtcCrossPreciseXmx");
    }
    bool visit_attributes(ov::AttributeVisitor&) override { return true; }
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        return std::make_shared<OtcCrossPreciseXmx>(args);
    }
};

class OtcPlusSparseXmx final : public ov::op::Op {
public:
    OPENVINO_OP("OtcPlusSparseXmx", "extension");
    OtcPlusSparseXmx() = default;
    explicit OtcPlusSparseXmx(const ov::OutputVector& args) : ov::op::Op(args) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        validate_sparse_inputs(*this, SparseKind::direct, "OtcPlusSparseXmx");
    }
    bool visit_attributes(ov::AttributeVisitor&) override { return true; }
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        return std::make_shared<OtcPlusSparseXmx>(args);
    }
};

}  // namespace

std::shared_ptr<ov::Model> makeFlashSplitKeyAttentionModel(std::size_t maxKeys) {
    require(maxKeys >= 1U && maxKeys <= kSplitMaxKeys,
            "Flash split-key attention key capacity must be in [1,2550]");

    const Shape q_shape{kSplitHeads, kSplitQueries, kSplitHeadDim, kSplitHeadDim};
    const Shape kv_shape{maxKeys, kSplitHeadDim, kSplitHeads, kSplitHeadDim};
    const Shape ids_shape{kSplitHeads, kSplitQueries, 1U, maxKeys};
    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, q_shape);
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, kv_shape);
    auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, kv_shape);
    auto blocks = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ids_shape);
    q->set_friendly_name("Q");
    k->set_friendly_name("K");
    v->set_friendly_name("V");
    blocks->set_friendly_name("blocks");
    q->output(0).get_tensor().set_names({"Q"});
    k->output(0).get_tensor().set_names({"K"});
    v->output(0).get_tensor().set_names({"V"});
    blocks->output(0).get_tensor().set_names({"blocks"});
    const ov::ParameterVector parameters{q, k, v, blocks};
    const ov::OutputVector inputs{q->output(0), k->output(0), v->output(0),
                                  blocks->output(0)};
    auto partial = std::make_shared<OtcPlusSplitKeyPartial>(inputs);
    partial->set_friendly_name("split_key_partial");
    partial->output(0).get_tensor().set_names({"split_partial"});
    auto merge = std::make_shared<OtcPlusSplitKeyMerge>(
        ov::OutputVector{partial->output(0)});
    merge->set_friendly_name("split_key_merge");
    merge->output(0).get_tensor().set_names({"attention"});
    return std::make_shared<ov::Model>(ov::OutputVector{merge->output(0)},
                                        parameters,
                                        "flash_split_key_attention");
}

void registerFlashOps(ov::Core& core) {
    core.add_extension<OtcCrossQPack, OtcCrossKVPack, OtcCrossContextUnpack,
                       OtcCrossPreciseXmx, OtcPlusSparseXmx,
                       OtcPlusSplitKeyPartial, OtcPlusSplitKeyMerge>();
}

}  // namespace otc::ai
