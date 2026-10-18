#include "flash_ops.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/opsets/opset13.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using Shape = std::vector<std::size_t>;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::string dims(const Shape& shape) {
    std::ostringstream out;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    return out.str();
}

std::string dim_lines(const Shape& shape) {
    std::ostringstream out;
    for (const auto value : shape) out << "<dim>" << value << "</dim>";
    return out.str();
}

struct Fixture {
    std::string type;
    std::vector<Shape> inputs;
    std::vector<std::string> precisions;
    Shape output;
    std::string output_precision;
    std::string version = "extension";
};

std::string make_ir(const Fixture& fixture) {
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?><net name=\"flash_ops_fixture\" version=\"11\"><layers>";
    int id = 0;
    for (std::size_t i = 0; i < fixture.inputs.size(); ++i) {
        xml << "<layer id=\"" << id << "\" name=\"p" << i
            << "\" type=\"Parameter\" version=\"opset1\"><data shape=\""
            << dims(fixture.inputs[i]) << "\" element_type=\""
            << (fixture.precisions[i] == "FP16" ? "f16" : fixture.precisions[i] == "FP32" ? "f32" : "i32")
            << "\"/><output><port id=\"0\" precision=\"" << fixture.precisions[i]
            << "\">" << dim_lines(fixture.inputs[i]) << "</port></output></layer>";
        ++id;
    }
    const int op_id = id++;
    xml << "<layer id=\"" << op_id << "\" name=\"op\" type=\"" << fixture.type
        << "\" version=\"" << fixture.version << "\"><input>";
    for (std::size_t i = 0; i < fixture.inputs.size(); ++i) {
        xml << "<port id=\"" << i << "\" precision=\"" << fixture.precisions[i]
            << "\">" << dim_lines(fixture.inputs[i]) << "</port>";
    }
    xml << "</input><output><port id=\"" << fixture.inputs.size()
        << "\" precision=\"" << fixture.output_precision << "\">"
        << dim_lines(fixture.output) << "</port></output></layer>";
    const int result_id = id++;
    xml << "<layer id=\"" << result_id << "\" name=\"result\" type=\"Result\" version=\"opset1\">"
        << "<input><port id=\"0\" precision=\"" << fixture.output_precision << "\">"
        << dim_lines(fixture.output) << "</port></input><output><port id=\"1\" precision=\""
        << fixture.output_precision << "\">" << dim_lines(fixture.output)
        << "</port></output></layer></layers><edges>";
    for (std::size_t i = 0; i < fixture.inputs.size(); ++i) {
        xml << "<edge from-layer=\"" << i << "\" from-port=\"0\" to-layer=\""
            << op_id << "\" to-port=\"" << i << "\"/>";
    }
    xml << "<edge from-layer=\"" << op_id << "\" from-port=\"" << fixture.inputs.size()
        << "\" to-layer=\"" << result_id << "\" to-port=\"0\"/></edges></net>";
    return xml.str();
}

std::shared_ptr<ov::Model> read_fixture(ov::Core& core, const Fixture& fixture) {
    // The fixture has no Const layers, so a one-byte weights tensor is enough
    // and no file or device API is involved.
    ov::Tensor weights(ov::element::u8, ov::Shape{1});
    return core.read_model(make_ir(fixture), weights);
}

Fixture q_fixture() {
    return {"OtcCrossQPack", {{1, 1, 2048, 1536}}, {"FP32"},
            {12, 16, 128, 128}, "FP16"};
}

Fixture kv_fixture() {
    return {"OtcCrossKVPack", {{1, 1, 512, 1536}}, {"FP32"},
            {4, 128, 12, 128}, "FP16"};
}

Fixture unpack_fixture() {
    return {"OtcCrossContextUnpack", {{12, 16, 128, 128}}, {"FP16"},
            {1, 1, 2048, 1536}, "FP32"};
}

Fixture sparse_fixture(const std::string& type, std::size_t key_blocks, std::size_t slots) {
    return {type,
            {{12, 16, 128, 128}, {key_blocks, 128, 12, 128},
             {key_blocks, 128, 12, 128}, {12, 16, 1, slots}},
            {"FP16", "FP16", "FP16", "I32"}, {12, 16, 128, 128}, "FP16"};
}

Fixture split_partial_fixture(std::size_t key_blocks) {
    return {"OtcPlusSplitKeyPartial",
            {{12, 16, 128, 128}, {key_blocks, 128, 12, 128},
             {key_blocks, 128, 12, 128}, {12, 16, 1, key_blocks}},
            {"FP16", "FP16", "FP16", "I32"}, {48, 16, 128, 130}, "FP32"};
}

Fixture split_merge_fixture() {
    return {"OtcPlusSplitKeyMerge", {{48, 16, 128, 130}}, {"FP32"},
            {12, 16, 128, 128}, "FP16"};
}

void expect_failure(ov::Core& core, Fixture fixture, const std::string& label) {
    try {
        (void)read_fixture(core, fixture);
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(label + ": malformed fixture accepted");
}

void check_model_contract(ov::Core& core) {
    for (const auto& fixture : {q_fixture(), kv_fixture(), unpack_fixture(),
                                sparse_fixture("OtcCrossPreciseXmx", 4, 4),
                                sparse_fixture("OtcPlusSparseXmx", 5100, 5100),
                                split_partial_fixture(2550), split_merge_fixture()}) {
        const auto model = read_fixture(core, fixture);
        require(model->outputs().size() == 1, fixture.type + ": output count changed");
        std::size_t custom_count = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (node->get_type_name() == fixture.type) {
                ++custom_count;
                require(std::string(node->get_type_info().version_id) == "extension",
                        fixture.type + ": extension version changed");
            }
        }
        require(custom_count == 1, fixture.type + ": custom node was not retained");
        const auto expected_type = fixture.output_precision == "FP16"
                                       ? ov::element::f16
                                       : fixture.output_precision == "FP32" ? ov::element::f32 : ov::element::i32;
        require(model->output(0).get_element_type() == expected_type,
                fixture.type + ": output precision changed");
        require(model->output(0).get_shape() == fixture.output,
                fixture.type + ": output shape changed");
    }
}

void check_split_factory() {
    const auto model = otc::ai::makeFlashSplitKeyAttentionModel(2550U);
    require(model->inputs().size() == 4, "split factory input count changed");
    require(model->outputs().size() == 1, "split factory output count changed");
    const std::vector<Shape> expected_shapes = {
        {12, 16, 128, 128}, {2550, 128, 12, 128},
        {2550, 128, 12, 128}, {12, 16, 1, 2550},
    };
    const std::vector<ov::element::Type> expected_types = {
        ov::element::f16, ov::element::f16, ov::element::f16, ov::element::i32,
    };
    const std::vector<std::string> expected_names = {"Q", "K", "V", "blocks"};
    for (std::size_t i = 0; i < expected_shapes.size(); ++i) {
        require(model->inputs()[i].get_shape() == expected_shapes[i],
                "split factory input shape changed");
        require(model->inputs()[i].get_element_type() == expected_types[i],
                "split factory input type changed");
        require(model->inputs()[i].get_any_name() == expected_names[i],
                "split factory input name changed");
    }
    require(model->output(0).get_shape() == Shape{12, 16, 128, 128},
            "split factory output shape changed");
    require(model->output(0).get_element_type() == ov::element::f16,
            "split factory output type changed");
    require(model->output(0).get_any_name() == "attention",
            "split factory output name changed");

    std::size_t partial_count = 0U;
    std::size_t merge_count = 0U;
    for (const auto& node : model->get_ordered_ops()) {
        if (std::string(node->get_type_name()) == "OtcPlusSplitKeyPartial") {
            ++partial_count;
            require(node->get_output_element_type(0) == ov::element::f32,
                    "split partial type changed");
            require(node->get_output_shape(0) == Shape{48, 16, 128, 130},
                    "split partial shape changed");
        } else if (std::string(node->get_type_name()) == "OtcPlusSplitKeyMerge") {
            ++merge_count;
        }
    }
    require(partial_count == 1U && merge_count == 1U,
            "split factory custom-node count changed");

    bool rejected = false;
    try {
        (void)otc::ai::makeFlashSplitKeyAttentionModel(0U);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "split factory accepted zero key capacity");
    rejected = false;
    try {
        (void)otc::ai::makeFlashSplitKeyAttentionModel(2551U);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "split factory accepted key capacity above 2550");
}

void check_rejections(ov::Core& core) {
    const auto check_unary = [&](const Fixture& fixture, const std::string& label) {
        auto bad = fixture;
        bad.inputs.clear();
        bad.precisions.clear();
        expect_failure(core, bad, label + " zero arity");
        bad = fixture;
        bad.inputs.push_back(fixture.inputs.front());
        bad.precisions.push_back(fixture.precisions.front());
        expect_failure(core, bad, label + " two arity");
        bad = fixture;
        bad.precisions[0] = bad.precisions[0] == "FP16" ? "FP32" : "FP16";
        expect_failure(core, bad, label + " wrong type");
        bad = fixture;
        ++bad.inputs[0].back();
        expect_failure(core, bad, label + " wrong shape");
    };
    check_unary(q_fixture(), "QPack");
    check_unary(kv_fixture(), "KVPack");
    check_unary(unpack_fixture(), "ContextUnpack");

    const auto check_sparse = [&](const std::string& type, const std::string& label) {
        auto bad = sparse_fixture(type, 4, 4);
        bad.inputs.pop_back();
        bad.precisions.pop_back();
        expect_failure(core, bad, label + " three arity");
        bad = sparse_fixture(type, 4, 4);
        bad.precisions[0] = "FP32";
        expect_failure(core, bad, label + " wrong Q type");
        bad = sparse_fixture(type, 4, 4);
        bad.inputs[0][1] = 15;
        expect_failure(core, bad, label + " wrong query count");
        bad = sparse_fixture(type, 4, 4);
        bad.inputs[1][3] = 127;
        expect_failure(core, bad, label + " wrong K shape");
        bad = sparse_fixture(type, 4, 4);
        bad.inputs[2][0] = 5;
        expect_failure(core, bad, label + " K/V mismatch");
    };
    check_sparse("OtcCrossPreciseXmx", "Precise");
    check_sparse("OtcPlusSparseXmx", "Sparse");

    auto split = split_partial_fixture(4);
    split.inputs[0][1] = 8;
    expect_failure(core, split, "Split partial query count");
    split = split_partial_fixture(4);
    split.inputs[1][0] = 2551;
    split.inputs[2][0] = 2551;
    split.inputs[3][3] = 2551;
    expect_failure(core, split, "Split partial key cap");
    split = split_partial_fixture(4);
    split.inputs[3][3] = 3;
    expect_failure(core, split, "Split partial ID capacity");
    split = split_partial_fixture(4);
    split.precisions[3] = "FP16";
    expect_failure(core, split, "Split partial ID type");

    auto merge = split_merge_fixture();
    merge.inputs.clear();
    merge.precisions.clear();
    expect_failure(core, merge, "Split merge zero arity");
    merge = split_merge_fixture();
    merge.precisions[0] = "FP16";
    expect_failure(core, merge, "Split merge accumulator type");
    merge = split_merge_fixture();
    merge.inputs[0][3] = 129;
    expect_failure(core, merge, "Split merge accumulator shape");

    auto bad = sparse_fixture("OtcPlusSparseXmx", 5101, 1);
    expect_failure(core, bad, "key-block cap");
    bad = sparse_fixture("OtcPlusSparseXmx", 4, 0);
    expect_failure(core, bad, "zero slots");
    bad = sparse_fixture("OtcPlusSparseXmx", 4, 5);
    expect_failure(core, bad, "slots exceed keys");
    bad = sparse_fixture("OtcPlusSparseXmx", 4, 4);
    bad.inputs[0][0] = 11;
    expect_failure(core, bad, "head count");
    bad = sparse_fixture("OtcPlusSparseXmx", 4, 4);
    bad.inputs[1][1] = 127;
    bad.inputs[2][1] = 127;
    expect_failure(core, bad, "token-block shape");
    bad = q_fixture();
    bad.inputs[0][3] = 0;
    expect_failure(core, bad, "zero dimension");
    bad = sparse_fixture("OtcPlusSparseXmx", 4, 4);
    bad.precisions[3] = "FP16";
    expect_failure(core, bad, "ID type");
    bad = sparse_fixture("OtcCrossPreciseXmx", 5, 4);
    expect_failure(core, bad, "precise key count");
    bad = q_fixture();
    bad.version = "opset1";
    expect_failure(core, bad, "extension version");
}

std::shared_ptr<ov::Node> find_custom(const std::shared_ptr<ov::Model>& model,
                                      const std::string& type) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() == type)
            return node;
    }
    throw std::runtime_error(type + ": custom node missing");
}

void check_clone_validation(ov::Core& core) {
    const std::vector<Fixture> fixtures = {
        q_fixture(), kv_fixture(), unpack_fixture(),
        sparse_fixture("OtcCrossPreciseXmx", 4, 4),
        sparse_fixture("OtcPlusSparseXmx", 4, 4), split_partial_fixture(4),
        split_merge_fixture(),
    };
    for (const auto& fixture : fixtures) {
        const auto model = read_fixture(core, fixture);
        const auto node = find_custom(model, fixture.type);
        const auto clone = node->clone_with_new_inputs(node->input_values());
        require(clone->get_type_name() == fixture.type, fixture.type + ": clone type changed");
        require(clone->get_output_size() == 1, fixture.type + ": clone output count changed");
        require(clone->get_output_element_type(0) == node->get_output_element_type(0) &&
                    clone->get_output_shape(0) == node->get_output_shape(0),
                fixture.type + ": clone output ABI changed");

        std::vector<ov::Dimension> dynamic_dims;
        dynamic_dims.reserve(fixture.inputs[0].size());
        for (std::size_t i = 0; i < fixture.inputs[0].size(); ++i) {
            dynamic_dims.emplace_back(i == 0
                                          ? ov::Dimension::dynamic()
                                          : ov::Dimension(static_cast<std::int64_t>(fixture.inputs[0][i])));
        }
        auto dynamic = std::make_shared<ov::op::v0::Parameter>(
            node->get_input_element_type(0), ov::PartialShape(dynamic_dims));
        auto inputs = node->input_values();
        inputs[0] = dynamic->output(0);
        try {
            (void)node->clone_with_new_inputs(inputs);
        } catch (const std::exception&) {
            continue;
        }
        throw std::runtime_error(fixture.type + ": dynamic clone input was accepted");
    }
}

void check_serialize_roundtrip(ov::Core& core) {
    const auto root = std::filesystem::temp_directory_path() /
                      ("otc-flash-ops-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    // Never reuse or clean a pre-existing path, even if the name collided.
    const bool created = std::filesystem::create_directory(root, ec);
    require(created && !ec, "could not create fresh structural-test temporary directory");
    struct Cleanup final {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{root};

    const std::vector<Fixture> fixtures = {
        q_fixture(), kv_fixture(), unpack_fixture(),
        sparse_fixture("OtcCrossPreciseXmx", 4, 4),
        sparse_fixture("OtcPlusSparseXmx", 4, 3), split_partial_fixture(4),
        split_merge_fixture(),
    };
    std::size_t index = 0;
    for (const auto& fixture : fixtures) {
        const auto model = read_fixture(core, fixture);
        const auto xml = root / ("fixture-" + std::to_string(index) + ".xml");
        const auto bin = root / ("fixture-" + std::to_string(index) + ".bin");
        ov::serialize(model, xml, bin);
        const auto roundtrip = core.read_model(xml, bin);
        (void)find_custom(roundtrip, fixture.type);
        require(roundtrip->output(0).get_element_type() == model->output(0).get_element_type() &&
                    roundtrip->output(0).get_shape() == model->output(0).get_shape(),
                fixture.type + ": roundtrip output ABI changed");
        ++index;
    }
}

}  // namespace

int main() {
    try {
        ov::Core core;
        otc::ai::registerFlashOps(core);
        check_model_contract(core);
        check_split_factory();
        check_rejections(core);
        check_clone_validation(core);
        check_serialize_roundtrip(core);
        std::cout << "flash_ops_cpu_pass structural_read_model_clone_serialize_only\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "flash_ops_cpu_fail: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
