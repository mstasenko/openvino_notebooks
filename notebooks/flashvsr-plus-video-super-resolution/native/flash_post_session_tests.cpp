#include "flash_post_session.hpp"

#include "flash-assets/flash_post_assets.hpp"
#include "flash_ops.hpp"
#include "seedvr_assets.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/assign.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/result.hpp>

#include <filesystem>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using otc::ai::SeedVrOwnedIr;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct LoadedModel {
    // The model may retain pointers into the authenticated XML/BIN owner.
    SeedVrOwnedIr ir;
    std::shared_ptr<ov::Model> model;
};

LoadedModel load_fixed_state(const std::filesystem::path& assets) {
    auto ir = otc::ai::readSeedVrIr(assets,
                                    otc::ai::kFlashPostXml,
                                    otc::ai::kFlashPostStructuralBin);
    ov::Core core;
    otc::ai::registerFlashOps(core);
    const ov::Tensor weights(ov::element::u8,
                             ov::Shape{ir.binBytes()},
                             ir.binData());
    auto model = core.read_model(ir.xml(), weights);
    return {std::move(ir), std::move(model)};
}

template <class Mutator>
void expect_portable_rejected(const ov::Model& source, const std::string& label,
                              Mutator mutator) {
    auto malformed = source.clone();
    mutator(*malformed);
    try {
        otc::ai::prepareFlashPostModel(*malformed);
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(label + ": malformed portable graph was accepted");
}

template <class Mutator>
void expect_portable_rejected_before_state_mutation(const ov::Model& source,
                                                    const std::string& label,
                                                    Mutator mutator) {
    auto malformed = source.clone();
    mutator(*malformed);
    try {
        otc::ai::prepareFlashPostModel(*malformed);
    } catch (const std::exception&) {
        require(malformed->get_results().size() == source.get_results().size() &&
                    malformed->get_variables().empty() && malformed->get_sinks().empty(),
                label + ": rejection happened after state mutation");
        return;
    }
    throw std::runtime_error(label + ": malformed portable graph was accepted");
}

void check_portable_source(const ov::Model& source) {
    require(source.get_results().size() == 3 && source.get_sinks().empty() &&
                source.get_variables().empty(), "portable fixture ABI differs");
    expect_portable_rejected(source, "wrong FFN2 gate input", [](ov::Model& model) {
        const auto linear = model.output("ffn2_linear_storage").get_node()->input_value(0);
        const auto gate = model.output("ffn2_gate_storage").get_node()->input_value(0).get_node_shared_ptr();
        bool replaced = false;
        for (std::size_t i = 0; i < gate->get_input_size(); ++i) {
            if (gate->input_value(i) == linear) {
                gate->input(i).replace_source_output(model.input("x"));
                replaced = true;
            }
        }
        require(replaced, "gate mutation did not take effect");
    });
    expect_portable_rejected(source, "preexisting state", [](ov::Model& model) {
        model.add_variables({std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{1, 4096, 1536},
                                      ov::element::bf16, "preexisting"})});
    });
    expect_portable_rejected(source, "missing private Result", [](ov::Model& model) {
        model.remove_result(std::dynamic_pointer_cast<ov::op::v0::Result>(
            model.output("ffn2_linear_storage").get_node_shared_ptr()));
    });
    expect_portable_rejected_before_state_mutation(source, "FP16 intermediate", [](ov::Model& model) {
        const auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
            model.output("x_out").get_node_shared_ptr());
        const auto gate = model.output("ffn2_gate_storage").get_node()->input_value(0)
            .get_node_shared_ptr();
        const auto output = result->input_value(0).get_node_shared_ptr();
        bool replaced = false;
        for (std::size_t i = 0; i < output->get_input_size(); ++i) {
            if (output->input_value(i).get_node_shared_ptr() == gate) continue;
            const auto toF16 = std::make_shared<ov::op::v0::Convert>(
                output->input_value(i), ov::element::f16);
            const auto back = std::make_shared<ov::op::v0::Convert>(
                toF16, ov::element::bf16);
            output->input(i).replace_source_output(back);
            replaced = true;
            break;
        }
        require(replaced, "FP16 fixture could not find non-gate output input");
        model.validate_nodes_and_infer_types();
    });
    expect_portable_rejected_before_state_mutation(source, "numeric reader of raw BF16 transport",
                                                    [](ov::Model& model) {
        std::shared_ptr<ov::Node> packed;
        for (const auto& node : model.get_ordered_ops())
            if (std::string(node->get_type_name()) == "OtcCrossQPack") {
                packed = node;
                break;
            }
        require(packed != nullptr, "raw transport fixture has no QPack");
        // No new F16 producer: only the reader is wrong. Interpreting these
        // raw BF16 bits numerically as F16 must not bypass the transport guard.
        const auto numeric = std::make_shared<ov::op::v0::Convert>(packed, ov::element::f32);
        const auto axes = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 1, 2, 3});
        const auto sum = std::make_shared<ov::op::v1::ReduceSum>(numeric, axes, false);
        const auto scalar = std::make_shared<ov::op::v0::Convert>(sum, ov::element::bf16);
        const auto output = model.output("x_out").get_node()->input_value(0).get_node_shared_ptr();
        const auto gate = model.output("ffn2_gate_storage").get_node()->input_value(0).get_node_shared_ptr();
        bool replaced = false;
        for (std::size_t i = 0; i < output->get_input_size(); ++i) {
            if (output->input_value(i).get_node_shared_ptr() == gate) continue;
            output->input(i).replace_source_output(
                std::make_shared<ov::op::v1::Add>(output->input_value(i), scalar));
            replaced = true;
            break;
        }
        require(replaced, "raw transport fixture could not find residual input");
        model.validate_nodes_and_infer_types();
    });
    expect_portable_rejected_before_state_mutation(source, "oversized structural Constant",
                                                    [](ov::Model& model) {
        const auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(
            model.output("x_out").get_node_shared_ptr());
        const auto gate = model.output("ffn2_gate_storage").get_node()->input_value(0)
            .get_node_shared_ptr();
        const auto output = result->input_value(0).get_node_shared_ptr();
        const std::vector<float> values(4096, 0.0f);
        const auto oversized = ov::op::v0::Constant::create(
            ov::element::bf16, {1, 4096, 1}, values);
        bool replaced = false;
        for (std::size_t i = 0; i < output->get_input_size(); ++i) {
            if (output->input_value(i).get_node_shared_ptr() == gate) continue;
            const auto injected = std::make_shared<ov::op::v1::Add>(
                output->input_value(i), oversized);
            output->input(i).replace_source_output(injected);
            replaced = true;
            break;
        }
        require(replaced, "Constant fixture could not find non-gate output input");
        model.validate_nodes_and_infer_types();
    });
}

template <class Mutator>
void expect_rejected(const ov::Model& source, const std::string& label, Mutator mutator) {
    auto malformed = source.clone();
    mutator(*malformed);
    try {
        otc::ai::validateFlashPostModel(*malformed);
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(label + ": malformed model was accepted");
}

std::shared_ptr<ov::op::v0::Parameter> parameter(ov::Model& model, const std::string& name) {
    return std::dynamic_pointer_cast<ov::op::v0::Parameter>(
        model.input(name).get_node_shared_ptr());
}

std::shared_ptr<ov::op::v0::Parameter> first_weight(ov::Model& model) {
    for (const auto& input : model.inputs()) {
        if (input.get_any_name().rfind("weight__", 0) == 0)
            return std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.get_node_shared_ptr());
    }
    throw std::runtime_error("fixed POST model has no learned parameter");
}

void check_mutations(const ov::Model& source) {
    otc::ai::validateFlashPostModel(source);

    expect_rejected(source, "missing data port", [](ov::Model& model) {
        model.remove_parameter(parameter(model, "x"));
    });
    expect_rejected(source, "extra data port", [](ov::Model& model) {
        auto extra = std::make_shared<ov::op::v0::Parameter>(
            ov::element::bf16, ov::PartialShape{1, 4096, 1536});
        extra->output(0).set_names({"unexpected_input"});
        model.add_parameters({extra});
    });
    expect_rejected(source, "renamed data port", [](ov::Model& model) {
        auto x = parameter(model, "x");
        x->output(0).set_names({"x_renamed"});
    });
    expect_rejected(source, "wrong data type", [](ov::Model& model) {
        auto x = parameter(model, "x");
        x->set_element_type(ov::element::f16);
        x->validate_and_infer_types();
    });
    expect_rejected(source, "wrong data shape", [](ov::Model& model) {
        auto x = parameter(model, "x");
        x->set_partial_shape({1, 4096, 1535});
        x->validate_and_infer_types();
    });
    expect_rejected(source, "renamed learned port", [](ov::Model& model) {
        auto weight = first_weight(model);
        weight->set_friendly_name("learned_name_tampered");
        weight->output(0).set_names({"weight__tampered"});
    });
    expect_rejected(source, "wrong learned schema", [](ov::Model& model) {
        auto weight = first_weight(model);
        weight->set_partial_shape({1536, 1535});
        weight->validate_and_infer_types();
    });
    expect_rejected(source, "missing state sink", [](ov::Model& model) {
        require(model.get_sinks().size() == 2, "fixture state sink count changed");
        model.remove_sink(model.get_sinks().front());
    });
    expect_rejected(source, "extra state sink", [](ov::Model& model) {
        require(model.get_sinks().size() == 2, "fixture state sink count changed");
        // A second node is required: add_sinks may deduplicate the same handle.
        auto extra = std::make_shared<ov::op::v6::Assign>(
            model.input("x"), model.get_variables().front());
        model.add_sinks({extra});
        require(model.get_sinks().size() == 3, "extra sink mutation did not take effect");
    });
    expect_rejected(source, "missing state variable", [](ov::Model& model) {
        require(model.get_variables().size() == 2, "fixture state variable count changed");
        model.remove_variable(model.get_variables().front());
    });
    expect_rejected(source, "added ReadValue state", [](ov::Model& model) {
        auto read = std::make_shared<ov::op::v6::ReadValue>(model.get_variables().front());
        const auto result = model.get_results().front();
        // Preserve every original branch/state/port, so only the no-feedback
        // invariant rejects this otherwise shape-compatible mutation.
        auto add = std::make_shared<ov::op::v1::Add>(result->input_value(0), read);
        add->output(0).set_names({"x_out"});
        result->input(0).replace_source_output(add);
    });
    expect_rejected(source, "extra Assign state", [](ov::Model& model) {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
            ov::PartialShape{1, 4096, 1536}, ov::element::bf16, "unexpected_assign"});
        auto assign = std::make_shared<ov::op::v6::Assign>(model.input("x"), variable);
        model.add_variables({variable});
        model.add_sinks({assign});
    });
    expect_rejected(source, "extra output", [](ov::Model& model) {
        model.add_results({std::make_shared<ov::op::v0::Result>(model.output("x_out"))});
    });
    expect_rejected(source, "missing output", [](ov::Model& model) {
        require(model.get_results().size() == 1, "fixture output count changed");
        model.remove_result(model.get_results().front());
    });
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto assets = argc == 2 ? std::filesystem::path(argv[1])
                                     : std::filesystem::path("assets/flash-assets");
        auto loaded = load_fixed_state(assets);
        check_portable_source(*loaded.model);
        otc::ai::prepareFlashPostModel(*loaded.model);
        check_mutations(*loaded.model);
        expect_portable_rejected(*loaded.model, "already prepared graph", [](ov::Model&) {});
        std::cout << "flash_post_session_model_validation=pass device_calls=0 compile_calls=0 infer_calls=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "flash_post_session_model_validation=fail: " << error.what() << '\n';
        return 1;
    }
}
