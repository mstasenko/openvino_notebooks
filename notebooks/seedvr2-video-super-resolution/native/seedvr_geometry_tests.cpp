#include "seedvr_geometry.hpp"
#include "ai_enhance_core.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/op/loop.hpp>
#include <openvino/op/constant.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace op = ov::opset13;
using namespace otc::ai;
using Node = std::shared_ptr<ov::Node>;
using Value = ov::Output<ov::Node>;
using Constant = ov::op::v0::Constant;
constexpr std::array<std::array<int,2>,6> kShapes = {{{24,30},{30,24},{8,64},{16,24},{48,60},{135,240}}};

void check(bool yes, const char* message) { if (!yes) throw std::runtime_error(message); }
auto ci(const ov::Shape& dims, std::vector<std::int64_t> values) {
  return Constant::create(ov::element::i64,dims,values);
}
auto cf(const ov::Shape& dims, std::vector<float> values) {
  return Constant::create(ov::element::f32,dims,values);
}
auto parameter(ov::element::Type type, const ov::Shape& shape, const char* name) {
  auto p = std::make_shared<op::Parameter>(type,shape);
  p->output(0).get_tensor().set_names({name});
  // Parameter friendly names deliberately do not encode the public contract.
  p->set_friendly_name(std::string("fixture_")+name);
  return p;
}
Node named(ov::Model& model, const std::string& name) {
  for (auto n : model.get_ordered_ops()) if (n->get_friendly_name()==name) return n;
  throw std::runtime_error("test semantic node missing");
}

struct Attention { Value video,sum,count; };
Attention attention(const Value& idx, const Value& mask, const Value& rope) {
  const auto dims = std::make_shared<op::ShapeOf>(idx,ov::element::i64);
  const auto target = std::make_shared<op::Concat>(ov::OutputVector{dims,ci({1},{2560})},0);
  const auto maskSum = std::make_shared<op::ReduceSum>(std::make_shared<op::Convert>(mask,ov::element::f32),ci({2},{0,1}),false);
  const auto ropeSum = std::make_shared<op::ReduceSum>(rope,ci({3},{0,1,2}),false);
  const auto zero = std::make_shared<op::Multiply>(std::make_shared<op::Add>(maskSum,ropeSum),cf({}, {0}));
  const auto video = std::make_shared<op::Broadcast>(zero,target);
  const auto sum = std::make_shared<op::Broadcast>(zero,ci({2},{58,2560}));
  const auto count = std::make_shared<op::Convert>(std::make_shared<op::Gather>(dims,ci({1},{0}),ci({}, {0})),ov::element::f32);
  return {video,sum,count};
}

std::shared_ptr<ov::Model> fixture(bool odd) {
  const auto m = makeSeedVrWindowMetadata(1,48,60,odd);
  const auto n = m.windowCount, width = m.maxVideoLen;
  const std::size_t chunk = odd?7U:8U;
  const auto full = n/chunk*chunk;
  ov::ParameterVector qkv;
  for (int i=0;i<6;++i) qkv.push_back(parameter(ov::element::f32,{i<3?2880U:58U,20,128},("qkv"+std::to_string(i)).c_str()));
  const auto bodyIdx = parameter(ov::element::i64,{chunk,width},"body_idx");
  const auto bodyMask = parameter(ov::element::boolean,{chunk,width+58},"body_mask");
  const auto bodyRope = parameter(ov::element::f32,{chunk,width+58,126},"body_rope");
  const auto sumState = parameter(ov::element::f32,{58,2560},"sum_state");
  const auto countState = parameter(ov::element::f32,{1},"count_state");
  const auto a = attention(bodyIdx,bodyMask,bodyRope);
  const auto packed = std::make_shared<op::Reshape>(a.video,ci({3},{1,static_cast<std::int64_t>(chunk*width),2560}),false);
  packed->set_friendly_name("pack_full_loop_video_scan_slice");
  const auto sum = std::make_shared<op::Add>(a.sum,sumState);
  const auto count = std::make_shared<op::Add>(a.count,countState);
  auto bodyParams = qkv;
  bodyParams.insert(bodyParams.end(),{bodyIdx,bodyMask,bodyRope,sumState,countState});
  const auto yes = Constant::create(ov::element::boolean,{},std::vector<char>{1});
  const auto body = std::make_shared<ov::Model>(ov::OutputVector{yes,packed,sum,count},bodyParams);
  const auto vid = parameter(ov::element::f32,{2880,2560},"vid");
  const auto txt = parameter(ov::element::f32,{58,2560},"txt");
  const auto emb = parameter(ov::element::f32,{1,15360},"emb");
  const auto idx = parameter(ov::element::i64,{n,width},"window_idx");
  const auto mask = parameter(ov::element::boolean,{n,width+58},"window_key_mask");
  const auto rope = parameter(ov::element::f32,{n,width+58,126},"window_rope_freqs");
  const auto weight = parameter(ov::element::f16,{2,2},"weight_000");
  auto loop = std::make_shared<ov::op::v5::Loop>(ci({}, {static_cast<std::int64_t>(full/chunk)}),yes);
  loop->set_friendly_name("full_chunk_attention_loop");
  loop->set_function(body);
  loop->set_special_body_ports({-1,0});
  for (std::size_t i=0;i<6;++i) {
    const Value original = i<3?vid->output(0):txt->output(0);
    const auto tokens = std::make_shared<op::Gather>(std::make_shared<op::ShapeOf>(original,ov::element::i64),ci({1},{0}),ci({}, {0}));
    const auto shape = std::make_shared<op::Concat>(ov::OutputVector{tokens,ci({2},{20,128})},0);
    loop->set_invariant_input(qkv[i],std::make_shared<op::Reshape>(original,shape,false));
  }
  std::array<Value,3> tail;
  const std::array<Value,3> metadata = {idx,mask,rope};
  const std::array<const char*,3> names = {"window_idx","window_key_mask","window_rope_freqs"};
  const std::array<std::shared_ptr<op::Parameter>,3> bodyMeta = {bodyIdx,bodyMask,bodyRope};
  for (std::size_t i=0;i<3;++i) {
    auto first = std::make_shared<op::Slice>(metadata[i],ci({1},{0}),ci({1},{static_cast<std::int64_t>(full)}),ci({1},{1}),ci({1},{0}));
    first->set_friendly_name(std::string("slice_full_loop_")+names[i]);
    auto last = std::make_shared<op::Slice>(metadata[i],ci({1},{static_cast<std::int64_t>(full)}),ci({1},{static_cast<std::int64_t>(n)}),ci({1},{1}),ci({1},{0}));
    last->set_friendly_name(std::string("slice_exact_tail_")+names[i]);
    tail[i] = last;
    loop->set_sliced_input(bodyMeta[i],first,0,chunk,chunk,-1,0);
  }
  for (const auto& entry : std::array<std::pair<std::shared_ptr<op::Parameter>,Value>,2>{{{sumState,sum},{countState,count}}}) {
    const auto dims = entry.first->get_shape();
    const auto zeros = Constant::create(ov::element::f16,dims,std::vector<ov::float16>(ov::shape_size(dims),ov::float16(0)));
    loop->set_merged_input(entry.first,std::make_shared<op::Convert>(zeros,ov::element::f32),entry.second);
  }
  const auto scanned = loop->get_concatenated_slices(packed,0,1,1,-1,0);
  const auto fullVideo = std::make_shared<op::Reshape>(scanned,ci({3},{static_cast<std::int64_t>(full),static_cast<std::int64_t>(width),2560}),false);
  fullVideo->set_friendly_name("reshape_full_loop_video_windows");
  const auto t = attention(tail[0],tail[1],tail[2]);
  const auto videoMerge = std::make_shared<op::Concat>(ov::OutputVector{fullVideo,t.video},0);
  videoMerge->set_friendly_name("loop_plus_tail_video_windows");
  const auto sumMerge = std::make_shared<op::Add>(loop->get_iter_value(sum,-1),t.sum);
  sumMerge->set_friendly_name("add_tail_text_sum");
  const auto countMerge = std::make_shared<op::Add>(loop->get_iter_value(count,-1),t.count);
  countMerge->set_friendly_name("add_tail_text_count");
  const auto flat = std::make_shared<op::Reshape>(videoMerge,ci({2},{static_cast<std::int64_t>(n*width),2560}),false);
  flat->set_friendly_name("flatten_loop_plus_tail_video_windows");
  const auto video = std::make_shared<op::Gather>(flat,ci({2880},m.windowReverseIdx),ci({}, {0}));
  video->set_friendly_name("exact_source_order_video_gather");
  const auto text = std::make_shared<op::Divide>(sumMerge,countMerge);
  text->set_friendly_name("averaged_text_output");
  video->output(0).get_tensor().set_names({"vid_out"});
  text->output(0).get_tensor().set_names({"txt_out"});
  return std::make_shared<ov::Model>(ov::OutputVector{video,text},ov::ParameterVector{vid,txt,emb,idx,mask,rope,weight});
}

void verify(ov::Model& model, const SeedVrWindowMetadata& m) {
  const auto oldParams = model.get_parameters();
  std::map<Node,std::vector<unsigned char>> constants;
  const auto loop = named(model,"full_chunk_attention_loop");
  const auto oldLoop = std::dynamic_pointer_cast<ov::op::v5::Loop>(loop);
  const auto oldSlice = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::SliceInputDescription>(oldLoop->get_input_descriptions()[6]);
  const auto oldChunk = static_cast<std::size_t>(oldSlice->m_part_size);
  const auto originalOps = model.get_ordered_ops();
  const auto originalBodyOps = oldLoop->get_function()->get_ordered_ops();
  const auto originalDescriptions = oldLoop->get_input_descriptions();
  std::map<Node,std::vector<ov::Output<ov::Node>>> originalEdges;
  std::map<Node,std::size_t> originalIds;
  for (const auto& graph : {originalOps,originalBodyOps}) for (const auto& node : graph) {
    originalEdges.emplace(node,node->input_values());
    originalIds.emplace(node,node->get_instance_id());
  }
  const auto oldReverse = std::dynamic_pointer_cast<Constant>(named(model,"exact_source_order_video_gather")->input_value(1).get_node_shared_ptr());
  const bool identity = model.input("vid").get_shape()==ov::Shape{m.sourceTokenCount,2560} &&
      model.input("window_idx").get_shape()==ov::Shape{m.windowCount,m.maxVideoLen} &&
      model.input("window_key_mask").get_shape()==ov::Shape{m.windowCount,m.attentionLen} &&
      model.input("window_rope_freqs").get_shape()==ov::Shape{m.windowCount,m.attentionLen,126} &&
      oldReverse->cast_vector<std::int64_t>()==m.windowReverseIdx;
  std::set<Node> zeros;
  for (std::size_t port : {11U,12U}) {
    auto n = loop->input_value(port).get_node_shared_ptr();
    if (n->get_type_name()==std::string("Convert")) n=n->input_value(0).get_node_shared_ptr();
    zeros.insert(n);
  }
  std::set<Node> beforeHalf,afterHalf;
  for (const auto& n : model.get_ordered_ops()) if (const auto c = std::dynamic_pointer_cast<Constant>(n)) {
    const auto bytes = c->get_byte_size();
    const auto data = static_cast<const unsigned char*>(c->get_data_ptr());
    constants[n] = std::vector<unsigned char>(data,data+bytes);
    if (c->get_element_type()==ov::element::f16 && !zeros.count(n)) beforeHalf.insert(n);
  }
  patchSeedVrGeometry(model,m);
  if (identity) {
    check(model.get_ordered_ops()==originalOps,"identity changed root nodes/order");
    check(oldLoop->get_function()->get_ordered_ops()==originalBodyOps,"identity changed body nodes/order");
    check(oldLoop->get_input_descriptions()==originalDescriptions,"identity replaced Loop descriptors");
    for (const auto& entry : originalEdges) {
      check(entry.first->input_values()==entry.second,"identity rewired an edge");
      check(entry.first->get_instance_id()==originalIds.at(entry.first),"identity changed a node id");
    }
  }
  check(model.get_parameters()==oldParams,"Parameter set changed");
  check(model.output("vid_out").get_shape()==ov::Shape{m.sourceTokenCount,2560},"video shape differs");
  check(model.output("txt_out").get_shape()==ov::Shape{58,2560},"text shape differs");
  check(model.input("window_idx").get_shape()==ov::Shape{m.windowCount,m.maxVideoLen},"index shape differs");
  check(model.input("window_key_mask").get_shape()==ov::Shape{m.windowCount,m.attentionLen},"mask shape differs");
  check(model.input("window_rope_freqs").get_shape()==ov::Shape{m.windowCount,m.attentionLen,126},"RoPE shape differs");
  std::size_t loops = 0;
  for (auto n : model.get_ordered_ops()) {
    if (n->get_type_name()==std::string("Loop")) ++loops;
    const auto original = constants.find(n);
    const auto c = std::dynamic_pointer_cast<Constant>(n);
    if (c && c->get_element_type()==ov::element::f16 && !zeros.count(n)) afterHalf.insert(n);
    if (original!=constants.end()) {
      check(c->get_byte_size()==original->second.size() && std::memcmp(c->get_data_ptr(),original->second.data(),original->second.size())==0,"remaining Constant value changed");
    } else if (c) check(c->get_element_type()==ov::element::i64,"unexpected added Constant dtype");
  }
  check(loops==(m.windowCount>1?1U:0U),"wrong reachable Loop count");
  check(beforeHalf==afterHalf,"stored FP16 constants other than zero initializers pruned");
  if (m.windowCount>1) {
    const auto newLoop = std::dynamic_pointer_cast<ov::op::v5::Loop>(named(model,"full_chunk_attention_loop"));
    const auto chunk = std::min(oldChunk,m.windowCount-1);
    const auto full = (m.windowCount-1)/chunk*chunk;
    for (std::size_t i=6;i<9;++i) {
      const auto d = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::SliceInputDescription>(newLoop->get_input_descriptions()[i]);
      check(d->m_stride==static_cast<std::int64_t>(chunk) && d->m_part_size==static_cast<std::int64_t>(chunk),"template chunk policy differs");
    }
    check(newLoop->get_num_iterations()==static_cast<std::int64_t>(full/chunk),"Loop trip count differs");
  }
  const auto reverse = std::dynamic_pointer_cast<Constant>(named(model,"exact_source_order_video_gather")->input_value(1).get_node_shared_ptr());
  check(reverse->cast_vector<std::int64_t>()==m.windowReverseIdx,"reverse differs");
}

void wrapIf(ov::Model& model, std::size_t result) {
  const auto input = model.get_results()[result]->input_value(0);
  const auto thenParam = parameter(input.get_element_type(),input.get_shape(),"then_value");
  const auto elseParam = parameter(input.get_element_type(),input.get_shape(),"else_value");
  const auto thenBody = std::make_shared<ov::Model>(ov::OutputVector{thenParam},ov::ParameterVector{thenParam});
  const auto elseBody = std::make_shared<ov::Model>(ov::OutputVector{elseParam},ov::ParameterVector{elseParam});
  const auto conditional = std::make_shared<op::If>(Constant::create(ov::element::boolean,{},std::vector<char>{1}));
  conditional->set_then_body(thenBody); conditional->set_else_body(elseBody);
  conditional->set_input(input,thenParam,elseParam);
  const auto output = conditional->set_output(thenBody->get_results()[0],elseBody->get_results()[0]);
  output.get_tensor().set_names(model.get_results()[result]->output(0).get_names());
  model.get_results()[result]->input(0).replace_source_output(output);
  model.validate_nodes_and_infer_types();
}

void wrapIterator(ov::Model& model) {
  const auto input = model.get_results()[1]->input_value(0);
  const auto p = parameter(ov::element::f32,{1,2560},"iterator_value");
  const auto body = std::make_shared<ov::Model>(ov::OutputVector{p},ov::ParameterVector{p});
  const auto iterator = std::make_shared<op::TensorIterator>();
  iterator->set_function(body);
  iterator->set_sliced_input(p,input,0,1,1,-1,0);
  const auto output = iterator->get_concatenated_slices(p,0,1,1,-1,0);
  output.get_tensor().set_names({"txt_out"});
  model.get_results()[1]->input(0).replace_source_output(output);
  model.validate_nodes_and_infer_types();
}

template<class Change> void reject(Change change, bool identity = false) {
  auto model = fixture(true);
  auto m = makeSeedVrWindowMetadata(1,identity?48:8,identity?60:64,true);
  change(*model,m);
  bool rejected = false;
  try { patchSeedVrGeometry(*model,m); } catch (const std::invalid_argument&) { rejected = true; }
  check(rejected,"malformed graph/metadata accepted");
}

int main(int argc, char** argv) {
  try {
    for (bool odd : {false,true}) for (const auto hw : kShapes) {
      auto model = fixture(odd);
      verify(*model,makeSeedVrWindowMetadata(1,hw[0],hw[1],odd));
    }
    reject([](auto&,auto& m) { m.frames=2; });
    reject([](auto&,auto& m) { m.windowReverseIdx[1]=m.windowReverseIdx[0]; });
    reject([](auto&,auto& m) { m.windowReverseIdx[0]=-1; });
    reject([](auto&,auto& m) { m.windowLengths[0]=m.maxVideoLen+1; });
    reject([](auto&,auto& m) { m.windowReverseIdx[0]=static_cast<std::int64_t>(m.maxVideoLen-1); });
    reject([](auto& model,auto&) { named(model,"exact_source_order_video_gather")->set_friendly_name("missing"); });
    reject([](auto& model,auto&) { named(model,"slice_exact_tail_window_idx")->input(1).replace_source_output(ci({1},{0})); });
    reject([](auto& model,auto&) { named(model,"add_tail_text_sum")->input(0).replace_source_output(cf({58,2560},{0})); });
    // Matching public shapes/reverse data must not bypass any structural guards.
    reject([](auto& model,auto&) { named(model,"exact_source_order_video_gather")->set_friendly_name("missing"); },true);
    reject([](auto& model,auto&) { named(model,"slice_exact_tail_window_idx")->input(1).replace_source_output(ci({1},{0})); },true);
    reject([](auto& model,auto&) { named(model,"add_tail_text_sum")->input(0).replace_source_output(cf({58,2560},{0})); },true);
    reject([](auto& model,auto&) { wrapIf(model,1); });
    reject([](auto& model,auto&) { wrapIterator(model); });
    reject([](auto& model,auto&) {
      const auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(named(model,"full_chunk_attention_loop"));
      wrapIf(*loop->get_function(),2);
    });
    reject([](auto& model,auto&) {
      const auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(named(model,"full_chunk_attention_loop"));
      const auto d = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::SliceInputDescription>(loop->get_input_descriptions()[6]);
      d->m_stride+=1;
    });
    std::cout << "device-free synthetic geometry: PASS\n";
    if (argc==2) {
      const std::filesystem::path directory(argv[1]);
      ov::Core core;  // read_model only; no device discovery or compilation.
      for (int block=0;block<32;++block) {
        const bool odd = block%2!=0;
        const auto name = "block"+std::to_string(block)+(odd?"_odd":"_even")+"_fused_loop_tail_fp16.xml";
        for (const auto hw : kShapes) {
          auto model = core.read_model((directory/name).string());
          verify(*model,makeSeedVrWindowMetadata(1,hw[0],hw[1],odd));
        }
        std::cout << "block " << block << " six geometry mappings: PASS\n";
      }
    } else check(argc==1,"optional template directory required");
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
