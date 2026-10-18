#include "seedvr_geometry.hpp"
#include "ai_enhance_core.hpp"

#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/loop.hpp>
#include <openvino/op/util/multi_subgraph_base.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace otc::ai {
namespace {
using Node = std::shared_ptr<ov::Node>;
using Constant = ov::op::v0::Constant;
using Subgraph = ov::op::util::MultiSubGraphOp;
constexpr std::array<const char*, 3> kMetadata = {
    "window_idx", "window_key_mask", "window_rope_freqs"};

void require(bool value, const char* message) {
  if (!value) throw std::invalid_argument(std::string("SeedVR2 geometry: ") + message);
}

std::size_t product(std::size_t a, std::size_t b) {
  require(a == 0 || b <= std::numeric_limits<std::size_t>::max() / a, "size overflow");
  return a * b;
}

Node named(ov::Model& graph, const std::string& name, const char* type) {
  Node found;
  for (const auto& node : graph.get_ordered_ops()) {
    if (node->get_friendly_name() != name) continue;
    require(!found && node->get_type_name() == std::string(type), "ambiguous or wrong semantic node");
    found = node;
  }
  require(static_cast<bool>(found), "missing semantic node");
  return found;
}

void shape(const ov::Output<ov::Node>& value, const ov::Shape& expected,
           ov::element::Type type) {
  require(value.get_element_type() == type && value.get_partial_shape().is_static() &&
              value.get_shape() == expected, "tensor contract differs");
}

std::shared_ptr<Constant> integer(const Node& target, std::size_t port,
                                 const std::vector<std::int64_t>& expected,
                                 const ov::Shape& dims) {
  require(port < target->get_input_size(), "missing semantic input");
  auto value = std::dynamic_pointer_cast<Constant>(target->input_value(port).get_node_shared_ptr());
  require(value && value->get_element_type() == ov::element::i64 &&
              value->get_shape() == dims && value->cast_vector<std::int64_t>() == expected,
          "semantic Constant differs");
  return value;
}

struct Edit {
  Node target;
  std::size_t port;
  std::shared_ptr<Constant> original;
  ov::Shape shape;
  std::vector<std::int64_t> values;
};

void add(std::vector<Edit>& edits, const Node& target, std::size_t port,
         std::vector<std::int64_t> before, ov::Shape beforeShape,
         std::vector<std::int64_t> after, ov::Shape afterShape) {
  auto original = integer(target, port, before, beforeShape);
  edits.push_back({target, port, std::move(original), std::move(afterShape), std::move(after)});
}

std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>> publicParameters(ov::Model& model) {
  const std::set<std::string> required = {
      "vid", "txt", "emb", "window_idx", "window_key_mask", "window_rope_freqs"};
  std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>> result;
  for (const auto& parameter : model.get_parameters()) {
    std::size_t matches = 0;
    for (const auto& name : parameter->output(0).get_names()) {
      if (!required.count(name)) continue;
      require(result.emplace(name, parameter).second, "duplicate public tensor name");
      ++matches;
    }
    require(matches <= 1, "aliased public tensor names");
  }
  require(result.size() == required.size(), "missing public Parameter");
  return result;
}

void reverseContract(const std::vector<std::int64_t>& reverse, std::size_t tokens,
                     std::size_t windows, std::size_t length) {
  require(reverse.size() == tokens, "reverse map length differs");
  const auto positions = product(windows, length);
  std::set<std::int64_t> unique;
  for (auto index : reverse)
    require(index >= 0 && static_cast<std::uint64_t>(index) < positions &&
                unique.insert(index).second, "reverse map contains padding, duplicates or invalid positions");
}

void metadataContract(const SeedVrWindowMetadata& m) {
  require((m.frames == 1 || m.frames == 2) && m.height > 0 && m.width > 0, "only positive T1/T2 geometry is supported");
  require(m.sourceTokenCount == product(static_cast<std::size_t>(m.frames), product(static_cast<std::size_t>(m.height), static_cast<std::size_t>(m.width))) &&
              m.sourceTokenCount <= 64800 && m.windowCount > 0 && m.maxVideoLen > 0 &&
              m.maxVideoLen <= 966 && m.attentionLen == m.maxVideoLen + 58 &&
              product(product(m.windowCount, m.attentionLen), 126) <= 64U * 1024U * 1024U,
          "metadata dimensions exceed the bounded contract");
  require(m.windowLengths.size() == m.windowCount, "missing window lengths");
  std::size_t sum = 0;
  for (auto length : m.windowLengths) {
    require(length > 0 && length <= m.maxVideoLen && sum <= m.sourceTokenCount &&
                length <= m.sourceTokenCount - sum, "invalid window coverage");
    sum += length;
  }
  require(sum == m.sourceTokenCount && *std::max_element(m.windowLengths.begin(), m.windowLengths.end()) == m.maxVideoLen,
          "window coverage differs");
  reverseContract(m.windowReverseIdx, m.sourceTokenCount, m.windowCount, m.maxVideoLen);
  for (auto index : m.windowReverseIdx)
    require(static_cast<std::size_t>(index) % m.maxVideoLen <
                m.windowLengths[static_cast<std::size_t>(index) / m.maxVideoLen], "reverse map selects padding");
}

void edge(const Node& target, std::size_t port, const ov::Output<ov::Node>& source) {
  require(port < target->get_input_size() && target->input_value(port) == source, "semantic edge mapping differs");
}

void zeroState(const std::shared_ptr<ov::op::v5::Loop>& loop, std::size_t port,
               const ov::Shape& dims) {
  auto value = loop->input_value(port).get_node_shared_ptr();
  shape(value->output(0), dims, ov::element::f32);
  const auto initializer = value;
  if (value->get_type_name() == std::string("Convert")) {
    const auto targets = value->output(0).get_target_inputs();
    require(targets.size() == 1 && targets.begin()->get_node() == loop.get() &&
                targets.begin()->get_index() == port, "shared Loop initializer");
    value = value->input_value(0).get_node_shared_ptr();
  }
  const auto constant = std::dynamic_pointer_cast<Constant>(value);
  require(constant && constant->get_shape() == dims &&
              (constant->get_element_type() == ov::element::f16 || constant->get_element_type() == ov::element::f32),
          "unsupported Loop initializer");
  const auto targets = constant->output(0).get_target_inputs();
  require(targets.size() == 1 && targets.begin()->get_node() ==
              (initializer == constant ? loop.get() : initializer.get()) &&
              targets.begin()->get_index() == (initializer == constant ? port : 0),
          "shared zero-state Constant");
  for (auto v : constant->cast_vector<float>()) require(v == 0, "Loop initializer is not zero");
}
}  // namespace

void patchSeedVrGeometry(ov::Model& model, const SeedVrWindowMetadata& m) {
  metadataContract(m);
  const auto params = publicParameters(model);
  const auto originalParameters = model.get_parameters();
  std::map<Node, ov::PartialShape> extraShapes;
  for (const auto& p : originalParameters) {
    if (std::none_of(params.begin(), params.end(), [&](const auto& entry) { return entry.second == p; }))
      extraShapes.emplace(p, p->get_partial_shape());
  }
  const auto idx = params.at("window_idx")->output(0);
  require(idx.get_partial_shape().is_static() && idx.get_shape().size() == 2, "nonstatic original index shape");
  const auto oldN = idx.get_shape()[0], oldM = idx.get_shape()[1];
  const auto vid = params.at("vid")->output(0);
  require(vid.get_partial_shape().is_static() && vid.get_shape().size() == 2, "nonstatic original video shape");
  const auto oldS = vid.get_shape()[0];
  require(oldN > 1 && oldM > 0 && oldS > 0 && oldS <= 64800 && oldM <= 966 &&
              product(product(oldN,oldM+58),126) <= 64U*1024U*1024U,
          "unsupported original geometry");
  shape(vid, {oldS,2560}, ov::element::f32);
  shape(params.at("txt")->output(0), {58,2560}, ov::element::f32);
  shape(params.at("emb")->output(0), {1,15360}, ov::element::f32);
  shape(idx, {oldN,oldM}, ov::element::i64);
  shape(params.at("window_key_mask")->output(0), {oldN,oldM+58}, ov::element::boolean);
  shape(params.at("window_rope_freqs")->output(0), {oldN,oldM+58,126}, ov::element::f32);
  require(model.outputs().size() == 2, "unexpected output count");
  shape(model.output("vid_out"), {oldS,2560}, ov::element::f32);
  shape(model.output("txt_out"), {58,2560}, ov::element::f32);

  auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(named(model,"full_chunk_attention_loop","Loop"));
  for (const auto& n : model.get_ordered_ops())
    if (std::dynamic_pointer_cast<Subgraph>(n)) require(n == loop, "unsupported control flow");
  auto body = loop->get_function();
  for (const auto& n : body->get_ordered_ops()) require(!std::dynamic_pointer_cast<Subgraph>(n), "nested control flow");
  const auto ports = loop->get_special_body_ports();
  require(ports.current_iteration_input_idx == -1 && ports.body_condition_output_idx == 0 &&
              loop->get_input_size() == 13 && loop->get_output_size() == 3 &&
              body->get_parameters().size() == 11 && body->get_results().size() == 4,
          "unsupported Loop roles");
  const auto condition = std::dynamic_pointer_cast<Constant>(loop->input_value(1).get_node_shared_ptr());
  const auto bodyCondition = std::dynamic_pointer_cast<Constant>(body->get_results()[0]->input_value(0).get_node_shared_ptr());
  require(condition && bodyCondition && condition->get_element_type() == ov::element::boolean &&
              bodyCondition->get_element_type() == ov::element::boolean &&
              condition->cast_vector<char>() == std::vector<char>{1} && bodyCondition->cast_vector<char>() == std::vector<char>{1},
          "unsupported Loop conditions");
  const auto descriptions = loop->get_input_descriptions();
  require(descriptions.size() == 11, "Loop descriptor count differs");
  std::size_t oldChunk = 0;
  for (std::size_t i = 0; i < descriptions.size(); ++i) {
    const auto& d = descriptions[i];
    require(d->m_input_index == i+2 && d->m_body_parameter_index == i, "Loop input roles differ");
    if (i < 6) {
      require(static_cast<bool>(std::dynamic_pointer_cast<Subgraph::InvariantInputDescription>(d)), "QKV role differs");
      shape(body->get_parameters()[i]->output(0), {i<3?oldS:58,20,128}, ov::element::f32);
    } else if (i < 9) {
      auto slice = std::dynamic_pointer_cast<Subgraph::SliceInputDescription>(d);
      require(slice && slice->m_axis == 0 && slice->m_start == 0 && slice->m_end == -1 &&
                  slice->m_stride > 0 && slice->m_stride == slice->m_part_size, "metadata slice role differs");
      if (i == 6) oldChunk = static_cast<std::size_t>(slice->m_stride);
      require(oldChunk == static_cast<std::size_t>(slice->m_stride), "mixed chunk sizes");
      const ov::Shape dims = i==6?ov::Shape{oldChunk,oldM}:i==7?ov::Shape{oldChunk,oldM+58}:ov::Shape{oldChunk,oldM+58,126};
      shape(body->get_parameters()[i]->output(0), dims, i==6?ov::element::i64:i==7?ov::element::boolean:ov::element::f32);
    } else {
      auto merged = std::dynamic_pointer_cast<Subgraph::MergedInputDescription>(d);
      require(merged && merged->m_body_value_index == i-7, "accumulator role differs");
      shape(body->get_parameters()[i]->output(0),i==9?ov::Shape{58,2560}:ov::Shape{1},ov::element::f32);
    }
  }
  require(oldChunk < oldN, "original Loop has no tail");
  const auto oldFull = oldN / oldChunk * oldChunk;
  require(oldFull < oldN, "original Loop must have a nonempty tail");
  const auto chunk = m.windowCount > 1 ? std::min(oldChunk,m.windowCount-1) : 0;
  const auto full = chunk ? (m.windowCount-1)/chunk*chunk : 0;
  const auto outputs = loop->get_output_descriptions();
  require(outputs.size() == 3, "Loop output roles differ");
  auto scan = std::dynamic_pointer_cast<Subgraph::ConcatOutputDescription>(outputs[0]);
  require(scan && scan->m_output_index == 0 && scan->m_body_value_index == 1 && scan->m_start == 0 &&
              scan->m_stride == 1 && scan->m_part_size == 1 && scan->m_end == -1 && scan->m_axis == 0,
          "Loop scan role differs");
  for (std::size_t i=1;i<3;++i) {
    auto value = std::dynamic_pointer_cast<Subgraph::BodyOutputDescription>(outputs[i]);
    require(value && value->m_output_index == i && value->m_body_value_index == i+1 && value->m_iteration == -1,
            "Loop accumulator output differs");
  }
  zeroState(loop,11,{58,2560}); zeroState(loop,12,{1});
  std::vector<Edit> edits;
  if (chunk) add(edits,loop,0,{static_cast<std::int64_t>(oldFull/oldChunk)},{},
                 {static_cast<std::int64_t>(full/chunk)},{});
  else integer(loop,0,{static_cast<std::int64_t>(oldFull/oldChunk)},{});
  for (std::size_t i=0;i<3;++i) {
    auto first = named(model,std::string("slice_full_loop_")+kMetadata[i],"Slice");
    auto tail = named(model,std::string("slice_exact_tail_")+kMetadata[i],"Slice");
    for (auto slice : {first,tail}) {
      require(slice->get_input_size()==5,"Slice arity differs");
      edge(slice,0,params.at(kMetadata[i])->output(0));
      integer(slice,3,{1},{1}); integer(slice,4,{0},{1});
    }
    integer(first,1,{0},{1});
    if (chunk) add(edits,first,2,{static_cast<std::int64_t>(oldFull)},{1},{static_cast<std::int64_t>(full)},{1});
    else integer(first,2,{static_cast<std::int64_t>(oldFull)},{1});
    add(edits,tail,1,{static_cast<std::int64_t>(oldFull)},{1},{static_cast<std::int64_t>(full)},{1});
    add(edits,tail,2,{static_cast<std::int64_t>(oldN)},{1},{static_cast<std::int64_t>(m.windowCount)},{1});
    edge(loop,8+i,first->output(0));
  }
  auto fullVideo = named(model,"reshape_full_loop_video_windows","Reshape");
  auto pack = named(*body,"pack_full_loop_video_scan_slice","Reshape");
  if (chunk) {
    add(edits,fullVideo,1,{static_cast<std::int64_t>(oldFull),static_cast<std::int64_t>(oldM),2560},{3},
        {static_cast<std::int64_t>(full),static_cast<std::int64_t>(m.maxVideoLen),2560},{3});
    add(edits,pack,1,{1,static_cast<std::int64_t>(oldChunk*oldM),2560},{3},
        {1,static_cast<std::int64_t>(chunk*m.maxVideoLen),2560},{3});
  } else {
    integer(fullVideo,1,{static_cast<std::int64_t>(oldFull),static_cast<std::int64_t>(oldM),2560},{3});
    integer(pack,1,{1,static_cast<std::int64_t>(oldChunk*oldM),2560},{3});
  }
  edge(fullVideo,0,loop->output(0));
  auto flat = named(model,"flatten_loop_plus_tail_video_windows","Reshape");
  auto gather = named(model,"exact_source_order_video_gather","Gather");
  add(edits,flat,1,{static_cast<std::int64_t>(product(oldN,oldM)),2560},{2},
      {static_cast<std::int64_t>(product(m.windowCount,m.maxVideoLen)),2560},{2});
  auto reverse = std::dynamic_pointer_cast<Constant>(gather->input_value(1).get_node_shared_ptr());
  require(reverse && reverse->get_element_type()==ov::element::i64 && reverse->get_shape()==ov::Shape{oldS},"reverse Constant differs");
  reverseContract(reverse->cast_vector<std::int64_t>(),oldS,oldN,oldM);
  edits.push_back({gather,1,reverse,{m.sourceTokenCount},m.windowReverseIdx});
  edge(gather,0,flat->output(0)); integer(gather,2,{0},{});
  const std::array<Node,3> merges = {named(model,"loop_plus_tail_video_windows","Concat"),
      named(model,"add_tail_text_sum","Add"),named(model,"add_tail_text_count","Add")};
  auto average = named(model,"averaged_text_output","Divide");
  const std::array<Node,3> consumers = {flat,average,average};
  const std::array<std::size_t,3> portsToReplace = {0,0,1};
  const std::array<ov::Shape,3> tailShapes = {ov::Shape{oldN-oldFull,oldM,2560},ov::Shape{58,2560},ov::Shape{1}};
  for (std::size_t i=0;i<3;++i) {
    require(merges[i]->get_input_size()==2,"merge arity differs");
    edge(merges[i],0,i==0?fullVideo->output(0):loop->output(i));
    shape(merges[i]->input_value(1),tailShapes[i],ov::element::f32);
    edge(consumers[i],portsToReplace[i],merges[i]->output(0));
    require(merges[i]->output(0).get_target_inputs().size()==1,"shared merge output");
  }

  // Keep a matching verified template literally unchanged: replacing equal
  // Constants can still change compiler ordering. Never bypass the guards above.
  bool unchanged = chunk == oldChunk && full == oldFull &&
      vid.get_shape() == ov::Shape{m.sourceTokenCount,2560} &&
      idx.get_shape() == ov::Shape{m.windowCount,m.maxVideoLen} &&
      params.at("window_key_mask")->get_shape() == ov::Shape{m.windowCount,m.attentionLen} &&
      params.at("window_rope_freqs")->get_shape() == ov::Shape{m.windowCount,m.attentionLen,126};
  for (const auto& edit : edits) {
    edge(edit.target,edit.port,edit.original->output(0));
    unchanged = unchanged && edit.original->get_shape() == edit.shape &&
        edit.original->cast_vector<std::int64_t>() == edit.values;
  }
  if (unchanged) return;

  // All structural guards precede edits; Constants are replaced, never modified
  // in place, so stored learned weights and any extra weight Parameters survive.
  for (const auto& edit : edits) {
    edge(edit.target,edit.port,edit.original->output(0));
    edit.target->input(edit.port).replace_source_output(Constant::create(ov::element::i64,edit.shape,edit.values));
  }
  if (chunk) {
    for (std::size_t i=6;i<9;++i) {
      auto d = std::dynamic_pointer_cast<Subgraph::SliceInputDescription>(loop->get_input_descriptions()[i]);
      d = std::make_shared<Subgraph::SliceInputDescription>(*d);
      d->m_stride = d->m_part_size = static_cast<std::int64_t>(chunk);
      loop->get_input_descriptions()[i] = d;
    }
    for (std::size_t i=0;i<3;++i) body->get_parameters()[i]->set_partial_shape(ov::Shape{m.sourceTokenCount,20,128});
    body->get_parameters()[6]->set_partial_shape(ov::Shape{chunk,m.maxVideoLen});
    body->get_parameters()[7]->set_partial_shape(ov::Shape{chunk,m.attentionLen});
    body->get_parameters()[8]->set_partial_shape(ov::Shape{chunk,m.attentionLen,126});
  } else {
    for (std::size_t i=0;i<3;++i) {
      edge(consumers[i],portsToReplace[i],merges[i]->output(0));
      consumers[i]->input(portsToReplace[i]).replace_source_output(merges[i]->input_value(1));
    }
  }
  params.at("vid")->set_partial_shape(ov::Shape{m.sourceTokenCount,2560});
  params.at("window_idx")->set_partial_shape(ov::Shape{m.windowCount,m.maxVideoLen});
  params.at("window_key_mask")->set_partial_shape(ov::Shape{m.windowCount,m.attentionLen});
  params.at("window_rope_freqs")->set_partial_shape(ov::Shape{m.windowCount,m.attentionLen,126});
  // SmartReshape can validate a nested body against stale outer shapes midway.
  model.validate_nodes_and_infer_types();
  shape(model.output("vid_out"),{m.sourceTokenCount,2560},ov::element::f32);
  shape(model.output("txt_out"),{58,2560},ov::element::f32);
  require(model.get_parameters()==originalParameters,"Parameter ownership changed");
  for (const auto& extra : extraShapes) require(extra.first->get_output_partial_shape(0)==extra.second,"extra weight Parameter changed");
  if (!chunk) for (const auto& n : model.get_ordered_ops()) require(!std::dynamic_pointer_cast<Subgraph>(n),"control flow remains reachable");
}
}  // namespace otc::ai
