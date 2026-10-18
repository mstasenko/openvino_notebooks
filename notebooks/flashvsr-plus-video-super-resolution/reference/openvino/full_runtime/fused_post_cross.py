#!/usr/bin/env python3
"""Replace the fixed-HD post cross-attention chunk body with global SDPA.

The caller authenticates the source IR bytes.  This narrow matcher checks the
F16 output, shared F32 K/V, shared normalized query, exact chunk
coverage, and learned cross-O consumer before replacing only the Concat.
"""

from __future__ import annotations

import math

TOKENS, CONTEXT, DIM = 23040, 512, 1536
HEADS, HEAD_DIM, CHUNK = 12, 128, 128
CHUNKS = TOKENS // CHUNK
SCALE = float(1.0 / math.sqrt(HEAD_DIM))
SIX_GIB = 6 * 1024**3
Q_NORM = "weight__blocks_0_cross_attn_norm_q_weight"
O_WEIGHT = "weight__blocks_0_cross_attn_o_weight"
O_BIAS = "weight__blocks_0_cross_attn_o_bias"


def memory_inventory(query_tokens: int = TOKENS) -> dict:
    """Conservative graph-visible full-N inventory; not UHD admission."""
    if isinstance(query_tokens, bool) or not isinstance(query_tokens, int) or query_tokens < 1:
        raise ValueError(f"query_tokens must be a positive int, got {query_tokens!r}")
    q, kv, score = HEADS * query_tokens * HEAD_DIM, HEADS * CONTEXT * HEAD_DIM, HEADS * query_tokens * CONTEXT
    f32 = {"query_f32_bytes": q * 4, "key_f32_bytes": kv * 4, "value_f32_bytes": kv * 4,
           "scores_f32_bytes": score * 4, "probabilities_f32_bytes": score * 4, "attention_output_f32_bytes": q * 4}
    state = {"query_source_f16_bytes": query_tokens * DIM * 2, "cross_k_source_f16_bytes": CONTEXT * DIM * 2,
             "cross_v_source_f16_bytes": CONTEXT * DIM * 2, "context_output_f16_bytes": query_tokens * DIM * 2}
    base = sum(f32.values()) + sum(state.values())
    return {"query_tokens": query_tokens, "context_tokens": CONTEXT, "heads": HEADS, "head_dim": HEAD_DIM,
            **f32, **state, "base_bytes": base, "scratch_allowance_bytes": base,
            "conservative_peak_bytes": 2 * base, "limit_bytes": SIX_GIB, "within_6_gib": 2 * base <= SIX_GIB,
            "admission_basis": "full-N structural inventory only; not UHD admission"}


def rewrite_post(model, source_half=False, bounded_rows=None) -> dict:
    """Authenticate the fixed-HD f6/f2 chunk body, then replace only its Concat."""
    import openvino as ov
    from openvino import opset13 as ops

    def bad(message):
        raise ValueError(f"fused post cross rewrite rejected: {message}")

    token_shape = model.input("x").get_partial_shape()
    if not token_shape.is_static:
        bad("the fixed-HD input shape must be static")
    token_shape = tuple(token_shape.to_shape())
    uhd=token_shape in ((1,195840,DIM),(1,65280,DIM))
    if uhd and (not source_half or type(bounded_rows)!=int or bounded_rows not in (1280,3840)):
        bad("UHD metadata requires native half attention and bounded query rows")
    if bounded_rows is not None and not uhd:
        bad("Bounded destination is restricted to reviewed UHD metadata")
    if token_shape not in ((1,23040,DIM),(1,7680,DIM)) and not uhd:
        bad(f"unsupported fixed-HD input shape {token_shape}")
    # These two reviewed geometries both contain complete 128-token chunks.
    TOKENS, CHUNKS = token_shape[1], token_shape[1] // CHUNK

    def src(node, index=0):
        return node.input(index).get_source_output().get_node()

    def same(a, b):
        return a.get_instance_id() == b.get_instance_id()

    def shape(out, expected, label):
        p = out.get_partial_shape()
        if not p.is_static or tuple(int(x) for x in p.to_shape()) != tuple(expected):
            bad(f"{label} shape is {p}, expected {tuple(expected)!r}")

    def node(value, kind, label):
        if value.get_type_name() != kind:
            bad(f"{label} is {value.get_type_name()}, expected {kind}")

    def attrs(value, expected, label):
        for key, wanted in expected.items():
            actual = value.get_attributes().get(key)
            if str(actual).lower() != str(wanted).lower():
                bad(f"{label} attribute {key} is {actual!r}, expected {wanted!r}")

    def vector(value, expected=None, label="constant"):
        node(value, "Constant", label)
        if value.output(0).get_element_type() != ov.Type.i64:
            bad(f"{label} is not i64")
        got = tuple(int(x) for x in value.get_vector())
        if expected is not None and got != tuple(expected):
            bad(f"{label} is {got!r}, expected {tuple(expected)!r}")
        return got

    def scalar(value, expected, label):
        node(value, "Constant", label)
        if value.output(0).get_element_type() != ov.Type.f32:
            bad(f"{label} is not f32")
        got = list(value.get_vector())
        if len(got) != 1 or not math.isclose(float(got[0]), expected, rel_tol=0.0, abs_tol=1e-7):
            bad(f"{label} is {got!r}, expected {expected!r}")

    def one(out, label, index=None):
        targets = list(out.get_target_inputs())
        if len(targets) != 1 or (index is not None and targets[0].get_index() != index):
            bad(f"{label} does not have the expected single consumer")
        return targets[0].get_node()

    def names(out):
        try:
            return tuple(sorted(str(x) for x in out.get_names()))
        except RuntimeError:
            return ()

    params = {}
    for parameter in model.get_parameters():
        ns = names(parameter.output(0))
        if len(ns) != 1 or ns[0] in params:
            bad(f"model Parameter names are not unique: {ns!r}")
        params[ns[0]] = parameter

    def parameter(name, expected_shape):
        value = params.get(name)
        if value is None or value.output(0).get_element_type() != ov.Type.f16:
            bad(f"missing/non-F16 Parameter {name!r}")
        shape(value.output(0), expected_shape, f"Parameter {name}")
        return value

    def cache(name, label):
        value = parameter(name, (1, CONTEXT, DIM))
        reshape = one(value.output(0), f"{label} Parameter", 0)
        node(reshape, "Reshape", f"{label} reshape"); shape(reshape.output(0), (1, CONTEXT, HEADS, HEAD_DIM), label)
        attrs(reshape, {"special_zero": False}, label); vector(src(reshape, 1), (1, CONTEXT, HEADS, HEAD_DIM), f"{label} shape")
        transpose = one(reshape.output(0), label, 0); node(transpose, "Transpose", f"{label} transpose")
        vector(src(transpose, 1), (0, 2, 1, 3), f"{label} axes"); shape(transpose.output(0), (1, HEADS, CONTEXT, HEAD_DIM), label)
        converted = one(transpose.output(0), label, 0); node(converted, "Convert", f"{label} conversion")
        if converted.output(0).get_element_type() != ov.Type.f32: bad(f"{label} conversion is not F32")
        shape(converted.output(0), (1, HEADS, CONTEXT, HEAD_DIM), label)
        if len(converted.output(0).get_target_inputs()) != CHUNKS: bad(f"{label} is not shared by {CHUNKS} chunks")
        return value, converted

    def chunk(final, key, value, index):
        label = f"chunk {index}"; node(final, "Convert", f"{label} output"); shape(final.output(0), (1, CHUNK, DIM), label)
        reshape = src(final); node(reshape, "Reshape", label); shape(reshape.output(0), (1, CHUNK, DIM), label)
        transpose = src(reshape); node(transpose, "Transpose", label); shape(transpose.output(0), (1, CHUNK, HEADS, HEAD_DIM), label)
        pv = src(transpose); node(pv, "MatMul", label); attrs(pv, {"transpose_a": False, "transpose_b": False}, label)
        shape(pv.output(0), (1, HEADS, CHUNK, HEAD_DIM), label)
        if not same(src(pv, 1), value): bad(f"{label} PV does not use shared V")
        softmax = src(pv); node(softmax, "Softmax", label); attrs(softmax, {"axis": 3}, label)
        shape(softmax.output(0), (1, HEADS, CHUNK, CONTEXT), label)
        scaled = src(softmax); node(scaled, "Multiply", label); scalar(src(scaled, 1), SCALE, f"{label} scale")
        qk = src(scaled); node(qk, "MatMul", label); attrs(qk, {"transpose_a": False, "transpose_b": True}, label)
        shape(qk.output(0), (1, HEADS, CHUNK, CONTEXT), label)
        if not same(src(qk, 1), key): bad(f"{label} QK does not use shared K")
        query = src(qk); node(query, "Convert", label); shape(query.output(0), (1, HEADS, CHUNK, HEAD_DIM), label)
        qt = src(query); node(qt, "Transpose", label); vector(src(qt, 1), (0, 2, 1, 3), f"{label} axes")
        qr = src(qt); node(qr, "Reshape", label); shape(qr.output(0), (1, CHUNK, HEADS, HEAD_DIM), label)
        qs = src(qr); node(qs, "StridedSlice", label); shape(qs.output(0), (1, CHUNK, DIM), label)
        root = src(qs); vector(src(qs, 3), (1, 1, 1), f"{label} strides")
        start, end = vector(src(qs, 1), label=f"{label} start"), vector(src(qs, 2), label=f"{label} end")
        expected = (0, index * CHUNK, 0), (1, (index + 1) * CHUNK, DIM)
        if (start, end) != expected: bad(f"{label} coverage is {(start, end)!r}, expected {expected!r}")
        return root, start[1]

    candidates = [n for n in model.get_ordered_ops() if n.get_type_name() == "Concat" and n.get_attributes().get("axis") == 1 and len(n.inputs()) == CHUNKS and n.output(0).get_partial_shape().is_static and tuple(int(x) for x in n.output(0).get_partial_shape().to_shape()) == (1, TOKENS, DIM)]
    if len(candidates) != 1: bad(f"expected one authenticated {CHUNKS}-chunk Concat, got {len(candidates)}")
    concat = candidates[0]
    if concat.output(0).get_element_type() != ov.Type.f16: bad("cross Concat is not F16")
    for input_value in concat.inputs():
        shape(input_value.get_source_output(), (1, CHUNK, DIM), "cross chunk")
        if input_value.get_source_output().get_element_type() != ov.Type.f16: bad("cross chunk is not F16")
    cross_o = one(concat.output(0), "cross Concat", 0); node(cross_o, "MatMul", "learned cross-O")
    attrs(cross_o, {"transpose_a": False, "transpose_b": True}, "learned cross-O"); shape(cross_o.output(0), (1, TOKENS, DIM), "learned cross-O")
    if not same(src(cross_o), concat): bad("cross-O does not consume authenticated Concat")
    o_weight = parameter(O_WEIGHT, (DIM, DIM)); parameter(O_BIAS, (DIM,))
    if not same(src(cross_o, 1), o_weight): bad("cross-O weight is not exact learned Parameter")
    key_parameter, key = cache("cross_k", "K"); value_parameter, value = cache("cross_v", "V")
    roots, coverage = [], []
    for index, input_value in enumerate(concat.inputs()):
        root, start = chunk(input_value.get_source_output().get_node(), key, value, index)
        roots.append(root); coverage.append(start)
    if len({r.get_instance_id() for r in roots}) != 1: bad("chunks do not share one query root")
    root = roots[0]; node(root, "Multiply", "shared query root"); shape(root.output(0), (1, TOKENS, DIM), "shared query root")
    if root.output(0).get_element_type() != ov.Type.f16 or len(root.output(0).get_target_inputs()) != CHUNKS: bad("query root is not shared F16")
    attrs(root, {"auto_broadcast": "numpy"}, "shared query root")
    norm = next((src(root, i) for i in range(2) if src(root, i).get_type_name() == "Convert"), None)
    scale = next((src(root, i) for i in range(2) if src(root, i).get_type_name() == "Reshape"), None)
    if norm is None or scale is None or norm.output(0).get_element_type() != ov.Type.f16 or src(norm).get_type_name() != "Divide": bad("query root is not normed F16 query")
    if src(src(norm)).output(0).get_element_type() != ov.Type.f32: bad("query RMSNorm boundary is not F32")
    shape(scale.output(0), (1, 1, DIM), "query scale"); vector(src(scale, 1), (1, 1, -1), "query scale shape")
    if not same(src(scale), parameter(Q_NORM, (DIM,))): bad("query scale is not learned norm Parameter")
    if coverage != [i * CHUNK for i in range(CHUNKS)]: bad("chunk coverage is not contiguous full-N coverage")
    ports = (tuple(names(p.output(0)) for p in model.get_parameters()), tuple(names(r.output(0)) for r in model.get_results()))
    axes = ops.constant([0, 2, 1, 3], ov.Type.i64)
    q = ops.transpose(ops.reshape(root, ops.constant([1, TOKENS, HEADS, HEAD_DIM], ov.Type.i64), False), axes)
    if source_half:
        # Native FlashAttention takes half Q/K/V and an implicit float scale.
        # Keep the authenticated original half K/V, not the F32 CPU fallback.
        attended = ops.scaled_dot_product_attention(q,key.input_value(0),value.input_value(0),
                    causal=False,name="fused_post_cross_global_sdpa")
    else:
        q = ops.convert(q,ov.Type.f32)
        attended = ops.scaled_dot_product_attention(q, key.output(0), value.output(0), None, scale=ops.constant(SCALE, ov.Type.f32), causal=False, name="fused_post_cross_global_sdpa")
    fused = ops.convert(ops.reshape(ops.transpose(attended, axes), ops.constant([1, TOKENS, DIM], ov.Type.i64), False), ov.Type.f16, name="fused_post_cross_context_f16")
    if names(concat.output(0)): fused.output(0).get_tensor().set_names(set(names(concat.output(0))))
    concat.output(0).replace(fused.output(0)); model.validate_nodes_and_infer_types()
    if ports != (tuple(names(p.output(0)) for p in model.get_parameters()), tuple(names(r.output(0)) for r in model.get_results())): bad("model ports changed")
    if not any(n.get_instance_id() == cross_o.get_instance_id() for n in model.get_ordered_ops()): bad("learned cross-O was removed")
    inventory = memory_inventory(bounded_rows or TOKENS)
    if not inventory["within_6_gib"]: bad("full-N inventory exceeds 6 GiB")
    return {"variant": "full_n_source_f16_sdpa" if source_half else "full_n_global_f32_sdpa", "query_tokens": TOKENS, "context_tokens": CONTEXT, "heads": HEADS, "head_dim": HEAD_DIM,
            "bounded_destination_rows":bounded_rows,
            "query_layout": [1, HEADS, TOKENS, HEAD_DIM], "key_layout": [1, HEADS, CONTEXT, HEAD_DIM], "value_layout": [1, HEADS, CONTEXT, HEAD_DIM],
            ("attention_output_f16_layout" if source_half else "attention_output_f32_layout"): [1, HEADS, TOKENS, HEAD_DIM], "context_output_f16_shape": [1, TOKENS, DIM], "mask": "none", "causal": False, "scale": SCALE,
            "chunk_count_authenticated": CHUNKS, "chunk_tokens_authenticated": CHUNK, "source_concat": concat.get_friendly_name(), "cross_o_matmul": cross_o.get_friendly_name(),
            "cross_o_weight_parameter": o_weight.get_friendly_name(), "key_parameter": key_parameter.get_friendly_name(), "value_parameter": value_parameter.get_friendly_name(),
            "replacement_context": fused.get_friendly_name(), "ports_preserved": True, "chunk_coverage": {"first_start": coverage[0], "last_start": coverage[-1], "covered_tokens": TOKENS},
            "memory_inventory": inventory}


__all__ = ["memory_inventory", "rewrite_post"]
