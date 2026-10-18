"""Finite same-input check of reused Plus pre/post graphs and real weights.

This checks individual components with synthetic states, not model quality or
GPU speed. The pinned official PyTorch modules are the independent reference.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from frame_geometry import FrameGeometry
from plus_graphs import BlockGeometry, build_block, build_bounded_pre, build_post_rows
from prepare_real import parameter_modules
from reference_paths import DIT_CHECKPOINT, SOURCE_ROOT


SOURCE = SOURCE_ROOT
SOURCE_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
CHECKPOINT = DIT_CHECKPOINT
CHECKPOINT_SHA = "7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b"
LIMITS = {"normalized_max_abs": .015625, "relative_l2": .005}
BASE = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def typed_tensor(value, dtype):
    """Use actual storage I/O; a surrounding F32 graph can erase rounding."""
    import openvino as ov
    if dtype != ov.Type.bf16:
        return ov.Tensor(np.asarray(value, dtype=dtype.to_dtype()), shared_memory=False)
    import torch
    tensor = ov.Tensor(dtype, value.shape)
    # OpenVINO exposes BF16 data as a NumPy half view. Copy bits, NEVER values:
    # interpreting this view numerically as FP16 would corrupt every tensor.
    bits = torch.from_numpy(value).to(torch.bfloat16).view(torch.uint16).numpy()
    tensor.data.view(np.uint16)[...] = bits
    return tensor


def tensor_array(tensor):
    import openvino as ov
    if tensor.element_type == ov.Type.bf16:
        import torch
        bits = np.array(tensor.data.view(np.uint16), copy=True)
        return torch.from_numpy(bits).view(torch.bfloat16).float().numpy()
    return np.array(tensor.data, dtype=np.float32, copy=True)


def metric(expected, actual):
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite or different-shape component output")
    expected, actual = expected.astype(np.float64), actual.astype(np.float64)
    delta = actual - expected
    normalized = float(abs(delta).max() / max(abs(expected).max(), 1e-30))
    relative = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30))
    return {"normalized_max_abs": normalized, "relative_l2": relative,
            "pass": normalized <= LIMITS["normalized_max_abs"] and relative <= LIMITS["relative_l2"]}


def _source_node(model, name):
    """Return the producer of a named Result, preserving graph boundaries."""
    return model.output(name).get_node().input_value(0).get_node()


def _single_consumer(node):
    targets = list(node.output(0).get_target_inputs())
    if len(targets) != 1:
        raise ValueError(f"Diagnostic graph boundary is not single-consumer: {node.get_friendly_name()}")
    return targets[0].get_node()


def _weight_matmul(model, parameter_name):
    matches = []
    for node in model.get_ordered_ops():
        if node.get_type_name() != "MatMul":
            continue
        if any(node.input_value(index).get_node().get_friendly_name() == parameter_name
               for index in range(node.get_input_size())):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"Expected one MatMul using {parameter_name}, got {len(matches)}")
    return matches[0]


def _has_parameter(node, parameter_name):
    seen, pending = set(), [node]
    while pending:
        value = pending.pop()
        identity = value.get_friendly_name()
        if identity in seen:
            continue
        seen.add(identity)
        if value.get_type_name() == "Parameter" and value.get_friendly_name() == parameter_name:
            return True
        pending.extend(value.input_value(index).get_node() for index in range(value.get_input_size()))
    return False


def _storage_multiply(model, parameter_name, shape):
    matches = [node for node in model.get_ordered_ops()
               if node.get_type_name() == "Multiply"
               and str(node.output(0).get_partial_shape()) == str(shape)
               and any(node.input_value(index).get_node().get_type_name() == "Reshape"
                       and _has_parameter(node.input_value(index).get_node(), parameter_name)
                       for index in range(node.get_input_size()))
               and any(node.input_value(index).get_node().get_type_name() == "Convert"
                       for index in range(node.get_input_size()))]
    if len(matches) != 1:
        raise ValueError(f"Expected one storage Multiply using {parameter_name}, got {len(matches)}")
    return matches[0]


def _rope_source(model, window_name):
    """Follow one named window through partition reshapes to final RoPE output."""
    node = _source_node(model, window_name)
    for expected in ("Reshape", "Transpose", "Reshape", "Reshape"):
        if node.get_type_name() != expected:
            raise ValueError(f"{window_name} partition topology changed at {node.get_type_name()}")
        node = node.input_value(0).get_node()
    if node.get_type_name() != "Convert":
        raise ValueError(f"{window_name} RoPE output topology changed")
    return node


def _conv_output(conv, shape):
    """Follow the reviewed 1x1-convolution post projection to token-major output."""
    node = conv
    for _ in range(3):
        node = _single_consumer(node)
    if node.get_type_name() != "Transpose" or str(node.output(0).get_partial_shape()) != str(shape):
        raise ValueError("Post convolution projection topology changed")
    return node


def _diagnostic_nodes(model, role, token_shape, contract):
    """Expose first-order storage boundaries without changing graph arithmetic."""
    import openvino as ov
    from openvino import opset13 as ops

    tokens, dim = token_shape[1:]
    full_shape = f"[1,{tokens},{dim}]"
    selected = {}
    if role == "pre":
        qmat = _weight_matmul(model, contract.parameter_port("blocks.0.self_attn.q.weight"))
        input_x = qmat.input_value(0).get_node()
        if input_x.get_type_name() != "Add":
            raise ValueError("Pre modulation input topology changed")
        norm1 = input_x.input_value(0).get_node().input_value(0).get_node()
        modulation = input_x.input_value(1).get_node().input_value(0).get_node()
        selected.update(norm1=norm1, modulation=modulation, input_x=input_x)
        for lane in ("q", "k", "v"):
            prefix = f"blocks.0.self_attn.{lane}"
            matmul = _weight_matmul(model, contract.parameter_port(prefix + ".weight"))
            selected[lane + "_linear"] = _single_consumer(matmul)
            if lane != "v":
                selected[lane + "_rms"] = _storage_multiply(
                    model, contract.parameter_port("blocks.0.self_attn.norm_" + lane + ".weight"), full_shape)
                selected[lane + "_rope"] = _rope_source(model, lane + "_windows")
        selected["q_windows"] = _source_node(model, "q_windows")
        selected["k_windows"] = _source_node(model, "k_windows")
        selected["v_windows"] = _source_node(model, "v_windows")
    elif role == "post":
        convs = [node for node in model.get_ordered_ops() if node.get_type_name() == "Convolution"]
        if len(convs) != 2:
            raise ValueError(f"Expected two post projection Convolutions, got {len(convs)}")
        self_o = _conv_output(convs[0], full_shape)
        gate_product = _single_consumer(self_o)
        first = _single_consumer(gate_product)
        selected.update(self_o=self_o, first_residual=first)
        cross_q_mat = _weight_matmul(model, contract.parameter_port("blocks.0.cross_attn.q.weight"))
        selected["norm3"] = cross_q_mat.input_value(0).get_node()
        selected["cross_q_linear"] = _single_consumer(cross_q_mat)
        selected["cross_q_rms"] = _storage_multiply(
            model, contract.parameter_port("blocks.0.cross_attn.norm_q.weight"), full_shape)
        cross_o_mat = _weight_matmul(model, contract.parameter_port("blocks.0.cross_attn.o.weight"))
        selected["cross_context"] = cross_o_mat.input_value(0).get_node()
        selected["cross_o_linear"] = _single_consumer(cross_o_mat)
        selected["second_residual"] = _single_consumer(selected["cross_o_linear"])
        ffn0_mat = _weight_matmul(model, contract.parameter_port("blocks.0.ffn.0.weight"))
        modulated = ffn0_mat.input_value(0).get_node()
        norm2 = modulated.input_value(0).get_node().input_value(0).get_node()
        selected.update(norm2=norm2, modulated=modulated,
                        ffn0_linear=_single_consumer(ffn0_mat))
        selected["gelu"] = _single_consumer(selected["ffn0_linear"])
        selected["ffn2"] = _conv_output(convs[1], full_shape)
        selected["x_out"] = _source_node(model, "x_out")
    else:
        raise ValueError(f"Diagnostic role must be pre or post, got {role!r}")
    allowed_shapes = {full_shape, f"[1,6,{dim}]", f"[1,1,{dim}]",
                      f"[1,512,{dim}]", "[2,128,1536]", f"[1,{tokens},8960]"}
    for name, node in selected.items():
        if str(node.output(0).get_partial_shape()) not in allowed_shapes:
            raise ValueError(f"Unexpected diagnostic shape for {name}: {node.output(0).get_partial_shape()}")
        # These producers already have authenticated component Results. Adding
        # a second Result causes OpenVINO to reuse the new name and silently
        # remove the original output alias from compiled-model outputs.
        if name in {"q_windows", "k_windows", "v_windows", "x_out"}:
            selected[name] = node
            continue
        result = ops.result(node.output(0))
        result.output(0).get_tensor().set_names({"diagnostic_" + name})
        model.add_results([result])
    model.validate_nodes_and_infer_types()
    return {name: (name if name in {"q_windows", "k_windows", "v_windows", "x_out"}
                   else "diagnostic_" + name) for name in selected}


def _expected_boundaries(block, role, x, t_mod, context, self_context, shape, offset):
    """Compute the same named boundaries from the pinned official Torch block."""
    import torch.nn.functional as functional
    from models.blocks import modulate
    from models.attention_backends.sparse import partition_heads
    from models.embeddings import apply_rope_3d

    def array(value):
        return np.ascontiguousarray(value.float().numpy())

    shifts = (block.modulation + t_mod).chunk(6, dim=1)
    if role == "pre":
        attention = block.self_attn
        normalized = block.norm1(x)
        modulation = block.modulation + t_mod
        input_x = modulate(normalized, shifts[0], shifts[1])
        expected = {"norm1": array(normalized), "modulation": array(modulation),
                    "input_x": array(input_x)}
        for lane in ("q", "k", "v"):
            linear = getattr(attention, lane)(input_x)
            expected[lane + "_linear"] = array(linear)
            if lane != "v":
                rms = getattr(attention, "norm_" + lane)(linear)
                rotated = apply_rope_3d(attention._heads(rms), shape, temporal_offset=offset)
                flattened = rotated.transpose(1, 2).reshape(x.shape)
                expected[lane + "_rms"] = array(rms)
                expected[lane + "_rope"] = array(flattened)
                windows = partition_heads(rotated, shape, (2, 8, 8))[0]
            else:
                windows = partition_heads(attention._heads(linear), shape, (2, 8, 8))[0]
            expected[lane + "_windows"] = array(
                windows[0].permute(1, 2, 0, 3).reshape(2, 128, 1536))
            if lane != "v":
                expected[lane + "_pooled"] = array(windows[0].mean(2, dtype=__import__("torch").float32))
        return expected

    if role != "post":
        raise ValueError(f"Expected boundary role must be pre or post, got {role!r}")
    attention = block.self_attn
    cross = block.cross_attn
    shifts = (block.modulation + t_mod).chunk(6, dim=1)
    self_o = attention.o(self_context)
    first = x + shifts[2] * self_o
    norm3 = block.norm3(first)
    cross_q_linear = cross.q(norm3)
    cross_q_rms = cross.norm_q(cross_q_linear)
    q = cross_q_rms.view(1, -1, cross.num_heads, cross.head_dim).transpose(1, 2)
    k = cross.norm_k(cross.k(context)).view(1, -1, cross.num_heads, cross.head_dim).transpose(1, 2)
    v = cross.v(context).view(1, -1, cross.num_heads, cross.head_dim).transpose(1, 2)
    cross_context = functional.scaled_dot_product_attention(q, k, v)
    cross_context = cross_context.transpose(1, 2).reshape(x.shape)
    cross_o = cross.o(cross_context)
    second = first + cross_o
    norm2 = block.norm2(second)
    modulated = modulate(norm2, shifts[3], shifts[4])
    ffn0 = block.ffn[0](modulated)
    gelu = block.ffn[1](ffn0)
    ffn2 = block.ffn[2](gelu)
    return {"self_o": array(self_o), "first_residual": array(first),
            "norm3": array(norm3), "cross_q_linear": array(cross_q_linear),
            "cross_q_rms": array(cross_q_rms), "cross_context": array(cross_context),
            "cross_o_linear": array(cross_o), "second_residual": array(second),
            "norm2": array(norm2), "modulated": array(modulated),
            "ffn0_linear": array(ffn0), "gelu": array(gelu),
            "ffn2": array(ffn2), "x_out": array(second + shifts[5] * ffn2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("CPU", "B580"), default="CPU")
    parser.add_argument("--run", action="store_true",
                        help="explicitly enable B580 execution after external ownership preflight")
    parser.add_argument("--diagnose-boundaries", action="store_true",
                        help="add named graph boundaries and compare them to the official block")
    parser.add_argument("--diagnose-precision", choices=("bf16", "f16"), default="bf16")
    parser.add_argument("--accuracy", action="store_true",
                        help="honor explicit F32 reductions without a blanket low-precision hint")
    parser.add_argument("--bounded-rows", type=int,
                        help="qualify window/row scheduling with this finite slab size")
    parser.add_argument("--frame-width", type=int, choices=(128, 256), default=128)
    args = parser.parse_args()
    if args.bounded_rows is not None:
        from plus_bounded import _rows
        _rows(args.bounded_rows)
        if args.diagnose_boundaries:
            parser.error("Boundary diagnostics and bounded scheduling are separate controls")
    if args.output.exists():
        parser.error("Fresh output leaf required")
    if args.device == "B580" and not args.run:
        parser.error("B580 execution requires explicit --run after external preflight")
    if subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip() != SOURCE_COMMIT:
        raise ValueError("Official Plus revision changed")
    if subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Official reference source has local changes")
    if CHECKPOINT.stat().st_size != 5676070392 or sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Pinned Turbo DiT checkpoint changed")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "worker-source.py")
    shutil.copyfile(Path(__file__).with_name("plus_graphs.py"), args.output / "plus_graphs-source.py")
    sys.path.insert(0, str(SOURCE))
    import torch
    import openvino as ov
    from safetensors import safe_open
    from models.blocks import DiTBlock, modulate
    from models.config import AttentionConfig
    from models.embeddings import apply_rope_3d, rope_angles_3d
    from models.attention_backends.sparse import partition_heads
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    contract, _, _ = parameter_modules()
    core = ov.Core()
    device = "CPU"
    inventory = None
    if args.device == "B580":
        sys.path.insert(0, str(BASE / "openvino"))
        from attention_from_cuda import resolve
        device, inventory = resolve(core)
        if device != "GPU.1":
            raise ValueError("Exact B580 identifier changed")
    geometry = BlockGeometry(FrameGeometry(128, args.frame_width))
    shape = (4, geometry.h, geometry.w)
    report = {"status": "initializing", "scope": "Plus layer0 same-input components; synthetic states; not full-video quality",
              "source_commit": SOURCE_COMMIT, "checkpoint_sha256": CHECKPOINT_SHA,
              "runtime": ov.get_version(), "script_sha256": sha(Path(__file__)),
              "graph_helper_sha256": sha(Path(__file__).with_name("plus_graphs.py")),
              "limits": LIMITS, "requested_device": args.device, "run_requested": args.run,
              "device": device,
              "inventory": inventory, "diagnose_boundaries": args.diagnose_boundaries,
              "accuracy": args.accuracy,
              "bounded_rows": args.bounded_rows, "geometry": shape,
              "diagnose_precision": args.diagnose_precision if args.diagnose_boundaries else None,
              "cases": []}
    def array(tensor):
        return np.ascontiguousarray(tensor.float().numpy())
    def persist():
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def infer(engine, feeds):
        request = engine.create_infer_request()
        tensors = {p.get_any_name(): typed_tensor(feeds[p.get_any_name()], p.get_element_type())
                   for p in engine.inputs}
        for name, tensor in tensors.items():
            request.set_tensor(name, tensor)
        request.infer()
        return {p.get_any_name(): tensor_array(request.get_output_tensor(i))
                for i, p in enumerate(engine.outputs)}
    precision_specs = ((args.diagnose_precision,
                       torch.bfloat16 if args.diagnose_precision == "bf16" else torch.float16),) \
        if args.diagnose_boundaries else (("bf16", torch.bfloat16), ("f16", torch.float16))
    try:
        with safe_open(CHECKPOINT, framework="pt", device="cpu") as handle, torch.inference_mode():
            for precision, dtype in precision_specs:
                with torch.device("meta"):
                    block = DiTBlock(1536, 12, 8960, AttentionConfig(kernel="sdpa"), 1e-6)
                prefix = "blocks.0."
                keys = {name[len(prefix):] for name in handle.keys() if name.startswith(prefix)}
                if keys != set(block.state_dict()):
                    raise ValueError("Turbo block schema differs from official class")
                state = {}
                for name, empty in block.state_dict().items():
                    raw = handle.get_tensor(prefix + name)
                    if raw.dtype != torch.float32 or raw.shape != empty.shape or not torch.isfinite(raw).all():
                        raise ValueError("Invalid Turbo layer parameter: " + name)
                    state[name] = raw.to(dtype)
                block.load_state_dict(state, strict=True, assign=True)
                block.eval()
                rng = torch.Generator(device="cpu").manual_seed(731)
                def sample(size, scale=.2):
                    return (torch.randn(size, generator=rng, dtype=torch.float32) * scale).to(dtype)
                x = sample(geometry.token_shape)
                t_mod = sample((1, 6, 1536))
                context = sample((1, 512, 1536))
                self_context = sample(geometry.token_shape)
                shifts = (block.modulation + t_mod).chunk(6, dim=1)
                gate_names = ("gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
                gates = dict(zip(gate_names, shifts[2:]))
                inp = modulate(block.norm1(x), shifts[0], shifts[1])
                attn = block.self_attn
                q = attn._heads(attn.norm_q(attn.q(inp)))
                k = attn._heads(attn.norm_k(attn.k(inp)))
                v = attn._heads(attn.v(inp))
                cross_k = block.cross_attn.norm_k(block.cross_attn.k(context))
                cross_v = block.cross_attn.v(context)
                first = x + shifts[2] * attn.o(self_context)
                second = first + block.cross_attn(block.norm3(first), context)
                final = second + shifts[5] * block.ffn(modulate(block.norm2(second), shifts[3], shifts[4]))
                for role in ("pre", "post"):
                    if args.bounded_rows is None:
                        graph = build_block(role, geometry.frame, precision=precision)
                    elif role == "pre":
                        graph, bounded_geometry, _ = build_bounded_pre(
                            geometry.frame, rows=args.bounded_rows, precision=precision)
                        if bounded_geometry != shape:
                            raise ValueError("Bounded pre geometry differs")
                    else:
                        graph = build_post_rows(args.bounded_rows, precision=precision)
                    diagnostic_names = _diagnostic_nodes(graph, role, geometry.token_shape, contract) \
                        if args.diagnose_boundaries else None
                    start = time.perf_counter()
                    hint = ov.Type.bf16 if precision == "bf16" else ov.Type.f16
                    compile_config = {"INFERENCE_PRECISION_HINT": hint,
                                      "NUM_STREAMS": ov.properties.streams.Num(1)}
                    if device == "CPU":
                        compile_config["INFERENCE_NUM_THREADS"] = 2
                    else:
                        compile_config.update(
                            PERFORMANCE_HINT=ov.properties.hint.PerformanceMode.LATENCY,
                            CACHE_DIR="",
                        )
                    if args.accuracy:
                        compile_config.pop("INFERENCE_PRECISION_HINT")
                        compile_config["EXECUTION_MODE_HINT"] = ov.properties.hint.ExecutionMode.ACCURACY
                    engine = core.compile_model(graph, device, compile_config)
                    compile_seconds = time.perf_counter() - start
                    execution_devices = [str(value) for value in engine.get_property("EXECUTION_DEVICES")]
                    if execution_devices != [device]:
                        raise ValueError("Component test used an unintended device")
                    weights = {contract.parameter_port(key): array(state[key[len(prefix):]])
                               for key in contract.ROLE_KEYS[role]}
                    for offset in ((0, 4, 12) if role == "pre" else (0,)):
                        expected = {"x_out": array(final)}
                        feeds = {"x": array(x), **weights}
                        if role == "pre":
                            expected = {"residual": array(x), **{n: array(g) for n, g in gates.items()}}
                            for name, value in (("q", q), ("k", k), ("v", v)):
                                rotated = apply_rope_3d(value, shape, temporal_offset=offset) if name != "v" else value
                                windows = partition_heads(rotated, shape, (2, 8, 8))[0]
                                expected[name + "_windows"] = array(windows[0].permute(1, 2, 0, 3).reshape(geometry.query_windows, 128, 1536))
                                if name != "v":
                                    expected[name + "_pooled"] = array(windows[0].mean(2, dtype=torch.float32))
                            angles = rope_angles_3d(shape, 128, torch.device("cpu"), temporal_offset=offset).double()
                            feeds.update(t_mod=array(t_mod), rope_cos=array(angles.cos()).reshape(1, geometry.tokens, 1, 64),
                                         rope_sin=array(angles.sin()).reshape(1, geometry.tokens, 1, 64))
                        else:
                            feeds.update(self_context=array(self_context), cross_k=array(cross_k), cross_v=array(cross_v),
                                         **{n: array(g) for n, g in gates.items()})
                        start = time.perf_counter()
                        if args.bounded_rows is None:
                            actual = infer(engine, feeds)
                        else:
                            from plus_bounded import infer_pre, infer_post
                            callback = lambda slab: infer(engine, slab)
                            actual = infer_pre(callback, feeds, shape, args.bounded_rows) if role == "pre" \
                                else infer_post(callback, feeds, args.bounded_rows)
                        seconds = time.perf_counter() - start
                        if args.diagnose_boundaries:
                            if not expected.keys() <= actual.keys():
                                raise ValueError("Component result names differ")
                            boundary_expected = _expected_boundaries(
                                block, role, x, t_mod, context, self_context, shape, offset)
                            boundary_actual = {}
                            for name, expected_value in boundary_expected.items():
                                actual_name = diagnostic_names.get(name, name)
                                if actual_name not in actual:
                                    raise ValueError("Missing diagnostic result: " + actual_name)
                                boundary_actual[name] = actual[actual_name]
                        else:
                            if actual.keys() != expected.keys():
                                raise ValueError("Component result names differ")
                        scores = {name: metric(value, actual[name]) for name, value in expected.items()}
                        boundaries = None
                        if args.diagnose_boundaries:
                            boundaries = {name: metric(boundary_expected[name], boundary_actual[name])
                                          for name in boundary_expected}
                        raw = args.output / (f"diagnostic-{precision}-{role}-offset{offset}.npz"
                                             if args.diagnose_boundaries else f"{precision}-{role}-offset{offset}.npz")
                        raw_values = {f"expected_{n}": v for n, v in expected.items()}
                        raw_values.update({f"actual_{n}": v for n, v in actual.items()})
                        if boundaries is not None:
                            raw_values.update({f"boundary_expected_{n}": v for n, v in boundary_expected.items()})
                            raw_values.update({f"boundary_actual_{n}": v for n, v in boundary_actual.items()})
                        np.savez(raw, **raw_values)
                        passed = all(v["pass"] for v in scores.values()) and \
                            (boundaries is None or all(v["pass"] for v in boundaries.values()))
                        first_non_exact = None
                        first_over_limit = None
                        if boundaries is not None:
                            first_non_exact = next(
                                (name for name in boundary_expected
                                 if not np.array_equal(boundary_expected[name], boundary_actual[name])), None)
                            first_over_limit = next(
                                (name for name, score in boundaries.items() if not score["pass"]), None)
                        report["cases"].append({"precision": precision, "role": role, "offset": offset,
                            "status": "pass" if passed else "fail", "precision_hint": str(hint),
                            "io": "typed storage tensors", "device": device,
                            "execution_devices": execution_devices,
                            "compile_config": {key: str(value) for key, value in compile_config.items()},
                            "compile_seconds": compile_seconds, "infer_seconds": seconds,
                            "metrics": scores, "boundary_metrics": boundaries,
                            "boundary_first_non_exact": first_non_exact,
                            "boundary_first_over_limit": first_over_limit,
                            "raw_sha256": sha(raw), "pass": passed})
                        persist()
                    del engine, graph
                    gc.collect()
                del state, block
                gc.collect()
        report["status"] = "pass" if report["cases"] and all(case["pass"] for case in report["cases"]) else "fail"
    except BaseException as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        persist()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
