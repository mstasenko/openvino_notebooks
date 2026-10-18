#!/usr/bin/env python3
"""CPU-only graph/configuration helper for the opt-in split-BF16 PV path."""

from __future__ import annotations

from pathlib import Path
import os
import sys

BASE = Path(__file__).resolve().parents[1]
_runtime = os.environ.get("OPENVINO_RUNTIME")
RUNTIME = Path(_runtime).expanduser() if _runtime else None
sys.path[:0] = [str(path) for path in (RUNTIME, BASE / "openvino",
                BASE / "openvino" / "chunked_attention", BASE / "evidence")
                if path is not None and path.is_dir()]

import check_bf16_cross_bridge_b580_20261017 as bridge_runtime  # noqa: E402
import plus_sparse_xmx as kernel  # noqa: E402

PRECISE_TYPE, DEFAULT_TYPE, MACRO = "OtcCrossPreciseXmx", "OtcPlusSparseXmx", "OTC_CROSS_PV_SPLIT"
CONTROL_CL = BASE / "openvino" / "chunked_attention" / "plus_sparse_xmx.cl"
ARCHIVED_CL = (BASE / "evidence" /
               "plus-native-post-bridge-dynamic-b580-20261017-attempt2-lifecycle" /
               "source-37-plus_sparse_xmx.cl")


class OtcCrossPreciseXmx(kernel.OtcPlusSparseXmx):
    """Opt-in S1 node retaining the qualified four-input ABI."""

    def clone_with_new_inputs(self, inputs):
        return OtcCrossPreciseXmx(inputs)


def port_signature(port):
    shape = port.get_partial_shape()
    if not shape.is_static:
        raise ValueError("proof requires static ports")
    return (tuple(sorted(port.get_names())), str(port.get_element_type()),
            tuple(int(x) for x in shape.to_shape()))


def model_abi(model):
    return {"inputs": [port_signature(model.input(i)) for i in range(len(model.inputs))],
            "outputs": [port_signature(model.output(i)) for i in range(len(model.outputs))]}


def rewrite_clone(model, *, s1_count=1):
    """Clone and replace only the existing S1 node, retaining names/ABI."""
    if type(s1_count) is not int or s1_count not in (1, 2):
        raise ValueError("one standalone or two POST attention branches required")
    cloned = model.clone()
    nodes = [n for n in cloned.get_ordered_ops() if n.get_type_name() == DEFAULT_TYPE]
    if len(nodes) != s1_count or any(n.get_type_name() == PRECISE_TYPE for n in cloned.get_ordered_ops()):
        raise ValueError(f"expected {s1_count} unmodified {DEFAULT_TYPE} branches, found {len(nodes)}")
    for old in nodes:
        replacement = OtcCrossPreciseXmx([old.input_value(i) for i in range(old.get_input_size())])
        replacement.set_friendly_name(old.get_friendly_name())
        replacement.output(0).get_tensor().set_names(old.output(0).get_tensor().get_names())
        old.output(0).replace(replacement.output(0))
    cloned.validate_nodes_and_infer_types()
    if sum(n.get_type_name() == PRECISE_TYPE for n in cloned.get_ordered_ops()) != s1_count:
        raise ValueError("precise S1 replacement was not retained")
    return cloned


def make_precise_plugin_config(output: Path) -> Path:
    """Reuse the bridge config and rename only its S1 registration."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = bridge_runtime._make_plugin_config(output)
    text = config.read_text(encoding="utf-8")
    marker = '<CustomLayer name="OtcPlusSparseXmx" type="SimpleGPU" version="1">'
    start, end = text.find(marker), text.find("</CustomLayer>", text.find(marker))
    if start < 0 or end < 0 or text.find(marker, start + len(marker)) >= 0:
        raise ValueError("combined config does not have one complete default S1 entry")
    end += len("</CustomLayer>"); block = text[start:end]
    option = 'options="-cl-intel-256-GRF-per-thread"'
    entry = 'entry="otc_plus_sparse_xmx"'
    if block.count(option) != 1 or block.count(entry) != 1:
        raise ValueError("unexpected S1 compiler options")
    block = block.replace('name="OtcPlusSparseXmx"', f'name="{PRECISE_TYPE}"', 1)
    block = block.replace(entry, 'entry="otc_cross_precise_xmx"', 1)
    block = block.replace(option, f'options="-cl-intel-256-GRF-per-thread -D{MACRO}"', 1)
    config.write_text(text[:start] + block + text[end:], encoding="utf-8")
    return config


def default_source(text: str) -> str:
    """Select the no-macro branch for an exact archived-token check."""
    active, stack, out = True, [], []
    for line in text.splitlines(keepends=True):
        directive = line.strip()
        if directive == f"#if defined({MACRO})":
            stack.append(active); active = False
        elif directive == "#else":
            if not stack: raise ValueError("unmatched #else")
            active = stack[-1] and not active
        elif directive == "#endif":
            if not stack: raise ValueError("unmatched #endif")
            active = stack.pop()
        elif active:
            out.append(line)
    if stack: raise ValueError("unterminated conditional")
    return "".join(out)
