"""Bounded Plus learned-weight ownership.

The caller owns the 26 permanent PRE/POST banks and one active bank for
layers 26--29. This helper only maps an existing ResidentGraph ABI, validates
one complete ``load_layer`` result, and allocates or refills that flat bank.
Cross-K/V setup uses the same function with ``(cross,)``; output pairs are
owned by the caller and are not part of this bank.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from plus_storage import storage_tensor
from plus_weights import EXTRA_LAYER_KEYS, LAYER_KEYS, LAYER_SHAPES, ROLE_KEYS


def _port_name(key: str) -> str:
    return "weight__" + key.replace(".", "_")


def _expected_keys(graphs):
    if type(graphs) is not tuple or len(graphs) not in (1, 2):
        raise TypeError("graphs must be (pre, post) or (cross,)")
    if len(graphs) == 1:
        return ((graphs[0], "cross", tuple(EXTRA_LAYER_KEYS)),)
    return (
        (graphs[0], "pre", tuple(ROLE_KEYS["pre"])),
        (graphs[1], "post", tuple(ROLE_KEYS["post"])),
    )


def _graph_ports(graph, role, keys):
    inputs = getattr(graph, "inputs", None)
    if not isinstance(inputs, Mapping):
        raise TypeError(f"{role} graph must expose an input mapping")
    expected = {_port_name(key): key for key in keys}
    actual = {name for name in inputs if isinstance(name, str) and name.startswith("weight__")}
    if actual != set(expected):
        raise ValueError(
            f"{role} learned-input ABI changed: "
            f"missing={sorted(set(expected) - actual)}, extra={sorted(actual - set(expected))}"
        )
    if any(not isinstance(name, str) for name in inputs):
        raise ValueError(f"{role} graph has a non-string input name")
    return expected


def _ports(graphs):
    result = {}
    for graph, role, keys in _expected_keys(graphs):
        for name, key in _graph_ports(graph, role, keys).items():
            if name in result:
                raise ValueError("duplicate learned weight port")
            result[name] = (graph, key)
    expected = set(EXTRA_LAYER_KEYS if len(graphs) == 1
                   else (*ROLE_KEYS["pre"], *ROLE_KEYS["post"]))
    if {key for _, key in result.values()} != expected:
        raise ValueError("learned graph key mapping is incomplete")
    return result


def _validate_source(values):
    if not isinstance(values, Mapping) or set(values) != set(LAYER_KEYS):
        actual = set(values) if isinstance(values, Mapping) else set()
        raise ValueError(
            "complete authenticated layer values required: "
            f"missing={sorted(set(LAYER_KEYS) - actual)}, extra={sorted(actual - set(LAYER_KEYS))}"
        )
    for key in LAYER_KEYS:
        value = values[key]
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32)
                or tuple(value.shape) != tuple(LAYER_SHAPES[key])
                or not value.flags.c_contiguous):
            raise ValueError(f"layer source ABI changed: {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite layer source: {key}")


def _validate_owner(graph, name, owner):
    validator = getattr(graph, "_validate_tensor", None)
    if not callable(validator):
        raise TypeError("ResidentGraph._validate_tensor is required")
    validator(name, owner, graph.inputs[name])


def weight_bank(graphs, values, bank=None):
    """Allocate or synchronously refill one PRE/POST or cross-input bank.

    Complete source validation and all destination validation happen before
    any host storage is copied. ``storage_tensor`` is the sole BF16 converter;
    its temporary owning tensors are released before this function returns.
    """

    ports = _ports(graphs)
    _validate_source(values)
    names = tuple(ports)
    if bank is None:
        bank = {}
        for name in names:
            graph, _ = ports[name]
            bank[name] = graph.allocate(graph.inputs[name])
    elif not isinstance(bank, Mapping) or set(bank) != set(names):
        raise ValueError("destination bank keyset changed")
    for name in names:
        graph, _ = ports[name]
        _validate_owner(graph, name, bank[name])

    staged = {}
    try:
        # Stage only the inputs consumed by these graphs. This also performs
        # the existing BF16 finite/overflow checks before owner mutation.
        for name in names:
            _, key = ports[name]
            staged[name] = storage_tensor(values[key], "bf16")
        for name in names:
            staged[name].copy_to(bank[name])
    finally:
        staged.clear()
    return bank


