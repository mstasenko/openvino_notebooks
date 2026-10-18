#!/usr/bin/env python3
"""CPU-only contract tests for setup_only.SetupOnlySequence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import setup_only


class FakePayload:
    shape = (1, 16, 2, 270, 480)
    dtype = "float32"
    finite = True

    def tobytes(self, order="C"):
        assert order == "C"
        return b"signed-mean"


class FakeBackend:
    def __init__(self, *, free=setup_only.SETUP_MIN_FREE_BYTES + 1):
        self.active = None
        self.models = {"vae": object(), "dit": None}
        self.events = []
        self.encode_calls = 0
        self.dit_calls = 0
        self.free = free
        self.drop_cleared = False
        self.setup_cleared = False
        self.outputs = {}

    def activate(self, role):
        assert self.active is None
        self.active = role
        self.events.append("activate:" + role)
        if role == "dit":
            self.models["dit"] = object()

    def encode(self, call):
        assert self.active == "vae" and call == "setup"
        self.encode_calls += 1
        self.events.append("encode:setup")
        return FakePayload()

    def dit(self, *_args):
        self.dit_calls += 1
        raise AssertionError("setup-only sequence called DiT forward")

    def preserve(self, call, stage, payload):
        assert self.active == "vae" and call == "setup" and stage == "mean"
        self.events.append("preserve:mean")
        self.outputs[(call, stage)] = payload

    def validate(self, stage, payload):
        assert stage == "mean" and payload is self.outputs[("setup", "mean")]
        self.events.append("validate:mean")

    def release(self, role):
        assert self.active == role
        self.events.append("release:" + role)
        self.active = None

    def sync(self):
        self.events.append("sync")

    def allocator(self):
        self.events.append("allocator")
        cleared = self.drop_cleared or self.setup_cleared
        return {"memory_allocated": 0 if cleared else 1,
                "memory_reserved": 0 if cleared else 1}

    def sample_physical(self, label):
        self.events.append("snapshot:" + label)
        if label in {"dit_setup_before_clear", "dit_setup_after_clear", "after_dit_release"}:
            assert self.models.get("dit") is not None
        usage = 50 if self.setup_cleared else 100
        free = self.free + (50 if self.setup_cleared else 0)
        return {"size": 100 + self.free, "usage": usage, "free": free}


class FakeDropper:
    def __init__(self, backend):
        self.backend = backend

    def drop_cpu_vae_owner(self, call):
        assert call == "setup" and self.backend.active is None
        self.backend.events.append("drop:setup")
        self.backend.models["vae"] = None
        self.backend.drop_cleared = True
        return {"owner_dropped": True}


class MalformedSnapshotBackend:
    def __init__(self, physical, allocator=None):
        self.physical = physical
        self.allocator_row = allocator or {"memory_allocated": 0, "memory_reserved": 0}

    def sample_physical(self, _label):
        return self.physical

    def allocator(self):
        return self.allocator_row


def malformed_snapshot_cases():
    cases = {
        "negative_size": {"size": -1, "usage": 0, "free": 0},
        "bool_usage": {"size": 100, "usage": True, "free": 99},
        "inconsistent_free": {"size": 100, "usage": 40, "free": 61},
    }
    for label, physical in cases.items():
        try:
            setup_only._snapshot(MalformedSnapshotBackend(physical), label)
        except setup_only.ContractError:
            continue
        raise AssertionError(f"malformed physical snapshot accepted: {label}")
    try:
        setup_only._snapshot(
            MalformedSnapshotBackend(
                {"size": 100, "usage": 40, "free": 60},
                {"memory_allocated": -1, "memory_reserved": 0},
            ),
            "negative_allocator",
        )
    except setup_only.ContractError:
        return
    raise AssertionError("negative allocator counter accepted")


def run_case():
    backend = FakeBackend()
    dropper = FakeDropper(backend)

    def setup_clear():
        assert backend.active == "dit"
        backend.events.append("public_setup_clear")
        backend.setup_cleared = True

    def quiesce(value):
        assert value.active is None
        value.events.append("cpu_quiesce")
        value.models.clear()

    expected = hashlib.sha256(b"signed-mean").hexdigest()
    receipt = setup_only.SetupOnlySequence(
        backend, dropper, setup_clear, expected_mean_sha256=expected, quiesce=quiesce
    ).run()
    assert receipt["status"] == "PASS_SETUP_ONLY"
    assert receipt["vae_encode_calls"] == 1 and receipt["dit_forward_calls"] == 0
    assert receipt["mean_preserved_before_vae_release"] is True
    assert receipt["old_body_gate_invoked"] is False
    assert list(receipt["snapshots"]) == [
        "idle", "after_vae_release", "after_vae_drop",
        "dit_setup_before_clear", "dit_setup_after_clear",
        "after_dit_release", "final"
    ]
    events = backend.events
    assert events.index("preserve:mean") < events.index("release:vae") < events.index("drop:setup")
    assert events.index("snapshot:dit_setup_before_clear") < events.index("public_setup_clear")
    assert events.index("public_setup_clear") < events.index("snapshot:dit_setup_after_clear")
    assert events.index("snapshot:dit_setup_after_clear") < events.index("release:dit")
    assert receipt["snapshots"]["dit_setup_before_clear"]["physical"]["usage"] == 100
    assert receipt["snapshots"]["dit_setup_after_clear"]["physical"]["usage"] == 50
    assert "cpu_quiesce" in events and backend.dit_calls == 0
    assert backend.models == {}
    return receipt


def gate_refusal_case():
    backend = FakeBackend(free=setup_only.SETUP_MIN_FREE_BYTES - 1)
    dropper = FakeDropper(backend)
    try:
        setup_only.SetupOnlySequence(
            backend, dropper, lambda: (_ for _ in ()).throw(AssertionError("clear reached")),
            expected_mean_sha256=hashlib.sha256(b"signed-mean").hexdigest(),
        ).run()
    except setup_only.GateRefusal:
        assert "activate:dit" not in backend.events
        assert "public_setup_clear" not in backend.events
        assert backend.dit_calls == 0
        return
    raise AssertionError("setup gate unexpectedly passed")


def main():
    source = setup_only.source_metadata()
    malformed_snapshot_cases()
    receipt = run_case()
    gate_refusal_case()
    print(json.dumps({
        "status": "PASS",
        "source": source,
        "positive": {
            "vae_encode_calls": receipt["vae_encode_calls"],
            "dit_forward_calls": receipt["dit_forward_calls"],
            "snapshot_labels": list(receipt["snapshots"]),
        },
        "negative": "setup_gate_refuses_before_dit_setup",
        "torch_imports": 0,
        "gpu_queries": 0,
        "payload_reads": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
