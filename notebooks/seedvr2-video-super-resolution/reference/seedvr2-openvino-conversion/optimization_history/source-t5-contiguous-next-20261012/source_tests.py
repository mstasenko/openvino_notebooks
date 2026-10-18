#!/usr/bin/env python3
"""Behavioral CPU-only tests for the additive contiguous T5 lifecycle."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location("contiguous_under_test", HERE / "contiguous.py")
assert loader is not None and loader.loader is not None
runner = importlib.util.module_from_spec(loader)
sys.modules[loader.name] = runner
loader.loader.exec_module(runner)
assert "torch" not in sys.modules and "numpy" not in sys.modules


def raises(expected, action):
    try:
        action()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def test_metadata_and_contract() -> None:
    row = runner.source_metadata()
    assert row["status"] == "PASS_METADATA_ONLY"
    assert row["payload_reads"] == 0
    assert row["torch_imports"] == row["gpu_queries"] == 0
    assert row["geometry"]["rgb_thwc"] == [5, 2160, 3840, 3]
    assert runner.SHAPES["mean"] == (1, 16, 2, 270, 480)
    assert runner.SHAPES["prediction"] == (2, 270, 480, 16)


def test_one_process_order_and_reuse() -> None:
    backend = runner.CpuContractBackend()
    rows = runner.ContiguousSequence(backend).run()
    expected_prefix = [
        "activate:vae", "encode:cold", "preserve:mean", "validate:mean", "release:vae",
        "cleanup:vae", "gate:cold", "activate:dit", "dit:cold", "preserve:prediction",
        "validate:prediction", "release:dit", "cleanup:dit", "activate:vae", "decode:cold",
        "preserve:rgb", "validate:rgb", "release:vae", "cleanup:vae", "finish:cold",
    ]
    assert backend.events[: len(expected_prefix)] == expected_prefix
    assert backend.activate_count == {"vae": 4, "dit": 2}
    assert len(rows) == 2 and rows[0]["call"] == "cold" and rows[1]["call"] == "warm"
    assert all(row["whole_t5_seconds"] > 0 for row in rows)
    assert backend.active is None


def test_exact_call_schedule_and_no_overlap() -> None:
    backend = runner.CpuContractBackend()
    raises(runner.ContractError, lambda: runner.ContiguousSequence(backend).run(("cold",)))
    backend = runner.CpuContractBackend()
    runner.ContiguousSequence(backend).run()
    active = None
    for event in backend.events:
        if event.startswith("activate:"):
            assert active is None
            active = event.split(":", 1)[1]
        elif event.startswith("release:"):
            assert active == event.split(":", 1)[1]
            active = None
    assert active is None


def test_gate_is_measured_and_has_no_t1_default() -> None:
    # The tiny bound is deliberately refusal-only test data; it is never a
    # fit authority and cannot enter VramFileGate.require().
    authority = runner.T5GateAuthority.from_mapping({
        "mode": "T5_CONTIGUOUS", "min_free_bytes": 1,
        "authority_sha256": "a" * 64, "source": "synthetic measured authority",
        "decision": "REFUSE_PRE_DIT",
    })
    assert authority.min_free_bytes == 1
    assert authority.decision == "REFUSE_PRE_DIT"
    assert not hasattr(runner, "MIN_FREE")
    backend = runner.CpuContractBackend(gate=lambda _call: {"size": 1, "usage": 1, "free": 0})
    raises(runner.GateRefusal, lambda: runner.ContiguousSequence(backend).run())
    assert "activate:dit" not in backend.events
    assert backend.active is None


def test_gate_json_uses_nonrecursive_authority_digest() -> None:
    body = {"mode": "T5_CONTIGUOUS", "min_free_bytes": 7,
            "source": "synthetic measured authority", "decision": "REFUSE_PRE_DIT"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["authority_sha256"] = hashlib.sha256(canonical).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gate.json"
        path.write_text(json.dumps(body, sort_keys=True) + "\n")
        authority = runner.T5GateAuthority.from_json(path, require_approval=False)
        raises(runner.ContractError, lambda: runner.T5GateAuthority.from_json(path))
    assert authority.min_free_bytes == 7
    assert authority.decision == "REFUSE_PRE_DIT"


def test_physical_gate_parser_is_strict_and_measured() -> None:
    authority = runner.T5GateAuthority.from_mapping({
        "mode": "T5_CONTIGUOUS", "min_free_bytes": 7,
        "authority_sha256": "c" * 64, "source": "synthetic measured authority",
        "decision": "REFUSE_PRE_DIT",
    })
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "vram_mm"
        path.write_text("size: 10\nusage: 2\n")
        gate = runner.VramFileGate(path, authority)
        assert gate.sample()["free"] == 8
        raises(runner.ContractError, lambda: gate.require("cold"))
        path.write_text("size: 10\nusage: 2\nusage: 1\n")
        raises(runner.ContractError, lambda: gate.sample())


def test_pre_dit_report_never_claims_fit_from_small_bound() -> None:
    body = {"mode": "T5_CONTIGUOUS", "min_free_bytes": 1,
            "source": "synthetic measured authority", "decision": "REFUSE_PRE_DIT"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["authority_sha256"] = hashlib.sha256(canonical).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gate_path = root / "gate.json"
        gate_path.write_text(json.dumps(body, sort_keys=True) + "\n")
        vram_path = root / "vram_mm"
        vram_path.write_text("size: 10\nusage: 8\n")
        authority = runner.T5GateAuthority.from_json(gate_path, require_approval=False)
        report = runner.pre_dit_gate_report(vram_path, authority)
        measured_refusal = runner.T5GateAuthority.from_mapping({
            "mode": "T5_CONTIGUOUS", "min_free_bytes": 9,
            "authority_sha256": "d" * 64, "source": "synthetic measured authority",
            "decision": "APPROVE_EXTERNAL",
        })
        physical_report = runner.pre_dit_gate_report(vram_path, measured_refusal)
    assert report["status"] == "REFUSED_PRE_DIT_AUTHORITY"
    assert report["fit_claim"] is False
    assert report["independent_approval"] is False
    assert physical_report["status"] == "REFUSED_PRE_DIT"
    assert physical_report["fit_claim"] is False


class FakeTensor:
    def __init__(self, device, dtype="torch.bfloat16", is_meta=False):
        self.device = device
        self.dtype = dtype
        self.is_meta = is_meta

    def is_floating_point(self):
        return self.dtype in {"torch.bfloat16", "bfloat16", "torch.float32", "float32"}


class FakeModule:
    def __init__(self, device):
        self.tensors = [FakeTensor(device)]

    def parameters(self, recurse=True):
        return iter(self.tensors)

    def buffers(self, recurse=True):
        return iter(())

    def to(self, device):
        for tensor in self.tensors:
            tensor.device = device
        return self


class FakeBlocks(FakeModule):
    def __init__(self, device):
        self.items = [FakeModule(device) for _ in range(32)]

    def __iter__(self):
        return iter(self.items)

    def parameters(self, recurse=True):
        return iter([tensor for item in self.items for tensor in item.tensors])

    def to(self, device):
        for item in self.items:
            item.to(device)
        return self


class FakeN32Model:
    def __init__(self):
        self.blocks = FakeBlocks("cpu")
        self.input = FakeModule("xpu")
        self.output = FakeModule("xpu")
        self.rope_lengths = [9] * 32

    def named_children(self):
        return iter((("blocks", self.blocks), ("input", self.input), ("output", self.output)))

    def parameters(self, recurse=True):
        return iter([tensor for _, child in self.named_children() for tensor in child.parameters()])

    def buffers(self, recurse=True):
        return iter(())


class FakeN32Adapter:
    def __init__(self):
        self.configure_calls = 0
        self.wrap_count = 0

    def configure_prefix(self, model, count, device, offload, debug):
        assert count == 32 and debug is not None
        if self.configure_calls:
            raise ValueError("refusing a second wrapper pass")
        self.configure_calls += 1
        self.wrap_count += 32
        model.blocks.to(offload)
        model.input.to(device)
        model.output.to(device)
        return {"selected_blocks": list(range(32))}


class FakeN32Reset:
    def __init__(self):
        self.calls = 0

    def reset(self, model):
        assert all(tensor.device == "cpu" for tensor in model.blocks.parameters())
        self.calls += 1
        rows = [{"name": f"blocks.{index}.attn.rope.rope", "before": length, "after": 0}
                for index, length in enumerate(model.rope_lengths)]
        model.rope_lengths = [0] * 32
        return rows


def test_n32_partial_placement_configure_once_and_reset_reuse() -> None:
    model = FakeN32Model()
    adapter = FakeN32Adapter()
    reset = FakeN32Reset()
    adapter.configure_prefix(model, 32, "xpu", "cpu", object())
    assert runner.verify_dit_mixed_placement(model, "xpu", "cpu") == {"blocks": 32, "io": 2}
    model.input.tensors[0].dtype = "torch.float32"
    raises(runner.ContractError, lambda: runner.verify_dit_mixed_placement(model, "xpu", "cpu"))
    model.input.tensors[0].dtype = "torch.bfloat16"
    model.blocks.items[0].tensors[0].is_meta = True
    raises(runner.ContractError, lambda: runner.verify_dit_mixed_placement(model, "xpu", "cpu"))
    model.blocks.items[0].tensors[0].is_meta = False
    reset.reset(model)
    model.blocks.items[0].to("xpu")
    raises(AssertionError, lambda: reset.reset(model))
    model.blocks.items[0].to("cpu")
    model.input.to("cpu")
    model.output.to("cpu")
    assert runner.verify_dit_cpu_placement(model) == {"tensors": 34}
    model.input.tensors[0].dtype = "torch.float32"
    raises(runner.ContractError, lambda: runner.verify_dit_cpu_placement(model))
    model.input.tensors[0].dtype = "torch.bfloat16"
    model.blocks.items[1].tensors[0].is_meta = True
    raises(runner.ContractError, lambda: runner.verify_dit_cpu_placement(model))
    model.blocks.items[1].tensors[0].is_meta = False
    model.input.to("xpu")
    model.output.to("xpu")
    assert runner.verify_dit_mixed_placement(model, "xpu", "cpu") == {"blocks": 32, "io": 2}
    reset.reset(model)
    raises(ValueError, lambda: adapter.configure_prefix(model, 32, "xpu", "cpu", object()))
    assert adapter.configure_calls == 1 and adapter.wrap_count == 32 and reset.calls == 2


def test_late_guard_preserves_output_without_exception_cleanup() -> None:
    backend = runner.CpuContractBackend(bad_stage="rgb")
    raises(runner.ContractError, lambda: runner.ContiguousSequence(backend).run())
    assert ("cold", "rgb") in backend.outputs
    assert backend.active == "vae"
    assert "cleanup:vae" not in backend.events[-2:]


def test_failed_dit_does_not_run_recovery_cleanup() -> None:
    backend = runner.CpuContractBackend(fail="dit")
    raises(RuntimeError, lambda: runner.ContiguousSequence(backend).run())
    assert ("cold", "mean") in backend.outputs
    assert backend.active == "dit"
    assert "cleanup:dit" not in backend.events


def test_failed_movement_does_not_run_recovery_cleanup() -> None:
    backend = runner.CpuContractBackend(fail="activate:vae")
    raises(RuntimeError, lambda: runner.ContiguousSequence(backend).run())
    assert backend.active == "vae"
    assert "cleanup:vae" not in backend.events


def test_bad_contracts_reject_before_acceptance() -> None:
    bad = runner.Synthetic("mean", (1, 16, 2, 270, 479), "float32")
    raises(runner.ContractError, lambda: runner.validate_payload("mean", bad))
    bad = runner.Synthetic("prediction", runner.SHAPES["prediction"], "float16")
    raises(runner.ContractError, lambda: runner.validate_payload("prediction", bad))
    raises(runner.ContractError, lambda: runner.T5GateAuthority.from_mapping({
        "mode": "T5_CONTIGUOUS", "min_free_bytes": 0,
        "authority_sha256": "b" * 64, "source": "invalid",
    }))


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    print(f"PASS {len(tests)} behavioral CPU source tests; no model/tensor/GPU imports")


if __name__ == "__main__":
    main()
