#!/usr/bin/env python3
"""Additive contiguous UHD T5 reference.

The module is deliberately import-safe: source checks and the contract tests do
not import NumPy, Torch, Core, or checkpoint readers.  The XPU implementation is
entered only by the explicit root-controlled ``--run`` path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
SPEC_PATH = EVIDENCE / "temporal-uhd-complete-source-20261012" / "spec.py"
CHILD_PATH = EVIDENCE / "temporal-uhd-complete-source-20261012" / "child.py"
BENCHMARK_PATH = EVIDENCE / "temporal-uhd-complete-source-20261012" / "benchmark.py"
T1_DRIVER_PATH = EVIDENCE / "source-contiguous-t1-20261012" / "driver_v2.py"
ADAPTER_PATH = EVIDENCE / "temporal-uhd-n32-rope-reset-20261012" / "adapter.py"
RESET_PATH = EVIDENCE / "temporal-uhd-n32-rope-reset-20261012" / "rope_reset.py"
FIXTURE_DIR = EVIDENCE / "native-real-uhd-t5-fixture-20261011" / "fixture"
BDF = "0000:03:00.0"

# These are the small, already-reviewed producer seams.  The full execution
# closure remains owned by the frozen producer's SOURCE-PINS.json; this leaf
# does not copy or weaken that closure.
SOURCE_PINS = {
    SPEC_PATH: "62b0624561ff49e674989533ef29d97cc1e21fd27377e4ab2f32f826536a318f",
    CHILD_PATH: "5733e1ed600f885562c96bdcd0ae241b7e149dfa54a76fa487600d0ae6f662ab",
    BENCHMARK_PATH: "a23929ce0a8b079c4cfb0c414c6d109fbdc62d323913d9af8137540114da175c",
    T1_DRIVER_PATH: "f36433c131ccf21c4a512fab0de079b961364ce6f6d97c4f4b1302bd4cda86c3",
    ADAPTER_PATH: "4385b5b629ab714d664adcdc112806219fc8d966edef7fd64f25a981d7f0b8b4",
    RESET_PATH: "ebaea1b765f21e033dccb6f378b1641106d6b4854859a1d41ffe46d03932609c",
}

SHAPES = {
    "normalized": (1, 3, 5, 2160, 3840),
    "noise": (2, 270, 480, 16),
    "text": (58, 5120),
    "mean": (1, 16, 2, 270, 480),
    "prediction": (2, 270, 480, 16),
    "decoded": (1, 3, 5, 2160, 3840),
    "rgb": (5, 2160, 3840, 3),
}
CALLS = ("cold", "warm")


class ContractError(ValueError):
    """A malformed source, tensor contract, or lifecycle transition."""


class GateRefusal(RuntimeError):
    """A measured physical admission refusal, not a device fault."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ContractError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path: Path, name: str) -> Any:
    loader = importlib.util.spec_from_file_location(name, path)
    if loader is None or loader.loader is None:
        raise ContractError(f"cannot load source helper: {path}")
    module = importlib.util.module_from_spec(loader)
    sys.modules[name] = module
    loader.loader.exec_module(module)
    return module


def source_metadata() -> dict[str, Any]:
    """Verify only small source/receipt metadata; never read fixture payloads."""

    rows: dict[str, Any] = {}
    for path, expected in SOURCE_PINS.items():
        info = path.lstat()
        require(path.is_file() and not path.is_symlink(), f"source is not a regular file: {path}")
        actual = digest(path)
        require(actual == expected, f"source pin changed: {path}")
        rows[str(path)] = {"bytes": info.st_size, "sha256": actual}

    spec = load_module(SPEC_PATH, "contiguous_uhd_spec_metadata")
    receipt, checked = spec.metadata()
    require(tuple(spec.SHAPES["rgb"]) == SHAPES["rgb"], "UHD RGB shape differs")
    require(tuple(spec.SHAPES["mean"]) == SHAPES["mean"], "UHD latent shape differs")
    require(tuple(spec.SHAPES["prediction"]) == SHAPES["prediction"], "UHD prediction shape differs")
    # Exercise the actual producer authority in metadata-only mode.  This
    # authenticates the complete source closure and imports no model reader;
    # fixture payload files are explicitly skipped by the frozen helper.
    previous_spec = sys.modules.get("spec")
    sys.modules["spec"] = spec
    try:
        child = load_module(CHILD_PATH, "contiguous_uhd_child_metadata")
        ref, _adapter = child.authority(metadata_only=True)
        reset = load_module(Path(child.RESET), "contiguous_qualified_rope_reset_metadata")
    finally:
        if previous_spec is None:
            sys.modules.pop("spec", None)
        else:
            sys.modules["spec"] = previous_spec
    return {
        "status": "PASS_METADATA_ONLY",
        "source_files": rows,
        "fixture_receipt_sha256": spec.RECEIPT_SHA,
        "fixture_check_sha256": spec.CHECK_SHA,
        "geometry": receipt["geometry"],
        "checked_keys": sorted(checked),
        "producer_source_head": ref.HEAD,
        "authenticated_source_root": str(ref.SOURCE),
        "qualified_adapter": str(ADAPTER_PATH),
        "qualified_reset": str(RESET_PATH),
        "qualified_reset_expected_sites": len(reset.EXPECTED),
        "payload_reads": 0,
        "torch_imports": 0,
        "gpu_queries": 0,
    }


@dataclass(frozen=True)
class T5GateAuthority:
    """Measured T5 admission authority; no T1 threshold is implicit here.

    ``decision`` is intentionally separate from ``min_free_bytes``.  A
    measured number alone cannot authorize a launch: the refusal/report path
    can observe it without making a fit claim, while the runtime path accepts
    only an independently approved authority document.
    """

    mode: str
    min_free_bytes: int
    authority_sha256: str
    source: str
    decision: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "T5GateAuthority":
        required = ("mode", "min_free_bytes", "authority_sha256", "source", "decision")
        require(all(key in row for key in required), "T5 gate authority fields are incomplete")
        require(row["mode"] == "T5_CONTIGUOUS", "gate is not a T5 contiguous authority")
        minimum = row["min_free_bytes"]
        require(type(minimum) is int and minimum > 0, "T5 gate minimum must be a positive integer")
        authority = row["authority_sha256"]
        require(isinstance(authority, str) and re.fullmatch(r"[0-9a-f]{64}", authority) is not None,
                "T5 gate authority hash is malformed")
        source = row["source"]
        require(isinstance(source, str) and source.strip(), "T5 gate source is missing")
        decision = row["decision"]
        require(decision in {"REFUSE_PRE_DIT", "APPROVE_EXTERNAL"},
                "T5 gate decision must be REFUSE_PRE_DIT or APPROVE_EXTERNAL")
        return cls(row["mode"], minimum, authority, source, decision)

    @classmethod
    def from_json(cls, path: Path, *, require_approval: bool = True) -> "T5GateAuthority":
        info = path.lstat()
        require(path.is_file() and not path.is_symlink() and info.st_nlink == 1,
                "T5 gate authority must be an owned regular file")
        with path.open() as stream:
            row = json.load(stream)
        require(isinstance(row, dict), "T5 gate authority is not an object")
        # ``authority_sha256`` identifies the separately reviewed authority
        # body, excluding the digest field.  Hashing the complete JSON would
        # be self-referential and would make every valid gate impossible to
        # represent.
        result = cls.from_mapping(row)
        body = {key: value for key, value in row.items() if key != "authority_sha256"}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        require(hashlib.sha256(canonical).hexdigest() == result.authority_sha256,
                "T5 gate authority content hash differs")
        if require_approval:
            require(result.decision == "APPROVE_EXTERNAL",
                    "T5 runtime requires an independently approved gate authority")
        return result


class VramFileGate:
    """Read-only debugfs sampler used only after a root-approved gate is supplied."""

    def __init__(self, path: Path, authority: T5GateAuthority,
                 save: Callable[[dict[str, int]], None] | None = None) -> None:
        self.path = path
        self.authority = authority
        self.save = save
        self.observations: list[dict[str, int]] = []

    def sample(self) -> dict[str, int]:
        text = self.path.read_text()
        fields: dict[str, int] = {}
        for key in ("size", "usage"):
            rows = re.findall(r"^[ \t]*" + key + r":[ \t]+([0-9]+)[ \t]*$", text, re.MULTILINE)
            require(len(rows) == 1, f"T5 physical allocator schema differs: {key}")
            fields[key] = int(rows[0])
        require(fields["usage"] <= fields["size"], "T5 physical usage exceeds allocator size")
        row = {**fields, "free": fields["size"] - fields["usage"]}
        self.observations.append(row)
        if self.save is not None:
            self.save(row)
        return row

    def require(self, label: str) -> dict[str, int]:
        require(self.authority.decision == "APPROVE_EXTERNAL",
                "T5 gate has no independent approval; report refusal without launching")
        row = self.sample()
        if row["free"] < self.authority.min_free_bytes:
            raise GateRefusal(
                f"T5 physical admission denied before DiT ({label}): "
                f"free={row['free']} required={self.authority.min_free_bytes}"
            )
        return row


def pre_dit_gate_report(path: Path, authority: T5GateAuthority) -> dict[str, Any]:
    """Sample physical VRAM without device APIs and make no fit claim.

    This is the only operationally useful path while an independent launch
    authority is absent.  It reports a measured refusal when the supplied
    authority's bound is not met, and otherwise still reports observation-only
    status.  It never calls Torch/XPU or treats a small synthetic bound as fit.
    """

    row = VramFileGate(path, authority).sample()
    if authority.decision == "REFUSE_PRE_DIT":
        status = "REFUSED_PRE_DIT_AUTHORITY"
    elif row["free"] < authority.min_free_bytes:
        status = "REFUSED_PRE_DIT"
    else:
        status = "OBSERVED_GATE_PASS_NO_FIT_CLAIM"
    return {
        "status": status,
        "fit_claim": False,
        "independent_approval": authority.decision == "APPROVE_EXTERNAL",
        "authority_decision": authority.decision,
        "min_free_bytes": authority.min_free_bytes,
        "physical": row,
    }


def _shape_and_finite(payload: Any, expected: tuple[int, ...], dtype: str) -> None:
    require(tuple(getattr(payload, "shape", ())) == expected, f"{dtype} output shape differs")
    actual_dtype = str(getattr(payload, "dtype", ""))
    require(actual_dtype == dtype, f"{dtype} output dtype differs: {actual_dtype}")
    finite = getattr(payload, "finite", None)
    if finite is not None:
        require(bool(finite), f"{dtype} output is nonfinite")
    elif hasattr(payload, "__array__"):
        import numpy as np  # Runtime-only validation for real arrays.
        require(bool(np.isfinite(payload).all()), f"{dtype} output is nonfinite")


def validate_payload(stage: str, payload: Any) -> None:
    expected = {"mean": SHAPES["mean"], "prediction": SHAPES["prediction"], "rgb": SHAPES["rgb"]}
    dtype = {"mean": "float32", "prediction": "float32", "rgb": "uint8"}
    require(stage in expected, f"unknown contiguous output stage: {stage}")
    _shape_and_finite(payload, expected[stage], dtype[stage])


def _device_type(value: Any) -> str:
    return str(getattr(value, "type", value))


def _module_tensors(module: Any) -> list[Any]:
    return [*module.parameters(recurse=True), *module.buffers(recurse=True)]


def _verify_tensor(tensor: Any, expected_device: Any, label: str) -> None:
    require(not bool(getattr(tensor, "is_meta", False)), f"{label} retains a meta tensor")
    floating = getattr(tensor, "is_floating_point", False)
    floating = floating() if callable(floating) else floating
    if floating:
        require(str(getattr(tensor, "dtype", "")) in {"torch.bfloat16", "bfloat16"},
                f"{label} floating tensor is not BF16")
    require(_device_type(getattr(tensor, "device", "")) == _device_type(expected_device),
            f"{label} tensor device differs")


def verify_dit_mixed_placement(model: Any, main_device: Any = "xpu",
                               offload_device: Any = "cpu") -> dict[str, int]:
    """Require the qualified N32 shape: CPU blocks plus XPU non-block I/O."""

    blocks = list(getattr(model, "blocks", ()))
    require(len(blocks) == 32, "qualified DiT must expose exactly 32 blocks")
    block_tensors = [tensor for block in blocks for tensor in _module_tensors(block)]
    io_tensors = [tensor for name, child in model.named_children() if name != "blocks"
                  for tensor in _module_tensors(child)]
    require(block_tensors and io_tensors, "qualified DiT placement has no parameter-bearing regions")
    offload = _device_type(offload_device)
    main = _device_type(main_device)
    require(all(_device_type(tensor.device) == offload for tensor in block_tensors),
            "N32 block placement is not entirely CPU/offload")
    require(all(_device_type(tensor.device) == main for tensor in io_tensors),
            "N32 non-block placement is not entirely on the active device")
    for tensor in block_tensors:
        _verify_tensor(tensor, offload_device, "N32 block")
    for tensor in io_tensors:
        _verify_tensor(tensor, main_device, "N32 non-block")
    return {"blocks": len(block_tensors), "io": len(io_tensors)}


def verify_dit_cpu_placement(model: Any, device: Any = "cpu") -> dict[str, int]:
    """Require all DiT regions CPU only after a healthy active-stage release."""

    tensors = _module_tensors(model)
    require(tensors, "DiT CPU release has no parameter-bearing regions")
    for tensor in tensors:
        _verify_tensor(tensor, device, "DiT CPU release")
    return {"tensors": len(tensors)}


class ContiguousSequence:
    """The only phase scheduler: one owner process, no stage subprocesses."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def run(self, calls: tuple[str, ...] = CALLS) -> list[dict[str, Any]]:
        require(calls == CALLS, "contiguous runner requires exactly cold then warm calls")
        rows: list[dict[str, Any]] = []
        for call in calls:
            started = self.backend.now()
            row: dict[str, Any] = {"call": call}

            stage_started = self.backend.now()
            self.backend.activate("vae")
            mean = self.backend.encode(call)
            self.backend.preserve(call, "mean", mean)
            self.backend.validate("mean", mean)
            self.backend.release("vae")
            row["encode_move_infer_release_seconds"] = self.backend.now() - stage_started

            gate_started = self.backend.now()
            row["t5_gate"] = self.backend.require_t5_gate(call)
            row["t5_gate_seconds"] = self.backend.now() - gate_started

            stage_started = self.backend.now()
            self.backend.activate("dit")
            prediction = self.backend.dit(call, mean)
            self.backend.preserve(call, "prediction", prediction)
            self.backend.validate("prediction", prediction)
            self.backend.release("dit")
            row["dit_move_infer_release_seconds"] = self.backend.now() - stage_started

            stage_started = self.backend.now()
            self.backend.activate("vae")
            rgb = self.backend.decode(call, prediction)
            # Preserve owned CPU output before the late validation, so a failed
            # guard cannot erase useful evidence and cannot trigger cleanup.
            self.backend.preserve(call, "rgb", rgb)
            self.backend.validate("rgb", rgb)
            self.backend.release("vae")
            row["decode_move_infer_release_seconds"] = self.backend.now() - stage_started

            row["whole_t5_seconds"] = self.backend.now() - started
            self.backend.finish_call(call, rgb)
            rows.append(row)
        outputs = getattr(self.backend, "outputs", {})
        for stage in ("mean", "prediction", "rgb"):
            first = outputs.get(("cold", stage))
            second = outputs.get(("warm", stage))
            require(first is not None and second is not None, f"missing cold/warm {stage} output")
            if hasattr(first, "tobytes") and hasattr(second, "tobytes"):
                require(first.tobytes(order="C") == second.tobytes(order="C"),
                        f"cold/warm {stage} output differs")
            else:
                require(first == second, f"cold/warm {stage} output differs")
        return rows


class CpuContractBackend:
    """Small behavioral backend used by source tests; it owns no real tensors."""

    def __init__(self, *, bad_stage: str | None = None, fail: str | None = None,
                 gate: Callable[[str], Mapping[str, int]] | None = None) -> None:
        self.bad_stage = bad_stage
        self.fail = fail
        self.gate = gate or (lambda _call: {"size": 2, "usage": 0, "free": 2})
        self.active: str | None = None
        self.events: list[str] = []
        self.outputs: dict[tuple[str, str], Any] = {}
        self.gate_rows: list[Mapping[str, int]] = []
        self.models = {"vae": object(), "dit": object()}
        self.activate_count = {"vae": 0, "dit": 0}
        self._ticks = 0.0

    def now(self) -> float:
        self._ticks += 0.01
        return self._ticks

    def activate(self, role: str) -> None:
        require(self.active is None, "CPU contract activated overlapping stages")
        self.active = role
        self.activate_count[role] += 1
        self.events.append("activate:" + role)
        if self.fail == "activate:" + role:
            raise RuntimeError("synthetic movement failure")

    def encode(self, call: str) -> Any:
        require(self.active == "vae", "encode owner is not VAE")
        self.events.append("encode:" + call)
        return Synthetic("mean", SHAPES["mean"], "float32")

    def dit(self, call: str, _mean: Any) -> Any:
        require(self.active == "dit", "DiT owner is not active")
        self.events.append("dit:" + call)
        if self.fail == "dit":
            raise RuntimeError("synthetic DiT failure")
        return Synthetic("prediction", SHAPES["prediction"], "float32")

    def decode(self, call: str, _prediction: Any) -> Any:
        require(self.active == "vae", "decode owner is not VAE")
        self.events.append("decode:" + call)
        shape = (5, 2160, 3840, 3) if self.bad_stage != "rgb" else (5, 2160, 3840, 2)
        return Synthetic("rgb", shape, "uint8")

    def preserve(self, call: str, stage: str, payload: Any) -> None:
        self.events.append("preserve:" + stage)
        self.outputs[(call, stage)] = payload.copy()

    def validate(self, stage: str, payload: Any) -> None:
        self.events.append("validate:" + stage)
        validate_payload(stage, payload)

    def release(self, role: str) -> None:
        require(self.active == role, "CPU contract released the wrong owner")
        self.events.append("release:" + role)
        self.events.append("cleanup:" + role)
        self.active = None

    def require_t5_gate(self, call: str) -> Mapping[str, int]:
        self.events.append("gate:" + call)
        row = self.gate(call)
        self.gate_rows.append(row)
        if row["free"] <= 0:
            raise GateRefusal("synthetic T5 gate refusal")
        return row

    def finish_call(self, call: str, _rgb: Any) -> None:
        self.events.append("finish:" + call)


@dataclass(frozen=True)
class Synthetic:
    name: str
    shape: tuple[int, ...]
    dtype: str
    finite: bool = True

    def copy(self) -> "Synthetic":
        return Synthetic(self.name, self.shape, self.dtype, self.finite)


def contract_self_test() -> dict[str, Any]:
    backend = CpuContractBackend()
    rows = ContiguousSequence(backend).run()
    expected = [
        "activate:vae", "encode:cold", "preserve:mean", "validate:mean", "release:vae",
        "cleanup:vae", "gate:cold", "activate:dit", "dit:cold", "preserve:prediction",
        "validate:prediction", "release:dit", "cleanup:dit", "activate:vae", "decode:cold",
        "preserve:rgb", "validate:rgb", "release:vae", "cleanup:vae", "finish:cold",
    ]
    require(backend.events[: len(expected)] == expected, "cold lifecycle order differs")
    require(backend.activate_count == {"vae": 4, "dit": 2}, "model owner placement schedule differs")
    require(len(rows) == 2 and all(row["whole_t5_seconds"] > 0 for row in rows), "cold/warm clocks missing")
    return {"status": "PASS", "events": len(backend.events), "calls": len(rows),
            "models_loaded": {key: 1 for key in backend.models}}


def archive_outputs(output: Path, backend: Any, rows: list[dict[str, Any]], metadata: dict[str, Any],
                    *, runtime: Mapping[str, Any] | None = None,
                    status: str = "completed", error: BaseException | None = None) -> None:
    """Archive only CPU-owned outputs after the action clocks have closed."""

    output.mkdir(mode=0o700)
    files: dict[str, Any] = {}
    for (call, stage), payload in getattr(backend, "outputs", {}).items():
        if not hasattr(payload, "tobytes"):
            continue
        suffix = ".rgb24" if stage == "rgb" else ".f32"
        path = output / f"{call}.{stage}{suffix}"
        with path.open("xb") as stream:
            stream.write(payload.tobytes(order="C"))
        files[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path),
                            "shape": list(payload.shape), "dtype": str(payload.dtype)}
    receipt = {"status": status, "source": metadata, "calls": rows, "files": files,
               "sequence": "single-process encode->DiT->decode; CPU-owned handoffs",
               "quality_award": False}
    if runtime is not None:
        receipt["runtime"] = runtime
    if error is not None:
        receipt.update(error_type=type(error).__name__, error=str(error),
                       GPU_calls_in_handler=0, no_retry=True)
    with (output / ("receipt.json" if status == "completed" else "failure-receipt.json")).open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)


def _torch_backend(spec: Any, ref: Any, adapter: Any, reset: Any,
                   gate: VramFileGate, bdf: str) -> Any:
    """Construct the real backend only after all non-GPU source checks pass."""

    import numpy as np
    import torch

    require(torch.__version__ == "2.9.1+xpu" and torch.version.cuda is None and torch.version.hip is None
            and not hasattr(torch._C, "_cuda_getDeviceCount"), "pinned XPU-only Torch required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    # The authenticated producer reference owns the source root.  Never derive
    # it from this evidence leaf: that points at evidence/source, not the
    # reviewed checkout.
    sys.path.insert(0, str(ref.SOURCE))
    resolver = ref.load_helper("evidence/vae-backend-20260918/torch_vae_backend_probe.py",
                              "contiguous_xpu_resolver")
    device, identity = resolver._resolve_xpu_identity(torch, bdf)
    torch.xpu.set_device(device)
    from src.optimization.blockswap import _configure_io_components
    from src.optimization.memory_manager import clear_memory, clear_rope_lru_caches, _standard_model_movement
    from src.utils.debug import Debug

    class Backend:
        def __init__(self) -> None:
            self.torch = torch
            self.np = np
            self.spec = spec
            self.ref = ref
            self.device = device
            self.identity = identity
            self.adapter = adapter
            self.reset = reset
            self.gate = gate
            self.models: dict[str, Any] = {"vae": None, "dit": None}
            self.active: str | None = None
            self.outputs: dict[tuple[str, str], Any] = {}
            self.events: list[str] = []
            self.input_values: dict[str, Any] | None = None
            self.debug = Debug(enabled=False)
            # The strong reference is intentional: the qualified adapter keeps
            # weak references to Debug while N32 wrappers are active.
            self.dit_configured = False
            self.timings: dict[str, Any] = {
                "model_load_seconds": {}, "n32_configure_seconds": None,
                "n32_rebind_seconds": [], "movement_seconds": [], "clear_seconds": [],
                "rope_clear_seconds": [], "reset_seconds": [], "reset_rows": [],
                "n32_placement": None,
                "input_preload_seconds": None,
            }

        def now(self) -> float:
            return time.monotonic()

        def sync(self) -> None:
            torch.xpu.synchronize(self.device)

        def allocator(self) -> dict[str, int]:
            return {name: int(getattr(torch.xpu, name)(self.device))
                    for name in ("memory_allocated", "memory_reserved")}

        def prepare_inputs(self) -> None:
            started = time.monotonic()
            self.input_values = self.spec.values(("rgb", "noise", "text"))
            self.timings["input_preload_seconds"] = time.monotonic() - started

        def runtime_metadata(self) -> dict[str, Any]:
            return {
                "device_identity": self.identity,
                "cpu_input_ready_before_cold_clock": True,
                "input_preload_seconds": self.timings["input_preload_seconds"],
                "model_load_seconds": self.timings["model_load_seconds"],
                "n32_configure_seconds": self.timings["n32_configure_seconds"],
                "n32_rebind_seconds": self.timings["n32_rebind_seconds"],
                "movement_seconds": self.timings["movement_seconds"],
                "clear_seconds": self.timings["clear_seconds"],
                "rope_clear_seconds": self.timings["rope_clear_seconds"],
                "reset_seconds": self.timings["reset_seconds"],
                "reset_rows": self.timings["reset_rows"],
                "dit_topology": "32 qualified block wrappers; blocks CPU/offload, non-block I/O XPU",
                "cpu_model_owners": "VAE retained on CPU between encode/decode; N32 DiT blocks CPU/offload",
            }

        def _model(self, role: str) -> Any:
            model = self.models[role]
            if model is not None:
                return model
            started = time.monotonic()
            if role == "dit":
                # Preserve the qualified producer topology: streamed BF16
                # weights target XPU, then all 32 blocks are explicitly moved
                # to CPU/offload and wrapped once for dynamic forward movement.
                model = ref.streamed_dit(torch, self.device)
                configured = time.monotonic()
                placement = self.adapter.configure_prefix(
                    model, 32, self.device, torch.device("cpu"), self.debug
                )
                self.timings["n32_configure_seconds"] = time.monotonic() - configured
                self.timings["n32_placement"] = placement
                verify_dit_mixed_placement(model, self.device, torch.device("cpu"))
                self.dit_configured = True
            else:
                cpu = torch.device("cpu")
                loader = ref.load_helper("openvino-vae-probe.py", "contiguous_vae_loader")
                weight = ref.TRIAL / "models" / ref.WEIGHTS["vae"][0]
                model, _ = loader.load_reduced_vae(ref.SOURCE, weight)
                model = model.to(device=cpu, dtype=torch.bfloat16).eval()
                ref.vae_runtime(model)
            self.models[role] = model
            self.timings["model_load_seconds"][role] = time.monotonic() - started
            return model

        def activate(self, role: str) -> None:
            require(self.active is None, "GPU stages overlap")
            first_model = self.models[role] is None
            model = self._model(role)
            cpu = torch.device("cpu")
            if role == "dit":
                if not first_model:
                    started = time.monotonic()
                    _configure_io_components(model, self.device, cpu, False, self.debug)
                    self.timings["n32_rebind_seconds"].append(time.monotonic() - started)
                self.sync()
                verify_dit_mixed_placement(model, self.device, cpu)
            else:
                started = time.monotonic()
                _standard_model_movement(model, cpu, self.device, "xpu", "VAE")
                self.sync()
                self.timings["movement_seconds"].append({"role": role, "direction": "cpu_to_xpu",
                                                          "seconds": time.monotonic() - started})
                require(all(t.device == self.device and not t.is_meta
                            for t in (*model.parameters(), *model.buffers())),
                        f"{role} model placement incomplete")
            self.active = role
            self.events.append("activate:" + role)

        def release(self, role: str) -> None:
            require(self.active == role, "GPU stage owner differs")
            model = self.models[role]
            if role == "dit":
                # The output was already copied and preserved by the caller;
                # cache cleanup is therefore a healthy boundary operation and
                # cannot erase failure evidence.
                started = time.monotonic()
                clear_rope_lru_caches(model)
                self.timings["rope_clear_seconds"].append(time.monotonic() - started)
                started = time.monotonic()
                for name, child in model.named_children():
                    if name != "blocks":
                        child.to(torch.device("cpu"))
                self.timings["movement_seconds"].append({"role": role, "direction": "xpu_io_to_cpu",
                                                          "seconds": time.monotonic() - started})
                verify_dit_cpu_placement(model)
            else:
                started = time.monotonic()
                _standard_model_movement(model, self.device, torch.device("cpu"), "cpu", "VAE")
                self.timings["movement_seconds"].append({"role": role, "direction": "xpu_to_cpu",
                                                          "seconds": time.monotonic() - started})
                require(all(t.device.type == "cpu" and not t.is_meta
                            for t in (*model.parameters(), *model.buffers())),
                        f"{role} CPU ownership incomplete")
            self.sync()
            started = time.monotonic()
            clear_memory(deep=True, force=True)
            self.timings["clear_seconds"].append({"role": role, "seconds": time.monotonic() - started})
            self.sync()
            stats = self.allocator()
            require(stats["memory_allocated"] == stats["memory_reserved"] == 0,
                    "GPU allocator remains live after healthy release")
            self.active = None
            self.events.append("release:" + role)

        def _gpu(self, value: Any) -> Any:
            return torch.from_numpy(np.ascontiguousarray(value)).to(device=self.device, dtype=torch.bfloat16)

        def _readback(self, value: Any) -> Any:
            result = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
            self.sync()
            return result

        def _values(self) -> dict[str, Any]:
            require(self.input_values is not None,
                    "input fixture must be preloaded before the cold whole-T5 clock")
            return self.input_values

        def encode(self, _call: str) -> Any:
            require(self.active == "vae", "encode VAE is not active")
            values = self._values()
            with torch.inference_mode():
                value = self._gpu(self.spec.normalize(values["rgb"]))
                encoded = self.models["vae"].tiled_encode(value, tile_size=(512, 512), tile_overlap=(64, 64))
                self.sync()
                moments = self._readback(encoded[:, :16])
            del encoded, value
            return np.ascontiguousarray(moments, dtype=np.float32)

        def dit(self, _call: str, mean: Any) -> Any:
            require(self.active == "dit", "DiT is not active")
            started = time.monotonic()
            rows = self.reset.reset(self.models["dit"])
            self.timings["reset_seconds"].append(time.monotonic() - started)
            self.timings["reset_rows"].append(rows)
            values = self._values()
            condition = mean[0].transpose(1, 2, 3, 0) * np.float32(0.9152)
            video = np.concatenate((values["noise"], condition,
                                    np.ones((2, 270, 480, 1), np.float32)), axis=-1).reshape(-1, 33)
            with torch.inference_mode():
                value = self._gpu(video)
                text = self._gpu(values["text"])
                timestep = self._gpu(np.array([1000], np.float32))
                vid_shape = torch.tensor([[2, 270, 480]], device=self.device, dtype=torch.long)
                text_shape = torch.tensor([[58]], device=self.device, dtype=torch.long)
                output = self.models["dit"](value, text, vid_shape, text_shape, timestep,
                                             disable_cache=False).vid_sample
                prediction = self._readback(output).reshape(SHAPES["prediction"])
            del output, value, text, timestep, vid_shape, text_shape
            return np.ascontiguousarray(prediction, dtype=np.float32)

        def decode(self, _call: str, prediction: Any) -> Any:
            require(self.active == "vae", "decode VAE is not active")
            values = self._values()
            restored = self.spec.restore(values["noise"], prediction)
            with torch.inference_mode():
                value = self._gpu(restored)
                decoded = self.models["vae"].tiled_decode(value, tile_size=(512, 512), tile_overlap=(64, 64))
                self.sync()
                raw = self._readback(decoded)
            del decoded, value, restored
            return np.ascontiguousarray(self.spec.rgb(raw))

        def preserve(self, call: str, stage: str, payload: Any) -> None:
            self.outputs[(call, stage)] = np.array(payload, copy=True, order="C")

        def validate(self, stage: str, payload: Any) -> None:
            validate_payload(stage, payload)

        def require_t5_gate(self, call: str) -> Mapping[str, int]:
            row = self.gate.require(call)
            self.events.append("gate:" + call)
            return row

        def finish_call(self, _call: str, _rgb: Any) -> None:
            require(self.active is None, "call finished with an active GPU owner")

    return Backend()


def run_xpu(output: Path, gate_json: Path, fixture: Path, bdf: str) -> None:
    metadata = source_metadata()
    require(fixture.resolve() == FIXTURE_DIR.resolve(), "only the reviewed UHD fixture is accepted")
    backend = None
    failed = False
    try:
        authority = T5GateAuthority.from_json(gate_json)
        vram = Path("/sys/kernel/debug/dri") / bdf / "tile0" / "vram_mm"
        gate = VramFileGate(vram, authority)
        spec = load_module(SPEC_PATH, "contiguous_uhd_spec_runtime")
        # Full authority is deliberately entered only in the root-controlled path.
        previous_spec = sys.modules.get("spec")
        sys.modules["spec"] = spec
        try:
            child = load_module(CHILD_PATH, "contiguous_uhd_child_runtime")
        finally:
            if previous_spec is None:
                sys.modules.pop("spec", None)
            else:
                sys.modules["spec"] = previous_spec
        ref, adapter = child.authority(metadata_only=False)
        reset = load_module(Path(child.RESET), "contiguous_qualified_rope_reset")
        backend = _torch_backend(spec, ref, adapter, reset, gate, bdf)
        # Fixture hashing and input conversion are deliberately outside the
        # cold/warm whole-T5 clocks; the receipt still records their duration.
        backend.prepare_inputs()
        rows = ContiguousSequence(backend).run()
        archive_outputs(output, backend, rows, metadata, runtime=backend.runtime_metadata())
    except BaseException as error:
        failed = True
        # CPU-owned completed outputs and a plain failure receipt are safe to
        # retain; do not query, clear, retry, or otherwise touch the GPU here.
        if backend is not None and getattr(backend, "outputs", None):
            try:
                archive_outputs(output, backend, [], metadata, runtime=backend.runtime_metadata(),
                                status="failed", error=error)
            except Exception:
                pass
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                   "GPU_calls_in_handler": 0, "no_retry": True, "source": metadata}
        failure_path = output / "failure.json"
        if not failure_path.exists():
            output.mkdir(mode=0o700, exist_ok=True)
            failure_path.write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        # Models are already CPU-owned after every successful stage.  No GPU
        # operation is performed here, including on an exception path.
        if not failed and backend is not None and backend.active is None:
            backend.models.clear()
            gc.collect()


def root_run_checks(output: Path, fixture: Path, gate_json: Path, bdf: str) -> None:
    require(os.geteuid() == 0, "--run is root-controlled; use --check-metadata or --self-test without root")
    require(bdf == BDF, "this runner is restricted to the reviewed B580 BDF")
    require(output.is_absolute() and output.parent == HERE and not output.exists() and not output.is_symlink(),
            "output must be a new direct child of the owned evidence leaf")
    require(fixture.resolve() == FIXTURE_DIR.resolve(), "fixture path differs from the reviewed UHD fixture")
    require(gate_json.is_absolute() and gate_json.is_file(), "an explicit measured T5 gate authority is required")


def root_report_checks(gate_json: Path, bdf: str) -> None:
    require(os.geteuid() == 0,
            "--report-pre-dit-gate is root-controlled because debugfs VRAM is privileged")
    require(bdf == BDF, "this runner is restricted to the reviewed B580 BDF")
    require(gate_json.is_absolute() and gate_json.is_file(),
            "an explicit measured T5 gate authority is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-metadata", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--report-pre-dit-gate", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--t5-gate-json", type=Path)
    parser.add_argument("--bdf", default=BDF)
    args = parser.parse_args()
    if args.check_metadata:
        print(json.dumps(source_metadata(), indent=2, sort_keys=True))
        return
    if args.self_test:
        print(json.dumps(contract_self_test(), indent=2, sort_keys=True))
        return
    if args.report_pre_dit_gate:
        require(args.t5_gate_json is not None,
                "--report-pre-dit-gate requires --t5-gate-json")
        root_report_checks(args.t5_gate_json, args.bdf)
        authority = T5GateAuthority.from_json(args.t5_gate_json, require_approval=False)
        vram = Path("/sys/kernel/debug/dri") / args.bdf / "tile0" / "vram_mm"
        print(json.dumps(pre_dit_gate_report(vram, authority), indent=2, sort_keys=True))
        return
    require(args.output is not None and args.fixture_dir is not None and args.t5_gate_json is not None,
            "--run requires --output, --fixture-dir, and --t5-gate-json")
    root_run_checks(args.output, args.fixture_dir, args.t5_gate_json, args.bdf)
    run_xpu(args.output, args.t5_gate_json, args.fixture_dir, args.bdf)


if __name__ == "__main__":
    main()
