#!/usr/bin/env python3
"""Narrow VAE-owner drop/reload overlay for the frozen contiguous T5 source.

This module does not copy or edit the contiguous runner.  It loads that frozen
source for its contracts, adds only the lifetime seam, and remains import-safe:
the real backend helper is imported only if a root-controlled runtime wraps a
real backend.  No model, fixture payload, Torch, Core, or device is touched by
source metadata or CPU contract tests.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
BASE_PATH = EVIDENCE / "source-t5-contiguous-next-20261012" / "contiguous.py"
N32_DRIVER_PATH = EVIDENCE / "temporal-uhd-n32-replay-20261012" / "driver.py"
BASE_SHA256 = "481d3720bbd644e4c14d18af20a07bd98c5f43dc469cd49bca7594da3b50edbc"
N32_DRIVER_SHA256 = "980c75485f78dacf0bb754f8c26806e72c06ac75b3258630dc6fd0b3c2de219d"


def _assert_base_pin() -> None:
    """Authenticate the contiguous source before executing its module code."""

    info = BASE_PATH.lstat()
    if not BASE_PATH.is_file() or BASE_PATH.is_symlink() or info.st_nlink != 1:
        raise RuntimeError(f"frozen base is not regular/single-link: {BASE_PATH}")
    with BASE_PATH.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != BASE_SHA256:
        raise RuntimeError(f"frozen base pin changed: {BASE_PATH}")


def _load_base() -> Any:
    _assert_base_pin()
    loader = importlib.util.spec_from_file_location("frozen_contiguous_t5", BASE_PATH)
    if loader is None or loader.loader is None:
        raise RuntimeError(f"cannot load frozen contiguous source: {BASE_PATH}")
    module = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = module
    loader.loader.exec_module(module)
    return module


BASE = _load_base()
ContractError = BASE.ContractError
GateRefusal = BASE.GateRefusal
CALLS = BASE.CALLS
require = BASE.require


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_metadata() -> dict[str, Any]:
    """Pin the frozen contiguous seam and the actual ordinary N32 driver only."""

    rows: dict[str, Any] = {}
    for path, expected in ((BASE_PATH, BASE_SHA256), (N32_DRIVER_PATH, N32_DRIVER_SHA256)):
        info = path.lstat()
        require(path.is_file() and not path.is_symlink(), f"source is not regular: {path}")
        actual = _digest(path)
        require(actual == expected, f"source pin changed: {path}")
        rows[str(path)] = {"bytes": info.st_size, "sha256": actual}
    return {
        "status": "PASS_SOURCE_ONLY",
        "source_files": rows,
        "selected_n32_driver": str(N32_DRIVER_PATH),
        "lifetime": "preserve_mean_then_drop_cpu_vae_before_t5_gate_reload_after_dit",
        "physical_gate": "unchanged_existing_t5_authority",
        "payload_reads": 0,
        "torch_imports": 0,
        "gpu_queries": 0,
    }


class DropReloadBackend:
    """Wrap the frozen real backend without changing its math or phase methods."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._pending_reload: str | None = None
        self._last_reload: dict[str, Any] | None = None
        self.drop_rows: list[dict[str, Any]] = []
        self.reload_rows: list[dict[str, Any]] = []
        self.pre_gate_rows: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    @staticmethod
    def _device_name(value: Any) -> str:
        return str(getattr(value, "type", value))

    def _assert_cpu_nonmeta(self, model: Any) -> None:
        tensors = [*model.parameters(), *model.buffers()]
        require(tensors, "VAE owner has no parameter/buffer tensors")
        for tensor in tensors:
            is_meta = getattr(tensor, "is_meta", False)
            is_meta = is_meta() if callable(is_meta) else is_meta
            require(not bool(is_meta), "VAE owner retains a meta tensor")
            require(self._device_name(getattr(tensor, "device", "")) == "cpu",
                    "VAE owner was not CPU-resident before drop")

    def _public_deep_clear_and_sync(self) -> None:
        """Use the reviewed public clear helper after deleting the CPU owner."""

        source = Path(self.backend.ref.SOURCE)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from src.optimization.memory_manager import clear_memory

        clear_memory(deep=True, force=True)
        self.backend.torch.xpu.synchronize(self.backend.device)

    def drop_cpu_vae_owner(self, call: str) -> dict[str, Any]:
        """Drop only after mean preservation and the frozen VAE release path."""

        require(self.backend.active is None, "cannot drop VAE while a stage is active")
        model = self.backend.models.get("vae")
        require(model is not None, "VAE owner is already absent")
        self._assert_cpu_nonmeta(model)
        started = self.backend.now()
        self.backend.models["vae"] = None
        del model
        gc.collect()
        self._public_deep_clear_and_sync()
        stats = self.backend.allocator()
        require(stats.get("memory_allocated") == 0 and stats.get("memory_reserved") == 0,
                "GPU allocator remains live after VAE owner drop")
        row = {
            "call": call,
            "seconds": self.backend.now() - started,
            "allocator": dict(stats),
            "owner_dropped": True,
            "retry": False,
        }
        self.drop_rows.append(row)
        self._pending_reload = call
        return row

    def sample_before_t5_gate(self, call: str) -> Mapping[str, int]:
        """Record a physical sample before the unchanged authority gate."""

        row = dict(self.backend.gate.sample())
        row["call"] = call
        row["after_vae_owner_drop"] = True
        self.pre_gate_rows.append(row)
        return row

    def activate(self, role: str) -> None:
        started = self.backend.now()
        self.backend.activate(role)
        if role == "vae" and self._pending_reload is not None:
            call = self._pending_reload
            self._last_reload = {
                "call": call,
                "seconds": self.backend.now() - started,
                "owner_recreated": True,
                "bf16_loader": True,
            }
            self.reload_rows.append(self._last_reload)
            self._pending_reload = None

    def consume_reload(self, call: str) -> Mapping[str, Any]:
        row = self._last_reload
        require(isinstance(row, dict) and row.get("call") == call and
                row.get("owner_recreated") is True and row.get("bf16_loader") is True,
                "decode VAE was not recreated after DiT release")
        self._last_reload = None
        return row


class DropReloadSequence:
    """The frozen contiguous sequence with one explicit lifetime seam."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def run(self, calls: tuple[str, ...] = CALLS) -> list[dict[str, Any]]:
        require(calls == CALLS, "drop/reload runner requires exactly cold then warm calls")
        rows: list[dict[str, Any]] = []
        for call in calls:
            started = self.backend.now()
            row: dict[str, Any] = {"call": call, "lifetime_variant": "drop_reload_vae"}

            stage_started = self.backend.now()
            self.backend.activate("vae")
            mean = self.backend.encode(call)
            # The handoff is deliberately before owner deletion so refusal or
            # deletion failure cannot erase the only completed encoder artifact.
            self.backend.preserve(call, "mean", mean)
            self.backend.validate("mean", mean)
            self.backend.release("vae")
            row["encode_move_infer_release_seconds"] = self.backend.now() - stage_started
            row["vae_drop"] = self.backend.drop_cpu_vae_owner(call)
            row["pre_dit_physical"] = self.backend.sample_before_t5_gate(call)

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
            row["vae_reload"] = self.backend.consume_reload(call)
            rgb = self.backend.decode(call, prediction)
            # Preserve owned CPU RGB before late validation, matching the frozen
            # contiguous failure-evidence rule.
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
            require(first is not None and second is not None,
                    f"missing cold/warm {stage} output")
            if hasattr(first, "tobytes") and hasattr(second, "tobytes"):
                require(first.tobytes(order="C") == second.tobytes(order="C"),
                        f"cold/warm {stage} output differs")
            else:
                require(first == second, f"cold/warm {stage} output differs")
        return rows


def contract_self_test() -> dict[str, Any]:
    """Keep a root-safe entry point without constructing a real backend."""

    return {"status": "SOURCE_ONLY", "source": source_metadata(),
            "sequence": "DropReloadSequence", "GPU_execution": False}


if __name__ == "__main__":
    print(contract_self_test())
