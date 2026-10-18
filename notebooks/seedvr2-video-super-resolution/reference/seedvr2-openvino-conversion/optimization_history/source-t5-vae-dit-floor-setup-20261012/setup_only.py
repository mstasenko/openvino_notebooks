#!/usr/bin/env python3
"""Setup-only VAE-to-N32-DiT floor sequence.

This module reuses the frozen contiguous and VAE-owner-drop seams.  It is
import-safe: importing it authenticates only the two small source files and
does not import Torch, Core, a model, a fixture, or a device.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Mapping


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
BASE_PATH = EVIDENCE / "source-t5-contiguous-next-20261012" / "contiguous.py"
DROP_PATH = EVIDENCE / "source-t5-vae-owner-drop-diagnostic-20261012" / "drop_reload.py"
BASE_SHA256 = "481d3720bbd644e4c14d18af20a07bd98c5f43dc469cd49bca7594da3b50edbc"
DROP_SHA256 = "1506886aef1015ec84ec485f74b9cc8f19f25f6916266ad797e92c27290cd500"

SETUP_GROWTH_BYTES = 7_217_139_712
SETUP_MARGIN_BYTES = 1_073_741_824
SETUP_MIN_FREE_BYTES = SETUP_GROWTH_BYTES + SETUP_MARGIN_BYTES
MEAN_SHA256 = "cb448dc98de0fe6e5aab8c1191987cab871c05eb0824daacd9eb54511b73b13a"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _authenticate(path: Path, expected: str) -> tuple[Any, str]:
    """Authenticate a frozen source before any module code can execute."""

    info = path.lstat()
    if not path.is_file() or path.is_symlink() or info.st_nlink != 1:
        raise RuntimeError(f"frozen source is not regular/single-link: {path}")
    actual = _digest(path)
    if actual != expected:
        raise RuntimeError(f"frozen source pin changed: {path}")
    return info, actual


def _load(path: Path, name: str, expected: str) -> Any:
    # The digest/stat check must precede spec.loader.exec_module: this module
    # is itself a trusted source seam and must not execute from stale bytes.
    _authenticate(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DROP = _load(DROP_PATH, "setup_only_frozen_drop_reload", DROP_SHA256)
BASE = DROP.BASE
require = BASE.require
ContractError = BASE.ContractError
GateRefusal = BASE.GateRefusal
DropReloadBackend = DROP.DropReloadBackend


def source_metadata() -> dict[str, Any]:
    rows = {}
    for path, expected in ((BASE_PATH, BASE_SHA256), (DROP_PATH, DROP_SHA256)):
        info, actual = _authenticate(path, expected)
        rows[str(path)] = {"bytes": info.st_size, "sha256": actual}
    return {
        "status": "PASS_SOURCE_ONLY",
        "source_files": rows,
        "setup_growth_bytes": SETUP_GROWTH_BYTES,
        "setup_margin_bytes": SETUP_MARGIN_BYTES,
        "setup_min_free_bytes": SETUP_MIN_FREE_BYTES,
        "mean_sha256": MEAN_SHA256,
        "torch_imports": 0,
        "gpu_queries": 0,
        "payload_reads": 0,
    }


def _payload_bytes(payload: Any) -> bytes:
    tobytes = getattr(payload, "tobytes", None)
    require(callable(tobytes), "mean payload has no owned byte representation")
    try:
        value = tobytes(order="C")
    except TypeError:
        value = tobytes()
    require(isinstance(value, (bytes, bytearray)), "mean payload bytes are not concrete")
    return bytes(value)


def _snapshot(backend: Any, label: str) -> dict[str, Any]:
    sampler = getattr(backend, "sample_physical", None)
    if callable(sampler):
        physical = dict(sampler(label))
    else:
        gate = getattr(backend, "gate", None)
        require(gate is not None and callable(getattr(gate, "sample", None)),
                "backend has no filesystem physical sampler")
        physical = dict(gate.sample())
    require(set(physical) == {"size", "usage", "free"},
            "physical snapshot schema differs")
    require(all(type(physical[key]) is int and physical[key] >= 0
                for key in ("size", "usage", "free")),
            "physical snapshot counters are not non-negative integers")
    require(physical["usage"] <= physical["size"] and
            physical["free"] == physical["size"] - physical["usage"],
            "physical snapshot is inconsistent")
    allocator = dict(backend.allocator())
    require(all(type(value) is int and value >= 0 for value in allocator.values()),
            "allocator counters are not non-negative integers")
    return {"label": label, "physical": physical, "allocator": allocator}


def _cpu_quiesce(backend: Any) -> None:
    """Release CPU-only references after the final GPU observation."""

    models = getattr(backend, "models", None)
    if isinstance(models, dict):
        models.clear()
    gc.collect()


class SetupOnlySequence:
    """One encode, one DiT setup, no DiT forward, and explicit release proof."""

    def __init__(
        self,
        backend: Any,
        dropper: Any,
        setup_clear: Callable[[], None],
        *,
        expected_mean_sha256: str | None = MEAN_SHA256,
        snapshot: Callable[[Any, str], dict[str, Any]] = _snapshot,
        quiesce: Callable[[Any], None] = _cpu_quiesce,
        archive_mean: Callable[[Any], Any] | None = None,
        minimum_free_bytes: int = SETUP_MIN_FREE_BYTES,
    ) -> None:
        require(callable(getattr(backend, "activate", None)), "backend activate seam missing")
        require(callable(getattr(backend, "encode", None)), "backend encode seam missing")
        require(callable(getattr(backend, "preserve", None)), "backend preserve seam missing")
        require(callable(getattr(backend, "validate", None)), "backend validate seam missing")
        require(callable(getattr(backend, "release", None)), "backend release seam missing")
        require(callable(getattr(backend, "sync", None)), "backend sync seam missing")
        require(callable(getattr(dropper, "drop_cpu_vae_owner", None)),
                "VAE owner-drop seam missing")
        require(callable(setup_clear), "public shallow setup clear seam missing")
        require(callable(snapshot) and callable(quiesce), "observation seam missing")
        require(type(minimum_free_bytes) is int and minimum_free_bytes > 0,
                "setup minimum free bound is malformed")
        self.backend = backend
        self.dropper = dropper
        self.setup_clear = setup_clear
        self.expected_mean_sha256 = expected_mean_sha256
        self.snapshot = snapshot
        self.quiesce = quiesce
        self.archive_mean = archive_mean
        self.minimum_free_bytes = minimum_free_bytes

    def run(self) -> dict[str, Any]:
        backend = self.backend
        snapshots: dict[str, dict[str, Any]] = {}
        snapshots["idle"] = self.snapshot(backend, "idle")

        backend.activate("vae")
        mean = backend.encode("setup")
        backend.preserve("setup", "mean", mean)
        backend.validate("mean", mean)
        mean_sha256 = hashlib.sha256(_payload_bytes(mean)).hexdigest()
        if self.expected_mean_sha256 is not None:
            require(mean_sha256 == self.expected_mean_sha256,
                    "mean bytes differ from the authenticated contiguous source")
        mean_archive = None
        if self.archive_mean is not None:
            mean_archive = self.archive_mean(mean)
        backend.release("vae")
        snapshots["after_vae_release"] = self.snapshot(backend, "after_vae_release")

        self.dropper.drop_cpu_vae_owner("setup")
        snapshots["after_vae_drop"] = self.snapshot(backend, "after_vae_drop")
        free = snapshots["after_vae_drop"]["physical"]["free"]
        if free < self.minimum_free_bytes:
            raise GateRefusal(
                f"setup-only DiT admission denied: free={free} "
                f"required={self.minimum_free_bytes}"
            )

        backend.activate("dit")
        # activate("dit") already synchronizes the frozen backend; this second
        # explicit sync makes the before-clear observation boundary testable.
        backend.sync()
        snapshots["dit_setup_before_clear"] = self.snapshot(backend, "dit_setup_before_clear")

        # The caller binds this to the existing public clear_memory(force=True,
        # deep=False) function.  It is intentionally separate from release,
        # whose frozen path performs the ordinary deep clear.
        self.setup_clear()
        backend.sync()
        snapshots["dit_setup_after_clear"] = self.snapshot(backend, "dit_setup_after_clear")

        # There is deliberately no backend.dit(...) call in this sequence.
        backend.release("dit")
        snapshots["after_dit_release"] = self.snapshot(backend, "after_dit_release")
        self.quiesce(backend)
        snapshots["final"] = self.snapshot(backend, "final")
        return {
            "status": "PASS_SETUP_ONLY",
            "mean_sha256": mean_sha256,
            "mean_archive": mean_archive,
            "mean_preserved_before_vae_release": True,
            "vae_encode_calls": 1,
            "dit_setup_calls": 1,
            "dit_forward_calls": 0,
            "setup_gate": {
                "minimum_free_bytes": self.minimum_free_bytes,
                "decision": "PASS",
            },
            "snapshots": snapshots,
            "old_body_gate_invoked": False,
            "retry": False,
        }


if __name__ == "__main__":
    print(json.dumps(source_metadata(), indent=2, sort_keys=True))
