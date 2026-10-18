"""Lean contiguous-T5 sequence with the qualified public block-clear seam.

The real backend remains the authenticated frozen backend in
``source-t5-contiguous-next-20261012/contiguous.py``.  This module only adds
the explicit admission/clear ordering; it imports no Torch or model at module
import time.  The small protocol is also usable by the stdlib-only fake tests.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
FROZEN_CONTIGUOUS = EVIDENCE / "source-t5-contiguous-next-20261012" / "contiguous.py"
SPEC = EVIDENCE / "temporal-uhd-complete-source-20261012" / "spec.py"
CHILD = EVIDENCE / "temporal-uhd-complete-source-20261012" / "child.py"
BENCHMARK = EVIDENCE / "temporal-uhd-complete-source-20261012" / "benchmark.py"
T1_DRIVER = EVIDENCE / "source-contiguous-t1-20261012" / "driver_v2.py"
BLOCK_CLEAR = EVIDENCE / "torch-n32-block-clear-diagnostic-20261012" / "adapter.py"
QUALIFIED_MOVE = EVIDENCE / "temporal-uhd-n32-rope-reset-20261012" / "adapter.py"
ROPE_RESET = EVIDENCE / "temporal-uhd-n32-rope-reset-20261012" / "rope_reset.py"

# These are source-only pins.  They authenticate the reused helpers without
# reading fixture/model payloads or creating a second backend implementation.
SOURCE_PINS = {
    FROZEN_CONTIGUOUS: "481d3720bbd644e4c14d18af20a07bd98c5f43dc469cd49bca7594da3b50edbc",
    SPEC: "62b0624561ff49e674989533ef29d97cc1e21fd27377e4ab2f32f826536a318f",
    CHILD: "5733e1ed600f885562c96bdcd0ae241b7e149dfa54a76fa487600d0ae6f662ab",
    BENCHMARK: "a23929ce0a8b079c4cfb0c414c6d109fbdc62d323913d9af8137540114da175c",
    T1_DRIVER: "f36433c131ccf21c4a512fab0de079b961364ce6f6d97c4f4b1302bd4cda86c3",
    BLOCK_CLEAR: "644e423621b15caa2b3cfcd0e71d50c4dc511e496dcf937a8db5ef2a79b5ca8b",
    QUALIFIED_MOVE: "4385b5b629ab714d664adcdc112806219fc8d966edef7fd64f25a981d7f0b8b4",
    ROPE_RESET: "ebaea1b765f21e033dccb6f378b1641106d6b4854859a1d41ffe46d03932609c",
}

DEVICE_SIZE = 12_809_404_416
INITIAL_MIN_FREE = 12_400_000_000
INITIAL_MAX_ALLOCATED = 256 * 1024 * 1024
SETUP_MIN_FREE = 8_290_881_536
FIXED_PHASE_GROWTH = 10_435_710_976
MARGIN = 512 * 1024 * 1024
POST_REQUIRED_FREE = FIXED_PHASE_GROWTH + MARGIN
POST_ALLOCATED = 122_705_408
POST_RESERVED = 140_509_184
SNAPSHOT_KEYS = frozenset(("size", "usage", "free", "allocated", "reserved"))
CALLS = ("cold", "warm")
EXPECTED_COLD_MEAN_SHA256 = "cb448dc98de0fe6e5aab8c1191987cab871c05eb0824daacd9eb54511b73b13a"


class ContractError(ValueError):
    """Malformed source, snapshot, payload, or lifecycle contract."""


class GateRefusal(RuntimeError):
    """A measured admission refusal before the next GPU stage/forward."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ContractError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_frozen_sources() -> dict[str, dict[str, Any]]:
    """Verify the exact small source closure; never inspect payload files."""
    result: dict[str, dict[str, Any]] = {}
    for path, expected in SOURCE_PINS.items():
        info = path.lstat()
        require(path.is_file() and not path.is_symlink(), f"source is not regular: {path}")
        actual = digest(path)
        require(actual == expected, f"source pin changed: {path}")
        result[str(path)] = {"bytes": info.st_size, "sha256": actual}
    return result


def validate_snapshot(row: Mapping[str, Any]) -> dict[str, int]:
    """Validate the exact physical plus allocator-I/O snapshot schema."""
    require(isinstance(row, Mapping) and set(row) == SNAPSHOT_KEYS,
            "physical/allocator snapshot schema differs")
    values = {key: row[key] for key in SNAPSHOT_KEYS}
    require(all(type(value) is int for value in values.values()),
            "snapshot counters must be plain integers")
    require(values["size"] == DEVICE_SIZE, "B580 device size differs")
    require(values["size"] > 0 and 0 <= values["usage"] <= values["size"],
            "physical usage is out of range")
    require(values["free"] == values["size"] - values["usage"],
            "physical free does not match size minus usage")
    require(0 <= values["allocated"] <= values["reserved"] <= values["usage"],
            "allocator I/O counters are out of range")
    return values


class FixedAdmission:
    """Cold admission plus explicit warm/setup/post-setup gates."""

    def __init__(self, snapshot: Callable[[str], Mapping[str, Any]]) -> None:
        self.snapshot = snapshot
        self.rows: dict[str, dict[str, int]] = {}

    def _read(self, label: str) -> dict[str, int]:
        row = validate_snapshot(self.snapshot(label))
        self.rows[label] = row
        return row

    def require_initial(self, call: str) -> dict[str, int]:
        row = self._read("initial:" + call)
        if row["free"] < INITIAL_MIN_FREE:
            raise GateRefusal(f"initial free below {INITIAL_MIN_FREE}: {row['free']}")
        if row["allocated"] > INITIAL_MAX_ALLOCATED:
            raise GateRefusal("initial allocator allocation exceeds the existing body gate")
        return row

    def require_warm(self, call: str) -> dict[str, int]:
        """Admit a replay from the persisted post-cold floor, not cold idle."""
        row = self._read("warm_start:" + call)
        if row["free"] < SETUP_MIN_FREE:
            raise GateRefusal(f"warm-start free below {SETUP_MIN_FREE}: {row['free']}")
        if row["allocated"] != 0 or row["reserved"] != 0:
            raise GateRefusal("warm-start gate requires zero allocator live I/O")
        return row

    def require_setup(self, call: str) -> dict[str, int]:
        row = self._read("setup:" + call)
        if row["free"] < SETUP_MIN_FREE:
            raise GateRefusal(f"setup free below {SETUP_MIN_FREE}: {row['free']}")
        if row["allocated"] != 0 or row["reserved"] != 0:
            raise GateRefusal("setup gate requires zero allocator live I/O")
        return row

    def require_post_setup(self, call: str) -> dict[str, int]:
        row = self._read("post_setup:" + call)
        if row["allocated"] <= 0 or row["reserved"] <= 0:
            raise GateRefusal("post-setup allocator live-I/O pair is not positive")
        if row["allocated"] > POST_ALLOCATED or row["reserved"] > POST_RESERVED:
            raise GateRefusal("post-setup allocator live-I/O pair exceeds measured maxima")
        if row["free"] < POST_REQUIRED_FREE:
            raise GateRefusal(
                f"post-setup free below {POST_REQUIRED_FREE}: {row['free']}"
            )
        return row


class PublicSetupClear:
    """One public shallow clear after DiT activation, with no filesystem work."""

    def __init__(self, clear_memory: Callable[[Any, str], None], synchronize: Callable[[], None],
                 debug: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self.clear_memory = clear_memory
        self.synchronize = synchronize
        self.debug = debug
        self.clock = clock

    def run(self, call: str) -> dict[str, Any]:
        started = self.clock()
        self.clear_memory(self.debug, "setup")
        self.synchronize()
        return {"call": call, "seconds": self.clock() - started, "mode": "public_shallow"}


class BlockClearAdapter:
    """Adapt the frozen block-clear adapter without copying its backend."""

    def __init__(self, qualified_adapter: Any, clear_memory: Callable[[Any, str], None],
                 synchronize: Callable[[], None], clock: Callable[[], float] = time.monotonic) -> None:
        self.qualified_adapter = qualified_adapter
        self.clear_memory = clear_memory
        self.synchronize = synchronize
        self.clock = clock
        self.configured = False
        self.current_call: str | None = None
        self.next_index = 0
        self.records: dict[str, dict[str, Any]] = {}

    def configure_prefix(self, model: Any, count: int, device: Any, offload: Any,
                         debug: Any, configure: Any = None, wrap: Any = None) -> Any:
        require(not self.configured, "block-clear adapter configured more than once")
        self.configured = True

        def before_boundary_sync() -> None:
            self.synchronize()

        def boundary(index: int) -> None:
            require(self.current_call is not None, "block clear outside a forward")
            require(index == self.next_index, "block-clear boundary order differs")
            started = self.clock()
            self.clear_memory(debug, f"n32_block_{index:02d}")
            self.synchronize()
            self.next_index += 1
            self.records[self.current_call]["clear_seconds"] += self.clock() - started
            self.records[self.current_call]["count"] += 1

        return self.qualified_adapter.configure_prefix(
            model, count, device, offload, debug,
            variant="block-clear",
            synchronize=before_boundary_sync,
            boundary=boundary,
            configure=configure,
            wrap=wrap,
        )

    def begin_call(self, call: str) -> None:
        require(self.current_call is None, "block-clear call overlap")
        self.current_call = call
        self.next_index = 0
        self.records[call] = {"count": 0, "clear_seconds": 0.0}

    def end_call(self) -> dict[str, Any]:
        require(self.current_call is not None, "block-clear call missing")
        call = self.current_call
        record = dict(self.records[call])
        require(record["count"] == 32 and self.next_index == 32,
                "exactly 32 block clears are required")
        self.current_call = None
        return record


def make_frozen_backend(frozen_backend: Any, spec: Any, ref: Any, reset: Any,
                        gate: Any, bdf: str, qualified_adapter: Any,
                        clear_memory: Callable[[Any, str], None],
                        synchronize: Callable[[], None]) -> tuple[Any, PublicSetupClear, BlockClearAdapter]:
    """Wire the frozen backend to this adapter without importing its runtime here."""
    verify_frozen_sources()
    factory = getattr(frozen_backend, "_torch_backend", None)
    require(callable(factory), "frozen backend factory is unavailable")
    block_clear = BlockClearAdapter(qualified_adapter, clear_memory, synchronize)
    backend = factory(spec, ref, block_clear, reset, gate, bdf)
    setup_clear = PublicSetupClear(clear_memory, synchronize, getattr(backend, "debug", None))
    return backend, setup_clear, block_clear


def _payload_matches(left: Any, right: Any) -> bool:
    if hasattr(left, "tobytes") and hasattr(right, "tobytes"):
        return left.tobytes(order="C") == right.tobytes(order="C")
    return left == right


def payload_sha256(payload: Any) -> str:
    """Hash an owned contiguous payload without accepting an opaque proxy."""
    tobytes = getattr(payload, "tobytes", None)
    require(callable(tobytes), "mean payload does not expose tobytes")
    raw = tobytes(order="C")
    require(isinstance(raw, (bytes, bytearray, memoryview)),
            "mean payload tobytes did not return bytes")
    return hashlib.sha256(bytes(raw)).hexdigest()


class ContiguousClearedSequence:
    """Explicit cold/warm sequence; no implicit gate or cleanup side effects."""

    def __init__(self, backend: Any, admission: FixedAdmission, setup_clear: PublicSetupClear,
                 block_clear: BlockClearAdapter, archive: Callable[[str, Mapping[Any, Any]], Any],
                 expected_mean_sha256: str = EXPECTED_COLD_MEAN_SHA256) -> None:
        self.backend = backend
        self.admission = admission
        self.setup_clear = setup_clear
        self.block_clear = block_clear
        self.archive = archive
        require(
            isinstance(expected_mean_sha256, str)
            and len(expected_mean_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_mean_sha256),
            "mean authority digest must be lowercase SHA-256",
        )
        self.expected_mean_sha256 = expected_mean_sha256
        # Keep completed rows reachable if a later cold/warm gate raises before
        # run() can return them to the caller's receipt writer.
        self.completed_calls: list[dict[str, Any]] = []

    def run(self, calls: tuple[str, ...] = CALLS) -> list[dict[str, Any]]:
        require(calls == CALLS, "contiguous cleared runner requires cold then warm")
        require(callable(self.archive), "archive callback is required")
        # Inputs are loaded before the whole-call clock starts.
        self.backend.prepare_inputs()
        rows: list[dict[str, Any]] = []
        self.completed_calls = rows
        for call in calls:
            started = self.backend.now()
            row: dict[str, Any] = {"call": call}

            if call == "cold":
                row["initial_gate"] = self.admission.require_initial(call)
            else:
                row["warm_start_gate"] = self.admission.require_warm(call)
            self.backend.activate("vae")
            mean = self.backend.encode(call)
            self.backend.preserve(call, "mean", mean)
            self.backend.validate("mean", mean)
            require(
                payload_sha256(mean) == self.expected_mean_sha256,
                f"{call} mean digest differs from authenticated authority",
            )
            self.backend.release("vae")

            row["setup_gate"] = self.admission.require_setup(call)
            self.backend.activate("dit")
            row["setup_clear"] = self.setup_clear.run(call)
            row["post_setup_gate"] = self.admission.require_post_setup(call)

            self.block_clear.begin_call(call)
            prediction = self.backend.dit(call, mean)
            row["block_clear"] = self.block_clear.end_call()
            self.backend.preserve(call, "prediction", prediction)
            self.backend.validate("prediction", prediction)
            self.backend.release("dit")

            self.backend.activate("vae")
            rgb = self.backend.decode(call, prediction)
            self.backend.preserve(call, "rgb", rgb)
            self.backend.validate("rgb", rgb)
            self.backend.release("vae")

            row["whole_t5_seconds"] = self.backend.now() - started
            # Persistence/hash/image work is deliberately outside this clock.
            row["archive"] = self.archive(call, self.backend.outputs)
            self.backend.finish_call(call, rgb)
            rows.append(row)

        outputs = getattr(self.backend, "outputs", {})
        for stage in ("mean", "prediction", "rgb"):
            first = outputs.get(("cold", stage))
            second = outputs.get(("warm", stage))
            require(first is not None and second is not None, f"missing cold/warm {stage}")
            require(_payload_matches(first, second), f"cold/warm {stage} differs")
        return rows


@dataclass(frozen=True)
class Synthetic:
    name: str
    shape: tuple[int, ...]
    dtype: str
    finite: bool = True

    def copy(self) -> "Synthetic":
        return Synthetic(self.name, self.shape, self.dtype, self.finite)

    def tobytes(self, order: str = "C") -> bytes:
        require(order == "C", "synthetic payload only supports C order")
        return repr((self.name, self.shape, self.dtype, self.finite)).encode("utf-8")


EXPECTED_SHAPES = {
    "mean": (1, 16, 2, 270, 480),
    "prediction": (2, 270, 480, 16),
    "rgb": (5, 2160, 3840, 3),
}
EXPECTED_DTYPES = {"mean": "float32", "prediction": "float32", "rgb": "uint8"}


def validate_payload(stage: str, payload: Any) -> None:
    # Synthetic + this shape/dtype helper are test-only.  The real backend
    # supplies validation from the authenticated frozen runtime and never
    # calls this helper.
    require(stage in EXPECTED_SHAPES, f"unknown payload stage: {stage}")
    require(tuple(getattr(payload, "shape", ())) == EXPECTED_SHAPES[stage],
            f"{stage} shape differs")
    require(str(getattr(payload, "dtype", "")) == EXPECTED_DTYPES[stage],
            f"{stage} dtype differs")
    require(getattr(payload, "finite", False) is True, f"{stage} is nonfinite")
