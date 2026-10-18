"""Stdlib-only contract tests for the contiguous cleared sequence."""

import unittest

from cleared_adapter import (
    BLOCK_CLEAR,
    CALLS,
    DEVICE_SIZE,
    EXPECTED_COLD_MEAN_SHA256,
    INITIAL_MAX_ALLOCATED,
    SETUP_MIN_FREE,
    FixedAdmission,
    BlockClearAdapter,
    ContractError,
    GateRefusal,
    POST_ALLOCATED,
    POST_RESERVED,
    PublicSetupClear,
    Synthetic,
    ContiguousClearedSequence,
    make_frozen_backend,
    payload_sha256,
    validate_payload,
    verify_frozen_sources,
)


SETUP_USAGE = 1_164_312_576
SETUP_FREE = DEVICE_SIZE - SETUP_USAGE
INITIAL_USAGE = 25_178_112
INITIAL_FREE = DEVICE_SIZE - INITIAL_USAGE
POST_USAGE = 1_474_707_456
WARM_USAGE = 1_346_781_184


def row(label, *, usage=None, allocated=None, reserved=None, size=DEVICE_SIZE):
    if usage is None:
        if label.startswith("initial:"):
            usage = INITIAL_USAGE
        elif label.startswith("warm_start:"):
            # The cold call leaves this measured floor in the same process;
            # warm replay must not pretend that cold-idle memory returned.
            usage = WARM_USAGE
        elif label.startswith("setup:"):
            usage = SETUP_USAGE
        else:
            usage = POST_USAGE
    if allocated is None:
        allocated = 0 if label.startswith(("initial:", "warm_start:", "setup:")) else POST_ALLOCATED
    if reserved is None:
        reserved = 0 if label.startswith(("initial:", "warm_start:", "setup:")) else POST_RESERVED
    return {
        "size": size,
        "usage": usage,
        "free": size - usage,
        "allocated": allocated,
        "reserved": reserved,
    }


class FakeBlockClear:
    def __init__(self):
        self.current = None
        self.index = 0
        self.records = {}
        self.events = []

    def begin_call(self, call):
        if self.current is not None:
            raise ContractError("fake block-clear calls overlap")
        self.current = call
        self.index = 0
        self.records[call] = {"count": 0, "clear_seconds": 0.0}
        self.events.append("block:begin:" + call)

    def fire(self, index):
        if self.current is None or index != self.index:
            raise ContractError("fake block-clear index differs")
        self.index += 1
        self.records[self.current]["count"] += 1
        self.events.append(f"block:{self.current}:{index:02d}")

    def end_call(self):
        if self.current is None or self.index != 32:
            raise ContractError("fake block-clear count differs")
        call = self.current
        self.current = None
        self.events.append("block:end:" + call)
        return dict(self.records[call])


class FakeBackend:
    def __init__(self, *, bad_stage=None):
        self.bad_stage = bad_stage
        self.active = None
        self.events = []
        self.outputs = {}
        self.models = {"vae": object(), "dit": object()}
        self.debug = object()
        self.activate_count = {"vae": 0, "dit": 0}
        self.block_clear = None
        self.ticks = 0.0

    def prepare_inputs(self):
        self.events.append("prepare_inputs")

    def now(self):
        self.ticks += 0.01
        return self.ticks

    def activate(self, role):
        if self.active is not None:
            raise ContractError("fake stage overlap")
        self.active = role
        self.activate_count[role] += 1
        self.events.append("activate:" + role)

    def encode(self, call):
        if self.active != "vae":
            raise ContractError("fake encode owner differs")
        self.events.append("encode:" + call)
        return Synthetic("mean", (1, 16, 2, 270, 480), "float32")

    def dit(self, call, _mean):
        if self.active != "dit":
            raise ContractError("fake DiT owner differs")
        self.events.append("dit:" + call)
        if self.block_clear is not None:
            for index in range(32):
                self.block_clear.fire(index)
        finite = self.bad_stage != "prediction_nonfinite"
        return Synthetic("prediction", (2, 270, 480, 16), "float32", finite)

    def decode(self, call, _prediction):
        if self.active != "vae":
            raise ContractError("fake decode owner differs")
        self.events.append("decode:" + call)
        shape = (5, 2160, 3840, 2) if self.bad_stage == "rgb_shape" else (5, 2160, 3840, 3)
        finite = self.bad_stage != "rgb_nonfinite"
        return Synthetic("rgb", shape, "uint8", finite)

    def preserve(self, call, stage, payload):
        self.events.append("preserve:" + stage)
        self.outputs[(call, stage)] = payload.copy()

    def validate(self, stage, payload):
        self.events.append("validate:" + stage)
        validate_payload(stage, payload)

    def release(self, role):
        if self.active != role:
            raise ContractError("fake release owner differs")
        self.events.append("release:" + role)
        self.active = None

    def finish_call(self, call, _rgb):
        self.events.append("finish:" + call)


def make_sequence(*, bad_stage=None, snapshots=None, expected_mean_sha256=None):
    backend = FakeBackend(bad_stage=bad_stage)
    vae_owner = backend.models["vae"]
    block_clear = FakeBlockClear()
    backend.block_clear = block_clear
    snapshots = snapshots or {}
    gate_rows = []

    def read(label):
        gate_rows.append(label)
        backend.events.append("gate:" + label)
        return snapshots.get(label, row(label))

    admission = FixedAdmission(read)
    clear_events = []
    sync_events = []

    def clear(_debug, name):
        clear_events.append(name)

    def synchronize():
        sync_events.append("sync")

    setup_clear = PublicSetupClear(clear, synchronize, debug=object(), clock=backend.now)
    archives = []

    def archive(call, _outputs):
        archives.append(call)
        backend.events.append("archive:" + call)
        return {"status": "outside_whole_clock"}

    fake_mean = Synthetic("mean", (1, 16, 2, 270, 480), "float32")
    if expected_mean_sha256 is None:
        expected_mean_sha256 = payload_sha256(fake_mean)
    sequence = ContiguousClearedSequence(
        backend, admission, setup_clear, block_clear, archive,
        expected_mean_sha256=expected_mean_sha256,
    )
    return sequence, backend, vae_owner, block_clear, clear_events, sync_events, archives, gate_rows


class QualifiedStub:
    def __init__(self):
        self.kwargs = None

    def configure_prefix(self, *args, **kwargs):
        self.kwargs = kwargs
        return {"variant": kwargs["variant"], "selected_blocks": list(range(args[1]))}


class FrozenBackendStub:
    def __init__(self):
        self.received = None

    def _torch_backend(self, spec, ref, block_clear, reset, gate, bdf):
        self.received = (spec, ref, block_clear, reset, gate, bdf)
        return FakeBackend()


class ClearedSequenceTests(unittest.TestCase):
    def test_frozen_source_closure_is_exact(self):
        metadata = verify_frozen_sources()
        self.assertEqual(len(metadata), 8)
        self.assertTrue(str(BLOCK_CLEAR) in metadata)

    def test_cold_warm_order_gates_and_replay(self):
        sequence, backend, vae_owner, block_clear, clear_events, sync_events, archives, gate_rows = make_sequence()
        rows = sequence.run()

        self.assertEqual([item["call"] for item in rows], list(CALLS))
        self.assertTrue(all(item["whole_t5_seconds"] > 0 for item in rows))
        self.assertEqual(backend.activate_count, {"vae": 4, "dit": 2})
        self.assertIs(backend.models["vae"], vae_owner)
        self.assertEqual(archives, ["cold", "warm"])
        self.assertEqual(gate_rows, [
            "initial:cold", "setup:cold", "post_setup:cold",
            "warm_start:warm", "setup:warm", "post_setup:warm",
        ])
        self.assertEqual([block_clear.records[name]["count"] for name in CALLS], [32, 32])
        self.assertEqual(clear_events.count("setup"), 2)
        self.assertEqual(len([name for name in block_clear.events if name.startswith("block:")]), 68)
        self.assertEqual(len(sync_events), 2)

        events = backend.events
        self.assertLess(events.index("gate:setup:cold") if "gate:setup:cold" in events else -1,
                        events.index("activate:dit"))
        # Admission events are kept by the fake reader; the sequence itself
        # cannot perform a forward until the post-setup callback returns.
        self.assertLess(events.index("activate:dit"), events.index("dit:cold"))
        self.assertEqual(events[-2:], ["archive:warm", "finish:warm"])
        self.assertEqual(backend.outputs[("cold", "rgb")], backend.outputs[("warm", "rgb")])

    def test_warm_floor_persists_and_warm_gate_does_not_repeat_cold_gate(self):
        sequence, backend, _owner, _block, _clear, _sync, archives, gate_rows = make_sequence()
        sequence.run()
        warm_row = sequence.admission.rows["warm_start:warm"]
        self.assertEqual(warm_row["usage"], WARM_USAGE)
        self.assertEqual(warm_row["free"], DEVICE_SIZE - WARM_USAGE)
        self.assertEqual(warm_row["allocated"], 0)
        self.assertEqual(warm_row["reserved"], 0)
        self.assertNotIn("initial:warm", gate_rows)
        self.assertNotIn("gate:initial:warm", backend.events)

    def test_warm_gate_low_free_refuses_before_warm_vae_or_forward(self):
        low_warm = row("warm_start:warm", usage=DEVICE_SIZE - SETUP_MIN_FREE + 1)
        sequence, backend, _owner, _block, _clear, _sync, archives, _labels = make_sequence(
            snapshots={"warm_start:warm": low_warm}
        )
        with self.assertRaises(GateRefusal):
            sequence.run()
        self.assertEqual(archives, ["cold"])
        self.assertEqual(backend.events.count("activate:vae"), 2)
        self.assertNotIn("encode:warm", backend.events)
        self.assertNotIn("dit:warm", backend.events)
        self.assertNotIn("decode:warm", backend.events)

    def test_mean_digest_is_explicit_and_checked_after_preservation(self):
        payload = Synthetic("mean", (1, 16, 2, 270, 480), "float32")
        wrong_digest = EXPECTED_COLD_MEAN_SHA256
        if payload_sha256(payload) == EXPECTED_COLD_MEAN_SHA256:
            self.fail("synthetic digest unexpectedly equals authenticated runtime digest")
        sequence, backend, _owner, _block, _clear, _sync, archives, _labels = make_sequence(
            expected_mean_sha256=wrong_digest
        )
        with self.assertRaises(ContractError):
            sequence.run()
        self.assertIn(("cold", "mean"), backend.outputs)
        self.assertEqual(archives, [])

    def test_initial_gate_rejects_before_vae_activation(self):
        low_initial = row("initial:cold", usage=DEVICE_SIZE - 12_400_000_000 + 1)
        sequence, backend, _owner, _block, _clear, _sync, archives, _labels = make_sequence(
            snapshots={"initial:cold": low_initial}
        )
        with self.assertRaises(GateRefusal):
            sequence.run()
        self.assertNotIn("activate:vae", backend.events)
        self.assertEqual(archives, [])

    def test_initial_allocator_body_limit_is_preserved(self):
        too_allocated = row("initial:cold", usage=INITIAL_MAX_ALLOCATED + 1,
                            allocated=INITIAL_MAX_ALLOCATED + 1,
                            reserved=INITIAL_MAX_ALLOCATED + 1)
        sequence, backend, _owner, _block, _clear, _sync, archives, _labels = make_sequence(
            snapshots={"initial:cold": too_allocated}
        )
        with self.assertRaises(GateRefusal):
            sequence.run()
        self.assertNotIn("activate:vae", backend.events)
        self.assertEqual(archives, [])

    def test_setup_gate_rejects_before_dit_activation(self):
        low_setup = row("setup:cold", usage=DEVICE_SIZE - 8_290_881_535)
        sequence, backend, _owner, _block, _clear, _sync, archives, _labels = make_sequence(
            snapshots={"setup:cold": low_setup}
        )
        with self.assertRaises(GateRefusal):
            sequence.run()
        self.assertNotIn("activate:dit", backend.events)
        self.assertEqual(archives, [])

    def test_post_setup_gate_rejects_without_forward_or_cleanup(self):
        low_post = row("post_setup:cold", usage=POST_USAGE + 362_115_073,
                       allocated=POST_ALLOCATED, reserved=POST_RESERVED)
        sequence, backend, _owner, block_clear, _clear, _sync, archives, _labels = make_sequence(
            snapshots={"post_setup:cold": low_post}
        )
        with self.assertRaises(GateRefusal):
            sequence.run()
        self.assertNotIn("dit:cold", backend.events)
        self.assertNotIn("release:dit", backend.events)
        self.assertEqual(block_clear.records, {})
        self.assertEqual(archives, [])

    def test_post_setup_accepts_observed_warm_lower_positive_pair(self):
        observed_warm = row(
            "post_setup:warm",
            usage=WARM_USAGE,
            allocated=122_443_264,
            reserved=130_023_424,
        )
        admitted = FixedAdmission(lambda _label: observed_warm).require_post_setup("warm")
        self.assertEqual(admitted, observed_warm)

    def test_post_setup_rejects_allocator_pair_above_measured_maxima(self):
        for allocated, reserved in (
            (POST_ALLOCATED + 1, POST_RESERVED),
            (POST_ALLOCATED, POST_RESERVED + 1),
        ):
            with self.subTest(allocated=allocated, reserved=reserved):
                too_high = row(
                    "post_setup:cold", allocated=allocated, reserved=reserved
                )
                with self.assertRaises(GateRefusal):
                    FixedAdmission(lambda _label: too_high).require_post_setup("cold")

    def test_completed_rows_survive_warm_gate_failure(self):
        low_warm = row(
            "warm_start:warm",
            usage=DEVICE_SIZE - SETUP_MIN_FREE + 1,
            allocated=0,
            reserved=0,
        )
        sequence, backend, _owner, _block, _clear, _sync, _archives, _labels = make_sequence(
            snapshots={"warm_start:warm": low_warm}
        )
        with self.assertRaises(GateRefusal):
            sequence.run()
        self.assertEqual([item["call"] for item in sequence.completed_calls], ["cold"])
        self.assertIn(("cold", "rgb"), backend.outputs)

    def test_wrong_shape_and_nonfinite_preserve_then_stop(self):
        for bad_stage in ("rgb_shape", "prediction_nonfinite"):
            with self.subTest(bad_stage=bad_stage):
                sequence, backend, _owner, _block, _clear, _sync, archives, _labels = make_sequence(
                    bad_stage=bad_stage
                )
                with self.assertRaises(ContractError):
                    sequence.run()
                self.assertEqual(archives, [])
                if bad_stage == "rgb_shape":
                    self.assertIn(("cold", "rgb"), backend.outputs)
                    self.assertEqual(backend.events[-1], "validate:rgb")
                else:
                    self.assertIn(("cold", "prediction"), backend.outputs)
                    self.assertNotIn("release:dit", backend.events)

    def test_public_block_clear_uses_exact_qualified_variant(self):
        qualified = QualifiedStub()
        clear_events = []
        sync_events = []

        def clear(_debug, name):
            clear_events.append(name)

        def sync():
            sync_events.append("sync")

        adapter = BlockClearAdapter(qualified, clear, sync, clock=lambda: 1.0)
        result = adapter.configure_prefix(object(), 32, "xpu", "cpu", object())
        self.assertEqual(result["variant"], "block-clear")
        self.assertEqual(qualified.kwargs["variant"], "block-clear")
        adapter.begin_call("cold")
        for index in range(32):
            qualified.kwargs["synchronize"]()
            qualified.kwargs["boundary"](index)
        record = adapter.end_call()
        self.assertEqual(record["count"], 32)
        self.assertEqual(len(clear_events), 32)
        self.assertEqual(len(sync_events), 64)

        adapter.begin_call("bad")
        with self.assertRaises(ContractError):
            qualified.kwargs["boundary"](1)

    def test_factory_reuses_frozen_backend_and_injects_only_adapter(self):
        frozen = FrozenBackendStub()
        qualified = QualifiedStub()
        calls = []
        backend, setup_clear, block_clear = make_frozen_backend(
            frozen, "spec", "ref", "reset", "gate", "0000:03:00.0", qualified,
            lambda debug, name: calls.append((debug, name)),
            lambda: calls.append(("sync", None)),
        )
        self.assertIsInstance(backend, FakeBackend)
        self.assertIs(frozen.received[2], block_clear)
        self.assertIs(setup_clear.debug, backend.debug)
        self.assertEqual(frozen.received[:2], ("spec", "ref"))
        self.assertEqual(frozen.received[3:], ("reset", "gate", "0000:03:00.0"))

    def test_allocator_schema_and_exact_b580_size_fail_closed(self):
        bad_schema = dict(row("post_setup:cold", allocated=POST_ALLOCATED, reserved=POST_RESERVED))
        bad_schema.pop("reserved")
        with self.assertRaises(ContractError):
            FixedAdmission(lambda _label: bad_schema).require_post_setup("cold")

        wrong_size = row("post_setup:cold", size=DEVICE_SIZE - 1,
                         allocated=POST_ALLOCATED, reserved=POST_RESERVED)
        with self.assertRaises(ContractError):
            FixedAdmission(lambda _label: wrong_size).require_post_setup("cold")


if __name__ == "__main__":
    unittest.main(verbosity=2)
