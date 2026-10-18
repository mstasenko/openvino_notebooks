#!/usr/bin/env python3
"""Light fake-Core tests for the raw VAE roundtrip diagnostic.

Only NumPy and pure stubs are used.  No OpenVINO module, Core, IR, Torch, or
model is imported or constructed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).with_name("openvino-vae-roundtrip.py")


def load_target():
    spec = importlib.util.spec_from_file_location("otc_vae_roundtrip_target", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeElementType:
    def get_type_name(self):
        return "f32"


class FakePartialShape:
    is_static = True

    def __init__(self, shape):
        self.shape = tuple(shape)

    def to_shape(self):
        return self.shape

    def __str__(self):
        return str(self.shape)


class FakePort:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def get_partial_shape(self):
        return FakePartialShape(self.shape)

    def get_element_type(self):
        return FakeElementType()


class FakeModel:
    def __init__(self, target):
        self.inputs = [FakePort(target.LATENT_SHAPE)]
        self.outputs = [FakePort(target.VIDEO_SHAPE)]


class FakeRequest:
    def __init__(self, compiled, output):
        self.compiled = compiled
        self.output = output
        self.calls = 0

    def infer(self, _feed):
        self.calls += 1
        return {self.compiled.outputs[0]: self.output}


class FakeCompiled:
    def __init__(self, target, output):
        self.inputs = [FakePort(target.LATENT_SHAPE)]
        self.outputs = [FakePort(target.VIDEO_SHAPE)]
        self.output = output
        self.request = FakeRequest(self, output)

    def create_infer_request(self):
        return self.request


class FakeCore:
    def __init__(self, target):
        self.target = target
        self.model = None
        self.compiled = None
        self.compile_device = None
        self.compile_properties = None

    def read_model(self, _path):
        self.model = FakeModel(self.target)
        return self.model

    def compile_model(self, model, device, properties):
        self.compile_device = device
        self.compile_properties = dict(properties)
        self.compiled = FakeCompiled(self.target, np.zeros(self.target.VIDEO_SHAPE, dtype=np.float32))
        return self.compiled


class FakeOV:
    class Type:
        f32 = "f32"

    def __init__(self, core):
        self.core = core

    def Core(self):
        return self.core


class VaeRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = load_target()

    def test_shape_and_finite_contract_rejects_bad_array(self):
        with self.assertRaises(self.target.ContractError):
            self.target._array(
                np,
                np.zeros((1, 2), dtype=np.float32),
                "latent",
                self.target.LATENT_SHAPE,
            )
        bad = np.zeros(self.target.LATENT_SHAPE, dtype=np.float32)
        bad.flat[0] = np.nan
        with self.assertRaises(self.target.ContractError):
            self.target._array(np, bad, "latent", self.target.LATENT_SHAPE)

    def test_raw_to_rgb_clamps_and_rounds_without_wavelet(self):
        raw = np.zeros(self.target.VIDEO_SHAPE, dtype=np.float32)
        raw[0, 0, 0, 0, :3] = (-1.0, 0.0, 1.0)
        raw[0, 1, 0, 0, :3] = (-2.0, 0.5, 2.0)
        raw[0, 2, 0, 0, :3] = (-0.5, 0.5, 1.5)
        frames = self.target._raw_frames(np, raw)
        rgb = self.target._rgb_frames(np, frames)
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb.shape, self.target.FRAME_SHAPE)
        np.testing.assert_array_equal(
            rgb[0, 0, :3],
            np.asarray([[0, 0, 64], [128, 191, 191], [255, 255, 255]], dtype=np.uint8),
        )

    def test_decode_once_uses_cpu_contract_and_one_infer(self):
        core = FakeCore(self.target)
        raw, info = self.target.decode_once(
            np,
            FakeOV(core),
            np.zeros(self.target.LATENT_SHAPE, dtype=np.float32),
        )
        self.assertEqual(raw.shape, self.target.VIDEO_SHAPE)
        self.assertEqual(info["input_type"], "f32")
        self.assertEqual(info["output_type"], "f32")
        self.assertEqual(core.compile_device, "CPU")
        self.assertEqual(core.compiled.request.calls, 1)
        self.assertEqual(core.compile_properties["INFERENCE_NUM_THREADS"], "8")
        self.assertEqual(core.compile_properties["NUM_STREAMS"], "1")
        self.assertEqual(core.compile_properties["CACHE_DIR"], "")
        self.assertEqual(core.compile_properties["INFERENCE_PRECISION_HINT"], "f32")
        core.compiled.request.output.fill(3.0)
        self.assertEqual(float(raw.max()), 0.0)

    def test_output_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="vae-roundtrip-test-") as temporary:
            output = Path(temporary) / "roundtrip"
            with patch.object(self.target, "OUTPUT_DIR", output):
                self.assertEqual(self.target._new_output_dir(), output)
                with self.assertRaises(FileExistsError):
                    self.target._new_output_dir()


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
