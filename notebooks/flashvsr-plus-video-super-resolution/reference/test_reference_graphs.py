"""Bounded source-closure checks for the extracted FlashVSR+ graphs.

These checks construct OpenVINO graph objects only.  They deliberately patch
``ov.Core`` and Torch CUDA/XPU probes so a passing result cannot hide
compilation, device discovery, checkpoint loading, or inference.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


REFERENCE = Path(__file__).resolve().parent
RUNTIME = os.environ.get("FLASHVSR_PLUS_OPENVINO_RUNTIME")
if RUNTIME:
    sys.path.insert(0, str(Path(RUNTIME).expanduser().resolve()))
for _path in (
    REFERENCE / "openvino" / "full_runtime",
    REFERENCE / "openvino" / "chunked_attention",
    REFERENCE / "openvino",
    REFERENCE,
    REFERENCE / "evidence",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _import_openvino():
    try:
        import openvino as ov
    except ImportError as error:  # pragma: no cover - environment dependent
        raise unittest.SkipTest("OpenVINO Python runtime is not installed") from error
    return ov


@contextmanager
def _no_device_creation(ov):
    """Make accidental Core construction a hard test failure."""

    original = ov.Core

    class ForbiddenCore:
        def __init__(self, *args, **kwargs):
            raise AssertionError("graph-closure test must not construct ov.Core")

    ov.Core = ForbiddenCore
    try:
        yield
    finally:
        ov.Core = original


def _patch_torch_device_queries():
    try:
        import torch
    except ImportError:  # pragma: no cover - decoder source is optional here
        return None
    saved = []
    for backend_name in ("cuda", "xpu"):
        backend = getattr(torch, backend_name, None)
        if backend is None:
            continue
        for name in (
            "is_available", "device_count", "current_device",
            "get_device_properties", "init",
        ):
            function = getattr(backend, name, None)
            if callable(function):
                saved.append((backend, name, function))
                setattr(
                    backend,
                    name,
                    lambda *args, _backend=backend_name, _name=name, **kwargs:
                    (_ for _ in ()).throw(
                        AssertionError(f"Torch {_backend} probe: {_name}")),
                )
    return saved


def _restore_torch_device_queries(saved):
    for backend, name, function in saved or ():
        setattr(backend, name, function)


class ReferenceGraphClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        cls.ov = _import_openvino()
        # Cover deferred imports in every test, not only graph-building calls.
        guard = _no_device_creation(cls.ov)
        guard.__enter__()
        cls.addClassCleanup(guard.__exit__, None, None, None)
        torch_saved = _patch_torch_device_queries()
        cls.addClassCleanup(_restore_torch_device_queries, torch_saved)
        from prepare_real import parameter_modules
        from frame_geometry import FrameGeometry
        from plus_graphs import build_bounded_pre, preserve_modulation_product
        from device_query_graph import expose_device_queries
        from post_cross_bridge_20261017 import build_post_cross_bridge
        from cross_precise_20261017 import model_abi, rewrite_clone
        from post_ffn2_state_20261017 import replace_ffn2_results_with_assigns

        cls.parameter_modules = staticmethod(parameter_modules)
        cls.FrameGeometry = FrameGeometry
        cls.build_bounded_pre = staticmethod(build_bounded_pre)
        cls.preserve_modulation_product = staticmethod(preserve_modulation_product)
        cls.expose_device_queries = staticmethod(expose_device_queries)
        cls.build_post_cross_bridge = staticmethod(build_post_cross_bridge)
        cls.model_abi = staticmethod(model_abi)
        cls.rewrite_clone = staticmethod(rewrite_clone)
        cls.replace_ffn2_results_with_assigns = staticmethod(
            replace_ffn2_results_with_assigns)

    def test_exact_assets_and_parameter_module_closure(self):
        required = (
            "openvino/full_model/contract.py",
            "openvino/full_model/builders.py",
            "openvino/mask_pool_probe.py",
            "openvino/source_cross_cache/qualify.py",
            "openvino/chunked_attention/plus_sparse_xmx.cl",
            "openvino/chunked_attention/plus_sparse_xmx.xml",
            "openvino/chunked_attention/kv_finite.cl",
            "openvino/chunked_attention/kv_finite.xml",
            "evidence/bf16_cross_bridge_20261017.cl",
            "evidence/bf16_cross_bridge_20261017.xml",
            "evidence/check_native_post_bridge_b580_20261017.py",
        )
        for relative in required:
            self.assertTrue((REFERENCE / relative).is_file(), relative)
        with _no_device_creation(self.ov):
            contract, builders, full = self.parameter_modules()
        self.assertEqual(contract.__name__, "real_parameter_contract")
        self.assertEqual(builders.__name__, "real_parameter_builders")
        self.assertEqual(full.__name__, "real_geometry_contract")

    def test_pre_private_outputs_and_device_queries(self):
        with _no_device_creation(self.ov):
            model, geometry, residual = self.build_bounded_pre(
                self.FrameGeometry(2160, 3840), rows=4096, precision="bf16")
            self.preserve_modulation_product(
                model, linear_stores=True, norm_stores=True)
            self.expose_device_queries(model, remote_kv=True, source_type="bf16")
        self.assertEqual(geometry, (4, 136, 240))
        self.assertTrue(residual)
        names = {port.get_any_name() for port in model.outputs}
        self.assertTrue({
            "q_device_0", "q_device_1", "k_device", "v_device",
            "q_finite", "kv_finite", "modulation_product",
            "q_linear_storage", "k_linear_storage", "norm1_storage",
            "q_rms_storage", "k_rms_storage",
        } <= names)
        bf16_names = {
            "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp",
            "modulation_product", "q_linear_storage", "k_linear_storage",
            "norm1_storage", "q_rms_storage", "k_rms_storage",
            "q_device_0", "q_device_1", "k_device", "v_device",
        }
        for port in model.outputs:
            if port.get_any_name() in bf16_names:
                self.assertEqual(port.get_element_type(), self.ov.Type.bf16,
                                 port.get_any_name())
        self.assertEqual(model.output("q_pooled").get_element_type(), self.ov.Type.f32)
        self.assertEqual(model.output("k_pooled").get_element_type(), self.ov.Type.f32)
        self.assertEqual(model.output("q_finite").get_element_type(), self.ov.Type.f16)
        self.assertEqual(model.output("kv_finite").get_element_type(), self.ov.Type.f16)

    def test_precise_ffn2_post_and_public_abi(self):
        with _no_device_creation(self.ov):
            model = self.build_post_cross_bridge(ffn2_matmul=True)
            before = self.model_abi(model)
            model = self.rewrite_clone(model, s1_count=2)
            precise = self.model_abi(model)
            model, descriptor = self.replace_ffn2_results_with_assigns(model)
            model.validate_nodes_and_infer_types()
        self.assertEqual(before["outputs"], precise["outputs"])
        self.assertEqual([port.get_any_name() for port in model.outputs], ["x_out"])
        self.assertEqual(len(model.get_variables()), 2)
        self.assertEqual(len(model.get_sinks()), 2)
        self.assertFalse(any(node.get_type_name() == "ReadValue"
                             for node in model.get_ordered_ops()))
        self.assertEqual(descriptor["public_outputs"], ["x_out"])
        self.assertEqual(len(descriptor["states"]), 2)

    def test_serialized_abi_helper_is_fail_closed(self):
        """The compiled-runtime helper must reject a plain source serialization."""

        from check_native_post_bridge_b580_20261017 import (
            _serialized_post_transport_abi,
        )
        with _no_device_creation(self.ov):
            model = self.rewrite_clone(
                self.build_post_cross_bridge(ffn2_matmul=True), s1_count=2)
            with tempfile.TemporaryDirectory() as directory:
                xml = Path(directory) / "post.xml"
                self.ov.serialize(model, str(xml), str(xml.with_suffix(".bin")))
                with self.assertRaises(ValueError):
                    _serialized_post_transport_abi(
                        xml, ids_proven=True, sparse_type="OtcCrossPreciseXmx")

    def test_decoder_source_and_converter_import_without_loading(self):
        saved = _patch_torch_device_queries()
        try:
            import plus_decoder
            source = plus_decoder._load_source_module()
            converter = plus_decoder._load_native_converter()
        finally:
            _restore_torch_device_queries(saved)
        self.assertTrue(Path(source.__file__).is_file())
        self.assertTrue(Path(converter.__file__).is_file())
        self.assertTrue(plus_decoder.SOURCE.is_file())
        self.assertFalse(plus_decoder.CHECKPOINT.is_file(),
                         "weights must remain external to the extraction")

    def test_config_generation_is_local_and_device_free(self):
        import check_bf16_cross_bridge_b580_20261017 as bridge
        import cross_precise_20261017 as precise

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "combined").mkdir()
            (root / "precise").mkdir()
            with _no_device_creation(self.ov):
                combined = bridge._make_plugin_config(root / "combined")
                selected = precise.make_precise_plugin_config(root / "precise")
            self.assertTrue(combined.is_file())
            self.assertTrue(selected.is_file())
            self.assertIn("OtcPlusSparseXmx", combined.read_text())
            self.assertIn("OtcCrossPreciseXmx", selected.read_text())


if __name__ == "__main__":
    unittest.main()
