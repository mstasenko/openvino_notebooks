"""Static F16-window pooling graph used by the later mask speed probe."""

from __future__ import annotations

from numbers import Integral

import numpy as np


WINDOW_TOKENS = 128
CHANNELS = 1536


def build_pool_graph(windows: int):
    """Build one static F16 ``[windows,128,1536] -> [windows,1536]`` graph.

    Reduction arithmetic is explicitly F32 and the public result is rounded
    once to F16.  Q and K use separate instances of this same graph so the
    source mask's einsum, local-mask softmax, and top-k path stays on CPU.
    """

    if isinstance(windows, bool) or not isinstance(windows, Integral) or windows <= 0:
        raise ValueError(f"windows must be a positive integer, got {windows!r}")
    import openvino as ov
    from openvino import opset13 as ops

    data = ops.parameter([int(windows), WINDOW_TOKENS, CHANNELS], ov.Type.f16, name="windows")
    axes = ops.constant(np.array([1], dtype=np.int64))
    pooled_f32 = ops.reduce_mean(ops.convert(data, ov.Type.f32), axes, False)
    pooled = ops.convert(pooled_f32, ov.Type.f16)
    pooled.set_friendly_name("pooled")
    pooled.output(0).get_tensor().set_names({"pooled"})
    model = ov.Model([pooled], [data], f"mask_pool_windows{int(windows)}")
    model.validate_nodes_and_infer_types()
    return model


def add_pre_pool_results(model):
    """Expose small Q/K means from the existing F16 pre outputs, before readback."""
    import openvino as ov
    from openvino import opset13 as ops

    if any(name in port.get_names() for port in model.outputs for name in ("q_pooled", "k_pooled")):
        raise ValueError("Pre graph already exposes mask pooling")
    sources = [model.output(name).get_node().input_value(0) for name in ("q_windows", "k_windows")]
    if sources[0].get_partial_shape() != sources[1].get_partial_shape():
        raise ValueError("Pre Q/K pooling requires the same window geometry")
    for value in sources:
        shape = list(value.get_shape())
        if (value.get_element_type() != ov.Type.f16 or len(shape)!=3 or shape[0]not in (60,180,510,1530)
                or shape[1:]!=[WINDOW_TOKENS,CHANNELS]):
            raise ValueError('GPU mask pooling requires the reviewed HD/UHD pre ABI')
    for name, value in zip(("q_pooled", "k_pooled"), sources):
        mean = ops.reduce_mean(ops.convert(value, ov.Type.f32), ops.constant(np.array([1], np.int64)), False)
        pooled = ops.convert(mean, ov.Type.f16)
        pooled.output(0).get_tensor().set_names({name})
        model.add_results([ops.result(pooled)])
    model.validate_nodes_and_infer_types()
    return {"names": ["q_pooled", "k_pooled"], "input_shapes": [list(value.get_shape()) for value in sources],
            "math": "F32 mean over 128 source F16 tokens, once-rounded F16 output"}


__all__ = ["CHANNELS", "WINDOW_TOKENS", "build_pool_graph", "add_pre_pool_results"]
