"""Raw-bit finite validator for a complete canonical GPU K/V cache."""
from pathlib import Path

import numpy as np
import openvino as ov


HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "kv_finite.xml"
ALLOWED_WINDOWS = (60, 180, 240, 510, 1530, 2040)
WINDOW_TOKENS, HEADS, DEPTH = 128, 12, 128
TAIL = (WINDOW_TOKENS, HEADS, DEPTH)
GOOD_BITS = np.uint16(0x3C00)  # finite
BAD_BITS = np.uint16(0x0000)   # non-finite
U64_MAX = (1 << 64) - 1


def _shape(windows):
    if type(windows) is not int or windows not in ALLOWED_WINDOWS:
        raise ValueError(f"windows must be one of {ALLOWED_WINDOWS}")
    return [windows, *TAIL]


def _flag_shape(windows):
    _shape(windows)
    return [windows, 1, 1, 1]


def _admit(shapes, types):
    if (len(shapes) != 2 or len(types) != 2 or shapes[0] != shapes[1]
            or types != [ov.Type.f16, ov.Type.f16]):
        raise ValueError("two equal F16 K/V inputs are required")
    shape = shapes[0]
    if len(shape) != 4 or shape[1:] != list(TAIL):
        raise ValueError("canonical K/V shape must be [W,128,12,128]")
    if type(shape[0]) is not int or shape[0] not in ALLOWED_WINDOWS:
        raise ValueError(f"windows must be one of {ALLOWED_WINDOWS}")
    elements = int(np.prod(shape))
    if elements > U64_MAX // np.dtype(np.float16).itemsize:
        raise MemoryError("canonical K/V byte count exceeds 64-bit indexing")
    return int(shape[0])


class OtcCompleteKvFinite(ov.Op):
    def __init__(self, inputs):
        super().__init__(self, inputs)
        self.validate_and_infer_types()

    def validate_and_infer_types(self):
        if self.get_input_size() != 2:
            raise ValueError("complete K/V requires exactly two inputs")
        partial = [self.get_input_partial_shape(i) for i in range(2)]
        if any(not shape.is_static for shape in partial):
            raise ValueError("dynamic K/V shapes are not admitted")
        shapes = [list(shape.to_shape()) for shape in partial]
        types = [self.get_input_element_type(i) for i in range(2)]
        windows = _admit(shapes, types)
        self.set_output_type(0, ov.Type.f16, ov.PartialShape(_flag_shape(windows)))

    def clone_with_new_inputs(self, inputs):
        return OtcCompleteKvFinite(inputs)

    def visit_attributes(self, visitor):
        return True


def build(windows):
    """Build the standalone graph; it does not integrate attention."""
    from openvino import opset13 as ops

    shape = _shape(windows)
    k = ops.parameter(shape, ov.Type.f16, name="K_cache")
    v = ops.parameter(shape, ov.Type.f16, name="V_cache")
    for parameter in (k, v):
        parameter.output(0).get_tensor().set_names({parameter.get_friendly_name()})
    flags = OtcCompleteKvFinite([k.output(0), v.output(0)])
    flags.set_friendly_name("complete_kv_finite")
    flags.output(0).get_tensor().set_names({"flags"})
    return ov.Model([flags], [k, v], "complete_kv_finite")


def _device_name(value, name):
    """Require the 2026.5 device binding exposed by a remote object."""
    getter = getattr(value, "get_device_name", None)
    if not callable(getter):
        raise ValueError(f"{name} must expose get_device_name")
    try:
        actual = getter()
    except Exception as error:
        raise ValueError(f"{name} device binding could not be read") from error
    if type(actual) is not str or actual != "GPU.1":
        raise ValueError(f"{name} belongs to a different device")


def _marker(value, name):
    """Read the opaque context marker without unwrapping its OVAny value."""
    getter = getattr(value, "get_params", None)
    if not callable(getter):
        raise ValueError(f"{name} must expose get_params")
    try:
        params = getter()
        marker = (params.get("OCL_CONTEXT") if hasattr(params, "get")
                  else params["OCL_CONTEXT"])
    except Exception as error:
        raise ValueError(f"{name} context binding is malformed") from error
    if marker is None:
        raise ValueError(f"{name} has no OCL_CONTEXT marker")
    return marker


def _same_marker(left, right, name):
    """Compare opaque markers through their supported wrapper equality only."""
    if left is right:
        return
    try:
        equal = bool(left == right)
    except Exception as error:
        raise ValueError(f"{name} context markers are not comparable") from error
    if not equal:
        raise ValueError(f"{name} is not owned by the supplied context")


def _root(value, name, context, windows):
    if not isinstance(value, ov.RemoteTensor):
        raise ValueError(f"{name} must be an owned RemoteTensor root")
    expected = tuple(_shape(windows))
    try:
        shape, bytes_ = tuple(value.get_shape()), int(value.get_byte_size())
    except Exception as error:
        raise ValueError(f"{name} must expose shape and byte size") from error
    if shape != expected or bytes_ != int(np.prod(expected)) * 2:
        raise ValueError(f"{name} is not the complete canonical F16 cache")
    element_type = getattr(value, "get_element_type", None)
    if element_type is not None and element_type() != ov.Type.f16:
        raise ValueError(f"{name} must be F16")
    _device_name(value, name)
    _device_name(context, "context")
    _same_marker(_marker(value, name), _marker(context, "context"), name)


def validate_flags(flags, windows):
    """Return raw flag bits after validating the tiny binary F16 ABI."""
    if (not isinstance(flags, np.ndarray) or flags.dtype != np.dtype(np.float16)
            or tuple(flags.shape) != tuple(_flag_shape(windows))
            or not flags.flags.c_contiguous):
        raise ValueError("flags must be contiguous F16 [W,1,1,1]")
    bits = flags.view(np.uint16).reshape(-1)
    if np.any((bits != GOOD_BITS) & (bits != BAD_BITS)):
        raise ValueError("validator returned a non-binary or NaN flag")
    return bits.copy()


def cpu_window_flag(k_bits, v_bits):
    """CPU mirror of the kernel's exponent-only predicate."""
    k, v = np.asarray(k_bits, np.uint16), np.asarray(v_bits, np.uint16)
    if k.shape != v.shape:
        raise ValueError("K/V bit fixtures must have equal shape")
    bad = ((k & np.uint16(0x7C00)) == np.uint16(0x7C00)).any()
    bad = bool(bad or ((v & np.uint16(0x7C00)) == np.uint16(0x7C00)).any())
    return np.uint16(BAD_BITS if bad else GOOD_BITS)


class CompleteKvFinite:
    """Synchronous request over two complete context-owned K/V roots."""

    def __init__(self, context, core, device, remote_k, remote_v, config=None):
        if not str(ov.get_version()).startswith("2026.5."):
            raise ValueError("complete K/V validator requires OpenVINO 2026.5")
        if type(device) is not str or device != "GPU.1" or context is None or core is None:
            raise ValueError("authenticated GPU.1 and an owned context/Core are required")
        if id(remote_k) == id(remote_v):
            raise ValueError("K/V roots must be distinct")
        try:
            shape = tuple(remote_k.get_shape())
            other_shape = tuple(remote_v.get_shape())
        except Exception as error:
            raise ValueError("K/V roots must expose their complete shapes") from error
        if shape != other_shape or len(shape) != 4:
            raise ValueError("K/V roots must have the same complete shape")
        windows = _admit([list(shape), list(shape)], [ov.Type.f16, ov.Type.f16])
        _root(remote_k, "K_cache", context, windows)
        _root(remote_v, "V_cache", context, windows)
        options = dict(config or {})
        if ("CONFIG_FILE" in options
                and Path(options["CONFIG_FILE"]).resolve() != CONFIG_FILE.resolve()):
            raise ValueError("complete K/V validator owns its CONFIG_FILE")
        options["CONFIG_FILE"] = str(CONFIG_FILE)
        model = build(windows)
        compiled = core.compile_model(model, context, options)
        if list(compiled.get_property("EXECUTION_DEVICES")) != [device]:
            raise ValueError("complete K/V validator compiled for an unexpected device")
        request = compiled.create_infer_request()
        flags = ov.Tensor(ov.Type.f16, ov.Shape(_flag_shape(windows)))
        request.set_tensor("K_cache", remote_k)
        request.set_tensor("V_cache", remote_v)
        request.set_tensor("flags", flags)
        self.context, self.core, self.device = context, core, device
        self.remote_k, self.remote_v, self.windows = remote_k, remote_v, windows
        self._root_ids = (id(remote_k), id(remote_v))
        self.model, self.compiled, self.request, self._flags = model, compiled, request, flags

    def run(self):
        if (id(self.remote_k), id(self.remote_v)) != self._root_ids:
            raise ValueError("complete K/V validator roots were rebound")
        _root(self.remote_k, "K_cache", self.context, self.windows)
        _root(self.remote_v, "V_cache", self.context, self.windows)
        self._flags.data.fill(np.float16(0))
        self.request.start_async()
        self.request.wait()
        value = np.array(self._flags.data, dtype=np.float16, copy=True, order="C")
        bits = validate_flags(value, self.windows)
        bad = np.flatnonzero(bits == BAD_BITS)
        if bad.size:
            raise ValueError(f"complete K/V cache contains non-finite windows {bad.tolist()}")
        return value
