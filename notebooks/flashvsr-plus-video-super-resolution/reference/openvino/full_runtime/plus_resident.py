"""Typed, reusable Plus requests with resident weights and private GPU stores.

Sixteen-bit arrays in this controller are storage bits, never numeric F16
views of BF16. Only requested public outputs are copied back to the host.
"""
import time
from math import prod

import numpy as np

from plus_pipeline import compile_config
from plus_storage import is_finite_bf16, storage_tensor


def _build_bf16_finite_model():
    """Build the native-BF16 checker used by the optional context handoff."""
    import openvino as ov
    from openvino import opset13 as ops

    value = ops.parameter([12, 16, 128, 128], ov.Type.bf16, name="attention")
    # The custom attention primitive uses F16 only as a two-byte transport
    # port.  Its remote output is re-viewed as BF16 before this exact finite
    # check; the conversion below is BF16 -> F32, never transport-F16 -> F32.
    finite = ops.less_equal(ops.absolute(ops.convert(value, ov.Type.f32)),
                            ops.constant(np.finfo(np.float32).max, ov.Type.f32))
    # Numeric min uses the qualified parallel GPU reduction; boolean AND
    # dispatches a very slow generic reduction in the pinned GPU plugin.
    flag = ops.reduce_min(ops.convert(finite, ov.Type.f16),
                          ops.constant([0, 1, 2, 3], ov.Type.i32), False)
    flag.output(0).get_tensor().set_names({"flags"})
    return ov.Model([flag], [value], "resident_attention_bf16_finite")


def _checked_remote_shape(value, name):
    try:
        shape = tuple(value.get_shape())
        byte_size = int(value.get_byte_size())
    except Exception as error:
        raise ValueError(f"{name} must expose shape and byte size") from error
    if len(shape) != 4:
        raise ValueError(f"{name} must be a canonical four-dimensional tensor")
    elements = 1
    for size in shape:
        if not isinstance(size, (int, np.integer)) or int(size) <= 0:
            raise ValueError(f"{name} has an invalid shape")
        elements *= int(size)
        if elements > (1 << 62):
            raise ValueError(f"{name} shape is too large")
    if byte_size != 2 * elements:
        raise ValueError(f"{name} must expose its complete two-byte storage")
    return tuple(int(size) for size in shape)


def typed_remote_view(context, root, port_type):
    """Make a same-context typed view of a complete two-byte remote tensor.

    The parameter map is passed back to the owned RemoteContext verbatim.  No
    pointer or numeric data is exposed here; the BF16/F16 distinction is only
    the port type for the same byte extent.
    """
    import openvino as ov
    from kv_finite import _device_name, _marker, _same_marker

    if port_type not in (ov.Type.bf16, ov.Type.f16):
        raise ValueError("Typed remote views require BF16 or F16")
    remote_type = getattr(ov, "RemoteTensor", None)
    if remote_type is not None and not isinstance(root, remote_type):
        raise ValueError("Typed remote view requires a RemoteTensor root")
    shape = _checked_remote_shape(root, "remote root")
    source_type = getattr(root, "get_element_type", None)
    if callable(source_type) and source_type() not in (ov.Type.bf16, ov.Type.f16):
        raise ValueError("Typed remote view requires a two-byte source root")
    _device_name(root, "remote root")
    _device_name(context, "remote context")
    _same_marker(_marker(root, "remote root"), _marker(context, "remote context"),
                 "typed remote view")
    get_params = getattr(root, "get_params", None)
    if not callable(get_params):
        raise ValueError("Remote root does not expose its parameter map")
    alias = context.create_tensor(port_type, ov.Shape(shape), get_params())
    alias_shape = getattr(alias, "get_shape", None)
    if callable(alias_shape) and tuple(alias_shape()) != shape:
        raise ValueError("Typed remote view changed the root shape")
    alias_bytes = getattr(alias, "get_byte_size", None)
    if callable(alias_bytes) and int(alias_bytes()) != 2 * prod(shape):
        raise ValueError("Typed remote view changed the root byte extent")
    alias_type = getattr(alias, "get_element_type", None)
    if callable(alias_type) and alias_type() != port_type:
        raise ValueError("Typed remote view changed the requested port type")
    return alias


def bits_tensor(bits, port_type=None, *, share=False):
    """Finite BF16 bits; shared feeds must stay alive through synchronous wait."""
    import openvino as ov
    bits = np.asarray(bits)
    if bits.dtype != np.uint16 or not bits.size or not bits.flags.c_contiguous:
        raise ValueError("Finite uint16 BF16 storage bits required")
    if not is_finite_bf16(bits):
        raise ValueError("Finite uint16 BF16 storage bits required")
    port_type = ov.Type.bf16 if port_type is None else port_type
    if port_type not in (ov.Type.bf16, ov.Type.f16):
        raise ValueError("BF16 storage or explicit F16 bit transport required")
    if type(share) is not bool or (share and not bits.flags.writeable):
        raise ValueError("Shared input requires writable contiguous storage")
    if share:
        # Derive both shape and byte extent from the validated array. The
        # explicit-type constructor otherwise accepts undersized buffers.
        return ov.Tensor(bits, bits.shape, port_type)
    tensor = ov.Tensor(port_type, bits.shape)
    tensor.data.view(np.uint16)[...] = bits
    return tensor


def tensor_bits(tensor, *, copy=True):
    """Read host storage bits, not an FP16 numeric conversion.

    The default owns an independent array.  ``copy=False`` is reserved for a
    freshly allocated private host Tensor whose NumPy view retains that Tensor
    through its ``base`` chain.
    """
    import openvino as ov
    if type(copy) is not bool:
        raise ValueError("copy must be boolean")
    if tensor.element_type not in (ov.Type.bf16, ov.Type.f16):
        raise ValueError("A two-byte host tensor is required")
    view = tensor.data.view(np.uint16)
    return view if not copy else np.array(view, copy=True)


class ResidentGraph:
    """One request and fixed output buffers, shared by all checkpoint layers."""
    def __init__(self, core, model, device, *, context=None, config=None):
        import openvino as ov
        if device != "CPU" and context is None:
            raise ValueError("GPU requests require the caller's owned context")
        self.marker = None
        if context is not None:
            from kv_finite import _device_name, _marker
            _device_name(context, "resident context")
            self.marker = _marker(context, "resident context")
        self.ov, self.context = ov, context
        output_names = [port.get_any_name() for port in model.outputs]
        if len(output_names) != len(set(output_names)):
            raise ValueError("Resident logical output names must be unique")
        input_names = {port.get_any_name() for port in model.inputs}
        if input_names.intersection(output_names):
            # The pinned Python RemoteTensor API binds only by name, and a
            # history name shared by an input/output selects the input first.
            model = model.clone()
            for port, name in zip(model.outputs, output_names):
                if name in input_names:
                    physical = "resident_output__" + name
                    if physical in input_names or physical in output_names:
                        raise ValueError("Resident output namespace collision")
                    port.get_tensor().set_names({physical})
            if {port.get_any_name() for port in model.inputs} != input_names:
                raise ValueError("Output renaming changed an input alias")
        model_outputs = list(model.outputs)
        physical_names = [port.get_any_name() for port in model_outputs]
        if len(physical_names) != len(set(physical_names)):
            raise ValueError("Resident physical output names must be unique")
        options = compile_config(ov, device)
        options.update(config or {})
        started = time.perf_counter()
        self.compiled = core.compile_model(model, context if context is not None else device, options)
        self.compile_seconds = time.perf_counter() - started
        if list(self.compiled.get_property("EXECUTION_DEVICES")) != [device]:
            raise ValueError("Resident graph executed on an unintended device")
        compiled_outputs = list(self.compiled.outputs)
        if len(compiled_outputs) != len(model_outputs):
            raise ValueError("Resident compiled output count changed")
        for index, (expected, actual) in enumerate(zip(model_outputs, compiled_outputs)):
            if (actual.get_any_name() != physical_names[index]
                    or actual.get_element_type() != expected.get_element_type()
                    or tuple(actual.get_shape()) != tuple(expected.get_shape())):
                raise ValueError("Resident compiled output ABI changed at index " + str(index))
        self.request = self.compiled.create_infer_request()
        self.inputs = {p.get_any_name(): p for p in self.compiled.inputs}
        self.outputs = dict(zip(output_names, compiled_outputs))
        # Port ABI is immutable after compilation; cache only this static
        # metadata, never tensor identity or tensor contents.  Live tensor
        # shape/bytes/type/context checks remain in _validate_tensor().
        self._port_abi = {}
        for port in (*self.inputs.values(), *self.outputs.values()):
            shape = tuple(port.shape)
            kind = port.get_element_type()
            self._port_abi[id(port)] = (shape, int(kind.size) * prod(shape), kind)
        self.buffers = {name: self.allocate(port) for name, port in self.outputs.items()}
        for name, tensor in self.buffers.items():
            self.request.set_tensor(self.outputs[name].get_any_name(), tensor)

    def allocate(self, port):
        shape, kind = port.get_shape(), port.get_element_type()
        # The pinned GPU plugin crashes when copying rank-0 RemoteTensors.
        # Tiny scalar flags stay host-bound; non-scalar stages remain remote.
        return (self.context.create_tensor(kind, shape, {}) if self.context is not None and len(shape)
                else self.ov.Tensor(kind, shape))

    def upload(self, name, value):
        """Upload one F32 checkpoint tensor once, directly into native BF16."""
        port = self.inputs[name]
        if not name.startswith("weight__") or port.get_element_type() != self.ov.Type.bf16:
            raise ValueError("Only BF16 learned weight inputs are uploaded here")
        value = np.asarray(value)
        if value.dtype != np.float32 or tuple(value.shape) != tuple(port.shape):
            raise ValueError("Resident weight shape/dtype changed: " + name)
        host = storage_tensor(value, "bf16")
        target = self.allocate(port)
        host.copy_to(target)
        return target

    def _validate_tensor(self, name, value, port):
        if isinstance(value, self.ov.RemoteTensor):
            from kv_finite import _device_name, _marker, _same_marker
            if self.marker is None:
                raise ValueError("Remote tensors require an owned GPU context")
            _device_name(value, name)
            _same_marker(_marker(value, name), self.marker, name)
        port_abi = getattr(self, "_port_abi", None)
        # Lightweight test doubles and external direct-call ports may skip
        # __init__; compute their ABI without retaining an unowned object ID.
        expected = port_abi.get(id(port)) if port_abi is not None else None
        if expected is None:  # retain direct-call behavior for an external port
            shape = tuple(port.shape)
            kind = port.get_element_type()
            expected = (shape, int(kind.size) * prod(shape), kind)
        expected_shape, expected_bytes, expected_kind = expected
        if (tuple(value.get_shape()) != expected_shape
                or int(value.get_byte_size()) != expected_bytes):
            raise ValueError("Resident tensor shape/storage changed: " + name)
        kind = getattr(value, "element_type", None)
        if kind is not None and kind != expected_kind:
            raise ValueError("Resident tensor type changed: " + name)

    def bind_output(self, name, tensor):
        """Produce directly into a downstream buffer in the same context."""
        self._validate_tensor(name, tensor, self.outputs[name])
        self.request.set_tensor(self.outputs[name].get_any_name(), tensor)
        self.buffers[name] = tensor

    def infer(self, feeds, *, read):
        """Complete all GPU writes, then copy only named public results.

        Inputs are already typed owning tensors (host or same-context remote).
        Their owners must remain live until this synchronous call returns.
        """
        self.last_seconds = {}
        started = time.perf_counter()
        if set(feeds) != set(self.inputs) or not set(read) <= set(self.outputs):
            raise ValueError("Complete typed feeds and known public results required")
        for name, value in feeds.items():
            self._validate_tensor(name, value, self.inputs[name])
            self.request.set_tensor(name, value)
        bound = time.perf_counter()
        self.request.start_async()
        self.request.wait()
        completed = time.perf_counter()
        results = {}
        for name in read:
            port = self.outputs[name]
            host = self.buffers[name]
            private_host = False
            if isinstance(host, self.ov.RemoteTensor):
                host = self.ov.Tensor(port.get_element_type(), port.get_shape())
                self.buffers[name].copy_to(host)
                private_host = True
            if host.element_type in (self.ov.Type.bf16, self.ov.Type.f16):
                results[name] = tensor_bits(host, copy=not private_host)
            else:
                # ``host.data`` is a NumPy view whose base retains this fresh
                # private Tensor; public/reused host destinations still copy.
                results[name] = (host.data if private_host
                                 else np.array(host.data, copy=True))
        self.last_seconds = {"bind": bound - started, "execute": completed - bound,
                             "read": time.perf_counter() - completed}
        return results


class ResidentAttention:
    """Bounded Q dispatches with one shared active-layer K/V device root.

    Historical layers remain compact host BF16; K/V upload occurs once per
    active layer, not once per query batch. Full context is never truncated.
    """
    def __init__(self, core, device, context, geometry, *, query_windows=16,
                 split_key=False, binary_prefix=False):
        import plus_sparse_xmx as kernel
        if query_windows not in (1, 2, 3, 4, 8, 16):
            raise ValueError("Qualified bounded query size required")
        if type(split_key) is not bool or (split_key and query_windows != 16):
            raise ValueError("Split-key attention requires a boolean opt-in and C16 queries")
        if type(binary_prefix) is not bool or (binary_prefix and not split_key):
            raise ValueError("Binary prefix requires a boolean opt-in and split-key attention")
        self.kernel, self.core, self.context, self.device = kernel, core, context, device
        self.geometry, self.queries = geometry, query_windows
        self.split_key = split_key
        self.binary_prefix = binary_prefix
        self.max_keys = 5 * geometry.spatial_windows
        if split_key and self.max_keys > 2550:
            raise ValueError("Split-key attention exceeds the qualified key capacity")
        self.roots = tuple(context.create_tensor(kernel.STORAGE,
            kernel.ov.Shape([self.max_keys, 128, 12, 128]), {}) for _ in range(2))
        self.engine = None
        self.compile_seconds = 0.
        self.last_seconds = {}
        # These are allocated only when the controller opts into the remote
        # pre/attention boundary.  The ordinary infer() path stays unchanged.
        self._remote_q_slots = None
        self._remote_q_views = None
        self._remote_kv_views = None
        self._remote_tail = None
        self._remote_prefix_keys = None
        self._remote_total_windows = None
        self._remote_ready_windows = 0
        self._remote_next_start = 0
        self._remote_pending = None
        self._remote_consumed = False
        self._remote_context_checker = None
        self._remote_context_input = None
        self._remote_context_output = None
        self._remote_context_ready = False
        self._remote_context_failed = False
        self._remote_context_next_start = 0

    def _prepare_context_handoff(self, graph):
        """Reuse the request's distinct output with a native-BF16 checker."""
        import openvino as ov

        if not hasattr(graph, "buffers") or "attention" not in graph.buffers:
            raise ValueError("Resident attention graph has no context output buffer")
        output = graph.buffers["attention"]
        expected_shape = (12, 16, 128, 128)
        if tuple(output.get_shape()) != expected_shape:
            raise ValueError("Resident attention output shape changed")
        # This pinned RemoteTensor API exposes no type getter. The compiled
        # port and our allocate() call establish its type; validate any extra
        # type metadata when the runtime does expose it.
        if graph.outputs["attention"].get_element_type() != ov.Type.f16:
            raise ValueError("Resident attention output transport changed")
        graph._validate_tensor("attention", output, graph.outputs["attention"])
        if self._remote_context_output is None:
            # Never bind the output directly to a consumed Q slot: the custom
            # attention primitive may read Q and write its output concurrently.
            if any(output is slot for slot in (*self._remote_q_slots, *self._remote_q_views)):
                raise ValueError("Resident context output aliases a Q slot")
            self._remote_context_output = output
            self._remote_context_input = typed_remote_view(
                self.context, output, ov.Type.bf16
            )
        elif output is not self._remote_context_output:
            raise ValueError("Resident context output binding changed")
        if self._remote_context_checker is None:
            self._remote_context_checker = ResidentGraph(
                self.core, _build_bf16_finite_model(), self.device,
                context=self.context
            )
            self.compile_seconds += self._remote_context_checker.compile_seconds
        return self._remote_context_checker

    @staticmethod
    def _validate_context_flag(flag):
        flag = np.asarray(flag)
        if flag.shape != () or flag.dtype != np.uint16 or flag.item() != 0x3c00:
            raise FloatingPointError("Resident attention produced non-finite BF16 context")

    def _validate_context_slot(self, slot, name):
        import openvino as ov
        if _checked_remote_shape(slot, name) != (12, 16, 128, 128):
            raise ValueError(f"{name} shape changed")
        get_type = getattr(slot, "get_element_type", None)
        if callable(get_type) and get_type() != ov.Type.bf16:
            raise ValueError(f"{name} must be native BF16")
        from kv_finite import _device_name, _marker, _same_marker
        _device_name(slot, name)
        _device_name(self.context, "resident context")
        _same_marker(_marker(slot, name), _marker(self.context, "resident context"), name)

    def context_targets(self, start, count):
        """Return the two native-BF16 C16 slots for one post slab.

        ``count`` is the number of valid windows in this slab.  A short slab
        is permitted only at the terminal tail; callers must ignore unused
        rows in the second slot in that case.
        """
        if type(start) is not int or type(count) is not int:
            raise ValueError("Context slab coordinates must be integers")
        if not self._remote_context_ready or self._remote_context_failed:
            raise ValueError("Resident context is not ready")
        total = self._remote_total_windows
        if (total is None or start != self._remote_context_next_start or start < 0
                or count <= 0 or count > 32 or start % 32 or start + count > total):
            raise ValueError("Context slabs must be sequential 32-window blocks")
        if start + count < total and count != 32:
            raise ValueError("Only the terminal context slab may be short")
        slot = start // 16
        if self._remote_q_slots is None or slot + 1 >= len(self._remote_q_slots):
            raise ValueError("Context C16 slot geometry changed")
        first, second = self._remote_q_slots[slot], self._remote_q_slots[slot + 1]
        self._validate_context_slot(first, "self_context_0")
        self._validate_context_slot(second, "self_context_1")
        self._remote_context_next_start = start + count
        return {"self_context_0": first, "self_context_1": second}

    def _ensure_engine(self):
        if self.engine is None:
            model = self.kernel.build(
                12, self.queries, self.max_keys, self.max_keys,
                canonical_kv=True, split_key=self.split_key
            )
            if self.binary_prefix:
                config = self.kernel.HERE.parents[1] / "evidence" / "attention_splitkey_binary_prefix_20261017.xml"
            else:
                config = (self.kernel.HERE.parents[1] / "evidence" / "attention_splitkey_20261017.xml"
                          if self.split_key else self.kernel.HERE / "plus_sparse_xmx.xml")
            graph = ResidentGraph(self.core, model, self.device, context=self.context,
                config={"CONFIG_FILE": str(config),
                        "INFERENCE_PRECISION_HINT": self.kernel.STORAGE})
            self.compile_seconds += graph.compile_seconds
            self.engine = graph
        return self.engine

    def _copy_kv_prefix(self, k, v, keys):
        ov, kernel = self.kernel.ov, self.kernel
        for host, root in zip((k, v), self.roots):
            packed = host.reshape(keys, 128, 12, 128)
            # Private precondition: prepare_remote() or infer() has already
            # finite-validated both complete host operands.  These checks keep
            # the raw shared constructor strict without rescanning UHD bytes.
            if (packed.dtype != np.uint16 or packed.shape != (keys, 128, 12, 128)
                    or not packed.flags.c_contiguous or not packed.flags.writeable
                    or packed.nbytes != 2 * prod(packed.shape)):
                raise ValueError("Validated raw K/V storage must be writable and contiguous")
            if kernel.STORAGE not in (ov.Type.bf16, ov.Type.f16):
                raise ValueError("Raw K/V storage requires BF16 or F16 transport")
            prefix = ov.RemoteTensor(
                root,
                ov.Coordinate([0, 0, 0, 0]),
                ov.Coordinate([keys, 128, 12, 128]),
            )
            prefix.copy_from(ov.Tensor(packed, packed.shape, kernel.STORAGE))

    @staticmethod
    def _remote_previous(previous):
        if previous is None:
            return None, None
        if isinstance(previous, dict):
            if not {"k", "v"} <= set(previous):
                raise ValueError("Remote history requires compact K/V fields")
            return previous["k"], previous["v"]
        raise ValueError("Remote history must be None or the existing K/V mapping")

    def _remote_windows(self):
        spatial = getattr(self.geometry, "spatial_windows", None)
        if type(spatial) is not int or spatial <= 0:
            raise ValueError("Remote attention requires positive spatial windows")
        windows = 2 * spatial
        if self.max_keys != 5 * spatial:
            raise ValueError("Remote attention cache capacity changed")
        return windows

    @staticmethod
    def _remote_host_history(value, name):
        value = np.asarray(value)
        if value.dtype != np.uint16 or value.ndim != 3 or value.shape[1:] != (128, 1536):
            raise ValueError(f"Remote history {name} must be uint16 [K,128,1536]")
        if not value.flags.c_contiguous:
            raise ValueError(f"Remote history {name} must be contiguous")
        if not value.flags.writeable:
            raise ValueError(f"Remote history {name} must be writable")
        if not is_finite_bf16(value):
            raise ValueError(f"Remote history {name} contains non-finite BF16 bits")
        return value

    @staticmethod
    def _remote_roi(root, start, count):
        import openvino as ov
        if type(start) is not int or type(count) is not int or start < 0 or count <= 0:
            raise ValueError("Remote ROI coordinates must be positive integers")
        return ov.RemoteTensor(root, ov.Coordinate([start, 0, 0, 0]),
                               ov.Coordinate([start + count, 128, 12, 128]))

    def _ensure_remote_buffers(self, windows):
        import openvino as ov
        if self.queries != 16:
            raise ValueError("Remote pre/attention requires a fixed C16 query graph")
        if self._remote_q_slots is not None:
            if len(self._remote_q_slots) != max(2, ((windows + 15) // 16 + 1) // 2 * 2):
                raise ValueError("Remote query slot geometry changed")
            return
        slots = (windows + 15) // 16
        if slots % 2:
            slots += 1
        shape = ov.Shape([12, 16, 128, 128])
        self._remote_q_slots = tuple(
            self.context.create_tensor(ov.Type.bf16, shape, {}) for _ in range(slots)
        )
        self._remote_q_views = tuple(
            typed_remote_view(self.context, slot, ov.Type.f16)
            for slot in self._remote_q_slots
        )
        self._remote_kv_views = tuple(
            typed_remote_view(self.context, root, ov.Type.bf16)
            for root in self.roots
        )
        tail_shape = ov.Shape([32, 128, 12, 128])
        self._remote_tail = tuple(
            self.context.create_tensor(ov.Type.bf16, tail_shape, {}) for _ in range(2)
        )

    def prepare_remote(self, previous):
        """Stage one compact host history prefix for the remote pre path.

        Fresh K/V are written by the pre graph through pre_targets().  This
        method intentionally uploads only ``previous`` and never reads or
        invents a second history representation.
        """
        started = time.perf_counter()
        windows = self._remote_windows()
        old_k, old_v = self._remote_previous(previous)
        if old_k is None or old_v is None:
            if old_k is not None or old_v is not None:
                raise ValueError("Remote history requires both K and V")
            keys = 0
        else:
            old_k = self._remote_host_history(old_k, "K")
            old_v = self._remote_host_history(old_v, "V")
            if old_k.shape != old_v.shape or not old_k.shape[0]:
                raise ValueError("Remote history K/V shape or length changed")
            keys = int(old_k.shape[0])
        if keys + windows > self.max_keys:
            raise ValueError("Remote compact history plus current windows exceeds capacity")
        validated = time.perf_counter()
        self._ensure_remote_buffers(windows)
        allocated = time.perf_counter()
        if keys:
            self._copy_kv_prefix(old_k, old_v, keys)
        uploaded = time.perf_counter()
        self.last_prepare_seconds = {
            "validate": validated - started,
            "buffers": allocated - validated,
            "upload": uploaded - allocated,
        }
        self._remote_prefix_keys = keys
        self._remote_total_windows = windows
        self._remote_ready_windows = 0
        self._remote_next_start = 0
        self._remote_pending = None
        self._remote_consumed = False
        self._remote_context_ready = False
        self._remote_context_failed = False
        self._remote_context_next_start = 0

    def pre_targets(self, start, count):
        """Return owned BF16 output targets for one sequential 32-window slab."""
        if self._remote_total_windows is None:
            raise ValueError("prepare_remote must precede pre_targets")
        if type(start) is not int or type(count) is not int:
            raise ValueError("Remote pre coordinates must be integers")
        total = self._remote_total_windows
        if (start != self._remote_next_start or start < 0 or count <= 0
                or count > 32 or start % 32 or start + count > total):
            raise ValueError("Remote pre slabs must be sequential 32-window blocks")
        if start + count < total and count != 32:
            raise ValueError("Only the final remote pre slab may be shorter than 32 windows")
        if self._remote_pending is not None:
            raise ValueError("Previous remote pre slab has not completed")
        slot = start // 16
        if slot + 1 >= len(self._remote_q_slots):
            raise ValueError("Remote C16 query slot geometry changed")
        prefix = self._remote_prefix_keys
        if count == 32:
            if prefix + start + count > self.max_keys:
                raise ValueError("Remote K/V destination ROI exceeds cache capacity")
            k_target, v_target = tuple(
                self._remote_roi(root, prefix + start, count)
                for root in self._remote_kv_views
            )
        else:
            k_target, v_target = self._remote_tail
        self._remote_pending = (start, count)
        return {
            "q_device_0": self._remote_q_slots[slot],
            "q_device_1": self._remote_q_slots[slot + 1],
            "k_device": k_target,
            "v_device": v_target,
        }

    def complete_pre(self, start, count):
        """Retire one completed pre slab, copying only a final tail if needed."""
        if self._remote_pending != (start, count):
            raise ValueError("Remote pre completion does not match its pending slab")
        if count != 32:
            prefix = self._remote_prefix_keys
            for root, scratch in zip(self._remote_kv_views, self._remote_tail):
                destination = self._remote_roi(root, prefix + start, count)
                source = self._remote_roi(scratch, 0, count)
                destination.copy_from(source)
        self._remote_ready_windows = start + count
        self._remote_next_start = start + count
        self._remote_pending = None

    def read_retained_into(self, k_destination, v_destination, source_start):
        """Copy the exact retained current K/V suffix into its existing owner."""
        import openvino as ov
        from kv_finite import _device_name, _marker, _same_marker

        total = self._remote_windows()
        prefix = self._remote_prefix_keys
        if (self._remote_total_windows != total or self._remote_pending is not None
                or self._remote_ready_windows != total or self._remote_next_start != total
                or self._remote_consumed or self._remote_context_failed):
            raise ValueError("Retained K/V requires completed, unconsumed pre slabs")
        if (type(prefix) is not int or prefix < 0 or prefix + total > self.max_keys
                or type(source_start) is not int
                or source_start not in (0, total // 2)):
            raise ValueError("Retained K/V source coordinates changed")
        count = total - source_start
        destinations = (k_destination, v_destination)
        for destination in destinations:
            if (not isinstance(destination, np.ndarray) or destination.dtype != np.uint16
                    or destination.shape != (count, 128, 1536)
                    or not destination.flags.c_contiguous or not destination.flags.writeable):
                raise ValueError("Retained K/V requires writable contiguous uint16 destinations")
        if np.shares_memory(*destinations):
            raise ValueError("Retained K/V destinations must not alias")
        if len(self.roots) != 2 or self.roots[0] is self.roots[1]:
            raise ValueError("Retained K/V requires distinct roots")
        _device_name(self.context, "resident context")
        marker = _marker(self.context, "resident context")
        for root in self.roots:
            if (not isinstance(root, ov.RemoteTensor)
                    or _checked_remote_shape(root, "retained root")
                    != (self.max_keys, 128, 12, 128)):
                raise ValueError("Retained K/V root geometry changed")
            get_type = getattr(root, "get_element_type", None)
            if callable(get_type) and get_type() != ov.Type.f16:
                raise ValueError("Retained K/V root must use F16 bit transport")
            _device_name(root, "retained root")
            _same_marker(_marker(root, "retained root"), marker, "retained root")
        for root, destination in zip(self.roots, destinations):
            source = self._remote_roi(root, prefix + source_start, count)
            # Explicit same-size F16 transport shares these raw BF16 bytes;
            # no numeric conversion or temporary host K/V owner is involved.
            host = ov.Tensor(destination, (count, 128, 12, 128), ov.Type.f16)
            source.copy_to(host)

    def infer_remote(self, mask, *, retain_context=False):
        """Run attention over prepared remote Q/K/V without host Q/K/V uploads.

        With ``retain_context=True``, attention outputs stay in the consumed
        C16 Q slots after a native-BF16 finite check; the default path retains
        its owning dense host return value.
        """
        from window_pre import window_indices
        ov, kernel = self.kernel.ov, self.kernel
        if type(retain_context) is not bool:
            raise ValueError("retain_context must be boolean")
        total = self._remote_total_windows
        if total is None or self._remote_pending is not None:
            raise ValueError("Remote attention requires completed pre slabs")
        if (self._remote_ready_windows != total or self._remote_consumed
                or self._remote_context_failed):
            raise ValueError("Remote attention inputs are not ready")
        keys = self._remote_prefix_keys + total
        if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
                or mask.shape != (12, total, keys)):
            raise ValueError("Remote attention mask geometry changed")
        before_compile = self.compile_seconds
        graph = self._ensure_engine()
        stage = {"compile": self.compile_seconds - before_compile, "kv_upload": 0.,
                 "query_pack": 0., "infer_readback": 0., "scatter": 0.}
        if retain_context:
            stage.update(context_validate=0., context_copy=0.)
            checker = self._prepare_context_handoff(graph)
            stage["compile"] = self.compile_seconds - before_compile
        else:
            checker = None
            dense = np.empty((1, total * 128, 1536), np.uint16)
        for start in range(0, total, 16):
            started = time.perf_counter()
            count = min(16, total - start)
            ids = self._pack_mask(mask, start, count, keys)
            stage["query_pack"] += time.perf_counter() - started
            started = time.perf_counter()
            feeds = {"Q": self._remote_q_views[start // 16], "K": self.roots[0],
                     "V": self.roots[1], "blocks": ov.Tensor(ids, shared_memory=False)}
            if retain_context:
                try:
                    graph.infer(feeds, read=())
                    stage["infer_readback"] += time.perf_counter() - started
                    started = time.perf_counter()
                    flag = checker.infer(
                        {"attention": self._remote_context_input}, read=("flags",)
                    )["flags"]
                    self._validate_context_flag(flag)
                    stage["context_validate"] += time.perf_counter() - started
                    started = time.perf_counter()
                    # The output is a distinct remote allocation.  Copy only
                    # after the checker has synchronously accepted its bytes.
                    self._remote_q_views[start // 16].copy_from(
                        self._remote_context_output
                    )
                    stage["context_copy"] += time.perf_counter() - started
                except Exception:
                    self._remote_context_failed = True
                    self._remote_context_ready = False
                    raise
            else:
                result = graph.infer(feeds, read=("attention",))["attention"]
                stage["infer_readback"] += time.perf_counter() - started
                started = time.perf_counter()
                indices = window_indices((4, *self.geometry.token_hw), start, count)
                dense[0, indices] = result[:, :count].transpose(1, 2, 0, 3).reshape(-1, 1536)
                stage["scatter"] += time.perf_counter() - started
        self._remote_consumed = True
        self._remote_context_ready = retain_context
        self._remote_context_next_start = 0
        self.last_seconds = stage
        return None if retain_context else dense

    def _pack_mask(self, mask, start, count, keys):
        selected = np.zeros((12, self.queries, self.max_keys), np.bool_)
        selected[:, :count, :keys] = mask[:, start:start + count]
        # Padded query rows must have a valid ordered ID, but their inactive
        # tail remains false so the fixed-capacity graph never selects stale K/V.
        selected[:, count:, 0] = True
        return self.kernel.pack(selected, capacity=self.max_keys)

    def infer(self, q, k, v, mask):
        from window_pre import window_indices
        ov, kernel = self.kernel.ov, self.kernel
        windows = 2 * self.geometry.spatial_windows
        if (q.shape != (windows, 128, 1536) or k.shape != v.shape
                or len(k.shape) != 3 or k.shape[1:] != (128, 1536)
                or not 1 <= len(k) <= self.max_keys
                or mask.shape != (12, windows, len(k)) or mask.dtype != np.bool_
                or any(value.dtype != np.uint16 for value in (q, k, v))):
            raise ValueError("Canonical BF16 attention geometry changed")
        keys = len(k)
        # Validate both complete K/V operands before compilation or any remote
        # root mutation.  Non-contiguous legacy inputs retain their historical
        # acceptance through an explicit writable C-order pack.
        k = self._remote_host_history(np.require(k, requirements=["C", "W"]), "K")
        v = self._remote_host_history(np.require(v, requirements=["C", "W"]), "V")
        stage = {"compile": 0., "kv_upload": 0., "query_pack": 0., "infer_readback": 0., "scatter": 0.}
        before_compile = self.compile_seconds
        graph = self._ensure_engine()
        stage["compile"] = self.compile_seconds - before_compile
        started = time.perf_counter()
        self._copy_kv_prefix(k, v, keys)
        stage["kv_upload"] = time.perf_counter() - started
        dense = np.empty((1, windows * 128, 1536), np.uint16)
        for start in range(0, windows, self.queries):
            started = time.perf_counter()
            count = min(self.queries, windows - start)
            query = np.zeros((12, self.queries, 128, 128), np.uint16)
            query[:, :count] = q[start:start + count].reshape(count, 128, 12, 128).transpose(2, 0, 1, 3)
            ids = self._pack_mask(mask, start, count, keys)
            feeds = {"Q": bits_tensor(query, kernel.STORAGE, share=True), "K": self.roots[0], "V": self.roots[1],
                     "blocks": ov.Tensor(ids, shared_memory=False)}
            stage["query_pack"] += time.perf_counter() - started
            started = time.perf_counter()
            result = graph.infer(feeds, read=("attention",))["attention"]
            stage["infer_readback"] += time.perf_counter() - started
            started = time.perf_counter()
            indices = window_indices((4, *self.geometry.token_hw), start, count)
            dense[0, indices] = result[:, :count].transpose(1, 2, 0, 3).reshape(-1, 1536)
            stage["scatter"] += time.perf_counter() - started
        self.last_seconds = stage
        return dense
