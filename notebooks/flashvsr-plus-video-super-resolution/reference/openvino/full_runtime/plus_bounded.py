"""Bound independent Plus token work without truncating attention context.

The inference callback owns typed I/O and resident weights. These schedulers
only gather/scatter token rows; they never reinterpret or cast storage values.
"""
import numpy as np

from window_pre import GATES, LANES, window_indices
from plus_tokens import RemoteTokenState


def _rows(rows):
    if type(rows) is not int or not 128 <= rows <= 8192 or rows % 128:
        raise ValueError("Expected128..8192 rows in complete128-token windows")


def _pad(value, indices, rows):
    value = np.asarray(value)
    if value.ndim < 2 or value.shape[0] != 1:
        raise ValueError("Expected a single token batch")
    result = np.zeros((1, rows, *value.shape[2:]), dtype=value.dtype)
    result[:, :len(indices)] = value[:, indices]
    return result


def _remote_post_geometry(context_geometry, tokens):
    if (not isinstance(context_geometry, (tuple, list))
            or len(context_geometry) != 3
            or any(type(value) is not int for value in context_geometry)):
        raise ValueError("Remote post requires integer (f, token_h, token_w) geometry")
    f, h, w = context_geometry
    if (f != 4 or h < 8 or w < 8 or h % 8 or w % 8
            or max(h, w) > 240 or h * w > 136 * 240):
        raise ValueError("Remote post geometry is outside reviewed f4 scope")
    expected = f * h * w
    if expected != tokens or expected % 128:
        raise ValueError("Remote post geometry does not match canonical x")
    return (f, h, w), expected // 128


def _validate_post_out(feeds, shape, out):
    if out is None:
        return
    if not isinstance(out, np.ndarray):
        raise ValueError("Post out must be a NumPy array")
    if not isinstance(feeds["x"], np.ndarray):
        raise ValueError("Post out requires a NumPy x feed")
    if out.shape != shape:
        raise ValueError("Post out shape differs from x")
    if out.dtype != feeds["x"].dtype:
        raise ValueError("Post out dtype must match x")
    for name, value in feeds.items():
        if not isinstance(value, np.ndarray) or not np.shares_memory(out, value):
            continue
        if name == "x" and out is value:
            continue
        raise ValueError("Post out aliases input: " + name)
    if not out.flags.c_contiguous:
        raise ValueError("Post out must be C-contiguous")
    if not out.flags.writeable:
        raise ValueError("Post out must be writable")


def _validate_cross(feeds):
    for name in ("cross_k", "cross_v"):
        value = feeds[name]
        shape_of = getattr(value, "get_shape", None)
        actual = tuple(shape_of()) if callable(shape_of) else np.shape(value)
        if actual != (1, 512, 1536):
            raise ValueError("Post must retain the complete text context")


def _infer_post_remote(infer, feeds, rows, *, out, context_geometry):
    if rows != 4096:
        raise ValueError("Remote post requires rows=4096")
    if not callable(infer):
        raise ValueError("Remote post requires the existing infer callback")
    if not isinstance(feeds, dict) or "self_context" in feeds:
        raise ValueError("Remote post does not accept dense self_context")
    if not {"x", "cross_k", "cross_v"} <= set(feeds):
        raise ValueError("Remote post requires x and complete text context")
    x = feeds.get("x")
    tokens = isinstance(x, RemoteTokenState)
    if not tokens and (not isinstance(x, np.ndarray) or x.ndim != 3 or x.shape[0] != 1
            or x.shape[2] != 1536 or x.dtype != np.uint16
            or not x.flags.c_contiguous):
        raise ValueError("Remote post x must be contiguous native-BF16 uint16 storage")
    geometry, windows = _remote_post_geometry(context_geometry, x.shape[1])
    if tokens and (x.geometry != geometry or out is not None):
        raise ValueError("Remote token state must match geometry and owns its destination")
    _validate_cross(feeds)
    if not tokens:
        _validate_post_out(feeds, x.shape, out)
    result = x if tokens else out if out is not None else np.empty_like(x)
    slab_windows = rows // 128
    for start in range(0, windows, slab_windows):
        count = min(slab_windows, windows - start)
        if tokens:
            slab_x = x.view(start, count)
        else:
            indices = window_indices(geometry, start, count)
            valid = count * 128
            slab_x = np.zeros((1, rows, 1536), dtype=x.dtype)
            slab_x[:, :valid] = x[:, indices]
            slab_x.flags.writeable = False
        slab = dict(feeds)
        slab["x"] = slab_x
        values = infer(slab, start, count)
        if tokens:
            if not isinstance(values, dict) or values:
                raise ValueError("Remote token post must not read back x_out")
            # The callback has waited; copy the distinct output slab to only
            # its valid root rows. Never alias infer inputs and outputs.
            x.commit(start, count)
            continue
        if not isinstance(values, dict) or set(values) != {"x_out"}:
            raise ValueError("Remote bounded post output ABI changed")
        value = values["x_out"]
        if (not isinstance(value, np.ndarray) or value.shape != (1, rows, 1536)
                or value.dtype != x.dtype):
            raise ValueError("Remote bounded post output ABI changed")
        result[:, indices] = value[:, :valid]
    return {"x_out": result}


def infer_pre(infer, feeds, geometry, rows, *, include_residual=True, remote=False,
              read_kv=True, kv_start=0):
    """Evaluate windows in source partition order, including a masked tail.

    ``remote=True`` is the narrow resident-Q path: ``infer`` receives
    ``(slab, start, count)`` and returns the public host values except for
    ``q_windows``.  The callback owns the remote Q/K/V targets and must read
    the finite scalars before returning canonical ``k_windows`` and
    ``v_windows``. On the terminal call ``read_kv=False`` omits only those
    host copies: GPU K/V, finite checks and slab completion remain mandatory.
    With ``kv_start=windows//2`` the local-history ABI omits K/V only for
    wholly preceding slabs; a crossing slab keeps its full row shape and is
    cropped into the contiguous suffix. Pooled outputs remain full-width.
    The default callback ABI and output set are unchanged.
    """
    _rows(rows)
    if type(include_residual) is not bool:
        raise ValueError("include_residual must be boolean")
    if type(remote) is not bool:
        raise ValueError("remote must be boolean")
    if type(read_kv) is not bool:
        raise ValueError("read_kv must be boolean")
    if type(kv_start) is not int:
        raise ValueError("kv_start must be an integer")
    if not remote and not read_kv:
        raise ValueError("read_kv=False requires remote pre")
    f, h, w = geometry
    if f != 4 or h % 8 or w % 8 or min(h, w) < 8:
        raise ValueError("Expected complete Plus f4 windows")
    tokens, windows = f * h * w, f * h * w // 128
    suffix_start = windows // 2
    if kv_start < 0 or kv_start > windows:
        raise ValueError("kv_start is outside the current windows")
    if kv_start and (not remote or not read_kv or kv_start != suffix_start):
        raise ValueError("nonzero kv_start requires remote local-history suffix")
    token_state = isinstance(feeds["x"], RemoteTokenState)
    if token_state and (not remote or include_residual or rows != 4096
                        or feeds["x"].geometry != tuple(geometry)):
        raise ValueError("Remote tokens require matching remote pre without host residual")
    for name, shape in (("x", (1, tokens, 1536)),
                        ("rope_cos", (1, tokens, 1, 64)),
                        ("rope_sin", (1, tokens, 1, 64))):
        if np.shape(feeds[name]) != shape:
            raise ValueError("Unexpected pre input shape: " + name)
    outputs = {}
    lanes = tuple(name for name in LANES if not remote or name != "q_windows")
    if not read_kv:
        lanes = tuple(name for name in lanes if name not in ("k_windows", "v_windows"))
    for start in range(0, windows, rows // 128):
        count = min(rows // 128, windows - start)
        indices = window_indices(geometry, start, count)
        slab = dict(feeds)
        for name in ("x", "rope_cos", "rope_sin"):
            slab[name] = (feeds[name].view(start, count) if token_state and name == "x"
                          else _pad(feeds[name], indices, rows))
        values = infer(slab, start, count) if remote else infer(slab)
        expected = {*lanes, *GATES, "q_pooled", "k_pooled"}
        # In suffix mode the complete GPU slab still runs, but a slab wholly
        # before the explicit boundary must not expose host K/V keys.
        if remote and read_kv and kv_start and start + count <= kv_start:
            expected -= {"k_windows", "v_windows"}
        if set(values) != expected:
            raise ValueError("Bounded pre output ABI changed")
        for name, value in values.items():
            value = np.asarray(value)
            if name in LANES:
                shape, axis = (rows // 128, 128, 1536), 0
            elif name.endswith("_pooled"):
                shape, axis = (12, rows // 128, 128), 1
            else:
                shape, axis = (1, 1, 1536), None
            if value.shape != shape:
                raise ValueError("Unexpected bounded pre output: " + name)
            if axis is None:
                if name in outputs and not np.array_equal(outputs[name], value):
                    raise ValueError("Token-independent modulation changed between slabs")
                outputs[name] = value.copy()
            else:
                full = list(shape)
                offset = kv_start if name in ("k_windows", "v_windows") else 0
                full[axis] = windows - offset
                begin = max(start, offset)
                if begin >= start + count:
                    raise ValueError("Suffix K/V returned outside its boundary")
                if name not in outputs:
                    outputs[name] = np.empty(full, value.dtype)
                target, source = [slice(None)] * 3, [slice(None)] * 3
                target[axis] = slice(begin - offset, start + count - offset)
                source[axis] = slice(begin - start, count)
                outputs[name][tuple(target)] = value[tuple(source)]
    # The unbounded pre residual is exactly its input, not a computed tensor.
    if include_residual:
        outputs["residual"] = np.array(feeds["x"], copy=True)
    return outputs


def infer_post(infer, feeds, rows, *, out=None, remote_context=False,
               context_geometry=None):
    """Bound query/FFN rows, keeping all512 text keys and values in each call.

    ``out`` is an optional owning NumPy destination.  It may be ``feeds["x"]``:
    each callback consumes its read-only input slab before that slab is written
    back, so independent post rows can safely reuse the input allocation.

    Remote mode requires ``rows=4096`` and
    ``context_geometry=(4, token_h, token_w)``.  The existing callback is
    called as ``infer(slab, start_window, window_count)`` and returns
    ``{"x_out": slab_output}``; it owns the two BF16 context slots.
    Canonical x tokens are gathered in source window order, the final slab is
    zero-padded, and outputs are scattered back without a dense host context.
    A RemoteTokenState x instead supplies the same window-ordered GPU ROIs;
    the synchronous callback returns {} and commits its existing output slab.
    """
    _rows(rows)
    if type(remote_context) is not bool:
        raise ValueError("remote_context must be boolean")
    if remote_context:
        return _infer_post_remote(
            infer, feeds, rows, out=out, context_geometry=context_geometry,
        )
    if context_geometry is not None:
        raise ValueError("context_geometry is only valid for remote_context=True")
    shape = np.shape(feeds["x"])
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 1536 or shape[1] < 1:
        raise ValueError("Unexpected post token shape")
    if np.shape(feeds["self_context"]) != shape:
        raise ValueError("Post self context shape differs")
    _validate_cross(feeds)
    _validate_post_out(feeds, shape, out)
    result = None
    for start in range(0, shape[1], rows):
        stop = min(start + rows, shape[1])
        slab = dict(feeds)
        for name in ("x", "self_context"):
            # Post rows are contiguous, unlike pre windows. Borrow a read-only
            # view for full slabs; allocate/copy only the final padded tail.
            value = feeds[name][:, start:stop].view()
            if stop - start < rows:
                padded = np.zeros((1, rows, 1536), dtype=value.dtype)
                padded[:, :stop - start] = value
                value = padded
            value.flags.writeable = False
            slab[name] = value
        values = infer(slab)
        if not isinstance(values, dict) or set(values) != {"x_out"}:
            raise ValueError("Bounded post output ABI changed")
        value = values["x_out"]
        if np.shape(value) != (1, rows, 1536):
            raise ValueError("Bounded post output ABI changed")
        if out is not None:
            if not isinstance(value, np.ndarray) or value.dtype != out.dtype:
                raise ValueError("Bounded post output dtype does not match out")
            out[:, start:stop] = value[:, :stop - start]
            continue
        if result is None:
            result = np.empty(shape, value.dtype)
        result[:, start:stop] = value[:, :stop - start]
    return {"x_out": out if out is not None else result}
