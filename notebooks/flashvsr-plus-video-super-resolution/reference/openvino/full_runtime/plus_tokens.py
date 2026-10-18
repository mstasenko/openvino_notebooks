"""Reusable window-major BF16 tokens for the resident Plus scheduler.

The caller supplies the existing BF16 post graph.  Its ``x_out`` buffer is
the only output slab; this class owns one padded window-major BF16 root and
does only bounded storage-bit gathers/scatters around synchronous graph work.
"""

import numpy as np

from plus_resident import bits_tensor, tensor_bits
from plus_storage import is_finite_bf16, storage_tensor
from window_pre import window_indices


CHANNELS = 1536
WINDOW = 128
ROWS = 4096
SLAB_WINDOWS = ROWS // WINDOW


def _geometry(value, tokens):
    from plus_bounded import _remote_post_geometry

    geometry, windows = _remote_post_geometry(value, tokens)
    return tuple(geometry), int(windows)


def _range(geometry, windows, start, count):
    if (type(start) is not int or type(count) is not int or start < 0
            or count <= 0 or count > SLAB_WINDOWS or start % SLAB_WINDOWS
            or start + count > windows):
        raise ValueError("token slabs must be aligned 32-window ranges")
    if start + count < windows and count != SLAB_WINDOWS:
        raise ValueError("only the terminal token slab may be short")
    return count * WINDOW


def _roi(ov, root, token_start, token_count):
    if type(token_start) is not int or type(token_count) is not int:
        raise ValueError("ROI coordinates must be integers")
    if token_start < 0 or token_count <= 0:
        raise ValueError("ROI coordinates must be positive")
    return ov.RemoteTensor(
        root,
        ov.Coordinate([0, token_start, 0]),
        ov.Coordinate([1, token_start + token_count, CHANNELS]),
    )


class RemoteTokenState:
    """Own one padded root while reusing a graph's distinct ``x_out`` slab."""

    def __init__(self, post, geometry, rows=ROWS):
        import openvino as ov

        if type(rows) is not int or rows != ROWS:
            raise ValueError("Remote token state requires rows=4096")
        if (not hasattr(post, "context") or not hasattr(post, "buffers")
                or "x_out" not in post.buffers or not hasattr(post, "inputs")):
            raise ValueError("post graph must expose context, x input, and x_out buffer")
        output = post.buffers["x_out"]
        try:
            port = post.inputs["x"]
            if (tuple(port.shape) != (1, rows, CHANNELS)
                    or port.get_element_type() != ov.Type.bf16):
                raise ValueError("post x must be the reviewed4096-row BF16 slab")
            # Validate the existing output against the graph's BF16 x input;
            # ResidentGraph also checks same-context ownership here.
            post._validate_tensor("x", output, post.inputs["x"])
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("post graph must validate its BF16 x_out buffer") from error
        if (not isinstance(geometry, (tuple, list)) or len(geometry) != 3
                or any(type(item) is not int for item in geometry)):
            raise ValueError("geometry must be integer (4, token_h, token_w)")
        geometry, windows = _geometry(geometry, geometry[0] * geometry[1] * geometry[2])
        padded_windows = ((windows + SLAB_WINDOWS - 1) // SLAB_WINDOWS) * SLAB_WINDOWS
        root_shape = (1, padded_windows * WINDOW, CHANNELS)
        context = post.context
        root = context.create_tensor(ov.Type.bf16, ov.Shape(root_shape), {})
        if root is output:
            raise ValueError("post x_out must remain distinct from token root")
        if (tuple(root.get_shape()) != root_shape
                or int(root.get_byte_size()) != 2 * int(np.prod(root_shape))):
            raise ValueError("token root shape/storage changed")
        get_type = getattr(root, "get_element_type", None)
        if callable(get_type) and get_type() != ov.Type.bf16:
            raise ValueError("token root must be native BF16")
        self.ov, self.context = ov, context
        self.geometry = geometry
        self.tokens = geometry[0] * geometry[1] * geometry[2]
        self.shape = (1, self.tokens, CHANNELS)
        self.windows, self.padded_windows, self.rows = windows, padded_windows, rows
        self.root, self.output_slab = root, output
        self._loaded = False
        self._commit_next = 0
        self._pass_complete = True

    def load(self, canonical, condition=None):
        if self._loaded and not self._pass_complete:
            raise ValueError("cannot load while a post pass is incomplete")
        # A conditioned reload is transactional from the public state
        # contract's perspective: any validation or copy failure must not
        # expose the previous root as if it were still current.
        conditioned = condition is not None
        if conditioned:
            self._loaded = False
            self._pass_complete = False
        canonical = np.asarray(canonical)
        if (canonical.dtype != np.uint16 or canonical.shape != (1, self.tokens, CHANNELS)
                or not canonical.flags.c_contiguous):
            raise ValueError("canonical x must be contiguous BF16 bits [1,tokens,1536]")
        # Validate all source bits before the first root write.
        bits_tensor(canonical, share=True)
        if conditioned:
            condition = np.asarray(condition)
            if (condition.dtype != np.uint16 or condition.shape != canonical.shape
                    or not condition.flags.c_contiguous):
                raise ValueError("condition must be contiguous BF16 bits matching canonical x")
            if not is_finite_bf16(condition):
                raise ValueError("condition contains non-finite BF16 bits")

        # Keep one bounded upload slab for both paths.  The conditioned path
        # decodes storage bits to F32, adds, and rounds through the shared
        # storage helper; no numeric F16 conversion is involved.
        slab = np.zeros((1, self.rows, CHANNELS), dtype=np.uint16)
        for start in range(0, self.windows, SLAB_WINDOWS):
            count = min(SLAB_WINDOWS, self.windows - start)
            valid = count * WINDOW
            indices = window_indices(self.geometry, start, count)
            slab.fill(0)
            if conditioned:
                left = (canonical[:, indices].astype(np.uint32) << np.uint32(16)).view(np.float32)
                right = (condition[:, indices].astype(np.uint32) << np.uint32(16)).view(np.float32)
                with np.errstate(over="ignore", invalid="ignore"):
                    np.add(left, right, out=left)
                # storage_tensor is the single source of BF16 RNE and rejects
                # non-finite sums and finite values overflowing BF16.
                slab[:, :valid] = tensor_bits(
                    storage_tensor(left, "bf16"), copy=False
                )
            else:
                slab[:, :valid] = canonical[:, indices]
            bits_tensor(slab, share=True).copy_to(
                _roi(self.ov, self.root, start * WINDOW, self.rows)
            )
        self._loaded = True
        self._commit_next = 0
        self._pass_complete = False

    def view(self, start, count):
        if not self._loaded:
            raise ValueError("load must precede view")
        _range(self.geometry, self.windows, start, count)
        return _roi(self.ov, self.root, start * WINDOW, self.rows)

    def commit(self, start, count):
        if not self._loaded:
            raise ValueError("load must precede commit")
        valid = _range(self.geometry, self.windows, start, count)
        if start != self._commit_next:
            raise ValueError("post commits must be sequential 32-window ranges")
        destination = _roi(self.ov, self.root, start * WINDOW, valid)
        source = (self.output_slab if count == SLAB_WINDOWS
                  else _roi(self.ov, self.output_slab, 0, valid))
        destination.copy_from(source)
        self._commit_next = 0 if start + count == self.windows else start + count
        self._pass_complete = self._commit_next == 0

    def read(self):
        if not self._loaded or not self._pass_complete:
            raise ValueError("all post slabs must be committed before read")
        result = np.empty((1, self.tokens, CHANNELS), dtype=np.uint16)
        slab = np.zeros((1, self.rows, CHANNELS), dtype=np.uint16)
        for start in range(0, self.windows, SLAB_WINDOWS):
            count = min(SLAB_WINDOWS, self.windows - start)
            slab.fill(0)
            host = bits_tensor(slab, share=True)
            _roi(self.ov, self.root, start * WINDOW, self.rows).copy_to(host)
            result[:, window_indices(self.geometry, start, count)] = \
                tensor_bits(host, copy=False)[:, :count * WINDOW]
        return result

