"""Exact bounded NVMe history spill for the research CUDA/OpenVINO controls."""
import mmap
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
import weakref
import numpy as np
from reference_paths import SCRATCH_ROOT


MAX_LAYERS = 30
DISK_RESERVE_BYTES = 16 << 30
FALLBACK_SCRATCH_ROOT = SCRATCH_ROOT
_MAX_NBYTES = (1 << 63) - 1
PREFETCH_RANGE_BYTES = 2 << 20


def _scratch_root():
    """Return the caller-selected, bounded scratch parent.

    The fallback is deliberately one exact experiment directory; generic
    system temporary directories are not acceptable for multi-gigabyte spill.
    """

    raw = os.environ.get('TMPDIR')
    root = Path(raw) if raw else FALLBACK_SCRATCH_ROOT
    if not root.is_absolute():
        raise ValueError('TMPDIR must be an absolute scratch directory')
    root = root.resolve()
    if root in (Path('/tmp'), Path('/var/tmp')):
        raise ValueError('generic system temporary directory is not allowed')
    if not root.is_dir():
        raise FileNotFoundError(f'exact scratch directory does not exist: {root}')
    if not os.access(root, os.W_OK | os.X_OK):
        raise PermissionError(f'exact scratch directory is not writable: {root}')
    return root


def _validate_sync_eviction_api(sync_writeback):
    if not sync_writeback:
        return
    required = ('posix_fadvise', 'POSIX_FADV_DONTNEED')
    if any(not hasattr(os, name) for name in required):
        raise OSError('synchronous spill requires os.posix_fadvise(DONTNEED)')
    if sys.platform == 'linux' and not callable(getattr(os, 'fsync', None)):
        raise OSError('Linux synchronous spill requires os.fsync')
    if not hasattr(mmap.mmap, 'madvise') or not hasattr(mmap, 'MADV_DONTNEED'):
        raise OSError('synchronous spill requires mmap MADV_DONTNEED')


def _shape_tuple(value, label):
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f'{label} must be a non-empty shape sequence')
    shape = tuple(value)
    if any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError(f'{label} must contain positive integer dimensions')
    return shape


def _checked_nbytes(shape, itemsize):
    elements = math.prod(shape)
    if elements > _MAX_NBYTES // itemsize:
        raise OverflowError('history shape byte count overflows the supported file size')
    return elements * itemsize


def _normalize_max_shapes(value, layers):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError('max_shapes must be a mapping of layer to (K,V) shapes')
    result = {}
    for raw_layer, pair in value.items():
        if type(raw_layer) is not int or not 0 <= raw_layer < layers:
            raise ValueError('max_shapes contains an invalid layer')
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('max_shapes values must contain K and V shapes')
        result[raw_layer] = (
            _shape_tuple(pair[0], 'maximum K shape'),
            _shape_tuple(pair[1], 'maximum V shape'),
        )
    return result


def _normalize_active_axes(value, max_shapes):
    if not max_shapes:
        if value is not None:
            raise ValueError('active_axes requires max_shapes')
        return {}
    if not isinstance(value, Mapping) or set(value) != set(max_shapes):
        raise ValueError('active_axes must name exactly the max_shapes layers')
    result = {}
    for layer, raw_axis in value.items():
        if type(raw_axis) is not int:
            raise ValueError('active axis must be an integer')
        k_shape, v_shape = max_shapes[layer]
        if len(k_shape) != len(v_shape) or not 0 <= raw_axis < len(k_shape):
            raise ValueError('active axis is outside the K/V shape rank')
        result[layer] = raw_axis
    return result


class DiskKv:
    def __init__(self, torch=None, layers=8, prefix='otc-cuda-kv-',
                 sync_writeback=True, required_bytes=None, *,
                 storage_dtype='fp16', max_shapes=None, active_axes=None):
        if type(layers) is not int or not 0 <= layers <= MAX_LAYERS:
            raise ValueError(f'At most {MAX_LAYERS} exact history slots')
        if prefix not in ('otc-cuda-kv-', 'otc-ov-kv-'):
            raise ValueError('Reviewed exact scratch prefix required')
        if type(sync_writeback) is not bool:
            raise ValueError('Boolean scratch writeback mode required')
        if (required_bytes is not None
                and (type(required_bytes) is not int or required_bytes < 0)):
            raise ValueError('required_bytes must be a nonnegative integer')
        if type(storage_dtype) is not str or storage_dtype not in ('fp16', 'bf16'):
            raise ValueError("storage_dtype must be 'fp16' or 'bf16'")
        _validate_sync_eviction_api(sync_writeback)
        normalized_max = _normalize_max_shapes(max_shapes, layers)
        normalized_axes = _normalize_active_axes(active_axes, normalized_max)
        self._storage_dtype = storage_dtype
        self.storage_dtype = storage_dtype
        self._numpy_dtype = np.dtype(np.uint16 if storage_dtype == 'bf16' else np.float16)
        self._itemsize = self._numpy_dtype.itemsize
        self.max_shapes = normalized_max
        self.active_axes = normalized_axes
        self.max_storage_bytes = sum(
            _checked_nbytes(shape, self._itemsize)
            for pair in normalized_max.values() for shape in pair
        )
        if self.max_storage_bytes > _MAX_NBYTES:
            raise OverflowError('declared history files exceed the supported total size')
        if (required_bytes is not None
                and required_bytes < self.max_storage_bytes):
            raise ValueError('required_bytes is smaller than declared maximum files')
        effective_required = (self.max_storage_bytes if required_bytes is None
                              and self.max_storage_bytes else required_bytes)
        scratch = _scratch_root()
        required = DISK_RESERVE_BYTES + (0 if effective_required is None else effective_required)
        if shutil.disk_usage(scratch).free < required:
            raise MemoryError(
                f'insufficient scratch space: need {required} bytes including reserve'
            )
        self.sync_writeback = sync_writeback
        self.torch, self.layers = torch, layers
        self.required_bytes = effective_required
        self.scratch_root = scratch
        self.root = Path(tempfile.mkdtemp(prefix=prefix, dir=str(scratch)))
        self.slots = {}
        self._active_shapes = {}
        self._exports = {}
        self._closed = False
        self.seconds = 0.
        self.bytes_written = 0

    def _check_layer(self, layer, allow_passthrough=False):
        if type(layer) is not int or layer < 0:
            raise ValueError('Nonnegative integer layer required')
        if layer >= self.layers:
            if allow_passthrough:
                return False
            raise ValueError('Existing history layer index required')
        return True

    def _check_open(self):
        if self._closed:
            raise RuntimeError('History spill is already closed')

    def _validate_active_shape(self, layer, name, shape):
        shape = _shape_tuple(shape, f'active {name} shape')
        maximum = self.max_shapes.get(layer)
        if maximum is not None:
            axis = self.active_axes[layer]
            maximum = maximum[0 if name == 'k' else 1]
            if len(shape) != len(maximum):
                raise ValueError('Active history rank changed')
            if any(index != axis and dim != maximum[index]
                       for index, dim in enumerate(shape)):
                raise ValueError('Active history shape is not a declared prefix')
            if shape[axis] > maximum[axis]:
                raise ValueError('Active history shape exceeds its declared maximum')
        previous = self._active_shapes.get((layer, name))
        if previous is not None:
            if maximum is None:
                if shape != previous:
                    raise ValueError('History shape changed between chronological calls')
            else:
                axis = self.active_axes[layer]
                if any(index != axis and dim != previous[index]
                       for index, dim in enumerate(shape)):
                    raise ValueError('Active history shape changed outside its prefix axis')
                if shape[axis] < previous[axis]:
                    raise ValueError('Active history prefix cannot shrink')
        _checked_nbytes(shape, self._itemsize)
        return shape

    def _validate_array(self, layer, name, array):
        if not isinstance(array, np.ndarray):
            raise ValueError(f'Compact {self._storage_dtype.upper()} NumPy histories required')
        if array.dtype != self._numpy_dtype or not array.flags.c_contiguous:
            raise ValueError(f'Compact {self._storage_dtype.upper()} NumPy histories required')
        self._validate_active_shape(layer, name, array.shape)

    def _active_view(self, layer, name, slot, shape):
        maximum = self.max_shapes.get(layer)
        if maximum is None or tuple(shape) == tuple(slot.shape):
            return slot
        axis = self.active_axes[layer]
        slices = [slice(None)] * slot.ndim
        slices[axis] = slice(0, shape[axis])
        view = slot[tuple(slices)]
        if tuple(view.shape) != tuple(shape):
            raise ValueError('Active history view shape mismatch')
        return view

    def _ensure_slot(self, layer, name, active_shape):
        key = (layer, name)
        slot = self.slots.get(key)
        if slot is not None:
            return slot
        maximum = self.max_shapes.get(layer)
        capacity = (maximum[0 if name == 'k' else 1]
                    if maximum is not None else tuple(active_shape))
        nbytes = _checked_nbytes(capacity, self._itemsize)
        path = self.root / f'{layer}-{name}.raw'
        if path.exists():
            raise FileExistsError(f'owned history path already exists: {path}')
        with path.open('xb') as stream:
            stream.truncate(nbytes)
        slot = np.memmap(path, mode='r+', dtype=self._numpy_dtype, shape=capacity)
        self.slots[key] = slot
        return slot

    def _sync_slot(self, layer, name, slot, byte_count):
        if not self.sync_writeback:
            return
        path = self.root / f'{layer}-{name}.raw'
        with path.open('r+b') as stream:
            # Linux MAP_SHARED writes use the file's page cache. fsync waits
            # for writeback without mmap.flush's GIL stall, allowing the one
            # I/O worker to overlap GPU submission. Retain msync elsewhere.
            if sys.platform == 'linux':
                os.fsync(stream.fileno())
            else:
                slot.flush()
            # No eviction after failed writeback. Consumers must finish before
            # retire/close; a live mapping remains valid after MADV_DONTNEED.
            slot._mmap.madvise(mmap.MADV_DONTNEED)
            os.posix_fadvise(
                stream.fileno(), 0, byte_count, os.POSIX_FADV_DONTNEED
            )

    def _external_view(self, key, value):
        if self._storage_dtype != 'bf16':
            return value
        # Keep the internal mapping distinct from the caller's exported view.
        # This lets us fail closed before mmap.close instead of invalidating a
        # live NumPy/Torch consumer that still aliases the raw BF16 file.
        # Ordinary ndarray slices retain this export as their base. A memmap
        # subclass slice instead skips it, defeating the live-consumer guard.
        view = value.view(np.ndarray)
        self._exports.setdefault(key, []).append(weakref.ref(view))
        return view

    def _live_exports(self, key):
        refs = self._exports.get(key, [])
        live = [ref for ref in refs if ref() is not None]
        if live:
            self._exports[key] = live
        else:
            self._exports.pop(key, None)
        return live

    def prefetch(self, layer, *, active_bytes=None):
        """Read-ahead hint for one existing next-layer pair; no new buffers.

        With ``active_bytes=(k_bytes, v_bytes)``, issue exact 2 MiB
        ``POSIX_FADV_WILLNEED`` prefix hints for initialized files.  This
        opt-in path requires ``os.posix_fadvise`` and raises ``OSError`` when
        the platform cannot provide it; the default MADV path is unchanged.
        """
        if type(layer) is not int or layer < 0:
            raise ValueError('Nonnegative integer layer required')
        # Keep this helper usable with the tiny object.__new__ fakes used by
        # the CPU contract tests, which intentionally provide only slots.
        if hasattr(self, 'layers') and layer >= self.layers:
            if active_bytes is not None:
                raise ValueError('Existing history layer index required')
            return
        if active_bytes is not None:
            if (not isinstance(active_bytes, (tuple, list))
                    or len(active_bytes) != 2
                    or any(type(value) is not int or value <= 0
                           for value in active_bytes)):
                raise ValueError('active_bytes must contain two positive integers')
            if not callable(getattr(os, 'posix_fadvise', None)) or not hasattr(
                    os, 'POSIX_FADV_WILLNEED'):
                raise OSError(
                    'ranged prefetch requires os.posix_fadvise(WILLNEED)'
                )
            self._check_open()
            names = ('k', 'v')
            slots = tuple(self.slots.get((layer, name)) for name in names)
            if any(slot is None for slot in slots):
                raise ValueError(
                    'ranged prefetch requires initialized K/V slots'
                )
            for name, slot, requested in zip(names, slots, active_bytes):
                extent = slot.nbytes
                if (extent > _MAX_NBYTES or requested > extent):
                    raise ValueError(
                        f'active {name} bytes exceed the declared slot extent'
                    )
                path = self.root / f'{layer}-{name}.raw'
                with path.open('rb') as stream:
                    if os.fstat(stream.fileno()).st_size != extent:
                        raise OSError('history file extent changed before prefetch')
                    for offset in range(0, requested, PREFETCH_RANGE_BYTES):
                        length = min(PREFETCH_RANGE_BYTES, requested - offset)
                        os.posix_fadvise(
                            stream.fileno(), offset, length,
                            os.POSIX_FADV_WILLNEED,
                        )
            return
        for name in ('k','v'):
            slot = self.slots.get((layer,name))
            if slot is not None:
                slot._mmap.madvise(mmap.MADV_WILLNEED)

    def allocate_arrays(self, layer):
        """Expose declared BF16 backing for a caller that fills it directly.

        The caller owns valid-prefix bookkeeping and must call flush after
        writes, then release every exported view before retire/close.
        """
        self._check_layer(layer)
        self._check_open()
        if self._storage_dtype != 'bf16' or layer not in self.max_shapes:
            raise ValueError('Direct backing requires declared maximum BF16 shapes')
        return tuple(self._external_view(
            (layer, name), self._ensure_slot(layer, name, shape))
            for name, shape in zip(('k', 'v'), self.max_shapes[layer]))

    def flush(self, layer):
        """Durably write and evict only this layer's complete owned mappings."""
        self._check_layer(layer)
        self._check_open()
        if any((layer, name) not in self.slots for name in ('k', 'v')):
            raise ValueError('Both history backing slots must exist before flush')
        started = time.perf_counter()
        for name in ('k', 'v'):
            slot = self.slots[(layer, name)]
            self._sync_slot(layer, name, slot, slot.nbytes)
        self.seconds += time.perf_counter() - started

    def store(self, layer, k, v):
        if not self._check_layer(layer, allow_passthrough=True):
            return k, v
        self._check_open()
        if self.torch is None:
            raise ValueError('Use store_arrays for NumPy histories')
        if any(t.device.type != 'cpu' or not t.is_contiguous() for t in (k, v)):
            raise ValueError('Compact CPU history required')
        if self._storage_dtype == 'bf16':
            if any(t.dtype != self.torch.bfloat16 for t in (k, v)):
                raise ValueError('BF16 Torch histories are required')
            arrays = tuple(t.detach().view(self.torch.uint16).numpy() for t in (k, v))
        else:
            arrays = (k.numpy(), v.numpy())
        return self._torch_views(
            self.store_arrays(layer, *arrays),
            ((layer, 'k'), (layer, 'v')),
        )

    def store_arrays(self, layer, k, v):
        if not self._check_layer(layer, allow_passthrough=True):
            return k, v
        self._check_open()
        self._validate_array(layer, 'k', k)
        self._validate_array(layer, 'v', v)
        started = time.perf_counter()
        values = []
        for name, array in (('k', k), ('v', v)):
            slot = self._ensure_slot(layer, name, array.shape)
            target = self._active_view(layer, name, slot, array.shape)
            np.copyto(target, array, casting='no')
            # A prefix along the source's head/group axis is strided in the
            # fixed-capacity file; advise the complete owned mapping rather
            # than treating its logical nbytes as one contiguous extent.
            self._sync_slot(layer, name, slot, slot.nbytes)
            values.append(self._external_view((layer, name), target))
            self.bytes_written += array.nbytes
        self._active_shapes[(layer, 'k')] = tuple(k.shape)
        self._active_shapes[(layer, 'v')] = tuple(v.shape)
        self.seconds += time.perf_counter() - started
        return tuple(values)

    def _torch_views(self, arrays, keys=None):
        if self.torch is None:
            raise ValueError('Torch views require the torch module')
        if self._storage_dtype == 'bf16':
            result = tuple(self.torch.from_numpy(a).view(self.torch.bfloat16)
                           for a in arrays)
            if keys is not None:
                for key, value in zip(keys, result):
                    self._exports.setdefault(key, []).append(weakref.ref(value))
            return result
        return tuple(self.torch.from_numpy(a) for a in arrays)

    def load_arrays(self, layer, shapes=None):
        """Return the latest raw K/V prefix views without copying.

        For a predeclared maximum, the returned views expose only the active
        prefix.  The caller must finish all consumers before ``retire`` or
        ``close``; an exported view intentionally prevents mmap closure.
        """
        self._check_layer(layer)
        self._check_open()
        current = tuple(self._active_shapes.get((layer, name))
                        for name in ('k', 'v'))
        if any(shape is None for shape in current):
            raise ValueError('No complete active history pair is available')
        if shapes is not None:
            if not isinstance(shapes, (tuple, list)) or len(shapes) != 2:
                raise ValueError('active shapes must contain K and V shapes')
            requested = tuple(_shape_tuple(shape, 'requested active shape')
                              for shape in shapes)
            if requested != current:
                raise ValueError('Requested active shape is not the stored prefix')
        return tuple(self._external_view((layer, name),
                                         self._active_view(layer, name,
                                                           self.slots[(layer, name)], shape))
                     for name, shape in zip(('k', 'v'), current))

    def load(self, layer, shapes=None):
        """Return Torch views of the latest raw pair, including BF16 transport."""
        if self.torch is None:
            raise ValueError('Torch views require the torch module')
        arrays = self.load_arrays(layer, shapes=shapes)
        return self._torch_views(arrays, ((layer, 'k'), (layer, 'v')))

    def retire(self, layer):
        """Release a layer only after its final attention consumer has finished."""
        self._check_layer(layer)
        keys = tuple((layer, name) for name in ('k', 'v'))
        # Check the pair before closing either mapping so a live V consumer
        # cannot leave K half-retired after a failed cleanup attempt.
        for key in keys:
            if self.slots.get(key) is not None and self._live_exports(key):
                raise BufferError(
                    'BF16 history consumer is still live; release views before retire'
                )
        for name in ('k', 'v'):
            key = (layer,name)
            slot = self.slots.get(key)
            if slot is not None:
                slot._mmap.close()
                del self.slots[key]
                self._active_shapes.pop(key, None)
                self._exports.pop(key, None)
            # Only exact files owned by this instance; no recursive cleanup.
            (self.root / f'{layer}-{name}.raw').unlink(missing_ok=True)

    def close(self):
        if self._closed:
            return
        for layer in range(self.layers):
            self.retire(layer)
        self.root.rmdir()
        self._closed = True
