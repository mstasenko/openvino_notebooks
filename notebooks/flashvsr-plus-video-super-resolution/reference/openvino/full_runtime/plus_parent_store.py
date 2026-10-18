"""Bounded, bit-preserving disk storage for staged FlashVSR+ decoder parents.

The decoder's staged phases need to keep a chronological set of BF16 parent
frames while a later phase runs.  This class deliberately stores the raw
``uint16`` representation: it does not interpret BF16 values, make a NumPy
copy, or expose a memory map that could retain the file in the page cache.
"""

import errno
import os
from pathlib import Path
import shutil
import stat
import tempfile

import numpy as np
from reference_paths import SCRATCH_ROOT


DISK_RESERVE_BYTES = 16 << 30
FALLBACK_SCRATCH_ROOT = SCRATCH_ROOT
_MAX_SIGNED_BYTES = (1 << 63) - 1
_MAX_PARENT_SLOTS = 1 << 20  # one byte of completeness metadata per slot
_PARENT_PREFIX = 'otc-plus-parent-'
_UINT16 = np.dtype(np.uint16)
_CACHE_ROOT = SCRATCH_ROOT.parent
_CACHE_ANCHOR = SCRATCH_ROOT.parent


def _scratch_root(scratch_root=None):
    raw = scratch_root
    if raw is None:
        raw = os.environ.get('TMPDIR') or FALLBACK_SCRATCH_ROOT
    if not isinstance(raw, (str, os.PathLike)):
        raise TypeError('scratch_root must be a path')
    root = Path(raw)
    if not root.is_absolute():
        raise ValueError('scratch root must be absolute')
    lexical = Path(os.path.abspath(root))
    for generic in (Path('/tmp'), Path('/var/tmp')):
        try:
            lexical.relative_to(generic)
        except ValueError:
            continue
        raise ValueError('generic temporary directories are not allowed')
    root = root.resolve(strict=True)
    try:
        root.relative_to(_CACHE_ANCHOR)
    except ValueError as exc:
        raise ValueError('scratch root must resolve under the configured extraction root') from exc
    if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise PermissionError(f'exact scratch directory is not writable: {root}')
    if root.stat().st_dev != _CACHE_ROOT.stat().st_dev:
        raise ValueError('scratch root must share the configured extraction filesystem')
    return root


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _shape_tuple(shape):
    if not isinstance(shape, (tuple, list)) or not shape:
        raise ValueError('non-empty integer shape required')
    values = tuple(shape)
    for dimension in values:
        if type(dimension) is not int or dimension <= 0:
            raise ValueError('shape dimensions must be positive integers')
    return values


def _storage_bytes(shape, count):
    shape = _shape_tuple(shape)
    count = _positive_int(count, 'count')
    if count > _MAX_PARENT_SLOTS:
        raise ValueError('parent slot count exceeds bounded metadata limit')
    elements = count
    for dimension in shape:
        elements *= dimension
        if elements > _MAX_SIGNED_BYTES // _UINT16.itemsize:
            raise OverflowError('parent storage size exceeds signed file offset range')
    result = elements * _UINT16.itemsize
    if result <= 0 or result > _MAX_SIGNED_BYTES:
        raise OverflowError('parent storage size exceeds signed file offset range')
    return result


def _check_free(root, required):
    if shutil.disk_usage(root).free < required:
        raise MemoryError(
            f'insufficient scratch space: need {required} bytes including reserve'
        )


def _validate_io_api():
    required = ('fdatasync', 'pwrite', 'preadv', 'posix_fadvise',
                'POSIX_FADV_DONTNEED')
    missing = [name for name in required if not hasattr(os, name)]
    if missing:
        raise OSError(f'parent-store I/O API is unavailable: {missing}')


def _validate_array(array, shape, *, destination=False):
    if not isinstance(array, np.ndarray):
        raise TypeError('a NumPy uint16 array is required')
    if array.dtype != _UINT16:
        raise TypeError('raw BF16 storage requires native uint16 dtype')
    if array.shape != shape:
        raise ValueError(f'array shape must be exactly {shape}')
    if not array.flags.c_contiguous:
        raise ValueError('array must be C-contiguous')
    if destination and not array.flags.writeable:
        raise ValueError('read destination must be writable')


def _write_all(fd, source, offset):
    view = memoryview(source).cast('B')
    done = 0
    while done < len(view):
        try:
            written = os.pwrite(fd, view[done:], offset + done)
        except InterruptedError:
            continue
        if not isinstance(written, int) or written <= 0:
            raise OSError(errno.EIO, 'short parent-store write')
        if written > len(view) - done:
            raise OSError(errno.EIO, 'invalid parent-store write length')
        done += written


def _read_exact(fd, destination, offset):
    view = memoryview(destination).cast('B')
    done = 0
    while done < len(view):
        try:
            read = os.preadv(fd, [view[done:]], offset + done)
        except InterruptedError:
            continue
        if not isinstance(read, int) or read < 0:
            raise OSError(errno.EIO, 'invalid parent-store read length')
        if read == 0:
            raise EOFError('truncated parent-store slot')
        if read > len(view) - done:
            raise OSError(errno.EIO, 'invalid parent-store read length')
        done += read


def _drop_cache(fd, offset, length):
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except AttributeError as exc:
        raise OSError('parent-store cache eviction API is unavailable') from exc


class BF16ParentStore:
    """One chronological raw-BF16 store backed by an anonymous disk file.

    Each slot is written at most once.  ``read_into`` always fills the caller's
    preallocated C-contiguous ``uint16`` destination directly via ``preadv``.
    The per-file check reserves 16 GiB plus this store's bytes.  A controller
    creating unequal front/history stores must sum their bytes and perform a
    joint peak check before creating either file; existing files then reduce
    free space for each subsequent per-file check.
    """

    def __init__(self, shape, count, scratch_root=None):
        _validate_io_api()
        self.shape = _shape_tuple(shape)
        self.count = _positive_int(count, 'count')
        self.storage_bytes = _storage_bytes(self.shape, self.count)
        root = _scratch_root(scratch_root)
        required = DISK_RESERVE_BYTES + self.storage_bytes
        if required > _MAX_SIGNED_BYTES:
            raise OverflowError('parent preflight exceeds signed file offset range')
        _check_free(root, required)
        self.scratch_root = root
        self._written = bytearray(self.count)
        self._closed = False
        self._file = None
        try:
            self._file = tempfile.TemporaryFile(
                mode='w+b', prefix=_PARENT_PREFIX, dir=str(root)
            )
            mode = stat.S_IMODE(os.fstat(self._file.fileno()).st_mode)
            if mode != 0o600:
                raise PermissionError(f'parent store mode is {oct(mode)}, not 0o600')
            os.ftruncate(self._file.fileno(), self.storage_bytes)
        except Exception:
            if self._file is not None:
                self._file.close()
            self._file = None
            raise

    def _ensure_open(self):
        if self._closed or self._file is None:
            raise RuntimeError('parent store is closed')

    def _slot_offset(self, index):
        if type(index) is not int or not 0 <= index < self.count:
            raise IndexError('parent slot index is out of range')
        return index * (self.storage_bytes // self.count)

    def write(self, index, array):
        self._ensure_open()
        offset = self._slot_offset(index)
        if self._written[index]:
            raise ValueError('parent slot has already been written')
        _validate_array(array, self.shape)
        _write_all(self._file.fileno(), array, offset)
        os.fdatasync(self._file.fileno())
        _drop_cache(self._file.fileno(), offset, self.storage_bytes // self.count)
        self._written[index] = 1

    def read_into(self, index, destination):
        self._ensure_open()
        offset = self._slot_offset(index)
        if not self._written[index]:
            raise RuntimeError('parent slot has not been written')
        _validate_array(destination, self.shape, destination=True)
        _read_exact(self._file.fileno(), destination, offset)
        _drop_cache(self._file.fileno(), offset, self.storage_bytes // self.count)
        return destination

    def read_rows_into(self, index, start, stop, destination):
        """Read a bounded NCHW row tile without materializing its full parent."""
        self._ensure_open()
        offset = self._slot_offset(index)
        if len(self.shape) != 4 or self.shape[0] != 1:
            raise ValueError('row reads require a single NCHW parent')
        _, channels, height, width = self.shape
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= height):
            raise ValueError('row range must be nonempty and within the parent')
        _validate_array(destination, (1, channels, stop - start, width), destination=True)
        if not self._written[index]:
            raise RuntimeError('parent slot has not been written')
        row_bytes = width * _UINT16.itemsize
        fd = self._file.fileno()
        # NCHW rows are contiguous within each channel, not across channels.
        for channel in range(channels):
            stripe = offset + (channel * height + start) * row_bytes
            _read_exact(fd, destination[0, channel], stripe)
            _drop_cache(fd, stripe, (stop - start) * row_bytes)
        return destination

    @property
    def complete(self):
        return all(self._written)

    def require_complete(self):
        self._ensure_open()
        if not self.complete:
            raise RuntimeError('parent store has unwritten slots')

    def fileno(self):
        self._ensure_open()
        return self._file.fileno()

    def close(self):
        if self._closed:
            return
        self._closed = True
        file_obj, self._file = self._file, None
        if file_obj is not None:
            file_obj.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        file_obj = getattr(self, '_file', None)
        if file_obj is not None:
            try:
                file_obj.close()
            except Exception:
                pass
