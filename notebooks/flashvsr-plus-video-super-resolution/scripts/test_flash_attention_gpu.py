#!/usr/bin/env python3
"""Lean stdlib/ctypes OpenCL gate; default execution is CPU-only."""
import argparse
import array
import ctypes as c
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

H, QW, ROWS, D = 1, 1, 128, 128
DIRECT_KEYS, CROSS_KEYS, SPLITS = 2, 4, 4
ATOL, RTOL, PARTIAL_ATOL = 5.0e-4, 1.0e-2, 1.0e-3
OFFSETS = (32, 32, 64, 4, 32)

def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]

def bfword(x):
    bits = struct.unpack("<I", struct.pack("<f", f32(x)))[0]
    word = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    word &= 0xFFFF
    if word & 0x7F80 == 0x7F80:
        raise ValueError("BF16 reference conversion overflow")
    return word

def bffloat(word):
    return struct.unpack("<f", struct.pack("<I", (word & 0xFFFF) << 16))[0]

def finite(word): return (word & 0x7F80) != 0x7F80

def word_pattern(i, salt):
    sign = 0x8000 if (i + 3 * salt) % 7 == 0 else 0
    return sign | (0x3D00 + 0x80 * ((i + salt) % 4)) | ((17 * i + 11 * salt) & 0x7F)

def pack_u16(values):
    a = array.array("H", (int(x) & 0xFFFF for x in values))
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()

def pack_i32(values):
    a = array.array("i", values)
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()

def pack_f32(values):
    return struct.pack("<" + "f" * len(values), *(f32(x) for x in values))

def unpack_u16(data):
    return list(struct.unpack("<" + "H" * (len(data) // 2), data))

def unpack_f32(data):
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))

def fsum(values):
    total = 0.0
    for value in values:
        total = f32(total + value)
    return total

def strides(shape):
    result = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        result[i] = result[i + 1] * shape[i + 1]
    return tuple(result)

def case(key_blocks, ids, salt):
    q = [word_pattern(i, salt) for i in range(ROWS * D)]
    k = [word_pattern(i, salt + 17) for i in range(key_blocks * ROWS * D)]
    v = [word_pattern(i, salt + 31) for i in range(key_blocks * ROWS * D)]
    return q, k, v, list(ids)

def prefix_length(ids, binary=False):
    if not binary:
        return next((i for i, block in enumerate(ids) if block < 0), len(ids))
    stop = len(ids)
    valid = 0
    while valid < stop:
        middle = valid + ((stop - valid) >> 1)
        if ids[middle] >= 0:
            valid = middle + 1
        else:
            stop = middle
    return valid

def selected(ids, binary=False):
    return ids[:prefix_length(ids, binary)]

def state(qw, kw, vw, key_blocks, ids, span=None, precise=False, binary=False):
    """One FP32 online-softmax oracle used by direct and split paths."""
    q, k, v = [list(map(bffloat, x)) for x in (qw, kw, vw)]
    chosen = selected(ids, binary)
    if span is not None:
        chosen = chosen[span[0]:span[1]]
    maximum = [float("-inf")] * ROWS
    denominator = [0.0] * ROWS
    accumulator = [[0.0] * D for _ in range(ROWS)]
    scale = f32(1.0 / math.sqrt(D))
    for block in chosen:
        if block >= key_blocks:
            continue
        for start in range(0, ROWS, 32):
            for row in range(ROWS):
                qbase = row * D
                logits = []
                for keyrow in range(start, start + 32):
                    base = (block * ROWS + keyrow) * D
                    dot = 0.0
                    for depth in range(D):
                        dot = f32(dot + f32(q[qbase + depth] * k[base + depth]))
                    logits.append(f32(dot * scale))
                updated = max(maximum[row], max(logits))
                correction = (0.0 if maximum[row] == float("-inf") else
                              f32(math.exp(maximum[row] - updated)))
                probabilities = [f32(math.exp(x - updated)) for x in logits]
                denominator[row] = f32(f32(denominator[row] * correction) + fsum(probabilities))
                maximum[row] = updated
                high = [bffloat(bfword(x)) for x in probabilities]
                low = ([bffloat(bfword(f32(x - y))) for x, y in zip(probabilities, high)]
                       if precise else [0.0] * 32)
                acc = [f32(x * correction) for x in accumulator[row]]
                for key_offset in range(32):
                    base = (block * ROWS + start + key_offset) * D
                    for depth in range(D):
                        value = v[base + depth]
                        acc[depth] = f32(acc[depth] + f32(high[key_offset] * value))
                        if precise:
                            acc[depth] = f32(acc[depth] + f32(low[key_offset] * value))
                accumulator[row] = acc
    return accumulator, maximum, denominator

def final_words(the_state):
    acc, _, den = the_state
    return [bfword(acc[row][depth] / den[row]) if den[row] else 0
            for row in range(ROWS) for depth in range(D)]

def split_expected(qw, kw, vw, key_blocks, ids, binary=False):
    valid = len(selected(ids, binary))
    states = [state(qw, kw, vw, key_blocks, ids,
                    (valid * split // SPLITS, valid * (split + 1) // SPLITS),
                    binary=binary)
              for split in range(SPLITS)]
    partial = []
    for acc, maximum, denominator in states:
        for row in range(ROWS):
            partial.extend(acc[row])
            partial.extend((maximum[row], denominator[row]))
    merged = [0] * (ROWS * D)
    for row in range(ROWS):
        global_max = max(part[1][row] for part in states)
        for depth in range(D):
            den = num = 0.0
            for acc, maximum, local_den in states:
                if local_den[row] <= 0.0:
                    continue
                weight = f32(math.exp(maximum[row] - global_max))
                den = f32(den + f32(weight * local_den[row]))
                num = f32(num + f32(weight * acc[row][depth]))
            if den > 0.0:
                merged[row * D + depth] = bfword(f32(num / den))
    return partial, merged

def prefix_numerical(qw, kw, vw, key_blocks, ids, binary=False):
    """Small finite online-softmax check for every prefix-search boundary."""
    q, k, v = [list(map(bffloat, x)) for x in (qw, kw, vw)]
    maximum, denominator, accumulator = float("-inf"), 0.0, 0.0
    scale = f32(1.0 / math.sqrt(D))
    for block in selected(ids, binary):
        if block >= key_blocks:
            continue
        base = block * ROWS * D
        dot = 0.0
        for depth in range(D):
            dot = f32(dot + f32(q[depth] * k[base + depth]))
        logit = f32(dot * scale)
        updated = max(maximum, logit)
        correction = (0.0 if maximum == float("-inf") else
                      f32(math.exp(maximum - updated)))
        probability = f32(math.exp(logit - updated))
        denominator = f32(f32(denominator * correction) + probability)
        accumulator = f32(f32(accumulator * correction) +
                           f32(probability * v[base]))
        maximum = updated
    return 0 if not denominator else bfword(f32(accumulator / denominator))

def cpu_check():
    direct = case(DIRECT_KEYS, [0, 1, -1, -1], 3)
    cross = case(CROSS_KEYS, [0, 1, 2, 3, -1, -1, -1, -1], 9)
    direct_out = final_words(state(*direct[:3], DIRECT_KEYS, direct[3]))
    precise_out = final_words(state(*cross[:3], CROSS_KEYS, cross[3], precise=True))
    empty_partial, empty_out = split_expected(*cross[:3], CROSS_KEYS, [0, 1, -1, -1])
    all_partial, all_out = split_expected(*cross[:3], CROSS_KEYS,
                                           [0, 1, 2, 3, -1, -1, -1, -1])
    binary_empty_partial, binary_empty_out = split_expected(
        *cross[:3], CROSS_KEYS, [0, 1, -1, -1], binary=True)
    binary_all_partial, binary_all_out = split_expected(
        *cross[:3], CROSS_KEYS, [0, 1, 2, 3, -1, -1, -1, -1], binary=True)
    for label, values in (("direct", direct_out), ("precise", precise_out),
                          ("split-empty", empty_out), ("split-all", all_out)):
        if not any(values) or any(not finite(x) for x in values):
            raise AssertionError(f"{label} reference is zero/nonfinite")
    if direct_out == precise_out or len(empty_partial) != SPLITS * ROWS * 130:
        raise AssertionError("reference variants did not diverge or split extent is wrong")
    if (binary_empty_partial != empty_partial or binary_empty_out != empty_out or
            binary_all_partial != all_partial or binary_all_out != all_out):
        raise AssertionError("linear and binary split references differ")

    prefix_lengths = (0, 1, 2, 3, 4, 5, 7, 8)
    prefix_case = case(8, [-1] * 8, 41)
    prefix_reports = []
    for length in prefix_lengths:
        ids = list(range(length)) + [-1] * (8 - length)
        qw, kw, vw, _ = prefix_case
        linear = prefix_numerical(qw, kw, vw, 8, ids)
        binary = prefix_numerical(qw, kw, vw, 8, ids, binary=True)
        if prefix_length(ids) != length or prefix_length(ids, True) != length:
            raise AssertionError(f"prefix length mismatch at {length}")
        if linear != binary or not finite(linear) or not finite(binary):
            raise AssertionError(f"linear/binary finite mismatch at {length}")
        prefix_reports.append({"length": length, "linear": linear, "binary": binary})
    return {"status": "pass", "numpy": False, "direct_nonzero": True, "precise_nonzero": True,
            "empty_splits": [0, 2], "all_splits_nonempty": True,
            "binary_prefix": {"lengths": list(prefix_lengths), "cases": prefix_reports},
            "partial_atol": PARTIAL_ATOL}

def bind(lib, name, result, args):
    fn = getattr(lib, name)
    fn.restype, fn.argtypes = result, args
    return fn

class OpenCL:
    def __init__(self, expected):
        p, u, s, e = c.c_void_p, c.c_uint, c.c_size_t, c.c_int
        self.lib = c.CDLL("libOpenCL.so.1")
        self.platforms = bind(self.lib, "clGetPlatformIDs", e, [u, c.POINTER(p), c.POINTER(u)])
        self.devices = bind(self.lib, "clGetDeviceIDs", e, [p, c.c_ulong, u, c.POINTER(p), c.POINTER(u)])
        self.info = bind(self.lib, "clGetDeviceInfo", e, [p, u, s, p, c.POINTER(s)])
        self.context_fn = bind(self.lib, "clCreateContext", p, [p, u, c.POINTER(p), p, p, c.POINTER(e)])
        self.queue_fn = bind(self.lib, "clCreateCommandQueue", p, [p, p, c.c_ulong, c.POINTER(e)])
        self.program_fn = bind(self.lib, "clCreateProgramWithSource", p,
                               [p, u, c.POINTER(c.c_char_p), c.POINTER(s), c.POINTER(e)])
        self.build_fn = bind(self.lib, "clBuildProgram", e, [p, u, c.POINTER(p), c.c_char_p, p, p])
        self.buildinfo = bind(self.lib, "clGetProgramBuildInfo", e, [p, p, u, s, p, c.POINTER(s)])
        self.kernel_fn = bind(self.lib, "clCreateKernel", p, [p, c.c_char_p, c.POINTER(e)])
        self.buffer_fn = bind(self.lib, "clCreateBuffer", p, [p, c.c_ulong, s, p, c.POINTER(e)])
        self.setarg = bind(self.lib, "clSetKernelArg", e, [p, u, s, p])
        self.write = bind(self.lib, "clEnqueueWriteBuffer", e, [p, p, u, s, s, p, u, p, p])
        self.read = bind(self.lib, "clEnqueueReadBuffer", e, [p, p, u, s, s, p, u, p, p])
        self.enqueue = bind(self.lib, "clEnqueueNDRangeKernel", e,
                            [p, p, u, p, c.POINTER(s), c.POINTER(s), u, p, p])
        self.finish = bind(self.lib, "clFinish", e, [p])
        self.release_mem = bind(self.lib, "clReleaseMemObject", e, [p])
        self.release_kernel = bind(self.lib, "clReleaseKernel", e, [p])
        self.release_program = bind(self.lib, "clReleaseProgram", e, [p])
        self.release_queue = bind(self.lib, "clReleaseCommandQueue", e, [p])
        self.release_context = bind(self.lib, "clReleaseContext", e, [p])
        self.expected = expected.upper()
        self.platform, device, self.name = self.select()
        self.device = c.c_void_p(device)
        self.context = self.queue = None
        try:
            status = e()
            self.context = self.context_fn(None, 1, c.byref(self.device), None, None, c.byref(status))
            self.check(status.value, "clCreateContext")
            self.queue = self.queue_fn(self.context, self.device, 0, c.byref(status))
            self.check(status.value, "clCreateCommandQueue")
        except Exception:
            self.close()
            raise
    @staticmethod
    def check(code, label):
        if code: raise RuntimeError(f"{label} failed with OpenCL error {code}")
    def info_string(self, device):
        size = c.c_size_t()
        self.check(self.info(device, 0x102B, 0, None, c.byref(size)), "device name size")
        buf = (c.c_ubyte * size.value)()
        self.check(self.info(device, 0x102B, size.value, buf, None), "device name")
        return bytes(buf).split(b"\0", 1)[0].decode(errors="replace")
    def select(self):
        if self.expected not in ("A770", "B580"):
            raise ValueError("--expected-device must name A770 or B580")
        count = c.c_uint()
        self.check(self.platforms(0, None, c.byref(count)), "platform count")
        ps = (c.c_void_p * count.value)()
        self.check(self.platforms(count.value, ps, None), "platforms")
        matches = []
        for platform in ps:
            n = c.c_uint()
            if self.devices(platform, 4, 0, None, c.byref(n)) != 0 or not n.value:
                continue
            ds = (c.c_void_p * n.value)()
            self.check(self.devices(platform, 4, n.value, ds, None), "devices")
            for device in ds:
                name = self.info_string(device)
                if self.expected.casefold() in name.casefold():
                    matches.append((platform, device, name))
        if len(matches) != 1:
            raise RuntimeError("expected exactly one matching GPU: " + ", ".join(x[2] for x in matches))
        return matches[0]
    def build(self, text, options, entry):
        source = text.encode()
        pointer = c.c_char_p(source)
        length, status = c.c_size_t(len(source)), c.c_int()
        program = self.program_fn(self.context, 1, c.byref(pointer), c.byref(length), c.byref(status))
        self.check(status.value, "clCreateProgramWithSource")
        try:
            code = self.build_fn(program, 1, c.byref(self.device), options.encode(), None, None)
            size = c.c_size_t()
            self.check(self.buildinfo(program, self.device, 0x1183, 0, None, c.byref(size)), "build log size")
            buf = (c.c_ubyte * max(size.value, 1))()
            self.check(self.buildinfo(program, self.device, 0x1183, size.value, buf, None), "build log")
            log = bytes(buf).decode(errors="replace").rstrip("\0\n")
            if code:
                raise RuntimeError(f"{entry} build failed: {log}")
            kernel = self.kernel_fn(program, entry.encode(), c.byref(status))
            self.check(status.value, f"clCreateKernel({entry})")
            return program, kernel, log
        except Exception:
            self.release_program(program)
            raise
    def close(self):
        if getattr(self, "queue", None):
            self.release_queue(self.queue)
            self.queue = None
        if getattr(self, "context", None):
            self.release_context(self.context)
            self.context = None

class Buffer:
    def __init__(self, api, data, item_bytes, offset, guards, fill):
        if len(fill) != item_bytes:
            raise ValueError("bad canary width")
        self.api, self.item_bytes, self.offset, self.logical_data = api, item_bytes, offset, data
        prefix, suffix = fill * offset, fill * guards
        self.initial = bytearray(prefix + data + suffix)
        self.host, self.object = bytearray(self.initial), None
        status = c.c_int()
        self.object = api.buffer_fn(api.context, 1, len(self.host), None, c.byref(status))
        api.check(status.value, "clCreateBuffer")
        try:
            self.upload()
        except Exception:
            api.release_mem(self.object)
            self.object = None
            raise
    def pointer(self):
        view = (c.c_ubyte * len(self.host)).from_buffer(self.host)
        return view, c.cast(view, c.c_void_p)
    def upload(self):
        view, pointer = self.pointer()
        try:
            self.api.check(self.api.write(self.api.queue, self.object, 1, 0, len(self.host), pointer, 0, None, None), "upload")
            self.api.check(self.api.finish(self.api.queue), "upload finish")
        finally:
            del view
    def download(self):
        view, pointer = self.pointer()
        try:
            self.api.check(self.api.read(self.api.queue, self.object, 1, 0, len(self.host), pointer, 0, None, None), "download")
        finally:
            del view
    def logical(self):
        start = self.offset * self.item_bytes
        return bytes(self.host[start:start + len(self.logical_data)])
    def guards_ok(self):
        start = self.offset * self.item_bytes
        return (self.host[:start] == self.initial[:start] and
                self.host[start + len(self.logical_data):] == self.initial[start + len(self.logical_data):])
    def close(self):
        if self.object is not None:
            self.api.release_mem(self.object)
            self.object = None

class BufferSet:
    def __init__(self):
        self.items = []
    def add(self, item): self.items.append(item); return item
    def close(self):
        for item in reversed(self.items):
            item.close()
        self.items.clear()
    def download(self):
        for item in self.items:
            item.download()

    def snapshot(self, count):
        return [item.logical() for item in self.items[:count]]

    def require_unchanged(self, expected, label):
        if self.snapshot(len(expected)) != expected:
            raise RuntimeError(f"{label} logical input changed")

    def guards_ok(self): return all(item.guards_ok() for item in self.items)

def contracts(kind, key_blocks, capacity, offsets=True):
    q = (H, QW, ROWS, D); kv = (key_blocks, ROWS, H, D); ids = (H, QW, 1, capacity)
    out = q; partial = (SPLITS * H, QW, ROWS, 130)
    off = OFFSETS if offsets else (0, 0, 0, 0, 0)
    qp, kp, ip, op, pp = strides(q), strides(kv), strides(ids), strides(out), strides(partial)
    if kind == "split-partial":
        dims = {"INPUT0_DIMS": q, "INPUT1_DIMS": kv, "INPUT2_DIMS": kv,
                "INPUT3_DIMS": ids, "OUTPUT0_DIMS": partial,
                "INPUT0_PITCHES": qp, "INPUT1_PITCHES": kp, "INPUT2_PITCHES": kp,
                "INPUT3_PITCHES": ip, "OUTPUT0_PITCHES": pp}
    elif kind == "split-merge":
        dims = {"INPUT0_DIMS": partial, "OUTPUT0_DIMS": out,
                "INPUT0_PITCHES": pp, "OUTPUT0_PITCHES": op}
    else:
        dims = {"INPUT0_DIMS": q, "INPUT1_DIMS": kv, "INPUT2_DIMS": kv,
                "INPUT3_DIMS": ids, "OUTPUT0_DIMS": out,
                "INPUT0_PITCHES": qp, "INPUT1_PITCHES": kp, "INPUT2_PITCHES": kp,
                "INPUT3_PITCHES": ip, "OUTPUT0_PITCHES": op}
    dims.update({f"INPUT{i}_OFFSET": off[i] for i in range(4)})
    dims["OUTPUT0_OFFSET"] = off[4]
    return dims, out, partial, off

def prefix(dims):
    return "".join(f"#define {k} ((int4)({','.join(map(str, v))}))\n" if isinstance(v, tuple)
                   else f"#define {k} {v}\n" for k, v in dims.items())

def source_for(root, split=False):
    path = root / ("plus_split_key.cl" if split else "plus_sparse_xmx.cl")
    if path.is_file():
        return path
    path = root / "plus_sparse_xmx.cl"
    text = path.read_text()
    if split and not all(name in text for name in ("otc_plus_split_key_partial", "otc_plus_split_key_merge")):
        raise FileNotFoundError("shared source has no split entries")
    return path

def make_buffers(api, kind, key_blocks, ids):
    dims, out, partial, off = contracts(kind, key_blocks, len(ids))
    qw, kw, vw, _ = case(key_blocks, ids, 3 if key_blocks == 2 else 9)
    owner = BufferSet()
    try:
        owner.add(Buffer(api, pack_u16(qw), 2, off[0], 16, b"\x5a\xa5"))
        owner.add(Buffer(api, pack_u16(kw), 2, off[1], 16, b"\x5a\xa5"))
        owner.add(Buffer(api, pack_u16(vw), 2, off[2], 16, b"\x5a\xa5"))
        owner.add(Buffer(api, pack_i32(ids), 4, off[3], 4, b"\x4d\x3c\x2b\x1a"))
        raw = (pack_f32([float("nan")] * (partial[0] * partial[1] * partial[2] * partial[3]))
               if kind == "split-partial" else pack_u16([0] * (out[0] * out[1] * out[2] * out[3])))
        owner.add(Buffer(api, raw, 4 if kind == "split-partial" else 2, off[4],
                         4 if kind == "split-partial" else 16,
                         b"\x11\x22\x33\x44" if kind == "split-partial" else b"\x11\x22"))
    except Exception:
        owner.close()
        raise
    return owner, qw, kw, vw, dims, off

def execute(api, kernel, args, global_z):
    for i, item in enumerate(args):
        handle = c.c_void_p(item.object)
        api.check(api.setarg(kernel, i, c.sizeof(handle), c.byref(handle)), f"arg{i}")
    gs = (c.c_size_t * 3)(16, 16, global_z)
    ls = (c.c_size_t * 3)(16, 1, 1)
    api.check(api.enqueue(api.queue, kernel, 3, None, gs, ls, 0, None, None), "enqueue")
    api.check(api.finish(api.queue), "kernel finish")

def compare(actual, expected, label, atol, rtol, raw_actual=None, raw_expected=None):
    if len(actual) != len(expected):
        raise RuntimeError(f"{label} extent differs")
    worst = None; max_abs = max_rel = 0.0; passed = True
    for i, (a, b) in enumerate(zip(actual, expected)):
        if math.isinf(a) or math.isinf(b):
            same = math.isinf(a) and math.isinf(b) and ((a > 0) == (b > 0))
            if same: continue
            absolute = relative = float("inf")
        elif not math.isfinite(a) or not math.isfinite(b):
            absolute = relative = float("inf")
        else:
            absolute = abs(a - b); relative = absolute / max(abs(b), 1e-8)
        if absolute > max_abs or relative > max_rel:
            max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
            worst = {"index": i, "actual_f32": a, "expected_f32": b}
            if raw_actual is not None: worst["actual"] = raw_actual[i]
            if raw_expected is not None: worst["expected"] = raw_expected[i]
        if not math.isfinite(a) or not math.isfinite(b) or absolute > atol + rtol * abs(b):
            passed = False
    return {"passed": passed, "max_abs": max_abs, "max_rel": max_rel,
            "worst": worst, "atol": atol, "rtol": rtol}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def dump(directory, name, data):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists():
        raise RuntimeError(f"refusing overwrite: {path}")
    path.write_bytes(data)
    return str(path), sha(data)

def run_direct(api, root, kind, dump_dir, atol, rtol):
    keys = CROSS_KEYS if kind == "precise-cross" else DIRECT_KEYS
    ids = list(range(keys)) + [-1] * keys
    path = source_for(root)
    raw = path.read_bytes(); text = raw.decode(); dims, *_ = contracts(kind, keys, len(ids))
    flags = "-cl-std=CL2.0 -cl-intel-256-GRF-per-thread"
    entry = "otc_cross_precise_xmx" if kind == "precise-cross" else "otc_plus_sparse_xmx"
    if kind == "precise-cross": flags += " -DOTC_CROSS_PV_SPLIT=1"
    program = kernel = owner = None
    try:
        program, kernel, log = api.build(prefix(dims) + text, flags, entry)
        owner, qw, kw, vw, _, off = make_buffers(api, kind, keys, ids)
        inputs = owner.snapshot(4)
        execute(api, kernel, owner.items, H); owner.download()
        if not owner.guards_ok(): raise RuntimeError(f"{kind} canary changed")
        owner.require_unchanged(inputs, kind)
        actual_bytes = owner.items[-1].logical(); actual = unpack_u16(actual_bytes)
        expected = final_words(state(qw, kw, vw, keys, ids, precise=kind == "precise-cross"))
        report = compare([bffloat(x) for x in actual], [bffloat(x) for x in expected], kind, atol, rtol, actual, expected)
        if not report["passed"]: raise RuntimeError(f"{kind} numerical mismatch: {report}")
        item = {"kind": kind, "source_sha256": sha(raw), "comparison": report,
                "output_sha256": sha(actual_bytes), "offsets": off, "build_log": log}
        if dump_dir: item["dump_path"], item["dump_sha256"] = dump(dump_dir, f"{kind}.u16le", actual_bytes)
        return item
    finally:
        if owner: owner.close()
        if kernel: api.release_kernel(kernel)
        if program: api.release_program(program)

def run_split(api, root, dump_dir, atol, rtol):
    path = source_for(root, True); raw = path.read_bytes(); text = raw.decode(); reports = []
    cases = (
        # Existing partition-boundary fixtures: one has empty split ranges,
        # and one has every split range nonempty.
        ("empty", CROSS_KEYS, [0, 1, -1, -1]),
        ("all", CROSS_KEYS, [0, 1, 2, 3, -1, -1, -1, -1]),
        ("zero-prefix", CROSS_KEYS, [-1] * 8),
        ("odd3-prefix", CROSS_KEYS, [0, 1, 2, -1, -1, -1, -1, -1]),
        ("full-prefix", 8, list(range(8))),
    )
    for variant, binary in (("linear", False), ("binary", True)):
        for label, key_blocks, ids in cases:
            owner = program = kernel = merge_program = merge_kernel = None
            try:
                dims, *_ = contracts("split-partial", key_blocks, len(ids))
                flags = "-cl-std=CL2.0 -cl-intel-256-GRF-per-thread -DOTC_SPLIT_PARTIAL=1 -DOTC_SPLIT_KEY=4"
                if binary:
                    flags += " -DOTC_BINARY_PREFIX=1"
                program, kernel, plog = api.build(prefix(dims) + text, flags,
                    "otc_plus_split_key_partial")
                owner, qw, kw, vw, _, off = make_buffers(api, "split-partial", key_blocks, ids)
                inputs = owner.snapshot(4)
                execute(api, kernel, owner.items, SPLITS * H); owner.download()
                if not owner.guards_ok(): raise RuntimeError(f"split/{variant}/{label} partial canary changed")
                owner.require_unchanged(inputs, f"split/{variant}/{label} partial")
                partial_bytes = owner.items[-1].logical(); actual_partial = unpack_f32(partial_bytes)
                expected_partial, expected_final = split_expected(
                    qw, kw, vw, key_blocks, ids, binary=binary)
                partial_report = compare(actual_partial, expected_partial,
                                         f"partial/{variant}/{label}", PARTIAL_ATOL, 0.0)
                if not partial_report["passed"]:
                    raise RuntimeError(f"partial/{variant}/{label} mismatch: {partial_report}")
                mdims, *_ = contracts("split-merge", key_blocks, len(ids))
                merge_program, merge_kernel, mlog = api.build(prefix(mdims) + text,
                    "-cl-std=CL2.0 -cl-intel-256-GRF-per-thread -DOTC_SPLIT_MERGE=1 -DOTC_SPLIT_KEY=4",
                    "otc_plus_split_key_merge")
                final = owner.add(Buffer(api, pack_u16([0] * ROWS * D), 2, OFFSETS[4], 16, b"\x11\x22"))
                partial_before_merge = owner.items[-2].logical()
                execute(api, merge_kernel, [owner.items[-2], final], H); owner.download()
                if not owner.guards_ok(): raise RuntimeError(f"split/{variant}/{label} merge canary changed")
                owner.require_unchanged(inputs, f"split/{variant}/{label} merge")
                if owner.items[-2].logical() != partial_before_merge:
                    raise RuntimeError(f"split/{variant}/{label} partial changed during merge")
                final_bytes = final.logical(); final_actual = unpack_u16(final_bytes)
                report = compare([bffloat(x) for x in final_actual], [bffloat(x) for x in expected_final],
                                 f"merge/{variant}/{label}", atol, rtol, final_actual, expected_final)
                if not report["passed"]:
                    raise RuntimeError(f"merge/{variant}/{label} mismatch: {report}")
                item = {"variant": variant, "case": label, "source_sha256": sha(raw),
                        "partial": partial_report, "comparison": report,
                        "output_sha256": sha(final_bytes), "partial_build_log": plog,
                        "merge_build_log": mlog, "offsets": off}
                if dump_dir:
                    stem = f"split-{variant}-{label}"
                    item["partial_dump"], item["partial_dump_sha256"] = dump(
                        dump_dir, f"{stem}.f32le", partial_bytes)
                    item["dump_path"], item["dump_sha256"] = dump(
                        dump_dir, f"merge-{variant}-{label}.u16le", final_bytes)
                reports.append(item)
            finally:
                if owner: owner.close()
                if kernel: api.release_kernel(kernel)
                if program: api.release_program(program)
                if merge_kernel: api.release_kernel(merge_kernel)
                if merge_program: api.release_program(merge_program)
    return {"source_sha256": sha(raw), "cases": reports}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-root", type=Path, required=True)
    parser.add_argument("--expected-device", choices=("A770", "B580"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--rtol", type=float, default=RTOL)
    args = parser.parse_args()
    cpu = cpu_check()
    if not args.run:
        print(json.dumps({"cpu_self_check": cpu}, indent=2, sort_keys=True)); return 0
    if args.expected_device is None: parser.error("--expected-device is required with --run")
    if not (args.kernel_root / "plus_sparse_xmx.cl").is_file():
        raise SystemExit("plus_sparse_xmx.cl is required")
    source_for(args.kernel_root, True)
    if args.dump_dir and args.dump_dir.exists() and any(args.dump_dir.iterdir()):
        raise SystemExit("refusing non-empty dump directory")
    started = time.monotonic(); api = OpenCL(args.expected_device)
    try:
        variants = [run_direct(api, args.kernel_root, "direct", args.dump_dir, args.atol, args.rtol),
                    run_direct(api, args.kernel_root, "precise-cross", args.dump_dir, args.atol, args.rtol),
                    run_split(api, args.kernel_root, args.dump_dir, args.atol, args.rtol)]
        result = {"status": "pass", "device": api.name, "expected_device": args.expected_device,
                  "cpu_self_check": cpu, "variants": variants,
                  "elapsed_seconds": time.monotonic() - started,
                  "kernel_root": str(args.kernel_root)}
    finally:
        api.close()
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        path = args.dump_dir / "kernel-gate-lean.json"
        if path.exists(): raise SystemExit("refusing report overwrite")
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        result["report_path"] = str(path)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
