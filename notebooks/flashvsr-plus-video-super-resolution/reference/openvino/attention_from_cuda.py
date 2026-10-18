#!/usr/bin/env python3
"""Finite BDF-specific OV operator gate consuming authenticated CUDA BSA inputs.

Launch only under the main agent's fresh GPU ownership/health preflight. This
script acquires no locks and makes no full-model or performance qualification.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parents[1]
PROBE = BASE.parent / 'flashvsr-audit/probe'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROBE))

BDF = '0000:03:00.0'
CASES = ('all', 'per_head', 'single', 'empty_row')
LIMITS = {'f32': {'max_abs': 5e-5, 'rms_relative': 5e-5},
          'f16': {'max_abs': 1e-3, 'rms_relative': 5e-3}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metrics(expected, actual):
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError('Invalid actual attention shape/finiteness')
    error = actual.astype(np.float64) - expected.astype(np.float64)
    rms = float(np.sqrt(np.mean(error * error)))
    denominator = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    return {'max_abs': float(np.abs(error).max()), 'rms_abs': rms,
            'rms_relative': rms / max(denominator, 1e-30), 'all_finite': True}


def load_case(path, cuda_record):
    from probe import Case, pack_selection, validate_selection_pack
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'q', 'k', 'v', 'mask', 'expected', 'actual'}:
            raise ValueError(f'Unexpected NPZ fields: {path}')
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    shapes = {'q': (256, 3, 128), 'k': (384, 3, 128), 'v': (384, 3, 128),
              'mask': (1, 3, 2, 3), 'expected': (256, 3, 128), 'actual': (256, 3, 128)}
    for key, value in arrays.items():
        dtype = np.dtype('bool') if key == 'mask' else np.dtype('float32')
        if value.shape != shapes[key] or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid {key} dtype/shape/finiteness in {path}')
    if hashlib.sha256(arrays['actual'].tobytes()).hexdigest() != cuda_record['raw_sha256']:
        raise ValueError(f'CUDA actual raw hash mismatch: {path}')
    # Input NPZ stores token-major CUDA tensors [tokens,heads,depth]. Windows
    # remain chronological inside each head; do not reshape without transposing.
    q = np.ascontiguousarray(arrays['q'].reshape(2, 128, 3, 128).transpose(2, 0, 1, 3))
    k = np.ascontiguousarray(arrays['k'].reshape(3, 128, 3, 128).transpose(2, 0, 1, 3))
    v = np.ascontiguousarray(arrays['v'].reshape(3, 128, 3, 128).transpose(2, 0, 1, 3))
    mask = arrays['mask'][0]
    selected, valid = pack_selection(mask)
    validate_selection_pack(mask, selected, valid)
    case = Case(path.stem, mask, selected, valid)
    return case, (q, k, v, selected, valid), arrays


def resolve(core, *, expected_name='B580'):
    if expected_name not in ('B580', 'A770'):
        raise ValueError('An explicit supported Intel GPU model is required')
    candidates, matches = [], []
    for device in core.available_devices:
        if device == 'GPU' or device.startswith('GPU.'):
            pci = core.get_property(device, 'DEVICE_PCI_INFO')
            fields = {key: int(getattr(pci, key)) for key in ('domain', 'bus', 'device', 'function')}
            if not (0 <= fields['domain'] <= 65535 and 0 <= fields['bus'] <= 255 and
                    0 <= fields['device'] <= 31 and 0 <= fields['function'] <= 7):
                raise ValueError('Malformed GPU PCIInfo fields')
            bdf = '{domain:04x}:{bus:02x}:{device:02x}.{function:x}'.format(**fields)
            name = str(core.get_property(device, 'FULL_DEVICE_NAME'))
            candidates.append({'device': device, 'bdf': bdf, 'full_name': name})
            if bdf == BDF:
                matches.append(device)
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one GPU at {BDF}: {candidates}')
    chosen = next(c for c in candidates if c['device'] == matches[0])
    if expected_name not in chosen['full_name']:
        raise ValueError(f'Unexpected exact-BDF device: {chosen}')
    return matches[0], candidates


def main():
    from cpu_layout_diagnostic import build_layout_c_graph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Completed sparse_oracle output directory')
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    parser.add_argument('--precision', choices=('f32', 'f16'), default='f32')
    parser.add_argument('--run', action='store_true', help='Explicitly enable GPU execution after external ownership preflight')
    args = parser.parse_args()
    if not args.run:
        parser.error('GPU execution requires explicit --run; main agent owns launch/preflight')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'initializing', 'precision_hint': args.precision,
              'limits_against_stored_fp32': LIMITS[args.precision], 'target_bdf': BDF,
              'scope': 'small supplied-mask attention operator; not FlashVSR model/fit/throughput qualification',
              'source_sha256': {str(p): sha(p) for p in (Path(__file__), BASE / 'sparse_oracle.py',
                                                       PROBE / 'probe.py', PROBE / 'cpu_layout_diagnostic.py')},
              'cases': []}

    def persist():
        (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    persist()
    core = compiled = request = None
    try:
        cuda = json.loads((args.input / 'results.json').read_text())
        if cuda.get('passed') is not True or {c['case'] for c in cuda['cases']} != set(CASES):
            raise ValueError('CUDA sparse oracle must pass all four prescribed cases first')
        if len(cuda['cases']) != len(CASES) or any(c['passed'] is not True for c in cuda['cases']):
            raise ValueError('Incomplete CUDA case records')
        report['cuda_report_sha256'] = sha(args.input / 'results.json')
        report['cuda_environment'] = {k: cuda[k] for k in ('torch', 'cuda', 'device', 'capability')}
        inputs = [(name, *load_case(args.input / f'{name}.npz',
                                  next(c for c in cuda['cases'] if c['case'] == name))) for name in CASES]
        import openvino as ov
        version = ov.get_version()
        if not version.startswith('2026.3.'):
            raise ValueError(f'Requires qualified OpenVINO2026.3, got {version}')
        report['runtime'] = {'openvino': version, 'numpy': np.__version__, 'python': sys.version}
        core = ov.Core()
        core.set_property({'CACHE_DIR': ''})
        device, candidates = resolve(core)
        report['devices'] = candidates
        report['selected_device'] = device
        config = {'INFERENCE_PRECISION_HINT': getattr(ov.Type, args.precision),
                  'PERFORMANCE_HINT': ov.properties.hint.PerformanceMode.LATENCY,
                  'NUM_STREAMS': ov.properties.streams.Num(1), 'CACHE_DIR': ''}
        report['compile_config'] = {'INFERENCE_PRECISION_HINT': args.precision,
                                    'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': 1,
                                    'CACHE_DIR': ''}
        persist()
        for name, case, values, arrays in inputs:
            graph, counts = build_layout_c_graph(case)
            start = time.perf_counter()
            compiled = core.compile_model(graph, device, config)
            compile_seconds = time.perf_counter() - start
            if list(compiled.get_property('EXECUTION_DEVICES')) != [device]:
                raise ValueError('Execution device differs from selected exact BDF')
            resolved_hint = str(compiled.get_property('INFERENCE_PRECISION_HINT'))
            request = compiled.create_infer_request()
            runs, first = [], None
            for repeat in range(2):
                start = time.perf_counter()
                request.infer(dict(zip(('Q', 'K', 'V', 'selectedKVblock', 'validSlot'), values)),
                              share_inputs=False, share_outputs=False)
                result = np.array(request.get_output_tensor(0).data, dtype=np.float32, copy=True)
                seconds = time.perf_counter() - start
                if result.shape != (3, 2, 128, 128):
                    raise ValueError(f'Invalid OV output shape: {result.shape}')
                actual = result.transpose(1, 2, 0, 3).reshape(256, 3, 128).copy()
                oracle_metrics = metrics(arrays['expected'], actual)
                cuda_metrics = metrics(arrays['actual'], actual)
                equal = first is None or np.array_equal(first, actual)
                if first is None:
                    first = actual
                empty_zero = bool((actual.reshape(2, 128, 3, 128).transpose(2, 0, 1, 3)[
                    ~case.block_mask.any(axis=-1)] == 0).all())
                limits = LIMITS[args.precision]
                passed = bool(oracle_metrics['max_abs'] <= limits['max_abs'] and
                              oracle_metrics['rms_relative'] <= limits['rms_relative'] and equal and empty_zero)
                np.save(args.output / f'{name}-repeat{repeat}.npy', actual, allow_pickle=False)
                runs.append({'repeat': repeat, 'passed': passed, 'seconds_infer_and_full_readback': seconds,
                             'against_stored_fp32_oracle': oracle_metrics,
                             'against_cuda_bf16_output_diagnostic_only': cuda_metrics,
                             'repeat_exact': bool(equal), 'empty_rows_zero': empty_zero,
                             'raw_sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
            report['cases'].append({'case': name, 'input_npz_sha256': sha(args.input / f'{name}.npz'),
                                    'compile_seconds': compile_seconds, 'resolved_precision_hint': resolved_hint,
                                    'ops': counts, 'runs': runs,
                                    'passed': all(r['passed'] for r in runs)})
            persist()
            print(json.dumps({'case': name, 'passed': report['cases'][-1]['passed']}), flush=True)
            del graph, first, result, actual
            request = compiled = None
            gc.collect()
        report['status'] = 'pass' if all(c['passed'] for c in report['cases']) else 'fail'
    except Exception as exc:
        report['status'] = 'fail'
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        request = compiled = core = None
        gc.collect()
        report['cleanup'] = 'references released; finite process exits naturally'
        persist()
    if report['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
