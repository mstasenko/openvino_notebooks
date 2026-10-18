#!/usr/bin/env python3
"""Minimal BF16 source archive/timer adapter; frozen model math and phase isolation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
E = HERE.parent
OLD = E / 'source-phase-major-uhd-guard-20261011'
NATIVE = E / 'uhd-phase-major-frame-20261011'
HEALTH = E / 'uhd-phase-major-boundary-review-20261011/health.py'
PINS = {
    OLD / 'benchmark.py': '13480afda494b7ffb439a6a4409233a5b4dea1d367b32ead7d1496e04d45de7b',
    OLD / 'seal.py': 'bdd04b7ecfbc581deabc38b6e81ebc6b8c1a17463151adfdc96c521795a9890b',
    OLD / 'seal.json': '2b7c50a0c1a66c29673e4570c752df2f6a05b54f8d485fefc969bf99d39414b5',
    NATIVE / 'benchmark.py': 'e1d71eb635f52ad6017f6a1777e66c9b856d37e9f4c0250ba09fea6f8af3a35d',
    NATIVE / 'execution-closure-v1.json': '353926ccfe52e0fc31e170b9aa50dad45235d159300a5f54e84d8f434babdf82',
    HEALTH: 'b2591479539b5b5f32892e431ef86bfa4c7ae932b6a00a1329ade549fd7e54eb',
}

def load(path, name):
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[path]:
        raise ValueError('frozen dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def source():
    # The old child captures its original seal and admission globals. Avoid
    # resolving those generic imports to this new wrapper's local seal helper.
    if digest(OLD / 'seal.json') != PINS[OLD / 'seal.json']: raise ValueError('old source table changed')
    authority = json.loads((OLD / 'seal.json').read_text())['files']
    admission_path = OLD / 'admission.py'
    PINS[admission_path] = authority[str(admission_path)]['sha256']
    admission = load(admission_path, 'frozen_source_admission')
    seal = load(OLD / 'seal.py', 'frozen_source_seal')
    previous = {name: sys.modules.get(name) for name in ('seal', 'admission')}
    sys.modules['seal'] = seal
    sys.modules['admission'] = admission
    sys.path.insert(0, str(OLD))
    try:
        return load(OLD / 'benchmark.py', 'frozen_source_benchmark')
    finally:
        sys.path.pop(0)
        for name, module in previous.items():
            if module is None: sys.modules.pop(name, None)
            else: sys.modules[name] = module

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): value.update(block)
    return value.hexdigest()

from artifacts import files, audit, validate as validate_artifacts
from child_impl import child
from reference_bridge import bridge

def validate(root,phase,old):
    validate_artifacts(root,phase,old)
    bridge(root,phase)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--child', choices=('encode', 'dit', 'decode'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    old = source()
    if args.check:
        _, _, values = old.inputs()  # Exact CPU fixture producer, no Torch/model load.
        import shlex
        pins = {}
        for line in (NATIVE / 'fixtures.contract').read_text().splitlines():
            name, path, size, value = shlex.split(line)
            pins[name] = (int(size), value)
        arrays = {**{'frame' + name: value for name, value in values['frames'].items()},
                  'text': values['text_positive'], 'noise': values['fixed_noise']}
        for name, value in arrays.items():
            data = value.tobytes(order='C')
            if (len(data), hashlib.sha256(data).hexdigest()) != pins[name]:
                raise ValueError('source/native fixed fixture differs: ' + name)
        if 'torch' in sys.modules: raise ValueError('unexpected Torch import during fixture check')
        print('PASS frozen BF16 source fixtures; no Torch/model/device calls')
        return
    root = args.output
    if root is None or not root.is_absolute() or root.parent != HERE or root.is_symlink():
        parser.error('output must be a direct absolute owned child')
    if args.child:
        if not root.is_dir(): parser.error('supervisor root missing')
        old.seal.verify_local()
        child(root, args.child, old)  # Same BF16 forwards/load/cleanup; compact raw/timer adapter.
        audit(root, args.child, old)  # CPU-only after healthy completion, never after exception.
        return
    import seal
    seal.verify_local()
    old.seal.verify()  # Unchanged full runtime authority, outside sequence timer.
    native = load(NATIVE / 'benchmark.py', 'frozen_native_observer')
    health = load(HEALTH, 'frozen_phase_health')
    root.mkdir()
    def launch(phase):
        with (root / (phase + '.log')).open('xb') as log:
            def action():
                return subprocess.run([sys.executable, '-u', '-B', str(HERE / 'benchmark.py'),
                    '--child', phase, '--output', str(root)], stdout=log, stderr=subprocess.STDOUT).returncode
            return native.observed_launch(root, phase, action, lambda save: native.physical_sample(health, save))
    try:
        receipt = native.sequence(root, 'source_bf16', launch, health.health,
                                  lambda root, phase, variant: validate(root, phase, old))
        receipt.update(label='process-isolated UHD source BF16; all child load/handoff/teardown charged',
            mean_contract='source first16 mean BCTHW16', prediction_contract='source BCTHW16',
            artifact_payload_bytes=165888000, npy_header_bytes=0,
            native_artifact_payload_bytes=165888000, extra_archival_bytes=0,
            source_math_unchanged=True, archival_payload_identical_to_native_phase_protocol=True,
            not_equal_to_monolithic_native_warm_sequence=True)
        with (root / 'receipt.json').open('x') as stream: json.dump(receipt, stream, indent=2, allow_nan=False)
    except BaseException as error:
        with (root / 'error.json').open('x') as stream:
            json.dump(dict(error=str(error), no_next_child=True, quality_pass=False), stream)
        raise

if __name__ == '__main__': main()
