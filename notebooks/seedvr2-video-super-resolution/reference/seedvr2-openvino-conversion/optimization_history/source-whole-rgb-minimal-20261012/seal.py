"""Additive wrapper source pins; frozen original source/runtime remains authority."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TABLE = HERE / 'seal.json'
REVIEW = HERE / 'REVIEW-MANIFEST.json'
def pin(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): value.update(block)
    return dict(bytes=path.stat().st_size, sha256=value.hexdigest())

def verify_local():
    for name, expected in json.loads(TABLE.read_text())['files'].items():
        if pin(Path(name)) != expected: raise ValueError('matched source wrapper drift: ' + name)

if __name__ == '__main__':
    import benchmark
    if sys.argv[1:] == ['prepare']:
        receipt = json.loads((HERE / 'CPU-RECEIPT.json').read_text())
        assert receipt['status'] == 'PASS' and receipt['before'] == receipt['after']
        files = {str(HERE / name): pin(HERE / name) for name in
                 ('benchmark.py', 'child_impl.py', 'artifacts.py', 'reference_bridge.py', 'tests.py', 'README.md', 'DIFF.patch', 'run_cpu.py', 'seal.py', 'CPU-RECEIPT.json', 'cpu-attempt1.log')}
        for path, expected in benchmark.PINS.items():
            assert pin(path)['sha256'] == expected
            files[str(path)] = pin(path)
        matched = HERE.parent / 'source-phase-major-matched-20261011/seal.json'
        assert pin(matched)['sha256'] == '8c8cdf9d17ab15844b231e5527e6c939a3474e714a28aee99ea136183ac63315'
        files[str(matched)] = pin(matched)
        from reference_bridge import AUTHORITY, SHA
        assert pin(AUTHORITY)['sha256'] == SHA
        files[str(AUTHORITY)] = pin(AUTHORITY)
        old = json.loads((benchmark.OLD / 'seal.json').read_text())['files']
        admission = benchmark.OLD / 'admission.py'
        assert pin(admission) == old[str(admission)]
        files[str(admission)] = pin(admission)
        native = json.loads((benchmark.NATIVE / 'execution-closure-v1.json').read_text())
        for name, expected in native['files'].items():
            # Reuse only actual CPU supervisor/runtime authority, not model assets.
            if ('/runtime-python311/' in name or '/numpy/' in name or '/numpy.libs/' in name
                or '/numpy-' in name or name.endswith('/health.py')):
                assert pin(Path(name)) == expected
                files[name] = expected
        files[str(benchmark.NATIVE / 'fixtures.contract')] = pin(benchmark.NATIVE / 'fixtures.contract')
        with REVIEW.open('x') as stream:
            json.dump(dict(status='CPU_READY_UNSEALED', files=files, GPU_calls=0,
                original_source_seal_sha256=benchmark.PINS[benchmark.OLD / 'seal.json'],
                native_observer_authority_sha256=benchmark.PINS[benchmark.NATIVE / 'execution-closure-v1.json']), stream, indent=2)
        print(json.dumps(dict(status='CPU_READY_UNSEALED', manifest=pin(REVIEW))))
    elif len(sys.argv) == 3 and sys.argv[1] == 'freeze':
        assert pin(REVIEW)['sha256'] == sys.argv[2]
        files = json.loads(REVIEW.read_text())['files']
        for name, expected in files.items(): assert pin(Path(name)) == expected
        files[str(REVIEW)] = pin(REVIEW)
        with TABLE.open('x') as stream: json.dump(dict(files=files, inherited_runtime='old source full verify before sequence'), stream, indent=2)
        print(json.dumps(dict(status='PASS', seal=pin(TABLE))))
    elif sys.argv[1:] == ['verify-local']:
        verify_local(); print('PASS matched wrapper local pins; no Torch/model/device work')
    else: raise SystemExit('prepare | freeze REVIEWED_SHA | verify-local')
