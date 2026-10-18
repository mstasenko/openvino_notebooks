"""Metadata-only preparation; no learned/large fixture payload reads or models."""
import json
from pathlib import Path
import spec

def pin(path):return {'bytes':path.stat().st_size,'sha256':spec.digest(path)}
def main():
    parent=spec.E/'temporal-native-prototype/xpu-pins.json'
    spec.require(spec.digest(parent)=='df7bc49d5f30c99ae3ccbae7f0f7057f68e471c544dae4b27f3861da5fbbbcad','Parent runtime authority changed')
    small=spec.E/'temporal-blockswap-n20-20261012'
    original_small=spec.E/'temporal-blockswap-small-20261012'
    closure=spec.E/'temporal-blockswap-small-run-20261012/execution-closure-v1.json'
    spec.require(spec.digest(closure)=='d23f2005c39e6de00d5e9fba973989d50d4479c6af5c2c461655892bacf7dff5','Qualified N16 adapter/control authority changed')
    trusted=json.loads(closure.read_text())['files']
    spec.require(pin(small/'SOURCE-PINS.json')==trusted[str(original_small/'SOURCE-PINS.json')],'Qualified selected source table changed')
    recipe=json.loads((small/'SOURCE-PINS.json').read_text())
    selected={path:row for path,row in recipe['files'].items()
              if Path(path).parent not in (spec.E/'temporal-native-prototype/fixture',
                                          spec.E/'temporal-native-prototype/xpu-t5-results')}
    for path,row in selected.items():spec.require(pin(Path(path))==row,'Selected source changed')
    adapter=small/'adapter.py'
    spec.require(spec.digest(adapter)=='b5533055be456c591e8da24dabf76100d62ce4f3543059604068a580ebc4c59c','Reviewed N20 adapter changed')
    selected[str(adapter)]=pin(adapter)
    for path in [spec.E/'temporal-native-prototype/xpu_t5_reference.py']:
        row=pin(path);spec.require(row==trusted[str(path)],'Qualified loader/adapter changed');selected[str(path)]=row
    fixture,checked=spec.metadata()
    for name,row in checked['files'].items():
        path=spec.FIXTURE/name
        spec.require(path.stat().st_size==row['bytes'],'Existing fixture size differs')
        selected[str(path)]=row # Inherit exact closed receipt, NOT a newly computed payload SHA.
    for path in [parent,closure,small/'REVIEW-MANIFEST.json',spec.FIXTURE/'receipt.json',spec.FIXTURE.parent/'CHECK-RECEIPT.json',
                 spec.FIXTURE.parent/'REVIEW.md',spec.FIXTURE.parent/'prepare.py',spec.FIXTURE.parent/'check.py',
                 spec.FIXTURE.parent/'native_fixture.cpp',spec.T/'media-evidence.json']:
        selected[str(path)]=pin(path)
    for name in ('child.py','spec.py','benchmark.py','tests.py','prepare.py','README.md','run_cpu.py'):
        selected[str(spec.HERE/name)]=pin(spec.HERE/name)
    old=spec.E/'source-phase-major-uhd-guard-20261011'
    spec.require(spec.digest(old/'seal.json')=='2b7c50a0c1a66c29673e4570c752df2f6a05b54f8d485fefc969bf99d39414b5','UHD admission authority changed')
    oldpins=json.loads((old/'seal.json').read_text())['files']
    spec.require(pin(old/'admission.py')==oldpins[str(old/'admission.py')],'UHD admission parser changed')
    selected[str(old/'admission.py')]=pin(old/'admission.py');selected[str(old/'seal.json')]=pin(old/'seal.json')
    value={'parent_runtime_authority_sha256':spec.digest(parent),'qualified_small_execution_sha256':spec.digest(closure),
           'selected_files':selected,'large_fixture_hashes_inherited_from':str(spec.FIXTURE.parent/'CHECK-RECEIPT.json'),
           'original_mp4_hash_and_extraction_command_available':False,'Torch_imports':0,'GPU_calls':0,
           'large_fixture_materialized':False,'checkpoint_payload_read':False,
           'offload_blocks':20,'small_N20_GPU_parity_root_admission_pending':True}
    with (spec.HERE/'SOURCE-PINS-v2.json').open('x') as file:json.dump(value,file,indent=2)
    print(json.dumps({'status':'SOURCE_METADATA_PREPARED','selected_files':len(selected),'large_payload_reads':0}))

if __name__=='__main__':main()
