"""Strict same-source bridge; never read old99MB decoded-float dumps."""
import json
import stat
from pathlib import Path
from artifacts import CALLS, SHAPES, array, digest, mean_shape, prediction_shape
E=Path(__file__).resolve().parent.parent
OLD=E/'source-phase-major-matched-20261011/gpu-attempt1'
AUTHORITY=E/'source-phase-major-matched-20261011/results-review/RESULTS.json'
SHA='58555a73326a5c94c875e674162ccb98b5048e8b1694ed3cd3df080344bcf15d'
def bridge(root,phase):
    import numpy as np
    if digest(AUTHORITY)!=SHA:raise ValueError('source output authority changed')
    table=json.loads(AUTHORITY.read_text())
    if table['status']!='PASS' or any(not row['passed'] for row in table['gates']):
        raise ValueError('old source protocol gate failed')
    rows={}
    for call in CALLS:
        reference=OLD/phase/(call+('.rgb24' if phase=='decode' else '.npy'))
        info=reference.lstat();pin=table['files'][str(reference)]
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=1000 or info.st_nlink!=1 or info.st_size!=pin['bytes'] or digest(reference)!=pin['sha256']:
            raise ValueError('old source raw identity differs')
        if phase=='decode':
            expected=reference.read_bytes()
            actual=array(root/phase/(call+'.rgb24'),SHAPES[phase],np.uint8).tobytes(order='C')
        else:
            with reference.open('rb') as stream:
                version=np.lib.format.read_magic(stream)
                shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                wanted=(1,32,270,480) if phase=='encode' else (1,270,480,16)
                if version!=(1,0) or shape!=wanted or fortran or dtype!=np.float32 or info.st_size!=stream.tell()+int(np.prod(shape))*4:
                    raise ValueError('old source NPY schema differs')
            value=np.load(reference,allow_pickle=False)
            if not np.isfinite(value).all():raise ValueError('old source raw nonfinite')
            converted=mean_shape(value[:,:16]) if phase=='encode' else prediction_shape(value)
            expected=converted.tobytes(order='C')
            actual=array(root/phase/(call+'.f32'),SHAPES[phase]).tobytes(order='C')
        if actual!=expected:raise ValueError('strict source adapter bridge failed: '+phase+'/'+call)
        rows[call]=dict(status='BYTE_EXACT',reference=str(reference),reference_pin=pin)
    paths=[root/phase/(call+('.rgb24' if phase=='decode' else '.f32')) for call in CALLS]
    hashes=[digest(path) for path in paths]
    if hashes[0]!=hashes[2] or len(set(hashes[1:]))!=3:
        raise ValueError('source replay/distinction failed')
    with (root/phase/'reference-bridge.json').open('x') as stream:
        json.dump(dict(status='PASS',rows=rows,warmup_replay='PASS',three_distinct='PASS',
                       source_only_not_cross_engine_quality=True),stream,indent=2)
