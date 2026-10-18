"""Fixed compact raw handoff for the existing four-call source protocol."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

CALLS=('warmup02','01','02','03')
PHASES=('encode','dit','decode')
HEIGHT,WIDTH=2160,3840
LATENT_HEIGHT,LATENT_WIDTH=270,480
SHAPES={'encode':(1,16,1,LATENT_HEIGHT,LATENT_WIDTH),
        'dit':(1,16,1,LATENT_HEIGHT,LATENT_WIDTH),'decode':(HEIGHT,WIDTH,3)}
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def json_object(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    return json.loads(path.read_text(),object_pairs_hook=unique)
def array(path,shape,dtype=None):
    import numpy as np
    dtype=np.dtype(np.float32 if dtype is None else dtype)
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1 or info.st_size!=math.prod(shape)*dtype.itemsize:
        raise ValueError('handoff ownership/type/size differs')
    raw=path.read_bytes()
    if len(raw)!=info.st_size:raise ValueError('handoff read truncated')
    value=np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
    if not np.isfinite(value).all():raise ValueError('nonfinite handoff')
    return value
def save(path,value):
    if path.exists() or path.is_symlink():raise ValueError('existing raw refused')
    with path.open('xb') as stream:stream.write(value.tobytes(order='C'))
def mean_shape(raw):
    import numpy as np
    if raw.ndim!=4 or raw.shape[:2]!=(1,16):raise ValueError('source mean shape differs')
    return np.ascontiguousarray(raw[:,:,None])
def prediction_shape(raw):
    import numpy as np
    return np.ascontiguousarray(raw.reshape((1,LATENT_HEIGHT,LATENT_WIDTH,16)).transpose(3,0,1,2)[None])
def files(root,phase,old=None):
    extension='.rgb24' if phase=='decode' else '.f32'
    return [root/phase/(call+extension) for call in CALLS]
def validate_raws(root,phase):
    import numpy as np
    for path in files(root,phase):array(path,SHAPES[phase],np.uint8 if phase=='decode' else np.float32)
def audit(root,phase,old=None):
    started=time.monotonic()
    rows={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in files(root,phase)}
    with (root/phase/'producer-hashes.json').open('x') as stream:
        json.dump(dict(files=rows,cpu_archive_audit_seconds=time.monotonic()-started,
                       charged_child_wall=True,no_GPU_calls=True),stream,indent=2)
def validate(root,phase,old=None):
    validate_raws(root,phase)
    receipt=json_object(root/phase/'receipt.json')
    if receipt['phase']!=phase or receipt['calls']!=list(CALLS) or receipt['status']!='PASS':
        raise ValueError('child receipt differs')
    clocks=[receipt['seconds'],receipt['model_load_seconds'],receipt['normal_cleanup_seconds'],
            *receipt['call_seconds'],*receipt['raw_write_seconds'],*receipt['handoff_read_seconds']]
    if any(len(receipt[name])!=4 for name in ('call_seconds','raw_write_seconds','handoff_read_seconds')) or any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in clocks):
        raise ValueError('child clock schema differs')
    rows=json_object(root/phase/'producer-hashes.json')['files']
    expected={path.name:path for path in files(root,phase)}
    if set(rows)!=set(expected):raise ValueError('exact four producer rows required')
    for name,path in expected.items():
        if rows[name]!=dict(bytes=path.stat().st_size,sha256=digest(path)):
            raise ValueError('producer hash differs')
