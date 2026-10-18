"""Fixed UHD T5 fixture metadata; importing this module materializes no tensors."""
import hashlib
import json
from pathlib import Path
import stat

HERE=Path(__file__).resolve().parent
E=HERE.parent
T=E.parents[1]
FIXTURE=E/'native-real-uhd-t5-fixture-20261011/fixture'
RECEIPT_SHA='9a57579d46fd1e1aaf186c52e1c41fbdd5cb44e61832cccf038e3cdc856e92c2'
CHECK_SHA='8ebba1dcbeca4f8c0083a646f3551a108ddc8f10e5154c70773344b4daffb7bb'
CALLS=('warmup','replay')
SHAPES={'normalized':(1,3,5,2160,3840),'noise':(2,270,480,16),'text':(58,5120),
        'mean':(1,16,2,270,480),'prediction':(2,270,480,16),'decoded':(1,3,5,2160,3840),
        'rgb':(5,2160,3840,3)}
FILES={'normalized':'normalized.bcthw.f32','noise':'noise.thwc.f32','text':'text_positive.f32','rgb':'frames.rgb24'}

def digest(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()

def require(okay,message):
    if not okay:raise ValueError(message)

def metadata():
    require(digest(FIXTURE/'receipt.json')==RECEIPT_SHA,'Fixture producer receipt changed')
    check=FIXTURE.parent/'CHECK-RECEIPT.json'
    require(digest(check)==CHECK_SHA,'Full fixture check receipt changed')
    receipt=json.loads((FIXTURE/'receipt.json').read_text());checked=json.loads(check.read_text())
    require(receipt['geometry']=={'rgb_thwc':[5,2160,3840,3],'noise_thwc':[2,270,480,16],
                                'patch_hidden':[1,2,135,240,2560],'total_tokens':64800},'Temporal geometry differs')
    require([f['timestamp'] for f in receipt['frames']]==[10.0,10.033,10.067,10.1,10.133] and
            all(f['sample_rgb_exact'] for f in receipt['frames']),'Chronology/sample identity differs')
    require(checked['noise_slices_distinct'] and checked['first_last_rgb_changed_channels']>0,'Repeated input/noise')
    require(receipt['no_wavelet'] and receipt['resampler']=={'library':'Pillow','version':'12.3.0',
        'filter':'BICUBIC','reducing_gap':None,'scale':3},'Fixture transforms differ')
    return receipt,checked

def values(roles,verify_payload=True):
    import numpy as np
    _,checked=metadata();result={}
    require(len(set(roles))==len(roles) and set(roles)<=set(FILES),'Fixture role selection differs')
    for role in roles:
        path=FIXTURE/FILES[role];row=checked['files'][path.name];info=path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_size==row['bytes'],'Fixture regular size differs')
        if verify_payload:require(digest(path)==row['sha256'],'Fixture payload hash differs')
        dtype='u1' if role=='rgb' else '<f4'
        result[role]=np.fromfile(path,dtype=dtype).reshape(SHAPES[role])
    return result

def normalize(rgb):
    import numpy as np
    require(rgb.shape==SHAPES['rgb'] and rgb.dtype==np.uint8,'RGB input schema differs')
    return (rgb.astype(np.float32)/np.float32(127.5)-np.float32(1)).transpose(3,0,1,2)[None]

def restore(noise,prediction):
    import numpy as np
    require(noise.shape==prediction.shape==SHAPES['prediction'],'Temporal prediction shape differs')
    return ((noise-prediction)/np.float32(0.9152)).transpose(3,0,1,2)[None]

def rgb(decoded):
    import numpy as np
    require(decoded.shape==SHAPES['decoded'],'Decoded temporal shape differs')
    return np.floor((np.clip(decoded[0].transpose(1,2,3,0),-1,1)+np.float32(1))*np.float32(127.5)+np.float32(.5)).astype(np.uint8)
