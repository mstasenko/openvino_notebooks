#!/usr/bin/env python3
"""Root-launched T1 setup diagnostic; metadata mode is source-only.

The runtime reuses the frozen T1 parent/ref/loader and the public memory
manager.  It encodes frame 02 once, retains the VAE on CPU, activates the
stock streamed DiT, records the public shallow-clear boundary, and performs
zero DiT forwards.  Root owns any runtime launch and GPU lifecycle.
"""
from __future__ import annotations

import argparse,gc,hashlib,importlib.util,json,os,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
E=HERE.parent
T1=E/'source-contiguous-t1-20261012'
DRIVER=T1/'driver_v2.py'
ROOT_CHECK=E/'source-whole-rgb-minimal-run-20261012/results-review/ROOT-CHECK.json'
RECOVERY=E/'uhd-sdpa-unprofiled-run-20261012/authority.py'
HEALTH=E/'uhd-phase-major-boundary-review-20261011/health.py'
OBSERVER=E/'uhd-phase-major-frame-20261011/benchmark.py'
DRIVER_SHA='f36433c131ccf21c4a512fab0de079b961364ce6f6d97c4f4b1302bd4cda86c3'
ROOT_CHECK_SHA='745c425ed75304ec38a25243a8c39d4203c7937aa24e03055e6ded67a4bf2767'
RECOVERY_SHA='254d8870cc1a11e9761fde9d50f57cb09d32c1b3f47cc9c9b70a74134766b252'
HEALTH_SHA='b2591479539b5b5f32892e431ef86bfa4c7ae932b6a00a1329ade549fd7e54eb'
OBSERVER_SHA='e1d71eb635f52ad6017f6a1777e66c9b856d37e9f4c0250ba09fea6f8af3a35d'
MEAN_SHA='39527231a9535eec056842116640906ac063ffd8860b866d679ddf9a70abbfd5'

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def source_metadata():
    rows={}
    for path,expected in ((DRIVER,DRIVER_SHA),(ROOT_CHECK,ROOT_CHECK_SHA),
                          (RECOVERY,RECOVERY_SHA),(HEALTH,HEALTH_SHA),(OBSERVER,OBSERVER_SHA)):
        info=path.lstat();require(path.is_file() and not path.is_symlink() and info.st_nlink==1,'source is not regular: '+str(path))
        actual=digest(path);require(actual==expected,'source pin changed: '+str(path))
        rows[str(path)]={'bytes':info.st_size,'sha256':actual}
    return {'status':'PASS_SOURCE_ONLY','source_files':rows,'torch_imports':0,
            'core_calls':0,'device_queries':0,'model_payload_reads':0,'dit_forward_calls':0}
def load(path,name,expected):
    require(digest(path)==expected,'runtime authority changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'cannot load '+str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def snapshot(observer,health,torch,device,output,label):
    directory=output/label;directory.mkdir()
    def save(name,raw):
        with (directory/name).open('xb') as stream:stream.write(raw)
    physical=observer.physical_sample(health,save)
    allocator={key:int(getattr(torch.xpu,key)(device)) for key in
               ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')}
    require(physical['free']==physical['size']-physical['usage'],'physical snapshot arithmetic differs')
    row={'label':label,'physical':physical,'allocator':allocator}
    (directory/'snapshot.json').write_text(json.dumps(row,indent=2)+'\n');return row
def run(output):
    require(os.geteuid()==0,'setup diagnostic is root-controlled')
    require(output.is_absolute() and output.parent==HERE and not output.exists() and not output.is_symlink(),
            'output must be an absent direct child')
    output.mkdir(mode=0o700)
    receipt={'status':'running','vae_encode_calls':0,'dit_setup_calls':0,'dit_forward_calls':0,
             'old_body_gate_invoked':False,'no_retry':True}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        driver=load(DRIVER,'frozen_t1_driver_setup_only',DRIVER_SHA)
        old,artifacts=driver.parent();old.seal.verify()
        require(digest(ROOT_CHECK)==ROOT_CHECK_SHA,'closed T1 source authority changed')
        require(json.loads(ROOT_CHECK.read_text())['status']=='PASS_EXACT_SOURCE_BRIDGES_AND_CLOSED_LIFECYCLE',
                'closed T1 source result is not PASS')
        import numpy as np
        import torch
        require(torch.__version__=='2.9.1+xpu' and torch.version.cuda is None and torch.version.hip is None,
                'pinned XPU Torch required')
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        ref,uhd,values=old.inputs()
        resolver=ref.load_helper('evidence/vae-backend-20260918/torch_vae_backend_probe.py','setup_only_resolver')
        device,identity=resolver._resolve_xpu_identity(torch,'0000:03:00.0');torch.xpu.set_device(device)
        sys.path.insert(0,str(ref.SOURCE))
        from src.optimization.memory_manager import clear_memory,_standard_model_movement
        authority=load(RECOVERY,'setup_only_recovery',RECOVERY_SHA).approved()
        health=load(HEALTH,'setup_only_health',HEALTH_SHA);health.BOOT,health.CURSOR=authority['boot'],authority['cursor']
        observer=load(OBSERVER,'setup_only_observer',OBSERVER_SHA)
        snapshot(observer,health,torch,device,output,'idle')
        cpu=torch.device('cpu');loader=ref.load_helper('openvino-vae-probe.py','setup_only_vae')
        vae,_=loader.load_reduced_vae(ref.SOURCE,ref.TRIAL/'models'/ref.WEIGHTS['vae'][0])
        vae=vae.to(device=cpu,dtype=torch.bfloat16).eval();uhd.vae_runtime(vae);vae=vae.to(device)
        torch.xpu.synchronize(device)
        rgb=values['frames']['02'];normalized=np.ascontiguousarray((rgb.astype(np.float32)/np.float32(127.5)-np.float32(1)).transpose(2,0,1)[None,:,None])
        with torch.inference_mode():encoded=vae.tiled_encode(torch.from_numpy(normalized).to(device=device,dtype=torch.bfloat16),tile_size=(512,512),tile_overlap=(64,64))[:,:16]
        torch.xpu.synchronize(device);mean=encoded.detach().to(device=cpu,dtype=torch.float32).contiguous().numpy().copy()
        mean=artifacts.mean_shape(mean);require(bool(np.isfinite(mean).all()),'T1 setup mean is nonfinite');artifacts.save(output/'mean.f32',mean)
        receipt.update(vae_encode_calls=1,mean_sha256=digest(output/'mean.f32'));require(receipt['mean_sha256']==MEAN_SHA,'T1 setup mean differs from closed source')
        del encoded,normalized
        _standard_model_movement(vae,device,cpu,'cpu','VAE');torch.xpu.synchronize(device);clear_memory(deep=True,force=True);torch.xpu.synchronize(device)
        require(int(torch.xpu.memory_allocated(device))==int(torch.xpu.memory_reserved(device))==0,'VAE release allocator is not zero')
        snapshot(observer,health,torch,device,output,'after-vae-release')
        dit=uhd.streamed_dit(torch,device);torch.xpu.synchronize(device);receipt['dit_setup_calls']=1
        snapshot(observer,health,torch,device,output,'dit-setup-before-clear')
        clear_memory(deep=False,force=True);torch.xpu.synchronize(device)
        snapshot(observer,health,torch,device,output,'dit-setup-after-shallow-clear');require(receipt['dit_forward_calls']==0,'setup path executed a DiT forward')
        _standard_model_movement(dit,device,cpu,'cpu','DiT');torch.xpu.synchronize(device);clear_memory(deep=True,force=True);torch.xpu.synchronize(device)
        require(int(torch.xpu.memory_allocated(device))==int(torch.xpu.memory_reserved(device))==0,'DiT release allocator is not zero')
        snapshot(observer,health,torch,device,output,'after-dit-release');del dit,vae,old,ref,uhd,values;gc.collect()
        snapshot(observer,health,torch,device,output,'final');receipt.update(status='completed',identity=identity)
    except BaseException as error:
        receipt.update(status='failed',error=str(error),dit_forward_calls=0)
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');raise
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    print(json.dumps(source_metadata(),indent=2,sort_keys=True)) if args.output is None else run(args.output)
