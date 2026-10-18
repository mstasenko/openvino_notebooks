"""ROOT ONLY: unchanged temporal BF16 source calls, three finite cold phases."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import stat
import time
import spec

PARENT=spec.E/'temporal-native-prototype'
PARENT_SHA='df7bc49d5f30c99ae3ccbae7f0f7057f68e471c544dae4b27f3861da5fbbbcad'
ADAPTER=spec.E/'temporal-blockswap-n20-20261012/adapter.py'

def load(path,name):
    module_spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(module_spec);sys.modules[name]=module;module_spec.loader.exec_module(module)
    return module

def authority(metadata_only=False):
    pins=json.loads((spec.HERE/'SOURCE-PINS-v2.json').read_text())
    spec.require(spec.digest(PARENT/'xpu-pins.json')==PARENT_SHA,'Parent source/runtime table changed')
    for path,row in pins['selected_files'].items():
        p=Path(path);spec.require(p.stat().st_size==row['bytes'],'Source input size changed: '+path)
        # Metadata check authenticates the existing closed fixture receipt, not its large payloads.
        if metadata_only and p in {spec.FIXTURE/name for name in spec.FILES.values()}:continue
        spec.require(spec.digest(p)==row['sha256'],'Source input changed: '+path)
    ref=load(PARENT/'xpu_t5_reference.py','original_temporal_reference')
    ref.source_check()
    return ref,load(ADAPTER,'qualified_move_only_adapter')

def prior(root,phase,call,role):
    import numpy as np
    receipt=json.loads((root/phase/'receipt.json').read_text());path=root/phase/(call+'.f32')
    info=path.lstat()
    spec.require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_uid==os.getuid(),'Handoff must be owned regular file')
    spec.require(receipt['status']=='completed' and receipt['fixture_sha256']==spec.RECEIPT_SHA,'Prior phase incomplete/input changed')
    row=receipt['outputs'][call+'.f32'];shape=spec.SHAPES[role]
    spec.require(row['shape']==list(shape) and row['dtype']=='float32' and
                 path.stat().st_size==row['bytes'] and spec.digest(path)==row['sha256'],'Handoff pin/schema changed')
    array=np.memmap(path,dtype='<f4',mode='r',shape=shape)
    spec.require(np.isfinite(array).all(),'Handoff nonfinite')
    return np.array(array,copy=True,order='C') # Owned CPU handoff; file I/O stays outside call clock.

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('encode','dit','decode'))
    parser.add_argument('--root',type=Path);parser.add_argument('--check-metadata',action='store_true')
    args=parser.parse_args();ref,adapter=authority(args.check_metadata);spec.metadata()
    if args.check_metadata:
        print(json.dumps({'status':'PASS_METADATA_ONLY','Torch_imports':0,'large_fixture_reads':0}));return
    spec.require(args.phase is not None and args.root is not None,'Exact phase/root required')
    root=args.root.resolve();spec.require(root.parent==spec.HERE and root.is_dir() and not args.root.is_symlink(),'Reviewed direct task-owned root required')
    output=root/args.phase;output.mkdir(mode=0o700)
    receipt={'status':'running','phase':args.phase,'fixture_sha256':spec.RECEIPT_SHA,'source_head':ref.HEAD,
             'blocks_to_swap':20 if args.phase=='dit' else None,'outputs':{},'calls':[],
             'no_retry':True,'input_frames':5,'latent_frames':2,'no_wavelet':True,'quality_award':False}
    started=time.monotonic()
    def record():
        receipt['wall_seconds']=time.monotonic()-started
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    def save(call,array):
        path=output/(call+'.f32')
        with path.open('xb') as file:array.tofile(file)
        receipt['outputs'][path.name]={'bytes':path.stat().st_size,'sha256':spec.digest(path),
                                      'shape':list(array.shape),'dtype':str(array.dtype)}
        record() # Every completed raw persists before late finite/shape/replay gates.
    record()
    try:
        import numpy as np
        values=spec.values({'encode':('rgb',),'dit':('noise','text'),'decode':('noise',)}[args.phase])
        import torch
        spec.require(torch.__version__=='2.9.1+xpu' and torch.version.cuda is None and torch.version.hip is None and
                     not hasattr(torch._C,'_cuda_getDeviceCount'),'Pinned XPU-only Torch required')
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        resolver=ref.load_helper('evidence/vae-backend-20260918/torch_vae_backend_probe.py','temporal_identity')
        device,identity=resolver._resolve_xpu_identity(torch,'0000:03:00.0')
        spec.require(identity['torch_uuid_raw'].replace('-','').lower()=='86800be2000000000300000000000000','B580 UUID differs')
        torch.xpu.set_device(device);receipt['identity']=identity
        # Same four healthy allocator fields as the qualified T1 StageDiagnostics.
        def allocator():
            return {name:getattr(torch.xpu,name)(device) for name in
                    ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')}
        receipt['allocator_before_model']=allocator();record()
        sys.path.insert(0,str(ref.SOURCE))
        from src.utils.debug import Debug
        from src.optimization.memory_manager import clear_rope_lru_caches,clear_memory
        if args.phase=='dit':
            admission=load(spec.E/'source-phase-major-uhd-guard-20261011/admission.py','original_uhd_admission')
            def preserve(observation):
                with (output/'admission.json').open('x') as file:json.dump(observation,file)
                if 'raw' in observation:
                    with (output/'vram_mm.txt').open('x') as file:file.write(observation['raw'])
            admission.physical_admission(Path('/sys/kernel/debug/dri/0000:03:00.0/tile0/vram_mm'),
                                         torch.xpu.memory_allocated(device),preserve)
        kind='dit' if args.phase=='dit' else 'vae';name,size,sha=ref.WEIGHTS[kind]
        weight=ref.TRIAL/'models'/name
        spec.require(weight.stat().st_size==size and spec.digest(weight)==sha,'Checkpoint changed')
        load_started=time.monotonic();debug=None
        if args.phase=='dit':
            model=ref.streamed_dit(torch,device);debug=Debug(enabled=False)
            receipt['placement']=adapter.configure_prefix(model,20,device,torch.device('cpu'),debug)
        else:
            loader=ref.load_helper('openvino-vae-probe.py','original_temporal_vae_loader')
            model,_=loader.load_reduced_vae(ref.SOURCE,weight)
            model=model.to(device=device,dtype=torch.bfloat16).eval();receipt['vae_runtime']=ref.vae_runtime(model)
        torch.xpu.synchronize(device);receipt['model_load_and_setup_seconds']=time.monotonic()-load_started
        receipt['allocator_after_model']=allocator();record()
        gpu=lambda value:torch.from_numpy(np.ascontiguousarray(value)).to(device=device,dtype=torch.bfloat16)
        with torch.inference_mode():
            for call in spec.CALLS:
                handoff_started=time.monotonic()
                if args.phase=='dit':mean=prior(root,'encode',call,'mean')
                if args.phase=='decode':prediction=prior(root,'dit',call,'prediction')
                handoff_seconds=time.monotonic()-handoff_started
                call_started=time.monotonic()
                if args.phase=='encode':
                    normalized=spec.normalize(values['rgb']);value=gpu(normalized)
                    encoded=model.tiled_encode(value,tile_size=(512,512),tile_overlap=(64,64))
                    output_tensor=encoded[:,:16] # Selected posterior mean; no unused variance archive/readback.
                elif args.phase=='dit':
                    noise=values['noise']
                    condition=mean[0].transpose(1,2,3,0)*np.float32(.9152)
                    video=np.concatenate((noise,condition,np.ones((2,270,480,1),np.float32)),axis=-1).reshape(-1,33)
                    value,text,timestep=gpu(video),gpu(values['text']),gpu(np.array([1000],np.float32))
                    vs=torch.tensor([[2,270,480]],device=device,dtype=torch.long);ts=torch.tensor([[58]],device=device,dtype=torch.long)
                    output_tensor=model(value,text,vs,ts,timestep,disable_cache=False).vid_sample
                else:
                    value=gpu(spec.restore(values['noise'],prediction))
                    output_tensor=model.tiled_decode(value,tile_size=(512,512),tile_overlap=(64,64))
                torch.xpu.synchronize(device)
                raw=output_tensor.detach().to(device='cpu',dtype=torch.float32).contiguous().numpy().copy()
                torch.xpu.synchronize(device);del output_tensor,value
                if args.phase=='encode':del encoded,normalized
                source_return_shape=list(raw.shape)
                if args.phase=='decode' and raw.shape==spec.SHAPES['decoded'] and np.isfinite(raw).all():
                    rgb=spec.rgb(raw)
                else:rgb=None
                call_seconds=time.monotonic()-call_started
                archive_started=time.monotonic()
                save(call,raw)
                receipt['outputs'][call+'.f32']['source_return_shape']=source_return_shape;record()
                if args.phase=='dit':
                    del text,timestep,vs,ts
                    raw=raw.reshape(spec.SHAPES['prediction'])
                    receipt['outputs'][call+'.f32']['shape']=list(raw.shape);record()
                role={'encode':'mean','dit':'prediction','decode':'decoded'}[args.phase]
                spec.require(raw.shape==spec.SHAPES[role] and raw.dtype==np.float32 and np.isfinite(raw).all(),'Output finite/shape guard failed')
                if args.phase=='decode':
                    spec.require(rgb is not None,'Completed RGB conversion required');path=output/(call+'.rgb24')
                    with path.open('xb') as file:rgb.tofile(file)
                    receipt['outputs'][path.name]={'bytes':path.stat().st_size,'sha256':spec.digest(path),'shape':list(rgb.shape),'dtype':'uint8'}
                    record();del rgb
                archive_seconds=time.monotonic()-archive_started
                # Never queried until completed raw finite/shape checks succeeded.
                call_allocator=allocator()
                boundary_started=time.monotonic()
                if args.phase=='dit':clear_rope_lru_caches(model) # Healthy only, AFTER saved raw passes finite/shape guard.
                cleanup_seconds=time.monotonic()-boundary_started
                receipt['calls'].append({'call':call,'owned_input_to_owned_output_seconds':call_seconds,
                    'handoff_read_seconds':handoff_seconds,'archive_hash_guard_seconds':archive_seconds,
                    'healthy_boundary_cleanup_seconds':cleanup_seconds,
                    'owned_call_plus_healthy_cleanup_seconds':call_seconds+cleanup_seconds,
                    'allocator_after_guard':call_allocator});record();del raw
        cleanup_started=time.monotonic()
        model.to(torch.device('cpu'));del model,debug;gc.collect();clear_memory(deep=True,force=True)
        torch.xpu.synchronize(device)
        receipt['phase_cleanup_seconds']=time.monotonic()-cleanup_started
        receipt['final_allocator']=allocator()
        receipt['status']='completed';receipt['replay_byte_exact']=receipt['outputs']['warmup.f32']['sha256']==receipt['outputs']['replay.f32']['sha256'];record()
        spec.require(receipt['replay_byte_exact'],'Warmup/replay byte mismatch; outputs retained')
    except BaseException as error:
        receipt.update(status='failed',error=str(error),GPU_calls_in_handler=0);record();raise

if __name__=='__main__':main()
