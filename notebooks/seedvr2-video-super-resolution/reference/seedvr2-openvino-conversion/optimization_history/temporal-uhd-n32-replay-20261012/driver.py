"""Root-only fixed N32 warmup plus three replay diagnostic; never retry or query after error."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time

HERE=Path(__file__).resolve().parent
E=HERE.parent
TU=E/'temporal-uhd-n20-source-20261012'
MEAN=TU/'n20-attempt1/encode/warmup.f32'
MEAN_SHA='cb448dc98de0fe6e5aab8c1191987cab871c05eb0824daacd9eb54511b73b13a'
SPEC=TU/'spec.py'
REF=E/'temporal-native-prototype/xpu_t5_reference.py'
ADAPTER=HERE/'adapter.py'
F=E/'uhd-phase-major-frame-20261011/benchmark.py'
B=E/'uhd-phase-major-boundary-review-20261011/health.py'
RECOVERY_AUTH=E/'temporal-uhd-n32-replay-run-20261012/authority.py'
CALLS=('warmup','replay1','replay2','replay3')
FIRSTFIT=E/'temporal-uhd-n32-clear-20261012/clear-attempt1/prediction.f32'
VRAM=Path('/sys/kernel/debug/dri/0000:03:00.0/tile0/vram_mm')

def require(value,message):
    if not value:raise ValueError(message)
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(path,name):
    loader=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(loader)
    sys.modules[name]=module;loader.loader.exec_module(module);return module
def authority(metadata_only=False):
    recipe=json.loads((HERE/'SOURCE-PINS-v1.json').read_text())
    require(digest(E/'temporal-uhd-n20-run-20261012/execution-closure-v1.json')==
            'e2f055dc71fb20f01a267e65beed8a6230cf83180b73fc04845dc1c90b020282','Authenticated parent changed')
    fixture=E/'native-real-uhd-t5-fixture-20261011/fixture'
    payloads={MEAN,FIRSTFIT,fixture/'noise.thwc.f32',fixture/'text_positive.f32'}
    for name,row in recipe['selected_files'].items():
        path=Path(name);info=path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_size==row['bytes'],'Selected regular size changed')
        if metadata_only and path in payloads:continue
        require(digest(path)==row['sha256'],'Selected input/source changed: '+name)
    return recipe

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--check-metadata',action='store_true')
    args=parser.parse_args();authority(args.check_metadata)
    if args.check_metadata:
        print('PASS_METADATA_ONLY no_Torch_models_GPU_or_payload_reads');return
    require(os.geteuid()==0,'Root-controlled diagnostic only')
    require(args.output is not None and args.output.is_absolute() and args.output.parent==HERE and
            not args.output.exists() and not args.output.is_symlink(),'Exact absent direct output required')
    args.output.mkdir(mode=0o700);output=args.output
    receipt={'status':'running','blocks_to_swap':32,'fixed_warmup_plus_three_replays':True,'no_retry':True,'outputs':{},'calls':{},
             'mean_sha256':MEAN_SHA,'diagnostic_not_fastest_source_baseline':True}
    started=time.monotonic()
    def record():
        receipt['wall_seconds']=time.monotonic()-started
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    record()
    try:
        recovery=load(RECOVERY_AUTH,'explicit_root_recovery').approved()
        receipt['root_recovery_authority']=recovery
        import numpy as np
        spec=load(SPEC,'qualified_temporal_input_spec');spec.metadata()
        receipt['fixture_sha256']=spec.RECEIPT_SHA
        ref=load(REF,'original_temporal_reference');ref.source_check()
        adapter=load(ADAPTER,'qualified_N32_move_only')
        handoff_started=time.monotonic()
        require(digest(MEAN)==MEAN_SHA and MEAN.stat().st_size==16588800,'Completed encoder mean changed')
        mean=np.fromfile(MEAN,dtype='<f4').reshape(1,16,2,270,480)
        require(np.isfinite(mean).all(),'Completed mean nonfinite')
        values=spec.values(('noise','text'))
        receipt['handoff_read_hash_seconds']=time.monotonic()-handoff_started
        import torch
        require(torch.__version__=='2.9.1+xpu' and torch.version.cuda is None and torch.version.hip is None and
                not hasattr(torch._C,'_cuda_getDeviceCount'),'Pinned XPU-only Torch required')
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        resolver=ref.load_helper('evidence/vae-backend-20260918/torch_vae_backend_probe.py','cache_clear_identity')
        device,identity=resolver._resolve_xpu_identity(torch,'0000:03:00.0')
        require(identity['torch_uuid_raw'].replace('-','').lower()=='86800be2000000000300000000000000','B580 UUID differs')
        torch.xpu.set_device(device);receipt['identity']=identity
        def allocator():
            return {name:int(getattr(torch.xpu,name)(device)) for name in
                    ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')}
        sys.path.insert(0,str(ref.SOURCE))
        from src.utils.debug import Debug
        from src.optimization.memory_manager import clear_rope_lru_caches,clear_memory
        admission=load(E/'source-phase-major-uhd-guard-20261011/admission.py','original_uhd_admission')
        def preserve(row):
            with (output/'admission.json').open('x') as stream:json.dump(row,stream)
        admission.physical_admission(VRAM,torch.xpu.memory_allocated(device),preserve)
        weight=ref.TRIAL/'models'/ref.WEIGHTS['dit'][0]
        require(weight.stat().st_size==ref.WEIGHTS['dit'][1],'Checkpoint size changed; root full closure verification required')
        receipt['allocator_before_model']=allocator()
        setup_started=time.monotonic()
        model=ref.streamed_dit(torch,device);debug=Debug(enabled=False)
        receipt['placement']=adapter.configure_prefix(model,32,device,torch.device('cpu'),debug)
        torch.xpu.synchronize(device)
        observer=load(F,'qualified_physical_observer');guard=load(B,'qualified_health_constants')
        guard.BOOT=recovery['boot'];guard.CURSOR=recovery['cursor']
        def snapshot(label):
            def save(suffix,value):
                with (output/(label+'.'+suffix)).open('xb') as stream:stream.write(value)
            physical=observer.physical_sample(guard,save) # Kernel-first before allocator queries.
            row={'physical':physical,'Torch_allocator':allocator()}
            receipt[label]=row;record()
        snapshot('before_clear')
        clear_started=time.monotonic()
        clear_memory(force=True,deep=False) # ONLY controlled change, once after healthy placement.
        torch.xpu.synchronize(device)
        receipt['forced_minimal_clear_seconds']=time.monotonic()-clear_started
        snapshot('after_clear')
        receipt['model_setup_clear_observation_seconds']=time.monotonic()-setup_started;record()
        gpu=lambda value:torch.from_numpy(np.ascontiguousarray(value)).to(device=device,dtype=torch.bfloat16)
        for label in CALLS:
            snapshot('before_'+label) # Fresh kernel gate BEFORE any next forward.
            call_started=time.monotonic()
            with torch.inference_mode():
                noise=values['noise'];condition=mean[0].transpose(1,2,3,0)*np.float32(.9152)
                video=np.concatenate((noise,condition,np.ones((2,270,480,1),np.float32)),axis=-1).reshape(-1,33)
                value,text,timestep=gpu(video),gpu(values['text']),gpu(np.array([1000],np.float32))
                vs=torch.tensor([[2,270,480]],device=device,dtype=torch.long);ts=torch.tensor([[58]],device=device,dtype=torch.long)
                result=model(value,text,vs,ts,timestep,disable_cache=False).vid_sample
                torch.xpu.synchronize(device)
                raw=result.detach().to(device='cpu',dtype=torch.float32).contiguous().numpy().copy()
                torch.xpu.synchronize(device);del result,value,text,timestep,vs,ts
            call={'owned_input_to_owned_output_seconds':time.monotonic()-call_started}
            receipt['calls'][label]=call
            archive_started=time.monotonic();path=output/(label+'.f32')
            with path.open('xb') as stream:raw.tofile(stream)
            receipt['outputs'][path.name]={'bytes':path.stat().st_size,'sha256':digest(path),'source_return_shape':list(raw.shape),'dtype':str(raw.dtype)};record()
            raw=raw.reshape(2,270,480,16)
            require(raw.dtype==np.float32 and np.isfinite(raw).all(),'Prediction finite/shape guard failed; raw retained')
            receipt['outputs'][path.name]['shape']=list(raw.shape)
            call['archive_hash_guard_seconds']=time.monotonic()-archive_started
            cleanup_started=time.monotonic();clear_rope_lru_caches(model)
            call['healthy_boundary_cleanup_seconds']=time.monotonic()-cleanup_started
            call['owned_call_plus_healthy_cleanup_seconds']=call['owned_input_to_owned_output_seconds']+call['healthy_boundary_cleanup_seconds']
            record()
        for key in ('owned_input_to_owned_output_seconds','archive_hash_guard_seconds','healthy_boundary_cleanup_seconds','owned_call_plus_healthy_cleanup_seconds'):
            receipt[key]=sum(receipt['calls'][label][key] for label in CALLS)
        snapshot('after_replays')
        receipt['allocator_after_guard']=receipt['after_replays']['Torch_allocator']
        teardown_started=time.monotonic()
        model.to(torch.device('cpu'));del model,debug;gc.collect();clear_memory(deep=True,force=True)
        torch.xpu.synchronize(device);receipt['final_allocator']=allocator()
        receipt['phase_teardown_seconds']=time.monotonic()-teardown_started
        receipt['status']='completed';record()
    except BaseException as error:
        receipt.update(status='failed',error=str(error),GPU_calls_in_handler=0,no_retry=True);record();raise
if __name__=='__main__':main()
