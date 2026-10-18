"""Private original BF16 contiguous T1 reference; no execution during preparation."""
import argparse,gc,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
E=HERE.parent
PARENT=E/'source-whole-rgb-minimal-20261012'
PARENT_SHA='88a44fc742937d8aac5eac0715d880ee7e823c7dde013ecff099443b62068fd9'
ROOT_CHECK=E/'source-whole-rgb-minimal-run-20261012/results-review/ROOT-CHECK.json'
ROOT_CHECK_SHA='745c425ed75304ec38a25243a8c39d4203c7937aa24e03055e6ded67a4bf2767'
PHASES=('encode','dit','decode');CALLS=('warmup02','01','02','03')
LATENT_HEIGHT,LATENT_WIDTH=270,480
def require(value,message):
    if not value:raise ValueError(message)
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def parent():
    table=PARENT/'seal.json';require(digest(table)==PARENT_SHA,'Original source authority changed')
    pins=json.loads(table.read_text())['files']
    def load(path,name):
        raw=path.read_bytes();require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==pins[str(path)],'Parent helper changed')
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module);return module
    previous={n:sys.modules.get(n) for n in ('artifacts','child_impl','reference_bridge')}
    sys.path.insert(0,str(PARENT))
    try:
        for n in previous:sys.modules[n]=load(PARENT/(n+'.py'),n)
        module=load(PARENT/'benchmark.py','frozen_minimal_source')
        return module.source(),sys.modules['artifacts']
    finally:
        sys.path.pop(0)
        for n,m in previous.items():
            if m is None:sys.modules.pop(n,None)
            else:sys.modules[n]=m
def schedule(calls,place,forward,release):
    # Exactly three phase waves of original B1 calls; no larger Torch forward batch.
    require(calls in (('warmup02',),('01','02','03')),'Fixed schedule differs')
    for phase in PHASES:
        model=place(phase)
        for call in calls:forward(phase,call,model)
        release(phase,model)
def forward(phase,call,model,values,loaded,gpu,torch,device,clear_rope_lru_caches,np,mean_shape,prediction_shape):
    if phase == "encode":
        rgb = values["frames"]["02" if call == "warmup02" else call]
        normalized = np.ascontiguousarray((rgb.astype(np.float32) / np.float32(127.5) - np.float32(1)).transpose(2, 0, 1)[None, :, None])
        value = gpu(normalized)
        torch.xpu.synchronize(device)
        forward_started=time.monotonic()
        output = model.tiled_encode(value, tile_size=(512, 512), tile_overlap=(64, 64))
        output = output[:, :16]  # Same posterior mean; do not transfer unused variance.
    elif phase == "dit":
        moments = loaded
        condition = moments[0, :, 0].transpose(1, 2, 0)[None] * np.float32(0.9152)
        video = np.concatenate((values["fixed_noise"], condition, np.ones((1, LATENT_HEIGHT, LATENT_WIDTH, 1), np.float32)), axis=-1).reshape(-1, 33)
        value, text, timestep = gpu(video), gpu(values["text_positive"]), gpu(values["timestep"])
        vs = torch.tensor([[1, LATENT_HEIGHT, LATENT_WIDTH]], device=device, dtype=torch.long)
        ts = torch.tensor([[58]], device=device, dtype=torch.long)
        torch.xpu.synchronize(device)
        forward_started=time.monotonic()
        output = model(value, text, vs, ts, timestep, disable_cache=False).vid_sample
    else:
        prediction = np.ascontiguousarray(loaded[0].transpose(1, 2, 3, 0))
        restored = ((values["fixed_noise"] - prediction) / np.float32(0.9152)).transpose(3, 0, 1, 2)[None]
        value = gpu(restored)
        torch.xpu.synchronize(device)
        forward_started=time.monotonic()
        output = model.tiled_decode(value, tile_size=(512, 512), tile_overlap=(64, 64))
    torch.xpu.synchronize(device)
    forward_seconds=time.monotonic()-forward_started
    raw = output.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
    torch.xpu.synchronize(device)
    finite=bool(np.isfinite(raw).all())
    if phase == "encode":
        raw = mean_shape(raw)
    elif phase == "dit":
        raw = prediction_shape(raw)
    del output, value
    if phase == "dit":
        del text, timestep, vs, ts
        clear_rope_lru_caches(model)
    if phase == "decode":
        raw = np.floor((np.clip(raw[0].transpose(1, 2, 0), -1, 1) + np.float32(1)) * np.float32(127.5) + np.float32(0.5)).astype(np.uint8)
        raw = np.ascontiguousarray(raw)
    return raw,finite,forward_seconds
def live_snapshot(observer,health,directory):
    def save(name,raw):
        with (directory/name).open('xb') as stream:stream.write(raw)
    # Qualified live-owner sampler: kernel first, then physical fields; NO DRM-idle gate.
    row=observer.physical_sample(health,save)
    with (directory/'physical.json').open('x') as stream:json.dump(row,stream,indent=2)
    return row
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output
    require(root.is_absolute() and root.parent==HERE and not root.exists() and not root.is_symlink(),'Exclusive direct output required')
    require(os.geteuid()==0,'Root-controlled GPU scope only')
    old,artifacts=parent()
    old.seal.verify() # Before receipt/warm/sequence clocks; a future outer process clock must charge full preverify.
    require(digest(ROOT_CHECK)==ROOT_CHECK_SHA,'Completed original source authority changed')
    reference_pins=json.loads(ROOT_CHECK.read_text())
    require(reference_pins['status']=='PASS_EXACT_SOURCE_BRIDGES_AND_CLOSED_LIFECYCLE','Original source result failed')
    root.mkdir(mode=0o700)
    for phase in PHASES:(root/phase).mkdir()
    wall=time.monotonic();receipt={'status':'running','controlled_path':'original BF16 B1 phase waves','warmup_calls':1,'measured_calls':3,'CPU_model_owners':'separate encode/decode VAE + original DiT','calls':{},'placements':[],'GPU_calls_in_handler':0,'quality_award':False}
    completed={};owners={};active=None
    def record():
        receipt['wall_seconds']=time.monotonic()-wall
        (root/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    def archive_completed():
        # CPU-only even on failed admission/runtime/finite guard; all completed raws survive.
        for (phase,call),raw in completed.items():
            path=root/phase/(call+('.rgb24' if phase=='decode' else '.f32'))
            if not path.exists():artifacts.save(path,raw)
    try:
        import numpy as np
        ref,uhd,values=old.inputs()
        import torch
        require(torch.__version__=='2.9.1+xpu' and torch.version.cuda is None and torch.version.hip is None and not hasattr(torch._C,'_cuda_getDeviceCount'),'Pinned XPU-only Torch')
        torch.set_num_threads(4);torch.set_num_interop_threads(1);sys.path.insert(0,str(ref.SOURCE))
        resolver=ref.load_helper('evidence/vae-backend-20260918/torch_vae_backend_probe.py','original_resolver')
        device,identity=resolver._resolve_xpu_identity(torch,'0000:03:00.0');torch.xpu.set_device(device)
        from src.optimization.memory_manager import clear_memory,clear_rope_lru_caches,_standard_model_movement
        # Existing recovered root epoch and unmodified CPU-only B580 health.
        authority_path=E/'uhd-sdpa-unprofiled-run-20261012/authority.py'
        require(digest(authority_path)=='254d8870cc1a11e9761fde9d50f57cb09d32c1b3f47cc9c9b70a74134766b252','Recovery authority changed')
        spec=importlib.util.spec_from_file_location('fixed_authority',authority_path);authority=importlib.util.module_from_spec(spec);spec.loader.exec_module(authority)
        certificate=authority.approved()
        health_path=E/'uhd-phase-major-boundary-review-20261011/health.py'
        require(digest(health_path)=='b2591479539b5b5f32892e431ef86bfa4c7ae932b6a00a1329ade549fd7e54eb','Health authority changed')
        spec=importlib.util.spec_from_file_location('fixed_health',health_path);health=importlib.util.module_from_spec(spec);spec.loader.exec_module(health)
        health.BOOT=certificate['boot'];health.CURSOR=certificate['cursor']
        observer_path=E/'uhd-phase-major-frame-20261011/benchmark.py'
        require(digest(observer_path)=='e1d71eb635f52ad6017f6a1777e66c9b856d37e9f4c0250ba09fea6f8af3a35d','Observer authority changed')
        spec=importlib.util.spec_from_file_location('fixed_observer',observer_path);observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
        receipt['identity']=identity
        def allocator():
            return {key:int(getattr(torch.xpu,key)(device)) for key in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')}
        def place(phase):
            nonlocal active
            require(active is None,'GPU phases may not overlap')
            started=time.monotonic();directory=root/('placement'+str(len(receipt['placements'])));directory.mkdir()
            live_snapshot(observer,health,directory)
            if phase=='dit':
                current=allocator();require(current['memory_allocated']==current['memory_reserved']==0,'Before DiT Torch current must be zero')
                def preserve(row):
                    (directory/'admission.json').write_text(json.dumps(row,indent=2))
                    if 'raw' in row:(directory/'vram_mm.txt').write_text(row['raw'])
                old.physical_admission(Path('/sys/kernel/debug/dri/0000:03:00.0/tile0/vram_mm'),current['memory_allocated'],preserve)
            first=phase not in owners
            if first:
                if phase=='dit':model=uhd.streamed_dit(torch,device)
                else:
                    loader=ref.load_helper('openvino-vae-probe.py','original_vae')
                    model,_=loader.load_reduced_vae(ref.SOURCE,ref.TRIAL/'models'/ref.WEIGHTS['vae'][0])
                    model=model.to(device=device,dtype=torch.bfloat16).eval();uhd.vae_runtime(model)
                owners[phase]=model
            else:model=owners[phase];model.to(device)
            torch.xpu.synchronize(device);active=phase
            receipt['placements'].append(dict(phase=phase,first_load=first,seconds=time.monotonic()-started,allocator=allocator()))
            record();return model
        def release(phase,model):
            nonlocal active
            started=time.monotonic();directory=root/('release'+str(len(receipt['placements'])-1));directory.mkdir()
            live_snapshot(observer,health,directory)
            _standard_model_movement(model,device,torch.device('cpu'),'cpu','DiT' if phase=='dit' else 'VAE')
            require(all(v.device.type=='cpu' and not v.is_meta for v in (*model.parameters(),*model.buffers())),'CPU owner placement incomplete')
            torch.xpu.synchronize(device);clear_memory(deep=True,force=True);torch.xpu.synchronize(device)
            after_directory=directory/'after';after_directory.mkdir();live_snapshot(observer,health,after_directory)
            stats=allocator();require(stats['memory_allocated']==stats['memory_reserved']==0,'GPU allocator remains live after phase release')
            receipt['placements'][-1].update(release_seconds=time.monotonic()-started,after_release=stats)
            active=None;record()
        gpu=lambda value:torch.from_numpy(np.ascontiguousarray(value)).to(device=device,dtype=torch.bfloat16)
        def call(phase,label,model):
            loaded=completed.get(('encode' if phase=='dit' else 'dit',label))
            started=time.monotonic()
            with torch.inference_mode():raw,finite,forward_seconds=forward(phase,label,model,values,loaded,gpu,torch,device,clear_rope_lru_caches,np,artifacts.mean_shape,artifacts.prediction_shape)
            completed[phase,label]=raw
            require(finite and raw.shape==artifacts.SHAPES[phase],'Completed output finite/shape guard failed')
            receipt['calls'][phase+'/'+label]=dict(owned_input_to_owned_output_seconds=time.monotonic()-started,original_forward_synchronized_seconds=forward_seconds)
            record()
        started=time.monotonic();schedule(('warmup02',),place,call,release);receipt['warmup_seconds']=time.monotonic()-started
        archive_completed() # Warm archival outside both action clocks.
        started=time.monotonic();schedule(('01','02','03'),place,call,release);receipt['sequence_seconds']=time.monotonic()-started
        archive_started=time.monotonic();archive_completed();receipt['archive_seconds']=time.monotonic()-archive_started
        for phase in PHASES:
            artifacts.validate_raws(root,phase)
            reference=PARENT/'gpu-attempt1'/phase
            for label in CALLS:
                suffix='.rgb24' if phase=='decode' else '.f32';a=root/phase/(label+suffix);b=reference/a.name
                pin=reference_pins['files'][str(b)]
                info=b.lstat()
                require(b.is_file() and not b.is_symlink() and info.st_uid==1000 and info.st_nlink==1 and info.st_size==pin['bytes'] and digest(b)==pin['sha256'],'Original raw identity changed')
                require(a.read_bytes()==b.read_bytes(),'Strict original source bridge failed: '+phase+'/'+label)
            require(digest(root/phase/('warmup02'+suffix))==digest(root/phase/('02'+suffix)),'Replay differs')
            require(len({digest(root/phase/(label+suffix)) for label in CALLS[1:]})==3,'Frames not distinct')
        teardown_started=time.monotonic();owners.clear();gc.collect()
        receipt['CPU_owner_teardown_seconds']=time.monotonic()-teardown_started
        final_directory=root/'final-live-sample';final_directory.mkdir();live_snapshot(observer,health,final_directory)
        receipt['final_allocator']=allocator();receipt['status']='completed';record()
    except BaseException as error:
        archive_completed()
        receipt.update(status='failed',error=str(error),GPU_calls_in_handler=0);record();raise
if __name__=='__main__':main()
