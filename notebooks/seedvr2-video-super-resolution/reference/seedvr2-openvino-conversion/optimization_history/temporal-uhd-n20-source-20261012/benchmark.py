"""Future root-only three cold children; source preparation has no execution."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import spec

PINS={spec.E/'uhd-phase-major-frame-20261011/benchmark.py':'e1d71eb635f52ad6017f6a1777e66c9b856d37e9f4c0250ba09fea6f8af3a35d',
      spec.E/'uhd-phase-major-boundary-review-20261011/health.py':'b2591479539b5b5f32892e431ef86bfa4c7ae932b6a00a1329ade549fd7e54eb'}

def load(path,name):
    spec.require(spec.digest(path)==PINS[path],'Fixed operational helper changed')
    loader=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module);return module

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve()
    spec.require(os.geteuid()==0 and root.parent==spec.HERE and not args.output.exists() and not args.output.is_symlink(),'Root-only exact absent direct output required')
    helper=load(spec.E/'uhd-phase-major-frame-20261011/benchmark.py','fixed_observer')
    guard=load(spec.E/'uhd-phase-major-boundary-review-20261011/health.py','fixed_health')
    root.mkdir(mode=0o700);started=time.monotonic();phases=[]
    try:
        for phase in ('encode','dit','decode'):
            with (root/(phase+'-child.log')).open('xb') as file:
                def action():
                    status=subprocess.run([sys.executable,'-u','-B',str(spec.HERE/'child.py'),'--phase',phase,'--root',str(root)],stdout=file,stderr=subprocess.STDOUT).returncode
                    with (root/(phase+'-status.json')).open('x') as out:json.dump({'natural_status':status},out)
                    return status
                status=helper.observed_launch(root,phase,action,lambda save:helper.physical_sample(guard,save))
            spec.require(status==0,'Child failed naturally; no retry/next phase')
            # Same reviewed kernel-first boundary prevents next launch on health failure.
            health=guard.health(root,phase)
            receipt=json.loads((root/phase/'receipt.json').read_text())
            spec.require(receipt['status']=='completed' and receipt['replay_byte_exact'],'Incomplete or replay-failed phase')
            phases.append({'phase':phase,'receipt':receipt,'boundary_health':health})
        with (root/'supervisor-receipt.json').open('x') as file:
            json.dump({'status':'completed','whole_three_process_seconds':time.monotonic()-started,'phases':phases,
                       'no_quality_or_speed_award':True,'initial_offload_blocks':20},file,indent=2)
    except BaseException as error:
        with (root/'supervisor-error.json').open('x') as file:json.dump({'error':str(error),'GPU_calls_in_handler':0,'no_retry':True},file)
        raise

if __name__=='__main__':main()
