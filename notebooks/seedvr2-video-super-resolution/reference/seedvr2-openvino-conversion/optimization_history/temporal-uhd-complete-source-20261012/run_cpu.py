"""Exclusive pure AST/tiny-array and metadata checks, never large execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
P=Path(__file__).resolve().parent
FILES=('child.py','spec.py','benchmark.py','tests.py','prepare.py','run_cpu.py','README.md')
def pins():return {name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in FILES}
before=pins();started=time.monotonic()
commands=[[sys.executable,'-B',str(P/'prepare.py')],[sys.executable,'-B',str(P/'tests.py')],
          [sys.executable,'-B',str(P/'child.py'),'--check-metadata']]
statuses=[]
with (P/'cpu.stdout').open('xb') as out,(P/'cpu.stderr').open('xb') as err:
    for command in commands:
        status=subprocess.run(command,cwd=P,stdout=out,stderr=err).returncode
        statuses.append(status)
        if status:break
after=pins();passed=statuses==[0]*3 and before==after
with (P/'CPU-RECEIPT.json').open('x') as file:
    json.dump({'status':'PASS' if passed else 'FAIL','commands':commands,'natural_statuses':statuses,
               'before':before,'after':after,'seconds':time.monotonic()-started,
               'Torch_imports':0,'models_materialized':0,'large_fixture_materialized':0,'GPU_calls':0},file,indent=2)
print(json.dumps({'status':'PASS' if passed else 'FAIL','natural_statuses':statuses}));raise SystemExit(0 if passed else 1)
