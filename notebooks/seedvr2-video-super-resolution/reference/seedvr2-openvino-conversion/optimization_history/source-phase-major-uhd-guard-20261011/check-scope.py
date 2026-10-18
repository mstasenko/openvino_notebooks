"""Evidence-only device-cgroup admission; no inference, resets or permission changes."""
import errno
import json
import os
from pathlib import Path
import stat
import sys

HERE = Path(__file__).resolve().parent
if os.getuid() != 0 or len(sys.argv) != 2:
    raise RuntimeError('Expected root and an absent receipt basename')
receipt = HERE / sys.argv[1]
if receipt.parent != HERE or receipt.exists():
    raise RuntimeError('Exact absent receipt required')

def denied(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as error:
        if error.errno != errno.EPERM:
            raise RuntimeError('Expected device-cgroup EPERM: ' + str(path)) from error
        return
    os.close(descriptor)
    raise RuntimeError('Device confinement not enforced: ' + str(path))

# A harmless omitted pseudo-device proves enforcement before any GPU open.
denied('/dev/full')
allowed = {'card2', 'renderD130'}
nodes = list(Path('/dev/dri').iterdir())
identities = {}
gpu_rdevs = set()
for node in nodes:
    info = node.lstat()
    if not stat.S_ISCHR(info.st_mode):
        continue
    device = (Path('/sys/class/drm') / node.name / 'device').resolve(strict=True)
    is_target = device.name == '0000:03:00.0'
    if is_target != (node.name in allowed):
        raise RuntimeError('B580 node identity changed: ' + str(node))
    identities[node.name] = {'rdev': [os.major(info.st_rdev), os.minor(info.st_rdev)],
                             'device': str(device), 'allowed': is_target}
    gpu_rdevs.add(info.st_rdev)
    if not is_target:
        denied(node)
if {name for name, value in identities.items() if value['allowed']} != allowed:
    raise RuntimeError('Both B580 nodes required')
for node in map(Path, ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools']):
    if node.exists():
        gpu_rdevs.add(node.stat().st_rdev)
        denied(node)
for name in sorted(allowed):
    descriptor = os.open('/dev/dri/' + name, os.O_RDWR | os.O_CLOEXEC)
    os.close(descriptor)
for fd in Path('/proc/self/fd').iterdir():
    try:
        target = os.readlink(fd)
    except FileNotFoundError:
        continue
    if os.stat(fd).st_rdev in gpu_rdevs or target.startswith('/dev/dri/') or target.startswith('/dev/nvidia'):
        raise RuntimeError('Unexpected inherited GPU descriptor')
with receipt.open('x') as output:
    json.dump({'PASS': True, 'GPU_inferences': 0, 'denied_pseudo_device': '/dev/full',
               'nodes': identities, 'cgroup': Path('/proc/self/cgroup').read_text()}, output, indent=2)
    output.write('\n')
print('PASS enforced B580-only device opens; other DRM nodes denied by cgroup')
