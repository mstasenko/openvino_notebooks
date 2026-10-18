"""Pinned Ubuntu26 native pack recipe. No Core, inference or learned-weight conversion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/seedvr"
WHEEL = ("openvino-2026.3.0-22451-cp311-cp311-manylinux_2_28_x86_64.whl", 57472600,
         "93e111164ccfd5c801edcfc3437683210837463bf57a2bd0518b940598664c2c",
         "https://files.pythonhosted.org/packages/b9/8e/b162407883ad51958d481c2af29d59e694caf4d8216b0c409c59294fd1aa/")
DEBS = [
    ("libssl3t64_3.5.5-1ubuntu3.7_amd64.deb", 2368026,
     "3ff73e121f6cd4f19e80ab4329dfd914108b3936c9db3708200203a9b245b34d"),
    ("libssl-dev_3.5.5-1ubuntu3.7_amd64.deb", 2936724,
     "d1cf6cafb141490b7020e392964b1f901835d60cb88c3a68ef0f95ef5e930de4"),
]
ARCHIVES = {
    "structure.zip": (1580909, "d23c7af7eb47a219b24a0698eca6e4fa51c06afdd037bc9050ddc6cb39b7fb12"),
    "rife-ir.zip": (21210600, "912fea8a9a405799cbe122605da12d6e4b5330aec83a0e026910651cb34d39fa"),
    "notices.zip": (64876, "21b9b99d79af32fe66800fc68efb104367e1e93bf4092b6e6395eed4b48d5578"),
}


def check(path, size, sha):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
        raise ValueError("nonregular/wrong size: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    if digest.hexdigest() != sha:
        raise ValueError("fixed hash differs: " + str(path))


def fetch(cache, name, size, sha, url):
    path = cache / name
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix=name + ".part-", dir=cache)
        partial = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as stream, urllib.request.urlopen(url) as response:
                total = 0
                while block := response.read(1048576):
                    total += len(block)
                    if total > size:
                        raise ValueError("download exceeds fixed size")
                    stream.write(block)
            check(partial, size, sha)
            try:
                os.link(partial, path)  # Publish complete bytes without overwriting a racing download.
            except FileExistsError:
                check(path, size, sha)
        finally:
            partial.unlink()  # Only this call's exact owned temporary file.
    check(path, size, sha)
    return path


def safe(name):
    return bool(name) and not name.startswith("/") and "\\" not in name and all(x not in ("", ".", "..") for x in name.split("/"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def unpack_fixed(path, directory, rows):
    expected = {row["path"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("duplicate expected archive entry")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != len(expected) or len({x.filename for x in infos}) != len(infos):
            raise ValueError("archive inventory differs")
        for info in infos:
            if not safe(info.filename) or info.filename not in expected or info.is_dir() or not stat.S_ISREG(info.external_attr >> 16):
                raise ValueError("unsafe/unexpected archive entry")
            row = expected[info.filename]
            if info.file_size != row["bytes"] or info.file_size > 32 * 1024**2:
                raise ValueError("archive member bound differs")
            data = archive.read(info)
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("member hash differs")
            write(directory / info.filename, data)


def target(path, parent, directory):
    path = Path(os.path.abspath(path))
    if path == parent or parent not in path.parents or path.is_symlink():
        raise ValueError("output target is outside generated root or a symlink")
    # Reject symlink ancestors without resolving them into some other destination.
    for ancestor in path.parents:
        if ancestor == ROOT:
            break
        if ancestor.is_symlink():
            raise ValueError("output target has a symlink ancestor")
    if path.exists() and (not path.is_dir() if directory else not path.is_file()):
        raise ValueError("existing output target has unexpected type")
    return path


def publish(staged, destination, directory):
    destination = target(destination, ROOT / ("dist" if directory else "release"), directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        holder = Path(tempfile.mkdtemp(prefix=destination.name + ".previous-", dir=destination.parent))
        backup = holder / "previous"
        destination.rename(backup)  # Preserve old versions without inspecting/executing their worker.
    try:
        if directory:
            staged.rename(destination)
        else:
            os.link(staged, destination)  # Publish complete bytes without clobbering a racing archive.
            staged.unlink()
    except Exception:
        if backup is not None and not destination.exists() and not destination.is_symlink():
            backup.rename(destination)
        raise
    return str(backup) if backup else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="CPU artifact closure without native build")
    parser.add_argument("--native-root", type=Path, default=ROOT / "native", help="Evidence validation only; default is checked-out source")
    parser.add_argument("--out", type=Path, default=ROOT / "dist/seedvr2-openvino-pack")
    parser.add_argument("--archive", type=Path, default=ROOT / "release/seedvr2-openvino-pack.zip")
    args = parser.parse_args()
    output = target(args.out, ROOT / "dist", True)
    archive_output = target(args.archive, ROOT / "release", False)
    cache = ROOT / "dist/seedvr2-openvino-pack-cache/artifacts"
    cache.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="prepared-", dir=cache.parent))
    inventory = json.loads((ASSETS / "assets.json").read_text())
    for name, (size, sha) in ARCHIVES.items():
        check(ASSETS / name, size, sha)
        unpack_fixed(ASSETS / name, work / ("licenses" if name == "notices.zip" else "structure" if name == "structure.zip" else "rife"), inventory[name]["files"])
    wheel = fetch(cache, WHEEL[0], WHEEL[1], WHEEL[2], WHEEL[3] + WHEEL[0])
    sdk = work / "sdk"
    with zipfile.ZipFile(wheel) as archive:
        seen = set()
        for info in archive.infolist():
            if not info.filename.startswith(("openvino/include/", "openvino/cmake/", "openvino/libs/")) or info.is_dir():
                continue
            if not safe(info.filename) or info.filename in seen or stat.S_ISLNK(info.external_attr >> 16) or info.file_size > 128 * 1024**2:
                raise ValueError("invalid authenticated SDK entry")
            seen.add(info.filename)
            write(sdk / info.filename, archive.read(info))
    crypto = work / "openssl"
    for name, size, sha in DEBS:
        package = fetch(cache, name, size, sha, "https://us.archive.ubuntu.com/ubuntu/pool/main/o/openssl/" + name)
        subprocess.run(["dpkg-deb", "-x", str(package), str(crypto)], check=True)
    runtime = work / "runtime/lib"
    runtime.mkdir(parents=True)
    pins = json.loads(subprocess.check_output(["node", "--input-type=module", "-e",
        "import {payloadPins} from './scripts/ai-pack-pins.mjs'; console.log(JSON.stringify(payloadPins))"], cwd=ROOT))
    for name, pin in pins.items():
        if not name.startswith("lib/"):
            continue
        basename = name[4:]
        source = crypto / "usr/lib/x86_64-linux-gnu/libcrypto.so.3" if basename == "libcrypto.so.3" else sdk / "openvino/libs" / basename
        check(source, pin["bytes"], pin["sha256"])
        shutil.copyfile(source, runtime / basename)
    if args.prepare_only:
        print("PASS pinned assets + wheel SDK/runtime + matched OpenSSL headers/library; GPU_queries=0 learned_bytes=0; staging=" + str(work))
        return
    build = work / "native-build"  # CMake's cached library paths must belong to this SDK instance.
    env = {**os.environ, "LD_LIBRARY_PATH": str(runtime) + ":" + str(sdk / "openvino/libs"), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "otc_CPU_ONLY": "1"}
    subprocess.run(["cmake", "-S", str(args.native_root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DAI_ENHANCE_BUILD_TESTS=ON",
        "-DOpenVINO_DIR=" + str(sdk / "openvino/cmake"), "-DOPENSSL_INCLUDE_DIR=" + str(crypto / "usr/include"),
        "-DOPENSSL_CRYPTO_LIBRARY=" + str(runtime / "libcrypto.so.3"),
        "-DCMAKE_CXX_FLAGS=-I" + str(crypto / "usr/include/x86_64-linux-gnu")], check=True, env=env)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True, env=env)
    subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure"], check=True, env=env)
    subprocess.run(["node", str(ROOT / "scripts/build-ai-pack-assemble.mjs"), "--runtime", str(work / "runtime"), "--worker", str(build / "seedvr2-openvino"),
        "--models", str(work / "structure/seedvr"), "--complements", str(work / "structure/seedvr-complements"), "--rife", str(work / "rife"),
        "--licenses", str(work / "licenses"), "--provenance", str(ASSETS / "provenance.json"),
        "--out", str(work / "pack"), "--archive", str(work / "pack.zip")], check=True, env=env)
    subprocess.run(["node", str(ROOT / "scripts/verify-ai-pack.mjs"), "--root", str(work / "pack"), "--archive", str(work / "pack.zip")], check=True, env=env)
    # Two independent publications, not an atomic directory+archive transaction.
    backups = {"directory": publish(work / "pack", output, True),
               "archive": publish(work / "pack.zip", archive_output, False)}
    print(json.dumps({"published": str(output), "archive": str(archive_output), "preserved_backups": backups}))


if __name__ == "__main__":
    main()
