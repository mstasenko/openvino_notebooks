"""Additive preparation seal; original runtime authority is inherited unchanged."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "source-phase-major-uhd-plan-20261011"
PARENT_TABLE = "b4f05125d14c8da1e75b16d19112bb59b243734a24d44dbfbf4bf61e3c0d7cfc"
PARENT_HELPER = "d406d53a7ac9abf27cce1eb2facb6421865afa1b646cd1702ce12b01a8b517b9"
PRODUCER = HERE.parent / "matched-throughput-source-20261011/pilot.py"
PRODUCER_SHA = "1a0f83e9ea15784d39999b0aa870271b38bd5f258c5fb19b9cecf6fa6a151095"
FIXTURE_RECEIPT = PRODUCER.parent / "fixture/receipt.json"
FIXTURE_RECEIPT_SHA = "3fa02f92b26bf731892949695770421524af04b3a123f58670ab7277b1cc1c91"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if sha(PARENT / "seal.json") != PARENT_TABLE or sha(PARENT / "seal.py") != PARENT_HELPER:
    raise ValueError("Frozen phase-major parent changed")
parent = load(PARENT / "seal.py", "original_phase_major_seal")
authority = parent.authority
load_probe = parent.load_probe

def load_fixture_producer():
    if sha(PRODUCER) != PRODUCER_SHA:
        raise ValueError("Frozen fixture producer changed")
    if sha(FIXTURE_RECEIPT) != FIXTURE_RECEIPT_SHA:
        raise ValueError("Frozen UHD fixture receipt changed")
    return load(PRODUCER, "frozen_uhd_fixture_producer")

def verify_local():
    for name, expected in json.loads((HERE / "seal.json").read_text())["files"].items():
        if authority.pin(Path(name)) != expected:
            raise ValueError("UHD additive source/fixture changed: " + name)

def verify():
    parent.verify()  # Exact original full runtime closure, no model materialization.
    verify_local()
    print("PASS additive UHD authority; no Torch/model/device operations")

def freeze():
    # Preparation does not rehash the inherited multi-GB runtime/checkpoints.
    files = [HERE / name for name in ("benchmark.py", "seal.py", "test_contracts.py", "README.md", "admission.py", "run-root.sh", "run-inside.sh", "check-scope.py",
                                     "BENCHMARK-DIFF.patch", "CPU-RESULTS.json")]
    files += [PARENT / "seal.json", PARENT / "seal.py", PARENT / "benchmark.py", PRODUCER]
    fixture = PRODUCER.parent / "fixture"
    files += [fixture / name for name in ("receipt.json", "frame01.rgb24", "frame03.rgb24")]
    files += [HERE.parent / "uhd-frame-prototype/fixture" / name
              for name in ("resized.rgb24", "noise.hwc.f32", "text_positive.f32")]
    with (HERE / "seal.json").open("x") as stream:
        json.dump({"parent_table": PARENT_TABLE, "gpu_prepared_only": True,
                   "runtime_policy_changed": False,
                   "files": {str(path): authority.pin(path) for path in files}}, stream, indent=2)
        stream.write("\n")
    print("Additive freeze PASS; no model copies/runtime rehash")

if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["verify-local"]:
        verify_local()
        print("Local additive verification PASS")
    elif sys.argv[1:] == ["verify"]:
        verify()
    else:
        raise ValueError("freeze, verify-local or verify required")
