#!/usr/bin/env python3
"""Lightweight no-model dispatch checks for the resumable video qualifier."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types


SCRIPT = Path(__file__).with_name("openvino-video-probe.py")


class FakeTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")

    def set_num_threads(self, _threads: int) -> None:
        return None

    def set_num_interop_threads(self, _threads: int) -> None:
        return None

    def manual_seed(self, _seed: int) -> None:
        return None


def load_probe() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("seedvr2_video_probe_dispatch", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(module: types.ModuleType, phase: str, output_dir: Path) -> tuple[int, dict]:
    fake_torch = FakeTorch()
    fake_numpy = types.ModuleType("numpy")
    fake_openvino = types.ModuleType("openvino")
    helper = types.SimpleNamespace(force_cpu_only=lambda _torch: {"cuda": False})
    original_modules = {
        name: sys.modules.get(name)
        for name in ("torch", "numpy", "openvino")
    }
    original_load_helper = module.load_helper
    original_validate_paths = module.validate_paths
    original_prepare = module.run_dit_prepare_phase
    original_validate = module.run_dit_validate_phase
    calls: list[str] = []

    def load_helper_stub(_path: Path, _name: str) -> types.SimpleNamespace:
        return helper

    def validate_paths_stub(*_args: object) -> None:
        return None

    def prepare_stub(*args: object) -> None:
        calls.append("dit-prepare")
        result = args[7]
        result["stages"]["dit-prepare"] = {
            "status": "prepared_unvalidated",
            "candidate_validation_status": "UNVALIDATED",
            "validation_required": True,
        }

    def validate_stub(*args: object) -> None:
        calls.append("dit-validate")
        result = args[6]
        result["stages"]["dit-validate"] = {"status": "passed"}
        result["artifacts"]["dit_ir"] = str(output_dir / "ir" / "dit.xml")

    try:
        sys.modules.update({"torch": fake_torch, "numpy": fake_numpy, "openvino": fake_openvino})
        module.load_helper = load_helper_stub
        module.validate_paths = validate_paths_stub
        module.run_dit_prepare_phase = prepare_stub
        module.run_dit_validate_phase = validate_stub
        argv = [
            "--source",
            str(output_dir / "source"),
            "--inputs-dir",
            str(output_dir / "inputs"),
            "--vae-weights",
            str(output_dir / "vae.safetensors"),
            "--dit-weights",
            str(output_dir / "dit.safetensors"),
            "--output-dir",
            str(output_dir),
            "--phase",
            phase,
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            status = module.main(argv)
        evidence = json.loads((output_dir / "evidence.json").read_text(encoding="utf-8"))
        assert calls == [phase], calls
        return status, evidence
    finally:
        module.load_helper = original_load_helper
        module.validate_paths = original_validate_paths
        module.run_dit_prepare_phase = original_prepare
        module.run_dit_validate_phase = original_validate
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def main() -> None:
    module = load_probe()
    with tempfile.TemporaryDirectory(prefix="seedvr2-dispatch-") as temporary:
        output_dir = Path(temporary)
        prepare_exit, prepare_evidence = run_main(module, "dit-prepare", output_dir / "prepare")
        assert prepare_exit == 0, prepare_evidence
        assert prepare_evidence["status"] == "prepared_unvalidated", prepare_evidence
        assert prepare_evidence["validation_required"] is True, prepare_evidence
        assert prepare_evidence["stages"]["dit-prepare"]["candidate_validation_status"] == "UNVALIDATED"

        validate_exit, validate_evidence = run_main(module, "dit-validate", output_dir / "validate")
        assert validate_exit == 0, validate_evidence
        assert validate_evidence["status"] == "passed", validate_evidence
        assert validate_evidence["stages"]["dit-validate"]["status"] == "passed"
        assert validate_evidence["artifacts"]["dit_ir"].endswith("/ir/dit.xml")
    print("dispatch stub checks passed: dit-prepare exit0/unvalidated; dit-validate exit0/promotion artifact")


if __name__ == "__main__":
    main()
