"""Bound every experiment in its own systemd scope.

This module deliberately fails closed.  It does not fall back to an
unbounded child when systemd or cgroup memory accounting is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from reference_paths import SCRATCH_ROOT


GIB = 1 << 30
MAX_BYTES = 24 * GIB
# Keep the hard ceiling; throttling below it distorted the GPU comparison.
HIGH_BYTES = MAX_BYTES
RESERVE_BYTES = 12 * GIB
MIN_CAP_BYTES = 1 * GIB

SYSTEMD_RUN = Path("/usr/bin/systemd-run")
SYSTEMCTL = Path("/usr/bin/systemctl")
FINDMNT = Path("/usr/bin/findmnt")
MEMINFO = Path("/proc/meminfo")
CGROUP_ROOT = Path("/sys/fs/cgroup")
SCRATCH_ANCHOR = SCRATCH_ROOT.parent

_UNIT_RE = re.compile(r"^otc-experiment-[a-f0-9]{16}\.scope$")
_SHOW_FIELDS = (
    "Id",
    "ControlGroup",
    "MainPID",
    "ActiveState",
    "SubState",
    "MemoryAccounting",
    "MemoryCurrent",
    "MemoryPeak",
    "MemoryMax",
    "MemoryHigh",
    "MemorySwapMax",
    "MemorySwapPeak",
    "Result",
    "OOMPolicy",
)
_CGROUP_FILES = (
    "memory.current",
    "memory.peak",
    "memory.max",
    "memory.high",
    "memory.swap.max",
    "memory.swap.current",
    "memory.swap.peak",
    "memory.oom.group",
    "memory.events",
    "cgroup.procs",
)


def _mem_available_bytes() -> int:
    for line in MEMINFO.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "MemAvailable:" and len(fields) >= 2:
            try:
                value = int(fields[1])
            except ValueError as exc:
                raise RuntimeError("MemAvailable is malformed") from exc
            if value < 0 or (len(fields) >= 3 and fields[2] != "kB"):
                raise RuntimeError("MemAvailable is invalid")
            return value * 1024
    raise RuntimeError("MemAvailable is unavailable; refusing an unbounded run")


def _systemd_manager_ready() -> bool:
    try:
        result = subprocess.run(
            [str(SYSTEMCTL), "show", "--no-pager", "--value", "-p", "Version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _require_support() -> None:
    if os.geteuid() != 0:
        raise PermissionError("experiment scopes require root")
    if not SYSTEMD_RUN.is_file() or not os.access(SYSTEMD_RUN, os.X_OK):
        raise RuntimeError(f"systemd-run is unavailable: {SYSTEMD_RUN}")
    if not SYSTEMCTL.is_file() or not os.access(SYSTEMCTL, os.X_OK):
        raise RuntimeError(f"systemctl is unavailable: {SYSTEMCTL}")
    if not _systemd_manager_ready():
        raise RuntimeError("systemd manager is unavailable")
    if not CGROUP_ROOT.is_dir():
        raise RuntimeError(f"cgroup root is unavailable: {CGROUP_ROOT}")
    controllers = CGROUP_ROOT / "cgroup.controllers"
    if not controllers.is_file() or "memory" not in controllers.read_text().split():
        raise RuntimeError("cgroup memory controller is unavailable")
    if not (CGROUP_ROOT / "cgroup.procs").is_file():
        raise RuntimeError("cgroup process membership is unavailable")


def _scratch_filesystem(path: Path) -> str:
    """Require a mounted, non-RAM filesystem for worker temporary files."""

    try:
        if path.stat().st_dev != SCRATCH_ANCHOR.stat().st_dev:
            raise RuntimeError("experiment scratch is on a different filesystem than the configured root")
    except OSError as exc:
        raise RuntimeError(f"cannot stat experiment scratch: {path}") from exc
    if not FINDMNT.is_file() or not os.access(FINDMNT, os.X_OK):
        raise RuntimeError(f"findmnt is unavailable: {FINDMNT}")
    try:
        result = subprocess.run(
            [str(FINDMNT), "-T", str(path), "--noheadings", "--output", "FSTYPE"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot verify experiment scratch filesystem") from exc
    fstype = result.stdout.strip().split()[0] if result.returncode == 0 and result.stdout.strip() else ""
    if not fstype or fstype.lower() in {"tmpfs", "ramfs"}:
        raise RuntimeError(f"unsafe experiment scratch filesystem: {fstype or 'unknown'}")
    return fstype


def _new_unit(lifecycle: Path) -> str:
    material = f"{lifecycle.resolve()}:{os.getpid()}:{time.time_ns()}".encode()
    return f"otc-experiment-{hashlib.sha256(material).hexdigest()[:16]}.scope"


def memory_limit(lifecycle: Path, env: MutableMapping[str, str]) -> tuple[list[str], dict[str, Any]]:
    """Return a bounded ``systemd-run`` prefix and the launch receipt.

    ``env`` is intentionally mutated so the caller's existing worker launch
    path retains every inherited setting while receiving an isolated scratch
    directory.  There is no unbounded fallback.
    """

    if not isinstance(env, MutableMapping):
        raise TypeError("env must be a mutable environment mapping")
    lifecycle = Path(lifecycle).resolve()
    _require_support()
    available = _mem_available_bytes()
    cap = min(MAX_BYTES, available - RESERVE_BYTES)
    if cap <= MIN_CAP_BYTES:
        raise RuntimeError(
            f"insufficient MemAvailable for bounded experiment: {available} bytes"
        )
    high = min(HIGH_BYTES, cap)
    unit = _new_unit(lifecycle)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    scratch = SCRATCH_ROOT / unit
    try:
        scratch.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError(f"experiment scratch collision: {scratch}") from exc
    os.chmod(scratch, 0o700)
    fstype = _scratch_filesystem(scratch)
    for key in ("TMPDIR", "TMP", "TEMP"):
        env[key] = str(scratch)

    prefix = [
        str(SYSTEMD_RUN),
        "--scope",
        "--quiet",
        f"--unit={unit}",
        "-p",
        "MemoryAccounting=yes",
        "-p",
        f"MemoryMax={cap}",
        "-p",
        f"MemoryHigh={high}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=kill",
        "--",
    ]
    receipt: dict[str, Any] = {
        "schema": 1,
        "lifecycle": str(lifecycle),
        "unit": unit,
        "scope": True,
        "mem_available_bytes": available,
        "available_bytes": available,
        "cap_bytes": cap,
        "high_bytes": high,
        "swap_max_bytes": 0,
        "oom_policy": "kill",
        "oom_group": 1,
        "scratch": str(scratch),
        "scratch_fstype": fstype,
        "scratch_root": str(SCRATCH_ROOT),
        "systemd_run": str(SYSTEMD_RUN),
        "systemctl": str(SYSTEMCTL),
        "cgroup_root": str(CGROUP_ROOT),
        "ownership_policy": {
            "fail_closed": True,
            "require_exact_unit": True,
            "require_control_group": True,
            "require_memory_files": list(_CGROUP_FILES),
            "require_nonempty_procs_while_running": False,
        },
    }
    policy_path = lifecycle / "memory-policy.json"
    receipt["policy_path"] = str(policy_path)
    prefix.extend((sys.executable, str(Path(__file__).resolve()), "--exec", str(policy_path), "--"))
    receipt["prefix"] = prefix
    try:
        policy_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.chmod(policy_path, 0o600)
    except OSError as exc:
        raise RuntimeError(f"cannot write memory policy receipt: {policy_path}") from exc
    return prefix, receipt


def _parse_show(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key] = value
    return values


def _safe_cgroup_path(control_group: str) -> Path | None:
    if not control_group.startswith("/") or ".." in Path(control_group).parts:
        return None
    root = CGROUP_ROOT.resolve()
    candidate = (root / control_group.lstrip("/")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _read_cgroup(path: Path) -> tuple[dict[str, str], dict[str, int], list[int], list[str]]:
    values: dict[str, str] = {}
    events: dict[str, int] = {}
    procs: list[int] = []
    errors: list[str] = []
    for name in _CGROUP_FILES:
        target = path / name
        try:
            text = target.read_text(encoding="utf-8").strip()
        except OSError as exc:
            errors.append(f"{name}: {exc}")
            continue
        values[name] = text
        if name == "memory.events":
            for line in text.splitlines():
                fields = line.split()
                if len(fields) != 2:
                    errors.append(f"memory.events malformed: {line!r}")
                    continue
                try:
                    events[fields[0]] = int(fields[1])
                except ValueError:
                    errors.append(f"memory.events non-integer: {line!r}")
        elif name == "cgroup.procs":
            for value in text.splitlines():
                try:
                    procs.append(int(value))
                except ValueError:
                    errors.append(f"cgroup.procs non-integer: {value!r}")
    return values, events, sorted(procs), errors


def _limit_matches(value: str | None, expected: int) -> bool:
    try:
        return int(value or "") == expected
    except ValueError:
        return False


def _policy_limits(receipt: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        raw_cap = receipt["cap_bytes"]
        raw_high = receipt["high_bytes"]
        raw_swap = receipt["swap_max_bytes"]
        if any(isinstance(value, bool) for value in (raw_cap, raw_high, raw_swap)):
            raise ValueError("boolean memory limit")
        cap = int(raw_cap)
        high = int(raw_high)
        swap = int(raw_swap)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("memory policy limits are malformed") from exc
    if cap <= 0 or cap > MAX_BYTES:
        raise RuntimeError("memory policy cap is outside the safe range")
    if high <= 0 or high > cap:
        raise RuntimeError("memory policy high limit is outside the safe range")
    if swap != 0:
        raise RuntimeError("memory policy swap limit is not zero")
    return cap, high, swap


def _self_cgroup_path() -> str:
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        hierarchy, separator, rest = line.partition(":")
        if separator and hierarchy == "0":
            controller, separator, path = rest.partition(":")
            if separator and not controller:
                return path
    raise RuntimeError("unified self cgroup is unavailable")


def process_snapshot(
    *,
    control_group: str | None = None,
    status_path: Path = Path("/proc/self/status"),
) -> dict[str, Any]:
    """Return trusted process and current unified-cgroup memory evidence.

    This is intentionally independent of systemd and never invokes a child
    process.  The caller may provide a status path (and tests may provide the
    already-read cgroup identity); production callers use both defaults.
    Numeric cgroup fields in ``memory`` are bytes/counters, while ``cgroup``
    retains the existing raw file names and values for audit receipts.
    """

    control_group = _self_cgroup_path() if control_group is None else control_group
    if (
        not isinstance(control_group, str)
        or not control_group.startswith("/")
        or "\x00" in control_group
        or not _UNIT_RE.fullmatch(Path(control_group).name)
    ):
        raise RuntimeError("worker cgroup scope identity is invalid")
    try:
        cgroup_path = _safe_cgroup_path(control_group)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("worker cgroup path is invalid") from exc
    if cgroup_path is None or not cgroup_path.is_dir():
        raise RuntimeError("worker cgroup path is unavailable or escaping")

    values, events, procs, errors = _read_cgroup(cgroup_path)
    if errors:
        raise RuntimeError("worker cgroup evidence is incomplete: " + "; ".join(errors))
    if os.getpid() not in procs or any(pid <= 0 for pid in procs):
        raise RuntimeError("worker pid is not owned by its cgroup")
    if any(not re.fullmatch(r"[0-9]+", line) for line in values["cgroup.procs"].splitlines()):
        raise RuntimeError("worker cgroup cgroup.procs is malformed")
    required = (
        "memory.current", "memory.peak", "memory.max", "memory.high",
        "memory.swap.max", "memory.swap.current", "memory.swap.peak",
        "memory.oom.group",
    )
    if any(name not in values for name in required):
        raise RuntimeError("worker cgroup memory evidence is incomplete")

    def nonnegative(name: str) -> int:
        raw = values.get(name, "")
        if not re.fullmatch(r"[0-9]+", raw):
            raise RuntimeError(f"worker cgroup {name} is not a nonnegative integer")
        return int(raw)

    numbers = {name: nonnegative(name) for name in required if name != "memory.oom.group"}
    if numbers["memory.max"] <= 0 or numbers["memory.max"] > MAX_BYTES:
        raise RuntimeError("worker cgroup memory.max is outside the safe range")
    if numbers["memory.high"] != numbers["memory.max"]:
        raise RuntimeError("worker cgroup memory.high does not equal memory.max")
    if numbers["memory.swap.max"] != 0:
        raise RuntimeError("worker cgroup swap.max is not zero")
    if values["memory.oom.group"] != "1":
        raise RuntimeError("worker cgroup memory.oom.group is not 1")
    if numbers["memory.peak"] < numbers["memory.current"]:
        raise RuntimeError("worker cgroup memory.peak precedes memory.current")
    if numbers["memory.swap.peak"] < numbers["memory.swap.current"]:
        raise RuntimeError("worker cgroup swap.peak precedes swap.current")

    event_names = ("low", "high", "max", "oom", "oom_kill", "oom_group_kill")
    if any(name not in events or events[name] < 0 for name in event_names):
        raise RuntimeError("worker cgroup memory.events is incomplete or negative")
    seen_events: set[str] = set()
    for line in values["memory.events"].splitlines():
        fields = line.split()
        if (
            len(fields) != 2
            or fields[0] in seen_events
            or not re.fullmatch(r"[0-9]+", fields[1])
        ):
            raise RuntimeError("worker cgroup memory.events is malformed")
        seen_events.add(fields[0])

    status: dict[str, int] = {}
    try:
        status_lines = Path(status_path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("/proc/self/status is unavailable") from exc
    for line in status_lines:
        fields = line.split()
        if not fields or fields[0] not in ("VmRSS:", "VmHWM:"):
            continue
        name = fields[0][:-1]
        if name in status or len(fields) != 3 or fields[2] != "kB" or not re.fullmatch(r"[0-9]+", fields[1]):
            raise RuntimeError(f"/proc/self/status {name} is malformed")
        status[name] = int(fields[1]) * 1024
    if set(status) != {"VmRSS", "VmHWM"}:
        raise RuntimeError("/proc/self/status lacks strict VmRSS/VmHWM fields")

    return {
        "schema": 1,
        "pid": os.getpid(),
        "control_group": control_group,
        "cgroup_path": str(cgroup_path),
        "process": status,
        "memory": {
            "current": numbers["memory.current"],
            "peak": numbers["memory.peak"],
            "max": numbers["memory.max"],
            "high": numbers["memory.high"],
            "swap_max": numbers["memory.swap.max"],
            "swap_current": numbers["memory.swap.current"],
            "swap_peak": numbers["memory.swap.peak"],
            "events": events,
        },
        "cgroup": {
            "path": str(cgroup_path),
            "files": values,
            "events": events,
            "procs": procs,
        },
    }


def _verified_child_cgroup(
    receipt: Mapping[str, Any],
) -> tuple[Path, dict[str, str], dict[str, int], list[int]]:
    unit = str(receipt.get("unit", ""))
    if not _UNIT_RE.fullmatch(unit):
        raise RuntimeError("invalid memory policy unit")
    cap, high, swap = _policy_limits(receipt)
    control_group = _self_cgroup_path()
    if not control_group.startswith("/") or Path(control_group).name != unit:
        raise RuntimeError(f"worker is not in expected scope: {control_group!r}")
    path = _safe_cgroup_path(control_group)
    if path is None or not path.is_dir():
        raise RuntimeError("worker cgroup path is unavailable")
    values, events, procs, errors = _read_cgroup(path)
    if errors:
        raise RuntimeError("worker cgroup evidence is incomplete: " + "; ".join(errors))
    expected = {
        "memory.max": cap,
        "memory.high": high,
        "memory.swap.max": swap,
    }
    for name, value in expected.items():
        if not _limit_matches(values.get(name), value):
            raise RuntimeError(f"worker cgroup {name} does not match policy")
    if values.get("memory.oom.group") != "1":
        raise RuntimeError("worker cgroup memory.oom.group is not 1")
    return path, values, events, procs


def memory_complete(
    receipt: Mapping[str, Any],
    verified: Mapping[str, Any],
    final: Mapping[str, Any],
) -> bool:
    """Accept only complete, exact-scope evidence with no memory OOM signal."""

    try:
        cap, high, swap = _policy_limits(receipt)
        unit = str(receipt["unit"])
        limits = verified["limits"]
        if verified.get("verified") is not True or verified.get("unit") != unit:
            return False
        if Path(str(verified["control_group"])).name != unit:
            return False
        if not _limit_matches(str(limits["memory.max"]), cap):
            return False
        if not _limit_matches(str(limits["memory.high"]), high):
            return False
        if not _limit_matches(str(limits["memory.swap.max"]), swap):
            return False
        if str(limits["memory.oom.group"]) != "1":
            return False

        if final.get("unit") != unit:
            return False
        ownership = final["ownership"]
        if not isinstance(ownership, Mapping) or ownership.get("ok") is not True:
            return False
        systemctl = final["systemctl"]
        result = str(systemctl["Result"]).strip().lower()
        if result != "success":
            return False
        cgroup = final["cgroup"]
        events = cgroup["events"]
        for key in ("oom", "oom_kill", "oom_group_kill"):
            if key not in events or int(events[key]) != 0:
                return False
        files = cgroup["files"]
        if int(files["memory.swap.current"]) != 0 or int(files["memory.swap.peak"]) != 0:
            return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError, RuntimeError):
        return False


def _set_oom_score_adj() -> None:
    if os.geteuid() != 0:
        return
    target = Path("/proc/self/oom_score_adj")
    try:
        target.write_text("1000\n", encoding="ascii")
        if target.read_text(encoding="ascii").strip() != "1000":
            raise RuntimeError("oom_score_adj was not applied")
    except OSError as exc:
        raise RuntimeError("cannot set root worker oom_score_adj=1000") from exc


def _exec_verified(policy_path: Path, worker_argv: list[str]) -> int:
    if not worker_argv:
        raise ValueError("verified scope worker command is empty")
    receipt = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("memory policy receipt is not an object")
    path, values, events, procs = _verified_child_cgroup(receipt)
    _set_oom_score_adj()
    verified = {
        "schema": 1,
        "verified": True,
        "pid": os.getpid(),
        "unit": receipt["unit"],
        "control_group": _self_cgroup_path(),
        "cgroup_path": str(path),
        "limits": {
            "memory.max": values["memory.max"],
            "memory.high": values["memory.high"],
            "memory.swap.max": values["memory.swap.max"],
            "memory.oom.group": values["memory.oom.group"],
        },
        "memory": {
            "current": values["memory.current"],
            "peak": values["memory.peak"],
            "swap_current": values["memory.swap.current"],
            "swap_peak": values["memory.swap.peak"],
            "events": events,
        },
        "member_pids": procs,
        "oom_score_adj": 1000 if os.geteuid() == 0 else None,
        "worker_argv": worker_argv,
    }
    verified_path = Path(str(receipt["lifecycle"])) / "memory-verified.json"
    verified_path.write_text(json.dumps(verified, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(verified_path, 0o600)
    # Keep the same inherited environment and let the wrapper return naturally.
    completed = subprocess.run(worker_argv, env=os.environ, check=False)
    final = snapshot(receipt)
    final_path = Path(str(receipt["lifecycle"])) / "memory-final.json"
    final_path.write_text(json.dumps(final, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(final_path, 0o600)
    return completed.returncode if completed.returncode >= 0 else 128 - completed.returncode


def _main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3 or args[0] != "--exec" or args[2] != "--":
        raise SystemExit("usage: experiment_memory.py --exec POLICY.json -- WORKER [ARGS...]")
    return _exec_verified(Path(args[1]).resolve(), args[3:])


def snapshot(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Capture exact-unit and cgroup memory evidence, failing closed on trust gaps."""

    unit = str(receipt.get("unit", ""))
    if not _UNIT_RE.fullmatch(unit):
        raise ValueError("invalid experiment scope unit")
    command = [str(SYSTEMCTL), "show", unit, "--no-pager", "--no-legend"]
    for field in _SHOW_FIELDS:
        command.extend(("-p", field))
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
        show_error = None if result.returncode == 0 else f"systemctl exit {result.returncode}"
        show = _parse_show(result.stdout)
    except (OSError, subprocess.SubprocessError) as exc:
        result = None
        show_error = str(exc)
        show = {}

    reasons: list[str] = []
    if show_error:
        reasons.append(show_error)
    if show.get("Id") != unit:
        reasons.append("systemd unit identity mismatch")
    if show.get("MemoryAccounting") != "yes":
        reasons.append("systemd MemoryAccounting is not yes")
    if show.get("OOMPolicy") != "kill":
        reasons.append("systemd OOMPolicy is not kill")
    control_group = show.get("ControlGroup", "")
    cgroup_path = _safe_cgroup_path(control_group)
    if cgroup_path is None:
        reasons.append("invalid or missing ControlGroup")
    elif Path(control_group).name != unit:
        reasons.append("ControlGroup is not the exact experiment scope")

    cgroup_values: dict[str, str] = {}
    events: dict[str, int] = {}
    procs: list[int] = []
    cgroup_errors: list[str] = []
    if cgroup_path is not None:
        cgroup_values, events, procs, cgroup_errors = _read_cgroup(cgroup_path)
        reasons.extend(cgroup_errors)
        try:
            cap, high, swap = _policy_limits(receipt)
            expected = {
                "memory.max": cap,
                "memory.high": high,
                "memory.swap.max": swap,
            }
            for name, value in expected.items():
                if not _limit_matches(cgroup_values.get(name), value):
                    reasons.append(f"{name} does not match requested policy")
        except RuntimeError as exc:
            reasons.append(str(exc))
        if cgroup_values.get("memory.oom.group") != "1":
            reasons.append("memory.oom.group is not 1")
    ownership = {
        "ok": not reasons,
        "fail_closed": True,
        "unit_exact": show.get("Id") == unit,
        "control_group": control_group,
        "cgroup_path": str(cgroup_path) if cgroup_path is not None else None,
        "member_pids": procs,
        "errors": reasons,
    }
    return {
        "unit": unit,
        "systemctl_command": command,
        "systemctl": show,
        "systemctl_error": show_error,
        "systemctl_returncode": None if result is None else result.returncode,
        "cgroup": {
            "path": str(cgroup_path) if cgroup_path is not None else None,
            "files": cgroup_values,
            "events": events,
            "procs": procs,
            "errors": cgroup_errors,
        },
        "ownership": ownership,
    }


if __name__ == "__main__":
    raise SystemExit(_main())

