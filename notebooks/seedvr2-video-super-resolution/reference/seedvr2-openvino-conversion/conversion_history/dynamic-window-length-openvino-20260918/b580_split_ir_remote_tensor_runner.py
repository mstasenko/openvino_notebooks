#!/usr/bin/env python3
"""Finite GPU.1 proof of the existing split prefix/chunk/suffix IRs.

The runner requires a fresh, independently produced B580 preflight marker and
an explicit new receipt path. It never discovers devices or selects a fallback.
All 32 weightless chunk IRs are checked on CPU; one shared chunk model/request
then consumes each block's six prefix QKV RemoteTensors without host copies.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
import resource
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import b580_chunked_window_validation_runner as base


HERE = Path(__file__).resolve().parent
DEVICE = "GPU.1"
TARGET = {
    "bdf": "0000:03:00.0",
    "driver": "xe",
    "render_node": "renderD128",
    "device": "B580",
}
BLOCKS = (0, 16, 31)
ALL_BLOCKS = tuple(range(32))
EXPECTED_WINDOWS = {0: 162, 16: 162, 31: 200}
PRODUCTION_SHAPE = (2, 135, 240)
CHUNK_SIZE = 16
MAX_MARKER_AGE_SECONDS = 60.0
UNSHIFTED_WINDOW_METHOD = "720pwin_by_size_bysize"
QKV_NAMES = ("vid_q", "vid_k", "vid_v", "txt_q", "txt_k", "txt_v")
VIDEO_QKV_NAMES = QKV_NAMES[:3]
TEXT_QKV_NAMES = QKV_NAMES[3:]
PREFIX_OUTPUTS = {
    **{name: (base.production_token_count(), base.HEADS, base.HEAD_DIM) for name in VIDEO_QKV_NAMES},
    **{name: (base.TXT_TOKENS, base.HEADS, base.HEAD_DIM) for name in TEXT_QKV_NAMES},
}
CHUNK_INPUT_SHAPES = {
    **{name: (-1, base.HEADS, base.HEAD_DIM) for name in VIDEO_QKV_NAMES},
    **{name: (base.TXT_TOKENS, base.HEADS, base.HEAD_DIM) for name in TEXT_QKV_NAMES},
    "window_idx": (-1, -1),
    "window_key_mask": (-1, -1),
    "window_rope_freqs": (-1, -1, base.ROPE_DIM),
}
CHUNK_OUTPUTS = ("vid_window_out", "txt_sum", "txt_count")
DISTINCT_THRESHOLD = 1.0e-7
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class RunnerInputError(ValueError):
    """A missing, stale, or unsafe caller-supplied input."""


class ContractError(RuntimeError):
    """An IR, receipt, tensor, coverage, or execution-device contract failed."""


class QuietParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - main path
        raise RunnerInputError("invalid runner arguments")


def _parser() -> argparse.ArgumentParser:
    parser = QuietParser(
        description=(
            "Run the exact split IR cases on GPU.1. Requires a fresh JSON marker "
            "with status=pass, checked_at_utc, B580 target fields, and selector GPU.1. "
            "--help exits before any file or device access."
        )
    )
    parser.add_argument("--preflight-marker", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def _utc_epoch(value: Any) -> float:
    if not isinstance(value, str):
        raise RunnerInputError("preflight timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunnerInputError("preflight timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RunnerInputError("preflight timestamp is not UTC")
    return parsed.timestamp()


def parse_preflight_marker(text: str, *, now: float | None = None) -> dict[str, Any]:
    """Parse the exact, affirmative target marker without touching OpenVINO."""

    try:
        values = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise RunnerInputError("preflight marker is not valid JSON") from exc
    if not isinstance(values, Mapping) or values.get("status") != "pass":
        raise RunnerInputError("preflight marker is not affirmative")
    target = values.get("target", values)
    if not isinstance(target, Mapping):
        raise RunnerInputError("preflight target is not an object")
    bdf = target.get("bdf", target.get("pci_bdf"))
    driver = target.get("driver")
    render_node = str(target.get("render_node", "")).rstrip("/").rsplit("/", 1)[-1]
    raw_device = target.get("device", target.get("pci_device", ""))
    device = "B580" if str(raw_device) == "B580" or "B580" in str(raw_device) else ""
    selector = target.get("openvino_selector", values.get("openvino_selector"))
    if (bdf, driver, render_node, device, selector) != (
        TARGET["bdf"], TARGET["driver"], TARGET["render_node"], TARGET["device"], DEVICE
    ):
        raise RunnerInputError("preflight target does not match the GPU.1 contract")
    checked_at = _utc_epoch(values.get("checked_at_utc"))
    current = time.time() if now is None else float(now)
    age = current - checked_at
    if age < 0.0 or age > MAX_MARKER_AGE_SECONDS:
        raise RunnerInputError("preflight marker is stale")
    return {
        "checked_at_utc": values["checked_at_utc"],
        "age_seconds": round(age, 3),
        "target": dict(TARGET),
        "openvino_selector": DEVICE,
    }


def validate_preflight_marker(
    path: Path,
    *,
    now: float | None = None,
    max_age_seconds: float = MAX_MARKER_AGE_SECONDS,
) -> dict[str, Any]:
    if max_age_seconds <= 0:
        raise RunnerInputError("preflight freshness window is invalid")
    try:
        stat = path.stat()
        if not path.is_file():
            raise RunnerInputError("preflight marker is not a regular file")
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunnerInputError("preflight marker is unavailable") from exc
    current = time.time() if now is None else float(now)
    mtime_age = current - float(stat.st_mtime)
    if mtime_age < 0.0 or mtime_age > max_age_seconds:
        raise RunnerInputError("preflight marker file is stale")
    record = parse_preflight_marker(text, now=current)
    if record["age_seconds"] > max_age_seconds:
        raise RunnerInputError("preflight marker is stale")
    if abs(float(stat.st_mtime) - (current - record["age_seconds"])) > max_age_seconds:
        raise RunnerInputError("preflight marker timestamp and file age disagree")
    record["marker_name"] = path.name
    return record


def chunk_ranges(window_count: int, chunk_size: int = CHUNK_SIZE) -> tuple[tuple[int, int], ...]:
    ranges = base.chunk_ranges(window_count, chunk_size)
    flattened = [item for start, end in ranges for item in range(start, end)]
    if flattened != list(range(window_count)):
        raise ContractError("chunk ranges do not cover each window exactly once")
    return ranges


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError("current receipt or manifest is unavailable") from exc
    if not isinstance(value, dict):
        raise ContractError("current receipt or manifest is not an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_artifact(filename: Any) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ContractError("manifest artifact name is invalid")
    path = HERE / filename
    if path.is_symlink() or not path.is_file() or path.resolve().parent != HERE:
        raise ContractError("manifest artifact is unavailable")
    return path


def _receipt_path(block_index: int) -> Path:
    return HERE / f"receipt.chunked_decomposition_block{block_index}.json"


def _check_file_against_receipts(
    *,
    block_index: int,
    receipt: Mapping[str, Any],
    manifest_file: Mapping[str, Any],
    stage: str,
    extension: str,
) -> tuple[Path, str]:
    filename = manifest_file.get("file")
    path = _resolved_artifact(filename)
    expected_name = base.ARTIFACT_NAMES[stage][0 if extension == "xml" else 1].format(
        block=f"block{block_index}"
    )
    if path.name != expected_name:
        raise ContractError("manifest artifact is not the split IR used by the runner")
    size = path.stat().st_size
    if manifest_file.get("bytes") != size:
        raise ContractError("manifest artifact size changed")
    expected_hash = manifest_file.get("sha256")
    if not isinstance(expected_hash, str) or not HASH_RE.fullmatch(expected_hash):
        raise ContractError("manifest artifact hash is invalid")
    if _sha256(path) != expected_hash:
        raise ContractError("artifact hash differs from the current manifest")
    artifact = receipt.get("artifacts", {}).get(stage, {})
    receipt_path_value = artifact.get(extension)
    receipt_size_value = artifact.get(f"{extension}_bytes")
    if (
        not isinstance(receipt_path_value, str)
        or Path(receipt_path_value).resolve() != path.resolve()
        or receipt_size_value != size
    ):
        raise ContractError("artifact differs from its current block receipt")
    return path, expected_hash


def _dims(port: ET.Element) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for child in port.findall("./dim"):
        text = (child.text or "").strip()
        result.append(None if text in {"", "-1", "?"} else int(text))
    return tuple(result)


def _names(port: ET.Element) -> tuple[str, ...]:
    return tuple(name.strip() for name in port.get("names", "").split(",") if name.strip())


def _port_record(port: ET.Element) -> dict[str, Any]:
    return {"shape": list(_dims(port)), "precision": port.get("precision")}


def inspect_ir(path: Path) -> dict[str, Any]:
    """Return named model ports and a name-insensitive graph signature from IR XML."""

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ContractError("IR XML metadata cannot be parsed") from exc
    layers = root.findall("./layers/layer")
    edges = root.findall("./edges/edge")
    if not layers or not edges:
        raise ContractError("IR XML graph metadata is incomplete")
    by_id = {layer.get("id"): layer for layer in layers}
    if None in by_id or len(by_id) != len(layers):
        raise ContractError("IR layer identifiers are invalid")

    inputs: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for layer in layers:
        kind = layer.get("type")
        if kind == "Parameter":
            for port in layer.findall("./output/port"):
                for name in _names(port):
                    inputs[name] = _port_record(port)
        elif kind == "Result":
            for port in layer.findall("./input/port"):
                incoming = [e for e in edges if e.get("to-layer") == layer.get("id") and e.get("to-port") == port.get("id")]
                if len(incoming) != 1 or incoming[0].get("from-layer") not in by_id:
                    raise ContractError("IR result connection is unexpected")
                edge = incoming[0]
                producer = by_id[edge.get("from-layer")]
                source_ports = [p for p in producer.findall("./output/port") if p.get("id") == edge.get("from-port")]
                if len(source_ports) != 1:
                    raise ContractError("IR result source is unavailable")
                for name in _names(source_ports[0]):
                    outputs[name] = _port_record(source_ports[0])

    layers = root.findall("./layers/layer")
    layer_ids = {layer.get("id"): str(index) for index, layer in enumerate(layers)}
    for index, layer in enumerate(layers):
        layer.set("id", str(index))
    edge_parent = root.find("./edges")
    graph_edges = root.findall("./edges/edge")
    for edge in graph_edges:
        edge.set("from-layer", layer_ids[edge.get("from-layer")])
        edge.set("to-layer", layer_ids[edge.get("to-layer")])
    if edge_parent is not None:
        for edge in graph_edges:
            edge_parent.remove(edge)
        edge_parent.extend(sorted(
            graph_edges,
            key=lambda e: (e.get("from-layer"), e.get("from-port"), e.get("to-layer"), e.get("to-port")),
        ))
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"meta_data", "rt_info"}:
                parent.remove(child)
    for element in root.iter():
        if element.text is not None and not element.text.strip():
            element.text = None
        if element.tail is not None and not element.tail.strip():
            element.tail = None
        for key in ("name", "names", "originalLayersNames"):
            element.attrib.pop(key, None)
    fingerprint = hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()
    return {
        "inputs": dict(sorted(inputs.items())),
        "outputs": dict(sorted(outputs.items())),
        "graph_fingerprint": fingerprint,
        "layer_count": len(layers),
    }


def _expected_ir_contract(stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage == "prefix":
        return (
            {
                "vid": {"shape": [None, base.VID_DIM], "precision": "FP32"},
                "txt": {"shape": [base.TXT_TOKENS, base.VID_DIM], "precision": "FP32"},
                "emb": {"shape": [1, base.EMB_DIM], "precision": "FP32"},
            },
            {
                **{
                    name: {"shape": [None, base.HEADS, base.HEAD_DIM], "precision": "FP32"}
                    for name in VIDEO_QKV_NAMES
                },
                **{
                    name: {"shape": [base.TXT_TOKENS, base.HEADS, base.HEAD_DIM], "precision": "FP32"}
                    for name in TEXT_QKV_NAMES
                },
            },
        )
    if stage == "chunk":
        inputs = {
            **{name: {"shape": list(shape), "precision": "FP32"} for name, shape in CHUNK_INPUT_SHAPES.items()},
            "window_idx": {"shape": [None, None], "precision": "I64"},
            "window_key_mask": {"shape": [None, None], "precision": "BOOL"},
        }
        return (
            inputs,
            {
                "vid_window_out": {"shape": [None, None, base.INNER_DIM], "precision": "FP32"},
                "txt_sum": {"shape": [base.TXT_TOKENS, base.INNER_DIM], "precision": "FP32"},
                "txt_count": {"shape": [1], "precision": "FP32"},
            },
        )
    if stage == "suffix":
        return (
            {
                "vid": {"shape": [None, base.VID_DIM], "precision": "FP32"},
                "txt": {"shape": [base.TXT_TOKENS, base.VID_DIM], "precision": "FP32"},
                "emb": {"shape": [1, base.EMB_DIM], "precision": "FP32"},
                "attn_vid": {"shape": [None, base.INNER_DIM], "precision": "FP32"},
                "attn_txt": {"shape": [base.TXT_TOKENS, base.INNER_DIM], "precision": "FP32"},
            },
            {
                "vid_out": {"shape": [None, base.VID_DIM], "precision": "FP32"},
                "txt_out": {"shape": [base.TXT_TOKENS, base.VID_DIM], "precision": "FP32"},
            },
        )
    raise ContractError("unknown split IR stage")


def _same_port_contract(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    for name, wanted in expected.items():
        got = actual[name]
        if got.get("precision") != wanted["precision"]:
            return False
        got_shape = got.get("shape", [])
        if len(got_shape) != len(wanted["shape"]):
            return False
        if any(target not in (None, -1) and value != target for value, target in zip(got_shape, wanted["shape"])):
            return False
    return True


def _verify_stage_ir(path: Path, stage: str) -> dict[str, Any]:
    metadata = inspect_ir(path)
    expected_inputs, expected_outputs = _expected_ir_contract(stage)
    if not _same_port_contract(metadata["inputs"], expected_inputs):
        raise ContractError("IR input port contract differs from the runner")
    if not _same_port_contract(metadata["outputs"], expected_outputs):
        raise ContractError("IR output port contract differs from the runner")
    return metadata


def _production_metadata(receipt: Mapping[str, Any], block_index: int) -> dict[str, int]:
    candidates = receipt.get("window_contract", {}).get("production_audit", [])
    match = next(
        (row for row in candidates if tuple(row.get("shape", ())) == PRODUCTION_SHAPE),
        None,
    )
    if not isinstance(match, Mapping):
        raise ContractError("current receipt lacks the production shape audit")
    count = int(match.get("window_count", 0))
    max_len = int(match.get("max_video_len", 0))
    if count != EXPECTED_WINDOWS[block_index] or max_len != 405:
        raise ContractError("current production window contract changed")
    if len(chunk_ranges(count)) != math.ceil(count / CHUNK_SIZE):
        raise ContractError("current chunk accounting is inconsistent")
    return {"window_count": count, "max_video_len": max_len}


def validate_current_artifacts() -> dict[str, Any]:
    """Hash current IR files against manifest receipts and prove one shared chunk graph."""

    manifest_path = HERE / "manifest.chunked_decomposition_all_blocks.json"
    manifest = _load_json(manifest_path)
    if not (
        manifest.get("all_artifact_receipts_pass") is True
        and manifest.get("all_measured_matrix_parity_pass") is True
        and manifest.get("all_traced_forbidden_ops_absent") is True
    ):
        raise ContractError("current all-block manifest is not passing")
    entries = manifest.get("blocks")
    if not isinstance(entries, list) or len(entries) != len(ALL_BLOCKS):
        raise ContractError("current all-block manifest has incomplete coverage")

    receipts: dict[int, dict[str, Any]] = {}
    chunk_fingerprints: set[str] = set()
    chunk_ports: dict[str, Any] | None = None
    chunk_xml_hashes: set[str] = set()
    chunk_bin_hashes: set[str] = set()
    artifact_hashes: dict[str, Any] = {
        "manifest_sha256": _sha256(manifest_path),
        "block_receipts_sha256": {},
        "chunk_ir_sha256": {},
        "target_split_ir_sha256": {},
    }
    files_checked = 0
    for block_index in ALL_BLOCKS:
        entry = entries[block_index]
        if entry.get("block_index") != block_index or entry.get("status") != "pass":
            raise ContractError("current block manifest identity is invalid")
        receipt = _load_json(_receipt_path(block_index))
        if receipt.get("status") != "pass" or receipt.get("block_index") != block_index:
            raise ContractError("current block receipt is not passing")
        receipts[block_index] = receipt
        artifact_hashes["block_receipts_sha256"][str(block_index)] = _sha256(_receipt_path(block_index))
        stage_entry = entry.get("artifacts", {}).get("chunk", {}).get("files", {})
        chunk_xml = _check_file_against_receipts(
            block_index=block_index, receipt=receipt,
            manifest_file=stage_entry.get("xml", {}), stage="chunk", extension="xml"
        )
        chunk_bin = _check_file_against_receipts(
            block_index=block_index, receipt=receipt,
            manifest_file=stage_entry.get("bin", {}), stage="chunk", extension="bin"
        )
        files_checked += 2
        chunk_xml_hashes.add(chunk_xml[1])
        chunk_bin_hashes.add(chunk_bin[1])
        artifact_hashes["chunk_ir_sha256"][str(block_index)] = {
            "xml": chunk_xml[1], "bin": chunk_bin[1]
        }
        meta = _verify_stage_ir(chunk_xml[0], "chunk")
        chunk_fingerprints.add(meta["graph_fingerprint"])
        port_schema = {"inputs": meta["inputs"], "outputs": meta["outputs"]}
        if chunk_ports is None:
            chunk_ports = port_schema
        elif port_schema != chunk_ports:
            raise ContractError("chunk IR port names, shapes, or types differ across blocks")

    if len(chunk_bin_hashes) != 1 or len(chunk_fingerprints) != 1 or len(chunk_xml_hashes) > 4:
        raise ContractError("chunk IRs do not satisfy the shared-model equivalence proof")
    shared_chunk_proof = {
        "blocks_checked": len(ALL_BLOCKS),
        "unique_binary_hashes": len(chunk_bin_hashes),
        "normalized_graph_variants": len(chunk_fingerprints),
        "port_contract_variants": 1,
        "xml_hash_variants": len(chunk_xml_hashes),
        "compiled_source_block": 0,
    }

    windows: dict[int, dict[str, int]] = {}
    selected_artifacts: dict[str, Any] = {}
    for block_index in BLOCKS:
        receipt = receipts[block_index]
        model_args = receipt.get("model_args", {})
        if model_args.get("block_index") != block_index:
            raise ContractError("current block model identity is invalid")
        windows[block_index] = _production_metadata(receipt, block_index)
        helper_name = f"dynamic_window_reverse_gather_block{block_index}_export.py"
        source = receipt.get("source", {})
        if (
            Path(str(source.get("base_helper", ""))).name != helper_name
            or not HASH_RE.fullmatch(str(source.get("base_helper_sha256", "")))
            or _sha256(HERE / helper_name) != source.get("base_helper_sha256")
        ):
            raise ContractError("current window helper differs from its block receipt")
        if block_index == 16 and (
            model_args.get("window_method") != UNSHIFTED_WINDOW_METHOD
            or model_args.get("shifted", False) is not False
        ):
            raise ContractError("even block 16 is not using the corrected unshifted method")
        selected_artifacts[str(block_index)] = {}
        artifact_hashes["target_split_ir_sha256"][str(block_index)] = {}
        for stage in ("prefix", "suffix"):
            manifest_files = entry_for_block(entries, block_index, stage)
            pair = {}
            for extension in ("xml", "bin"):
                path, digest = _check_file_against_receipts(
                    block_index=block_index,
                    receipt=receipt,
                    manifest_file=manifest_files[extension],
                    stage=stage,
                    extension=extension,
                )
                pair[extension] = path
                artifact_hashes["target_split_ir_sha256"][str(block_index)].setdefault(stage, {})[
                    extension
                ] = digest
                files_checked += 1
            _verify_stage_ir(pair["xml"], stage)
            selected_artifacts[str(block_index)][stage] = {"files_checked": 2, "hashes_match": True}

    return {
        "receipts": receipts,
        "window_contracts": windows,
        "shared_chunk_ir": selected_artifacts,
        "artifact_hashes": artifact_hashes,
        "proof": {
            "artifact_files_checked": files_checked,
            "artifact_hashes_match": files_checked,
            "shared_chunk": shared_chunk_proof,
            "target_blocks": {
                str(block): {
                    **windows[block],
                    "chunk_calls": len(chunk_ranges(windows[block]["window_count"])),
                }
                for block in BLOCKS
            },
            "block16_unshifted": True,
        },
    }


def entry_for_block(entries: Sequence[Mapping[str, Any]], block_index: int, stage: str) -> Mapping[str, Any]:
    try:
        return entries[block_index]["artifacts"][stage]["files"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ContractError("manifest artifact stage is unavailable") from exc


def _rss_snapshot() -> dict[str, int]:
    peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        current_bytes = resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        current_bytes = peak_kib * 1024
    return {
        "current_rss_bytes": current_bytes,
        "peak_rss_kib": peak_kib,
        "peak_rss_bytes": peak_kib * 1024,
    }


def _compile_model(
    ov: Any,
    core: Any,
    context: Any,
    path: Path,
    expected_shapes: Mapping[str, Sequence[int]],
) -> tuple[Any, Any, dict[str, Any]]:
    before = _rss_snapshot()
    started = time.perf_counter()
    model = core.read_model(str(path))
    read_seconds = time.perf_counter() - started
    input_ports = {port.get_any_name(): port for port in model.inputs}
    if set(input_ports) != set(expected_shapes):
        raise ContractError("compiled IR input names differ from CPU metadata")
    started = time.perf_counter()
    model.reshape({name: ov.PartialShape(list(shape)) for name, shape in expected_shapes.items()})
    reshape_seconds = time.perf_counter() - started
    started = time.perf_counter()
    compiled = core.compile_model(
        model,
        context,
        {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": ov.Type.f16,
            "NUM_STREAMS": "1",
        },
    )
    compile_seconds = time.perf_counter() - started
    devices = [str(item) for item in compiled.get_property("EXECUTION_DEVICES")]
    if devices != [DEVICE]:
        raise ContractError("compiled model execution device is not exactly GPU.1")
    return model, compiled, {
        "read_model_seconds": read_seconds,
        "reshape_seconds": reshape_seconds,
        "compile_seconds": compile_seconds,
        "execution_devices": devices,
        "rss_before": before,
        "rss_after": _rss_snapshot(),
    }


def bind_remote_qkv(request: Any, tensors: Mapping[str, Any], remote_tensor_type: type) -> int:
    """Bind each block's six remote prefix outputs once before its chunk loop."""

    if set(tensors) != set(QKV_NAMES):
        raise ContractError("prefix remote QKV tensor set is incomplete")
    for name in QKV_NAMES:
        tensor = tensors[name]
        if not isinstance(tensor, remote_tensor_type):
            raise ContractError("prefix QKV output is not a RemoteTensor")
        if str(tensor.get_device_name()) != DEVICE:
            raise ContractError("prefix QKV RemoteTensor is not on GPU.1")
        request.set_tensor(name, tensor)
    return len(QKV_NAMES)


def gather_exact_chunk(
    *,
    video_windows: Any,
    window_idx: Any,
    window_mask: Any,
    reverse: Any,
    start: int,
    max_video_len: int,
    video_accumulator: Any,
    gather_counts: Any,
    np: Any,
) -> int:
    """Apply the saved inverse map for this chunk and reject duplicates/gaps."""

    video_mask = window_mask[:, :max_video_len]
    source_indices = window_idx[:, :max_video_len][video_mask]
    source_indices, positions = inverse_gather_positions(
        source_indices,
        reverse,
        start=start,
        max_video_len=max_video_len,
        chunk_windows=video_windows.shape[0],
        source_token_count=video_accumulator.shape[0],
        np=np,
    )
    if np.any(gather_counts[source_indices] != 0):
        raise ContractError("inverse gather repeated a source token")
    flat_video = video_windows.reshape(-1, video_windows.shape[-1])
    video_accumulator[source_indices] = flat_video[positions]
    gather_counts[source_indices] = 1
    return int(source_indices.size)


def inverse_gather_positions(
    source_indices: Any,
    reverse: Any,
    *,
    start: int,
    max_video_len: int,
    chunk_windows: int,
    source_token_count: int,
    np: Any | None = None,
) -> tuple[Any, Any]:
    """Validate a chunk's source-token set and map it into chunk-local rows."""

    if np is None:
        indices = [int(value) for value in source_indices]
        if not indices or len(set(indices)) != len(indices):
            raise ContractError("chunk contains duplicate or missing source tokens")
        if any(value < 0 or value >= source_token_count for value in indices):
            raise ContractError("chunk source index is out of bounds")
        positions = [int(reverse[value]) - start * max_video_len for value in indices]
        if any(value < 0 or value >= chunk_windows * max_video_len for value in positions):
            raise ContractError("inverse gather selected outside the active chunk")
        return indices, positions

    indices = np.asarray(source_indices, dtype=np.int64)
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ContractError("chunk contains duplicate or missing source tokens")
    if np.any(indices < 0) or np.any(indices >= source_token_count):
        raise ContractError("chunk source index is out of bounds")
    positions = reverse[indices] - start * max_video_len
    if np.any(positions < 0) or np.any(positions >= chunk_windows * max_video_len):
        raise ContractError("inverse gather selected outside the active chunk")
    return indices, positions


def _host_tensor(ov: Any, np: Any, array: Any) -> Any:
    return ov.Tensor(np.ascontiguousarray(array), shared_memory=True)


def _check_host_output(ov: Any, np: Any, tensor: Any, shape: Sequence[int]) -> Any:
    if isinstance(tensor, ov.RemoteTensor):
        raise ContractError("chunk or suffix output did not return to host")
    value = np.asarray(tensor.data)
    if tuple(value.shape) != tuple(shape) or not bool(np.isfinite(value).all()):
        raise ContractError("host output shape or finite-value check failed")
    return value


def _compile_shapes_for_chunk(max_video_len: int) -> dict[str, tuple[int, ...]]:
    return {
        **{name: (-1, base.HEADS, base.HEAD_DIM) for name in VIDEO_QKV_NAMES},
        **{name: (base.TXT_TOKENS, base.HEADS, base.HEAD_DIM) for name in TEXT_QKV_NAMES},
        "window_idx": (-1, max_video_len),
        "window_key_mask": (-1, max_video_len + base.TXT_TOKENS),
        "window_rope_freqs": (-1, max_video_len + base.TXT_TOKENS, base.ROPE_DIM),
    }


def _run_block(
    block_index: int,
    *,
    ov: Any,
    np: Any,
    torch: Any,
    core: Any,
    context: Any,
    decomposition: Any,
    chunk_request: Any,
    qkv_remote: Mapping[str, Any],
    window_info: Mapping[str, Any],
) -> dict[str, Any]:
    helper_path = HERE / f"dynamic_window_reverse_gather_block{block_index}_export.py"
    helper = base._load_module(helper_path, f"remote_tensor_window_helper_{block_index}")
    if int(getattr(helper, "BLOCK_INDEX", 0)) != block_index:
        raise ContractError("window helper block identity is invalid")
    if block_index == 16 and helper.WINDOW_METHOD != UNSHIFTED_WINDOW_METHOD:
        raise ContractError("even block 16 helper is not unshifted")
    contract, summary = base._window_metadata(helper, decomposition, torch)
    arrays = base._contract_arrays(contract, np)
    accounting = arrays["accounting"]
    if (
        int(summary["window_count"]) != window_info["window_count"]
        or int(summary["max_video_len"]) != window_info["max_video_len"]
    ):
        raise ContractError("runtime CPU metadata differs from current receipt")
    source_tokens = int(accounting["source_tokens"])
    values = base._random_inputs(source_tokens, 2026091800 + block_index, np)
    record: dict[str, Any] = {
        "block_index": block_index,
        "shape": list(PRODUCTION_SHAPE),
        "window_count": window_info["window_count"],
        "max_video_len": window_info["max_video_len"],
        "chunk_count": len(chunk_ranges(window_info["window_count"])),
        "prefix": {},
        "chunks": [],
        "gather": {},
        "suffix": {},
    }

    prefix_shapes = {
        "vid": (source_tokens, base.VID_DIM),
        "txt": (base.TXT_TOKENS, base.VID_DIM),
        "emb": (1, base.EMB_DIM),
    }
    prefix_model, prefix_compiled, prefix_compile = _compile_model(
        ov, core, context, HERE / base.ARTIFACT_NAMES["prefix"][0].format(block=f"block{block_index}"), prefix_shapes
    )
    prefix_request = prefix_compiled.create_infer_request()
    for name, value in values.items():
        prefix_request.set_tensor(name, _host_tensor(ov, np, value))
    if set(qkv_remote) != set(QKV_NAMES):
        raise ContractError("shared prefix QKV buffer is incomplete")
    for name, tensor in qkv_remote.items():
        prefix_request.set_tensor(name, tensor)
    rss_before = _rss_snapshot()
    started = time.perf_counter()
    # Python's infer() wrapper tries to expose every output as a host array;
    # wait explicitly so the six device-resident outputs remain remote.
    prefix_request.start_async()
    prefix_request.wait()
    prefix_seconds = time.perf_counter() - started
    for name, tensor in qkv_remote.items():
        if (
            not isinstance(tensor, ov.RemoteTensor)
            or tuple(int(value) for value in tensor.get_shape()) != PREFIX_OUTPUTS[name]
            or str(tensor.get_device_name()) != DEVICE
        ):
            raise ContractError("prefix output is not a GPU.1 RemoteTensor")
    record["prefix"] = {
        "compile": prefix_compile,
        "infer_seconds": prefix_seconds,
        "qkv_remote_tensor_count": len(qkv_remote),
        "qkv_host_copies": 0,
        "rss_before": rss_before,
        "rss_after": _rss_snapshot(),
    }
    del prefix_request, prefix_compiled, prefix_model
    gc.collect()

    bind_started = time.perf_counter()
    bound_count = bind_remote_qkv(chunk_request, qkv_remote, ov.RemoteTensor)
    remote_bind_seconds = time.perf_counter() - bind_started
    if bound_count != len(QKV_NAMES):
        raise ContractError("not all prefix QKV tensors were bound to the shared chunk request")

    vid_acc = np.zeros((source_tokens, base.INNER_DIM), dtype=np.float32)
    txt_sum_acc = np.zeros((base.TXT_TOKENS, base.INNER_DIM), dtype=np.float32)
    gather_counts = np.zeros(source_tokens, dtype=np.uint8)
    text_count = 0.0
    chunk_events = []
    gather_seconds_total = 0.0
    for start, end in chunk_ranges(int(accounting["window_count"])):
        chunk_windows = end - start
        idx = arrays["index"][start:end]
        mask = arrays["mask"][start:end]
        freqs = arrays["freqs"][start:end]
        chunk_inputs = {
            "window_idx": idx,
            "window_key_mask": mask,
            "window_rope_freqs": freqs,
        }
        expected_inputs = {
            "window_idx": (chunk_windows, int(accounting["max_video_len"])),
            "window_key_mask": (chunk_windows, int(accounting["max_attention_len"])),
            "window_rope_freqs": (chunk_windows, int(accounting["max_attention_len"]), base.ROPE_DIM),
        }
        if any(tuple(chunk_inputs[name].shape) != shape for name, shape in expected_inputs.items()):
            raise ContractError("chunk metadata shape differs from the compiled IR")
        chunk_rss_before = _rss_snapshot()
        bind_started = time.perf_counter()
        for name, value in chunk_inputs.items():
            chunk_request.set_tensor(name, _host_tensor(ov, np, value))
        output_shapes = {
            "vid_window_out": (chunk_windows, int(accounting["max_video_len"]), base.INNER_DIM),
            "txt_sum": (base.TXT_TOKENS, base.INNER_DIM),
            "txt_count": (1,),
        }
        host_outputs = {
            name: ov.Tensor(ov.Type.f32, list(shape))
            for name, shape in output_shapes.items()
        }
        for name, tensor in host_outputs.items():
            chunk_request.set_tensor(name, tensor)
        input_bind_seconds = time.perf_counter() - bind_started
        infer_started = time.perf_counter()
        chunk_request.infer()
        infer_seconds = time.perf_counter() - infer_started
        output_started = time.perf_counter()
        output_values = {
            name: _check_host_output(ov, np, chunk_request.get_tensor(name), shape)
            for name, shape in output_shapes.items()
        }
        output_view_seconds = time.perf_counter() - output_started
        text_count_part = float(output_values["txt_count"].reshape(-1)[0])
        if not math.isfinite(text_count_part) or abs(text_count_part - chunk_windows) > 1.0e-6:
            raise ContractError("chunk text count differs from its window count")
        gather_rss_before = _rss_snapshot()
        gather_started = time.perf_counter()
        gathered = gather_exact_chunk(
            video_windows=output_values["vid_window_out"],
            window_idx=idx,
            window_mask=mask.astype(bool, copy=False),
            reverse=arrays["reverse"],
            start=start,
            max_video_len=int(accounting["max_video_len"]),
            video_accumulator=vid_acc,
            gather_counts=gather_counts,
            np=np,
        )
        txt_sum_acc += output_values["txt_sum"]
        text_count += text_count_part
        gather_seconds = time.perf_counter() - gather_started
        gather_seconds_total += gather_seconds
        chunk_events.append(
            {
                "start": start,
                "end": end,
                "windows": chunk_windows,
                "input_bind_seconds": input_bind_seconds,
                "infer_seconds": infer_seconds,
                "host_output_view_seconds": output_view_seconds,
                "gather_text_sum_seconds": gather_seconds,
                "gathered_source_tokens": gathered,
                "outputs_finite": True,
                "rss_before": chunk_rss_before,
                "rss_after": _rss_snapshot(),
                "gather_rss_before": gather_rss_before,
                "gather_rss_after": _rss_snapshot(),
            }
        )
        del output_values, host_outputs

    if len(chunk_events) != record["chunk_count"]:
        raise ContractError("shared chunk request call count differs from the block contract")
    if text_count != float(accounting["window_count"]):
        raise ContractError("text sum count does not equal the full window count")
    if not bool(np.all(gather_counts == 1)):
        raise ContractError("inverse gather did not cover every source token exactly once")
    if not bool(np.isfinite(vid_acc).all()) or not bool(np.isfinite(txt_sum_acc).all()):
        raise ContractError("gathered video or text sum is non-finite")
    gather_started = time.perf_counter()
    attn_txt = txt_sum_acc / np.float32(text_count)
    if not bool(np.isfinite(attn_txt).all()):
        raise ContractError("text mean is non-finite")
    text_mean_seconds = time.perf_counter() - gather_started
    record["chunks"] = chunk_events
    record["remote_qkv"] = {
        "bound_to_shared_request_once_for_block": bound_count,
        "set_again_inside_chunk_loop": False,
        "binding_seconds": remote_bind_seconds,
        "same_remote_objects_reused_for_calls": len(chunk_events),
    }
    record["gather"] = {
        "seconds": gather_seconds_total + text_mean_seconds,
        "text_mean_seconds": text_mean_seconds,
        "text_count": text_count,
        "text_count_equals_window_count": True,
        "source_tokens": source_tokens,
        "unique_source_tokens": int(np.count_nonzero(gather_counts)),
        "full_unique_coverage": True,
        "rss_after": _rss_snapshot(),
    }

    suffix_shapes = {
        "vid": (source_tokens, base.VID_DIM),
        "txt": (base.TXT_TOKENS, base.VID_DIM),
        "emb": (1, base.EMB_DIM),
        "attn_vid": (source_tokens, base.INNER_DIM),
        "attn_txt": (base.TXT_TOKENS, base.INNER_DIM),
    }
    suffix_path = HERE / base.ARTIFACT_NAMES["suffix"][0].format(block=f"block{block_index}")
    suffix_model, suffix_compiled, suffix_compile = _compile_model(
        ov, core, context, suffix_path, suffix_shapes
    )
    suffix_request = suffix_compiled.create_infer_request()
    suffix_inputs = {
        "vid": values["vid"],
        "txt": values["txt"],
        "emb": values["emb"],
        "attn_vid": vid_acc,
        "attn_txt": attn_txt,
    }
    for name, value in suffix_inputs.items():
        suffix_request.set_tensor(name, _host_tensor(ov, np, value))
    suffix_shapes_out = {
        "vid_out": (source_tokens, base.VID_DIM),
        "txt_out": (base.TXT_TOKENS, base.VID_DIM),
    }
    suffix_outputs = {
        name: ov.Tensor(ov.Type.f32, list(shape))
        for name, shape in suffix_shapes_out.items()
    }
    for name, tensor in suffix_outputs.items():
        suffix_request.set_tensor(name, tensor)
    suffix_rss_before = _rss_snapshot()
    started = time.perf_counter()
    suffix_request.infer()
    suffix_seconds = time.perf_counter() - started
    outputs = {
        name: _check_host_output(ov, np, suffix_request.get_tensor(name), shape)
        for name, shape in suffix_shapes_out.items()
    }
    deltas = {
        "vid": float(np.max(np.abs(outputs["vid_out"] - values["vid"]))),
        "txt": float(np.max(np.abs(outputs["txt_out"] - values["txt"]))),
    }
    if any(value <= DISTINCT_THRESHOLD for value in deltas.values()):
        raise ContractError("suffix output does not differ from its input")
    record["suffix"] = {
        "compile": suffix_compile,
        "infer_seconds": suffix_seconds,
        "outputs_finite": True,
        "output_input_max_abs_delta": deltas,
        "outputs_differ_from_inputs": True,
        "rss_before": suffix_rss_before,
        "rss_after": _rss_snapshot(),
    }
    del suffix_request, suffix_compiled, suffix_model
    gc.collect()
    return record


def _resolve_receipt(path: Path) -> Path:
    candidate = path if path.is_absolute() else HERE / path
    resolved = candidate.resolve(strict=False)
    if resolved.parent != HERE or resolved.suffix.lower() != ".json":
        raise RunnerInputError("receipt must be a JSON file directly inside the evidence directory")
    if candidate.exists() or candidate.is_symlink():
        raise RunnerInputError("refusing to overwrite an existing receipt")
    return resolved


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sanitized_error(exc: BaseException, stage: str) -> dict[str, str]:
    kind = type(exc).__name__
    code = {
        "RunnerInputError": "invalid_input",
        "ContractError": "contract_failure",
        "AssertionError": "assertion_failure",
        "FileNotFoundError": "artifact_unavailable",
        "PermissionError": "permission_denied",
        "RuntimeError": "runtime_failure",
    }.get(kind, "unexpected_failure")
    return {"stage": stage, "type": kind, "code": code}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        receipt_path = _resolve_receipt(args.receipt)
    except RunnerInputError as exc:
        print(json.dumps({"status": "blocked", "receipt_written": False}, sort_keys=True))
        return 2

    receipt: dict[str, Any] = {
        "runner": "b580_split_ir_remote_tensor_runner",
        "status": "not_started",
        "device_argument": DEVICE,
        "blocks": list(BLOCKS),
        "chunk_size": CHUNK_SIZE,
        "errors": [],
    }
    try:
        receipt["preflight"] = validate_preflight_marker(args.preflight_marker)
        audit = validate_current_artifacts()
        receipt["artifact_proof"] = audit["proof"]
        receipt["artifact_hashes"] = audit["artifact_hashes"]
        import numpy as np
        import openvino as ov
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        decomposition = base._load_module(
            HERE / "chunked_attention_decomposition_export.py",
            "chunked_attention_decomposition_for_remote_tensor_runner",
        )
        setup_rss = _rss_snapshot()
        setup_started = time.perf_counter()
        core = ov.Core()
        context = core.get_default_context(DEVICE)
        context_seconds = time.perf_counter() - setup_started
        if str(context.get_device_name()) != DEVICE:
            raise ContractError("default remote context is not exactly GPU.1")
        receipt["device_context"] = {
            "default_context": True,
            "device_name": DEVICE,
            "setup_seconds": context_seconds,
            "rss_before": setup_rss,
            "rss_after": _rss_snapshot(),
        }

        first_info = audit["window_contracts"][0]
        shared_chunk_shapes = _compile_shapes_for_chunk(first_info["max_video_len"])
        chunk_xml = HERE / base.ARTIFACT_NAMES["chunk"][0].format(block="block0")
        chunk_model, chunk_compiled, chunk_compile = _compile_model(
            ov, core, context, chunk_xml, shared_chunk_shapes
        )
        chunk_request = chunk_compiled.create_infer_request()
        receipt["shared_chunk_compile"] = chunk_compile
        qkv_remote = {
            name: context.create_tensor(
                ov.Type.f32,
                ov.Shape(shape),
                {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"},
            )
            for name, shape in PREFIX_OUTPUTS.items()
        }
        receipt["blocks_run"] = []
        for block_index in BLOCKS:
            receipt["blocks_run"].append(
                _run_block(
                    block_index,
                    ov=ov,
                    np=np,
                    torch=torch,
                    core=core,
                    context=context,
                    decomposition=decomposition,
                    chunk_request=chunk_request,
                    qkv_remote=qkv_remote,
                    window_info=audit["window_contracts"][block_index],
                )
            )
        del qkv_remote, chunk_request, chunk_compiled, chunk_model, context, core
        gc.collect()
        receipt["status"] = "pass"
        receipt["rss_final"] = _rss_snapshot()
    except RunnerInputError as exc:
        receipt["status"] = "blocked"
        receipt["errors"].append(_sanitized_error(exc, "preflight"))
    except Exception as exc:
        receipt["status"] = "failure"
        receipt["errors"].append(_sanitized_error(exc, "runner"))
        receipt["rss_final"] = _rss_snapshot()

    try:
        _write_receipt(receipt_path, receipt)
        written = True
    except OSError:
        written = False
    print(json.dumps({"status": receipt["status"], "receipt_written": written}, sort_keys=True))
    return 0 if receipt["status"] == "pass" and written else 2


if __name__ == "__main__":
    raise SystemExit(main())
