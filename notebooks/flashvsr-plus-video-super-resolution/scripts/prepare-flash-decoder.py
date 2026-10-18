#!/usr/bin/env python3
"""Prepare authenticated Flash TCDecoder weight-binding tables.

The checkpoint remains the only learned source.  Its ZIP payload is never
extracted: FakeTensorMode locates the raw F32 spans, while full-file and
per-span hashes authenticate them.  The output is only a C++ table; native
preparation later reads the pinned checkpoint and recreates the three BF16
BINs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any, Iterable
import zipfile

from flash_export import fresh_output


CHECKPOINT_BYTES = 189_018_333
CHECKPOINT_SHA256 = "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161"
SOURCE_SHA256 = "85369a0fcd41073a2acc1050cfeb37a85d123e84c12cbee6339bb4308dc04ac6"
SOURCE_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
PARAMETER_MANIFEST_SHA256 = "b7edfda7d33cca8bbb35979e4580f3bfd07d6e759d3e22f73c78291ab4042b8d"
DATA_MEMBER = "step-step=54000_lora/data/0"
SOURCE_PIN_NAME = "TCDecoder.ckpt"
DYNAMIC = {
    "grow_memory": ("grow_weight", "decoder.15.conv.weight", 256),
    "remote_frame_child": ("frame_grow", "decoder.21.conv.weight", 128),
}
PHASES = ("front", "grow_memory", "remote_frame_child")
PHASE_BIN_NAMES = {
    "front": "flash_decoder_front_bf16.bin",
    "grow_memory": "flash_decoder_grow_memory_bf16.bin",
    "remote_frame_child": "flash_decoder_remote_frame_child_bf16.bin",
}
PHASE_EXPECTED_COUNTS = {"front": 41, "grow_memory": 19, "remote_frame_child": 4}


def _load_post_helpers():
    path = Path(__file__).with_name("prepare-flash-post.py")
    spec = importlib.util.spec_from_file_location("flash_post_pin_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_post_helpers()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str, *, root: Path | None = None) -> Path:
    """Resolve one trusted file without accepting symlink or traversal input."""

    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must be a regular file, not a symlink: {path}")
    candidate = raw.resolve(strict=True)
    if root is not None:
        root = root.resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} escapes its trusted root: {path}") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} is not a regular file: {path}")
    return candidate


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} is not a real directory: {path}")
    return resolved


def _safe_child(root: Path, relative: str, label: str) -> Path:
    if type(relative) is not str or not relative:
        raise ValueError(f"{label} path is not a simple relative asset name: {relative!r}")
    if any(character in relative for character in ("/", "\\", ":", "\x00")):
        raise ValueError(f"{label} path is not a simple relative asset name: {relative!r}")
    candidate = Path(relative)
    if (candidate.is_absolute() or candidate.name != relative or
            any(part in ("", ".", "..") for part in candidate.parts)):
        raise ValueError(f"{label} path is not a simple relative asset name: {relative!r}")
    return _regular(root / candidate, label, root=root)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _pin(name: str, byte_count: int, digest: str) -> str:
    return HELPERS._cpp_pin(name, byte_count, digest)


def _span_cpp(span: dict[str, int]) -> str:
    return f'{{{span["offset"]}ULL, {span["bytes"]}ULL, {span["elements"]}ULL}}'


def _load_plus_decoder(reference_root: Path, decoder_source: Path, checkpoint: Path):
    runtime = _regular(reference_root / "openvino/full_runtime/plus_decoder.py",
                        "plus_decoder.py", root=reference_root)
    _regular(reference_root / "openvino/full_runtime/plus_storage.py",
             "plus_storage.py", root=reference_root)
    old_path = list(sys.path)
    sys.path[:0] = [str(runtime.parent), str(reference_root / "openvino")]
    try:
        spec = importlib.util.spec_from_file_location("flash_decoder_pin_plus_decoder", runtime)
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to load pinned plus_decoder.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.SOURCE = decoder_source
        module.CHECKPOINT = checkpoint
        decoder, state, metadata = module.load_decoder()
        return module, decoder, state, metadata
    finally:
        sys.path[:] = old_path


def _zip_data_payload(path: Path) -> tuple[int, int]:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 5 or any(item.compress_type != zipfile.ZIP_STORED for item in entries):
            raise ValueError("checkpoint must contain exactly five stored ZIP entries")
        item = next((entry for entry in entries if entry.filename == DATA_MEMBER), None)
        if item is None:
            raise ValueError(f"checkpoint is missing {DATA_MEMBER}")
        with path.open("rb") as stream:
            stream.seek(item.header_offset)
            header = stream.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("checkpoint data/0 local ZIP header is invalid")
        name_bytes, extra_bytes = struct.unpack_from("<HH", header, 26)
        payload = item.header_offset + 30 + name_bytes + extra_bytes
        if payload % 4:
            raise ValueError("checkpoint data/0 payload is not four-byte aligned")
        return payload, int(item.file_size)


def _hash_range(stream: Any, offset: int, size: int) -> str:
    stream.seek(offset)
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        block = stream.read(min(1 << 20, remaining))
        if not block:
            raise ValueError("checkpoint raw tensor span is truncated")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def _raw_offsets(checkpoint: Path, state: dict[str, Any], metadata: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    """Authenticate all 66 raw F32 ranges without extracting the ZIP payload."""

    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake = torch.load(checkpoint, map_location="meta", weights_only=True, mmap=True)
    if not isinstance(fake, dict) or set(fake) != set(state) or len(fake) != 66:
        raise ValueError("checkpoint fake tensor keys are not the authenticated 66-key set")
    payload, payload_bytes = _zip_data_payload(checkpoint)
    payload_end = payload + payload_bytes
    records: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int, str]] = []
    with checkpoint.open("rb") as stream:
        for key, value in state.items():
            tensor = fake[key]
            if str(value.dtype) != "torch.float32" or str(tensor.dtype) != "torch.float32":
                raise ValueError(f"checkpoint parameter is not F32: {key}")
            shape = tuple(int(dim) for dim in value.shape)
            if tuple(int(dim) for dim in tensor.shape) != shape or not tensor.is_contiguous():
                raise ValueError(f"checkpoint tensor metadata differs: {key}")
            storage = tensor.untyped_storage()
            checkpoint_offset = getattr(storage, "_checkpoint_offset", None)
            if type(checkpoint_offset) is not int:
                raise ValueError(f"checkpoint offset metadata is unavailable: {key}")
            storage_offset = int(tensor.storage_offset())
            size = int(tensor.numel()) * 4
            raw_start = checkpoint_offset + storage_offset * 4
            raw_end = raw_start + size
            if checkpoint_offset % 4 or raw_start % 4 or raw_start < payload or raw_end > payload_end:
                raise ValueError(f"checkpoint raw span alignment/range differs: {key}")
            actual = _hash_range(stream, raw_start, size)
            expected = metadata["parameter_hashes"].get(key)
            if actual != expected:
                raise ValueError(f"checkpoint raw span hash differs: {key}")
            records[key] = {
                "shape": list(shape),
                "elements": int(tensor.numel()),
                "source_offset": raw_start,
                "source_bytes": size,
                "raw_sha256": actual,
                "checkpoint_offset": checkpoint_offset,
                "storage_offset_elements": storage_offset,
            }
            intervals.append((raw_start, raw_end, key))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"checkpoint raw tensor spans overlap: {previous[2]} / {current[2]}")
    return records, torch.__version__


def _expected_graph_shape(source_shape: tuple[int, ...]) -> list[int]:
    return [1, source_shape[0], 1, 1] if len(source_shape) == 1 else list(source_shape)


def _exact_shape(value: Any, expected: list[int]) -> bool:
    return (type(value) is list and len(value) == len(expected) and
            all(type(item) is int and item == wanted for item, wanted in zip(value, expected)))


def _validate_dynamic_contract(phase: str, graph: dict[str, Any]) -> None:
    contract = graph.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{phase} graph contract is malformed")
    expected_name = DYNAMIC.get(phase, (None, "", 0))[0]
    expected_names = [] if expected_name is None else [expected_name]
    if contract.get("dynamic_inputs") != expected_names:
        raise ValueError(f"{phase} dynamic input contract differs")
    inputs = contract.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError(f"{phase} graph input records are missing")
    dynamic_names = {name for name, _, _ in DYNAMIC.values()}
    if any(isinstance(item, dict) and item.get("name") in dynamic_names and
           item.get("name") != expected_name for item in inputs):
        raise ValueError(f"{phase} exposes an unexpected dynamic input")
    matching = [item for item in inputs
                if isinstance(item, dict) and item.get("name") == expected_name]
    if expected_name is None:
        return
    if len(matching) != 1:
        raise ValueError(f"{phase} dynamic input port is missing or duplicated")
    expected_shape = [DYNAMIC[phase][2], DYNAMIC[phase][2], 1, 1]
    port = matching[0]
    if port.get("element_type") != "bf16" or not _exact_shape(port.get("shape"), expected_shape):
        raise ValueError(f"{phase} dynamic input port ABI differs")


def _validate_bundle(manifest: dict[str, Any], manifest_path: Path,
                     state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if manifest.get("schema") != 1 or manifest.get("status") != "pass" or manifest.get("bundle") != "phased":
        raise ValueError("bundle manifest is not the trusted reviewed phased schema")
    if manifest.get("precision") != "bf16" or manifest.get("geometry", {}).get("source_hw") != [2176, 3840]:
        raise ValueError("bundle geometry/precision differs from the reviewed decoder bundle")
    geometry = manifest.get("geometry", {})
    if geometry.get("tile_count") != 16 or geometry.get("halo_rows") != 12:
        raise ValueError("bundle tile geometry differs from the reviewed 16-tile/12-halo plan")
    trust = manifest.get("trust", {})
    storage = manifest.get("learned_storage", {})
    if not isinstance(trust, dict) or not isinstance(storage, dict):
        raise ValueError("bundle trust/storage records are malformed")
    if (trust.get("learned_payloads_loaded") is not False or
            storage.get("materialized_learned_payloads") is not False):
        raise ValueError("structural bundle must not contain materialized learned payloads")
    decoder_source = manifest.get("decoder_source", {})
    if not isinstance(decoder_source, dict):
        raise ValueError("bundle decoder source record is malformed")
    if decoder_source.get("sha256") != SOURCE_SHA256 or decoder_source.get("expected_sha256") != SOURCE_SHA256:
        raise ValueError("bundle decoder source fingerprint differs")
    if manifest.get("graphs", {}).keys() != set(PHASES):
        raise ValueError("bundle phase set differs")
    static = storage.get("static_bindings")
    if not isinstance(static, list) or len(static) != 64:
        raise ValueError("bundle must declare exactly 64 static bindings")
    source_names = set(state)
    dynamic_names = {item[1] for item in DYNAMIC.values()}
    expected_static = source_names - dynamic_names
    static_by_name: dict[str, dict[str, Any]] = {}
    for record in static:
        if not isinstance(record, dict):
            raise ValueError("static binding is malformed")
        name = record.get("source_parameter_name")
        if not isinstance(name, str) or name not in expected_static or name in static_by_name:
            raise ValueError(f"static source identity is invalid: {name!r}")
        source_shape = tuple(int(x) for x in state[name].shape)
        if (record.get("source_shape") != list(source_shape) or record.get("source_dtype") != "f32" or
                record.get("storage") != "bf16" or record.get("graph_input_name") != "weight__" + name.replace(".", "_") or
                record.get("graph_shape") != _expected_graph_shape(source_shape)):
            raise ValueError(f"static source ABI differs: {name}")
        static_by_name[name] = record
    if set(static_by_name) != expected_static:
        raise ValueError("static binding coverage is not exactly 64 authenticated parameters")
    graph_to_source = {
        record["graph_input_name"]: source_name
        for source_name, record in static_by_name.items()
    }
    phase_names: dict[str, set[str]] = {}
    for phase in PHASES:
        graph = manifest["graphs"][phase]
        _validate_dynamic_contract(phase, graph)
        contract = graph["contract"]
        graph_names = contract.get("static_parameter_names")
        if not isinstance(graph_names, list) or len(graph_names) != PHASE_EXPECTED_COUNTS[phase]:
            raise ValueError(f"{phase} static phase ownership count differs")
        if any(name not in graph_to_source for name in graph_names):
            raise ValueError(f"{phase} static phase ownership contains an unknown graph input")
        phase_names[phase] = {graph_to_source[name] for name in graph_names}
        if not phase_names[phase] <= expected_static or len(phase_names[phase]) != len(graph_names):
            raise ValueError(f"{phase} static phase ownership is malformed")
        metadata = manifest["graphs"][phase].get("metadata", {})
        if metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise ValueError(f"{phase} checkpoint provenance differs")
    if set().union(*phase_names.values()) != expected_static or sum(map(len, phase_names.values())) != 64:
        raise ValueError("static phase ownership is not disjoint/exhaustive")
    banks = storage.get("source_bank_descriptors")
    if not isinstance(banks, list) or len(banks) != 2 or storage.get("source_bank_slice_count") != 4:
        raise ValueError("bundle dynamic bank descriptor count differs")
    dynamic_records: dict[str, dict[str, Any]] = {}
    for record in banks:
        if not isinstance(record, dict):
            raise ValueError("dynamic bank descriptor is malformed")
        source_name = record.get("source_parameter_name")
        if source_name in dynamic_records:
            raise ValueError("duplicate dynamic source bank")
        found = next((value for value in DYNAMIC.values() if value[1] == source_name), None)
        if found is None or record.get("storage") != "bf16" or record.get("source_dtype") != "f32":
            raise ValueError(f"dynamic source identity is invalid: {source_name!r}")
        shape = tuple(int(x) for x in state[source_name].shape)
        if record.get("source_shape") != list(shape):
            raise ValueError(f"dynamic source shape differs: {source_name}")
        width = found[2]
        slices = record.get("slices")
        if not isinstance(slices, list) or len(slices) != 2:
            raise ValueError(f"dynamic slice count differs: {source_name}")
        for child, item in enumerate(slices):
            expected_shape = [width, *shape[1:]]
            if (item.get("child") != child or item.get("axis") != 0 or
                    item.get("start") != child * width or item.get("stop") != (child + 1) * width or
                    item.get("shape") != expected_shape or item.get("source_dtype") != "f32" or
                    item.get("storage") != "bf16"):
                raise ValueError(f"dynamic slice ABI differs: {source_name}")
        dynamic_records[source_name] = record
    if set(dynamic_records) != dynamic_names:
        raise ValueError("dynamic source bank coverage differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("bundle artifact records are malformed")
    artifact_records: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        if not isinstance(artifacts.get(phase), dict):
            raise ValueError(f"missing structural artifacts for {phase}")
        phase_artifacts = {}
        for kind in ("xml", "bin"):
            record = artifacts[phase].get(kind)
            if not isinstance(record, dict):
                raise ValueError(f"missing {phase} {kind} artifact")
            relative = record.get("path")
            if type(relative) is not str:
                raise ValueError(f"{phase} {kind} artifact path is not a string")
            bytes_value = record.get("bytes")
            lower_bound = 1 if kind == "xml" else 0
            if (type(bytes_value) is not int or bytes_value < lower_bound or
                    bytes_value > 0xFFFFFFFFFFFFFFFF):
                raise ValueError(f"{phase} {kind} artifact byte count is invalid")
            digest = record.get("sha256")
            try:
                HELPERS._bytes_literal(digest)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{phase} {kind} artifact SHA-256 is invalid") from error
            path = _safe_child(manifest_path.parent, relative, f"{phase} {kind}")
            if path.stat().st_size != bytes_value or _sha256(path) != digest:
                raise ValueError(f"{phase} {kind} artifact fingerprint differs")
            phase_artifacts[kind] = {**record, "path": path}
        artifact_records[phase] = phase_artifacts
    return {"static": static_by_name, "phases": phase_names, "dynamic": dynamic_records}, artifact_records


def _item(raw: dict[str, Any], key: str, *, start: int | None = None,
          elements: int | None = None) -> dict[str, Any]:
    source_offset = raw["source_offset"] if start is None else start
    count = raw["elements"] if elements is None else elements
    return {"key": key, "source_key": key, "source_offset": source_offset, "bytes": count * 4}


def _binding_from_segment(record: dict[str, Any], segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_key": record["source_parameter_name"],
        "port_name": record["graph_input_name"],
        "span": {"offset": int(segment["offset"]), "bytes": int(segment["bytes"]),
                 "elements": int(segment["bytes"]) // 2},
        "dimensions": list(record["graph_shape"]),
        "rank": len(record["graph_shape"]),
        "source_shape": list(record["source_shape"]),
    }


def _render_header(source_pin: str, source_bytes: int, source_sha: str,
                   manifest_sha: str, artifact_records: dict[str, dict[str, Any]],
                   phase_records: dict[str, dict[str, Any]]) -> str:
    lines = [
        "// Generated by scripts/prepare-flash-decoder.py; do not edit.",
        "// Learned payloads are not embedded; plans read the authenticated checkpoint during native preparation.",
        "#pragma once", "#include <array>", "#include <cstddef>", "#include <cstdint>",
        '#include "seedvr_prepare_detail.hpp"', "", "namespace otc::ai {",
        "struct FlashDecoderWeightBinding {",
        "  const char* canonical_key;", "  const char* port_name;",
        "  SeedVrWeightSpan span;", "  std::array<std::size_t, 4> dimensions;",
        "  std::size_t rank;", "};", "",
        f"inline const SeedVrAssetPin kFlashDecoderSource = {_pin(source_pin, source_bytes, source_sha)};",
        "inline constexpr std::uint64_t kFlashDecoderHeaderBytes = 0ULL;",
        "inline constexpr std::uint32_t kFlashDecoderTableVersion = 1U;",
        f"inline constexpr const char* kFlashDecoderSourceCommit = {json.dumps(SOURCE_COMMIT)};",
        f"inline constexpr const char* kFlashDecoderCheckpointManifest = {json.dumps(manifest_sha)};",
    ]
    for phase in PHASES:
        cap = "Front" if phase == "front" else ("Grow" if phase == "grow_memory" else "Frame")
        artifact = artifact_records[phase]
        lines.extend([
            f"inline const SeedVrAssetPin kFlashDecoder{cap}Xml = "
            + _pin(artifact["xml"]["path"].name, artifact["xml"]["bytes"], artifact["xml"]["sha256"]) + ";",
            f"inline const SeedVrAssetPin kFlashDecoder{cap}StructuralBin = "
            + _pin(artifact["bin"]["path"].name, artifact["bin"]["bytes"], artifact["bin"]["sha256"]) + ";",
        ])
    for phase in PHASES:
        cap = "Front" if phase == "front" else ("Grow" if phase == "grow_memory" else "Frame")
        bindings = phase_records[phase]["bindings"]
        lines.extend([
            f"inline const std::array<FlashDecoderWeightBinding, {len(bindings)}> kFlashDecoder{cap}Weights{{{{",
        ])
        for record in bindings:
            dimensions = record["dimensions"] + [1] * (4 - len(record["dimensions"]))
            lines.append(
                f'  {{{json.dumps(record["canonical_key"])}, {json.dumps(record["port_name"])}, '
                f'{_span_cpp(record["span"])}, '
                f'{{{", ".join(str(int(value)) + "ULL" for value in dimensions)}}}, '
                f'{record["rank"]}}},'
            )
        lines.extend(["}};", ""])
    lines.extend([
        "inline const SeedVrWeightSpan kFlashDecoderGrowChildSpans[2] = {",
        *[f"  {_span_cpp(item)}," for item in phase_records["grow_memory"]["child_spans"]],
        "};",
        "inline const SeedVrWeightSpan kFlashDecoderFrameChildSpans[2] = {",
        *[f"  {_span_cpp(item)}," for item in phase_records["remote_frame_child"]["child_spans"]],
        "};", "",
    ])
    for phase in PHASES:
        cap = "Front" if phase == "front" else ("Grow" if phase == "grow_memory" else "Frame")
        record = phase_records[phase]
        lines.append(f"inline const prepare_detail::Segment kFlashDecoder{cap}Segments[] = {{")
        lines.extend(f"  {HELPERS._segment_cpp(segment)}," for segment in record["segments"])
        lines.extend(["};", ""])
        lines.append(
            f"inline const SeedVrAssetPin kFlashDecoder{cap}Bin = "
            + _pin(record["bin"]["name"], record["bin"]["bytes"], record["bin"]["sha256"]) + ";"
        )
    lines.extend([
        "inline const prepare_detail::Plan kFlashDecoderPlans[3] = {",
        "  {kFlashDecoderFrontBin, kFlashDecoderFrontStructuralBin, kFlashDecoderFrontSegments, "
        f"{len(phase_records['front']['segments'])}U}},",
        "  {kFlashDecoderGrowBin, kFlashDecoderGrowStructuralBin, kFlashDecoderGrowSegments, "
        f"{len(phase_records['grow_memory']['segments'])}U}},",
        "  {kFlashDecoderFrameBin, kFlashDecoderFrameStructuralBin, kFlashDecoderFrameSegments, "
        f"{len(phase_records['remote_frame_child']['segments'])}U}},",
        "};", "", "}  // namespace otc::ai", "",
    ])
    return "\n".join(lines)


def build(*, reference_root: Path, checkpoint: Path, decoder_source: Path,
          bundle_manifest: Path, bf16_converter: Path, output: Path) -> dict[str, Any]:
    reference_root = _directory(reference_root, "reference root")
    checkpoint = _regular(checkpoint, "checkpoint")
    decoder_source = _regular(decoder_source, "decoder source")
    bf16_converter = _regular(bf16_converter, "BF16 converter")
    if not os.access(bf16_converter, os.X_OK):
        raise ValueError("BF16 converter is not executable")
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("checkpoint size differs from the authenticated pin")
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint fingerprint differs from the authenticated pin")
    if _sha256(decoder_source) != SOURCE_SHA256:
        raise ValueError("decoder source fingerprint differs from the authenticated pin")
    manifest = _load_json(bundle_manifest, "bundle manifest")
    manifest_sha = _sha256(_regular(bundle_manifest, "bundle manifest"))
    module, _decoder, state, metadata = _load_plus_decoder(reference_root, decoder_source, checkpoint)
    if (metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
            metadata.get("parameter_manifest_sha256") != PARAMETER_MANIFEST_SHA256 or
            metadata.get("source_sha256") != SOURCE_SHA256):
        raise ValueError("authenticated decoder metadata differs from the pinned receipt")
    raw, torch_version = _raw_offsets(checkpoint, state, metadata)
    validated, artifacts = _validate_bundle(manifest, bundle_manifest, state)
    destination = fresh_output(output)
    converter = Path(bf16_converter)
    phase_outputs: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        static_names = sorted(validated["phases"][phase])
        static_items = [_item(raw[name], name) for name in static_names]
        dynamic_items: list[dict[str, Any]] = []
        child_source = None
        if phase in DYNAMIC:
            input_name, child_source, width = DYNAMIC[phase]
            source = raw[child_source]
            row_elements = source["elements"] // source["shape"][0]
            for child in range(2):
                start = source["source_offset"] + child * width * row_elements * 4
                dynamic_items.append(_item(source, child_source, start=start, elements=width * row_elements))
        structural = artifacts[phase]["bin"]["path"].read_bytes()
        all_items = static_items + dynamic_items
        segments, output_bytes, output_sha = HELPERS._plan_from_items(
            checkpoint, all_items, structural, converter, learned_source=0
        )
        bindings = []
        for index, name in enumerate(static_names, 1):
            bindings.append(_binding_from_segment(validated["static"][name], segments[index]))
        child_spans = []
        for index, item in enumerate(dynamic_items, len(static_items) + 1):
            segment = segments[index]
            child_spans.append({"offset": int(segment["offset"]), "bytes": int(segment["bytes"]),
                                "elements": int(segment["bytes"]) // 2})
        phase_outputs[phase] = {
            "bindings": bindings,
            "child_spans": child_spans,
            "segments": segments,
            "bin": {"name": PHASE_BIN_NAMES[phase], "bytes": output_bytes, "sha256": output_sha},
            "static_count": len(static_names),
            "dynamic_count": len(dynamic_items),
            "dynamic_source_parameter": child_source,
        }
    header = _render_header(
        SOURCE_PIN_NAME, checkpoint.stat().st_size, CHECKPOINT_SHA256, manifest_sha,
        artifacts, phase_outputs,
    )
    header_path = destination / "flash_decoder_assets.hpp"
    header_path.write_text(header, encoding="utf-8")
    receipt = {
        "status": "pass",
        "generator": "scripts/prepare-flash-decoder.py",
        "checkpoint": {"name": checkpoint.name, "bytes": checkpoint.stat().st_size,
                        "sha256": CHECKPOINT_SHA256, "parameter_count": len(state),
                        "parameter_manifest_sha256": metadata["parameter_manifest_sha256"],
                        "source_sha256": metadata["source_sha256"]},
        "decoder_source": {"path": decoder_source.name, "sha256": _sha256(decoder_source),
                           "source_commit": SOURCE_COMMIT},
        "bundle_manifest": {"path": Path(bundle_manifest).name, "sha256": manifest_sha},
        "raw_span_audit": {"parameter_count": len(raw), "raw_f32_bytes": sum(item["source_bytes"] for item in raw.values()),
                            "torch_version": torch_version, "checkpoint_offset": sorted({item["checkpoint_offset"] for item in raw.values()}),
                            "all_contiguous": True, "all_raw_hashes_match": True,
                            "full_checkpoint_sha256_verified": True},
        "phases": {
            phase: {**{key: value for key, value in record.items() if key != "segments"},
                    "segment_count": len(record["segments"]),
                    "source_indices": sorted({int(segment["source"]) for segment in record["segments"]})}
            for phase, record in phase_outputs.items()
        },
        "prepared_payloads_written": False,
        "header": {"path": header_path.name, "bytes": header_path.stat().st_size,
                   "sha256": _sha256(header_path), "source_only": True},
        "bf16_converter": {"name": converter.name, "sha256": _sha256(converter)},
    }
    (destination / "decoder-preparation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-source", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--bf16-converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        receipt = build(
            reference_root=args.reference_root, checkpoint=args.checkpoint,
            decoder_source=args.decoder_source, bundle_manifest=args.bundle_manifest,
            bf16_converter=args.bf16_converter, output=args.output,
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(json.dumps({"status": receipt["status"], "output": str(args.output),
                      "header_sha256": receipt["header"]["sha256"],
                      "prepared_payloads_written": receipt["prepared_payloads_written"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
