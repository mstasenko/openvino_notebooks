#!/usr/bin/env python3
"""Build-time pin generator for fixed Flash PRE/POST/CROSS/LQ native assets.

This tool authenticates the pinned Turbo or Base safetensors files, reads only
their bounded safetensors headers, and asks the reviewed native BF16 converter
to hash each fixed stage span.  It emits C++ tables consumed by native code;
there is no runtime safetensors parser, JSON override, OpenVINO import, or GPU
operation here.  The POST graph is loaded before the native stage recreates
its two owning Assign barriers; the PRE graph remains ordinary Result output
and the native stage recreates its six state sinks.  Learned per-layer BINs
are prepared later by the native fixed Plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
from typing import Any

from flash_export import fresh_output

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "flash-assets"
LAYERS = 30
ROWS = 4096
DIM = 1536
MAX_HEADER_BYTES = 1 << 20
CHUNK_BYTES = 1 << 20
UINT64_MAX = (1 << 64) - 1
ZERO_SHA = "00" * 32
EXTERNAL_WEIGHTS_LABEL = "external_authenticated_safetensors"

MODE_PINS = {
    "turbo": {
        "dit": (5_676_070_392, "7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b"),
        "lq": (1_151_384_272, "e77b378ae12342eefbb58f924256f362a151b93ce73dd32808fcdaca9c051496"),
    },
    "base": {
        "dit": (5_676_070_392, "58d64c1f1b3a559bafb69bb014f3ab2745e800c89435afdc154966eb96bc7459"),
        "lq": (1_151_384_272, "a1fe997f19060171db8dd4d58c30debc78ddfcc357b63481798792a38a356d8d"),
    },
}
SOURCE_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
MODEL_REPO_REVISION = "46b79520946ac67dec94d1566490690953c7debc"

# The reusable POST graph has 13 learned ports.  The other nine Parameters
# (x, two contexts, cross K/V, and four modulation tensors) are call inputs.
POST_SCHEMA = (
    ("blocks.0.self_attn.o.weight", (1536, 1536)),
    ("blocks.0.self_attn.o.bias", (1536,)),
    ("blocks.0.norm3.weight", (1536,)),
    ("blocks.0.norm3.bias", (1536,)),
    ("blocks.0.cross_attn.q.weight", (1536, 1536)),
    ("blocks.0.cross_attn.q.bias", (1536,)),
    ("blocks.0.cross_attn.norm_q.weight", (1536,)),
    ("blocks.0.cross_attn.o.weight", (1536, 1536)),
    ("blocks.0.cross_attn.o.bias", (1536,)),
    ("blocks.0.ffn.0.weight", (8960, 1536)),
    ("blocks.0.ffn.0.bias", (8960,)),
    ("blocks.0.ffn.2.weight", (1536, 8960)),
    ("blocks.0.ffn.2.bias", (1536,)),
)
PRE_SCHEMA = (
    ("blocks.0.modulation", (1, 6, 1536)),
    ("blocks.0.self_attn.q.weight", (1536, 1536)),
    ("blocks.0.self_attn.q.bias", (1536,)),
    ("blocks.0.self_attn.norm_q.weight", (1536,)),
    ("blocks.0.self_attn.k.weight", (1536, 1536)),
    ("blocks.0.self_attn.k.bias", (1536,)),
    ("blocks.0.self_attn.norm_k.weight", (1536,)),
    ("blocks.0.self_attn.v.weight", (1536, 1536)),
    ("blocks.0.self_attn.v.bias", (1536,)),
)
CROSS_SCHEMA = (
    ("blocks.0.cross_attn.k.weight", (1536, 1536)),
    ("blocks.0.cross_attn.k.bias", (1536,)),
    ("blocks.0.cross_attn.norm_k.weight", (1536,)),
    ("blocks.0.cross_attn.v.weight", (1536, 1536)),
    ("blocks.0.cross_attn.v.bias", (1536,)),
)
TEXT_SCHEMA = (
    ("text_embedding.0.weight", (1536, 4096)),
    ("text_embedding.0.bias", (1536,)),
    ("text_embedding.2.weight", (1536, 1536)),
    ("text_embedding.2.bias", (1536,)),
)

# The edge graphs are global one-plan stages, rather than per-block graphs.
# Their learned Parameters are still sourced from DIT (source 0) through the
# same authenticated DIT+LQ pair used by the other native tables.
EDGE_PREFIX_SCHEMA = (
    ("patch_embedding.weight", (1536, 16, 1, 2, 2)),
    ("patch_embedding.bias", (1536,)),
)
EDGE_SUFFIX_SCHEMA = (
    ("head.modulation", (1, 2, 1536)),
    ("head.proj.weight", (64, 1536)),
    ("head.proj.bias", (64,)),
)
EDGE_PREFIX_ASSET_PINS = {
    "xml": (
        "flash_edges_prefix_bf16.xml", 8_874,
        "a1ab4509d9c4a1b67a1ec54e1d555ee1ceec30e2930c7600326dda9c396c1004",
    ),
    "bin": (
        "flash_edges_prefix_bf16.bin", 664,
        "69bf6dce98cf98395030464b350ca048a648b7617ca864ced452b1db3d209bd1",
    ),
}
EDGE_SUFFIX_ASSET_PINS = {
    "xml": (
        "flash_edges_suffix_bf16.xml", 17_778,
        "20900cf95b1b32ab3b21d879a1f56210a4918b755acc34fb96b85e0c68f87fe5",
    ),
    "bin": (
        "flash_edges_suffix_bf16.bin", 328,
        "9bec50f4ae9fa668adee56efac658baba7e6f2a73301bc24b6fb613238cb39bb",
    ),
}

# The LQ projection is split into the same three causal roles used by the
# reference stream: a one-frame startup, a four-frame prefix (stage 1), and a
# suffix (stage 2).  The startup and prefix deliberately carry independent
# conv1/norm1 spans even though the source tensor bytes are the same.
LQ_ROLE_ORDER = ("startup", "prefix", "suffix")
LQ_ROLE_SCHEMAS = {
    "startup": (
        ("conv1.bias", (2048,)),
        ("conv1.weight", (2048, 768, 4, 3, 3)),
        ("norm1.weight", (2048, 1, 1, 1)),
    ),
    "prefix": (
        ("conv1.bias", (2048,)),
        ("conv1.weight", (2048, 768, 4, 3, 3)),
        ("norm1.weight", (2048, 1, 1, 1)),
    ),
    "suffix": (
        ("conv2.bias", (3072,)),
        ("conv2.weight", (3072, 2048, 4, 3, 3)),
        ("linear_layers.0.bias", (1536,)),
        ("linear_layers.0.weight", (1536, 3072)),
        ("norm2.weight", (3072, 1, 1, 1)),
    ),
}
LQ_SCHEMA_KEYS = frozenset(
    key for schema in LQ_ROLE_SCHEMAS.values() for key, _shape in schema
)

# These six pins come from the two identical, independently serialized
# attempt-8/attempt-9 exports.  The native build copies the matching files
# into ASSET_ROOT before invoking this generator; authentication remains
# fail-closed if any file is absent, renamed, resized, or changed.
LQ_ASSET_PINS: dict[str, dict[str, tuple[str, int, str]]] = {
    "startup": {
        "xml": (
            "flash_lq_projection_startup_bf16.xml", 19580,
            "0f7ac5a5908eeccfc811e323ef37824d09102ac1e653e0b812b3f9a59addf3e9",
        ),
        "bin": (
            "flash_lq_projection_startup_bf16.bin", 416,
            "34eac0b87a2297e790754b0fc3e9647a973ee1ba97dbe375243172fc96c970f8",
        ),
    },
    "prefix": {
        "xml": (
            "flash_lq_projection_stage1_bf16.xml", 19096,
            "376a58a703181e6acb619c0052b51139e7a0c15478080a25ca139d69e25460d1",
        ),
        "bin": (
            "flash_lq_projection_stage1_bf16.bin", 336,
            "8f79ea44054cc236f34c6836981475f280e1851193e5e9bf10c1f3a6853ceaf0",
        ),
    },
    "suffix": {
        "xml": (
            "flash_lq_projection_stage2_bf16.xml", 17783,
            "2496660b73fd26240a80a7f88acf5947f2d2f48e75082e8d099301292a94f8eb",
        ),
        "bin": (
            "flash_lq_projection_stage2_bf16.bin", 168,
            "99d49b0a03c208ba2067ce955fdc3b30bef5a4af7a65f8f55f7291fee3244b76",
        ),
    },
}
ASSET_PINS = {
    "xml": ("flash_post_4096_bf16.xml", 69_969,
            "1de0453729011b0d6e61f43a993e04b559fe27c3aa0e4894c69cf42f3afc3d37"),
    "bin": ("flash_post_4096_bf16.bin", 3_474,
            "4fbc48e7bdaf2e07ba701b43367a4b573e5aef6cdad16b75219395aedad4ab5a"),
    "config": ("bf16_cross_bridge_custom_layers.xml", 2_118,
                "7c00d36f62fe9e5e3e7d3b152ac9c543bc0df00aba5fd2ead6b47e4ef8d6e9f6"),
    "bridge": ("bf16_cross_bridge_20261017.cl", 5_221,
                "cc16f1801411f833fc447fd11e28531512e59e5223aecc28ebc36b6b50ca7395"),
    "sparse": ("plus_sparse_xmx.cl", 25_101,
                "eef7a585e37c4277d764fbe5f62110a012d68b02db0be4e852c722edc75738c2"),
}
PRE_ASSET_PINS = {
    "xml": ("flash_pre_4096_bf16.xml", 87_093,
            "a70c2bbb5fb0f65add9e93da461b9dee324a2fb5216c649ccf69edf84876087d"),
    "bin": ("flash_pre_4096_bf16.bin", 768,
            "b4ce6a73d4445949f43264f8334d57fcedf5b29504bc2222c110be3e7c60646e"),
}
CROSS_ASSET_PINS = {
    "xml": ("flash_cross_kv_bf16.xml", 9_493,
            "e57aee54756cd34d775371b8137358fff7ddfb4680ee680dd1c93dc1f299a82d"),
    "bin": ("flash_cross_kv_bf16.bin", 12,
            "862ca08291ff0bf5f96f630f2f8e3a42384ddc458dde33057ed473fe26612aa5"),
}
TEXT_ASSET_PINS = {
    "xml": ("flash_text_bf16.xml", 4_964,
            "d4f3a191a52f6bd499300310b90b9cfb70cf5bf92455b892549a6f0a81d03e21"),
    # The structural text graph contains no Constant nodes; its authenticated
    # complement is intentionally an exact empty file.
    "bin": ("flash_text_bf16.bin", 0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _inside(path: Path, root: Path, label: str) -> Path:
    root = root.resolve()
    raw = Path(os.path.abspath(path))
    cursor = raw
    while True:
        if cursor.is_symlink():
            raise ValueError(f"{label} uses a symlink: {path}")
        if cursor == root or cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        candidate = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its authenticated root: {path}") from exc
    if not stat.S_ISREG(candidate.stat().st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return candidate

def authenticate_file(path: Path, *, expected_bytes: int, expected_sha256: str,
                       root: Path | None, label: str) -> dict[str, Any]:
    candidate = _inside(path, root, label) if root is not None else Path(path).resolve(strict=True)
    if candidate.stat().st_size != expected_bytes:
        raise ValueError(f"{label} size differs: {candidate.stat().st_size} != {expected_bytes}")
    digest = sha256_file(candidate)
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs: {digest}")
    return {"name": candidate.name, "bytes": expected_bytes, "sha256": digest, "path": candidate}

def _checked_product(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        if type(dim) is not int or dim <= 0 or value > UINT64_MAX // dim:
            raise ValueError(f"invalid tensor shape: {shape!r}")
        value *= dim
    return value

def _parse_header(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    with path.open("rb") as stream:
        encoded = stream.read(8)
        if len(encoded) != 8:
            raise ValueError(f"{path} has no safetensors header length")
        header_bytes = struct.unpack("<Q", encoded)[0]
        if header_bytes < 2 or header_bytes > MAX_HEADER_BYTES:
            raise ValueError(f"unsafe safetensors header length: {header_bytes}")
        raw = stream.read(header_bytes)
        if len(raw) != header_bytes:
            raise ValueError(f"{path} safetensors header is truncated")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} safetensors header is invalid") from exc
    if type(header) is not dict:
        raise ValueError("safetensors header must be an object")
    tensors: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int, str]] = []
    payload_bytes = path.stat().st_size - 8 - header_bytes
    if payload_bytes < 0:
        raise ValueError("safetensors payload is truncated")
    for key, record in header.items():
        if key == "__metadata__":
            if type(record) is not dict:
                raise ValueError("safetensors metadata must be an object")
            continue
        if type(key) is not str or type(record) is not dict:
            raise ValueError("safetensors tensor record is malformed")
        if record.get("dtype") != "F32":
            raise ValueError(f"{key} is not an F32 source tensor")
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if type(shape) is not list or type(offsets) is not list or len(offsets) != 2:
            raise ValueError(f"{key} shape/offset metadata is malformed")
        if any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError(f"{key} has an invalid shape")
        if any(type(x) is not int or x < 0 for x in offsets):
            raise ValueError(f"{key} has invalid offsets")
        start, end = offsets
        elements = _checked_product(tuple(shape))
        if start > end or end > payload_bytes or end - start != elements * 4:
            raise ValueError(f"{key} F32 byte range differs from its shape")
        if start % 4 or (end - start) % 4:
            raise ValueError(f"{key} F32 range is not four-byte aligned")
        intervals.append((start, end, key))
        tensors[key] = {
            "shape": tuple(shape), "elements": elements, "relative_offset": start,
            "bytes": end - start, "source_offset": 8 + header_bytes + start,
        }
    for previous, current in zip(sorted(intervals), sorted(intervals)[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"safetensors data ranges overlap: {previous[2]} / {current[2]}")
    return header_bytes, tensors

def _source_layer(path: Path, layer: int, tensors: dict[str, dict[str, Any]],
                  schema: tuple[tuple[str, tuple[int, ...]], ...] = POST_SCHEMA,
                  stage: str = "POST") -> list[dict[str, Any]]:
    result = []
    for canonical, shape in schema:
        key = canonical.replace("blocks.0.", f"blocks.{layer}.", 1)
        if key not in tensors:
            raise ValueError(f"missing authenticated {stage} source tensor: {key}")
        record = tensors[key]
        if record["shape"] != shape:
            raise ValueError(f"{stage} source shape differs for {key}: {record['shape']!r}")
        result.append({"key": canonical, "source_key": key, **record})
    return result

def _convert(converter: Path, source: Path, offset: int, byte_count: int) -> bytes:
    with source.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(byte_count)
    if len(raw) != byte_count:
        raise ValueError("source span was truncated")
    completed = subprocess.run(
        [str(converter)], input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", "replace")[-4096:]
        raise RuntimeError(f"BF16 converter failed ({completed.returncode}): {detail}")
    expected = byte_count // 2
    if len(completed.stdout) != expected:
        raise ValueError(f"BF16 converter output size differs: {len(completed.stdout)} != {expected}")
    return completed.stdout

def _bytes_literal(digest: str) -> str:
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid SHA-256")
    return ", ".join(f"0x{digest[i:i+2]}" for i in range(0, 64, 2))

def _cpp_string(value: str) -> str:
    return json.dumps(value)

def _cpp_pin(name: str, size: int, digest: str) -> str:
    return f'{{{_cpp_string(name)}, {size}ULL, {{{_bytes_literal(digest)}}}}}'

def _plan_from_items(source: Path, items: list[dict[str, Any]], prefix: bytes,
                     converter: Path, *, learned_source: int = 0) -> tuple[list[dict[str, Any]], int, str]:
    if learned_source not in (0, 1):
        raise ValueError("learned source index must be 0 or 1")
    segments = []
    if prefix:
        segments.append({
            "offset": 0, "bytes": len(prefix), "source_offset": 0,
            "source": 2, "learned": False, "sha256": ZERO_SHA,
            "encoding": "copy",
        })
    whole = hashlib.sha256()
    whole.update(prefix)
    output_offset = len(prefix)
    for item in items:
        source_offset = item.get("source_offset")
        byte_count = item.get("bytes")
        if (type(source_offset) is not int or source_offset < 0 or
                source_offset > UINT64_MAX or type(byte_count) is not int or
                byte_count <= 0 or byte_count > UINT64_MAX or byte_count % 4):
            raise ValueError("learned source span is malformed")
        converted = _convert(converter, source, item["source_offset"], item["bytes"])
        digest = hashlib.sha256(converted).hexdigest()
        output_bytes = len(converted)
        segments.append({
            "offset": output_offset, "bytes": output_bytes,
            "source_offset": item["source_offset"], "source": learned_source,
            "learned": True, "sha256": digest,
            "encoding": "f32_to_bf16_rne",
        })
        whole.update(converted)
        output_offset += output_bytes
    return segments, output_offset, whole.hexdigest()


def _layer_plan(mode: str, layer: int, source: Path, source_tensors: dict[str, dict[str, Any]],
                prefix: bytes, converter: Path, *,
                schema: tuple[tuple[str, tuple[int, ...]], ...] = POST_SCHEMA,
                stage: str = "POST") -> tuple[list[dict[str, Any]], int, str]:
    return _plan_from_items(
        source, _source_layer(source, layer, source_tensors, schema=schema, stage=stage),
        prefix, converter,
    )


def _lq_source_items(source_tensors: dict[str, dict[str, Any]], role: str) -> list[dict[str, Any]]:
    if role not in LQ_ROLE_SCHEMAS:
        raise ValueError(f"unsupported LQ role: {role}")
    if frozenset(source_tensors) != LQ_SCHEMA_KEYS:
        missing = sorted(LQ_SCHEMA_KEYS - set(source_tensors))
        extra = sorted(set(source_tensors) - LQ_SCHEMA_KEYS)
        raise ValueError(f"LQ source key set differs: missing={missing}, extra={extra}")
    result = []
    for canonical, shape in LQ_ROLE_SCHEMAS[role]:
        record = source_tensors[canonical]
        if not isinstance(record, dict):
            raise ValueError(f"LQ source record is malformed for {canonical}")
        if record.get("shape") != shape:
            raise ValueError(f"LQ source shape differs for {canonical}: {record.get('shape')!r}")
        elements = _checked_product(shape)
        if (record.get("bytes") != elements * 4 or
                type(record.get("source_offset")) is not int or
                record["source_offset"] < 0 or
                record["source_offset"] > UINT64_MAX):
            raise ValueError(f"LQ source span differs for {canonical}")
        result.append({"key": canonical, "source_key": canonical, **record})
    return result


def _lq_plan(role: str, source: Path, source_tensors: dict[str, dict[str, Any]],
             prefix: bytes, converter: Path) -> tuple[list[dict[str, Any]], int, str]:
    return _plan_from_items(
        source, _lq_source_items(source_tensors, role), prefix, converter,
        learned_source=1,
    )

def _segment_cpp(segment: dict[str, Any]) -> str:
    sha = _bytes_literal(segment["sha256"])
    encoding = ("prepare_detail::SegmentEncoding::f32_to_bf16_rne"
                if segment["encoding"] == "f32_to_bf16_rne"
                else "prepare_detail::SegmentEncoding::copy")
    return (
        f'{{{segment["offset"]}ULL, {segment["bytes"]}ULL, '
        f'{segment["source_offset"]}ULL, {segment["source"]}u, '
        f'{"true" if segment["learned"] else "false"}, {{{sha}}}, {encoding}}}'
    )

def _render_stage_header(mode_records: dict[str, dict[str, Any]],
                         mode_plans: dict[str, list[dict[str, Any]]],
                         stage: str) -> str:
    if stage == "POST":
        schema = POST_SCHEMA
        asset_pins = ASSET_PINS
        graph_variant = "result_ir_assign_runtime"
        rows = ROWS
        family = "FlashPost"
        dimension_width = 3
    elif stage == "PRE":
        schema = PRE_SCHEMA
        asset_pins = PRE_ASSET_PINS
        graph_variant = "result_ir_ordinary_results"
        rows = ROWS
        family = "FlashPre"
        dimension_width = 3
    elif stage == "CROSS":
        schema = CROSS_SCHEMA
        asset_pins = CROSS_ASSET_PINS
        graph_variant = "result_ir_ordinary_results"
        rows = 512
        family = "FlashCross"
        dimension_width = 3
    elif stage == "TEXT":
        schema = TEXT_SCHEMA
        asset_pins = TEXT_ASSET_PINS
        graph_variant = "result_ir_ordinary_results"
        rows = 512
        family = "FlashText"
        dimension_width = 3
    elif stage == "EDGES_PREFIX":
        schema = EDGE_PREFIX_SCHEMA
        asset_pins = EDGE_PREFIX_ASSET_PINS
        graph_variant = "result_ir_ordinary_results"
        rows = 130560
        family = "FlashEdgesPrefix"
        dimension_width = 5
    elif stage == "EDGES_SUFFIX":
        schema = EDGE_SUFFIX_SCHEMA
        asset_pins = EDGE_SUFFIX_ASSET_PINS
        graph_variant = "result_ir_ordinary_results"
        rows = 130560
        family = "FlashEdgesSuffix"
        dimension_width = 5
    else:
        raise ValueError(f"unsupported stage: {stage}")
    structural_name = f"k{family}StructuralBin"

    if not mode_plans:
        raise ValueError("stage plans are empty")
    plan_counts = {len(plans) for plans in mode_plans.values()}
    if len(plan_counts) != 1:
        raise ValueError("stage plan counts differ between modes")
    plan_count = plan_counts.pop()
    expected_plan_count = 1 if stage.startswith("EDGES_") or stage == "TEXT" else LAYERS
    if plan_count != expected_plan_count:
        raise ValueError(
            f"{stage} requires {expected_plan_count} plan(s), got {plan_count}"
        )

    binding = []
    out_offset = len(next(iter(mode_records.values()))["prefix"])
    for canonical, shape in schema:
        elements = _checked_product(shape)
        if len(shape) > dimension_width:
            raise ValueError(f"{stage} binding rank exceeds generated width: {canonical}")
        dimensions = tuple(shape) + (1,) * (dimension_width - len(shape))
        binding.append(
            f'    {{{_cpp_string(canonical)}, '
            f'{_cpp_string("weight__" + canonical.replace(".", "_"))}, '
            f'{{{out_offset}ULL, {elements * 2}ULL, {elements}ULL}}, '
            f'{{{", ".join(f"{value}ULL" for value in dimensions)}}}, '
            f'{len(shape)}}},'
        )
        out_offset += elements * 2

    lines = [
        "// Generated by scripts/prepare-flash-post.py; do not edit.",
        "#pragma once",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        '#include "seedvr_prepare_detail.hpp"',
        "",
        "namespace otc::ai {",
        f"struct {family}WeightBinding {{",
        "  const char* canonical_key;",
        "  const char* port_name;",
        "  SeedVrWeightSpan span;",
        f"  std::array<std::size_t, {dimension_width}> dimensions;",
        "  std::size_t rank;",
        "};",
        f"inline const SeedVrAssetPin k{family}Xml = "
        + _cpp_pin(asset_pins["xml"][0], asset_pins["xml"][1], asset_pins["xml"][2]) + ";",
        f"inline const SeedVrAssetPin k{family}StructuralBin = "
        + _cpp_pin(asset_pins["bin"][0], asset_pins["bin"][1], asset_pins["bin"][2]) + ";",
    ]
    if stage == "POST":
        lines.extend([
            f"inline const SeedVrAssetPin k{family}Config = "
            + _cpp_pin(asset_pins["config"][0], asset_pins["config"][1], asset_pins["config"][2]) + ";",
            f"inline const SeedVrAssetPin k{family}BridgeSource = "
            + _cpp_pin(asset_pins["bridge"][0], asset_pins["bridge"][1], asset_pins["bridge"][2]) + ";",
            f"inline const SeedVrAssetPin k{family}SparseSource = "
            + _cpp_pin(asset_pins["sparse"][0], asset_pins["sparse"][1], asset_pins["sparse"][2]) + ";",
        ])
    lines.extend([
        f"inline constexpr std::uint32_t k{family}TableVersion = 1U;",
        f"inline constexpr std::size_t k{family}Rows = {rows}U;",
        f"inline constexpr std::size_t k{family}Layers = {plan_count}U;",
        f"inline const std::array<{family}WeightBinding, {len(schema)}> k{family}Weights{{{{",
    ])
    lines.extend(binding)
    lines.extend([
        "}};", "",
        f'inline constexpr const char* k{family}GraphVariant = "{graph_variant}";',
        "",
    ])
    for mode, source_records in mode_records.items():
        plans = mode_plans[mode]
        dit = source_records["dit"]
        mode_cap = mode.capitalize()
        selected_bins = f"k{family}{mode_cap}Bins"
        selected_plans = f"k{family}{mode_cap}Plans"
        selected_segments = f"k{family}{mode_cap}Segments"
        source_name = f"k{family}{mode_cap}Sources"
        header_name = f"k{family}{mode_cap}HeaderBytes"
        # All fixed-stage tables use the common two-source prepare contract.
        # CROSS consumes learned spans from DIT (source 0), but LQ remains an
        # authenticated source because prepare_detail::prepare validates the
        # complete DIT+LQ pair before binding any plan.
        lines.append(f"inline const SeedVrAssetPin {source_name}[2] = {{")
        lines.append("  " + _cpp_pin(dit["name"], dit["bytes"], dit["sha256"]) + ",")
        lq = source_records["lq"]
        lines.append("  " + _cpp_pin(lq["name"], lq["bytes"], lq["sha256"]))
        lines.append("};")
        lines.append(f"inline constexpr std::uint64_t {header_name}[2] = "
                     f"{{{dit['header_bytes']}ULL, {lq['header_bytes']}ULL}};")
        for index, plan in enumerate(plans):
            lines.append(f"inline const prepare_detail::Segment {selected_segments}_{index:02d}[] = {{")
            lines.extend("  " + _segment_cpp(segment) + "," for segment in plan["segments"])
            lines.append("};")
        lines.append(f"inline const std::array<SeedVrAssetPin, {plan_count}> {selected_bins}{{{{")
        lines.extend(
            "  " + _cpp_pin(record["name"], record["bytes"], record["sha256"]) + ","
            for record in (plan["bin"] for plan in plans)
        )
        lines.extend(["}};", ""])
        lines.append(f"inline const std::array<prepare_detail::Plan, {plan_count}> {selected_plans}{{{{")
        lines.extend(
            f"  {{{selected_bins}[{index}], {structural_name}, "
            f"{selected_segments}_{index:02d}, {len(plan['segments'])}U}},"
            for index, plan in enumerate(plans)
        )
        lines.extend(["}};", ""])
        lines.extend([
            f"inline constexpr const char* k{family}{mode_cap}SourceCommit = {_cpp_string(SOURCE_COMMIT)};",
            f"inline constexpr const char* k{family}{mode_cap}ModelRevision = {_cpp_string(MODEL_REPO_REVISION)};",
            "",
        ])
    lines.extend(["}", ""])
    return "\n".join(lines)


def _render_header(mode_records: dict[str, dict[str, Any]],
                   mode_plans: dict[str, list[dict[str, Any]]]) -> str:
    return _render_stage_header(mode_records, mode_plans, "POST")


def _render_pre_header(mode_records: dict[str, dict[str, Any]],
                       mode_plans: dict[str, list[dict[str, Any]]]) -> str:
    return _render_stage_header(mode_records, mode_plans, "PRE")


def _render_lq_header(
    mode_records: dict[str, dict[str, Any]],
    mode_plans: dict[str, dict[str, dict[str, Any]]],
) -> str:
    """Render the three-role LQ table without changing PRE/POST output."""

    if not mode_records or set(mode_records) - set(MODE_PINS):
        raise ValueError("LQ header requires one or more known modes")
    if set(mode_plans) != set(mode_records):
        raise ValueError("LQ mode records and plans differ")
    first = next(iter(mode_records.values()))
    role_records = first.get("roles")
    if not isinstance(role_records, dict) or tuple(role_records) != LQ_ROLE_ORDER:
        raise ValueError("LQ structural role pins are incomplete")
    for source_records in mode_records.values():
        if source_records.get("roles") != role_records:
            raise ValueError("LQ structural role pins differ between modes")

    lines = [
        "// Generated by scripts/prepare-flash-post.py; do not edit.",
        "#pragma once",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        '#include "seedvr_prepare_detail.hpp"',
        "",
        "namespace otc::ai {",
        "struct FlashLqWeightBinding {",
        "  const char* canonical_key;",
        "  const char* port_name;",
        "  SeedVrWeightSpan span;",
        "  std::array<std::size_t, 5> dimensions;",
        "  std::size_t rank;",
        "};",
        "inline constexpr std::uint32_t kFlashLqTableVersion = 1U;",
        "inline constexpr std::size_t kFlashLqRoleCount = 3U;",
        'inline constexpr const char* kFlashLqGraphVariant = "three_role_projection";',
        "",
    ]

    for role in LQ_ROLE_ORDER:
        role_cap = role.capitalize()
        asset = role_records[role]
        if set(asset) != {"xml", "bin"}:
            raise ValueError(f"LQ {role} structural pins must contain XML and BIN")
        lines.extend([
            f"inline const SeedVrAssetPin kFlashLq{role_cap}Xml = "
            + _cpp_pin(asset["xml"]["name"], asset["xml"]["bytes"], asset["xml"]["sha256"]) + ";",
            f"inline const SeedVrAssetPin kFlashLq{role_cap}StructuralBin = "
            + _cpp_pin(asset["bin"]["name"], asset["bin"]["bytes"], asset["bin"]["sha256"]) + ";",
        ])
        schema = LQ_ROLE_SCHEMAS[role]
        plan = next(iter(mode_plans.values()))[role]
        offset = asset["bin"]["bytes"]
        lines.append(
            f"inline const std::array<FlashLqWeightBinding, {len(schema)}> "
            f"kFlashLq{role_cap}Weights{{{{"
        )
        for canonical, shape in schema:
            elements = _checked_product(shape)
            if len(shape) > 5:
                raise ValueError(f"LQ rank exceeds generated binding width: {canonical}")
            dimensions = tuple(shape) + (1,) * (5 - len(shape))
            lines.append(
                f'    {{{_cpp_string(canonical)}, '
                f'{_cpp_string("weight__" + canonical.replace(".", "_"))}, '
                f'{{{offset}ULL, {elements * 2}ULL, {elements}ULL}}, '
                f'{{{", ".join(f"{value}ULL" for value in dimensions)}}}, '
                f'{len(shape)}U}},'
            )
            offset += elements * 2
        if offset != plan["bin"]["bytes"]:
            raise ValueError(f"LQ {role} binding span total differs from plan")
        lines.extend(["}};", ""])

    for mode, source_records in mode_records.items():
        mode_cap = mode.capitalize()
        plans = mode_plans.get(mode)
        if not isinstance(plans, dict) or tuple(plans) != LQ_ROLE_ORDER:
            raise ValueError(f"LQ {mode} plans are incomplete")
        dit = source_records["dit"]
        lq = source_records["lq"]
        lines.extend([
            f"inline const SeedVrAssetPin kFlashLq{mode_cap}Sources[2] = {{",
            "  " + _cpp_pin(dit["name"], dit["bytes"], dit["sha256"]) + ",",
            "  " + _cpp_pin(lq["name"], lq["bytes"], lq["sha256"]),
            "};",
            f"inline constexpr std::uint64_t kFlashLq{mode_cap}HeaderBytes[2] = "
            f"{{{dit['header_bytes']}ULL, {lq['header_bytes']}ULL}};",
        ])
        for role in LQ_ROLE_ORDER:
            role_cap = role.capitalize()
            plan = plans[role]
            segments_name = f"kFlashLq{mode_cap}{role_cap}Segments"
            lines.append(
                f"inline const prepare_detail::Segment {segments_name}[] = {{"
            )
            lines.extend("  " + _segment_cpp(segment) + "," for segment in plan["segments"])
            lines.extend(["};", ""])
        lines.append(
            f"inline const std::array<SeedVrAssetPin, 3> kFlashLq{mode_cap}Bins{{{{"
        )
        lines.extend(
            "  " + _cpp_pin(plans[role]["bin"]["name"], plans[role]["bin"]["bytes"],
                              plans[role]["bin"]["sha256"]) + ","
            for role in LQ_ROLE_ORDER
        )
        lines.extend(["}};", ""])
        lines.append(
            f"inline const std::array<prepare_detail::Plan, 3> kFlashLq{mode_cap}Plans{{{{"
        )
        for index, role in enumerate(LQ_ROLE_ORDER):
            role_cap = role.capitalize()
            plan = plans[role]
            lines.append(
                f"  {{kFlashLq{mode_cap}Bins[{index}], kFlashLq{role_cap}StructuralBin, "
                f"kFlashLq{mode_cap}{role_cap}Segments, {len(plan['segments'])}U}},"
            )
        lines.extend([
            "}};",
            f"inline constexpr const char* kFlashLq{mode_cap}SourceCommit = "
            f"{_cpp_string(SOURCE_COMMIT)};",
            f"inline constexpr const char* kFlashLq{mode_cap}ModelRevision = "
            f"{_cpp_string(MODEL_REPO_REVISION)};",
            "",
        ])
    lines.extend(["}", ""])
    return "\n".join(lines)

def _asset_records_for(pins: dict[str, tuple[str, int, str]]) -> dict[str, dict[str, Any]]:
    records = {}
    for label, (name, size, digest) in pins.items():
        records[label] = authenticate_file(
            ASSET_ROOT / name, expected_bytes=size, expected_sha256=digest,
            root=ASSET_ROOT, label=f"fixed {label} asset",
        )
    return records


def _asset_records() -> dict[str, dict[str, Any]]:
    return _asset_records_for(ASSET_PINS)


def _lq_asset_records() -> dict[str, dict[str, dict[str, Any]]]:
    if tuple(LQ_ASSET_PINS) != LQ_ROLE_ORDER:
        raise ValueError(
            "LQ structural asset pins are not installed; refusing an unpinned table"
        )
    result = {}
    for role in LQ_ROLE_ORDER:
        pins = LQ_ASSET_PINS[role]
        if set(pins) != {"xml", "bin"}:
            raise ValueError(f"LQ {role} structural pins must contain XML and BIN")
        result[role] = _asset_records_for(pins)
    return result

def _write_stage_provenance(
    path: Path,
    modes: tuple[str, ...],
    records: dict[str, Any],
    plans: dict[str, list[dict[str, Any]]],
    weights_root: Path,
    converter: Path,
    stage: str,
) -> None:
    if stage == "POST":
        lines = [
            "Flash POST native build-time pins",
            f"modes={','.join(modes)}",
            "graph_variant=result_ir_assign_runtime",
            f"source_commit={SOURCE_COMMIT}",
            f"model_repo_revision={MODEL_REPO_REVISION}",
            f"converter={converter.name}",
            f"weights_root={EXTERNAL_WEIGHTS_LABEL}",
            "learned_payload=external; this output contains tables, not learned data",
        ]
    elif stage == "PRE":
        lines = [
            "Flash PRE native build-time pins",
            f"modes={','.join(modes)}",
            "graph_variant=result_ir_ordinary_results",
            f"source_commit={SOURCE_COMMIT}",
            f"model_repo_revision={MODEL_REPO_REVISION}",
            f"converter={converter.name}",
            f"weights_root={EXTERNAL_WEIGHTS_LABEL}",
            f"fixed_xml={records['xml']['name']} bytes={records['xml']['bytes']} sha256={records['xml']['sha256']}",
            f"fixed_bin={records['bin']['name']} bytes={records['bin']['bytes']} sha256={records['bin']['sha256']}",
            "learned_payload=external; this output contains tables, not learned data",
        ]
    elif stage == "CROSS":
        lines = [
            "Flash CROSS K/V native build-time pins",
            f"modes={','.join(modes)}",
            "graph_variant=result_ir_ordinary_results",
            "rows=512",
            "learned_schema=5 per-layer cross-attention K/V tensors",
            f"source_commit={SOURCE_COMMIT}",
            f"model_repo_revision={MODEL_REPO_REVISION}",
            f"converter={converter.name}",
            f"weights_root={EXTERNAL_WEIGHTS_LABEL}",
            "authenticated_sources=DIT+LQ; cross learned spans use source 0",
            f"fixed_xml={records['xml']['name']} bytes={records['xml']['bytes']} sha256={records['xml']['sha256']}",
            f"fixed_bin={records['bin']['name']} bytes={records['bin']['bytes']} sha256={records['bin']['sha256']}",
            "learned_payload=external; this output contains tables, not learned data",
        ]
    elif stage == "TEXT":
        lines = [
            "Flash TEXT native build-time pins",
            f"modes={','.join(modes)}",
            "graph_variant=result_ir_ordinary_results",
            "rows=512",
            "learned_schema=4 global text-embedding tensors",
            "storage_boundary=text_linear before GELU(tanh)",
            f"source_commit={SOURCE_COMMIT}",
            f"model_repo_revision={MODEL_REPO_REVISION}",
            f"converter={converter.name}",
            f"weights_root={EXTERNAL_WEIGHTS_LABEL}",
            "authenticated_sources=DIT+LQ; text learned spans use source 0",
            f"fixed_xml={records['xml']['name']} bytes={records['xml']['bytes']} sha256={records['xml']['sha256']}",
            f"fixed_bin={records['bin']['name']} bytes={records['bin']['bytes']} sha256={records['bin']['sha256']}",
            "learned_payload=external; this output contains tables, not learned data",
        ]
    elif stage in {"EDGES_PREFIX", "EDGES_SUFFIX"}:
        role = "prefix" if stage == "EDGES_PREFIX" else "suffix"
        learned_count = len(EDGE_PREFIX_SCHEMA if role == "prefix" else EDGE_SUFFIX_SCHEMA)
        lines = [
            f"Flash EDGES {role.upper()} native build-time pins",
            f"modes={','.join(modes)}",
            "graph_variant=result_ir_ordinary_results",
            "geometry=working_2176x3840_four_latents",
            f"edge_role={role}",
            f"learned_schema={learned_count} global edge tensors",
            f"source_commit={SOURCE_COMMIT}",
            f"model_repo_revision={MODEL_REPO_REVISION}",
            f"converter={converter.name}",
            f"weights_root={EXTERNAL_WEIGHTS_LABEL}",
            "authenticated_sources=DIT+LQ; edge learned spans use source 0",
            f"fixed_xml={records['xml']['name']} bytes={records['xml']['bytes']} sha256={records['xml']['sha256']}",
            f"fixed_bin={records['bin']['name']} bytes={records['bin']['bytes']} sha256={records['bin']['sha256']}",
            "conditioning=caller prepares fixed t/t_mod from authenticated source-F32 values",
            "learned_payload=external; this output contains tables, not learned data",
        ]
    else:
        raise ValueError(f"unsupported stage: {stage}")
    for mode in modes:
        lines.append(f"[{mode}]")
        for index, plan in enumerate(plans[mode]):
            lines.append(
                f"layer_{index:02d}_bin={plan['bin']['name']} "
                f"bytes={plan['bin']['bytes']} sha256={plan['bin']['sha256']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_provenance(path: Path, modes: tuple[str, ...], records: dict[str, Any], plans: dict[str, list[dict[str, Any]]],
                      weights_root: Path, converter: Path) -> None:
    _write_stage_provenance(path, modes, records, plans, weights_root, converter, "POST")


def _write_pre_provenance(path: Path, modes: tuple[str, ...], records: dict[str, Any], plans: dict[str, list[dict[str, Any]]],
                          weights_root: Path, converter: Path) -> None:
    _write_stage_provenance(path, modes, records, plans, weights_root, converter, "PRE")


def _write_lq_provenance(
    path: Path,
    modes: tuple[str, ...],
    records: dict[str, dict[str, dict[str, Any]]],
    mode_records: dict[str, dict[str, Any]],
    plans: dict[str, dict[str, dict[str, Any]]],
    weights_root: Path,
    converter: Path,
) -> None:
    lines = [
        "Flash LQ projection native build-time pins",
        f"modes={','.join(modes)}",
        "graph_variant=three_role_projection_startup_prefix_suffix",
        f"source_commit={SOURCE_COMMIT}",
        f"model_repo_revision={MODEL_REPO_REVISION}",
        f"converter={converter.name}",
        f"weights_root={EXTERNAL_WEIGHTS_LABEL}",
        "learned_payload=external; this output contains tables, not learned data",
    ]
    for role in LQ_ROLE_ORDER:
        xml = records[role]["xml"]
        structural = records[role]["bin"]
        lines.append(
            f"[{role}] xml={xml['name']} bytes={xml['bytes']} sha256={xml['sha256']} "
            f"structural_bin={structural['name']} bytes={structural['bytes']} "
            f"sha256={structural['sha256']}"
        )
    for mode in modes:
        source = mode_records[mode]
        lines.extend([
            f"[{mode}]",
            f"dit_source={source['dit']['name']} bytes={source['dit']['bytes']} "
            f"sha256={source['dit']['sha256']} header_bytes={source['dit']['header_bytes']}",
            f"lq_source={source['lq']['name']} bytes={source['lq']['bytes']} "
            f"sha256={source['lq']['sha256']} header_bytes={source['lq']['header_bytes']}",
        ])
        for role in LQ_ROLE_ORDER:
            plan = plans[mode][role]
            lines.append(
                f"{role}_bin={plan['bin']['name']} bytes={plan['bin']['bytes']} "
                f"sha256={plan['bin']['sha256']} learned_weights="
                f"{len(LQ_ROLE_SCHEMAS[role])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _collect_stage(
    args: argparse.Namespace,
    *,
    schema: tuple[tuple[str, tuple[int, ...]], ...],
    asset_pins: dict[str, tuple[str, int, str]],
    stage: str,
    bin_pattern: str,
    plan_count: int = LAYERS,
) -> tuple[
    tuple[str, ...], Path, dict[str, dict[str, Any]],
    dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], Path,
]:
    if type(plan_count) is not int or plan_count <= 0:
        raise ValueError("stage plan count must be a positive integer")
    modes = tuple(MODE_PINS) if args.mode == "both" else (args.mode,)
    converter = _inside(Path(args.bf16_converter), Path(args.bf16_converter).resolve().parent, "BF16 converter")
    if not os.access(converter, os.X_OK):
        raise ValueError("BF16 converter is not executable")
    records = _asset_records_for(asset_pins)
    weights_root_arg = Path(args.weights_root).expanduser()
    if weights_root_arg.is_symlink():
        raise ValueError("weights-root must not be a symlink")
    weights_root = weights_root_arg.resolve(strict=True)
    if not weights_root.is_dir():
        raise ValueError("weights-root must be a real directory")
    mode_records = {}
    mode_plans = {}
    prefix = records["bin"]["path"].read_bytes()
    for mode in modes:
        mode_root = weights_root / mode
        dit_path = mode_root / "dit.safetensors"
        dit_pin = MODE_PINS[mode]["dit"]
        dit = authenticate_file(dit_path, expected_bytes=dit_pin[0], expected_sha256=dit_pin[1],
                                root=weights_root, label=f"{mode} DIT")
        dit_header, dit_tensors = _parse_header(dit["path"])
        source_records = {"dit": {**dit, "header_bytes": dit_header}, "prefix": prefix}
        lq_path = mode_root / "lq_proj_in.safetensors"
        lq_pin = MODE_PINS[mode]["lq"]
        lq = authenticate_file(lq_path, expected_bytes=lq_pin[0], expected_sha256=lq_pin[1],
                               root=weights_root, label=f"{mode} LQ")
        lq_header, _ = _parse_header(lq["path"])
        source_records["lq"] = {**lq, "header_bytes": lq_header}
        mode_records[mode] = source_records
        plans = []
        for layer in range(plan_count):
            segments, output_bytes, output_sha = _layer_plan(
                mode, layer, dit["path"], dit_tensors, prefix, converter,
                schema=schema, stage=stage,
            )
            plans.append({
                "bin": {
                    "name": bin_pattern.format(mode=mode, layer=layer),
                    "bytes": output_bytes, "sha256": output_sha,
                },
                "segments": segments,
            })
        mode_plans[mode] = plans
    return modes, converter, records, mode_records, mode_plans, weights_root


def _collect_lq(
    args: argparse.Namespace,
) -> tuple[
    tuple[str, ...], Path, dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]], Path,
]:
    """Collect authenticated DIT/LQ sources and the three LQ role plans."""

    modes = tuple(MODE_PINS) if args.mode == "both" else (args.mode,)
    converter = _inside(
        Path(args.bf16_converter), Path(args.bf16_converter).resolve().parent,
        "BF16 converter",
    )
    if not os.access(converter, os.X_OK):
        raise ValueError("BF16 converter is not executable")
    role_records = _lq_asset_records()
    weights_root_arg = Path(args.weights_root).expanduser()
    if weights_root_arg.is_symlink():
        raise ValueError("weights-root must not be a symlink")
    weights_root = weights_root_arg.resolve(strict=True)
    if not weights_root.is_dir():
        raise ValueError("weights-root must be a real directory")

    mode_records: dict[str, dict[str, Any]] = {}
    mode_plans: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in modes:
        mode_root = weights_root / mode
        dit_pin = MODE_PINS[mode]["dit"]
        lq_pin = MODE_PINS[mode]["lq"]
        dit = authenticate_file(
            mode_root / "dit.safetensors", expected_bytes=dit_pin[0],
            expected_sha256=dit_pin[1], root=weights_root, label=f"{mode} DIT",
        )
        lq = authenticate_file(
            mode_root / "lq_proj_in.safetensors", expected_bytes=lq_pin[0],
            expected_sha256=lq_pin[1], root=weights_root, label=f"{mode} LQ",
        )
        dit_header, _dit_tensors = _parse_header(dit["path"])
        lq_header, lq_tensors = _parse_header(lq["path"])
        # The DIT source is authenticated because the native full-pack
        # preparation API requires both source pins, even though LQ segments
        # use source index 1 only.
        mode_records[mode] = {
            "dit": {**dit, "header_bytes": dit_header},
            "lq": {**lq, "header_bytes": lq_header},
            "roles": role_records,
        }
        plans: dict[str, dict[str, Any]] = {}
        for role in LQ_ROLE_ORDER:
            prefix = role_records[role]["bin"]["path"].read_bytes()
            segments, output_bytes, output_sha = _lq_plan(
                role, lq["path"], lq_tensors, prefix, converter
            )
            plans[role] = {
                "bin": {
                    "name": f"flash_lq_projection_bf16_{mode}_{role}.bin",
                    "bytes": output_bytes,
                    "sha256": output_sha,
                },
                "complement": role_records[role]["bin"],
                "segments": segments,
            }
        mode_plans[mode] = plans
    return modes, converter, role_records, mode_records, mode_plans, weights_root


def _build_stage(
    args: argparse.Namespace,
    *,
    stage: str,
    schema: tuple[tuple[str, tuple[int, ...]], ...],
    asset_pins: dict[str, tuple[str, int, str]],
    bin_pattern: str,
    header_name: str,
    provenance_name: str,
    readme: str,
    plan_count: int = LAYERS,
) -> Path:
    modes, converter, records, mode_records, mode_plans, weights_root = _collect_stage(
        args, schema=schema, asset_pins=asset_pins, stage=stage,
        bin_pattern=bin_pattern, plan_count=plan_count,
    )
    output = fresh_output(args.output)
    (output / header_name).write_text(
        _render_stage_header(mode_records, mode_plans, stage), encoding="utf-8"
    )
    _write_stage_provenance(
        output / provenance_name, modes, records, mode_plans,
        weights_root, converter, stage,
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")
    return output


def _build_post(args: argparse.Namespace) -> Path:
    return _build_stage(
        args, stage="POST", schema=POST_SCHEMA, asset_pins=ASSET_PINS,
        bin_pattern="flash_post_4096_bf16_state_{mode}_layer{layer:02d}.bin",
        header_name="flash_post_assets.hpp",
        provenance_name="flash_post_provenance.txt",
        readme=(
            "Generated native Flash POST table. Learned per-layer BINs are prepared by "
            "the fixed native Segment plan from authenticated external safetensors; "
            "this directory contains no learned payload.\\n"
        ),
    )


def _build_pre(args: argparse.Namespace) -> Path:
    return _build_stage(
        args, stage="PRE", schema=PRE_SCHEMA, asset_pins=PRE_ASSET_PINS,
        bin_pattern="flash_pre_4096_bf16_{mode}_layer{layer:02d}.bin",
        header_name="flash_pre_assets.hpp",
        provenance_name="flash_pre_provenance.txt",
        readme=(
            "Generated native Flash PRE table. The ordinary Result graph is "
            "authenticated by the fixed structural asset; six PRE state sinks "
            "are recreated by the native stage. Learned per-layer BINs are "
            "prepared from authenticated external safetensors.\n"
        ),
    )


def _build_cross(args: argparse.Namespace) -> Path:
    return _build_stage(
        args, stage="CROSS", schema=CROSS_SCHEMA, asset_pins=CROSS_ASSET_PINS,
        bin_pattern="flash_cross_kv_bf16_{mode}_layer{layer:02d}.bin",
        header_name="flash_cross_assets.hpp",
        provenance_name="flash_cross_provenance.txt",
        readme=(
            "Generated native Flash cross K/V table. The ordinary Result graph is "
            "authenticated by the fixed BF16 structural asset; five per-layer "
            "cross-attention tensors are prepared from authenticated DIT safetensors. "
            "This directory contains no learned payload.\n"
        ),
    )


def _build_text(args: argparse.Namespace) -> Path:
    return _build_stage(
        args, stage="TEXT", schema=TEXT_SCHEMA, asset_pins=TEXT_ASSET_PINS,
        bin_pattern="flash_text_bf16_{mode}.bin",
        header_name="flash_text_assets.hpp",
        provenance_name="flash_text_provenance.txt",
        plan_count=1,
        readme=(
            "Generated native Flash text-embedding table. The fixed BF16 graph "
            "exposes prompt, context, and text_linear with a pre-GELU storage "
            "barrier; learned tensors are prepared from authenticated external "
            "DIT safetensors and are not stored here.\n"
        ),
    )


def _build_edges_prefix(args: argparse.Namespace) -> Path:
    return _build_stage(
        args, stage="EDGES_PREFIX", schema=EDGE_PREFIX_SCHEMA,
        asset_pins=EDGE_PREFIX_ASSET_PINS,
        bin_pattern="flash_edges_prefix_bf16_{mode}.bin",
        header_name="flash_edges_prefix_assets.hpp",
        provenance_name="flash_edges_prefix_provenance.txt",
        plan_count=1,
        readme=(
            "Generated native Flash edge-prefix table. The two learned BF16 "
            "spans are prepared from authenticated external DIT source data; "
            "fixed timestep conditioning is prepared by the caller.\n"
        ),
    )


def _build_edges_suffix(args: argparse.Namespace) -> Path:
    return _build_stage(
        args, stage="EDGES_SUFFIX", schema=EDGE_SUFFIX_SCHEMA,
        asset_pins=EDGE_SUFFIX_ASSET_PINS,
        bin_pattern="flash_edges_suffix_bf16_{mode}.bin",
        header_name="flash_edges_suffix_assets.hpp",
        provenance_name="flash_edges_suffix_provenance.txt",
        plan_count=1,
        readme=(
            "Generated native Flash edge-suffix table. The three learned BF16 "
            "spans are prepared from authenticated external DIT source data; "
            "the table contains no activation or conditioning payload.\n"
        ),
    )


def _build_lq(args: argparse.Namespace) -> Path:
    modes, converter, role_records, mode_records, mode_plans, weights_root = _collect_lq(args)
    output = fresh_output(args.output)
    (output / "flash_lq_assets.hpp").write_text(
        _render_lq_header(mode_records, mode_plans), encoding="utf-8"
    )
    _write_lq_provenance(
        output / "flash_lq_provenance.txt", modes, role_records, mode_records,
        mode_plans, weights_root, converter,
    )
    (output / "README.txt").write_text(
        "Generated native Flash LQ projection table. The three startup/prefix/suffix "
        "BINs are prepared from the authenticated DIT+LQ source pair and pinned "
        "structural role complements; no learned payload is stored in this table "
        "directory.\n",
        encoding="utf-8",
    )
    return output


def build(args: argparse.Namespace) -> Path:
    stage = getattr(args, "stage", "post")
    if stage == "post":
        return _build_post(args)
    if stage == "pre":
        return _build_pre(args)
    if stage == "cross":
        return _build_cross(args)
    if stage == "text":
        return _build_text(args)
    if stage == "edges-prefix":
        return _build_edges_prefix(args)
    if stage == "edges-suffix":
        return _build_edges_suffix(args)
    if stage == "lq":
        return _build_lq(args)
    raise ValueError(f"unsupported stage: {stage}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("turbo", "base", "both"), default="both")
    parser.add_argument(
        "--stage",
        choices=("pre", "post", "cross", "text", "edges-prefix", "edges-suffix", "lq"),
        default="post",
    )
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--bf16-converter", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = build(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"generated {output}")

if __name__ == "__main__":
    main()
