#!/usr/bin/env python3
"""Prepare fixed Flash timestep conditioning as derived BF16 asset bytes.

This is a bounded build-time exporter.  The reference root is trusted
executable source; recorded hashes are provenance, not a sandbox.  It loads
only the eleven authenticated edge tensors from each pinned DIT checkpoint,
derives the fixed ``t`` and ``t_mod`` values with the reviewed source helper,
and writes the resulting raw BF16 bytes to a fresh cache directory.  No
OpenVINO Core, device, graph compilation, or checkpoint tensor bytes are
copied into the output; the eleven learned edge tensors are read only to
derive the fixed conditioning values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from flash_export import (
    checked_reference_file,
    checked_reference_root,
    fresh_output,
    graph_source_records,
    reference_import_path,
    require_reference_modules,
    require_runtime_version,
    sha256_file,
)


OPENVINO_VERSION = "2026.5.0-23457-7d81aad55a5"
T_SHAPE = (1, 1, 1536)
T_MOD_SHAPE = (1, 6, 1536)
T_BYTES = 3072
T_MOD_BYTES = 18432
CONDITIONING_BYTES = T_BYTES + T_MOD_BYTES
FIXED_TIMESTEP = 1000.0
MODES = ("turbo", "base")

REFERENCE_MODULES = {
    "plus_edges": "openvino/full_runtime/plus_edges.py",
    "plus_storage": "openvino/full_runtime/plus_storage.py",
    "plus_modes": "openvino/full_runtime/plus_modes.py",
}
REQUIRED_REFERENCE_FILES = tuple(REFERENCE_MODULES.values())

# This is the independently audited ZIP member span; it is metadata only.  No
# ZIP parser or prompt payload is read by this exporter.
PROMPT_OUTER = {
    "name": "posi_prompt.pth",
    "bytes": 4_195_504,
    "sha256": "4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef",
}
PROMPT_RAW = {
    # Native validation consumes the installed leaf, while the archive member
    # is retained as provenance rather than being used as a filesystem name.
    "name": "flash_prompt_bf16.bin",
    "archive_member": "posi_prompt/data/0",
    "bytes": 4_194_304,
    "sha256": "ae3fbad78e1bc23148d736d95fa269517d76ecf21dc2ff17ba63a54f6cafd3e9",
}
PROMPT_RAW_OFFSET = 448
PROMPT_SHAPE = [1, 512, 4096]
SOURCE_ORACLE_EVIDENCE = {
    "receipt": "fixed-conditioning-source-review-20261016",
    "receipt_sha256": "e243a9d359f3e7f39d5f2fa212af682fce56b2034db9962d9d7a6d4059de81fd",
    "mode": "turbo",
    "embedding_bits_different": 0,
    "time_embedding_different_elements": 2,
    "time_modulation_different_elements": 2,
    "interpretation": (
        "Independent CPU source comparison found only two BF16 differences in each "
        "derived time output from arithmetic ordering; no missing storage boundary."
    ),
}


def _sha_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _pin_literal(record: Mapping[str, Any]) -> str:
    digest = record["sha256"]
    if (type(digest) is not str or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("asset hash must be lowercase SHA-256")
    byte_list = ", ".join(f"0x{digest[index:index + 2]}"
                           for index in range(0, len(digest), 2))
    return "{" + byte_list + "}"


def _validate_words(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    import numpy as np

    words = np.asarray(value)
    if words.dtype != np.dtype(np.uint16) or words.shape != shape:
        raise ValueError(f"{name} must be contiguous uint16 BF16 words with shape {shape}")
    if not words.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    flat = words.reshape(-1)
    for start in range(0, flat.size, 1 << 20):
        if np.any((flat[start:start + (1 << 20)] & np.uint16(0x7FFF)) >= np.uint16(0x7F80)):
            raise ValueError(f"{name} contains non-finite BF16 words")
    return words.astype("<u2", copy=False)


def pack_conditioning_words(t_words: Any, t_mod_words: Any) -> bytes:
    """Pack the exact little-endian ``t`` then ``t_mod`` BF16 storage bytes."""

    t = _validate_words(t_words, T_SHAPE, "t")
    t_mod = _validate_words(t_mod_words, T_MOD_SHAPE, "t_mod")
    payload = t.tobytes(order="C") + t_mod.tobytes(order="C")
    if len(payload) != CONDITIONING_BYTES:
        raise ValueError("fixed conditioning payload size changed")
    return payload


def _storage_words(value: Any, storage_tensor: Any, name: str,
                   shape: tuple[int, ...]) -> np.ndarray:
    import numpy as np

    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32) or array.shape != shape:
        raise ValueError(f"source {name} has unexpected F32 shape/dtype")
    if not array.flags.c_contiguous or not np.isfinite(array).all():
        raise ValueError(f"source {name} must be finite contiguous F32")
    tensor = storage_tensor(np.ascontiguousarray(array), "bf16")
    # storage_tensor owns BF16 bits; never reinterpret through numeric F16.
    words = np.asarray(tensor.data).view(np.uint16).reshape(shape).copy()
    return _validate_words(words, shape, name)


def derive_conditioning_words(weights: Mapping[str, Any], fixed_conditioning: Any,
                              storage_tensor: Any) -> tuple[np.ndarray, np.ndarray]:
    """Run the reviewed fused-boundary source helper and retain raw BF16 bits."""

    result = fixed_conditioning(dict(weights), "bf16")
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("fixed_conditioning returned an unexpected result")
    t, t_mod = result[:2]
    return (
        _storage_words(t, storage_tensor, "t", T_SHAPE),
        _storage_words(t_mod, storage_tensor, "t_mod", T_MOD_SHAPE),
    )


def _mode_asset_record(mode: str, payload: bytes) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("mode must be exactly turbo or base")
    return {
        "name": f"flash_fixed_conditioning_bf16_{mode}.bin",
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
        "dtype": "bf16_raw_le",
        "derived": True,
        "t": {"offset": 0, "bytes": T_BYTES, "shape": list(T_SHAPE)},
        "t_mod": {"offset": T_BYTES, "bytes": T_MOD_BYTES, "shape": list(T_MOD_SHAPE)},
    }


def _render_header(records: Mapping[str, Mapping[str, Any]],
                   modes: tuple[str, ...]) -> str:
    def pin(name: str, record: Mapping[str, Any]) -> str:
        return (f'{{"{name}", {int(record["bytes"])}ULL, '
                f'{_pin_literal(record)}}}')

    lines = [
        "// Generated by scripts/export-flash-conditioning.py; do not edit.",
        "#pragma once",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        '#include "seedvr_assets.hpp"',
        "",
        "namespace otc::ai {",
        "struct FlashFixedConditioningAsset {",
        "  const char* mode;",
        "  SeedVrAssetPin bin;",
        "};",
        "struct FlashFixedPromptSpan {",
        "  SeedVrAssetPin outer, raw;",
        "  std::uint64_t rawOffset;",
        "  std::array<std::size_t, 3> shape;",
        "};",
        f"inline constexpr std::uint32_t kFlashFixedTableVersion = 1U;",
        f"inline constexpr std::uint64_t kFlashFixedConditioningBytes = {CONDITIONING_BYTES}ULL;",
        f"inline constexpr std::uint64_t kFlashFixedTBytes = {T_BYTES}ULL;",
        f"inline constexpr std::uint64_t kFlashFixedTModBytes = {T_MOD_BYTES}ULL;",
        "inline constexpr std::uint64_t kFlashFixedTOffset = 0ULL;",
        f"inline constexpr std::uint64_t kFlashFixedTModOffset = {T_BYTES}ULL;",
        "inline constexpr std::array<std::size_t, 3> kFlashFixedTShape{1ULL, 1ULL, 1536ULL};",
        "inline constexpr std::array<std::size_t, 3> kFlashFixedTModShape{1ULL, 6ULL, 1536ULL};",
        "inline constexpr const char* kFlashFixedGraphVariant = \"derived_fixed_bf16_conditioning\";",
        "inline constexpr const char* kFlashFixedSourceCommit = \"b548a2cc2c929d3d4676f2973abe27876ec4dee7\";",
        "inline constexpr const char* kFlashFixedModelRevision = \"46b79520946ac67dec94d1566490690953c7debc\";",
        "inline const FlashFixedPromptSpan kFlashFixedPrompt = {",
        f'  {pin(PROMPT_OUTER["name"], PROMPT_OUTER)},',
        f'  {pin(PROMPT_RAW["name"], PROMPT_RAW)},',
        f"  {PROMPT_RAW_OFFSET}ULL, {{1ULL, 512ULL, 4096ULL}}",
        "};",
    ]
    lines.extend([
        f"inline const std::array<FlashFixedConditioningAsset, {len(modes)}> kFlashFixedAssets{{{{",
    ])
    for mode in modes:
        record = records[mode]
        lines.append(
            f'  {{"{mode}", {pin(record["name"], record)}}},'
        )
    lines.extend([
        "}};",
        "}  // namespace otc::ai",
        "",
    ])
    return "\n".join(lines)


def _write_outputs(destination: Path, records: Mapping[str, Mapping[str, Any]],
                   source_records: Mapping[str, Mapping[str, Any]],
                   modes: tuple[str, ...],
                   source_files: list[dict[str, Any]], runtime_version: str,
                   exporter_sha: str) -> dict[str, Any]:
    header = _render_header(records, modes).encode("utf-8")
    if any(token in header for token in (b"/home/", b"/cache/", b"/tmp/")):
        raise ValueError("generated conditioning header contains a private path")
    header_path = destination / "flash_fixed_assets.hpp"
    header_path.write_bytes(header)
    provenance = {
        "schema": 1,
        "status": "pass",
        "scope": "derived fixed Flash timestep conditioning; no checkpoint tensor bytes copied into output",
        "trust": {
            "reference_root": "explicit curated executable source; not sandboxed",
            "source_hashes_are_provenance_only": True,
            "derived_arrays_are_not_source_spans": True,
            "learned_payloads_written": False,
        },
        "conditioning": {
            "fixed_timestep": FIXED_TIMESTEP,
            "storage": "raw little-endian BF16 words",
            "conditioning_bytes": CONDITIONING_BYTES,
            "t_shape": list(T_SHAPE),
            "t_mod_shape": list(T_MOD_SHAPE),
            "t_offset": 0,
            "t_mod_offset": T_BYTES,
            "source_helper": "plus_edges.fixed_conditioning(weights, 'bf16')",
        },
        "source_parity": SOURCE_ORACLE_EVIDENCE,
        "prompt": {
            "outer": PROMPT_OUTER,
            "raw_span": {**PROMPT_RAW, "source_offset": PROMPT_RAW_OFFSET},
            "shape": PROMPT_SHAPE,
            "dtype": "bf16_raw_le",
            "runtime_parser": False,
        },
        "openvino": {
            "required_version": OPENVINO_VERSION,
            "actual_version": runtime_version,
            "core_created": False,
            "device_queries": False,
            "compilation_performed": False,
            "inference_performed": False,
        },
        "source_files": source_files,
        "modes": {mode: {"dit": source_records[mode], "artifact": records[mode]}
                  for mode in modes},
        "exporter_sha256": exporter_sha,
    }
    provenance_bytes = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    if any(token in provenance_bytes for token in (b"/home/", b"/cache/", b"/tmp/")):
        raise ValueError("generated conditioning provenance contains a private path")
    provenance_path = destination / "flash_fixed_provenance.json"
    provenance_path.write_bytes(provenance_bytes)
    return {
        "header": {"path": header_path.name, "bytes": len(header),
                   "sha256": sha256_file(header_path)},
        "provenance": {"path": provenance_path.name, "bytes": len(provenance_bytes),
                        "sha256": sha256_file(provenance_path)},
    }


def export_conditioning(reference_root: Path, model_root: Path, output: Path,
                        modes: Iterable[str] = MODES) -> dict[str, Any]:
    selected = tuple(modes)
    if not selected or any(mode not in MODES for mode in selected) or len(set(selected)) != len(selected):
        raise ValueError("modes must be a non-empty subset of turbo/base")
    root = checked_reference_root(reference_root, REQUIRED_REFERENCE_FILES)
    model_root = checked_reference_root(
        Path(model_root), [f"{mode}/dit.safetensors" for mode in selected]
    )
    destination = fresh_output(output)
    with reference_import_path(root):
        import openvino as ov
        import plus_edges
        import plus_storage
        import plus_modes
        from plus_modes import resolve_plus_mode

        runtime_version = str(ov.get_version())
        require_runtime_version(runtime_version, OPENVINO_VERSION)
        require_reference_modules(root, REFERENCE_MODULES)
        source_records: dict[str, dict[str, Any]] = {}
        records: dict[str, dict[str, Any]] = {}
        for mode in selected:
            spec = resolve_plus_mode(mode)
            dit = checked_reference_file(model_root, f"{mode}/dit.safetensors")
            if dit.stat().st_size != spec.dit_bytes:
                raise ValueError(f"unexpected {mode} DIT size")
            weights, metadata = plus_edges.load_edge_weights(dit, verify_hash=True, mode=mode)
            t_words, t_mod_words = derive_conditioning_words(
                weights, plus_edges.fixed_conditioning, plus_storage.storage_tensor
            )
            payload = pack_conditioning_words(t_words, t_mod_words)
            record = _mode_asset_record(mode, payload)
            (destination / record["name"]).write_bytes(payload)
            records[mode] = record
            source_records[mode] = {
                "name": "dit.safetensors",
                "bytes": int(metadata["checkpoint_bytes"]),
                "sha256": str(metadata["checkpoint_sha256"]),
                "source_commit": str(metadata["source_commit"]),
                "model_revision": str(plus_modes.MODEL_REPO_REVISION),
                "selected_keys": list(metadata["selected_keys"]),
                "selected_metadata": metadata["selected_metadata"],
            }
        source_files = graph_source_records(root, REQUIRED_REFERENCE_FILES)
    outputs = _write_outputs(
        destination, records, source_records, selected, source_files, runtime_version,
        sha256_file(Path(__file__).resolve()),
    )
    manifest = {
        "schema": 1,
        "status": "pass",
        "modes": list(selected),
        "artifacts": {mode: records[mode] for mode in selected},
        **outputs,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("turbo", "base", "both"), default="both")
    args = parser.parse_args(list(argv) if argv is not None else None)
    modes = MODES if args.mode == "both" else (args.mode,)
    try:
        manifest = export_conditioning(args.reference_root, args.model_root, args.output, modes)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({"status": manifest["status"], "artifacts": manifest["artifacts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
