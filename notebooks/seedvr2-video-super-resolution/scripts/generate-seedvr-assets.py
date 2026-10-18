#!/usr/bin/env python3
"""Generate reviewed native asset tables; never promote arbitrary model hashes to trust."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import xml.etree.ElementTree as ET

SOURCE_SHA = "b77d41d1a4d3a0010265ffff293ffa019b6edcd50770cad40c38f61657717493"
CONTROL_SHA = "ed9a870b8a9d4f6f33832edefb4ab01a91c99bac036a6bbe252d5ff83c2b0bb7"
REPS = (0, 1, 2, 10, 11, 31)
U64_MAX = 2**64 - 1
PIN_RE = r'^\{"([^"\n]+)", (\d+)ULL, "([0-9a-f]{64})"\},$'
WEIGHT_RE = r'^\{"(weight_\d+)", ov::Shape\{([0-9,]+)\}, (\d+)ULL, (\d+)ULL, "([0-9a-f]{64})"\},$'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trusted_text(path, expected):
    require(path.stat().st_size < 1024 * 1024, "Source manifest is oversized")
    with path.open("rb") as file:
        data = file.read(1024 * 1024)
    require(hashlib.sha256(data).hexdigest() == expected, "Fixed source-manifest SHA differs")
    return data.decode("utf-8")


def section(text, declaration):
    return text.split(declaration + " = {\n", 1)[1].split("\n};", 1)[0]


def parse_pins(text, declaration):
    rows = re.findall(PIN_RE, section(text, declaration), re.M)
    pins = {name: (int(size), sha) for name, size, sha in rows}
    require(len(rows) == len(pins), "Duplicate source pin")
    return pins


def representative(block):
    if block in (0, 31):
        return block
    return (1 if block % 2 else 2) if block < 10 else (11 if block % 2 else 10)


def parse(source, control):
    pins = parse_pins(source, "const std::vector<Pin> kPins")
    controls = parse_pins(control, "const std::vector<Pin> kControlPins")
    blocks = []
    body = section(source, "const std::vector<BlockSpec> kBlocks")
    for number, binary, weights in re.findall(r'^\{(\d+), "([^"\n]+)", \{\n(.*?)^\}\},$', body, re.M | re.S):
        rows = [(name, tuple(map(int, dims.split(","))), int(offset), int(size), sha)
                for name, dims, offset, size, sha in re.findall(WEIGHT_RE, weights, re.M)]
        index = int(number)
        require(index == len(blocks) and len(rows) == (28 if index < 10 else 16 if index == 31 else 23),
                "Block order or span count differs")
        require(len({row[0] for row in rows}) == len(rows), "Duplicate weight name")
        require(binary in pins, "BIN lacks trusted full-file pin")
        for name, dims, offset, size, _ in rows:
            elements = math.prod(dims)
            require(1 <= len(dims) <= 3 and all(d > 0 for d in dims) and elements <= U64_MAX // 2,
                    "Invalid or overflowing weight shape")
            require(size == elements * 2 and offset <= pins[binary][0]
                    and size <= pins[binary][0] - offset and offset % 2 == 1, "Invalid learned span")
        previous = None
        for row in sorted(rows, key=lambda row: row[2]):
            # OpenVINO serialized equal constants share exact byte ranges.
            if previous is not None and row[2] < previous[2] + previous[3]:
                require(row[1:] == previous[1:], "Learned spans partially overlap or disagree")
            previous = row
        blocks.append({"bin": binary, "weights": rows})
    require(len(blocks) == 32 and sum(len(x["weights"]) for x in blocks) == 779, "Frozen block count differs")
    templates = re.findall(r'^\{(\d+), "([^"\n]+)", "([^"\n]+)"\},$',
                           section(source, "const std::vector<TemplateSpec> kTemplates"), re.M)
    require(tuple(int(x[0]) for x in templates) == REPS, "Template representatives differ")
    for index, block in enumerate(blocks):
        # SHA differs between blocks; the reusable schema deliberately excludes learned values.
        schema = lambda b: [row[:4] for row in b["weights"]]
        require(schema(block) == schema(blocks[representative(index)]), "Six-class span schema is not exact")
        xml = str(Path(block["bin"]).with_suffix(".xml"))
        require(xml in controls, "Original XML lacks trusted pin")
        block["xml"] = xml
    for rep, xml, binary in templates:
        require(xml in pins and binary == blocks[int(rep)]["bin"], "Template source association differs")
    return pins, controls, blocks, templates


def verify(path, pin, weights=()):
    require(stat.S_ISREG(path.lstat().st_mode), "Asset is not a regular non-symlink file")
    size, expected = pin
    digest = hashlib.sha256()
    span_hashes = [hashlib.sha256() for _ in weights]
    xml = bytearray() if path.suffix == ".xml" else None
    require(0 < size <= (8 * 1024**2 if xml is not None else 1024**3), "Asset size exceeds native bound")
    offset = 0
    with path.open("rb") as file:
        require(os.fstat(file.fileno()).st_size == size, "Asset size differs")
        while data := file.read(1024 * 1024):
            digest.update(data)
            if xml is not None:
                xml.extend(data)
            for index, (_, _, start, length, _) in enumerate(weights):
                lo, hi = max(offset, start), min(offset + len(data), start + length)
                if hi > lo:
                    span_hashes[index].update(memoryview(data)[lo - offset:hi - offset])
            offset += len(data)
            require(offset <= size, "Asset grew beyond its pin")
    require(offset == size and digest.hexdigest() == expected, "Actual asset SHA/size differs from trusted pin")
    for row, span_digest in zip(weights, span_hashes):
        require(span_digest.hexdigest() == row[4], "Learned bytes differ from source span pin")
    return ET.fromstring(xml) if xml is not None else None


def constants(root, bin_size):
    result = set()
    widths = {"f16": 2, "f32": 4, "i64": 8, "i32": 4, "boolean": 1, "u8": 1}
    for layer in root.iter("layer"):
        if layer.get("type") != "Const":
            continue
        data = layer.find("data").attrib
        shape = tuple(map(int, data["shape"].split(","))) if data["shape"] else ()
        offset, size = int(data["offset"]), int(data["size"])
        require(data["element_type"] in widths and all(x > 0 for x in shape), "Unknown constant dtype/shape")
        require(size == math.prod(shape) * widths[data["element_type"]] and offset <= bin_size
                and size <= bin_size - offset, "Constant bounds/shape differ")
        result.add((data["element_type"], shape, offset, size))
    return result


def emit(pins, controls, blocks, templates):
    assets = []
    for block in blocks:
        assets.extend(((block["xml"], controls[block["xml"]]), (block["bin"], pins[block["bin"]])))
    assets.extend((xml, pins[xml]) for _, xml, _ in templates)
    names = [Path(path).name for path, _ in assets]
    require(len(assets) == len(set(names)) == 70, "Pack asset names collide")
    lines = ["// Generated by scripts/generate-seedvr-assets.py; reviewed fixed source digests.",
             "#pragma once", '#include "seedvr_assets.hpp"', "", "namespace otc::ai {",
             "struct SeedVrWeightBinding { const char* name; SeedVrWeightSpan span; std::array<std::size_t, 3> dimensions; std::size_t rank; };",
             "struct SeedVrWeightSchema { const SeedVrWeightBinding* weights; std::size_t count; };",
             "struct SeedVrBlockAssets { std::size_t xmlPin, binPin, schema, templateIndex; };",
             "struct SeedVrTemplateAssets { std::size_t representative, xmlPin, binPin, schema; };",
             f'inline constexpr const char* kSeedVrSourceManifestSha = "{SOURCE_SHA}";',
             f'inline constexpr const char* kSeedVrControlManifestSha = "{CONTROL_SHA}";',
             "inline constexpr std::array<std::size_t, 3> kSeedVrAssetSourceGrid{1, 48, 60};",
             "inline constexpr bool kSeedVrFinalTemplateNeedsTextPrune = true;",
             "inline const std::array<SeedVrAssetPin, 70> kSeedVrModelAssetPins{{"]
    for name, (_, (size, sha)) in zip(names, assets):
        digest = ",".join("0x" + sha[i:i + 2] for i in range(0, 64, 2))
        lines.append(f'  {{"{name}", {size}ULL, {{{digest}}}}},')
    lines.append("}};")
    for rep in REPS:
        rows = blocks[rep]["weights"]
        lines.append(f"inline constexpr std::array<SeedVrWeightBinding, {len(rows)}> kSeedVrClass{rep}Weights{{{{")
        for name, dims, offset, size, _ in rows:
            padded = dims + (1,) * (3 - len(dims))
            lines.append(f'  {{"{name}", {{{offset}ULL,{size}ULL,{math.prod(dims)}ULL}}, '
                         f'{{{",".join(map(str, padded))}}}, {len(dims)}}},')
        lines.append("}};")
    lines.append("inline constexpr std::array<SeedVrWeightSchema, 6> kSeedVrWeightSchemas{{")
    lines.extend(f"  {{kSeedVrClass{rep}Weights.data(), kSeedVrClass{rep}Weights.size()}}," for rep in REPS)
    lines.extend(("}};", "inline constexpr std::array<SeedVrBlockAssets, 32> kSeedVrBlockAssets{{"))
    for index in range(32):
        schema = REPS.index(representative(index))
        lines.append(f"  {{{2 * index}, {2 * index + 1}, {schema}, {schema}}},")
    lines.extend(("}};", "inline constexpr std::array<SeedVrTemplateAssets, 6> kSeedVrTemplateAssets{{"))
    lines.extend(f"  {{{rep}, {64 + index}, {2 * rep + 1}, {index}}}," for index, rep in enumerate(REPS))
    lines.extend(("}};", f"inline constexpr std::uint64_t kSeedVrSnapshotBinBytes = {sum(pins[b['bin']][0] for b in blocks)}ULL;",
                  f"inline constexpr std::uint64_t kSeedVrSnapshotXmlBytes = {sum(size for name, (_, (size, _)) in zip(names, assets) if name.endswith('.xml'))}ULL;",
                  "} // namespace otc::ai", ""))
    return "\n".join(lines), assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--models", type=Path, help="Optional flattened pack directory; fixed pins remain authoritative")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True, help="New receipt file, never overwrite")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    require(not args.receipt.exists(), "Receipt must be new")
    source = trusted_text(args.source_manifest, SOURCE_SHA)
    control = trusted_text(args.control_manifest, CONTROL_SHA)
    pins, controls, blocks, templates = parse(source, control)
    output, assets = emit(pins, controls, blocks, templates)
    if args.output.exists():
        require(args.output.stat().st_size == len(output.encode()) and args.output.read_bytes() == output.encode(),
                "Existing generated header differs; refuse overwrite")
    else:
        require(not args.check, "--check requires an existing header")
    source_path = lambda path: args.models / Path(path).name if args.models else Path(path)
    for index, block in enumerate(blocks):
        verify(source_path(block["bin"]), pins[block["bin"]], block["weights"])
        root = verify(source_path(block["xml"]), controls[block["xml"]])
        stored = constants(root, pins[block["bin"]][0])
        for _, dims, offset, size, _ in block["weights"]:
            require(("f16", dims, offset, size) in stored, "Original XML learned mapping differs")
    for rep, xml, binary in templates:
        root = verify(source_path(xml), pins[xml])
        constants(root, pins[binary][0])
        parameters = {layer.get("name"): layer.find("data").attrib
                      for layer in root.findall("./layers/layer") if layer.get("type") == "Parameter"
                      and layer.get("name", "").startswith("weight_")}
        expected = {row[0] for row in blocks[int(rep)]["weights"]}
        require(set(parameters) == expected | ({"weight_008"} if int(rep) == 31 else set()), "Template learned ports differ")
        for name, dims, _, _, _ in blocks[int(rep)]["weights"]:
            require(parameters[name]["element_type"] == "f16" and tuple(map(int, parameters[name]["shape"].split(","))) == dims,
                    "Template parameter dtype/shape differs")
    if not args.output.exists():
        with args.output.open("xb") as file:
            file.write(output.encode())
    receipt = {"status": "pass", "check": args.check, "gpu_calls": 0, "models_compiled": 0,
               "fixed_source_sha256": SOURCE_SHA, "fixed_control_sha256": CONTROL_SHA,
               "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
               "asset_count": 70, "verified_asset_bytes": sum(pin[0] for _, pin in assets),
               "verified_learned_bytes": sum(row[3] for block in blocks for row in block["weights"]),
               "original_spans": 779, "schema_spans": 146, "all32_schema_comparisons_exact": True,
               "schema_members": {str(rep): [i for i in range(32) if representative(i) == rep] for rep in REPS},
               "elapsed_seconds": time.monotonic() - start,
               "output_contains_absolute_paths": "/home/" in output or "trial" in output,
               "assets": [{"name": Path(path).name, "bytes": pin[0], "sha256": pin[1]} for path, pin in assets]}
    with args.receipt.open("x") as file:
        json.dump(receipt, file, indent=2)
        file.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "check", "asset_count", "verified_asset_bytes", "schema_spans", "elapsed_seconds", "output_sha256")}))


if __name__ == "__main__":
    main()
