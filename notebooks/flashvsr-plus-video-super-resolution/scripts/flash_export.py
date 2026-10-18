"""Small shared utilities for build-time Flash structural exporters.

Graph construction and role-specific ABI contracts remain in each exporter
so a PRE, LQ or cross-attention export
cannot silently accept the other graph's ports.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, Iterable, Mapping


# Match the native LQ structural validator: each literal is tiny and the
# complete graph carries at most one bounded constants budget.  This also
# keeps a learned tensor from being smuggled in as one large unnamed literal.
MAX_CONSTANT_BYTES = 4096
MAX_TOTAL_CONSTANT_BYTES = 65536
_TYPE_BYTES = {
    "boolean": 1, "bool": 1,
    "u8": 1, "i8": 1,
    "u16": 2, "i16": 2, "f16": 2, "bf16": 2,
    "u32": 4, "i32": 4, "f32": 4,
    "u64": 8, "i64": 8, "f64": 8,
    "uint8": 1, "int8": 1, "uint16": 2, "int16": 2,
    "uint32": 4, "int32": 4, "uint64": 8, "int64": 8,
    "float16": 2, "float32": 4, "float64": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def helper_fingerprint() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"name": path.name, "sha256": sha256_file(path)}


def serialize_structural_ir(ov: Any, model: Any, output: Path,
                            stem: str) -> dict[str, dict[str, Any]]:
    """Write portable IR and reject public-port changes introduced by its reader."""
    if not stem or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in stem):
        raise ValueError("structural artifact stem must be a plain lowercase name")
    paths = {kind: output / f"{stem}.{kind}" for kind in ("xml", "bin")}
    expected_ports = {
        kind: [canonical_port_record(port) for port in getattr(model, kind)]
        for kind in ("inputs", "outputs")
    }
    ov.serialize(model, str(paths["xml"]), str(paths["bin"]))
    # A valid in-memory graph may still have input/output name collisions that
    # the IR reader silently renames. Check the consumer ABI before publishing.
    frontend = ov.frontend.FrontEndManager().load_by_framework("ir")
    if frontend is None:
        raise ValueError("OpenVINO IR frontend is unavailable")
    reloaded = frontend.convert(frontend.load(str(paths["xml"])))
    for kind, expected in expected_ports.items():
        actual = [canonical_port_record(port) for port in getattr(reloaded, kind)]
        if actual != expected:
            raise ValueError(f"serialized {stem} {kind} ABI changed during IR readback")
    records = {}
    for kind, path in paths.items():
        assert_no_private_paths(path, path.read_bytes())
        records[kind] = {"path": path.name, "bytes": path.stat().st_size,
                         "sha256": sha256_file(path)}
    return records


def graph_source_records(root: Path, required_files: Iterable[str]) -> list[dict[str, Any]]:
    records = [source_record(root, path)
               for path in loaded_source_paths(root, under=root / "openvino")]
    if not {Path(item).as_posix() for item in required_files}.issubset(
            {record["path"] for record in records}):
        raise ValueError("structural graph source provenance is incomplete")
    return records


def write_structural_manifest(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = {**manifest, "export_helper": helper_fingerprint()}
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path = output / "manifest.json"
    assert_no_private_paths(path, encoded)
    path.write_bytes(encoded)
    return manifest


def _reject_symlink_components(path: Path, root: Path | None = None) -> None:
    cursor = Path(path)
    if cursor.is_symlink():
        raise ValueError(f"symlink is not accepted: {path}")
    stop = root.resolve() if root is not None else None
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError(f"symlink component is not accepted: {path}")
        if stop is not None and cursor == stop:
            break
        cursor = cursor.parent


def checked_reference_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"reference file is not relative: {relative}")
    path = Path(root) / rel
    _reject_symlink_components(path, root)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"required reference file is missing: {relative}") from error
    try:
        resolved.relative_to(Path(root).resolve())
    except ValueError as error:
        raise ValueError(f"reference file escapes root: {relative}") from error
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"reference file is not regular: {relative}")
    return resolved


def checked_reference_root(value: Path, required_files: Iterable[str]) -> Path:
    _reject_symlink_components(value)
    try:
        root = Path(value).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"reference root does not exist: {value}") from error
    if not root.is_dir():
        raise ValueError(f"reference root is not a directory: {value}")
    for relative in required_files:
        checked_reference_file(root, relative)
    return root


def fresh_output(value: Path) -> Path:
    path = Path(value)
    if path.exists() or path.is_symlink():
        raise ValueError(f"output must be a fresh directory: {path}")
    # Check every existing parent before mkdir can follow a caller-provided
    # symlink.  The output leaf is deliberately private because it will hold
    # authenticated manifests and generated build-time tables.
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path.resolve()


@contextmanager
def reference_import_path(root: Path):
    old_path = sys.path[:]
    sys.path[:0] = [
        str(root), str(root / "openvino" / "full_runtime"),
        str(root / "openvino"), str(root / "openvino" / "chunked_attention"),
    ]
    try:
        yield
    finally:
        sys.path[:] = old_path


def require_reference_modules(root: Path, modules: Mapping[str, str]) -> None:
    """Require each imported top-level module to be the exact expected file."""

    root = Path(root).resolve()
    for name, relative in modules.items():
        expected = checked_reference_file(root, relative)
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None)
        if not filename:
            raise ValueError(f"reference module was not loaded: {name}")
        try:
            actual = Path(filename).resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise ValueError(f"reference module source is unavailable: {name}") from error
        if actual != expected:
            raise ValueError(
                f"reference module escaped its exact source pin: {name} ({actual})"
            )


def loaded_source_paths(root: Path, *, under: Path | None = None) -> list[Path]:
    root = Path(root).resolve()
    boundary = root if under is None else Path(under).resolve()
    paths = set()
    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        filename = getattr(module, "__file__", None)
        if not filename or not str(filename).endswith(".py"):
            continue
        path = Path(filename).resolve()
        try:
            path.relative_to(boundary)
        except ValueError:
            continue
        if path.is_file():
            paths.add(path)
    return sorted(paths)


def canonical_port_record(port: Any, *, include_names: bool = False) -> dict[str, Any]:
    names = sorted(str(name) for name in port.get_names())
    if len(names) != 1 or str(port.get_any_name()) != names[0]:
        raise ValueError("every ABI port requires exactly one canonical tensor name")
    try:
        shape = [int(value) for value in port.get_shape()]
    except Exception as error:
        raise ValueError("structural export requires static port shapes") from error
    element_type = port.get_element_type()
    to_string = getattr(element_type, "to_string", None)
    record = {
        "name": names[0], "shape": shape,
        "element_type": str(to_string() if callable(to_string) else element_type),
    }
    if include_names:
        record["names"] = names
    return record


def check_port_records(records: list[dict[str, Any]],
                       contract: dict[str, tuple[list[int], str]], role: str) -> dict[str, Any]:
    names = [record["name"] for record in records]
    if len(names) != len(set(names)) or set(names) != set(contract):
        raise ValueError(f"{role} ABI names changed")
    by_name = {record["name"]: record for record in records}
    for name, (shape, element_type) in contract.items():
        record = by_name[name]
        if record["shape"] != shape or record["element_type"] != element_type:
            raise ValueError(f"{role} ABI changed: {name}")
    return by_name


def require_runtime_version(version: str, expected: str) -> None:
    if version != expected:
        raise ValueError(f"pinned OpenVINO {expected} required, got {version}")


def reject_f16_intermediates(model: Any, role: str) -> None:
    for node in model.get_ordered_ops():
        for output in node.outputs():
            element_type = output.get_element_type()
            to_string = getattr(element_type, "to_string", None)
            name = str(to_string() if callable(to_string) else element_type).lower()
            if name in {"f16", "float16"}:
                raise ValueError(f"{role} graph contains an FP16 intermediate")


def source_record(root: Path, path: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    resolved = Path(path).resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"loaded graph source escaped reference root: {path}") from error
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"loaded graph source is not regular: {path}")
    return {"path": relative.as_posix(), "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved)}


def assert_no_private_paths(path: Path, data: bytes) -> None:
    forbidden = (b"/home/", b"/cache/", b"/tmp/", b"/var/tmp/", b"C:\\Users\\")
    if any(token in data for token in forbidden):
        raise ValueError(f"generated artifact contains an absolute private path: {path.name}")


def _element_bytes(element_type: Any) -> int:
    to_string = getattr(element_type, "to_string", None)
    value = str(to_string() if callable(to_string) else element_type).lower()
    if value not in _TYPE_BYTES:
        raise ValueError(f"constant has an unknown element type: {value}")
    return _TYPE_BYTES[value]


def validate_constant_nodes(model: Any, *, forbidden_prefixes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Bound literals as accidental-embedding guardrails.

    The explicit reference root is trusted executable input; size limits and
    name prefixes do not prove that an arbitrary graph literal is not learned
    data.  External-weight graph contracts remain the actual ABI boundary.
    """

    names: list[str] = []
    total_bytes = 0
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Constant":
            continue
        friendly = str(node.get_friendly_name())
        output_size = getattr(node, "get_output_size", None)
        if not callable(output_size):
            raise ValueError(f"Constant has no inspectable outputs: {friendly}")
        try:
            output_count = output_size()
            if isinstance(output_count, bool) or not isinstance(output_count, int):
                raise TypeError("output count is not an integer")
        except Exception as error:
            raise ValueError(f"Constant output count is invalid: {friendly}") from error
        if output_count <= 0:
            raise ValueError(f"Constant has no outputs: {friendly}")
        node_bytes = 0
        output_names: set[str] = set()
        for index in range(output_count):
            output = node.output(index)
            output_names.update(str(name) for name in output.get_names())
            try:
                shape = [int(value) for value in output.get_shape()]
            except Exception as error:
                raise ValueError("structural Constant shape must be static") from error
            if any(value < 0 for value in shape):
                raise ValueError(f"structural Constant shape is invalid: {friendly}")
            elements = math.prod(shape) if shape else 1
            node_bytes += elements * _element_bytes(output.get_element_type())
        if any(name.startswith(prefix) for name in output_names | {friendly}
               for prefix in forbidden_prefixes):
            raise ValueError(f"learned value became a Constant: {friendly}")
        if node_bytes > MAX_CONSTANT_BYTES:
            raise ValueError(f"structural Constant is too large: {friendly}")
        if total_bytes > MAX_TOTAL_CONSTANT_BYTES - node_bytes:
            raise ValueError("structural Constant budget exceeded")
        names.append(friendly)
        total_bytes += node_bytes
    return {"constant_node_count": len(names), "constant_names": sorted(names),
            "constant_bytes": total_bytes}
