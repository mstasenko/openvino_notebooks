#!/usr/bin/env python3
"""Evidence-only B580 validation for the split SeedVR2 attention graphs.

This runner deliberately has no device discovery or host-coordination logic.  A
caller must provide a fresh JSON marker produced by an independently reviewed
preflight.  The marker is the only admission control used here; the OpenVINO
selector is fixed to ``GPU.1`` and is never a command-line option.

The runner validates one complete production-shape case for block 0, block 16,
and block 31, in that order.  For each block it compiles and infers prefix,
chunk, and suffix one at a time, performs the inverse gather and text
sum/count in the caller, then releases the compiled object before the next
block.  It writes a compact receipt and never prints command lines or argv.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEVICE_ARGUMENT = "GPU.1"
PRODUCTION_SHAPE = (2, 135, 240)
CHUNK_SIZE = 16
BLOCKS = ("block0", "block16", "block31")
TXT_TOKENS = 58
VID_DIM = 2560
EMB_DIM = 15360
HEADS = 20
HEAD_DIM = 128
INNER_DIM = HEADS * HEAD_DIM
ROPE_DIM = 126
PREFLIGHT_MAX_AGE_SECONDS = 60.0
DISTINCT_THRESHOLD = 1.0e-7
DEFAULT_RECEIPT = HERE / "receipt.b580-chunked-window-validation.json"

REQUIRED_PREFLIGHT = {
    "bdf": "0000:03:00.0",
    "driver": "xe",
    "render_node": "renderD128",
    "device": "B580",
}

STAGES = ("prefix", "chunk", "suffix")
ARTIFACT_NAMES = {
    "prefix": ("{block}_chunked_prefix_fp16.xml", "{block}_chunked_prefix_fp16.bin"),
    "chunk": ("{block}_chunked_attention_fp16.xml", "{block}_chunked_attention_fp16.bin"),
    "suffix": ("{block}_chunked_suffix_fp16.xml", "{block}_chunked_suffix_fp16.bin"),
}


class RunnerInputError(ValueError):
    """An invalid caller-supplied marker or runner argument."""


class ContractError(RuntimeError):
    """A local artifact, shape, output, or accounting contract failed."""


class QuietArgumentParser(argparse.ArgumentParser):
    """Do not emit usage, command lines, or argv when rejecting input."""

    def error(self, message: str) -> None:  # pragma: no cover - exercised by main
        raise RunnerInputError("invalid runner arguments")


def _parser() -> argparse.ArgumentParser:
    parser = QuietArgumentParser(add_help=False)
    parser.add_argument("--preflight-marker", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser


def _required_target(values: Mapping[str, Any]) -> dict[str, str]:
    target: Mapping[str, Any]
    nested = values.get("target")
    if nested is None:
        target = values
    elif isinstance(nested, Mapping):
        target = nested
    else:
        raise RunnerInputError("preflight target is not an object")

    bdf = target.get("bdf", target.get("pci_bdf"))
    driver = target.get("driver")
    render_node = target.get("render_node")
    device = target.get("device", target.get("pci_device"))
    if not all(isinstance(value, str) for value in (bdf, driver, render_node, device)):
        raise RunnerInputError("preflight target does not match the B580 contract")
    normalized_render = str(render_node).rstrip("/").rsplit("/", 1)[-1]
    if "device" in target:
        normalized_device = str(device)
    elif "B580" in str(device):
        normalized_device = "B580"
    else:
        normalized_device = ""
    if (
        str(bdf) != REQUIRED_PREFLIGHT["bdf"]
        or str(driver) != REQUIRED_PREFLIGHT["driver"]
        or normalized_render != REQUIRED_PREFLIGHT["render_node"]
        or normalized_device != REQUIRED_PREFLIGHT["device"]
    ):
        raise RunnerInputError("preflight target does not match the B580 contract")
    selector = target.get("openvino_selector", values.get("openvino_selector"))
    if selector is not None and selector != DEVICE_ARGUMENT:
        raise RunnerInputError("preflight selector does not match GPU.1")
    result = dict(REQUIRED_PREFLIGHT)
    status = values.get("status")
    if status is not None and status not in {"pass", "ready"}:
        raise RunnerInputError("preflight marker is not affirmative")
    return result


def parse_preflight_marker(text: str) -> dict[str, str]:
    """Parse and strictly validate the caller-supplied marker JSON.

    The accepted shape is either a top-level object containing the four target
    fields or an object with those fields under ``target``.  The preflight
    receipt spelling ``pci_bdf``/``pci_device`` and a ``/dev/dri/renderD128``
    render-node path are accepted as exact aliases.  Additional fields are
    ignored, but any supplied status must be ``pass`` or ``ready``.
    """

    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise RunnerInputError("preflight marker is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RunnerInputError("preflight marker is not an object")
    return _required_target(payload)


def validate_preflight_marker(
    path: Path,
    *,
    now: float | None = None,
    max_age_seconds: float = PREFLIGHT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Read a marker only after checking that its mtime is fresh."""

    if max_age_seconds <= 0:
        raise RunnerInputError("preflight freshness window is invalid")
    try:
        stat = path.stat()
    except OSError as exc:
        raise RunnerInputError("preflight marker is unavailable") from exc
    if not path.is_file():
        raise RunnerInputError("preflight marker is not a regular file")
    observed_now = time.time() if now is None else float(now)
    age = observed_now - float(stat.st_mtime)
    if age < 0.0 or age > max_age_seconds:
        raise RunnerInputError("preflight marker is stale")
    try:
        fields = parse_preflight_marker(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunnerInputError("preflight marker cannot be read") from exc
    return {
        "marker_name": path.name,
        "age_seconds": round(age, 3),
        "max_age_seconds": float(max_age_seconds),
        "target": fields,
    }


def production_token_count(shape: Sequence[int] = PRODUCTION_SHAPE) -> int:
    """Return the source-token count, accepting only the full production shape."""

    normalized = tuple(int(value) for value in shape)
    if normalized != PRODUCTION_SHAPE:
        raise ContractError("shape is not the required production shape")
    return math.prod(normalized)


def chunk_ranges(window_count: int, chunk_size: int = CHUNK_SIZE) -> tuple[tuple[int, int], ...]:
    """Return contiguous, non-overlapping [start, end) window ranges."""

    if isinstance(window_count, bool) or isinstance(chunk_size, bool):
        raise ContractError("window and chunk counts must be integers")
    if window_count <= 0 or chunk_size <= 0:
        raise ContractError("window and chunk counts must be positive")
    ranges = tuple(
        (start, min(window_count, start + chunk_size))
        for start in range(0, window_count, chunk_size)
    )
    if not ranges or ranges[-1][1] != window_count:
        raise ContractError("chunk ranges do not end at the window count")
    return ranges


def validate_shape_chunk_accounting(
    *,
    shape: Sequence[int],
    window_count: int,
    max_video_len: int,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, Any]:
    """Pure contract math used before any OpenVINO model is compiled."""

    source_tokens = production_token_count(shape)
    if window_count <= 0 or max_video_len <= 0:
        raise ContractError("window metadata dimensions must be positive")
    ranges = chunk_ranges(window_count, chunk_size)
    if any(end <= start or end - start > chunk_size for start, end in ranges):
        raise ContractError("chunk range exceeds the requested chunk size")
    return {
        "shape": list(PRODUCTION_SHAPE),
        "source_tokens": source_tokens,
        "window_count": int(window_count),
        "max_video_len": int(max_video_len),
        "max_attention_len": int(max_video_len) + TXT_TOKENS,
        "chunk_size": int(chunk_size),
        "chunk_count": len(ranges),
        "ranges": [list(item) for item in ranges],
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError("local helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(8 * 1024 * 1024)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def _artifact_inventory(block: str) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for stage in STAGES:
        xml_path = HERE / ARTIFACT_NAMES[stage][0].format(block=block)
        bin_path = HERE / ARTIFACT_NAMES[stage][1].format(block=block)
        entries = []
        for path in (xml_path, bin_path):
            if not path.is_file():
                raise ContractError("required split IR artifact is unavailable")
            stat = path.stat()
            entries.append(
                {
                    "name": path.name,
                    "bytes": int(stat.st_size),
                    "sha256": _sha256(path),
                }
            )
        inventory[stage] = entries
    return inventory


def _all_artifacts() -> dict[str, Any]:
    return {block: _artifact_inventory(block) for block in BLOCKS}


def _peak_rss() -> dict[str, int]:
    # Linux reports ru_maxrss in KiB; this runner is intentionally Linux-only
    # because its admission marker names the xe DRM render node.
    kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"peak_rss_kib": kib, "peak_rss_bytes": kib * 1024}


def _error_record(stage: str, exc: BaseException) -> dict[str, str]:
    """Return fixed, sanitized error text without paths, commands, or argv."""

    kind = type(exc).__name__
    code = {
        "RunnerInputError": "invalid_input",
        "ContractError": "contract_failure",
        "AssertionError": "assertion_failure",
        "FileNotFoundError": "artifact_unavailable",
        "PermissionError": "permission_denied",
        "ValueError": "value_error",
        "RuntimeError": "runtime_failure",
    }.get(kind, "unexpected_failure")
    return {
        "stage": stage,
        "type": kind,
        "code": code,
        "message": "sanitized runner error; see stage and type",
    }


def _to_numpy(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _model_inputs(model: Any) -> dict[str, Any]:
    return {port.get_any_name(): port for port in model.inputs}


def _reshape_model(ov: Any, model: Any, expected_shapes: Mapping[str, Sequence[int]]) -> None:
    actual = _model_inputs(model)
    if set(actual) != set(expected_shapes):
        raise ContractError("split IR input names do not match the runner contract")
    model.reshape({name: ov.PartialShape(list(shape)) for name, shape in expected_shapes.items()})


def _compile(
    ov: Any,
    core: Any,
    model_path: Path,
    expected_shapes: Mapping[str, Sequence[int]],
) -> tuple[Any, Any, float, dict[str, Any]]:
    model = core.read_model(str(model_path))
    _reshape_model(ov, model, expected_shapes)
    started = time.perf_counter()
    compiled = core.compile_model(
        model,
        DEVICE_ARGUMENT,
        {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": ov.Type.f16,
            "NUM_STREAMS": "1",
        },
    )
    elapsed = time.perf_counter() - started
    query_name = "EXECUTION_DEVICES"
    execution_devices = [str(value) for value in compiled.get_property(query_name)]
    query = {"property": query_name, "value": execution_devices}
    if execution_devices != [DEVICE_ARGUMENT]:
        raise ContractError("compiled model execution device does not match GPU.1")
    return model, compiled, elapsed, query


def _infer(compiled: Any, inputs: Mapping[str, Any], np: Any) -> tuple[dict[str, Any], float]:
    request = compiled.create_infer_request()
    started = time.perf_counter()
    result = request.infer(dict(inputs))
    elapsed = time.perf_counter() - started
    outputs = {
        port.get_any_name(): np.array(result[port], copy=True)
        for port in compiled.outputs
    }
    del request
    return outputs, elapsed


def _check_outputs(
    outputs: Mapping[str, Any],
    expected_shapes: Mapping[str, Sequence[int]],
    np: Any,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, shape in expected_shapes.items():
        if name not in outputs:
            raise ContractError("compiled model output name is missing")
        value = outputs[name]
        expected = tuple(int(item) for item in shape)
        if tuple(value.shape) != expected:
            raise ContractError("compiled model output shape is unexpected")
        finite = bool(np.isfinite(value).all())
        if not finite:
            raise ContractError("compiled model output is non-finite")
        checks[name] = {"shape": list(value.shape), "finite": finite}
    return checks


def _contract_arrays(contract: Mapping[str, Any], np: Any) -> dict[str, Any]:
    index = _to_numpy(contract["window_idx"], np).astype(np.int64, copy=False)
    mask = _to_numpy(contract["window_key_mask"], np).astype(bool, copy=False)
    freqs = _to_numpy(contract["window_rope_freqs"], np).astype(np.float32, copy=False)
    reverse = _to_numpy(contract["window_reverse_idx"], np).astype(np.int64, copy=False)
    window_count = int(contract["window_count"])
    max_video_len = int(contract["max_video_len"])
    accounting = validate_shape_chunk_accounting(
        shape=PRODUCTION_SHAPE,
        window_count=window_count,
        max_video_len=max_video_len,
        chunk_size=CHUNK_SIZE,
    )
    source_tokens = accounting["source_tokens"]
    if tuple(index.shape) != (window_count, max_video_len):
        raise ContractError("window index shape is unexpected")
    if tuple(mask.shape) != (window_count, max_video_len + TXT_TOKENS):
        raise ContractError("window mask shape is unexpected")
    if tuple(freqs.shape) != (window_count, max_video_len + TXT_TOKENS, ROPE_DIM):
        raise ContractError("window RoPE shape is unexpected")
    if tuple(reverse.shape) != (source_tokens,):
        raise ContractError("window reverse index shape is unexpected")

    video_mask = mask[:, :max_video_len]
    valid_source = index[:, :max_video_len][video_mask]
    if valid_source.size != source_tokens:
        raise ContractError("window metadata does not cover every source token")
    occurrences = np.bincount(valid_source, minlength=source_tokens)
    if occurrences.size != source_tokens or not np.all(occurrences == 1):
        raise ContractError("window metadata has duplicate or missing source tokens")
    if np.unique(reverse).size != source_tokens or np.any(reverse < 0):
        raise ContractError("window reverse map is not full and unique")
    if np.any(reverse >= window_count * max_video_len):
        raise ContractError("window reverse map selects outside padded windows")
    if not np.all(video_mask.reshape(-1)[reverse]):
        raise ContractError("window reverse map selects padding")
    return {
        "index": index,
        "mask": mask,
        "freqs": freqs,
        "reverse": reverse,
        "accounting": accounting,
        "coverage": {
            "source_tokens": int(source_tokens),
            "valid_tokens": int(valid_source.size),
            "unique_tokens": int(np.unique(valid_source).size),
            "occurrence_min": int(occurrences.min()),
            "occurrence_max": int(occurrences.max()),
            "reverse_unique": True,
        },
    }


def _window_metadata(helper: Any, decomposition: Any, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = decomposition.window_metadata(helper, PRODUCTION_SHAPE)
    if str(helper.SOURCE) not in sys.path:
        sys.path.insert(0, str(helper.SOURCE))
    from src.models.dit_3b.rope import get_na_rope

    rope = get_na_rope("mmrope3d", helper.HEAD_DIM)
    contract = helper.make_window_contract(
        PRODUCTION_SHAPE,
        max_video_len=int(summary["max_video_len"]),
        txt_len=helper.TXT_TOKENS,
        rope=rope,
        torch=torch,
    )
    if int(contract["window_count"]) != int(summary["window_count"]):
        raise ContractError("trusted helpers disagree on window count")
    if int(contract["max_video_len"]) != int(summary["max_video_len"]):
        raise ContractError("trusted helpers disagree on maximum window length")
    return contract, {
        "window_count": int(summary["window_count"]),
        "max_video_len": int(summary["max_video_len"]),
        "window_lengths": [int(value) for value in summary["window_lengths"]],
    }


def _random_inputs(source_tokens: int, seed: int, np: Any) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    def normal(shape: Sequence[int]) -> Any:
        return (rng.standard_normal(tuple(shape)).astype(np.float32) * np.float32(0.02))

    return {
        "vid": normal((source_tokens, VID_DIM)),
        "txt": normal((TXT_TOKENS, VID_DIM)),
        "emb": normal((1, EMB_DIM)),
    }


def _stage_path(block: str, stage: str) -> Path:
    return HERE / ARTIFACT_NAMES[stage][0].format(block=block)


def _run_block(
    block: str,
    *,
    artifacts: Mapping[str, Any],
    ov: Any,
    np: Any,
    torch: Any,
    decomposition: Any,
) -> dict[str, Any]:
    suffix = block.removeprefix("block")
    helper_path = HERE / f"dynamic_window_reverse_gather_block{suffix}_export.py"
    helper = _load_module(helper_path, f"dynamic_window_helper_{suffix}")
    contract, window_summary = _window_metadata(helper, decomposition, torch)
    arrays = _contract_arrays(contract, np)
    accounting = arrays["accounting"]
    source_tokens = int(accounting["source_tokens"])
    values = _random_inputs(source_tokens, 2026091800 + int(suffix), np)
    core = ov.Core()
    block_receipt: dict[str, Any] = {
        "block": block,
        "shape": list(PRODUCTION_SHAPE),
        "chunk_size": CHUNK_SIZE,
        "window": window_summary,
        "accounting": accounting,
        "coverage": arrays["coverage"],
        "artifacts": dict(artifacts),
        "stages": {},
    }
    prefix_outputs: dict[str, Any]
    try:
        prefix_shapes = {
            "vid": (source_tokens, VID_DIM),
            "txt": (TXT_TOKENS, VID_DIM),
            "emb": (1, EMB_DIM),
        }
        prefix_model, prefix_compiled, compile_seconds, execution_query = _compile(
            ov, core, _stage_path(block, "prefix"), prefix_shapes
        )
        try:
            prefix_outputs, infer_seconds = _infer(prefix_compiled, values, np)
            checks = _check_outputs(
                prefix_outputs,
                {
                    "vid_q": (source_tokens, HEADS, HEAD_DIM),
                    "vid_k": (source_tokens, HEADS, HEAD_DIM),
                    "vid_v": (source_tokens, HEADS, HEAD_DIM),
                    "txt_q": (TXT_TOKENS, HEADS, HEAD_DIM),
                    "txt_k": (TXT_TOKENS, HEADS, HEAD_DIM),
                    "txt_v": (TXT_TOKENS, HEADS, HEAD_DIM),
                },
                np,
            )
            block_receipt["stages"]["prefix"] = {
                "compile_seconds": compile_seconds,
                "infer_seconds": infer_seconds,
                "execution_device_query": execution_query,
                "outputs": checks,
                **_peak_rss(),
            }
        finally:
            del prefix_compiled, prefix_model
            gc.collect()

        chunk_shapes = {
            "vid_q": (source_tokens, HEADS, HEAD_DIM),
            "vid_k": (source_tokens, HEADS, HEAD_DIM),
            "vid_v": (source_tokens, HEADS, HEAD_DIM),
            "txt_q": (TXT_TOKENS, HEADS, HEAD_DIM),
            "txt_k": (TXT_TOKENS, HEADS, HEAD_DIM),
            "txt_v": (TXT_TOKENS, HEADS, HEAD_DIM),
            # Keep the window batch dynamic so the final partial chunk is
            # represented exactly; the per-call input is still capped at 16.
            "window_idx": (-1, int(accounting["max_video_len"])),
            "window_key_mask": (-1, int(accounting["max_attention_len"])),
            "window_rope_freqs": (-1, int(accounting["max_attention_len"]), ROPE_DIM),
        }
        chunk_model, chunk_compiled, compile_seconds, execution_query = _compile(
            ov, core, _stage_path(block, "chunk"), chunk_shapes
        )
        try:
            vid_acc = np.zeros((source_tokens, INNER_DIM), dtype=np.float32)
            txt_sum_acc = np.zeros((TXT_TOKENS, INNER_DIM), dtype=np.float32)
            gather_counts = np.zeros(source_tokens, dtype=np.uint8)
            text_count = 0.0
            chunk_events = []
            for start, end in chunk_ranges(int(accounting["window_count"]), CHUNK_SIZE):
                chunk_windows = end - start
                idx = arrays["index"][start:end]
                mask = arrays["mask"][start:end]
                freqs = arrays["freqs"][start:end]
                # Prefix outputs are copied before their compiled model is released.
                chunk_inputs = {
                    "vid_q": prefix_outputs["vid_q"],
                    "vid_k": prefix_outputs["vid_k"],
                    "vid_v": prefix_outputs["vid_v"],
                    "txt_q": prefix_outputs["txt_q"],
                    "txt_k": prefix_outputs["txt_k"],
                    "txt_v": prefix_outputs["txt_v"],
                    "window_idx": idx,
                    "window_key_mask": mask,
                    "window_rope_freqs": freqs,
                }
                chunk_inputs_shapes = {
                    "vid_q": (source_tokens, HEADS, HEAD_DIM),
                    "vid_k": (source_tokens, HEADS, HEAD_DIM),
                    "vid_v": (source_tokens, HEADS, HEAD_DIM),
                    "txt_q": (TXT_TOKENS, HEADS, HEAD_DIM),
                    "txt_k": (TXT_TOKENS, HEADS, HEAD_DIM),
                    "txt_v": (TXT_TOKENS, HEADS, HEAD_DIM),
                    "window_idx": (chunk_windows, int(accounting["max_video_len"])),
                    "window_key_mask": (chunk_windows, int(accounting["max_attention_len"])),
                    "window_rope_freqs": (chunk_windows, int(accounting["max_attention_len"]), ROPE_DIM),
                }
                for name, shape in chunk_inputs_shapes.items():
                    if tuple(chunk_inputs[name].shape) != tuple(shape):
                        raise ContractError("chunk input shape is unexpected")
                chunk_outputs, infer_seconds = _infer(chunk_compiled, chunk_inputs, np)
                checks = _check_outputs(
                    chunk_outputs,
                    {
                        "vid_window_out": (chunk_windows, int(accounting["max_video_len"]), INNER_DIM),
                        "txt_sum": (TXT_TOKENS, INNER_DIM),
                        "txt_count": (1,),
                    },
                    np,
                )
                local_reverse = arrays["reverse"] - start * int(accounting["max_video_len"])
                source_idx = idx[:, : int(accounting["max_video_len"])][
                    mask[:, : int(accounting["max_video_len"])]
                ]
                selected_positions = local_reverse[source_idx]
                if np.any(selected_positions < 0) or np.any(
                    selected_positions >= chunk_windows * int(accounting["max_video_len"])
                ):
                    raise ContractError("inverse gather selected outside the active chunk")
                if np.unique(source_idx).size != source_idx.size:
                    raise ContractError("chunk inverse gather saw duplicate source tokens")
                if np.any(gather_counts[source_idx] != 0):
                    raise ContractError("chunk inverse gather repeated a source token")
                flat_video = chunk_outputs["vid_window_out"].reshape(-1, INNER_DIM)
                vid_acc[source_idx] = flat_video[selected_positions]
                gather_counts[source_idx] = 1
                txt_sum_acc += chunk_outputs["txt_sum"]
                count = float(chunk_outputs["txt_count"].reshape(-1)[0])
                if not math.isfinite(count) or count <= 0.0:
                    raise ContractError("chunk text count is not finite and positive")
                text_count += count
                chunk_events.append(
                    {
                        "start": start,
                        "end": end,
                        "windows": chunk_windows,
                        "infer_seconds": infer_seconds,
                        "outputs": checks,
                    }
                )
            if abs(text_count - float(accounting["window_count"])) > 1.0e-6:
                raise ContractError("text count does not equal window count")
            if not bool(np.all(gather_counts == 1)):
                raise ContractError("inverse gather did not cover every source token exactly once")
            if not bool(np.isfinite(vid_acc).all()) or not bool(np.isfinite(txt_sum_acc).all()):
                raise ContractError("inverse-gather or text-sum output is non-finite")
            attn_txt = txt_sum_acc / np.float32(text_count)
            if not bool(np.isfinite(attn_txt).all()):
                raise ContractError("text average is non-finite")
            block_receipt["stages"]["chunk"] = {
                "compile_seconds": compile_seconds,
                "infer_seconds": float(sum(item["infer_seconds"] for item in chunk_events)),
                "execution_device_query": execution_query,
                "chunk_events": chunk_events,
                "text_count": text_count,
                "text_count_equals_window_count": True,
                "inverse_gather": {
                    "source_tokens": source_tokens,
                    "unique_source_tokens": int(np.count_nonzero(gather_counts)),
                    "full_unique_coverage": True,
                },
                **_peak_rss(),
            }
        finally:
            del chunk_compiled, chunk_model
            gc.collect()

        suffix_shapes = {
            "vid": (source_tokens, VID_DIM),
            "txt": (TXT_TOKENS, VID_DIM),
            "emb": (1, EMB_DIM),
            "attn_vid": (source_tokens, INNER_DIM),
            "attn_txt": (TXT_TOKENS, INNER_DIM),
        }
        suffix_model, suffix_compiled, compile_seconds, execution_query = _compile(
            ov, core, _stage_path(block, "suffix"), suffix_shapes
        )
        try:
            suffix_inputs = {
                "vid": values["vid"],
                "txt": values["txt"],
                "emb": values["emb"],
                "attn_vid": vid_acc,
                "attn_txt": attn_txt,
            }
            suffix_outputs, infer_seconds = _infer(suffix_compiled, suffix_inputs, np)
            checks = _check_outputs(
                suffix_outputs,
                {
                    "vid_out": (source_tokens, VID_DIM),
                    "txt_out": (TXT_TOKENS, VID_DIM),
                },
                np,
            )
            vid_delta = float(np.max(np.abs(suffix_outputs["vid_out"].astype(np.float64) - values["vid"].astype(np.float64))))
            txt_delta = float(np.max(np.abs(suffix_outputs["txt_out"].astype(np.float64) - values["txt"].astype(np.float64))))
            if vid_delta <= DISTINCT_THRESHOLD or txt_delta <= DISTINCT_THRESHOLD:
                raise ContractError("suffix outputs do not differ from inputs")
            block_receipt["stages"]["suffix"] = {
                "compile_seconds": compile_seconds,
                "infer_seconds": infer_seconds,
                "execution_device_query": execution_query,
                "outputs": checks,
                "output_input_max_abs_delta": {"vid": vid_delta, "txt": txt_delta},
                "outputs_differ_from_inputs": True,
                **_peak_rss(),
            }
        finally:
            del suffix_compiled, suffix_model
            gc.collect()
        block_receipt["status"] = "pass"
        block_receipt.update(_peak_rss())
        return block_receipt
    finally:
        del core
        gc.collect()


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    receipt_path = DEFAULT_RECEIPT
    receipt: dict[str, Any] = {
        "status": "not_started",
        "runner": "b580_chunked_window_validation_runner",
        "device_argument": DEVICE_ARGUMENT,
        "shape": list(PRODUCTION_SHAPE),
        "chunk_size": CHUNK_SIZE,
        "blocks": list(BLOCKS),
        "errors": [],
    }
    try:
        args = _parser().parse_args(argv)
        receipt_path = args.receipt
        if receipt_path.exists():
            raise RunnerInputError("refusing to overwrite an existing receipt")
        receipt["preflight"] = validate_preflight_marker(args.preflight_marker)
        # Inventory every block before importing OpenVINO.  A missing block-31
        # IR therefore fails closed before any device plugin can be initialized.
        artifacts = _all_artifacts()
        receipt["artifact_inventory"] = artifacts
        import numpy as np
        import openvino as ov
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        decomposition = _load_module(
            HERE / "chunked_attention_decomposition_export.py",
            "chunked_attention_decomposition_for_b580_runner",
        )
        receipt["block_receipts"] = []
        for block in BLOCKS:
            block_receipt = _run_block(
                block,
                artifacts=artifacts[block],
                ov=ov,
                np=np,
                torch=torch,
                decomposition=decomposition,
            )
            receipt["block_receipts"].append(block_receipt)
        receipt["status"] = "pass"
        receipt.update(_peak_rss())
        _write_receipt(receipt_path, receipt)
        print(json.dumps({"status": "pass", "receipt": receipt_path.name}, sort_keys=True))
        return 0
    except RunnerInputError as exc:
        receipt["status"] = "blocked"
        receipt["errors"].append(_error_record("preflight", exc))
    except Exception as exc:  # sanitized into the receipt; no traceback/argv is printed
        receipt["status"] = "failure"
        receipt["errors"].append(_error_record("runner", exc))
    receipt.update(_peak_rss())
    try:
        _write_receipt(receipt_path, receipt)
    except Exception:
        # The caller still gets a fixed status line; the original sanitized
        # receipt error remains the only information emitted by this runner.
        pass
    print(json.dumps({"status": receipt["status"], "receipt": receipt_path.name}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
