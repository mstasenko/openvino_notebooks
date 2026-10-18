"""Authenticated real-video boundaries shared by finite CUDA/OV replay tests."""
import hashlib
import json
from pathlib import Path

import numpy as np

from reference_paths import EVIDENCE_ROOT, ROOT as REFERENCE_ROOT

BASE = REFERENCE_ROOT
ROOT = EVIDENCE_ROOT / "true-upscale-1280x768-attempt1"
REPORT_SHA = "057aa41bb76bc550160a947a126195b7c15c043a29b58fde28d27d74655db08a"
WIDTH, HEIGHT, SPATIAL_WINDOWS = 1280, 768, 60
FIELDS = {"x", "t", "t_mod", "lq_0", "prediction"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs():
    report_path = ROOT / "report.json"
    if sha(report_path) != REPORT_SHA:
        raise ValueError("Real-video producer receipt changed")
    report = json.loads(report_path.read_text())
    if report["status"] != "complete" or report["geometry"]["working"] != [WIDTH, HEIGHT]:
        raise ValueError("Completed actual-video geometry required")
    records = report["model_captures"]
    if len(records) != 4 or [record["chunk"] for record in records] != list(range(4)):
        raise ValueError("Exact four streaming chunks required")
    fixtures = []
    for call, record in enumerate(records):
        path = ROOT / f"model-chunk{call}.npz"
        if Path(record["path"]).resolve() != path.resolve() or sha(path) != record["sha256"]:
            raise ValueError("Actual model input capture changed")
        frames = 6 if call == 0 else 2
        shapes = {"x": (1,16,frames,96,160), "t": (1,1536), "t_mod": (1,6,1536),
                  "lq_0": (1,frames*3840,1536), "prediction": (1,16,frames,96,160)}
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != FIELDS:
                raise ValueError("Actual input capture fields changed")
            values = {}
            for key in FIELDS:
                value = archive[key]
                if value.shape != shapes[key] or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"Invalid actual input ABI: {call}/{key}")
                values[key] = np.ascontiguousarray(value.astype(np.float16))
                if not np.isfinite(values[key]).all():
                    raise ValueError("Actual input F16 conversion overflow")
        fixtures.append(values)
    return fixtures, {
        "report_sha256": REPORT_SHA, "archive_sha256": [record["sha256"] for record in records],
        "working_geometry": [WIDTH,HEIGHT], "calls": 4,
        "boundary_dtype": "saved BF16-origin F32 arrays -> F16 for both replay backends",
        "scope": "actual-video denoiser replay; not an end-to-end FP16 pipeline or video speed test",
        "input_frames": 49, "reference_output_frames": 45,
    }
