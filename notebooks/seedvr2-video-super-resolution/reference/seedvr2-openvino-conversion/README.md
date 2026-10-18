# SeedVR2 OpenVINO conversion research snapshot

This directory preserves the reusable, source-only SeedVR2 conversion probes
that were found in the bounded OneTrackCat trial workspace. It is deliberately
separate from the portable native worker in the parent directory: these files
are a research conversion closure, not a shipping runtime and not a quality
qualification.

## Provenance and licenses

The community model implementation snapshot under `source_snapshot/` is copied from the
Apache-2.0 licensed checkout
`a86582751/ComfyUI-SeedVR2_VideoUpscaler-intel-xpu` at commit
`8d4f3c4bd854695bfcbd25e775b9ea5852ca5f2c`. Its original license is preserved
as `source_snapshot/LICENSE-APACHE-2.0`. The checkout is a community
ComfyUI/PyTorch Intel-XPU fork; it is not an OpenVINO exporter and it is not
claimed to be the original ByteDance release.

`official_source_snapshot/` is a separate Apache-2.0 snapshot of the official
ByteDance-Seed `SeedVR` repository at revision
`e4de8c24441a67e1b7df56abea10645059bb1185`. Its original license and sparse
source manifest are preserved there. It supplies the official DiT v2/VAE v3,
configuration, preprocessing, and inference source used to audit this
boundary. The official checkout is CUDA/NCCL/FlashAttention-oriented and does
not itself contain an OpenVINO exporter.

`conversion_history/` preserves the real GPL-3.0-only OpenVINO conversion
closure recovered from the bounded trial archive: block-shard and pre/post
exporters, chunked/dynamic attention decomposition, representative block
exporters, loop-chunk builders, and fused proof/dispatcher sources.
`optimization_history/` preserves selected source-only T1/T5 contiguous,
residency, and VAE optimization helpers from the later review. Both are
historical research source, not generated model packs or product qualification.

The seven files in `probes/` are GPL-3.0-only research helpers from the
OneTrackCat trial. Their license is preserved as
`LICENSE-GPL-3.0-only`. `SOURCE-MANIFEST.json` records the byte count and
SHA-256 of every copied file, as well as the source commit and probe origin.

No learned checkpoint, `.pt`, `.safetensors`, generated IR/BIN, source
footage, intermediate NPZ, profile, cache, virtual environment, wheel, or
private host path is included. The source tree's ComfyUI entry point and
ComfyUI-only interface modules and example media are intentionally omitted.
The snapshot contains Python/YAML model and preprocessing code only.

## What the closure contains

* `source_snapshot/src/` preserves the source model graph and reusable support
  modules, including the 3B and 7B DiT implementations, the VAE v3 modules,
  temporal/window helpers, configuration, tiling, color correction, and
  memory/offload helpers.
* `source_snapshot/configs_3b/` and `configs_7b/` preserve the matching model
  configuration files. The upstream `pyproject.toml` and requirements file are
  included as provenance, not as a lockfile or a claim that all dependencies
  are bundled.
* `official_source_snapshot/` preserves the official source/configuration
  boundary; it intentionally contains no checkpoints or generated artifacts.
* `conversion_history/` preserves the source-level OpenVINO conversion and
  fused-graph closure. Its `provenance.json` records origin-relative paths and
  original/copied hashes; generated XML/BIN and private evidence remain excluded.
* `optimization_history/` preserves selected late T1/T5/residency/VAE source
  experiments without lifecycle evidence or private paths.
* `probes/openvino-probe.py` builds a bounded CPU structural DiT probe. With an
  externally supplied checkpoint it loads the 3B model, traces/freezes it,
  calls `openvino.convert_model`, and performs CPU shape/finite checks.
* `probes/openvino-vae-probe.py` performs the analogous bounded VAE
  structural probe.
* `probes/openvino-video-probe.py` composes the five-frame crop path: source
  preprocessing, VAE encode/decode, DiT preparation/validation, and a
  finite/shape comparison. `openvino-video-probe-dispatch-test.py` exercises
  its no-model phase dispatch using stubs.
* `canonical-rms.py` is the pure OpenVINO graph-rewrite utility used by the
  RMS experiments. `openvino-vae-roundtrip.py` and its fake-Core test preserve
  the bounded CPU decoder roundtrip contract; they require external encoded
  intermediates and IR when used beyond the test.

The probe scripts use `torch.jit.trace`/`torch.jit.freeze` and
`openvino.convert_model`; they do not bundle exported learned tensors,
checkpoints, or a native weight packer. Conversion requires an independently authenticated
source checkout, matching external `safetensors` checkpoints, input frames,
the external positive/negative conditioning tensors expected by the source,
PyTorch/OpenVINO versions, and enough CPU memory. The scripts' default
reduced-width/no-weight path is useful for structural checks only. The video
probe's qualified research geometry is a five-frame crop (not arbitrary
resolution or long-video streaming), and its output is not a replacement for
the parent native worker's T1/T5 contract.

## Reproduction boundary

Run from this directory with a Python environment containing the packages in
`requirements-conversion.txt` (the upstream dependency list remains in
`source_snapshot/upstream-requirements.txt`). The smallest no-model check is:

```sh
python3 probes/openvino-video-probe-dispatch-test.py
```

A real CPU conversion is intentionally explicit about every external input;
the following is a template, not a bundled fixture:

```sh
python3 probes/openvino-video-probe.py \
  --source source_snapshot \
  --inputs-dir /path/to/authenticated/seedvr2-inputs \
  --vae-weights /path/to/authenticated/ema_vae_fp16.safetensors \
  --dit-weights /path/to/authenticated/seedvr2_ema_3b_fp16.safetensors \
  --output-dir /path/to/fresh/probe-output \
  --phase all --threads 2
```

The output directory must be fresh and must be treated as disposable
research evidence. Do not copy generated IR, weights, footage, or result logs
into this notebook subtree. Authenticate external source/checkpoint hashes
before the run and record the exact Python, PyTorch, OpenVINO, and source
versions beside any private experiment.

## Explicit limitations

The recovered source closes the previously missing *research conversion
closure*, but it does not make a learned production pack. The official source
and GPL conversion helpers still require independently authenticated
checkpoints, external conditioning assets, matching dependency versions, and
fresh generated XML/BIN outputs. No claim is made that the reduced probes or
historical fused scripts are numerically/temporally equivalent to the native
worker, that a GPU path is qualified, or that the community XPU fork is
suitable for redistribution without its terms and notices. Future native
integration must pin the official source revision, conversion helper hashes,
checkpoint hashes, conversion options, generated output hashes, port ABI, and
an independent quality comparison before replacing reviewed structural assets.
