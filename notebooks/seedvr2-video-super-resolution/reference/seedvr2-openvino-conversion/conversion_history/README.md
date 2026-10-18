# OpenVINO conversion history

This directory is a source-only record of the actual SeedVR2-to-OpenVINO
conversion work that preceded the native worker.  It is research history, not
a shipped model pack: generated XML/BIN, checkpoints, private footage, logs,
receipts, and profiler output are intentionally excluded.

## Upstream source boundary

`../official_source_snapshot/` is the official ByteDance-Seed `SeedVR` source
at revision `e4de8c24441a67e1b7df56abea10645059bb1185`.  It is kept separate
under its original Apache-2.0 license.  The official checkout contains the
real DiT v2, VAE v3, configuration, preprocessing, and inference sources, but
it is CUDA/NCCL/FlashAttention-oriented and does not contain an OpenVINO
exporter.  The scripts below are GPL-3.0-only OneTrackCat research helpers
that adapt the model boundary to OpenVINO; their license is recorded at the
parent `LICENSE-GPL-3.0-only`.

`provenance.json` records every copied file's archive-relative origin and
pre-sanitization SHA-256.  A small number of archived diagnostic strings that
contained a private host path are replaced by portable placeholders in the
copied source; those entries record both hashes and the reason.

## Conversion closure

The selected closure follows the real path rather than a toy exporter:

* `block-shard-openvino-20260918/` validates and exports one real block shard.
* `seedvr-prepost-openvino-20260918/` contains the prelude/postlude audit.
* `dynamic-window-length-openvino-20260918/` contains the dynamic window,
  chunked-attention decomposition, split-IR runners, all block exporters,
  contract tests, and the fixed-envelope runner. The archive has 32
  block-specialized reverse-gather scripts (the shared-weight family and
  final-layer behavior change across the sequence) and no parameterized
  generator; all 32 are therefore preserved with their original hashes so
  the all-block contract remains executable rather than silently reduced to
  three representative blocks.
* `loop-chunk-attention-openvino-20260919/` preserves the loop-chunk builder,
  production graph builder, and CPU contract test.
* `fused-loop-block0-openvino-20260919/` and
  `fused-all-blocks-openvino-20260919/` preserve the fused graph proof and
  all-block dispatcher source.  Their generated artifacts are not included.

The scripts remain explicit about external source, checkpoint, runtime, and
output inputs. The dynamic exporters accept the portable environment overrides
`SEEDVR_SOURCE`, `SEEDVR_WEIGHTS`, and `SEEDVR_OPENVINO_ROOT`; their defaults
point at the sibling `source_snapshot/` and a caller-supplied external weight
or runtime location. They are not a claim of full-width quality or GPU
qualification; reproducibility still requires independently authenticated
weights, the matching Python/PyTorch/OpenVINO versions, and a fresh output
directory.

