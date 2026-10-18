# FlashVSR+ OpenVINO

This directory is a portable, research-only extraction of the FlashVSR+ source
closure used by the qualified local OpenVINO experiments.  It is a reusable
reference for studying the model and its bounded-memory execution; it is not a
drop-in OTC product backend and it does not claim a production speed target.

The `reference/` tree contains Python/OpenVINO graph builders, the exact
BF16-storage and window-attention helpers, the phased decoder/history helpers,
and the small parameter-layer closure needed to rebuild the graphs.  It does
not contain model weights, videos, generated NPZ/NPY files, private caches,
native binaries, or a patched OpenVINO installation.  A native C++ component
reference is included separately under `native/`; it is not a complete product
worker and still requires authenticated external layer assets and the pinned
runtime.  The Python source closure includes the
full-model contract/builders, source-cross-cache and mask-pooling graph
dependencies, native BF16 attention/finite-check CL/XML assets, and the
opt-in bridge qualification helpers.  The full OpenVINO runner
under `reference/openvino/` is a captured B580 research runner, not a portable
arbitrary-device backend: its qualification helpers intentionally require the
recorded adapter/device policy and pinned runtime.  Treat the runner command
below as a reproduction recipe for that environment, and adapt and requalify
the device policy before using another accelerator.

## What is included

The execution path is deliberately explicit:

* the official FlashVSR+ source-side LQ projection and DiT/TCDecoder loaders;
* resident 30-layer DiT PRE/POST graph construction with authenticated layer
  schemas and BF16 raw-storage boundaries;
* sparse window attention, remote token/context, compact K/V, precise BF16
  cross-attention, FFN2 MatMul, and the bounded phased decoder used by the
  measured research runs;
* CPU structural checks and evidence helpers that validate shapes, finite
  values, source/model identities, and byte-level outputs.

`--cross-precise --ffn2-matmul` selects the latest qualified research variant.
It retains the original Result/state boundaries and uses a named BF16 bridge;
it is not an implicit default.  The `turbo` and `base` descriptors are
separate and authenticated in `reference/openvino/full_runtime/plus_modes.py`.

The measured local reference was a 45-frame 4K-output B580 experiment.  The
latest qualified Turbo and Base clocks were about 3.6177 and 6.032 seconds per
frame respectively, under the captured machine, input, OpenVINO build, memory
policy, and diagnostic-clock contract.  These are provenance facts, not
promises: rerun the full gate on the target system before drawing a performance
conclusion.  The Base path was not silently substituted for Turbo, and no
CPU fallback is hidden behind either mode.

The reported processing clock excludes one-time graph compilation and the
initial model/weight upload, but includes per-call weight-bank refills.  It
also excludes source resize/preparation, RIFE and video encoding, archive/hash
work, and optional diagnostic profiling.  Check the captured receipt and its
exact clock contract before comparing a rerun; these exclusions do not make
the reference a product throughput claim.

## External resources and portability

The closure test is intentionally split from the graph test.  Run
`reference/test_reference_graphs.py` with the pinned OpenVINO Python directory
in `FLASHVSR_PLUS_OPENVINO_RUNTIME` to construct the 4K PRE graph, its BF16
private/device views, and the precise+FFN2 POST graph without creating an
OpenVINO Core, querying a device, loading weights, or running inference.  The
test also checks that the local custom-layer configuration files can be
assembled.  It is a structural source check, not a GPU qualification.

`openvino/full_runtime/prepare_real.py` remains a historical qualification
utility.  Its external evidence, checkpoints, generated IR manifests, and
runtime-version gate are intentionally unchanged; this extraction does not
claim that utility CLI is independently qualified merely because the bounded
graph builders are constructible.

All large or machine-specific inputs are external.  Set these variables (or
use the equivalent launcher options):

| Variable | Meaning |
| --- | --- |
| `FLASHVSR_PLUS_SOURCE` | checked-out official FlashVSR+ source tree |
| `FLASHVSR_PLUS_MODELS` | directory containing `turbo/` and/or `base/` checkpoints |
| `FLASHVSR_PLUS_DIT`, `FLASHVSR_PLUS_LQ` | optional explicit checkpoint paths |
| `FLASHVSR_PLUS_TCDECODER` | TCDecoder checkpoint |
| `FLASHVSR_PLUS_PROMPT` | authenticated prompt tensor |
| `FLASHVSR_PLUS_SCRATCH` | bounded disk-history scratch directory |
| `FLASHVSR_PLUS_EVIDENCE` | external evidence/receipt directory |
| `FLASHVSR_PLUS_RGB_REFERENCE` | optional authenticated RGB24 source directory |

No variable defaults to the original workstation.  With no variables, the
official source resolves to the bundled `reference/source/` closure and the
large model/input/scratch paths resolve under `reference/external/`; execution
fails closed when a required external file is not present.  The runner
additionally needs the pinned OpenVINO 2026.5
development runtime, NumPy, safetensors, Pillow, and (for the official
projection/decoder source path) the matching CPU Torch environment.  The
qualified runtime was built from OpenVINO revision
`7d81aad55a5`; a different OpenVINO build is an unqualified experiment.

## Running the reference

The thin launcher keeps flat research imports isolated from the notebook tree
and forwards all model options unchanged.  `--input` is a prepared-input
directory, not an NPZ filename: it must contain the authenticated
`manifest.json` and the prepared arrays named by that manifest.  The value
passed to `--input-sha256` is the SHA-256 of that `manifest.json` (the
directory is the namespace for the manifest), not a hash of an arbitrary video
file.  Supply a fresh output path and the external model/source resources:

```bash
python scripts/flashvsr_plus_openvino.py \
  --source-root /data/flashvsr-plus/source \
  --models-root /data/flashvsr-plus/models \
  --tcdecoder /data/flashvsr-plus/models/TCDecoder.ckpt \
  --prompt /data/flashvsr-plus/source/prompt_tensor/posi_prompt.pth \
  --scratch-root /fast/flashvsr-plus/scratch \
  --evidence-root /data/flashvsr-plus/evidence \
  --run --mode turbo \
  --input /data/flashvsr-plus/input/common-video-3840x2160-paired45-attempt1 \
  --input-sha256 60b7a1a57ccf363e0be1af01aeea16014d0d3a7488d133904f1d252728c1aa4c \
  --output /data/flashvsr-plus/results/turbo-run-001 \
  --state-barriers \
  --remote-pre --remote-context --remote-tokens \
  --cross-precise --ffn2-matmul \
  --decode --disk-decoder --rgb-decoder \
  --fuse-memories --remote-history --fuse-growth --remote-frames \
  --compact-kv --retained-kv-roi --split-key --bounded-prefix
```

This is the complete staged-decoder/history configuration from the qualified
45-frame Turbo receipt; it is a reproduction recipe, not a claim that the
placeholder `/data` resources are authenticated.  To select Base, use `--mode
base` and provide the authenticated Base checkpoints under `models/base/` (or
explicit `--dit`/`--lq` paths).  RGB input is opt-in and must use an
authenticated RGB reference; it is never silently inferred from an arbitrary
video directory.  The qualified receipt used prepared source arrays, so it
did not add `--rgb-input`.

The notebook is intentionally a provenance/setup guide rather than a hidden
download cell.  It does not fetch weights or write caches.  Review the output
manifest, source fingerprints, memory admission, and retirement receipts
before comparing runs.

## Precision, memory, and security boundaries

The research closure treats BF16 tensors as raw `uint16` storage at the
explicit graph boundaries.  It does not insert an F16 numerical cast in place
of a BF16 boundary.  Every selected checkpoint and source identity is
allow-listed; tensor names, shapes, dtypes, finite values, and output counts are
validated before graph ownership is published.  Disk history is bounded and
must be explicitly enabled; it is not a general unbounded video spool.

Inputs and receipts are read with `allow_pickle=False` where applicable.  Mode
checkpoint paths are resolved beneath the selected model root, scratch and
evidence roots are caller-owned, and malformed manifests fail before device
allocation.  These checks are research safety properties, not a sandbox: run
the code only with trusted external directories and the intended OpenVINO
device policy.

## Provenance and licenses

See `reference/provenance.json` for the source/model/runtime pins and qualified
clock scope, and `reference/provenance_sources.json` for the per-file source
fingerprints and extraction roles.  Each manifest record keeps both the
authenticated `origin_sha256` (when a source file exists) and the shipped
`sanitized_sha256`; the legacy `sha256` field aliases the latter.  `native/README.md` describes
the native component reference and its intentionally omitted learned layer assets.
`reference/LICENSE.GPL-3.0` covers the extracted research and OTC native code;
`reference/LICENSE.APACHE-2.0` covers the copied official FlashVSR+ source
subtree.  Preserve both notices when redistributing this reference extraction.
Weights and generated evidence remain outside this tree and are not
redistributed here.

Run the offline structural check with:

```bash
python reference/test_portable_closure.py
```

That check does not import OpenVINO, Torch, weights, or media.
