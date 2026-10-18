# SeedVR2 OpenVINO

This directory is a self-contained, source-first extraction of the SeedVR2
OpenVINO worker from OneTrackCat v0.1.1. It is kept on the `seedvr2-openvino`
branch as a reusable reference while a separate FlashVSR+ backend is developed;
FlashVSR+ code, private profiling evidence, source footage, SeedVR learned
weights and host-specific paths are deliberately not part of this directory.

## License boundary

The copied worker and build/packaging scripts retain the source project's
GPL-3.0-only terms. Read [LICENSE-GPL-3.0-only](./LICENSE-GPL-3.0-only) before
redistributing or combining this subtree with the Apache-2.0-licensed notebook
repository. The notebook repository's top-level license does not relicense the
files here. Third-party attribution is summarized in
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md), and the byte-preserved
notice archive is `assets/seedvr/notices.zip`.
The separate research conversion snapshot has its own Apache-2.0 source and
GPL-3.0-only probe notices; its boundary is documented in
`reference/seedvr2-openvino-conversion/README.md`.

## What is included

* `native/` is the 33-file C++17 worker, its authenticated graph
  asset tables, streaming/controller code, and CPU/math tests. It uses the
  standard OpenVINO runtime; Flash-specific custom operations are not included.
* `scripts/` contains only the seven SeedVR pack scripts from the clean v0.1.1
  baseline. They are root-relative (`ROOT` is this directory), so they do not
  depend on the original checkout layout. Python uses the standard library;
  the Node assembly/verification scripts retain the baseline's Node.js 22.22+
  and `adm-zip` 0.6.0 requirements.
  The small `package.json` records that exact build-time dependency.
* `assets/seedvr/` contains the reviewed structural graph/complement archive,
  the separately attributed RIFE IR archive, provenance manifests and notices.
  It contains no SeedVR learned checkpoint or prepared SeedVR learned BIN;
  the RIFE archive is intentionally a separate pre-trained asset with its
  own notice and pin.
* `reference/seedvr2-openvino-conversion/` is a separately licensed,
  source-only conversion snapshot. It preserves the official Apache-2.0
  SeedVR source, the GPL-3.0-only historical OpenVINO exporter/fused-graph
  helpers, bounded probes, and selected T1/T5 optimization source. It is not
  part of the native worker or its pack. The `reference/` directory is only a
  layout namespace; internal historical paths and provenance manifests remain
  unchanged.

The clean source snapshot is
`2d2e792c14a00253843ad297bb64fbb90cecb59f` (OneTrackCat v0.1.1). Files were
copied from that commit, rather than from a dirty research checkout. The
public `native/` layout and `seedvr2-openvino` command names are this fork's
adaptation of that baseline; native namespaces/source provenance and the
byte-preserved historical reference remain intact. The
structural archive and notices are checked by fixed size/SHA-256 pins in the
scripts and provenance files:

| artifact | bytes | SHA-256 |
| --- | ---: | --- |
| `assets/seedvr/structure.zip` | 1,580,909 | `d23c7af7eb47a219b24a0698eca6e4fa51c06afdd037bc9050ddc6cb39b7fb12` |
| `assets/seedvr/rife-ir.zip` | 21,210,600 | `912fea8a9a405799cbe122605da12d6e4b5330aec83a0e026910651cb34d39fa` |
| `assets/seedvr/notices.zip` | 64,876 | `21b9b99d79af32fe66800fc68efb104367e1e93bf4092b6e6395eed4b48d5578` |

The `structure.zip` size above is the compressed archive size. Its
authenticated members unpack to 24,684,720 bytes; that unpacked total is not
the file size used by the pin.

## Runtime model and precision contract

The worker accepts chronological RGB24 frames on stdin and writes RGB24 frames
on stdout. The frame geometry, frame rate, duration and device are explicit CLI
arguments. It keeps only the current temporal batch and model intermediates;
it does not load a whole video into memory. `--device CPU` is deterministic
CPU execution; `--device AUTO` performs OpenVINO device discovery and may select
an available GPU before falling back to CPU. The worker validates graph file
names, byte sizes, SHA-256 pins, port shapes/types, execution device and
finite outputs before publishing a result.

The numerical boundaries are intentional:

1. RGB24 is normalized to FP32 and fed through the FP32 public encoder/prelude
   ports. The latent condition is mean plus a fixed, versioned noise field.
2. The 32 fused DiT blocks keep video/text state on the selected device between
   blocks. The embedding boundary is rounded to FP16 before block/postlude
   modulation. Learned graph parameters and the fused block graphs are FP16;
   this is a storage/graph contract, not a claim of bitwise PyTorch parity.
3. The decoder uses FP16 on GPU with an audited residual boundary separated
   before compilation; encoder and prelude/postlude public ports remain FP32.
   VAE work is spatially tiled at 512 pixels with 64-pixel overlap and blended
   with finite-value checks. Spatial tiling can change context and therefore is
   not silently described as equivalent to an untiled reference.
4. RIFE is a separate FP32 interpolation stage. It is used only when the output
   FPS is higher than the source FPS; SeedVR itself enhances source frames and
   never interpolates generated output-clock frames.

### T1 and T5 temporal semantics

`--mode temporal` groups up to five contiguous source frames. Scene-cut flags
end a group, and a short tail of two to four frames is padded by repeating its
last source frame so the temporal graph still receives T5; only the original
frames are emitted. A true one-frame group uses T1. `--mode per-frame` always
uses T1. This is temporal batching, not a promise of five output frames or
frame interpolation.

The frame graph also has a latent-time contract: latent T1 is the qualified
large-frame path, while latent T2 reuse is bounded to 960x768 in this snapshot.
T2/UHD-T5, arbitrary devices, and complete PyTorch-quality parity remain
unqualified. Higher output FPS is handled by the separately pinned RIFE model.

## Models and source-to-IR status

The structural archive is reproducible and authenticated, but it is not a
learned model export. The pack recipe records the external source artifacts:

* `numz/SeedVR2_comfyUI`, revision
  `09ced71023636e9bc8cdf9cdecfb2625d1e691e8`,
  `seedvr2_ema_3b_fp16.safetensors` (6,783,018,808 bytes, SHA-256
  `2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304`);
* `ema_vae_fp16.safetensors` (501,324,814 bytes, SHA-256
  `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1`);
* the published ByteDance-Seed positive-conditioning asset and its Apache-2.0
  attribution, recorded in `assets/seedvr/PROVENANCE.md`.

No SeedVR learned checkpoint, prepared learned BIN, or creator text-encoder
source is included. The research subtree preserves the historical exporter
source, but not its generated learned outputs. `generate-seedvr-assets.py` validates
already-authenticated XML/BIN manifests and emits a fixed C++ table; it is not
an exporter and cannot reconstruct missing learned files. The official source
checkout itself has no OpenVINO exporter; the separately preserved research
history contains the actual OneTrackCat conversion helpers and representative
fused closure. A future source-to-IR reproduction must pin the official source
revision, helper hashes, OpenVINO/PyTorch versions, conversion options,
checkpoint/conditioning hashes, and every generated XML/BIN hash before
replacing the reviewed structural assets. Do not silently regenerate them with
a different runtime.

## Build and test

From this directory, a bounded CPU/math build does not require OpenVINO or any
learned model:

```sh
cmake -S native -B /tmp/seedvr2-math-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAI_ENHANCE_MATH_ONLY=ON -DAI_ENHANCE_BUILD_TESTS=ON
cmake --build /tmp/seedvr2-math-build --parallel 2
ctest --test-dir /tmp/seedvr2-math-build --output-on-failure
```

The full worker additionally needs a matching OpenVINO CMake package and
OpenSSL. The pinned native pack recipe is build-only and never downloads or
converts learned weights implicitly:

```sh
python3 scripts/prepare-ai-pack.py --prepare-only
```

The public worker is `seedvr2-openvino` (with the ELF payload named
`seedvr2-openvino.bin`); the default archive is
`release/seedvr2-openvino-pack.zip`.

The seven scripts preserve their original trust boundaries: fixed archive
member inventories, regular-file/no-symlink checks, bounded streaming hashes,
atomic output publication and explicit model/source pins. The pack assembly
expects a separately prepared structural model directory and refuses learned
`.safetensors`, `.pth`, `.onnx` or untrusted `seedvr/*.bin` payloads. Run the
Node scripts from this root after installing the recorded dependency, for
example `npm install --ignore-scripts` followed by the script's documented
arguments. No GPU, Core compilation or large model download is required for
the source/math checks above.

## Honest limitations

This extraction is a portable native reference, not a shipping quality claim.
The public subtree has no SeedVR learned checkpoint/prepared BIN, no
copyrighted source footage, no private lifecycle/profile evidence and no
end-to-end benchmark receipt. It does include the separately attributed
pre-trained RIFE IR listed above. CPU
math/stream/asset tests can verify contracts but cannot establish GPU speed or
model quality. The B580/T1 observations that motivated the worker were local
research evidence and are intentionally not repeated here as a portable
qualification. Production use still needs a separately authenticated model
pack, device-specific qualification, long-video memory testing, cancellation
testing, and a native quality comparison against a trusted reference.
