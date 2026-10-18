# Native FlashVSR+ OpenVINO structural reference subset

This directory is a source-and-structural-asset subset of the OTC native
FlashVSR+ PRE/POST, text, mask-layout, LQ projection, and cross-attention components.  It is
included for source review and regeneration provenance; it is not a complete
FlashVSR+ video worker.  A small standalone `CMakeLists.txt` builds the static
reference library and CPU-only structural tests; it deliberately does not build
the OTC application or run a GPU inference.

The baseline build has 21 native CTest entries; when a Python interpreter is
available, three additional stdlib exporter checks are registered. Enabling
`FLASH_POST_BUILD_PREPARER=ON` adds the preparation-library,
flash-preparation, and preparation-CLI checks (24 native, or 27 including
those Python checks).
The synchronized CPU builds passed both variants under the documented 3 GiB,
two-thread, no-swap envelope.  These are structural checks, not full-worker or
throughput qualification.

The selected load format is `assets/flash-assets/flash_post_4096_bf16.xml` plus its
structural `.bin`.  `flash_post_session.cpp` reads that ordinary Result model
and reconstructs the two owning BF16 `Assign` sinks in memory.  The older
Assign-only `_state` XML/BIN reproduction assets are deliberately omitted:
the pinned stock OpenVINO reader rejects serialized Assigns without a matching
`ReadValue`. No numeric conversion or feedback edge is introduced by the in-memory
rewrite.

The copied C++ sources, generated asset tables, custom-layer XML, and sealed
kernel sources are pinned copies from the OTC `flash-post` working checkpoint.
The shared stage validates the complete remote-root ABI; the PRE model/session
keeps the six BF16 storage barriers and uploads one authenticated nine-weight
bank at a time.  These are reference sources, not a public worker API.
The shared `flashDeviceTensor` seam keeps learned banks, activations, and LQ
cache roots in explicit USM storage; the bounded front session separately uses
authenticated `USM_HOST_BUFFER` roots where its ledger permits.  The earlier
unwritten-destination symptom was traced to the oneDNN same-engine stream-copy
path when buffer-backed and USM-backed tensors were mixed; it was not a static
remote flag.  The storage guard rejects a non-USM device root before request
submission.  This is a runtime-specific component correction, not a full
worker or performance claim.
The shared `plus_sparse_xmx.cl` source supplies the direct, precise-cross, and
split partial/merge entries.  It selects the qualified SG16/2D path when the
active device exposes 2D block I/O and the SG8 raw-word scalar-load path when
that extension is absent, while retaining the shared raw-BF16 and FP32
partial-buffer ABI.  The current A770 finite numerical gate passed direct and
split-merge outputs exactly against an independent CPU reference; precise
cross differed by at most one BF16 step (`0.00048828125`).  The earlier
45-frame 1280x720-to-3840x2160 native spatial-worker smoke run is historical
functional evidence: its 630.92-second wall time was contended by another
GPU owner's final 78 seconds and included initialization, compilation,
resizing, and writes, so it is not a clean benchmark.  A current Turbo
repeat-eight run completed 45 frames in 487.78 seconds (10.8396 seconds per
source frame), versus the repeat-four baseline's 597.03 seconds (13.2673
seconds per frame).  Its full 1,119,744,000-byte output has SHA-256
`6a7648efa05cf0f965fd4976a252f447db2130b5751ef36c22077530643256b9` and is
byte-identical to the baseline's 45 unique frames; the receipt reports empty
scratch, zero swap, no OOM, and no new GPU faults.  The Base repeat-eight run
also completed 45 frames in 1322.86 seconds (29.3969 seconds per source
frame), versus the repeat-four A770 baseline's 1959.24 seconds (43.5387
seconds per frame), a 32.5% wall-time reduction.  Its full
1,119,744,000-byte output has SHA-256
`29dbb3b66a0a5b2d5013d729244bd8debd31a72ef363bf2e558ea451f869b99b` and is
byte-identical to the baseline.  The receipt reports empty scratch, zero
swap, no OOM, no new GPU faults, and no competing workload observed during
the candidate.  Cgroup memory reached its 10 GiB cap (`memory.events
max102038`) with substantial I/O pressure; maximum process RSS was
6,655,108 KiB.  These capped process-wall timings include initialization,
compilation, resizing, and disk writes, exclude decode/RIFE/encoding, and are
not pure-GPU execution or end-to-end editor export timing.  The later LOW
queue-throttle policy was not in this binary, and the prior A770 baseline has
an ownership-observation gap after 12:20 PDT.
`../reference/provenance_sources.json` records each source fingerprint, and
`../reference/provenance.json` pins the generated asset table. The table still names the 30 learned
Turbo/Base layer BINs; those large learned files, all model weights, built
libraries, and runtime binaries are intentionally not redistributed here.
Consequently the native PRE/POST sessions are not independently runnable until
a caller supplies those authenticated external layer assets and the pinned
OpenVINO runtime.  The CPU structural tests do not need learned layer assets,
but they do need the OpenVINO development package.

`flash_model.cpp` builds the separate, weight-free BF16 output-check model used
by the GPU stage after LQ/Cross data requests.  It keeps BF16 activations in
their original domain and reduces independent Boolean masks to one host-side
scalar F16 result; it does not mutate the ordinary data Result graph.
`flash_cross_session.*` provides the source-only per-layer Cross K/V session
with the same explicit context and external-weight ownership rules.  The
asset verifier and preparation library also authenticate an empty structural
BIN when an IR has no Constants, while XML, learned spans, and reconstructed
output remain strictly positive and hash-checked.  This is an authenticated
structural-asset case, not permission to omit learned payloads.

`flash_text_model.*` validates the ordinary Result-based text graph and its
four external BF16 learned Parameters.  `flash_text_bf16.xml/.bin` and
`flash_text_assets.hpp` contain only the empty structural complement and
authenticated source-span table; learned text BINs remain external.  The
CPU test uses `read_model` only and does not qualify text or end-to-end video
execution.  `flash_mask.*` is the bounded CPU-side grouped mask/index layout
helper used by the research path.  Its finite, tie, retention, and malformed
input tests are not a GPU attention or full-quality qualification.

A historical previous-source snapshot was qualified as a POST component on the
captured Intel B580 research device for layers 0, 23, and 29: nine public
outputs were byte-exact
across A/B/A weight refills, the late-A result and input-preservation checks
passed, and the final tracked GPU allocations were zero.  The completed
shared POST audit recorded a 5.563609172 s component gate, zero new
kernel-log lines, 40 authenticated source snapshots, and nine raw outputs
with exact A/B/A restoration.  Its host/cgroup peak was 1,492,938,752 bytes;
that is not a GPU-allocation peak and not a per-frame throughput measurement.
These are historical component evidence only, not a fresh GPU qualification of
the synchronized guard/lifetime sources, and not a full 45-frame quality,
product, or end-to-end speed claim.

An earlier 5.9237-second component gate included setup/compilation, weight uploads,
readback, and output hashing. Separate warm POST samples averaged about
7.2–7.3 ms per call and exclude setup/readback. Neither number is seconds per
video frame: this component does not run source preparation, the other DiT
stages, the decoder, RIFE, or video encoding.

## Structural build and source preparation

The build needs the pinned OpenVINO development package and OpenSSL headers.
The default build has no model weights and no GPU inference:

```bash
cmake -S native -B /tmp/flashvsr-plus-native-build \
  -DOpenVINO_DIR=/path/to/openvino/runtime/cmake
cmake --build /tmp/flashvsr-plus-native-build -j2
ctest --test-dir /tmp/flashvsr-plus-native-build --output-on-failure
```

The Linux reference target also compiles `flash_video.cpp` as a controller
API/lifetime seam, without building OTC `main.cpp`, an application worker, or a
GPU-launch executable.  With `FLASH_POST_BUILD_PREPARER=ON`, the preparation
library includes the mode-selecting `flash_prepare_cli.cpp` and its structural
CLI test; these are build-time checks and do not run a device graph.

The native session path is intentionally narrower than this structural build:
it requires an explicitly resolved Intel OpenCL GPU context and the exact
OpenVINO build `2026.5.0-23457-7d81aad55a5`.  Another accelerator or runtime
build is an unqualified experiment.

The CTest entries exercise constant-memory source/latent chronology,
custom-op model contracts, the ordinary
Result-model-to-Assign rewrite, the bounded RoPE and setup/session seams, and
sealed kernel-file ownership.  They use
small CPU-side OpenVINO graph reads only; they do not compile or submit a
device graph.

The optional `scripts/test_flash_attention_gpu.py` gate is manual rather than
a CTest or CI action.  Without `--run` it performs only a deterministic CPU
oracle self-check.  With `--run`, pass the explicit kernel root and one
expected device, and run it only on an idle matching GPU:

```bash
python scripts/test_flash_attention_gpu.py \
  --kernel-root assets/flash-assets
OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd python scripts/test_flash_attention_gpu.py --run \
  --expected-device A770 --kernel-root assets/flash-assets \
  --dump-dir /path/to/fresh/attention-gate-output
```

The gate defaults are `atol=0.0005`, `rtol=0.01`, and partial-buffer
`atol=0.001`.  Its CPU oracle is a reference sanity check; an actual raw
OpenCL pass is component evidence only and does not establish product-worker
throughput.  No automatic GPU execution is implied.

The synchronized source set also contains the bounded attention/token seams,
the Linux mmap-backed history owner, and the Linux-denoiser orchestration
source.  These are build-only component references with their own structural
guards; they do not turn this extraction into a full streaming worker or a
quality/throughput qualification.

The intended bounded controller is phase-scoped: LQ, denoiser, and decoder
sessions can suspend compiled request workspaces between chunks while keeping
causal roots and temporal history alive.  It must stream chunks rather than
retain full-clip conditions or an output spool.  This reference remains fixed
to 2176x3840 and compression 4; hardware-stage failures are sticky, and the
controller, RIFE path, and CPU fallback are not product-qualified.

On Linux, the history backing uses `posix_fallocate` before mapping so ENOSPC
fails before publication instead of becoming a later SIGBUS.  The retained
history ledger is 14.94 GiB for Turbo and 33.62 GiB for Base, plus a 16 GiB
scratch-disk reserve; these are disk figures, not RAM or `/tmp` storage.

An external functional gate (receipt SHA-256
`f993ae46541c4ca1539f2775fc25e92929df31b93a6e9079a45510a9e2de1956`)
covered 17 ordered finite frames in two calls, eight LQ groups, and five
decoder parents.  It observed at most 10,931,526,318 device bytes and
3,877,109,760 host-USM bytes separately; host RSS was 3,719,192 KiB and wall
time was 120.01 seconds including cold compilation, host work, and disk checks.
The baseline exited cleanly with an empty history scratch area.  This is a
bounded functional component gate, not visual-quality, full-worker, or speed
evidence.

The exact Python checks for the build-time converter and table generator are
separate and require no weights:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/test_prepare_flash_post.py
```

The decoder learned-weight preparer is a separate build-time-only tool.  It
authenticates the pinned `TCDecoder.ckpt`, verifies 64 static and two dynamic
source-parameter bindings, and emits only source-span tables; no checkpoint or
prepared learned BIN is redistributed.  Its nine-test stdlib contract suite
is run with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/test_prepare_flash_decoder.py
```

The native `flash_decoder_session.*` is a bounded fixed-UHD decoder-session
building block, not a qualified full worker.  Its current policy uses
USM_HOST_BUFFER roots for the large front-side state while the 51 tiled BF16
history/tail, compact inputs, and frame tiles remain device-backed.  It exposes
at most four visible planar F32 frames on the host.  The
fixed compression-4 packed-parent ABI is caller-owned: packed roots must be
source-authenticated, finite BF16 with the exact layout.  The session skips
the three causal startup frames and trims the padded tail; it does not provide
another temporal-compression mode or an end-of-stream flush.  The CPU
`flash_decoder_frame` test proves tile geometry, validates each whole raw BF16
tile before normalization/scatter, and proves BF16-RNE normalization plus
bottom/right cropping.

The front CPU raw-BF16 check uses one reusable host scratch buffer, at most
267,386,880 bytes, and performs no FP16 narrowing.  Growth uses its existing
per-shape GPU checker graph sequentially; no growth-output readback or
all-shape checker residency is introduced.  The CPU tests cover subnormals,
values beyond numeric F16, infinities, NaNs, and shape-overflow rejection.  A
historical co-resident UHD attempt reached about 18.974 GB of tracked device
residency and stopped with VM error -12; targeted accelerator recovery and a
small ROI smoke completed without a new kernel fault.  Removing the front
checker accounted for an 802,160,648-byte reduction in compiled allocation
accounting.  A later conservative admission check observed 6,351,895,056
bytes in use (including a 204,718,080-byte caller-packed root) and rejected
fixed roots requiring 6,391,377,920 bytes against 12,168,933,376 bytes of
capacity before reserve.  Within this history, attempt 3 encountered a
compute/driver fault and required targeted recovery, attempts 4 and 5 stopped
at safe guards, and the older attempt-6 harness stopped at its all-bytes
memory-budget guard.  These negative results do not describe the current
host-USM component path.

The current front host-USM component passed its bounded checks on the pinned
B580 runtime: 7.61 seconds wall time, 9,206,615,570 bytes of construction
device-root accounting, 3,877,109,760 bytes of host-USM root allocation, and
6,211,968,530 bytes of sampled steady device usage after the first lane.
Progress reached 17; excess/sticky-error rejection passed; the frame SHA-256
was 40ee40e0167405cabf47216d66dca8a7e495116053cd8acc6824bed7d1c11d4e; and
tracked allocations returned to zero with clean logs and no holders.

The same bounded session passed a causal 49-frame run: 13 parents, 221
progress callbacks, 49 unique finite/nonzero frame hashes, frame zero
byte-identical to the component result, excess/sticky rejection, and zero
final tracked allocations.  Its 68.33-second wall time includes CPU input
generation, hashing, and full-frame scans, so it is not a speed measurement.
Sampled device usage was 9,206,615,570 bytes and host-USM root allocation was
3,877,109,760 bytes; process RSS was 1,679,216 KiB with no swap.  RSS is not
the complete host-USM allocation budget.  The receipt SHA-256 is
d50e6abccd9fdf325d1fa5de7938f70ee1c7facb59b32b207653f129c782692a; no
independent CPU oracle was run for later frames.

An independent CPU decoder diagnostic, using the same authenticated source
weights but a constant/F32-hint reference representation rather than the
native external-Parameter GPU representation, reported max absolute error
0.015625, relative L2 0.00493494 (0.49%), and PSNR 57.0164 dB.  Its receipt
SHA-256 is 9bd5420c0be1ba01ebdec8a89bf764878ebd6c789366fbd1b6d3c7c0bfbf2219;
it is not byte-parity or full-quality evidence.

The growth-stage checker also completed a bounded inference slice.  These
results are bounded session/component evidence only: geometry-wide controller
admission and retirement, cancellation, CPU fallback, and the complete native
worker remain open; no full UHD numerical-quality, product, or FPS claim is
made.

`FLASH_POST_BUILD_PREPARER=ON` adds the build-only authenticated preparation
library and raw-F32-to-BF16 converter.  `scripts/prepare-flash-post.py` then
authenticates caller-supplied Turbo/Base safetensors, invokes that converter,
and writes a fresh generated table directory.  The converter executable is a
caller-trusted build input; the safetensors are pinned by size and SHA-256.
The generated provenance records the fixed external-weights label rather than
a local weights path, so it does not leak a caller's filesystem location.  The
generator never downloads weights or ships learned payloads.  A typical
invocation is:

```bash
cmake -S native -B /tmp/flashvsr-plus-native-build \
  -DOpenVINO_DIR=/path/to/openvino/runtime/cmake \
  -DFLASH_POST_BUILD_PREPARER=ON
cmake --build /tmp/flashvsr-plus-native-build --target flashvsr-plus-openvino-bf16-converter -j2
PYTHONDONTWRITEBYTECODE=1 python scripts/test_bf16_convert.py \
  /tmp/flashvsr-plus-native-build/flashvsr-plus-openvino-bf16-converter
PYTHONDONTWRITEBYTECODE=1 python scripts/prepare-flash-post.py \
  --output /tmp/flashvsr-plus-flash-assets \
  --mode turbo --weights-root /authenticated/models \
  --bf16-converter /tmp/flashvsr-plus-native-build/flashvsr-plus-openvino-bf16-converter
```

The generated output is intentionally outside this extraction. Learned layer
BINs are not redistributed; `assets/flash-assets/` contains only small structural
graphs, custom-layer sources and provenance.

The fixed-conditioning header records separate 21,504-byte Turbo and Base
pins, and the corresponding derived raw mode assets are included in
`assets/flash-assets/` with their portable SHA-256 provenance.  They are fixed
conditioning data, not learned checkpoint payloads; the reference neither
downloads weights nor treats these small derived files as source spans.

The build-only graph exporters include text, edge, and Cross structural IR
paths under `scripts/`.  They require an explicit curated reference
root, keep learned tensors as BF16 Parameters, record only relative artifact
paths, and perform no OpenVINO device compilation or inference.  The public
path policy in `reference_paths.py` supplies external prompt/checkpoint roots;
no workstation cache path is a default in this extraction.

The shared exporter serializer reads each newly written IR back through the
stock OpenVINO frontend before publishing its manifest.  It rejects a
serialized input/output name, order, shape, or element-type drift.  Learned
decoder Parameters are selected by source Parameter identity rather than by
shape or value; the phased bank slices retain that identity across the
external factory.  Source hashes in the provenance files describe the
reproducible input closure; they are not a cryptographic runtime pin or a
sandbox for the trusted build-time reference code.

The native Cross, LQ, and edge validators use one shared structural guard for
ordinary BF16 Parameter/Result graphs, bounded constants, and rejection of
hidden state or numeric F16 intermediates; each role keeps its own exact
ports and topology checks.  That is a current source/CPU structural
qualification, distinct from the separately reported fixed-UHD component
receipts below.

The decoder exporter is an additional complete-graph structural building block at
the tiny 8x8 test geometry; it is not an UHD export.  Its first
eight-test XML/BIN/manifest receipt (XML
`3335de180beab2d6e47f9f9fc8b68699c39aa6a79fe0c44c079b2d6cd4fb35c6`, BIN
`b545661c58527706805bbc996cf2103ea18c3a6e424c8ab5cb49e16a9e96ac48`,
manifest `85dac00445a07752361d527476d7be670e2be179cc2c685e14ad80a8d4f7eb1c`)
is retained as a historical source-control receipt.  The current strict
complete export is manifest
`a060095455b32f34a5ca1acaacf9eae9025b6943f8ae5517e79d1d432f26e4ec` (XML
`a1908bd715773eb085187b644a82279cc8845f413a3ff7c22936660cf4ffb55c`, BIN
`7ccb1528f8e8850781dfc5687462f0427da148bc73cfe61237577fcb231a6e21`).
The current decoder contract suite has 12 stdlib-only tests, the external
factory suite has 7, and the source-span preparer suite has 9; neither creates
an OpenVINO Core, queries a device, compiles, or performs inference.  This
complete export remains an 8x8 structural graph; the separate phased bundle
is the fixed-UHD working-geometry source/building block.  Full trajectory
numerical execution, a full worker, and throughput are not claimed.

The lean learned-factory traversal was also checked by an independent
ABI-linked CPU graph audit.  The fresh receipt
`report-lean-traversal-v4.json` has SHA-256
`52e8044a6f24f9d7c8cc0d14f25a24d56322c129d9682324dd8b7d8d5139de27`; control
and candidate semantic DAG fingerprints were identical
(`ff8404662fd6b85daa6cc3a4247aee8fb6e5ea680df0eb6aab91005bece8ba5b`).
The audit created no Core, device, compilation, inference, or checkpoint
state.  This is structural CPU evidence only and does not qualify GPU or UHD
execution.

The opt-in `--bundle phased` decoder was then exported and strictly read back
at reviewed 2176x3840 geometry with 16 tiles and a 12-row halo.  The current
three-graph manifest is
`039d2cbf423cb45e90fe7b66802d1eafeaeab14239632e8fb349498aca2b98e5`; its
portable boundary uses `middle_in` and explicit `*_out` state names rather
than relying on frontend renaming.  No device compilation or inference was
performed, and this is not a UHD numerical or full-worker claim.

The synchronized native decoder structural assets are the three phased
XML/BIN pairs plus `flash_decoder_assets.hpp`; the header records 64 static and
two dynamic authenticated source parameters exposed through four child slices
(two grow-memory and two remote-frame-child).  The independent native
preparation audit (`audit-main.json`, SHA-256
`1b8491f5ac8599ee154356364e71d6469ee422e0fb90a21d2ad7dc26ea11cef4`) matched
those source spans byte-for-byte on CPU.  The Python build-time pin generator
authenticates the pinned `TCDecoder.ckpt` (189,018,333 bytes, SHA-256
`e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161`) and
emits fixed source-span records; native preparation consumes those records with
BF16 RNE, so runtime needs no ZIP/pickle parser, Python, Torch, or derivative
checkpoint.  The structural model tests use CPU `read_model` only, and
the preparation tests use temporary synthetic contracts; neither performs
OpenVINO device compilation or inference.  These assets cover the fixed UHD
working geometry only and do not qualify a decoder worker, video quality, or
throughput.

The latest independent default-equivalence receipt is the
`report-node-normalization-v3.json`, SHA-256
`83a30f5751329ac8830b7df66940a77a728e6c5196da888405825eda67ce64da`, built
from phased source SHA
`e6c3d53cea1268525b4ed7074aac23bfccb5b915ed0e192719b098e09ebec6e0`.
Seven default graphs and four default bank arrays remained unchanged, while
the identity-linked factory covered 64 static and two dynamic source
Parameters (66 total).  This is structural CPU evidence only; it does not
qualify GPU, UHD numerical execution, a full worker, or throughput.

The native `flash_lq_model.*` and `flash_lq_session.*` sources add structural
validation and a context-owned three-stage projection session to the reference
library.  Their runtime path remains a component-only reference, is fixed to
the 2176x3840 working geometry, and is not a full FlashVSR+ worker.  The LQ
CTest only reads the structural IR on CPU; it does not compile or submit a
device graph.  After the USM storage correction, the current Turbo and Base
native component reruns passed their prefix/suffix output,
finite/sticky-error, and authenticated A/B/A rebind checks.  Their raw
comparison reports are
`native-lq-usm-turbo-20261018-attempt1/comparison.json` and
`native-lq-usm-base-20261018-attempt1/comparison.json`; all three projection
groups in each mode were byte-exact against the independent constant-weight
LQ references.  Neither result is a full-worker, video-quality, or throughput
claim.

An earlier historical LQ GPU attempt failed because its inline finite-sidecar
fan-out differed from both matched constant- and parameter-weight Python
reference variants; the parameterized graph path itself matched.  The current
workaround uses the separate
weight-free BF16 checker described above.  Frozen Turbo and Base component
qualifications of this fixed-UHD path each passed startup plus three projection
groups with byte-exact raw outputs, non-finite rejection with sticky failure,
and zero final tracked GPU allocations.  Their setup/phase timings are
component diagnostics, not frame throughput.  This is component evidence
only: it is not a full FlashVSR+ trajectory, product-quality, FPS, or
end-to-end speed claim.  Six LQ BINs cover two modes and 22 learned spans
total, with 688,947,096 prepared bytes per mode.
The standalone finite-checker test exercises signed zero, ±1, max-BF16,
65,536, both-sign infinities, and NaN on the host; this is CPU evidence only.

The `flash_cross_kv_bf16.xml/.bin` pair is a build-only cross-attention K/V
export.  It covers two modes and 30 layers with five learned weights per layer:
60 BINs and 300 learned spans total, 512 rows, and 283,392,360 prepared bytes
per mode.  The installed header SHA-256 is
`73cbd2f7de0b5038a755b2fb786647ce1334d947ce15bba9330aeb89cb873709`.
Its external CPU preparation receipt has SHA-256
`819eee0555b4fd7212612faab1814745022479c6813a48ba55da0b8a206cdade`.
The synchronized public CPU builds passed all 21 baseline CTest entries with
`FLASH_POST_BUILD_PREPARER=OFF`; the preparer configuration registers 24 entries
with
`FLASH_POST_BUILD_PREPARER=ON`, including the stream, asset, kernel, optional
preparation, fixed-conditioning, stage, PRE, bounded RoPE/setup, shared
finite-checker, text, mask, LQ, Cross, and Edges checks, alongside the Python
exporter/closure groups.
These are CPU-only structural checks: OpenVINO used `read_model` and host-side
finite evaluation only; no device graph was compiled or submitted.
The added native Cross validator is a
structural ABI/graph check.  The current synchronized USM component receipts
(`native-cross-usm-turbo-20261018-attempt1/component-results.json` and
`native-cross-usm-base-20261018-attempt1/component-results.json`) exercised
Turbo and Base in the layer sequence `0, 23, 0, 29, 0`; their ten K/V output
arrays per mode were byte-exact against the independent resident
Parameter-weight references.  The corresponding `comparison.json` reports
preserve the raw checks.  The constant-weight graph comparison remains a
diagnostic with relative L2 about 0.0027--0.0029, not a full quality or
performance claim.
The edge tables have CPU-audited preparation for four external BINs
(prefix/suffix × Turbo/Base), ten learned spans across both modes.  Before the
USM correction, an isolated native/Python suffix comparison produced all-zero
outputs; after the correction, bounded native edge component gates passed for Turbo
(`native-edges-usm-turbo-20261018-attempt3`) and Base
(`native-edges-usm-base-20261018-attempt2`): prefix/suffix output hashes were
nonzero, suffix outputs matched their separate host-oracle receipts, finite
and sticky-error checks passed, and authenticated A/B/A rebind recovery was
byte-exact.  The lifecycle receipts were natural, idle postflight, and had no
new kernel-log lines.  These are edge-component results only; they are not a
full controller, video-quality, FPS, or end-to-end performance claim.

The additional `flash_pre_4096_bf16.xml/.bin` pair is a structural PRE export,
not a GPU qualification. Its 13 Parameters keep all nine learned tensors
external; its 18 Results include six BF16 rounding barriers. Two fresh
exports were byte-identical. The independent learned-preparation audit checks
both Turbo and Base tables: 60 layer BINs and 540 learned spans, with every
source F32 span independently converted to BF16 RNE and compared by raw bytes.
That audit is CPU evidence only; the main review independently reran the same
receipt, and it is not GPU qualification.

A historical native PRE component run on the B580 with Turbo weights and
bounded synthetic 4096-row activations at layers 0, 23, and 29 matched all 90
public output arrays (ten fields per A/B/A phase and layer) against the pinned
Python reference byte-for-byte. Late-A replay, unchanged input roots, malformed
binding rejection, both scalar finite flags, and zero final tracked GPU
allocations passed. Its external audit receipt is
`audit_native_pre_cpp_20261017.json` (SHA-256
`89c3a0c662126b9a9f06dec8b14d3e99aabcaf2cbd35eccb04773bb95d99ad84`).
The 2.1074-second component gate and 895,311,872-byte host/cgroup peak are not
video-frame speed or GPU-memory measurements, and that historical receipt is not
a fresh GPU qualification of the synchronized guard/lifetime sources. That
historical record did not qualify Base inference, a full trajectory, or output
video quality; its raw fixtures and generated evidence remain external to this
source extraction.

A separate current-source USM PRE component receipt
(`native-pre-usm-20261018-attempt1/component-results.json`) reran the same
Turbo PRE boundary and its 90 raw output arrays with the synchronized
storage-guard/lifetime sources.  The finite flags, input-root preservation,
negative API checks, and final tracked GPU allocation check passed.  This is
still a single-stage component result, not a full trajectory, frame-rate, or
video-quality qualification.

Regenerate the structural export from the notebook directory with the pinned
Python runtime available:

```bash
python scripts/export-flash-structure.py \
  --reference-root reference --output /path/to/fresh/pre-structure
python scripts/test_export_flash_structure.py
```

The shared table generator supports both stages.  This extracted generator
uses only the checked-in `assets/flash-assets` directory;
`--stage pre` selects PRE, while the option defaults to POST:

```bash
python scripts/prepare-flash-post.py \
  --stage pre \
  --output /path/to/fresh/pre-table --mode both \
  --weights-root /authenticated/models \
  --bf16-converter /tmp/flashvsr-plus-native-build/flashvsr-plus-openvino-bf16-converter
```

`flash_stream.hpp` preserves the fixed four-latent call schedule, including
startup/tail handling, without storing a video-length frame table. This is
source-frame chronology, not RIFE interpolation or an end-to-end worker.

`../reference/LICENSE.GPL-3.0` covers these OTC native sources.  The official
FlashVSR+ source subtree remains under `../reference/source/`, and its Apache
notice is kept separately at `../reference/LICENSE.APACHE-2.0`.
