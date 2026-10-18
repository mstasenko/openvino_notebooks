# FlashVSR+ structural assets

These are fixed, CPU-generated OpenVINO structural assets for the qualified
4096-row BF16 POST graph. Learned tensors remain Parameters and are appended
by the native authenticated preparation table; these files contain no learned
checkpoint payload.

The selected graph is `flash_post_4096_bf16.xml`, with its two private BF16
Result barriers intact. `prepareFlashPostModel` verifies the fixed ABI and
FFN2 chain, then replaces just those Results with the two qualified owning
Assign sinks in memory. This avoids the stock IR reader's assumption that
every serialized Assign has a matching ReadValue. No feedback, new numeric
conversion, or computation is introduced. The two state owners retain their
full memory charge. The original Assign-only IR is a reproduction artifact,
not the selected load format.
The custom-layer XML uses only relative filenames and the checked-in CL files.

The decoder structural bundle adds `flash_decoder_assets.hpp` and three phased
UHD-working-geometry XML/BIN pairs (`front`, `grow_memory`, and
`remote_frame_child`).  The generated table authenticates 64 static and two
dynamic source parameters exposed through four child slices (two grow-memory
and two remote-frame-child); the learned checkpoint and prepared learned BINs
remain external.  The fixed bundle uses 16 tiles with a 12-row halo.  The
independent native preparation audit (`audit-main.json`, SHA-256
`1b8491f5ac8599ee154356364e71d6469ee422e0fb90a21d2ad7dc26ea11cef4`) matched
the source spans byte-for-byte on CPU.  This is structural/reference evidence,
not a complete decoder worker or UHD numerical/performance qualification.
The Python build-time pin generator authenticates the original `TCDecoder.ckpt`
(189,018,333 bytes, SHA-256
`e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161`); native
preparation consumes fixed raw F32 spans with BF16 round-to-nearest-even
conversion, so runtime needs no ZIP/pickle parser, Python, Torch, or derivative
checkpoint.
This README is an intentional public-extraction documentation adaptation; its
provenance record retains the actual OTC origin hash separately from the
shipped documentation hash.

Research regeneration used the pinned POST bridge/precise helpers. Runtime
pack generation must consume these files
and must not import the research helper tree.

Provenance (2026-10-17): source commit `b548a2cc2c929d3d4676f2973abe27876ec4dee7`,
model revision `46b79520946ac67dec94d1566490690953c7debc`, and the current
synchronized native preparation helpers are `seedvr_prepare.cpp` SHA-256
`c200e0c51ea91a7b4232b16b8665e99a8044746fd96f813026b954ff1c769501` and
`seedvr_prepare_detail.hpp` SHA-256
`dc5dc3040446116ee61bc4f275e0deb34ece7bb727dff27de5b9ae81127e8404`.
The generator is `scripts/prepare-flash-post.py` and uses the native build-only
BF16 converter. Its fixed XML SHA-256 is
`1de0453729011b0d6e61f43a993e04b559fe27c3aa0e4894c69cf42f3afc3d37`;
the structural BIN is unchanged. This selects a portable representation, not
a different learned checkpoint or precision policy.
The installed combined header SHA-256 is
`6ade2f94ecac684932d180d0ae9e61624e696a61f7e224f7c21acfa15ea52119`.
It passes a C++17 syntax-only check including 30 Turbo plans, 30 Base plans,
and 13 bindings.

The PRE pair `flash_pre_4096_bf16.xml/.bin` is a separate structural export
from `scripts/export-flash-structure.py`; its provenance records the 15 exact
graph-builder source hashes. The 768-byte BIN contains only small structural
constants, not learned tensors. Two fresh exports are byte-identical, with
13 Parameters, 18 Results and no state nodes.
The native PRE wrapper now verifies that ABI and recreates only its six
owning BF16 Assign barriers. PRE and POST share request/weight-bank handling,
but retain their separately qualified precision policies. The build-time
generator's `--stage pre` option emits the nine-weight plans for both modes.
All 60 prepared PRE layer BINs (540 learned spans) were independently checked
against the external source checkpoints with a separate NumPy BF16 RNE
implementation. A finite B580 Turbo PRE component test matched all 90 output
arrays from the Python reference, including ABA and input-preservation checks.
This does not qualify Base or the full video pipeline. Neither PRE nor POST is
linked into the export worker yet.

The three `flash_lq_projection_*_bf16.xml/.bin` pairs are build-only exports
from `scripts/export-flash-projection.py`. Their small BINs hold structural
constants only; the 3/3/5 learned weights for startup/prefix/suffix stay external.
Two fresh exports are byte-identical. The serialized graphs pass the stock IR
reader and native ABI/hidden-state/precision checks without compiling a device.
The prefix input is named `old_cache1`, distinct from its `cache1` output, so
the IR reader cannot silently rename it. `flash_lq_provenance.json` records the
builder/exporter hashes and working geometry (2176×3840, visible 2160×3840).
After the shared USM-device storage correction, the fixed-UHD native LQ Turbo
component passed the constant-weight Python output comparison across startup
and three consecutive groups, including non-finite rejection, failed-session
reuse rejection, and zero final tracked GPU allocations.  The corresponding
Base reference rerun also passed the same fixed-UHD component checks.  Other
geometries and a complete export worker remain unqualified.
The data graphs stay unchanged: a separate weight-free GPU request checks
their completed BF16 output roots. An inline finite-predicate fanout left the
LQ cache Results unwritten on the pinned runtime; using a Boolean mask with
staged ReduceMin in the separate request avoids both that issue and the slow
monolithic logical reduction. This is a workaround, not a proven core fix.
Startup retires before steady-stage compilation. The largest observed
allocation snapshot fell from 8,659,360,924 to 5,790,325,844 bytes; neither
number is a continuously sampled physical VRAM peak.
Both modes' three prepared LQ BINs (22 learned spans total) were independently
compared with the original F32 checkpoints using chunked NumPy BF16 RNE.
The native structural test authenticates the fixed asset pins before parsing.

The text structural pair `flash_text_bf16.xml/.bin` is an ordinary Result graph
with an intentionally empty structural complement; the four learned text
Parameters remain external and are described by `flash_text_assets.hpp`.
`flash_text_provenance.txt` records the source and conversion pins.  The
empty BIN is authenticated and is not a substitute for learned weights.

The fixed timestep-conditioning exporter and its `flash_fixed_assets.hpp` /
`flash_fixed_provenance.json` pins describe the 21,504-byte raw-BF16 `t` plus
`t_mod` layout for Turbo and Base.  The derived mode BINs are external and are
not bundled here; no learned checkpoint payload is hidden in this structural
subset.  The six exporter contract tests include optional NumPy packing cases:
the system-Python run reports four standard-library passes and two explicit
NumPy skips, while the pinned numeric environment is validated separately.

The split-attention configuration `plus_split_key.xml`, shared kernel source
`plus_sparse_xmx.cl`, and `flash_attention_assets.hpp` preserve the measured
BF16 transport and FP32 partial-accumulator policy.  The shared source
contains the SG16/2D path and the SG8 raw-word fallback selected when 2D
block I/O is absent.  The current A770 finite numerical gate passed direct and
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
an ownership-observation gap after 12:20 PDT.  These are pinned kernel/config
components only; the native full attention/controller path remains
unqualified here.

The optional `../../scripts/test_flash_attention_gpu.py` gate is not registered
with CTest.  Its default invocation runs only the CPU oracle; `--run` must
name `--kernel-root` and an explicit `--expected-device A770` or `B580`, and
requires an idle matching GPU.  Raw OpenCL results are component evidence, not
an automatic CI action or a product-throughput qualification.

The Cross component was separately checked on the pinned B580 for both Turbo
and Base using the layer sequence `0, 23, 0, 29, 0`; outputs matched the
resident Parameter-weight reference byte-for-byte.  The constant-weight graph
comparison is retained as a diagnostic (relative L2 about 0.0027--0.0029),
not as a quality or performance claim.  Edge preparation was CPU-audited for
four external BINs (prefix/suffix × Turbo/Base), ten learned spans total across
the two modes.  Before the USM correction, an isolated native/Python suffix
comparison produced all-zero outputs; after the correction, bounded native edge
component gates passed for Turbo and Base, including nonzero prefix/suffix
output hashes, suffix comparison against separate host-oracle receipts,
finite/sticky-error checks, and authenticated A/B/A rebind recovery.  The
unwritten-destination symptom was traced to the pinned oneDNN same-engine
stream-copy path mixing buffer-backed and USM-backed roots, not to a static
remote flag.  The shared stage now requires USM device roots.  These are
edge-component results only; the full worker, output video, and performance
remain unqualified.
