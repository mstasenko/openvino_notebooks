# Lean contiguous PyTorch T5 baseline — source-only plan

Status: source-only design review. No model/checkpoint payload, tensor/array
payload, OpenVINO Core, Torch device, build, or GPU operation was performed.
This plan proposes no production edit or launch authority; root owns any later
implementation and device admission.

## Evidence and exact source closure

The review read only the named native sources and small source/plan leaves:

- `native/ai-enhance/seedvr_frame.cpp/.hpp`, `seedvr_session.cpp/.hpp`,
  `seedvr_component_cache.hpp`, `seedvr_assets.cpp/.hpp`, `main.cpp`, and
  `CMakeLists.txt`.
- `source-contiguous-matched-plan-20261012/PLAN.md`.
- Existing source scripts `temporal-uhd-complete-source-20261012/{spec.py,
  child.py,benchmark.py}`, `source-contiguous-t1-20261012/driver_v2.py`,
  `source-contiguous-t1-run-20261012/{supervisor.py,run-root.sh}`, and
  `temporal-native-prototype/xpu_t5_reference.py`.
- Named source-quality/performance plans: `native-real-t5-session-20261011`,
  `native-uhd-t5-session-plan-20261011`,
  `native-f16-collapse-full-quality-plan-20261011`,
  `native-next-throughput-plan-20261011`,
  `temporal-baseline-fit-plan-20261012`, and
  `temporal-lean-integration-plan-20261012`.

Current source pins useful for a future additive wrapper are:

```text
seedvr_frame.cpp       d99cca0cd0a7b0bd4ee2a3bc09284f9f8e8ae8b27a9e873c9521518332aa5815
seedvr_session.cpp     089c1febad0484299ee8d7ef7b83fbd62bf0a5794de55275ca2550a17882feba
seedvr_component_cache.hpp
                        1dd99845e53b78964333b2f478778fbfd4680526258cf119f87fc366028a758c
seedvr_assets.cpp      d89316ac1367a189eb43ecbfcf10f3f315267d967e14434b1a8f2977c8fe0c2c
main.cpp               caca7e12d5487ec69e355f907c5194be212ad00aee97e91566809ff63d7095b8
contiguous source plan 604a40c39d710fc670d5c400d01ff261f139c456731aa6a752929d8bd5710875
T5 spec.py             62b0624561ff49e674989533ef29d97cc1e21fd27377e4ab2f32f826536a318f
T5 child.py            5733e1ed600f885562c96bdcd0ae241b7e149dfa54a76fa487600d0ae6f662ab
T5 benchmark.py        a23929ce0a8b079c4cfb0c414c6d109fbdc62d323913d9af8137540114da175c
T1 driver_v2.py        f36433c131ccf21c4a512fab0de079b961364ce6f6d97c4f4b1302bd4cda86c3
T1 supervisor.py        30b2aec7d138b75c8216ae99fe2f7a9e6c1f1195f640c0ded2907414b5890c76
T5 reference            5730a8a03a8cbc51cd13a407144d9f4c7c09f46057d9fcc3878bcdaefb6f9d7a
```

## What the current OpenVINO path actually retains

`main.cpp:504-545` creates one `SeedVrFrameSession` for a 1- or 5-frame
batch and calls `run` on the packed chronological RGB. In
`seedvr_frame.cpp:262-286`, the constructor creates `dit_` before the first
`run`; that `SeedVrSession` allocates its persistent remote embedding, two
video/text state slots, and parity metadata in its constructor
(`seedvr_session.cpp:301-361,562-640`). The DiT object is a member and is not
offloaded or destroyed during `run`.

The actual T5 order is `vaeSpatial(encode)` → prelude → `dit_->run` → postlude
→ `vaeSpatial(decode)` (`seedvr_frame.cpp:289-359`). VAE request/compiled-model
objects and the active XML/BIN owner are released at tile-shape/stage
boundaries (`:363-403,462-505`), but the DiT remote state, `ov::Core`, and
session-owned assets remain live across both VAE stages. This is the key
co-residency hazard for a matched baseline: it must not be mistaken for a safe
stage sum or copied into a PyTorch runner as simultaneous GPU ownership.

For T2 UHD, `SeedVrSession::prepareAssets` retains all block snapshots only for
the explicitly qualified geometry; larger/non-T1 geometry falls back to
per-block owned IR (`seedvr_session.cpp:507-560`). The retained path uses a
shared weight bank and releases an old active request/model before a new parity
template for large geometry (`:430-458`); the two persistent state slots remain
live until session destruction. `SeedVrFrameSession::cacheGeometryQualified`
allows component-cache reuse only for GPU T1 at 960×768 or 3840×2160
(`seedvr_frame.cpp:407-412`), so a 5-frame T5 call must not claim that cache
policy.

The host cache is a separate contract, not a VRAM gate:
`ComponentCache::admit` requires observed host and cgroup headroom and can
disable/drop all entries; `store` rejects oversized exporter capacity, storage,
or headroom before compact copy (`seedvr_component_cache.hpp:31-109`). Asset
owners wrap authenticated bytes and must outlive any graph using their tensor
view (`seedvr_assets.hpp`); host/cgroup observations are read from the owning
native process and unknown values disable retention (`seedvr_assets.cpp:160-243`).

## Why the T1 physical gate matters for T5

The named contiguous-T1 plan records a natural pre-DiT refusal: after encoder
movement/deep clear, Torch allocated/reserved were zero, but physical usage was
1,271,721,984 bytes and free was 11,537,682,432 bytes, below the original
12,400,000,000-byte pre-DiT gate. This is backend/driver residency, not a Torch
allocator leak and not a GPU fault. A contiguous T5 runner must sample the
physical gate immediately before DiT activation; zero Torch counters are not a
waiver.

The named T5 plans establish only an envelope, not permission to launch:

- True T5 is five chronological RGB frames → latent T2, with source-token
  bound 64,800. The UHD logical BF16 model inventory is about 6.8 GB, while
  all 32 learned blocks are about 6.68 GB; persistent state, metadata, output
  handoffs, compiled scratch, and allocator padding are additional.
- The UHD T5 logical state/metadata accounting lists two persistent video/text
  states at about 664 MB, parity metadata about 86 MB, and final residual
  auxiliary storage about 332 MB; these are logical figures, not physical
  high-water measurements.
- A prior three-process T5 receipt observed scope peak 9,774,178,304 bytes and
  post-boundary physical free 12,784,226,304 bytes with a minimum-free gate of
  11,811,160,064 bytes. Its process boundaries prevent it from proving a
  single-process envelope.
- The fit plan's N16 CPU-offload projection is 4,135,481,120 logical bytes
  before checkpoint mappings, arrays, Python/runtime memory, and temporary
  movement. The proposed host cap is therefore a launch condition, not a fit
  proof. Separate Torch allocated, reserved, physical, host, and cgroup samples
  remain separately labelled.

## Recommended single-process runner

Create an additive private `contiguous.py` and one thin root controller in the
new leaf. Do not edit the frozen phase child, benchmark helper, native source,
or production CMake. Reuse the existing source authority and helper pins.

1. Import `spec.py` for the exact five-frame/T2 shapes, normalization,
   restoration factor, RGB quantizer, fixture receipt checks, and chronological
   roles. Reuse `child.py`'s source/fixture authority and condition packing,
   but do not call its phase CLI or write disk handoffs between stages.
2. Reuse `xpu_t5_reference.py`'s `source_check`, `streamed_dit`, `vae_runtime`,
   and helper loading only through the authenticated source closure. Its own
   fixture constants are 960×768, so the UHD wrapper must take geometry and
   arrays only from the exact UHD `spec.py`/child contract; do not mix its small
   shapes into the target.
3. Keep one CPU BF16 DiT object and one CPU VAE object for the process. The
   existing reference loader can construct DiT on CPU by passing a CPU target;
   the VAE loader returns a CPU object. Before each active stage, move only that
   stage to XPU using the existing source memory-manager movement/cleanup policy
   used by `driver_v2.py:160-169`, then synchronize and assert every parameter
   and buffer has the expected device/dtype and is non-meta.
4. Run exactly one complete sequence per call:
   `encode → DiT → decode`, using one T5 batch, not five T1 calls. Encode reads
   the five RGBs and returns the posterior mean; DiT consumes the fixed T2
   noise/text and `disable_cache=False`; decode returns all five RGB frames.
   Keep CPU handoffs in owned contiguous arrays and never alias a live device
   tensor to a mutable/reused host buffer.
5. At every transition, synchronize; delete stage tensors and temporary outputs;
   move the active model back to CPU (or destroy it only after its successful
   output is preserved); run healthy-path allocator cleanup; synchronize again;
   require `memory_allocated == memory_reserved == 0` before activating the
   next stage. No two stage models or requests may be live on XPU.
6. Use one process for cold and warm/replay calls. Retain the CPU model objects
   between calls only; reset the qualified RoPE/cache state at the same healthy
   boundary as the source reference. Do not retain GPU graphs, VAE state, or
   stale remote buffers across a call. On inference failure, preserve the
   receipt/raw outputs and stop naturally; do not query GPU state, clear/retry,
   or invoke a fallback from the exception handler.

The controller may reuse the fixed observer/health modules once around this one
child invocation, but must not inherit `benchmark.py`'s three subprocess loop.
The outer controller records child status, cgroup limits/events, kernel-first
health, physical pre/postflight, and holders; it must not turn a stage failure
into a successful whole result. No external timeout or signal is part of the
finite run.

## Clock and comparison contract

Record all of these fields rather than reporting a phase sum as T5 throughput:

- `process_wall_seconds`: child entry through final model release and natural
  exit, including output persistence and teardown.
- `cold_whole_t5_seconds`: from the first active-stage/model setup boundary to
  final synchronized CPU RGB, including model movement, all three forwards,
  CPU handoffs, synchronization, and healthy stage cleanup.
- `warm_whole_t5_seconds`: the same boundary for the replay call with the same
  five inputs/noise/text and retained CPU objects.
- Per-stage input-to-owned-output clocks, model setup/movement clocks, handoff
  clocks, allocator cleanup clocks, and archive/hash clocks separately.

Primary whole clocks exclude disk hashing/image analysis but include every
computation, transfer, synchronization, and ownership transition needed to
produce CPU-owned RGB. The native comparator must expose the same input-ready
to CPU-output boundary. Do not compare this clock with the existing three-phase
inference sum or with a single DiT stage timer. Torch BF16 versus mixed native
F16/F32 is an independent-source/deployment comparison, not byte-equality proof.

## CPU-only preflight before any root run

Add no model or array fixtures. A source/AST checker should prove:

- one runtime child and one ordered `encode → dit → decode` call chain, with no
  phase subprocesses, per-frame T1 loop, CUDA/HIP path, or stage disk handoff;
- a release guard after each stage that checks CPU/non-meta ownership, allocator
  zero, synchronization, and no simultaneous active request/model;
- outputs are preserved before late finite/shape/replay checks, while exception
  handling performs no GPU query, cleanup retry, or fallback;
- exact five-frame/T2 shape, condition/restoration/quantization contracts and
  source/helper pins come from the current UHD producer, not copied small-T1
  constants;
- negative fixtures cover missing handoff, nonfinite output, wrong shape, stale
  cache, failed movement, nonzero allocator, partial model movement, and a
  physical-gate refusal. These are tiny CPU doubles only.

## Limits and next decision

This plan provides a reliable whole-action methodology and identifies the
current native overlap hazard; it does not establish T5 fit, speed, quality, or
production readiness. Root must review the additive wrapper/source closure,
run the CPU-only preflight, then choose fresh physical/host admission and one
natural contiguous baseline/candidate attempt. Any failure remains a preserved
failure; no threshold, memory waiver, retry, precision change, or production
integration follows from this source-only plan.
