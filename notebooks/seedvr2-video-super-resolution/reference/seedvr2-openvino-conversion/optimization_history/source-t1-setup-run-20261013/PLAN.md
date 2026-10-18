# T1 setup-only measurement

Date: 2026-10-13.  This leaf is source preparation only.  No GPU, Torch,
OpenVINO/Core, checkpoint, payload, build, or runtime launch was performed
while preparing it.  Root owns review, the fresh B580 occupancy/preflight,
launch, and teardown.

## One finite next action

Root may invoke `source-only.py --output <new-direct-child>` once, using the
existing frozen T1 outer runner.  The output must be an absent direct child of
this leaf; a failed output is evidence and is not reused.  The root wrapper
must establish fresh ownership of B580 `0000:03:00.0`, verify the current
post-reset health/occupancy, and retain the unchanged initial 12.4-GB root
preflight.  Those checks are outside this adapter.  This leaf introduces no
T5 floor, no new memory threshold, and no authorization protocol.

The runtime sequence is deliberately finite and setup-only:

1. Call `source_metadata()` before any Torch, device, model, or payload action.
   Verify the frozen T1 driver/seal and closed `ROOT-CHECK.json`.
2. Reuse `old.inputs()`, frame 02, the original RGB normalization, the
   existing reduced VAE loader, and the existing `uhd.vae_runtime` registration.
   Encode once with the frozen `512x512` tiles and `64x64` overlap, retain the
   first 16 channels, and copy the CPU F32 result.
3. The frozen tiled VAE returns direct 4-D `(1,16,270,480)` for this singleton
   frame.  Require that source shape, pass it unchanged to the frozen
   4-D `artifacts.mean_shape` boundary, which adds the singleton temporal axis
   and produces `(1,16,1,270,480)`.  Hash the saved BCTHW result against the
   closed-source `MEAN_SHA`.  This matches `driver_v2.py:48–49,69–73` and the
   pinned VAE implementation's singleton-frame squeeze.
4. Move the same VAE owner to CPU with the existing
   `_standard_model_movement`, synchronize, deep-clear, and require Torch
   allocated/reserved `0/0`.  Keep the CPU VAE reference alive and record
   `after-vae-release`.
5. Construct the original `uhd.streamed_dit(torch,device)` while that CPU VAE
   owner remains retained.  Record `dit-setup-before-clear`, invoke only the
   existing public `clear_memory(deep=False,force=True)`, synchronize, and
   record `dit-setup-after-shallow-clear`.  The receipt requires zero DiT
   forwards.  Then use the existing DiT CPU movement/deep-clear path, verify
   allocator `0/0`, and record release/final snapshots.

Each snapshot retains the observer's kernel-first physical sample and Torch
allocated, reserved, and monotonic maximum counters.  The only adapter
arithmetic check is `free == size - usage`; sampled physical usage and
allocator maxima are observations, not an invented peak or admission bound.

## Current recovery authority

The source pins the current stacked-GN recovery epoch, not the superseded
authority.  The files and exact SHA-256 pins are:

| source | SHA-256 |
| --- | --- |
| `source-contiguous-t1-20261012/driver_v2.py` | `f36433c131ccf21c4a512fab0de079b961364ce6f6d97c4f4b1302bd4cda86c3` |
| `source-whole-rgb-minimal-run-20261012/results-review/ROOT-CHECK.json` | `745c425ed75304ec38a25243a8c39d4203c7937aa24e03055e6ded67a4bf2767` |
| `b580-stacked-gn-recovery-20261013/authority.py` | `2c058c714b1d5d738fb3ea0a82ed3c57b423253fc9799b195f79a8146d401f50` |
| `b580-stacked-gn-recovery-20261013/RECOVERY.json` | `491688187e2e7969262e4e163ee53a03def93e737b2c10a42f99d4a70aee8123` |
| `b580-stacked-gn-recovery-20261013/ROOT-RECOVERY.json` | `d5bc0b624194471786f9798f10dca768a0b88243aa734059e69a4afc6622e563` |
| `uhd-phase-major-boundary-review-20261011/health.py` | `b2591479539b5b5f32892e431ef86bfa4c7ae932b6a00a1329ade549fd7e54eb` |
| `uhd-phase-major-frame-20261011/benchmark.py` | `e1d71eb635f52ad6017f6a1777e66c9b856d37e9f4c0250ba09fea6f8af3a35d` |

The recovery JSON is `ROOT_APPROVED_POST_RECOVERY`, and its root evidence is
`TINY_NATIVE_AND_XPU_PASS_SAME_BOOT` for B580 UUID
`86800be2-0000-0000-0300-000000000000`, with no reboot or other-GT/GPU reset.
The runtime loads `authority.py` and calls its existing `approved()` method;
the metadata pass independently pins both JSON artifacts.

## Existing facts, not gates

The frozen T1 README records a separate DiT model-ready GPU observation of
`6,862,435,840` B.  It also records `6,782,952,896` B of unique F16 checkpoint
storage and `7,785,545,804` B for three CPU BF16 owners.  These are inventory
and historical observations, not a CPU-RSS prediction or reservation and do
not become a threshold here; this run measures the actual VAE-retained to DiT
setup transition.  Likewise, the prior isolated physical sample
`12,755,009,536` B and the `17.920090376244236 s/frame` stage sum remain
labelled historical facts only.

The adapter does not call `old.physical_admission()`, weaken the root initial
floor, or reuse any T5 admission predicate.  It adds no backend, model
loader, allocator policy, dataclass, retry, or generic verification layer.

## Follow-on decision boundary

This setup receipt does not award quality, speed, or a T1 baseline.  After a
successful real measurement, root may run the separate isolated stock versus
qualified N32 block-clear adapter-32 comparison with the original T1 inputs,
byte-exact stock output, the existing `17.92009 s/frame` stage-floor bar, and
raw physical/allocator evidence.  Only after that comparison should a true
full cold/warm T1 reference preserve the frozen normalization, conditioning,
noise/timestep, output bridge, synchronized phase clocks, and CPU ownership.

## Local source validation

The owned files are `PLAN.md`, `source-only.py`, `source-v1.py`, and `tests.py`.
The test suite is CPU-only and checks the real artifact mean-shape contract,
current recovery pins, source seams, shallow-clear ordering, retained VAE
ownership, zero DiT forwards, and direct-child/no-retry behavior:

```text
python3 -B tests.py
python3 -B source-only.py
```

The second command is metadata mode and must report `PASS_SOURCE_ONLY` with
zero Torch imports, Core calls, device queries, model/payload reads, and DiT
forwards.  `source-v1.py` is preserved provenance for review only and is not a
launch candidate.

## Root wrapper prepared for review

The only launch files added after the source preparation are
`run-root.sh` and `run-inside.sh`.  They are not run in this preparation turn.
The intended command, once root reviews it, is:

```text
bash run-root.sh setup
```

The outer wrapper first refuses any existing output, observation, scope proof,
log, cgroup, source-pin, or authority artifact; then it records the current
source metadata, exact `source-only.py` SHA, current stacked-GN authority,
boot/BDF/vendor/device/xe identity, empty B580 clients/holders, journal
boundary, kernel-first journal, VRAM, and unchanged initial gates.  It uses
the existing strict `card2`/`renderD130` device scope and the existing
`check-scope.py` proof.  The inner wrapper snapshots cgroup state before and
after, confirms the inherited scope, and invokes only `source-only.py
--output`; it does not add a loader, backend, admission predicate, or retry.
After natural child exit, the outer wrapper captures kernel-first postflight,
clients/holders/VRAM, source metadata and source SHA again, and the authority
again.  A changed source or authority, new kernel fault/query error, holder,
scope mismatch, or child failure is retained and returns failure; there is no
timeout, signal, retry, or quality/performance award.

The scope is `MemoryMax=24G`, `MemorySwapMax=0`, and `CPUQuota=400%`.  The
requested 18-GiB default was not silently imposed: the authentic frozen T1
outer runner and T1 README explicitly use a 24-GiB/no-swap/4-CPU enforcement
cap because retained CPU BF16 owners and unknown framework/file-cache/RSS
overhead are not a proven bound.  This wrapper preserves that existing T1
scope fact; it is not a T5 gate or a new T1 admission threshold.  The root
preflight remains the unchanged 12,400,000,000-B driver-free and 25,165,824
KiB `MemAvailable` safety check, and postflight records raw VRAM without
inventing a new free-floor threshold.
