# T1 setup-only diagnostic: source contract

Date: 2026-10-13

This leaf is source-only preparation.  It does not build, import Torch or
OpenVINO, initialize Core, query the device, open a checkpoint/fixture, or
launch a GPU job when run in metadata mode.  Root owns review, fresh device
occupancy/health checks, the eventual B580 launch, and teardown.

## Concrete next action

Run `source-only.py --output <new-direct-child>` once under the root-owned
GPU lifecycle.  This is a finite setup measurement, not another readiness
predicate:

1. Run `source_metadata()` before any Torch/device action.  Reuse frozen T1
   `driver_v2.py:parent()`, its authenticated `old.inputs()` and
   `ref.load_helper` seams, then verify the closed `ROOT-CHECK.json`.  Resolve
   the existing `0000:03:00.0` XPU identity with the frozen resolver and the
   current post-GT-reset authority below.
2. Record `idle`.  Load the existing reduced VAE, use the original frame 02
   RGB normalization and `512x512`/`64` tiled encode, retain only its first 16
   channels, and copy the CPU F32 mean.  The live T1 encode is 5-D
   `(1,16,1,270,480)`; pass its singleton temporal slice (`mean[:,:,0]`) to
   the frozen `artifacts.mean_shape` 4-D contract so the saved result remains
   byte-identical BCTHW `(1,16,1,270,480)`.  Then hash it against the closed
   T1 mean.
3. Move that same VAE owner to CPU with the existing
   `_standard_model_movement`, synchronize, call the existing deep clear, and
   require current Torch allocated/reserved to be zero.  The VAE Python owner
   remains retained on CPU.  Record `after-vae-release`.
4. Activate the original `uhd.streamed_dit(torch, device)` owner while the
   CPU VAE owner is still retained.  Do not call the DiT.  Record
   `dit-setup-before-clear`, invoke only the existing public
   `clear_memory(deep=False, force=True)`, synchronize, and record
   `dit-setup-after-shallow-clear`.  The receipt requires zero DiT forwards.
5. Move the DiT owner to CPU, perform the normal existing deep clear and
   zero-allocator check, then record `after-dit-release` and `final`.

Every snapshot keeps the observer's kernel-first raw physical sample and a
JSON row containing physical size/usage/free plus Torch allocated, reserved,
and max counters.  The adapter checks only `free == size - usage`; it does
not claim that a sampled maximum is a peak bound.  An exception writes one
failed receipt and exits without retry, fallback, or post-failure GPU work.

The adapter deliberately does not call `old.physical_admission()` and does
not embed a new 12.4-GB post-VAE refusal or any T5 phase/setup threshold.  A
root-owned outer preflight may still perform the unchanged initial 12.4-GB
health gate before importing this runtime.  The purpose of this run is to
measure the actual T1 VAE-retained-to-DiT
activation memory transition, including the before/after public shallow
clear, so the next T1 decision has a real observed floor instead of the same
known refusal.

## Reused source seams and pins

`source-only.py` pins the exact runtime sources it loads:

| source | SHA-256 |
| --- | --- |
| `source-contiguous-t1-20261012/driver_v2.py` | `f36433c131ccf21c4a512fab0de079b961364ce6f6d97c4f4b1302bd4cda86c3` |
| `source-whole-rgb-minimal-run-20261012/results-review/ROOT-CHECK.json` | `745c425ed75304ec38a25243a8c39d4203c7937aa24e03055e6ded67a4bf2767` |
| `b580-post-gt-reset-authority-20261012/authority.py` | `8ead0b4791dd1df83d1396a834084bd426545acf8c25ea2759ce8d0d707eee41` |
| `b580-post-gt-reset-authority-20261012/ROOT-RECOVERY.json` | `701c4de2b2d9ad074ddb5e73e147287161ce04e8f937d641b491fe04427685cf` |
| `uhd-phase-major-boundary-review-20261011/health.py` | `b2591479539b5b5f32892e431ef86bfa4c7ae932b6a00a1329ade549fd7e54eb` |
| `uhd-phase-major-frame-20261011/benchmark.py` | `e1d71eb635f52ad6017f6a1777e66c9b856d37e9f4c0250ba09fea6f8af3a35d` |

The frozen driver already binds the original BF16 Torch/XPU VAE, VAE runtime
registration, streamed DiT construction, output shape helper, and memory
movement path.  This leaf adds no backend, model, adapter framework,
dataclass, predicate, allocator policy, or source-wide search.  Metadata mode
reads only these explicit source pins and reports zero Torch/Core/device
activity.

Runtime binding hazards for root review:

- `old.seal.verify()` happens before model construction; `ROOT-CHECK` must
  still report `PASS_EXACT_SOURCE_BRIDGES_AND_CLOSED_LIFECYCLE`.
- The root wrapper must establish fresh B580 ownership and health first.  The
  observer is kernel-first and has no DRM-idle gate; it is not a replacement
  for root occupancy checks.
- The CPU VAE must stay referenced until after the DiT setup snapshots.  Do
  not replace `uhd.streamed_dit` with a new loader or introduce a CPU-VAE
  dropper.
- `clear_memory(deep=False, force=True)` is intentionally the public shallow
  boundary being measured.  The later owner release uses the existing deep
  clear and current allocator-zero boundary.
- Snapshot maxima are monotonic allocator counters and physical samples are
  sampled observations.  Preserve all raw files and do not convert either
  into an invented admission margin.
- The output directory must be a new direct child of this leaf.  A failed
  run is evidence and must not be rerun into the same directory.

## Closed facts retained without making a new gate

The previous contiguous T1 attempt reached post-VAE physical usage
`1,271,721,984 B`, free `11,537,682,432 B`, with Torch allocated/reserved at
zero, and therefore refused the unchanged 12.4-GB gate before DiT.  The
closed isolated T1 phase sampled maximum physical usage `12,755,009,536 B`
(free `54,394,880 B`) and allocator maxima `11,120,621,568 /
12,559,843,328 B` allocated/reserved.  Those are historical observations,
not a new peak bound.  The prior `17.920090376244236 s/frame` value is an
isolated warm stage sum, not a contiguous T1 baseline.

The healthy T5 setup evidence and its fixed measured increment remain T5
only.  In particular, this leaf contains no T5 free floor or T5 gate.  The
latest actual T5 comparison (`53.713391533 s / 5` Torch versus
`57.014993821 s / 5` OpenVINO) does not establish a T1 result.

## Follow-on T1 reference and acceptance

After this setup measurement, root should run a separate isolated T1 stock
versus qualified N32 block-clear adapter-32 comparison using the original T1
inputs.  That comparison must preserve byte-exact stock output, the existing
faster `17.92009 s/frame` stage-floor bar, and raw physical/allocator evidence;
this setup leaf must not award a quality or speed result.

Only then should root run a true full T1 cold/warm reference through the
frozen phase schedule: warmup frame 02, measured frames 01/02/03, original
normalization/conditioning/noise/timestep/output bridge, synchronized
encode/DiT/decode, separate clocks, and CPU ownership plus deep-clear at each
phase.  Cold process wall and warm measured sequence remain distinct.  A T1
comparison with OpenVINO requires the same B580 identity
(`86800be2-0000-0000-0300-000000000000`), exact inputs/settings, byte-level
raw-output review, and comparable quality.  No production edit or v0.1.1
commit is part of this source-only leaf; FlashVSR follows only after that
milestone.

## Local validation and ownership

Root should inspect all four files and launch any runtime.  The local test is
CPU-only and does not import the frozen backend:

```text
python3 -B tests.py
python3 -B source-only.py
```

The second command is metadata mode and must report `PASS_SOURCE_ONLY` with
zero Torch imports, Core calls, device queries, model/payload reads, and DiT
forwards.  `source-v1.py` preserves the pre-fix candidate for review only; it
is not a runtime candidate.  The owned paths are this `PLAN.md`,
`source-only.py`, `source-v1.py`, and `tests.py`.
