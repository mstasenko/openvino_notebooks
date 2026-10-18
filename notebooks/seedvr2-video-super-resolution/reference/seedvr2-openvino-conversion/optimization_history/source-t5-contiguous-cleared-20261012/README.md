# Contiguous T5 with measured public clears

Status: source-only adapter and fake CPU contract tests.  No Torch, model,
fixture payload, OpenVINO Core, device query, build, or GPU execution occurs
in this leaf.  There is intentionally no CMake target: the adapter is Python
only and reuses the authenticated frozen backend rather than copying it.

## Composition

`cleared_adapter.py` adds two narrow pieces around the frozen
`source-t5-contiguous-next-20261012/contiguous.py` backend:

* `FixedAdmission` validates the complete physical/allocator snapshot.  The
  one cold call starts at the existing 12,400,000,000-byte body gate and its
  allocator ceiling.  A warm replay starts from the measured persistent
  post-cold floor and therefore uses a separate same-B580/kernel-first
  physical observation: at least 8,290,881,536 bytes free and zero allocator
  live I/O.  Reapplying the cold-idle gate to warm would reject every replay.
  Before DiT activation both calls keep the 8,290,881,536-byte setup gate and
  zero allocator requirement.  After activation plus a public shallow clear,
  the exact B580 size, allocated/reserved pair `122,705,408/140,509,184`, and
  the fixed measured increment `10,435,710,976 + 512 MiB = 10,972,581,888`
  free are required.  It never computes a usage-subtracted observed-peak
  expression.
* `ContiguousClearedSequence` takes an `expected_mean_sha256` scalar (default
  `cb448dc98de0fe6e5aab8c1191987cab871c05eb0824daacd9eb54511b73b13a`) and,
  after the mean is preserved and the frozen backend validates it, hashes the
  owned contiguous bytes and requires that exact digest.  Producer/checkpoint
  provenance remains the root-owned authenticated source/fixture authority;
  this adapter does not invent or trust a producer string on a payload.
* `ContiguousClearedSequence` is an explicit two-call `cold -> warm` sequence:
  inputs are prepared before the whole-call clock; VAE encode/preserve/validate
  and CPU release; setup gate; DiT activation; one public shallow clear and
  synchronize; post-setup gate; DiT forward with the authenticated N32
  block-clear adapter; preserve/validate/release; VAE decode/preserve/validate/
  release.  The CPU VAE object is retained between stages and calls; no owner
  drop or reload is introduced.

The sequence does not reuse the frozen `ContiguousSequence`, whose old gate
method is intentionally not called.  It performs no exception cleanup, retry,
fallback, or second attempt.  A failed gate or validation leaves the preserved
raw output and active state for the root-owned outer receipt to report.
Per-block clear callbacks record only in-memory count/time; they do not write
filesystems or hash outputs.  Archive/hash work is an injected callback after
the whole-T5 clock and is explicitly outside that clock.

For a future root-owned runtime integration, load and verify the frozen
backend, then call `make_frozen_backend` (not run here).  It invokes the
frozen `_torch_backend(spec, ref, block_adapter, reset, gate, bdf)` with a
`BlockClearAdapter` whose clear callback invokes the existing public
`clear_memory(debug=..., deep=False, force=True, timer_name=...)` and whose
synchronize callback invokes the existing XPU synchronize primitive.  The
frozen `spec.py`, `child.py`, reference loader, and `rope_reset.py` remain the
only math/weight/shape authorities.

## Authenticated source closure

The adapter pins and verifies these exact source files before any future real
backend construction:

```text
source-t5-contiguous-next-20261012/contiguous.py
  481d3720bbd644e4c14d18af20a07bd98c5f43dc469cd49bca7594da3b50edbc
temporal-uhd-complete-source-20261012/spec.py
  62b0624561ff49e674989533ef29d97cc1e21fd27377e4ab2f32f826536a318f
temporal-uhd-complete-source-20261012/child.py
  5733e1ed600f885562c96bdcd0ae241b7e149dfa54a76fa487600d0ae6f662ab
temporal-uhd-complete-source-20261012/benchmark.py
  a23929ce0a8b079c4cfb0c414c6d109fbdc62d323913d9af8137540114da175c
source-contiguous-t1-20261012/driver_v2.py
  f36433c131ccf21c4a512fab0de079b961364ce6f6d97c4f4b1302bd4cda86c3
temporal-uhd-n32-rope-reset-20261012/adapter.py
  4385b5b629ab714d664adcdc112806219fc8d966edef7fd64f25a981d7f0b8b4
temporal-uhd-n32-rope-reset-20261012/rope_reset.py
  ebea1b765f21e033dccb6f378b1641106d6b4854859a1d41ffe46d03932609c
torch-n32-block-clear-diagnostic-20261012/adapter.py
  644e423621b15caa2b3cfcd0e71d50c4dc511e496dcf937a8db5ef2a79b5ca8b
```

The frozen block-clear adapter remains the source of the exact 32-block
wrapper behavior; this leaf only supplies callbacks and enforces the observed
count.  The old stock `51.201711896`-second contiguous phase floor remains a
separate target.  Public clearing overhead is measured separately and is not a
performance or PyTorch-parity award.

`Synthetic` and `validate_payload` exist only for the stdlib contract tests.
The real backend keeps the frozen runtime's output/shape/finite validation and
does not use the synthetic helper.

## CPU contract test

`source_tests.py` uses only the standard library and synthetic payload objects.
It checks source pins, the one-time cold gate plus persistent-floor warm gate,
the explicit mean digest after preservation/validation, retained CPU VAE ownership,
setup/post gates, exact 32-block callback count, replay identity, malformed
allocator schema/size, low-floor refusal before warm VAE/forward, and
preservation of wrong-shape/nonfinite outputs before a natural stop.

Run only the bounded source test command:

```text
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B source_tests.py
```

