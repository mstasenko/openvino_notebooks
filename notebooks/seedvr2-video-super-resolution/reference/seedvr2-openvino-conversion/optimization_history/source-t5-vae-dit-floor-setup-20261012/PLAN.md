# Setup-only VAE-to-DiT floor measurement plan

This is a source-only proposal. It does not import Torch/Core, open a model or
fixture payload, query a device, build code, or change either frozen producer.
It answers one narrow question: after one real VAE encode has been preserved
and its CPU owner dropped, is the N32 DiT setup floor shared with that VAE
floor or additive? No DiT prediction is part of this experiment.

## Frozen source seams

The proposal reuses the existing source seams without copying their runtime
implementation:

- `source-t5-contiguous-next-20261012/contiguous.py`, 40,611 bytes,
  SHA-256 `481d3720bbd644e4c14d18af20a07bd98c5f43dc469cd49bca7594da3b50edbc`.
  `_torch_backend()` owns the reviewed streamed-DiT construction,
  `activate("dit")` configures 32 CPU blocks plus XPU I/O, and
  `release("dit")` performs the existing CPU move, synchronization, deep
  public clear, and allocator-zero assertion.
- `source-t5-vae-owner-drop-diagnostic-20261012/drop_reload.py`, 10,522 bytes,
  SHA-256 `1506886aef1015ec84ec485f74b9cc8f19f25f6916266ad797e92c27290cd500`.
  `DropReloadBackend.drop_cpu_vae_owner()` verifies CPU/non-meta ownership,
  drops the VAE owner, runs the reviewed public deep clear and sync, and
  requires zero live GPU allocator bytes.

No new math, placement policy, allocator API, gate, or output schema is
proposed. The future root-owned diagnostic should call these existing methods
directly rather than `ContiguousSequence.run()` or `DropReloadSequence.run()`.

## One finite setup-only sequence

In one fresh root-controlled process, after the normal source/model/fixture
authority and unchanged old body gate have passed:

1. Record the filesystem VRAM sample and `backend.allocator()` counters at
   idle. Activate VAE once, run exactly one original `encode`, preserve and
   validate the mean with the frozen shape/dtype contract, then release VAE.
   Archive the mean using the existing producer convention; preserve its raw
   bytes/hash for later source-identical comparison.
2. Invoke the existing `drop_cpu_vae_owner("setup")` seam. Record the
   post-VAE-drop physical sample and allocated/reserved counters. Require the
   existing zero-allocator result and retain the CPU-owner/lifetime evidence.
3. Activate DiT once. This loads/configures the existing streamed N32 model,
   applies the existing 32-block CPU placement, and synchronizes. Do **not**
   call `backend.dit`, construct a prediction, or run any DiT forward. Record
   the setup physical sample, allocator counters, and the exact placement
   result.
4. Release DiT through the frozen `release("dit")` path, including its normal
   cache cleanup, CPU movement, deep clear, sync, and allocator-zero check.
   Clear only CPU references after the final observation, run the ordinary
   garbage collection path, record the final physical/allocator counters, and
   exit naturally. There is no retry or forced exception teardown.

The receipt should retain finite natural statuses and, for every phase, the
filesystem `vram_mm` raw sample plus `memory_allocated` and
`memory_reserved`. It should explicitly record `dit_forward_calls=0`, one VAE
encode, one DiT setup, and the mean shape/dtype/hash. A failure preserves the
partial evidence and stops; it does not attempt a prediction to complete the
receipt.

There is no launch command at this source-only stage. If approved, root should
run one fresh bounded scope with the trial `venv/py311/bin/python -u -B` and a
reviewed setup-only entrypoint derived from these two frozen modules. It must
not invoke `contiguous.py --run`, because that entrypoint intentionally calls
the DiT forward; the setup-only entrypoint must expose the lifecycle above and
record its evidence in a new root-selected attempt leaf.

## Shared-versus-additive calculation

Use the already closed stock setup measurement supplied with this task:

```text
closed setup maximum before clear = 7,242,317,824 bytes
fresh idle baseline               =    25,178,112 bytes
setup growth estimate             = 7,217,139,712 bytes
post-VAE floor                    = 1,164,189,696 bytes
```

With a deliberately explicit 1-GiB planning margin, the conservative setup-only
additive estimate is:

```text
1,164,189,696 + 7,217,139,712 + 1,073,741,824
= 9,455,071,232 bytes
```

Against the recorded 12,809,404,416-byte device size, that leaves
3,354,333,184 bytes. This is only a planning estimate, not an admission or a
prediction authorization. The actual setup run must compare:

- idle physical usage (`I`);
- post-VAE-owner-drop floor (`V`);
- maximum physical usage during DiT setup (`D`); and
- post-DiT-release usage and allocator counters (`F`).

`D - I` is the measured total setup growth, while `D - V` shows the observed
increment in the presence of the released-VAE floor. A future independent body
run is required before interpreting that comparison as shared or additive.
If the complete body and DiT setup are ever concurrent, the additive worst case
must remain rejected unless a measured growth bound plus the explicit margin
fits. The existing body 12.4-GB gate is unchanged.

## Acceptance and limits

Acceptance is limited to: natural status zero; exact one-encode/zero-DiT-forward
markers; finite, source-identical mean; all required placement and owner
lifetime checks; zero allocator bytes after each healthy release; and complete
physical/allocator evidence. A separate fresh repeat and a full body repeat
must establish any production contiguous-prediction claim. This setup-only
measurement cannot award speed, quality, or body admission, and it must not
weaken the existing gate merely because the estimate fits.

The bounded CPU contract test uses a small signed sentinel in place of a real
mean payload; its digest check proves the preserve/validate ordering contract,
not a measurement of the authenticated runtime mean. The real setup receipt
must compute and compare the mean bytes after the authorized VAE encode.

