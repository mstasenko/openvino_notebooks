# Conditional fresh UHD phase-major source control

Prepared only. Root owns execution, fresh B580 confinement/admission and natural postflight. Frozen parent `source-phase-major-uhd-plan-20261011` is unchanged; its evidence and preparation failure remain authoritative. This additive variant changes only the pre-DiT admission guard and adds reviewed root launch wrappers. Source BF16 math, original FP16 weights, SDPA, tile 512/64 order, runtime allocator defaults, source cleanup and quality gates are unchanged.

## Exact guards

After exact BDF/UUID resolution and runtime identity initialization, the DiT child records Torch allocated bytes and exact driver `vram_mm` contents. It requires allocated <= 268,435,456 B and driver free >= 12,400,000,000 B. Malformed/duplicate fields, read errors, invalid allocation values and insufficient free space are rejected; current observations are persisted before rejection. A receipt-write failure stops before weight upload. The old 9 GiB parser is not called.

Root wrapper requires a supplied approved boot ID/kernel cursor, exact card2/renderD130 BDF/vendor/device/xe identity, empty clients and no holders/query errors, no new B580 fault/reset, physical driver free >= 12,750,000,000 B and MemAvailable >= 24 GiB. The bounded scope is 18 GiB/noSwap/CPU400%, strict B580 nodes plus standard pseudo-devices. The inside helper verifies actual cgroup limits and denied other GPU opens before starting the source child. All planned result paths must be absent, including symlinks.

These are conditional current-observation gates, not physical peak guarantees or allocator reservations. Prior isolated UHD DiT completed, but its Torch peak reservation was 12,559,843,328 B; that counter is not exclusively physical VRAM. Small fresh pre-DiT usage was 312,717,312 B versus primed 1,271,590,912 B. Removing that history is evidence supporting this controlled question, not proof that four UHD calls fit. Past source OOM/kernel resets remain failures; no retry is included.

## Timing and lifecycle

Fresh encode, DiT and decode processes each reuse only their own stage model for warmup02 then real distinct frames01/02/03. All existing RGB/noise/text bytes are authenticated; no media or model conversion is added. Whole wall charges startup, authentication/loading, warmup, all calls, transfers, packing, raw handoffs, quantization and natural cleanup. It is not a streaming online worker and steady stage sums are not a whole-video denominator. Complete raw payload is 597,196,800 B plus 1,536 B of NPY headers. Quality and matched speed awards remain pending.

On child failure, the supervisor does not start the next stage or make GPU calls in its exception handler. The in-scope shell saves natural child status plus final cgroup current/peak/events/stat/PSI before root postflight can reject. Root scope/tee status is preserved first; fault handling has no GPU runtime calls, timeout, signals or retry.

## CPU validation and seal

Seven bounded CPU contracts cover inherited geometry/packing/timing/fault behavior, exact fixture authorities, actual reshape adaptation, new threshold equality/one-byte failure/schema/read/persist failure, and the actual shell busy guard with fake fuser outcomes. Shell syntax and fixture check require no Torch import, checkpoint materialization, root action or device enumeration.

    TRIAL/venv/py311/bin/python -B test_contracts.py
    TRIAL/venv/py311/bin/python -B benchmark.py --check
    /usr/bin/python3 -B seal.py verify-local

Additive seal inherits the frozen full runtime authority. Preparation does not rehash multi-GB checkpoints. Full `seal.py verify` runs in the CPU supervisor before source model/device work and outside the disclosed benchmark clock.

## Root-only reviewed command

After root reviews all files and approves current boot/cursor and admission:

    /bin/bash run-root.sh APPROVED_CURRENT_BOOT_ID APPROVED_KERNEL_CURSOR

Fixed output is this leaf's new `gpu-attempt1/`, log `gpu-attempt1.log`. Human escalation at 600 seconds is observational, never a forced timer. The outer wrapper and strict scope have only CPU syntax/guard validation in this variant until root actually runs them. No GPU action occurred during preparation.
