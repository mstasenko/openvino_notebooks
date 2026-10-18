# Small true-T5 move-only BlockSwap preparation

Source-ready for root review, **unsealed and not GPU-admitted**. No UHD test,
Torch import, tensor materialization, device query, build or GPU work occurred.

`adapter.py` calls only the pinned upstream `_configure_blocks` and
`_wrap_block_forward`, selecting blocks0..15 with `blocks_to_swap=15`.
N0 returns without setting fields or wrapping/moving any block. The imported
upstream `Debug(enabled=False)` remains alive for the complete model lifetime;
upstream wrappers otherwise retain only weak model/debug references. Both I/O
and unselected blocks remain on XPU. No RoPE retry patch, CPU frequency fallback,
whole-model move protection, speculative precision change or new scheduler.

`control.py` reuses the original qualified meta/BF16 streamed DiT loader,
config/SDPA/fixture and exact original prelude-equivalent input packing. Genuine
five960×768 frames share the authenticated tiled BF16 encoder mean at
`temporal-native-prototype/xpu-t5-results/mean.npy` (SHA31f2071b...4e70f6),
the frozen THWC T2 noise and58×5120 text. It does NOT rerun VAE or substitute
synthetic inputs/five T1 calls. Mean receipt and file are hard-pinned. Source
revision is8d4f3c4bd854695bfcbd25e775b9ea5852ca5f2c.

SOURCE-PINS.json binds selected source/input bytes and the frozen parent runtime
authority `xpu-pins.json` SHA df7bc49d5f30c99ae3ccbae7f0f7057f68e471c544dae4b27f3861da5fbbbcad.
Additional observed files absent from that table are explicitly identified,
not mislabeled as inherited pins. Root must authenticate/pin this derivative
and verify the inherited runtime/checkpoint authority before launching. The
6.78GB checkpoint was not redundantly hashed or its payload read here; the
execution size/checkpoint-table check is not a substitute for root verification.

## Exact finite controls (root only, separate natural processes)

Use the trial py311 interpreter, strict B580-only scope/exact BDF+UUID health
preflight and fresh unused output paths. Commands after approved closure:

```
T/venv/py311/bin/python -u -B E/temporal-blockswap-small-20261012/control.py --blocks 0 --output E/temporal-blockswap-small-20261012/resident-attempt1
T/venv/py311/bin/python -u -B E/temporal-blockswap-small-20261012/control.py --blocks 16 --output E/temporal-blockswap-small-20261012/swap16-attempt1
```

Each process runs one warmup then identical replay. Both completed CPU F32
prediction[23040,16] and restored[1,16,2,96,120] arrays are saved exclusively
before finite/shape gates. Prediction is saved even if restoration reshaping
fails. Each file carries SHA/type/shape in receipt.json. Failure handler records
only CPU status: no query/cleanup/retry/next inference. Healthy teardown moves
the model to CPU and exits normally; root physical postflight is authoritative.
No counter/peak claim is made by this source-only harness.

Calls include CPU packing, uploads, every synchronous per-block move, original
forward, readback, restoration, disk archival and healthy RoPE LRU clearing.
Loading+swap setup is separate and charged in whole wall. This is an extension
of the historical single-call T5 probe to a two-call replay; the same healthy
per-call LRU clear used by the qualified phase-major reference is applied to
BOTH controls. The original forward Cache remains enabled. Upstream wrapper
pressure-clear policy is retained and its time included: XPU availability check
uses host RAM as a proxy, not physical B580 capacity. No altered allocator policy.

Predeclared later acceptance: both natural processes complete; all four arrays
per process have exact expected sizes/types/shapes and are finite; replay is
hash/byte-identical within each mode; the two modes' corresponding arrays are
full-byte identical. Preserve failed arrays/status. These are unchanged-math
move-only correctness gates, not a temporal image-quality or speed award.
Historical source prediction can be an additional bridge, not replace the new
matched N0 control. Any difference must be reported, not waived. Timing with16
swapped blocks is fit-diagnostic only, not the final fastest qualified denominator.

## Closed CPU preparation

`run_cpu.py` completed naturally0 under2GiB/noSwap/CPU200%/nice15, all three
commands0, source before/after identical. `cpu.stdout`, `cpu.stderr` and
CPU-RECEIPT.json retain exact commands/status. Five fake tests execute the ACTUAL
upstream function bodies via AST (no module import/Torch), proving placement,
field15, chronology/rebinding over two calls, N0 unchanged, invalid/preconfigured
rejection, and exception propagation with no later move/cache/next block/retry.
Both actual fixture checks passed via `control.py --blocks N --check` without
Torch import. These fake tests do NOT prove actual XPU transfer/buffer lifetime,
math equivalence, memory fit or driver health. Root owns those subsequent gates.
