# Complete genuine UHD T5 BF16 source — preparation only

One finite three-process encode→DiT32→decode job, warmup/replay of the exact chronological five-frame fixture. This is not fiveT1 calls. Each DiT consumes its fresh corresponding encoder mean; each decoder consumes its corresponding fresh prediction. No fixed A/B prediction is injected into the chain.

Exact original BF16 loaders/forward, timestep1000, fixed native T2 noise/text, conditioning/restoration0.9152, 512×512 tiles/64 overlap, original BF16 VAE blending, unsliced T5/T2 memory-disabled policy and floor/clamp RGB conversion remain unchanged. Selected16 mean only; no unused variance readback/archive. Successful artifacts per call: mean16,588,800B + THWC prediction16,588,800B + fiveRGB124,416,000B =157,593,600B. Two calls total315,187,200B. Normal decoded F32 dump497,664,000B is removed; if decoded shape/finite validation prevents RGB, completed failed decoded CPU data is preserved before rejection.

DiT reuses qualified N32 synchronous move-only placement, once-after-placement healthy minimal clear, and validated CPU RoPE cache-length reset. Reset is explicitly a diagnostic stability recipe, not a proved root-cause fix or fastest-source configuration. Original coldB/warmA mismatch and temporal artifacts are not waived. Reset is charged inside owned-call time; cached original forward math and per-block movements/offloads remain unchanged. Healthy outer RoPE LRU cleanup remains separate and charged in phase/wholewall. No retry/fallback/fault GPU cleanup.

Runtime root preverify must authenticate the full inherited runtime/checkpoints: temporal-uhd-n32-rope-reset-run-20261012 execution closure6c16a71329ad567314b74462e4fcdc39601a85074e0a8a11b9160235fceb78eb. Child checkpoint size checks avoid redundant multiGB hashing inside phase clocks; this is not permission to launch without full root closure verification. SOURCE-PINS inherits existing fixture/checkpoint authority; CPU prep reads no payload or model tensors. Unused normalized fixture dump is not a runtime child hash dependency; normalization is recomputed from exact RGB with the unchanged formula.

Only one stage model per process. Fresh kernel-first snapshots precede every next forward and healthy DiT clear. Mandatory parent physical sampler/health precede every next child, including final decode health. Source pre-DiT admission remains12.4GB driver free and Torch allocated≤256MiB before upload; future outer scope remains root-owned exact B580,18GiB/noSwap/CPU400%, host24GiB/idle12.75GB. No source/model/device operation is authorized by this preparation.

Future root-reviewed command (not executed):

    T/venv/py311/bin/python -u -B P/benchmark.py --output P/gpu-attempt1

Children: child.py --phase encode|dit|decode --root P/gpu-attempt1. Exact absent direct root; no signals/timeouts/automatic retries, human600s observation. Recovery constants derive from the same pinned root-approved epoch as the qualified N32/decoder controls.

Report true whole_three_process_seconds separately from model_load_and_setup, per-call owned_input_to_owned_output, handoff read/hash, archive/hash/guards, healthy boundary cleanup and phase teardown. Source wholewall includes process startup, required handoffs, all archives, sampling/health and cleanup. Do not compare it directly with the native full32 one-process pilot whole_seconds or call a sum of warm stage actions end-to-end. Native first remains fit/quality; matched sustained performance is a later same-boundary test.

Quality first: retain bothwarm/replay complete RGB, require own replay/finite/shape, compare encoded mean/prediction/RGB to closed source controls separately, and inspect allfive frames/temporal differences/fixed input-selected crops. Stable A and anomalous B both decode with later-frame distortion in existing evidence; repeatability is not acceptable quality. No new threshold, temporal/speed/release award or original-failure erasure.

CPU only: qualified python -B run_cpu.py. Tiny arrays/AST/metadata, actual fresh-handoff corruption cases, no Torch/model/GPU or large payload materialization. This leaf is not frozen and has no GPU launcher.
