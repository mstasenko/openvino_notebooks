# Selected late optimization source

These are source-only, historical T1/T5/residency/VAE optimization helpers
from the final `resume-20261010` work.  They are preserved separately from the
conversion scripts because they explore controller and memory contracts, not
the official model implementation.  No lifecycle directories, private logs,
GPU receipts, checkpoints, footage, generated graphs, or absolute host paths
are included.

The copied plans and tests retain their original source references and hashes
where useful, but remain historical experiments.  They do not silently become
part of the native worker: the current native C++ T1/T5 behavior and its
license boundary are documented by the parent README.  The temporal source
closure is included to make the residency/streaming design reviewable; it is
not an arbitrary-resolution or long-video quality qualification.

The small source dependencies referenced by those experiments are also kept
here: the contiguous T1 driver and RGB artifact helpers, the VAE owner-drop
overlay, the N20/N32 block-placement and RoPE helpers, the T5 reference
loader, and the phase-admission contracts.  Only source and contract code was
copied.  The historical source-pin manifests, model/checkpoint files, UHD
fixtures, generated outputs, receipts, logs, and lifecycle directories remain
excluded.  Consequently, scripts whose original contract authenticates one of
those omitted inputs intentionally remain metadata-only until a caller supplies
fresh external inputs and pins; this is not presented as a reproducible GPU
qualification.

The source-only contract suites are useful without those payloads: the cleared
T5 controller suite (16 tests), VAE owner-drop setup suite, and the small
source/AST checks run without Torch, OpenVINO Core, or device access.  The
dynamic conversion contracts likewise pass their source-only cases.  A few
historical tests intentionally fail closed when they look for an omitted
`ROOT-CHECK.json`, production XML/BIN, or UHD fixture; those failures are
evidence that the private input is absent, not a substituted pass.

