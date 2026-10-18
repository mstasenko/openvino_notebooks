# Third-party notices for the SeedVR2 subtree

The files under this directory are a mixed-license extraction. The copied
OneTrackCat worker and its pack scripts are GPL-3.0-only; see
`LICENSE-GPL-3.0-only`. The top-level OpenVINO Notebooks Apache-2.0 license is
not a substitute for that file.

The complete byte-preserved notices are in `assets/seedvr/notices.zip`. The
important upstream obligations recorded there and in `assets/seedvr/PROVENANCE.md`
are:

* OpenVINO Runtime and its oneDNN/TBB notices: retain the corresponding Apache
  and runtime notices when redistributing a built runtime.
* OpenSSL: retain the OpenSSL license/Ubuntu notice for the native pack's
  cryptographic dependency.
* ByteDance-Seed/SeedVR positive conditioning: Apache-2.0 attribution; this
  repository does not redistribute the original learned SeedVR checkpoint.
* Practical-RIFE v4.25 / packed-7 model: MIT attribution; the pinned RIFE IR is
  retained only in `assets/seedvr/rife-ir.zip`.
* OneTrackCat's GPL notice: retained in both the explicit license file and the
  pack notices. Do not relicense the copied native worker as Apache-2.0.

Before publishing a runtime or model pack, unpack and ship the exact notices
required by the resulting payload. The structural archive alone is not a
learned-weight license grant, and external checkpoint downloads remain subject
to their upstream terms.
