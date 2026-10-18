# Official SeedVR source snapshot

This tree is the non-generated source portion of ByteDance-Seed's official
`SeedVR` checkout, revision `e4de8c24441a67e1b7df56abea10645059bb1185`.
The sparse source manifest and original `LICENSE` are preserved beside it as
`OFFICIAL-SOURCE-MANIFEST.json` and `LICENSE-APACHE-2.0`.

It contains the official model/configuration/inference implementation only;
checkpoints and embeddings were deliberately excluded.  The official runner
uses CUDA/NCCL/FlashAttention and is not itself an OpenVINO exporter.  The
OpenVINO conversion history lives in the sibling `../conversion_history/`
directory under the separate GPL-3.0-only research boundary.

