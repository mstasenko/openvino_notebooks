// Fixed release provenance and authenticated payload bytes, not writable manifest authority.
export const releasePins = {
  "openssl": {
    "version": "3.5.5-1ubuntu3.7",
    "source": "openssl",
    "distribution": "Ubuntu",
    "artifactSha256": "3ff73e121f6cd4f19e80ab4329dfd914108b3936c9db3708200203a9b245b34d",
    "url": "https://us.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl3t64_3.5.5-1ubuntu3.7_amd64.deb",
    "glibcMinimum": "2.38"
  },
  "runtime": {
    "version": "2026.3.0",
    "artifactSha256": "93e111164ccfd5c801edcfc3437683210837463bf57a2bd0518b940598664c2c",
    "artifact": "openvino-2026.3.0-22451-cp311-cp311-manylinux_2_28_x86_64.whl"
  },
  "positiveText": {
    "repository": "ByteDance-Seed/SeedVR",
    "revision": "e4de8c24441a67e1b7df56abea10645059bb1185",
    "source": "pos_emb.pt",
    "sourceSha256": "fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799",
    "sha256": "124724ab11176f0143f9f4975a5e784dade59fc04d762948361bb36d23e0ff49",
    "licenseSha256": "5967a0c153e731109b76c5d83b5e0fae2382b6aaf7b898d08ff03702eea59430",
    "recipe": "BF16-LE-u16-to-FP32-LE-u32-shift16"
  },
  "rife": {
    "variant": "v2-packed7",
    "sourceSha256": "65c57a5e4abb17ad67faf35054291ac53affab0506d509d2e66698f6ecd75584"
  }
}
export const payloadPins = {
  "lib/libcrypto.so.3": { "bytes": 6386528, "sha256": "5385f0436ac2e284ba8c5fb1488874057c29c04be72dea32add832694f381e28" },
  "lib/cache.json": { "bytes": 8840377, "sha256": "0b2ad2a99516a3411ce3fef274e8ae2880b41a6f00c9bac6768cd4a94b87c41c" },
  "LICENSE-OPENSSL.txt": { "bytes": 11358, "sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30" },
  "NOTICE-OPENSSL-UBUNTU.txt": { "bytes": 2543, "sha256": "6a7da622fe0637a334d2a8fc470852d2ffb77d9a2b2f930f854e32a41ad6ef35" },
  "lib/libhwloc.so.15": {
    "bytes": 471657,
    "sha256": "57f2f1875fb301ad493bff365cb355220a8cb28bb75d76c5584c411811906b4f"
  },
  "lib/libopenvino.so.2630": {
    "bytes": 20124281,
    "sha256": "90eea88b2de0b904e6ee205e3a1768d32615359000c1c257c22068585fb32389"
  },
  "lib/libopenvino_intel_cpu_plugin.so": {
    "bytes": 64276761,
    "sha256": "e7f2d023754a2a5ad10438faa875f23f257e22d531a9052a96c52f0f5f10a498"
  },
  "lib/libopenvino_intel_gpu_plugin.so": {
    "bytes": 40994545,
    "sha256": "4eba241070b0c262c6df72dbd8e3686485370bdde440a3d605aba2510029035c"
  },
  "lib/libopenvino_ir_frontend.so.2630": {
    "bytes": 600273,
    "sha256": "d90a1ac8ef689d90b3e76b534c9ba3d79fbe289e27d162ae9f9cfc38ea6f336c"
  },
  "lib/libtbb.so.12": {
    "bytes": 367649,
    "sha256": "6781dc6aa8b2ca237d0164258e70266a44667756c02a9a039782ec1d1a67b653"
  },
  "lib/libtbbbind_2_5.so.3": {
    "bytes": 31361,
    "sha256": "baec0c4299339170eb55ae905b57efc5b89a1e10f8780fb9e9e27acb5446f341"
  },
  "lib/libtbbmalloc.so.2": {
    "bytes": 158129,
    "sha256": "d7166b1e54c17d8f25be63e22fc2366bbe7b1b8cc3a7ea629dea59caeba9c020"
  },
  "lib/libtbbmalloc_proxy.so.2": {
    "bytes": 26713,
    "sha256": "ee6b319fd28005147b40a6d5a152218998ea6eda54e13f7e8e72ca462ce6797a"
  },
  "rife.xml": {
    "bytes": 335754,
    "sha256": "7b5e8df0dc32b206569b81c1e69a65269ece571420005a989679dd523f8d0fe3"
  },
  "rife.bin": {
    "bytes": 22643376,
    "sha256": "c166cc6df78e76a50950c26988c64c30dc33e1e5867a3ebcb4cb4e630a8afaa9"
  },
  "LICENSE-SEEDVR2.txt": {
    "bytes": 11334,
    "sha256": "5967a0c153e731109b76c5d83b5e0fae2382b6aaf7b898d08ff03702eea59430"
  },
  "LICENSE-RIFE-MIT.txt": {
    "bytes": 1066,
    "sha256": "f629619f078e3ed699f1ea401248cb7796c36334e3b8ec0eb2c082eceaa5f084"
  },
  "LICENSE-OPENVINO.txt": {
    "bytes": 11357,
    "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
  },
  "NOTICE-OPENVINO-ONEDNN.txt": {
    "bytes": 28475,
    "sha256": "f1b16991dfe8488f3f5f07cccba575c31c91175a9f2c8d9e1569665f86e0b2f6"
  },
  "NOTICE-OPENVINO-TBB.txt": {
    "bytes": 46979,
    "sha256": "b10997cc50652ba6ab365e3bc43e13057681cd4a56533b46171529bd249749f3"
  },
  "NOTICE-OPENVINO-RUNTIME.txt": {
    "bytes": 99330,
    "sha256": "85481d0f8a01308fb8c837e0007897ab5f35c90b6ee6333e6e6dfeff4ca3f46c"
  }
}
export function matchesReleaseProvenance(record) {
  return Object.entries(releasePins).every(([group, values]) =>
    Object.entries(values).every(([key, value]) => record?.[group]?.[key] === value))
}
export const modelNotice = 'SeedVR2 positive conditioning: ByteDance-Seed/SeedVR e4de8c24441a67e1b7df56abea10645059bb1185 pos_emb.pt; Apache-2.0, copyright 2025 seed. Modified only by deterministic BF16-to-FP32 bit expansion. RIFE v4.25: hzwer/Practical-RIFE de9a989bb7b8a71d94f058297e603633aaa43ad6, MIT; vs-mlrt packed-7 ONNX. Original training checkpoint and exporter revision are not retained.'
