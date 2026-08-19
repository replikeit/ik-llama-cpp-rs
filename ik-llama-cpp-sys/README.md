# ik-llama-cpp-sys

[![Crates.io](https://img.shields.io/crates/v/ik-llama-cpp-sys.svg)](https://crates.io/crates/ik-llama-cpp-sys)
[![Docs.rs](https://docs.rs/ik-llama-cpp-sys/badge.svg)](https://docs.rs/ik-llama-cpp-sys)

Low-level FFI bindings to **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** (ikawrakow's
SOTA-quantization fork of llama.cpp), generated with bindgen. `links = "ik_llama"`.

**You almost certainly want the safe wrapper [`ik-llama-cpp-2`](https://crates.io/crates/ik-llama-cpp-2)
instead of this crate directly.**

The ik_llama.cpp source is vendored, so `cargo add ik-llama-cpp-sys` needs no submodule step — but
the build compiles it from source, so **CMake**, a **C/C++ toolchain**, and `libclang` (for bindgen)
are required. A prebuilt library can be linked instead via `IK_LLAMA_CPP_LIB_DIR` (+ `IK_LLAMA_CPP_SRC`
for headers).

Features: `cuda`, `vulkan`, `openmp`, `native`, `common` (ik `common/` + the MTP/json-schema glue),
`mtmd` (libmtmd). Default = CPU core, with OpenMP on (off on macOS; `IK_LLAMA_OPENMP=0` disables).

The default x86 ISA baseline is deliberately conservative (AVX2/FMA/F16C — **no AVX-VNNI**, so ik's
iqk int8 kernels take their generic path). Opt in per-ISA with `avx_vnni`, `avx512`, `avx512_vbmi`,
`avx512_vnni`, `avx512_bf16` — chosen for the oldest machine the artifact will run on, since a
missing ISA means SIGILL. `native` is host-only and not portable. See the
[workspace README](https://github.com/replikeit/ik-llama-cpp-rs#cpu-isa-baseline-read-this-before-building-for-distribution).

## License

Licensed under either of [Apache-2.0](https://github.com/replikeit/ik-llama-cpp-rs/blob/main/LICENSE-APACHE)
or [MIT](https://github.com/replikeit/ik-llama-cpp-rs/blob/main/LICENSE-MIT) at your option. Bundles
ik_llama.cpp (MIT).
