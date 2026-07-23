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
`mtmd` (libmtmd). Default = CPU core.

## License

Licensed under either of [Apache-2.0](https://github.com/replikeit/ik-llama-cpp-rs/blob/main/LICENSE-APACHE)
or [MIT](https://github.com/replikeit/ik-llama-cpp-rs/blob/main/LICENSE-MIT) at your option. Bundles
ik_llama.cpp (MIT).
