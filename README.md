# ik-llama-cpp-rs

Safe Rust bindings for **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** (ikawrakow's
SOTA-quantization fork of llama.cpp), matching the API and codestyle of the production crate
**[llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)** (`llama-cpp-2` / `llama-cpp-sys-2`).

Use this instead of `llama-cpp-2` when you need ik's SOTA quant runtime (IQ*_K, IQ*_KT, R4/R8
repacks, BitNet, …) or its Multi-Token-Prediction (NextN) path. Production code swaps
`use llama_cpp_2` → `use ik_llama_cpp_2`.

- `ik-llama-cpp-sys` — low-level FFI (bindgen + CMake/prebuilt build). `links = "ik_llama"`.
- `ik-llama-cpp-2` — safe wrapper. Submodule pinned at `ikawrakow/ik_llama.cpp @ 9d07d868`.

> ik and stock llama.cpp export the same `llama_*`/`ggml_*` symbols and have incompatible ggml
> ABIs — **never link both into one process** (feature-gate one runtime per binary).

## Build

Two modes (Linux CPU + CUDA supported in v1):

**Prebuilt fast-path** (link an existing ik build, skip CMake):
```bash
export IK_LLAMA_CPP_LIB_DIR=/path/to/ik_llama.cpp/build   # has libllama.so, libggml.so, common/libcommon.a
export IK_LLAMA_CPP_SRC=/path/to/ik_llama.cpp             # headers for bindgen
export LD_LIBRARY_PATH="$IK_LLAMA_CPP_LIB_DIR/src:$IK_LLAMA_CPP_LIB_DIR/ggml/src:$LD_LIBRARY_PATH"
cargo build -p ik-llama-cpp-2
```

**From source** (CMake builds ik with `-DGGML_MAX_CONTEXTS=2048`):
```bash
export IK_LLAMA_CPP_SRC=/path/to/ik_llama.cpp   # or use the vendored submodule
cargo build -p ik-llama-cpp-2                    # CPU
CUDACXX=/opt/cuda/bin/nvcc CUDAARCHS=89 PATH=/opt/cuda/bin:$PATH \
  cargo build -p ik-llama-cpp-2 --features cuda  # CUDA (NCCL disabled; single-GPU)
```

### Features / backends
Drivers: `cuda`, `vulkan`, `metal` (Apple/macOS; no-op off-macOS), CPU (default). Plus `openmp`,
`native` (host-CPU tuning), `static-stdcxx`, `dynamic-link`, and `common` (builds ik `common/` + the
MTP glue). Prebuilt sharing (ik's equivalent of a "system ggml") is the `IK_LLAMA_CPP_LIB_DIR` link
path — ik has no `LLAMA_USE_SYSTEM_GGML` since its ggml carries the SOTA-quant kernels.

> **`GGML_MAX_CONTEXTS=2048`** is set for all CMake builds: ik allocates one `ggml_context` per GGUF
> shard, so loading a split set of >64 shards fails at shard 65 with the default cap of 64. (Only
> matters for many-shard split sets / offline merges; single merged files are unaffected.)

## Models (Thireus SPECIAL_SPLIT)

Thireus repos ship a per-tensor *distribution* format, not run-ready files. Prepare runnable GGUFs:
```bash
scripts/prepare_models.sh <ik_build_bin> <general-00001-of-*.gguf> <mtp-00001-of-*.gguf> .models
# -> .models/general.gguf  (merged)   and   .models/mtp-combined.gguf  (general + NextN half)
```
(MTP needs a *single combined NextN model*: ik MTP is embedded, not a `-md` sidecar.)

## Examples

```bash
cargo run -p simple -- --model .models/general.gguf --prompt "The capital of France is" -n 32
cargo run -p mtp    -- --model .models/mtp-combined.gguf -n 16          # validates n_nextn_layer>0 + gen
```

## Test

```bash
IK_TEST_MODEL=.models/general.gguf cargo test -p ik-llama-cpp-2 --features _smoke -- --test-threads=1
```

## Status (verified)

| Area | State | Notes |
|---|---|---|
| General (load→tokenize→decode→sample) | ✅ CPU + CUDA | |
| MTP: NextN load + activation + full speculative driver | ✅ CPU + CUDA | `MtpSpeculative` new→begin→step; `common` feature; wraps ik `common_speculative_*` incl. the recurrent checkpoint path |
| `gguf` metadata, `timing`, `context/kv_cache` | ✅ | Batch A |
| `context/session`, `context/embeddings`+reranker, `model/{chat,lora,meta}` | ✅ | Batch B |
| `sampling` — chain-object `LlamaSampler` (greedy/dist/temp/temp_ext/top_k/top_p/min_p/typical/top_n_sigma/tail_free/penalties/grammar/grammar_lazy + `chain_simple`/`apply`/`accept`/`sample`) | ✅ CPU + CUDA | matches the `llama-cpp-2` chain API; emulated over ik's null-ctx-safe legacy `llama_sample_*` + a seeded Rust `dist` draw; grammar stages via a core ctx-free glue |
| `grammar` (GBNF grammar + DRY stateful samplers) | ✅ CPU + CUDA | `LlamaGrammar` (lifetime-tied to model) + `LlamaDrySampler`; smoke-tested on-device (33/33 layers offloaded) |
| `json_schema_to_grammar` (JSON Schema → GBNF, for tool/function calling) | ✅ CPU + CUDA | `common` feature; C-glue over libcommon's converter (mirrors the anchor); feeds `LlamaGrammar::new` |
| `llama-cpp-2` drop-in surface (0.1.x) | ✅ non-MTP, CPU + CUDA | `tests/dropin_surface.rs` exercises the `llama-cpp-2` API a downstream consumer uses; only the granular `MtpSpeculative` API differs (ik's embedded recurrent MTP vs the fork's 2-context driver — ik uses the higher-level `new→begin→step`) |
| `mtmd` (multimodal) | ✅ builds/links; no-model tests | `mtmd` feature; full vision path needs a vision GGUF + mmproj fixture (not present) |
| `quantize` write-path | ✅ builds/links; params tests | full requantize needs an f16 source GGUF (not present) |
| lifecycle / negative / tokenizer tests | ✅ | |

Every module was smoke-tested and passed an **independent opus code review** (all findings fixed:
general, MTP, Batch A+B, mtmd, grammar+dry, chain-sampler, grammar-sampler glue). No known memory-safety UB; FFI signatures verified against the ik
headers; MTP faithful to native `--spec-type mtp`. Notably, the mtmd review fixed a `Clone`
double-free + a borrowed-chunk UAF that are latent in the upstream `llama-cpp-2` anchor too — so this
crate is more correct there.

**Deferred:** the vision + full-requantize smoke tests (blocked on model fixtures); CI.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Bundles [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
(MIT) as a submodule. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in this crate, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or conditions.
