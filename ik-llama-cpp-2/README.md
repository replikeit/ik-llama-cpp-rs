# ik-llama-cpp-2

[![Crates.io](https://img.shields.io/crates/v/ik-llama-cpp-2.svg)](https://crates.io/crates/ik-llama-cpp-2)
[![Docs.rs](https://docs.rs/ik-llama-cpp-2/badge.svg)](https://docs.rs/ik-llama-cpp-2)
[![License](https://img.shields.io/crates/l/ik-llama-cpp-2.svg)](https://github.com/replikeit/ik-llama-cpp-rs)

Safe Rust bindings for **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** — ikawrakow's
SOTA-quantization fork of llama.cpp. Mirrors the API and codestyle of
**[llama-cpp-2](https://crates.io/crates/llama-cpp-2)**, so most code ports by swapping
`use llama_cpp_2` → `use ik_llama_cpp_2`.

Reach for this instead of `llama-cpp-2` when you want ik's SOTA quant runtime (IQ\*\_K, IQ\*\_KT,
R4/R8 repacks, BitNet, …) or its Multi-Token-Prediction (NextN) path.

- `ik-llama-cpp-2` — this crate: the safe wrapper.
- [`ik-llama-cpp-sys`](https://crates.io/crates/ik-llama-cpp-sys) — the low-level FFI (bindgen +
  CMake build). ik_llama.cpp is vendored, so `cargo add` works with no submodule step.

> ik and stock llama.cpp export the same `llama_*`/`ggml_*` symbols with incompatible ggml ABIs —
> never link both into one process.

## Requirements

The `-sys` crate builds ik_llama.cpp from source on first build, so you need **CMake** and a
**C/C++ toolchain** (plus `libclang` for bindgen). The initial build is heavy. Linux CPU + CUDA are
exercised in CI/smoke tests.

```toml
[dependencies]
ik-llama-cpp-2 = "0.1"
```

### Features

`cuda`, `vulkan`, `openmp`, `native` (host-CPU tuning), `common` (ik `common/` — enables the MTP
speculative driver + `json_schema_to_grammar`), `mtmd` (multimodal). Default = CPU core.

## Example

```rust,no_run
use ik_llama_cpp_2::{
    llama_backend::LlamaBackend, model::{AddBos, LlamaModel, params::LlamaModelParams},
    context::params::LlamaContextParams, llama_batch::LlamaBatch, sampling::LlamaSampler,
};
use std::num::NonZeroU32;

let backend = LlamaBackend::init()?;
let model = LlamaModel::load_from_file(&backend, "model.gguf", &LlamaModelParams::default())?;

let mut ctx = model.new_context(
    &backend,
    LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048)),
)?;

let prompt = model.str_to_token("The capital of France is", AddBos::Always)?;
let mut batch = LlamaBatch::new(prompt.len().max(64), 1);
batch.add_sequence(&prompt, 0, false)?;
ctx.decode(&mut batch)?;

// chain-object sampler, exactly like llama-cpp-2
let mut sampler = LlamaSampler::chain_simple([
    LlamaSampler::top_k(40),
    LlamaSampler::top_p(0.95, 1),
    LlamaSampler::temp(0.8),
    LlamaSampler::dist(1234),
]);
let mut idx = batch.n_tokens() - 1;
let mut decoder = encoding_rs::UTF_8.new_decoder();
for _ in 0..32 {
    let tok = sampler.sample(&ctx, idx);
    sampler.accept(tok);
    if model.is_eog_token(tok) { break; }
    print!("{}", model.token_to_piece(tok, &mut decoder, false, None)?);
    batch.clear();
    batch.add(tok, /* pos */ batch.n_tokens(), &[0], true)?;
    ctx.decode(&mut batch)?;
    idx = 0;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Structured output / tool calling: `json_schema_to_grammar(schema)` → `LlamaSampler::grammar(...)`
(the `common` feature). MTP (NextN) speculative decoding: `MtpSpeculative::new → begin → step`
(also `common`).

## License

Licensed under either of [Apache-2.0](https://github.com/replikeit/ik-llama-cpp-rs/blob/main/LICENSE-APACHE)
or [MIT](https://github.com/replikeit/ik-llama-cpp-rs/blob/main/LICENSE-MIT) at your option.
