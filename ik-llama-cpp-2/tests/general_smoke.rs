//! General-path smoke test: load → tokenize (roundtrip) → decode → sample.
//!
//! Gated behind the `_smoke` feature and the `IK_TEST_MODEL` env var (a single
//! merged GGUF, e.g. `.models/qwen35-4b-iq1s-general.gguf`). Asserts *workflow*
//! correctness, not text quality (IQ1_S output is expected to be degenerate).
#![cfg(feature = "_smoke")]

use ik_llama_cpp_2::{
    LlamaBackend, LlamaBatch, LlamaContext, LlamaContextParams, LlamaModel, LlamaModelParams,
    LlamaSampler,
};

fn model_path() -> String {
    std::env::var("IK_TEST_MODEL").expect("set IK_TEST_MODEL to a merged GGUF path")
}

#[test]
fn general_load_tokenize_decode_sample() {
    let backend = LlamaBackend::init().expect("backend init");

    let mparams = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, model_path(), &mparams).expect("load model");
    assert!(model.n_vocab() > 0, "vocab should be non-empty");

    // tokenize <-> detokenize roundtrip
    let toks = model.tokenize("Hello, world!", true).expect("tokenize");
    assert!(!toks.is_empty(), "tokenize produced no tokens");
    let text = model.detokenize(&toks).expect("detokenize");
    assert!(text.contains("Hello"), "roundtrip lost text: {text:?}");

    // context + prompt decode
    let cparams = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(2048))
        .with_n_threads(8)
        .with_seed(42);
    let mut ctx = LlamaContext::new(&model, &cparams).expect("context");

    let prompt = model
        .tokenize("The capital of France is", true)
        .expect("tokenize prompt");
    let mut batch = LlamaBatch::new(prompt.len().max(64), 1);
    batch.add_sequence(&prompt, 0, false).expect("add prompt");
    ctx.decode(&mut batch).expect("decode prompt");

    let last = batch.n_tokens() - 1;
    let logits = ctx.get_logits_ith(last).expect("logits");
    assert_eq!(
        logits.len(),
        model.n_vocab() as usize,
        "logits length must equal vocab size"
    );

    // greedy decode a handful of tokens
    let mut sampler = LlamaSampler::greedy();
    let mut n_past = batch.n_tokens();
    let mut logits_idx = last;
    let mut generated = 0usize;
    for _ in 0..8 {
        let tok = sampler.sample(&ctx, logits_idx);
        generated += 1;
        if model.is_eog(tok) {
            break;
        }
        batch.clear();
        batch.add(tok, n_past, &[0], true).expect("add token");
        n_past += 1;
        ctx.decode(&mut batch).expect("decode token");
        logits_idx = 0;
    }
    assert!(generated >= 1, "should have generated at least one token");
    println!(
        "GENERAL SMOKE OK: vocab={} prompt_tokens={} generated={}",
        model.n_vocab(),
        prompt.len(),
        generated
    );
}
