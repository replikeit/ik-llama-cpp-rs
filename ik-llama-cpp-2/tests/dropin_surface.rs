//! `llama-cpp-2` drop-in "usage mirror": exercises the `llama-cpp-2` API surface
//! a downstream consumer relies on, *except* the MTP speculative path (ik's
//! embedded recurrent MTP uses a higher-level `new→begin→step` driver rather
//! than the stock crate's granular two-context `MtpSpeculative` API).
//!
//! Compiling this file proves the non-MTP surface is drop-in; running it
//! (needs `IK_TEST_MODEL`) proves it works end to end. Gated behind `_smoke`.
#![cfg(feature = "_smoke")]

use std::num::NonZeroU32;

use ik_llama_cpp_2::context::params::LlamaContextType;
use ik_llama_cpp_2::context::session::LlamaStateSeqFlags;
use ik_llama_cpp_2::token::data::LlamaTokenData;
use ik_llama_cpp_2::token::data_array::LlamaTokenDataArray;
use ik_llama_cpp_2::{
    AddBos, LlamaBackend, LlamaBatch, LlamaContext, LlamaContextParams, LlamaModel,
    LlamaModelParams, LlamaSampler, LlamaToken,
};

fn model_path() -> String {
    std::env::var("IK_TEST_MODEL").expect("set IK_TEST_MODEL to a merged GGUF path")
}

fn n_gpu_layers() -> u32 {
    std::env::var("IK_TEST_NGL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// A consumer's model/context construction + generation loop (minus the MTP
/// driver): model params, context params (incl. fork-specific builders), batch,
/// the sampler chain + grammar gate, streaming detokenize, and kv-cache ops.
#[test]
fn dropin_non_mtp_surface_compiles_and_runs() {
    let backend = LlamaBackend::init().expect("backend");

    // --- model params + load ---
    let mparams = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers()); // u32
    let model = LlamaModel::load_from_file(&backend, model_path(), &mparams).expect("load");

    // --- token accessors ---
    let _eos: LlamaToken = model.token_eos();
    let _bos: LlamaToken = model.token_bos();
    let _tmpl = model.chat_template(None); // Option<&str> arg
    assert!(model.n_vocab() > 0);

    // --- tokenize via AddBos ---
    let prompt = model
        .str_to_token("Return JSON: ", AddBos::Always)
        .expect("str_to_token");
    let _never = model
        .str_to_token("x", AddBos::Never)
        .expect("str_to_token");

    // --- context params: every builder a consumer uses, incl. fork-specific ---
    let cparams = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_n_batch(2048)
        .with_n_ubatch(512)
        .with_n_threads(8)
        .with_n_threads_batch(8)
        .with_n_rs_seq(4)
        .with_context_type(LlamaContextType::Default);
    // model.new_context(&backend, params)
    let mut ctx: LlamaContext = model.new_context(&backend, cparams).expect("new_context");

    // --- batch prefill ---
    let mut batch = LlamaBatch::new(prompt.len().max(64), 1);
    for (i, &t) in prompt.iter().enumerate() {
        batch
            .add(t, i as i32, &[0], i == prompt.len() - 1)
            .expect("add");
    }
    ctx.decode(&mut batch).expect("decode");
    let last = batch.n_tokens() - 1;

    // --- token data array ---
    let arr = ctx.token_data_array_ith(last);
    assert_eq!(arr.data.len(), model.n_vocab() as usize);
    let _first: Option<&LlamaTokenData> = arr.data.first();
    let _built =
        LlamaTokenDataArray::new(vec![LlamaTokenData::new(LlamaToken(0), 1.0, 0.0)], false);
    let _sel: Option<LlamaToken> = arr.selected_token();

    // --- sampler chain (grammar gate + penalties + filters + dist) ---
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.8),
        LlamaSampler::dist(0),
    ]);
    // standalone grammar gate + lazy grammar
    let mut gate = LlamaSampler::grammar(&model, "root ::= [0-9]+", "root").expect("grammar");
    let _lazy = LlamaSampler::grammar_lazy(
        &model,
        "root ::= \"yes\" | \"no\"",
        "root",
        vec![b"<tool>".as_slice()],
        &[],
    )
    .expect("grammar_lazy");

    // --- generation loop (apply gate, apply chain, accept) ---
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_past = batch.n_tokens();
    let mut logits_idx = last;
    let mut text = String::new();
    for _ in 0..6 {
        let mut cand = ctx.token_data_array_ith(logits_idx);
        gate.apply(&mut cand); // grammar gate first
        sampler.apply(&mut cand); // then the chain (ends in dist selector)
        let tok = cand
            .selected_token()
            .unwrap_or_else(|| sampler.sample(&ctx, logits_idx));
        gate.accept(tok);
        sampler.accept(tok);
        if model.is_eog_token(tok) {
            break;
        }
        // token_to_piece with an incremental decoder
        text.push_str(
            &model
                .token_to_piece(tok, &mut decoder, true, None)
                .unwrap_or_default(),
        );
        batch.clear();
        batch.add(tok, n_past, &[0], true).expect("add");
        n_past += 1;
        ctx.decode(&mut batch).expect("decode");
        logits_idx = 0;
    }

    // --- kv-cache rollback ops ---
    ctx.clear_kv_cache_seq(Some(0), Some(2), None)
        .expect("kv rm");
    ctx.clear_kv_cache();

    println!("DROP-IN NON-MTP SURFACE OK: produced={text:?}");
}

/// Snapshot save/restore — the in-memory
/// `state_seq_*_ext` + `LlamaStateSeqFlags` path.
#[test]
fn dropin_snapshot_surface() {
    let backend = LlamaBackend::init().expect("backend");
    let mparams = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers());
    let model = LlamaModel::load_from_file(&backend, model_path(), &mparams).expect("load");
    let cparams = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048));
    let mut ctx = model.new_context(&backend, cparams).expect("ctx");

    let prompt = model
        .str_to_token("hello world", AddBos::Always)
        .expect("tok");
    let mut batch = LlamaBatch::new(prompt.len().max(16), 1);
    for (i, &t) in prompt.iter().enumerate() {
        batch
            .add(t, i as i32, &[0], i == prompt.len() - 1)
            .expect("add");
    }
    ctx.decode(&mut batch).expect("decode");

    let flags = LlamaStateSeqFlags::empty();
    let size = ctx.state_seq_get_size_ext(0, flags);
    assert!(size > 0, "seq state size should be non-zero");
    let mut buf = vec![0u8; size];
    // SAFETY: buf has `size` bytes (== state_seq_get_size_ext).
    let written = unsafe { ctx.state_seq_get_data_ext(buf.as_mut_ptr(), 0, flags) };
    assert_eq!(written, size, "written bytes == reported size");
    // SAFETY: `buf` is a state produced for this same context.
    let ok = unsafe { ctx.state_seq_set_data_ext(&buf, 0, flags) };
    assert!(ok, "restore should succeed");
    // PARTIAL_ONLY variant must exist.
    let _partial = LlamaStateSeqFlags::PARTIAL_ONLY;

    println!("DROP-IN SNAPSHOT SURFACE OK: state_bytes={size}");
}
