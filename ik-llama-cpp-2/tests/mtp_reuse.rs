//! Cross-call MTP KV-prefix reuse: correctness parity + reuse instrumentation.
//!
//! The caller-driven prefill (`warm` + `finalize_prompt`) plus companion-state
//! snapshot/restore (`companion_state_{size,get,set}`) let a consumer restore a
//! shared prefix on BOTH the target and the NextN companion, then warm only the
//! suffix — skipping the (dominant) target prefill of the prefix.
//!
//! # Parity gate (`mtp_reuse_parity`)
//!
//! Committed tokens are grammar-gated off the **target** logits, and a draft only
//! advances `n_past` when it *equals* the committed token — so the emitted
//! sequence is independent of draft quality. Restoring the target KV to
//! `reuse_from` and decoding only the suffix must therefore produce **byte-
//! identical** output to a full warmup (the companion §5 boundary-conditioning
//! difference moves acceptance, never which tokens are emitted). This is the
//! correctness gate: reuse must never change output.
//!
//! Gated behind `_smoke` + `common` and `IK_MTP_MODEL` (a combined NextN GGUF).
#![cfg(all(feature = "_smoke", feature = "common"))]
// A model-backed bench/smoke: casts, long fns, and terse doc prose are expected.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::pedantic,
    clippy::doc_lazy_continuation
)]

use std::num::NonZeroU32;
use std::time::Instant;

use ik_llama_cpp_2::context::session::LlamaStateSeqFlags;
use ik_llama_cpp_2::{
    LlamaBackend, LlamaBatch, LlamaContext, LlamaContextParams, LlamaGrammar, LlamaModel,
    LlamaModelParams, LlamaSampler, LlamaToken, LlamaTokenData, LlamaTokenDataArray,
    MtpSpeculative, MtpSpeculativeParams,
};

const N_CTX: u32 = 2048;
const N_THREADS: u32 = 8;
/// Prefill chunk size — well under the default `n_batch` (`llama_decode` aborts
/// on `n_tokens > n_batch`). Small enough to exercise the multi-chunk warm loop.
const PREFILL_CHUNK: usize = 128;
/// Broad prose charset: masks every step (grammar genuinely exercised) yet never
/// dead-ends, so acceptance reflects the model's natural distribution.
const GRAMMAR: &str = "root ::= [a-zA-Z0-9 ,.:;'\"()\\n-]+";

fn mtp_model_path() -> String {
    std::env::var("IK_MTP_MODEL").expect("set IK_MTP_MODEL to a combined NextN GGUF path")
}

fn n_gen() -> usize {
    std::env::var("IK_N_GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
}

fn gpu_layers() -> u32 {
    std::env::var("IK_N_GPU_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn cparams() -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_threads(N_THREADS)
        .with_mtp(true)
        .with_seed(42)
}

/// Select-then-commit grammar gate (matches edge-ai's `GrammarGate` and the
/// throughput bench): argmax off raw logits, grammar-check that one token, only
/// full-vocab resample on a violation.
fn gate_pick(
    ctx: &mut LlamaContext,
    grammar: &mut LlamaGrammar,
    sampler: &mut LlamaSampler,
    idx: i32,
) -> Option<LlamaToken> {
    let logits = ctx.get_logits_ith(idx).ok()?;
    if logits.is_empty() {
        return None;
    }
    let (mut best, mut best_v) = (0i32, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as i32;
        }
    }
    let cand = LlamaToken(best);
    let mut one = LlamaTokenDataArray::from_iter(
        std::iter::once(LlamaTokenData::new(cand, best_v, 0.0)),
        false,
    );
    grammar.apply(ctx, &mut one);
    let tok = if one.data.first().is_some_and(|d| d.logit().is_finite()) {
        cand
    } else {
        let mut arr = ctx.token_data_array_ith(idx);
        grammar.apply(ctx, &mut arr);
        sampler.apply(&mut arr);
        arr.selected_token()?
    };
    grammar.accept_token(ctx, tok);
    sampler.accept(tok);
    Some(tok)
}

/// Full target-KV + companion snapshot at a prefix boundary.
struct Snapshot {
    target: Vec<u8>,
    companion: Vec<u8>,
}

/// Caller-driven prefill of `tokens[reuse_from..]` on `spec`. The caller must
/// have already established the target + companion KV up to `reuse_from` (a
/// `clear_kv_cache()` for `reuse_from == 0`, or a restore for a reuse). Decodes
/// the suffix in chunks with **output on every position** (the warmup reads a
/// per-position hidden row), warms the companion over each chunk, and — if
/// `capture_at` is set and a chunk boundary lands on it — snapshots BOTH the
/// target and companion states *there* (contemporaneous capture: the recurrent
/// `s_l` cannot be sliced from a later state). Finalizes the prompt so `draft`
/// is ready. Returns the snapshot if `capture_at` was hit.
fn prefill(
    spec: &mut MtpSpeculative,
    tokens: &[LlamaToken],
    reuse_from: usize,
    capture_at: Option<usize>,
) -> Option<Snapshot> {
    let n = tokens.len();
    let mut pos = reuse_from;
    let mut last_chunk_len = 0usize;
    let mut snap = None;
    while pos < n {
        let mut end = (pos + PREFILL_CHUNK).min(n);
        if let Some(b) = capture_at {
            if b > pos && b < end {
                end = b; // force a chunk boundary onto the snapshot point
            }
        }
        let mut batch = LlamaBatch::new(end - pos, 1);
        for (j, &t) in tokens[pos..end].iter().enumerate() {
            batch
                .add(t, (pos + j) as i32, &[0], true) // output on EVERY position
                .expect("prefill add");
        }
        spec.target_context_mut()
            .decode(&mut batch)
            .expect("prefill decode");
        spec.warm(&batch).expect("warm companion over chunk");
        last_chunk_len = end - pos;
        pos = end;

        if capture_at == Some(pos) {
            let target = capture_target(spec.target_context());
            let companion = capture_companion(spec);
            assert!(!target.is_empty(), "target snapshot empty at boundary");
            assert!(
                !companion.is_empty(),
                "companion snapshot empty at boundary"
            );
            snap = Some(Snapshot { target, companion });
        }
    }
    spec.finalize_prompt((last_chunk_len - 1) as i32, n)
        .expect("finalize_prompt");
    snap
}

fn capture_target(ctx: &LlamaContext) -> Vec<u8> {
    let size = ctx.state_seq_get_size_ext(0, LlamaStateSeqFlags::empty());
    let mut buf = vec![0u8; size];
    // SAFETY: buf is exactly `size` bytes (the size llama.cpp just reported).
    let n = unsafe { ctx.state_seq_get_data_ext(buf.as_mut_ptr(), 0, LlamaStateSeqFlags::empty()) };
    buf.truncate(n);
    buf
}

fn capture_companion(spec: &MtpSpeculative) -> Vec<u8> {
    let size = spec.companion_state_size();
    let mut buf = vec![0u8; size];
    // SAFETY: buf is exactly `size` bytes (the size the companion just reported).
    let n = unsafe { spec.companion_state_get(buf.as_mut_ptr()) };
    buf.truncate(n);
    buf
}

/// Restore both contexts to the snapshot boundary. Caller passes `reuse_from`
/// separately (the token count the snapshot represents).
fn restore(spec: &mut MtpSpeculative, snap: &Snapshot) -> bool {
    spec.target_context_mut().clear_kv_cache();
    // SAFETY: bytes produced by capture_* on the same model/params; llama.cpp
    // validates length internally and rejects (→ false) rather than reading OOB.
    let ok_t = unsafe {
        spec.target_context_mut().state_seq_set_data_ext(
            &snap.target,
            0,
            LlamaStateSeqFlags::empty(),
        )
    };
    let ok_c = unsafe { spec.companion_state_set(&snap.companion) };
    ok_t && ok_c
}

/// Draft → verify → grammar-gate → commit generation loop (mirrors the bench's
/// `run_mtp`). Returns (emitted tokens, proposed drafts, accepted drafts).
fn generate(
    spec: &mut MtpSpeculative,
    model: &LlamaModel,
    grammar: &mut LlamaGrammar,
    sampler: &mut LlamaSampler,
    mut n_past: i32,
) -> (Vec<LlamaToken>, usize, usize) {
    let mut out = Vec::new();
    let mut id_last = {
        let ctx = spec.target_context_mut();
        gate_pick(ctx, grammar, sampler, -1).expect("first token")
    };
    out.push(id_last);
    let (mut proposed, mut accepted) = (0usize, 0usize);
    'gen: while out.len() < n_gen() {
        let drafts = spec.draft(n_past, id_last).expect("draft");
        let k = drafts.len();
        proposed += k;

        let mut batch = LlamaBatch::new(k + 1, 1);
        batch.add(id_last, n_past, &[0], true).expect("add id_last");
        for (i, d) in drafts.iter().enumerate() {
            batch
                .add(*d, n_past + 1 + i as i32, &[0], true)
                .expect("add draft");
        }
        spec.target_context_mut()
            .decode(&mut batch)
            .expect("verify decode");

        let mut committed = Vec::new();
        let mut na = 0usize;
        let mut done = false;
        for (i, &d) in drafts.iter().enumerate() {
            let tok = {
                let ctx = spec.target_context_mut();
                gate_pick(ctx, grammar, sampler, i as i32).expect("pick")
            };
            committed.push(tok);
            out.push(tok);
            if model.is_eog(tok) || out.len() >= n_gen() {
                done = true;
                break;
            }
            if tok == d {
                na += 1;
            } else {
                break;
            }
        }
        if !done && na == k {
            let tok = {
                let ctx = spec.target_context_mut();
                gate_pick(ctx, grammar, sampler, k as i32).expect("bonus")
            };
            committed.push(tok);
            out.push(tok);
            if model.is_eog(tok) {
                done = true;
            }
        }
        accepted += na;
        spec.commit(id_last, &committed, k).expect("commit");
        n_past += committed.len() as i32;
        id_last = *committed.last().expect("committed");
        if done {
            break 'gen;
        }
    }
    (out, proposed, accepted)
}

fn load_model(backend: &LlamaBackend) -> LlamaModel {
    let mparams = LlamaModelParams::default()
        .with_mtp(true)
        .with_n_gpu_layers(gpu_layers());
    let model =
        LlamaModel::load_from_file(backend, mtp_model_path(), &mparams).expect("load model");
    assert!(
        model.n_nextn_layer() > 0,
        "IK_MTP_MODEL must be a NextN model"
    );
    model
}

/// Correctness gate: restore-then-suffix-warm emits byte-identical tokens to a
/// full warmup on the same prompt (greedy + same grammar). Reuse must never
/// change output — only acceptance/speed.
fn parity_phase(model: &LlamaModel) {
    let prompt = model
        .tokenize(
            "You are a helpful assistant. Paris is the capital of France. \
             Write three short sentences about the city and its history.\n",
            true,
        )
        .expect("tokenize");
    assert!(prompt.len() > 8, "need a non-trivial prompt");
    let split = prompt.len() / 2;

    let ctx = LlamaContext::new(model, &cparams()).expect("mtp ctx");
    let mut spec =
        MtpSpeculative::new(model, ctx, MtpSpeculativeParams::default()).expect("driver");

    // (A) Full warmup — snapshot the target + companion at `split`, then generate.
    spec.target_context_mut().clear_kv_cache();
    let snap = prefill(&mut spec, &prompt, 0, Some(split)).expect("snapshot at split");
    let (out_full, _p_full, _a_full) = {
        let mut g = LlamaGrammar::new(model, GRAMMAR, "root").expect("grammar");
        let mut s = LlamaSampler::greedy();
        generate(&mut spec, model, &mut g, &mut s, prompt.len() as i32)
    };

    // (B) Reuse — restore both contexts to `split`, warm only the suffix, generate.
    assert!(restore(&mut spec, &snap), "restore must succeed");
    let _ = prefill(&mut spec, &prompt, split, None);
    let (out_reuse, _p_r, _a_r) = {
        let mut g = LlamaGrammar::new(model, GRAMMAR, "root").expect("grammar");
        let mut s = LlamaSampler::greedy();
        generate(&mut spec, model, &mut g, &mut s, prompt.len() as i32)
    };

    eprintln!(
        "MTP REUSE PARITY: prompt={} split={} full={} reuse={}\n  full : {:?}\n  reuse: {:?}",
        prompt.len(),
        split,
        out_full.len(),
        out_reuse.len(),
        model
            .detokenize(&out_full)
            .unwrap_or_default()
            .chars()
            .take(64)
            .collect::<String>(),
        model
            .detokenize(&out_reuse)
            .unwrap_or_default()
            .chars()
            .take(64)
            .collect::<String>(),
    );

    assert_eq!(
        out_full, out_reuse,
        "restore+suffix-warm must emit byte-identical tokens to a full warmup \
         (committed tokens are gated off the restored target logits, independent \
         of companion draft quality) — a mismatch means the target snapshot \
         round-trip or the prefill plumbing is broken"
    );
}

/// `reuse_from` fractions to sweep (override with IK_REUSE_SWEEP="0.5,0.75,0.9").
fn reuse_sweep() -> Vec<f64> {
    std::env::var("IK_REUSE_SWEEP")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<f64>| !v.is_empty())
        .unwrap_or_else(|| vec![0.25, 0.5, 0.75, 0.9])
}

/// Instrumentation: prefill wall-time saved and NextN acceptance, reuse vs. a
/// full warmup, across `reuse_from` fractions of a shared prefix. The savings
/// should track the skipped fraction (target decode of `[0..reuse_from)` is not
/// re-run); acceptance should stay near the full-warmup rate (the §5 boundary
/// perturbation decays over the suffix). Set IK_N_GPU_LAYERS to measure on GPU,
/// where the prefill win is largest. Run with `--nocapture`.
fn bench_phase(model: &LlamaModel) {
    // A sizable shared prefix + short query, so a skipped prefix is a real win.
    let sys_block = "You are a meticulous IT support assistant. Follow the company \
        troubleshooting policy: confirm the symptom, identify the single most likely \
        root cause, and propose one concrete next step. Be concise and specific. "
        .repeat(6);
    let prompt = model
        .tokenize(
            &format!("{sys_block}\nUser: my laptop is slow today. What should I do?\n"),
            true,
        )
        .expect("tokenize");
    let n = prompt.len();
    assert!(n > 32, "need a non-trivial prompt (got {n})");

    let ctx = LlamaContext::new(model, &cparams()).expect("mtp ctx");
    let mut spec =
        MtpSpeculative::new(model, ctx, MtpSpeculativeParams::default()).expect("driver");

    // Warm up once (allocator / caches) so timings compare fairly.
    spec.target_context_mut().clear_kv_cache();
    let _ = prefill(&mut spec, &prompt, 0, None);

    // Full-warmup baseline: timed clean prefill + generate.
    spec.target_context_mut().clear_kv_cache();
    let t = Instant::now();
    let _ = prefill(&mut spec, &prompt, 0, None);
    let full_ms = t.elapsed().as_secs_f64() * 1000.0;
    let (out_full, prop_full, acc_full) = {
        let mut g = LlamaGrammar::new(model, GRAMMAR, "root").expect("grammar");
        let mut s = LlamaSampler::greedy();
        generate(&mut spec, model, &mut g, &mut s, n as i32)
    };
    let rate_full = if prop_full > 0 {
        acc_full as f64 / prop_full as f64
    } else {
        0.0
    };

    eprintln!(
        "\nMTP PREFIX REUSE  (model={}, gpu_layers={}, prompt={}, n_gen={})\n\
         full warmup: prefill {:.1} ms | accept {:.2}\n\
         --------------------------------------------------------------\n\
         reuse_from |  prefill ms | saved | accept | parity",
        mtp_model_path().rsplit('/').nth(2).unwrap_or("?"),
        gpu_layers(),
        n,
        n_gen(),
        full_ms,
        rate_full,
    );

    for frac in reuse_sweep() {
        let split = ((n as f64 * frac) as usize).clamp(1, n - 1);

        // Create the snapshot at `split` (untimed — in production this was captured
        // on a prior call). A full prefill that snapshots as it passes `split`.
        spec.target_context_mut().clear_kv_cache();
        let snap = prefill(&mut spec, &prompt, 0, Some(split)).expect("snapshot at split");

        // Timed reuse: restore both contexts to `split`, warm only the suffix.
        let t = Instant::now();
        assert!(restore(&mut spec, &snap), "restore failed at split={split}");
        let _ = prefill(&mut spec, &prompt, split, None);
        let reuse_ms = t.elapsed().as_secs_f64() * 1000.0;

        let (out_reuse, prop_r, acc_r) = {
            let mut g = LlamaGrammar::new(model, GRAMMAR, "root").expect("grammar");
            let mut s = LlamaSampler::greedy();
            generate(&mut spec, model, &mut g, &mut s, n as i32)
        };
        let rate_r = if prop_r > 0 {
            acc_r as f64 / prop_r as f64
        } else {
            0.0
        };
        let parity = out_reuse == out_full;
        let saved = 1.0 - reuse_ms / full_ms.max(f64::MIN_POSITIVE);

        // Degeneracy guard (corrupt model spamming one token fakes high acceptance).
        let mut counts = std::collections::HashMap::new();
        for tk in &out_reuse {
            *counts.entry(tk.0).or_insert(0usize) += 1;
        }
        let top_frac = counts.values().copied().max().unwrap_or(0) as f64 / out_reuse.len() as f64;

        eprintln!(
            "   {:>5.0}% ({:>4}) | {:>7.1} ms | {:>4.0}% |  {:.2}  | {}",
            frac * 100.0,
            split,
            reuse_ms,
            saved * 100.0,
            rate_r,
            if parity { "ok" } else { "MISMATCH" },
        );

        // Correctness gate holds across the whole sweep: reuse never changes output.
        assert!(
            parity,
            "reuse output diverged from full warmup at split={split}"
        );
        assert!(
            top_frac < 0.60,
            "degenerate output at split={split} (one token = {:.0}%) — corrupt model?",
            top_frac * 100.0
        );
    }
}

/// Single entry point: the backend is process-global (init once) and the model
/// is expensive to load, so both phases share one init + one load. Runs the
/// correctness parity gate first, then the reuse instrumentation sweep.
#[test]
fn mtp_reuse() {
    let backend = LlamaBackend::init().expect("backend");
    let model = load_model(&backend);
    parity_phase(&model);
    bench_phase(&model);
}
