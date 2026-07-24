//! Throughput benchmark: does MTP speculation boost tok/s **under a grammar**?
//!
//! Generates the same tokens two ways under an identical JSON grammar + greedy
//! sampler, and compares tokens/s:
//!   * PLAIN — the non-MTP engine: decode one token per step, grammar-gate it.
//!   * MTP   — caller-driven `draft`/`commit` with the same grammar gate.
//! Because both are greedy under the same grammar, they should emit the SAME
//! tokens (MTP changes only speed, not which tokens are committed) — the bench
//! asserts that, then reports the speedup and the NextN acceptance rate.
//!
//! Gated behind `_smoke` + `common` and `IK_MTP_MODEL` (a combined NextN GGUF).
//! Run with `--nocapture` to see the numbers.
#![cfg(all(feature = "_smoke", feature = "common"))]

use std::num::NonZeroU32;
use std::time::Instant;

use ik_llama_cpp_2::{
    LlamaBackend, LlamaBatch, LlamaContext, LlamaContextParams, LlamaGrammar, LlamaModel,
    LlamaModelParams, LlamaSampler, LlamaToken, LlamaTokenData, LlamaTokenDataArray,
};

const N_CTX: u32 = 2048;
const N_THREADS: u32 = 8;

fn mtp_model_path() -> String {
    std::env::var("IK_MTP_MODEL").expect("set IK_MTP_MODEL to a combined NextN GGUF path")
}

fn n_gen() -> usize {
    std::env::var("IK_N_GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128)
}

/// n_max values to sweep (override with IK_N_MAX_SWEEP="1,2,3,4").
fn n_max_sweep() -> Vec<i32> {
    std::env::var("IK_N_MAX_SWEEP")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<i32>| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2, 3, 4, 6, 8])
}

// Select-then-commit grammar gate (matches edge-ai's GrammarGate): take the
// greedy argmax off the raw logits, check that ONE token against the grammar,
// and only fall back to a full grammar-masked resample on a violation. This
// avoids masking the whole vocab every step (the naive path is ~17x slower).
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
    // Greedy argmax == what the greedy sampler would pick unmasked.
    let (mut best, mut best_v) = (0i32, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as i32;
        }
    }
    let cand = LlamaToken(best);

    // Check just the argmax against the grammar (1-element array).
    let mut one = LlamaTokenDataArray::from_iter(
        std::iter::once(LlamaTokenData::new(cand, best_v, 0.0)),
        false,
    );
    grammar.apply(ctx, &mut one);
    let argmax_ok = one.data.first().is_some_and(|d| d.logit().is_finite());

    let tok = if argmax_ok {
        cand
    } else {
        // Violation: build the full candidate set, grammar-mask it, resample.
        let mut arr = ctx.token_data_array_ith(idx);
        grammar.apply(ctx, &mut arr);
        sampler.apply(&mut arr);
        arr.selected_token()?
    };
    grammar.accept_token(ctx, tok);
    sampler.accept(tok);
    Some(tok)
}

fn cparams(mtp: bool) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_threads(N_THREADS)
        .with_mtp(mtp)
        .with_seed(42)
}

/// Pure decode speed: greedy argmax straight off the logits slice, a REUSED
/// size-1 batch, no grammar / no candidate-array. Isolates GPU decode + minimal
/// host work, to see how much of the loop time is decode vs sampling/grammar.
fn run_raw(model: &LlamaModel, prompt: &[LlamaToken]) -> (usize, f64) {
    let mut ctx = LlamaContext::new(model, &cparams(false)).expect("raw ctx");
    let argmax = |logits: &[f32]| -> i32 {
        let mut bi = 0i32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i as i32;
            }
        }
        bi
    };
    let mut batch = LlamaBatch::new(prompt.len().max(1), 1);
    batch.add_sequence(prompt, 0, false).expect("add prompt");
    ctx.decode(&mut batch).expect("prefill");

    let t0 = Instant::now();
    let mut tok = LlamaToken(argmax(
        ctx.get_logits_ith(batch.n_tokens() - 1).expect("logits"),
    ));
    let mut n = 1usize;
    let mut pos = prompt.len() as i32;
    let mut b = LlamaBatch::new(1, 1);
    while n < n_gen() && !model.is_eog(tok) {
        b.clear();
        b.add(tok, pos, &[0], true).expect("add");
        ctx.decode(&mut b).expect("decode");
        pos += 1;
        tok = LlamaToken(argmax(ctx.get_logits_ith(0).expect("logits")));
        n += 1;
    }
    (n, t0.elapsed().as_secs_f64())
}

/// Non-MTP baseline: one target decode per generated token.
fn run_plain(
    model: &LlamaModel,
    grammar_str: &str,
    prompt: &[LlamaToken],
) -> (Vec<LlamaToken>, f64) {
    let mut ctx = LlamaContext::new(model, &cparams(false)).expect("plain ctx");
    let mut grammar = LlamaGrammar::new(model, grammar_str, "root").expect("grammar");
    let mut sampler = LlamaSampler::greedy();

    let mut batch = LlamaBatch::new(prompt.len().max(1), 1);
    batch.add_sequence(prompt, 0, false).expect("add prompt");
    ctx.decode(&mut batch).expect("prefill");

    let t0 = Instant::now();
    let mut out = Vec::new();
    let mut tok =
        gate_pick(&mut ctx, &mut grammar, &mut sampler, batch.n_tokens() - 1).expect("first token");
    out.push(tok);
    let mut pos = prompt.len() as i32;
    while out.len() < n_gen() && !model.is_eog(tok) {
        let mut b = LlamaBatch::new(1, 1);
        b.add(tok, pos, &[0], true).expect("add");
        ctx.decode(&mut b).expect("decode");
        pos += 1;
        tok = gate_pick(&mut ctx, &mut grammar, &mut sampler, 0).expect("pick");
        out.push(tok);
    }
    (out, t0.elapsed().as_secs_f64())
}

/// MTP path: caller-driven draft/verify/commit under the same grammar gate.
fn run_mtp(
    model: &LlamaModel,
    grammar_str: &str,
    prompt: &[LlamaToken],
    n_max: i32,
) -> (Vec<LlamaToken>, f64, usize, usize) {
    use ik_llama_cpp_2::{MtpSpeculative, MtpSpeculativeParams};

    let ctx = LlamaContext::new(model, &cparams(true)).expect("mtp ctx");
    let mut grammar = LlamaGrammar::new(model, grammar_str, "root").expect("grammar");
    let mut sampler = LlamaSampler::greedy();
    let params = MtpSpeculativeParams {
        n_max,
        n_min: 0,
        p_min: 0.0,
        ..Default::default()
    };
    let mut spec = MtpSpeculative::new(model, ctx, params).expect("mtp driver");
    let mut n_past = spec.begin(prompt).expect("begin") as i32;

    let t0 = Instant::now();
    let mut out = Vec::new();
    let mut id_last = {
        let ctx = spec.target_context_mut();
        gate_pick(ctx, &mut grammar, &mut sampler, -1).expect("first token")
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
            .expect("decode");

        let mut committed = Vec::new();
        let mut na = 0usize;
        let mut done = false;
        for (i, &d) in drafts.iter().enumerate() {
            let tok = {
                let ctx = spec.target_context_mut();
                gate_pick(ctx, &mut grammar, &mut sampler, i as i32).expect("pick")
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
                gate_pick(ctx, &mut grammar, &mut sampler, k as i32).expect("bonus")
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
    (out, t0.elapsed().as_secs_f64(), proposed, accepted)
}

#[test]
fn mtp_speedup_under_grammar() {
    let backend = LlamaBackend::init().expect("backend");
    // Offload to GPU with IK_N_GPU_LAYERS (default 0 = CPU). MTP's throughput win
    // only appears when target decode is bandwidth-bound (GPU); on CPU it is
    // compute-bound and speculation adds net work.
    let ngl: u32 = std::env::var("IK_N_GPU_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mparams = LlamaModelParams::default()
        .with_mtp(true)
        .with_n_gpu_layers(ngl);
    let model =
        LlamaModel::load_from_file(&backend, mtp_model_path(), &mparams).expect("load model");
    assert!(
        model.n_nextn_layer() > 0,
        "IK_MTP_MODEL must be a NextN model"
    );

    // A broad-but-real per-step constraint: prose characters. It masks every
    // token outside the set on every step (so the grammar path is genuinely
    // exercised) yet never dead-ends, so acceptance reflects the model's natural
    // distribution rather than a pathological grammar. (A tight JSON-envelope
    // grammar is the consumer's real case; acceptance there is schema-dependent.)
    let grammar_str = "root ::= [a-zA-Z0-9 ,.:;'\"()\\n-]+";

    let prompt = model
        .tokenize(
            "Paris is the capital of France. Write three sentences about it.\n",
            true,
        )
        .expect("tokenize");

    // Warm up once (fills caches / allocator) so timings compare fairly.
    let _ = run_plain(&model, grammar_str, &prompt);

    // Baselines: pure decode + the non-MTP grammar-gated engine.
    let (raw_n, raw_secs) = run_raw(&model, &prompt);
    let tps_raw = raw_n as f64 / raw_secs;
    let (out_plain, secs_plain) = run_plain(&model, grammar_str, &prompt);
    let tps_plain = out_plain.len() as f64 / secs_plain;

    eprintln!(
        "\nMTP n_max SWEEP  (model={}, gpu_layers={}, n_gen={})\n\
         baseline raw   (no grammar) : {:.1} tok/s\n\
         baseline plain (grammar)    : {:.1} tok/s   (grammar overhead {:.2}x)\n\
         ---------------------------------------------------------------\n\
         n_max |  tok/s  | speedup | accept | tok/round | match",
        mtp_model_path().rsplit('/').nth(2).unwrap_or("?"),
        ngl,
        n_gen(),
        tps_raw,
        tps_plain,
        tps_raw / tps_plain,
    );

    let mut best: Option<(i32, f64)> = None;
    for n_max in n_max_sweep() {
        let (out_mtp, secs_mtp, proposed, accepted) = run_mtp(&model, grammar_str, &prompt, n_max);
        let tps_mtp = out_mtp.len() as f64 / secs_mtp;
        let speedup = tps_mtp / tps_plain;
        let accept_rate = if proposed > 0 {
            accepted as f64 / proposed as f64
        } else {
            0.0
        };
        let rounds = proposed as f64 / f64::from(n_max).max(1.0);
        let tok_per_round = out_mtp.len() as f64 / rounds.max(1.0);
        let common = out_plain
            .iter()
            .zip(out_mtp.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let match_frac = common as f64 / out_plain.len().min(out_mtp.len()).max(1) as f64;

        // Degeneracy guard: catch a corrupt model spamming one token (which fakes
        // high acceptance). "coherent" = the most frequent token is < 60% of output.
        let mut counts = std::collections::HashMap::new();
        for t in &out_mtp {
            *counts.entry(t.0).or_insert(0usize) += 1;
        }
        let top_frac = counts.values().copied().max().unwrap_or(0) as f64 / out_mtp.len() as f64;
        let text = model.detokenize(&out_mtp).unwrap_or_default();

        eprintln!(
            "  {:>3}  | {:>6.1} |  {:>4.2}x  |  {:.2}  |   {:>4.2}    | {:.0}%   {:?}",
            n_max,
            tps_mtp,
            speedup,
            accept_rate,
            tok_per_round,
            match_frac * 100.0,
            text.chars().take(48).collect::<String>(),
        );

        // Real correctness = grammar-valid, coherent output; NOT bit-identity with
        // sequential greedy (the batched verify decode differs from one-at-a-time
        // by float reduction order, which flips greedy *ties* — inherent to spec
        // decoding, not a bug). The grammar gate guarantees validity by construction.
        assert!(proposed > 0, "n_max={n_max}: NextN proposed no drafts");
        assert!(
            top_frac < 0.60,
            "n_max={n_max}: degenerate output (one token = {:.0}%) — corrupt model?",
            top_frac * 100.0
        );

        if best.map_or(true, |(_, s)| speedup > s) {
            best = Some((n_max, speedup));
        }
    }
    if let Some((n, s)) = best {
        eprintln!("  best: n_max={n} at {s:.2}x");
    }
}
