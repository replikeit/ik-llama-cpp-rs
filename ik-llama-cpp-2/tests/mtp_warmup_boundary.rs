//! Regression guard: MTP prompt-warmup conditioning must not depend on how the
//! prompt happened to be chunked.
//!
//! `common_speculative_on_target_batch` (in the vendored
//! `ik-llama-cpp-sys/ik_llama.cpp/common/speculative.cpp`) feeds the NextN
//! companion a **shifted** hidden-state array during prompt warmup: the
//! conditioned row 0 must carry the target's hidden state from the token
//! immediately BEFORE this warmup call's batch — or zeros, if there is no such
//! token (a fresh position-0 warmup). Before the fix already applied in this
//! working tree, a write-before-read on `target_hidden_by_seq` meant row 0
//! instead got **this batch's own last hidden** (a state up to `n_tokens - 1`
//! positions in the FUTURE relative to row 0), and the genuine zero-boundary
//! branch was unreachable. See the fix:
//! `git -C ik-llama-cpp-sys/ik_llama.cpp diff -- common/speculative.cpp`.
//!
//! The bug is invisible on any prompt shorter than `n_batch`
//! ([`MtpSpeculative::begin`] chunks the warmup into `llama_n_batch`-sized
//! calls): a single warmup call has no "preceding chunk" to mis-attribute, so
//! it always took the (harmless, and — pre-fix — the ONLY reachable) branch.
//! It only bites once a prompt warmup crosses a chunk boundary, which is
//! exactly how it shipped unnoticed.
//!
//! This test is DIFFERENTIAL: the same prompt, generated to the same length
//! under the same grammar + greedy sampler, once with a small `n_batch`
//! (forcing several warmup chunks, i.e. several row-0 boundary crossings) and
//! once with a large `n_batch` (a single warmup call, no boundary crossed at
//! all). It measures NextN draft acceptance both ways and asserts the
//! multi-chunk run does not dip relative to the single-chunk run beyond a
//! lenient margin.
//!
//! Honesty note: the task brief's own measurement (30 real cases, Qwen3.5-4B,
//! CUDA) put the fix's acceptance effect at +0.20% — at the edge of noise.
//! This test does NOT assert an improvement or a speedup; it asserts the
//! ABSENCE of the regression the bug's failure mode would cause (a corrupted
//! row 0 propagated through an entire chunk by the recurrent companion), with
//! a margin loose enough to absorb ordinary run-to-run noise. Treat it as a
//! regression guard for a warmup path (multi-chunk prompts) that had zero test
//! coverage before this fix — not a performance benchmark.
//!
//! A cheap *structural* observation covers the "fresh position-0 warmup
//! receives zeros" half of the acceptance criterion, which is otherwise
//! C++-internal and not directly observable from Rust: every
//! [`MtpSpeculative::new`] call below allocates a brand-new
//! `common_speculative` driver, which constructs a fresh (empty)
//! `target_hidden_by_seq` map (`ik-llama-cpp-sys/wrapper_common.cpp`'s
//! `ik_llama_rs_mtp_init` -> `common_speculative_init` ->
//! `std::make_unique<common_speculative_state_mtp>`). So the FIRST warmup
//! chunk of every scenario here — both the multi-chunk and the single-chunk
//! one — necessarily takes the fix's `assign(0.0f)` branch (there is no prior
//! call for that fresh driver's `seq_id`). If that branch read garbage instead
//! of zero-filling, both scenarios below would be affected identically, not
//! just the multi-chunk one; the healthy-floor and degeneracy assertions on
//! BOTH scenarios are what would catch that.
//!
//! Gated behind `_smoke` + `common` and `IK_MTP_MODEL` (a combined NextN GGUF).
//! Run with `--nocapture` to see the numbers.
#![cfg(all(feature = "_smoke", feature = "common"))]
// A model-backed bench/smoke: casts, long fns, and terse doc prose are expected.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::pedantic,
    clippy::doc_lazy_continuation
)]

use std::collections::HashMap;
use std::num::NonZeroU32;

use ik_llama_cpp_2::{
    LlamaBackend, LlamaBatch, LlamaContext, LlamaContextParams, LlamaGrammar, LlamaModel,
    LlamaModelParams, LlamaSampler, LlamaToken, LlamaTokenData, LlamaTokenDataArray,
    MtpSpeculative, MtpSpeculativeParams,
};

const N_CTX: u32 = 2048;
const N_THREADS: u32 = 8;

/// Small enough that the prompt below (a few hundred tokens) needs several
/// warmup chunks (`MtpSpeculative::begin` chunks the prompt by
/// `llama_n_batch`, mirroring native `--spec-type mtp`) — this is the case
/// that crosses the row-0 boundary the fix addresses. Must stay well under
/// `PROMPT_TEXT`'s tokenized length (checked at runtime below).
const N_BATCH_MULTICHUNK: u32 = 64;
/// Large enough to hold the whole prompt in a single warmup chunk — the case
/// that shipped without ever exercising the boundary logic at all.
const N_BATCH_SINGLECHUNK: u32 = 2048;

/// Minimum warmup chunks the multi-chunk scenario must produce for this test
/// to actually exercise the bug (a single chunk cannot catch it).
const MIN_MULTICHUNK_CHUNKS: usize = 4;

/// Broad prose charset: masks every step (grammar genuinely exercised) yet
/// never dead-ends, so acceptance reflects the model's natural distribution
/// rather than a pathological grammar (matches the other MTP smoke tests).
const GRAMMAR: &str = "root ::= [a-zA-Z0-9 ,.:;'\"()\\n-]+";

/// Real, coherent, information-dense prose (not a repeated phrase, so the
/// model's continuation is not itself degenerate) long enough to tokenize to
/// several times `N_BATCH_MULTICHUNK`, short enough to keep a two-scenario
/// CPU run fast.
const PROMPT_TEXT: &str = "The history of computing stretches back centuries before the first \
electronic machines were built. Mechanical calculators designed by inventors such as Blaise \
Pascal and Gottfried Leibniz automated arithmetic using gears and levers, laying groundwork for \
later designs. In the nineteenth century, Charles Babbage conceived the Analytical Engine, a \
general-purpose mechanical computer that could be programmed using punched cards borrowed from \
the textile industry's Jacquard looms. Ada Lovelace, working alongside Babbage, wrote what many \
now consider the first published computer program, along with notes describing how such a \
machine might one day manipulate symbols beyond mere numbers.\n\
The twentieth century brought electromechanical and then fully electronic computers. During the \
Second World War, codebreakers at Bletchley Park built specialized machines to decrypt enemy \
communications, while in the United States the ENIAC demonstrated that vacuum tubes could \
perform general calculations far faster than any mechanical device. These early machines were \
enormous, consumed vast amounts of power, and required teams of engineers working around the \
clock to keep them running, yet they proved beyond doubt that electronic computation was \
practical at scale. Within a decade, research laboratories and universities across several \
countries had begun building their own experimental machines, each one refining the ideas that \
came before it.\n\
The invention of the transistor in 1947 transformed the field completely. Transistors were \
smaller, cheaper, more reliable, and far more energy efficient than vacuum tubes, and they \
enabled an entirely new generation of computers throughout the 1950s and 1960s. The subsequent \
development of the integrated circuit allowed engineers to place many transistors on a single \
chip, and by the 1970s microprocessors made it possible to build an entire computer's central \
processing unit on one small piece of silicon. This relentless miniaturization, roughly doubling \
in density every two years, set the stage for the personal computer revolution that would soon \
follow.\n\
Companies such as Apple, IBM, and Commodore brought computing into homes and small offices \
during the late 1970s and early 1980s. Graphical user interfaces, popularized first by Xerox \
PARC and later refined by Apple and Microsoft, made computers approachable for people without \
any specialized training. The rise of the public internet during the 1990s connected these \
once-isolated machines into a single global network, and the decades that followed saw computing \
power migrate from bulky desktops into laptops, then into smartphones, and eventually into \
countless small embedded devices scattered throughout everyday life, often unnoticed by the \
people who now depend on them constantly.\n";

fn mtp_model_path() -> String {
    std::env::var("IK_MTP_MODEL").expect("set IK_MTP_MODEL to a combined NextN GGUF path")
}

fn n_gen() -> usize {
    std::env::var("IK_N_GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80)
}

fn gpu_layers() -> u32 {
    std::env::var("IK_N_GPU_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn cparams(n_batch: u32) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_batch) // n_ubatch must never exceed n_batch
        .with_n_threads(N_THREADS)
        .with_mtp(true)
        .with_seed(42)
}

/// Select-then-commit grammar gate (matches edge-ai's `GrammarGate` and the
/// other MTP smoke tests): argmax off raw logits, grammar-check that one
/// token, only full-vocab resample on a violation.
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

/// Draft -> verify -> grammar-gate -> commit generation loop (mirrors
/// `mtp_reuse.rs` / `mtp_grammar_bench.rs`'s `generate`/`run_mtp`). Returns
/// (emitted tokens, proposed drafts, accepted drafts).
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

/// Outcome of one warmup-chunking scenario, run to the same generation length
/// under the same grammar + greedy sampler.
struct ScenarioResult {
    label: &'static str,
    n_batch: u32,
    n_chunks: usize,
    proposed: usize,
    accepted: usize,
    rate: f64,
    top_frac: f64,
    text_preview: String,
}

/// Build a fresh context + `MtpSpeculative` at `n_batch`, warm the prompt
/// (crossing `n_chunks` row-0 boundaries), then generate `n_gen()` tokens.
fn run_scenario(
    model: &LlamaModel,
    prompt: &[LlamaToken],
    label: &'static str,
    n_batch: u32,
) -> ScenarioResult {
    // Mirrors `MtpSpeculative::begin`'s own chunking (`ik_llama_rs_mtp_begin`
    // in wrapper_common.cpp: `chunk = min(n_batch, n - offset)` in a loop) so
    // this is the exact number of `common_speculative_on_target_batch` warmup
    // calls the run below makes, i.e. the number of row-0 boundaries crossed.
    let n_chunks = prompt.len().div_ceil(n_batch as usize);

    let ctx = LlamaContext::new(model, &cparams(n_batch)).expect("mtp ctx");
    let mut spec =
        MtpSpeculative::new(model, ctx, MtpSpeculativeParams::default()).expect("mtp driver");
    let n_past = spec.begin(prompt).expect("begin") as i32;

    let mut grammar = LlamaGrammar::new(model, GRAMMAR, "root").expect("grammar");
    let mut sampler = LlamaSampler::greedy();
    let (out, proposed, accepted) = generate(&mut spec, model, &mut grammar, &mut sampler, n_past);

    let rate = if proposed > 0 {
        accepted as f64 / proposed as f64
    } else {
        0.0
    };

    // Degeneracy guard input: a corrupt model / broken conditioning can spam
    // one token, which would otherwise fake a high acceptance rate.
    let mut counts = HashMap::new();
    for t in &out {
        *counts.entry(t.0).or_insert(0usize) += 1;
    }
    let top_frac = counts.values().copied().max().unwrap_or(0) as f64 / out.len().max(1) as f64;
    let text_preview = model
        .detokenize(&out)
        .unwrap_or_default()
        .chars()
        .take(80)
        .collect::<String>();

    ScenarioResult {
        label,
        n_batch,
        n_chunks,
        proposed,
        accepted,
        rate,
        top_frac,
        text_preview,
    }
}

/// Single entry point: the backend is process-global (init once) and the
/// model is expensive to load, so both scenarios share one init + one load.
#[test]
fn mtp_warmup_boundary() {
    let backend = LlamaBackend::init().expect("backend");
    let model = load_model(&backend);

    let prompt = model.tokenize(PROMPT_TEXT, true).expect("tokenize");
    eprintln!(
        "MTP WARMUP BOUNDARY: prompt tokenizes to {} tokens",
        prompt.len()
    );
    assert!(
        prompt.len() >= 250,
        "prompt too short ({} tokens) to give the multi-chunk scenario several \
         warmup boundaries; lengthen PROMPT_TEXT",
        prompt.len()
    );
    assert!(
        prompt.len() <= N_BATCH_SINGLECHUNK as usize,
        "prompt ({} tokens) does not fit in one N_BATCH_SINGLECHUNK ({}) chunk \
         -- the 'single-chunk' scenario would no longer be single-chunk",
        prompt.len(),
        N_BATCH_SINGLECHUNK
    );

    let n_chunks_multi = prompt.len().div_ceil(N_BATCH_MULTICHUNK as usize);
    assert!(
        n_chunks_multi >= MIN_MULTICHUNK_CHUNKS,
        "need >= {MIN_MULTICHUNK_CHUNKS} warmup chunks to cross several row-0 \
         boundaries; got {n_chunks_multi} chunks for a {}-token prompt at \
         n_batch={N_BATCH_MULTICHUNK} -- lengthen PROMPT_TEXT or shrink \
         N_BATCH_MULTICHUNK",
        prompt.len()
    );

    // (A) Small n_batch: `begin()` warms the prompt over `n_chunks_multi`
    // calls, each one (after the first) needing the fix's snapshot-before-
    // store to condition row 0 on the PRECEDING chunk's last hidden rather
    // than its own.
    let multi = run_scenario(&model, &prompt, "multi-chunk", N_BATCH_MULTICHUNK);
    // (B) Large n_batch: `begin()` warms the prompt in a single call -- the
    // case that shipped without ever exercising the boundary logic.
    let single = run_scenario(&model, &prompt, "single-chunk", N_BATCH_SINGLECHUNK);

    eprintln!(
        "\nMTP WARMUP BOUNDARY  (model={}, prompt={} tokens, n_gen={})\n\
         scenario     | n_batch | chunks | proposed | accepted | accept | top_tok | text\n\
         ------------------------------------------------------------------------------------\n\
         {:<12} | {:>7} | {:>6} | {:>8} | {:>8} | {:>6.4} | {:>6.1}% | {:?}\n\
         {:<12} | {:>7} | {:>6} | {:>8} | {:>8} | {:>6.4} | {:>6.1}% | {:?}\n",
        mtp_model_path().rsplit('/').nth(2).unwrap_or("?"),
        prompt.len(),
        n_gen(),
        multi.label,
        multi.n_batch,
        multi.n_chunks,
        multi.proposed,
        multi.accepted,
        multi.rate,
        multi.top_frac * 100.0,
        multi.text_preview,
        single.label,
        single.n_batch,
        single.n_chunks,
        single.proposed,
        single.accepted,
        single.rate,
        single.top_frac * 100.0,
        single.text_preview,
    );

    // Structural sanity: the scenarios really did chunk the way their names
    // claim (see the module doc for why this matters).
    assert!(
        multi.n_chunks >= MIN_MULTICHUNK_CHUNKS,
        "multi-chunk scenario only produced {} chunk(s)",
        multi.n_chunks
    );
    assert_eq!(
        single.n_chunks, 1,
        "single-chunk scenario produced {} chunks, not 1",
        single.n_chunks
    );

    // Sanity: NextN actually proposed drafts in both scenarios.
    assert!(multi.proposed > 0, "multi-chunk: NextN proposed no drafts");
    assert!(
        single.proposed > 0,
        "single-chunk: NextN proposed no drafts"
    );

    // Degeneracy guard: catch a corrupt model / broken conditioning spamming
    // one token, which would otherwise fake a high acceptance rate.
    assert!(
        multi.top_frac < 0.60,
        "multi-chunk: degenerate output (one token = {:.0}%) -- corrupt model \
         or broken conditioning?",
        multi.top_frac * 100.0
    );
    assert!(
        single.top_frac < 0.60,
        "single-chunk: degenerate output (one token = {:.0}%) -- corrupt model?",
        single.top_frac * 100.0
    );

    // Conservative floor: both scenarios must be reasonably healthy. This is
    // deliberately loose -- it is not a quality benchmark, just a guard
    // against a badly broken conditioning path tanking acceptance outright.
    assert!(
        multi.rate > 0.30,
        "multi-chunk acceptance too low ({:.4}) -- possible regression at a \
         warmup chunk boundary",
        multi.rate
    );
    assert!(
        single.rate > 0.30,
        "single-chunk acceptance too low ({:.4})",
        single.rate
    );

    // The actual regression guard: chunking the SAME prompt's warmup into
    // several calls must not measurably hurt acceptance relative to a single
    // warmup call. The task brief's own measurement puts the fix's effect at
    // the edge of noise (+0.20%), so this margin is intentionally lenient --
    // it guards against the bug's failure mode (a corrupted row 0 propagated
    // through an entire chunk by the recurrent companion), not a performance
    // target. Do not tighten this into a speedup/improvement assertion.
    assert!(
        multi.rate >= single.rate - 0.08,
        "multi-chunk warmup acceptance ({:.4}) dipped more than the 0.08 \
         margin below single-chunk ({:.4}) -- possible regression of the \
         row-0 warmup conditioning fix at a chunk boundary",
        multi.rate,
        single.rate
    );
}
