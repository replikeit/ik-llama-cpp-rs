//! Grammar-gated MTP smoke test — proves ik's NextN speculation runs while the
//! CALLER applies its own GBNF grammar + sampler on every committed token
//! (the capability `step()` cannot offer, since it samples inside the glue).
//!
//! This is also the reference caller-driven loop: `begin` → per round
//! `draft` → build+decode the verify batch on the target → grammar-gate each
//! position → `commit`. No sampling happens inside the crate.
//!
//! Gated behind `_smoke` + `common` and the `IK_MTP_MODEL` env var (a combined
//! NextN GGUF, e.g. `.models/qwen35-4b-iq1s-mtp-combined.gguf`).
#![cfg(all(feature = "_smoke", feature = "common"))]

use std::num::NonZeroU32;

use ik_llama_cpp_2::{
    LlamaBackend, LlamaBatch, LlamaContext, LlamaContextParams, LlamaGrammar, LlamaModel,
    LlamaModelParams, LlamaSampler, LlamaToken, MtpSpeculative, MtpSpeculativeParams,
};

fn mtp_model_path() -> String {
    std::env::var("IK_MTP_MODEL").expect("set IK_MTP_MODEL to a combined NextN GGUF path")
}

/// The consumer's grammar-constrained selector, in miniature: read the logits at
/// output index `idx`, mask them with the grammar, pick with the sampler, then
/// advance both the grammar and sampler by the committed token. Returns `None`
/// only if no token is selectable (logits unavailable / grammar dead-ends).
fn gate_pick(
    ctx: &mut LlamaContext,
    grammar: &mut LlamaGrammar,
    sampler: &mut LlamaSampler,
    idx: i32,
) -> Option<LlamaToken> {
    let mut arr = ctx.token_data_array_ith(idx);
    grammar.apply(ctx, &mut arr); // mask grammar-invalid tokens to -inf
    sampler.apply(&mut arr); // then select (greedy)
    let tok = arr.selected_token()?;
    grammar.accept_token(ctx, tok);
    sampler.accept(tok);
    Some(tok)
}

#[test]
fn mtp_runs_under_grammar_constraint() {
    const MAX_TOKENS: usize = 24;
    const MAX_ROUNDS: usize = 128;
    const N_MAX: i32 = 2;

    let backend = LlamaBackend::init().expect("backend init");

    // NextN model + MTP-enabled context (same setup as the `mtp` example).
    let mparams = LlamaModelParams::default().with_mtp(true);
    let model =
        LlamaModel::load_from_file(&backend, mtp_model_path(), &mparams).expect("load MTP model");
    assert!(
        model.n_nextn_layer() > 0,
        "IK_MTP_MODEL must be a NextN/MTP model (n_nextn_layer > 0)"
    );

    let cparams = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_n_threads(8)
        .with_mtp(true)
        .with_seed(42);
    let ctx = LlamaContext::new(&model, &cparams).expect("create MTP context");

    // Envelope grammar: digits only. Because the grammar masks every non-digit
    // token, any produced text MUST be digits regardless of model quality — so a
    // digits-only output proves the grammar was applied to the committed tokens.
    let mut grammar = LlamaGrammar::new(&model, "root ::= [0-9]+", "root").expect("grammar");
    let mut sampler = LlamaSampler::greedy();

    let prompt = model.tokenize("Count: 1 2 3", true).expect("tokenize");

    let params = MtpSpeculativeParams {
        n_max: N_MAX,
        n_min: 0,
        p_min: 0.0,
        ..Default::default()
    };
    let mut spec = MtpSpeculative::new(&model, ctx, params).expect("init MTP driver");
    let mut n_past = spec.begin(&prompt).expect("MTP begin (warmup)") as i32;

    // First committed token from the prompt's last logits (index -1 = last).
    let mut generated: Vec<LlamaToken> = Vec::new();
    let mut id_last = {
        let ctx = spec.target_context_mut();
        gate_pick(ctx, &mut grammar, &mut sampler, -1).expect("first token")
    };
    generated.push(id_last);

    let mut proposed_total = 0usize;
    let mut accepted_total = 0usize;
    let mut rounds = 0usize;

    'gen: for _ in 0..MAX_ROUNDS {
        if generated.len() >= MAX_TOKENS {
            break;
        }
        rounds += 1;

        // 1. DRAFT (no sampling) — up to N_MAX NextN candidates after id_last.
        let drafts = spec.draft(n_past, id_last).expect("draft");
        let k = drafts.len();
        proposed_total += k;

        // 2. VERIFY BATCH = [id_last] + drafts, logits on ALL positions, decoded
        //    on the target context by the caller.
        let mut batch = LlamaBatch::new(k + 1, 1);
        batch.add(id_last, n_past, &[0], true).expect("add id_last");
        for (i, d) in drafts.iter().enumerate() {
            batch
                .add(*d, n_past + 1 + i as i32, &[0], true)
                .expect("add draft");
        }
        spec.target_context_mut()
            .decode(&mut batch)
            .expect("decode verify batch");

        // 3. GRAMMAR-GATED COMMIT — pick each position ourselves; a draft is
        //    accepted only if the grammar-gated pick equals it.
        let mut committed: Vec<LlamaToken> = Vec::new();
        let mut n_accepted = 0usize;
        let mut done = false;
        for (i, &drafted) in drafts.iter().enumerate() {
            let tok = {
                let ctx = spec.target_context_mut();
                gate_pick(ctx, &mut grammar, &mut sampler, i as i32).expect("gate pick")
            };
            committed.push(tok);
            generated.push(tok);
            if model.is_eog(tok) || generated.len() >= MAX_TOKENS {
                done = true;
                break;
            }
            if tok == drafted {
                n_accepted += 1; // draft correct -> keep going
            } else {
                break; // correction -> stop accepting this round
            }
        }
        // Bonus token on a full accept (all k drafts matched).
        if !done && n_accepted == k {
            let tok = {
                let ctx = spec.target_context_mut();
                gate_pick(ctx, &mut grammar, &mut sampler, k as i32).expect("bonus pick")
            };
            committed.push(tok);
            generated.push(tok);
            if model.is_eog(tok) {
                done = true;
            }
        }
        accepted_total += n_accepted;

        // 4. COMMIT — advance the companion + roll target KV to the committed
        //    prefix (ik owns the KV; we never touch it).
        assert!(
            !committed.is_empty(),
            "each round commits at least one token"
        );
        spec.commit(id_last, &committed, k).expect("commit");
        n_past += committed.len() as i32;
        id_last = *committed.last().expect("committed non-empty");

        if done {
            break 'gen;
        }
    }

    // --- Assertions: MTP ran under the grammar and stayed grammar-valid. ---
    assert!(
        proposed_total > 0,
        "NextN proposed no drafts across {rounds} rounds — MTP did not run"
    );
    assert!(!generated.is_empty(), "no tokens generated");

    let text = model.detokenize(&generated).expect("detokenize");
    let stripped: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit()),
        "grammar was not applied on the MTP path: output = {text:?}"
    );

    let accept_rate = accepted_total as f64 / proposed_total as f64;
    let n_tokens = generated.len();
    eprintln!(
        "MTP+GRAMMAR OK: rounds={rounds} tokens={n_tokens} drafts_proposed={proposed_total} \
         drafts_accepted={accepted_total} accept_rate={accept_rate:.2} output={text:?}"
    );
}
