//! Grammar + DRY sampler smoke test: load → decode a prompt → constrain the
//! next-token candidates with a GBNF grammar and the DRY sampler.
//!
//! Gated behind the `_smoke` feature and the `IK_TEST_MODEL` env var (a single
//! merged GGUF, e.g. `.models/qwen35-4b-iq1s-general.gguf`). Asserts *workflow*
//! correctness (the grammar actually masks the candidate set), not text quality
//! (IQ1_S output is expected to be degenerate).
#![cfg(feature = "_smoke")]

use ik_llama_cpp_2::sampling::LlamaTokenDataArray;
use ik_llama_cpp_2::{
    DryParams, LlamaBackend, LlamaBatch, LlamaContext, LlamaContextParams, LlamaDrySampler,
    LlamaGrammar, LlamaModel, LlamaModelParams, LlamaSampler,
};

fn model_path() -> String {
    std::env::var("IK_TEST_MODEL").expect("set IK_TEST_MODEL to a merged GGUF path")
}

/// Number of layers to offload to the GPU (`IK_TEST_NGL`, default 0 = CPU). Set
/// e.g. `IK_TEST_NGL=99` to exercise a CUDA build on-device.
fn n_gpu_layers() -> u32 {
    std::env::var("IK_TEST_NGL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Load the model + a small context and decode a fixed prompt, returning the
/// backend/model/context plus the logits index of the last prompt token.
fn setup() -> (LlamaBackend, LlamaModel) {
    let backend = LlamaBackend::init().expect("backend init");
    let mparams = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers());
    let model = LlamaModel::load_from_file(&backend, model_path(), &mparams).expect("load model");
    assert!(model.n_vocab() > 0, "vocab should be non-empty");
    (backend, model)
}

#[test]
fn grammar_constrains_next_token_to_digits() {
    let (_backend, model) = setup();

    let cparams = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(2048))
        .with_n_threads(8)
        .with_seed(42);
    let mut ctx = LlamaContext::new(&model, &cparams).expect("context");

    let prompt = model
        .tokenize("The year the Berlin Wall fell was", true)
        .expect("tokenize prompt");
    let mut batch = LlamaBatch::new(prompt.len().max(64), 1);
    batch.add_sequence(&prompt, 0, false).expect("add prompt");
    ctx.decode(&mut batch).expect("decode prompt");
    let last = batch.n_tokens() - 1;

    // A digits-only grammar: the constrained argmax must be a digit token, even
    // though the unconstrained continuation of this prompt would not be.
    let mut grammar = LlamaGrammar::new(&model, "root ::= [0-9]+", "root").expect("build grammar");

    let mut greedy = LlamaSampler::greedy();
    let mut n_past = batch.n_tokens();
    let mut logits_idx = last;
    let mut produced = String::new();

    for _ in 0..6 {
        let logits = ctx.get_logits_ith(logits_idx).expect("logits").to_vec();
        let mut arr = LlamaTokenDataArray::from_logits(&logits);
        // Constrain, then draw the argmax over the surviving candidates.
        grammar.apply(&mut ctx, &mut arr);
        greedy.apply(&mut arr);
        let tok = arr.selected_token().expect("selector picked a token");
        if model.is_eog(tok) {
            break;
        }
        let piece = model.detokenize(&[tok]).unwrap_or_default();
        // Every ASCII char the grammar let through must be a digit. (Skip the
        // check for byte/partial-UTF-8 pieces, which the grammar handles at the
        // byte level.)
        if piece.is_ascii() {
            assert!(
                !piece.is_empty() && piece.chars().all(|c| c.is_ascii_digit()),
                "grammar let a non-digit piece through: {piece:?}"
            );
        }
        produced.push_str(&piece);

        grammar.accept_token(&mut ctx, tok);

        batch.clear();
        batch.add(tok, n_past, &[0], true).expect("add token");
        n_past += 1;
        ctx.decode(&mut batch).expect("decode token");
        logits_idx = 0;
    }

    // try_clone must yield an independent, usable grammar.
    let cloned = grammar.try_clone().expect("clone grammar");
    drop(cloned);

    // An invalid grammar must fail to build rather than panic.
    assert!(
        LlamaGrammar::new(&model, "root ::= (((", "root").is_err(),
        "malformed GBNF should fail to parse"
    );

    println!("GRAMMAR SMOKE OK: digit-constrained output = {produced:?}");
}

/// Tool-calling path: a JSON Schema is converted to a GBNF grammar, the grammar
/// compiles, and it constrains generation to a JSON object. `common` feature.
#[cfg(feature = "common")]
#[test]
fn json_schema_to_grammar_constrains_output_to_json() {
    use ik_llama_cpp_2::json_schema_to_grammar;

    let (_backend, model) = setup();

    // A function/tool argument schema, as used for tool calling.
    let schema = r#"{
        "type": "object",
        "properties": {
            "city": { "type": "string" },
            "unit": { "enum": ["c", "f"] }
        },
        "required": ["city"]
    }"#;
    let grammar_str = json_schema_to_grammar(schema).expect("schema -> grammar");
    assert!(
        grammar_str.contains("root ::="),
        "converted grammar missing a root rule:\n{grammar_str}"
    );

    // Invalid JSON must surface an error, not panic.
    assert!(
        json_schema_to_grammar("{ not valid json").is_err(),
        "invalid schema should fail to convert"
    );

    let cparams = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(2048))
        .with_n_threads(8)
        .with_seed(3);
    let mut ctx = LlamaContext::new(&model, &cparams).expect("context");

    let prompt = model
        .tokenize("Return the weather query as JSON: ", true)
        .expect("tokenize");
    let mut batch = LlamaBatch::new(prompt.len().max(64), 1);
    batch.add_sequence(&prompt, 0, false).expect("add prompt");
    ctx.decode(&mut batch).expect("decode prompt");
    let last = batch.n_tokens() - 1;

    // The converted grammar must itself round-trip into the constraint engine.
    let mut grammar =
        LlamaGrammar::new(&model, &grammar_str, "root").expect("build grammar from schema");

    let mut greedy = LlamaSampler::greedy();
    let mut n_past = batch.n_tokens();
    let mut logits_idx = last;
    let mut produced = String::new();

    for _ in 0..12 {
        let logits = ctx.get_logits_ith(logits_idx).expect("logits").to_vec();
        let mut arr = LlamaTokenDataArray::from_logits(&logits);
        grammar.apply(&mut ctx, &mut arr);
        greedy.apply(&mut arr);
        let tok = arr.selected_token().expect("selector picked a token");
        if model.is_eog(tok) {
            break;
        }
        produced.push_str(&model.detokenize(&[tok]).unwrap_or_default());
        grammar.accept_token(&mut ctx, tok);
        batch.clear();
        batch.add(tok, n_past, &[0], true).expect("add token");
        n_past += 1;
        ctx.decode(&mut batch).expect("decode token");
        logits_idx = 0;
    }

    // An object-typed schema forces the output to open a JSON object. (Tolerant
    // of empty/partial-UTF-8 prefixes from a 1.5bpw model.)
    let trimmed = produced.trim_start();
    if trimmed.is_ascii() && !trimmed.is_empty() {
        assert!(
            trimmed.starts_with('{'),
            "object-schema grammar should force a JSON object, got {produced:?}"
        );
    }
    println!("JSON-SCHEMA SMOKE OK: grammar constrained output = {produced:?}");
}

/// The chain-object grammar sampler (`LlamaSampler::grammar`) applies + advances
/// a GBNF constraint context-free (via the core glue), constraining greedy
/// output to digits.
#[test]
fn sampler_grammar_constrains_via_glue() {
    let (_backend, model) = setup();

    let cparams = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(2048))
        .with_n_threads(8)
        .with_seed(11);
    let mut ctx = LlamaContext::new(&model, &cparams).expect("context");

    let prompt = model
        .str_to_token(
            "The number of days in a week is",
            ik_llama_cpp_2::AddBos::Always,
        )
        .expect("tokenize");
    let mut batch = LlamaBatch::new(prompt.len().max(64), 1);
    batch.add_sequence(&prompt, 0, false).expect("add prompt");
    ctx.decode(&mut batch).expect("decode prompt");
    let last = batch.n_tokens() - 1;

    // grammar stage (ctx-free) + greedy selector, composed as a chain.
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::grammar(&model, "root ::= [0-9]+", "root").expect("grammar stage"),
        LlamaSampler::greedy(),
    ]);

    let mut n_past = batch.n_tokens();
    let mut logits_idx = last;
    let mut produced = String::new();
    for _ in 0..6 {
        let mut arr = ctx.token_data_array_ith(logits_idx);
        sampler.apply(&mut arr);
        let tok = arr.selected_token().expect("selector picked a token");
        if model.is_eog_token(tok) {
            break;
        }
        let piece = model.token_to_piece_lossy(tok).unwrap_or_default();
        if piece.is_ascii() {
            assert!(
                !piece.is_empty() && piece.chars().all(|c| c.is_ascii_digit()),
                "sampler grammar let a non-digit piece through: {piece:?}"
            );
        }
        produced.push_str(&piece);
        sampler.accept(tok);
        batch.clear();
        batch.add(tok, n_past, &[0], true).expect("add token");
        n_past += 1;
        ctx.decode(&mut batch).expect("decode token");
        logits_idx = 0;
    }

    // malformed GBNF must fail to build, not panic.
    assert!(
        LlamaSampler::grammar(&model, "root ::= (((", "root").is_err(),
        "malformed GBNF should fail"
    );
    println!("SAMPLER-GRAMMAR SMOKE OK: digit-constrained output = {produced:?}");
}

#[test]
fn dry_sampler_applies_and_tracks_history() {
    let (_backend, model) = setup();

    let cparams = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(2048))
        .with_n_threads(8)
        .with_seed(7);
    let mut ctx = LlamaContext::new(&model, &cparams).expect("context");

    let prompt = model
        .tokenize("one two three four", true)
        .expect("tokenize");
    let mut batch = LlamaBatch::new(prompt.len().max(64), 1);
    batch.add_sequence(&prompt, 0, false).expect("add prompt");
    ctx.decode(&mut batch).expect("decode prompt");
    let last = batch.n_tokens() - 1;

    let params = DryParams {
        multiplier: 0.8,
        base: 1.75,
        allowed_length: 2,
        penalty_last_n: 256,
        ..DryParams::default()
    };
    let mut dry = LlamaDrySampler::new(&model, &params).expect("build dry sampler");
    // Seed the history with the prompt so DRY has something to penalize.
    for &t in &prompt {
        dry.accept(t);
    }

    let n_vocab = model.n_vocab();
    let mut greedy = LlamaSampler::greedy();
    let mut n_past = batch.n_tokens();
    let mut logits_idx = last;
    let mut generated = 0usize;

    for _ in 0..8 {
        let logits = ctx.get_logits_ith(logits_idx).expect("logits").to_vec();
        let mut arr = LlamaTokenDataArray::from_logits(&logits);
        dry.apply(&mut ctx, &mut arr);
        greedy.apply(&mut arr);
        let tok = arr.selected_token().expect("selector picked a token");
        assert!(
            tok.raw() >= 0 && tok.raw() < n_vocab,
            "DRY produced an out-of-range token id {}",
            tok.raw()
        );
        dry.accept(tok);
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

    // reset + clone lifecycle must not crash / leak.
    dry.reset();
    let cloned = dry.try_clone().expect("clone dry sampler");
    drop(cloned);

    assert!(generated >= 1, "should have generated at least one token");
    println!("DRY SMOKE OK: generated={generated}");
}
