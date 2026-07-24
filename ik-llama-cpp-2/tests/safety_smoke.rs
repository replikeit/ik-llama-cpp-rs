//! Safety-guard smoke tests: verify that the bounds/range guards added to the
//! safe API turn what would otherwise be a process abort or out-of-bounds read
//! into a normal `Err`/`None` — using a real model so the C side is exercised.
//!
//! Gated behind the `_smoke` feature and the `IK_TEST_MODEL` env var (any GGUF,
//! e.g. `.models/qwen35-4b-iq1s-general.gguf`).
#![cfg(feature = "_smoke")]

use ik_llama_cpp_2::{
    gguf::GgufContext, LlamaBackend, LlamaError, LlamaModel, LlamaModelParams, LlamaToken,
};

fn model_path() -> String {
    std::env::var("IK_TEST_MODEL").expect("set IK_TEST_MODEL to a GGUF path")
}

/// An out-of-vocab token id must be rejected as `TokenOutOfRange`, NOT forwarded
/// to ik's tokenizer (whose `cache.at(token)` would throw and abort the process
/// across the FFI boundary).
#[test]
fn out_of_range_token_is_error_not_abort() {
    let backend = LlamaBackend::init().expect("backend init");
    let model = LlamaModel::load_from_file(&backend, model_path(), &LlamaModelParams::default())
        .expect("load model");
    let n_vocab = model.n_vocab();
    assert!(n_vocab > 0);

    for bad in [n_vocab, i32::MAX, -1] {
        let err = model
            .token_to_piece_lossy(LlamaToken(bad))
            .expect_err("out-of-range token must error");
        assert!(
            matches!(err, LlamaError::TokenOutOfRange { token, n_vocab: nv } if token == bad && nv == n_vocab),
            "expected TokenOutOfRange for {bad}, got {err:?}"
        );
    }

    // detokenize funnels through the same guard.
    assert!(
        matches!(
            model.detokenize(&[LlamaToken(999_999_999)]),
            Err(LlamaError::TokenOutOfRange { .. })
        ),
        "detokenize with an out-of-range id must error"
    );

    // A valid token still converts.
    let tok = model
        .tokenize("hi", true)
        .expect("tokenize")
        .first()
        .copied()
        .expect("at least one token");
    assert!(
        model.token_to_piece_lossy(tok).is_ok(),
        "a valid token must still convert"
    );
}

/// Out-of-range tensor/element indices on `GgufContext` must return `None`
/// rather than driving an out-of-bounds read (ik's `gguf_get_tensor_*` have no
/// bounds assert).
#[test]
fn gguf_out_of_range_index_is_none_not_oob() {
    let path = model_path();
    let gguf = GgufContext::from_file(std::path::Path::new(&path)).expect("open gguf");

    let n = gguf.n_tensors();
    assert!(n > 0, "model should describe at least one tensor");

    // Valid index works.
    assert!(gguf.tensor_name(0).is_some(), "tensor 0 should have a name");
    assert!(gguf.tensor_type(0).is_some());
    assert!(gguf.tensor_offset(0).is_some());

    // Out-of-range indices are rejected, not read out of bounds.
    for bad in [n, n + 1, i32::MAX, -1] {
        assert!(gguf.tensor_name(bad).is_none(), "tensor_name({bad})");
        assert!(gguf.tensor_type(bad).is_none(), "tensor_type({bad})");
        assert!(gguf.tensor_offset(bad).is_none(), "tensor_offset({bad})");
    }
}
