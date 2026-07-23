//! Negative-path test: bad inputs produce typed errors, not panics/UB.
//!
//! Gated behind `_smoke` + `IK_TEST_MODEL` (a valid model is loaded to prove the
//! happy path, then invalid loads are checked).
#![cfg(feature = "_smoke")]

use ik_llama_cpp_2::{LlamaBackend, LlamaError, LlamaModel, LlamaModelParams};

fn model_path() -> String {
    std::env::var("IK_TEST_MODEL").expect("set IK_TEST_MODEL")
}

#[test]
fn invalid_loads_return_typed_errors() {
    let backend = LlamaBackend::init().expect("backend init");

    // happy path still works
    let ok = LlamaModel::load_from_file(&backend, model_path(), &LlamaModelParams::default());
    assert!(ok.is_ok(), "valid model should load");
    drop(ok);

    // missing file -> ModelLoad error (no panic)
    let missing = LlamaModel::load_from_file(
        &backend,
        "/nonexistent/definitely-not-a-model.gguf",
        &LlamaModelParams::default(),
    );
    assert!(
        matches!(missing, Err(LlamaError::ModelLoad(_))),
        "missing file should be ModelLoad error, got {missing:?}"
    );

    // a real file that is not a GGUF -> ModelLoad error (no panic)
    let not_gguf = LlamaModel::load_from_file(
        &backend,
        concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"),
        &LlamaModelParams::default(),
    );
    assert!(
        matches!(not_gguf, Err(LlamaError::ModelLoad(_))),
        "non-gguf should be ModelLoad error, got {not_gguf:?}"
    );

    // interior NUL in a path -> Nul error
    let nul = LlamaModel::load_from_file(&backend, "bad\0path.gguf", &LlamaModelParams::default());
    assert!(
        matches!(nul, Err(LlamaError::Nul(_))),
        "NUL path should be Nul error, got {nul:?}"
    );

    println!("NEGATIVE OK");
}
