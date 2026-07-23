//! Batch B smoke: model metadata accessors (exercises the two-call FFI string
//! buffer pattern against a real model). Gated `_smoke` + `IK_TEST_MODEL`.
#![cfg(feature = "_smoke")]

use ik_llama_cpp_2::{LlamaBackend, LlamaModel, LlamaModelParams};

fn model_path() -> String {
    std::env::var("IK_TEST_MODEL").expect("set IK_TEST_MODEL")
}

#[test]
fn model_metadata_accessors() {
    let backend = LlamaBackend::init().expect("backend init");
    let model =
        LlamaModel::load_from_file(&backend, model_path(), &LlamaModelParams::default()).unwrap();

    assert_eq!(
        model.meta_val_str("general.architecture").as_deref(),
        Some("qwen35"),
        "architecture metadata"
    );
    assert!(model.meta_count() > 0, "meta_count");
    assert!(model.n_embd() > 0, "n_embd");
    assert!(model.n_layer() > 0, "n_layer");
    assert!(model.n_ctx_train() > 0, "n_ctx_train");
    assert!(!model.desc().is_empty(), "desc");
    assert!(model.n_params() > 0, "n_params");

    // key-by-index roundtrip: the first key must be readable + re-findable.
    let k0 = model.meta_key_by_index(0).expect("key 0");
    assert!(!k0.is_empty());
    assert!(model.meta_val_str(&k0).is_some(), "value for {k0}");

    println!(
        "BATCH-B META OK: arch=qwen35 n_embd={} n_layer={} n_ctx_train={} meta_count={} params={}",
        model.n_embd(),
        model.n_layer(),
        model.n_ctx_train(),
        model.meta_count(),
        model.n_params()
    );
}
