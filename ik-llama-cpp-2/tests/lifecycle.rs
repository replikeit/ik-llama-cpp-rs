//! Lifecycle test: repeated load → context → decode → teardown, no leak/crash.
//!
//! Gated behind `_smoke` + `IK_TEST_MODEL`. A single `#[test]` runs the cycles
//! sequentially (the backend is a process-wide singleton).
#![cfg(feature = "_smoke")]

use ik_llama_cpp_2::{
    LlamaBackend, LlamaBatch, LlamaContext, LlamaContextParams, LlamaModel, LlamaModelParams,
};

fn model_path() -> String {
    std::env::var("IK_TEST_MODEL").expect("set IK_TEST_MODEL")
}

#[test]
fn load_context_decode_teardown_repeat() {
    for cycle in 0..2 {
        let backend = LlamaBackend::init().expect("backend init");
        {
            let model =
                LlamaModel::load_from_file(&backend, model_path(), &LlamaModelParams::default())
                    .expect("load model");
            {
                let mut ctx = LlamaContext::new(
                    &model,
                    &LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512)),
                )
                .expect("context");
                let toks = model.tokenize("hello", true).expect("tokenize");
                let mut batch = LlamaBatch::new(toks.len().max(8), 1);
                batch.add_sequence(&toks, 0, false).expect("add");
                ctx.decode(&mut batch).expect("decode");
                // ctx dropped here
            }
            // model dropped here
        }
        // backend dropped here (frees + resets the singleton)
        println!("lifecycle cycle {cycle} OK");
    }
    // A second `init` after the first was dropped must succeed.
    let backend = LlamaBackend::init().expect("re-init after drop");
    drop(backend);
    println!("LIFECYCLE OK");
}
