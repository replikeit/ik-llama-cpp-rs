//! Tokenizer test across a mixed domain: ASCII, Cyrillic, Unicode
//! punctuation, JSON, tool-call-like payloads, filesystem paths + CLI flags.
//!
//! Asserts tokenize produces tokens and detokenize round-trips the meaningful
//! content (exact byte-identity is not guaranteed for all normalizers, so we
//! check non-emptiness + substring survival).
#![cfg(feature = "_smoke")]

use ik_llama_cpp_2::{LlamaBackend, LlamaModel, LlamaModelParams};

fn model_path() -> String {
    std::env::var("IK_TEST_MODEL").expect("set IK_TEST_MODEL")
}

#[test]
fn tokenize_roundtrip_domain_strings() {
    let backend = LlamaBackend::init().expect("backend init");
    let model =
        LlamaModel::load_from_file(&backend, model_path(), &LlamaModelParams::default()).unwrap();

    // (input, a substring that must survive the roundtrip)
    let cases: &[(&str, &str)] = &[
        ("Hello, world!", "Hello"),
        ("Привет, мир — как дела?", "мир"),
        ("Unicode: “quotes” — dash… café", "café"),
        (r#"{"tool":"disk_check","args":{"drive":"C:","deep":true}}"#, "disk_check"),
        (
            "<tool_call>{\"name\":\"restart_service\",\"arguments\":{\"svc\":\"spooler\"}}</tool_call>",
            "restart_service",
        ),
        ("Run `chkdsk C:\\ /f /r` then reboot; exit code 0x80070057.", "chkdsk"),
        ("/var/log/syslog:1234: error E_FAIL at --path=/etc/foo.conf", "syslog"),
    ];

    for (input, needle) in cases {
        let toks = model.tokenize(input, true).expect("tokenize");
        assert!(!toks.is_empty(), "no tokens for {input:?}");
        let out = model.detokenize(&toks).expect("detokenize");
        assert!(
            out.contains(needle),
            "roundtrip of {input:?} lost {needle:?}: got {out:?}"
        );
    }

    // BOS is prepended when requested, not when not.
    let with_bos = model.tokenize("test", true).unwrap();
    let without_bos = model.tokenize("test", false).unwrap();
    assert!(
        with_bos.len() >= without_bos.len(),
        "add_bos should not shrink the token count"
    );

    println!("TOKENIZER OK ({} cases)", cases.len());
}
