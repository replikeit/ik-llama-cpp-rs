//! `send_logs_to_tracing` end-to-end: a real model load must land in the tracing
//! subscriber and *nothing* of it may leak to raw stderr.
//!
//! The stderr half can only be checked at process level (the C side writes to
//! fd 2 directly, which libtest's capture does not intercept), so the test
//! re-executes itself as a child and inspects the child's stderr pipe.
//!
//! Gated behind the `_smoke` feature and the `IK_TEST_MODEL` env var.
#![cfg(feature = "_smoke")]

use std::sync::{Arc, Mutex};

use ik_llama_cpp_2::{
    send_logs_to_tracing, LlamaBackend, LlamaModel, LlamaModelParams, LogOptions,
};

const CHILD_MARKER: &str = "IK_TEST_LOG_BRIDGE_CHILD";
const TEST_NAME: &str = "model_load_logs_reach_tracing_and_not_stderr";

/// ik logs as `LLAMA_LOG_*("%s: ...", __func__)`, i.e. `load_tensors: ...`.
/// Matching the *shape* rather than specific lines keeps this test alive across
/// fork updates.
fn looks_like_c_log(line: &str) -> bool {
    match line.split_once(": ") {
        Some((head, _)) => {
            !head.is_empty()
                && head.len() < 64
                && head
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        }
        None => false,
    }
}

#[test]
fn model_load_logs_reach_tracing_and_not_stderr() {
    if std::env::var_os(CHILD_MARKER).is_some() {
        return child();
    }

    let out = std::process::Command::new(std::env::current_exe().expect("current_exe"))
        .args([TEST_NAME, "--exact"])
        .env(CHILD_MARKER, "1")
        .output()
        .expect("re-exec test binary");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "child failed ({:?})\n--- stdout ---\n{}\n--- stderr ---\n{stderr}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
    );

    let leaked: Vec<&str> = stderr.lines().filter(|l| looks_like_c_log(l)).collect();
    assert!(
        leaked.is_empty(),
        "{} C log line(s) still went to raw stderr, first few: {:?}",
        leaked.len(),
        &leaked[..leaked.len().min(5)],
    );
}

/// Runs in the re-executed child: bridge installed *before* backend init, then a
/// full model load, then assert the subscriber saw it.
fn child() {
    let logs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));

    // A global subscriber, not `set_default`: ik logs from its own worker threads
    // and a thread-local dispatcher would miss those events.
    let writer = VecWriter(logs.clone());
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .without_time()
        .with_target(false)
        .with_level(false)
        .with_writer(move || writer.clone())
        .init();

    send_logs_to_tracing(LogOptions::default());

    let backend = LlamaBackend::init().expect("backend");
    let path = std::env::var("IK_TEST_MODEL").expect("set IK_TEST_MODEL to a merged GGUF path");
    let model = LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default())
        .expect("load model");
    assert!(model.n_vocab() > 0);

    let captured = logs.lock().unwrap();
    let c_logs = captured.iter().filter(|l| looks_like_c_log(l)).count();
    // A model load is hundreds of lines; 20 is a floor that only a broken bridge
    // (or a total-silence fork) could miss.
    assert!(
        c_logs >= 20,
        "bridge captured only {c_logs} C log line(s) out of {} events: {:?}",
        captured.len(),
        &captured[..captured.len().min(10)],
    );
}

#[derive(Clone)]
struct VecWriter(Arc<Mutex<Vec<String>>>);

impl std::io::Write for VecWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap()
            .push(String::from_utf8_lossy(buf).trim().to_string());
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
