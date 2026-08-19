//! Bridge ik_llama.cpp's C log callback into [`tracing`].
//!
//! Without [`send_logs_to_tracing`] the C side writes straight to stderr — in
//! release builds too — which a consumer has no way to filter (a single model
//! load is ~330 lines of `llama_init_from_model:` / `load_tensors:`).
//!
//! One callback, one buffer: unlike upstream llama.cpp, this fork has no
//! `ggml_log_set`, so `llama_log_set` is the only sink and every message —
//! llama.cpp's and the active backend's — arrives through it in call order.
//! `GGML_LOG_LEVEL_CONT` therefore always continues the line this same buffer
//! is holding, and the two-state split upstream needs to keep llama.cpp and
//! ggml `CONT`s from interleaving has nothing to guard here.

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, OnceLock, PoisonError};

use ik_llama_cpp_sys as sys;

/// Options to configure how ik_llama.cpp logs are intercepted.
#[derive(Default, Debug, Clone)]
pub struct LogOptions {
    disabled: bool,
}

impl LogOptions {
    /// If enabled, logs are sent to tracing. If disabled, all logs are suppressed.
    /// Default is for logs to be sent to tracing.
    #[must_use]
    pub fn with_logs_enabled(mut self, enabled: bool) -> Self {
        self.disabled = !enabled;
        self
    }
}

/// Callback state, reachable from C via a raw pointer for the rest of the
/// process (see [`send_logs_to_tracing`]).
#[derive(Debug)]
struct State {
    disabled: bool,
    /// The partial line built so far plus the level it started at. ik emits a
    /// single log line over several calls (`GGML_LOG_LEVEL_CONT` continues the
    /// previous one), so text is buffered until a `\n` closes it.
    pending: Mutex<(sys::ggml_log_level, String)>,
}

impl State {
    fn new(options: LogOptions) -> Self {
        Self {
            disabled: options.disabled,
            pending: Mutex::new((sys::GGML_LOG_LEVEL_INFO, String::new())),
        }
    }
}

/// `tracing` wants a const level and a literal target, so dispatch by hand.
fn emit(level: sys::ggml_log_level, line: &str) {
    match level {
        sys::GGML_LOG_LEVEL_DEBUG => tracing::debug!(target: "ik_llama_cpp", "{line}"),
        sys::GGML_LOG_LEVEL_WARN => tracing::warn!(target: "ik_llama_cpp", "{line}"),
        sys::GGML_LOG_LEVEL_ERROR => tracing::error!(target: "ik_llama_cpp", "{line}"),
        // INFO, NONE (ik's own default callback prints those like plain output)
        // and anything a future fork adds.
        _ => tracing::info!(target: "ik_llama_cpp", "{line}"),
    }
}

extern "C" fn logs_to_trace(level: sys::ggml_log_level, text: *const c_char, data: *mut c_void) {
    // Unwinding across the FFI boundary is UB and a user's `tracing` subscriber
    // is arbitrary code, so nothing here is allowed to escape as a panic.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if text.is_null() || data.is_null() {
            return;
        }
        // SAFETY: `data` is the `&'static State` handed to `llama_log_set`, and
        // `text` is a NUL-terminated C string owned by the caller for the
        // duration of the callback.
        let state = unsafe { &*data.cast::<State>() };
        if state.disabled {
            return;
        }
        let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();

        // Poisoning only means a previous call panicked mid-buffer; keep bridging.
        let mut pending = state.pending.lock().unwrap_or_else(PoisonError::into_inner);
        let (level_of_buf, buf) = &mut *pending;
        if level != sys::GGML_LOG_LEVEL_CONT {
            // A non-CONT message abandons any half-line before it (a format
            // string in ik missing its trailing `\n`): flush it at its own level.
            if !buf.is_empty() {
                emit(*level_of_buf, buf);
                buf.clear();
            }
            *level_of_buf = level;
        }
        buf.push_str(&text);
        while let Some(nl) = buf.find('\n') {
            emit(*level_of_buf, &buf[..nl]);
            buf.drain(..=nl);
        }
    }));
}

/// Redirect ik_llama.cpp's C-side logs into [`tracing`].
///
/// Call this *before* [`LlamaBackend::init`](crate::LlamaBackend::init), or the
/// backend's own startup lines go to stderr before the callback is in place.
/// Only the first call takes effect — the state handed to C has to outlive every
/// later log call, so it is leaked into a `static` and never replaced.
///
/// # ggml coverage
///
/// ik's vendored ggml predates upstream's `ggml_log_set`, so there is no second
/// sink to install: `llama_log_set` is the whole hook. It covers llama.cpp's
/// logger and, in this fork, also hands the same callback to the active
/// backend's logger (CUDA / Metal / CANN). What it cannot reach is core
/// `ggml.c`'s own `fprintf(stderr, ...)` diagnostics, which this fork emits
/// unconditionally.
pub fn send_logs_to_tracing(options: LogOptions) {
    // A `static` OnceLock's storage is fixed for the whole program, so `&State`
    // from it is a pointer C can keep — no `Box` indirection needed.
    static STATE: OnceLock<State> = OnceLock::new();
    let state: *const State = STATE.get_or_init(|| State::new(options));
    // SAFETY: `logs_to_trace` is a valid `ggml_log_callback` and `state` points
    // at a `static` that lives for the rest of the process.
    unsafe { sys::llama_log_set(Some(logs_to_trace), state.cast_mut().cast()) };
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing::subscriber::DefaultGuard;
    use tracing_subscriber::util::SubscriberInitExt;

    use super::*;

    #[derive(Clone)]
    struct VecWriter(Arc<Mutex<Vec<String>>>);

    impl std::io::Write for VecWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap()
                // fmt pads the level to 5 chars ("` WARN`"); trim both ends.
                .push(String::from_utf8_lossy(buf).trim().to_string());
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Collects rendered `LEVEL message` lines from the bridge.
    fn capture(max: tracing::Level) -> (DefaultGuard, Arc<Mutex<Vec<String>>>) {
        let logs = Arc::new(Mutex::new(vec![]));
        let writer = VecWriter(logs.clone());
        let guard = tracing_subscriber::fmt()
            .with_max_level(max)
            .with_ansi(false)
            .without_time()
            .with_target(false)
            .with_writer(move || writer.clone())
            .finish()
            .set_default();
        (guard, logs)
    }

    /// Feeds bytes through the real `extern "C"` callback.
    fn feed(state: &State, calls: &[(sys::ggml_log_level, &str)]) {
        let ptr = (state as *const State).cast_mut().cast::<c_void>();
        for (level, text) in calls {
            let c = std::ffi::CString::new(*text).unwrap();
            logs_to_trace(*level, c.as_ptr(), ptr);
        }
    }

    #[test]
    fn cont_stitches_one_line_and_holds_until_newline() {
        let (_g, logs) = capture(tracing::Level::TRACE);
        let state = State::new(LogOptions::default());

        feed(
            &state,
            &[
                (sys::GGML_LOG_LEVEL_INFO, "load_tensors: "),
                (sys::GGML_LOG_LEVEL_CONT, "buffer size ="),
            ],
        );
        assert!(
            logs.lock().unwrap().is_empty(),
            "an unterminated line must not be emitted yet"
        );

        feed(&state, &[(sys::GGML_LOG_LEVEL_CONT, " 1234 MiB\n")]);
        assert_eq!(
            *logs.lock().unwrap(),
            vec!["INFO load_tensors: buffer size = 1234 MiB"]
        );
    }

    #[test]
    fn levels_map_and_multiple_lines_split_per_call() {
        let (_g, logs) = capture(tracing::Level::TRACE);
        let state = State::new(LogOptions::default());

        feed(
            &state,
            &[
                (sys::GGML_LOG_LEVEL_DEBUG, "dbg\n"),
                (sys::GGML_LOG_LEVEL_WARN, "warn\n"),
                (sys::GGML_LOG_LEVEL_ERROR, "err\n"),
                (sys::GGML_LOG_LEVEL_NONE, "plain\n"),
                (sys::GGML_LOG_LEVEL_INFO, "one\ntwo\n"),
            ],
        );
        assert_eq!(
            *logs.lock().unwrap(),
            vec![
                "DEBUG dbg",
                "WARN warn",
                "ERROR err",
                "INFO plain",
                "INFO one",
                "INFO two",
            ]
        );
    }

    #[test]
    fn non_cont_flushes_an_abandoned_half_line_at_its_own_level() {
        let (_g, logs) = capture(tracing::Level::TRACE);
        let state = State::new(LogOptions::default());

        feed(
            &state,
            &[
                (sys::GGML_LOG_LEVEL_WARN, "half line, no newline"),
                (sys::GGML_LOG_LEVEL_INFO, "next\n"),
            ],
        );
        assert_eq!(
            *logs.lock().unwrap(),
            vec!["WARN half line, no newline", "INFO next"]
        );
    }

    #[test]
    fn disabled_suppresses_everything() {
        let (_g, logs) = capture(tracing::Level::TRACE);
        let state = State::new(LogOptions::default().with_logs_enabled(false));

        feed(&state, &[(sys::GGML_LOG_LEVEL_ERROR, "boom\n")]);
        assert!(logs.lock().unwrap().is_empty());
    }

    #[test]
    fn invalid_utf8_and_null_pointers_do_not_panic() {
        let (_g, logs) = capture(tracing::Level::TRACE);
        let state = State::new(LogOptions::default());
        let ptr = (&state as *const State).cast_mut().cast::<c_void>();

        let bad = std::ffi::CString::new(b"caf\xff\n".to_vec()).unwrap();
        logs_to_trace(sys::GGML_LOG_LEVEL_INFO, bad.as_ptr(), ptr);
        logs_to_trace(sys::GGML_LOG_LEVEL_INFO, std::ptr::null(), ptr);
        logs_to_trace(
            sys::GGML_LOG_LEVEL_INFO,
            c"x\n".as_ptr(),
            std::ptr::null_mut(),
        );

        assert_eq!(*logs.lock().unwrap(), vec!["INFO caf\u{fffd}"]);
    }
}
