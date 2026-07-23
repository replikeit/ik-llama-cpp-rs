//! Process-wide ik_llama.cpp backend initialization.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::LlamaError;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// RAII proof that the ik_llama.cpp backend has been initialized.
///
/// Initialization is a process-wide singleton (like llama.cpp itself). Hold one
/// `LlamaBackend` for the lifetime of your models/contexts; dropping it calls
/// `llama_backend_free`.
#[derive(Debug)]
pub struct LlamaBackend {
    _private: (),
}

impl LlamaBackend {
    /// Initialize the backend. Returns [`LlamaError::BackendAlreadyInitialized`]
    /// if a `LlamaBackend` is already live in this process.
    pub fn init() -> Result<Self, LlamaError> {
        if INITIALIZED.swap(true, Ordering::SeqCst) {
            return Err(LlamaError::BackendAlreadyInitialized);
        }
        // SAFETY: guarded by the atomic above; called at most once at a time.
        unsafe { ik_llama_cpp_sys::llama_backend_init() };
        Ok(Self { _private: () })
    }
}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        // SAFETY: matches the single `llama_backend_init` above.
        unsafe { ik_llama_cpp_sys::llama_backend_free() };
        INITIALIZED.store(false, Ordering::SeqCst);
    }
}
