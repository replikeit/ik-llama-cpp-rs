//! Model loading parameters ([`LlamaModelParams`]) over ik's `llama_model_params`.

use ik_llama_cpp_sys as sys;

/// Parameters controlling how a model is loaded.
///
/// Starts from `llama_model_default_params()`; builder methods override the
/// fields v1 needs. ik's `llama_model_params` carries many fork-specific fields
/// (mla, ncmoe, fit, repack_tensors, per-layer K/V types, …); those keep their
/// defaults unless exposed here.
#[derive(Debug, Clone)]
pub struct LlamaModelParams {
    pub(crate) params: sys::llama_model_params,
}

impl Default for LlamaModelParams {
    fn default() -> Self {
        // SAFETY: returns a fully-initialized POD struct by value.
        Self {
            params: unsafe { sys::llama_model_default_params() },
        }
    }
}

impl LlamaModelParams {
    /// Number of layers to offload to the GPU (0 = CPU only).
    ///
    /// Takes `u32` to match `llama-cpp-2`; clamped into ik's `i32` field.
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n: u32) -> Self {
        self.params.n_gpu_layers = i32::try_from(n).unwrap_or(i32::MAX);
        self
    }

    /// Whether to memory-map the model file (default true).
    #[must_use]
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.params.use_mmap = use_mmap;
        self
    }

    /// Force the model into RAM (mlock).
    #[must_use]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.params.use_mlock = use_mlock;
        self
    }

    /// Load only the vocabulary (no weights).
    #[must_use]
    pub fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    /// Load the MTP / NextN prediction layers if present in the model.
    #[must_use]
    pub fn with_mtp(mut self, mtp: bool) -> Self {
        self.params.mtp = mtp;
        self
    }

    /// Access the raw params (advanced/escape hatch).
    #[must_use]
    pub fn as_raw(&self) -> &sys::llama_model_params {
        &self.params
    }
}
