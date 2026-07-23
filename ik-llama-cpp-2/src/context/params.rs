//! Context parameters ([`LlamaContextParams`]) over ik's `llama_context_params`.

use std::num::NonZeroU32;

use ik_llama_cpp_sys as sys;

/// The kind of context to create.
///
/// Mirrors `llama-cpp-2`'s `context::params::LlamaContextType`. ik has no
/// `llama_context_type` field — instead it toggles MTP via a `bool mtp`, so
/// this shim maps [`LlamaContextType::Mtp`] onto that flag.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaContextType {
    /// Default decoder context.
    Default,
    /// Multi-token-prediction (NextN) draft context.
    Mtp,
}

/// Parameters controlling a [`crate::LlamaContext`].
///
/// Starts from `llama_context_default_params()`. Note ik keeps a `seed` field in
/// the context params (stock removed it) and uses a `bool flash_attn` plus the
/// MTP fields `mtp` / `mtp_op_type`. There is no `ctx_type` in ik.
#[derive(Debug, Clone)]
pub struct LlamaContextParams {
    pub(crate) params: sys::llama_context_params,
}

impl Default for LlamaContextParams {
    fn default() -> Self {
        Self {
            params: unsafe { sys::llama_context_default_params() },
        }
    }
}

impl LlamaContextParams {
    /// Context size (tokens); `None` means "take it from the model" (0).
    ///
    /// Takes `Option<NonZeroU32>` to match `llama-cpp-2`.
    #[must_use]
    pub fn with_n_ctx(mut self, n_ctx: Option<NonZeroU32>) -> Self {
        self.params.n_ctx = n_ctx.map_or(0, NonZeroU32::get);
        self
    }

    /// Logical batch size.
    #[must_use]
    pub fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.params.n_batch = n_batch;
        self
    }

    /// Physical (micro) batch size.
    #[must_use]
    pub fn with_n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.params.n_ubatch = n_ubatch;
        self
    }

    /// Maximum number of sequences (distinct recurrent states).
    ///
    /// `llama-cpp-2`'s NextN fork exposes this as `n_rs_seq`; ik's equivalent is
    /// `n_seq_max`, which this sets.
    #[must_use]
    pub fn with_n_rs_seq(mut self, n_rs_seq: u32) -> Self {
        self.params.n_seq_max = n_rs_seq;
        self
    }

    /// Select the context kind. [`LlamaContextType::Mtp`] enables ik's MTP path
    /// (equivalent to [`Self::with_mtp(true)`](Self::with_mtp)).
    #[must_use]
    pub fn with_context_type(mut self, context_type: LlamaContextType) -> Self {
        self.params.mtp = matches!(context_type, LlamaContextType::Mtp);
        self
    }

    /// RNG seed (ik retains this in the context params).
    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.params.seed = seed;
        self
    }

    /// Threads used for generation.
    #[must_use]
    pub fn with_n_threads(mut self, n_threads: u32) -> Self {
        self.params.n_threads = n_threads;
        self
    }

    /// Threads used for batch/prompt processing.
    #[must_use]
    pub fn with_n_threads_batch(mut self, n_threads_batch: u32) -> Self {
        self.params.n_threads_batch = n_threads_batch;
        self
    }

    /// Enable flash attention.
    #[must_use]
    pub fn with_flash_attn(mut self, flash_attn: bool) -> Self {
        self.params.flash_attn = flash_attn;
        self
    }

    /// Activate the MTP path (requires a model loaded with `.with_mtp(true)`).
    #[must_use]
    pub fn with_mtp(mut self, mtp: bool) -> Self {
        self.params.mtp = mtp;
        self
    }

    /// Produce embeddings on decode (sets `params.embeddings`).
    ///
    /// Required for embedding and reranker models; usually paired with
    /// [`Self::with_pooling_type`].
    #[must_use]
    pub fn with_embeddings(mut self, embeddings: bool) -> Self {
        self.params.embeddings = embeddings;
        self
    }

    /// Set the pooling strategy for embeddings (sets `params.pooling_type`).
    ///
    /// Takes the raw `sys::llama_pooling_type` (e.g. `LLAMA_POOLING_TYPE_MEAN`,
    /// `LLAMA_POOLING_TYPE_CLS`, or `LLAMA_POOLING_TYPE_LAST`) to avoid
    /// introducing a new enum.
    #[must_use]
    pub fn with_pooling_type(mut self, pooling_type: sys::llama_pooling_type) -> Self {
        self.params.pooling_type = pooling_type;
        self
    }

    /// Access the raw params (advanced/escape hatch).
    #[must_use]
    pub fn as_raw(&self) -> &sys::llama_context_params {
        &self.params
    }
}
