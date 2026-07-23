//! Safe wrapper around `llama_context` ([`LlamaContext`]).

pub mod embeddings;
pub mod kv_cache;
pub mod params;
pub mod session;

use std::marker::PhantomData;
use std::ptr::NonNull;

use ik_llama_cpp_sys as sys;

use crate::llama_batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::speculative::MtpOpType;
use crate::LlamaError;

pub use params::LlamaContextParams;

/// An inference context bound to a [`LlamaModel`].
///
/// The lifetime `'a` ties the context to its model (the model must outlive it).
#[derive(Debug)]
pub struct LlamaContext<'a> {
    pub(crate) context: NonNull<sys::llama_context>,
    n_vocab: i32,
    /// The raw params the context was built with (needed to derive the MTP
    /// companion context in the speculative glue; only read under `common`).
    #[cfg_attr(not(feature = "common"), allow(dead_code))]
    pub(crate) raw_params: sys::llama_context_params,
    _model: PhantomData<&'a LlamaModel>,
}

impl<'a> LlamaContext<'a> {
    /// Create a context for `model`.
    pub fn new(model: &'a LlamaModel, params: &LlamaContextParams) -> Result<Self, LlamaError> {
        // SAFETY: valid model ptr + initialized params.
        let raw = unsafe { sys::llama_init_from_model(model.model.as_ptr(), params.params) };
        NonNull::new(raw)
            .map(|context| Self {
                context,
                n_vocab: model.n_vocab(),
                raw_params: params.params,
                _model: PhantomData,
            })
            .ok_or(LlamaError::ContextCreation)
    }

    /// Run a decode over `batch`. Errors on a non-zero status from `llama_decode`.
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), LlamaError> {
        // SAFETY: valid ctx + batch owned by the caller.
        let ret = unsafe { sys::llama_decode(self.context.as_ptr(), batch.as_raw()) };
        if ret != 0 {
            return Err(LlamaError::Decode(ret));
        }
        Ok(())
    }

    /// Logits for batch index `i` of the last decode (length = `n_vocab`).
    ///
    /// `i` is the **batch position** (0..n_tokens), not a logits-rank; that entry
    /// must have been added with `logits = true` (else ik returns null → `NoLogits`).
    pub fn get_logits_ith(&self, i: i32) -> Result<&[f32], LlamaError> {
        // SAFETY: ptr valid until the next decode; slice length is n_vocab.
        let ptr = unsafe { sys::llama_get_logits_ith(self.context.as_ptr(), i) };
        if ptr.is_null() {
            return Err(LlamaError::NoLogits(i));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.n_vocab as usize) })
    }

    /// Candidate token data for batch index `i` (id + logit, `p = 0`).
    ///
    /// Returns an empty iterator if logits are unavailable at `i` (e.g. that
    /// position was not decoded with `logits = true`).
    pub fn candidates_ith(
        &self,
        i: i32,
    ) -> impl Iterator<Item = crate::token::data::LlamaTokenData> + '_ {
        let logits = self.get_logits_ith(i).unwrap_or(&[]);
        (0_i32..).zip(logits).map(|(idx, &logit)| {
            crate::token::data::LlamaTokenData::new(crate::token::LlamaToken::new(idx), logit, 0.0)
        })
    }

    /// Build a [`LlamaTokenDataArray`](crate::token::data_array::LlamaTokenDataArray)
    /// from the logits at batch index `i` (matches `llama-cpp-2`). Empty if
    /// logits are unavailable at `i`.
    #[must_use]
    pub fn token_data_array_ith(&self, i: i32) -> crate::token::data_array::LlamaTokenDataArray {
        crate::token::data_array::LlamaTokenDataArray::from_iter(self.candidates_ith(i), false)
    }

    /// The context size (tokens).
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { sys::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Set the MTP operation mode for subsequent decodes.
    ///
    /// Only meaningful for a context created with `.with_mtp(true)` on a model
    /// loaded with `.with_mtp(true)` (a NextN model). Drives the low-level MTP
    /// state machine (`llama_set_mtp_op_type`).
    pub fn set_mtp_op_type(&mut self, op: MtpOpType) {
        unsafe { sys::llama_set_mtp_op_type(self.context.as_ptr(), op.to_raw()) };
    }

    /// Vocabulary size (cached from the model at creation).
    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        self.n_vocab
    }

    /// Raw context pointer (used by the sampling module; escape hatch).
    #[must_use]
    pub(crate) fn as_ptr(&self) -> *mut sys::llama_context {
        self.context.as_ptr()
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe { sys::llama_free(self.context.as_ptr()) };
    }
}
