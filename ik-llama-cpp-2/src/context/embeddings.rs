//! Embeddings + reranker support: the `llama_get_embeddings*` accessors.
//!
//! Embeddings are produced when the context is built with
//! `LlamaContextParams::with_embeddings(true)` and (usually) a pooling type set
//! via `with_pooling_type`. Fetch a per-token vector with
//! [`LlamaContext::embeddings_ith`] or a pooled per-sequence vector with
//! [`LlamaContext::embeddings_seq_ith`]; each slice is `n_embd` long.
//!
//! A **reranker** is just an embedding model run with a rank/classification
//! pooling head: the pooled "embedding" of a query+document sequence is a
//! relevance score. Read it with [`LlamaContext::embeddings_seq_ith`] after a
//! decode (with a rank pooling type, the slice collapses to the score). ik does
//! not expose a dedicated `LLAMA_POOLING_TYPE_RANK` constant — the available
//! pooling types are `NONE`, `MEAN`, `CLS`, and `LAST`.

use ik_llama_cpp_sys as sys;

use crate::context::LlamaContext;

/// Errors returned by the embedding accessors.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum EmbeddingsError {
    /// The context was created without `with_embeddings(true)`.
    #[error("embeddings were not enabled in the context params")]
    NotEnabled,
    /// No embeddings for the given token: its batch entry did not request
    /// logits/embeddings, or the index was out of range.
    #[error("no embeddings available for token index {0}")]
    LogitsNotEnabled(i32),
    /// No per-sequence embeddings for the given sequence: the pooling type is
    /// `LLAMA_POOLING_TYPE_NONE`, or the sequence id was out of range.
    #[error("no sequence embeddings for seq {0} (pooling type NONE, or out of range)")]
    NonePoolType(i32),
}

impl LlamaContext<'_> {
    /// The model's embedding dimension (`n_embd`).
    ///
    /// ik exposes only `llama_model_n_embd` (there is no plain `llama_n_embd`),
    /// so we resolve the model from the context pointer first. The context does
    /// not hold a model reference, hence the round-trip through `llama_get_model`.
    #[must_use]
    fn n_embd(&self) -> usize {
        // SAFETY: the context owns a valid model pointer for its whole lifetime.
        let n = unsafe { sys::llama_model_n_embd(sys::llama_get_model(self.context.as_ptr())) };
        usize::try_from(n).unwrap_or(0)
    }

    /// Get the embeddings for the `i`th token in the current context.
    ///
    /// # Returns
    ///
    /// A slice with the embeddings for the last decoded batch of the given
    /// token. The size corresponds to the `n_embd` of the context's model.
    ///
    /// # Errors
    ///
    /// - [`EmbeddingsError::NotEnabled`] if the context was built without
    ///   `with_embeddings(true)`.
    /// - [`EmbeddingsError::LogitsNotEnabled`] if that token was not decoded
    ///   with logits enabled (ik returns a null pointer), or `i` is out of range.
    pub fn embeddings_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.raw_params.embeddings {
            return Err(EmbeddingsError::NotEnabled);
        }
        let n_embd = self.n_embd();
        // SAFETY: ptr valid until the next decode; slice length is n_embd.
        let ptr = unsafe { sys::llama_get_embeddings_ith(self.context.as_ptr(), i) };
        if ptr.is_null() {
            return Err(EmbeddingsError::LogitsNotEnabled(i));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, n_embd) })
    }

    /// Get the pooled embeddings for the `seq`th sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice with the pooled embeddings for the last decoded batch. The size
    /// corresponds to the `n_embd` of the context's model.
    ///
    /// # Errors
    ///
    /// - [`EmbeddingsError::NotEnabled`] if the context was built without
    ///   `with_embeddings(true)`.
    /// - [`EmbeddingsError::NonePoolType`] if the model uses
    ///   `LLAMA_POOLING_TYPE_NONE` (ik returns a null pointer), or `seq` exceeds
    ///   the max sequence id.
    pub fn embeddings_seq_ith(&self, seq: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.raw_params.embeddings {
            return Err(EmbeddingsError::NotEnabled);
        }
        let n_embd = self.n_embd();
        // SAFETY: ptr valid until the next decode; slice length is n_embd.
        let ptr = unsafe { sys::llama_get_embeddings_seq(self.context.as_ptr(), seq) };
        if ptr.is_null() {
            return Err(EmbeddingsError::NonePoolType(seq));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, n_embd) })
    }
}
