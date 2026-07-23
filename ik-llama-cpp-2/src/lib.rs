//! Safe Rust bindings for `ik_llama.cpp` (ikawrakow's SOTA-quant fork).
//!
//! Mirrors the API and codestyle of the `llama-cpp-2` crate
//! (`utilityai/llama-cpp-rs`), adapted for ik_llama.cpp's divergent C API:
//! legacy `llama_sample_*` sampling, model-pointer tokenizer, ik-specific
//! model/context params, and the MTP / NextN speculative path.
//!
//! v1 covers the general generation path (load → tokenize → decode → sample)
//! plus the MTP scaffolding.
#![allow(clippy::pedantic, clippy::module_name_repetitions)]

use std::ffi::NulError;

pub mod context;
pub mod gguf;
pub mod grammar;
pub mod llama_backend;
pub mod llama_batch;
pub mod model;
#[cfg(feature = "mtmd")]
pub mod mtmd;
pub mod quantize;
pub mod sampling;
pub mod speculative;
pub mod timing;
pub mod token;

pub use context::params::{LlamaContextParams, LlamaContextType};
pub use context::LlamaContext;
#[cfg(feature = "common")]
pub use grammar::{json_schema_to_grammar, JsonSchemaError};
pub use grammar::{DryInitError, DryParams, GrammarInitError, LlamaDrySampler, LlamaGrammar};
pub use llama_backend::LlamaBackend;
pub use llama_batch::LlamaBatch;
pub use model::params::LlamaModelParams;
pub use model::{AddBos, LlamaModel};
#[cfg(feature = "mtmd")]
pub use mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams,
    MtmdEncodeError, MtmdEvalError, MtmdInitError, MtmdInputChunk, MtmdInputChunkError,
    MtmdInputChunkType, MtmdInputChunks, MtmdInputChunksError, MtmdInputText, MtmdTokenizeError,
};
pub use sampling::LlamaSampler;
pub use speculative::{MtpOpType, MtpSpeculativeParams};
#[cfg(feature = "common")]
pub use speculative::{MtpSpeculative, MtpStep};
pub use token::data::LlamaTokenData;
pub use token::data_array::LlamaTokenDataArray;
pub use token::LlamaToken;

/// Errors returned by the safe wrapper.
#[derive(Debug, thiserror::Error)]
pub enum LlamaError {
    /// The global backend was already initialized (it is a process-wide singleton).
    #[error("llama backend already initialized")]
    BackendAlreadyInitialized,
    /// `llama_model_load_from_file` returned null for the given path.
    #[error("failed to load model from {0}")]
    ModelLoad(String),
    /// `llama_init_from_model` returned null.
    #[error("failed to create llama context")]
    ContextCreation,
    /// A path or prompt contained an interior NUL byte.
    #[error("string contained an interior NUL byte")]
    Nul(#[from] NulError),
    /// The model path was not valid UTF-8 / could not be converted to a C string.
    #[error("invalid model path")]
    InvalidPath,
    /// Tokenization failed (negative count from `llama_tokenize`).
    #[error("tokenization failed")]
    Tokenize,
    /// Detokenization produced invalid UTF-8 that could not be recovered.
    #[error("token could not be converted to text")]
    TokenToPiece,
    /// `llama_decode` returned a non-zero status.
    #[error("llama_decode failed with status {0}")]
    Decode(i32),
    /// A batch operation exceeded the batch's allocated capacity.
    #[error("batch capacity {capacity} exceeded (tried to add token {index})")]
    BatchOverflow {
        /// Allocated token capacity of the batch.
        capacity: usize,
        /// Index that overflowed.
        index: usize,
    },
    /// More sequence ids were supplied for a token than the batch's `n_seq_max`.
    #[error("too many seq_ids for a batch token: got {got}, max {max}")]
    TooManySeqIds {
        /// Number of seq_ids supplied.
        got: usize,
        /// Batch's configured `n_seq_max`.
        max: usize,
    },
    /// Requested logits for a position that was not computed / out of range.
    #[error("no logits available at index {0}")]
    NoLogits(i32),
    /// MTP speculative driver initialization failed (e.g. 0 NextN layers, or an
    /// openPangu/recurrent target, or a context not created with `.with_mtp(true)`).
    #[error("failed to initialize MTP speculative driver")]
    MtpInit,
    /// MTP begin (prompt warmup) failed.
    #[error("MTP begin failed with status {0}")]
    MtpBegin(i32),
    /// MTP step (draft/verify/accept/commit) failed.
    #[error("MTP step failed with status {0}")]
    MtpStep(i32),
}
