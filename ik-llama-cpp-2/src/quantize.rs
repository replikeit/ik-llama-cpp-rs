//! Model quantization write-path over ik's `llama_model_quantize`.
//!
//! The anchor crate (`utilityai/llama-cpp-rs`, `llama-cpp-2`) ships no quantize
//! wrapper, so this is implemented fresh in the crate's module-local style
//! (own `thiserror` error, builder over the raw params struct, `as_raw` escape
//! hatch), mirroring [`crate::gguf`] and [`crate::model::params`].
//!
//! Quantizing a source GGUF (usually f16/bf16/higher precision) produces a
//! smaller GGUF at `output`. ik_llama.cpp extends the upstream params struct
//! with many fork-specific per-tensor override fields (`attn_*_type`,
//! `ffn_*_type`, `token_embedding_type`, …) plus extra flags; those keep the
//! values from `llama_model_quantize_default_params()` unless the caller reaches
//! through [`LlamaQuantizeParams::as_raw_mut`].

use std::ffi::{CString, NulError};
use std::path::Path;

use ik_llama_cpp_sys as sys;

/// Errors returned by [`quantize`].
#[derive(Debug, thiserror::Error)]
pub enum QuantizeError {
    /// An input or output path was not valid UTF-8 (cannot become a C string).
    #[error("quantize path was not valid UTF-8")]
    InvalidPath,
    /// A path contained an interior NUL byte.
    #[error("quantize path contained an interior NUL byte")]
    Nul(#[from] NulError),
    /// `llama_model_quantize` returned a non-zero status code.
    #[error("llama_model_quantize failed with status {0}")]
    Quantize(u32),
}

/// Common `llama_ftype` target types, re-exported from the sys crate.
///
/// These are the raw `ik_llama_cpp_sys::LLAMA_FTYPE_*` constants (each a
/// `llama_ftype`, i.e. a `u32`); the full set (84 variants) lives in the sys
/// crate and any of them may be passed to [`LlamaQuantizeParams::with_ftype`].
/// The list below is a curated subset covering the standard llama.cpp quants
/// plus ik_llama.cpp's SOTA "K"/"KS"/"KT" and repacked "R4/R8" families.
pub mod ftype {
    pub use ik_llama_cpp_sys::llama_ftype;

    // --- Standard llama.cpp float / legacy / K-quants ---
    pub use ik_llama_cpp_sys::{
        LLAMA_FTYPE_ALL_F32, LLAMA_FTYPE_MOSTLY_BF16, LLAMA_FTYPE_MOSTLY_F16,
        LLAMA_FTYPE_MOSTLY_Q2_K, LLAMA_FTYPE_MOSTLY_Q2_K_S, LLAMA_FTYPE_MOSTLY_Q3_K_L,
        LLAMA_FTYPE_MOSTLY_Q3_K_M, LLAMA_FTYPE_MOSTLY_Q3_K_S, LLAMA_FTYPE_MOSTLY_Q4_0,
        LLAMA_FTYPE_MOSTLY_Q4_1, LLAMA_FTYPE_MOSTLY_Q4_K_M, LLAMA_FTYPE_MOSTLY_Q4_K_S,
        LLAMA_FTYPE_MOSTLY_Q5_0, LLAMA_FTYPE_MOSTLY_Q5_1, LLAMA_FTYPE_MOSTLY_Q5_K_M,
        LLAMA_FTYPE_MOSTLY_Q5_K_S, LLAMA_FTYPE_MOSTLY_Q6_K, LLAMA_FTYPE_MOSTLY_Q8_0,
    };

    // --- Standard llama.cpp i-quants ---
    pub use ik_llama_cpp_sys::{
        LLAMA_FTYPE_MOSTLY_IQ1_M, LLAMA_FTYPE_MOSTLY_IQ1_S, LLAMA_FTYPE_MOSTLY_IQ2_M,
        LLAMA_FTYPE_MOSTLY_IQ2_S, LLAMA_FTYPE_MOSTLY_IQ2_XS, LLAMA_FTYPE_MOSTLY_IQ2_XXS,
        LLAMA_FTYPE_MOSTLY_IQ3_M, LLAMA_FTYPE_MOSTLY_IQ3_S, LLAMA_FTYPE_MOSTLY_IQ3_XXS,
        LLAMA_FTYPE_MOSTLY_IQ4_NL, LLAMA_FTYPE_MOSTLY_IQ4_XS,
    };

    // --- ik_llama.cpp SOTA quants (the reason this fork exists) ---
    pub use ik_llama_cpp_sys::{
        LLAMA_FTYPE_MOSTLY_IQ1_KT, LLAMA_FTYPE_MOSTLY_IQ2_K, LLAMA_FTYPE_MOSTLY_IQ2_KL,
        LLAMA_FTYPE_MOSTLY_IQ2_KS, LLAMA_FTYPE_MOSTLY_IQ2_KT, LLAMA_FTYPE_MOSTLY_IQ3_K,
        LLAMA_FTYPE_MOSTLY_IQ3_KL, LLAMA_FTYPE_MOSTLY_IQ3_KS, LLAMA_FTYPE_MOSTLY_IQ3_KT,
        LLAMA_FTYPE_MOSTLY_IQ4_K, LLAMA_FTYPE_MOSTLY_IQ4_KS, LLAMA_FTYPE_MOSTLY_IQ4_KSS,
        LLAMA_FTYPE_MOSTLY_IQ4_KT, LLAMA_FTYPE_MOSTLY_IQ5_K, LLAMA_FTYPE_MOSTLY_IQ5_KS,
        LLAMA_FTYPE_MOSTLY_IQ6_K, LLAMA_FTYPE_MOSTLY_Q8_KV,
    };
}

/// Parameters controlling a [`quantize`] run.
///
/// Starts from `llama_model_quantize_default_params()`; the builder methods
/// override the handful of fields common quantization runs need. ik's
/// per-tensor override fields (`output_tensor_type`, `token_embedding_type`,
/// `attn_q_type`, `attn_k_type`, `attn_v_type`, `attn_qkv_type`,
/// `attn_output_type`, `ffn_gate_type`, `ffn_down_type`, `ffn_up_type`,
/// `ffn_gate_inp_type`, `extra_output_type`, `per_layer_token_embedding_type`)
/// and the extra flags (`only_copy`, `pure_`, `keep_split`,
/// `ignore_imatrix_rules`, `only_repack`, `dry_run`, `partial_requant`) plus the
/// opaque pointers (`imatrix`, `kv_overrides`, `custom_quants`,
/// `repack_pattern`, `user_data`) keep their default values unless set through
/// [`as_raw_mut`](LlamaQuantizeParams::as_raw_mut).
#[derive(Debug, Clone)]
pub struct LlamaQuantizeParams {
    pub(crate) raw: sys::llama_model_quantize_params,
}

impl Default for LlamaQuantizeParams {
    fn default() -> Self {
        // SAFETY: returns a fully-initialized POD struct by value; pointer
        // fields are set to null by the C side.
        Self {
            raw: unsafe { sys::llama_model_quantize_default_params() },
        }
    }
}

impl LlamaQuantizeParams {
    /// Target quantization format (e.g. [`ftype::LLAMA_FTYPE_MOSTLY_Q4_K_M`] or
    /// an ik SOTA quant like [`ftype::LLAMA_FTYPE_MOSTLY_IQ4_K`]).
    #[must_use]
    pub fn with_ftype(mut self, ftype: sys::llama_ftype) -> Self {
        self.raw.ftype = ftype;
        self
    }

    /// Number of threads to use (0 lets the C side pick a default).
    #[must_use]
    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.raw.nthread = n_threads;
        self
    }

    /// Allow requantizing tensors that are already quantized (default false).
    #[must_use]
    pub fn with_allow_requantize(mut self, allow_requantize: bool) -> Self {
        self.raw.allow_requantize = allow_requantize;
        self
    }

    /// Also quantize the `output.weight` tensor (default false keeps it larger
    /// for quality).
    #[must_use]
    pub fn with_quantize_output_tensor(mut self, quantize_output_tensor: bool) -> Self {
        self.raw.quantize_output_tensor = quantize_output_tensor;
        self
    }

    /// Immutable access to the raw params (advanced/escape hatch).
    #[must_use]
    pub fn as_raw(&self) -> &sys::llama_model_quantize_params {
        &self.raw
    }

    /// Mutable access to the raw params, for setting ik's extended per-tensor
    /// override fields, flags, or the opaque `imatrix`/`kv_overrides` pointers
    /// that are not surfaced by dedicated builders.
    #[must_use]
    pub fn as_raw_mut(&mut self) -> &mut sys::llama_model_quantize_params {
        &mut self.raw
    }
}

/// Quantize the GGUF model at `input`, writing the result to `output`.
///
/// Wraps `llama_model_quantize(fname_inp, fname_out, *const params)`; a non-zero
/// return code becomes [`QuantizeError::Quantize`].
///
/// # Errors
/// - [`QuantizeError::InvalidPath`] if a path is not valid UTF-8.
/// - [`QuantizeError::Nul`] if a path contains an interior NUL byte.
/// - [`QuantizeError::Quantize`] if the underlying C call reports failure.
pub fn quantize(
    input: &Path,
    output: &Path,
    params: &LlamaQuantizeParams,
) -> Result<(), QuantizeError> {
    let inp = CString::new(input.to_str().ok_or(QuantizeError::InvalidPath)?)?;
    let out = CString::new(output.to_str().ok_or(QuantizeError::InvalidPath)?)?;

    // SAFETY: both C strings outlive the call; `&params.raw` is a valid,
    // fully-initialized `*const llama_model_quantize_params`.
    let ret = unsafe { sys::llama_model_quantize(inp.as_ptr(), out.as_ptr(), &params.raw) };

    if ret == 0 {
        Ok(())
    } else {
        Err(QuantizeError::Quantize(ret))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // No-model unit test: params construct and builders round-trip. Actually
    // quantizing needs a higher-precision source GGUF we do not have in
    // fixtures (the test model is already IQ1_S), so the write-path itself is
    // only exercised by build/link here.
    #[test]
    fn params_default_and_builders_round_trip() {
        let default = LlamaQuantizeParams::default();
        // Default builds and exposes its raw view.
        let _ = default.as_raw();

        let params = LlamaQuantizeParams::default()
            .with_ftype(ftype::LLAMA_FTYPE_MOSTLY_Q4_K_M)
            .with_n_threads(4)
            .with_allow_requantize(true)
            .with_quantize_output_tensor(true);

        assert_eq!(params.as_raw().nthread, 4);
        assert_eq!(params.as_raw().ftype, ftype::LLAMA_FTYPE_MOSTLY_Q4_K_M);
        assert!(params.as_raw().allow_requantize);
        assert!(params.as_raw().quantize_output_tensor);
    }

    #[test]
    fn ik_sota_ftype_is_exposed() {
        // ik SOTA quant const is reachable through the re-export module.
        let params = LlamaQuantizeParams::default().with_ftype(ftype::LLAMA_FTYPE_MOSTLY_IQ4_K);
        assert_eq!(params.as_raw().ftype, ftype::LLAMA_FTYPE_MOSTLY_IQ4_K);
    }
}
