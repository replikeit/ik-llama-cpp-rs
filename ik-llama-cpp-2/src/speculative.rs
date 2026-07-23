//! Multi-Token Prediction (MTP / NextN) support.
//!
//! ik_llama.cpp's MTP is an **embedded** mechanism: a single model carrying
//! NextN prediction layers (loaded with [`crate::LlamaModelParams::with_mtp`]),
//! driven by a low-level op-type state machine plus a `common_speculative` driver.
//!
//! * [`MtpOpType`] + [`crate::LlamaContext::set_mtp_op_type`] expose the raw
//!   op-type primitive (available without `common`).
//! * [`MtpSpeculative`] (requires the **`common`** feature) is the full
//!   draft/accept driver, backed by a C++ glue over ik's `common_speculative_*`
//!   (`ik_llama_rs_mtp_*`). It reproduces the native `--spec-type mtp` flow.

use ik_llama_cpp_sys as sys;

/// MTP decode-graph operation mode (maps to ik's `llama_mtp_op_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpOpType {
    /// Normal decode (no MTP graph).
    None,
    /// Prompt warmup — populate MTP KV + hidden state (no logits).
    Warmup,
    /// Advance MTP state after accepted tokens.
    UpdateAccepted,
    /// Generate draft tokens from the NextN head.
    DraftGen,
}

impl MtpOpType {
    /// Raw `llama_mtp_op_type` value.
    #[must_use]
    pub fn to_raw(self) -> sys::llama_mtp_op_type {
        match self {
            MtpOpType::None => sys::MTP_OP_NONE,
            MtpOpType::Warmup => sys::MTP_OP_WARMUP,
            MtpOpType::UpdateAccepted => sys::MTP_OP_UPDATE_ACCEPTED,
            MtpOpType::DraftGen => sys::MTP_OP_DRAFT_GEN,
        }
    }
}

/// Parameters for MTP speculative decoding. Mirrors the shape of llama-cpp-2's
/// `MtpSpeculativeParams` (plus `temp`/`mtp_heads` for ik's driver).
#[derive(Debug, Clone, Copy)]
pub struct MtpSpeculativeParams {
    /// Maximum draft tokens per step (n_max=1 = one NextN head).
    pub n_max: i32,
    /// Minimum draft tokens per step.
    pub n_min: i32,
    /// Minimum draft acceptance probability (0.0 = never early-stop drafting).
    pub p_min: f32,
    /// MTP heads to use (1 = default; >1 experimental).
    pub mtp_heads: i32,
    /// Target-sampler temperature (<= 0 = greedy).
    pub temp: f32,
}

impl Default for MtpSpeculativeParams {
    fn default() -> Self {
        Self {
            n_max: 1,
            n_min: 0,
            p_min: 0.0,
            mtp_heads: 1,
            temp: 0.0,
        }
    }
}

#[cfg(feature = "common")]
pub use driver::{MtpSpeculative, MtpStep};

#[cfg(feature = "common")]
mod driver {
    use super::MtpSpeculativeParams;
    use ik_llama_cpp_sys as sys;
    use std::ptr::NonNull;

    use crate::context::LlamaContext;
    use crate::model::LlamaModel;
    use crate::token::LlamaToken;
    use crate::LlamaError;

    /// Result of one [`MtpSpeculative::step`].
    #[derive(Debug, Clone)]
    pub struct MtpStep {
        /// Tokens emitted this step — append to the output; each token is emitted
        /// exactly once across steps.
        pub tokens: Vec<LlamaToken>,
        /// Number of draft tokens accepted this step (0..=n_max).
        pub n_accepted: i32,
    }

    /// Full MTP speculative driver (requires the `common` feature).
    ///
    /// Wraps ik's `common_speculative_*` via the `ik_llama_rs_mtp_*` C++ glue,
    /// which OWNS the draft→verify→accept→commit critical section (multiple
    /// internal `llama_decode`s on the target + companion contexts). The borrow of
    /// the [`LlamaContext`] enforces that Rust does not touch it mid-loop; drive it
    /// as `new` → `begin(prompt)` → `step()`* .
    #[derive(Debug)]
    pub struct MtpSpeculative<'ctx, 'model> {
        raw: NonNull<sys::ik_llama_rs_mtp>,
        ctx: &'ctx mut LlamaContext<'model>,
        params: MtpSpeculativeParams,
    }

    impl<'ctx, 'model> MtpSpeculative<'ctx, 'model> {
        /// Initialize the driver over a NextN model + its (mtp-enabled) context.
        ///
        /// `model` must have been loaded with `.with_mtp(true)` and `ctx` created
        /// with `.with_mtp(true)`. Fails (`MtpInit`) for a model with 0 NextN
        /// layers or an openPangu/recurrent target (unsupported in v1).
        pub fn new(
            model: &LlamaModel,
            ctx: &'ctx mut LlamaContext<'model>,
            params: MtpSpeculativeParams,
        ) -> Result<Self, LlamaError> {
            let cparams = ctx.raw_params; // the exact params ctx was built with
                                          // SAFETY: valid model/ctx; cparams matches ctx; glue validates preconditions.
            let raw = unsafe {
                sys::ik_llama_rs_mtp_init(
                    model.model.as_ptr(),
                    ctx.context.as_ptr(),
                    &cparams,
                    params.n_max,
                    params.n_min,
                    params.p_min,
                    params.mtp_heads,
                    params.temp,
                )
            };
            NonNull::new(raw)
                .map(|raw| Self { raw, ctx, params })
                .ok_or(LlamaError::MtpInit)
        }

        /// Prompt warmup + begin. Runs the prompt through the MTP path and prepares
        /// the first draft. Returns `n_past`. Call once before [`Self::step`].
        pub fn begin(&mut self, prompt: &[LlamaToken]) -> Result<usize, LlamaError> {
            let toks: Vec<sys::llama_token> = prompt.iter().map(|t| t.0).collect();
            // SAFETY: valid handle + token buffer.
            let n =
                unsafe { sys::ik_llama_rs_mtp_begin(self.raw.as_ptr(), toks.as_ptr(), toks.len()) };
            if n < 0 {
                return Err(LlamaError::MtpBegin(n as i32));
            }
            Ok(n as usize)
        }

        /// Run one draft→verify→accept→commit cycle. Returns the tokens emitted
        /// this step (to append to the output) and how many drafts were accepted.
        pub fn step(&mut self) -> Result<MtpStep, LlamaError> {
            let cap = self.params.n_max.max(1) as usize + 2;
            let mut out = vec![0 as sys::llama_token; cap];
            let mut n_out: usize = 0;
            let mut n_accepted: i32 = 0;
            // SAFETY: out has `cap` slots; n_out/n_accepted are valid out-params.
            let st = unsafe {
                sys::ik_llama_rs_mtp_step(
                    self.raw.as_ptr(),
                    out.as_mut_ptr(),
                    cap,
                    &mut n_out,
                    &mut n_accepted,
                )
            };
            if st as i32 != 0 {
                // LLAMA_RS_STATUS_OK == 0
                return Err(LlamaError::MtpStep(st as i32));
            }
            out.truncate(n_out);
            Ok(MtpStep {
                tokens: out.into_iter().map(LlamaToken).collect(),
                n_accepted,
            })
        }

        /// The configured parameters.
        #[must_use]
        pub fn params(&self) -> MtpSpeculativeParams {
            self.params
        }

        /// Mutable access to the wrapped context (only between full generations).
        pub fn context_mut(&mut self) -> &mut LlamaContext<'model> {
            self.ctx
        }
    }

    impl Drop for MtpSpeculative<'_, '_> {
        fn drop(&mut self) {
            // SAFETY: frees the common_speculative + common_sampler owned by the glue.
            unsafe { sys::ik_llama_rs_mtp_free(self.raw.as_ptr()) };
        }
    }
}
