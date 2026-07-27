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
    /// Maximum draft tokens proposed per step. Defaults to `2`, the throughput
    /// sweet spot measured on a 1-NextN-layer model (IQ4_K, ~1.2x over the plain
    /// engine); higher values draft deeper but acceptance falls off for a
    /// single-head NextN model. `n_max=1` = one NextN head.
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
            n_max: 2,
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
    use crate::llama_batch::LlamaBatch;
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
    /// Wraps ik's `common_speculative_*` via the `ik_llama_rs_mtp_*` C++ glue.
    /// Two ways to drive it after `new` → `begin(prompt)`:
    ///
    /// * **`step()`** — the convenience path: the glue owns the whole
    ///   draft→verify→**sample**→commit cycle (greedy / `temp`), returning the
    ///   committed tokens. Use this when you don't need to constrain sampling.
    /// * **`draft()` + `commit()`** — the caller-driven path: `draft` returns the
    ///   NextN candidates without sampling; you build the verify batch
    ///   `[id_last] + drafts` (logits on every position), decode it on
    ///   [`target_context_mut`](Self::target_context_mut), pick each committed
    ///   token yourself (e.g. through a [`crate::LlamaGrammar`] + sampler over
    ///   [`target_context`](Self::target_context)'s logits), then `commit` them.
    ///   This is what lets MTP speculation run **together with** grammar-
    ///   constrained / custom sampling — the crate never samples on this path.
    ///
    /// Either way ik owns all KV bookkeeping (the companion cache and the target
    /// rollback happen inside the glue); the caller must not clear or roll back
    /// either cache itself.
    #[derive(Debug)]
    pub struct MtpSpeculative<'model> {
        // NOTE: `raw` (the C driver) references `ctx`, so it must be freed first.
        // Fields drop in declaration order, but `Drop::drop` (which frees `raw`)
        // runs before any field is dropped, so `ctx` is still alive then. Keep
        // `raw` declared before `ctx` regardless, for clarity.
        raw: NonNull<sys::ik_llama_rs_mtp>,
        ctx: LlamaContext<'model>,
        params: MtpSpeculativeParams,
    }

    impl<'model> MtpSpeculative<'model> {
        /// Initialize the driver over a NextN model + its (mtp-enabled) context.
        ///
        /// Takes ownership of `ctx` (so the driver is self-contained and can be
        /// stored in a struct); read/decode it back out via
        /// [`target_context`](Self::target_context) /
        /// [`target_context_mut`](Self::target_context_mut).
        ///
        /// `model` must have been loaded with `.with_mtp(true)` and `ctx` created
        /// with `.with_mtp(true)`. Fails with [`LlamaError::MtpInit`] for a model
        /// with 0 NextN layers (not an MTP model). Recurrent / openPangu targets
        /// (which includes the Qwen3.5 NextN family) are supported: the driver
        /// takes an internal rollback checkpoint before each verify so rejected
        /// drafts restore the recurrent state correctly.
        pub fn new(
            model: &LlamaModel,
            ctx: LlamaContext<'model>,
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

        /// Propose up to `n_max` draft tokens following `id_last` (the last
        /// committed token) at position `n_past`, **without sampling**.
        ///
        /// This is the caller-driven entry point for running MTP speculation
        /// alongside a grammar / custom sampler (see the module example). After
        /// `draft`, the caller must:
        /// 1. build the verify batch `[id_last] + drafts` (one sequence, logits on
        ///    **every** position), in that order;
        /// 2. decode it on [`target_context_mut`](Self::target_context_mut);
        /// 3. pick the committed tokens with its own grammar/sampler, reading
        ///    output index `i` via [`target_context`](Self::target_context) for
        ///    each verify position `i`;
        /// 4. call [`commit`](Self::commit).
        ///
        /// Use one round strictly as `draft` → verify-decode → `commit`: `commit`
        /// takes its anchor position from the `n_past` you pass here, so do not
        /// `draft` twice before committing, and build the verify batch with
        /// `id_last` at exactly `n_past`. Do not mix this path with
        /// [`step`](Self::step) on the same driver — they own the loop state
        /// differently.
        ///
        /// # Errors
        ///
        /// [`LlamaError::MtpStep`] if the C glue rejects the draft.
        pub fn draft(
            &mut self,
            n_past: i32,
            id_last: LlamaToken,
        ) -> Result<Vec<LlamaToken>, LlamaError> {
            let cap = self.params.n_max.max(1) as usize;
            let mut out = vec![0 as sys::llama_token; cap];
            let mut n_out: usize = 0;
            // SAFETY: `out` has `cap` slots; `n_out` is a valid out-param.
            let st = unsafe {
                sys::ik_llama_rs_mtp_draft(
                    self.raw.as_ptr(),
                    n_past,
                    id_last.0,
                    out.as_mut_ptr(),
                    cap,
                    &mut n_out,
                )
            };
            if st as i32 != 0 {
                return Err(LlamaError::MtpStep(st as i32));
            }
            out.truncate(n_out);
            Ok(out.into_iter().map(LlamaToken).collect())
        }

        /// Finalize one speculative round after the caller has decoded the verify
        /// batch and chosen its committed tokens.
        ///
        /// `id_last` is the anchor token that led the verify batch (the same one
        /// passed to [`draft`](Self::draft)); `committed` is the accepted-draft
        /// prefix followed by exactly **one** correction/bonus token (so
        /// `committed.len() == n_accepted + 1`); `n_draft` is the number of tokens
        /// the matching [`draft`](Self::draft) returned.
        ///
        /// Advances the MTP companion by the accepted count and rolls the target
        /// KV cache back to the committed prefix. The caller must **not** clear or
        /// roll back either KV cache itself — ik owns that bookkeeping.
        ///
        /// # Errors
        ///
        /// [`LlamaError::MtpStep`] if `committed` is empty, longer than
        /// `n_draft + 1` (more tokens than the verify batch produced — which would
        /// desync the KV cache), or the glue rejects the commit.
        pub fn commit(
            &mut self,
            id_last: LlamaToken,
            committed: &[LlamaToken],
            n_draft: usize,
        ) -> Result<(), LlamaError> {
            // `committed` is the accepted-draft prefix (<= n_draft) plus exactly
            // one correction/bonus token, so it can never exceed `n_draft + 1`.
            // Rejecting a longer slice stops a caller from advancing `n_past` past
            // positions the verify batch never decoded (a silent KV desync).
            if committed.is_empty() || committed.len() > n_draft + 1 {
                return Err(LlamaError::MtpStep(-1));
            }
            let ids: Vec<sys::llama_token> = committed.iter().map(|t| t.0).collect();
            // SAFETY: `ids` is a valid `ids.len()`-long token array for the call.
            let st = unsafe {
                sys::ik_llama_rs_mtp_commit(
                    self.raw.as_ptr(),
                    id_last.0,
                    ids.as_ptr(),
                    ids.len(),
                    n_draft,
                )
            };
            if st as i32 != 0 {
                return Err(LlamaError::MtpStep(st as i32));
            }
            Ok(())
        }

        /// Warm the NextN companion over a decoded prompt chunk — the
        /// caller-driven prefill primitive for cross-call KV-prefix reuse.
        ///
        /// Call immediately after decoding a prompt chunk on
        /// [`target_context_mut`](Self::target_context_mut), passing the **same**
        /// batch, which must carry output/logits on **every** position (the
        /// warmup reads a per-position hidden row). Advances the companion KV at
        /// the batch's positions and marks its heads warmed; performs no decode
        /// or sampling. [`begin`](Self::begin) is exactly this in a loop over
        /// `n_batch`-sized chunks, followed by
        /// [`finalize_prompt`](Self::finalize_prompt).
        ///
        /// For a reuse, restore the target KV (via
        /// [`target_context_mut`](Self::target_context_mut)'s
        /// `state_seq_set_data_ext`) and the companion
        /// ([`companion_state_set`](Self::companion_state_set)) to `reuse_from`
        /// first, then decode+`warm` only the suffix.
        ///
        /// # Errors
        ///
        /// [`LlamaError::MtpStep`] if the glue rejects the warmup.
        pub fn warm(&mut self, batch: &LlamaBatch) -> Result<(), LlamaError> {
            let batch_raw = batch.as_raw();
            // SAFETY: `batch_raw` is a valid llama_batch for the call; its buffers
            // are owned by `batch` and outlive the call. The caller must have just
            // decoded this batch on the target context (see method docs).
            let st = unsafe { sys::ik_llama_rs_mtp_warm(self.raw.as_ptr(), &batch_raw) };
            if st as i32 != 0 {
                return Err(LlamaError::MtpStep(st as i32));
            }
            Ok(())
        }

        /// Finalize a caller-driven prefill: reset the draft cache/history,
        /// capture the seed hidden at position `n_past - 1` (required before
        /// [`draft`](Self::draft)), set the driver's `n_past`, and clear any
        /// [`step`](Self::step) carry.
        ///
        /// `last_output_index` is the index of the last prompt token within the
        /// last decoded chunk (e.g. `batch.n_tokens() - 1` of the final chunk);
        /// `n_past` is the total prompt length. Call once, after the last
        /// [`warm`](Self::warm).
        ///
        /// # Errors
        ///
        /// [`LlamaError::MtpStep`] if `n_past == 0`, `last_output_index < 0`, or
        /// the glue's hidden capture fails.
        pub fn finalize_prompt(
            &mut self,
            last_output_index: i32,
            n_past: usize,
        ) -> Result<(), LlamaError> {
            // SAFETY: valid handle; the glue validates the indices.
            let st = unsafe {
                sys::ik_llama_rs_mtp_finalize_prompt(self.raw.as_ptr(), last_output_index, n_past)
            };
            if st as i32 != 0 {
                return Err(LlamaError::MtpStep(st as i32));
            }
            Ok(())
        }

        /// Bytes needed to snapshot the NextN companion's state (KV + recurrent
        /// state) for the bound sequence. `0` means nothing is cached yet, or
        /// state IO is unsupported for this architecture (e.g. openPangu) — in
        /// which case cross-call reuse is impossible and the caller must fall back
        /// to a full [`begin`](Self::begin).
        ///
        /// The companion caches only the NextN tail layers, so this is a small
        /// fraction of the target context's own
        /// [`state_seq_get_size_ext`](LlamaContext::state_seq_get_size_ext).
        #[must_use]
        pub fn companion_state_size(&self) -> usize {
            // SAFETY: valid handle; returns 0 for a null companion / unsupported arch.
            unsafe { sys::ik_llama_rs_mtp_companion_state_size(self.raw.as_ptr()) }
        }

        /// Serialize the companion's state into `dst`, which must point to at
        /// least [`companion_state_size`](Self::companion_state_size) bytes.
        /// Returns the number of bytes written. Pair with the target context's
        /// [`state_seq_get_data_ext`](LlamaContext::state_seq_get_data_ext) to
        /// snapshot a full reuse boundary (capture both *at the same position* —
        /// the companion's recurrent state cannot be sliced from a later state).
        ///
        /// # Safety
        ///
        /// `dst` must be valid for writes of `companion_state_size()` bytes.
        #[must_use]
        pub unsafe fn companion_state_get(&self, dst: *mut u8) -> usize {
            // SAFETY: caller guarantees `dst` is sized per companion_state_size().
            unsafe { sys::ik_llama_rs_mtp_companion_state_get(self.raw.as_ptr(), dst) }
        }

        /// Restore the companion's state from `src` (bytes previously produced by
        /// [`companion_state_get`](Self::companion_state_get) on the same model /
        /// context params). Returns `false` on a length/format mismatch or an
        /// unsupported arch — the caller then falls back to a full
        /// [`begin`](Self::begin). Pair with the target context's
        /// [`state_seq_set_data_ext`](LlamaContext::state_seq_set_data_ext).
        ///
        /// # Safety
        ///
        /// `src` must contain valid companion state produced by
        /// [`companion_state_get`](Self::companion_state_get).
        pub unsafe fn companion_state_set(&mut self, src: &[u8]) -> bool {
            // SAFETY: caller guarantees `src` is valid companion state; the glue
            // validates the length and rejects rather than reading out of bounds.
            unsafe {
                sys::ik_llama_rs_mtp_companion_state_set(self.raw.as_ptr(), src.as_ptr(), src.len())
            }
        }

        /// The configured parameters.
        #[must_use]
        pub fn params(&self) -> MtpSpeculativeParams {
            self.params
        }

        /// Shared access to the target context — read per-position logits here for
        /// a grammar-gated commit (e.g. [`LlamaContext::token_data_array_ith`]).
        #[must_use]
        pub fn target_context(&self) -> &LlamaContext<'model> {
            &self.ctx
        }

        /// Mutable access to the target context — decode the verify batch here.
        pub fn target_context_mut(&mut self) -> &mut LlamaContext<'model> {
            &mut self.ctx
        }

        /// Mutable access to the wrapped context (only between full generations).
        ///
        /// Alias of [`target_context_mut`](Self::target_context_mut), kept for
        /// back-compat.
        pub fn context_mut(&mut self) -> &mut LlamaContext<'model> {
            &mut self.ctx
        }
    }

    impl Drop for MtpSpeculative<'_> {
        fn drop(&mut self) {
            // SAFETY: frees the common_speculative + common_sampler owned by the
            // glue. Runs before `self.ctx` is dropped, so the glue's reference to
            // the target context is still valid here.
            unsafe { sys::ik_llama_rs_mtp_free(self.raw.as_ptr()) };
        }
    }
}
