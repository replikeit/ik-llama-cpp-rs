#pragma once

// C ABI glue for ik_llama.cpp's MTP (Multi-Token-Prediction / NextN) speculative
// decoding, compiled only under the `common` feature. Wraps ik's C++
// `common_speculative_*` + `common_sampler` (in libcommon.a) so Rust binds a
// small, stable `extern "C"` surface (bindgen allowlist `ik_llama_rs_.*`).
//
// This header is pure C (parsed by bindgen); the heavy C++ common headers are
// included only by wrapper_common.cpp. `llama_*` types come from llama.h, which
// wrapper.h includes before this header.

#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// Convert a JSON Schema (as a JSON string) into a GBNF grammar string. Wraps
// libcommon's C++ `json_schema_to_grammar`. On LLAMA_RS_STATUS_OK, writes a
// heap-allocated NUL-terminated grammar string to `*out_grammar` (the caller
// frees it with ik_llama_rs_string_free). On any error `*out_grammar` is left
// NULL. Returns LLAMA_RS_STATUS_INVALID_ARGUMENT for null args,
// LLAMA_RS_STATUS_EXCEPTION on parse/conversion failure, or
// LLAMA_RS_STATUS_ALLOCATION_FAILED if the result string could not be duplicated.
llama_rs_status ik_llama_rs_json_schema_to_grammar(
        const char  * schema_json,
        bool          force_gbnf,
        char       ** out_grammar);

// Free a string returned via `out_grammar` by the function above.
void ik_llama_rs_string_free(char * ptr);

// Opaque MTP speculative driver handle.
typedef struct ik_llama_rs_mtp ik_llama_rs_mtp;

// Initialize the MTP speculative driver over an existing target context.
//
// Preconditions (the caller MUST satisfy these — see the Rust MtpSpeculative):
//   * `model`   was loaded with model_params.mtp = true (NextN tensors present), and
//   * `ctx_tgt` was created with context_params.mtp = true.
// `cparams_tgt` must be the same `llama_context_params` used to build `ctx_tgt`;
// the companion MTP context is derived from it (+ MTP warmup/embeddings overrides).
// `temp <= 0` selects a greedy target sampler.
//
// Returns NULL on error: null args, or a model with 0 NextN layers (not MTP).
// Recurrent / openPangu targets (incl. Qwen3.5 NextN) ARE supported via an
// internal rollback checkpoint taken before each verify.
ik_llama_rs_mtp * ik_llama_rs_mtp_init(
        struct llama_model                * model,
        struct llama_context              * ctx_tgt,
        const struct llama_context_params * cparams_tgt,
        int32_t                             n_max,
        int32_t                             n_min,
        float                               p_min,
        int32_t                             mtp_heads,
        float                               temp);

// Prompt warmup + begin + capture final hidden state. The driver OWNS the prompt
// decode (it must feed an all-logits batch). Call once before stepping.
// Returns n_past (>= 0) on success, or a negative `llama_rs_status` on error.
long ik_llama_rs_mtp_begin(
        ik_llama_rs_mtp   * spec,
        const llama_token * prompt,
        size_t              n_prompt);

// Run one draft -> verify -> accept -> commit cycle. Writes the tokens emitted
// this step into `out_tokens[0..*n_out)` (append these to the visible output;
// each token is emitted exactly once across steps). `*n_accepted` = number of
// draft tokens accepted this step. The caller decides when to stop (EOG / budget).
// Returns LLAMA_RS_STATUS_OK on success.
llama_rs_status ik_llama_rs_mtp_step(
        ik_llama_rs_mtp * spec,
        llama_token     * out_tokens,
        size_t            cap,
        size_t          * n_out,
        int32_t         * n_accepted);

// Caller-driven MTP (grammar-gated / custom sampling): NO sampling in the glue.
//
// `draft` proposes up to n_max candidate tokens following `id_last` at position
// `n_past`, writing them to out_tokens[0..*n_out) (ALLOCATION_FAILED if the count
// exceeds cap). The caller then builds the verify batch `[id_last] + drafts`
// (logits on ALL positions), decodes it on the target context, picks the
// committed tokens with its own grammar/sampler by reading each output index,
// and calls `commit`.
llama_rs_status ik_llama_rs_mtp_draft(
        ik_llama_rs_mtp * spec,
        int32_t           n_past,
        llama_token       id_last,
        llama_token     * out_tokens,
        size_t            cap,
        size_t          * n_out);

// `commit` finalizes one round: `committed` is the accepted-draft prefix plus
// exactly one correction/bonus token (n_committed == n_accepted + 1), `n_draft`
// is the count returned by the matching `draft`. It advances the MTP companion by
// the accepted count and rolls the target KV back to the committed prefix. The
// caller must NOT manipulate the target or companion KV itself. No sampling.
llama_rs_status ik_llama_rs_mtp_commit(
        ik_llama_rs_mtp   * spec,
        llama_token         id_last,
        const llama_token * committed,
        size_t              n_committed,
        size_t              n_draft);

// --- Caller-driven prefill (for cross-call KV-prefix reuse) ---------------
//
// `begin` above is the atomic no-reuse prefill. These split its body so the
// caller can drive prefill in chunks and snapshot at a boundary (needed for
// reuse — the recurrent companion state cannot be sliced from a later snapshot).
// One prefill is: for each prompt chunk, the caller decodes it on the target
// context (all-logits) then calls `warm`; after the last chunk it calls
// `finalize_prompt`. For a reuse the caller first restores the target KV and the
// companion state (below) to `reuse_from`, then warms only the suffix.

// Warm the NextN companion over `batch` (which the caller has just decoded on the
// target context, with output on EVERY position). Advances the companion KV at
// the batch's positions and marks its heads warmed. Does NOT sample or decode.
llama_rs_status ik_llama_rs_mtp_warm(
        ik_llama_rs_mtp          * spec,
        const struct llama_batch * batch);

// Finalize a caller-driven prefill: reset the draft cache/history, capture the
// seed hidden at position `n_past - 1` from target output index `last_output_index`
// (required before `draft`), set the driver's `n_past`, and clear any `step()`
// carry. `last_output_index` is the index of the last prompt token within the
// last decoded chunk.
llama_rs_status ik_llama_rs_mtp_finalize_prompt(
        ik_llama_rs_mtp * spec,
        int32_t           last_output_index,
        size_t            n_past);

// Companion (NextN) context state snapshot/restore, forwarding llama_state_seq_*
// on common_speculative's internal companion context with flags=0 (full: KV +
// recurrent state). Enable cross-call reuse by pairing with the target context's
// own state_seq snapshot. Return 0 / false if the companion ctx is null or state
// IO is unsupported for the arch (e.g. openPangu) — the caller then full-prefills.
size_t ik_llama_rs_mtp_companion_state_size(ik_llama_rs_mtp * spec);
size_t ik_llama_rs_mtp_companion_state_get(ik_llama_rs_mtp * spec, uint8_t * dst);
bool   ik_llama_rs_mtp_companion_state_set(
        ik_llama_rs_mtp * spec,
        const uint8_t   * src,
        size_t            len);

void ik_llama_rs_mtp_free(ik_llama_rs_mtp * spec);

#ifdef __cplusplus
}
#endif
