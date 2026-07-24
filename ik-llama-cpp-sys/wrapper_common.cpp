// C ABI glue implementation for ik_llama.cpp MTP speculative decoding.
//
// Wraps ik's C++ `common_speculative_*` + `common_sampler` (libcommon.a). The
// call sequence mirrors examples/main/main.cpp's `--spec-type mtp` path.
// Single-sequence, embedded NextN MTP. Recurrent / openPangu targets (incl.
// Qwen3.5 NextN) are supported via an internal rollback checkpoint taken before
// each verify (see `needs_ckpt`); only a model with 0 NextN layers is rejected.

#include "speculative.h" // common_speculative_*, common_params_speculative, stage params
#include "sampling.h"     // common_sampler_*, common_params_sampling
#include "common.h"       // common_batch_add
#include "json-schema-to-grammar.h" // json_schema_to_grammar
#include "llama.h"        // MTP_OP_*, llama_*

#include "wrapper_common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
// RAII guard: frees a llama_batch on scope exit (incl. exception unwinding),
// so no batch leaks on the throw paths (fixes the manual-free leak windows).
struct batch_guard {
    llama_batch b;
    explicit batch_guard(llama_batch b_) : b(b_) {}
    ~batch_guard() { llama_batch_free(b); }
    batch_guard(const batch_guard &) = delete;
    batch_guard & operator=(const batch_guard &) = delete;
};
} // namespace

// --- JSON Schema -> GBNF grammar (wraps libcommon's json_schema_to_grammar) ---

extern "C" llama_rs_status ik_llama_rs_json_schema_to_grammar(
        const char * schema_json,
        bool         force_gbnf,
        char      ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_grammar = nullptr;
    try {
        const auto schema  = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void ik_llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

struct ik_llama_rs_mtp {
    llama_model              * model   = nullptr;
    llama_context            * ctx_tgt = nullptr;
    common_speculative       * spec    = nullptr;
    common_sampler           * smpl    = nullptr;
    common_params_speculative  params;
    common_params_sampling     sparams;      // used for the checkpoint before_draft call
    bool                       needs_ckpt = false; // recurrent/openPangu target
    llama_seq_id               seq_id  = 0;
    llama_pos                  n_past  = 0;
    bool                       have_carry = false;
    llama_token                carry   = 0;
};

extern "C" {

ik_llama_rs_mtp * ik_llama_rs_mtp_init(
        llama_model * model,
        llama_context * ctx_tgt,
        const llama_context_params * cparams_tgt,
        int32_t n_max,
        int32_t n_min,
        float   p_min,
        int32_t mtp_heads,
        float   temp) {
    if (!model || !ctx_tgt || !cparams_tgt) {
        fprintf(stderr, "ik_llama_rs_mtp_init: null arg (model=%p ctx=%p cparams=%p)\n",
                (void *) model, (void *) ctx_tgt, (const void *) cparams_tgt);
        return nullptr;
    }
    // Use the model actually backing ctx_tgt (native uses llama_get_model(ctx)).
    llama_model * mdl = const_cast<llama_model *>(llama_get_model(ctx_tgt));
    if (!mdl) {
        mdl = model;
    }
    if (llama_model_n_nextn_layer(mdl) <= 0) {
        fprintf(stderr, "ik_llama_rs_mtp_init: 0 NextN layers\n");
        return nullptr;
    }
    try {
        ik_llama_rs_mtp * h = new ik_llama_rs_mtp();
        h->model   = mdl;
        h->ctx_tgt = ctx_tgt;

        // one MTP stage
        common_speculative_stage_params stage;
        stage.type  = COMMON_SPECULATIVE_TYPE_MTP;
        stage.n_max = n_max;
        if (n_min >= 0) {
            stage.n_min = n_min;
        }
        stage.p_min = p_min; // 0.0 is a valid override (>= 0)
        if (mtp_heads > 0) {
            stage.mtp_heads = mtp_heads;
        }
        h->params.stages.push_back(stage);
        h->params.n_max     = n_max;
        h->params.n_min     = (n_min >= 0) ? n_min : 0;
        h->params.p_min     = p_min;
        h->params.mtp_heads = (mtp_heads > 0) ? mtp_heads : 1;

        // companion MTP context params = target's params + MTP overrides
        // (mirrors common_speculative_prepare_mtp_runtime).
        h->params.cparams_dft              = *cparams_tgt;
        h->params.cparams_dft.mtp          = true;
        h->params.cparams_dft.mtp_op_type  = MTP_OP_WARMUP;
        h->params.cparams_dft.embeddings   = true;
        h->params.cparams_dft.pooling_type = LLAMA_POOLING_TYPE_NONE;

        h->spec = common_speculative_init(h->params, ctx_tgt);
        if (!h->spec) {
            fprintf(stderr, "ik_llama_rs_mtp_init: common_speculative_init returned null\n");
            delete h;
            return nullptr;
        }

        common_params_sampling sparams;
        sparams.temp = temp; // <= 0 => greedy
        h->smpl = common_sampler_init(mdl, sparams);
        if (!h->smpl) {
            fprintf(stderr, "ik_llama_rs_mtp_init: common_sampler_init returned null\n");
            common_speculative_free(h->spec);
            delete h;
            return nullptr;
        }
        h->sparams = sparams;
        // Recurrent / openPangu targets (incl. Qwen3.5 NextN) drive a checkpoint
        // save before each draft; commit restores on rejection.
        h->needs_ckpt = llama_model_has_recurrent(mdl) || llama_model_is_openpangu(mdl);
        return h;
    } catch (const std::exception & e) {
        fprintf(stderr, "ik_llama_rs_mtp_init: exception: %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "ik_llama_rs_mtp_init: unknown exception\n");
        return nullptr;
    }
}

long ik_llama_rs_mtp_begin(ik_llama_rs_mtp * h, const llama_token * prompt, size_t n) {
    if (!h || !prompt || n == 0) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        // Decode the prompt in n_batch-sized chunks (all-logits, for the warmup
        // hidden capture). `llama_decode` asserts n_tokens <= n_batch (uncatchable
        // abort), so single-shot on a long prompt would crash — chunk like native.
        const int32_t n_batch = (int32_t) llama_n_batch(h->ctx_tgt);
        int32_t last_chunk_len = 0;
        size_t offset = 0;
        while (offset < n) {
            const int32_t chunk = (int32_t) std::min<size_t>((size_t) n_batch, n - offset);
            batch_guard bg(llama_batch_init(chunk, 0, 1));
            for (int32_t j = 0; j < chunk; ++j) {
                common_batch_add(bg.b, prompt[offset + (size_t) j],
                                 (llama_pos) (offset + (size_t) j), { h->seq_id }, true);
            }
            if (llama_decode(h->ctx_tgt, bg.b) != 0) {
                return LLAMA_RS_STATUS_EXCEPTION;
            }
            if (common_speculative_on_target_seq_batch(
                    h->spec, h->ctx_tgt, bg.b, h->seq_id, /*is_prompt_warmup=*/true) != 0) {
                return LLAMA_RS_STATUS_EXCEPTION;
            }
            last_chunk_len = chunk;
            offset += (size_t) chunk;
        }
        const int32_t   final_index = last_chunk_len - 1; // output index within the last chunk
        const llama_pos final_pos   = (llama_pos) n - 1;
        h->n_past = (llama_pos) n;

        std::vector<llama_token> empty;
        common_speculative_begin(h->spec, empty);
        if (!common_speculative_capture_output_hidden(h->spec, h->ctx_tgt, final_index, h->seq_id, final_pos)) {
            fprintf(stderr, "ik_llama_rs_mtp_begin: capture_output_hidden failed\n");
            return LLAMA_RS_STATUS_EXCEPTION;
        }
        h->have_carry = false;
        return (long) h->n_past;
    } catch (const std::exception & e) {
        fprintf(stderr, "ik_llama_rs_mtp_begin: exception: %s\n", e.what());
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

llama_rs_status ik_llama_rs_mtp_step(
        ik_llama_rs_mtp * h,
        llama_token     * out,
        size_t            cap,
        size_t          * n_out,
        int32_t         * n_accepted) {
    if (!h || !out || !n_out || !n_accepted) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *n_out = 0;
    *n_accepted = 0;
    try {
        // anchor token: reuse the carried bonus, else sample+accept
        const bool  from_carry = h->have_carry;
        llama_token sampled_before;
        if (from_carry) {
            sampled_before = h->carry;
            h->have_carry = false;
        } else {
            sampled_before = common_sampler_sample_legacy(h->smpl, h->ctx_tgt, nullptr, -1);
            common_sampler_accept(h->smpl, h->ctx_tgt, sampled_before, /*is_generated=*/true);
        }

        // draft (empty history for pure self-spec MTP)
        std::vector<llama_token> draft_history;
        common_speculative_draft_result dr = common_speculative_draft_ex(
            h->spec, h->ctx_tgt, h->params, draft_history, sampled_before, h->n_past, h->seq_id);
        std::vector<llama_token> & draft = dr.tokens;

        // recurrent / openPangu targets: save a rollback checkpoint before verify.
        // If the save fails, DROP the draft (fall back to single-token): commit's
        // restore is gated on a valid checkpoint, so drafting past a failed save
        // would leave the recurrent/hidden state wrongly advanced on a reject
        // (matches main.cpp:988-1001).
        if (h->needs_ckpt) {
            const bool ok = common_speculative_before_draft(
                h->spec, h->model, h->ctx_tgt, h->smpl, h->sparams,
                h->seq_id, h->n_past, sampled_before, (int) draft.size() + 1,
                h->params.recurrent_ckpt_mode);
            if (!ok) {
                draft.clear();
            }
        }

        // clamp draft to context/batch limits (main.cpp:975-983) so the verify
        // batch never places a token at/over n_ctx (assert-abort) near the end.
        {
            const int32_t n_ctx   = (int32_t) llama_n_ctx(h->ctx_tgt);
            const int32_t n_batch = (int32_t) llama_n_batch(h->ctx_tgt);
            int32_t max_draft = std::min(n_ctx - (int32_t) h->n_past - 2, n_batch - 1);
            if (max_draft < 0) {
                max_draft = 0;
            }
            if ((int32_t) draft.size() > max_draft) {
                draft.resize((size_t) max_draft);
            }
        }

        // target verify batch: sampled_before @ n_past, drafts @ n_past+1..
        // (empty draft => single-token step; sample_and_accept_n still returns the bonus)
        batch_guard vbg(llama_batch_init((int32_t) draft.size() + 1, 0, 1));
        std::vector<int> verify_indices;
        common_batch_add(vbg.b, sampled_before, h->n_past, { h->seq_id }, true);
        verify_indices.push_back(0);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(vbg.b, draft[i], h->n_past + 1 + (llama_pos) i, { h->seq_id }, true);
            verify_indices.push_back((int) i + 1);
        }
        if (llama_decode(h->ctx_tgt, vbg.b) != 0) {
            return LLAMA_RS_STATUS_EXCEPTION; // vbg frees the batch
        }

        std::vector<llama_token> ids =
            common_sampler_sample_and_accept_n(h->smpl, h->ctx_tgt, verify_indices, draft);

        std::vector<int32_t> accepted_output_indices;
        if (!ids.empty()) {
            accepted_output_indices.assign(verify_indices.begin(), verify_indices.begin() + ids.size());
        }
        common_speculative_commit(
            h->spec, h->ctx_tgt, h->smpl, h->seq_id, sampled_before, ids,
            (int) draft.size(), h->n_past + 1, accepted_output_indices);
        // vbg frees the verify batch at scope end (incl. exception paths)

        // emit: [sampled_before if not carried] + ids ; carry the bonus (ids.back())
        if (!from_carry) {
            if (*n_out >= cap) { return LLAMA_RS_STATUS_ALLOCATION_FAILED; }
            out[(*n_out)++] = sampled_before;
        }
        if (!ids.empty()) {
            h->carry = ids.back();
            h->have_carry = true;
            for (size_t i = 0; i < ids.size(); ++i) {
                if (*n_out >= cap) { return LLAMA_RS_STATUS_ALLOCATION_FAILED; }
                out[(*n_out)++] = ids[i];
            }
            *n_accepted = (int32_t) (ids.size() - 1); // ids includes the bonus token
            h->n_past  += (llama_pos) ids.size();
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

// --- Caller-driven MTP primitives (for grammar-gated / custom sampling) ---
//
// These split step()'s draft->verify->sample->commit into pieces so the CALLER
// owns token selection: no sampling happens in the glue. One round is:
//   1. draft(n_past, id_last)               -> up to n_max candidate tokens
//   2. [caller builds the verify batch `[id_last] + drafts` with logits on ALL
//      positions, decodes it on the target context, and picks the committed
//      tokens with its own grammar/sampler by reading each output index]
//   3. commit(id_last, committed, n_committed, n_draft)
// `committed` is the accepted-draft prefix followed by exactly ONE
// correction/bonus token (so n_committed == n_accepted + 1). commit advances the
// MTP companion by the accepted count and rolls the target KV back to the
// committed prefix — the caller must NOT touch either KV cache itself.

llama_rs_status ik_llama_rs_mtp_draft(
        ik_llama_rs_mtp * h,
        int32_t           n_past,
        llama_token       id_last,
        llama_token     * out,
        size_t            cap,
        size_t          * n_out) {
    if (!h || !out || !n_out || n_past < 0) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *n_out = 0;
    try {
        h->n_past = (llama_pos) n_past;

        // draft (empty history for pure self-spec MTP; base pos = n_past)
        std::vector<llama_token> draft_history;
        common_speculative_draft_result dr = common_speculative_draft_ex(
            h->spec, h->ctx_tgt, h->params, draft_history, id_last, h->n_past, h->seq_id);
        std::vector<llama_token> & draft = dr.tokens;

        // recurrent / openPangu targets: checkpoint before the caller verifies so
        // commit() can roll the recurrent/hidden state back on a rejection. If the
        // save fails, drop the draft (single-token fallback), matching step().
        if (h->needs_ckpt) {
            const bool ok = common_speculative_before_draft(
                h->spec, h->model, h->ctx_tgt, h->smpl, h->sparams,
                h->seq_id, h->n_past, id_last, (int) draft.size() + 1,
                h->params.recurrent_ckpt_mode);
            if (!ok) {
                draft.clear();
            }
        }

        // clamp to context/batch limits so the caller's verify batch never places
        // a token at/over n_ctx (assert-abort) near the end (matches step()).
        {
            const int32_t n_ctx   = (int32_t) llama_n_ctx(h->ctx_tgt);
            const int32_t n_batch = (int32_t) llama_n_batch(h->ctx_tgt);
            int32_t max_draft = std::min(n_ctx - (int32_t) h->n_past - 2, n_batch - 1);
            if (max_draft < 0) {
                max_draft = 0;
            }
            if ((int32_t) draft.size() > max_draft) {
                draft.resize((size_t) max_draft);
            }
        }

        *n_out = draft.size();
        if (draft.size() > cap) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        if (!draft.empty()) {
            std::memcpy(out, draft.data(), draft.size() * sizeof(llama_token));
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

llama_rs_status ik_llama_rs_mtp_commit(
        ik_llama_rs_mtp   * h,
        llama_token         id_last,
        const llama_token * committed,
        size_t              n_committed,
        size_t              n_draft) {
    if (!h || !committed || n_committed == 0) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::vector<llama_token> ids(committed, committed + n_committed);
        // The caller built the verify batch as [id_last] + drafts (logits on all),
        // so output index i holds the prediction after verify position i; the
        // committed prefix occupies output indices 0..n_committed.
        std::vector<int32_t> accepted_output_indices;
        accepted_output_indices.reserve(n_committed);
        for (size_t i = 0; i < n_committed; ++i) {
            accepted_output_indices.push_back((int32_t) i);
        }
        // Advances the MTP companion by the accepted count and rolls the target KV
        // back to `n_past + n_committed` (drops rejected drafts). Consumes the
        // already-sampled `ids`; performs no sampling itself.
        common_speculative_commit(
            h->spec, h->ctx_tgt, h->smpl, h->seq_id, id_last, ids,
            (int) n_draft, h->n_past + 1, accepted_output_indices);
        h->n_past += (llama_pos) n_committed;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

void ik_llama_rs_mtp_free(ik_llama_rs_mtp * h) {
    if (!h) {
        return;
    }
    if (h->smpl) {
        common_sampler_free(h->smpl);
    }
    if (h->spec) {
        common_speculative_free(h->spec);
    }
    delete h;
}

} // extern "C"
