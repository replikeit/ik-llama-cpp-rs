#pragma once
// Core C ABI glue: context-free GBNF grammar constraint apply/accept.
//
// ik's public `llama_grammar_apply` / `llama_grammar_accept_token` require a
// `llama_context` (they read `ctx->model.vocab` / `ctx->sampling`). The
// chain-object `LlamaSampler` applies a grammar WITHOUT a context, so this shim
// calls ik's internal `llama_grammar_sample_impl` / `llama_grammar_accept_impl`
// directly with the model's vocab (stashed by the Rust wrapper at construction)
// and a NULL sampling pointer — both impls null-guard the sampling arg. Grammar
// is a core feature, so this is compiled unconditionally (bindgen allowlist
// `ik_llama_rs_grammar_.*`).
//
// `llama_*` types come from llama.h, which wrapper.h includes before this header.

#ifdef __cplusplus
extern "C" {
#endif

// Apply grammar constraints to `candidates` in place (drives tokens the grammar
// cannot currently accept to -inf). `vocab` must be the vocab the grammar was
// built from.
void ik_llama_rs_grammar_apply(
        const struct llama_grammar * grammar,
        const struct llama_vocab   * vocab,
        llama_token_data_array     * candidates);

// Advance the grammar state by accepting `token`.
void ik_llama_rs_grammar_accept(
        struct llama_grammar     * grammar,
        const struct llama_vocab * vocab,
        llama_token                token);

#ifdef __cplusplus
}
#endif
