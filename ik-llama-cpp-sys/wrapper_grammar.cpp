// Core grammar-constraint glue (see wrapper_grammar.h). Wraps ik's internal
// llama_grammar_sample_impl / llama_grammar_accept_impl with a NULL sampling
// pointer so a grammar can be applied/advanced without a llama_context.
#include "llama-grammar.h" // llama_grammar_sample_impl/accept_impl + struct llama_grammar
#include "llama.h"          // llama_vocab, llama_token, llama_token_data_array

#include "wrapper_grammar.h"

extern "C" void ik_llama_rs_grammar_apply(
        const struct llama_grammar * grammar,
        const struct llama_vocab   * vocab,
        llama_token_data_array     * candidates) {
    llama_grammar_sample_impl(grammar, vocab, nullptr, candidates);
}

extern "C" void ik_llama_rs_grammar_accept(
        struct llama_grammar     * grammar,
        const struct llama_vocab * vocab,
        llama_token                token) {
    llama_grammar_accept_impl(*grammar, vocab, nullptr, token);
}
