// Bindgen entry header for ik_llama.cpp.
//
// Unlike stock llama.cpp there is NO `ggml/include/gguf.h` in ik — the gguf
// API lives in `ggml/include/ggml.h`, which is pulled in transitively by
// `include/llama.h` (`#include "ggml.h"`). So a single include is enough to
// surface the `llama_*`, `ggml_*` and `gguf_*` symbols (plan correction [B1]).
#include "ik_llama.cpp/include/llama.h"

// Core `ik_llama_rs_grammar_*` glue (context-free grammar apply/accept). Grammar
// is a core feature, so this is always compiled and always bound.
#include "wrapper_grammar.h"

// The `ik_llama_rs_*` C ABI glue (JSON-schema-to-grammar + MTP speculative) is
// emitted by `wrapper_common.cpp`, which is only compiled — and only has its
// header included here — when the `common` feature is enabled.
#ifdef LLAMA_RS_BUILD_COMMON
#include "wrapper_common.h"
#endif
