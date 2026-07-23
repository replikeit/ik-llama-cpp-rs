// bindgen entry-point for the `mtmd` (multimodal) feature.
//
// mtmd lives under ik_llama.cpp's `examples/mtmd/` (not `include/`), so these
// are resolved via `-I <src>/examples/mtmd`. mtmd.h pulls in ggml.h / llama.h
// (`-I <src>/include`, `-I <src>/ggml/include`) and, under C++, nlohmann/json.hpp
// (`-I <src>/vendor`) — which is why this feature runs its OWN C++-mode bindgen
// pass, separate from the main C bindings. See build.rs::generate_mtmd_bindings.
#include "mtmd.h"
#include "mtmd-helper.h"
