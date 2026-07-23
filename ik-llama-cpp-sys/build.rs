//! Build script for `ik-llama-cpp-sys`.
//!
//! Two build modes:
//!   * **Prebuilt fast-path** — if `IK_LLAMA_CPP_LIB_DIR` is set, skip CMake and
//!     link the prebuilt `libllama`/`libggml` (+ static `libcommon.a` under
//!     `common`) found there. Bindgen still runs off the source headers.
//!   * **CMake build** — otherwise build ik_llama.cpp from source (submodule or
//!     `IK_LLAMA_CPP_SRC`) with `-DGGML_MAX_CONTEXTS=2048`.
//!
//! ik has no `install()` rules for the static archives and no
//! `ggml-config.cmake` / `LLAMA_USE_SYSTEM_GGML` — so this crate's build.rs is
//! the single source of link truth (prebuilt layout is our convention).
//! Linux (CPU + CUDA) focused for v1.

use std::env;
use std::path::{Path, PathBuf};

fn feat(name: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{name}")).is_ok()
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let src = env::var("IK_LLAMA_CPP_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("ik_llama.cpp"));
    assert!(
        src.join("include/llama.h").exists(),
        "ik_llama.cpp source not found at {src:?} — set IK_LLAMA_CPP_SRC or run `git submodule update --init`"
    );

    let want_common = feat("COMMON");
    let want_mtmd = feat("MTMD");
    let want_cuda = feat("CUDA");
    let want_vulkan = feat("VULKAN");
    let want_openmp = feat("OPENMP");
    let want_native = feat("NATIVE");
    let dynamic_link = feat("DYNAMIC_LINK");
    let static_stdcxx = feat("STATIC_STDCXX");

    for f in [
        "wrapper.h",
        "wrapper_common.h",
        "wrapper_common.cpp",
        "wrapper_grammar.h",
        "wrapper_grammar.cpp",
        "wrapper_utils.h",
        "wrapper_mtmd.h",
        "build.rs",
    ] {
        println!("cargo:rerun-if-changed={f}");
    }
    println!("cargo:rerun-if-env-changed=IK_LLAMA_CPP_SRC");
    println!("cargo:rerun-if-env-changed=IK_LLAMA_CPP_LIB_DIR");

    // ---- bindgen (both modes) ----
    generate_bindings(&src, &out_dir, want_common);

    // mtmd gets its OWN C++-mode bindgen pass (writes mtmd_bindings.rs). It is
    // separate from the main C pass because mtmd.h pulls in nlohmann/json.hpp and
    // declares two functions taking a C++ `json&` (blocklisted below) that a C
    // bindgen cannot parse. See generate_mtmd_bindings.
    if want_mtmd {
        generate_mtmd_bindings(&src, &out_dir);
    }

    // ---- docs.rs: stop after bindgen ----
    // docs.rs builds have no network and tight time/memory limits, and rustdoc
    // only needs the generated bindings to typecheck — not the compiled native
    // library (there is no final link for a rlib's docs). The full from-source
    // CMake build exceeds docs.rs's limits, so on docs.rs we skip the C++ glue,
    // the CMake build, and all linking.
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // ---- compile the core grammar glue (cc) ALWAYS ----
    // Emitted before the native libs (GNU ld left-to-right): the glue references
    // `llama_grammar_sample_impl`/`accept_impl` in libllama.
    compile_grammar_glue(&src, &manifest_dir);

    // ---- compile the common glue (cc) under `common` FIRST ----
    // Emitting the glue's `static=ik_llama_rs_common` before `static=common`
    // matters: GNU ld resolves archives left-to-right, and the glue references
    // `common_*` symbols, so the glue archive must precede libcommon.a (m2).
    if want_common {
        compile_common_glue(&src, &manifest_dir);
    }

    // ---- compile libmtmd (cc) BEFORE the native-lib link block ----
    // Same left-to-right link-order rationale as the common glue: `static=mtmd`
    // references `llama_*`/`ggml_*` symbols, so it must precede `llama`/`ggml`
    // on the link line.
    if want_mtmd {
        compile_mtmd(&src);
    }

    // ---- native libs: prebuilt fast-path OR CMake build ----
    let backend = if let Some(lib_dir) = env::var("IK_LLAMA_CPP_LIB_DIR").ok().map(PathBuf::from) {
        link_prebuilt(&lib_dir, want_common, want_cuda);
        format!("prebuilt:{}", lib_dir.display())
    } else {
        let dst = cmake_build(
            &src,
            want_common,
            want_cuda,
            want_vulkan,
            want_openmp,
            want_native,
            dynamic_link,
        );
        link_built(&dst, want_common, dynamic_link);
        "cmake".to_string()
    };

    // ---- CUDA runtime libs ----
    if want_cuda {
        link_cuda();
    }

    // ---- C++ stdlib + system libs ----
    link_system(static_stdcxx, want_openmp);

    write_manifest(
        &src,
        &out_dir,
        want_cuda,
        want_vulkan,
        want_openmp,
        want_common,
        &backend,
    );
}

fn generate_bindings(src: &Path, out_dir: &Path, want_common: bool) {
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", src.join("include").display()))
        .clang_arg(format!("-I{}", src.join("ggml/include").display()))
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*")
        .allowlist_var("gguf_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("llama_.*")
        // Core grammar glue (wrapper_grammar.h) — always compiled + always bound.
        .allowlist_function("ik_llama_rs_grammar_.*")
        .prepend_enum_name(false)
        .derive_partialeq(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    if want_common {
        builder = builder
            .clang_arg("-DLLAMA_RS_BUILD_COMMON")
            .clang_arg(format!("-I{}", src.join("common").display()))
            .allowlist_function("ik_llama_rs_.*")
            .allowlist_type("ik_llama_rs_.*");
    }

    builder
        .generate()
        .expect("bindgen failed to generate ik_llama.cpp bindings")
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}

/// Generate the `mtmd_*` bindings in a dedicated **C++-mode** bindgen pass
/// (`mtmd_bindings.rs` in OUT_DIR).
///
/// Why a separate pass (not appended to the main C bindings):
///   * `mtmd.h` (under `examples/mtmd/`) `#include`s `nlohmann/json.hpp` and
///     declares `mtmd_input_chunk_from_json`/`_to_json`, which take a C++
///     `json&`. Those cannot be expressed in a C bindgen, so we parse in C++
///     mode (`-x c++ -std=c++17`) and **blocklist** the two json functions.
///
/// Type sharing with the main C bindings:
///   * We `allowlist` only `mtmd_.*` and **blocklist** all `llama_.*`/`ggml_.*`/
///     `gguf_.*` types and functions. bindgen then emits *only* the `mtmd_*`
///     items, referencing the shared llama/ggml types (`llama_model`,
///     `llama_context`, `llama_token`, `llama_pos`, `ggml_log_level`,
///     `llama_flash_attn_type`, `ggml_type`, ...) by bare name. The generated
///     file is `include!`d inside a `mod mtmd_bindings { use super::*; ... }`
///     (see src/lib.rs) so those names resolve to the main bindings' defs — no
///     duplicate/clashing `llama_*`/`ggml_*` type definitions.
fn generate_mtmd_bindings(src: &Path, out_dir: &Path) {
    let mtmd_dir = src.join("examples/mtmd");
    assert!(
        mtmd_dir.join("mtmd.h").exists(),
        "mtmd feature set but {mtmd_dir:?}/mtmd.h not found"
    );

    bindgen::Builder::default()
        .header("wrapper_mtmd.h")
        // C++ mode: mtmd.h's C++ block includes nlohmann/json.hpp.
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{}", mtmd_dir.display()))
        .clang_arg(format!("-I{}", src.join("include").display()))
        .clang_arg(format!("-I{}", src.join("ggml/include").display()))
        .clang_arg(format!("-I{}", src.join("vendor").display()))
        .allowlist_function("mtmd_.*")
        .allowlist_type("mtmd_.*")
        // These two take a C++ `json&` — parseable in C++ mode but not bindable.
        .blocklist_function("mtmd_input_chunk_from_json")
        .blocklist_function("mtmd_input_chunk_to_json")
        // llama/ggml/gguf types & functions come from the main C bindings; keep
        // this pass to `mtmd_*` only and reference the shared types via
        // `use super::*` at the include! site.
        .blocklist_type("llama_.*")
        .blocklist_type("ggml_.*")
        .blocklist_type("gguf_.*")
        .blocklist_function("llama_.*")
        .blocklist_function("ggml_.*")
        .blocklist_function("gguf_.*")
        // The C++-mode pass also surfaces `nlohmann_ordered_json` / `json` type
        // aliases (from `mtmd.h`'s json include) that would otherwise be emitted
        // and re-exported. The C mtmd API references none of them, so drop them.
        .blocklist_type("nlohmann.*")
        .blocklist_type("json")
        .blocklist_item("json")
        // In C++ mode the header's `namespace mtmd {...}` C++ wrappers match the
        // `mtmd_.*` allowlist and drag in their std/nlohmann template
        // dependencies (std::unique_ptr / std::string / std::vector, nlohmann
        // ordered_json). Emitting those verbatim fails to compile (unions with
        // non-Copy fields, bogus size asserts) and blocklisting them leaves a
        // dangling template type-parameter that panics bindgen. The C mtmd API
        // references none of that machinery, so emit it as opaque blobs (Copy,
        // no template descent). Combined with layout_tests(false) so we don't
        // assert on the blobs' internals.
        .opaque_type("std_.*")
        .opaque_type("std::.*")
        .opaque_type("nlohmann.*")
        .opaque_type("__gnu_cxx.*")
        .layout_tests(false)
        .prepend_enum_name(false)
        .derive_partialeq(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed to generate mtmd bindings")
        .write_to_file(out_dir.join("mtmd_bindings.rs"))
        .expect("failed to write mtmd_bindings.rs");
}

/// Recursively collect directories under `root` that contain a file matching `pred`.
fn dirs_with(root: &Path, pred: &dyn Fn(&str) -> bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root).into_iter().flatten() {
        if entry.file_type().is_file() {
            if let Some(name) = entry.file_name().to_str() {
                if pred(name) {
                    if let Some(parent) = entry.path().parent() {
                        let p = parent.to_path_buf();
                        if !out.contains(&p) {
                            out.push(p);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Prebuilt fast-path: link the shared `libllama`/`libggml` (+ static `libcommon.a`
/// under `common`) found under `lib_dir`.
///
/// NOTE: the `-Wl,-rpath` link-arg below applies only to THIS crate's own
/// artifacts — `cargo:rustc-link-arg` does not propagate to downstream binaries
/// (only link-search/link-lib do). Consumers (examples/tests/apps) must set
/// `LD_LIBRARY_PATH` to the `.so` dirs at runtime, or add rpath in their own
/// build script. See examples/simple for the documented `LD_LIBRARY_PATH` usage.
fn link_prebuilt(lib_dir: &Path, want_common: bool, _want_cuda: bool) {
    assert!(
        lib_dir.exists(),
        "IK_LLAMA_CPP_LIB_DIR does not exist: {lib_dir:?}"
    );

    let so_dirs = dirs_with(lib_dir, &|n| {
        n.starts_with("libllama.so") || n.starts_with("libggml") && n.contains(".so")
    });
    assert!(
        !so_dirs.is_empty(),
        "no libllama.so/libggml*.so found under {lib_dir:?}"
    );
    for d in &so_dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
    }

    // static libcommon.a first (it depends on symbols provided by the shared libs)
    if want_common {
        let common_dirs = dirs_with(lib_dir, &|n| n == "libcommon.a");
        assert!(
            !common_dirs.is_empty(),
            "`common` feature set but no libcommon.a found under {lib_dir:?} (expected {lib_dir:?}/common)"
        );
        for d in &common_dirs {
            println!("cargo:rustc-link-search=native={}", d.display());
        }
        println!("cargo:rustc-link-lib=static=common");
    }

    // ik ggml is monolithic: libggml.so -> `ggml`, libllama.so -> `llama`
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
}

/// CMake build of ik from source. Returns the cmake crate's output dir (install prefix).
fn cmake_build(
    src: &Path,
    _want_common: bool,
    want_cuda: bool,
    want_vulkan: bool,
    want_openmp: bool,
    want_native: bool,
    dynamic_link: bool,
) -> PathBuf {
    let mut cfg = cmake::Config::new(src);
    cfg.define("GGML_MAX_CONTEXTS", "2048") // load >64-shard split sets / large merges
        .define("LLAMA_CURL", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("BUILD_SHARED_LIBS", if dynamic_link { "ON" } else { "OFF" })
        .define("GGML_NATIVE", if want_native { "ON" } else { "OFF" })
        .define("GGML_OPENMP", if want_openmp { "ON" } else { "OFF" });
    // NOTE: ik always builds `common` (target `common` -> libcommon.a); there is
    // no LLAMA_BUILD_COMMON flag ([M2]).
    if want_cuda {
        cfg.define("GGML_CUDA", "ON");
        // Disable NCCL (multi-GPU all-reduce): single-GPU inference doesn't need
        // it and it otherwise pulls a libnccl link/runtime dep. Matches the
        // anchor's GGML_CUDA_NCCL=OFF intent.
        cfg.define("GGML_NCCL", "OFF");
    }
    if want_vulkan {
        cfg.define("GGML_VULKAN", "ON");
    }
    cfg.build()
}

/// Link the archives/libs from a from-source CMake build. ik has no install rules
/// for the static archives, so we glob the build tree.
fn link_built(dst: &Path, want_common: bool, dynamic_link: bool) {
    let build = dst.join("build");
    let search_root = if build.exists() {
        build
    } else {
        dst.to_path_buf()
    };
    let kind = if dynamic_link { "dylib" } else { "static" };
    let ext_ok = |n: &str| {
        if dynamic_link {
            n.contains(".so")
        } else {
            n.ends_with(".a")
        }
    };

    // Link order (static): most-dependent first — common -> llama -> ggml.
    let mut wanted: Vec<(&str, &str)> = Vec::new();
    if want_common {
        wanted.push(("common", "libcommon"));
    }
    wanted.push(("llama", "libllama"));
    wanted.push(("ggml", "libggml"));

    for (link_name, file_prefix) in wanted {
        let dirs = dirs_with(&search_root, &|n| n.starts_with(file_prefix) && ext_ok(n));
        assert!(
            !dirs.is_empty(),
            "could not find {file_prefix}.* under {search_root:?} after CMake build"
        );
        for d in &dirs {
            println!("cargo:rustc-link-search=native={}", d.display());
            if dynamic_link {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
            }
        }
        println!("cargo:rustc-link-lib={kind}={link_name}");
    }
}

/// Compile `wrapper_common.cpp` (the `ik_llama_rs_*` glue) with cc.
/// Compile the always-on core grammar glue (`wrapper_grammar.cpp`) into
/// `libik_llama_rs_grammar.a` (cc). It calls ik's internal
/// `llama_grammar_sample_impl` / `accept_impl`, so it must precede libllama on
/// the link line (emitted before the native-lib block in `main`).
fn compile_grammar_glue(src: &Path, manifest_dir: &Path) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(manifest_dir.join("wrapper_grammar.cpp"))
        .include(manifest_dir)
        .include(src.join("include"))
        .include(src.join("ggml/include"))
        .include(src.join("src")) // llama-grammar.h + llama-impl.h
        .flag_if_supported("-fPIC")
        .warnings(false);
    build.compile("ik_llama_rs_grammar");
}

fn compile_common_glue(src: &Path, manifest_dir: &Path) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(manifest_dir.join("wrapper_common.cpp"))
        .include(manifest_dir)
        .include(src.join("include"))
        .include(src.join("ggml/include"))
        .include(src.join("common")) // [m5] common/speculative.h drags in common.h/sampling.h
        .include(src.join("src")) // common/speculative.h includes src/llama-spec-features.h
        .include(src.join("vendor")) // json-schema-to-grammar.h -> <nlohmann/json.hpp>
        .define("LLAMA_RS_BUILD_COMMON", None)
        .flag_if_supported("-fPIC")
        .warnings(false);
    build.compile("ik_llama_rs_common");
}

/// Compile ik's `examples/mtmd/` multimodal sources into `libmtmd.a` (cc).
///
/// Sources: `mtmd.cpp`, `mtmd-audio.cpp`, `clip.cpp`, `mtmd-helper.cpp`.
/// Deliberately NOT `mtmd-cli.cpp` (a `main()` binary) or
/// `deprecation-warning.cpp`. Includes cover mtmd headers, the ik source root,
/// public headers, ggml headers, and `vendor/` (stb / miniaudio / nlohmann).
fn compile_mtmd(src: &Path) {
    let mtmd_dir = src.join("examples/mtmd");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(mtmd_dir.join("mtmd.cpp"))
        .file(mtmd_dir.join("mtmd-audio.cpp"))
        .file(mtmd_dir.join("clip.cpp"))
        .file(mtmd_dir.join("mtmd-helper.cpp"))
        .include(&mtmd_dir)
        .include(src)
        .include(src.join("include"))
        .include(src.join("ggml/include"))
        .include(src.join("vendor"))
        .flag_if_supported("-Wno-cast-qual")
        .pic(true)
        .warnings(false);
    build.compile("mtmd");
}

fn link_cuda() {
    // honor CUDA_PATH / /opt/cuda; find_cuda_helper resolves the toolkit dir
    let candidates = [
        env::var("CUDA_PATH").ok(),
        Some("/opt/cuda".to_string()),
        Some("/usr/local/cuda".to_string()),
    ];
    for c in candidates.into_iter().flatten() {
        let lib64 = PathBuf::from(&c).join("lib64");
        if lib64.exists() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
            // CUDA driver API (cuGetErrorString/cuDeviceGet/...) links against the
            // stub libcuda.so here; the real driver resolves at runtime.
            let stubs = lib64.join("stubs");
            if stubs.exists() {
                println!("cargo:rustc-link-search=native={}", stubs.display());
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    // driver API (needed by ggml-cuda common.cuh / device init)
    println!("cargo:rustc-link-lib=dylib=cuda");
}

fn link_system(static_stdcxx: bool, want_openmp: bool) {
    if static_stdcxx {
        println!("cargo:rustc-link-lib=static=stdc++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    if want_openmp {
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
}

fn write_manifest(
    src: &Path,
    out_dir: &Path,
    cuda: bool,
    vulkan: bool,
    openmp: bool,
    common: bool,
    backend: &str,
) {
    let sha = std::process::Command::new("git")
        .args(["-C", &src.display().to_string(), "rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let target = env::var("TARGET").unwrap_or_default();
    let mut backends = vec!["cpu"];
    if cuda {
        backends.push("cuda");
    }
    if vulkan {
        backends.push("vulkan");
    }
    if openmp {
        backends.push("openmp");
    }
    if common {
        backends.push("common");
    }
    let manifest = format!(
        "ik_sha={sha}\ntarget={target}\nGGML_MAX_CONTEXTS=2048\nbackends={}\nlink_backend={backend}\n",
        backends.join("+")
    );
    let _ = std::fs::write(out_dir.join("ik-build.txt"), &manifest);
    println!(
        "cargo:warning=ik-llama-cpp-sys: ik={} target={} backends={} ({})",
        &sha[..sha.len().min(8)],
        target,
        backends.join("+"),
        backend
    );
}
