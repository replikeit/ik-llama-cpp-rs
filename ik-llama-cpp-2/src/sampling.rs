//! Sampling — a chain-object [`LlamaSampler`] matching `llama-cpp-2`'s API,
//! emulated over ik_llama.cpp's **legacy** `llama_sample_*` array functions.
//!
//! ik has no `llama_sampler_chain_*` and only `grammar`/`dry` sampler-init
//! functions, so this reconstructs the anchor's chain model in Rust: each
//! constructor builds a one-stage sampler, [`LlamaSampler::chain_simple`]
//! concatenates them, and [`apply`](LlamaSampler::apply) runs the stages over a
//! [`LlamaTokenDataArray`] in order. The legacy transforms accept a null
//! `llama_context` (ik guards it internally), so no context is needed until the
//! final draw; the `dist` selector draws in Rust from a seeded RNG.

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr::NonNull;

use ik_llama_cpp_sys as sys;

use crate::context::LlamaContext;
use crate::grammar::GrammarInitError;
use crate::model::LlamaModel;
use crate::token::LlamaToken;

// The candidate array lives in `token::data_array` (matching the anchor);
// re-exported for ergonomics / back-compat.
pub use crate::token::data_array::LlamaTokenDataArray;

/// A small deterministic RNG for the `dist` selector, so a stochastic sampler
/// is reproducible from its seed without pulling in a `rand` dependency
/// (xorshift64\* seeded via SplitMix64).
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u32) -> Self {
        let mut z = u64::from(seed).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self {
            state: (z ^ (z >> 31)) | 1,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f32` in `[0, 1)` (top 24 bits / 2^24).
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        (bits as f32) / (1u32 << 24) as f32
    }
}

#[derive(Debug)]
enum Stage {
    TopK(i32),
    TopP {
        p: f32,
        min_keep: usize,
    },
    MinP {
        p: f32,
        min_keep: usize,
    },
    Typical {
        p: f32,
        min_keep: usize,
    },
    TopNSigma(f32),
    TailFree {
        z: f32,
        min_keep: usize,
    },
    Temp(f32),
    TempExt {
        t: f32,
        delta: f32,
        exponent: f32,
    },
    Penalties {
        last_n: i32,
        repeat: f32,
        freq: f32,
        presence: f32,
    },
    /// Stochastic selector: softmax + a seeded weighted draw.
    Dist(Rng),
    /// Deterministic selector: argmax over logits.
    Greedy,
    /// GBNF grammar constraint (masks disallowed tokens). Owns the C grammar;
    /// `vocab` is stashed from the model so apply/accept need no context.
    Grammar {
        grammar: NonNull<sys::llama_grammar>,
        vocab: *const sys::llama_vocab,
    },
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Stage::Grammar { grammar, .. } = self {
            // SAFETY: `grammar` was produced by `llama_sampler_init_grammar[_lazy]`
            // and is owned solely by this stage (Stage is not Clone).
            unsafe { sys::llama_grammar_free(grammar.as_ptr()) };
        }
    }
}

impl Stage {
    fn run(&mut self, arr: &mut LlamaTokenDataArray, hist: &[sys::llama_token]) {
        let null = std::ptr::null_mut();
        match self {
            Stage::TopK(k) => {
                // SAFETY: null ctx is guarded by ik; `c` describes `arr.data`.
                unsafe {
                    arr.modify_as_c_llama_token_data_array(|c| {
                        sys::llama_sample_top_k(null, c, *k, 1)
                    });
                }
            }
            Stage::TopP { p, min_keep } => unsafe {
                arr.modify_as_c_llama_token_data_array(|c| {
                    sys::llama_sample_top_p(null, c, *p, *min_keep)
                });
            },
            Stage::MinP { p, min_keep } => unsafe {
                arr.modify_as_c_llama_token_data_array(|c| {
                    sys::llama_sample_min_p(null, c, *p, *min_keep)
                });
            },
            Stage::Typical { p, min_keep } => unsafe {
                arr.modify_as_c_llama_token_data_array(|c| {
                    sys::llama_sample_typical(null, c, *p, *min_keep)
                });
            },
            Stage::TopNSigma(n) => unsafe {
                arr.modify_as_c_llama_token_data_array(|c| {
                    sys::llama_sample_top_n_sigma(null, c, *n)
                });
            },
            Stage::TailFree { z, min_keep } => unsafe {
                arr.modify_as_c_llama_token_data_array(|c| {
                    sys::llama_sample_tail_free(null, c, *z, *min_keep)
                });
            },
            Stage::Temp(t) => unsafe {
                arr.modify_as_c_llama_token_data_array(|c| sys::llama_sample_temp(null, c, *t));
            },
            Stage::TempExt { t, delta, exponent } => unsafe {
                arr.modify_as_c_llama_token_data_array(|c| {
                    sys::llama_sample_entropy(null, c, *t - *delta, *t + *delta, *exponent);
                });
            },
            Stage::Penalties {
                last_n,
                repeat,
                freq,
                presence,
            } => {
                let use_n = (*last_n).max(0) as usize;
                let use_n = use_n.min(hist.len());
                let start = hist.len() - use_n;
                let window = &hist[start..];
                // SAFETY: null ctx guarded by ik; `window` lives for the call.
                unsafe {
                    arr.modify_as_c_llama_token_data_array(|c| {
                        sys::llama_sample_repetition_penalties(
                            null,
                            c,
                            window.as_ptr(),
                            window.len(),
                            *repeat,
                            *freq,
                            *presence,
                        );
                    });
                }
            }
            Stage::Dist(rng) => {
                let r = rng.next_f32();
                // SAFETY: null ctx guarded; softmax fills `p`; we read the C
                // array within `[0, size)` and set `selected`.
                unsafe {
                    arr.modify_as_c_llama_token_data_array(|c| {
                        // Guard BEFORE softmax: llama_sample_softmax_impl does
                        // GGML_ASSERT(size > 0) and would abort on an empty set.
                        if c.size == 0 {
                            c.selected = -1;
                            return;
                        }
                        sys::llama_sample_softmax(null, c);
                        let data = std::slice::from_raw_parts(c.data, c.size);
                        let mut cum = 0.0_f32;
                        let mut chosen = (c.size - 1) as i64;
                        for (i, d) in data.iter().enumerate() {
                            cum += d.p;
                            if r < cum {
                                chosen = i as i64;
                                break;
                            }
                        }
                        c.selected = chosen;
                    });
                }
            }
            Stage::Greedy => {
                // SAFETY: we only read `[0, size)` and set `selected`.
                unsafe {
                    arr.modify_as_c_llama_token_data_array(|c| {
                        if c.size == 0 {
                            c.selected = -1;
                            return;
                        }
                        let data = std::slice::from_raw_parts(c.data, c.size);
                        let mut best = 0i64;
                        let mut best_logit = f32::NEG_INFINITY;
                        for (i, d) in data.iter().enumerate() {
                            if d.logit > best_logit {
                                best_logit = d.logit;
                                best = i as i64;
                            }
                        }
                        c.selected = best;
                    });
                }
            }
            Stage::Grammar { grammar, vocab } => {
                let g = grammar.as_ptr();
                let v = *vocab;
                // SAFETY: `g`/`v` valid; the glue applies grammar constraints to
                // `c` in place (masks disallowed tokens), null-`smpl` internally.
                unsafe {
                    arr.modify_as_c_llama_token_data_array(|c| {
                        sys::ik_llama_rs_grammar_apply(g, v, c);
                    });
                }
            }
        }
    }
}

/// Build the single lazy-grammar trigger *pattern* from `trigger_words`,
/// reproducing byte-for-byte what ik's C++ `llama_sampler_init_grammar_lazy`
/// assembles internally: `[\s\S]*?(w1|w2|…)[\s\S]*`, with each word's regex
/// metacharacters backslash-escaped. Returns `None` when there are no words (so
/// the caller passes no pattern — a token-only lazy trigger), matching the C++
/// which leaves `trigger_patterns` empty in that case.
fn build_lazy_trigger_pattern(
    trigger_words: impl IntoIterator<Item = impl AsRef<[u8]>>,
) -> Option<Vec<u8>> {
    let mut words = trigger_words.into_iter();
    let first = words.next()?;
    let mut pattern: Vec<u8> = b"[\\s\\S]*?(".to_vec();
    push_regex_escaped(&mut pattern, first.as_ref());
    for word in words {
        pattern.push(b'|');
        push_regex_escaped(&mut pattern, word.as_ref());
    }
    pattern.extend_from_slice(b")[\\s\\S]*");
    Some(pattern)
}

/// Append `word` to `out`, backslash-escaping the regex metacharacters ik's C++
/// escapes (`std::regex special_chars = "[.^$|()*+?\\[\\]{}\\\\]"`). All are
/// ASCII, so escaping byte-wise matches the C++ (which runs `std::regex_replace`
/// over the raw bytes) even for multi-byte UTF-8 words.
fn push_regex_escaped(out: &mut Vec<u8>, word: &[u8]) {
    for &b in word {
        if matches!(
            b,
            b'.' | b'^'
                | b'$'
                | b'|'
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b'?'
                | b'['
                | b']'
                | b'{'
                | b'}'
                | b'\\'
        ) {
            out.push(b'\\');
        }
        out.push(b);
    }
}

/// A sampler: an ordered chain of transform stages ending in a selector,
/// matching `llama-cpp-2`'s `LlamaSampler`.
///
/// Build single stages with the constructors ([`greedy`](Self::greedy),
/// [`temp`](Self::temp), [`top_k`](Self::top_k), …) and compose them with
/// [`chain_simple`](Self::chain_simple). During generation, either call
/// [`sample`](Self::sample) (which reads logits from the context) or
/// [`apply`](Self::apply) on a candidate array you already built, then read
/// [`LlamaTokenDataArray::selected_token`]. Call [`accept`](Self::accept) with
/// each drawn token so stateful stages (penalties, grammar) stay in sync.
///
/// Not `Clone` (a grammar stage owns a C grammar object freed on drop).
#[derive(Debug)]
pub struct LlamaSampler {
    stages: Vec<Stage>,
    history: Vec<LlamaToken>,
}

impl LlamaSampler {
    fn single(stage: Stage) -> Self {
        Self {
            stages: vec![stage],
            history: Vec::new(),
        }
    }

    /// Compose several samplers into one chain, applied left to right.
    #[must_use]
    pub fn chain_simple(samplers: impl IntoIterator<Item = LlamaSampler>) -> Self {
        let mut stages = Vec::new();
        for s in samplers {
            stages.extend(s.stages);
        }
        Self {
            stages,
            history: Vec::new(),
        }
    }

    /// Greedy (argmax) selector.
    #[must_use]
    pub fn greedy() -> Self {
        Self::single(Stage::Greedy)
    }

    /// Stochastic distribution selector seeded by `seed` (softmax + draw).
    #[must_use]
    pub fn dist(seed: u32) -> Self {
        Self::single(Stage::Dist(Rng::new(seed)))
    }

    /// Temperature scaling.
    #[must_use]
    pub fn temp(t: f32) -> Self {
        Self::single(Stage::Temp(t))
    }

    /// Dynamic-temperature ("entropy") scaling over `[t-delta, t+delta]`.
    #[must_use]
    pub fn temp_ext(t: f32, delta: f32, exponent: f32) -> Self {
        Self::single(Stage::TempExt { t, delta, exponent })
    }

    /// Top-k filtering.
    #[must_use]
    pub fn top_k(k: i32) -> Self {
        Self::single(Stage::TopK(k))
    }

    /// Top-p (nucleus) filtering.
    #[must_use]
    pub fn top_p(p: f32, min_keep: usize) -> Self {
        Self::single(Stage::TopP { p, min_keep })
    }

    /// Min-p filtering.
    #[must_use]
    pub fn min_p(p: f32, min_keep: usize) -> Self {
        Self::single(Stage::MinP { p, min_keep })
    }

    /// Locally-typical filtering.
    #[must_use]
    pub fn typical(p: f32, min_keep: usize) -> Self {
        Self::single(Stage::Typical { p, min_keep })
    }

    /// Top-nσ filtering.
    #[must_use]
    pub fn top_n_sigma(n: f32) -> Self {
        Self::single(Stage::TopNSigma(n))
    }

    /// Tail-free filtering.
    #[must_use]
    pub fn tail_free(z: f32, min_keep: usize) -> Self {
        Self::single(Stage::TailFree { z, min_keep })
    }

    /// A GBNF grammar constraint stage.
    ///
    /// `grammar_str` is GBNF source, `root` the start symbol (usually `"root"`).
    /// The stage masks tokens the grammar cannot currently accept; call
    /// [`accept`](Self::accept) with each drawn token to advance grammar state.
    ///
    /// # Safety contract
    ///
    /// The returned sampler stashes a pointer into `model`'s vocab, so **`model`
    /// must outlive this sampler** — using it after the model is dropped is
    /// undefined behavior. This is not encoded in the type (the sampler stays
    /// `'static`) so that it drops in for `llama-cpp-2`, which makes the same
    /// choice; keep the model alive at least as long as any sampler built from
    /// it (a `'static` model trivially satisfies this). Prefer
    /// [`crate::grammar::LlamaGrammar`] if you want the model lifetime enforced.
    ///
    /// # Errors
    ///
    /// [`GrammarInitError::Nul`] on an interior NUL byte; [`GrammarInitError::Parse`]
    /// if the GBNF fails to parse.
    pub fn grammar(
        model: &LlamaModel,
        grammar_str: &str,
        root: &str,
    ) -> Result<Self, GrammarInitError> {
        let c_gbnf = CString::new(grammar_str)?;
        let c_root = CString::new(root)?;
        // SAFETY: model is valid; vocab is const and lives with the model.
        let vocab = unsafe { sys::llama_model_get_vocab(model.model.as_ptr()) };
        // SAFETY: valid vocab + two valid C strings (copied C-side). Null = parse fail.
        let raw =
            unsafe { sys::llama_sampler_init_grammar(vocab, c_gbnf.as_ptr(), c_root.as_ptr()) };
        let grammar = NonNull::new(raw).ok_or(GrammarInitError::Parse)?;
        Ok(Self::single(Stage::Grammar { grammar, vocab }))
    }

    /// A lazy GBNF grammar constraint: it stays inert until one of
    /// `trigger_words` / `trigger_tokens` appears, then constrains generation.
    ///
    /// # Errors
    ///
    /// [`GrammarInitError::Nul`] on an interior NUL byte in the grammar, root, or
    /// a trigger word; [`GrammarInitError::Parse`] if the GBNF fails to parse.
    pub fn grammar_lazy(
        model: &LlamaModel,
        grammar_str: &str,
        root: &str,
        trigger_words: impl IntoIterator<Item = impl AsRef<[u8]>>,
        trigger_tokens: &[LlamaToken],
    ) -> Result<Self, GrammarInitError> {
        let c_gbnf = CString::new(grammar_str)?;
        let c_root = CString::new(root)?;
        // Build the trigger *pattern* in Rust and call `..._lazy_patterns`, rather
        // than passing `trigger_words` to the deprecated `..._lazy`. The words path
        // has a use-after-scope bug in ik's C++: it assembles the pattern in a local
        // `std::string`, stashes a pointer to that local, then reads the pointer
        // *after* the string has dropped (llama-sampling.cpp:1372-1387). The
        // `_patterns` entry point copies each pattern into an owned `std::string`
        // immediately (llama-grammar.cpp:1318), so a live-for-the-call pattern is
        // safe. We reproduce the exact pattern the C++ would have built.
        let c_pattern = match build_lazy_trigger_pattern(trigger_words) {
            Some(pattern) => Some(CString::new(pattern)?),
            None => None,
        };
        // 0 or 1 pattern pointers, borrowing `c_pattern` (alive across the call).
        let mut pattern_ptrs: Vec<*const c_char> = c_pattern.iter().map(|c| c.as_ptr()).collect();
        // Match the original semantics when there are no words: (null, 0).
        let (pat_ptr, pat_len) = if pattern_ptrs.is_empty() {
            (std::ptr::null_mut::<*const c_char>(), 0usize)
        } else {
            (pattern_ptrs.as_mut_ptr(), pattern_ptrs.len())
        };
        let tokens: Vec<sys::llama_token> = trigger_tokens.iter().map(|t| t.0).collect();
        // SAFETY: model valid → const vocab.
        let vocab = unsafe { sys::llama_model_get_vocab(model.model.as_ptr()) };
        // SAFETY: valid vocab + C strings + trigger arrays live for the call
        // (ik copies what it needs). Null = parse failure.
        let raw = unsafe {
            sys::llama_sampler_init_grammar_lazy_patterns(
                vocab,
                c_gbnf.as_ptr(),
                c_root.as_ptr(),
                pat_ptr,
                pat_len,
                tokens.as_ptr(),
                tokens.len(),
            )
        };
        let grammar = NonNull::new(raw).ok_or(GrammarInitError::Parse)?;
        Ok(Self::single(Stage::Grammar { grammar, vocab }))
    }

    /// Repetition / frequency / presence penalties over the accepted history.
    #[must_use]
    pub fn penalties(
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> Self {
        Self::single(Stage::Penalties {
            last_n: penalty_last_n,
            repeat: penalty_repeat,
            freq: penalty_freq,
            presence: penalty_present,
        })
    }

    /// Run every stage over `arr` in order (transforms + selector). Read the
    /// result with [`LlamaTokenDataArray::selected_token`].
    ///
    /// A no-op on an empty candidate array (leaving `selected = None`): ik's
    /// legacy samplers `GGML_ASSERT(size > 0)` and would abort the process, so
    /// this safe wrapper must not forward an empty set into them.
    pub fn apply(&mut self, arr: &mut LlamaTokenDataArray) {
        if arr.data.is_empty() {
            return;
        }
        let hist: Vec<sys::llama_token> = self.history.iter().map(|t| t.0).collect();
        for stage in &mut self.stages {
            stage.run(arr, &hist);
        }
    }

    /// Record `token`: appends to the history (penalties stage) and advances any
    /// grammar stage.
    ///
    /// If a grammar stage is present, only feed tokens the grammar permits at the
    /// current position (i.e. tokens surviving [`apply`](Self::apply)). Accepting
    /// a grammar-disallowed end-of-generation token aborts the process (ik's
    /// `llama_grammar_accept_impl` treats it as fatal).
    pub fn accept(&mut self, token: LlamaToken) {
        self.history.push(token);
        for stage in &mut self.stages {
            if let Stage::Grammar { grammar, vocab } = stage {
                // SAFETY: `grammar`/`vocab` valid; the glue advances grammar
                // state with null `smpl` internally.
                unsafe { sys::ik_llama_rs_grammar_accept(grammar.as_ptr(), *vocab, token.0) };
            }
        }
    }

    /// Clear the accepted-token history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Build the candidate array from the context's logits at batch index `idx`,
    /// run the chain, and return the selected token (argmax fallback if no
    /// selector ran).
    pub fn sample(&mut self, ctx: &LlamaContext, idx: i32) -> LlamaToken {
        let mut arr = ctx.token_data_array_ith(idx);
        self.apply(&mut arr);
        arr.selected_token().unwrap_or_else(|| {
            arr.data
                .iter()
                .max_by(|a, b| {
                    a.logit()
                        .partial_cmp(&b.logit())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map_or(LlamaToken(0), crate::token::data::LlamaTokenData::id)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::data::LlamaTokenData;

    #[test]
    fn greedy_selects_argmax() {
        let mut arr = LlamaTokenDataArray::from_logits(&[0.1, 2.5, -1.0, 2.4, 0.0]);
        let mut s = LlamaSampler::greedy();
        s.apply(&mut arr);
        assert_eq!(s.stages.len(), 1);
        assert_eq!(arr.selected_token(), Some(LlamaToken(1)));
    }

    #[test]
    fn dist_is_deterministic_for_a_seed() {
        let logits = [0.2_f32, 1.0, 0.5, 3.0, -0.5];
        let mut a = LlamaSampler::dist(1234);
        let mut b = LlamaSampler::dist(1234);
        let mut arr_a = LlamaTokenDataArray::from_logits(&logits);
        let mut arr_b = LlamaTokenDataArray::from_logits(&logits);
        a.apply(&mut arr_a);
        b.apply(&mut arr_b);
        assert!(arr_a.selected_token().is_some());
        assert_eq!(arr_a.selected_token(), arr_b.selected_token());
    }

    #[test]
    fn apply_on_empty_array_is_noop_not_abort() {
        // ik's legacy samplers GGML_ASSERT(size > 0); apply must short-circuit
        // an empty candidate set rather than forward it (which would SIGABRT).
        let mut arr = LlamaTokenDataArray::new(Vec::new(), false);
        let mut s = LlamaSampler::chain_simple([
            LlamaSampler::top_k(3),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::temp(0.8),
            LlamaSampler::dist(1),
        ]);
        s.apply(&mut arr);
        assert_eq!(arr.selected_token(), None);
    }

    #[test]
    fn chain_composes_stages() {
        let s = LlamaSampler::chain_simple([
            LlamaSampler::top_k(3),
            LlamaSampler::temp(0.8),
            LlamaSampler::greedy(),
        ]);
        assert_eq!(s.stages.len(), 3);
        // history + accept
        let mut s = s;
        s.accept(LlamaTokenData::new(LlamaToken(5), 0.0, 0.0).id());
        assert_eq!(s.history, vec![LlamaToken(5)]);
    }

    #[test]
    fn lazy_trigger_pattern_none_when_no_words() {
        let empty: [&[u8]; 0] = [];
        assert_eq!(build_lazy_trigger_pattern(empty), None);
    }

    #[test]
    fn lazy_trigger_pattern_wraps_and_joins() {
        // Two plain words -> `[\s\S]*?(<tool>|call)[\s\S]*` (byte-for-byte the
        // string ik's C++ would have assembled from these trigger words).
        let pat = build_lazy_trigger_pattern([b"<tool>".as_slice(), b"call".as_slice()])
            .expect("some pattern");
        assert_eq!(pat, br"[\s\S]*?(<tool>|call)[\s\S]*".to_vec());
    }

    #[test]
    fn lazy_trigger_pattern_escapes_metacharacters() {
        // Every regex metacharacter ik escapes must be backslash-prefixed so the
        // trigger matches the literal word, not a regex.
        let pat =
            build_lazy_trigger_pattern([br".^$|()*+?[]{}\".as_slice()]).expect("some pattern");
        assert_eq!(
            pat,
            br"[\s\S]*?(\.\^\$\|\(\)\*\+\?\[\]\{\}\\)[\s\S]*".to_vec()
        );
    }

    #[test]
    fn lazy_trigger_pattern_preserves_non_meta_utf8() {
        // Multi-byte UTF-8 passes through unescaped (its bytes are all >= 0x80,
        // none collide with the ASCII metacharacters).
        let pat = build_lazy_trigger_pattern(["café".as_bytes()]).expect("some pattern");
        let mut expected = br"[\s\S]*?(".to_vec();
        expected.extend_from_slice("café".as_bytes());
        expected.extend_from_slice(br")[\s\S]*");
        assert_eq!(pat, expected);
    }
}
