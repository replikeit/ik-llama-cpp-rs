//! Stateful GBNF grammar + DRY samplers.
//!
//! ik_llama.cpp exposes two *stateful* sampler objects that do not fit the
//! stateless `llama_sample_*` array API modelled in [`crate::sampling`] (they
//! own C-side state that advances as tokens are accepted):
//!
//! * [`LlamaGrammar`] — a GBNF grammar constraint. Unlike the stock
//!   `llama-cpp-2` anchor (which parses GBNF in Rust and calls the now
//!   commented-out `llama_grammar_init`), ik parses the grammar C-side via
//!   `llama_sampler_init_grammar(vocab, grammar_str, root)`. The returned
//!   grammar retains a pointer to the model's vocab, so it is lifetime-tied to
//!   the [`LlamaModel`] it was built from.
//! * [`LlamaDrySampler`] — the DRY ("Don't Repeat Yourself") repetition
//!   sampler. It processes its sequence breakers against the vocab only at
//!   init, so it does not retain the model.
//!
//! Both own a raw C object and free it on `Drop`. Typical use inside a
//! generation loop: [`LlamaGrammar::apply`] / [`LlamaDrySampler::apply`] mutate
//! the candidate array in place before you draw a token; then
//! [`LlamaGrammar::accept_token`] / [`LlamaDrySampler::accept`] advance the
//! sampler state with the token you drew.

use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::ptr::NonNull;

use ik_llama_cpp_sys as sys;

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::sampling::LlamaTokenDataArray;
use crate::token::LlamaToken;

/// Error building a [`LlamaGrammar`].
#[derive(Debug, thiserror::Error)]
pub enum GrammarInitError {
    /// The grammar string or its root symbol contained an interior NUL byte.
    #[error("grammar string contained an interior NUL byte")]
    Nul(#[from] std::ffi::NulError),
    /// `llama_sampler_init_grammar` returned null: the GBNF failed to parse.
    #[error("failed to parse GBNF grammar")]
    Parse,
}

/// A stateful GBNF grammar constraint.
///
/// Build with [`LlamaGrammar::new`], then during generation call
/// [`apply`](Self::apply) on the candidate array before sampling and
/// [`accept_token`](Self::accept_token) with the token you drew to advance the
/// grammar state.
///
/// Lifetime-tied to the [`LlamaModel`]: ik's grammar retains a pointer to the
/// model's vocab (used to map tokens to text when accepting), so the model must
/// outlive the grammar. This is enforced at compile time by the `'model`
/// borrow.
///
/// The [`LlamaContext`] passed to [`apply`](Self::apply) /
/// [`accept_token`](Self::accept_token) must belong to the **same model** this
/// grammar was built from — `apply` reads the context's vocab while
/// `accept_token` reads the grammar's stored vocab, so mixing models would
/// constrain and advance against different vocabularies.
#[derive(Debug)]
pub struct LlamaGrammar<'model> {
    grammar: NonNull<sys::llama_grammar>,
    _model: PhantomData<&'model LlamaModel>,
}

impl<'model> LlamaGrammar<'model> {
    /// Compile a GBNF grammar for `model`.
    ///
    /// `grammar_str` is GBNF source; `root` is the name of the start symbol
    /// (conventionally `"root"`). Returns [`GrammarInitError::Parse`] if the
    /// grammar fails to parse C-side.
    ///
    /// # Errors
    ///
    /// [`GrammarInitError::Nul`] if `grammar_str` or `root` contains an interior
    /// NUL byte; [`GrammarInitError::Parse`] if the GBNF is invalid.
    pub fn new(
        model: &'model LlamaModel,
        grammar_str: &str,
        root: &str,
    ) -> Result<Self, GrammarInitError> {
        let c_grammar = CString::new(grammar_str)?;
        let c_root = CString::new(root)?;
        // SAFETY: `model` is a live, valid model; its vocab is const and lives
        // as long as the model (which outlives `self` via the `'model` borrow).
        let vocab = unsafe { sys::llama_model_get_vocab(model.model.as_ptr()) };
        // SAFETY: `vocab` is valid; both C strings are valid for the call. ik
        // parses `grammar_str`/`root` into rules (copying what it needs), so the
        // `CString`s may drop after this returns. Null => parse failure.
        let raw =
            unsafe { sys::llama_sampler_init_grammar(vocab, c_grammar.as_ptr(), c_root.as_ptr()) };
        NonNull::new(raw)
            .map(|grammar| Self {
                grammar,
                _model: PhantomData,
            })
            .ok_or(GrammarInitError::Parse)
    }

    /// Apply the grammar's constraints to `arr` in place: tokens the grammar
    /// cannot currently accept have their logit driven to `-inf`.
    ///
    /// Call this before sampling / drawing a token.
    pub fn apply(&self, ctx: &mut LlamaContext, arr: &mut LlamaTokenDataArray) {
        let mut c = arr.as_c();
        // SAFETY: `self.grammar` and `ctx` are valid; `c` describes `arr.data`
        // and is mutated in place through its pointer for the call.
        unsafe {
            sys::llama_grammar_apply(self.grammar.as_ptr(), ctx.as_ptr(), &mut c);
        }
    }

    /// Advance the grammar state by accepting `token`.
    ///
    /// Call this after you draw a token, so subsequent [`apply`](Self::apply)
    /// calls constrain the next position correctly.
    pub fn accept_token(&mut self, ctx: &mut LlamaContext, token: LlamaToken) {
        // SAFETY: `self.grammar` (exclusively borrowed) and `ctx` are valid.
        unsafe {
            sys::llama_grammar_accept_token(self.grammar.as_ptr(), ctx.as_ptr(), token.0);
        }
    }

    /// Deep-copy this grammar (independent parse state), or `None` if the C-side
    /// copy fails.
    #[must_use]
    pub fn try_clone(&self) -> Option<Self> {
        // SAFETY: `self.grammar` is valid; `llama_grammar_copy` yields an
        // independent grammar (or null on failure).
        let raw = unsafe { sys::llama_grammar_copy(self.grammar.as_ptr()) };
        NonNull::new(raw).map(|grammar| Self {
            grammar,
            _model: PhantomData,
        })
    }
}

impl Drop for LlamaGrammar<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.grammar` was produced by `new`/`try_clone` and is freed
        // exactly once (ownership is not shared).
        unsafe { sys::llama_grammar_free(self.grammar.as_ptr()) };
    }
}

/// Parameters for the DRY ("Don't Repeat Yourself") sampler.
///
/// DRY penalizes tokens that would extend a repeated sequence. See
/// <https://github.com/oobabooga/text-generation-webui/pull/5677>.
#[derive(Debug, Clone)]
pub struct DryParams {
    /// Penalty strength. `0.0` disables DRY.
    pub multiplier: f32,
    /// Exponential base for the length-scaled penalty (llama.cpp default `1.75`).
    pub base: f32,
    /// Minimum repeated-sequence length before a penalty applies (default `2`).
    pub allowed_length: i32,
    /// How many recent tokens to scan. `-1` = the whole context, `0` = disabled.
    pub penalty_last_n: i32,
    /// Strings that reset the repetition scan (e.g. newlines, quotes).
    pub seq_breakers: Vec<String>,
}

impl Default for DryParams {
    /// llama.cpp's stock defaults (DRY disabled via `multiplier = 0.0`).
    fn default() -> Self {
        Self {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: -1,
            seq_breakers: ["\n", ":", "\"", "*"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        }
    }
}

/// Error building a [`LlamaDrySampler`].
#[derive(Debug, thiserror::Error)]
pub enum DryInitError {
    /// A sequence breaker contained an interior NUL byte.
    #[error("a DRY sequence breaker contained an interior NUL byte")]
    Nul(#[from] std::ffi::NulError),
    /// `llama_sampler_init_dry` returned null.
    #[error("failed to initialize the DRY sampler")]
    Init,
}

/// The stateful DRY repetition sampler.
///
/// Build with [`LlamaDrySampler::new`], then [`apply`](Self::apply) it to the
/// candidate array before sampling and [`accept`](Self::accept) each drawn
/// token so its repetition history stays in sync.
#[derive(Debug)]
pub struct LlamaDrySampler {
    dry: NonNull<sys::llama_sampler_dry>,
}

impl LlamaDrySampler {
    /// Build a DRY sampler for `model` from `params`.
    ///
    /// # Errors
    ///
    /// [`DryInitError::Nul`] if a sequence breaker contains an interior NUL
    /// byte; [`DryInitError::Init`] if the C-side initializer returns null.
    pub fn new(model: &LlamaModel, params: &DryParams) -> Result<Self, DryInitError> {
        // Own the breaker C strings and collect their pointers. ik copies each
        // breaker into a `std::string` at init (and processes them against the
        // vocab), so neither the `CString`s nor the model need outlive this call.
        let c_breakers: Vec<CString> = params
            .seq_breakers
            .iter()
            .map(|s| CString::new(s.as_str()))
            .collect::<Result<_, _>>()?;
        let mut ptrs: Vec<*const c_char> = c_breakers.iter().map(|c| c.as_ptr()).collect();
        // SAFETY: `model` is a live, valid model; its vocab is const.
        let vocab = unsafe { sys::llama_model_get_vocab(model.model.as_ptr()) };
        // SAFETY: `vocab` is valid; `ptrs` is a `ptrs.len()`-long array of valid
        // C strings that live for the call. ik only reads `seq_breakers`
        // (declared `*mut` but not mutated). Null => init failure.
        let raw = unsafe {
            sys::llama_sampler_init_dry(
                vocab,
                params.multiplier,
                params.base,
                params.allowed_length,
                params.penalty_last_n,
                ptrs.as_mut_ptr(),
                ptrs.len(),
            )
        };
        NonNull::new(raw)
            .map(|dry| Self { dry })
            .ok_or(DryInitError::Init)
    }

    /// Apply the DRY penalty to `arr` in place, based on the accepted history.
    ///
    /// Call this before sampling / drawing a token.
    pub fn apply(&mut self, ctx: &mut LlamaContext, arr: &mut LlamaTokenDataArray) {
        let mut c = arr.as_c();
        // SAFETY: `ctx` and `self.dry` are valid; `c` describes `arr.data`,
        // mutated in place through its pointer.
        unsafe { sys::llama_sample_dry(ctx.as_ptr(), self.dry.as_ptr(), &mut c) };
    }

    /// Record a drawn `token` in the sampler's repetition history.
    pub fn accept(&mut self, token: LlamaToken) {
        // SAFETY: `self.dry` is exclusively borrowed and valid.
        unsafe { sys::llama_sampler_dry_accept(self.dry.as_ptr(), token.0) };
    }

    /// Clear the sampler's repetition history.
    pub fn reset(&mut self) {
        // SAFETY: `self.dry` is exclusively borrowed and valid.
        unsafe { sys::llama_sampler_dry_reset(self.dry.as_ptr()) };
    }

    /// Deep-copy this sampler (independent history), or `None` if the C-side
    /// copy fails.
    #[must_use]
    pub fn try_clone(&self) -> Option<Self> {
        // SAFETY: `self.dry` is valid; `llama_sampler_dry_clone` yields an
        // independent sampler (or null on failure).
        let raw = unsafe { sys::llama_sampler_dry_clone(self.dry.as_ptr()) };
        NonNull::new(raw).map(|dry| Self { dry })
    }
}

impl Drop for LlamaDrySampler {
    fn drop(&mut self) {
        // SAFETY: `self.dry` was produced by `new`/`try_clone` and is freed
        // exactly once (ownership is not shared).
        unsafe { sys::llama_sampler_dry_free(self.dry.as_ptr()) };
    }
}

/// Error converting a JSON Schema into a GBNF grammar via
/// [`json_schema_to_grammar`] (the `common` feature).
#[cfg(feature = "common")]
#[derive(Debug, thiserror::Error)]
pub enum JsonSchemaError {
    /// The schema string contained an interior NUL byte.
    #[error("JSON schema string contained an interior NUL byte")]
    Nul(#[from] std::ffi::NulError),
    /// The C-side conversion failed (invalid schema / parse error). The wrapped
    /// value is the raw `llama_rs_status` code.
    #[error("JSON schema to grammar conversion failed (status {0})")]
    Convert(i32),
    /// The produced grammar was not valid UTF-8 (should not happen).
    #[error("converted grammar was not valid UTF-8")]
    Utf8,
}

/// Convert a JSON Schema (given as a JSON string) into a GBNF grammar string,
/// ready to pass to [`LlamaGrammar::new`].
///
/// Wraps ik's `common/json-schema-to-grammar` (hence the `common` feature), the
/// same conversion `llama-server` uses. Useful for tool / function calling:
/// convert a function's parameter schema into a grammar, build a
/// [`LlamaGrammar`] from it, and constrain generation so the model can only emit
/// arguments that satisfy the schema.
///
/// # Errors
///
/// [`JsonSchemaError::Nul`] if `schema_json` has an interior NUL byte;
/// [`JsonSchemaError::Convert`] if the schema is invalid or fails to convert;
/// [`JsonSchemaError::Utf8`] if the produced grammar is not valid UTF-8.
#[cfg(feature = "common")]
pub fn json_schema_to_grammar(schema_json: &str) -> Result<String, JsonSchemaError> {
    let schema = CString::new(schema_json)?;
    let mut out: *mut c_char = std::ptr::null_mut();
    // SAFETY: `schema` is a valid C string; `out` is a valid out-pointer. On
    // success the C side writes a heap-allocated NUL-terminated string to `out`
    // (allocated by its matching allocator) that we free below.
    let status =
        unsafe { sys::ik_llama_rs_json_schema_to_grammar(schema.as_ptr(), false, &mut out) };
    // LLAMA_RS_STATUS_OK == 0 (see wrapper_utils.h).
    if status as i32 != 0 || out.is_null() {
        // On a non-OK status the C side leaves `out` NULL; free defensively in
        // case a future change sets it before failing.
        if !out.is_null() {
            // SAFETY: `out` was produced by the matching C allocator.
            unsafe { sys::ik_llama_rs_string_free(out) };
        }
        return Err(JsonSchemaError::Convert(status as i32));
    }
    // SAFETY: `out` is a valid NUL-terminated C string that we own.
    let bytes = unsafe { std::ffi::CStr::from_ptr(out) }.to_bytes().to_vec();
    // SAFETY: `out` came from the matching C allocator and is freed exactly once.
    unsafe { sys::ik_llama_rs_string_free(out) };
    String::from_utf8(bytes).map_err(|_| JsonSchemaError::Utf8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_params_default_is_disabled() {
        let p = DryParams::default();
        assert_eq!(p.multiplier, 0.0, "DRY is disabled by default");
        assert_eq!(p.seq_breakers.len(), 4);
        assert!(p.seq_breakers.iter().any(|b| b == "\n"));
    }
}
