//! Safe wrapper around `llama_model` ([`LlamaModel`]).

pub mod chat;
pub mod lora;
pub mod meta;
pub mod params;

use std::ffi::CString;
use std::num::NonZeroU16;
use std::os::raw::c_char;
use std::path::Path;
use std::ptr::NonNull;

use ik_llama_cpp_sys as sys;

use crate::llama_backend::LlamaBackend;
use crate::token::LlamaToken;
use crate::LlamaError;

pub use params::LlamaModelParams;

/// A loaded ik_llama.cpp model.
///
/// Uses ik's model-pointer tokenizer/vocab API (`llama_tokenize(model, …)`,
/// `llama_token_bos(model)`, …), which differs from modern stock llama.cpp's
/// vocab-pointer API.
#[derive(Debug)]
pub struct LlamaModel {
    pub(crate) model: NonNull<sys::llama_model>,
}

// SAFETY: after loading, the model is immutable and safe to share across threads
// (matches llama-cpp-2's contract).
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

/// Whether to prepend a BOS (beginning-of-sequence) token when tokenizing.
///
/// Mirrors `llama-cpp-2`'s `model::AddBos`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddBos {
    /// Prepend the BOS token.
    Always,
    /// Do not prepend the BOS token.
    Never,
}

impl LlamaModel {
    /// Load a model from a single GGUF file (for split models, pass the merged
    /// file or the first shard — see the crate docs on SPECIAL_SPLIT).
    ///
    /// # Invariant
    ///
    /// The [`LlamaBackend`] passed here (and any derived [`crate::LlamaContext`])
    /// **must outlive** the returned model. Dropping the backend first runs
    /// `llama_backend_free` before the model/context are freed, which is unsound.
    /// The natural nested/reverse-drop order (backend → model → context declared
    /// outer-to-inner, dropped inner-to-outer) satisfies this. (This mirrors the
    /// `llama-cpp-2` anchor's contract.)
    pub fn load_from_file(
        _backend: &LlamaBackend,
        path: impl AsRef<Path>,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaError> {
        let path = path.as_ref();
        let c_path = CString::new(path.to_str().ok_or(LlamaError::InvalidPath)?)?;
        // SAFETY: valid C string + a fully-initialized params struct.
        let raw = unsafe { sys::llama_model_load_from_file(c_path.as_ptr(), params.params) };
        NonNull::new(raw)
            .map(|model| Self { model })
            .ok_or_else(|| LlamaError::ModelLoad(path.display().to_string()))
    }

    /// Number of vocabulary tokens.
    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        unsafe { sys::llama_n_vocab(self.model.as_ptr()) }
    }

    /// Number of MTP / NextN prediction layers (0 if the model has none).
    #[must_use]
    pub fn n_nextn_layer(&self) -> i32 {
        unsafe { sys::llama_model_n_nextn_layer(self.model.as_ptr()) }
    }

    /// Beginning-of-sequence token.
    #[must_use]
    pub fn token_bos(&self) -> LlamaToken {
        LlamaToken(unsafe { sys::llama_token_bos(self.model.as_ptr()) })
    }

    /// End-of-sequence token.
    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        LlamaToken(unsafe { sys::llama_token_eos(self.model.as_ptr()) })
    }

    /// Whether `token` marks end-of-generation (EOS/EOT/etc.).
    #[must_use]
    pub fn is_eog(&self, token: LlamaToken) -> bool {
        unsafe { sys::llama_token_is_eog(self.model.as_ptr(), token.0) }
    }

    /// Whether `token` marks end-of-generation (alias of [`Self::is_eog`],
    /// matching the `llama-cpp-2` method name).
    #[must_use]
    pub fn is_eog_token(&self, token: LlamaToken) -> bool {
        self.is_eog(token)
    }

    /// Create a context for this model (matches `llama-cpp-2`'s
    /// `model.new_context(&backend, params)`).
    ///
    /// The [`LlamaBackend`] argument is accepted for API parity; the returned
    /// context borrows `self`, so the model must outlive it.
    ///
    /// # Errors
    ///
    /// [`LlamaError::ContextCreation`] if `llama_init_from_model` returns null.
    pub fn new_context<'a>(
        &'a self,
        _backend: &LlamaBackend,
        params: crate::context::params::LlamaContextParams,
    ) -> Result<crate::context::LlamaContext<'a>, LlamaError> {
        crate::context::LlamaContext::new(self, &params)
    }

    /// Tokenize `text`, choosing whether to prepend BOS via [`AddBos`] (matches
    /// `llama-cpp-2`'s `str_to_token`). Special tokens are parsed.
    ///
    /// # Errors
    ///
    /// [`LlamaError::Nul`] on an interior NUL byte; [`LlamaError::Tokenize`] if
    /// tokenization fails.
    pub fn str_to_token(&self, text: &str, add_bos: AddBos) -> Result<Vec<LlamaToken>, LlamaError> {
        self.tokenize(text, matches!(add_bos, AddBos::Always))
    }

    /// Tokenize `text`. `add_bos` prepends the BOS token; special tokens are parsed.
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<LlamaToken>, LlamaError> {
        let c_text = CString::new(text)?;
        let text_len = text.len() as i32;
        // Generous first guess; on overflow ik returns -(required).
        let mut cap = (text.len() + 16) as i32;
        let mut buf = vec![0 as sys::llama_token; cap as usize];
        // SAFETY: buf has `cap` slots.
        let mut n = unsafe {
            sys::llama_tokenize(
                self.model.as_ptr(),
                c_text.as_ptr(),
                text_len,
                buf.as_mut_ptr(),
                cap,
                add_bos,
                true,
            )
        };
        if n < 0 {
            cap = -n;
            buf = vec![0 as sys::llama_token; cap as usize];
            n = unsafe {
                sys::llama_tokenize(
                    self.model.as_ptr(),
                    c_text.as_ptr(),
                    text_len,
                    buf.as_mut_ptr(),
                    cap,
                    add_bos,
                    true,
                )
            };
        }
        if n < 0 {
            return Err(LlamaError::Tokenize);
        }
        buf.truncate(n as usize);
        Ok(buf.into_iter().map(LlamaToken).collect())
    }

    /// Raw bytes of a single token's piece.
    ///
    /// `special` controls whether special/control tokens render as text;
    /// `lstrip` strips that many leading spaces. `buf_size` is the initial
    /// buffer guess — on overflow the call retries once with the exact size.
    ///
    /// # Errors
    ///
    /// [`LlamaError::TokenToPiece`] if the C conversion fails.
    pub fn token_to_piece_bytes(
        &self,
        token: LlamaToken,
        buf_size: usize,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, LlamaError> {
        let lstrip_i = lstrip.map_or(0, |n| i32::from(n.get()));
        let mut buf = vec![0u8; buf_size.max(1)];
        let write = |buf: &mut [u8]| unsafe {
            sys::llama_token_to_piece(
                self.model.as_ptr(),
                token.0,
                buf.as_mut_ptr().cast::<c_char>(),
                i32::try_from(buf.len()).unwrap_or(i32::MAX),
                lstrip_i,
                special,
            )
        };
        let mut n = write(&mut buf);
        if n < 0 {
            buf = vec![0u8; (-n) as usize];
            n = write(&mut buf);
            if n < 0 {
                return Err(LlamaError::TokenToPiece);
            }
        }
        buf.truncate(n as usize);
        Ok(buf)
    }

    /// Convert a single token to text, decoding its bytes incrementally through
    /// `decoder` (so a multi-byte UTF-8 sequence split across token boundaries is
    /// reassembled). Matches `llama-cpp-2`'s `token_to_piece`.
    ///
    /// `special` renders special/control tokens as text; `lstrip` strips leading
    /// spaces. Use one decoder (`encoding_rs::UTF_8.new_decoder()`) per stream.
    ///
    /// A multi-byte sequence split at the very end of the stream stays buffered
    /// in `decoder` (this never flushes with `last = true`); that matches the
    /// `llama-cpp-2` anchor and is a non-issue for a stream ending on a token
    /// boundary.
    ///
    /// # Errors
    ///
    /// [`LlamaError::TokenToPiece`] if the C conversion fails.
    pub fn token_to_piece(
        &self,
        token: LlamaToken,
        decoder: &mut encoding_rs::Decoder,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<String, LlamaError> {
        let bytes = self.token_to_piece_bytes(token, 8, special, lstrip)?;
        let mut out = String::with_capacity(bytes.len() + 16);
        let _ = decoder.decode_to_string(&bytes, &mut out, false);
        Ok(out)
    }

    /// Convert a single token to its text piece, lossily (UTF-8 replacement on
    /// invalid bytes). Convenience over [`Self::token_to_piece`] for callers not
    /// doing incremental streaming.
    ///
    /// # Errors
    ///
    /// [`LlamaError::TokenToPiece`] if the C conversion fails.
    pub fn token_to_piece_lossy(&self, token: LlamaToken) -> Result<String, LlamaError> {
        let bytes = self.token_to_piece_bytes(token, 8, false, None)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Detokenize a slice of tokens into a string, decoding incrementally through
    /// a single UTF-8 decoder (`special = false`).
    ///
    /// Note: this concatenates per-token pieces, so special tokens are dropped and
    /// spacing may differ slightly from a true detokenizer. ik only exposes the
    /// vocab-pointer `llama_detokenize` (the model-pointer overload is commented
    /// out in the header); a full detokenizer path is a follow-up.
    pub fn detokenize(&self, tokens: &[LlamaToken]) -> Result<String, LlamaError> {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut s = String::new();
        for &t in tokens {
            s.push_str(&self.token_to_piece(t, &mut decoder, false, None)?);
        }
        Ok(s)
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { sys::llama_free_model(self.model.as_ptr()) };
    }
}
