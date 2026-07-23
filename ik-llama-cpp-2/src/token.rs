//! Token newtype.

pub mod data;
pub mod data_array;

/// A single vocabulary token id (newtype over ik_llama.cpp's `llama_token` = `i32`).
///
/// `#[repr(transparent)]` guarantees identical layout to `llama_token`, so
/// `*const llama_token` ↔ `*const LlamaToken` casts (e.g. in mtmd `text_tokens`)
/// are sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct LlamaToken(pub ik_llama_cpp_sys::llama_token);

impl LlamaToken {
    /// Construct from a raw token id.
    #[must_use]
    pub fn new(id: ik_llama_cpp_sys::llama_token) -> Self {
        Self(id)
    }

    /// The raw token id.
    #[must_use]
    pub fn raw(self) -> ik_llama_cpp_sys::llama_token {
        self.0
    }
}
