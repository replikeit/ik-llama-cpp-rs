//! Safe wrapper around `llama_token_data`.
use crate::token::LlamaToken;

use ik_llama_cpp_sys as sys;

/// A transparent wrapper around `llama_token_data` (a token id + its logit and
/// probability).
///
/// Do not rely on `repr(transparent)` for this type — it is an implementation
/// detail (it lets `[LlamaTokenData]` be reinterpreted as `[llama_token_data]`
/// for the C sampling calls) and may change.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct LlamaTokenData {
    data: sys::llama_token_data,
}

impl LlamaTokenData {
    /// Create a new token data from a token, logit, and probability.
    #[must_use]
    pub fn new(LlamaToken(id): LlamaToken, logit: f32, p: f32) -> Self {
        LlamaTokenData {
            data: sys::llama_token_data { id, logit, p },
        }
    }

    /// The token's id.
    #[must_use]
    pub fn id(&self) -> LlamaToken {
        LlamaToken(self.data.id)
    }

    /// The token's logit.
    #[must_use]
    pub fn logit(&self) -> f32 {
        self.data.logit
    }

    /// The token's probability.
    #[must_use]
    pub fn p(&self) -> f32 {
        self.data.p
    }

    /// Set the token's id.
    pub fn set_id(&mut self, id: LlamaToken) {
        self.data.id = id.0;
    }

    /// Set the token's logit.
    pub fn set_logit(&mut self, logit: f32) {
        self.data.logit = logit;
    }

    /// Set the token's probability.
    pub fn set_p(&mut self, p: f32) {
        self.data.p = p;
    }
}
