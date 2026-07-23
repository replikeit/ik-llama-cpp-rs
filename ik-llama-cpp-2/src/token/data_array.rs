//! A rusty equivalent of `llama_token_data_array`.
use std::ptr;

use ik_llama_cpp_sys as sys;

use crate::context::LlamaContext;
use crate::token::data::LlamaTokenData;
use crate::token::LlamaToken;

/// A safe wrapper around `llama_token_data_array` — the candidate set consumed
/// (and mutated) by sampling.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaTokenDataArray {
    /// The underlying candidate data.
    pub data: Vec<LlamaTokenData>,
    /// The index of the selected token in `data`, if a sampler has chosen one.
    pub selected: Option<usize>,
    /// Whether `data` is sorted by logit (descending).
    pub sorted: bool,
}

impl LlamaTokenDataArray {
    /// Create a new `LlamaTokenDataArray` from a vector and whether the data is
    /// sorted by logit.
    #[must_use]
    pub fn new(data: Vec<LlamaTokenData>, sorted: bool) -> Self {
        Self {
            data,
            selected: None,
            sorted,
        }
    }

    /// Create a new `LlamaTokenDataArray` from an iterator of [`LlamaTokenData`].
    pub fn from_iter<T>(data: T, sorted: bool) -> LlamaTokenDataArray
    where
        T: IntoIterator<Item = LlamaTokenData>,
    {
        Self::new(data.into_iter().collect(), sorted)
    }

    /// Build from a logits slice (`id` = index). Probabilities start at 0 and
    /// the data is unsorted.
    #[must_use]
    pub fn from_logits(logits: &[f32]) -> Self {
        let data = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| LlamaTokenData::new(LlamaToken(i as sys::llama_token), logit, 0.0))
            .collect();
        Self::new(data, false)
    }

    /// The currently selected token, if a sampler has chosen one.
    #[must_use]
    pub fn selected_token(&self) -> Option<LlamaToken> {
        self.data.get(self.selected?).map(LlamaTokenData::id)
    }

    /// Apply repetition / frequency / presence penalties in place over the most
    /// recent `last_tokens`.
    ///
    /// Wraps ik's legacy `llama_sample_repetition_penalties`. An empty
    /// `last_tokens` slice is a no-op. Logits are mutated in place; candidate
    /// order is preserved.
    pub fn apply_repetition_penalties(
        &mut self,
        ctx: &mut LlamaContext,
        last_tokens: &[LlamaToken],
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) {
        let raw: Vec<sys::llama_token> = last_tokens.iter().map(|t| t.0).collect();
        let ctx_ptr = ctx.as_ptr();
        // SAFETY: valid ctx; `c` describes `self.data` (mutated in place through
        // its pointer); `raw` lives for the call.
        unsafe {
            self.modify_as_c_llama_token_data_array(|c| {
                sys::llama_sample_repetition_penalties(
                    ctx_ptr,
                    c,
                    raw.as_ptr(),
                    raw.len(),
                    penalty_repeat,
                    penalty_freq,
                    penalty_present,
                );
            });
        }
    }

    /// Build a `llama_token_data_array` snapshot pointing at `self.data`.
    ///
    /// The result borrows `self.data`'s buffer; it is valid only while `self` is
    /// not moved/reallocated. C sampling calls that only rewrite logits in place
    /// (no resize) may use this directly; anything that can resize / reselect
    /// must go through [`Self::modify_as_c_llama_token_data_array`].
    pub(crate) fn as_c(&mut self) -> sys::llama_token_data_array {
        sys::llama_token_data_array {
            data: self.data.as_mut_ptr().cast::<sys::llama_token_data>(),
            size: self.data.len(),
            selected: self.selected.and_then(|s| s.try_into().ok()).unwrap_or(-1),
            sorted: self.sorted,
        }
    }

    /// Expose `self` as a C `llama_token_data_array` for the duration of
    /// `modify`, then sync `size` / `sorted` / `selected` back.
    ///
    /// # Safety
    ///
    /// `modify` must leave the array's `data`/`size` describing initialized
    /// token data within this buffer's capacity, and must set `sorted` honestly.
    pub(crate) unsafe fn modify_as_c_llama_token_data_array<T>(
        &mut self,
        modify: impl FnOnce(&mut sys::llama_token_data_array) -> T,
    ) -> T {
        let data = self.data.as_mut_ptr().cast::<sys::llama_token_data>();
        let mut c = sys::llama_token_data_array {
            data,
            size: self.data.len(),
            selected: self.selected.and_then(|s| s.try_into().ok()).unwrap_or(-1),
            sorted: self.sorted,
        };

        let result = modify(&mut c);

        assert!(
            c.size <= self.data.capacity(),
            "sampler returned an array larger than the candidate buffer capacity"
        );
        if !ptr::eq(c.data, data) {
            ptr::copy(c.data, data, c.size);
        }
        // SAFETY: `modify` guarantees `0..c.size` is initialized (see contract).
        self.data.set_len(c.size);
        self.sorted = c.sorted;
        self.selected = c.selected.try_into().ok().filter(|&s| s < self.data.len());
        result
    }
}
