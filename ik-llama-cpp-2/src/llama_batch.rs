//! Token batch for decode ([`LlamaBatch`]) over ik's `llama_batch`.

use ik_llama_cpp_sys as sys;

use crate::token::LlamaToken;
use crate::LlamaError;

/// A batch of tokens to feed to [`crate::LlamaContext::decode`].
///
/// Wraps `llama_batch_init`/`llama_batch_free`. ik's `llama_batch` has legacy
/// `all_pos_*`/`all_seq_id` tail fields (zeroed by `llama_batch_init`); we use
/// the explicit per-token `pos`/`seq_id`/`logits` arrays.
#[derive(Debug)]
pub struct LlamaBatch {
    batch: sys::llama_batch,
    capacity: usize,
    n_seq_max: usize,
}

impl LlamaBatch {
    /// Allocate a batch holding up to `capacity` tokens, each assignable to up to
    /// `n_seq_max` sequences.
    #[must_use]
    pub fn new(capacity: usize, n_seq_max: usize) -> Self {
        assert!(
            capacity <= i32::MAX as usize && n_seq_max <= i32::MAX as usize,
            "batch dims exceed i32::MAX (capacity={capacity}, n_seq_max={n_seq_max})"
        );
        // SAFETY: allocates owned arrays sized for `capacity`/`n_seq_max`.
        let batch = unsafe { sys::llama_batch_init(capacity as i32, 0, n_seq_max as i32) };
        Self {
            batch,
            capacity,
            n_seq_max,
        }
    }

    /// Reset the batch to empty (does not free).
    pub fn clear(&mut self) {
        self.batch.n_tokens = 0;
    }

    /// Number of tokens currently in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.batch.n_tokens
    }

    /// Add one token at position `pos` for the given sequence ids; `logits` marks
    /// whether logits should be computed for it.
    pub fn add(
        &mut self,
        token: LlamaToken,
        pos: i32,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), LlamaError> {
        let i = self.batch.n_tokens as usize;
        if i >= self.capacity {
            return Err(LlamaError::BatchOverflow {
                capacity: self.capacity,
                index: i,
            });
        }
        if seq_ids.len() > self.n_seq_max {
            return Err(LlamaError::TooManySeqIds {
                got: seq_ids.len(),
                max: self.n_seq_max,
            });
        }
        // SAFETY: `i` is within capacity; arrays were sized by llama_batch_init.
        unsafe {
            *self.batch.token.add(i) = token.0;
            *self.batch.pos.add(i) = pos;
            *self.batch.n_seq_id.add(i) = seq_ids.len() as i32;
            let seq_row = *self.batch.seq_id.add(i);
            for (j, &s) in seq_ids.iter().enumerate() {
                *seq_row.add(j) = s;
            }
            *self.batch.logits.add(i) = i8::from(logits);
        }
        self.batch.n_tokens += 1;
        Ok(())
    }

    /// Append a whole prompt for a single sequence at positions `0..len`.
    /// `logits_all` requests logits for every token; otherwise only the last.
    ///
    /// Positions always start at 0 — this primes a **fresh** sequence. To append
    /// a continuation to an existing sequence, use [`Self::add`] with explicit
    /// positions.
    pub fn add_sequence(
        &mut self,
        tokens: &[LlamaToken],
        seq_id: i32,
        logits_all: bool,
    ) -> Result<(), LlamaError> {
        let n = tokens.len();
        for (i, &t) in tokens.iter().enumerate() {
            let want_logits = logits_all || i + 1 == n;
            self.add(t, i as i32, &[seq_id], want_logits)?;
        }
        Ok(())
    }

    /// Raw batch, by value (for `llama_decode`).
    pub(crate) fn as_raw(&self) -> sys::llama_batch {
        self.batch
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        unsafe { sys::llama_batch_free(self.batch) };
    }
}
