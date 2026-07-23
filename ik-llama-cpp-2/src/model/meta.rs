//! Model metadata + dimension accessors.
//!
//! Mirrors the `llama-cpp-2` anchor's metadata/dim accessors on `LlamaModel`,
//! adapted to ik's C API. ik exposes no head-count accessor (`llama_n_head` /
//! `llama_model_n_head` are absent), so that getter is omitted.

use std::os::raw::c_char;

use ik_llama_cpp_sys as sys;

use crate::model::LlamaModel;

/// Read a C string produced by an `snprintf`-style FFI getter using the
/// two-call size-then-fill pattern. Returns `None` if the getter reports the
/// item is absent (negative return) or the bytes are not valid UTF-8.
fn read_ffi_string<F: Fn(*mut c_char, usize) -> i32>(fill: F) -> Option<String> {
    // Probe the required length (snprintf semantics: null buf, size 0).
    let needed = fill(std::ptr::null_mut(), 0);
    if needed < 0 {
        return None;
    }
    let cap = needed as usize + 1;
    let mut buf = vec![0u8; cap];
    let written = fill(buf.as_mut_ptr().cast::<c_char>(), cap);
    if written < 0 {
        return None;
    }
    buf.truncate(written as usize);
    String::from_utf8(buf).ok()
}

impl LlamaModel {
    /// Value of a metadata key (e.g. `"general.architecture"`), or `None` if absent.
    #[must_use]
    pub fn meta_val_str(&self, key: &str) -> Option<String> {
        let c_key = std::ffi::CString::new(key).ok()?;
        let model = self.model.as_ptr();
        // SAFETY: valid model + NUL-terminated key; buf/size honored by the getter.
        read_ffi_string(|buf, size| unsafe {
            sys::llama_model_meta_val_str(model, c_key.as_ptr(), buf, size)
        })
    }

    /// Number of metadata key-value pairs.
    #[must_use]
    pub fn meta_count(&self) -> i32 {
        unsafe { sys::llama_model_meta_count(self.model.as_ptr()) }
    }

    /// Metadata key name at `index`, or `None` if out of range.
    #[must_use]
    pub fn meta_key_by_index(&self, index: i32) -> Option<String> {
        let model = self.model.as_ptr();
        // SAFETY: valid model; buf/size honored by the getter.
        read_ffi_string(|buf, size| unsafe {
            sys::llama_model_meta_key_by_index(model, index, buf, size)
        })
    }

    /// Metadata value at `index`, or `None` if out of range.
    #[must_use]
    pub fn meta_val_str_by_index(&self, index: i32) -> Option<String> {
        let model = self.model.as_ptr();
        // SAFETY: valid model; buf/size honored by the getter.
        read_ffi_string(|buf, size| unsafe {
            sys::llama_model_meta_val_str_by_index(model, index, buf, size)
        })
    }

    /// Human-readable model description.
    #[must_use]
    pub fn desc(&self) -> String {
        let model = self.model.as_ptr();
        // SAFETY: valid model; buf/size honored by the getter.
        read_ffi_string(|buf, size| unsafe { sys::llama_model_desc(model, buf, size) })
            .unwrap_or_default()
    }

    /// Total model size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        unsafe { sys::llama_model_size(self.model.as_ptr()) }
    }

    /// Total number of parameters.
    #[must_use]
    pub fn n_params(&self) -> u64 {
        unsafe { sys::llama_model_n_params(self.model.as_ptr()) }
    }

    /// Training context length.
    #[must_use]
    pub fn n_ctx_train(&self) -> i32 {
        unsafe { sys::llama_n_ctx_train(self.model.as_ptr()) }
    }

    /// Embedding dimension.
    #[must_use]
    pub fn n_embd(&self) -> i32 {
        unsafe { sys::llama_model_n_embd(self.model.as_ptr()) }
    }

    /// Number of layers.
    #[must_use]
    pub fn n_layer(&self) -> i32 {
        unsafe { sys::llama_n_layer(self.model.as_ptr()) }
    }

    /// RoPE type (raw `llama_rope_type`).
    #[must_use]
    pub fn rope_type(&self) -> sys::llama_rope_type {
        unsafe { sys::llama_rope_type(self.model.as_ptr()) }
    }
}
