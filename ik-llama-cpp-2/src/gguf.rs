//! Safe wrapper around `gguf_context` for reading GGUF file metadata.
//!
//! Provides metadata-only access to GGUF files without loading tensor data.
//! Useful for inspecting model architecture parameters before loading a model.
//!
//! Mirrors `llama-cpp-2`'s `gguf` module, adapted for ik_llama.cpp's C API:
//! ik's `gguf_*` functions index key-value pairs and tensors with `int`
//! (`i32`), whereas upstream llama.cpp migrated these to `int64_t`.

use std::ffi::{CStr, CString, NulError};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

/// Errors returned when opening a GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    /// The path contained an interior NUL byte and could not be converted to a
    /// C string.
    #[error("gguf path contained an interior NUL byte")]
    Nul(#[from] NulError),
    /// The path was not valid UTF-8 and could not be converted to a C string.
    #[error("gguf path was not valid UTF-8")]
    InvalidPath,
    /// `gguf_init_from_file` returned null (missing file or not a valid GGUF).
    #[error("failed to open or parse GGUF file: {0}")]
    Init(PathBuf),
}

/// A safe wrapper around `gguf_context`.
///
/// Opens a GGUF file and parses only the metadata header; tensor weights are
/// never loaded into memory (`no_alloc = true`).
///
/// # Aborts
///
/// The typed value getters (`val_u32`, `val_str`, `arr_str`, …) and index-based
/// accessors call ik functions guarded by `GGML_ASSERT`: passing an
/// out-of-range index (`>= n_kv()` / `>= n_tensors()`) or reading a value with
/// the wrong type aborts the **process**. Validate first via [`Self::n_kv`],
/// [`Self::find_key`], and [`Self::kv_type`] before calling a typed getter.
#[derive(Debug)]
pub struct GgufContext {
    ctx: NonNull<ik_llama_cpp_sys::gguf_context>,
}

impl GgufContext {
    /// Open a GGUF file and parse its metadata header.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] if the path is not valid UTF-8, contains a NUL
    /// byte, or the file does not exist / is not a valid GGUF file.
    pub fn from_file(path: &Path) -> Result<Self, GgufError> {
        let path_str = path.to_str().ok_or(GgufError::InvalidPath)?;
        let c_path = CString::new(path_str)?;
        let params = ik_llama_cpp_sys::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: `c_path` is a valid NUL-terminated C string that outlives the
        // call, and `params` is a valid, fully-initialized struct.
        let ptr = unsafe { ik_llama_cpp_sys::gguf_init_from_file(c_path.as_ptr(), params) };
        let ctx = NonNull::new(ptr).ok_or_else(|| GgufError::Init(path.to_path_buf()))?;
        Ok(Self { ctx })
    }

    /// The GGUF format version of the file.
    #[must_use]
    pub fn version(&self) -> i32 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_version(self.ctx.as_ptr()) }
    }

    /// The tensor-data alignment (in bytes) declared by the file.
    #[must_use]
    pub fn alignment(&self) -> usize {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_alignment(self.ctx.as_ptr()) }
    }

    /// Total number of key-value pairs in the metadata.
    #[must_use]
    pub fn n_kv(&self) -> i32 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_n_kv(self.ctx.as_ptr()) }
    }

    /// Find the index of a key by name. Returns `-1` if not found.
    #[must_use]
    pub fn find_key(&self, key: &str) -> i32 {
        let Ok(c_key) = CString::new(key) else {
            return -1;
        };
        // SAFETY: `self.ctx` is valid; `c_key` is a valid NUL-terminated string
        // that outlives the call.
        unsafe { ik_llama_cpp_sys::gguf_find_key(self.ctx.as_ptr(), c_key.as_ptr()) }
    }

    /// Return the key name at the given index, or `None` if the pointer is null
    /// or not valid UTF-8.
    #[must_use]
    pub fn key_at(&self, idx: i32) -> Option<&str> {
        // SAFETY: `self.ctx` is valid; the returned pointer (if non-null) points
        // to a NUL-terminated string owned by the context.
        let ptr = unsafe { ik_llama_cpp_sys::gguf_get_key(self.ctx.as_ptr(), idx) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and NUL-terminated; it lives as long as
        // `self`, which bounds the returned `&str`.
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Return the value type of the KV pair at `idx`.
    #[must_use]
    pub fn kv_type(&self, idx: i32) -> ik_llama_cpp_sys::gguf_type {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_kv_type(self.ctx.as_ptr(), idx) }
    }

    /// Return the element type of the array KV pair at `idx`.
    ///
    /// Only meaningful when [`kv_type`](Self::kv_type) is `GGUF_TYPE_ARRAY`.
    #[must_use]
    pub fn arr_type(&self, idx: i32) -> ik_llama_cpp_sys::gguf_type {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_arr_type(self.ctx.as_ptr(), idx) }
    }

    /// Read a `uint8` value. Aborts (inside ik_llama.cpp) if the stored type is
    /// not `GGUF_TYPE_UINT8` — check [`kv_type`](Self::kv_type) first if unsure.
    #[must_use]
    pub fn val_u8(&self, idx: i32) -> u8 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_u8(self.ctx.as_ptr(), idx) }
    }

    /// Read an `int8` value.
    #[must_use]
    pub fn val_i8(&self, idx: i32) -> i8 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_i8(self.ctx.as_ptr(), idx) }
    }

    /// Read a `uint16` value.
    #[must_use]
    pub fn val_u16(&self, idx: i32) -> u16 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_u16(self.ctx.as_ptr(), idx) }
    }

    /// Read an `int16` value.
    #[must_use]
    pub fn val_i16(&self, idx: i32) -> i16 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_i16(self.ctx.as_ptr(), idx) }
    }

    /// Read a `uint32` value. Aborts (inside ik_llama.cpp) if the stored type is
    /// not `GGUF_TYPE_UINT32` — check [`kv_type`](Self::kv_type) first if unsure.
    #[must_use]
    pub fn val_u32(&self, idx: i32) -> u32 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_u32(self.ctx.as_ptr(), idx) }
    }

    /// Read an `int32` value.
    #[must_use]
    pub fn val_i32(&self, idx: i32) -> i32 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_i32(self.ctx.as_ptr(), idx) }
    }

    /// Read a `float32` value.
    #[must_use]
    pub fn val_f32(&self, idx: i32) -> f32 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_f32(self.ctx.as_ptr(), idx) }
    }

    /// Read a `uint64` value.
    #[must_use]
    pub fn val_u64(&self, idx: i32) -> u64 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_u64(self.ctx.as_ptr(), idx) }
    }

    /// Read an `int64` value.
    #[must_use]
    pub fn val_i64(&self, idx: i32) -> i64 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_i64(self.ctx.as_ptr(), idx) }
    }

    /// Read a `float64` value.
    #[must_use]
    pub fn val_f64(&self, idx: i32) -> f64 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_f64(self.ctx.as_ptr(), idx) }
    }

    /// Read a `bool` value.
    #[must_use]
    pub fn val_bool(&self, idx: i32) -> bool {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_val_bool(self.ctx.as_ptr(), idx) }
    }

    /// Read a string value. Returns `None` if the pointer is null or not valid
    /// UTF-8.
    #[must_use]
    pub fn val_str(&self, idx: i32) -> Option<&str> {
        // SAFETY: `self.ctx` is valid; the returned pointer (if non-null) points
        // to a NUL-terminated string owned by the context.
        let ptr = unsafe { ik_llama_cpp_sys::gguf_get_val_str(self.ctx.as_ptr(), idx) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and NUL-terminated; it lives as long as
        // `self`, which bounds the returned `&str`.
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Number of elements in the array KV pair at `idx`.
    ///
    /// Only meaningful when [`kv_type`](Self::kv_type) is `GGUF_TYPE_ARRAY`.
    #[must_use]
    pub fn arr_len(&self, idx: i32) -> i32 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_arr_n(self.ctx.as_ptr(), idx) }
    }

    /// Read the `i`-th string of a string-array KV pair.
    ///
    /// Returns `None` if the pointer is null or not valid UTF-8. Only valid
    /// when [`arr_type`](Self::arr_type) is `GGUF_TYPE_STRING`.
    #[must_use]
    pub fn arr_str(&self, idx: i32, i: i32) -> Option<&str> {
        // SAFETY: `self.ctx` is valid; the returned pointer (if non-null) points
        // to a NUL-terminated string owned by the context.
        let ptr = unsafe { ik_llama_cpp_sys::gguf_get_arr_str(self.ctx.as_ptr(), idx, i) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and NUL-terminated; it lives as long as
        // `self`, which bounds the returned `&str`.
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Total number of tensors described in the file.
    #[must_use]
    pub fn n_tensors(&self) -> i32 {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_n_tensors(self.ctx.as_ptr()) }
    }

    /// Return the name of the tensor at index `i`, or `None` if the pointer is
    /// null or not valid UTF-8.
    #[must_use]
    pub fn tensor_name(&self, i: i32) -> Option<&str> {
        // SAFETY: `self.ctx` is valid; the returned pointer (if non-null) points
        // to a NUL-terminated string owned by the context.
        let ptr = unsafe { ik_llama_cpp_sys::gguf_get_tensor_name(self.ctx.as_ptr(), i) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and NUL-terminated; it lives as long as
        // `self`, which bounds the returned `&str`.
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Return the [`ggml_type`](ik_llama_cpp_sys::ggml_type) of the tensor at
    /// index `i`.
    #[must_use]
    pub fn tensor_type(&self, i: i32) -> ik_llama_cpp_sys::ggml_type {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_tensor_type(self.ctx.as_ptr(), i) }
    }

    /// Return the data offset (in bytes, relative to the start of the tensor
    /// data section) of the tensor at index `i`.
    #[must_use]
    pub fn tensor_offset(&self, i: i32) -> usize {
        // SAFETY: `self.ctx` is a valid, non-null context for its lifetime.
        unsafe { ik_llama_cpp_sys::gguf_get_tensor_offset(self.ctx.as_ptr(), i) }
    }
}

/// Return the human-readable name of a [`gguf_type`](ik_llama_cpp_sys::gguf_type)
/// (e.g. `"u32"`, `"str"`, `"arr"`), or `None` if the pointer is null or not
/// valid UTF-8.
#[must_use]
pub fn type_name(kind: ik_llama_cpp_sys::gguf_type) -> Option<&'static str> {
    // SAFETY: `gguf_type_name` returns a pointer to a static NUL-terminated
    // string (or null) for any input; it borrows nothing from a context.
    let ptr = unsafe { ik_llama_cpp_sys::gguf_type_name(kind) };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `ptr` is non-null and points to a `'static` NUL-terminated string.
    unsafe { CStr::from_ptr(ptr).to_str().ok() }
}

impl Drop for GgufContext {
    fn drop(&mut self) {
        // SAFETY: `self.ctx` was returned by `gguf_init_from_file` and has not
        // been freed; it is freed exactly once here.
        unsafe { ik_llama_cpp_sys::gguf_free(self.ctx.as_ptr()) }
    }
}

#[cfg(all(test, feature = "_smoke"))]
mod tests {
    use super::*;

    /// Reads `general.architecture` from the GGUF pointed to by `IK_TEST_MODEL`
    /// and asserts it is `qwen35`. Skips (does not fail) when the env var is
    /// unset so the suite stays green without a model on disk.
    #[test]
    fn reads_architecture() {
        let Ok(model) = std::env::var("IK_TEST_MODEL") else {
            eprintln!("IK_TEST_MODEL not set; skipping gguf smoke test");
            return;
        };
        let ctx = GgufContext::from_file(Path::new(&model)).expect("open gguf");
        assert!(ctx.n_kv() > 0, "expected at least one KV pair");

        let idx = ctx.find_key("general.architecture");
        assert!(idx >= 0, "general.architecture key missing");
        assert_eq!(ctx.val_str(idx), Some("qwen35"));
    }
}
