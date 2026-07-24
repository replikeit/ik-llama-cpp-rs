//! LoRA adapters (`llama_lora_adapter_*`).
//!
//! Mirrors `llama-cpp-2`'s LoRA API ([`crate::model::LlamaModel::lora_adapter_init`]
//! and the `LlamaContext::lora_adapter_*` methods), adapted for ik_llama.cpp's
//! older C API: ik exposes per-adapter `llama_lora_adapter_set` / `_remove` /
//! `_clear` over an `llama_lora_adapter` handle, whereas modern stock llama.cpp
//! replaced these with `llama_set_adapters_lora` over an `llama_adapter_lora`
//! type.

use std::ffi::{CString, NulError};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use ik_llama_cpp_sys as sys;

/// A safe wrapper around `llama_lora_adapter`.
///
/// The adapter is initialized from a GGUF file via
/// [`crate::model::LlamaModel::lora_adapter_init`].
///
/// # Ownership
///
/// In ik_llama.cpp the adapter is **owned by the model**: it registers itself
/// with the model on init and is freed automatically when the model is freed
/// (`llama.h`: "will be freed when the model is deleted"). This type therefore
/// has **no `Drop`** — freeing it here would double-free (once here, once by the
/// model).
///
/// The `'model` lifetime ties the adapter to the [`crate::model::LlamaModel`] it
/// was built from, so the compiler enforces that the model outlives the adapter
/// (using an adapter after its model drops would be a use-after-free, since the
/// model frees it). This is stricter than the `llama-cpp-2` anchor, whose
/// adapter carries no lifetime — a deliberate divergence for compile-time safety.
/// Do not use an adapter after `lora_adapter_clear`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaLoraAdapter<'model> {
    pub(crate) lora_adapter: NonNull<sys::llama_lora_adapter>,
    pub(crate) _model: PhantomData<&'model crate::model::LlamaModel>,
}

/// An error that can occur when initializing a lora adapter.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterInitError {
    /// There was a null byte in a provided string and thus it could not be converted to a C string.
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    /// llama.cpp returned a nullptr - this could be many different causes.
    #[error("null result from llama cpp")]
    NullResult,
    /// Failed to convert the path to a rust str. This means the path was not valid unicode.
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

/// An error that can occur when setting a lora adapter on a context.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterSetError {
    /// llama.cpp returned a non-zero error code.
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}

/// An error that can occur when removing a lora adapter from a context.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterRemoveError {
    /// llama.cpp returned a non-zero error code.
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}

impl crate::model::LlamaModel {
    /// Initializes a lora adapter from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterInitError`] for more information.
    pub fn lora_adapter_init(
        &self,
        path: &Path,
    ) -> Result<LlamaLoraAdapter<'_>, LlamaLoraAdapterInitError> {
        debug_assert!(path.exists(), "{path:?} does not exist");

        let path_str = path
            .to_str()
            .ok_or_else(|| LlamaLoraAdapterInitError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path_str)?;
        // SAFETY: `self.model` is a valid, non-null model pointer and `cstr` is a
        // valid NUL-terminated C string that outlives the call.
        let adapter = unsafe { sys::llama_lora_adapter_init(self.model.as_ptr(), cstr.as_ptr()) };

        let adapter = NonNull::new(adapter).ok_or(LlamaLoraAdapterInitError::NullResult)?;

        tracing::debug!(?path, "Initialized lora adapter");
        // The returned adapter borrows `&self` (the model), so the model is
        // guaranteed to outlive it — the model owns and frees the adapter.
        Ok(LlamaLoraAdapter {
            lora_adapter: adapter,
            _model: PhantomData,
        })
    }
}

// NOTE: intentionally NO `impl Drop` — the model owns and frees the adapter
// (see the `LlamaLoraAdapter` "Ownership" docs). A `Drop` calling
// `llama_lora_adapter_free` here would double-free (the model frees it too) and
// could UAF if the adapter were still set on a live context.

impl crate::context::LlamaContext<'_> {
    /// Sets a lora adapter on this context with the given scale.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterSetError`] for more information.
    pub fn lora_adapter_set(
        &mut self,
        adapter: &LlamaLoraAdapter<'_>,
        scale: f32,
    ) -> Result<(), LlamaLoraAdapterSetError> {
        // SAFETY: both the context and adapter pointers are valid and non-null
        // for the duration of the call.
        let err_code = unsafe {
            sys::llama_lora_adapter_set(self.context.as_ptr(), adapter.lora_adapter.as_ptr(), scale)
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterSetError::ErrorResult(err_code));
        }

        tracing::debug!(scale, "Set lora adapter");
        Ok(())
    }

    /// Removes a specific lora adapter from this context.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterRemoveError`] for more information.
    pub fn lora_adapter_remove(
        &mut self,
        adapter: &LlamaLoraAdapter<'_>,
    ) -> Result<(), LlamaLoraAdapterRemoveError> {
        // SAFETY: both the context and adapter pointers are valid and non-null
        // for the duration of the call.
        let err_code = unsafe {
            sys::llama_lora_adapter_remove(self.context.as_ptr(), adapter.lora_adapter.as_ptr())
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterRemoveError::ErrorResult(err_code));
        }

        tracing::debug!("Removed lora adapter");
        Ok(())
    }

    /// Removes all lora adapters from this context.
    pub fn lora_adapter_clear(&mut self) {
        // SAFETY: `self.context` is a valid, non-null context pointer.
        unsafe { sys::llama_lora_adapter_clear(self.context.as_ptr()) }
        tracing::debug!("Cleared lora adapters");
    }
}
