//! Chat templates (`llama_chat_apply_template`).
//!
//! Mirrors the `llama-cpp-2` anchor's chat-template API ([`LlamaChatMessage`],
//! [`LlamaChatTemplate`], [`LlamaModel::chat_template`],
//! [`LlamaModel::apply_chat_template`]), adapted to ik_llama.cpp's C signatures.
//!
//! [`LlamaModel`]: crate::model::LlamaModel

use std::ffi::{CStr, CString, NulError};
use std::os::raw::c_char;
use std::str::Utf8Error;
use std::string::FromUtf8Error;

use ik_llama_cpp_sys as sys;

/// A performance-friendly wrapper around [`LlamaModel::chat_template`] which is then
/// fed into [`LlamaModel::apply_chat_template`] to convert a list of messages into an LLM
/// prompt. Internally the template is stored as a `CString` to avoid round-trip conversions
/// within the FFI.
///
/// [`LlamaModel::chat_template`]: crate::model::LlamaModel::chat_template
/// [`LlamaModel::apply_chat_template`]: crate::model::LlamaModel::apply_chat_template
#[derive(Eq, PartialEq, Clone, PartialOrd, Ord, Hash)]
pub struct LlamaChatTemplate(CString);

impl LlamaChatTemplate {
    /// Create a new template from a string. This can either be the name of a built-in
    /// chat template like "chatml" or "llama3" or an actual Jinja template for
    /// ik_llama.cpp to interpret.
    ///
    /// # Errors
    /// If `template` contains a null byte and thus cannot be converted to a C string.
    pub fn new(template: &str) -> Result<Self, NulError> {
        Ok(Self(CString::new(template)?))
    }

    /// Accesses the template as a c string reference.
    #[must_use]
    pub fn as_c_str(&self) -> &CStr {
        &self.0
    }

    /// Attempts to convert the `CString` into a Rust str reference.
    ///
    /// # Errors
    /// If the template is not valid UTF-8.
    pub fn to_str(&self) -> Result<&str, Utf8Error> {
        self.0.to_str()
    }

    /// Convenience method to create an owned String.
    ///
    /// # Errors
    /// If the template is not valid UTF-8.
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        self.to_str().map(str::to_string)
    }
}

impl std::fmt::Debug for LlamaChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A safe wrapper around `llama_chat_message`.
///
/// Owns its `role` and `content` as `CString`s so the borrowed pointers handed to
/// the FFI stay valid for the duration of the call.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    role: CString,
    content: CString,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`.
    ///
    /// # Errors
    /// If either of `role` or `content` contain null bytes.
    pub fn new(role: String, content: String) -> Result<Self, NewLlamaChatMessageError> {
        Ok(Self {
            role: CString::new(role)?,
            content: CString::new(content)?,
        })
    }
}

/// Failed to construct a [`LlamaChatMessage`].
#[derive(Debug, thiserror::Error)]
pub enum NewLlamaChatMessageError {
    /// The string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
}

/// Failed to apply a chat template (see [`LlamaModel::apply_chat_template`]).
///
/// [`LlamaModel::apply_chat_template`]: crate::model::LlamaModel::apply_chat_template
#[derive(Debug, thiserror::Error)]
pub enum ApplyChatTemplateError {
    /// The string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
    /// The formatted prompt could not be converted to utf8.
    #[error("{0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    /// `llama_chat_apply_template` returned an error code.
    #[error("ffi error {0}")]
    FfiError(i32),
}

impl crate::model::LlamaModel {
    /// Get the chat template from the model by name. If the `name` parameter is `None`,
    /// the model's default chat template will be returned.
    ///
    /// You supply this into [`Self::apply_chat_template`] to get back a string with the
    /// appropriate template substitution applied to convert a list of messages into a
    /// prompt the LLM can use to complete the chat.
    ///
    /// Returns `None` if the model has no chat template by that name (or if `name`
    /// contains an interior null byte).
    ///
    /// You could also use an external jinja parser, like [minijinja](https://github.com/mitsuhiko/minijinja),
    /// to parse jinja templates not supported by the ik_llama.cpp template engine.
    #[must_use]
    pub fn chat_template(&self, name: Option<&str>) -> Option<LlamaChatTemplate> {
        // Keep the CString alive for the duration of the FFI call.
        let name_cstr = match name {
            Some(name) => match CString::new(name) {
                Ok(name) => Some(name),
                Err(_) => return None,
            },
            None => None,
        };
        let name_ptr = name_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        // SAFETY: `self.model` is a valid model pointer; `name_ptr` is either null or a
        // valid, NUL-terminated C string.
        let result = unsafe { sys::llama_model_chat_template(self.model.as_ptr(), name_ptr) };
        if result.is_null() {
            return None;
        }

        // SAFETY: `result` is a valid, NUL-terminated C string owned by the model.
        let bytes = unsafe { CStr::from_ptr(result) }.to_bytes();
        // `bytes` has no interior NUL (it came from a C string), so this cannot fail;
        // `.ok()` keeps this `unwrap`-free.
        CString::new(bytes).ok().map(LlamaChatTemplate)
    }

    /// Apply a chat template to a list of messages, returning the formatted prompt.
    ///
    /// Inspired by hf `apply_chat_template()` on python. Use [`Self::chat_template`] to
    /// retrieve the template baked into the model (this is the preferred path), or build
    /// one directly with [`LlamaChatTemplate::new`].
    ///
    /// `add_assistant` controls whether the prompt ends with the token(s) that indicate
    /// the start of an assistant message.
    ///
    /// # Errors
    /// There are many ways this can fail. See [`ApplyChatTemplateError`] for more information.
    pub fn apply_chat_template(
        &self,
        tmpl: &LlamaChatTemplate,
        chat: &[LlamaChatMessage],
        add_assistant: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        // Build the ik_llama_cpp_sys chat messages. These borrow the `CString`s owned by
        // `chat`, which outlive every FFI call below.
        let c_chat: Vec<sys::llama_chat_message> = chat
            .iter()
            .map(|c| sys::llama_chat_message {
                role: c.role.as_ptr(),
                content: c.content.as_ptr(),
            })
            .collect();
        let n_msg = c_chat.len();
        let tmpl_ptr = tmpl.0.as_ptr();

        // Recommended starting size is 2 * total message bytes (per the C API docs).
        let hint = chat
            .iter()
            .fold(0usize, |acc, c| {
                acc + c.role.as_bytes().len() + c.content.as_bytes().len()
            })
            .saturating_mul(2);
        let mut cap = i32::try_from(hint).unwrap_or(i32::MAX);
        let mut buf = vec![0u8; cap.max(0) as usize];

        // SAFETY: `buf` holds `cap` bytes and `c_chat` holds `n_msg` messages, all
        // borrowed pointers stay valid across the call.
        let mut n = unsafe {
            sys::llama_chat_apply_template(
                tmpl_ptr,
                c_chat.as_ptr(),
                n_msg,
                add_assistant,
                buf.as_mut_ptr().cast::<c_char>(),
                cap,
            )
        };
        if n < 0 {
            return Err(ApplyChatTemplateError::FfiError(n));
        }

        // ik returns the total required length; if it exceeds the buffer, re-alloc and
        // re-apply (two-call buffer sizing, like tokenize).
        if n > cap {
            cap = n;
            buf = vec![0u8; cap as usize];
            // SAFETY: as above, now with a buffer sized to the required length.
            n = unsafe {
                sys::llama_chat_apply_template(
                    tmpl_ptr,
                    c_chat.as_ptr(),
                    n_msg,
                    add_assistant,
                    buf.as_mut_ptr().cast::<c_char>(),
                    cap,
                )
            };
            if n < 0 {
                return Err(ApplyChatTemplateError::FfiError(n));
            }
        }

        buf.truncate(n as usize);
        Ok(String::from_utf8(buf)?)
    }
}
