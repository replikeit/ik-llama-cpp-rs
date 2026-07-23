//! Safe wrapper around multimodal (MTMD) functionality in `ik_llama.cpp`.
//!
//! This module provides Rust bindings for ik_llama.cpp's multimodal support,
//! allowing processing of text, image, and audio inputs through a unified
//! interface. Mirrors `llama-cpp-2`'s `mtmd` module, adapted to ik's divergent
//! C API (see the `D*` notes below).
//!
//! # Warning
//! This API is experimental and subject to breaking changes.
//!
//! # ik deltas vs. the `llama-cpp-2` anchor
//! * D1 `mtmd_decode_use_non_causal(ctx)` is 1-arg in ik (no chunk arg).
//! * D2 ik exposes `mtmd_get_audio_bitrate` (the anchor's name is
//!   `mtmd_get_audio_sample_rate`); [`MtmdContext::get_audio_sample_rate`] is a
//!   thin alias of [`MtmdContext::get_audio_bitrate`].
//! * D3 `mtmd_helper_bitmap_init_from_file(ctx, fname)` is 2-arg (no `placeholder`).
//! * D4 `mtmd_helper_bitmap_init_from_buf(ctx, buf, len)` is 3-arg (no `placeholder`).
//! * D6 `mtmd_context_params` has extra fields (`verbosity`, `image_marker`,
//!   `flash_attn_type`, `kq_type`) — we build from `mtmd_context_params_default()`
//!   and override only the fields we expose.
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::slice;

use ik_llama_cpp_sys as sys;

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::token::LlamaToken;

/// Input chunk types for multimodal data
///
/// # Examples
///
/// ```
/// use ik_llama_cpp_2::mtmd::MtmdInputChunkType;
///
/// let text_chunk = MtmdInputChunkType::Text;
/// let image_chunk = MtmdInputChunkType::Image;
/// let audio_chunk = MtmdInputChunkType::Audio;
///
/// assert_eq!(text_chunk, MtmdInputChunkType::Text);
/// assert_eq!(text_chunk, ik_llama_cpp_sys::MTMD_INPUT_CHUNK_TYPE_TEXT.into());
/// assert_ne!(text_chunk, image_chunk);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MtmdInputChunkType {
    /// Text input chunk
    Text = sys::MTMD_INPUT_CHUNK_TYPE_TEXT as _,
    /// Image input chunk
    Image = sys::MTMD_INPUT_CHUNK_TYPE_IMAGE as _,
    /// Audio input chunk
    Audio = sys::MTMD_INPUT_CHUNK_TYPE_AUDIO as _,
}

impl From<sys::mtmd_input_chunk_type> for MtmdInputChunkType {
    fn from(chunk_type: sys::mtmd_input_chunk_type) -> Self {
        match chunk_type {
            sys::MTMD_INPUT_CHUNK_TYPE_TEXT => MtmdInputChunkType::Text,
            sys::MTMD_INPUT_CHUNK_TYPE_IMAGE => MtmdInputChunkType::Image,
            sys::MTMD_INPUT_CHUNK_TYPE_AUDIO => MtmdInputChunkType::Audio,
            _ => panic!("Unknown MTMD input chunk type: {chunk_type}"),
        }
    }
}

/// Configuration parameters for MTMD context
///
/// # Examples
///
/// ```
/// use ik_llama_cpp_2::mtmd::{MtmdContextParams, mtmd_default_marker};
/// use std::ffi::CString;
///
/// let params = MtmdContextParams {
///     use_gpu: false,
///     print_timings: true,
///     n_threads: 4,
///     media_marker: CString::new(mtmd_default_marker()).unwrap(),
///     image_min_tokens: -1,
///     image_max_tokens: -1,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MtmdContextParams {
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Whether to print timing information
    pub print_timings: bool,
    /// Number of threads to use for processing
    pub n_threads: i32,
    /// Media marker string used to identify media positions in text
    pub media_marker: CString,
    /// Minimum number of tokens used to represent an image.
    /// Controls the visual token budget lower bound. Use -1 for the model default.
    pub image_min_tokens: i32,
    /// Maximum number of tokens used to represent an image.
    /// Controls the visual token budget upper bound. Use -1 for the model default.
    pub image_max_tokens: i32,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        unsafe { sys::mtmd_context_params_default() }.into()
    }
}

impl From<&MtmdContextParams> for sys::mtmd_context_params {
    /// D6: start from the ik default (which fills the extra `verbosity` /
    /// `image_marker` / `flash_attn_type` / `kq_type` fields) and override only
    /// the fields this wrapper exposes.
    fn from(params: &MtmdContextParams) -> Self {
        let mut context = unsafe { sys::mtmd_context_params_default() };
        let MtmdContextParams {
            use_gpu,
            print_timings,
            n_threads,
            media_marker,
            image_min_tokens,
            image_max_tokens,
        } = params;

        context.use_gpu = *use_gpu;
        context.print_timings = *print_timings;
        context.n_threads = *n_threads;
        context.media_marker = media_marker.as_ptr();
        context.image_min_tokens = *image_min_tokens;
        context.image_max_tokens = *image_max_tokens;

        context
    }
}

impl From<sys::mtmd_context_params> for MtmdContextParams {
    fn from(params: sys::mtmd_context_params) -> Self {
        Self {
            use_gpu: params.use_gpu,
            print_timings: params.print_timings,
            n_threads: params.n_threads,
            media_marker: unsafe { CStr::from_ptr(params.media_marker) }.to_owned(),
            image_min_tokens: params.image_min_tokens,
            image_max_tokens: params.image_max_tokens,
        }
    }
}

/// Text input configuration
///
/// # Examples
///
/// ```
/// use ik_llama_cpp_2::mtmd::MtmdInputText;
///
/// let input = MtmdInputText {
///     text: "Describe this image.".to_string(),
///     add_special: true,
///     parse_special: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MtmdInputText {
    /// The input text string
    pub text: String,
    /// Whether to add special tokens
    pub add_special: bool,
    /// Whether to parse special tokens
    pub parse_special: bool,
}

/// Safe wrapper around `mtmd_context`.
///
/// This represents an initialized multimodal context that can process
/// text, images, and audio through ik_llama.cpp's multimodal interface.
#[derive(Debug)]
pub struct MtmdContext {
    pub(crate) context: NonNull<sys::mtmd_context>,
}

// MtmdContext is thread safe
unsafe impl Send for MtmdContext {}
unsafe impl Sync for MtmdContext {}

impl MtmdContext {
    /// Initialize MTMD context from a multimodal projection file.
    ///
    /// # Arguments
    ///
    /// * `mmproj_path` - Path to the multimodal projection file
    /// * `text_model` - Reference to the text model
    /// * `params` - Configuration parameters for the MTMD context
    ///
    /// # Errors
    ///
    /// Returns an error if the path cannot be converted to a C string, or if the
    /// underlying C function returns null (indicating initialization failure).
    pub fn init_from_file(
        mmproj_path: &str,
        text_model: &LlamaModel,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdInitError> {
        let path_cstr = CString::new(mmproj_path)?;
        let ctx_params = sys::mtmd_context_params::from(params);

        let context = unsafe {
            sys::mtmd_init_from_file(path_cstr.as_ptr(), text_model.model.as_ptr(), ctx_params)
        };

        let context = NonNull::new(context).ok_or(MtmdInitError::NullResult)?;
        Ok(Self { context })
    }

    /// Check whether non-causal attention mask is needed before `llama_decode`.
    ///
    /// D1: ik's `mtmd_decode_use_non_causal` takes only the context (the anchor's
    /// signature has an extra chunk argument).
    #[must_use]
    pub fn decode_use_non_causal(&self) -> bool {
        unsafe { sys::mtmd_decode_use_non_causal(self.context.as_ptr()) }
    }

    /// Check whether the current model uses M-RoPE for `llama_decode`.
    ///
    /// M-RoPE (Multimodal Rotary Position Embedding) affects how positions
    /// are calculated for multimodal inputs.
    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { sys::mtmd_decode_use_mrope(self.context.as_ptr()) }
    }

    /// Check whether the current model supports vision input.
    #[must_use]
    pub fn support_vision(&self) -> bool {
        unsafe { sys::mtmd_support_vision(self.context.as_ptr()) }
    }

    /// Check whether the current model supports audio input.
    #[must_use]
    pub fn support_audio(&self) -> bool {
        unsafe { sys::mtmd_support_audio(self.context.as_ptr()) }
    }

    /// Get audio bitrate (sample rate) in Hz (e.g., 16000 for Whisper).
    /// Returns `None` if audio is not supported.
    ///
    /// D2: backed by ik's `mtmd_get_audio_bitrate`.
    #[must_use]
    pub fn get_audio_bitrate(&self) -> Option<u32> {
        let rate = unsafe { sys::mtmd_get_audio_bitrate(self.context.as_ptr()) };
        (rate > 0).then_some(rate.unsigned_abs())
    }

    /// Anchor-compatible alias for [`MtmdContext::get_audio_bitrate`] (ik names
    /// the underlying getter `mtmd_get_audio_bitrate`).
    #[must_use]
    pub fn get_audio_sample_rate(&self) -> Option<u32> {
        self.get_audio_bitrate()
    }

    /// Tokenize input text and bitmaps into chunks.
    ///
    /// The input text must contain media markers (default: `<__media__>`) that
    /// will be replaced with the corresponding bitmap data from `bitmaps`. The
    /// number of bitmaps must equal the number of markers in the text.
    ///
    /// # Errors
    ///
    /// * `BitmapCountMismatch` - Number of bitmaps doesn't match number of markers
    /// * `ImagePreprocessingError` - Error occurred during image preprocessing
    /// * `CStringError` - Text contains an interior NUL byte
    /// * `UnknownError` - Other tokenization error occurred
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ik_llama_cpp_2::mtmd::*;
    /// # fn example(ctx: &MtmdContext, bitmap: &MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// let text = MtmdInputText {
    ///     text: "Here is an image: <__media__>\nDescribe it.".to_string(),
    ///     add_special: true,
    ///     parse_special: true,
    /// };
    /// let chunks = ctx.tokenize(text, &[bitmap])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn tokenize(
        &self,
        text: MtmdInputText,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<MtmdInputChunks, MtmdTokenizeError> {
        let chunks = MtmdInputChunks::new();
        let text_cstring = CString::new(text.text)?;
        let input_text = sys::mtmd_input_text {
            text: text_cstring.as_ptr(),
            add_special: text.add_special,
            parse_special: text.parse_special,
        };

        // Create bitmap pointers
        let bitmap_ptrs: Vec<*const sys::mtmd_bitmap> = bitmaps
            .iter()
            .map(|b| b.bitmap.as_ptr().cast_const())
            .collect();

        let result = unsafe {
            sys::mtmd_tokenize(
                self.context.as_ptr(),
                chunks.chunks.as_ptr(),
                &raw const input_text,
                bitmap_ptrs.as_ptr().cast_mut(),
                bitmaps.len(),
            )
        };

        match result {
            0 => Ok(chunks),
            1 => Err(MtmdTokenizeError::BitmapCountMismatch),
            2 => Err(MtmdTokenizeError::ImagePreprocessingError),
            _ => Err(MtmdTokenizeError::UnknownError(result)),
        }
    }

    /// Encode a chunk for image/audio processing.
    ///
    /// Processes image or audio chunks into embeddings usable by the language
    /// model. The embeddings can be retrieved via `mtmd_get_output_embd`.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEncodeError::EncodeFailure` if encoding fails.
    pub fn encode_chunk(&mut self, chunk: &MtmdInputChunk<'_>) -> Result<(), MtmdEncodeError> {
        let result = unsafe { sys::mtmd_encode_chunk(self.context.as_ptr(), chunk.chunk.as_ptr()) };

        if result == 0 {
            Ok(())
        } else {
            Err(MtmdEncodeError::EncodeFailure(result))
        }
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { sys::mtmd_free(self.context.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_bitmap`.
///
/// Represents bitmap data for images or audio that can be processed by the
/// multimodal system. For images, data is stored in RGB format. For audio, data
/// is stored as PCM F32 samples.
// NOTE: intentionally does NOT derive `Clone`. `bitmap` is a `NonNull` owning
// handle freed by `mtmd_bitmap_free` in `Drop`; a derived (shallow) `Clone`
// would produce two owners of the same pointer -> double free. There is no
// `mtmd_bitmap_copy` in the C API to implement a deep clone.
#[derive(Debug)]
pub struct MtmdBitmap {
    pub(crate) bitmap: NonNull<sys::mtmd_bitmap>,
}

// MtmdBitmap is thread safe
unsafe impl Send for MtmdBitmap {}
unsafe impl Sync for MtmdBitmap {}

impl MtmdBitmap {
    /// Create a bitmap from image data in RGB format.
    ///
    /// # Arguments
    ///
    /// * `nx` - Width of the image in pixels
    /// * `ny` - Height of the image in pixels
    /// * `data` - Image data in RGBRGBRGB... format (must be exactly `nx * ny * 3` bytes)
    ///
    /// # Errors
    ///
    /// * `InvalidDataSize` - Data length doesn't match `nx * ny * 3`
    /// * `NullResult` - Underlying C function returned null
    ///
    /// # Examples
    ///
    /// ```
    /// use ik_llama_cpp_2::mtmd::MtmdBitmap;
    ///
    /// // Create a 2x2 red image
    /// let red_pixel = [255, 0, 0]; // RGB values for red
    /// let image_data = red_pixel.repeat(4); // 2x2 = 4 pixels
    ///
    /// let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data);
    /// assert!(bitmap.is_ok());
    /// ```
    pub fn from_image_data(nx: u32, ny: u32, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        // Multiply in `usize` (64-bit on target platforms) to match the C side's
        // `(size_t)nx * ny * 3`. Computing `nx * ny * 3` in `u32` would panic in
        // debug and wrap in release, letting an undersized `data` slip past the
        // check and causing the C `memcpy` to over-read.
        if data.len() != (nx as usize) * (ny as usize) * 3 {
            return Err(MtmdBitmapError::InvalidDataSize);
        }

        let bitmap = unsafe { sys::mtmd_bitmap_init(nx, ny, data.as_ptr()) };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from audio data in PCM F32 format.
    ///
    /// # Errors
    ///
    /// * `NullResult` - Underlying C function returned null
    pub fn from_audio_data(data: &[f32]) -> Result<Self, MtmdBitmapError> {
        let bitmap = unsafe { sys::mtmd_bitmap_init_from_audio(data.len(), data.as_ptr()) };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a file.
    ///
    /// Supported formats:
    /// - Images: formats supported by `stb_image` (jpg, png, bmp, gif, etc.)
    /// - Audio: formats supported by miniaudio (wav, mp3, flac)
    ///
    /// Audio files are auto-detected based on magic bytes.
    ///
    /// D3: ik's `mtmd_helper_bitmap_init_from_file` takes only `(ctx, fname)` —
    /// there is no `placeholder` argument.
    ///
    /// # Errors
    ///
    /// * `CStringError` - Path contains an interior NUL byte
    /// * `NullResult` - File could not be loaded or processed
    ///
    /// This function is thread-safe.
    pub fn from_file(ctx: &MtmdContext, path: &str) -> Result<Self, MtmdBitmapError> {
        let path_cstr = CString::new(path)?;
        let bitmap = unsafe {
            sys::mtmd_helper_bitmap_init_from_file(ctx.context.as_ptr(), path_cstr.as_ptr())
        };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a buffer containing file data.
    ///
    /// Supported formats are the same as [`MtmdBitmap::from_file`]. Audio files
    /// are auto-detected based on magic bytes.
    ///
    /// D4: ik's `mtmd_helper_bitmap_init_from_buf` takes only `(ctx, buf, len)` —
    /// there is no `placeholder` argument.
    ///
    /// # Errors
    ///
    /// * `NullResult` - Buffer could not be processed
    ///
    /// This function is thread-safe.
    pub fn from_buffer(ctx: &MtmdContext, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        let bitmap = unsafe {
            sys::mtmd_helper_bitmap_init_from_buf(ctx.context.as_ptr(), data.as_ptr(), data.len())
        };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Get bitmap width in pixels.
    #[must_use]
    pub fn nx(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_nx(self.bitmap.as_ptr()) }
    }

    /// Get bitmap height in pixels.
    #[must_use]
    pub fn ny(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_ny(self.bitmap.as_ptr()) }
    }

    /// Get bitmap data as a byte slice.
    ///
    /// For images: RGB format with length `nx * ny * 3`.
    /// For audio: PCM F32 bytes with length `n_samples * 4`.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        let ptr = unsafe { sys::mtmd_bitmap_get_data(self.bitmap.as_ptr()) };
        let len = unsafe { sys::mtmd_bitmap_get_n_bytes(self.bitmap.as_ptr()) };
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    /// Check if this bitmap contains audio data (vs image data).
    #[must_use]
    pub fn is_audio(&self) -> bool {
        unsafe { sys::mtmd_bitmap_is_audio(self.bitmap.as_ptr()) }
    }

    /// Get the bitmap's optional ID string.
    ///
    /// Bitmap ID is useful for KV cache tracking and can e.g. be calculated
    /// based on a hash of the bitmap data.
    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { sys::mtmd_bitmap_get_id(self.bitmap.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            let id = unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned();
            Some(id)
        }
    }

    /// Set the bitmap's ID string.
    ///
    /// # Errors
    ///
    /// Returns an error if the ID string contains an interior NUL byte.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use ik_llama_cpp_2::mtmd::MtmdBitmap;
    /// # fn example(bitmap: &MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// bitmap.set_id("image_001")?;
    /// assert_eq!(bitmap.id(), Some("image_001".to_string()));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_id(&self, id: &str) -> Result<(), std::ffi::NulError> {
        let id_cstr = CString::new(id)?;
        unsafe {
            sys::mtmd_bitmap_set_id(self.bitmap.as_ptr(), id_cstr.as_ptr());
        }
        Ok(())
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        unsafe { sys::mtmd_bitmap_free(self.bitmap.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunks`.
///
/// A collection of input chunks created from tokenizing text and media. Text
/// chunks contain tokens; media chunks contain embeddings.
#[derive(Debug)]
pub struct MtmdInputChunks {
    pub(crate) chunks: NonNull<sys::mtmd_input_chunks>,
}

impl Default for MtmdInputChunks {
    fn default() -> Self {
        Self::new()
    }
}

impl MtmdInputChunks {
    /// Create a new empty input chunks collection.
    ///
    /// # Panics
    ///
    /// Panics only if the underlying `mtmd_input_chunks_init` returns null, which
    /// indicates allocation failure and should not happen in practice.
    ///
    /// # Examples
    ///
    /// ```
    /// use ik_llama_cpp_2::mtmd::MtmdInputChunks;
    ///
    /// let chunks = MtmdInputChunks::new();
    /// assert_eq!(chunks.len(), 0);
    /// assert!(chunks.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        let chunks = unsafe { sys::mtmd_input_chunks_init() };
        let chunks = NonNull::new(chunks).expect("mtmd_input_chunks_init returned null");
        Self { chunks }
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { sys::mtmd_input_chunks_size(self.chunks.as_ptr()) }
    }

    /// Check if the chunks collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a chunk by index.
    ///
    /// The returned [`MtmdInputChunk`] borrows this collection: the chunk's
    /// pointer is owned by the `MtmdInputChunks` (freed when the collection is
    /// dropped), so the returned handle is tied to `&self` via its `'a` lifetime
    /// and cannot outlive the collection.
    #[must_use]
    pub fn get<'a>(&'a self, index: usize) -> Option<MtmdInputChunk<'a>> {
        if index >= self.len() {
            return None;
        }

        let chunk_ptr = unsafe { sys::mtmd_input_chunks_get(self.chunks.as_ptr(), index) };

        // Note: We don't own this chunk, it's owned by the chunks collection.
        NonNull::new(chunk_ptr.cast_mut()).map(|ptr| MtmdInputChunk {
            chunk: ptr,
            owned: false,
            _marker: PhantomData,
        })
    }

    /// Get total number of tokens across all chunks (useful for KV cache size).
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        unsafe { sys::mtmd_helper_get_n_tokens(self.chunks.as_ptr()) }
    }

    /// Get total position count across all chunks (useful for `n_past`).
    ///
    /// Normally `n_pos` equals `n_tokens`, but for M-RoPE it differs.
    #[must_use]
    pub fn total_positions(&self) -> i32 {
        unsafe { sys::mtmd_helper_get_n_pos(self.chunks.as_ptr()) }
    }

    /// Evaluate chunks using the multimodal context and LLAMA context.
    ///
    /// This helper automatically:
    /// 1. Runs `llama_decode()` on text chunks.
    /// 2. Runs `mtmd_encode()` on image chunks, then `mtmd_get_output_embd()`,
    ///    then `llama_decode()`.
    ///
    /// If any `mtmd_encode()` or `llama_decode()` call returns non-zero, the
    /// function stops and forwards the error.
    ///
    /// Returns the new `n_past` value on success.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEvalError::EvalFailure` if any encoding or decoding fails.
    ///
    /// This function is NOT thread-safe.
    pub fn eval_chunks(
        &mut self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &LlamaContext<'_>,
        n_past: sys::llama_pos,
        seq_id: sys::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<sys::llama_pos, MtmdEvalError> {
        let mut new_n_past: sys::llama_pos = 0;

        let result = unsafe {
            sys::mtmd_helper_eval_chunks(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunks.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &raw mut new_n_past,
            )
        };

        if result == 0 {
            Ok(new_n_past)
        } else {
            Err(MtmdEvalError::EvalFailure(result))
        }
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        unsafe { sys::mtmd_input_chunks_free(self.chunks.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunk`.
///
/// Represents a single chunk of input data — text tokens, image tokens, or audio
/// tokens. The chunk type determines what data and operations are available.
///
/// The `'a` lifetime ties a **borrowed** chunk (from [`MtmdInputChunks::get`]) to
/// the collection that owns its backing memory, so the chunk cannot outlive the
/// collection (which would free the underlying `mtmd_input_chunk` out from under
/// it). An **owned** chunk from [`MtmdInputChunk::copy`] carries its own
/// allocation and is `MtmdInputChunk<'static>`.
#[derive(Debug)]
pub struct MtmdInputChunk<'a> {
    pub(crate) chunk: NonNull<sys::mtmd_input_chunk>,
    /// Whether this handle owns the underlying chunk (only owned copies are freed
    /// on drop; borrowed chunks are owned by their `MtmdInputChunks`).
    owned: bool,
    /// Ties a borrowed chunk to the lifetime of its owning `MtmdInputChunks`.
    _marker: PhantomData<&'a MtmdInputChunks>,
}

impl<'a> MtmdInputChunk<'a> {
    /// Get the type of this chunk.
    #[must_use]
    pub fn chunk_type(&self) -> MtmdInputChunkType {
        let chunk_type = unsafe { sys::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::from(chunk_type)
    }

    /// Get text tokens from this chunk.
    ///
    /// Only valid for text chunks. Returns `None` for image or audio chunks.
    ///
    /// The returned slice borrows `self`, so it cannot outlive this chunk handle.
    /// For a borrowed chunk that borrow is itself bounded by the chunk's `'a`
    /// (the owning collection), and for an owned chunk it is bounded by the
    /// owned allocation — so the slice can never outlive its backing buffer.
    #[must_use]
    pub fn text_tokens(&self) -> Option<&[LlamaToken]> {
        if self.chunk_type() != MtmdInputChunkType::Text {
            return None;
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            sys::mtmd_input_chunk_get_tokens_text(self.chunk.as_ptr(), &raw mut n_tokens)
        };

        if tokens_ptr.is_null() || n_tokens == 0 {
            None
        } else {
            // LlamaToken is `#[repr(transparent)]` over `llama_token` (i32);
            // reinterpret the token buffer in place (matches the `llama-cpp-2`
            // anchor). The slice lifetime is tied to `&self` (see above).
            unsafe {
                Some(slice::from_raw_parts(
                    tokens_ptr.cast::<LlamaToken>(),
                    n_tokens,
                ))
            }
        }
    }

    /// Get the number of tokens in this chunk.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    /// Get the number of positions in this chunk.
    ///
    /// Returns the number of temporal positions (always 1 for M-RoPE, `n_tokens`
    /// otherwise).
    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { sys::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    /// Get chunk ID if available.
    ///
    /// Returns `None` for text chunks, may return an ID for image/audio chunks.
    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { sys::mtmd_input_chunk_get_id(self.chunk.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
                .into()
        }
    }

    /// Create a copy of this chunk that you own.
    ///
    /// Useful if you want custom logic to handle the chunk (e.g. KV cache
    /// management) by moving ownership into your own code. The returned copy is
    /// freed when it is dropped.
    ///
    /// # Errors
    ///
    /// Returns `MtmdInputChunkError::NullResult` if copying fails.
    pub fn copy(&self) -> Result<MtmdInputChunk<'static>, MtmdInputChunkError> {
        let chunk = unsafe { sys::mtmd_input_chunk_copy(self.chunk.as_ptr()) };
        let chunk = NonNull::new(chunk).ok_or(MtmdInputChunkError::NullResult)?;
        // Owned copy: it carries its own allocation (freed on drop) and borrows
        // nothing, so it is `'static`.
        Ok(MtmdInputChunk {
            chunk,
            owned: true,
            _marker: PhantomData,
        })
    }
}

impl Drop for MtmdInputChunk<'_> {
    fn drop(&mut self) {
        if self.owned {
            unsafe { sys::mtmd_input_chunk_free(self.chunk.as_ptr()) }
        }
    }
}

/// Get the default media marker string.
///
/// Returns the default marker used to identify media positions in text
/// (typically `"<__media__>"`). Use this marker in your input text to indicate
/// where media content should be inserted.
///
/// # Examples
///
/// ```
/// use ik_llama_cpp_2::mtmd::mtmd_default_marker;
///
/// let marker = mtmd_default_marker();
/// assert!(!marker.is_empty());
///
/// let text = format!("Describe this image: {}", marker);
/// assert!(text.contains(marker));
/// ```
#[must_use]
pub fn mtmd_default_marker() -> &'static str {
    unsafe {
        let c_str = sys::mtmd_default_marker();
        CStr::from_ptr(c_str).to_str().unwrap_or("<__media__>")
    }
}

// Error types

/// Errors that can occur when initializing MTMD context.
#[derive(thiserror::Error, Debug)]
pub enum MtmdInitError {
    /// Failed to create `CString` from input.
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// MTMD context initialization returned null.
    #[error("MTMD context initialization returned null")]
    NullResult,
}

/// Errors that can occur when working with MTMD bitmaps.
#[derive(thiserror::Error, Debug)]
pub enum MtmdBitmapError {
    /// Failed to create `CString` from input.
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// Invalid data size for bitmap.
    #[error("Invalid data size for bitmap")]
    InvalidDataSize,
    /// Bitmap creation returned null.
    #[error("Bitmap creation returned null")]
    NullResult,
}

/// Errors that can occur when working with MTMD input chunk collections.
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunksError {
    /// Input chunks creation returned null.
    #[error("Input chunks creation returned null")]
    NullResult,
}

/// Errors that can occur when working with individual MTMD input chunks.
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunkError {
    /// Input chunk operation returned null.
    #[error("Input chunk operation returned null")]
    NullResult,
}

/// Errors that can occur during tokenization.
#[derive(thiserror::Error, Debug)]
pub enum MtmdTokenizeError {
    /// Number of bitmaps does not match number of markers in text.
    #[error("Number of bitmaps does not match number of markers")]
    BitmapCountMismatch,
    /// Image preprocessing error occurred.
    #[error("Image preprocessing error")]
    ImagePreprocessingError,
    /// Text contains characters that cannot be converted to a C string.
    #[error("Failed to create CString from text: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// Unknown error occurred during tokenization.
    #[error("Unknown error: {0}")]
    UnknownError(i32),
}

/// Errors that can occur during encoding.
#[derive(thiserror::Error, Debug)]
pub enum MtmdEncodeError {
    /// Encode operation failed.
    #[error("Encode failed with code: {0}")]
    EncodeFailure(i32),
}

/// Errors that can occur during evaluation.
#[derive(thiserror::Error, Debug)]
pub enum MtmdEvalError {
    /// Evaluation operation failed.
    #[error("Eval failed with code: {0}")]
    EvalFailure(i32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_marker_is_non_empty() {
        let marker = mtmd_default_marker();
        assert!(
            !marker.is_empty(),
            "default media marker should be non-empty"
        );
    }

    #[test]
    fn bitmap_from_image_data_valid_size_is_ok() {
        // 2x2 RGB image = 2 * 2 * 3 = 12 bytes.
        let bitmap = MtmdBitmap::from_image_data(2, 2, &[0u8; 12]);
        assert!(bitmap.is_ok(), "expected Ok for correctly-sized image data");
    }

    #[test]
    fn bitmap_from_image_data_wrong_size_errors() {
        // 5 bytes != 2 * 2 * 3 = 12 -> InvalidDataSize.
        let bitmap = MtmdBitmap::from_image_data(2, 2, &[0u8; 5]);
        assert!(matches!(bitmap, Err(MtmdBitmapError::InvalidDataSize)));
    }
}
