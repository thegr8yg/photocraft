//! Local machine-learning models for PhotoCraft.
//!
//! The first user is Content-Aware Fill with **LaMa** (large-mask inpainting): the engine hands
//! this crate the pixels around a selection and gets them back filled in. Everything here runs
//! on the user's own computer; nothing is downloaded, uploaded or bundled.
//!
//! - [`inpaint`]: the model contract ([`InpaintModel`]) and the pipeline that chooses the window,
//!   scales it to the model's size, runs the model and writes back only the hole. Pure Rust, tested
//!   with a stub model, builds for wasm.
//! - [`manifest`]: what we know about each model (id, source, licence, optional pinned hash) and
//!   how a model file is hashed and checked.
//! - [`onnx`] (feature `onnx`, native only): [`OrtLama`], the ONNX Runtime backend. `ort` loads
//!   the runtime library at run time, so building needs no download and a missing library is an
//!   error message, not a crash.
//!
//! Layer L4. Depends on no other workspace crate: callers convert their pixels to display RGB.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod error;
pub mod inpaint;
pub mod manifest;
#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
pub mod onnx;

pub use error::{MlError, Result};
pub use inpaint::{InpaintModel, Options, Output, Region, inpaint};
pub use manifest::{FileInfo, ModelSpec};
#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
pub use onnx::{Device, OrtLama};

/// Is the ONNX Runtime backend compiled into this build?
pub const fn onnx_compiled() -> bool {
    cfg!(all(feature = "onnx", not(target_arch = "wasm32")))
}
