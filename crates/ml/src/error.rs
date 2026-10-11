//! The error type of the local-model seam. Every failure is a value the caller can show: a
//! missing runtime, a bad model file or a model that misbehaves never panics.

/// Why a local model could not be loaded or run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MlError {
    /// This build or platform has no way to run the model (feature off, wasm).
    #[error("{0}")]
    Unavailable(String),
    /// The ONNX Runtime library could not be found or loaded.
    #[error("ONNX Runtime library: {0}")]
    Runtime(String),
    /// The model file is missing, unreadable, does not match its pinned hash or is not usable.
    #[error("model: {0}")]
    Model(String),
    /// The inputs are not something the model can take (size, buffer lengths).
    #[error("{0}")]
    BadInput(String),
    /// The caller cancelled between stages.
    #[error("cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, MlError>;
