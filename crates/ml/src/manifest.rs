//! What PhotoCraft knows about a downloadable model, and how a model file is checked.
//!
//! Model files are never committed or bundled. The user points Preferences at a file they
//! downloaded; this module hashes it so the status line can show what is loaded and, when the user
//! recorded the expected SHA-256 (or the manifest carries a pin), refuse anything else.

use crate::error::{MlError, Result};

/// One model PhotoCraft can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// Stable id (`lama`), used in preferences and command params.
    pub id: &'static str,
    pub label: &'static str,
    /// Where the user can get a compatible ONNX export. Informational; PhotoCraft never fetches it.
    pub source: &'static str,
    /// The model's own licence (check it before use; it is not PhotoCraft's).
    pub license: &'static str,
    /// Exact size in bytes, when pinned.
    pub size: Option<u64>,
    /// Pinned SHA-256 (lower-case hex), when known. `None` = accept any file, show its hash.
    pub sha256: Option<&'static str>,
}

/// LaMa ("Resolution-robust Large Mask Inpainting with Fourier Convolutions", Suvorov et al.,
/// WACV 2022), exported to ONNX at a fixed square size (512 in the common exports).
///
/// No hash is pinned: exports differ (fp32/fp16, opset) and none has been verified by this
/// project yet. Set `localModels.lamaSha256` to pin the one you use.
pub const LAMA: ModelSpec = ModelSpec {
    id: "lama",
    label: "LaMa inpainting",
    source: "https://huggingface.co/Carve/LaMa-ONNX",
    license: "Apache-2.0 (LaMa, Samsung AI Center); check the export you use",
    size: None,
    sha256: None,
};

/// Every model the seam knows.
pub const MODELS: &[ModelSpec] = &[LAMA];

/// Look a model up by id.
pub fn find(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

/// A model file's measured size and hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileInfo {
    pub size: u64,
    /// Lower-case hex SHA-256.
    pub sha256: String,
}

/// Refuse to hash anything bigger than this (a mistaken path to a disk image, say).
pub const MAX_MODEL_BYTES: u64 = 4 << 30;

/// Hash a model file (streamed, so a 200 MB model does not need 200 MB of memory).
#[cfg(not(target_arch = "wasm32"))]
pub fn hash_file(path: &std::path::Path) -> Result<FileInfo> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| MlError::Model(format!("cannot open `{}`: {e}", path.display())))?;
    let meta = f.metadata().map_err(|e| MlError::Model(format!("cannot read `{}`: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(MlError::Model(format!("`{}` is not a file", path.display())));
    }
    if meta.len() == 0 {
        return Err(MlError::Model(format!("`{}` is empty", path.display())));
    }
    if meta.len() > MAX_MODEL_BYTES {
        return Err(MlError::Model(format!("`{}` is too large to be a model ({} bytes)", path.display(), meta.len())));
    }
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf).map_err(|e| MlError::Model(format!("cannot read `{}`: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        let chunk = buf.get(..n).ok_or_else(|| MlError::Model("short read".into()))?;
        hasher.update(chunk);
        total = total.saturating_add(n as u64);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest.iter() {
        hex.push_str(&format!("{b:02x}"));
    }
    Ok(FileInfo { size: total, sha256: hex })
}

#[cfg(target_arch = "wasm32")]
pub fn hash_file(_: &std::path::Path) -> Result<FileInfo> {
    Err(MlError::Unavailable("local models run in the desktop app only".into()))
}

/// Check a measured file against what is expected (the manifest's pin, then the user's).
/// `expected_sha256` is compared case-insensitively and ignoring surrounding whitespace; an empty
/// string means "no pin".
pub fn check(spec: &ModelSpec, info: &FileInfo, expected_sha256: &str) -> Result<()> {
    if let Some(size) = spec.size
        && size != info.size
    {
        return Err(MlError::Model(format!("{} has {} bytes, expected {size}", spec.label, info.size)));
    }
    let pins = [spec.sha256.unwrap_or(""), expected_sha256];
    for pin in pins.iter().map(|p| p.trim()).filter(|p| !p.is_empty()) {
        if !pin.eq_ignore_ascii_case(&info.sha256) {
            return Err(MlError::Model(format!("{} does not match its pinned SHA-256 (file {}, expected {pin})", spec.label, info.sha256)));
        }
    }
    Ok(())
}
