//! The ONNX Runtime backend: LaMa (or any two-input inpainting model with an `image` and a `mask`)
//! as an [`InpaintModel`].
//!
//! `ort` is built with `load-dynamic`: nothing is linked or downloaded when PhotoCraft is built.
//! The user installs ONNX Runtime (a release archive, a package manager, `pip`'s
//! `onnxruntime/capi/`) and points Preferences at `libonnxruntime.so` / `onnxruntime.dll` /
//! `libonnxruntime.dylib`, or sets `PHOTOCRAFT_ORT_LIB`. A GPU build of the library (CUDA,
//! DirectML, CoreML) is used automatically when present; otherwise the CPU runs the model.
//!
//! `ort` panics in a few places on a hostile environment (a library that is not ONNX Runtime, or
//! a version it cannot talk to). Every call into it here is inside `catch_unwind` and becomes an
//! [`MlError::Runtime`], so a wrong library path is a message, never a crash.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use ort::ep::{self, ExecutionProvider};
use ort::session::Session;
use ort::value::{Shape, Tensor, TensorElementType};

use crate::error::{MlError, Result};
use crate::inpaint::InpaintModel;
use crate::manifest;

/// Which hardware the model may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Device {
    /// A GPU provider if the runtime library has one, else the CPU.
    #[default]
    Auto,
    /// Always the CPU.
    Cpu,
}

impl Device {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// The library the process loaded (ONNX Runtime can be loaded once per process). Only a success
/// is remembered: a wrong path can be corrected without restarting.
static LOADED: Mutex<Option<PathBuf>> = Mutex::new(None);

fn default_library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// Where to look for the runtime: the explicit path, `PHOTOCRAFT_ORT_LIB`, `ORT_DYLIB_PATH`, then
/// the platform's library name (found by the system loader's search path).
fn library_candidate(explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit.filter(|p| !p.as_os_str().is_empty()) {
        return p.to_path_buf();
    }
    for var in ["PHOTOCRAFT_ORT_LIB", "ORT_DYLIB_PATH"] {
        if let Some(v) = std::env::var_os(var).filter(|v| !v.is_empty()) {
            return PathBuf::from(v);
        }
    }
    PathBuf::from(default_library_name())
}

/// Load the runtime library; once it is loaded, later calls return its path. Returns the library path.
pub fn load_runtime(explicit: Option<&Path>) -> Result<PathBuf> {
    let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
    let asked = explicit.filter(|p| !p.as_os_str().is_empty());
    if let Some(p) = loaded.as_ref() {
        return match asked {
            Some(a) if a != p.as_path() => {
                Err(MlError::Runtime(format!("ONNX Runtime is already loaded from `{}`; restart PhotoCraft to use another library", p.display())))
            }
            _ => Ok(p.clone()),
        };
    }
    let wanted = library_candidate(explicit);
    let hint = "Install ONNX Runtime and set its library in Preferences › Local Models (or PHOTOCRAFT_ORT_LIB)";
    if wanted.components().count() > 1 && !wanted.is_file() {
        return Err(MlError::Runtime(format!("`{}` does not exist. {hint}", wanted.display())));
    }
    let attempt = catch_unwind(AssertUnwindSafe(|| match ort::init_from(&wanted) {
        Ok(builder) => {
            builder.commit();
            Ok(())
        }
        Err(e) => Err(e.to_string()),
    }));
    match attempt {
        Ok(Ok(())) => {
            *loaded = Some(wanted.clone());
            Ok(wanted)
        }
        Ok(Err(e)) => Err(MlError::Runtime(format!("could not load `{}`: {e}. {hint}", wanted.display()))),
        Err(_) => Err(MlError::Runtime(format!("`{}` is not a compatible ONNX Runtime library. {hint}", wanted.display()))),
    }
}

/// An inpainting model running in ONNX Runtime.
pub struct OrtLama {
    session: Mutex<Session>,
    size: usize,
    image_input: String,
    mask_input: String,
    label: String,
    /// What the model runs on, for the status line.
    pub device: String,
}

impl std::fmt::Debug for OrtLama {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtLama").field("label", &self.label).field("size", &self.size).field("device", &self.device).finish()
    }
}

/// Which GPU providers the loaded library offers, best first.
fn gpu_providers() -> Vec<(&'static str, ort::ep::ExecutionProviderDispatch)> {
    let mut v = Vec::new();
    if ep::CUDA::default().is_available().unwrap_or(false) {
        v.push(("CUDA", ep::CUDA::default().build()));
    }
    if ep::DirectML::default().is_available().unwrap_or(false) {
        v.push(("DirectML", ep::DirectML::default().build()));
    }
    if ep::CoreML::default().is_available().unwrap_or(false) {
        v.push(("CoreML", ep::CoreML::default().build()));
    }
    v
}

fn square_side(shape: &Shape) -> Result<usize> {
    let dims: Vec<i64> = shape.iter().copied().collect();
    let (h, w) = match dims.as_slice() {
        [_, _, h, w] => (*h, *w),
        _ => return Err(MlError::Model("the image input is not a 4-D tensor (batch, channels, height, width)".into())),
    };
    // A dynamic dimension (-1) means the export takes any size; use the size LaMa was trained at.
    let side = |d: i64| if d < 0 { Some(512usize) } else { usize::try_from(d).ok().filter(|d| *d > 0) };
    match (side(h), side(w)) {
        (Some(h), Some(w)) if h == w => Ok(h),
        _ => Err(MlError::Model(format!("the model works at {h}×{w}; only square models are supported"))),
    }
}

impl OrtLama {
    /// Load `model` (an ONNX file) with the runtime library at `runtime` (or the default search).
    pub fn load(runtime: Option<&Path>, model: &Path, device: Device) -> Result<Self> {
        load_runtime(runtime)?;
        let info = manifest::hash_file(model)?;
        let _ = info;
        let inner = catch_unwind(AssertUnwindSafe(|| Self::build(model, device)));
        match inner {
            Ok(r) => r,
            Err(_) => Err(MlError::Runtime("ONNX Runtime failed while loading the model (it is probably too old for this model; try a newer runtime)".into())),
        }
    }

    fn build(model: &Path, device: Device) -> Result<Self> {
        let gpus = if device == Device::Auto { gpu_providers() } else { Vec::new() };
        let names: Vec<&str> = gpus.iter().map(|(n, _)| *n).collect();
        let mut builder = Session::builder().map_err(|e| MlError::Runtime(e.to_string()))?;
        if !gpus.is_empty() {
            let eps: Vec<_> = gpus.iter().map(|(_, e)| e.clone()).collect();
            builder = builder.with_execution_providers(eps).map_err(|e| MlError::Runtime(e.to_string()))?;
        }
        let session = builder.commit_from_file(model).map_err(|e| MlError::Model(format!("cannot load `{}`: {e}", model.display())))?;

        let inputs = session.inputs();
        if inputs.len() != 2 {
            return Err(MlError::Model(format!("expected an image and a mask input, the model has {}", inputs.len())));
        }
        // Prefer the input named like a mask; otherwise the second one is the mask.
        let mask_idx = inputs.iter().position(|i| i.name().to_ascii_lowercase().contains("mask")).unwrap_or(1);
        let image_idx = 1 - mask_idx;
        let image = inputs.get(image_idx).ok_or_else(|| MlError::Model("the model has no image input".into()))?;
        let mask = inputs.get(mask_idx).ok_or_else(|| MlError::Model("the model has no mask input".into()))?;
        for i in [image, mask] {
            if i.dtype().tensor_type() != Some(TensorElementType::Float32) {
                return Err(MlError::Model(format!(
                    "input `{}` is not 32-bit float (half-precision models are not supported yet; use the fp32 export)",
                    i.name()
                )));
            }
        }
        let shape = image.dtype().tensor_shape().ok_or_else(|| MlError::Model("the image input is not a tensor".into()))?;
        let size = square_side(shape)?;
        let label = model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
        let device = if names.is_empty() { "CPU".to_string() } else { format!("{} (falls back to CPU)", names.join(" / ")) };
        Ok(Self { size, image_input: image.name().to_string(), mask_input: mask.name().to_string(), label, device, session: Mutex::new(session) })
    }

    fn infer(&self, rgb: &[f32], mask: &[f32]) -> Result<Vec<f32>> {
        let n = self.size;
        let plane = n * n;
        // Interleaved HWC -> planar CHW.
        let mut chw = vec![0.0f32; plane * 3];
        for (i, px) in rgb.chunks_exact(3).enumerate() {
            for (c, v) in px.iter().enumerate() {
                if let Some(d) = chw.get_mut(c * plane + i) {
                    *d = *v;
                }
            }
        }
        let dim = n as i64;
        let image = Tensor::from_array((Shape::new([1, 3, dim, dim]), chw)).map_err(|e| MlError::Runtime(e.to_string()))?;
        let mask_t = Tensor::from_array((Shape::new([1, 1, dim, dim]), mask.to_vec())).map_err(|e| MlError::Runtime(e.to_string()))?;
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs = session
            .run(ort::inputs![self.image_input.as_str() => image, self.mask_input.as_str() => mask_t])
            .map_err(|e| MlError::Model(format!("the model failed: {e}")))?;
        let first = outputs.iter().next().ok_or_else(|| MlError::Model("the model returned nothing".into()))?.1;
        let (shape, data) = first.try_extract_tensor::<f32>().map_err(|e| MlError::Model(format!("unexpected model output: {e}")))?;
        if data.len() != plane * 3 {
            return Err(MlError::Model(format!("the model returned {} values, expected {} for a {n}×{n} image", data.len(), plane * 3)));
        }
        // Exports differ: planar (1,3,H,W) or interleaved (1,H,W,3); 0..1 or 0..255.
        let planar = shape.last().copied() != Some(3);
        let peak = data.iter().copied().fold(0.0f32, |m, v| if v.is_finite() { m.max(v) } else { m });
        let unit = if peak > 1.5 { 255.0 } else { 1.0 };
        let mut out = vec![0.0f32; plane * 3];
        for i in 0..plane {
            for c in 0..3 {
                let src = if planar { c * plane + i } else { i * 3 + c };
                if let (Some(d), Some(s)) = (out.get_mut(i * 3 + c), data.get(src)) {
                    *d = *s / unit;
                }
            }
        }
        Ok(out)
    }
}

impl InpaintModel for OrtLama {
    fn name(&self) -> String {
        format!("{} on {}", self.label, self.device)
    }
    fn size(&self) -> usize {
        self.size
    }
    fn run(&self, rgb: &[f32], mask: &[f32]) -> Result<Vec<f32>> {
        let plane = self.size * self.size;
        if rgb.len() != plane * 3 || mask.len() != plane {
            return Err(MlError::BadInput("buffer sizes do not match the model".into()));
        }
        match catch_unwind(AssertUnwindSafe(|| self.infer(rgb, mask))) {
            Ok(r) => r,
            Err(_) => Err(MlError::Runtime("ONNX Runtime failed while running the model".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_runtime_is_an_error_not_a_panic() {
        let r = OrtLama::load(Some(Path::new("/nonexistent/dir/libonnxruntime.so")), Path::new("/nonexistent/lama.onnx"), Device::Cpu);
        assert!(matches!(r, Err(MlError::Runtime(_))), "{r:?}");
    }

    #[test]
    fn the_device_names_parse() {
        assert_eq!(Device::parse("auto"), Some(Device::Auto));
        assert_eq!(Device::parse("cpu"), Some(Device::Cpu));
        assert_eq!(Device::parse("gpu"), None);
    }

    #[test]
    fn model_sizes_come_from_the_input_shape() {
        assert_eq!(square_side(&Shape::new([1, 3, 512, 512])).ok(), Some(512));
        assert_eq!(square_side(&Shape::new([1, 3, -1, -1])).ok(), Some(512));
        assert!(square_side(&Shape::new([1, 3, 256, 512])).is_err());
        assert!(square_side(&Shape::new([3, 512, 512])).is_err());
        assert!(square_side(&Shape::new([1, 3, 0, 0])).is_err());
    }
}
