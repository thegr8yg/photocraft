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
use ort::session::{Session, builder::GraphOptimizationLevel};
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

/// How much graph rewriting ONNX Runtime does when the model loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OptLevel {
    Off,
    /// Constant folding and redundant-node removal.
    Basic,
    /// Plus node fusions that stay portable across hardware.
    Extended,
    /// Plus layout and hardware specific rewrites (ONNX Runtime's own default).
    #[default]
    All,
}

impl OptLevel {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "off" => Some(Self::Off),
            "basic" => Some(Self::Basic),
            "extended" => Some(Self::Extended),
            "all" => Some(Self::All),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Basic => "basic",
            Self::Extended => "extended",
            Self::All => "all",
        }
    }
    fn ort(self) -> GraphOptimizationLevel {
        match self {
            Self::Off => GraphOptimizationLevel::Disable,
            Self::Basic => GraphOptimizationLevel::Level1,
            Self::Extended => GraphOptimizationLevel::Level2,
            Self::All => GraphOptimizationLevel::All,
        }
    }
}

/// Session knobs that change speed but not what the model computes (up to float rounding).
/// `Tuning::default()` leaves every choice to ONNX Runtime, i.e. what PhotoCraft did before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Tuning {
    /// Threads inside one operator (convolutions, FFTs). 0 = ONNX Runtime's choice (physical cores).
    pub intra_threads: usize,
    /// Graph rewriting level at load time.
    pub opt: OptLevel,
    /// Let idle worker threads spin briefly instead of sleeping. Lowers latency for back-to-back
    /// runs, burns CPU between them.
    pub no_spinning: bool,
    /// Pre-plan tensor memory from the first run's shapes (CPU only; helps fixed-size models).
    pub no_memory_pattern: bool,
}

impl Tuning {
    /// One-line description, for benchmark tables.
    pub fn label(&self) -> String {
        let threads = if self.intra_threads == 0 { "auto".to_string() } else { self.intra_threads.to_string() };
        format!("threads={threads} opt={} spin={} mempattern={}", self.opt.name(), !self.no_spinning, !self.no_memory_pattern)
    }
}

/// Seconds-scale timings of the last run, for benchmarks and the status line.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunTiming {
    /// Layout conversion into the model's tensors.
    pub prepare: std::time::Duration,
    /// `Session::run`.
    pub session: std::time::Duration,
    /// Output extraction and layout conversion.
    pub finish: std::time::Duration,
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
    tuning: Tuning,
    last: Mutex<RunTiming>,
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
        Self::load_tuned(runtime, model, device, Tuning::default())
    }

    /// [`load`](Self::load) with explicit session settings (see [`Tuning`]).
    pub fn load_tuned(runtime: Option<&Path>, model: &Path, device: Device, tuning: Tuning) -> Result<Self> {
        load_runtime(runtime)?;
        let info = manifest::hash_file(model)?;
        let _ = info;
        let inner = catch_unwind(AssertUnwindSafe(|| Self::build(model, device, tuning)));
        match inner {
            Ok(r) => r,
            Err(_) => Err(MlError::Runtime("ONNX Runtime failed while loading the model (it is probably too old for this model; try a newer runtime)".into())),
        }
    }

    /// The settings this session was created with.
    pub fn tuning(&self) -> Tuning {
        self.tuning
    }

    /// Where the last `run` spent its time.
    pub fn last_timing(&self) -> RunTiming {
        *self.last.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn build(model: &Path, device: Device, tuning: Tuning) -> Result<Self> {
        let gpus = if device == Device::Auto { gpu_providers() } else { Vec::new() };
        let names: Vec<&str> = gpus.iter().map(|(n, _)| *n).collect();
        fn rt<E: std::fmt::Display>(e: E) -> MlError {
            MlError::Runtime(e.to_string())
        }
        let mut builder = Session::builder().map_err(rt)?.with_optimization_level(tuning.opt.ort()).map_err(rt)?;
        if tuning.intra_threads > 0 {
            builder = builder.with_intra_threads(tuning.intra_threads).map_err(rt)?;
        }
        if tuning.no_spinning {
            builder = builder.with_intra_op_spinning(false).map_err(rt)?;
        }
        if tuning.no_memory_pattern {
            builder = builder.with_memory_pattern(false).map_err(rt)?;
        }
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
        Ok(Self { size, image_input: image.name().to_string(), mask_input: mask.name().to_string(), label, device, tuning, last: Mutex::new(RunTiming::default()), session: Mutex::new(session) })
    }

    fn infer(&self, rgb: &[f32], mask: &[f32]) -> Result<Vec<f32>> {
        let n = self.size;
        let plane = n * n;
        let t0 = std::time::Instant::now();
        // Interleaved HWC -> planar CHW, one pass per channel plane (no per-element bounds checks).
        let mut chw = vec![0.0f32; plane * 3];
        let (r, rest) = chw.split_at_mut(plane);
        let (g, b) = rest.split_at_mut(plane);
        for (((px, r), g), b) in rgb.chunks_exact(3).zip(r.iter_mut()).zip(g.iter_mut()).zip(b.iter_mut()) {
            if let [pr, pg, pb] = px {
                *r = *pr;
                *g = *pg;
                *b = *pb;
            }
        }
        let dim = n as i64;
        let image = Tensor::from_array((Shape::new([1, 3, dim, dim]), chw)).map_err(|e| MlError::Runtime(e.to_string()))?;
        let mask_t = Tensor::from_array((Shape::new([1, 1, dim, dim]), mask.to_vec())).map_err(|e| MlError::Runtime(e.to_string()))?;
        let prepare = t0.elapsed();
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let t1 = std::time::Instant::now();
        let outputs = session
            .run(ort::inputs![self.image_input.as_str() => image, self.mask_input.as_str() => mask_t])
            .map_err(|e| MlError::Model(format!("the model failed: {e}")))?;
        let ran = t1.elapsed();
        let t2 = std::time::Instant::now();
        let first = outputs.iter().next().ok_or_else(|| MlError::Model("the model returned nothing".into()))?.1;
        let (shape, data) = first.try_extract_tensor::<f32>().map_err(|e| MlError::Model(format!("unexpected model output: {e}")))?;
        if data.len() != plane * 3 {
            return Err(MlError::Model(format!("the model returned {} values, expected {} for a {n}×{n} image", data.len(), plane * 3)));
        }
        // Exports differ: planar (1,3,H,W) or interleaved (1,H,W,3); 0..1 or 0..255.
        let planar = shape.last().copied() != Some(3);
        let peak = data.iter().copied().fold(0.0f32, |m, v| if v.is_finite() { m.max(v) } else { m });
        let inv = if peak > 1.5 { 1.0 / 255.0 } else { 1.0 };
        let mut out = vec![0.0f32; plane * 3];
        if planar {
            let (r, rest) = data.split_at(plane);
            let (g, b) = rest.split_at(plane);
            for (((o, r), g), b) in out.chunks_exact_mut(3).zip(r).zip(g).zip(b) {
                if let [or, og, ob] = o {
                    *or = *r * inv;
                    *og = *g * inv;
                    *ob = *b * inv;
                }
            }
        } else {
            for (o, s) in out.iter_mut().zip(data) {
                *o = *s * inv;
            }
        }
        *self.last.lock().unwrap_or_else(PoisonError::into_inner) = RunTiming { prepare, session: ran, finish: t2.elapsed() };
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
    fn timing_split_ms(&self) -> Option<(f64, f64, f64)> {
        let t = self.last_timing();
        Some((t.prepare.as_secs_f64() * 1e3, t.session.as_secs_f64() * 1e3, t.finish.as_secs_f64() * 1e3))
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
