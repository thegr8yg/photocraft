//! Local models: the LaMa neural fill for Edit › Content-Aware Fill, and `ml.status` to see what
//! is set up.
//!
//! Everything runs on the user's computer through ONNX Runtime ([`photocraft_ml`]). The runtime
//! library and the model file are the user's: Preferences › Local Models points at them
//! (`localModels.onnxRuntimeLibrary`, `localModels.lamaModel`), nothing is downloaded or
//! bundled. The model loads on a worker the first time it is needed and stays loaded while the
//! settings do not change.
//!
//! A build without the `local-ml` feature, a missing library, a missing or mismatching model
//! file and a model that misbehaves are all errors that say what to do; the document is left
//! alone.

use std::sync::{Arc, Mutex};

use photocraft_color::PixelFormat;
use photocraft_ml::{InpaintModel, MlError, Options, Region};
use photocraft_raster::{from_rgba_into, to_rgba};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::jobs::JobCtx;
use crate::prefs::LocalModels;
use crate::{EngineError, Result, Session};

/// The loaded model, shared with the workers that fill.
#[derive(Default)]
#[cfg_attr(not(feature = "local-ml"), allow(dead_code))]
struct Cache {
    key: String,
    model: Option<Arc<dyn InpaintModel>>,
}

/// Session state of the local models.
#[derive(Default)]
pub struct MlState {
    cache: Arc<Mutex<Cache>>,
    /// A model that stands in for LaMa (tests of the code around it, without a model file).
    stand_in: Option<Arc<dyn InpaintModel>>,
}

/// What a background job carries to get the model: cheap to clone, `Send`.
#[derive(Clone)]
pub struct MlHandle {
    cache: Arc<Mutex<Cache>>,
    stand_in: Option<Arc<dyn InpaintModel>>,
}

impl MlState {
    /// Use `model` instead of loading LaMa from the preferences (tests).
    #[doc(hidden)]
    pub fn set_stand_in(&mut self, model: Arc<dyn InpaintModel>) {
        self.stand_in = Some(model);
    }

    pub fn handle(&self) -> MlHandle {
        MlHandle { cache: self.cache.clone(), stand_in: self.stand_in.clone() }
    }
}

fn ml_error(e: MlError) -> EngineError {
    match e {
        MlError::Cancelled => EngineError::Cancelled,
        other => EngineError::Other(other.to_string()),
    }
}

/// Fail early, on the UI thread, when LaMa cannot possibly run: nothing is configured.
pub fn precheck(cfg: &LocalModels, handle: &MlHandle) -> Result<()> {
    if handle.stand_in.is_some() {
        return Ok(());
    }
    if !photocraft_ml::onnx_compiled() {
        return Err(EngineError::Other(
            "this build of PhotoCraft has no local-model support; build with `--features local-ml` (docs/local-models.md), or use the classical Content-Aware Fill".into(),
        ));
    }
    if cfg.lama_model.trim().is_empty() {
        return Err(EngineError::Other(
            "LaMa needs a model file: choose one in Preferences › Local Models (an ONNX export of LaMa, see docs/local-models.md), or use the classical Content-Aware Fill".into(),
        ));
    }
    Ok(())
}

impl MlHandle {
    /// The LaMa model for these settings, loading it the first time (seconds for the first fill,
    /// instant afterwards).
    pub fn lama(&self, cfg: &LocalModels) -> Result<Arc<dyn InpaintModel>> {
        if let Some(m) = &self.stand_in {
            return Ok(m.clone());
        }
        precheck(cfg, self)?;
        self.load(cfg)
    }

    #[cfg(all(feature = "local-ml", not(target_arch = "wasm32")))]
    fn load(&self, cfg: &LocalModels) -> Result<Arc<dyn InpaintModel>> {
        use std::path::Path;
        let key = format!("{}|{}|{}|{}", cfg.onnx_runtime_library, cfg.lama_model, cfg.device.name(), cfg.lama_sha256);
        let mut cache = self.cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.key == key
            && let Some(m) = &cache.model
        {
            return Ok(m.clone());
        }
        let model_path = Path::new(cfg.lama_model.trim());
        let info = photocraft_ml::manifest::hash_file(model_path).map_err(ml_error)?;
        photocraft_ml::manifest::check(&photocraft_ml::manifest::LAMA, &info, &cfg.lama_sha256).map_err(ml_error)?;
        let runtime = (!cfg.onnx_runtime_library.trim().is_empty()).then(|| Path::new(cfg.onnx_runtime_library.trim()));
        let device = photocraft_ml::Device::parse(cfg.device.name()).unwrap_or_default();
        let model: Arc<dyn InpaintModel> = Arc::new(photocraft_ml::OrtLama::load(runtime, model_path, device).map_err(ml_error)?);
        cache.key = key;
        cache.model = Some(model.clone());
        Ok(model)
    }

    #[cfg(not(all(feature = "local-ml", not(target_arch = "wasm32"))))]
    fn load(&self, _: &LocalModels) -> Result<Arc<dyn InpaintModel>> {
        let _ = &self.cache;
        Err(EngineError::Other("this build of PhotoCraft has no local-model support".into()))
    }
}

/// Fill the hole of a window of document pixels with `model`. `img` is `w × h` pixels of `fmt`
/// (interleaved); the result has the same layout with the hole's pixels replaced and everything
/// else equal to `img`. Colours go through the sRGB RGBA intermediate every mode conversion in
/// the engine uses (gray, CMYK and Lab included); hole pixels come back opaque.
pub fn lama_fill(model: &dyn InpaintModel, w: usize, h: usize, img: &[f32], hole: &[bool], fmt: &PixelFormat, ctx: &JobCtx) -> Result<Vec<f32>> {
    let n = fmt.channels();
    if n == 0 || img.len() != w.saturating_mul(h).saturating_mul(n) || hole.len() != w.saturating_mul(h) {
        return Err(EngineError::Other("internal error: the fill window does not match its pixels".into()));
    }
    let mut rgb = Vec::with_capacity(w * h * 3);
    for px in img.chunks_exact(n) {
        let c = to_rgba(fmt, px);
        rgb.extend(c.iter().take(3).map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }));
    }
    ctx.check()?;
    let out = photocraft_ml::inpaint(model, Region { width: w, height: h, rgb: &rgb, hole }, &Options::default(), &|| ctx.check().is_err()).map_err(ml_error)?;
    let mut filled = img.to_vec();
    let mut enc = [0.0f32; 8];
    for (i, (&is_hole, px)) in hole.iter().zip(out.rgb.chunks_exact(3)).enumerate() {
        if !is_hole {
            continue;
        }
        let k = from_rgba_into(fmt, [px.first().copied().unwrap_or(0.0), px.get(1).copied().unwrap_or(0.0), px.get(2).copied().unwrap_or(0.0), 1.0], &mut enc);
        let k = k.min(n);
        if let (Some(dst), Some(src)) = (filled.get_mut(i * n..i * n + k), enc.get(..k)) {
            dst.copy_from_slice(src);
        }
    }
    Ok(filled)
}

// ------------------------------------------------------------------ ml.status

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

fn path_status(path: &str) -> Value {
    let p = path.trim();
    if p.is_empty() {
        return json!({"set": false});
    }
    match std::fs::metadata(p) {
        Ok(m) if m.is_file() => json!({"set": true, "path": p, "found": true, "bytes": m.len()}),
        Ok(_) => json!({"set": true, "path": p, "found": false, "error": "not a file"}),
        Err(e) => json!({"set": true, "path": p, "found": false, "error": e.to_string()}),
    }
}

/// What is set up, and with `verify` / `load` whether it works. Reads only; `load` starts the
/// model (seconds, once).
fn status(s: &mut Session, p: &Value) -> Result<Value> {
    let mut cfg = s.prefs.get().local_models.clone();
    // Settings can be tried before they are saved (the Preferences dialog's "Check setup").
    for (key, field) in [("onnxRuntimeLibrary", &mut cfg.onnx_runtime_library), ("lamaModel", &mut cfg.lama_model), ("lamaSha256", &mut cfg.lama_sha256)] {
        match p.get(key) {
            None | Some(Value::Null) => {}
            Some(Value::String(v)) if v.len() <= 4096 && !v.contains('\0') => *field = v.clone(),
            Some(_) => return Err(EngineError::BadParams { cmd: "ml.status".into(), msg: format!("`{key}` must be text of at most 4096 characters") }),
        }
    }
    if let Some(d) = p.get("device") {
        cfg.device = d.as_str().and_then(crate::prefs::LocalDevice::parse).ok_or_else(|| EngineError::BadParams { cmd: "ml.status".into(), msg: "`device` must be auto|cpu".into() })?;
    }
    let verify = p.get("verify").and_then(Value::as_bool).unwrap_or(false);
    let load = p.get("load").and_then(Value::as_bool).unwrap_or(false);
    let model = photocraft_ml::manifest::LAMA;
    let mut lama = path_status(&cfg.lama_model);
    if verify && lama.get("found") == Some(&json!(true)) {
        match photocraft_ml::manifest::hash_file(std::path::Path::new(cfg.lama_model.trim())) {
            Ok(info) => {
                let ok = photocraft_ml::manifest::check(&model, &info, &cfg.lama_sha256);
                lama["sha256"] = json!(info.sha256);
                lama["pinMatches"] = json!(ok.is_ok());
                if let Err(e) = ok {
                    lama["error"] = json!(e.to_string());
                }
            }
            Err(e) => lama["error"] = json!(e.to_string()),
        }
    }
    let mut out = json!({
        "compiled": photocraft_ml::onnx_compiled(),
        "runtime": path_status(&cfg.onnx_runtime_library),
        "lama": lama,
        "device": cfg.device.name(),
        "contentAwareFill": cfg.content_aware_fill.name(),
        "model": {"id": model.id, "label": model.label, "source": model.source, "license": model.license},
    });
    let ready = if load {
        let handle = s.ml.handle();
        match handle.lama(&cfg) {
            Ok(m) => {
                out["loaded"] = json!({"name": m.name(), "size": m.size()});
                true
            }
            Err(e) => {
                out["loadError"] = json!(e.to_string());
                false
            }
        }
    } else {
        precheck(&cfg, &s.ml.handle()).is_ok()
    };
    out["configured"] = json!(ready);
    Ok(out)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "ml.status",
        label: "Local Models Status",
        menu: &[],
        shortcut: None,
        params: r##"{"verify":bool=false,"load":bool=false,"onnxRuntimeLibrary":path?,"lamaModel":path?,"lamaSha256":hex?,"device":"auto|cpu"?} (verify hashes the LaMa file; load starts the model and reports the device; the path/device params try settings before they are saved)"##,
        enabled: always,
        run: status,
        journal: false,
    }]
}

#[cfg(test)]
mod tests;
