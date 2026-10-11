//! Inpainting benchmark: latency and output fidelity of LaMa-style models across ONNX Runtime
//! session settings, on fixed synthetic cases or your own images.
//!
//! ```text
//! cargo run --release -p photocraft-ml --features onnx --example lama_bench -- \
//!     --model lama_fp32.onnx --runtime /path/libonnxruntime.so [--images DIR] [--reps 10] [--json out.json]
//! ```
//!
//! What it measures, per configuration (the first one is the reference):
//! - `load`: time to create the session;
//! - `e2e`: p50/p95 of the whole `photocraft_ml::inpaint` call (window cut, resize, model, composite);
//! - `model`: p50 of `InpaintModel::run`, split into prepare / session / finish where the backend reports it;
//! - `drift`: PSNR of the filled pixels against the reference configuration's (a speed setting that
//!   is lossless shows > 60 dB; below ~40 dB it changed what the model computes);
//! - `truth`: PSNR / MAE of the filled pixels against the untouched source image (meaningful for real
//!   photos; the synthetic cases only prove the harness). Use it to compare *models*, not settings.
//!
//! `--stub` swaps the model for a flat-colour filler so the harness can be checked without a model or
//! ONNX Runtime. `--config "threads=4,opt=basic,nospin,nomem"` (repeatable) replaces the default sweep.
//! `--images DIR` reads binary PPM (P6) files (convert with any tool); holes are generated from a
//! fixed seed so runs compare. Not a test: it prints a table and (with `--json`) a report.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use photocraft_ml::inpaint::{InpaintModel, Options, Region, inpaint};
use photocraft_ml::{MlError, Result};

#[cfg(feature = "onnx")]
use photocraft_ml::{Device, OptLevel, OrtLama, Tuning};

/// A source image (interleaved RGB, 0..1) and its hole mask.
struct Case {
    name: String,
    w: usize,
    h: usize,
    rgb: Vec<f32>,
    hole: Vec<bool>,
}

/// Deterministic pseudo-random numbers (xorshift): same cases on every run and machine.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn synthetic_image(w: usize, h: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let mut v = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let stripes = (((x + y) / 24) % 2) as f32 * 0.15;
            let base = [0.2 + 0.6 * fx, 0.3 + 0.4 * fy, 0.7 - 0.5 * fx * fy];
            for (c, b) in base.iter().enumerate() {
                v[(y * w + x) * 3 + c] = (b + stripes + (rng.unit() - 0.5) * 0.04).clamp(0.0, 1.0);
            }
        }
    }
    v
}

fn blob(w: usize, h: usize, cx: f32, cy: f32, rx: f32, ry: f32) -> Vec<bool> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 - cx, (i / w) as f32 - cy);
            (x / rx).powi(2) + (y / ry).powi(2) <= 1.0
        })
        .collect()
}

fn scratch(w: usize, h: usize) -> Vec<bool> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            (y - (0.2 * x + h as f32 * 0.3)).abs() < 3.0 && x > w as f32 * 0.2 && x < w as f32 * 0.8
        })
        .collect()
}

fn synthetic_cases() -> Vec<Case> {
    let (w, h) = (1024, 768);
    let rgb = synthetic_image(w, h, 7);
    let mk = |name: &str, hole: Vec<bool>| Case { name: name.into(), w, h, rgb: rgb.clone(), hole };
    vec![
        mk("small-blob", blob(w, h, 500.0, 380.0, 40.0, 30.0)),
        mk("medium-blob", blob(w, h, 520.0, 400.0, 150.0, 110.0)),
        mk("large-blob (shrunk)", blob(w, h, 512.0, 384.0, 420.0, 300.0)),
        mk("scratch", scratch(w, h)),
    ]
}

/// Binary PPM (P6, maxval 255 or 65535).
fn read_ppm(path: &Path) -> Option<(usize, usize, Vec<f32>)> {
    let bytes = std::fs::read(path).ok()?;
    let mut pos = 0usize;
    let mut tokens: Vec<String> = Vec::new();
    while tokens.len() < 4 && pos < bytes.len() {
        while bytes.get(pos).is_some_and(|b| b.is_ascii_whitespace()) {
            pos += 1;
        }
        if bytes.get(pos) == Some(&b'#') {
            while bytes.get(pos).is_some_and(|b| *b != b'\n') {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while bytes.get(pos).is_some_and(|b| !b.is_ascii_whitespace()) {
            pos += 1;
        }
        tokens.push(String::from_utf8_lossy(bytes.get(start..pos)?).into_owned());
    }
    pos += 1; // the single whitespace after maxval
    if tokens.first().map(String::as_str) != Some("P6") {
        return None;
    }
    let (w, h, max): (usize, usize, usize) = (tokens.get(1)?.parse().ok()?, tokens.get(2)?.parse().ok()?, tokens.get(3)?.parse().ok()?);
    if w == 0 || h == 0 || w.checked_mul(h)? > 1 << 26 || max == 0 || max > 65535 {
        return None;
    }
    let data = bytes.get(pos..)?;
    let wide = max > 255;
    let n = w * h * 3;
    if data.len() < n * if wide { 2 } else { 1 } {
        return None;
    }
    let rgb = (0..n)
        .map(|i| if wide { u16::from_be_bytes([data[i * 2], data[i * 2 + 1]]) as f32 / max as f32 } else { data[i] as f32 / max as f32 })
        .collect();
    Some((w, h, rgb))
}

fn dir_cases(dir: &Path) -> Vec<Case> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "ppm")).collect();
    files.sort();
    let mut out = Vec::new();
    for (k, f) in files.iter().enumerate() {
        let Some((w, h, rgb)) = read_ppm(f) else {
            eprintln!("skipping {} (not a binary PPM)", f.display());
            continue;
        };
        let mut rng = Rng(0x9e37_79b9 + k as u64 * 7919);
        let stem = f.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let side = w.min(h) as f32;
        for (label, frac) in [("small", 0.08f32), ("medium", 0.2), ("large", 0.45)] {
            let (rx, ry) = (side * frac * (0.7 + 0.6 * rng.unit()), side * frac * (0.7 + 0.6 * rng.unit()));
            let cx = rx + rng.unit() * (w as f32 - 2.0 * rx).max(1.0);
            let cy = ry + rng.unit() * (h as f32 - 2.0 * ry).max(1.0);
            out.push(Case { name: format!("{stem}/{label}"), w, h, rgb: rgb.clone(), hole: blob(w, h, cx, cy, rx, ry) });
        }
    }
    out
}

/// Fills with the mean colour of the pixels outside the hole. Only for checking the harness.
struct Stub(usize);
impl InpaintModel for Stub {
    fn name(&self) -> String {
        "stub".into()
    }
    fn size(&self) -> usize {
        self.0
    }
    fn run(&self, rgb: &[f32], mask: &[f32]) -> Result<Vec<f32>> {
        let mut sum = [0.0f64; 3];
        let mut n = 0.0f64;
        for (px, m) in rgb.chunks_exact(3).zip(mask) {
            if *m < 0.5 {
                for (s, v) in sum.iter_mut().zip(px) {
                    *s += *v as f64;
                }
                n += 1.0;
            }
        }
        let mean = sum.map(|s| (s / n.max(1.0)) as f32);
        Ok(rgb.chunks_exact(3).zip(mask).flat_map(|(px, m)| if *m >= 0.5 { mean } else { [px[0], px[1], px[2]] }).collect())
    }
}

#[derive(Clone, Debug, Default)]
struct Config {
    label: String,
    threads: usize,
    opt: String,
    no_spin: bool,
    no_mem: bool,
}

fn parse_config(s: &str) -> Option<Config> {
    let mut c = Config { opt: "all".into(), ..Config::default() };
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('=') {
            Some(("threads", v)) => c.threads = v.parse().ok()?,
            Some(("opt", v)) => c.opt = v.into(),
            None if part == "nospin" => c.no_spin = true,
            None if part == "nomem" => c.no_mem = true,
            None if part == "default" => {}
            _ => return None,
        }
    }
    c.label = s.to_string();
    Some(c)
}

fn default_sweep() -> Vec<Config> {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut specs = vec!["default".to_string(), "threads=1".to_string()];
    for t in [2usize, 4, 8, cores] {
        if t > 1 && t <= cores && !specs.contains(&format!("threads={t}")) {
            specs.push(format!("threads={t}"));
        }
    }
    specs.extend(["opt=basic", "opt=extended", "nospin", "nomem"].map(String::from));
    specs.iter().filter_map(|s| parse_config(s)).collect()
}

struct Stats {
    p50: f64,
    p95: f64,
}
fn stats(mut ms: Vec<f64>) -> Stats {
    ms.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| ms.get(((ms.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(f64::NAN);
    Stats { p50: at(0.5), p95: at(0.95) }
}
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// PSNR (dB, peak 1) and MAE over the pixels where `hole` is set.
fn diff(a: &[f32], b: &[f32], hole: &[bool]) -> (f64, f64) {
    let (mut se, mut ae, mut n) = (0.0f64, 0.0f64, 0.0f64);
    for (i, _) in hole.iter().enumerate().filter(|(_, h)| **h) {
        for c in 0..3 {
            let d = (a.get(i * 3 + c).copied().unwrap_or(0.0) - b.get(i * 3 + c).copied().unwrap_or(0.0)) as f64;
            se += d * d;
            ae += d.abs();
            n += 1.0;
        }
    }
    if n == 0.0 {
        return (f64::INFINITY, 0.0);
    }
    let mse = se / n;
    (if mse <= 1e-12 { 200.0 } else { 10.0 * (1.0 / mse).log10() }, ae / n)
}

struct Row {
    label: String,
    load_ms: f64,
    e2e: Stats,
    model_p50: f64,
    breakdown: Option<(f64, f64, f64)>,
    drift_db: f64,
    truth_db: f64,
    truth_mae: f64,
}

struct Args {
    model: Option<PathBuf>,
    runtime: Option<PathBuf>,
    images: Option<PathBuf>,
    reps: usize,
    warmup: usize,
    json: Option<PathBuf>,
    stub: bool,
    configs: Vec<Config>,
    cpu: bool,
}

fn parse_args() -> std::result::Result<Args, String> {
    let mut a = Args { model: None, runtime: None, images: None, reps: 10, warmup: 2, json: None, stub: false, configs: Vec::new(), cpu: true };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--model" => a.model = Some(val("--model")?.into()),
            "--runtime" => a.runtime = Some(val("--runtime")?.into()),
            "--images" => a.images = Some(val("--images")?.into()),
            "--reps" => a.reps = val("--reps")?.parse().map_err(|_| "--reps must be a number")?,
            "--warmup" => a.warmup = val("--warmup")?.parse().map_err(|_| "--warmup must be a number")?,
            "--json" => a.json = Some(val("--json")?.into()),
            "--config" => {
                let s = val("--config")?;
                a.configs.push(parse_config(&s).ok_or_else(|| format!("bad --config `{s}` (threads=N, opt=off|basic|extended|all, nospin, nomem)"))?);
            }
            "--stub" => a.stub = true,
            "--gpu" => a.cpu = false,
            "-h" | "--help" => return Err("see the header of crates/ml/examples/lama_bench.rs".into()),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    if a.reps == 0 {
        return Err("--reps must be at least 1".into());
    }
    Ok(a)
}

#[cfg(feature = "onnx")]
fn load_model(a: &Args, c: &Config) -> Result<Box<dyn InpaintModel>> {
    if a.stub {
        return Ok(Box::new(Stub(512)));
    }
    let model = a.model.as_deref().ok_or_else(|| MlError::BadInput("pass --model <lama.onnx> (or --stub)".into()))?;
    let opt = OptLevel::parse(&c.opt).ok_or_else(|| MlError::BadInput(format!("bad opt level `{}`", c.opt)))?;
    let tuning = Tuning { intra_threads: c.threads, opt, no_spinning: c.no_spin, no_memory_pattern: c.no_mem };
    let device = if a.cpu { Device::Cpu } else { Device::Auto };
    Ok(Box::new(OrtLama::load_tuned(a.runtime.as_deref(), model, device, tuning)?))
}

#[cfg(not(feature = "onnx"))]
fn load_model(a: &Args, _c: &Config) -> Result<Box<dyn InpaintModel>> {
    if a.stub { Ok(Box::new(Stub(512))) } else { Err(MlError::Runtime("build with --features onnx".into())) }
}

fn run_config(a: &Args, c: &Config, cases: &[Case], reference: &mut Vec<Vec<f32>>) -> Result<Row> {
    let t = Instant::now();
    let model = load_model(a, c)?;
    let load_ms = ms(t.elapsed());
    let opts = Options::default();
    let never = || false;
    let mut e2e = Vec::new();
    let mut model_ms = Vec::new();
    let (mut drift, mut truth_se, mut truth_mae, mut truth_n) = (f64::INFINITY, 0.0f64, 0.0f64, 0.0f64);
    let fill_reference = reference.is_empty();
    for (k, case) in cases.iter().enumerate() {
        let region = Region { width: case.w, height: case.h, rgb: &case.rgb, hole: &case.hole };
        for _ in 0..a.warmup {
            inpaint(model.as_ref(), region, &opts, &never)?;
        }
        let mut last = None;
        for _ in 0..a.reps {
            let t = Instant::now();
            let out = inpaint(model.as_ref(), region, &opts, &never)?;
            e2e.push(ms(t.elapsed()));
            last = Some(out);
        }
        // One bare model call at the model's size, to separate it from the window work.
        let sz = model.size();
        let probe_rgb = vec![0.5f32; sz * sz * 3];
        let mut probe_mask = vec![0.0f32; sz * sz];
        for y in sz / 3..2 * sz / 3 {
            for x in sz / 3..2 * sz / 3 {
                probe_mask[y * sz + x] = 1.0;
            }
        }
        for _ in 0..a.reps.min(5) {
            let t = Instant::now();
            model.run(&probe_rgb, &probe_mask)?;
            model_ms.push(ms(t.elapsed()));
        }
        let out = last.ok_or_else(|| MlError::BadInput("no repetitions".into()))?;
        if fill_reference {
            reference.push(out.rgb.clone());
        } else if let Some(r) = reference.get(k) {
            drift = drift.min(diff(&out.rgb, r, &case.hole).0);
        }
        let (psnr, mae) = diff(&out.rgb, &case.rgb, &case.hole);
        truth_se += psnr.min(100.0);
        truth_mae += mae;
        truth_n += 1.0;
    }
    let breakdown = model.timing_split_ms();
    Ok(Row {
        label: c.label.clone(),
        load_ms,
        e2e: stats(e2e),
        model_p50: stats(model_ms).p50,
        breakdown,
        drift_db: if fill_reference { f64::INFINITY } else { drift },
        truth_db: truth_se / truth_n.max(1.0),
        truth_mae: truth_mae / truth_n.max(1.0),
    })
}

fn main() {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let cases = match &a.images {
        Some(d) => dir_cases(d),
        None => synthetic_cases(),
    };
    if cases.is_empty() {
        eprintln!("error: no cases (no binary .ppm files in --images?)");
        std::process::exit(2);
    }
    for c in &cases {
        println!("  case {:<28} {}x{}, hole {} px", c.name, c.w, c.h, c.hole.iter().filter(|h| **h).count());
    }
    let configs = if a.configs.is_empty() { default_sweep() } else { a.configs.clone() };
    println!("{} case(s), {} rep(s) each, {} config(s){}", cases.len(), a.reps, configs.len(), if a.stub { " [stub model]" } else { "" });
    println!("{:<34} {:>8} {:>9} {:>9} {:>9} {:>8} {:>9} {:>8}", "config", "load ms", "e2e p50", "e2e p95", "model p50", "speedup", "drift dB", "truth dB");
    let mut reference = Vec::new();
    let mut rows: Vec<Row> = Vec::new();
    for c in &configs {
        match run_config(&a, c, &cases, &mut reference) {
            Ok(r) => {
                let base = rows.first().map(|b| b.e2e.p50).unwrap_or(r.e2e.p50);
                let drift = if r.drift_db.is_finite() { format!("{:.1}", r.drift_db) } else { "ref".into() };
                println!(
                    "{:<34} {:>8.0} {:>9.1} {:>9.1} {:>9.1} {:>7.2}x {:>9} {:>8.2}",
                    r.label, r.load_ms, r.e2e.p50, r.e2e.p95, r.model_p50, base / r.e2e.p50, drift, r.truth_db
                );
                if let Some((p, r2, f)) = r.breakdown {
                    println!("{:<34}   last run: prepare {p:.1} ms, session {r2:.1} ms, finish {f:.1} ms", "");
                }
                rows.push(r);
            }
            Err(e) => {
                eprintln!("{}: {e}", c.label);
                std::process::exit(1);
            }
        }
    }
    if let Some(path) = &a.json {
        let body: Vec<String> = rows
            .iter()
            .map(|r| {
                let num = |v: f64| if v.is_finite() { format!("{v:.4}") } else { "null".into() };
                format!(
                    "{{\"config\":\"{}\",\"load_ms\":{},\"e2e_p50_ms\":{},\"e2e_p95_ms\":{},\"model_p50_ms\":{},\"drift_db\":{},\"truth_psnr_db\":{},\"truth_mae\":{}}}",
                    r.label.replace('"', "'"),
                    num(r.load_ms),
                    num(r.e2e.p50),
                    num(r.e2e.p95),
                    num(r.model_p50),
                    num(r.drift_db),
                    num(r.truth_db),
                    num(r.truth_mae)
                )
            })
            .collect();
        if let Err(e) = std::fs::write(path, format!("{{\"cases\":{},\"reps\":{},\"rows\":[{}]}}\n", cases.len(), a.reps, body.join(","))) {
            eprintln!("cannot write {}: {e}", path.display());
            std::process::exit(1);
        }
    }
}
