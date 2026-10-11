//! Inpainting with a fixed-size model (LaMa and friends): the model sees a square window around
//! the hole and returns the same window filled; this module chooses that window, scales it to the
//! model's size, runs the model and writes only the hole's pixels back.
//!
//! The model contract ([`InpaintModel`]) is deliberately tiny so the ONNX backend, a test stub and
//! any future model (MI-GAN, a GPU build) are interchangeable:
//!
//! - `rgb`: `size × size × 3` interleaved, display-referred, 0..1;
//! - `mask`: `size × size`, 1 inside the hole;
//! - returns `size × size × 3`, 0..1.
//!
//! Choices made here (all measured on the model's size, not the image's):
//!
//! - The window is `max(size, hole extent × (1 + 2 × context))` pixels wide, centred on the hole
//!   and clamped to the image. At that size most holes are processed 1:1 with the pixels around
//!   them; a bigger hole is scaled down to fit and the fill comes back softer (the model has to
//!   invent the structure either way). Never scaled up: that would only blur.
//! - A window smaller than the model (small images) is padded by replicating its edge, with the
//!   padding marked as known so the model does not fill it.
//! - The mask given to the model is grown by `dilate` pixels so a half-covered edge pixel of the
//!   selection is repainted rather than left as a halo.
//! - Pixels outside the hole are never modified, bit for bit.

use crate::error::{MlError, Result};

/// A model that fills holes in a square image of fixed size.
pub trait InpaintModel: Send + Sync {
    /// A short name for status lines and command results.
    fn name(&self) -> String;
    /// Side of the square image the model takes, in pixels.
    fn size(&self) -> usize;
    /// Fill: see the module docs for the buffer layout.
    fn run(&self, rgb: &[f32], mask: &[f32]) -> Result<Vec<f32>>;
}

/// The pixels to fill: interleaved RGB (0..1) and a hole flag per pixel.
#[derive(Clone, Copy, Debug)]
pub struct Region<'a> {
    pub width: usize,
    pub height: usize,
    pub rgb: &'a [f32],
    pub hole: &'a [bool],
}

/// Tuning, with defaults that match the model's training distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// Extra context around the hole's bounding box, as a fraction of its larger side, on each side.
    pub context: f32,
    /// Pixels (at model resolution) the mask is grown by.
    pub dilate: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { context: 0.5, dilate: 3 }
    }
}

/// The result: the whole region, with only hole pixels changed.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub rgb: Vec<f32>,
    /// The window the model saw: `[x, y, width, height]` in region pixels.
    pub window: [usize; 4],
    /// Model pixels per region pixel (1.0 = not scaled; below 1.0 the window was shrunk).
    pub scale: f32,
}

/// Largest side accepted, and the cap on pixels (same bounds as the rest of the engine's windows).
pub const MAX_SIDE: usize = 1 << 16;
pub const MAX_PIXELS: usize = 1 << 28;
/// Model sizes outside this are not a real inpainting model.
pub const MIN_MODEL: usize = 16;
pub const MAX_MODEL: usize = 4096;

/// A planar-free image: `data` is `w × h × c` interleaved.
#[derive(Clone, Debug)]
struct Plane {
    w: usize,
    h: usize,
    c: usize,
    data: Vec<f32>,
}

impl Plane {
    fn new(w: usize, h: usize, c: usize) -> Self {
        Self { w, h, c, data: vec![0.0; w.saturating_mul(h).saturating_mul(c)] }
    }
    fn at(&self, x: usize, y: usize, ch: usize) -> f32 {
        self.data.get((y * self.w + x) * self.c + ch).copied().unwrap_or(0.0)
    }
    fn set(&mut self, x: usize, y: usize, ch: usize, v: f32) {
        let i = (y * self.w + x) * self.c + ch;
        if let Some(s) = self.data.get_mut(i) {
            *s = v;
        }
    }
}

/// Weights mapping `src` samples to each of `dst` samples along one axis: area-averaging when
/// shrinking (no aliasing), centre-aligned linear when growing.
fn axis_weights(src: usize, dst: usize) -> Vec<Vec<(usize, f32)>> {
    let mut out = Vec::with_capacity(dst);
    if src == 0 || dst == 0 {
        return out;
    }
    let ratio = src as f64 / dst as f64;
    for i in 0..dst {
        let mut taps: Vec<(usize, f32)> = Vec::new();
        if dst < src {
            let (a, b) = (i as f64 * ratio, (i + 1) as f64 * ratio);
            let first = a.floor() as usize;
            let last = (b.ceil() as usize).min(src);
            for j in first..last.max(first + 1) {
                let w = (b.min((j + 1) as f64) - a.max(j as f64)).max(0.0);
                if w > 0.0 && j < src {
                    taps.push((j, w as f32));
                }
            }
        } else {
            let pos = (i as f64 + 0.5) * ratio - 0.5;
            let lo = pos.floor();
            let frac = (pos - lo) as f32;
            let clampi = |v: f64| v.clamp(0.0, (src - 1) as f64) as usize;
            let (a, b) = (clampi(lo), clampi(lo + 1.0));
            taps.push((a, 1.0 - frac));
            taps.push((b, frac));
        }
        let sum: f32 = taps.iter().map(|t| t.1).sum();
        if sum > 0.0 {
            for t in &mut taps {
                t.1 /= sum;
            }
        } else {
            taps.push((0, 1.0));
        }
        out.push(taps);
    }
    out
}

/// Resize to `dw × dh` (separable; each axis shrinks by area or grows linearly).
fn resize(src: &Plane, dw: usize, dh: usize) -> Plane {
    if src.w == dw && src.h == dh {
        return src.clone();
    }
    let wx = axis_weights(src.w, dw);
    let wy = axis_weights(src.h, dh);
    let mut mid = Plane::new(dw, src.h, src.c);
    for y in 0..src.h {
        for (x, taps) in wx.iter().enumerate() {
            for ch in 0..src.c {
                let v: f32 = taps.iter().map(|&(j, w)| src.at(j, y, ch) * w).sum();
                mid.set(x, y, ch, v);
            }
        }
    }
    let mut out = Plane::new(dw, dh, src.c);
    for (y, taps) in wy.iter().enumerate() {
        for x in 0..dw {
            for ch in 0..src.c {
                let v: f32 = taps.iter().map(|&(j, w)| mid.at(x, j, ch) * w).sum();
                out.set(x, y, ch, v);
            }
        }
    }
    out
}

/// Grow the 1-channel mask by `r` pixels (square structuring element, two passes).
fn dilate(mask: &mut Plane, r: usize) {
    if r == 0 {
        return;
    }
    let (w, h) = (mask.w, mask.h);
    let mut tmp = Plane::new(w, h, 1);
    for y in 0..h {
        for x in 0..w {
            let mut m = 0.0f32;
            for xx in x.saturating_sub(r)..=(x + r).min(w.saturating_sub(1)) {
                m = m.max(mask.at(xx, y, 0));
            }
            tmp.set(x, y, 0, m);
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut m = 0.0f32;
            for yy in y.saturating_sub(r)..=(y + r).min(h.saturating_sub(1)) {
                m = m.max(tmp.at(x, yy, 0));
            }
            mask.set(x, y, 0, m);
        }
    }
}

fn check_region(r: &Region<'_>) -> Result<usize> {
    if r.width == 0 || r.height == 0 || r.width > MAX_SIDE || r.height > MAX_SIDE {
        return Err(MlError::BadInput(format!("the region is {}×{} pixels; it must be between 1 and {MAX_SIDE} on each side", r.width, r.height)));
    }
    let px = r.width.checked_mul(r.height).filter(|p| *p <= MAX_PIXELS).ok_or_else(|| MlError::BadInput("the region is too large to fill".into()))?;
    if r.rgb.len() != px * 3 || r.hole.len() != px {
        return Err(MlError::BadInput("the region's buffers do not match its size".into()));
    }
    Ok(px)
}

/// Fill the hole of `region` with `model`. `cancelled` is polled between stages.
pub fn inpaint(model: &dyn InpaintModel, region: Region<'_>, opts: &Options, cancelled: &dyn Fn() -> bool) -> Result<Output> {
    let px = check_region(&region)?;
    let (w, h) = (region.width, region.height);
    let size = model.size();
    if !(MIN_MODEL..=MAX_MODEL).contains(&size) {
        return Err(MlError::Model(format!("the model works at {size}×{size} pixels, which is not a usable size")));
    }
    // The hole's bounding box.
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for (i, &hole) in region.hole.iter().enumerate() {
        if hole {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    if x0 == usize::MAX {
        return Ok(Output { rgb: region.rgb.to_vec(), window: [0, 0, w, h], scale: 1.0 });
    }
    let (bw, bh) = (x1 - x0, y1 - y0);
    // The window: model-sized or the hole plus context, whichever is larger; centred, clamped.
    let ctx = if opts.context.is_finite() { opts.context.clamp(0.0, 4.0) } else { 0.5 };
    let want = ((bw.max(bh) as f64) * (1.0 + 2.0 * ctx as f64)).ceil() as usize;
    let side = want.max(size);
    let (ww, wh) = (side.min(w), side.min(h));
    let wx = ((x0 + x1) / 2).saturating_sub(ww / 2).min(w - ww);
    let wy = ((y0 + y1) / 2).saturating_sub(wh / 2).min(h - wh);
    let scale = if ww.max(wh) > size { size as f32 / ww.max(wh) as f32 } else { 1.0 };
    let sw = ((ww as f32 * scale).round() as usize).clamp(1, size);
    let sh = ((wh as f32 * scale).round() as usize).clamp(1, size);

    // Cut the window out.
    let mut rgb = Plane::new(ww, wh, 3);
    let mut mask = Plane::new(ww, wh, 1);
    for y in 0..wh {
        for x in 0..ww {
            let i = (wy + y) * w + wx + x;
            for ch in 0..3 {
                rgb.set(x, y, ch, region.rgb.get(i * 3 + ch).copied().unwrap_or(0.0));
            }
            mask.set(x, y, 0, if region.hole.get(i).copied().unwrap_or(false) { 1.0 } else { 0.0 });
        }
    }
    if cancelled() {
        return Err(MlError::Cancelled);
    }
    // Scale to the model's resolution, pad to a square by replicating the edge.
    let small = resize(&rgb, sw, sh);
    let mut small_mask = resize(&mask, sw, sh);
    for v in &mut small_mask.data {
        *v = if *v > 0.01 { 1.0 } else { 0.0 };
    }
    let mut input = Plane::new(size, size, 3);
    let mut input_mask = Plane::new(size, size, 1);
    for y in 0..size {
        for x in 0..size {
            let (sx, sy) = (x.min(sw - 1), y.min(sh - 1));
            for ch in 0..3 {
                input.set(x, y, ch, small.at(sx, sy, ch));
            }
            if x < sw && y < sh {
                input_mask.set(x, y, 0, small_mask.at(x, y, 0));
            }
        }
    }
    dilate(&mut input_mask, opts.dilate.min(16));
    // Padding stays known to the model.
    for y in 0..size {
        for x in 0..size {
            if x >= sw || y >= sh {
                input_mask.set(x, y, 0, 0.0);
            }
        }
    }
    if cancelled() {
        return Err(MlError::Cancelled);
    }

    let mut out = model.run(&input.data, &input_mask.data)?;
    if out.len() != size * size * 3 {
        return Err(MlError::Model(format!("{} returned {} values, expected {}", model.name(), out.len(), size * size * 3)));
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(MlError::Model(format!("{} produced values that are not numbers", model.name())));
    }
    for v in &mut out {
        *v = v.clamp(0.0, 1.0);
    }
    if cancelled() {
        return Err(MlError::Cancelled);
    }

    // Back to the window's size, then into the hole only.
    let mut filled_small = Plane::new(sw, sh, 3);
    for y in 0..sh {
        for x in 0..sw {
            for ch in 0..3 {
                filled_small.set(x, y, ch, out.get((y * size + x) * 3 + ch).copied().unwrap_or(0.0));
            }
        }
    }
    let filled = resize(&filled_small, ww, wh);
    let mut result = region.rgb.to_vec();
    debug_assert_eq!(result.len(), px * 3);
    for y in 0..wh {
        for x in 0..ww {
            if mask.at(x, y, 0) > 0.0 {
                let i = (wy + y) * w + wx + x;
                for ch in 0..3 {
                    if let Some(d) = result.get_mut(i * 3 + ch) {
                        *d = filled.at(x, y, ch);
                    }
                }
            }
        }
    }
    Ok(Output { rgb: result, window: [wx, wy, ww, wh], scale })
}

/// A model that fills the hole with one colour and copies everything else. For tests of the code
/// around a model (this crate's, the engine's, the app's) without a model file.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SolidFill {
    pub size: usize,
    pub color: [f32; 3],
}

impl InpaintModel for SolidFill {
    fn name(&self) -> String {
        "solid fill (test)".into()
    }
    fn size(&self) -> usize {
        self.size
    }
    fn run(&self, rgb: &[f32], mask: &[f32]) -> Result<Vec<f32>> {
        if rgb.len() != self.size * self.size * 3 || mask.len() != self.size * self.size {
            return Err(MlError::BadInput("buffer sizes do not match the model".into()));
        }
        let mut out = rgb.to_vec();
        for (i, m) in mask.iter().enumerate() {
            if *m > 0.5 {
                for (ch, c) in self.color.iter().enumerate() {
                    if let Some(d) = out.get_mut(i * 3 + ch) {
                        *d = *c;
                    }
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
