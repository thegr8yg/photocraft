use super::*;

fn gradient(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            v.extend([x as f32 / w as f32, y as f32 / h as f32, 0.25]);
        }
    }
    v
}

fn rect_hole(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<bool> {
    (0..w * h).map(|i| (x0..x1).contains(&(i % w)) && (y0..y1).contains(&(i / w))).collect()
}

fn never() -> bool {
    false
}

const RED: [f32; 3] = [1.0, 0.0, 0.0];

#[test]
fn fills_only_the_hole() {
    let (w, h) = (200, 160);
    let rgb = gradient(w, h);
    let hole = rect_hole(w, h, 90, 70, 110, 90);
    let m = SolidFill { size: 64, color: RED };
    let out = inpaint(&m, Region { width: w, height: h, rgb: &rgb, hole: &hole }, &Options::default(), &never).unwrap();
    for i in 0..w * h {
        let px = &out.rgb[i * 3..i * 3 + 3];
        if hole[i] {
            // Resampling blends a little at the rim, so compare loosely; the centre is exact.
            let (x, y) = (i % w, i / w);
            if (94..106).contains(&x) && (74..86).contains(&y) {
                assert_eq!(px, RED, "pixel {x},{y}");
            }
        } else {
            assert_eq!(px, &rgb[i * 3..i * 3 + 3], "outside the hole must be untouched (pixel {i})");
        }
    }
    assert_eq!(out.scale, 1.0, "a hole that fits is processed 1:1");
}

#[test]
fn an_image_smaller_than_the_model_is_padded_not_stretched() {
    let (w, h) = (20, 12);
    let rgb = gradient(w, h);
    let hole = rect_hole(w, h, 8, 4, 12, 8);
    let m = SolidFill { size: 64, color: RED };
    let out = inpaint(&m, Region { width: w, height: h, rgb: &rgb, hole: &hole }, &Options::default(), &never).unwrap();
    assert_eq!(out.window, [0, 0, w, h]);
    assert_eq!(&out.rgb[(5 * w + 9) * 3..(5 * w + 9) * 3 + 3], RED);
    assert_eq!(&out.rgb[0..3], &rgb[0..3]);
}

#[test]
fn a_hole_larger_than_the_model_is_scaled_down_and_still_filled() {
    let (w, h) = (640, 480);
    let rgb = gradient(w, h);
    let hole = rect_hole(w, h, 100, 80, 500, 400);
    let m = SolidFill { size: 64, color: RED };
    let out = inpaint(&m, Region { width: w, height: h, rgb: &rgb, hole: &hole }, &Options::default(), &never).unwrap();
    assert!(out.scale < 0.2, "scale {}", out.scale);
    assert_eq!(&out.rgb[(240 * w + 300) * 3..(240 * w + 300) * 3 + 3], RED);
    assert_eq!(&out.rgb[0..3], &rgb[0..3]);
    assert!(out.rgb.iter().all(|v| (0.0..=1.0).contains(v)));
}

#[test]
fn holes_on_the_border_and_the_whole_image_work() {
    let m = SolidFill { size: 32, color: RED };
    for (w, h, hole) in [
        (90, 70, rect_hole(90, 70, 0, 0, 10, 10)),
        (90, 70, rect_hole(90, 70, 80, 60, 90, 70)),
        (90, 70, vec![true; 90 * 70]),
        (1, 1, vec![true]),
        (3, 500, rect_hole(3, 500, 0, 250, 3, 260)),
    ] {
        let rgb = gradient(w, h);
        let out = inpaint(&m, Region { width: w, height: h, rgb: &rgb, hole: &hole }, &Options::default(), &never).unwrap();
        assert_eq!(out.rgb.len(), rgb.len());
        for i in 0..w * h {
            if !hole[i] {
                assert_eq!(&out.rgb[i * 3..i * 3 + 3], &rgb[i * 3..i * 3 + 3]);
            }
        }
        let [wx, wy, ww, wh] = out.window;
        assert!(wx + ww <= w && wy + wh <= h, "window {:?} outside {w}x{h}", out.window);
    }
}

#[test]
fn no_hole_returns_the_input() {
    let rgb = gradient(30, 30);
    let m = SolidFill { size: 32, color: RED };
    let out = inpaint(&m, Region { width: 30, height: 30, rgb: &rgb, hole: &vec![false; 900] }, &Options::default(), &never).unwrap();
    assert_eq!(out.rgb, rgb);
}

#[test]
fn bad_regions_are_errors_not_panics() {
    let m = SolidFill { size: 32, color: RED };
    let rgb = gradient(10, 10);
    let hole = [true; 100];
    let o = Options::default();
    for (w, h, rgb, hole) in [
        (0, 10, &rgb[..], &hole[..]),
        (10, 0, &rgb[..], &hole[..]),
        (10, 10, &rgb[..299], &hole[..]),
        (10, 10, &rgb[..], &hole[..99]),
        (usize::MAX, 2, &rgb[..], &hole[..]),
        (1 << 20, 1, &rgb[..], &hole[..]),
        (1 << 15, 1 << 15, &rgb[..], &hole[..]),
    ] {
        assert!(inpaint(&m, Region { width: w, height: h, rgb, hole }, &o, &never).is_err(), "{w}x{h}");
    }
}

#[test]
fn hostile_options_are_clamped() {
    let m = SolidFill { size: 32, color: RED };
    let rgb = gradient(80, 80);
    let hole = rect_hole(80, 80, 30, 30, 50, 50);
    for o in [Options { context: f32::NAN, dilate: usize::MAX }, Options { context: f32::INFINITY, dilate: 0 }, Options { context: -5.0, dilate: 1000 }] {
        assert!(inpaint(&m, Region { width: 80, height: 80, rgb: &rgb, hole: &hole }, &o, &never).is_ok());
    }
}

struct Broken(Vec<f32>, usize);
impl InpaintModel for Broken {
    fn name(&self) -> String {
        "broken".into()
    }
    fn size(&self) -> usize {
        self.1
    }
    fn run(&self, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        Ok(self.0.clone())
    }
}

#[test]
fn a_misbehaving_model_is_an_error() {
    let rgb = gradient(40, 40);
    let hole = rect_hole(40, 40, 10, 10, 20, 20);
    let region = Region { width: 40, height: 40, rgb: &rgb, hole: &hole };
    let o = Options::default();
    let nan = Broken(vec![f32::NAN; 32 * 32 * 3], 32);
    assert!(matches!(inpaint(&nan, region, &o, &never), Err(MlError::Model(_))));
    let short = Broken(vec![0.5; 10], 32);
    assert!(matches!(inpaint(&short, region, &o, &never), Err(MlError::Model(_))));
    for size in [0, 1, 15, 4097, usize::MAX] {
        assert!(matches!(inpaint(&Broken(vec![], size), region, &o, &never), Err(MlError::Model(_))), "size {size}");
    }
    // Out-of-range values are clamped, not passed on.
    let loud = Broken(vec![9.0; 32 * 32 * 3], 32);
    let out = inpaint(&loud, region, &o, &never).unwrap();
    assert!(out.rgb.iter().all(|v| (0.0..=1.0).contains(v)));
}

#[test]
fn cancelling_stops_the_fill() {
    let rgb = gradient(40, 40);
    let hole = rect_hole(40, 40, 10, 10, 20, 20);
    let m = SolidFill { size: 32, color: RED };
    let r = inpaint(&m, Region { width: 40, height: 40, rgb: &rgb, hole: &hole }, &Options::default(), &|| true);
    assert_eq!(r.unwrap_err(), MlError::Cancelled);
}

#[test]
fn resizing_keeps_constants_and_sizes() {
    let mut p = Plane::new(37, 23, 3);
    p.data.iter_mut().for_each(|v| *v = 0.4);
    for (dw, dh) in [(5, 5), (37, 23), (100, 90), (1, 1), (64, 1)] {
        let r = resize(&p, dw, dh);
        assert_eq!((r.w, r.h, r.data.len()), (dw, dh, dw * dh * 3));
        assert!(r.data.iter().all(|v| (v - 0.4).abs() < 1e-4), "{dw}x{dh}");
    }
}

/// The row-based resize and dilate must equal the obvious per-sample versions.
#[test]
fn fast_resize_and_dilate_match_the_naive_versions() {
    fn naive_resize(src: &Plane, dw: usize, dh: usize) -> Plane {
        let wx = axis_weights(src.w, dw);
        let wy = axis_weights(src.h, dh);
        let mut mid = Plane::new(dw, src.h, src.c);
        for y in 0..src.h {
            for (x, taps) in wx.iter().enumerate() {
                for ch in 0..src.c {
                    mid.set(x, y, ch, taps.iter().map(|&(j, w)| src.at(j, y, ch) * w).sum());
                }
            }
        }
        let mut out = Plane::new(dw, dh, src.c);
        for (y, taps) in wy.iter().enumerate() {
            for x in 0..dw {
                for ch in 0..src.c {
                    out.set(x, y, ch, taps.iter().map(|&(j, w)| mid.at(x, j, ch) * w).sum());
                }
            }
        }
        out
    }
    fn naive_dilate(mask: &Plane, r: usize) -> Plane {
        let mut out = Plane::new(mask.w, mask.h, 1);
        for y in 0..mask.h {
            for x in 0..mask.w {
                let mut m = 0.0f32;
                for yy in y.saturating_sub(r)..=(y + r).min(mask.h - 1) {
                    for xx in x.saturating_sub(r)..=(x + r).min(mask.w - 1) {
                        m = m.max(mask.at(xx, yy, 0));
                    }
                }
                out.set(x, y, 0, m);
            }
        }
        out
    }
    let mut seed = 12345u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    for (sw, sh, dw, dh, c) in [(37, 23, 16, 9, 3), (16, 9, 37, 23, 3), (50, 50, 50, 20, 1), (7, 7, 7, 7, 3), (1, 5, 4, 2, 3)] {
        let mut p = Plane::new(sw, sh, c);
        p.data.iter_mut().for_each(|v| *v = rnd());
        let (fast, slow) = (resize(&p, dw, dh), naive_resize(&p, dw, dh));
        assert_eq!((fast.w, fast.h, fast.c), (slow.w, slow.h, slow.c));
        assert!(fast.data.iter().zip(&slow.data).all(|(a, b)| (a - b).abs() < 1e-5), "resize {sw}x{sh} -> {dw}x{dh}");
    }
    for (w, h, r) in [(20, 13, 3), (5, 5, 16), (1, 9, 2), (9, 1, 1)] {
        let mut m = Plane::new(w, h, 1);
        m.data.iter_mut().for_each(|v| *v = if rnd() > 0.9 { 1.0 } else { 0.0 });
        let want = naive_dilate(&m, r);
        dilate(&mut m, r);
        assert_eq!(m.data, want.data, "dilate {w}x{h} r={r}");
    }
}
