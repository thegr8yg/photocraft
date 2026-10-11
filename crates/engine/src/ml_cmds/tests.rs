use super::*;
use photocraft_ml::inpaint::SolidFill;

const BLUE: [f32; 3] = [0.0, 0.0, 1.0];

/// A 128×96 white document with a red 20×20 square at (40,30), selected.
fn session(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 128, "height": 96, "depth": depth, "background": "white"})).unwrap();
    s.execute("select.rect", json!({"x": 40, "y": 30, "width": 20, "height": 20})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
    s
}

fn with_model(s: &mut Session) {
    s.ml.set_stand_in(Arc::new(SolidFill { size: 64, color: BLUE }));
}

fn px(s: &Session, x: i32, y: i32) -> Vec<f32> {
    let st = s.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().pixel(x, y)
}

fn undo_steps(s: &Session) -> usize {
    s.active().unwrap().history.past_len()
}

#[test]
fn lama_fills_the_selection_with_the_model_at_all_depths() {
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        with_model(&mut s);
        let before = undo_steps(&s);
        let r = s.execute("edit.contentAwareFill", json!({"method": "lama"})).unwrap();
        assert_eq!(r["method"], "lama");
        assert!(r["filled"].as_u64().unwrap() >= 300, "{r}");
        let c = px(&s, 50, 40);
        assert!(c[0] < 0.1 && c[1] < 0.1 && c[2] > 0.9, "depth {depth}: the hole is the model's colour, got {c:?}");
        assert_eq!(px(&s, 5, 5), vec![1.0, 1.0, 1.0, 1.0][..px(&s, 5, 5).len()], "depth {depth}: outside the selection is untouched");
        assert_eq!(undo_steps(&s), before + 1, "one undo step");
        assert_eq!(s.active().unwrap().history.undo_label(), Some("Content-Aware Fill"));
        s.undo();
        let r = px(&s, 50, 40);
        assert!(r[0] > 0.9 && r[1] < 0.1, "depth {depth}: undo restores the red square, got {r:?}");
    }
}

#[test]
fn lama_can_write_a_new_layer() {
    let mut s = session(8);
    with_model(&mut s);
    let base = s.active().unwrap().active_layer.unwrap();
    let r = s.execute("edit.contentAwareFill", json!({"method": "lama", "output": "new"})).unwrap();
    let nid = photocraft_doc::LayerId(r["layer"].as_u64().unwrap());
    assert_ne!(nid, base);
    let b = s.active().unwrap().doc.layer(nid).unwrap().surface().unwrap().content_bounds();
    assert!(photocraft_geom::Rect::new(38, 28, 62, 52).contains_rect(&b), "{b:?}");
}

#[test]
fn the_preference_picks_the_method_when_the_command_does_not() {
    let mut s = session(8);
    with_model(&mut s);
    assert_eq!(s.execute("edit.contentAwareFill", json!({})).unwrap()["method"], "classical");
    s.undo();
    s.execute("prefs.set", json!({"path": "localModels.contentAwareFill", "value": "lama"})).unwrap();
    assert_eq!(s.execute("edit.contentAwareFill", json!({})).unwrap()["method"], "lama");
    s.undo();
    // Delete and Fill Selection (the right-click item) follows the preference too.
    assert_eq!(s.execute("edit.deleteAndFillSelection", json!({})).unwrap()["method"], "lama");
    // An explicit method wins.
    s.undo();
    assert_eq!(s.execute("edit.contentAwareFill", json!({"method": "classical"})).unwrap()["method"], "classical");
}

#[test]
fn without_a_model_lama_explains_what_to_do_and_changes_nothing() {
    let mut s = session(8);
    let before = undo_steps(&s);
    let err = s.execute("edit.contentAwareFill", json!({"method": "lama"})).unwrap_err().to_string();
    assert!(err.contains("classical"), "{err}");
    assert!(err.contains("Local Models") || err.contains("local-model"), "{err}");
    assert_eq!(undo_steps(&s), before);
    assert!(px(&s, 50, 40)[0] > 0.9, "the document is unchanged");
    // A configured path to nothing is an error too (or "no support" in a build without ONNX).
    s.execute("prefs.set", json!({"path": "localModels.lamaModel", "value": "/nonexistent/lama.onnx"})).unwrap();
    s.execute("prefs.set", json!({"path": "localModels.onnxRuntimeLibrary", "value": "/nonexistent/libonnxruntime.so"})).unwrap();
    assert!(s.execute("edit.contentAwareFill", json!({"method": "lama"})).is_err());
    assert_eq!(undo_steps(&s), before);
}

#[test]
fn bad_method_values_are_errors() {
    let mut s = session(8);
    with_model(&mut s);
    for m in [json!("gan"), json!(3), json!(["lama"]), json!(true), json!({"a": 1})] {
        assert!(s.execute("edit.contentAwareFill", json!({"method": m})).is_err(), "{m}");
    }
}

struct Nan;
impl InpaintModel for Nan {
    fn name(&self) -> String {
        "nan".into()
    }
    fn size(&self) -> usize {
        32
    }
    fn run(&self, _: &[f32], _: &[f32]) -> photocraft_ml::Result<Vec<f32>> {
        Ok(vec![f32::NAN; 32 * 32 * 3])
    }
}

#[test]
fn a_misbehaving_model_leaves_the_document_alone() {
    let mut s = session(8);
    s.ml.set_stand_in(Arc::new(Nan));
    let before = undo_steps(&s);
    assert!(s.execute("edit.contentAwareFill", json!({"method": "lama"})).is_err());
    assert_eq!(undo_steps(&s), before);
    assert!(px(&s, 50, 40)[0] > 0.9);
}

#[test]
fn lama_handles_other_colour_modes() {
    let mut ran = 0;
    for mode in ["grayscale", "cmyk", "lab"] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 64, "background": "white"})).unwrap();
        s.execute(&format!("image.mode.{mode}"), json!({})).unwrap();
        ran += 1;
        s.execute("select.rect", json!({"x": 20, "y": 20, "width": 10, "height": 10})).unwrap();
        with_model(&mut s);
        let r = s.execute("edit.contentAwareFill", json!({"method": "lama"}));
        assert!(r.is_ok(), "{mode}: {r:?}");
        let v = px(&s, 25, 25);
        assert!(v.iter().all(|c| c.is_finite()), "{mode}: {v:?}");
    }
    assert_eq!(ran, 3, "every colour mode was exercised");
}

#[test]
fn local_model_preferences_are_validated() {
    let mut s = Session::new();
    for (path, value) in [
        ("localModels.device", json!("gpu")),
        ("localModels.device", json!(1)),
        ("localModels.contentAwareFill", json!("magic")),
        ("localModels.lamaModel", json!(5)),
        ("localModels.lamaModel", json!("a\0b")),
        ("localModels.lamaModel", json!("x".repeat(5000))),
        ("localModels.onnxRuntimeLibrary", json!(null)),
    ] {
        assert!(s.execute("prefs.set", json!({"path": path, "value": value})).is_err(), "{path} = {value}");
    }
    s.execute("prefs.set", json!({"path": "localModels.device", "value": "cpu"})).unwrap();
    s.execute("prefs.reset", json!({"path": "localModels"})).unwrap();
    assert_eq!(s.prefs.get().local_models, LocalModels::default());
}

#[test]
fn status_reports_what_is_set_up() {
    let mut s = Session::new();
    let r = s.execute("ml.status", json!({})).unwrap();
    assert_eq!(r["configured"], false);
    assert_eq!(r["lama"]["set"], false);
    assert_eq!(r["model"]["id"], "lama");
    assert_eq!(r["compiled"], photocraft_ml::onnx_compiled());
    // A file that is there: found, and with `verify` its hash.
    let dir = std::env::temp_dir().join(format!("photocraft-ml-status-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("lama.onnx");
    std::fs::write(&file, b"not really a model").unwrap();
    s.execute("prefs.set", json!({"path": "localModels.lamaModel", "value": file.to_string_lossy()})).unwrap();
    let r = s.execute("ml.status", json!({"verify": true})).unwrap();
    assert_eq!(r["lama"]["found"], true);
    assert_eq!(r["lama"]["bytes"], 18);
    assert_eq!(r["lama"]["sha256"].as_str().map(str::len), Some(64));
    // A pin that does not match is reported, not accepted.
    s.execute("prefs.set", json!({"path": "localModels.lamaSha256", "value": "00".repeat(32)})).unwrap();
    let r = s.execute("ml.status", json!({"verify": true})).unwrap();
    assert_eq!(r["lama"]["pinMatches"], false);
    // Loading junk is an error value, never a crash.
    let r = s.execute("ml.status", json!({"load": true})).unwrap();
    assert!(r.get("loadError").is_some() || r.get("loaded").is_some(), "{r}");
    // Unsaved settings can be tried; bad ones are errors.
    let r = s.execute("ml.status", json!({"lamaModel": "/nonexistent/other.onnx", "device": "cpu"})).unwrap();
    assert_eq!(r["lama"]["found"], false);
    assert_eq!(r["device"], "cpu");
    assert!(s.execute("ml.status", json!({"lamaModel": 5})).is_err());
    assert!(s.execute("ml.status", json!({"device": "tpu"})).is_err());
    assert!(s.execute("ml.status", json!({"lamaModel": "a\0b"})).is_err());
    // Hostile params.
    for p in [json!(null), json!([]), json!({"verify": "yes", "load": 3}), json!(5)] {
        let _ = s.execute("ml.status", p);
    }
    std::fs::remove_dir_all(&dir).ok();
}
