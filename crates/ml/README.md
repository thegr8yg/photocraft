# photocraft-ml

Local machine-learning models for PhotoCraft (layer L4, no workspace dependencies). Today: **LaMa
inpainting** for Content-Aware Fill. User guide and limits: [`docs/local-models.md`](../../docs/local-models.md).

```rust
use photocraft_ml::{inpaint, Options, Region};
// `model` is any `InpaintModel` (the ONNX one is `OrtLama`, feature `onnx`).
let out = inpaint(&model, Region { width, height, rgb: &rgb, hole: &hole }, &Options::default(), &|| cancelled())?;
// out.rgb: the region with only the hole's pixels replaced.
```

- `inpaint`: the model contract (`InpaintModel`) and the pipeline: choose a window around the hole,
  scale it to the model's square size (never up), pad, run, write back only the hole. Pure Rust,
  builds for wasm, tested with `SolidFill`.
- `manifest`: `ModelSpec`s (id, source, licence, optional pinned SHA-256/size) and streamed file
  hashing/checking. Nothing is downloaded or bundled.
- `onnx` (feature `onnx`, native): `OrtLama`, ONNX Runtime through `ort` with `load-dynamic`, so a
  build needs no runtime download; `load_runtime(path)` loads the library once, and every call
  into `ort` is under a panic guard (a wrong library is an `MlError`, never a crash).

Errors are `MlError` values; nothing here panics on a bad file, size, buffer or model.
