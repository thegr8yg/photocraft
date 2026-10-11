# Local models (LaMa Content-Aware Fill)

PhotoCraft can fill a selection with **LaMa** (large-mask inpainting, Suvorov et al., WACV 2022), a
neural model that runs **on your own computer** through ONNX Runtime. It is an option next to the
classical Content-Aware Fill (patch completion), which stays the default. Nothing is downloaded,
uploaded or bundled by PhotoCraft: the runtime library and the model file are yours.

> Status: the code path is tested end to end with a stand-in model (all bit depths, gray, CMYK and
> Lab, undo, new-layer output, error cases). It has **not** yet been run against a real LaMa file or
> ONNX Runtime build in CI: do a **Check setup** (below) with yours and report what it says.

## What you need

1. A build with local-model support: `cargo build --release --features local-ml` (desktop app).
   Without the feature the `lama` method answers "this build has no local-model support". The
   feature adds no build-time download: `ort` loads the runtime library when the app runs.
2. **ONNX Runtime** (1.17 or newer), as a shared library:
   - Linux `libonnxruntime.so`, macOS `libonnxruntime.dylib`, Windows `onnxruntime.dll`;
   - from a release archive of <https://github.com/microsoft/onnxruntime/releases>, a package
     manager, or Python's `onnxruntime` wheel (`.../site-packages/onnxruntime/capi/`);
   - for a GPU, take that project's GPU build (CUDA, DirectML on Windows, CoreML on macOS).
     PhotoCraft uses a GPU provider when the library has one and falls back to the CPU.
3. A **LaMa ONNX file** with an `image` and a `mask` input, float32, square (512 × 512 in the common
   exports; an export with a dynamic size is treated as 512). One export is
   <https://huggingface.co/Carve/LaMa-ONNX>. LaMa is Apache-2.0; **check the licence of the
   export you use**, it is not PhotoCraft's. Half-precision (fp16) exports are not supported yet.

## Set it up

Edit › Preferences › **Local Models**:

| Setting | Meaning |
|---|---|
| ONNX Runtime library | Path of the library above. Empty: `PHOTOCRAFT_ORT_LIB`, then `ORT_DYLIB_PATH`, then the system's library search path. |
| LaMa model | Path of the `.onnx` file. |
| LaMa SHA-256 | Optional. When set, any other file is refused. **Check setup** shows the hash of yours. |
| Device | Auto (GPU if the library offers one) or CPU. |
| Content-Aware Fill | Which method fills when a command does not say: Classical or LaMa. |

Press **Check setup**: it hashes the file, loads the runtime and the model, and tells you what it is
running on (or exactly what is wrong). The runtime library can be loaded once per process: to switch
to another library, restart PhotoCraft.

## Use it

- **Edit › Content-Aware Fill…** has a **Method** choice (Classical, LaMa).
- **Delete and Fill Selection** (the selection tools' right-click menu) and Edit › Fill's
  Content-Aware use the method chosen in Preferences.
- Commands and agents: `edit.contentAwareFill` with `{"method": "lama"}`; `ml.status` reports the
  setup (`{"verify": true, "load": true}` to hash and start the model; it also takes `lamaModel`,
  `onnxRuntimeLibrary`, `lamaSha256` and `device` to try settings before saving them).

The first LaMa fill loads the model (seconds); later ones reuse it. The fill is a background job
(progress, cancel) and one undo step.

## How it works, and its limits

- The model sees a square window around the selection: at least its working size (512 px), or the
  selection plus 50 % context on each side, whichever is larger, shrunk to fit and never enlarged. A
  selection much bigger than the model's size is therefore filled from a shrunken view and comes back
  softer; that is the price of a fixed-size model. Split a huge fill into several.
- Only the pixels inside the selection change; the edge feathers by the selection's own coverage.
  The mask given to the model is grown by 3 model pixels so a half-selected edge is repainted.
- It ignores the sampling area and the colour/rotation/scale/mirror options of the classical fill
  (the model decides what to copy); `output` (current/new/duplicate) works.
- Colours go through the engine's sRGB RGBA intermediate (gray, CMYK and Lab included). An RGB
  document in a wide-gamut working space is handed to the model as its stored numbers, not
  converted to sRGB first; a proper colour-managed path is open work.
- Pixels the fill writes are opaque. Transparent areas around the selection look like their stored
  colour to the model (usually black).
- Desktop only. The web build has no ONNX Runtime and reports it.

## Safety

- A wrong library path, a library that is not ONNX Runtime or is too old, a missing, truncated or
  mismatching model file, and a model that returns nonsense are all **errors with a message**, and
  the document is left alone. Calls into `ort` run under a panic guard because `ort` itself panics
  on some bad environments.
- Over automation (MCP, the control channel) the `localModels` preferences and `ml.status`'s `lamaModel` / `onnxRuntimeLibrary` params are refused: choosing which native library the app loads, or which file it reads, is the person's decision at the keyboard, not an agent's. An agent can still run `edit.contentAwareFill` with `method: lama` and `ml.status` against what the person configured.
- A model file is hashed in a stream (no 200 MB read into memory), refused above 4 GiB, and only
  the first successful runtime load is remembered, so fixing a wrong path needs no restart.

## Benchmarking and tuning

`crates/ml/examples/lama_bench.rs` times the whole fill (and the bare model call) across ONNX Runtime
session settings (threads, graph optimisation, spinning, memory pattern), and reports how far each
setting's output drifts from the reference, so a speed-up that changes the picture is caught:

```sh
cargo run --release -p photocraft-ml --features onnx --example lama_bench -- \
    --model lama_fp32.onnx --runtime /path/to/libonnxruntime.so --reps 10 --json out.json
# your own photos (binary P6 .ppm; holes are generated from a fixed seed):  --images DIR
# one or more specific settings instead of the sweep:  --config "threads=4,opt=basic,nospin"
# no model or runtime at all, to check the harness:  --stub
```

Columns: `e2e` is the whole `inpaint` call, `model` one bare model call, `drift dB` the PSNR of the
filled pixels against the first (reference) setting (above ~60 dB = lossless), `truth dB` the PSNR
against the untouched source image (compare *models* with it, on real photos). `photocraft_ml::Tuning`
carries the same settings in code (`OrtLama::load_tuned`); the app still uses ONNX Runtime's defaults.
Status: measured so far only with a small stand-in convolution network, not real LaMa weights.

## For maintainers

- Code: `crates/ml` (L4; the window/scale/composite pipeline, the model contract, the manifest and
  the `onnx` backend), `crates/engine/src/ml_cmds.rs` (preferences, `ml.status`, the model cache,
  the colour conversion) and the `method` branch of `content_aware_fill_as` in
  `edit_menu_cmds.rs`. Everything but the real model is covered by tests; `photocraft_ml::inpaint::SolidFill`
  is the stand-in used by tests of the code around a model.
- Adding another model (MI-GAN, a face landmark network for Face-Aware Liquify): add a
  `ModelSpec` to `crates/ml/src/manifest.rs`, a backend implementing `InpaintModel` (or a sibling
  trait), and a preference; the runtime loading, hashing, status and UI follow the same shape.
- `ml` is in `xtask/src/layers.rs`; `cargo xtask wasm` builds it without `onnx`.
