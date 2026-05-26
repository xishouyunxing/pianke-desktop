# Rust-only Release Candidate Checklist

This checklist is the release gate for the Rust-only desktop package. It records what is automated, what is intentionally out of scope, and what still needs human confirmation before calling a build a fully aligned public release.

## Automated Gates

- `npm run smoke:rc` is the one-command release-candidate gate. It runs core/backend tests, Tauri check, UI build, OpenCV ORB release smoke, Rust-only release build, no-Python bundle scan, release backend smoke, and `git diff --check`.
- `npm run build` builds the Rust-only Tauri/NSIS package with OpenCV ORB enabled.
- `npm run smoke:release-rust` verifies the release executable starts without launching a Python backend child process.
- `npm run check:no-python-bundle` fails if release outputs contain Python runtime, Flask worker resources, Python package resources, or missing OpenCV runtime DLLs.

## Current Rust-only Capability Boundary

- Fast is installed-in-package and runs without system Python.
- Watermark is implemented in Rust and is covered by endpoint/wire-shape smoke for templates, preview, batch export, status, cancel error shape, output counts, and output files.
- OpenCV ORB is part of the release build path and has a release smoke test.
- RAW support means embedded JPEG preview extraction; it does not promise LibRaw demosaic.
- HEIC/HEIF support uses Windows WIC and depends on the system codec. Missing codec should produce a clear skipped reason, not a job crash.
- Expert requires the Pianke official ONNX component package. DINOv2 and InsightFace are validated through gated local parity tests when the component and private fixtures are configured.
- MUSIQ and CLIP-IQA+ are optional Expert quality models and should only be presented as available when ONNX parity is passing.
- NIMA is intentionally unavailable in Rust because the Python legacy path replaces the MobileNetV2 classifier without loading a verifiable NIMA-trained head. Rust reports `nima_legacy_unavailable=true` and keeps `aesthetic_score=null`.
- Tycoon supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages compatible providers. Mock E2E is automated; real provider calls require user-supplied credentials and are not part of the default RC gate.

## Gated Local Checks

Run these only on a machine that has the local private Expert component and fixture photos:

```powershell
$env:PIANKE_TYCOON_E2E='1'
$env:PIANKE_EXPERT_COMPONENT_DIR='C:\Users\ero29\Desktop\pianke\.tmp_backend\model_components\expert'
cargo test --release --manifest-path crates\pianke-backend\Cargo.toml tycoon_mock_e2e_runs_with_complete_expert_component_when_configured -- --nocapture
```

Optional open-directory smoke for the watermark output folder:

```powershell
$env:PIANKE_WATERMARK_OPEN_OUT_DIR_SMOKE='1'
cargo test --manifest-path crates\pianke-backend\Cargo.toml rust_watermark_preview_and_batch_export_work_after_fast_selection -- --nocapture
```

## Human Confirmation Before Public Release

- Install the generated NSIS package on a clean Windows user machine without relying on system Python.
- Confirm the large-screen home page folder picker opens the system folder dialog and fills the selected path.
- Run Fast copy and move flows, including undo and reopen.
- Run watermark preview and batch export from Fast winners, then visually confirm templates, typography, logo placement, EXIF text, margins, and output dimensions are acceptable.
- Confirm RAW+JPG+XMP companion behavior with real camera samples.
- Confirm HEIC behavior on a machine with and without the Windows HEIF/HEVC codec.
- Confirm the official Expert component package URL, checksum manifest, CDN speed, and upgrade policy.
- Decide whether real OpenAI/Anthropic-compatible provider calls should be tested before marketing Tycoon as production-ready.

## Non-release Artifacts

Do not commit private photos, ONNX model files, generated parity fixtures, `.tmp_*` directories, or `src-tauri/opencv-runtime/`.
