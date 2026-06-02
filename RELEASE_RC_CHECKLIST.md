# Rust-only Release Candidate Checklist

This checklist is the release gate for the Rust-only desktop package. It records what is automated, what is intentionally out of scope, and what still needs human confirmation before calling a build a fully aligned public release.

## Automated Gates

- `npm run smoke:rc` is the one-command release-candidate gate. It runs core/backend tests, Tauri check, UI build, OpenCV ORB release smoke, Rust-only release build, no-Python bundle scan, release backend smoke, and `git diff --check`.
- `npm run build` builds the Rust-only Tauri/NSIS package with OpenCV ORB enabled.
- `npm run smoke:release-rust` verifies the release executable starts without launching a Python backend child process.
- `npm run check:no-python-bundle` fails if release outputs contain Python runtime, Flask worker resources, Python package resources, or missing OpenCV runtime DLLs.

## 0.3.0 Update Note

0.3.0 focuses on the Rust-only complete-capability release candidate: Expert is split into a standard local model package and an optional quality-model extension, the add-on manager now shows clearer package size and capability status, and the Fast, watermark, Expert local model, component install, and software update flows continue to be tightened for public release.

## Current Rust-only Capability Boundary

- Fast is installed-in-package and runs without system Python.
- Watermark is implemented in Rust and is covered by endpoint/wire-shape smoke for templates, preview, batch export, status, cancel error shape, output counts, and output files.
- OpenCV ORB is part of the release build path and has a release smoke test.
- RAW support means embedded JPEG preview extraction; it does not promise LibRaw demosaic.
- HEIC/HEIF support uses Windows WIC and depends on the system codec. Missing codec should produce a clear skipped reason, not a job crash.
- Expert requires the Pianke official ONNX component package. DINOv2 and InsightFace are validated through gated local parity tests when the component and private fixtures are configured.
- MUSIQ and CLIP-IQA+ are Expert quality models and should only be presented as available when ONNX parity is passing.
- The Python legacy random NIMA classifier is intentionally not replicated. Rust can use the real `nima_vgg16_ava.onnx` AVA weight exported from pyiqa/IQA-PyTorch for free open-source use; without that ONNX file it reports `nima_legacy_unavailable=true` and keeps `aesthetic_score=null`. Optional Inception-AVA, KonIQ, and SPAQ NIMA exports are reported as extra quality signals, not as replacements for `aesthetic_score`.
- Tycoon supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages compatible providers. Mock E2E is automated; real provider calls require user-supplied credentials and are not part of the default RC gate.
- Software update checking is prompt-only. The app reads `https://pianke.moeuu.cn/pianke/desktop/latest.json`, shows a small "有新版本" button when the remote version is newer, and opens the download URL when clicked. It does not perform silent auto-update.

## Official Server Upload Layout

Upload the desktop software update manifest and installer to:

```text
/pianke/desktop/latest.json
/pianke/desktop/片刻桌面版_<version>_x64-setup.exe
```

`latest.json` must use this shape:

```json
{
  "version": "0.3.0",
  "url": "https://pianke.moeuu.cn/pianke/desktop/片刻桌面版_0.3.0_x64-setup.exe",
  "notes": "Expert 拆分为标准组件与可选质量模型扩展，降低首次下载体积；附加组件管理增加包体说明和能力状态；继续完善 Rust-only Fast、水印、Expert 本地模型与更新检测体验。",
  "published_at": "2026-06-02"
}
```

Upload the Expert standard component package to:

```text
/pianke/components/expert/onnx-v1/component.json
/pianke/components/expert/onnx-v1/models/dinov2-small.onnx
/pianke/components/expert/onnx-v1/models/insightface/det_10g.onnx
/pianke/components/expert/onnx-v1/models/insightface/w600k_r50.onnx
/pianke/components/expert/onnx-v1/models/insightface/1k3d68.onnx
```

Upload the optional Expert quality extension package to:

```text
/pianke/components/expert-quality/onnx-v1/component.json
/pianke/components/expert-quality/onnx-v1/quality_preprocessor.json
/pianke/components/expert-quality/onnx-v1/models/quality/musiq.onnx
/pianke/components/expert-quality/onnx-v1/models/quality/clipiqa_plus.onnx
/pianke/components/expert-quality/onnx-v1/models/quality/nima_vgg16_ava.onnx
/pianke/components/expert-quality/onnx-v1/models/quality/nima_inception_ava.onnx
/pianke/components/expert-quality/onnx-v1/models/quality/nima_koniq.onnx
/pianke/components/expert-quality/onnx-v1/models/quality/nima_spaq.onnx
```

The standard Expert package is about 423 MB. The optional quality extension is about 1.1 GB when all NIMA variants are included.

The desktop app installs Expert standard models from `https://pianke.moeuu.cn/pianke/components/expert/onnx-v1/component.json` by default. The optional quality extension uses `https://pianke.moeuu.cn/pianke/components/expert-quality/onnx-v1/component.json`. Development builds can still override these with `PIANKE_EXPERT_*` and `PIANKE_EXPERT_QUALITY_*` environment variables.

## Gated Local Checks

Run these only on a machine that has the local private Expert component and fixture photos:

```powershell
$env:PIANKE_TYCOON_E2E='1'
$env:PIANKE_EXPERT_COMPONENT_DIR='C:\Users\ero29\Desktop\pianke\.tmp_backend\model_components\expert'
$env:PIANKE_EXPERT_QUALITY_COMPONENT_DIR='C:\Users\ero29\Desktop\pianke\.tmp_backend\model_components\expert-quality'
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
