# Expert parity fixtures

This directory is used for local Python-vs-Rust Expert parity JSON.

The generated `*.json` files may reference private photos under
`.tmp_expert_parity/private_photos`, so they are ignored by Git by default.
Generate them with:

```powershell
python scripts\export_dinov2_golden.py .tmp_expert_parity\private_photos fixtures\expert_parity\dinov2.json --limit 10
python scripts\export_dinov2_golden.py .tmp_expert_parity\private_photos fixtures\expert_parity\insightface.json --faces --limit 10
python scripts\export_dinov2_golden.py .tmp_expert_parity\private_photos fixtures\expert_parity\quality.json --faces --quality --limit 10
```

Run parity checks with:

```powershell
$env:PIANKE_EXPERT_COMPONENT_DIR="C:\Users\ero29\Desktop\pianke\.tmp_backend\model_components\expert"
cargo test --manifest-path crates\pianke-backend\Cargo.toml dinov2_golden_fixture_matches_when_configured -- --nocapture
cargo test --manifest-path crates\pianke-backend\Cargo.toml insightface_golden_fixture_matches_when_configured -- --nocapture
cargo test --manifest-path crates\pianke-backend\Cargo.toml quality_golden_fixture_matches_when_configured -- --nocapture
```

Quality parity is gated: it only runs when `models/quality/musiq.onnx`,
`models/quality/clipiqa_plus.onnx`, and `fixtures/expert_parity/quality.json`
exist. The legacy Python NIMA classifier is intentionally not reproduced.
