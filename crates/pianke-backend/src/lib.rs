use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::oneshot;

mod clustering;
mod job;
mod quality;
mod routes;
mod session;
mod types;
mod util;

mod model_components;
use model_components::ModelManager;

mod expert_vision;
mod llm_provider;
use llm_provider::LlmProviderManager;
mod watermark;

use routes::build_router;
use types::*;
pub use types::{ServerHandle, ServerOptions};

const STATE_FILENAME: &str = ".pic_selecter_state.json";
const INFO_CACHE_FILENAME: &str = ".pic_selecter_infos.json";
const PIC_DIR: &str = "_pic_selecter";
const THUMB_MAX: u32 = 1600;
const ANALYSIS_MAX_SIDE: u32 = 2048;
const DEFAULT_APP_UPDATE_URL: &str = "https://pianke.moeuu.cn/pianke/desktop/latest.json";

const IMAGE_EXTS: &[&str] = &[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"];
const RAW_EXTS: &[&str] = &[
    ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng", ".raf", ".orf", ".rw2",
    ".pef", ".rwl", ".srw", ".x3f",
];
const HEIC_EXTS: &[&str] = &[".heic", ".heif"];
const SIDECAR_EXTS: &[&str] = &[".xmp"];

pub fn start(options: ServerOptions) -> Result<ServerHandle, String> {
    let addr: SocketAddr = format!("127.0.0.1:{}", options.port)
        .parse()
        .map_err(|e| format!("invalid backend address: {e}"))?;
    let listener = TcpListener::bind(addr).map_err(|e| format!("bind Rust backend failed: {e}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("configure backend listener failed: {e}"))?;
    let ctx = AppCtx {
        inner: Arc::new(Mutex::new(AppState::default())),
        token: options.token,
        models: ModelManager::new(options.backend_dir.join("model_components")),
        llm: LlmProviderManager::new(options.backend_dir.join("llm_provider")),
        backend_dir: options.backend_dir,
        folder_picker: options.folder_picker,
    };
    let app = build_router(ctx);
    let (tx, rx) = oneshot::channel::<()>();
    let thread = thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().expect("create backend runtime");
        runtime.block_on(async move {
            let listener =
                tokio::net::TcpListener::from_std(listener).expect("create tokio listener");
            let server = axum::serve(listener, app).with_graceful_shutdown(async {
                let _ = rx.await;
            });
            if let Err(err) = server.await {
                eprintln!("[rust-backend] server error: {err}");
            }
        });
    });

    Ok(ServerHandle {
        shutdown: Some(tx),
        thread: Some(thread),
    })
}

// Re-export internal items for tests
#[cfg(test)]
use clustering::{expert_pair_similarity, face_overlap_similarity};
#[cfg(test)]
use job::{extract_embedded_jpeg_preview, process_one, scan_folder};
#[cfg(test)]
use session::{advance, unique_target};
#[cfg(test)]
use util::{format_kind_for_path, heic_decode_available};

#[cfg(all(test, feature = "opencv-orb"))]
use job::{compute_orb_inliers_for_records, load_fast_analysis_image};
#[cfg(all(test, feature = "opencv-orb"))]
use quality::quality_signals;

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat};
    use pianke_core::fast::{ExifSummary, FastImageInfo};
    use serde::Deserialize;
    #[cfg(feature = "opencv-orb")]
    use serde_json::Value;
    use std::{
        collections::HashMap,
        fs,
        io::Cursor,
        path::{Path, PathBuf},
    };

    #[test]
    fn scan_pairs_raw_with_same_stem_jpg() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("IMG_0001.CR2");
        let jpg = dir.path().join("IMG_0001.JPG");
        fs::write(&raw, b"raw").expect("write raw");
        fs::write(&jpg, b"jpg").expect("write jpg");

        let pairs = scan_folder(dir.path());
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].primary, raw);
        assert_eq!(pairs[0].analysis.as_ref(), Some(&jpg));
        assert_eq!(pairs[0].companions, vec![jpg]);
    }

    #[test]
    fn raw_embedded_jpeg_preview_classifies_failures() {
        let missing = extract_embedded_jpeg_preview(b"fake raw without preview")
            .expect_err("missing embedded jpeg should fail");
        assert!(matches!(missing, RawPreviewError::NoEmbeddedJpeg));
        assert!(missing.to_message().contains("RAW"));

        let broken = extract_embedded_jpeg_preview(b"raw\xff\xd8not a jpeg\xff\xd9tail")
            .expect_err("broken embedded jpeg should fail");
        assert!(matches!(broken, RawPreviewError::DecodeFailed(_)));
        assert!(broken.to_message().contains("RAW"));

        let mut cursor = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(8, 8, image::Rgb([4, 8, 16])))
            .write_to(&mut cursor, ImageFormat::Jpeg)
            .expect("encode embedded jpeg");
        let jpeg = cursor.into_inner();
        let mut raw = b"raw header".to_vec();
        raw.extend_from_slice(&jpeg);
        raw.extend_from_slice(b"raw trailer");
        let extracted =
            extract_embedded_jpeg_preview(&raw).expect("valid embedded jpeg should be found");
        assert_eq!(extracted, jpeg.as_slice());
    }

    #[test]
    fn unique_target_suffixes_existing_names() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("a.jpg"), b"x").expect("write existing");
        assert_eq!(
            unique_target(dir.path(), "a.jpg"),
            dir.path().join("a_1.jpg")
        );
    }

    #[test]
    fn advance_pick_left_finishes_two_image_group() {
        let mut group = GroupState::new(vec!["a.jpg".to_string(), "b.jpg".to_string()]);
        advance(&mut group, "right");
        assert!(group.finished);
        assert_eq!(group.winner.as_deref(), Some("a.jpg"));
        assert_eq!(group.losers, vec!["b.jpg"]);
    }

    #[test]
    fn face_overlap_matches_arcface_threshold_logic() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![0.99, 0.01], vec![1.0, 0.0]];
        assert_eq!(face_overlap_similarity(&a, &[]), 0.0);
        assert_eq!(face_overlap_similarity(&[], &[]), 1.0);
        assert!((face_overlap_similarity(&a, &b) - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn expert_pair_similarity_uses_face_signal_for_portraits() {
        let mut a = FastImageInfo {
            path: "IMG_0001.jpg".to_string(),
            mtime: Some(100.0),
            dinov2: Some(vec![0.9, 0.1]),
            face_embeddings: vec![vec![1.0, 0.0]],
            ..FastImageInfo::default()
        };
        let mut b = FastImageInfo {
            path: "IMG_0002.jpg".to_string(),
            mtime: Some(101.0),
            dinov2: Some(vec![0.9, 0.1]),
            face_embeddings: vec![vec![1.0, 0.0]],
            ..FastImageInfo::default()
        };
        let same_face = expert_pair_similarity(&a, &b);
        b.face_embeddings = vec![vec![0.0, 1.0]];
        let different_face = expert_pair_similarity(&a, &b);
        assert!(same_face > different_face);
        assert!(same_face > 0.8);
        a.face_embeddings.clear();
        b.face_embeddings.clear();
        assert!(expert_pair_similarity(&a, &b) > 0.5);
    }

    #[test]
    fn fast_format_fixture_matches_when_configured() {
        #[derive(Deserialize)]
        struct Fixture {
            #[serde(default)]
            source_folder: Option<String>,
            #[serde(default)]
            images: Vec<FixtureImage>,
            #[serde(default)]
            skipped: Vec<FixtureSkipped>,
        }

        #[derive(Deserialize)]
        struct FixtureImage {
            path: String,
            #[serde(default)]
            format_kind: Option<String>,
            #[serde(default)]
            companions: Vec<String>,
            #[serde(default)]
            companion_kinds: Vec<String>,
        }

        #[derive(Deserialize)]
        struct FixtureSkipped {
            path: String,
            #[serde(default)]
            format_kind: Option<String>,
        }

        fn resolve_fixture_path(
            workspace: &Path,
            fixture_dir: &Path,
            source_folder: Option<&str>,
            path: &str,
        ) -> PathBuf {
            let raw = PathBuf::from(path);
            if raw.is_absolute() {
                return raw;
            }
            let workspace_path = workspace.join(&raw);
            if workspace_path.exists() {
                return workspace_path;
            }
            if let Some(source_folder) = source_folder {
                let source = PathBuf::from(source_folder);
                let source = if source.is_absolute() {
                    source
                } else {
                    workspace.join(source)
                };
                let source_path = source.join(&raw);
                if source_path.exists() {
                    return source_path;
                }
            }
            fixture_dir.join(raw)
        }

        fn path_key(path: &Path) -> String {
            path.to_string_lossy().replace('\\', "/").to_lowercase()
        }

        let Ok(fixture_path) = std::env::var("PIANKE_FAST_FORMAT_FIXTURE") else {
            return;
        };
        let fixture_path = PathBuf::from(fixture_path);
        if !fixture_path.exists() {
            return;
        }
        let text = fs::read_to_string(&fixture_path).expect("read fast format fixture");
        let fixture: Fixture = serde_json::from_str(&text).expect("parse fast format fixture");
        if fixture.images.is_empty() && fixture.skipped.is_empty() {
            return;
        }

        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let fixture_dir = fixture_path.parent().unwrap_or(Path::new("."));
        let source_folder = fixture
            .source_folder
            .as_deref()
            .map(|folder| resolve_fixture_path(&workspace, fixture_dir, None, folder))
            .unwrap_or_else(|| fixture_dir.to_path_buf());
        if !source_folder.exists() {
            return;
        }

        let pairs = scan_folder(&source_folder);
        let pair_by_primary = pairs
            .iter()
            .map(|pair| (path_key(&pair.primary), pair))
            .collect::<HashMap<_, _>>();

        let mut checked = 0usize;
        for image in fixture.images {
            let path = resolve_fixture_path(
                &workspace,
                fixture_dir,
                fixture.source_folder.as_deref(),
                &image.path,
            );
            let Some(pair) = pair_by_primary.get(&path_key(&path)) else {
                continue;
            };
            let companion_kinds = pair
                .companions
                .iter()
                .map(|p| format_kind_for_path(p).to_string())
                .collect::<Vec<_>>();
            if !image.companion_kinds.is_empty() {
                assert_eq!(
                    companion_kinds,
                    image.companion_kinds,
                    "companion kind mismatch for {}",
                    path.display()
                );
            }
            if !image.companions.is_empty() {
                assert_eq!(
                    pair.companions.len(),
                    image.companions.len(),
                    "companion count mismatch for {}",
                    path.display()
                );
            }
            let record = process_one(pair, "standard");
            match image.format_kind.as_deref() {
                Some("heic") if heic_decode_available() => assert!(
                    record.is_ok(),
                    "HEIC fixture should decode when WIC HEIF capability is enabled: {}",
                    path.display()
                ),
                Some("heic") => assert!(
                    record
                        .as_ref()
                        .err()
                        .is_some_and(|reason| reason.contains("HEIC/HEIF")),
                    "HEIC fixture should report a clear skip when capability is unavailable"
                ),
                Some("raw") => {
                    assert!(
                        record.is_ok() || pair.analysis.as_ref() == Some(&pair.primary),
                        "RAW fixture should be queued for embedded-preview analysis when no JPG companion exists"
                    );
                }
                _ => assert!(
                    record.is_ok(),
                    "image fixture should process: {}",
                    path.display()
                ),
            }
            checked += 1;
        }

        for skipped in fixture.skipped {
            let path = resolve_fixture_path(
                &workspace,
                fixture_dir,
                fixture.source_folder.as_deref(),
                &skipped.path,
            );
            let Some(pair) = pair_by_primary.get(&path_key(&path)) else {
                continue;
            };
            let result = process_one(pair, "standard");
            match skipped.format_kind.as_deref() {
                Some("raw") => assert!(
                    result
                        .as_ref()
                        .err()
                        .is_some_and(|reason| reason.contains("RAW")),
                    "skipped RAW should return RAW-specific reason"
                ),
                Some("heic") => assert!(
                    result
                        .as_ref()
                        .err()
                        .is_some_and(|reason| reason.contains("HEIC/HEIF")),
                    "skipped HEIC should return HEIC/HEIF-specific reason"
                ),
                _ => {}
            }
            checked += 1;
        }
        assert!(checked > 0, "fast format fixture had no comparable entries");
    }

    #[test]
    fn fast_analysis_hash_fixture_matches_when_configured() {
        #[derive(Deserialize)]
        struct Fixture {
            #[serde(default)]
            source_folder: Option<String>,
            #[serde(default)]
            images: Vec<FixtureImage>,
        }

        #[derive(Deserialize)]
        struct FixtureImage {
            path: String,
            #[serde(default)]
            hashes: FixtureHashes,
            #[serde(default)]
            color_hist: Option<Vec<f32>>,
            #[serde(default)]
            exif_summary: Option<ExifSummary>,
        }

        #[derive(Default, Deserialize)]
        struct FixtureHashes {
            #[serde(default)]
            phash: Option<String>,
            #[serde(default)]
            dhash: Option<String>,
            #[serde(default)]
            whash: Option<String>,
            #[serde(default)]
            ahash: Option<String>,
        }

        fn resolve_fixture_path(
            workspace: &Path,
            fixture_dir: &Path,
            source_folder: Option<&str>,
            path: &str,
        ) -> PathBuf {
            let raw = PathBuf::from(path);
            if raw.is_absolute() {
                return raw;
            }
            let workspace_path = workspace.join(&raw);
            if workspace_path.exists() {
                return workspace_path;
            }
            if let Some(source_folder) = source_folder {
                let source = PathBuf::from(source_folder);
                let source = if source.is_absolute() {
                    source
                } else {
                    workspace.join(source)
                };
                let source_path = source.join(&raw);
                if source_path.exists() {
                    return source_path;
                }
            }
            fixture_dir.join(raw)
        }

        let fixture_path = std::env::var("PIANKE_FAST_ANALYSIS_FIXTURE")
            .or_else(|_| std::env::var("PIANKE_FAST_ORB_FIXTURE"));
        let Ok(fixture_path) = fixture_path else {
            return;
        };
        let fixture_path = PathBuf::from(fixture_path);
        if !fixture_path.exists() {
            return;
        }
        let text = fs::read_to_string(&fixture_path).expect("read fast analysis fixture");
        let fixture: Fixture = serde_json::from_str(&text).expect("parse fast analysis fixture");
        if fixture.images.is_empty() {
            return;
        }
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let fixture_dir = fixture_path.parent().unwrap_or(Path::new("."));
        for image in fixture.images {
            let path = resolve_fixture_path(
                &workspace,
                fixture_dir,
                fixture.source_folder.as_deref(),
                &image.path,
            );
            let pair = ScanPair {
                primary: path.clone(),
                companions: Vec::new(),
                analysis: Some(path.clone()),
            };
            let record = process_one(&pair, "standard").expect("process fast fixture image");
            assert_eq!(
                record.info.phash,
                image.hashes.phash,
                "pHash mismatch for {}",
                path.display()
            );
            assert_eq!(
                record.info.dhash,
                image.hashes.dhash,
                "dHash mismatch for {}",
                path.display()
            );
            assert_eq!(
                record.info.whash,
                image.hashes.whash,
                "wHash mismatch for {}",
                path.display()
            );
            assert_eq!(
                record.info.ahash,
                image.hashes.ahash,
                "aHash mismatch for {}",
                path.display()
            );
            if let Some(expected_exif) = image.exif_summary {
                assert_eq!(
                    record
                        .info
                        .exif_summary
                        .as_ref()
                        .and_then(|exif| exif.width),
                    expected_exif.width,
                    "analysis width mismatch for {}",
                    path.display()
                );
                assert_eq!(
                    record
                        .info
                        .exif_summary
                        .as_ref()
                        .and_then(|exif| exif.height),
                    expected_exif.height,
                    "analysis height mismatch for {}",
                    path.display()
                );
            }
            if let (Some(actual), Some(expected)) = (&record.info.color_hist, image.color_hist) {
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "HSV length for {}",
                    path.display()
                );
                let max_delta = actual
                    .iter()
                    .zip(expected.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_delta <= 0.00001,
                    "HSV histogram mismatch for {}: max_delta={max_delta}",
                    path.display()
                );
            }
        }
    }

    #[cfg(feature = "opencv-orb")]
    #[test]
    fn fast_quality_fixture_matches_when_configured() {
        #[derive(Deserialize)]
        struct Fixture {
            #[serde(default)]
            source_folder: Option<String>,
            #[serde(default)]
            strength: Option<String>,
            #[serde(default)]
            images: Vec<FixtureImage>,
        }

        #[derive(Deserialize)]
        struct FixtureImage {
            path: String,
            #[serde(default)]
            companions: Vec<String>,
            #[serde(default)]
            quality: Option<Value>,
            #[serde(default)]
            quality_signals: Option<Value>,
        }

        fn resolve_path(
            workspace: &Path,
            fixture_dir: &Path,
            source_folder: Option<&str>,
            path: &str,
        ) -> PathBuf {
            let raw = PathBuf::from(path);
            if raw.is_absolute() {
                return raw;
            }
            let workspace_path = workspace.join(&raw);
            if workspace_path.exists() {
                return workspace_path;
            }
            if let Some(source_folder) = source_folder {
                let source = PathBuf::from(source_folder);
                let source = if source.is_absolute() {
                    source
                } else {
                    workspace.join(source)
                };
                let source_path = source.join(&raw);
                if source_path.exists() {
                    return source_path;
                }
            }
            fixture_dir.join(raw)
        }

        fn value_f64(value: &Value, key: &str) -> Option<f64> {
            value.get(key).and_then(Value::as_f64)
        }

        fn assert_close(label: &str, actual: Option<f64>, expected: Option<f64>, tolerance: f64) {
            match (actual, expected) {
                (Some(actual), Some(expected)) => assert!(
                    (actual - expected).abs() <= tolerance,
                    "{label}: actual={actual}, expected={expected}, tolerance={tolerance}"
                ),
                (None, None) => {}
                other => panic!("{label}: optional mismatch {other:?}"),
            }
        }

        let Ok(fixture_path) = std::env::var("PIANKE_FAST_QUALITY_FIXTURE") else {
            return;
        };
        let fixture_path = PathBuf::from(fixture_path);
        if !fixture_path.exists() {
            return;
        }
        let text = fs::read_to_string(&fixture_path).expect("read fast quality fixture");
        let fixture: Fixture = serde_json::from_str(&text).expect("parse fast quality fixture");
        if fixture.images.is_empty() {
            return;
        }
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let fixture_dir = fixture_path.parent().unwrap_or(Path::new("."));
        let strength = fixture.strength.as_deref().unwrap_or("standard");

        let mut checked = 0usize;
        for image in fixture.images {
            let path = resolve_path(
                &workspace,
                fixture_dir,
                fixture.source_folder.as_deref(),
                &image.path,
            );
            if !path.exists() {
                continue;
            }
            let companions = image
                .companions
                .iter()
                .map(|path| {
                    resolve_path(
                        &workspace,
                        fixture_dir,
                        fixture.source_folder.as_deref(),
                        path,
                    )
                })
                .collect::<Vec<_>>();
            let pair = ScanPair {
                primary: path.clone(),
                analysis: Some(path.clone()),
                companions,
            };
            let record = process_one(&pair, strength).expect("process fast quality fixture image");
            let actual_quality = record.info.quality.expect("quality output");
            if let Some(expected_quality) = image.quality {
                assert_eq!(
                    actual_quality.flags,
                    expected_quality
                        .get("flags")
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(Value::as_str)
                                .map(str::to_string)
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default(),
                    "flags mismatch for {}",
                    path.display()
                );
                assert_eq!(
                    actual_quality.auto_reject,
                    expected_quality.get("auto_reject").and_then(Value::as_bool),
                    "auto_reject mismatch for {}",
                    path.display()
                );
                assert_eq!(
                    actual_quality.reject_reason.as_deref(),
                    expected_quality
                        .get("reject_reason")
                        .and_then(Value::as_str),
                    "reject_reason mismatch for {}",
                    path.display()
                );
                assert_close(
                    &format!("quality_score {}", path.display()),
                    actual_quality.quality_score,
                    value_f64(&expected_quality, "quality_score"),
                    0.50,
                );
                assert_close(
                    &format!("blur_score {}", path.display()),
                    actual_quality.blur_score,
                    value_f64(&expected_quality, "blur_score"),
                    1.0,
                );
                assert_close(
                    &format!("brightness_mean {}", path.display()),
                    actual_quality.brightness_mean,
                    value_f64(&expected_quality, "brightness_mean"),
                    0.02,
                );
                assert_close(
                    &format!("entropy {}", path.display()),
                    actual_quality.entropy,
                    value_f64(&expected_quality, "entropy"),
                    0.0005,
                );
                assert_close(
                    &format!("blur_combined {}", path.display()),
                    actual_quality.blur_combined,
                    value_f64(&expected_quality, "blur_combined"),
                    0.02,
                );
                assert_close(
                    &format!("edge_width_pix {}", path.display()),
                    actual_quality.edge_width_pix,
                    value_f64(&expected_quality, "edge_width_pix"),
                    0.50,
                );
                assert_close(
                    &format!("horizon_tilt_deg {}", path.display()),
                    actual_quality.horizon_tilt_deg,
                    value_f64(&expected_quality, "horizon_tilt_deg"),
                    0.25,
                );
            }

            if let Some(expected_signals) = image.quality_signals {
                let analysis = load_fast_analysis_image(&path).expect("load analysis image");
                let file_size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let signals = quality_signals(&analysis, file_size);
                assert_close(
                    &format!("signal lap {}", path.display()),
                    Some(signals.blur_score),
                    value_f64(&expected_signals, "lap"),
                    1.0,
                );
                assert_close(
                    &format!("signal motion_anisotropy {}", path.display()),
                    Some(signals.motion_anisotropy),
                    value_f64(&expected_signals, "motion_anisotropy"),
                    0.02,
                );
                assert_close(
                    &format!("signal salient_sharpness {}", path.display()),
                    signals.salient_sharpness,
                    value_f64(&expected_signals, "salient_sharpness"),
                    5.0,
                );
                assert_close(
                    &format!("signal focus_ratio {}", path.display()),
                    signals.focus_ratio,
                    value_f64(&expected_signals, "focus_ratio"),
                    0.05,
                );
                assert_close(
                    &format!("signal composition {}", path.display()),
                    signals.composition,
                    value_f64(&expected_signals, "composition"),
                    0.02,
                );
            }
            checked += 1;
        }
        assert!(checked > 0, "fast quality fixture had no readable images");
    }

    #[cfg(feature = "opencv-orb")]
    #[test]
    fn opencv_orb_inliers_detect_repeated_scene_geometry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a_path = dir.path().join("scene_0001.png");
        let b_path = dir.path().join("scene_0002.png");
        let c_path = dir.path().join("other_0100.png");

        let mut a = image::RgbImage::from_pixel(420, 320, image::Rgb([8, 8, 8]));
        for i in 0..24u32 {
            let x = 20 + (i * 37) % 340;
            let y = 24 + (i * 53) % 250;
            for yy in y..(y + 14).min(320) {
                for xx in x..(x + 22).min(420) {
                    a.put_pixel(
                        xx,
                        yy,
                        image::Rgb([(40 + i * 7) as u8, (190 - i * 3) as u8, (80 + i * 5) as u8]),
                    );
                }
            }
        }
        let mut b = image::RgbImage::from_pixel(420, 320, image::Rgb([8, 8, 8]));
        for y in 0..300u32 {
            for x in 0..395u32 {
                let px = *a.get_pixel(x, y);
                b.put_pixel(x + 18, y + 11, px);
            }
        }
        let mut c = image::RgbImage::from_pixel(420, 320, image::Rgb([18, 18, 18]));
        for i in 0..18u32 {
            let cx = 30 + (i * 71) % 350;
            let cy = 28 + (i * 41) % 250;
            for yy in cy..(cy + 28).min(320) {
                for xx in cx..(cx + 7).min(420) {
                    c.put_pixel(xx, yy, image::Rgb([220, (20 + i * 11) as u8, 40]));
                }
            }
        }

        a.save(&a_path).expect("save a");
        b.save(&b_path).expect("save b");
        c.save(&c_path).expect("save c");

        let records = [&a_path, &b_path, &c_path]
            .into_iter()
            .map(|path| InfoRecord {
                info: FastImageInfo {
                    path: path.to_string_lossy().to_string(),
                    ..FastImageInfo::default()
                },
                companions: Vec::new(),
            })
            .collect::<Vec<_>>();
        let inliers = compute_orb_inliers_for_records(&records);
        let same_scene = *inliers.get(&(0, 1)).unwrap_or(&0);
        let different_scene = *inliers.get(&(0, 2)).unwrap_or(&0);

        assert!(
            same_scene >= 8,
            "translated same-scene pair should have ORB inliers, got {same_scene}"
        );
        assert!(
            same_scene > different_scene,
            "same scene should outrank different scene: same={same_scene}, different={different_scene}"
        );
    }

    #[cfg(feature = "opencv-orb")]
    #[test]
    fn opencv_orb_fixture_matches_when_configured() {
        #[derive(Deserialize)]
        struct Fixture {
            #[serde(default)]
            source_folder: Option<String>,
            #[serde(default)]
            images: Vec<FixtureImage>,
            #[serde(default)]
            orb_pairs: Vec<FixtureOrbPair>,
        }

        #[derive(Deserialize)]
        struct FixtureImage {
            path: String,
        }

        #[derive(Deserialize)]
        struct FixtureOrbPair {
            i: usize,
            j: usize,
            #[serde(default)]
            base_sim: f64,
            orb_inliers: usize,
            #[serde(default)]
            time_hard_split: bool,
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum OrbEffect {
            Strong,
            Medium,
            WeakDowngrade,
            Neutral,
        }

        fn orb_effect(inliers: usize, base_sim: f64) -> OrbEffect {
            if inliers >= 80 {
                OrbEffect::Strong
            } else if inliers >= 30 {
                OrbEffect::Medium
            } else if inliers < 5 && base_sim > 0.55 {
                OrbEffect::WeakDowngrade
            } else {
                OrbEffect::Neutral
            }
        }

        fn resolve_path(
            workspace: &Path,
            fixture_dir: &Path,
            source_folder: Option<&str>,
            path: &str,
        ) -> PathBuf {
            let raw = PathBuf::from(path);
            if raw.is_absolute() {
                return raw;
            }
            let workspace_path = workspace.join(&raw);
            if workspace_path.exists() {
                return workspace_path;
            }
            if let Some(source_folder) = source_folder {
                let source = PathBuf::from(source_folder);
                let source = if source.is_absolute() {
                    source
                } else {
                    workspace.join(source)
                };
                let source_path = source.join(&raw);
                if source_path.exists() {
                    return source_path;
                }
            }
            fixture_dir.join(raw)
        }

        let Ok(fixture_path) = std::env::var("PIANKE_FAST_ORB_FIXTURE") else {
            return;
        };
        let fixture_path = PathBuf::from(fixture_path);
        if !fixture_path.exists() {
            return;
        }
        let text = fs::read_to_string(&fixture_path).expect("read fast ORB fixture");
        let fixture: Fixture = serde_json::from_str(&text).expect("parse fast ORB fixture");
        if fixture.images.is_empty() || fixture.orb_pairs.is_empty() {
            return;
        }

        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let fixture_dir = fixture_path.parent().unwrap_or(Path::new("."));
        let records = fixture
            .images
            .iter()
            .map(|image| InfoRecord {
                info: FastImageInfo {
                    path: resolve_path(
                        &workspace,
                        fixture_dir,
                        fixture.source_folder.as_deref(),
                        &image.path,
                    )
                    .to_string_lossy()
                    .to_string(),
                    ..FastImageInfo::default()
                },
                companions: Vec::new(),
            })
            .collect::<Vec<_>>();
        let actual = compute_orb_inliers_for_records(&records);
        let mut checked = 0usize;
        for pair in fixture.orb_pairs {
            if pair.time_hard_split {
                continue;
            }
            let key = (pair.i.min(pair.j), pair.i.max(pair.j));
            let actual_inliers = *actual.get(&key).unwrap_or(&0);
            let expected_effect = orb_effect(pair.orb_inliers, pair.base_sim);
            let actual_effect = orb_effect(actual_inliers, pair.base_sim);
            assert_eq!(
                actual_effect, expected_effect,
                "ORB behavior bucket mismatch for pair {:?}: actual={} ({:?}), expected={} ({:?}), base_sim={}",
                key, actual_inliers, actual_effect, pair.orb_inliers, expected_effect, pair.base_sim
            );
            let tolerance = 8usize.max(((pair.orb_inliers as f64) * 0.2).ceil() as usize);
            let delta = actual_inliers.abs_diff(pair.orb_inliers);
            assert!(
                delta <= tolerance,
                "ORB inliers mismatch for pair {:?}: actual={}, expected={}, tolerance={}",
                key,
                actual_inliers,
                pair.orb_inliers,
                tolerance
            );
            checked += 1;
        }
        assert!(checked > 0, "fast ORB fixture had no comparable pairs");
    }
}
