use image::{Rgb, RgbImage};
use pianke_backend::{start, ServerOptions};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test port");
    listener.local_addr().expect("local addr").port()
}

fn backend_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("backend dir");
    fs::create_dir_all(dir.path().join("static")).expect("static dir");
    fs::write(
        dir.path().join("static").join("index.html"),
        "<!doctype html><meta charset='utf-8'><title>test</title>",
    )
    .expect("write index");
    dir
}

fn start_mock_llm_server(response_body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("mock llm bind");
    let port = listener.local_addr().expect("mock llm addr").port();
    let body = response_body.to_string();
    thread::spawn(move || {
        for _ in 0..8 {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    format!("http://127.0.0.1:{port}/v1")
}

fn write_jpg(path: &Path, seed: u8) {
    let mut img = RgbImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            img.put_pixel(
                x,
                y,
                Rgb([
                    seed.wrapping_add((x * 3) as u8),
                    seed.wrapping_add((y * 5) as u8),
                    seed.wrapping_add(((x + y) * 2) as u8),
                ]),
            );
        }
    }
    img.save(path).expect("save jpg");
}

fn wait_for_done(client: &Client, base: &str, token: &str) -> Value {
    wait_for_done_with_timeout(client, base, token, Duration::from_secs(10))
}

fn wait_for_done_with_timeout(
    client: &Client,
    base: &str,
    token: &str,
    timeout: Duration,
) -> Value {
    let deadline = Instant::now() + timeout;
    loop {
        let job: Value = client
            .get(format!("{base}/api/job"))
            .header("X-Token", token)
            .send()
            .expect("job response")
            .json()
            .expect("job json");
        if job["status"] == "done" {
            return job;
        }
        if job["status"] == "error" {
            panic!("job failed: {job}");
        }
        assert!(Instant::now() < deadline, "job timeout: {job}");
        thread::sleep(Duration::from_millis(100));
    }
}

fn wait_for_watermark_done(client: &Client, base: &str, token: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status: Value = client
            .get(format!("{base}/api/watermark/status"))
            .header("X-Token", token)
            .send()
            .expect("watermark status response")
            .json()
            .expect("watermark status json");
        if status["status"] == "done" {
            return status;
        }
        if status["status"] == "error" {
            panic!("watermark failed: {status}");
        }
        assert!(Instant::now() < deadline, "watermark timeout: {status}");
        thread::sleep(Duration::from_millis(100));
    }
}

fn start_test_backend(token: &str) -> (tempfile::TempDir, pianke_backend::ServerHandle, String) {
    let backend = backend_dir();
    let port = reserve_port();
    let handle = start(ServerOptions {
        port,
        token: Some(token.to_string()),
        backend_dir: backend.path().to_path_buf(),
    })
    .expect("start backend");
    let base = format!("http://127.0.0.1:{port}");
    (backend, handle, base)
}

fn start_test_backend_in(
    backend: tempfile::TempDir,
    token: &str,
) -> (tempfile::TempDir, pianke_backend::ServerHandle, String) {
    fs::create_dir_all(backend.path().join("static")).expect("static dir");
    fs::write(
        backend.path().join("static").join("index.html"),
        "<!doctype html><meta charset='utf-8'><title>test</title>",
    )
    .expect("write index");
    let port = reserve_port();
    let handle = start(ServerOptions {
        port,
        token: Some(token.to_string()),
        backend_dir: backend.path().to_path_buf(),
    })
    .expect("start backend");
    let base = format!("http://127.0.0.1:{port}");
    (backend, handle, base)
}

fn materialize_component_for_test(source: &Path, backend: &Path) -> PathBuf {
    let target = backend.join("model_components").join("expert");
    if target.exists() {
        fs::remove_dir_all(&target).expect("remove old test component");
    }
    materialize_tree(source, &target);
    fs::write(
        target.join("component.json"),
        r#"{"id":"expert","version":"onnx-v1","runtime":"onnxruntime","models":["dinov2-small","insightface-det_10g","insightface-w600k_r50","insightface-1k3d68"],"checksum_status":"verified"}"#,
    )
    .expect("write test expert manifest");
    target
}

fn materialize_tree(source: &Path, target: &Path) {
    fs::create_dir_all(target).expect("create component target");
    for entry in fs::read_dir(source).expect("read component source") {
        let entry = entry.expect("component source entry");
        let src = entry.path();
        let dst = target.join(entry.file_name());
        if src.is_dir() {
            materialize_tree(&src, &dst);
        } else {
            if fs::hard_link(&src, &dst).is_err() {
                fs::copy(&src, &dst).expect("copy component file");
            }
        }
    }
}

fn tycoon_e2e_photo_dir() -> Option<PathBuf> {
    if std::env::var("PIANKE_TYCOON_E2E").ok().as_deref() != Some("1") {
        return None;
    }
    if let Ok(dir) = std::env::var("PIANKE_TYCOON_E2E_PHOTO_DIR") {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let path = workspace
        .join(".tmp_expert_parity")
        .join("private_photos")
        .join("cos");
    path.is_dir().then_some(path)
}

fn copy_e2e_photos(source_dir: &Path, target_dir: &Path, limit: usize) -> usize {
    let entries = fs::read_dir(source_dir)
        .expect("read e2e photo dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        })
        .take(limit)
        .collect::<Vec<_>>();
    if limit == 2 && !entries.is_empty() {
        fs::copy(&entries[0], target_dir.join("TYCOON_E2E_0001.jpg"))
            .expect("copy first e2e photo");
        fs::copy(&entries[0], target_dir.join("TYCOON_E2E_0002.jpg"))
            .expect("copy duplicated e2e photo");
        return 2;
    }
    let mut copied = 0;
    for (idx, src) in entries.into_iter().enumerate() {
        let name = format!("TYCOON_E2E_{:04}.jpg", idx + 1);
        fs::copy(&src, target_dir.join(name)).expect("copy e2e photo");
        copied += 1;
    }
    copied
}

#[test]
fn token_guard_rejects_wrong_token_and_allows_valid_token() {
    let token = "secret-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let client = Client::new();

    let denied = client
        .get(format!("{base}/api/capabilities"))
        .header("X-Token", "wrong")
        .send()
        .expect("denied response");
    assert_eq!(denied.status(), 403);

    let ok = client
        .get(format!("{base}/api/capabilities"))
        .header("X-Token", token)
        .send()
        .expect("ok response");
    assert!(ok.status().is_success());
}

#[test]
fn frontend_compat_endpoints_keep_expected_shape() {
    let token = "compat-token";
    let (backend, _handle, base) = start_test_backend(token);
    let llm_base = start_mock_llm_server(
        r#"{"choices":[{"message":{"content":"{\"verdict\":\"reject\",\"reason\":\"too dark\",\"flaws\":\"low light\",\"fixable\":\"exposure\"}"}}]}"#,
    );
    let photos = tempfile::tempdir().expect("photos dir");
    let a = photos.path().join("BURST_0001.jpg");
    let b = photos.path().join("BURST_0002.jpg");
    write_jpg(&a, 45);
    fs::copy(&a, &b).expect("copy identical jpg");

    let client = Client::new();
    let capabilities: Value = client
        .get(format!("{base}/api/capabilities"))
        .header("X-Token", token)
        .send()
        .expect("capabilities response")
        .json()
        .expect("capabilities json");
    assert_eq!(capabilities["face_aware"], false);
    assert_eq!(capabilities["backend"], "rust-fast");
    assert_eq!(capabilities["engines"], json!(["fast", "tycoon"]));
    assert_eq!(capabilities["watermark"], true);
    assert_eq!(capabilities["expert_installed"], false);
    assert_eq!(capabilities["tycoon_ready"], false);
    assert_eq!(capabilities["python_required"], false);
    assert_eq!(capabilities["quality_models"], false);
    assert_eq!(capabilities["nima_legacy_unavailable"], true);
    assert_eq!(capabilities["formats"]["raw_thumbnail"], true);
    assert_eq!(capabilities["formats"]["raw_strategy"], "embedded_jpeg");
    assert_eq!(
        capabilities["opencv_orb"],
        cfg!(feature = "opencv-orb"),
        "top-level OpenCV ORB capability should match compiled feature"
    );
    assert_eq!(
        capabilities["formats"]["opencv_orb"],
        cfg!(feature = "opencv-orb"),
        "format OpenCV ORB capability should match compiled feature"
    );
    assert!(capabilities["formats"]["heic"].is_boolean());
    assert!(matches!(
        capabilities["formats"]["heic_strategy"].as_str(),
        Some("windows_wic" | "unavailable")
    ));
    if capabilities["formats"]["heic"].as_bool().unwrap_or(false) {
        assert_eq!(capabilities["formats"]["heic_strategy"], "windows_wic");
    } else {
        assert_eq!(capabilities["formats"]["heic_strategy"], "unavailable");
    }
    assert_eq!(capabilities["expert_capabilities"]["dinov2"], false);
    assert_eq!(
        capabilities["expert_capabilities"]["insightface_detection"],
        false
    );
    assert_eq!(capabilities["expert_capabilities"]["quality_models"], false);
    assert_eq!(
        capabilities["expert_capabilities"]["nima_legacy_unavailable"],
        true
    );

    let components: Value = client
        .get(format!("{base}/api/model_components"))
        .header("X-Token", token)
        .send()
        .expect("model components response")
        .json()
        .expect("model components json");
    let components_list = components["components"]
        .as_array()
        .expect("components list");
    assert_eq!(components_list.len(), 1);
    assert!(components_list
        .iter()
        .any(|c| c["id"] == "expert" && c["status"] == "not_installed"));
    assert!(components_list.iter().all(|c| c["id"] != "tycoon"));
    assert!(components["cache_dir"]
        .as_str()
        .expect("cache dir")
        .contains("model_components"));

    let install_resp = client
        .post(format!("{base}/api/model_components/install"))
        .header("X-Token", token)
        .json(&json!({"id": "expert"}))
        .send()
        .expect("install response");
    assert_eq!(install_resp.status(), 428);
    let install_json: Value = install_resp.json().expect("install json");
    assert_eq!(install_json["manual_supported"], true);
    assert_eq!(install_json["component"]["id"], "expert");

    let unknown_install = client
        .post(format!("{base}/api/model_components/install"))
        .header("X-Token", token)
        .json(&json!({"id": "unknown"}))
        .send()
        .expect("unknown install response");
    assert_eq!(unknown_install.status(), 404);

    let start_resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start response");
    assert!(start_resp.status().is_success());
    wait_for_done(&client, &base, token);

    let confirm: Value = client
        .post(format!("{base}/api/confirm_prescreen"))
        .header("X-Token", token)
        .send()
        .expect("confirm response")
        .json()
        .expect("confirm json");
    assert_eq!(confirm["async"], true);
    assert_eq!(confirm["all_paths"].as_array().expect("all_paths").len(), 2);

    let progress: Value = client
        .get(format!("{base}/api/grouping_progress?since=0"))
        .header("X-Token", token)
        .send()
        .expect("grouping progress response")
        .json()
        .expect("grouping progress json");
    assert_eq!(progress["status"], "done");
    assert!(progress["groups"].as_array().expect("groups").len() >= 1);
    assert!(progress["total"].as_u64().expect("total") >= 1);
    assert!(progress["multi"].as_u64().expect("multi") >= 1);
    assert!(progress.get("error").is_some());

    let preview: Value = client
        .get(format!("{base}/api/preview_groups"))
        .header("X-Token", token)
        .send()
        .expect("preview response")
        .json()
        .expect("preview json");
    let first = preview["groups"]
        .as_array()
        .expect("preview groups")
        .first()
        .expect("preview group")
        .as_object()
        .expect("preview group object");
    for key in [
        "id",
        "size",
        "samples",
        "best_path",
        "earliest_dt",
        "span_seconds",
    ] {
        assert!(first.contains_key(key), "missing preview field {key}");
    }

    let provider: Value = client
        .get(format!("{base}/api/llm/provider"))
        .header("X-Token", token)
        .send()
        .expect("provider response")
        .json()
        .expect("provider json");
    assert_eq!(provider["configured"], false);
    assert_eq!(provider["key_configured"], false);
    assert!(provider["protocols"]
        .as_array()
        .expect("protocols")
        .contains(&json!("openai_chat_completions")));

    let models_without_provider = client
        .get(format!("{base}/api/llm_models"))
        .header("X-Token", token)
        .send()
        .expect("models response");
    assert_eq!(models_without_provider.status(), 428);
    let models_json: Value = models_without_provider.json().expect("models json");
    assert_eq!(models_json["manual_supported"], true);

    let save_provider: Value = client
        .post(format!("{base}/api/llm/provider"))
        .header("X-Token", token)
        .json(&json!({
            "protocol": "openai_chat_completions",
            "base_url": llm_base,
            "api_key": "secret-key",
            "model": "vision-model",
            "display_name": "Example AI"
        }))
        .send()
        .expect("save provider response")
        .json()
        .expect("save provider json");
    assert_eq!(save_provider["configured"], true);
    assert_eq!(save_provider["config"]["api_key_ref"], "local_secret");
    assert!(
        !fs::read_to_string(backend.path().join("llm_provider").join("provider.json"))
            .expect("provider config")
            .contains("secret-key")
    );

    let ark_status: Value = client
        .get(format!("{base}/api/ark_key"))
        .header("X-Token", token)
        .send()
        .expect("ark alias response")
        .json()
        .expect("ark alias json");
    assert_eq!(ark_status["deprecated"], true);
    assert_eq!(ark_status["configured"], true);

    let tycoon_capabilities: Value = client
        .get(format!("{base}/api/capabilities"))
        .header("X-Token", token)
        .send()
        .expect("tycoon capabilities response")
        .json()
        .expect("tycoon capabilities json");
    assert_eq!(tycoon_capabilities["tycoon_ready"], true);

    let tycoon_start = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "tycoon",
            "llm_model": "vision-model"
        }))
        .send()
        .expect("tycoon start response");
    assert!(tycoon_start.status().is_success());
    let job: Value = client
        .get(format!("{base}/api/job"))
        .header("X-Token", token)
        .send()
        .expect("tycoon job response")
        .json()
        .expect("tycoon job json");
    if job["status"] != "error" {
        wait_for_done(&client, &base, token);
    }
    let job: Value = client
        .get(format!("{base}/api/job"))
        .header("X-Token", token)
        .send()
        .expect("tycoon job response")
        .json()
        .expect("tycoon job json");
    assert_eq!(job["status"], "error");
    assert!(job["error"]
        .as_str()
        .expect("tycoon missing component error")
        .contains("Expert"));
}

#[test]
fn job_log_endpoint_lists_and_reads_session_logs() {
    let token = "job-log-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    let a = photos.path().join("LOG_0001.jpg");
    let b = photos.path().join("LOG_0002.jpg");
    write_jpg(&a, 51);
    fs::copy(&a, &b).expect("copy identical jpg");

    let client = Client::new();
    let no_session = client
        .get(format!("{base}/api/job_log"))
        .header("X-Token", token)
        .send()
        .expect("job log no session response");
    assert_eq!(no_session.status(), 400);

    client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "dry_run": true,
            "mode": "copy",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start response");
    wait_for_done(&client, &base, token);

    let empty_logs: Value = client
        .get(format!("{base}/api/job_log"))
        .header("X-Token", token)
        .send()
        .expect("empty job log response")
        .json()
        .expect("empty job log json");
    assert_eq!(empty_logs["logs"], json!([]));

    let jobs_dir = photos.path().join("_pic_selecter").join("jobs");
    fs::create_dir_all(&jobs_dir).expect("create jobs dir");
    fs::write(jobs_dir.join("20260526-fast.log"), "hello job log\n").expect("write job log");
    fs::write(jobs_dir.join("not-log.txt"), "ignored").expect("write ignored log");

    let logs: Value = client
        .get(format!("{base}/api/job_log"))
        .header("X-Token", token)
        .send()
        .expect("job log list response")
        .json()
        .expect("job log list json");
    let list = logs["logs"].as_array().expect("logs list");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], "20260526-fast.log");
    assert_eq!(list[0]["size"], 14);
    assert!(list[0]["mtime"].as_f64().unwrap_or_default() > 0.0);

    let content = client
        .get(format!("{base}/api/job_log?name=20260526-fast.log"))
        .header("X-Token", token)
        .send()
        .expect("job log read response")
        .text()
        .expect("job log text");
    assert_eq!(content, "hello job log\n");

    let bad_name = client
        .get(format!("{base}/api/job_log?name=../secret.log"))
        .header("X-Token", token)
        .send()
        .expect("bad job log name response");
    assert_eq!(bad_name.status(), 400);
}

#[test]
fn unsupported_raw_and_heic_are_reported_as_skipped() {
    let token = "unsupported-formats-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    fs::write(photos.path().join("ONLY_RAW.CR2"), b"raw without jpg").expect("raw file");
    fs::write(photos.path().join("ONLY_HEIC.HEIC"), b"not decoded").expect("heic file");

    let client = Client::new();
    let start = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start unsupported response");
    assert!(start.status().is_success());
    let job = wait_for_done(&client, &base, token);
    assert_eq!(job["skipped_count"], 2);

    let skipped: Value = client
        .get(format!("{base}/api/skipped"))
        .header("X-Token", token)
        .send()
        .expect("skipped response")
        .json()
        .expect("skipped json");
    let reasons = skipped["skipped"]
        .as_array()
        .expect("skipped list")
        .iter()
        .filter_map(|item| item["reason"].as_str())
        .collect::<Vec<_>>();
    assert!(reasons.iter().any(|reason| reason.contains("RAW")));
    assert!(reasons.iter().any(|reason| reason.contains("HEIC/HEIF")));
}

#[test]
fn pure_raw_with_embedded_jpeg_preview_enters_fast_flow() {
    let token = "raw-preview-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    let jpg = photos.path().join("preview.jpg");
    write_jpg(&jpg, 91);
    let jpeg_bytes = fs::read(&jpg).expect("read preview jpg");
    fs::remove_file(&jpg).expect("remove preview jpg");
    let raw = photos.path().join("ONLY_RAW.CR2");
    let mut raw_bytes = b"fake raw header".to_vec();
    raw_bytes.extend_from_slice(&jpeg_bytes);
    raw_bytes.extend_from_slice(b"fake raw trailer");
    fs::write(&raw, raw_bytes).expect("write raw with preview");

    let client = Client::new();
    let start = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start raw preview response");
    assert!(start.status().is_success());
    let job = wait_for_done(&client, &base, token);
    assert_eq!(job["skipped_count"], 0);

    let status: Value = client
        .get(format!("{base}/api/status"))
        .header("X-Token", token)
        .send()
        .expect("status response")
        .json()
        .expect("status json");
    assert_eq!(status["image_count"], 1);

    let skipped: Value = client
        .get(format!("{base}/api/skipped"))
        .header("X-Token", token)
        .send()
        .expect("skipped response")
        .json()
        .expect("skipped json");
    assert_eq!(
        skipped["skipped"].as_array().expect("skipped list").len(),
        0
    );
}

#[test]
fn llm_test_accepts_all_supported_protocols_with_mock_provider() {
    let token = "llm-test-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let llm_base = start_mock_llm_server(r#"{"ok":true,"output_text":"pong"}"#);
    let client = Client::new();

    for protocol in [
        "openai_chat_completions",
        "openai_responses",
        "anthropic_messages",
    ] {
        let resp: Value = client
            .post(format!("{base}/api/llm/test"))
            .header("X-Token", token)
            .json(&json!({
                "protocol": protocol,
                "base_url": llm_base,
                "api_key": "secret-key",
                "model": "vision-model"
            }))
            .send()
            .expect("llm test response")
            .json()
            .expect("llm test json");
        assert_eq!(
            resp["ok"], true,
            "protocol {protocol} should pass mock test"
        );
        assert_eq!(resp["protocol"], protocol);
    }
}

#[test]
fn installed_model_manifest_updates_capabilities() {
    let token = "models-token";
    let (backend, _handle, base) = start_test_backend(token);
    let expert_dir = backend.path().join("model_components").join("expert");
    fs::create_dir_all(expert_dir.join("models")).expect("expert component dir");
    fs::write(expert_dir.join("models").join("dinov2-small.onnx"), b"fake")
        .expect("expert model marker");
    fs::write(
        expert_dir.join("component.json"),
        r#"{"id":"expert","version":"onnx-v1","runtime":"onnxruntime","models":["dinov2-small"],"checksum_status":"verified"}"#,
    )
    .expect("expert manifest");

    let client = Client::new();
    let capabilities: Value = client
        .get(format!("{base}/api/capabilities"))
        .header("X-Token", token)
        .send()
        .expect("capabilities response")
        .json()
        .expect("capabilities json");
    assert_eq!(
        capabilities["expert_installed"], true,
        "expected materialized Expert component to be installed: {capabilities}"
    );
    assert_eq!(capabilities["face_aware"], false);
    assert_eq!(capabilities["quality_models"], false);
    assert_eq!(capabilities["expert_capabilities"]["dinov2"], true);
    assert_eq!(
        capabilities["expert_capabilities"]["insightface_detection"],
        false
    );
    assert_eq!(capabilities["expert_capabilities"]["musiq"], false);
    assert_eq!(capabilities["expert_capabilities"]["clipiqa"], false);
    assert_eq!(capabilities["expert_capabilities"]["quality_models"], false);
    assert_eq!(capabilities["expert_capabilities"]["nima_legacy"], false);
    assert_eq!(capabilities["tycoon_ready"], false);
    assert_eq!(capabilities["model_components"]["expert"], "installed");
    assert_eq!(capabilities["engines"], json!(["expert", "fast", "tycoon"]));
}

#[test]
fn tycoon_mock_e2e_runs_with_complete_expert_component_when_configured() {
    let Some(photo_dir) = tycoon_e2e_photo_dir() else {
        return;
    };
    let Ok(component_dir) = std::env::var("PIANKE_EXPERT_COMPONENT_DIR") else {
        return;
    };
    let component_dir = PathBuf::from(component_dir);
    if !component_dir.join("component.json").exists()
        || !component_dir
            .join("models")
            .join("dinov2-small.onnx")
            .exists()
        || !component_dir
            .join("models")
            .join("insightface")
            .join("det_10g.onnx")
            .exists()
        || !component_dir
            .join("models")
            .join("insightface")
            .join("w600k_r50.onnx")
            .exists()
        || !component_dir
            .join("models")
            .join("insightface")
            .join("1k3d68.onnx")
            .exists()
    {
        return;
    }

    let token = "tycoon-e2e-token";
    let backend = backend_dir();
    materialize_component_for_test(&component_dir, backend.path());
    let (_backend, _handle, base) = start_test_backend_in(backend, token);
    let photos = tempfile::tempdir().expect("tycoon e2e photos dir");
    let copied = copy_e2e_photos(&photo_dir, photos.path(), 2);
    if copied == 0 {
        return;
    }

    let llm_base = start_mock_llm_server(
        r#"{"choices":[{"message":{"content":"{\"verdict\":\"pass\",\"reason\":\"mock ok\",\"flaws\":\"none\",\"fixable\":\"none\"}"}}]}"#,
    );
    let client = Client::new();
    let save_provider: Value = client
        .post(format!("{base}/api/llm/provider"))
        .header("X-Token", token)
        .json(&json!({
            "protocol": "openai_chat_completions",
            "base_url": llm_base,
            "api_key": "mock-key",
            "model": "mock-vision",
            "display_name": "Mock Vision",
            "timeout_seconds": 30
        }))
        .send()
        .expect("save mock provider response")
        .json()
        .expect("save mock provider json");
    assert_eq!(save_provider["configured"], true);

    let capabilities: Value = client
        .get(format!("{base}/api/capabilities"))
        .header("X-Token", token)
        .send()
        .expect("capabilities response")
        .json()
        .expect("capabilities json");
    assert_eq!(
        capabilities["expert_installed"], true,
        "expected materialized Expert component to be installed: {capabilities}"
    );
    assert_eq!(
        capabilities["expert_capabilities"]["dinov2"], true,
        "expected DINOv2 capability: {capabilities}"
    );
    assert_eq!(
        capabilities["expert_capabilities"]["insightface_detection"], true,
        "expected InsightFace detection capability: {capabilities}"
    );
    assert_eq!(
        capabilities["expert_capabilities"]["insightface_recognition"], true,
        "expected InsightFace recognition capability: {capabilities}"
    );
    assert_eq!(
        capabilities["expert_capabilities"]["insightface_landmark"], true,
        "expected InsightFace landmark capability: {capabilities}"
    );
    assert_eq!(
        capabilities["tycoon_ready"], true,
        "expected Tycoon provider readiness: {capabilities}"
    );

    let start_resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "tycoon",
            "llm_model": "mock-vision",
            "prescreen_enabled": false
        }))
        .send()
        .expect("tycoon start response");
    assert!(start_resp.status().is_success());

    let job = wait_for_done_with_timeout(&client, &base, token, Duration::from_secs(120));
    assert_eq!(job["engine"], "tycoon");
    let events = job["events"].as_array().expect("job events");
    assert!(
        events.iter().any(|event| {
            event["signals"].as_array().is_some_and(|signals| {
                signals.iter().any(|signal| {
                    signal["kind"] == "llm" && signal["value"].as_str() == Some("pass")
                })
            }) && event["reason"].as_str() == Some("mock ok")
        }),
        "expected Tycoon job event with LLM verdict and reason: {job}"
    );

    let group: Value = client
        .get(format!("{base}/api/group"))
        .header("X-Token", token)
        .send()
        .expect("tycoon group response")
        .json()
        .expect("tycoon group json");
    assert_eq!(group["done"], false, "expected Tycoon group: {group}");
    for side in ["left_meta", "right_meta"] {
        if group["group"][side].is_object() {
            assert_eq!(group["group"][side]["llm_verdict"], "pass");
            assert_eq!(group["group"][side]["llm_reason"], "mock ok");
            assert_eq!(group["group"][side]["dinov2_dim"], 384);
            assert!(
                group["group"][side]["face_embedding_count"]
                    .as_u64()
                    .unwrap_or(0)
                    >= 1,
                "expected face embeddings in {side}: {group}"
            );
        }
    }
}

#[test]
fn expert_start_requires_installed_component() {
    let token = "expert-missing-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    write_jpg(&photos.path().join("EXPERT_0001.jpg"), 90);

    let client = Client::new();
    let resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "expert"
        }))
        .send()
        .expect("expert start response");
    assert_eq!(resp.status(), 428);
    let body: Value = resp.json().expect("expert error json");
    assert!(body["error"]
        .as_str()
        .expect("error message")
        .contains("Expert"));
}

#[test]
fn expert_start_with_incomplete_component_requires_reinstall() {
    let token = "expert-incomplete-token";
    let (backend, _handle, base) = start_test_backend(token);
    let expert_dir = backend.path().join("model_components").join("expert");
    fs::create_dir_all(&expert_dir).expect("expert component dir");
    fs::write(
        expert_dir.join("component.json"),
        r#"{"id":"expert","version":"onnx-v1","runtime":"onnxruntime","models":["dinov2-small"],"checksum_status":"verified"}"#,
    )
    .expect("expert manifest");
    let photos = tempfile::tempdir().expect("photos dir");
    write_jpg(&photos.path().join("EXPERT_0001.jpg"), 90);

    let client = Client::new();
    let resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "expert",
            "prescreen_enabled": false
        }))
        .send()
        .expect("expert start response");
    assert_eq!(resp.status(), 428);
    let body: Value = resp.json().expect("expert start json");
    assert!(body["error"]
        .as_str()
        .expect("error message")
        .contains("Expert"));
}

#[test]
fn model_component_install_from_source_dir_updates_capabilities() {
    let token = "model-install-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let source = tempfile::tempdir().expect("source component dir");
    fs::create_dir_all(source.path().join("models")).expect("models dir");
    let model_path = source.path().join("models").join("dinov2-small.onnx");
    fs::write(&model_path, b"fake onnx bytes").expect("model bytes");
    fs::write(
        source.path().join("component.json"),
        r#"{
            "id":"expert",
            "version":"onnx-v1",
            "runtime":"onnxruntime",
            "models":["dinov2-small"],
            "files":[{"path":"models/dinov2-small.onnx","size_bytes":15}]
        }"#,
    )
    .expect("component manifest");

    let client = Client::new();
    let install: Value = client
        .post(format!("{base}/api/model_components/install"))
        .header("X-Token", token)
        .json(&json!({
            "id": "expert",
            "source_dir": source.path()
        }))
        .send()
        .expect("install response")
        .json()
        .expect("install json");
    assert_eq!(install["ok"], true);
    assert_eq!(install["component"]["status"], "installed");
    assert_eq!(
        install["component"]["manifest"]["checksum_status"],
        "verified"
    );
    let install_dir = PathBuf::from(
        install["component"]["install_dir"]
            .as_str()
            .expect("install dir"),
    );
    assert!(install_dir
        .join("models")
        .join("dinov2-small.onnx")
        .exists());

    let capabilities: Value = client
        .get(format!("{base}/api/capabilities"))
        .header("X-Token", token)
        .send()
        .expect("capabilities response")
        .json()
        .expect("capabilities json");
    assert_eq!(capabilities["expert_installed"], true);
    assert_eq!(capabilities["model_components"]["expert"], "installed");
}

#[test]
fn fast_main_flow_copies_winner_and_loser_without_moving_sources() {
    let token = "flow-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    let a = photos.path().join("IMG_0001.jpg");
    let b = photos.path().join("IMG_0002.jpg");
    let c = photos.path().join("IMG_0100.jpg");
    write_jpg(&a, 20);
    fs::copy(&a, &b).expect("copy identical jpg");
    write_jpg(&c, 180);

    let client = Client::new();
    let start_resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start response");
    assert!(start_resp.status().is_success());

    wait_for_done(&client, &base, token);
    let group_resp: Value = client
        .get(format!("{base}/api/group"))
        .header("X-Token", token)
        .send()
        .expect("group response")
        .json()
        .expect("group json");
    assert_eq!(
        group_resp["done"], false,
        "expected one pair group: {group_resp}"
    );
    let left = group_resp["group"]["left"]
        .as_str()
        .expect("left path")
        .to_string();
    let right = group_resp["group"]["right"]
        .as_str()
        .expect("right path")
        .to_string();

    let choose_resp = client
        .post(format!("{base}/api/choose"))
        .header("X-Token", token)
        .json(&json!({"loser": "right"}))
        .send()
        .expect("choose response");
    assert!(choose_resp.status().is_success());

    assert!(a.exists(), "copy mode must keep source a");
    assert!(b.exists(), "copy mode must keep source b");
    assert!(PathBuf::from(&left).exists(), "winner source still exists");
    assert!(PathBuf::from(&right).exists(), "loser source still exists");
    assert!(
        photos
            .path()
            .join("winners")
            .join(file_name(&left))
            .exists(),
        "winner copy exists"
    );
    assert!(
        photos
            .path()
            .join("losers")
            .join(file_name(&right))
            .exists(),
        "loser copy exists"
    );
}

#[test]
fn rust_watermark_preview_and_batch_export_work_after_fast_selection() {
    let token = "watermark-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    let a = photos.path().join("WM_0001.jpg");
    let b = photos.path().join("WM_0002.jpg");
    write_jpg(&a, 75);
    fs::copy(&a, &b).expect("copy identical jpg");

    let client = Client::new();
    let templates: Value = client
        .get(format!("{base}/api/watermark/templates"))
        .header("X-Token", token)
        .send()
        .expect("templates response")
        .json()
        .expect("templates json");
    let template_list = templates["templates"].as_array().expect("templates");
    assert!(template_list.len() >= 1);
    let first_template = template_list.first().expect("first template");
    for key in ["id", "name", "desc"] {
        assert!(
            first_template.get(key).and_then(|v| v.as_str()).is_some(),
            "missing template field {key}: {templates}"
        );
    }
    assert!(templates["logos"].as_array().expect("logos").is_empty());

    let idle_cancel = client
        .post(format!("{base}/api/watermark/cancel"))
        .header("X-Token", token)
        .json(&json!({}))
        .send()
        .expect("idle cancel response");
    assert_eq!(idle_cancel.status(), 400);

    let start_resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start response");
    assert!(start_resp.status().is_success());
    wait_for_done(&client, &base, token);

    let group_resp: Value = client
        .get(format!("{base}/api/group"))
        .header("X-Token", token)
        .send()
        .expect("group response")
        .json()
        .expect("group json");
    assert_eq!(group_resp["done"], false, "expected comparable group");
    let left = group_resp["group"]["left"]
        .as_str()
        .expect("left path")
        .to_string();

    let choose_resp = client
        .post(format!("{base}/api/choose"))
        .header("X-Token", token)
        .json(&json!({"loser": "right"}))
        .send()
        .expect("choose response");
    assert!(choose_resp.status().is_success());

    let preview: Value = client
        .post(format!("{base}/api/watermark/preview"))
        .header("X-Token", token)
        .json(&json!({"template": "A", "preview_index": 0}))
        .send()
        .expect("preview response")
        .json()
        .expect("preview json");
    assert!(preview["image_b64"].as_str().expect("preview b64").len() > 100);
    assert!(preview["size_kb"].as_f64().expect("preview size") > 0.0);
    assert_eq!(preview["preview_index"], 0);
    assert_eq!(preview["source_name"], file_name(&left));
    assert_eq!(preview["total_winners"], 1);
    assert!(preview["exif"].is_object());

    let export: Value = client
        .post(format!("{base}/api/watermark/start"))
        .header("X-Token", token)
        .json(&json!({"template": "A"}))
        .send()
        .expect("watermark start response")
        .json()
        .expect("watermark start json");
    assert_eq!(export["ok"], true);
    assert_eq!(export["total"], 1);
    assert!(PathBuf::from(export["out_dir"].as_str().expect("export out dir")).exists());

    let status = wait_for_watermark_done(&client, &base, token);
    assert_eq!(status["status"], "done");
    assert_eq!(status["done"], 1);
    assert_eq!(status["total"], 1);
    assert_eq!(status["ok"], 1);
    assert_eq!(status["failed_count"], 0);
    assert!(status["failed"].as_array().expect("failed list").is_empty());
    assert!(status["failed_sample"]
        .as_array()
        .expect("failed sample")
        .is_empty());
    assert!(status["elapsed"].as_f64().expect("elapsed") >= 0.0);
    let out_dir = PathBuf::from(status["out_dir"].as_str().expect("out dir"));
    assert!(out_dir.exists(), "watermark output dir exists");
    assert!(
        fs::read_dir(&out_dir)
            .expect("read output dir")
            .flatten()
            .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jpg")),
        "watermark output jpg exists"
    );

    if std::env::var("PIANKE_WATERMARK_OPEN_OUT_DIR_SMOKE").ok().as_deref() == Some("1") {
        let open_out_dir: Value = client
            .post(format!("{base}/api/watermark/open_out_dir"))
            .header("X-Token", token)
            .json(&json!({}))
            .send()
            .expect("open out dir response")
            .json()
            .expect("open out dir json");
        assert_eq!(open_out_dir["ok"], true);
    }
}

#[test]
fn move_mode_moves_files_and_undo_restores_sources() {
    let token = "move-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    let a = photos.path().join("MOVE_0001.jpg");
    let b = photos.path().join("MOVE_0002.jpg");
    write_jpg(&a, 35);
    fs::copy(&a, &b).expect("copy identical jpg");

    let client = Client::new();
    let start_resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "move",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start response");
    assert!(start_resp.status().is_success());
    wait_for_done(&client, &base, token);

    let group_resp: Value = client
        .get(format!("{base}/api/group"))
        .header("X-Token", token)
        .send()
        .expect("group response")
        .json()
        .expect("group json");
    let left = group_resp["group"]["left"]
        .as_str()
        .expect("left path")
        .to_string();
    let right = group_resp["group"]["right"]
        .as_str()
        .expect("right path")
        .to_string();

    let choose_resp = client
        .post(format!("{base}/api/choose"))
        .header("X-Token", token)
        .json(&json!({"loser": "right"}))
        .send()
        .expect("choose response");
    assert!(choose_resp.status().is_success());

    assert!(!PathBuf::from(&left).exists(), "winner source was moved");
    assert!(!PathBuf::from(&right).exists(), "loser source was moved");
    assert!(photos
        .path()
        .join("winners")
        .join(file_name(&left))
        .exists());
    assert!(photos
        .path()
        .join("losers")
        .join(file_name(&right))
        .exists());

    let undo_resp: Value = client
        .post(format!("{base}/api/undo"))
        .header("X-Token", token)
        .send()
        .expect("undo response")
        .json()
        .expect("undo json");
    assert_eq!(undo_resp["done"], false);
    assert!(PathBuf::from(&left).exists(), "undo restores winner source");
    assert!(PathBuf::from(&right).exists(), "undo restores loser source");
    assert!(
        !photos
            .path()
            .join("winners")
            .join(file_name(&left))
            .exists(),
        "undo removes winner target"
    );
    assert!(
        !photos
            .path()
            .join("losers")
            .join(file_name(&right))
            .exists(),
        "undo removes loser target"
    );
}

#[test]
fn move_mode_reopen_group_restores_sources() {
    let token = "reopen-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    let a = photos.path().join("REOPEN_0001.jpg");
    let b = photos.path().join("REOPEN_0002.jpg");
    write_jpg(&a, 55);
    fs::copy(&a, &b).expect("copy identical jpg");

    let client = Client::new();
    let start_resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "move",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start response");
    assert!(start_resp.status().is_success());
    wait_for_done(&client, &base, token);

    let group_resp: Value = client
        .get(format!("{base}/api/group"))
        .header("X-Token", token)
        .send()
        .expect("group response")
        .json()
        .expect("group json");
    let group_id = group_resp["group"]["id"]
        .as_str()
        .expect("group id")
        .to_string();
    let left = group_resp["group"]["left"]
        .as_str()
        .expect("left path")
        .to_string();
    let right = group_resp["group"]["right"]
        .as_str()
        .expect("right path")
        .to_string();

    let choose_resp = client
        .post(format!("{base}/api/choose"))
        .header("X-Token", token)
        .json(&json!({"loser": "right"}))
        .send()
        .expect("choose response");
    assert!(choose_resp.status().is_success());
    assert!(!PathBuf::from(&left).exists());
    assert!(!PathBuf::from(&right).exists());

    let reopen: Value = client
        .post(format!("{base}/api/reopen_group"))
        .header("X-Token", token)
        .json(&json!({"group_id": group_id}))
        .send()
        .expect("reopen response")
        .json()
        .expect("reopen json");
    assert_eq!(reopen["ok"], true);
    assert_eq!(reopen["failed"].as_array().expect("failed").len(), 0);
    assert!(
        PathBuf::from(&left).exists(),
        "reopen restores winner source"
    );
    assert!(
        PathBuf::from(&right).exists(),
        "reopen restores loser source"
    );
    assert!(!photos
        .path()
        .join("winners")
        .join(file_name(&left))
        .exists());
    assert!(!photos
        .path()
        .join("losers")
        .join(file_name(&right))
        .exists());
}

#[test]
fn raw_with_jpg_companion_is_copied_together() {
    let token = "raw-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    let raw = photos.path().join("PAIR_0001.CR2");
    let jpg = photos.path().join("PAIR_0001.JPG");
    let xmp = photos.path().join("PAIR_0001.XMP");
    fs::write(&raw, b"fake raw primary").expect("write raw");
    write_jpg(&jpg, 90);
    fs::write(&xmp, b"<xmpmeta />").expect("write xmp");

    let client = Client::new();
    let resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "copy",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start response");
    assert!(resp.status().is_success());
    wait_for_done(&client, &base, token);

    assert!(raw.exists(), "copy mode keeps raw primary");
    assert!(jpg.exists(), "copy mode keeps jpg companion");
    assert!(xmp.exists(), "copy mode keeps xmp companion");
    assert!(photos.path().join("winners").join("PAIR_0001.CR2").exists());
    assert!(photos.path().join("winners").join("PAIR_0001.JPG").exists());
    assert!(photos.path().join("winners").join("PAIR_0001.XMP").exists());
}

#[test]
fn move_mode_moves_raw_jpg_xmp_companions_together() {
    let token = "raw-move-token";
    let (_backend, _handle, base) = start_test_backend(token);
    let photos = tempfile::tempdir().expect("photos dir");
    let raw = photos.path().join("MOVEPAIR_0001.CR2");
    let jpg = photos.path().join("MOVEPAIR_0001.JPG");
    let xmp = photos.path().join("MOVEPAIR_0001.XMP");
    fs::write(&raw, b"fake raw primary").expect("write raw");
    write_jpg(&jpg, 110);
    fs::write(&xmp, b"<xmpmeta />").expect("write xmp");

    let client = Client::new();
    let resp = client
        .post(format!("{base}/api/start"))
        .header("X-Token", token)
        .json(&json!({
            "folder": photos.path(),
            "mode": "move",
            "engine": "fast",
            "prescreen_enabled": false
        }))
        .send()
        .expect("start response");
    assert!(resp.status().is_success());
    wait_for_done(&client, &base, token);

    assert!(!raw.exists(), "move mode moves raw primary");
    assert!(!jpg.exists(), "move mode moves jpg companion");
    assert!(!xmp.exists(), "move mode moves xmp companion");
    assert!(photos
        .path()
        .join("winners")
        .join("MOVEPAIR_0001.CR2")
        .exists());
    assert!(photos
        .path()
        .join("winners")
        .join("MOVEPAIR_0001.JPG")
        .exists());
    assert!(photos
        .path()
        .join("winners")
        .join("MOVEPAIR_0001.XMP")
        .exists());
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}
