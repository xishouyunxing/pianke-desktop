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
    let deadline = Instant::now() + Duration::from_secs(10);
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
    assert_eq!(capabilities["expert_installed"], true);
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
    assert!(templates["templates"].as_array().expect("templates").len() >= 1);
    assert!(templates["logos"].as_array().expect("logos").is_empty());

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

    let status = wait_for_watermark_done(&client, &base, token);
    assert_eq!(status["ok"], 1);
    assert_eq!(status["failed_count"], 0);
    let out_dir = PathBuf::from(status["out_dir"].as_str().expect("out dir"));
    assert!(out_dir.exists(), "watermark output dir exists");
    assert!(
        fs::read_dir(&out_dir)
            .expect("read output dir")
            .flatten()
            .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jpg")),
        "watermark output jpg exists"
    );
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
