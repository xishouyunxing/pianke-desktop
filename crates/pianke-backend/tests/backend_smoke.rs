use image::{Rgb, RgbImage};
use pianke_backend::{start, ServerOptions};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::{
    fs,
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
    let (_backend, _handle, base) = start_test_backend(token);
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
    assert_eq!(capabilities["engines"], json!(["fast"]));

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

    let unavailable = client
        .get(format!("{base}/api/llm_models"))
        .header("X-Token", token)
        .send()
        .expect("unavailable response");
    assert_eq!(unavailable.status(), 501);
    let unavailable_json: Value = unavailable.json().expect("unavailable json");
    assert_eq!(unavailable_json["unavailable"], true);
    assert!(unavailable_json["error"].as_str().unwrap_or("").len() > 0);
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
