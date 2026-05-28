use axum::{
    body::Body,
    extract::{Path as AxumPath, Query, State},
    http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use chrono::{DateTime, Local};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageFormat};
use pianke_core::fast::{
    analyze_from_signals, average_hash_from_luma, cluster, cluster_with_options,
    difference_hash_from_luma, perceptual_hash_from_luma, wavelet_hash_from_luma, ExifSummary,
    FastClusterOptions, FastImageInfo, FastQualityProfile, FastQualitySignals, QualityInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    env, fs,
    io::Cursor,
    net::{SocketAddr, TcpListener},
    path::{Component, Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::oneshot;
use uuid::Uuid;

#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;

mod model_components;
use model_components::{ComponentInstallRequest, ModelManager};

mod expert_vision;
mod llm_provider;
use llm_provider::JudgeVerdict;
use llm_provider::{LlmProviderManager, SaveProviderRequest};
mod watermark;
use watermark::WatermarkConfig;

const STATE_FILENAME: &str = ".pic_selecter_state.json";
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

pub struct ServerOptions {
    pub port: u16,
    pub token: Option<String>,
    pub backend_dir: PathBuf,
    pub folder_picker: Option<Arc<dyn Fn() -> Result<Option<PathBuf>, String> + Send + Sync>>,
}

#[derive(Debug)]
pub struct ServerHandle {
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl ServerHandle {
    pub fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

#[derive(Clone)]
struct AppCtx {
    inner: Arc<Mutex<AppState>>,
    token: Option<String>,
    backend_dir: PathBuf,
    models: ModelManager,
    llm: LlmProviderManager,
    folder_picker: Option<Arc<dyn Fn() -> Result<Option<PathBuf>, String> + Send + Sync>>,
}

#[derive(Debug, Default)]
struct AppState {
    session: Option<SessionState>,
    job: JobState,
    last_infos: Vec<InfoRecord>,
    grouping: GroupingState,
    watermark: WatermarkState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroupState {
    images: Vec<String>,
    #[serde(default)]
    pending: Vec<String>,
    left: Option<String>,
    right: Option<String>,
    #[serde(default)]
    losers: Vec<String>,
    winner: Option<String>,
    #[serde(default)]
    extra_winners: Vec<String>,
    #[serde(default)]
    finished: bool,
    #[serde(default)]
    applied: bool,
    id: String,
    #[serde(default)]
    move_log: Vec<Value>,
    #[serde(default)]
    auto_rejected: Vec<String>,
    #[serde(default)]
    auto_reject_reasons: HashMap<String, String>,
    #[serde(default)]
    auto_selected: bool,
    #[serde(default)]
    manual_restored: Vec<String>,
}

impl GroupState {
    fn new(images: Vec<String>) -> Self {
        let mut group = Self {
            images,
            pending: Vec::new(),
            left: None,
            right: None,
            losers: Vec::new(),
            winner: None,
            extra_winners: Vec::new(),
            finished: false,
            applied: false,
            id: Uuid::new_v4().to_string().replace('-', ""),
            move_log: Vec::new(),
            auto_rejected: Vec::new(),
            auto_reject_reasons: HashMap::new(),
            auto_selected: false,
            manual_restored: Vec::new(),
        };
        let live = group.images.clone();
        if live.len() == 1 {
            group.winner = live.first().cloned();
            group.finished = true;
        } else {
            group.left = live.get(0).cloned();
            group.right = live.get(1).cloned();
            group.pending = live.into_iter().skip(2).collect();
        }
        group
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionState {
    folder: String,
    dry_run: bool,
    mode: String,
    engine: String,
    groups: Vec<GroupState>,
    current_group: usize,
    threshold_near: i32,
    threshold_far: i32,
    near_seconds: i32,
    prescreen_enabled: bool,
    prescreen_strength: String,
    prescreen_reviewed: bool,
    #[serde(default)]
    prescreen_rejected: Vec<String>,
    #[serde(default)]
    prescreen_reject_reasons: HashMap<String, String>,
    #[serde(default)]
    prescreen_restored: Vec<String>,
    #[serde(default)]
    undo_stack: Vec<UndoEntry>,
    #[serde(default)]
    meta: HashMap<String, Value>,
    #[serde(default)]
    companions: HashMap<String, Vec<String>>,
    #[serde(default)]
    selection_started: bool,
    #[serde(default)]
    skipped: Vec<SkippedItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UndoEntry {
    group_index: usize,
    group: GroupState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SkippedItem {
    path: String,
    reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InfoRecord {
    info: FastImageInfo,
    companions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct JobEvent {
    seq: u64,
    name: String,
    path: String,
    ok: bool,
    reject: bool,
    reason: Option<String>,
    verdict: String,
    engine: String,
    shutter: Option<String>,
    aperture: Option<String>,
    iso: Option<String>,
    signals: Vec<Value>,
}

#[derive(Debug, Clone, Serialize)]
struct JobState {
    folder: String,
    dry_run: bool,
    mode: String,
    engine: String,
    status: String,
    done: usize,
    total: usize,
    label: String,
    error: Option<String>,
    skipped: Vec<SkippedItem>,
    started_at: f64,
    finished_at: f64,
    cancel_requested: bool,
    threshold_near: i32,
    threshold_far: i32,
    near_seconds: i32,
    prescreen_enabled: bool,
    prescreen_strength: String,
    recent_events: Vec<JobEvent>,
    event_seq: u64,
}

impl Default for JobState {
    fn default() -> Self {
        Self {
            folder: String::new(),
            dry_run: false,
            mode: "copy".to_string(),
            engine: "fast".to_string(),
            status: "idle".to_string(),
            done: 0,
            total: 0,
            label: String::new(),
            error: None,
            skipped: Vec::new(),
            started_at: 0.0,
            finished_at: 0.0,
            cancel_requested: false,
            threshold_near: 10,
            threshold_far: 6,
            near_seconds: 300,
            prescreen_enabled: true,
            prescreen_strength: "standard".to_string(),
            recent_events: Vec::new(),
            event_seq: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct GroupingState {
    status: String,
    groups: Vec<Value>,
    all_paths: Vec<String>,
    total: usize,
    multi: usize,
    error: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
struct WatermarkState {
    status: String,
    done: usize,
    total: usize,
    current: String,
    out_dir: Option<String>,
    ok: usize,
    failed: Vec<(String, String)>,
    error: Option<String>,
    started_at: f64,
    finished_at: f64,
    cancel_requested: bool,
}

#[derive(Debug, Deserialize)]
struct StartRequest {
    folder: String,
    #[serde(default)]
    dry_run: bool,
    #[serde(default = "default_copy")]
    mode: String,
    #[serde(default = "default_fast")]
    engine: String,
    #[serde(default = "default_true")]
    prescreen_enabled: bool,
    #[serde(default = "default_standard")]
    prescreen_strength: String,
    #[serde(default)]
    llm_model: Option<String>,
    #[serde(default = "default_threshold_near")]
    threshold_near: i32,
    #[serde(default = "default_threshold_far")]
    threshold_far: i32,
    #[serde(default = "default_near_seconds")]
    near_seconds: i32,
}

fn default_copy() -> String {
    "copy".to_string()
}
fn default_fast() -> String {
    "fast".to_string()
}
fn default_standard() -> String {
    "standard".to_string()
}
fn default_true() -> bool {
    true
}
fn default_threshold_near() -> i32 {
    10
}
fn default_threshold_far() -> i32 {
    6
}
fn default_near_seconds() -> i32 {
    300
}

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

fn build_router(ctx: AppCtx) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/static/*path", get(static_file))
        .route("/api/desktop/health", get(health))
        .route("/api/app_update", get(app_update))
        .route("/api/capabilities", get(capabilities))
        .route("/api/model_components", get(model_components))
        .route("/api/model_components/status", get(model_components))
        .route(
            "/api/model_components/install",
            post(model_component_install),
        )
        .route(
            "/api/model_components/delete",
            post(model_component_delete),
        )
        .route(
            "/api/model_components/pause",
            post(model_component_pause),
        )
        .route(
            "/api/model_components/cancel",
            post(model_component_cancel),
        )
        .route(
            "/api/llm/provider",
            get(llm_provider_status)
                .post(llm_provider_save)
                .delete(llm_provider_clear),
        )
        .route("/api/llm/models", get(llm_models))
        .route("/api/llm/test", post(llm_test))
        .route("/api/start", post(start_job))
        .route("/api/reset_session", post(reset_session))
        .route("/api/cancel_job", post(cancel_job))
        .route("/api/job", get(job_status))
        .route("/api/status", get(session_status))
        .route("/api/group", get(current_group))
        .route("/api/choose", post(choose))
        .route("/api/kick", post(kick))
        .route("/api/undo", post(undo))
        .route("/api/skip_group", post(skip_group))
        .route("/api/reopen_group", post(reopen_group))
        .route("/api/image", get(image_endpoint))
        .route("/api/image_original", get(image_original))
        .route("/api/winners", get(winners))
        .route("/api/auto_rejected", get(auto_rejected))
        .route("/api/restore_rejected", post(restore_rejected))
        .route("/api/confirm_prescreen", post(confirm_prescreen))
        .route("/api/grouping_progress", get(grouping_progress))
        .route("/api/skipped", get(skipped))
        .route("/api/regroup", post(regroup))
        .route("/api/preview_groups", get(preview_groups))
        .route("/api/peek_folder", post(peek_folder))
        .route("/api/open_folder", post(open_folder))
        .route("/api/browse_folder", post(browse_folder))
        .route(
            "/api/ark_key",
            get(ark_key_status)
                .post(ark_key_save_compat)
                .delete(ark_key_clear),
        )
        .route("/api/llm_models", get(llm_models))
        .route("/api/job_log", get(job_log))
        .route("/api/llm_concurrency", get(llm_concurrency))
        .route("/api/watermark/templates", get(watermark_templates))
        .route("/api/watermark/preview", post(watermark_preview))
        .route("/api/watermark/start", post(watermark_start))
        .route("/api/watermark/status", get(watermark_status))
        .route("/api/watermark/cancel", post(watermark_cancel))
        .route("/api/watermark/open_out_dir", post(watermark_open_out_dir))
        .layer(middleware::from_fn_with_state(ctx.clone(), token_guard))
        .with_state(ctx)
}

async fn token_guard(
    State(ctx): State<AppCtx>,
    req: Request<Body>,
    next: Next,
) -> Result<Response, Response> {
    let path = req.uri().path();
    if path == "/" || path.starts_with("/static/") {
        return Ok(next.run(req).await);
    }
    if let Some(expected) = &ctx.token {
        let headers = req.headers();
        let header_token = headers
            .get("X-Token")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let query_token = req
            .uri()
            .query()
            .and_then(|q| {
                q.split('&').find_map(|part| {
                    let (k, v) = part.split_once('=')?;
                    (k == "token").then_some(v)
                })
            })
            .unwrap_or("");
        if header_token != expected && query_token != expected {
            return Err(json_error(StatusCode::FORBIDDEN, "forbidden token"));
        }
    }
    if req.method() != Method::GET && path != "/api/start" && path != "/api/reset_session" {
        // The token already gates mutating requests. This placeholder keeps the
        // guard explicit without relying on browser Origin behavior in Tauri.
    }
    Ok(next.run(req).await)
}

async fn index(State(ctx): State<AppCtx>) -> Response {
    match fs::read_to_string(ctx.backend_dir.join("static").join("index.html")) {
        Ok(text) => Html(text).into_response(),
        Err(_) => Html("<!doctype html><meta charset='utf-8'><h1>片刻 Rust Fast 后端</h1>")
            .into_response(),
    }
}

async fn static_file(State(ctx): State<AppCtx>, AxumPath(path): AxumPath<String>) -> Response {
    let Some(path) = safe_join(&ctx.backend_dir.join("static"), &path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    file_response(path, None)
}

async fn health() -> impl IntoResponse {
    Json(json!({"ok": true, "backend": "rust-fast"}))
}

#[derive(Debug, Deserialize)]
struct AppUpdateManifest {
    version: String,
    url: String,
    #[serde(default)]
    notes: String,
    #[serde(default)]
    published_at: String,
}

async fn app_update() -> impl IntoResponse {
    let current_version = env!("CARGO_PKG_VERSION").to_string();
    let manifest_url =
        env::var("PIANKE_APP_UPDATE_URL").unwrap_or_else(|_| DEFAULT_APP_UPDATE_URL.to_string());

    match fetch_app_update_manifest(&manifest_url).await {
        Ok(manifest) => {
            let update_available = compare_versions(&manifest.version, &current_version).is_gt();
            Json(json!({
                "current_version": current_version,
                "latest_version": manifest.version,
                "update_available": update_available,
                "url": manifest.url,
                "notes": manifest.notes,
                "published_at": manifest.published_at
            }))
        }
        Err(err) => Json(json!({
            "current_version": current_version,
            "latest_version": Value::Null,
            "update_available": false,
            "url": Value::Null,
            "notes": "",
            "published_at": "",
            "error": err
        })),
    }
}

async fn fetch_app_update_manifest(url: &str) -> Result<AppUpdateManifest, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(4))
        .build()
        .map_err(|e| format!("创建软件更新检查客户端失败：{e}"))?;
    let manifest = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("检查软件更新失败：{e}"))?
        .error_for_status()
        .map_err(|e| format!("检查软件更新失败：{e}"))?
        .text()
        .await
        .map_err(|e| format!("读取软件更新清单失败：{e}"))?;
    let manifest: AppUpdateManifest = serde_json::from_str(manifest.trim_start_matches('\u{feff}'))
        .map_err(|e| format!("解析软件更新清单失败：{e}"))?;
    if manifest.version.trim().is_empty() {
        return Err("软件更新清单缺少 version 或 url".to_string());
    }
    Ok(manifest)
}

fn compare_versions(remote: &str, current: &str) -> std::cmp::Ordering {
    let remote = remote.trim().trim_start_matches('v');
    let current = current.trim().trim_start_matches('v');
    if remote == current {
        return std::cmp::Ordering::Equal;
    }
    let remote_parts = version_parts(remote);
    let current_parts = version_parts(current);
    let len = remote_parts.len().max(current_parts.len()).max(1);
    for idx in 0..len {
        let left = *remote_parts.get(idx).unwrap_or(&0);
        let right = *current_parts.get(idx).unwrap_or(&0);
        match left.cmp(&right) {
            std::cmp::Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    std::cmp::Ordering::Equal
}

fn version_parts(version: &str) -> Vec<u64> {
    version
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<u64>().unwrap_or(0))
        .collect()
}

async fn capabilities(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let expert_installed = ctx.models.is_installed("expert");
    let expert_caps = ctx.models.expert_capabilities();
    let face_aware = expert_installed
        && expert_caps.insightface_detection
        && expert_caps.insightface_recognition
        && expert_caps.insightface_landmark;
    let quality_models = expert_installed && expert_caps.quality_models;
    let llm_status = ctx.llm.provider_status();
    let tycoon_ready = llm_status.configured;
    let mut engines = ctx.models.available_engines();
    engines.push("tycoon".to_string());
    engines.sort();
    engines.dedup();
    Json(json!({
        "face_aware": face_aware,
        "engines": engines,
        "backend": "rust-fast",
        "rust_fast": true,
        "watermark": true,
        "expert_installed": expert_installed,
        "expert_capabilities": expert_caps,
        "quality_models": quality_models,
        "nima_legacy_unavailable": true,
        "opencv_orb": opencv_orb_available(),
        "formats": format_capabilities(),
        "tycoon_ready": tycoon_ready,
        "model_components": ctx.models.status_map(),
        "llm_provider": {
            "configured": llm_status.configured,
            "key_configured": llm_status.key_configured,
            "protocols": llm_status.protocols
        },
        "install_mode": "base",
        "python_required": false
    }))
}

fn format_capabilities() -> Value {
    json!({
        "raw_thumbnail": true,
        "raw_strategy": "embedded_jpeg",
        "heic": heic_decode_available(),
        "heic_strategy": heic_decode_strategy(),
        "opencv_orb": opencv_orb_available(),
    })
}

fn opencv_orb_available() -> bool {
    cfg!(feature = "opencv-orb")
}

fn heic_decode_available() -> bool {
    heic_runtime_available()
}

fn heic_decode_strategy() -> &'static str {
    if heic_decode_available() {
        "windows_wic"
    } else {
        "unavailable"
    }
}

#[cfg(windows)]
fn heic_runtime_available() -> bool {
    windows_heif_decoder_available().unwrap_or(false)
}

#[cfg(not(windows))]
fn heic_runtime_available() -> bool {
    false
}

async fn model_components(State(ctx): State<AppCtx>) -> impl IntoResponse {
    Json(json!({
        "components": ctx.models.list(),
        "cache_dir": ctx.models.cache_dir().to_string_lossy(),
        "backend": "rust-fast"
    }))
}

async fn model_component_install(
    State(ctx): State<AppCtx>,
    Json(req): Json<ComponentInstallRequest>,
) -> impl IntoResponse {
    match ctx.models.install(req).await {
        Ok((status, body)) | Err((status, body)) => (status, Json(body)),
    }
}

async fn model_component_delete(
    State(ctx): State<AppCtx>,
    Json(req): Json<ComponentInstallRequest>,
) -> impl IntoResponse {
    match ctx.models.delete(&req.id) {
        Ok(component) => (
            StatusCode::OK,
            Json(json!({
                "ok": true,
                "component": component,
                "cache_dir": ctx.models.cache_dir().to_string_lossy()
            })),
        ),
        Err((status, body)) => (status, Json(body)),
    }
}

async fn model_component_pause(
    State(ctx): State<AppCtx>,
    Json(req): Json<ComponentInstallRequest>,
) -> impl IntoResponse {
    match ctx.models.pause(&req.id) {
        Ok(component) => (
            StatusCode::OK,
            Json(json!({
                "ok": true,
                "component": component,
                "cache_dir": ctx.models.cache_dir().to_string_lossy()
            })),
        ),
        Err((status, body)) => (status, Json(body)),
    }
}

async fn model_component_cancel(
    State(ctx): State<AppCtx>,
    Json(req): Json<ComponentInstallRequest>,
) -> impl IntoResponse {
    match ctx.models.cancel(&req.id) {
        Ok(component) => (
            StatusCode::OK,
            Json(json!({
                "ok": true,
                "component": component,
                "cache_dir": ctx.models.cache_dir().to_string_lossy()
            })),
        ),
        Err((status, body)) => (status, Json(body)),
    }
}

async fn llm_provider_status(State(ctx): State<AppCtx>) -> impl IntoResponse {
    Json(json!(ctx.llm.provider_status()))
}

async fn llm_provider_save(
    State(ctx): State<AppCtx>,
    Json(req): Json<SaveProviderRequest>,
) -> impl IntoResponse {
    match ctx.llm.save_provider(req) {
        Ok(status) => (StatusCode::OK, Json(json!(status))),
        Err(err) => err.into_json(),
    }
}

async fn llm_provider_clear(State(ctx): State<AppCtx>) -> impl IntoResponse {
    match ctx.llm.clear_provider() {
        Ok(()) => (StatusCode::OK, Json(json!({"ok": true}))),
        Err(err) => err.into_json(),
    }
}

async fn llm_models(State(ctx): State<AppCtx>) -> impl IntoResponse {
    match ctx.llm.list_models().await {
        Ok(body) => (StatusCode::OK, Json(body)),
        Err(err) => err.into_json(),
    }
}

async fn llm_test(
    State(ctx): State<AppCtx>,
    Json(req): Json<Option<SaveProviderRequest>>,
) -> impl IntoResponse {
    match ctx.llm.test_provider(req).await {
        Ok(body) => (StatusCode::OK, Json(body)),
        Err(err) => err.into_json(),
    }
}

async fn ark_key_status(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let status = ctx.llm.provider_status();
    Json(json!({
        "configured": status.key_configured,
        "source": if status.key_configured { json!("file") } else { Value::Null },
        "masked": if status.key_configured { json!("***") } else { Value::Null },
        "deprecated": true,
        "provider": status
    }))
}

async fn ark_key_save_compat() -> impl IntoResponse {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": "Rust 版已改用通用 AI 服务商配置，请使用 /api/llm/provider",
            "deprecated": true
        })),
    )
}

async fn ark_key_clear(State(ctx): State<AppCtx>) -> impl IntoResponse {
    llm_provider_clear(State(ctx)).await
}

async fn llm_concurrency(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let status = ctx.llm.provider_status();
    let limit = status
        .config
        .as_ref()
        .map(|c| c.max_concurrency)
        .unwrap_or(0);
    Json(json!({"limit": limit}))
}

#[derive(Debug, Deserialize)]
struct JobLogQuery {
    name: Option<String>,
}

async fn job_log(State(ctx): State<AppCtx>, Query(q): Query<JobLogQuery>) -> Response {
    let folder = {
        let state = ctx.inner.lock().expect("backend state lock");
        state.session.as_ref().map(|s| s.folder.clone())
    };
    let Some(folder) = folder else {
        return json_error(StatusCode::BAD_REQUEST, "no session");
    };

    let jobs_dir = Path::new(&folder).join(PIC_DIR).join("jobs");
    if let Some(name) = q.name.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        if name.contains('/')
            || name.contains('\\')
            || name.contains("..")
            || !name.ends_with(".log")
        {
            return json_error(StatusCode::BAD_REQUEST, "非法文件名");
        }
        let target = jobs_dir.join(name);
        if !target.exists() {
            return json_error(StatusCode::NOT_FOUND, "文件不存在");
        }
        match fs::read_to_string(&target) {
            Ok(content) => {
                return (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                    content,
                )
                    .into_response();
            }
            Err(err) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("读取日志失败: {err}"),
                );
            }
        }
    }

    if !jobs_dir.exists() {
        return Json(json!({"logs": []})).into_response();
    }

    let mut logs = match fs::read_dir(&jobs_dir) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("log") {
                    return None;
                }
                let meta = entry.metadata().ok()?;
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                Some(json!({
                    "name": entry.file_name().to_string_lossy(),
                    "size": meta.len(),
                    "mtime": mtime,
                }))
            })
            .collect::<Vec<_>>(),
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("读取日志目录失败: {err}"),
            );
        }
    };

    logs.sort_by(|a, b| {
        b["mtime"]
            .as_f64()
            .partial_cmp(&a["mtime"].as_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    logs.truncate(50);
    Json(json!({"logs": logs})).into_response()
}

async fn watermark_templates(State(ctx): State<AppCtx>) -> impl IntoResponse {
    Json(json!({
        "templates": watermark::list_templates(),
        "logos": watermark::available_logos(&ctx.backend_dir),
    }))
}

async fn watermark_preview(State(ctx): State<AppCtx>, Json(req): Json<Value>) -> Response {
    let cfg = serde_json::from_value::<WatermarkConfig>(req.clone()).unwrap_or_default();
    let winners = {
        let state = ctx.inner.lock().expect("backend state lock");
        watermark_winner_paths(state.session.as_ref())
    };
    if winners.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "没有 winner 照片可预览");
    }
    let idx = cfg.preview_index.min(winners.len().saturating_sub(1));
    let src = PathBuf::from(&winners[idx]);
    match watermark::render(&ctx.backend_dir, &src, &cfg, Some(1400)) {
        Ok(data) => {
            let exif = watermark::parse_exif(&src);
            Json(json!({
                "image_b64": BASE64_STANDARD.encode(&data),
                "size_kb": (data.len() as f64 / 1024.0 * 10.0).round() / 10.0,
                "source_name": src.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                "total_winners": winners.len(),
                "preview_index": idx,
                "exif": exif,
            }))
            .into_response()
        }
        Err(err) => json_error(StatusCode::INTERNAL_SERVER_ERROR, &err),
    }
}

async fn watermark_start(State(ctx): State<AppCtx>, Json(req): Json<Value>) -> Response {
    let cfg = serde_json::from_value::<WatermarkConfig>(req).unwrap_or_default();
    let winners = {
        let state = ctx.inner.lock().expect("backend state lock");
        if state.watermark.status == "running" {
            return json_error(StatusCode::CONFLICT, "已有水印任务正在运行");
        }
        watermark_winner_paths(state.session.as_ref())
    };
    if winners.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "没有 winner 照片可导出");
    }

    let folder = {
        let state = ctx.inner.lock().expect("backend state lock");
        state.session.as_ref().map(|s| s.folder.clone())
    };
    let Some(folder) = folder else {
        return json_error(StatusCode::BAD_REQUEST, "当前没有可用会话");
    };

    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let out_dir = Path::new(&folder)
        .join("winners")
        .join(format!("watermarked_{stamp}"));
    if let Err(err) = fs::create_dir_all(&out_dir) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("创建输出目录失败: {err}"),
        );
    }

    {
        let mut state = ctx.inner.lock().expect("backend state lock");
        state.watermark = WatermarkState {
            status: "running".to_string(),
            total: winners.len(),
            out_dir: Some(out_dir.to_string_lossy().to_string()),
            started_at: now_secs(),
            ..WatermarkState::default()
        };
    }

    let total = winners.len();
    let out_dir_text = out_dir.to_string_lossy().to_string();
    let worker_ctx = ctx.clone();
    thread::spawn(move || run_watermark_job(worker_ctx, winners, out_dir, cfg));
    Json(json!({"ok": true, "total": total, "out_dir": out_dir_text})).into_response()
}

async fn watermark_status(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let state = ctx.inner.lock().expect("backend state lock");
    let wm = state.watermark.clone();
    let end = if wm.finished_at > 0.0 {
        wm.finished_at
    } else {
        now_secs()
    };
    let elapsed = if wm.started_at > 0.0 {
        (end - wm.started_at).max(0.0)
    } else {
        0.0
    };
    let failed_count = wm.failed.len();
    let failed_sample = wm.failed.iter().take(5).cloned().collect::<Vec<_>>();
    Json(json!({
        "status": wm.status,
        "done": wm.done,
        "total": wm.total,
        "current": wm.current,
        "out_dir": wm.out_dir,
        "ok": wm.ok,
        "failed": wm.failed,
        "failed_count": failed_count,
        "failed_sample": failed_sample,
        "error": wm.error,
        "elapsed": elapsed,
    }))
}

async fn watermark_cancel(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let mut state = ctx.inner.lock().expect("backend state lock");
    if state.watermark.status != "running" {
        return json_error(StatusCode::BAD_REQUEST, "没有运行中的水印任务");
    }
    state.watermark.cancel_requested = true;
    Json(json!({"ok": true})).into_response()
}

async fn watermark_open_out_dir(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let out_dir = {
        let state = ctx.inner.lock().expect("backend state lock");
        state.watermark.out_dir.clone()
    };
    let Some(out_dir) = out_dir else {
        return json_error(StatusCode::BAD_REQUEST, "没有可打开的输出目录");
    };
    let path = PathBuf::from(out_dir);
    if !path.exists() {
        return json_error(StatusCode::BAD_REQUEST, "输出目录不存在");
    }
    let result = if cfg!(windows) {
        Command::new("explorer").arg(&path).spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(&path).spawn()
    } else {
        Command::new("xdg-open").arg(&path).spawn()
    };
    match result {
        Ok(_) => Json(json!({"ok": true})).into_response(),
        Err(err) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("打开失败: {err}"),
        ),
    }
}

async fn browse_folder(State(ctx): State<AppCtx>) -> impl IntoResponse {
    if let Some(picker) = ctx.folder_picker.clone() {
        return match (picker)() {
            Ok(Some(path)) => Json(json!({
                "ok": true,
                "folder": path.to_string_lossy(),
            }))
            .into_response(),
            Ok(None) => Json(json!({"ok": true, "cancelled": true})).into_response(),
            Err(err) => json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("打开选择对话框失败: {err}"),
            ),
        };
    }
    json_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "当前运行环境不支持系统选择对话框，请手动粘贴照片文件夹路径。",
    )
}

async fn start_job(State(ctx): State<AppCtx>, Json(req): Json<StartRequest>) -> Response {
    if req.engine != "fast" && req.engine != "tycoon" && req.engine != "expert" {
        return json_error(
            StatusCode::BAD_REQUEST,
            "Rust backend only supports fast, expert, or tycoon for now",
        );
    }
    if req.engine == "expert" && !ctx.models.is_installed("expert") {
        return json_error(
            StatusCode::PRECONDITION_REQUIRED,
            "Expert 模型组件尚未安装，请先安装增强组件",
        );
    }
    if req.mode != "copy" && req.mode != "move" {
        return json_error(
            StatusCode::BAD_REQUEST,
            "Rust Fast 后端仅支持 copy / move 模式",
        );
    }
    let folder = PathBuf::from(&req.folder);
    if !folder.is_dir() {
        return json_error(StatusCode::BAD_REQUEST, "照片文件夹不存在");
    }

    {
        let mut state = ctx.inner.lock().expect("backend state lock");
        state.session = None;
        state.last_infos.clear();
        state.grouping = GroupingState {
            status: "idle".to_string(),
            ..GroupingState::default()
        };
        state.job = JobState {
            folder: req.folder.clone(),
            dry_run: req.dry_run,
            mode: req.mode.clone(),
            engine: req.engine.clone(),
            status: "scanning".to_string(),
            label: "扫描文件夹…".to_string(),
            started_at: now_secs(),
            threshold_near: req.threshold_near,
            threshold_far: req.threshold_far,
            near_seconds: req.near_seconds,
            prescreen_enabled: req.prescreen_enabled,
            prescreen_strength: req.prescreen_strength.clone(),
            ..JobState::default()
        };
    }

    let worker_ctx = ctx.clone();
    thread::spawn(move || run_job(worker_ctx, req));
    Json(json!({"started": true, "backend": "rust-fast"})).into_response()
}

fn run_job(ctx: AppCtx, req: StartRequest) {
    let expert_dir = if req.engine == "expert" || req.engine == "tycoon" {
        match ctx.models.installed_dir("expert") {
            Ok(dir) => Some(dir),
            Err(err) if req.engine == "tycoon" => {
                let mut state = ctx.inner.lock().expect("backend state lock");
                state.job.status = "error".to_string();
                state.job.error = Some(format!(
                    "Tycoon 模式需要先安装 Expert DINOv2 + 人脸组件: {err}"
                ));
                state.job.finished_at = now_secs();
                return;
            }
            Err(err) => {
                let mut state = ctx.inner.lock().expect("backend state lock");
                state.job.status = "error".to_string();
                state.job.error = Some(err);
                state.job.finished_at = now_secs();
                return;
            }
        }
    } else {
        None
    };
    let mut expert_model = if let Some(dir) = &expert_dir {
        match expert_vision::Dinov2Model::from_component_dir(dir) {
            Ok(model) => Some(model),
            Err(err) => {
                let mut state = ctx.inner.lock().expect("backend state lock");
                state.job.status = "error".to_string();
                state.job.error = Some(err);
                state.job.finished_at = now_secs();
                return;
            }
        }
    } else {
        None
    };
    let mut face_model = if let Some(dir) = &expert_dir {
        if expert_vision::InsightFaceModels::component_files_ready(dir) {
            match expert_vision::InsightFaceModels::from_component_dir(dir) {
                Ok(model) => Some(model),
                Err(err) if req.engine == "tycoon" => {
                    let mut state = ctx.inner.lock().expect("backend state lock");
                    state.job.status = "error".to_string();
                    state.job.error = Some(format!(
                        "Tycoon 模式需要可用的人脸组件，当前加载失败: {err}"
                    ));
                    state.job.finished_at = now_secs();
                    return;
                }
                Err(_) => None,
            }
        } else if req.engine == "tycoon" {
            let mut state = ctx.inner.lock().expect("backend state lock");
            state.job.status = "error".to_string();
            state.job.error =
                Some("Tycoon 模式需要 Expert 人脸组件：det_10g / w600k_r50 / 1k3d68".to_string());
            state.job.finished_at = now_secs();
            return;
        } else {
            None
        }
    } else {
        None
    };
    let mut quality_model = if req.engine == "expert" {
        if let Some(dir) = &expert_dir {
            if expert_vision::ExpertQualityModels::component_files_ready(dir) {
                match expert_vision::ExpertQualityModels::from_component_dir(dir) {
                    Ok(model) => Some(model),
                    Err(err) => {
                        let mut state = ctx.inner.lock().expect("backend state lock");
                        state.job.label =
                            format!("Expert 质量模型暂不可用，继续使用 DINOv2/人脸能力: {err}");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    let tycoon_config = if req.engine == "tycoon" {
        match ctx.llm.require_tycoon_ready(req.llm_model.as_deref()) {
            Ok(config) => Some(config),
            Err(err) => {
                let mut state = ctx.inner.lock().expect("backend state lock");
                state.job.status = "error".to_string();
                state.job.error = Some(err.message);
                state.job.finished_at = now_secs();
                return;
            }
        }
    } else {
        None
    };
    let tycoon_runtime = tycoon_config
        .as_ref()
        .map(|_| tokio::runtime::Runtime::new().expect("create tycoon runtime"));

    let folder = PathBuf::from(&req.folder);
    let pairs = scan_folder(&folder);
    {
        let mut state = ctx.inner.lock().expect("backend state lock");
        state.job.total = pairs.len();
        state.job.status = "hashing".to_string();
        state.job.label = "读取图片与计算 Fast 特征…".to_string();
    }

    let mut infos = Vec::new();
    let mut skipped = Vec::new();
    for (idx, pair) in pairs.into_iter().enumerate() {
        if ctx
            .inner
            .lock()
            .expect("backend state lock")
            .job
            .cancel_requested
        {
            let mut state = ctx.inner.lock().expect("backend state lock");
            state.job.status = "cancelled".to_string();
            state.job.finished_at = now_secs();
            return;
        }

        match process_one(&pair, &req.prescreen_strength) {
            Ok(mut record) => {
                if let Some(model) = expert_model.as_mut() {
                    let Some(analysis) = pair.analysis.as_ref() else {
                        let mut state = ctx.inner.lock().expect("backend state lock");
                        state.job.status = "error".to_string();
                        state.job.error =
                            Some("Expert 模式需要可解码的 JPG/PNG/WebP/TIFF companion".to_string());
                        state.job.finished_at = now_secs();
                        return;
                    };
                    match model.extract_path(analysis) {
                        Ok(dinov2) => apply_dinov2_embedding(&mut record, dinov2),
                        Err(reason) => {
                            let mut state = ctx.inner.lock().expect("backend state lock");
                            state.job.status = "error".to_string();
                            state.job.error = Some(reason);
                            state.job.finished_at = now_secs();
                            return;
                        }
                    }
                }
                if let Some(model) = face_model.as_mut() {
                    let Some(analysis) = pair.analysis.as_ref() else {
                        let mut state = ctx.inner.lock().expect("backend state lock");
                        state.job.status = "error".to_string();
                        state.job.error = Some(
                            "Expert/Tycoon 人脸模式需要可解码的 JPG/PNG/WebP/TIFF companion"
                                .to_string(),
                        );
                        state.job.finished_at = now_secs();
                        return;
                    };
                    match apply_face_analysis(model, analysis, &mut record) {
                        Ok(()) => {}
                        Err(reason) if req.engine == "tycoon" => {
                            let mut state = ctx.inner.lock().expect("backend state lock");
                            state.job.status = "error".to_string();
                            state.job.error = Some(reason);
                            state.job.finished_at = now_secs();
                            return;
                        }
                        Err(reason) => {
                            let mut quality = record.info.quality.clone().unwrap_or_default();
                            quality
                                .extra
                                .insert("face_unavailable".to_string(), json!(reason));
                            record.info.quality = Some(quality);
                        }
                    }
                }
                if req.engine == "expert" {
                    apply_expert_quality_availability(&mut record, quality_model.is_some(), None);
                    if let Some(model) = quality_model.as_mut() {
                        let Some(analysis) = pair.analysis.as_ref() else {
                            let mut quality = record.info.quality.clone().unwrap_or_default();
                            quality.extra.insert(
                                "quality_models_unavailable".to_string(),
                                json!("Expert 质量模型需要可解码的 JPG/PNG/WebP/TIFF companion"),
                            );
                            record.info.quality = Some(quality);
                            push_job_event(&ctx, &record, None);
                            infos.push(record);
                            continue;
                        };
                        match apply_expert_quality_models(
                            model,
                            analysis,
                            &mut record,
                            &req.prescreen_strength,
                        ) {
                            Ok(()) => {}
                            Err(reason) => {
                                apply_expert_quality_availability(&mut record, false, Some(reason));
                            }
                        }
                    }
                }
                if let Some(config) = &tycoon_config {
                    match run_tycoon_judge(
                        &ctx,
                        config,
                        tycoon_runtime.as_ref().expect("tycoon runtime"),
                        &pair,
                        &req.prescreen_strength,
                    ) {
                        Ok(verdict) => apply_llm_verdict(&mut record, &verdict),
                        Err(reason) => {
                            let mut state = ctx.inner.lock().expect("backend state lock");
                            state.job.status = "error".to_string();
                            state.job.error = Some(reason);
                            state.job.finished_at = now_secs();
                            return;
                        }
                    }
                }
                push_job_event(&ctx, &record, None);
                infos.push(record);
            }
            Err(reason) => {
                let item = SkippedItem {
                    path: pair.primary.to_string_lossy().to_string(),
                    reason,
                };
                push_skip_event(&ctx, &item);
                skipped.push(item);
            }
        }
        let mut state = ctx.inner.lock().expect("backend state lock");
        state.job.done = idx + 1;
        state.job.skipped = skipped.clone();
    }

    {
        let mut state = ctx.inner.lock().expect("backend state lock");
        state.job.status = "grouping".to_string();
        state.job.label = "正在组连拍…".to_string();
    }

    let fast_infos = infos.iter().map(|r| r.info.clone()).collect::<Vec<_>>();
    let idx_groups = if req.engine == "expert" || req.engine == "tycoon" {
        expert_cluster(&fast_infos)
    } else {
        let orb_inliers = compute_orb_inliers_for_records(&infos);
        if orb_inliers.is_empty() {
            cluster(&fast_infos)
        } else {
            let mut options = FastClusterOptions::default();
            options.orb_inliers = orb_inliers;
            cluster_with_options(&fast_infos, &options)
        }
    };
    let mut groups = Vec::new();
    for group in idx_groups {
        let paths = group
            .into_iter()
            .filter_map(|i| infos.get(i))
            .map(|r| r.info.path.clone())
            .collect::<Vec<_>>();
        if !paths.is_empty() {
            groups.push(GroupState::new(paths));
        }
    }

    if groups.is_empty() && !infos.is_empty() {
        groups = infos
            .iter()
            .map(|r| GroupState::new(vec![r.info.path.clone()]))
            .collect();
    }

    let mut meta = HashMap::new();
    let mut companions = HashMap::new();
    let mut prescreen_rejected = Vec::new();
    let mut prescreen_reasons = HashMap::new();
    for record in &infos {
        meta.insert(record.info.path.clone(), meta_entry(record));
        if !record.companions.is_empty() {
            companions.insert(record.info.path.clone(), record.companions.clone());
        }
        if req.prescreen_enabled {
            if let Some(q) = &record.info.quality {
                if q.auto_reject.unwrap_or(false) {
                    let reason = q
                        .reject_reason
                        .clone()
                        .unwrap_or_else(|| "智能初筛".to_string());
                    prescreen_rejected.push(record.info.path.clone());
                    prescreen_reasons.insert(record.info.path.clone(), reason);
                }
            }
        }
    }

    for group in &mut groups {
        for path in &prescreen_rejected {
            if group.images.contains(path) && !group.auto_rejected.contains(path) {
                group.auto_rejected.push(path.clone());
                group.losers.push(path.clone());
                if let Some(reason) = prescreen_reasons.get(path) {
                    group
                        .auto_reject_reasons
                        .insert(path.clone(), reason.clone());
                }
            }
        }
        let rejected = group.auto_rejected.iter().cloned().collect::<HashSet<_>>();
        let live = group
            .images
            .iter()
            .filter(|p| !rejected.contains(*p))
            .cloned()
            .collect::<Vec<_>>();
        reset_group_live(group, live);
    }

    let all_paths = infos
        .iter()
        .map(|r| r.info.path.clone())
        .collect::<Vec<_>>();
    let grouping_cards = preview_cards(&groups, &meta, true);
    let mut session = SessionState {
        folder: req.folder.clone(),
        dry_run: req.dry_run,
        mode: req.mode.clone(),
        engine: req.engine.clone(),
        groups,
        current_group: 0,
        threshold_near: req.threshold_near,
        threshold_far: req.threshold_far,
        near_seconds: req.near_seconds,
        prescreen_enabled: req.prescreen_enabled,
        prescreen_strength: req.prescreen_strength,
        prescreen_reviewed: false,
        prescreen_rejected,
        prescreen_reject_reasons: prescreen_reasons,
        prescreen_restored: Vec::new(),
        undo_stack: Vec::new(),
        meta,
        companions,
        selection_started: false,
        skipped,
    };
    if !session.prescreen_enabled {
        let _ = apply_finished_groups(&mut session);
    }

    let _ = save_state(&session);
    let mut state = ctx.inner.lock().expect("backend state lock");
    state.grouping = GroupingState {
        status: "done".to_string(),
        groups: grouping_cards,
        all_paths,
        total: session.groups.len(),
        multi: session.groups.iter().filter(|g| g.images.len() > 1).count(),
        error: None,
    };
    state.last_infos = infos;
    state.session = Some(session);
    state.job.status = "done".to_string();
    state.job.label = "完成".to_string();
    state.job.finished_at = now_secs();
}

#[derive(Debug, Clone)]
struct ScanPair {
    primary: PathBuf,
    companions: Vec<PathBuf>,
    analysis: Option<PathBuf>,
}

fn scan_folder(folder: &Path) -> Vec<ScanPair> {
    let mut groups: HashMap<(PathBuf, String), Vec<PathBuf>> = HashMap::new();
    scan_dir(folder, folder, &mut groups);
    let mut out = Vec::new();
    for mut files in groups.into_values() {
        files.sort();
        let raws = files
            .iter()
            .filter(|p| RAW_EXTS.contains(&ext_lower(p).as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let images = files
            .iter()
            .filter(|p| IMAGE_EXTS.contains(&ext_lower(p).as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let heics = files
            .iter()
            .filter(|p| HEIC_EXTS.contains(&ext_lower(p).as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let sidecars = files
            .iter()
            .filter(|p| SIDECAR_EXTS.contains(&ext_lower(p).as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if let Some(primary) = raws.first().cloned() {
            let companions = raws
                .iter()
                .skip(1)
                .chain(images.iter())
                .chain(sidecars.iter())
                .cloned()
                .collect::<Vec<_>>();
            out.push(ScanPair {
                primary: primary.clone(),
                companions,
                analysis: images.first().cloned().or(Some(primary)),
            });
        } else if let Some(primary) = images.first().cloned() {
            out.push(ScanPair {
                primary: primary.clone(),
                companions: images
                    .iter()
                    .skip(1)
                    .chain(sidecars.iter())
                    .cloned()
                    .collect(),
                analysis: Some(primary),
            });
        } else if let Some(primary) = heics.first().cloned() {
            out.push(ScanPair {
                primary: primary.clone(),
                companions: heics
                    .iter()
                    .skip(1)
                    .chain(sidecars.iter())
                    .cloned()
                    .collect(),
                analysis: Some(primary),
            });
        }
    }
    out.sort_by(|a, b| a.primary.cmp(&b.primary));
    out
}

fn scan_dir(root: &Path, dir: &Path, groups: &mut HashMap<(PathBuf, String), Vec<PathBuf>>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if matches!(name, "winners" | "losers" | PIC_DIR) {
                continue;
            }
            scan_dir(root, &path, groups);
            continue;
        }
        let ext = ext_lower(&path);
        if !IMAGE_EXTS.contains(&ext.as_str())
            && !RAW_EXTS.contains(&ext.as_str())
            && !HEIC_EXTS.contains(&ext.as_str())
            && !SIDECAR_EXTS.contains(&ext.as_str())
        {
            continue;
        }
        let parent = path.parent().unwrap_or(root).to_path_buf();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        groups.entry((parent, stem)).or_default().push(path);
    }
}

fn run_tycoon_judge(
    ctx: &AppCtx,
    config: &llm_provider::ProviderConfig,
    runtime: &tokio::runtime::Runtime,
    pair: &ScanPair,
    strength: &str,
) -> Result<JudgeVerdict, String> {
    let analysis = pair
        .analysis
        .as_ref()
        .ok_or_else(|| "tycoon requires a decodable companion image".to_string())?;
    let (data_url, _bytes, _size) = image_to_data_url(analysis)?;
    let prompt = tycoon_prompt(strength);
    runtime
        .block_on(ctx.llm.judge_image(config, &data_url, &prompt))
        .map_err(|e| e.message)
}

fn apply_llm_verdict(record: &mut InfoRecord, verdict: &JudgeVerdict) {
    let is_reject = verdict.verdict.eq_ignore_ascii_case("reject");
    let mut quality = record.info.quality.clone().unwrap_or_default();
    if is_reject {
        quality.auto_reject = Some(true);
        quality.reject_reason = Some(verdict.reason.clone());
        if !quality.flags.iter().any(|f| f == "llm_reject") {
            quality.flags.push("llm_reject".to_string());
        }
    } else {
        quality.auto_reject = Some(false);
        if !quality.flags.iter().any(|f| f == "llm_pass") {
            quality.flags.push("llm_pass".to_string());
        }
    }
    quality
        .extra
        .insert("llm_verdict".to_string(), json!(verdict.verdict));
    quality
        .extra
        .insert("llm_reason".to_string(), json!(verdict.reason));
    if let Some(flaws) = &verdict.flaws {
        quality.extra.insert("llm_flaws".to_string(), json!(flaws));
    }
    if let Some(fixable) = &verdict.fixable {
        quality
            .extra
            .insert("llm_fixable".to_string(), json!(fixable));
    }
    record.info.quality = Some(quality);
}

fn apply_expert_quality_availability(
    record: &mut InfoRecord,
    available: bool,
    reason: Option<String>,
) {
    let mut quality = record.info.quality.clone().unwrap_or_default();
    quality
        .extra
        .insert("quality_models".to_string(), json!(available));
    quality
        .extra
        .insert("nima_legacy_unavailable".to_string(), json!(true));
    if let Some(reason) = reason {
        quality
            .extra
            .insert("quality_models_unavailable".to_string(), json!(reason));
    }
    record.info.quality = Some(quality);
}

fn apply_expert_quality_models(
    model: &mut expert_vision::ExpertQualityModels,
    analysis: &Path,
    record: &mut InfoRecord,
    strength: &str,
) -> Result<(), String> {
    let img = image::open(analysis).map_err(|e| format!("加载 Expert 质量分析图片失败: {e}"))?;
    let scores = model.analyze(&img)?;
    apply_expert_quality_scores(record, scores, strength);
    Ok(())
}

fn apply_expert_quality_scores(
    record: &mut InfoRecord,
    scores: expert_vision::ExpertQualityScores,
    strength: &str,
) {
    let mut quality = record.info.quality.clone().unwrap_or_default();
    quality
        .extra
        .insert("quality_models".to_string(), json!(true));
    quality
        .extra
        .insert("nima_legacy_unavailable".to_string(), json!(true));
    quality
        .extra
        .insert("aesthetic_score".to_string(), serde_json::Value::Null);
    if let Some(v) = scores.musiq_score {
        quality.extra.insert("musiq_score".to_string(), json!(v));
    }
    if let Some(v) = scores.clipiqa_score {
        quality.extra.insert("clipiqa_score".to_string(), json!(v));
    }
    apply_expert_aesthetic_rule(&mut quality, strength);
    record.info.quality = Some(quality);
}

fn apply_expert_aesthetic_rule(quality: &mut QualityInfo, strength: &str) {
    let (musiq_low, clipiqa_low) = match strength {
        "advanced" | "aggressive" => (68.0, 0.65),
        _ => (55.0, 0.55),
    };
    let musiq_low_hit = quality
        .extra
        .get("musiq_score")
        .and_then(|v| v.as_f64())
        .is_some_and(|v| v < musiq_low);
    let clipiqa_low_hit = quality
        .extra
        .get("clipiqa_score")
        .and_then(|v| v.as_f64())
        .is_some_and(|v| v < clipiqa_low);
    if musiq_low_hit && clipiqa_low_hit && !quality.flags.iter().any(|f| f == "low_aesthetic") {
        quality.flags.push("low_aesthetic".to_string());
    }
    if quality.flags.iter().any(|f| f == "low_aesthetic") {
        quality.auto_reject = Some(true);
        if quality.reject_reason.is_none() {
            quality.reject_reason = Some("美学评分偏低".to_string());
        }
        if let Some(score) = quality.quality_score {
            quality.quality_score = Some((score - 8.0).max(0.0));
        }
    }
}

fn apply_dinov2_embedding(record: &mut InfoRecord, dinov2: Vec<f32>) {
    let mut quality = record.info.quality.clone().unwrap_or_default();
    quality
        .extra
        .insert("dinov2_dim".to_string(), json!(dinov2.len()));
    quality
        .extra
        .insert("expert_stage".to_string(), json!("dinov2"));
    record.info.quality = Some(quality);
    record.info.dinov2 = Some(dinov2);
}

fn apply_face_analysis(
    model: &mut expert_vision::InsightFaceModels,
    analysis: &Path,
    record: &mut InfoRecord,
) -> Result<(), String> {
    let (img, faces) = model.extract_path(analysis)?;
    apply_face_infos(record, &img, faces);
    Ok(())
}

fn apply_face_infos(
    record: &mut InfoRecord,
    img: &DynamicImage,
    faces: Vec<expert_vision::FaceInfo>,
) {
    let signals = expert_vision::face_signals_from_data(&faces, img);
    let mut quality = record.info.quality.clone().unwrap_or_default();
    quality
        .extra
        .insert("face_count".to_string(), json!(signals.face_count));
    if let Some(v) = signals.face_sharpness {
        quality.extra.insert("face_sharpness".to_string(), json!(v));
    }
    quality
        .extra
        .insert("face_clipped".to_string(), json!(signals.face_clipped));
    if let Some(v) = signals.eyes_open_score {
        quality
            .extra
            .insert("eyes_open_score".to_string(), json!(v));
    }
    if let Some(v) = signals.face_area_ratio {
        quality
            .extra
            .insert("face_area_ratio".to_string(), json!(v));
    }
    if let Some(v) = signals.det_score {
        quality.extra.insert("det_score".to_string(), json!(v));
    }
    quality
        .extra
        .insert("faces_detail".to_string(), json!(signals.faces_detail));
    apply_face_quality_flags(&mut quality);
    record.info.face_embeddings = faces.into_iter().map(|face| face.embedding).collect();
    record.info.quality = Some(quality);
}

fn apply_face_quality_flags(quality: &mut QualityInfo) {
    let face_count = quality
        .extra
        .get("face_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let face_sharp = quality.extra.get("face_sharpness").and_then(|v| v.as_f64());
    let eyes = quality
        .extra
        .get("eyes_open_score")
        .and_then(|v| v.as_f64());
    let clipped = quality
        .extra
        .get("face_clipped")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if face_count > 0 {
        if let Some(sharp) = face_sharp {
            if sharp < 60.0 && !quality.flags.iter().any(|f| f == "face_very_blurry") {
                quality.flags.push("face_very_blurry".to_string());
            } else if sharp < 120.0 && !quality.flags.iter().any(|f| f == "face_blurry") {
                quality.flags.push("face_blurry".to_string());
            }
        }
        if let Some(eye) = eyes {
            if eye < 0.22 && !quality.flags.iter().any(|f| f == "eyes_closed") {
                quality.flags.push("eyes_closed".to_string());
            }
        }
        if clipped && !quality.flags.iter().any(|f| f == "face_clipped") {
            quality.flags.push("face_clipped".to_string());
        }
    }
    if quality.flags.iter().any(|f| {
        matches!(
            f.as_str(),
            "face_very_blurry" | "eyes_closed" | "all_eyes_closed" | "face_clipped"
        )
    }) {
        quality.auto_reject = Some(true);
        if quality.reject_reason.is_none() {
            quality.reject_reason = Some("人脸质量不达标".to_string());
        }
    }
}

fn process_one(pair: &ScanPair, strength: &str) -> Result<InfoRecord, String> {
    let analysis = pair.analysis.as_ref().ok_or_else(|| {
        if RAW_EXTS.contains(&ext_lower(&pair.primary).as_str()) {
            "RAW without same-stem JPG could not be queued for analysis".to_string()
        } else {
            "HEIC/HEIF could not be queued for analysis".to_string()
        }
    })?;
    let img = load_fast_analysis_image(analysis)?;
    let meta = fs::metadata(&pair.primary).map_err(|e| format!("stat 失败: {e}"))?;
    let file_size = meta.len();
    let mtime = meta
        .modified()
        .ok()
        .and_then(system_time_secs)
        .unwrap_or_else(now_secs);
    let (width, height) = img.dimensions();
    let (exif_width, exif_height) = image::image_dimensions(analysis).unwrap_or((width, height));
    let exif = ExifSummary {
        width: Some(exif_width),
        height: Some(exif_height),
        file_size: Some(file_size),
        datetime: meta
            .modified()
            .ok()
            .map(|t| DateTime::<Local>::from(t).naive_local().isoformat()),
        ..ExifSummary::default()
    };

    let gray8 = pillow_luma(&img.to_rgb8());
    let small_a = expert_vision::pillow_lanczos_resize_luma(&gray8, 8, 8);
    let small_d = expert_vision::pillow_lanczos_resize_luma(&gray8, 9, 8);
    let small_p = expert_vision::pillow_lanczos_resize_luma(&gray8, 32, 32);
    let whash_scale = whash_image_scale(width, height);
    let small_w = expert_vision::pillow_lanczos_resize_luma(&gray8, whash_scale, whash_scale);
    let ahash = average_hash_from_luma(small_a.as_raw(), 8).unwrap_or_default();
    let dhash = difference_hash_from_luma(small_d.as_raw(), 8).unwrap_or_default();
    let phash = perceptual_hash_from_luma(small_p.as_raw(), 8).unwrap_or_default();
    let whash =
        wavelet_hash_from_luma(small_w.as_raw(), 8, whash_scale as usize).unwrap_or_default();

    let signals = quality_signals(&img, file_size);
    let quality_result = analyze_from_signals(&signals, FastQualityProfile::from_name(strength));
    let quality = QualityInfo {
        blur_score: Some(round3(signals.blur_score)),
        brightness_mean: Some(round3(signals.brightness_mean)),
        brightness_std: Some(round3(signals.brightness_std)),
        contrast_score: Some(round3(signals.contrast_score)),
        overexposed_ratio: Some(round5(signals.overexposed_ratio)),
        underexposed_ratio: Some(round5(signals.underexposed_ratio)),
        entropy: Some(round5(signals.entropy)),
        width: Some(width),
        height: Some(height),
        file_size: Some(file_size),
        quality_score: Some(quality_result.quality_score),
        flags: quality_result.flags,
        auto_reject: Some(quality_result.auto_reject),
        reject_reason: quality_result.reject_reason,
        blur_combined: Some(round3(signals.blur_combined)),
        motion_anisotropy: Some(round3(signals.motion_anisotropy)),
        edge_width_pix: signals.edge_width_pix.map(round2),
        focus_ratio: signals.focus_ratio.map(round3),
        horizon_tilt_deg: signals.horizon_tilt_deg.map(round2),
        composition: signals.composition.map(round3),
        salient_sharpness: signals.salient_sharpness.map(round3),
        extra: HashMap::new(),
    };

    let info = FastImageInfo {
        path: pair.primary.to_string_lossy().to_string(),
        phash: Some(phash),
        dhash: Some(dhash),
        whash: Some(whash),
        ahash: Some(ahash),
        timestamp: None,
        size: Some(file_size),
        mtime: Some(mtime),
        exif_summary: Some(exif),
        quality: Some(quality),
        color_hist: compute_color_hist(&img),
        orb_descs_len: None,
        orb_kps_len: None,
        dinov2: None,
        face_embeddings: Vec::new(),
    };
    Ok(InfoRecord {
        info,
        companions: pair
            .companions
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
    })
}

fn load_fast_analysis_image(path: &Path) -> Result<DynamicImage, String> {
    let rgb = load_rgb_image(path)?;
    let oriented = apply_exif_orientation(rgb, read_exif_orientation(path).unwrap_or(1));
    let (w, h) = oriented.dimensions();
    let analysis = if w.max(h) <= ANALYSIS_MAX_SIDE {
        oriented
    } else {
        let scale = ANALYSIS_MAX_SIDE as f32 / w.max(h) as f32;
        let new_w = ((w as f32 * scale) as u32).max(1);
        let new_h = ((h as f32 * scale) as u32).max(1);
        expert_vision::pillow_lanczos_resize_rgb(&oriented, new_w, new_h)
    };
    Ok(DynamicImage::ImageRgb8(analysis))
}

fn load_rgb_image(path: &Path) -> Result<image::RgbImage, String> {
    let ext = ext_lower(path);
    if RAW_EXTS.contains(&ext.as_str()) {
        let bytes = fs::read(path).map_err(|e| format!("读取 RAW 文件失败: {e}"))?;
        let jpeg = extract_embedded_jpeg_preview(&bytes).map_err(|e| e.to_message())?;
        return image::load_from_memory_with_format(jpeg, ImageFormat::Jpeg)
            .map(|img| img.to_rgb8())
            .map_err(|e| format!("RAW 内嵌 JPEG 预览图解码失败: {e}"));
    }
    if HEIC_EXTS.contains(&ext.as_str()) {
        return load_heic_rgb_image(path);
    }
    #[cfg(feature = "opencv-orb")]
    {
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
            .unwrap_or(false)
        {
            let bytes = fs::read(path).map_err(|e| format!("读取 Fast JPEG 图片失败: {e}"))?;
            let rgb: image::RgbImage = turbojpeg::decompress_image(&bytes)
                .map_err(|e| format!("libjpeg-turbo 解码 Fast JPEG 失败: {e}"))?;
            return Ok(rgb);
        }
    }
    image::open(path)
        .map(|img| img.to_rgb8())
        .map_err(|e| format!("加载失败: {e}"))
}

#[derive(Debug)]
enum RawPreviewError {
    NoEmbeddedJpeg,
    DecodeFailed(String),
}

impl RawPreviewError {
    fn to_message(&self) -> String {
        match self {
            RawPreviewError::NoEmbeddedJpeg => "纯 RAW 暂未找到内嵌 JPEG 预览图".to_string(),
            RawPreviewError::DecodeFailed(e) => {
                format!("RAW 内嵌 JPEG 预览图解码失败: {e}")
            }
        }
    }
}

fn extract_embedded_jpeg_preview(bytes: &[u8]) -> Result<&[u8], RawPreviewError> {
    let mut best: Option<&[u8]> = None;
    let mut saw_jpeg = false;
    let mut last_decode_error: Option<String> = None;
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        if bytes[i] == 0xFF && bytes[i + 1] == 0xD8 {
            saw_jpeg = true;
            let start = i;
            let mut j = i + 2;
            while j + 1 < bytes.len() {
                if bytes[j] == 0xFF && bytes[j + 1] == 0xD9 {
                    let end = j + 2;
                    let candidate = &bytes[start..end];
                    match image::load_from_memory_with_format(candidate, ImageFormat::Jpeg) {
                        Ok(_) if best.map_or(true, |current| candidate.len() > current.len()) => {
                            best = Some(candidate);
                        }
                        Ok(_) => {}
                        Err(e) => last_decode_error = Some(e.to_string()),
                    }
                    i = end;
                    break;
                }
                j += 1;
            }
            if j + 1 >= bytes.len() {
                break;
            }
            continue;
        }
        i += 1;
    }
    if let Some(best) = best {
        Ok(best)
    } else if saw_jpeg {
        Err(RawPreviewError::DecodeFailed(
            last_decode_error.unwrap_or_else(|| "未找到可解码的 JPEG 片段".to_string()),
        ))
    } else {
        Err(RawPreviewError::NoEmbeddedJpeg)
    }
}

fn load_heic_rgb_image(path: &Path) -> Result<image::RgbImage, String> {
    #[cfg(windows)]
    {
        return load_heic_rgb_image_wic(path).map_err(|e| {
            format!("HEIC/HEIF 系统 WIC 解码失败，请确认 Windows HEIF 图像扩展已安装: {e}")
        });
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        Err("HEIC/HEIF 解码仅在 Windows WIC 路径下启用，当前平台暂未支持".to_string())
    }
}

#[cfg(windows)]
fn windows_wic_factory(
) -> Result<windows::Win32::Graphics::Imaging::IWICImagingFactory, windows::core::Error> {
    use windows::Win32::Graphics::Imaging::{CLSID_WICImagingFactory2, IWICImagingFactory};
    use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};

    unsafe {
        ensure_wic_com_initialized()?;
        CoCreateInstance::<_, IWICImagingFactory>(
            &CLSID_WICImagingFactory2,
            None,
            CLSCTX_INPROC_SERVER,
        )
    }
}

#[cfg(windows)]
fn ensure_wic_com_initialized() -> Result<(), windows::core::Error> {
    use std::cell::Cell;
    use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};

    thread_local! {
        static COM_INITIALIZED: Cell<bool> = const { Cell::new(false) };
    }

    COM_INITIALIZED.with(|initialized| {
        if initialized.get() {
            return Ok(());
        }
        unsafe {
            let hr = CoInitializeEx(None, COINIT_MULTITHREADED);
            if hr.is_err() && hr.0 as u32 != 0x80010106 {
                return Err(windows::core::Error::from_hresult(hr));
            }
        }
        initialized.set(true);
        Ok(())
    })
}

#[cfg(windows)]
fn windows_heif_decoder_available() -> Result<bool, windows::core::Error> {
    use windows::Win32::Graphics::Imaging::GUID_ContainerFormatHeif;

    let factory = windows_wic_factory()?;
    unsafe {
        Ok(factory
            .CreateDecoder(&GUID_ContainerFormatHeif, std::ptr::null())
            .is_ok())
    }
}

#[cfg(windows)]
fn load_heic_rgb_image_wic(path: &Path) -> Result<image::RgbImage, String> {
    use windows::core::{Interface, PCWSTR};
    use windows::Win32::Foundation::GENERIC_READ;
    use windows::Win32::Graphics::Imaging::{
        GUID_WICPixelFormat24bppRGB, IWICBitmapSource, WICBitmapDitherTypeNone,
        WICBitmapPaletteTypeCustom, WICDecodeMetadataCacheOnLoad,
    };

    let factory = windows_wic_factory().map_err(|e| format!("初始化 WIC 失败: {e}"))?;
    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();

    unsafe {
        let decoder = factory
            .CreateDecoderFromFilename(
                PCWSTR(wide.as_ptr()),
                None,
                GENERIC_READ,
                WICDecodeMetadataCacheOnLoad,
            )
            .map_err(|e| format!("打开 HEIC/HEIF 文件失败: {e}"))?;
        let frame = decoder
            .GetFrame(0)
            .map_err(|e| format!("读取 HEIC/HEIF 第一帧失败: {e}"))?;
        let converter = factory
            .CreateFormatConverter()
            .map_err(|e| format!("创建 WIC 像素格式转换器失败: {e}"))?;
        converter
            .Initialize(
                &frame,
                &GUID_WICPixelFormat24bppRGB,
                WICBitmapDitherTypeNone,
                None,
                0.0,
                WICBitmapPaletteTypeCustom,
            )
            .map_err(|e| format!("转换 HEIC/HEIF 到 RGB 失败: {e}"))?;

        let source: IWICBitmapSource = converter
            .cast()
            .map_err(|e| format!("读取 WIC RGB 图像失败: {e}"))?;
        let mut width = 0u32;
        let mut height = 0u32;
        source
            .GetSize(&mut width, &mut height)
            .map_err(|e| format!("读取 HEIC/HEIF 尺寸失败: {e}"))?;
        if width == 0 || height == 0 {
            return Err("HEIC/HEIF 图像尺寸为空".to_string());
        }
        let stride = width
            .checked_mul(3)
            .ok_or_else(|| "HEIC/HEIF 图像宽度过大".to_string())?;
        let len = stride
            .checked_mul(height)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| "HEIC/HEIF 图像尺寸过大".to_string())?;
        let mut buffer = vec![0u8; len];
        source
            .CopyPixels(std::ptr::null(), stride, &mut buffer)
            .map_err(|e| format!("复制 HEIC/HEIF 像素失败: {e}"))?;
        image::RgbImage::from_raw(width, height, buffer)
            .ok_or_else(|| "HEIC/HEIF RGB 缓冲区尺寸不匹配".to_string())
    }
}

fn read_exif_orientation(path: &Path) -> Option<u16> {
    let bytes = fs::read(path).ok()?;
    let mut cursor = Cursor::new(bytes);
    let exif = exif::Reader::new().read_from_container(&mut cursor).ok()?;
    let field = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    field.value.get_uint(0).map(|v| v as u16)
}

fn apply_exif_orientation(img: image::RgbImage, orientation: u16) -> image::RgbImage {
    match orientation {
        2 => image::imageops::flip_horizontal(&img),
        3 => image::imageops::rotate180(&img),
        4 => image::imageops::flip_vertical(&img),
        5 => image::imageops::rotate90(&image::imageops::flip_horizontal(&img)),
        6 => image::imageops::rotate90(&img),
        7 => image::imageops::rotate270(&image::imageops::flip_horizontal(&img)),
        8 => image::imageops::rotate270(&img),
        _ => img,
    }
}

fn pillow_luma(rgb: &image::RgbImage) -> image::GrayImage {
    let (w, h) = rgb.dimensions();
    let mut out = image::GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = rgb.get_pixel(x, y).0;
            let luma =
                (19595u32 * r as u32 + 38470u32 * g as u32 + 7471u32 * b as u32 + 32768) >> 16;
            out.put_pixel(x, y, image::Luma([luma.min(255) as u8]));
        }
    }
    out
}

#[cfg(feature = "opencv-orb")]
fn compute_orb_inliers_for_records(records: &[InfoRecord]) -> HashMap<(usize, usize), usize> {
    use opencv::{
        calib3d,
        core::{self, Mat, Point2f, Vector, NORM_HAMMING},
        features2d, imgproc,
        prelude::*,
    };

    struct OrbFeatures {
        descriptors: Mat,
        keypoints: Vec<Point2f>,
    }

    fn features_from_rgb(rgb: &image::RgbImage) -> Result<Option<OrbFeatures>, String> {
        let gray = pillow_luma(rgb);
        let (w, h) = gray.dimensions();
        if w < 32 || h < 32 {
            return Ok(None);
        }
        let src = Mat::from_slice(gray.as_raw())
            .map_err(|e| format!("create ORB Mat failed: {e}"))?
            .reshape(1, h as i32)
            .map_err(|e| format!("reshape ORB Mat failed: {e}"))?
            .try_clone()
            .map_err(|e| format!("clone ORB Mat failed: {e}"))?;
        let mut work = src;
        if w.max(h) > 800 {
            let scale = 800.0f64 / w.max(h) as f64;
            let size = core::Size::new(
                ((w as f64 * scale) as i32).max(1),
                ((h as f64 * scale) as i32).max(1),
            );
            let mut resized = Mat::default();
            imgproc::resize(&work, &mut resized, size, 0.0, 0.0, imgproc::INTER_AREA)
                .map_err(|e| format!("resize ORB image failed: {e}"))?;
            work = resized;
        }
        let mut orb = features2d::ORB::create(
            500,
            1.2,
            8,
            31,
            0,
            2,
            features2d::ORB_ScoreType::HARRIS_SCORE,
            31,
            20,
        )
        .map_err(|e| format!("create ORB failed: {e}"))?;
        let mut keypoints = Vector::<core::KeyPoint>::new();
        let mut descriptors = Mat::default();
        orb.detect_and_compute(
            &work,
            &Mat::default(),
            &mut keypoints,
            &mut descriptors,
            false,
        )
        .map_err(|e| format!("compute ORB failed: {e}"))?;
        if descriptors.empty() || descriptors.rows() < 8 {
            return Ok(None);
        }
        let points = keypoints.iter().map(|kp| kp.pt()).collect::<Vec<Point2f>>();
        if points.len() < 8 {
            return Ok(None);
        }
        Ok(Some(OrbFeatures {
            descriptors,
            keypoints: points,
        }))
    }

    fn features(path: &str) -> Result<Option<OrbFeatures>, String> {
        let img = load_fast_analysis_image(Path::new(path))?;
        features_from_rgb(&img.to_rgb8())
    }

    fn inliers(a: &OrbFeatures, b: &OrbFeatures) -> Result<usize, String> {
        let matcher = features2d::BFMatcher::create(NORM_HAMMING, true)
            .map_err(|e| format!("create BFMatcher failed: {e}"))?;
        let mut matches = Vector::<core::DMatch>::new();
        matcher
            .train_match_def(&a.descriptors, &b.descriptors, &mut matches)
            .map_err(|e| format!("match ORB descriptors failed: {e}"))?;
        let good = matches
            .iter()
            .filter(|m| m.distance < 60.0)
            .collect::<Vec<_>>();
        if good.len() < 8 {
            return Ok(0);
        }
        let mut pts_a = Vector::<Point2f>::new();
        let mut pts_b = Vector::<Point2f>::new();
        for m in good {
            let qa = m.query_idx as usize;
            let tb = m.train_idx as usize;
            if let (Some(pa), Some(pb)) = (a.keypoints.get(qa), b.keypoints.get(tb)) {
                pts_a.push(*pa);
                pts_b.push(*pb);
            }
        }
        if pts_a.len() < 8 {
            return Ok(0);
        }
        let mut mask = Mat::default();
        let _ = calib3d::find_homography(&pts_a, &pts_b, &mut mask, calib3d::RANSAC, 4.0)
            .map_err(|e| format!("find ORB homography failed: {e}"))?;
        if mask.empty() {
            return Ok(0);
        }
        let count =
            core::count_non_zero(&mask).map_err(|e| format!("read ORB mask failed: {e}"))?;
        Ok(count.max(0) as usize)
    }

    let features = records
        .iter()
        .map(|record| features(&record.info.path).ok().flatten())
        .collect::<Vec<_>>();
    let mut out = HashMap::new();
    for i in 0..features.len() {
        for j in (i + 1)..features.len() {
            let Some(a) = features[i].as_ref() else {
                continue;
            };
            let Some(b) = features[j].as_ref() else {
                continue;
            };
            if let Ok(value) = inliers(a, b) {
                if value > 0 {
                    out.insert((i, j), value);
                }
            }
        }
    }
    out
}

#[cfg(not(feature = "opencv-orb"))]
fn compute_orb_inliers_for_records(_records: &[InfoRecord]) -> HashMap<(usize, usize), usize> {
    HashMap::new()
}

fn image_to_data_url(path: &Path) -> Result<(String, usize, (u32, u32)), String> {
    let img = image::open(path).map_err(|e| format!("load image failed: {e}"))?;
    let (w, h) = img.dimensions();
    let max_side = 896u32;
    let resized = if w.max(h) > max_side {
        let scale = max_side as f32 / w.max(h) as f32;
        img.resize(
            (w as f32 * scale).round().max(1.0) as u32,
            (h as f32 * scale).round().max(1.0) as u32,
            FilterType::Lanczos3,
        )
    } else {
        img
    };
    let mut buf = Cursor::new(Vec::new());
    resized
        .write_to(&mut buf, ImageFormat::Jpeg)
        .map_err(|e| format!("encode jpeg failed: {e}"))?;
    let bytes = buf.into_inner();
    Ok((
        format!("data:image/jpeg;base64,{}", BASE64_STANDARD.encode(&bytes)),
        bytes.len(),
        resized.dimensions(),
    ))
}

fn tycoon_prompt(strength: &str) -> String {
    let _ = strength;
    "You are a photo quality judge. Return exactly one JSON object with keys verdict, reason, flaws, fixable. verdict must be pass or reject. reason must be short and concrete. flaws lists all defects. fixable is only meaningful when verdict is pass.".to_string()
}

fn expert_cluster(infos: &[FastImageInfo]) -> Vec<Vec<usize>> {
    const DISTANCE_THRESHOLD: f64 = 0.46;
    const HARD_BREAK_SECONDS: f64 = 45.0 * 60.0;
    let n = infos.len();
    if n == 0 {
        return Vec::new();
    }

    let mut sorted_idx = (0..n).collect::<Vec<_>>();
    sorted_idx.sort_by(|a, b| {
        expert_time_for_info(&infos[*a])
            .unwrap_or(0.0)
            .partial_cmp(&expert_time_for_info(&infos[*b]).unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut segments: Vec<Vec<usize>> = Vec::new();
    for idx in sorted_idx {
        let split = segments
            .last()
            .and_then(|seg| seg.last().copied())
            .is_some_and(|prev| {
                match (
                    expert_time_for_info(&infos[idx]),
                    expert_time_for_info(&infos[prev]),
                ) {
                    (Some(cur), Some(prev)) => cur - prev > HARD_BREAK_SECONDS,
                    _ => false,
                }
            });
        if split || segments.is_empty() {
            segments.push(vec![idx]);
        } else if let Some(seg) = segments.last_mut() {
            seg.push(idx);
        }
    }

    let mut groups = Vec::new();
    for segment in segments {
        groups.extend(expert_complete_linkage_segment(
            infos,
            segment,
            DISTANCE_THRESHOLD,
        ));
    }
    groups = expert_split_oversized(groups, infos, 25);
    for group in &mut groups {
        group.sort_by(|a, b| {
            expert_time_for_info(&infos[*a])
                .unwrap_or(0.0)
                .partial_cmp(&expert_time_for_info(&infos[*b]).unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    groups.sort_by(|a, b| {
        let ta = a
            .first()
            .and_then(|idx| expert_time_for_info(&infos[*idx]))
            .unwrap_or(0.0);
        let tb = b
            .first()
            .and_then(|idx| expert_time_for_info(&infos[*idx]))
            .unwrap_or(0.0);
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });
    groups
}

fn expert_complete_linkage_segment(
    infos: &[FastImageInfo],
    members: Vec<usize>,
    threshold: f64,
) -> Vec<Vec<usize>> {
    if members.is_empty() {
        return Vec::new();
    }
    let mut clusters = members
        .into_iter()
        .map(|idx| (idx, vec![idx]))
        .collect::<HashMap<usize, Vec<usize>>>();
    let mut cache: HashMap<(usize, usize), f64> = HashMap::new();
    loop {
        let ids = clusters.keys().copied().collect::<Vec<_>>();
        if ids.len() < 2 {
            break;
        }
        let mut best_pair = None;
        let mut best_d = f64::INFINITY;
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let d = expert_cluster_distance(
                    ids[i], ids[j], &clusters, infos, &mut cache, threshold,
                );
                if d < best_d {
                    best_d = d;
                    best_pair = Some((ids[i], ids[j]));
                }
            }
        }
        let Some((a, b)) = best_pair else {
            break;
        };
        if best_d > threshold {
            break;
        }
        let moved = clusters.remove(&b).unwrap_or_default();
        clusters.entry(a).or_default().extend(moved);
        cache.retain(|(x, y), _| *x != a && *y != a && *x != b && *y != b);
    }
    clusters.into_values().collect()
}

fn expert_cluster_distance(
    ca: usize,
    cb: usize,
    clusters: &HashMap<usize, Vec<usize>>,
    infos: &[FastImageInfo],
    cache: &mut HashMap<(usize, usize), f64>,
    threshold: f64,
) -> f64 {
    let key = (ca.min(cb), ca.max(cb));
    if let Some(value) = cache.get(&key) {
        return *value;
    }
    let mut max_d = 0.0;
    let Some(a_members) = clusters.get(&ca) else {
        return 1.0;
    };
    let Some(b_members) = clusters.get(&cb) else {
        return 1.0;
    };
    for i in a_members {
        for j in b_members {
            let d = 1.0 - expert_pair_similarity(&infos[*i], &infos[*j]);
            if d > max_d {
                max_d = d;
                if max_d > threshold {
                    cache.insert(key, max_d);
                    return max_d;
                }
            }
        }
    }
    cache.insert(key, max_d);
    max_d
}

fn expert_split_oversized(
    groups: Vec<Vec<usize>>,
    infos: &[FastImageInfo],
    max_size: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut stack = groups;
    while let Some(group) = stack.pop() {
        if group.len() <= max_size {
            out.push(group);
            continue;
        }
        let mut timed = group
            .into_iter()
            .map(|idx| (expert_time_for_info(&infos[idx]).unwrap_or(0.0), idx))
            .collect::<Vec<_>>();
        timed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut best_gap = -1.0;
        let mut best_k = timed.len() / 2;
        for k in 1..timed.len() {
            let gap = timed[k].0 - timed[k - 1].0;
            if gap > best_gap {
                best_gap = gap;
                best_k = k;
            }
        }
        stack.push(timed[..best_k].iter().map(|(_, idx)| *idx).collect());
        stack.push(timed[best_k..].iter().map(|(_, idx)| *idx).collect());
    }
    out
}

fn expert_time_for_info(info: &FastImageInfo) -> Option<f64> {
    info.timestamp.or(info.mtime)
}

fn expert_pair_similarity(a: &FastImageInfo, b: &FastImageInfo) -> f64 {
    const W_DINOV2: f64 = 0.35;
    const W_TIME: f64 = 0.20;
    const W_EXIF: f64 = 0.10;
    const W_FACE: f64 = 0.15;
    const W_GPS: f64 = 0.10;
    const W_FILENAME: f64 = 0.10;
    const PORTRAIT_W_DINOV2: f64 = 0.20;
    const PORTRAIT_W_TIME: f64 = 0.15;
    const PORTRAIT_W_EXIF: f64 = 0.05;
    const PORTRAIT_W_FACE: f64 = 0.50;
    const PORTRAIT_W_GPS: f64 = 0.05;
    const PORTRAIT_W_FILENAME: f64 = 0.05;

    let dino = cosine_similarity(a.dinov2.as_deref(), b.dinov2.as_deref())
        .expect("Expert/Tycoon clustering requires DINOv2 embeddings");
    let time = time_similarity(a.mtime, b.mtime);
    let exif = expert_exif_similarity(a.exif_summary.as_ref(), b.exif_summary.as_ref());
    let (gps, has_gps) = expert_gps_similarity(a.exif_summary.as_ref(), b.exif_summary.as_ref())
        .map_or((0.0, false), |v| (v, true));
    let face = face_overlap_similarity(&a.face_embeddings, &b.face_embeddings);
    let name = expert_filename_similarity(&a.path, &b.path);
    let portrait = !a.face_embeddings.is_empty() && !b.face_embeddings.is_empty();
    let (mut w_dino, w_time, mut w_exif, mut w_face, mut w_gps, w_name) = if portrait {
        (
            PORTRAIT_W_DINOV2,
            PORTRAIT_W_TIME,
            PORTRAIT_W_EXIF,
            PORTRAIT_W_FACE,
            PORTRAIT_W_GPS,
            PORTRAIT_W_FILENAME,
        )
    } else {
        (W_DINOV2, W_TIME, W_EXIF, W_FACE, W_GPS, W_FILENAME)
    };
    if !has_gps {
        w_exif += w_gps;
        w_gps = 0.0;
    }
    if !portrait && face == 0.0 && (a.face_embeddings.is_empty() || b.face_embeddings.is_empty()) {
        w_dino += w_face * 0.4;
        let w_time_adj = w_time + w_face * 0.3;
        w_exif += w_face * 0.3;
        w_face = 0.0;
        let _ = w_face;
        return w_dino * dino + w_time_adj * time + w_exif * exif + w_gps * gps + w_name * name;
    }
    w_dino * dino + w_time * time + w_exif * exif + w_face * face + w_gps * gps + w_name * name
}

fn cosine_similarity(a: Option<&[f32]>, b: Option<&[f32]>) -> Option<f64> {
    let (Some(a), Some(b)) = (a, b) else {
        return None;
    };
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na < 1e-12 || nb < 1e-12 {
        return None;
    }
    Some((dot / (na.sqrt() * nb.sqrt())).clamp(0.0, 1.0))
}

fn face_overlap_similarity(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    let n1 = a.len();
    let n2 = b.len();
    if n1 == 0 && n2 == 0 {
        1.0
    } else if n1 == 0 || n2 == 0 {
        0.0
    } else {
        let mut matched_a = HashSet::new();
        let mut matched_b = HashSet::new();
        for (i, emb_a) in a.iter().enumerate() {
            if matched_a.contains(&i) {
                continue;
            }
            let mut best_j = None;
            let mut best_sim = -1.0;
            for (j, emb_b) in b.iter().enumerate() {
                if matched_b.contains(&j) {
                    continue;
                }
                let sim = cosine_similarity(Some(emb_a), Some(emb_b)).unwrap_or(0.0);
                if sim > best_sim {
                    best_sim = sim;
                    best_j = Some(j);
                }
            }
            if let Some(j) = best_j {
                if best_sim > 0.45 {
                    matched_a.insert(i);
                    matched_b.insert(j);
                }
            }
        }
        let matched = matched_a.len();
        let denom = n1 + n2 - matched;
        if denom == 0 {
            0.0
        } else {
            matched as f64 / denom as f64
        }
    }
}

fn time_similarity(a: Option<f64>, b: Option<f64>) -> f64 {
    match (a, b) {
        (Some(x), Some(y)) => (-(x - y).abs() / 60.0).exp(),
        _ => 0.0,
    }
}

fn expert_exif_similarity(a: Option<&ExifSummary>, b: Option<&ExifSummary>) -> f64 {
    let (Some(a), Some(b)) = (a, b) else {
        return 0.0;
    };
    let mut score = 0.0;
    let mut parts = 0.0;
    if let (Some(x), Some(y)) = (&a.camera, &b.camera) {
        parts += 1.0;
        if x == y {
            score += 1.0;
        }
    }
    if let (Some(x), Some(y)) = (&a.lens, &b.lens) {
        parts += 1.0;
        if x == y {
            score += 1.0;
        }
    }
    if parts > 0.0 {
        score / parts
    } else {
        0.0
    }
}

fn expert_gps_similarity(a: Option<&ExifSummary>, b: Option<&ExifSummary>) -> Option<f64> {
    let a = a?;
    let b = b?;
    let (lat1, lon1, lat2, lon2) = (a.gps_lat?, a.gps_lon?, b.gps_lat?, b.gps_lon?);
    let avg_lat_rad = ((lat1 + lat2) / 2.0).to_radians();
    let dist = ((lat1 - lat2).powi(2) + ((lon1 - lon2) * avg_lat_rad.cos()).powi(2)).sqrt();
    Some((-dist / 0.0009).exp())
}

fn expert_filename_similarity(a: &str, b: &str) -> f64 {
    let name_a = file_name(a);
    let name_b = file_name(b);
    let (Some(n1), Some(n2)) = (
        filename_number_from_name(&name_a),
        filename_number_from_name(&name_b),
    ) else {
        return 0.0;
    };
    if filename_prefix_from_name(&name_a) != filename_prefix_from_name(&name_b) {
        return 0.0;
    }
    let delta = n1.abs_diff(n2) as f64;
    if delta == 0.0 {
        1.0
    } else {
        (1.0 - delta / 30.0).max(0.0)
    }
}

fn filename_number_from_name(name: &str) -> Option<u64> {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name);
    let digits = stem
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.chars().rev().collect::<String>().parse().ok()
    }
}

fn filename_prefix_from_name(name: &str) -> String {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name);
    stem.trim_end_matches(|ch: char| ch.is_ascii_digit())
        .to_string()
}

fn whash_image_scale(width: u32, height: u32) -> u32 {
    let natural = previous_power_of_two(width.min(height));
    natural.max(8)
}

fn previous_power_of_two(value: u32) -> u32 {
    if value <= 1 {
        return 1;
    }
    1 << (31 - value.leading_zeros())
}

fn quality_signals(img: &DynamicImage, file_size: u64) -> FastQualitySignals {
    let rgb = img.to_rgb8();
    let gray = pillow_luma(&rgb);
    let work = expert_vision::pillow_thumbnail_luma(&gray, 768, 768);
    let arr = GrayMatrix::from_luma(&work);
    let (width, height) = img.dimensions();

    let brightness_mean = arr.mean();
    let brightness_std = arr.std();
    let contrast_score = brightness_std;
    let underexposed_ratio = arr.ratio_le(8.0);
    let overexposed_ratio = arr.ratio_ge(247.0);
    let entropy = matrix_entropy(&arr);
    let lap = laplacian_variance(&arr).max(laplacian_variance(&arr.center_crop(0.6)));
    let tenengrad = tenengrad(&arr);
    let (high_ratio, motion_anisotropy) = fft_high_freq_ratio(&arr);
    let edge_width = edge_width_marziliano(&arr);

    let lap_norm = (lap.max(0.0).ln_1p() / 900.0f64.ln_1p()).min(1.0);
    let tenengrad_norm = (tenengrad.max(0.0).ln_1p() / 2000.0f64.ln_1p()).min(1.0);
    let high_norm = (high_ratio.max(0.0) / 0.40).min(1.0);
    let mut parts = vec![lap_norm, tenengrad_norm, high_norm];
    if let Some(width) = edge_width {
        parts.push(((10.0 - width) / 7.0).clamp(0.0, 1.0));
    }
    let blur_combined = parts.iter().sum::<f64>() / parts.len().max(1) as f64;

    let saliency = saliency_map(&arr);
    let salient_sharpness = saliency
        .as_ref()
        .and_then(|smap| salient_region_sharpness(&arr, smap));
    let focus_ratio = saliency
        .as_ref()
        .and_then(|smap| saliency_focus_consistency(&arr, smap));
    let composition = saliency.as_ref().map(|smap| composition_score(&arr, smap));
    let exposure = nine_grid_exposure(&arr);
    let horizon_tilt_deg = horizon_tilt_degrees(&arr);

    FastQualitySignals {
        width,
        height,
        file_size,
        blur_score: lap,
        brightness_mean,
        brightness_std,
        contrast_score,
        overexposed_ratio,
        underexposed_ratio,
        entropy,
        blur_combined,
        salient_sharpness,
        motion_anisotropy,
        edge_width_pix: edge_width,
        focus_ratio,
        horizon_tilt_deg,
        composition,
        worst_clip_dark: exposure.worst_clip_dark,
        worst_clip_bright: exposure.worst_clip_bright,
    }
}

#[derive(Clone)]
struct GrayMatrix {
    width: usize,
    height: usize,
    data: Vec<f64>,
}

impl GrayMatrix {
    fn new(width: usize, height: usize, data: Vec<f64>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    fn from_luma(img: &image::GrayImage) -> Self {
        let (w, h) = img.dimensions();
        let data = img.as_raw().iter().map(|v| f64::from(*v)).collect();
        Self::new(w as usize, h as usize, data)
    }

    fn zeros(width: usize, height: usize) -> Self {
        Self::new(width, height, vec![0.0; width.saturating_mul(height)])
    }

    fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    fn set(&mut self, x: usize, y: usize, value: f64) {
        self.data[y * self.width + x] = value;
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / self.len().max(1) as f64
    }

    fn std(&self) -> f64 {
        let mean = self.mean();
        (self.data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / self.len().max(1) as f64)
            .sqrt()
    }

    fn ratio_le(&self, threshold: f64) -> f64 {
        self.data.iter().filter(|v| **v <= threshold).count() as f64 / self.len().max(1) as f64
    }

    fn ratio_ge(&self, threshold: f64) -> f64 {
        self.data.iter().filter(|v| **v >= threshold).count() as f64 / self.len().max(1) as f64
    }

    fn center_crop(&self, ratio: f64) -> Self {
        let crop_h = ((self.height as f64 * ratio) as usize).max(1);
        let crop_w = ((self.width as f64 * ratio) as usize).max(1);
        let y0 = (self.height.saturating_sub(crop_h)) / 2;
        let x0 = (self.width.saturating_sub(crop_w)) / 2;
        let mut data = Vec::with_capacity(crop_w * crop_h);
        for y in y0..(y0 + crop_h).min(self.height) {
            for x in x0..(x0 + crop_w).min(self.width) {
                data.push(self.get(x, y));
            }
        }
        Self::new(crop_w, crop_h, data)
    }
}

#[derive(Debug, Clone, Copy)]
struct NineGridExposure {
    worst_clip_dark: f64,
    worst_clip_bright: f64,
}

fn matrix_entropy(arr: &GrayMatrix) -> f64 {
    let mut hist = [0usize; 256];
    for value in &arr.data {
        let idx = value.round().clamp(0.0, 255.0) as usize;
        hist[idx] += 1;
    }
    let total = hist.iter().sum::<usize>();
    if total == 0 {
        return 0.0;
    }
    hist.iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let p = *count as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

fn laplacian_variance(arr: &GrayMatrix) -> f64 {
    if arr.height < 3 || arr.width < 3 {
        return 0.0;
    }
    let mut values = Vec::with_capacity((arr.width - 2) * (arr.height - 2));
    for y in 1..arr.height - 1 {
        for x in 1..arr.width - 1 {
            let lap = arr.get(x, y) * 4.0
                - arr.get(x, y - 1)
                - arr.get(x, y + 1)
                - arr.get(x - 1, y)
                - arr.get(x + 1, y);
            values.push(lap);
        }
    }
    variance(&values)
}

fn tenengrad(arr: &GrayMatrix) -> f64 {
    if arr.height < 3 || arr.width < 3 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for y in 1..arr.height - 1 {
        for x in 1..arr.width - 1 {
            let gx = arr.get(x + 1, y) - arr.get(x - 1, y);
            let gy = arr.get(x, y + 1) - arr.get(x, y - 1);
            total += gx * gx + gy * gy;
            count += 1;
        }
    }
    total / count.max(1) as f64
}

fn fft_high_freq_ratio(arr: &GrayMatrix) -> (f64, f64) {
    use rustfft::{num_complex::Complex, FftPlanner};

    if arr.height < 16 || arr.width < 16 {
        return (0.0, 0.0);
    }
    let side = arr.height.min(arr.width);
    let y0 = (arr.height - side) / 2;
    let x0 = (arr.width - side) / 2;
    let mut crop = GrayMatrix::zeros(side, side);
    for y in 0..side {
        for x in 0..side {
            crop.set(x, y, arr.get(x0 + x, y0 + y));
        }
    }
    if side > 256 {
        crop = resize_area_matrix(&crop, 256, 256);
    }
    let side = crop.width;
    let mean = crop.mean();
    let hann = hann_window(side);
    let mut spec = vec![Complex::new(0.0, 0.0); side * side];
    for y in 0..side {
        for x in 0..side {
            spec[y * side + x] = Complex::new((crop.get(x, y) - mean) * hann[y] * hann[x], 0.0);
        }
    }
    fft2_in_place(&mut spec, side, side, false, &mut FftPlanner::new());

    let mut mag = vec![0.0; side * side];
    for y in 0..side {
        for x in 0..side {
            let src_y = (y + side / 2) % side;
            let src_x = (x + side / 2) % side;
            mag[y * side + x] = spec[src_y * side + src_x].norm();
        }
    }
    mag[(side / 2) * side + side / 2] = 0.0;

    let center = side as f64 / 2.0;
    let r_max = side as f64 / 2.0;
    let mut total = 1e-8;
    let mut high = 0.0;
    let mut sums = [0.0f64; 12];
    let mut counts = [1e-6f64; 12];
    let mut band_count = 0usize;
    for y in 0..side {
        for x in 0..side {
            let dy = y as f64 - center;
            let dx = x as f64 - center;
            let r = (dy * dy + dx * dx).sqrt();
            let value = mag[y * side + x];
            total += value;
            if r > 0.30 * r_max {
                high += value;
            }
            if r > 0.10 * r_max && r < 0.50 * r_max {
                let mut theta = dy.atan2(dx);
                if theta < 0.0 {
                    theta += std::f64::consts::PI;
                }
                let bin = ((theta / std::f64::consts::PI * 12.0) as usize).min(11);
                sums[bin] += value;
                counts[bin] += 1.0;
                band_count += 1;
            }
        }
    }
    let high_ratio = high / total;
    if band_count < 50 {
        return (high_ratio, 0.0);
    }
    let avgs = sums
        .iter()
        .zip(counts.iter())
        .map(|(sum, count)| sum / count)
        .collect::<Vec<_>>();
    let max_avg = avgs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_avg = avgs.iter().copied().fold(f64::INFINITY, f64::min);
    let aniso = (max_avg - min_avg) / (max_avg + 1e-8);
    (high_ratio, aniso)
}

fn nine_grid_exposure(arr: &GrayMatrix) -> NineGridExposure {
    if arr.height < 9 || arr.width < 9 {
        return NineGridExposure {
            worst_clip_dark: 0.0,
            worst_clip_bright: 0.0,
        };
    }
    let ys = [0, arr.height / 3, 2 * arr.height / 3, arr.height];
    let xs = [0, arr.width / 3, 2 * arr.width / 3, arr.width];
    let mut worst_clip_dark = 0.0f64;
    let mut worst_clip_bright = 0.0f64;
    for gy in 0..3 {
        for gx in 0..3 {
            let mut dark = 0usize;
            let mut bright = 0usize;
            let mut count = 0usize;
            for y in ys[gy]..ys[gy + 1] {
                for x in xs[gx]..xs[gx + 1] {
                    let value = arr.get(x, y);
                    if value <= 8.0 {
                        dark += 1;
                    }
                    if value >= 247.0 {
                        bright += 1;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                worst_clip_dark = worst_clip_dark.max(dark as f64 / count as f64);
                worst_clip_bright = worst_clip_bright.max(bright as f64 / count as f64);
            }
        }
    }
    NineGridExposure {
        worst_clip_dark,
        worst_clip_bright,
    }
}

#[cfg(feature = "opencv-orb")]
fn edge_width_marziliano(arr: &GrayMatrix) -> Option<f64> {
    use opencv::{core, imgproc, prelude::*};

    if arr.height < 16 || arr.width < 16 {
        return None;
    }
    let bytes = arr
        .data
        .iter()
        .map(|v| v.round().clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    let src = core::Mat::from_slice(&bytes)
        .ok()?
        .reshape(1, arr.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut edges = core::Mat::default();
    imgproc::canny(&src, &mut edges, 50.0, 150.0, 3, false).ok()?;
    let edge_bytes = edges.data_typed::<u8>().ok()?;
    let points = edge_bytes
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            if *value > 0 {
                Some((idx / arr.width, idx % arr.width))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if points.len() < 50 {
        return None;
    }
    let mut widths = Vec::new();
    for (y, x) in points {
        if x < 3 || x > arr.width.saturating_sub(4) {
            continue;
        }
        let mut left = x;
        for k in 1..12 {
            if x < k + 1 {
                break;
            }
            if arr.get(x - k, y) >= arr.get(x - k + 1, y) {
                left = x - k;
            } else {
                break;
            }
        }
        let mut right = x;
        for k in 1..12 {
            if x + k > arr.width.saturating_sub(2) {
                break;
            }
            if arr.get(x + k, y) <= arr.get(x + k - 1, y) {
                right = x + k;
            } else {
                break;
            }
        }
        let width = right.saturating_sub(left);
        if (1..=25).contains(&width) {
            widths.push(width as f64);
        }
    }
    if widths.len() < 30 {
        None
    } else {
        Some(widths.iter().sum::<f64>() / widths.len() as f64)
    }
}

#[cfg(not(feature = "opencv-orb"))]
fn edge_width_marziliano(_arr: &GrayMatrix) -> Option<f64> {
    None
}

#[cfg(feature = "opencv-orb")]
fn horizon_tilt_degrees(arr: &GrayMatrix) -> Option<f64> {
    use opencv::{core, imgproc, prelude::*};

    if arr.height < 64 || arr.width < 64 {
        return None;
    }
    let bytes = arr
        .data
        .iter()
        .map(|v| v.round().clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    let src = core::Mat::from_slice(&bytes)
        .ok()?
        .reshape(1, arr.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut edges = core::Mat::default();
    imgproc::canny(&src, &mut edges, 50.0, 150.0, 3, false).ok()?;
    let min_len = 40.max((arr.height.min(arr.width) as f64 * 0.35) as i32);
    let mut lines = core::Vector::<core::Vec4i>::new();
    imgproc::hough_lines_p(
        &edges,
        &mut lines,
        1.0,
        std::f64::consts::PI / 180.0,
        80,
        min_len as f64,
        10.0,
    )
    .ok()?;
    if lines.is_empty() {
        return None;
    }
    let mut pairs = Vec::<(f64, f64)>::new();
    for line in lines.iter().take(200) {
        let dx = f64::from(line[2] - line[0]);
        let dy = f64::from(line[3] - line[1]);
        let length = dx.hypot(dy);
        if length < f64::from(min_len) {
            continue;
        }
        let mut theta = dy.atan2(dx).to_degrees();
        if theta > 90.0 {
            theta -= 180.0;
        } else if theta < -90.0 {
            theta += 180.0;
        }
        let dev = theta.abs().min((90.0 - theta.abs()).abs());
        pairs.push((dev, length));
    }
    if pairs.is_empty() {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_take = (pairs.len() / 3).max(1);
    let (weighted, total) = pairs
        .iter()
        .take(n_take)
        .fold((0.0, 0.0), |(s, w), (angle, weight)| {
            (s + angle * weight, w + weight)
        });
    Some(weighted / total.max(1e-8))
}

#[cfg(not(feature = "opencv-orb"))]
fn horizon_tilt_degrees(_arr: &GrayMatrix) -> Option<f64> {
    None
}

fn saliency_map(arr: &GrayMatrix) -> Option<GrayMatrix> {
    use rustfft::{num_complex::Complex, FftPlanner};

    if arr.height < 8 || arr.width < 8 {
        return None;
    }
    let small = resize_area_matrix(arr, 64, 64);
    let mut spectrum = small
        .data
        .iter()
        .map(|v| Complex::new(*v, 0.0))
        .collect::<Vec<_>>();
    fft2_in_place(&mut spectrum, 64, 64, false, &mut FftPlanner::new());
    let log_amp = GrayMatrix::new(
        64,
        64,
        spectrum
            .iter()
            .map(|value| (value.norm() + 1e-8).ln())
            .collect(),
    );
    let smooth =
        opencv_box_filter_3x3(&log_amp).unwrap_or_else(|| box_filter_3x3_reflect101(&log_amp));
    let mut recon = Vec::with_capacity(spectrum.len());
    for (idx, value) in spectrum.iter().enumerate() {
        let residual = log_amp.data[idx] - smooth.data[idx];
        recon.push(Complex::from_polar(residual.exp(), value.arg()));
    }
    fft2_in_place(&mut recon, 64, 64, true, &mut FftPlanner::new());
    let norm = (64 * 64) as f64;
    let squared = GrayMatrix::new(
        64,
        64,
        recon
            .iter()
            .map(|value| (value / norm).norm_sqr())
            .collect(),
    );
    let blurred =
        opencv_gaussian_blur(&squared, 9, 2.5).unwrap_or_else(|| gaussian_blur(&squared, 9, 2.5));
    let mut out = opencv_resize_linear_matrix(&blurred, arr.width, arr.height)
        .unwrap_or_else(|| resize_bilinear_matrix(&blurred, arr.width, arr.height));
    let min_v = out.data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = out.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_v - min_v < 1e-8 {
        return None;
    }
    for value in &mut out.data {
        *value = (*value - min_v) / (max_v - min_v);
    }
    if out.std() < 0.01 {
        None
    } else {
        Some(out)
    }
}

fn salient_region_sharpness(arr: &GrayMatrix, smap: &GrayMatrix) -> Option<f64> {
    if arr.height < 16 || arr.width < 16 {
        return None;
    }
    if smap.std() < 0.01 {
        return None;
    }
    let smap = if smap.width == arr.width && smap.height == arr.height {
        smap.clone()
    } else {
        resize_bilinear_matrix(smap, arr.width, arr.height)
    };
    let threshold = quantile(&smap.data, 0.80)?;
    let mut selected = Vec::new();
    for y in 1..arr.height - 1 {
        for x in 1..arr.width - 1 {
            if smap.get(x, y) >= threshold {
                let lap = arr.get(x, y) * 4.0
                    - arr.get(x, y - 1)
                    - arr.get(x, y + 1)
                    - arr.get(x - 1, y)
                    - arr.get(x + 1, y);
                selected.push(lap);
            }
        }
    }
    if selected.len() < 100 {
        None
    } else {
        Some(variance(&selected))
    }
}

fn saliency_focus_consistency(arr: &GrayMatrix, smap: &GrayMatrix) -> Option<f64> {
    if arr.height < 32 || arr.width < 32 {
        return None;
    }
    let sub_thr = quantile(&smap.data, 0.80)?;
    let bg_thr = quantile(&smap.data, 0.30)?;
    let mut sub_lap = Vec::new();
    let mut bg_lap = Vec::new();
    for y in 1..arr.height - 1 {
        for x in 1..arr.width - 1 {
            let sal = smap.get(x, y);
            let lap = arr.get(x, y) * 4.0
                - arr.get(x, y - 1)
                - arr.get(x, y + 1)
                - arr.get(x - 1, y)
                - arr.get(x + 1, y);
            if sal >= sub_thr {
                sub_lap.push(lap);
            }
            if sal <= bg_thr {
                bg_lap.push(lap);
            }
        }
    }
    if sub_lap.len() < 100 || bg_lap.len() < 100 {
        None
    } else {
        Some(variance(&sub_lap) / (variance(&bg_lap) + 1e-6))
    }
}

fn composition_score(arr: &GrayMatrix, smap: &GrayMatrix) -> f64 {
    let total = smap.data.iter().sum::<f64>() + 1e-8;
    if total < 1e-3 {
        return 0.4;
    }
    let mut cx_sum = 0.0;
    let mut cy_sum = 0.0;
    for y in 0..smap.height {
        for x in 0..smap.width {
            let value = smap.get(x, y);
            cx_sum += x as f64 * value;
            cy_sum += y as f64 * value;
        }
    }
    let cy = cy_sum / total / (arr.height.saturating_sub(1).max(1) as f64);
    let cx = cx_sum / total / (arr.width.saturating_sub(1).max(1) as f64);
    let grid_pts = [
        (1.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
    ];
    let d_grid = grid_pts
        .iter()
        .map(|(py, px)| (cy - py).hypot(cx - px))
        .fold(f64::INFINITY, f64::min);
    let d_center = (cy - 0.5).hypot(cx - 0.5);
    let pos_score = (1.0 - d_grid.min(d_center) / 0.35).max(0.0);

    let threshold = quantile(&smap.data, 0.80).unwrap_or(0.0);
    let mut subject = 0usize;
    let mut edge_subject = 0usize;
    let edge_h = (arr.height as f64 * 0.05) as usize;
    let edge_w = (arr.width as f64 * 0.05) as usize;
    let edge_h = edge_h.max(2);
    let edge_w = edge_w.max(2);
    for y in 0..smap.height {
        for x in 0..smap.width {
            if smap.get(x, y) >= threshold {
                subject += 1;
                if y < edge_h
                    || y >= smap.height.saturating_sub(edge_h)
                    || x < edge_w
                    || x >= smap.width.saturating_sub(edge_w)
                {
                    edge_subject += 1;
                }
            }
        }
    }
    let frac = subject as f64 / smap.len().max(1) as f64;
    let size_score = if frac < 0.04 {
        frac / 0.04
    } else if frac > 0.55 {
        (1.0 - (frac - 0.55) / 0.45).max(0.0)
    } else {
        1.0
    };
    let edge_frac = edge_subject as f64 / (subject as f64 + 1e-6);
    let edge_score = if edge_frac < 0.25 {
        1.0
    } else {
        (1.0 - (edge_frac - 0.25) / 0.5).max(0.0)
    };
    0.5 * pos_score + 0.3 * size_score + 0.2 * edge_score
}

fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = (sorted.len() - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Some(sorted[lo])
    } else {
        let weight = pos - lo as f64;
        Some(sorted[lo] * (1.0 - weight) + sorted[hi] * weight)
    }
}

fn hann_window(size: usize) -> Vec<f64> {
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())
        .collect()
}

fn fft2_in_place(
    data: &mut [rustfft::num_complex::Complex<f64>],
    width: usize,
    height: usize,
    inverse: bool,
    planner: &mut rustfft::FftPlanner<f64>,
) {
    let row_fft = if inverse {
        planner.plan_fft_inverse(width)
    } else {
        planner.plan_fft_forward(width)
    };
    for y in 0..height {
        row_fft.process(&mut data[y * width..(y + 1) * width]);
    }
    let col_fft = if inverse {
        planner.plan_fft_inverse(height)
    } else {
        planner.plan_fft_forward(height)
    };
    let mut column = vec![rustfft::num_complex::Complex::new(0.0, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            column[y] = data[y * width + x];
        }
        col_fft.process(&mut column);
        for y in 0..height {
            data[y * width + x] = column[y];
        }
    }
}

fn box_filter_3x3_reflect101(src: &GrayMatrix) -> GrayMatrix {
    let mut out = GrayMatrix::zeros(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let mut sum = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let sx = reflect101(x as isize + dx, src.width);
                    let sy = reflect101(y as isize + dy, src.height);
                    sum += src.get(sx, sy);
                }
            }
            out.set(x, y, sum / 9.0);
        }
    }
    out
}

#[cfg(feature = "opencv-orb")]
fn opencv_box_filter_3x3(src: &GrayMatrix) -> Option<GrayMatrix> {
    use opencv::{core, imgproc, prelude::*};

    let input = src.data.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let mat = core::Mat::from_slice(&input)
        .ok()?
        .reshape(1, src.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let kernel_data = vec![1.0f32 / 9.0; 9];
    let kernel = core::Mat::from_slice(&kernel_data)
        .ok()?
        .reshape(1, 3)
        .ok()?
        .try_clone()
        .ok()?;
    let mut dst = core::Mat::default();
    imgproc::filter_2d(
        &mat,
        &mut dst,
        -1,
        &kernel,
        core::Point::new(-1, -1),
        0.0,
        core::BORDER_DEFAULT,
    )
    .ok()?;
    let data = dst.data_typed::<f32>().ok()?;
    Some(GrayMatrix::new(
        src.width,
        src.height,
        data.iter().map(|v| f64::from(*v)).collect(),
    ))
}

#[cfg(not(feature = "opencv-orb"))]
fn opencv_box_filter_3x3(_src: &GrayMatrix) -> Option<GrayMatrix> {
    None
}

fn gaussian_blur(src: &GrayMatrix, kernel_size: usize, sigma: f64) -> GrayMatrix {
    let radius = kernel_size / 2;
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as isize - radius as isize;
        let value = (-(x * x) as f64 / (2.0 * sigma * sigma)).exp();
        kernel.push(value);
        sum += value;
    }
    for value in &mut kernel {
        *value /= sum;
    }
    let mut tmp = GrayMatrix::zeros(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let mut value = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let sx = reflect101(x as isize + k as isize - radius as isize, src.width);
                value += src.get(sx, y) * weight;
            }
            tmp.set(x, y, value);
        }
    }
    let mut out = GrayMatrix::zeros(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let mut value = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let sy = reflect101(y as isize + k as isize - radius as isize, src.height);
                value += tmp.get(x, sy) * weight;
            }
            out.set(x, y, value);
        }
    }
    out
}

#[cfg(feature = "opencv-orb")]
fn opencv_gaussian_blur(src: &GrayMatrix, kernel_size: usize, sigma: f64) -> Option<GrayMatrix> {
    use opencv::{core, imgproc, prelude::*};

    let input = src.data.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let mat = core::Mat::from_slice(&input)
        .ok()?
        .reshape(1, src.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut dst = core::Mat::default();
    imgproc::gaussian_blur(
        &mat,
        &mut dst,
        core::Size::new(kernel_size as i32, kernel_size as i32),
        sigma,
        sigma,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .ok()?;
    let data = dst.data_typed::<f32>().ok()?;
    Some(GrayMatrix::new(
        src.width,
        src.height,
        data.iter().map(|v| f64::from(*v)).collect(),
    ))
}

#[cfg(not(feature = "opencv-orb"))]
fn opencv_gaussian_blur(_src: &GrayMatrix, _kernel_size: usize, _sigma: f64) -> Option<GrayMatrix> {
    None
}

fn reflect101(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let mut idx = idx;
    let len_i = len as isize;
    while idx < 0 || idx >= len_i {
        if idx < 0 {
            idx = -idx;
        } else {
            idx = 2 * len_i - idx - 2;
        }
    }
    idx as usize
}

#[cfg(feature = "opencv-orb")]
fn resize_area_matrix(src: &GrayMatrix, width: usize, height: usize) -> GrayMatrix {
    opencv_resize_matrix(src, width, height, opencv::imgproc::INTER_AREA)
        .unwrap_or_else(|| resize_bilinear_matrix(src, width, height))
}

#[cfg(not(feature = "opencv-orb"))]
fn resize_area_matrix(src: &GrayMatrix, width: usize, height: usize) -> GrayMatrix {
    resize_bilinear_matrix(src, width, height)
}

#[cfg(feature = "opencv-orb")]
fn opencv_resize_linear_matrix(
    src: &GrayMatrix,
    width: usize,
    height: usize,
) -> Option<GrayMatrix> {
    opencv_resize_matrix(src, width, height, opencv::imgproc::INTER_LINEAR)
}

#[cfg(not(feature = "opencv-orb"))]
fn opencv_resize_linear_matrix(
    _src: &GrayMatrix,
    _width: usize,
    _height: usize,
) -> Option<GrayMatrix> {
    None
}

#[cfg(feature = "opencv-orb")]
fn opencv_resize_matrix(
    src: &GrayMatrix,
    width: usize,
    height: usize,
    interpolation: i32,
) -> Option<GrayMatrix> {
    use opencv::{core, imgproc, prelude::*};

    let input = src.data.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let mat = core::Mat::from_slice(&input)
        .ok()?
        .reshape(1, src.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut dst = core::Mat::default();
    imgproc::resize(
        &mat,
        &mut dst,
        core::Size::new(width as i32, height as i32),
        0.0,
        0.0,
        interpolation,
    )
    .ok()?;
    let data = dst.data_typed::<f32>().ok()?;
    Some(GrayMatrix::new(
        width,
        height,
        data.iter().map(|v| f64::from(*v)).collect(),
    ))
}

fn resize_bilinear_matrix(src: &GrayMatrix, width: usize, height: usize) -> GrayMatrix {
    if width == 0 || height == 0 || src.width == 0 || src.height == 0 {
        return GrayMatrix::zeros(width, height);
    }
    if src.width == width && src.height == height {
        return src.clone();
    }
    let mut out = GrayMatrix::zeros(width, height);
    let scale_x = src.width as f64 / width as f64;
    let scale_y = src.height as f64 / height as f64;
    for y in 0..height {
        let fy = (y as f64 + 0.5) * scale_y - 0.5;
        let y0 = fy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(src.height - 1);
        let wy = fy - y0 as f64;
        for x in 0..width {
            let fx = (x as f64 + 0.5) * scale_x - 0.5;
            let x0 = fx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(src.width - 1);
            let wx = fx - x0 as f64;
            let top = src.get(x0, y0) * (1.0 - wx) + src.get(x1, y0) * wx;
            let bottom = src.get(x0, y1) * (1.0 - wx) + src.get(x1, y1) * wx;
            out.set(x, y, top * (1.0 - wy) + bottom * wy);
        }
    }
    out
}

fn compute_color_hist(img: &DynamicImage) -> Option<Vec<f32>> {
    #[cfg(feature = "opencv-orb")]
    {
        if let Some(hist) = compute_color_hist_opencv(img) {
            return Some(hist);
        }
    }
    let rgb = img.resize(384, 384, FilterType::Triangle).to_rgb8();
    let (w, h) = rgb.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    let mut feats = Vec::with_capacity(144);
    for gy in 0..3 {
        for gx in 0..3 {
            let x0 = gx * w / 3;
            let x1 = (gx + 1) * w / 3;
            let y0 = gy * h / 3;
            let y1 = (gy + 1) * h / 3;
            let mut hh = [0f32; 16];
            let mut ss = [0f32; 16];
            let mut vv = [0f32; 16];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = rgb.get_pixel(x, y);
                    let (hue, sat, val) = rgb_to_hsv(p[0], p[1], p[2]);
                    hh[((hue / 180.0 * 16.0).floor() as usize).min(15)] += 1.0;
                    ss[((sat * 16.0).floor() as usize).min(15)] += 1.0;
                    vv[((val * 16.0).floor() as usize).min(15)] += 1.0;
                }
            }
            normalize_sum(&mut hh);
            normalize_sum(&mut ss);
            normalize_sum(&mut vv);
            feats.extend(hh);
            feats.extend(ss);
            feats.extend(vv);
        }
    }
    let norm = feats.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return None;
    }
    for v in &mut feats {
        *v /= norm;
    }
    Some(feats)
}

#[cfg(feature = "opencv-orb")]
fn compute_color_hist_opencv(img: &DynamicImage) -> Option<Vec<f32>> {
    use opencv::{core, imgproc, prelude::*};

    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    let src = core::Mat::from_slice(rgb.as_raw())
        .ok()?
        .reshape(3, h as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut work = src;
    let mut work_w = w;
    let mut work_h = h;
    if w.max(h) > 384 {
        let scale = 384.0f64 / w.max(h) as f64;
        work_w = ((w as f64 * scale) as i32).max(1) as u32;
        work_h = ((h as f64 * scale) as i32).max(1) as u32;
        let mut resized = core::Mat::default();
        imgproc::resize(
            &work,
            &mut resized,
            core::Size::new(work_w as i32, work_h as i32),
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )
        .ok()?;
        work = resized;
    }
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(
        &work,
        &mut hsv,
        imgproc::COLOR_RGB2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .ok()?;
    let mut feats = Vec::with_capacity(144);
    for gy in 0..3 {
        for gx in 0..3 {
            let x0 = gx * work_w / 3;
            let x1 = (gx + 1) * work_w / 3;
            let y0 = gy * work_h / 3;
            let y1 = (gy + 1) * work_h / 3;
            let mut hh = [0f32; 16];
            let mut ss = [0f32; 16];
            let mut vv = [0f32; 16];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = *hsv.at_2d::<core::Vec3b>(y as i32, x as i32).ok()?;
                    hh[((p[0] as usize * 16) / 180).min(15)] += 1.0;
                    ss[((p[1] as usize * 16) / 256).min(15)] += 1.0;
                    vv[((p[2] as usize * 16) / 256).min(15)] += 1.0;
                }
            }
            normalize_sum(&mut hh);
            normalize_sum(&mut ss);
            normalize_sum(&mut vv);
            feats.extend(hh);
            feats.extend(ss);
            feats.extend(vv);
        }
    }
    let norm = feats.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return None;
    }
    for v in &mut feats {
        *v /= norm;
    }
    Some(feats)
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = f32::from(r) / 255.0;
    let g = f32::from(g) / 255.0;
    let b = f32::from(b) / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let hue = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let hue = if hue < 0.0 { hue + 360.0 } else { hue } / 2.0;
    let sat = if max == 0.0 { 0.0 } else { delta / max };
    (hue, sat, max)
}

fn normalize_sum(values: &mut [f32]) {
    let sum = values.iter().sum::<f32>();
    if sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

async fn reset_session(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let mut state = ctx.inner.lock().expect("backend state lock");
    state.session = None;
    state.last_infos.clear();
    state.grouping = GroupingState::default();
    state.job = JobState::default();
    Json(json!({"ok": true}))
}

async fn cancel_job(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let mut state = ctx.inner.lock().expect("backend state lock");
    state.job.cancel_requested = true;
    Json(json!({"ok": true}))
}

#[derive(Debug, Deserialize)]
struct SinceQuery {
    since: Option<u64>,
}

async fn job_status(State(ctx): State<AppCtx>, Query(q): Query<SinceQuery>) -> impl IntoResponse {
    let state = ctx.inner.lock().expect("backend state lock");
    let job = &state.job;
    let since = q.since.unwrap_or(0);
    let events = job
        .recent_events
        .iter()
        .filter(|ev| ev.seq > since)
        .cloned()
        .collect::<Vec<_>>();
    Json(json!({
        "folder": job.folder,
        "dry_run": job.dry_run,
        "mode": job.mode,
        "engine": job.engine,
        "status": job.status,
        "done": job.done,
        "total": job.total,
        "label": job.label,
        "error": job.error,
        "elapsed": if job.started_at > 0.0 { now_secs() - job.started_at } else { 0.0 },
        "skipped_count": job.skipped.len(),
        "skipped_sample": job.skipped.iter().take(5).collect::<Vec<_>>(),
        "events": events,
        "rejected_running": job.recent_events.iter().filter(|e| e.reject).count(),
    }))
}

async fn session_status(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let state = ctx.inner.lock().expect("backend state lock");
    Json(status_json(state.session.as_ref()))
}

fn status_json(session: Option<&SessionState>) -> Value {
    let Some(s) = session else {
        return json!({"ready": false});
    };
    let total_groups = s.groups.len();
    let multi_groups = s.groups.iter().filter(|g| g.images.len() > 1).count();
    let finished_groups = s.groups.iter().filter(|g| g.finished).count();
    let finished_multi_groups = s
        .groups
        .iter()
        .filter(|g| g.images.len() > 1 && g.finished)
        .count();
    let winner_count = s.groups.iter().filter(|g| g.winner.is_some()).count()
        + s.groups
            .iter()
            .map(|g| g.extra_winners.len())
            .sum::<usize>();
    let loser_count = s.groups.iter().map(|g| g.losers.len()).sum::<usize>();
    let restored = s.prescreen_restored.iter().collect::<HashSet<_>>();
    let pending_count = s
        .prescreen_rejected
        .iter()
        .filter(|p| !restored.contains(p))
        .count();
    json!({
        "ready": true,
        "folder": s.folder,
        "dry_run": s.dry_run,
        "mode": s.mode,
        "engine": s.engine,
        "current": s.current_group,
        "total_groups": total_groups,
        "multi_groups": multi_groups,
        "finished_groups": finished_groups,
        "finished_multi_groups": finished_multi_groups,
        "unfinished_groups": total_groups.saturating_sub(finished_groups),
        "image_count": s.groups.iter().map(|g| g.images.len()).sum::<usize>(),
        "winner_count": winner_count,
        "loser_count": loser_count,
        "threshold_near": s.threshold_near,
        "threshold_far": s.threshold_far,
        "near_seconds": s.near_seconds,
        "prescreen_enabled": s.prescreen_enabled,
        "prescreen_strength": s.prescreen_strength,
        "prescreen_reviewed": s.prescreen_reviewed,
        "prescreen_auto_rejected_count": s.prescreen_rejected.len(),
        "prescreen_pending_count": pending_count,
        "selection_started": s.selection_started,
        "preferences": {"decisions": 0},
    })
}

async fn current_group(State(ctx): State<AppCtx>) -> Response {
    let mut state = ctx.inner.lock().expect("backend state lock");
    let Some(session) = state.session.as_mut() else {
        return Json(json!({"done": true, "group": null})).into_response();
    };
    advance_current_index(session);
    if session.current_group >= session.groups.len() {
        return Json(json!({"done": true, "group": null})).into_response();
    }
    Json(json!({"done": false, "group": serialize_group(session, session.current_group)}))
        .into_response()
}

#[derive(Debug, Deserialize)]
struct ChooseRequest {
    loser: String,
}

async fn choose(State(ctx): State<AppCtx>, Json(req): Json<ChooseRequest>) -> Response {
    mutate_current_group(ctx, |session, idx| {
        let snapshot = session.groups[idx].clone();
        session.undo_stack.push(UndoEntry {
            group_index: idx,
            group: snapshot,
        });
        advance(&mut session.groups[idx], &req.loser);
    })
}

#[derive(Debug, Deserialize)]
struct KickRequest {
    side: String,
}

async fn kick(State(ctx): State<AppCtx>, Json(req): Json<KickRequest>) -> Response {
    mutate_current_group(ctx, |session, idx| {
        let snapshot = session.groups[idx].clone();
        session.undo_stack.push(UndoEntry {
            group_index: idx,
            group: snapshot,
        });
        kick_side(&mut session.groups[idx], &req.side);
    })
}

async fn undo(State(ctx): State<AppCtx>) -> Response {
    let mut state = ctx.inner.lock().expect("backend state lock");
    let Some(session) = state.session.as_mut() else {
        return json_error(StatusCode::BAD_REQUEST, "没有会话");
    };
    if let Some(entry) = session.undo_stack.pop() {
        if entry.group_index < session.groups.len() {
            let _ = revert_group_files(session, entry.group_index);
            session.groups[entry.group_index] = entry.group;
            session.current_group = entry.group_index;
        }
    }
    let _ = save_state(session);
    current_group_payload(session)
}

async fn skip_group(State(ctx): State<AppCtx>) -> Response {
    let mut state = ctx.inner.lock().expect("backend state lock");
    let Some(session) = state.session.as_mut() else {
        return json_error(StatusCode::BAD_REQUEST, "没有会话");
    };
    if session.current_group < session.groups.len() {
        let group = session.groups.remove(session.current_group);
        session.groups.push(group);
    }
    let _ = save_state(session);
    current_group_payload(session)
}

#[derive(Debug, Deserialize)]
struct ReopenRequest {
    group_id: String,
}

async fn reopen_group(State(ctx): State<AppCtx>, Json(req): Json<ReopenRequest>) -> Response {
    let mut state = ctx.inner.lock().expect("backend state lock");
    let Some(session) = state.session.as_mut() else {
        return json_error(StatusCode::BAD_REQUEST, "没有会话");
    };
    let mut failed = Vec::new();
    if let Some(idx) = session.groups.iter().position(|g| g.id == req.group_id) {
        failed = revert_group_files(session, idx);
        reset_group_for_reopen(&mut session.groups[idx]);
        session.current_group = idx;
    }
    let _ = save_state(session);
    Json(json!({"ok": true, "failed": failed})).into_response()
}

fn mutate_current_group<F>(ctx: AppCtx, f: F) -> Response
where
    F: FnOnce(&mut SessionState, usize),
{
    let mut state = ctx.inner.lock().expect("backend state lock");
    let Some(session) = state.session.as_mut() else {
        return json_error(StatusCode::BAD_REQUEST, "没有会话");
    };
    advance_current_index(session);
    if session.current_group >= session.groups.len() {
        return Json(json!({"done": true})).into_response();
    }
    let idx = session.current_group;
    f(session, idx);
    if session.groups[idx].finished {
        let _ = apply_group(session, idx);
    }
    advance_current_index(session);
    let _ = save_state(session);
    current_group_payload(session)
}

fn current_group_payload(session: &mut SessionState) -> Response {
    advance_current_index(session);
    if session.current_group >= session.groups.len() {
        Json(json!({"done": true})).into_response()
    } else {
        Json(json!({
            "done": false,
            "group": serialize_group(session, session.current_group)
        }))
        .into_response()
    }
}

fn advance_current_index(session: &mut SessionState) {
    while session.current_group < session.groups.len()
        && session.groups[session.current_group].finished
    {
        if !session.groups[session.current_group].applied {
            let idx = session.current_group;
            let _ = apply_group(session, idx);
        }
        session.current_group += 1;
    }
}

fn advance(group: &mut GroupState, loser_side: &str) {
    if group.finished {
        return;
    }
    let drained_both = matches!(loser_side, "both" | "neither");
    match loser_side {
        "both" => {
            if let Some(left) = group.left.take() {
                group.losers.push(left);
            }
            if let Some(right) = group.right.take() {
                group.losers.push(right);
            }
        }
        "neither" => {
            if let Some(left) = group.left.take() {
                group.extra_winners.push(left);
            }
            if let Some(right) = group.right.take() {
                group.extra_winners.push(right);
            }
        }
        "left" => {
            if let Some(left) = group.left.take() {
                group.losers.push(left);
            }
        }
        "right" => {
            if let Some(right) = group.right.take() {
                group.losers.push(right);
            }
        }
        _ => return,
    }
    refill_and_finalize(group, drained_both);
}

fn kick_side(group: &mut GroupState, side: &str) -> bool {
    if group.finished {
        return false;
    }
    match side {
        "left" => {
            if let Some(left) = group.left.take() {
                group.losers.push(left);
            } else {
                return false;
            }
        }
        "right" => {
            if let Some(right) = group.right.take() {
                group.losers.push(right);
            } else {
                return false;
            }
        }
        _ => return false,
    }
    refill_and_finalize(group, false);
    true
}

fn refill_and_finalize(group: &mut GroupState, drained_both: bool) {
    if group.left.is_none() && !group.pending.is_empty() {
        group.left = Some(group.pending.remove(0));
    }
    if group.right.is_none() && !group.pending.is_empty() {
        group.right = Some(group.pending.remove(0));
    }
    if group.pending.is_empty() {
        match (group.left.clone(), group.right.clone()) {
            (Some(left), None) if !drained_both => {
                group.winner = Some(left);
                group.finished = true;
            }
            (None, Some(right)) if !drained_both => {
                group.winner = Some(right);
                group.finished = true;
            }
            (None, None) => {
                group.finished = true;
            }
            _ => {}
        }
    }
}

fn reset_group_live(group: &mut GroupState, live: Vec<String>) {
    group.left = None;
    group.right = None;
    group.pending.clear();
    group.winner = None;
    group.finished = false;
    group.applied = false;
    if live.is_empty() {
        group.finished = true;
        group.auto_selected = true;
    } else if live.len() == 1 {
        group.winner = live.first().cloned();
        group.finished = true;
        group.auto_selected = true;
    } else {
        group.left = live.get(0).cloned();
        group.right = live.get(1).cloned();
        group.pending = live.into_iter().skip(2).collect();
    }
}

fn serialize_group(session: &SessionState, idx: usize) -> Value {
    let g = &session.groups[idx];
    let loser_set = g.losers.iter().collect::<HashSet<_>>();
    let extra_set = g.extra_winners.iter().collect::<HashSet<_>>();
    let pending_set = g.pending.iter().collect::<HashSet<_>>();
    let members = g
        .images
        .iter()
        .map(|p| {
            let status = if Some(p) == g.left.as_ref() {
                "current-left"
            } else if Some(p) == g.right.as_ref() {
                "current-right"
            } else if loser_set.contains(p) {
                "loser"
            } else if extra_set.contains(p) || (g.finished && Some(p) == g.winner.as_ref()) {
                "winner"
            } else if pending_set.contains(p) {
                "pending"
            } else {
                "pending"
            };
            json!({"path": p, "name": file_name(p), "status": status})
        })
        .collect::<Vec<_>>();
    let decided = g.losers.len() + g.extra_winners.len();
    json!({
        "best_path": best_path(g, &session.meta),
        "earliest_dt": earliest_dt(g, &session.meta),
        "index": idx,
        "id": g.id,
        "id_short": g.id.chars().take(6).collect::<String>(),
        "total_images": g.images.len(),
        "decided": decided,
        "remaining_in_group": usize::from(g.left.is_some()) + usize::from(g.right.is_some()) + g.pending.len(),
        "left": g.left,
        "right": g.right,
        "left_meta": g.left.as_ref().and_then(|p| session.meta.get(p)).cloned(),
        "right_meta": g.right.as_ref().and_then(|p| session.meta.get(p)).cloned(),
        "members": members,
        "next_preload": g.pending.first(),
        "pending_count": g.pending.len(),
        "loser_count": g.losers.len(),
        "winner": g.winner,
        "finished": g.finished,
        "applied": g.applied,
        "can_undo": session.undo_stack.last().is_some_and(|u| u.group_index == idx),
    })
}

fn apply_group(session: &mut SessionState, idx: usize) -> Result<(), String> {
    if idx >= session.groups.len() || session.groups[idx].applied || !session.groups[idx].finished {
        return Ok(());
    }
    if session.dry_run {
        session.groups[idx].applied = true;
        return Ok(());
    }
    let folder = PathBuf::from(&session.folder);
    let win_dir = folder.join("winners");
    let lose_dir = folder.join("losers");
    fs::create_dir_all(&win_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&lose_dir).map_err(|e| e.to_string())?;
    let mode = session.mode.clone();
    let group = &mut session.groups[idx];
    let mut failed = false;
    if let Some(winner) = group.winner.clone() {
        match transfer_with_companions(&winner, &win_dir, session.companions.get(&winner), &mode) {
            Ok(result) => {
                record_transfer(group, &winner, "winner", &result);
                if mode == "move" {
                    group.winner = Some(result.main_target);
                }
            }
            Err(_) => failed = true,
        }
    }
    let mut new_extra_winners = Vec::with_capacity(group.extra_winners.len());
    for winner in group.extra_winners.clone() {
        match transfer_with_companions(&winner, &win_dir, session.companions.get(&winner), &mode) {
            Ok(result) => {
                record_transfer(group, &winner, "winner", &result);
                if mode == "move" {
                    new_extra_winners.push(result.main_target);
                } else {
                    new_extra_winners.push(winner);
                }
            }
            Err(_) => {
                failed = true;
                new_extra_winners.push(winner);
            }
        }
    }
    group.extra_winners = new_extra_winners;
    let mut new_losers = Vec::with_capacity(group.losers.len());
    for loser in group.losers.clone() {
        match transfer_with_companions(&loser, &lose_dir, session.companions.get(&loser), &mode) {
            Ok(result) => {
                record_transfer(group, &loser, "loser", &result);
                if mode == "move" {
                    new_losers.push(result.main_target);
                } else {
                    new_losers.push(loser);
                }
            }
            Err(_) => {
                failed = true;
                new_losers.push(loser);
            }
        }
    }
    group.losers = new_losers;
    group.applied = true;
    if failed {
        return Err("部分文件处理失败".to_string());
    }
    Ok(())
}

fn apply_finished_groups(session: &mut SessionState) -> Result<(), String> {
    for idx in 0..session.groups.len() {
        if session.groups[idx].finished && !session.groups[idx].applied {
            apply_group(session, idx)?;
        }
    }
    Ok(())
}

#[derive(Debug)]
struct TransferResult {
    main_target: String,
    companion_pairs: Vec<(String, String)>,
}

fn transfer_with_companions(
    src: &str,
    target_dir: &Path,
    companions: Option<&Vec<String>>,
    mode: &str,
) -> Result<TransferResult, String> {
    let main_target = unique_target(
        target_dir,
        Path::new(src)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("image"),
    );
    transfer_one(src, &main_target, mode)?;
    let stem = main_target
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image")
        .to_string();
    let mut companion_pairs = Vec::new();
    for comp in companions.into_iter().flatten() {
        let suffix = Path::new(comp)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let name = if suffix.is_empty() {
            stem.clone()
        } else {
            format!("{stem}.{suffix}")
        };
        let target = unique_target(target_dir, &name);
        if transfer_one(comp, &target, mode).is_ok() {
            companion_pairs.push((comp.clone(), target.to_string_lossy().to_string()));
        }
    }
    Ok(TransferResult {
        main_target: main_target.to_string_lossy().to_string(),
        companion_pairs,
    })
}

fn transfer_one(src: &str, target: &Path, mode: &str) -> Result<(), String> {
    if mode == "move" {
        fs::rename(src, target).map_err(|e| e.to_string())
    } else {
        fs::copy(src, target).map(|_| ()).map_err(|e| e.to_string())
    }
}

fn record_transfer(group: &mut GroupState, src: &str, kind: &str, result: &TransferResult) {
    group
        .move_log
        .push(json!({"src": src, "dst": result.main_target, "kind": kind}));
    let companion_kind = format!("{kind}_companion");
    for (comp_src, comp_dst) in &result.companion_pairs {
        group
            .move_log
            .push(json!({"src": comp_src, "dst": comp_dst, "kind": companion_kind}));
    }
}

fn revert_group_files(session: &mut SessionState, idx: usize) -> Vec<Value> {
    if idx >= session.groups.len() || session.groups[idx].move_log.is_empty() {
        return Vec::new();
    }
    let mode = session.mode.clone();
    let root = PathBuf::from(&session.folder);
    let mut failed = Vec::new();
    for entry in session.groups[idx].move_log.clone() {
        let src = entry.get("src").and_then(|v| v.as_str()).unwrap_or("");
        let dst = entry.get("dst").and_then(|v| v.as_str()).unwrap_or("");
        if dst.is_empty() {
            continue;
        }
        let dst_path = PathBuf::from(dst);
        if !dst_path.exists() {
            failed.push(json!({"path": dst, "reason": "target missing"}));
            continue;
        }
        if mode == "copy" {
            if let Err(err) = fs::remove_file(&dst_path) {
                failed.push(json!({"path": dst, "reason": err.to_string()}));
            }
            continue;
        }
        let mut restore_target = if src.is_empty() {
            root.join(
                dst_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("image"),
            )
        } else {
            PathBuf::from(src)
        };
        if restore_target.exists() {
            let name = restore_target
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("image")
                .to_string();
            restore_target = unique_target(restore_target.parent().unwrap_or(&root), &name);
        }
        if let Err(err) = fs::rename(&dst_path, &restore_target) {
            failed.push(json!({"path": dst, "reason": err.to_string()}));
        }
    }
    session.groups[idx].move_log.clear();
    failed
}

fn reset_group_for_reopen(group: &mut GroupState) {
    group.winner = None;
    group.extra_winners.clear();
    group.losers.clear();
    group.applied = false;
    group.finished = false;
    group.left = group.images.first().cloned();
    group.right = group.images.get(1).cloned();
    group.pending = group.images.iter().skip(2).cloned().collect();
}

fn unique_target(folder: &Path, name: &str) -> PathBuf {
    let target = folder.join(name);
    if !target.exists() {
        return target;
    }
    let path = Path::new(name);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    for i in 1.. {
        let candidate = if ext.is_empty() {
            folder.join(format!("{stem}_{i}"))
        } else {
            folder.join(format!("{stem}_{i}.{ext}"))
        };
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!()
}

async fn auto_rejected(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let state = ctx.inner.lock().expect("backend state lock");
    let items = state
        .session
        .as_ref()
        .map(|s| {
            s.groups
                .iter()
                .flat_map(|g| {
                    g.auto_rejected.iter().map(|p| {
                        json!({
                            "group_id": g.id,
                            "path": p,
                            "name": file_name(p),
                            "reason": g.auto_reject_reasons.get(p).cloned().unwrap_or_else(|| "智能初筛".to_string()),
                            "restored": s.prescreen_restored.contains(p),
                        })
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Json(json!({"items": items}))
}

#[derive(Debug, Deserialize)]
struct RestoreRequest {
    group_id: String,
    path: String,
}

async fn restore_rejected(State(ctx): State<AppCtx>, Json(req): Json<RestoreRequest>) -> Response {
    let mut state = ctx.inner.lock().expect("backend state lock");
    let Some(session) = state.session.as_mut() else {
        return json_error(StatusCode::BAD_REQUEST, "没有会话");
    };
    if let Some(group) = session.groups.iter_mut().find(|g| g.id == req.group_id) {
        group.auto_rejected.retain(|p| p != &req.path);
        group.losers.retain(|p| p != &req.path);
        if !group.manual_restored.contains(&req.path) {
            group.manual_restored.push(req.path.clone());
        }
        if !session.prescreen_restored.contains(&req.path) {
            session.prescreen_restored.push(req.path);
        }
        let live = group
            .images
            .iter()
            .filter(|p| !group.auto_rejected.contains(*p))
            .cloned()
            .collect::<Vec<_>>();
        reset_group_live(group, live);
    }
    let _ = save_state(session);
    Json(json!({"ok": true})).into_response()
}

async fn confirm_prescreen(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let mut state = ctx.inner.lock().expect("backend state lock");
    if let Some(session) = state.session.as_mut() {
        session.prescreen_reviewed = true;
        session.selection_started = true;
        let _ = apply_finished_groups(session);
        let _ = save_state(session);
    }
    Json(json!({
        "async": true,
        "all_paths": state.grouping.all_paths,
    }))
}

async fn grouping_progress(
    State(ctx): State<AppCtx>,
    Query(q): Query<SinceQuery>,
) -> impl IntoResponse {
    let state = ctx.inner.lock().expect("backend state lock");
    let since = q.since.unwrap_or(0) as usize;
    Json(json!({
        "status": state.grouping.status,
        "groups": state.grouping.groups.iter().skip(since).cloned().collect::<Vec<_>>(),
        "total": state.grouping.total,
        "multi": state.grouping.multi,
        "error": state.grouping.error,
    }))
}

#[derive(Debug, Deserialize)]
struct RegroupRequest {
    threshold_near: Option<i32>,
    threshold_far: Option<i32>,
    near_seconds: Option<i32>,
}

async fn regroup(State(ctx): State<AppCtx>, Json(req): Json<RegroupRequest>) -> Response {
    let mut state = ctx.inner.lock().expect("backend state lock");
    let infos = state.last_infos.clone();
    let Some(session) = state.session.as_mut() else {
        return json_error(StatusCode::BAD_REQUEST, "没有会话");
    };
    if let Some(v) = req.threshold_near {
        session.threshold_near = v;
    }
    if let Some(v) = req.threshold_far {
        session.threshold_far = v;
    }
    if let Some(v) = req.near_seconds {
        session.near_seconds = v;
    }
    let fast_infos = infos.iter().map(|r| r.info.clone()).collect::<Vec<_>>();
    session.groups = cluster(&fast_infos)
        .into_iter()
        .map(|idxs| {
            GroupState::new(
                idxs.into_iter()
                    .filter_map(|i| infos.get(i))
                    .map(|r| r.info.path.clone())
                    .collect(),
            )
        })
        .collect();
    session.current_group = 0;
    let _ = save_state(session);
    Json(json!({"ok": true})).into_response()
}

async fn preview_groups(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let state = ctx.inner.lock().expect("backend state lock");
    let groups = state
        .session
        .as_ref()
        .map(|s| preview_cards(&s.groups, &s.meta, false))
        .unwrap_or_default();
    Json(json!({"groups": groups}))
}

fn preview_cards(
    groups: &[GroupState],
    meta: &HashMap<String, Value>,
    include_single: bool,
) -> Vec<Value> {
    groups
        .iter()
        .filter(|g| include_single || g.images.len() > 1)
        .map(|g| {
            let samples = g.images.iter().take(4).cloned().collect::<Vec<_>>();
            let times = g
                .images
                .iter()
                .filter_map(|p| meta.get(p)?.get("mtime")?.as_f64())
                .collect::<Vec<_>>();
            let span = if times.len() >= 2 {
                let min = times.iter().copied().fold(f64::INFINITY, f64::min);
                let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (max - min).max(0.0)
            } else {
                0.0
            };
            json!({
                "id": g.id,
                "size": g.images.len(),
                "samples": samples,
                "best_path": best_path(g, meta),
                "earliest_dt": earliest_dt(g, meta),
                "span_seconds": span,
            })
        })
        .collect()
}

async fn skipped(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let state = ctx.inner.lock().expect("backend state lock");
    let skipped = state
        .session
        .as_ref()
        .map(|s| s.skipped.clone())
        .unwrap_or_else(|| state.job.skipped.clone());
    Json(json!({"skipped": skipped}))
}

async fn winners(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let state = ctx.inner.lock().expect("backend state lock");
    let entries = state
        .session
        .as_ref()
        .map(|s| {
            s.groups
                .iter()
                .flat_map(|g| {
                    let mut out = Vec::new();
                    if let Some(w) = &g.winner {
                        out.push(json!({"path": w, "name": file_name(w), "group_id": g.id, "group_size": g.images.len()}));
                    }
                    for w in &g.extra_winners {
                        out.push(json!({"path": w, "name": file_name(w), "group_id": g.id, "group_size": g.images.len()}));
                    }
                    out
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Json(json!({"entries": entries, "winners": entries}))
}

fn watermark_winner_paths(session: Option<&SessionState>) -> Vec<String> {
    session
        .map(|s| {
            s.groups
                .iter()
                .flat_map(|g| {
                    let mut out = Vec::new();
                    if let Some(w) = &g.winner {
                        out.push(w.clone());
                    }
                    out.extend(g.extra_winners.iter().cloned());
                    out
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn run_watermark_job(ctx: AppCtx, winners: Vec<String>, out_dir: PathBuf, cfg: WatermarkConfig) {
    let mut progress = |done: usize, total: usize, name: String| {
        let mut state = ctx.inner.lock().expect("backend state lock");
        state.watermark.done = done;
        state.watermark.total = total;
        state.watermark.current = name;
    };
    let mut cancel = || {
        let state = ctx.inner.lock().expect("backend state lock");
        state.watermark.cancel_requested
    };
    let result = watermark::batch_export(
        &ctx.backend_dir,
        &winners,
        &out_dir,
        &cfg,
        Some(&mut progress),
        Some(&mut cancel),
    );
    let mut state = ctx.inner.lock().expect("backend state lock");
    state.watermark.finished_at = now_secs();
    match result {
        Ok(report) => {
            state.watermark.ok = report["ok"].as_u64().unwrap_or(0) as usize;
            state.watermark.failed = report["failed"]
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| {
                            Some((
                                item.get(0)?.as_str()?.to_string(),
                                item.get(1)?.as_str()?.to_string(),
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default();
            state.watermark.status = if state.watermark.cancel_requested {
                "cancelled".to_string()
            } else {
                "done".to_string()
            };
        }
        Err(err) => {
            state.watermark.status = "error".to_string();
            state.watermark.error = Some(err);
        }
    }
}

async fn peek_folder(Json(req): Json<Value>) -> Response {
    let Some(folder) = req.get("folder").and_then(|v| v.as_str()) else {
        return json_error(StatusCode::BAD_REQUEST, "缺少 folder");
    };
    let folder = PathBuf::from(folder);
    if !folder.is_dir() {
        return Json(json!({"ok": false, "count": 0, "error": "目录不存在"})).into_response();
    }
    let pairs = scan_folder(&folder);
    let samples = pairs
        .iter()
        .take(3)
        .map(|p| p.primary.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let size = pairs
        .iter()
        .filter_map(|p| fs::metadata(&p.primary).ok().map(|m| m.len()))
        .sum::<u64>();
    Json(json!({
        "ok": true,
        "count": pairs.len(),
        "samples": samples,
        "earliest": "",
        "latest": "",
        "active_period": "",
        "span_days": 1,
        "size_text": format_bytes(size),
    }))
    .into_response()
}

async fn open_folder(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let folder = ctx
        .inner
        .lock()
        .expect("backend state lock")
        .session
        .as_ref()
        .map(|s| s.folder.clone());
    if let Some(folder) = folder {
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("explorer").arg(folder).spawn();
        }
    }
    Json(json!({"ok": true}))
}

async fn image_endpoint(Query(q): Query<HashMap<String, String>>) -> Response {
    let Some(path) = q.get("path") else {
        return json_error(StatusCode::BAD_REQUEST, "缺少 path");
    };
    let max_side = q
        .get("w")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(THUMB_MAX);
    let path = PathBuf::from(path);
    let Ok(img) = image::open(&path) else {
        return broken_placeholder();
    };
    let resized = img.resize(max_side, max_side, FilterType::Lanczos3);
    let mut bytes = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    if resized.write_to(&mut cursor, ImageFormat::Jpeg).is_err() {
        return broken_placeholder();
    }
    binary_response(bytes, "image/jpeg", None)
}

async fn image_original(Query(q): Query<HashMap<String, String>>) -> Response {
    let Some(path) = q.get("path") else {
        return json_error(StatusCode::BAD_REQUEST, "缺少 path");
    };
    file_response(PathBuf::from(path), Some("application/octet-stream"))
}

fn push_job_event(ctx: &AppCtx, record: &InfoRecord, reason: Option<String>) {
    let mut state = ctx.inner.lock().expect("backend state lock");
    let q = record.info.quality.as_ref();
    let reject = q.and_then(|q| q.auto_reject).unwrap_or(false);
    let engine = state.job.engine.clone();
    let llm_verdict = q
        .and_then(|q| q.extra.get("llm_verdict"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let llm_reason = q
        .and_then(|q| q.extra.get("llm_reason"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let signals = if engine == "expert" || engine == "tycoon" {
        let dino_value = record
            .info
            .dinov2
            .as_ref()
            .map(|v| format!("{} dims", v.len()))
            .unwrap_or_else(|| "missing".to_string());
        let face_value = if record.info.face_embeddings.is_empty() {
            q.and_then(|q| q.extra.get("face_unavailable"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| "0 faces".to_string())
        } else {
            format!("{} faces", record.info.face_embeddings.len())
        };
        let quality_value = q
            .and_then(|q| q.extra.get("quality_models"))
            .and_then(|v| v.as_bool())
            .map(|available| {
                if available {
                    "MUSIQ + CLIP-IQA+"
                } else {
                    "not installed"
                }
            })
            .unwrap_or("not installed");
        vec![
            json!({"kind": "dino", "label": "DINOv2", "value": dino_value}),
            json!({"kind": "face", "label": "Face", "value": face_value}),
            json!({"kind": "quality", "label": "Quality", "value": quality_value}),
            json!({"kind": "llm", "label": "LLM", "value": llm_verdict.clone().unwrap_or_else(|| "none".to_string())}),
        ]
    } else {
        vec![
            json!({"kind": "hash", "label": "hash", "value": record.info.ahash.clone().unwrap_or_default()}),
            json!({"kind": "color", "label": "HSV", "value": if record.info.color_hist.is_some() { "ready" } else { "missing" }}),
            json!({"kind": "orb", "label": "ORB", "value": "pending"}),
            json!({"kind": "llm", "label": "LLM", "value": llm_verdict.clone().unwrap_or_else(|| "none".to_string())}),
        ]
    };
    state.job.event_seq += 1;
    let event = JobEvent {
        seq: state.job.event_seq,
        name: file_name(&record.info.path),
        path: record.info.path.clone(),
        ok: true,
        reject,
        reason: reason
            .or_else(|| llm_reason.clone())
            .or_else(|| q.and_then(|q| q.reject_reason.clone())),
        verdict: if reject { "reject" } else { "pass" }.to_string(),
        engine,
        shutter: record
            .info
            .exif_summary
            .as_ref()
            .and_then(|e| e.shutter.clone()),
        aperture: record
            .info
            .exif_summary
            .as_ref()
            .and_then(|e| e.aperture.clone()),
        iso: record
            .info
            .exif_summary
            .as_ref()
            .and_then(|e| e.iso.clone()),
        signals,
    };
    state.job.recent_events.push(event);
    if state.job.recent_events.len() > 200 {
        state.job.recent_events.remove(0);
    }
}

fn push_skip_event(ctx: &AppCtx, skipped: &SkippedItem) {
    let mut state = ctx.inner.lock().expect("backend state lock");
    state.job.event_seq += 1;
    let event = JobEvent {
        seq: state.job.event_seq,
        name: file_name(&skipped.path),
        path: skipped.path.clone(),
        ok: false,
        reject: false,
        reason: Some(skipped.reason.clone()),
        verdict: "skip".to_string(),
        engine: "fast".to_string(),
        shutter: None,
        aperture: None,
        iso: None,
        signals: vec![
            json!({"kind": "skip", "label": "—", "value": "—"}),
            json!({"kind": "skip", "label": "—", "value": "—"}),
            json!({"kind": "skip", "label": "—", "value": "—"}),
        ],
    };
    state.job.recent_events.push(event);
}

fn meta_entry(record: &InfoRecord) -> Value {
    let mut map = serde_json::Map::new();
    if let Some(exif) = &record.info.exif_summary {
        if let Ok(Value::Object(obj)) = serde_json::to_value(exif) {
            map.extend(obj.into_iter().filter(|(_, v)| !v.is_null()));
        }
    }
    if let Some(q) = &record.info.quality {
        if let Ok(Value::Object(obj)) = serde_json::to_value(q) {
            map.extend(obj.into_iter().filter(|(_, v)| !v.is_null()));
        }
    }
    if let Some(mtime) = record.info.mtime {
        map.insert("mtime".to_string(), json!(mtime));
    }
    if let Some(dinov2) = &record.info.dinov2 {
        map.insert("dinov2_dim".to_string(), json!(dinov2.len()));
    }
    if !record.info.face_embeddings.is_empty() {
        map.insert(
            "face_embedding_count".to_string(),
            json!(record.info.face_embeddings.len()),
        );
    }
    Value::Object(map)
}

fn best_path(group: &GroupState, meta: &HashMap<String, Value>) -> Option<String> {
    group
        .images
        .iter()
        .max_by(|a, b| {
            score_for(a, meta)
                .partial_cmp(&score_for(b, meta))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
}

fn score_for(path: &str, meta: &HashMap<String, Value>) -> f64 {
    meta.get(path)
        .and_then(|m| m.get("quality_score"))
        .and_then(|v| v.as_f64())
        .unwrap_or(50.0)
}

fn earliest_dt(group: &GroupState, meta: &HashMap<String, Value>) -> Option<String> {
    group
        .images
        .iter()
        .filter_map(|p| meta.get(p)?.get("datetime")?.as_str().map(str::to_string))
        .min()
}

fn save_state(session: &SessionState) -> Result<(), String> {
    let path = Path::new(&session.folder).join(STATE_FILENAME);
    let text = serde_json::to_string_pretty(&json!({
        "schema": 6,
        "folder": session.folder,
        "dry_run": session.dry_run,
        "mode": session.mode,
        "engine": session.engine,
        "current_group": session.current_group,
        "threshold_near": session.threshold_near,
        "threshold_far": session.threshold_far,
        "near_seconds": session.near_seconds,
        "prescreen_enabled": session.prescreen_enabled,
        "prescreen_strength": session.prescreen_strength,
        "prescreen_reviewed": session.prescreen_reviewed,
        "prescreen_rejected": session.prescreen_rejected,
        "prescreen_reject_reasons": session.prescreen_reject_reasons,
        "prescreen_restored": session.prescreen_restored,
        "companions": session.companions,
        "groups": session.groups,
    }))
    .map_err(|e| e.to_string())?;
    fs::write(path, text).map_err(|e| e.to_string())
}

fn file_response(path: PathBuf, content_type: Option<&str>) -> Response {
    match fs::read(&path) {
        Ok(bytes) => {
            let ct = content_type
                .map(str::to_string)
                .unwrap_or_else(|| content_type_for(&path).to_string());
            binary_response(bytes, &ct, None)
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

fn binary_response(bytes: Vec<u8>, content_type: &str, extra: Option<(&str, &str)>) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(content_type)
            .unwrap_or(HeaderValue::from_static("application/octet-stream")),
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-store, must-revalidate"),
    );
    if let Some((k, v)) = extra {
        if let (Ok(name), Ok(value)) = (
            header::HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(v),
        ) {
            headers.insert(name, value);
        }
    }
    (headers, bytes).into_response()
}

fn broken_placeholder() -> Response {
    let svg = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 480 360'><rect width='100%' height='100%' fill='#efeadd'/><text x='50%' y='52%' text-anchor='middle' font-family='sans-serif' font-size='22' fill='#6b6256'>无法读取</text></svg>";
    binary_response(
        svg.as_bytes().to_vec(),
        "image/svg+xml",
        Some(("X-Image-Status", "failed")),
    )
}

fn json_error(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({"error": msg}))).into_response()
}

fn safe_join(base: &Path, rel: &str) -> Option<PathBuf> {
    let mut out = base.to_path_buf();
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(p) => out.push(p),
            _ => return None,
        }
    }
    Some(out)
}

fn content_type_for(path: &Path) -> &'static str {
    match ext_lower(path).as_str() {
        ".html" => "text/html; charset=utf-8",
        ".js" => "application/javascript; charset=utf-8",
        ".css" => "text/css; charset=utf-8",
        ".svg" => "image/svg+xml",
        ".png" => "image/png",
        ".jpg" | ".jpeg" => "image/jpeg",
        ".webp" => "image/webp",
        _ => "application/octet-stream",
    }
}

fn ext_lower(path: &Path) -> String {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| format!(".{}", s.to_lowercase()))
        .unwrap_or_default()
}

#[cfg(test)]
fn format_kind_for_path(path: &Path) -> &'static str {
    let ext = ext_lower(path);
    if RAW_EXTS.contains(&ext.as_str()) {
        "raw"
    } else if HEIC_EXTS.contains(&ext.as_str()) {
        "heic"
    } else if IMAGE_EXTS.contains(&ext.as_str()) {
        "image"
    } else if SIDECAR_EXTS.contains(&ext.as_str()) {
        "sidecar"
    } else {
        "other"
    }
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

fn now_secs() -> f64 {
    system_time_secs(SystemTime::now()).unwrap_or(0.0)
}

fn system_time_secs(t: SystemTime) -> Option<f64> {
    let d = t.duration_since(UNIX_EPOCH).ok()?;
    Some(d.as_secs_f64())
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn round5(v: f64) -> f64 {
    (v * 100000.0).round() / 100000.0
}

fn format_bytes(n: u64) -> String {
    if n < 1024 {
        format!("{n} B")
    } else if n < 1024 * 1024 {
        format!("{:.0} KB", n as f64 / 1024.0)
    } else {
        format!("{:.1} MB", n as f64 / 1024.0 / 1024.0)
    }
}

trait IsoFormat {
    fn isoformat(&self) -> String;
}

impl IsoFormat for chrono::NaiveDateTime {
    fn isoformat(&self) -> String {
        self.format("%Y-%m-%dT%H:%M:%S").to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    // Python samples Marziliano edge points with NumPy default_rng(0);
                    // Rust keeps deterministic scan order, so the score can drift slightly
                    // while flags and edge-width buckets remain checked below.
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
        let a_path = dir.path().join("scene_a.png");
        let b_path = dir.path().join("scene_b.png");
        let c_path = dir.path().join("scene_c.png");

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
