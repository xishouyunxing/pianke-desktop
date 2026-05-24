use axum::{
    body::Body,
    extract::{Path as AxumPath, Query, State},
    http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Local};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageFormat};
use pianke_core::fast::{
    analyze_from_signals, average_hash_from_luma, cluster, difference_hash_from_luma,
    perceptual_hash_from_luma, ExifSummary, FastImageInfo, FastQualityProfile, FastQualitySignals,
    QualityInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Cursor,
    net::{SocketAddr, TcpListener},
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::oneshot;
use uuid::Uuid;

const STATE_FILENAME: &str = ".pic_selecter_state.json";
const PIC_DIR: &str = "_pic_selecter";
const THUMB_MAX: u32 = 1600;

const IMAGE_EXTS: &[&str] = &[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"];
const RAW_EXTS: &[&str] = &[
    ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng", ".raf", ".orf", ".rw2",
    ".pef", ".rwl", ".srw", ".x3f",
];
const UNSUPPORTED_IMAGE_EXTS: &[&str] = &[".heic", ".heif"];
const SIDECAR_EXTS: &[&str] = &[".xmp"];

#[derive(Debug)]
pub struct ServerOptions {
    pub port: u16,
    pub token: Option<String>,
    pub backend_dir: PathBuf,
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
}

#[derive(Debug, Default)]
struct AppState {
    session: Option<SessionState>,
    job: JobState,
    last_infos: Vec<InfoRecord>,
    grouping: GroupingState,
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
        backend_dir: options.backend_dir,
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
        .route("/api/capabilities", get(capabilities))
        .route("/api/model_components", get(model_components))
        .route(
            "/api/model_components/install",
            post(model_component_install),
        )
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
        .route("/api/browse_folder", post(unavailable))
        .route(
            "/api/ark_key",
            get(unavailable).post(unavailable).delete(unavailable),
        )
        .route("/api/llm_models", get(unavailable))
        .route("/api/job_log", get(unavailable))
        .route("/api/llm_concurrency", get(unavailable))
        .route("/api/watermark/templates", get(unavailable))
        .route("/api/watermark/preview", post(unavailable))
        .route("/api/watermark/start", post(unavailable))
        .route("/api/watermark/status", get(unavailable))
        .route("/api/watermark/cancel", post(unavailable))
        .route("/api/watermark/open_out_dir", post(unavailable))
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

async fn capabilities() -> impl IntoResponse {
    Json(json!({
        "face_aware": false,
        "engines": ["fast"],
        "backend": "rust-fast",
        "rust_fast": true,
        "watermark": false,
        "expert_installed": false,
        "tycoon_ready": false,
        "model_components": {
            "expert": "not_installed",
            "tycoon": "not_installed"
        },
        "install_mode": "base",
        "python_required": false
    }))
}

async fn model_components() -> impl IntoResponse {
    Json(json!({
        "components": [
            {
                "id": "expert",
                "label": "Expert local models",
                "status": "not_installed",
                "estimated_size_mb": 850,
                "engines": ["expert"],
                "models": ["dinov2-small", "insightface-buffalo_l", "nima", "musiq", "clipiqa+"],
                "runtime": "onnxruntime",
                "download_required": true
            },
            {
                "id": "tycoon",
                "label": "Tycoon local grouping models",
                "status": "not_installed",
                "estimated_size_mb": 400,
                "engines": ["tycoon"],
                "models": ["dinov2-small", "insightface-buffalo_l"],
                "runtime": "onnxruntime+reqwest",
                "download_required": true
            }
        ],
        "cache_dir": null,
        "backend": "rust-fast"
    }))
}

async fn model_component_install() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "Rust model component installer is not implemented yet",
            "unavailable": true
        })),
    )
}

async fn unavailable() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({"error": "Rust Fast 后端暂不支持该功能", "unavailable": true})),
    )
}

async fn start_job(State(ctx): State<AppCtx>, Json(req): Json<StartRequest>) -> Response {
    if req.engine != "fast" {
        return json_error(
            StatusCode::BAD_REQUEST,
            "Rust Fast 后端第一轮仅支持 fast 模式",
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
    thread::spawn(move || run_fast_job(worker_ctx, req));
    Json(json!({"started": true, "backend": "rust-fast"})).into_response()
}

fn run_fast_job(ctx: AppCtx, req: StartRequest) {
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
            Ok(record) => {
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
    let idx_groups = cluster(&fast_infos);
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
        engine: "fast".to_string(),
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
        let unsupported = files
            .iter()
            .filter(|p| UNSUPPORTED_IMAGE_EXTS.contains(&ext_lower(p).as_str()))
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
                primary,
                companions,
                analysis: images.first().cloned(),
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
        } else if let Some(primary) = unsupported.first().cloned() {
            out.push(ScanPair {
                primary,
                companions: Vec::new(),
                analysis: None,
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
            && !UNSUPPORTED_IMAGE_EXTS.contains(&ext.as_str())
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

fn process_one(pair: &ScanPair, strength: &str) -> Result<InfoRecord, String> {
    let analysis = pair.analysis.as_ref().ok_or_else(|| {
        if RAW_EXTS.contains(&ext_lower(&pair.primary).as_str()) {
            "纯 RAW 暂未在 Rust Fast 后端启用，请保留同名 JPG".to_string()
        } else {
            "HEIC/HEIF 暂未在 Rust Fast 后端启用".to_string()
        }
    })?;
    let img = image::open(analysis).map_err(|e| format!("加载失败: {e}"))?;
    let meta = fs::metadata(&pair.primary).map_err(|e| format!("stat 失败: {e}"))?;
    let file_size = meta.len();
    let mtime = meta
        .modified()
        .ok()
        .and_then(system_time_secs)
        .unwrap_or_else(now_secs);
    let (width, height) = img.dimensions();
    let exif = ExifSummary {
        width: Some(width),
        height: Some(height),
        file_size: Some(file_size),
        datetime: meta
            .modified()
            .ok()
            .map(|t| DateTime::<Local>::from(t).naive_local().isoformat()),
        ..ExifSummary::default()
    };

    let gray8 = img.to_luma8();
    let small_a = DynamicImage::ImageLuma8(gray8.clone())
        .resize_exact(8, 8, FilterType::Lanczos3)
        .to_luma8();
    let small_d = DynamicImage::ImageLuma8(gray8.clone())
        .resize_exact(9, 8, FilterType::Lanczos3)
        .to_luma8();
    let small_p = DynamicImage::ImageLuma8(gray8.clone())
        .resize_exact(32, 32, FilterType::Lanczos3)
        .to_luma8();
    let ahash = average_hash_from_luma(small_a.as_raw(), 8).unwrap_or_default();
    let dhash = difference_hash_from_luma(small_d.as_raw(), 8).unwrap_or_default();
    let phash = perceptual_hash_from_luma(small_p.as_raw(), 8).unwrap_or_default();
    let whash = ahash.clone();

    let signals = quality_signals(&img, file_size);
    let quality_result = analyze_from_signals(&signals, FastQualityProfile::from_name(strength));
    let quality = QualityInfo {
        blur_score: Some(signals.blur_score),
        brightness_mean: Some(signals.brightness_mean),
        brightness_std: Some(signals.brightness_std),
        contrast_score: Some(signals.contrast_score),
        overexposed_ratio: Some(signals.overexposed_ratio),
        underexposed_ratio: Some(signals.underexposed_ratio),
        entropy: Some(signals.entropy),
        width: Some(width),
        height: Some(height),
        file_size: Some(file_size),
        quality_score: Some(quality_result.quality_score),
        flags: quality_result.flags,
        auto_reject: Some(quality_result.auto_reject),
        reject_reason: quality_result.reject_reason,
        blur_combined: Some(signals.blur_combined),
        motion_anisotropy: Some(signals.motion_anisotropy),
        edge_width_pix: signals.edge_width_pix,
        focus_ratio: signals.focus_ratio,
        horizon_tilt_deg: signals.horizon_tilt_deg,
        composition: signals.composition,
        salient_sharpness: signals.salient_sharpness,
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

fn quality_signals(img: &DynamicImage, file_size: u64) -> FastQualitySignals {
    let gray = img
        .resize(768, 768, FilterType::Triangle)
        .to_luma8()
        .into_raw();
    let count = gray.len().max(1) as f64;
    let mean = gray.iter().map(|v| f64::from(*v)).sum::<f64>() / count;
    let var = gray
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / count;
    let std = var.sqrt();
    let under = gray.iter().filter(|v| **v <= 8).count() as f64 / count;
    let over = gray.iter().filter(|v| **v >= 247).count() as f64 / count;
    let entropy = entropy(&gray);
    let blur_combined = (std / 64.0).clamp(0.0, 1.0);
    let (w, h) = img.dimensions();
    FastQualitySignals {
        width: w,
        height: h,
        file_size,
        blur_score: std * std,
        brightness_mean: round3(mean),
        brightness_std: round3(std),
        contrast_score: round3(std),
        overexposed_ratio: round5(over),
        underexposed_ratio: round5(under),
        entropy: round5(entropy),
        blur_combined: round3(blur_combined),
        salient_sharpness: None,
        motion_anisotropy: 0.0,
        edge_width_pix: None,
        focus_ratio: None,
        horizon_tilt_deg: None,
        composition: Some(0.5),
        worst_clip_dark: under,
        worst_clip_bright: over,
    }
}

fn entropy(bytes: &[u8]) -> f64 {
    let mut hist = [0usize; 256];
    for b in bytes {
        hist[*b as usize] += 1;
    }
    let total = bytes.len().max(1) as f64;
    hist.iter()
        .filter(|v| **v > 0)
        .map(|v| {
            let p = *v as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn compute_color_hist(img: &DynamicImage) -> Option<Vec<f32>> {
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
    state.job.event_seq += 1;
    let event = JobEvent {
        seq: state.job.event_seq,
        name: file_name(&record.info.path),
        path: record.info.path.clone(),
        ok: true,
        reject,
        reason: reason.or_else(|| q.and_then(|q| q.reject_reason.clone())),
        verdict: if reject { "reject" } else { "pass" }.to_string(),
        engine: "fast".to_string(),
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
        signals: vec![
            json!({"kind": "hash", "label": "hash", "value": record.info.ahash.clone().unwrap_or_default()}),
            json!({"kind": "color", "label": "HSV", "value": if record.info.color_hist.is_some() { "已建" } else { "数据不足" }}),
            json!({"kind": "orb", "label": "ORB", "value": "待接入"}),
        ],
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
}
