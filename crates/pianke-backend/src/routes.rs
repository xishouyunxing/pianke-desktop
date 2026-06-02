use axum::{
    body::Body,
    extract::{Path as AxumPath, Query, State},
    http::{header, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    env, fs,
    io::Cursor,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, UNIX_EPOCH},
};

use crate::job::*;
use crate::llm_provider::SaveProviderRequest;
use crate::model_components::ComponentInstallRequest;
use crate::session::*;
use crate::types::*;
use crate::util::*;
use crate::watermark;
use crate::{DEFAULT_APP_UPDATE_URL, PIC_DIR, THUMB_MAX};

pub(crate) fn build_router(ctx: AppCtx) -> Router {
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
        .route("/api/model_components/delete", post(model_component_delete))
        .route("/api/model_components/pause", post(model_component_pause))
        .route("/api/model_components/cancel", post(model_component_cancel))
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

async fn app_update() -> impl IntoResponse {
    let current_version = env!("CARGO_PKG_VERSION").to_string();
    let manifest_url =
        env::var("PIANKE_APP_UPDATE_URL").unwrap_or_else(|_| DEFAULT_APP_UPDATE_URL.to_string());

    match fetch_app_update_manifest(&manifest_url).await {
        Ok(manifest) => {
            let valid_url =
                manifest.url.starts_with("https://") || manifest.url.starts_with("http://");
            let update_available =
                valid_url && compare_versions(&manifest.version, &current_version).is_gt();
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
    if manifest.version.trim().is_empty()
        || !(manifest.url.starts_with("https://") || manifest.url.starts_with("http://"))
    {
        return Err("软件更新清单缺少 version 或 url".to_string());
    }
    Ok(manifest)
}

async fn capabilities(State(ctx): State<AppCtx>) -> impl IntoResponse {
    let expert_installed = ctx.models.is_installed("expert");
    let expert_caps = ctx.models.expert_capabilities();
    let face_aware = expert_installed
        && expert_caps.insightface_detection
        && expert_caps.insightface_recognition
        && expert_caps.insightface_landmark;
    let quality_models = expert_installed && expert_caps.quality_models;
    let quality_models_reason = if quality_models {
        "available"
    } else if !expert_installed {
        "expert_component_not_installed"
    } else if !ctx.models.is_installed("expert-quality") {
        "expert_quality_component_not_installed"
    } else if expert_caps.musiq && expert_caps.clipiqa {
        "quality_models_fixed_shape_not_parity_verified"
    } else {
        "quality_models_not_installed"
    };
    let llm_status = ctx.llm.provider_status();
    let tycoon_ready = llm_status.configured
        && expert_installed
        && expert_caps.dinov2
        && expert_caps.insightface_detection
        && expert_caps.insightface_recognition
        && expert_caps.insightface_landmark;
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
        "quality_models_reason": quality_models_reason,
        "nima_legacy_unavailable": expert_caps.nima_legacy_unavailable,
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
    let cfg = serde_json::from_value::<watermark::WatermarkConfig>(req.clone()).unwrap_or_default();
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
    let cfg = serde_json::from_value::<watermark::WatermarkConfig>(req).unwrap_or_default();
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
    if req.engine == "tycoon" {
        if !ctx.models.is_installed("expert") {
            return json_error(
                StatusCode::PRECONDITION_REQUIRED,
                "Tycoon 模式需要先安装 Expert 标准组件",
            );
        }
        let expert_caps = ctx.models.expert_capabilities();
        if !expert_caps.dinov2
            || !expert_caps.insightface_detection
            || !expert_caps.insightface_recognition
            || !expert_caps.insightface_landmark
        {
            return json_error(
                StatusCode::PRECONDITION_REQUIRED,
                "Tycoon 模式需要 Expert 标准组件：DINOv2 + InsightFace 三个 ONNX 模型",
            );
        }
        if let Err(err) = ctx.llm.require_tycoon_ready(req.llm_model.as_deref()) {
            return json_error(err.status, &err.message);
        }
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
    thread::spawn(move || run_job_guarded(worker_ctx, req));
    Json(json!({"started": true, "backend": "rust-fast"})).into_response()
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

async fn choose(State(ctx): State<AppCtx>, Json(req): Json<ChooseRequest>) -> Response {
    mutate_current_group(ctx, |session, idx| {
        let snapshot = session.groups[idx].clone();
        session.undo_stack.push(UndoEntry {
            group_index: idx,
            group: snapshot,
        });
        if session.undo_stack.len() > 50 {
            session.undo_stack.remove(0);
        }
        advance(&mut session.groups[idx], &req.loser);
    })
}

async fn kick(State(ctx): State<AppCtx>, Json(req): Json<KickRequest>) -> Response {
    mutate_current_group(ctx, |session, idx| {
        let snapshot = session.groups[idx].clone();
        session.undo_stack.push(UndoEntry {
            group_index: idx,
            group: snapshot,
        });
        if session.undo_stack.len() > 50 {
            session.undo_stack.remove(0);
        }
        kick_side(&mut session.groups[idx], &req.side);
    })
}

async fn undo(State(ctx): State<AppCtx>) -> Response {
    // Phase 1: plan revert
    let (mut response_val, revert_plan) = {
        let mut state = ctx.inner.lock().expect("backend state lock");
        let Some(session) = state.session.as_mut() else {
            return json_error(StatusCode::BAD_REQUEST, "没有会话");
        };
        let plan = if let Some(entry) = session.undo_stack.last().cloned() {
            if entry.group_index < session.groups.len() {
                let plan = plan_group_revert(session, entry.group_index);
                session.groups[entry.group_index] = entry.group;
                session.current_group = entry.group_index;
                Some(plan)
            } else {
                None
            }
        } else {
            None
        };
        let _ = save_state(session);
        (current_group_payload(session), plan)
    };
    // Phase 2: execute revert without lock
    let failed = if let Some(ref plan) = revert_plan {
        execute_revert_plan(plan)
    } else {
        Vec::new()
    };
    // Phase 3: commit revert
    if let Some(plan) = revert_plan {
        let mut state = ctx.inner.lock().expect("backend state lock");
        if let Some(session) = state.session.as_mut() {
            if failed.is_empty() {
                if session
                    .undo_stack
                    .last()
                    .is_some_and(|entry| entry.group_index == plan.group_index)
                {
                    session.undo_stack.pop();
                }
                commit_revert(session, plan.group_index);
                let _ = save_state(session);
                response_val = current_group_payload(session);
            } else {
                restore_revert_log(session, &plan);
                let _ = save_state(session);
                return (
                    StatusCode::CONFLICT,
                    Json(json!({"error": "撤销文件恢复失败", "failed": failed})),
                )
                    .into_response();
            }
        }
    }
    response_val
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

async fn reopen_group(State(ctx): State<AppCtx>, Json(req): Json<ReopenRequest>) -> Response {
    // Phase 1: plan revert
    let revert_plan = {
        let mut state = ctx.inner.lock().expect("backend state lock");
        let Some(session) = state.session.as_mut() else {
            return json_error(StatusCode::BAD_REQUEST, "没有会话");
        };
        let plan = if let Some(idx) = session.groups.iter().position(|g| g.id == req.group_id) {
            let plan = plan_group_revert(session, idx);
            reset_group_for_reopen(&mut session.groups[idx]);
            session.current_group = idx;
            Some(plan)
        } else {
            None
        };
        let _ = save_state(session);
        plan
    };
    // Phase 2: execute revert without lock
    let failed = if let Some(ref plan) = revert_plan {
        execute_revert_plan(plan)
    } else {
        Vec::new()
    };
    // Phase 3: commit revert
    if let Some(plan) = revert_plan {
        let mut state = ctx.inner.lock().expect("backend state lock");
        if let Some(session) = state.session.as_mut() {
            if failed.is_empty() {
                commit_revert(session, plan.group_index);
            } else {
                restore_revert_log(session, &plan);
            }
            let _ = save_state(session);
        }
    }
    Json(json!({"ok": failed.is_empty(), "failed": failed})).into_response()
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
    let mut options = pianke_core::fast::FastClusterOptions::default();
    options.time_halflife = (session.near_seconds as f64 / 5.0).max(5.0);
    options.threshold = (0.80 - session.threshold_near as f64 * 0.04)
        .max(0.10)
        .min(0.75);
    options.max_group_size = max_group_size_from_threshold_far(session.threshold_far);
    session.groups = pianke_core::fast::cluster_with_options(&fast_infos, &options)
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

fn thumb_cache_key(folder: &Path, img_path: &str, width: u32) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(img_path.as_bytes());
    hasher.update(width.to_le_bytes());
    let hex = format!("{:x}", hasher.finalize());
    folder
        .join(PIC_DIR)
        .join("thumbs")
        .join(format!("{}.jpg", &hex[..16]))
}

async fn image_endpoint(
    State(ctx): State<AppCtx>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let Some(path) = q.get("path") else {
        return json_error(StatusCode::BAD_REQUEST, "缺少 path");
    };
    let session_folder = ctx
        .inner
        .lock()
        .expect("backend state lock")
        .session
        .as_ref()
        .map(|s| s.folder.clone());
    let src = PathBuf::from(path);
    if let Some(ref folder) = session_folder {
        if !src.starts_with(folder) {
            return json_error(StatusCode::FORBIDDEN, "路径不在会话文件夹内");
        }
    }
    let max_side = q
        .get("w")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(THUMB_MAX);
    let src_mtime = fs::metadata(&src).ok().and_then(|m| m.modified().ok());

    let cache_path = src.parent().map(|p| thumb_cache_key(p, path, max_side));

    if let Some(ref cp) = cache_path {
        if let Ok(cached_meta) = fs::metadata(cp) {
            let cache_valid = match (src_mtime, cached_meta.modified().ok()) {
                (Some(s), Some(c)) => c >= s,
                _ => false,
            };
            if cache_valid {
                if let Ok(bytes) = fs::read(cp) {
                    return binary_response(bytes, "image/jpeg", None);
                }
            }
        }
    }

    let Ok(img) = image::open(&src) else {
        return broken_placeholder();
    };
    let resized = img.resize(max_side, max_side, image::imageops::FilterType::Lanczos3);
    let mut bytes = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    if resized
        .write_to(&mut cursor, image::ImageFormat::Jpeg)
        .is_err()
    {
        return broken_placeholder();
    }

    if let Some(cp) = cache_path {
        if let Some(parent) = cp.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&cp, &bytes);
    }

    binary_response(bytes, "image/jpeg", None)
}

async fn image_original(
    State(ctx): State<AppCtx>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let Some(path) = q.get("path") else {
        return json_error(StatusCode::BAD_REQUEST, "缺少 path");
    };
    let session_folder = ctx
        .inner
        .lock()
        .expect("backend state lock")
        .session
        .as_ref()
        .map(|s| s.folder.clone());
    let src = PathBuf::from(path);
    if let Some(ref folder) = session_folder {
        if !src.starts_with(folder) {
            return json_error(StatusCode::FORBIDDEN, "路径不在会话文件夹内");
        }
    }
    file_response(src, Some("application/octet-stream"))
}
