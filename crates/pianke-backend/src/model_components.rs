use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

const MANIFEST_FILENAME: &str = "component.json";
const INSTALL_STATE_FILENAME: &str = "install_state.json";
const INSTALL_CONTROL_FILENAME: &str = "install_control.json";
pub const DEFAULT_EXPERT_MANIFEST_URL: &str =
    "https://pianke.moeuu.cn/pianke/components/expert/onnx-v1/component.json";

#[derive(Debug, Clone)]
pub struct ModelManager {
    cache_dir: PathBuf,
}

impl ModelManager {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn list(&self) -> Vec<ComponentView> {
        component_catalog()
            .into_iter()
            .map(|def| self.view_for(def))
            .collect()
    }

    pub fn status_map(&self) -> Value {
        let mut map = serde_json::Map::new();
        for component in self.list() {
            map.insert(component.id, json!(component.status));
        }
        Value::Object(map)
    }

    pub fn is_installed(&self, id: &str) -> bool {
        self.list()
            .into_iter()
            .any(|component| component.id == id && component.status == "installed")
    }

    pub fn expert_capabilities(&self) -> ExpertComponentCapabilities {
        let dir = self.cache_dir.join("expert");
        let musiq = dir
            .join("models")
            .join("quality")
            .join("musiq.onnx")
            .exists();
        let clipiqa = dir
            .join("models")
            .join("quality")
            .join("clipiqa_plus.onnx")
            .exists();
        let nima = crate::expert_vision::ExpertQualityModels::nima_component_file_ready(&dir);
        let nima_extra =
            crate::expert_vision::ExpertQualityModels::extra_nima_component_files_ready(&dir);
        let quality_models =
            musiq && clipiqa && crate::expert_vision::quality_preprocessor_allows_parity(&dir);
        ExpertComponentCapabilities {
            dinov2: dir.join("models").join("dinov2-small.onnx").exists(),
            insightface_detection: dir
                .join("models")
                .join("insightface")
                .join("det_10g.onnx")
                .exists(),
            insightface_recognition: dir
                .join("models")
                .join("insightface")
                .join("w600k_r50.onnx")
                .exists(),
            insightface_landmark: dir
                .join("models")
                .join("insightface")
                .join("1k3d68.onnx")
                .exists(),
            musiq,
            clipiqa,
            nima,
            nima_extra,
            quality_models,
            nima_legacy: false,
            nima_legacy_unavailable: !nima,
        }
    }

    pub fn installed_dir(&self, id: &str) -> Result<PathBuf, String> {
        let component = self
            .list()
            .into_iter()
            .find(|component| component.id == id)
            .ok_or_else(|| format!("unknown model component: {id}"))?;
        if component.status != "installed" {
            return Err(format!("模型组件 {id} 尚未安装"));
        }
        Ok(PathBuf::from(component.install_dir))
    }

    pub fn available_engines(&self) -> Vec<String> {
        let mut engines = vec!["fast".to_string()];
        for component in self.list() {
            if component.status == "installed" {
                engines.extend(component.engines);
            }
        }
        engines.sort();
        engines.dedup();
        engines
    }

    pub async fn install(
        &self,
        req: ComponentInstallRequest,
    ) -> Result<(StatusCode, Value), (StatusCode, Value)> {
        let Some(def) = component_catalog().into_iter().find(|c| c.id == req.id) else {
            return Err((
                StatusCode::NOT_FOUND,
                json!({
                    "error": format!("unknown model component: {}", req.id),
                    "unavailable": true
                }),
            ));
        };
        match self.install_component(def, req).await {
            Ok(component) => Ok((
                StatusCode::OK,
                json!({
                    "ok": true,
                    "component": component,
                    "cache_dir": self.cache_dir.to_string_lossy()
                }),
            )),
            Err(err) => Err(err),
        }
    }

    pub fn delete(&self, id: &str) -> Result<ComponentView, (StatusCode, Value)> {
        let Some(def) = component_catalog().into_iter().find(|c| c.id == id) else {
            return Err((
                StatusCode::NOT_FOUND,
                json!({
                    "error": format!("unknown model component: {id}"),
                    "unavailable": true
                }),
            ));
        };
        let install_dir = self.cache_dir.join(def.id);
        let temp_dir = self.cache_dir.join(format!("{}.installing", def.id));
        if let Some(state) = read_install_state(&install_dir.join(INSTALL_STATE_FILENAME)) {
            if state.status == "running" {
                return Err((
                    StatusCode::CONFLICT,
                    json!({
                        "error": "组件正在安装中，当前版本请等待安装结束后再删除。",
                        "component": self.view_for(def)
                    }),
                ));
            }
        }
        if install_dir.exists() {
            fs::remove_dir_all(&install_dir)
                .map_err(|e| server_error(format!("remove component failed: {e}")))?;
        }
        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir)
                .map_err(|e| server_error(format!("remove temp component failed: {e}")))?;
        }
        Ok(self.view_for(def))
    }

    pub fn pause(&self, id: &str) -> Result<ComponentView, (StatusCode, Value)> {
        self.write_control(id, InstallControl::pause())
    }

    pub fn cancel(&self, id: &str) -> Result<ComponentView, (StatusCode, Value)> {
        self.write_control(id, InstallControl::cancel())
    }

    fn write_control(
        &self,
        id: &str,
        control: InstallControl,
    ) -> Result<ComponentView, (StatusCode, Value)> {
        let Some(def) = component_catalog().into_iter().find(|c| c.id == id) else {
            return Err((
                StatusCode::NOT_FOUND,
                json!({
                    "error": format!("unknown model component: {id}"),
                    "unavailable": true
                }),
            ));
        };
        let install_dir = self.cache_dir.join(def.id);
        fs::create_dir_all(&install_dir)
            .map_err(|e| server_error(format!("create component dir failed: {e}")))?;
        write_install_control(&install_dir.join(INSTALL_CONTROL_FILENAME), &control)
            .map_err(server_error)?;
        let state_path = install_dir.join(INSTALL_STATE_FILENAME);
        if let Some(mut state) = read_install_state(&state_path) {
            if state.status == "running" || state.status == "pause_requested" {
                state.status = if control.cancel_requested {
                    "cancel_requested".to_string()
                } else {
                    "pause_requested".to_string()
                };
                write_install_state(&state_path, &state).map_err(server_error)?;
            }
        }
        Ok(self.view_for(def))
    }

    fn view_for(&self, def: ComponentDef) -> ComponentView {
        let install_dir = self.cache_dir.join(def.id);
        let manifest_path = install_dir.join(MANIFEST_FILENAME);
        let manifest = read_manifest(&manifest_path);
        let install_state = read_install_state(&install_dir.join(INSTALL_STATE_FILENAME));
        let status = match &manifest {
            Some(manifest)
                if manifest.id == def.id
                    && manifest.version == def.version
                    && manifest.checksum_status.as_deref() == Some("verified")
                    && component_files_ready(def.id, &install_dir) =>
            {
                "installed"
            }
            Some(manifest) if manifest.id == def.id && manifest.version == def.version => {
                "unverified"
            }
            Some(_) => "version_mismatch",
            None => "not_installed",
        };

        ComponentView {
            id: def.id.to_string(),
            label: def.label.to_string(),
            status: status.to_string(),
            version: def.version.to_string(),
            estimated_size_mb: def.estimated_size_mb,
            engines: def.engines.iter().map(|s| s.to_string()).collect(),
            models: def.models.iter().map(|s| s.to_string()).collect(),
            runtime: def.runtime.to_string(),
            download_required: status != "installed",
            install_dir: install_dir.to_string_lossy().to_string(),
            manifest,
            install_state,
        }
    }

    async fn install_component(
        &self,
        def: ComponentDef,
        req: ComponentInstallRequest,
    ) -> Result<ComponentView, (StatusCode, Value)> {
        let source = match InstallSource::from_request(&req) {
            Some(source) => source,
            None => {
                return Err((
                    StatusCode::PRECONDITION_REQUIRED,
                    json!({
                        "error": "缺少 Expert 模型组件 manifest_url、manifest_path 或 source_dir。基础包不会内置大模型，需要先配置组件清单。",
                        "component": self.view_for(def),
                        "manual_supported": true,
                        "cache_dir": self.cache_dir.to_string_lossy()
                    }),
                ));
            }
        };

        let install_dir = self.cache_dir.join(def.id);
        let temp_dir = self.cache_dir.join(format!("{}.installing", def.id));
        let state_path = install_dir.join(INSTALL_STATE_FILENAME);
        let can_resume_temp = read_install_state(&state_path)
            .map(|state| state.status == "paused")
            .unwrap_or(false);
        if let Err(err) = fs::create_dir_all(&install_dir) {
            return Err(server_error(format!("create component dir failed: {err}")));
        }
        write_install_state(
            &state_path,
            &InstallState::running(def.id, "reading_manifest", 0, 1),
        )
        .map_err(server_error)?;
        let control_path = install_dir.join(INSTALL_CONTROL_FILENAME);
        let _ = fs::remove_file(&control_path);

        let loaded = load_manifest(source).await.map_err(server_error)?;
        let mut manifest = loaded.manifest;
        if manifest.id != def.id {
            return Err(bad_request(format!(
                "manifest id mismatch: expected {}, got {}",
                def.id, manifest.id
            )));
        }
        if manifest.runtime != def.runtime {
            return Err(bad_request(format!(
                "runtime mismatch: expected {}, got {}",
                def.runtime, manifest.runtime
            )));
        }
        if manifest.version != def.version {
            return Err(bad_request(format!(
                "version mismatch: expected {}, got {}",
                def.version, manifest.version
            )));
        }

        if temp_dir.exists() && !can_resume_temp {
            fs::remove_dir_all(&temp_dir).map_err(|e| {
                server_error(format!("remove stale temp component dir failed: {e}"))
            })?;
        }
        fs::create_dir_all(&temp_dir)
            .map_err(|e| server_error(format!("create temp component dir failed: {e}")))?;

        let total = manifest.files.len().max(1);
        let total_bytes: u64 = manifest
            .files
            .iter()
            .filter_map(|file| file.size_bytes)
            .sum();
        let mut completed_bytes = 0_u64;
        for (idx, file) in manifest.files.iter().enumerate() {
            let rel = safe_relative_path(&file.path)
                .ok_or_else(|| bad_request(format!("unsafe model file path: {}", file.path)))?;
            let target = temp_dir.join(&rel);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| server_error(format!("create model subdir failed: {e}")))?;
            }
            let label = rel.to_string_lossy().to_string();
            if target.exists() {
                let existing_len = fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
                if file
                    .size_bytes
                    .map(|size| size == existing_len)
                    .unwrap_or(true)
                {
                    completed_bytes = completed_bytes.saturating_add(existing_len);
                    continue;
                }
            }
            write_install_state(
                &state_path,
                &InstallState::running(def.id, &label, idx, total)
                    .with_total_bytes(completed_bytes, total_bytes)
                    .with_file_bytes(0, file.size_bytes.unwrap_or(0)),
            )
            .map_err(server_error)?;
            let outcome = load_component_file_to_path(
                file,
                loaded.base_dir.as_deref(),
                &target,
                &state_path,
                &control_path,
                DownloadProgress {
                    id: def.id,
                    label: &label,
                    done: idx,
                    total,
                    completed_bytes,
                    total_bytes,
                },
            )
            .await
            .map_err(server_error)?;
            match outcome {
                DownloadOutcome::Complete(bytes_written) => {
                    completed_bytes = completed_bytes.saturating_add(bytes_written);
                }
                DownloadOutcome::Paused => {
                    write_install_state(
                        &state_path,
                        &InstallState::paused(def.id, &label, idx, total)
                            .with_total_bytes(completed_bytes, total_bytes),
                    )
                    .map_err(server_error)?;
                    return Err((
                        StatusCode::OK,
                        json!({
                            "ok": false,
                            "paused": true,
                            "component": self.view_for(def),
                            "cache_dir": self.cache_dir.to_string_lossy()
                        }),
                    ));
                }
                DownloadOutcome::Cancelled => {
                    write_install_state(
                        &state_path,
                        &InstallState::cancelled(def.id, &label, idx, total)
                            .with_total_bytes(completed_bytes, total_bytes),
                    )
                    .map_err(server_error)?;
                    let _ = fs::remove_dir_all(&temp_dir);
                    return Err((
                        StatusCode::OK,
                        json!({
                            "ok": false,
                            "cancelled": true,
                            "component": self.view_for(def),
                            "cache_dir": self.cache_dir.to_string_lossy()
                        }),
                    ));
                }
            }
            let bytes = fs::read(&target)
                .map_err(|e| server_error(format!("read downloaded component file failed: {e}")))?;
            if let Some(expected_size) = file.size_bytes {
                if expected_size != bytes.len() as u64 {
                    return Err(bad_request(format!(
                        "size mismatch for {}: expected {}, got {}",
                        file.path,
                        expected_size,
                        bytes.len()
                    )));
                }
            }
            if let Some(expected_hash) = &file.sha256 {
                let actual = sha256_hex(&bytes);
                if !actual.eq_ignore_ascii_case(expected_hash) {
                    return Err(bad_request(format!(
                        "sha256 mismatch for {}: expected {}, got {}",
                        file.path, expected_hash, actual
                    )));
                }
            }
        }

        manifest.installed_at = Some(chrono_like_timestamp());
        manifest.checksum_status = Some("verified".to_string());
        fs::write(
            temp_dir.join(MANIFEST_FILENAME),
            serde_json::to_vec_pretty(&manifest)
                .map_err(|e| server_error(format!("serialize manifest failed: {e}")))?,
        )
        .map_err(|e| server_error(format!("write manifest failed: {e}")))?;

        if install_dir.exists() {
            fs::remove_dir_all(&install_dir)
                .map_err(|e| server_error(format!("remove old component failed: {e}")))?;
        }
        fs::rename(&temp_dir, &install_dir)
            .map_err(|e| server_error(format!("activate component failed: {e}")))?;
        write_install_state(
            &install_dir.join(INSTALL_STATE_FILENAME),
            &InstallState::done(def.id, total).with_total_bytes(completed_bytes, total_bytes),
        )
        .map_err(server_error)?;
        Ok(self.view_for(def))
    }
}

#[derive(Debug, Deserialize)]
pub struct ComponentInstallRequest {
    pub id: String,
    #[serde(default)]
    pub manifest_url: Option<String>,
    #[serde(default)]
    pub manifest_path: Option<String>,
    #[serde(default)]
    pub source_dir: Option<String>,
}

#[derive(Debug, Clone)]
struct ComponentDef {
    id: &'static str,
    label: &'static str,
    version: &'static str,
    estimated_size_mb: u64,
    engines: &'static [&'static str],
    models: &'static [&'static str],
    runtime: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentManifest {
    pub id: String,
    pub version: String,
    pub runtime: String,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub files: Vec<ComponentFile>,
    #[serde(default)]
    pub installed_at: Option<String>,
    #[serde(default)]
    pub checksum_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentFile {
    pub path: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallState {
    pub id: String,
    pub status: String,
    pub done: usize,
    pub total: usize,
    pub current: String,
    #[serde(default)]
    pub bytes_done: u64,
    #[serde(default)]
    pub bytes_total: u64,
    #[serde(default)]
    pub current_file_bytes_done: u64,
    #[serde(default)]
    pub current_file_bytes_total: u64,
    #[serde(default)]
    pub error: Option<String>,
}

impl InstallState {
    fn running(id: &str, current: &str, done: usize, total: usize) -> Self {
        Self {
            id: id.to_string(),
            status: "running".to_string(),
            done,
            total,
            current: current.to_string(),
            bytes_done: 0,
            bytes_total: 0,
            current_file_bytes_done: 0,
            current_file_bytes_total: 0,
            error: None,
        }
    }

    fn paused(id: &str, current: &str, done: usize, total: usize) -> Self {
        Self {
            id: id.to_string(),
            status: "paused".to_string(),
            done,
            total,
            current: current.to_string(),
            bytes_done: 0,
            bytes_total: 0,
            current_file_bytes_done: 0,
            current_file_bytes_total: 0,
            error: None,
        }
    }

    fn cancelled(id: &str, current: &str, done: usize, total: usize) -> Self {
        Self {
            id: id.to_string(),
            status: "cancelled".to_string(),
            done,
            total,
            current: current.to_string(),
            bytes_done: 0,
            bytes_total: 0,
            current_file_bytes_done: 0,
            current_file_bytes_total: 0,
            error: None,
        }
    }

    fn done(id: &str, total: usize) -> Self {
        Self {
            id: id.to_string(),
            status: "done".to_string(),
            done: total,
            total,
            current: String::new(),
            bytes_done: 0,
            bytes_total: 0,
            current_file_bytes_done: 0,
            current_file_bytes_total: 0,
            error: None,
        }
    }

    fn with_total_bytes(mut self, done: u64, total: u64) -> Self {
        self.bytes_done = done;
        self.bytes_total = total;
        self
    }

    fn with_file_bytes(mut self, done: u64, total: u64) -> Self {
        self.current_file_bytes_done = done;
        self.current_file_bytes_total = total;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct InstallControl {
    #[serde(default)]
    pause_requested: bool,
    #[serde(default)]
    cancel_requested: bool,
}

impl InstallControl {
    fn pause() -> Self {
        Self {
            pause_requested: true,
            cancel_requested: false,
        }
    }

    fn cancel() -> Self {
        Self {
            pause_requested: false,
            cancel_requested: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ComponentView {
    pub id: String,
    pub label: String,
    pub status: String,
    pub version: String,
    pub estimated_size_mb: u64,
    pub engines: Vec<String>,
    pub models: Vec<String>,
    pub runtime: String,
    pub download_required: bool,
    pub install_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest: Option<ComponentManifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub install_state: Option<InstallState>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExpertComponentCapabilities {
    pub dinov2: bool,
    pub insightface_detection: bool,
    pub insightface_recognition: bool,
    pub insightface_landmark: bool,
    pub musiq: bool,
    pub clipiqa: bool,
    pub nima: bool,
    pub nima_extra: bool,
    pub quality_models: bool,
    pub nima_legacy: bool,
    pub nima_legacy_unavailable: bool,
}

fn component_catalog() -> Vec<ComponentDef> {
    vec![ComponentDef {
        id: "expert",
        label: "Expert local models",
        version: "onnx-v1",
        estimated_size_mb: 850,
        engines: &["expert"],
        models: &[
            "dinov2-small",
            "insightface-det_10g",
            "insightface-w600k_r50",
            "insightface-1k3d68",
            "quality-musiq-optional",
            "quality-clipiqa-plus-optional",
        ],
        runtime: "onnxruntime",
    }]
}

fn component_files_ready(id: &str, install_dir: &Path) -> bool {
    match id {
        "expert" => {
            install_dir
                .join("models")
                .join("dinov2-small.onnx")
                .exists()
                && crate::expert_vision::InsightFaceModels::component_files_ready(install_dir)
                && crate::expert_vision::ExpertQualityModels::component_ready_for_live_scoring(
                    install_dir,
                )
        }
        _ => true,
    }
}

fn read_manifest(path: &Path) -> Option<ComponentManifest> {
    let text = fs::read_to_string(path).ok()?;
    parse_manifest_text(&text).ok()
}

fn read_install_state(path: &Path) -> Option<InstallState> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_install_state(path: &Path, state: &InstallState) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let data = serde_json::to_vec_pretty(state).map_err(|e| e.to_string())?;
    fs::write(path, data).map_err(|e| e.to_string())
}

fn read_install_control(path: &Path) -> InstallControl {
    let Ok(text) = fs::read_to_string(path) else {
        return InstallControl::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn write_install_control(path: &Path, control: &InstallControl) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let data = serde_json::to_vec_pretty(control).map_err(|e| e.to_string())?;
    fs::write(path, data).map_err(|e| e.to_string())
}

enum InstallSource {
    SourceDir(PathBuf),
    ManifestPath(PathBuf),
    ManifestUrl(String),
}

impl InstallSource {
    fn from_request(req: &ComponentInstallRequest) -> Option<Self> {
        req.source_dir
            .as_ref()
            .map(|p| Self::SourceDir(PathBuf::from(p)))
            .or_else(|| {
                req.manifest_path
                    .as_ref()
                    .map(|p| Self::ManifestPath(PathBuf::from(p)))
            })
            .or_else(|| {
                req.manifest_url
                    .as_ref()
                    .map(|u| Self::ManifestUrl(u.clone()))
            })
            .or_else(|| Self::from_environment(&req.id))
            .or_else(|| Self::default_official(&req.id))
            .or_else(|| Self::local_packaged(&req.id))
    }

    fn from_environment(id: &str) -> Option<Self> {
        let prefix = id.to_ascii_uppercase().replace('-', "_");
        env::var(format!("PIANKE_{prefix}_SOURCE_DIR"))
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(|p| Self::SourceDir(PathBuf::from(p)))
            .or_else(|| {
                env::var(format!("PIANKE_{prefix}_MANIFEST_PATH"))
                    .ok()
                    .filter(|v| !v.trim().is_empty())
                    .map(|p| Self::ManifestPath(PathBuf::from(p)))
            })
            .or_else(|| {
                env::var(format!("PIANKE_{prefix}_MANIFEST_URL"))
                    .ok()
                    .filter(|v| !v.trim().is_empty())
                    .map(Self::ManifestUrl)
            })
    }

    fn default_official(id: &str) -> Option<Self> {
        match id {
            "expert" => Some(Self::ManifestUrl(DEFAULT_EXPERT_MANIFEST_URL.to_string())),
            _ => None,
        }
    }

    fn local_packaged(id: &str) -> Option<Self> {
        if id != "expert" {
            return None;
        }
        let candidates = [
            PathBuf::from("model_components").join(id),
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .unwrap_or_else(|| Path::new("."))
                .join("model_components")
                .join(id),
        ];
        candidates
            .into_iter()
            .find(|dir| dir.join(MANIFEST_FILENAME).exists())
            .map(Self::SourceDir)
    }
}

struct LoadedManifest {
    manifest: ComponentManifest,
    base_dir: Option<PathBuf>,
}

async fn load_manifest(source: InstallSource) -> Result<LoadedManifest, String> {
    match source {
        InstallSource::SourceDir(dir) => {
            let manifest_path = dir.join(MANIFEST_FILENAME);
            let manifest = read_manifest_required(&manifest_path)?;
            Ok(LoadedManifest {
                manifest,
                base_dir: Some(dir),
            })
        }
        InstallSource::ManifestPath(path) => {
            let manifest = read_manifest_required(&path)?;
            Ok(LoadedManifest {
                manifest,
                base_dir: path.parent().map(Path::to_path_buf),
            })
        }
        InstallSource::ManifestUrl(url) if url.starts_with("file://") => {
            let path = file_url_to_path(&url);
            let manifest = read_manifest_required(&path)?;
            Ok(LoadedManifest {
                manifest,
                base_dir: path.parent().map(Path::to_path_buf),
            })
        }
        InstallSource::ManifestUrl(url) => {
            let text = reqwest::get(&url)
                .await
                .map_err(|e| format!("download manifest failed: {e}"))?
                .error_for_status()
                .map_err(|e| format!("download manifest failed: {e}"))?
                .text()
                .await
                .map_err(|e| format!("read manifest body failed: {e}"))?;
            let manifest =
                parse_manifest_text(&text).map_err(|e| format!("parse manifest failed: {e}"))?;
            Ok(LoadedManifest {
                manifest,
                base_dir: None,
            })
        }
    }
}

fn read_manifest_required(path: &Path) -> Result<ComponentManifest, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read manifest failed: {e}"))?;
    parse_manifest_text(&text).map_err(|e| format!("parse manifest failed: {e}"))
}

fn parse_manifest_text(text: &str) -> Result<ComponentManifest, serde_json::Error> {
    serde_json::from_str(text.trim_start_matches('\u{feff}').trim_start())
}

struct DownloadProgress<'a> {
    id: &'a str,
    label: &'a str,
    done: usize,
    total: usize,
    completed_bytes: u64,
    total_bytes: u64,
}

enum DownloadOutcome {
    Complete(u64),
    Paused,
    Cancelled,
}

async fn load_component_file_to_path(
    file: &ComponentFile,
    base_dir: Option<&Path>,
    target: &Path,
    state_path: &Path,
    control_path: &Path,
    progress: DownloadProgress<'_>,
) -> Result<DownloadOutcome, String> {
    if let Some(url) = &file.url {
        if url.starts_with("file://") {
            let path = file_url_to_path(url);
            return copy_component_file_to_path(
                &path,
                target,
                state_path,
                control_path,
                progress,
                file.size_bytes.unwrap_or(0),
            );
        }
        return download_component_url_to_path(
            file,
            url,
            target,
            state_path,
            control_path,
            progress,
        )
        .await
        .map_err(|e| format!("download component file failed: {e}"));
    }
    let Some(base_dir) = base_dir else {
        return Err(format!("component file {} has no url", file.path));
    };
    let rel = safe_relative_path(&file.path)
        .ok_or_else(|| format!("unsafe model file path: {}", file.path))?;
    copy_component_file_to_path(
        &base_dir.join(rel),
        target,
        state_path,
        control_path,
        progress,
        file.size_bytes.unwrap_or(0),
    )
}

fn copy_component_file_to_path(
    source: &Path,
    target: &Path,
    state_path: &Path,
    control_path: &Path,
    progress: DownloadProgress<'_>,
    expected_size: u64,
) -> Result<DownloadOutcome, String> {
    let part = part_path(target);
    let mut input =
        fs::File::open(source).map_err(|e| format!("read component file failed: {e}"))?;
    let mut output =
        fs::File::create(&part).map_err(|e| format!("create component temp file failed: {e}"))?;
    let mut buffer = [0_u8; 1024 * 256];
    let mut file_done = 0_u64;
    loop {
        let control = read_install_control(control_path);
        if control.cancel_requested {
            return Ok(DownloadOutcome::Cancelled);
        }
        if control.pause_requested {
            return Ok(DownloadOutcome::Paused);
        }
        let n = input
            .read(&mut buffer)
            .map_err(|e| format!("read component file failed: {e}"))?;
        if n == 0 {
            break;
        }
        output
            .write_all(&buffer[..n])
            .map_err(|e| format!("write component temp file failed: {e}"))?;
        file_done += n as u64;
        write_install_state(
            state_path,
            &InstallState::running(progress.id, progress.label, progress.done, progress.total)
                .with_total_bytes(progress.completed_bytes + file_done, progress.total_bytes)
                .with_file_bytes(file_done, expected_size),
        )?;
    }
    output
        .flush()
        .map_err(|e| format!("flush component temp file failed: {e}"))?;
    fs::rename(&part, target).map_err(|e| format!("activate component file failed: {e}"))?;
    Ok(DownloadOutcome::Complete(file_done))
}

async fn download_component_url_to_path(
    file: &ComponentFile,
    url: &str,
    target: &Path,
    state_path: &Path,
    control_path: &Path,
    progress: DownloadProgress<'_>,
) -> Result<DownloadOutcome, String> {
    let part = part_path(target);
    let existing = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let client = reqwest::Client::new();
    let mut request = client.get(url);
    if existing > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={existing}-"));
    }
    let mut response = request
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("server returned error: {e}"))?;
    let range_accepted = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let append = existing > 0 && range_accepted;
    let mut output = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(&part)
        .map_err(|e| format!("create component temp file failed: {e}"))?;
    let mut file_done = if append { existing } else { 0 };
    let file_total = file.size_bytes.unwrap_or_else(|| {
        response
            .content_length()
            .map(|len| len + file_done)
            .unwrap_or(0)
    });
    loop {
        let control = read_install_control(control_path);
        if control.cancel_requested {
            return Ok(DownloadOutcome::Cancelled);
        }
        if control.pause_requested {
            return Ok(DownloadOutcome::Paused);
        }
        let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| format!("read response body failed: {e}"))?
        else {
            break;
        };
        output
            .write_all(&chunk)
            .map_err(|e| format!("write component temp file failed: {e}"))?;
        file_done += chunk.len() as u64;
        write_install_state(
            state_path,
            &InstallState::running(progress.id, progress.label, progress.done, progress.total)
                .with_total_bytes(progress.completed_bytes + file_done, progress.total_bytes)
                .with_file_bytes(file_done, file_total),
        )?;
    }
    output
        .flush()
        .map_err(|e| format!("flush component temp file failed: {e}"))?;
    fs::rename(&part, target).map_err(|e| format!("activate component file failed: {e}"))?;
    Ok(DownloadOutcome::Complete(file_done))
}

fn part_path(target: &Path) -> PathBuf {
    let mut name = target
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    name.push(".part");
    target.with_file_name(name)
}

fn file_url_to_path(url: &str) -> PathBuf {
    let mut path = url.trim_start_matches("file://").to_string();
    if cfg!(windows)
        && path.starts_with('/')
        && path.len() > 3
        && path.as_bytes().get(2) == Some(&b':')
    {
        path.remove(0);
    }
    PathBuf::from(path.replace("%20", " "))
}

fn safe_relative_path(path: &str) -> Option<PathBuf> {
    let p = Path::new(path);
    if p.is_absolute() {
        return None;
    }
    let mut out = PathBuf::new();
    for component in p.components() {
        match component {
            std::path::Component::Normal(part) => out.push(part),
            _ => return None,
        }
    }
    (!out.as_os_str().is_empty()).then_some(out)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn server_error(message: String) -> (StatusCode, Value) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": message, "unavailable": false}),
    )
}

fn bad_request(message: String) -> (StatusCode, Value) {
    (
        StatusCode::BAD_REQUEST,
        json!({"error": message, "unavailable": false}),
    )
}

fn chrono_like_timestamp() -> String {
    // Keep the component module independent from chrono; seconds are enough for diagnostics.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    format!("unix:{now}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fake_full_expert_component(root: &Path) -> Vec<(String, Vec<u8>)> {
        let files = vec![
            (
                "models/dinov2-small.onnx".to_string(),
                b"fake-dinov2".to_vec(),
            ),
            (
                "models/insightface/det_10g.onnx".to_string(),
                b"fake-det".to_vec(),
            ),
            (
                "models/insightface/w600k_r50.onnx".to_string(),
                b"fake-rec".to_vec(),
            ),
            (
                "models/insightface/1k3d68.onnx".to_string(),
                b"fake-landmark".to_vec(),
            ),
            (
                "models/quality/musiq.onnx".to_string(),
                b"fake-musiq".to_vec(),
            ),
            (
                "models/quality/clipiqa_plus.onnx".to_string(),
                b"fake-clipiqa".to_vec(),
            ),
            (
                "models/quality/nima_vgg16_ava.onnx".to_string(),
                b"fake-nima".to_vec(),
            ),
            (
                "models/quality/nima_inception_ava.onnx".to_string(),
                b"fake-nima-inception".to_vec(),
            ),
            (
                "models/quality/nima_koniq.onnx".to_string(),
                b"fake-nima-koniq".to_vec(),
            ),
            (
                "models/quality/nima_spaq.onnx".to_string(),
                b"fake-nima-spaq".to_vec(),
            ),
            (
                "quality_preprocessor.json".to_string(),
                br#"{"max_side":1024,"musiq_input_width":null,"musiq_input_height":null,"musiq_input_kind":"pyiqa_multiscale_patches","musiq_patch_size":32,"musiq_patch_stride":32,"musiq_hse_grid_size":10,"musiq_longer_side_lengths":[224,384],"musiq_max_seq_len_from_original_res":-1,"clipiqa_input_width":null,"clipiqa_input_height":null,"resize_filter":"pillow_lanczos","resize_rounding":"floor","nima_model":"pyiqa-nima-vgg16-ava","nima_input_width":224,"nima_input_height":224,"nima_resize_shorter":224,"nima_mean":[0.485,0.456,0.406],"nima_std":[0.229,0.224,0.225],"nima_input_name":"input","nima_output_name":"score","nima_extra_models":[{"field":"nima_inception_ava_score","path":"models/quality/nima_inception_ava.onnx","input_width":299,"input_height":299,"resize_shorter":299,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5],"input_name":"input","output_name":"score"},{"field":"nima_koniq_score","path":"models/quality/nima_koniq.onnx","input_width":299,"input_height":299,"resize_shorter":299,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5],"input_name":"input","output_name":"score"},{"field":"nima_spaq_score","path":"models/quality/nima_spaq.onnx","input_width":299,"input_height":299,"resize_shorter":299,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5],"input_name":"input","output_name":"score"}]}"#.to_vec(),
            ),
        ];
        for (rel, bytes) in &files {
            let path = root.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("fake component parent");
            }
            fs::write(path, bytes).expect("fake component file");
        }
        files
    }

    fn fake_full_expert_manifest(files: &[(String, Vec<u8>)]) -> String {
        let manifest_files = files
            .iter()
            .map(|(path, bytes)| {
                json!({
                    "path": path,
                    "sha256": sha256_hex(bytes),
                    "size_bytes": bytes.len()
                })
            })
            .collect::<Vec<_>>();
        serde_json::to_string_pretty(&json!({
            "id": "expert",
            "version": "onnx-v1",
            "runtime": "onnxruntime",
            "models": [
                "dinov2-small",
                "insightface-det_10g",
                "insightface-w600k_r50",
                "insightface-1k3d68",
                "quality-musiq",
                "quality-clipiqa-plus",
                "quality-nima-vgg16-ava",
                "quality-nima-inception-ava",
                "quality-nima-koniq",
                "quality-nima-spaq"
            ],
            "files": manifest_files,
            "checksum_status": "verified"
        }))
        .expect("fake full expert manifest")
    }

    #[test]
    fn lists_not_installed_components_without_cache_dir() {
        let temp = tempfile::tempdir().expect("temp dir");
        let manager = ModelManager::new(temp.path().join("models"));
        let components = manager.list();
        assert_eq!(components.len(), 1);
        assert!(components.iter().all(|c| c.status == "not_installed"));
    }

    #[test]
    fn installed_manifest_marks_component_installed() {
        let temp = tempfile::tempdir().expect("temp dir");
        let install_dir = temp.path().join("models").join("expert");
        let files = write_fake_full_expert_component(&install_dir);
        fs::write(
            install_dir.join(MANIFEST_FILENAME),
            fake_full_expert_manifest(&files),
        )
        .expect("manifest");

        let manager = ModelManager::new(temp.path().join("models"));
        let expert = manager
            .list()
            .into_iter()
            .find(|c| c.id == "expert")
            .expect("expert component");
        assert_eq!(expert.status, "installed");
        assert!(!expert.download_required);
        let caps = manager.expert_capabilities();
        assert!(caps.dinov2);
        assert!(caps.insightface_detection);
        assert!(caps.insightface_recognition);
        assert!(caps.insightface_landmark);
        assert!(caps.quality_models);
        assert!(caps.nima);
        assert!(caps.nima_extra);
        assert!(!caps.nima_legacy);
        assert!(!caps.nima_legacy_unavailable);
    }

    #[test]
    fn dinov2_only_manifest_is_unverified_not_installed() {
        let temp = tempfile::tempdir().expect("temp dir");
        let install_dir = temp.path().join("models").join("expert");
        fs::create_dir_all(install_dir.join("models")).expect("install dir");
        fs::write(
            install_dir.join("models").join("dinov2-small.onnx"),
            b"fake",
        )
        .expect("model");
        fs::write(
            install_dir.join(MANIFEST_FILENAME),
            r#"{"id":"expert","version":"onnx-v1","runtime":"onnxruntime","models":["dinov2-small"],"checksum_status":"verified"}"#,
        )
        .expect("manifest");

        let manager = ModelManager::new(temp.path().join("models"));
        let expert = manager
            .list()
            .into_iter()
            .find(|c| c.id == "expert")
            .expect("expert component");
        assert_eq!(expert.status, "unverified");
        assert!(expert.download_required);
        assert!(!manager.available_engines().contains(&"expert".to_string()));
        assert!(manager.expert_capabilities().dinov2);
    }

    #[test]
    fn optional_quality_models_without_parity_preprocessor_do_not_claim_parity() {
        let temp = tempfile::tempdir().expect("temp dir");
        let install_dir = temp.path().join("models").join("expert");
        fs::create_dir_all(install_dir.join("models").join("quality")).expect("quality dir");
        fs::write(
            install_dir
                .join("models")
                .join("quality")
                .join("musiq.onnx"),
            b"fake",
        )
        .expect("musiq");
        fs::write(
            install_dir
                .join("models")
                .join("quality")
                .join("clipiqa_plus.onnx"),
            b"fake",
        )
        .expect("clipiqa");

        let manager = ModelManager::new(temp.path().join("models"));
        let caps = manager.expert_capabilities();
        assert!(caps.musiq);
        assert!(caps.clipiqa);
        assert!(!caps.quality_models);
    }

    #[test]
    fn optional_quality_models_update_expert_capabilities_with_parity_preprocessor() {
        let temp = tempfile::tempdir().expect("temp dir");
        let install_dir = temp.path().join("models").join("expert");
        fs::create_dir_all(install_dir.join("models").join("quality")).expect("quality dir");
        fs::write(
            install_dir
                .join("models")
                .join("quality")
                .join("musiq.onnx"),
            b"fake",
        )
        .expect("musiq");
        fs::write(
            install_dir
                .join("models")
                .join("quality")
                .join("clipiqa_plus.onnx"),
            b"fake",
        )
        .expect("clipiqa");
        fs::write(
            install_dir.join("quality_preprocessor.json"),
            r#"{"max_side":1024,"musiq_input_kind":"pyiqa_multiscale_patches","clipiqa_input_width":null,"clipiqa_input_height":null,"resize_filter":"pillow_lanczos","resize_rounding":"floor"}"#,
        )
        .expect("quality preprocessor");

        let manager = ModelManager::new(temp.path().join("models"));
        let caps = manager.expert_capabilities();
        assert!(caps.musiq);
        assert!(caps.clipiqa);
        assert!(caps.quality_models);
    }

    #[test]
    fn fixed_shape_quality_models_do_not_claim_parity() {
        let temp = tempfile::tempdir().expect("temp dir");
        let install_dir = temp.path().join("models").join("expert");
        fs::create_dir_all(install_dir.join("models").join("quality")).expect("quality dir");
        fs::write(
            install_dir
                .join("models")
                .join("quality")
                .join("musiq.onnx"),
            b"fake",
        )
        .expect("musiq");
        fs::write(
            install_dir
                .join("models")
                .join("quality")
                .join("clipiqa_plus.onnx"),
            b"fake",
        )
        .expect("clipiqa");
        fs::write(
            install_dir.join("quality_preprocessor.json"),
            r#"{"musiq_input_width":256,"musiq_input_height":192,"clipiqa_input_width":256,"clipiqa_input_height":192}"#,
        )
        .expect("quality preprocessor");

        let manager = ModelManager::new(temp.path().join("models"));
        let caps = manager.expert_capabilities();
        assert!(caps.musiq);
        assert!(caps.clipiqa);
        assert!(!caps.quality_models);
    }

    #[test]
    fn expert_install_defaults_to_official_full_manifest_url() {
        let req = ComponentInstallRequest {
            id: "expert".to_string(),
            source_dir: None,
            manifest_path: None,
            manifest_url: None,
        };
        match InstallSource::from_request(&req).expect("default expert source") {
            InstallSource::ManifestUrl(url) => assert_eq!(url, DEFAULT_EXPERT_MANIFEST_URL),
            _ => panic!("expected official manifest url"),
        }
    }

    #[test]
    fn parses_manifest_with_utf8_bom() {
        let manifest = parse_manifest_text(
            "\u{feff}{\"id\":\"expert\",\"version\":\"onnx-v1\",\"runtime\":\"onnxruntime\"}",
        )
        .expect("manifest with bom");
        assert_eq!(manifest.id, "expert");
        assert_eq!(manifest.version, "onnx-v1");
    }

    #[tokio::test]
    async fn installs_component_from_source_dir_and_verifies_files() {
        let temp = tempfile::tempdir().expect("temp dir");
        let source = temp.path().join("source");
        let files = write_fake_full_expert_component(&source);
        let total_bytes: usize = files.iter().map(|(_, bytes)| bytes.len()).sum();
        fs::write(
            source.join(MANIFEST_FILENAME),
            fake_full_expert_manifest(&files),
        )
        .expect("manifest");

        let manager = ModelManager::new(temp.path().join("models"));
        let (status, body) = manager
            .install(ComponentInstallRequest {
                id: "expert".to_string(),
                source_dir: Some(source.to_string_lossy().to_string()),
                manifest_path: None,
                manifest_url: None,
            })
            .await
            .expect("install component");
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["ok"], true);
        let expert = manager
            .list()
            .into_iter()
            .find(|c| c.id == "expert")
            .expect("expert component");
        assert_eq!(expert.status, "installed");
        assert_eq!(
            expert.manifest.as_ref().unwrap().checksum_status.as_deref(),
            Some("verified")
        );
        assert!(Path::new(&expert.install_dir)
            .join("models")
            .join("dinov2-small.onnx")
            .exists());
        let state = expert.install_state.as_ref().expect("install state");
        assert_eq!(state.status, "done");
        assert_eq!(state.bytes_done, total_bytes as u64);
        assert_eq!(state.bytes_total, total_bytes as u64);
        assert!(manager.available_engines().contains(&"expert".to_string()));
    }

    #[test]
    fn pause_and_cancel_update_running_install_state() {
        let temp = tempfile::tempdir().expect("temp dir");
        let manager = ModelManager::new(temp.path().join("models"));
        let install_dir = manager.cache_dir.join("expert");
        fs::create_dir_all(&install_dir).expect("install dir");
        write_install_state(
            &install_dir.join(INSTALL_STATE_FILENAME),
            &InstallState::running("expert", "models/dinov2-small.onnx", 0, 1)
                .with_total_bytes(4, 10)
                .with_file_bytes(4, 10),
        )
        .expect("running state");

        let paused = manager.pause("expert").expect("pause");
        let paused_state = paused.install_state.expect("paused state");
        assert_eq!(paused_state.status, "pause_requested");
        let control = read_install_control(&install_dir.join(INSTALL_CONTROL_FILENAME));
        assert!(control.pause_requested);

        let cancelled = manager.cancel("expert").expect("cancel");
        let cancelled_state = cancelled.install_state.expect("cancel state");
        assert_eq!(cancelled_state.status, "cancel_requested");
        let control = read_install_control(&install_dir.join(INSTALL_CONTROL_FILENAME));
        assert!(control.cancel_requested);
    }

    #[tokio::test]
    async fn install_rejects_bad_checksum() {
        let temp = tempfile::tempdir().expect("temp dir");
        let source = temp.path().join("source");
        fs::create_dir_all(source.join("models")).expect("source models dir");
        fs::write(source.join("models").join("bad.onnx"), b"actual").expect("model file");
        fs::write(
            source.join(MANIFEST_FILENAME),
            r#"{
                "id":"expert",
                "version":"onnx-v1",
                "runtime":"onnxruntime",
                "models":["dinov2-small"],
                "files":[{"path":"models/bad.onnx","sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]
            }"#,
        )
        .expect("manifest");

        let manager = ModelManager::new(temp.path().join("models"));
        let err = manager
            .install(ComponentInstallRequest {
                id: "expert".to_string(),
                source_dir: Some(source.to_string_lossy().to_string()),
                manifest_path: None,
                manifest_url: None,
            })
            .await
            .expect_err("checksum mismatch");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }
}
