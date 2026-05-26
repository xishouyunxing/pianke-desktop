use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const MANIFEST_FILENAME: &str = "component.json";
const INSTALL_STATE_FILENAME: &str = "install_state.json";
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
            quality_models: musiq && clipiqa,
            nima_legacy: false,
            nima_legacy_unavailable: true,
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
        if let Err(err) = fs::create_dir_all(&install_dir) {
            return Err(server_error(format!("create component dir failed: {err}")));
        }
        write_install_state(
            &state_path,
            &InstallState::running(def.id, "reading_manifest", 0, 1),
        )
        .map_err(server_error)?;

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

        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir).map_err(|e| {
                server_error(format!("remove stale temp component dir failed: {e}"))
            })?;
        }
        fs::create_dir_all(&temp_dir)
            .map_err(|e| server_error(format!("create temp component dir failed: {e}")))?;

        let total = manifest.files.len().max(1);
        for (idx, file) in manifest.files.iter().enumerate() {
            let rel = safe_relative_path(&file.path)
                .ok_or_else(|| bad_request(format!("unsafe model file path: {}", file.path)))?;
            let target = temp_dir.join(&rel);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| server_error(format!("create model subdir failed: {e}")))?;
            }
            let label = rel.to_string_lossy().to_string();
            write_install_state(
                &state_path,
                &InstallState::running(def.id, &label, idx, total),
            )
            .map_err(server_error)?;
            let bytes = load_component_file(file, loaded.base_dir.as_deref())
                .await
                .map_err(server_error)?;
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
            fs::write(&target, bytes)
                .map_err(|e| server_error(format!("write model file failed: {e}")))?;
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
            &InstallState::done(def.id, total),
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
            error: None,
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
        "expert" => install_dir
            .join("models")
            .join("dinov2-small.onnx")
            .exists(),
        _ => true,
    }
}

fn read_manifest(path: &Path) -> Option<ComponentManifest> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
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
                serde_json::from_str(&text).map_err(|e| format!("parse manifest failed: {e}"))?;
            Ok(LoadedManifest {
                manifest,
                base_dir: None,
            })
        }
    }
}

fn read_manifest_required(path: &Path) -> Result<ComponentManifest, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read manifest failed: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse manifest failed: {e}"))
}

async fn load_component_file(
    file: &ComponentFile,
    base_dir: Option<&Path>,
) -> Result<Vec<u8>, String> {
    if let Some(url) = &file.url {
        if url.starts_with("file://") {
            let path = file_url_to_path(url);
            return fs::read(path).map_err(|e| format!("read component file failed: {e}"));
        }
        return reqwest::get(url)
            .await
            .map_err(|e| format!("download component file failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("download component file failed: {e}"))?
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| format!("read component file body failed: {e}"));
    }
    let Some(base_dir) = base_dir else {
        return Err(format!("component file {} has no url", file.path));
    };
    let rel = safe_relative_path(&file.path)
        .ok_or_else(|| format!("unsafe model file path: {}", file.path))?;
    fs::read(base_dir.join(rel)).map_err(|e| format!("read component file failed: {e}"))
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
        assert_eq!(expert.status, "installed");
        assert!(!expert.download_required);
        let caps = manager.expert_capabilities();
        assert!(caps.dinov2);
        assert!(!caps.quality_models);
        assert!(!caps.nima_legacy);
        assert!(caps.nima_legacy_unavailable);
    }

    #[test]
    fn optional_quality_models_update_expert_capabilities() {
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
        assert!(caps.quality_models);
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

    #[tokio::test]
    async fn installs_component_from_source_dir_and_verifies_files() {
        let temp = tempfile::tempdir().expect("temp dir");
        let source = temp.path().join("source");
        fs::create_dir_all(source.join("models")).expect("source models dir");
        let model_bytes = b"tiny fake onnx model";
        fs::write(source.join("models").join("dinov2-small.onnx"), model_bytes)
            .expect("model file");
        fs::write(
            source.join(MANIFEST_FILENAME),
            format!(
                r#"{{
                    "id":"expert",
                    "version":"onnx-v1",
                    "runtime":"onnxruntime",
                    "models":["dinov2-small"],
                    "files":[{{"path":"models/dinov2-small.onnx","sha256":"{}","size_bytes":{}}}]
                }}"#,
                sha256_hex(model_bytes),
                model_bytes.len()
            ),
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
        assert!(manager.available_engines().contains(&"expert".to_string()));
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
