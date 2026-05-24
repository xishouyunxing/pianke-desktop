use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

const MANIFEST_FILENAME: &str = "component.json";

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

    pub fn install_unavailable(
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
        let component = self.view_for(def);
        Ok((
            StatusCode::NOT_IMPLEMENTED,
            json!({
                "error": "Rust model component downloader is not implemented yet",
                "unavailable": true,
                "component": component,
                "cache_dir": self.cache_dir.to_string_lossy()
            }),
        ))
    }

    fn view_for(&self, def: ComponentDef) -> ComponentView {
        let install_dir = self.cache_dir.join(def.id);
        let manifest_path = install_dir.join(MANIFEST_FILENAME);
        let manifest = read_manifest(&manifest_path);
        let status = match &manifest {
            Some(manifest) if manifest.id == def.id && manifest.version == def.version => {
                "installed"
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
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ComponentInstallRequest {
    pub id: String,
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
    pub installed_at: Option<String>,
    #[serde(default)]
    pub checksum_status: Option<String>,
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
            "insightface-buffalo_l",
            "nima",
            "musiq",
            "clipiqa+",
        ],
        runtime: "onnxruntime",
    }]
}

fn read_manifest(path: &Path) -> Option<ComponentManifest> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
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
        fs::create_dir_all(&install_dir).expect("install dir");
        fs::write(
            install_dir.join(MANIFEST_FILENAME),
            r#"{"id":"expert","version":"onnx-v1","runtime":"onnxruntime","models":["dinov2-small"]}"#,
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
    }
}
