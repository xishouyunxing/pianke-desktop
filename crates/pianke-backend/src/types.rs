use chrono::NaiveDateTime;
use pianke_core::fast::FastImageInfo;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
};
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::llm_provider::LlmProviderManager;
use crate::model_components::ModelManager;

pub struct ServerOptions {
    pub port: u16,
    pub token: Option<String>,
    pub backend_dir: PathBuf,
    pub folder_picker: Option<Arc<dyn Fn() -> Result<Option<PathBuf>, String> + Send + Sync>>,
}

#[derive(Debug)]
pub struct ServerHandle {
    pub(crate) shutdown: Option<oneshot::Sender<()>>,
    pub(crate) thread: Option<thread::JoinHandle<()>>,
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
pub(crate) struct AppCtx {
    pub(crate) inner: Arc<Mutex<AppState>>,
    pub(crate) token: Option<String>,
    pub(crate) backend_dir: PathBuf,
    pub(crate) models: ModelManager,
    pub(crate) llm: LlmProviderManager,
    pub(crate) folder_picker:
        Option<Arc<dyn Fn() -> Result<Option<PathBuf>, String> + Send + Sync>>,
}

#[derive(Debug, Default)]
pub(crate) struct AppState {
    pub(crate) session: Option<SessionState>,
    pub(crate) job: JobState,
    pub(crate) last_infos: Vec<InfoRecord>,
    pub(crate) grouping: GroupingState,
    pub(crate) watermark: WatermarkState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GroupState {
    pub(crate) images: Vec<String>,
    #[serde(default)]
    pub(crate) pending: Vec<String>,
    pub(crate) left: Option<String>,
    pub(crate) right: Option<String>,
    #[serde(default)]
    pub(crate) losers: Vec<String>,
    pub(crate) winner: Option<String>,
    #[serde(default)]
    pub(crate) extra_winners: Vec<String>,
    #[serde(default)]
    pub(crate) finished: bool,
    #[serde(default)]
    pub(crate) applied: bool,
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) move_log: Vec<Value>,
    #[serde(default)]
    pub(crate) auto_rejected: Vec<String>,
    #[serde(default)]
    pub(crate) auto_reject_reasons: HashMap<String, String>,
    #[serde(default)]
    pub(crate) auto_selected: bool,
    #[serde(default)]
    pub(crate) manual_restored: Vec<String>,
}

impl GroupState {
    pub(crate) fn new(images: Vec<String>) -> Self {
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
pub(crate) struct SessionState {
    pub(crate) folder: String,
    pub(crate) dry_run: bool,
    pub(crate) mode: String,
    pub(crate) engine: String,
    pub(crate) groups: Vec<GroupState>,
    pub(crate) current_group: usize,
    pub(crate) threshold_near: i32,
    pub(crate) threshold_far: i32,
    pub(crate) near_seconds: i32,
    pub(crate) prescreen_enabled: bool,
    pub(crate) prescreen_strength: String,
    pub(crate) prescreen_reviewed: bool,
    #[serde(default)]
    pub(crate) prescreen_rejected: Vec<String>,
    #[serde(default)]
    pub(crate) prescreen_reject_reasons: HashMap<String, String>,
    #[serde(default)]
    pub(crate) prescreen_restored: Vec<String>,
    #[serde(default)]
    pub(crate) undo_stack: Vec<UndoEntry>,
    #[serde(default)]
    pub(crate) meta: HashMap<String, Value>,
    #[serde(default)]
    pub(crate) companions: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub(crate) selection_started: bool,
    #[serde(default)]
    pub(crate) skipped: Vec<SkippedItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct UndoEntry {
    pub(crate) group_index: usize,
    pub(crate) group: GroupState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SkippedItem {
    pub(crate) path: String,
    pub(crate) reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct InfoRecord {
    pub(crate) info: FastImageInfo,
    pub(crate) companions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct JobEvent {
    pub(crate) seq: u64,
    pub(crate) name: String,
    pub(crate) path: String,
    pub(crate) ok: bool,
    pub(crate) reject: bool,
    pub(crate) reason: Option<String>,
    pub(crate) verdict: String,
    pub(crate) engine: String,
    pub(crate) shutter: Option<String>,
    pub(crate) aperture: Option<String>,
    pub(crate) iso: Option<String>,
    pub(crate) signals: Vec<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct JobState {
    pub(crate) folder: String,
    pub(crate) dry_run: bool,
    pub(crate) mode: String,
    pub(crate) engine: String,
    pub(crate) status: String,
    pub(crate) done: usize,
    pub(crate) total: usize,
    pub(crate) label: String,
    pub(crate) error: Option<String>,
    pub(crate) skipped: Vec<SkippedItem>,
    pub(crate) started_at: f64,
    pub(crate) finished_at: f64,
    pub(crate) cancel_requested: bool,
    pub(crate) threshold_near: i32,
    pub(crate) threshold_far: i32,
    pub(crate) near_seconds: i32,
    pub(crate) prescreen_enabled: bool,
    pub(crate) prescreen_strength: String,
    pub(crate) recent_events: Vec<JobEvent>,
    pub(crate) event_seq: u64,
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
pub(crate) struct GroupingState {
    pub(crate) status: String,
    pub(crate) groups: Vec<Value>,
    pub(crate) all_paths: Vec<String>,
    pub(crate) total: usize,
    pub(crate) multi: usize,
    pub(crate) error: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub(crate) struct WatermarkState {
    pub(crate) status: String,
    pub(crate) done: usize,
    pub(crate) total: usize,
    pub(crate) current: String,
    pub(crate) out_dir: Option<String>,
    pub(crate) ok: usize,
    pub(crate) failed: Vec<(String, String)>,
    pub(crate) error: Option<String>,
    pub(crate) started_at: f64,
    pub(crate) finished_at: f64,
    pub(crate) cancel_requested: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StartRequest {
    pub(crate) folder: String,
    #[serde(default)]
    pub(crate) dry_run: bool,
    #[serde(default = "default_copy")]
    pub(crate) mode: String,
    #[serde(default = "default_fast")]
    pub(crate) engine: String,
    #[serde(default = "default_true")]
    pub(crate) prescreen_enabled: bool,
    #[serde(default = "default_standard")]
    pub(crate) prescreen_strength: String,
    #[serde(default)]
    pub(crate) llm_model: Option<String>,
    #[serde(default = "default_threshold_near")]
    pub(crate) threshold_near: i32,
    #[serde(default = "default_threshold_far")]
    pub(crate) threshold_far: i32,
    #[serde(default = "default_near_seconds")]
    pub(crate) near_seconds: i32,
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

#[derive(Debug, Deserialize)]
pub(crate) struct AppUpdateManifest {
    pub(crate) version: String,
    pub(crate) url: String,
    #[serde(default)]
    pub(crate) notes: String,
    #[serde(default)]
    pub(crate) published_at: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ScanPair {
    pub(crate) primary: PathBuf,
    pub(crate) companions: Vec<PathBuf>,
    pub(crate) analysis: Option<PathBuf>,
}

#[derive(Debug)]
pub(crate) enum RawPreviewError {
    NoEmbeddedJpeg,
    DecodeFailed(String),
}

impl RawPreviewError {
    pub(crate) fn to_message(&self) -> String {
        match self {
            RawPreviewError::NoEmbeddedJpeg => "纯 RAW 暂未找到内嵌 JPEG 预览图".to_string(),
            RawPreviewError::DecodeFailed(e) => {
                format!("RAW 内嵌 JPEG 预览图解码失败: {e}")
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct SinceQuery {
    pub(crate) since: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChooseRequest {
    pub(crate) loser: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct KickRequest {
    pub(crate) side: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReopenRequest {
    pub(crate) group_id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RestoreRequest {
    pub(crate) group_id: String,
    pub(crate) path: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RegroupRequest {
    pub(crate) threshold_near: Option<i32>,
    pub(crate) threshold_far: Option<i32>,
    pub(crate) near_seconds: Option<i32>,
}

#[derive(Debug)]
pub(crate) struct TransferResult {
    pub(crate) main_target: String,
    pub(crate) companion_pairs: Vec<(String, String)>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct JobLogQuery {
    pub(crate) name: Option<String>,
}

pub(crate) trait IsoFormat {
    fn isoformat(&self) -> String;
}

impl IsoFormat for NaiveDateTime {
    fn isoformat(&self) -> String {
        self.format("%Y-%m-%dT%H:%M:%S").to_string()
    }
}
