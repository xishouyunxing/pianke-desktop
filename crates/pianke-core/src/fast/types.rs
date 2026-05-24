use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExifSummary {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub file_size: Option<u64>,
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub aperture: Option<String>,
    pub shutter: Option<String>,
    pub iso: Option<String>,
    pub focal_length: Option<String>,
    pub datetime: Option<String>,
    pub gps_lat: Option<f64>,
    pub gps_lon: Option<f64>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityInfo {
    pub blur_score: Option<f64>,
    pub brightness_mean: Option<f64>,
    pub brightness_std: Option<f64>,
    pub contrast_score: Option<f64>,
    pub overexposed_ratio: Option<f64>,
    pub underexposed_ratio: Option<f64>,
    pub entropy: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub file_size: Option<u64>,
    pub quality_score: Option<f64>,
    #[serde(default)]
    pub flags: Vec<String>,
    pub auto_reject: Option<bool>,
    pub reject_reason: Option<String>,
    pub salient_sharpness: Option<f64>,
    pub blur_combined: Option<f64>,
    pub motion_anisotropy: Option<f64>,
    pub edge_width_pix: Option<f64>,
    pub focus_ratio: Option<f64>,
    pub horizon_tilt_deg: Option<f64>,
    pub composition: Option<f64>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FastImageInfo {
    pub path: String,
    pub phash: Option<String>,
    pub dhash: Option<String>,
    pub whash: Option<String>,
    pub ahash: Option<String>,
    pub timestamp: Option<f64>,
    pub size: Option<u64>,
    pub mtime: Option<f64>,
    pub exif_summary: Option<ExifSummary>,
    pub quality: Option<QualityInfo>,
    pub color_hist: Option<Vec<f32>>,
    pub orb_descs_len: Option<usize>,
    pub orb_kps_len: Option<usize>,
    #[serde(default)]
    pub dinov2: Option<Vec<f32>>,
    #[serde(default)]
    pub face_embeddings: Vec<Vec<f32>>,
}
