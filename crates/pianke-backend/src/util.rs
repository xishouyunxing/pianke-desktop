use axum::{
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(test)]
use crate::{HEIC_EXTS, IMAGE_EXTS, RAW_EXTS, SIDECAR_EXTS};

pub(crate) fn file_response(path: PathBuf, content_type: Option<&str>) -> Response {
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

pub(crate) fn binary_response(
    bytes: Vec<u8>,
    content_type: &str,
    extra: Option<(&str, &str)>,
) -> Response {
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

pub(crate) fn broken_placeholder() -> Response {
    let svg = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 480 360'><rect width='100%' height='100%' fill='#efeadd'/><text x='50%' y='52%' text-anchor='middle' font-family='sans-serif' font-size='22' fill='#6b6256'>无法读取</text></svg>";
    binary_response(
        svg.as_bytes().to_vec(),
        "image/svg+xml",
        Some(("X-Image-Status", "failed")),
    )
}

pub(crate) fn json_error(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({"error": msg}))).into_response()
}

pub(crate) fn safe_join(base: &Path, rel: &str) -> Option<PathBuf> {
    let mut out = base.to_path_buf();
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(p) => out.push(p),
            _ => return None,
        }
    }
    Some(out)
}

pub(crate) fn content_type_for(path: &Path) -> &'static str {
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

pub(crate) fn ext_lower(path: &Path) -> String {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| format!(".{}", s.to_lowercase()))
        .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn format_kind_for_path(path: &Path) -> &'static str {
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

pub(crate) fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

pub(crate) fn now_secs() -> f64 {
    system_time_secs(SystemTime::now()).unwrap_or(0.0)
}

pub(crate) fn system_time_secs(t: SystemTime) -> Option<f64> {
    let d = t.duration_since(UNIX_EPOCH).ok()?;
    Some(d.as_secs_f64())
}

pub(crate) fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

pub(crate) fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

pub(crate) fn round5(v: f64) -> f64 {
    (v * 100000.0).round() / 100000.0
}

pub(crate) fn pillow_luma(rgb: &image::RgbImage) -> image::GrayImage {
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

pub(crate) fn format_bytes(n: u64) -> String {
    if n < 1024 {
        format!("{n} B")
    } else if n < 1024 * 1024 {
        format!("{:.0} KB", n as f64 / 1024.0)
    } else {
        format!("{:.1} MB", n as f64 / 1024.0 / 1024.0)
    }
}

pub(crate) fn whash_image_scale(width: u32, height: u32) -> u32 {
    let natural = previous_power_of_two(width.min(height));
    natural.max(8)
}

pub(crate) fn previous_power_of_two(value: u32) -> u32 {
    if value <= 1 {
        return 1;
    }
    1 << (31 - value.leading_zeros())
}

pub(crate) fn compare_versions(remote: &str, current: &str) -> std::cmp::Ordering {
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

pub(crate) fn format_capabilities() -> Value {
    json!({
        "raw_thumbnail": true,
        "raw_strategy": "embedded_jpeg",
        "heic": heic_decode_available(),
        "heic_strategy": heic_decode_strategy(),
        "opencv_orb": opencv_orb_available(),
    })
}

pub(crate) fn opencv_orb_available() -> bool {
    cfg!(feature = "opencv-orb")
}

pub(crate) fn heic_decode_available() -> bool {
    heic_runtime_available()
}

pub(crate) fn heic_decode_strategy() -> &'static str {
    if heic_decode_available() {
        "windows_wic"
    } else {
        "unavailable"
    }
}

#[cfg(windows)]
fn heic_runtime_available() -> bool {
    crate::job::windows_heif_decoder_available().unwrap_or(false)
}

#[cfg(not(windows))]
fn heic_runtime_available() -> bool {
    false
}
