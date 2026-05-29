use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastQualityProfile {
    Standard,
    Advanced,
}

impl FastQualityProfile {
    pub fn from_name(name: &str) -> Self {
        match name {
            "advanced" | "aggressive" => Self::Advanced,
            _ => Self::Standard,
        }
    }

    fn thresholds(self) -> Thresholds {
        match self {
            Self::Standard => Thresholds {
                subject_sharp: 750.0,
                very_subject_sharp: 550.0,
                very_blur_combined: 0.28,
                motion_anisotropy: 0.62,
                edge_width_pix: 5.0,
                dark_mean: 22.0,
                bright_mean: 235.0,
                dead_shadow: 0.82,
                dead_highlight: 0.82,
                low_contrast: 10.0,
                low_entropy: 0.85,
                min_long_side: 640,
                min_file_size: 25_000,
                horizon_tilt_deg: 4.5,
                horizon_severe_deg: 15.0,
                score_adjust: 0.0,
                score_floor: 35.0,
                color_noise_high: 0.75,
                color_cast_high: 28.0,
                dynamic_range_low: 30.0,
            },
            Self::Advanced => Thresholds {
                subject_sharp: 1100.0,
                very_subject_sharp: 650.0,
                very_blur_combined: 0.35,
                motion_anisotropy: 0.55,
                edge_width_pix: 4.0,
                dark_mean: 28.0,
                bright_mean: 228.0,
                dead_shadow: 0.70,
                dead_highlight: 0.70,
                low_contrast: 14.0,
                low_entropy: 1.20,
                min_long_side: 900,
                min_file_size: 40_000,
                horizon_tilt_deg: 3.0,
                horizon_severe_deg: 12.0,
                score_adjust: -6.0,
                score_floor: 45.0,
                color_noise_high: 0.60,
                color_cast_high: 22.0,
                dynamic_range_low: 40.0,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Thresholds {
    subject_sharp: f64,
    very_subject_sharp: f64,
    very_blur_combined: f64,
    motion_anisotropy: f64,
    edge_width_pix: f64,
    dark_mean: f64,
    bright_mean: f64,
    dead_shadow: f64,
    dead_highlight: f64,
    low_contrast: f64,
    low_entropy: f64,
    min_long_side: u32,
    min_file_size: u64,
    horizon_tilt_deg: f64,
    horizon_severe_deg: f64,
    score_adjust: f64,
    score_floor: f64,
    color_noise_high: f64,
    color_cast_high: f64,
    dynamic_range_low: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastQualitySignals {
    pub width: u32,
    pub height: u32,
    pub file_size: u64,
    pub blur_score: f64,
    pub brightness_mean: f64,
    pub brightness_std: f64,
    pub contrast_score: f64,
    pub overexposed_ratio: f64,
    pub underexposed_ratio: f64,
    pub entropy: f64,
    pub blur_combined: f64,
    pub salient_sharpness: Option<f64>,
    pub motion_anisotropy: f64,
    pub edge_width_pix: Option<f64>,
    pub focus_ratio: Option<f64>,
    pub horizon_tilt_deg: Option<f64>,
    pub composition: Option<f64>,
    pub worst_clip_dark: f64,
    pub worst_clip_bright: f64,
    pub color_noise: Option<f64>,
    pub color_cast: Option<f64>,
    pub dynamic_range: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FastQualityResult {
    pub quality_score: f64,
    pub flags: Vec<String>,
    pub auto_reject: bool,
    pub reject_reason: Option<String>,
}

pub fn analyze_from_signals(
    signals: &FastQualitySignals,
    profile: FastQualityProfile,
) -> FastQualityResult {
    let p = profile.thresholds();
    let mut flags: Vec<String> = Vec::new();

    if signals.width.max(signals.height) < p.min_long_side {
        flags.push("too_small".to_string());
    }
    if signals.file_size > 0 && signals.file_size < p.min_file_size {
        flags.push("tiny_file".to_string());
    }

    if let Some(salient) = signals.salient_sharpness {
        if salient < p.very_subject_sharp {
            flags.push("very_blurry".to_string());
        } else if salient < p.subject_sharp {
            flags.push("subject_blurry".to_string());
        }
    }

    if !flags.iter().any(|f| f == "very_blurry") && signals.blur_combined < p.very_blur_combined {
        flags.push("very_blurry".to_string());
    }

    let is_motion = signals.motion_anisotropy > p.motion_anisotropy
        && signals.edge_width_pix.is_some_and(|v| v > p.edge_width_pix)
        && signals
            .salient_sharpness
            .is_some_and(|v| v < p.subject_sharp)
        && signals.focus_ratio.map_or(true, |v| v < 5.0);
    if is_motion {
        flags.retain(|f| f != "blurry" && f != "subject_blurry");
        if !flags.iter().any(|f| f == "very_blurry") {
            flags.push("motion_blur".to_string());
        }
    }

    if signals.brightness_mean < p.dark_mean || signals.underexposed_ratio >= p.dead_shadow {
        flags.push("underexposed".to_string());
    } else if signals.worst_clip_dark >= p.dead_shadow
        && signals.brightness_mean < p.dark_mean * 3.0
    {
        flags.push("underexposed".to_string());
    }

    if signals.brightness_mean > p.bright_mean || signals.overexposed_ratio >= p.dead_highlight {
        flags.push("overexposed".to_string());
    } else if signals.worst_clip_bright >= p.dead_highlight
        && signals.brightness_mean > p.bright_mean - 20.0
    {
        flags.push("overexposed".to_string());
    }

    if signals.contrast_score < p.low_contrast {
        flags.push("low_contrast".to_string());
    }
    if signals.entropy < p.low_entropy {
        flags.push("low_information".to_string());
    }

    if let Some(tilt) = signals.horizon_tilt_deg {
        if tilt > p.horizon_severe_deg {
            flags.push("horizon_severe".to_string());
        } else if tilt > p.horizon_tilt_deg {
            flags.push("horizon_tilt".to_string());
        }
    }

    if let Some(cn) = signals.color_noise {
        if cn > p.color_noise_high {
            flags.push("color_noise".to_string());
        }
    }
    if let Some(cc) = signals.color_cast {
        if cc > p.color_cast_high {
            flags.push("color_cast".to_string());
        }
    }
    if let Some(dr) = signals.dynamic_range {
        if dr < p.dynamic_range_low {
            flags.push("flat_tone".to_string());
        }
    }

    let quality_score = compute_score(
        signals.blur_combined,
        signals.brightness_mean,
        signals.contrast_score,
        signals.entropy,
        signals.composition,
        &flags,
        p.score_adjust,
    );
    let hard = rejecting_flags(&flags);
    let auto_reject = !hard.is_empty() || quality_score < p.score_floor;
    let reject_reason = if !hard.is_empty() {
        reason_for_fast(&flags)
    } else if auto_reject {
        Some("综合质量不达标".to_string())
    } else {
        None
    };

    FastQualityResult {
        quality_score: round3(quality_score),
        flags,
        auto_reject,
        reject_reason,
    }
}

fn compute_score(
    blur_combined: f64,
    brightness_mean: f64,
    contrast_score: f64,
    entropy: f64,
    composition: Option<f64>,
    flags: &[String],
    score_adjust: f64,
) -> f64 {
    let blur_component = blur_combined * 35.0;
    let exposure_component = (25.0 - (brightness_mean - 128.0).abs() / 128.0 * 25.0).max(0.0);
    let contrast_component = (contrast_score / 64.0 * 20.0).min(20.0);
    let entropy_component = (entropy / 7.0 * 10.0).min(10.0);
    let comp_component = composition.unwrap_or(0.5) * 10.0;
    let mut score = blur_component
        + exposure_component
        + contrast_component
        + entropy_component
        + comp_component
        + score_adjust;

    for flag in flags {
        match flag.as_str() {
            "very_blurry" | "motion_blur" | "underexposed" | "overexposed" | "low_information"
            | "horizon_severe" => score -= 22.0,
            "subject_blurry" | "low_contrast" | "horizon_tilt" => score -= 12.0,
            "color_noise" | "color_cast" | "flat_tone" => score -= 8.0,
            "too_small" | "tiny_file" => score -= 6.0,
            _ => {}
        }
    }
    score.clamp(0.0, 100.0)
}

fn rejecting_flags(flags: &[String]) -> Vec<&str> {
    flags
        .iter()
        .map(String::as_str)
        .filter(|f| {
            matches!(
                *f,
                "very_blurry"
                    | "motion_blur"
                    | "underexposed"
                    | "overexposed"
                    | "low_information"
                    | "too_small"
                    | "tiny_file"
                    | "horizon_severe"
            )
        })
        .collect()
}

fn reason_for_fast(flags: &[String]) -> Option<String> {
    for flag in [
        "motion_blur",
        "very_blurry",
        "subject_blurry",
        "horizon_severe",
        "underexposed",
        "overexposed",
        "low_information",
        "too_small",
        "tiny_file",
        "horizon_tilt",
        "low_contrast",
        "color_noise",
        "color_cast",
        "flat_tone",
    ] {
        if flags.iter().any(|f| f == flag) {
            return Some(
                match flag {
                    "motion_blur" => "运动模糊 · 手抖或模特动",
                    "very_blurry" => "主体失焦",
                    "subject_blurry" => "主体不够清晰",
                    "horizon_severe" => "歪斜严重 · 失控构图",
                    "underexposed" => "曝光严重不足",
                    "overexposed" => "高光溢出 · 细节流失",
                    "low_information" => "画面缺少内容",
                    "too_small" => "非拍摄文件（疑似截图）",
                    "tiny_file" => "文件异常小",
                    "horizon_tilt" => "地平线明显歪斜",
                    "low_contrast" => "反差不足",
                    "color_noise" => "高 ISO 彩噪明显",
                    "color_cast" => "白平衡偏色",
                    "flat_tone" => "动态范围不足 · 画面发灰",
                    _ => flag,
                }
                .to_string(),
            );
        }
    }
    None
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> FastQualitySignals {
        FastQualitySignals {
            width: 1200,
            height: 800,
            file_size: 200_000,
            blur_score: 1000.0,
            brightness_mean: 128.0,
            brightness_std: 48.0,
            contrast_score: 48.0,
            overexposed_ratio: 0.0,
            underexposed_ratio: 0.0,
            entropy: 5.5,
            blur_combined: 0.8,
            salient_sharpness: Some(1200.0),
            motion_anisotropy: 0.1,
            edge_width_pix: Some(2.0),
            focus_ratio: Some(2.0),
            horizon_tilt_deg: None,
            composition: Some(0.7),
            worst_clip_dark: 0.0,
            worst_clip_bright: 0.0,
            color_noise: Some(0.3),
            color_cast: Some(10.0),
            dynamic_range: Some(120.0),
        }
    }

    #[test]
    fn severe_blur_is_rejected() {
        let mut s = base();
        s.salient_sharpness = Some(100.0);
        s.blur_combined = 0.1;
        let q = analyze_from_signals(&s, FastQualityProfile::Standard);
        assert!(q.auto_reject);
        assert!(q.flags.contains(&"very_blurry".to_string()));
    }

    #[test]
    fn advanced_profile_is_stricter() {
        let mut s = base();
        s.salient_sharpness = Some(700.0);
        let standard = analyze_from_signals(&s, FastQualityProfile::Standard);
        let advanced = analyze_from_signals(&s, FastQualityProfile::Advanced);
        assert!(advanced.quality_score <= standard.quality_score);
        assert!(advanced.flags.contains(&"subject_blurry".to_string()));
    }
}
