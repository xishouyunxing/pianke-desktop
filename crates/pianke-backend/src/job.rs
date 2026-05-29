use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use chrono::{DateTime, Local};
use image::{DynamicImage, GenericImageView, ImageFormat};
use pianke_core::fast::{
    analyze_from_signals, average_hash_from_luma, cluster_with_options, difference_hash_from_luma,
    perceptual_hash_from_luma, wavelet_hash_from_luma, ExifSummary, FastClusterOptions,
    FastImageInfo, FastQualityProfile, QualityInfo,
};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;

use crate::clustering::expert_cluster;
use crate::expert_vision;
use crate::llm_provider::JudgeVerdict;
use crate::quality::{compute_color_hist, quality_signals};
use crate::session::*;
use crate::types::*;
use crate::util::*;
use crate::watermark::WatermarkConfig;
use crate::{ANALYSIS_MAX_SIDE, HEIC_EXTS, IMAGE_EXTS, PIC_DIR, RAW_EXTS, SIDECAR_EXTS};

pub(crate) fn max_group_size_from_threshold_far(threshold_far: i32) -> usize {
    (threshold_far.clamp(3, 12) as usize * 2 + 8).max(8).min(60)
}

pub(crate) fn run_job(ctx: AppCtx, req: StartRequest) {
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
    let cached = load_info_cache(&req.folder).unwrap_or_default();
    let (mut infos, pairs) = if cached.is_empty() {
        (Vec::new(), pairs)
    } else {
        filter_cached_infos(&pairs, &cached)
    };
    let reused_count = infos.len();
    {
        let mut state = ctx.inner.lock().expect("backend state lock");
        state.job.total = pairs.len() + reused_count;
        state.job.done = reused_count;
        state.job.status = "hashing".to_string();
        state.job.label = if reused_count > 0 {
            format!(
                "增量扫描：复用 {reused_count} 张，处理 {} 张新图片…",
                pairs.len()
            )
        } else {
            "读取图片与计算 Fast 特征…".to_string()
        };
    }

    let mut skipped = Vec::new();

    if req.engine == "fast" {
        // Fast mode: parallelize process_one across rayon thread pool
        let cancel = AtomicBool::new(false);
        let done_count = AtomicUsize::new(0);
        let skipped_count = AtomicUsize::new(0);
        let strength = req.prescreen_strength.clone();
        let ctx_clone = ctx.clone();

        let mut results: Vec<(usize, Result<InfoRecord, SkippedItem>)> = pairs
            .into_par_iter()
            .enumerate()
            .map(|(idx, pair)| {
                if cancel.load(Ordering::Relaxed) {
                    let item = SkippedItem {
                        path: pair.primary.to_string_lossy().to_string(),
                        reason: "cancelled".to_string(),
                    };
                    return (idx, Err(item));
                }
                match process_one(&pair, &strength) {
                    Ok(record) => {
                        let done = reused_count + done_count.fetch_add(1, Ordering::Relaxed) + 1;
                        let mut state = ctx_clone.inner.lock().expect("backend state lock");
                        state.job.done = done;
                        if state.job.cancel_requested {
                            cancel.store(true, Ordering::Relaxed);
                        }
                        (idx, Ok(record))
                    }
                    Err(reason) => {
                        let item = SkippedItem {
                            path: pair.primary.to_string_lossy().to_string(),
                            reason,
                        };
                        done_count.fetch_add(1, Ordering::Relaxed);
                        skipped_count.fetch_add(1, Ordering::Relaxed);
                        (idx, Err(item))
                    }
                }
            })
            .collect();

        if cancel.load(Ordering::Relaxed) {
            let mut state = ctx.inner.lock().expect("backend state lock");
            state.job.status = "cancelled".to_string();
            state.job.finished_at = now_secs();
            return;
        }

        results.sort_by_key(|(idx, _)| *idx);
        for (_, result) in results {
            match result {
                Ok(record) => {
                    push_job_event(&ctx, &record, None);
                    infos.push(record);
                }
                Err(item) => {
                    push_skip_event(&ctx, &item);
                    skipped.push(item);
                }
            }
        }
        let mut state = ctx.inner.lock().expect("backend state lock");
        state.job.done = reused_count + done_count.load(Ordering::Relaxed);
        state.job.skipped = skipped.clone();
    } else {
        // Expert/Tycoon mode: sequential (model state not Send+Sync)
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
                            state.job.error = Some(
                                "Expert 模式需要可解码的 JPG/PNG/WebP/TIFF companion".to_string(),
                            );
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
                        apply_expert_quality_availability(
                            &mut record,
                            quality_model.is_some(),
                            None,
                        );
                        if let Some(model) = quality_model.as_mut() {
                            let Some(analysis) = pair.analysis.as_ref() else {
                                let mut quality = record.info.quality.clone().unwrap_or_default();
                                quality.extra.insert(
                                    "quality_models_unavailable".to_string(),
                                    json!(
                                        "Expert 质量模型需要可解码的 JPG/PNG/WebP/TIFF companion"
                                    ),
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
                                    apply_expert_quality_availability(
                                        &mut record,
                                        false,
                                        Some(reason),
                                    );
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
            state.job.done = reused_count + idx + 1;
            state.job.skipped = skipped.clone();
        }
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
        let mut options = FastClusterOptions::default();
        options.time_halflife = (req.near_seconds as f64 / 5.0).max(5.0);
        options.threshold = (0.80 - req.threshold_near as f64 * 0.04)
            .max(0.10)
            .min(0.75);
        options.max_group_size = max_group_size_from_threshold_far(req.threshold_far);
        if !orb_inliers.is_empty() {
            options.orb_inliers = orb_inliers;
        }
        cluster_with_options(&fast_infos, &options)
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
    let _ = save_info_cache(&req.folder, &infos);
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
    state.job.done = state.job.total;
    state.job.label = "完成".to_string();
    state.job.finished_at = now_secs();
}

pub(crate) fn scan_folder(folder: &Path) -> Vec<ScanPair> {
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

pub(crate) fn run_tycoon_judge(
    ctx: &AppCtx,
    config: &crate::llm_provider::ProviderConfig,
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

pub(crate) fn apply_llm_verdict(record: &mut InfoRecord, verdict: &JudgeVerdict) {
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

pub(crate) fn apply_expert_quality_availability(
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

pub(crate) fn apply_expert_quality_models(
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

pub(crate) fn apply_expert_quality_scores(
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

pub(crate) fn process_one(pair: &ScanPair, strength: &str) -> Result<InfoRecord, String> {
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

pub(crate) fn load_fast_analysis_image(path: &Path) -> Result<DynamicImage, String> {
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

pub(crate) fn extract_embedded_jpeg_preview(bytes: &[u8]) -> Result<&[u8], RawPreviewError> {
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
pub(crate) fn windows_heif_decoder_available() -> Result<bool, windows::core::Error> {
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

#[cfg(feature = "opencv-orb")]
pub(crate) fn compute_orb_inliers_for_records(
    records: &[InfoRecord],
) -> HashMap<(usize, usize), usize> {
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
pub(crate) fn compute_orb_inliers_for_records(
    _records: &[InfoRecord],
) -> HashMap<(usize, usize), usize> {
    HashMap::new()
}

pub(crate) fn image_to_data_url(path: &Path) -> Result<(String, usize, (u32, u32)), String> {
    let img = image::open(path).map_err(|e| format!("load image failed: {e}"))?;
    let (w, h) = img.dimensions();
    let max_side = 896u32;
    let resized = if w.max(h) > max_side {
        let scale = max_side as f32 / w.max(h) as f32;
        img.resize(
            (w as f32 * scale).round().max(1.0) as u32,
            (h as f32 * scale).round().max(1.0) as u32,
            image::imageops::FilterType::Lanczos3,
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
    let header = "你是一位专业的照片质量评审员。请审查照片并返回一个 JSON 对象，包含以下键：verdict、reason、flaws、fixable。\n- verdict: \"pass\" 或 \"reject\"\n- reason: 简短具体的判定理由\n- flaws: 列出所有缺陷（若无缺陷为空字符串）\n- fixable: 仅在 verdict 为 \"pass\" 时有意义，说明后期是否可修复\n";
    let criteria = match strength {
        "advanced" | "aggressive" => concat!(
            "【进阶模式 · 严格评审】请以高标准审查，以下任一情况均应 reject：\n",
            "1. 主体模糊、对焦偏移、运动拖影\n",
            "2. 曝光偏差（过曝死白、欠曝丢失暗部细节）\n",
            "3. 构图严重失衡、水平线明显歪斜、主体被裁切\n",
            "4. 人物表情不自然、闭眼、尴尬瞬间\n",
            "5. 背景杂乱干扰主体、出现路人/杂物\n",
            "6. 色彩异常（白平衡偏色、高 ISO 彩噪明显）\n",
            "7. 整体美学水准偏低，不具备保留价值\n",
            "只有在所有方面都表现良好时才判定为 pass。",
        ),
        _ => concat!(
            "【标准模式 · 宽松评审】仅在存在明显硬伤时 reject：\n",
            "1. 严重模糊（主体完全不可辨认）\n",
            "2. 严重过曝或欠曝（大面积死白/死黑）\n",
            "3. 明显的构图失误（水平线大幅歪斜、主体被截断）\n",
            "4. 人物明显闭眼\n",
            "轻微的对焦不实、曝光偏差、构图不够完美等不应 reject。",
        ),
    };
    format!("{header}\n{criteria}")
}

pub(crate) fn push_job_event(ctx: &AppCtx, record: &InfoRecord, reason: Option<String>) {
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

pub(crate) fn push_skip_event(ctx: &AppCtx, skipped: &SkippedItem) {
    let mut state = ctx.inner.lock().expect("backend state lock");
    let engine = state.job.engine.clone();
    state.job.event_seq += 1;
    let event = JobEvent {
        seq: state.job.event_seq,
        name: file_name(&skipped.path),
        path: skipped.path.clone(),
        ok: false,
        reject: false,
        reason: Some(skipped.reason.clone()),
        verdict: "skip".to_string(),
        engine,
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

pub(crate) fn meta_entry(record: &InfoRecord) -> Value {
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

pub(crate) fn run_watermark_job(
    ctx: AppCtx,
    winners: Vec<String>,
    out_dir: PathBuf,
    cfg: WatermarkConfig,
) {
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
    let result = crate::watermark::batch_export(
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

#[cfg(test)]
mod tests {
    use super::max_group_size_from_threshold_far;

    #[test]
    fn threshold_far_is_clamped_before_usize_conversion() {
        assert_eq!(max_group_size_from_threshold_far(-1), 14);
        assert_eq!(max_group_size_from_threshold_far(2), 14);
        assert_eq!(max_group_size_from_threshold_far(6), 20);
        assert_eq!(max_group_size_from_threshold_far(12), 32);
        assert_eq!(max_group_size_from_threshold_far(99), 32);
    }
}
