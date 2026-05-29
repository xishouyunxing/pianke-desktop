use axum::{
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use crate::types::*;
use crate::util::file_name;
use crate::{INFO_CACHE_FILENAME, STATE_FILENAME};

pub(crate) fn advance(group: &mut GroupState, loser_side: &str) {
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

pub(crate) fn kick_side(group: &mut GroupState, side: &str) -> bool {
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

pub(crate) fn reset_group_live(group: &mut GroupState, live: Vec<String>) {
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

pub(crate) fn advance_current_index(session: &mut SessionState) {
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

pub(crate) fn mutate_current_group<F>(ctx: AppCtx, f: F) -> Response
where
    F: FnOnce(&mut SessionState, usize),
{
    // Phase 1: lock → mutate state + compute transfer plan
    let (response, transfer_plan) = {
        let mut state = ctx.inner.lock().expect("backend state lock");
        let Some(session) = state.session.as_mut() else {
            return crate::util::json_error(axum::http::StatusCode::BAD_REQUEST, "没有会话");
        };
        advance_current_index(session);
        if session.current_group >= session.groups.len() {
            return Json(json!({"done": true})).into_response();
        }
        let idx = session.current_group;
        f(session, idx);
        let plan = if session.groups[idx].finished {
            plan_group_apply(session, idx).ok()
        } else {
            None
        };
        advance_current_index(session);
        let _ = save_state(session);
        (current_group_payload(session), plan)
    };
    // Phase 2: execute transfer plan without lock
    if let Some(ref plan) = transfer_plan {
        execute_transfer_plan(plan, &{
            let state = ctx.inner.lock().expect("backend state lock");
            state
                .session
                .as_ref()
                .map(|s| s.mode.clone())
                .unwrap_or_default()
        });
    }
    // Phase 3: lock → commit transfer
    if let Some(plan) = transfer_plan {
        let mut state = ctx.inner.lock().expect("backend state lock");
        if let Some(session) = state.session.as_mut() {
            commit_group_apply(session, &plan);
            let _ = save_state(session);
        }
    }
    response
}

pub(crate) fn current_group_payload(session: &mut SessionState) -> Response {
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

pub(crate) fn serialize_group(session: &SessionState, idx: usize) -> Value {
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

pub(crate) struct TransferPlan {
    pub mkdirs: Vec<PathBuf>,
    pub transfers: Vec<PlannedTransfer>,
    pub group_index: usize,
}

pub(crate) struct PlannedTransfer {
    pub src: String,
    pub target: PathBuf,
    pub companions: Vec<(String, PathBuf)>,
    pub kind: String,
}

fn plan_group_apply(session: &SessionState, idx: usize) -> Result<TransferPlan, String> {
    if idx >= session.groups.len() || session.groups[idx].applied || !session.groups[idx].finished {
        return Err("group not ready".to_string());
    }
    if session.dry_run {
        return Ok(TransferPlan {
            mkdirs: Vec::new(),
            transfers: Vec::new(),
            group_index: idx,
        });
    }
    let folder = PathBuf::from(&session.folder);
    let win_dir = folder.join("winners");
    let lose_dir = folder.join("losers");
    let mode = &session.mode;
    let group = &session.groups[idx];
    let mut transfers = Vec::new();

    if let Some(ref winner) = group.winner {
        let main_target = unique_target(
            &win_dir,
            Path::new(winner)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("image"),
        );
        let comps = plan_companion_targets(
            winner,
            &main_target,
            &win_dir,
            session.companions.get(winner),
            mode,
        );
        transfers.push(PlannedTransfer {
            src: winner.clone(),
            target: main_target,
            companions: comps,
            kind: "winner".to_string(),
        });
    }
    for extra in &group.extra_winners {
        let main_target = unique_target(
            &win_dir,
            Path::new(extra)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("image"),
        );
        let comps = plan_companion_targets(
            extra,
            &main_target,
            &win_dir,
            session.companions.get(extra),
            mode,
        );
        transfers.push(PlannedTransfer {
            src: extra.clone(),
            target: main_target,
            companions: comps,
            kind: "winner".to_string(),
        });
    }
    for loser in &group.losers {
        let main_target = unique_target(
            &lose_dir,
            Path::new(loser)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("image"),
        );
        let comps = plan_companion_targets(
            loser,
            &main_target,
            &lose_dir,
            session.companions.get(loser),
            mode,
        );
        transfers.push(PlannedTransfer {
            src: loser.clone(),
            target: main_target,
            companions: comps,
            kind: "loser".to_string(),
        });
    }
    Ok(TransferPlan {
        mkdirs: vec![win_dir, lose_dir],
        transfers,
        group_index: idx,
    })
}

fn plan_companion_targets(
    _main_src: &str,
    main_target: &Path,
    target_dir: &Path,
    companions: Option<&Vec<String>>,
    _mode: &str,
) -> Vec<(String, PathBuf)> {
    let stem = main_target
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image")
        .to_string();
    let mut result = Vec::new();
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
        result.push((comp.clone(), target));
    }
    result
}

pub(crate) fn execute_transfer_plan(plan: &TransferPlan, mode: &str) {
    for dir in &plan.mkdirs {
        let _ = fs::create_dir_all(dir);
    }
    for t in &plan.transfers {
        let _ = transfer_one(&t.src, &t.target, mode);
        for (comp_src, comp_target) in &t.companions {
            let _ = transfer_one(comp_src, comp_target, mode);
        }
    }
}

pub(crate) fn commit_group_apply(session: &mut SessionState, plan: &TransferPlan) {
    let idx = plan.group_index;
    if idx >= session.groups.len() || session.groups[idx].applied {
        return;
    }
    if session.dry_run {
        session.groups[idx].applied = true;
        return;
    }
    let mode = session.mode.clone();
    let group = &mut session.groups[idx];
    for t in &plan.transfers {
        let result = TransferResult {
            main_target: t.target.to_string_lossy().to_string(),
            companion_pairs: t
                .companions
                .iter()
                .map(|(s, d)| (s.clone(), d.to_string_lossy().to_string()))
                .collect(),
        };
        record_transfer(group, &t.src, &t.kind, &result);
        if mode == "move" {
            match t.kind.as_str() {
                "winner" => {
                    if group.winner.as_deref() == Some(&t.src) {
                        group.winner = Some(result.main_target.clone());
                    } else if let Some(pos) = group.extra_winners.iter().position(|w| w == &t.src) {
                        group.extra_winners[pos] = result.main_target.clone();
                    }
                }
                "loser" => {
                    if let Some(pos) = group.losers.iter().position(|l| l == &t.src) {
                        group.losers[pos] = result.main_target.clone();
                    }
                }
                _ => {}
            }
        }
    }
    group.applied = true;
}

pub(crate) struct RevertPlan {
    pub entries: Vec<RevertEntry>,
    pub group_index: usize,
    pub move_log: Vec<Value>,
}

pub(crate) struct RevertEntry {
    pub dst: PathBuf,
    pub restore_target: PathBuf,
    pub mode_is_copy: bool,
}

pub(crate) fn plan_group_revert(session: &SessionState, idx: usize) -> RevertPlan {
    let mut entries = Vec::new();
    let move_log = if idx < session.groups.len() {
        session.groups[idx].move_log.clone()
    } else {
        Vec::new()
    };
    if idx >= session.groups.len() || session.groups[idx].move_log.is_empty() {
        return RevertPlan {
            entries,
            group_index: idx,
            move_log,
        };
    }
    let mode = &session.mode;
    let root = PathBuf::from(&session.folder);
    for log_entry in &session.groups[idx].move_log {
        let src = log_entry.get("src").and_then(|v| v.as_str()).unwrap_or("");
        let dst = log_entry.get("dst").and_then(|v| v.as_str()).unwrap_or("");
        if dst.is_empty() {
            continue;
        }
        let dst_path = PathBuf::from(dst);
        let restore_target = if mode == "copy" {
            dst_path.clone()
        } else if src.is_empty() {
            root.join(
                dst_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("image"),
            )
        } else {
            let mut rt = PathBuf::from(src);
            if rt.exists() {
                let name = rt
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("image")
                    .to_string();
                rt = unique_target(rt.parent().unwrap_or(&root), &name);
            }
            rt
        };
        entries.push(RevertEntry {
            dst: dst_path,
            restore_target,
            mode_is_copy: mode == "copy",
        });
    }
    RevertPlan {
        entries,
        group_index: idx,
        move_log,
    }
}

pub(crate) fn execute_revert_plan(plan: &RevertPlan) -> Vec<Value> {
    let mut failed = Vec::new();
    for entry in &plan.entries {
        if !entry.dst.exists() {
            failed.push(json!({"path": entry.dst.to_string_lossy(), "reason": "target missing"}));
            continue;
        }
        if entry.mode_is_copy {
            if let Err(err) = fs::remove_file(&entry.dst) {
                failed
                    .push(json!({"path": entry.dst.to_string_lossy(), "reason": err.to_string()}));
            }
        } else if let Err(err) = fs::rename(&entry.dst, &entry.restore_target) {
            failed.push(json!({"path": entry.dst.to_string_lossy(), "reason": err.to_string()}));
        }
    }
    failed
}

pub(crate) fn commit_revert(session: &mut SessionState, idx: usize) {
    if idx < session.groups.len() {
        session.groups[idx].move_log.clear();
    }
}

pub(crate) fn restore_revert_log(session: &mut SessionState, plan: &RevertPlan) {
    if plan.group_index < session.groups.len() {
        session.groups[plan.group_index].move_log = plan.move_log.clone();
    }
}

pub(crate) fn apply_group(session: &mut SessionState, idx: usize) -> Result<(), String> {
    let plan = plan_group_apply(session, idx)?;
    if plan.transfers.is_empty() && session.dry_run {
        session.groups[idx].applied = true;
        return Ok(());
    }
    execute_transfer_plan(&plan, &session.mode);
    commit_group_apply(session, &plan);
    Ok(())
}

pub(crate) fn apply_finished_groups(session: &mut SessionState) -> Result<(), String> {
    for idx in 0..session.groups.len() {
        if session.groups[idx].finished && !session.groups[idx].applied {
            apply_group(session, idx)?;
        }
    }
    Ok(())
}

pub(crate) fn transfer_with_companions(
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

pub(crate) fn revert_group_files(session: &mut SessionState, idx: usize) -> Vec<Value> {
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

pub(crate) fn reset_group_for_reopen(group: &mut GroupState) {
    group.winner = None;
    group.extra_winners.clear();
    group.losers.clear();
    group.applied = false;
    group.finished = false;
    group.left = group.images.first().cloned();
    group.right = group.images.get(1).cloned();
    group.pending = group.images.iter().skip(2).cloned().collect();
}

pub(crate) fn unique_target(folder: &Path, name: &str) -> PathBuf {
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

pub(crate) fn save_state(session: &SessionState) -> Result<(), String> {
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
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, &text).map_err(|e| format!("write state tmp: {e}"))?;
    fs::rename(&tmp, &path).map_err(|e| format!("rename state: {e}"))
}

pub(crate) fn save_info_cache(folder: &str, infos: &[InfoRecord]) -> Result<(), String> {
    let path = Path::new(folder).join(INFO_CACHE_FILENAME);
    let text = serde_json::to_string_pretty(infos).map_err(|e| format!("serialize infos: {e}"))?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, &text).map_err(|e| format!("write infos tmp: {e}"))?;
    fs::rename(&tmp, &path).map_err(|e| format!("rename infos: {e}"))
}

pub(crate) fn load_info_cache(folder: &str) -> Option<Vec<InfoRecord>> {
    let path = Path::new(folder).join(INFO_CACHE_FILENAME);
    let text = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&text).ok()
}

pub(crate) fn filter_cached_infos(
    scanned: &[ScanPair],
    cached: &[InfoRecord],
) -> (Vec<InfoRecord>, Vec<ScanPair>) {
    let mut cache_map: HashMap<&str, &InfoRecord> = HashMap::new();
    for rec in cached {
        cache_map.insert(&rec.info.path, rec);
    }
    let mut reused = Vec::new();
    let mut to_process = Vec::new();
    for pair in scanned {
        let key = pair.primary.to_string_lossy().to_string();
        if let Some(cached_rec) = cache_map.get(key.as_str()) {
            let meta = fs::metadata(&pair.primary).ok();
            let cur_mtime = meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(crate::util::system_time_secs);
            let cur_size = meta.as_ref().map(|m| m.len());
            let same_mtime = match (cur_mtime, cached_rec.info.mtime) {
                (Some(a), Some(b)) => (a - b).abs() < 2.0,
                (None, None) => true,
                _ => false,
            };
            let same_size = cur_size == cached_rec.info.size;
            if same_mtime && same_size {
                let mut rec = (*cached_rec).clone();
                rec.info.path = pair.primary.to_string_lossy().to_string();
                rec.companions = pair
                    .companions
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect();
                reused.push(rec);
                continue;
            }
        }
        to_process.push(pair.clone());
    }
    (reused, to_process)
}

pub(crate) fn status_json(session: Option<&SessionState>) -> Value {
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

pub(crate) fn best_path(group: &GroupState, meta: &HashMap<String, Value>) -> Option<String> {
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

pub(crate) fn score_for(path: &str, meta: &HashMap<String, Value>) -> f64 {
    meta.get(path)
        .and_then(|m| m.get("quality_score"))
        .and_then(|v| v.as_f64())
        .unwrap_or(50.0)
}

pub(crate) fn earliest_dt(group: &GroupState, meta: &HashMap<String, Value>) -> Option<String> {
    group
        .images
        .iter()
        .filter_map(|p| meta.get(p)?.get("datetime")?.as_str().map(str::to_string))
        .min()
}

pub(crate) fn preview_cards(
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

pub(crate) fn watermark_winner_paths(session: Option<&SessionState>) -> Vec<String> {
    session
        .map(|s| {
            s.groups
                .iter()
                .flat_map(|g| {
                    let mut out = Vec::new();
                    if let Some(w) = &g.winner {
                        out.push(w.clone());
                    }
                    out.extend(g.extra_winners.iter().cloned());
                    out
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_new_single_image_finishes_immediately() {
        let g = GroupState::new(vec!["a.jpg".to_string()]);
        assert!(g.finished);
        assert_eq!(g.winner.as_deref(), Some("a.jpg"));
        assert!(g.pending.is_empty());
    }

    #[test]
    fn group_new_two_images_sets_left_right() {
        let g = GroupState::new(vec!["a.jpg".to_string(), "b.jpg".to_string()]);
        assert!(!g.finished);
        assert_eq!(g.left.as_deref(), Some("a.jpg"));
        assert_eq!(g.right.as_deref(), Some("b.jpg"));
        assert!(g.pending.is_empty());
    }

    #[test]
    fn group_new_three_images_puts_third_in_pending() {
        let g = GroupState::new(vec![
            "a.jpg".to_string(),
            "b.jpg".to_string(),
            "c.jpg".to_string(),
        ]);
        assert!(!g.finished);
        assert_eq!(g.pending, vec!["c.jpg"]);
    }

    #[test]
    fn advance_right_loser_picks_left_as_winner() {
        let mut g = GroupState::new(vec!["a.jpg".to_string(), "b.jpg".to_string()]);
        advance(&mut g, "right");
        assert!(g.finished);
        assert_eq!(g.winner.as_deref(), Some("a.jpg"));
        assert_eq!(g.losers, vec!["b.jpg"]);
    }

    #[test]
    fn advance_left_loser_picks_right_as_winner() {
        let mut g = GroupState::new(vec!["a.jpg".to_string(), "b.jpg".to_string()]);
        advance(&mut g, "left");
        assert!(g.finished);
        assert_eq!(g.winner.as_deref(), Some("b.jpg"));
        assert_eq!(g.losers, vec!["a.jpg"]);
    }

    #[test]
    fn advance_both_loser_finishes_with_no_winner() {
        let mut g = GroupState::new(vec!["a.jpg".to_string(), "b.jpg".to_string()]);
        advance(&mut g, "both");
        assert!(g.finished);
        assert!(g.winner.is_none());
        assert_eq!(g.losers.len(), 2);
    }

    #[test]
    fn advance_neither_keeps_both_as_winners() {
        let mut g = GroupState::new(vec!["a.jpg".to_string(), "b.jpg".to_string()]);
        advance(&mut g, "neither");
        assert!(g.finished);
        assert!(g.winner.is_none());
        assert_eq!(g.extra_winners.len(), 2);
        assert!(g.losers.is_empty());
    }

    #[test]
    fn advance_three_images_continues_after_first_pick() {
        let mut g = GroupState::new(vec![
            "a.jpg".to_string(),
            "b.jpg".to_string(),
            "c.jpg".to_string(),
        ]);
        advance(&mut g, "right");
        assert!(!g.finished);
        assert_eq!(g.left.as_deref(), Some("a.jpg"));
        assert_eq!(g.right.as_deref(), Some("c.jpg"));
        assert!(g.pending.is_empty());
        advance(&mut g, "left");
        assert!(g.finished);
        assert_eq!(g.winner.as_deref(), Some("c.jpg"));
    }

    #[test]
    fn kick_side_removes_one_and_refills() {
        let mut g = GroupState::new(vec![
            "a.jpg".to_string(),
            "b.jpg".to_string(),
            "c.jpg".to_string(),
        ]);
        assert!(kick_side(&mut g, "left"));
        assert_eq!(g.losers, vec!["a.jpg"]);
        assert_eq!(g.left.as_deref(), Some("c.jpg"));
        assert_eq!(g.right.as_deref(), Some("b.jpg"));
    }

    #[test]
    fn reset_group_live_empty_finishes() {
        let mut g = GroupState::new(vec!["a.jpg".to_string(), "b.jpg".to_string()]);
        reset_group_live(&mut g, vec![]);
        assert!(g.finished);
        assert!(g.winner.is_none());
        assert!(g.auto_selected);
    }

    #[test]
    fn reset_group_live_single_auto_selects() {
        let mut g = GroupState::new(vec!["a.jpg".to_string(), "b.jpg".to_string()]);
        reset_group_live(&mut g, vec!["b.jpg".to_string()]);
        assert!(g.finished);
        assert_eq!(g.winner.as_deref(), Some("b.jpg"));
        assert!(g.auto_selected);
    }

    #[test]
    fn save_state_atomic_write() {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = SessionState {
            folder: dir.path().to_string_lossy().to_string(),
            dry_run: false,
            mode: "copy".to_string(),
            engine: "fast".to_string(),
            groups: vec![],
            current_group: 0,
            threshold_near: 10,
            threshold_far: 6,
            near_seconds: 300,
            prescreen_enabled: true,
            prescreen_strength: "standard".to_string(),
            prescreen_reviewed: false,
            prescreen_rejected: vec![],
            prescreen_reject_reasons: HashMap::new(),
            prescreen_restored: vec![],
            undo_stack: vec![],
            meta: HashMap::new(),
            companions: HashMap::new(),
            selection_started: false,
            skipped: vec![],
        };
        save_state(&session).expect("save_state");
        let state_path = dir.path().join(STATE_FILENAME);
        assert!(state_path.exists());
        let tmp_path = state_path.with_extension("json.tmp");
        assert!(!tmp_path.exists(), "tmp file should be removed by rename");
        let text = std::fs::read_to_string(&state_path).expect("read state");
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("parse state");
        assert_eq!(parsed["schema"], 6);
        assert_eq!(parsed["mode"], "copy");
    }

    #[test]
    fn unique_target_avoids_collision() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.jpg"), b"x").expect("write");
        let target = unique_target(dir.path(), "a.jpg");
        assert_eq!(target, dir.path().join("a_1.jpg"));
    }
}
