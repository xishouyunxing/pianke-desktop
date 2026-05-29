use pianke_core::fast::FastImageInfo;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::util::file_name;

pub(crate) fn expert_cluster(infos: &[FastImageInfo]) -> Vec<Vec<usize>> {
    const DISTANCE_THRESHOLD: f64 = 0.46;
    const HARD_BREAK_SECONDS: f64 = 45.0 * 60.0;
    let n = infos.len();
    if n == 0 {
        return Vec::new();
    }

    let mut sorted_idx = (0..n).collect::<Vec<_>>();
    sorted_idx.sort_by(|a, b| {
        expert_time_for_info(&infos[*a])
            .unwrap_or(0.0)
            .partial_cmp(&expert_time_for_info(&infos[*b]).unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut segments: Vec<Vec<usize>> = Vec::new();
    for idx in sorted_idx {
        let split = segments
            .last()
            .and_then(|seg| seg.last().copied())
            .is_some_and(|prev| {
                match (
                    expert_time_for_info(&infos[idx]),
                    expert_time_for_info(&infos[prev]),
                ) {
                    (Some(cur), Some(prev)) => cur - prev > HARD_BREAK_SECONDS,
                    _ => false,
                }
            });
        if split || segments.is_empty() {
            segments.push(vec![idx]);
        } else if let Some(seg) = segments.last_mut() {
            seg.push(idx);
        }
    }

    let mut groups = Vec::new();
    for segment in segments {
        groups.extend(expert_complete_linkage_segment(
            infos,
            segment,
            DISTANCE_THRESHOLD,
        ));
    }
    groups = expert_split_oversized(groups, infos, 25);
    for group in &mut groups {
        group.sort_by(|a, b| {
            expert_time_for_info(&infos[*a])
                .unwrap_or(0.0)
                .partial_cmp(&expert_time_for_info(&infos[*b]).unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    groups.sort_by(|a, b| {
        let ta = a
            .first()
            .and_then(|idx| expert_time_for_info(&infos[*idx]))
            .unwrap_or(0.0);
        let tb = b
            .first()
            .and_then(|idx| expert_time_for_info(&infos[*idx]))
            .unwrap_or(0.0);
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });
    groups
}

fn expert_complete_linkage_segment(
    infos: &[FastImageInfo],
    members: Vec<usize>,
    threshold: f64,
) -> Vec<Vec<usize>> {
    if members.is_empty() {
        return Vec::new();
    }
    let mut clusters = members
        .into_iter()
        .map(|idx| (idx, vec![idx]))
        .collect::<HashMap<usize, Vec<usize>>>();
    let mut cache: HashMap<(usize, usize), f64> = HashMap::new();
    loop {
        let ids = clusters.keys().copied().collect::<Vec<_>>();
        if ids.len() < 2 {
            break;
        }
        let mut best_pair = None;
        let mut best_d = f64::INFINITY;
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let d = expert_cluster_distance(
                    ids[i], ids[j], &clusters, infos, &mut cache, threshold,
                );
                if d < best_d {
                    best_d = d;
                    best_pair = Some((ids[i], ids[j]));
                }
            }
        }
        let Some((a, b)) = best_pair else {
            break;
        };
        if best_d > threshold {
            break;
        }
        let moved = clusters.remove(&b).unwrap_or_default();
        clusters.entry(a).or_default().extend(moved);
        cache.retain(|(x, y), _| *x != a && *y != a && *x != b && *y != b);
    }
    clusters.into_values().collect()
}

fn expert_cluster_distance(
    ca: usize,
    cb: usize,
    clusters: &HashMap<usize, Vec<usize>>,
    infos: &[FastImageInfo],
    cache: &mut HashMap<(usize, usize), f64>,
    threshold: f64,
) -> f64 {
    let key = (ca.min(cb), ca.max(cb));
    if let Some(value) = cache.get(&key) {
        return *value;
    }
    let mut max_d = 0.0;
    let Some(a_members) = clusters.get(&ca) else {
        return 1.0;
    };
    let Some(b_members) = clusters.get(&cb) else {
        return 1.0;
    };
    for i in a_members {
        for j in b_members {
            let d = 1.0 - expert_pair_similarity(&infos[*i], &infos[*j]);
            if d > max_d {
                max_d = d;
                if max_d > threshold {
                    cache.insert(key, max_d);
                    return max_d;
                }
            }
        }
    }
    cache.insert(key, max_d);
    max_d
}

fn expert_split_oversized(
    groups: Vec<Vec<usize>>,
    infos: &[FastImageInfo],
    max_size: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut stack = groups;
    while let Some(group) = stack.pop() {
        if group.len() <= max_size {
            out.push(group);
            continue;
        }
        let mut timed = group
            .into_iter()
            .map(|idx| (expert_time_for_info(&infos[idx]).unwrap_or(0.0), idx))
            .collect::<Vec<_>>();
        timed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut best_gap = -1.0;
        let mut best_k = timed.len() / 2;
        for k in 1..timed.len() {
            let gap = timed[k].0 - timed[k - 1].0;
            if gap > best_gap {
                best_gap = gap;
                best_k = k;
            }
        }
        stack.push(timed[..best_k].iter().map(|(_, idx)| *idx).collect());
        stack.push(timed[best_k..].iter().map(|(_, idx)| *idx).collect());
    }
    out
}

fn expert_time_for_info(info: &FastImageInfo) -> Option<f64> {
    info.timestamp.or(info.mtime)
}

pub(crate) fn expert_pair_similarity(a: &FastImageInfo, b: &FastImageInfo) -> f64 {
    const W_DINOV2: f64 = 0.35;
    const W_TIME: f64 = 0.20;
    const W_EXIF: f64 = 0.10;
    const W_FACE: f64 = 0.15;
    const W_GPS: f64 = 0.10;
    const W_FILENAME: f64 = 0.10;
    const PORTRAIT_W_DINOV2: f64 = 0.20;
    const PORTRAIT_W_TIME: f64 = 0.15;
    const PORTRAIT_W_EXIF: f64 = 0.05;
    const PORTRAIT_W_FACE: f64 = 0.50;
    const PORTRAIT_W_GPS: f64 = 0.05;
    const PORTRAIT_W_FILENAME: f64 = 0.05;

    let dino = cosine_similarity(a.dinov2.as_deref(), b.dinov2.as_deref())
        .expect("Expert/Tycoon clustering requires DINOv2 embeddings");
    let time = time_similarity(a.mtime, b.mtime);
    let exif = expert_exif_similarity(a.exif_summary.as_ref(), b.exif_summary.as_ref());
    let (gps, has_gps) = expert_gps_similarity(a.exif_summary.as_ref(), b.exif_summary.as_ref())
        .map_or((0.0, false), |v| (v, true));
    let face = face_overlap_similarity(&a.face_embeddings, &b.face_embeddings);
    let name = expert_filename_similarity(&a.path, &b.path);
    let portrait = !a.face_embeddings.is_empty() && !b.face_embeddings.is_empty();
    let (mut w_dino, w_time, mut w_exif, w_face, mut w_gps, w_name) = if portrait {
        (
            PORTRAIT_W_DINOV2,
            PORTRAIT_W_TIME,
            PORTRAIT_W_EXIF,
            PORTRAIT_W_FACE,
            PORTRAIT_W_GPS,
            PORTRAIT_W_FILENAME,
        )
    } else {
        (W_DINOV2, W_TIME, W_EXIF, W_FACE, W_GPS, W_FILENAME)
    };
    if !has_gps {
        w_exif += w_gps;
        w_gps = 0.0;
    }
    if !portrait && face == 0.0 && (a.face_embeddings.is_empty() || b.face_embeddings.is_empty()) {
        w_dino += w_face * 0.4;
        let w_time_adj = w_time + w_face * 0.3;
        w_exif += w_face * 0.3;
        return w_dino * dino + w_time_adj * time + w_exif * exif + w_gps * gps + w_name * name;
    }
    w_dino * dino + w_time * time + w_exif * exif + w_face * face + w_gps * gps + w_name * name
}

pub(crate) fn cosine_similarity(a: Option<&[f32]>, b: Option<&[f32]>) -> Option<f64> {
    let (Some(a), Some(b)) = (a, b) else {
        return None;
    };
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na < 1e-12 || nb < 1e-12 {
        return None;
    }
    Some((dot / (na.sqrt() * nb.sqrt())).clamp(0.0, 1.0))
}

pub(crate) fn face_overlap_similarity(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    let n1 = a.len();
    let n2 = b.len();
    if n1 == 0 && n2 == 0 {
        1.0
    } else if n1 == 0 || n2 == 0 {
        0.0
    } else {
        let mut matched_a = HashSet::new();
        let mut matched_b = HashSet::new();
        for (i, emb_a) in a.iter().enumerate() {
            if matched_a.contains(&i) {
                continue;
            }
            let mut best_j = None;
            let mut best_sim = -1.0;
            for (j, emb_b) in b.iter().enumerate() {
                if matched_b.contains(&j) {
                    continue;
                }
                let sim = cosine_similarity(Some(emb_a), Some(emb_b)).unwrap_or(0.0);
                if sim > best_sim {
                    best_sim = sim;
                    best_j = Some(j);
                }
            }
            if let Some(j) = best_j {
                if best_sim > 0.45 {
                    matched_a.insert(i);
                    matched_b.insert(j);
                }
            }
        }
        let matched = matched_a.len();
        let denom = n1 + n2 - matched;
        if denom == 0 {
            0.0
        } else {
            matched as f64 / denom as f64
        }
    }
}

fn time_similarity(a: Option<f64>, b: Option<f64>) -> f64 {
    match (a, b) {
        (Some(x), Some(y)) => (-(x - y).abs() / 60.0).exp(),
        _ => 0.0,
    }
}

fn expert_exif_similarity(
    a: Option<&pianke_core::fast::ExifSummary>,
    b: Option<&pianke_core::fast::ExifSummary>,
) -> f64 {
    let (Some(a), Some(b)) = (a, b) else {
        return 0.0;
    };
    let mut score = 0.0;
    let mut parts = 0.0;
    if let (Some(x), Some(y)) = (&a.camera, &b.camera) {
        parts += 1.0;
        if x == y {
            score += 1.0;
        }
    }
    if let (Some(x), Some(y)) = (&a.lens, &b.lens) {
        parts += 1.0;
        if x == y {
            score += 1.0;
        }
    }
    if parts > 0.0 {
        score / parts
    } else {
        0.0
    }
}

fn expert_gps_similarity(
    a: Option<&pianke_core::fast::ExifSummary>,
    b: Option<&pianke_core::fast::ExifSummary>,
) -> Option<f64> {
    let a = a?;
    let b = b?;
    let (lat1, lon1, lat2, lon2) = (a.gps_lat?, a.gps_lon?, b.gps_lat?, b.gps_lon?);
    let avg_lat_rad = ((lat1 + lat2) / 2.0).to_radians();
    let dist = ((lat1 - lat2).powi(2) + ((lon1 - lon2) * avg_lat_rad.cos()).powi(2)).sqrt();
    Some((-dist / 0.0009).exp())
}

fn expert_filename_similarity(a: &str, b: &str) -> f64 {
    let name_a = file_name(a);
    let name_b = file_name(b);
    let (Some(n1), Some(n2)) = (
        filename_number_from_name(&name_a),
        filename_number_from_name(&name_b),
    ) else {
        return 0.0;
    };
    if filename_prefix_from_name(&name_a) != filename_prefix_from_name(&name_b) {
        return 0.0;
    }
    let delta = n1.abs_diff(n2) as f64;
    if delta == 0.0 {
        1.0
    } else {
        (1.0 - delta / 30.0).max(0.0)
    }
}

fn filename_number_from_name(name: &str) -> Option<u64> {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name);
    let digits = stem
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.chars().rev().collect::<String>().parse().ok()
    }
}

fn filename_prefix_from_name(name: &str) -> String {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name);
    stem.trim_end_matches(|ch: char| ch.is_ascii_digit())
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let sim = cosine_similarity(Some(&a), Some(&a)).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(Some(&a), Some(&b)).unwrap();
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_none_returns_none() {
        assert!(cosine_similarity(None, Some(&[1.0])).is_none());
        assert!(cosine_similarity(Some(&[1.0]), None).is_none());
        assert!(cosine_similarity(None, None).is_none());
    }

    #[test]
    fn time_similarity_zero_delta() {
        assert!((time_similarity(Some(100.0), Some(100.0)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn time_similarity_large_delta_near_zero() {
        let sim = time_similarity(Some(0.0), Some(600.0));
        assert!(sim < 0.01);
    }

    #[test]
    fn time_similarity_none_returns_zero() {
        assert!((time_similarity(None, Some(100.0))).abs() < 1e-10);
        assert!((time_similarity(Some(100.0), None)).abs() < 1e-10);
    }

    #[test]
    fn expert_filename_similarity_adjacent_numbers() {
        let sim = expert_filename_similarity("IMG_0001.jpg", "IMG_0002.jpg");
        assert!(
            sim > 0.9,
            "adjacent numbers should have high similarity, got {sim}"
        );
    }

    #[test]
    fn expert_filename_similarity_far_numbers() {
        let sim = expert_filename_similarity("IMG_0001.jpg", "IMG_0100.jpg");
        assert!(
            sim < 0.01,
            "far numbers should have low similarity, got {sim}"
        );
    }

    #[test]
    fn expert_filename_similarity_different_prefix() {
        let sim = expert_filename_similarity("IMG_0001.jpg", "DSC_0001.jpg");
        assert!(sim.abs() < 1e-10, "different prefix should be 0, got {sim}");
    }

    #[test]
    fn face_overlap_both_empty_is_one() {
        assert!((face_overlap_similarity(&[], &[]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn face_overlap_one_empty_is_zero() {
        let a = vec![vec![1.0, 0.0]];
        assert!((face_overlap_similarity(&a, &[])).abs() < 1e-10);
        assert!((face_overlap_similarity(&[], &a)).abs() < 1e-10);
    }
}
