use crate::fast::types::{ExifSummary, FastImageInfo};
use chrono::{DateTime, NaiveDateTime};
use std::collections::{HashMap, HashSet};
use std::path::Path;

const HARD_BREAK_SECONDS: f64 = 30.0 * 60.0;
const TIME_HALFLIFE: f64 = 60.0;
const BURST_NUMBER_DELTA: u64 = 3;
const BURST_TIME_GAP: f64 = 1.2;
const BURST_HASH_MAX: u32 = 18;

const W_PHASH: f64 = 0.40;
const W_DHASH: f64 = 0.30;
const W_WHASH: f64 = 0.20;
const W_AHASH: f64 = 0.10;

const W_HASH: f64 = 0.32;
const W_COLOR: f64 = 0.22;
const W_TIME: f64 = 0.18;
const W_EXIF: f64 = 0.12;
const W_NAME: f64 = 0.10;
const W_GPS: f64 = 0.06;

const ORB_CANDIDATE_BASE: f64 = 0.45;
const ORB_INLIERS_STRONG: usize = 80;
const ORB_INLIERS_MEDIUM: usize = 30;
const ORB_INLIERS_WEAK: usize = 5;
const CLUSTER_DISTANCE_THRESHOLD: f64 = 0.38;
const MAX_GROUP_SIZE: usize = 25;

#[derive(Debug, Clone)]
pub struct FastClusterOptions {
    pub threshold: f64,
    pub max_group_size: usize,
    pub orb_inliers: HashMap<(usize, usize), usize>,
    pub time_halflife: f64,
}

impl Default for FastClusterOptions {
    fn default() -> Self {
        Self {
            threshold: CLUSTER_DISTANCE_THRESHOLD,
            max_group_size: MAX_GROUP_SIZE,
            orb_inliers: HashMap::new(),
            time_halflife: TIME_HALFLIFE,
        }
    }
}

pub fn cluster(infos: &[FastImageInfo]) -> Vec<Vec<usize>> {
    cluster_with_options(infos, &FastClusterOptions::default())
}

pub fn cluster_with_options(
    infos: &[FastImageInfo],
    options: &FastClusterOptions,
) -> Vec<Vec<usize>> {
    let n = infos.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![vec![0]];
    }

    let bursts = detect_bursts(infos);
    let segments = split_by_time_gaps(infos);
    let mut pair_cache: HashMap<(usize, usize), f64> = HashMap::new();
    let mut all_groups = Vec::new();

    for segment in segments {
        let seg_set: HashSet<usize> = segment.iter().copied().collect();
        let local_forced = bursts
            .iter()
            .map(|group| {
                group
                    .intersection(&seg_set)
                    .copied()
                    .collect::<HashSet<_>>()
            })
            .filter(|group| group.len() >= 2)
            .collect::<Vec<_>>();

        let sub_groups = complete_linkage(&segment, options.threshold, &local_forced, |i, j| {
            if i == j {
                return 0.0;
            }
            let key = ordered_pair(i, j);
            if let Some(d) = pair_cache.get(&key) {
                return *d;
            }
            let sim = pair_final_sim(i, j, infos, options);
            let dist = 1.0 - sim;
            pair_cache.insert(key, dist);
            dist
        });
        all_groups.extend(sub_groups);
    }

    let mut all_groups = split_oversized(all_groups, infos, options.max_group_size);
    for group in &mut all_groups {
        group.sort_by(|a, b| cmp_f64(time_for_info(&infos[*a]), time_for_info(&infos[*b])));
    }
    all_groups.sort_by(|a, b| {
        let ta = a.first().and_then(|i| time_for_info(&infos[*i]));
        let tb = b.first().and_then(|i| time_for_info(&infos[*i]));
        cmp_f64(ta, tb)
    });
    all_groups
}

fn detect_bursts(infos: &[FastImageInfo]) -> Vec<HashSet<usize>> {
    let n = infos.len();
    let mut keyed = infos
        .iter()
        .enumerate()
        .filter_map(|(i, info)| {
            let name = file_name(&info.path);
            let number = filename_number(&name)?;
            Some((filename_prefix(&name), number, time_for_info(info), i))
        })
        .collect::<Vec<_>>();
    if keyed.is_empty() {
        return Vec::new();
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut union = (0..n).collect::<Vec<_>>();
    for j in 1..keyed.len() {
        let (p0, n0, t0, i0) = (
            &keyed[j - 1].0,
            keyed[j - 1].1,
            keyed[j - 1].2,
            keyed[j - 1].3,
        );
        let (p1, n1, t1, i1) = (&keyed[j].0, keyed[j].1, keyed[j].2, keyed[j].3);
        if p0 != p1 || n1.saturating_sub(n0) > BURST_NUMBER_DELTA {
            continue;
        }
        if let (Some(a), Some(b)) = (t0, t1) {
            if b - a > BURST_TIME_GAP {
                continue;
            }
        }
        if let (Some(ph0), Some(ph1)) = (&infos[i0].phash, &infos[i1].phash) {
            if let (Some(a), Some(b)) = (hex_to_u64(ph0), hex_to_u64(ph1)) {
                if (a ^ b).count_ones() > BURST_HASH_MAX {
                    continue;
                }
            }
        }
        union_merge(&mut union, i0, i1);
    }

    let mut groups: HashMap<usize, HashSet<usize>> = HashMap::new();
    for i in 0..n {
        let root = union_find(&mut union, i);
        groups.entry(root).or_default().insert(i);
    }
    groups
        .into_values()
        .filter(|group| group.len() >= 2)
        .collect()
}

fn pair_final_sim(
    ia: usize,
    ib: usize,
    infos: &[FastImageInfo],
    options: &FastClusterOptions,
) -> f64 {
    let a = &infos[ia];
    let b = &infos[ib];
    let base = pair_base_sim(a, b, options.time_halflife);
    let hash_sim = hash_combined_sim(a, b);
    if base < ORB_CANDIDATE_BASE && hash_sim < 0.65 {
        return base;
    }

    let ta = time_for_info(a);
    let tb = time_for_info(b);
    if let (Some(ta), Some(tb)) = (ta, tb) {
        if (ta - tb).abs() > HARD_BREAK_SECONDS {
            return base * 0.5;
        }
    }

    let inliers = *options.orb_inliers.get(&ordered_pair(ia, ib)).unwrap_or(&0);
    if inliers >= ORB_INLIERS_STRONG {
        return 0.95;
    }
    if inliers >= ORB_INLIERS_MEDIUM {
        return base.max(0.85);
    }
    if inliers < ORB_INLIERS_WEAK && base > 0.55 {
        return 0.55;
    }
    base
}

fn pair_base_sim(a: &FastImageInfo, b: &FastImageInfo, time_halflife: f64) -> f64 {
    let sim_hash = hash_combined_sim(a, b);
    let (sim_color, has_color) = color_sim(a, b).map_or((0.0, false), |v| (v, true));
    let sim_time = time_sim(time_for_info(a), time_for_info(b), time_halflife);
    let sim_exif = exif_sim(a.exif_summary.as_ref(), b.exif_summary.as_ref());
    let sim_name = name_sim(&file_name(&a.path), &file_name(&b.path));
    let (sim_gps, has_gps) = gps_sim(a.exif_summary.as_ref(), b.exif_summary.as_ref())
        .map_or((0.0, false), |v| (v, true));

    let (mut w_hash, mut w_color, w_time, mut w_exif, w_name, mut w_gps) =
        (W_HASH, W_COLOR, W_TIME, W_EXIF, W_NAME, W_GPS);
    if !has_color {
        w_hash += w_color;
        w_color = 0.0;
    }
    if !has_gps {
        w_exif += w_gps;
        w_gps = 0.0;
    }

    let base = w_hash * sim_hash
        + w_color * sim_color
        + w_time * sim_time
        + w_exif * sim_exif
        + w_name * sim_name
        + w_gps * sim_gps;

    let penalty = quality_penalty(a, b);
    (base - penalty).max(0.0)
}

fn quality_penalty(a: &FastImageInfo, b: &FastImageInfo) -> f64 {
    let qa = a.quality.as_ref();
    let qb = b.quality.as_ref();
    let (Some(qa), Some(qb)) = (qa, qb) else {
        return 0.0;
    };
    let reject_a = qa.auto_reject.unwrap_or(false);
    let reject_b = qb.auto_reject.unwrap_or(false);
    if reject_a != reject_b {
        return 0.08;
    }
    if let (Some(sa), Some(sb)) = (qa.quality_score, qb.quality_score) {
        let gap = (sa - sb).abs();
        if gap > 25.0 {
            return (gap - 25.0) * 0.002;
        }
    }
    0.0
}

fn hash_combined_sim(a: &FastImageInfo, b: &FastImageInfo) -> f64 {
    let pairs = [
        (W_PHASH, hash_sim(a.phash.as_deref(), b.phash.as_deref())),
        (W_DHASH, hash_sim(a.dhash.as_deref(), b.dhash.as_deref())),
        (W_WHASH, hash_sim(a.whash.as_deref(), b.whash.as_deref())),
        (W_AHASH, hash_sim(a.ahash.as_deref(), b.ahash.as_deref())),
    ];
    let mut total_w = 0.0;
    let mut total_s = 0.0;
    for (weight, sim) in pairs {
        if let Some(sim) = sim {
            total_w += weight;
            total_s += weight * sim;
        }
    }
    if total_w < 1e-6 {
        0.0
    } else {
        total_s / total_w
    }
}

fn hash_sim(h1: Option<&str>, h2: Option<&str>) -> Option<f64> {
    let a = hex_to_u64(h1?)?;
    let b = hex_to_u64(h2?)?;
    Some((1.0 - (a ^ b).count_ones() as f64 / 64.0).max(0.0))
}

fn color_sim(a: &FastImageInfo, b: &FastImageInfo) -> Option<f64> {
    let ha = a.color_hist.as_ref()?;
    let hb = b.color_hist.as_ref()?;
    if ha.len() != hb.len() {
        return None;
    }
    let dot = ha
        .iter()
        .zip(hb.iter())
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum::<f64>();
    Some(dot.clamp(0.0, 1.0))
}

fn time_sim(t1: Option<f64>, t2: Option<f64>, halflife: f64) -> f64 {
    match (t1, t2) {
        (Some(a), Some(b)) => (-(a - b).abs() / halflife).exp(),
        _ => 0.0,
    }
}

fn exif_sim(a: Option<&ExifSummary>, b: Option<&ExifSummary>) -> f64 {
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
    if let (Some(x), Some(y)) = (
        parse_number(a.focal_length.as_deref()),
        parse_number(b.focal_length.as_deref()),
    ) {
        parts += 1.0;
        score += (1.0 - (x - y).abs() / 25.0).max(0.0);
    }
    if let (Some(x), Some(y)) = (
        parse_aperture(a.aperture.as_deref()),
        parse_aperture(b.aperture.as_deref()),
    ) {
        parts += 1.0;
        score += (1.0 - (x - y).abs() / 4.0).max(0.0);
    }
    if parts > 0.0 {
        score / parts
    } else {
        0.0
    }
}

fn gps_sim(a: Option<&ExifSummary>, b: Option<&ExifSummary>) -> Option<f64> {
    let a = a?;
    let b = b?;
    let (lat1, lon1, lat2, lon2) = (a.gps_lat?, a.gps_lon?, b.gps_lat?, b.gps_lon?);
    let avg_lat_rad = ((lat1 + lat2) / 2.0).to_radians();
    let dist = ((lat1 - lat2).powi(2) + ((lon1 - lon2) * avg_lat_rad.cos()).powi(2)).sqrt();
    Some((-dist / 0.0009).exp())
}

fn split_by_time_gaps(infos: &[FastImageInfo]) -> Vec<Vec<usize>> {
    if infos.is_empty() {
        return Vec::new();
    }
    let mut sorted_idx = (0..infos.len()).collect::<Vec<_>>();
    sorted_idx.sort_by(|a, b| cmp_f64(time_for_info(&infos[*a]), time_for_info(&infos[*b])));

    let mut segments = vec![vec![sorted_idx[0]]];
    for idx in sorted_idx.into_iter().skip(1) {
        let previous = *segments
            .last()
            .and_then(|s| s.last())
            .expect("segment exists");
        let should_break = match (time_for_info(&infos[idx]), time_for_info(&infos[previous])) {
            (Some(cur), Some(prev)) => cur - prev > HARD_BREAK_SECONDS,
            _ => false,
        };
        if should_break {
            segments.push(vec![idx]);
        } else {
            segments.last_mut().expect("segment exists").push(idx);
        }
    }
    segments
}

fn complete_linkage<F>(
    members: &[usize],
    threshold: f64,
    forced_groups: &[HashSet<usize>],
    mut dist_fn: F,
) -> Vec<Vec<usize>>
where
    F: FnMut(usize, usize) -> f64,
{
    if members.is_empty() {
        return Vec::new();
    }
    let member_set = members.iter().copied().collect::<HashSet<_>>();
    let mut clusters = members
        .iter()
        .copied()
        .map(|i| (i, HashSet::from([i])))
        .collect::<HashMap<_, _>>();

    for group in forced_groups {
        let group = group.intersection(&member_set).copied().collect::<Vec<_>>();
        if let Some((&anchor, rest)) = group.split_first() {
            for idx in rest {
                if let Some(moved) = clusters.remove(idx) {
                    clusters.entry(anchor).or_default().extend(moved);
                }
            }
        }
    }

    let mut cache: HashMap<(usize, usize), f64> = HashMap::new();
    loop {
        let ids = clusters.keys().copied().collect::<Vec<_>>();
        if ids.len() < 2 {
            break;
        }
        let mut best_pair = None;
        let mut best_dist = f64::INFINITY;

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let key = ordered_pair(ids[i], ids[j]);
                let dist = if let Some(d) = cache.get(&key) {
                    *d
                } else {
                    let mut max_dist = 0.0;
                    for a in &clusters[&ids[i]] {
                        for b in &clusters[&ids[j]] {
                            let d = dist_fn(*a, *b);
                            if d > max_dist {
                                max_dist = d;
                                if max_dist > threshold {
                                    break;
                                }
                            }
                        }
                        if max_dist > threshold {
                            break;
                        }
                    }
                    cache.insert(key, max_dist);
                    max_dist
                };
                if dist < best_dist {
                    best_dist = dist;
                    best_pair = Some((ids[i], ids[j]));
                }
                if best_dist <= 0.0 {
                    break;
                }
            }
            if best_dist <= 0.0 {
                break;
            }
        }

        let Some((a, b)) = best_pair else {
            break;
        };
        if best_dist > threshold {
            break;
        }
        if let Some(moved) = clusters.remove(&b) {
            clusters.entry(a).or_default().extend(moved);
        }
        cache.retain(|(x, y), _| *x != a && *x != b && *y != a && *y != b);
    }

    clusters
        .into_values()
        .map(|group| {
            let mut group = group.into_iter().collect::<Vec<_>>();
            group.sort_unstable();
            group
        })
        .collect()
}

fn split_oversized(
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
            .map(|idx| (time_for_info(&infos[idx]).unwrap_or(0.0), idx))
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

fn time_for_info(info: &FastImageInfo) -> Option<f64> {
    if let Some(meta) = &info.exif_summary {
        if let Some(datetime) = &meta.datetime {
            if let Some(t) = parse_iso_datetime(datetime) {
                return Some(t);
            }
        }
    }
    info.timestamp.or(info.mtime)
}

fn parse_iso_datetime(value: &str) -> Option<f64> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return Some(dt.timestamp() as f64 + f64::from(dt.timestamp_subsec_micros()) / 1_000_000.0);
    }
    let naive = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").ok()?;
    Some(
        naive.and_utc().timestamp() as f64
            + f64::from(naive.and_utc().timestamp_subsec_micros()) / 1_000_000.0,
    )
}

fn name_sim(name1: &str, name2: &str) -> f64 {
    let (Some(n1), Some(n2)) = (filename_number(name1), filename_number(name2)) else {
        return 0.0;
    };
    if filename_prefix(name1) != filename_prefix(name2) {
        return 0.0;
    }
    let delta = n1.abs_diff(n2) as f64;
    if delta == 0.0 {
        1.0
    } else {
        (1.0 - delta / 30.0).max(0.0)
    }
}

fn filename_number(name: &str) -> Option<u64> {
    let stem = file_stem(name);
    let digits = stem
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    digits.chars().rev().collect::<String>().parse().ok()
}

fn filename_prefix(name: &str) -> String {
    let stem = file_stem(name);
    stem.trim_end_matches(|ch: char| ch.is_ascii_digit())
        .to_string()
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

fn file_stem(name: &str) -> &str {
    Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name)
}

fn parse_number(value: Option<&str>) -> Option<f64> {
    let value = value?;
    let cleaned = value.trim().trim_end_matches("mm");
    cleaned.parse().ok()
}

fn parse_aperture(value: Option<&str>) -> Option<f64> {
    let value = value?;
    value
        .trim()
        .strip_prefix("f/")
        .unwrap_or(value.trim())
        .parse()
        .ok()
}

fn hex_to_u64(value: &str) -> Option<u64> {
    u64::from_str_radix(value, 16).ok()
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn cmp_f64(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
    a.unwrap_or(0.0)
        .partial_cmp(&b.unwrap_or(0.0))
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn union_find(union: &mut [usize], mut x: usize) -> usize {
    while union[x] != x {
        union[x] = union[union[x]];
        x = union[x];
    }
    x
}

fn union_merge(union: &mut [usize], a: usize, b: usize) {
    let ra = union_find(union, a);
    let rb = union_find(union, b);
    if ra != rb {
        union[ra] = rb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(path: &str, phash: &str, time: f64) -> FastImageInfo {
        FastImageInfo {
            path: path.to_string(),
            phash: Some(phash.to_string()),
            dhash: Some(phash.to_string()),
            whash: Some(phash.to_string()),
            ahash: Some(phash.to_string()),
            timestamp: Some(time),
            color_hist: Some(vec![1.0, 0.0, 0.0]),
            ..FastImageInfo::default()
        }
    }

    #[test]
    fn groups_near_identical_hashes() {
        let infos = vec![
            info("IMG_0001.jpg", "ffffffffffffffff", 100.0),
            info("IMG_0002.jpg", "fffffffffffffffe", 101.0),
            info("IMG_0100.jpg", "0000000000000000", 4000.0),
        ];
        assert_eq!(cluster(&infos), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn time_gap_hard_splits_even_similar_hashes() {
        let infos = vec![
            info("IMG_0001.jpg", "ffffffffffffffff", 100.0),
            info("IMG_0002.jpg", "ffffffffffffffff", 4000.0),
        ];
        assert_eq!(cluster(&infos), vec![vec![0], vec![1]]);
    }

    #[test]
    fn orb_override_can_promote_medium_match() {
        let mut a = info("A_0001.jpg", "ffffffffffffffff", 100.0);
        let mut b = info("B_0001.jpg", "ffffffffffff0000", 105.0);
        a.color_hist = Some(vec![0.7, 0.3, 0.0]);
        b.color_hist = Some(vec![0.7, 0.3, 0.0]);
        let mut options = FastClusterOptions::default();
        options.orb_inliers.insert((0, 1), ORB_INLIERS_MEDIUM);
        assert_eq!(cluster_with_options(&[a, b], &options), vec![vec![0, 1]]);
    }
}
