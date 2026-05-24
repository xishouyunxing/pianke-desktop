use pianke_core::fast::{cluster_with_options, ExifSummary, FastClusterOptions, FastImageInfo};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct GoldenFile {
    #[serde(default)]
    images: Vec<GoldenImage>,
    #[serde(default)]
    orb_pairs: Vec<GoldenOrbPair>,
    #[serde(default)]
    groups: Vec<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct GoldenImage {
    path: String,
    #[serde(default)]
    timestamp: Option<f64>,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    mtime: Option<f64>,
    #[serde(default)]
    exif_summary: Option<ExifSummary>,
    #[serde(default)]
    hashes: GoldenHashes,
    #[serde(default)]
    color_hist: Option<Vec<f32>>,
}

#[derive(Debug, Default, Deserialize)]
struct GoldenHashes {
    #[serde(default)]
    phash: Option<String>,
    #[serde(default)]
    dhash: Option<String>,
    #[serde(default)]
    whash: Option<String>,
    #[serde(default)]
    ahash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GoldenOrbPair {
    i: usize,
    j: usize,
    orb_inliers: usize,
}

#[test]
fn python_fast_golden_groups_match() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("fast_parity");
    if !fixture_dir.exists() {
        return;
    }

    let files = fs::read_dir(&fixture_dir)
        .expect("read fast parity fixture directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect::<Vec<_>>();
    if files.is_empty() {
        return;
    }

    for path in files {
        let text = fs::read_to_string(&path).expect("read fast parity fixture");
        let golden: GoldenFile = serde_json::from_str(&text).expect("parse fast parity fixture");
        let infos = golden
            .images
            .into_iter()
            .map(|image| FastImageInfo {
                path: image.path,
                phash: image.hashes.phash,
                dhash: image.hashes.dhash,
                whash: image.hashes.whash,
                ahash: image.hashes.ahash,
                timestamp: image.timestamp,
                size: image.size,
                mtime: image.mtime,
                exif_summary: image.exif_summary,
                color_hist: image.color_hist,
                ..FastImageInfo::default()
            })
            .collect::<Vec<_>>();
        let mut options = FastClusterOptions::default();
        options.orb_inliers = golden
            .orb_pairs
            .into_iter()
            .map(|pair| ((pair.i.min(pair.j), pair.i.max(pair.j)), pair.orb_inliers))
            .collect::<HashMap<_, _>>();

        assert_eq!(
            cluster_with_options(&infos, &options),
            golden.groups,
            "fixture {}",
            path.display()
        );
    }
}
