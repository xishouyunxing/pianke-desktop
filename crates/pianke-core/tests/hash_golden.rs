use pianke_core::fast::{
    average_hash_from_luma, difference_hash_from_luma, perceptual_hash_from_luma,
    wavelet_hash_from_luma,
};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Debug, Deserialize)]
struct HashGolden {
    average_8x8: AverageGolden,
    difference_9x8: DifferenceGolden,
    perceptual_32x32: PerceptualGolden,
    wavelet_32x32: Option<WaveletGolden>,
}

#[derive(Debug, Deserialize)]
struct AverageGolden {
    samples: Vec<u8>,
    ahash: String,
}

#[derive(Debug, Deserialize)]
struct DifferenceGolden {
    samples: Vec<u8>,
    dhash: String,
}

#[derive(Debug, Deserialize)]
struct PerceptualGolden {
    phash: String,
}

#[derive(Debug, Deserialize)]
struct WaveletGolden {
    whash: String,
}

#[test]
fn synthetic_hashes_match_python_imagehash_golden() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("fast_parity")
        .join("hash_synthetic.json");
    let text = fs::read_to_string(&fixture).expect("read hash golden fixture");
    let golden: HashGolden = serde_json::from_str(&text).expect("parse hash golden fixture");

    assert_eq!(
        average_hash_from_luma(&golden.average_8x8.samples, 8).as_deref(),
        Some(golden.average_8x8.ahash.as_str())
    );
    assert_eq!(
        difference_hash_from_luma(&golden.difference_9x8.samples, 8).as_deref(),
        Some(golden.difference_9x8.dhash.as_str())
    );

    let mut phash_samples = Vec::with_capacity(32 * 32);
    for y in 0..32 {
        for x in 0..32 {
            phash_samples.push(((x * 7 + y * 11 + (x * y) % 17) % 256) as u8);
        }
    }
    assert_eq!(
        perceptual_hash_from_luma(&phash_samples, 8).as_deref(),
        Some(golden.perceptual_32x32.phash.as_str())
    );
    if let Some(wavelet) = golden.wavelet_32x32 {
        assert_eq!(
            wavelet_hash_from_luma(&phash_samples, 8, 32).as_deref(),
            Some(wavelet.whash.as_str())
        );
    }
}
