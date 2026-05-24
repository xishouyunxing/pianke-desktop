/// Convert row-major boolean hash bits to the same hex string format used by
/// Python imagehash.ImageHash.__str__.
pub fn bits_to_imagehash_hex(bits: &[bool]) -> String {
    if bits.is_empty() {
        return String::new();
    }
    let width = bits.len().div_ceil(4);
    let mut value = String::with_capacity(width);
    for chunk in bits.chunks(4) {
        let mut nibble = 0u8;
        for (i, bit) in chunk.iter().enumerate() {
            if *bit {
                nibble |= 1 << (3 - i);
            }
        }
        value.push(char::from_digit(nibble as u32, 16).expect("nibble is hex"));
    }
    value
}

/// Average hash for already resized grayscale samples.
///
/// Python reference:
/// image.convert("L").resize((hash_size, hash_size), LANCZOS), then
/// pixels > numpy.mean(pixels), row-major hex packing.
pub fn average_hash_from_luma(samples: &[u8], hash_size: usize) -> Option<String> {
    if hash_size < 2 || samples.len() != hash_size * hash_size {
        return None;
    }
    let avg = samples.iter().map(|v| f64::from(*v)).sum::<f64>() / samples.len() as f64;
    let bits = samples
        .iter()
        .map(|pixel| f64::from(*pixel) > avg)
        .collect::<Vec<_>>();
    Some(bits_to_imagehash_hex(&bits))
}

/// Horizontal difference hash for already resized grayscale samples.
///
/// The sample matrix must be hash_size rows by hash_size + 1 columns. This
/// isolates imagehash-compatible comparison and bit order from the later,
/// harder question of matching Pillow's resize exactly.
pub fn difference_hash_from_luma(samples: &[u8], hash_size: usize) -> Option<String> {
    if hash_size < 2 || samples.len() != hash_size * (hash_size + 1) {
        return None;
    }
    let stride = hash_size + 1;
    let mut bits = Vec::with_capacity(hash_size * hash_size);
    for row in 0..hash_size {
        let offset = row * stride;
        for col in 0..hash_size {
            bits.push(samples[offset + col + 1] > samples[offset + col]);
        }
    }
    Some(bits_to_imagehash_hex(&bits))
}

/// Perceptual hash for already resized grayscale samples.
///
/// Python reference:
/// image.convert("L").resize((hash_size * highfreq_factor, ...), LANCZOS),
/// scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1), then
/// low-frequency hash_size x hash_size block > numpy.median(block).
pub fn perceptual_hash_from_luma(samples: &[u8], hash_size: usize) -> Option<String> {
    if hash_size < 2 {
        return None;
    }
    let size = hash_size * 4;
    if samples.len() != size * size {
        return None;
    }

    let mut stage = vec![0.0; size * size];
    for col in 0..size {
        let input = (0..size)
            .map(|row| f64::from(samples[row * size + col]))
            .collect::<Vec<_>>();
        let output = dct_type_ii(&input);
        for row in 0..size {
            stage[row * size + col] = output[row];
        }
    }

    let mut dct = vec![0.0; size * size];
    for row in 0..size {
        let input = (0..size)
            .map(|col| stage[row * size + col])
            .collect::<Vec<_>>();
        let output = dct_type_ii(&input);
        for col in 0..size {
            dct[row * size + col] = output[col];
        }
    }

    let mut low = Vec::with_capacity(hash_size * hash_size);
    for row in 0..hash_size {
        for col in 0..hash_size {
            low.push(dct[row * size + col]);
        }
    }
    let median = median_like_numpy(low.clone());
    let bits = low.iter().map(|value| *value > median).collect::<Vec<_>>();
    Some(bits_to_imagehash_hex(&bits))
}

/// Haar wavelet hash for already resized grayscale samples.
///
/// This mirrors Python imagehash.whash defaults for mode="haar" and
/// remove_max_haar_ll=true. The caller must resize to image_scale x image_scale,
/// where image_scale is a power of two and at least hash_size.
pub fn wavelet_hash_from_luma(
    samples: &[u8],
    hash_size: usize,
    image_scale: usize,
) -> Option<String> {
    if hash_size < 2
        || !hash_size.is_power_of_two()
        || !image_scale.is_power_of_two()
        || image_scale < hash_size
        || samples.len() != image_scale * image_scale
    {
        return None;
    }

    let mut current = samples
        .iter()
        .map(|value| f64::from(*value) / 255.0)
        .collect::<Vec<_>>();

    // imagehash.whash removes the single lowest-frequency LL coefficient. For
    // Haar at full depth this is equivalent to subtracting the global mean.
    let mean = current.iter().sum::<f64>() / current.len() as f64;
    for value in &mut current {
        *value -= mean;
    }

    let mut current_size = image_scale;
    while current_size > hash_size {
        let next_size = current_size / 2;
        let mut next = vec![0.0; next_size * next_size];
        for y in 0..next_size {
            for x in 0..next_size {
                let y0 = y * 2;
                let x0 = x * 2;
                let a = current[y0 * current_size + x0];
                let b = current[y0 * current_size + x0 + 1];
                let c = current[(y0 + 1) * current_size + x0];
                let d = current[(y0 + 1) * current_size + x0 + 1];
                next[y * next_size + x] = (a + b + c + d) / 4.0;
            }
        }
        current = next;
        current_size = next_size;
    }

    let median = median_like_numpy(current.clone());
    let bits = current
        .iter()
        .map(|value| *value > median)
        .collect::<Vec<_>>();
    Some(bits_to_imagehash_hex(&bits))
}

fn dct_type_ii(input: &[f64]) -> Vec<f64> {
    let n = input.len() as f64;
    (0..input.len())
        .map(|k| {
            let k = k as f64;
            2.0 * input
                .iter()
                .enumerate()
                .map(|(n_idx, value)| {
                    let angle = std::f64::consts::PI * (n_idx as f64 + 0.5) * k / n;
                    *value * angle.cos()
                })
                .sum::<f64>()
        })
        .collect()
}

fn median_like_numpy(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_bits_like_python_imagehash() {
        let mut bits = vec![false; 64];
        bits[0] = true;
        bits[63] = true;
        assert_eq!(bits_to_imagehash_hex(&bits), "8000000000000001");
    }

    #[test]
    fn average_hash_uses_strict_greater_than_mean() {
        let samples = (0u8..64).collect::<Vec<_>>();
        assert_eq!(
            average_hash_from_luma(&samples, 8).as_deref(),
            Some("00000000ffffffff")
        );
    }

    #[test]
    fn difference_hash_compares_adjacent_columns() {
        let mut increasing = Vec::new();
        let mut decreasing = Vec::new();
        for _ in 0..8 {
            increasing.extend(0u8..9);
            decreasing.extend((0u8..9).rev());
        }
        assert_eq!(
            difference_hash_from_luma(&increasing, 8).as_deref(),
            Some("ffffffffffffffff")
        );
        assert_eq!(
            difference_hash_from_luma(&decreasing, 8).as_deref(),
            Some("0000000000000000")
        );
    }

    #[test]
    fn perceptual_hash_matches_python_reference_fixture() {
        let mut samples = Vec::with_capacity(32 * 32);
        for y in 0..32 {
            for x in 0..32 {
                samples.push(((x * 7 + y * 11 + (x * y) % 17) % 256) as u8);
            }
        }
        assert_eq!(
            perceptual_hash_from_luma(&samples, 8).as_deref(),
            Some("9748943f602d5ef8")
        );
    }

    #[test]
    fn wavelet_hash_matches_python_imagehash_fixture() {
        let mut samples = Vec::with_capacity(32 * 32);
        for y in 0..32 {
            for x in 0..32 {
                samples.push(((x * 7 + y * 11 + (x * y) % 17) % 256) as u8);
            }
        }
        assert_eq!(
            wavelet_hash_from_luma(&samples, 8, 32).as_deref(),
            Some("0f3e78e0c1870f3e")
        );
    }
}
