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
}
