use image::imageops::FilterType;
use image::DynamicImage;
use image::GenericImageView;
use pianke_core::fast::FastQualitySignals;

use crate::expert_vision;
use crate::util;

#[derive(Clone)]
pub(crate) struct GrayMatrix {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) data: Vec<f64>,
}

impl GrayMatrix {
    pub(crate) fn new(width: usize, height: usize, data: Vec<f64>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    pub(crate) fn from_luma(img: &image::GrayImage) -> Self {
        let (w, h) = img.dimensions();
        let data = img.as_raw().iter().map(|v| f64::from(*v)).collect();
        Self::new(w as usize, h as usize, data)
    }

    pub(crate) fn zeros(width: usize, height: usize) -> Self {
        Self::new(width, height, vec![0.0; width.saturating_mul(height)])
    }

    pub(crate) fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    pub(crate) fn set(&mut self, x: usize, y: usize, value: f64) {
        self.data[y * self.width + x] = value;
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / self.len().max(1) as f64
    }

    pub(crate) fn std(&self) -> f64 {
        let mean = self.mean();
        (self.data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / self.len().max(1) as f64)
            .sqrt()
    }

    pub(crate) fn ratio_le(&self, threshold: f64) -> f64 {
        self.data.iter().filter(|v| **v <= threshold).count() as f64 / self.len().max(1) as f64
    }

    pub(crate) fn ratio_ge(&self, threshold: f64) -> f64 {
        self.data.iter().filter(|v| **v >= threshold).count() as f64 / self.len().max(1) as f64
    }

    pub(crate) fn center_crop(&self, ratio: f64) -> Self {
        let crop_h = ((self.height as f64 * ratio) as usize).max(1);
        let crop_w = ((self.width as f64 * ratio) as usize).max(1);
        let y0 = (self.height.saturating_sub(crop_h)) / 2;
        let x0 = (self.width.saturating_sub(crop_w)) / 2;
        let mut data = Vec::with_capacity(crop_w * crop_h);
        for y in y0..(y0 + crop_h).min(self.height) {
            for x in x0..(x0 + crop_w).min(self.width) {
                data.push(self.get(x, y));
            }
        }
        Self::new(crop_w, crop_h, data)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NineGridExposure {
    pub(crate) worst_clip_dark: f64,
    pub(crate) worst_clip_bright: f64,
}

pub(crate) fn quality_signals(img: &DynamicImage, file_size: u64) -> FastQualitySignals {
    let rgb = img.to_rgb8();
    let gray = util::pillow_luma(&rgb);
    let work = expert_vision::pillow_thumbnail_luma(&gray, 768, 768);
    let arr = GrayMatrix::from_luma(&work);
    let (width, height) = img.dimensions();

    let brightness_mean = arr.mean();
    let brightness_std = arr.std();
    let contrast_score = brightness_std;
    let underexposed_ratio = arr.ratio_le(8.0);
    let overexposed_ratio = arr.ratio_ge(247.0);
    let entropy = matrix_entropy(&arr);
    let lap = laplacian_variance(&arr).max(laplacian_variance(&arr.center_crop(0.6)));
    let tenengrad = tenengrad(&arr);
    let (high_ratio, motion_anisotropy) = fft_high_freq_ratio(&arr);
    let edge_width = edge_width_marziliano(&arr);

    let lap_norm = (lap.max(0.0).ln_1p() / 900.0f64.ln_1p()).min(1.0);
    let tenengrad_norm = (tenengrad.max(0.0).ln_1p() / 2000.0f64.ln_1p()).min(1.0);
    let high_norm = (high_ratio.max(0.0) / 0.40).min(1.0);
    let mut parts = vec![lap_norm, tenengrad_norm, high_norm];
    if let Some(width) = edge_width {
        parts.push(((10.0 - width) / 7.0).clamp(0.0, 1.0));
    }
    let blur_combined = parts.iter().sum::<f64>() / parts.len().max(1) as f64;

    let saliency = saliency_map(&arr);
    let salient_sharpness = saliency
        .as_ref()
        .and_then(|smap| salient_region_sharpness(&arr, smap));
    let focus_ratio = saliency
        .as_ref()
        .and_then(|smap| saliency_focus_consistency(&arr, smap));
    let composition = saliency.as_ref().map(|smap| composition_score(&arr, smap));
    let exposure = nine_grid_exposure(&arr);
    let horizon_tilt_deg = horizon_tilt_degrees(&arr);
    let color_noise = color_noise_ratio(&rgb);
    let color_cast = color_cast_deviation(&rgb);
    let dynamic_range = brightness_dynamic_range(&arr);

    FastQualitySignals {
        width,
        height,
        file_size,
        blur_score: lap,
        brightness_mean,
        brightness_std,
        contrast_score,
        overexposed_ratio,
        underexposed_ratio,
        entropy,
        blur_combined,
        salient_sharpness,
        motion_anisotropy,
        edge_width_pix: edge_width,
        focus_ratio,
        horizon_tilt_deg,
        composition,
        worst_clip_dark: exposure.worst_clip_dark,
        worst_clip_bright: exposure.worst_clip_bright,
        color_noise,
        color_cast,
        dynamic_range,
    }
}

fn color_noise_ratio(rgb: &image::RgbImage) -> Option<f64> {
    let (w, h) = rgb.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    let work = expert_vision::pillow_thumbnail_luma(&util::pillow_luma(rgb), 384, 384);
    let (sw, sh) = work.dimensions();
    if sw < 8 || sh < 8 {
        return None;
    }
    let mut luma_vals = Vec::with_capacity((sw * sh) as usize);
    let mut chroma_diffs = Vec::with_capacity((sw * sh) as usize);
    for y in 0..sh {
        for x in 0..sw {
            let l = work.get_pixel(x, y)[0] as f64;
            luma_vals.push(l);
            if x > 0 {
                let prev = work.get_pixel(x - 1, y)[0] as f64;
                chroma_diffs.push((l - prev).abs());
            }
        }
    }
    let luma_std = {
        let mean = luma_vals.iter().sum::<f64>() / luma_vals.len() as f64;
        (luma_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / luma_vals.len() as f64).sqrt()
    };
    if luma_std < 1.0 {
        return Some(0.0);
    }
    let chroma_std = {
        let mean = chroma_diffs.iter().sum::<f64>() / chroma_diffs.len().max(1) as f64;
        (chroma_diffs.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / chroma_diffs.len().max(1) as f64)
            .sqrt()
    };
    Some(chroma_std / luma_std)
}

fn color_cast_deviation(rgb: &image::RgbImage) -> Option<f64> {
    let (w, h) = rgb.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut count = 0.0f64;
    let step = ((w * h / 10_000).max(1)) as u32;
    for y in (0..h).step_by(step as usize) {
        for x in (0..w).step_by(step as usize) {
            let p = rgb.get_pixel(x, y);
            r_sum += f64::from(p[0]);
            g_sum += f64::from(p[1]);
            b_sum += f64::from(p[2]);
            count += 1.0;
        }
    }
    if count < 1.0 {
        return None;
    }
    let r_mean = r_sum / count;
    let g_mean = g_sum / count;
    let b_mean = b_sum / count;
    let gray = (r_mean + g_mean + b_mean) / 3.0;
    Some(((r_mean - gray).powi(2) + (g_mean - gray).powi(2) + (b_mean - gray).powi(2)).sqrt())
}

fn brightness_dynamic_range(arr: &GrayMatrix) -> Option<f64> {
    if arr.len() < 100 {
        return None;
    }
    let mut sorted = arr.data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = sorted[(sorted.len() as f64 * 0.05) as usize];
    let hi = sorted[((sorted.len() - 1) as f64 * 0.95) as usize];
    Some(hi - lo)
}

pub(crate) fn matrix_entropy(arr: &GrayMatrix) -> f64 {
    let mut hist = [0usize; 256];
    for value in &arr.data {
        let idx = value.round().clamp(0.0, 255.0) as usize;
        hist[idx] += 1;
    }
    let total = hist.iter().sum::<usize>();
    if total == 0 {
        return 0.0;
    }
    hist.iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let p = *count as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

fn laplacian_variance(arr: &GrayMatrix) -> f64 {
    if arr.height < 3 || arr.width < 3 {
        return 0.0;
    }
    let mut values = Vec::with_capacity((arr.width - 2) * (arr.height - 2));
    for y in 1..arr.height - 1 {
        for x in 1..arr.width - 1 {
            let lap = arr.get(x, y) * 4.0
                - arr.get(x, y - 1)
                - arr.get(x, y + 1)
                - arr.get(x - 1, y)
                - arr.get(x + 1, y);
            values.push(lap);
        }
    }
    variance(&values)
}

fn tenengrad(arr: &GrayMatrix) -> f64 {
    if arr.height < 3 || arr.width < 3 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for y in 1..arr.height - 1 {
        for x in 1..arr.width - 1 {
            let gx = arr.get(x + 1, y) - arr.get(x - 1, y);
            let gy = arr.get(x, y + 1) - arr.get(x, y - 1);
            total += gx * gx + gy * gy;
            count += 1;
        }
    }
    total / count.max(1) as f64
}

fn fft_high_freq_ratio(arr: &GrayMatrix) -> (f64, f64) {
    use rustfft::{num_complex::Complex, FftPlanner};

    if arr.height < 16 || arr.width < 16 {
        return (0.0, 0.0);
    }
    let side = arr.height.min(arr.width);
    let y0 = (arr.height - side) / 2;
    let x0 = (arr.width - side) / 2;
    let mut crop = GrayMatrix::zeros(side, side);
    for y in 0..side {
        for x in 0..side {
            crop.set(x, y, arr.get(x0 + x, y0 + y));
        }
    }
    if side > 256 {
        crop = resize_area_matrix(&crop, 256, 256);
    }
    let side = crop.width;
    let mean = crop.mean();
    let hann = hann_window(side);
    let mut spec = vec![Complex::new(0.0, 0.0); side * side];
    for y in 0..side {
        for x in 0..side {
            spec[y * side + x] = Complex::new((crop.get(x, y) - mean) * hann[y] * hann[x], 0.0);
        }
    }
    fft2_in_place(&mut spec, side, side, false, &mut FftPlanner::new());

    let mut mag = vec![0.0; side * side];
    for y in 0..side {
        for x in 0..side {
            let src_y = (y + side / 2) % side;
            let src_x = (x + side / 2) % side;
            mag[y * side + x] = spec[src_y * side + src_x].norm();
        }
    }
    mag[(side / 2) * side + side / 2] = 0.0;

    let center = side as f64 / 2.0;
    let r_max = side as f64 / 2.0;
    let mut total = 1e-8;
    let mut high = 0.0;
    let mut sums = [0.0f64; 12];
    let mut counts = [1e-6f64; 12];
    let mut band_count = 0usize;
    for y in 0..side {
        for x in 0..side {
            let dy = y as f64 - center;
            let dx = x as f64 - center;
            let r = (dy * dy + dx * dx).sqrt();
            let value = mag[y * side + x];
            total += value;
            if r > 0.30 * r_max {
                high += value;
            }
            if r > 0.10 * r_max && r < 0.50 * r_max {
                let mut theta = dy.atan2(dx);
                if theta < 0.0 {
                    theta += std::f64::consts::PI;
                }
                let bin = ((theta / std::f64::consts::PI * 12.0) as usize).min(11);
                sums[bin] += value;
                counts[bin] += 1.0;
                band_count += 1;
            }
        }
    }
    let high_ratio = high / total;
    if band_count < 50 {
        return (high_ratio, 0.0);
    }
    let avgs = sums
        .iter()
        .zip(counts.iter())
        .map(|(sum, count)| sum / count)
        .collect::<Vec<_>>();
    let max_avg = avgs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_avg = avgs.iter().copied().fold(f64::INFINITY, f64::min);
    let aniso = (max_avg - min_avg) / (max_avg + 1e-8);
    (high_ratio, aniso)
}

pub(crate) fn nine_grid_exposure(arr: &GrayMatrix) -> NineGridExposure {
    if arr.height < 9 || arr.width < 9 {
        return NineGridExposure {
            worst_clip_dark: 0.0,
            worst_clip_bright: 0.0,
        };
    }
    let ys = [0, arr.height / 3, 2 * arr.height / 3, arr.height];
    let xs = [0, arr.width / 3, 2 * arr.width / 3, arr.width];
    let mut worst_clip_dark = 0.0f64;
    let mut worst_clip_bright = 0.0f64;
    for gy in 0..3 {
        for gx in 0..3 {
            let mut dark = 0usize;
            let mut bright = 0usize;
            let mut count = 0usize;
            for y in ys[gy]..ys[gy + 1] {
                for x in xs[gx]..xs[gx + 1] {
                    let value = arr.get(x, y);
                    if value <= 8.0 {
                        dark += 1;
                    }
                    if value >= 247.0 {
                        bright += 1;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                worst_clip_dark = worst_clip_dark.max(dark as f64 / count as f64);
                worst_clip_bright = worst_clip_bright.max(bright as f64 / count as f64);
            }
        }
    }
    NineGridExposure {
        worst_clip_dark,
        worst_clip_bright,
    }
}

#[cfg(feature = "opencv-orb")]
fn edge_width_marziliano(arr: &GrayMatrix) -> Option<f64> {
    use opencv::{core, imgproc, prelude::*};

    if arr.height < 16 || arr.width < 16 {
        return None;
    }
    let bytes = arr
        .data
        .iter()
        .map(|v| v.round().clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    let src = core::Mat::from_slice(&bytes)
        .ok()?
        .reshape(1, arr.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut edges = core::Mat::default();
    imgproc::canny(&src, &mut edges, 50.0, 150.0, 3, false).ok()?;
    let edge_bytes = edges.data_typed::<u8>().ok()?;
    let points = edge_bytes
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            if *value > 0 {
                Some((idx / arr.width, idx % arr.width))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if points.len() < 50 {
        return None;
    }
    let mut widths = Vec::new();
    for (y, x) in points {
        if x < 3 || x > arr.width.saturating_sub(4) {
            continue;
        }
        let mut left = x;
        for k in 1..12 {
            if x < k + 1 {
                break;
            }
            if arr.get(x - k, y) >= arr.get(x - k + 1, y) {
                left = x - k;
            } else {
                break;
            }
        }
        let mut right = x;
        for k in 1..12 {
            if x + k > arr.width.saturating_sub(2) {
                break;
            }
            if arr.get(x + k, y) <= arr.get(x + k - 1, y) {
                right = x + k;
            } else {
                break;
            }
        }
        let width = right.saturating_sub(left);
        if (1..=25).contains(&width) {
            widths.push(width as f64);
        }
    }
    if widths.len() < 30 {
        None
    } else {
        Some(widths.iter().sum::<f64>() / widths.len() as f64)
    }
}

#[cfg(not(feature = "opencv-orb"))]
fn edge_width_marziliano(_arr: &GrayMatrix) -> Option<f64> {
    None
}

#[cfg(feature = "opencv-orb")]
fn horizon_tilt_degrees(arr: &GrayMatrix) -> Option<f64> {
    use opencv::{core, imgproc, prelude::*};

    if arr.height < 64 || arr.width < 64 {
        return None;
    }
    let bytes = arr
        .data
        .iter()
        .map(|v| v.round().clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    let src = core::Mat::from_slice(&bytes)
        .ok()?
        .reshape(1, arr.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut edges = core::Mat::default();
    imgproc::canny(&src, &mut edges, 50.0, 150.0, 3, false).ok()?;
    let min_len = 40.max((arr.height.min(arr.width) as f64 * 0.35) as i32);
    let mut lines = core::Vector::<core::Vec4i>::new();
    imgproc::hough_lines_p(
        &edges,
        &mut lines,
        1.0,
        std::f64::consts::PI / 180.0,
        80,
        min_len as f64,
        10.0,
    )
    .ok()?;
    if lines.is_empty() {
        return None;
    }
    let mut pairs = Vec::<(f64, f64)>::new();
    for line in lines.iter().take(200) {
        let dx = f64::from(line[2] - line[0]);
        let dy = f64::from(line[3] - line[1]);
        let length = dx.hypot(dy);
        if length < f64::from(min_len) {
            continue;
        }
        let mut theta = dy.atan2(dx).to_degrees();
        if theta > 90.0 {
            theta -= 180.0;
        } else if theta < -90.0 {
            theta += 180.0;
        }
        let dev = theta.abs().min((90.0 - theta.abs()).abs());
        pairs.push((dev, length));
    }
    if pairs.is_empty() {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_take = (pairs.len() / 3).max(1);
    let (weighted, total) = pairs
        .iter()
        .take(n_take)
        .fold((0.0, 0.0), |(s, w), (angle, weight)| {
            (s + angle * weight, w + weight)
        });
    Some(weighted / total.max(1e-8))
}

#[cfg(not(feature = "opencv-orb"))]
fn horizon_tilt_degrees(_arr: &GrayMatrix) -> Option<f64> {
    None
}

fn saliency_map(arr: &GrayMatrix) -> Option<GrayMatrix> {
    use rustfft::{num_complex::Complex, FftPlanner};

    if arr.height < 8 || arr.width < 8 {
        return None;
    }
    let small = resize_area_matrix(arr, 64, 64);
    let mut spectrum = small
        .data
        .iter()
        .map(|v| Complex::new(*v, 0.0))
        .collect::<Vec<_>>();
    fft2_in_place(&mut spectrum, 64, 64, false, &mut FftPlanner::new());
    let log_amp = GrayMatrix::new(
        64,
        64,
        spectrum
            .iter()
            .map(|value| (value.norm() + 1e-8).ln())
            .collect(),
    );
    let smooth =
        opencv_box_filter_3x3(&log_amp).unwrap_or_else(|| box_filter_3x3_reflect101(&log_amp));
    let mut recon = Vec::with_capacity(spectrum.len());
    for (idx, value) in spectrum.iter().enumerate() {
        let residual = log_amp.data[idx] - smooth.data[idx];
        recon.push(Complex::from_polar(residual.exp(), value.arg()));
    }
    fft2_in_place(&mut recon, 64, 64, true, &mut FftPlanner::new());
    let norm = (64 * 64) as f64;
    let squared = GrayMatrix::new(
        64,
        64,
        recon
            .iter()
            .map(|value| (value / norm).norm_sqr())
            .collect(),
    );
    let blurred =
        opencv_gaussian_blur(&squared, 9, 2.5).unwrap_or_else(|| gaussian_blur(&squared, 9, 2.5));
    let mut out = opencv_resize_linear_matrix(&blurred, arr.width, arr.height)
        .unwrap_or_else(|| resize_bilinear_matrix(&blurred, arr.width, arr.height));
    let min_v = out.data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = out.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_v - min_v < 1e-8 {
        return None;
    }
    for value in &mut out.data {
        *value = (*value - min_v) / (max_v - min_v);
    }
    if out.std() < 0.01 {
        None
    } else {
        Some(out)
    }
}

fn salient_region_sharpness(arr: &GrayMatrix, smap: &GrayMatrix) -> Option<f64> {
    if arr.height < 16 || arr.width < 16 {
        return None;
    }
    if smap.std() < 0.01 {
        return None;
    }
    let smap = if smap.width == arr.width && smap.height == arr.height {
        smap.clone()
    } else {
        resize_bilinear_matrix(smap, arr.width, arr.height)
    };
    let threshold = quantile(&smap.data, 0.80)?;
    let mut selected = Vec::new();
    for y in 1..arr.height - 1 {
        for x in 1..arr.width - 1 {
            if smap.get(x, y) >= threshold {
                let lap = arr.get(x, y) * 4.0
                    - arr.get(x, y - 1)
                    - arr.get(x, y + 1)
                    - arr.get(x - 1, y)
                    - arr.get(x + 1, y);
                selected.push(lap);
            }
        }
    }
    if selected.len() < 100 {
        None
    } else {
        Some(variance(&selected))
    }
}

fn saliency_focus_consistency(arr: &GrayMatrix, smap: &GrayMatrix) -> Option<f64> {
    if arr.height < 32 || arr.width < 32 {
        return None;
    }
    let sub_thr = quantile(&smap.data, 0.80)?;
    let bg_thr = quantile(&smap.data, 0.30)?;
    let mut sub_lap = Vec::new();
    let mut bg_lap = Vec::new();
    for y in 1..arr.height - 1 {
        for x in 1..arr.width - 1 {
            let sal = smap.get(x, y);
            let lap = arr.get(x, y) * 4.0
                - arr.get(x, y - 1)
                - arr.get(x, y + 1)
                - arr.get(x - 1, y)
                - arr.get(x + 1, y);
            if sal >= sub_thr {
                sub_lap.push(lap);
            }
            if sal <= bg_thr {
                bg_lap.push(lap);
            }
        }
    }
    if sub_lap.len() < 100 || bg_lap.len() < 100 {
        None
    } else {
        Some(variance(&sub_lap) / (variance(&bg_lap) + 1e-6))
    }
}

fn composition_score(arr: &GrayMatrix, smap: &GrayMatrix) -> f64 {
    let total = smap.data.iter().sum::<f64>() + 1e-8;
    if total < 1e-3 {
        return 0.4;
    }
    let mut cx_sum = 0.0;
    let mut cy_sum = 0.0;
    for y in 0..smap.height {
        for x in 0..smap.width {
            let value = smap.get(x, y);
            cx_sum += x as f64 * value;
            cy_sum += y as f64 * value;
        }
    }
    let cy = cy_sum / total / (arr.height.saturating_sub(1).max(1) as f64);
    let cx = cx_sum / total / (arr.width.saturating_sub(1).max(1) as f64);
    let grid_pts = [
        (1.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
    ];
    let d_grid = grid_pts
        .iter()
        .map(|(py, px)| (cy - py).hypot(cx - px))
        .fold(f64::INFINITY, f64::min);
    let d_center = (cy - 0.5).hypot(cx - 0.5);
    let pos_score = (1.0 - d_grid.min(d_center) / 0.35).max(0.0);

    let threshold = quantile(&smap.data, 0.80).unwrap_or(0.0);
    let mut subject = 0usize;
    let mut edge_subject = 0usize;
    let edge_h = (arr.height as f64 * 0.05) as usize;
    let edge_w = (arr.width as f64 * 0.05) as usize;
    let edge_h = edge_h.max(2);
    let edge_w = edge_w.max(2);
    for y in 0..smap.height {
        for x in 0..smap.width {
            if smap.get(x, y) >= threshold {
                subject += 1;
                if y < edge_h
                    || y >= smap.height.saturating_sub(edge_h)
                    || x < edge_w
                    || x >= smap.width.saturating_sub(edge_w)
                {
                    edge_subject += 1;
                }
            }
        }
    }
    let frac = subject as f64 / smap.len().max(1) as f64;
    let size_score = if frac < 0.04 {
        frac / 0.04
    } else if frac > 0.55 {
        (1.0 - (frac - 0.55) / 0.45).max(0.0)
    } else {
        1.0
    };
    let edge_frac = edge_subject as f64 / (subject as f64 + 1e-6);
    let edge_score = if edge_frac < 0.25 {
        1.0
    } else {
        (1.0 - (edge_frac - 0.25) / 0.5).max(0.0)
    };
    0.5 * pos_score + 0.3 * size_score + 0.2 * edge_score
}

pub(crate) fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

pub(crate) fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = (sorted.len() - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Some(sorted[lo])
    } else {
        let weight = pos - lo as f64;
        Some(sorted[lo] * (1.0 - weight) + sorted[hi] * weight)
    }
}

fn hann_window(size: usize) -> Vec<f64> {
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())
        .collect()
}

fn fft2_in_place(
    data: &mut [rustfft::num_complex::Complex<f64>],
    width: usize,
    height: usize,
    inverse: bool,
    planner: &mut rustfft::FftPlanner<f64>,
) {
    let row_fft = if inverse {
        planner.plan_fft_inverse(width)
    } else {
        planner.plan_fft_forward(width)
    };
    for y in 0..height {
        row_fft.process(&mut data[y * width..(y + 1) * width]);
    }
    let col_fft = if inverse {
        planner.plan_fft_inverse(height)
    } else {
        planner.plan_fft_forward(height)
    };
    let mut column = vec![rustfft::num_complex::Complex::new(0.0, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            column[y] = data[y * width + x];
        }
        col_fft.process(&mut column);
        for y in 0..height {
            data[y * width + x] = column[y];
        }
    }
}

fn box_filter_3x3_reflect101(src: &GrayMatrix) -> GrayMatrix {
    let mut out = GrayMatrix::zeros(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let mut sum = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let sx = reflect101(x as isize + dx, src.width);
                    let sy = reflect101(y as isize + dy, src.height);
                    sum += src.get(sx, sy);
                }
            }
            out.set(x, y, sum / 9.0);
        }
    }
    out
}

#[cfg(feature = "opencv-orb")]
fn opencv_box_filter_3x3(src: &GrayMatrix) -> Option<GrayMatrix> {
    use opencv::{core, imgproc, prelude::*};

    let input = src.data.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let mat = core::Mat::from_slice(&input)
        .ok()?
        .reshape(1, src.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let kernel_data = vec![1.0f32 / 9.0; 9];
    let kernel = core::Mat::from_slice(&kernel_data)
        .ok()?
        .reshape(1, 3)
        .ok()?
        .try_clone()
        .ok()?;
    let mut dst = core::Mat::default();
    imgproc::filter_2d(
        &mat,
        &mut dst,
        -1,
        &kernel,
        core::Point::new(-1, -1),
        0.0,
        core::BORDER_DEFAULT,
    )
    .ok()?;
    let data = dst.data_typed::<f32>().ok()?;
    Some(GrayMatrix::new(
        src.width,
        src.height,
        data.iter().map(|v| f64::from(*v)).collect(),
    ))
}

#[cfg(not(feature = "opencv-orb"))]
fn opencv_box_filter_3x3(_src: &GrayMatrix) -> Option<GrayMatrix> {
    None
}

fn gaussian_blur(src: &GrayMatrix, kernel_size: usize, sigma: f64) -> GrayMatrix {
    let radius = kernel_size / 2;
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as isize - radius as isize;
        let value = (-(x * x) as f64 / (2.0 * sigma * sigma)).exp();
        kernel.push(value);
        sum += value;
    }
    for value in &mut kernel {
        *value /= sum;
    }
    let mut tmp = GrayMatrix::zeros(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let mut value = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let sx = reflect101(x as isize + k as isize - radius as isize, src.width);
                value += src.get(sx, y) * weight;
            }
            tmp.set(x, y, value);
        }
    }
    let mut out = GrayMatrix::zeros(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let mut value = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let sy = reflect101(y as isize + k as isize - radius as isize, src.height);
                value += tmp.get(x, sy) * weight;
            }
            out.set(x, y, value);
        }
    }
    out
}

#[cfg(feature = "opencv-orb")]
fn opencv_gaussian_blur(src: &GrayMatrix, kernel_size: usize, sigma: f64) -> Option<GrayMatrix> {
    use opencv::{core, imgproc, prelude::*};

    let input = src.data.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let mat = core::Mat::from_slice(&input)
        .ok()?
        .reshape(1, src.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut dst = core::Mat::default();
    imgproc::gaussian_blur(
        &mat,
        &mut dst,
        core::Size::new(kernel_size as i32, kernel_size as i32),
        sigma,
        sigma,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .ok()?;
    let data = dst.data_typed::<f32>().ok()?;
    Some(GrayMatrix::new(
        src.width,
        src.height,
        data.iter().map(|v| f64::from(*v)).collect(),
    ))
}

#[cfg(not(feature = "opencv-orb"))]
fn opencv_gaussian_blur(_src: &GrayMatrix, _kernel_size: usize, _sigma: f64) -> Option<GrayMatrix> {
    None
}

fn reflect101(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let mut idx = idx;
    let len_i = len as isize;
    while idx < 0 || idx >= len_i {
        if idx < 0 {
            idx = -idx;
        } else {
            idx = 2 * len_i - idx - 2;
        }
    }
    idx as usize
}

#[cfg(feature = "opencv-orb")]
pub(crate) fn resize_area_matrix(src: &GrayMatrix, width: usize, height: usize) -> GrayMatrix {
    opencv_resize_matrix(src, width, height, opencv::imgproc::INTER_AREA)
        .unwrap_or_else(|| resize_bilinear_matrix(src, width, height))
}

#[cfg(not(feature = "opencv-orb"))]
pub(crate) fn resize_area_matrix(src: &GrayMatrix, width: usize, height: usize) -> GrayMatrix {
    resize_bilinear_matrix(src, width, height)
}

#[cfg(feature = "opencv-orb")]
fn opencv_resize_linear_matrix(
    src: &GrayMatrix,
    width: usize,
    height: usize,
) -> Option<GrayMatrix> {
    opencv_resize_matrix(src, width, height, opencv::imgproc::INTER_LINEAR)
}

#[cfg(not(feature = "opencv-orb"))]
fn opencv_resize_linear_matrix(
    _src: &GrayMatrix,
    _width: usize,
    _height: usize,
) -> Option<GrayMatrix> {
    None
}

#[cfg(feature = "opencv-orb")]
fn opencv_resize_matrix(
    src: &GrayMatrix,
    width: usize,
    height: usize,
    interpolation: i32,
) -> Option<GrayMatrix> {
    use opencv::{core, imgproc, prelude::*};

    let input = src.data.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let mat = core::Mat::from_slice(&input)
        .ok()?
        .reshape(1, src.height as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut dst = core::Mat::default();
    imgproc::resize(
        &mat,
        &mut dst,
        core::Size::new(width as i32, height as i32),
        0.0,
        0.0,
        interpolation,
    )
    .ok()?;
    let data = dst.data_typed::<f32>().ok()?;
    Some(GrayMatrix::new(
        width,
        height,
        data.iter().map(|v| f64::from(*v)).collect(),
    ))
}

pub(crate) fn resize_bilinear_matrix(src: &GrayMatrix, width: usize, height: usize) -> GrayMatrix {
    if width == 0 || height == 0 || src.width == 0 || src.height == 0 {
        return GrayMatrix::zeros(width, height);
    }
    if src.width == width && src.height == height {
        return src.clone();
    }
    let mut out = GrayMatrix::zeros(width, height);
    let scale_x = src.width as f64 / width as f64;
    let scale_y = src.height as f64 / height as f64;
    for y in 0..height {
        let fy = (y as f64 + 0.5) * scale_y - 0.5;
        let y0 = fy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(src.height - 1);
        let wy = fy - y0 as f64;
        for x in 0..width {
            let fx = (x as f64 + 0.5) * scale_x - 0.5;
            let x0 = fx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(src.width - 1);
            let wx = fx - x0 as f64;
            let top = src.get(x0, y0) * (1.0 - wx) + src.get(x1, y0) * wx;
            let bottom = src.get(x0, y1) * (1.0 - wx) + src.get(x1, y1) * wx;
            out.set(x, y, top * (1.0 - wy) + bottom * wy);
        }
    }
    out
}

pub(crate) fn compute_color_hist(img: &DynamicImage) -> Option<Vec<f32>> {
    #[cfg(feature = "opencv-orb")]
    {
        if let Some(hist) = compute_color_hist_opencv(img) {
            return Some(hist);
        }
    }
    let rgb = img.resize(384, 384, FilterType::Triangle).to_rgb8();
    let (w, h) = rgb.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    let mut feats = Vec::with_capacity(144);
    for gy in 0..3 {
        for gx in 0..3 {
            let x0 = gx * w / 3;
            let x1 = (gx + 1) * w / 3;
            let y0 = gy * h / 3;
            let y1 = (gy + 1) * h / 3;
            let mut hh = [0f32; 16];
            let mut ss = [0f32; 16];
            let mut vv = [0f32; 16];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = rgb.get_pixel(x, y);
                    let (hue, sat, val) = rgb_to_hsv(p[0], p[1], p[2]);
                    hh[((hue / 180.0 * 16.0).floor() as usize).min(15)] += 1.0;
                    ss[((sat * 16.0).floor() as usize).min(15)] += 1.0;
                    vv[((val * 16.0).floor() as usize).min(15)] += 1.0;
                }
            }
            normalize_sum(&mut hh);
            normalize_sum(&mut ss);
            normalize_sum(&mut vv);
            feats.extend(hh);
            feats.extend(ss);
            feats.extend(vv);
        }
    }
    let norm = feats.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return None;
    }
    for v in &mut feats {
        *v /= norm;
    }
    Some(feats)
}

#[cfg(feature = "opencv-orb")]
fn compute_color_hist_opencv(img: &DynamicImage) -> Option<Vec<f32>> {
    use opencv::{core, imgproc, prelude::*};

    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    let src = core::Mat::from_slice(rgb.as_raw())
        .ok()?
        .reshape(3, h as i32)
        .ok()?
        .try_clone()
        .ok()?;
    let mut work = src;
    let mut work_w = w;
    let mut work_h = h;
    if w.max(h) > 384 {
        let scale = 384.0f64 / w.max(h) as f64;
        work_w = ((w as f64 * scale) as i32).max(1) as u32;
        work_h = ((h as f64 * scale) as i32).max(1) as u32;
        let mut resized = core::Mat::default();
        imgproc::resize(
            &work,
            &mut resized,
            core::Size::new(work_w as i32, work_h as i32),
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )
        .ok()?;
        work = resized;
    }
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(
        &work,
        &mut hsv,
        imgproc::COLOR_RGB2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .ok()?;
    let mut feats = Vec::with_capacity(144);
    for gy in 0..3 {
        for gx in 0..3 {
            let x0 = gx * work_w / 3;
            let x1 = (gx + 1) * work_w / 3;
            let y0 = gy * work_h / 3;
            let y1 = (gy + 1) * work_h / 3;
            let mut hh = [0f32; 16];
            let mut ss = [0f32; 16];
            let mut vv = [0f32; 16];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = *hsv.at_2d::<core::Vec3b>(y as i32, x as i32).ok()?;
                    hh[((p[0] as usize * 16) / 180).min(15)] += 1.0;
                    ss[((p[1] as usize * 16) / 256).min(15)] += 1.0;
                    vv[((p[2] as usize * 16) / 256).min(15)] += 1.0;
                }
            }
            normalize_sum(&mut hh);
            normalize_sum(&mut ss);
            normalize_sum(&mut vv);
            feats.extend(hh);
            feats.extend(ss);
            feats.extend(vv);
        }
    }
    let norm = feats.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return None;
    }
    for v in &mut feats {
        *v /= norm;
    }
    Some(feats)
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = f32::from(r) / 255.0;
    let g = f32::from(g) / 255.0;
    let b = f32::from(b) / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let hue = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let hue = if hue < 0.0 { hue + 360.0 } else { hue } / 2.0;
    let sat = if max == 0.0 { 0.0 } else { delta / max };
    (hue, sat, max)
}

fn normalize_sum(values: &mut [f32]) {
    let sum = values.iter().sum::<f32>();
    if sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_matrix_mean_and_std() {
        let m = GrayMatrix::new(2, 2, vec![0.0, 2.0, 4.0, 6.0]);
        assert!((m.mean() - 3.0).abs() < 1e-10);
        let expected_std = (5.0f64).sqrt();
        assert!((m.std() - expected_std).abs() < 1e-10);
    }

    #[test]
    fn gray_matrix_ratio_le_ge() {
        let m = GrayMatrix::new(4, 1, vec![5.0, 10.0, 240.0, 250.0]);
        assert!((m.ratio_le(8.0) - 0.25).abs() < 1e-10);
        assert!((m.ratio_ge(247.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn laplacian_variance_uniform_is_zero() {
        let m = GrayMatrix::new(5, 5, vec![100.0; 25]);
        assert!((laplacian_variance(&m)).abs() < 1e-10);
    }

    #[test]
    fn tenengrad_uniform_is_zero() {
        let m = GrayMatrix::new(5, 5, vec![100.0; 25]);
        assert!((tenengrad(&m)).abs() < 1e-10);
    }

    #[test]
    fn matrix_entropy_uniform_distribution() {
        let mut data = Vec::new();
        for i in 0..256 {
            data.push(i as f64);
        }
        let m = GrayMatrix::new(256, 1, data);
        let e = matrix_entropy(&m);
        assert!(
            (e - 8.0).abs() < 0.01,
            "entropy of uniform 256 bins should be ~8.0, got {e}"
        );
    }

    #[test]
    fn matrix_entropy_constant_is_zero() {
        let m = GrayMatrix::new(10, 10, vec![128.0; 100]);
        assert!((matrix_entropy(&m)).abs() < 1e-10);
    }

    #[test]
    fn variance_constant_is_zero() {
        assert!((variance(&[5.0, 5.0, 5.0])).abs() < 1e-10);
    }

    #[test]
    fn variance_known_values() {
        let v = variance(&[1.0, 2.0, 3.0]);
        let expected = 2.0 / 3.0;
        assert!((v - expected).abs() < 1e-10);
    }

    #[test]
    fn quantile_basic() {
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5), Some(3.0));
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0), Some(1.0));
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 1.0), Some(5.0));
        assert_eq!(quantile(&[], 0.5), None);
    }

    #[test]
    fn center_crop_preserves_center() {
        let m = GrayMatrix::new(4, 4, (0..16).map(|i| i as f64).collect());
        let cropped = m.center_crop(0.5);
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
        assert!((cropped.get(0, 0) - m.get(1, 1)).abs() < 1e-10);
    }

    #[test]
    fn rgb_to_hsv_red() {
        let (h, s, v) = rgb_to_hsv(255, 0, 0);
        assert!((h).abs() < 0.01);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn rgb_to_hsv_black() {
        let (h, s, v) = rgb_to_hsv(0, 0, 0);
        assert!((h).abs() < 0.01);
        assert!((s).abs() < 0.01);
        assert!((v).abs() < 0.01);
    }
}
