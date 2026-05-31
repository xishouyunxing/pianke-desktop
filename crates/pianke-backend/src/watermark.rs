use ab_glyph::{FontArc, PxScale};
use exif;
use image::{
    imageops::{self, FilterType},
    DynamicImage, ImageBuffer, Rgba, RgbaImage,
};
use imageproc::{
    drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_line_segment_mut, draw_text_mut},
    rect::Rect,
};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
};

static FONT_CACHE: OnceCell<Option<FontArc>> = OnceCell::new();

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WatermarkConfig {
    #[serde(default = "default_template")]
    pub template: String,
    #[serde(default)]
    pub preview_index: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExifInfo {
    pub make: String,
    pub model: String,
    pub lens: String,
    pub focal_length: String,
    pub f_number: String,
    pub exposure: String,
    pub iso: String,
    pub datetime: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemplateView {
    pub id: String,
    pub name: String,
    pub desc: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogoView {
    pub name: String,
    pub file: String,
    pub size_kb: f64,
}

fn default_template() -> String {
    "A".to_string()
}

pub fn list_templates() -> Vec<TemplateView> {
    vec![
        TemplateView {
            id: "A".into(),
            name: "标准底栏".into(),
            desc: "完整底栏 + 品牌 + 参数".into(),
        },
        TemplateView {
            id: "B_full".into(),
            name: "极简底栏-详细".into(),
            desc: "更轻的底栏，保留参数".into(),
        },
        TemplateView {
            id: "B_clean".into(),
            name: "极简底栏-极简".into(),
            desc: "只保留品牌和型号".into(),
        },
        TemplateView {
            id: "C_full".into(),
            name: "毛玻璃悬浮-详细".into(),
            desc: "半透明悬浮块 + 参数".into(),
        },
        TemplateView {
            id: "C_clean".into(),
            name: "毛玻璃悬浮-极简".into(),
            desc: "半透明悬浮块 + 品牌".into(),
        },
        TemplateView {
            id: "D_full".into(),
            name: "经典白边相框-详细".into(),
            desc: "白边相框 + 参数".into(),
        },
        TemplateView {
            id: "D_clean".into(),
            name: "经典白边相框-极简".into(),
            desc: "白边相框 + 品牌".into(),
        },
        TemplateView {
            id: "F_full".into(),
            name: "杂志风-详细".into(),
            desc: "大图 + 侧边色块 + 参数".into(),
        },
        TemplateView {
            id: "F_clean".into(),
            name: "杂志风-极简".into(),
            desc: "大图 + 侧边色块 + 品牌".into(),
        },
        TemplateView {
            id: "G".into(),
            name: "极简白边".into(),
            desc: "只有细边和品牌".into(),
        },
        TemplateView {
            id: "H".into(),
            name: "相机回放".into(),
            desc: "相机背屏风格的玩味样式".into(),
        },
    ]
}

pub fn available_logos(root_dir: &Path) -> Vec<LogoView> {
    let logos_dir = root_dir.join("assets").join("logos");
    let Ok(entries) = fs::read_dir(logos_dir) else {
        return vec![];
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase()
            != "png"
        {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        out.push(LogoView {
            name: path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string(),
            file: path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string(),
            size_kb: (meta.len() as f64 / 1024.0 * 10.0).round() / 10.0,
        });
    }
    out.sort_by(|a, b| a.file.cmp(&b.file));
    out
}

pub fn parse_exif(path: &Path) -> ExifInfo {
    let mut info = ExifInfo {
        make: String::new(),
        model: String::new(),
        lens: String::new(),
        focal_length: String::new(),
        f_number: String::new(),
        exposure: String::new(),
        iso: String::new(),
        datetime: String::new(),
    };
    let Ok(bytes) = fs::read(path) else {
        return info;
    };
    let mut cursor = Cursor::new(bytes);
    let Ok(exif) = exif::Reader::new().read_from_container(&mut cursor) else {
        return info;
    };
    for field in exif.fields() {
        let value = field.display_value().with_unit(&exif).to_string();
        match field.tag {
            exif::Tag::Make => info.make = clean(&value),
            exif::Tag::Model => info.model = clean(&value),
            exif::Tag::LensModel => info.lens = clean(&value),
            exif::Tag::FocalLength => info.focal_length = clean(&value),
            exif::Tag::FNumber => info.f_number = clean(&value),
            exif::Tag::ExposureTime => info.exposure = clean(&value),
            exif::Tag::PhotographicSensitivity | exif::Tag::ISOSpeed => info.iso = clean(&value),
            exif::Tag::DateTimeOriginal | exif::Tag::DateTime => info.datetime = clean(&value),
            _ => {}
        }
    }
    info
}

pub fn render(
    root_dir: &Path,
    img_path: &Path,
    cfg: &WatermarkConfig,
    preview_max_side: Option<u32>,
) -> Result<Vec<u8>, String> {
    let src = image::open(img_path).map_err(|e| format!("load image failed: {e}"))?;
    let exif = parse_exif(img_path);
    let mut img = src.to_rgb8();
    if let Some(max_side) = preview_max_side {
        let (w, h) = img.dimensions();
        if w.max(h) > max_side {
            let scale = max_side as f32 / w.max(h) as f32;
            img = imageops::resize(
                &img,
                (w as f32 * scale).round().max(1.0) as u32,
                (h as f32 * scale).round().max(1.0) as u32,
                FilterType::Lanczos3,
            );
        }
    }
    let canvas = render_template(root_dir, &img, &exif, &cfg.template);
    encode_jpeg(&canvas)
}

pub fn batch_export(
    root_dir: &Path,
    src_paths: &[String],
    dst_dir: &Path,
    cfg: &WatermarkConfig,
    mut progress_cb: Option<&mut dyn FnMut(usize, usize, String)>,
    mut cancel_check: Option<&mut dyn FnMut() -> bool>,
) -> Result<Value, String> {
    fs::create_dir_all(dst_dir).map_err(|e| format!("create output dir failed: {e}"))?;
    let total = src_paths.len();
    let mut ok = 0usize;
    let mut failed = Vec::new();
    for (i, src) in src_paths.iter().enumerate() {
        if cancel_check.as_mut().is_some_and(|f| f()) {
            break;
        }
        let src_path = Path::new(src);
        match render(root_dir, src_path, cfg, None) {
            Ok(data) => {
                let out = dst_dir.join(
                    src_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("image")
                        .to_string()
                        + ".jpg",
                );
                fs::write(&out, data).map_err(|e| format!("write output failed: {e}"))?;
                ok += 1;
            }
            Err(err) => failed.push((
                src_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string(),
                err,
            )),
        }
        if let Some(cb) = progress_cb.as_mut() {
            cb(
                i + 1,
                total,
                Path::new(src)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string(),
            );
        }
    }
    Ok(json!({"ok": ok, "failed": failed, "total": total}))
}

fn render_template(
    root_dir: &Path,
    img: &image::RgbImage,
    exif: &ExifInfo,
    template: &str,
) -> RgbaImage {
    let style = template_family(template);
    match style {
        TemplateFamily::A => render_bottom_bar(root_dir, img, exif, 0.13, true, true, false),
        TemplateFamily::B => render_bottom_bar(
            root_dir,
            img,
            exif,
            0.10,
            true,
            !template.ends_with("_clean"),
            false,
        ),
        TemplateFamily::C => render_glass_overlay(root_dir, img, exif, !template.ends_with("_clean")),
        TemplateFamily::D => render_framed(root_dir, img, exif, !template.ends_with("_clean")),
        TemplateFamily::F => render_magazine(root_dir, img, exif, !template.ends_with("_clean")),
        TemplateFamily::G => render_minimal(root_dir, img, exif),
        TemplateFamily::H => render_camera_replay(root_dir, img, exif),
    }
}

enum TemplateFamily {
    A,
    B,
    C,
    D,
    F,
    G,
    H,
}

fn template_family(template: &str) -> TemplateFamily {
    let base = template.split('_').next().unwrap_or(template);
    match base {
        "A" => TemplateFamily::A,
        "B" => TemplateFamily::B,
        "C" => TemplateFamily::C,
        "D" => TemplateFamily::D,
        "F" => TemplateFamily::F,
        "G" => TemplateFamily::G,
        "H" => TemplateFamily::H,
        _ => TemplateFamily::A,
    }
}

fn render_bottom_bar(
    root_dir: &Path,
    img: &image::RgbImage,
    exif: &ExifInfo,
    bar_ratio: f32,
    show_logo: bool,
    show_params: bool,
    dark: bool,
) -> RgbaImage {
    let (w, h) = img.dimensions();
    let bar_h = ((w as f32) * bar_ratio).max(88.0) as u32;
    let mut canvas = ImageBuffer::from_pixel(w, h + bar_h, Rgba([255, 255, 255, 255]));
    imageops::overlay(
        &mut canvas,
        &DynamicImage::ImageRgb8(img.clone()).to_rgba8(),
        0,
        0,
    );
    let bg = if dark {
        Rgba([28, 28, 30, 255])
    } else {
        Rgba([255, 255, 255, 255])
    };
    draw_filled_rect_mut(&mut canvas, Rect::at(0, h as i32).of_size(w, bar_h), bg);
    let fg = if dark {
        Rgba([255, 255, 255, 255])
    } else {
        Rgba([29, 29, 31, 255])
    };
    let sub = if dark {
        Rgba([210, 210, 215, 255])
    } else {
        Rgba([134, 134, 139, 255])
    };
    let pad = ((w as f32) * 0.04) as i32;
    let font = load_font();
    let scale_main = PxScale::from((bar_h as f32 * 0.22).clamp(16.0, 42.0));
    let scale_sub = PxScale::from((bar_h as f32 * 0.15).clamp(12.0, 28.0));
    let left_main = trim_value(&format!("{} {}", exif.make, exif.model));
    let left_sub = if show_params {
        trim_value(
            &[
                exif.lens.clone(),
                exif.focal_length.clone(),
                exif.f_number.clone(),
                exif.exposure.clone(),
                exif.iso.clone(),
            ]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("  "),
        )
    } else {
        String::new()
    };
    let y_base = h as i32 + (bar_h as i32 / 2);
    if let Some(font) = font.as_ref() {
        if !left_main.is_empty() {
            draw_text_mut(
                &mut canvas,
                fg,
                pad,
                y_base - (scale_main.y as i32),
                scale_main,
                font,
                &left_main,
            );
        }
        if !left_sub.is_empty() {
            draw_text_mut(
                &mut canvas,
                sub,
                pad,
                y_base + 4,
                scale_sub,
                font,
                &left_sub,
            );
        }
    }
    if show_logo {
        if let Some(logo) = load_logo(root_dir, &exif.make, (bar_h as f32 * 0.55) as u32) {
            let logo_w = logo.width();
            let x = w.saturating_sub(logo_w + pad as u32) as i64;
            let y = h as i64 + ((bar_h - logo.height()) / 2) as i64;
            imageops::overlay(&mut canvas, &logo, x, y);
        }
    }
    canvas
}

fn render_framed(
    root_dir: &Path,
    img: &image::RgbImage,
    exif: &ExifInfo,
    show_params: bool,
) -> RgbaImage {
    let (w, h) = img.dimensions();
    let frame = ((w.min(h) as f32) * 0.05).max(24.0) as u32;
    let mut canvas = ImageBuffer::from_pixel(
        w + frame * 2,
        h + frame * 2 + 72,
        Rgba([255, 255, 255, 255]),
    );
    imageops::overlay(
        &mut canvas,
        &DynamicImage::ImageRgb8(img.clone()).to_rgba8(),
        frame as i64,
        frame as i64,
    );
    let font = load_font();
    let scale = PxScale::from(((frame as f32) * 0.5).clamp(14.0, 34.0));
    let text = if show_params {
        trim_value(&format!(
            "{} {}  {}  {}",
            exif.make, exif.model, exif.f_number, exif.exposure
        ))
    } else {
        trim_value(&format!("{} {}", exif.make, exif.model))
    };
    if !text.is_empty() {
        if let Some(font) = font.as_ref() {
            draw_text_mut(
                &mut canvas,
                Rgba([29, 29, 31, 255]),
                frame as i32,
                (h + frame + 18) as i32,
                scale,
                font,
                &text,
            );
        }
    }
    if let Some(logo) = load_logo(root_dir, &exif.make, (frame as f32 * 0.9) as u32) {
        let x = canvas.width().saturating_sub(logo.width() + frame) as i64;
        imageops::overlay(&mut canvas, &logo, x, (h + frame + 8) as i64);
    }
    canvas
}

fn render_magazine(
    root_dir: &Path,
    img: &image::RgbImage,
    exif: &ExifInfo,
    show_params: bool,
) -> RgbaImage {
    let (w, h) = img.dimensions();
    let margin = ((w.min(h) as f32) * 0.075).clamp(46.0, 180.0) as u32;
    let side = ((w as f32) * 0.20).clamp(150.0, 360.0) as u32;
    let footer = ((h as f32) * 0.08).clamp(54.0, 140.0) as u32;
    let mut canvas = ImageBuffer::from_pixel(
        w + side + margin * 2,
        h + footer + margin * 2,
        Rgba([247, 244, 238, 255]),
    );
    let photo_x = margin;
    let photo_y = margin;
    imageops::overlay(
        &mut canvas,
        &DynamicImage::ImageRgb8(img.clone()).to_rgba8(),
        photo_x as i64,
        photo_y as i64,
    );
    let side_x = (photo_x + w + margin / 2) as i32;
    draw_filled_rect_mut(
        &mut canvas,
        Rect::at(side_x, photo_y as i32).of_size(side, h),
        Rgba([38, 35, 31, 255]),
    );
    let font = load_font();
    let scale_kicker = PxScale::from(((side as f32) * 0.08).clamp(13.0, 22.0));
    let scale_main = PxScale::from(((side as f32) * 0.17).clamp(24.0, 52.0));
    let scale_sub = PxScale::from(((side as f32) * 0.095).clamp(14.0, 26.0));
    let title = trim_value(&format!("{} {}", fallback(&exif.make, "CAMERA"), exif.model));
    let lens = fallback(&exif.lens, "LENS DATA");
    let body = param_line(exif);
    if let Some(font) = font.as_ref() {
        draw_text_mut(
            &mut canvas,
            Rgba([214, 177, 111, 255]),
            side_x + 24,
            photo_y as i32 + 28,
            scale_kicker,
            font,
            "PIANKE PHOTO",
        );
        draw_text_mut(
            &mut canvas,
            Rgba([250, 247, 240, 255]),
            side_x + 24,
            photo_y as i32 + 72,
            scale_main,
            font,
            &title,
        );
        if !body.is_empty() {
            draw_text_mut(
                &mut canvas,
                Rgba([210, 204, 192, 255]),
                side_x + 24,
                photo_y as i32 + 140,
                scale_sub,
                font,
                &body,
            );
        }
        if show_params {
            draw_text_mut(
                &mut canvas,
                Rgba([158, 151, 140, 255]),
                side_x + 24,
                photo_y as i32 + 190,
                scale_sub,
                font,
                &lens,
            );
        }
        draw_text_mut(
            &mut canvas,
            Rgba([78, 72, 64, 255]),
            margin as i32,
            (photo_y + h + 22) as i32,
            scale_sub,
            font,
            &trim_value(&format!("{}  {}", fallback(&exif.datetime, "PHOTO SERIES"), body)),
        );
    }
    draw_line_segment_mut(
        &mut canvas,
        (side_x as f32 + 24.0, (photo_y + h - 82) as f32),
        ((side_x + side as i32 - 24) as f32, (photo_y + h - 82) as f32),
        Rgba([104, 94, 80, 255]),
    );
    if let Some(logo) = load_logo(root_dir, &exif.make, 46) {
        imageops::overlay(
            &mut canvas,
            &logo,
            (side_x + 24) as i64,
            (photo_y + h - 62) as i64,
        );
    }
    canvas
}

fn render_minimal(root_dir: &Path, img: &image::RgbImage, exif: &ExifInfo) -> RgbaImage {
    let (w, h) = img.dimensions();
    let frame = 10u32;
    let mut canvas = ImageBuffer::from_pixel(
        w + frame * 2,
        h + frame * 2 + 24,
        Rgba([255, 255, 255, 255]),
    );
    imageops::overlay(
        &mut canvas,
        &DynamicImage::ImageRgb8(img.clone()).to_rgba8(),
        frame as i64,
        frame as i64,
    );
    let font = load_font();
    let scale = PxScale::from(14.0);
    let text = trim_value(&format!("{} {}", exif.make, exif.model));
    if !text.is_empty() {
        if let Some(font) = font.as_ref() {
            draw_text_mut(
                &mut canvas,
                Rgba([134, 134, 139, 255]),
                frame as i32,
                (h + frame + 4) as i32,
                scale,
                font,
                &text,
            );
        }
    }
    if let Some(logo) = load_logo(root_dir, &exif.make, 18) {
        let x = canvas.width().saturating_sub(logo.width() + frame) as i64;
        imageops::overlay(&mut canvas, &logo, x, (h + frame + 3) as i64);
    }
    canvas
}

fn render_camera_replay(root_dir: &Path, img: &image::RgbImage, exif: &ExifInfo) -> RgbaImage {
    let (w, h) = img.dimensions();
    let pad = ((w.min(h) as f32) * 0.055).clamp(28.0, 96.0) as u32;
    let band = (h as f32 * 0.18).max(120.0) as u32;
    let mut canvas = ImageBuffer::from_pixel(w + pad * 2, h + band + pad * 2, Rgba([13, 14, 14, 255]));
    let photo_x = pad;
    let photo_y = pad;
    imageops::overlay(
        &mut canvas,
        &DynamicImage::ImageRgb8(img.clone()).to_rgba8(),
        photo_x as i64,
        photo_y as i64,
    );
    let frame = Rect::at(photo_x as i32, photo_y as i32).of_size(w, h);
    draw_hollow_rect_mut(&mut canvas, frame, Rgba([235, 238, 230, 210]));
    let corner = (w.min(h) as f32 * 0.07).clamp(28.0, 86.0);
    let l = photo_x as f32;
    let t = photo_y as f32;
    let r = (photo_x + w) as f32;
    let b = (photo_y + h) as f32;
    let hud = Rgba([238, 242, 232, 230]);
    for (x0, y0, x1, y1) in [
        (l + 18.0, t + 18.0, l + 18.0 + corner, t + 18.0),
        (l + 18.0, t + 18.0, l + 18.0, t + 18.0 + corner),
        (r - 18.0 - corner, t + 18.0, r - 18.0, t + 18.0),
        (r - 18.0, t + 18.0, r - 18.0, t + 18.0 + corner),
        (l + 18.0, b - 18.0, l + 18.0 + corner, b - 18.0),
        (l + 18.0, b - 18.0 - corner, l + 18.0, b - 18.0),
        (r - 18.0 - corner, b - 18.0, r - 18.0, b - 18.0),
        (r - 18.0, b - 18.0 - corner, r - 18.0, b - 18.0),
    ] {
        draw_line_segment_mut(&mut canvas, (x0, y0), (x1, y1), hud);
    }
    let center_x = photo_x + w / 2;
    let center_y = photo_y + h / 2;
    draw_line_segment_mut(
        &mut canvas,
        ((center_x - 28) as f32, center_y as f32),
        ((center_x + 28) as f32, center_y as f32),
        Rgba([238, 242, 232, 150]),
    );
    draw_line_segment_mut(
        &mut canvas,
        (center_x as f32, (center_y - 28) as f32),
        (center_x as f32, (center_y + 28) as f32),
        Rgba([238, 242, 232, 150]),
    );
    draw_filled_rect_mut(
        &mut canvas,
        Rect::at((photo_x + 28) as i32, (photo_y + 28) as i32).of_size(16, 16),
        Rgba([224, 48, 40, 255]),
    );
    let font = load_font();
    let scale_small = PxScale::from((band as f32 * 0.14).clamp(14.0, 24.0));
    let scale_main = PxScale::from((band as f32 * 0.20).clamp(18.0, 34.0));
    let text = trim_value(&format!("{} {}", fallback(&exif.make, "CAMERA"), exif.model));
    if let Some(font) = font.as_ref() {
        draw_text_mut(
            &mut canvas,
            Rgba([244, 244, 238, 255]),
            (photo_x + 52) as i32,
            (photo_y + 22) as i32,
            scale_small,
            font,
            "REC",
        );
        draw_text_mut(
            &mut canvas,
            Rgba([244, 244, 238, 255]),
            pad as i32,
            (photo_y + h + 28) as i32,
            scale_main,
            font,
            &text,
        );
        draw_text_mut(
            &mut canvas,
            Rgba([178, 184, 172, 255]),
            pad as i32,
            (photo_y + h + 74) as i32,
            scale_small,
            font,
            &trim_value(&format!("{}   {}", param_line(exif), fallback(&exif.lens, ""))),
        );
        draw_text_mut(
            &mut canvas,
            Rgba([178, 184, 172, 255]),
            (photo_x + w.saturating_sub(150)) as i32,
            (photo_y + 22) as i32,
            scale_small,
            font,
            "PLAY  100-0001",
        );
    }
    if let Some(logo) = load_logo(root_dir, &exif.make, 44) {
        let x = canvas.width().saturating_sub(logo.width() + pad) as i64;
        imageops::overlay(&mut canvas, &logo, x, (photo_y + h + 28) as i64);
    }
    canvas
}

fn render_glass_overlay(
    root_dir: &Path,
    img: &image::RgbImage,
    exif: &ExifInfo,
    show_params: bool,
) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut canvas = DynamicImage::ImageRgb8(img.clone()).to_rgba8();
    let panel_w = ((w as f32) * 0.48).clamp(260.0, 720.0) as u32;
    let panel_h = if show_params {
        ((h as f32) * 0.18).clamp(116.0, 210.0) as u32
    } else {
        ((h as f32) * 0.13).clamp(86.0, 160.0) as u32
    };
    let margin = ((w.min(h) as f32) * 0.055).clamp(24.0, 90.0) as u32;
    let x = margin.min(w.saturating_sub(panel_w));
    let y = h.saturating_sub(panel_h + margin);
    let crop = imageops::crop_imm(&canvas, x, y, panel_w, panel_h).to_image();
    let blurred = DynamicImage::ImageRgba8(crop).blur(12.0).to_rgba8();
    imageops::overlay(&mut canvas, &blurred, x as i64, y as i64);
    draw_filled_rect_mut(
        &mut canvas,
        Rect::at(x as i32, y as i32).of_size(panel_w, panel_h),
        Rgba([255, 255, 255, 94]),
    );
    draw_hollow_rect_mut(
        &mut canvas,
        Rect::at(x as i32, y as i32).of_size(panel_w, panel_h),
        Rgba([255, 255, 255, 150]),
    );
    let font = load_font();
    let title = trim_value(&format!("{} {}", fallback(&exif.make, "CAMERA"), exif.model));
    let scale_main = PxScale::from((panel_h as f32 * 0.20).clamp(16.0, 36.0));
    let scale_sub = PxScale::from((panel_h as f32 * 0.13).clamp(12.0, 24.0));
    if let Some(font) = font.as_ref() {
        draw_text_mut(
            &mut canvas,
            Rgba([255, 255, 255, 245]),
            (x + 24) as i32,
            (y + 22) as i32,
            scale_main,
            font,
            &title,
        );
        if show_params {
            draw_text_mut(
                &mut canvas,
                Rgba([240, 240, 236, 220]),
                (x + 24) as i32,
                (y + panel_h / 2) as i32,
                scale_sub,
                font,
                &param_line(exif),
            );
            draw_text_mut(
                &mut canvas,
                Rgba([220, 224, 220, 205]),
                (x + 24) as i32,
                (y + panel_h / 2 + 32) as i32,
                scale_sub,
                font,
                &fallback(&exif.lens, ""),
            );
        }
    }
    if let Some(logo) = load_logo(root_dir, &exif.make, (panel_h as f32 * 0.34) as u32) {
        let lx = x + panel_w.saturating_sub(logo.width() + 24);
        let ly = y + (panel_h.saturating_sub(logo.height())) / 2;
        imageops::overlay(&mut canvas, &logo, lx as i64, ly as i64);
    }
    canvas
}

fn fallback(value: &str, default: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        default.to_string()
    } else {
        value.to_string()
    }
}

fn param_line(exif: &ExifInfo) -> String {
    trim_value(
        &[
            exif.focal_length.clone(),
            exif.f_number.clone(),
            exif.exposure.clone(),
            exif.iso.clone(),
        ]
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>()
        .join("  "),
    )
}

fn load_logo(root_dir: &Path, make: &str, target_h: u32) -> Option<RgbaImage> {
    let path = logo_path(root_dir, make)?;
    let img = image::open(path).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    if h == 0 || target_h == 0 {
        return Some(img);
    }
    let scale = target_h as f32 / h as f32;
    Some(imageops::resize(
        &img,
        (w as f32 * scale).round().max(1.0) as u32,
        target_h,
        FilterType::Lanczos3,
    ))
}

fn logo_path(root_dir: &Path, make: &str) -> Option<PathBuf> {
    let logos = root_dir.join("assets").join("logos");
    let lower = make.to_lowercase();
    let map = [
        ("fuji", "fujifilm.png"),
        ("fujifilm", "fujifilm.png"),
        ("canon", "canon.png"),
        ("nikon", "nikon.png"),
        ("sony", "sony.png"),
        ("leica", "leica_logo.png"),
        ("hasselblad", "hasselblad.png"),
        ("olympus", "olympus_blue_gold.png"),
        ("om digital", "olympus_blue_gold.png"),
        ("om system", "olympus_blue_gold.png"),
        ("panasonic", "panasonic.png"),
        ("pentax", "pentax.png"),
        ("ricoh", "ricoh.png"),
        ("apple", "apple.png"),
        ("xiaomi", "xmage.png"),
    ];
    for (needle, file) in map {
        if lower.contains(needle) {
            let p = logos.join(file);
            if p.exists() {
                return Some(p);
            }
        }
    }
    let default = logos.join("default.png");
    default.exists().then_some(default)
}

fn load_font() -> Option<FontArc> {
    FONT_CACHE
        .get_or_init(|| {
            let candidates = [
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\msyhbd.ttc",
                r"C:\Windows\Fonts\simhei.ttf",
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ];
            for candidate in candidates {
                if let Ok(bytes) = fs::read(candidate) {
                    if let Ok(font) = FontArc::try_from_vec(bytes) {
                        return Some(font);
                    }
                }
            }
            None
        })
        .clone()
}

fn encode_jpeg(img: &RgbaImage) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let rgb = DynamicImage::ImageRgba8(img.clone()).to_rgb8();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 92);
    encoder
        .encode_image(&rgb)
        .map_err(|e| format!("encode jpeg failed: {e}"))?;
    Ok(buf)
}

fn clean(value: &str) -> String {
    value.replace('\u{0}', "").trim().to_string()
}

fn trim_value(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}
