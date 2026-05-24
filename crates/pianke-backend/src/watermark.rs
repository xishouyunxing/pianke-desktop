use ab_glyph::{FontArc, PxScale};
use exif;
use image::{
    imageops::{self, FilterType},
    DynamicImage, ImageBuffer, Rgba, RgbaImage,
};
use imageproc::{
    drawing::{draw_filled_rect_mut, draw_text_mut},
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
        TemplateFamily::C => render_bottom_bar(
            root_dir,
            img,
            exif,
            0.12,
            false,
            !template.ends_with("_clean"),
            true,
        ),
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
    let side = ((w as f32) * 0.12).max(60.0) as u32;
    let mut canvas = ImageBuffer::from_pixel(w + side, h + 32, Rgba([255, 255, 255, 255]));
    imageops::overlay(
        &mut canvas,
        &DynamicImage::ImageRgb8(img.clone()).to_rgba8(),
        0,
        0,
    );
    draw_filled_rect_mut(
        &mut canvas,
        Rect::at(w as i32, 0).of_size(side, h),
        Rgba([245, 245, 247, 255]),
    );
    let font = load_font();
    let scale_main = PxScale::from(((side as f32) * 0.28).clamp(16.0, 34.0));
    let scale_sub = PxScale::from(((side as f32) * 0.16).clamp(12.0, 22.0));
    let title = trim_value(&format!("{} {}", exif.make, exif.model));
    let body = if show_params {
        trim_value(&format!(
            "{}  {}  {}",
            exif.f_number, exif.exposure, exif.iso
        ))
    } else {
        String::new()
    };
    if let Some(font) = font.as_ref() {
        draw_text_mut(
            &mut canvas,
            Rgba([29, 29, 31, 255]),
            w as i32 + 16,
            24,
            scale_main,
            font,
            &title,
        );
        if !body.is_empty() {
            draw_text_mut(
                &mut canvas,
                Rgba([134, 134, 139, 255]),
                w as i32 + 16,
                72,
                scale_sub,
                font,
                &body,
            );
        }
    }
    if let Some(logo) = load_logo(root_dir, &exif.make, 36) {
        imageops::overlay(
            &mut canvas,
            &logo,
            (w + 14) as i64,
            (h.saturating_sub(50)) as i64,
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
    let band = (h as f32 * 0.32).max(120.0) as u32;
    let mut canvas = ImageBuffer::from_pixel(w, h + band, Rgba([20, 20, 20, 255]));
    imageops::overlay(
        &mut canvas,
        &DynamicImage::ImageRgb8(img.clone()).to_rgba8(),
        0,
        0,
    );
    let font = load_font();
    let scale = PxScale::from((band as f32 * 0.14).clamp(14.0, 28.0));
    let text = trim_value(&format!("{} {}", exif.make, exif.model));
    if !text.is_empty() {
        if let Some(font) = font.as_ref() {
            draw_text_mut(
                &mut canvas,
                Rgba([255, 255, 255, 255]),
                24,
                (h + 18) as i32,
                scale,
                font,
                &text,
            );
        }
    }
    if let Some(logo) = load_logo(root_dir, &exif.make, 44) {
        let x = w.saturating_sub(logo.width() + 24) as i64;
        imageops::overlay(&mut canvas, &logo, x, (h + 18) as i64);
    }
    canvas
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
