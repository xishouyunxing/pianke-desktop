use image::{imageops::FilterType, DynamicImage, GenericImageView, GrayImage};
use ndarray::Array4;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

const DINO_MODEL_PATH: &str = "models/dinov2-small.onnx";
const PREPROCESSOR_PATH: &str = "preprocessor.json";
const FACE_DET_MODEL_PATH: &str = "models/insightface/det_10g.onnx";
const FACE_REC_MODEL_PATH: &str = "models/insightface/w600k_r50.onnx";
const FACE_LANDMARK_MODEL_PATH: &str = "models/insightface/1k3d68.onnx";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dinov2Preprocessor {
    #[serde(default = "default_resize_shorter")]
    pub resize_shorter: u32,
    #[serde(default = "default_input_size")]
    pub input_width: u32,
    #[serde(default = "default_input_size")]
    pub input_height: u32,
    #[serde(default = "default_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_std")]
    pub std: [f32; 3],
    #[serde(default)]
    pub input_name: Option<String>,
    #[serde(default)]
    pub output_name: Option<String>,
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
}

impl Default for Dinov2Preprocessor {
    fn default() -> Self {
        Self {
            resize_shorter: default_resize_shorter(),
            input_width: default_input_size(),
            input_height: default_input_size(),
            mean: default_mean(),
            std: default_std(),
            input_name: None,
            output_name: None,
            embedding_dim: default_embedding_dim(),
        }
    }
}

pub struct Dinov2Model {
    session: Session,
    preprocessor: Dinov2Preprocessor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceInfo {
    pub bbox: [f32; 4],
    pub det_score: f32,
    #[serde(default)]
    pub kps: Option<Vec<[f32; 2]>>,
    #[serde(default)]
    pub landmark_2d_68: Option<Vec<[f32; 2]>>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetail {
    pub bbox: [i32; 4],
    pub sharpness: f64,
    pub eye_score: Option<f64>,
    pub det_score: f64,
    pub area_ratio: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FaceSignals {
    pub face_count: usize,
    pub face_sharpness: Option<f64>,
    pub face_clipped: bool,
    pub eyes_open_score: Option<f64>,
    pub face_area_ratio: Option<f64>,
    pub det_score: Option<f64>,
    pub faces_detail: Vec<FaceDetail>,
}

pub struct InsightFaceModels {
    detector: Session,
    recognizer: Session,
    landmark: Session,
}

impl InsightFaceModels {
    pub fn component_files_ready(component_dir: &Path) -> bool {
        component_dir.join(FACE_DET_MODEL_PATH).exists()
            && component_dir.join(FACE_REC_MODEL_PATH).exists()
            && component_dir.join(FACE_LANDMARK_MODEL_PATH).exists()
    }

    pub fn from_component_dir(component_dir: &Path) -> Result<Self, String> {
        let det_path = component_dir.join(FACE_DET_MODEL_PATH);
        let rec_path = component_dir.join(FACE_REC_MODEL_PATH);
        let landmark_path = component_dir.join(FACE_LANDMARK_MODEL_PATH);
        for path in [&det_path, &rec_path, &landmark_path] {
            if !path.exists() {
                return Err(format!("Expert 组件缺少 InsightFace ONNX 文件: {}", path.display()));
            }
        }
        Ok(Self {
            detector: load_session(&det_path, "InsightFace detector")?,
            recognizer: load_session(&rec_path, "InsightFace recognizer")?,
            landmark: load_session(&landmark_path, "InsightFace landmark")?,
        })
    }

    pub fn extract_faces(&mut self, img: &DynamicImage) -> Result<Vec<FaceInfo>, String> {
        let rgb = resize_max_side(img, 1024).to_rgb8();
        let _detector_preflight = preprocess_bgr_chw(&DynamicImage::ImageRgb8(rgb.clone()), 640, 640)?;
        let faces = run_detector_placeholder(&mut self.detector)?;
        let mut out = Vec::new();
        for face in faces {
            let rec = preprocess_face_crop(img, face.bbox, 112, 112)?;
            let mut embedding = run_embedding(&mut self.recognizer, rec)?;
            normalize_l2(&mut embedding)?;
            let landmark = run_landmark_placeholder(&mut self.landmark)?;
            out.push(FaceInfo {
                bbox: face.bbox,
                det_score: face.det_score,
                kps: face.kps,
                landmark_2d_68: landmark,
                embedding,
            });
        }
        Ok(out)
    }
}

impl Dinov2Model {
    pub fn from_component_dir(component_dir: &Path) -> Result<Self, String> {
        let model_path = component_dir.join(DINO_MODEL_PATH);
        if !model_path.exists() {
            return Err(format!(
                "Expert 组件缺少 DINOv2 ONNX 文件：{}",
                model_path.display()
            ));
        }
        let preprocessor = load_preprocessor(component_dir)?;
        let session = Session::builder()
            .map_err(|e| format!("初始化 ONNX Runtime 失败: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("配置 ONNX Runtime 失败: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("加载 DINOv2 ONNX 失败: {e}"))?;
        Ok(Self {
            session,
            preprocessor,
        })
    }

    pub fn extract_path(&mut self, path: &Path) -> Result<Vec<f32>, String> {
        let img = image::open(path).map_err(|e| format!("加载图片失败: {e}"))?;
        self.extract(&img)
    }

    pub fn extract(&mut self, img: &DynamicImage) -> Result<Vec<f32>, String> {
        let input = preprocess_to_nchw(img, &self.preprocessor)?;
        let input_name = self.preprocessor.input_name.as_deref();
        let output_name = self.preprocessor.output_name.as_deref();
        let input_view =
            TensorRef::from_array_view(&input).map_err(|e| format!("创建 DINOv2 输入失败: {e}"))?;
        let outputs = if let Some(name) = input_name {
            self.session
                .run(ort::inputs![name => input_view])
                .map_err(|e| format!("运行 DINOv2 失败: {e}"))?
        } else {
            self.session
                .run(ort::inputs![input_view])
                .map_err(|e| format!("运行 DINOv2 失败: {e}"))?
        };
        let output = if let Some(name) = output_name {
            outputs
                .get(name)
                .ok_or_else(|| format!("DINOv2 输出缺少字段: {name}"))?
        } else {
            &outputs[0]
        };
        let tensor = output
            .try_extract_array::<f32>()
            .map_err(|e| format!("读取 DINOv2 输出失败: {e}"))?;
        let raw = tensor.iter().copied().collect::<Vec<_>>();
        if raw.len() < self.preprocessor.embedding_dim {
            return Err(format!(
                "DINOv2 输出维度不足：{} < {}",
                raw.len(),
                self.preprocessor.embedding_dim
            ));
        }
        let mut cls = raw
            .into_iter()
            .take(self.preprocessor.embedding_dim)
            .collect::<Vec<_>>();
        normalize_l2(&mut cls)?;
        Ok(cls)
    }
}

fn load_session(path: &Path, label: &str) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("初始化 ONNX Runtime 失败: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("配置 {label} ONNX Runtime 失败: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("加载 {label} ONNX 失败: {e}"))
}

pub fn load_preprocessor(component_dir: &Path) -> Result<Dinov2Preprocessor, String> {
    let path = component_dir.join(PREPROCESSOR_PATH);
    if !path.exists() {
        return Ok(Dinov2Preprocessor::default());
    }
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("读取 DINOv2 preprocessor.json 失败: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("解析 DINOv2 preprocessor.json 失败: {e}"))
}

#[derive(Debug, Clone)]
struct DetectorFace {
    bbox: [f32; 4],
    det_score: f32,
    kps: Option<Vec<[f32; 2]>>,
}

fn run_detector_placeholder(session: &mut Session) -> Result<Vec<DetectorFace>, String> {
    let input = Array4::<f32>::zeros((1, 3, 640, 640));
    let input_view = TensorRef::from_array_view(&input)
        .map_err(|e| format!("创建 InsightFace detector 输入失败: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_view])
        .map_err(|e| format!("运行 InsightFace detector 失败: {e}"))?;
    if outputs.len() == 0 {
        return Err("InsightFace detector 没有输出".to_string());
    }
    Ok(Vec::new())
}

fn run_embedding(session: &mut Session, input: Array4<f32>) -> Result<Vec<f32>, String> {
    let input_view = TensorRef::from_array_view(&input)
        .map_err(|e| format!("创建 InsightFace recognizer 输入失败: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_view])
        .map_err(|e| format!("运行 InsightFace recognizer 失败: {e}"))?;
    if outputs.len() == 0 {
        return Err("InsightFace recognizer 没有输出".to_string());
    }
    let tensor = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|e| format!("读取 InsightFace embedding 失败: {e}"))?;
    let embedding = tensor.iter().copied().collect::<Vec<_>>();
    if embedding.len() < 128 {
        return Err(format!("InsightFace embedding 维度异常: {}", embedding.len()));
    }
    Ok(embedding)
}

fn run_landmark_placeholder(session: &mut Session) -> Result<Option<Vec<[f32; 2]>>, String> {
    let input = Array4::<f32>::zeros((1, 3, 192, 192));
    let input_view = TensorRef::from_array_view(&input)
        .map_err(|e| format!("创建 InsightFace landmark 输入失败: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_view])
        .map_err(|e| format!("运行 InsightFace landmark 失败: {e}"))?;
    if outputs.len() == 0 {
        return Err("InsightFace landmark 没有输出".to_string());
    }
    Ok(None)
}

pub fn preprocess_bgr_chw(
    img: &DynamicImage,
    width: u32,
    height: u32,
) -> Result<Array4<f32>, String> {
    if width == 0 || height == 0 {
        return Err("InsightFace 输入尺寸不能为 0".to_string());
    }
    let resized = image::imageops::resize(&img.to_rgb8(), width, height, FilterType::Triangle);
    let plane = (width * height) as usize;
    let mut data = vec![0.0f32; plane * 3];
    for y in 0..height {
        for x in 0..width {
            let p = resized.get_pixel(x, y).0;
            let idx = (y * width + x) as usize;
            data[idx] = p[2] as f32;
            data[plane + idx] = p[1] as f32;
            data[plane * 2 + idx] = p[0] as f32;
        }
    }
    Array4::from_shape_vec((1, 3, height as usize, width as usize), data)
        .map_err(|e| format!("创建 InsightFace BGR NCHW 输入失败: {e}"))
}

fn preprocess_face_crop(
    img: &DynamicImage,
    bbox: [f32; 4],
    width: u32,
    height: u32,
) -> Result<Array4<f32>, String> {
    let (full_w, full_h) = img.dimensions();
    let x1 = bbox[0].floor().clamp(0.0, full_w.saturating_sub(1) as f32) as u32;
    let y1 = bbox[1].floor().clamp(0.0, full_h.saturating_sub(1) as f32) as u32;
    let x2 = bbox[2].ceil().clamp((x1 + 1) as f32, full_w as f32) as u32;
    let y2 = bbox[3].ceil().clamp((y1 + 1) as f32, full_h as f32) as u32;
    let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
    preprocess_bgr_chw(&crop, width, height)
}

fn resize_max_side(img: &DynamicImage, max_side: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w.max(h) <= max_side {
        return img.clone();
    }
    let scale = max_side as f32 / w.max(h) as f32;
    img.resize(
        (w as f32 * scale).round().max(1.0) as u32,
        (h as f32 * scale).round().max(1.0) as u32,
        FilterType::Lanczos3,
    )
}

pub fn face_signals_from_data(faces: &[FaceInfo], img: &DynamicImage) -> FaceSignals {
    if faces.is_empty() {
        return FaceSignals::default();
    }
    let (full_w, full_h) = img.dimensions();
    let mut details = Vec::new();
    for face in faces {
        let [x1, y1, x2, y2] = face.bbox;
        let cx1 = x1.floor().max(0.0).min(full_w as f32) as u32;
        let cy1 = y1.floor().max(0.0).min(full_h as f32) as u32;
        let cx2 = x2.ceil().max(0.0).min(full_w as f32) as u32;
        let cy2 = y2.ceil().max(0.0).min(full_h as f32) as u32;
        if cx2 <= cx1 || cy2 <= cy1 {
            continue;
        }
        let crop = img.crop_imm(cx1, cy1, cx2 - cx1, cy2 - cy1).to_luma8();
        let sharpness = laplacian_variance(&downscale_gray(crop, 256));
        let eye_score = compute_eye_open_score(face);
        let area_ratio = ((cx2 - cx1) as f64 * (cy2 - cy1) as f64)
            / (full_w.max(1) as f64 * full_h.max(1) as f64);
        details.push(FaceDetail {
            bbox: [cx1 as i32, cy1 as i32, cx2 as i32, cy2 as i32],
            sharpness: round3(sharpness),
            eye_score: eye_score.map(|v| (v * 10000.0).round() / 10000.0),
            det_score: face.det_score as f64,
            area_ratio: (area_ratio * 100000.0).round() / 100000.0,
        });
    }
    if details.is_empty() {
        return FaceSignals {
            face_count: faces.len(),
            ..FaceSignals::default()
        };
    }
    let main_idx = details
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.area_ratio
                .partial_cmp(&b.area_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let main = details[main_idx].clone();
    let margin = 2.0f64.max(full_w.min(full_h) as f64 * 0.008);
    let clipped = main.bbox[0] as f64 <= margin
        || main.bbox[2] as f64 >= full_w as f64 - margin
        || main.bbox[1] as f64 <= margin
        || main.bbox[3] as f64 >= full_h as f64 - margin;
    FaceSignals {
        face_count: details.len(),
        face_sharpness: Some(main.sharpness),
        face_clipped: clipped,
        eyes_open_score: main.eye_score,
        face_area_ratio: Some(main.area_ratio),
        det_score: Some(main.det_score),
        faces_detail: details,
    }
}

pub fn compute_eye_open_score(face: &FaceInfo) -> Option<f64> {
    let lm = face.landmark_2d_68.as_ref()?;
    if lm.len() < 48 {
        return None;
    }
    fn ear(points: &[[f32; 2]]) -> f64 {
        fn dist(a: [f32; 2], b: [f32; 2]) -> f64 {
            let dx = a[0] as f64 - b[0] as f64;
            let dy = a[1] as f64 - b[1] as f64;
            (dx * dx + dy * dy).sqrt()
        }
        let horiz = dist(points[0], points[3]);
        if horiz < 1e-6 {
            return 0.0;
        }
        (dist(points[1], points[5]) + dist(points[2], points[4])) / (2.0 * horiz)
    }
    let left = ear(&lm[36..42]);
    let right = ear(&lm[42..48]);
    let score = (left + right) / 2.0;
    (score <= 0.55).then_some(score)
}

fn downscale_gray(img: GrayImage, max_side: u32) -> GrayImage {
    let (w, h) = img.dimensions();
    if w.max(h) <= max_side {
        img
    } else {
        DynamicImage::ImageLuma8(img)
            .resize(max_side, max_side, FilterType::Lanczos3)
            .to_luma8()
    }
}

fn laplacian_variance(img: &GrayImage) -> f64 {
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return 0.0;
    }
    let mut values = Vec::with_capacity(((w - 2) * (h - 2)) as usize);
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let center = img.get_pixel(x, y).0[0] as f64 * -4.0;
            let v = center
                + img.get_pixel(x - 1, y).0[0] as f64
                + img.get_pixel(x + 1, y).0[0] as f64
                + img.get_pixel(x, y - 1).0[0] as f64
                + img.get_pixel(x, y + 1).0[0] as f64;
            values.push(v);
        }
    }
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len().max(1) as f64
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

pub fn preprocess_to_nchw(
    img: &DynamicImage,
    cfg: &Dinov2Preprocessor,
) -> Result<Array4<f32>, String> {
    if cfg.resize_shorter == 0 || cfg.input_width == 0 || cfg.input_height == 0 {
        return Err("DINOv2 预处理尺寸不能为 0".to_string());
    }
    if cfg.std.iter().any(|v| v.abs() < 1e-8) {
        return Err("DINOv2 预处理 std 不能为 0".to_string());
    }
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let shorter = w.min(h).max(1);
    let scale = cfg.resize_shorter as f32 / shorter as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(cfg.input_width);
    let new_h = ((h as f32 * scale).round() as u32).max(cfg.input_height);
    let resized = image::imageops::resize(&rgb, new_w, new_h, FilterType::CatmullRom);
    let x = new_w.saturating_sub(cfg.input_width) / 2;
    let y = new_h.saturating_sub(cfg.input_height) / 2;
    let cropped =
        image::imageops::crop_imm(&resized, x, y, cfg.input_width, cfg.input_height).to_image();

    let mut data = vec![
        0.0f32;
        (3 * cfg.input_width * cfg.input_height)
            .try_into()
            .unwrap_or(0)
    ];
    let plane = (cfg.input_width * cfg.input_height) as usize;
    for yy in 0..cfg.input_height {
        for xx in 0..cfg.input_width {
            let pixel = cropped.get_pixel(xx, yy).0;
            let idx = (yy * cfg.input_width + xx) as usize;
            for c in 0..3 {
                data[c * plane + idx] = (pixel[c] as f32 / 255.0 - cfg.mean[c]) / cfg.std[c];
            }
        }
    }
    Array4::from_shape_vec(
        (1, 3, cfg.input_height as usize, cfg.input_width as usize),
        data,
    )
    .map_err(|e| format!("创建 DINOv2 NCHW 输入失败: {e}"))
}

pub fn normalize_l2(values: &mut [f32]) -> Result<(), String> {
    let norm = values
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    if norm < 1e-8 {
        return Err("DINOv2 输出零向量".to_string());
    }
    for value in values {
        *value = (*value as f64 / norm) as f32;
    }
    Ok(())
}

fn default_resize_shorter() -> u32 {
    256
}

fn default_input_size() -> u32 {
    224
}

fn default_embedding_dim() -> usize {
    384
}

fn default_mean() -> [f32; 3] {
    [0.485, 0.456, 0.406]
}

fn default_std() -> [f32; 3] {
    [0.229, 0.224, 0.225]
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn preprocess_outputs_nchw_normalized_tensor() {
        let mut src = RgbImage::new(320, 240);
        for y in 0..240 {
            for x in 0..320 {
                src.put_pixel(x, y, Rgb([(x % 255) as u8, (y % 255) as u8, 128]));
            }
        }
        let arr = preprocess_to_nchw(
            &DynamicImage::ImageRgb8(src),
            &Dinov2Preprocessor::default(),
        )
        .expect("preprocess");
        assert_eq!(arr.shape(), &[1, 3, 224, 224]);
        assert!(arr.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn missing_onnx_file_is_clear_error() {
        let temp = tempfile::tempdir().expect("temp dir");
        let err = match Dinov2Model::from_component_dir(temp.path()) {
            Ok(_) => panic!("missing model should fail"),
            Err(err) => err,
        };
        assert!(err.contains("dinov2-small.onnx"));
    }

    #[test]
    fn l2_normalize_outputs_unit_vector() {
        let mut values = vec![3.0, 4.0];
        normalize_l2(&mut values).expect("normalize");
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn face_signals_compute_main_face_sharpness_and_eye_score() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(120, 120, |x, y| {
            let v = if (x + y) % 2 == 0 { 255 } else { 0 };
            Rgb([v, v, v])
        }));
        let mut lm = vec![[0.0f32, 0.0f32]; 68];
        lm[36] = [40.0, 50.0];
        lm[37] = [43.0, 49.0];
        lm[38] = [47.0, 49.0];
        lm[39] = [50.0, 50.0];
        lm[40] = [47.0, 51.0];
        lm[41] = [43.0, 51.0];
        lm[42] = [70.0, 50.0];
        lm[43] = [73.0, 49.0];
        lm[44] = [77.0, 49.0];
        lm[45] = [80.0, 50.0];
        lm[46] = [77.0, 51.0];
        lm[47] = [73.0, 51.0];
        let face = FaceInfo {
            bbox: [20.0, 20.0, 100.0, 100.0],
            det_score: 0.91,
            kps: None,
            landmark_2d_68: Some(lm),
            embedding: vec![1.0, 0.0],
        };
        let signals = face_signals_from_data(&[face], &img);
        assert_eq!(signals.face_count, 1);
        assert!(signals.face_sharpness.expect("sharpness") > 0.0);
        assert!(signals.eyes_open_score.expect("eye score") > 0.05);
        assert!(!signals.face_clipped);
        assert_eq!(signals.faces_detail.len(), 1);
    }

    #[test]
    fn dinov2_golden_fixture_matches_when_configured() {
        let Ok(component_dir) = std::env::var("PIANKE_EXPERT_COMPONENT_DIR") else {
            return;
        };
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root")
            .join("fixtures")
            .join("expert_parity")
            .join("dinov2.json");
        if !fixture.exists() {
            return;
        }
        let text = fs::read_to_string(&fixture).expect("read DINOv2 golden fixture");
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("parse DINOv2 golden fixture");
        let mut model =
            Dinov2Model::from_component_dir(Path::new(&component_dir)).expect("load DINOv2 model");
        for item in value["items"].as_array().expect("items") {
            let path = Path::new(item["path"].as_str().expect("path"));
            let expected = item["embedding"]
                .as_array()
                .expect("embedding")
                .iter()
                .map(|v| v.as_f64().expect("embedding value") as f32)
                .collect::<Vec<_>>();
            let actual = model.extract_path(path).expect("extract DINOv2");
            assert_eq!(actual.len(), expected.len());
            let cosine = actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>();
            assert!(
                cosine >= 0.999,
                "DINOv2 cosine below threshold for {}: {cosine}",
                path.display()
            );
        }
    }
}
