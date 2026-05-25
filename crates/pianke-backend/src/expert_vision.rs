use image::{imageops::FilterType, DynamicImage, GenericImageView, GrayImage, Rgb, RgbImage};
use imageproc::geometric_transformations::Projection;
#[cfg(not(feature = "opencv-orb"))]
use imageproc::geometric_transformations::{warp_into, Interpolation};
use ndarray::Array4;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

const DINO_MODEL_PATH: &str = "models/dinov2-small.onnx";
const PREPROCESSOR_PATH: &str = "preprocessor.json";
const FACE_DET_MODEL_PATH: &str = "models/insightface/det_10g.onnx";
const FACE_REC_MODEL_PATH: &str = "models/insightface/w600k_r50.onnx";
const FACE_LANDMARK_MODEL_PATH: &str = "models/insightface/1k3d68.onnx";
const MUSIQ_MODEL_PATH: &str = "models/quality/musiq.onnx";
const CLIPIQA_MODEL_PATH: &str = "models/quality/clipiqa_plus.onnx";
const QUALITY_PREPROCESSOR_PATH: &str = "quality_preprocessor.json";
const DET_SIZE: u32 = 640;
const FACE_MAX_DIM: u32 = 1024;
const DET_SCORE_THRESHOLD: f32 = 0.5;
const DET_NMS_THRESHOLD: f32 = 0.4;
const PILLOW_PRECISION_BITS: i32 = 32 - 8 - 2;
const ARC_FACE_TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

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
pub struct ExpertQualityPreprocessor {
    #[serde(default = "default_quality_max_side")]
    pub max_side: u32,
    #[serde(default)]
    pub musiq_input_width: Option<u32>,
    #[serde(default)]
    pub musiq_input_height: Option<u32>,
    #[serde(default = "default_quality_clip_input_size")]
    pub clipiqa_input_width: u32,
    #[serde(default = "default_quality_clip_input_size")]
    pub clipiqa_input_height: u32,
    #[serde(default)]
    pub mean: Option<[f32; 3]>,
    #[serde(default)]
    pub std: Option<[f32; 3]>,
    #[serde(default)]
    pub scale_255: bool,
    #[serde(default)]
    pub musiq_input_name: Option<String>,
    #[serde(default)]
    pub musiq_output_name: Option<String>,
    #[serde(default)]
    pub clipiqa_input_name: Option<String>,
    #[serde(default)]
    pub clipiqa_output_name: Option<String>,
}

impl Default for ExpertQualityPreprocessor {
    fn default() -> Self {
        Self {
            max_side: default_quality_max_side(),
            musiq_input_width: None,
            musiq_input_height: None,
            clipiqa_input_width: default_quality_clip_input_size(),
            clipiqa_input_height: default_quality_clip_input_size(),
            mean: None,
            std: None,
            scale_255: false,
            musiq_input_name: None,
            musiq_output_name: None,
            clipiqa_input_name: None,
            clipiqa_output_name: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExpertQualityScores {
    pub musiq_score: Option<f64>,
    pub clipiqa_score: Option<f64>,
}

pub struct ExpertQualityModels {
    musiq: Session,
    clipiqa: Session,
    preprocessor: ExpertQualityPreprocessor,
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

impl ExpertQualityModels {
    pub fn component_files_ready(component_dir: &Path) -> bool {
        component_dir.join(MUSIQ_MODEL_PATH).exists()
            && component_dir.join(CLIPIQA_MODEL_PATH).exists()
    }

    pub fn from_component_dir(component_dir: &Path) -> Result<Self, String> {
        let musiq_path = component_dir.join(MUSIQ_MODEL_PATH);
        let clipiqa_path = component_dir.join(CLIPIQA_MODEL_PATH);
        for path in [&musiq_path, &clipiqa_path] {
            if !path.exists() {
                return Err(format!(
                    "Expert 组件缺少质量模型 ONNX 文件: {}",
                    path.display()
                ));
            }
        }
        Ok(Self {
            musiq: load_session(&musiq_path, "MUSIQ")?,
            clipiqa: load_session(&clipiqa_path, "CLIP-IQA+")?,
            preprocessor: load_quality_preprocessor(component_dir)?,
        })
    }

    pub fn analyze(&mut self, img: &DynamicImage) -> Result<ExpertQualityScores, String> {
        let musiq_input = preprocess_quality_nchw(
            img,
            self.preprocessor.musiq_input_width,
            self.preprocessor.musiq_input_height,
            self.preprocessor.max_side,
            &self.preprocessor,
        )?;
        let clipiqa_input = preprocess_quality_nchw(
            img,
            Some(self.preprocessor.clipiqa_input_width),
            Some(self.preprocessor.clipiqa_input_height),
            self.preprocessor.max_side,
            &self.preprocessor,
        )?;
        let musiq_score = run_scalar_model(
            &mut self.musiq,
            musiq_input,
            self.preprocessor.musiq_input_name.as_deref(),
            self.preprocessor.musiq_output_name.as_deref(),
            "MUSIQ",
        )?;
        let clipiqa_score = run_scalar_model(
            &mut self.clipiqa,
            clipiqa_input,
            self.preprocessor.clipiqa_input_name.as_deref(),
            self.preprocessor.clipiqa_output_name.as_deref(),
            "CLIP-IQA+",
        )?;
        Ok(ExpertQualityScores {
            musiq_score: Some(musiq_score),
            clipiqa_score: Some(clipiqa_score),
        })
    }
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
                return Err(format!(
                    "Expert 组件缺少 InsightFace ONNX 文件: {}",
                    path.display()
                ));
            }
        }
        Ok(Self {
            detector: load_session(&det_path, "InsightFace detector")?,
            recognizer: load_session(&rec_path, "InsightFace recognizer")?,
            landmark: load_session(&landmark_path, "InsightFace landmark")?,
        })
    }

    pub fn extract_faces(&mut self, img: &DynamicImage) -> Result<Vec<FaceInfo>, String> {
        let det_input = DetectorInput::from_image(img, DET_SIZE)?;
        let det_source = det_input.source_rgb.clone();
        let faces = run_detector(&mut self.detector, &det_input)?;
        let mut out = Vec::new();
        for face in faces {
            if face.kps.is_none() {
                return Err("InsightFace detector did not return 5-point landmarks".to_string());
            }
            let pre_kps = face.pre_kps.clone().ok_or_else(|| {
                "InsightFace detector did not return pre-resized 5-point landmarks".to_string()
            })?;
            let (aligned, _inverse) = align_face_rgb(&det_source, &pre_kps, 112)?;
            let rec = rgb_to_chw(&aligned, 127.5, 127.5)?;
            let mut embedding = run_embedding(&mut self.recognizer, rec)?;
            normalize_l2(&mut embedding)?;
            let landmark = run_landmark(
                &mut self.landmark,
                &det_source,
                face.pre_bbox,
                det_input.pre_scale,
            )?;
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

    pub fn extract_path(&mut self, path: &Path) -> Result<(DynamicImage, Vec<FaceInfo>), String> {
        let img = load_insightface_image(path)?;
        let faces = self.extract_faces(&img)?;
        Ok((img, faces))
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

pub fn load_insightface_image(path: &Path) -> Result<DynamicImage, String> {
    #[cfg(feature = "opencv-orb")]
    {
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
            .unwrap_or(false)
        {
            let bytes =
                fs::read(path).map_err(|e| format!("读取 InsightFace JPEG 图片失败: {e}"))?;
            let rgb: RgbImage = turbojpeg::decompress_image(&bytes)
                .map_err(|e| format!("libjpeg-turbo 解码 InsightFace JPEG 失败: {e}"))?;
            return Ok(DynamicImage::ImageRgb8(rgb));
        }
    }
    image::open(path).map_err(|e| format!("加载 InsightFace 图片失败: {e}"))
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

pub fn load_quality_preprocessor(
    component_dir: &Path,
) -> Result<ExpertQualityPreprocessor, String> {
    let path = component_dir.join(QUALITY_PREPROCESSOR_PATH);
    if !path.exists() {
        return Ok(ExpertQualityPreprocessor::default());
    }
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("读取 Expert quality_preprocessor.json 失败: {e}"))?;
    serde_json::from_str(&text)
        .map_err(|e| format!("解析 Expert quality_preprocessor.json 失败: {e}"))
}

#[derive(Debug, Clone)]
struct DetectorFace {
    bbox: [f32; 4],
    pre_bbox: [f32; 4],
    det_score: f32,
    kps: Option<Vec<[f32; 2]>>,
    pre_kps: Option<Vec<[f32; 2]>>,
}

struct DetectorInput {
    tensor: Array4<f32>,
    source_rgb: RgbImage,
    pre_scale: f32,
    det_scale: f32,
    pad_x: f32,
    pad_y: f32,
}

impl DetectorInput {
    fn from_image(img: &DynamicImage, det_size: u32) -> Result<Self, String> {
        #[cfg(feature = "opencv-orb")]
        {
            return Self::from_image_opencv(img, det_size);
        }
        #[cfg(not(feature = "opencv-orb"))]
        {
            Self::from_image_rust(img, det_size)
        }
    }

    #[cfg(not(feature = "opencv-orb"))]
    fn from_image_rust(img: &DynamicImage, det_size: u32) -> Result<Self, String> {
        let (w, h) = img.dimensions();
        if w == 0 || h == 0 || det_size == 0 {
            return Err("InsightFace detector input size is invalid".to_string());
        }
        let original = img.to_rgb8();
        let max_side = w.max(h);
        let (det_source, pre_scale) = if max_side > FACE_MAX_DIM {
            let pre_scale = FACE_MAX_DIM as f32 / max_side as f32;
            let pre_w = ((w as f32 * pre_scale) as u32).max(1);
            let pre_h = ((h as f32 * pre_scale) as u32).max(1);
            (
                pillow_lanczos_resize_rgb(&original, pre_w, pre_h),
                pre_scale,
            )
        } else {
            (original, 1.0)
        };
        let (src_w, src_h) = det_source.dimensions();
        let im_ratio = src_h as f32 / src_w as f32;
        let model_ratio = 1.0f32;
        let (resized_w, resized_h) = if im_ratio > model_ratio {
            let resized_h = det_size;
            let resized_w = (resized_h as f32 / im_ratio).max(1.0) as u32;
            (resized_w, resized_h)
        } else {
            let resized_w = det_size;
            let resized_h = (resized_w as f32 * im_ratio).max(1.0) as u32;
            (resized_w, resized_h)
        };
        let det_scale = resized_h as f32 / src_h as f32;
        let resized = image::imageops::resize(
            &det_source,
            resized_w.max(1),
            resized_h.max(1),
            FilterType::Triangle,
        );
        let mut canvas = RgbImage::from_pixel(det_size, det_size, Rgb([0, 0, 0]));
        image::imageops::replace(&mut canvas, &resized, 0, 0);
        let tensor = rgb_to_chw(&canvas, 127.5, 128.0)?;
        Ok(Self {
            tensor,
            source_rgb: det_source,
            pre_scale,
            det_scale,
            pad_x: 0.0,
            pad_y: 0.0,
        })
    }

    #[cfg(feature = "opencv-orb")]
    fn from_image_opencv(img: &DynamicImage, det_size: u32) -> Result<Self, String> {
        use opencv::{core, imgproc, prelude::*};

        let (w, h) = img.dimensions();
        if w == 0 || h == 0 || det_size == 0 {
            return Err("InsightFace detector input size is invalid".to_string());
        }
        let original_rgb = img.to_rgb8();
        let max_side = w.max(h);
        let (source_rgb, pre_scale) = if max_side > FACE_MAX_DIM {
            let pre_scale = FACE_MAX_DIM as f32 / max_side as f32;
            let pre_w = ((w as f32 * pre_scale) as u32).max(1);
            let pre_h = ((h as f32 * pre_scale) as u32).max(1);
            (
                pillow_lanczos_resize_rgb(&original_rgb, pre_w, pre_h),
                pre_scale,
            )
        } else {
            (original_rgb, 1.0)
        };
        let source_mat = rgb_image_to_mat(&source_rgb)?;
        let src_w = source_rgb.width().max(1);
        let src_h = source_rgb.height().max(1);
        let im_ratio = src_h as f32 / src_w as f32;
        let model_ratio = 1.0f32;
        let (resized_w, resized_h) = if im_ratio > model_ratio {
            let resized_h = det_size;
            let resized_w = (resized_h as f32 / im_ratio).max(1.0) as u32;
            (resized_w, resized_h)
        } else {
            let resized_w = det_size;
            let resized_h = (resized_w as f32 * im_ratio).max(1.0) as u32;
            (resized_w, resized_h)
        };
        let det_scale = resized_h as f32 / src_h as f32;
        let mut resized = core::Mat::default();
        imgproc::resize(
            &source_mat,
            &mut resized,
            core::Size::new(resized_w as i32, resized_h as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )
        .map_err(|e| format!("InsightFace OpenCV detector resize failed: {e}"))?;

        let mut canvas = RgbImage::from_pixel(det_size, det_size, Rgb([0, 0, 0]));
        for y in 0..resized_h {
            for x in 0..resized_w {
                let p = *resized
                    .at_2d::<core::Vec3b>(y as i32, x as i32)
                    .map_err(|e| format!("read InsightFace OpenCV pixel failed: {e}"))?;
                canvas.put_pixel(x, y, Rgb([p[0], p[1], p[2]]));
            }
        }
        let tensor = rgb_to_chw(&canvas, 127.5, 128.0)?;
        Ok(Self {
            tensor,
            source_rgb,
            pre_scale,
            det_scale,
            pad_x: 0.0,
            pad_y: 0.0,
        })
    }
}

#[cfg(feature = "opencv-orb")]
fn rgb_image_to_mat(img: &RgbImage) -> Result<opencv::core::Mat, String> {
    use opencv::prelude::*;

    let (_, h) = img.dimensions();
    opencv::core::Mat::from_slice(img.as_raw())
        .map_err(|e| format!("create OpenCV RGB Mat failed: {e}"))?
        .reshape(3, h as i32)
        .map_err(|e| format!("reshape OpenCV RGB Mat failed: {e}"))?
        .try_clone()
        .map_err(|e| format!("clone OpenCV RGB Mat failed: {e}"))
}

#[derive(Debug, Clone)]
struct PillowCoeffs {
    bounds: Vec<(usize, usize)>,
    coeffs: Vec<i32>,
    ksize: usize,
}

pub(crate) fn pillow_lanczos_resize_rgb(src: &RgbImage, dst_w: u32, dst_h: u32) -> RgbImage {
    let (src_w, src_h) = src.dimensions();
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return RgbImage::new(dst_w, dst_h);
    }
    if src_w == dst_w && src_h == dst_h {
        return src.clone();
    }

    let horiz = pillow_precompute_coeffs(src_w as usize, 0.0, src_w as f32, dst_w as usize);
    let vert = pillow_precompute_coeffs(src_h as usize, 0.0, src_h as f32, dst_h as usize);
    let y_first = vert.bounds.first().map(|(start, _)| *start).unwrap_or(0);
    let y_last = vert
        .bounds
        .last()
        .map(|(start, len)| start + len)
        .unwrap_or(src_h as usize);
    let tmp_h = y_last.saturating_sub(y_first).max(1);
    let mut tmp = vec![[0u8; 3]; dst_w as usize * tmp_h];

    for yy in y_first..y_last {
        let tmp_y = yy - y_first;
        for xx in 0..dst_w as usize {
            let (xmin, xmax) = horiz.bounds[xx];
            let coeff = &horiz.coeffs[xx * horiz.ksize..xx * horiz.ksize + horiz.ksize];
            let mut ss = [
                1i64 << (PILLOW_PRECISION_BITS - 1),
                1i64 << (PILLOW_PRECISION_BITS - 1),
                1i64 << (PILLOW_PRECISION_BITS - 1),
            ];
            for x in 0..xmax {
                let p = src.get_pixel((xmin + x) as u32, yy as u32).0;
                let k = coeff[x] as i64;
                ss[0] += p[0] as i64 * k;
                ss[1] += p[1] as i64 * k;
                ss[2] += p[2] as i64 * k;
            }
            tmp[tmp_y * dst_w as usize + xx] = [
                pillow_clip8(ss[0]),
                pillow_clip8(ss[1]),
                pillow_clip8(ss[2]),
            ];
        }
    }

    let mut out = RgbImage::from_pixel(dst_w, dst_h, Rgb([0, 0, 0]));
    for yy in 0..dst_h as usize {
        let (ymin, ymax) = vert.bounds[yy];
        let coeff = &vert.coeffs[yy * vert.ksize..yy * vert.ksize + vert.ksize];
        for xx in 0..dst_w as usize {
            let mut ss = [
                1i64 << (PILLOW_PRECISION_BITS - 1),
                1i64 << (PILLOW_PRECISION_BITS - 1),
                1i64 << (PILLOW_PRECISION_BITS - 1),
            ];
            for y in 0..ymax {
                let p = tmp[(ymin - y_first + y) * dst_w as usize + xx];
                let k = coeff[y] as i64;
                ss[0] += p[0] as i64 * k;
                ss[1] += p[1] as i64 * k;
                ss[2] += p[2] as i64 * k;
            }
            out.put_pixel(
                xx as u32,
                yy as u32,
                Rgb([
                    pillow_clip8(ss[0]),
                    pillow_clip8(ss[1]),
                    pillow_clip8(ss[2]),
                ]),
            );
        }
    }
    out
}

pub(crate) fn pillow_lanczos_resize_luma(src: &GrayImage, dst_w: u32, dst_h: u32) -> GrayImage {
    let (src_w, src_h) = src.dimensions();
    pillow_lanczos_resize_luma_box(src, dst_w, dst_h, 0.0, 0.0, src_w as f32, src_h as f32)
}

fn pillow_lanczos_resize_luma_box(
    src: &GrayImage,
    dst_w: u32,
    dst_h: u32,
    in0_x: f32,
    in0_y: f32,
    in1_x: f32,
    in1_y: f32,
) -> GrayImage {
    let (src_w, src_h) = src.dimensions();
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return GrayImage::new(dst_w, dst_h);
    }
    if src_w == dst_w
        && src_h == dst_h
        && in0_x == 0.0
        && in0_y == 0.0
        && in1_x == src_w as f32
        && in1_y == src_h as f32
    {
        return src.clone();
    }

    let horiz = pillow_precompute_coeffs(src_w as usize, in0_x, in1_x, dst_w as usize);
    let vert = pillow_precompute_coeffs(src_h as usize, in0_y, in1_y, dst_h as usize);
    let y_first = vert.bounds.first().map(|(start, _)| *start).unwrap_or(0);
    let y_last = vert
        .bounds
        .last()
        .map(|(start, len)| start + len)
        .unwrap_or(src_h as usize);
    let tmp_h = y_last.saturating_sub(y_first).max(1);
    let mut tmp = vec![0u8; dst_w as usize * tmp_h];

    for yy in y_first..y_last {
        let tmp_y = yy - y_first;
        for xx in 0..dst_w as usize {
            let (xmin, xmax) = horiz.bounds[xx];
            let coeff = &horiz.coeffs[xx * horiz.ksize..xx * horiz.ksize + horiz.ksize];
            let mut ss = 1i64 << (PILLOW_PRECISION_BITS - 1);
            for x in 0..xmax {
                let p = src.get_pixel((xmin + x) as u32, yy as u32).0[0];
                ss += p as i64 * coeff[x] as i64;
            }
            tmp[tmp_y * dst_w as usize + xx] = pillow_clip8(ss);
        }
    }

    let mut out = GrayImage::new(dst_w, dst_h);
    for yy in 0..dst_h as usize {
        let (ymin, ymax) = vert.bounds[yy];
        let coeff = &vert.coeffs[yy * vert.ksize..yy * vert.ksize + vert.ksize];
        for xx in 0..dst_w as usize {
            let mut ss = 1i64 << (PILLOW_PRECISION_BITS - 1);
            for y in 0..ymax {
                let p = tmp[(ymin - y_first + y) * dst_w as usize + xx];
                ss += p as i64 * coeff[y] as i64;
            }
            out.put_pixel(xx as u32, yy as u32, image::Luma([pillow_clip8(ss)]));
        }
    }
    out
}

fn pillow_precompute_coeffs(in_size: usize, in0: f32, in1: f32, out_size: usize) -> PillowCoeffs {
    let scale = (in1 as f64 - in0 as f64) / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 3.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut coeffs = vec![0i32; out_size * ksize];
    let ss = 1.0 / filterscale;
    let precision = (1i64 << PILLOW_PRECISION_BITS) as f64;

    for xx in 0..out_size {
        let center = in0 as f64 + (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size as i32 {
            xmax = in_size as i32;
        }
        let count = (xmax - xmin).max(0) as usize;
        let mut weights = vec![0.0f64; count];
        let mut sum = 0.0f64;
        for (x, weight) in weights.iter_mut().enumerate() {
            let w = pillow_lanczos_filter((x as f64 + xmin as f64 - center + 0.5) * ss);
            *weight = w;
            sum += w;
        }
        if sum != 0.0 {
            for weight in &mut weights {
                *weight /= sum;
            }
        }
        let offset = xx * ksize;
        for (x, weight) in weights.iter().enumerate() {
            coeffs[offset + x] = if *weight < 0.0 {
                (-0.5 + weight * precision) as i32
            } else {
                (0.5 + weight * precision) as i32
            };
        }
        bounds.push((xmin as usize, count));
    }

    PillowCoeffs {
        bounds,
        coeffs,
        ksize,
    }
}

fn pillow_lanczos_filter(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        pillow_sinc_filter(x) * pillow_sinc_filter(x / 3.0)
    } else {
        0.0
    }
}

fn pillow_sinc_filter(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let pix = x * std::f64::consts::PI;
        pix.sin() / pix
    }
}

fn pillow_clip8(value: i64) -> u8 {
    (value >> PILLOW_PRECISION_BITS).clamp(0, 255) as u8
}

fn run_detector(session: &mut Session, input: &DetectorInput) -> Result<Vec<DetectorFace>, String> {
    let input_view = TensorRef::from_array_view(&input.tensor)
        .map_err(|e| format!("create InsightFace detector input failed: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_view])
        .map_err(|e| format!("run InsightFace detector failed: {e}"))?;
    if outputs.len() < 6 {
        return Err(format!(
            "InsightFace detector output count is invalid: {}",
            outputs.len()
        ));
    }
    let strides = [8usize, 16, 32];
    let fmc = 3usize;
    let mut candidates = Vec::new();
    for (level, stride) in strides.iter().enumerate() {
        let score = output_vec(&outputs, level)?;
        let bbox = output_vec(&outputs, level + fmc)?;
        let kps = if outputs.len() >= 9 {
            Some(output_vec(&outputs, level + fmc * 2)?)
        } else {
            None
        };
        decode_detector_level(
            score,
            bbox,
            kps,
            *stride,
            DET_SIZE as usize,
            input,
            &mut candidates,
        )?;
    }
    candidates.sort_by(|a, b| {
        b.det_score
            .partial_cmp(&a.det_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep = nms(&candidates, DET_NMS_THRESHOLD);
    Ok(keep
        .into_iter()
        .map(|idx| candidates[idx].clone())
        .collect())
}

fn output_vec(
    outputs: &ort::session::SessionOutputs<'_>,
    index: usize,
) -> Result<Vec<f32>, String> {
    outputs[index]
        .try_extract_array::<f32>()
        .map(|arr| arr.iter().copied().collect())
        .map_err(|e| format!("read InsightFace detector output {index} failed: {e}"))
}

fn decode_detector_level(
    scores: Vec<f32>,
    boxes: Vec<f32>,
    landmarks: Option<Vec<f32>>,
    stride: usize,
    det_size: usize,
    input: &DetectorInput,
    out: &mut Vec<DetectorFace>,
) -> Result<(), String> {
    let height = det_size / stride;
    let width = det_size / stride;
    let anchor_count = height * width * 2;
    if scores.len() < anchor_count || boxes.len() < anchor_count * 4 {
        return Err(format!(
            "InsightFace detector output shape mismatch at stride {stride}: scores={}, boxes={}",
            scores.len(),
            boxes.len()
        ));
    }
    if let Some(values) = &landmarks {
        if values.len() < anchor_count * 10 {
            return Err(format!(
                "InsightFace detector landmark output shape mismatch at stride {stride}: {}",
                values.len()
            ));
        }
    }
    for idx in 0..anchor_count {
        let det_score = scores[idx];
        if det_score < DET_SCORE_THRESHOLD {
            continue;
        }
        let cell = idx / 2;
        let y = cell / width;
        let x = cell % width;
        let anchor = [x as f32 * stride as f32, y as f32 * stride as f32];
        let b = &boxes[idx * 4..idx * 4 + 4];
        let decoded = distance2bbox(
            anchor,
            [
                b[0] * stride as f32,
                b[1] * stride as f32,
                b[2] * stride as f32,
                b[3] * stride as f32,
            ],
            det_size as f32,
        );
        let pre_bbox = [
            (decoded[0] - input.pad_x) / input.det_scale,
            (decoded[1] - input.pad_y) / input.det_scale,
            (decoded[2] - input.pad_x) / input.det_scale,
            (decoded[3] - input.pad_y) / input.det_scale,
        ];
        let export_bbox = [
            python_export_bbox_coord(pre_bbox[0], input.pre_scale),
            python_export_bbox_coord(pre_bbox[1], input.pre_scale),
            python_export_bbox_coord(pre_bbox[2], input.pre_scale),
            python_export_bbox_coord(pre_bbox[3], input.pre_scale),
        ];
        let kps = landmarks.as_ref().map(|values| {
            let mut points = Vec::with_capacity(5);
            for p in 0..5 {
                let dx = values[idx * 10 + p * 2];
                let dy = values[idx * 10 + p * 2 + 1];
                points.push([
                    (anchor[0] + dx * stride as f32 - input.pad_x) / input.det_scale,
                    (anchor[1] + dy * stride as f32 - input.pad_y) / input.det_scale,
                ]);
            }
            points
        });
        out.push(DetectorFace {
            bbox: export_bbox,
            pre_bbox,
            det_score,
            kps: kps.as_ref().map(|points| {
                points
                    .iter()
                    .map(|[x, y]| [x / input.pre_scale, y / input.pre_scale])
                    .collect()
            }),
            pre_kps: kps,
        });
    }
    Ok(())
}

fn python_export_bbox_coord(pre_resized_coord: f32, pre_scale: f32) -> f32 {
    if pre_scale <= 0.0 {
        return pre_resized_coord;
    }
    (pre_resized_coord as i32) as f32 / pre_scale
}

fn distance2bbox(anchor: [f32; 2], distance: [f32; 4], max_shape: f32) -> [f32; 4] {
    [
        (anchor[0] - distance[0]).clamp(0.0, max_shape),
        (anchor[1] - distance[1]).clamp(0.0, max_shape),
        (anchor[0] + distance[2]).clamp(0.0, max_shape),
        (anchor[1] + distance[3]).clamp(0.0, max_shape),
    ]
}

fn nms(faces: &[DetectorFace], threshold: f32) -> Vec<usize> {
    let mut order = (0..faces.len()).collect::<Vec<_>>();
    order.sort_by(|a, b| {
        faces[*b]
            .det_score
            .partial_cmp(&faces[*a].det_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = Vec::new();
    while let Some(current) = order.first().copied() {
        keep.push(current);
        order = order
            .into_iter()
            .skip(1)
            .filter(|idx| bbox_iou(faces[current].bbox, faces[*idx].bbox) <= threshold)
            .collect();
    }
    keep
}

fn bbox_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let xx1 = a[0].max(b[0]);
    let yy1 = a[1].max(b[1]);
    let xx2 = a[2].min(b[2]);
    let yy2 = a[3].min(b[3]);
    let w = (xx2 - xx1 + 1.0).max(0.0);
    let h = (yy2 - yy1 + 1.0).max(0.0);
    let inter = w * h;
    let area_a = ((a[2] - a[0] + 1.0).max(0.0)) * ((a[3] - a[1] + 1.0).max(0.0));
    let area_b = ((b[2] - b[0] + 1.0).max(0.0)) * ((b[3] - b[1] + 1.0).max(0.0));
    inter / (area_a + area_b - inter).max(1e-6)
}

fn run_embedding(session: &mut Session, input: Array4<f32>) -> Result<Vec<f32>, String> {
    let input_view = TensorRef::from_array_view(&input)
        .map_err(|e| format!("create InsightFace recognizer input failed: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_view])
        .map_err(|e| format!("run InsightFace recognizer failed: {e}"))?;
    if outputs.len() == 0 {
        return Err("InsightFace recognizer produced no outputs".to_string());
    }
    let tensor = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|e| format!("read InsightFace embedding failed: {e}"))?;
    let embedding = tensor.iter().copied().collect::<Vec<_>>();
    if embedding.len() < 128 {
        return Err(format!(
            "InsightFace embedding dimension is invalid: {}",
            embedding.len()
        ));
    }
    Ok(embedding)
}

fn run_scalar_model(
    session: &mut Session,
    input: Array4<f32>,
    input_name: Option<&str>,
    output_name: Option<&str>,
    label: &str,
) -> Result<f64, String> {
    let input_view = TensorRef::from_array_view(&input)
        .map_err(|e| format!("create {label} input failed: {e}"))?;
    let outputs = if let Some(name) = input_name {
        session
            .run(ort::inputs![name => input_view])
            .map_err(|e| format!("run {label} failed: {e}"))?
    } else {
        session
            .run(ort::inputs![input_view])
            .map_err(|e| format!("run {label} failed: {e}"))?
    };
    if outputs.len() == 0 {
        return Err(format!("{label} produced no outputs"));
    }
    let output = if let Some(name) = output_name {
        outputs
            .get(name)
            .ok_or_else(|| format!("{label} output missing field: {name}"))?
    } else {
        &outputs[0]
    };
    let tensor = output
        .try_extract_array::<f32>()
        .map_err(|e| format!("read {label} output failed: {e}"))?;
    let Some(score) = tensor.iter().next() else {
        return Err(format!("{label} output is empty"));
    };
    Ok((*score as f64 * 10000.0).round() / 10000.0)
}

fn run_landmark(
    session: &mut Session,
    img: &RgbImage,
    bbox: [f32; 4],
    pre_scale: f32,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    let (crop, inverse) = align_landmark_crop_rgb(img, bbox, 192)?;
    let input = rgb_to_chw(&crop, 0.0, 1.0)?;
    let input_view = TensorRef::from_array_view(&input)
        .map_err(|e| format!("create InsightFace landmark input failed: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_view])
        .map_err(|e| format!("run InsightFace landmark failed: {e}"))?;
    if outputs.len() == 0 {
        return Err("InsightFace landmark produced no outputs".to_string());
    }
    let tensor = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|e| format!("read InsightFace landmark output failed: {e}"))?;
    let values = tensor.iter().copied().collect::<Vec<_>>();
    let mut points = Vec::with_capacity(68);
    if values.len() >= 3000 {
        let rows = values.len() / 3;
        if rows < 68 {
            return Ok(None);
        }
        let start = rows - 68;
        for i in 0..68 {
            let row = start + i;
            let x = (values[row * 3] + 1.0) * 96.0;
            let y = (values[row * 3 + 1] + 1.0) * 96.0;
            let mapped = apply_affine(inverse, [x, y]);
            points.push([mapped[0] / pre_scale, mapped[1] / pre_scale]);
        }
    } else if values.len() >= 136 {
        for i in 0..68 {
            let x = (values[i * 2] + 1.0) * 96.0;
            let y = (values[i * 2 + 1] + 1.0) * 96.0;
            let mapped = apply_affine(inverse, [x, y]);
            points.push([mapped[0] / pre_scale, mapped[1] / pre_scale]);
        }
    } else {
        return Ok(None);
    }
    Ok(Some(points))
}

#[allow(dead_code)]
pub fn preprocess_rgb_chw(
    img: &DynamicImage,
    width: u32,
    height: u32,
    mean: f32,
    std: f32,
) -> Result<Array4<f32>, String> {
    if width == 0 || height == 0 {
        return Err("InsightFace input size must not be zero".to_string());
    }
    let rgb = img.to_rgb8();
    let resized = if rgb.width() == width && rgb.height() == height {
        rgb
    } else {
        image::imageops::resize(&rgb, width, height, FilterType::Triangle)
    };
    rgb_to_chw(&resized, mean, std)
}

fn preprocess_quality_nchw(
    img: &DynamicImage,
    target_width: Option<u32>,
    target_height: Option<u32>,
    max_side: u32,
    cfg: &ExpertQualityPreprocessor,
) -> Result<Array4<f32>, String> {
    let mut rgb = img.to_rgb8();
    let (mut width, mut height) = rgb.dimensions();
    if width == 0 || height == 0 {
        return Err("Expert quality input image is empty".to_string());
    }
    if let (Some(w), Some(h)) = (target_width, target_height) {
        if w == 0 || h == 0 {
            return Err("Expert quality fixed input size must not be zero".to_string());
        }
        if width != w || height != h {
            rgb = image::imageops::resize(&rgb, w, h, FilterType::Triangle);
            width = w;
            height = h;
        }
    } else if max_side > 0 && width.max(height) > max_side {
        let scale = max_side as f32 / width.max(height) as f32;
        width = ((width as f32 * scale).round() as u32).max(1);
        height = ((height as f32 * scale).round() as u32).max(1);
        rgb = image::imageops::resize(&rgb, width, height, FilterType::Triangle);
    }

    let mean = cfg.mean.unwrap_or([0.0, 0.0, 0.0]);
    let std = cfg.std.unwrap_or([1.0, 1.0, 1.0]);
    if std.iter().any(|v| v.abs() < 1e-8) {
        return Err("Expert quality preprocessor std must not contain zero".to_string());
    }
    let plane = (width * height) as usize;
    let mut data = vec![0.0f32; plane * 3];
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y).0;
            let idx = (y * width + x) as usize;
            for c in 0..3 {
                let raw = if cfg.scale_255 {
                    pixel[c] as f32
                } else {
                    pixel[c] as f32 / 255.0
                };
                data[c * plane + idx] = (raw - mean[c]) / std[c];
            }
        }
    }
    Array4::from_shape_vec((1, 3, height as usize, width as usize), data)
        .map_err(|e| format!("create Expert quality NCHW input failed: {e}"))
}

fn rgb_to_chw(img: &RgbImage, mean: f32, std: f32) -> Result<Array4<f32>, String> {
    if std.abs() < 1e-8 {
        return Err("InsightFace std must not be zero".to_string());
    }
    let (width, height) = img.dimensions();
    let plane = (width * height) as usize;
    let mut data = vec![0.0f32; plane * 3];
    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x, y).0;
            let idx = (y * width + x) as usize;
            data[idx] = (p[0] as f32 - mean) / std;
            data[plane + idx] = (p[1] as f32 - mean) / std;
            data[plane * 2 + idx] = (p[2] as f32 - mean) / std;
        }
    }
    Array4::from_shape_vec((1, 3, height as usize, width as usize), data)
        .map_err(|e| format!("create InsightFace RGB NCHW input failed: {e}"))
}

#[cfg(test)]
fn align_face(
    img: &DynamicImage,
    kps: &[[f32; 2]],
    image_size: u32,
) -> Result<(RgbImage, [f64; 6]), String> {
    align_face_rgb(&img.to_rgb8(), kps, image_size)
}

fn align_face_rgb(
    rgb: &RgbImage,
    kps: &[[f32; 2]],
    image_size: u32,
) -> Result<(RgbImage, [f64; 6]), String> {
    if kps.len() < 5 {
        return Err("InsightFace detector returned fewer than 5 landmarks".to_string());
    }
    let dst = ARC_FACE_TEMPLATE
        .map(|[x, y]| [x * image_size as f32 / 112.0, y * image_size as f32 / 112.0]);
    let src = [kps[0], kps[1], kps[2], kps[3], kps[4]];
    let forward = estimate_similarity(&src, &dst)?;
    let inverse = invert_affine(forward)?;
    let projection = Projection::from_matrix([
        forward[0] as f32,
        forward[1] as f32,
        forward[2] as f32,
        forward[3] as f32,
        forward[4] as f32,
        forward[5] as f32,
        0.0,
        0.0,
        1.0,
    ])
    .ok_or_else(|| "create face alignment projection failed".to_string())?;
    let aligned = warp_rgb_affine(rgb, image_size, forward, &projection)?;
    Ok((aligned, inverse))
}

#[cfg(feature = "opencv-orb")]
fn warp_rgb_affine(
    rgb: &RgbImage,
    image_size: u32,
    forward: [f64; 6],
    _projection: &Projection,
) -> Result<RgbImage, String> {
    warp_rgb_affine_with_matrix(rgb, image_size, forward)
}

#[cfg(feature = "opencv-orb")]
fn warp_rgb_affine_with_matrix(
    rgb: &RgbImage,
    image_size: u32,
    forward: [f64; 6],
) -> Result<RgbImage, String> {
    use opencv::{core, imgproc, prelude::*};

    let src = rgb_image_to_mat(rgb)?;
    let matrix = core::Mat::from_slice_2d(&[&forward[0..3], &forward[3..6]])
        .map_err(|e| format!("create OpenCV affine matrix failed: {e}"))?;
    let mut dst = core::Mat::default();
    imgproc::warp_affine(
        &src,
        &mut dst,
        &matrix,
        core::Size::new(image_size as i32, image_size as i32),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        core::Scalar::all(0.0),
    )
    .map_err(|e| format!("InsightFace OpenCV warpAffine failed: {e}"))?;
    let mut aligned = RgbImage::from_pixel(image_size, image_size, Rgb([0, 0, 0]));
    for y in 0..image_size {
        for x in 0..image_size {
            let p = *dst
                .at_2d::<core::Vec3b>(y as i32, x as i32)
                .map_err(|e| format!("read InsightFace OpenCV aligned pixel failed: {e}"))?;
            aligned.put_pixel(x, y, Rgb([p[0], p[1], p[2]]));
        }
    }
    Ok(aligned)
}

#[cfg(not(feature = "opencv-orb"))]
#[allow(dead_code)]
fn warp_rgb_affine_with_matrix(
    _rgb: &RgbImage,
    _image_size: u32,
    _forward: [f64; 6],
) -> Result<RgbImage, String> {
    Err("OpenCV warpAffine is unavailable".to_string())
}

#[cfg(not(feature = "opencv-orb"))]
fn warp_rgb_affine(
    rgb: &RgbImage,
    image_size: u32,
    _forward: [f64; 6],
    projection: &Projection,
) -> Result<RgbImage, String> {
    let mut aligned = RgbImage::from_pixel(image_size, image_size, Rgb([0, 0, 0]));
    warp_into(
        rgb,
        projection,
        Interpolation::Bilinear,
        Rgb([0, 0, 0]),
        &mut aligned,
    );
    Ok(aligned)
}

#[cfg(test)]
#[allow(dead_code)]
fn align_landmark_crop(
    img: &DynamicImage,
    bbox: [f32; 4],
    image_size: u32,
) -> Result<(RgbImage, [f64; 6]), String> {
    align_landmark_crop_rgb(&img.to_rgb8(), bbox, image_size)
}

fn align_landmark_crop_rgb(
    rgb: &RgbImage,
    bbox: [f32; 4],
    image_size: u32,
) -> Result<(RgbImage, [f64; 6]), String> {
    let w = (bbox[2] - bbox[0]).max(1.0);
    let h = (bbox[3] - bbox[1]).max(1.0);
    let center = [(bbox[2] + bbox[0]) * 0.5, (bbox[3] + bbox[1]) * 0.5];
    let scale = image_size as f32 / (w.max(h) * 1.5);
    let forward = [
        scale as f64,
        0.0,
        (image_size as f32 * 0.5 - center[0] * scale) as f64,
        0.0,
        scale as f64,
        (image_size as f32 * 0.5 - center[1] * scale) as f64,
    ];
    let inverse = invert_affine(forward)?;
    #[cfg(feature = "opencv-orb")]
    let crop = warp_rgb_affine_with_matrix(rgb, image_size, forward)?;
    #[cfg(not(feature = "opencv-orb"))]
    let crop = {
        let projection = Projection::from_matrix([
            forward[0] as f32,
            forward[1] as f32,
            forward[2] as f32,
            forward[3] as f32,
            forward[4] as f32,
            forward[5] as f32,
            0.0,
            0.0,
            1.0,
        ])
        .ok_or_else(|| "create landmark crop projection failed".to_string())?;
        let mut crop = RgbImage::from_pixel(image_size, image_size, Rgb([0, 0, 0]));
        warp_into(
            rgb,
            &projection,
            Interpolation::Bilinear,
            Rgb([0, 0, 0]),
            &mut crop,
        );
        crop
    };
    Ok((crop, inverse))
}

fn estimate_similarity(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> Result<[f64; 6], String> {
    let num = src.len() as f32;
    let src_mean = [
        src.iter().map(|p| p[0]).sum::<f32>() / num,
        src.iter().map(|p| p[1]).sum::<f32>() / num,
    ];
    let dst_mean = [
        dst.iter().map(|p| p[0]).sum::<f32>() / num,
        dst.iter().map(|p| p[1]).sum::<f32>() / num,
    ];
    let mut src_demean = [[0.0f32; 2]; 5];
    let mut dst_demean = [[0.0f32; 2]; 5];
    for i in 0..src.len() {
        src_demean[i] = [src[i][0] - src_mean[0], src[i][1] - src_mean[1]];
        dst_demean[i] = [dst[i][0] - dst_mean[0], dst[i][1] - dst_mean[1]];
    }
    let mut a = nalgebra::Matrix2::<f32>::zeros();
    for row in 0..2 {
        for col in 0..2 {
            let mut sum = 0.0f32;
            for i in 0..src.len() {
                sum += dst_demean[i][row] * src_demean[i][col];
            }
            a[(row, col)] = sum / num;
        }
    }
    let src_demean_mean = [
        src_demean.iter().map(|p| p[0]).sum::<f32>() / num,
        src_demean.iter().map(|p| p[1]).sum::<f32>() / num,
    ];
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;
    for i in 0..src.len() {
        let x = src_demean[i][0] - src_demean_mean[0];
        let y = src_demean[i][1] - src_demean_mean[1];
        var_x += x * x;
        var_y += y * y;
    }
    let src_var = (var_x / num) + (var_y / num);
    if src_var.abs() < 1e-12 {
        return Err("face similarity source landmarks are degenerate".to_string());
    }
    // Mirror skimage.transform._umeyama, which InsightFace uses through
    // SimilarityTransform. The SVD path matters: the closed-form 2D rotation can
    // differ by tiny ULPs, enough to flip pixels in OpenCV's warpAffine output.
    let svd = a.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| "face similarity SVD missing U".to_string())?;
    let v_t = svd
        .v_t
        .ok_or_else(|| "face similarity SVD missing Vt".to_string())?;
    let mut d = nalgebra::Vector2::<f32>::new(1.0, 1.0);
    if a.determinant() < 0.0 {
        d[1] = -1.0;
    }
    let rank = svd.singular_values.iter().filter(|v| **v > 1e-6).count();
    if rank == 0 {
        return Err("face similarity covariance is degenerate".to_string());
    }
    let rotation = if rank == 1 {
        if u.determinant() * v_t.determinant() > 0.0 {
            u * v_t
        } else {
            let saved = d[1];
            d[1] = -1.0;
            let r = u * nalgebra::Matrix2::<f32>::from_diagonal(&d) * v_t;
            d[1] = saved;
            r
        }
    } else {
        u * nalgebra::Matrix2::<f32>::from_diagonal(&d) * v_t
    };
    let scale = (svd.singular_values.dot(&d) / src_var) as f64;
    let m00 = scale * rotation[(0, 0)] as f64;
    let m01 = scale * rotation[(0, 1)] as f64;
    let m10 = scale * rotation[(1, 0)] as f64;
    let m11 = scale * rotation[(1, 1)] as f64;
    let tx = dst_mean[0] as f64 - (m00 * src_mean[0] as f64 + m01 * src_mean[1] as f64);
    let ty = dst_mean[1] as f64 - (m10 * src_mean[0] as f64 + m11 * src_mean[1] as f64);
    Ok([m00, m01, tx, m10, m11, ty])
}

fn invert_affine(m: [f64; 6]) -> Result<[f64; 6], String> {
    let det = m[0] * m[4] - m[1] * m[3];
    if det.abs() < 1e-8 {
        return Err("face alignment transform is singular".to_string());
    }
    let inv00 = m[4] / det;
    let inv01 = -m[1] / det;
    let inv10 = -m[3] / det;
    let inv11 = m[0] / det;
    let inv02 = -(inv00 * m[2] + inv01 * m[5]);
    let inv12 = -(inv10 * m[2] + inv11 * m[5]);
    Ok([inv00, inv01, inv02, inv10, inv11, inv12])
}

fn apply_affine(m: [f64; 6], p: [f32; 2]) -> [f32; 2] {
    [
        (m[0] * p[0] as f64 + m[1] * p[1] as f64 + m[2]) as f32,
        (m[3] * p[0] as f64 + m[4] * p[1] as f64 + m[5]) as f32,
    ]
}

pub fn face_signals_from_data(faces: &[FaceInfo], img: &DynamicImage) -> FaceSignals {
    if faces.is_empty() {
        return FaceSignals::default();
    }
    let (full_w, full_h) = img.dimensions();
    let mut details = Vec::new();
    for face in faces {
        let [x1, y1, x2, y2] = face.bbox;
        let cx1 = (x1 as i32).clamp(0, full_w as i32) as u32;
        let cy1 = (y1 as i32).clamp(0, full_h as i32) as u32;
        let cx2 = (x2 as i32).clamp(0, full_w as i32) as u32;
        let cy2 = (y2 as i32).clamp(0, full_h as i32) as u32;
        if cx2 <= cx1 || cy2 <= cy1 {
            continue;
        }
        let crop_rgb = img.crop_imm(cx1, cy1, cx2 - cx1, cy2 - cy1).to_rgb8();
        let crop = pillow_rgb_to_luma(&crop_rgb);
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
        pillow_thumbnail_luma(&img, max_side, max_side)
    }
}

fn pillow_thumbnail_luma(src: &GrayImage, max_w: u32, max_h: u32) -> GrayImage {
    let (w, h) = src.dimensions();
    if w == 0 || h == 0 || max_w == 0 || max_h == 0 {
        return GrayImage::new(0, 0);
    }
    if max_w >= w && max_h >= h {
        return src.clone();
    }

    let aspect = w as f64 / h as f64;
    let mut dst_w = max_w as i32;
    let mut dst_h = max_h as i32;
    if max_w as f64 / max_h as f64 >= aspect {
        dst_w = pillow_round_aspect(max_h as f64 * aspect, |n| {
            (aspect - n as f64 / max_h as f64).abs()
        });
    } else {
        dst_h = pillow_round_aspect(max_w as f64 / aspect, |n| {
            if n == 0 {
                0.0
            } else {
                (aspect - max_w as f64 / n as f64).abs()
            }
        });
    }
    let dst_w = dst_w.max(1) as u32;
    let dst_h = dst_h.max(1) as u32;

    let factor_x = ((w as f64 / dst_w as f64 / 2.0) as u32).max(1);
    let factor_y = ((h as f64 / dst_h as f64 / 2.0) as u32).max(1);
    if factor_x > 1 || factor_y > 1 {
        let reduced = pillow_reduce_luma(src, factor_x, factor_y);
        pillow_lanczos_resize_luma_box(
            &reduced,
            dst_w,
            dst_h,
            0.0,
            0.0,
            w as f32 / factor_x as f32,
            h as f32 / factor_y as f32,
        )
    } else {
        pillow_lanczos_resize_luma(src, dst_w, dst_h)
    }
}

fn pillow_round_aspect<F>(number: f64, key: F) -> i32
where
    F: Fn(i32) -> f64,
{
    let floor = number.floor() as i32;
    let ceil = number.ceil() as i32;
    if key(floor) <= key(ceil) {
        floor.max(1)
    } else {
        ceil.max(1)
    }
}

fn pillow_reduce_luma(src: &GrayImage, factor_x: u32, factor_y: u32) -> GrayImage {
    let (w, h) = src.dimensions();
    let out_w = w.div_ceil(factor_x);
    let out_h = h.div_ceil(factor_y);
    let mut out = GrayImage::new(out_w, out_h);
    for oy in 0..out_h {
        let y0 = oy * factor_y;
        let y1 = ((oy + 1) * factor_y).min(h);
        for ox in 0..out_w {
            let x0 = ox * factor_x;
            let x1 = ((ox + 1) * factor_x).min(w);
            let mut sum = 0u64;
            let mut count = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += src.get_pixel(x, y).0[0] as u64;
                    count += 1;
                }
            }
            let value = if count == 0 {
                0
            } else {
                (sum + count / 2) / count
            };
            out.put_pixel(ox, oy, image::Luma([value.min(255) as u8]));
        }
    }
    out
}

fn pillow_rgb_to_luma(img: &RgbImage) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = img.get_pixel(x, y).0;
            let luma =
                (19595u32 * r as u32 + 38470u32 * g as u32 + 7471u32 * b as u32 + 32768) >> 16;
            out.put_pixel(x, y, image::Luma([luma.min(255) as u8]));
        }
    }
    out
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
    let new_w = ((w as f32 * scale).floor() as u32).max(cfg.input_width);
    let new_h = ((h as f32 * scale).floor() as u32).max(cfg.input_height);
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

fn default_quality_max_side() -> u32 {
    1024
}

fn default_quality_clip_input_size() -> u32 {
    224
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
    fn distance_bbox_and_nms_follow_scrfd_geometry() {
        let bbox = distance2bbox([10.0, 20.0], [2.0, 3.0, 4.0, 5.0], 640.0);
        assert_eq!(bbox, [8.0, 17.0, 14.0, 25.0]);
        let faces = vec![
            DetectorFace {
                bbox: [0.0, 0.0, 20.0, 20.0],
                pre_bbox: [0.0, 0.0, 20.0, 20.0],
                det_score: 0.9,
                kps: None,
                pre_kps: None,
            },
            DetectorFace {
                bbox: [2.0, 2.0, 22.0, 22.0],
                pre_bbox: [2.0, 2.0, 22.0, 22.0],
                det_score: 0.8,
                kps: None,
                pre_kps: None,
            },
            DetectorFace {
                bbox: [80.0, 80.0, 100.0, 100.0],
                pre_bbox: [80.0, 80.0, 100.0, 100.0],
                det_score: 0.7,
                kps: None,
                pre_kps: None,
            },
        ];
        assert_eq!(nms(&faces, 0.4), vec![0, 2]);
    }

    #[test]
    fn rgb_chw_preprocess_matches_swap_rb_effect() {
        let img = RgbImage::from_pixel(1, 1, Rgb([10, 20, 30]));
        let arr = rgb_to_chw(&img, 0.0, 1.0).expect("preprocess");
        assert_eq!(arr.shape(), &[1, 3, 1, 1]);
        assert_eq!(arr[[0, 0, 0, 0]], 10.0);
        assert_eq!(arr[[0, 1, 0, 0]], 20.0);
        assert_eq!(arr[[0, 2, 0, 0]], 30.0);
    }

    #[test]
    fn arcface_alignment_transform_maps_landmarks_to_template() {
        let src = ARC_FACE_TEMPLATE;
        let dst = ARC_FACE_TEMPLATE;
        let m = estimate_similarity(&src, &dst).expect("similarity");
        for point in src {
            let mapped = apply_affine(m, point);
            assert!((mapped[0] - point[0]).abs() < 1e-3);
            assert!((mapped[1] - point[1]).abs() < 1e-3);
        }
        let inv = invert_affine(m).expect("invert");
        let p = apply_affine(inv, apply_affine(m, [12.0, 34.0]));
        assert!((p[0] - 12.0).abs() < 1e-3);
        assert!((p[1] - 34.0).abs() < 1e-3);
    }

    #[test]
    fn arcface_alignment_warps_source_landmarks_to_template() {
        let src = ARC_FACE_TEMPLATE.map(|[x, y]| [x + 8.0, y + 5.0]);
        let mut img = RgbImage::from_pixel(128, 128, Rgb([0, 0, 0]));
        for [x, y] in src {
            for yy in (y as i32 - 1)..=(y as i32 + 1) {
                for xx in (x as i32 - 1)..=(x as i32 + 1) {
                    if xx >= 0 && yy >= 0 {
                        img.put_pixel(xx as u32, yy as u32, Rgb([255, 255, 255]));
                    }
                }
            }
        }
        let (aligned, _) =
            align_face(&DynamicImage::ImageRgb8(img), &src, 112).expect("align face");
        for [x, y] in ARC_FACE_TEMPLATE {
            let mut local_sum = 0u32;
            for yy in (y as i32 - 1)..=(y as i32 + 1) {
                for xx in (x as i32 - 1)..=(x as i32 + 1) {
                    local_sum += aligned.get_pixel(xx as u32, yy as u32).0[0] as u32;
                }
            }
            assert!(local_sum > 200, "landmark was not warped to template");
        }
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
    fn quality_preprocess_respects_fixed_clip_input_shape() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 24, Rgb([128, 64, 32])));
        let cfg = ExpertQualityPreprocessor {
            clipiqa_input_width: 16,
            clipiqa_input_height: 12,
            ..ExpertQualityPreprocessor::default()
        };
        let arr = preprocess_quality_nchw(&img, Some(16), Some(12), cfg.max_side, &cfg)
            .expect("quality preprocess");
        assert_eq!(arr.shape(), &[1, 3, 12, 16]);
        assert!(arr.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dinov2_golden_fixture_matches_when_configured() {
        let Ok(component_dir) = std::env::var("PIANKE_EXPERT_COMPONENT_DIR") else {
            return;
        };
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let fixture = workspace
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
        let items = value["items"].as_array().expect("items");
        assert!(!items.is_empty(), "DINOv2 golden fixture has no items");
        let mut rust_infos = Vec::new();
        for (idx, item) in items.iter().enumerate() {
            let path = fixture_image_path(workspace, item);
            let expected = item["embedding"]
                .as_array()
                .expect("embedding")
                .iter()
                .map(|v| v.as_f64().expect("embedding value") as f32)
                .collect::<Vec<_>>();
            let actual = model.extract_path(&path).expect("extract DINOv2");
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual.len(), 384, "DINOv2 embedding dimension changed");
            assert!(
                (l2_norm(&actual) - 1.0).abs() <= 1e-4,
                "Rust DINOv2 embedding is not L2 normalized for {}",
                path.display()
            );
            assert!(
                (l2_norm(&expected) - 1.0).abs() <= 1e-4,
                "Python DINOv2 embedding is not L2 normalized for {}",
                path.display()
            );
            let cosine = cosine(&actual, &expected);
            assert!(
                cosine >= 0.999,
                "DINOv2 cosine below threshold for {}: {cosine}",
                path.display()
            );
            rust_infos.push(fixture_fast_info(item, idx));
        }
        assert_expert_grouping_matches_fixture(&value, &rust_infos);
    }

    #[test]
    fn insightface_golden_fixture_matches_when_configured() {
        let Ok(component_dir) = std::env::var("PIANKE_EXPERT_COMPONENT_DIR") else {
            return;
        };
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let fixture = workspace
            .join("fixtures")
            .join("expert_parity")
            .join("insightface.json");
        if !fixture.exists() {
            return;
        }
        let text = fs::read_to_string(&fixture).expect("read InsightFace golden fixture");
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("parse InsightFace golden fixture");
        let mut model = InsightFaceModels::from_component_dir(Path::new(&component_dir))
            .expect("load InsightFace models");
        let items = value["items"].as_array().expect("items");
        assert!(!items.is_empty(), "InsightFace golden fixture has no items");
        let mut rust_infos = Vec::new();
        for (item_idx, item) in items.iter().enumerate() {
            let path = fixture_image_path(workspace, item);
            let img = load_insightface_image(&path).expect("load fixture image");
            let actual = model.extract_faces(&img).expect("extract faces");
            let expected = item["faces"].as_array().expect("faces");
            assert_eq!(
                actual.len(),
                expected.len(),
                "face count mismatch for {}",
                path.display()
            );
            for (idx, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
                let exp_bbox = exp["bbox"]
                    .as_array()
                    .expect("bbox")
                    .iter()
                    .map(|v| v.as_f64().expect("bbox value") as f32)
                    .collect::<Vec<_>>();
                let iou = bbox_iou(
                    act.bbox,
                    [exp_bbox[0], exp_bbox[1], exp_bbox[2], exp_bbox[3]],
                );
                assert!(
                    iou >= 0.98 || bbox_max_abs_delta(act.bbox, &exp_bbox) <= 2.0,
                    "bbox mismatch for {} face {}: iou={iou}, actual={:?}, expected={:?}",
                    path.display(),
                    idx,
                    act.bbox,
                    exp_bbox
                );
                let expected_embedding = exp["embedding"]
                    .as_array()
                    .expect("embedding")
                    .iter()
                    .map(|v| v.as_f64().expect("embedding value") as f32)
                    .collect::<Vec<_>>();
                let cosine = cosine(&act.embedding, &expected_embedding);
                assert!(
                    cosine >= 0.999,
                    "embedding cosine below threshold for {} face {}: {cosine}, actual_bbox={:?}, expected_bbox={:?}, actual_kps={:?}, expected_kps={:?}",
                    path.display(),
                    idx,
                    act.bbox,
                    exp_bbox,
                    act.kps,
                    optional_points(exp.get("kps"))
                );
                if let Some(exp_kps) = optional_points(exp.get("kps")) {
                    let act_kps = act.kps.as_ref().expect("actual 5-point landmarks");
                    assert_points_close(act_kps, &exp_kps, 2.0, &path, idx, "5-point landmarks");
                }
                if let Some(exp_landmark) = optional_points(exp.get("landmark_2d_68")) {
                    let act_landmark = act
                        .landmark_2d_68
                        .as_ref()
                        .expect("actual 68-point landmarks");
                    assert_points_close(
                        act_landmark,
                        &exp_landmark,
                        3.0,
                        &path,
                        idx,
                        "68-point landmarks",
                    );
                }
            }
            let signals = face_signals_from_data(&actual, &img);
            assert_face_signals_close(item.get("face_signals"), &signals, &path);
            rust_infos.push(fixture_fast_info_with_actual_faces(item, item_idx, &actual));
        }
        assert_expert_grouping_matches_fixture(&value, &rust_infos);
    }

    #[test]
    fn insightface_debug_fixture_matches_when_configured() {
        let Ok(component_dir) = std::env::var("PIANKE_EXPERT_COMPONENT_DIR") else {
            return;
        };
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let fixture = workspace
            .join("fixtures")
            .join("expert_parity")
            .join("insightface_debug.json");
        if !fixture.exists() {
            return;
        }
        let text = fs::read_to_string(&fixture).expect("read InsightFace debug fixture");
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("parse InsightFace debug fixture");
        let mut detector = load_session(
            &Path::new(&component_dir).join(FACE_DET_MODEL_PATH),
            "InsightFace detector",
        )
        .expect("load detector");
        let items = value["items"].as_array().expect("items");
        assert!(
            items.iter().any(|item| item.get("faces_debug").is_some()),
            "InsightFace debug fixture has no faces_debug entries"
        );
        for item in items {
            let Some(debug) = item.get("faces_debug") else {
                continue;
            };
            let path = fixture_image_path(workspace, item);
            let img = load_insightface_image(&path).expect("load fixture image");
            let det_input = DetectorInput::from_image(&img, DET_SIZE).expect("detector input");
            let expected_size = debug["pre_size"].as_array().expect("pre_size");
            assert_eq!(
                [
                    det_input.source_rgb.width() as u64,
                    det_input.source_rgb.height() as u64
                ],
                [
                    expected_size[0].as_u64().expect("pre width"),
                    expected_size[1].as_u64().expect("pre height")
                ],
                "pre-resized dimensions mismatch for {}",
                path.display()
            );
            if let Some(expected_hash) = debug["source_rgb_sha256"].as_str() {
                assert_eq!(
                    rgb_sha256(&det_input.source_rgb),
                    expected_hash,
                    "pre-resized RGB checksum mismatch for {}",
                    path.display()
                );
            }
            let actual = run_detector(&mut detector, &det_input).expect("run detector");
            let expected_faces = debug["faces"].as_array().expect("debug faces");
            assert_eq!(
                actual.len(),
                expected_faces.len(),
                "debug face count mismatch for {}",
                path.display()
            );
            for (idx, (act, exp)) in actual.iter().zip(expected_faces.iter()).enumerate() {
                let exp_pre_bbox = exp["pre_bbox"]
                    .as_array()
                    .expect("pre_bbox")
                    .iter()
                    .map(|v| v.as_f64().expect("pre_bbox value") as f32)
                    .collect::<Vec<_>>();
                assert!(
                    bbox_max_abs_delta(act.pre_bbox, &exp_pre_bbox) <= 2.0,
                    "pre_bbox mismatch for {} face {}: actual={:?}, expected={:?}",
                    path.display(),
                    idx,
                    act.pre_bbox,
                    exp_pre_bbox
                );
                if let Some(exp_pre_kps) = optional_points(exp.get("pre_kps")) {
                    let act_pre_kps = act.pre_kps.as_ref().expect("actual pre_kps");
                    assert_points_close(
                        act_pre_kps,
                        &exp_pre_kps,
                        0.5,
                        &path,
                        idx,
                        "pre-resized 5-point landmarks",
                    );
                    if let Some(expected_crop_hash) = exp["arcface_crop_sha256"].as_str() {
                        let expected_matrix = exp
                            .get("arcface")
                            .and_then(|v| v.get("matrix"))
                            .map(read_arcface_matrix);
                        let exp_src = [
                            exp_pre_kps[0],
                            exp_pre_kps[1],
                            exp_pre_kps[2],
                            exp_pre_kps[3],
                            exp_pre_kps[4],
                        ];
                        let exp_dst = ARC_FACE_TEMPLATE;
                        let debug_forward =
                            estimate_similarity(&exp_src, &exp_dst).expect("estimate debug face");
                        let expected_matrix_bgr_hash = expected_matrix
                            .map(|matrix| {
                                let matrix_aligned =
                                    warp_rgb_affine_with_matrix(&det_input.source_rgb, 112, matrix)
                                        .expect("warp fixture matrix");
                                rgb_bgr_sha256(&matrix_aligned)
                            })
                            .unwrap_or_default();
                        assert_eq!(
                            expected_matrix_bgr_hash,
                            expected_crop_hash,
                            "fixture ArcFace matrix checksum mismatch for {} face {}",
                            path.display(),
                            idx
                        );
                        assert_matrix_close(
                            debug_forward,
                            expected_matrix.expect("fixture ArcFace matrix"),
                            0.001,
                            &path,
                            idx,
                            "ArcFace matrix",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn quality_golden_fixture_matches_when_configured() {
        let Ok(component_dir) = std::env::var("PIANKE_EXPERT_COMPONENT_DIR") else {
            return;
        };
        let component_dir = Path::new(&component_dir);
        if !ExpertQualityModels::component_files_ready(component_dir) {
            return;
        }
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let fixture = workspace
            .join("fixtures")
            .join("expert_parity")
            .join("quality.json");
        if !fixture.exists() {
            return;
        }
        let text = fs::read_to_string(&fixture).expect("read Expert quality golden fixture");
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("parse Expert quality golden fixture");
        let items = value["items"].as_array().expect("items");
        if items
            .iter()
            .all(|item| item.get("quality_scores").is_none())
        {
            return;
        }
        let preprocessor =
            load_quality_preprocessor(component_dir).expect("load quality preprocessor");
        if quality_fixture_exceeds_fixed_export_shape(workspace, items, &preprocessor) {
            eprintln!(
                "skip quality golden parity: current quality ONNX component uses fixed export input shape and does not match real-photo Python pyiqa resizing"
            );
            return;
        }
        let mut model =
            ExpertQualityModels::from_component_dir(component_dir).expect("load quality models");
        for item in items {
            let Some(expected_scores) = item.get("quality_scores") else {
                continue;
            };
            let path = fixture_image_path(workspace, item);
            let img = image::open(&path).expect("load fixture image");
            let actual = model.analyze(&img).expect("extract quality scores");
            assert_optional_score_close(
                actual.musiq_score,
                expected_scores.get("musiq_score"),
                0.1,
                &path,
                "musiq_score",
            );
            assert_optional_score_close(
                actual.clipiqa_score,
                expected_scores.get("clipiqa_score"),
                0.02,
                &path,
                "clipiqa_score",
            );

            let mut record = crate::InfoRecord {
                info: pianke_core::fast::FastImageInfo {
                    quality: Some(pianke_core::fast::QualityInfo {
                        quality_score: Some(80.0),
                        ..pianke_core::fast::QualityInfo::default()
                    }),
                    ..pianke_core::fast::FastImageInfo::default()
                },
                companions: Vec::new(),
            };
            let strength = expected_scores
                .get("strength")
                .and_then(|v| v.as_str())
                .unwrap_or("standard");
            crate::apply_expert_quality_scores(&mut record, actual, strength);
            let q = record.info.quality.expect("quality");
            if let Some(expected_quality) = item.get("quality") {
                assert_eq!(
                    q.flags.iter().any(|f| f == "low_aesthetic"),
                    expected_quality["flags"]
                        .as_array()
                        .map(|flags| flags.iter().any(|f| f.as_str() == Some("low_aesthetic")))
                        .unwrap_or(false),
                    "low_aesthetic flag mismatch for {}",
                    path.display()
                );
                if q.flags.iter().any(|f| f == "low_aesthetic") {
                    assert_eq!(q.auto_reject, Some(true));
                    assert!(q.reject_reason.is_some());
                }
            }
        }
    }

    fn quality_fixture_exceeds_fixed_export_shape(
        workspace: &Path,
        items: &[serde_json::Value],
        cfg: &ExpertQualityPreprocessor,
    ) -> bool {
        let Some(musiq_w) = cfg.musiq_input_width else {
            return false;
        };
        let Some(musiq_h) = cfg.musiq_input_height else {
            return false;
        };
        items.iter().any(|item| {
            if item.get("quality_scores").is_none() {
                return false;
            }
            let path = fixture_image_path(workspace, item);
            let Ok((w, h)) = image::image_dimensions(&path) else {
                return false;
            };
            (w, h) != (musiq_w, musiq_h)
                || (w, h) != (cfg.clipiqa_input_width, cfg.clipiqa_input_height)
        })
    }

    fn fixture_image_path(workspace: &Path, item: &serde_json::Value) -> std::path::PathBuf {
        let raw = item["path"].as_str().expect("path");
        let path = Path::new(raw);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            workspace.join(path)
        }
    }

    fn fixture_fast_info(item: &serde_json::Value, idx: usize) -> pianke_core::fast::FastImageInfo {
        let dinov2 = item["embedding"]
            .as_array()
            .expect("embedding")
            .iter()
            .map(|v| v.as_f64().expect("embedding value") as f32)
            .collect::<Vec<_>>();
        let face_embeddings = item["faces"]
            .as_array()
            .map(|faces| {
                faces
                    .iter()
                    .map(|face| {
                        face["embedding"]
                            .as_array()
                            .expect("face embedding")
                            .iter()
                            .map(|v| v.as_f64().expect("face embedding value") as f32)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        pianke_core::fast::FastImageInfo {
            path: item["path"].as_str().expect("path").to_string(),
            phash: Some("0".repeat(16)),
            timestamp: Some(idx as f64 * 10.0),
            mtime: Some(idx as f64 * 10.0),
            dinov2: Some(dinov2),
            face_embeddings,
            ..pianke_core::fast::FastImageInfo::default()
        }
    }

    fn fixture_fast_info_with_actual_faces(
        item: &serde_json::Value,
        idx: usize,
        faces: &[FaceInfo],
    ) -> pianke_core::fast::FastImageInfo {
        let mut info = fixture_fast_info(item, idx);
        info.face_embeddings = faces.iter().map(|face| face.embedding.clone()).collect();
        info
    }

    fn assert_expert_grouping_matches_fixture(
        value: &serde_json::Value,
        infos: &[pianke_core::fast::FastImageInfo],
    ) {
        let Some(expected) = value["grouping"]["group_indices"].as_array() else {
            return;
        };
        let expected = expected
            .iter()
            .map(|group| {
                group
                    .as_array()
                    .expect("group array")
                    .iter()
                    .map(|v| v.as_u64().expect("group index") as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let actual = crate::expert_cluster(infos);
        assert_eq!(actual, expected, "Expert/Tycoon grouping parity mismatch");
    }

    fn bbox_max_abs_delta(actual: [f32; 4], expected: &[f32]) -> f32 {
        actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0, f32::max)
    }

    fn optional_points(value: Option<&serde_json::Value>) -> Option<Vec<[f32; 2]>> {
        let value = value?;
        if value.is_null() {
            return None;
        }
        Some(
            value
                .as_array()
                .expect("points array")
                .iter()
                .map(|point| {
                    let arr = point.as_array().expect("point");
                    [
                        arr[0].as_f64().expect("x") as f32,
                        arr[1].as_f64().expect("y") as f32,
                    ]
                })
                .collect(),
        )
    }

    fn rgb_sha256(img: &RgbImage) -> String {
        let mut hasher = Sha256::new();
        hasher.update(img.as_raw());
        format!("{:x}", hasher.finalize())
    }

    fn rgb_bgr_sha256(img: &RgbImage) -> String {
        let mut hasher = Sha256::new();
        for pixel in img.pixels() {
            let [r, g, b] = pixel.0;
            hasher.update([b, g, r]);
        }
        format!("{:x}", hasher.finalize())
    }

    fn read_arcface_matrix(value: &serde_json::Value) -> [f64; 6] {
        let rows = value.as_array().expect("arcface matrix rows");
        let r0 = rows[0].as_array().expect("arcface matrix row 0");
        let r1 = rows[1].as_array().expect("arcface matrix row 1");
        [
            r0[0].as_f64().expect("m00"),
            r0[1].as_f64().expect("m01"),
            r0[2].as_f64().expect("m02"),
            r1[0].as_f64().expect("m10"),
            r1[1].as_f64().expect("m11"),
            r1[2].as_f64().expect("m12"),
        ]
    }

    fn assert_matrix_close(
        actual: [f64; 6],
        expected: [f64; 6],
        tolerance: f64,
        path: &Path,
        face_idx: usize,
        label: &str,
    ) {
        let max_delta = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_delta <= tolerance,
            "{label} mismatch for {} face {}: max_delta={max_delta}, actual={actual:?}, expected={expected:?}",
            path.display(),
            face_idx
        );
    }

    fn assert_points_close(
        actual: &[[f32; 2]],
        expected: &[[f32; 2]],
        tolerance_px: f32,
        path: &Path,
        face_idx: usize,
        label: &str,
    ) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label} count mismatch for {} face {}",
            path.display(),
            face_idx
        );
        let max_delta = actual
            .iter()
            .zip(expected.iter())
            .flat_map(|(a, b)| [(a[0] - b[0]).abs(), (a[1] - b[1]).abs()])
            .fold(0.0, f32::max);
        assert!(
            max_delta <= tolerance_px,
            "{label} mismatch for {} face {}: max_delta={max_delta}",
            path.display(),
            face_idx
        );
    }

    fn assert_face_signals_close(
        expected: Option<&serde_json::Value>,
        actual: &FaceSignals,
        path: &Path,
    ) {
        let Some(expected) = expected else {
            return;
        };
        assert_eq!(
            actual.face_count,
            expected["face_count"].as_u64().unwrap_or(0) as usize,
            "face_count signal mismatch for {}",
            path.display()
        );
        assert_eq!(
            actual.face_clipped,
            expected["face_clipped"].as_bool().unwrap_or(false),
            "face_clipped signal mismatch for {}",
            path.display()
        );
        assert_optional_f64_close(
            actual.face_sharpness,
            expected.get("face_sharpness"),
            5.0,
            path,
            "face_sharpness",
            Some(&format!(
                "actual_faces={:?}, expected_faces={}",
                actual.faces_detail,
                expected
                    .get("faces_detail")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "null".to_string())
            )),
        );
        assert_optional_f64_close(
            actual.eyes_open_score,
            expected.get("eyes_open_score"),
            0.03,
            path,
            "eyes_open_score",
            None,
        );
        assert_eq!(
            actual.faces_detail.len(),
            expected["faces_detail"].as_array().map_or(0, Vec::len),
            "faces_detail count mismatch for {}",
            path.display()
        );
    }

    fn assert_optional_f64_close(
        actual: Option<f64>,
        expected: Option<&serde_json::Value>,
        tolerance: f64,
        path: &Path,
        label: &str,
        context: Option<&str>,
    ) {
        let expected = expected.and_then(|v| v.as_f64());
        match (actual, expected) {
            (Some(a), Some(e)) => assert!(
                (a - e).abs() <= tolerance,
                "{label} mismatch for {}: actual={a}, expected={e}{}{}",
                path.display(),
                context.map(|_| "\n").unwrap_or(""),
                context.unwrap_or("")
            ),
            (None, None) => {}
            other => panic!(
                "{label} presence mismatch for {}: {other:?}",
                path.display()
            ),
        }
    }

    fn assert_optional_score_close(
        actual: Option<f64>,
        expected: Option<&serde_json::Value>,
        tolerance: f64,
        path: &Path,
        label: &str,
    ) {
        let expected = expected.and_then(|v| v.as_f64());
        match (actual, expected) {
            (Some(a), Some(e)) => assert!(
                (a - e).abs() <= tolerance,
                "{label} mismatch for {}: actual={a}, expected={e}",
                path.display()
            ),
            (None, None) => {}
            other => panic!(
                "{label} presence mismatch for {}: {other:?}",
                path.display()
            ),
        }
    }

    fn l2_norm(values: &[f32]) -> f64 {
        values
            .iter()
            .map(|v| {
                let v = *v as f64;
                v * v
            })
            .sum::<f64>()
            .sqrt()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
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
        dot / (na.sqrt() * nb.sqrt()).max(1e-12)
    }
}
