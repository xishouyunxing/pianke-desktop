use image::{imageops::FilterType, DynamicImage};
use ndarray::Array4;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

const DINO_MODEL_PATH: &str = "models/dinov2-small.onnx";
const PREPROCESSOR_PATH: &str = "preprocessor.json";

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

pub fn load_preprocessor(component_dir: &Path) -> Result<Dinov2Preprocessor, String> {
    let path = component_dir.join(PREPROCESSOR_PATH);
    if !path.exists() {
        return Ok(Dinov2Preprocessor::default());
    }
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("读取 DINOv2 preprocessor.json 失败: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("解析 DINOv2 preprocessor.json 失败: {e}"))
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
