use axum::http::StatusCode;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

const CONFIG_FILENAME: &str = "provider.json";
const SECRET_FILENAME: &str = "provider.key";
const TEST_IMAGE_DATA_URL: &str = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/ASP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/ASP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Ar//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/EFBQRAQAAAAAAAAAAAAAAAAAAARD/2gAIAQMBAT8QH//EFBQRAQAAAAAAAAAAAAAAAAAAARD/2gAIAQIBAT8QH//EFBABAQAAAAAAAAAAAAAAAAAAARD/2gAIAQEAAT8QH//Z";

#[derive(Debug, Clone)]
pub struct LlmProviderManager {
    config_path: PathBuf,
    secret_path: PathBuf,
}

impl LlmProviderManager {
    pub fn new(root_dir: PathBuf) -> Self {
        Self {
            config_path: root_dir.join(CONFIG_FILENAME),
            secret_path: root_dir.join(SECRET_FILENAME),
        }
    }

    pub fn provider_status(&self) -> ProviderStatus {
        let config = self.load_config().ok().flatten();
        let key_configured = self.load_api_key().is_some();
        ProviderStatus {
            configured: config
                .as_ref()
                .map(|c| c.is_usable() && key_configured)
                .unwrap_or(false),
            key_configured,
            config,
            protocols: LlmProtocol::all(),
        }
    }

    pub fn save_provider(&self, req: SaveProviderRequest) -> Result<ProviderStatus, ApiError> {
        let config = ProviderConfig::from_request(&req)?;
        fs::create_dir_all(
            self.config_path
                .parent()
                .ok_or_else(|| ApiError::internal("invalid provider config path"))?,
        )
        .map_err(|e| ApiError::internal(format!("create provider config dir failed: {e}")))?;
        let text = serde_json::to_string_pretty(&config)
            .map_err(|e| ApiError::internal(format!("serialize provider config failed: {e}")))?;
        fs::write(&self.config_path, text)
            .map_err(|e| ApiError::internal(format!("write provider config failed: {e}")))?;

        if let Some(key) = req
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            fs::write(&self.secret_path, key)
                .map_err(|e| ApiError::internal(format!("write provider api key failed: {e}")))?;
        }

        Ok(self.provider_status())
    }

    pub fn clear_provider(&self) -> Result<(), ApiError> {
        remove_if_exists(&self.config_path)?;
        remove_if_exists(&self.secret_path)?;
        Ok(())
    }

    pub async fn list_models(&self) -> Result<Value, ApiError> {
        let Some(config) = self.load_config()? else {
            return Err(ApiError::precondition("请先配置 AI 服务商"));
        };
        let Some(api_key) = self.load_api_key() else {
            return Err(ApiError::precondition("请先保存 AI 服务商 API Key"));
        };

        if config.protocol == LlmProtocol::AnthropicMessages {
            return Ok(json!({
                "models": [],
                "manual_supported": true,
                "error": "Anthropic Messages API 第一版不自动拉取模型，请手填 model id"
            }));
        }

        let url = config.endpoint("models")?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| ApiError::internal(format!("create http client failed: {e}")))?;
        let resp = client
            .get(url)
            .headers(config.headers(&api_key)?)
            .send()
            .await
            .map_err(|e| ApiError::bad_gateway(format!("拉取模型列表失败: {e}")))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| ApiError::bad_gateway(format!("读取模型列表失败: {e}")))?;
        if !status.is_success() {
            return Ok(json!({
                "models": [],
                "manual_supported": true,
                "error": format!("模型列表接口返回 HTTP {status}: {}", truncate(&body, 180))
            }));
        }
        let value: Value = serde_json::from_str(&body)
            .map_err(|e| ApiError::bad_gateway(format!("模型列表 JSON 解析失败: {e}")))?;
        Ok(json!({
            "models": parse_openai_models(&value),
            "manual_supported": true
        }))
    }

    pub async fn test_provider(&self, req: Option<SaveProviderRequest>) -> Result<Value, ApiError> {
        let (config, api_key) = if let Some(req) = req {
            let config = ProviderConfig::from_request(&req)?;
            let key = req
                .api_key
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .or_else(|| self.load_api_key())
                .ok_or_else(|| ApiError::precondition("请填写 AI 服务商 API Key"))?;
            (config, key)
        } else {
            (
                self.load_config()?
                    .ok_or_else(|| ApiError::precondition("请先配置 AI 服务商"))?,
                self.load_api_key()
                    .ok_or_else(|| ApiError::precondition("请先保存 AI 服务商 API Key"))?,
            )
        };
        let payloads = build_probe_payloads(&config);
        let url = config.completion_endpoint()?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| ApiError::internal(format!("create http client failed: {e}")))?;
        let headers = config.headers(&api_key)?;
        let mut last_error = None;
        for (payload_index, payload) in payloads.iter().enumerate() {
            let resp = client
                .post(&url)
                .headers(headers.clone())
                .json(payload)
                .send()
                .await
                .map_err(|e| ApiError::bad_gateway(format!("AI 服务商连接失败: {e}")))?;
            let status = resp.status();
            let body = resp
                .text()
                .await
                .map_err(|e| ApiError::bad_gateway(format!("读取 AI 服务商响应失败: {e}")))?;
            if !status.is_success() {
                let compatibility_retry = config.protocol == LlmProtocol::OpenaiChatCompletions
                    && payload_index == 0
                    && is_chat_payload_compatibility_error(status, &body);
                if compatibility_retry {
                    last_error = Some(format!("HTTP {status}: {}", truncate(&body, 180)));
                    continue;
                }
                return Err(ApiError::bad_gateway(format!(
                    "AI 服务商返回 HTTP {status}: {}",
                    truncate(&body, 180)
                )));
            }
            return Ok(json!({
                "ok": true,
                "protocol": config.protocol,
                "model": config.model,
                "response_preview": truncate(&body, 240)
            }));
        }
        Err(ApiError::bad_gateway(format!(
            "AI 服务商测试失败: {}",
            last_error.unwrap_or_else(|| "unknown error".to_string())
        )))
    }

    pub async fn judge_image(
        &self,
        config: &ProviderConfig,
        image_data_url: &str,
        prompt: &str,
    ) -> Result<JudgeVerdict, ApiError> {
        let api_key = self
            .load_api_key()
            .ok_or_else(|| ApiError::precondition("AI provider API key missing"))?;
        let payloads = build_judge_payloads(config, image_data_url, prompt);
        let url = config.completion_endpoint()?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| ApiError::internal(format!("create http client failed: {e}")))?;

        let attempts = 4u32;
        let mut last_error: Option<String> = None;
        for (payload_index, payload) in payloads.iter().enumerate() {
            let is_fallback_payload = payload_index > 0;
            for attempt in 1..=attempts {
                let resp = client
                    .post(&url)
                    .headers(config.headers(&api_key)?)
                    .json(payload)
                    .send()
                    .await;
                let resp = match resp {
                    Ok(resp) => resp,
                    Err(e) => {
                        last_error = Some(e.to_string());
                        if attempt < attempts {
                            tokio::time::sleep(Duration::from_millis(300 * u64::from(attempt)))
                                .await;
                            continue;
                        }
                        return Err(ApiError::bad_gateway(format!(
                            "AI 服务商请求失败: {e} (base_url={}, model={})",
                            config.base_url, config.model
                        )));
                    }
                };
                let status = resp.status();
                let body = resp
                    .text()
                    .await
                    .map_err(|e| ApiError::bad_gateway(format!("read AI response failed: {e}")))?;
                if status.as_u16() == 429 || status.is_server_error() {
                    last_error = Some(format!("HTTP {status}: {}", truncate(&body, 180)));
                    if attempt < attempts {
                        tokio::time::sleep(Duration::from_millis(400 * u64::from(attempt))).await;
                        continue;
                    }
                }
                if !status.is_success() {
                    let compatibility_retry = config.protocol == LlmProtocol::OpenaiChatCompletions
                        && !is_fallback_payload
                        && is_chat_payload_compatibility_error(status, &body);
                    if compatibility_retry {
                        last_error = Some(format!("HTTP {status}: {}", truncate(&body, 180)));
                        break;
                    }
                    return Err(ApiError::bad_gateway(format!(
                        "AI 服务商返回 HTTP {status} (base_url={}, model={}): {}",
                        config.base_url,
                        config.model,
                        truncate(&body, 240)
                    )));
                }
                let value: Value = serde_json::from_str(&body).map_err(|e| {
                    ApiError::bad_gateway(format!(
                        "AI 响应不是 JSON: {e} (base_url={}, model={}): {}",
                        config.base_url,
                        config.model,
                        truncate(&body, 240)
                    ))
                })?;
                let parsed = parse_judge_response(&config.protocol, &value).map_err(|e| {
                    ApiError::bad_gateway(format!(
                        "AI 响应解析失败 (base_url={}, model={}): {e}",
                        config.base_url, config.model
                    ))
                })?;
                return Ok(JudgeVerdict {
                    verdict: parsed["verdict"].as_str().unwrap_or("reject").to_string(),
                    reason: parsed["reason"].as_str().unwrap_or("").to_string(),
                    flaws: parsed["flaws"].as_str().map(str::to_string),
                    fixable: parsed["fixable"].as_str().map(str::to_string),
                });
            }
        }
        Err(ApiError::bad_gateway(format!(
            "AI 服务商多次重试仍失败 (base_url={}, model={}): {}",
            config.base_url,
            config.model,
            last_error.unwrap_or_else(|| "unknown error".to_string())
        )))
    }

    pub fn require_tycoon_ready(
        &self,
        model_override: Option<&str>,
    ) -> Result<ProviderConfig, ApiError> {
        let mut config = self
            .load_config()?
            .ok_or_else(|| ApiError::precondition("请先配置 AI 服务商"))?;
        if let Some(model) = model_override.map(str::trim).filter(|s| !s.is_empty()) {
            config.model = model.to_string();
        }
        if !config.is_usable() {
            return Err(ApiError::precondition(
                "AI 服务商配置不完整，请填写协议、Base URL 和模型 ID",
            ));
        }
        if self.load_api_key().is_none() {
            return Err(ApiError::precondition("请先保存 AI 服务商 API Key"));
        }
        Ok(config)
    }

    fn load_config(&self) -> Result<Option<ProviderConfig>, ApiError> {
        if !self.config_path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&self.config_path)
            .map_err(|e| ApiError::internal(format!("read provider config failed: {e}")))?;
        let config = serde_json::from_str(&text)
            .map_err(|e| ApiError::internal(format!("parse provider config failed: {e}")))?;
        Ok(Some(config))
    }

    fn load_api_key(&self) -> Option<String> {
        fs::read_to_string(&self.secret_path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LlmProtocol {
    OpenaiChatCompletions,
    OpenaiResponses,
    AnthropicMessages,
}

impl LlmProtocol {
    fn all() -> Vec<&'static str> {
        vec![
            "openai_chat_completions",
            "openai_responses",
            "anthropic_messages",
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub protocol: LlmProtocol,
    pub base_url: String,
    pub api_key_ref: String,
    pub model: String,
    pub display_name: String,
    pub max_concurrency: u32,
    pub timeout_seconds: u64,
}

impl ProviderConfig {
    fn from_request(req: &SaveProviderRequest) -> Result<Self, ApiError> {
        let protocol = req
            .protocol
            .clone()
            .ok_or_else(|| ApiError::bad_request("请选择 AI 协议"))?;
        let base_url = clean_required(&req.base_url, "请填写 Base URL")?;
        let model = clean_required(&req.model, "请填写模型 ID")?;
        Ok(Self {
            protocol,
            base_url: trim_trailing_slashes(&base_url),
            api_key_ref: "local_secret".to_string(),
            model,
            display_name: req
                .display_name
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("AI Provider")
                .to_string(),
            max_concurrency: req.max_concurrency.unwrap_or(4).clamp(1, 32),
            timeout_seconds: req.timeout_seconds.unwrap_or(45).clamp(5, 300),
        })
    }

    fn is_usable(&self) -> bool {
        !self.base_url.trim().is_empty() && !self.model.trim().is_empty()
    }

    fn endpoint(&self, path: &str) -> Result<String, ApiError> {
        Ok(format!(
            "{}/{}",
            trim_trailing_slashes(&self.base_url),
            path.trim_start_matches('/')
        ))
    }

    fn completion_endpoint(&self) -> Result<String, ApiError> {
        let trimmed = trim_trailing_slashes(&self.base_url);
        match self.protocol {
            LlmProtocol::OpenaiChatCompletions => {
                if ends_with_path(&trimmed, "chat/completions") {
                    Ok(trimmed)
                } else {
                    self.endpoint("chat/completions")
                }
            }
            LlmProtocol::OpenaiResponses => {
                if ends_with_path(&trimmed, "responses") {
                    Ok(trimmed)
                } else {
                    self.endpoint("responses")
                }
            }
            LlmProtocol::AnthropicMessages => {
                if ends_with_path(&trimmed, "messages") {
                    Ok(trimmed)
                } else {
                    self.endpoint("messages")
                }
            }
        }
    }

    fn headers(&self, api_key: &str) -> Result<HeaderMap, ApiError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        match self.protocol {
            LlmProtocol::OpenaiChatCompletions | LlmProtocol::OpenaiResponses => {
                let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
                    .map_err(|_| ApiError::bad_request("API Key 包含非法字符"))?;
                headers.insert(AUTHORIZATION, value);
            }
            LlmProtocol::AnthropicMessages => {
                let key = HeaderValue::from_str(api_key)
                    .map_err(|_| ApiError::bad_request("API Key 包含非法字符"))?;
                headers.insert("x-api-key", key);
                headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            }
        }
        Ok(headers)
    }
}

#[derive(Debug, Deserialize)]
pub struct SaveProviderRequest {
    pub protocol: Option<LlmProtocol>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub display_name: Option<String>,
    pub max_concurrency: Option<u32>,
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ProviderStatus {
    pub configured: bool,
    pub key_configured: bool,
    pub config: Option<ProviderConfig>,
    pub protocols: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeVerdict {
    pub verdict: String,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flaws: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixable: Option<String>,
}

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn precondition(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PRECONDITION_REQUIRED,
            message: message.into(),
        }
    }

    fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    pub fn into_json(self) -> (StatusCode, axum::Json<Value>) {
        (
            self.status,
            axum::Json(json!({
                "error": self.message,
                "manual_supported": true
            })),
        )
    }
}

pub fn build_judge_payload(config: &ProviderConfig, image_data_url: &str, prompt: &str) -> Value {
    build_judge_payloads(config, image_data_url, prompt)
        .into_iter()
        .next()
        .expect("judge payloads are non-empty")
}

fn build_judge_payloads(config: &ProviderConfig, image_data_url: &str, prompt: &str) -> Vec<Value> {
    match config.protocol {
        LlmProtocol::OpenaiChatCompletions => vec![
            json!({
                "model": config.model,
                "temperature": 0.0,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个图片质检助手。必须只输出严格 JSON，不要任何额外文字。格式：{\"verdict\":\"pass|reject\",\"reason\":\"...\",\"flaws\":\"...\",\"fixable\":\"...\"}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            }),
            json!({
                "model": config.model,
                "temperature": 0.0,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个图片质检助手。必须只输出严格 JSON，不要任何额外文字。格式：{\"verdict\":\"pass|reject\",\"reason\":\"...\",\"flaws\":\"...\",\"fixable\":\"...\"}"
                    },
                    {
                        "role": "user",
                        "content": format!("{prompt}\n\n图片：\n![image]({image_data_url})")
                    }
                ]
            }),
        ],
        LlmProtocol::OpenaiResponses => vec![json!({
            "model": config.model,
            "temperature": 0.0,
            "max_output_tokens": 512,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_data_url},
                    {"type": "input_text", "text": prompt}
                ]
            }]
        })],
        LlmProtocol::AnthropicMessages => vec![json!({
            "model": config.model,
            "max_tokens": 512,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    anthropic_image_block(image_data_url),
                    {"type": "text", "text": prompt}
                ]
            }]
        })],
    }
}

#[allow(dead_code)]
pub fn parse_judge_response(protocol: &LlmProtocol, value: &Value) -> Result<Value, String> {
    let text = match protocol {
        LlmProtocol::OpenaiChatCompletions => extract_chat_text(value),
        LlmProtocol::OpenaiResponses => value["output_text"]
            .as_str()
            .or_else(|| {
                value["output"].as_array()?.iter().find_map(|item| {
                    item["content"].as_array()?.iter().find_map(|part| {
                        (part["type"] == "output_text")
                            .then(|| part["text"].as_str())
                            .flatten()
                    })
                })
            })
            .unwrap_or("")
            .to_string(),
        LlmProtocol::AnthropicMessages => value["content"]
            .as_array()
            .and_then(|parts| {
                parts.iter().find_map(|part| {
                    (part["type"] == "text")
                        .then(|| part["text"].as_str())
                        .flatten()
                })
            })
            .unwrap_or("")
            .to_string(),
    };
    let cleaned = strip_code_fence(&text);
    // 1) 整体就是 JSON
    // 2) 否则尝试从字符串里抽出第一个 { ... } 子串
    let parsed: Value = match serde_json::from_str(&cleaned) {
        Ok(v) => v,
        Err(_) => match extract_first_json_object(&cleaned) {
            Some(candidate) => serde_json::from_str(&candidate).map_err(|e| {
                format!(
                    "AI 响应不是有效 JSON: {e}; content={}",
                    truncate(&cleaned, 200)
                )
            })?,
            None => {
                return Err(format!(
                    "AI 响应不是有效 JSON: 无法解析为对象; content={}",
                    truncate(&cleaned, 200)
                ));
            }
        },
    };
    let verdict = parsed["verdict"]
        .as_str()
        .unwrap_or("")
        .to_ascii_lowercase();
    if verdict != "pass" && verdict != "reject" {
        return Err("AI 响应 verdict 必须是 pass 或 reject".to_string());
    }
    if parsed["reason"].as_str().unwrap_or("").trim().is_empty() {
        return Err("AI 响应缺少 reason".to_string());
    }
    Ok(parsed)
}

/// 兼容多种 chat/completions 响应里 content 字段的形状：
/// - 官方 OpenAI: `choices[0].message.content` = string
/// - 某些第三方代理会把 content 包成数组 `[{type:"text", text:"..."}]`
/// - 还有一些直接 `choices[0].text` (legacy)
fn extract_chat_text(value: &Value) -> String {
    // 1) 官方格式
    if let Some(s) = value["choices"][0]["message"]["content"].as_str() {
        return s.to_string();
    }
    // 2) content 是数组（部分代理会重排成多模态数组）
    if let Some(arr) = value["choices"][0]["message"]["content"].as_array() {
        for part in arr {
            if let Some(s) = part.get("text").and_then(|v| v.as_str()) {
                return s.to_string();
            }
        }
        // 拼接所有 string 元素
        let joined: Vec<String> = arr
            .iter()
            .filter_map(|p| p.as_str().map(str::to_string))
            .collect();
        if !joined.is_empty() {
            return joined.join("\n");
        }
    }
    // 3) legacy text 字段
    if let Some(s) = value["choices"][0]["text"].as_str() {
        return s.to_string();
    }
    // 4) 错误信息字段
    if let Some(err) = value["error"]["message"].as_str() {
        return err.to_string();
    }
    String::new()
}

/// 去掉 markdown 代码块包裹：```json ... ```、```JSON ... ```、``` ... ```
fn strip_code_fence(text: &str) -> String {
    let trimmed = text.trim();
    // 找到开头的 ``` 行
    let after_open = if let Some(rest) = trimmed.strip_prefix("```") {
        // 跳过可选的语言标识
        let mut newline_idx = rest.find('\n').unwrap_or(rest.len());
        if newline_idx == 0 {
            newline_idx = 0;
        }
        rest[newline_idx..]
            .trim_start_matches('\n')
            .trim_start_matches('\r')
    } else {
        trimmed
    };
    let stripped = if let Some(rest) = after_open.strip_suffix("```") {
        rest.trim_end_matches('\n').trim_end_matches('\r')
    } else {
        after_open
    };
    stripped.trim().to_string()
}

/// 在文本中查找第一个 { ... } 平衡的 JSON 对象子串。
/// 简单实现：扫左括号，跟踪字符串字面量和转义，找到匹配的右括号。
fn extract_first_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut start: Option<usize> = None;
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        let c = b as char;
        if in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => {
                if start.is_none() {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(s) = start {
                            return Some(text[s..=i].to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn build_probe_payloads(config: &ProviderConfig) -> Vec<Value> {
    build_judge_payloads(
        config,
        TEST_IMAGE_DATA_URL,
        r#"只输出一行 JSON：{"verdict":"pass","reason":"连接测试","flaws":"无","fixable":"无"}"#,
    )
}

fn anthropic_image_block(data_url: &str) -> Value {
    let base64 = data_url
        .split_once(',')
        .map(|(_, data)| data)
        .unwrap_or(data_url);
    json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64
        }
    })
}

fn parse_openai_models(value: &Value) -> Vec<Value> {
    value["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|model| {
            let id = model["id"].as_str()?;
            Some(json!({
                "id": id,
                "label": id,
                "tier": "other"
            }))
        })
        .collect()
}

fn clean_required(value: &Option<String>, message: &str) -> Result<String, ApiError> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ApiError::bad_request(message))
}

fn trim_trailing_slashes(value: &str) -> String {
    value.trim().trim_end_matches('/').to_string()
}

fn ends_with_path(url: &str, path: &str) -> bool {
    url.trim_end_matches('/')
        .rsplit_once('/')
        .map(|(_, last)| last.eq_ignore_ascii_case(path.rsplit('/').next().unwrap_or(path)))
        .unwrap_or(false)
        && url
            .trim_end_matches('/')
            .to_ascii_lowercase()
            .ends_with(&format!("/{}", path.to_ascii_lowercase()))
}

fn is_chat_payload_compatibility_error(status: StatusCode, body: &str) -> bool {
    if status.as_u16() != 400 && status.as_u16() != 422 {
        return false;
    }
    let lower = body.to_ascii_lowercase();
    [
        "image_url",
        "content",
        "message",
        "schema",
        "invalid type",
        "unsupported",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn remove_if_exists(path: &Path) -> Result<(), ApiError> {
    match fs::remove_file(path) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(ApiError::internal(format!("remove file failed: {e}"))),
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut out = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(protocol: LlmProtocol) -> ProviderConfig {
        ProviderConfig {
            protocol,
            base_url: "https://api.example.com/v1".to_string(),
            api_key_ref: "local_secret".to_string(),
            model: "vision-model".to_string(),
            display_name: "test".to_string(),
            max_concurrency: 4,
            timeout_seconds: 30,
        }
    }

    #[test]
    fn chat_completions_payload_uses_image_url_parts() {
        let payload = build_judge_payload(
            &config(LlmProtocol::OpenaiChatCompletions),
            "data:image/jpeg;base64,abc",
            "prompt",
        );
        assert_eq!(payload["messages"][0]["role"], "system");
        assert_eq!(payload["messages"][1]["content"][0]["type"], "image_url");
        assert_eq!(payload["messages"][1]["content"][1]["type"], "text");
    }

    #[test]
    fn chat_completions_payloads_include_markdown_fallback() {
        let payloads = build_judge_payloads(
            &config(LlmProtocol::OpenaiChatCompletions),
            "data:image/jpeg;base64,abc",
            "prompt",
        );
        assert_eq!(payloads.len(), 2);
        assert!(payloads[1]["messages"][1]["content"]
            .as_str()
            .expect("fallback content")
            .contains("![image](data:image/jpeg;base64,abc)"));
    }

    #[test]
    fn completion_endpoint_accepts_root_or_full_endpoint_url() {
        let mut root = config(LlmProtocol::OpenaiChatCompletions);
        root.base_url = "https://api.example.com/v1".to_string();
        assert_eq!(
            root.completion_endpoint().expect("root endpoint"),
            "https://api.example.com/v1/chat/completions"
        );

        let mut full = config(LlmProtocol::OpenaiChatCompletions);
        full.base_url = "https://api.example.com/v1/chat/completions/".to_string();
        assert_eq!(
            full.completion_endpoint().expect("full endpoint"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn responses_payload_uses_input_image_parts() {
        let payload = build_judge_payload(
            &config(LlmProtocol::OpenaiResponses),
            "data:image/jpeg;base64,abc",
            "prompt",
        );
        assert_eq!(payload["input"][0]["content"][0]["type"], "input_image");
        assert_eq!(payload["input"][0]["content"][1]["type"], "input_text");
    }

    #[test]
    fn anthropic_payload_uses_base64_image_block() {
        let payload = build_judge_payload(
            &config(LlmProtocol::AnthropicMessages),
            "data:image/jpeg;base64,abc",
            "prompt",
        );
        assert_eq!(payload["messages"][0]["content"][0]["type"], "image");
        assert_eq!(
            payload["messages"][0]["content"][0]["source"]["data"],
            "abc"
        );
    }

    #[test]
    fn provider_config_does_not_serialize_api_key() {
        let temp = tempfile::tempdir().expect("temp dir");
        let manager = LlmProviderManager::new(temp.path().join("llm"));
        manager
            .save_provider(SaveProviderRequest {
                protocol: Some(LlmProtocol::OpenaiChatCompletions),
                base_url: Some("https://api.example.com/v1".to_string()),
                api_key: Some("secret-key".to_string()),
                model: Some("gpt-vision".to_string()),
                display_name: None,
                max_concurrency: None,
                timeout_seconds: None,
            })
            .expect("save provider");
        let config_text =
            fs::read_to_string(temp.path().join("llm").join(CONFIG_FILENAME)).expect("config file");
        assert!(!config_text.contains("secret-key"));
        let secret_text =
            fs::read_to_string(temp.path().join("llm").join(SECRET_FILENAME)).expect("secret file");
        assert_eq!(secret_text, "secret-key");
    }

    #[test]
    fn parses_chat_json_verdict() {
        let value = json!({"choices":[{"message":{"content":"{\"verdict\":\"pass\",\"reason\":\"连接测试\"}"}}]});
        let parsed = parse_judge_response(&LlmProtocol::OpenaiChatCompletions, &value)
            .expect("parsed response");
        assert_eq!(parsed["verdict"], "pass");
    }

    #[test]
    fn parses_chat_with_markdown_fence_case_insensitive() {
        let value = json!({"choices":[{"message":{"content":"```JSON\n{\"verdict\":\"reject\",\"reason\":\"糊了\"}\n```"}}]});
        let parsed = parse_judge_response(&LlmProtocol::OpenaiChatCompletions, &value)
            .expect("parsed response with JSON fence");
        assert_eq!(parsed["verdict"], "reject");
        assert_eq!(parsed["reason"], "糊了");
    }

    #[test]
    fn parses_chat_with_prose_around_json() {
        let value = json!({"choices":[{"message":{"content":"下面是结果：{\"verdict\":\"pass\",\"reason\":\"清晰\"} 仅供参考"}}]});
        let parsed = parse_judge_response(&LlmProtocol::OpenaiChatCompletions, &value)
            .expect("parsed response with prose around json");
        assert_eq!(parsed["verdict"], "pass");
    }

    #[test]
    fn parses_chat_with_array_content() {
        // 某些第三方代理把 content 拼成 [{type:"text", text:"..."}]
        let value = json!({"choices":[{"message":{"content":[{"type":"text","text":"{\"verdict\":\"pass\",\"reason\":\"ok\"}"}]}}]});
        let parsed = parse_judge_response(&LlmProtocol::OpenaiChatCompletions, &value)
            .expect("parsed array content");
        assert_eq!(parsed["verdict"], "pass");
    }

    #[test]
    fn rejects_invalid_verdict_value() {
        let value =
            json!({"choices":[{"message":{"content":"{\"verdict\":\"maybe\",\"reason\":\"x\"}"}}]});
        assert!(parse_judge_response(&LlmProtocol::OpenaiChatCompletions, &value).is_err());
    }

    #[test]
    fn strip_code_fence_handles_variants() {
        assert_eq!(strip_code_fence("```json\n{}\n```"), "{}");
        assert_eq!(strip_code_fence("```JSON\n{}\n```"), "{}");
        assert_eq!(strip_code_fence("```\n{}\n```"), "{}");
        assert_eq!(strip_code_fence("{}"), "{}");
        assert_eq!(strip_code_fence("  ```\n{\"a\":1}\n```  "), "{\"a\":1}");
    }

    #[test]
    fn extract_first_json_object_basic() {
        assert_eq!(
            extract_first_json_object("noise {\"a\":1,\"b\":2} more"),
            Some("{\"a\":1,\"b\":2}".to_string())
        );
        assert_eq!(extract_first_json_object("no json here"), None);
        assert_eq!(
            extract_first_json_object("{\"k\":\"v\\\"q}\"}"),
            Some("{\"k\":\"v\\\"q}\"}".to_string())
        );
    }
}
