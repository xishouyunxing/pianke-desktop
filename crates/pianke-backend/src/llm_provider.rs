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
        let payload = build_probe_payload(&config)?;
        let url = config.completion_endpoint()?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| ApiError::internal(format!("create http client failed: {e}")))?;
        let resp = client
            .post(url)
            .headers(config.headers(&api_key)?)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ApiError::bad_gateway(format!("AI 服务商连接失败: {e}")))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| ApiError::bad_gateway(format!("读取 AI 服务商响应失败: {e}")))?;
        if !status.is_success() {
            return Err(ApiError::bad_gateway(format!(
                "AI 服务商返回 HTTP {status}: {}",
                truncate(&body, 180)
            )));
        }
        Ok(json!({
            "ok": true,
            "protocol": config.protocol,
            "model": config.model,
            "response_preview": truncate(&body, 240)
        }))
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
        let payload = build_judge_payload(config, image_data_url, prompt);
        let url = config.completion_endpoint()?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| ApiError::internal(format!("create http client failed: {e}")))?;

        let attempts = 4u32;
        let mut last_error: Option<String> = None;
        for attempt in 1..=attempts {
            let resp = client
                .post(&url)
                .headers(config.headers(&api_key)?)
                .json(&payload)
                .send()
                .await;
            let resp = match resp {
                Ok(resp) => resp,
                Err(e) => {
                    last_error = Some(e.to_string());
                    if attempt < attempts {
                        tokio::time::sleep(Duration::from_millis(300 * u64::from(attempt))).await;
                        continue;
                    }
                    return Err(ApiError::bad_gateway(format!(
                        "AI provider request failed: {e}"
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
                return Err(ApiError::bad_gateway(format!(
                    "AI provider returned HTTP {status}: {}",
                    truncate(&body, 180)
                )));
            }
            let value: Value = serde_json::from_str(&body)
                .map_err(|e| ApiError::bad_gateway(format!("parse AI response failed: {e}")))?;
            let parsed =
                parse_judge_response(&config.protocol, &value).map_err(ApiError::bad_gateway)?;
            return Ok(JudgeVerdict {
                verdict: parsed["verdict"].as_str().unwrap_or("reject").to_string(),
                reason: parsed["reason"].as_str().unwrap_or("").to_string(),
                flaws: parsed["flaws"].as_str().map(str::to_string),
                fixable: parsed["fixable"].as_str().map(str::to_string),
            });
        }
        Err(ApiError::bad_gateway(format!(
            "AI provider request failed after retries: {}",
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
        match self.protocol {
            LlmProtocol::OpenaiChatCompletions => self.endpoint("chat/completions"),
            LlmProtocol::OpenaiResponses => self.endpoint("responses"),
            LlmProtocol::AnthropicMessages => self.endpoint("messages"),
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
    match config.protocol {
        LlmProtocol::OpenaiChatCompletions => json!({
            "model": config.model,
            "temperature": 0.0,
            "max_tokens": 384,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt}
                ]
            }]
        }),
        LlmProtocol::OpenaiResponses => json!({
            "model": config.model,
            "temperature": 0.0,
            "max_output_tokens": 384,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_data_url},
                    {"type": "input_text", "text": prompt}
                ]
            }]
        }),
        LlmProtocol::AnthropicMessages => json!({
            "model": config.model,
            "max_tokens": 384,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    anthropic_image_block(image_data_url),
                    {"type": "text", "text": prompt}
                ]
            }]
        }),
    }
}

#[allow(dead_code)]
pub fn parse_judge_response(protocol: &LlmProtocol, value: &Value) -> Result<Value, String> {
    let text = match protocol {
        LlmProtocol::OpenaiChatCompletions => value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or(""),
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
            .unwrap_or(""),
        LlmProtocol::AnthropicMessages => value["content"]
            .as_array()
            .and_then(|parts| {
                parts.iter().find_map(|part| {
                    (part["type"] == "text")
                        .then(|| part["text"].as_str())
                        .flatten()
                })
            })
            .unwrap_or(""),
    };
    let cleaned = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let parsed: Value = serde_json::from_str(cleaned).map_err(|e| {
        format!(
            "AI 响应不是有效 JSON: {e}; content={}",
            truncate(cleaned, 160)
        )
    })?;
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

fn build_probe_payload(config: &ProviderConfig) -> Result<Value, ApiError> {
    Ok(build_judge_payload(
        config,
        TEST_IMAGE_DATA_URL,
        r#"只输出一行 JSON：{"verdict":"pass","reason":"连接测试","flaws":"无","fixable":"无"}"#,
    ))
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
        assert_eq!(payload["messages"][0]["content"][0]["type"], "image_url");
        assert_eq!(payload["messages"][0]["content"][1]["type"], "text");
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
}
