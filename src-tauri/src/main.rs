#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::{
    env,
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};
use tauri::{Manager, State};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
struct BackendPayload {
    running: bool,
    healthy: bool,
    url: Option<String>,
    token: Option<String>,
    port: Option<u16>,
    python: Option<String>,
    backend_kind: String,
    backend_dir: Option<String>,
    message: String,
}

#[derive(Debug)]
struct BackendRuntime {
    rust_backend: Option<pianke_backend::ServerHandle>,
    payload: BackendPayload,
}

impl Default for BackendRuntime {
    fn default() -> Self {
        Self {
            rust_backend: None,
            payload: BackendPayload {
                running: false,
                healthy: false,
                url: None,
                token: None,
                port: None,
                python: None,
                backend_kind: "rust-fast".to_string(),
                backend_dir: None,
                message: "片刻 Rust 后端尚未启动".to_string(),
            },
        }
    }
}

#[derive(Default)]
struct BackendState(Mutex<BackendRuntime>);

#[tauri::command]
fn backend_status(state: State<'_, BackendState>) -> Result<BackendPayload, String> {
    let mut runtime = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(port) = runtime.payload.port {
        runtime.payload.healthy = can_connect(port);
        runtime.payload.running = runtime.rust_backend.is_some();
        if runtime.payload.healthy {
            runtime.payload.message = "Rust Fast 后端已就绪".to_string();
        }
    }
    Ok(runtime.payload.clone())
}

#[tauri::command]
fn restart_backend(
    app: tauri::AppHandle,
    state: State<'_, BackendState>,
) -> Result<BackendPayload, String> {
    start_or_restart_backend(&app, &state)
}

#[tauri::command]
fn pick_folder() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("选择照片文件夹")
        .pick_folder()
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn open_external_url(url: String) -> Result<(), String> {
    let url = url.trim();
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("只能打开 http/https 下载链接".to_string());
    }
    open_url_with_system(url)
}

fn main() {
    tauri::Builder::default()
        .manage(BackendState::default())
        .setup(|app| {
            let handle = app.handle().clone();
            let state = app.state::<BackendState>();
            if let Err(err) = start_or_restart_backend(&handle, &state) {
                if let Ok(mut runtime) = state.0.lock() {
                    runtime.payload = BackendPayload {
                        running: false,
                        healthy: false,
                        url: None,
                        token: None,
                        port: None,
                        python: None,
                        backend_kind: "rust-fast".to_string(),
                        backend_dir: None,
                        message: err,
                    };
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            backend_status,
            restart_backend,
            pick_folder,
            open_external_url
        ])
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                let state = window.state::<BackendState>();
                if let Ok(mut runtime) = state.0.lock() {
                    stop_rust_backend(&mut runtime.rust_backend);
                };
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run pianke desktop");
}

#[cfg(windows)]
fn open_url_with_system(url: &str) -> Result<(), String> {
    Command::new("rundll32")
        .arg("url.dll,FileProtocolHandler")
        .arg(url)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("打开下载链接失败：{e}"))
}

#[cfg(target_os = "macos")]
fn open_url_with_system(url: &str) -> Result<(), String> {
    Command::new("open")
        .arg(url)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("打开下载链接失败：{e}"))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_url_with_system(url: &str) -> Result<(), String> {
    Command::new("xdg-open")
        .arg(url)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("打开下载链接失败：{e}"))
}

fn start_or_restart_backend(
    app: &tauri::AppHandle,
    state: &State<'_, BackendState>,
) -> Result<BackendPayload, String> {
    let mut runtime = state.0.lock().map_err(|e| e.to_string())?;
    stop_rust_backend(&mut runtime.rust_backend);

    let backend_dir = resolve_backend_dir(app)?;
    let port = reserve_port()?;
    let token = Uuid::new_v4().to_string();
    let url = format!("http://127.0.0.1:{port}");
    let requested_python = matches!(env::var("PIANKE_BACKEND").ok().as_deref(), Some("python"));

    let handle = pianke_backend::start(pianke_backend::ServerOptions {
        port,
        token: Some(token.clone()),
        backend_dir: backend_dir.clone(),
    })?;
    let healthy = wait_for_port(port, Duration::from_secs(10));
    runtime.rust_backend = Some(handle);
    runtime.payload = BackendPayload {
        running: true,
        healthy,
        url: Some(url),
        token: Some(token),
        port: Some(port),
        python: None,
        backend_kind: "rust-fast".to_string(),
        backend_dir: Some(backend_dir.to_string_lossy().to_string()),
        message: if requested_python {
            "当前 Rust-only 开发包不包含 Python 后端，已启动 Rust Fast 后端。".to_string()
        } else if healthy {
            "Rust Fast 后端已就绪".to_string()
        } else {
            "Rust Fast 后端已启动，但健康检查暂未通过".to_string()
        },
    };
    Ok(runtime.payload.clone())
}

fn stop_rust_backend(handle: &mut Option<pianke_backend::ServerHandle>) {
    if let Some(handle) = handle.take() {
        handle.stop();
    }
}

fn resolve_backend_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if let Ok(dir) = env::var("PIANKE_BACKEND_DIR") {
        let p = PathBuf::from(dir);
        if p.join("static").exists() {
            return Ok(p);
        }
    }

    let dev_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("无法解析开发目录")?
        .to_path_buf();
    if dev_dir.join("static").exists() {
        return Ok(dev_dir);
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        if resource_dir.join("static").exists() {
            return Ok(resource_dir);
        }
        let nested = resource_dir.join("backend");
        if nested.join("static").exists() {
            return Ok(nested);
        }
    }

    Err(
        "未找到 static 资源目录。开发时请在项目根目录运行；打包时请确认 static 资源已随包分发。"
            .to_string(),
    )
}

fn reserve_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    drop(listener);
    Ok(port)
}

fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let started = Instant::now();
    while started.elapsed() < timeout {
        if can_connect(port) {
            return true;
        }
        thread::sleep(Duration::from_millis(120));
    }
    false
}

fn can_connect(port: u16) -> bool {
    let addr = format!("127.0.0.1:{port}");
    addr.parse()
        .ok()
        .and_then(|addr| TcpStream::connect_timeout(&addr, Duration::from_millis(250)).ok())
        .is_some()
}
