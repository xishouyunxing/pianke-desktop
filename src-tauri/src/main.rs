#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::{
    env,
    io::{BufRead, BufReader},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
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
    child: Option<Child>,
    rust_backend: Option<pianke_backend::ServerHandle>,
    payload: BackendPayload,
}

impl Default for BackendRuntime {
    fn default() -> Self {
        Self {
            child: None,
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
                message: "片刻引擎尚未启动".to_string(),
            },
        }
    }
}

#[derive(Default)]
struct BackendState(Mutex<BackendRuntime>);

struct PythonCommand {
    program: String,
    prefix_args: Vec<String>,
    label: String,
}

#[tauri::command]
fn backend_status(state: State<'_, BackendState>) -> Result<BackendPayload, String> {
    let mut runtime = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(child) = runtime.child.as_mut() {
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            runtime.child = None;
            runtime.payload.running = false;
            runtime.payload.healthy = false;
            runtime.payload.message = "片刻引擎已退出，请重启引擎".to_string();
        } else if let Some(port) = runtime.payload.port {
            runtime.payload.healthy = can_connect(port);
            runtime.payload.running = true;
            if runtime.payload.healthy {
                runtime.payload.message = "片刻引擎已就绪".to_string();
            }
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
            pick_folder
        ])
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                let state = window.state::<BackendState>();
                {
                    if let Ok(mut runtime) = state.0.lock() {
                        stop_child(&mut runtime.child);
                        stop_rust_backend(&mut runtime.rust_backend);
                    };
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run pianke desktop");
}

fn start_or_restart_backend(
    app: &tauri::AppHandle,
    state: &State<'_, BackendState>,
) -> Result<BackendPayload, String> {
    let mut runtime = state.0.lock().map_err(|e| e.to_string())?;
    stop_child(&mut runtime.child);
    stop_rust_backend(&mut runtime.rust_backend);

    let backend_dir = resolve_backend_dir(app)?;
    let port = reserve_port()?;
    let token = Uuid::new_v4().to_string();
    let url = format!("http://127.0.0.1:{port}");

    let requested_python = matches!(env::var("PIANKE_BACKEND").ok().as_deref(), Some("python"));
    let use_python = requested_python && python_fallback_allowed();

    if !use_python {
        let handle = pianke_backend::start(pianke_backend::ServerOptions {
            port,
            token: Some(token.clone()),
            backend_dir: backend_dir.clone(),
        })?;
        let healthy = wait_for_port(port, Duration::from_secs(10));
        runtime.payload = BackendPayload {
            running: true,
            healthy,
            url: Some(url),
            token: Some(token),
            port: Some(port),
            python: None,
            backend_kind: "rust-fast".to_string(),
            backend_dir: Some(backend_dir.to_string_lossy().to_string()),
            message: if healthy {
                "Rust Fast 引擎已就绪".to_string()
            } else {
                "Rust Fast 引擎已启动，但健康检查暂未通过".to_string()
            },
        };
        if requested_python {
            runtime.payload.message =
                "正式安装包不包含 Python 后端，已启动 Rust Fast 后端。".to_string();
        }
        runtime.rust_backend = Some(handle);
        return Ok(runtime.payload.clone());
    }

    let python = resolve_python(app);

    let mut command = Command::new(&python.program);
    command
        .args(&python.prefix_args)
        .arg("app.py")
        .arg("--port")
        .arg(port.to_string())
        .arg("--no-browser")
        .current_dir(&backend_dir)
        .env("PIANKE_DESKTOP", "1")
        .env("PIC_SELECTER_TOKEN", &token)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|e| {
        format!(
            "无法启动 Python 后端：{e}。请安装 Python 3.10，或设置 PIANKE_PYTHON 指向 python.exe。"
        )
    })?;

    drain_output(child.stdout.take(), "stdout");
    drain_output(child.stderr.take(), "stderr");

    let healthy = wait_for_port(port, Duration::from_secs(10));
    let message = if healthy {
        "片刻引擎已就绪".to_string()
    } else {
        "Python 后端已启动，但健康检查暂未通过；请稍等或查看终端日志".to_string()
    };

    runtime.payload = BackendPayload {
        running: true,
        healthy,
        url: Some(url),
        token: Some(token),
        port: Some(port),
        python: Some(python.label),
        backend_kind: "python".to_string(),
        backend_dir: Some(backend_dir.to_string_lossy().to_string()),
        message,
    };
    runtime.child = Some(child);
    Ok(runtime.payload.clone())
}

fn stop_rust_backend(handle: &mut Option<pianke_backend::ServerHandle>) {
    if let Some(handle) = handle.take() {
        handle.stop();
    }
}

fn python_fallback_allowed() -> bool {
    cfg!(debug_assertions)
}

fn stop_child(child: &mut Option<Child>) {
    if let Some(mut child) = child.take() {
        let _ = child.kill();
        let _ = child.wait();
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

fn resolve_python(app: &tauri::AppHandle) -> PythonCommand {
    if let Ok(py) = env::var("PIANKE_PYTHON") {
        return PythonCommand {
            label: py.clone(),
            program: py,
            prefix_args: vec![],
        };
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        let bundled = resource_dir.join("python").join("python.exe");
        if bundled.exists() {
            let py = bundled.to_string_lossy().to_string();
            return PythonCommand {
                label: py.clone(),
                program: py,
                prefix_args: vec![],
            };
        }
    }

    let dev_bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("binaries")
        .join("python")
        .join("python.exe");
    if dev_bundled.exists() {
        let py = dev_bundled.to_string_lossy().to_string();
        return PythonCommand {
            label: py.clone(),
            program: py,
            prefix_args: vec![],
        };
    }

    if cfg!(windows) {
        PythonCommand {
            program: "py".to_string(),
            prefix_args: vec!["-3.10".to_string()],
            label: "py -3.10".to_string(),
        }
    } else {
        PythonCommand {
            program: "python3".to_string(),
            prefix_args: vec![],
            label: "python3".to_string(),
        }
    }
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

fn drain_output<T: std::io::Read + Send + 'static>(stream: Option<T>, label: &'static str) {
    if let Some(stream) = stream {
        thread::spawn(move || {
            let reader = BufReader::new(stream);
            for line in reader.lines().map_while(Result::ok) {
                eprintln!("[python:{label}] {line}");
            }
        });
    }
}
