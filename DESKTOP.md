# 片刻桌面版开发说明

桌面版采用 Tauri v2 + Vue 3 + Rust backend。当前 GitHub 开发包是 Rust-only 结构，不再包含 Python 后端、Python embeddable runtime、Flask worker 或旧的一键 Python 启动器。

## 开发运行

```powershell
npm install
npm run dev
```

Tauri 启动后会在本机随机端口拉起 Rust backend，并把 `backend_kind` 标记为 `rust-fast`。如果环境变量里误设了 `PIANKE_BACKEND=python`，应用仍会启动 Rust backend，并提示当前开发包不包含 Python 后端。

## 构建发布包

```powershell
npm run build
```

构建脚本会：

- 构建前端静态资源；
- 构建 Tauri/Rust release；
- 收集 OpenCV ORB 所需 runtime 文件；
- 生成 Windows NSIS 安装包；
- 保持发布包不包含 Python runtime、`app.py`、Python package 或旧 Flask worker。

发布包检查：

```powershell
npm run smoke:release-rust
npm run check:no-python-bundle
```

RC 总验收：

```powershell
npm run smoke:rc
```

## 能力边界

- Fast 与水印属于基础安装包能力。
- RAW 第一阶段读取 embedded JPEG preview，不做完整 demosaic。
- HEIC/HEIF 在 Windows 上依赖系统 WIC/HEIF codec；缺 codec 时应 graceful skip。
- Expert 模型不进 Git、不进基础安装包，默认从片刻官方完整组件 manifest 安装。
- Tycoon 支持通用 provider 协议；真实远程 API 调用由用户配置，仓库验收默认使用 mock。
- NIMA legacy classifier 不复刻，避免把旧版随机初始化行为包装成可验证能力。

## 服务器发布文件

软件更新 manifest：

```text
/pianke/desktop/latest.json
/pianke/desktop/片刻桌面版_<version>_x64-setup.exe
```

Expert 完整组件：

```text
/pianke/components/expert/onnx-v1/component.json
/pianke/components/expert/onnx-v1/models/...
```

本机整理好的服务器上传目录通常位于 `.tmp_expert_upload/pianke/`，该目录不进入 Git。
