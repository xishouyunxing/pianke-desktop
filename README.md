# 片刻桌面版

片刻是一个面向摄影选片的 Rust/Tauri 桌面应用。当前 GitHub 开发包只保留 Rust 后端、Vue 前端和 Tauri 桌面壳，不包含 Python runtime、Flask worker、Python 参考实现或旧启动脚本。

## 当前能力

- Fast 模式：安装即用，本地完成扫描、hash/HSV/quality/ORB 分析、预筛、分组、copy/move、撤销、重开和 RAW+JPG/XMP 伴随文件归档。
- 水印：Rust 后端接管 `templates / preview / start / status / cancel / open_out_dir`，基础包内可用。
- RAW/HEIC：RAW 读取 embedded JPEG preview；HEIC/HEIF 在 Windows 上优先使用系统 WIC codec，缺 codec 时进入 skipped 并给出中文原因。
- Expert 模式：通过片刻官方 ONNX 组件包安装本地模型；当前组件约定包含 DINOv2、InsightFace/ArcFace/68 点 landmark，可扩展 MUSIQ/CLIP-IQA+。NIMA legacy classifier 不复刻，能力中明确标记不可用。
- Tycoon 模式：复用 Expert 本地分组，并支持 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 兼容 provider。真实远程调用需要用户自行配置 API；仓库内默认只做 mock 验收。
- 软件更新提示：启动后检查 `https://pianke.moeuu.cn/pianke/desktop/latest.json`，发现新版本时在首页显示下载按钮。

## 开发环境

需要安装：

- Node.js 20+
- Rust stable
- Windows 构建 release 时需要 Tauri/NSIS 相关工具链；OpenCV runtime 由构建脚本收集到发布包

安装依赖：

```powershell
npm install
```

开发运行：

```powershell
npm run dev
```

构建 UI：

```powershell
npm run build:ui
```

构建 Rust-only 安装包：

```powershell
npm run build
```

RC 验收：

```powershell
npm run smoke:rc
```

检查发布包不含 Python：

```powershell
npm run check:no-python-bundle
```

## Expert 组件

基础安装包不内置大模型文件。Expert 首次启用时默认从片刻官方完整组件 manifest 安装：

```text
https://pianke.moeuu.cn/pianke/components/expert/onnx-v1/component.json
```

开发时可用环境变量覆盖：

- `PIANKE_EXPERT_MANIFEST_URL`
- `PIANKE_EXPERT_MANIFEST_PATH`
- `PIANKE_EXPERT_SOURCE_DIR`
- `PIANKE_EXPERT_COMPONENT_DIR`

组件文件和 ONNX 模型不进入 Git。服务器上传结构请参考 [RELEASE_RC_CHECKLIST.md](RELEASE_RC_CHECKLIST.md)。

## 软件更新

应用会读取：

```text
https://pianke.moeuu.cn/pianke/desktop/latest.json
```

示例：

```json
{
  "version": "0.1.1",
  "url": "https://pianke.moeuu.cn/pianke/desktop/片刻桌面版_0.1.1_x64-setup.exe",
  "notes": "更新说明",
  "published_at": "2026-05-27"
}
```

当前实现只提示并打开下载链接，不做静默自更新。

## 仓库边界

这个仓库面向 Rust-only 开发包：

- 不提交私有照片、生成 fixture、`.tmp_*`、ONNX 模型或安装包产物。
- 不提交 Python backend/reference 代码。
- 不提交 `src-tauri/opencv-runtime/` 或本机收集的 native runtime 缓存。

发布前请按 [RELEASE_RC_CHECKLIST.md](RELEASE_RC_CHECKLIST.md) 做安装包 smoke 和人工视觉检查。
