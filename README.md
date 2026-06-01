<p align="center">
  <img src="src-tauri/icons/icon.png" alt="片刻 Logo" width="120" />
</p>

<h1 align="center">片刻桌面版</h1>

<p align="center">
  摄影选片、相似分组、批量水印和本地 AI 辅助判断工具。
  <br />
  安装即用 Fast 模式，按需启用 Expert 本地模型与 Tycoon 远程 AI。
</p>

<p align="center">
  <a href="https://pianke.moeuu.cn">官网</a>
  ·
  <a href="#简介">用户介绍</a>
  ·
  <a href="#开发者">开发包</a>
  ·
  <a href="#重构">技术细节</a>
</p>

<p align="center">
  <img alt="Rust-only" src="https://img.shields.io/badge/backend-Rust--only-f97316" />
  <img alt="Tauri" src="https://img.shields.io/badge/desktop-Tauri%202-24c8db" />
  <img alt="Vue" src="https://img.shields.io/badge/frontend-Vue%203-42b883" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-lightgrey" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green" />
</p>

> 本项目由原项目二次开发而来。原项目地址：[https://github.com/zhaoyue4810/pianke.git](https://github.com/zhaoyue4810/pianke.git)

---

## 简介

片刻是一款给摄影师、摄影爱好者和大量照片整理场景使用的桌面选片工具。你只需要选择一个照片文件夹，软件会自动扫描里面的照片，把相似连拍、相近构图或同一场景的照片分到一起，并先帮你过滤明显模糊、欠曝、过曝、闭眼或质量较差的照片。

之后你可以在左右对比界面里快速保留更好的那张，最后把胜出照片复制或移动到结果目录。RAW+JPG/XMP 伴随文件会一起归档，避免只移动 JPG 却漏掉 RAW。

官网和下载入口：[https://pianke.moeuu.cn](https://pianke.moeuu.cn)

### 工作模式

| 模式 | 是否联网 | 是否需要模型 | 适合场景 |
| --- | --- | --- | --- |
| Fast | 不需要 | 不需要 | 快速整理、连拍筛选、普通照片归档 |
| Expert | 不需要 | 需要官方 ONNX 组件 | 人像、语义相似、需要更强本地判断 |
| Tycoon | 需要 | 需要 Expert 组件和 API Key | 需要远程大模型给出文字判断理由 |

### 基础功能

- **相似分组**：自动把连拍、相近构图和同一场景照片放到同组。
- **质量预筛**：识别明显模糊、曝光异常、闭眼等问题照片。
- **左右对比**：用更直观的 A/B 选择方式快速保留胜出图。
- **批量水印**：给胜出照片添加相机参数风格水印。
- **安全归档**：支持 copy/move、撤销、重开，RAW+JPG/XMP companion 一起处理。
- **更新提示**：有新版本时首页显示提示按钮，点击后打开官网下载。

---

## 开发者

当前 GitHub 仓库是 Rust-only 开发包，只保留 Tauri 桌面壳、Vue 前端、Rust 后端和构建/验收脚本，不再包含旧 Python 后端、Python runtime、Flask worker、requirements 文件或旧的一键启动器。

### 环境要求

- Node.js 20+
- Rust stable
- Windows release 构建需要 Tauri/NSIS 相关工具链
- OpenCV runtime 由构建脚本收集到发布包

### 常用命令

```powershell
npm install
npm run dev
npm run build:ui
npm run build
npm run smoke:rc
```

发布包检查：

```powershell
npm run smoke:release-rust
npm run check:no-python-bundle
```

### Expert 组件来源

Expert 组件默认从由我本地验证过的 manifest 安装：

```text
https://pianke.moeuu.cn/pianke/components/expert/onnx-v1/component.json
```

开发时可以用这些环境变量覆盖组件来源或本机组件目录：

```text
PIANKE_EXPERT_MANIFEST_URL
PIANKE_EXPERT_MANIFEST_PATH
PIANKE_EXPERT_SOURCE_DIR
PIANKE_EXPERT_COMPONENT_DIR
```

### 软件更新

软件更新检查默认读取：

```text
https://pianke.moeuu.cn/pianke/desktop/latest.json
```

当前更新功能只提示有新版本并打开下载链接，不做静默自动更新。

---

## 重构

这次重构的核心目标是把最终安装包收口为 Rust 默认、无 Python runtime、安装即用的桌面软件。Tauri 负责桌面壳和系统能力，Vue 负责前端交互，Rust backend 负责本地 HTTP API、任务状态、图片分析、归档、水印、模型组件和 provider 配置。

### 主要技术点

| 模块 | 技术点 |
| --- | --- |
| Rust Fast 后端 | 扫描、EXIF/方向处理、感知 hash、HSV 特征、质量评分、prescreen、分组、copy/move、undo/reopen |
| OpenCV ORB | ORB keypoints/descriptors、BFMatcher、RANSAC homography inliers，用于增强相似图分组 |
| RAW/HEIC | RAW 读取 embedded JPEG preview；HEIC/HEIF 在 Windows 下优先走系统 WIC codec |
| 水印模块 | Rust 接管 `templates / preview / start / status / cancel / open_out_dir` |
| Expert 本地组件 | 官方 ONNX 组件包安装 DINOv2、InsightFace/ArcFace、68 点 landmark、MUSIQ、CLIP-IQA+、NIMA 等模型 |
| Tycoon provider | 支持 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 三类兼容协议 |
| 打包守卫 | 构建和 smoke 脚本检查发布包不包含 Python runtime、Flask worker、旧 Python package 或缺失 native runtime |

### 能力边界

- NIMA legacy classifier 不复刻旧版随机初始化行为；当前只在安装真实 `nima_vgg16_ava.onnx` 权重后启用 NIMA 分数。
- MUSIQ/CLIP-IQA+ 只有在 ONNX 与 Python golden parity 通过时才声明可用。
- Expert 组件和 ONNX 模型不进入 Git，也不内置在基础安装包中。
- Tycoon 的真实远程调用由用户自行配置 API，仓库默认使用 mock 验收本地流程。

---
