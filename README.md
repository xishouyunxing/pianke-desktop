# 片刻桌面版

> 本项目由原项目二次开发而来。原项目地址：[https://github.com/zhaoyue4810/pianke.git](https://github.com/zhaoyue4810/pianke.git)

## 给小白用户看的介绍

片刻是一款给摄影师、摄影爱好者和大量照片整理场景使用的桌面选片工具。你只需要选择一个照片文件夹，软件会自动扫描里面的照片，把相似连拍、相近构图或同一场景的照片分到一起，并先帮你过滤明显模糊、欠曝、过曝、闭眼或质量较差的照片。之后你可以在左右对比界面里快速保留更好的那张，最后把胜出照片复制或移动到结果目录。

软件的基础功能安装后即可使用：Fast 模式不需要本地模型，也不需要联网；水印功能可以给选出的照片批量添加相机参数风格水印；RAW+JPG/XMP 伴随文件会一起归档，避免只移动 JPG 却漏掉 RAW。需要更强的人像和语义分组时，可以安装官方 Expert 组件包；需要远程大模型给出“为什么退片”的文字判断时，再配置自己的 AI provider。

官网和下载入口：[https://pianke.moeuu.cn](https://pianke.moeuu.cn)

## Rust-only 开发包介绍

当前 GitHub 仓库是 Rust-only 开发包，只保留 Tauri 桌面壳、Vue 前端、Rust 后端和构建/验收脚本，不再包含旧 Python 后端、Python runtime、Flask worker、requirements 文件或旧的一键启动器。

开发环境需要：

- Node.js 20+
- Rust stable
- Windows release 构建需要 Tauri/NSIS 相关工具链
- OpenCV runtime 由构建脚本收集到发布包

常用命令：

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

Expert 组件默认从片刻官方完整组件 manifest 安装：

```text
https://pianke.moeuu.cn/pianke/components/expert/onnx-v1/component.json
```

开发时可以用这些环境变量覆盖组件来源或本机组件目录：

- `PIANKE_EXPERT_MANIFEST_URL`
- `PIANKE_EXPERT_MANIFEST_PATH`
- `PIANKE_EXPERT_SOURCE_DIR`
- `PIANKE_EXPERT_COMPONENT_DIR`

软件更新检查默认读取：

```text
https://pianke.moeuu.cn/pianke/desktop/latest.json
```

当前更新功能只提示有新版本并打开下载链接，不做静默自动更新。

仓库边界：

- 不提交私有照片、生成 fixture、`.tmp_*`、ONNX 模型或安装包产物。
- 不提交 `src-tauri/opencv-runtime/` 或本机收集的 native runtime 缓存。
- 不提交旧 Python backend/reference 代码。

## Rust 重构技术细节

这次重构的核心目标是把最终安装包收口为 Rust 默认、无 Python runtime、安装即用的桌面软件。Tauri 负责桌面壳和系统能力，Vue 负责前端交互，Rust backend 负责本地 HTTP API、任务状态、图片分析、归档、水印、模型组件和 provider 配置。

主要技术点：

- **Rust Fast 后端**：实现本地扫描、EXIF/方向处理、感知 hash、HSV 特征、质量评分、prescreen、分组、copy/move、undo/reopen、RAW+JPG/XMP companion 归档。
- **OpenCV ORB**：release 路径纳入 ORB keypoints/descriptors、BFMatcher 和 RANSAC homography inliers，用于增强相似图分组。
- **RAW/HEIC 支持**：RAW 第一阶段读取 embedded JPEG preview，不做完整 demosaic；HEIC/HEIF 在 Windows 下优先走系统 WIC codec，缺少系统 codec 时进入 skipped 并给出中文原因。
- **Rust 水印模块**：接管 `templates / preview / start / status / cancel / open_out_dir`，基础包内可用，前端无需切回 Python。
- **Expert 本地组件**：通过官方 ONNX 组件包安装 DINOv2、InsightFace/ArcFace、68 点 landmark 等模型；基础包不内置大模型，组件文件不进入 Git。
- **Tycoon provider 层**：支持 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 三类兼容协议；本地流程可用 mock 验收，真实远程调用由用户自行配置 API。
- **能力边界显式化**：NIMA legacy classifier 不复刻旧版随机初始化行为，能力中明确标记不可用；MUSIQ/CLIP-IQA+ 只有在 ONNX 与 Python golden parity 通过时才声明可用。
- **Rust-only 打包守卫**：构建和 smoke 脚本会检查发布包不包含 Python runtime、Flask worker、旧 Python package 或缺失的 native runtime。

发布前请按 [RELEASE_RC_CHECKLIST.md](RELEASE_RC_CHECKLIST.md) 做安装包 smoke、首页选择文件夹、Fast copy/move、水印视觉、Expert 组件下载速度和真实 provider 配置等人工确认。
