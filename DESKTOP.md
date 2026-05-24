# 片刻桌面版开发说明

桌面版第一阶段采用 Tauri v2 + Vue3 壳，保留现有 Flask/Python 后端。

## 开发运行

```powershell
npm install
npm run dev
```

Tauri 会启动 Vue 壳，并由 Rust 侧拉起 `app.py --no-browser`。默认优先使用：

1. `PIANKE_PYTHON` 指向的 Python；
2. 打包资源中的 `python/python.exe`；
3. Windows 的 `py -3.10`。

## 打包方向

当前仓库已经把 `app.py`、`pic_selecter/`、`static/`、`assets/` 和
`requirements-fast.txt` 声明为 Tauri 资源。真正发布给小白前，还需要把
固定版本的 Python 3.10 embeddable runtime 和 Fast 依赖放入
`src-tauri/binaries/python/` 或打包资源的 `python/` 目录。

Expert 模式依赖仍应作为后续可选组件下载，不建议放入第一版基础安装包。

## 准备内置 Python

发布给小白前先运行：

```powershell
npm run prepare:python
npm run build
```

`prepare:python` 会下载 Python 3.10 embeddable runtime，启用 pip，并把
`requirements-fast.txt` 中的 Fast 模式依赖安装到 `src-tauri/binaries/python/`。
该目录体积较大，不入库，但会被 Tauri 作为 `python/` 资源打进 NSIS 安装包。
