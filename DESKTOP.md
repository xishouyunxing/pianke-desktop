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
