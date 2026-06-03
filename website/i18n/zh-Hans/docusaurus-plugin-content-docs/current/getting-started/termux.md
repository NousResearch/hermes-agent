---
sidebar_position: 3
title: "Android / Termux"
description: "通过 Termux 在 Android 手机上直接运行 Hermes Agent"
---

# 在 Android 上通过 Termux 运行 Hermes

这是在 Android 手机上通过 [Termux](https://termux.dev/) 直接运行 Hermes Agent 的已验证路径。

它为你提供手机上可用的本地 CLI，以及目前已知可在 Android 上干净安装的核心扩展功能。

## 已验证路径支持哪些功能？

已验证的 Termux 安装包含：
- Hermes CLI
- cron 支持
- PTY（伪终端）/后台终端支持
- Telegram gateway 支持（手动 / 尽力而为的后台运行）
- MCP 支持
- Honcho 记忆支持
- ACP 支持

具体对应以下命令：

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

## 哪些功能尚未纳入已验证路径？

部分功能仍依赖桌面/服务器风格的依赖项，这些依赖项尚未为 Android 发布，或尚未在手机上验证：

- `.[all]` 目前不支持 Android
- `voice` 扩展被 `faster-whisper -> ctranslate2` 阻塞，`ctranslate2` 未发布 Android wheel 包
- 自动浏览器 / Playwright 引导在 Termux 安装程序中被跳过
- 基于 Docker 的终端隔离在 Termux 内不可用
- Android 可能仍会挂起 Termux 后台任务，因此 gateway 持久化是尽力而为，而非正常的托管服务

这并不妨碍 Hermes 作为手机原生 CLI agent 正常工作——只是意味着推荐的移动端安装有意比桌面/服务器安装更精简。

---

## 方式一：一行安装命令

Hermes 现已内置 Termux 感知的安装路径：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

在 Termux 上，安装程序会自动：
- 使用 `pkg` 安装系统包
- 使用 `python -m venv` 创建虚拟环境
- 优先尝试较大的 `.[termux-all]` 扩展，失败后回退到较小的 `.[termux]` 扩展（再次失败则进行基础安装）——curl 安装程序自动按此顺序执行
- 将 `hermes` 链接到 `$PREFIX/bin`，使其保留在 Termux PATH 中
- 跳过未经验证的浏览器 / WhatsApp 引导

如果你需要显式命令或需要调试失败的安装，请使用下方的手动安装路径。

---

## 方式二：手动安装（完全显式）

### 1. 更新 Termux 并安装系统包

```bash
pkg update
pkg install -y git python clang rust make pkg-config libffi openssl nodejs ripgrep ffmpeg
```

各包用途说明：
- `python` — 运行时 + 虚拟环境支持
- `git` — 克隆/更新仓库
- `clang`、`rust`、`make`、`pkg-config`、`libffi`、`openssl` — 在 Android 上构建部分 Python 依赖所需
- `nodejs` — 可选的 Node 运行时，用于已验证核心路径之外的实验
- `ripgrep` — 快速文件搜索
- `ffmpeg` — 媒体 / TTS 转换

### 2. 克隆 Hermes

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

### 3. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
```

`ANDROID_API_LEVEL` 对于基于 Rust / maturin 的包（如 `jiter`）非常重要。

### 4. 安装已验证的 Termux 包

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

如果你只需要最小化的核心 agent，以下命令同样有效：

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 5. 将 `hermes` 添加到 Termux PATH

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

`$PREFIX/bin` 在 Termux 中已默认在 PATH 中，因此这样做可以让 `hermes` 命令在新 shell 中持续可用，无需每次重新激活虚拟环境。

### 6. 验证安装

```bash
hermes version
hermes doctor
```

### 7. 启动 Hermes

```bash
hermes
```

---

## 推荐的后续配置

### 配置模型

```bash
hermes model
```

或直接在 `~/.hermes/.env` 中设置密钥。

### 稍后重新运行完整的交互式设置向导

```bash
hermes setup
```

### 手动安装可选的 Node 依赖

已验证的 Termux 路径有意跳过 Node/浏览器引导。如果你之后想尝试浏览器工具：

```bash
pkg install nodejs-lts
npm install
```

浏览器工具会自动将 Termux 目录（`/data/data/com.termux/files/usr/bin`）纳入 PATH 搜索，因此无需额外配置 PATH 即可发现 `agent-browser` 和 `npx`。

在另有文档说明之前，请将 Android 上的浏览器 / WhatsApp 工具视为实验性功能。

---

## 故障排查

### 安装 `.[all]` 时出现 `No solution found`

改用已验证的 Termux 包：

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

当前阻塞原因是 `voice` 扩展：
- `voice` 依赖 `faster-whisper`
- `faster-whisper` 依赖 `ctranslate2`
- `ctranslate2` 未发布 Android wheel 包

### `uv pip install` 在 Android 上失败

改用标准库 venv + `pip` 的 Termux 路径：

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` 报错提示缺少 `ANDROID_API_LEVEL`

在安装前显式设置 API 级别：

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `hermes doctor` 提示缺少 ripgrep 或 Node

使用 Termux 包安装：

```bash
pkg install ripgrep nodejs
```

### 安装 Python 包时构建失败

确保已安装构建工具链：

```bash
pkg install clang rust make pkg-config libffi openssl
```

然后重试：

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

---

## 手机上的已知限制

- Docker 后端不可用
- 通过 `faster-whisper` 进行的本地语音转录在已验证路径中不可用
- 安装程序有意跳过浏览器自动化配置
- 部分可选扩展可能可用，但目前仅 `.[termux]` 和 `.[termux-all]` 被记录为已验证的 Android 安装包

如果你遇到新的 Android 特定问题，请在 GitHub 上提交 issue，并附上：
- 你的 Android 版本
- `termux-info`
- `python --version`
- `hermes doctor`
- 确切的安装命令及完整错误输出