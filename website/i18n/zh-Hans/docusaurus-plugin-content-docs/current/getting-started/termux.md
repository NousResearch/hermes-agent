---
sidebar_position: 3
title: "Android / Termux"
description: "通过 Termux 在 Android 手机上直接运行 Hermes Agent"
---

# 在 Termux 上使用 Android 版 Hermes

这是在 [Termux](https://termux.dev/) 通过 Android 手机直接运行 Hermes Agent 的测试路径。

它为你提供手机上可工作的本地 CLI，以及目前已知可干净安装在 Android 上的核心额外功能。

## 测试路径支持什么

测试的 Termux 包安装：
- Hermes CLI
- cron 支持
- PTY/后台终端支持
- Telegram 网关支持（手动/尽力后台运行）
- MCP 支持
- Honcho 记忆支持
- ACP 支持

具体来说，它映射到：

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

## 测试路径目前不包含什么

一些功能仍然需要桌面/服务器风格的依赖项，这些依赖项尚未为 Android 发布，或者尚未在手机上验证：

- `.[all]` 目前不支持 Android
- `voice` 额外功能被 `faster-whisper -> ctranslate2` 阻塞，而 `ctranslate2` 不发布 Android wheel
- 自动浏览器 / Playwright 引导在 Termux 安装程序中被跳过
- Docker 风格的终端隔离在 Termux 内部不可用
- Android 可能仍然挂起 Termux 后台作业，因此网关持久化是尽力而为，而不是正常托管服务

这并不妨碍 Hermes 作为手机本地 CLI Agent 良好工作 — 只是推荐的手机安装有意比桌面/服务器安装更窄。

---

## 选项 1：一行安装程序

Hermes 现在提供支持 Termux 的安装路径：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

在 Termux 上，安装程序自动：
- 使用 `pkg` 安装系统包
- 使用 `python -m venv` 创建 venv
- 使用 `pip` 安装 `.[termux]`
- 将 `hermes` 符号链接到 `$PREFIX/bin`，以便它保持在你的 Termux PATH 上
- 跳过未经测试的浏览器 / WhatsApp 引导

如果你想要明确的命令或需要调试失败的安装，请使用下面的手动路径。

---

## 选项 2：手动安装（完全明确）

### 1. 更新 Termux 并安装系统包

```bash
pkg update
pkg install -y git python clang rust make pkg-config libffi openssl nodejs ripgrep ffmpeg
```

为什么是这些包？
- `python` — 运行时 + venv 支持
- `git` — 克隆/更新仓库
- `clang`、`rust`、`make`、`pkg-config`、`libffi`、`openssl` — 在 Android 上构建某些 Python 依赖项所需
- `nodejs` — 可选的 Node 运行时，用于测试路径之外的实验
- `ripgrep` — 快速文件搜索
- `ffmpeg` — 媒体 / TTS 转换

### 2. 克隆 Hermes

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

如果你已经克隆但没有子模块：

```bash
git submodule update --init --recursive
```

### 3. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
```

`ANDROID_API_LEVEL` 对于 `jiter` 等基于 Rust / maturin 的包很重要。

### 4. 安装测试的 Termux 包

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

如果你只想要最小的核心 Agent，这也可以：

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 5. 将 `hermes` 放到你的 Termux PATH

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

`$PREFIX/bin` 已经在 Termux 的 PATH 上，所以这使得 `hermes` 命令在新 shell 中持久化，而无需每次都重新激活 venv。

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

## 推荐的跟进设置

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

测试的 Termux 路径有意跳过 Node/浏览器引导。如果你以后想尝试浏览器工具：

```bash
pkg install nodejs-lts
npm install
```

浏览器工具自动在 PATH 搜索中包含 Termux 目录（`/data/data/com.termux/files/usr/bin`），因此 `agent-browser` 和 `npx` 无需任何额外 PATH 配置即可被发现。

将 Android 上的浏览器 / WhatsApp 工具视为实验性的，直到有文档记录。

---

## 故障排除

### 安装 `.[all]` 时"No solution found"

改用测试的 Termux 包：

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

目前的阻塞因素是 `voice` 额外功能：
- `voice` 拉取 `faster-whisper`
- `faster-whisper` 依赖 `ctranslate2`
- `ctranslate2` 不发布 Android wheel

### `uv pip install` 在 Android 上失败

改用 stdlib venv + `pip` 的 Termux 路径：

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` 抱怨 `ANDROID_API_LEVEL`

在安装前明确设置 API 级别：

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `hermes doctor` 说缺少 ripgrep 或 Node

用 Termux 包安装它们：

```bash
pkg install ripgrep nodejs
```

### 安装 Python 包时构建失败

确保构建工具链已安装：

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
- 通过 `faster-whisper` 的本地语音转录在测试路径中不可用
- 浏览器自动化设置被安装程序有意跳过
- 某些可选额外功能可能有效，但只有 `.[termux]` 目前被记录为测试过的 Android 包

如果你遇到新的 Android 特定问题，请使用以下信息打开 GitHub issue：
- 你的 Android 版本
- `termux-info`
- `python --version`
- `hermes doctor`
- 确切的安装命令和完整错误输出
