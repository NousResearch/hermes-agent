---
sidebar_position: 3
title: "Android / Termux"
description: "通过 Termux 在 Android 手机上直接运行 Hermes Agent"
---

# 在 Android 上使用 Termux 运行 Hermes

这是通过 [Termux](https://termux.dev/) 在 Android 手机上直接运行 Hermes Agent 的经过测试的路径。

它为你提供手机上可用的本地 CLI，以及当前已知可在 Android 上干净安装的核心额外功能。

## 测试路径中支持什么？

经过测试的 Termux 包安装：
- Hermes CLI
- cron 支持
- PTY/后台终端支持
- Telegram 网关支持（手动/尽力后台运行）
- MCP 支持
- Honcho 记忆支持
- ACP 支持

具体对应：

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

## 测试路径中尚不支持什么？

一些功能仍需要未针对 Android 发布的桌面/服务器风格依赖，或尚未在手机上验证：

- `.[all]` 目前在 Android 上不支持
- `voice` 额外功能被 `faster-whisper -> ctranslate2` 阻塞，`ctranslate2` 不发布 Android wheel
- Termux 安装程序跳过自动浏览器/Playwright 引导
- Termux 内不可用基于 Docker 的终端隔离
- Android 可能仍然暂停 Termux 后台作业，因此网关持久性是尽力而为，而非正常的托管服务

这并不妨碍 Hermes 作为手机原生 CLI 智能体良好工作——只是意味着推荐的移动安装有意比桌面/服务器安装更精简。

---

## 方案一：一键安装

Hermes 现在提供 Termux 感知安装路径：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

在 Termux 上，安装程序自动：
- 使用 `pkg` 安装系统包
- 使用 `python -m venv` 创建虚拟环境
- 先尝试广泛的 `.[termux-all]` 额外功能，回退到较小的 `.[termux]` 额外功能（然后是基础安装）——curl 安装程序自动匹配此顺序
- 将 `hermes` 链接到 `$PREFIX/bin`，使其保持在 Termux PATH 上
- 跳过未经测试的浏览器/WhatsApp 引导

如果你想要显式命令或需要调试失败的安装，请使用下面的手动路径。

---

## 方案二：手动安装（完全显式）

### 1. 更新 Termux 并安装系统包

```bash
pkg update
pkg install -y git python clang rust make pkg-config libffi openssl nodejs ripgrep ffmpeg
```

为什么需要这些包？
- `python` —— 运行时 + venv 支持
- `git` —— 克隆/更新仓库
- `clang`、`rust`、`make`、`pkg-config`、`libffi`、`openssl` —— 在 Android 上构建一些 Python 依赖所需
- `nodejs` —— 可选 Node 运行时，用于测试核心路径之外的实验
- `ripgrep` —— 快速文件搜索
- `ffmpeg` —— 媒体/TTS 转换

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

`ANDROID_API_LEVEL` 对基于 Rust/maturin 的包（如 `jiter`）很重要。

### 4. 安装经过测试的 Termux 包

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

如果你只想要最小核心智能体，这也有效：

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 5. 将 `hermes` 放到 Termux PATH 上

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

`$PREFIX/bin` 已经在 Termux 的 PATH 上，因此这使 `hermes` 命令在新 shell 中持久可用，无需每次重新激活虚拟环境。

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

## 推荐的后续设置

### 配置模型

```bash
hermes model
```

或直接在 `~/.hermes/.env` 中设置密钥。

### 稍后重新运行完整交互设置向导

```bash
hermes setup
```

### 手动安装可选 Node 依赖

经过测试的 Termux 路径有意跳过 Node/浏览器引导。如果你想稍后实验浏览器工具：

```bash
pkg install nodejs-lts
npm install
```

浏览器工具自动在 PATH 搜索中包含 Termux 目录（`/data/data/com.termux/files/usr/bin`），因此无需额外 PATH 配置即可发现 `agent-browser` 和 `npx`。

在另行记录之前，将 Android 上的浏览器/WhatsApp 工具视为实验性。

---

## 故障排除

### 安装 `.[all]` 时出现 `No solution found`

改用经过测试的 Termux 包：

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

当前阻塞是 `voice` 额外功能：
- `voice` 拉取 `faster-whisper`
- `faster-whisper` 依赖 `ctranslate2`
- `ctranslate2` 不发布 Android wheel

### `uv pip install` 在 Android 上失败

改用 Termux 路径，使用 stdlib venv + `pip`：

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` 抱怨 `ANDROID_API_LEVEL`

在安装前显式设置 API 级别：

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `hermes doctor` 说缺少 ripgrep 或 Node

使用 Termux 包安装它们：

```bash
pkg install ripgrep nodejs
```

### 安装 Python 包时出现构建失败

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
- 测试路径中不可用通过 `faster-whisper` 的本地语音转录
- 安装程序有意跳过浏览器自动化设置
- 一些可选额外功能可能工作，但目前仅 `.[termux]` 和 `.[termux-all]` 被记录为经过测试的 Android 包

如果你遇到新的 Android 特定问题，请提交 GitHub issue 并包含：
- 你的 Android 版本
- `termux-info`
- `python --version`
- `hermes doctor`
- 确切的安装命令和完整错误输出
