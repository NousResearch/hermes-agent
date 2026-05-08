# Hermes 安装引导

**语言 / Language:** [简体中文](#简体中文) | [English](#english)

---

# 简体中文

## 快速安装

使用一键安装脚本，在两分钟内即可启动并运行 Hermes Agent。

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Android / Termux

Hermes 现已提供 Termux 专用安装路径：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

安装程序会自动检测 Termux 环境并切换到经过验证的 Android 安装流程：

- 使用 Termux 的 `pkg` 安装系统依赖（`git`、`python`、`nodejs`、`ripgrep`、`ffmpeg` 及构建工具）
- 使用 `python -m venv` 创建虚拟环境
- 自动导出 `ANDROID_API_LEVEL` 以兼容 Android wheel 构建
- 通过 `pip` 安装特制的 `.[termux]` 扩展依赖
- 默认跳过未经测试的浏览器 / WhatsApp 引导程序

如需完整的步骤说明，请参阅 [Termux 指南](./termux.md)。

> **⚠️ Windows 注意**
> **不支持**原生 Windows。请安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)，然后在 WSL2 内运行 Hermes Agent。上述安装命令在 WSL2 中同样适用。

### 安装程序的工作内容

安装程序会自动处理所有事项：全部依赖项（Python、Node.js、ripgrep、ffmpeg）、仓库克隆、虚拟环境创建、全局 `hermes` 命令设置以及 LLM 提供商配置。安装完成后即可开始对话。

#### 安装目录布局

安装文件的存放位置取决于您是以普通用户还是 root 用户身份安装：

| 安装模式 | 代码位置 | `hermes` 可执行文件 | 数据目录 |
|---|---|---|---|
| 每用户（普通） | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes`（符号链接） | `~/.hermes/` |
| Root 模式 (`sudo curl … \| sudo bash`) | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/`（或 `$HERMES_HOME`） |

Root 模式的 **FHS 标准布局**（`/usr/local/lib/…`、`/usr/local/bin/hermes`）与 Linux 上其他系统级开发工具的安装位置一致，适用于共享机器部署——一个系统安装可服务所有用户。每个用户的配置（认证、技能、会话）仍然存放在各自的 `~/.hermes/` 或显式设置的 `HERMES_HOME` 路径下。

### 安装后

重新加载 shell 并启动对话：

```bash
source ~/.bashrc   # 或：source ~/.zshrc
hermes             # 开始对话！
```

如需重新配置各项设置，请使用专用命令：

```bash
hermes model          # 选择 LLM 提供商和模型
hermes tools          # 配置启用哪些工具
hermes gateway setup  # 设置消息平台
hermes config set     # 设置单个配置值
hermes setup          # 运行完整设置向导，一次性配置所有内容
```

---

## 前提条件

唯一的先决条件是 **Git**。安装程序会自动处理其他所有事项：

- **uv**（快速 Python 包管理器）
- **Python 3.11**（通过 uv 安装，无需 sudo 权限）
- **Node.js v22**（用于浏览器自动化和 WhatsApp 桥接）
- **ripgrep**（快速文件搜索）
- **ffmpeg**（用于 TTS 的音频格式转换）

> **ℹ️ 提示**
> 您**无需**手动安装 Python、Node.js、ripgrep 或 ffmpeg。安装程序会自动检测缺失的组件并为您安装。只需确保 `git` 可用（`git --version`）。

> **💡 Nix 用户**
> 如果您使用 Nix（NixOS、macOS 或 Linux），有专用的安装路径，包含 Nix flake、声明式 NixOS 模块和可选的容器模式。请参阅 **[Nix & NixOS 安装指南](./nix-setup.md)**。

---

## 手动 / 开发者安装

如果您想克隆仓库并从源代码安装——用于贡献代码、运行特定分支或完全控制虚拟环境——请参阅贡献指南中的[开发环境配置](../developer-guide/contributing.md#development-setup)章节。

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| `hermes: command not found` | 重新加载 shell（`source ~/.bashrc`）或检查 PATH 配置 |
| `API key not set` | 运行 `hermes model` 配置提供商，或执行 `hermes config set OPENROUTER_API_KEY your_key` |
| 更新后配置丢失 | 运行 `hermes config check` 然后执行 `hermes config migrate` |

如需更详细的诊断信息，请运行 `hermes doctor`——它会准确指出缺失的内容及修复方法。

---

# English

## Quick Install

Get Hermes Agent up and running in under two minutes with the one-line installer.

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Android / Termux

Hermes now ships a Termux-aware installer path too:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

The installer detects Termux automatically and switches to a tested Android flow:

- Uses Termux `pkg` for system dependencies (`git`, `python`, `nodejs`, `ripgrep`, `ffmpeg`, build tools)
- Creates the virtualenv with `python -m venv`
- Exports `ANDROID_API_LEVEL` automatically for Android wheel builds
- Installs a curated `.[termux]` extra with `pip`
- Skips the untested browser / WhatsApp bootstrap by default

If you want the fully explicit path, follow the dedicated [Termux guide](./termux.md).

> **⚠️ Windows Notice**
> Native Windows is **not supported**. Please install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and run Hermes Agent from there. The install command above works inside WSL2.

### What the Installer Does

The installer handles everything automatically — all dependencies (Python, Node.js, ripgrep, ffmpeg), the repo clone, virtual environment, global `hermes` command setup, and LLM provider configuration. By the end, you're ready to chat.

#### Install Layout

Where the installer puts things depends on whether you're installing as a normal user or as root:

| Installer | Code lives at | `hermes` binary | Data directory |
|---|---|---|---|
| Per-user (normal) | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes` (symlink) | `~/.hermes/` |
| Root-mode (`sudo curl … \| sudo bash`) | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/` (or `$HERMES_HOME`) |

The root-mode **FHS layout** (`/usr/local/lib/…`, `/usr/local/bin/hermes`) matches where other system-wide developer tools land on Linux. It's useful for shared-machine deployments where one system install should serve every user. Per-user config (auth, skills, sessions) still lives under each user's `~/.hermes/` or explicit `HERMES_HOME`.

### After Installation

Reload your shell and start chatting:

```bash
source ~/.bashrc   # or: source ~/.zshrc
hermes             # Start chatting!
```

To reconfigure individual settings later, use the dedicated commands:

```bash
hermes model          # Choose your LLM provider and model
hermes tools          # Configure which tools are enabled
hermes gateway setup  # Set up messaging platforms
hermes config set     # Set individual config values
hermes setup          # Or run the full setup wizard to configure everything at once
```

---

## Prerequisites

The only prerequisite is **Git**. The installer automatically handles everything else:

- **uv** (fast Python package manager)
- **Python 3.11** (via uv, no sudo needed)
- **Node.js v22** (for browser automation and WhatsApp bridge)
- **ripgrep** (fast file search)
- **ffmpeg** (audio format conversion for TTS)

> **ℹ️ Info**
> You do **not** need to install Python, Node.js, ripgrep, or ffmpeg manually. The installer detects what's missing and installs it for you. Just make sure `git` is available (`git --version`).

> **💡 Nix Users**
> If you use Nix (on NixOS, macOS, or Linux), there's a dedicated setup path with a Nix flake, declarative NixOS module, and optional container mode. See the **[Nix & NixOS Setup](./nix-setup.md)** guide.

---

## Manual / Developer Installation

If you want to clone the repo and install from source — for contributing, running from a specific branch, or having full control over the virtual environment — see the [Development Setup](../developer-guide/contributing.md#development-setup) section in the Contributing guide.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `hermes: command not found` | Reload your shell (`source ~/.bashrc`) or check PATH |
| `API key not set` | Run `hermes model` to configure your provider, or `hermes config set OPENROUTER_API_KEY your_key` |
| Missing config after update | Run `hermes config check` then `hermes config migrate` |

For more diagnostics, run `hermes doctor` — it will tell you exactly what's missing and how to fix it.
