---
sidebar_position: 2
title: "安装"
description: "在 Linux、macOS、WSL2、原生 Windows（早期 beta）或通过 Termux 的 Android 上安装 Hermes Agent"
---

# 安装

使用一行命令安装程序，在不到两分钟内启动并运行 Hermes Agent。

## 快速安装

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Windows（原生，PowerShell）—— 早期 Beta {#windows-native-powershell--early-beta}

:::warning 早期 BETA
原生 Windows 支持处于 **早期 beta** 阶段。它可以在常见路径下安装和运行，但尚未像我们的 POSIX 安装程序那样经过广泛的道路测试。遇到问题时请 [提交 issue](https://github.com/NousResearch/hermes-agent/issues)。对于目前在 Windows 上最成熟的设置，请改为在上述 **WSL2** 中使用 Linux/macOS 一行命令安装程序。
:::

打开 PowerShell 并运行：

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

安装程序会处理 **所有内容**：`uv`、Python 3.11、Node.js 22、`ripgrep`、`ffmpeg`，**以及一个便携版 Git Bash**（PortableGit —— 一个自包含的 Git-for-Windows 发行版，提供 `bash.exe` 和 Hermes 用于 shell 命令的完整 POSIX 工具链；在 32 位 Windows 上，安装程序会回退到 MinGit，后者缺少 bash 并禁用 terminal-tool / agent-browser 功能）。它会将仓库克隆到 `%LOCALAPPDATA%\hermes\hermes-agent` 下，创建 virtualenv，并将 `hermes` 添加到您的 **用户 PATH**。安装完成后请重启终端（或打开新的 PowerShell 窗口），以便 PATH 生效。

**Git 的处理方式：**
1. 如果 `git` 已在您的 PATH 上，安装程序会使用您现有的安装。
2. 否则，它会下载便携版 **PortableGit**（约 50MB，来自官方 `git-for-windows` GitHub 发布版）并解压到 `%LOCALAPPDATA%\hermes\git`。无需管理员权限。完全隔离 —— 不会干扰任何系统 Git 安装，无论其是否损坏。（在 32 位 Windows 上，由于 PortableGit 仅提供 64 位和 ARM64 资源，因此会回退到 MinGit；依赖 bash 的 Hermes 功能在 32 位主机上无法使用。）

**为什么不使用 winget？** 早期设计通过 `winget install Git.Git` 自动安装 Git，但当系统 Git 安装处于部分或损坏状态时，winget 会严重失败（而这正是用户需要安装程序正常工作的时候）。便携版 Git 方法绕过了 winget、Windows 安装程序注册表以及任何现有系统 Git。如果 Hermes 的 Git 安装本身损坏，只需执行 `Remove-Item %LOCALAPPDATA%\hermes\git` 并重新运行安装程序 —— 对系统无影响，无需卸载操作。

安装程序还会设置 `HERMES_GIT_BASH_PATH` 指向找到的 `bash.exe`，以便 Hermes 在新的 shell 中确定性地解析它。

如果您更喜欢 WSL2，上述 Linux 安装程序在其中同样适用；原生和 WSL 安装可以共存而不冲突（原生数据位于 `%LOCALAPPDATA%\hermes`，WSL 数据位于 `~/.hermes`）。

### Android / Termux

Hermes 现在也提供 Termux 感知安装路径：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

安装程序会自动检测 Termux 并切换到经过测试的 Android 流程：
- 使用 Termux `pkg` 安装系统依赖（`git`、`python`、`nodejs`、`ripgrep`、`ffmpeg`、构建工具）
- 使用 `python -m venv` 创建 virtualenv
- 自动导出 `ANDROID_API_LEVEL` 用于 Android wheel 构建
- 优先使用范围更广的 `.[termux-all]` extra，如果第一次编译失败则回退到较小的 `.[termux]` extra（最后回退到基础安装）
- 默认跳过未经测试的浏览器 / WhatsApp 引导程序

如果您想要完全显式的路径，请遵循专门的 [Termux 指南](./termux.md)。

:::note Windows 功能对等性（早期 Beta）

原生 Windows 处于 **早期 beta** 阶段。除了基于浏览器的仪表板聊天终端外，其他所有功能都可在 Windows 上原生运行：
- **CLI（`hermes chat`、`hermes setup`、`hermes gateway` 等）** —— 原生，使用您的默认终端
- **Gateway（Telegram、Discord、Slack 等）** —— 原生，以后台 PowerShell 进程运行
- **Cron 调度器** —— 原生
- **浏览器工具** —— 原生（通过 Node.js 使用 Chromium）
- **MCP 服务器** —— 原生（支持 stdio 和 HTTP 两种传输方式）
- **Dashboard `/chat` 终端面板** —— **仅限 WSL2**（使用 POSIX PTY；原生 Windows 没有等效实现）。仪表板的其余部分（sessions、jobs、metrics）可以原生运行 —— 只有嵌入式 PTY 终端标签页被限制。

如果遇到编码相关的 bug，请在环境中设置 `HERMES_DISABLE_WINDOWS_UTF8=1`，以回退到传统的 cp1252 stdio 路径（有助于排查问题）。
:::

### 安装程序的作用

安装程序自动处理所有内容 —— 所有依赖（Python、Node.js、ripgrep、ffmpeg）、仓库克隆、虚拟环境、全局 `hermes` 命令设置以及 LLM provider 配置。完成后，您就可以开始聊天了。

#### 安装布局

安装程序放置文件的位置取决于您是作为普通用户还是 root 安装：

| 安装程序 | 代码位置 | `hermes` 二进制文件 | 数据目录 |
|---|---|---|---|
| 每用户（普通） | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes`（符号链接） | `~/.hermes/` |
| Root 模式（`sudo curl … \| sudo bash`） | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/`（或 `$HERMES_HOME`） |

Root 模式的 **FHS 布局**（`/usr/local/lib/…`、`/usr/local/bin/hermes`）与 Linux 上其他系统级开发者工具的安装位置一致。这对于共享机器部署很有用，其中一个系统安装应服务于所有用户。每个用户的配置（auth、skills、sessions）仍然位于各自的 `~/.hermes/` 或显式的 `HERMES_HOME` 下。

### 安装完成后

重新加载您的 shell 并开始聊天：

```bash
source ~/.bashrc   # 或：source ~/.zshrc
hermes             # 开始聊天！
```

以后要重新配置单个设置，请使用专用命令：

```bash
hermes model          # 选择您的 LLM provider 和模型
hermes tools          # 配置启用哪些工具
hermes gateway setup  # 设置消息平台
hermes config set     # 设置单个配置值
hermes setup          # 或运行完整的设置向导，一次性配置所有内容
```

---

## 前提条件

唯一的前提条件是 **Git**。安装程序会自动处理其他所有内容：

- **uv**（快速 Python 包管理器）
- **Python 3.11**（通过 uv，无需 sudo）
- **Node.js v22**（用于浏览器自动化和 WhatsApp bridge）
- **ripgrep**（快速文件搜索）
- **ffmpeg**（TTS 音频格式转换）

:::info
您 **无需** 手动安装 Python、Node.js、ripgrep 或 ffmpeg。安装程序会检测缺失的内容并为您安装。只需确保 `git` 可用（`git --version`）。
:::

:::tip Nix 用户
如果您使用 Nix（在 NixOS、macOS 或 Linux 上），有专门的设置路径，包括 Nix flake、声明式 NixOS 模块和可选的容器模式。请参阅 **[Nix & NixOS 设置](./nix-setup.md)** 指南。
:::

---

## 手动 / 开发者安装

如果您想要克隆仓库并从源码安装 —— 为了贡献代码、从特定分支运行，或对虚拟环境有完全控制 —— 请参阅贡献指南中的 [开发设置](/developer-guide/contributing#development-setup) 部分。

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| `hermes: command not found` | 重新加载 shell（`source ~/.bashrc`）或检查 PATH |
| `API key not set` | 运行 `hermes model` 配置您的 provider，或 `hermes config set OPENROUTER_API_KEY your_key` |
| 更新后配置丢失 | 运行 `hermes config check` 然后 `hermes config migrate` |

如需更多诊断信息，请运行 `hermes doctor` —— 它会准确告诉您缺少什么以及如何修复。
