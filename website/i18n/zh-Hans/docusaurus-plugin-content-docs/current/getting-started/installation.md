---
sidebar_position: 2
title: "安装"
description: "在 Linux、macOS、WSL2、原生 Windows 或通过 Termux 在 Android 上安装 Hermes Agent"
---

# 安装

使用一行安装命令，两分钟内即可启动并运行 Hermes Agent。

## 快速安装

### 一行安装命令（Linux / macOS / WSL2）

基于 git 的安装方式，跟踪 `main` 分支，可立即获取最新变更：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Windows（原生，PowerShell）

原生 Windows 无需 WSL 即可运行 Hermes——CLI、gateway、TUI 和工具均可原生运行。（原生安装与 WSL2 安装可干净共存；唯一仅限 WSL2 的功能见下方功能说明。）遇到 bug 请[提交 issue](https://github.com/NousResearch/hermes-agent/issues)。

打开 PowerShell 并运行：

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

安装程序处理**一切**：`uv`、Python 3.11、Node.js 22、`ripgrep`、`ffmpeg`，**以及一个便携式 Git Bash**（PortableGit——一个自包含的 Git-for-Windows 发行版，附带 `bash.exe` 和 Hermes 用于 shell 命令的完整 POSIX 工具链；在 32 位 Windows 上安装程序会回退到 MinGit，后者缺少 bash，终端工具和 agent 浏览器功能将被禁用）。它将仓库克隆到 `%LOCALAPPDATA%\hermes\hermes-agent`，创建虚拟环境，并将 `hermes` 添加到**用户 PATH**。安装完成后请重启终端（或打开新的 PowerShell 窗口）以使 PATH 生效。

**Git 的处理方式：**
1. 如果 `git` 已在你的 PATH 中，安装程序将使用现有安装。
2. 否则，它会下载便携式 **PortableGit**（约 50MB，来自官方 `git-for-windows` GitHub 发布页）并解压到 `%LOCALAPPDATA%\hermes\git`。无需管理员权限，完全隔离——不会干扰任何系统 Git 安装，无论其状态如何。（在 32 位 Windows 上会回退到 MinGit，因为 PortableGit 仅提供 64 位和 ARM64 资产；依赖 bash 的 Hermes 功能在 32 位主机上无法使用。）

**为什么不使用 winget？** 早期设计通过 `winget install Git.Git` 自动安装 Git，但当系统 Git 安装处于部分损坏状态时，winget 会严重失败（而这恰恰是用户最需要安装程序正常工作的时候）。便携式 Git 方案绕过了 winget、Windows 安装程序注册表以及任何现有系统 Git。如果 Hermes 的 Git 安装本身出现问题，执行 `Remove-Item %LOCALAPPDATA%\hermes\git` 并重新运行安装程序即可——对系统无影响，无需卸载操作。

安装程序还会将 `HERMES_GIT_BASH_PATH` 设置为找到的 `bash.exe` 路径，以便 Hermes 在新 shell 中确定性地解析它。

如果你偏好 WSL2，上方的 Linux 安装程序可在其中运行；原生安装和 WSL 安装可以共存而不冲突（原生数据位于 `%LOCALAPPDATA%\hermes`，WSL 数据位于 `~/.hermes`）。

**桌面安装程序（替代方案）：** 也提供一个轻量 GUI 安装程序——下载 Hermes Desktop，运行 `.exe`，首次启动时它会在后台调用 `install.ps1` 来配置 Python（通过 `uv`）、Node、PortableGit 及其余依赖。桌面应用和 PowerShell 安装的 CLI 共享相同的安装目录和数据目录，可以单独或同时使用。详见 [Windows（原生）指南](../user-guide/windows-native#desktop-installer-alternative)。

### Android / Termux

Hermes 现在也提供 Termux 感知的安装路径：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

安装程序会自动检测 Termux 并切换到经过测试的 Android 流程：
- 使用 Termux `pkg` 安装系统依赖（`git`、`python`、`nodejs`、`ripgrep`、`ffmpeg`、构建工具）
- 使用 `python -m venv` 创建虚拟环境
- 自动导出 `ANDROID_API_LEVEL` 以用于 Android wheel 构建
- 优先使用较宽泛的 `.[termux-all]` extra，若首次编译失败则回退到较小的 `.[termux]` extra（最终回退到基础安装）
- 默认跳过未经测试的浏览器 / WhatsApp 引导

如需完整的显式步骤，请参阅专门的 [Termux 指南](./termux.md)。

:::note Windows 功能对等性

除基于浏览器的 dashboard 聊天终端外，其余功能均可在 Windows 上原生运行：
- **CLI（`hermes chat`、`hermes setup`、`hermes gateway` 等）** — 原生，使用默认终端
- **Gateway（Telegram、Discord、Slack 等）** — 原生，作为后台 PowerShell 进程运行
- **Cron 调度器** — 原生
- **浏览器工具** — 原生（通过 Node.js 使用 Chromium）
- **MCP 服务器** — 原生（stdio 和 HTTP 传输均支持）
- **Dashboard `/chat` 终端面板** — **仅限 WSL2**（使用 POSIX PTY（伪终端），原生 Windows 无等效实现）。Dashboard 的其余部分（会话、任务、指标）可原生运行——仅嵌入式 PTY 终端标签页受限。

如果遇到编码相关的 bug 并希望回退到旧版 cp1252 stdio 路径（用于问题定位），请在环境中设置 `HERMES_DISABLE_WINDOWS_UTF8=1`。
:::

### 安装程序做了什么

安装程序自动处理一切——所有依赖（Python、Node.js、ripgrep、ffmpeg）、仓库克隆、虚拟环境、全局 `hermes` 命令配置以及 LLM 提供商配置。完成后即可开始聊天。

#### 安装目录结构

安装程序的存放位置取决于你是以普通用户还是 root 身份安装：

| 安装方式 | 代码位置 | `hermes` 二进制 | 数据目录 |
|---|---|---|---|
| pip install | Python site-packages | `~/.local/bin/hermes`（console_scripts） | `~/.hermes/` |
| 用户级（git 安装程序） | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes`（符号链接） | `~/.hermes/` |
| Root 模式（`sudo curl … \| sudo bash`） | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/`（或 `$HERMES_HOME`） |

Root 模式的 **FHS 布局**（`/usr/local/lib/…`、`/usr/local/bin/hermes`）与其他系统级开发工具在 Linux 上的安装位置一致。适用于共享机器部署场景，一次系统安装可服务所有用户。每个用户的个人配置（认证、技能、会话）仍位于各自的 `~/.hermes/` 或显式指定的 `HERMES_HOME` 下。

### 安装后

重新加载 shell 并开始聊天：

```bash
source ~/.bashrc   # 或：source ~/.zshrc
hermes             # 开始聊天！
```

如需稍后重新配置单项设置，使用以下专用命令：

```bash
hermes model          # 选择 LLM 提供商和模型
hermes tools          # 配置启用的工具
hermes gateway setup  # 配置消息平台
hermes config set     # 设置单个配置项
hermes setup          # 或运行完整的设置向导一次性配置所有内容
```

:::tip 最快路径：Nous Portal
一个订阅涵盖 300+ 个模型以及 [Tool Gateway](/user-guide/features/tool-gateway)（网络搜索、图像生成、TTS、云端浏览器）。无需逐一管理各工具的密钥：

```bash
hermes setup --portal
```

该命令一次性完成登录、设置 Nous 为提供商并开启 Tool Gateway。
:::

---

## 前置条件

**pip install：** 除 Python 3.11+ 外无其他前置条件，其余均自动处理。

**Git 安装程序：** 唯一的前置条件是 **Git**。安装程序自动处理其余一切：

- **uv**（快速 Python 包管理器）
- **Python 3.11**（通过 uv，无需 sudo）
- **Node.js v22**（用于浏览器自动化和 WhatsApp 桥接）
- **ripgrep**（快速文件搜索）
- **ffmpeg**（TTS 的音频格式转换）

:::info
你**无需**手动安装 Python、Node.js、ripgrep 或 ffmpeg。安装程序会检测缺失的依赖并自动安装。只需确保 `git` 可用（`git --version`）。
:::

:::tip Nix 用户
如果你使用 Nix（在 NixOS、macOS 或 Linux 上），有专门的配置路径，包含 Nix flake、声明式 NixOS 模块和可选容器模式。请参阅 **[Nix & NixOS 配置](./nix-setup.md)** 指南。
:::

---

## 手动 / 开发者安装

如果你想克隆仓库并从源码安装——用于贡献代码、从特定分支运行或完全控制虚拟环境——请参阅贡献指南中的[开发环境配置](../developer-guide/contributing.md#development-setup)章节。

---

## 非 Sudo / 系统服务用户安装

支持以专用非特权用户身份运行 Hermes（例如 `hermes` systemd 服务账户，或任何没有 `sudo` 权限的用户）。安装路径中真正需要 root 权限的只有 Playwright 的 `--with-deps` 步骤，该步骤通过 `apt` 安装 Chromium 所需的共享库（`libnss3`、`libxkbcommon` 等）。安装程序会检测 sudo 是否可用，并在不可用时优雅降级——它会将 Chromium 二进制安装到服务用户自己的 Playwright 缓存中，并打印管理员需要单独运行的确切命令。

**推荐的分步方式（Debian/Ubuntu）：**

1. **一次性操作，以具有 sudo 权限的管理员用户身份**，安装 Chromium 所需的系统库：
   ```bash
   sudo npx playwright install-deps chromium
   ```
   （可在任意位置运行——`npx` 会自动获取 Playwright。）

2. **以非特权服务用户身份**，运行常规安装程序。它会检测到缺少 sudo，跳过 `--with-deps`，并将 Chromium 安装到用户本地的 Playwright 缓存中：
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
   ```

   如果想完全跳过 Playwright 步骤——例如在无头环境中运行且不需要浏览器自动化——传入 `--skip-browser`：
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-browser
   ```

3. **使 `hermes` 对服务用户的 shell 可用。** 安装程序将启动器写入 `~/.local/bin/hermes`。系统服务账户通常具有不包含 `~/.local/bin` 的最小 PATH。可以将其添加到用户环境，或将启动器符号链接到系统位置：
   ```bash
   # 方案 A — 添加到服务用户的 profile
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # 方案 B — 系统级符号链接（以管理员身份运行）
   sudo ln -s /home/hermes/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
   ```

4. **验证：** `hermes doctor` 现在应能正常运行。如果出现 `ModuleNotFoundError: No module named 'dotenv'`，说明你在用系统 Python 调用仓库源码中的 `hermes` 文件（`~/.hermes/hermes-agent/hermes`），而非 venv 启动器（`~/.hermes/hermes-agent/venv/bin/hermes`）——请修正步骤 3。

同样的方式适用于 Arch（安装程序使用 pacman，具有相同的 sudo 检测逻辑）、Fedora/RHEL 和 openSUSE——这些发行版完全不支持 `--with-deps`，因此管理员始终需要单独安装系统库。安装程序会打印相应的 `dnf`/`zypper` 命令。

---

## 故障排查

| 问题 | 解决方案 |
|---------|----------|
| `hermes: command not found` | 重新加载 shell（`source ~/.bashrc`）或检查 PATH |
| `API key not set` | 运行 `hermes model` 配置提供商，或 `hermes config set OPENROUTER_API_KEY your_key` |
| 更新后配置丢失 | 运行 `hermes config check`，然后运行 `hermes config migrate` |

如需更多诊断信息，运行 `hermes doctor`——它会告诉你确切缺少什么以及如何修复。

## 安装方式自动检测

Hermes 会自动检测安装方式（`pip`、git 安装程序、Homebrew 或 NixOS），`hermes update` 会打印对应路径的更新命令。无需设置任何环境变量——检测基于安装目录结构（Python site-packages、`~/.hermes/hermes-agent/`、Homebrew 前缀或 Nix store 路径）。`hermes doctor` 也会在其环境摘要中显示检测到的安装方式。