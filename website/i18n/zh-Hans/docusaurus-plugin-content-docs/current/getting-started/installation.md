---
sidebar_position: 2
title: "安装指南"
description: "在 Linux、macOS、WSL2、Windows 或 Android Termux 上安装 Hermes Agent"
---

# 安装指南

两分钟内完成 Hermes Agent 的安装和运行。

:::tip 平台支持
完整的平台支持矩阵（支持的操作系统、分发方式和平台特性），请参阅 **[平台支持](./platform-support.md)**。
:::

## 快速安装

### 使用 Hermes Desktop 安装器（推荐，macOS / Windows）

如需同时安装命令行和桌面应用，请从官网下载 [Hermes Desktop 安装器](https://hermes-agent.nousresearch.com/) 并运行。

### 纯命令行安装

#### Linux / macOS / WSL2 / Android (Termux)

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows（原生）

在 PowerShell 中运行：

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

如果你先做了纯命令行安装，之后想加装 Hermes Desktop，只需运行：

```bash
hermes desktop
```

### 安装器做了什么

安装器会自动处理所有依赖——Python、Node.js、ripgrep、ffmpeg，克隆仓库，创建虚拟环境，配置全局 `hermes` 命令，以及 LLM 提供商配置。安装完成后即可直接使用。

#### 安装目录结构

安装位置取决于你以普通用户还是 root 身份运行：

| 安装方式 | 代码位置 | `hermes` 命令 | 数据目录 |
| --- | --- | --- | --- |
| 普通用户（git 安装器） | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes`（符号链接） | `~/.hermes/` |
| root 模式（`sudo curl … \| sudo bash`） | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/`（或 `$HERMES_HOME`） |

root 模式的 **FHS 布局**（`/usr/local/lib/…`、`/usr/local/bin/hermes`）与其他系统级开发工具一致，适合共享服务器部署——一次安装，多用户使用。每个用户的个人配置（认证、技能、会话）仍然存储在各自 `~/.hermes/` 下。

### 安装后

重新加载 Shell 然后开始对话：

```bash
source ~/.bashrc   # 或：source ~/.zshrc
hermes              # 开始对话！
```

后续如需调整配置，使用以下命令：

```bash
hermes model          # 选择 LLM 提供商和模型
hermes tools          # 配置启用的工具
hermes gateway setup  # 设置消息平台
hermes config set     # 设置单个配置值
hermes config get     # 查看单个配置值
hermes setup          # 运行完整配置向导，一次性配置所有项
```

:::tip 最快路径：Nous Portal
一个订阅即可覆盖 300+ 模型和 [工具网关](/user-guide/features/tool-gateway)（网页搜索、图像生成、TTS、云端浏览器），无需为每个工具单独配置密钥：

```bash
hermes setup --portal
```

一条命令完成登录、设置 Nous 为提供商并启用工具网关。
:::

---

## 前置依赖

**安装器：** 非 Windows 平台唯一的前置依赖是 **Git**。在 Linux 上，还需确保 `curl` 和 `xz-utils` 可用（安装器以 `.tar.xz` 格式下载 Node.js）。桌面应用额外需要 `g++`（Debian/Ubuntu 上的 `build-essential`）来编译原生模块。安装器会自动处理其余所有依赖：

- **uv**（快速 Python 包管理器）
- **Python 3.11**（通过 uv，无需 sudo）
- **Node.js v22**（用于浏览器自动化和 WhatsApp 桥接）
- **ripgrep**（快速文件搜索）
- **ffmpeg**（TTS 音频格式转换）

:::info
你**不需要**手动安装 Python、Node.js、ripgrep 或 ffmpeg。安装器会检测缺失项并自动安装。只需确保 `git` 可用（`git --version`）。在 Linux 上，确保 `curl` 和 `xz-utils` 已安装（Debian/Ubuntu：`sudo apt install curl xz-utils`）。桌面应用还需安装 `build-essential`（`sudo apt install build-essential`）。
:::

:::tip Nix 用户
Nix **不再是官方支持的安装路径**（仅提供尽力而为的支持）。如果你已经在使用 Nix（NixOS、macOS 或 Linux），有专门的 Nix flake 安装路径，包含声明式 NixOS 模块和可选的容器模式。详见 **[Nix 和 NixOS 安装指南](./nix-setup.md)**。
:::

---

## 手动/开发者安装

如果你需要克隆仓库并从源码安装——用于贡献代码、从特定分支运行，或完全控制虚拟环境——请参阅贡献指南中的 [开发环境搭建](../developer-guide/contributing.md#开发环境搭建) 章节。

---

## 无 Sudo / 服务用户安装

Hermes 支持以专用非特权用户身份运行（如 `hermes` systemd 服务账户，或任何无 `sudo` 权限的用户）。安装路径中唯一真正需要 root 权限的步骤是 Playwright 的 `--with-deps`，它会通过 `apt` 安装 Chromium 所需的共享库（`libnss3`、`libxkbcommon` 等）。安装器会检测 sudo 是否可用，不可用时自动降级——它会将 Chromium 二进制安装到服务用户自己的 Playwright 缓存中，并打印管理员需要手动执行的命令。

**推荐的分步方案（Debian/Ubuntu）：**

1. **一次性操作：以有 sudo 权限的管理员身份**，安装 Chromium 需要的系统库：

   ```bash
   sudo npx playwright install-deps chromium
   ```
   （可以在任意目录运行——`npx` 会自动获取 Playwright。）

2. **以非特权服务用户身份**，运行标准安装器。它会检测到缺少 sudo，跳过 `--with-deps`，将 Chromium 安装到用户本地 Playwright 缓存：

   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

   如果你不需要浏览器自动化功能（例如纯无头运行），可以通过 `--skip-browser` 跳过 Playwright：

   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-browser
   ```

3. **让 `hermes` 命令对服务用户可用。** 安装器将启动器写入 `~/.local/bin/hermes`。系统服务账户的 PATH 通常不包含 `~/.local/bin`。你可以将其添加到用户环境变量中，或者将启动器链接到系统路径：

   ```bash
   # 方案 A — 添加到服务用户的 profile
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # 方案 B — 系统级符号链接（以管理员身份运行）
   sudo ln -s /home/hermes/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
   ```

4. **验证：** `hermes doctor` 应该正常运行。如果出现 `ModuleNotFoundError: No module named 'dotenv'`，说明你使用系统 Python 调用了仓库源文件（`~/.hermes/hermes-agent/hermes`），而不是 venv 启动器（`~/.hermes/hermes-agent/venv/bin/hermes`）——请修正步骤 3。

5. **从此账户运行消息网关？** 用户级服务会在登出时停止，且不会在启动时自动运行。你需要为服务用户启用 lingering：

   ```bash
   sudo loginctl enable-linger <服务用户名>
   ```

   服务本身的配置详见 [消息网关](/user-guide/messaging/)。

同样的方案适用于 Arch（安装器使用 pacman，有相同的 sudo 检测逻辑）、Fedora/RHEL 和 openSUSE——这些发行版完全不支持 `--with-deps`，因此系统库始终由管理员单独安装。安装器会打印相应的 `dnf` / `zypper` 命令。

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| `hermes: command not found` | 重新加载 Shell（`source ~/.bashrc`）或检查 PATH |
| `API key not set` | 运行 `hermes model` 配置提供商，或 `hermes config set OPENROUTER_API_KEY your_key` |
| 更新后配置丢失 | 运行 `hermes config check` 然后 `hermes config migrate` |

更多诊断信息，运行 `hermes doctor`——它会准确告诉你缺少什么以及如何修复。

## 安装方式自动检测

Hermes 会自动检测是通过 git 安装器、Docker 还是 NixOS 安装的，`hermes update` 会打印对应路径的更新命令。不需要设置环境变量——检测基于安装布局（`~/.hermes/hermes-agent/` 仓库检出、Docker 镜像标记或 Nix 存储路径）。`hermes doctor` 也会在环境摘要中显示检测到的安装方式。
