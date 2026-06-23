---
title: "Windows (WSL2) 指南"
description: "通过 WSL2 在 Windows 上运行 Hermes Agent —— 安装配置、Windows 与 Linux 之间的文件系统访问、网络设置及常见问题"
sidebar_label: "Windows (WSL2)"
sidebar_position: 2
---

# Windows (WSL2) 指南

Hermes Agent 现已同时支持原生 Windows 和 WSL2。本页介绍 WSL2 路径；如需原生 PowerShell 安装方式，请参阅专属的 **[Windows（原生）指南](./windows-native.md)**。

**何时选择 WSL2 而非原生：**
- 你想使用 dashboard 内嵌终端（`/chat` 标签页）—— 该面板需要 POSIX PTY（伪终端），仅 WSL2 支持。
- 你在进行大量 POSIX 相关的开发工作，希望 Hermes 会话与开发工具共享同一文件系统和路径。
- 你已有 WSL2 环境，不想维护第二套安装。

**何时原生更合适（甚至更好）：**
- 交互式聊天、gateway（Telegram/Discord 等）、cron 调度器、浏览器工具、MCP 服务器以及大多数 Hermes 功能均可在 Windows 上原生运行。
- 你不想在每次引用文件或打开 URL 时都考虑跨越 WSL↔Windows 边界的问题。

在 WSL2 中，实际上有两台"计算机"同时运行：你的 Windows 宿主机，以及由 WSL 管理的 Linux 虚拟机。大多数困惑都源于不清楚自己当前处于哪一侧。

本指南涵盖这种分离中专门影响 Hermes 的部分：安装 WSL2、在 Windows 与 Linux 之间传输文件、双向网络配置，以及实际遇到的常见问题。

:::info 简体中文
最小安装路径的中文说明维护在本页 —— 通过右上角的**语言**菜单切换，选择**简体中文**即可查看。
:::

## 为什么选择 WSL2（而非原生 Windows）

原生 Windows 安装直接运行在 Windows 上：使用 Windows 终端（PowerShell、Windows Terminal 等）、Windows 文件系统路径（`C:\Users\…`）和 Windows 进程。Hermes 使用 Git Bash 执行 shell 命令，这也是 Claude Code 等 agent 目前处理 Windows 的方式 —— 无需完整重写即可绕过 POSIX 与 Windows 的差异。

WSL2 在轻量级虚拟机中运行真实的 Linux 内核，因此其中的 Hermes 与在 Ubuntu 上运行几乎完全相同。当你需要真正的 POSIX 环境时，这非常有价值：`fork`、`/tmp`、UNIX socket、信号语义、PTY 支持的终端、`bash`/`zsh` 等 shell，以及 `rg`、`git`、`ffmpeg` 等在 Linux 上行为一致的工具。

WSL2 的实际影响：

- Hermes CLI、gateway、会话、内存、技能和工具运行时均位于 Linux 虚拟机内部。
- Windows 程序（浏览器、原生应用、带登录 profile 的 Chrome）位于虚拟机外部。
- 每次需要两者通信时 —— 共享文件、打开 URL、控制 Chrome、访问本地模型服务器、将 Hermes gateway 暴露给手机 —— 都需要跨越一道边界。这些边界正是本指南要讲的内容。

## 安装 WSL2

在**管理员 PowerShell** 或 Windows Terminal 中执行：

```powershell
wsl --install
```

在全新的 Windows 10 22H2+ 或 Windows 11 上，此命令会安装 WSL2 内核、虚拟机平台功能以及默认的 Ubuntu 发行版。按提示重启。重启后 Ubuntu 会打开并要求设置 Linux 用户名和密码 —— 这是一个**全新的 Linux 用户**，与你的 Windows 账户无关。

验证你确实在使用 WSL2（而非旧版 WSL1）：

```powershell
wsl --list --verbose
```

应显示 `VERSION  2`。如果某个发行版显示 `VERSION  1`，请转换：

```powershell
wsl --set-version Ubuntu 2
wsl --set-default-version 2
```

Hermes 在 WSL1 上无法可靠运行 —— WSL1 会动态转译 Linux 系统调用，某些行为（procfs、信号、网络）与真实 Linux 存在偏差。

### 发行版选择

我们以 Ubuntu（LTS）为测试基准。Debian 同样可用。Arch 和 NixOS 也有人在用，但一键安装脚本假设使用基于 Debian 的 `apt` 系统 —— 如需其他路径，请参阅 [Nix 安装指南](/getting-started/nix-setup)。

### 启用 systemd（推荐）

Hermes gateway（以及任何你希望持续运行的服务）在 systemd 下更易管理。在现代 WSL 上，在发行版内执行一次即可启用：

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true

[interop]
enabled=true
appendWindowsPath=true

[automount]
options = "metadata,umask=22,fmask=11"
EOF
```

然后在 PowerShell 中执行：

```powershell
wsl --shutdown
```

重新打开 WSL 终端。`ps -p 1 -o comm=` 应输出 `systemd`。

上面的 `metadata` 挂载选项很重要 —— 没有它，`/mnt/c/...` 上的文件无法存储真实的 Linux 权限位，这会导致在 Windows 路径下对脚本执行 `chmod +x` 等操作失效。

### 在 WSL 内安装 Hermes

打开 WSL2 shell 后执行：

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc
hermes
```

安装程序将 WSL2 视为普通 Linux —— 无需任何 WSL 专属配置。完整目录结构请参阅[安装说明](/getting-started/installation)。

## 文件系统：跨越 Windows ↔ WSL2 边界

这是最容易让人踩坑的部分。存在**两套文件系统**，文件放在哪里至关重要 —— 影响性能、正确性以及哪些工具能访问到它。

### 两个方向

| 方向 | 内部路径 | 使用的路径 |
|---|---|---|
| Windows 磁盘，从 WSL 访问 | `C:\Users\you\Documents` | `/mnt/c/Users/you/Documents` |
| WSL 磁盘，从 Windows 访问 | `/home/you/code` | `\\wsl$\Ubuntu\home\you\code`（较新版本为 `\\wsl.localhost\Ubuntu\...`） |

两者都是真实存在的，都可以使用，但它们**不是同一个文件系统** —— 底层通过 9P 网络协议桥接。这带来了真实的性能和语义差异。

### Hermes 和项目应放在哪里

**经验法则：将所有 Linux 相关内容保留在 Linux 文件系统内。**

- 你的 Hermes 安装目录（`~/.hermes/`）—— Linux 侧。安装程序已自动处理。
- 你在 WSL 中开发的 git 仓库 —— Linux 侧（`~/code/...`、`~/projects/...`）。
- 你的模型、数据集、venv —— Linux 侧。

遵循此规则的好处：

- **I/O 速度快。** 对 `/mnt/c/...` 的操作需经过 9P，比原生 ext4 慢 10–100 倍。在 `~/code` 下感觉瞬间完成的 `git status`（10k 文件仓库），在 `/mnt/c` 下可能需要 15 秒以上。
- **权限正确。** Linux 权限位在 `/mnt/c` 上只是尽力模拟。`ssh` 因"权限不当"拒绝密钥，或 `chmod +x` 静默失败，都是常见问题。
- **文件监听可靠。** 跨 9P 的 inotify 不稳定 —— 文件监听器（开发服务器、测试运行器）在 `/mnt/c` 上经常漏报变更。
- **无大小写敏感问题。** Windows 路径默认不区分大小写；Linux 区分大小写。同时包含 `Readme.md` 和 `README.md` 的项目在两侧行为不同。

只有当你**确实需要**文件存在于 Windows 侧时，才将其放在 `/mnt/c` 下 —— 例如需要从 Windows GUI 应用打开，或 Windows Chrome 的 DevTools MCP 需要当前目录是 Windows 可访问的路径。

### 在两侧之间传输文件

**从 Windows → 传入 WSL：** 最简单的方式是打开资源管理器，在地址栏输入 `\\wsl.localhost\Ubuntu`，然后拖放到 `\home\<you>\...`。或者在 PowerShell 中：

```powershell
wsl cp /mnt/c/Users/you/Downloads/file.pdf ~/incoming/
```

**从 WSL → 传入 Windows：** 复制到 `/mnt/c/Users/<you>/...`，Windows 资源管理器会立即看到：

```bash
cp ~/reports/output.pdf /mnt/c/Users/you/Desktop/
```

**在 Windows 应用中打开 WSL 文件**（GUI 编辑器、浏览器等）：使用 `explorer.exe` 或 `wslview`：

```bash
sudo apt install wslu     # 安装一次 —— 提供 wslview、wslpath、wslopen 等工具
wslview ~/reports/output.pdf    # 用 Windows 默认程序打开
explorer.exe .                  # 在 Windows 资源管理器中打开当前 WSL 目录
```

**在两个世界之间转换路径：**

```bash
wslpath -w ~/code/project        # → \\wsl.localhost\Ubuntu\home\you\code\project
wslpath -u 'C:\Users\you'        # → /mnt/c/Users/you
```

### 行尾符、BOM 与 git

如果你在 Windows 侧用 Windows 编辑器编辑文件，可能会产生 `CRLF` 行尾符。当 Linux 侧的 `bash` 或 Python 读取这些文件时，shell 脚本会报错 `bad interpreter: /bin/bash^M`，带 BOM 的 `.env` 文件也可能导致 Python 失败。

解决方法是在 WSL 内（而非 Windows 上）配置合理的 git 设置：

```bash
git config --global core.autocrlf input
git config --global core.eol lf
```

对于已有 CRLF 的文件：

```bash
sudo apt install dos2unix
dos2unix path/to/script.sh
```

### "在 WSL 内 clone 还是在 `/mnt/c` 上 clone？"

在 WSL 内 clone。始终如此，除非有特殊原因。典型的 Hermes 工作流（`hermes chat`、调用 `rg`/`ripgrep` 搜索仓库的工具、文件监听器、后台 gateway）在 `~/code/myrepo` 下会比在 `/mnt/c/Users/you/myrepo` 下快得多，也更可靠。

一个例外：**启动 Windows 二进制文件的 MCP bridge。** 如果你通过 `cmd.exe` 使用 `chrome-devtools-mcp`（参见 [MCP 指南：WSL → Windows Chrome](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)），当 Hermes 的当前工作目录是 `~` 时，Windows 可能会报 `UNC` 警告。此时请从 `/mnt/c/` 下的某个目录启动 Hermes，以便 Windows 进程拥有一个带盘符的工作目录。

## 网络：WSL ↔ Windows

WSL2 在轻量级虚拟机中运行，拥有独立的网络栈。这意味着 WSL 内的 `localhost` 与 Windows 上的 `localhost` **并不相同** —— 从网络角度看，它们是两台独立的主机。对于每个服务，你需要确定流量方向，并选择正确的桥接方式。

以下两种情况最为常见。

### 情况一 —— WSL 中的 Hermes 访问 Windows 上的服务

最常见的场景：你在 **Windows 上运行 Ollama、LM Studio 或 llama-server**，而 WSL 内的 Hermes 需要访问它。

此场景的权威说明在 providers 指南中：**[WSL2 本地模型网络配置 →](/integrations/providers#wsl2-networking-windows-users)**

简要说明：

- **Windows 11 22H2+：** 启用镜像网络模式（在 `%USERPROFILE%\.wslconfig` 中设置 `networkingMode=mirrored`，然后执行 `wsl --shutdown`）。之后 `localhost` 在两侧均可互通。
- **Windows 10 或旧版本：** 使用 Windows 宿主机 IP（WSL 虚拟网络的默认网关），并确保 Windows 上的服务绑定到 `0.0.0.0` 而非仅 `127.0.0.1`。通常还需要在 Windows 防火墙中为该端口添加规则。

完整表格（Ollama / LM Studio / vLLM / SGLang 绑定地址、防火墙规则一行命令、动态 IP 辅助工具、Hyper-V 防火墙解决方案）请点击上方链接 —— 此处不再重复。

### 情况二 —— Windows（或局域网）上的设备访问 WSL 中的 Hermes

这是反向情况，其他地方较少记录，但以下场景需要用到：

- 从 Windows 浏览器使用 Hermes **Web Dashboard**。
- 从 Windows 侧工具使用 **OpenAI 兼容 API 服务器**（当 `API_SERVER_ENABLED=true` 时由 `hermes gateway` 暴露）。参见 [API Server 功能页](/user-guide/features/api-server)。
- 测试**消息 gateway**（Telegram、Discord 等），平台会向本地 webhook URL 发送请求 —— 通常建议使用 `cloudflared`/`ngrok` 而非原始端口转发。

#### 子情况 2a：从 Windows 宿主机本身访问

在**启用了镜像模式的 Windows 11 22H2+** 上，无需任何额外操作。WSL 中绑定到 `0.0.0.0:8080`（甚至 `127.0.0.1:8080`）的进程，可直接从 Windows 浏览器通过 `http://localhost:8080` 访问。WSL 会自动将绑定发布回宿主机。

在 **NAT 模式**（Windows 10 / 旧版 Windows 11）下，WSL2 默认的"localhost 转发"通常会将 Linux 侧的 `127.0.0.1` 绑定转发到 Windows 的 `localhost`，因此以 `--host 127.0.0.1` 启动的 Hermes 服务通常可从 Windows 通过 `http://localhost:PORT` 访问。如果无法访问：

- 在 WSL 内显式绑定到 `0.0.0.0`。
- 用 `ip -4 addr show eth0 | grep inet` 获取 WSL 虚拟机的 IP，然后从 Windows 直接访问该 IP。

#### 子情况 2b：从局域网中的其他设备访问（手机、平板、另一台 PC）

这才是真正麻烦的地方。流量路径为 **局域网设备 → Windows 宿主机 → WSL 虚拟机**，你需要分别配置两段：

1. **在 WSL 内绑定所有网络接口。** 监听 `127.0.0.1` 的进程永远无法从虚拟机外部访问。请使用 `0.0.0.0`。

2. **配置 Windows → WSL 虚拟机的端口转发。** 镜像模式下自动完成。NAT 模式下需要在管理员 PowerShell 中手动配置，每个端口单独设置：

   ```powershell
   # 获取 WSL 虚拟机当前 IP（NAT 模式下每次重启 WSL 都会变化）
   $wslIp = (wsl hostname -I).Trim().Split(' ')[0]

   # 将 Windows 端口 8080 转发到 WSL:8080
   netsh interface portproxy add v4tov4 `
     listenaddress=0.0.0.0 listenport=8080 `
     connectaddress=$wslIp connectport=8080

   # 在 Windows 防火墙中放行该端口
   New-NetFirewallRule -DisplayName "Hermes WSL 8080" `
     -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
   ```

   之后可用以下命令删除：`netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080`。

3. **让局域网设备访问 `http://<windows-lan-ip>:8080`。**

由于 NAT 模式下 WSL 虚拟机 IP 在每次重启后都会变化，一次性配置的规则在下次 `wsl --shutdown` 后即失效。如需持久化，要么启用镜像模式，要么将端口代理步骤写入 Windows 登录时自动运行的脚本。

对于来自云端消息服务商的 webhook（Telegram `setWebhook`、Slack 事件等），不建议折腾端口转发 —— 请使用 `cloudflared` 隧道。参见 [webhook 指南](/user-guide/messaging/webhooks)。

## 在 Windows 上长期运行 Hermes 服务

Hermes 的 [Tool Gateway](/user-guide/features/tool-gateway) 和 API 服务器都是长期运行的进程。在 WSL2 中，有以下几种方式保持它们持续运行。

### 在 WSL 内使用 systemd（推荐）

如果你按照上面的安装步骤启用了 systemd，`hermes gateway` 和 API 服务器的使用方式与任何 Linux 机器上完全相同。使用 gateway 设置向导：

```bash
hermes gateway setup
```

它会提示是否安装 systemd 用户单元，以便在 WSL 启动时自动拉起 gateway。

### 让 WSL 在 Windows 登录时自动启动

WSL 虚拟机只在有进程使用时保持运行。若要在没有终端窗口的情况下保持 gateway 可访问，可通过任务计划程序在 Windows 登录时启动一个 WSL 进程：

- **触发器：** 用户登录时（你的账户）。
- **操作：** 启动程序
  - 程序：`C:\Windows\System32\wsl.exe`
  - 参数：`-d Ubuntu --exec /bin/sh -c "sleep infinity"`

这样可以保持虚拟机存活，使 systemd 管理的 gateway 持续运行。在 Windows 11 上，较新的 `wsl --install --no-launch` + 自动启动流程也可以实现；`sleep infinity` 方案是兼容性最好的版本。

## GPU 直通（本地模型）

WSL2 自 WSL 内核 5.10.43+ 起原生支持 **NVIDIA** GPU —— 在 Windows 上安装标准 NVIDIA 驱动（**不要**在 WSL 内安装 Linux NVIDIA 驱动），WSL 内的 `nvidia-smi` 即可识别 GPU。之后，CUDA 工具链、`torch`、`vllm`、`sglang` 和 `llama-server` 均可正常使用真实 GPU。

AMD ROCm 和 Intel Arc 在 WSL2 内的支持仍在发展中，不在 Hermes 的测试范围内 —— 使用当前驱动可能可以工作，但我们暂无推荐方案。

如果你运行的是**原生 Windows** 本地模型服务器（Windows 版 Ollama、LM Studio），它已通过 Windows 驱动使用 GPU，则完全不需要 WSL GPU 直通 —— 只需按照上面的情况一，从 WSL 通过网络访问即可。

## 常见问题

**连接 Windows 上的 Ollama / LM Studio 时报"Connection refused"。**
参见 [WSL2 网络配置](/integrations/providers#wsl2-networking-windows-users)。九成情况是服务绑定在 `127.0.0.1` 上，需要改为 `0.0.0.0`（Ollama：`OLLAMA_HOST=0.0.0.0`），或者缺少防火墙规则。

**`git status` / `hermes chat` 在仓库中极慢。**
你很可能在 `/mnt/c/...` 下工作。将仓库移到 `~/code/...`（Linux 侧），速度会有数量级的提升。

**脚本报错 `bad interpreter: /bin/bash^M`。**
Windows 编辑器产生的 CRLF 行尾符。执行 `dos2unix script.sh`，并在 WSL git 配置中设置 `core.autocrlf input`。

**通过 MCP 启动 Windows 二进制文件时出现"UNC paths are not supported"警告。**
Hermes 的工作目录在 Linux 文件系统内，Windows `cmd.exe` 无法识别。在该会话中从 `/mnt/c/...` 下启动 Hermes，或使用一个在调用 Windows 可执行文件前先 `cd` 到 Windows 可访问路径的包装脚本。

**休眠/睡眠后时钟漂移。**
宿主机从睡眠恢复后，WSL2 的时钟可能滞后数分钟，导致所有基于证书的操作失败（OAuth、HTTPS API）。按需修复：

```bash
sudo hwclock -s
```

或安装 `ntpdate` 并在登录时运行。

**启用镜像模式后或连接 VPN 时 DNS 停止工作。**
镜像模式会将宿主机网络设置代理到 WSL —— 如果 Windows DNS 有问题（VPN 分流隧道、企业解析器），WSL 会继承这些问题。解决方法：手动覆盖 `resolv.conf`（在 `/etc/wsl.conf` 中设置 `generateResolvConf=false`，然后手动编写 `/etc/resolv.conf`，填入 `1.1.1.1` 或你的 VPN DNS）。

**运行安装程序后找不到 `hermes` 命令。**
安装程序通过 `~/.bashrc` 将 `~/.local/bin` 添加到 shell 的 PATH 中。需要执行 `source ~/.bashrc`（或打开新终端）才能在当前会话中生效。

**Windows Defender 对 WSL 文件扫描很慢。**
Defender 通过 9P 桥接扫描从 Windows 访问的文件，这会放大 `/mnt/c` 风格跨边界访问的延迟。如果你只在 WSL 内部访问 WSL 文件，则不受影响。如果你频繁使用 Windows 工具访问 `\\wsl$\...`，可考虑将 WSL 发行版路径排除在实时扫描之外。

**磁盘空间不足。**
WSL2 将虚拟机磁盘存储为 `%LOCALAPPDATA%\Packages\...` 下的稀疏 VHDX 文件。它会自动增长，但删除文件后不会自动收缩。回收空间的方法：执行 `wsl --shutdown`，然后在管理员 PowerShell 中运行 `Optimize-VHD -Path <path-to-ext4.vhdx> -Mode Full`（需要 Hyper-V 工具），或使用 WSL 文档中记录的更简单的 `diskpart` 方式。

## 下一步

- **[安装说明](/getting-started/installation)** —— 实际安装步骤（Linux/WSL2/Termux 均使用同一安装程序）。
- **[集成 → Providers → WSL2 网络配置](/integrations/providers#wsl2-networking-windows-users)** —— 本地模型服务器网络配置的权威深度说明。
- **[MCP 指南 → WSL → Windows Chrome](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)** —— 从 WSL 中的 Hermes 控制你已登录的 Windows Chrome。
- **[Tool Gateway](/user-guide/features/tool-gateway)** 和 **[Web Dashboard](/user-guide/features/web-dashboard)** —— 你最常需要从 WSL 暴露到网络其他部分的长期运行服务。