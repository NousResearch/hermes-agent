P---
title: "Windows（原生）指南"
description: "在 Windows 10 / 11 上原生运行 Hermes Agent — 安装、功能矩阵、UTF-8 控制台、Git Bash、将 gateway 作为计划任务、编辑器处理、PATH、卸载及常见问题"
sidebar_label: "Windows（原生）"
sidebar_position: 3
---

# Windows（原生）指南

Hermes 可在 Windows 10 和 Windows 11 上原生运行——无需 WSL、Cygwin 或 Docker。本页是深度指南：原生支持哪些功能、哪些仅限 WSL、安装程序实际做了什么，以及你可能需要调整的 Windows 专属配置项。

如果你只是想安装，[首页](/) 或[安装页面](../getting-started/installation#windows原生powershell)上的一行命令就够了。遇到意外情况时再回来查阅本页。

:::tip 想用 WSL？
如果你更倾向于真正的 POSIX 环境（用于 dashboard 内嵌终端、`fork` 语义、Linux 风格文件监视器等），请参阅 **[Windows（WSL2）指南](./windows-wsl-quickstart.md)**。两者可以干净共存：原生数据存放在 `%LOCALAPPDATA%\hermes`，WSL 数据存放在 `~/.hermes`。
:::

## 快速安装

打开 **PowerShell**（或 Windows Terminal）并运行：

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

无需管理员权限。安装程序会写入 `%LOCALAPPDATA%\hermes\`，并将 `hermes` 添加到你的**用户 PATH**——安装完成后打开新终端即可使用。

**安装程序选项**（需要使用 scriptblock 形式传递参数）：

```powershell
& ([scriptblock]::Create((irm https://hermes-agent.nousresearch.com/install.ps1))) -NoVenv -SkipSetup -Branch main
```

| 参数          | 默认值                               | 用途                                            |
| ------------- | ------------------------------------ | ----------------------------------------------- |
| `-Branch`     | `main`                               | 克隆指定分支（用于测试 PR）                     |
| `-Commit`     | 未设置                               | 将安装固定到指定 commit SHA（覆盖 `-Branch`）   |
| `-Tag`        | 未设置                               | 将安装固定到指定 git tag（如 `v0.14.0`）        |
| `-NoVenv`     | 关闭                                 | 跳过 venv 创建（高级用法——由你自行管理 Python） |
| `-SkipSetup`  | 关闭                                 | 跳过安装后的 `hermes setup` 向导                |
| `-HermesHome` | `%LOCALAPPDATA%\hermes`              | 覆盖数据目录                                    |
| `-InstallDir` | `%LOCALAPPDATA%\hermes\hermes-agent` | 覆盖代码存放位置                                |

安装程序会自动重试不稳定的 git 拉取，并剥离下载的 `install.ps1` 内容中的 BOM，因此 HTTP 传输中携带的 UTF-8 BOM 不再会破坏 `[scriptblock]::Create((irm ...))` 形式。

### 桌面安装程序（备选方案）

也提供了一个轻量 GUI 安装程序——如果你更倾向于双击 `.exe` 而非打开 PowerShell，可以使用它。下载 Hermes Desktop，运行安装程序，首次启动时 GUI 会在后台调用 `install.ps1` 来配置 Python（通过 `uv`）、Node、PortableGit 以及下文描述的其余依赖引导流程。首次运行后，桌面应用与 PowerShell 安装的 `hermes` CLI 共享同一个 `%LOCALAPPDATA%\hermes\hermes-agent` 安装目录和 `%USERPROFILE%\.hermes` 数据目录——可以在 GUI 和 CLI 之间自由切换。

如果你想要熟悉的 Windows 安装体验，或者要将 Hermes 交给非开发者使用，请使用桌面安装程序；如果你已经在终端中，请使用 PowerShell 一行命令。

### 依赖引导（`dep_ensure`）

在首次启动时（以及检测到缺少工具时按需触发），Hermes 会运行一个小型 Python 引导程序——`hermes_cli/dep_ensure.py`——检查并懒加载安装所需的非 Python 依赖。在 Windows 上，相关依赖如下：

| 依赖            | Hermes 需要它的原因                                                                             |
| --------------- | ----------------------------------------------------------------------------------------------- |
| **PortableGit** | 为终端工具提供 `bash.exe`，为会话内克隆提供 `git`。在安装时配置，而非由 `dep_ensure` 负责。     |
| **Node.js 22**  | 浏览器工具（`agent-browser`）、TUI 的 web 桥接以及 WhatsApp 桥接所必需。                        |
| **ffmpeg**      | TTS / 语音消息的音频格式转换。                                                                  |
| **ripgrep**     | 快速文件搜索——不可用时回退到 `grep`。                                                           |
| **npm 包**      | `agent-browser`、Playwright Chromium 以及各工具集的 Node 依赖，在首次使用浏览器工具时安装一次。 |

每个依赖都有类似 `shutil.which(...)` 的检查；如果二进制文件缺失且当前为交互式运行，`dep_ensure` 会提示安装（实际安装逻辑委托给 `scripts\install.ps1 -ensure <dep>`）。非交互式运行（gateway、cron、无头桌面启动）会跳过提示，并直接给出清晰的 `this feature needs <dep>` 错误。

## 安装程序实际做了什么

从头到尾，按顺序：

1. **引导 `uv`** — Astral 的快速 Python 管理器。安装到 `%USERPROFILE%\.local\bin`。
2. **通过 `uv` 安装 Python 3.11**。无需预先安装 Python。
3. **安装 Node.js 22**（优先使用 winget，否则将便携式 Node 压缩包解压到 `%LOCALAPPDATA%\hermes\node`）。用于浏览器工具和 WhatsApp 桥接。
4. **安装便携式 Git** — 如果 `git` 已在 PATH 中，安装程序直接使用；否则从官方 `git-for-windows` 发布版下载精简的自包含 **PortableGit**（约 45 MB）到 `%LOCALAPPDATA%\hermes\git`。无需管理员权限，不写入 Windows 安装程序注册表，不干扰系统上的其他任何内容。
5. **将仓库克隆**到 `%LOCALAPPDATA%\hermes\hermes-agent` 并在其中创建 virtualenv。
6. **分层 `uv pip install`** — 先尝试 `.[all]`，如果 `git+https` 依赖在 GitHub 限速时失败，则逐步回退到更小的集合（`[messaging,dashboard,ext]` → `[messaging]` → `.`）。防止"单次失败导致裸安装"的故障模式。
7. **根据 `.env` 自动安装消息 SDK** — 如果存在 `TELEGRAM_BOT_TOKEN` / `DISCORD_BOT_TOKEN` / `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` / `WHATSAPP_ENABLED`，则运行 `python -m ensurepip --upgrade` 并针对性地调用 `pip install`，确保各平台 SDK 可正常导入。
8. **设置 `HERMES_GIT_BASH_PATH`** 为解析后的 `bash.exe` 路径，使 Hermes 在新 shell 中能确定性地找到它。
9. **将 `%LOCALAPPDATA%\hermes\bin` 添加到用户 PATH** — 打开新终端后即可使用 `hermes` 命令。
10. **运行 `hermes setup`** — 正常的首次运行向导（模型、提供商、工具集）。使用 `-SkipSetup` 跳过。

:::tip 在 Windows 上跳过繁琐的提供商配置
在 Windows 上，逐个配置工具 API key（Firecrawl、FAL、Browser Use、OpenAI TTS）是获得可用 agent 摩擦最大的部分。[Nous Portal](/user-guide/features/tool-gateway) 订阅通过一次 OAuth 登录即可覆盖模型**以及**所有这些工具。安装程序完成后，运行 `hermes setup --portal` 完成配置。
:::

## 功能矩阵

除 dashboard 内嵌终端面板外，所有功能均可在 Windows 上原生运行。

| 功能                                                         | 原生 Windows        | WSL2               |
| ------------------------------------------------------------ | ------------------- | ------------------ |
| CLI（`hermes chat`、`hermes setup`、`hermes gateway` 等）    | ✓                   | ✓                  |
| 交互式 TUI（`hermes --tui`）                                 | ✓                   | ✓                  |
| 消息 gateway（Telegram、Discord、Slack、WhatsApp，15+ 平台） | ✓                   | ✓                  |
| Cron 调度器                                                  | ✓                   | ✓                  |
| 浏览器工具（通过 Node 驱动 Chromium）                        | ✓                   | ✓                  |
| MCP 服务器（stdio 和 HTTP）                                  | ✓                   | ✓                  |
| 本地 Ollama / LM Studio / llama-server                       | ✓                   | ✓（通过 WSL 网络） |
| Web dashboard（会话、任务、指标、配置）                      | ✓                   | ✓                  |
| Dashboard `/chat` 内嵌终端面板                               | ✗（需要 POSIX PTY） | ✓                  |
| 登录时自动启动                                               | ✓（schtasks）       | ✓（systemd）       |

Dashboard 的 `/chat` 标签页通过 POSIX PTY（`ptyprocess`）内嵌了真实终端。原生 Windows 没有等效的原语；Python 的 `pywinpty` / Windows ConPTY 可以实现，但需要单独的实现——视为未来工作。**dashboard 的其余部分均可原生运行**——只有该标签页会显示"请使用 WSL2"的提示横幅。

## Hermes 在 Windows 上如何运行 shell 命令

Hermes 的终端工具通过 **Git Bash** 运行命令，与 Claude Code 采用相同策略。这在不重写每个工具的情况下绕过了 POSIX 与 Windows 的差异。

`bash.exe` 的解析顺序：

1. 如果设置了 `HERMES_GIT_BASH_PATH` 环境变量，优先使用。
2. `%LOCALAPPDATA%\hermes\git\usr\bin\bash.exe`（安装程序管理的 PortableGit）。
3. `%LOCALAPPDATA%\hermes\git\bin\bash.exe`（旧版 Git-for-Windows 布局）。
4. 系统 Git-for-Windows 安装（`%ProgramFiles%\Git\bin\bash.exe` 等）。
5. MSYS2、Cygwin 或 PATH 上任意 `bash.exe` 作为最后手段。

安装程序会显式设置 `HERMES_GIT_BASH_PATH`，使新 PowerShell 会话无需重新发现。如果你想让 Hermes 使用特定的 bash——例如系统 Git Bash 或通过符号链接的 WSL bash——可以覆盖此变量。

**注意事项：** MinGit 的目录布局与完整 Git-for-Windows 安装程序不同——bash 位于 `usr\bin\bash.exe`，而非 `bin\bash.exe`。Hermes 会同时检查两个路径。如果你手动解压 MinGit zip，请确保选择**非 busybox** 变体（`MinGit-*-64-bit.zip`，而非 `MinGit-*-busybox*.zip`）——busybox 构建附带的是 `ash` 而非 `bash`，且大多数 coreutils 工具缺失。

## Windows 上的 UTF-8 控制台

Python 在 Windows 上的默认 stdio 使用控制台的活动代码页（通常是 cp1252 或 cp437）。Hermes 的横幅、斜杠命令列表、工具输出、Rich 面板和技能描述均包含 Unicode 字符。若不加干预，任何此类内容都会导致 `UnicodeEncodeError: 'charmap' codec can't encode character…` 崩溃。

修复逻辑位于 `hermes_cli/stdio.py::configure_windows_stdio()`，在每个入口点（`cli.py::main`、`hermes_cli/main.py::main`、`gateway/run.py::main`）的早期调用。它会：

1. 通过 `kernel32.SetConsoleCP` / `SetConsoleOutputCP` 将控制台代码页切换为 CP_UTF8（65001）。
2. 使用 `errors='replace'` 将 `sys.stdout` / `sys.stderr` / `sys.stdin` 重新配置为 UTF-8。
3. 通过 `setdefault` 设置 `PYTHONIOENCODING=utf-8` 和 `PYTHONUTF8=1`（用户显式设置的值优先），使子 Python 进程继承 UTF-8。
4. 如果 `EDITOR` 和 `VISUAL` 均未设置，则设置 `EDITOR=notepad`（详见下方编辑器章节）。

此函数是幂等的，在非 Windows 系统上为空操作。

**禁用方式：** 在环境中设置 `HERMES_DISABLE_WINDOWS_UTF8=1` 可回退到旧版 cp1252 stdio 路径。用于排查编码 bug；正常使用中不建议设置。

## 编辑器（`Ctrl-X Ctrl-E`、`/edit`）

在 PR #21561 之前，在 Windows 上按 `Ctrl-X Ctrl-E` 或输入 `/edit` 会静默无响应。prompt_toolkit 有一个硬编码的 POSIX 绝对路径回退列表（`/usr/bin/nano`、`/usr/bin/pico`、`/usr/bin/vi` 等），在 Windows 上永远无法解析——即使安装了完整的 Git for Windows 也不行。

Hermes 的 Windows stdio 垫片现在将 `EDITOR=notepad` 设为默认值。Notepad 随每个 Windows 安装附带，可作为阻塞式编辑器使用——`subprocess.call(["notepad", file])` 会阻塞直到窗口关闭。

**用户覆盖仍然优先**（在 setdefault 之前检查）：

| 编辑器    | PowerShell 命令                                                                    |
| --------- | ---------------------------------------------------------------------------------- |
| VS Code   | `$env:EDITOR = "code --wait"`                                                      |
| Notepad++ | `$env:EDITOR = "'C:\Program Files\Notepad++\notepad++.exe' -multiInst -nosession"` |
| Neovim    | `$env:EDITOR = "nvim"`                                                             |
| Helix     | `$env:EDITOR = "hx"`                                                               |

VS Code 的 `--wait` 标志至关重要——没有它，编辑器会立即返回，Hermes 收到的是空缓冲区。

在 PowerShell profile 中永久设置：

```powershell
# In $PROFILE
$env:EDITOR = "code --wait"
```

或在系统设置的用户环境变量中设置，使每个新 shell 都能获取。

## CLI 中用 `Ctrl+Enter` 换行

Windows Terminal 将 `Ctrl+Enter` 作为独立按键序列传递。Hermes 将其绑定为"插入换行"，使你可以在 CLI 中编写多行 prompt（提示词）而无需回退到 `Esc`-然后-`Enter`。适用于 Windows Terminal、VS Code 集成终端以及任何支持 VT 转义序列的现代 Windows 控制台宿主。

在旧版 `cmd.exe` 控制台上，`Ctrl+Enter` 会折叠为普通 `Enter`——请改用 `Esc Enter`，或升级到 Windows Terminal（免费，Windows 11 默认已安装）。

## 在 Windows 登录时运行 gateway

Windows 上的 `hermes gateway install` 使用**计划任务**，并以 Startup 文件夹作为回退——无需管理员权限。

### 安装

```powershell
hermes gateway install
```

底层发生的事情：

1. `schtasks /Create /SC ONLOGON /RL LIMITED /TN HermesGateway` — 注册一个在你登录时以标准（非提升）权限运行的任务。无 UAC 提示。
2. 如果 schtasks 被组策略阻止，则回退到在 `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup` 中写入 `start /min cmd.exe /d /c <wrapper>` 快捷方式。效果相同，稍显粗糙。
3. 通过 **`pythonw.exe`** 以分离方式生成 gateway——而非 `python.exe`。`pythonw.exe` 没有附加控制台，可免疫来自同一进程组中兄弟进程的 `CTRL_C_EVENT` 广播（这是一个真实问题，曾导致在同一进程组中 Ctrl+C 任何进程时 gateway 被杀死）。

生成时使用的标志：`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`。

### 管理

```powershell
hermes gateway status      # 合并视图：schtasks + Startup 文件夹 + 运行中的 PID
hermes gateway start       # 立即启动计划任务
hermes gateway stop        # 等效于优雅的 SIGTERM（通过 psutil 调用 TerminateProcess）
hermes gateway restart
hermes gateway uninstall   # 移除 schtasks 条目、Startup 快捷方式、pid 文件
```

`hermes gateway status` 是幂等的——调用一千次也不会意外杀死 gateway。（PR #21561 之前它会静默地这样做，原因是 `os.kill(pid, 0)` 在 C 层与 `CTRL_C_EVENT` 发生碰撞——如果你想了解来龙去脉，请参阅下方"进程管理内部机制"。）

### 为什么不用 Windows 服务？

服务需要管理员权限安装，并将 gateway 的生命周期绑定到机器启动，而非用户登录。典型的 Hermes 用户希望：登录 → gateway 可用，注销 → gateway 消失。计划任务无需提权即可实现这一点。如果你确实需要服务，可以手动使用 `nssm` 或 `sc create`——但你可能并不需要。

## 数据布局

| 路径                                  | 内容                                                            |
| ------------------------------------- | --------------------------------------------------------------- |
| `%LOCALAPPDATA%\hermes\hermes-agent\` | Git 检出 + venv。可安全执行 `Remove-Item -Recurse` 后重新安装。 |
| `%LOCALAPPDATA%\hermes\git\`          | PortableGit（仅在安装程序配置时存在）。                         |
| `%LOCALAPPDATA%\hermes\node\`         | 便携式 Node.js（仅在安装程序配置时存在）。                      |
| `%LOCALAPPDATA%\hermes\bin\`          | `hermes.cmd` 垫片，已添加到用户 PATH。                          |
| `%USERPROFILE%\.hermes\`              | 你的配置、认证、技能、会话、日志。**重装后保留。**              |

这种分离是有意为之：`%LOCALAPPDATA%\hermes` 是可丢弃的基础设施（可以删除后用一行命令恢复）。`%USERPROFILE%\.hermes` 是你的数据——配置、记忆、技能、会话历史——其结构与 Linux 安装完全相同。在机器间同步它，你的 Hermes 就随之迁移。

**覆盖 `HERMES_HOME`：** 设置该环境变量以指向不同的数据目录。与 Linux 上的用法相同。

## 浏览器工具

浏览器工具使用 `agent-browser`（一个 Node 辅助程序）驱动 Chromium。在 Windows 上：

- 安装程序通过 npm 将 `agent-browser` 添加到 PATH。
- `shutil.which("agent-browser", path=...)` 会自动找到 `.cmd` 垫片——`CreateProcessW` 无法执行无扩展名的 shebang 脚本，因此 Hermes 始终解析到 `.CMD` 包装器。不要手动调用 shebang 脚本；始终通过 `.cmd` 调用。
- Playwright Chromium 在首次运行时自动安装（`npx playwright install chromium`）。如果安装失败，`hermes doctor` 会给出修复提示。

## 在 Windows 上运行 Hermes — 实用说明

### 安装后的 PATH

安装程序通过 `[Environment]::SetEnvironmentVariable` 将 `%LOCALAPPDATA%\hermes\bin` 添加到你的**用户 PATH**。已打开的终端不会获取此更新——安装完成后请打开新的 PowerShell 窗口（或 Windows Terminal 标签页）。关闭并重新打开，不要手动执行 `$env:PATH += …`，除非你清楚自己在做什么。

验证：

```powershell
Get-Command hermes        # 应输出 C:\Users\<you>\AppData\Local\hermes\bin\hermes.cmd
hermes --version
```

### 环境变量

Hermes 同时支持 `$env:X`（进程作用域）和用户环境变量（永久，在系统属性 → 环境变量中设置）。将 API key 放在 `%USERPROFILE%\.hermes\.env` 中是标准做法——与 Linux 相同：

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

不要将密钥放在用户环境变量中，除非你明确希望系统上的每个 Windows 进程都能看到它们（通常不是你想要的）。

### Windows 专属环境变量

这些变量仅影响原生 Windows 安装：

| 变量                          | 效果                                                                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `HERMES_GIT_BASH_PATH`        | 覆盖 bash.exe 的发现逻辑。可指向任意 bash——完整 Git-for-Windows、通过符号链接的 WSL bash、MSYS2、Cygwin。安装程序会自动设置此变量。 |
| `HERMES_DISABLE_WINDOWS_UTF8` | 设为 `1` 可禁用 UTF-8 stdio 垫片，回退到区域设置代码页。用于排查编码 bug。                                                          |
| `EDITOR` / `VISUAL`           | 用于 `/edit` 和 `Ctrl-X Ctrl-E` 的编辑器。如果两者均未设置，Hermes 默认使用 `notepad`。                                             |

## 卸载

在 PowerShell 中执行：

```powershell
hermes uninstall
```

这是干净的卸载路径——移除 schtasks 条目、Startup 文件夹快捷方式、`hermes.cmd` 垫片，删除 `%LOCALAPPDATA%\hermes\hermes-agent\`，并从用户 PATH 中移除相关条目。它会保留 `%USERPROFILE%\.hermes\`（你的配置、认证、技能、会话、日志），以防你需要重新安装。

彻底清除所有内容：

```powershell
hermes uninstall
Remove-Item -Recurse -Force "$env:USERPROFILE\.hermes"
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\hermes"
```

`hermes uninstall` CLI 子命令还能处理 schtasks 条目以不同任务名注册的情况（旧版安装）——它通过安装路径而非硬编码任务名来搜索。

## 进程管理内部机制

这是背景资料——除非你在调试"它在自杀"的奇怪现象，否则可以跳过。

在 Linux 和 macOS 上，POSIX 惯用法 `os.kill(pid, 0)` 是一个无操作的权限检查："这个 PID 是否存活且我能向它发信号？"在 Windows 上，Python 的 `os.kill` 将 `sig=0` 映射到 `CTRL_C_EVENT`——两者在整数值 0 上发生碰撞——并通过 `GenerateConsoleCtrlEvent(0, pid)` 将 Ctrl+C 广播到包含目标 PID 的**整个控制台进程组**。这是 [bpo-14484](https://bugs.python.org/issue14484)，自 2012 年起一直未修复，因为修改它会破坏依赖当前行为的脚本。

后果：任何通过 `os.kill(pid, 0)` 检查"此 PID 是否存活"的代码路径，在 Windows 上都会静默地杀死目标进程。Hermes 已将所有此类位置（11 个文件中的 14 处）迁移到 `gateway.status._pid_exists()`，该函数使用 `psutil.pid_exists()`（在 Windows 上底层使用 `OpenProcess + GetExitCodeProcess`——无信号）。如果你在编写插件或补丁，请直接使用 `psutil.pid_exists()` 或 `gateway.status._pid_exists()`——永远不要用 `os.kill(pid, 0)`。

`scripts/check-windows-footguns.py` 在 CI 中强制执行此规则：任何新的 `os.kill(pid, 0)` 调用都会导致 `Windows footguns (blocking)` 检查失败，除非该行带有 `# windows-footgun: ok — <reason>` 标记。

## 常见问题

**安装后立即出现 `hermes: command not found`。**
打开新的 PowerShell 窗口。安装程序已将 `%LOCALAPPDATA%\hermes\bin` 添加到用户 PATH，但现有 shell 需要重启才能获取更新。在此期间可以运行 `& "$env:LOCALAPPDATA\hermes\bin\hermes.cmd"`。

**运行工具时出现 `WinError 193: %1 is not a valid Win32 application`。**
你触发了绕过 `.cmd` 垫片的 shebang 脚本调用。Hermes 通过 `shutil.which(cmd, path=local_bin)` 解析命令，使 PATHEXT 能识别 `.CMD`——如果你通过硬编码路径调用工具，请切换到 `.cmd` 变体（例如使用 `npx.cmd` 而非 `npx`）。

**`[scriptblock]::Create(...)` 失败，提示 `The assignment expression is not valid`。**
你下载的 `install.ps1` 携带了 UTF-8 BOM。`irm | iex` 形式会自动剥离 BOM；`[scriptblock]::Create((irm ...))` 不会。请改用简单的 `irm | iex` 形式，或手动下载脚本并通过 `[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))` 保存为不带 BOM 的纯 UTF-8。

**重启后 gateway 无法持续运行。**
运行 `hermes gateway status`——它会合并 schtasks 条目、Startup 文件夹快捷方式（如有）和运行中的 PID。如果 schtasks 已注册但未运行，组策略可能阻止了 `ONLOGON` 触发器。运行 `schtasks /Query /TN HermesGateway /V /FO LIST` 查看任务失败原因，或通过卸载后使用 `HERMES_GATEWAY_FORCE_STARTUP=1` 重新安装来回退到 Startup 文件夹路径。

**设置 `$env:EDITOR` 后 `/edit` 仍然无响应。**
你只在当前进程中设置了它；请关闭并重新打开 shell，或在系统属性 → 环境变量中以用户作用域设置。在新 PowerShell 窗口中用 `echo $env:EDITOR` 验证。

**浏览器工具启动了，但工具调用超时。**
Chromium 在首次运行时自动安装。如果安装失败（GitHub 限速、Playwright CDN 故障），运行 `hermes doctor`——它会检测缺失的 Chromium 并打印修复所需的确切 `npx playwright install chromium` 命令。

**`agent-browser` 报奇怪的 Node 版本错误。**
安装程序在 `%LOCALAPPDATA%\hermes\node` 配置了 Node 22，但你的 PATH 中可能有更靠前的旧版系统 Node 18。要么将 Hermes 的 node 目录移到 PATH 前面，要么如果你不在其他地方使用 Node，删除系统安装。

**CLI 中中文/日文/阿拉伯文字符显示为 `?`。**
UTF-8 stdio 垫片未激活。检查 `HERMES_DISABLE_WINDOWS_UTF8` 是否**未**设置（`Get-ChildItem env:HERMES_DISABLE_WINDOWS_UTF8`）。如果该变量为空但仍然看到 `?`，控制台宿主（非常旧的 `cmd.exe`）可能完全不支持 UTF-8——请切换到 Windows Terminal。

**Gateway 无法发送 Telegram 图片——"`BadRequest: payload contains invalid characters`"。**
这与 Windows 无关，但有时首先在 Windows 上暴露。通常意味着 JSON 请求体中的文件路径包含未转义的反斜杠。Telegram 应该收到 Hermes 规范化后的路径，而非原始 Windows 路径——如果你在自定义插件中看到此问题，请确保传递的是 Hermes 提供的路径，而非来自用户输入的 `str(Path(...))`。

**`git pull` 后出现"在我另一台机器上能用"的编码怪象。**
如果你在 Windows 上使用非 UTF-8 编辑器（旧版 Windows 的 Notepad、某些中文输入法）编辑了 Hermes 配置或技能文件，该文件可能带 BOM 保存。Hermes 在大多数配置读取中能容忍 `utf-8-sig`，但折叠 YAML 标量（`description: >`）内部的 BOM 会静默破坏 YAML 解析。请将文件重新保存为不带 BOM 的纯 UTF-8。

## 下一步

- **[安装](../getting-started/installation.md)** — 完整安装页面，包括 Linux/macOS/WSL2/Termux。
- **[Windows（WSL2）指南](./windows-wsl-quickstart.md)** — 如果你需要 POSIX 语义或 dashboard 终端面板。
- **[CLI 参考](../reference/cli-commands.md)** — 所有 `hermes` 子命令。
- **[FAQ](../reference/faq.md)** — 常见的非 Windows 专属问题。
- **[消息 Gateway](./messaging/index.md)** — 在 Windows 上运行 Telegram/Discord/Slack。
