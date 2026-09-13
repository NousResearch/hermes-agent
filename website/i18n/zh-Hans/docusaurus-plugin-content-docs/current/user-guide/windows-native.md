---
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

**安装程序选项：**

```powershell
& ([scriptblock]::Create((irm https://hermes-agent.nousresearch.com/install.ps1))) -NonInteractive -Branch main
```

| 参数 | 用途 |
|---|---|
| `-Branch NAME` | 选择源码分支，默认 `main`。 |
| `-Commit SHA` | 在分支检出后固定到指定 commit。 |
| `-HermesHome PATH` | 选择数据目录。 |
| `-InstallDir PATH` | 选择源码目录。 |
| `-NonInteractive` | 跳过需要输入的 setup/gateway 阶段。 |
| `-IncludeDesktop` | 构建桌面应用并创建快捷方式。 |
| `-ShowResolvedPaths` | 只输出解析后的路径 JSON，不安装。 |
| `-Manifest` / `-ProtocolVersion` | 查看引导 GUI 使用的阶段协议。 |
| `-Stage NAME -Json` | 执行单个阶段并输出结果帧。 |

当前脚本不接受 `-NoVenv`、`-SkipSetup` 或 `-Tag`。

### MSIX / App Installer 和 Microsoft Store

自包含 MSIX 要求 Windows 11 22H2 或更新版本。
Windows 10 源码安装支持不代表 MSIX 支持 Windows 10。
打开 `.appinstaller` 文件，Windows 会安装签名包并记录更新源。
软件包包含 Python、Node 和基础依赖，首次启动无需克隆或编译源码。

执行别名提供 `hermes`、`hermes-agent` 和 `hermes-acp`。
用 `Get-Command hermes -All` 检查是否被其他安装覆盖。
在 Windows 的应用执行别名设置中管理这些入口。

侧载版通过桌面 Update 控件交给 App Installer 更新。
Hermes 先下载本地描述文件，再停止自己的后端、退出并等待包替换。
它不依赖 `ms-appinstaller:` URL 协议。商店版本由 Microsoft Store 更新。

`Hermes-Setup.exe` 是另一种引导安装程序，会下载并配置源码安装。
不要把它与自包含 MSIX 混为一谈。

## 源码安装程序实际做了什么

1. 使用现有 Git；缺少时下载经过验证的 Git for Windows 工具包。
2. 克隆源码分支，并应用可选的 commit pin。
3. 引导 uv，再委托 PM 准备 Python 3.14、必要工具和名为 `all` 的 Python extra。
4. 在数据目录的 `bin` 下生成启动器，并加入用户 PATH。
5. 准备配置；非交互模式跳过输入阶段。
6. 按需构建桌面应用，并写入完成标记。

PM 通过 `pm/lock.json` 管理工具版本，不使用旧的 winget/分层 pip 回退。
启动器运行工具存储中的 Python，并在导入依赖前选择完整环境。
`hermes_cli/dep_ensure.py` 将功能需求交给 PM，不再调用 `install.ps1 -Ensure`。

:::tip 在 Windows 上跳过繁琐的提供商配置
在 Windows 上，逐个配置工具 API key（Firecrawl、FAL、Browser Use、OpenAI TTS）是获得可用 agent 摩擦最大的部分。[Nous Portal](/user-guide/features/tool-gateway) 订阅通过一次 OAuth 登录即可覆盖模型**以及**所有这些工具。安装程序完成后，运行 `hermes setup --portal` 完成配置。
:::

## 功能矩阵

Windows 支持取决于功能和架构。部分可选 SDK 不支持所有 Windows 目标。

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
| Dashboard `/chat` 内嵌终端面板 | `pywinpty`/ConPTY | POSIX PTY |
| 登录时自动启动                                               | ✓（schtasks）       | ✓（systemd）       |

Dashboard 已有 Windows ConPTY 实现，依赖 `pywinpty`。SDK 缺失或损坏时终端仍可能不可用。原生 Windows ARM64 不包含 Mem0/Google Chat SDK、Faster-Whisper 或 openWakeWord。Sherpa 支持原生 Windows ARM64，且是该平台自动选择的唤醒词引擎。

## Hermes 在 Windows 上如何运行 shell 命令

Hermes 的终端工具通过 **Git Bash** 运行命令，与 Claude Code 采用相同策略。这在不重写每个工具的情况下绕过了 POSIX 与 Windows 的差异。

`pm.shell()` 先读取 PM facts 中的 Git/Bash，再检查 PATH。
当前脚本不再设置 `HERMES_GIT_BASH_PATH`。MinGit 不能替代带 Bash 的 Git for Windows。

WindowsApps 软件包中的可执行文件可能无法由包外 Python 启动，并返回 `WinError 5`。
请使用包自己的入口，或为源码环境使用常规工具安装，不要关闭系统安全控制。

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

| 路径 | 内容 |
|---|---|
| `%LOCALAPPDATA%\hermes\hermes-agent\` | 源码安装的 checkout；纯 MSIX 安装没有此目录。 |
| `%LOCALAPPDATA%\hermes\tools\` | 可写工具存储；MSIX 基础工具保留在包内。 |
| `%LOCALAPPDATA%\hermes\installs\` | 每个安装的环境选择、事务日志和 Python 代际。 |
| `%LOCALAPPDATA%\hermes\bin\` | 源码安装启动器；MSIX 使用执行别名。 |
| `%LOCALAPPDATA%\hermes\` | 用户配置、密钥、会话、插件、技能和日志。 |

这些是默认路径，`HERMES_HOME` 可以更改数据位置。
不要删除整个 `%LOCALAPPDATA%\hermes` 来修复应用，否则会丢失共享数据。

## 浏览器工具

内置浏览器后端使用 PM 管理的 `agent-browser` 和 Chromium。
Browser Use 则通过 `hermes tools` 配置自己的 CLI。
ARM64 Windows 上的 Chromium/agent-browser 可以使用 x64 模拟，这与原生 Python 不同。
详见 [浏览器自动化](./features/browser.md)。

## 在 Windows 上运行 Hermes — 实用说明

### 安装后的 PATH

安装程序通过 `[Environment]::SetEnvironmentVariable` 将 `%LOCALAPPDATA%\hermes\bin` 添加到你的**用户 PATH**。已打开的终端不会获取此更新——安装完成后请打开新的 PowerShell 窗口（或 Windows Terminal 标签页）。关闭并重新打开，不要手动执行 `$env:PATH += …`，除非你清楚自己在做什么。

验证：

```powershell
Get-Command hermes        # 应输出 C:\Users\<you>\AppData\Local\hermes\bin\hermes.exe
hermes --version
```

### 环境变量

Hermes 同时支持 `$env:X`（进程作用域）和用户环境变量（永久，在系统属性 → 环境变量中设置）。将 API key 放在所选 `HERMES_HOME` 的 `.env` 中（默认 `%LOCALAPPDATA%\hermes\.env`）——与 Linux 相同：

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

不要将密钥放在用户环境变量中，除非你明确希望系统上的每个 Windows 进程都能看到它们（通常不是你想要的）。

### Windows 专属环境变量

这些变量仅影响原生 Windows 安装：

| 变量                          | 效果                                                                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `HERMES_DISABLE_WINDOWS_UTF8` | 设为 `1` 可禁用 UTF-8 stdio 垫片，回退到区域设置代码页。用于排查编码 bug。                                                          |
| `EDITOR` / `VISUAL`           | 用于 `/edit` 和 `Ctrl-X Ctrl-E` 的编辑器。如果两者均未设置，Hermes 默认使用 `notepad`。                                             |

## 卸载

在 PowerShell 中执行：

```powershell
hermes uninstall
```

源码安装可先用 `hermes uninstall --dry-run` 查看范围。`--full` 同时删除数据，`--data` 只删除数据。MSIX/Store 应通过 Windows 设置的“已安装的应用”移除，CLI 不删除包所有的代码。

:::caution 删除用户数据
删除前先停止使用所选 `HERMES_HOME` 的全部进程，并备份数据。
通过 `hermes uninstall --dry-run` 检查范围，再选择数据删除模式。
不要为了修复一个应用或 profile 而递归删除默认数据根目录。
自定义 `HERMES_HOME` 可以位于其他位置，移除应用包也不会删除这些数据。
:::

`hermes uninstall` CLI 子命令还能处理 schtasks 条目以不同任务名注册的情况（旧版安装）——它通过安装路径而非硬编码任务名来搜索。

## 进程管理内部机制

这是背景资料——除非你在调试"它在自杀"的奇怪现象，否则可以跳过。

在 Linux 和 macOS 上，POSIX 惯用法 `os.kill(pid, 0)` 是一个无操作的权限检查："这个 PID 是否存活且我能向它发信号？"在 Windows 上，Python 的 `os.kill` 将 `sig=0` 映射到 `CTRL_C_EVENT`——两者在整数值 0 上发生碰撞——并通过 `GenerateConsoleCtrlEvent(0, pid)` 将 Ctrl+C 广播到包含目标 PID 的**整个控制台进程组**。这是 [bpo-14484](https://bugs.python.org/issue14484)，自 2012 年起一直未修复，因为修改它会破坏依赖当前行为的脚本。

后果：任何通过 `os.kill(pid, 0)` 检查"此 PID 是否存活"的代码路径，在 Windows 上都会静默地杀死目标进程。Hermes 已将所有此类位置（11 个文件中的 14 处）迁移到 `gateway.status._pid_exists()`，该函数使用 `psutil.pid_exists()`（在 Windows 上底层使用 `OpenProcess + GetExitCodeProcess`——无信号）。如果你在编写插件或补丁，请直接使用 `psutil.pid_exists()` 或 `gateway.status._pid_exists()`——永远不要用 `os.kill(pid, 0)`。

`scripts/check-windows-footguns.py` 在 CI 中强制执行此规则：任何新的 `os.kill(pid, 0)` 调用都会导致 `Windows footguns (blocking)` 检查失败，除非该行带有 `# windows-footgun: ok — <reason>` 标记。

## 常见问题

**安装后立即出现 `hermes: command not found`。**
打开新的 PowerShell 窗口。安装程序已将 `%LOCALAPPDATA%\hermes\bin` 添加到用户 PATH，但现有 shell 需要重启才能获取更新。在此期间可以运行 `& "$env:LOCALAPPDATA\hermes\bin\hermes.exe"`。

**运行工具时出现 `WinError 193: %1 is not a valid Win32 application`。**
你触发了绕过 `.cmd` 垫片的 shebang 脚本调用。Hermes 通过 `shutil.which(cmd, path=local_bin)` 解析命令，使 PATHEXT 能识别 `.CMD`——如果你通过硬编码路径调用工具，请切换到 `.cmd` 变体（例如使用 `npx.cmd` 而非 `npx`）。

**`[scriptblock]::Create(...)` 失败，提示 `The assignment expression is not valid`。**
你下载的 `install.ps1` 携带了 UTF-8 BOM。`irm | iex` 形式会自动剥离 BOM；`[scriptblock]::Create((irm ...))` 不会。请改用简单的 `irm | iex` 形式，或手动下载脚本并通过 `[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))` 保存为不带 BOM 的纯 UTF-8。

**重启后 gateway 无法持续运行。**
运行 `hermes gateway status`——它会合并 schtasks 条目、Startup 文件夹快捷方式（如有）和运行中的 PID。如果 schtasks 已注册但未运行，组策略可能阻止了 `ONLOGON` 触发器。运行 `schtasks /Query /TN HermesGateway /V /FO LIST` 查看任务失败原因，或通过卸载后使用 `HERMES_GATEWAY_FORCE_STARTUP=1` 重新安装来回退到 Startup 文件夹路径。

**设置 `$env:EDITOR` 后 `/edit` 仍然无响应。**
你只在当前进程中设置了它；请关闭并重新打开 shell，或在系统属性 → 环境变量中以用户作用域设置。在新 PowerShell 窗口中用 `echo $env:EDITOR` 验证。

**浏览器工具启动了，但工具调用超时。**
运行 `hermes doctor` 和 `hermes pm doctor`，并通过 `hermes tools` 检查所选浏览器后端。不要向签名包写入另一个 Playwright 版本。

**`agent-browser` 报奇怪的 Node 版本错误。**
运行 `hermes pm doctor` 并检查当前 Hermes 入口。PM 提供固定的 Node 版本，不要为了修复 Hermes 而删除其他程序使用的系统 Node。

**CLI 中中文/日文/阿拉伯文字符显示为 `?`。**
UTF-8 stdio 垫片未激活。检查 `HERMES_DISABLE_WINDOWS_UTF8` 是否**未**设置（`Get-ChildItem env:HERMES_DISABLE_WINDOWS_UTF8`）。如果该变量为空但仍然看到 `?`，控制台宿主（非常旧的 `cmd.exe`）可能完全不支持 UTF-8——请切换到 Windows Terminal。

**Gateway 无法发送 Telegram 图片——"`BadRequest: payload contains invalid characters`"。**
这与 Windows 无关，但有时首先在 Windows 上暴露。通常意味着 JSON 请求体中的文件路径包含未转义的反斜杠。Telegram 应该收到 Hermes 规范化后的路径，而非原始 Windows 路径——如果你在自定义插件中看到此问题，请确保传递的是 Hermes 提供的路径，而非来自用户输入的 `str(Path(...))`。

**`git pull` 后出现"在我另一台机器上能用"的编码怪象。**
如果你在 Windows 上使用非 UTF-8 编辑器（旧版 Windows 的 Notepad、某些中文输入法）编辑了 Hermes 配置或技能文件，该文件可能带 BOM 保存。Hermes 在大多数配置读取中能容忍 `utf-8-sig`，但折叠 YAML 标量（`description: >`）内部的 BOM 会静默破坏 YAML 解析。请将文件重新保存为不带 BOM 的纯 UTF-8。

## 下一步

- **[安装](../getting-started/installation.md)** — 完整安装页面，包括 Linux/macOS/WSL2。
- **[Windows（WSL2）指南](./windows-wsl-quickstart.md)** — 如果你需要 POSIX 语义或 dashboard 终端面板。
- **[CLI 参考](../reference/cli-commands.md)** — 所有 `hermes` 子命令。
- **[FAQ](../reference/faq.md)** — 常见的非 Windows 专属问题。
- **[消息 Gateway](./messaging/index.md)** — 在 Windows 上运行 Telegram/Discord/Slack。
