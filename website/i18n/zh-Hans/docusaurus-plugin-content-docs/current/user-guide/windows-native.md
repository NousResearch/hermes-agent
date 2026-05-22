---
title: "Windows（原生）指南 - 早期测试版"
description: "早期测试版：在 Windows 10 / 11 上原生运行 Hermes Agent - 安装、功能矩阵、UTF-8 控制台、Git Bash、以计划任务方式运行网关、编辑器处理、PATH、卸载和常见坑"
sidebar_label: "Windows（原生） - 测试版"
sidebar_position: 3
---

# Windows（原生）指南 - 早期测试版

:::warning 早期测试版
原生 Windows 支持仍处于**早期测试版**。它可以安装、运行，并通过我们的 Windows footgun lint，但还没有像 Linux / macOS / WSL2 那样经过大规模实战。你仍可能遇到一些粗糙边缘 - 尤其是子进程处理、路径细节和非 ASCII 控制台输出。如果碰到了问题，请带上复现步骤在 [issues](https://github.com/NousResearch/hermes-agent/issues) 里反馈。如果你现在就想要一套更稳定的方案，请改用 [WSL2 下的 Linux/macOS 安装路径](./windows-wsl-quickstart.md)。
:::

Hermes 可以原生运行在 Windows 10 和 Windows 11 上 - 不需要 WSL、Cygwin 或 Docker。本页是深入说明：原生模式能做什么、哪些功能仍然是 WSL-only、安装程序到底做了什么，以及你可能需要调整的 Windows 专属开关。

如果你只是想安装，主页上的一行命令或者 [安装页](../getting-started/installation#windows-native-powershell--early-beta) 已经足够。只有当某些行为让你意外时，再回来读这篇深挖文档。

:::tip 想用 WSL？
如果你更喜欢真正的 POSIX 环境（比如 dashboard 的内嵌终端、`fork` 语义、Linux 风格文件 watcher 等），可以看 **[Windows（WSL2）指南](./windows-wsl-quickstart.md)**。两者可以并存：原生数据放在 `%LOCALAPPDATA%\hermes`，WSL 数据放在 `~/.hermes`。
:::

## 快速安装

打开 **PowerShell**（或 Windows Terminal）并运行：

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

不需要管理员权限。安装程序会把文件放到 `%LOCALAPPDATA%\hermes\`，并把 `hermes` 加入你的 **User PATH** - 完成后请打开一个新终端。

**安装器参数**（需要使用 scriptblock 形式才能传参）：

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1))) -NoVenv -SkipSetup -Branch main
```

| 参数 | 默认值 | 用途 |
|---|---|---|
| `-Branch` | `main` | 克隆指定分支（适合测试 PR） |
| `-NoVenv` | 关闭 | 跳过创建虚拟环境（高级用法 - 你自己管理 Python） |
| `-SkipSetup` | 关闭 | 跳过安装后的 `hermes setup` 向导 |
| `-HermesHome` | `%LOCALAPPDATA%\hermes` | 覆盖数据目录 |
| `-InstallDir` | `%LOCALAPPDATA%\hermes\hermes-agent` | 覆盖代码位置 |

## 安装器到底做了什么

按顺序，完整流程如下：

1. **引导安装 `uv`** - Astral 的快速 Python 管理器。会安装到 `%USERPROFILE%\.local\bin`。
2. **通过 `uv` 安装 Python 3.11**。不需要你系统里先装 Python。
3. **安装 Node.js 22**（优先 winget；如果没有，就在 `%LOCALAPPDATA%\hermes\node` 下解压一个便携 Node 包）。它用于浏览器工具和 WhatsApp bridge。
4. **安装便携 Git** - 如果 PATH 上已经有 `git`，安装器就用现成的；否则会下载一个精简、自包含的 **PortableGit**（大约 45 MB，来自官方 `git-for-windows` 发布包）到 `%LOCALAPPDATA%\hermes\git`。不需要管理员权限，不会改 Windows 安装注册表，也不会干扰机器上其他 Git 安装。
5. **克隆仓库** 到 `%LOCALAPPDATA%\hermes\hermes-agent`，并在里面创建虚拟环境。
6. **分层 `uv pip install`** - 先尝试 `.[all]`，如果某个 `git+https` 依赖因为 GitHub 限流失败，就逐步回退到更小的依赖集（`[messaging,dashboard,ext]` → `[messaging]` → `.`）。这样可以避免“一个依赖 flake 把你降级成裸安装”的失败模式。
7. **按 `.env` 自动安装消息平台 SDK** - 如果存在 `TELEGRAM_BOT_TOKEN` / `DISCORD_BOT_TOKEN` / `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` / `WHATSAPP_ENABLED`，就运行 `python -m ensurepip --upgrade` 以及有针对性的 `pip install`，确保每个平台的 SDK 都真的可以导入。
8. **设置 `HERMES_GIT_BASH_PATH`** 到解析出来的 `bash.exe`，这样 Hermes 在新 shell 里也能确定性地找到它。
9. **把 `%LOCALAPPDATA%\hermes\bin` 加入 User PATH** - 新开终端后就能直接使用 `hermes`。
10. **运行 `hermes setup`** - 正常的首次运行向导（模型、提供商、工具集）。如果你想跳过，可以用 `-SkipSetup`。

## 功能矩阵

除了 dashboard 的内嵌终端面板外，其余功能都可以原生在 Windows 上运行。

| 功能 | 原生 Windows | WSL2 |
|---|---|---|
| CLI（`hermes chat`、`hermes setup`、`hermes gateway` 等） | ✓ | ✓ |
| 交互式 TUI（`hermes --tui`） | ✓ | ✓ |
| 消息网关（Telegram、Discord、Slack、WhatsApp、15+ 平台） | ✓ | ✓ |
| Cron 调度器 | ✓ | ✓ |
| 浏览器工具（通过 Node 驱动 Chromium） | ✓ | ✓ |
| MCP 服务（stdio 和 HTTP） | ✓ | ✓ |
| 本地 Ollama / LM Studio / llama-server | ✓ | ✓（通过 WSL 网络） |
| Web dashboard（会话、任务、指标、配置） | ✓ | ✓ |
| Dashboard 的 `/chat` 内嵌终端面板 | ✗（需要 POSIX PTY） | ✓ |
| 登录后自动启动 | ✓（schtasks） | ✓（systemd） |

dashboard 的 `/chat` 标签页会通过 POSIX PTY（`ptyprocess`）嵌入一个真实终端。原生 Windows 没有等价的原语；Python 的 `pywinpty` / Windows ConPTY 理论上可行，但那是单独的实现 - 现在请把它看作未来工作。**dashboard 的其余部分在原生 Windows 上都可工作** - 只有那一个标签页会显示“请用 WSL2”的提示。

## Hermes 在 Windows 上如何运行 shell 命令

Hermes 的终端工具会通过 **Git Bash** 运行命令，和 Claude Code 采用的策略类似。这样不用为每个工具都重写一套 POSIX / Windows 差异处理。

`bash.exe` 的解析顺序如下：

1. 如果设置了 `HERMES_GIT_BASH_PATH`，优先使用它。
2. `%LOCALAPPDATA%\hermes\git\usr\bin\bash.exe`（安装器管理的 PortableGit）。
3. `%LOCALAPPDATA%\hermes\git\bin\bash.exe`（旧版 Git-for-Windows 布局）。
4. 系统安装的 Git-for-Windows（如 `%ProgramFiles%\Git\bin\bash.exe` 等）。
5. 最后才是 PATH 上任何能找到的 `bash.exe`（MSYS2、Cygwin 等）。

安装器会显式设置 `HERMES_GIT_BASH_PATH`，这样新的 PowerShell 会话就不需要重新发现一次。你也可以手动覆盖它，比如想让 Hermes 使用某个特定的 bash - 例如系统 Git Bash，或者通过符号链接指向的 WSL bash。

**坑点：** MinGit 的布局和完整 Git-for-Windows 安装器不同 - bash 位于 `usr\bin\bash.exe`，而不是 `bin\bash.exe`。Hermes 会同时检查这两个位置。如果你是手动解压 MinGit zip，请务必选择**非 busybox** 版本（`MinGit-*-64-bit.zip`，不要选 `MinGit-*-busybox*.zip`） - busybox 版本只带 `ash`，缺少大部分 coreutils。

## Windows 上的 UTF-8 控制台

Windows 下 Python 默认的 stdio 会使用控制台当前代码页（通常是 cp1252 或 cp437）。Hermes 的横幅、斜杠命令列表、工具流、Rich 面板和技能描述都包含 Unicode。如果不处理，任何这些内容都可能因为 `UnicodeEncodeError: 'charmap' codec can't encode character…` 而崩掉。

相关修复在 `hermes_cli/stdio.py::configure_windows_stdio()` 中，它会在每个入口点早期调用（`cli.py::main`、`hermes_cli/main.py::main`、`gateway/run.py::main`）。它会：

1. 通过 `kernel32.SetConsoleCP` / `SetConsoleOutputCP` 把控制台代码页切换为 CP_UTF8（65001）。
2. 把 `sys.stdout` / `sys.stderr` / `sys.stdin` 重新配置为 UTF-8，并设置 `errors='replace'`。
3. 设置 `PYTHONIOENCODING=utf-8` 和 `PYTHONUTF8=1`（使用 `setdefault`，因此用户显式设置的值仍然优先），让子 Python 进程也继承 UTF-8。
4. 如果 `EDITOR` 和 `VISUAL` 都没有设置，就默认把 `EDITOR=notepad`（见下文编辑器部分）。

这个处理是幂等的，在非 Windows 平台上不会做任何事。

**可选择退出：** 在环境里设置 `HERMES_DISABLE_WINDOWS_UTF8=1`，就会回退到旧的 cp1252 stdio 路径。这个选项适合用来定位编码问题，不太可能是你日常真正想要的设置。

## 编辑器（`Ctrl-X Ctrl-E`、`/edit`）

在 #21561 之前，在 Windows 上按 `Ctrl-X Ctrl-E` 或输入 `/edit` 都会悄悄失效。prompt_toolkit 有一个硬编码的 POSIX 绝对路径回退列表（`/usr/bin/nano`、`/usr/bin/pico`、`/usr/bin/vi` 等），在 Windows 上永远找不到 - 即使你装了完整的 Git for Windows 也是一样。

Hermes 的 Windows stdio shim 现在会把 `EDITOR=notepad` 作为默认值。Notepad 每台 Windows 都有，而且是阻塞式编辑器 - `subprocess.call(["notepad", file])` 会一直阻塞到窗口关闭。

**用户自己设置的值仍然优先**（它们会在 setdefault 之前检查）：

| 编辑器 | PowerShell 命令 |
|---|---|
| VS Code | `$env:EDITOR = "code --wait"` |
| Notepad++ | `$env:EDITOR = "'C:\Program Files\Notepad++\notepad++.exe' -multiInst -nosession"` |
| Neovim | `$env:EDITOR = "nvim"` |
| Helix | `$env:EDITOR = "hx"` |

VS Code 上的 `--wait` 标志非常关键 - 没有它，编辑器会立刻返回，Hermes 收到的就是一个空缓冲区。

你可以把它永久写到 PowerShell 配置里：

```powershell
# 写在 $PROFILE 里
$env:EDITOR = "code --wait"
```

或者把它设置成系统设置里的用户环境变量，这样每个新 shell 都会自动带上。

## 在 CLI 里用 `Ctrl+Enter` 换行

Windows Terminal 会把 `Ctrl+Enter` 作为独立按键序列传给程序。Hermes 会把它绑定为“插入换行”，这样你就能在 CLI 里编辑多行提示词，而不必退回到 `Esc` 再 `Enter`。Windows Terminal、VS Code 集成终端以及任何支持 VT 转义序列的现代 Windows 控制台宿主都可以工作。

在老式 `cmd.exe` 控制台里，`Ctrl+Enter` 会退化成普通 `Enter` - 这时请用 `Esc` + `Enter`，或者升级到 Windows Terminal（它免费，而且 Windows 11 默认自带）。

## 在 Windows 登录时运行网关

Windows 上的 `hermes gateway install` 使用 **计划任务**，并带有启动文件夹回退 - 不需要管理员权限。

### 安装

```powershell
hermes gateway install
```

底层会做这些事：

1. `schtasks /Create /SC ONLOGON /RL LIMITED /TN HermesGateway` - 注册一个在你登录时运行、使用标准（非提升）权限的任务。没有 UAC 弹窗。
2. 如果 schtasks 被组策略拦截，就改为在 `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup` 里写一个 `start /min cmd.exe /d /c <wrapper>` 快捷方式。效果一样，只是稍微粗糙一点。
3. 通过 **`pythonw.exe`** 启动网关 - 不是 `python.exe`。`pythonw.exe` 没有关联控制台，因此它不会受到同一控制台进程组里其他进程的 `CTRL_C_EVENT` 广播影响（这曾经是一个真实问题，会在你 Ctrl+C 其他进程时把网关一起杀掉）。

启动时使用的 flags：`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`。

### 管理

```powershell
hermes gateway status      # 合并视图：schtasks + 启动文件夹 + 正在运行的 PID
hermes gateway start       # 现在就启动计划任务
hermes gateway stop        # 类似优雅 SIGTERM（通过 psutil 调用 TerminateProcess）
hermes gateway restart
hermes gateway uninstall   # 删除 schtasks 条目、启动快捷方式和 pid 文件
```

`hermes gateway status` 是幂等的 - 就算连续调用一千次，也绝不会误杀网关。（在 #21561 之前，它会通过 `os.kill(pid, 0)` 在 C 层和 `CTRL_C_EVENT` 冲突，从而静默杀死进程 - 如果你感兴趣，后面“进程管理内部原理”会讲这个故事。）

### 为什么不用 Windows Service？

Service 需要管理员权限安装，而且它把网关生命周期绑到机器启动，而不是用户登录。典型 Hermes 用户更想要的是：登录后网关可用，注销后网关退出。计划任务正好满足这一点，而且不需要提升权限。如果你真的想要 service，可以手工用 `nssm` 或 `sc create` 做，但大概率没必要。

## 数据布局

| 路径 | 内容 |
|---|---|
| `%LOCALAPPDATA%\hermes\hermes-agent\` | Git checkout + venv。可以安全地 `Remove-Item -Recurse` 后重装。 |
| `%LOCALAPPDATA%\hermes\git\` | PortableGit（仅当安装器帮你配置时才会有）。 |
| `%LOCALAPPDATA%\hermes\node\` | Portable Node.js（仅当安装器帮你配置时才会有）。 |
| `%LOCALAPPDATA%\hermes\bin\` | `hermes.cmd` shim，会被加入 User PATH。 |
| `%USERPROFILE%\.hermes\` | 你的配置、认证、技能、会话、日志。**重装不会丢。** |

这个拆分是刻意设计的：`%LOCALAPPDATA%\hermes` 是可丢弃的基础设施（你可以直接删掉，再用一行安装命令恢复），而 `%USERPROFILE%\.hermes` 才是你的数据 - 配置、记忆、技能、会话历史 - 它的结构和 Linux 安装完全一致。你可以把它在机器之间镜像迁移，Hermes 也会跟着走。

**覆盖 `HERMES_HOME`：** 你可以设置环境变量把数据目录指向别处。行为与 Linux 完全相同。

## 浏览器工具

浏览器工具使用 `agent-browser`（一个 Node 辅助程序）来控制 Chromium。在 Windows 上：

- 安装器会通过 npm 把 `agent-browser` 放到 PATH 上。
- `shutil.which("agent-browser", path=...)` 会自动找到 `.cmd` shim - `CreateProcessW` 不能直接执行没有扩展名的 shebang，因此 Hermes 总是会解析到 `.CMD` 包装器。不要手动运行 shebang 脚本；永远通过 `.cmd` 入口。
- Playwright Chromium 会在第一次运行时自动安装（`npx playwright install chromium`）。如果安装失败，`hermes doctor` 会给你修复提示。

## 在 Windows 上运行 Hermes 的实践说明

### 安装后的 PATH

安装器会通过 `[Environment]::SetEnvironmentVariable` 把 `%LOCALAPPDATA%\hermes\bin` 加入你的 **User PATH**。现有终端不会立刻感知到 - 安装完成后请打开一个新的 PowerShell 窗口（或 Windows Terminal 标签页）。不要手工 `$env:PATH += …`，除非你知道自己在做什么。

验证方式：

```powershell
Get-Command hermes        # 应该输出 C:\Users\<you>\AppData\Local\hermes\bin\hermes.cmd
hermes --version
```

### 环境变量

Hermes 同时支持 `$env:X`（当前进程范围）和 User 环境变量（持久化，保存在系统属性 → 环境变量里）。把 API Key 放在 `%USERPROFILE%\.hermes\.env` 中是常规方式 - 和 Linux 一样：

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

不要把密钥放进 User 环境变量里，除非你真的希望整台 Windows 上的所有进程都能看到它们（大多数情况下你并不想这样）。

### Windows 专属环境变量

这些变量只影响原生 Windows 安装：

| 变量 | 效果 |
|---|---|
| `HERMES_GIT_BASH_PATH` | 覆盖 bash.exe 的发现逻辑。可以指向任何 bash - 完整 Git-for-Windows、通过 symlink 的 WSL bash、MSYS2、Cygwin 都行。安装器会自动设置这个值。 |
| `HERMES_DISABLE_WINDOWS_UTF8` | 设为 `1` 可禁用 UTF-8 stdio shim，回退到 locale 代码页。适合调试编码 bug。 |
| `EDITOR` / `VISUAL` | `/edit` 和 `Ctrl-X Ctrl-E` 用的编辑器。如果两者都没设置，Hermes 默认用 `notepad`。 |

## 卸载

在 PowerShell 中运行：

```powershell
hermes uninstall
```

这是干净卸载路径 - 会移除 schtasks 条目、启动文件夹快捷方式、`hermes.cmd` shim，删除 `%LOCALAPPDATA%\hermes\hermes-agent\`，并清理 User PATH。它会保留 `%USERPROFILE%\.hermes\`（你的配置、认证、技能、会话、日志），以便你后续重装。

如果你要把所有东西都清掉：

```powershell
hermes uninstall
Remove-Item -Recurse -Force "$env:USERPROFILE\.hermes"
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\hermes"
```

`hermes uninstall` CLI 子命令还能处理 schtasks 条目被注册成别的任务名（旧安装）的情况 - 它会按安装路径搜索，而不是只认一个硬编码任务名。

## 进程管理内部原理

这部分是背景材料 - 如果你不是在排查“它自己把自己杀了”这种怪问题，可以跳过。

在 Linux 和 macOS 上，POSIX 惯例里的 `os.kill(pid, 0)` 只是一个无操作的权限检查：意思是“这个 PID 活着吗？我能不能向它发信号？”。但在 Windows 上，Python 的 `os.kill` 会把 `sig=0` 映射到 `CTRL_C_EVENT` - 它们的整数值都等于 0 - 然后通过 `GenerateConsoleCtrlEvent(0, pid)` 广播一个 Ctrl+C 给**整个**包含目标 PID 的控制台进程组。这就是 [bpo-14484](https://bugs.python.org/issue14484)，从 2012 年起就一直开着。它不会被修，因为改动会破坏依赖当前行为的脚本。

因此，Hermes 过去所有用 `os.kill(pid, 0)` 检查 PID 是否存活的路径，在 Windows 上都会静默把目标进程杀掉。Hermes 已经把这类调用（11 个文件里共 14 处）全部迁移到 `gateway.status._pid_exists()`，后者使用 `psutil.pid_exists()`（而 `psutil` 在 Windows 上内部会用 `OpenProcess + GetExitCodeProcess` - 不走信号机制）。如果你在写插件或补丁，请直接使用 `psutil.pid_exists()` 或 `gateway.status._pid_exists()` - 不要再用 `os.kill(pid, 0)`。

`scripts/check-windows-footguns.py` 会在 CI 里强制检查这一点：任何新的 `os.kill(pid, 0)` 调用都会让 `Windows footguns (blocking)` 检查失败，除非那一行带有 `# windows-footgun: ok — <reason>` 标记。

## 常见坑

**刚安装完就出现 `hermes: command not found`。**
打开一个新的 PowerShell 窗口。安装器已经把 `%LOCALAPPDATA%\hermes\bin` 加入 User PATH，但现有 shell 需要重启才会看到它。在这之前，你可以先用 `& "$env:LOCALAPPDATA\hermes\bin\hermes.cmd"` 临时运行。

**运行某个工具时出现 `WinError 193: %1 is not a valid Win32 application`。**
你遇到了一个绕过 `.cmd` shim 的 shebang 脚本调用。Hermes 会通过 `shutil.which(cmd, path=local_bin)` 来解析命令，所以 PATHEXT 会吃到 `.CMD` - 如果你是硬编码路径在调用工具，改用 `.cmd` 版本（例如 `npx.cmd`，不要直接用 `npx`）。

**`[scriptblock]::Create(...)` 报错 `The assignment expression is not valid`.**
你下载的 `install.ps1` 带上了 UTF-8 BOM。`irm | iex` 形式会自动去掉 BOM；但 `[scriptblock]::Create((irm ...))` 不会。请改用最简单的 `irm | iex` 形式，或者手动下载脚本并用不带 BOM 的 UTF-8 保存，比如 `[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))`。

**重启后网关没有继续运行。**
先检查 `hermes gateway status` - 它会合并显示 schtasks 条目、启动文件夹快捷方式（如果有）以及当前 PID。如果 schtasks 已注册但没有运行，可能是组策略阻止了 `ONLOGON` 触发器。运行 `schtasks /Query /TN HermesGateway /V /FO LIST` 查看任务失败原因，或者通过卸载后重新安装并设置 `HERMES_GATEWAY_FORCE_STARTUP=1` 来回退到启动文件夹路径。

**设置了 `$env:EDITOR` 之后 `/edit` 还是没反应。**
你把它只设在当前进程里了；请关闭并重新打开 shell，或者把它放到系统设置里的 User 级别环境变量中。在新的 PowerShell 窗口里用 `echo $env:EDITOR` 验证一下。

**Browser 工具能启动，但工具超时。**