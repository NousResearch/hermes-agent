---
sidebar_position: 1
title: "CLI 界面"
description: "掌握 Hermes Agent 终端界面——命令、快捷键、个性化设置等"
---

# CLI 界面

Hermes Agent 的 CLI 是一个完整的终端用户界面（TUI）——不是 Web UI。它支持多行编辑、斜杠命令自动补全、对话历史、中断重定向和流式工具输出。为常驻终端的用户打造。

:::tip
Hermes 还提供了一个现代 TUI，支持模态覆盖层、鼠标选择和非阻塞输入。使用 `hermes --tui` 启动——详见 [TUI](tui.md) 指南。
:::

## 运行 CLI

```bash
# 启动交互式会话（默认）
hermes

# 单查询模式（非交互式）
hermes chat -q "Hello"

# 使用指定模型
hermes chat --model "anthropic/claude-sonnet-4"

# 使用指定提供商
hermes chat --provider nous        # 使用 Nous Portal
hermes chat --provider openrouter  # 强制使用 OpenRouter

# 使用指定工具集
hermes chat --toolsets "web,terminal,skills"

# 启动时预加载一个或多个技能
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -q "open a draft PR"

# 恢复之前的会话
hermes --continue             # 恢复最近的 CLI 会话 (-c)
hermes --resume <session_id>  # 按 ID 恢复指定会话 (-r)

# 详细模式（调试输出）
hermes chat --verbose

# 隔离的 git worktree（用于并行运行多个代理）
hermes -w                         # 在 worktree 中交互模式
hermes -w -q "Fix issue #123"     # 在 worktree 中单查询
```

## 界面布局

<img className="docs-terminal-figure" src="/img/docs/cli-layout.svg" alt="Hermes CLI 布局的风格化预览，展示横幅、对话区域和固定输入提示。" />
<p className="docs-figure-caption">Hermes CLI 横幅、对话流和固定输入提示，以稳定的文档图形而非脆弱的文本艺术渲染。</p>

欢迎横幅会一目了然地显示你的模型、终端后端、工作目录、可用工具和已安装技能。

### 状态栏

输入区域上方有一个持久状态栏，实时更新：

```
 ⚕ claude-sonnet-4-20250514 │ 12.4K/200K │ [██████░░░░] 6% │ $0.06 │ 15m
```

| 元素 | 说明 |
|---------|-------------|
| 模型名称 | 当前模型（超过 26 字符则截断） |
| Token 计数 | 已用上下文 token / 最大上下文窗口 |
| 上下文条 | 带颜色阈值指示的视觉填充指示器 |
| 成本 | 预估会话成本（未知/零价格模型显示 `n/a`） |
| 持续时间 | 已用会话时间 |

状态栏会根据终端宽度自适应——≥ 76 列显示完整布局，52–75 列紧凑布局，低于 52 列最小布局（仅显示模型 + 持续时间）。

**上下文颜色编码：**

| 颜色 | 阈值 | 含义 |
|-------|-----------|---------|
| 绿色 | < 50% | 空间充足 |
| 黄色 | 50–80% | 逐渐变满 |
| 橙色 | 80–95% | 接近限制 |
| 红色 | ≥ 95% | 即将溢出——考虑使用 `/compress` |

使用 `/usage` 查看详细分解，包括按类别统计的成本（输入 vs 输出 token）。

### 会话恢复显示

恢复之前的会话时（`hermes -c` 或 `hermes --resume <id>`），一个"Previous Conversation"面板会出现在横幅和输入提示之间，显示对话历史的紧凑摘要。详见 [Sessions — 恢复时的对话摘要](sessions.md#conversation-recap-on-resume) 了解详情和配置。

## 快捷键 {#keybindings}

| 按键 | 操作 |
|-----|--------|
| `Enter` | 发送消息 |
| `Alt+Enter`、`Ctrl+J` 或 `Shift+Enter` | 换行（多行输入）。`Shift+Enter` 需要终端能区分它和 `Enter`——见下文。在 Windows Terminal 上，`Alt+Enter` 被终端捕获（全屏切换）；请改用 `Ctrl+Enter` 或 `Ctrl+J`。 |
| `Alt+V` | 在终端支持时从剪贴板粘贴图片 |
| `Ctrl+V` | 粘贴文本并尝试附加剪贴板图片 |
| `Ctrl+B` | 在语音模式启用时开始/停止录音（`voice.record_key`，默认：`ctrl+b`） |
| `Ctrl+G` | 在 `$EDITOR`（vim/nvim/nano/VS Code 等）中打开当前输入缓冲区。保存并退出以发送编辑后的文本作为下一个提示——非常适合长段落提示。 |
| `Ctrl+X Ctrl+E` | Emacs 风格的外部编辑器替代绑定（与 `Ctrl+G` 行为相同）。 |
| `Ctrl+C` | 中断代理（2 秒内按两次强制退出） |
| `Ctrl+D` | 退出 |
| `Ctrl+Z` | 将 Hermes 挂起到后台（仅限 Unix）。在 shell 中运行 `fg` 恢复。 |
| `Tab` | 接受自动建议（幽灵文本）或自动补全斜杠命令 |

**多行粘贴预览。** 当你粘贴多行文本块时，CLI 会回显一个紧凑的单行预览（`[pasted: 47 lines, 1,842 chars — press Enter to send]`），而不是将整个内容转储到滚动缓冲区。完整内容仍然会被发送；这只是显示优化。

**最终响应中的 Markdown 剥离。** CLI 会从*最终*的代理回复中剥离最冗长的 markdown 围栏和 `**粗体**` / `*斜体*` 包装器，使其呈现为可读的终端散文而非原始源码。代码块和列表会被保留。这不会影响网关平台或工具结果——它们保留 markdown 以供原生渲染。

## 斜杠命令

输入 `/` 查看自动补全下拉菜单。Hermes 支持大量 CLI 斜杠命令、动态技能命令和用户定义的快捷命令。

常用示例：

| 命令 | 说明 |
|---------|-------------|
| `/help` | 显示命令帮助 |
| `/model` | 显示或更改当前模型 |
| `/tools` | 列出当前可用工具 |
| `/skills browse` | 浏览技能中心和官方可选技能 |
| `/background <prompt>` | 在单独的后台会话中运行提示 |
| `/skin` | 显示或切换当前 CLI 皮肤 |
| `/voice on` | 启用 CLI 语音模式（按 `Ctrl+B` 录音） |
| `/voice tts` | 切换 Hermes 回复的语音播放 |
| `/reasoning high` | 增加推理强度 |
| `/title My Session` | 为当前会话命名 |

完整的内置 CLI 和消息列表，详见 [斜杠命令参考](/reference/slash-commands)。

关于设置、提供商、静音调节和消息/Discord 语音使用，详见 [语音模式](features/voice-mode.md)。

:::tip
命令不区分大小写——`/HELP` 与 `/help` 效果相同。已安装的技能也会自动成为斜杠命令。
:::

## 快捷命令

你可以定义自定义命令，即时运行 shell 命令而无需调用 LLM。这些在 CLI 和消息平台（Telegram、Discord 等）中都可用。

```yaml
# ~/.hermes/config.yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

然后在任何聊天中输入 `/status`、`/gpu` 或 `/restart`。更多示例详见 [配置指南](/user-guide/configuration#quick-commands)。

## 启动时预加载技能

如果你已经知道会话中需要哪些技能，可以在启动时传入：

```bash
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -s github-auth
```

Hermes 会在第一回合前将每个指定技能加载到会话提示中。同一标志在交互模式和单查询模式下都有效。

## 技能斜杠命令

`~/.hermes/skills/` 中的每个已安装技能都会自动注册为斜杠命令。技能名称即命令：

```
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor

# 仅输入技能名即可加载并让代理询问你的需求：
/excalidraw
```

## 个性化设置

设置预定义的个性来改变代理的语气：

```
/personality pirate
/personality kawaii
/personality concise
```

内置个性包括：`helpful`、`concise`、`technical`、`creative`、`teacher`、`kawaii`、`catgirl`、`pirate`、`shakespeare`、`surfer`、`noir`、`uwu`、`philosopher`、`hype`。

你也可以在 `~/.hermes/config.yaml` 中定义自定义个性：

```yaml
personalities:
  helpful: "You are a helpful, friendly AI assistant."
  kawaii: "You are a kawaii assistant! Use cute expressions..."
  pirate: "Arrr! Ye be talkin' to Captain Hermes..."
  # 添加你自己的！
```

## 多行输入

有两种方式输入多行消息：

1. **`Alt+Enter`、`Ctrl+J` 或 `Shift+Enter`** — 插入新行
2. **反斜杠续行** — 在行尾使用 `\` 继续：

```
❯ Write a function that:\
  1. Takes a list of numbers\
  2. Returns the sum
```

:::info
支持粘贴多行文本——使用上述任意换行键，或直接粘贴内容。
:::

### Shift+Enter 兼容性

大多数终端默认发送相同的字节序列来表示 `Enter` 和 `Shift+Enter`，因此应用程序无法区分它们。Hermes 仅在终端通过 [Kitty 键盘协议](https://sw.kovidgoyal.net/kitty/keyboard-protocol/) 或 xterm 的 `modifyOtherKeys` 模式发送不同序列时才能识别 `Shift+Enter`。

| 终端 | 状态 |
|---|---|
| Kitty、foot、WezTerm、Ghostty | 默认启用独立的 `Shift+Enter` |
| iTerm2（新版）、Alacritty、VS Code terminal、Warp | 在设置中启用 Kitty 协议后支持 |
| Windows Terminal Preview 1.25+ | 在设置中启用 Kitty 协议后支持 |
| macOS Terminal.app、原版 Windows Terminal（稳定版） | 不支持——`Shift+Enter` 与 `Enter` 无法区分 |

在终端无法区分它们的地方，`Alt+Enter` 和 `Ctrl+J` 仍然到处可用。**在 Windows Terminal 上，`Alt+Enter` 被终端捕获（切换全屏）且永远不会到达 Hermes——请使用 `Ctrl+Enter`（作为 `Ctrl+J` 传递）或直接用 `Ctrl+J` 换行。**

## 中断代理

你可以随时中断代理：

- 在代理工作时**输入新消息 + Enter**——它会中断并处理你的新指令
- **`Ctrl+C`** — 中断当前操作（2 秒内按两次强制退出）
- 进行中的终端命令会立即被终止（SIGTERM，1 秒后 SIGKILL）
- 中断期间输入的多条消息会合并为一个提示

### 忙碌输入模式

`display.busy_input_mode` 配置键控制当代理工作时你按 Enter 会发生什么：

| 模式 | 行为 |
|------|----------|
| `"interrupt"`（默认） | 你的消息中断当前操作并立即处理 |
| `"queue"` | 你的消息被静默排队，在代理完成后作为下一回合发送 |
| `"steer"` | 你的消息通过 `/steer` 注入当前运行，在下一个工具调用后到达代理——不中断，不开启新回合 |

```yaml
# ~/.hermes/config.yaml
display:
  busy_input_mode: "steer"   # 或 "queue" 或 "interrupt"（默认）
```

`"queue"` 模式在你想准备后续消息而不意外取消进行中的工作时很有用。`"steer"` 模式在你想中途重定向代理而不中断时很有用——例如在它还在编辑代码时说"顺便也检查一下测试"。未知值回退到 `"interrupt"`。

`"steer"` 有两个自动回退：如果代理尚未开始，或附加了图片，消息会回退到 `"queue"` 行为以确保不会丢失任何内容。

你也可以在 CLI 内部更改：

```text
/busy queue
/busy steer
/busy interrupt
/busy status
```

:::tip 首次触摸提示
你第一次在使用 Hermes 工作时按 Enter，Hermes 会打印一行提示来解释 `/busy` 旋钮（`"(tip) Your message interrupted the current run…"`）。每个安装仅触发一次——`config.yaml` 中 `onboarding.seen.busy_input_prompt` 的一个标志会锁定它。删除该键可再次看到提示。
:::

### 挂起到后台

在 Unix 系统上，按 **`Ctrl+Z`** 将 Hermes 挂起到后台——就像任何终端进程一样。Shell 会打印确认：

```
Hermes Agent has been suspended. Run `fg` to bring Hermes Agent back.
```

在 shell 中输入 `fg` 即可从你离开的地方精确恢复会话。Windows 不支持此功能。

## 工具进度显示

CLI 在代理工作时显示动画反馈：

**思考动画**（API 调用期间）：
```
  ◜ (｡•́︿•̀｡) pondering... (1.2s)
  ◠ (⊙_⊙) contemplating... (2.4s)
  ✧٩(ˊᗜˋ*)و✧ got it! (3.1s)
```

**工具执行流：**
```
  ┊ 💻 terminal `ls -la` (0.3s)
  ┊ 🔍 web_search (1.2s)
  ┊ 📄 web_extract (2.1s)
```

使用 `/verbose` 循环切换显示模式：`off → new → all → verbose`。此命令也可为消息平台启用——详见 [配置](/user-guide/configuration#display-settings)。

### 工具预览长度

`display.tool_preview_length` 配置键控制工具调用预览行中显示的最大字符数（例如文件路径、终端命令）。默认是 `0`，表示无限制——显示完整路径和命令。

```yaml
# ~/.hermes/config.yaml
display:
  tool_preview_length: 80   # 将工具预览截断为 80 字符（0 = 无限制）
```

这在窄终端或工具参数包含非常长的文件路径时很有用。

## 会话管理

### 恢复会话

退出 CLI 会话时，会打印恢复命令：

```
Resume this session with:
  hermes --resume 20260225_143052_a1b2c3

Session:        20260225_143052_a1b2c3
Duration:       12m 34s
Messages:       28 (5 user, 18 tool calls)
```

恢复选项：

```bash
hermes --continue                          # 恢复最近的 CLI 会话
hermes -c                                  # 简写形式
hermes -c "my project"                     # 按名称恢复会话（血统中最近的）
hermes --resume 20260225_143052_a1b2c3     # 按 ID 恢复指定会话
hermes --resume "refactoring auth"         # 按标题恢复
hermes -r 20260225_143052_a1b2c3           # 简写形式
```

恢复会从 SQLite 还原完整对话历史。代理能看到所有之前的消息、工具调用和响应——就像你从未离开过一样。

在聊天中使用 `/title My Session Name` 为当前会话命名，或从命令行使用 `hermes sessions rename <id> <title>`。使用 `hermes sessions list` 浏览过去的会话。

### 会话存储

CLI 会话存储在 Hermes 的 SQLite 状态数据库中，位于 `~/.hermes/state.db`。数据库保存：

- 会话元数据（ID、标题、时间戳、token 计数器）
- 消息历史
- 跨压缩/恢复会话的血统
- `session_search` 使用的全文搜索索引

一些消息适配器也会在数据库旁边保留每平台的转录文件，但 CLI 本身从 SQLite 会话存储中恢复。

### 上下文压缩

长对话在接近上下文限制时会自动总结：

```yaml
# 在 ~/.hermes/config.yaml 中
compression:
  enabled: true
  threshold: 0.50    # 默认在上下文限制的 50% 时压缩

# 总结模型在 auxiliary 下配置：
auxiliary:
  compression:
    model: ""  # 留空使用主聊天模型（默认）。或指定一个便宜快速的模型，例如 "google/gemini-3-flash-preview"。
```

压缩触发时，中间回合会被总结，而前 3 回合和后 20 回合始终保留。

## 后台会话

在单独的后台会话中运行提示，同时继续使用 CLI 进行其他工作：

```
/background Analyze the logs in /var/log and summarize any errors from today
```

Hermes 立即确认任务并将提示交还给你：

```
🔄 Background task #1 started: "Analyze the logs in /var/log and summarize..."
   Task ID: bg_143022_a1b2c3
```

### 工作原理

每个 `/background` 提示会在守护线程中生成一个**完全独立的代理会话**：

- **隔离的对话** — 后台代理不知道你当前会话的历史。它只收到你提供的提示。
- **相同配置** — 后台代理继承你的模型、提供商、工具集、推理设置和回退模型。
- **非阻塞** — 你的前台会话保持完全交互。你可以聊天、运行命令，甚至启动更多后台任务。
- **多任务** — 你可以同时运行多个后台任务。每个都获得一个编号 ID。

### 结果

后台任务完成时，结果会以面板形式出现在你的终端中：

```
╭─ ⚕ Hermes (background #1) ──────────────────────────────────╮
│ Found 3 errors in syslog from today:                         │
│ 1. OOM killer invoked at 03:22 — killed process nginx        │
│ 2. Disk I/O error on /dev/sda1 at 07:15                      │
│ 3. Failed SSH login attempts from 192.168.1.50 at 14:30      │
╰──────────────────────────────────────────────────────────────╯
```

如果任务失败，你会看到错误通知。如果配置中启用了 `display.bell_on_complete`，任务完成时终端会响铃。

### 使用场景

- **长时间研究** — 在编写代码时运行 "/background research the latest developments in quantum error correction"
- **文件处理** — 在继续对话时运行 "/background analyze all Python files in this repo and list any security issues"
- **并行调查** — 启动多个后台任务同时从不同角度探索

:::info
后台会话不会出现在你的主对话历史中。它们是独立的会话，有自己的任务 ID（例如 `bg_143022_a1b2c3`）。
:::

## 安静模式

默认情况下，CLI 以安静模式运行，它会：
- 抑制工具的详细日志
- 启用 kawaii 风格动画反馈
- 保持输出简洁友好

如需调试输出：
```bash
hermes chat --verbose
```
