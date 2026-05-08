---
sidebar_position: 1
title: "CLI 界面"
description: "掌握 Hermes Agent 终端界面——命令、按键绑定、人格设置等"
---

# CLI 界面

Hermes Agent 的 CLI 是一个完整的终端用户界面（TUI）——不是 Web 界面。它具有多行编辑、斜杠命令自动补全、对话历史、中断和重定向，以及流式工具输出。为那些工作在终端中的人打造。

:::tip
Hermes 还附带了一个现代化的 TUI，具有模态叠加层、鼠标选择和非阻塞输入。使用 `hermes --tui` 启动——请参阅 [TUI](tui.md) 指南。
:::

## 运行 CLI

```bash
# 启动交互式会话（默认）
hermes

# 单查询模式（非交互式）
hermes chat -q "Hello"

# 使用特定模型
hermes chat --model "anthropic/claude-sonnet-4"

# 使用特定 provider
hermes chat --provider nous        # 使用 Nous Portal
hermes chat --provider openrouter  # 强制使用 OpenRouter

# 使用特定工具集
hermes chat --toolsets "web,terminal,skills"

# 启动时预加载一个或多个 skills
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -q "open a draft PR"

# 恢复之前的会话
hermes --continue             # 恢复最近的 CLI 会话 (-c)
hermes --resume <session_id>  # 按 ID 恢复特定会话 (-r)

# 详细模式（调试输出）
hermes chat --verbose

# 隔离的 git worktree（用于并行运行多个 agent）
hermes -w                         # 在 worktree 中交互式运行
hermes -w -q "Fix issue #123"     # 在 worktree 中单次查询
```

## 界面布局

<img className="docs-terminal-figure" src="/img/docs/cli-layout.svg" alt="Stylized preview of the Hermes CLI layout showing the banner, conversation area, and fixed input prompt." />
<p className="docs-figure-caption">Hermes CLI 横幅、对话流和固定输入提示符，渲染为稳定的文档图形而非脆弱的文字艺术。</p>

欢迎横幅一目了然地显示您的模型、终端后端、工作目录、可用的工具和已安装的 skills。

### 状态栏

一个持久的状态栏位于输入区域上方，实时更新：

```
 ⚕ claude-sonnet-4-20250514 │ 12.4K/200K │ [██████░░░░] 6% │ $0.06 │ 15m
```

| 元素 | 描述 |
|---------|-------------|
| 模型名称 | 当前模型（如果超过 26 个字符则截断） |
| Token 计数 | 已用上下文 token / 最大上下文窗口 |
| 上下文条 | 带颜色编码阈值的可视化填充指示器 |
| 成本 | 估计的会话成本（对于未知/零价格模型为 `n/a`） |
| 时长 | 已过会话时间 |

该栏适应终端宽度——≥76 列时完整布局，52–75 列时紧凑，52 列以下最小化（仅显示模型和时长）。

**上下文颜色编码：**

| 颜色 | 阈值 | 含义 |
|-------|-----------|---------|
| 绿色 | < 50% | 空间充足 |
| 黄色 | 50–80% | 正在填满 |
| 橙色 | 80–95% | 接近限制 |
| 红色 | ≥ 95% | 接近溢出——考虑使用 `/compress` |

使用 `/usage` 获取详细的费用明细，包括按类别的成本（输入 vs 输出 token）。

### 会话恢复显示

恢复之前的会话（`hermes -c` 或 `hermes --resume <id>`）时，会在横幅和输入提示之间显示一个"上一个对话"面板，显示对话历史的简要概述。请参阅 [Sessions — 恢复时的对话概述](sessions.md#conversation-recap-on-resume) 获取详情和配置。

## 按键绑定

| 按键 | 操作 |
|-----|--------|
| `Enter` | 发送消息 |
| `Alt+Enter` 或 `Ctrl+J` | 新行（多行输入） |
| `Alt+V` | 当终端支持时，从剪贴板粘贴图片 |
| `Ctrl+V` | 粘贴文本并在适当时附加剪贴板图片 |
| `Ctrl+B` | 启用语音模式时开始/停止语音录制（`voice.record_key`，默认：`ctrl+b`） |
| `Ctrl+G` | 在 `$EDITOR`（vim/nvim/nano/VS Code/等）中打开当前输入缓冲区。保存并退出后将编辑的文本作为下一个提示发送——非常适合长的、多段的提示。 |
| `Ctrl+X Ctrl+E` | 外部编辑器的 Emacs 风格备用绑定（与 `Ctrl+G` 行为相同）。 |
| `Ctrl+C` | 中断 agent（2 秒内双击强制退出） |
| `Ctrl+D` | 退出 |
| `Ctrl+Z` | 将 Hermes 挂起到后台（仅 Unix）。在 shell 中运行 `fg` 恢复。 |
| `Tab` | 接受自动建议（幽灵文本）或自动补全斜杠命令 |

**多行粘贴预览。** 当您粘贴多行块时，CLI 会回显一个紧凑的单行预览（`[pasted: 47 lines, 1,842 chars — press Enter to send]`）而不是将整个内容转储到滚动缓冲区。发送的仍是完整内容；这只是显示上的优化。

**最终响应中的 Markdown 剥离。** CLI 从*最终* agent 回复中剥离最冗长的 markdown 代码块和 `**bold**` / `*italic*` 包装器，使其呈现为可读的终端散文而非原始源代码。代码块和列表会被保留。这不会影响 gateway 平台或工具结果——它们保留 markdown 以便原生渲染。

## 斜杠命令

输入 `/` 查看自动补全下拉菜单。Hermes 支持大量 CLI 斜杠命令、动态 skill 命令和用户定义的快速命令。

常见示例：

| 命令 | 描述 |
|---------|-------------|
| `/help` | 显示命令帮助 |
| `/model` | 显示或更改当前模型 |
| `/tools` | 列出当前可用的工具 |
| `/skills browse` | 浏览 skills hub 和官方可选 skills |
| `/background <prompt>` | 在单独的后台会话中运行提示 |
| `/skin` | 显示或切换活动的 CLI 皮肤 |
| `/voice on` | 启用 CLI 语音模式（按 `Ctrl+B` 录制） |
| `/voice tts` | 切换 Hermes 回复的语音播放 |
| `/reasoning high` | 增加推理力度 |
| `/title My Session` | 为当前会话命名 |

完整的内置 CLI 和消息传递列表，请参阅 [斜杠命令参考](../reference/slash-commands.md)。

有关设置、provider、静音调优和消息传递/Discord 语音使用，请参阅 [语音模式](features/voice-mode.md)。

:::tip
命令不区分大小写——`/HELP` 与 `/help` 一样工作。已安装的 skills 也会自动成为斜杠命令。
:::

## 快速命令

您可以定义自定义命令，无需调用 LLM 即可立即运行 shell 命令。这些命令在 CLI 和消息传递平台（Telegram、Discord 等）中都可用。

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

然后在任何聊天中输入 `/status`、`/gpu` 或 `/restart`。请参阅 [配置指南](/docs/user-guide/configuration#quick-commands) 获取更多示例。

## 启动时预加载 Skills

如果您已经知道会话需要哪些 skills，请在启动时传递它们：

```bash
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -s github-auth
```

Hermes 在第一轮之前将每个命名的 skill 加载到会话提示中。同样的标志在交互模式和单查询模式下都有效。

## Skill 斜杠命令

`~/.hermes/skills/` 中的每个已安装 skill 都会自动注册为斜杠命令。skill 名称成为命令：

```
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor

# 仅 skill 名称会加载它并让 agent 询问您需要什么：
/excalidraw
```

## 人格设置

设置预定义的人格以更改 agent 的语气：

```
/personality pirate
/personality kawaii
/personality concise
```

内置人格包括：`helpful`、`concise`、`technical`、`creative`、`teacher`、`kawaii`、`catgirl`、`pirate`、`shakespeare`、`surfer`、`noir`、`uwu`、`philosopher`、`hype`。

您也可以在 `~/.hermes/config.yaml` 中定义自定义人格：

```yaml
personalities:
  helpful: "You are a helpful, friendly AI assistant."
  kawaii: "You are a kawaii assistant! Use cute expressions..."
  pirate: "Arrr! Ye be talkin' to Captain Hermes..."
  # 添加您自己的！
```

## 多行输入

有两种输入多行消息的方法：

1. **`Alt+Enter` 或 `Ctrl+J`** — 插入新行
2. **反斜杠续行** — 以 `\` 结束一行以继续：

```
❯ Write a function that:\
  1. Takes a list of numbers\
  2. Returns the sum
```

:::info
支持粘贴多行文本——使用 `Alt+Enter` 或 `Ctrl+J` 插入换行符，或直接粘贴内容。
:::

## 中断 Agent

您可以随时中断 agent：

- **在 agent 工作时输入新消息 + Enter** — 它会中断并处理您的新指示
- **`Ctrl+C`** — 中断当前操作（2 秒内双击强制退出）
- 进行中的终端命令会立即终止（SIGTERM，然后 1 秒后 SIGKILL）
- 在中断期间输入的多条消息会合并为一个提示

### 忙碌输入模式

`display.busy_input_mode` 配置键控制当您在 agent 工作时按下 Enter 时会发生什么：

| 模式 | 行为 |
|------|----------|
| `"interrupt"` (默认) | 您的消息会中断当前操作并立即处理 |
| `"queue"` | 您的消息会被静默排队，并在 agent 完成后作为下一轮发送 |
| `"steer"` | 您的消息通过 `/steer` 注入到当前运行中，在下一个工具调用后到达 agent——无需中断，无需新轮次 |

```yaml
# ~/.hermes/config.yaml
display:
  busy_input_mode: "steer"   # 或 "queue" 或 "interrupt" (默认)
```

`"queue"` 模式在您想准备后续消息而不意外取消进行中的工作时很有用。`"steer"` 模式在您想在不中断的情况下重定向 agent 完成任务时很有用——例如，当它仍在编辑代码时，"实际上，也检查一下测试"。未知值会回退到 `"interrupt"`。

`"steer"` 有两个自动回退：如果 agent 尚未开始，或如果附加了图片，消息会回退到 `"queue"` 行为以确保不会丢失任何内容。

您也可以在 CLI 中更改它：

```text
/busy queue
/busy steer
/busy interrupt
/busy status
```

:::tip 首次提示
当 Hermes 工作时，您第一次按 Enter，Hermes 会打印一行提示解释 `/busy` 旋钮（`"(tip) Your message interrupted the current run…"`）。它只在每次安装时触发一次——`config.yaml` 中 `onboarding.seen.busy_input_prompt` 下的一个标志会锁定它。删除该键可以再次看到提示。
:::

### 挂起到后台

在 Unix 系统上，按 **`Ctrl+Z`** 将 Hermes 挂起到后台——就像任何终端进程一样。shell 打印确认：

```
Hermes Agent has been suspended. Run `fg` to bring Hermes Agent back.
```

在 shell 中输入 `fg` 以精确恢复会话。这在 Windows 上不支持。

## 工具进度显示

CLI 在 agent 工作时显示动画反馈：

**思考动画**（API 调用期间）：
```
  ◜ (｡•́︿•̀｡) pondering... (1.2s)
  ◠ (⊙_⊙) contemplating... (2.4s)
  ✧٩(ˊᗜˋ*)و✧ got it! (3.1s)
```

**工具执行信息：**
```
  ┊ 💻 terminal `ls -la` (0.3s)
  ┊ 🔍 web_search (1.2s)
  ┊ 📄 web_extract (2.1s)
```

使用 `/verbose` 循环切换显示模式：`off → new → all → verbose`。此命令也可为消息传递平台启用——请参阅 [配置](/docs/user-guide/configuration#display-settings)。

### 工具预览长度

`display.tool_preview_length` 配置键控制工具调用预览行中显示的最大字符数（例如文件路径、终端命令）。默认值为 `0`，表示无限制——显示完整路径和命令。

```yaml
# ~/.hermes/config.yaml
display:
  tool_preview_length: 80   # 将工具预览截断为 80 个字符（0 = 无限制）
```

这在窄终端或工具参数包含非常长的文件路径时很有用。

## 会话管理

### 恢复会话

当您退出 CLI 会话时，会打印恢复命令：

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
hermes -c                                  # 短格式
hermes -c "my project"                     # 恢复命名会话（谱系中最新的）
hermes --resume 20260225_143052_a1b2c3     # 按 ID 恢复特定会话
hermes --resume "refactoring auth"         # 按标题恢复
hermes -r 20260225_143052_a1b2c3           # 短格式
```

恢复会从 SQLite 完整恢复对话历史。agent 可以看到所有先前的消息、工具调用和响应——就像您从未离开过一样。

在聊天中使用 `/title My Session Name` 为当前会话命名，或从命令行使用 `hermes sessions rename <id> <title>`。使用 `hermes sessions list` 浏览过去的会话。

### 会话存储

CLI 会话存储在 Hermes 的 SQLite 状态数据库中，路径为 `~/.hermes/state.db`。数据库保留：

- 会话元数据（ID、标题、时间戳、token 计数器）
- 消息历史
- 压缩/恢复会话的谱系
- `session_search` 使用的全文搜索索引

某些消息传递适配器也会在数据库旁边保留每个平台的转录文件，但 CLI 本身从 SQLite 会话存储恢复。

### 上下文压缩

当接近上下文限制时，长对话会自动摘要：

```yaml
# 在 ~/.hermes/config.yaml 中
compression:
  enabled: true
  threshold: 0.50    # 默认在上下文限制的 50% 时压缩

# 在 auxiliary 下配置的摘要模型：
auxiliary:
  compression:
    model: "google/gemini-3-flash-preview"  # 用于摘要的模型
```

当压缩触发时，中间轮次会被摘要，而前 3 轮和后 20 轮始终保留。

## 后台会话

在继续使用 CLI 进行其他工作时，在单独的后台会话中运行提示：

```
/background Analyze the logs in /var/log and summarize any errors from today
```

Hermes 立即确认任务并返回提示：

```
🔄 Background task #1 started: "Analyze the logs in /var/log and summarize..."
   Task ID: bg_143022_a1b2c3
```

### 工作原理

每个 `/background` 提示会在守护线程中生成一个**完全独立的 agent 会话**：

- **隔离对话** — 后台 agent 不了解您当前会话的历史。它只接收您提供的提示。
- **相同配置** — 后台 agent 从当前会话继承模型、provider、工具集、推理设置和回退模型。
- **非阻塞** — 您的前台会话保持完全交互。您可以聊天、运行命令，甚至启动更多后台任务。
- **多个任务** — 您可以同时运行多个后台任务。每个任务都有一个编号 ID。

### 结果

当后台任务完成时，结果会作为面板显示在您的终端中：

```
╭─ ⚕ Hermes (background #1) ──────────────────────────────────╮
│ Found 3 errors in syslog from today:                         │
│ 1. OOM killer invoked at 03:22 — killed process nginx        │
│ 2. Disk I/O error on /dev/sda1 at 07:15                      │
│ 3. Failed SSH login attempts from 192.168.1.50 at 14:30      │
╰──────────────────────────────────────────────────────────────╯
```

如果任务失败，您会看到错误通知。如果您的配置中启用了 `display.bell_on_complete`，任务完成时终端铃声会响。

### 使用场景

- **长时间运行的研究** — "/background research the latest developments in quantum error correction" 而您继续编写代码
- **文件处理** — "/background analyze all Python files in this repo and list any security issues" 而您继续对话
- **并行调查** — 启动多个后台任务同时探索不同的角度

:::info
后台会话不会出现在您的主要对话历史中。它们是独立的会话，有自己的任务 ID（例如 `bg_143022_a1b2c3`）。
:::

## 安静模式

默认情况下，CLI 以安静模式运行，这会：
- 禁止工具的详细日志
- 启用 kawaii 风格的动画反馈
- 保持输出简洁和用户友好

用于调试输出：
```bash
hermes chat --verbose
```

## 高级用法

### 配置文件路径

默认情况下，Hermes 在 `~/.hermes/` 中存储配置、会话和状态。您可以使用 `HERMES_HOME` 环境变量覆盖此位置：

```bash
HERMES_HOME=/path/to/custom/hermes hermes
```

### 环境变量配置

某些设置可以通过环境变量配置，而不是配置文件：

```bash
# 覆盖默认模型
HERMES_MODEL=anthropic/claude-sonnet-4

# 启用详细输出
HERMES_VERBOSE=1

# 自定义数据目录
HERMES_HOME=/custom/path
```

### 调试和诊断

如果遇到问题，Hermes 提供了几个诊断命令：

```bash
# 检查 Hermes 安装的健康状况
hermes doctor

# 显示当前配置
hermes config show

# 查看最近的日志
hermes logs --tail 100

# 调试模式运行
hermes --verbose chat
```

## 与其他工具集成

### Git 集成

Hermes 与 git 无缝协作：

```bash
# 在 git 仓库中启动
cd my-project
hermes

# Hermes 自动检测 .git 目录
# 并相应地设置工作目录
```

### 终端多路复用器

Hermes 与 tmux、screen 和其他终端多路复用器兼容：

```bash
# 在 tmux 中运行
tmux new -s hermes
hermes

# 或在现有会话中附加
hermes attach -t hermes
```

### SSH 远程使用

通过 SSH 在远程机器上运行 Hermes：

```bash
ssh user@remote-server
hermes
```

确保 SSH 会话支持 PTY（默认情况）。使用 `-T` 标志禁用 PTY 会限制某些功能。

## 性能提示

- **上下文管理** — 长时间对话会自动压缩。使用 `/compress` 手动触发。
- **工具选择** — 使用 `--toolsets` 限制可用工具可以减少延迟。
- **模型选择** — 较便宜的模型通常适合日常任务。
- **后台任务** — 对于不需要立即结果的任务使用 `/background`。

## 故障排除

### Hermes 无响应

1. 按 `Ctrl+C` 中断当前操作
2. 等待 agent 确认中断
3. 如果仍然无响应，按 `Ctrl+C` 两次强制退出
4. 使用 `hermes --resume <session_id>` 恢复会话

### 连接问题

如果看到 API 连接错误：
- 检查您的 API 密钥是否正确设置
- 验证网络连接
- 检查 `hermes doctor` 的连接测试

### 性能问题

如果 Hermes 运行缓慢：
- 检查上下文使用情况（状态栏中的百分比）
- 考虑使用 `/compress` 压缩上下文
- 减少会话中的消息数量

## 获取帮助

- `/help` — 在 CLI 中查看所有可用命令
- `hermes doctor` — 诊断安装问题
- 查看 [文档](/docs/) 获取详细指南
