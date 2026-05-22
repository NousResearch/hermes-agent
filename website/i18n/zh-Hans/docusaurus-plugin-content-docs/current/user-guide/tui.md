---
sidebar_position: 2
title: "TUI"
description: "启动 Hermes 的现代终端界面 - 更适合鼠标、支持富交互覆盖层、输入不会阻塞"
---

# TUI

TUI 是 Hermes 的现代前端 - 它依然运行在与 [经典 CLI](cli.md) 相同的 Python 运行时之上。还是同一个 agent、同一批会话、同一套斜杠命令，只是交互界面更简洁、更流畅。

如果你要交互式运行 Hermes，TUI 是更推荐的方式。

## 启动

```bash
# 启动 TUI
hermes --tui

# 恢复最近的 TUI 会话（若没有则回退到最近的经典 CLI 会话）
hermes --tui -c
hermes --tui --continue

# 按会话 ID 或标题恢复指定会话
hermes --tui -r 20260409_000000_aa11bb
hermes --tui --resume "my t0p session"

# 直接运行源码 - 跳过 prebuild 步骤（给 TUI 贡献者用）
hermes --tui --dev
```

你也可以通过环境变量开启：

```bash
export HERMES_TUI=1
hermes          # 现在会使用 TUI
hermes chat     # 同样如此
```

经典 CLI 仍然是默认可用的。所有在 [CLI 界面](cli.md) 中记载的功能 - 斜杠命令、快捷命令、技能预加载、人格、多行输入、中断等 - 在 TUI 里都完全一样。

## 为什么使用 TUI

- **首帧更快** - 横幅会在应用彻底加载完成前就先渲染出来，所以 Hermes 启动时终端不会显得卡住。
- **输入不阻塞** - 在会话准备好之前就可以先输入和排队消息。你的第一条提示会在 agent 上线的瞬间发送。
- **富交互覆盖层** - 模型选择器、会话选择器、审批和澄清提示都会以模态面板呈现，而不是嵌在正文里。
- **实时会话面板** - 工具和技能在初始化时会逐步填充显示。
- **更适合鼠标操作** - 拖动高亮时使用统一背景，而不是 SGR 反显。复制时直接用终端自带的复制手势即可。
- **备用屏渲染** - 差量更新意味着流式输出时不会闪烁，退出后滚动缓冲也不会被刷得很乱。
- **输入器辅助能力** - 长内容自动折叠粘贴，`Cmd+V` / `Ctrl+V` 先尝试文本粘贴并在必要时回退到剪贴板图片，支持 bracketed paste 安全机制，以及图片/文件路径附件规范化。

同样的 [皮肤](features/skins.md) 和 [人格](features/personality.md) 也都适用。你可以在会话中用 `/skin ares`、`/personality pirate` 随时切换，界面会实时重绘。完整的可定制键位和哪些设置对经典 CLI / TUI 生效，请参阅 [Skins & Themes](features/skins.md) - TUI 支持横幅配色、UI 颜色、提示符号/颜色、会话显示、补全菜单、选择背景、`tool_prefix` 和 `help_header`。

## 需求

- **Node.js** ≥ 20 - TUI 作为 Python CLI 启动的子进程运行。`hermes doctor` 会检查这一点。
- **TTY** - 和经典 CLI 一样，如果把 stdin 管道化或在非交互环境里运行，会回退到单次提问模式。

首次启动时，Hermes 会把 TUI 的 Node 依赖安装到 `ui-tui/node_modules`（一次性操作，只需几秒）。后续启动会很快。如果你拉取了新的 Hermes 版本，只要源码比 dist 新，TUI bundle 会自动重建。

### 外部预构建

自带预构建 bundle 的发行版（如 Nix、系统包）可以让 Hermes 指向它：

```bash
export HERMES_TUI_DIR=/path/to/prebuilt/ui-tui
hermes --tui
```

该目录必须包含 `dist/entry.js`。

## 键位

键位与 [经典 CLI](cli.md#keybindings) 完全一致。唯一的行为差异是：

- **鼠标拖动** 会用统一的选区背景高亮文本。
- **`Cmd+V` / `Ctrl+V`** 会先尝试正常文本粘贴，然后回退到 OSC52 / 原生剪贴板读取，最后在剪贴板或粘贴内容解析为图片时尝试作为图片附件。
- **`/terminal-setup`** 会为本地 VS Code / Cursor / Windsurf 安装终端绑定，便于在 macOS 上获得更好的 `Cmd+Enter` 和撤销/重做一致性。
- **斜杠自动补全** 以浮动面板形式打开，并显示描述信息，而不是内联下拉框。
- **`Ctrl+X`** - 当一条排队消息被高亮时（消息是在 agent 仍在运行时发送的），可将其从队列中删除。**`Esc`** 会取消编辑并取消高亮，但不会删除。
- **`Ctrl+G` / `Ctrl+X Ctrl+E`** - 在 `$EDITOR` 中打开当前输入缓冲，适合多行 / 长提示词编辑；保存并退出后会把内容作为提示词发回。

## 斜杠命令 {#slash-commands}

所有斜杠命令都照常可用。少数命令是 TUI 专属的 - 它们会以更丰富的形式输出，或者以覆盖层而不是内联面板渲染：

| 命令 | TUI 行为 |
|------|----------|
| `/help` | 带分类的命令覆盖层，可用方向键导航 |
| `/sessions` | 模态会话选择器 - 预览、标题、token 总数、内联恢复 |
| `/model` | 按提供商分组的模态模型选择器，带成本提示 |
| `/skin` | 实时预览 - 浏览时主题立即生效 |
| `/details` | 切换详细工具调用信息（全局或按分区） |
| `/usage` | 丰富的 token / 成本 / 上下文面板 |
| `/agents`（别名 `/tasks`） | 可观测性覆盖层 - 实时子 agent 树，支持 kill / pause 控制、按分支的成本 / token / 文件汇总、逐轮历史 |
| `/reload` | 重新读取 `~/.hermes/.env` 到正在运行的 TUI 进程，以便新添加的 API Key 无需重启即可生效 |
| `/mouse` | 运行时切换鼠标追踪开关（同时会持久化到 `config.yaml` 的 `display.mouse_tracking`） |

其余所有斜杠命令（包括已安装技能、快捷命令和人格切换）都与经典 CLI 完全一致。见 [Slash Commands Reference](/reference/slash-commands)。

## LaTeX 数学渲染

TUI 的 markdown 管线会直接渲染 LaTeX 数学公式：`$E = mc^2$` 和 `$$\frac{a}{b}$$` 会显示为 Unicode 格式的数学内容，而不是原始 TeX 源码。行内和块级公式都支持；不支持的语法会回退为显示原始 TeX，并包在代码样式中，方便复制。

这项功能默认开启，不需要任何配置。经典 CLI 仍会保留原始 TeX。

## 亮色终端检测

TUI 会自动识别亮色终端，并切换到亮色主题。检测分三层：

1. `HERMES_TUI_THEME` 环境变量 - 优先级最高。可用值：`light`、`dark`，或任意 6 位背景色十六进制（例如 `ffffff`、`1a1a2e`）。
2. `COLORFGBG` 环境变量 - xterm 系终端常用的“我的背景色是什么”提示。
3. 通过 OSC 11 探测终端背景 - 适用于现代终端（Ghostty、Warp、iTerm2、WezTerm、Kitty），即使它们没有设置 `COLORFGBG`。

如果你想永久强制使用亮色主题：

```bash
export HERMES_TUI_THEME=light
```

## 忙碌指示器样式

状态栏里的忙碌指示器是可插拔的 - 默认样式会在 agent 工作期间每 2.5 秒轮换一次 Hermes 的 kawaii 表情。你可以通过配置或 `/indicator` 斜杠命令选择其他样式：

```yaml
display:
  tui_status_indicator: kaomoji   # kaomoji | emoji | unicode | ascii
```

或者在会话里运行：`/indicator emoji`（等等）。这些样式使用匹配宽度的字形，因此旋转时状态栏其他部分不会抖动。

## 自动恢复

默认情况下，`hermes --tui` 每次启动都会开启一个新的会话。如果你希望在终端或 SSH 连接意外中断后，自动重新连接到最近的 TUI 会话，可以启用：

```bash
export HERMES_TUI_RESUME=1          # 最近的 TUI 会话
# 或：
export HERMES_TUI_RESUME=<session-id>   # 指定会话
```

取消设置该变量，或显式传入 `--resume <id>`，即可按每次启动覆盖默认行为。

## 状态行

TUI 的状态行会实时跟踪 agent 状态：

| 状态 | 含义 |
|------|------|
| `starting agent…` | 会话 ID 已就绪；工具和技能仍在上线中。你可以先输入 - 消息会排队，准备好后发送。 |
| `ready` | agent 空闲，接受输入。 |
| `thinking…` / `running…` | agent 正在推理或运行工具。 |
| `interrupted` | 当前轮次已取消；按 Enter 重新发送即可。 |
| `forging session…` / `resuming…` | 初次连接或 `--resume` 握手中。 |

状态栏的皮肤颜色和阈值与经典 CLI 共享 - 自定义方式见 [Skins](features/skins.md)。

状态行还会显示：

- **工作目录与 git 分支** - `~/projects/hermes-agent (docs/two-week-gap-sweep)`。你在侧边终端里 `git checkout` 时，分支后缀会基于 mtime 缓存更新，因此 TUI 会反映你实际活跃的分支，而不是启动时的那个分支。
- **每条提示的耗时** - 运行中显示为 `⏱ 12s/3m 45s`（实时），轮次结束后冻结为 `⏲ 32s / 3m 45s`。前一个数字是自上一条用户消息以来的时间；后一个数字是总会话时长。每次新提示都会重置。

## 配置

TUI 会遵循所有标准 Hermes 配置：`~/.hermes/config.yaml`、profiles、personality、skins、quick commands、credential pools、memory providers、工具/技能启用状态。不存在单独的 TUI 配置文件。

少数键会专门影响 TUI 外观：

```yaml
display:
  skin: default              # 任意内置或自定义 skin
  personality: helpful
  details_mode: collapsed    # hidden | collapsed | expanded - 全局折叠状态默认值
  sections:                  # 可选：按分区覆盖（任意子集）
    thinking: expanded       # 始终展开
    tools: expanded          # 始终展开
    activity: collapsed      # 重新启用 activity 面板（默认隐藏）
  mouse_tracking: true       # 如果终端和鼠标报告冲突，可关闭
```

运行时切换：

- `/details [hidden|collapsed|expanded|cycle]` - 设置全局模式
- `/details <section> [hidden|collapsed|expanded|reset]` - 覆盖单个分区
  （分区：`thinking`、`tools`、`subagents`、`activity`）

**默认可见性**

TUI 预设了一组按分区的默认值，尽量把当前轮次作为实时转录流展示，而不是一串折叠 chevron：

- `thinking` - **expanded**。模型推理会在输出时直接流式显示。
- `tools` - **expanded**。工具调用及其结果会展开渲染。
- `subagents` - 继承全局 `details_mode`（默认折叠在 chevron 下 - 只有真的发生委派时才展开）。
- `activity` - **hidden**。环境级元信息（网关提示、终端一致性提示、后台通知）对于日常使用通常是噪音。工具失败仍会直接显示在失败的工具行上；当所有面板都隐藏时，环境错误/警告会通过浮层兜底。

按分区覆盖会优先于分区默认值和全局 `details_mode`。如果你想调整布局：

- `display.sections.thinking: collapsed` - 把 thinking 再折叠回 chevron 下
- `display.sections.tools: collapsed` - 把 tool calls 再折叠回 chevron 下
- `display.sections.activity: collapsed` - 重新启用 activity 面板
- 运行时使用 `/details <section> <mode>`

在 `display.sections` 里显式设置的任何值都会覆盖默认值，因此现有配置可以原样继续工作。

## 会话

TUI 与经典 CLI 共享会话 - 两者都写入同一个 `~/.hermes/state.db`。你可以在一个界面开始会话，然后在另一个界面继续。会话选择器会显示来自两个来源的会话，并带上来源标签。

会话生命周期、搜索、压缩和导出说明请见 [Sessions](sessions.md)。

## 回到经典 CLI

直接运行 `hermes`（不带 `--tui`）仍会进入经典 CLI。若你想让某台机器默认偏向 TUI，可以在 shell 配置里设置 `HERMES_TUI=1`。想切回去时，取消该设置即可。

如果 TUI 启动失败（没有 Node、bundle 丢失、TTY 有问题），Hermes 会给出诊断信息并回退，而不是让你卡住。

## 另见

- [CLI 界面](cli.md) - 共享的斜杠命令和键位完整参考
- [会话](sessions.md) - 恢复、分支和历史
- [Skins & Themes](features/skins.md) - 横幅、状态栏和覆盖层主题化
- [语音模式](features/voice-mode.md) - 两种界面都可用
- [配置](configuration.md) - 所有配置键