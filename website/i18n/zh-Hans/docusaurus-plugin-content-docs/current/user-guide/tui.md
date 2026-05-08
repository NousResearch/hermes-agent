---
sidebar_position: 2
title: "TUI"
description: "启动 Hermes 的现代化终端界面——鼠标友好、丰富叠加层和非阻塞输入。"
---

# TUI

TUI 是 Hermes 的现代化前端——一个由与 [经典 CLI](cli.md) 相同的 Python 运行时支持的终端 UI。同样的 agent、同样的会话、同样的斜杠命令；一个更干净、响应更快的界面来与它们交互。

这是推荐以交互方式运行 Hermes 的方式。

## 启动

```bash
# 启动 TUI
hermes --tui

# 恢复最新的 TUI 会话（回退到最新的经典会话）
hermes --tui -c
hermes --tui --continue

# 按 ID 或标题恢复特定会话
hermes --tui -r 20260409_000000_aa11bb
hermes --tui --resume "my t0p session"

# 直接运行源——跳过预构建步骤（供 TUI 贡献者使用）
hermes --tui --dev
```

您也可以通过环境变量启用：

```bash
export HERMES_TUI=1
hermes          # 现在使用 TUI
hermes chat     # 同样
```

经典 CLI 仍然可用作默认。 [CLI 界面](cli.md) 中记录的任何内容——斜杠命令、快速命令、skill 预加载、人格设置、多行输入、中断——在 TUI 中完全相同地工作。

## 为什么使用 TUI

- **即时第一帧** — 横幅在应用程序完成加载之前绘制，因此当 Hermes 启动时终端不会感觉冻结。
- **非阻塞输入** — 在会话准备好之前键入和排队消息。Agent 上线的瞬间发送您的第一个提示。
- **丰富的叠加层** — 模型选择器、会话选择器、审批和澄清提示都作为模态面板呈现，而不是内联流程。
- **实时会话面板** — 工具和 skills 在初始化时逐步填充。
- **鼠标友好选择** — 拖动以突出显示，使用统一背景而不是 SGR 反转。使用终端的正常复制手势复制。
- **交替屏幕渲染** — 差异更新意味着流式传输时无闪烁，退出后无滚动回溯混乱。
- **Composer 便利** — 长代码片段的内联粘贴折叠、`Cmd+V`/`Ctrl+V` 文本粘贴与剪贴板图像回退、带保护粘贴的括号粘贴，以及图像/文件路径附件规范化。

相同的 [皮肤](features/skins.md) 和 [人格](features/personality.md) 适用。在会话中切换 `/skin ares`、`/personality pirate`，UI 实时重绘。请参阅 [皮肤和主题](features/skins.md) 获取可自定义键的完整列表以及哪些适用于经典版 vs TUI — TUI 遵守横幅调色板、UI 颜色、提示符号/颜色、会话显示、补全菜单、选择背景、`tool_prefix` 和 `help_header`。

## 要求

- **Node.js** ≥ 20 — TUI 作为从 Python CLI 启动的子进程运行。`hermes doctor` 验证这一点。
- **TTY** — 与经典 CLI 一样，管道传输 stdin 或在非交互环境中运行会回退到单查询模式。

首次启动时，Hermes 会将 TUI 的 Node 依赖安装到 `ui-tui/node_modules`（一次性的，几秒钟）。后续启动很快。如果您拉取新的 Hermes 版本，当源比 dist 更新时，TUI bundle 会自动重建。

### 外部预构建

分发预构建 bundle 的发行版（Nix、系统包）可以指向 Hermes：

```bash
export HERMES_TUI_DIR=/path/to/prebuilt/ui-tui
hermes --tui
```

目录必须包含 `dist/entry.js` 和最新的 `node_modules`。

## 按键绑定

按键绑定与 [经典 CLI](cli.md#keybindings) 完全匹配。唯一的行为差异：

- **鼠标拖动** 用统一选择背景突出显示文本。
- **`Cmd+V` / `Ctrl+V`** 首先尝试正常文本粘贴，然后回退到 OSC52/原生剪贴板读取，最后在剪贴板或粘贴的有效内容解析为图像时进行图像附加。
- **`/terminal-setup`** 安装本地 VS Code / Cursor / Windsurf 终端绑定，以在 macOS 上获得更好的 `Cmd+Enter` 和撤消/重做对等。
- **斜杠自动补全** 作为带有描述的浮动面板打开，而不是内联下拉菜单。
- **`Ctrl+X`** — 当排队的消息被突出显示时（在 agent 仍在运行时发送），从队列中删除它。**`Esc`** 取消编辑并在不删除的情况下取消突出显示。
- **`Ctrl+G` / `Ctrl+X Ctrl+E`** — 在 `$EDITOR` 中打开当前输入缓冲区，用于多行/长提示组成；保存并退出将内容作为提示发送回。

## 斜杠命令

所有斜杠命令的工作方式不变。有几个是 TUI 独有的——它们产生更丰富的输出或作为叠加层呈现而不是内联面板：

| 命令 | TUI 行为 |
|---------|--------------|
| `/help` | 带分类命令的可叠加层，可通过箭头键导航 |
| `/sessions` | 模态会话选择器——预览、标题、token 总计、内联恢复 |
| `/model` | 按 provider 分组的模态模型选择器，带成本提示 |
| `/skin` | 实时预览——主题更改在您浏览时应用 |
| `/details` | 切换详细工具调用详情（全局或按部分） |
| `/usage` | 丰富的 token / 成本 / 上下文面板 |
| `/agents`（别名 `/tasks`） | 可观测性叠加层——带有 kill/pause 控件的实时子代理树，按分支成本 / token / 文件汇总，轮次历史 |
| `/reload` | 重新读取运行中的 TUI 进程中的 `~/.hermes/.env`，因此新添加的 API 密钥无需重启即可生效 |
| `/mouse` | 运行时切换鼠标跟踪开/关（也持久化到 `config.yaml` 中的 `display.mouse_tracking`） |

每个其他斜杠命令（包括已安装的 skills、快速命令和人格切换）的工作方式与经典 CLI 完全相同。请参阅 [斜杠命令参考](../reference/slash-commands.md)。

## LaTeX 数学渲染

TUI 的 markdown 管道渲染 LaTeX 数学内联：`$E = mc^2$` 和 `$$\frac{a}{b}$$` 渲染为 Unicode 格式的数学，而不是原始 TeX 源代码。内联和块数学都支持；不支持的语法会回退到显示包装在代码跨度中的字面 TeX，因此仍然可以复制。

这是始终开启的——无需配置。经典 CLI 保留原始 TeX。

## 浅色终端检测

TUI 自动检测浅色终端并相应地切换到浅色主题。检测在三个层面工作：

1. `HERMES_TUI_THEME` 环境变量 — 最高优先级。值：`light`、`dark` 或原始 6 字符背景十六进制（例如 `ffffff`、`1a1a2e`）。
2. `COLORFGBG` 环境变量 — 经典的"我的背景颜色是什么？"提示，用于 xterm 派生终端。
3. 终端背景探测通过 OSC 11 — 适用于现代终端（Ghostty、Warp、iTerm2、WezTerm、Kitty），这些终端不设置 `COLORFGBG`。

如果您希望永久使用浅色主题，无论终端如何：

```bash
export HERMES_TUI_THEME=light
```

## 忙碌指示器样式

状态栏 FaceTicker 是可插拔的——默认在 agent 工作时每 2.5 秒轮换 Hermes kawaii 面部调色板。通过配置选择不同的样式（或 `none` 以获得最小圆点）：

```yaml
display:
  busy_indicator:
    style: kawaii     # kawaii | minimal | dots | wings | none
```

样式附带匹配的字形宽度，因此状态栏的其余部分在轮换时不会抖动。

## 自动恢复

默认情况下，`hermes --tui` 每次启动都会开始一个新的会话。要自动重新附加到最新的 TUI 会话（当您的终端或 SSH 连接意外断开时很有用），请选择加入：

```bash
export HERMES_TUI_RESUME=1          # 最新的 TUI 会话
# 或：
export HERMES_TUI_RESUME=<session-id>   # 特定会话
```

取消设置变量或显式传递 `--resume <id>` 以在每次启动时覆盖。

## 状态行

TUI 的状态行实时跟踪 agent 状态：

| 状态 | 含义 |
|--------|---------|
| `starting agent…` | 会话 ID 处于活动状态；工具和 skills 仍在上线。您可以输入——消息排队并在准备好时发送。 |
| `ready` | Agent 空闲，接受输入。 |
| `thinking…` / `running…` | Agent 正在推理或运行工具。 |
| `interrupted` | 当前轮次已取消；按 Enter 再次发送。 |
| `forging session…` / `resuming…` | 初始连接或 `--resume` 握手。 |

每个皮肤的狀態列顏色和阈值与经典 CLI 共享——请参阅 [皮肤](features/skins.md) 获取自定义。

状态行还显示：

- **带 git 分支的工作目录** — `~/projects/hermes-agent (docs/two-week-gap-sweep)`。当您在侧终端中 `git checkout` 时，分支后缀会更新（mtime 缓存）因此 TUI 反映您实际的活动分支，而不是启动时的分支。
- **每个提示的经过时间** — 轮次运行时为 `⏱ 12s/3m 45s`（实时），轮次完成后冻结为 `⏲ 32s / 3m 45s`。第一个数字是自上一条用户消息以来的时间；第二个是总会话时长。每次新提示时重置。

## 配置

TUI 遵守所有标准 Hermes 配置：`~/.hermes/config.yaml`、profiles、人格、皮肤、快速命令、凭证池、内存 provider、工具/skill 启用。不存在 TUI 特定的配置文件。

少数键专门调整 TUI 界面：

```yaml
display:
  skin: default              # 任何内置或自定义皮肤
  personality: helpful
  details_mode: collapsed    # hidden | collapsed | expanded — 全局手风琴默认
  sections:                  # 可选：按部分覆盖（任何子集）
    thinking: expanded       # 始终打开
    tools: expanded          # 始终打开
    activity: collapsed      # 重新选择加入活动面板（默认隐藏）
  mouse_tracking: true       # 如果您的终端与鼠标报告冲突则禁用
```

运行时切换：

- `/details [hidden|collapsed|expanded|cycle]` — 设置全局模式
- `/details <section> [hidden|collapsed|expanded|reset]` — 覆盖一个部分
  （部分：`thinking`、`tools`、`subagents`、`activity`）

**默认可见性**

TUI 带有固执己见的按部分默认值，将轮次流式传输为实时转录而不是 chevron 墙：

- `thinking` — **展开**。推理在模型发出时内联流式传输。
- `tools` — **展开**。工具调用及其结果呈现为打开状态。
- `subagents` — 遵循全局 `details_mode`（默认折叠在 chevron 下——在委托实际发生之前保持安静）。
- `activity` — **隐藏**。环境元（gateway 提示、终端对等提示、后台通知）对大多数日常使用来说是噪音。工具失败仍然在失败工具行上内联呈现；当每个面板都隐藏时，环境错误/警告通过浮动警报后备浮出水面。

按部分覆盖优先于部分默认值和全局 `details_mode`。要重塑布局：

- `display.sections.thinking: collapsed` — 将思维放回 chevron 下
- `display.sections.tools: collapsed` — 将工具调用放回 chevron 下
- `display.sections.activity: collapsed` — 重新选择加入活动面板
- `/details <section> <mode>` 在运行时

在 `display.sections` 中明确设置的任何内容优先于默认值，因此现有配置保持工作不变。

## 会话

会话在 TUI 和经典 CLI 之间共享——两者都写入同一个 `~/.hermes/state.db`。您可以在一处开始会话，在另一处恢复。会话选择器显示来自两个来源的会话，带有源标签。

请参阅 [会话](sessions.md) 了解生命周期、搜索、压缩和导出。

## 回退到经典 CLI

启动 `hermes`（不带 `--tui`）留在经典 CLI 上。要使机器偏好 TUI，请在您的 shell 配置文件中设置 `HERMES_TUI=1`。要返回，取消设置它。

如果 TUI 启动失败（无 Node、缺少 bundle、TTY 问题），Hermes 会打印诊断并回退——而不是让您卡住。

## 另请参阅

- [CLI 界面](cli.md) — 完整的斜杠命令和按键绑定参考（共享）
- [会话](sessions.md) — 恢复、分支和历史
- [皮肤和主题](features/skins.md) — 主题化横幅、状态栏和叠加层
- [语音模式](features/voice-mode.md) — 在两个界面中都可用
- [配置](configuration.md) — 所有配置键
