---
sidebar_position: 2
title: "TUI"
description: "启动 Hermes 的现代终端 UI——支持鼠标操作、丰富的浮层面板和非阻塞输入。"
---

# TUI

TUI 是 Hermes 的现代前端——一个终端 UI（用户界面），与 [Classic CLI](cli.md) 共享同一 Python 运行时。相同的 agent、相同的会话、相同的斜杠命令；交互界面更简洁、响应更流畅。

这是以交互方式运行 Hermes 的推荐方式。

## 启动

```bash
# 启动 TUI
hermes --tui

# 恢复最近的 TUI 会话（若无则回退到最近的 classic 会话）
hermes --tui -c
hermes --tui --continue

# 通过 ID 或标题恢复指定会话
hermes --tui -r 20260409_000000_aa11bb
hermes --tui --resume "my t0p session"

# 直接运行源码——跳过预构建步骤（供 TUI 贡献者使用）
hermes --tui --dev
```

也可以通过环境变量启用：

```bash
export HERMES_TUI=1
hermes          # 现在使用 TUI
hermes chat     # 同上
```

Classic CLI 仍作为默认方式保留。[CLI 界面](cli.md)中记录的所有内容——斜杠命令、快捷命令、skill 预加载、personality、多行输入、中断——在 TUI 中均完全一致。

## 为什么选择 TUI

- **即时首帧** — banner 在应用加载完成前就已渲染，因此 Hermes 启动时终端不会出现卡顿感。
- **非阻塞输入** — 会话就绪前即可输入并排队消息。agent 上线后立即发送第一条 prompt（提示词）。
- **丰富的浮层面板** — 模型选择器、会话选择器、审批和澄清提示均以模态面板形式渲染，而非内联流程。
- **实时会话面板** — 工具和 skill 在初始化过程中逐步填充。
- **鼠标友好的选择** — 拖拽高亮时使用统一背景色，而非 SGR 反色。使用终端的常规复制手势即可复制。
- **备用屏幕渲染** — 差量更新意味着流式传输时无闪烁，退出后无滚动历史残留。
- **编辑器增强** — 长片段的内联折叠粘贴、`Cmd+V` / `Ctrl+V` 文本粘贴（带剪贴板图片回退）、括号粘贴安全保护，以及图片/文件路径附件规范化。

同样的 [skins](features/skins.md) 和 [personalities](features/personality.md) 均适用。会话中途使用 `/skin ares`、`/personality pirate` 切换，UI 实时重绘。完整的可定制键列表及其对 classic 与 TUI 的适用范围，请参阅 [Skins & Themes](features/skins.md)——TUI 支持 banner 调色板、UI 颜色、prompt 字形/颜色、会话显示、补全菜单、选区背景色、`tool_prefix` 和 `help_header`。

### 可折叠的 banner 区块

TUI 启动 banner 将运行时信息分为四个可折叠区块，每个区块标题旁渲染 `▸` / `▾` 折叠箭头：

| 区块 | 默认状态 |
|------|---------|
| Tools | 展开 |
| Skills | 折叠 |
| System Prompt | 折叠 |
| MCP Servers | 折叠 |

点击区块标题（或其折叠箭头）的任意位置即可切换展开/折叠状态。Tools 列表默认展开，因为它是会话开始时最常查看的区块；Skills、System Prompt 和 MCP Servers 默认折叠，即使安装了大量 skill 或接入了多个 MCP server，banner 也能保持紧凑。状态仅对当前 banner 实例有效，下次启动将重置为默认值。

## 环境要求

- **Node.js** ≥ 20 — TUI 作为从 Python CLI 启动的子进程运行。`hermes doctor` 会验证此项。
- **TTY** — 与 classic CLI 一样，通过管道传入 stdin 或在非交互式环境中运行时，将回退到单次查询模式。

首次启动时，Hermes 会将 TUI 的 Node 依赖安装到 `ui-tui/node_modules`（一次性操作，耗时数秒）。后续启动速度很快。拉取新版 Hermes 后，若源文件比 dist 更新，TUI bundle 将自动重新构建。

### 外部预构建

发行版若附带预构建 bundle（如 Nix、系统包），可将 Hermes 指向该 bundle：

```bash
export HERMES_TUI_DIR=/path/to/prebuilt/ui-tui
hermes --tui
```

该目录必须包含 `dist/entry.js`。

## 快捷键

快捷键与 [Classic CLI](cli.md#keybindings) 完全一致。仅有以下行为差异：

- **鼠标拖拽** — 以统一选区背景色高亮文本。
- **`Cmd+V` / `Ctrl+V`** — 优先尝试普通文本粘贴，然后回退到 OSC52/原生剪贴板读取，最后在剪贴板或粘贴内容解析为图片时进行图片附件操作。
- **`/terminal-setup`** — 安装本地 VS Code / Cursor / Windsurf 终端绑定，以在 macOS 上获得更好的 `Cmd+Enter` 和撤销/重做一致性。
- **斜杠自动补全** — 以带描述的浮动面板形式展开，而非内联下拉菜单。
- **`Ctrl+X`** — 当排队消息被高亮（在 agent 仍在运行时发送的消息）时，从队列中删除该消息。**`Esc`** 取消编辑并取消高亮，但不删除。
- **`Ctrl+G` / `Ctrl+X Ctrl+E`** — 在 `$EDITOR` 中打开当前输入缓冲区，用于多行/长 prompt 编写；保存并退出后，内容将作为 prompt 发送回来。

## 斜杠命令

所有斜杠命令均可正常使用。部分命令由 TUI 独有——它们会产生更丰富的输出或以浮层而非内联面板形式渲染：

| 命令 | TUI 行为 |
|------|---------|
| `/help` | 带分类命令的浮层，可用方向键导航 |
| `/sessions` | 模态会话选择器——预览、标题、token 总量、内联恢复 |
| `/model` | 按提供商分组的模态模型选择器，带费用提示 |
| `/skin` | 实时预览——浏览时主题变更即时生效 |
| `/details` | 切换详细工具调用详情（全局或按区块） |
| `/usage` | 丰富的 token / 费用 / 上下文面板 |
| `/agents`（别名 `/tasks`） | 可观测性浮层——带终止/暂停控制的实时子 agent 树、按分支的费用/token/文件汇总、逐轮历史记录 |
| `/reload` | 将 `~/.hermes/.env` 重新读入运行中的 TUI 进程，使新添加的 API 密钥无需重启即可生效 |
| `/mouse [on\|off\|toggle\|wheel\|buttons\|all]` | 在运行时选择鼠标跟踪预设（同时持久化到 `config.yaml` 的 `display.mouse_tracking`）。`wheel`（1000+1006）保留滚轮滚动而不产生悬停事件，避免在 tmux 中向 prompt 行发送"No image in clipboard"垃圾信息；`buttons` 添加 1002 以支持终端侧拖拽选择；`all` 是带悬停 UI 的默认值。 |

其他所有斜杠命令（包括已安装的 skill、快捷命令和 personality 切换）与 classic CLI 完全一致。请参阅[斜杠命令参考](../reference/slash-commands.md)。

## LaTeX 数学渲染

TUI 的 Markdown 渲染管线支持内联 LaTeX 数学：`$E = mc^2$` 和 `$$\frac{a}{b}$$` 渲染为 Unicode 格式的数学表达式，而非原始 TeX 源码。支持内联和块级数学；不支持的语法将回退为显示包裹在代码 span 中的原始 TeX，以保持可复制性。

此功能始终开启，无需配置。Classic CLI 保留原始 TeX。

## 浅色终端检测

TUI 自动检测浅色终端并相应切换到浅色主题。检测分三层进行：

1. `HERMES_TUI_THEME` 环境变量——最高优先级。可选值：`light`、`dark`，或原始 6 位背景十六进制色值（如 `ffffff`、`1a1a2e`）。
2. `COLORFGBG` 环境变量——xterm 衍生终端使用的经典"背景色查询"提示。
3. 通过 OSC 11 探测终端背景——适用于不设置 `COLORFGBG` 的现代终端（Ghostty、Warp、iTerm2、WezTerm、Kitty）。

若要无论终端如何都永久使用浅色主题：

```bash
export HERMES_TUI_THEME=light
```

## 忙碌指示器样式

状态栏忙碌指示器可插拔——默认显示不带前导字形的 `working` 标签。通过配置或 `/indicator` 斜杠命令选择不同样式：

```yaml
display:
  tui_status_indicator: plain   # plain | kaomoji | emoji | unicode | ascii
```

或在会话中：`/indicator emoji`（等）。各样式附带匹配的字形宽度，轮换时状态栏其余部分不会抖动。

## 自动恢复

默认情况下，`hermes --tui` 每次启动都会开启新会话。若要自动重新连接到最近的 TUI 会话（在终端或 SSH 连接意外断开时很有用），可选择启用：

```bash
export HERMES_TUI_RESUME=1          # 最近的 TUI 会话
# 或：
export HERMES_TUI_RESUME=<session-id>   # 指定会话
```

取消设置该变量，或在每次启动时显式传入 `--resume <id>` 以覆盖。

## 状态栏

TUI 的状态栏实时跟踪 agent 状态：

| 状态 | 含义 |
|------|------|
| `starting agent…` | 会话 ID 已激活；工具和 skill 仍在上线中。可以输入——消息将排队，就绪后发送。 |
| `ready` | Agent 空闲，等待输入。 |
| `thinking…` / `running…` | Agent 正在推理或运行工具。 |
| `interrupted` | 当前轮次已取消；按 Enter 重新发送。 |
| `forging session…` / `resuming…` | 初始连接或 `--resume` 握手中。 |

各 skin 的状态栏颜色和阈值与 classic CLI 共享——请参阅 [Skins](features/skins.md) 了解自定义方式。

状态栏还显示：

- **工作目录及 git 分支** — `~/projects/hermes-agent (docs/two-week-gap-sweep)`。在旁边的终端执行 `git checkout` 时，分支后缀会更新（mtime 缓存），TUI 反映的是实际活跃分支，而非启动时的分支。
- **每条 prompt 的耗时** — 轮次运行时显示 `⏱ 12s/3m 45s`（实时），轮次完成后冻结为 `⏲ 32s / 3m 45s`。第一个数字是自上次用户消息以来的时间；第二个是会话总时长。每次新 prompt 时重置。
- **`🗜️ N`** — 当前会话被自动压缩的次数。首次压缩触发后显示。
- **`▶ N`** — 当前会话中正在运行的 `/background` 任务数量。至少有一个任务在执行时显示。
- **`⚠ YOLO`** — 每当 YOLO 模式开启时（`hermes --yolo`、`/yolo` 或 `HERMES_YOLO_MODE=1`）显示的可见警告。同一徽章也出现在启动 banner 中，确保你不会在未注意到的情况下启动自动审批会话。

## 配置

TUI 遵循所有标准 Hermes 配置：`~/.hermes/config.yaml`、profile、personality、skin、快捷命令、凭证池、内存提供商、工具/skill 启用状态。不存在 TUI 专属配置文件。

少数键专门用于调整 TUI 界面：

```yaml
display:
  skin: default              # 任意内置或自定义 skin
  personality: helpful
  details_mode: collapsed    # hidden | collapsed | expanded — 全局折叠面板默认值
  sections:                  # 可选：按区块覆盖（任意子集）
    thinking: expanded       # 始终展开
    tools: expanded          # 始终展开
    activity: collapsed      # 重新启用 activity 面板（默认隐藏）
  mouse_tracking: all        # off | wheel | buttons | all（或 true/false 以向后兼容）
                             #   wheel   — 1000+1006（滚轮+点击；无拖拽，无悬停——
                             #             在 tmux 内推荐使用，可消除悬停事件导致的
                             #             prompt 行"No image in clipboard"垃圾信息）
                             #   buttons — 添加 1002 以支持终端侧拖拽选择
                             #   all     — 添加 1003 以支持悬停（滚动条悬停翻页、
                             #             链接 mouseenter 等）
```

运行时切换：

- `/details [hidden|collapsed|expanded|cycle]` — 设置全局模式
- `/details <section> [hidden|collapsed|expanded|reset]` — 覆盖单个区块
  （区块：`thinking`、`tools`、`subagents`、`activity`）

**默认可见性**

TUI 附带有主见的按区块默认值，将轮次以实时转录形式流式展示，而非一堆折叠箭头：

- `thinking` — **展开**。推理过程随模型输出内联流式显示。
- `tools` — **展开**。工具调用及其结果以展开状态渲染。
- `subagents` — 沿用全局 `details_mode`（默认折叠在箭头下——在实际发生委托之前保持安静）。
- `activity` — **隐藏**。环境元信息（gateway 提示、终端一致性提醒、后台通知）对日常使用来说是噪音。工具失败仍会在失败的工具行内联渲染；当所有面板均隐藏时，环境错误/警告通过浮动警告兜底显示。

按区块覆盖优先于区块默认值和全局 `details_mode`。调整布局的方式：

- `display.sections.thinking: collapsed` — 将 thinking 折叠到箭头下
- `display.sections.tools: collapsed` — 将工具调用折叠到箭头下
- `display.sections.activity: collapsed` — 重新启用 activity 面板
- 运行时使用 `/details <section> <mode>`

在 `display.sections` 中显式设置的内容优先于默认值，因此现有配置保持不变。

## 会话

会话在 TUI 和 classic CLI 之间共享——两者均写入同一个 `~/.hermes/state.db`。可以在一个界面开始会话，在另一个界面恢复。会话选择器显示来自两个来源的会话，并带有来源标签。

会话生命周期、搜索、压缩和导出，请参阅[会话](sessions.md)。

## TUI 如何与其 gateway 通信

默认情况下，TUI 会在进程内启动自己的 gateway，因此每个 TUI 实例是自包含的——无需任何配置。

你可能会在代码或日志中看到 `HERMES_TUI_GATEWAY_URL` 环境变量。它是 **Web 仪表板的内部接线细节**，并非面向用户的远程连接开关。当你打开仪表板的 "Chat" 标签页（`hermes dashboard` → `/chat`）时，仪表板的 Web 服务器会派生一个内嵌的 TUI 子进程，并注入 `HERMES_TUI_GATEWAY_URL`，让该子进程通过本地回环 WebSocket（`/api/ws`）连接到仪表板自己的进程内 `tui_gateway`。`/api/ws` 端点仅存在于仪表板服务器内部（`hermes_cli/web_server.py`），并绑定到该进程的生命周期和认证。

不存在通用的"将任意 TUI 指向任意独立 gateway 端口"的模式。特别是，OpenAI 兼容 API 服务器（`hermes gateway` / `api_server` 平台）**不**提供 `/api/ws`——它是模型后端接口（`/v1/chat/completions`、`/v1/models` 等），并刻意不暴露 TUI 的 JSON-RPC 控制通道。将 `HERMES_TUI_GATEWAY_URL` 设置为该端口将返回 404。

如果你希望多个界面共享同一组会话，请使用共享的 `~/.hermes/state.db`（参见[会话](sessions.md)）或 Web 仪表板的内嵌聊天（参见 [Web Dashboard](features/web-dashboard.md#chat)）——而不是手动设置 gateway URL。

## 回退到 Classic CLI

不带 `--tui` 启动 `hermes` 将继续使用 classic CLI。若要让某台机器默认使用 TUI，在 shell profile 中设置 `HERMES_TUI=1`。若要回退，取消设置即可。

如果 TUI 启动失败（无 Node、缺少 bundle、TTY 问题），Hermes 会打印诊断信息并回退——而不是让你陷入困境。

## 另请参阅

- [CLI 界面](cli.md) — 完整的斜杠命令和快捷键参考（共享）
- [会话](sessions.md) — 恢复、分支和历史记录
- [Skins & Themes](features/skins.md) — 自定义 banner、状态栏和浮层主题
- [语音模式](features/voice-mode.md) — 在两种界面中均可使用
- [配置](configuration.md) — 所有配置键