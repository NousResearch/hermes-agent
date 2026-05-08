---
sidebar_position: 15
title: "Web Dashboard"
description: "基于浏览器的仪表板，用于管理配置、API 密钥、会话、日志、分析、计划任务和技能"
---

# Web Dashboard

Web Dashboard 是一个基于浏览器的 UI，用于管理 Hermes Agent 安装。你可以配置设置、管理 API 密钥和监控会话，而无需编辑 YAML 文件或运行 CLI 命令。

## 快速开始

```bash
hermes dashboard
```

这将启动一个本地 Web 服务器并在浏览器中打开 `http://127.0.0.1:9119`。仪表板完全在你的机器上运行——没有数据离开本地主机。

### 选项

| 标志 | 默认值 | 描述 |
|------|---------|-------------|
| `--port` | `9119` | Web 服务器运行的端口 |
| `--host` | `127.0.0.1` | 绑定地址 |
| `--no-open` | — | 不自动打开浏览器 |
| `--insecure` | 关闭 | 允许绑定到非本地主机地址（**危险**——会在网络上暴露 API 密钥；请配合防火墙和强认证使用） |
| `--tui` | 关闭 | 暴露浏览器内聊天选项卡（通过 PTY/WebSocket 嵌入的 `hermes --tui`）。或者设置 `HERMES_DASHBOARD_TUI=1`。 |

```bash
# 自定义端口
hermes dashboard --port 8080

# 绑定到所有接口（在共享网络上请谨慎使用）
hermes dashboard --host 0.0.0.0

# 启动但不打开浏览器
hermes dashboard --no-open
```

## 前置条件

默认的 `hermes-agent` 安装不包含 HTTP 栈或 PTY 辅助工具——这些是可选的附加组件。**Web Dashboard** 需要 FastAPI 和 Uvicorn（`web` 额外组件）。**聊天**选项卡还需要 `ptyprocess` 来通过伪终端（POSIX 上的 `pty` 额外组件）在后台生成嵌入式 TUI。使用以下命令安装两者：

```bash
pip install 'hermes-agent[web,pty]'
```

`web` 额外组件引入 FastAPI/Uvicorn；`pty` 额外组件引入 `ptyprocess`（POSIX）或 `pywinpty`（原生 Windows——注意嵌入式 TUI 本身仍需要 WSL）。`pip install hermes-agent[all]` 包含这两个额外组件，是最简单的方式如果你也想要消息/语音等功能。

当你运行 `hermes dashboard` 但没有依赖项时，它会告诉你需要安装什么。如果前端尚未构建且 `npm` 可用，它会在首次启动时自动构建。

## 页面

### 状态

登录页面显示你的安装概览：

- **Agent 版本**和发布日期
- **Gateway 状态**——运行中/已停止、PID、已连接的平台及其状态
- **活跃会话**——过去 5 分钟内活跃的会话数量
- **最近会话**——最近 20 个会话的列表，包含模型、消息数量、令牌使用量以及对话预览

状态页面每 5 秒自动刷新。

### 聊天

**聊天**选项卡直接在浏览器中嵌入完整的 Hermes TUI（与 `hermes --tui` 相同的界面）。你可以在终端 TUI 中执行的所有操作——斜杠命令、模型选择器、工具调用卡片、Markdown 流式输出、clarify/sudo/批准提示、皮肤主题——在这里都完全相同，因为仪表板运行的是真正的 TUI 二进制文件，并通过 [xterm.js](https://xtermjs.org/) 及其 WebGL 渲染器将其 ANSI 输出渲染为像素精确的单元格布局。

**工作原理：**

- `/api/pty` 打开一个使用仪表板会话令牌进行身份验证的 WebSocket
- 服务器在 POSIX 伪终端后面生成 `hermes --tui`
- 按键发送到 PTY；ANSI 输出流返回到浏览器
- xterm.js 的 WebGL 渲染器将每个单元格绘制到整数像素网格；鼠标跟踪（SGR 1006）、宽字符（Unicode 11）和制图字符都原生渲染
- 调整浏览器窗口大小会通过 `@xterm/addon-fit` 附加组件调整 TUI 大小

**恢复现有会话：**从**会话**选项卡，点击任何会话旁边的播放图标（▶）。这会跳转到 `/chat?resume=<id>` 并使用 `--resume` 启动 TUI，加载完整历史记录。

**前置条件：**

- Node.js（与 `hermes --tui` 相同的要求；TUI 包在首次启动时构建）
- `ptyprocess`——由 `pty` 额外组件安装（`pip install 'hermes-agent[web,pty]'`，或 `[all]` 两者都包含）
- POSIX 内核（Linux、macOS 或 WSL）。不支持原生 Windows Python——请使用 WSL。

关闭浏览器标签页，PTY 会在服务器上被干净地回收。重新打开会生成一个新的会话。

### 配置

`config.yaml` 的表单编辑器。所有 150+ 配置字段从 `DEFAULT_CONFIG` 自动发现并组织成选项卡类别：

- **model**——默认模型、提供商、基础 URL、推理设置
- **terminal**——后端（本地/docker/ssh/modal）、超时、shell 偏好
- **display**——皮肤、工具进度、恢复显示、微调器设置
- **agent**——最大迭代次数、gateway 超时、服务层级
- **delegation**——子代理限制、推理努力
- **memory**——提供商选择、上下文注入设置
- **approvals**——危险命令批准模式（ask/yolo/deny）
- 以及更多——config.yaml 的每个部分都有相应的表单字段

具有已知有效值的字段（terminal 后端、皮肤、批准模式等）渲染为下拉菜单。布尔值渲染为切换开关。其他都是文本输入框。

**操作：**

- **保存**——立即将更改写入 `config.yaml`
- **重置为默认值**——将所有字段恢复为其默认值（点击保存前不会保存）
- **导出**——将当前配置下载为 JSON
- **导入**——上传 JSON 配置文件以替换当前值

:::tip
配置更改在下一个 agent 会话或 gateway 重启后生效。Web Dashboard 编辑的是 `hermes config set` 和 gateway 读取的同一个 `config.yaml` 文件。
:::

### API 密钥

管理存储 API 密钥和凭据的 `.env` 文件。密钥按类别分组：

- **LLM 提供商**——OpenRouter、Anthropic、OpenAI、DeepSeek 等
- **工具 API 密钥**——Browserbase、Firecrawl、Tavily、ElevenLabs 等
- **消息平台**——Telegram、Discord、Slack 机器人令牌等
- **Agent 设置**——非密钥环境变量如 `API_SERVER_ENABLED`

每个密钥显示：
- 是否已设置（带值的脱敏预览）
- 用途描述
- 指向提供商注册/密钥页面的链接
- 用于设置或更新值的输入字段
- 删除按钮

高级/不常用的密钥默认隐藏在切换开关后面。

### 会话

浏览和检查所有 agent 会话。每行显示会话标题、源平台图标（CLI、Telegram、Discord、Slack、cron）、模型名称、消息数量、工具调用数量以及上次活跃时间。实时会话用脉冲徽章标记。

- **搜索**——使用 FTS5 对所有消息内容进行全文搜索。结果显示高亮片段，展开时自动滚动到第一条匹配消息。
- **展开**——点击会话加载其完整的消息历史。消息按角色（用户、助手、系统、工具）着色，并以带语法高亮的 Markdown 渲染。
- **工具调用**——带有工具调用的助手消息显示可折叠的块，包含函数名称和 JSON 参数。
- **删除**——使用垃圾桶图标删除会话及其消息历史。

### 日志

查看 agent、gateway 和错误日志文件，支持过滤和实时追踪。

- **文件**——在 `agent`、`errors` 和 `gateway` 日志文件之间切换
- **级别**——按日志级别过滤：全部、调试、信息、警告或错误
- **组件**——按源组件过滤：全部、gateway、agent、tools、cli 或 cron
- **行数**——选择显示多少行（50、100、200 或 500）
- **自动刷新**——切换实时追踪，每 5 秒轮询新日志行
- **颜色编码**——日志行按严重程度着色（红色表示错误，黄色表示警告，灰色表示调试）

### 分析

从会话历史计算的使用和成本分析。选择时间周期（7、30 或 90 天）查看：

- **摘要卡片**——总令牌数（输入/输出）、缓存命中率、总预估或实际成本以及总会话数和日均值
- **每日令牌图表**——堆叠条形图显示每天的输入和输出令牌使用情况，悬停时显示细分和成本的工具提示
- **每日明细表**——每天的日期、会话数、输入令牌、输出令牌、缓存命中率和成本
- **每个模型的明细**——显示每个使用过的模型、其会话数、令牌使用量和预估成本

### Cron

创建和管理按计划运行 agent 提示的计划任务。

- **创建**——填写名称（可选）、提示、cron 表达式（例如 `0 9 * * *`）和交付目标（本地、Telegram、Discord、Slack 或电子邮件）
- **任务列表**——每个任务显示其名称、提示预览、计划表达式、状态徽章（启用/暂停/错误）、交付目标、上次运行时间和下次运行时间
- **暂停/恢复**——在活动状态和暂停状态之间切换任务
- **立即触发**——在正常计划之外立即执行任务
- **删除**——永久删除计划任务

### 技能

浏览、搜索和切换技能及工具集。技能从 `~/.hermes/skills/` 加载并按类别分组。

- **搜索**——按名称、描述或类别过滤技能和工具集
- **类别过滤器**——点击类别标签缩小列表（例如 MLOps、MCP、红队、AI）
- **切换**——使用开关启用或禁用单个技能。更改在下一个会话生效。
- **工具集**——单独的部分显示内置工具集（文件操作、Web 浏览等）及其活动/非活动状态、设置要求和包含的工具列表

:::warning 安全
Web Dashboard 读取和写入你的 `.env` 文件，其中包含 API 密钥和密钥。它默认绑定到 `127.0.0.1`——只能从你的本地机器访问。如果你绑定到 `0.0.0.0`，网络上的任何人都可以查看和修改你的凭据。仪表板本身没有身份验证。
:::

## `/reload` 斜杠命令

Dashboard PR 还将 `/reload` 斜杠命令添加到交互式 CLI。在通过 Web Dashboard 更改 API 密钥后（或直接编辑 `.env`），在活动的 CLI 会话中使用 `/reload` 来获取更改而无需重启：

```
You → /reload
  Reloaded .env (3 个变量已更新)
```

这会将 `~/.hermes/.env` 重新读取到运行进程的 environment 中。当你已经通过仪表板添加了新的提供商密钥并想立即使用它时，这很有用。

## REST API

Web Dashboard 暴露了前端使用的 REST API。你也可以直接调用这些端点进行自动化：

### GET /api/status

返回 agent 版本、gateway 状态、平台状态和活跃会话数。

### GET /api/sessions

返回最近 20 个会话及其元数据（模型、令牌计数、时间戳、预览）。

### GET /api/config

以 JSON 格式返回当前 `config.yaml` 内容。

### GET /api/config/defaults

返回默认配置值。

### GET /api/config/schema

返回描述每个配置字段的模式——类型、描述、类别以及适用时的选择选项。前端使用它为每个字段呈现正确的输入小部件。

### PUT /api/config

保存新配置。Body: `{"config": {...}}`。

### GET /api/env

返回所有已知环境变量及其设置/未设置状态、脱敏值、描述和类别。

### PUT /api/env

设置环境变量。Body: `{"key": "VAR_NAME", "value": "secret"}`。

### DELETE /api/env

删除环境变量。Body: `{"key": "VAR_NAME"}`。

### GET /api/sessions/\{session_id\}

返回单个会话的元数据。

### GET /api/sessions/\{session_id\}/messages

返回会话的完整消息历史，包括工具调用和时间戳。

### GET /api/sessions/search

跨消息内容进行全文搜索。查询参数：`q`。返回匹配的会话 ID 及其高亮片段。

### DELETE /api/sessions/\{session_id\}

删除会话及其消息历史。

### GET /api/logs

返回日志行。查询参数：`file`（agent/errors/gateway）、`lines`（数量）、`level`、`component`。

### GET /api/analytics/usage

返回令牌使用量、成本和会话分析。查询参数：`days`（默认 30）。响应包含每日明细和每个模型的聚合。

### GET /api/cron/jobs

返回所有已配置的计划任务及其状态、计划和运行历史。

### POST /api/cron/jobs

创建新的计划任务。Body: `{"prompt": "...", "schedule": "0 9 * * *", "name": "...", "deliver": "local"}`。

### POST /api/cron/jobs/\{job_id\}/pause

暂停计划任务。

### POST /api/cron/jobs/\{job_id\}/resume

恢复暂停的计划任务。

### POST /api/cron/jobs/\{job_id\}/trigger

在计划之外立即触发计划任务。

### DELETE /api/cron/jobs/\{job_id\}

删除计划任务。

### GET /api/skills

返回所有技能及其名称、描述、类别和启用状态。

### PUT /api/skills/toggle

启用或禁用技能。Body: `{"name": "skill-name", "enabled": true}`。

### GET /api/tools/toolsets

返回所有工具集及其标签、描述、工具列表和活动/配置状态。

## CORS

Web 服务器将 CORS 限制为仅本地主机源：

- `http://localhost:9119` / `http://127.0.0.1:9119`（生产环境）
- `http://localhost:3000` / `http://127.0.0.1:3000`
- `http://localhost:5173` / `http://127.0.0.1:5173`（Vite 开发服务器）

如果你在自定义端口上运行服务器，该源会自动添加。

## 开发

如果你正在为 Web Dashboard 前端做贡献：

```bash
# 终端 1：启动后端 API
hermes dashboard --no-open

# 终端 2：使用 HMR 启动 Vite 开发服务器
cd web/
npm install
npm run dev
```

位于 `http://localhost:5173` 的 Vite 开发服务器将 `/api` 请求代理到 `http://127.0.0.1:9119` 的 FastAPI 后端。

前端使用 React 19、TypeScript、Tailwind CSS v4 和 shadcn/ui 风格组件构建。生产构建输出到 `hermes_cli/web_dist/`，FastAPI 服务器将其作为静态 SPA 提供服务。

## 更新时自动构建

当你运行 `hermes update` 时，如果 `npm` 可用，Web 前端会自动重建。这使仪表板与代码更新保持同步。如果未安装 `npm`，更新会跳过前端构建，`hermes dashboard` 会在首次启动时构建它。

## 主题和插件

仪表板附带六个内置主题，可以通过用户定义的主题、插件选项卡和后端 API 路由进行扩展——所有这些都是即插即用的，无需克隆仓库。

**从标题栏实时切换主题**——点击语言切换器旁边的调色板图标。选择会持久化到 `config.yaml` 下的 `dashboard.theme`，并在页面加载时恢复。

内置主题：

| 主题 | 特点 |
|-------|-----------|
| **Hermes Teal** (`default`) | 深青色 + 奶油色，系统字体，舒适的间距 |
| **Hermes Teal (Large)** (`default-large`) | 与默认相同，18px 文本和更宽敞的间距 |
| **Midnight** (`midnight`) | 深蓝紫色，Inter + JetBrains Mono |
| **Ember** (`ember`) | 暖深红色 + 青铜色，Spectral 衬线 + IBM Plex Mono |
| **Mono** (`mono`) | 灰度，IBM Plex，紧凑 |
| **Cyberpunk** (`cyberpunk`) | 黑色霓虹绿，Share Tech Mono |
| **Rosé** (`rose`) | 粉红色 + 象牙色，Fraunces 衬线，宽敞 |

要构建自己的主题、添加插件选项卡、注入 shell 槽或公开插件特定的 REST 端点，请参阅**[扩展仪表板](./extending-the-dashboard)**——完整指南涵盖：

- 主题 YAML 模式——调色板、排版、布局、资源、`componentStyles`、`colorOverrides`、`customCSS`
- 布局变体——`standard`、`cockpit`、`tiled`
- 插件清单、SDK、shell 槽、页面作用域槽（在不完全覆盖的情况下将小部件注入内置页面）、后端 FastAPI 路由
- 完整的组合主题+插件演练（Strike Freedom cockpit 演示）
- 发现、重新加载和故障排除
