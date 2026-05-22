---
sidebar_position: 15
title: "Web Dashboard"
description: "基于浏览器的仪表板，用于管理配置、API 密钥、会话、日志、分析、cron 任务和技能"
---

# Web Dashboard

Web Dashboard 是一个基于浏览器的 UI，用于管理你的 Hermes Agent 安装。无需编辑 YAML 文件或运行 CLI 命令，你可以通过简洁的 Web 界面配置设置、管理 API 密钥并监控会话。

## 快速开始

```bash
hermes dashboard
```

这会启动一个本地 Web 服务器，并在浏览器中打开 `http://127.0.0.1:9119`。仪表板完全在你的机器上运行 —— 没有任何数据会离开 localhost。

### 选项

| Flag | 默认值 | 说明 |
|------|---------|-------------|
| `--port` | `9119` | Web 服务器运行的端口 |
| `--host` | `127.0.0.1` | 绑定地址 |
| `--no-open` | — | 不自动打开浏览器 |
| `--insecure` | off | 允许绑定到非 localhost 的主机（**危险** —— 会在网络上暴露 API 密钥；需配合防火墙和强认证使用） |
| `--tui` | off | 暴露浏览器内的 Chat 标签页（通过 PTY/WebSocket 嵌入 `hermes --tui`）。或者设置环境变量 `HERMES_DASHBOARD_TUI=1`。 |

```bash
# 自定义端口
hermes dashboard --port 8080

# 绑定到所有接口（在共享网络上请谨慎使用）
hermes dashboard --host 0.0.0.0

# 启动时不打开浏览器
hermes dashboard --no-open
```

## 前置条件

默认的 `hermes-agent` 安装不包含 HTTP 协议栈或 PTY 辅助工具 —— 这些是可选的额外组件。**Web Dashboard** 需要 FastAPI 和 Uvicorn（`web` extra）。**Chat** 标签页还需要 `ptyprocess` 来在伪终端（`pty` extra，POSIX 系统）后启动嵌入的 TUI。同时安装两者：

```bash
pip install 'hermes-agent[web,pty]'
```

`web` extra 会拉取 FastAPI/Uvicorn；`pty` 会拉取 `ptyprocess`（POSIX）或 `pywinpty`（原生 Windows —— 注意嵌入的 TUI 本身仍需要 WSL）。`pip install hermes-agent[all]` 包含所有 extra，如果你还需要消息/语音等功能，这是最简便的方式。

当你在没有安装依赖的情况下运行 `hermes dashboard` 时，它会提示你需要安装什么。如果前端尚未构建且 `npm` 可用，它会在首次启动时自动构建。

## 页面

### 状态 (Status)

着陆页显示你安装的实时概览：

- **Agent 版本** 和发布日期
- **Gateway 状态** —— 运行中/已停止、PID、已连接的平台及其状态
- **活跃会话** —— 最近 5 分钟内活跃的会话数量
- **最近会话** —— 最近 20 个会话的列表，包含模型、消息数量、token 用量以及对话预览

状态页面每 5 秒自动刷新。

### 聊天 (Chat)

**Chat** 标签页将完整的 Hermes TUI（与 `hermes --tui` 相同的界面）直接嵌入浏览器。你在终端 TUI 中可以执行的所有操作 —— slash 命令、模型选择器、工具调用卡片、markdown 流式输出、clarify/sudo/approval 提示、皮肤主题 —— 在这里都能以完全相同的方式工作，因为仪表板运行的是真实的 TUI 二进制文件，并通过 [xterm.js](https://xtermjs.org/) 渲染其 ANSI 输出，使用 WebGL 渲染器实现像素级精确的单元格布局。

**工作原理：**

- `/api/pty` 打开一个通过仪表板 session token 认证的 WebSocket
- 服务器在 POSIX 伪终端后启动 `hermes --tui`
- 按键传输到 PTY；ANSI 输出流回浏览器
- xterm.js 的 WebGL 渲染器将每个单元格绘制到整数像素网格；鼠标跟踪（SGR 1006）、宽字符（Unicode 11）和框线字形都能原生渲染
- 调整浏览器窗口大小会通过 `@xterm/addon-fit` 插件调整 TUI 大小

**恢复已有会话：** 在 **Sessions** 标签页中，点击任意会话旁的播放图标（▶）。这会跳转到 `/chat?resume=<id>` 并使用 `--resume` 启动 TUI，加载完整历史记录。

**前置条件：**

- Node.js（与 `hermes --tui` 的要求相同；TUI 包在首次启动时构建）
- `ptyprocess` —— 由 `pty` extra 安装（`pip install 'hermes-agent[web,pty]'`，或 `[all]` 包含两者）
- POSIX 内核（Linux、macOS 或 WSL2）。`/chat` 终端面板特别需要 POSIX PTY —— 原生 Windows Python 没有等效实现，因此在原生 Windows 安装上，仪表板的其余部分（会话、任务、指标、配置编辑器）可以工作，但 `/chat` 标签页会显示横幅，提示你使用 WSL2 来使用该功能。

关闭浏览器标签页后，PTY 会在服务器端被干净地回收。重新打开会生成一个新会话。

### 配置 (Config)

基于表单的 `config.yaml` 编辑器。所有 150 多个配置字段都通过 `DEFAULT_CONFIG` 自动发现，并按标签页分类组织：

- **model** —— 默认模型、provider、base URL、reasoning 设置
- **terminal** —— 后端（local/docker/ssh/modal）、超时、shell 偏好
- **display** —— 皮肤、工具进度、恢复显示、spinner 设置
- **agent** —— 最大迭代次数、gateway 超时、service tier
- **delegation** —— subagent 限制、reasoning effort
- **memory** —— provider 选择、上下文注入设置
- **approvals** —— 危险命令审批模式（ask/yolo/deny）
- 以及更多 —— `config.yaml` 的每个部分都有对应的表单字段

具有已知有效值的字段（terminal 后端、皮肤、审批模式等）会渲染为下拉框。布尔值渲染为开关。其他所有内容都是文本输入。

**操作：**

- **Save** —— 立即将更改写入 `config.yaml`
- **Reset to defaults** —— 将所有字段恢复为默认值（在点击 Save 之前不会保存）
- **Export** —— 将当前配置下载为 JSON
- **Import** —— 上传 JSON 配置文件以替换当前值

:::tip
配置更改在下一个 agent 会话或 gateway 重启时生效。Web Dashboard 编辑的是与 `hermes config set` 和 gateway 读取的相同的 `config.yaml` 文件。
:::

### API 密钥 (API Keys)

管理存储 API 密钥和凭证的 `.env` 文件。密钥按类别分组：

- **LLM Providers** —— OpenRouter、Anthropic、OpenAI、DeepSeek 等
- **Tool API Keys** —— Browserbase、Firecrawl、Tavily、ElevenLabs 等
- **Messaging Platforms** —— Telegram、Discord、Slack bot token 等
- **Agent Settings** —— 非机密环境变量，如 `API_SERVER_ENABLED`

每个密钥显示：
- 当前是否已设置（带有脱敏的值预览）
- 用途说明
- 指向 provider 注册/密钥页面的链接
- 用于设置或更新值的输入字段
- 删除按钮

高级/不常用的密钥默认隐藏，可通过开关显示。

### 会话 (Sessions)

浏览和检查所有 agent 会话。每行显示会话标题、来源平台图标（CLI、Telegram、Discord、Slack、cron）、模型名称、消息数量、工具调用数量以及上次活跃时间。活跃会话会标有脉冲徽章。

- **Search** —— 使用 FTS5 对所有消息内容进行全文搜索。结果会显示高亮片段，展开时自动滚动到第一条匹配消息。
- **Expand** —— 点击会话可加载其完整消息历史。消息按角色（user、assistant、system、tool）进行颜色编码，并以 Markdown 格式渲染，带有语法高亮。
- **Tool calls** —— 包含工具调用的 assistant 消息会显示可折叠的块，包含函数名和 JSON 参数。
- **Delete** —— 使用垃圾桶图标删除会话及其消息历史。

### 日志 (Logs)

查看 agent、gateway 和错误日志文件，支持过滤和实时追踪。

- **File** —— 在 `agent`、`errors` 和 `gateway` 日志文件之间切换
- **Level** —— 按日志级别过滤：ALL、DEBUG、INFO、WARNING 或 ERROR
- **Component** —— 按来源组件过滤：all、gateway、agent、tools、cli 或 cron
- **Lines** —— 选择显示多少行（50、100、200 或 500）
- **Auto-refresh** —— 开启每 5 秒轮询新日志行的实时追踪
- **Color-coded** —— 日志行按严重程度着色（错误为红色、警告为黄色、调试为暗色）

### 分析 (Analytics)

基于会话历史计算的用量和成本分析。选择时间段（7、30 或 90 天）可查看：

- **Summary cards** —— 总 token（输入/输出）、缓存命中率、总估算或实际成本，以及总会话数和日均值
- **Daily token chart** —— 堆叠柱状图，显示每天的输入和输出 token 用量，悬停提示显示细分和成本
- **Daily breakdown table** —— 每天的日期、会话数、输入 token、输出 token、缓存命中率和成本
- **Per-model breakdown** —— 表格显示每个使用的模型、其会话数、token 用量和估算成本

### 定时任务 (Cron)

创建和管理按计划重复运行 agent prompt 的定时 cron 任务。

- **Create** —— 填写名称（可选）、prompt、cron 表达式（例如 `0 9 * * *`）和投递目标（local、Telegram、Discord、Slack 或 email）
- **Job list** —— 每个任务显示其名称、prompt 预览、计划表达式、状态徽章（enabled/paused/error）、投递目标、上次运行时间和下次运行时间
- **Pause / Resume** —— 在活跃和暂停状态之间切换任务
- **Trigger now** —— 立即在计划外执行任务
- **Delete** —— 永久删除 cron 任务

### 技能 (Skills)

浏览、搜索和切换技能及工具集。技能从 `~/.hermes/skills/` 加载，并按类别分组。

- **Search** —— 按名称、描述或类别过滤技能和工具集
- **Category filter** —— 点击类别标签缩小列表（例如 MLOps、MCP、Red Teaming、AI）
- **Toggle** —— 使用开关启用或禁用单个技能。更改在下次会话时生效。
- **Toolsets** —— 单独的部分显示内置工具集（文件操作、网页浏览等）及其活跃/非活跃状态、设置要求和包含的工具列表

:::warning 安全提示
Web Dashboard 会读取和写入你的 `.env` 文件，其中包含 API 密钥和机密。它默认绑定到 `127.0.0.1` —— 仅可从你的本地机器访问。如果你绑定到 `0.0.0.0`，网络上的任何人都可以查看和修改你的凭证。仪表板本身没有认证机制。
:::

## `/reload` Slash 命令

仪表板 PR 还为交互式 CLI 添加了 `/reload` slash 命令。通过 Web Dashboard（或直接编辑 `.env`）更改 API 密钥后，在活跃的 CLI 会话中使用 `/reload` 即可在不重启的情况下应用更改：

```
You → /reload
  Reloaded .env (3 var(s) updated)
```

这会重新读取 `~/.hermes/.env` 到运行进程的上下文中。当你通过仪表板添加了新的 provider 密钥并希望立即使用时，非常有用。

## REST API

Web Dashboard 暴露了一个 REST API，前端会消费这些端点。你也可以直接调用这些端点进行自动化：

### GET /api/status

返回 agent 版本、gateway 状态、平台状态和活跃会话数。

### GET /api/sessions

返回最近 20 个会话及其元数据（模型、token 数量、时间戳、预览）。

### GET /api/config

以 JSON 格式返回当前 `config.yaml` 的内容。

### GET /api/config/defaults

返回默认配置值。

### GET /api/config/schema

返回描述每个配置字段的 schema —— 类型、描述、类别以及适用的选项。前端使用此 schema 为每个字段渲染正确的输入组件。

### PUT /api/config

保存新配置。请求体：`{"config": {...}}`。

### GET /api/env

返回所有已知环境变量及其设置/未设置状态、脱敏值、描述和类别。

### PUT /api/env

设置环境变量。请求体：`{"key": "VAR_NAME", "value": "[REDACTED]"}`。

### DELETE /api/env

删除环境变量。请求体：`{"key": "VAR_NAME"}`。

### GET /api/sessions/\{session_id\}

返回单个会话的元数据。

### GET /api/sessions/\{session_id\}/messages

返回会话的完整消息历史，包括工具调用和时间戳。

### GET /api/sessions/search

对消息内容进行全文搜索。查询参数：`q`。返回匹配的会话 ID 和高亮片段。

### DELETE /api/sessions/\{session_id\}

删除会话及其消息历史。

### GET /api/logs

返回日志行。查询参数：`file`（agent/errors/gateway）、`lines`（数量）、`level`、`component`。

### GET /api/analytics/usage

返回 token 用量、成本和会话分析。查询参数：`days`（默认 30）。响应包含每日细分和按模型聚合的数据。

### GET /api/cron/jobs

返回所有配置的 cron 任务及其状态、计划和运行历史。

### POST /api/cron/jobs

创建新的 cron 任务。请求体：`{"prompt": "...", "schedule": "0 9 * * *", "name": "...", "deliver": "local"}`。

### POST /api/cron/jobs/\{job_id\}/pause

暂停 cron 任务。

### POST /api/cron/jobs/\{job_id\}/resume

恢复已暂停的 cron 任务。

### POST /api/cron/jobs/\{job_id\}/trigger

在计划外立即触发 cron 任务。

### DELETE /api/cron/jobs/\{job_id\}

删除 cron 任务。

### GET /api/skills

返回所有技能及其名称、描述、类别和启用状态。

### PUT /api/skills/toggle

启用或禁用技能。请求体：`{"name": "skill-name", "enabled": true}`。

### GET /api/tools/toolsets

返回所有工具集及其标签、描述、工具列表和活跃/已配置状态。

## CORS

Web 服务器将 CORS 限制为仅 localhost 来源：

- `http://localhost:9119` / `http://127.0.0.1:9119`（生产环境）
- `http://localhost:3000` / `http://127.0.0.1:3000`
- `http://localhost:5173` / `http://127.0.0.1:5173`（Vite 开发服务器）

如果你在自定义端口上运行服务器，该来源会自动添加。

## 开发

如果你正在为 Web Dashboard 前端做贡献：

```bash
# 终端 1：启动后端 API
hermes dashboard --no-open

# 终端 2：启动 Vite 开发服务器（带 HMR）
cd web/
npm install
npm run dev
```

Vite 开发服务器在 `http://localhost:5173` 运行，将 `/api` 请求代理到 `http://127.0.0.1:9119` 的 FastAPI 后端。

前端使用 React 19、TypeScript、Tailwind CSS v4 和 shadcn/ui 风格的组件构建。生产构建输出到 `hermes_cli/web_dist/`，由 FastAPI 服务器作为静态 SPA 提供服务。

## 更新时自动构建

当你运行 `hermes update` 时，如果 `npm` 可用，Web 前端会自动重新构建。这会使仪表板与代码更新保持同步。如果未安装 `npm`，更新会跳过前端构建，`hermes dashboard` 将在首次启动时构建它。

## 主题和插件

仪表板自带六种内置主题，并可通过用户定义的主题、插件标签页和后端 API 路由进行扩展 —— 全部即插即用，无需克隆仓库。

**实时切换主题** —— 从顶部栏点击语言切换器旁边的调色板图标。选择会持久化到 `config.yaml` 的 `dashboard.theme` 中，并在页面加载时恢复。

内置主题：

| 主题 | 特点 |
|-------|-----------|
| **Hermes Teal** (`default`) | 深青色 + 奶油色，系统字体，舒适的间距 |
| **Hermes Teal (Large)** (`default-large`) | 与默认主题相同，但使用 18px 文本和更宽松的间距 |
| **Midnight** (`midnight`) | 深蓝紫色，Inter + JetBrains Mono |
| **Ember** (`ember`) | 暖深红色 + 青铜色，Spectral 衬线体 + IBM Plex Mono |
| **Mono** (`mono`) | 灰度，IBM Plex，紧凑 |
| **Cyberpunk** (`cyberpunk`) | 霓虹绿配黑色，Share Tech Mono |
| **Rosé** (`rose`) | 粉色 + 象牙色，Fraunces 衬线体，宽敞 |

要构建自己的主题、添加插件标签页、注入到 shell 插槽或暴露插件特定的 REST 端点，请参阅 **[扩展仪表板 (Extending the Dashboard)](./extending-the-dashboard)** —— 完整指南涵盖：

- Theme YAML schema —— palette、typography、layout、assets、componentStyles、colorOverrides、customCSS
- Layout 变体 —— `standard`、`cockpit`、`tiled`
- Plugin manifest、SDK、shell slots、page-scoped slots（将组件注入内置页面而无需覆盖它们）、后端 FastAPI 路由
- 完整的主题+插件组合演练（Strike Freedom cockpit 演示）
- 发现、重载和故障排除
