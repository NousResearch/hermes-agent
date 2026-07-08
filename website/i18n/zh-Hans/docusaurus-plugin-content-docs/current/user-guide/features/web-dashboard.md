---
sidebar_position: 15
title: "Web Dashboard"
description: "基于浏览器的仪表板，用于管理配置、API 密钥、会话、日志、分析、定时任务和技能"
---

# Web Dashboard

Web Dashboard 是一个基于浏览器的 UI，用于管理你的 Hermes Agent 安装。无需编辑 YAML 文件或运行 CLI 命令，即可通过简洁的 Web 界面配置设置、管理 API 密钥并监控会话。

## 快速开始

```bash
hermes dashboard
```

这将启动一个本地 Web 服务器，并在浏览器中打开 `http://127.0.0.1:9119`。Dashboard 完全在你的机器上运行——数据不会离开 localhost。

### 选项

| 标志 | 默认值 | 描述 |
|------|---------|-------------|
| `--port` | `9119` | Web 服务器运行端口 |
| `--host` | `127.0.0.1` | 绑定地址 |
| `--no-open` | — | 不自动打开浏览器 |
| `--insecure` | 关闭 | 允许绑定到非 localhost 主机（**危险**——会在网络上暴露 API 密钥；请配合防火墙和强认证使用） |

```bash
# 自定义端口
hermes dashboard --port 8080

# 绑定到所有接口（在共享网络上请谨慎使用）
hermes dashboard --host 0.0.0.0

# 启动时不打开浏览器
hermes dashboard --no-open
```

## 前置条件

默认的 `hermes-agent` 安装不包含 HTTP 栈或 PTY 辅助工具——这些是可选扩展。**Web Dashboard** 需要 FastAPI 和 Uvicorn（`web` 扩展）。**Chat** 标签页还需要 `ptyprocess` 来在伪终端（pseudo-terminal）后面启动嵌入式 TUI（POSIX 上的 `pty` 扩展）。使用以下命令同时安装：

```bash
pip install 'hermes-agent[web,pty]'
```

`web` 扩展会引入 FastAPI/Uvicorn；`pty` 扩展会引入 `ptyprocess`（POSIX）或 `pywinpty`（原生 Windows——注意嵌入式 TUI 本身仍需要 WSL）。`pip install hermes-agent[all]` 包含两个扩展，如果你还需要消息/语音等功能，这是最简便的方式。

在没有依赖项的情况下运行 `hermes dashboard` 时，它会告诉你需要安装什么。如果前端尚未构建且 `npm` 可用，则会在首次启动时自动构建。

Chat 标签页是每次 `hermes dashboard` 启动的一部分——内嵌的浏览器聊天面板（通过 PTY/WebSocket 运行 TUI）始终可用，无需任何额外参数。

## 页面

### Status（状态）

首页显示你的安装的实时概览：

- **Agent 版本**和发布日期
- **Gateway 状态**——运行中/已停止、PID、已连接平台及其状态
- **活跃会话**——过去 5 分钟内活跃的会话数量
- **最近会话**——最近 20 个会话的列表，包含模型、消息数、token 用量和对话预览

状态页每 5 秒自动刷新一次。

### Chat（聊天）

**Chat** 标签页将完整的 Hermes TUI（与 `hermes --tui` 相同的界面）直接嵌入浏览器。你在终端 TUI 中能做的一切——斜杠命令、模型选择器、工具调用卡片、Markdown 流式输出、clarify/sudo/approval 提示、皮肤主题——在这里都完全一致，因为 Dashboard 运行的是真实的 TUI 二进制文件，并通过 [xterm.js](https://xtermjs.org/) 的 WebGL 渲染器以像素级精度渲染其 ANSI 输出。

**工作原理：**

- `/api/pty` 打开一个经 Dashboard 会话 token 认证的 WebSocket
- 服务器在 POSIX 伪终端后面启动 `hermes --tui`
- 按键传输到 PTY；ANSI 输出流式返回浏览器
- xterm.js 的 WebGL 渲染器将每个单元格绘制到整数像素网格；鼠标追踪（SGR 1006）、宽字符（Unicode 11）和方框绘制字形均原生渲染
- 调整浏览器窗口大小会通过 `@xterm/addon-fit` 插件调整 TUI 大小

**恢复已有会话：** 在 **Sessions** 标签页中，点击任意会话旁的播放图标（▶）。这会跳转到 `/chat?resume=<id>` 并以 `--resume` 参数启动 TUI，加载完整历史记录。

**前置条件：**

- Node.js（与 `hermes --tui` 相同的要求；TUI 包在首次启动时构建）
- `ptyprocess`——由 `pty` 扩展安装（`pip install 'hermes-agent[web,pty]'`，或 `[all]` 同时包含两者）
- POSIX 内核（Linux、macOS 或 WSL2）。`/chat` 终端面板特别需要 POSIX PTY——原生 Windows Python 没有等效实现，因此在原生 Windows 安装上，Dashboard 的其余部分（sessions、jobs、metrics、config editor）可以正常工作，但 `/chat` 标签页会显示提示，告知你需要使用 WSL2 才能使用该功能。

关闭浏览器标签页后，PTY 会在服务器端被干净地回收。重新打开会启动一个新会话。

### Config（配置）

`config.yaml` 的表单式编辑器。所有 150+ 个配置字段均从 `DEFAULT_CONFIG` 自动发现，并按标签页分类组织：

- **model** — 默认模型、提供商、基础 URL、推理设置
- **terminal** — 后端（local/docker/ssh/modal）、超时、Shell 偏好
- **display** — 皮肤、工具进度、恢复显示、spinner 设置
- **agent** — 最大迭代次数、gateway 超时、服务层级
- **delegation** — 子 agent 限制、推理力度
- **memory** — 提供商选择、上下文注入设置
- **approvals** — 危险命令审批模式（ask/yolo/deny）
- 更多——config.yaml 的每个部分都有对应的表单字段

具有已知有效值的字段（terminal 后端、皮肤、审批模式等）渲染为下拉菜单。布尔值渲染为开关。其余均为文本输入框。

**操作：**

- **Save** — 立即将更改写入 `config.yaml`
- **Reset to defaults** — 将所有字段恢复为默认值（点击 Save 前不会保存）
- **Export** — 将当前配置下载为 JSON
- **Import** — 上传 JSON 配置文件以替换当前值

:::tip
配置更改在下一次 agent 会话或 gateway 重启时生效。Web Dashboard 编辑的是 `hermes config set` 和 gateway 读取的同一个 `config.yaml` 文件。
:::

### API Keys（API 密钥）

管理存储 API 密钥和凭据的 `.env` 文件。密钥按类别分组：

- **LLM Providers** — OpenRouter、Anthropic、OpenAI、DeepSeek 等
- **Tool API Keys** — Browserbase、Firecrawl、Tavily、ElevenLabs 等
- **Messaging Platforms** — Telegram、Discord、Slack bot token 等
- **Agent Settings** — 非敏感环境变量，如 `API_SERVER_ENABLED`

每个密钥显示：
- 是否已设置（带有值的脱敏预览）
- 用途说明
- 提供商注册/密钥页面的链接
- 用于设置或更新值的输入框
- 删除按钮

高级/不常用的密钥默认隐藏，可通过开关显示。

### Sessions（会话）

浏览和检查所有 agent 会话。每行显示会话标题、来源平台图标（CLI、Telegram、Discord、Slack、cron）、模型名称、消息数、工具调用数以及最后活跃时间。实时会话以脉冲徽章标记。

- **Search** — 使用 FTS5 对所有消息内容进行全文搜索。结果显示高亮片段，展开时自动滚动到第一条匹配消息。
- **Expand** — 点击会话以加载完整消息历史。消息按角色（user、assistant、system、tool）用颜色区分，并以带语法高亮的 Markdown 渲染。
- **Tool calls** — 包含工具调用的 assistant 消息显示可折叠块，包含函数名和 JSON 参数。
- **Delete** — 使用垃圾桶图标删除会话及其消息历史。

### Logs（日志）

查看 agent、gateway 和错误日志文件，支持过滤和实时追踪。

- **File** — 在 `agent`、`errors` 和 `gateway` 日志文件之间切换
- **Level** — 按日志级别过滤：ALL、DEBUG、INFO、WARNING 或 ERROR
- **Component** — 按来源组件过滤：all、gateway、agent、tools、cli 或 cron
- **Lines** — 选择显示行数（50、100、200 或 500）
- **Auto-refresh** — 切换实时追踪，每 5 秒轮询新日志行
- **Color-coded** — 日志行按严重程度着色（错误为红色，警告为黄色，debug 为暗色）

### Analytics（分析）

基于会话历史计算的用量和成本分析。选择时间段（7、30 或 90 天）查看：

- **Summary cards** — 总 token 数（输入/输出）、缓存命中率、总估算或实际成本，以及总会话数和日均值
- **Daily token chart** — 堆叠柱状图，显示每日输入和输出 token 用量，悬停提示显示明细和成本
- **Daily breakdown table** — 每日日期、会话数、输入 token、输出 token、缓存命中率和成本
- **Per-model breakdown** — 显示每个使用模型的会话数、token 用量和估算成本的表格

### Cron（定时任务）

创建和管理按定期计划运行 agent prompt 的定时任务。

- **Create** — 填写名称（可选）、prompt、cron 表达式（如 `0 9 * * *`）和投递目标（local、Telegram、Discord、Slack 或 email）
- **Job list** — 每个任务显示其名称、prompt 预览、计划表达式、状态徽章（enabled/paused/error）、投递目标、上次运行时间和下次运行时间
- **Pause / Resume** — 在活跃和暂停状态之间切换任务
- **Trigger now** — 在正常计划之外立即执行任务
- **Delete** — 永久删除定时任务

### Skills（技能）

浏览、搜索和切换技能与工具集。技能从 `~/.hermes/skills/` 加载，并按类别分组。

- **Search** — 按名称、描述或类别过滤技能和工具集
- **Category filter** — 点击类别标签缩小列表范围（如 MLOps、MCP、Red Teaming、AI）
- **Toggle** — 使用开关启用或禁用单个技能。更改在下一次会话时生效。
- **Toolsets** — 单独的部分显示内置工具集（文件操作、Web 浏览等），包含其活跃/非活跃状态、设置要求和包含的工具列表

:::warning 安全提示
Web Dashboard 会读写包含 API 密钥和机密的 `.env` 文件。它默认绑定到 `127.0.0.1`——只能从本机访问。如果绑定到 `0.0.0.0`，网络上的任何人都可以查看和修改你的凭据。Dashboard 本身没有任何认证机制。
:::

## `/reload` 斜杠命令

Dashboard 还为交互式 CLI 添加了 `/reload` 斜杠命令。通过 Web Dashboard（或直接编辑 `.env`）更改 API 密钥后，在活跃的 CLI 会话中使用 `/reload` 即可获取更改，无需重启：

```
You → /reload
  Reloaded .env (3 var(s) updated)
```

这会将 `~/.hermes/.env` 重新读取到运行中进程的环境中。当你通过 Dashboard 添加了新的提供商密钥并希望立即使用时非常有用。

## REST API

Web Dashboard 暴露了一个供前端使用的 REST API。你也可以直接调用这些端点进行自动化操作：

### GET /api/status

返回 agent 版本、gateway 状态、平台状态和活跃会话数。

### GET /api/sessions

返回最近 20 个会话的元数据（模型、token 数、时间戳、预览）。

### GET /api/config

以 JSON 格式返回当前 `config.yaml` 内容。

### GET /api/config/defaults

返回默认配置值。

### GET /api/config/schema

返回描述每个配置字段的 schema——类型、描述、类别，以及适用时的选项。前端使用此 schema 为每个字段渲染正确的输入控件。

### PUT /api/config

保存新配置。请求体：`{"config": {...}}`。

### GET /api/env

返回所有已知环境变量，包含其设置/未设置状态、脱敏值、描述和类别。

### PUT /api/env

设置环境变量。请求体：`{"key": "VAR_NAME", "value": "secret"}`。

### DELETE /api/env

删除环境变量。请求体：`{"key": "VAR_NAME"}`。

### GET /api/sessions/\{session_id\}

返回单个会话的元数据。

### GET /api/sessions/\{session_id\}/messages

返回会话的完整消息历史，包含工具调用和时间戳。

### GET /api/sessions/search

对消息内容进行全文搜索。查询参数：`q`。返回匹配的会话 ID 和高亮片段。

### DELETE /api/sessions/\{session_id\}

删除会话及其消息历史。

### GET /api/logs

返回日志行。查询参数：`file`（agent/errors/gateway）、`lines`（数量）、`level`、`component`。

### GET /api/analytics/usage

返回 token 用量、成本和会话分析。查询参数：`days`（默认 30）。响应包含每日明细和按模型聚合数据。

### GET /api/cron/jobs

返回所有已配置的定时任务，包含其状态、计划和运行历史。

### POST /api/cron/jobs

创建新定时任务。请求体：`{"prompt": "...", "schedule": "0 9 * * *", "name": "...", "deliver": "local"}`。

### POST /api/cron/jobs/\{job_id\}/pause

暂停定时任务。

### POST /api/cron/jobs/\{job_id\}/resume

恢复已暂停的定时任务。

### POST /api/cron/jobs/\{job_id\}/trigger

在计划之外立即触发定时任务。

### DELETE /api/cron/jobs/\{job_id\}

删除定时任务。

### GET /api/skills

返回所有技能，包含其名称、描述、类别和启用状态。

### PUT /api/skills/toggle

启用或禁用技能。请求体：`{"name": "skill-name", "enabled": true}`。

### GET /api/tools/toolsets

返回所有工具集，包含其标签、描述、工具列表以及活跃/已配置状态。

## CORS

Web 服务器将 CORS 限制为仅 localhost 来源：

- `http://localhost:9119` / `http://127.0.0.1:9119`（生产环境）
- `http://localhost:3000` / `http://127.0.0.1:3000`
- `http://localhost:5173` / `http://127.0.0.1:5173`（Vite 开发服务器）

如果你在自定义端口上运行服务器，该来源会自动添加。

## 开发

如果你要为 Web Dashboard 前端做贡献：

```bash
# 终端 1：启动后端 API
hermes dashboard --no-open

# 终端 2：启动带 HMR 的 Vite 开发服务器
cd web/
npm install
npm run dev
```

`http://localhost:5173` 上的 Vite 开发服务器会将 `/api` 请求代理到 `http://127.0.0.1:9119` 上的 FastAPI 后端。

前端使用 React 19、TypeScript、Tailwind CSS v4 和 shadcn/ui 风格组件构建。生产构建输出到 `hermes_cli/web_dist/`，由 FastAPI 服务器作为静态 SPA 提供服务。

## 更新时自动构建

运行 `hermes update` 时，如果 `npm` 可用，Web 前端会自动重新构建。这使 Dashboard 与代码更新保持同步。如果未安装 `npm`，更新会跳过前端构建，`hermes dashboard` 将在首次启动时构建。

## 主题与插件

Dashboard 内置六个主题，并可通过用户自定义主题、插件标签页和后端 API 路由进行扩展——全部即插即用，无需克隆仓库。

**实时切换主题**：点击顶部栏语言切换器旁的调色板图标。选择会持久化到 `config.yaml` 的 `dashboard.theme` 下，并在页面加载时恢复。

内置主题：

| 主题 | 特点 |
|-------|-----------|
| **Hermes Teal** (`default`) | 深青色 + 奶油色，系统字体，舒适间距 |
| **Hermes Teal (Large)** (`default-large`) | 与 default 相同，但使用 18px 文字和更宽松的间距 |
| **Midnight** (`midnight`) | 深蓝紫色，Inter + JetBrains Mono |
| **Ember** (`ember`) | 暖深红 + 古铜色，Spectral 衬线体 + IBM Plex Mono |
| **Mono** (`mono`) | 灰度，IBM Plex，紧凑 |
| **Cyberpunk** (`cyberpunk`) | 黑底霓虹绿，Share Tech Mono |
| **Rosé** (`rose`) | 粉色 + 象牙色，Fraunces 衬线体，宽松 |

如需构建自定义主题、添加插件标签页、注入 shell 插槽或暴露插件专属 REST 端点，请参阅 **[扩展 Dashboard](./extending-the-dashboard)**——完整指南涵盖：

- 主题 YAML schema——调色板、排版、布局、资源、componentStyles、colorOverrides、customCSS
- 布局变体——`standard`、`cockpit`、`tiled`
- 插件 manifest、SDK、shell 插槽、页面级插槽（在不覆盖内置页面的情况下注入控件）、后端 FastAPI 路由
- 完整的主题加插件综合演示（Strike Freedom cockpit 示例）
- 发现、重载和故障排查