---
sidebar_position: 14
title: "API 服务器"
description: "将 hermes-agent 作为 OpenAI 兼容 API 暴露给任何前端"
---

# API 服务器

API 服务器将 hermes-agent 暴露为 OpenAI 兼容的 HTTP 端点。任何使用 OpenAI 格式的前端 — Open WebUI、LobeChat、LibreChat、NextChat、ChatBox 等等 — 都可以连接到 hermes-agent 并将其用作后端。

您的代理使用完整的工具集（终端、文件操作、Web 搜索、内存、技能）处理请求并返回最终响应。流式传输时，工具进度指示器内联出现，以便前端显示代理正在做什么。

## 快速开始

### 1. 启用 API 服务器

添加到 `~/.hermes/.env`：

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# 可选：仅当浏览器必须直接调用 Hermes 时
# API_SERVER_CORS_ORIGINS=http://localhost:3000
```

### 2. 启动网关

```bash
hermes gateway
```

您会看到：

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. 连接前端

将任何 OpenAI 兼容客户端指向 `http://localhost:8642/v1`：

```bash
# 用 curl 测试
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change-me-local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

或者连接 Open WebUI、LobeChat 或任何其他前端 — 请参阅 [Open WebUI 集成指南](/docs/user-guide/messaging/open-webui) 获取分步说明。

## 端点

### POST /v1/chat/completions

标准 OpenAI Chat Completions 格式。无状态 — 每个请求通过 `messages` 数组包含完整对话。

**请求：**
```json
{
  "model": "hermes-agent",
  "messages": [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "Write a fibonacci function"}
  ],
  "stream": false
}
```

**响应：**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "hermes-agent",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Here's a fibonacci function..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
}
```

**内联图像输入：** 用户消息可以将 `content` 作为 `text` 和 `image_url` 部分的数组发送。支持远程 `http(s)` URL 和 `data:image/...` URL：

```json
{
  "model": "hermes-agent",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}}
      ]
    }
  ]
}
```

上传的文件（`file` / `input_file` / `file_id`）和非图像 `data:` URL 返回 `400 unsupported_content_type`。

**流式传输**（`"stream": true`）：返回带逐令牌响应块的 Server-Sent Events（SSE）。对于 **Chat Completions**，流使用标准 `chat.completion.chunk` 事件加上 Hermes 的自定义 `hermes.tool.progress` 事件用于工具启动 UX。对于 **Responses**，流使用 OpenAI Responses 事件类型，如 `response.created`、`response.output_text.delta`、`response.output_item.added`、`response.output_item.done` 和 `response.completed`。

**流中的工具进度**：
- **Chat Completions**：Hermes 为工具启动可见性发出 `event: hermes.tool.progress`，而不污染持久化的助手文本。
- **Responses**：Hermes 在 SSE 流中发出规范原生的 `function_call` 和 `function_call_output` 输出项，以便客户端可以实时呈现结构化工具 UI。

### POST /v1/responses

OpenAI Responses API 格式。通过 `previous_response_id` 支持服务器端对话状态 — 服务器存储完整对话历史（包括工具调用和结果），因此多轮上下文保持不变，而无需客户端管理。

**请求：**
```json
{
  "model": "hermes-agent",
  "input": "What files are in my project?",
  "instructions": "You are a helpful coding assistant.",
  "store": true
}
```

**响应：**
```json
{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "model": "hermes-agent",
  "output": [
    {"type": "function_call", "name": "terminal", "arguments": "{\"command\": \"ls\"}", "call_id": "call_1"},
    {"type": "function_call_output", "call_id": "call_1", "output": "README.md src/ tests/"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Your project has..."}]}
  ],
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

**内联图像输入：** `input[].content` 可以包含 `input_text` 和 `input_image` 部分。支持远程 URL 和 `data:image/...` URL：

```json
{
  "model": "hermes-agent",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this screenshot."},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K..."}
      ]
    }
  ]
}
```

上传的文件（`input_file` / `file_id`）和非图像 `data:` URL 返回 `400 unsupported_content_type`。

#### 通过 previous_response_id 进行多轮对话

链接响应以跨轮次保持完整上下文（包括工具调用）：

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

服务器从存储的响应链重建完整对话 — 所有之前的工具调用和结果都被保留。链接的请求也共享同一个会话，因此多轮对话在仪表板和会话历史中显示为单个条目。

#### 命名对话

使用 `conversation` 参数而不是跟踪响应 ID：

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

服务器自动链接到该对话中最新的响应。像网关会话的 `/title` 命令一样。

### GET /v1/responses/\{id\}

通过 ID 检索先前存储的响应。

### DELETE /v1/responses/\{id\}

删除存储的响应。

### GET /v1/models

将代理列为可用模型。广告的模型名称默认为[配置文件](/docs/user-guide/profiles)名称（或默认配置文件的 `hermes-agent`）。大多数前端需要用于模型发现。

### GET /v1/capabilities

返回 API 服务器稳定表面的机器可读描述，供外部 UI、编排器和插件桥接使用。

```json
{
  "object": "hermes.api_server.capabilities",
  "platform": "hermes-agent",
  "model": "hermes-agent",
  "auth": {"type": "bearer", "required": true},
  "features": {
    "chat_completions": true,
    "responses_api": true,
    "run_submission": true,
    "run_status": true,
    "run_events_sse": true,
    "run_stop": true
  }
}
```

在集成仪表板、浏览器 UI 或控制平面时使用此端点，以便它们可以发现运行中的 Hermes 版本是否支持运行、流式传输、取消和会话连续性，而不依赖私有 Python 内部。

### GET /health

健康检查。返回 `{"status": "ok"}`。也可在 **GET /v1/health** 获取，适用于期望 `/v1/` 前缀的 OpenAI 兼容客户端。

### GET /health/detailed

扩展健康检查，还报告活动会话、运行中的代理和资源使用情况。对监控/可观察性工具很有用。

## Runs API（流式友好替代）

除了 `/v1/chat/completions` 和 `/v1/responses`，服务器还暴露了一个 **runs** API，适用于客户端想要订阅进度事件而不是自己管理流式传输的长格式会话。

### POST /v1/runs

创建新的代理运行。返回可用于订阅进度事件的 `run_id`。

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

Runs 接受简单的 `input` 字符串和可选的 `session_id`、`instructions`、`conversation_history` 或 `previous_response_id`。当提供 `session_id` 时，Hermes 在运行状态中呈现它，以便外部 UI 可以将其自己的对话 ID 与运行相关联。

### GET /v1/runs/\{run_id\}

轮询当前运行状态。这对需要在保持 SSE 连接打开的情况下获取状态的仪表板很有用，或者对在导航后重新连接的 UI 有用。

```json
{
  "object": "hermes.run",
  "run_id": "run_abc123",
  "status": "completed",
  "session_id": "space-session",
  "model": "hermes-agent",
  "output": "Done.",
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

终端状态（`completed`、`failed` 或 `cancelled`）后的状态会短暂保留，用于轮询和 UI 协调。

### GET /v1/runs/\{run_id\}/events

运行工具调用进度、令牌增量和工作流事件的 Server-Sent Events 流。专为想要附加/分离而不丢失状态的仪表板和厚客户端设计。

### POST /v1/runs/\{run_id\}/stop

中断正在运行的代理轮次。端点立即返回 `{"status": "stopping"}`，而 Hermes 要求活动代理在下一个安全中断点停止。

## Jobs API（后台计划工作）

服务器暴露了一个轻量级 jobs CRUD 表面，用于从远程客户端管理计划/后台代理运行。所有端点都通过相同的 bearer 认证进行门控。

### GET /api/jobs

列出所有计划的作业。

### POST /api/jobs

创建新的计划作业。请求体接受与 `hermes cron` 相同的形状 — prompt、schedule、skills、提供商覆盖、传递目标。

### GET /api/jobs/\{job_id\}

获取单个作业的定义和上次运行状态。

### PATCH /api/jobs/\{job_id\}

更新现有作业的字段（prompt、schedule 等）。部分更新会被合并。

### DELETE /api/jobs/\{job_id\}

删除作业。也会取消任何正在进行的运行。

### POST /api/jobs/\{job_id\}/pause

暂停作业而不删除它。下次计划运行时间戳被挂起，直到恢复。

### POST /api/jobs/\{job_id\}/resume

恢复先前暂停的作业。

### POST /api/jobs/\{job_id\}/run

在计划外立即触发作业运行。

## 系统提示处理

当前端发送 `system` 消息（Chat Completions）或 `instructions` 字段（Responses API）时，hermes-agent **将其叠加在其核心系统提示之上**。您的代理保留其所有工具、内存和技能 — 前端的系统提示添加额外的指令。

这意味着您可以自定义每个前端的行为而不丢失功能：
- Open WebUI 系统提示："You are a Python expert. Always include type hints."
- 代理仍然有终端、文件工具、Web 搜索、内存等。

## 认证

通过 `Authorization` 头部的 Bearer 令牌认证：

```
Authorization: Bearer ***
```

通过 `API_SERVER_KEY` env 变量配置密钥。如果您需要浏览器直接调用 Hermes，还要将 `API_SERVER_CORS_ORIGINS` 设置为明确的允许列表。

:::warning 安全
API 服务器提供对 hermes-agent 完整工具集的访问权限，**包括终端命令**。当绑定到非回环地址如 `0.0.0.0` 时，`API_SERVER_KEY` 是**必需的**。还要将 `API_SERVER_CORS_ORIGINS` 保持窄以控制浏览器访问。

默认绑定地址（`127.0.0.1`）仅用于本地使用。浏览器访问默认禁用；仅为明确的受信任来源启用。
:::

## 配置

### 环境变量

| 变量 | 默认 | 描述 |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | 启用 API 服务器 |
| `API_SERVER_PORT` | `8642` | HTTP 服务器端口 |
| `API_SERVER_HOST` | `127.0.0.1` | 绑定地址（默认仅本地）|
| `API_SERVER_KEY` | _(无)_ | 认证的 Bearer 令牌 |
| `API_SERVER_CORS_ORIGINS` | _(无)_ | 逗号分隔的允许浏览器来源 |
| `API_SERVER_MODEL_NAME` | _(配置文件名称)_ | `/v1/models` 上的模型名称。默认为配置文件名称，或默认配置文件的 `hermes-agent`。 |

### config.yaml

```yaml
# 尚不支持 — 使用环境变量。
# config.yaml 支持将在未来版本中提供。
```

## 安全头

所有响应都包含安全头：
- `X-Content-Type-Options: nosniff` — 防止 MIME 类型嗅探
- `Referrer-Policy: no-referrer` — 防止引用者泄漏

## CORS

API 服务器**默认不启用**浏览器 CORS。

对于直接浏览器访问，设置明确的允许列表：

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

启用 CORS 时：
- **预检响应**包含 `Access-Control-Max-Age: 600`（10 分钟缓存）
- **SSE 流式响应**包含 CORS 头，以便浏览器 EventSource 客户端正常工作
- **`Idempotency-Key`** 是允许的请求头 — 客户端可以发送它用于去重（响应按密钥缓存 5 分钟）

大多数记录的前端如 Open WebUI 都是服务器到服务器连接，根本不需要 CORS。

## 兼容的前端

任何支持 OpenAI API 格式的前端都可以工作。已测试/记录的集成：

| 前端 | Stars | 连接 |
|----------|-------|------------|
| [Open WebUI](/docs/user-guide/messaging/open-webui) | 126k | 有完整指南 |
| LobeChat | 73k | 自定义提供商端点 |
| LibreChat | 34k | librechat.yaml 中的自定义端点 |
| AnythingLLM | 56k | 通用 OpenAI 提供商 |
| NextChat | 87k | BASE_URL env 变量 |
| ChatBox | 39k | API Host 设置 |
| Jan | 26k | 远程模型配置 |
| HF Chat-UI | 8k | OPENAI_BASE_URL |
| big-AGI | 7k | 自定义端点 |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | 直接 HTTP 请求 |

## 使用配置文件的多用户设置

为多个用户提供各自隔离的 Hermes 实例（单独的 config、内存、技能），请使用[配置文件](/docs/user-guide/profiles)：

```bash
# 为每个用户创建配置文件
hermes profile create alice
hermes profile create bob

# 在不同端口上配置每个配置文件的 API 服务器
hermes -p alice config set API_SERVER_ENABLED true
hermes -p alice config set API_SERVER_PORT 8643
hermes -p alice config set API_SERVER_KEY alice-secret

hermes -p bob config set API_SERVER_ENABLED true
hermes -p bob config set API_SERVER_PORT 8644
hermes -p bob config set API_SERVER_KEY bob-secret

# 启动每个配置文件的网关
hermes -p alice gateway &
hermes -p bob gateway &
```

每个配置文件的 API 服务器自动广告配置文件名称作为模型 ID：

- `http://localhost:8643/v1/models` → 模型 `alice`
- `http://localhost:8644/v1/models` → 模型 `bob`

在 Open WebUI 中，将每个添加为单独连接。模型下拉菜单显示 `alice` 和 `bob` 作为不同的模型，每个都由完全隔离的 Hermes 实例支持。请参阅 [Open WebUI 指南](/docs/user-guide/messaging/open-webui#multi-user-setup-with-profiles) 获取详情。

## 限制

- **响应存储** — 存储的响应（用于 `previous_response_id`）持久化在 SQLite 中，在网关重启后保持。最多 100 个存储响应（LRU 驱逐）。
- **不支持文件上传** — 内联图像在 `/v1/chat/completions` 和 `/v1/responses` 上都支持，但上传的文件（`file`、`input_file`、`file_id`）和非图像文档输入不支持通过 API。
- **模型字段是装饰性的** — 请求中的 `model` 字段被接受，但实际使用的 LLM 模型在 config.yaml 中服务器端配置。

## 代理模式

API 服务器还用作**网关代理模式**的后端。当另一个 Hermes 网关实例配置了 `GATEWAY_PROXY_URL` 指向此 API 服务器时，它将所有消息转发到这里，而不是运行自己的代理。这支持分离部署 — 例如，处理 Matrix E2EE 的 Docker 容器中继到主机侧代理。

请参阅 [Matrix 代理模式](/docs/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos) 获取完整的设置指南。
