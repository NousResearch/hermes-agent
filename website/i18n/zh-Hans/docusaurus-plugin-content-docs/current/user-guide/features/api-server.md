---
sidebar_position: 14
title: "API 服务器"
description: "将 hermes-agent 作为 OpenAI 兼容的 API 暴露给任意前端"
---

# API 服务器

API 服务器将 hermes-agent 作为 OpenAI 兼容的 HTTP 端点暴露出来。任何支持 OpenAI 格式的前端——Open WebUI、LobeChat、LibreChat、NextChat、ChatBox 以及数百个其他工具——都可以连接到 hermes-agent 并将其用作后端。

你的 agent 使用完整工具集（终端、文件操作、网络搜索、记忆、技能）处理请求，并返回最终响应。在流式传输时，工具进度指示器会内联显示，让前端能够展示 agent 正在执行的操作。

:::tip 一个后端同时覆盖模型与工具
Hermes 本身需要配置好 provider（提供商）和工具后端，API 服务器才能发挥作用。[Nous Portal](/user-guide/features/tool-gateway) 订阅同时处理两者——300+ 个模型，以及通过 Tool Gateway 提供的网络/图像/TTS/浏览器功能。在启动 API 服务器之前运行一次 `hermes setup --portal`，Open WebUI 或 LobeChat 等前端即可获得一个完整配备工具的后端。
:::

## 快速开始

### 1. 启用 API 服务器

在 `~/.hermes/.env` 中添加：

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# 可选：仅当浏览器需要直接调用 Hermes 时
# API_SERVER_CORS_ORIGINS=http://localhost:3000
```

### 2. 启动 gateway

```bash
hermes gateway
```

你将看到：

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. 连接前端

将任何 OpenAI 兼容客户端指向 `http://localhost:8642/v1`：

```bash
# 使用 curl 测试
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change-me-local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

或连接 Open WebUI、LobeChat 或其他任意前端——参见 [Open WebUI 集成指南](/user-guide/messaging/open-webui)获取分步说明。

## 端点

### POST /v1/chat/completions

标准 OpenAI Chat Completions 格式。无状态——完整对话通过每次请求的 `messages` 数组传入。

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

上传的文件（`file` / `input_file` / `file_id`）和非图像 `data:` URL 将返回 `400 unsupported_content_type`。

**流式传输**（`"stream": true`）：返回逐 token 响应块的 Server-Sent Events（SSE）。对于 **Chat Completions**，流使用标准 `chat.completion.chunk` 事件，以及 Hermes 自定义的 `hermes.tool.progress` 事件用于工具启动的 UX 展示。对于 **Responses**，流使用 OpenAI Responses 事件类型，如 `response.created`、`response.output_text.delta`、`response.output_item.added`、`response.output_item.done` 和 `response.completed`。

**流中的工具进度：**
- **Chat Completions**：Hermes 发出 `event: hermes.tool.progress` 以提供工具启动可见性，同时不污染持久化的 assistant 文本。
- **Responses**：Hermes 在 SSE 流期间发出符合规范的 `function_call` 和 `function_call_output` 输出项，让客户端能够实时渲染结构化工具 UI。

### POST /v1/responses

OpenAI Responses API 格式。通过 `previous_response_id` 支持服务端对话状态——服务器存储完整的对话历史（包括工具调用和结果），因此多轮上下文无需客户端自行管理。

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

上传的文件（`input_file` / `file_id`）和非图像 `data:` URL 将返回 `400 unsupported_content_type`。

#### 使用 previous_response_id 进行多轮对话

链式响应以在多轮之间保持完整上下文（包括工具调用）：

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

服务器从存储的响应链重建完整对话——所有之前的工具调用和结果均被保留。链式请求还共享同一个 session，因此多轮对话在仪表板和 session 历史中显示为单个条目。

#### 命名对话

使用 `conversation` 参数代替追踪响应 ID：

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

服务器自动链接到该对话中的最新响应。类似于 gateway session 的 `/title` 命令。

### GET /v1/responses/\{id\}

通过 ID 检索之前存储的响应。

### DELETE /v1/responses/\{id\}

删除存储的响应。

### GET /v1/models

将 agent 列为可用模型。广播的模型名称默认为 [profile](/user-guide/profiles) 名称（默认 profile 则为 `hermes-agent`）。大多数前端进行模型发现时需要此端点。

### GET /v1/capabilities

返回 API 服务器稳定接口的机器可读描述，供外部 UI、编排器和插件桥接使用。

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

在集成仪表板、浏览器 UI 或控制平面时使用此端点，以便它们能够发现当前运行的 Hermes 版本是否支持 runs、流式传输、取消和 session 连续性，而无需依赖私有 Python 内部实现。

### GET /health

健康检查。返回 `{"status": "ok"}`。也可通过 **GET /v1/health** 访问，供期望 `/v1/` 前缀的 OpenAI 兼容客户端使用。

### GET /health/detailed

面向监控和控制平面的已认证就绪检查。它会报告当前 profile 的配置、状态数据库、已配置模型、磁盘空间、gateway/platform 状态、活跃 API run、待处理进程完成通知和活跃 delegation 的有限状态。响应只暴露状态与计数，不包含配置值、凭据、路径、命令、队列载荷或原始错误。

公开的 `/health` 路由仍是低开销的存活探针，不运行就绪检查。就绪状态降级时仍返回 HTTP 200；请检查顶层 `status` 和 `readiness.checks` 字段。

## Runs API（流式友好的替代方案）

除 `/v1/chat/completions` 和 `/v1/responses` 外，服务器还暴露了一个 **runs** API，适用于客户端希望订阅进度事件而非自行管理流式传输的长时 session。

### POST /v1/runs

创建新的 agent run。返回可用于订阅进度事件的 `run_id`。

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

Runs 接受简单的 `input` 字符串，以及可选的 `session_id`、`instructions`、`conversation_history` 或 `previous_response_id`。当提供 `session_id` 时，Hermes 会在 run 状态中暴露它，以便外部 UI 将 run 与自己的对话 ID 关联。

### GET /v1/runs/\{run_id\}

轮询当前 run 状态。适用于需要状态但不想保持 SSE 连接的仪表板，或在导航后重新连接的 UI。

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

状态在终态（`completed`、`failed` 或 `cancelled`）之后会短暂保留，以供轮询和 UI 对账使用。

### GET /v1/runs/\{run_id\}/events

run 的工具调用进度、token 增量和生命周期事件的 Server-Sent Events 流。专为需要附加/分离而不丢失状态的仪表板和厚客户端设计。

### POST /v1/runs/\{run_id\}/stop

中断正在运行的 agent 轮次。端点立即返回 `{"status": "stopping"}`，同时 Hermes 要求活跃 agent 在下一个安全中断点停止。

## Jobs API（后台计划任务）

服务器暴露了一个轻量级 jobs CRUD 接口，用于从远程客户端管理计划/后台 agent run。所有端点均受同一 bearer 认证保护。

### GET /api/jobs

列出所有计划任务。

### POST /api/jobs

创建新的计划任务。请求体接受与 `hermes cron` 相同的结构——prompt（提示词）、schedule（计划）、skills（技能）、provider 覆盖、投递目标。

### GET /api/jobs/\{job_id\}

获取单个任务的定义和最后一次运行状态。

### PATCH /api/jobs/\{job_id\}

更新现有任务的字段（prompt、schedule 等）。部分更新会被合并。

### DELETE /api/jobs/\{job_id\}

删除任务。同时取消任何正在进行的 run。

### POST /api/jobs/\{job_id\}/pause

暂停任务而不删除它。下次计划运行的时间戳将被挂起，直到恢复。

### POST /api/jobs/\{job_id\}/resume

恢复之前暂停的任务。

### POST /api/jobs/\{job_id\}/run

立即触发任务运行，不受计划限制。

## 系统 Prompt 处理

当前端发送 `system` 消息（Chat Completions）或 `instructions` 字段（Responses API）时，hermes-agent 会将其**叠加在**核心系统 prompt 之上。你的 agent 保留所有工具、记忆和技能——前端的系统 prompt 只是添加额外指令。

这意味着你可以按前端自定义行为，而不会失去能力：
- Open WebUI 系统 prompt："You are a Python expert. Always include type hints."
- agent 仍然拥有终端、文件工具、网络搜索、记忆等。

## 认证

通过 `Authorization` 请求头进行 Bearer token 认证：

```
Authorization: Bearer ***
```

通过 `API_SERVER_KEY` 环境变量配置密钥。如果需要浏览器直接调用 Hermes，还需将 `API_SERVER_CORS_ORIGINS` 设置为明确的允许列表。

:::warning 安全
API 服务器提供对 hermes-agent 工具集的完整访问权限，**包括终端命令**。当绑定到非回环地址（如 `0.0.0.0`）时，**必须**设置 `API_SERVER_KEY`。同时保持 `API_SERVER_CORS_ORIGINS` 范围尽量小，以控制浏览器访问。

默认绑定地址（`127.0.0.1`）仅供本地使用。浏览器访问默认禁用；仅为明确的可信来源启用。
:::

## 配置

### 环境变量

| 变量 | 默认值 | 描述 |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | 启用 API 服务器 |
| `API_SERVER_PORT` | `8642` | HTTP 服务器端口 |
| `API_SERVER_HOST` | `127.0.0.1` | 绑定地址（默认仅限本地） |
| `API_SERVER_KEY` | _（无）_ | 认证用 Bearer token |
| `API_SERVER_CORS_ORIGINS` | _（无）_ | 逗号分隔的允许浏览器来源 |
| `API_SERVER_MODEL_NAME` | _（profile 名称）_ | `/v1/models` 上的模型名称。默认为 profile 名称，默认 profile 则为 `hermes-agent`。 |

### config.yaml

```yaml
# 暂不支持——请使用环境变量。
# config.yaml 支持将在未来版本中推出。
```

## 安全响应头

所有响应均包含安全响应头：
- `X-Content-Type-Options: nosniff` — 防止 MIME 类型嗅探
- `Referrer-Policy: no-referrer` — 防止 referrer 泄露

## CORS

API 服务器默认**不**启用浏览器 CORS。

如需直接浏览器访问，请设置明确的允许列表：

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

启用 CORS 后：
- **预检响应**包含 `Access-Control-Max-Age: 600`（10 分钟缓存）
- **SSE 流式响应**包含 CORS 头，使浏览器 EventSource 客户端能够正常工作
- **`Idempotency-Key`** 是允许的请求头——客户端可发送它用于去重（响应按 key 缓存 5 分钟）

大多数已记录的前端（如 Open WebUI）采用服务器到服务器连接，完全不需要 CORS。

## 兼容前端

任何支持 OpenAI API 格式的前端均可使用。已测试/记录的集成：

| 前端 | Stars | 连接方式 |
|----------|-------|------------|
| [Open WebUI](/user-guide/messaging/open-webui) | 126k | 提供完整指南 |
| LobeChat | 73k | 自定义 provider 端点 |
| LibreChat | 34k | librechat.yaml 中的自定义端点 |
| AnythingLLM | 56k | 通用 OpenAI provider |
| NextChat | 87k | BASE_URL 环境变量 |
| ChatBox | 39k | API Host 设置 |
| Jan | 26k | 远程模型配置 |
| HF Chat-UI | 8k | OPENAI_BASE_URL |
| big-AGI | 7k | 自定义端点 |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | 直接 HTTP 请求 |

## 使用 Profiles 的多用户设置

要为多个用户提供各自隔离的 Hermes 实例（独立的配置、记忆、技能），请使用 [profiles](/user-guide/profiles)：

```bash
# 为每个用户创建 profile
hermes profile create alice
hermes profile create bob

# 在不同端口上配置每个 profile 的 API 服务器。API_SERVER_* 是环境变量
# （不是 config.yaml 键），因此将它们写入每个 profile 的 .env：
cat >> ~/.hermes/profiles/alice/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8643
API_SERVER_KEY=alice-secret
EOF

cat >> ~/.hermes/profiles/bob/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8644
API_SERVER_KEY=bob-secret
EOF

# 启动每个 profile 的 gateway
hermes -p alice gateway &
hermes -p bob gateway &
```

每个 profile 的 API 服务器自动将 profile 名称作为模型 ID 广播：

- `http://localhost:8643/v1/models` → 模型 `alice`
- `http://localhost:8644/v1/models` → 模型 `bob`

在 Open WebUI 中，将每个添加为单独的连接。模型下拉列表显示 `alice` 和 `bob` 作为不同模型，每个均由完全隔离的 Hermes 实例支持。详见 [Open WebUI 指南](/user-guide/messaging/open-webui#multi-user-setup-with-profiles)。

## 限制

- **响应存储** — 存储的响应（用于 `previous_response_id`）持久化在 SQLite 中，gateway 重启后仍然存在。最多存储 100 个响应（LRU 淘汰）。
- **不支持文件上传** — 两个端点（`/v1/chat/completions` 和 `/v1/responses`）均支持内联图像，但不支持通过 API 上传文件（`file`、`input_file`、`file_id`）和非图像文档输入。
- **model 字段仅为展示用途** — 请求中的 `model` 字段会被接受，但实际使用的 LLM 模型在服务端的 config.yaml 中配置。

## 代理模式

API 服务器还作为 **gateway 代理模式**的后端。当另一个 Hermes gateway 实例配置了指向此 API 服务器的 `GATEWAY_PROXY_URL` 时，它会将所有消息转发到这里，而不是运行自己的 agent。这支持分离部署——例如，一个处理 Matrix E2EE 的 Docker 容器将请求中继到宿主机侧的 agent。

完整设置指南参见 [Matrix 代理模式](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)。