---
sidebar_position: 14
title: "API 服务器"
description: "将 hermes-agent 作为兼容 OpenAI 的 API 暴露给任何前端"
---

# API 服务器

API 服务器将 hermes-agent 暴露为兼容 OpenAI 的 HTTP 端点。任何使用 OpenAI 格式的前端 —— Open WebUI、LobeChat、LibreChat、NextChat、ChatBox 等 —— 都可以连接到 hermes-agent 并将其用作后端。

您的智能体使用其完整工具集（终端、文件操作、网页搜索、记忆、技能）处理请求并返回最终响应。流式传输时，工具进度指示器以内联方式显示，以便前端可以展示智能体正在做什么。

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

您将看到：

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. 连接前端

将任何兼容 OpenAI 的客户端指向 `http://localhost:8642/v1`：

```bash
# 使用 curl 测试
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change...dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

或连接 Open WebUI、LobeChat 或任何其他前端 —— 请参阅 [Open WebUI 集成指南](/user-guide/messaging/open-webui) 获取分步说明。

## 端点

### POST /v1/chat/completions

标准 OpenAI 聊天补全格式。无状态 —— 通过 `messages` 数组在每次请求中包含完整对话。

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

上传的文件 (`file` / `input_file` / `file_id`) 和非图像 `data:` URL 返回 `400 unsupported_content_type`。

**流式传输** (`"stream": true`)：返回带有逐 token 响应块的 Server-Sent Events (SSE)。对于**聊天补全**，流使用标准 `chat.completion.chunk` 事件以及 Hermes 的自定义 `hermes.tool.progress` 事件用于工具启动 UX。对于**响应**，流使用 OpenAI 响应事件类型，例如 `response.created`、`response.output_text.delta`、`response.output_item.added`、`response.output_item.done` 和 `response.completed`。

**流中的工具进度**：
- **聊天补全**：Hermes 发出 `event: hermes.tool.progress` 用于工具启动可见性，而不会污染持久化的助手文本。
- **响应**：Hermes 在 SSE 流期间发出规范原生的 `function_call` 和 `function_call_output` 输出项，因此客户端可以实时渲染结构化工具 UI。

### POST /v1/responses

OpenAI 响应 API 格式。支持通过 `previous_response_id` 进行服务端对话状态 —— 服务器存储完整对话历史（包括工具调用和结果），因此多轮上下文无需客户端管理即可保留。

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

上传的文件 (`input_file` / `file_id`) 和非图像 `data:` URL 返回 `400 unsupported_content_type`。

#### 使用 previous_response_id 进行多轮对话

链式响应以在多次交互中保持完整上下文（包括工具调用）：

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

服务器从存储的响应链重建完整对话 —— 所有先前的工具调用和结果都被保留。链式请求还共享相同的会话，因此多轮对话在仪表板和会话历史记录中显示为单个条目。

#### 命名对话

使用 `conversation` 参数代替跟踪响应 ID：

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

服务器自动链接到该对话中的最新响应。类似于网关会话的 `/title` 命令。

### GET /v1/responses/\{id\}

通过 ID 检索先前存储的响应。

### DELETE /v1/responses/\{id\}

删除存储的响应。

### GET /v1/models

将智能体列为可用模型。宣传的模型名称默认为[配置文件](/user-guide/profiles)名称（或默认配置文件的 `hermes-agent`）。大多数前端需要此端点进行模型发现。

### GET /v1/capabilities

返回 API 服务器稳定表面的机器可读描述，供外部 UI、编排器和插件桥接器使用。

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

在集成仪表板、浏览器 UI 或控制平面时使用此端点，以便它们可以发现运行的 Hermes 版本是否支持运行、流式传输、取消和会话连续性，而无需依赖私有 Python 内部实现。

### GET /health

健康检查。返回 `{"status": "ok"}`。对于期望 `/v1/` 前缀的兼容 OpenAI 客户端，也可在 **GET /v1/health** 获取。

### GET /health/detailed

扩展健康检查，还报告活动会话、运行中的智能体和资源使用情况。对监控/可观测性工具很有用。

## 运行 API（流式友好的替代方案）

除了 `/v1/chat/completions` 和 `/v1/responses`，服务器还暴露了一个**运行** API，用于客户端希望订阅进度事件而非自行管理流式传输的长会话。

### POST /v1/runs

创建新的智能体运行。返回可用于订阅进度事件的 `run_id`。

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

运行接受简单的 `input` 字符串和可选的 `session_id`、`instructions`、`conversation_history` 或 `previous_response_id`。当提供 `session_id` 时，Hermes 在运行状态中显示它，以便外部 UI 可以将运行与其自己的对话 ID 关联。

### GET /v1/runs/\{run_id\}

轮询当前运行状态。这对于需要状态而不保持 SSE 连接打开的仪表板很有用，或对于导航后重新连接的 UI。

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

状态在终止状态（`completed`、`failed` 或 `cancelled`）后短暂保留，用于轮询和 UI 协调。

### GET /v1/runs/\{run_id\}/events

运行的工具调用进度、token 增量和生命周期事件的 Server-Sent Events 流。专为想要在不丢失状态的情况下附加/分离的仪表板和厚客户端设计。

### POST /v1/runs/\{run_id\}/stop

中断正在运行的智能体回合。端点立即返回 `{"status": "stopping"}`，同时 Hermes 要求活动智能体在下一个安全中断点停止。

## 任务 API（后台计划工作）

服务器暴露了一个轻量级的任务 CRUD 接口，用于从远程客户端管理计划/后台智能体运行。所有端点都在相同的 bearer 认证之后。

### GET /api/jobs

列出所有计划任务。

### POST /api/jobs

创建新的计划任务。请求体接受与 `hermes cron` 相同的形状 —— 提示、计划、技能、提供商覆盖、交付目标。

### GET /api/jobs/\{job_id\}

获取单个任务的定义和上次运行状态。

### PATCH /api/jobs/\{job_id\}

更新现有任务的字段（提示、计划等）。部分更新被合并。

### DELETE /api/jobs/\{job_id\}

删除任务。同时取消任何正在进行的运行。

### POST /api/jobs/\{job_id\}/pause

暂停任务而不删除它。下次计划运行时间戳被挂起，直到恢复。

### POST /api/jobs/\{job_id\}/resume

恢复先前暂停的任务。

### POST /api/jobs/\{job_id\}/run

立即触发任务运行，不按计划。

## 系统提示处理

当前端发送 `system` 消息（聊天补全）或 `instructions` 字段（响应 API）时，hermes-agent **将其叠加**在其核心系统提示之上。您的智能体保留所有工具、记忆和技能 —— 前端的系统提示添加额外指令。

这意味着您可以按前端自定义行为而不会丢失功能：
- Open WebUI 系统提示："You are a Python expert. Always include type hints."
- 智能体仍然拥有终端、文件工具、网页搜索、记忆等。

## 认证

通过 `Authorization` 标头进行 Bearer token 认证：

```
Authorization: Bearer ***
```

通过 `API_SERVER_KEY` 环境变量配置密钥。如果您需要浏览器直接调用 Hermes，还将 `API_SERVER_CORS_ORIGINS` 设置为显式允许列表。

:::warning 安全
API 服务器提供对 hermes-agent 工具集的完全访问，**包括终端命令**。当绑定到 `0.0.0.0` 等非环回地址时，`API_SERVER_KEY` **是必需的**。还要保持 `API_SERVER_CORS_ORIGINS` 狭窄以控制浏览器访问。

默认绑定地址 (`127.0.0.1`) 仅供本地使用。默认情况下禁用浏览器访问；仅为显式受信任的来源启用它。
:::

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | 启用 API 服务器 |
| `API_SERVER_PORT` | `8642` | HTTP 服务器端口 |
| `API_SERVER_HOST` | `127.0.0.1` | 绑定地址（默认仅限本地主机） |
| `API_SERVER_KEY` | _(无)_ | 认证用的 Bearer token |
| `API_SERVER_CORS_ORIGINS` | _(无)_ | 逗号分隔的允许浏览器来源 |
| `API_SERVER_MODEL_NAME` | _(配置文件名称)_ | `/v1/models` 上的模型名称。默认为配置文件名称，或默认配置文件的 `hermes-agent`。 |

### config.yaml

```yaml
# 尚不支持 —— 使用环境变量。
# 未来版本将支持 config.yaml。
```

## 安全标头

所有响应都包含安全标头：
- `X-Content-Type-Options: nosniff` —— 防止 MIME 类型嗅探
- `Referrer-Policy: no-referrer` —— 防止 referrer 泄漏

## CORS

API 服务器默认**不**启用浏览器 CORS。

对于直接浏览器访问，设置显式允许列表：

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

启用 CORS 时：
- **预检响应** 包含 `Access-Control-Max-Age: 600`（10 分钟缓存）
- **SSE 流式响应** 包含 CORS 标头，以便浏览器 EventSource 客户端正常工作
- **`Idempotency-Key`** 是允许的请求标头 —— 客户端可以发送它进行去重（响应按键缓存 5 分钟）

大多数记录的前端（如 Open WebUI）采用服务器到服务器连接，根本不需要 CORS。

## 兼容的前端

任何支持 OpenAI API 格式的前端都有效。已测试/记录的集成：

| 前端 | Stars | 连接 |
|----------|-------|------------|
| [Open WebUI](/user-guide/messaging/open-webui) | 126k | 提供完整指南 |
| LobeChat | 73k | 自定义提供商端点 |
| LibreChat | 34k | librechat.yaml 中的自定义端点 |
| AnythingLLM | 56k | 通用 OpenAI 提供商 |
| NextChat | 87k | BASE_URL 环境变量 |
| ChatBox | 39k | API Host 设置 |
| Jan | 26k | 远程模型配置 |
| HF Chat-UI | 8k | OPENAI_BASE_URL |
| big-AGI | 7k | 自定义端点 |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | 直接 HTTP 请求 |

## 使用配置文件的多用户设置

要为多个用户提供各自隔离的 Hermes 实例（单独配置、记忆、技能），请使用[配置文件](/user-guide/profiles)：

```bash
# 为每个用户创建配置文件
hermes profile create alice
hermes profile create bob

# 在每个配置文件的不同端口上配置 API 服务器。API_SERVER_* 是环境
# 变量（不是 config.yaml 键），因此将它们写入每个配置文件的 .env：
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

# 启动每个配置文件的网关
hermes -p alice gateway &
hermes -p bob gateway &
```

每个配置文件的 API 服务器自动将配置文件名称宣传为模型 ID：

- `http://localhost:8643/v1/models` → 模型 `alice`
- `http://localhost:8644/v1/models` → 模型 `bob`

在 Open WebUI 中，将每个添加为单独连接。模型下拉菜单将 `alice` 和 `bob` 显示为不同的模型，每个都由完全隔离的 Hermes 实例支持。有关详细信息，请参阅 [Open WebUI 指南](/user-guide/messaging/open-webui#multi-user-setup-with-profiles)。

## 限制

- **响应存储** —— 存储的响应（用于 `previous_response_id`）在 SQLite 中持久化，并在网关重启后保留。最多 100 个存储的响应（LRU 驱逐）。
- **无文件上传** —— `/v1/chat/completions` 和 `/v1/responses` 都支持内联图像，但不支持上传的文件（`file`、`input_file`、`file_id`）和非图像文档输入。
- **模型字段是装饰性的** —— 请求中的 `model` 字段被接受，但实际使用的 LLM 模型在 config.yaml 中服务端配置。

## 代理模式

API 服务器还作为**网关代理模式**的后端。当另一个 Hermes 网关实例配置 `GATEWAY_PROXY_URL` 指向此 API 服务器时，它会将所有消息转发到这里，而不是运行自己的智能体。这支持拆分部署 —— 例如，处理 Matrix E2EE 的 Docker 容器，中继到主机端智能体。

请参阅 [Matrix 代理模式](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos) 获取完整设置指南。
