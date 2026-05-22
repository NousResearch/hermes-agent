---
sidebar_position: 4
title: "MCP（模型上下文协议）"
description: "通过 MCP 将 Hermes Agent 连接到外部工具服务器，并精确控制 Hermes 加载哪些 MCP 工具"
---

# MCP（模型上下文协议）

MCP 允许 Hermes Agent 连接到外部工具服务器，使代理能够使用运行在 Hermes 之外的工具——例如 GitHub、数据库、文件系统、浏览器集群、内部 API 等。

如果你希望 Hermes 使用已经存在于其他位置的工具，MCP 通常是最简洁的实现方式。

## MCP 为你带来什么

- 在不先编写原生 Hermes 工具的情况下访问外部工具生态
- 在同一配置中支持本地 stdio 服务器与远程 HTTP MCP 服务器
- 启动时自动发现并注册工具
- 当服务器支持时提供 MCP 资源与提示的实用封装
- 每个服务器的过滤策略，允许你只向 Hermes 暴露所需的工具

## 快速上手

1. 安装 MCP 支持（如果使用标准安装脚本，通常已包含）：

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
```

2. 在 `~/.hermes/config.yaml` 中添加 MCP 服务器：

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
```

3. 启动 Hermes：

```bash
hermes chat
```

4. 请求 Hermes 使用 MCP 提供的能力，例如：

```text
列出 /home/user/projects 下的文件并总结仓库结构。
```

Hermes 会发现 MCP 服务器的工具并像使用其他工具一样调用它们。

## 两类 MCP 服务器

### Stdio 服务器

Stdio 服务器作为本地子进程运行，通过 stdin/stdout 通信。

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
```

适用场景：本地已安装服务器、需要低延迟访问本地资源、或参考服务器文档使用 `command`/`args`/`env`。

### HTTP 服务器

HTTP MCP 服务器是 Hermes 直接连接的远程端点。

```yaml
mcp_servers:
  remote_api:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer ***"
```

适用场景：服务器托管在他处、组织内部暴露的 MCP 端点、或不希望 Hermes 启动本地子进程。

## 基本配置参考

Hermes 从 `~/.hermes/config.yaml` 的 `mcp_servers` 读取 MCP 配置。

### 常用字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `command` | string | stdio MCP 服务器的可执行命令 |
| `args` | list | 传给 stdio 服务器的参数 |
| `env` | mapping | 传给 stdio 服务器的环境变量 |
| `url` | string | HTTP MCP 端点 |
| `headers` | mapping | 远程服务器的 HTTP 头 |
| `timeout` | number | 工具调用超时 |
| `connect_timeout` | number | 初始连接超时 |
| `enabled` | bool | 若为 `false` 则跳过该服务器 |
| `tools` | mapping | 每服务器的工具过滤与策略 |

### 最小 stdio 示例

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### 最小 HTTP 示例

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
```

## Hermes 如何注册 MCP 工具

Hermes 会对 MCP 工具加前缀以避免与内置工具冲突：

```text
mcp_<server_name>_<tool_name>
```

示例：`mcp_filesystem_read_file`、`mcp_github_create_issue`。

一般情况下无需手动调用带前缀的名称——Hermes 在推理时会选择合适的工具。

## MCP 实用工具

当服务器支持时，Hermes 还会注册围绕 MCP 资源与提示的实用工具：

- `list_resources`
- `read_resource`
- `list_prompts`
- `get_prompt`

这些工具按相同前缀模式注册，例如 `mcp_github_list_resources`。

### 注意

- 仅当 MCP 会话确实支持资源操作时，Hermes 才会注册资源实用工具。
- 仅当服务器支持提示操作时，才会注册提示相关工具。

## 每服务器过滤

你可以控制每个 MCP 服务器贡献给 Hermes 的工具，从而精细管理工具命名空间。

### 完全禁用某个服务器

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

### 白名单工具

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [create_issue, list_issues]
```

### 黑名单工具

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

### 优先级规则

若同时存在 `include` 与 `exclude`：`include` 优先。

### 也可以过滤实用工具

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

`tools.resources: false` 将禁用 `list_resources` / `read_resource`，`tools.prompts: false` 将禁用 `list_prompts` / `get_prompt`。

### 完整示例

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, search_code]
      prompts: false

  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer]
      resources: false

  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

## 若全部被过滤会怎样？

如果你的配置过滤掉了所有可调用工具并禁用了实用工具，Hermes 不会为该服务器创建空的运行时 MCP 工具集，从而保持工具列表简洁。

## 运行时行为

### 发现时机

Hermes 在启动时发现 MCP 服务器并将其工具注册到常规工具注册表中。

### 动态工具发现

MCP 服务器可以在运行时通过 `notifications/tools/list_changed` 通知 Hermes 工具列表发生改变。Hermes 收到通知后会自动重新获取该服务器的工具列表并更新注册表——无需手动 `/reload-mcp`。

刷新操作受锁保护，防止来自同一服务器的速发通知导致并发刷新。提示与资源变更通知目前可接收，但尚未完全触发对应自动处理。

### 重新加载

如果你修改了 MCP 配置，使用：

```text
/reload-mcp
```

可从配置重新加载 MCP 服务器并刷新可用工具列表。对于服务器主动推送的运行时变更，请参见“动态工具发现”。

### 工具集

每个贡献了至少一个已注册工具的 MCP 服务器也会创建一个运行时工具集：

```text
mcp-<server>
```

这使得从工具集角度理解 MCP 服务器更容易。

## 安全模型

### Stdio 环境过滤

对于 stdio 服务器，Hermes 不会盲目传递完整的 shell 环境。仅传递显式配置的 `env` 加上安全基线，减少意外泄露机密的风险。

### 配置级暴露控制

通过过滤支持你可以：
- 禁用可能危险的工具
- 仅暴露最小白名单以保护敏感服务器
- 在不希望暴露资源/提示时禁用相应封装

## 示例用例

（此处略去若干示例以保持文档简洁 — 英文原文包含更多实例）

---
---
sidebar_position: 4
title: "MCP (Model Context Protocol)"
description: "通过 MCP 将 Hermes Agent 连接到外部工具服务器 —— 并精确控制 Hermes 加载哪些 MCP 工具"
---

# MCP (Model Context Protocol)

MCP 允许 Hermes Agent 连接到外部工具服务器，使智能体可以使用存在于 Hermes 本身之外的工具 —— GitHub、数据库、文件系统、浏览器栈、内部 API 等。

如果您曾经希望 Hermes 使用已存在于别处的工具，MCP 通常是最简洁的方式。

## MCP 能为您做什么

- 无需先编写原生 Hermes 工具即可访问外部工具生态系统
- 同一配置中支持本地 stdio 服务器和远程 HTTP MCP 服务器
- 启动时自动发现工具并注册
- 当服务器支持时，为 MCP 资源和提示提供实用包装器
- 按服务器过滤，因此您可以仅暴露您希望 Hermes 看到的 MCP 工具

## 快速开始

1. 安装 MCP 支持（如果您使用标准安装脚本，则已包含）：

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
```

2. 将 MCP 服务器添加到 `~/.hermes/config.yaml`：

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
```

3. 启动 Hermes：

```bash
hermes chat
```

4. 要求 Hermes 使用 MCP 支持的能力。

例如：

```text
列出 /home/user/projects 中的文件并总结仓库结构。
```

Hermes 将发现 MCP 服务器的工具并像使用任何其他工具一样使用它们。

## 两种 MCP 服务器

### Stdio 服务器

Stdio 服务器作为本地子进程运行，通过 stdin/stdout 通信。

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
```

在以下情况使用 stdio 服务器：
- 服务器已本地安装
- 您想要对本地资源的低延迟访问
- 您遵循显示 `command`、`args` 和 `env` 的 MCP 服务器文档

### HTTP 服务器

HTTP MCP 服务器是 Hermes 直接连接的远程端点。

```yaml
mcp_servers:
  remote_api:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer ***"
```

在以下情况使用 HTTP 服务器：
- MCP 服务器托管在其他地方
- 您的组织暴露内部 MCP 端点
- 您不希望 Hermes 为该集成生成本地子进程

## 基本配置参考

Hermes 从 `~/.hermes/config.yaml` 的 `mcp_servers` 下读取 MCP 配置。

### 常用键

| 键 | 类型 | 含义 |
|---|---|---|
| `command` | string | stdio MCP 服务器的可执行文件 |
| `args` | list | stdio 服务器的参数 |
| `env` | mapping | 传递给 stdio 服务器的环境变量 |
| `url` | string | HTTP MCP 端点 |
| `headers` | mapping | 远程服务器的 HTTP 标头 |
| `timeout` | number | 工具调用超时 |
| `connect_timeout` | number | 初始连接超时 |
| `enabled` | bool | 如果为 `false`，Hermes 完全跳过该服务器 |
| `tools` | mapping | 按服务器工具过滤和实用策略 |

### 最小 stdio 示例

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### 最小 HTTP 示例

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
```

## Hermes 如何注册 MCP 工具

Hermes 为 MCP 工具添加前缀，使其不会与内置名称冲突：

```text
mcp_<server_name>_<tool_name>
```

示例：

| 服务器 | MCP 工具 | 注册名称 |
|---|---|---|
| `filesystem` | `read_file` | `mcp_filesystem_read_file` |
| `github` | `create-issue` | `mcp_github_create_issue` |
| `my-api` | `query.data` | `mcp_my_api_query_data` |

实际上，您通常不需要手动调用带前缀的名称 —— Hermes 看到工具并在正常推理期间选择它。

## MCP 实用工具

当受支持时，Hermes 还会注册围绕 MCP 资源和提示的实用工具：

- `list_resources`
- `read_resource`
- `list_prompts`
- `get_prompt`

这些按服务器以相同的前缀模式注册，例如：

- `mcp_github_list_resources`
- `mcp_github_get_prompt`

### 重要

这些实用工具现在是能力感知的：
- 仅当 MCP 会话实际支持资源操作时，Hermes 才会注册资源实用工具
- 仅当 MCP 会话实际支持提示操作时，Hermes 才会注册提示实用工具

因此，暴露可调用工具但没有资源/提示的服务器不会获得这些额外的包装器。

## 按服务器过滤

您可以控制每个 MCP 服务器向 Hermes 贡献哪些工具，从而对工具命名空间进行细粒度管理。

### 完全禁用服务器

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

如果 `enabled: false`，Hermes 完全跳过该服务器，甚至不尝试连接。

### 白名单服务器工具

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [create_issue, list_issues]
```

仅注册这些 MCP 服务器工具。

### 黑名单服务器工具

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

注册除被排除之外的所有服务器工具。

### 优先级规则

如果两者都存在：

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

`include` 胜出。

### 也过滤实用工具

您还可以单独禁用 Hermes 添加的实用包装器：

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

这意味着：
- `tools.resources: false` 禁用 `list_resources` 和 `read_resource`
- `tools.prompts: false` 禁用 `list_prompts` 和 `get_prompt`

### 完整示例

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [create_issue, list_issues, search_code]
      prompts: false

  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer]
      resources: false

  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

## 如果所有内容都被过滤掉会怎样？

如果您的配置过滤掉所有可调用工具并禁用或省略所有支持的实用工具，Hermes 不会为该服务器创建空的运行时 MCP 工具集。

这保持工具列表整洁。

## 运行时行为

### 发现时间

Hermes 在启动时发现 MCP 服务器并将其工具注册到正常工具注册表中。

### 动态工具发现

MCP 服务器可以通过发送 `notifications/tools/list_changed` 通知来通知 Hermes 其可用工具在运行时发生变化。当 Hermes 收到此通知时，它会自动重新获取服务器的工具列表并更新注册表 —— 无需手动 `/reload-mcp`。

这对于能力动态变化的 MCP 服务器很有用（例如，当加载新数据库模式时添加工具的服务器，或当服务离线时移除工具的服务器）。

刷新受锁保护，因此来自同一服务器的快速连续通知不会导致重叠刷新。提示和资源更改通知 (`prompts/list_changed`、`resources/list_changed`) 被接收但尚未处理。

### 重新加载

如果您更改 MCP 配置，请使用：

```text
/reload-mcp
```

这会从配置重新加载 MCP 服务器并刷新可用工具列表。对于服务器自身推送的运行时工具更改，请参阅上面的[动态工具发现](#动态工具发现)。

### 工具集

每个配置的 MCP 服务器在贡献至少一个注册工具时还会创建一个运行时工具集：

```text
mcp-<server>
```

这使 MCP 服务器在工具集级别更容易理解。

## 安全模型

### Stdio 环境过滤

对于 stdio 服务器，Hermes 不会盲目传递您的完整 shell 环境。

仅显式配置的 `env` 加上安全基线被传递。这减少了意外密钥泄漏。

### 配置级暴露控制

新的过滤支持也是一种安全控制：
- 禁用您不希望模型看到的危险工具
- 为敏感服务器暴露最小白名单
- 当您不希望暴露该表面时，禁用资源/提示包装器

## 示例用例

### 具有最小 issue 管理表面的 GitHub 服务器

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue]
      prompts: false
      resources: false
```

像这样使用：

```text
显示标记为 bug 的开放 issue，然后为不稳定的 MCP 重连行为起草一个新 issue。
```

### 移除危险操作的 Stripe 服务器

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

像这样使用：

```text
查找最近 10 次失败的付款并总结常见失败原因。
```

### 用于单个项目根目录的文件系统服务器

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

像这样使用：

```text
检查项目根目录并解释目录布局。
```

## 故障排除

### MCP 服务器未连接

检查：

```bash
# 验证 MCP 依赖是否已安装（标准安装中已包含）
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

node --version
npx --version
```

然后验证您的配置并重新启动 Hermes。

### 工具未出现

可能的原因：
- 服务器连接失败
- 发现失败
- 您的过滤配置排除了工具
- 该服务器不存在实用能力
- 服务器被 `enabled: false` 禁用

如果您有意过滤，这是预期的。

### 为什么资源或提示实用工具没有出现？

因为 Hermes 现在仅在两者都为真时才注册这些包装器：
1. 您的配置允许它们
2. 服务器会话实际支持该能力

这是有意为之，保持工具列表诚实。

## MCP 采样支持

MCP 服务器可以通过 `sampling/createMessage` 协议从 Hermes 请求 LLM 推理。这允许 MCP 服务器要求 Hermes 代表其生成文本 —— 对于需要 LLM 能力但没有自己模型访问权限的服务器很有用。

当 MCP SDK 支持时，采样**默认对所有 MCP 服务器启用**。在 `sampling` 键下按服务器配置：

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    sampling:
      enabled: true            # 启用采样（默认：true）
      model: "openai/gpt-4o"  # 覆盖采样请求的模型（可选）
      max_tokens_cap: 4096     # 每次采样响应的最大 token 数（默认：4096）
      timeout: 30              # 每次请求的超时秒数（默认：30）
      max_rpm: 10              # 速率限制：每分钟最大请求数（默认：10）
      max_tool_rounds: 5       # 采样循环中的最大工具使用轮数（默认：5）
      allowed_models: []       # 服务器可请求的模型名称允许列表（空 = 任何）
      log_level: "info"        # 审计日志级别：debug、info 或 warning（默认：info）
```

采样处理程序包括滑动窗口速率限制器、每次请求超时和工具循环深度限制以防止失控使用。指标（请求计数、错误、使用的 token）按服务器实例跟踪。

要为特定服务器禁用采样：

```yaml
mcp_servers:
  untrusted_server:
    url: "https://mcp.example.com"
    sampling:
      enabled: false
```

## 将 Hermes 作为 MCP 服务器运行

除了连接**到** MCP 服务器外，Hermes 还可以**成为** MCP 服务器。这让其他支持 MCP 的智能体（Claude Code、Cursor、Codex 或任何 MCP 客户端）可以使用 Hermes 的消息传递能力 —— 列出对话、阅读消息历史记录，并在所有已连接平台上发送消息。

### 何时使用

- 您希望 Claude Code、Cursor 或另一个编码智能体通过 Hermes 发送和阅读 Telegram/Discord/Slack 消息
- 您希望单个 MCP 服务器同时桥接到 Hermes 的所有已连接消息平台
- 您已有一个运行中的 Hermes 网关，连接了平台

### 快速开始

```bash
hermes mcp serve
```

这会启动一个 stdio MCP 服务器。MCP 客户端（而非您）管理进程生命周期。

### MCP 客户端配置

将 Hermes 添加到您的 MCP 客户端配置。例如，在 Claude Code 的 `~/.claude/claude_desktop_config.json` 中：

```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

或者如果您在特定位置安装了 Hermes：

```json
{
  "mcpServers": {
    "hermes": {
      "command": "/home/user/.hermes/hermes-agent/venv/bin/hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

### 可用工具

MCP 服务器暴露 10 个工具，匹配 OpenClaw 的频道桥接表面加上 Hermes 特定的频道浏览器：

| 工具 | 说明 |
|------|-------------|
| `conversations_list` | 列出活跃的消息对话。按平台过滤或按名称搜索。 |
| `conversation_get` | 通过会话键获取单个对话的详细信息。 |
| `messages_read` | 阅读对话的近期消息历史记录。 |
| `attachments_fetch` | 从特定消息中提取非文本附件（图像、媒体）。 |
| `events_poll` | 自光标位置以来轮询新对话事件。 |
| `events_wait` | 长轮询 / 阻塞直到下一个事件到达（近实时）。 |
| `messages_send` | 通过平台发送消息（例如 `telegram:123456`、`discord:#general`）。 |
| `channels_list` | 列出所有平台上的可用消息传递目标。 |
| `permissions_list_open` | 列出此桥接会话期间观察到的待处理批准请求。 |
| `permissions_respond` | 允许或拒绝待处理的批准请求。 |

### 事件系统

MCP 服务器包含一个实时事件桥接器，轮询 Hermes 的会话数据库以获取新消息。这让 MCP 客户端近实时地感知传入对话：

```
# 轮询新事件（非阻塞）
events_poll(after_cursor=0)

# 等待下一个事件（阻塞直到超时）
events_wait(after_cursor=42, timeout_ms=30000)
```

事件类型：`message`、`approval_requested`、`approval_resolved`

事件队列是内存中的，在桥接器连接时启动。旧消息可通过 `messages_read` 获取。

### 选项

```bash
hermes mcp serve              # 正常模式
hermes mcp serve --verbose    # stderr 上的调试日志
```

### 工作原理

MCP 服务器直接从 Hermes 的会话存储（`~/.hermes/sessions/sessions.json` 和 SQLite 数据库）读取对话数据。后台线程轮询数据库以获取新消息并维护内存中的事件队列。对于发送消息，它使用与 Hermes 智能体本身相同的 `send_message` 基础设施。

网关**不需要**为读取操作运行（列出对话、阅读历史记录、轮询事件）。它**需要**为发送操作运行，因为平台适配器需要活动连接。

### 当前限制

- 仅 stdio 传输（尚无 HTTP MCP 传输）
- 通过 mtime 优化的 DB 轮询以约 200ms 间隔进行事件轮询（文件未更改时跳过工作）
- 尚无 `claude/channel` 推送通知协议
- 仅文本发送（无通过 `messages_send` 的媒体/附件发送）

## 相关文档

- [将 MCP 与 Hermes 一起使用](/guides/use-mcp-with-hermes)
- [CLI 命令](/reference/cli-commands)
- [斜杠命令](/reference/slash-commands)
- [常见问题](/reference/faq)
