---
sidebar_position: 8
title: "MCP 配置参考"
description: "Hermes Agent MCP 配置键、过滤语义及工具策略参考"
---

# MCP 配置参考

本页是主 MCP 文档的简明参考手册。

概念说明请参阅：
- [MCP（Model Context Protocol）](/user-guide/features/mcp)
- [在 Hermes 中使用 MCP](/guides/use-mcp-with-hermes)

## 根配置结构

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # stdio servers
    args: []
    env: {}

    # OR
    url: "..."          # HTTP servers
    headers: {}

    enabled: true
    timeout: 120
    connect_timeout: 60
    supports_parallel_tool_calls: false
    tools:
      include: []
      exclude: []
      resources: true
      prompts: true
```

## 服务器键

| 键 | 类型 | 适用范围 | 含义 |
|---|---|---|---|
| `command` | string | stdio | 要启动的可执行文件 |
| `args` | list | stdio | 子进程的参数 |
| `env` | mapping | stdio | 传递给子进程的环境变量 |
| `url` | string | HTTP | 远程 MCP 端点 |
| `headers` | mapping | HTTP | 远程服务器请求的请求头 |
| `enabled` | bool | 两者 | 为 false 时完全跳过该服务器 |
| `timeout` | number | 两者 | 工具调用超时时间 |
| `connect_timeout` | number | 两者 | 初始连接超时时间 |
| `supports_parallel_tool_calls` | bool | 两者 | 允许该服务器的工具并发执行 |
| `tools` | mapping | 两者 | 过滤及工具策略 |
| `auth` | string | HTTP | 认证方式。设为 `oauth` 可启用带 PKCE 的 OAuth 2.1 |
| `sampling` | mapping | 两者 | 服务器发起的 LLM 请求策略；默认关闭，除非显式设置 `sampling.enabled: true`（参见 MCP 指南） |

## `tools` 策略键

| 键 | 类型 | 含义 |
|---|---|---|
| `include` | string 或 list | 白名单：指定允许注册的服务器原生 MCP 工具 |
| `exclude` | string 或 list | 黑名单：指定禁止注册的服务器原生 MCP 工具 |
| `resources` | bool-like | 启用/禁用 `list_resources` + `read_resource` |
| `prompts` | bool-like | 启用/禁用 `list_prompts` + `get_prompt` |

## 过滤语义

### `include`

若设置了 `include`，则只注册其中列出的服务器原生 MCP 工具。

```yaml
tools:
  include: [create_issue, list_issues]
```

### `exclude`

若设置了 `exclude` 且未设置 `include`，则注册除列出名称之外的所有服务器原生 MCP 工具。

```yaml
tools:
  exclude: [delete_customer]
```

### 优先级

若两者同时设置，`include` 优先。

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

结果：
- `create_issue` 仍被允许
- `delete_issue` 被忽略，因为 `include` 优先级更高

## 工具策略

Hermes 可为每个 MCP 服务器注册以下工具包装器：

Resources（资源）：
- `list_resources`
- `read_resource`

Prompts（提示词）：
- `list_prompts`
- `get_prompt`

### 禁用 resources

```yaml
tools:
  resources: false
```

### 禁用 prompts

```yaml
tools:
  prompts: false
```

### 能力感知注册

即使设置了 `resources: true` 或 `prompts: true`，Hermes 也只在 MCP 会话实际暴露对应能力时才注册相应工具。

因此以下情况属于正常现象：
- 你启用了 prompts
- 但没有出现任何 prompt 工具
- 原因是该服务器不支持 prompts

## `enabled: false`

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

行为：
- 不发起连接
- 不进行服务发现
- 不注册工具
- 配置保留，供后续复用

## 空结果行为

若过滤后服务器原生工具全部被移除，且没有工具被注册，Hermes 不会为该服务器创建空的 MCP 运行时工具集。

## 配置示例

### GitHub 安全白名单

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      resources: false
      prompts: false
```

### Stripe 黑名单

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### 仅资源的文档服务器

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      include: []
      resources: true
      prompts: false
```

## 重新加载配置

修改 MCP 配置后，使用以下命令重新加载服务器：

```text
/reload-mcp
```

## 工具命名

服务器原生 MCP 工具的命名格式为：

```text
mcp_<server>_<tool>
```

示例：
- `mcp_github_create_issue`
- `mcp_filesystem_read_file`
- `mcp_my_api_query_data`

工具包装器遵循相同的前缀规则：
- `mcp_<server>_list_resources`
- `mcp_<server>_read_resource`
- `mcp_<server>_list_prompts`
- `mcp_<server>_get_prompt`

### 名称规范化

服务器名称和工具名称中的连字符（`-`）和点号（`.`）在注册前均会替换为下划线。这确保工具名称是 LLM function-calling API 的合法标识符。

例如，名为 `my-api` 的服务器暴露了名为 `list-items.v2` 的工具，注册后变为：

```text
mcp_my_api_list_items_v2
```

编写 `include` / `exclude` 过滤器时请注意——使用**原始** MCP 工具名称（含连字符/点号），而非规范化后的名称。

## OAuth 2.1 认证

对于需要 OAuth 的 HTTP 服务器，在服务器条目中设置 `auth: oauth`：

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
```

行为：
- Hermes 使用 MCP SDK 的 OAuth 2.1 PKCE 流程（元数据发现、动态客户端注册、token 交换及刷新）
- 首次连接时，浏览器窗口将打开以完成授权
- Token 持久化至 `~/.hermes/mcp-tokens/<server>.json`，跨会话复用
- Token 刷新自动进行；仅在刷新失败时才需重新授权
- 仅适用于 HTTP/StreamableHTTP 传输（基于 `url` 的服务器）
