---
sidebar_position: 4
title: "MCP（模型上下文协议）"
description: "通过 MCP 将 Hermes Agent 连接到外部工具服务器，并精确控制 Hermes 加载哪些 MCP 工具"
---

# MCP（模型上下文协议）

本页的 Python 依赖命令使用 [PM 准备的源码环境](../../reference/package-management.md#developer-workflow)。
依赖变更后，请重新激活该 checkout 并重启 Hermes。

MCP 让 Hermes Agent 连接到外部工具服务器，使 agent 能够使用 Hermes 本身之外的工具——GitHub、数据库、文件系统、浏览器栈、内部 API 等等。

如果你曾经希望 Hermes 使用某个已经存在于其他地方的工具，MCP 通常是最简洁的方式。

## MCP 能给你带来什么

- 无需先编写原生 Hermes 工具，即可访问外部工具生态系统
- 在同一配置中同时支持本地 stdio 服务器和远程 HTTP MCP 服务器
- 启动时自动发现并注册工具
- 在服务器支持的情况下，提供针对 MCP 资源和 prompt（提示词）的实用工具封装
- 按服务器过滤，只向 Hermes 暴露你真正需要的 MCP 工具

## 快速开始

1. 安装 MCP 支持（如果你使用了标准安装脚本，已包含在内）：

```bash
cd ~/.hermes/hermes-agent
python -c "import pm; pm.sync_venv(['mcp'], explicit=True)"
```

2. 在 `~/.hermes/config.yaml` 中添加一个 MCP 服务器：

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

4. 让 Hermes 使用 MCP 支持的能力。

例如：

```text
List the files in /home/user/projects and summarize the repo structure.
```

Hermes 会发现 MCP 服务器的工具，并像使用其他工具一样使用它们。

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

适合使用 stdio 服务器的场景：
- 服务器已在本地安装
- 需要低延迟访问本地资源
- 你参考的 MCP 服务器文档中使用了 `command`、`args` 和 `env`

### HTTP 服务器

HTTP MCP 服务器是 Hermes 直接连接的远程端点。

```yaml
mcp_servers:
  remote_api:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer ***"
```

适合使用 HTTP 服务器的场景：
- MCP 服务器托管在其他地方
- 你的组织暴露了内部 MCP 端点
- 你不希望 Hermes 为该集成在本地启动子进程

## 基本配置参考

Hermes 从 `~/.hermes/config.yaml` 的 `mcp_servers` 下读取 MCP 配置。

### 常用字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `command` | string | stdio MCP 服务器的可执行文件 |
| `args` | list | stdio 服务器的参数 |
| `env` | mapping | 传递给 stdio 服务器的环境变量 |
| `url` | string | HTTP MCP 端点 |
| `headers` | mapping | 远程服务器的 HTTP 头 |
| `timeout` | number | 工具调用超时时间 |
| `connect_timeout` | number | 初始连接超时时间 |
| `enabled` | bool | 若为 `false`，Hermes 完全跳过该服务器 |
| `supports_parallel_tool_calls` | bool | 若为 `true`，该服务器的工具可并发运行 |
| `tools` | mapping | 按服务器过滤工具及实用工具策略 |

### 最简 stdio 示例

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### 最简 HTTP 示例

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
```

## 内置预设

对于知名 MCP 服务器，`hermes mcp add` 接受 `--preset` 标志，自动填写传输层细节，无需手动查找命令和参数。预设只提供默认值——你在同一命令行传入的其他内容（环境变量、头信息、过滤规则）仍然优先生效。

| 预设 | 配置内容 |
|---|---|
| `codex` | Codex CLI 的 MCP 服务器（通过 stdio 运行 `codex mcp-server`）。需要 PATH 中存在 `codex` CLI。 |

```bash
# 一行命令将 Codex CLI 添加为 MCP 服务器
hermes mcp add codex --preset codex
```

等价于写入：

```yaml
mcp_servers:
  codex:
    command: "codex"
    args: ["mcp-server"]
```

你可以使用任意本地名称（`hermes mcp add my-codex --preset codex` 完全可以）；预设只提供 `command`/`args` 默认值。

## Hermes 注册 MCP 工具的方式

Hermes 为 MCP 工具添加前缀，避免与内置名称冲突：

```text
mcp_<server_name>_<tool_name>
```

示例：

| 服务器 | MCP 工具 | 注册名称 |
|---|---|---|
| `filesystem` | `read_file` | `mcp_filesystem_read_file` |
| `github` | `create-issue` | `mcp_github_create_issue` |
| `my-api` | `query.data` | `mcp_my_api_query_data` |

实际使用中，你通常不需要手动调用带前缀的名称——Hermes 在正常推理过程中会自动识别并选择该工具。

## MCP 实用工具

在服务器支持的情况下，Hermes 还会围绕 MCP 资源和 prompt 注册实用工具：

- `list_resources`
- `read_resource`
- `list_prompts`
- `get_prompt`

这些工具按服务器注册，遵循相同的前缀规则，例如：

- `mcp_github_list_resources`
- `mcp_github_get_prompt`

### 重要说明

这些实用工具现在具备能力感知：
- 只有当 MCP 会话实际支持资源操作时，Hermes 才注册资源实用工具
- 只有当 MCP 会话实际支持 prompt 操作时，Hermes 才注册 prompt 实用工具

因此，一个只暴露可调用工具而没有资源/prompt 的服务器，不会获得这些额外的封装。

## 按服务器过滤

你可以控制每个 MCP 服务器向 Hermes 贡献哪些工具，从而精细管理工具命名空间。

### 完全禁用某个服务器

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

若 `enabled: false`，Hermes 完全跳过该服务器，甚至不尝试连接。

### 白名单过滤服务器工具

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

只有列出的 MCP 服务器工具会被注册。

### 黑名单过滤服务器工具

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

除排除项外，所有服务器工具均被注册。

### 优先级规则

若两者同时存在：

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

`include` 优先生效。

### 同样可过滤实用工具

你也可以单独禁用 Hermes 添加的实用工具封装：

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

含义：
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

## 如果所有工具都被过滤掉会怎样？

如果你的配置过滤掉了所有可调用工具，并禁用或省略了所有支持的实用工具，Hermes 不会为该服务器创建空的运行时 MCP 工具集。

这样可以保持工具列表整洁。

## 运行时行为

### 发现时机

Hermes 在启动时发现 MCP 服务器，并将其工具注册到普通工具注册表中。

### 动态工具发现

MCP 服务器可以在运行时通过发送 `notifications/tools/list_changed` 通知，告知 Hermes 其可用工具发生了变化。Hermes 收到该通知后，会自动重新获取服务器的工具列表并更新注册表——无需手动执行 `/reload-mcp`。

这对于能力动态变化的 MCP 服务器非常有用（例如，加载新数据库 schema 时添加工具，或服务下线时移除工具）。

刷新操作受锁保护，因此同一服务器快速连续发送的通知不会导致重叠刷新。prompt 和资源变更通知（`prompts/list_changed`、`resources/list_changed`）会被接收，但暂未处理。

### 重新加载

如果你修改了 MCP 配置，请使用：

```text
/reload-mcp
```

这会从配置重新加载 MCP 服务器并刷新可用工具列表。对于服务器主动推送的运行时工具变更，请参阅上方的[动态工具发现](#dynamic-tool-discovery)。

### 工具集

每个已配置的 MCP 服务器，在贡献至少一个已注册工具时，也会创建一个运行时工具集：

```text
mcp-<server>
```

这使得在工具集层面更容易理解 MCP 服务器的情况。

## 安全模型

### Stdio 环境变量过滤

对于 stdio 服务器，Hermes 不会盲目传递你的完整 shell 环境。

只有显式配置的 `env` 加上安全基线才会被传递。这减少了意外泄露密钥的风险。

### 配置层面的暴露控制

新的过滤支持同时也是一种安全控制：
- 禁用你不希望模型看到的危险工具
- 对敏感服务器只暴露最小白名单
- 在不需要暴露该接口时，禁用资源/prompt 封装

## 示例用例

### GitHub 服务器，仅暴露最小 issue 管理接口

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

使用方式：

```text
Show me open issues labeled bug, then draft a new issue for the flaky MCP reconnection behavior.
```

### Stripe 服务器，移除危险操作

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

使用方式：

```text
Look up the last 10 failed payments and summarize common failure reasons.
```

### 文件系统服务器，限定单个项目根目录

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

使用方式：

```text
Inspect the project root and explain the directory layout.
```

## 故障排查

### MCP 服务器无法连接

检查：

```bash
# 验证 MCP 依赖已安装（标准安装已包含）
cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['mcp'], explicit=True)"

node --version
npx --version
```

然后验证你的配置并重启 Hermes。

### 工具未出现

可能原因：
- 服务器连接失败
- 发现过程失败
- 你的过滤配置排除了这些工具
- 该服务器不存在对应的实用工具能力
- 服务器通过 `enabled: false` 被禁用

如果你是有意过滤，这是预期行为。

### 为什么资源或 prompt 实用工具没有出现？

因为 Hermes 现在只在以下两个条件同时满足时才注册这些封装：
1. 你的配置允许它们
2. 服务器会话实际支持该能力

这是有意为之，保持工具列表的真实性。

## 并行工具调用

默认情况下，MCP 工具按顺序执行——一次一个。如果你的 MCP 服务器暴露的工具可以安全并发运行（例如只读查询、独立 API 调用），可以选择启用并行执行：

```yaml
mcp_servers:
  docs:
    command: "docs-server"
    supports_parallel_tool_calls: true
```

当 `supports_parallel_tool_calls` 为 `true` 时，Hermes 可能在单次工具调用批次中同时执行该服务器的多个工具，就像对内置只读工具（`web_search`、`read_file` 等）的处理方式一样。

:::caution
只对工具可以安全同时运行的 MCP 服务器启用并行调用。如果工具会读写共享状态、文件、数据库或外部资源，请在启用此设置前仔细评估读写竞争条件。
:::

## MCP Sampling 支持

MCP 服务器可以通过 `sampling/createMessage` 协议向 Hermes 请求 LLM 推理。这允许 MCP 服务器代表自己请求 Hermes 生成文本——适用于需要 LLM 能力但没有自己模型访问权限的服务器。

Sampling 对所有 MCP 服务器**默认启用**（当 MCP SDK 支持时）。可在 `sampling` 键下按服务器配置：

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    sampling:
      enabled: true            # 启用 sampling（默认：true）
      model: "openai/gpt-4o"  # 覆盖 sampling 请求使用的模型（可选）
      max_tokens_cap: 4096     # 每次 sampling 响应的最大 token 数（默认：4096）
      timeout: 30              # 每次请求的超时时间，单位秒（默认：30）
      max_rpm: 10              # 速率限制：每分钟最大请求数（默认：10）
      max_tool_rounds: 5       # sampling 循环中的最大工具调用轮数（默认：5）
      allowed_models: []       # 服务器可请求的模型名称白名单（空 = 不限）
      log_level: "info"        # 审计日志级别：debug、info 或 warning（默认：info）
```

sampling 处理器包含滑动窗口速率限制器、按请求超时和工具循环深度限制，防止失控使用。每个服务器实例会跟踪指标（请求数、错误数、已用 token 数）。

如需对特定服务器禁用 sampling：

```yaml
mcp_servers:
  untrusted_server:
    url: "https://mcp.example.com"
    sampling:
      enabled: false
```

## 将 Hermes 作为 MCP 服务器运行

Hermes 也可以**作为** MCP 服务器运行，让 Claude Code、Claude Desktop、Cursor、Codex 或任何 MCP 客户端列出和读取你的消息会话、等待新消息、跨所有已连接平台发送消息，并可选地调用 Hermes 自己的网页、浏览器、视觉、图像生成、技能和 TTS 工具。该功能现由官方 **mcp-serve** 插件提供：

```bash
hermes plugins install mcp-serve
```

```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp-serve"]
    }
  }
}
```

它以前内置为 `hermes mcp serve`。配置了 `"args": ["mcp", "serve"]` 的 MCP 客户端需改为 `["mcp-serve"]`；工具名称和参数保持不变，`hermes mcp serve` 现在会提示新的写法和安装命令。工具列表、`--tools` 模式及其安全限制见[插件 README](https://github.com/NousResearch/hermes-official-plugins/tree/main/mcp-serve)。

## 相关文档

- [在 Hermes 中使用 MCP](../../guides/use-mcp-with-hermes.md)
- [CLI 命令](../../reference/cli-commands.md)
- [斜杠命令](../../reference/slash-commands.md)
- [常见问题](../../reference/faq.md)