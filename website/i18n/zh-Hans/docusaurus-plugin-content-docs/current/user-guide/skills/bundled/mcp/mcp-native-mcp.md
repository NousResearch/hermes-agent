---
title: "Native Mcp — MCP 客户端：连接服务器、注册工具（stdio/HTTP）"
sidebar_label: "Native Mcp"
description: "MCP 客户端：连接服务器、注册工具（stdio/HTTP）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Native Mcp

MCP 客户端：连接服务器、注册工具（stdio/HTTP）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mcp/native-mcp` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `MCP`, `Tools`, `Integrations` |
| 相关 skill | [`mcporter`](/user-guide/skills/optional/mcp/mcp-mcporter) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Native MCP 客户端

Hermes Agent 内置了一个 MCP 客户端，它在启动时连接到 MCP 服务器，发现其工具，并将其作为一等工具直接提供给 agent 调用。无需桥接 CLI——来自 MCP 服务器的工具与 `terminal`、`read_file` 等内置工具并列显示。

## 使用场景

在以下情况下使用此 skill：
- 连接到 MCP 服务器并在 Hermes Agent 中使用其工具
- 通过 MCP 添加外部能力（文件系统访问、GitHub、数据库、API）
- 运行基于 stdio 的本地 MCP 服务器（npx、uvx 或任意命令）
- 连接到远程 HTTP/StreamableHTTP MCP 服务器
- 让 MCP 工具自动发现并在每次对话中可用

如需从终端进行临时、一次性的 MCP 工具调用而无需任何配置，请改用 `mcporter` skill。

## 前置条件

- **mcp Python 包** — 可选依赖；通过 `pip install mcp` 安装。若未安装，MCP 支持将静默禁用。
- **Node.js** — 基于 `npx` 的 MCP 服务器（大多数社区服务器）所需
- **uv** — 基于 `uvx` 的 MCP 服务器（Python 服务器）所需

安装 MCP SDK：

```bash
pip install mcp
# 或者，如果使用 uv：
uv pip install mcp
```

## 快速开始

在 `~/.hermes/config.yaml` 的 `mcp_servers` 键下添加 MCP 服务器：

```yaml
mcp_servers:
  time:
    command: "uvx"
    args: ["mcp-server-time"]
```

重启 Hermes Agent。启动时它将：
1. 连接到服务器
2. 发现可用工具
3. 以 `mcp_time_*` 前缀注册它们
4. 将其注入所有平台工具集

之后即可自然地使用这些工具——只需让 agent 获取当前时间即可。

## 配置参考

`mcp_servers` 下的每个条目是一个服务器名称到其配置的映射。有两种传输类型：**stdio**（基于命令）和 **HTTP**（基于 url）。

### Stdio 传输（command + args）

```yaml
mcp_servers:
  server_name:
    command: "npx"             # （必填）要运行的可执行文件
    args: ["-y", "pkg-name"]   # （可选）命令参数，默认：[]
    env:                       # （可选）子进程的环境变量
      SOME_API_KEY: "value"
    timeout: 120               # （可选）每次工具调用超时（秒），默认：120
    connect_timeout: 60        # （可选）初始连接超时（秒），默认：60
```

### HTTP 传输（url）

```yaml
mcp_servers:
  server_name:
    url: "https://my-server.example.com/mcp"   # （必填）服务器 URL
    headers:                                     # （可选）HTTP 请求头
      Authorization: "Bearer sk-..."
    timeout: 180               # （可选）每次工具调用超时（秒），默认：120
    connect_timeout: 60        # （可选）初始连接超时（秒），默认：60
```

### 所有配置选项

| 选项              | 类型   | 默认值  | 描述                                              |
|-------------------|--------|---------|---------------------------------------------------|
| `command`         | string | --      | 要运行的可执行文件（stdio 传输，必填）            |
| `args`            | list   | `[]`    | 传递给命令的参数                                  |
| `env`             | dict   | `{}`    | 子进程的额外环境变量                              |
| `url`             | string | --      | 服务器 URL（HTTP 传输，必填）                     |
| `headers`         | dict   | `{}`    | 每次请求发送的 HTTP 请求头                        |
| `timeout`         | int    | `120`   | 每次工具调用超时（秒）                            |
| `connect_timeout` | int    | `60`    | 初始连接和发现的超时时间                          |

注意：服务器配置必须有 `command`（stdio）或 `url`（HTTP）之一，不能同时存在。

## 工作原理

### 启动发现

Hermes Agent 启动时，`discover_mcp_tools()` 在工具初始化期间被调用：

1. 从 `~/.hermes/config.yaml` 读取 `mcp_servers`
2. 对每个服务器，在专用后台事件循环中生成连接
3. 初始化 MCP 会话并调用 `list_tools()` 发现可用工具
4. 在 Hermes 工具注册表中注册每个工具

### 工具命名规范

MCP 工具按以下命名模式注册：

```
mcp_{server_name}_{tool_name}
```

名称中的连字符和点号会替换为下划线，以兼容 LLM API。

示例：
- 服务器 `filesystem`，工具 `read_file` → `mcp_filesystem_read_file`
- 服务器 `github`，工具 `list-issues` → `mcp_github_list_issues`
- 服务器 `my-api`，工具 `fetch.data` → `mcp_my_api_fetch_data`

### 自动注入

发现完成后，MCP 工具会自动注入所有 `hermes-*` 平台工具集（CLI、Discord、Telegram 等）。这意味着 MCP 工具无需任何额外配置即可在每次对话中使用。

### 连接生命周期

- 每个服务器作为长期存活的 asyncio Task 运行在后台守护线程中
- 连接在 agent 进程的整个生命周期内持续存在
- 若连接断开，将自动以指数退避方式重连（最多重试 5 次，最大退避 60 秒）
- agent 关闭时，所有连接将优雅关闭

### 幂等性

`discover_mcp_tools()` 是幂等的——多次调用只会连接尚未连接的服务器。失败的服务器将在后续调用时重试。

## 传输类型

### Stdio 传输

最常见的传输方式。Hermes 将 MCP 服务器作为子进程启动，并通过 stdin/stdout 通信。

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
```

子进程继承**经过过滤的**环境（见下方安全章节）以及你在 `env` 中指定的任何变量。

### HTTP / StreamableHTTP 传输

用于远程或共享 MCP 服务器。要求 `mcp` 包包含 HTTP 客户端支持（`mcp.client.streamable_http`）。

```yaml
mcp_servers:
  remote_api:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer sk-..."
```

如果你安装的 `mcp` 版本不支持 HTTP 客户端，该服务器将以 ImportError 失败，其他服务器将正常继续运行。

## 安全

### 环境变量过滤

对于 stdio 服务器，Hermes **不会**将你的完整 shell 环境传递给 MCP 子进程。只有以下安全基线变量会被继承：

- `PATH`、`HOME`、`USER`、`LANG`、`LC_ALL`、`TERM`、`SHELL`、`TMPDIR`
- 所有 `XDG_*` 变量

所有其他环境变量（API 密钥、token、密钥等）均被排除，除非你通过 `env` 配置键显式添加。这可防止凭据意外泄露给不受信任的 MCP 服务器。

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      # 只有此 token 会传递给子进程
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
```

### 错误消息中的凭据脱敏

若 MCP 工具调用失败，错误消息中任何类似凭据的模式都会在展示给 LLM 之前自动脱敏。涵盖：

- GitHub PAT（`ghp_...`）
- OpenAI 风格密钥（`sk-...`）
- Bearer token
- 通用的 `token=`、`key=`、`API_KEY=`、`password=`、`secret=` 模式

## 故障排查

### "MCP SDK not available -- skipping MCP tool discovery"

`mcp` Python 包未安装。请安装：

```bash
pip install mcp
```

### "No MCP servers configured"

`~/.hermes/config.yaml` 中没有 `mcp_servers` 键，或该键为空。请至少添加一个服务器。

### "Failed to connect to MCP server 'X'"

常见原因：
- **命令未找到**：`command` 指定的二进制文件不在 PATH 中。请确保 `npx`、`uvx` 或相关命令已安装。
- **包未找到**：对于 npx 服务器，npm 包可能不存在，或需要在 args 中加入 `-y` 以自动安装。
- **超时**：服务器启动耗时过长。请增大 `connect_timeout`。
- **端口冲突**：对于 HTTP 服务器，URL 可能无法访问。

### "MCP server 'X' requires HTTP transport but mcp.client.streamable_http is not available"

你安装的 `mcp` 包版本不包含 HTTP 客户端支持。请升级：

```bash
pip install --upgrade mcp
```

### 工具未出现

- 检查服务器是否列在 `mcp_servers` 下（而非 `mcp` 或 `servers`）
- 确保 YAML 缩进正确
- 查看 Hermes Agent 启动日志中的连接信息
- 工具名称以 `mcp_{server}_{tool}` 为前缀——请查找该模式

### 连接持续断开

客户端以指数退避方式最多重试 5 次（1s、2s、4s、8s、16s，上限 60s）。若服务器根本无法访问，5 次尝试后将放弃。请检查服务器进程和网络连通性。

## 示例

### 时间服务器（uvx）

```yaml
mcp_servers:
  time:
    command: "uvx"
    args: ["mcp-server-time"]
```

注册如 `mcp_time_get_current_time` 等工具。

### 文件系统服务器（npx）

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"]
    timeout: 30
```

注册如 `mcp_filesystem_read_file`、`mcp_filesystem_write_file`、`mcp_filesystem_list_directory` 等工具。

### 带认证的 GitHub 服务器

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxxxxxxxxxx"
    timeout: 60
```

注册如 `mcp_github_list_issues`、`mcp_github_create_pull_request` 等工具。

### 远程 HTTP 服务器

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.mycompany.com/v1/mcp"
    headers:
      Authorization: "Bearer sk-xxxxxxxxxxxxxxxxxxxx"
      X-Team-Id: "engineering"
    timeout: 180
    connect_timeout: 30
```

### 多服务器

```yaml
mcp_servers:
  time:
    command: "uvx"
    args: ["mcp-server-time"]

  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxxxxxxxxxx"

  company_api:
    url: "https://mcp.internal.company.com/mcp"
    headers:
      Authorization: "Bearer sk-xxxxxxxxxxxxxxxxxxxx"
    timeout: 300
```

所有服务器的所有工具同时注册并可用。每个服务器的工具以其名称为前缀，避免冲突。

## Sampling（服务器发起的 LLM 请求）

Hermes 支持 MCP 的 `sampling/createMessage` 能力——MCP 服务器可在工具执行期间通过 agent 请求 LLM 补全。这支持 agent-in-the-loop 工作流（数据分析、内容生成、决策制定）。

Sampling **默认启用**。可按服务器配置：

```yaml
mcp_servers:
  my_server:
    command: "npx"
    args: ["-y", "my-mcp-server"]
    sampling:
      enabled: true           # 默认：true
      model: "gemini-3-flash" # 模型覆盖（可选）
      max_tokens_cap: 4096    # 每次请求最大 token 数
      timeout: 30             # LLM 调用超时（秒）
      max_rpm: 10             # 每分钟最大请求数
      allowed_models: []      # 模型白名单（空 = 全部允许）
      max_tool_rounds: 5      # 工具循环上限（0 = 禁用）
      log_level: "info"       # 审计日志详细程度
```

服务器还可以在 sampling 请求中包含 `tools`，用于多轮工具增强工作流。`max_tool_rounds` 配置可防止无限工具循环。每个服务器的审计指标（请求数、错误数、token 数、工具使用次数）通过 `get_mcp_status()` 追踪。

对不受信任的服务器，可通过 `sampling: { enabled: false }` 禁用 sampling。

## 注意事项

- MCP 工具从 agent 角度同步调用，但在专用后台事件循环上异步运行
- 工具结果以 JSON 形式返回，格式为 `{"result": "..."}` 或 `{"error": "..."}`
- native MCP 客户端与 `mcporter` 相互独立——可同时使用两者
- 服务器连接在同一 agent 进程的所有对话中持久共享
- 添加或移除服务器需要重启 agent（当前不支持热重载）