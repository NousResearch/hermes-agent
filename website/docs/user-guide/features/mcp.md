---
sidebar_position: 4
title: "MCP (Model Context Protocol)"
description: "Connect Hermes Agent to external tool servers via MCP — databases, APIs, filesystems, and more"
---

# MCP (Model Context Protocol)

MCP lets Hermes Agent connect to external tool servers — giving the agent access to databases, APIs, filesystems, and more without any code changes.

## Overview

The [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) is an open standard for connecting AI agents to external tools and data sources. MCP servers expose tools over a lightweight RPC protocol, and Hermes Agent can connect to any compliant server automatically.

What this means for you:

- **Thousands of ready-made tools** — browse the [MCP server directory](https://github.com/modelcontextprotocol/servers) for servers covering GitHub, Slack, databases, file systems, web scraping, and more
- **No code changes needed** — add a few lines to `~/.hermes/config.yaml` and the tools appear alongside built-in ones
- **Mix and match** — run multiple MCP servers simultaneously, combining stdio-based and HTTP-based servers
- **Secure by default** — environment variables are filtered and credentials are stripped from error messages

## Prerequisites

```bash
pip install hermes-agent[mcp]
```

| Server Type | Runtime Needed | Example |
|-------------|---------------|---------|
| HTTP/remote | Nothing extra | `url: "https://mcp.example.com"` |
| npm-based (npx) | Node.js 18+ | `command: "npx"` |
| Python-based | uv (recommended) | `command: "uvx"` |

## Configuration

MCP servers are configured in `~/.hermes/config.yaml` under the `mcp_servers` key.

### Stdio Servers

Stdio servers run as local subprocesses, communicating over stdin/stdout:

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    env: {}

  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxx"
```

| Key | Required | Description |
|-----|----------|-------------|
| `command` | Yes | Executable to run (`npx`, `uvx`, `python`) |
| `args` | No | Command-line arguments |
| `env` | No | Environment variables for the subprocess |

:::info Security
Only explicitly listed `env` variables plus a safe baseline (`PATH`, `HOME`, `USER`, `LANG`, `SHELL`, `TMPDIR`, `XDG_*`) are passed to the subprocess. Your API keys and secrets are **not** leaked.
:::

### HTTP Servers

```yaml
mcp_servers:
  remote_api:
    url: "https://my-mcp-server.example.com/mcp"
    headers:
      Authorization: "Bearer sk-xxxxxxxxxxxx"
```

### Per-Server Timeouts

```yaml
mcp_servers:
  slow_database:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      DATABASE_URL: "postgres://user:pass@localhost/mydb"
    timeout: 300          # Tool call timeout (default: 120s)
    connect_timeout: 90   # Initial connection timeout (default: 60s)
```

### Mixed Configuration Example

```yaml
mcp_servers:
  # Local filesystem via stdio
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

  # GitHub API via stdio with auth
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxx"

  # Remote database via HTTP
  company_db:
    url: "https://mcp.internal.company.com/db"
    headers:
      Authorization: "Bearer sk-xxxxxxxxxxxx"
    timeout: 180

  # Python-based server via uvx
  memory:
    command: "uvx"
    args: ["mcp-server-memory"]
```

## Translating from Claude Desktop Config

Many MCP server docs show Claude Desktop JSON format. Here's the translation:

**Claude Desktop JSON:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

**Hermes YAML:**
```yaml
mcp_servers:                          # mcpServers → mcp_servers (snake_case)
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

Rules: `mcpServers` → `mcp_servers` (snake_case), JSON → YAML. Keys like `command`, `args`, `env` are identical.

## How It Works

### Tool Registration

Each MCP tool is registered with a prefixed name:

```
mcp_{server_name}_{tool_name}
```

| Server Name | MCP Tool Name | Registered As |
|-------------|--------------|---------------|
| `filesystem` | `read_file` | `mcp_filesystem_read_file` |
| `github` | `create-issue` | `mcp_github_create_issue` |
| `my-api` | `query.data` | `mcp_my_api_query_data` |

Tools appear alongside built-in tools — the agent calls them like any other tool.

:::info
In addition to the server's own tools, each MCP server also gets 4 utility tools auto-registered: `list_resources`, `read_resource`, `list_prompts`, and `get_prompt`. These allow the agent to discover and use MCP resources and prompts exposed by the server.

Each configured server also creates a **runtime toolset** named `mcp-<server>`. This means you can filter or reason about MCP servers at the toolset level in the same way you do with built-in toolsets.
:::

### Reconnection

If an MCP server disconnects, Hermes automatically reconnects with exponential backoff (1s, 2s, 4s, 8s, 16s — max 5 attempts). Initial connection failures are reported immediately.

### Shutdown

On agent exit, all MCP server connections are cleanly shut down.

## Popular MCP Servers

| Server | Package | Description |
|--------|---------|-------------|
| Filesystem | `@modelcontextprotocol/server-filesystem` | Read/write/search local files |
| GitHub | `@modelcontextprotocol/server-github` | Issues, PRs, repos, code search |
| Git | `@modelcontextprotocol/server-git` | Git operations on local repos |
| Fetch | `@modelcontextprotocol/server-fetch` | HTTP fetching and web content |
| Memory | `@modelcontextprotocol/server-memory` | Persistent key-value memory |
| SQLite | `@modelcontextprotocol/server-sqlite` | Query SQLite databases |
| PostgreSQL | `@modelcontextprotocol/server-postgres` | Query PostgreSQL databases |
| Brave Search | `@modelcontextprotocol/server-brave-search` | Web search via Brave API |
| Puppeteer | `@modelcontextprotocol/server-puppeteer` | Browser automation |

### Example Configs

```yaml
mcp_servers:
  # No API key needed
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]

  git:
    command: "uvx"
    args: ["mcp-server-git", "--repository", "/home/user/my-repo"]

  fetch:
    command: "uvx"
    args: ["mcp-server-fetch"]

  sqlite:
    command: "uvx"
    args: ["mcp-server-sqlite", "--db-path", "/home/user/data.db"]

  # Requires API key
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxx"

  brave_search:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-brave-search"]
    env:
      BRAVE_API_KEY: "BSA_xxxxxxxxxxxx"
```

## Troubleshooting

### "MCP SDK not available"

```bash
pip install hermes-agent[mcp]
```

### Server fails to start

The MCP server command (`npx`, `uvx`) is not on PATH. Install the required runtime:

```bash
# For npm-based servers
npm install -g npx    # or ensure Node.js 18+ is installed

# For Python-based servers
pip install uv        # then use "uvx" as the command
```

### Server connects but tools fail with auth errors

Ensure the key is in the server's `env` block:

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_your_actual_token"  # Check this
```

### Connection timeout

Increase `connect_timeout` for slow-starting servers:

```yaml
mcp_servers:
  slow_server:
    command: "npx"
    args: ["-y", "heavy-server-package"]
    connect_timeout: 120   # default is 60
```

### Reload MCP Servers

You can reload MCP servers without restarting Hermes:

- In the CLI: the agent reconnects automatically
- In messaging: send `/reload-mcp`

## Sampling (Server-Initiated LLM Requests)

MCP's `sampling/createMessage` capability allows MCP servers to request LLM completions through the Hermes agent. This enables agent-in-the-loop workflows where servers can leverage the LLM during tool execution — for example, a database server asking the LLM to interpret query results, or a code analysis server requesting the LLM to review findings.

### How It Works

When an MCP server sends a `sampling/createMessage` request:

1. The sampling callback validates against rate limits and model whitelist
2. Resolves which model to use (config override > server hint > default)
3. Converts MCP messages to OpenAI-compatible format
4. Offloads the LLM call to a thread via `asyncio.to_thread()` (non-blocking)
5. Returns the response (text or tool use) back to the server

### Configuration

Sampling is **enabled by default** for all MCP servers. No extra setup needed — if you have an auxiliary LLM client configured, sampling works automatically.

```yaml
mcp_servers:
  analysis_server:
    command: "npx"
    args: ["-y", "my-analysis-server"]
    sampling:
      enabled: true           # default: true
      model: "gemini-3-flash" # override model (optional)
      max_tokens_cap: 4096    # max tokens per request (default: 4096)
      timeout: 30             # LLM call timeout in seconds (default: 30)
      max_rpm: 10             # max requests per minute (default: 10)
      allowed_models: []      # model whitelist (empty = allow all)
      max_tool_rounds: 5      # max consecutive tool use rounds (0 = disable)
      log_level: "info"       # audit verbosity: debug, info, warning
```

### Tool Use in Sampling

Servers can include `tools` and `toolChoice` in sampling requests, enabling multi-turn tool-augmented workflows within a single sampling session. The callback forwards tool definitions to the LLM, handles tool use responses with proper `ToolUseContent` types, and enforces `max_tool_rounds` to prevent infinite loops.

### Security

- **Rate limiting**: Per-server sliding window (default: 10 req/min)
- **Token cap**: Servers can't request more than `max_tokens_cap` (default: 4096)
- **Model whitelist**: `allowed_models` restricts which models a server can use
- **Tool loop limit**: `max_tool_rounds` caps consecutive tool use rounds
- **Credential stripping**: LLM responses are sanitized before returning to the server
- **Non-blocking**: LLM calls run in a separate thread via `asyncio.to_thread()`
- **Typed errors**: All failures return structured `ErrorData` per MCP spec

To disable sampling for untrusted servers:

```yaml
mcp_servers:
  untrusted:
    command: "npx"
    args: ["-y", "untrusted-server"]
    sampling:
      enabled: false
```
