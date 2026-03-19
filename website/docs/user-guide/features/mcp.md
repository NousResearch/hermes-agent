---
sidebar_position: 4
title: "MCP (Model Context Protocol)"
description: "Connect Hermes Agent to external tool servers via MCP — and control exactly which MCP tools Hermes loads"
---

# MCP (Model Context Protocol)

MCP lets Hermes Agent connect to external tool servers so the agent can use tools that live outside Hermes itself — GitHub, databases, file systems, browser stacks, internal APIs, and more.

If you have ever wanted Hermes to use a tool that already exists somewhere else, MCP is usually the cleanest way to do it.

## What MCP gives you

- Access to external tool ecosystems without writing a native Hermes tool first
- Local stdio servers and remote HTTP MCP servers in the same config
- Automatic tool discovery and registration at startup
- Utility wrappers for MCP resources and prompts when supported by the server
- Per-server filtering so you can expose only the MCP tools you actually want Hermes to see

## Quick start

### Option A: Use the CLI (recommended)

The `hermes mcp` command adds servers with interactive tool discovery:

```bash
# HTTP server
hermes mcp add ink --url "https://mcp.ml.ink/mcp"

# Stdio server
hermes mcp add github --command npx --args -y @modelcontextprotocol/server-github

# OAuth-authenticated server
hermes mcp add my-api --url "https://api.example.com/mcp" --auth oauth
```

Hermes connects to the server, discovers available tools, and lets you select which ones to enable before saving to config.

### Option B: Edit config manually

1. Install MCP support (already included if you used the standard install script):

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
```

2. Add an MCP server to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
```

3. Start Hermes — tools are auto-discovered:

```bash
hermes chat
```

### Try it

```text
List the files in /home/user/projects and summarize the repo structure.
```

Hermes will discover the MCP server's tools and use them like any other tool.

## Two kinds of MCP servers

### Stdio servers

Stdio servers run as local subprocesses and talk over stdin/stdout.

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
```

Use stdio servers when:
- the server is installed locally
- you want low-latency access to local resources
- you are following MCP server docs that show `command`, `args`, and `env`

### HTTP servers

HTTP MCP servers are remote endpoints Hermes connects to directly.

```yaml
mcp_servers:
  remote_api:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer ***"
```

Use HTTP servers when:
- the MCP server is hosted elsewhere
- your organization exposes internal MCP endpoints
- you do not want Hermes spawning a local subprocess for that integration

### OAuth-authenticated servers

For servers that require OAuth 2.1 (e.g., third-party SaaS APIs), Hermes handles the full PKCE flow automatically:

```yaml
mcp_servers:
  my-api:
    url: "https://api.example.com/mcp"
    auth: oauth
```

Or via the CLI:

```bash
hermes mcp add my-api --url "https://api.example.com/mcp" --auth oauth
```

On first connection, Hermes opens your browser for authorization. Tokens are cached in `~/.hermes/mcp-tokens/` with restrictive permissions (`0600`) and auto-refreshed when they expire.

You can combine OAuth with static headers for servers that need both:

```yaml
mcp_servers:
  my-api:
    url: "https://api.example.com/mcp"
    auth: oauth
    headers:
      X-Custom-Header: "value"
```

## Basic configuration reference

Hermes reads MCP config from `~/.hermes/config.yaml` under `mcp_servers`.

### Common keys

| Key | Type | Meaning |
|---|---|---|
| `command` | string | Executable for a stdio MCP server |
| `args` | list | Arguments for the stdio server |
| `env` | mapping | Environment variables passed to the stdio server |
| `url` | string | HTTP MCP endpoint |
| `headers` | mapping | HTTP headers for remote servers |
| `auth` | string | Authentication type: `oauth` for OAuth 2.1 PKCE |
| `timeout` | number | Tool call timeout |
| `connect_timeout` | number | Initial connection timeout |
| `enabled` | bool | If `false`, Hermes skips the server entirely |
| `tools` | mapping | Per-server tool filtering and utility policy |

### Minimal stdio example

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### Minimal HTTP example

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
```

### Environment variable interpolation

You can reference environment variables from `~/.hermes/.env` in config values using `${VAR}` syntax:

```yaml
mcp_servers:
  ink:
    url: "https://mcp.ml.ink/mcp"
    headers:
      Authorization: "Bearer ${INK_API_KEY}"
```

This keeps secrets out of `config.yaml`. The `hermes mcp add` command sets this up automatically when you provide an API key.

## How Hermes registers MCP tools

Hermes prefixes MCP tools so they do not collide with built-in names:

```text
mcp_<server_name>_<tool_name>
```

Examples:

| Server | MCP tool | Registered name |
|---|---|---|
| `filesystem` | `read_file` | `mcp_filesystem_read_file` |
| `github` | `create-issue` | `mcp_github_create_issue` |
| `my-api` | `query.data` | `mcp_my_api_query_data` |

In practice, you usually do not need to call the prefixed name manually — Hermes sees the tool and chooses it during normal reasoning.

## MCP utility tools

When supported, Hermes also registers utility tools around MCP resources and prompts:

- `list_resources`
- `read_resource`
- `list_prompts`
- `get_prompt`

These are registered per server with the same prefix pattern, for example:

- `mcp_github_list_resources`
- `mcp_github_get_prompt`

### Important

These utility tools are now capability-aware:
- Hermes only registers resource utilities if the MCP session actually supports resource operations
- Hermes only registers prompt utilities if the MCP session actually supports prompt operations

So a server that exposes callable tools but no resources/prompts will not get those extra wrappers.

## Per-server filtering

This is the main feature added by the PR work.

You can now control which tools each MCP server contributes to Hermes.

### Disable a server entirely

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

If `enabled: false`, Hermes skips the server completely and does not even attempt a connection.

### Whitelist server tools

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

Only those MCP server tools are registered.

### Blacklist server tools

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

All server tools are registered except the excluded ones.

### Precedence rule

If both are present:

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

`include` wins.

### Filter utility tools too

You can also separately disable Hermes-added utility wrappers:

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

That means:
- `tools.resources: false` disables `list_resources` and `read_resource`
- `tools.prompts: false` disables `list_prompts` and `get_prompt`

### Full example

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

## What happens if everything is filtered out?

If your config filters out all callable tools and disables or omits all supported utilities, Hermes does not create an empty runtime MCP toolset for that server.

That keeps the tool list clean.

## Runtime behavior

### Discovery time

Hermes discovers MCP servers at startup and registers their tools into the normal tool registry.

### Reloading

If you change MCP config, use:

```text
/reload-mcp
```

This reloads MCP servers from config and refreshes the available tool list.

### Toolsets

Each configured MCP server also creates a runtime toolset when it contributes at least one registered tool:

```text
mcp-<server>
```

That makes MCP servers easier to reason about at the toolset level.

## Security model

### Stdio env filtering

For stdio servers, Hermes does not blindly pass your full shell environment.

Only explicitly configured `env` plus a safe baseline are passed through. This reduces accidental secret leakage.

### Config-level exposure control

The new filtering support is also a security control:
- disable dangerous tools you do not want the model to see
- expose only a minimal whitelist for a sensitive server
- disable resource/prompt wrappers when you do not want that surface exposed

## Example use cases

### GitHub server with a minimal issue-management surface

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

Use it like:

```text
Show me open issues labeled bug, then draft a new issue for the flaky MCP reconnection behavior.
```

### Stripe server with dangerous actions removed

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

Use it like:

```text
Look up the last 10 failed payments and summarize common failure reasons.
```

### Filesystem server for a single project root

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

Use it like:

```text
Inspect the project root and explain the directory layout.
```

## Managing MCP servers with the CLI

The `hermes mcp` command provides a full lifecycle manager for MCP servers.

### `hermes mcp add`

Add a server with interactive tool discovery:

```bash
# HTTP server
hermes mcp add ink --url "https://mcp.ml.ink/mcp"

# Stdio server
hermes mcp add github --command npx --args -y @modelcontextprotocol/server-github

# OAuth-authenticated server
hermes mcp add my-api --url "https://api.example.com/mcp" --auth oauth
```

The command:
1. Connects to the server and discovers available tools
2. Shows the full tool list
3. Lets you enable all, select specific tools, or cancel
4. Saves the config to `~/.hermes/config.yaml`

For HTTP servers, it also prompts for authentication (API key or OAuth) and securely stores credentials in `~/.hermes/.env`.

### `hermes mcp list`

View all configured servers with transport, tool count, and status:

```bash
hermes mcp list
```

### `hermes mcp test`

Test a connection and verify tools are discoverable:

```bash
hermes mcp test ink
```

Shows connection latency, auth info (masked), and the full tool list.

### `hermes mcp configure`

Reconfigure which tools are enabled for a server using an interactive checklist:

```bash
hermes mcp configure ink
```

### `hermes mcp remove`

Remove a server from config and clean up OAuth tokens:

```bash
hermes mcp remove ink
```

## Troubleshooting

### MCP server not connecting

Check:

```bash
# Test the connection
hermes mcp test <server-name>

# Verify MCP deps are installed (already included in standard install)
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

node --version
npx --version
```

Then verify your config and restart Hermes.

### Tools not appearing

Possible causes:
- the server failed to connect
- discovery failed
- your filter config excluded the tools
- the utility capability does not exist on that server
- the server is disabled with `enabled: false`

If you are intentionally filtering, this is expected.

### Why didn't resource or prompt utilities appear?

Because Hermes now only registers those wrappers when both are true:
1. your config allows them
2. the server session actually supports the capability

This is intentional and keeps the tool list honest.

## Related docs

- [Use MCP with Hermes](/docs/guides/use-mcp-with-hermes)
- [MCP Config Reference](/docs/reference/mcp-config-reference)
- [CLI Commands](/docs/reference/cli-commands)
- [Slash Commands](/docs/reference/slash-commands)
- [FAQ](/docs/reference/faq)
