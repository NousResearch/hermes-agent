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

1. MCP support ships with the standard install — no extra step needed.

2. Add an MCP server to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
```

3. Start Hermes:

```bash
hermes chat
```

4. Ask Hermes to use the MCP-backed capability.

For example:

```text
List the files in /home/user/projects and summarize the repo structure.
```

Hermes will discover the MCP server's tools and use them like any other tool.

## Catalog: one-click install for Nous-approved MCPs

Hermes ships a curated catalog of MCP servers that Nous staff has reviewed
and merged. They're disabled by default — install only what you actually
want.

```bash
hermes mcp                # interactive picker (default)
hermes mcp catalog        # plain-text list, scriptable
hermes mcp install n8n    # install a catalog entry by name
```

The picker shows each entry with its current status:

```
n8n          available              Manage and inspect n8n workflows from Hermes
linear       enabled                Linear issue/project management (remote OAuth)
github       installed (disabled)   GitHub repo + PR tools
```

Hit `Enter` on a row to install (and walk through any required credentials),
enable, disable, or uninstall. Catalog entries are stored under
`optional-mcps/` in the hermes-agent repo — presence in that directory means
Nous approval. There is no community submission tier; entries are added by
merging a PR.

Catalog entries can require:

- **API key** — Hermes prompts at install time and writes the value to
  `~/.hermes/.env`. Non-secret values (base URLs) go to the same file.
- **OAuth** (remote MCP) — written as `auth: oauth` in your config; the MCP
  client opens a browser on first connection.
- **OAuth** (third-party provider like Google/GitHub) — Hermes points you at
  `hermes auth <provider>` if you haven't authenticated already.

### Tool selection at install time

After credentials are configured, Hermes probes the MCP server to list every
tool it exposes and presents a checklist:

```
Select tools for 'linear' (SPACE toggle, ENTER confirm)
  [x] find_issues       Find issues matching a query
  [x] get_issue         Get a single issue
  [x] create_issue      Create a new issue
  [ ] delete_workspace  Delete a Linear workspace
  ...
```

The pre-checked rows come from:

1. **Your prior selection** if you've installed this entry before (reinstalls
   preserve what you had — the manifest's defaults don't override it)
2. **The manifest's `tools.default_enabled`** if the entry declares one (some
   catalog entries pre-prune mutating or rarely-useful tools)
3. **Everything** if neither applies

Submit the checklist with ENTER. Only the checked tools end up in
`mcp_servers.<name>.tools.include`. If you select everything, no filter is
written (cleanest config shape, identical behavior).

**If the probe fails** (server unreachable, OAuth not yet completed,
backing service not running), the install still succeeds: the manifest's
`tools.default_enabled` is applied directly (if declared), or no filter is
written (if not). Re-run `hermes mcp configure <name>` once the server is
reachable to refine.

### Trust model

Installing a catalog entry runs whatever the manifest specifies — `git clone`,
the entry's `bootstrap` commands (`pip install`, `npm install`, etc.), and
ultimately the MCP server's own code. Manifests are gated by PR review into
the hermes-agent repo, so Nous has reviewed each entry before it shipped —
**but you should still read the manifest before installing**, especially the
`source:` field's repository, the `install.bootstrap:` commands, and any
`transport.command:` invocation.

Manifests live at
[`optional-mcps/<name>/manifest.yaml`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps)
on GitHub. The picker also prints the manifest's `source:` URL at install
time so you can quickly verify the upstream repo. The web dashboard's MCP
page surfaces the same detail per catalog entry — transport, auth type, the
endpoint URL (HTTP) or command + args (stdio), the git install source/ref and
bootstrap commands, and setup notes — with the `source:` rendered as a
clickable link, so you can inspect exactly what an entry connects to or runs
before clicking Install.

### Manifest version compatibility

Manifests pin a `manifest_version`. The catalog is forward-compatible: if a
PR adds an entry with a newer `manifest_version` than your installed Hermes
understands, the picker will surface a warning (`⚠ '<name>' requires a newer
Hermes`) for that entry instead of silently hiding it. Run `hermes update`
to install the latest Hermes when you see that.

### Runtime `${ENV_VAR}` substitution

Inside an entry's `transport.command`, `transport.args`, `transport.url`,
and `headers`, `${VAR}` placeholders are resolved at server-connect time
from environment variables (which include everything in `~/.hermes/.env`).
This is useful when a catalog entry wants to reference a value the user
configured elsewhere — e.g. `${HOME}/foo` or `${MY_PROVIDER_TOKEN}`.

Note this is distinct from `${INSTALL_DIR}` in catalog manifests, which is
substituted at install-time with the path the catalog cloned the entry's
repo into.

### Updating tool selection later

```bash
hermes mcp configure linear
```

Reopens the same checklist with your current selection pre-checked. Use this
when you want more tools enabled, or when the server has added new tools that
you want to opt into.

### Updating the catalog manifest

MCPs are never auto-updated. Re-run `hermes mcp install <name>` to refresh
after a Hermes update if a manifest version changed.

To add an MCP to the catalog, open a PR against
[`optional-mcps/`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps).

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

### OAuth-authenticated HTTP servers

Most hosted MCP servers (Linear, Sentry, Atlassian, Asana, Figma, Stripe, …) require OAuth 2.1 instead of a static bearer token. Set `auth: oauth` and Hermes handles discovery, dynamic client registration, PKCE, token exchange, refresh, and step-up auth via the MCP Python SDK.

```yaml
mcp_servers:
  linear:
    url: "https://mcp.linear.app/mcp"
    auth: oauth
```

On first connect, Hermes prints an authorize URL, opens your browser when possible, and waits for the OAuth callback on a local loopback port. Tokens are cached at `~/.hermes/mcp-tokens/<server>.json` with 0o600 perms; subsequent runs reuse them silently until refresh fails.

**Remote / headless hosts.** When Hermes runs on a different machine than your browser, the loopback callback can't reach your laptop. Two ways to complete the flow:

- **Paste-back (no setup):** on an interactive terminal Hermes prints "Or paste the redirect URL here…" alongside the authorize URL. Open the URL in your browser, approve, copy the full URL the browser ends up on (the redirect will show a connection error — that's expected), paste it at the prompt. Bare `?code=…&state=…` query strings work too.
- **SSH port forward:** `ssh -N -L <port>:127.0.0.1:<port> user@host` in a separate terminal, then let the redirect flow normally.

See [OAuth over SSH / Remote Hosts](../../guides/oauth-over-ssh.md#mcp-servers) for the full walkthrough, including DCR-less servers (e.g. Slack), pre-registered `client_id`/`client_secret`, scope customization, and re-auth via `hermes mcp login <server>`.

**Pitfall — providers that don't support automatic registration (Google Drive, Atlassian).** Some servers reject the dynamic client registration step (RFC 7591) that bare `auth: oauth` relies on — Google's official Drive server (`https://drivemcp.googleapis.com/mcp/v1`) returns a `400 Bad Request`, so no OAuth client is created and no token is acquired. The symptom is subtle: these servers also serve `tools/list` *without* auth, so `hermes mcp login` can list the tools and look like it worked, but every real tool call later times out. `hermes mcp login` now detects this (it checks that a token actually landed on disk) and tells you to supply your own OAuth client. Create one in the provider's console and add it to config:

```yaml
mcp_servers:
  googledrive:
    url: "https://drivemcp.googleapis.com/mcp/v1"
    auth: oauth
    oauth:
      client_id: "<your-oauth-client-id>"
      client_secret: "<your-oauth-client-secret>"
```

Then run `hermes mcp login googledrive` — with the pre-registered client, Hermes skips registration and runs the normal browser authorization flow.

**Pitfall — config auto-reload race.** When you edit `~/.hermes/config.yaml` from inside a running Hermes session, the CLI auto-reloads MCP connections with a 30s timeout. That's not enough for an interactive OAuth flow. Add the entry, then run `hermes mcp login <server>` from a fresh terminal — it waits the full 5 minutes for you to complete auth.

## mTLS / client certificates

Remote HTTP MCP servers that require mutual TLS (client-certificate authentication) are supported via `client_cert` / `client_key`. Hermes passes the resolved certificate to the underlying HTTP client for the TLS handshake.

`client_cert` accepts three shapes:

- **A single combined PEM path** — one file holding both the certificate and the private key:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/.certs/mcp-client.pem"
```

- **A `[cert, key]` 2-tuple** — certificate and key in separate files (equivalent to setting `client_cert` + `client_key`):

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key"]
```

- **A `[cert, key, password]` 3-tuple** — when the private key is encrypted, the third element is the key passphrase:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key", "${MCP_KEY_PASSWORD}"]
```

You can also keep the cert and key fully separate via `client_cert` (combined PEM) plus an explicit `client_key`. Paths support `~` expansion; a missing file raises a clear, server-scoped error rather than an opaque TLS handshake failure.

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
| `client_cert` | string \| list | Client certificate for mTLS — a combined PEM path, or `[cert, key]` / `[cert, key, password]` |
| `client_key` | string | Client private-key PEM path (when separate from `client_cert`) |
| `timeout` | number | Tool call timeout |
| `connect_timeout` | number | Initial connection timeout |
| `enabled` | bool | If `false`, Hermes skips the server entirely |
| `supports_parallel_tool_calls` | bool | If `true`, tools from this server may run concurrently |
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

## Built-in presets

For well-known MCP servers, `hermes mcp add` accepts a `--preset` flag that fills in the transport details so you don't have to look up the command and args. The preset only supplies defaults — anything else (env vars, headers, filtering) you pass on the same command line still wins.

| Preset | What it wires up |
|---|---|
| `codex` | The Codex CLI's MCP server (`codex mcp-server` over stdio). Requires the `codex` CLI on PATH. |

```bash
# Add Codex CLI as an MCP server in one line
hermes mcp add codex --preset codex
```

That writes the equivalent of:

```yaml
mcp_servers:
  codex:
    command: "codex"
    args: ["mcp-server"]
```

You can pick any local name (`hermes mcp add my-codex --preset codex` is fine); the preset only provides the `command`/`args` defaults.

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

You can control which tools each MCP server contributes to Hermes, allowing fine-grained management of your tool namespace.

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

### Dynamic Tool Discovery

MCP servers can notify Hermes when their available tools change at runtime by sending a `notifications/tools/list_changed` notification. When Hermes receives this notification, it automatically re-fetches the server's tool list and updates the registry — no manual `/reload-mcp` required.

This is useful for MCP servers whose capabilities change dynamically (e.g. a server that adds tools when a new database schema is loaded, or removes tools when a service goes offline).

The refresh is lock-protected so rapid-fire notifications from the same server don't cause overlapping refreshes. Prompt and resource change notifications (`prompts/list_changed`, `resources/list_changed`) are received but not yet acted on.

### Reloading

If you change MCP config, use:

```text
/reload-mcp
```

This reloads MCP servers from config and refreshes the available tool list. For runtime tool changes pushed by the server itself, see [Dynamic Tool Discovery](#dynamic-tool-discovery) above.

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

## Troubleshooting

### MCP server not connecting

Check:

```bash
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

## Parallel Tool Calls

By default, MCP tools run sequentially — one at a time. If your MCP server exposes tools that are safe to run concurrently (e.g. read-only queries, independent API calls), you can opt-in to parallel execution:

```yaml
mcp_servers:
  docs:
    command: "docs-server"
    supports_parallel_tool_calls: true
```

When `supports_parallel_tool_calls` is `true`, Hermes may execute multiple tools from that server at the same time within a single tool-call batch, just like it does for built-in read-only tools (web_search, read_file, etc.).

:::caution
Only enable parallel calls for MCP servers whose tools are safe to run at the same time. If tools read and write shared state, files, databases, or external resources, review the read/write race conditions before enabling this setting.
:::

## MCP Sampling Support

MCP servers can request LLM inference from Hermes via the `sampling/createMessage` protocol. This allows an MCP server to ask Hermes to generate text on its behalf — useful for servers that need LLM capabilities but don't have their own model access.

Sampling is **enabled by default** for all MCP servers (when the MCP SDK supports it). Configure it per-server under the `sampling` key:

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    sampling:
      enabled: true            # Enable sampling (default: true)
      model: "openai/gpt-4o"  # Override model for sampling requests (optional)
      max_tokens_cap: 4096     # Max tokens per sampling response (default: 4096)
      timeout: 30              # Timeout in seconds per request (default: 30)
      max_rpm: 10              # Rate limit: max requests per minute (default: 10)
      max_tool_rounds: 5       # Max tool-use rounds in sampling loops (default: 5)
      allowed_models: []       # Allowlist of model names the server may request (empty = any)
      log_level: "info"        # Audit log level: debug, info, or warning (default: info)
```

The sampling handler includes a sliding-window rate limiter, per-request timeouts, and tool-loop depth limits to prevent runaway usage. Metrics (request count, errors, tokens used) are tracked per server instance.

To disable sampling for a specific server:

```yaml
mcp_servers:
  untrusted_server:
    url: "https://mcp.example.com"
    sampling:
      enabled: false
```

## Running Hermes as an MCP server

In addition to connecting **to** MCP servers, Hermes can also **be** an MCP server. This lets other MCP-capable agents (Claude Code, Cursor, Codex, or any MCP client) use Hermes's messaging capabilities — list conversations, read message history, and send messages across all your connected platforms.

### When to use this

- You want Claude Code, Cursor, or another coding agent to send and read Telegram/Discord/Slack messages through Hermes
- You want a single MCP server that bridges to all of Hermes's connected messaging platforms at once
- You already have a running Hermes gateway with connected platforms

:::note
**This bridge observes and sends — it does not task the agent.** `messages_send` delivers through Hermes's own bot identity (the same `send_message` path the Hermes agent uses), so a message sent this way is an *outbound bot message* — the Hermes agent does not receive it as a prompt and will not process or reply to it. `events_poll`, `events_wait`, and `messages_read` let an MCP client *observe* conversations.

An MCP client can therefore **watch** Hermes's conversations and **post as the Hermes bot**, but it cannot hand the Hermes agent a task through this bridge. To have the Hermes agent act on a request, the request must arrive through a real inbound channel — a message from an allowlisted user on a connected platform, the terminal UI, or a [scheduled cron job](/docs/user-guide/features/cron).
:::

### Quick start

```bash
hermes mcp serve
```

This starts a stdio MCP server. The MCP client (not you) manages the process lifecycle.

### MCP client configuration

Add Hermes to your MCP client config. For example, in Claude Code's `~/.claude/claude_desktop_config.json`:

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

Or if you installed Hermes in a specific location:

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

### Available tools

The MCP server exposes 10 tools, matching OpenClaw's channel bridge surface plus a Hermes-specific channel browser:

| Tool | Description |
|------|-------------|
| `conversations_list` | List active messaging conversations. Filter by platform or search by name. |
| `conversation_get` | Get detailed info about one conversation by session key. |
| `messages_read` | Read recent message history for a conversation. |
| `attachments_fetch` | Extract non-text attachments (images, media) from a specific message. |
| `events_poll` | Poll for new conversation events since a cursor position. |
| `events_wait` | Long-poll / block until the next event arrives (near-real-time). |
| `messages_send` | Send a message through a platform **as the Hermes bot** (e.g. `telegram:123456`, `discord:#general`). |
| `channels_list` | List available messaging targets across all platforms. |
| `permissions_list_open` | List pending approval requests observed during this bridge session. |
| `permissions_respond` | Allow or deny a pending approval request. |

### Event system

The MCP server includes a live event bridge that polls Hermes's session database for new messages. This gives MCP clients near-real-time awareness of incoming conversations:

```
# Poll for new events (non-blocking)
events_poll(after_cursor=0)

# Wait for next event (blocks up to timeout)
events_wait(after_cursor=42, timeout_ms=30000)
```

Event types: `message`, `approval_requested`, `approval_resolved`

The event queue is in-memory and starts when the bridge connects. Older messages are available through `messages_read`.

### Options

```bash
hermes mcp serve              # Normal mode
hermes mcp serve --verbose    # Debug logging on stderr
```

### How it works

The MCP server reads conversation data directly from Hermes's session store (`~/.hermes/sessions/sessions.json` and the SQLite database). A background thread polls the database for new messages and maintains an in-memory event queue. For sending messages, it uses the same `send_message` infrastructure as the Hermes agent itself — so a message sent through `messages_send` goes out as the Hermes bot, not as inbound input to the Hermes agent.

The gateway does NOT need to be running for read operations (listing conversations, reading history, polling events). It DOES need to be running for send operations, since the platform adapters need active connections.

### Current limits

- The embedded `hermes mcp serve` exposes a **stdio-only** MCP server today. If you need an HTTP MCP server, run a separate adapter — or, much more commonly, use the MCP **client** side of Hermes, which already speaks both stdio and HTTP (`url` + `headers` in `mcp_servers.yaml` / `config.yaml`; see [HTTP servers](#http-servers) above).
- Event polling at ~200ms intervals via mtime-optimized DB polling (skips work when files are unchanged)
- No `claude/channel` push notification protocol yet
- Text-only sends (no media/attachment sending through `messages_send`)
- Observe-and-send only — the bridge cannot task the Hermes agent (`messages_send` posts as the bot; no tool injects a prompt into Hermes's reasoning loop)

## Related docs

- [Use MCP with Hermes](/guides/use-mcp-with-hermes)
- [CLI Commands](/reference/cli-commands)
- [Slash Commands](/reference/slash-commands)
- [FAQ](/reference/faq)
