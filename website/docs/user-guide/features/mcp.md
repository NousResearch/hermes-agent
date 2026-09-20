---
sidebar_position: 4
title: "MCP (Model Context Protocol)"
description: "Connect Hermes Agent to external tool servers via MCP — and control exactly which MCP tools Hermes loads"
---

# MCP (Model Context Protocol)

MCP lets Hermes Agent connect to external tool servers so the agent can use tools that live outside Hermes itself — GitHub, databases, file systems, browser stacks, internal APIs, and more.

If you have ever wanted Hermes to use a tool that already exists somewhere else, MCP is usually the cleanest way to do it.

:::tip Coming from Claude Code?
The `mcpServers` block in your `~/.claude.json` maps to `mcp_servers` in Hermes' `config.yaml` — and `hermes import-agent claude-code` migrates it (along with skills and instructions) automatically. See [Import from Other Agents](../import-from-other-agents.md).
:::

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

In the desktop app you can also ask: "add the Linear MCP". The agent calls
`manage_connections` with an `mcp: true` target, an approval card appears in
the chat, and Install writes the same config the CLI would. On the CLI and in
messaging apps the agent relays the commands below instead.

```bash
hermes mcp                   # interactive picker (default)
hermes mcp catalog           # plain-text list, scriptable
hermes mcp install deepwiki  # install a catalog entry by name
```

The picker shows each entry with its current status:

```
deepwiki     available              Ask questions about public GitHub repositories
linear       enabled                Linear issue/project management (remote OAuth)
github       installed (disabled)   GitHub repo + PR tools
```

Hit `Enter` on a row to install (and walk through any required credentials),
enable, disable, or uninstall. Catalog entries are stored under
`optional-mcps/` in the hermes-agent repo — presence in that directory means
Nous approval. There is no community submission tier; entries are added by
merging a PR.

The third-party n8n bridge is no longer available for catalog installation.
Existing installations keep their `mcp_servers` configuration, credentials,
installed files, and selected tools. They continue to load as configured MCP
servers and appear as custom entries in the picker, where you can still
configure tools or enable and disable them. Catalog reinstall is no longer
available. This change does not migrate existing connections to
[n8n's official MCP server](https://docs.n8n.io/connect/connect-to-n8n-mcp-server/).

Catalog entries can require:

- **API key** — Hermes prompts at install time and writes the value to
  `~/.hermes/.env`. Non-secret values (base URLs) go to the same file.
- **OAuth** (remote MCP) — written as `auth: oauth` in your config; the MCP
  client opens a browser on first connection.
- **OAuth** (third-party provider like Google/GitHub) — Hermes points you at
  `hermes auth <provider>` if you haven't authenticated already.

### n8n's official MCP server

The `n8n-official` catalog entry connects directly to your n8n Cloud or
self-hosted instance over HTTP with browser OAuth. No local bridge or n8n
API key is required.

1. Ask an owner or admin to enable **Settings > Instance-level MCP** in n8n.
2. Open **Connect** and copy the full **Server URL** ending in
   `/mcp-server/http`, not the editor URL. Older versions show the endpoint
   directly on the MCP settings page.
3. Run `hermes mcp install n8n-official` and enter that URL when prompted.
4. Complete browser OAuth. If needed, run `hermes mcp login n8n-official`
   or use **Authorize** on the configured server in Desktop or the dashboard.
5. Review tools with `hermes mcp configure n8n-official`, then start a new
   session or use `/reload-mcp`.

The Hermes backend must be able to reach the URL. n8n controls permissions
and workflow exposure; some tools modify or run workflows. See
[n8n's connection guide](https://docs.n8n.io/connect/connect-to-n8n-mcp-server/).

This entry uses the existing catalog setup and storage behavior. It is
separate from the retired `n8n` bridge, so existing connections, credentials,
installed files, and tool selections are not replaced.

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

Some entries with very large auto-generated surfaces (e.g. `cloudflare`,
~3,300 OpenAPI endpoint tools) instead declare `tools.default_excluded` — a
curated block-list of names and glob patterns. Installing one of these skips
the checklist entirely and writes `tools.exclude`; everything not matched
stays enabled, including tools the server adds later. Edit
`mcp_servers.<name>.tools.exclude` in config.yaml to re-enable a family.

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

Cursor-style context variables are also substituted (case-sensitive):
`${userHome}` (home directory), `${workspaceFolder}` (session workspace
root), `${workspaceFolderBasename}`, and `${pathSeparator}` / `${/}`
(the OS path separator). See the
[MCP config reference](../../reference/mcp-config-reference.md) for details.

Note this is distinct from `${INSTALL_DIR}` in catalog manifests, which is
substituted at install-time with the path the catalog cloned the entry's
repo into.

### Entries that need your own OAuth app (no DCR)

Some vendors run their remote MCP behind OAuth but do **not** offer Dynamic
Client Registration — every client must be an app the user pre-registers in
the vendor's developer console. Asana's V2 server
(`https://mcp.asana.com/v2/mcp`) is the shipped example: the retired V1
`https://mcp.asana.com/sse` server accepted any client; V2 does not.

Such a manifest declares the credentials under `auth.env` and pins the
client under `auth.oauth`, so installing it (CLI picker, web dashboard or
Desktop) prompts for the Client ID / Client secret, stores them in the
profile's `.env`, and writes only `${VAR}` references to `config.yaml`:

```yaml
mcp_servers:
  asana:
    url: https://mcp.asana.com/v2/mcp
    auth: oauth
    oauth:
      client_id: "${ASANA_CLIENT_ID}"
      client_secret: "${ASANA_CLIENT_SECRET}"
      redirect_host: localhost      # the vendor matches the redirect URL exactly
      redirect_port: 27890          # register http://localhost:27890/callback on the app
```

Read the entry's `post_install` notes for the exact app type and redirect URL
to register, then run `hermes mcp login <name>` and restart (or
`/reload-mcp`) the session or gateway that should expose the tools. The
dashboard / Desktop **Authorize** button works too: because the client is
pre-registered with a pinned `redirect_port`, Hermes keeps the registered
loopback callback (`http://localhost:27890/callback`) instead of the
dashboard's own callback URL — so the browser you approve in must run on the
same machine as the Hermes process. For a remote host, use `hermes mcp login`
over SSH port-forwarding.

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

### Suggestion metadata (`suggest:`)

A manifest may declare an optional `suggest:` block with `keywords:` and/or
`hosts:` lists. UI surfaces (currently the Desktop app's composer) use it to
offer a one-click "Add &lt;server&gt;" pill when your draft mentions one of the
keywords as a completed word, or contains a pasted link whose hostname ends
with one of the host suffixes. It is purely advisory — installs still flow
through the same validated catalog/config paths — and most hosted remote
entries (Atlassian, Sentry, Notion, Stripe, Vercel, Supabase, and friends)
declare it.

GitHub is deliberately **not** in the catalog: its hosted MCP requires each
client to bring its own OAuth app (generic dynamic client registration is
rejected), and Hermes's bundled `github/*` skills driving the `gh` CLI are a
more capable integration. On Desktop, GitHub mentions instead offer the
`github-auth` skill when `gh` isn't signed in yet.

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

HTTP and SSE servers honor the standard proxy settings: `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` (a `socks://` alias is normalized to `socks5://`), then the OS proxy (Windows registry, macOS system settings), with `NO_PROXY` hosts — including CIDR ranges and `*.example.com` patterns — connecting directly.

### OAuth-authenticated HTTP servers

Most hosted MCP servers (Cloudflare, Linear, Sentry, Atlassian, Asana, Figma, Stripe, …) require OAuth 2.1 instead of a static bearer token. Set `auth: oauth` and Hermes handles discovery, client identification, PKCE, token exchange, refresh, and step-up auth via the MCP Python SDK.

Hermes identifies itself with a [Client ID Metadata Document](../../reference/mcp-config-reference.md#client-identification-cimd-and-dcr) on servers that support one, and falls back to Dynamic Client Registration on those that don't. Both are automatic; there is nothing to configure.

:::tip Figma remote MCP
Figma's hosted endpoint (`https://mcp.figma.com/mcp`) allowlists Dynamic Client Registration by **exact `client_name`** — bare `"Hermes Agent"` 403s, while `"Claude Code"` and `"Codex"` succeed. Hermes auto-sets `oauth.client_name: "Claude Code"` for `mcp.figma.com` so install/login works without a special trick:

```yaml
mcp_servers:
  figma:
    url: "https://mcp.figma.com/mcp"
    auth: oauth
```

Or: `hermes mcp install figma`, then `hermes mcp login figma`.
:::

```yaml
mcp_servers:
  linear:
    url: "https://mcp.linear.app/mcp"
    auth: oauth
```

On first connect, Hermes prints an authorize URL, opens your browser when possible, and waits for the OAuth callback on a local loopback port. Tokens are cached at `~/.hermes/mcp-tokens/<server>.json` with 0o600 perms; subsequent runs reuse them silently until refresh fails.

Refresh tokens are bound to the authorization server that granted them: Hermes records the discovered issuer alongside the cached tokens and, if a server's advertised authorization server ever changes (server migration, metadata edit, or hijack), the stored refresh token is dropped instead of being sent to the new issuer. The current access token keeps working until it expires, then a normal re-authorization runs against the new issuer.

The redirect back from the authorization server is checked against RFC 9207: when the server's metadata advertises `authorization_response_iss_parameter_supported`, a redirect without a matching `iss` is rejected. Figma's authorization server (`https://api.figma.com`) advertises that support and then omits `iss`; Hermes fills the missing value from the discovered issuer for that one issuer and logs a warning, so `hermes mcp login figma` completes. A present-but-different `iss` is still rejected, and no other server gets the exemption.

The authorization server's metadata document must name the server the resource advertised (RFC 8414 §3.3); a document for a different server is rejected before any registration or login. One shape is accepted without an exact match: a server advertised with a path (`https://host/path`) whose document, fetched from `https://host/.well-known/oauth-authorization-server/path`, names the origin `https://host` as its issuer — Strava's MCP connector publishes exactly that pair. Only the origin's operator controls that well-known location, so the document is treated as the advertised server's own; a document naming another origin or another path, or one reached only through a redirect or a fallback location, still fails with `Authorization server metadata issuer mismatch`.

**Remote / headless hosts.** When Hermes runs on a different machine than your browser, the loopback callback can't reach your laptop. Ways to complete the flow:

- **Hermes Desktop (automatic):** when you run the OAuth sign-in from the Desktop app's MCP setup UI against a remote backend, Desktop hosts the callback listener on *your* machine and relays the authorization back to the gateway automatically — no tunnel, paste, or proxy needed. Requires both the Desktop app and the backend to be up to date.
- **Paste-back (no setup):** on an interactive terminal Hermes prints "Or paste the redirect URL here…" alongside the authorize URL. Open the URL in your browser, approve, copy the full URL the browser ends up on (the redirect will show a connection error — that's expected), paste it at the prompt. Bare `?code=…&state=…` query strings work too.
- **Device-code login (no callback at all):** if the server's authorization server advertises a device authorization endpoint, run `hermes mcp login <server> --flow device` on the machine running Hermes. It prints a verification URL and a short code; open the URL on any device, enter the code, and Hermes polls for approval. No browser is launched on the host and no callback listener is needed. Set `oauth.flow: device` on the server to make `login` and `reauth` use it by default. Details: [Device-code login](../../reference/mcp-config-reference.md#device-code-login-rfc-8628).
- **SSH port forward:** `ssh -N -L <port>:127.0.0.1:<port> user@host` in a separate terminal, then let the redirect flow normally.
- **Proxied callback (`redirect_uri`):** when a public HTTPS endpoint forwards to the host (e.g. a Tailscale Funnel or reverse proxy pointed at the callback port), set `oauth.redirect_uri` and the browser redirect reaches Hermes on its own — no tunnel or paste needed:

```yaml
mcp_servers:
  myserver:
    url: "https://mcp.example.com/mcp"
    auth: oauth
    oauth:
      redirect_port: 8765                                # fixed port for the proxy to target
      redirect_uri: "https://oauth.example.ts.net/callback"
```

For fully headless gateways (messaging bot, no interactive terminal at all), the optional [`mcp-oauth-remote-gateway` skill](../skills/optional/mcp/mcp-mcp-oauth-remote-gateway.md) walks the agent through completing the flow manually and writing tokens where Hermes expects them.

**Pitfall — WAF rejects `127.0.0.1` redirect URIs.** A few providers front their authorization server with a WAF that 403s any authorize request whose query string contains a literal `127.0.0.1` (Reclaim.ai's AWS API Gateway is a known example — every attempt returns `{"message":"Forbidden"}` before reaching the OAuth app). Set `oauth.redirect_host: localhost` to use `http://localhost:<port>/callback` instead; the callback listener still binds `127.0.0.1` either way.

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

## Per-user identity header

Remote HTTP/SSE MCP servers that key behavior on a caller identity (per-user rate limits, audit trails, multi-tenant routing) can be sent an identity header on every request via `identity_header`:

```yaml
mcp_servers:
  team_api:
    url: "https://mcp.team.example.com/mcp"
    identity_header:
      name: "X-User-Id"
      value_from: "static"   # "static" (default) or "profile"
      value: "alice"         # required for static
```

- `value_from: static` sends the literal `value` from config.yaml.
- `value_from: profile` sends the active Hermes profile name, resolved once at connect time — useful when multiple profiles on one machine talk to the same server and it needs to tell them apart.

An explicit entry in the server's `headers` mapping with the same name (any casing) always wins; the identity header never overrides your own header config. Invalid `identity_header` blocks are warned about and ignored — they never block the server from connecting. On stdio servers the key is ignored with a warning (stdio transports have no headers).

## Basic configuration reference

Hermes reads MCP config from `~/.hermes/config.yaml` under `mcp_servers`.

### Common keys

| Key | Type | Meaning |
|---|---|---|
| `command` | string | Executable for a stdio MCP server |
| `args` | list | Arguments for the stdio server |
| `env` | mapping | Environment variables passed to the stdio server |
| `cwd` | string | Working directory for the stdio server process. Default: the session working directory when one is pinned (ACP/gateway sessions, `terminal.cwd`), else the Hermes process directory |
| `url` | string | HTTP MCP endpoint |
| `headers` | mapping | HTTP headers for remote servers |
| `client_cert` | string \| list | Client certificate for mTLS — a combined PEM path, or `[cert, key]` / `[cert, key, password]` |
| `client_key` | string | Client private-key PEM path (when separate from `client_cert`) |
| `identity_header` | mapping | Optional per-user identity header for HTTP/SSE servers — `{name, value_from: static\|profile, value}` |
| `timeout` | number | Tool call timeout |
| `connect_timeout` | number | Initial connection timeout (also bounds the MCP `initialize` handshake) |
| `lazy` | bool | If `true`, register the server's tools from the schema cache at startup and only start/connect it on the first tool call (default `false`). Needs one prior live connect to fill the cache. |
| `idle_timeout_seconds` | number | Recycle a stdio server after this many seconds without a tool call (`0` = never, default). The server restarts transparently on the next tool call. |
| `max_lifetime_seconds` | number | Recycle a stdio server after this total age (`0` = never, default). Restarts transparently on next use. |
| `enabled` | bool | If `false`, Hermes skips the server entirely |
| `supports_parallel_tool_calls` | bool | If `true`, tools from this server may run concurrently |
| `tools` | mapping | Per-server tool filtering and utility policy |

### Minimal stdio example

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
```

### Recycling memory-heavy stdio servers

Browser-based MCP servers (e.g. `@playwright/mcp`) keep a full Chromium
resident after their first tool call — hundreds of MB that never get
released. Opt in to automatic recycling and the server is torn down after
the idle/lifetime limit, then restarted transparently the next time one of
its tools is called (its tools stay registered the whole time):

```yaml
mcp_servers:
  playwright:
    command: "npx"
    args: ["-y", "@playwright/mcp@latest", "--headless"]
    idle_timeout_seconds: 900     # recycle after 15 min without a tool call
    max_lifetime_seconds: 86400   # and at least once a day regardless
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

### Tool-result sanitization and `_meta`

Two behaviors apply to every MCP tool result before the model sees it:

- **Invisible Unicode TAG characters are stripped.** Characters in the U+E0000–U+E007F range render as nothing in terminals and chat UIs but are fully visible to the model — a classic prompt-injection smuggling channel for a malicious or compromised server. Hermes strips them from tool results, resource content, and tool descriptions. Legitimate emoji tag sequences (regional flags like 🏴󠁧󠁢󠁳󠁣󠁴󠁿) are preserved.
- **Vendor `_meta` is surfaced; protocol-reserved keys are not.** When a server attaches a `_meta` mapping to a tool result (vendor namespaces like `com.example/handoff`), Hermes passes it through to the model alongside the result content. Keys under protocol-reserved prefixes — a `modelcontextprotocol` or `mcp` label followed by another label, e.g. `modelcontextprotocol.io/...` or `tools.mcp.com/...` — are dropped, matching the MCP spec's key-name rules. If nothing model-facing remains, the `_meta` field is omitted entirely.

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

Entries in `include`/`exclude` may also be glob patterns (`*`, `?`, `[...]`,
matched case-sensitively): `include: ["*_dns_*"]` registers every tool whose
name contains `_dns_`. Plain entries without metacharacters stay exact-match.
Globs are the practical way to filter servers that expose thousands of
auto-generated endpoint tools by product family.

### Blacklist server tools

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

All server tools are registered except the excluded ones.

### Glob patterns

Both lists accept fnmatch-style globs alongside exact names — essential for
huge flat surfaces like Cloudflare's API MCP (`?codemode=false`, ~3,300
tools) where excluding product areas one endpoint at a time is impractical:

```yaml
mcp_servers:
  cloudflare:
    url: "https://mcp.cloudflare.com/mcp?codemode=false"
    auth: oauth
    tools:
      exclude: ["*_radar_*", "*_accounts_dlp_*", "*_zones_web3_*"]
```

Entries without glob metacharacters (`*`, `?`, `[`) match exactly — `docs`
excludes only the tool named `docs`, never `docs_search`.

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

### Lazy start

A server with `lazy: true` is registered from the on-disk schema cache instead: its tools appear in the registry immediately, and the process is spawned (or the HTTP endpoint connected) on the first tool call. The cache is written on every live connect, so the first run of a new or changed server is always eager. The banner and the TUI session panel show such a server as **lazy** with its cached tool count (`3 tool(s) (lazy, starts on first use)`) — it is a working server, not a failed one — and the startup discovery summary counts it as `N lazy, not spawned yet`.

### Dynamic Tool Discovery

MCP servers can notify Hermes when their available tools change at runtime by sending a `notifications/tools/list_changed` notification. When Hermes receives this notification, it automatically re-fetches the server's tool list and updates the registry — no manual `/reload-mcp` required.

This is useful for MCP servers whose capabilities change dynamically (e.g. a server that adds tools when a new database schema is loaded, or removes tools when a service goes offline).

The refresh is lock-protected so rapid-fire notifications from the same server don't cause overlapping refreshes. Prompt and resource change notifications (`prompts/list_changed`, `resources/list_changed`) are received but not yet acted on.

### Reloading

If you change MCP config, use:

```text
/reload-mcp
```

This reloads MCP servers from config and refreshes the available tool list. It is also the explicit way to re-probe availability-gated tools (Docker, `HASS_TOKEN`, OAuth…): a session's tool set is otherwise frozen, so a credential or daemon that appears mid-session is only picked up on `/reload-mcp`, `/new`, or context compaction. For runtime tool changes pushed by the server itself, see [Dynamic Tool Discovery](#dynamic-tool-discovery) above.

A running messaging gateway (`hermes gateway run`) also watches `config.yaml` on its own: within about a minute of you removing an `mcp_servers` entry or setting `enabled: false`, that server's connection is torn down; a newly added entry is connected. A server whose first connect failed (an unreachable host, or an OAuth server on a headless box that had no token yet) is retried automatically on its connect cooldown schedule (30 s, doubling up to 10 min) once you fix the cause. No restart or `/reload-mcp` needed for the edit to take effect.

**Expired OAuth tokens in the background.** The gateway, `/reload-mcp`, and the periodic self-probe of a parked server never open a browser — nobody is there to complete the flow. When a refresh token dies, the server parks with a warning in `gateway.log` and you re-authorize once with `hermes mcp login <server>` (or the Desktop/dashboard *Authorize* button); the parked server picks the new token up on its next probe.

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

The startup summary in `agent.log` names every server that did not register, with the recorded
connect error, so you never have to work out the failing one by elimination:

```
MCP: registered 116 tool(s) from 4 server(s) (2 failed: github (Connection closed); notion (HTTP 401 from POST https://mcp.notion.com/mcp))
```

A server that was skipped this pass because it is still inside its retry cooldown from an earlier
failure is listed as `not attempted (in retry cooldown)`.

### Remote (HTTP) server rejects the connection

`hermes mcp test <name>` reports what the server actually answered. When the MCP SDK can only say
`Server returned an error response` (a 4xx/5xx whose body is not a JSON-RPC error), Hermes appends
the HTTP status, the URL it requested and the start of the response body:

```
Streamable HTTP: Server returned an error response (HTTP 400 from POST http://host:27200/mcp:
{"jsonrpc":"2.0","error":{"code":-32020,"message":"Unsupported MCP-Protocol-Version"}})
```

Read the status and body first: a `400`/`405` on the `initialize` POST usually means the endpoint
speaks SSE only (set `transport: sse`) or a proxy in front of it rejects the request; a `401`/`403`
means the token or OAuth grant is wrong; an HTML body means the URL points at a web page, not an MCP
endpoint. `hermes logs --level debug` additionally shows the exact endpoint each connect attempt used.

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

## MCP Elicitation Support

MCP servers can ask the user for structured input mid-tool-call via the `elicitation/create` protocol (mcp Python SDK ≥ 1.11.0). Hermes routes **form-mode** elicitations through its existing approval surface — an interactive prompt in the CLI/TUI, or approval buttons on gateway platforms like Telegram and Slack — so the request reaches you wherever the session lives. **URL-mode** elicitations (where a server points you at an external URL) are declined as unsupported.

Elicitation is **enabled by default** per server. Configure it under the `elicitation` key:

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    elicitation:
      enabled: true    # default: true
      timeout: 300     # seconds to wait for your answer (default: 300)
```

The 5-minute default timeout mirrors the gateway approval default so users on async surfaces have time to respond before the server gives up. Per-server metrics (requests, accepted, declined, errors) are tracked on the handler.

## Running Hermes as an MCP server

In addition to connecting **to** MCP servers, Hermes can also **be** an MCP server. This lets other MCP-capable agents (Claude Code, Cursor, Codex, or any MCP client) use Hermes's messaging capabilities — list conversations, read message history, and send messages across all your connected platforms.

### When to use this

- You want Claude Code, Cursor, or another coding agent to send and read Telegram/Discord/Slack messages through Hermes
- You want a single MCP server that bridges to all of Hermes's connected messaging platforms at once
- You already have a running Hermes gateway with connected platforms

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
| `messages_send` | Send a message through a platform (e.g. `telegram:123456`, `discord:#general`). |
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

The MCP server reads conversation data directly from Hermes's session store — `~/.hermes/state.db` is the primary source, with `sessions.json` kept only as a legacy fallback. A background thread polls the database for new messages and maintains an in-memory event queue. For sending messages, it uses the same internal send engine (`tools/send_message_tool.py`) that powers cron delivery and the `hermes send` CLI.

The gateway does NOT need to be running for read operations (listing conversations, reading history, polling events). It DOES need to be running for send operations, since the platform adapters need active connections.

### Current limits

- The embedded `hermes mcp serve` exposes a **stdio-only** MCP server today. If you need an HTTP MCP server, run a separate adapter — or, much more commonly, use the MCP **client** side of Hermes, which already speaks both stdio and HTTP (`url` + `headers` in `mcp_servers.yaml` / `config.yaml`; see [HTTP servers](#http-servers) above).
- Event polling at ~200ms intervals via mtime-optimized DB polling (skips work when files are unchanged)
- No `claude/channel` push notification protocol yet
- Text-only sends (no media/attachment sending through `messages_send`)

## Related docs

- [Use MCP with Hermes](../../guides/use-mcp-with-hermes.md)
- [CLI Commands](../../reference/cli-commands.md)
- [Slash Commands](../../reference/slash-commands.md)
- [FAQ](../../reference/faq.md)
