---
sidebar_position: 8
title: "MCP Config Reference"
description: "Reference for Hermes Agent MCP configuration keys, filtering semantics, and utility-tool policy"
---

# MCP Config Reference

This page is the compact reference companion to the main MCP docs.

For conceptual guidance, see:
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [Use MCP with Hermes](/guides/use-mcp-with-hermes)

## Root config shape

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # stdio servers
    args: []
    env: {}

    # OR
    url: "..."          # HTTP servers
    headers: {}

    # Optional HTTP/SSE TLS settings:
    ssl_verify: true                # bool or path to a CA bundle (PEM)
    client_cert: "/path/to/cert.pem"  # mTLS client certificate (see below)
    # client_key: "/path/to/key.pem"  # optional, when key lives in a separate file

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

## Server keys

| Key | Type | Applies to | Meaning |
|---|---|---|---|
| `command` | string | stdio | Executable to launch |
| `args` | list | stdio | Arguments for the subprocess |
| `env` | mapping | stdio | Environment passed to the subprocess |
| `url` | string | HTTP | Remote MCP endpoint |
| `headers` | mapping | HTTP | Headers for remote server requests |
| `ssl_verify` | bool or string | HTTP | TLS verification. `true` (default) uses system CAs, `false` disables verification (insecure), or a string path to a custom CA bundle (PEM) |
| `client_cert` | string or list | HTTP | mTLS client certificate. String = path to a PEM file containing cert + key. List `[cert, key]` = separate files. List `[cert, key, password]` = encrypted key |
| `client_key` | string | HTTP | Path to the client private key, when `client_cert` is a string and the key is in a separate file |
| `enabled` | bool | both | Skip the server entirely when false |
| `timeout` | number | both | Tool call timeout in seconds (default: `300`) |
| `connect_timeout` | number | both | Initial connection timeout in seconds (default: `60`) |
| `protocol` | string | both | Protocol-era negotiation: `auto` (default — legacy `initialize` handshake first, falling back to the 2026-07-28 `server/discover` stateless probe when the server rejects the handshake as modern-only), `stateless` (probe `server/discover` first; one legacy retry), or `legacy` (handshake only, no fallback) |
| `supports_parallel_tool_calls` | bool | both | Allow tools from this server to run concurrently |
| `skip_preflight` | bool | HTTP | Bypass the fail-fast content-type probe for valid Streamable HTTP endpoints whose HEAD/GET answers a non-MCP content type (default: `false`) |
| `transport` | string | HTTP | Set to `sse` to use the SSE transport instead of Streamable HTTP |
| `keepalive_interval` | number | both | Liveness ping cadence in seconds (default: `180`, floored at 5s). Set below the server's session TTL for servers that GC idle sessions quickly |
| `idle_timeout_seconds` | number | stdio | Optional stdio server recycle after idle time (`0` disables). May also live under a `lifecycle:` mapping |
| `max_lifetime_seconds` | number | stdio | Optional stdio server recycle after age (`0` disables). May also live under a `lifecycle:` mapping |
| `tools` | mapping | both | Filtering and utility-tool policy |
| `auth` | string | HTTP | Authentication method. Set to `oauth` to enable OAuth 2.1 with PKCE |
| `sampling` | mapping | both | Server-initiated LLM request policy (see MCP guide) |
| `elicitation` | mapping | both | Server-initiated user-input requests. `enabled` (default `true`) and `timeout` in seconds (default `300`). Form-mode requests route through the approval surface; URL-mode is declined (see MCP guide) |
| `trust` | string | both | Trust tier: `full` (default) or `untrusted`. On an `untrusted` server, every write-capable tool call (any tool without a `readOnlyHint: true` annotation) requires user approval through the standard approval surface before it runs. `readOnlyHint` is a server-supplied *hint* — a lying server can at most skip approval for tools it claims are read-only, never gain extra access — so mark any server you don't fully control as `untrusted`. Unrecognized values are treated as `untrusted` (fail-closed) |

## Environment variable references

String values anywhere in a server entry (`env`, `headers`, `args`, `url`, …) may reference environment variables with `${VAR}` or the Cursor-style SecretRef form `${env:VAR}` — both resolve to the same variable, so MCP snippets copied from Cursor / Claude configs work unchanged:

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${env:GITHUB_TOKEN}"   # same as "${GITHUB_TOKEN}"
```

Values resolve from the active profile's secret scope (falling back to the process environment), so put the secret in `~/.hermes/.env`. An unset variable keeps its literal placeholder.

### Context variables

Beyond env vars, the Cursor-style context variables are interpolated too (names are case-sensitive):

| Variable | Resolves to |
|---|---|
| `${userHome}` | The current user's home directory |
| `${workspaceFolder}` | The session workspace root (the session's terminal cwd when known, else the process cwd) |
| `${workspaceFolderBasename}` | The basename of `${workspaceFolder}` |
| `${pathSeparator}` / `${/}` | The OS path separator (`os.sep`) |

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
    env:
      CACHE_DIR: "${userHome}${/}.cache${/}mcp"
```

Any other `${...}` reference falls through to the env-var lookup above.

## `tools` policy keys

| Key | Type | Meaning |
|---|---|---|
| `include` | string or list | Whitelist server-native MCP tools. Entries may be exact names or fnmatch-style globs (`*_radar_*`, `get_zones_*`) |
| `exclude` | string or list | Blacklist server-native MCP tools. Same exact-name / glob semantics as `include` |
| `resources` | bool-like | Enable/disable `list_resources` + `read_resource` |
| `prompts` | bool-like | Enable/disable `list_prompts` + `get_prompt` |

## Filtering semantics

### `include`

If `include` is set, only those server-native MCP tools are registered.

```yaml
tools:
  include: [create_issue, list_issues]
```

### `exclude`

If `exclude` is set and `include` is not, every server-native MCP tool except those names is registered.

```yaml
tools:
  exclude: [delete_customer]
```

### Precedence

If both are set, `include` wins.

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

Result:
- `create_issue` is still allowed
- `delete_issue` is ignored because `include` takes precedence

## Utility-tool policy

Hermes may register these utility wrappers per MCP server:

Resources:
- `list_resources`
- `read_resource`

Prompts:
- `list_prompts`
- `get_prompt`

### Disable resources

```yaml
tools:
  resources: false
```

### Disable prompts

```yaml
tools:
  prompts: false
```

### Capability-aware registration

Even when `resources: true` or `prompts: true`, Hermes only registers those utility tools if the MCP session actually exposes the corresponding capability.

So this is normal:
- you enable prompts
- but no prompt utilities appear
- because the server does not support prompts

## `enabled: false`

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

Behavior:
- no connection attempt
- no discovery
- no tool registration
- config remains in place for later reuse

## Empty result behavior

If filtering removes all server-native tools and no utility tools are registered, Hermes does not create an empty MCP runtime toolset for that server.

## Example configs

### Safe GitHub allowlist

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

### Stripe blacklist

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### Resource-only docs server

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      include: []
      resources: true
      prompts: false
```

### TLS client certificate (mTLS)

For HTTP/SSE servers that require a client certificate, set `client_cert` (and optionally `client_key`):

```yaml
mcp_servers:
  # Combined cert + key in a single PEM file
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/secrets/mcp-client.pem"

  # Separate cert and key files
  partner_api:
    url: "https://mcp.partner.example.com/mcp"
    client_cert: "~/secrets/client.crt"
    client_key: "~/secrets/client.key"

  # Encrypted key with a passphrase (3-element list form)
  bank_api:
    url: "https://mcp.bank.example.com/mcp"
    client_cert: ["~/secrets/client.crt", "~/secrets/client.key", "my-passphrase"]

  # Custom CA bundle (private CA / self-signed server)
  lab_api:
    url: "https://mcp.lab.local/mcp"
    ssl_verify: "~/secrets/lab-ca.pem"
    client_cert: "~/secrets/lab-client.pem"
```

Notes:
- Paths support `~` expansion. Missing files fail fast at connect time with a server-scoped error message.
- `ssl_verify: false` disables server certificate verification entirely. Don't use this with real services.
- Works on both Streamable HTTP and SSE transports.

## Reloading config

After changing MCP config, reload servers with:

```text
/reload-mcp
```

## Tool naming

Server-native MCP tools become:

```text
mcp__<server>__<tool>
```

Examples:
- `mcp__github__create_issue`
- `mcp__filesystem__read_file`
- `mcp__my_api__query_data`

Utility tools follow the same prefixing pattern:
- `mcp__<server>__list_resources`
- `mcp__<server>__read_resource`
- `mcp__<server>__list_prompts`
- `mcp__<server>__get_prompt`

The double-underscore delimiter (`mcp__…__…`) matches the convention used by Claude Code, Codex, and OpenCode, and disambiguates the server/tool boundary even when either component contains underscores.

### Name sanitization

Any character that is not a letter, digit, or underscore (hyphens, dots, spaces, etc.) in both server names and tool names is replaced with an underscore before registration. This ensures tool names are valid identifiers for LLM function-calling APIs.

For example, a server named `my-api` exposing a tool called `list-items.v2` becomes:

```text
mcp__my_api__list_items_v2
```

Keep this in mind when writing `include` / `exclude` filters — use the **original** MCP tool name (with hyphens/dots), not the sanitized version.

## OAuth 2.1 authentication

For HTTP servers that require OAuth, set `auth: oauth` on the server entry:

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
```

Behavior:
- Hermes uses the MCP SDK's OAuth 2.1 PKCE flow (metadata discovery, client identification, token exchange, and refresh)
- On first connect, a browser window opens for authorization
- Tokens are persisted to `~/.hermes/mcp-tokens/<server>.json` and reused across sessions
- Token refresh is automatic; re-authorization only happens when refresh fails
- Only applies to HTTP/StreamableHTTP transport (`url`-based servers)

### Client identification: CIMD and DCR

Hermes identifies itself to authorization servers with a **Client ID Metadata Document** (CIMD), the mechanism the MCP `2026-07-28` spec adopted in place of Dynamic Client Registration. The document is published at
`https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json`, and that URL *is* the `client_id` — the authorization server fetches it to learn Hermes' name, logo, and permitted redirect URIs. Nothing is registered per install, and nothing is user-specific.

The final choice belongs to the authorization server: the SDK sends the document URL as the `client_id` only when the server advertises `client_id_metadata_document_supported: true` in its metadata, and otherwise registers via DCR exactly as before. DCR is deprecated in the MCP spec but still what almost every deployed server uses today.

#### Callback ports

The document declares a fixed set of loopback redirect URIs, and the spec requires the redirect URI in an authorization request to be an *exact string match* against one of them — so a CIMD flow cannot use the random high port Hermes normally picks. Hermes therefore pins the callback to one of ports `27890`–`27894`.

That pin has to be chosen before the server's capabilities are known, because the redirect URI is fixed at the start of the flow while the server's metadata only arrives partway through. So Hermes pins the port for any flow that *could* end up using CIMD, and reverts to a random port for the rest:

- A server Hermes has connected to before, whose cached metadata does not advertise CIMD, keeps the random port it has always used.
- A server Hermes has never reached gets a pinned port on that first login, since guessing is the only way CIMD can ever be used.
- Anything that would move the callback elsewhere reverts too: a pre-registered `oauth.client_id`, an `oauth.client_secret`, a custom `oauth.client_name` or `oauth.token_endpoint_auth_method`, an `oauth.redirect_uri` or `oauth.redirect_port` override, a dashboard- or desktop-driven login, an existing client registration on disk, or all five ports being held by other processes.

Each pinned port is bound as soon as it is chosen and held until the browser redirect arrives, so two concurrent logins — a second profile, or another server in the same process — cannot land on the same listener.

#### When a server rejects the document

If a server fetches the document and refuses it at the *token* endpoint (`invalid_client`), Hermes logs the rejection, records it under `~/.hermes/mcp-tokens/<server>.cimd-off`, and uses DCR for that server from then on.

A server that cannot fetch or validate the document at all aborts at the *authorization* endpoint instead, before any redirect happens. There is no signal Hermes can observe there, so the browser shows an invalid-client error and the login times out after five minutes. The timeout message names the document and points at `cimd: false`. Running `hermes mcp login <server>` clears the recorded rejection, so a corrected document gets another chance.

#### Optional per-server keys

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
    oauth:
      scope: "user:read company:read"                           # least-privilege ceiling
      client_metadata_url: "https://example.com/my-cimd.json"  # self-hosted document
      cimd: false                                              # force DCR
      user_agent: "My-MCP-Client/1.0"                          # token-request User-Agent
```

`oauth.scope` is a **hard ceiling** for Dynamic Client Registration and the authorize request. The MCP SDK otherwise requests every scope listed in the server's `scopes_supported`. When this key is set, Hermes keeps that value even if discovery advertises a wider list, and logs a warning if the SDK tries to overwrite it. Omit the key to keep the SDK's default (all advertised scopes).

`client_metadata_url` must be an HTTPS URL with a path (no bare origin, no fragment, no userinfo, no `.`/`..` segments) that returns `200` and `Content-Type: application/json` with **no redirect** — authorization servers are forbidden from following redirects when fetching it. Hermes still pins its callback to the same `27890`–`27894` range, so a self-hosted document must declare all ten loopback URIs (`http://127.0.0.1:<port>/callback` and `http://localhost:<port>/callback` for each port), and its `client_id` must be its own URL.

`user_agent` replaces the HTTP library's default `User-Agent` on **token-endpoint requests only** (authorization-code exchange and refresh) — some authorization servers and WAFs reject the default `python-httpx/...` value there. It never applies to MCP traffic or OAuth discovery, and no other token-request headers are configurable. Empty or null values are ignored.

## Add to Hermes link

MCP vendors and docs can offer a one-click **"Add to Hermes"** button that opens the Hermes desktop app with a pre-filled server config, mirroring Cursor's `cursor://anysphere.cursor-deeplink/mcp/install` scheme:

```text
hermes://mcp/install?name=NAME&config=BASE64
```

- `name` — the server name. Must match `^[A-Za-z0-9._-]{1,64}$`.
- `config` — the server config object as **base64url-encoded JSON** (standard base64 is also accepted). The decoded JSON must be an object with either a string `url` field (`http://`/`https://` only) or a string `command` field, and may carry any of the server keys documented above. Payloads over 32KB are rejected.

Example (JavaScript):

```js
const config = { url: 'https://mcp.example.com/mcp' }
const link = `hermes://mcp/install?name=example&config=${btoa(JSON.stringify(config))
  .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')}`
```

Opening the link never installs anything by itself: the desktop app shows a confirmation dialog with the server name and the full pretty-printed config (with an extra caution for `command`-based servers, which run a local process), and the user must explicitly confirm. Existing server names are never overwritten — the user is asked to rename or cancel.
