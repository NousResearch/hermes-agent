---
sidebar_position: 5
title: "UTCP Tool Providers"
description: "Give Hermes native UTCP (Universal Tool Calling Protocol) tools — discovery, auth, and execution without an MCP server in between"
---

# UTCP Tool Providers

Hermes can consume [UTCP](https://github.com/universal-tool-calling-protocol/utcp-specification)
providers natively. A UTCP provider publishes a **manual** — a list of tool
definitions, each with the transport, URL and auth needed to call it — and
Hermes fetches that manual, registers the tools, and calls them directly.

There is no MCP server in the middle. The agent talks to your provider's own
API, using the provider's own transport.

## Why UTCP (and how it relates to MCP)

| | MCP | UTCP |
|---|---|---|
| What it wraps | An MCP **server** you run or connect to | Your **existing HTTP API** |
| Extra process | Usually yes (stdio server, bridge) | No |
| Tool discovery | `tools/list` over the protocol | A manual at a URL (or an OpenAPI spec converted on the fly) |
| Auth | Server-defined | Declared in the provider (API key / basic / OAuth2) |
| Best for | Servers that already speak MCP | REST/OpenAPI APIs you want to expose directly |

MCP remains fully supported — the two providers coexist in the same session.
Use MCP when an MCP server already exists; use UTCP when the tool is just an
HTTP API (or when you'd otherwise deploy a server only to wrap one).

## Install

UTCP support is optional and isolated. Nothing is imported until you enable it.

```bash
hermes pm install --extra utcp
```

Enabling UTCP in `config.yaml` without the extra installed makes discovery
prepare it on first use (interactively, `hermes pm` asks before syncing). To
pre-install — the right thing for servers, containers and CI — run the command
above.

## Quick start

1. Add the provider to `config.yaml` (or use `hermes utcp add`):

```yaml
utcp:
  enabled: true
  providers:
    - name: weather
      call_template_type: http
      url: https://weather.example.com/.well-known/utcp
```

2. Put any credential in `~/.hermes/.env` — never in `config.yaml`:

```bash
# ~/.hermes/.env
WEATHER_UTCP_API_KEY=sk-live-...
```

3. Reference it from the provider:

```yaml
utcp:
  enabled: true
  providers:
    - name: weather
      call_template_type: http
      url: https://weather.example.com/.well-known/utcp
      variables:
        API_KEY: env:WEATHER_UTCP_API_KEY
      headers:
        Authorization: "Bearer ${API_KEY}"
```

4. Verify:

```bash
hermes utcp status     # connected? how many tools?
hermes utcp tools      # registry names, e.g. utcp__weather__get_forecast
```

5. Ask the agent to use the tool. It appears in the model's tool list as
`utcp__weather__get_forecast`.

## What a provider serves

At the configured `url`, Hermes accepts either:

- **A UTCP manual** — JSON (or YAML) with `utcp_version` and `tools`; or
- **An OpenAPI document** — the SDK converts operations into UTCP tools. This
  is the fastest path for an existing REST API: point `url` at `/openapi.json`.

Each tool entry names a transport and a URL. Hermes registers the tool only if
its transport matches the provider's (`http`, `sse`, `streamable_http`).

## Configuration reference

```yaml
utcp:
  enabled: true                       # master switch (default: false)
  allowed_transports:                 # accepted transports (default below)
    - http
    - sse
    - streamable_http
  discovery_timeout: 1.5              # bounded join at agent build (seconds)
  single_query_discovery_timeout: 15  # same, for `hermes chat -q`
  max_concurrent_calls: 4             # per-provider default
  max_result_chars: 200000            # per-provider default
  max_arg_bytes: 262144               # argument payload cap
  providers:
    - name: weather                   # [A-Za-z0-9_], part of every tool name
      enabled: true
      call_template_type: http        # http | sse | streamable_http
      url: https://weather.example.com/.well-known/utcp
      timeout: 60                     # per tool call (1–600s)
      connect_timeout: 15             # discovery budget (1–120s)
      max_result_chars: 200000
      max_concurrent_calls: 4
      variables:                      # NAME -> env:ENV_VAR (secrets live in .env)
        API_KEY: env:WEATHER_UTCP_API_KEY
      headers:                        # static headers, ${VAR} allowed
        Authorization: "Bearer ${API_KEY}"
      auth:                           # optional SDK auth block (api_key/basic/oauth2)
        auth_type: api_key
        api_key: "${API_KEY}"
        var_name: X-Api-Key
        location: header
      tools:
        include: []                   # optional whitelist (globs allowed)
        exclude: []                   # optional blacklist
```

Non-default transports are refused even if the SDK is installed: `cli`, `file`,
`socket` and `mcp` are **not** in the allowlist because they execute local
commands or read local files, and a provider config must never be able to do
that on the host.

## Tool identity

Every discovered tool is registered as:

```
utcp__<provider>__<tool>
```

- The provider component is the `name` from `config.yaml`.
- The tool component is the provider-local tool name.
- Non-alphanumeric characters become `_`; long names are truncated with a
  stable digest so two long names can never collide silently.

Each provider gets its own toolset (`utcp-<provider>`), aliased to the bare
provider name, so `--toolsets utcp-weather` and `--toolsets weather` both work.

Tools already registered by Hermes or by another provider are never
overwritten: if a name is taken, the UTCP tool is skipped and a warning is
logged.

## CLI

```bash
hermes utcp list                  # configured providers (no network)
hermes utcp status                # live status: connected, tools, last error
hermes utcp tools [provider]      # discovered tools + registry names
hermes utcp refresh [provider]    # re-run discovery, re-register tools
hermes utcp add weather --url https://weather.example.com/.well-known/utcp \
    --variable API_KEY=env:WEATHER_UTCP_API_KEY
hermes utcp remove weather
```

`status` only ever prints the *names* of the environment variables a provider
consumes, never their values.

## Lifecycle

Discovery happens at startup, not on every tool call:

1. Entrypoint starts discovery in the background (CLI, TUI, desktop, gateway,
   ACP) or inline for entrypoints without a later startup path.
2. Before the agent snapshots its tool list, Hermes joins that thread for at
   most `discovery_timeout` (interactive) or `single_query_discovery_timeout`
   (one-shot runs). The join returns the instant discovery completes.
3. Each provider is discovered independently. A provider that is down, slow, or
   misconfigured leaves the other providers' tools intact and reports its own
   error in `hermes utcp status`.
4. `hermes utcp refresh` re-discovers on demand. Tool lists are not polled
   during a session.

Each provider runs its own UTCP client, so its transports, credential scope,
concurrency limit and failures are isolated from every other provider.

## Authentication

UTCP's own auth mechanisms are supported (API key in header/query/cookie, basic,
OAuth2 with token caching and refresh) via the `auth:` block, and any template
field can interpolate `${VAR}` variables.

Rules Hermes enforces on top of the SDK:

- **Secrets live in `.env`.** `variables:` maps a name to `env:ENV_VAR`; literal
  values are rejected so a credential can't be committed to `config.yaml`.
- **Variables are provider-scoped.** The SDK resolves `${VAR}` as
  `<provider>_<VAR>`. Hermes additionally requires every reference — in the
  provider config *and* in every discovered tool template — to be declared in
  that provider's `variables:`. A hostile manual that says
  `${OPENAI_API_KEY}` is rejected instead of being resolved against an
  unrelated secret.
- **Errors are scrubbed.** Configured values are redacted from every error
  string, along with `Bearer …`, `sk-…`-style tokens and `api_key=…` pairs.

## Security model

Providers are external, untrusted inputs. Hermes applies:

- **Transport allowlist** — HTTP-family transports only (no CLI/file/socket).
- **URL policy** — HTTPS, or plain HTTP only for loopback hosts; no credentials
  embedded in URLs; no placeholder hosts (the SDK re-checks the resolved URL on
  every call and re-validates redirects).
- **Credential scoping** — see above; declared variables only, per provider.
- **No code execution from metadata** — descriptions and schemas are data.
  Schema text is never evaluated; unknown transports and non-object input
  schemas are skipped.
- **Limits** — per-call timeout, per-provider concurrency semaphore, argument
  size cap, result size cap, and a circuit breaker (5 consecutive failures parks
  a provider for 60s so a dead endpoint can't burn a whole turn).
- **No retries** — a failed tool call is reported, never resent. Hermes cannot
  know whether an arbitrary tool is idempotent.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `UTCP support is disabled` | `utcp.enabled: false` or section missing. |
| `UTCP support requires the optional 'utcp' extra` | Run `hermes pm install --extra utcp`. |
| `url must use http(s)` / `plain http is only allowed for loopback` | Use HTTPS, or a loopback host for local dev. |
| `references unset environment variable` | Add the variable to `~/.hermes/.env`. |
| `transport 'x' is not allowed` | Only `http`, `sse`, `streamable_http` are accepted. |
| Provider connected, 0 tools | The manual was empty, every tool was filtered (`tools.include/exclude`), or tools were rejected (see log). `hermes utcp tools` shows what registered. |
| `error_type: "auth"` | Provider returned 401/403, or a referenced variable is missing. |
| `error_type: "timeout"` | Raise the provider's `timeout`; the SDK's own HTTP ceiling is 30s per request. |
| Tool missing from the model's list | Run `hermes utcp status`; if the provider connected late, `hermes utcp refresh`. Large catalogs are also deferred behind `tool_search`. |

Logs: `~/.hermes/logs/agent.log` (discovery and per-tool warnings).

## Known limitations

- Transports are limited to the HTTP family (`http`, `sse`,
  `streamable_http`). CLI, file, socket, WebSocket, GraphQL and MCP transports
  exist in the UTCP SDK but are intentionally not enabled.
- Result streaming (`call_tool_streaming`) is not surfaced as a Hermes tool
  result because the registry's handler contract is a single JSON payload.
- Tool lists are refreshed at startup or on explicit `hermes utcp refresh`;
  there is no push/invalidation channel in UTCP.
- Tool arguments are validated structurally (required fields, primitive types,
  enums, closed objects) rather than with a full JSON-Schema validator.
