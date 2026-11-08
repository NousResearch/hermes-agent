---
sidebar_position: 66
title: "UTCP Tool Provider Internals"
description: "How Hermes consumes UTCP providers natively: architecture, SDK mapping, lifecycle, and security boundaries"
---

# Native UTCP tool provider — design

Status: implemented on `feature/native-utcp-support`.
Scope: Hermes consumes UTCP providers directly; no MCP server in the middle.

## 1. Current tool architecture (as implemented today)

- `tools/registry.py` is the single source of truth: `ToolRegistry.register()`
  stores `(schema, handler, toolset, check_fn, is_async, …)`; `dispatch()`
  executes a handler and normalizes results to a JSON string (or the multimodal
  envelope).
- Tool modules self-register at import time (`discover_builtin_tools()` AST-
  scans `tools/*.py`), **or** register dynamically from a provider. `mcp_tool.py`
  is the existing dynamic provider: it registers `mcp__<server>__<tool>` into the
  `mcp-<server>` toolset, with a per-server `check_fn` and a background asyncio
  loop for the async MCP SDK.
- Toolset resolution (`toolsets.py`) treats any registry-only toolset as a
  plugin toolset, and `hermes_cli/tools_config.py::_get_platform_tools()` maps
  enabled MCP server names onto their toolsets, so MCP tools reach
  `get_tool_definitions()` without being in the static `TOOLSETS` map.
- Discovery lifecycle (`hermes_cli/mcp_startup.py`): background thread at
  startup, bounded join before the agent snapshots its tool list.

UTCP mirrors all four points rather than introducing a parallel mechanism.

## 2. UTCP SDK capabilities (verified against utcp 1.2.0 / utcp-http 1.1.15)

| Capability | API | Notes |
|---|---|---|
| Client | `UtcpClient.create(root_dir, config={"variables": {...}})` | Async; plugins are discovered via the `utcp.plugins` entry-point group. |
| Discovery | `await client.register_manual(template)` → `RegisterManualResult(manual.tools)` | HTTP transport accepts a UTCP manual (`utcp_version` + `tools`) **or** an OpenAPI document, converted on the fly. |
| Tool model | `Tool(name, description, inputs: JsonSchema, outputs, tags, tool_call_template)` | `inputs` is a Pydantic JSON-Schema subset; serialized with `model_dump(by_alias=True)`. |
| Execution | `await client.call_tool("<manual>.<tool>", args)` | Returns parsed JSON (dict/list/scalars) or text. No timeout parameter. |
| Auth | `ApiKeyAuth` (header/query/cookie), `BasicAuth`, `OAuth2Auth` (cached, refreshed) | Declared per call template (`auth`, `auth_tools`). |
| Variables | `${VAR}` / `$VAR`, namespaced per manual: `<manual>_<VAR>` | Resolution order: config `variables` → loaders → `os.environ`. |
| Transports | `http`, `sse`, `streamable_http` (utcp-http); `cli`, `file`, `gql`, `mcp`, `socket`, `text`, `websocket` (separate plugins) | Only the HTTP family is stable *and* cannot execute local code. |
| Errors | Raises `aiohttp.ClientResponseError` (body included), `UtcpVariableNotFound`, `ValueError("Tool not found")` | No structured error type; Hermes classifies. |
| Timeouts | Discovery 10s, HTTP call 30s (hardcoded in utcp-http) | Hermes enforces its own, smaller-or-equal timeout via `asyncio.wait_for`. |
| Teardown | `await client.close()` | Closes only the protocol instances the client created. |

Unsupported / not claimed: streaming results, push tool-list invalidation,
transports outside the HTTP family, full JSON-Schema validation of arguments.

## 3. Integration architecture

```
Agent runtime
   |
get_tool_definitions()  <-- toolsets.resolve_toolset("utcp-<provider>")
   |
ToolRegistry (tools/registry.py)          <-- unchanged
   |-- built-in tools
   |-- mcp-<server> tools                 (tools/mcp_tool.py)
   `-- utcp-<provider> tools              (tools/utcp_tool.py)
             |
        one UtcpClient per provider  (SDK: utcp 1.2.0 + utcp-http 1.1.15)
             |
        transport: http | sse | streamable_http
```

Key decisions:

1. **One `UtcpClient` per provider.** The SDK's client owns transport instances,
   the tool repository, and the variable scope. Per-provider clients give
   independent failure domains, independent credential scopes, and independent
   teardown — the isolation MCP gets from per-server tasks.
2. **One background asyncio loop** for all clients, matching `mcp_tool.py`.
   Registry handlers are synchronous, so calls are scheduled with
   `run_coroutine_threadsafe(...).result(timeout)`.
3. **No new execution path.** Handlers are ordinary registry handlers; results
   are normalized to the same `{"result": ...}` JSON envelope MCP uses.
4. **Provider-level toolset + alias.** Tools register into `utcp-<provider>`
   with the bare provider name as a toolset alias, so `--toolsets utcp-demo`
   and `--toolsets demo` both resolve, and `_get_platform_tools()` can map
   configured provider names onto toolsets the way it does for MCP servers.
5. **Discovery at lifecycle boundaries.** The same entrypoints that start MCP
   discovery start UTCP discovery; the join before agent construction is
   bounded by `utcp.discovery_timeout` / `single_query_discovery_timeout`.

## 4. Security boundaries

| Risk | Control |
|---|---|
| Secret exfiltration via provider metadata | Declared-variable allowlist: every `${VAR}` in the provider *and* in each discovered tool template must be declared under `variables:`. Rejected otherwise. |
| Secret in config | `variables:` accepts only `env:NAME`; literal values are a config error. |
| Secret in logs / model context | Configured values + `Bearer …` / `sk-…` / `api_key=…` shapes redacted from all error strings; status shows variable *names* only. |
| SSRF / plaintext transport | HTTPS, or plain HTTP to loopback only; no credentials in URL; no placeholder hosts; the SDK re-validates the resolved URL and each redirect per call. |
| Local code execution | CLI/file/socket/MCP transports are not allowlisted at all. |
| Tool-name shadowing | Registry rejects cross-toolset shadowing; UTCP additionally pre-checks and skips collisions with built-ins and other toolsets. |
| Resource exhaustion | Per-call timeout, per-provider semaphore, argument-size cap, result-size cap, 5-failure/60s circuit breaker. |
| Non-idempotent retries | None. Errors are reported, never retried. |

## 5. Modules

| File | Purpose |
|---|---|
| `tools/utcp_tool.py` | Config model, discovery, registration, execution, redaction, status. |
| `hermes_cli/utcp_startup.py` | Background discovery + bounded join, mirroring `mcp_startup.py`. |
| `hermes_cli/utcp_config.py`, `hermes_cli/subcommands/utcp.py` | `hermes utcp list/status/tools/refresh/add/remove`. |
| `hermes_cli/config_defaults.py` | `utcp:` defaults (disabled). |
| `hermes_cli/tools_config.py` | `enabled_utcp_provider_names()` + platform resolution. |
| `cron/scheduler.py` | Per-job toolset merge includes UTCP providers. |
| `pyproject.toml`, `uv.lock`, `pm/extras.py` | Optional `utcp` extra + `pm.extras` prepare-on-demand. |
| Lifecycle hooks | `hermes_cli/main.py`, `cli.py`, `hermes_cli/console_engine.py`, `acp_adapter/session.py`, `gateway/run.py`, `tui_gateway/{entry,agent_factory,session_registry}.py` |

## 6. Compatibility

- No existing module changes behavior when `utcp.enabled` is false (the default):
  discovery is config-gated before any SDK import, and every hook is additive.
- Optional dependency: `utcp`/`utcp-http` are imported only on first discovery.
- The registry, toolset, and dispatch contracts are unchanged.
