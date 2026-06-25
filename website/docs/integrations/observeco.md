---
title: "ObserveCo"
sidebar_label: "ObserveCo"
sidebar_position: 4
---

# ObserveCo

> Real-time observability for your Hermes agents — token usage, session lifecycle, tool calls, errors, and gateway health, all in one dashboard.

ObserveCo is an open-source observability platform for local AI agents. The `observability/observeco` Hermes plugin exports telemetry from every conversation to an ObserveCo OTEL listener, which writes to a local SQLite database that powers the ObserveCo dashboard.

## Quick Start

```bash
# 1. Enable the plugin
hermes plugins enable observability/observeco

# 2. Set the endpoint (default: http://127.0.0.1:4318)
echo 'HERMES_OBSERVECO_ENDPOINT=http://127.0.0.1:4318' >> ~/.hermes/.env

# 3. Restart the gateway
launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway

# 4. Start the OTEL listener (if not already running)
observeco otel listen start --port 4318

# 5. Open the dashboard
open http://localhost:8123
```

## How It Works

```
Hermes Gateway
  └── observability/observeco plugin
        ├── post_api_request  → token usage (input, output, cache, cost, model)
        ├── api_request_error → LLM failures
        ├── on_session_start  → session begin
        ├── on_session_end    → session end
        ├── pre_tool_call     → tool invoked
        ├── post_tool_call    → tool result
        ├── subagent_start    → child spawned
        ├── subagent_stop     → child done
        └── pre_gateway_dispatch → message routed
              │
              │ OTLP/HTTP POST
              ▼
        OTEL Listener (:4318)
              │ writes
              ▼
        pulse.db (SQLite)
              │ reads
              ▼
        ObserveCo Dashboard (:8123)
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HERMES_OBSERVECO_ENDPOINT` | No | `http://127.0.0.1:4318` | OTLP HTTP endpoint for span export |
| `HERMES_OBSERVECO_SERVICE` | No | `hermes-agent` | `service.name` in OTLP resource attributes |
| `HERMES_OBSERVECO_DISABLED` | No | — | Set to `true` to disable export without removing the plugin |

**Backward compatibility:** If `HERMES_OBSERVECO_ENDPOINT` is not set, the plugin falls back to `HERMES_OTEL_ENDPOINT`. This lets you migrate from the old `observability/otel` plugin without changing your `.env`.

### Plugin Activation

```bash
hermes plugins enable observability/observeco
```

The plugin is opt-in. It will not load until you enable it and restart the gateway.

## What Gets Exported

Every hook fires a fire-and-forget OTLP/HTTP POST to the configured endpoint. The plugin uses stdlib only (`urllib.request`, `json`, `logging`) — no external dependencies.

### Token Usage (`post_api_request`)

Fires after every LLM API call. This is the primary data source for the ObserveCo Token Analytics tab.

| OTLP Attribute | Source | Example |
|----------------|--------|---------|
| `llm.usage.token_count.prompt` | `usage.input_tokens` | `142791` |
| `llm.usage.token_count.completion` | `usage.output_tokens` | `131` |
| `llm.usage.cache_creation_input_tokens` | `usage.cache_write_tokens` | `0` |
| `llm.usage.cache_read_input_tokens` | `usage.cache_read_tokens` | `0` |
| `gen_ai.system` | `provider` | `custom` |
| `gen_ai.request.model` | `model` | `deepseek-v4-flash` |
| `hermes.session_id` | `session_id` | `20260625_114604_7edb00` |
| `hermes.cost_usd` | `usage.estimated_cost_usd` | `0.0214383` |
| `hermes.api_duration_ms` | `api_duration` | `1234` |
| `hermes.finish_reason` | `finish_reason` | `stop` |
| `hermes.reasoning_tokens` | `usage.reasoning_tokens` | `0` |

### LLM Errors (`api_request_error`)

Fires when an LLM API call fails. The span has `status_code=2` (ERROR) and includes the error message.

| OTLP Attribute | Source |
|----------------|--------|
| `hermes.error` | `error` (stringified) |
| `gen_ai.request.model` | `model` |
| `gen_ai.system` | `provider` |

### Session Lifecycle (`on_session_start`, `on_session_end`, `on_session_finalize`)

Fires at session boundaries. The `trace_id` is derived from `session_id` so all spans in a session share the same trace.

| OTLP Attribute | Source |
|----------------|--------|
| `hermes.session_id` | `session_id` |
| `hermes.event` | `session_start` / `session_end` / `session_finalize` |

### Tool Calls (`pre_tool_call`, `post_tool_call`)

Fires before and after every tool execution. The span name is `tool.<tool_name>`.

| OTLP Attribute | Source |
|----------------|--------|
| `hermes.tool_name` | `tool_name` |
| `hermes.tool_result` | `post_tool_call` only — summary of result |

### Subagents (`subagent_start`, `subagent_stop`)

Fires when a subagent is spawned or completes.

| OTLP Attribute | Source |
|----------------|--------|
| `hermes.parent_session_id` | `parent_session_id` |
| `hermes.subagent_goal` | `goal` (truncated to 200 chars) |

### Gateway Dispatch (`pre_gateway_dispatch`)

Fires for every incoming message before it's routed to an agent session. This hook is **observer-only** — it does not modify dispatch behaviour.

| OTLP Attribute | Source |
|----------------|--------|
| `hermes.platform` | `event.platform` |
| `hermes.chat_id` | `event.chat_id` |
| `hermes.user_id` | `event.user_id` |
| `hermes.topic_id` | `event.topic_id` |
| `hermes.text_preview` | `event.text` (first 100 chars) |

## Migrating from the OTEL Plugin

If you were using `observability/otel`, migration is straightforward:

```bash
# 1. Enable the new plugin
hermes plugins enable observability/observeco

# 2. Disable the old one
hermes plugins disable observability/otel

# 3. Restart the gateway
launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway
```

The ObserveCo plugin reads `HERMES_OTEL_ENDPOINT` as a fallback, so your existing `.env` configuration continues to work without changes.

## Troubleshooting

### No data in the dashboard

1. **Is the plugin enabled?**
   ```bash
   hermes plugins list | grep observeco
   ```

2. **Is the OTEL listener running?**
   ```bash
   lsof -i :4318 | grep LISTEN
   ```

3. **Is the endpoint configured?**
   ```bash
   grep HERMES_OBSERVECO_ENDPOINT ~/.hermes/.env
   ```

4. **Check the gateway log for plugin registration:**
   ```bash
   grep -i "observeco" ~/.hermes/logs/gateway.log
   ```

5. **Test with a manual span:**
   ```bash
   curl -s http://127.0.0.1:4318/v1/traces -X POST \
     -H "Content-Type: application/json" \
     -d '{"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"test"}}]},"scopeSpans":[{"scope":{"name":"test"},"spans":[{"traceId":"abc123","spanId":"def456","name":"test","startTimeUnixNano":"1000000000","endTimeUnixNano":"1000000000","attributes":[{"key":"llm.usage.token_count.prompt","value":{"intValue":100}},{"key":"llm.usage.token_count.completion","value":{"intValue":50}},{"key":"gen_ai.request.model","value":{"stringValue":"test-model"}}],"status":{"code":1}}]}]}]}'
   ```

6. **Check the database:**
   ```bash
   sqlite3 ~/Library/Application\ Support/observeco/pulse.db \
     "SELECT source, model, input_tokens, output_tokens, datetime(recorded_at, 'unixepoch') \
      FROM token_logs WHERE source='otel' ORDER BY recorded_at DESC LIMIT 5;"
   ```

### Plugin loads but no spans are sent

The plugin is fire-and-forget with a 5-second timeout. Failures are logged at DEBUG level:

```bash
tail -f ~/.hermes/logs/gateway.log | grep -i "observeco"
```

### Stale listener

If the OTEL listener was started before a code update, it may be running stale code. Restart it:

```bash
observeco otel listen stop
observeco otel listen start --port 4318
```

## Data Flow Summary

```
Hermes LLM call → post_api_request hook → OTLP span → OTEL listener → pulse.db → Dashboard
```

The entire pipeline is local. No data leaves your machine unless you configure the endpoint to point to a remote server.
