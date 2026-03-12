---
sidebar_position: 15
title: "Streaming"
description: "Token-by-token live response display across all platforms"
---

# Streaming Responses

When enabled, hermes-agent streams LLM responses token-by-token instead of waiting for the full generation. Users see the response typing out live — the same experience as ChatGPT, Claude, or Gemini.

Streaming is **disabled by default** and can be enabled globally or per-platform.

## How It Works

```
LLM generates tokens → callback fires per token → queue → consumer displays

Telegram/Discord/Slack:
  Token arrives → Accumulate → Every 1.5s, edit the message with new text + ▌ cursor
  Done → Final edit removes cursor

API Server:
  Token arrives → SSE event sent to client immediately
  Done → finish chunk + [DONE]
```

The agent's internal operation doesn't change — tools still execute normally, memory and skills work as before. Streaming only affects how the **final text response** is delivered to the user.

## Enable Streaming

### Option 1: Environment variable

```bash
# Enable for all platforms
export HERMES_STREAMING_ENABLED=true
hermes gateway
```

### Option 2: config.yaml

```yaml
streaming:
  enabled: true     # Master switch
```

### Option 3: Per-platform

```yaml
streaming:
  enabled: false    # Off by default
  telegram: true    # But on for Telegram
  discord: true     # And Discord
  api_server: true  # And the API server
```

## Platform Support

| Platform | Streaming Method | Rate Limit | Notes |
|----------|-----------------|------------|-------|
| **Telegram** | Progressive message editing | ~20 edits/min | 1.5s edit interval, ▌ cursor |
| **Discord** | Progressive message editing | 5 edits/5s | 1.5s edit interval |
| **Slack** | Progressive message editing | ~50 calls/min | 1.5s edit interval |
| **API Server** | SSE (Server-Sent Events) | No limit | Real token-by-token events |
| **WhatsApp** | ❌ Not supported | — | No message editing API |
| **Home Assistant** | ❌ Not supported | — | No message editing API |
| **CLI** | ❌ Not yet implemented | — | KawaiiSpinner provides feedback |

Platforms without message editing support automatically fall back to non-streaming (the response appears all at once, as before).

## What Users See

### Telegram/Discord/Slack

1. Agent starts working (typing indicator shows)
2. After ~20 tokens, a message appears with partial text and a ▌ cursor
3. Every 1.5 seconds, the message is edited with more accumulated text
4. When the response is complete, the cursor disappears

Tool progress messages still work alongside streaming — tool names/previews appear as before, and the streamed response is shown in a separate message.

### API Server (frontends like Open WebUI)

When `stream: true` is set in the request, the API server returns Server-Sent Events:

```
data: {"choices":[{"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"content":"Here"}}]}

data: {"choices":[{"delta":{"content":" is"}}]}

data: {"choices":[{"delta":{"content":" the"}}]}

data: {"choices":[{"delta":{"content":" answer"}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Frontends like Open WebUI display this as live typing.

## How It Works Internally

### Architecture

```
┌─────────────┐    stream_callback(delta)    ┌──────────────────┐
│  LLM API    │ ──────────────────────────► │   queue.Queue()  │
│  (stream)   │    (runs in agent thread)   │   (thread-safe)  │
└─────────────┘                              └────────┬─────────┘
                                                      │
                                       ┌──────────────┼──────────┐
                                       │              │          │
                                 ┌─────▼─────┐ ┌─────▼────┐ ┌──▼──────┐
                                 │  Gateway   │ │ API Svr  │ │  CLI    │
                                 │  edit msg  │ │ SSE evt  │ │ (TODO)  │
                                 └───────────┘ └──────────┘ └─────────┘
```

1. `AIAgent.__init__` accepts an optional `stream_callback` function
2. When set, `_interruptible_api_call()` routes to `_run_streaming_chat_completion()` instead of the normal non-streaming path
3. The streaming method calls the OpenAI API with `stream=True`, iterates chunks, and calls `stream_callback(delta_text)` for each text token
4. Tool call deltas are accumulated silently (no streaming for tool arguments)
5. When the stream ends, `stream_callback(None)` signals completion
6. The method returns a fake response object compatible with the existing code path
7. If streaming fails for any reason, it falls back to a normal non-streaming API call

### Thread Safety

The agent runs in a background thread (via `_interruptible_api_call`). The consumer (gateway async task, API server SSE writer) runs in the main event loop. A `queue.Queue` bridges them — it's thread-safe by design.

### Graceful Fallback

If the LLM provider doesn't support `stream=True` or the streaming connection fails, the agent automatically falls back to a non-streaming API call. The user gets the response normally, just without the live typing effect. No error is shown.

## Configuration Reference

```yaml
streaming:
  enabled: false          # Master switch (default: off)

  # Per-platform overrides (optional):
  telegram: true          # Enable for Telegram
  discord: true           # Enable for Discord
  slack: true             # Enable for Slack
  api_server: true        # Enable for API server

  # Tuning (optional):
  edit_interval: 1.5      # Seconds between message edits (default: 1.5)
  min_tokens: 20          # Tokens before first display (default: 20)
```

| Variable | Default | Description |
|----------|---------|-------------|
| `HERMES_STREAMING_ENABLED` | `false` | Master switch via env var |
| `streaming.enabled` | `false` | Master switch via config |
| `streaming.<platform>` | _(unset)_ | Per-platform override |
| `streaming.edit_interval` | `1.5` | Seconds between Telegram/Discord edits |
| `streaming.min_tokens` | `20` | Minimum tokens before first message |

## Interaction with Other Features

### Tool Execution

When the agent calls tools (terminal, file operations, web search, etc.), no text tokens are generated — tool arguments are accumulated silently. Tool progress messages continue to work as before. After tools finish, the next LLM call may produce the final text response, which streams normally.

### Context Compression

Compression happens between API calls, not during streaming. No interaction.

### Interrupts

If the user sends a new message while streaming, the agent is interrupted. The HTTP connection is closed (stopping token generation), accumulated text is shown as-is, and the new message is processed.

### Prompt Caching

Streaming doesn't affect prompt caching — the request is identical, just with `stream=True` added.

### Responses API (Codex)

The Codex/Responses API streaming path also supports the `stream_callback`. Token deltas from `response.output_text.delta` events are emitted via the callback.

## Troubleshooting

### Streaming isn't working

1. Check the config: `streaming.enabled: true` in config.yaml or `HERMES_STREAMING_ENABLED=true`
2. Check per-platform: `streaming.telegram: true` overrides the master switch
3. Restart the gateway after changing config
4. Check logs for "Streaming failed, falling back" — indicates the provider may not support streaming

### Response appears twice

If you see the response both in a progressively-edited message AND as a separate final message, this is a bug. The streaming system should suppress the normal send when tokens were delivered via streaming. Please file an issue.

### Messages update too slowly

The default edit interval is 1.5 seconds (to respect platform rate limits). You can lower it in config:

```yaml
streaming:
  edit_interval: 1.0   # Faster updates (may hit rate limits)
```

Going below 1.0s risks Telegram rate limiting (429 errors).

### No streaming on WhatsApp/HomeAssistant

These platforms don't support message editing, so streaming automatically falls back to non-streaming. This is expected behavior.
