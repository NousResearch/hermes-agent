---
sidebar_position: 7
title: "Gateway Internals"
description: "How the messaging gateway boots, authorizes users, routes sessions, and delivers messages"
---

# Gateway Internals

The messaging gateway is the long-running process that connects Hermes to external platforms.

Key files:

- `gateway/run.py`
- `gateway/config.py`
- `gateway/session.py`
- `gateway/delivery.py`
- `gateway/pairing.py`
- `gateway/channel_directory.py`
- `gateway/hooks.py`
- `gateway/mirror.py`
- `gateway/platforms/*`

## Core responsibilities

The gateway process is responsible for:

- loading configuration from `.env`, `config.yaml`, and `gateway.json`
- starting platform adapters
- authorizing users
- routing incoming events to sessions
- maintaining per-chat session continuity
- dispatching messages to `AIAgent`
- running cron ticks and background maintenance tasks
- mirroring/proactively delivering output to configured channels

## Config sources

The gateway has a multi-source config model:

- environment variables
- `~/.hermes/gateway.json`
- selected bridged values from `~/.hermes/config.yaml`

## Session routing

`gateway/session.py` and `GatewayRunner` cooperate to map incoming messages to active session IDs.

Session keying can depend on:

- platform
- user/chat identity
- thread/topic identity
- special platform-specific routing behavior

## Authorization layers

The gateway can authorize through:

- platform allowlists
- gateway-wide allowlists
- DM pairing flows
- explicit allow-all settings

Pairing support is implemented in `gateway/pairing.py`.

## Delivery path

Outgoing deliveries are handled by `gateway/delivery.py`, which knows how to:

- deliver to a home channel
- resolve explicit targets
- mirror some remote deliveries back into local history/session tracking

## Hooks

Gateway events emit hook callbacks through `gateway/hooks.py`. Hooks are local trusted Python code and can observe or extend gateway lifecycle events.

## Background maintenance

The gateway also runs maintenance tasks such as:

- cron ticking
- cache refreshes
- session expiry checks
- proactive memory flush before reset/expiry

## Honcho interaction

When a memory provider plugin (e.g. Honcho) is enabled, the gateway creates an AIAgent per incoming message with the same session ID. The memory provider's `initialize()` receives the session ID and creates the appropriate backend session. Tools are routed through the `MemoryManager`, which handles all provider lifecycle hooks (prefetch, sync, session end).

### Memory provider session routing

Memory provider tools (e.g. `honcho_profile`, `viking_search`) are routed through the MemoryManager in `_invoke_tool()`:

```
AIAgent._invoke_tool()
  → self._memory_manager.handle_tool_call(name, args)
    → provider.handle_tool_call(name, args)
```

Each memory provider manages its own session lifecycle internally. The `initialize()` method receives the session ID, and `on_session_end()` handles cleanup and final flush.

### Memory flush lifecycle

When a session is reset, resumed, or expires, the gateway flushes built-in memories before discarding context. The flush creates a temporary `AIAgent` that runs a memory-only conversation turn. The memory provider's `on_session_end()` hook fires during this process, giving external providers a chance to persist any buffered data.

## Related docs

- [Session Storage](./session-storage.md)
- [Cron Internals](./cron-internals.md)
- [ACP Internals](./acp-internals.md)
