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

When Honcho is enabled, the gateway keeps persistent Honcho managers aligned with session lifetimes and platform-specific session keys.

### Session routing

Honcho tools (`honcho_profile`, `honcho_search`, `honcho_context`, `honcho_conclude`) need to execute against the correct user's Honcho session. In a multi-user gateway, the process-global module state in `tools/honcho_tools.py` is insufficient — multiple sessions may be active concurrently.

The solution threads session context through the call chain:

```
AIAgent._invoke_tool()
  → handle_function_call(honcho_manager=..., honcho_session_key=...)
    → registry.dispatch(**kwargs)
      → _handle_honcho_*(args, **kw)
        → _resolve_session_context(**kw)   # prefers explicit kwargs over module globals
```

`_resolve_session_context()` in `honcho_tools.py` checks for `honcho_manager` and `honcho_session_key` in the kwargs first, falling back to the module-global `_session_manager` / `_session_key` for CLI mode where there's only one session.

### Memory flush lifecycle

When a session is reset, resumed, or expires, the gateway flushes memories before discarding context. The flush creates a temporary `AIAgent` with:

- `session_id` set to the old session's ID (so transcripts load correctly)
- `honcho_session_key` set to the gateway session key (so Honcho writes go to the right place)
- `sync_honcho=False` passed to `run_conversation()` (so the synthetic flush turn doesn't write back to Honcho's conversation history)

After the flush completes, any queued Honcho writes are drained and the gateway-level Honcho manager is shut down for that session key.

## Related docs

- [Session Storage](./session-storage.md)
- [Cron Internals](./cron-internals.md)
- [ACP Internals](./acp-internals.md)
