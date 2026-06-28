---
sidebar_position: 9
title: "Realtime Orchestration API"
description: "Low-latency conversation coordination for voice and chat clients"
---

# Realtime Orchestration API

Hermes' realtime orchestration API gives external clients a low-latency
"talk now, work in the background" surface. It is designed for voice calls,
live chat, and other interactive clients where waiting for a full tool-using
agent turn would create dead air.

The realtime API does not replace the normal `AIAgent` loop. It coordinates a
live conversation around that loop:

1. A client sends the latest user text to `POST /v1/realtime/turn`.
2. Hermes returns a short talker response and an action such as `start_task`,
   `confirm`, `cancel`, or `none`.
3. The client or a worker starts slow tool work through `/v1/runs`, an existing
   session surface, or its own background executor.
4. Workers patch the live context document and emit progress through realtime
   tasks/events.
5. The next talker turn sees the updated live context and can report verified
   progress without waiting for slow work in the foreground.

## Components

- `LiveContextStore` keeps a mutable per-session context document. It is
  bounded in memory and intended as live working state, not long-term memory.
- `RealtimeLoop` builds the no-tools talker/orchestrator prompt and parses the
  resulting JSON decision.
- `RealtimeTask` records background work intent and lifecycle state for clients
  that need progress UI.
- `/v1/realtime/events` streams context/task events over Server-Sent Events.

The implementation is intentionally platform-neutral. Voice clients, chat
clients, and dashboards can use the same API.

## Endpoints

All endpoints use the API server authentication rules. If an API key is
configured, send `Authorization: Bearer <key>`.

Use `X-Hermes-Session-Key` to scope live context and realtime tasks to a
conversation:

```bash
curl -sS http://localhost:8080/v1/realtime/turn \
  -H "Authorization: Bearer $HERMES_API_KEY" \
  -H "X-Hermes-Session-Key: voice:demo" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Can you check the latest invoice email?",
    "transcript": [
      {"role": "user", "text": "Can you check the latest invoice email?"}
    ],
    "timeout": 1.5
  }'
```

Typical response:

```json
{
  "object": "hermes.realtime.turn",
  "session_key": "voice:demo",
  "say": "I am checking that now. This may take a moment, and I will keep you updated.",
  "action": "start_task",
  "action_request": "Can you check the latest invoice email?",
  "task": {
    "object": "hermes.realtime.task",
    "task_id": "rtask_...",
    "status": "queued"
  },
  "latency_seconds": 0.012,
  "degraded": true
}
```

`degraded: true` means Hermes did not use an optional talker model and returned
the deterministic fallback decision instead. This is intentional for live
clients: provider auto-detection can block on credential discovery, so Hermes
only calls the no-tools talker model when a concrete `provider` is supplied in
the request or `HERMES_REALTIME_TALKER_PROVIDER` is configured.

### `POST /v1/realtime/turn`

Request fields:

| Field | Type | Purpose |
| --- | --- | --- |
| `input` | string | Latest user text. Aliases: `message`, `text`, `user_text`. |
| `transcript` | array | Recent conversation turns for the talker prompt. |
| `context_patch` | object | Optional patch applied before the turn. |
| `base_prompt` | string | Optional talker/orchestrator prompt override. |
| `provider` | string | Optional explicit auxiliary provider for the talker model. |
| `model` | string | Optional talker model name. |
| `timeout` | number | Max seconds for the talker model. |
| `max_tokens` | number | Talker response token cap. |
| `extra_body` | object | Provider-specific passthrough body. |

The talker model should return one JSON object:

```json
{
  "say": "Short user-facing response.",
  "action": "none|start_task|confirm|cancel",
  "action_request": "Background work request, if any.",
  "context_patch": {}
}
```

The talker should not use tools or make unverified claims. Slow work belongs in
the normal agent loop or a background worker.

### `GET /v1/realtime/context`

Returns the current live context document for the session key.

### `POST /v1/realtime/context`

Patches the live context document. Patches support a small append form for list
fields:

```json
{
  "known_facts": {
    "_append": ["The invoice thread was found in the user's inbox."]
  }
}
```

### `GET /v1/realtime/tasks`

Lists realtime task records for the session key.

### `POST /v1/realtime/tasks`

Creates a task record without running tools directly:

```json
{
  "request": "Research the latest invoice email.",
  "source": "voice-client"
}
```

### `GET /v1/realtime/events`

Streams recent and future realtime events as Server-Sent Events. Event names
include `context.updated`, `task.created`, `task.updated`, `turn.degraded`, and
`turn.completed`.

## Client Pattern

For a live voice or chat client:

1. Send every completed user utterance to `/v1/realtime/turn`.
2. Speak `say` immediately.
3. If `action` is `start_task`, start the slow work asynchronously.
4. Patch `/v1/realtime/context` as workers learn verified facts.
5. Subscribe to `/v1/realtime/events` for UI/progress updates.
6. On the next user turn, include the recent transcript and the same session key.

This keeps the interactive path fast while preserving Hermes' normal
tool-using depth for work that needs it.
