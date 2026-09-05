---
sidebar_position: 14
title: "API Server"
description: "Expose hermes-agent as an OpenAI-compatible API for any frontend"
---

# API Server

The API server exposes hermes-agent as an OpenAI-compatible HTTP endpoint. Any frontend that speaks the OpenAI format — Open WebUI, LobeChat, LibreChat, NextChat, ChatBox, and hundreds more — can connect to hermes-agent and use it as a backend.

Your agent handles requests with its full toolset (terminal, file operations, web search, memory, skills) and returns the final response. When streaming, tool progress indicators appear inline so frontends can show what the agent is doing.

:::tip One backend covers models + tools
Hermes itself needs a configured provider and tool backends for the API server to be useful. A [Nous Portal](/user-guide/features/tool-gateway) subscription handles both — 300+ models plus web/image/TTS/browser via the Tool Gateway. Run `hermes setup --portal` once before starting the API server and frontends like Open WebUI or LobeChat get a fully tool-equipped backend.
:::

## Quick Start

### 1. Enable the API server

Add to `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# Optional: only if a browser must call Hermes directly
# API_SERVER_CORS_ORIGINS=http://localhost:3000
```

### 2. Start the gateway

```bash
hermes gateway
```

You'll see:

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. Connect a frontend

Point any OpenAI-compatible client at `http://localhost:8642/v1`:

```bash
# Test with curl
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change-me-local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Or connect Open WebUI, LobeChat, or any other frontend — see the [Open WebUI integration guide](/user-guide/messaging/open-webui) for step-by-step instructions.

## Endpoints

### POST /v1/chat/completions

Standard OpenAI Chat Completions format. Stateless — the full conversation is included in each request via the `messages` array.

**Request:**
```json
{
  "model": "hermes-agent",
  "messages": [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "Write a fibonacci function"}
  ],
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "hermes-agent",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Here's a fibonacci function..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
}
```

**Inline image input:** user messages may send `content` as an array of `text` and `image_url` parts. Both remote `http(s)` URLs and `data:image/...` URLs are supported:

```json
{
  "model": "hermes-agent",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}}
      ]
    }
  ]
}
```

Uploaded files (`file` / `input_file` / `file_id`) and non-image `data:` URLs return `400 unsupported_content_type`.

**Streaming** (`"stream": true`): Returns Server-Sent Events (SSE) with token-by-token response chunks. For **Chat Completions**, the stream uses standard `chat.completion.chunk` events plus Hermes' custom `hermes.tool.progress` event for tool-start UX. For **Responses**, the stream uses OpenAI Responses event types such as `response.created`, `response.output_text.delta`, `response.output_item.added`, `response.output_item.done`, and `response.completed`.

**Tool progress in streams**:
- **Chat Completions**: Hermes emits `event: hermes.tool.progress` for tool-start visibility without polluting persisted assistant text.
- **Responses**: Hermes emits spec-native `function_call` and `function_call_output` output items during the SSE stream, so clients can render structured tool UI in real time.

### POST /v1/responses

OpenAI Responses API format. Supports server-side conversation state via `previous_response_id` — the server stores full conversation history (including tool calls and results) so multi-turn context is preserved without the client managing it.

**Request:**
```json
{
  "model": "hermes-agent",
  "input": "What files are in my project?",
  "instructions": "You are a helpful coding assistant.",
  "store": true
}
```

**Response:**
```json
{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "model": "hermes-agent",
  "output": [
    {"type": "function_call", "status": "completed", "name": "terminal", "arguments": "{\"command\": \"ls\"}", "call_id": "call_1"},
    {"type": "function_call_output", "status": "completed", "call_id": "call_1", "output": "README.md src/ tests/"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Your project has..."}]}
  ],
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

Tool calls in the `output` array were already executed server-side by the Hermes agent — they are replayed with `"status": "completed"` for structured tool UI, never as pending calls for the client to execute.

**Inline image input:** `input[].content` can contain `input_text` and `input_image` parts. Both remote URLs and `data:image/...` URLs are supported:

```json
{
  "model": "hermes-agent",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this screenshot."},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K..."}
      ]
    }
  ]
}
```

Uploaded files (`input_file` / `file_id`) and non-image `data:` URLs return `400 unsupported_content_type`.

#### Multi-turn with previous_response_id

Chain responses to maintain full context (including tool calls) across turns:

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

The server reconstructs the full conversation from the stored response chain — all previous tool calls and results are preserved. Chained requests also share the same session, so multi-turn conversations appear as a single entry in the dashboard and session history.

#### Named conversations

Use the `conversation` parameter instead of tracking response IDs:

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

The server automatically chains to the latest response in that conversation. Like the `/title` command for gateway sessions.

### GET /v1/responses/\{id\}

Retrieve a previously stored response by ID.

### DELETE /v1/responses/\{id\}

Delete a stored response.

### GET /v1/models

Lists the agent as an available model. The advertised model name defaults to the [profile](/user-guide/profiles) name (or `hermes-agent` for the default profile). Required by most frontends for model discovery.

`/v1/models` is intentionally the cheap OpenAI-compat surface. It does **not**
enumerate every authenticated provider/model combination Hermes can route to,
and it does not do pricing or capability enrichment.

### GET /api/model/options

Hermes-aware clients can request the same curated provider/model inventory used
by the dashboard and TUI. This route uses the API server's normal bearer
authentication and returns provider rows, model capability hints, and pricing
metadata that do not belong in the OpenAI-compatible `/v1/models` response:

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options"
```

That payload is the same substrate the dashboard Models page and the TUI
`model.options` RPC use. It returns authenticated providers, curated model
lists, per-model pricing, and model capability hints.

Normal opens are intentionally conservative for custom providers: Hermes probes
only the **currently selected** custom endpoint so a stale or offline saved
endpoint does not block the picker. An explicit refresh flips to full probing
and busts the provider model cache:

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options?refresh=1"
```

Use `/v1/models` when an OpenAI-compatible client only needs a model name to
send back in chat/responses requests. Use `/api/model/options` when an
authenticated UI needs the richer Hermes-specific picker metadata.

### GET /v1/capabilities

Returns a machine-readable description of the API server's stable surface for external UIs, orchestrators, and plugin bridges.

```json
{
  "object": "hermes.api_server.capabilities",
  "platform": "hermes-agent",
  "model": "hermes-agent",
  "auth": {"type": "bearer", "required": true},
  "features": {
    "chat_completions": true,
    "responses_api": true,
    "run_submission": true,
    "run_status": true,
    "run_events_sse": true,
    "run_stop": true
  }
}
```

Use this endpoint when integrating dashboards, browser UIs, or control planes so they can discover whether the running Hermes version supports runs, streaming, cancellation, and session continuity without depending on private Python internals.

## Browser-extension control

Hermes can route browser tools through an authenticated extension that controls
the browser session associated with the current Hermes session. The feature is
disabled by default; set `browser.extension_control.enabled` to `true` to opt in:

```yaml
browser:
  extension_control:
    enabled: true
```

The local API path also requires the API server bearer key. A controller may
register only for an existing server session. Hermes derives the controller
principal from authenticated server state; a client-supplied `principal_id` is
ignored.

Discover the live contract through `GET /v1/capabilities`. The
`browser_extension_control` object reports whether the feature is enabled, the
protocol version, transport names, and the exact capability allowlist:

```text
controller.noop
browser_back
browser_click
browser_navigate
browser_press
browser_screenshot
browser_scroll
browser_snapshot
browser_tab_activate
browser_tabs
browser_type
```

Requested capabilities outside that list are filtered out. Raw CDP, arbitrary
script evaluation, console access, uploads, image extraction, and vision are not
part of the controller protocol.

When a request has no bound controller identity, or when the feature is disabled,
Hermes preserves the existing browser backend. Once the gateway binds a
controller principal and transport family to the request, that extension lane
is authoritative: missing, ambiguous, disconnected, or incapable controllers
fail closed instead of silently switching to a different local/cloud browser.
After an exact controller is selected, its result or error is authoritative and
Hermes never retries the same action through another backend.

### Local API registration

1. Send an authenticated `POST /v1/browser-control/register` with
   `protocol_version`, `session_id`, `controller_id`, `browser_profile_id`, and
   the requested `capabilities`.
2. Hermes returns a single-use ticket with a 30-second TTL and the filtered,
   server-bound controller scope.
3. Open `GET /v1/browser-control/ws` with both WebSocket subprotocols:
   `hermes-browser-control-v1` and
   `hermes-browser-control-ticket.<ticket>`.

The ticket is never accepted in the query string. Unknown, expired, reused, or
malformed tickets fail before WebSocket upgrade.

### Controller frames

Hermes sends `browser.controller.command` frames containing `command_id`,
`action`, immutable `arguments`, browser/controller ids, and the originating
`tool_call_id`. The controller replies with `browser.controller.result`, the
same `command_id`, an exact boolean `ok`, and either `result` or `error`.
Cancellation and timeout emit `browser.controller.cancel`; late results are
ignored.

An unexpected socket loss marks the controller offline and preserves work
already in flight until each command's original deadline. A reconnect with the
same principal, profile, session, controller id, browser profile, and transport
identity refreshes the transport without admitting new work before any deferred
cancels are flushed. Negotiated capabilities may change on that reconnect; they
are not an identity field. A different controller id or browser profile in the
same authenticated session lane is a hard replacement: old pending work is
cancelled before the successor becomes routable. Send
`browser.controller.detach` on the authenticated controller transport for an
intentional hard detach — that immediately cancels pending work. Merely closing
the socket is treated as a recoverable disconnect.

The authenticated dashboard transport exposes the same registration, result,
heartbeat, capability, and ownership semantics over its Gateway RPC/event
channel. In both transports, selection requires one unambiguous exact match on
principal, profile, session, controller, browser profile, transport family, and
capability. Once selected, a controller failure is authoritative and is never
retried through a different browser backend.

## Per-request model selection

Authenticated clients can override Hermes' default model selection per request
by sending:

- `model` — the target model id for this turn
- `provider` — the Hermes provider slug to resolve credentials/runtime for this turn
- `model_options` — request-scoped reasoning / service-tier controls

The same request fields are accepted on:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/runs`
- `POST /api/sessions/{session_id}/chat`
- `POST /api/sessions/{session_id}/chat/stream`

Precedence is deterministic:

1. Session `/model` override, if that session already has one
2. A static `gateway.platforms.api_server.model_routes` mapping selected when
   the request's `model` is a configured route alias
3. Direct request `model` / `provider` when no route alias matches
4. Global gateway config / environment defaults

`model_options` stays request-scoped regardless of which model/provider wins.
If a request sends a `provider` that conflicts with a configured `model_routes`
alias, Hermes rejects the request with `400` instead of silently remixing route
credentials with another provider.

**Bare `model` values on the OpenAI-compatible endpoints are opt-in.** Generic
OpenAI clients routinely hardcode model names (`gpt-4o`, ...), and existing
deployments rely on those falling back to the gateway default. On
`POST /v1/chat/completions` and `POST /v1/responses`, a `model` value sent
WITHOUT a `provider` is therefore ignored unless you enable:

```yaml
gateway:
  platforms:
    api_server:
      direct_model_requests: true
```

Requests that include an explicit `provider` — and the Hermes-native
`/v1/runs` and session-chat endpoints — always honor the requested model
regardless of this flag.

Example:

```json
{
  "model": "MiniMax-M3",
  "provider": "minimax",
  "model_options": {
    "reasoning_effort": "high",
    "service_tier": "priority"
  },
  "messages": [
    {"role": "user", "content": "Summarize the repo status."}
  ]
}
```

### GET /health

Health check. Returns `{"status": "ok"}`. Also available at **GET /v1/health** for OpenAI-compatible clients that expect the `/v1/` prefix.

### GET /health/detailed

Authenticated readiness check for monitoring and control planes. It reports
bounded status for the active profile's config, state database, configured
model, disk space, gateway/platform state, active API runs, pending process
completions, and active delegations. The response exposes status and counts,
not config values, credentials, paths, commands, queue payloads, or raw errors.

The public `/health` route remains a cheap liveness probe and does not run
readiness checks. A degraded readiness result still uses HTTP 200; inspect the
top-level `status` and `readiness.checks` fields.

## Runs API (streaming-friendly alternative)

In addition to `/v1/chat/completions` and `/v1/responses`, the server exposes a **runs** API for long-form sessions where the client wants to subscribe to progress events instead of managing streaming themselves.

### POST /v1/runs

Create a new agent run. Returns a `run_id` that can be used to subscribe to progress events.

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

Runs accept a simple `input` string and optional `session_id`, `instructions`, `conversation_history`, or `previous_response_id`. When `session_id` is provided, Hermes surfaces it in the run status so external UIs can correlate runs with their own conversation IDs.

For safely retryable creation, send an `Idempotency-Key` header (1–255 visible ASCII characters). Hermes durably reserves the key before starting work. An identical retry returns the original `run_id` with HTTP 202 and `Idempotency-Replayed: true`, including after a gateway restart and after the run has completed, failed, or been cancelled. Reusing the same key with a different JSON payload returns HTTP 409 with code `idempotency_key_conflict`. Keys are isolated by authenticated API profile/credential and retained for 24 hours after their last status update; clients should use unique, unguessable keys and must not reuse them for unrelated operations. Requests without the header retain the legacy behavior and always create a new run.

When `session_id` identifies an existing Hermes session and no explicit
`conversation_history` or `previous_response_id` is supplied, the run loads
that session's active transcript. Session turn leases serialize concurrent
writers and refresh the transcript after a contended wait.

### GET /v1/runs/\{run_id\}

Poll the current run state. This is useful for dashboards that need status without holding an SSE connection open, or for UIs that reconnect after navigation.

```json
{
  "object": "hermes.run",
  "run_id": "run_abc123",
  "status": "completed",
  "session_id": "space-session",
  "model": "hermes-agent",
  "output": "Done.",
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

Statuses are retained briefly after terminal states (`completed`, `failed`, or `cancelled`) for polling and UI reconciliation.

### GET /v1/runs/\{run_id\}/events

Server-Sent Events stream of the run's tool-call progress, token deltas, and lifecycle events. Designed for dashboards and thick clients that want to attach/detach without losing state.

When the agent delegates work to background subagents, the stream also carries
`subagent.start` and `subagent.complete` lifecycle events, so clients can
observe delegation outcomes — including timeouts and failures — instead of the
run going silent while a child works. The `subagent.complete` payload carries
the child's status, summary, duration, token/cost figures, a
`child_session_id` for correlation, and the `delegation_id` of the batch it
belongs to (so concurrent or nested fan-outs stay distinguishable); free-text fields pass forced secret
redaction before leaving the process. Per-tool child events
(`subagent.tool`, progress ticks) are intentionally **not** forwarded — they
are high-volume UI noise; use the per-child live transcript files for
play-by-play.

Unconsumed event buffers expire after five minutes so a detached client cannot
grow memory indefinitely. This expires transport state only: a run that is
still executing remains visible to status polling, approval, stop control, and
concurrency accounting until its executor work actually exits. A connected SSE
subscriber continues draining normally.

### POST /v1/runs/\{run_id\}/stop

Interrupt a running agent turn. The endpoint returns immediately with `{"status": "stopping"}` while Hermes asks the active agent to stop at the next safe interruption point.
The run stays tracked as `stopping` until the executor-backed work exits, then
settles as `cancelled`; requesting stop never hides a worker that is still
running.

### POST /v1/runs/\{run_id\}/approval

Resolve a pending approval for a run that is waiting on a human decision (for example, a tool call gated behind an approval policy). The body carries the approval decision; the run resumes once the decision is recorded. This endpoint is advertised in `/v1/capabilities` as the `run_approval` feature so external UIs can detect support before surfacing an approval prompt.

## Jobs API (background scheduled work)

The server exposes a lightweight jobs CRUD surface for managing scheduled / background agent runs from a remote client. All endpoints are gated behind the same bearer auth.

### GET /api/jobs

List all scheduled jobs.

### POST /api/jobs

Create a new scheduled job. Body accepts the same shape as `hermes cron` — prompt, schedule, skills, provider override, delivery target.

### GET /api/jobs/\{job_id\}

Fetch a single job's definition and last-run state.

### PATCH /api/jobs/\{job_id\}

Update fields on an existing job (prompt, schedule, etc.). Partial updates are merged.

### DELETE /api/jobs/\{job_id\}

Remove a job. Also cancels any in-flight run.

### POST /api/jobs/\{job_id\}/pause

Pause a job without deleting it. Next-scheduled-run timestamps are suspended until resumed.

### POST /api/jobs/\{job_id\}/resume

Resume a previously paused job.

### POST /api/jobs/\{job_id\}/run

Trigger the job to run immediately, out of schedule.

## Sessions API (session control over REST)

External UIs can manage Hermes sessions over REST without standing up the dashboard. All endpoints are gated by `API_SERVER_KEY` and live under `/api/sessions/*`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sessions` | List sessions (paginated — `limit`, `offset`, `source`, `include_children`) |
| `POST` | `/api/sessions` | Create an empty session |
| `GET` | `/api/sessions/{id}` | Read session metadata |
| `PATCH` | `/api/sessions/{id}` | Update title or `end_reason` |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `GET` | `/api/sessions/{id}/messages` | Message history for a session |
| `POST` | `/api/sessions/{id}/fork` | Branch the session via `SessionDB` lineage (matches CLI `/branch` semantics) |
| `POST` | `/api/sessions/{id}/chat` | Run one synchronous agent turn |
| `POST` | `/api/sessions/{id}/chat/stream` | SSE wrapper over a single turn — emits `assistant.delta`, `tool.started`, `tool.completed`, `run.completed` events |

`/v1/capabilities` advertises the full surface via `session_*` feature flags and `endpoints.session_*` entries so external UIs can detect support and fall back safely. Inline images are supported in `chat` and `chat/stream` payloads (multimodal-aware path).

Session list and detail responses include `cwd`, `git_repo_root`,
`project_id`, and `project_slug`. The project fields are `null` when the
session is not owned by an explicit Project.

```bash
# fork a session and run one turn
curl -X POST http://localhost:8642/api/sessions/$ID/fork \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"title": "explore alt path"}'

# stream a turn over SSE
curl -N -X POST http://localhost:8642/api/sessions/$ID/chat/stream \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"input": "what files changed in the last hour?"}'
```

## Projects API (read-only workspace discovery)

`GET /api/projects` exposes the active profile's Project catalog to external
UIs. The `data` array contains non-archived projects from `projects.db`, with
their folders and a `session_count`. `discovered_repos` contains cached
repo-first discovery rows with the same count field, and `active_id` identifies
the selected explicit Project when one is active.

Counts include the same active, non-hidden, client-visible conversations as
the Sessions API and are not limited by session-list pagination. This endpoint
does not create `projects.db` for a profile that has never used Projects; it
returns empty arrays instead.

```bash
curl http://localhost:8642/api/projects \
  -H "Authorization: Bearer $API_SERVER_KEY"
```

`/v1/capabilities` advertises this surface through
`features.project_resources` and `endpoints.projects`.

## Skills and toolsets discovery

`GET /v1/skills` and `GET /v1/toolsets` let external clients enumerate the agent's capabilities deterministically over REST instead of asking the model. Both are read-only and gated by `API_SERVER_KEY`.

```bash
curl http://localhost:8642/v1/skills \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "github-pr-workflow", "description": "...", "category": "..."}, ...]

curl http://localhost:8642/v1/toolsets \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "core", "label": "...", "description": "...", "enabled": true,
#     "configured": true, "tools": ["read_file", "write_file", ...]}, ...]
```

`/v1/skills` returns the same metadata the skills hub uses internally. `/v1/toolsets` returns toolsets resolved for the `api_server` platform with the concrete `tools` list each one expands to. Both are advertised under `endpoints.*` in `/v1/capabilities`.

## Long-term memory scoping (`X-Hermes-Session-Key`)

Multi-user frontends like Open WebUI need a stable per-channel identifier for long-term memory (Honcho, etc.) that is **independent** of the transcript-scoped `X-Hermes-Session-Id` (which rotates on `/new`). Pass `X-Hermes-Session-Key` on `/v1/chat/completions`, `/v1/responses`, or `/v1/runs` and Hermes threads it through to `AIAgent(gateway_session_key=...)`, where the Honcho memory provider uses it to derive a stable scope.

```http
POST /v1/chat/completions HTTP/1.1
Authorization: Bearer ***
X-Hermes-Session-Id: transcript-alpha
X-Hermes-Session-Key: agent:main:webui:dm:user-42
```

Rules: max 256 chars, control characters (`\r`, `\n`, `\x00`) are rejected, and the value is echoed back on responses (JSON + SSE). `/v1/capabilities` advertises support via `"session_key_header": "X-Hermes-Session-Key"`. Without the key, Honcho's `per-session` strategy produces a different scope per `session_id` — exactly the behavior Hermes had before.

## System Prompt Handling

When a frontend sends a `system` message (Chat Completions) or `instructions` field (Responses API), hermes-agent **layers it on top** of its core system prompt. Your agent keeps all its tools, memory, and skills — the frontend's system prompt adds extra instructions.

This means you can customize behavior per-frontend without losing capabilities:
- Open WebUI system prompt: "You are a Python expert. Always include type hints."
- The agent still has terminal, file tools, web search, memory, etc.

## Authentication

Bearer token auth via the `Authorization` header:

```
Authorization: Bearer ***
```

Configure the key via `API_SERVER_KEY` env var. If you need a browser to call Hermes directly, also set `API_SERVER_CORS_ORIGINS` to an explicit allowlist.

### Multi-profile routing (`/p/<profile>/…`)

When [multi-profile gateway routing](/user-guide/multi-profile-gateways) is
enabled (`gateway.multiplex_profiles`), the shared listener serves every
profile through a `/p/<profile>/` URL prefix — and **authentication is bound
to the routed profile**:

- Requests to `/p/<profile>/v1/...` must present that profile's own
  `API_SERVER_KEY` (from `~/.hermes/profiles/<profile>/.env`). The default
  listener's key is rejected on named-profile prefixes.
- Unprefixed routes and `/p/default/...` keep using the default profile's key.
- A named profile with no `API_SERVER_KEY` of its own fails closed — its
  prefix is unreachable until you set one.
- Runs are per-profile scoped: `/v1/runs/{run_id}` and its `events`, `stop`,
  `steer`, and `approval` routes only answer for the profile that created
  the run (including runs started via `/api/sessions/{id}/chat/stream`);
  another profile's run id returns `404`, never `403`.

:::warning Breaking change (July 2026)
Before this fix, a valid default-profile key was accepted on any
`/p/<profile>/` prefix. If you relied on one shared key across profile
prefixes, set a distinct `API_SERVER_KEY` in each profile's `.env` — reused
default keys on named prefixes now return `401`.
:::

:::warning Security
The API server gives full access to hermes-agent's toolset, **including terminal commands**. `API_SERVER_KEY` is **required for every deployment**, including the default loopback bind on `127.0.0.1`. Keep `API_SERVER_CORS_ORIGINS` narrow to control browser access when you explicitly allow browser callers.
:::

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | Enable the API server |
| `API_SERVER_PORT` | `8642` | HTTP server port |
| `API_SERVER_HOST` | `127.0.0.1` | Bind address (localhost only by default) |
| `API_SERVER_KEY` | _(required)_ | Bearer token for auth |
| `API_SERVER_CORS_ORIGINS` | _(none)_ | Comma-separated allowed browser origins |
| `API_SERVER_MODEL_NAME` | _(profile name)_ | Model name on `/v1/models`. Defaults to profile name, or `hermes-agent` for default profile. |

### config.yaml

The same settings can live in `~/.hermes/config.yaml` under a nested `gateway.api_server:` section:

```yaml
gateway:
  api_server:
    enabled: true
    port: 8642
    host: 127.0.0.1
    key: your-secret-key
    cors_origins: http://localhost:3000
    model_name: my-hermes
    max_concurrent_runs: 10   # concurrent-run cap; 0 disables the limit
```

`port`, `key`, `host`, `cors_origins`, and `model_name` are automatically bridged into the platform's `extra` settings, so they behave exactly like their `API_SERVER_*` environment-variable counterparts. Environment variables take precedence over `config.yaml` values. The block is also accepted under `gateway.platforms.api_server:` or a top-level `platforms.api_server:` section.

### Concurrent-run cap

The API server limits how many agent runs may execute at once across the OpenAI-compatible and Runs endpoints. The cap is read from `gateway.api_server.max_concurrent_runs` (default **10**; `0` disables the limit, negative values clamp to 0). When the cap is reached, new run-starting requests are rejected with **HTTP 429** `Too many concurrent runs (max N)` — clients should back off and retry.

## Security Headers

All responses include security headers:
- `X-Content-Type-Options: nosniff` — prevents MIME type sniffing
- `Referrer-Policy: no-referrer` — prevents referrer leakage

## CORS

The API server does **not** enable browser CORS by default.

For direct browser access, set an explicit allowlist:

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

When CORS is enabled:
- **Preflight responses** include `Access-Control-Max-Age: 600` (10 minute cache)
- **SSE streaming responses** include CORS headers so browser EventSource clients work correctly
- **`Idempotency-Key`** is an allowed request header — clients can send it for deduplication (responses are cached by key for 5 minutes)

Most documented frontends such as Open WebUI connect server-to-server and do not need CORS at all.

## Compatible Frontends

Any frontend that supports the OpenAI API format works. Tested/documented integrations:

| Frontend | Stars | Connection |
|----------|-------|------------|
| [Open WebUI](/user-guide/messaging/open-webui) | 126k | Full guide available |
| LobeChat | 73k | Custom provider endpoint |
| LibreChat | 34k | Custom endpoint in librechat.yaml |
| AnythingLLM | 56k | Generic OpenAI provider |
| NextChat | 87k | BASE_URL env var |
| ChatBox | 39k | API Host setting |
| Jan | 26k | Remote model config |
| HF Chat-UI | 8k | OPENAI_BASE_URL |
| big-AGI | 7k | Custom endpoint |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | Direct HTTP requests |

## Multi-User Setup with Profiles

To give multiple users their own isolated Hermes instance (separate config, memory, skills), use [profiles](/user-guide/profiles):

```bash
# Create a profile per user
hermes profile create alice
hermes profile create bob

# Configure each profile's API server on a different port. API_SERVER_* are env
# vars (not config.yaml keys), so write them to each profile's .env:
cat >> ~/.hermes/profiles/alice/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8643
API_SERVER_KEY=alice-secret
EOF

cat >> ~/.hermes/profiles/bob/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8644
API_SERVER_KEY=bob-secret
EOF

# Start each profile's gateway
hermes -p alice gateway &
hermes -p bob gateway &
```

Each profile's API server automatically advertises the profile name as the model ID:

- `http://localhost:8643/v1/models` → model `alice`
- `http://localhost:8644/v1/models` → model `bob`

In Open WebUI, add each as a separate connection. The model dropdown shows `alice` and `bob` as distinct models, each backed by a fully isolated Hermes instance. See the [Open WebUI guide](/user-guide/messaging/open-webui#multi-user-setup-with-profiles) for details.

## Limitations

- **Response storage** — stored responses (for `previous_response_id`) are persisted in SQLite and survive gateway restarts. Max 100 stored responses (LRU eviction).
- **No file upload** — inline images are supported on both `/v1/chat/completions` and `/v1/responses`, but uploaded files (`file`, `input_file`, `file_id`) and non-image document inputs are not supported through the API.
- **Simple OpenAI clients still see an alias** — `/v1/models` advertises the
  stable Hermes alias (`hermes-agent` or the active profile name). Richer
  clients can send explicit `provider` / `model_options` overrides on requests.

## Proxy Mode

The API server also serves as the backend for **gateway proxy mode**. When another Hermes gateway instance is configured with `GATEWAY_PROXY_URL` pointing at this API server, it forwards all messages here instead of running its own agent. This enables split deployments — for example, a Docker container handling Matrix E2EE that relays to a host-side agent.

See [Matrix Proxy Mode](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos) for the full setup guide.
