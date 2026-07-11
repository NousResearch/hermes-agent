---
title: Claude Max Kanban Runtime
sidebar_label: Claude Max Kanban Runtime
---

# Claude Max Kanban Runtime

Hermes can run a Kanban worker through the official Claude Agent SDK and the
Claude Code login attached to a Claude Max subscription. This is an opt-in
whole-agent runtime: Claude owns the worker loop while Hermes keeps ownership
of the board, session record, fallback chain, and worker lifecycle.

The current boundary is intentionally narrow: **Kanban workers on macOS only**.
Keep the board orchestrator on Hermes' normal runtime.

## Cost prerequisite

Before enabling this runtime, disable **Usage Credits / extra usage** on the
Anthropic account. The runtime requires `claude auth status` to attest all of
the following before a worker starts:

- logged in through `claude.ai`;
- first-party Anthropic service;
- `max` subscription type.

It also aborts when the SDK reports an overage route and records attested usage
as subscription-included rather than estimating an API cost.

:::warning No software-only zero-charge guarantee
The Claude CLI auth response proves that the process is using a Max login, but
it does not expose the account-level Usage Credits switch. Hermes cannot turn
that setting off or verify it locally. An SDK rate-limit/overage event is only
observable after a request has begun. Treat disabled Usage Credits as a hard
operator prerequisite, not something the runtime can enforce completely.
:::

Do not put `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or another Anthropic API
credential in the worker profile. The Claude subprocess receives an exact
allowlisted environment and will not receive those values, but removing them
also prevents a confusing native API route elsewhere in the profile.

Zero-extra operation also requires **every** fallback and Mixture-of-Agents
reference/aggregator route to be subscription-backed or local. A Max-backed
acting worker does not make a native Anthropic, OpenAI, OpenRouter, or other API
advisor free; those routes keep their own billing semantics.

Automatic auxiliary tasks do not reinterpret a Claude SDK main route as native
Anthropic and do not search ambient credentials or main fallback routes. Title
generation, compression, and similar `provider: auto` tasks skip gracefully.
To enable one, configure `auxiliary.<task>.provider` and `.model` explicitly;
that route has its own billing semantics and may cost extra.

## Prerequisites

1. macOS, because the worker command boundary uses Seatbelt (`sandbox-exec`);
2. Claude Code signed into the intended Max account:

   ```bash
   claude auth login
   claude auth status
   ```

3. Usage Credits / extra usage disabled in the Anthropic account UI;
4. a dedicated Hermes worker profile and an initialized Kanban board.

The exact pinned Agent SDK dependency is installed lazily on first use. To
preinstall it for an immutable deployment, install the optional extra:

```bash
pip install 'hermes-agent[claude-agent-sdk]'
```

## Worker profile configuration

Create a dedicated profile rather than changing the orchestrator profile:

```bash
hermes profile create claude-worker --clone
```

In that profile's `config.yaml`, select the provider-neutral runtime:

```yaml
model:
  default: claude-opus-4-6
  provider: anthropic
  runtime: claude_agent_sdk

# Optional, but recommended: another subscription/local runtime to use when
# Max is capped or temporarily unavailable. Each entry owns its own runtime.
fallback_providers:
  - provider: openai-codex
    model: gpt-5.4
    runtime: codex_app_server
```

Do not use a native `provider: anthropic` fallback without
`runtime: claude_agent_sdk` if the goal is subscription-only operation. Native
Anthropic is an API billing route.

Assign ready Kanban tasks to `claude-worker`. The gateway-embedded dispatcher
spawns `hermes -p claude-worker chat -q ...` with the task and workspace
identities pinned in its environment; direct chat sessions do not satisfy this
boundary and fail closed.

## Worker capabilities and isolation

The SDK worker receives a small in-process Hermes MCP surface:

- Kanban status, heartbeat, comment, complete, block, create, link, and related
  worker-safe board operations available to the assigned task;
- `read_file` and `write_file`, confined to regular files inside the worker
  workspace with symlink and linked-file rejection;
- `terminal`, run with an exact environment, workspace-only writes, network
  disabled, and a stable default-deny Seatbelt profile that reads only the
  workspace, read-only Git common directory, and explicit system/toolchain
  roots;
- `process`, limited to processes owned by the current worker task.

When the same Claude runtime performs a background memory/skill review, it gets
an independent capability mode containing only the requested `memory` and/or
`skill_manage` tool. Curator mode gets only `skill_manage`. Kanban, terminal,
process, and file capabilities are absent from those auxiliary SDK sessions;
missing required tool definitions fail closed.

Claude's built-in Bash, web tools, plugins, skills, and user setting sources are
disabled for this runtime. The worker receives the real host `HOME` only so the
Claude login can resolve from secure OS storage; arbitrary Hermes/provider
credentials are not inherited.

## Fallback behavior

Runtime identity belongs to each route, not to the provider name. This lets one
turn move between `claude_agent_sdk`, `codex_app_server`, a local model, or the
normal Hermes loop without starting a second Hermes process.

When a Max request is rejected for auth, quota, rate limit, overload, billing,
or a server failure, Hermes advances the configured fallback chain in the same
worker process. The circuit is persisted by profile, attested account, runtime,
provider, and model, so newly created agents respect an active reset window.

Hermes replays the original user turn only while it is safe. An unresolved tool
call fails closed. If Claude completed and recorded a tool action before the
failure, the next external runtime instead receives a continuation instruction
to inspect durable workspace, board, and process state and not repeat completed
actions.

## Observability

Hermes writes structured runtime events to a dedicated logger named
`hermes.runtime_events`. Every event is logged as a single-line JSON object at
the `INFO` level. Configure any standard Python logging handler to capture this
logger (file sink, syslog, structured backend) independently of the main
`hermes` logger.

### Payload structure

Every event includes the following base fields:

| Field | Type | Description |
|-------|------|-------------|
| `event` | string | Event name (see below). |
| `ts` | float | Unix timestamp at emission time. |
| `provider` | string | Active provider at the time of the event. |
| `model` | string | Active model at the time of the event. |
| `runtime` | string | Active runtime identifier (e.g. `claude_agent_sdk`, `hermes`). |

Additional fields are appended per event type as documented below.

### Events

#### `runtime_attempt_start`

Emitted immediately before each external-runtime attempt begins. No extra
fields beyond the base payload. Use this event to correlate the start of an
attempt with its eventual outcome.

```json
{"event": "runtime_attempt_start", "ts": 1783000000.0, "provider": "anthropic", "model": "claude-opus-4-6", "runtime": "claude_agent_sdk"}
```

#### `runtime_attempt_failure`

Emitted when a runtime attempt ends in a classified failure. Extra fields:

| Field | Type | Description |
|-------|------|-------------|
| `reason` | string | Failover reason — one of `auth`, `auth_permanent`, `rate_limit`, `billing`, `overloaded`, `server_error`, `timeout`, `network`, or `unknown`. |
| `replay_safe` | bool | Whether the original user turn can be replayed into the next runtime. `false` means a tool side effect was recorded and fallback was blocked to avoid repeating it. |

```json
{"event": "runtime_attempt_failure", "ts": 1783000001.2, "provider": "anthropic", "model": "claude-opus-4-6", "runtime": "claude_agent_sdk", "reason": "rate_limit", "replay_safe": true}
```

#### `runtime_circuit_open`

Emitted when Hermes opens a subscription circuit after a quota, billing, auth,
overload, or server-error failure. The circuit is persisted to
`~/.hermes/state/runtime-circuits.json` (profile-aware) and is checked before
each subsequent attempt — a newly created agent on the same profile skips the
circuit-protected runtime until the reset window expires. Extra fields:

| Field | Type | Description |
|-------|------|-------------|
| `reset_at` | int | Unix timestamp when the circuit resets and the runtime is eligible for retry. |

```json
{"event": "runtime_circuit_open", "ts": 1783000001.3, "provider": "anthropic", "model": "claude-opus-4-6", "runtime": "claude_agent_sdk", "reset_at": 1783003600}
```

#### `runtime_fallback_activated`

Emitted when Hermes successfully advances to a fallback route after a failure.
Records both the outgoing and incoming route for end-to-end tracing. Extra fields:

| Field | Type | Description |
|-------|------|-------------|
| `reason` | string | Failover reason that triggered the fallback (same vocabulary as `runtime_attempt_failure`). |
| `from_provider` | string | Provider that failed. |
| `from_model` | string | Model that failed. |
| `from_runtime` | string | Runtime that failed. |
| `to_provider` | string | Provider of the activated fallback route. |
| `to_model` | string | Model of the activated fallback route. |
| `to_runtime` | string | Runtime of the activated fallback route. |

```json
{"event": "runtime_fallback_activated", "ts": 1783000001.4, "provider": "openai", "model": "gpt-5.4", "runtime": "hermes", "reason": "rate_limit", "from_provider": "anthropic", "from_model": "claude-opus-4-6", "from_runtime": "claude_agent_sdk", "to_provider": "openai", "to_model": "gpt-5.4", "to_runtime": "hermes"}
```

#### `runtime_billing_mode`

Emitted after each Claude Agent SDK attempt, regardless of success or failure,
when usage metadata is available. Indicates whether the usage was attested as
subscription-included. Extra fields:

| Field | Type | Description |
|-------|------|-------------|
| `billing_mode` | string | `subscription_included` if a valid Max attestation was present; `unattested` otherwise. |
| `cost_status` | string | `included` (subscription-included) or `unknown` (unattested). |

```json
{"event": "runtime_billing_mode", "ts": 1783000002.1, "provider": "anthropic", "model": "claude-opus-4-6", "runtime": "claude_agent_sdk", "billing_mode": "subscription_included", "cost_status": "included"}
```

### Capturing the event stream

Route the logger to a file with standard Python logging config, or to any
structured backend by replacing the handler:

```python
import logging

handler = logging.FileHandler("runtime-events.jsonl")
handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("hermes.runtime_events").addHandler(handler)
```

To grep the events from the main agent log in real time:

```bash
tail -f ~/.hermes/profiles/<profile>/logs/agent.log | grep '"event".*runtime_'
```

## Supported surfaces

The provider-neutral route is propagated consistently by the CLI, messaging
gateway (including Telegram), TUI/Desktop gateway, ACP, and cron constructors.
The Claude SDK execution boundary itself still requires a dispatcher-created
Kanban task on macOS.

That means Desktop or Telegram can create, monitor, and receive results from a
Claude Max worker through the normal Kanban flow. They are not alternate ways
to start a direct Claude SDK chat. ACP and cron also fail closed if configured
for this runtime outside a Kanban worker instead of silently becoming native
Anthropic.

## Verification

Safe preflight, with no generative model call:

```bash
claude auth status
```

This checks the system Claude CLI and is a useful operator preflight, but it is
not the final runtime attestation. Hermes runs the SDK's pinned bundled Claude
CLI through an exact-environment wrapper and executes `auth status` against that
same wrapped binary before the worker starts. Confirm the system check reports
a first-party `claude.ai` Max login, then separately verify in the Anthropic
account UI that Usage Credits / extra usage is disabled.

The repository's automated tests verify route propagation, exact-environment
credential isolation, Max attestation parsing, workspace/process boundaries,
fallback replay rules, circuit persistence, message projection, and token
accounting without consuming Claude tokens. A true end-to-end Kanban smoke test
necessarily consumes Max-plan usage and cannot prove the account-level Usage
Credits switch; run it only after the manual account check.

## Current limitations

- macOS only;
- Kanban workers only; the orchestrator remains on Hermes' native runtime;
- no direct Desktop, TUI, ACP, Telegram, or cron Claude SDK chat;
- no Claude built-in Bash, network/web tools, plugins, or user settings;
- Usage Credits state cannot be attested through the CLI;
- the terminal rejects pre-existing hard-linked regular files before each
  sandbox start. Default-deny path rules also stop the sandboxed worker from
  creating an external hard link or following a late symlink outside the
  workspace. macOS Seatbelt is not a separate-user or VM boundary, however: a
  hostile same-UID host process could race a new hard link into a long-running
  sandbox after preflight. Use a dedicated OS account, copied private
  workspace, container, or VM when hostile sibling host processes are in the
  threat model. Workspaces whose dependency manager created hard-linked files
  must be recreated in copy mode first (for uv, set `UV_LINK_MODE=copy`).

For board setup and worker assignment, see [Kanban](./kanban) and
[Kanban worker lanes](./kanban-worker-lanes). For the fallback-chain format,
see [Fallback Providers](./fallback-providers).
