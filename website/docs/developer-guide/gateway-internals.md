---
sidebar_position: 7
title: "Gateway Internals"
description: "How the messaging gateway boots, authorizes users, routes sessions, and delivers messages"
---

# Gateway Internals

The messaging gateway is the long-running process that connects Hermes to 20+ external messaging platforms through a unified architecture.

## Canonical group metadata and profile discovery

The authenticated canonical WebSocket exposes `groups.capabilities`, `groups.list`,
`groups.state`, and `groups.log` with `session:read`; `groups.create`,
`groups.rename`, and `groups.disband` require `session:control`. These reuse the
hosted-room metadata protocol in the owning authority's database. An explicit
`profile` selector must resolve to that authority's home. Disband refuses rooms
with unsettled driver work rather than bypassing Stop.

This is **metadata management, not hosted-room inference**. The existing local
hosted-room driver still uses legacy TUI session handlers; the canonical surface
therefore reports `driver: false` and does not advertise send, retry, approval,
or peer-grant operations. It never imports that driver to execute a group turn.

`profiles.list` requires `session:read` and returns only the authenticated
profile, including its display metadata and avatar presence. `include_sessions`
defaults to true: previews are limited to that actor's canonical local bindings,
and `canonical_session` resolves the exact hidden `Bot Chat` title, not the most
recent conversation. Discovery does not create, restore, or unarchive sessions,
open sibling profile databases, or run inference. Cross-profile discovery requires
a separately authenticated connection to the relevant owner.

## Canonical ancillary reads

Desktop and Ink use authenticated `session.control.read`, `process.list`,
`subagent.list`, and `subagent.tail` with `session_id`. They require
`session:read` and the owning profile/session identity. These polls never restore
a cold session: use `session.resume` first. A retired route binding returns
`stale_generation` rather than exposing its replacement conversation.

Control reads project the physical transcript's persisted goal, loop and heartbeat
through the existing frontend field allowlists, without constructing managers or
clearing wait barriers. Revisions hash the visible snapshot; timed wait barriers
can disappear when their deadline passes without changing persisted state.
Process reads use already-owned registry objects and include a bounded output tail;
they do not recover processes, consume notifications or scan retained receipts.
Subagent reads require the current parent's live object ancestry, not merely equal
session IDs; tails are bounded to 16 KiB. Missing or retired children return
`available: false`. These are reads, not child steering or stop controls.

Managed-worker policies return `unsupported_projection` for process/subagent reads:
the owner's process-local registry cannot truthfully describe another interpreter.
Control mutations (`session.control`), process mutations and subagent steering are
not enabled by these read projections.

## Shared authority pending-input snapshots

`session.resume` returns each pending admission with its original `input_id` and
committed public `text`, alongside the canonical `admission_id`, sequence and status.
This includes native messaging inputs and local-client inputs. Snapshots do not expose
private native routing, source, or authorization envelopes. Submission and receipt
ACKs retain their existing shape. Clients must use `admission_id` for cancellation
and receipt queries rather than assuming it equals the original input ID.
Queue admission, claim, cancellation and settlement also publish `session.info` to
attached viewers. Its `pending` rows use the same projection as resume. Settlement
publishes this state before `message.complete`, retaining the terminal event as the
last event of that turn. Desktop translates both paths into its existing
`pending_submissions` queue store.

A turn that was `started` when its owning gateway died is recovered by the next owner
as `unknown`: Hermes cannot prove which side effects happened, so it neither replays
the input nor lets the queued turns behind it run. `prompt.resolve_unknown`
(`session:control`, params `session_id`, `admission_id`, `execution_generation`)
acknowledges the loss: the row settles as `interrupted`, the session FIFO resumes and
the follower runs exactly once. The generation must be the one stamped on the unknown
row (visible in the resume/`session.info` projection), so a stale or already-resolved
row is refused with `stale_generation`. Desktop shows such a row in the queue panel as
"Turn lost during restart" with a **Discard** button; the gateway CLI exposes it as
`/discard <admission_id>`. The lost input stays in the transcript for the user to resend.

## Shared authority prompt attachments

`prompt.submit` accepts an optional `attachments: [{path, mime}]` list beside `text`.
The client stages image bytes in the profile image cache (`cache/images/` under the
served `HERMES_HOME`, the same place messaging adapters stage downloads); paths must be
regular files directly in that directory, `mime` must be an image type
(`image/png`, `image/jpeg`, `image/gif`, `image/webp`), at most 10 per submission, each
under `gateway.max_inbound_media_bytes`. The authority copies the bytes into its
immutable `native-inputs` store at admission and stores the sha256-checked reference in
the private payload, so later mutation or cache cleanup of the staging file cannot
change what executes, including after an owner restart. Execution restores them onto
`MessageEvent.media_urls`/`media_types`, and the ordinary image routing
(`agent.image_input_mode`, native `image_url` parts vs. text pre-analysis) applies.
Text-only submissions keep the `{text}` payload and fingerprint. The ACP transport
agent uses this for image, image resource-link and embedded-image prompt blocks;
Ink and Desktop still attach through the legacy `image.attach`/`file.attach` RPCs.

## Native admission startup recovery

Ordinary gateway startup recovers accepted native inputs after adapter connection,
route publication, and the startup restore gate. Only never-started queued work is
eligible. Recovery uses the current routing index and connected adapter, rechecks
sender authorization and the retained connector/profile binding, and leaves rows
queued when a route, transport, or authorization is unavailable. Restoring that
configuration permits recovery on a later startup.

An interrupted started admission becomes `unknown` and pauses its session's FIFO;
its followers do not automatically run. Legacy synthetic restart turns are excluded
from sessions with canonical admission history, so they cannot race the durable
queue or bypass an unknown pause. Completed inputs are not reinferred on restart.
Under `gateway.multiplex_profiles` the bootstrap builds one session authority per
served profile home (the default plus every directory under `profiles/`), each bound
to that home's `state.db` and entered under that profile's runtime scope. Every
admission path (bot message, `/p/<profile>/` API and webhook, cron fire, kanban
dispatch, native CLI/Desktop/ACP attach) resolves its authority from the routed
profile's home, so recovery, FIFO order and exactly-once settlement hold per profile
and a secondary's rows never land in the default's ledger. A served secondary's
clients discover the multiplexer through the default home's descriptor
(`served_profiles`) and attach to its control socket with their own `profile_id`.

## Shared authority local slash commands

Authenticated local sessions accept `slash.exec({session_id, command})` and
`command.dispatch({session_id, name, arg})`. Both use the existing command registry
and gateway handlers, without a legacy slash worker or second execution runtime.
An optional `profile` selector must match the owning daemon's profile. Caller source
and routing overrides are rejected; another authenticated actor cannot operate the
session.

Supported read commands are `/help`, `/commands`, `/status`, `/context`, `/version`,
and `/whoami` (`session:read`). `/title` requires `session:control`, even for a query,
and retains the existing busy rejection. Registry aliases work. Handler results use
`{type: "exec", output}`.

Skill names and configured quick-command aliases to supported commands or skills
are resolved in the owner's profile. Skill resolution requires `session:submit` and
returns `{type: "skill", name, message, display}`. Desktop/Ink send the expanded
message through their normal identified `prompt.submit`; resolution is not an
admission ACK. The authority's durable FIFO owns retries and execution, and skill
content remains a user message without changing the cached system prefix.

Other commands return `unsupported_command`: shell quick commands, plugin execution,
bundles, configuration/runtime changes, lifecycle commands, and approval/secret
slash shortcuts are not exposed. Existing generation-bound approval and interrupt
RPCs remain separate. Catalog discovery is broader than this reviewed execution set.

## Shared authority busy input and corrections

`config.get({session_id, key: "busy"})` projects the session's frozen busy-input
preference. `config.set({session_id, key: "busy", value})` accepts `interrupt`,
`steer`, or `queue` and changes only that live session's preference, shared by its
viewers until owner restart. Neither operation changes profile settings or the
cached agent/system prompt. Other config keys are not handled by this adapter.

`session.steer` and `session.redirect` require `session_id`, `text`, and the current
`execution_generation`, plus authenticated session-control rights. Optional
`profile` must match the authority. Controls target the existing running agent:
steer uses the ordinary correction drain; redirect cancels an active model request
and continues the same turn (or steers at a tool boundary). They never create a
queued admission or silently fall back to a new turn. `queued` (steer) and
`redirected` acknowledge runtime acceptance, not crash-durable delivery; do not
automatically retry an ambiguous reply. `rejected` means the runtime missed the
active correction window, while `stale_generation` rejects settled/replaced work.
Managed-worker corrections currently return `unsupported_control`; normal durable
queue submission and Stop remain separate controls.

## Shared authority setup readiness

Authenticated local clients can call `setup.status` and `setup.runtime_check` before
creating a session. Both require `session:create` and run against the daemon's owned
profile; a selector for another profile is rejected. Runtime readiness reuses the
existing provider resolver off the event loop and does not create a legacy session
runtime. Missing provider credentials remain a failed readiness result, not a bypass
of onboarding.

## Shared authority clarification protocol

When the HTTP/WebSocket surface is bound to a `SessionAuthority`, attached authorized
viewers receive `clarify.request` events and pending questions in the `session.resume`
`prompts` snapshot. This is an authority API contract; it does not imply that every
native frontend already renders the shared form or that ordinary bootstrap is complete.

A response uses `clarify.respond` with `session_id`, `execution_generation`, `prompt_id`,
and an `answer` string. The connected viewer must be subscribed and have
`session:respond`. The authority checks the running generation and resolves the existing
native clarify waiter by exact prompt ID, without adding a user prompt to the admission
queue. The first response wins; detach does not cancel it. Native answers and timeouts
retire the same prompt and emit `clarify.settled`; response text is not placed in the
control replay stream, although the normal clarify tool result becomes part of the
conversation. Questions and choice labels use the existing display redactor.

Publication follows successful native card delivery. A failed native delivery does not
create an actionable shared form. Ordinary approvals use the separate `approval.respond`
permission path; sudo/secret prompts and durable control restoration after daemon restart
are not implemented by this clarification path.

## Revision-checked session metadata

The authority-bound WebSocket accepts `session.mutate` for a registered session.
Parameters are `session_id`, `request_id`, `expected_revision`, `operation`, and
`payload`. `rename` accepts only `{"title": "New title"}`; `archive` accepts only
`{"archived": true}` or `{"archived": false}`. The caller needs `session:control`.
The session handle in `session.resume` supplies the current revision.

The edit and its retry receipt commit together. Repeating the same request returns
the original result without incrementing the revision again; reusing its ID with
different contents returns `admission_conflict`. A competing edit with a stale
revision returns `revision_conflict`. Draining rejects new mutations and fresh
session registration before routing or transcript creation. These edits
do not interrupt a running turn. This RPC does not yet migrate legacy direct
writers, expose arbitrary SQL, or implement reset, delete, or rewind.

## Private native HTTP authentication

The authoritative daemon accepts a **fresh, one-use HTTP grant per request** from
its same-user private control socket. Send `session-ticket` with exactly
`profile_id` (the canonical served primary home), `instance_id` (the ready daemon's
boot ID), and `purpose: "native-http"`. The returned ticket expires after 30 seconds
and carries only `http:owner`; send it in `X-Hermes-Gateway-Ticket` from the native
main process. Never place it in a URL, cookie, public descriptor, renderer config,
or log, and never obtain a dashboard token by scraping the public page.

The existing HTTP authentication seam redeems the grant only for the matching
ready daemon and primary profile. It requires an actual loopback socket peer,
captured before trusted-proxy rewriting, and **no Origin header at all** (even an
empty Origin is refused). An invalid, duplicate, expired, replayed, or wrong-purpose
header returns 401 without falling back to a supplied static token or OAuth cookie.
Draining/unready listeners refuse admission; restart and shutdown revoke grants.
HTTP grants do not authenticate WebSockets, and interactive/exposure/worker grants
do not authenticate HTTP. Requests without this header keep the existing static
session-token, OAuth bearer/cookie, Host, and route-specific token-provider gates.

This local bootstrap is an explicit same-OS-user administrative boundary, not an
OAuth bypass for remote clients. An exposure proxy must not replace remote callers'
credentials with its own native grant. It is not a filesystem sandbox against code
already running as the same OS user.

A native grant does not retarget the daemon through a URL or JSON `profile` selector:
each supplied selector must resolve to the served primary home. The existing
config `current`/empty/default-name contracts are preserved rather than treating
`default` as an alias for any arbitrary launch home. Cross-profile targets return
403 `profile_scope_mismatch`, including conflicting duplicate query selectors.
`GET /api/profiles` and `GET /api/profiles/active` retain their discovery metadata
contracts; discovery does not confer another profile's data authority. Aggregate
session/sidebar routes require an explicit own-profile selector, and cron job routes
require an explicit own-profile query. All-profile project/transcript aggregation,
profile creation/import/activation, and profile rename/delete are not granted by
this primary-profile credential. Native clients must not rewrite `all` to `current`
or borrow the primary grant to hide a cross-profile failure.

The native HTTP integration is covered by real ordinary-daemon/control-socket/TCP
probes in `tests/gateway/test_native_http_auth.py`, alongside gated OAuth/cookie and
actual non-loopback-peer controls. This does not establish native Desktop UI parity
or platform-specific Windows/macOS bootstrap validation.

## Frozen local launch credentials

Local `session.create` receipts retain the launch policy, not plaintext API keys.
Inline credentials in the profile config are borrowed from that existing private
source after restart: the canonical profile home, credential path, value fingerprint,
and frozen policy identity must match. Provider endpoints and request settings still
come from the frozen policy, not the current config. Missing or changed credentials
fail closed rather than selecting a different key. No second secret store is created.
Credential-bearing headers, environment maps, and terminal projections are redacted
from durable policy JSON and hydrated privately through the same source checks.

Retrying a creation request returns its original session and policy without binding
current replacement credentials. An ad-hoc `--api-key` remains authority-lifetime-only;
restart revokes that key, not config-based sessions. Older receipts that contain only
an instance-bound credential reference cannot securely recover a lost value and remain
fail-closed. Their history is still readable; create a new session to adopt current
credentials. Credential fingerprints are private metadata, never public API responses.

CLI cost/data-training consent runs before gateway discovery and session creation,
including top-level one-shot launches. It does not become a process-wide option.

## Key Files

| File | Purpose |
|------|---------|
| `gateway/run.py` | `GatewayRunner` facade — composes the `gateway/run_*.py` sibling mixins (startup, adapters, inbound, turn, busy, goals, notifications, shutdown, …) and `gateway/slash_commands_*.py` handlers |
| `gateway/session.py` | `SessionStore` — conversation persistence and session key construction |
| `gateway/delivery.py` | Outbound message delivery to target platforms/channels |
| `gateway/pairing.py` | DM pairing flow for user authorization |
| `gateway/channel_directory.py` | Maps chat IDs to human-readable names for cron delivery |
| `gateway/hooks.py` | Hook discovery, loading, and lifecycle event dispatch |
| `gateway/mirror.py` | Cross-session message mirroring for `send_message` |
| `gateway/status.py` | Token lock management for profile-scoped gateway instances |
| `gateway/builtin_hooks/` | Extension point for always-registered hooks (none shipped) |
| `gateway/platform_registry.py` | Adapter registry, factories, and deferred (lazy) loaders for bundled platform plugins |
| `plugins/platforms/<name>/` | Bundled messaging adapters (most platforms: `adapter.py` + `plugin.yaml`) |
| `gateway/platforms/` | Shared `base.py` plus legacy/direct adapters (Signal, API server, webhooks, …) |

## Optional service installation

A running gateway and an installed OS service are different things. `hermes gateway run`
works without a service, including with no messaging platforms configured.

`hermes setup`, quick setup, and `hermes gateway setup` ask interactive users once:

> Install the gateway service so Hermes starts automatically at login and keeps scheduled jobs and messaging available?

The profile's `gateway.service_install_choice` stores `null`, `install`, or `decline`.
An unset value is not permission to install. Declining records `decline` without creating,
enabling, or removing a service. In the gateway wizard, choosing **Start now** but declining
service installation launches an unmanaged gateway; it does not create a disabled service.
Without service persistence, messaging and scheduled work stop at logout/reboot. Jobs cannot
run while the host is off.

Noninteractive setup and imports do not install a missing service, even when the imported
config says `install`. Existing installed services remain authoritative and can be started
regardless of the saved preference. To opt in explicitly later, run `hermes gateway install`;
the choice is recorded only after successful installation. To be asked again during setup,
reset the choice with `hermes config set gateway.service_install_choice null`.

Setup orchestration lives in `hermes_cli/gateway_setup_service.py`. Its
`ensure_gateway_service` helper is not a general runtime discovery or auto-start API:
ordinary calls never reinstall a missing service based only on stored preference.

## Architecture Overview

```text
┌─────────────────────────────────────────────────┐
│                  GatewayRunner                  │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Telegram │  │ Discord  │  │  Slack   │       │
│  │ Adapter  │  │ Adapter  │  │ Adapter  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │
│       └─────────────┼─────────────┘             │
│                     ▼                           │
│              _handle_message()                  │
│                     │                           │
│         ┌───────────┼───────────┐               │
│         ▼           ▼           ▼               │
│  Slash command   AIAgent    Queue/BG            │
│    dispatch      creation   sessions            │
│                     │                           │
│                     ▼                           │
│                 SessionStore                    │
│              (SQLite persistence)               │
└───────┴─────────────┴─────────────┴─────────────┘
```

## Message Flow

When a message arrives from any platform:

1. **Platform adapter** receives raw event, normalizes it into a `MessageEvent`
2. **Base adapter** checks active session guard:
   - If agent is running for this session → queue message, set interrupt event
   - If `/approve`, `/deny`, `/stop` → bypass guard (dispatched inline)
3. **GatewayRunner._handle_message()** receives the event:
   - Resolve session key via `_session_key_for_source()` (format: `agent:main:{platform}:{chat_type}:{chat_id}`)
   - Check authorization (see Authorization below)
   - Check if it's a slash command → dispatch to command handler
   - Check if agent is already running → intercept commands like `/stop`, `/status`
   - Otherwise → create `AIAgent` instance and run conversation
4. **Response** is sent back through the platform adapter

### Session Key Format

Session keys encode the full routing context:

```
agent:main:{platform}:{chat_type}:{chat_id}
```

For example: `agent:main:telegram:private:123456789`

Thread-aware platforms (Telegram forum topics, Discord threads, Slack threads) may include thread IDs in the chat_id portion. **Never construct session keys manually** — always use `build_session_key()` from `gateway/session.py`.

### Two-Level Message Guard

When an agent is actively running, incoming messages pass through two sequential guards:

1. **Level 1 — Base adapter** (`gateway/platforms/base.py`): Checks `_active_sessions`. If the session is active, queues the message in `_pending_messages` and sets an interrupt event. This catches messages *before* they reach the gateway runner.

2. **Level 2 — Gateway runner** (`gateway/run_inbound.py`): Checks `_running_agents`. Intercepts specific commands (`/stop`, `/new`, `/queue`, `/status`, `/approve`, `/deny`) and routes them appropriately. Everything else triggers `running_agent.interrupt()`.

Commands that must reach the runner while the agent is blocked (like `/approve`) are dispatched **inline** via `await self._message_handler(event)` — they bypass the background task system to avoid race conditions.

## Authorization

The gateway uses a multi-layer authorization check, evaluated in order:

1. **Per-platform allow-all flag** (e.g., `TELEGRAM_ALLOW_ALL_USERS`) — if set, all users on that platform are authorized
2. **Platform allowlist** (e.g., `TELEGRAM_ALLOWED_USERS`) — comma-separated user IDs
3. **DM pairing** — authenticated users can pair new users via a pairing code
4. **Global allow-all** (`GATEWAY_ALLOW_ALL_USERS`) — if set, all users across all platforms are authorized
5. **Default: deny** — unauthorized users are rejected

### DM Pairing Flow

```text
Admin: /pair
Gateway: "Pairing code: ABC123. Share with the user."
New user: ABC123
Gateway: "Paired! You're now authorized."
```

Pairing state is persisted in `gateway/pairing.py` and survives restarts.

## Slash Command Dispatch

All slash commands in the gateway flow through the same resolution pipeline:

1. `resolve_command()` from `hermes_cli/commands.py` maps input to canonical name (handles aliases, prefix matching)
2. The canonical name is checked against `GATEWAY_KNOWN_COMMANDS`
3. `_handle_message()` (`gateway/run_inbound.py`) looks the handler up by name — `_handle_<name>_command` on the `gateway/slash_commands_*.py` mixins — via `_command_handler_table` over `_IDLE_COMMANDS` / `_PLAIN_COMMANDS` in `gateway/run_busy.py`; there is no `if canonical == ...` chain
4. Some commands are gated on config (`gateway_config_gate` on `CommandDef`)

### Running-Agent Guard

Commands that must NOT execute while the agent is processing are rejected early:

While `_quick_key in self._running_agents`, `_dispatch_busy_slash_command()` in `gateway/run_busy.py` routes each recognized command by its `CommandDef.busy_policy` / `busy_handler`: a mid-run variant (`_busy_<key>_command`) if one exists, otherwise the normal handler when `busy_policy` allows it, otherwise a reject message ("⏳ Agent is running — `/model` can't run mid-turn…").

Bypass commands (`/stop`, `/new`, `/approve`, `/deny`, `/queue`, `/status`) have mid-run handlers and are dispatched inline.

## Config Sources

The gateway reads configuration from multiple sources:

| Source | What it provides |
|--------|-----------------|
| `~/.hermes/.env` | API keys, bot tokens, platform credentials |
| `~/.hermes/config.yaml` | Model settings, tool configuration, display options |
| Environment variables | Override any of the above |

Unlike the CLI (which uses `load_cli_config()` with hardcoded defaults), the gateway reads `config.yaml` directly via YAML loader. This means config keys that exist in the CLI's defaults dict but not in the user's config file may behave differently between CLI and gateway.

## Platform Adapters

Most messaging platforms ship as plugin adapters under `plugins/platforms/<name>/adapter.py`; a few legacy adapters still live directly in `gateway/platforms/`. All extend `BasePlatformAdapter` from `gateway/platforms/base.py`:

```text
plugins/platforms/                  # plugin-packaged adapters (one dir each)
├── telegram/adapter.py     # Telegram Bot API (long polling or webhook)
├── discord/adapter.py      # Discord bot via discord.py
├── slack/adapter.py        # Slack Socket Mode
├── whatsapp/adapter.py     # WhatsApp Business Cloud API
├── matrix/adapter.py       # Matrix via mautrix (optional E2EE)
├── mattermost/adapter.py   # Mattermost WebSocket API
├── email/adapter.py        # Email via IMAP/SMTP
├── sms/adapter.py          # SMS via Twilio
├── dingtalk/adapter.py     # DingTalk WebSocket
├── feishu/adapter.py       # Feishu/Lark WebSocket or webhook
├── wecom/adapter.py        # WeCom (WeChat Work) callback
├── line/adapter.py         # LINE Messaging API
├── teams/adapter.py        # Microsoft Teams
├── irc/adapter.py          # IRC (canonical scoped-lock example)
├── homeassistant/adapter.py # Home Assistant conversation integration
└── …                       # google_chat, ntfy, photon, raft, simplex, …

gateway/platforms/                  # core base + legacy direct adapters
├── base.py              # BasePlatformAdapter — shared logic for all platforms
├── signal.py            # Signal via signal-cli REST API
├── weixin.py            # Weixin (personal WeChat) via iLink Bot API
├── bluebubbles.py       # Apple iMessage via BlueBubbles macOS server
├── qqbot/               # QQ Bot (Tencent QQ) via Official API v2 (sub-package)
├── yuanbao.py           # Yuanbao (Tencent) DM/group adapter
├── msgraph_webhook.py   # Microsoft Graph change-notification webhook (Teams, Outlook, etc.)
├── webhook.py           # Inbound/outbound webhook adapter
└── api_server.py        # REST API server adapter
```

**Deferred loading:** Bundled `kind: platform` plugins register cheap `register_deferred` loaders in `gateway/platform_registry.py` (via `hermes_cli/plugins.py`) so platform SDKs import only when the gateway starts, delivers, or runs setup/status — not on plain `hermes chat`. Resolution loads one adapter on lookup; full enumeration runs pending loaders only on paths that need every platform.

Experimental connector-backed platforms use the generic relay adapter in `gateway/relay/` instead of a direct platform module. When `GATEWAY_RELAY_URL` or `gateway.relay_url` is configured, the gateway registers the `relay` platform, dials the connector over an outbound WebSocket, and receives `descriptor`, `inbound`, and `interrupt_inbound` frames on that same socket. The connector advertises a `CapabilityDescriptor`; Hermes can send normal outbound replies, token-less `follow_up` operations, and interrupt frames back through the relay. The source-grounded wire contract lives in [`docs/relay-connector-contract.md`](https://github.com/NousResearch/hermes-agent/blob/main/docs/relay-connector-contract.md).

Adapters implement a common interface:
- `connect()` / `disconnect()` — lifecycle management
- `send()` — outbound message delivery
- inbound events are normalized into a `MessageEvent` and forwarded via `handle_message()`

Internal push wakes use `gateway.wake.admit_internal_event`: the public
`handle_message()` still returns `None`, but the event's process-local
`_gateway_accepted` receipt is set only after scheduling or queue insertion.
A missing handler, mismatched explicit session key, or queue-cap drop is not
acceptance. Custom adapters overriding ingress should delegate internal events to
`BasePlatformAdapter.handle_message()` (or explicitly record actual admission),
not equate a consumed/dropped callback with acceptance. This receipt is separate
from heartbeat execution accounting and does not bypass authorization, emergency
stop, or later turn-preparation gates.

### Token Locks

Adapters that connect with unique credentials call `acquire_scoped_lock()` in `connect()` and `release_scoped_lock()` in `disconnect()`. This prevents two profiles from using the same bot token simultaneously.

A lock conflict is emitted as `{scope}_lock` with `retryable=True` so a **mid-run** reconnect can recover once the other holder exits. At **startup**, though, a live foreign holder is a configuration conflict: `gateway/restart.py::is_global_startup_conflict()` recognizes the `*_lock` / `lock_conflict` code families and the startup router parks the platform `fatal` instead of retry-queueing it. With nothing else connected the gateway exits `78` (`EX_CONFIG`, `gateway_state=startup_failed`) so the supervisor stops restarting it; alongside a genuinely transient peer failure the gateway stays alive and only the peer retries.

## Delivery Path

Outgoing deliveries (`gateway/delivery.py`) handle:

- **Direct reply** — send response back to the originating chat
- **Home channel delivery** — route cron job outputs and background results to a configured home channel
- **Explicit target delivery** — the send engine specifying `telegram:-1001234567890`, exposed via the [`hermes send` CLI](/guides/pipe-script-output) for shell scripts and via cron `deliver:` targets
- **Cross-platform delivery** — deliver to a different platform than the originating message

Cron job deliveries are NOT mirrored into gateway session history — they live in their own cron session only. This is a deliberate design choice to avoid message alternation violations.

## Hooks

Gateway hooks are Python modules that respond to lifecycle events:

### Gateway Hook Events

| Event | When fired |
|-------|-----------|
| `gateway:startup` | Gateway process starts |
| `session:start` | New conversation session begins |
| `session:end` | Session completes or times out |
| `session:reset` | User resets session with `/new` |
| `agent:start` | Agent begins processing a message |
| `agent:step` | Agent completes one tool-calling iteration |
| `agent:end` | Agent finishes and returns response |
| `command:*` | Any slash command is executed |

Hooks are discovered from `gateway/builtin_hooks/` (an extension point — currently empty in the shipped distribution; `_register_builtin_hooks()` is a no-op stub) and `~/.hermes/hooks/` (user-installed). Each hook is a directory with a `HOOK.yaml` manifest and `handler.py`.

## Memory Provider Integration

When a memory provider plugin (e.g., Honcho) is enabled:

1. Gateway creates an `AIAgent` per message with the session ID
2. The `MemoryManager` initializes the provider with the session context
3. Provider tools (e.g., `honcho_profile`, `viking_search`) are routed through:

```text
AIAgent._invoke_tool()
  → self._memory_manager.handle_tool_call(name, args)
    → provider.handle_tool_call(name, args)
```

4. On session end/reset, `on_session_end()` fires for cleanup and final data flush

### Memory Flush Lifecycle

Explicit conversation boundaries (such as `/new`, `/reset`, or `/resume`) flush and finalize the outgoing session. Idle time and daily boundaries never finalize it.

Resource-only TTL, LRU, and memory-pressure eviction commits the cached transcript to configured memory providers before releasing the agent's clients. It does not close the durable conversation: the next turn reloads the same transcript and identity.

## Background Maintenance

The gateway runs periodic maintenance alongside message handling:

- **Cron ticking** — checks job schedules and fires due jobs
- **Session housekeeping** — reclaims cached resources without ending transcripts
- **Memory flush** — commits memory before soft cache eviction
- **Cache refresh** — refreshes model lists and provider status

## Process Management

The gateway runs as a long-lived process, managed via:

- `hermes gateway start` / `hermes gateway stop` — manual control
- `systemctl` (Linux) or `launchctl` (macOS) — service management
- PID file at `~/.hermes/gateway.pid` — profile-scoped process tracking

**Profile-scoped vs global**: `start_gateway()` uses profile-scoped PID files. `hermes gateway stop` stops only the current profile's gateway. `hermes gateway stop --all` uses global `ps aux` scanning to kill all gateway processes (used during updates).

## Related Docs

- [Session Storage](./session-storage.md)
- [Cron Internals](./cron-internals.md)
- [ACP Internals](./acp-internals.md)
- [Agent Loop Internals](./agent-loop.md)
- [Messaging Gateway (User Guide)](/user-guide/messaging)
