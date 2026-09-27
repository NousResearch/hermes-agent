---
sidebar_position: 7
title: "Gateway Internals"
description: "How the messaging gateway boots, authorizes users, routes sessions, and delivers messages"
---

# Gateway Internals

The messaging gateway is the long-running process that connects Hermes to 20+ external messaging platforms through a unified architecture.

## Key Files

| File | Purpose |
|------|---------|
| `gateway/run.py` | `GatewayRunner` facade — composes the `gateway/run_*.py` sibling mixins (startup, adapters, inbound, turn, busy, goals, notifications, shutdown, …) and `gateway/slash_commands_*.py` handlers |
| `gateway/session.py` | `SessionStore` — conversation persistence and session key construction |
| `gateway/run_session_options.py` | Per-session runtime options: the durable-first commit, the session admission lock and `apply_session_options` (see [Per-Session Runtime Options](#per-session-runtime-options)) |
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
   - Resolve session key via `_session_key_for_source()` (format: `agent:{namespace}:{platform}:{chat_type}:{chat_id}`; the namespace is `main` for the default profile, `<profile>` under multiplexing — see [Multiplexed profiles](#multiplexed-profiles))
   - Check authorization (see Authorization below)
   - Check if it's a slash command → dispatch to command handler
   - Check if agent is already running → intercept commands like `/stop`, `/status`
   - Otherwise → create `AIAgent` instance and run conversation
4. **Response** is sent back through the platform adapter

### Session Key Format

Session keys encode the full routing context:

```
agent:{namespace}:{platform}:{chat_type}:{chat_id}
```

For example: `agent:main:telegram:private:123456789` for the default profile, or
`agent:work:telegram:private:123456789` when the multiplexer routes that chat to profile `work`
(`gateway/session.py::_session_key_namespace`; a profile literally named `main` is marked `main~`).

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
4. **Global allow-all** (`GATEWAY_ALLOW_ALL_USERS`, or `gateway.allow_all_users` in `config.yaml`, bridged to the env var by `gateway/config_loader.py::bridge_core_env_settings`) — if set, all users across all platforms are authorized
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

Experimental connector-backed platforms use the generic relay adapter in `gateway/relay/` instead of a direct platform module. When `GATEWAY_RELAY_URL` or `gateway.relay_url` is configured, the gateway registers the `relay` platform, dials the connector over an outbound WebSocket, and receives `descriptor`, `inbound`, and `interrupt_inbound` frames on that same socket. The connector advertises a `CapabilityDescriptor`; Hermes can send normal outbound replies, token-less `follow_up` operations, and interrupt frames back through the relay. The source-grounded wire contract lives in [Relay ↔ Connector contract](relay-connector-contract.md).

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

A lock conflict is emitted as `{scope}_lock` with `retryable=True` so a **mid-run** reconnect can recover once the other holder exits. At **startup**, though, a live foreign holder is a configuration conflict: `gateway/restart.py::is_global_startup_conflict()` recognizes the `*_lock` / `lock_conflict` code families and the startup router parks the platform `fatal` instead of retry-queueing it. With nothing else connected the gateway exits `78` (`EX_CONFIG`, `gateway_state=startup_failed`) so the supervisor stops restarting it: systemd via `RestartPreventExitStatus=78`, s6 via finish→125, launchd via `KeepAlive.SuccessfulExit=false` after the stderr wrapper maps 78→0. Alongside a genuinely transient peer failure the gateway stays alive and only the peer retries.

## Delivery Path

Outgoing deliveries (`gateway/delivery.py`) handle:

- **Direct reply** — send response back to the originating chat
- **Home channel delivery** — route cron job outputs and background results to a configured home channel
- **Explicit target delivery** — the send engine specifying `telegram:-1001234567890`, exposed via the [`hermes send` CLI](../guides/pipe-script-output.md) for shell scripts and via cron `deliver:` targets
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

Hooks are discovered from `gateway/builtin_hooks/` (an extension point — currently empty in the shipped distribution; `_register_builtin_hooks()` is a no-op stub) and `<profile home>/hooks/` (user-installed; `~/.hermes/hooks/` for the default profile, one directory per served profile under multiplexing — paths resolve at call time, never at import). Each hook is a directory with a `HOOK.yaml` manifest and `handler.py`.

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

**Profile-scoped vs global**: `start_gateway()` uses profile-scoped PID files. Standalone (one gateway per profile), `hermes -p x gateway stop` stops only that profile's gateway. Under multiplexing there is ONE gateway process per host, owned by whichever profile launched it (`gateway/host_rendezvous.py` publishes its PID, home and served set; `gateway/host_attach.py` is the attach/rescan/refuse decision every lifecycle verb goes through): `hermes gateway stop` on the owner takes every served profile down, and `hermes -p x gateway stop` for a served secondary refuses with exit 78 (it has no gateway of its own). A second `gateway run` for a served profile attaches and exits 0 — under a service supervisor it exits 75 (EX_TEMPFAIL) instead, so the redundant unit is RETRIED rather than parked: "someone else serves me right now" is a runtime observation that ends when that process does, and 78 (which systemd, s6 and launchd all treat as permanent) would strand the profile. ATTACH requires a live `identify` answer from the owner; a rendezvous record with nothing answering behind it proves an owner exists but never that it serves you, so it yields a transient refusal (exit 75), never an attach. An owner that answers `multiplex: False` to the rescan is another profile's *standalone* gateway, not a multiplexer that excluded you: the verb starts this profile's own gateway beside it (the one-process-per-profile topology), it does not refuse — refusing there exited 78 and parked every launchd unit but the first to claim the host lock. `hermes gateway stop --all` uses global `ps aux` scanning to kill all gateway processes (used during updates). Liveness is decided by `gateway.status.live_gateway_pid_for_home` (PID + start-time fingerprint), never bare PID existence.

## Multiplexed profiles

With `gateway.multiplex_profiles: true` one process serves the default profile plus every live directory under `profiles/` (`hermes_cli/profiles.py::profiles_to_serve(multiplex=True)`). `os.environ` and module globals hold the **launch** profile's values, so every activity for a secondary binds its scope explicitly — a profile is home + secret scope + terminal scope together:

| Activity | Binding |
|---|---|
| Routed turn | `gateway/run.py::_profile_runtime_scope(home)` via `run_turn.py::_profile_scope_for_source` |
| Agent release / eviction (TTL, LRU, memory pressure) | `gateway/run_agent_cache.py::_run_release_in_profile_scope` |
| Shutdown | `gateway/run_shutdown.py::_finalize_session` |
| Post-turn media delivery | `gateway/platforms/base.py::_media_delivery_scope` |
| Cron tick | `cron/scheduler_provider.py::_profile_cron_scope(home)` (one ticker, profiles in sequence) |
| Child processes (`hermes -p X` workers, relay turns, browser drivers) | `tools/environments/local.py::served_profile_child_env` |
| Background threads | `agent/memory_provider.py::spawn_context_thread` |

Secret reads fail closed (`agent.secret_scope.get_secret` raises `UnscopedSecretError`) only after `set_multiplex_active(True)`, which the gateway, cron, `gateway migrate` and the Desktop/dashboard `serve` backend set. Adapter YAML never reaches `os.environ` under multiplex: `gateway/platforms/_shared.py::apply_yaml_bridge` seeds `PlatformConfig.extra` and skips the environ write under a secondary's scope; gates read through `platform_gate_env`. Shared-ingress platforms (WhatsApp bridge, Relay) run on the default profile only; a secondary that enables one is logged once and stamped into runtime status (`run_adapters.py::_note_unserved_secondary_platform`). Per-profile isolation as the user sees it: [Multi-profile gateways § What is isolated per profile](../user-guide/multi-profile-gateways.md#what-is-isolated-per-profile).

## Mid-run plugin loading

Plugins that load after the adapters connected (install/enable from the CLI, Desktop, dashboard or
`plugins.manage`; a tool-triggered force re-discovery) re-wire their platform handlers without a restart
(#87770). The pieces, all in `gateway/run_plugin_rewire.py`:

- **Discovery listener** — `_start_recover_previous_run` subscribes `PluginManager.on_plugin_loaded` for the
  launch profile and `_load_secondary_profile_config` does so per served profile. The event fires from inside
  `discover_and_load` (never from an RPC) for the newly loaded plugins; the callback hops onto the gateway
  loop with `call_soon_threadsafe`.
- **Idempotent re-wire** — `BasePlatformAdapter.rewire_plugin_handlers()` re-reads
  `get_platform_handler_factories(platform)` and runs only factories not yet wired on the live native
  client (keyed `(plugin, qualname)` because a force reload hands back new function objects). Telegram
  hoists the added handlers ahead of core's catch-alls; Slack also re-registers missing
  `register_slack_action_handler` callbacks once per `AsyncApp`.
- **`reload-plugins` control verb** — other processes (`hermes plugins install`, `hermes serve`) ask the
  running gateway to force-rescan the requested (served) home; the answer carries `plugins`, per-plugin
  `activations` and `adapters_rewired`, so the caller can say "active now" truthfully.
- **Scope limit** — handlers only. Tools and system-prompt sections of a late plugin wait for the next
  session (prompt-cache invariant); portable MCP servers wait for `mcp.reload`. Nothing un-wires on disable.

## Per-Session Runtime Options

A session's model, reasoning effort and `/fast` tier can be set by the user (`/model`,
`/reasoning`, `/fast`, typed or from a picker) or by a host UI through the structured API. Both
paths share one write in `gateway/run_session_options.py` (`GatewaySessionOptionsMixin`):

- **Durable first.** `_commit_session_runtime_options(source, patch)` persists the patched fields
  with `SessionStore.set_runtime_options` (one strict save: it raises `OSError` when state.db was the
  routing source and did not save, even if the `sessions.json` mirror did) and only then assigns live
  state. A failed save raises with memory, the queued model note and any one-turn restore untouched,
  so a slash command fails visibly instead of replying "set" for a value a restart would drop.
- **Admission lock.** One per-session `asyncio.Lock` is shared by the commit, the idle->running claim in
  `_handle_message` (no await inside, so an uncontended claim never yields), the `/login` free-tier clear
  and boot-resume. Boot-resume cannot await: while a commit holds the lock it defers the session, and the
  commit reschedules it on release. The commit re-checks busy under the lock, so a picker tap mid-turn
  gets the same busy reply as a typed command. Lock order: `_model_switch_lock`, then the admission lock.
  The lock is not re-entrant.
- **Boundaries.** The commit resolves the route first and consumes an auto-reset boundary (#48031,
  #58403). The store write is a compare-and-swap on the entry's `session_id`, and
  `_clear_conversation_scope` bumps a conversation epoch, so a `/new`, `/resume` or reset that lands while
  a write is in flight leaves the value in neither memory nor the new entry (`SessionMissing`).
- **One-turn overrides are never saved.** `/model --once` and `/moa` never commit. A commit writes only
  the fields it names, and a durable model commit pops an armed one-turn restore when it is assigned.
- **Cancellation.** Live state is assigned by a done-callback on the executor Future of the write, so no
  task cancellation (including `asyncio.run` teardown) separates the two. The caller waits for that
  callback under the lock across any number of cancels, then re-raises the last cancel without
  `uncancel()`; a write failure behind a cancel is logged at WARNING with its traceback.
- **What is stored.** `model_override` keeps only model/provider/base_url (credentials are re-resolved on
  rehydrate), `reasoning_override` only `{enabled, effort}`, and `service_tier_override` is `"normal"`
  (explicit normal), `"priority"`, `"auto"` or `"cold"`; absent means inherit `agent.service_tier`.
  `_rehydrate_session_runtime_options` reads them back once per process, and a live value wins.

`GatewayRunner.apply_session_options(source, options)` is the structured API. Keys: `model` (`""` means
inherit the configured model), `provider`, `reasoning_effort` (`""` means inherit, `none`, or a
`VALID_REASONING_EFFORTS` level), `fast` (bool), `confirm_model_selection` and `initial` (no next-turn
model note). The source goes through `_canonicalize` and runs under `_profile_scope_for_source`. The whole
patch is validated first (same switch resolution and selection guards as `/model`); nothing moves on a
rejection. Only changed fields are written, so re-sending current values is a no-op. A slash command
that lands between validation and commit wins (`conflict`). Results: `accepted` (with `applied` and
`effective`), `confirmation_required` (with `title`, `message` and `error`), or `rejected` with `code`
and `error`: `invalid_options`, `invalid_session`, `session_busy`, `model_rejected`,
`reasoning_rejected`, `fast_rejected`, `fast_unsupported`, `conflict`, `session_missing` or
`durable_write_failed` (an `OSError` from the store).

## Related Docs

- [Session Storage](./session-storage.md)
- [Cron Internals](./cron-internals.md)
- [ACP Internals](./acp-internals.md)
- [Agent Loop Internals](./agent-loop.md)
- [Messaging Gateway (User Guide)](../user-guide/messaging/index.md)
