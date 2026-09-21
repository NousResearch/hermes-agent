---
title: "Multiplexing Gateway Internals"
description: "Design of the one-gateway-for-all-profiles mode: scope composition, secret scope, inbound routing, persistence"
---

# Multiplexing Gateway

One gateway process can serve every profile in the install. The mode is on by
default (`gateway.multiplex_profiles`, default `true`), and everything it
changes reverts the moment the flag is off. An *unset* flag is settled at boot
by `hermes_cli/gateway_multiplex_mode.py::resolve_multiplex_mode`, which runs
the `hermes gateway migrate` preflight and keeps the gateway standalone when a
secondary still runs its own gateway, a blocker exists, or the host cannot be
migrated (see "The mode flag"). This document is the design rationale
referenced from `agent/secret_scope.py` ("Workstream A"): what is isolated per
profile, the mechanism that isolates it, and what deliberately stays
process-global.

## Overview

Without multiplexing, one gateway process serves exactly one profile — its
`.env`, sessions, skills, and platform adapters — and multi-profile installs
run one process per profile. Multiplexing collapses that into a single
process: the default profile plus every served named profile get their own
adapters, secrets, sessions, and cron ticks, while sharing one event loop, one
HTTP listener, one process lock, and one status surface.

The design constraint that shapes everything below: **profile A's turns must
never observe profile B's state**. Secrets, homes, sessions, and adapter lanes
are isolated per profile; anything that cannot yet be isolated fails closed or
is documented as a known limitation at the end of this document.

## The mode flag

- Config: `gateway.multiplex_profiles` (also accepted at top level). Parsed in
  `gateway/config.py` with precedence env > config > unset. `GatewayConfig`
  keeps an unset flag as `None` (readers test truthiness, so it reads as off);
  `load_gateway_config_for_runner` then calls `resolve_multiplex_mode`, which
  writes the boot verdict — `True` on a quiet multi-profile default install,
  `False` with a logged reason otherwise. Explicit values pass through
  verbatim; a config injected into `GatewayRunner(config=...)` is not resolved.
- Other processes read the LIVE gateway's `served_profiles` record first and
  the explicit flag second (`gateway_multiplex_mode.default_gateway_multiplexes`
  / `explicit_multiplex_flag`), never the merged default: `named_profile_served_
  by_running_multiplexer`, the enroll warning, the dashboard's listener guard,
  the cron-fire port resolver, container boot, and the migration plan
  (`_read_multiplex_flag`, so an unset default reads as "not yet multiplexed"
  and the fold proceeds).
- Env override: `GATEWAY_MULTIPLEX_PROFILES` accepts explicit truthy/falsy
  tokens only; a blank or unrecognized value returns "no override" so an empty
  deployment secret cannot shadow a config opt-in.
- At startup, `GatewayRunner.__init__` calls
  `agent.secret_scope.set_multiplex_active(...)` once. `_MULTIPLEX_ACTIVE` is
  a plain module global, not a contextvar: it describes the deployment mode,
  not a per-task value. Its only job is to arm the fail-closed behavior in
  `get_secret()`.
- The dashboard/Desktop backend (`hermes serve`) has no such flag, so
  `hermes_cli/web_server.py::start_server` calls
  `tui_gateway.launch_profile_policy.activate_multi_profile_hosting_eagerly()`
  at boot: the host arms the guard when the machine has more than one servable
  profile home, instead of waiting for the first `?profile=<other>` request.
  Activation is one-way, and anything the backend had already done by then
  (idle-reaper transcript flushes, hosted rooms, cron) was never re-scoped. A
  genuinely single-profile host still never activates.
- With the guard armed the **launch profile is a tenant too**: a body with no
  routed profile binds `launch_profile_scope_if_multiplexed()` (its `.env` over
  the env frozen at activation) rather than running unscoped on ambient
  `os.environ`.

## Scope composition

Every inbound event composes the same two context-local scopes before any
profile-owned code runs:

```
platform event
   │
   ▼
profile_routes match ──► served-set check ──► SessionSource.profile stamped
   │                                           (gateway/profile_routing.py)
   ▼
_profile_runtime_scope(profile_home)           (gateway/run.py)
   ├── set_hermes_home_override(home)          config / state.db / skills /
   │                                           memory / sessions resolve here
   └── set_secret_scope(profile .env + secret sources)
   │                                           provider keys, platform tokens
   ▼
agent turn (worker thread via copy_context())
   │
   ▼
scope unwound in finally
```

`_profile_runtime_scope` wraps every seam where profile-owned code executes:
secondary adapter startup, connect and reconnect, the primary platform event
handler, inbound preprocessing, `/model` and session-info resolution,
background tasks, and the agent turn itself. Config reloads run under the
default profile's scope so global gateway settings (`#64674`) resolve
consistently.

Both scopes are `contextvars`, so they propagate into executor worker threads
via `copy_context()` and unwind deterministically — nothing is written to
`os.environ`, ever.

## Workstream A: context-local secret scope

`agent/secret_scope.py` exists because the obvious implementation — union all
profile `.env` files into `os.environ` — leaks profile A's keys into profile
B's turns and into every subprocess spawned with `env=dict(os.environ)`.

- `build_profile_secret_scope(home)` merges the profile's `.env` with its
  configured secret sources, skipping globals.
- `set_secret_scope(mapping)` installs it for the current task.
- `get_secret(name)` resolves: global allowlist → active scope → fallback.
  The fallback is the load-bearing part:
  - multiplexing **off**: reads `os.environ`, so single-profile gateways and
    every non-gateway caller behave exactly as before;
  - multiplexing **on**, no scope installed: **raises `UnscopedSecretError`**
    rather than silently reading the process environment. An un-migrated call
    site fails loud at that exact line instead of leaking another profile's
    value.
- A small allowlist (`HERMES_HOME`, `HERMES_PROFILE`, proxy settings,
  `API_SERVER_*` listener settings — but deliberately not `API_SERVER_KEY`)
  stays global because those describe the process, not a profile.
- Cloud SDK *default credential chains* are ambient by construction
  (`google.auth.default()`, `DefaultAzureCredential`, `boto3.Session()` with
  no keys): every source they walk — process env, CLI caches, instance
  metadata — is the launch context's identity. Under multiplexing a served
  profile without a complete credential of its own is **refused** by the
  Vertex, Entra ID and Bedrock adapters rather than minting that identity
  against its own `base_url`; standalone runs keep the chain.

Because the per-turn `.env` reload is a no-op under multiplexing, rotated
credentials are picked up through the profile scope on the next turn — never
via `os.environ`. This holds at the loader boundary, not just the gateway's
reload helper: `hermes_cli.env_loader.load_hermes_dotenv` skips the
process-global load whenever multiplexing is active *and* a profile-home
override is installed (import-time and cron callers hit it mid-turn), while
still hydrating the profile's external secret sources into its private
snapshot (`#77562`). The unscoped startup load is unchanged.

The same scope-authoritative rule covers the other `os.environ` seams a
routed turn can reach: `${VAR}` / `${env:VAR}` references in a profile's
`config.yaml` resolve through `get_secret` when a scope is installed
(`#84079`), and `.env` writes made under a scope (`save_env_value`, e.g. a
`/pair` grant mirror) update the installed scope mapping instead of the
process environment (`#88441`).

## The HERMES_HOME override

`hermes_constants.py` holds a context-local override consulted by
`get_hermes_home()` before the `HERMES_HOME` env var. Everything that resolves
paths through it — config, `state.db`, skills, memory, SOUL, sessions, kanban,
goals, plugin discovery, MCP startup — follows the active profile
automatically. `get_process_hermes_home()` exists for the few machine-level
assets that must not follow the override. `hermes_home_key()` gives
per-home registries a stable scope key. A one-shot warning (`#18594`) fires if
profile-scoped code runs without the override where one is expected.

## Inbound routing

`gateway.profile_routes` maps `(platform, user_id, guild_id, chat_id, thread_id)` to a
profile; matching is conjunctive, most-specific-first, with parent-chain chat
matching for threads. Routing only runs when multiplexing is active, and a
matched route whose target is outside the served set is rejected (the event is
dropped, not misdelivered). Full schema and matching rules:
[Routing shared-bot chats to profiles](../user-guide/multi-profile-gateways.md#routing-shared-bot-chats-to-profiles-profile_routes).

## Serving selected profiles

`profiles_to_serve(multiplex, profile_allowlist)` in `hermes_cli/profiles.py`
is the single chokepoint for which profiles a multiplexer serves: default plus
every valid profile directory, optionally filtered by allowlist. A malformed
allowlist fails safe to default-only. The served set gates adapter startup,
cron ticking (`#69377`), `/p/<profile>/` HTTP admission, route eligibility,
and the runtime status surface. An excluded profile stays installed and can
still run its own standalone gateway.

## Per-profile persistence

`SessionStore` binds no database handle at construction (`#88532`). Session
DB handles are resolved at call time through the active HERMES_HOME override —
one cached handle per resolved `profiles/<name>/state.db` — so sessions land
in the owning profile's store even when the store object itself is shared.
Pairing stores are constructed per served profile.

## Per-bot session lanes

Session keys are namespaced by profile (`agent:main` for default,
`agent:<name>` for named profiles). Every inbound event carries ONE frozen
`RoutingIdentity` (`gateway/session_identity.py`), resolved by
`resolve_identity()` at the runner's ingress handlers and pinned on the source
as a wire-invisible attribute: `transport_profile` (the bot that received it —
credential, allowlist, `authorization_home`), `runtime_profile` (the routed
profile that executes — `runtime_home`, key `namespace`, `store_path`) and a
weak `transport` ref to the receiving adapter. `"default"` is spelled out;
`None` never means default. Under multiplexing a route to an unserved profile
raises `IdentityUnresolved` and the event is dropped.

Adapters also carry `_owner_profile` (installed at adapter configuration time,
before any inbound event). Every ingress path canonicalizes the identity FIRST
— `BasePlatformAdapter._canonicalize` runs at `handle_message`, text/photo/album
batching, the busy path and every adapter-derived session key; the runner's
per-profile and default handlers, the auth-check callback and the shared
`_handle_message` gate do the same — so no lane is keyed before the receiving
bot is known. Text/media batching, active-session tracking, the busy-session
guard, `/stop` `/new` `/reset` and clarify replies are all keyed per lane, so
two bots sharing a chat do not share a session lane and a control command on one
bot cannot reach the other's run. A route to an unserved profile is dropped with
one WARNING at the first seam it reaches, never keyed into `agent:main`. Copy a
source with `session_identity.replace_source`, not `dataclasses.replace`, or the
copy loses its transport and identity.

## Intake vs delivery: which bot acts on an event

Two runner seams answer the two questions a multiplexed gateway keeps
conflating (`gateway/authz_mixin.py`):

- `_intake_adapter_for(source)` — the bot that **received** the event. Live
  provenance only: the transport ref `build_source` pinned, the process-level
  relay adapter for relay-delivered events, or the adapter currently
  registered for the identity's `transport_profile` after a reconnect. It
  gates intake policy (Slack ignored channels, relay fronting, re-dispatch of
  a queued live event) and returns `None` for a source with no live
  provenance — nothing may re-admit a restored row on a guessed bot.
- `_delivery_adapter_for(source)` — the bot that **answers**: sends, edits,
  typing, progress, pickers, pending-message slots. The receiving bot whenever
  it is known; otherwise the unique owner of `(platform, runtime_profile)` —
  a secondary's own adapter, the primary for a shared-bot satellite, and
  `None` for a secondary whose bot is disconnected (it never borrows the
  default bot).

| Topology | Runtime (`runtime_profile`, key namespace, home) | Intake | Delivery |
| --- | --- | --- | --- |
| Per-credential bot, no route | The bot's own profile | Owner adapter | Owner adapter |
| Shared credential → satellite via `profile_routes` | Routed profile | Receiving (shared) adapter | Receiving adapter; after a restart the satellite still drains through the primary |
| Shared bot → a profile that owns its own bot | Routed profile | Receiving adapter | Receiving adapter — the conversation stays with the bot the user wrote to |
| Secondary-owned bot → `default` (`bot_profile: <secondary>`) | `default` (`agent:main`, default home) | Receiving (secondary) adapter | Receiving adapter |
| Restored / synthetic source, no live provenance | Stored `source.profile` | **None** (fail closed) | Unique owner of `(platform, runtime)`, else `None` |

Outside multiplexing there is one adapter per platform, so both seams return
it. `tests/gateway/test_multiplex_transport_matrix.py` asserts every row.

### Restore, relay, callbacks and thread hops

The routing entry persists `transport_profile` next to the key (and the
`sessions.transport_profile` column in `state.db`), so after a restart a
revived lane still knows which bot received it: `_restored_source(entry)`
re-pins a `RoutingIdentity` with no live adapter and `_delivery_adapter_for`
delivers through that bot's adapter or fails closed — a satellite routed
through the default bot keeps answering from the default bot, a lane owned by
a secondary never falls back to the default bot's credential. Entries written
before the column existed carry `null` and keep the shared-bot heuristics.
Over the relay, every outbound frame's `metadata.profile` (and `follow_up`'s
key namespace) tells the connector which profile to stamp on the next
`passthrough_forward`, so a button press after a routed slash command stays in
the same profile. Deferred callbacks (`/model` picker) capture the routed home
at command time and the gateway's executor hops copy the ContextVar scope.

## Control plane

Desktop plugins reach the gateway only through the ws JSON-RPC door, so
profile enumeration and configuration live in
`tui_gateway/methods_profiles.py`: `profiles.list`, `profiles.create`,
`profiles.describe`, `profiles.configure`, `profiles.set_asset`,
`profiles.get_asset`. Reads and writes run under the target profile's
HERMES_HOME override. Asset writes are atomic, type- and size-capped.

## Failure modes

- Fatal at startup: multiplex config errors and a secondary profile enabling a
  port-binding platform (`MultiplexConfigError`,
  `SecondaryPortBindingConfigError`) — one shared HTTP listener is owned by
  the default profile.
- Skipped, not fatal: a single misconfigured secondary adapter is skipped with
  a warning rather than taking down the multiplexer.
- Fail-closed: unscoped `get_secret()` under multiplexing raises; a routed
  event targeting an unserved profile is dropped; an unscoped `/p/` request
  enters the default profile's scope (`#61276`) rather than an undefined one.
- Fallback: an external `cron.provider` does not support multiplexing and
  falls back to the built-in ticker with a warning.

## Known limitations

Process-global state that is not yet profile-scoped:

| Surface | State at time of writing |
| --- | --- |
| MCP discovery and tool registration | Process-global; the first profile to build an agent wins the discovery slot. Full per-profile MCP registries are tracked in `#67605`. |
| Terminal / sandbox env (`TERMINAL_*`) | Global by allowlist; tools read it from the process environment. |
| Built-in tool registry | Built-ins are process-global; plugin-registered tools are overlaid per profile via `hermes_home_key()`. |
| Provider/capability registries | Same hybrid overlay pattern (browser, image-gen, TTS, transcription, video-gen, web-search, secret sources). |
| HTTP listener, relay ingress, process lock | One per process, owned by the default/active profile. Per-profile `runtime_status.json` is still written. |

## Non-goals

Multiplexing isolates *profiles*; it does not authenticate or authorize *end
users*. A profile is a configuration, not a person: the gateway trusts its
transport and its routing table to decide which profile an event belongs to.
Request-level identity and per-user authorization above the profile layer are
out of scope for this document.

## Related

- [Multi-profile gateways](../user-guide/multi-profile-gateways.md) — user-facing guide, including `profile_routes`
  and the standalone one-gateway-per-profile alternative.
- `agent/secret_scope.py`, `hermes_constants.py`, `gateway/profile_routing.py`,
  `gateway/run.py` (`_profile_runtime_scope`), `hermes_cli/profiles.py`
  (`profiles_to_serve`), `gateway/session.py`, `tui_gateway/methods_profiles.py`.
