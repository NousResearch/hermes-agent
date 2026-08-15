---
sidebar_position: 5
---

# Connecting Desktop to Many Hermes Instances

Register every Hermes backend you own — the local runtime, remote gateways on
your LAN or VPS, SSH hosts, and Hermes Cloud instances — in one desktop app,
and use the agents on all of them side by side.

This is the desktop-side complement to
[Running Many Gateways at Once](./multi-profile-gateways.md): that page is
about hosting several gateways on one machine; this one is about one desktop
app talking to several machines.

## The connection registry

**Settings → Connections** manages a named registry of agent sources. Each
entry is a *connection*:

| Kind | What it is | Auth |
|---|---|---|
| **Local** | The runtime this app manages on your machine | automatic |
| **Remote gateway** | A `hermes serve` backend reachable over HTTP(S) — LAN, Tailscale, VPS | session token or OAuth |
| **SSH** | A Hermes install reached over SSH; the app opens the tunnel and starts the dashboard for you | SSH key + adopted token |
| **Hermes Cloud** | A hosted instance discovered through your Nous account | portal sign-in |

Rules worth knowing:

- **Every connection needs a unique device name** ("Homelab", "Work laptop").
  The name shows up everywhere the instance appears — roster badges, handles,
  update results.
- The **local** entry is managed by the app and cannot be removed. Removing
  any other connection tears down its live backends and tunnels; the instance
  itself is untouched.
- **Test** probes the connection's own HTTP *and* WebSocket legs, so a pass
  means chat will actually work — not just that the host pinged.
- Cloud entries come from the Hermes Cloud sign-in/discovery flow, not a
  hand-typed URL.
- Tokens are encrypted with the OS keyring (same plain-text opt-in as
  Settings → Gateway on keyring-less Linux), and never leave the Electron
  main process.

### Migrating from the single-connection settings

The first launch of a registry-capable build imports your existing settings
automatically: the global connection mode and any per-profile overrides from
Settings → Gateway become named registry entries (deduplicated by URL/host).
The legacy settings file is left untouched, so older builds on the same
machine keep working.

## Agents across sources

Every profile on every registered connection is an *agent*. The union roster
is what multi-source surfaces (and plugins like
[Bot Mode](https://github.com/NousResearch/Hermes-Bot-Mode)) render:

- When the same profile name exists on several sources, handles disambiguate
  as **`@name-device`** — `research` on your Homelab renders as
  `@research-homelab`, while a profile unique across all sources keeps its
  bare name.
- Enumeration is eager but sockets are lazy: the app lists agents over REST
  without dialing every source's WebSocket. An unreachable source reports
  per-row instead of breaking the roster; SSH sources stay connect-on-demand
  until you first open an agent on them (no surprise tunnels).
- Opening an agent dials **its own source** — chats, sessions, and memory
  live on the machine that owns the profile, exactly as if you were using
  that instance directly.

Each `(connection, profile)` pair gets its own backend and socket, pooled
with the same idle-reaping as local per-profile backends — background agents
keep streaming while you look at another source.

## Updating every instance at once

**Settings → Connections → Update all instances** dispatches `hermes update`
to every eligible connection in parallel:

- **Local** updates through the app's own update pipeline (the same flow as
  Settings → Updates).
- **Remote and SSH** connections are told to update themselves via their own
  backend — the update runs on *that* machine.
- **Hermes Cloud** instances are skipped: the platform manages their
  versions.

Each instance reports independently, so one unreachable box never wedges the
batch. Backends that manage updates externally (Docker, Nix) refuse politely
with their own message, per row.

## For plugin authors

The Desktop [plugin SDK](../developer-guide/desktop-plugin-sdk.md) exposes the
multi-source surface directly:

- `host.connections()` — the registered connection list (labels, kinds,
  primary; never token bytes).
- `host.agents()` — the union roster: one row per `(source, profile)` with
  the precomputed `@name-device` handle.
- `host.ensureAgent(connectionId, profile)` — activate an agent's gateway so
  subsequent `host.request` calls hit its backend.
- `host.warmAgent(connectionId, profile)` — fire-and-forget socket pre-warm
  (hover-intent).

All four are feature-detected: on an older Desktop build they're absent and a
plugin should fall back to the single-source `profiles.list` flow. Bot Mode's
multi-source roster is the reference consumer.

## Troubleshooting

- **An agent shows but won't open** — run **Test** on its connection. The
  WebSocket leg failing while HTTP passes usually means a proxy, firewall, or
  gateway auth/origin guard is blocking `/api/ws`.
- **A remote source is missing from the roster** — its backend is down or
  unreachable; the roster lists it under sources with the error. SSH sources
  show *connect-on-demand* until first use — that's by design, not a failure.
- **"Update Hermes Desktop to chat with agents on other connections"** — the
  app predates the multi-connection stack; update the desktop app itself.
- **Duplicate device names** — not possible; names are enforced unique at
  save time. If a migrated name collided, it was suffixed (`Homelab 2`).
