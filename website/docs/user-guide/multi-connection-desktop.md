---
sidebar_position: 5
---

# Connecting Desktop to Many Hermes Instances

Register every Hermes backend you own — the local runtime, remote gateways on
your LAN or VPS, SSH hosts, and Hermes Cloud instances — in one desktop app,
and use the agents on all of them side by side. Connections are persistent:
each registered source dials its own backends and WebSockets on demand, and
background agents keep streaming while you look at another source.

This is the desktop-side complement to
[Running Many Gateways at Once](./multi-profile-gateways.md): that page is
about hosting several gateways on one machine; this one is about one desktop
app talking to several machines.

## Where to find it

Everything lives on the unified **Settings → Gateways** page (older builds had
separate **Gateway** and **Connections** pages; legacy Connections deep links
redirect there). Three doors lead to it:

- **Settings → Gateways** — the page itself (**Cmd/Ctrl+,**, then
  **Gateways** in the settings nav). The connections registry is a section
  of that page, below the machine-level connection-mode controls.
- **The sidebar profile rail** — the plug button at the right end of the rail
  (tooltip: **"Connect another Hermes gateway…"**) deep-links straight to
  the Gateways page. It is always visible, even before you have created
  a second profile or a second connection.
- **The command palette** — **Cmd/Ctrl+K**, then type *Gateways* (also
  matches *connections*, *add gateway*, *remote*, *ssh*, *instances*).

## The connection registry

The registry section of **Settings → Gateways** manages a named list of agent
sources. The
pane's intro says it plainly: *"Register every place your agents live — this
device, remote gateways on your network, and Hermes Cloud instances. All of
them are stored here."* Each entry is a *connection*:

| Kind | What it is | Auth |
|---|---|---|
| **Local** | "The Hermes runtime managed by this app." | automatic |
| **Remote gateway** | "A Hermes gateway reachable over HTTP(S) — LAN, Tailscale, or the internet." | session token or OAuth |
| **SSH** | "A Hermes install reached over SSH." The app opens the tunnel and starts the dashboard for you | SSH key + adopted token |
| **Hermes Cloud** | "A hosted instance discovered through your Hermes Cloud account." | portal sign-in |

Rules worth knowing:

- **Every connection needs a unique device name** ("Homelab", "Work laptop").
  The name shows up everywhere the instance appears — roster badges, handles,
  update results. Uniqueness is case-insensitive, so `Homelab` and `homelab`
  cannot coexist.
- The **local** entry is managed by the app (it wears a **This device** pill)
  and cannot be removed. Removing any other connection tears down its live
  backends and tunnels; the instance itself is untouched.
- One connection is always the **Primary** (pill on its row): it is the
  registry fallback for multi-source calls that do not name a source.
  **Make primary** does not switch the current Sessions workspace; removing
  the primary falls back to the local entry.
- **Test** probes the connection's own HTTP *and* WebSocket legs, so a pass
  (the *"Reachable"* toast) means chat will actually work — not just that the
  host pinged.
- **Duplicates are rejected when you save**: there is only ever one **local**
  entry; **remote** and **cloud** entries are deduplicated on the normalized
  URL (trimmed, trailing slashes stripped, lowercased — and across both
  kinds, so a cloud entry and a remote entry can't point at the same URL);
  **SSH** entries are deduplicated on the normalized `user@host:port` plus
  remote profile.
- Cloud entries normally come from the Hermes Cloud sign-in/discovery flow at
  the top of the Gateways page — the **Hermes Cloud** kind in the add-connection
  editor points you there.

Switch sources from the **Sessions** sidebar. Profiles, chats, messaging, and
cron stay scoped to that source; the app-managed window backend is still chosen
by the connection-mode controls above. **Primary** is the registry fallback and
does not switch the current workspace.

## Adding a connection, step by step

1. Open **Settings → Gateways** and scroll to the connections registry (or
   click the plug in the profile rail).
2. Click **Add connection**.
3. Pick the kind: **Local**, **Hermes Cloud**, **Remote gateway**, or **SSH**.
   (**Local** is disabled while the app-managed local entry exists — which is
   almost always; **Hermes Cloud** directs you to the cloud sign-in/discovery
   flow above.)
4. Fill the fields:
   - **Name** — required, unique; the "device name" shown everywhere this
     instance appears (placeholder: `Homelab`). Max 64 characters.
   - *Remote gateway only:*
     - **Gateway URL** — the base URL of a running `hermes serve` backend,
       e.g. `http://homelab.lan:9119`. Reverse-proxy path prefixes work.
     - **Authentication** — choose **Session token** or **OAuth**:
       - **Session token** — paste the dashboard session token from the
         remote gateway. When editing, *"Leave blank to keep the saved
         token."*
       - **OAuth** — sign in through the Nous Portal browser flow; no token
         to paste.
   - *SSH only:*
     - **SSH host** — one composite field in `user@host:22` form (user and
       port optional). Your SSH key is used; the app adopts a dashboard
       token over the tunnel.
5. Click **Save connection** (or **Cancel**).
6. Click **Test** on the new row and wait for *"Reachable"*.

Edit any non-local entry later with the pencil button, or remove it with the
trash button — removal asks for confirmation and reminds you that *"The
instance itself is not touched — you can add it again any time."*

:::info The remote backend is a running `hermes serve` process
Nothing here works unless the backend is actually up and reachable on the
other machine. The desktop app attaches to it; it does not start it for you
(except for SSH connections, where the app starts the dashboard over the
tunnel on demand). See
[Connecting to a remote backend](./desktop.md#connecting-to-a-remote-backend)
for backend-side setup — auth providers, binding to a non-loopback address,
and Tailscale guidance.
:::

### Migrating from the single-connection settings

The first launch of a registry-capable build imports your existing settings
automatically: the global connection mode and any legacy per-profile
overrides from Settings → Gateway become named registry entries (deduplicated
by URL/host). (Newer builds no longer offer per-profile overrides in the
Gateways settings page — gateway connections are machine-level, and profiles
are discovered from the gateways you connect.)
The legacy settings file is left untouched, so older builds on the same
machine keep working. If a migrated name collided, it was suffixed
(`Homelab 2`).

## Agents across sources

Every [profile](./profiles.md) on every registered connection is an *agent*.
The union roster is what multi-source surfaces (and the built-in
[Bot Mode](./bot-mode.md) roster) render:

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

### Switching and scoping

The sidebar foot follows one hierarchy: **source → profile → sessions**.
Sources are machines or hosted backends; profiles are isolated Hermes agents
that live on one source.

- With one registered source, no source control is added. Local-only Desktop
  keeps the same profile rail and keyboard flow as before.
- With several sources, the sidebar shows one named source selector. Its device,
  cloud, network, or terminal icon identifies the connection type; profile
  avatars remain a separate control after the divider. The same selector scales
  from two sources to a larger fleet without turning backends into profile-like
  glyphs or crowding profile actions out of the rail.
- Selecting a source restores the last profile used there. The profile rail
  then shows only that source's profiles; the home pill returns to its default
  profile and the layers pill shows **All profiles on this source**.
  **Cmd/Ctrl+1–9** continue to switch profiles within the active source.
- With more than thirteen profiles on the active source, their avatar strip
  condenses into a named profile selector. Large source and profile sets can
  therefore coexist without changing the **source → profile → sessions** model.
- **This device** remains a first-class source even when a remote connection is
  Primary. It can keep local sessions available during a remote outage, but the
  app does not call it "offline mode": the selected model or tools may still
  require internet access.
- The session list, messaging channels, cron jobs, settings, files, and memory
  are all scoped to the active `(source, profile)`. Switching from a Telegram
  source to a Signal source cannot leave the previous source's channel groups
  or sessions in the sidebar.
- Merely displaying the switcher reads Electron's local connection registry.
  Remote gateways are opened only when selected; there is no periodic fleet
  polling.
- Hovering an agent pre-warms its backend so the switch doesn't pay a cold
  boot.
- The **Capabilities** page (Skills / Tools / MCP) has a matching scope: its
  **Configuring** selector lists every `(profile, device)` agent from the
  union roster, and picking one reads and writes **that machine's** skills,
  toolsets, and MCP servers without switching the Sessions workspace. Hub
  installs, env keys, and MCP setup all land on the selected agent's backend.
  The MCP tab's *hot-reload into a live session* button appears only for agents
  on the gateway the window is connected to; edits on other machines apply on
  their next session.

Add, test, rename, or remove sources in **Settings → Connections**. The plug
button beside the profile actions is a shortcut to that single management
home, not a second add flow.

### Sessions and Bot Mode

Sessions intentionally show one active source at a time: this keeps files,
tools, channels, cron, and session history in one understandable execution
context. Bot Mode serves a different job and may present the union roster,
grouped by source, so a user can open one agent on a NAS and another on a VPS
from one surface. Opening a bot still activates its exact `(source, profile)`
route.

Direct bot mentions and delegation remain source-local by default. Crossing a
backend boundary changes filesystem, credentials, tools, and trust context, so
cross-source execution should be an explicit bridge rather than an accidental
side effect of sharing one Desktop window.

## Updating every instance at once

**Settings → Gateways → Update all instances** (shown once more than one
connection is registered) dispatches `hermes update` to every eligible
connection in parallel:

- **Local** updates through the app's own update pipeline (the same flow as
  Settings → Updates).
- **Remote and SSH** connections are told to update themselves via their own
  backend — the update runs on *that* machine.
- **Hermes Cloud** instances are skipped with a *"Managed by Hermes Cloud"*
  note: the platform manages their versions.

Each instance reports independently, so one unreachable box never wedges the
batch. Backends that manage updates externally (Docker, Nix) refuse politely
with their own message, per row.

## Security notes

- **Where tokens live.** Remote-gateway session tokens are encrypted at rest
  with Electron's `safeStorage` (the OS keychain — Keychain on macOS, DPAPI
  on Windows, the session keyring backend on Linux) and stay in the Electron
  main process; the renderer and plugins never see token bytes. OAuth tokens
  for native sign-in are stored the same way, keyed by gateway base URL, and
  refreshed automatically before expiry.
- **Keyring-less Linux.** On a Linux session without a usable keychain the
  app cannot encrypt the token; saving one raises an explicit opt-in dialog
  before it will store the
  token in plain text.
- **The registry file** (`connections.json` under the app's user-data
  directory) holds labels, URLs, and hosts — secrets only ever appear inside
  encrypted envelopes.
- The plugin SDK's `host.connections()` deliberately returns labels, kinds,
  and the primary id — never token material.

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

- **"Connection test failed"** — the backend isn't reachable at that URL from
  this machine. Check that `hermes serve` is running on the remote host, the
  port is open, and (for token auth) the token is current. Re-run **Test**
  after fixing.
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
- **"Could not save the connection"** — most commonly a missing **Name**, a
  name already in use, or a malformed **Gateway URL** / **SSH host**; the
  error message names the exact violation.
