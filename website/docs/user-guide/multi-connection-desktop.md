---
sidebar_position: 5
---

# Connecting Desktop to Many Hermes Instances

Register every Hermes backend you own — the local runtime, remote gateways on
your LAN or VPS, SSH hosts, and Hermes Cloud instances — in one desktop app,
and use the agents on all of them side by side. Connections are persistent:
each registered gateway dials its own backends and WebSockets on demand, and
background agents keep streaming while you look at another gateway.

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

## The gateway registry

The **Registered gateways** section of **Settings → Gateways** manages a named
list of Hermes gateways. Its intro says it plainly: *"Manage this device and
every Hermes gateway it can reach through remote, SSH, or Cloud connections."*
Each entry is a *connection*:

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
- The **local** entry is managed by the app (it wears an **App-managed** pill)
  and cannot be removed. Removing any other connection tears down its live
  backends and tunnels; the instance itself is untouched.
- One connection is always the **Primary** (pill on its row): it is the
  registry fallback for multi-gateway calls that do not name a gateway.
  **Make primary** does not switch the current Sessions workspace; removing
  the primary falls back to the local entry.
- **At startup, return to Sessions on the last-used gateway** controls which
  gateway Sessions opens after a full app restart. It is off by default, so
  Sessions opens on **Primary**. Turn it on to resume the most recent gateway
  that connected successfully. A failed switch is never remembered, and a
  removed or unavailable saved gateway falls back to Primary.
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

Switch gateways from the **Sessions** sidebar. Profiles, chats, messaging, and
cron stay scoped to that gateway; the app-managed window backend is still chosen
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

## Agents across gateways

Every [profile](./profiles.md) on every registered connection is an *agent*.
The union roster is what multi-gateway surfaces (and the built-in
[Bot Mode](./bot-mode.md) roster) render:

- When the same profile name exists on several gateways, handles disambiguate
  as **`@name-device`** — `research` on your Homelab renders as
  `@research-homelab`, while a profile unique across all gateways keeps its
  bare name.
- Enumeration is eager but sockets are lazy: the app lists agents over REST
  without dialing every gateway's WebSocket. An unreachable gateway reports
  per-row instead of breaking the roster; SSH connections stay connect-on-demand
  until you first open an agent on them (no surprise tunnels).
- Opening an agent dials **its own gateway** — chats, sessions, and memory
  live on the machine that owns the profile, exactly as if you were using
  that instance directly.

Each `(connection, profile)` pair gets its own backend and socket, pooled
with the same idle-reaping as local per-profile backends — background agents
keep streaming while you look at another gateway.

### Switching and scoping

The sidebar foot follows one hierarchy: **gateway → profile → sessions**.
Gateways are machines or hosted backends; profiles are isolated Hermes agents
that live on one gateway.

- With one registered gateway, no gateway control is added. Local-only Desktop
  keeps the same profile rail and keyboard flow as before.
- With several gateways, the sidebar shows one named gateway selector. Its device,
  cloud, network, or terminal icon identifies the connection type; profile
  avatars remain a separate control after the divider. The same selector scales
  from two gateways to a larger fleet without turning backends into profile-like
  glyphs or crowding profile actions out of the rail.
- Selecting a gateway restores the last profile used there. The home pill
  returns to its default profile and the layers pill shows **All profiles on
  this gateway**. **Cmd/Ctrl+1–9** continue to switch profiles within the
  active gateway.
- With several gateways the profile rail is a **fleet rail**: every registered
  gateway's profiles sit on the one strip, each group headed by that gateway's
  kind glyph (device, network, terminal, cloud) — the same glyph the gateway
  selector uses. The active gateway's squares look exactly as they do on a
  single-gateway Desktop; the other gateways' squares are dimmed ("at rest").
  Hovering an at-rest square names its machine (`omer · This device`), so two
  same-named profiles on different machines never read alike.
- Clicking an at-rest square performs the same switch as the gateway selector,
  landing on that exact `(gateway, profile)`: the square spins while the
  target is dialed, the previous gateway stays painted until the target
  answers, and a dead target fails the click with a message rather than
  leaving the window half-switched. Groups keep registry order whichever
  gateway is active, so a square never moves under the pointer that clicked
  it. Right-click on an at-rest square offers **Switch to**, **Color**,
  **Rename**, **Edit SOUL.md** and **Delete**, all executed on the square's
  own gateway; the delete confirmation names the machine.
- A gateway the last enumeration could not reach keeps its squares, marked
  with an amber dot on its glyph — a sleeping box is still yours. Two
  registrations of one backend collapse to a single group. Past thirteen
  squares across the fleet, the strip condenses into one menu sectioned by
  gateway.
- The selected gateway survives a quit and relaunch only when **Settings →
  Gateways → At startup, return to Sessions on the last-used gateway** is on.
  The preference and gateway id live in the app's user-data registry, so
  replacing or updating the application bundle does not reset them.
- With more than thirteen profiles on the active gateway, their avatar strip
  condenses into a named profile selector. Large gateway and profile sets can
  therefore coexist without changing the **gateway → profile → sessions** model.
- **This device** remains a first-class gateway even when a remote connection is
  Primary. It can keep local sessions available during a remote outage, but the
  app does not call it "offline mode": the selected model or tools may still
  require internet access.
- The session list, messaging channels, cron jobs, settings, files, and memory
  are all scoped to the active `(gateway, profile)`. Switching from a Telegram
  gateway to a Signal gateway cannot leave the previous gateway's channel groups
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

Add, test, rename, or remove gateways in **Settings → Gateways**. The plug
button beside the profile actions is a shortcut to that single management
home, not a second add flow.

### Sessions and Bot Mode

Sessions intentionally show one active gateway at a time: this keeps files,
tools, channels, cron, and session history in one understandable execution
context. The fleet profile rail widens only the *picker* — the workspace still
lives on exactly one `(gateway, profile)` after every click. Bot Mode serves a
different job and may present the union roster, grouped by gateway, so a user
can open one agent on a NAS and another on a VPS from one surface. Opening a
bot still activates its exact `(gateway, profile)` route.

Direct bot mentions and delegation remain gateway-local by default. Crossing a
backend boundary changes filesystem, credentials, tools, and trust context, so
cross-gateway execution should be an explicit bridge rather than an accidental
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

You rarely need the Settings button, though: once more than one update target
exists, the app's regular update affordances (**Update now** on the About
panel, ⌘K **Update Hermes**, the update-ready toast) run the same fan-out
automatically — active backend first, then every other eligible gateway, then
the desktop app itself last. See
[Updating](./desktop.md#updating) in the desktop guide.

## Security notes

- **Where tokens live.** Remote-gateway session tokens (and native sign-in
  OAuth tokens, keyed by gateway base URL) are stored in the app's user-data
  directory as owner-only (0600) files, in the Electron main process; the
  renderer and plugins never see token bytes.
- **Optional keychain encryption.** By default the tokens are **not** run
  through the OS keychain — on macOS in particular, Electron's `safeStorage`
  parks a per-app key in the login keychain, and a locked or broken keychain
  turns that into a password prompt on every launch. If you want at-rest
  encryption on top of the file permissions, turn on **Settings → Gateway →
  "Encrypt saved secrets with the OS keychain"**; existing stored secrets are
  re-encrypted in place (Keychain on macOS, DPAPI on Windows, the session
  keyring backend on Linux). Turning it back off decrypts them again.
- **The registry file** (`connections.json` under the app's user-data
  directory) holds labels, URLs, and hosts — secrets only ever appear inside
  encrypted envelopes.
- The plugin SDK's `host.connections()` deliberately returns labels, kinds,
  and the primary id — never token material.

## For plugin authors

The Desktop [plugin SDK](../developer-guide/desktop-plugin-sdk.md) exposes the
multi-gateway surface directly:

- `host.connections()` — the registered connection list (labels, kinds,
  primary; never token bytes).
- `host.agents()` — the union roster: one row per `(gateway, profile)` with
  the precomputed `@name-device` handle.
- `host.ensureAgent(connectionId, profile)` — activate an agent's gateway so
  subsequent `host.request` calls hit its backend.
- `host.warmAgent(connectionId, profile)` — fire-and-forget socket pre-warm
  (hover-intent).

All four are feature-detected: on an older Desktop build they're absent and a
plugin should fall back to the single-gateway `profiles.list` flow. Bot Mode's
multi-gateway roster is the reference consumer.

## Troubleshooting

- **"Connection test failed"** — the backend isn't reachable at that URL from
  this machine. Check that `hermes serve` is running on the remote host, the
  port is open, and (for token auth) the token is current. Re-run **Test**
  after fixing.
- **An agent shows but won't open** — run **Test** on its connection. The
  WebSocket leg failing while HTTP passes usually means a proxy, firewall, or
  gateway auth/origin guard is blocking `/api/ws`.
- **A remote gateway is missing from the roster** — its backend is down or
  unreachable; the roster lists it under gateways with the error. SSH connections
  show *connect-on-demand* until first use — that's by design, not a failure.
- **"Update Hermes Desktop to chat with agents on other connections"** — the
  app predates the multi-connection stack; update the desktop app itself.
- **Duplicate device names** — not possible; names are enforced unique at
  save time. If a migrated name collided, it was suffixed (`Homelab 2`).
- **"Could not save the connection"** — most commonly a missing **Name**, a
  name already in use, or a malformed **Gateway URL** / **SSH host**; the
  error message names the exact violation.
