---
sidebar_position: 3
title: "Desktop App"
description: "The native Hermes desktop app — a polished experience for chatting with Hermes, with streaming tool output, side-by-side previews, a file browser, voice, cron, profiles, skills, and settings. macOS, Windows, and Linux."
---

# Desktop App

The Hermes desktop app is a native app built around the **same** agent you get from the CLI and the gateway — same config, same API keys, same sessions, same skills, same memory. It is not a separate product or a lightweight clone; it uses the same Hermes Agent core and settings, and drives it through a modern & thoughtfully designed UI. If you have used `hermes` in a terminal, everything you set up there is already here, and anything you do here shows up there.

It runs on **macOS, Windows, and Linux**.

:::tip Which interface is which?
Hermes has several front ends that all talk to the same agent:

- **Desktop App** (this page) — a native application with a purpose-built UI for chat, configuration, and management.
- **CLI** (`hermes`) and **[TUI](./tui.md)** (`hermes --tui`) — terminal interfaces.
- **[Web Dashboard](./features/web-dashboard.md)** (`hermes dashboard`) — a browser admin panel; its optional **Chat** tab embeds the TUI through a pseudo-terminal.

Pick whichever fits the moment. They share state, so you can start a session in one and resume it in another.
:::

## Install

### With the Hermes Desktop installer on MacOS or Windows (recommended)

[Download the Hermes Desktop installer](https://hermes-agent.nousresearch.com/desktop) from our website and run it.

### With the CLI installer on Linux, MacOS, or Windows

Add `--include-desktop` to the regular install script.

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --include-desktop
```

### With an existing Hermes installation

If you already have Hermes installed, simply run

```bash
hermes desktop
```

That uses your current config, keys, sessions, and skills.

## What's in the app

The desktop app is organized as a chat-first window with a left sidebar for navigation. It's built to allow managing multiple simultaneous agent conversations, configuring messaging providers, creating artifacts, browsing projects' folder structures, and working on multiple projects at once.

### Chat

The center of the app. You get:

- **Streaming responses** with live tool activity and structured tool-call summaries as the agent works.
- **The same conversation history** as every other Hermes surface — sessions started here resume in the CLI/TUI and vice versa.
- **Drag-and-drop files** anywhere in the chat area to attach them to your next message.
- **A right-hand preview rail** — render web pages, files, and tool outputs side by side while you keep chatting.

Chatting against a Hermes instance on another machine instead of the bundled local backend? See [Connecting to a remote backend](#connecting-to-a-remote-backend) below — and for the full picture of how the remote-hosted dashboard connection works (the `/api/ws` chat socket, session-token pinning, and WebSocket close-code triage), see [Web Dashboard → Connecting Hermes Desktop to a remote backend](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend).

### File browser

Explore and preview the working directory without leaving the app — useful for following along as the agent reads, writes, and edits files. Set the initial project directory with `hermes desktop --cwd <path>` (or the `HERMES_DESKTOP_CWD` environment variable).

### Voice

Talk to Hermes and hear it back, the same [voice mode](./features/voice-mode.md) available elsewhere. On macOS the OS will prompt once for microphone access.

### Settings & onboarding

Manage providers, models, tools, and credentials from a real UI instead of editing YAML. First-run onboarding gets you to your first message in seconds. The settings panes cover providers/keys, model selection, toolset configuration, MCP servers, the gateway, and session management.

### Management panes

The app also surfaces the broader Hermes management surface so you don't have to drop to a terminal:

- **Skills** — browse, install, and manage [skills](./features/skills.md).
- **Cron** — view and manage [scheduled jobs](../reference/cli-commands.md#hermes-cron).
- **Profiles** — switch between [Hermes profiles](./profiles.md) (isolated config/skills/sessions).
- **Messaging** — set up gateway channels.
- **Agents** and **Command Center** — orchestration surfaces for multi-agent work.

## Updating

The app checks for updates in the background and offers a one-click update when one is ready.

The [manual update process](https://hermes-agent.nousresearch.com/docs/getting-started/updating) also works with the GUI.

## CLI reference: `hermes desktop`

To launch via the CLI, simply run `hermes desktop`. By default it installs workspace Node dependencies, builds the current OS's unpacked Electron app, then launches that packaged artifact.

| Flag                 | Description                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `--skip-build`       | Skip npm install/package and launch the existing unpacked app from `apps/desktop/release` |
| `--force-build`      | Force a full rebuild even if the content stamp matches                                    |
| `--build-only`       | Build the desktop app but do not launch it (used by `hermes update`)                      |
| `--source`           | Launch via `electron .` against `apps/desktop/dist` instead of the packaged app           |
| `--cwd PATH`         | Initial project directory for desktop chat sessions (sets `HERMES_DESKTOP_CWD`)           |
| `--hermes-root PATH` | Override the Hermes source root the app uses (sets `HERMES_DESKTOP_HERMES_ROOT`)          |
| `--ignore-existing`  | Force the app to ignore any `hermes` CLI already on `PATH` during backend resolution      |
| `--fake-boot`        | Enable deterministic boot delays for validating the startup UI                            |

## How it works

The packaged app ships only the Electron shell. On first launch it installs the Hermes Agent runtime into `HERMES_HOME` (`~/.hermes`, or `%LOCALAPPDATA%\hermes` on Windows) — **the same layout a CLI install uses**, which is why the two are interchangeable. The React renderer talks to a `hermes dashboard` backend over the standard gateway APIs and reuses the agent rather than reimplementing it. Install, backend-resolution, and self-update logic live in the Electron main process.

## Connecting to a remote backend

By default the app starts and manages its own **local** backend. You can instead point it at a Hermes backend running on another machine — a VPS, a home server, or a Mini behind Tailscale — under **Settings → Gateway → Remote gateway**. It asks for two things:

- **Remote URL** — the backend's dashboard URL, e.g. `http://<host>:9119`
- **Session token** — the backend's dashboard session token

The session token is the part that trips people up. **Hermes does not print it for you to copy** — by default the backend mints a fresh random token on every boot and injects it straight into the served HTML, so there is nothing in `config.yaml`, in `/gateway`, or in the logs to grab. For a remote connection you pin the token yourself on the backend, then paste that same value into the app.

### On the backend (the remote machine)

```bash
# 1. Mint a stable token and store it in ~/.hermes/.env (secrets file, 0600).
#    Without HERMES_DASHBOARD_SESSION_TOKEN the token is random per boot and
#    uncopyable; setting it pins the value the desktop app will use.
TOKEN=$(openssl rand -base64 32)
echo "HERMES_DASHBOARD_SESSION_TOKEN=$TOKEN" >> ~/.hermes/.env
chmod 600 ~/.hermes/.env
echo "$TOKEN"   # copy this value into the desktop app

# 2. Run the dashboard bound to a reachable address.
#    --insecure is required for any non-loopback bind and keeps the legacy
#    session-token auth path (a non-loopback bind WITHOUT --insecure engages
#    the OAuth gate, which ignores the session token).
hermes dashboard --no-open --insecure --host 0.0.0.0 --port 9119
```

Running the dashboard as a systemd service? Give the unit `EnvironmentFile=%h/.hermes/.env` so the token is in the environment at boot.

:::warning
`--insecure` exposes a port that reads/writes your `.env` (API keys, secrets) and can run agent commands. Never expose it to the open internet — put it behind a VPN. [Tailscale](https://tailscale.com/) is the clean option: bind to the machine's tailscale IP (`--host <tailscale-ip>`) and use `http://<tailscale-ip>:9119` as the Remote URL so only your tailnet can reach it.
:::

### In the app

**Settings → Gateway → Remote gateway:**

1. **Remote URL** — `http://<backend-host>:9119` (path prefixes like `/hermes` work if you front it with a reverse proxy)
2. **Session token** — paste the `$TOKEN` value from step 1
3. **Test remote** — confirms the backend is reachable and the token is accepted
4. **Save and reconnect** — switches the desktop shell onto the remote backend

The token is stored encrypted in the app's local config; leave the field blank on a later edit to keep the saved one. You can also set it without the UI via the `HERMES_DESKTOP_REMOTE_URL` + `HERMES_DESKTOP_REMOTE_TOKEN` environment variables before launching the app (both must be set together; they override the in-app settings).

### Troubleshooting

- **Test fails with 401** — the token doesn't match the backend's `HERMES_DASHBOARD_SESSION_TOKEN`, or the backend is bound non-loopback *without* `--insecure` (OAuth gate is on, ignoring the token). Verify with `curl -s -H "X-Hermes-Session-Token: $TOKEN" http://<host>:9119/api/status` — that should return JSON, not a 401.
- **Connection refused / times out** — the backend bound to `127.0.0.1` (the default) or a firewall/VPN is blocking the port. Bind to `0.0.0.0` or the tailscale IP and open the port to your trusted network.
- **No token to copy** — expected. You mint it yourself; Hermes never surfaces the default ephemeral one.

For the same setup from the web-dashboard angle, see [Web Dashboard → Connecting Hermes Desktop to a remote backend](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend); the env vars are catalogued under [Environment Variables → Web Dashboard & Hermes Desktop](../reference/environment-variables.md#web-dashboard--hermes-desktop).

## Troubleshooting

Boot logs land in `HERMES_HOME/logs/desktop.log` (it includes backend output and recent Python tracebacks) — check it first if the app reports a boot failure. You can also tail it from the CLI:

```bash
hermes logs gui -f
```

Common resets:

```bash
# Force a clean first-launch setup (macOS/Linux)
rm "$HOME/.hermes/hermes-agent/.hermes-bootstrap-complete"

# Rebuild a broken Python venv (macOS/Linux)
rm -rf "$HOME/.hermes/hermes-agent/venv"

# Reset a stuck macOS microphone prompt
tccutil reset Microphone com.nousresearch.hermes
```

## Building from source

If you want to hack on the app itself, install workspace deps from the repo root once, then run the dev server from `apps/desktop`:

```bash
npm install          # from repo root — links apps/desktop, web, apps/shared
cd apps/desktop
npm run dev          # Vite renderer + Electron, which boots the Python backend
```

Point the app at a specific checkout, or sandbox it from your real config:

```bash
HERMES_DESKTOP_HERMES_ROOT=/path/to/clone npm run dev
HERMES_HOME=/tmp/throwaway npm run dev
npm run dev:fake-boot   # exercise the startup overlay with deterministic delays
```

Build installers:

```bash
npm run dist:mac     # DMG + zip
npm run dist:win     # NSIS + MSI
npm run dist:linux   # AppImage + deb + rpm
npm run pack         # unpacked app under release/ (no installer)
```

macOS/Windows signing and notarization run automatically when the relevant credentials are present in the environment (`CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*` for macOS, `WIN_CSC_*` for Windows).

## See also

- [CLI Guide](./cli.md) — the terminal interface
- [TUI](./tui.md) — the modern terminal UI the desktop backend reuses
- [Web Dashboard](./features/web-dashboard.md) — browser admin panel with an embedded chat tab
- [Configuration](./configuration.md) — config that the desktop app reads and writes
- [Windows (Native)](./windows-native.md) — native Windows install path
