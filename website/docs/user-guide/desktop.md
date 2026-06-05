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

Chatting against a Hermes instance on another machine instead of the bundled local backend? See [Connecting to a remote backend](#connecting-to-a-remote-backend) below — and for the full picture of how the remote-hosted dashboard connection works (the auth gate, the `/api/ws` chat socket, and WebSocket close-code triage), see [Web Dashboard → Connecting Hermes Desktop to a remote backend](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend).

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

By default the app starts and manages its own **local** backend. You can instead point it at a Hermes backend running on another machine — a VPS, a home server, or a Mini behind Tailscale.

:::info The remote backend is a running `hermes dashboard` process
"Remote backend" means a **`hermes dashboard`** server running on the remote machine — that is the process the desktop app connects to. Nothing in this section works unless that dashboard is actually up and reachable. The desktop app does not start it for you; you (or a `systemd` service) keep `hermes dashboard` running on the remote host, and the app attaches to it. If you also use messaging channels (Telegram, Discord, etc.), the **gateway** is a *separate* long-running process you start independently — see the note after the setup steps.
:::

The connection has two halves: on the backend you protect the dashboard with a **username and password**, and in the app you enter the backend's URL and sign in with those credentials. Binding the dashboard to a non-loopback address automatically engages its auth gate, so the username/password provider is what lets the desktop app through.

### On the backend (the remote machine)

Set a username and password, then start the dashboard bound to a reachable address. The credentials live in `~/.hermes/.env` (the secrets file, mode 0600):

```bash
# 1. Set the dashboard login credentials.
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
# Recommended: a stable signing secret so sessions survive restarts.
# Without it a random key is generated per boot and you'll be logged out
# on every restart.
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env

# 2. Run the dashboard bound to a reachable address. The non-loopback bind
#    engages the auth gate; the username/password provider handles login.
hermes dashboard --no-open --host 0.0.0.0 --port 9119
```

Keep that `hermes dashboard` process running for as long as you want the desktop app to be able to connect — if it stops, the app can no longer reach the backend. Run it under `systemd`, `tmux`, or your process manager of choice so it survives logout and reboots.

Separately, make sure the **gateway is running** on the remote host if you rely on messaging channels — the dashboard backend is what the desktop app talks to, but your Telegram/Discord/Slack gateway sessions are a different process that you start and keep running on their own. See [Messaging](./messaging/index.md) for gateway setup.

Prefer not to keep a plaintext password at rest? Set `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` to a scrypt hash instead — compute it with `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`. Full configuration surface (config.yaml keys, every env var, the rate limiter): [Web Dashboard → Username/password provider](./features/web-dashboard.md#usernamepassword-provider-no-oauth-idp).

Running the dashboard as a systemd service? Give the unit `EnvironmentFile=%h/.hermes/.env` so the credentials are in the environment at boot.

:::warning
The dashboard reads and writes your `.env` (API keys, secrets) and can run agent commands. Even behind a username and password, never expose it directly to the open internet — put it behind a VPN. [Tailscale](https://tailscale.com/) is the clean option: bind to the machine's tailscale IP (`--host <tailscale-ip>`) and use `http://<tailscale-ip>:9119` as the Remote URL so only your tailnet can reach it.
:::

### In the app

**Settings → Gateway → Remote gateway:**

1. **Remote URL** — `http://<backend-host>:9119` (path prefixes like `/hermes` work if you front it with a reverse proxy)
2. **Sign in** — the app detects that the backend requires a username and password and shows a **Sign in** button. Click it, enter the credentials from step 1, and the app authenticates against the backend's login page.
3. **Save and reconnect** — switches the desktop shell onto the remote backend. The session refreshes automatically; you stay signed in across restarts when `HERMES_DASHBOARD_BASIC_AUTH_SECRET` is set.

You can also set the backend URL without the UI via the `HERMES_DESKTOP_REMOTE_URL` environment variable before launching the app (it overrides the in-app setting); you still sign in with your username and password from the Gateway settings panel.

### Troubleshooting

- **Sign-in fails with 401 / "Invalid credentials"** — the username or password doesn't match the backend's `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`. The backend returns the same generic error for an unknown user and a wrong password (no enumeration oracle), so double-check both. Confirm the gate is on with `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'` — it should report `true` and include `"basic"`.
- **No "Sign in" button — it asks for a session token instead** — the backend's username/password provider isn't active. `/api/status` won't list `"basic"` in `auth_providers`. Make sure both the username and a password (or password hash) are set in `~/.hermes/.env` and that the dashboard process actually loaded them.
- **Signed out on every restart** — set `HERMES_DASHBOARD_BASIC_AUTH_SECRET` to a stable value. Without it the token-signing key is regenerated per boot, invalidating all sessions.
- **Connection refused / times out** — the backend bound to `127.0.0.1` (the default) or a firewall/VPN is blocking the port. Bind to `0.0.0.0` or the tailscale IP and open the port to your trusted network.

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
