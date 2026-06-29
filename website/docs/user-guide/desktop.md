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

Follow the [installation instructions for Hermes Desktop](../getting-started/installation.md).

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
- **Composer history and queue editing** — press the up/down arrow keys in an empty composer to recall and reuse previous prompts, and edit messages you've queued up before they're sent.

#### Status bar

The bar along the bottom of the chat shows live session state and exposes quick controls without opening Settings:

- **Per-session YOLO toggle** — flip YOLO on or off for just this session (matching the TUI). YOLO bypasses the dangerous-command approval prompts, so know what you're turning off — see [Security → YOLO Mode](./security.md#yolo-mode).

Chatting against a Hermes instance on another machine instead of the bundled local backend? See [Connecting to a remote backend](#connecting-to-a-remote-backend) below — and for the full picture of how the remote-hosted dashboard connection works (the auth gate, the `/api/ws` chat socket, and WebSocket close-code triage), see [Web Dashboard → Connecting Hermes Desktop to a remote backend](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend).

#### Choosing a model

The model picker lives in the **composer**, just left of the microphone. Click it to switch the model, reasoning effort, and fast mode from one dropdown.

- **The composer picker is sticky UI state and never touches your default.** It's remembered locally (per device) and **follows** across new chats and restarts instead of snapping back to the default — pick a model once and the next `Cmd/Ctrl+N` opens on it. With a live chat, switching models scopes the change to that **current chat**; either way the selection rides along when the session is created/switched and is **never** written to the profile default. (Switching [profiles](#sessions--profiles) reseeds to that profile's own default.)
- **Set the default in Settings → Model.** That "main" model is your **per-profile global default** — it's what new chats, crons, subagents, and auxiliary tasks start from, and it's the only place that writes it. Each [profile](#sessions--profiles) keeps its own default.
- **Per-model effort/fast presets.** Each model remembers its own reasoning effort and fast-mode choice in the desktop app, re-applied to the session whenever you pick that model. These presets are a desktop convenience and don't change crons or subagents.

### File browser

Explore and preview the working directory without leaving the app — useful for following along as the agent reads, writes, and edits files. Set the initial project directory with `hermes desktop --cwd <path>` (or the `HERMES_DESKTOP_CWD` environment variable).

### Voice

Talk to Hermes and hear it back, the same [voice mode](./features/voice-mode.md) available elsewhere. On macOS the OS will prompt once for microphone access.

### Settings & onboarding

Manage providers, models, tools, and credentials from a real UI instead of editing YAML. First-run onboarding gets you to your first message in seconds. The settings panes cover providers/keys, model selection, toolset configuration, MCP servers, the gateway, and session management.

- **Providers settings pane** — a dedicated place to manage inference providers, with an Accounts / API-keys UX for signing in and storing credentials per provider.
- **Every provider and model in the menus** — the GUI surfaces the full provider list and every model that `hermes model` knows about, so you pick from the same catalog the CLI sees rather than a curated subset.
- **xAI Grok OAuth** — Grok is a first-class OAuth provider in the launcher; sign in through the browser flow like the other OAuth providers.
- **Tool-backend installs from the GUI** — run a tool backend's post-setup install steps directly from the app instead of dropping to a terminal.
- **Auxiliary-model warning** — if you switch the main model to a new provider while auxiliary tasks (titling, summarization, and similar helpers) are still pinned to another provider, the app warns you so you don't unknowingly split work across two providers.

First-run onboarding has been redesigned on a unified overlay design system, and you can pick **Choose provider later** to skip provider setup and get into the app first.

### Management panes

The app also surfaces the broader Hermes management surface so you don't have to drop to a terminal:

- **Skills** — browse, install, and manage [skills](./features/skills.md).
- **Cron** — view and manage [scheduled jobs](../reference/cli-commands.md#hermes-cron).
- **Profiles** — switch between [Hermes profiles](./profiles.md) (isolated config/skills/sessions).
- **Messaging** — set up gateway channels.
- **Agents** and **Command Center** — orchestration surfaces for multi-agent work.

### Keyboard & navigation

- **Command palette** — press **Cmd+K** (Ctrl+K on Windows/Linux) to jump to actions and navigate the app from the keyboard.
- **Rebindable shortcuts** — a shortcuts panel in Settings lets you remap the app's keyboard shortcuts to your own keys.
- **Custom zoom shortcuts** — zoom the interface in half-step increments for finer control over text size.
- **UI language switcher** — change the app's interface language in-app, including Simplified Chinese (zh-Hans).

### Sessions & profiles

- **Session-list overhaul** — a reworked session list with archiving and general session hygiene to keep the list manageable as it grows.
- **Search sessions by id** — find a specific session directly by its id.
- **Concurrent multi-profile sessions** — run sessions across multiple [profiles](./profiles.md) at the same time, and reference a session in another profile with cross-profile `@session` links.

## Updating

The app checks for updates in the background and offers a one-click update when one is ready.

The [manual update process](https://hermes-agent.nousresearch.com/docs/getting-started/updating) also works with the GUI.

## Uninstalling

Open **Settings → About → Danger zone** and pick how much to remove:

- **Uninstall Chat GUI only** — removes the desktop app and its data; the Hermes agent, your config, and your chats stay. (Same as `hermes uninstall --gui`.)
- **Uninstall GUI + agent, keep my data** — removes the app and the agent but keeps config, chats, and secrets for a future reinstall. (Same as `hermes uninstall`.)
- **Uninstall everything** — removes the app, the agent, and all user data. (Same as `hermes uninstall --full`.)

The app closes to finish the job (the cleanup runs after it exits so it can remove the running app bundle and its own venv). The agent-removing options are hidden automatically when no local agent is installed (for example, a GUI-only "lite" client connected to a remote backend).

You can do the same from the terminal — `hermes uninstall --gui` for the GUI alone, or `hermes uninstall` / `hermes uninstall --full` for the agent too.

:::note
Running `hermes uninstall --gui` from a **source checkout** (a `hermes desktop` dev build) also removes the workspace `node_modules` and `apps/desktop/{dist,release}` build output, since those are GUI build artifacts. They're recoverable with `hermes desktop` (or `npm install` + a rebuild) — but if you're actively hacking on the desktop app, expect to reinstall dependencies afterward.
:::

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

The packaged app ships the Electron shell and a native React chat surface. On first launch it can install the Hermes Agent runtime into `HERMES_HOME` (`~/.hermes`, or `%LOCALAPPDATA%\hermes` on Windows) — **the same layout a CLI install uses**, which is why the two are interchangeable. Backend resolution first honours `HERMES_DESKTOP_HERMES_ROOT`, then a completed managed install, then a probed `hermes` on `PATH` (unless `--ignore-existing` / `HERMES_DESKTOP_IGNORE_EXISTING=1` is set), and finally an explicit `HERMES_DESKTOP_HERMES` command override for packagers such as Nix. The React renderer talks to a headless backend the app launches for you — a `hermes serve` process that serves the `tui_gateway` JSON-RPC/WebSocket API — and reuses the agent runtime rather than embedding `hermes --tui`. The desktop app is **self-contained**: it runs its own `hermes serve` backend and never launches or requires the [web dashboard](./features/web-dashboard.md). Install, backend-resolution, and self-update logic live in the Electron main process.

## Connecting to a remote backend

By default the app starts and manages its own **local** backend. You can instead point it at a Hermes backend running on another machine — a VPS, a home server, or a Mini behind Tailscale.

:::info The remote backend is a running `hermes serve` process
"Remote backend" means a **`hermes serve`** server running on the remote machine — that is the process the desktop app connects to. Nothing in this section works unless that backend is actually up and reachable. The desktop app does not start it for you; you (or a `systemd` service) keep `hermes serve` running on the remote host, and the app attaches to it. If you also use messaging channels (Telegram, Discord, etc.), the **gateway** is a *separate* long-running process you start independently — see the note after the setup steps.
:::

The connection has two halves: on the backend you protect it with an **auth provider**, and in the app you enter the backend's URL and sign in. Binding the backend to a non-loopback address automatically engages its auth gate, and the provider you configure is what lets the desktop app through.

**Pick a provider based on where the backend lives:**

- **OAuth (Nous Portal) — preferred for anything reachable beyond your own machine.** Logins are verified against your Nous account, so this is the option suitable for a VPS, a public host, or any remote backend. Register the dashboard with `hermes dashboard register` (or the Portal [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards) page) to provision its OAuth client, then sign in from the app with **Sign in with Nous Research**. A self-hosted OIDC provider works the same way if you run your own identity provider.
- **Username/password — local / trusted-network use only.** The simplest option when the backend is on the same trusted LAN or reachable only over a VPN (e.g. Tailscale). It protects a single shared credential with no external identity provider, so **do not use it for a dashboard exposed to the public internet** — reach for OAuth there instead.

The rest of this section shows the username/password path because it's the quickest to stand up on a trusted network; for the OAuth path see [Web Dashboard → Default provider: Nous Research](./features/web-dashboard.md#default-provider-nous-research).

### On the backend (the remote machine)

Set a username and password, then start the backend bound to a reachable address. The credentials live in `~/.hermes/.env` (the secrets file, mode 0600):

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

# 2. Run the backend bound to a reachable address. The non-loopback bind
#    engages the auth gate; the username/password provider handles login.
hermes serve --host 0.0.0.0 --port 9119
```

Keep that `hermes serve` process running for as long as you want the desktop app to be able to connect — if it stops, the app can no longer reach the backend. Run it under `systemd`, `tmux`, or your process manager of choice so it survives logout and reboots.

Separately, make sure the **gateway is running** on the remote host if you rely on messaging channels — the `hermes serve` backend is what the desktop app talks to, but your Telegram/Discord/Slack gateway sessions are a different process that you start and keep running on their own. See [Messaging](./messaging/index.md) for gateway setup.

Prefer not to keep a plaintext password at rest? Set `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` to a scrypt hash instead — compute it with `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`. Full configuration surface (config.yaml keys, every env var, the rate limiter): [Web Dashboard → Username/password provider](./features/web-dashboard.md#usernamepassword-provider-no-oauth-idp).

Running the backend as a systemd service? Give the unit `EnvironmentFile=%h/.hermes/.env` so the credentials are in the environment at boot.

:::warning
The backend reads and writes your `.env` (API keys, secrets) and can run agent commands. The **username/password** setup shown above is for a trusted network — never expose a password-protected backend directly to the open internet; put it behind a VPN. [Tailscale](https://tailscale.com/) is the clean option: bind to the machine's tailscale IP (`--host <tailscale-ip>`) and use `http://<tailscale-ip>:9119` as the Remote URL so only your tailnet can reach it. To reach a backend over the public internet, use the **OAuth (Nous Portal)** provider instead.
:::

### In the app

**Settings → Gateway → Remote gateway:**

1. **Remote URL** — `http://<backend-host>:9119` (path prefixes like `/hermes` work if you front it with a reverse proxy)
2. **Sign in** — the app detects which provider the backend advertises and adapts the button. For a username/password backend it shows a **Sign in** button that opens a credential form (enter the credentials from step 1). For an OAuth backend it shows **Sign in with `<provider>`** (e.g. *Sign in with Nous Research*), which runs the provider's browser sign-in. Either way the app ends up with an authenticated session against the backend.
3. **Save and reconnect** — switches the desktop shell onto the remote backend. The session refreshes automatically; you stay signed in across restarts when `HERMES_DASHBOARD_BASIC_AUTH_SECRET` is set.

You can also set the backend URL without the UI via the `HERMES_DESKTOP_REMOTE_URL` environment variable before launching the app (it overrides the in-app setting); you still sign in from the Gateway settings panel.

:::note Per-profile remote hosts
The remote gateway host is configured per [profile](./profiles.md), so each profile can point at its own remote backend (or stay on its local one). Switching profiles switches which remote host the app connects to.
:::

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

### "Build desktop app" stuck on Electron download

The build downloads the Electron runtime (~114&nbsp;MB) from `github.com/electron/electron/releases`. If the installer hangs on the **Build desktop app** step with the live output repeating `retrying attempt=…`, GitHub is being blocked or throttled on your network (firewall, proxy, or region).

The installer self-heals this automatically: on a failed build it (1) clears a corrupt cached Electron zip and retries, then (2) if it still fails and you haven't set `ELECTRON_MIRROR`, retries once more through `npmmirror.com`, the de-facto Electron community mirror. `@electron/get` SHASUM-checks the download, but the checksums come from the same mirror — that catches a corrupt or partial download, not a compromised mirror. If you'd rather not trust a third-party host, pin your own `ELECTRON_MIRROR` (below); the build never overrides one you've set.

To **choose your own mirror** (e.g. a corporate/trusted one), set `ELECTRON_MIRROR` before installing or rebuild manually — the build honors it and won't override it:

```bash
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ \
  bash -c 'cd "$HOME/.hermes/hermes-agent/apps/desktop" && CSC_IDENTITY_AUTO_DISCOVERY=false npm run pack'
```

To clear a corrupt cached zip by hand:

```bash
rm -f "$HOME/Library/Caches/electron"/electron-*.zip   # macOS
rm -f "$HOME/.cache/electron"/electron-*.zip            # Linux
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
- [TUI](./tui.md) — the modern terminal UI used by `hermes --tui` and the dashboard chat tab
- [Web Dashboard](./features/web-dashboard.md) — browser admin panel with an embedded chat tab
- [Configuration](./configuration.md) — config that the desktop app reads and writes
- [Windows (Native)](./windows-native.md) — native Windows install path
