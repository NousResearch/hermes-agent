---
sidebar_position: 3
title: "Desktop App"
description: "The native Hermes desktop app — a polished Electron window for chatting with the full agent, with streaming tool output, side-by-side previews, a file browser, voice, cron, profiles, skills, and settings. macOS, Windows, and Linux."
---

# Desktop App

The Hermes desktop app is a native window around the **same** agent you get from the CLI and the gateway — same config, same API keys, same sessions, same skills, same memory. It is not a separate product or a lightweight clone; it installs the standard Hermes Agent runtime into the standard `~/.hermes` layout and drives it through a real UI. If you have used `hermes` in a terminal, everything you set up there is already here, and anything you do here shows up there.

It's built on Electron and runs on **macOS, Windows, and Linux**.

:::tip Which interface is which?
Hermes has several front ends that all talk to the same agent:

- **CLI** (`hermes`) and **[TUI](./tui.md)** (`hermes --tui`) — terminal interfaces.
- **[Web Dashboard](./features/web-dashboard.md)** (`hermes dashboard`) — a browser admin panel; its optional **Chat** tab embeds the TUI through a pseudo-terminal.
- **Desktop App** (this page) — a native window with a purpose-built React UI for chat, previews, and management.

Pick whichever fits the moment. They share state, so you can start a session in one and resume it in another.
:::

## Install

### With the Hermes installer (recommended)

Add `--include-desktop` to the one-line installer and it provisions the agent and builds the desktop app in a single pass:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --include-desktop
```

Already have the Hermes CLI? Build and launch against your existing install:

```bash
hermes desktop
```

That uses your current config, keys, sessions, and skills. On first launch the app walks you through picking a provider and model; there's nothing else to configure.

### Prebuilt installers

When a release ships desktop installers they're attached to the [releases page](https://github.com/NousResearch/hermes-agent/releases/latest):

| Platform | Artifacts |
|----------|-----------|
| macOS | `.dmg` (signed + notarized) |
| Windows | `.exe` (NSIS) / `.msi` |
| Linux | `.AppImage` / `.deb` / `.rpm` |

These are published manually, so the install-with-Hermes path above is the most reliable way to get the latest build.

### Windows GUI installer

On Windows there's also a thin GUI installer: download **Hermes Desktop**, run the `.exe`, and on first launch it calls `install.ps1` under the hood to provision a bundled Python (via `uv`), a portable Git, ripgrep, and the rest of the dependencies — no admin rights or system changes required. The desktop app and a PowerShell-installed CLI share the same install and data directories, so you can use either or both. See the [Windows (Native) guide](./windows-native.md#desktop-installer-alternative) for details.

## Requirements

The installer handles the toolchain for you (Python 3.11+, a portable Git, ripgrep). What's worth knowing:

- **Windows** — the installer bundles its own Git and Python; no admin rights needed.
- **macOS / Linux** — uses your system Python 3.11+, installed automatically if missing.

## What's in the app

The desktop app is organized as a chat-first window with a left sidebar for navigation. The headline surface is the conversation; the rest are management panes that mirror what you'd otherwise do through `hermes` subcommands or the web dashboard.

### Chat

The center of the app. You get:

- **Streaming responses** with live tool activity and structured tool-call summaries as the agent works.
- **The same conversation history** as every other Hermes surface — sessions started here resume in the CLI/TUI and vice versa.
- **Drag-and-drop files** anywhere in the chat area to attach them to your next message.
- **A right-hand preview rail** — render web pages, files, and tool outputs side by side while you keep chatting.
- **A model picker** for switching models mid-session without leaving the window.

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

The app checks for updates in the background and offers a one-click update when one is ready. You can also update any time from the CLI — this pulls the latest agent and rebuilds the app in place:

```bash
hermes update
```

## CLI reference: `hermes desktop`

The canonical command is `hermes desktop` (the older `hermes gui` is kept as a deprecated alias). By default it installs workspace Node dependencies, builds the current OS's unpacked Electron app, then launches that packaged artifact.

| Flag | Description |
|------|-------------|
| `--skip-build` | Skip npm install/package and launch the existing unpacked app from `apps/desktop/release` |
| `--source` | Launch via `electron .` against `apps/desktop/dist` instead of the packaged app |
| `--build-only` | Build the desktop app but do not launch it (used by the installer's `--update` flow) |
| `--cwd PATH` | Initial project directory for desktop chat sessions (sets `HERMES_DESKTOP_CWD`) |
| `--hermes-root PATH` | Override the Hermes source root the app uses (sets `HERMES_DESKTOP_HERMES_ROOT`) |
| `--ignore-existing` | Force the app to ignore any `hermes` CLI already on `PATH` during backend resolution |
| `--fake-boot` | Enable deterministic boot delays for validating the startup UI |

## How it works

The packaged app ships only the Electron shell. On first launch it installs the Hermes Agent runtime into `HERMES_HOME` (`~/.hermes`, or `%LOCALAPPDATA%\hermes` on Windows) — **the same layout a CLI install uses**, which is why the two are interchangeable. The React renderer talks to a `hermes dashboard --tui` backend over the standard gateway APIs and reuses the agent rather than reimplementing it. Install, backend-resolution, and self-update logic live in the Electron main process.

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
