---
sidebar_position: 4
title: "Hermes Webapp"
description: "The planned browser-native Hermes workspace: chat, sessions, files, previews, and per-chat browser workspaces without replacing Hermes Dashboard or Hermes Desktop."
---

# Hermes Webapp

Hermes Webapp is the browser-native workspace surface for Hermes Agent. It is not a replacement for Hermes Desktop and it is not just a rename of the Web Dashboard.

Today, `hermes webapp` starts the same browser UI bundle as `hermes dashboard` while the dedicated workspace surface is built out. The separate command is intentional: it gives the portable browser workspace a stable product boundary without breaking existing Dashboard users.

```bash
hermes webapp
```

## Which surface is which?

- **Hermes Dashboard** (`hermes dashboard`) — browser management UI for config, models, keys, sessions, logs, cron, skills, plugins, and profile administration. Its Chat tab embeds the TUI over a pseudo-terminal.
- **Hermes Desktop** (`hermes desktop`) — native Electron app. It can use Electron-only APIs such as native file dialogs, OS integration, packaged app updates, and Electron webviews.
- **Hermes Webapp** (`hermes webapp`) — portable browser workspace for chat, active sessions, files, previews, and future per-chat browser/RPA work. It should work from any modern browser when the user intentionally exposes it to that device.
- **Hermes Serve** (`hermes serve`) — headless backend used by Desktop, Webapp, and other clients.

## Network scope

Hermes Webapp follows the same safety model as the Dashboard server: it is local by default.

| Scope | What it means | Typical command |
| --- | --- | --- |
| Localhost | Only this computer can use it. Safest default. | `hermes webapp` |
| LAN | Devices on the local network can reach it. Requires deliberate bind/auth. | `hermes webapp --host 0.0.0.0` |
| Tailscale | Devices in the tailnet can reach it. Requires deliberate bind/auth. | bind to the Tailscale address |
| Public internet | Not an ad-hoc bind target. Use a deliberate hosted/deployment path. | out of scope |

Hermes Webapp is not a remote-desktop product. It can be reachable from a chosen network, but it should not pretend to be Chrome Remote Desktop, RustDesk, or a public browser-control service.

## Browser workspace vision

The intended model is: every chat can own its own browser workspace.

A chat workspace may include:

- the chat transcript and active/inactive run state;
- file browser and file preview tabs;
- responsive preview tabs;
- one or more browser tabs tied to that chat;
- viewport presets for phone, foldable, desktop, and ultrawide review;
- an agent-visible tab registry;
- screenshots/annotations for browser RPA;
- visible click/type/scroll actions when the agent controls a tab.

That makes Hermes Webapp useful for work such as:

- reviewing a local web app in multiple responsive sizes;
- helping with Google Sheets, Canva, Notion, dashboards, and SaaS workflows;
- browser RPA where the user can watch and interrupt;
- debugging a web page with screenshots, DOM/accessibility context, and agent-guided actions.

## Extension support direction

Hermes Webapp should provide a browser profile/extension mechanism, not bundle personal extensions. Users may choose to install tools such as password managers or dark-mode extensions in their own Webapp browser context.

Design constraints:

- per-user and preferably per-workspace browser profile isolation;
- no raw attachment to a user's default Chrome profile;
- no bundled LastPass, Dark Reader, or equivalent user-specific extension;
- clear security boundaries when a browser workspace is reachable over LAN or Tailscale.

## Current status

Implemented foundation:

- `hermes webapp` command exists as a browser UI entrypoint;
- it shares the current dashboard server/runtime flags;
- it is not headless (`hermes serve` remains the headless backend);
- dashboard process lifecycle management includes `webapp` processes.

Planned next steps:

1. Add a persisted Webapp workspace state contract.
2. Add browser tab registry APIs per chat session.
3. Build a native Webapp chat workspace instead of relying only on xterm/TUI embedding.
4. Add an extension-capable local browser worker.
5. Add visible, interruptible browser RPA controls.

See `docs/plans/2026-07-08-hermes-webapp-browser-workspace.md` in the repository for the implementation plan.
