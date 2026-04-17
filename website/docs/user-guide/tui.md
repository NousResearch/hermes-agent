---
sidebar_position: 2
title: "TUI"
description: "Launch the modern terminal UI for Hermes — mouse-friendly, rich overlays, and non-blocking input."
---

# TUI

The TUI is the modern front-end for Hermes — a terminal UI backed by the same Python runtime as the [Classic CLI](cli.md). Same agent, same sessions, same slash commands; a cleaner, more responsive surface for interacting with them.

It's the recommended way to run Hermes interactively.

## Launch

```bash
# Launch the TUI
hermes --tui

# Resume the latest TUI session (falls back to the latest classic session)
hermes --tui -c
hermes --tui --continue

# Resume a specific session by ID or title
hermes --tui -r 20260409_000000_aa11bb
hermes --tui --resume "my t0p session"

# Run source directly — skips the prebuild step (for TUI contributors)
hermes --tui --dev
```

You can also enable it via env var:

```bash
export HERMES_TUI=1
hermes          # now uses the TUI
hermes chat     # same
```

The classic CLI remains available as the default. Anything documented in [CLI Interface](cli.md) — slash commands, quick commands, skill preloading, personalities, multi-line input, interrupts — works in the TUI identically.

## Why the TUI

- **Instant first frame** — the banner paints before the app finishes loading, so the terminal never feels frozen while Hermes is starting.
- **Non-blocking input** — type and queue messages before the session is ready. Your first prompt sends the moment the agent comes online.
- **Rich overlays** — model picker, session picker, approval and clarification prompts all render as modal panels rather than inline flows.
- **Live session panel** — tools and skills fill in progressively as they initialize.
- **Mouse-friendly selection** — drag to highlight with a uniform background instead of SGR inverse. Copy with your terminal's normal copy gesture.
- **Alternate-screen rendering** — differential updates mean no flicker when streaming, no scrollback clutter after you quit.
- **Composer affordances** — inline paste-collapse for long snippets, image paste from the clipboard (`Alt+V`), bracketed-paste safety.

Same [skins](features/skins.md) and [personalities](features/personality.md) apply. Switch mid-session with `/skin ares`, `/personality pirate`, and the UI repaints live. Skin keys are marked `(both)`, `(classic)`, or `(tui)` in [`example-skin.yaml`](https://github.com/NousResearch/hermes-agent/blob/main/docs/skins/example-skin.yaml) so you can see at a glance what applies where — the TUI honors the banner palette, UI colors, prompt glyph/color, session display, completion menu, selection bg, `tool_prefix`, and `help_header`.

## Requirements

- **Node.js** ≥ 20 — the TUI runs as a subprocess launched from the Python CLI. `hermes doctor` verifies this.
- **TTY** — like the classic CLI, piping stdin or running in non-interactive environments falls back to single-query mode.

On first launch Hermes installs the TUI's Node dependencies into `ui-tui/node_modules` (one-time, a few seconds). Subsequent launches are fast. If you pull a new Hermes version, the TUI bundle is rebuilt automatically when sources are newer than the dist.

### External prebuild

Distributions that ship a prebuilt bundle (Nix, system packages) can point Hermes at it:

```bash
export HERMES_TUI_DIR=/path/to/prebuilt/ui-tui
hermes --tui
```

The directory must contain `dist/entry.js` and an up-to-date `node_modules`.

## Keybindings

Keybindings match the [Classic CLI](cli.md#keybindings) exactly. The only behavioral differences:

- **Mouse drag** highlights text with a uniform selection background.
- **`Ctrl+V`** pastes text from your clipboard directly into the composer; multi-line pastes stay on one row until you expand them.
- **Slash autocompletion** opens as a floating panel with descriptions, not an inline dropdown.

## Slash commands

All slash commands work unchanged. A few are TUI-owned — they produce richer output or render as overlays rather than inline panels:

| Command | TUI behavior |
|---------|--------------|
| `/help` | Overlay with categorized commands, arrow-key navigable |
| `/sessions` | Modal session picker — preview, title, token totals, resume inline |
| `/model` | Modal model picker grouped by provider, with cost hints |
| `/skin` | Live preview — theme change applies as you browse |
| `/details` | Toggle verbose tool-call details in the transcript |
| `/usage` | Rich token / cost / context panel |

Every other slash command (including installed skills, quick commands, and personality toggles) works identically to the classic CLI. See [Slash Commands Reference](../reference/slash-commands.md).

## Status line

The TUI's status line tracks agent state in real time:

| Status | Meaning |
|--------|---------|
| `starting agent…` | Session ID is live; tools and skills still coming online. You can type — messages queue and send when ready. |
| `ready` | Agent is idle, accepting input. |
| `thinking…` / `running…` | Agent is reasoning or running a tool. |
| `interrupted` | Current turn was cancelled; press Enter to send again. |
| `forging session…` / `resuming…` | Initial connect or `--resume` handshake. |

The per-skin status-bar colors and thresholds are shared with the classic CLI — see [Skins](features/skins.md) for customization.

## Configuration

The TUI respects all standard Hermes config: `~/.hermes/config.yaml`, profiles, personalities, skins, quick commands, credential pools, memory providers, tool/skill enablement. No TUI-specific config file exists.

A handful of keys tune the TUI surface specifically:

```yaml
display:
  skin: default          # any built-in or custom skin
  personality: helpful
  details_mode: compact  # or "verbose" — default tool-call detail level
  mouse_tracking: true   # disable if your terminal conflicts with mouse reporting
```

`/details on` / `/details off` / `/details cycle` toggle this at runtime.

## Sessions

Sessions are shared between the TUI and the classic CLI — both write to the same `~/.hermes/state.db`. You can start a session in one, resume in the other. The session picker surfaces sessions from both sources, with a source tag.

See [Sessions](sessions.md) for lifecycle, search, compression, and export.

## Reverting to the classic CLI

Launching `hermes` (without `--tui`) stays on the classic CLI. To make a machine prefer the TUI, set `HERMES_TUI=1` in your shell profile. To go back, unset it.

If the TUI fails to launch (no Node, missing bundle, TTY issue), Hermes prints a diagnostic and falls back — rather than leaving you stuck.

## See also

- [CLI Interface](cli.md) — full slash command and keybinding reference (shared)
- [Sessions](sessions.md) — resume, branch, and history
- [Skins & Themes](features/skins.md) — theme the banner, status bar, and overlays
- [Voice Mode](features/voice-mode.md) — works in both interfaces
- [Configuration](configuration.md) — all config keys
