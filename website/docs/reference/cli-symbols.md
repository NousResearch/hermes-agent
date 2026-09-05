---
title: "CLI Symbols Glossary"
description: "What every symbol in the Hermes terminal UI means — transcript markers, status-bar badges, overlay glyphs, and approval prompts."
---

# CLI Symbols Glossary

The Hermes terminal interfaces speak a compact visual language: dots, chevrons, braille spinners, and status glyphs. This page is the decoder ring. It covers the [TUI](../user-guide/tui.md) (where most of these render) and notes the pieces shared with the [Classic CLI](../user-guide/cli.md).

:::note Skins can restyle some of these
Glyphs marked *themeable* below are brand defaults — a [skin](../user-guide/features/skins.md) can override them (for example `tool_prefix` and the prompt symbol). Everything else is fixed in the renderer.
:::

## Transcript symbols

What you see in the conversation flow while the agent works.

| Symbol | Meaning |
|--------|---------|
| `❯` | Input prompt — where you type. *Themeable* (skins set their own prompt symbol). |
| `●` | A tool call. The bullet precedes the tool name and its arguments. |
| `┊` | Tool-activity rail shown with tool lines. *Themeable* (`tool_prefix`). |
| `✓` / `✗` | Tool result: succeeded / failed. Appended to the end of a completed tool line. |
| `▸` / `▾` | Collapsed / expanded section chevron (thinking, tools, subagents, banner sections). Click to toggle. |
| `▍` | Streaming cursor — blinks while the model is still emitting text. |
| `│` `├─` `└─` | Tree rails — connect a parent (a delegation, a journey) to its child entries; `└─` marks the last child. |
| `◈` | Display-only timeline event (session notices rendered inline, not user messages). |
| `◇` | An injected reference block, e.g. `◇ Reference 1/2 — <label>`. |
| `↳` | The sticky prompt — echoes the user message the agent is currently working on. |
| `☐` / `☑` / `•` | Markdown task-list items (open / done) and plain list bullets. |
| `▶` | Collapsed `<details>` summary inside rendered markdown. |

## Status-bar symbols

The single line at the bottom of the TUI. Segments appear only when relevant and drop off first on narrow terminals.

| Symbol | Meaning |
|--------|---------|
| `⠋⠙⠹…` (braille patterns) | Busy spinner. Thinking and tool phases use different braille animation sets. |
| `⚕ 🌀 🤔 ✨ 🍵 🔮` | Frames of the `emoji` busy-indicator style (`/indicator emoji`). The default style rotates kaomoji faces instead. |
| <code>&#124; / - &#92;</code> | Frames of the `ascii` busy-indicator style. |
| `⏱` | Per-prompt elapsed time while the turn runs, e.g. `⏱ 12s/3m 45s` (turn time / session time). |
| `⏲` | The same timer, frozen after the turn completes. |
| `cmp N` | The session has been auto-compressed N times. |
| `▶ N` | N `/bg` tasks currently running. |
| `⚠ YOLO` | YOLO mode is on (auto-approval). Also shown in the startup banner. |
| `⛓ N` | N subagents currently active. |
| `↩ resumes when subagent finishes` | Reassurance shown while you are idle but delegated work is still in flight — the result returns on its own. |
| `● REC` | Voice mode is recording. |
| `◉ STT` | Voice recording stopped; speech-to-text is transcribing. |
| `◉ focus` | Focus view is on (reduced output). Pinned so it never drops off a narrow terminal. |
| `♥` | Affection flash — Hermes noticed you being nice to it. |
| `⚡` / `🔋` | Battery indicator (opt-in): plugged in / on battery, with percentage. |
| `N bg` | N background terminal processes tracked in this session. |
| `N live sessions` | Open TUI sessions in this process — click to open the session switcher. |

## Notices

Short-lived status-bar notices carry their own leading glyph, set by severity:

| Symbol | Meaning |
|--------|---------|
| `✓` | Success notice. |
| `•` | Informational notice. |
| `⚠` | Warning (also used for credit warnings). |
| `✕` | Error notice. |

## Approval and confirmation prompts

| Symbol | Meaning |
|--------|---------|
| `⚠ approval required` | A tool wants to run something that needs your explicit yes (bordered panel with the command preview). |
| `⚠` / `?` | Confirmation dialog title: dangerous action / ordinary question. |
| `🔐` | Sudo password prompt (input is masked). |
| `🔑` | Credential/secret input prompt (input is masked). |

## Subagents overlay (`/agents`)

| Symbol | Meaning |
|--------|---------|
| `●` | Subagent running. |
| `○` | Queued. |
| `✓` | Completed. |
| `■` | Interrupted. |
| `✗` | Failed. |
| `⌛` | Timed out. |
| `⚠` | Errored. |
| `⚡N` | N currently-active agents in a rollup row. |
| `▁▂▃▄▅▆▇█` | Activity sparkline — recent event volume per branch. |

## Session switcher (`Ctrl+X`)

| Symbol | Meaning |
|--------|---------|
| `✓` | Session idle. |
| `…` | Starting. |
| `?` | Waiting for input. |
| `▶` | Working. |
| `✎ draft` | The composer in that session holds an unsent draft. |

## Pickers and hubs

| Symbol | Meaning |
|--------|---------|
| `▸` | Current selection row (model picker and friends). |
| `*` | The currently-active model / provider. |
| `●` / `○` | Provider authenticated / not authenticated (model picker); plugin state fallback (plugins hub). |
| `✓` / `✗` | Plugin enabled / disabled (plugins hub). |
| `↑ N more` / `↓ N more` | More rows above/below the visible window of a list. |
| `┃` | Scrollbar thumb on scrollable overlays. |

## Goals

Goal lifecycle notices (from [goals](../user-guide/features/goals.md)) lead with their state:

| Symbol | Meaning |
|--------|---------|
| `✓` | Goal complete. |
| `↻` | Goal continuing — another iteration was scheduled. |
| `⏸` | Goal paused. |

## See also

- [TUI](../user-guide/tui.md) — status line, details modes, busy-indicator styles
- [Classic CLI](../user-guide/cli.md) — shared keybindings and slash commands
- [Skins & Themes](../user-guide/features/skins.md) — which glyphs and colors you can customize
