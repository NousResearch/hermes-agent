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

The single line at the bottom of the TUI (toggle with `/statusbar`). Segments appear only when relevant and drop off first on narrow terminals; a session-title badge is pinned to the far-right edge once the session has a name. Which segments render can be restricted with the `display.status_bar.fields` config list.

| Symbol | Meaning |
|--------|---------|
| `⠋⠙⠹…` (braille patterns) | Busy spinner. Thinking and tool phases use different braille animation sets. |
| `☤ 🌀 🤔 ✨ 🍵 🔮` | Frames of the `emoji` busy-indicator style (`/indicator emoji`). The default style rotates kaomoji faces instead. |
| <code>&#124; / - &#92;</code> | Frames of the `ascii` busy-indicator style. |
| `☤ <model>` | Current model (Classic CLI). The caduceus marks the bar's start. |
| `<used>/<window>` | Context tokens used out of the model's context window (e.g. `~66.6K/1M`). |
| `[█░░░] N%` | Context fill as a bar plus percentage. Color: green < 50%, yellow 50-80%, orange 80-95%, red >= 95%. |
| `~` prefix | The context number is a local estimate (reasoning models replay thinking, so the last request's count can overshoot); provider-exact readings have no `~`. |
| `◎ N%` | Prompt cache hit rate since the last baseline reset (a model switch or compression resets it). >= 70% is good, < 40% is bad. |
| `◷ Ns` | Rolling average API latency per request (Classic CLI). |
| `↑ N t/s` | Rolling average output throughput, tokens per second (Classic CLI). |
| `🗜️ N` / `cmp N` | The session has been auto-compressed N times (`🗜️` in the Classic CLI, `cmp N` in the TUI). |
| `⚙ N` / `N bg` | N background processes tracked in this session (Classic CLI: `⚙ N`, TUI: `N bg`). |
| `▶ N` | N `/bg` tasks currently running. |
| `⛓ N` | N subagents currently active. |
| `⊙ goal a/b` | A standing [goal](../user-guide/features/goals.md) is active - a of b turns used. |
| `⎇ <branch>` | Git branch of the working directory (opt-in via `display.status_bar.fields`). |
| `Σ<N>` | Session token total (opt-in via an explicit fields list, Classic CLI). |
| `⏱` | Per-prompt elapsed time while the turn runs, e.g. `⏱ 12s/3m 45s` (turn time / session time). |
| `⏲` | The same timer, frozen after the turn completes. |
| `✓ <dur>` | Time since the last final response - idle time (Classic CLI); hidden while a turn is live. |
| `📌 N` | N prompts parked in the stash (Ctrl+S); `▲` appended while the stash panel is open. |
| `⚠ YOLO` | YOLO mode is on (auto-approval). Also shown in the startup banner. |
| `↩ resumes when subagent finishes` | Reassurance shown while you are idle but delegated work is still in flight — the result returns on its own. |
| `● REC` | Voice mode is recording. |
| `◉ STT` | Voice recording stopped; speech-to-text is transcribing. |
| `◉ focus` | Focus view is on (reduced output). Pinned so it never drops off a narrow terminal. |
| `♥` | Affection flash — Hermes noticed you being nice to it. |
| `⚡` / `🔋` | Battery indicator (opt-in): plugged in / on battery, with percentage. |
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
