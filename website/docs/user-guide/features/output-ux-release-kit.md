---
sidebar_position: 39
title: "Output UX Release Kit"
description: "Ready-to-use output mode profiles, config examples, and CLI tags for richer Hermes UX across CLI, TUI, desktop, and messaging."
---

# Output UX Release Kit

This is a ready-to-use upgrade kit for Hermes output UX. It bundles the three highest-leverage display settings into copy-paste configs, plus the CLI commands to switch modes at runtime.

- `final_response_markdown` — controls how the agent's closing reply is rendered
- `tool_progress` — controls how much tool-call progress is visible
- `tool_progress_grouping` — controls whether gateway tool progress edits one bubble or emits many

## Quick start

Pick one profile and paste it into `~/.hermes/config.yaml` under `display:`.

## Profiles

### Rich terminal (CLI / TUI / desktop) — maximize output fidelity

```yaml
display:
  interface: tui              # or omit for CLI
  final_response_markdown: render
  tool_progress: verbose
  tool_preview_length: 0      # full paths/commands
  streaming: true
  show_reasoning: true
  show_cost: true
  timestamps: true
```

Use this when:
- Your terminal/TUI width is comfortable
- You want full markdown rendering in responses
- You want tool args, previews, and timing visible by default

### Messaging-light (Telegram / Discord / Slack) — unobtrusive but available

```yaml
display:
  final_response_markdown: strip
  tool_progress: new
  tool_progress_grouping: accumulate
  tool_progress_command: true  # lets users toggle richer detail
```

Use this when:
- You want tidy chat output
- You still want easy opt-in to detail via `/verbose`
- You prefer one editable progress bubble rather than many messages

## CLI schema tags and commands

These are the runtime controls for output modes in the interactive CLI/TUI.

### Output mode tags

| Tag | Scope | what it changes | default | notes |
|------|-------|----------------|---------|-------|
| final_response_markdown | CLI/TUI/desktop | renders agent's final reply: `render` keeps markdown, `strip` removes noisy decorators, `raw` passes unchanged | strip | Use `render` in wide terminals/TUI/desktop; `strip` is safer on narrow terminals |
| tool_progress | all surfaces | tool progress density: `off` | `new` | default `off` on some surfaces; CLI often default `all` or higher |
| tool_progress_grouping | gateway messaging | message shape on editing platforms: `accumulate` = one bubble edited in place, `separate` = one message per tool | accumulate | N/A for CLI/TUI; relevant on Telegram/Discord/Slack/messaging |
| friendly_tool_labels | CLI + gateway | human-phrased status labels for built-in tools instead of raw tool names | true | affects spinner/progress only |
| timestamps | CLI | prepend `[HH:MM]` timestamps to user/assistant labels | false | off in minimal setups |
| show_cost | CLI | show estimated `$ cost` in the status bar | false | opt-in where budget tracking matters |
| streaming | CLI | stream token output live instead of replacing the line | true in ideal configs | can be set per response |

### Commands

| Command | Effect |
|--------|--------|
| `/verbose` | Cycle `tool_progress` for this run: `off → new → all → verbose → off` |
| `/reasoning show` | Show model reasoning/thinking above each response |
| `/reasoning hide` | Hide reasoning again |
| `/details <section> <mode>` | Collapse or expand TUI sections: `thinking` | `tools` | `subagents` | `activity` |
| `/indicator emoji` | Switch TUI busy indicator style: `kaomoji | emoji | unicode | ascii` |
| `/title My Session` | Name the active session for easier resume later |

## Platform cheat sheet

| Surface | Recommended `tool_progress` | Recommended `final_response_markdown` | Recommended `tool_progress_grouping` |
|---------|----------------------------|--------------------------------------|--------------------------------------|
| Classic CLI | `all` or `verbose` | `render` | N/A |
| TUI | `all` or `verbose` | `render` | N/A |
| Desktop | `all` or `verbose` | `render` | N/A |
| Telegram | `new` (opt-in `all`) | `strip` | `accumulate` |
| Discord | `new` (opt-in `all`) | `strip` | `accumulate` |
| Slack | `new` (opt-in `all`) | `strip` | `accumulate` |
| Signal / SMS | `off` | `strip` or `raw` | N/A |
| Web dashboard | follows CLI/TUI defaults | `render` preferred | N/A |

## Troubleshooting

- If rich responses look broken in your terminal, switch `final_response_markdown` to `strip`.
- If tool progress feels noisy in chat, switch `tool_progress` to `new` and keep `tool_progress_grouping: accumulate`.
- If edited bubbles feel jittery or clipped, try `tool_progress_grouping: separate` temporarily.
- If you see markdown symbols leaking into external wrappers, prefer `raw` only when piping to another renderer.

## See also

- [CLI Interface](/user-guide/cli)
- [TUI Interface](/user-guide/tui)
- [Configuration](/user-guide/configuration)
- [Tool Progress Grouping](/user-guide/features/tool-progress-grouping)
- [Skins & Themes](/user-guide/features/skins)
