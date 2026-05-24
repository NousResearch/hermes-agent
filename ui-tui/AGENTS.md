# TUI and Dashboard Chat Guide

These rules apply to `ui-tui/` and `tui_gateway/`.

## Architecture

`hermes --tui` runs an Ink/React terminal UI in Node and a Python JSON-RPC
backend over stdio.

```text
hermes --tui
  `- Node (Ink) --stdio JSON-RPC-- Python (tui_gateway)
       |                             `- AIAgent + tools + sessions
       `- transcript, composer, prompts, activity
```

TypeScript owns rendering. Python owns sessions, tools, model calls, slash
commands that fall through, and agent execution.

## Main Surfaces

| Surface | Ink component | Gateway method/event |
| --- | --- | --- |
| chat streaming | `app.tsx`, `messageLine.tsx` | `prompt.submit`, `message.delta`, `message.complete` |
| tool activity | `thinking.tsx` | `tool.start`, `tool.progress`, `tool.complete` |
| approvals | `prompts.tsx` | `approval.request`, `approval.respond` |
| clarify/sudo/secret | `prompts.tsx`, `maskedPrompt.tsx` | response methods |
| session picker | `sessionPicker.tsx` | `session.list`, `session.resume` |
| slash commands | local handler + fallthrough | `slash.exec`, `command.dispatch` |
| completions | `useCompletion` | `complete.slash`, `complete.path` |
| theming | `theme.ts`, `branding.tsx` | `gateway.ready` skin data |

## Slash Commands

Handle client-only commands locally in `app.tsx` where appropriate. Everything
else should fall through to the persistent slash worker and Python command
dispatch so behavior stays consistent with classic CLI/gateway surfaces.

## Dev Commands

```bash
cd ui-tui
npm install
npm run dev
npm start
npm run build
npm run type-check
npm run lint
npm run fmt
npm test
```

Use the repo's existing package manager and scripts. Do not hand-edit generated
build artifacts unless the surrounding codebase already tracks them.

## Dashboard `/chat`

The dashboard embeds the real `hermes --tui` via `hermes_cli/pty_bridge.py` and
the `/api/pty` WebSocket. Do not re-implement the main transcript, composer, or
terminal in React.

Structured React UI around the TUI is allowed when it complements the PTY-backed
chat: sidebars, inspectors, model pickers, tool-call panels, summaries, and
status panels. Keep those failures non-destructive so the terminal pane remains
usable.

## Resize and Transport

PTY frames are raw bytes. Resize is sent as the special
`\x1b[RESIZE:<cols>;<rows>]` frame and handled server-side with `TIOCSWINSZ`.

Authentication for `/api/pty` uses the dashboard session token in the query
string because browsers cannot set `Authorization` on WebSocket upgrade.
