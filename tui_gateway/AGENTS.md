# tui_gateway/ + ui-tui/ — the TUI and its JSON-RPC backend

Applies on top of the root `AGENTS.md`. The TUI fully replaces the classic prompt_toolkit CLI;
activate with `hermes --tui` or `HERMES_TUI=1`. `tui_gateway` is ALSO the backend the Desktop app
and the dashboard `/chat` talk to — changes here have three consumers.

## Process model

```
hermes --tui
  └─ Node (Ink)  ──stdio JSON-RPC──  Python (tui_gateway)
       │                                  └─ AIAgent + tools + sessions
       └─ renders transcript, composer, prompts, activity
```

TypeScript owns the screen. Python owns sessions, tools, model calls, and slash-command logic.
Never move agent behaviour into the renderer.

## Transport

Newline-delimited JSON-RPC over stdio: requests from Ink, events from Python. `tui_gateway/server.py`
is the facade with the method/event catalog; methods live in `methods_*.py` siblings (`methods_config`,
`methods_complete`, `methods_browser`, `methods_bot_relay`, ...), event publishing in
`event_publisher.py` / `event_replay.py`. Desktop reaches the same server over WebSocket via
`apps/shared` (`JsonRpcGatewayClient`). New RPC = a new `methods_<topic>.py` or an entry in an
existing topical sibling, registered in the table — no `if method == ...` chain (root shape rules).

## Key surfaces

| Surface | Ink component | Gateway method / event |
|---|---|---|
| Chat streaming | `app.tsx` + `messageLine.tsx` | `prompt.submit` → `message.delta` / `message.complete` |
| Tool activity | `thinking.tsx` | `tool.start` / `tool.progress` / `tool.complete` |
| Approvals | `prompts.tsx` | `approval.request` → `approval.respond` |
| Clarify / sudo / secret | `prompts.tsx`, `maskedPrompt.tsx` | `clarify.respond`, `sudo.respond`, `secret.respond` |
| Session picker | `sessionPicker.tsx` | `session.list` / `session.resume` |
| Slash commands | local handler + fallthrough | `slash.exec` → `_SlashWorker`; `command.dispatch` |
| Completions | `useCompletion` hook | `complete.slash`, `complete.path` |
| Theming | `theme.ts` + `branding.tsx` | `gateway.ready` carries skin data |
| Plugin compat notice | — | `plugins.compat_report` (see `plugins/AGENTS.md`) |

## Slash command flow

1. Built-in client commands (`/help`, `/quit`, `/clear`, `/resume`, `/copy`, `/paste`, ...) are
   handled locally in `app.tsx`.
2. Everything else → `slash.exec`, which runs in the persistent `_SlashWorker` subprocess →
   `command.dispatch` fallback, which the gateway resolves into a skill / alias / exec directive
   (a skill command resolves to `{type: "skill", message}` and is submitted as a normal prompt).

`commands.catalog` (empty-query list) and `complete.slash` (typed-query completions) already include
built-ins, user `quick_commands`, AND skill-derived commands (`scan_skill_commands()` /
`get_skill_commands()`) — clients do not need a new RPC to see skills. The command definitions
themselves come from `hermes_cli/commands.py` (`hermes_cli/AGENTS.md`).

## Dev commands

```bash
cd ui-tui
npm install       # first time
npm run dev       # watch mode (rebuilds hermes-ink + tsx --watch)
npm start         # production
npm run build     # full build (hermes-ink + tsc)
npm run typecheck # tsc --noEmit
npm run lint      # eslint
npm run fmt       # prettier
npm test          # vitest
```

Python tests: `tests/tui_gateway/` via `scripts/run_tests.sh`. TS tests: vitest in `ui-tui`. A
Python test that asserts about `package.json` / `.ts` sources will not run on a JS-only PR — keep
JS-side assertions in vitest (root testing rules). Root TypeScript style rules apply.

Related: `web/AGENTS.md` (dashboard embeds this TUI over a PTY), `apps/desktop/AGENTS.md` (own
renderer on the same backend).
