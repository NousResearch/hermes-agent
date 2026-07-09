# TUI Subtree Instructions

This file scopes TUI guidance to `ui-tui/` work. Root `AGENTS.md` still contains the non-negotiable project rules. Full reference: `docs/agent-context/tui-and-desktop.md`.

## TUI architecture

- `hermes --tui` is an Ink/Node UI connected to Python `tui_gateway` over newline-delimited JSON-RPC on stdio.
- TypeScript owns rendering and input; Python owns sessions, tools, model calls, and slash-command backend logic.
- Keep route/app roots thin; prefer small feature-owned nanostores and focused hooks/modules.
- Use explicit async handler intent such as `onClick={() => void save()}`.

## Dashboard relationship

- Dashboard `/chat` embeds the real TUI through the PTY bridge. Do not rebuild the primary transcript or composer in dashboard React; extend Ink when the main chat UX changes.

## Verification

- Run relevant `ui-tui` checks (`npm run typecheck`, `npm test`, etc.) for TUI changes.
