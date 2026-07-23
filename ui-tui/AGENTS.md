# Hermes TUI Development Guide

These instructions apply to `ui-tui/`. The TUI is the Ink/React terminal client
for the Python `tui_gateway` JSON-RPC backend.

## Ownership boundary

TypeScript owns rendering, interaction, local client commands, and UI state.
Python owns sessions, model calls, tools, approvals, and the canonical slash
command dispatch.

Transport is newline-delimited JSON-RPC over stdio:

```text
hermes --tui
  └─ Ink client ── JSON-RPC/stdin/stdout ── tui_gateway
                                            └─ AIAgent
```

Key flow:

- chat: `prompt.submit` → `message.delta` / `message.complete`;
- tools: `tool.start` / `tool.progress` / `tool.complete`;
- approvals and prompts: request events → matching `.respond` methods;
- sessions: `session.list` / `session.resume`;
- commands: local client handler, then `slash.exec`;
- completion: `complete.slash` and `complete.path`.

## Dashboard boundary

The dashboard `/chat` embeds the real TUI through a PTY. Do not recreate the
transcript, composer, terminal, or slash-command behavior in React. Extend Ink
so the dashboard inherits the change.

Structured React UI around the PTY—sidebars, inspectors, summaries, status
panels—is allowed when it does not become a second chat surface. Keep its state
independent from the PTY child and make failures non-destructive.

The Electron desktop app is a separate chat client and has its own scoped guide
at `apps/desktop/AGENTS.md`.

## TypeScript conventions

- Shared state belongs in small, feature-owned nanostores.
- Rendering components subscribe with `useStore`; actions read atoms directly.
- Keep route roots thin and hooks single-purpose.
- Avoid prop-drilling state through three layers.
- Keep persistence with the atom that owns it.
- Prefer interfaces for public object shapes.
- Extend React primitives with `React.ComponentProps`, `Pick`, or `Omit`.
- Prefer table-driven mappings over long condition ladders.
- Make ignored promises explicit with `void`.

## Commands

```bash
npm install
npm run dev
npm run build
npm run typecheck
npm run lint
npm run fmt
npm test
```

Run the smallest relevant command during iteration, then the build, typecheck,
and focused tests before handoff.
