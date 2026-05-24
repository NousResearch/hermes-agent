# TUI Gateway Guide

Follow `ui-tui/AGENTS.md` for the overall TUI architecture.

This directory owns the Python JSON-RPC backend used by `hermes --tui`. Keep it
aligned with the classic CLI and gateway command behavior:

- Python owns sessions, tools, model calls, slash-command fallthrough, approval
  handling, and session persistence.
- TypeScript owns rendering and local client-only commands.
- Do not move primary chat behavior into dashboard React components; it should
  flow through Ink and this gateway.

When adding methods or events, keep the newline-delimited JSON-RPC transport
stable and update the matching TypeScript client code in `ui-tui/src/`.
