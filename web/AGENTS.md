# Web dashboard

The root guidance applies. Backend routes live in `hermes_cli/web_routers/` and mount through the `hermes_cli/web_server.py` facade. The frontend lives in `web/src/`. JSON-RPC/WebSocket code shared with Desktop lives in `apps/shared`.

## Chat boundary

The dashboard chat embeds the real `hermes --tui` through `hermes_cli/pty_bridge.py` and `/api/pty`. It does not reimplement the transcript, composer, or slash-command behavior in React. Extend the Ink TUI for primary chat behavior.

Structured React UI may surround the terminal for inspectors, model selection, summaries, and status. Keep that state separate from the PTY child and make its failures non-destructive so terminal chat continues.

The PTY WebSocket uses the same ephemeral session token as REST. Browsers pass it in the upgrade query because they cannot set the authorization header there. Frames remain raw PTY bytes. Resize uses the existing escape-frame protocol and server-side terminal resize path.

The PTY implementation is POSIX-backed. WSL is supported through that path. Native Windows is not. Do not claim native support without a real transport implementation and host test.

## `dashboard` and `serve`

`dashboard` and `serve` share startup code but are separate surfaces. `serve` is the headless backend used by Desktop and does not mount the SPA, even if stale web assets exist. Desktop has no build or runtime dependency on `web/`. See `apps/desktop/src/AGENTS.md`.

## Local rules

- Every REST and WebSocket endpoint uses the existing session-token scheme.
- Add a dashboard backend surface as a topical router rather than growing `web_server.py`.
- Keep primary chat and slash behavior in the TUI contract. Keep optional dashboard panels outside the PTY session.
- Run Python router/PTY tests from `tests/hermes_cli/` through `scripts/run_tests.sh`. Run frontend tests through scripts declared in `web/package.json`.
- JavaScript behavior is tested in Vitest, not by Python tests that inspect TypeScript or package files.

Read `tui_gateway/AGENTS.md` for generated JSON-RPC contracts and `website/docs/developer-guide/architecture.md` for system context.