# web/ + hermes_cli/web_routers/ — the dashboard (`hermes dashboard` → `/chat`)

Applies on top of the root `AGENTS.md`. Backend routers: `hermes_cli/web_routers/*.py`, one file per
dashboard surface, mounted by `hermes_cli/web_server.py` (+ `web_server_*.py` siblings). Frontend:
`web/src/`. Shared JSON-RPC/WS client: `apps/shared` (`@hermes/shared`), also used by the desktop.

## The dashboard embeds the REAL `hermes --tui` — not a rewrite

`hermes_cli/pty_bridge.py` + the `@app.websocket("/api/pty")` endpoint in `web_server.py`:

- `web/src/pages/ChatPage.tsx` mounts xterm.js `Terminal` with the WebGL renderer, `@xterm/addon-fit`
  (container-driven resize) and `@xterm/addon-unicode11` (wide-character widths).
- `/api/pty?token=…` upgrades to a WebSocket; auth uses the same ephemeral `_SESSION_TOKEN` as REST,
  passed as a query param because browsers cannot set `Authorization` on a WS upgrade.
- The server spawns exactly what `hermes --tui` would spawn, through `ptyprocess` (POSIX PTY — WSL
  works, native Windows does not).
- Frames are raw PTY bytes each way; resize travels as `\x1b[RESIZE:<cols>;<rows>]`, intercepted
  on the server and applied with `TIOCSWINSZ`.

**Do not re-implement the primary chat experience in React.** Transcript, composer/input flow
(including slash-command behaviour), and the PTY-backed terminal belong to the embedded TUI; anything
added to Ink shows up here automatically. If you are rebuilding the transcript or composer for the
dashboard, stop and extend Ink (`tui_gateway/AGENTS.md`).

**Structured React UI around the TUI is fine when it is not a second chat surface.** Sidebar
widgets, inspectors, summaries, status panels (`ChatSidebar`, `ModelPickerDialog`, `ToolCall`)
complement the embedded TUI. Keep their state independent of the PTY child's session and surface
their failures non-destructively so the terminal pane keeps working.

## `dashboard` vs `serve`

`dashboard` and `serve` share `cmd_dashboard` / `start_server` but are independent surfaces — neither
launches the other. `serve` is the headless backend the desktop app spawns (`headless_backend=True`:
`cmd_dashboard` skips `_build_web_ui` and exports `HERMES_SERVE_HEADLESS=1` so `mount_spa()`
disables the SPA even if a stray `web_dist/` exists — only JSON-RPC/WS/API is reachable). The
desktop has no build/runtime dependency on this frontend. Details: `apps/desktop/src/AGENTS.md`.

## Rules

- Auth: every new REST route and WS endpoint uses the same session token; never a second scheme.
- Routers are one-file-per-surface; a new surface is a new `web_routers/<surface>.py`, not a growing
  `web_server.py`.
- Tests: Python in `tests/hermes_cli/` (routers, pty bridge); JS in the `web/` vitest suite. Python
  tests must not assert about `package.json` / `.tsx` sources (root testing rules). Root TypeScript
  style rules apply.

## Localization framework (Dashboard + Ink TUI)


- English is the complete source catalog and only final fallback. Every non-English pack
  overlays English independently; Simplified Chinese is the first complete non-English
  implementation, not a privileged runtime branch.
- `locales/registry.json` is the single authority for locale identities, endonyms, picker
  labels, ordinary aliases and protocol compatibility aliases. Python, Ink and Dashboard
  consume it directly; do not recreate locale lists or normalization tables elsewhere.
- Product language choices expose `zh` (Simplified Chinese) and `zh-hant` (Traditional
  Chinese) as independent languages. Region-tagged inputs such as `zh-TW`, `zh-HK` and
  `zh-MO` are boundary-only compatibility values that normalize immediately to `zh-hant`.
- Stable translation keys belong at presentation boundaries. Components, commands, schema
  renderers and bundled Dashboard extensions must not branch on a named locale or use
  translated labels as control-flow identifiers.
- This rollout covers Dashboard/Web UI, Ink TUI, its Dashboard embedding, and bundled
  Dashboard extensions. Classic CLI, Electron Desktop chat, website and messaging-platform
  presentation need deliberate follow-up work rather than incidental changes here.
- A display-language refresh must not rebuild the agent or reload MCP/tool schemas. Tool
  reload remains an explicit user action so per-conversation prompt caching stays stable.
- Acceptance tests assert behavior: complete English, complete declared packs, direct English
  fallback for partial packs, cross-runtime normalization parity, and no feature-code edits
  required when registering a new locale. Never freeze locale counts.
