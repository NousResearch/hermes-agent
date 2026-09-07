# web governance

Read repository-root `AGENTS.md` and its required routing table before acting.
These are supplemental current-lineage rules, not exceptions to global rules.
Read this file completely; automatic hints may truncate it. Scope follows the
behavior affected, including tests, docs and callers outside this directory.

### TUI in the Dashboard (`hermes dashboard` → `/chat`)

The dashboard embeds the real `hermes --tui` — **not** a rewrite.  See `hermes_cli/pty_bridge.py` + the `@app.websocket("/api/pty")` endpoint in `hermes_cli/web_server.py`.

- Browser loads `web/src/pages/ChatPage.tsx`, which mounts xterm.js's `Terminal` with the WebGL renderer, `@xterm/addon-fit` for container-driven resize, and `@xterm/addon-unicode11` for modern wide-character widths.
- `/api/pty?token=…` upgrades to a WebSocket; auth uses the same ephemeral `_SESSION_TOKEN` as REST, via query param (browsers can't set `Authorization` on WS upgrade).
- The server spawns whatever `hermes --tui` would spawn, through `ptyprocess` (POSIX PTY — WSL works, native Windows does not).
- Frames: raw PTY bytes each direction; resize via `\x1b[RESIZE:<cols>;<rows>]` intercepted on the server and applied with `TIOCSWINSZ`.

**Do not re-implement the primary chat experience in React.** The main transcript, composer/input flow (including slash-command behavior), and PTY-backed terminal belong to the embedded `hermes --tui` — anything new you add to Ink shows up in the dashboard automatically. If you find yourself rebuilding the transcript or composer for the dashboard, stop and extend Ink instead.

**Structured React UI around the TUI is allowed when it is not a second chat surface.** Sidebar widgets, inspectors, summaries, status panels, and similar supporting views (e.g. `ChatSidebar`, `ModelPickerDialog`, `ToolCall`) are fine when they complement the embedded TUI rather than replacing the transcript / composer / terminal. Keep their state independent of the PTY child's session and surface their failures non-destructively so the terminal pane keeps working unimpaired.

<!-- End of relocated baseline guidance. -->
