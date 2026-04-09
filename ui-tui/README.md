# Hermes TUI

React + Ink terminal UI for Hermes. TypeScript owns the screen. Python owns sessions, tools, model calls, and most command logic.

```bash
hermes --tui
```

## What runs

The client entrypoint is `src/entry.tsx`. It exits early if `stdin` is not a TTY, starts `GatewayClient`, then renders `App`.

`GatewayClient` spawns:

```text
python -m tui_gateway.entry
```

By default it uses `venv/bin/python` from the repo root. Set `HERMES_PYTHON` to override.

The transport is newline-delimited JSON-RPC over stdio:

```text
ui-tui/src                  tui_gateway/
-----------                 -------------
entry.tsx                   entry.py
  -> GatewayClient            -> request loop
  -> App                      -> server.py RPC handlers

stdin/stdout: JSON-RPC requests, responses, events
stderr: captured into an in-memory log ring
```

Malformed stdout lines are treated as protocol noise and surfaced as `gateway.protocol_error`. Stderr lines become `gateway.stderr`. Neither writes directly into the terminal.

## Running it

From the repo root, the normal path is:

```bash
hermes --tui
```

The CLI expects `ui-tui/node_modules` to exist. If the TUI deps are missing:

```bash
cd ui-tui
npm install
```

Local package commands:

```bash
npm run dev
npm start
npm run build
npm run lint
npm run fmt
npm run fix
```

There is no package-local test script today.

## App model

`src/app.tsx` is the center of the UI. It holds:

- transcript and streaming state
- queued messages and input history
- session lifecycle
- tool progress and reasoning text
- prompt flows for approval, clarify, sudo, and secret input
- slash command routing
- tab completion and path completion
- theme state from gateway skin data

The UI renders as a normal Ink tree with `Static` transcript output, a live streaming assistant row, prompt overlays, queue preview, status rule, input line, and completion list.

The intro panel is driven by `session.info` and rendered through `branding.tsx`.

## Hotkeys and interactions

Current input behavior is split across `app.tsx`, `components/textInput.tsx`, and the prompt/picker components.

### Main chat input

| Key | Behavior |
|---|---|
| `Enter` | Submit the current draft |
| empty `Enter` twice | If queued messages exist and the agent is busy, interrupt the current run. If queued messages exist and the agent is idle, send the next queued message |
| `\` + `Enter` | Append the line to the multiline buffer instead of sending |
| `Ctrl+C` | Interrupt active run, or clear the current draft, or exit if nothing is pending |
| `Ctrl+D` | Exit |
| `Ctrl+G` | Open `$EDITOR` with the current draft |
| `Ctrl+L` | New session (same as `/clear`) |
| `Ctrl+V` | Paste clipboard image (same as `/paste`) |
| `Esc` | Clear the current draft |
| `Tab` | Apply the active completion |
| `Up/Down` | Cycle completions if the completion list is open; otherwise edit queued messages first, then walk input history |
| `Left/Right` | Move the cursor |
| modified `Left/Right` | Move by word when the terminal sends `Ctrl` or `Meta` with the arrow key |
| `Home` / `Ctrl+A` | Start of line |
| `End` / `Ctrl+E` | End of line |
| `Backspace` / `Delete` | Delete the character to the left of the cursor |
| modified `Backspace` / `Delete` | Delete the previous word |
| `Ctrl+W` | Delete the previous word |
| `Ctrl+U` | Delete from the cursor back to the start of the line |
| `Ctrl+K` | Delete from the cursor to the end of the line |
| `Meta+B` / `Meta+F` | Move by word |
| `!cmd` | Run a shell command through the gateway |
| `{!cmd}` | Inline shell interpolation before send or queue |

Notes:

- `Tab` only applies completions when completions are present and you are not in multiline mode.
- Queue/history navigation only applies when you are not in multiline mode.
- `PgUp` / `PgDn` are left to the terminal emulator; the TUI does not handle them.

### Prompt and picker modes

| Context | Keys | Behavior |
|---|---|---|
| approval prompt | `Up/Down`, `Enter` | Move and confirm the selected approval choice |
| approval prompt | `o`, `s`, `a`, `d` | Quick-pick `once`, `session`, `always`, `deny` |
| approval prompt | `Esc`, `Ctrl+C` | Deny |
| clarify prompt with choices | `Up/Down`, `Enter` | Move and confirm the selected choice |
| clarify prompt with choices | single-digit number | Quick-pick the matching numbered choice |
| clarify prompt with choices | `Enter` on "Other" | Switch into free-text entry |
| clarify free-text mode | `Enter` | Submit typed answer |
| sudo / secret prompt | `Enter` | Submit typed value |
| sudo / secret prompt | `Ctrl+C` | Cancel by sending an empty response |
| resume picker | `Up/Down`, `Enter` | Move and resume the selected session |
| resume picker | `1-9` | Quick-pick one of the first nine visible sessions |
| resume picker | `Esc`, `Ctrl+C` | Close the picker |

Notes:

- Clarify free-text mode and masked prompts use `ink-text-input`, so text editing there follows the library's default bindings rather than `components/textInput.tsx`.
- When a blocking prompt is open, the main chat input hotkeys are suspended.
- Clarify mode has no dedicated cancel shortcut in the current client. Sudo and secret prompts only expose `Ctrl+C` cancellation from the app-level blocked handler.

### Interaction rules

- Plain text entered while the agent is busy is queued instead of sent immediately.
- Slash commands and `!cmd` do not queue; they execute immediately even while a run is active.
- Queue auto-drains after each assistant response, unless a queued item is currently being edited.
- `Up/Down` prioritizes queued-message editing over history. History only activates when there is no queue to edit.
- If you load a queued item into the input and resubmit plain text, that queue item is replaced, removed from the queue preview, and promoted to send next. If the agent is still busy, the edited item is moved to the front of the queue and the current run is interrupted first.
- Completion requests are debounced by 60 ms. Input starting with `/` uses `complete.slash`. A trailing token that starts with `./`, `../`, `~/`, `/`, or `@` uses `complete.path`.
- Text pastes are captured into a local paste shelf and inserted as `[[paste:<id>]]` tokens. Nothing is newline-flattened.
- Small pastes default to `excerpt` mode. Larger pastes default to `attach` mode.
- Very large paste references trigger a confirmation prompt before send.
- Pasted content is scanned for obvious secret patterns before send and redacted in the outbound payload.
- `Ctrl+G` writes the current draft, including any multiline buffer, to a temp file, temporarily swaps screen buffers, launches `$EDITOR`, then restores the TUI and submits the saved text if the editor exits cleanly.
- Input history is stored in `~/.hermes/.hermes_history` or under `HERMES_HOME`.

## Rendering

Assistant output is rendered in one of two ways:

- if the payload already contains ANSI, `messageLine.tsx` prints it directly
- otherwise `components/markdown.tsx` renders a small Markdown subset into Ink components

The Markdown renderer handles headings, lists, block quotes, tables, fenced code blocks, diff coloring, inline code, emphasis, links, and plain URLs.

Tool/status activity is shown in a live activity lane. Transcript rows stay focused on user/assistant turns.

## Prompt flows

The Python gateway can pause the main loop and request structured input:

- `approval.request`: allow once, allow for session, allow always, or deny
- `clarify.request`: pick from choices or type a custom answer
- `sudo.request`: masked password entry
- `secret.request`: masked value entry for a named env var
- `session.list`: used by `SessionPicker` for `/resume`

These are stateful UI branches in `app.tsx`, not separate screens.

## Commands

The local slash handler covers the built-ins that need direct client behavior:

- `/help`
- `/quit`, `/exit`, `/q`
- `/clear`
- `/new`
- `/compact`
- `/resume`
- `/copy`
- `/paste`
- `/logs`
- `/statusbar`, `/sb`
- `/queue`
- `/undo`
- `/retry`

Notes:

- `/copy` sends the selected assistant response through OSC 52.
- `/paste` with no args asks the gateway for clipboard image attachment state.
- `/paste list|mode|drop|clear` manages text paste-shelf items.
- `/statusbar` toggles the status rule on/off.

Anything else falls through to:

1. `slash.exec`
2. `command.dispatch`

That lets Python own aliases, plugins, skills, and registry-backed commands without duplicating the logic in the TUI.

## Event surface

Primary event types the client handles today:

| Event | Payload |
|---|---|
| `gateway.ready` | `{ skin? }` |
| `session.info` | session metadata for banner + tool/skill panels |
| `message.start` | start assistant streaming |
| `message.delta` | `{ text, rendered? }` |
| `message.complete` | `{ text, rendered?, usage, status }` |
| `thinking.delta` | `{ text }` |
| `reasoning.delta` | `{ text }` |
| `status.update` | `{ kind, text }` |
| `tool.start` | `{ tool_id, name, context? }` |
| `tool.progress` | `{ name, preview }` |
| `tool.complete` | `{ tool_id, name }` |
| `clarify.request` | `{ question, choices?, request_id }` |
| `approval.request` | `{ command, description }` |
| `sudo.request` | `{ request_id }` |
| `secret.request` | `{ prompt, env_var, request_id }` |
| `background.complete` | `{ task_id, text }` |
| `btw.complete` | `{ text }` |
| `error` | `{ message }` |
| `gateway.stderr` | synthesized from child stderr |
| `gateway.protocol_error` | synthesized from malformed stdout |

## Theme model

The client starts with `DEFAULT_THEME` from `theme.ts`, then merges in gateway skin data from `gateway.ready`.

Current branding overrides:

- agent name
- prompt symbol
- welcome text
- goodbye text

Current color overrides:

- banner title, accent, border, body, dim
- label, ok, error, warn

`branding.tsx` uses those values for the logo, session panel, and update notice.

## File map

```text
ui-tui/src/
  entry.tsx              TTY gate + render()
  app.tsx                main state machine and UI
  gatewayClient.ts       child process + JSON-RPC bridge
  theme.ts               default palette + skin merge
  constants.ts           display constants, hotkeys, tool labels
  types.ts               shared client-side types
  banner.ts              ASCII art data

  components/
    branding.tsx         banner + session summary
    markdown.tsx         Markdown-to-Ink renderer
    maskedPrompt.tsx     masked input for sudo / secrets
    messageLine.tsx      transcript rows
    prompts.tsx          approval + clarify flows
    queuedMessages.tsx   queued input preview
    sessionPicker.tsx    session resume picker
    textInput.tsx        custom line editor
    thinking.tsx         spinner, reasoning, tool activity

  lib/
    history.ts           persistent input history
    osc52.ts             OSC 52 clipboard copy
    text.ts              text helpers, ANSI detection, previews
```

Related Python side:

```text
tui_gateway/
  entry.py               stdio entrypoint
  server.py              RPC handlers and session logic
  render.py              optional rich/ANSI bridge
```

## Notes

- No dead code: `main.tsx`, `altScreen.tsx`, `commandPalette.tsx`, and `lib/slash.ts` have been removed.
