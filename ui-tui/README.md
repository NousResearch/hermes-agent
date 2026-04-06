# Hermes TUI

Ink-based terminal UI for the Hermes agent. Two processes, one stdio pipe: TypeScript renders the screen, Python runs the brain.

```
hermes --tui
```

## How it works

The TUI spawns `tui_gateway.entry` as a child process. They talk newline-delimited JSON-RPC over stdin/stdout. Python handles sessions, tools, and LLM calls. Ink handles layout, input, and rendering. Stderr is piped into a ring buffer (never hits the terminal).

```
Ink (ui-tui/)                    Python (tui_gateway/)
─────────────                    ─────────────────────
  TextInput                        entry.py
      │                               │
      ├─ JSON-RPC request ──────────► handle_request()
      │                               │
      │                           server.py
      │                            40 RPC methods
      │                            agent threads
      │                               │
      ◄── JSON-RPC response ─────────┤
      ◄── event push (streaming) ────┘
```

All Python writes go through a locked `write_json()` so concurrent agent threads can't interleave bytes on stdout.

## Layout

The app runs in the alternate screen buffer. Everything fits in one fixed frame -- no native scroll. The message area gets whatever rows are left after subtracting chrome:

```
rows - header - thinking - queue - palette - statusbar - separator - input
```

The viewport walks backward from the newest message, estimating each one's rendered height, until the row budget runs out. PgUp/PgDn shift the window.

## Hotkeys

| Key | What it does |
|-----|-------------|
| Ctrl+C | Interrupt / clear / exit (contextual) |
| Ctrl+D | Exit |
| Ctrl+G | Open `$EDITOR` for multiline prompt |
| Ctrl+L | Clear messages |
| Ctrl+V | Paste clipboard image |
| Tab | Complete `/commands` |
| Up/Down | Cycle queue or input history |
| PgUp/PgDn | Scroll |
| Esc | Clear input |
| `\` + Enter | Continue on next line |
| `!cmd` | Shell command |
| `{!cmd}` | Interpolate shell output inline |

## Ctrl+G editor

Writes the current input to a temp file, leaves the alt screen, opens your `$EDITOR`, then reads the file back and submits it on save. Multiline `\`-continued input pre-populates the file.

## Message queue

Input typed while the agent is busy gets queued. The queue drains automatically after each response. Double-Enter sends the next queued item. Arrow keys let you edit queued messages before they send.

## Rendering

Assistant text goes through `markdown.tsx` -- a zero-dependency JSX renderer that handles code blocks (with diff coloring), headings, lists, quotes, tables, and inline formatting. If the Python side provides pre-rendered ANSI (via `agent.rich_output`), that takes priority.

## Slash commands

60+ commands wired in the local `slash()` switch. Anything unrecognized falls through to `command.dispatch` on the Python side (quick commands, plugins, skill commands) with alias resolution. `/help` lists them all.

## Events (Python -> Ink)

| Event | Payload |
|-------|---------|
| `gateway.ready` | `{ skin? }` |
| `session.info` | `{ model, tools, skills }` |
| `message.start` | -- |
| `message.delta` | `{ text, rendered? }` |
| `message.complete` | `{ text, rendered?, usage, status }` |
| `thinking.delta` | `{ text }` |
| `reasoning.delta` | `{ text }` |
| `status.update` | `{ kind, text }` |
| `tool.generating` | `{ name }` |
| `tool.start` | `{ tool_id, name }` |
| `tool.progress` | `{ name, preview }` |
| `tool.complete` | `{ tool_id, name }` |
| `clarify.request` | `{ question, choices?, request_id }` |
| `approval.request` | `{ command, description }` |
| `sudo.request` | `{ request_id }` |
| `secret.request` | `{ prompt, env_var, request_id }` |
| `background.complete` | `{ task_id, text }` |
| `btw.complete` | `{ text }` |
| `error` | `{ message }` |

The client also synthesizes `gateway.stderr` and `gateway.protocol_error` from the child process.

## RPC methods (40)

Session: `session.create`, `session.list`, `session.resume`, `session.branch`, `session.title`, `session.usage`, `session.history`, `session.undo`, `session.compress`, `session.save`, `session.interrupt`, `terminal.resize`

Prompts: `prompt.submit`, `prompt.background`, `prompt.btw`

Responses: `clarify.respond`, `approval.respond`, `sudo.respond`, `secret.respond`, `clipboard.paste`

Config: `config.set`, `config.get`

System: `process.stop`, `reload.mcp`, `shell.exec`, `cli.exec`, `commands.catalog`, `command.resolve`, `command.dispatch`

Features: `voice.toggle`, `voice.record`, `voice.tts`, `insights.get`, `rollback.list`, `rollback.restore`, `rollback.diff`, `browser.manage`, `plugins.list`, `cron.manage`, `skills.manage`

## Performance notes

- `MessageLine` and `Thinking` are `React.memo`'d so they skip re-render when the user types
- The spinner uses `useRef` instead of `useState` -- no parent re-renders at 80ms intervals
- `rpc` and `newSession` are `useCallback`-stable so the gateway event listener doesn't re-subscribe every render
- On a normal keystroke, only the input line re-renders. Viewport recalc only triggers on message/scroll changes.

## Themes

Python loads a skin from `~/.hermes/config.yaml` at startup. The `gateway.ready` event carries colors and branding to the client, which merges them into the default palette (gold/amber/bronze/cornsilk). Branding overrides the agent name, prompt symbol, and welcome text.

## Files

```
ui-tui/src/
  entry.tsx           entrypoint
  app.tsx             state, events, input, commands, layout
  altScreen.tsx       alternate screen buffer
  gatewayClient.ts    JSON-RPC child process bridge
  constants.ts        hotkeys, tool verbs, spinner frames
  theme.ts            palette + skin mapping
  types.ts            shared interfaces
  banner.ts           ASCII art

  lib/
    text.ts           ANSI strip, row estimation
    messages.ts       streaming message upsert
    history.ts        persistent input history
    slash.ts          command palette + tab completion
    osc52.ts          clipboard via OSC 52

  components/
    messageLine.tsx   chat message (memoized)
    markdown.tsx      MD renderer (code, diff, tables)
    thinking.tsx      spinner / reasoning / tool progress
    queuedMessages.tsx  queue display
    prompts.tsx       approval + clarify
    maskedPrompt.tsx  sudo / secret input
    sessionPicker.tsx session resume picker
    commandPalette.tsx  slash suggestions
    branding.tsx      welcome banner + session panel

tui_gateway/
  entry.py            stdin loop, JSON-RPC dispatch
  server.py           40 RPC methods, session management
  render.py           optional Rich ANSI rendering bridge
```
