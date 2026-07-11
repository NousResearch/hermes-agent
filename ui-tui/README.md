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

Interpreter resolution order is: `HERMES_PYTHON` → `PYTHON` → `$VIRTUAL_ENV/bin/python` → `./.venv/bin/python` → `./venv/bin/python` → `python3` (or `python` on Windows).

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

The CLI expects `ui-tui/dist/entry.js` to exist, or the whole source code available in which to run `npm install` and `npm run dev`.

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

Tests use vitest:

```bash
npm test         # single run
npm run test:watch
```

## App model

`src/app.tsx` is the center of the UI. Heavy logic is split into `src/app/`:

- `src/app/createGatewayEventHandler.ts` — maps gateway events to state updates
- `src/app/createSlashHandler.ts` — local slash command dispatch
- `src/app/useComposerState.ts` — draft, multiline buffer, queue editing
- `src/app/useInputHandlers.ts` — keypress routing
- `src/app/useMainApp.ts` — top-level composition hook: wires all sub-hooks, manages transcript history, session polling, and exposes props consumed by `app.tsx`
- `src/app/useSessionLifecycle.ts` — session create / resume / activate / close and visible-history reset
- `src/app/useSubmission.ts` — message send, shell exec (`!cmd`), inline interpolation (`{!cmd}`), and busy-input-mode dispatch (queue / steer / interrupt)
- `src/app/turnController.ts` — stateful class that drives the turn lifecycle: buffers streaming deltas, manages tool/reasoning state, handles interrupt and message-complete transitions
- `src/app/turnStore.ts` — nanostore for turn state (streaming text, tools, reasoning, subagents, todos, activity trail)
- `src/app/useConfigSync.ts` — fetches `config.get full` on session start and polls config mtime every 5 s; applies display settings and triggers MCP reload on change
- `src/app/useLongRunToolCharms.ts` — fires ambient activity messages for tools running longer than 8 s
- `src/app/overlayStore.ts` / `src/app/uiStore.ts` — nanostores for overlay and UI state
- `src/app/delegationStore.ts` — nanostore for subagent spawning caps and overlay accordion state
- `src/app/spawnHistoryStore.ts` — in-memory ring (last 10) of finished subagent fan-out snapshots; populated at turn end for `/replay`
- `src/app/inputSelectionStore.ts` — nanostore exposing the active text-input selection handle
- `src/app/gatewayContext.tsx` — React context for the gateway client
- `src/app/gatewayRecovery.ts` — pure function that decides whether to respawn and resume after a gateway crash, with a 3-attempt / 60 s budget
- `src/app/setupHandoff.ts` — launches external `hermes setup`, suspends Ink while it runs, opens a new session on success
- `src/app/scroll.ts` — scrolls the viewport while keeping the text selection anchor in sync
- `src/app/interfaces.ts` — internal interfaces (ComposerActions, GatewayRpc, etc.)

### Slash command subsystem (`src/app/slash/`)

- `types.ts` — `SlashCommand` interface and `SlashRunCtx` execution context (gateway rpc, transcript helpers, session refs, stale-guard)
- `registry.ts` — assembles `SLASH_COMMANDS` from all command files in registration order (core → billing → credits → session → ops → setup → debug) and exposes `findSlashCommand(name)` for case-insensitive lookup
- `commands/core.ts` — general TUI commands
- `commands/billing.ts` — `/billing`: manage Nous terminal billing — buy credits, auto-reload, limits
- `commands/credits.ts` — `/credits`
- `commands/session.ts` — session and agent commands
- `commands/ops.ts` — operations commands
- `commands/setup.ts` — `/setup`
- `commands/debug.ts` — `/heapdump`, `/mem`

The top-level `app.tsx` composes these into the Ink tree with `Static` transcript output, a live streaming assistant row, prompt overlays, queue preview, status rule, input line, and completion list.

State managed at the top level includes:

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

| Key                             | Behavior                                                                                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Enter`                         | Submit the current draft                                                                                                                                |
| empty `Enter` twice             | If queued messages exist and the agent is busy, interrupt the current run. If queued messages exist and the agent is idle, send the next queued message |
| `Shift+Enter` / `Alt+Enter`     | Insert a newline in the current draft                                                                                                                   |
| `\` + `Enter`                   | Append the line to the multiline buffer (fallback for terminals without modifier support)                                                               |
| `Ctrl+C`                        | Interrupt active run, or clear the current draft, or exit if nothing is pending                                                                         |
| `Ctrl+D`                        | Exit                                                                                                                                                    |
| `Cmd/Ctrl+G` / `Alt+G`          | Open `$EDITOR` with the current draft (use `Alt+G` in VSCode/Cursor — they bind the primary keystroke to Find Next)                                     |
| `Ctrl+L`                        | New session (same as `/clear`)                                                                                                                          |
| `Ctrl+V` / `Alt+V`              | Paste text first, then fall back to image/path attachment when applicable                                                                               |
| `Tab`                           | Apply the active completion                                                                                                                             |
| `Up/Down`                       | Cycle completions if the completion list is open; otherwise edit queued messages first, then walk input history                                         |
| `Left/Right`                    | Move the cursor                                                                                                                                         |
| modified `Left/Right`           | Move by word when the terminal sends `Ctrl` or `Meta` with the arrow key                                                                                |
| `Home` / `Ctrl+A`               | Start of line                                                                                                                                           |
| `End` / `Ctrl+E`                | End of line                                                                                                                                             |
| `Backspace`                     | Delete the character to the left of the cursor                                                                                                          |
| `Delete`                        | Delete the character to the right of the cursor                                                                                                         |
| modified `Backspace`            | Delete the previous word                                                                                                                                |
| modified `Delete`               | Delete the next word                                                                                                                                    |
| `Ctrl+W`                        | Delete the previous word                                                                                                                                |
| `Ctrl+U`                        | Delete from the cursor back to the start of the line                                                                                                    |
| `Ctrl+K`                        | Delete from the cursor to the end of the line                                                                                                           |
| `Meta+B` / `Meta+F`             | Move by word                                                                                                                                            |
| `!cmd`                          | Run a shell command through the gateway                                                                                                                 |
| `{!cmd}`                        | Inline shell interpolation before send; queued drafts keep the raw text until they are sent                                                            |

Notes:

- `Tab` only applies completions when completions are present and you are not in multiline mode.
- Queue/history navigation only applies when you are not in multiline mode.
- `PgUp` / `PgDn` are left to the terminal emulator; the TUI does not handle them.

### Prompt and picker modes

| Context                     | Keys                | Behavior                                          |
| --------------------------- | ------------------- | ------------------------------------------------- |
| approval prompt             | `Up/Down`, `Enter`  | Move and confirm the selected approval choice     |
| approval prompt             | `o`, `s`, `a`, `d`  | Quick-pick `once`, `session`, `always`, `deny`    |
| approval prompt             | `Esc`, `Ctrl+C`     | Deny                                              |
| clarify prompt with choices | `Up/Down`, `Enter`  | Move and confirm the selected choice              |
| clarify prompt with choices | single-digit number | Quick-pick the matching numbered choice           |
| clarify prompt with choices | `Enter` on "Other"  | Switch into free-text entry                       |
| clarify free-text mode      | `Enter`             | Submit typed answer                               |
| sudo / secret prompt        | `Enter`             | Submit typed value                                |
| sudo / secret prompt        | `Ctrl+C`            | Cancel by sending an empty response               |
| resume picker               | `Up/Down`, `Enter`  | Move and resume the selected session              |
| resume picker               | `1-9`               | Quick-pick one of the first nine visible sessions |
| resume picker               | `Esc`, `Ctrl+C`     | Close the picker                                  |

Notes:

- Clarify free-text mode and masked prompts use `ink-text-input`, so text editing there follows the library's default bindings rather than `components/textInput.tsx`.
- When a blocking prompt is open, the main chat input hotkeys are suspended.
- Clarify mode has no dedicated cancel shortcut in the current client. Sudo and secret prompts only expose `Ctrl+C` cancellation from the app-level blocked handler.

### Interaction rules

- Plain text entered while the agent is busy is queued instead of sent immediately.
- Slash commands and `!cmd` do not queue; they execute immediately even while a run is active.
- Queue auto-drains after each assistant response, unless a queued item is currently being edited.
- `Up/Down` prioritizes queued-message editing over history. History only activates when there is no queue to edit.
- Queued drafts keep their original `!cmd` and `{!cmd}` text while you edit them. Shell commands and interpolation run when the queued item is actually sent.
- If you load a queued item into the input and resubmit plain text, that queue item is replaced, removed from the queue preview, and promoted to send next. If the agent is still busy, the edited item is moved to the front of the queue and sent after the current run completes.
- Completion requests are debounced by 60 ms. Input starting with `/` uses `complete.slash`. A trailing token that starts with `./`, `../`, `~/`, `/`, or `@` uses `complete.path`.
- Text pastes are inserted inline directly into the draft. Nothing is newline-flattened.
- `Cmd/Ctrl+G` (or `Alt+G` in VSCode/Cursor, which intercept the primary keystroke for Find Next) writes the current draft, including any multiline buffer, to a temp file, suspends Ink, launches `$EDITOR`, then restores the TUI and submits the saved text if the editor exits cleanly.
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

The following commands are handled directly by the TUI client. Unrecognized commands fall through to the Python gateway via `slash.exec` and `command.dispatch`.

### Core (`core.ts`)
`/help`, `/quit` (alias `/exit`), `/update`, `/clear` (alias `/new`),
`/compact`, `/copy`, `/paste`, `/details` (alias `/detail`),
`/statusbar` (alias `/sb`), `/queue` (alias `/q`), `/logs`, `/history`,
`/save`, `/undo`, `/retry`, `/steer`, `/mouse` (alias `/scroll`),
`/status`, `/title`, `/fortune`, `/redraw`, `/terminal-setup`

### Billing (`billing.ts`)
`/billing` — manage Nous terminal billing — buy credits, auto-reload, limits

### Session (`session.ts`)
`/model`, `/sessions` (aliases `/switch`, `/session`, `/resume`),
`/background` (aliases `/bg`, `/btw`), `/image`, `/personality`,
`/compress`, `/branch` (alias `/fork`), `/voice`, `/skin`,
`/indicator`, `/yolo`, `/reasoning`, `/fast`, `/busy`, `/verbose`, `/usage`

### Ops (`ops.ts`)
`/stop`, `/reload-mcp` (alias `/reload_mcp`), `/reload`, `/browser`,
`/rollback`, `/agents` (alias `/tasks`), `/replay`, `/replay-diff`,
`/skills`, `/reload-skills` (alias `/reload_skills`), `/plugins`, `/tools`

### Credits (`credits.ts`)
`/credits` — Nous credit balance and browser top-up

### Setup (`setup.ts`)
`/setup` — launches external `hermes setup` wizard, suspends Ink while it runs

### Debug (`debug.ts`)
`/heapdump`, `/mem` — V8 memory diagnostics

---

Anything not matched above falls through to:

1. `slash.exec`
2. `command.dispatch`

That lets Python own aliases, plugins, skills, and registry-backed commands without duplicating the logic in the TUI.

## Event surface

Primary event types the client handles today:

| Event                      | Payload                                                                     |
| -------------------------- | --------------------------------------------------------------------------- |
| `gateway.ready`            | `{ skin? }`                                                                 |
| `skin.changed`             | `{ skin }`                                                                  |
| `session.info`             | session metadata for banner + tool/skill panels                             |
| `message.start`            | start assistant streaming                                                   |
| `message.delta`            | `{ text, rendered? }`                                                       |
| `message.complete`         | `{ text, rendered?, usage, status }`                                        |
| `thinking.delta`           | `{ text }`                                                                  |
| `reasoning.delta`          | `{ text, verbose? }`                                                        |
| `reasoning.available`      | `{ text, verbose? }`                                                        |
| `commentary.delta`         | `{ text }` — Codex mid-turn narration (desktop: "Working" lane)             |
| `status.update`            | `{ kind, text }`                                                            |
| `notification.show`        | `{ id, key, kind, level, text, ttl_ms? }`                                   |
| `notification.clear`       | `{ key }`                                                                   |
| `tool.start`               | `{ tool_id, name, context?, args_text? }`                                   |
| `tool.generating`          | `{ name }`                                                                  |
| `tool.progress`            | `{ name, preview }`                                                         |
| `tool.complete`            | `{ tool_id, name, error?, summary?, duration_s?, inline_diff?, todos? }`    |
| `clarify.request`          | `{ question, choices?, request_id }`                                        |
| `approval.request`         | `{ command, description, allow_permanent? }`                                |
| `sudo.request`             | `{ request_id }`                                                            |
| `secret.request`           | `{ prompt, env_var, request_id }`                                           |
| `background.complete`      | `{ task_id, text }`                                                         |
| `billing.step_up.verification` | `{ verification_url, user_code }`                                       |
| `review.summary`           | `{ text }`                                                                  |
| `browser.progress`         | `{ message }`                                                               |
| `voice.status`             | `{ state }`                                                                 |
| `voice.transcript`         | `{ text, no_speech_limit? }`                                                |
| `subagent.spawn_requested` | `{ subagent_id?, task_index, goal?, depth?, parent_id? }`                   |
| `subagent.start`           | `{ subagent_id?, task_index, goal?, depth?, parent_id? }`                   |
| `subagent.thinking`        | `{ text }`                                                                  |
| `subagent.tool`            | `{ tool_name?, tool_preview?, text? }`                                      |
| `subagent.progress`        | `{ text }`                                                                  |
| `subagent.complete`        | `{ status, summary?, text?, duration_seconds? }`                            |
| `error`                    | `{ message }`                                                               |
| `gateway.stderr`           | synthesized from child stderr                                               |
| `gateway.protocol_error`   | synthesized from malformed stdout                                           |
| `gateway.start_timeout`    | `{ cwd?, python?, stderr_tail? }`                                           |

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
ui-tui/
  packages/hermes-ink/   forked Ink renderer (local dep)
  src/
    entry.tsx            TTY gate + render()
    app.tsx              top-level Ink tree, composes src/app/*
    gatewayClient.ts     child process + JSON-RPC bridge
    gatewayTypes.ts      gateway event and RPC response type definitions
    theme.ts             theme colors and skin merge
    banner.ts            ASCII art renderer (parses Rich color tags)
    types.ts             shared client-side types (ActiveTool, Msg, etc.)

    app/
      createGatewayEventHandler.ts  event → state mapping
      createSlashHandler.ts         local slash dispatch
      delegationStore.ts            nanostore for subagent spawning caps and overlay accordion state
      gatewayContext.tsx            React context for gateway client
      gatewayRecovery.ts            crash-recovery budget: respawn+resume capped to 3 attempts / 60 s
      inputSelectionStore.ts        nanostore exposing the active text-input selection handle
      interfaces.ts                 internal interfaces (ComposerActions, GatewayRpc, etc.)
      overlayStore.ts               nanostores for overlay state
      scroll.ts                     viewport scroll with text-selection anchor sync
      setupHandoff.ts               launches external hermes setup, suspends Ink while it runs
      spawnHistoryStore.ts          ring buffer of finished subagent fan-out snapshots
      turnController.ts             stateful turn lifecycle driver (streaming, tools, reasoning)
      turnStore.ts                  nanostore for turn state (streaming, tools, reasoning, subagents)
      uiStore.ts                    nanostores for UI flags (busy, sid, mouseTracking, etc.)
      useComposerState.ts           draft + multiline buffer + queue editing
      useConfigSync.ts              config polling and MCP reload on mtime change
      useInputHandlers.ts           keypress routing
      useLongRunToolCharms.ts       ambient activity messages for tools running longer than 8 s
      useMainApp.ts                 top-level composition hook
      useSessionLifecycle.ts        session create / resume / activate / close
      useSubmission.ts              message send, shell exec, interpolation, busy-input-mode dispatch

      slash/
        types.ts                    SlashCommand interface and SlashRunCtx execution context
        registry.ts                 SLASH_COMMANDS assembly and findSlashCommand lookup
        commands/
          billing.ts                /billing — manage Nous terminal billing
          core.ts                   general TUI commands
          credits.ts                /credits
          debug.ts                  /heapdump, /mem
          ops.ts                    operations commands
          session.ts                session and agent commands
          setup.ts                  /setup wizard

    components/
      activeSessionSwitcher.tsx  active session switch overlay
      agentsOverlay.tsx          subagent delegation overlay
      appChrome.tsx              status bar, input row, completions
      appLayout.tsx              top-level layout composition
      appOverlays.tsx            overlay routing (pickers, prompts)
      billingOverlay.tsx         billing overlay
      branding.tsx               banner + session summary
      fpsOverlay.tsx             FPS debug overlay
      helpHint.tsx               contextual help hint
      markdown.tsx               Markdown-to-Ink renderer
      maskedPrompt.tsx           masked input for sudo / secrets
      messageLine.tsx            transcript rows
      modelPicker.tsx            model switch picker
      overlayControls.tsx        shared overlay control buttons
      pluginsHub.tsx             plugins hub overlay
      prompts.tsx                approval + clarify flows
      queuedMessages.tsx         queued input preview
      skillsHub.tsx              skills hub overlay
      streamingAssistant.tsx     live streaming assistant row
      streamingMarkdown.tsx      streaming Markdown renderer
      textInput.tsx              custom line editor
      themed.tsx                 theme-aware wrapper
      thinking.tsx               spinner, reasoning, tool activity
      todoPanel.tsx              todo list panel

    config/
      env.ts                     environment variable resolution and Termux/mouse defaults
      limits.ts                  paste size, live-render and history limits
      timing.ts                  streaming batch and debounce timing constants

    content/
      charms.ts                  ambient activity strings for long-running tools
      faces.ts                   agent face / kaomoji pool
      fortunes.ts                /fortune quote pool
      hotkeys.ts                 platform-aware hotkey display strings
      placeholders.ts            rotating input placeholder strings
      setup.ts                   setup-required panel content
      verbs.ts                   tool activity verb map (browser → browsing, etc.)

    domain/
      blockLayout.ts             block layout and lead-gap helpers
      details.ts                 details visibility mode resolution (hidden/collapsed/expanded)
      messages.ts                message formatting and transcript helpers
      paths.ts                   cwd shortening and path display helpers
      providers.ts               provider display name helpers
      roles.ts                   message role color and label helpers
      slash.ts                   slash command parsing and TUI session model flag
      usage.ts                   token usage zero value and helpers
      viewport.ts                viewport height estimation helpers

    hooks/
      useCompletion.ts           tab completion (slash + path)
      useGitBranch.ts            current git branch via child_process execFile
      useInputHistory.ts         persistent history navigation
      useQueue.ts                queued message management
      useVirtualHistory.ts       virtual list scroll and height tracking

    lib/
      circularBuffer.ts          fixed-size generic ring buffer
      clipboard.ts               clipboard read / write via child_process
      editor.ts                  $EDITOR launch, PATH resolution, and Ink suspend
      emoji.ts                   emoji and variation selector width helpers
      externalCli.ts             external CLI subprocess launcher
      externalLink.ts            open URLs in the system browser
      forceTruecolor.ts          24-bit truecolor override before chalk imports
      fpsStore.ts                Ink frame FPS tracker nanostore
      fuzzy.ts                   lightweight fuzzy subsequence scorer
      gracefulExit.ts            clean shutdown with failsafe timeout
      history.ts                 persistent input history (read/append to disk)
      inputMetrics.ts            input width and wrap metrics
      liveProgress.ts            todo helpers and tool-shelf message assembly
      mathUnicode.ts             best-effort LaTeX → Unicode for inline math
      memory.ts                  V8 heap snapshot and diagnostics helpers
      memoryMonitor.ts           automatic heap-dump trigger on high usage
      messages.ts                transcript message append helpers
      openExternalUrl.ts         platform-aware URL opener (macOS/Linux/Windows)
      osc52.ts                   OSC 52 terminal clipboard copy sequence
      parentLog.ts               append-only log to ~/.hermes/tui-parent.log
      perfPane.tsx               FPS / render perf overlay pane
      platform.ts                platform-aware keybinding and SSH detection helpers
      precisionWheel.ts          high-precision scroll wheel with sticky-frame budget
      prompt.ts                  composer prompt text helpers (Termux-safe)
      reasoning.ts               reasoning tag detection and split helpers
      rpc.ts                     JSON-RPC result and command dispatch helpers
      subagentTree.ts            subagent tree flattening and aggregate helpers
      syntax.ts                  syntax token types and theme-aware highlighting
      terminalModes.ts           terminal mode reset sequences (kitty, mouse, etc.)
      terminalParity.ts          VSCode-like terminal detection and hint helpers
      terminalSetup.ts           IDE keybinding config file install helpers
      termux.ts                  Termux platform detection helpers
      text.ts                    text helpers, ANSI detection, tool trail builders
      todo.ts                    todo item tone and display helpers
      viewportStore.ts           viewport height nanostore via ScrollBoxHandle
      virtualHeights.ts          virtual list row height estimation
      wheelAccel.ts              scroll wheel acceleration state machine

    protocol/
      interpolation.ts           {!cmd} inline shell interpolation regex and helpers
      paste.ts                   bracketed paste snippet token regex

    types/
      hermes-ink.d.ts            type declarations for @hermes/ink

    __tests__/                   vitest suite
```

Related Python side:

```text
tui_gateway/
  entry.py               stdio entrypoint
  server.py              RPC handlers and session logic
  render.py              optional rich/ANSI bridge
  slash_worker.py        persistent HermesCLI subprocess for slash commands
```