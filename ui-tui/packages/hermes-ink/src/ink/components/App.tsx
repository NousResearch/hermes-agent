import React, { PureComponent, type ReactNode } from 'react'

import { updateLastInteractionTime } from '../../bootstrap/state.js'
import { logForDebugging } from '../../utils/debug.js'
import { stopCapturingEarlyInput } from '../../utils/earlyInput.js'
import { isMouseClicksDisabled } from '../../utils/fullscreen.js'
import { logError } from '../../utils/log.js'
import type { DOMElement } from '../dom.js'
import { EventEmitter } from '../events/emitter.js'
import { InputEvent } from '../events/input-event.js'
import { TerminalFocusEvent } from '../events/terminal-focus-event.js'
import {
  INITIAL_STATE,
  type ParsedInput,
  type ParsedKey,
  type ParsedMouse,
  parseMultipleKeypresses
} from '../parse-keypress.js'
import reconciler from '../reconciler.js'
import { finishSelection, hasSelection, type SelectionState, startSelection } from '../selection.js'
import { getTerminalFocused, setTerminalFocused } from '../terminal-focus-state.js'
import { TerminalQuerier, xtversion } from '../terminal-querier.js'
import { isXtermJs, setXtversionName, supportsExtendedKeys } from '../terminal.js'
import {
  DISABLE_KITTY_KEYBOARD,
  DISABLE_MODIFY_OTHER_KEYS,
  ENABLE_KITTY_KEYBOARD,
  ENABLE_MODIFY_OTHER_KEYS,
  FOCUS_IN,
  FOCUS_OUT
} from '../termio/csi.js'
import { DBP, DFE, DISABLE_MOUSE_TRACKING, EBP, EFE, HIDE_CURSOR, SHOW_CURSOR } from '../termio/dec.js'

import AppContext from './AppContext.js'
import { ClockProvider } from './ClockContext.js'
import CursorDeclarationContext, { type CursorDeclarationSetter } from './CursorDeclarationContext.js'
import ErrorOverview from './ErrorOverview.js'
import StdinContext from './StdinContext.js'
import { TerminalFocusProvider } from './TerminalFocusContext.js'
import { TerminalSizeContext } from './TerminalSizeContext.js'

// Platforms that support Unix-style process suspension (SIGSTOP/SIGCONT)
const SUPPORTS_SUSPEND = false

// After this many milliseconds of stdin silence, the next chunk triggers
// a terminal mode re-assert (mouse tracking). Catches tmux detach→attach,
// ssh reconnect, and laptop wake — the terminal resets DEC private modes
// but no signal reaches us. 5s is well above normal inter-keystroke gaps
// but short enough that the first scroll after reattach works.
const STDIN_RESUME_GAP_MS = 5000
type Props = {
  readonly children: ReactNode
  readonly stdin: NodeJS.ReadStream
  readonly stdout: NodeJS.WriteStream
  readonly stderr: NodeJS.WriteStream
  readonly exitOnCtrlC: boolean
  readonly onExit: (error?: Error) => void
  readonly terminalColumns: number
  readonly terminalRows: number
  // Text selection state. App mutates this directly from mouse events
  // and calls onSelectionChange to trigger a repaint. Mouse events only
  // arrive when <AlternateScreen> (or similar) enables mouse tracking,
  // so the handler is always wired but dormant until tracking is on.
  readonly selection: SelectionState
  readonly onSelectionChange: () => void
  // Dispatch a click at (col, row) — hit-tests the DOM tree and bubbles
  // onClick handlers. Returns true if a DOM handler consumed the click.
  // No-op (returns false) outside fullscreen mode (Ink.dispatchClick
  // gates on altScreenActive).
  readonly onClickAt: (col: number, row: number) => boolean
  readonly onMouseDownAt: (col: number, row: number, button: number) => DOMElement | undefined
  readonly onMouseUpAt: (target: DOMElement, col: number, row: number, button: number) => void
  readonly onMouseDragAt: (target: DOMElement, col: number, row: number, button: number) => void
  // Dispatch hover (onMouseEnter/onMouseLeave) as the pointer moves over
  // DOM elements. Called for mode-1003 motion events with no button held.
  // No-op outside fullscreen (Ink.dispatchHover gates on altScreenActive).
  readonly onHoverAt: (col: number, row: number) => void
  // Look up the OSC 8 hyperlink at (col, row) synchronously at click
  // time. Returns the URL or undefined. The browser-open is deferred by
  // MULTI_CLICK_TIMEOUT_MS so double-click can cancel it.
  readonly getHyperlinkAt: (col: number, row: number) => string | undefined
  // Open a hyperlink URL in the browser. Called after the timer fires.
  readonly onOpenHyperlink: (url: string) => void
  // Called on double/triple-click PRESS at (col, row). count=2 selects
  // the word under the cursor; count=3 selects the line. Ink reads the
  // screen buffer to find word/line boundaries and mutates selection,
  // setting isDragging=true so a subsequent drag extends by word/line.
  readonly onMultiClick: (col: number, row: number, count: 2 | 3) => void
  // Called on drag-motion. Mode-aware: char mode updates focus to the
  // exact cell; word/line mode snaps to word/line boundaries. Needs
  // screen-buffer access (word boundaries) so lives on Ink, not here.
  readonly onSelectionDrag: (col: number, row: number) => void
  // Called when stdin data arrives after a >STDIN_RESUME_GAP_MS gap.
  // Ink re-asserts terminal modes: extended key reporting, and (when in
  // fullscreen) re-enters alt-screen + mouse tracking. Idempotent on the
  // terminal side. Optional so testing.tsx doesn't need to stub it.
  readonly onStdinResume?: () => void
  // Receives the declared native-cursor position from useDeclaredCursor
  // so ink.tsx can park the terminal cursor there after each frame.
  // Enables IME composition at the input caret and lets screen readers /
  // magnifiers track the input. Optional so testing.tsx doesn't stub it.
  readonly onCursorDeclaration?: CursorDeclarationSetter
  // Dispatch a keyboard event through the DOM tree. Called for each
  // parsed key alongside the legacy EventEmitter path.
  readonly dispatchKeyboardEvent: (parsedKey: ParsedKey) => void
}

// Multi-click detection thresholds. 500ms is the macOS default; a small
// position tolerance allows for trackpad jitter between clicks.
const MULTI_CLICK_TIMEOUT_MS = 500
const MULTI_CLICK_DISTANCE = 1
type State = {
  readonly error?: Error
}

// Root component for all Ink apps
// It renders stdin and stdout contexts, so that children can access them if needed
// It also handles Ctrl+C exiting and cursor visibility
export default class App extends PureComponent<Props, State> {
  static displayName = 'InternalApp'
  static getDerivedStateFromError(error: Error) {
    return {
      error
    }
  }
  override state = {
    error: undefined
  }

  // Count how many components enabled raw mode to avoid disabling
  // raw mode until all components don't need it anymore
  rawModeEnabledCount = 0
  inputEmitter = new EventEmitter()
  keyParseState = INITIAL_STATE
  // Timer for flushing incomplete escape sequences
  incompleteEscapeTimer: NodeJS.Timeout | null = null
  // Timeout durations for incomplete sequences (ms)
  readonly NORMAL_TIMEOUT = 50 // Short timeout for regular esc sequences
  readonly PASTE_TIMEOUT = 500 // Longer timeout for paste operations

  // Terminal query/response dispatch. Responses arrive on stdin (parsed
  // out by parse-keypress) and are routed to pending promise resolvers.
  querier = new TerminalQuerier(this.props.stdout)

  // Multi-click tracking for double/triple-click text selection. A click
  // within MULTI_CLICK_TIMEOUT_MS and MULTI_CLICK_DISTANCE of the previous
  // click increments clickCount; otherwise it resets to 1.
  lastClickTime = 0
  lastClickCol = -1
  lastClickRow = -1
  clickCount = 0
  // Deferred hyperlink-open timer — cancelled if a second click arrives
  // within MULTI_CLICK_TIMEOUT_MS (so double-clicking a hyperlink selects
  // the word without also opening the browser). DOM onClick dispatch is
  // NOT deferred — it returns true from onClickAt and skips this timer.
  pendingHyperlinkTimer: ReturnType<typeof setTimeout> | null = null
  // Last mode-1003 motion position. Terminals already dedupe to cell
  // granularity but this also lets us skip dispatchHover entirely on
  // repeat events (drag-then-release at same cell, etc.).
  lastHoverCol = -1
  lastHoverRow = -1
  mouseCaptureTarget: DOMElement | undefined

  // Timestamp of last stdin chunk. Used to detect long gaps (tmux attach,
  // ssh reconnect, laptop wake) and trigger terminal mode re-assert.
  // Initialized to now so startup doesn't false-trigger.
  lastStdinTime = Date.now()

  // Determines if TTY is supported on the provided stdin
  isRawModeSupported(): boolean {
    return this.props.stdin.isTTY
  }
  override render() {
    return (
      <TerminalSizeContext.Provider
        value={{
          columns: this.props.terminalColumns,
          rows: this.props.terminalRows
        }}
      >
        <AppContext.Provider
          value={{
            exit: this.handleExit
          }}
        >
          <StdinContext.Provider
            value={{
              stdin: this.props.stdin,
              setRawMode: this.handleSetRawMode,
              isRawModeSupported: this.isRawModeSupported(),
              exitOnCtrlC: this.props.exitOnCtrlC,
              inputEmitter: this.inputEmitter,
              querier: this.querier
            }}
          >
            <TerminalFocusProvider>
              <ClockProvider>
                <CursorDeclarationContext.Provider value={this.props.onCursorDeclaration ?? (() => {})}>
                  {this.state.error ? <ErrorOverview error={this.state.error as Error} /> : this.props.children}
                </CursorDeclarationContext.Provider>
              </ClockProvider>
            </TerminalFocusProvider>
          </StdinContext.Provider>
        </AppContext.Provider>
      </TerminalSizeContext.Provider>
    )
  }
  override componentDidMount() {
    // In accessibility mode, keep the native cursor visible for screen magnifiers and other tools
    if (this.props.stdout.isTTY) {
      this.props.stdout.write(HIDE_CURSOR)
    }
  }
  override componentWillUnmount() {
    if (this.props.stdout.isTTY) {
      this.props.stdout.write(SHOW_CURSOR)
    }

    // Clear any pending timers
    if (this.incompleteEscapeTimer) {
      clearTimeout(this.incompleteEscapeTimer)
      this.incompleteEscapeTimer = null
    }

    if (this.pendingHyperlinkTimer) {
      clearTimeout(this.pendingHyperlinkTimer)
      this.pendingHyperlinkTimer = null
    }

    // ignore calling setRawMode on an handle stdin it cannot be called
    if (this.isRawModeSupported()) {
      this.handleSetRawMode(false)
    }
  }
  override componentDidCatch(error: Error) {
    this.handleExit(error)
  }
  handleSetRawMode = (isEnabled: boolean): void => {
    const { stdin } = this.props

    if (!this.isRawModeSupported()) {
      if (stdin === process.stdin) {
        throw new Error(
          'Raw mode is not supported on the current process.stdin, which Ink uses as input stream by default.\nRead about how to prevent this error on https://github.com/vadimdemedes/ink/#israwmodesupported'
        )
      } else {
        throw new Error(
          'Raw mode is not supported on the stdin provided to Ink.\nRead about how to prevent this error on https://github.com/vadimdemedes/ink/#israwmodesupported'
        )
      }
    }

    stdin.setEncoding('utf8')

    if (isEnabled) {
      // Ensure raw mode is enabled only once
      if (this.rawModeEnabledCount === 0) {
        // Stop early input capture right before we add our own readable handler.
        // Both use the same stdin 'readable' + read() pattern, so they can't
        // coexist -- our handler would drain stdin before Ink's can see it.
        // The buffered text is preserved for REPL.tsx via consumeEarlyInput().
        stopCapturingEarlyInput()
        stdin.ref()
        stdin.setRawMode(true)
        stdin.addListener('readable', this.handleReadable)
        // Enable bracketed paste mode
        this.props.stdout.write(EBP)
        // Enable terminal focus reporting (DECSET 1004)
        this.props.stdout.write(EFE)

        // Enable extended key reporting so ctrl+shift+<letter> is
        // distinguishable from ctrl+<letter>. We write both the kitty stack
        // push (CSI >1u) and xterm modifyOtherKeys level 2 (CSI >4;2m) —
        // terminals honor whichever they implement (tmux only accepts the
        // latter).
        if (supportsExtendedKeys()) {
          this.props.stdout.write(ENABLE_KITTY_KEYBOARD)
          this.props.stdout.write(ENABLE_MODIFY_OTHER_KEYS)
        }

        // Probe terminal identity. XTVERSION survives SSH (query/reply goes
        // through the pty), unlike TERM_PROGRAM. Used for wheel-scroll base
        // detection when env vars are absent. Fire-and-forget: the DA1
        // sentinel bounds the round-trip, and if the terminal ignores the
        // query, flush() still resolves and name stays undefined.
        // Deferred to next tick so it fires AFTER the current synchronous
        // init sequence completes — avoids interleaving with alt-screen/mouse
        // tracking enable writes that may happen in the same render cycle.
        setImmediate(() => {
          void Promise.all([this.querier.send(xtversion()), this.querier.flush()]).then(([r]) => {
            if (r) {
              setXtversionName(r.name)
              logForDebugging(`XTVERSION: terminal identified as "${r.name}"`)
            } else {
              logForDebugging('XTVERSION: no reply (terminal ignored query)')
            }
          })
        })
      }

      this.rawModeEnabledCount++

      return
    }

    // Disable raw mode only when no components left that are using it
    if (--this.rawModeEnabledCount === 0) {
      this.props.stdout.write(DISABLE_MODIFY_OTHER_KEYS)
      this.props.stdout.write(DISABLE_KITTY_KEYBOARD)
      // Disable terminal focus reporting (DECSET 1004)
      this.props.stdout.write(DFE)
      // Disable bracketed paste mode
      this.props.stdout.write(DBP)
      stdin.setRawMode(false)
      stdin.removeListener('readable', this.handleReadable)
      stdin.unref()
    }
  }

  // Helper to flush incomplete escape sequences
  flushIncomplete = (): void => {
    // Clear the timer reference
    this.incompleteEscapeTimer = null

    // Only proceed if we have incomplete sequences
    if (!this.keyParseState.incomplete) {
      return
    }

    // Fullscreen: if stdin has data waiting, it's almost certainly the
    // continuation of the buffered sequence (e.g. `[<64;74;16M` after a
    // lone ESC). Node's event loop runs the timers phase before the poll
    // phase, so when a heavy render blocks the loop past 50ms, this timer
    // fires before the queued readable event even though the bytes are
    // already buffered. Re-arm instead of flushing: handleReadable will
    // drain stdin next and clear this timer. Prevents both the spurious
    // Escape key and the lost scroll event.
    if (this.props.stdin.readableLength > 0) {
      this.incompleteEscapeTimer = setTimeout(this.flushIncomplete, this.NORMAL_TIMEOUT)

      return
    }

    // Process incomplete as a flush operation (input=null)
    // This reuses all existing parsing logic
    this.processInput(null)
  }

  // Process input through the parser and handle the results
  processInput = (input: string | Buffer | null): void => {
    // Parse input using our state machine
    const [keys, newState] = parseMultipleKeypresses(this.keyParseState, input)
    this.keyParseState = newState

    // Process ALL keys in a SINGLE discreteUpdates call to prevent
    // "Maximum update depth exceeded" error when many keys arrive at once
    // (e.g., from paste operations or holding keys rapidly).
    // This batches all state updates from handleInput and all useInput
    // listeners together within one high-priority update context.
    if (keys.length > 0) {
      reconciler.discreteUpdates(processKeysInBatch, this, keys, undefined, undefined)
    }

    // If we have incomplete escape sequences, set a timer to flush them
    if (this.keyParseState.incomplete) {
      // Cancel any existing timer first
      if (this.incompleteEscapeTimer) {
        clearTimeout(this.incompleteEscapeTimer)
      }

      this.incompleteEscapeTimer = setTimeout(
        this.flushIncomplete,
        this.keyParseState.mode === 'IN_PASTE' ? this.PASTE_TIMEOUT : this.NORMAL_TIMEOUT
      )
    }
  }
  handleReadable = (): void => {
    // Detect long stdin gaps (tmux attach, ssh reconnect, laptop wake).
    // The terminal may have reset DEC private modes; re-assert mouse
    // tracking. Checked before the read loop so one Date.now() covers
    // all chunks in this readable event.
    const now = Date.now()

    if (now - this.lastStdinTime > STDIN_RESUME_GAP_MS) {
      this.props.onStdinResume?.()
    }

    this.lastStdinTime = now

    try {
      let chunk

      while ((chunk = this.props.stdin.read() as string | null) !== null) {
        // Process the input chunk
        this.processInput(chunk)
      }
    } catch (error) {
      // In Bun, an uncaught throw inside a stream 'readable' handler can
      // permanently wedge the stream: data stays buffered and 'readable'
      // never re-emits. Catching here ensures the stream stays healthy so
      // subsequent keystrokes are still delivered.
      logError(error)

      // Re-attach the listener in case the exception detached it.
      // Bun may remove the listener after an error; without this,
      // the session freezes permanently (stdin reader dead, event loop alive).
      const { stdin } = this.props

      if (this.rawModeEnabledCount > 0 && !stdin.listeners('readable').includes(this.handleReadable)) {
        logForDebugging('handleReadable: re-attaching stdin readable listener after error recovery', {
          level: 'warn'
        })
        stdin.addListener('readable', this.handleReadable)
      }
    }
  }
  handleInput = (input: string | undefined): void => {
    // Exit on Ctrl+C
    if (input === '\x03' && this.props.exitOnCtrlC) {
      this.handleExit()
    }

    // Note: Ctrl+Z (suspend) is now handled in processKeysInBatch using the
    // parsed key to support both raw (\x1a) and CSI u format from Kitty
    // keyboard protocol terminals (Ghostty, iTerm2, kitty, WezTerm)
  }
  handleExit = (error?: Error): void => {
    if (this.isRawModeSupported()) {
      this.handleSetRawMode(false)
    }

    this.props.onExit(error)
  }
  handleTerminalFocus = (isFocused: boolean): void => {
    // setTerminalFocused notifies subscribers: TerminalFocusProvider (context)
    // and Clock (interval speed) — no App setState needed.
    setTerminalFocused(isFocused)
  }
  handleSuspend = (): void => {
    if (!this.isRawModeSupported()) {
      return
    }

    // Store the exact raw mode count to restore it properly
    const rawModeCountBeforeSuspend = this.rawModeEnabledCount

    // Completely disable raw mode before suspending
    while (this.rawModeEnabledCount > 0) {
      this.handleSetRawMode(false)
    }

    // Show cursor, disable focus reporting, and disable mouse tracking
    // before suspending. DISABLE_MOUSE_TRACKING is a no-op if tracking
    // wasn't enabled, so it's safe to emit unconditionally — without
    // it, SGR mouse sequences would appear as garbled text at the
    // shell prompt while suspended.
    if (this.props.stdout.isTTY) {
      this.props.stdout.write(SHOW_CURSOR + DFE + DISABLE_MOUSE_TRACKING)
    }

    this.inputEmitter.emit('suspend')

    // Set up resume handler
    const resumeHandler = () => {
      // Restore raw mode to exact previous state
      for (let i = 0; i < rawModeCountBeforeSuspend; i++) {
        if (this.isRawModeSupported()) {
          this.handleSetRawMode(true)
        }
      }

      if (this.props.stdout.isTTY) {
        this.props.stdout.write(HIDE_CURSOR + EFE)
      }

      this.inputEmitter.emit('resume')
      process.removeListener('SIGCONT', resumeHandler)
    }

    process.on('SIGCONT', resumeHandler)
    process.kill(process.pid, 'SIGSTOP')
  }
}

// Helper to process all keys within a single discrete update context.
// discreteUpdates expects (fn, a, b, c, d) -> fn(a, b, c, d)
function processKeysInBatch(app: App, items: ParsedInput[], _unused1: undefined, _unused2: undefined): void {
  // Update interaction time for notification timeout tracking.
  // This is called from the central input handler to avoid having multiple
  // stdin listeners that can cause race conditions and dropped input.
  // Terminal responses (kind: 'response') are automated, not user input.
  // Mode-1003 no-button motion is also excluded — passive cursor drift is
  // not engagement (would suppress idle notifications + defer housekeeping).
  if (
    items.some(i => i.kind === 'key' || (i.kind === 'mouse' && !((i.button & 0x20) !== 0 && (i.button & 0x03) === 3)))
  ) {
    updateLastInteractionTime()
  }

  for (const item of items) {
    // Terminal responses (DECRPM, DA1, OSC replies, etc.) are not user
    // input — route them to the querier to resolve pending promises.
    if (item.kind === 'response') {
      app.querier.onResponse(item.response)

      continue
    }

    // Mouse click/drag events update selection state (fullscreen only).
    // Terminal sends 1-indexed col/row; convert to 0-indexed for the
    // screen buffer. Button bit 0x20 = drag (motion while button held).
    if (item.kind === 'mouse') {
      handleMouseEvent(app, item)

      continue
    }

    const sequence = item.sequence

    // Handle terminal focus events (DECSET 1004)
    if (sequence === FOCUS_IN) {
      app.handleTerminalFocus(true)
      const event = new TerminalFocusEvent('terminalfocus')
      app.inputEmitter.emit('terminalfocus', event)

      continue
    }

    if (sequence === FOCUS_OUT) {
      app.handleTerminalFocus(false)

      // Defensive: if we lost the release event (mouse released outside
      // terminal window — some emulators drop it rather than capturing the
      // pointer), focus-out is the next observable signal that the drag is
      // over. Without this, drag-to-scroll's timer runs until the scroll
      // boundary is hit.
      if (app.props.selection.isDragging) {
        finishSelection(app.props.selection)
        app.props.onSelectionChange()
      }

      const event = new TerminalFocusEvent('terminalblur')
      app.inputEmitter.emit('terminalblur', event)

      continue
    }

    // Failsafe: if we receive input, the terminal must be focused
    if (!getTerminalFocused()) {
      setTerminalFocused(true)
    }

    // Handle Ctrl+Z (suspend) using parsed key to support both raw (\x1a) and
    // CSI u format (\x1b[122;5u) from Kitty keyboard protocol terminals
    if (item.name === 'z' && item.ctrl && SUPPORTS_SUSPEND) {
      app.handleSuspend()

      continue
    }

    app.handleInput(sequence)
    const event = new InputEvent(item)
    app.inputEmitter.emit('input', event)

    // Also dispatch through the DOM tree so onKeyDown handlers fire.
    app.props.dispatchKeyboardEvent(item)
  }
}

/** Exported for testing. Mutates app.props.selection and click/hover state. */
export function handleMouseEvent(app: App, m: ParsedMouse): void {
  // Allow disabling click handling while keeping wheel scroll (which goes
  // through the keybinding system as 'wheelup'/'wheeldown', not here).
  if (isMouseClicksDisabled()) {
    return
  }

  const sel = app.props.selection
  // Terminal coords are 1-indexed; screen buffer is 0-indexed
  const col = m.col - 1
  const row = m.row - 1
  const baseButton = m.button & 0x03

  if (m.action === 'press') {
    if ((m.button & 0x20) !== 0 && baseButton === 3) {
      if (app.mouseCaptureTarget) {
        app.props.onMouseUpAt(app.mouseCaptureTarget, col, row, baseButton)
        app.mouseCaptureTarget = undefined
      }

      // Mode-1003 motion with no button held. Dispatch hover; skip the
      // rest of this handler (no selection, no click-count side effects).
      // Lost-release recovery: no-button motion while isDragging=true means
      // the release happened outside the terminal window (iTerm2 doesn't
      // capture the pointer past window bounds, so the SGR 'm' never
      // arrives). Finish the selection here so copy-on-select fires. The
      // FOCUS_OUT handler covers the "switched apps" case but not "released
      // past the edge, came back" — and tmux drops focus events unless
      // `focus-events on` is set, so this is the more reliable signal.
      if (sel.isDragging) {
        finishSelection(sel)
        app.props.onSelectionChange()
      }

      if (col === app.lastHoverCol && row === app.lastHoverRow) {
        return
      }

      app.lastHoverCol = col
      app.lastHoverRow = row
      app.props.onHoverAt(col, row)

      return
    }

    if (baseButton !== 0) {
      // Non-left press breaks the multi-click chain.
      app.clickCount = 0

      // Forward middle/right button presses to the DOM so components can
      // react (e.g. right-click-to-paste on input fields). Middle/right
      // don't participate in selection, multi-click, or hyperlink UX, so
      // we just dispatch and exit without setting mouseCaptureTarget —
      // the matching release (if any) falls through the release path and
      // is ignored there because baseButton !== 0 && !sel.isDragging.
      app.props.onMouseDownAt(col, row, baseButton)

      return
    }

    if ((m.button & 0x20) !== 0) {
      if (app.mouseCaptureTarget) {
        app.props.onMouseDragAt(app.mouseCaptureTarget, col, row, baseButton)

        return
      }

      // Drag motion: mode-aware extension (char/word/line). onSelectionDrag
      // calls notifySelectionChange internally — no extra onSelectionChange.
      app.props.onSelectionDrag(col, row)

      return
    }

    // Lost-release fallback for mode-1002-only terminals: a fresh press
    // while isDragging=true means the previous release was dropped (cursor
    // left the window). Finish that selection so copy-on-select fires
    // before startSelection/onMultiClick clobbers it. Mode-1003 terminals
    // hit the no-button-motion recovery above instead, so this is rare.
    if (sel.isDragging) {
      finishSelection(sel)
      app.props.onSelectionChange()
    }

    const capture = app.props.onMouseDownAt(col, row, baseButton)

    if (capture) {
      app.mouseCaptureTarget = capture
      app.clickCount = 0

      return
    }

    // Fresh left press. Detect multi-click HERE (not on release) so the
    // word/line highlight appears immediately and a subsequent drag can
    // extend by word/line like native macOS. Previously detected on
    // release, which meant (a) visible latency before the word highlights
    // and (b) double-click+drag fell through to char-mode selection.
    const now = Date.now()

    const nearLast =
      now - app.lastClickTime < MULTI_CLICK_TIMEOUT_MS &&
      Math.abs(col - app.lastClickCol) <= MULTI_CLICK_DISTANCE &&
      Math.abs(row - app.lastClickRow) <= MULTI_CLICK_DISTANCE

    app.clickCount = nearLast ? app.clickCount + 1 : 1
    app.lastClickTime = now
    app.lastClickCol = col
    app.lastClickRow = row

    if (app.clickCount >= 2) {
      // Cancel any pending hyperlink-open from the first click — this is
      // a double-click, not a single-click on a link.
      if (app.pendingHyperlinkTimer) {
        clearTimeout(app.pendingHyperlinkTimer)
        app.pendingHyperlinkTimer = null
      }

      // Cap at 3 (line select) for quadruple+ clicks.
      const count = app.clickCount === 2 ? 2 : 3
      app.props.onMultiClick(col, row, count)

      return
    }

    startSelection(sel, col, row)
    // SGR bit 0x08 = alt (xterm.js wires altKey here, not metaKey — see
    // comment at the hyperlink-open guard below). On macOS xterm.js,
    // receiving alt means macOptionClickForcesSelection is OFF (otherwise
    // xterm.js would have consumed the event for native selection).
    sel.lastPressHadAlt = (m.button & 0x08) !== 0
    app.props.onSelectionChange()

    return
  }

  // Release: end the drag even for non-zero button codes. Some terminals
  // encode release with the motion bit or button=3 "no button" (carried
  // over from pre-SGR X10 encoding) — filtering those would orphan
  // isDragging=true and leave drag-to-scroll's timer running until the
  // scroll boundary. Only act on non-left releases when we ARE dragging
  // (so an unrelated middle/right click-release doesn't touch selection).
  if (app.mouseCaptureTarget) {
    app.props.onMouseUpAt(app.mouseCaptureTarget, col, row, baseButton)
    app.mouseCaptureTarget = undefined

    return
  }

  if (baseButton !== 0) {
    if (!sel.isDragging) {
      return
    }

    finishSelection(sel)
    app.props.onSelectionChange()

    return
  }

  finishSelection(sel)

  // NOTE: unlike the old release-based detection we do NOT reset clickCount
  // on release-after-drag. This aligns with NSEvent.clickCount semantics:
  // an intervening drag doesn't break the click chain. Practical upside:
  // trackpad jitter during an intended double-click (press→wobble→release
  // →press) now correctly resolves to word-select instead of breaking to a
  // fresh single click. The nearLast window (500ms, 1 cell) bounds the
  // effect — a deliberate drag past that just starts a fresh chain.
  // A press+release with no drag in char mode is a click: anchor set,
  // focus null → hasSelection false. In word/line mode the press already
  // set anchor+focus (hasSelection true), so release just keeps the
  // highlight. The anchor check guards against an orphaned release (no
  // prior press — e.g. button was held when mouse tracking was enabled).
  if (!hasSelection(sel) && sel.anchor) {
    // Single click: dispatch DOM click immediately (cursor repositioning
    // etc. are latency-sensitive). If no DOM handler consumed it, defer
    // the hyperlink check so a second click can cancel it.
    if (!app.props.onClickAt(col, row)) {
      // Resolve the hyperlink URL synchronously while the screen buffer
      // still reflects what the user clicked — deferring only the
      // browser-open so double-click can cancel it.
      const url = app.props.getHyperlinkAt(col, row)

      // xterm.js (VS Code, Cursor, Windsurf, etc.) has its own OSC 8 link
      // handler that fires on Cmd+click *without consuming the mouse event*
      // (Linkifier._handleMouseUp calls link.activate() but never
      // preventDefault/stopPropagation). The click is also forwarded to the
      // pty as SGR, so both VS Code's terminalLinkManager AND our handler
      // here would open the URL — twice. We can't filter on Cmd: xterm.js
      // drops metaKey before SGR encoding (ICoreMouseEvent has no meta
      // field; the SGR bit we call 'meta' is wired to alt). Let xterm.js
      // own link-opening; Cmd+click is the native UX there anyway.
      // TERM_PROGRAM is the sync fast-path; isXtermJs() is the XTVERSION
      // probe result (catches SSH + non-VS Code embedders like Hyper).
      if (url && process.env.TERM_PROGRAM !== 'vscode' && !isXtermJs()) {
        // Clear any prior pending timer — clicking a second link
        // supersedes the first (only the latest click opens).
        if (app.pendingHyperlinkTimer) {
          clearTimeout(app.pendingHyperlinkTimer)
        }

        app.pendingHyperlinkTimer = setTimeout(
          (app, url) => {
            app.pendingHyperlinkTimer = null
            app.props.onOpenHyperlink(url)
          },
          MULTI_CLICK_TIMEOUT_MS,
          app,
          url
        )
      }
    }
  }

  app.props.onSelectionChange()
}
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJuYW1lcyI6WyJSZWFjdCIsIlB1cmVDb21wb25lbnQiLCJSZWFjdE5vZGUiLCJ1cGRhdGVMYXN0SW50ZXJhY3Rpb25UaW1lIiwibG9nRm9yRGVidWdnaW5nIiwic3RvcENhcHR1cmluZ0Vhcmx5SW5wdXQiLCJpc0VudlRydXRoeSIsImlzTW91c2VDbGlja3NEaXNhYmxlZCIsImxvZ0Vycm9yIiwiRXZlbnRFbWl0dGVyIiwiSW5wdXRFdmVudCIsIlRlcm1pbmFsRm9jdXNFdmVudCIsIklOSVRJQUxfU1RBVEUiLCJQYXJzZWRJbnB1dCIsIlBhcnNlZEtleSIsIlBhcnNlZE1vdXNlIiwicGFyc2VNdWx0aXBsZUtleXByZXNzZXMiLCJyZWNvbmNpbGVyIiwiZmluaXNoU2VsZWN0aW9uIiwiaGFzU2VsZWN0aW9uIiwiU2VsZWN0aW9uU3RhdGUiLCJzdGFydFNlbGVjdGlvbiIsImlzWHRlcm1KcyIsInNldFh0dmVyc2lvbk5hbWUiLCJzdXBwb3J0c0V4dGVuZGVkS2V5cyIsImdldFRlcm1pbmFsRm9jdXNlZCIsInNldFRlcm1pbmFsRm9jdXNlZCIsIlRlcm1pbmFsUXVlcmllciIsInh0dmVyc2lvbiIsIkRJU0FCTEVfS0lUVFlfS0VZQk9BUkQiLCJESVNBQkxFX01PRElGWV9PVEhFUl9LRVlTIiwiRU5BQkxFX0tJVFRZX0tFWUJPQVJEIiwiRU5BQkxFX01PRElGWV9PVEhFUl9LRVlTIiwiRk9DVVNfSU4iLCJGT0NVU19PVVQiLCJEQlAiLCJERkUiLCJESVNBQkxFX01PVVNFX1RSQUNLSU5HIiwiRUJQIiwiRUZFIiwiSElERV9DVVJTT1IiLCJTSE9XX0NVUlNPUiIsIkFwcENvbnRleHQiLCJDbG9ja1Byb3ZpZGVyIiwiQ3Vyc29yRGVjbGFyYXRpb25Db250ZXh0IiwiQ3Vyc29yRGVjbGFyYXRpb25TZXR0ZXIiLCJFcnJvck92ZXJ2aWV3IiwiU3RkaW5Db250ZXh0IiwiVGVybWluYWxGb2N1c1Byb3ZpZGVyIiwiVGVybWluYWxTaXplQ29udGV4dCIsIlNVUFBPUlRTX1NVU1BFTkQiLCJwcm9jZXNzIiwicGxhdGZvcm0iLCJTVERJTl9SRVNVTUVfR0FQX01TIiwiUHJvcHMiLCJjaGlsZHJlbiIsInN0ZGluIiwiTm9kZUpTIiwiUmVhZFN0cmVhbSIsInN0ZG91dCIsIldyaXRlU3RyZWFtIiwic3RkZXJyIiwiZXhpdE9uQ3RybEMiLCJvbkV4aXQiLCJlcnJvciIsIkVycm9yIiwidGVybWluYWxDb2x1bW5zIiwidGVybWluYWxSb3dzIiwic2VsZWN0aW9uIiwib25TZWxlY3Rpb25DaGFuZ2UiLCJvbkNsaWNrQXQiLCJjb2wiLCJyb3ciLCJvbkhvdmVyQXQiLCJnZXRIeXBlcmxpbmtBdCIsIm9uT3Blbkh5cGVybGluayIsInVybCIsIm9uTXVsdGlDbGljayIsImNvdW50Iiwib25TZWxlY3Rpb25EcmFnIiwib25TdGRpblJlc3VtZSIsIm9uQ3Vyc29yRGVjbGFyYXRpb24iLCJkaXNwYXRjaEtleWJvYXJkRXZlbnQiLCJwYXJzZWRLZXkiLCJNVUxUSV9DTElDS19USU1FT1VUX01TIiwiTVVMVElfQ0xJQ0tfRElTVEFOQ0UiLCJTdGF0ZSIsIkFwcCIsImRpc3BsYXlOYW1lIiwiZ2V0RGVyaXZlZFN0YXRlRnJvbUVycm9yIiwic3RhdGUiLCJ1bmRlZmluZWQiLCJyYXdNb2RlRW5hYmxlZENvdW50IiwiaW50ZXJuYWxfZXZlbnRFbWl0dGVyIiwia2V5UGFyc2VTdGF0ZSIsImluY29tcGxldGVFc2NhcGVUaW1lciIsIlRpbWVvdXQiLCJOT1JNQUxfVElNRU9VVCIsIlBBU1RFX1RJTUVPVVQiLCJxdWVyaWVyIiwicHJvcHMiLCJsYXN0Q2xpY2tUaW1lIiwibGFzdENsaWNrQ29sIiwibGFzdENsaWNrUm93IiwiY2xpY2tDb3VudCIsInBlbmRpbmdIeXBlcmxpbmtUaW1lciIsIlJldHVyblR5cGUiLCJzZXRUaW1lb3V0IiwibGFzdEhvdmVyQ29sIiwibGFzdEhvdmVyUm93IiwibGFzdFN0ZGluVGltZSIsIkRhdGUiLCJub3ciLCJpc1Jhd01vZGVTdXBwb3J0ZWQiLCJpc1RUWSIsInJlbmRlciIsImNvbHVtbnMiLCJyb3dzIiwiZXhpdCIsImhhbmRsZUV4aXQiLCJzZXRSYXdNb2RlIiwiaGFuZGxlU2V0UmF3TW9kZSIsImludGVybmFsX2V4aXRPbkN0cmxDIiwiaW50ZXJuYWxfcXVlcmllciIsImNvbXBvbmVudERpZE1vdW50IiwiZW52IiwiQ0xBVURFX0NPREVfQUNDRVNTSUJJTElUWSIsIndyaXRlIiwiY29tcG9uZW50V2lsbFVubW91bnQiLCJjbGVhclRpbWVvdXQiLCJjb21wb25lbnREaWRDYXRjaCIsImlzRW5hYmxlZCIsInNldEVuY29kaW5nIiwicmVmIiwiYWRkTGlzdGVuZXIiLCJoYW5kbGVSZWFkYWJsZSIsInNldEltbWVkaWF0ZSIsIlByb21pc2UiLCJhbGwiLCJzZW5kIiwiZmx1c2giLCJ0aGVuIiwiciIsIm5hbWUiLCJyZW1vdmVMaXN0ZW5lciIsInVucmVmIiwiZmx1c2hJbmNvbXBsZXRlIiwiaW5jb21wbGV0ZSIsInJlYWRhYmxlTGVuZ3RoIiwicHJvY2Vzc0lucHV0IiwiaW5wdXQiLCJCdWZmZXIiLCJrZXlzIiwibmV3U3RhdGUiLCJsZW5ndGgiLCJkaXNjcmV0ZVVwZGF0ZXMiLCJwcm9jZXNzS2V5c0luQmF0Y2giLCJtb2RlIiwiY2h1bmsiLCJyZWFkIiwibGlzdGVuZXJzIiwiaW5jbHVkZXMiLCJsZXZlbCIsImhhbmRsZUlucHV0IiwiaGFuZGxlVGVybWluYWxGb2N1cyIsImlzRm9jdXNlZCIsImhhbmRsZVN1c3BlbmQiLCJyYXdNb2RlQ291bnRCZWZvcmVTdXNwZW5kIiwiZW1pdCIsInJlc3VtZUhhbmRsZXIiLCJpIiwib24iLCJraWxsIiwicGlkIiwiYXBwIiwiaXRlbXMiLCJfdW51c2VkMSIsIl91bnVzZWQyIiwic29tZSIsImtpbmQiLCJidXR0b24iLCJpdGVtIiwib25SZXNwb25zZSIsInJlc3BvbnNlIiwiaGFuZGxlTW91c2VFdmVudCIsInNlcXVlbmNlIiwiZXZlbnQiLCJpc0RyYWdnaW5nIiwiY3RybCIsIm0iLCJzZWwiLCJiYXNlQnV0dG9uIiwiYWN0aW9uIiwibmVhckxhc3QiLCJNYXRoIiwiYWJzIiwibGFzdFByZXNzSGFkQWx0IiwiYW5jaG9yIiwiVEVSTV9QUk9HUkFNIl0sInNvdXJjZXMiOlsiQXBwLnRzeCJdLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgUHVyZUNvbXBvbmVudCwgdHlwZSBSZWFjdE5vZGUgfSBmcm9tICdyZWFjdCdcbmltcG9ydCB7IHVwZGF0ZUxhc3RJbnRlcmFjdGlvblRpbWUgfSBmcm9tICcuLi8uLi9ib290c3RyYXAvc3RhdGUuanMnXG5pbXBvcnQgeyBsb2dGb3JEZWJ1Z2dpbmcgfSBmcm9tICcuLi8uLi91dGlscy9kZWJ1Zy5qcydcbmltcG9ydCB7IHN0b3BDYXB0dXJpbmdFYXJseUlucHV0IH0gZnJvbSAnLi4vLi4vdXRpbHMvZWFybHlJbnB1dC5qcydcbmltcG9ydCB7IGlzRW52VHJ1dGh5IH0gZnJvbSAnLi4vLi4vdXRpbHMvZW52VXRpbHMuanMnXG5pbXBvcnQgeyBpc01vdXNlQ2xpY2tzRGlzYWJsZWQgfSBmcm9tICcuLi8uLi91dGlscy9mdWxsc2NyZWVuLmpzJ1xuaW1wb3J0IHsgbG9nRXJyb3IgfSBmcm9tICcuLi8uLi91dGlscy9sb2cuanMnXG5pbXBvcnQgeyBFdmVudEVtaXR0ZXIgfSBmcm9tICcuLi9ldmVudHMvZW1pdHRlci5qcydcbmltcG9ydCB7IElucHV0RXZlbnQgfSBmcm9tICcuLi9ldmVudHMvaW5wdXQtZXZlbnQuanMnXG5pbXBvcnQgeyBUZXJtaW5hbEZvY3VzRXZlbnQgfSBmcm9tICcuLi9ldmVudHMvdGVybWluYWwtZm9jdXMtZXZlbnQuanMnXG5pbXBvcnQge1xuICBJTklUSUFMX1NUQVRFLFxuICB0eXBlIFBhcnNlZElucHV0LFxuICB0eXBlIFBhcnNlZEtleSxcbiAgdHlwZSBQYXJzZWRNb3VzZSxcbiAgcGFyc2VNdWx0aXBsZUtleXByZXNzZXMsXG59IGZyb20gJy4uL3BhcnNlLWtleXByZXNzLmpzJ1xuaW1wb3J0IHJlY29uY2lsZXIgZnJvbSAnLi4vcmVjb25jaWxlci5qcydcbmltcG9ydCB7XG4gIGZpbmlzaFNlbGVjdGlvbixcbiAgaGFzU2VsZWN0aW9uLFxuICB0eXBlIFNlbGVjdGlvblN0YXRlLFxuICBzdGFydFNlbGVjdGlvbixcbn0gZnJvbSAnLi4vc2VsZWN0aW9uLmpzJ1xuaW1wb3J0IHtcbiAgaXNYdGVybUpzLFxuICBzZXRYdHZlcnNpb25OYW1lLFxuICBzdXBwb3J0c0V4dGVuZGVkS2V5cyxcbn0gZnJvbSAnLi4vdGVybWluYWwuanMnXG5pbXBvcnQge1xuICBnZXRUZXJtaW5hbEZvY3VzZWQsXG4gIHNldFRlcm1pbmFsRm9jdXNlZCxcbn0gZnJvbSAnLi4vdGVybWluYWwtZm9jdXMtc3RhdGUuanMnXG5pbXBvcnQgeyBUZXJtaW5hbFF1ZXJpZXIsIHh0dmVyc2lvbiB9IGZyb20gJy4uL3Rlcm1pbmFsLXF1ZXJpZXIuanMnXG5pbXBvcnQge1xuICBESVNBQkxFX0tJVFRZX0tFWUJPQVJELFxuICBESVNBQkxFX01PRElGWV9PVEhFUl9LRVlTLFxuICBFTkFCTEVfS0lUVFlfS0VZQk9BUkQsXG4gIEVOQUJMRV9NT0RJRllfT1RIRVJfS0VZUyxcbiAgRk9DVVNfSU4sXG4gIEZPQ1VTX09VVCxcbn0gZnJvbSAnLi4vdGVybWlvL2NzaS5qcydcbmltcG9ydCB7XG4gIERCUCxcbiAgREZFLFxuICBESVNBQkxFX01PVVNFX1RSQUNLSU5HLFxuICBFQlAsXG4gIEVGRSxcbiAgSElERV9DVVJTT1IsXG4gIFNIT1dfQ1VSU09SLFxufSBmcm9tICcuLi90ZXJtaW8vZGVjLmpzJ1xuaW1wb3J0IEFwcENvbnRleHQgZnJvbSAnLi9BcHBDb250ZXh0LmpzJ1xuaW1wb3J0IHsgQ2xvY2tQcm92aWRlciB9IGZyb20gJy4vQ2xvY2tDb250ZXh0LmpzJ1xuaW1wb3J0IEN1cnNvckRlY2xhcmF0aW9uQ29udGV4dCwge1xuICB0eXBlIEN1cnNvckRlY2xhcmF0aW9uU2V0dGVyLFxufSBmcm9tICcuL0N1cnNvckRlY2xhcmF0aW9uQ29udGV4dC5qcydcbmltcG9ydCBFcnJvck92ZXJ2aWV3IGZyb20gJy4vRXJyb3JPdmVydmlldy5qcydcbmltcG9ydCBTdGRpbkNvbnRleHQgZnJvbSAnLi9TdGRpbkNvbnRleHQuanMnXG5pbXBvcnQgeyBUZXJtaW5hbEZvY3VzUHJvdmlkZXIgfSBmcm9tICcuL1Rlcm1pbmFsRm9jdXNDb250ZXh0LmpzJ1xuaW1wb3J0IHsgVGVybWluYWxTaXplQ29udGV4dCB9IGZyb20gJy4vVGVybWluYWxTaXplQ29udGV4dC5qcydcblxuLy8gUGxhdGZvcm1zIHRoYXQgc3VwcG9ydCBVbml4LXN0eWxlIHByb2Nlc3Mgc3VzcGVuc2lvbiAoU0lHU1RPUC9TSUdDT05UKVxuY29uc3QgU1VQUE9SVFNfU1VTUEVORCA9IHByb2Nlc3MucGxhdGZvcm0gIT09ICd3aW4zMidcblxuLy8gQWZ0ZXIgdGhpcyBtYW55IG1pbGxpc2Vjb25kcyBvZiBzdGRpbiBzaWxlbmNlLCB0aGUgbmV4dCBjaHVuayB0cmlnZ2Vyc1xuLy8gYSB0ZXJtaW5hbCBtb2RlIHJlLWFzc2VydCAobW91c2UgdHJhY2tpbmcpLiBDYXRjaGVzIHRtdXggZGV0YWNo4oaSYXR0YWNoLFxuLy8gc3NoIHJlY29ubmVjdCwgYW5kIGxhcHRvcCB3YWtlIOKAlCB0aGUgdGVybWluYWwgcmVzZXRzIERFQyBwcml2YXRlIG1vZGVzXG4vLyBidXQgbm8gc2lnbmFsIHJlYWNoZXMgdXMuIDVzIGlzIHdlbGwgYWJvdmUgbm9ybWFsIGludGVyLWtleXN0cm9rZSBnYXBzXG4vLyBidXQgc2hvcnQgZW5vdWdoIHRoYXQgdGhlIGZpcnN0IHNjcm9sbCBhZnRlciByZWF0dGFjaCB3b3Jrcy5cbmNvbnN0IFNURElOX1JFU1VNRV9HQVBfTVMgPSA1MDAwXG5cbnR5cGUgUHJvcHMgPSB7XG4gIHJlYWRvbmx5IGNoaWxkcmVuOiBSZWFjdE5vZGVcbiAgcmVhZG9ubHkgc3RkaW46IE5vZGVKUy5SZWFkU3RyZWFtXG4gIHJlYWRvbmx5IHN0ZG91dDogTm9kZUpTLldyaXRlU3RyZWFtXG4gIHJlYWRvbmx5IHN0ZGVycjogTm9kZUpTLldyaXRlU3RyZWFtXG4gIHJlYWRvbmx5IGV4aXRPbkN0cmxDOiBib29sZWFuXG4gIHJlYWRvbmx5IG9uRXhpdDogKGVycm9yPzogRXJyb3IpID0+IHZvaWRcbiAgcmVhZG9ubHkgdGVybWluYWxDb2x1bW5zOiBudW1iZXJcbiAgcmVhZG9ubHkgdGVybWluYWxSb3dzOiBudW1iZXJcbiAgLy8gVGV4dCBzZWxlY3Rpb24gc3RhdGUuIEFwcCBtdXRhdGVzIHRoaXMgZGlyZWN0bHkgZnJvbSBtb3VzZSBldmVudHNcbiAgLy8gYW5kIGNhbGxzIG9uU2VsZWN0aW9uQ2hhbmdlIHRvIHRyaWdnZXIgYSByZXBhaW50LiBNb3VzZSBldmVudHMgb25seVxuICAvLyBhcnJpdmUgd2hlbiA8QWx0ZXJuYXRlU2NyZWVuPiAob3Igc2ltaWxhcikgZW5hYmxlcyBtb3VzZSB0cmFja2luZyxcbiAgLy8gc28gdGhlIGhhbmRsZXIgaXMgYWx3YXlzIHdpcmVkIGJ1dCBkb3JtYW50IHVudGlsIHRyYWNraW5nIGlzIG9uLlxuICByZWFkb25seSBzZWxlY3Rpb246IFNlbGVjdGlvblN0YXRlXG4gIHJlYWRvbmx5IG9uU2VsZWN0aW9uQ2hhbmdlOiAoKSA9PiB2b2lkXG4gIC8vIERpc3BhdGNoIGEgY2xpY2sgYXQgKGNvbCwgcm93KSDigJQgaGl0LXRlc3RzIHRoZSBET00gdHJlZSBhbmQgYnViYmxlc1xuICAvLyBvbkNsaWNrIGhhbmRsZXJzLiBSZXR1cm5zIHRydWUgaWYgYSBET00gaGFuZGxlciBjb25zdW1lZCB0aGUgY2xpY2suXG4gIC8vIE5vLW9wIChyZXR1cm5zIGZhbHNlKSBvdXRzaWRlIGZ1bGxzY3JlZW4gbW9kZSAoSW5rLmRpc3BhdGNoQ2xpY2tcbiAgLy8gZ2F0ZXMgb24gYWx0U2NyZWVuQWN0aXZlKS5cbiAgcmVhZG9ubHkgb25DbGlja0F0OiAoY29sOiBudW1iZXIsIHJvdzogbnVtYmVyKSA9PiBib29sZWFuXG4gIC8vIERpc3BhdGNoIGhvdmVyIChvbk1vdXNlRW50ZXIvb25Nb3VzZUxlYXZlKSBhcyB0aGUgcG9pbnRlciBtb3ZlcyBvdmVyXG4gIC8vIERPTSBlbGVtZW50cy4gQ2FsbGVkIGZvciBtb2RlLTEwMDMgbW90aW9uIGV2ZW50cyB3aXRoIG5vIGJ1dHRvbiBoZWxkLlxuICAvLyBOby1vcCBvdXRzaWRlIGZ1bGxzY3JlZW4gKEluay5kaXNwYXRjaEhvdmVyIGdhdGVzIG9uIGFsdFNjcmVlbkFjdGl2ZSkuXG4gIHJlYWRvbmx5IG9uSG92ZXJBdDogKGNvbDogbnVtYmVyLCByb3c6IG51bWJlcikgPT4gdm9pZFxuICAvLyBMb29rIHVwIHRoZSBPU0MgOCBoeXBlcmxpbmsgYXQgKGNvbCwgcm93KSBzeW5jaHJvbm91c2x5IGF0IGNsaWNrXG4gIC8vIHRpbWUuIFJldHVybnMgdGhlIFVSTCBvciB1bmRlZmluZWQuIFRoZSBicm93c2VyLW9wZW4gaXMgZGVmZXJyZWQgYnlcbiAgLy8gTVVMVElfQ0xJQ0tfVElNRU9VVF9NUyBzbyBkb3VibGUtY2xpY2sgY2FuIGNhbmNlbCBpdC5cbiAgcmVhZG9ubHkgZ2V0SHlwZXJsaW5rQXQ6IChjb2w6IG51bWJlciwgcm93OiBudW1iZXIpID0+IHN0cmluZyB8IHVuZGVmaW5lZFxuICAvLyBPcGVuIGEgaHlwZXJsaW5rIFVSTCBpbiB0aGUgYnJvd3Nlci4gQ2FsbGVkIGFmdGVyIHRoZSB0aW1lciBmaXJlcy5cbiAgcmVhZG9ubHkgb25PcGVuSHlwZXJsaW5rOiAodXJsOiBzdHJpbmcpID0+IHZvaWRcbiAgLy8gQ2FsbGVkIG9uIGRvdWJsZS90cmlwbGUtY2xpY2sgUFJFU1MgYXQgKGNvbCwgcm93KS4gY291bnQ9MiBzZWxlY3RzXG4gIC8vIHRoZSB3b3JkIHVuZGVyIHRoZSBjdXJzb3I7IGNvdW50PTMgc2VsZWN0cyB0aGUgbGluZS4gSW5rIHJlYWRzIHRoZVxuICAvLyBzY3JlZW4gYnVmZmVyIHRvIGZpbmQgd29yZC9saW5lIGJvdW5kYXJpZXMgYW5kIG11dGF0ZXMgc2VsZWN0aW9uLFxuICAvLyBzZXR0aW5nIGlzRHJhZ2dpbmc9dHJ1ZSBzbyBhIHN1YnNlcXVlbnQgZHJhZyBleHRlbmRzIGJ5IHdvcmQvbGluZS5cbiAgcmVhZG9ubHkgb25NdWx0aUNsaWNrOiAoY29sOiBudW1iZXIsIHJvdzogbnVtYmVyLCBjb3VudDogMiB8IDMpID0+IHZvaWRcbiAgLy8gQ2FsbGVkIG9uIGRyYWctbW90aW9uLiBNb2RlLWF3YXJlOiBjaGFyIG1vZGUgdXBkYXRlcyBmb2N1cyB0byB0aGVcbiAgLy8gZXhhY3QgY2VsbDsgd29yZC9saW5lIG1vZGUgc25hcHMgdG8gd29yZC9saW5lIGJvdW5kYXJpZXMuIE5lZWRzXG4gIC8vIHNjcmVlbi1idWZmZXIgYWNjZXNzICh3b3JkIGJvdW5kYXJpZXMpIHNvIGxpdmVzIG9uIEluaywgbm90IGhlcmUuXG4gIHJlYWRvbmx5IG9uU2VsZWN0aW9uRHJhZzogKGNvbDogbnVtYmVyLCByb3c6IG51bWJlcikgPT4gdm9pZFxuICAvLyBDYWxsZWQgd2hlbiBzdGRpbiBkYXRhIGFycml2ZXMgYWZ0ZXIgYSA+U1RESU5fUkVTVU1FX0dBUF9NUyBnYXAuXG4gIC8vIEluayByZS1hc3NlcnRzIHRlcm1pbmFsIG1vZGVzOiBleHRlbmRlZCBrZXkgcmVwb3J0aW5nLCBhbmQgKHdoZW4gaW5cbiAgLy8gZnVsbHNjcmVlbikgcmUtZW50ZXJzIGFsdC1zY3JlZW4gKyBtb3VzZSB0cmFja2luZy4gSWRlbXBvdGVudCBvbiB0aGVcbiAgLy8gdGVybWluYWwgc2lkZS4gT3B0aW9uYWwgc28gdGVzdGluZy50c3ggZG9lc24ndCBuZWVkIHRvIHN0dWIgaXQuXG4gIHJlYWRvbmx5IG9uU3RkaW5SZXN1bWU/OiAoKSA9PiB2b2lkXG4gIC8vIFJlY2VpdmVzIHRoZSBkZWNsYXJlZCBuYXRpdmUtY3Vyc29yIHBvc2l0aW9uIGZyb20gdXNlRGVjbGFyZWRDdXJzb3JcbiAgLy8gc28gaW5rLnRzeCBjYW4gcGFyayB0aGUgdGVybWluYWwgY3Vyc29yIHRoZXJlIGFmdGVyIGVhY2ggZnJhbWUuXG4gIC8vIEVuYWJsZXMgSU1FIGNvbXBvc2l0aW9uIGF0IHRoZSBpbnB1dCBjYXJldCBhbmQgbGV0cyBzY3JlZW4gcmVhZGVycyAvXG4gIC8vIG1hZ25pZmllcnMgdHJhY2sgdGhlIGlucHV0LiBPcHRpb25hbCBzbyB0ZXN0aW5nLnRzeCBkb2Vzbid0IHN0dWIgaXQuXG4gIHJlYWRvbmx5IG9uQ3Vyc29yRGVjbGFyYXRpb24/OiBDdXJzb3JEZWNsYXJhdGlvblNldHRlclxuICAvLyBEaXNwYXRjaCBhIGtleWJvYXJkIGV2ZW50IHRocm91Z2ggdGhlIERPTSB0cmVlLiBDYWxsZWQgZm9yIGVhY2hcbiAgLy8gcGFyc2VkIGtleSBhbG9uZ3NpZGUgdGhlIGxlZ2FjeSBFdmVudEVtaXR0ZXIgcGF0aC5cbiAgcmVhZG9ubHkgZGlzcGF0Y2hLZXlib2FyZEV2ZW50OiAocGFyc2VkS2V5OiBQYXJzZWRLZXkpID0+IHZvaWRcbn1cblxuLy8gTXVsdGktY2xpY2sgZGV0ZWN0aW9uIHRocmVzaG9sZHMuIDUwMG1zIGlzIHRoZSBtYWNPUyBkZWZhdWx0OyBhIHNtYWxsXG4vLyBwb3NpdGlvbiB0b2xlcmFuY2UgYWxsb3dzIGZvciB0cmFja3BhZCBqaXR0ZXIgYmV0d2VlbiBjbGlja3MuXG5jb25zdCBNVUxUSV9DTElDS19USU1FT1VUX01TID0gNTAwXG5jb25zdCBNVUxUSV9DTElDS19ESVNUQU5DRSA9IDFcblxudHlwZSBTdGF0ZSA9IHtcbiAgcmVhZG9ubHkgZXJyb3I/OiBFcnJvclxufVxuXG4vLyBSb290IGNvbXBvbmVudCBmb3IgYWxsIEluayBhcHBzXG4vLyBJdCByZW5kZXJzIHN0ZGluIGFuZCBzdGRvdXQgY29udGV4dHMsIHNvIHRoYXQgY2hpbGRyZW4gY2FuIGFjY2VzcyB0aGVtIGlmIG5lZWRlZFxuLy8gSXQgYWxzbyBoYW5kbGVzIEN0cmwrQyBleGl0aW5nIGFuZCBjdXJzb3IgdmlzaWJpbGl0eVxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgQXBwIGV4dGVuZHMgUHVyZUNvbXBvbmVudDxQcm9wcywgU3RhdGU+IHtcbiAgc3RhdGljIGRpc3BsYXlOYW1lID0gJ0ludGVybmFsQXBwJ1xuXG4gIHN0YXRpYyBnZXREZXJpdmVkU3RhdGVGcm9tRXJyb3IoZXJyb3I6IEVycm9yKSB7XG4gICAgcmV0dXJuIHsgZXJyb3IgfVxuICB9XG5cbiAgb3ZlcnJpZGUgc3RhdGUgPSB7XG4gICAgZXJyb3I6IHVuZGVmaW5lZCxcbiAgfVxuXG4gIC8vIENvdW50IGhvdyBtYW55IGNvbXBvbmVudHMgZW5hYmxlZCByYXcgbW9kZSB0byBhdm9pZCBkaXNhYmxpbmdcbiAgLy8gcmF3IG1vZGUgdW50aWwgYWxsIGNvbXBvbmVudHMgZG9uJ3QgbmVlZCBpdCBhbnltb3JlXG4gIHJhd01vZGVFbmFibGVkQ291bnQgPSAwXG5cbiAgaW50ZXJuYWxfZXZlbnRFbWl0dGVyID0gbmV3IEV2ZW50RW1pdHRlcigpXG4gIGtleVBhcnNlU3RhdGUgPSBJTklUSUFMX1NUQVRFXG4gIC8vIFRpbWVyIGZvciBmbHVzaGluZyBpbmNvbXBsZXRlIGVzY2FwZSBzZXF1ZW5jZXNcbiAgaW5jb21wbGV0ZUVzY2FwZVRpbWVyOiBOb2RlSlMuVGltZW91dCB8IG51bGwgPSBudWxsXG4gIC8vIFRpbWVvdXQgZHVyYXRpb25zIGZvciBpbmNvbXBsZXRlIHNlcXVlbmNlcyAobXMpXG4gIHJlYWRvbmx5IE5PUk1BTF9USU1FT1VUID0gNTAgLy8gU2hvcnQgdGltZW91dCBmb3IgcmVndWxhciBlc2Mgc2VxdWVuY2VzXG4gIHJlYWRvbmx5IFBBU1RFX1RJTUVPVVQgPSA1MDAgLy8gTG9uZ2VyIHRpbWVvdXQgZm9yIHBhc3RlIG9wZXJhdGlvbnNcblxuICAvLyBUZXJtaW5hbCBxdWVyeS9yZXNwb25zZSBkaXNwYXRjaC4gUmVzcG9uc2VzIGFycml2ZSBvbiBzdGRpbiAocGFyc2VkXG4gIC8vIG91dCBieSBwYXJzZS1rZXlwcmVzcykgYW5kIGFyZSByb3V0ZWQgdG8gcGVuZGluZyBwcm9taXNlIHJlc29sdmVycy5cbiAgcXVlcmllciA9IG5ldyBUZXJtaW5hbFF1ZXJpZXIodGhpcy5wcm9wcy5zdGRvdXQpXG5cbiAgLy8gTXVsdGktY2xpY2sgdHJhY2tpbmcgZm9yIGRvdWJsZS90cmlwbGUtY2xpY2sgdGV4dCBzZWxlY3Rpb24uIEEgY2xpY2tcbiAgLy8gd2l0aGluIE1VTFRJX0NMSUNLX1RJTUVPVVRfTVMgYW5kIE1VTFRJX0NMSUNLX0RJU1RBTkNFIG9mIHRoZSBwcmV2aW91c1xuICAvLyBjbGljayBpbmNyZW1lbnRzIGNsaWNrQ291bnQ7IG90aGVyd2lzZSBpdCByZXNldHMgdG8gMS5cbiAgbGFzdENsaWNrVGltZSA9IDBcbiAgbGFzdENsaWNrQ29sID0gLTFcbiAgbGFzdENsaWNrUm93ID0gLTFcbiAgY2xpY2tDb3VudCA9IDBcbiAgLy8gRGVmZXJyZWQgaHlwZXJsaW5rLW9wZW4gdGltZXIg4oCUIGNhbmNlbGxlZCBpZiBhIHNlY29uZCBjbGljayBhcnJpdmVzXG4gIC8vIHdpdGhpbiBNVUxUSV9DTElDS19USU1FT1VUX01TIChzbyBkb3VibGUtY2xpY2tpbmcgYSBoeXBlcmxpbmsgc2VsZWN0c1xuICAvLyB0aGUgd29yZCB3aXRob3V0IGFsc28gb3BlbmluZyB0aGUgYnJvd3NlcikuIERPTSBvbkNsaWNrIGRpc3BhdGNoIGlzXG4gIC8vIE5PVCBkZWZlcnJlZCDigJQgaXQgcmV0dXJucyB0cnVlIGZyb20gb25DbGlja0F0IGFuZCBza2lwcyB0aGlzIHRpbWVyLlxuICBwZW5kaW5nSHlwZXJsaW5rVGltZXI6IFJldHVyblR5cGU8dHlwZW9mIHNldFRpbWVvdXQ+IHwgbnVsbCA9IG51bGxcbiAgLy8gTGFzdCBtb2RlLTEwMDMgbW90aW9uIHBvc2l0aW9uLiBUZXJtaW5hbHMgYWxyZWFkeSBkZWR1cGUgdG8gY2VsbFxuICAvLyBncmFudWxhcml0eSBidXQgdGhpcyBhbHNvIGxldHMgdXMgc2tpcCBkaXNwYXRjaEhvdmVyIGVudGlyZWx5IG9uXG4gIC8vIHJlcGVhdCBldmVudHMgKGRyYWctdGhlbi1yZWxlYXNlIGF0IHNhbWUgY2VsbCwgZXRjLikuXG4gIGxhc3RIb3ZlckNvbCA9IC0xXG4gIGxhc3RIb3ZlclJvdyA9IC0xXG5cbiAgLy8gVGltZXN0YW1wIG9mIGxhc3Qgc3RkaW4gY2h1bmsuIFVzZWQgdG8gZGV0ZWN0IGxvbmcgZ2FwcyAodG11eCBhdHRhY2gsXG4gIC8vIHNzaCByZWNvbm5lY3QsIGxhcHRvcCB3YWtlKSBhbmQgdHJpZ2dlciB0ZXJtaW5hbCBtb2RlIHJlLWFzc2VydC5cbiAgLy8gSW5pdGlhbGl6ZWQgdG8gbm93IHNvIHN0YXJ0dXAgZG9lc24ndCBmYWxzZS10cmlnZ2VyLlxuICBsYXN0U3RkaW5UaW1lID0gRGF0ZS5ub3coKVxuXG4gIC8vIERldGVybWluZXMgaWYgVFRZIGlzIHN1cHBvcnRlZCBvbiB0aGUgcHJvdmlkZWQgc3RkaW5cbiAgaXNSYXdNb2RlU3VwcG9ydGVkKCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiB0aGlzLnByb3BzLnN0ZGluLmlzVFRZXG4gIH1cblxuICBvdmVycmlkZSByZW5kZXIoKSB7XG4gICAgcmV0dXJuIChcbiAgICAgIDxUZXJtaW5hbFNpemVDb250ZXh0LlByb3ZpZGVyXG4gICAgICAgIHZhbHVlPXt7XG4gICAgICAgICAgY29sdW1uczogdGhpcy5wcm9wcy50ZXJtaW5hbENvbHVtbnMsXG4gICAgICAgICAgcm93czogdGhpcy5wcm9wcy50ZXJtaW5hbFJvd3MsXG4gICAgICAgIH19XG4gICAgICA+XG4gICAgICAgIDxBcHBDb250ZXh0LlByb3ZpZGVyXG4gICAgICAgICAgdmFsdWU9e3tcbiAgICAgICAgICAgIGV4aXQ6IHRoaXMuaGFuZGxlRXhpdCxcbiAgICAgICAgICB9fVxuICAgICAgICA+XG4gICAgICAgICAgPFN0ZGluQ29udGV4dC5Qcm92aWRlclxuICAgICAgICAgICAgdmFsdWU9e3tcbiAgICAgICAgICAgICAgc3RkaW46IHRoaXMucHJvcHMuc3RkaW4sXG4gICAgICAgICAgICAgIHNldFJhd01vZGU6IHRoaXMuaGFuZGxlU2V0UmF3TW9kZSxcbiAgICAgICAgICAgICAgaXNSYXdNb2RlU3VwcG9ydGVkOiB0aGlzLmlzUmF3TW9kZVN1cHBvcnRlZCgpLFxuXG4gICAgICAgICAgICAgIGludGVybmFsX2V4aXRPbkN0cmxDOiB0aGlzLnByb3BzLmV4aXRPbkN0cmxDLFxuXG4gICAgICAgICAgICAgIGludGVybmFsX2V2ZW50RW1pdHRlcjogdGhpcy5pbnRlcm5hbF9ldmVudEVtaXR0ZXIsXG4gICAgICAgICAgICAgIGludGVybmFsX3F1ZXJpZXI6IHRoaXMucXVlcmllcixcbiAgICAgICAgICAgIH19XG4gICAgICAgICAgPlxuICAgICAgICAgICAgPFRlcm1pbmFsRm9jdXNQcm92aWRlcj5cbiAgICAgICAgICAgICAgPENsb2NrUHJvdmlkZXI+XG4gICAgICAgICAgICAgICAgPEN1cnNvckRlY2xhcmF0aW9uQ29udGV4dC5Qcm92aWRlclxuICAgICAgICAgICAgICAgICAgdmFsdWU9e3RoaXMucHJvcHMub25DdXJzb3JEZWNsYXJhdGlvbiA/PyAoKCkgPT4ge30pfVxuICAgICAgICAgICAgICAgID5cbiAgICAgICAgICAgICAgICAgIHt0aGlzLnN0YXRlLmVycm9yID8gKFxuICAgICAgICAgICAgICAgICAgICA8RXJyb3JPdmVydmlldyBlcnJvcj17dGhpcy5zdGF0ZS5lcnJvciBhcyBFcnJvcn0gLz5cbiAgICAgICAgICAgICAgICAgICkgOiAoXG4gICAgICAgICAgICAgICAgICAgIHRoaXMucHJvcHMuY2hpbGRyZW5cbiAgICAgICAgICAgICAgICAgICl9XG4gICAgICAgICAgICAgICAgPC9DdXJzb3JEZWNsYXJhdGlvbkNvbnRleHQuUHJvdmlkZXI+XG4gICAgICAgICAgICAgIDwvQ2xvY2tQcm92aWRlcj5cbiAgICAgICAgICAgIDwvVGVybWluYWxGb2N1c1Byb3ZpZGVyPlxuICAgICAgICAgIDwvU3RkaW5Db250ZXh0LlByb3ZpZGVyPlxuICAgICAgICA8L0FwcENvbnRleHQuUHJvdmlkZXI+XG4gICAgICA8L1Rlcm1pbmFsU2l6ZUNvbnRleHQuUHJvdmlkZXI+XG4gICAgKVxuICB9XG5cbiAgb3ZlcnJpZGUgY29tcG9uZW50RGlkTW91bnQoKSB7XG4gICAgLy8gSW4gYWNjZXNzaWJpbGl0eSBtb2RlLCBrZWVwIHRoZSBuYXRpdmUgY3Vyc29yIHZpc2libGUgZm9yIHNjcmVlbiBtYWduaWZpZXJzIGFuZCBvdGhlciB0b29sc1xuICAgIGlmIChcbiAgICAgIHRoaXMucHJvcHMuc3Rkb3V0LmlzVFRZICYmXG4gICAgICAhaXNFbnZUcnV0aHkocHJvY2Vzcy5lbnYuQ0xBVURFX0NPREVfQUNDRVNTSUJJTElUWSlcbiAgICApIHtcbiAgICAgIHRoaXMucHJvcHMuc3Rkb3V0LndyaXRlKEhJREVfQ1VSU09SKVxuICAgIH1cbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXBvbmVudFdpbGxVbm1vdW50KCkge1xuICAgIGlmICh0aGlzLnByb3BzLnN0ZG91dC5pc1RUWSkge1xuICAgICAgdGhpcy5wcm9wcy5zdGRvdXQud3JpdGUoU0hPV19DVVJTT1IpXG4gICAgfVxuXG4gICAgLy8gQ2xlYXIgYW55IHBlbmRpbmcgdGltZXJzXG4gICAgaWYgKHRoaXMuaW5jb21wbGV0ZUVzY2FwZVRpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQodGhpcy5pbmNvbXBsZXRlRXNjYXBlVGltZXIpXG4gICAgICB0aGlzLmluY29tcGxldGVFc2NhcGVUaW1lciA9IG51bGxcbiAgICB9XG4gICAgaWYgKHRoaXMucGVuZGluZ0h5cGVybGlua1RpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQodGhpcy5wZW5kaW5nSHlwZXJsaW5rVGltZXIpXG4gICAgICB0aGlzLnBlbmRpbmdIeXBlcmxpbmtUaW1lciA9IG51bGxcbiAgICB9XG4gICAgLy8gaWdub3JlIGNhbGxpbmcgc2V0UmF3TW9kZSBvbiBhbiBoYW5kbGUgc3RkaW4gaXQgY2Fubm90IGJlIGNhbGxlZFxuICAgIGlmICh0aGlzLmlzUmF3TW9kZVN1cHBvcnRlZCgpKSB7XG4gICAgICB0aGlzLmhhbmRsZVNldFJhd01vZGUoZmFsc2UpXG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgY29tcG9uZW50RGlkQ2F0Y2goZXJyb3I6IEVycm9yKSB7XG4gICAgdGhpcy5oYW5kbGVFeGl0KGVycm9yKVxuICB9XG5cbiAgaGFuZGxlU2V0UmF3TW9kZSA9IChpc0VuYWJsZWQ6IGJvb2xlYW4pOiB2b2lkID0+IHtcbiAgICBjb25zdCB7IHN0ZGluIH0gPSB0aGlzLnByb3BzXG5cbiAgICBpZiAoIXRoaXMuaXNSYXdNb2RlU3VwcG9ydGVkKCkpIHtcbiAgICAgIGlmIChzdGRpbiA9PT0gcHJvY2Vzcy5zdGRpbikge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgJ1JhdyBtb2RlIGlzIG5vdCBzdXBwb3J0ZWQgb24gdGhlIGN1cnJlbnQgcHJvY2Vzcy5zdGRpbiwgd2hpY2ggSW5rIHVzZXMgYXMgaW5wdXQgc3RyZWFtIGJ5IGRlZmF1bHQuXFxuUmVhZCBhYm91dCBob3cgdG8gcHJldmVudCB0aGlzIGVycm9yIG9uIGh0dHBzOi8vZ2l0aHViLmNvbS92YWRpbWRlbWVkZXMvaW5rLyNpc3Jhd21vZGVzdXBwb3J0ZWQnLFxuICAgICAgICApXG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgJ1JhdyBtb2RlIGlzIG5vdCBzdXBwb3J0ZWQgb24gdGhlIHN0ZGluIHByb3ZpZGVkIHRvIEluay5cXG5SZWFkIGFib3V0IGhvdyB0byBwcmV2ZW50IHRoaXMgZXJyb3Igb24gaHR0cHM6Ly9naXRodWIuY29tL3ZhZGltZGVtZWRlcy9pbmsvI2lzcmF3bW9kZXN1cHBvcnRlZCcsXG4gICAgICAgIClcbiAgICAgIH1cbiAgICB9XG5cbiAgICBzdGRpbi5zZXRFbmNvZGluZygndXRmOCcpXG5cbiAgICBpZiAoaXNFbmFibGVkKSB7XG4gICAgICAvLyBFbnN1cmUgcmF3IG1vZGUgaXMgZW5hYmxlZCBvbmx5IG9uY2VcbiAgICAgIGlmICh0aGlzLnJhd01vZGVFbmFibGVkQ291bnQgPT09IDApIHtcbiAgICAgICAgLy8gU3RvcCBlYXJseSBpbnB1dCBjYXB0dXJlIHJpZ2h0IGJlZm9yZSB3ZSBhZGQgb3VyIG93biByZWFkYWJsZSBoYW5kbGVyLlxuICAgICAgICAvLyBCb3RoIHVzZSB0aGUgc2FtZSBzdGRpbiAncmVhZGFibGUnICsgcmVhZCgpIHBhdHRlcm4sIHNvIHRoZXkgY2FuJ3RcbiAgICAgICAgLy8gY29leGlzdCAtLSBvdXIgaGFuZGxlciB3b3VsZCBkcmFpbiBzdGRpbiBiZWZvcmUgSW5rJ3MgY2FuIHNlZSBpdC5cbiAgICAgICAgLy8gVGhlIGJ1ZmZlcmVkIHRleHQgaXMgcHJlc2VydmVkIGZvciBSRVBMLnRzeCB2aWEgY29uc3VtZUVhcmx5SW5wdXQoKS5cbiAgICAgICAgc3RvcENhcHR1cmluZ0Vhcmx5SW5wdXQoKVxuICAgICAgICBzdGRpbi5yZWYoKVxuICAgICAgICBzdGRpbi5zZXRSYXdNb2RlKHRydWUpXG4gICAgICAgIHN0ZGluLmFkZExpc3RlbmVyKCdyZWFkYWJsZScsIHRoaXMuaGFuZGxlUmVhZGFibGUpXG4gICAgICAgIC8vIEVuYWJsZSBicmFja2V0ZWQgcGFzdGUgbW9kZVxuICAgICAgICB0aGlzLnByb3BzLnN0ZG91dC53cml0ZShFQlApXG4gICAgICAgIC8vIEVuYWJsZSB0ZXJtaW5hbCBmb2N1cyByZXBvcnRpbmcgKERFQ1NFVCAxMDA0KVxuICAgICAgICB0aGlzLnByb3BzLnN0ZG91dC53cml0ZShFRkUpXG4gICAgICAgIC8vIEVuYWJsZSBleHRlbmRlZCBrZXkgcmVwb3J0aW5nIHNvIGN0cmwrc2hpZnQrPGxldHRlcj4gaXNcbiAgICAgICAgLy8gZGlzdGluZ3Vpc2hhYmxlIGZyb20gY3RybCs8bGV0dGVyPi4gV2Ugd3JpdGUgYm90aCB0aGUga2l0dHkgc3RhY2tcbiAgICAgICAgLy8gcHVzaCAoQ1NJID4xdSkgYW5kIHh0ZXJtIG1vZGlmeU90aGVyS2V5cyBsZXZlbCAyIChDU0kgPjQ7Mm0pIOKAlFxuICAgICAgICAvLyB0ZXJtaW5hbHMgaG9ub3Igd2hpY2hldmVyIHRoZXkgaW1wbGVtZW50ICh0bXV4IG9ubHkgYWNjZXB0cyB0aGVcbiAgICAgICAgLy8gbGF0dGVyKS5cbiAgICAgICAgaWYgKHN1cHBvcnRzRXh0ZW5kZWRLZXlzKCkpIHtcbiAgICAgICAgICB0aGlzLnByb3BzLnN0ZG91dC53cml0ZShFTkFCTEVfS0lUVFlfS0VZQk9BUkQpXG4gICAgICAgICAgdGhpcy5wcm9wcy5zdGRvdXQud3JpdGUoRU5BQkxFX01PRElGWV9PVEhFUl9LRVlTKVxuICAgICAgICB9XG4gICAgICAgIC8vIFByb2JlIHRlcm1pbmFsIGlkZW50aXR5LiBYVFZFUlNJT04gc3Vydml2ZXMgU1NIIChxdWVyeS9yZXBseSBnb2VzXG4gICAgICAgIC8vIHRocm91Z2ggdGhlIHB0eSksIHVubGlrZSBURVJNX1BST0dSQU0uIFVzZWQgZm9yIHdoZWVsLXNjcm9sbCBiYXNlXG4gICAgICAgIC8vIGRldGVjdGlvbiB3aGVuIGVudiB2YXJzIGFyZSBhYnNlbnQuIEZpcmUtYW5kLWZvcmdldDogdGhlIERBMVxuICAgICAgICAvLyBzZW50aW5lbCBib3VuZHMgdGhlIHJvdW5kLXRyaXAsIGFuZCBpZiB0aGUgdGVybWluYWwgaWdub3JlcyB0aGVcbiAgICAgICAgLy8gcXVlcnksIGZsdXNoKCkgc3RpbGwgcmVzb2x2ZXMgYW5kIG5hbWUgc3RheXMgdW5kZWZpbmVkLlxuICAgICAgICAvLyBEZWZlcnJlZCB0byBuZXh0IHRpY2sgc28gaXQgZmlyZXMgQUZURVIgdGhlIGN1cnJlbnQgc3luY2hyb25vdXNcbiAgICAgICAgLy8gaW5pdCBzZXF1ZW5jZSBjb21wbGV0ZXMg4oCUIGF2b2lkcyBpbnRlcmxlYXZpbmcgd2l0aCBhbHQtc2NyZWVuL21vdXNlXG4gICAgICAgIC8vIHRyYWNraW5nIGVuYWJsZSB3cml0ZXMgdGhhdCBtYXkgaGFwcGVuIGluIHRoZSBzYW1lIHJlbmRlciBjeWNsZS5cbiAgICAgICAgc2V0SW1tZWRpYXRlKCgpID0+IHtcbiAgICAgICAgICB2b2lkIFByb21pc2UuYWxsKFtcbiAgICAgICAgICAgIHRoaXMucXVlcmllci5zZW5kKHh0dmVyc2lvbigpKSxcbiAgICAgICAgICAgIHRoaXMucXVlcmllci5mbHVzaCgpLFxuICAgICAgICAgIF0pLnRoZW4oKFtyXSkgPT4ge1xuICAgICAgICAgICAgaWYgKHIpIHtcbiAgICAgICAgICAgICAgc2V0WHR2ZXJzaW9uTmFtZShyLm5hbWUpXG4gICAgICAgICAgICAgIGxvZ0ZvckRlYnVnZ2luZyhgWFRWRVJTSU9OOiB0ZXJtaW5hbCBpZGVudGlmaWVkIGFzIFwiJHtyLm5hbWV9XCJgKVxuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgbG9nRm9yRGVidWdnaW5nKCdYVFZFUlNJT046IG5vIHJlcGx5ICh0ZXJtaW5hbCBpZ25vcmVkIHF1ZXJ5KScpXG4gICAgICAgICAgICB9XG4gICAgICAgICAgfSlcbiAgICAgICAgfSlcbiAgICAgIH1cblxuICAgICAgdGhpcy5yYXdNb2RlRW5hYmxlZENvdW50KytcbiAgICAgIHJldHVyblxuICAgIH1cblxuICAgIC8vIERpc2FibGUgcmF3IG1vZGUgb25seSB3aGVuIG5vIGNvbXBvbmVudHMgbGVmdCB0aGF0IGFyZSB1c2luZyBpdFxuICAgIGlmICgtLXRoaXMucmF3TW9kZUVuYWJsZWRDb3VudCA9PT0gMCkge1xuICAgICAgdGhpcy5wcm9wcy5zdGRvdXQud3JpdGUoRElTQUJMRV9NT0RJRllfT1RIRVJfS0VZUylcbiAgICAgIHRoaXMucHJvcHMuc3Rkb3V0LndyaXRlKERJU0FCTEVfS0lUVFlfS0VZQk9BUkQpXG4gICAgICAvLyBEaXNhYmxlIHRlcm1pbmFsIGZvY3VzIHJlcG9ydGluZyAoREVDU0VUIDEwMDQpXG4gICAgICB0aGlzLnByb3BzLnN0ZG91dC53cml0ZShERkUpXG4gICAgICAvLyBEaXNhYmxlIGJyYWNrZXRlZCBwYXN0ZSBtb2RlXG4gICAgICB0aGlzLnByb3BzLnN0ZG91dC53cml0ZShEQlApXG4gICAgICBzdGRpbi5zZXRSYXdNb2RlKGZhbHNlKVxuICAgICAgc3RkaW4ucmVtb3ZlTGlzdGVuZXIoJ3JlYWRhYmxlJywgdGhpcy5oYW5kbGVSZWFkYWJsZSlcbiAgICAgIHN0ZGluLnVucmVmKClcbiAgICB9XG4gIH1cblxuICAvLyBIZWxwZXIgdG8gZmx1c2ggaW5jb21wbGV0ZSBlc2NhcGUgc2VxdWVuY2VzXG4gIGZsdXNoSW5jb21wbGV0ZSA9ICgpOiB2b2lkID0+IHtcbiAgICAvLyBDbGVhciB0aGUgdGltZXIgcmVmZXJlbmNlXG4gICAgdGhpcy5pbmNvbXBsZXRlRXNjYXBlVGltZXIgPSBudWxsXG5cbiAgICAvLyBPbmx5IHByb2NlZWQgaWYgd2UgaGF2ZSBpbmNvbXBsZXRlIHNlcXVlbmNlc1xuICAgIGlmICghdGhpcy5rZXlQYXJzZVN0YXRlLmluY29tcGxldGUpIHJldHVyblxuXG4gICAgLy8gRnVsbHNjcmVlbjogaWYgc3RkaW4gaGFzIGRhdGEgd2FpdGluZywgaXQncyBhbG1vc3QgY2VydGFpbmx5IHRoZVxuICAgIC8vIGNvbnRpbnVhdGlvbiBvZiB0aGUgYnVmZmVyZWQgc2VxdWVuY2UgKGUuZy4gYFs8NjQ7NzQ7MTZNYCBhZnRlciBhXG4gICAgLy8gbG9uZSBFU0MpLiBOb2RlJ3MgZXZlbnQgbG9vcCBydW5zIHRoZSB0aW1lcnMgcGhhc2UgYmVmb3JlIHRoZSBwb2xsXG4gICAgLy8gcGhhc2UsIHNvIHdoZW4gYSBoZWF2eSByZW5kZXIgYmxvY2tzIHRoZSBsb29wIHBhc3QgNTBtcywgdGhpcyB0aW1lclxuICAgIC8vIGZpcmVzIGJlZm9yZSB0aGUgcXVldWVkIHJlYWRhYmxlIGV2ZW50IGV2ZW4gdGhvdWdoIHRoZSBieXRlcyBhcmVcbiAgICAvLyBhbHJlYWR5IGJ1ZmZlcmVkLiBSZS1hcm0gaW5zdGVhZCBvZiBmbHVzaGluZzogaGFuZGxlUmVhZGFibGUgd2lsbFxuICAgIC8vIGRyYWluIHN0ZGluIG5leHQgYW5kIGNsZWFyIHRoaXMgdGltZXIuIFByZXZlbnRzIGJvdGggdGhlIHNwdXJpb3VzXG4gICAgLy8gRXNjYXBlIGtleSBhbmQgdGhlIGxvc3Qgc2Nyb2xsIGV2ZW50LlxuICAgIGlmICh0aGlzLnByb3BzLnN0ZGluLnJlYWRhYmxlTGVuZ3RoID4gMCkge1xuICAgICAgdGhpcy5pbmNvbXBsZXRlRXNjYXBlVGltZXIgPSBzZXRUaW1lb3V0KFxuICAgICAgICB0aGlzLmZsdXNoSW5jb21wbGV0ZSxcbiAgICAgICAgdGhpcy5OT1JNQUxfVElNRU9VVCxcbiAgICAgIClcbiAgICAgIHJldHVyblxuICAgIH1cblxuICAgIC8vIFByb2Nlc3MgaW5jb21wbGV0ZSBhcyBhIGZsdXNoIG9wZXJhdGlvbiAoaW5wdXQ9bnVsbClcbiAgICAvLyBUaGlzIHJldXNlcyBhbGwgZXhpc3RpbmcgcGFyc2luZyBsb2dpY1xuICAgIHRoaXMucHJvY2Vzc0lucHV0KG51bGwpXG4gIH1cblxuICAvLyBQcm9jZXNzIGlucHV0IHRocm91Z2ggdGhlIHBhcnNlciBhbmQgaGFuZGxlIHRoZSByZXN1bHRzXG4gIHByb2Nlc3NJbnB1dCA9IChpbnB1dDogc3RyaW5nIHwgQnVmZmVyIHwgbnVsbCk6IHZvaWQgPT4ge1xuICAgIC8vIFBhcnNlIGlucHV0IHVzaW5nIG91ciBzdGF0ZSBtYWNoaW5lXG4gICAgY29uc3QgW2tleXMsIG5ld1N0YXRlXSA9IHBhcnNlTXVsdGlwbGVLZXlwcmVzc2VzKHRoaXMua2V5UGFyc2VTdGF0ZSwgaW5wdXQpXG4gICAgdGhpcy5rZXlQYXJzZVN0YXRlID0gbmV3U3RhdGVcblxuICAgIC8vIFByb2Nlc3MgQUxMIGtleXMgaW4gYSBTSU5HTEUgZGlzY3JldGVVcGRhdGVzIGNhbGwgdG8gcHJldmVudFxuICAgIC8vIFwiTWF4aW11bSB1cGRhdGUgZGVwdGggZXhjZWVkZWRcIiBlcnJvciB3aGVuIG1hbnkga2V5cyBhcnJpdmUgYXQgb25jZVxuICAgIC8vIChlLmcuLCBmcm9tIHBhc3RlIG9wZXJhdGlvbnMgb3IgaG9sZGluZyBrZXlzIHJhcGlkbHkpLlxuICAgIC8vIFRoaXMgYmF0Y2hlcyBhbGwgc3RhdGUgdXBkYXRlcyBmcm9tIGhhbmRsZUlucHV0IGFuZCBhbGwgdXNlSW5wdXRcbiAgICAvLyBsaXN0ZW5lcnMgdG9nZXRoZXIgd2l0aGluIG9uZSBoaWdoLXByaW9yaXR5IHVwZGF0ZSBjb250ZXh0LlxuICAgIGlmIChrZXlzLmxlbmd0aCA+IDApIHtcbiAgICAgIHJlY29uY2lsZXIuZGlzY3JldGVVcGRhdGVzKFxuICAgICAgICBwcm9jZXNzS2V5c0luQmF0Y2gsXG4gICAgICAgIHRoaXMsXG4gICAgICAgIGtleXMsXG4gICAgICAgIHVuZGVmaW5lZCxcbiAgICAgICAgdW5kZWZpbmVkLFxuICAgICAgKVxuICAgIH1cblxuICAgIC8vIElmIHdlIGhhdmUgaW5jb21wbGV0ZSBlc2NhcGUgc2VxdWVuY2VzLCBzZXQgYSB0aW1lciB0byBmbHVzaCB0aGVtXG4gICAgaWYgKHRoaXMua2V5UGFyc2VTdGF0ZS5pbmNvbXBsZXRlKSB7XG4gICAgICAvLyBDYW5jZWwgYW55IGV4aXN0aW5nIHRpbWVyIGZpcnN0XG4gICAgICBpZiAodGhpcy5pbmNvbXBsZXRlRXNjYXBlVGltZXIpIHtcbiAgICAgICAgY2xlYXJUaW1lb3V0KHRoaXMuaW5jb21wbGV0ZUVzY2FwZVRpbWVyKVxuICAgICAgfVxuICAgICAgdGhpcy5pbmNvbXBsZXRlRXNjYXBlVGltZXIgPSBzZXRUaW1lb3V0KFxuICAgICAgICB0aGlzLmZsdXNoSW5jb21wbGV0ZSxcbiAgICAgICAgdGhpcy5rZXlQYXJzZVN0YXRlLm1vZGUgPT09ICdJTl9QQVNURSdcbiAgICAgICAgICA/IHRoaXMuUEFTVEVfVElNRU9VVFxuICAgICAgICAgIDogdGhpcy5OT1JNQUxfVElNRU9VVCxcbiAgICAgIClcbiAgICB9XG4gIH1cblxuICBoYW5kbGVSZWFkYWJsZSA9ICgpOiB2b2lkID0+IHtcbiAgICAvLyBEZXRlY3QgbG9uZyBzdGRpbiBnYXBzICh0bXV4IGF0dGFjaCwgc3NoIHJlY29ubmVjdCwgbGFwdG9wIHdha2UpLlxuICAgIC8vIFRoZSB0ZXJtaW5hbCBtYXkgaGF2ZSByZXNldCBERUMgcHJpdmF0ZSBtb2RlczsgcmUtYXNzZXJ0IG1vdXNlXG4gICAgLy8gdHJhY2tpbmcuIENoZWNrZWQgYmVmb3JlIHRoZSByZWFkIGxvb3Agc28gb25lIERhdGUubm93KCkgY292ZXJzXG4gICAgLy8gYWxsIGNodW5rcyBpbiB0aGlzIHJlYWRhYmxlIGV2ZW50LlxuICAgIGNvbnN0IG5vdyA9IERhdGUubm93KClcbiAgICBpZiAobm93IC0gdGhpcy5sYXN0U3RkaW5UaW1lID4gU1RESU5fUkVTVU1FX0dBUF9NUykge1xuICAgICAgdGhpcy5wcm9wcy5vblN0ZGluUmVzdW1lPy4oKVxuICAgIH1cbiAgICB0aGlzLmxhc3RTdGRpblRpbWUgPSBub3dcbiAgICB0cnkge1xuICAgICAgbGV0IGNodW5rXG4gICAgICB3aGlsZSAoKGNodW5rID0gdGhpcy5wcm9wcy5zdGRpbi5yZWFkKCkgYXMgc3RyaW5nIHwgbnVsbCkgIT09IG51bGwpIHtcbiAgICAgICAgLy8gUHJvY2VzcyB0aGUgaW5wdXQgY2h1bmtcbiAgICAgICAgdGhpcy5wcm9jZXNzSW5wdXQoY2h1bmspXG4gICAgICB9XG4gICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgIC8vIEluIEJ1biwgYW4gdW5jYXVnaHQgdGhyb3cgaW5zaWRlIGEgc3RyZWFtICdyZWFkYWJsZScgaGFuZGxlciBjYW5cbiAgICAgIC8vIHBlcm1hbmVudGx5IHdlZGdlIHRoZSBzdHJlYW06IGRhdGEgc3RheXMgYnVmZmVyZWQgYW5kICdyZWFkYWJsZSdcbiAgICAgIC8vIG5ldmVyIHJlLWVtaXRzLiBDYXRjaGluZyBoZXJlIGVuc3VyZXMgdGhlIHN0cmVhbSBzdGF5cyBoZWFsdGh5IHNvXG4gICAgICAvLyBzdWJzZXF1ZW50IGtleXN0cm9rZXMgYXJlIHN0aWxsIGRlbGl2ZXJlZC5cbiAgICAgIGxvZ0Vycm9yKGVycm9yKVxuXG4gICAgICAvLyBSZS1hdHRhY2ggdGhlIGxpc3RlbmVyIGluIGNhc2UgdGhlIGV4Y2VwdGlvbiBkZXRhY2hlZCBpdC5cbiAgICAgIC8vIEJ1biBtYXkgcmVtb3ZlIHRoZSBsaXN0ZW5lciBhZnRlciBhbiBlcnJvcjsgd2l0aG91dCB0aGlzLFxuICAgICAgLy8gdGhlIHNlc3Npb24gZnJlZXplcyBwZXJtYW5lbnRseSAoc3RkaW4gcmVhZGVyIGRlYWQsIGV2ZW50IGxvb3AgYWxpdmUpLlxuICAgICAgY29uc3QgeyBzdGRpbiB9ID0gdGhpcy5wcm9wc1xuICAgICAgaWYgKFxuICAgICAgICB0aGlzLnJhd01vZGVFbmFibGVkQ291bnQgPiAwICYmXG4gICAgICAgICFzdGRpbi5saXN0ZW5lcnMoJ3JlYWRhYmxlJykuaW5jbHVkZXModGhpcy5oYW5kbGVSZWFkYWJsZSlcbiAgICAgICkge1xuICAgICAgICBsb2dGb3JEZWJ1Z2dpbmcoXG4gICAgICAgICAgJ2hhbmRsZVJlYWRhYmxlOiByZS1hdHRhY2hpbmcgc3RkaW4gcmVhZGFibGUgbGlzdGVuZXIgYWZ0ZXIgZXJyb3IgcmVjb3ZlcnknLFxuICAgICAgICAgIHsgbGV2ZWw6ICd3YXJuJyB9LFxuICAgICAgICApXG4gICAgICAgIHN0ZGluLmFkZExpc3RlbmVyKCdyZWFkYWJsZScsIHRoaXMuaGFuZGxlUmVhZGFibGUpXG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgaGFuZGxlSW5wdXQgPSAoaW5wdXQ6IHN0cmluZyB8IHVuZGVmaW5lZCk6IHZvaWQgPT4ge1xuICAgIC8vIEV4aXQgb24gQ3RybCtDXG4gICAgaWYgKGlucHV0ID09PSAnXFx4MDMnICYmIHRoaXMucHJvcHMuZXhpdE9uQ3RybEMpIHtcbiAgICAgIHRoaXMuaGFuZGxlRXhpdCgpXG4gICAgfVxuXG4gICAgLy8gTm90ZTogQ3RybCtaIChzdXNwZW5kKSBpcyBub3cgaGFuZGxlZCBpbiBwcm9jZXNzS2V5c0luQmF0Y2ggdXNpbmcgdGhlXG4gICAgLy8gcGFyc2VkIGtleSB0byBzdXBwb3J0IGJvdGggcmF3IChcXHgxYSkgYW5kIENTSSB1IGZvcm1hdCBmcm9tIEtpdHR5XG4gICAgLy8ga2V5Ym9hcmQgcHJvdG9jb2wgdGVybWluYWxzIChHaG9zdHR5LCBpVGVybTIsIGtpdHR5LCBXZXpUZXJtKVxuICB9XG5cbiAgaGFuZGxlRXhpdCA9IChlcnJvcj86IEVycm9yKTogdm9pZCA9PiB7XG4gICAgaWYgKHRoaXMuaXNSYXdNb2RlU3VwcG9ydGVkKCkpIHtcbiAgICAgIHRoaXMuaGFuZGxlU2V0UmF3TW9kZShmYWxzZSlcbiAgICB9XG5cbiAgICB0aGlzLnByb3BzLm9uRXhpdChlcnJvcilcbiAgfVxuXG4gIGhhbmRsZVRlcm1pbmFsRm9jdXMgPSAoaXNGb2N1c2VkOiBib29sZWFuKTogdm9pZCA9PiB7XG4gICAgLy8gc2V0VGVybWluYWxGb2N1c2VkIG5vdGlmaWVzIHN1YnNjcmliZXJzOiBUZXJtaW5hbEZvY3VzUHJvdmlkZXIgKGNvbnRleHQpXG4gICAgLy8gYW5kIENsb2NrIChpbnRlcnZhbCBzcGVlZCkg4oCUIG5vIEFwcCBzZXRTdGF0ZSBuZWVkZWQuXG4gICAgc2V0VGVybWluYWxGb2N1c2VkKGlzRm9jdXNlZClcbiAgfVxuXG4gIGhhbmRsZVN1c3BlbmQgPSAoKTogdm9pZCA9PiB7XG4gICAgaWYgKCF0aGlzLmlzUmF3TW9kZVN1cHBvcnRlZCgpKSB7XG4gICAgICByZXR1cm5cbiAgICB9XG5cbiAgICAvLyBTdG9yZSB0aGUgZXhhY3QgcmF3IG1vZGUgY291bnQgdG8gcmVzdG9yZSBpdCBwcm9wZXJseVxuICAgIGNvbnN0IHJhd01vZGVDb3VudEJlZm9yZVN1c3BlbmQgPSB0aGlzLnJhd01vZGVFbmFibGVkQ291bnRcblxuICAgIC8vIENvbXBsZXRlbHkgZGlzYWJsZSByYXcgbW9kZSBiZWZvcmUgc3VzcGVuZGluZ1xuICAgIHdoaWxlICh0aGlzLnJhd01vZGVFbmFibGVkQ291bnQgPiAwKSB7XG4gICAgICB0aGlzLmhhbmRsZVNldFJhd01vZGUoZmFsc2UpXG4gICAgfVxuXG4gICAgLy8gU2hvdyBjdXJzb3IsIGRpc2FibGUgZm9jdXMgcmVwb3J0aW5nLCBhbmQgZGlzYWJsZSBtb3VzZSB0cmFja2luZ1xuICAgIC8vIGJlZm9yZSBzdXNwZW5kaW5nLiBESVNBQkxFX01PVVNFX1RSQUNLSU5HIGlzIGEgbm8tb3AgaWYgdHJhY2tpbmdcbiAgICAvLyB3YXNuJ3QgZW5hYmxlZCwgc28gaXQncyBzYWZlIHRvIGVtaXQgdW5jb25kaXRpb25hbGx5IOKAlCB3aXRob3V0XG4gICAgLy8gaXQsIFNHUiBtb3VzZSBzZXF1ZW5jZXMgd291bGQgYXBwZWFyIGFzIGdhcmJsZWQgdGV4dCBhdCB0aGVcbiAgICAvLyBzaGVsbCBwcm9tcHQgd2hpbGUgc3VzcGVuZGVkLlxuICAgIGlmICh0aGlzLnByb3BzLnN0ZG91dC5pc1RUWSkge1xuICAgICAgdGhpcy5wcm9wcy5zdGRvdXQud3JpdGUoU0hPV19DVVJTT1IgKyBERkUgKyBESVNBQkxFX01PVVNFX1RSQUNLSU5HKVxuICAgIH1cblxuICAgIC8vIEVtaXQgc3VzcGVuZCBldmVudCBmb3IgQ2xhdWRlIENvZGUgdG8gaGFuZGxlLiBNb3N0bHkganVzdCBoYXMgYSBub3RpZmljYXRpb25cbiAgICB0aGlzLmludGVybmFsX2V2ZW50RW1pdHRlci5lbWl0KCdzdXNwZW5kJylcblxuICAgIC8vIFNldCB1cCByZXN1bWUgaGFuZGxlclxuICAgIGNvbnN0IHJlc3VtZUhhbmRsZXIgPSAoKSA9PiB7XG4gICAgICAvLyBSZXN0b3JlIHJhdyBtb2RlIHRvIGV4YWN0IHByZXZpb3VzIHN0YXRlXG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHJhd01vZGVDb3VudEJlZm9yZVN1c3BlbmQ7IGkrKykge1xuICAgICAgICBpZiAodGhpcy5pc1Jhd01vZGVTdXBwb3J0ZWQoKSkge1xuICAgICAgICAgIHRoaXMuaGFuZGxlU2V0UmF3TW9kZSh0cnVlKVxuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIC8vIEhpZGUgY3Vyc29yICh1bmxlc3MgaW4gYWNjZXNzaWJpbGl0eSBtb2RlKSBhbmQgcmUtZW5hYmxlIGZvY3VzIHJlcG9ydGluZyBhZnRlciByZXN1bWluZ1xuICAgICAgaWYgKHRoaXMucHJvcHMuc3Rkb3V0LmlzVFRZKSB7XG4gICAgICAgIGlmICghaXNFbnZUcnV0aHkocHJvY2Vzcy5lbnYuQ0xBVURFX0NPREVfQUNDRVNTSUJJTElUWSkpIHtcbiAgICAgICAgICB0aGlzLnByb3BzLnN0ZG91dC53cml0ZShISURFX0NVUlNPUilcbiAgICAgICAgfVxuICAgICAgICAvLyBSZS1lbmFibGUgZm9jdXMgcmVwb3J0aW5nIHRvIHJlc3RvcmUgdGVybWluYWwgc3RhdGVcbiAgICAgICAgdGhpcy5wcm9wcy5zdGRvdXQud3JpdGUoRUZFKVxuICAgICAgfVxuXG4gICAgICAvLyBFbWl0IHJlc3VtZSBldmVudCBmb3IgQ2xhdWRlIENvZGUgdG8gaGFuZGxlXG4gICAgICB0aGlzLmludGVybmFsX2V2ZW50RW1pdHRlci5lbWl0KCdyZXN1bWUnKVxuXG4gICAgICBwcm9jZXNzLnJlbW92ZUxpc3RlbmVyKCdTSUdDT05UJywgcmVzdW1lSGFuZGxlcilcbiAgICB9XG5cbiAgICBwcm9jZXNzLm9uKCdTSUdDT05UJywgcmVzdW1lSGFuZGxlcilcbiAgICBwcm9jZXNzLmtpbGwocHJvY2Vzcy5waWQsICdTSUdTVE9QJylcbiAgfVxufVxuXG4vLyBIZWxwZXIgdG8gcHJvY2VzcyBhbGwga2V5cyB3aXRoaW4gYSBzaW5nbGUgZGlzY3JldGUgdXBkYXRlIGNvbnRleHQuXG4vLyBkaXNjcmV0ZVVwZGF0ZXMgZXhwZWN0cyAoZm4sIGEsIGIsIGMsIGQpIC0+IGZuKGEsIGIsIGMsIGQpXG5mdW5jdGlvbiBwcm9jZXNzS2V5c0luQmF0Y2goXG4gIGFwcDogQXBwLFxuICBpdGVtczogUGFyc2VkSW5wdXRbXSxcbiAgX3VudXNlZDE6IHVuZGVmaW5lZCxcbiAgX3VudXNlZDI6IHVuZGVmaW5lZCxcbik6IHZvaWQge1xuICAvLyBVcGRhdGUgaW50ZXJhY3Rpb24gdGltZSBmb3Igbm90aWZpY2F0aW9uIHRpbWVvdXQgdHJhY2tpbmcuXG4gIC8vIFRoaXMgaXMgY2FsbGVkIGZyb20gdGhlIGNlbnRyYWwgaW5wdXQgaGFuZGxlciB0byBhdm9pZCBoYXZpbmcgbXVsdGlwbGVcbiAgLy8gc3RkaW4gbGlzdGVuZXJzIHRoYXQgY2FuIGNhdXNlIHJhY2UgY29uZGl0aW9ucyBhbmQgZHJvcHBlZCBpbnB1dC5cbiAgLy8gVGVybWluYWwgcmVzcG9uc2VzIChraW5kOiAncmVzcG9uc2UnKSBhcmUgYXV0b21hdGVkLCBub3QgdXNlciBpbnB1dC5cbiAgLy8gTW9kZS0xMDAzIG5vLWJ1dHRvbiBtb3Rpb24gaXMgYWxzbyBleGNsdWRlZCDigJQgcGFzc2l2ZSBjdXJzb3IgZHJpZnQgaXNcbiAgLy8gbm90IGVuZ2FnZW1lbnQgKHdvdWxkIHN1cHByZXNzIGlkbGUgbm90aWZpY2F0aW9ucyArIGRlZmVyIGhvdXNla2VlcGluZykuXG4gIGlmIChcbiAgICBpdGVtcy5zb21lKFxuICAgICAgaSA9PlxuICAgICAgICBpLmtpbmQgPT09ICdrZXknIHx8XG4gICAgICAgIChpLmtpbmQgPT09ICdtb3VzZScgJiZcbiAgICAgICAgICAhKChpLmJ1dHRvbiAmIDB4MjApICE9PSAwICYmIChpLmJ1dHRvbiAmIDB4MDMpID09PSAzKSksXG4gICAgKVxuICApIHtcbiAgICB1cGRhdGVMYXN0SW50ZXJhY3Rpb25UaW1lKClcbiAgfVxuXG4gIGZvciAoY29uc3QgaXRlbSBvZiBpdGVtcykge1xuICAgIC8vIFRlcm1pbmFsIHJlc3BvbnNlcyAoREVDUlBNLCBEQTEsIE9TQyByZXBsaWVzLCBldGMuKSBhcmUgbm90IHVzZXJcbiAgICAvLyBpbnB1dCDigJQgcm91dGUgdGhlbSB0byB0aGUgcXVlcmllciB0byByZXNvbHZlIHBlbmRpbmcgcHJvbWlzZXMuXG4gICAgaWYgKGl0ZW0ua2luZCA9PT0gJ3Jlc3BvbnNlJykge1xuICAgICAgYXBwLnF1ZXJpZXIub25SZXNwb25zZShpdGVtLnJlc3BvbnNlKVxuICAgICAgY29udGludWVcbiAgICB9XG5cbiAgICAvLyBNb3VzZSBjbGljay9kcmFnIGV2ZW50cyB1cGRhdGUgc2VsZWN0aW9uIHN0YXRlIChmdWxsc2NyZWVuIG9ubHkpLlxuICAgIC8vIFRlcm1pbmFsIHNlbmRzIDEtaW5kZXhlZCBjb2wvcm93OyBjb252ZXJ0IHRvIDAtaW5kZXhlZCBmb3IgdGhlXG4gICAgLy8gc2NyZWVuIGJ1ZmZlci4gQnV0dG9uIGJpdCAweDIwID0gZHJhZyAobW90aW9uIHdoaWxlIGJ1dHRvbiBoZWxkKS5cbiAgICBpZiAoaXRlbS5raW5kID09PSAnbW91c2UnKSB7XG4gICAgICBoYW5kbGVNb3VzZUV2ZW50KGFwcCwgaXRlbSlcbiAgICAgIGNvbnRpbnVlXG4gICAgfVxuXG4gICAgY29uc3Qgc2VxdWVuY2UgPSBpdGVtLnNlcXVlbmNlXG5cbiAgICAvLyBIYW5kbGUgdGVybWluYWwgZm9jdXMgZXZlbnRzIChERUNTRVQgMTAwNClcbiAgICBpZiAoc2VxdWVuY2UgPT09IEZPQ1VTX0lOKSB7XG4gICAgICBhcHAuaGFuZGxlVGVybWluYWxGb2N1cyh0cnVlKVxuICAgICAgY29uc3QgZXZlbnQgPSBuZXcgVGVybWluYWxGb2N1c0V2ZW50KCd0ZXJtaW5hbGZvY3VzJylcbiAgICAgIGFwcC5pbnRlcm5hbF9ldmVudEVtaXR0ZXIuZW1pdCgndGVybWluYWxmb2N1cycsIGV2ZW50KVxuICAgICAgY29udGludWVcbiAgICB9XG4gICAgaWYgKHNlcXVlbmNlID09PSBGT0NVU19PVVQpIHtcbiAgICAgIGFwcC5oYW5kbGVUZXJtaW5hbEZvY3VzKGZhbHNlKVxuICAgICAgLy8gRGVmZW5zaXZlOiBpZiB3ZSBsb3N0IHRoZSByZWxlYXNlIGV2ZW50IChtb3VzZSByZWxlYXNlZCBvdXRzaWRlXG4gICAgICAvLyB0ZXJtaW5hbCB3aW5kb3cg4oCUIHNvbWUgZW11bGF0b3JzIGRyb3AgaXQgcmF0aGVyIHRoYW4gY2FwdHVyaW5nIHRoZVxuICAgICAgLy8gcG9pbnRlciksIGZvY3VzLW91dCBpcyB0aGUgbmV4dCBvYnNlcnZhYmxlIHNpZ25hbCB0aGF0IHRoZSBkcmFnIGlzXG4gICAgICAvLyBvdmVyLiBXaXRob3V0IHRoaXMsIGRyYWctdG8tc2Nyb2xsJ3MgdGltZXIgcnVucyB1bnRpbCB0aGUgc2Nyb2xsXG4gICAgICAvLyBib3VuZGFyeSBpcyBoaXQuXG4gICAgICBpZiAoYXBwLnByb3BzLnNlbGVjdGlvbi5pc0RyYWdnaW5nKSB7XG4gICAgICAgIGZpbmlzaFNlbGVjdGlvbihhcHAucHJvcHMuc2VsZWN0aW9uKVxuICAgICAgICBhcHAucHJvcHMub25TZWxlY3Rpb25DaGFuZ2UoKVxuICAgICAgfVxuICAgICAgY29uc3QgZXZlbnQgPSBuZXcgVGVybWluYWxGb2N1c0V2ZW50KCd0ZXJtaW5hbGJsdXInKVxuICAgICAgYXBwLmludGVybmFsX2V2ZW50RW1pdHRlci5lbWl0KCd0ZXJtaW5hbGJsdXInLCBldmVudClcbiAgICAgIGNvbnRpbnVlXG4gICAgfVxuXG4gICAgLy8gRmFpbHNhZmU6IGlmIHdlIHJlY2VpdmUgaW5wdXQsIHRoZSB0ZXJtaW5hbCBtdXN0IGJlIGZvY3VzZWRcbiAgICBpZiAoIWdldFRlcm1pbmFsRm9jdXNlZCgpKSB7XG4gICAgICBzZXRUZXJtaW5hbEZvY3VzZWQodHJ1ZSlcbiAgICB9XG5cbiAgICAvLyBIYW5kbGUgQ3RybCtaIChzdXNwZW5kKSB1c2luZyBwYXJzZWQga2V5IHRvIHN1cHBvcnQgYm90aCByYXcgKFxceDFhKSBhbmRcbiAgICAvLyBDU0kgdSBmb3JtYXQgKFxceDFiWzEyMjs1dSkgZnJvbSBLaXR0eSBrZXlib2FyZCBwcm90b2NvbCB0ZXJtaW5hbHNcbiAgICBpZiAoaXRlbS5uYW1lID09PSAneicgJiYgaXRlbS5jdHJsICYmIFNVUFBPUlRTX1NVU1BFTkQpIHtcbiAgICAgIGFwcC5oYW5kbGVTdXNwZW5kKClcbiAgICAgIGNvbnRpbnVlXG4gICAgfVxuXG4gICAgYXBwLmhhbmRsZUlucHV0KHNlcXVlbmNlKVxuICAgIGNvbnN0IGV2ZW50ID0gbmV3IElucHV0RXZlbnQoaXRlbSlcbiAgICBhcHAuaW50ZXJuYWxfZXZlbnRFbWl0dGVyLmVtaXQoJ2lucHV0JywgZXZlbnQpXG5cbiAgICAvLyBBbHNvIGRpc3BhdGNoIHRocm91Z2ggdGhlIERPTSB0cmVlIHNvIG9uS2V5RG93biBoYW5kbGVycyBmaXJlLlxuICAgIGFwcC5wcm9wcy5kaXNwYXRjaEtleWJvYXJkRXZlbnQoaXRlbSlcbiAgfVxufVxuXG4vKiogRXhwb3J0ZWQgZm9yIHRlc3RpbmcuIE11dGF0ZXMgYXBwLnByb3BzLnNlbGVjdGlvbiBhbmQgY2xpY2svaG92ZXIgc3RhdGUuICovXG5leHBvcnQgZnVuY3Rpb24gaGFuZGxlTW91c2VFdmVudChhcHA6IEFwcCwgbTogUGFyc2VkTW91c2UpOiB2b2lkIHtcbiAgLy8gQWxsb3cgZGlzYWJsaW5nIGNsaWNrIGhhbmRsaW5nIHdoaWxlIGtlZXBpbmcgd2hlZWwgc2Nyb2xsICh3aGljaCBnb2VzXG4gIC8vIHRocm91Z2ggdGhlIGtleWJpbmRpbmcgc3lzdGVtIGFzICd3aGVlbHVwJy8nd2hlZWxkb3duJywgbm90IGhlcmUpLlxuICBpZiAoaXNNb3VzZUNsaWNrc0Rpc2FibGVkKCkpIHJldHVyblxuXG4gIGNvbnN0IHNlbCA9IGFwcC5wcm9wcy5zZWxlY3Rpb25cbiAgLy8gVGVybWluYWwgY29vcmRzIGFyZSAxLWluZGV4ZWQ7IHNjcmVlbiBidWZmZXIgaXMgMC1pbmRleGVkXG4gIGNvbnN0IGNvbCA9IG0uY29sIC0gMVxuICBjb25zdCByb3cgPSBtLnJvdyAtIDFcbiAgY29uc3QgYmFzZUJ1dHRvbiA9IG0uYnV0dG9uICYgMHgwM1xuXG4gIGlmIChtLmFjdGlvbiA9PT0gJ3ByZXNzJykge1xuICAgIGlmICgobS5idXR0b24gJiAweDIwKSAhPT0gMCAmJiBiYXNlQnV0dG9uID09PSAzKSB7XG4gICAgICAvLyBNb2RlLTEwMDMgbW90aW9uIHdpdGggbm8gYnV0dG9uIGhlbGQuIERpc3BhdGNoIGhvdmVyOyBza2lwIHRoZVxuICAgICAgLy8gcmVzdCBvZiB0aGlzIGhhbmRsZXIgKG5vIHNlbGVjdGlvbiwgbm8gY2xpY2stY291bnQgc2lkZSBlZmZlY3RzKS5cbiAgICAgIC8vIExvc3QtcmVsZWFzZSByZWNvdmVyeTogbm8tYnV0dG9uIG1vdGlvbiB3aGlsZSBpc0RyYWdnaW5nPXRydWUgbWVhbnNcbiAgICAgIC8vIHRoZSByZWxlYXNlIGhhcHBlbmVkIG91dHNpZGUgdGhlIHRlcm1pbmFsIHdpbmRvdyAoaVRlcm0yIGRvZXNuJ3RcbiAgICAgIC8vIGNhcHR1cmUgdGhlIHBvaW50ZXIgcGFzdCB3aW5kb3cgYm91bmRzLCBzbyB0aGUgU0dSICdtJyBuZXZlclxuICAgICAgLy8gYXJyaXZlcykuIEZpbmlzaCB0aGUgc2VsZWN0aW9uIGhlcmUgc28gY29weS1vbi1zZWxlY3QgZmlyZXMuIFRoZVxuICAgICAgLy8gRk9DVVNfT1VUIGhhbmRsZXIgY292ZXJzIHRoZSBcInN3aXRjaGVkIGFwcHNcIiBjYXNlIGJ1dCBub3QgXCJyZWxlYXNlZFxuICAgICAgLy8gcGFzdCB0aGUgZWRnZSwgY2FtZSBiYWNrXCIg4oCUIGFuZCB0bXV4IGRyb3BzIGZvY3VzIGV2ZW50cyB1bmxlc3NcbiAgICAgIC8vIGBmb2N1cy1ldmVudHMgb25gIGlzIHNldCwgc28gdGhpcyBpcyB0aGUgbW9yZSByZWxpYWJsZSBzaWduYWwuXG4gICAgICBpZiAoc2VsLmlzRHJhZ2dpbmcpIHtcbiAgICAgICAgZmluaXNoU2VsZWN0aW9uKHNlbClcbiAgICAgICAgYXBwLnByb3BzLm9uU2VsZWN0aW9uQ2hhbmdlKClcbiAgICAgIH1cbiAgICAgIGlmIChjb2wgPT09IGFwcC5sYXN0SG92ZXJDb2wgJiYgcm93ID09PSBhcHAubGFzdEhvdmVyUm93KSByZXR1cm5cbiAgICAgIGFwcC5sYXN0SG92ZXJDb2wgPSBjb2xcbiAgICAgIGFwcC5sYXN0SG92ZXJSb3cgPSByb3dcbiAgICAgIGFwcC5wcm9wcy5vbkhvdmVyQXQoY29sLCByb3cpXG4gICAgICByZXR1cm5cbiAgICB9XG4gICAgaWYgKGJhc2VCdXR0b24gIT09IDApIHtcbiAgICAgIC8vIE5vbi1sZWZ0IHByZXNzIGJyZWFrcyB0aGUgbXVsdGktY2xpY2sgY2hhaW4uXG4gICAgICBhcHAuY2xpY2tDb3VudCA9IDBcbiAgICAgIHJldHVyblxuICAgIH1cbiAgICBpZiAoKG0uYnV0dG9uICYgMHgyMCkgIT09IDApIHtcbiAgICAgIC8vIERyYWcgbW90aW9uOiBtb2RlLWF3YXJlIGV4dGVuc2lvbiAoY2hhci93b3JkL2xpbmUpLiBvblNlbGVjdGlvbkRyYWdcbiAgICAgIC8vIGNhbGxzIG5vdGlmeVNlbGVjdGlvbkNoYW5nZSBpbnRlcm5hbGx5IOKAlCBubyBleHRyYSBvblNlbGVjdGlvbkNoYW5nZS5cbiAgICAgIGFwcC5wcm9wcy5vblNlbGVjdGlvbkRyYWcoY29sLCByb3cpXG4gICAgICByZXR1cm5cbiAgICB9XG4gICAgLy8gTG9zdC1yZWxlYXNlIGZhbGxiYWNrIGZvciBtb2RlLTEwMDItb25seSB0ZXJtaW5hbHM6IGEgZnJlc2ggcHJlc3NcbiAgICAvLyB3aGlsZSBpc0RyYWdnaW5nPXRydWUgbWVhbnMgdGhlIHByZXZpb3VzIHJlbGVhc2Ugd2FzIGRyb3BwZWQgKGN1cnNvclxuICAgIC8vIGxlZnQgdGhlIHdpbmRvdykuIEZpbmlzaCB0aGF0IHNlbGVjdGlvbiBzbyBjb3B5LW9uLXNlbGVjdCBmaXJlc1xuICAgIC8vIGJlZm9yZSBzdGFydFNlbGVjdGlvbi9vbk11bHRpQ2xpY2sgY2xvYmJlcnMgaXQuIE1vZGUtMTAwMyB0ZXJtaW5hbHNcbiAgICAvLyBoaXQgdGhlIG5vLWJ1dHRvbi1tb3Rpb24gcmVjb3ZlcnkgYWJvdmUgaW5zdGVhZCwgc28gdGhpcyBpcyByYXJlLlxuICAgIGlmIChzZWwuaXNEcmFnZ2luZykge1xuICAgICAgZmluaXNoU2VsZWN0aW9uKHNlbClcbiAgICAgIGFwcC5wcm9wcy5vblNlbGVjdGlvbkNoYW5nZSgpXG4gICAgfVxuICAgIC8vIEZyZXNoIGxlZnQgcHJlc3MuIERldGVjdCBtdWx0aS1jbGljayBIRVJFIChub3Qgb24gcmVsZWFzZSkgc28gdGhlXG4gICAgLy8gd29yZC9saW5lIGhpZ2hsaWdodCBhcHBlYXJzIGltbWVkaWF0ZWx5IGFuZCBhIHN1YnNlcXVlbnQgZHJhZyBjYW5cbiAgICAvLyBleHRlbmQgYnkgd29yZC9saW5lIGxpa2UgbmF0aXZlIG1hY09TLiBQcmV2aW91c2x5IGRldGVjdGVkIG9uXG4gICAgLy8gcmVsZWFzZSwgd2hpY2ggbWVhbnQgKGEpIHZpc2libGUgbGF0ZW5jeSBiZWZvcmUgdGhlIHdvcmQgaGlnaGxpZ2h0c1xuICAgIC8vIGFuZCAoYikgZG91YmxlLWNsaWNrK2RyYWcgZmVsbCB0aHJvdWdoIHRvIGNoYXItbW9kZSBzZWxlY3Rpb24uXG4gICAgY29uc3Qgbm93ID0gRGF0ZS5ub3coKVxuICAgIGNvbnN0IG5lYXJMYXN0ID1cbiAgICAgIG5vdyAtIGFwcC5sYXN0Q2xpY2tUaW1lIDwgTVVMVElfQ0xJQ0tfVElNRU9VVF9NUyAmJlxuICAgICAgTWF0aC5hYnMoY29sIC0gYXBwLmxhc3RDbGlja0NvbCkgPD0gTVVMVElfQ0xJQ0tfRElTVEFOQ0UgJiZcbiAgICAgIE1hdGguYWJzKHJvdyAtIGFwcC5sYXN0Q2xpY2tSb3cpIDw9IE1VTFRJX0NMSUNLX0RJU1RBTkNFXG4gICAgYXBwLmNsaWNrQ291bnQgPSBuZWFyTGFzdCA/IGFwcC5jbGlja0NvdW50ICsgMSA6IDFcbiAgICBhcHAubGFzdENsaWNrVGltZSA9IG5vd1xuICAgIGFwcC5sYXN0Q2xpY2tDb2wgPSBjb2xcbiAgICBhcHAubGFzdENsaWNrUm93ID0gcm93XG4gICAgaWYgKGFwcC5jbGlja0NvdW50ID49IDIpIHtcbiAgICAgIC8vIENhbmNlbCBhbnkgcGVuZGluZyBoeXBlcmxpbmstb3BlbiBmcm9tIHRoZSBmaXJzdCBjbGljayDigJQgdGhpcyBpc1xuICAgICAgLy8gYSBkb3VibGUtY2xpY2ssIG5vdCBhIHNpbmdsZS1jbGljayBvbiBhIGxpbmsuXG4gICAgICBpZiAoYXBwLnBlbmRpbmdIeXBlcmxpbmtUaW1lcikge1xuICAgICAgICBjbGVhclRpbWVvdXQoYXBwLnBlbmRpbmdIeXBlcmxpbmtUaW1lcilcbiAgICAgICAgYXBwLnBlbmRpbmdIeXBlcmxpbmtUaW1lciA9IG51bGxcbiAgICAgIH1cbiAgICAgIC8vIENhcCBhdCAzIChsaW5lIHNlbGVjdCkgZm9yIHF1YWRydXBsZSsgY2xpY2tzLlxuICAgICAgY29uc3QgY291bnQgPSBhcHAuY2xpY2tDb3VudCA9PT0gMiA/IDIgOiAzXG4gICAgICBhcHAucHJvcHMub25NdWx0aUNsaWNrKGNvbCwgcm93LCBjb3VudClcbiAgICAgIHJldHVyblxuICAgIH1cbiAgICBzdGFydFNlbGVjdGlvbihzZWwsIGNvbCwgcm93KVxuICAgIC8vIFNHUiBiaXQgMHgwOCA9IGFsdCAoeHRlcm0uanMgd2lyZXMgYWx0S2V5IGhlcmUsIG5vdCBtZXRhS2V5IOKAlCBzZWVcbiAgICAvLyBjb21tZW50IGF0IHRoZSBoeXBlcmxpbmstb3BlbiBndWFyZCBiZWxvdykuIE9uIG1hY09TIHh0ZXJtLmpzLFxuICAgIC8vIHJlY2VpdmluZyBhbHQgbWVhbnMgbWFjT3B0aW9uQ2xpY2tGb3JjZXNTZWxlY3Rpb24gaXMgT0ZGIChvdGhlcndpc2VcbiAgICAvLyB4dGVybS5qcyB3b3VsZCBoYXZlIGNvbnN1bWVkIHRoZSBldmVudCBmb3IgbmF0aXZlIHNlbGVjdGlvbikuXG4gICAgc2VsLmxhc3RQcmVzc0hhZEFsdCA9IChtLmJ1dHRvbiAmIDB4MDgpICE9PSAwXG4gICAgYXBwLnByb3BzLm9uU2VsZWN0aW9uQ2hhbmdlKClcbiAgICByZXR1cm5cbiAgfVxuXG4gIC8vIFJlbGVhc2U6IGVuZCB0aGUgZHJhZyBldmVuIGZvciBub24temVybyBidXR0b24gY29kZXMuIFNvbWUgdGVybWluYWxzXG4gIC8vIGVuY29kZSByZWxlYXNlIHdpdGggdGhlIG1vdGlvbiBiaXQgb3IgYnV0dG9uPTMgXCJubyBidXR0b25cIiAoY2FycmllZFxuICAvLyBvdmVyIGZyb20gcHJlLVNHUiBYMTAgZW5jb2RpbmcpIOKAlCBmaWx0ZXJpbmcgdGhvc2Ugd291bGQgb3JwaGFuXG4gIC8vIGlzRHJhZ2dpbmc9dHJ1ZSBhbmQgbGVhdmUgZHJhZy10by1zY3JvbGwncyB0aW1lciBydW5uaW5nIHVudGlsIHRoZVxuICAvLyBzY3JvbGwgYm91bmRhcnkuIE9ubHkgYWN0IG9uIG5vbi1sZWZ0IHJlbGVhc2VzIHdoZW4gd2UgQVJFIGRyYWdnaW5nXG4gIC8vIChzbyBhbiB1bnJlbGF0ZWQgbWlkZGxlL3JpZ2h0IGNsaWNrLXJlbGVhc2UgZG9lc24ndCB0b3VjaCBzZWxlY3Rpb24pLlxuICBpZiAoYmFzZUJ1dHRvbiAhPT0gMCkge1xuICAgIGlmICghc2VsLmlzRHJhZ2dpbmcpIHJldHVyblxuICAgIGZpbmlzaFNlbGVjdGlvbihzZWwpXG4gICAgYXBwLnByb3BzLm9uU2VsZWN0aW9uQ2hhbmdlKClcbiAgICByZXR1cm5cbiAgfVxuICBmaW5pc2hTZWxlY3Rpb24oc2VsKVxuICAvLyBOT1RFOiB1bmxpa2UgdGhlIG9sZCByZWxlYXNlLWJhc2VkIGRldGVjdGlvbiB3ZSBkbyBOT1QgcmVzZXQgY2xpY2tDb3VudFxuICAvLyBvbiByZWxlYXNlLWFmdGVyLWRyYWcuIFRoaXMgYWxpZ25zIHdpdGggTlNFdmVudC5jbGlja0NvdW50IHNlbWFudGljczpcbiAgLy8gYW4gaW50ZXJ2ZW5pbmcgZHJhZyBkb2Vzbid0IGJyZWFrIHRoZSBjbGljayBjaGFpbi4gUHJhY3RpY2FsIHVwc2lkZTpcbiAgLy8gdHJhY2twYWQgaml0dGVyIGR1cmluZyBhbiBpbnRlbmRlZCBkb3VibGUtY2xpY2sgKHByZXNz4oaSd29iYmxl4oaScmVsZWFzZVxuICAvLyDihpJwcmVzcykgbm93IGNvcnJlY3RseSByZXNvbHZlcyB0byB3b3JkLXNlbGVjdCBpbnN0ZWFkIG9mIGJyZWFraW5nIHRvIGFcbiAgLy8gZnJlc2ggc2luZ2xlIGNsaWNrLiBUaGUgbmVhckxhc3Qgd2luZG93ICg1MDBtcywgMSBjZWxsKSBib3VuZHMgdGhlXG4gIC8vIGVmZmVjdCDigJQgYSBkZWxpYmVyYXRlIGRyYWcgcGFzdCB0aGF0IGp1c3Qgc3RhcnRzIGEgZnJlc2ggY2hhaW4uXG4gIC8vIEEgcHJlc3MrcmVsZWFzZSB3aXRoIG5vIGRyYWcgaW4gY2hhciBtb2RlIGlzIGEgY2xpY2s6IGFuY2hvciBzZXQsXG4gIC8vIGZvY3VzIG51bGwg4oaSIGhhc1NlbGVjdGlvbiBmYWxzZS4gSW4gd29yZC9saW5lIG1vZGUgdGhlIHByZXNzIGFscmVhZHlcbiAgLy8gc2V0IGFuY2hvcitmb2N1cyAoaGFzU2VsZWN0aW9uIHRydWUpLCBzbyByZWxlYXNlIGp1c3Qga2VlcHMgdGhlXG4gIC8vIGhpZ2hsaWdodC4gVGhlIGFuY2hvciBjaGVjayBndWFyZHMgYWdhaW5zdCBhbiBvcnBoYW5lZCByZWxlYXNlIChub1xuICAvLyBwcmlvciBwcmVzcyDigJQgZS5nLiBidXR0b24gd2FzIGhlbGQgd2hlbiBtb3VzZSB0cmFja2luZyB3YXMgZW5hYmxlZCkuXG4gIGlmICghaGFzU2VsZWN0aW9uKHNlbCkgJiYgc2VsLmFuY2hvcikge1xuICAgIC8vIFNpbmdsZSBjbGljazogZGlzcGF0Y2ggRE9NIGNsaWNrIGltbWVkaWF0ZWx5IChjdXJzb3IgcmVwb3NpdGlvbmluZ1xuICAgIC8vIGV0Yy4gYXJlIGxhdGVuY3ktc2Vuc2l0aXZlKS4gSWYgbm8gRE9NIGhhbmRsZXIgY29uc3VtZWQgaXQsIGRlZmVyXG4gICAgLy8gdGhlIGh5cGVybGluayBjaGVjayBzbyBhIHNlY29uZCBjbGljayBjYW4gY2FuY2VsIGl0LlxuICAgIGlmICghYXBwLnByb3BzLm9uQ2xpY2tBdChjb2wsIHJvdykpIHtcbiAgICAgIC8vIFJlc29sdmUgdGhlIGh5cGVybGluayBVUkwgc3luY2hyb25vdXNseSB3aGlsZSB0aGUgc2NyZWVuIGJ1ZmZlclxuICAgICAgLy8gc3RpbGwgcmVmbGVjdHMgd2hhdCB0aGUgdXNlciBjbGlja2VkIOKAlCBkZWZlcnJpbmcgb25seSB0aGVcbiAgICAgIC8vIGJyb3dzZXItb3BlbiBzbyBkb3VibGUtY2xpY2sgY2FuIGNhbmNlbCBpdC5cbiAgICAgIGNvbnN0IHVybCA9IGFwcC5wcm9wcy5nZXRIeXBlcmxpbmtBdChjb2wsIHJvdylcbiAgICAgIC8vIHh0ZXJtLmpzIChWUyBDb2RlLCBDdXJzb3IsIFdpbmRzdXJmLCBldGMuKSBoYXMgaXRzIG93biBPU0MgOCBsaW5rXG4gICAgICAvLyBoYW5kbGVyIHRoYXQgZmlyZXMgb24gQ21kK2NsaWNrICp3aXRob3V0IGNvbnN1bWluZyB0aGUgbW91c2UgZXZlbnQqXG4gICAgICAvLyAoTGlua2lmaWVyLl9oYW5kbGVNb3VzZVVwIGNhbGxzIGxpbmsuYWN0aXZhdGUoKSBidXQgbmV2ZXJcbiAgICAgIC8vIHByZXZlbnREZWZhdWx0L3N0b3BQcm9wYWdhdGlvbikuIFRoZSBjbGljayBpcyBhbHNvIGZvcndhcmRlZCB0byB0aGVcbiAgICAgIC8vIHB0eSBhcyBTR1IsIHNvIGJvdGggVlMgQ29kZSdzIHRlcm1pbmFsTGlua01hbmFnZXIgQU5EIG91ciBoYW5kbGVyXG4gICAgICAvLyBoZXJlIHdvdWxkIG9wZW4gdGhlIFVSTCDigJQgdHdpY2UuIFdlIGNhbid0IGZpbHRlciBvbiBDbWQ6IHh0ZXJtLmpzXG4gICAgICAvLyBkcm9wcyBtZXRhS2V5IGJlZm9yZSBTR1IgZW5jb2RpbmcgKElDb3JlTW91c2VFdmVudCBoYXMgbm8gbWV0YVxuICAgICAgLy8gZmllbGQ7IHRoZSBTR1IgYml0IHdlIGNhbGwgJ21ldGEnIGlzIHdpcmVkIHRvIGFsdCkuIExldCB4dGVybS5qc1xuICAgICAgLy8gb3duIGxpbmstb3BlbmluZzsgQ21kK2NsaWNrIGlzIHRoZSBuYXRpdmUgVVggdGhlcmUgYW55d2F5LlxuICAgICAgLy8gVEVSTV9QUk9HUkFNIGlzIHRoZSBzeW5jIGZhc3QtcGF0aDsgaXNYdGVybUpzKCkgaXMgdGhlIFhUVkVSU0lPTlxuICAgICAgLy8gcHJvYmUgcmVzdWx0IChjYXRjaGVzIFNTSCArIG5vbi1WUyBDb2RlIGVtYmVkZGVycyBsaWtlIEh5cGVyKS5cbiAgICAgIGlmICh1cmwgJiYgcHJvY2Vzcy5lbnYuVEVSTV9QUk9HUkFNICE9PSAndnNjb2RlJyAmJiAhaXNYdGVybUpzKCkpIHtcbiAgICAgICAgLy8gQ2xlYXIgYW55IHByaW9yIHBlbmRpbmcgdGltZXIg4oCUIGNsaWNraW5nIGEgc2Vjb25kIGxpbmtcbiAgICAgICAgLy8gc3VwZXJzZWRlcyB0aGUgZmlyc3QgKG9ubHkgdGhlIGxhdGVzdCBjbGljayBvcGVucykuXG4gICAgICAgIGlmIChhcHAucGVuZGluZ0h5cGVybGlua1RpbWVyKSB7XG4gICAgICAgICAgY2xlYXJUaW1lb3V0KGFwcC5wZW5kaW5nSHlwZXJsaW5rVGltZXIpXG4gICAgICAgIH1cbiAgICAgICAgYXBwLnBlbmRpbmdIeXBlcmxpbmtUaW1lciA9IHNldFRpbWVvdXQoXG4gICAgICAgICAgKGFwcCwgdXJsKSA9PiB7XG4gICAgICAgICAgICBhcHAucGVuZGluZ0h5cGVybGlua1RpbWVyID0gbnVsbFxuICAgICAgICAgICAgYXBwLnByb3BzLm9uT3Blbkh5cGVybGluayh1cmwpXG4gICAgICAgICAgfSxcbiAgICAgICAgICBNVUxUSV9DTElDS19USU1FT1VUX01TLFxuICAgICAgICAgIGFwcCxcbiAgICAgICAgICB1cmwsXG4gICAgICAgIClcbiAgICAgIH1cbiAgICB9XG4gIH1cbiAgYXBwLnByb3BzLm9uU2VsZWN0aW9uQ2hhbmdlKClcbn1cbiJdLCJtYXBwaW5ncyI6IkFBQUEsT0FBT0EsS0FBSyxJQUFJQyxhQUFhLEVBQUUsS0FBS0MsU0FBUyxRQUFRLE9BQU87QUFDNUQsU0FBU0MseUJBQXlCLFFBQVEsMEJBQTBCO0FBQ3BFLFNBQVNDLGVBQWUsUUFBUSxzQkFBc0I7QUFDdEQsU0FBU0MsdUJBQXVCLFFBQVEsMkJBQTJCO0FBQ25FLFNBQVNDLFdBQVcsUUFBUSx5QkFBeUI7QUFDckQsU0FBU0MscUJBQXFCLFFBQVEsMkJBQTJCO0FBQ2pFLFNBQVNDLFFBQVEsUUFBUSxvQkFBb0I7QUFDN0MsU0FBU0MsWUFBWSxRQUFRLHNCQUFzQjtBQUNuRCxTQUFTQyxVQUFVLFFBQVEsMEJBQTBCO0FBQ3JELFNBQVNDLGtCQUFrQixRQUFRLG1DQUFtQztBQUN0RSxTQUNFQyxhQUFhLEVBQ2IsS0FBS0MsV0FBVyxFQUNoQixLQUFLQyxTQUFTLEVBQ2QsS0FBS0MsV0FBVyxFQUNoQkMsdUJBQXVCLFFBQ2xCLHNCQUFzQjtBQUM3QixPQUFPQyxVQUFVLE1BQU0sa0JBQWtCO0FBQ3pDLFNBQ0VDLGVBQWUsRUFDZkMsWUFBWSxF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
