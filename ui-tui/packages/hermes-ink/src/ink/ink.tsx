import { closeSync, constants as fsConstants, openSync, readSync, writeSync } from 'fs'
import { format } from 'util'

import autoBind from 'auto-bind'
import noop from 'lodash-es/noop.js'
import throttle from 'lodash-es/throttle.js'
import React, { type ReactNode } from 'react'
import type { FiberRoot } from 'react-reconciler'
import { ConcurrentRoot } from 'react-reconciler/constants.js'
import { onExit } from 'signal-exit'

import { flushInteractionTime } from '../bootstrap/state.js'
import { getYogaCounters } from '../native-ts/yoga-layout/index.js'
import { logForDebugging } from '../utils/debug.js'
import { logError } from '../utils/log.js'

import { colorize } from './colorize.js'
import App from './components/App.js'
import type { CursorDeclaration, CursorDeclarationSetter } from './components/CursorDeclarationContext.js'
import { FRAME_INTERVAL_MS } from './constants.js'
import * as dom from './dom.js'
import { markDirty } from './dom.js'
import { KeyboardEvent } from './events/keyboard-event.js'
import { FocusManager } from './focus.js'
import { emptyFrame, type Frame, type FrameEvent } from './frame.js'
import { dispatchClick, dispatchHover, dispatchMouse } from './hit-test.js'
import instances from './instances.js'
import { LogUpdate } from './log-update.js'
import { nodeCache } from './node-cache.js'
import { optimize } from './optimizer.js'
import Output from './output.js'
import type { ParsedKey } from './parse-keypress.js'
import reconciler, {
  dispatcher,
  getLastCommitMs,
  getLastYogaMs,
  recordYogaMs,
  resetProfileCounters
} from './reconciler.js'
import renderNodeToOutput, { consumeFollowScroll, didLayoutShift } from './render-node-to-output.js'
import { applyPositionedHighlight, type MatchPosition, scanPositions } from './render-to-screen.js'
import createRenderer, { type Renderer } from './renderer.js'
import {
  cellAt,
  CellWidth,
  CharPool,
  createScreen,
  HyperlinkPool,
  isEmptyCellAt,
  migrateScreenPools,
  StylePool
} from './screen.js'
import { applySearchHighlight } from './searchHighlight.js'
import {
  applySelectionOverlay,
  captureScrolledRows,
  clearSelection,
  createSelectionState,
  extendSelection,
  findPlainTextUrlAt,
  type FocusMove,
  getSelectedText,
  hasSelection,
  moveFocus,
  selectionBounds,
  selectionSignature,
  type SelectionState,
  selectLineAt,
  selectWordAt,
  shiftAnchor,
  shiftSelection,
  shiftSelectionForFollow,
  startSelection,
  updateSelection
} from './selection.js'
import {
  needsAltScreenResizeScrollbackClear,
  supportsExtendedKeys,
  SYNC_OUTPUT_SUPPORTED,
  type Terminal,
  writeDiffToTerminal
} from './terminal.js'
import {
  CURSOR_HOME,
  cursorMove,
  cursorPosition,
  DISABLE_KITTY_KEYBOARD,
  DISABLE_MODIFY_OTHER_KEYS,
  ENABLE_KITTY_KEYBOARD,
  ENABLE_MODIFY_OTHER_KEYS,
  ERASE_SCREEN,
  ERASE_SCROLLBACK
} from './termio/csi.js'
import {
  DBP,
  DFE,
  DISABLE_MOUSE_TRACKING,
  ENABLE_MOUSE_TRACKING,
  ENTER_ALT_SCREEN,
  EXIT_ALT_SCREEN,
  SHOW_CURSOR
} from './termio/dec.js'
import {
  CLEAR_ITERM2_PROGRESS,
  CLEAR_TAB_STATUS,
  setClipboard,
  supportsTabStatus,
  wrapForMultiplexer
} from './termio/osc.js'
import { TerminalWriteProvider } from './useTerminalNotification.js'

// Alt-screen: renderer.ts sets cursor.visible = !isTTY || screen.height===0,
// which is always false in alt-screen (TTY + content fills screen).
// Reusing a frozen object saves 1 allocation per frame.
const ALT_SCREEN_ANCHOR_CURSOR = Object.freeze({
  x: 0,
  y: 0,
  visible: false
})

const CURSOR_HOME_PATCH = Object.freeze({
  type: 'stdout' as const,
  content: CURSOR_HOME
})

const ERASE_THEN_HOME_PATCH = Object.freeze({
  type: 'stdout' as const,
  content: ERASE_SCREEN + CURSOR_HOME
})

const DEEP_ERASE_THEN_HOME_PATCH = Object.freeze({
  type: 'stdout' as const,
  content: ERASE_SCREEN + ERASE_SCROLLBACK + CURSOR_HOME
})

// Cached per-Ink-instance, invalidated on resize. frame.cursor.y for
// alt-screen is always terminalRows - 1 (renderer.ts).
function makeAltScreenParkPatch(terminalRows: number) {
  return Object.freeze({
    type: 'stdout' as const,
    content: cursorPosition(terminalRows, 1)
  })
}

export type Options = {
  stdout: NodeJS.WriteStream
  stdin: NodeJS.ReadStream
  stderr: NodeJS.WriteStream
  exitOnCtrlC: boolean
  patchConsole: boolean
  waitUntilExit?: () => Promise<void>
  onFrame?: (event: FrameEvent) => void
}
export default class Ink {
  private readonly log: LogUpdate
  private readonly terminal: Terminal
  private scheduleRender: (() => void) & {
    cancel?: () => void
  }
  // Ignore last render after unmounting a tree to prevent empty output before exit
  private isUnmounted = false
  private isPaused = false
  private readonly container: FiberRoot
  private rootNode: dom.DOMElement
  readonly focusManager: FocusManager
  private renderer: Renderer
  private readonly stylePool: StylePool
  private charPool: CharPool
  private hyperlinkPool: HyperlinkPool
  private exitPromise?: Promise<void>
  private restoreConsole?: () => void
  private restoreStderr?: () => void
  private readonly unsubscribeTTYHandlers?: () => void
  private terminalColumns: number
  private terminalRows: number
  private currentNode: ReactNode = null
  private frontFrame: Frame
  private backFrame: Frame
  private lastPoolResetTime = performance.now()
  private drainTimer: ReturnType<typeof setTimeout> | null = null
  // Write-drain telemetry: pendingWriteStart is the performance.now() of
  // the most recent stdout.write waiting for its drain callback.  Set to
  // null when the callback fires (drained).  Read on the NEXT frame and
  // reported as prevFrameDrainMs so the FrameEvent records how long the
  // previous write took to actually hit the terminal — distinguishes
  // "queued in Node" (write returned true) from "terminal accepted bytes"
  // (callback fired).
  private pendingWriteStart: number | null = null
  private lastDrainMs = 0
  private lastYogaCounters: {
    ms: number
    visited: number
    measured: number
    cacheHits: number
    live: number
  } = {
    ms: 0,
    visited: 0,
    measured: 0,
    cacheHits: 0,
    live: 0
  }
  private altScreenParkPatch: Readonly<{
    type: 'stdout'
    content: string
  }>
  // Text selection state (alt-screen only). Owned here so the overlay
  // pass in onRender can read it and App.tsx can update it from mouse
  // events. Public so instances.get() callers can access.
  readonly selection: SelectionState = createSelectionState()
  // Search highlight query (alt-screen only). Setter below triggers
  // scheduleRender; applySearchHighlight in onRender inverts matching cells.
  private searchHighlightQuery = ''
  // Position-based highlight. VML scans positions ONCE (via
  // scanElementSubtree, when the target message is mounted), stores them
  // message-relative, sets this for every-frame apply. rowOffset =
  // message's current screen-top. currentIdx = which position is
  // "current" (yellow). null clears. Positions are known upfront —
  // navigation is index arithmetic, no scan-feedback loop.
  private searchPositions: {
    positions: MatchPosition[]
    rowOffset: number
    currentIdx: number
  } | null = null
  // React-land subscribers for selection state changes (useHasSelection).
  // Fired alongside the terminal repaint whenever the selection mutates
  // so UI (e.g. footer hints) can react to selection appearing/clearing.
  private readonly selectionListeners = new Set<() => void>()
  private selectionVersion = 0
  private lastSelectionSignature = ''
  // DOM nodes currently under the pointer (mode-1003 motion). Held here
  // so App.tsx's handleMouseEvent is stateless — dispatchHover diffs
  // against this set and mutates it in place.
  private readonly hoveredNodes = new Set<dom.DOMElement>()
  // Set by <AlternateScreen> via setAltScreenActive(). Controls the
  // renderer's cursor.y clamping (keeps cursor in-viewport to avoid
  // LF-induced scroll when screen.height === terminalRows) and gates
  // alt-screen-aware SIGCONT/resize/unmount handling.
  private altScreenActive = false
  // Set alongside altScreenActive so SIGCONT resume knows whether to
  // re-enable mouse tracking (not all <AlternateScreen> uses want it).
  private altScreenMouseTracking = false
  // True when the previous frame's screen buffer cannot be trusted for
  // blit — selection overlay mutated it, resetFramesForAltScreen()
  // replaced it with blanks, or forceRedraw() reset it to 0×0. Forces
  // one full-render frame; steady-state frames after clear it and regain
  // the blit + narrow-damage fast path.
  private prevFrameContaminated = false
  // Set by handleResize: prepend ERASE_SCREEN to the next onRender's patches
  // INSIDE the BSU/ESU block so clear+paint is atomic. Writing ERASE_SCREEN
  // synchronously in handleResize would leave the screen blank for the ~80ms
  // render() takes; deferring into the atomic block means old content stays
  // visible until the new frame is fully ready.
  private needsEraseBeforePaint = false
  // Native cursor positioning: a component (via useDeclaredCursor) declares
  // where the terminal cursor should be parked after each frame. Terminal
  // emulators render IME preedit text at the physical cursor position, and
  // screen readers / screen magnifiers track it — so parking at the text
  // input's caret makes CJK input appear inline and lets a11y tools follow.
  private cursorDeclaration: CursorDeclaration | null = null
  // Main-screen: physical cursor position after the declared-cursor move,
  // tracked separately from frame.cursor (which must stay at content-bottom
  // for log-update's relative-move invariants). Alt-screen doesn't need
  // this — every frame begins with CSI H. null = no move emitted last frame.
  private displayCursor: {
    x: number
    y: number
  } | null = null
  // Burst of SIGWINCH (vscode panel drag) → one React commit per
  // microtask. Dims are captured sync in handleResize; only the
  // expensive tree rebuild defers.
  private pendingResizeRender = false
  private resizeSettleTimer: ReturnType<typeof setTimeout> | null = null

  // Fold synchronous re-entry (selection fanout, onFrame callback)
  // into one follow-up microtask instead of stacking renders.
  private isRendering = false
  private immediateRerenderRequested = false
  private selectionDragCell: { col: number; row: number } | null = null
  private selectionAutoScrollTimer: ReturnType<typeof setInterval> | null = null
  private selectionAutoScrollDir: -1 | 0 | 1 = 0
  constructor(private readonly options: Options) {
    autoBind(this)

    if (this.options.patchConsole) {
      this.restoreConsole = this.patchConsole()
      this.restoreStderr = this.patchStderr()
    }

    this.terminal = {
      stdout: options.stdout,
      stderr: options.stderr
    }
    this.terminalColumns = options.stdout.columns || 80
    this.terminalRows = options.stdout.rows || 24
    this.altScreenParkPatch = makeAltScreenParkPatch(this.terminalRows)
    this.stylePool = new StylePool()
    this.charPool = new CharPool()
    this.hyperlinkPool = new HyperlinkPool()
    this.frontFrame = emptyFrame(
      this.terminalRows,
      this.terminalColumns,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    )
    this.backFrame = emptyFrame(
      this.terminalRows,
      this.terminalColumns,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    )
    this.log = new LogUpdate({
      isTTY: (options.stdout.isTTY as boolean | undefined) || false,
      stylePool: this.stylePool
    })

    // scheduleRender is called from the reconciler's resetAfterCommit, which
    // runs BEFORE React's layout phase (ref attach + useLayoutEffect). Any
    // state set in layout effects — notably the cursorDeclaration from
    // useDeclaredCursor — would lag one commit behind if we rendered
    // synchronously. Deferring to a microtask runs onRender after layout
    // effects have committed, so the native cursor tracks the caret without
    // a one-keystroke lag. Same event-loop tick, so throughput is unchanged.
    // Test env uses onImmediateRender (direct onRender, no throttle) so
    // existing synchronous lastFrame() tests are unaffected.
    const deferredRender = (): void => queueMicrotask(this.onRender)
    this.scheduleRender = throttle(deferredRender, FRAME_INTERVAL_MS, {
      leading: true,
      trailing: true
    })

    // Ignore last render after unmounting a tree to prevent empty output before exit
    this.isUnmounted = false

    // Unmount when process exits
    this.unsubscribeExit = onExit(this.unmount, {
      alwaysLast: false
    })

    if (options.stdout.isTTY) {
      options.stdout.on('resize', this.handleResize)
      process.on('SIGCONT', this.handleResume)

      this.unsubscribeTTYHandlers = () => {
        options.stdout.off('resize', this.handleResize)
        process.off('SIGCONT', this.handleResume)
      }
    }

    this.rootNode = dom.createNode('ink-root')
    this.focusManager = new FocusManager((target, event) => dispatcher.dispatchDiscrete(target, event))
    this.rootNode.focusManager = this.focusManager
    this.renderer = createRenderer(this.rootNode, this.stylePool)
    this.rootNode.onRender = this.scheduleRender
    this.rootNode.onImmediateRender = this.onRender

    this.rootNode.onComputeLayout = () => {
      // Calculate layout during React's commit phase so useLayoutEffect hooks
      // have access to fresh layout data
      // Guard against accessing freed Yoga nodes after unmount
      if (this.isUnmounted) {
        return
      }

      if (this.rootNode.yogaNode) {
        const t0 = performance.now()
        this.rootNode.yogaNode.setWidth(this.terminalColumns)
        this.rootNode.yogaNode.calculateLayout(this.terminalColumns)
        const ms = performance.now() - t0
        recordYogaMs(ms)
        const c = getYogaCounters()
        this.lastYogaCounters = {
          ms,
          ...c
        }
      }
    }

    this.container = reconciler.createContainer(
      this.rootNode,
      ConcurrentRoot,
      null,
      false,
      null,
      'id',
      noop,
      // onUncaughtError
      noop,
      // onCaughtError
      noop,
      // onRecoverableError
      noop // onDefaultTransitionIndicator
    )

    if (process.env.NODE_ENV === 'development') {
      reconciler.injectIntoDevTools({
        bundleType: 0,
        // Reporting React DOM's version, not Ink's
        // See https://github.com/facebook/react/issues/16666#issuecomment-532639905
        version: '16.13.1',
        rendererPackageName: 'ink'
      })
    }
  }
  private handleResume = () => {
    if (!this.options.stdout.isTTY) {
      return
    }

    // Alt screen: after SIGCONT, content is stale (shell may have written
    // to main screen, switching focus away) and mouse tracking was
    // disabled by handleSuspend.
    if (this.altScreenActive) {
      this.reenterAltScreen()

      return
    }

    // Main screen: start fresh to prevent clobbering terminal content
    this.frontFrame = emptyFrame(
      this.frontFrame.viewport.height,
      this.frontFrame.viewport.width,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    )
    this.backFrame = emptyFrame(
      this.backFrame.viewport.height,
      this.backFrame.viewport.width,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    )
    this.log.reset()
    // Physical cursor position is unknown after the shell took over during
    // suspend. Clear displayCursor so the next frame's cursor preamble
    // doesn't emit a relative move from a stale park position.
    this.displayCursor = null
  }

  // Dims captured sync — closes the stale-dim window the original
  // debounce rejection warned about. Expensive React commit defers to
  // one microtask per burst: vscode fires many SIGWINCHes per panel
  // drag, each ~80ms uncoalesced = event loop visibly locks up.
  private handleResize = () => {
    const cols = this.options.stdout.columns || 80
    const rows = this.options.stdout.rows || 24

    // Terminals often emit 2+ resize events for one user action (window
    // settling). Same-dimension events are no-ops; skip to avoid redundant
    // frame resets and renders.
    if (cols === this.terminalColumns && rows === this.terminalRows) {
      return
    }

    this.terminalColumns = cols
    this.terminalRows = rows
    this.altScreenParkPatch = makeAltScreenParkPatch(this.terminalRows)

    // Pending throttled/drain work captured stale dims — cancel so
    // the upcoming microtask owns the next frame.
    this.scheduleRender.cancel?.()

    if (this.drainTimer !== null) {
      clearTimeout(this.drainTimer)
      this.drainTimer = null
    }

    if (this.resizeSettleTimer !== null) {
      clearTimeout(this.resizeSettleTimer)
      this.resizeSettleTimer = null
    }

    // Alt screen: reset frame buffers so the next render repaints from
    // scratch (prevFrameContaminated → every cell written, wrapped in
    // BSU/ESU — old content stays visible until the new frame swaps
    // atomically). Re-assert mouse tracking (some emulators reset it on
    // resize). Do NOT write ENTER_ALT_SCREEN: iTerm2 treats ?1049h as a
    // buffer clear even when already in alt — that's the blank flicker.
    // Self-healing re-entry (if something kicked us out of alt) is handled
    // by handleResume (SIGCONT) and the sleep-wake detector; resize itself
    // doesn't exit alt-screen. Do NOT write ERASE_SCREEN: render() below
    // can take ~80ms; erasing first leaves the screen blank that whole time.
    if (this.altScreenActive && !this.isPaused && this.options.stdout.isTTY) {
      if (this.altScreenMouseTracking) {
        this.options.stdout.write(ENABLE_MOUSE_TRACKING)
      }

      this.resetFramesForAltScreen()
      this.needsEraseBeforePaint = true

      // One last repaint after the resize burst settles closes any host-side
      // reflow drift the normal diff path can't see.
      this.resizeSettleTimer = setTimeout(() => {
        this.resizeSettleTimer = null

        if (!this.canAltScreenRepaint()) {
          return
        }

        this.resetFramesForAltScreen()
        this.needsEraseBeforePaint = true
        this.render(this.currentNode!)
      }, 160)
    }

    // Already queued: later events in this burst updated dims/alt-screen
    // prep above; the queued render picks up the latest values when it
    // fires (React commit → onComputeLayout → scheduleRender → onRender).
    if (this.pendingResizeRender) {
      return
    }

    this.pendingResizeRender = true

    queueMicrotask(() => {
      this.pendingResizeRender = false

      if (this.isUnmounted || this.currentNode === null) {
        return
      }

      this.render(this.currentNode)
    })
  }

  private canAltScreenRepaint(): boolean {
    return (
      !this.isUnmounted &&
      !this.isPaused &&
      this.altScreenActive &&
      !!this.options.stdout.isTTY &&
      this.currentNode !== null
    )
  }

  resolveExitPromise: () => void = () => {}
  rejectExitPromise: (reason?: Error) => void = () => {}
  unsubscribeExit: () => void = () => {}

  /**
   * Pause Ink and hand the terminal over to an external TUI (e.g. git
   * commit editor). In non-fullscreen mode this enters the alt screen;
   * in fullscreen mode we're already in alt so we just clear it.
   * Call `exitAlternateScreen()` when done to restore Ink.
   */
  enterAlternateScreen(): void {
    this.pause()
    this.suspendStdin()
    this.options.stdout.write(
      // Disable extended key reporting first — editors that don't speak
      // CSI-u (e.g. nano) show "Unknown sequence" for every Ctrl-<key> if
      // kitty/modifyOtherKeys stays active. exitAlternateScreen re-enables.
      DISABLE_KITTY_KEYBOARD +
        DISABLE_MODIFY_OTHER_KEYS +
        (this.altScreenMouseTracking ? DISABLE_MOUSE_TRACKING : '') +
        // disable mouse (no-op if off)
        (this.altScreenActive ? '' : '\x1b[?1049h') +
        // enter alt (already in alt if fullscreen)
        '\x1b[?1004l' +
        // disable focus reporting
        '\x1b[0m' +
        // reset attributes
        '\x1b[?25h' +
        // show cursor
        '\x1b[2J' +
        // clear screen
        '\x1b[H' // cursor home
    )
  }

  /**
   * Resume Ink after an external TUI handoff with a full repaint.
   * In non-fullscreen mode this exits the alt screen back to main;
   * in fullscreen mode we re-enter alt and clear + repaint.
   *
   * The re-enter matters: terminal editors (vim, nano, less) write
   * smcup/rmcup (?1049h/?1049l), so even though we started in alt,
   * the editor's rmcup on exit drops us to main screen. Without
   * re-entering, the 2J below wipes the user's main-screen scrollback
   * and subsequent renders land in main — native terminal scroll
   * returns, fullscreen scroll is dead.
   */
  exitAlternateScreen(): void {
    this.options.stdout.write(
      (this.altScreenActive ? ENTER_ALT_SCREEN : '') +
        // re-enter alt — vim's rmcup dropped us to main
        '\x1b[2J' +
        // clear screen (now alt if fullscreen)
        '\x1b[H' +
        // cursor home
        (this.altScreenMouseTracking ? ENABLE_MOUSE_TRACKING : '') +
        (this.altScreenActive ? '' : '\x1b[?1049l') +
        // exit alt (non-fullscreen only)
        '\x1b[?25l' // hide cursor (Ink manages)
    )
    this.resumeStdin()

    if (this.altScreenActive) {
      this.resetFramesForAltScreen()
    } else {
      this.repaint()
    }

    this.resume()
    // Re-enable focus reporting and extended key reporting — terminal
    // editors (vim, nano, etc.) write their own modifyOtherKeys level on
    // entry and reset it on exit, leaving us unable to distinguish
    // ctrl+shift+<letter> from ctrl+<letter>. Pop-before-push keeps the
    // Kitty stack balanced (a well-behaved editor restores our entry, so
    // without the pop we'd accumulate depth on each editor round-trip).
    this.options.stdout.write(
      '\x1b[?1004h' +
        (supportsExtendedKeys() ? DISABLE_KITTY_KEYBOARD + ENABLE_KITTY_KEYBOARD + ENABLE_MODIFY_OTHER_KEYS : '')
    )
  }
  onRender() {
    if (this.isUnmounted || this.isPaused) {
      return
    }

    // Fold synchronous re-entry (selection fanout, onFrame callback)
    // into one follow-up microtask — back-to-back renders within one
    // macrotask were the freeze multiplier.
    if (this.isRendering) {
      this.immediateRerenderRequested = true

      return
    }

    this.isRendering = true

    // Entering a render cancels any pending drain tick — this render will
    // handle the drain (and re-schedule below if needed). Prevents a
    // wheel-event-triggered render AND a drain-timer render both firing.
    if (this.drainTimer !== null) {
      clearTimeout(this.drainTimer)
      this.drainTimer = null
    }

    // Flush deferred interaction-time update before rendering so we call
    // Date.now() at most once per frame instead of once per keypress.
    // Done before the render to avoid dirtying state that would trigger
    // an extra React re-render cycle.
    flushInteractionTime()
    const renderStart = performance.now()
    const terminalWidth = this.options.stdout.columns || 80
    const terminalRows = this.options.stdout.rows || 24

    const frame = this.renderer({
      frontFrame: this.frontFrame,
      backFrame: this.backFrame,
      isTTY: this.options.stdout.isTTY,
      terminalWidth,
      terminalRows,
      altScreen: this.altScreenActive,
      prevFrameContaminated: this.prevFrameContaminated
    })

    const rendererMs = performance.now() - renderStart

    // Sticky/auto-follow scrolled the ScrollBox this frame. Translate the
    // selection by the same delta so the highlight stays anchored to the
    // TEXT (native terminal behavior — the selection walks up the screen
    // as content scrolls, eventually clipping at the top). frontFrame
    // still holds the PREVIOUS frame's screen (swap is at ~500 below), so
    // captureScrolledRows reads the rows that are about to scroll out
    // before they're overwritten — the text stays copyable until the
    // selection scrolls entirely off. During drag, focus tracks the mouse
    // (screen-local) so only anchor shifts — selection grows toward the
    // mouse as the anchor walks up. After release, both ends are text-
    // anchored and move as a block.
    const follow = consumeFollowScroll()

    if (
      follow &&
      this.selection.anchor &&
      // Only translate if the selection is ON scrollbox content. Selections
      // in the footer/prompt/StickyPromptHeader are on static text — the
      // scroll doesn't move what's under them. Without this guard, a
      // footer selection would be shifted by -delta then clamped to
      // viewportBottom, teleporting it into the scrollbox. Mirror the
      // bounds check the deleted check() in ScrollKeybindingHandler had.
      this.selection.anchor.row >= follow.viewportTop &&
      this.selection.anchor.row <= follow.viewportBottom
    ) {
      const { delta, viewportTop, viewportBottom } = follow

      // captureScrolledRows and shift* are a pair: capture grabs rows about
      // to scroll off, shift moves the selection endpoint so the same rows
      // won't intersect again next frame. Capturing without shifting leaves
      // the endpoint in place, so the SAME viewport rows re-intersect every
      // frame and scrolledOffAbove grows without bound — getSelectedText
      // then returns ever-growing text on each re-copy. Keep capture inside
      // each shift branch so the pairing can't be broken by a new guard.
      if (this.selection.isDragging) {
        if (hasSelection(this.selection)) {
          captureScrolledRows(this.selection, this.frontFrame.screen, viewportTop, viewportTop + delta - 1, 'above')
        }

        shiftAnchor(this.selection, -delta, viewportTop, viewportBottom)
      } else if (
        // Flag-3 guard: the anchor check above only proves ONE endpoint is
        // on scrollbox content. A drag from row 3 (scrollbox) into the
        // footer at row 6, then release, leaves focus outside the viewport
        // — shiftSelectionForFollow would clamp it to viewportBottom,
        // teleporting the highlight from static footer into the scrollbox.
        // Symmetric check: require BOTH ends inside to translate. A
        // straddling selection falls through to NEITHER shift NOR capture:
        // the footer endpoint pins the selection, text scrolls away under
        // the highlight, and getSelectedText reads the CURRENT screen
        // contents — no accumulation. Dragging branch doesn't need this:
        // shiftAnchor ignores focus, and the anchor DOES shift (so capture
        // is correct there even when focus is in the footer).
        !this.selection.focus ||
        (this.selection.focus.row >= viewportTop && this.selection.focus.row <= viewportBottom)
      ) {
        if (hasSelection(this.selection)) {
          captureScrolledRows(this.selection, this.frontFrame.screen, viewportTop, viewportTop + delta - 1, 'above')
        }

        const cleared = shiftSelectionForFollow(this.selection, -delta, viewportTop, viewportBottom)

        // Auto-clear (both ends overshot minRow) must notify React-land
        // so useHasSelection re-renders and the footer copy/escape hint
        // disappears. notifySelectionChange() would recurse into onRender;
        // fire the listeners directly — they schedule a React update for
        // LATER, they don't re-enter this frame.
        if (cleared) {
          for (const cb of this.selectionListeners) {
            cb()
          }
        }
      }
    }

    // Selection overlay: invert cell styles in the screen buffer itself,
    // so the diff picks up selection as ordinary cell changes and
    // LogUpdate remains a pure diff engine.
    //
    // Full-screen damage (PR #20120) is a correctness backstop for the
    // sibling-resize bleed: when flexbox siblings resize between frames
    // (spinner appears → bottom grows → scrollbox shrinks), the
    // cached-clear + clip-and-cull + setCellAt damage union can miss
    // transition cells at the boundary. But that only happens when layout
    // actually SHIFTS — didLayoutShift() tracks exactly this (any node's
    // cached yoga position/size differs from current, or a child was
    // removed). Steady-state frames (spinner rotate, clock tick, text
    // stream into fixed-height box) don't shift layout, so normal damage
    // bounds are correct and diffEach only compares the damaged region.
    //
    // Selection also requires full damage: overlay writes via setCellStyleId
    // which doesn't track damage, and prev-frame overlay cells need to be
    // compared when selection moves/clears. prevFrameContaminated covers
    // the frame-after-selection-clears case.
    let selActive = false
    let hlActive = false

    if (this.altScreenActive) {
      selActive = hasSelection(this.selection)

      if (selActive) {
        applySelectionOverlay(frame.screen, this.selection, this.stylePool)
      }

      // Scan-highlight: inverse on ALL visible matches (less/vim style).
      // Position-highlight (below) overlays CURRENT (yellow) on top.
      hlActive = applySearchHighlight(frame.screen, this.searchHighlightQuery, this.stylePool)

      // Position-based CURRENT: write yellow at positions[currentIdx] +
      // rowOffset. No scanning — positions came from a prior scan when
      // the message first mounted. Message-relative + rowOffset = screen.
      if (this.searchPositions) {
        const sp = this.searchPositions

        const posApplied = applyPositionedHighlight(
          frame.screen,
          this.stylePool,
          sp.positions,
          sp.rowOffset,
          sp.currentIdx
        )

        hlActive = hlActive || posApplied
      }
    }

    // Full-damage backstop: applies on BOTH alt-screen and main-screen.
    // Layout shifts (spinner appears, status line resizes) can leave stale
    // cells at sibling boundaries that per-node damage tracking misses.
    // Selection/highlight overlays write via setCellStyleId which doesn't
    // track damage. prevFrameContaminated covers the cleanup frame.
    if (didLayoutShift() || selActive || hlActive || this.prevFrameContaminated) {
      frame.screen.damage = {
        x: 0,
        y: 0,
        width: frame.screen.width,
        height: frame.screen.height
      }
    }

    // Alt-screen: anchor the physical cursor to (0,0) before every diff.
    // All cursor moves in log-update are RELATIVE to prev.cursor; if tmux
    // (or any emulator) perturbs the physical cursor out-of-band (status
    // bar refresh, pane redraw, Cmd+K wipe), the relative moves drift and
    // content creeps up 1 row/frame. CSI H resets the physical cursor;
    // passing prev.cursor=(0,0) makes the diff compute from the same spot.
    // Self-healing against any external cursor manipulation. Main-screen
    // can't do this — cursor.y tracks scrollback rows CSI H can't reach.
    // The CSI H write is deferred until after the diff is computed so we
    // can skip it for empty diffs (no writes → physical cursor unused).
    let prevFrame = this.frontFrame

    if (this.altScreenActive) {
      prevFrame = {
        ...this.frontFrame,
        cursor: ALT_SCREEN_ANCHOR_CURSOR
      }
    }

    const tDiff = performance.now()

    const diff = this.log.render(
      prevFrame,
      frame,
      this.altScreenActive,
      // DECSTBM needs BSU/ESU atomicity — without it the outer terminal
      // renders the scrolled-but-not-yet-repainted intermediate state.
      // tmux is the main case (re-emits DECSTBM with its own timing and
      // doesn't implement DEC 2026, so SYNC_OUTPUT_SUPPORTED is false).
      SYNC_OUTPUT_SUPPORTED
    )

    const diffMs = performance.now() - tDiff
    // Swap buffers
    this.backFrame = this.frontFrame
    this.frontFrame = frame

    // Periodically reset char/hyperlink pools to prevent unbounded growth
    // during long sessions. 5 minutes is infrequent enough that the O(cells)
    // migration cost is negligible. Reuses renderStart to avoid extra clock call.
    if (renderStart - this.lastPoolResetTime > 5 * 60 * 1000) {
      this.resetPools()
      this.lastPoolResetTime = renderStart
    }

    const flickers: FrameEvent['flickers'] = []

    for (const patch of diff) {
      if (patch.type === 'clearTerminal') {
        flickers.push({
          desiredHeight: frame.screen.height,
          availableHeight: frame.viewport.height,
          reason: patch.reason
        })
      }
    }

    const tOptimize = performance.now()
    const optimized = optimize(diff)
    const optimizeMs = performance.now() - tOptimize
    const hasDiff = optimized.length > 0

    if (this.altScreenActive && hasDiff) {
      // Prepend CSI H to anchor the physical cursor to (0,0) so
      // log-update's relative moves compute from a known spot (self-healing
      // against out-of-band cursor drift, see the ALT_SCREEN_ANCHOR_CURSOR
      // comment above). Append CSI row;1 H to park the cursor at the bottom
      // row (where the prompt input is) — without this, the cursor ends
      // wherever the last diff write landed (a different row every frame),
      // making iTerm2's cursor guide flicker as it chases the cursor.
      // BSU/ESU protects content atomicity but iTerm2's guide tracks cursor
      // position independently. Parking at bottom (not 0,0) keeps the guide
      // where the user's attention is.
      //
      // After resize, prepend a clear too. The diff only writes cells
      // that changed; cells where new=blank and prev-buffer=blank get skipped
      // — but the physical terminal still has stale content there (shorter
      // lines at new width leave old-width text tails visible). Apple Terminal
      // can also preserve alt-screen reflow artifacts in scrollback during
      // resize, so it gets CSI 3J in this one recovery path. When BSU/ESU is
      // supported, the clear+paint lands atomically; otherwise the final state
      // is still healed even if the repaint is visible.
      if (this.needsEraseBeforePaint) {
        this.needsEraseBeforePaint = false
        optimized.unshift(needsAltScreenResizeScrollbackClear() ? DEEP_ERASE_THEN_HOME_PATCH : ERASE_THEN_HOME_PATCH)
      } else {
        optimized.unshift(CURSOR_HOME_PATCH)
      }

      optimized.push(this.altScreenParkPatch)
    }

    // Native cursor positioning: park the terminal cursor at the declared
    // position so IME preedit text renders inline and screen readers /
    // magnifiers can follow the input. nodeCache holds the absolute screen
    // rect populated by renderNodeToOutput this frame (including scrollTop
    // translation) — if the declared node didn't render (stale declaration
    // after remount, or scrolled out of view), it won't be in the cache
    // and no move is emitted.
    const decl = this.cursorDeclaration
    const rect = decl !== null ? nodeCache.get(decl.node) : undefined

    const target =
      decl !== null && rect !== undefined
        ? {
            x: rect.x + decl.relativeX,
            y: rect.y + decl.relativeY
          }
        : null

    const parked = this.displayCursor

    // Preserve the empty-diff zero-write fast path: skip all cursor writes
    // when nothing rendered AND the park target is unchanged.
    const targetMoved = target !== null && (parked === null || parked.x !== target.x || parked.y !== target.y)

    if (hasDiff || targetMoved || (target === null && parked !== null)) {
      // Main-screen preamble: log-update's relative moves assume the
      // physical cursor is at prevFrame.cursor. If last frame parked it
      // elsewhere, move back before the diff runs. Alt-screen's CSI H
      // already resets to (0,0) so no preamble needed.
      if (parked !== null && !this.altScreenActive && hasDiff) {
        const pdx = prevFrame.cursor.x - parked.x
        const pdy = prevFrame.cursor.y - parked.y

        if (pdx !== 0 || pdy !== 0) {
          optimized.unshift({
            type: 'stdout',
            content: cursorMove(pdx, pdy)
          })
        }
      }

      if (target !== null) {
        if (this.altScreenActive) {
          // Absolute CUP (1-indexed); next frame's CSI H resets regardless.
          // Emitted after altScreenParkPatch so the declared position wins.
          const row = Math.min(Math.max(target.y + 1, 1), terminalRows)
          const col = Math.min(Math.max(target.x + 1, 1), terminalWidth)
          optimized.push({
            type: 'stdout',
            content: cursorPosition(row, col)
          })
        } else {
          // After the diff (or preamble), cursor is at frame.cursor. If no
          // diff AND previously parked, it's still at the old park position
          // (log-update wrote nothing). Otherwise it's at frame.cursor.
          const from =
            !hasDiff && parked !== null
              ? parked
              : {
                  x: frame.cursor.x,
                  y: frame.cursor.y
                }

          const dx = target.x - from.x
          const dy = target.y - from.y

          if (dx !== 0 || dy !== 0) {
            optimized.push({
              type: 'stdout',
              content: cursorMove(dx, dy)
            })
          }
        }

        this.displayCursor = target
      } else {
        // Declaration cleared (input blur, unmount). Restore physical cursor
        // to frame.cursor before forgetting the park position — otherwise
        // displayCursor=null lies about where the cursor is, and the NEXT
        // frame's preamble (or log-update's relative moves) computes from a
        // wrong spot. The preamble above handles hasDiff; this handles
        // !hasDiff (e.g. accessibility mode where blur doesn't change
        // renderedValue since invert is identity).
        if (parked !== null && !this.altScreenActive && !hasDiff) {
          const rdx = frame.cursor.x - parked.x
          const rdy = frame.cursor.y - parked.y

          if (rdx !== 0 || rdy !== 0) {
            optimized.push({
              type: 'stdout',
              content: cursorMove(rdx, rdy)
            })
          }
        }

        this.displayCursor = null
      }
    }

    const tWrite = performance.now()

    // Capture any stale pending write BEFORE starting this frame's write —
    // if the callback already fired, pendingWriteStart is null and lastDrainMs
    // already reflects the previous frame's drain.  If it hasn't fired, we
    // report "still pending" via a non-zero duration based on now-then so
    // backpressure shows up even if Node never flushes this session.
    const staleDrain = this.pendingWriteStart !== null ? performance.now() - this.pendingWriteStart : this.lastDrainMs

    const prevFrameDrainMs = Math.round(staleDrain * 100) / 100
    this.lastDrainMs = 0

    // Only track drain on TTY. Piped/non-TTY stdout bypasses flow control.
    const trackDrain = this.options.stdout.isTTY && hasDiff
    const drainStart = trackDrain ? tWrite : 0

    if (trackDrain) {
      this.pendingWriteStart = drainStart
    }

    const { bytes: writeBytes, backpressure } = writeDiffToTerminal(
      this.terminal,
      optimized,
      this.altScreenActive && !SYNC_OUTPUT_SUPPORTED,
      trackDrain
        ? () => {
            // Callback fires once Node has flushed the chunk to the OS.
            // Capture the drain time and clear pending so the NEXT frame's
            // staleDrain = the real end-to-end flush time.
            if (this.pendingWriteStart === drainStart) {
              this.lastDrainMs = performance.now() - drainStart
              this.pendingWriteStart = null
            }
          }
        : undefined
    )

    const writeMs = performance.now() - tWrite

    // Update blit safety for the NEXT frame. The frame just rendered
    // becomes frontFrame (= next frame's prevScreen). If we applied the
    // selection overlay, that buffer has inverted cells. selActive/hlActive
    // are only ever true in alt-screen; in main-screen this is false→false.
    this.prevFrameContaminated = selActive || hlActive || !!frame.absoluteOverlayMoved

    // Plain setTimeout (not scheduleRender) — lodash throttle's leading
    // edge would fire inside this trailing invocation and double-render.
    // Scroll drain only; absolute-overlay movement rides prevFrameContaminated
    // into the next natural render. Routing it here made caret re-layout a
    // 250fps self-oscillator that locked the event loop after resize.
    if (frame.scrollDrainPending) {
      this.drainTimer = setTimeout(() => this.onRender(), FRAME_INTERVAL_MS >> 2)
    }

    const yogaMs = getLastYogaMs()
    const commitMs = getLastCommitMs()
    const yc = this.lastYogaCounters
    // Reset so drain-only frames (no React commit) don't repeat stale values.
    resetProfileCounters()
    this.lastYogaCounters = {
      ms: 0,
      visited: 0,
      measured: 0,
      cacheHits: 0,
      live: 0
    }
    this.options.onFrame?.({
      durationMs: performance.now() - renderStart,
      phases: {
        renderer: rendererMs,
        diff: diffMs,
        optimize: optimizeMs,
        write: writeMs,
        patches: diff.length,
        optimizedPatches: optimized.length,
        writeBytes,
        backpressure,
        prevFrameDrainMs,
        yoga: yogaMs,
        commit: commitMs,
        yogaVisited: yc.visited,
        yogaMeasured: yc.measured,
        yogaCacheHits: yc.cacheHits,
        yogaLive: yc.live
      },
      flickers
    })

    this.isRendering = false

    if (this.immediateRerenderRequested) {
      this.immediateRerenderRequested = false
      queueMicrotask(() => this.onRender())
    }
  }
  pause(): void {
    // Flush pending React updates and render before pausing.
    reconciler.flushSyncFromReconciler()
    this.onRender()
    this.isPaused = true
  }
  resume(): void {
    this.isPaused = false
    this.onRender()
  }

  /**
   * Reset frame buffers so the next render writes the full screen from scratch.
   * Call this before resume() when the terminal content has been corrupted by
   * an external process (e.g. tmux, shell, full-screen TUI).
   */
  repaint(): void {
    this.frontFrame = emptyFrame(
      this.frontFrame.viewport.height,
      this.frontFrame.viewport.width,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    )
    this.backFrame = emptyFrame(
      this.backFrame.viewport.height,
      this.backFrame.viewport.width,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    )
    this.log.reset()
    // Physical cursor position is unknown after external terminal corruption.
    // Clear displayCursor so the cursor preamble doesn't emit a stale
    // relative move from where we last parked it.
    this.displayCursor = null
  }

  /**
   * Clear the physical terminal and force a full redraw.
   *
   * The traditional readline ctrl+l — clears the visible screen and
   * redraws the current content. Also the recovery path when the terminal
   * was cleared externally (macOS Cmd+K) and Ink's diff engine thinks
   * unchanged cells don't need repainting. Scrollback is preserved.
   */
  forceRedraw(): void {
    if (!this.options.stdout.isTTY || this.isUnmounted || this.isPaused) {
      return
    }

    this.options.stdout.write(ERASE_SCREEN + CURSOR_HOME)

    if (this.altScreenActive) {
      this.resetFramesForAltScreen()
    } else {
      this.repaint()
      // repaint() resets frontFrame to 0×0. Without this flag the next
      // frame's blit optimization copies from that empty screen and the
      // diff sees no content. onRender resets the flag at frame end.
      this.prevFrameContaminated = true
    }

    this.onRender()
  }

  /**
   * Mark the previous frame as untrustworthy for blit, forcing the next
   * render to do a full-damage diff instead of the per-node fast path.
   *
   * Lighter than forceRedraw() — no screen clear, no extra write. Call
   * from a useLayoutEffect cleanup when unmounting a tall overlay: the
   * blit fast path can copy stale cells from the overlay frame into rows
   * the shrunken layout no longer reaches, leaving a ghost title/divider.
   * onRender resets the flag at frame end so it's one-shot.
   */
  invalidatePrevFrame(): void {
    this.prevFrameContaminated = true
  }

  /**
   * Called by the <AlternateScreen> component on mount/unmount.
   * Controls cursor.y clamping in the renderer and gates alt-screen-aware
   * behavior in SIGCONT/resize/unmount handlers. Repaints on change so
   * the first alt-screen frame (and first main-screen frame on exit) is
   * a full redraw with no stale diff state.
   */
  setAltScreenActive(active: boolean, mouseTracking = false): void {
    if (this.altScreenActive === active) {
      return
    }

    this.altScreenActive = active
    this.altScreenMouseTracking = active && mouseTracking

    if (active) {
      this.resetFramesForAltScreen()
    } else {
      this.repaint()
    }
  }

  /**
   * Toggle mouse tracking at runtime while the alt screen is active.
   * Writes the appropriate DEC reset/set sequences so the terminal
   * (and ConPTY on Windows WSL2) reflects the change immediately.
   */
  setAltScreenMouseTracking(enabled: boolean): void {
    if (this.altScreenMouseTracking === enabled) {
      return
    }

    this.altScreenMouseTracking = enabled

    if (this.altScreenActive) {
      this.options.stdout.write(enabled ? ENABLE_MOUSE_TRACKING : DISABLE_MOUSE_TRACKING)
    }
  }
  get isAltScreenActive(): boolean {
    return this.altScreenActive
  }

  /**
   * Re-assert terminal modes after a gap (>5s stdin silence or event-loop
   * stall). Catches tmux detach→attach, ssh reconnect, and laptop
   * sleep/wake — none of which send SIGCONT. The terminal may reset DEC
   * private modes on reconnect; this method restores them.
   *
   * Always re-asserts extended key reporting and mouse tracking. Mouse
   * tracking is idempotent (DEC private mode set-when-set is a no-op). The
   * Kitty keyboard protocol is NOT — CSI >1u is a stack push, so we pop
   * first to keep depth balanced (pop on empty stack is a no-op per spec,
   * so after a terminal reset this still restores depth 0→1). Without the
   * pop, each >5s idle gap adds a stack entry, and the single pop on exit
   * or suspend can't drain them — the shell is left in CSI u mode where
   * Ctrl+C/Ctrl+D leak as escape sequences. The alt-screen
   * re-entry (ERASE_SCREEN + frame reset) is NOT idempotent — it blanks the
   * screen — so it's opt-in via includeAltScreen. The stdin-gap caller fires
   * on ordinary >5s idle + keypress and must not erase; the event-loop stall
   * detector fires on genuine sleep/wake and opts in. tmux attach / ssh
   * reconnect typically send a resize, which already covers alt-screen via
   * handleResize.
   */
  reassertTerminalModes = (includeAltScreen = false): void => {
    if (!this.options.stdout.isTTY) {
      return
    }

    // Don't touch the terminal during an editor handoff — re-enabling kitty
    // keyboard here would undo enterAlternateScreen's disable and nano would
    // start seeing CSI-u sequences again.
    if (this.isPaused) {
      return
    }

    // Extended keys — re-assert if enabled (App.tsx enables these on
    // allowlisted terminals at raw-mode entry; a terminal reset clears them).
    // Pop-before-push keeps Kitty stack depth at 1 instead of accumulating
    // on each call.
    if (supportsExtendedKeys()) {
      this.options.stdout.write(DISABLE_KITTY_KEYBOARD + ENABLE_KITTY_KEYBOARD + ENABLE_MODIFY_OTHER_KEYS)
    }

    if (!this.altScreenActive) {
      return
    }

    // Mouse tracking — idempotent, safe to re-assert on every stdin gap.
    if (this.altScreenMouseTracking) {
      this.options.stdout.write(ENABLE_MOUSE_TRACKING)
    }

    // Alt-screen re-entry — destructive (ERASE_SCREEN). Only for callers that
    // have a strong signal the terminal actually dropped mode 1049.
    if (includeAltScreen) {
      this.reenterAltScreen()
    }
  }

  /**
   * Mark this instance as unmounted so future unmount() calls early-return.
   * Called by gracefulShutdown's cleanupTerminalModes() after it has sent
   * EXIT_ALT_SCREEN but before the remaining terminal-reset sequences.
   * Without this, signal-exit's deferred ink.unmount() (triggered by
   * process.exit()) runs the full unmount path: onRender() + writeSync
   * cleanup block + updateContainerSync → AlternateScreen unmount cleanup.
   * The result is 2-3 redundant EXIT_ALT_SCREEN sequences landing on the
   * main screen AFTER printResumeHint(), which tmux (at least) interprets
   * as restoring the saved cursor position — clobbering the resume hint.
   */
  detachForShutdown(): void {
    this.isUnmounted = true
    // Cancel any pending throttled render so it doesn't fire between
    // cleanupTerminalModes() and process.exit() and write to main screen.
    this.scheduleRender.cancel?.()

    // Restore stdin from raw mode. unmount() used to do this via React
    // unmount (App.componentWillUnmount → handleSetRawMode(false)) but we're
    // short-circuiting that path. Must use this.options.stdin — NOT
    // process.stdin — because getStdinOverride() may have opened /dev/tty
    // when stdin is piped.
    const stdin = this.options.stdin as NodeJS.ReadStream & {
      isRaw?: boolean
      setRawMode?: (m: boolean) => void
    }

    this.drainStdin()

    if (stdin.isTTY && stdin.isRaw && stdin.setRawMode) {
      stdin.setRawMode(false)
    }
  }

  /** @see drainStdin */
  drainStdin(): void {
    drainStdin(this.options.stdin)
  }

  /**
   * Re-enter alt-screen, clear, home, re-enable mouse tracking, and reset
   * frame buffers so the next render repaints from scratch. Self-heal for
   * SIGCONT, resize, and stdin-gap/event-loop-stall (sleep/wake) — any of
   * which can leave the terminal in main-screen mode while altScreenActive
   * stays true. ENTER_ALT_SCREEN is a terminal-side no-op if already in alt.
   */
  private reenterAltScreen(): void {
    this.options.stdout.write(
      ENTER_ALT_SCREEN + ERASE_SCREEN + CURSOR_HOME + (this.altScreenMouseTracking ? ENABLE_MOUSE_TRACKING : '')
    )
    this.resetFramesForAltScreen()
  }

  /**
   * Seed prev/back frames with full-size BLANK screens (rows×cols of empty
   * cells, not 0×0). In alt-screen mode, next.screen.height is always
   * terminalRows; if prev.screen.height is 0 (emptyFrame's default),
   * log-update sees heightDelta > 0 ('growing') and calls renderFrameSlice,
   * whose trailing per-row CR+LF at the last row scrolls the alt screen,
   * permanently desyncing the virtual and physical cursors by 1 row.
   *
   * With a rows×cols blank prev, heightDelta === 0 → standard diffEach
   * → moveCursorTo (CSI cursorMove, no LF, no scroll).
   *
   * viewport.height = rows + 1 matches the renderer's alt-screen output,
   * preventing a spurious resize trigger on the first frame. cursor.y = 0
   * matches the physical cursor after ENTER_ALT_SCREEN + CSI H (home).
   */
  private resetFramesForAltScreen(): void {
    const rows = this.terminalRows
    const cols = this.terminalColumns

    const blank = (): Frame => ({
      screen: createScreen(cols, rows, this.stylePool, this.charPool, this.hyperlinkPool),
      viewport: {
        width: cols,
        height: rows + 1
      },
      cursor: {
        x: 0,
        y: 0,
        visible: true
      }
    })

    this.frontFrame = blank()
    this.backFrame = blank()
    this.log.reset()
    // Defense-in-depth: alt-screen skips the cursor preamble anyway (CSI H
    // resets), but a stale displayCursor would be misleading if we later
    // exit to main-screen without an intervening render.
    this.displayCursor = null
    // Fresh frontFrame is blank rows×cols — blitting from it would copy
    // blanks over content. Next alt-screen frame must full-render.
    this.prevFrameContaminated = true
  }

  /**
   * Copy the current text selection to the system clipboard without clearing the
   * selection. Returns the copied text when a clipboard path succeeded (native
   * tool fired, tmux buffer loaded, or OSC 52 emitted), or '' when no path was
   * taken (e.g. headless Linux without tmux). Matches iTerm2's copy-on-select
   * behavior where the selected region stays visible after the automatic copy.
   */
  async copySelectionNoClear(): Promise<string> {
    if (!hasSelection(this.selection)) {
      return ''
    }

    const text = getSelectedText(this.selection, this.frontFrame.screen)

    if (text) {
      try {
        const { sequence, success } = await setClipboard(text)

        if (sequence) {
          this.options.stdout.write(sequence)
        }

        if (success) {
          return text
        }

        if (process.env.HERMES_TUI_DEBUG_CLIPBOARD) {
          console.error(
            '[clipboard] no path reached the clipboard (headless + no tmux?) — set HERMES_TUI_FORCE_OSC52=1 to force the escape sequence'
          )
        }
      } catch (err) {
        if (process.env.HERMES_TUI_DEBUG_CLIPBOARD) {
          console.error('[clipboard] error:', err)
        }
      }
    }

    return ''
  }

  /**
   * Copy the current text selection to the system clipboard via OSC 52
   * and clear the selection. Returns the copied text (empty if no selection
   * or clipboard operation failed).
   */
  async copySelection(): Promise<string> {
    if (!hasSelection(this.selection)) {
      return ''
    }

    const text = await this.copySelectionNoClear()
    clearSelection(this.selection)
    this.notifySelectionChange()

    return text
  }

  /** Clear the current text selection without copying. */
  clearTextSelection(): void {
    if (!hasSelection(this.selection)) {
      return
    }

    clearSelection(this.selection)
    this.notifySelectionChange()
  }

  /**
   * Set the search highlight query. Non-empty → all visible occurrences
   * are inverted (SGR 7) on the next frame; first one also underlined.
   * Empty → clears (prevFrameContaminated handles the frame after). Same
   * damage-tracking machinery as selection — setCellStyleId doesn't track
   * damage, so the overlay forces full-frame damage while active.
   */
  setSearchHighlight(query: string): void {
    if (this.searchHighlightQuery === query) {
      return
    }

    this.searchHighlightQuery = query
    this.scheduleRender()
  }

  /** Paint an EXISTING DOM subtree to a fresh Screen at its natural
   *  height, scan for query. Returns positions relative to the element's
   *  bounding box (row 0 = element top).
   *
   *  The element comes from the MAIN tree — built with all real
   *  providers, yoga already computed. We paint it to a fresh buffer
   *  with offsets so it lands at (0,0). Same paint path as the main
   *  render. Zero drift. No second React root, no context bridge.
   *
   *  ~1-2ms (paint only, no reconcile — the DOM is already built). */
  scanElementSubtree(el: dom.DOMElement): MatchPosition[] {
    if (!this.searchHighlightQuery || !el.yogaNode) {
      return []
    }

    const width = Math.ceil(el.yogaNode.getComputedWidth())
    const height = Math.ceil(el.yogaNode.getComputedHeight())

    if (width <= 0 || height <= 0) {
      return []
    }

    // renderNodeToOutput adds el's OWN computedLeft/Top to offsetX/Y.
    // Passing -elLeft/-elTop nets to 0 → paints at (0,0) in our buffer.
    const elLeft = el.yogaNode.getComputedLeft()
    const elTop = el.yogaNode.getComputedTop()
    const screen = createScreen(width, height, this.stylePool, this.charPool, this.hyperlinkPool)

    const output = new Output({
      width,
      height,
      stylePool: this.stylePool,
      screen
    })

    renderNodeToOutput(el, output, {
      offsetX: -elLeft,
      offsetY: -elTop,
      prevScreen: undefined
    })
    const rendered = output.get()
    // renderNodeToOutput wrote our offset positions to nodeCache —
    // corrupts the main render (it'd blit from wrong coords). Mark the
    // subtree dirty so the next main render repaints + re-caches
    // correctly. One extra paint of this message, but correct > fast.
    dom.markDirty(el)
    const positions = scanPositions(rendered, this.searchHighlightQuery)
    logForDebugging(
      `scanElementSubtree: q='${this.searchHighlightQuery}' ` +
        `el=${width}x${height}@(${elLeft},${elTop}) n=${positions.length} ` +
        `[${positions
          .slice(0, 10)
          .map(p => `${p.row}:${p.col}`)
          .join(',')}` +
        `${positions.length > 10 ? ',…' : ''}]`
    )

    return positions
  }

  /** Set the position-based highlight state. Every frame, writes CURRENT
   *  style at positions[currentIdx] + rowOffset. null clears. The scan-
   *  highlight (inverse on all matches) still runs — this overlays yellow
   *  on top. rowOffset changes as the user scrolls (= message's current
   *  screen-top); positions stay stable (message-relative). */
  setSearchPositions(
    state: {
      positions: MatchPosition[]
      rowOffset: number
      currentIdx: number
    } | null
  ): void {
    this.searchPositions = state
    this.scheduleRender()
  }

  /**
   * Set the selection highlight background color. Replaces the per-cell
   * SGR-7 inverse with a solid theme-aware bg (matches native terminal
   * selection). Accepts the same color formats as Text backgroundColor
   * (rgb(), ansi:name, #hex, ansi256()) — colorize() routes through
   * chalk so the tmux/xterm.js level clamps in colorize.ts apply and
   * the emitted SGR is correct for the current terminal.
   *
   * Called by React-land once theme is known (ScrollKeybindingHandler's
   * useEffect watching useTheme). Before that call, withSelectionBg
   * falls back to withInverse so selection still renders on the first
   * frame; the effect fires before any mouse input so the fallback is
   * unobservable in practice.
   */
  setSelectionBgColor(color: string): void {
    // Wrap a NUL marker, then split on it to extract the open/close SGR.
    // colorize returns the input unchanged if the color string is bad —
    // no NUL-split then, so fall through to null (inverse fallback).
    const wrapped = colorize('\0', color, 'background')
    const nul = wrapped.indexOf('\0')

    if (nul <= 0 || nul === wrapped.length - 1) {
      this.stylePool.setSelectionBg(null)

      return
    }

    this.stylePool.setSelectionBg({
      type: 'ansi',
      code: wrapped.slice(0, nul),
      endCode: wrapped.slice(nul + 1) // always \x1b[49m for bg
    })
    // No scheduleRender: this is called from a React effect that already
    // runs inside the render cycle, and the bg only matters once a
    // selection exists (which itself triggers a full-damage frame).
  }

  /**
   * Capture text from rows about to scroll out of the viewport during
   * drag-to-scroll. Must be called BEFORE the ScrollBox scrolls so the
   * screen buffer still holds the outgoing content. Accumulated into
   * the selection state and joined back in by getSelectedText.
   */
  captureScrolledRows(firstRow: number, lastRow: number, side: 'above' | 'below'): void {
    captureScrolledRows(this.selection, this.frontFrame.screen, firstRow, lastRow, side)
  }

  /**
   * Shift anchor AND focus by dRow, clamped to [minRow, maxRow]. Used by
   * keyboard scroll handlers (PgUp/PgDn etc.) so the highlight tracks the
   * content instead of disappearing. Unlike shiftAnchor (drag-to-scroll),
   * this moves BOTH endpoints — the user isn't holding the mouse at one
   * edge. Supplies screen.width for the col-reset-on-clamp boundary.
   */
  shiftSelectionForScroll(dRow: number, minRow: number, maxRow: number): void {
    const hadSel = hasSelection(this.selection)
    shiftSelection(this.selection, dRow, minRow, maxRow, this.frontFrame.screen.width)

    // shiftSelection clears when both endpoints overshoot the same edge
    // (Home/g/End/G page-jump past the selection). Notify subscribers so
    // useHasSelection updates. Safe to call notifySelectionChange here —
    // this runs from keyboard handlers, not inside onRender().
    if (hadSel && !hasSelection(this.selection)) {
      this.notifySelectionChange()
    }
  }

  /**
   * Keyboard selection extension (shift+arrow/home/end). Moves focus;
   * anchor stays fixed so the highlight grows or shrinks relative to it.
   * Left/right wrap across row boundaries — native macOS text-edit
   * behavior: shift+left at col 0 wraps to end of the previous row.
   * Up/down clamp at viewport edges (no scroll-to-extend yet). Drops to
   * char mode. No-op outside alt-screen or without an active selection.
   */
  moveSelectionFocus(move: FocusMove): void {
    if (!this.altScreenActive) {
      return
    }

    const { focus } = this.selection

    if (!focus) {
      return
    }

    const { width, height } = this.frontFrame.screen

    const maxCol = width - 1
    const maxRow = height - 1

    let { col, row } = focus

    switch (move) {
      case 'left':
        if (col > 0) {
          col--
        } else if (row > 0) {
          col = maxCol
          row--
        }

        break

      case 'right':
        if (col < maxCol) {
          col++
        } else if (row < maxRow) {
          col = 0
          row++
        }

        break

      case 'up':
        if (row > 0) {
          row--
        }

        break

      case 'down':
        if (row < maxRow) {
          row++
        }

        break

      case 'lineStart':
        col = 0

        break

      case 'lineEnd':
        col = maxCol

        break
    }

    if (col === focus.col && row === focus.row) {
      return
    }

    moveFocus(this.selection, col, row)
    this.notifySelectionChange()
  }

  /** Whether there is an active text selection. */
  hasTextSelection(): boolean {
    return hasSelection(this.selection)
  }

  getSelectionVersion(): number {
    return this.selectionVersion
  }

  /**
   * Subscribe to selection state changes. Fires whenever the selection
   * mutates — anchor/focus moves, drag updates, programmatic clears.
   * Does NOT fire on `copySelectionNoClear()` (no mutation, no notify),
   * which is why version-based subscribers don't risk re-entrant copies.
   * Returns an unsubscribe fn.
   */
  subscribeToSelectionChange(cb: () => void): () => void {
    this.selectionListeners.add(cb)

    return () => this.selectionListeners.delete(cb)
  }
  private notifySelectionChange(): void {
    this.scheduleRender()

    // Only bump version when the selection range actually mutated.
    // Listeners still fire unconditionally — useHasSelection() snapshots
    // through React, which dedupes via Object.is on the boolean value.
    const sig = selectionSignature(this.selection)

    if (sig !== this.lastSelectionSignature) {
      this.lastSelectionSignature = sig
      this.selectionVersion += 1
    }

    for (const cb of this.selectionListeners) {
      cb()
    }
  }

  /**
   * Hit-test the rendered DOM tree at (col, row) and bubble a ClickEvent
   * from the deepest hit node up through ancestors with onClick handlers.
   * Returns true if a DOM handler consumed the click. Gated on
   * altScreenActive — clicks only make sense with a fixed viewport where
   * nodeCache rects map 1:1 to terminal cells (no scrollback offset).
   */
  dispatchClick(col: number, row: number): boolean {
    if (!this.altScreenActive) {
      return false
    }

    const blank = isEmptyCellAt(this.frontFrame.screen, col, row)

    return dispatchClick(this.rootNode, col, row, blank)
  }
  dispatchMouseDown(col: number, row: number, button: number): dom.DOMElement | undefined {
    if (!this.altScreenActive) {
      return undefined
    }

    this.stopSelectionAutoScroll()

    return dispatchMouse(
      this.rootNode,
      col,
      row,
      'onMouseDown',
      button,
      isEmptyCellAt(this.frontFrame.screen, col, row)
    )
  }
  dispatchMouseUp(target: dom.DOMElement, col: number, row: number, button: number): void {
    if (!this.altScreenActive) {
      return
    }

    this.stopSelectionAutoScroll()
    dispatchMouse(this.rootNode, col, row, 'onMouseUp', button, isEmptyCellAt(this.frontFrame.screen, col, row), target)
  }
  dispatchMouseDrag(target: dom.DOMElement, col: number, row: number, button: number): void {
    if (!this.altScreenActive) {
      return
    }

    dispatchMouse(
      this.rootNode,
      col,
      row,
      'onMouseDrag',
      button,
      isEmptyCellAt(this.frontFrame.screen, col, row),
      target
    )
  }
  dispatchHover(col: number, row: number): void {
    if (!this.altScreenActive) {
      return
    }

    dispatchHover(this.rootNode, col, row, this.hoveredNodes)
  }
  dispatchKeyboardEvent(parsedKey: ParsedKey): void {
    const target = this.focusManager.activeElement ?? this.rootNode
    const event = new KeyboardEvent(parsedKey)
    dispatcher.dispatchDiscrete(target, event)

    // Tab cycling is the default action — only fires if no handler
    // called preventDefault(). Mirrors browser behavior.
    if (!event.defaultPrevented && parsedKey.name === 'tab' && !parsedKey.ctrl && !parsedKey.meta) {
      if (parsedKey.shift) {
        this.focusManager.focusPrevious(this.rootNode)
      } else {
        this.focusManager.focusNext(this.rootNode)
      }
    }
  }
  /**
   * Look up the URL at (col, row) in the current front frame. Checks for
   * an OSC 8 hyperlink first, then falls back to scanning the row for a
   * plain-text URL (mouse tracking intercepts the terminal's native
   * Cmd+Click URL detection, so we replicate it). This is a pure lookup
   * with no side effects — call it synchronously at click time so the
   * result reflects the screen the user actually clicked on, then defer
   * the browser-open action via a timer.
   */
  getHyperlinkAt(col: number, row: number): string | undefined {
    if (!this.altScreenActive) {
      return undefined
    }

    const screen = this.frontFrame.screen
    const cell = cellAt(screen, col, row)
    let url = cell?.hyperlink

    // SpacerTail cells (right half of wide/CJK/emoji chars) store the
    // hyperlink on the head cell at col-1.
    if (!url && cell?.width === CellWidth.SpacerTail && col > 0) {
      url = cellAt(screen, col - 1, row)?.hyperlink
    }

    return url ?? findPlainTextUrlAt(screen, col, row)
  }

  /**
   * Optional callback fired when clicking an OSC 8 hyperlink in fullscreen
   * mode. Set by FullscreenLayout via useLayoutEffect.
   */
  onHyperlinkClick: ((url: string) => void) | undefined

  /**
   * Stable prototype wrapper for onHyperlinkClick. Passed to <App> as
   * onOpenHyperlink so the prop is a bound method (autoBind'd) that reads
   * the mutable field at call time — not the undefined-at-render value.
   */
  openHyperlink(url: string): void {
    this.onHyperlinkClick?.(url)
  }

  /**
   * Handle a double- or triple-click at (col, row): select the word or
   * line under the cursor by reading the current screen buffer. Called on
   * PRESS (not release) so the highlight appears immediately and drag can
   * extend the selection word-by-word / line-by-line. Falls back to
   * char-mode startSelection if the click lands on a noSelect cell.
   */
  handleMultiClick(col: number, row: number, count: 2 | 3): void {
    if (!this.altScreenActive) {
      return
    }

    const screen = this.frontFrame.screen
    // selectWordAt/selectLineAt no-op on noSelect/out-of-bounds. Seed with
    // a char-mode selection so the press still starts a drag even if the
    // word/line scan finds nothing selectable.
    startSelection(this.selection, col, row)

    if (count === 2) {
      selectWordAt(this.selection, screen, col, row)
    } else {
      selectLineAt(this.selection, screen, row)
    }

    // Ensure hasSelection is true so release doesn't re-dispatch onClickAt.
    // selectWordAt no-ops on noSelect; selectLineAt no-ops out-of-bounds.
    if (!this.selection.focus) {
      this.selection.focus = this.selection.anchor
    }

    this.notifySelectionChange()
  }

  /**
   * Handle a drag-motion at (col, row). In char mode updates focus to the
   * exact cell. In word/line mode snaps to word/line boundaries so the
   * selection extends by word/line like native macOS. Gated on
   * altScreenActive for the same reason as dispatchClick.
   */
  handleSelectionDrag(col: number, row: number): void {
    if (!this.altScreenActive) {
      return
    }

    if (this.selectionDragCell?.col === col && this.selectionDragCell.row === row) {
      this.updateSelectionAutoScroll(row)

      return
    }

    this.selectionDragCell = { col, row }
    this.applySelectionDrag(col, row)
    this.updateSelectionAutoScroll(row)
  }

  private applySelectionDrag(col: number, row: number): void {
    const sel = this.selection

    if (sel.anchorSpan) {
      extendSelection(sel, this.frontFrame.screen, col, row)
    } else {
      updateSelection(sel, col, row)
    }

    this.notifySelectionChange()
  }

  private updateSelectionAutoScroll(row: number): void {
    if (!this.selection.isDragging || !this.altScreenActive) {
      this.stopSelectionAutoScroll()

      return
    }

    const dir: -1 | 0 | 1 = row <= 0 ? -1 : row >= this.terminalRows - 1 ? 1 : 0

    if (dir === 0) {
      this.stopSelectionAutoScroll()

      return
    }

    if (this.selectionAutoScrollDir === dir && this.selectionAutoScrollTimer) {
      return
    }

    this.stopSelectionAutoScroll()
    this.selectionAutoScrollDir = dir
    this.selectionAutoScrollTimer = setInterval(() => this.stepSelectionAutoScroll(), 50)
  }

  private stepSelectionAutoScroll(): void {
    if (!this.selection.isDragging || !this.altScreenActive || this.selectionAutoScrollDir === 0) {
      this.stopSelectionAutoScroll()

      return
    }

    const box = this.findPrimaryScrollBox()

    if (!box) {
      this.stopSelectionAutoScroll()

      return
    }

    const viewport = Math.max(0, box.scrollViewportHeight ?? 0)
    const max = Math.max(0, (box.scrollHeight ?? 0) - viewport)
    const current = box.scrollTop ?? 0
    const next = Math.max(0, Math.min(max, current + this.selectionAutoScrollDir))

    if (next === current) {
      return
    }

    const top = box.scrollViewportTop ?? 0
    const bottom = top + viewport - 1
    const before = selectionBounds(this.selection)

    if (before) {
      if (this.selectionAutoScrollDir > 0) {
        captureScrolledRows(this.selection, this.frontFrame.screen, top, top, 'above')
      } else {
        captureScrolledRows(this.selection, this.frontFrame.screen, bottom, bottom, 'below')
      }
    }

    box.stickyScroll = false
    box.pendingScrollDelta = undefined
    box.scrollAnchor = undefined
    box.scrollTop = next
    markDirty(box)
    shiftAnchor(this.selection, -this.selectionAutoScrollDir, top, bottom)

    if (this.selectionDragCell) {
      this.selectionDragCell = {
        col: this.selectionDragCell.col,
        row: this.selectionAutoScrollDir > 0 ? bottom : top
      }
    }

    this.applySelectionDrag(
      this.selectionDragCell?.col ?? 0,
      this.selectionDragCell?.row ?? (this.selectionAutoScrollDir > 0 ? bottom : top)
    )
  }

  private stopSelectionAutoScroll(): void {
    if (this.selectionAutoScrollTimer) {
      clearInterval(this.selectionAutoScrollTimer)
      this.selectionAutoScrollTimer = null
    }

    this.selectionAutoScrollDir = 0
    this.selectionDragCell = null
  }

  private findPrimaryScrollBox(): dom.DOMElement | undefined {
    const stack = [this.rootNode]

    while (stack.length) {
      const node = stack.shift()!

      if (
        node.style.overflowY === 'scroll' &&
        node.scrollHeight !== undefined &&
        node.scrollViewportHeight !== undefined
      ) {
        return node
      }

      for (const child of node.childNodes) {
        if (child.nodeName !== '#text') {
          stack.push(child)
        }
      }
    }
  }

  // Methods to properly suspend stdin for external editor usage
  // This is needed to prevent Ink from swallowing keystrokes when an external editor is active
  private stdinListeners: Array<{
    event: string
    listener: (...args: unknown[]) => void
  }> = []
  private wasRawMode = false
  suspendStdin(): void {
    const stdin = this.options.stdin

    if (!stdin.isTTY) {
      return
    }

    // Store and remove all 'readable' event listeners temporarily
    // This prevents Ink from consuming stdin while the editor is active
    const readableListeners = stdin.listeners('readable')
    logForDebugging(
      `[stdin] suspendStdin: removing ${readableListeners.length} readable listener(s), wasRawMode=${
        (
          stdin as NodeJS.ReadStream & {
            isRaw?: boolean
          }
        ).isRaw ?? false
      }`
    )
    readableListeners.forEach(listener => {
      this.stdinListeners.push({
        event: 'readable',
        listener: listener as (...args: unknown[]) => void
      })
      stdin.removeListener('readable', listener as (...args: unknown[]) => void)
    })

    // If raw mode is enabled, disable it temporarily
    const stdinWithRaw = stdin as NodeJS.ReadStream & {
      isRaw?: boolean
      setRawMode?: (mode: boolean) => void
    }

    if (stdinWithRaw.isRaw && stdinWithRaw.setRawMode) {
      stdinWithRaw.setRawMode(false)
      this.wasRawMode = true
    }
  }
  resumeStdin(): void {
    const stdin = this.options.stdin

    if (!stdin.isTTY) {
      return
    }

    // Re-attach all the stored listeners
    if (this.stdinListeners.length === 0 && !this.wasRawMode) {
      logForDebugging('[stdin] resumeStdin: called with no stored listeners and wasRawMode=false (possible desync)', {
        level: 'warn'
      })
    }

    logForDebugging(
      `[stdin] resumeStdin: re-attaching ${this.stdinListeners.length} listener(s), wasRawMode=${this.wasRawMode}`
    )
    this.stdinListeners.forEach(({ event, listener }) => {
      stdin.addListener(event, listener)
    })
    this.stdinListeners = []

    // Re-enable raw mode if it was enabled before
    if (this.wasRawMode) {
      const stdinWithRaw = stdin as NodeJS.ReadStream & {
        setRawMode?: (mode: boolean) => void
      }

      if (stdinWithRaw.setRawMode) {
        stdinWithRaw.setRawMode(true)
      }

      this.wasRawMode = false
    }
  }

  // Stable identity for TerminalWriteContext. An inline arrow here would
  // change on every render() call (initial mount + each resize), which
  // cascades through useContext → <AlternateScreen>'s useLayoutEffect dep
  // array → spurious exit+re-enter of the alt screen on every SIGWINCH.
  private writeRaw(data: string): void {
    this.options.stdout.write(data)
  }
  private setCursorDeclaration: CursorDeclarationSetter = (decl, clearIfNode) => {
    if (decl === null && clearIfNode !== undefined && this.cursorDeclaration?.node !== clearIfNode) {
      return
    }

    this.cursorDeclaration = decl
  }
  render(node: ReactNode): void {
    this.currentNode = node

    const tree = (
      <App
        dispatchKeyboardEvent={this.dispatchKeyboardEvent}
        exitOnCtrlC={this.options.exitOnCtrlC}
        getHyperlinkAt={this.getHyperlinkAt}
        onClickAt={this.dispatchClick}
        onCursorDeclaration={this.setCursorDeclaration}
        onExit={this.unmount}
        onHoverAt={this.dispatchHover}
        onMouseDownAt={this.dispatchMouseDown}
        onMouseDragAt={this.dispatchMouseDrag}
        onMouseUpAt={this.dispatchMouseUp}
        onMultiClick={this.handleMultiClick}
        onOpenHyperlink={this.openHyperlink}
        onSelectionChange={this.notifySelectionChange}
        onSelectionDrag={this.handleSelectionDrag}
        onStdinResume={this.reassertTerminalModes}
        selection={this.selection}
        stderr={this.options.stderr}
        stdin={this.options.stdin}
        stdout={this.options.stdout}
        terminalColumns={this.terminalColumns}
        terminalRows={this.terminalRows}
      >
        <TerminalWriteProvider value={this.writeRaw}>{node}</TerminalWriteProvider>
      </App>
    )

    reconciler.updateContainerSync(tree, this.container, null, noop)
    reconciler.flushSyncWork()
  }
  unmount(error?: Error | number | null): void {
    if (this.isUnmounted) {
      return
    }

    this.onRender()
    this.unsubscribeExit()

    if (typeof this.restoreConsole === 'function') {
      this.restoreConsole()
    }

    this.restoreStderr?.()
    this.unsubscribeTTYHandlers?.()

    // Non-TTY environments don't handle erasing ansi escapes well, so it's better to
    // only render last frame of non-static output
    const diff = this.log.renderPreviousOutput_DEPRECATED(this.frontFrame)
    writeDiffToTerminal(this.terminal, optimize(diff))

    // Clean up terminal modes synchronously before process exit.
    // React's componentWillUnmount won't run in time when process.exit() is called,
    // so we must reset terminal modes here to prevent escape sequence leakage.
    // Use writeSync to stdout (fd 1) to ensure writes complete before exit.
    // We unconditionally send all disable sequences because terminal detection
    // may not work correctly (e.g., in tmux, screen) and these are no-ops on
    // terminals that don't support them.

    if (this.options.stdout.isTTY) {
      if (this.altScreenActive) {
        // <AlternateScreen>'s unmount effect won't run during signal-exit.
        // Exit alt screen FIRST so other cleanup sequences go to the main screen.
        writeSync(1, EXIT_ALT_SCREEN)
      }

      // Disable mouse tracking — unconditional because altScreenActive can be
      // stale if AlternateScreen's unmount (which flips the flag) raced a
      // blocked event loop + SIGINT. No-op if tracking was never enabled.
      writeSync(1, DISABLE_MOUSE_TRACKING)
      // Drain stdin so in-flight mouse events don't leak to the shell
      this.drainStdin()
      // Disable extended key reporting (both kitty and modifyOtherKeys)
      writeSync(1, DISABLE_MODIFY_OTHER_KEYS)
      writeSync(1, DISABLE_KITTY_KEYBOARD)
      // Disable focus events (DECSET 1004)
      writeSync(1, DFE)
      // Disable bracketed paste mode
      writeSync(1, DBP)
      // Show cursor
      writeSync(1, SHOW_CURSOR)
      // Clear iTerm2 progress bar
      writeSync(1, CLEAR_ITERM2_PROGRESS)

      // Clear tab status (OSC 21337) so a stale dot doesn't linger
      if (supportsTabStatus()) {
        writeSync(1, wrapForMultiplexer(CLEAR_TAB_STATUS))
      }
    }

    this.isUnmounted = true

    // Cancel any pending throttled renders to prevent accessing freed Yoga nodes
    this.scheduleRender.cancel?.()

    if (this.drainTimer !== null) {
      clearTimeout(this.drainTimer)
      this.drainTimer = null
    }

    if (this.resizeSettleTimer !== null) {
      clearTimeout(this.resizeSettleTimer)
      this.resizeSettleTimer = null
    }

    reconciler.updateContainerSync(null, this.container, null, noop)
    reconciler.flushSyncWork()
    instances.delete(this.options.stdout)

    // Free the root yoga node, then clear its reference. Children are already
    // freed by the reconciler's removeChildFromContainer; using .free() (not
    // .freeRecursive()) avoids double-freeing them.
    this.rootNode.yogaNode?.free()
    this.rootNode.yogaNode = undefined

    if (error instanceof Error) {
      this.rejectExitPromise(error)
    } else {
      this.resolveExitPromise()
    }
  }
  async waitUntilExit(): Promise<void> {
    this.exitPromise ||= new Promise((resolve, reject) => {
      this.resolveExitPromise = resolve
      this.rejectExitPromise = reject
    })

    return this.exitPromise
  }
  resetLineCount(): void {
    if (this.options.stdout.isTTY) {
      // Swap so old front becomes back (for screen reuse), then reset front
      this.backFrame = this.frontFrame
      this.frontFrame = emptyFrame(
        this.frontFrame.viewport.height,
        this.frontFrame.viewport.width,
        this.stylePool,
        this.charPool,
        this.hyperlinkPool
      )
      this.log.reset()
      // frontFrame is reset, so frame.cursor on the next render is (0,0).
      // Clear displayCursor so the preamble doesn't compute a stale delta.
      this.displayCursor = null
    }
  }

  /**
   * Replace char/hyperlink pools with fresh instances to prevent unbounded
   * growth during long sessions. Migrates the front frame's screen IDs into
   * the new pools so diffing remains correct. The back frame doesn't need
   * migration — resetScreen zeros it before any reads.
   *
   * Call between conversation turns or periodically.
   */
  resetPools(): void {
    this.charPool = new CharPool()
    this.hyperlinkPool = new HyperlinkPool()
    migrateScreenPools(this.frontFrame.screen, this.charPool, this.hyperlinkPool)
    // Back frame's data is zeroed by resetScreen before reads, but its pool
    // references are used by the renderer to intern new characters. Point
    // them at the new pools so the next frame's IDs are comparable.
    this.backFrame.screen.charPool = this.charPool
    this.backFrame.screen.hyperlinkPool = this.hyperlinkPool
  }
  patchConsole(): () => void {
    // biome-ignore lint/suspicious/noConsole: intentionally patching global console
    const con = console
    const originals: Partial<Record<keyof Console, Console[keyof Console]>> = {}
    const toDebug = (...args: unknown[]) => logForDebugging(`console.log: ${format(...args)}`)
    const toError = (...args: unknown[]) => logError(new Error(`console.error: ${format(...args)}`))

    for (const m of CONSOLE_STDOUT_METHODS) {
      originals[m] = con[m]
      con[m] = toDebug
    }

    for (const m of CONSOLE_STDERR_METHODS) {
      originals[m] = con[m]
      con[m] = toError
    }

    originals.assert = con.assert

    con.assert = (condition: unknown, ...args: unknown[]) => {
      if (!condition) {
        toError(...args)
      }
    }

    return () => Object.assign(con, originals)
  }

  /**
   * Intercept process.stderr.write so stray writes (config.ts, hooks.ts,
   * third-party deps) don't corrupt the alt-screen buffer. patchConsole only
   * hooks console.* methods — direct stderr writes bypass it, land at the
   * parked cursor, scroll the alt-screen, and desync frontFrame from the
   * physical terminal. Next diff writes only changed-in-React cells at
   * absolute coords → interleaved garbage.
   *
   * Swallows the write (routes text to the debug log) and, in alt-screen,
   * forces a full-damage repaint as a defensive recovery. Not patching
   * process.stdout — Ink itself writes there.
   */
  private patchStderr(): () => void {
    const stderr = process.stderr
    const originalWrite = stderr.write
    let reentered = false

    const intercept = (
      chunk: Uint8Array | string,
      encodingOrCb?: BufferEncoding | ((err?: Error | null) => void),
      cb?: (err?: Error | null) => void
    ): boolean => {
      const callback = typeof encodingOrCb === 'function' ? encodingOrCb : cb

      // Reentrancy guard: logForDebugging → writeToStderr → here. Pass
      // through to the original so --debug-to-stderr still works and we
      // don't stack-overflow.
      if (reentered) {
        const encoding = typeof encodingOrCb === 'string' ? encodingOrCb : undefined

        return originalWrite.call(stderr, chunk, encoding, callback)
      }

      reentered = true

      try {
        const text = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8')
        logForDebugging(`[stderr] ${text}`, {
          level: 'warn'
        })

        if (this.altScreenActive && !this.isUnmounted && !this.isPaused) {
          this.prevFrameContaminated = true
          this.scheduleRender()
        }
      } finally {
        reentered = false
        callback?.()
      }

      return true
    }

    stderr.write = intercept

    return () => {
      if (stderr.write === intercept) {
        stderr.write = originalWrite
      }
    }
  }
}

/**
 * Discard pending stdin bytes so in-flight escape sequences (mouse tracking
 * reports, bracketed-paste markers) don't leak to the shell after exit.
 *
 * Two layers of trickiness:
 *
 * 1. setRawMode is termios, not fcntl — the stdin fd stays blocking, so
 *    readSync on it would hang forever. Node doesn't expose fcntl, so we
 *    open /dev/tty fresh with O_NONBLOCK (all fds to the controlling
 *    terminal share one line-discipline input queue).
 *
 * 2. By the time forceExit calls this, detachForShutdown has already put
 *    the TTY back in cooked (canonical) mode. Canonical mode line-buffers
 *    input until newline, so O_NONBLOCK reads return EAGAIN even when
 *    mouse bytes are sitting in the buffer. We briefly re-enter raw mode
 *    so reads return any available bytes, then restore cooked mode.
 *
 * Safe to call multiple times. Call as LATE as possible in the exit path:
 * DISABLE_MOUSE_TRACKING has terminal round-trip latency, so events can
 * arrive for a few ms after it's written.
 */

export function drainStdin(stdin: NodeJS.ReadStream = process.stdin): void {
  if (!stdin.isTTY) {
    return
  }

  // Drain Node's stream buffer (bytes libuv already pulled in). read()
  // returns null when empty — never blocks.
  try {
    while (stdin.read() !== null) {
      /* discard */
    }
  } catch {
    /* stream may be destroyed */
  }

  // No /dev/tty on Windows; CONIN$ doesn't support O_NONBLOCK semantics.
  // Windows Terminal also doesn't buffer mouse reports the same way.
  if (process.platform === 'win32') {
    return
  }

  // termios is per-device: flip stdin to raw so canonical-mode line
  // buffering doesn't hide partial input from the non-blocking read.
  // Restored in the finally block.
  const tty = stdin as NodeJS.ReadStream & {
    isRaw?: boolean
    setRawMode?: (raw: boolean) => void
  }

  const wasRaw = tty.isRaw === true
  // Drain the kernel TTY buffer via a fresh O_NONBLOCK fd. Bounded at 64
  // reads (64KB) — a real mouse burst is a few hundred bytes; the cap
  // guards against a terminal that ignores O_NONBLOCK.
  let fd = -1

  try {
    // setRawMode inside try: on revoked TTY (SIGHUP/SSH disconnect) the
    // ioctl throws EBADF — same recovery path as openSync/readSync below.
    if (!wasRaw) {
      tty.setRawMode?.(true)
    }

    fd = openSync('/dev/tty', fsConstants.O_RDONLY | fsConstants.O_NONBLOCK)
    const buf = Buffer.alloc(1024)

    for (let i = 0; i < 64; i++) {
      if (readSync(fd, buf, 0, buf.length, null) <= 0) {
        break
      }
    }
  } catch {
    // EAGAIN (buffer empty — expected), ENXIO/ENOENT (no controlling tty),
    // EBADF/EIO (TTY revoked — SIGHUP, SSH disconnect)
  } finally {
    if (fd >= 0) {
      try {
        closeSync(fd)
      } catch {
        /* ignore */
      }
    }

    if (!wasRaw) {
      try {
        tty.setRawMode?.(false)
      } catch {
        /* TTY may be gone */
      }
    }
  }
}

const CONSOLE_STDOUT_METHODS = [
  'log',
  'info',
  'debug',
  'dir',
  'dirxml',
  'count',
  'countReset',
  'group',
  'groupCollapsed',
  'groupEnd',
  'table',
  'time',
  'timeEnd',
  'timeLog'
] as const

const CONSOLE_STDERR_METHODS = ['warn', 'error', 'trace'] as const
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJuYW1lcyI6WyJhdXRvQmluZCIsImNsb3NlU3luYyIsImNvbnN0YW50cyIsImZzQ29uc3RhbnRzIiwib3BlblN5bmMiLCJyZWFkU3luYyIsIndyaXRlU3luYyIsIm5vb3AiLCJ0aHJvdHRsZSIsIlJlYWN0IiwiUmVhY3ROb2RlIiwiRmliZXJSb290IiwiQ29uY3VycmVudFJvb3QiLCJvbkV4aXQiLCJmbHVzaEludGVyYWN0aW9uVGltZSIsImdldFlvZ2FDb3VudGVycyIsImxvZ0ZvckRlYnVnZ2luZyIsImxvZ0Vycm9yIiwiZm9ybWF0IiwiY29sb3JpemUiLCJBcHAiLCJDdXJzb3JEZWNsYXJhdGlvbiIsIkN1cnNvckRlY2xhcmF0aW9uU2V0dGVyIiwiRlJBTUVfSU5URVJWQUxfTVMiLCJkb20iLCJLZXlib2FyZEV2ZW50IiwiRm9jdXNNYW5hZ2VyIiwiZW1wdHlGcmFtZSIsIkZyYW1lIiwiRnJhbWVFdmVudCIsImRpc3BhdGNoQ2xpY2siLCJkaXNwYXRjaEhvdmVyIiwiaW5zdGFuY2VzIiwiTG9nVXBkYXRlIiwibm9kZUNhY2hlIiwib3B0aW1pemUiLCJPdXRwdXQiLCJQYXJzZWRLZXkiLCJyZWNvbmNpbGVyIiwiZGlzcGF0Y2hlciIsImdldExhc3RDb21taXRNcyIsImdldExhc3RZb2dhTXMiLCJpc0RlYnVnUmVwYWludHNFbmFibGVkIiwicmVjb3JkWW9nYU1zIiwicmVzZXRQcm9maWxlQ291bnRlcnMiLCJyZW5kZXJOb2RlVG9PdXRwdXQiLCJjb25zdW1lRm9sbG93U2Nyb2xsIiwiZGlkTGF5b3V0U2hpZnQiLCJhcHBseVBvc2l0aW9uZWRIaWdobGlnaHQiLCJNYXRjaFBvc2l0aW9uIiwic2NhblBvc2l0aW9ucyIsImNyZWF0ZVJlbmRlcmVyIiwiUmVuZGVyZXIiLCJDZWxsV2lkdGgiLCJDaGFyUG9vbCIsImNlbGxBdCIsImNyZWF0ZVNjcmVlbiIsIkh5cGVybGlua1Bvb2wiLCJpc0VtcHR5Q2VsbEF0IiwibWlncmF0ZVNjcmVlblBvb2xzIiwiU3R5bGVQb29sIiwiYXBwbHlTZWFyY2hIaWdobGlnaHQiLCJhcHBseVNlbGVjdGlvbk92ZXJsYXkiLCJjYXB0dXJlU2Nyb2xsZWRSb3dzIiwiY2xlYXJTZWxlY3Rpb24iLCJjcmVhdGVTZWxlY3Rpb25TdGF0ZSIsImV4dGVuZFNlbGVjdGlvbiIsIkZvY3VzTW92ZSIsImZpbmRQbGFpblRleHRVcmxBdCIsImdldFNlbGVjdGVkVGV4dCIsImhhc1NlbGVjdGlvbiIsIm1vdmVGb2N1cyIsIlNlbGVjdGlvblN0YXRlIiwic2VsZWN0TGluZUF0Iiwic2VsZWN0V29yZEF0Iiwic2hpZnRBbmNob3IiLCJzaGlmdFNlbGVjdGlvbiIsInNoaWZ0U2VsZWN0aW9uRm9yRm9sbG93Iiwic3RhcnRTZWxlY3Rpb24iLCJ1cGRhdGVTZWxlY3Rpb24iLCJTWU5DX09VVFBVVF9TVVBQT1JURUQiLCJzdXBwb3J0c0V4dGVuZGVkS2V5cyIsIlRlcm1pbmFsIiwid3JpdGVEaWZmVG9UZXJtaW5hbCIsIkNVUlNPUl9IT01FIiwiY3Vyc29yTW92ZSIsImN1cnNvclBvc2l0aW9uIiwiRElTQUJMRV9LSVRUWV9LRVlCT0FSRCIsIkRJU0FCTEVfTU9ESUZZX09USEVSX0tFWVMiLCJFTkFCTEVfS0lUVFlfS0VZQk9BUkQiLCJFTkFCTEVfTU9ESUZZX09USEVSX0tFWVMiLCJFUkFTRV9TQ1JFRU4iLCJEQlAiLCJERkUiLCJESVNBQkxFX01PVVNFX1RSQUNLSU5HIiwiRU5BQkxFX01PVVNFX1RSQUNLSU5HIiwiRU5URVJfQUxUX1NDUkVFTiIsIkVYSVRfQUxUX1NDUkVFTiIsIlNIT1dfQ1VSU09SIiwiQ0xFQVJfSVRFUk0yX1BST0dSRVNTIiwiQ0xFQVJfVEFCX1NUQVRVUyIsInNldENsaXBib2FyZCIsInN1cHBvcnRzVGFiU3RhdHVzIiwid3JhcEZvck11bHRpcGxleGVyIiwiVGVybWluYWxXcml0ZVByb3ZpZGVyIiwiQUxUX1NDUkVFTl9BTkNIT1JfQ1VSU09SIiwiT2JqZWN0IiwiZnJlZXplIiwieCIsInkiLCJ2aXNpYmxlIiwiQ1VSU09SX0hPTUVfUEFUQ0giLCJ0eXBlIiwiY29uc3QiLCJjb250ZW50IiwiRVJBU0VfVEhFTl9IT01FX1BBVENIIiwibWFrZUFsdFNjcmVlblBhcmtQYXRjaCIsInRlcm1pbmFsUm93cyIsIk9wdGlvbnMiLCJzdGRvdXQiLCJOb2RlSlMiLCJXcml0ZVN0cmVhbSIsInN0ZGluIiwiUmVhZFN0cmVhbSIsInN0ZGVyciIsImV4aXRPbkN0cmxDIiwicGF0Y2hDb25zb2xlIiwid2FpdFVudGlsRXhpdCIsIlByb21pc2UiLCJvbkZyYW1lIiwiZXZlbnQiLCJJbmsiLCJsb2ciLCJ0ZXJtaW5hbCIsInNjaGVkdWxlUmVuZGVyIiwiY2FuY2VsIiwiaXNVbm1vdW50ZWQiLCJpc1BhdXNlZCIsImNvbnRhaW5lciIsInJvb3ROb2RlIiwiRE9NRWxlbWVudCIsImZvY3VzTWFuYWdlciIsInJlbmRlcmVyIiwic3R5bGVQb29sIiwiY2hhclBvb2wiLCJoeXBlcmxpbmtQb29sIiwiZXhpdFByb21pc2UiLCJyZXN0b3JlQ29uc29sZSIsInJlc3RvcmVTdGRlcnIiLCJ1bnN1YnNjcmliZVRUWUhhbmRsZXJzIiwidGVybWluYWxDb2x1bW5zIiwiY3VycmVudE5vZGUiLCJmcm9udEZyYW1lIiwiYmFja0ZyYW1lIiwibGFzdFBvb2xSZXNldFRpbWUiLCJwZXJmb3JtYW5jZSIsIm5vdyIsImRyYWluVGltZXIiLCJSZXR1cm5UeXBlIiwic2V0VGltZW91dCIsImxhc3RZb2dhQ291bnRlcnMiLCJtcyIsInZpc2l0ZWQiLCJtZWFzdXJlZCIsImNhY2hlSGl0cyIsImxpdmUiLCJhbHRTY3JlZW5QYXJrUGF0Y2giLCJSZWFkb25seSIsInNlbGVjdGlvbiIsInNlYXJjaEhpZ2hsaWdodFF1ZXJ5Iiwic2VhcmNoUG9zaXRpb25zIiwicG9zaXRpb25zIiwicm93T2Zmc2V0IiwiY3VycmVudElkeCIsInNlbGVjdGlvbkxpc3RlbmVycyIsIlNldCIsImhvdmVyZWROb2RlcyIsImFsdFNjcmVlbkFjdGl2ZSIsImFsdFNjcmVlbk1vdXNlVHJhY2tpbmciLCJwcmV2RnJhbWVDb250YW1pbmF0ZWQiLCJuZWVkc0VyYXNlQmVmb3JlUGFpbnQiLCJjdXJzb3JEZWNsYXJhdGlvbiIsImRpc3BsYXlDdXJzb3IiLCJjb25zdHJ1Y3RvciIsIm9wdGlvbnMiLCJwYXRjaFN0ZGVyciIsImNvbHVtbnMiLCJyb3dzIiwiaXNUVFkiLCJkZWZlcnJlZFJlbmRlciIsInF1ZXVlTWljcm90YXNrIiwib25SZW5kZXIiLCJsZWFkaW5nIiwidHJhaWxpbmciLCJ1bnN1YnNjcmliZUV4aXQiLCJ1bm1vdW50IiwiYWx3YXlzTGFzdCIsIm9uIiwiaGFuZGxlUmVzaXplIiwicHJvY2VzcyIsImhhbmRsZVJlc3VtZSIsIm9mZiIsImNyZWF0ZU5vZGUiLCJ0YXJnZXQiLCJkaXNwYXRjaERpc2NyZXRlIiwib25JbW1lZGlhdGVSZW5kZXIiLCJvbkNvbXB1dGVMYXlvdXQiLCJ5b2dhTm9kZSIsInQwIiwic2V0V2lkdGgiLCJjYWxjdWxhdGVMYXlvdXQiLCJjIiwiY3JlYXRlQ29udGFpbmVyIiwiaW5qZWN0SW50b0RldlRvb2xzIiwiYnVuZGxlVHlwZSIsInZlcnNpb24iLCJyZW5kZXJlclBhY2thZ2VOYW1lIiwicmVlbnRlckFsdFNjcmVlbiIsInZpZXdwb3J0IiwiaGVpZ2h0Iiwid2lkdGgiLCJyZXNldCIsImNvbHMiLCJ3cml0ZSIsInJlc2V0RnJhbWVzRm9yQWx0U2NyZWVuIiwicmVuZGVyIiwicmVzb2x2ZUV4aXRQcm9taXNlIiwicmVqZWN0RXhpdFByb21pc2UiLCJyZWFzb24iLCJFcnJvciIsImVudGVyQWx0ZXJuYXRlU2NyZWVuIiwicGF1c2UiLCJzdXNwZW5kU3RkaW4iLCJleGl0QWx0ZXJuYXRlU2NyZWVuIiwicmVzdW1lU3RkaW4iLCJyZXBhaW50IiwicmVzdW1lIiwiY2xlYXJUaW1lb3V0IiwicmVuZGVyU3RhcnQiLCJ0ZXJtaW5hbFdpZHRoIiwiZnJhbWUiLCJhbHRTY3JlZW4iLCJyZW5kZXJlck1zIiwiZm9sbG93IiwiYW5jaG9yIiwicm93Iiwidmlld3BvcnRUb3AiLCJ2aWV3cG9ydEJvdHRvbSIsImRlbHRhIiwiaXNEcmFnZ2luZyIsInNjcmVlbiIsImZvY3VzIiwiY2xlYXJlZCIsImNiIiwic2VsQWN0aXZlIiwiaGxBY3RpdmUiLCJzcCIsInBvc0FwcGxpZWQiLCJkYW1hZ2UiLCJwcmV2RnJhbWUiLCJjdXJzb3IiLCJ0RGlmZiIsImRpZmYiLCJkaWZmTXMiLCJyZXNldFBvb2xzIiwiZmxpY2tlcnMiLCJwYXRjaCIsInB1c2giLCJkZXNpcmVkSGVpZ2h0IiwiYXZhaWxhYmxlSGVpZ2h0IiwiZGVidWciLCJjaGFpbiIsImZpbmRPd25lckNoYWluQXRSb3ciLCJ0cmlnZ2VyWSIsInByZXZMaW5lIiwibmV4dExpbmUiLCJsZW5ndGgiLCJqb2luIiwibGV2ZWwiLCJ0T3B0aW1pemUiLCJvcHRpbWl6ZWQiLCJvcHRpbWl6ZU1zIiwiaGFzRGlmZiIsInVuc2hpZnQiLCJkZWNsIiwicmVjdCIsImdldCIsIm5vZGUiLCJ1bmRlZmluZWQiLCJyZWxhdGl2ZVgiLCJyZWxhdGl2ZVkiLCJwYXJrZWQiLCJ0YXJnZXRNb3ZlZCIsInBkeCIsInBkeSIsIk1hdGgiLCJtaW4iLCJtYXgiLCJjb2wiLCJmcm9tIiwiZHgiLCJkeSIsInJkeCIsInJkeSIsInRXcml0ZSIsIndyaXRlTXMiLCJzY3JvbGxEcmFpblBlbmRpbmciLCJ5b2dhTXMiLCJjb21taXRNcyIsInljIiwiZHVyYXRpb25NcyIsInBoYXNlcyIsInBhdGNoZXMiLCJ5b2dhIiwiY29tbWl0IiwieW9nYVZpc2l0ZWQiLCJ5b2dhTWVhc3VyZWQiLCJ5b2dhQ2FjaGVIaXRzIiwieW9nYUxpdmUiLCJmbHVzaFN5bmNGcm9tUmVjb25jaWxlciIsImZvcmNlUmVkcmF3IiwiaW52YWxpZGF0ZVByZXZGcmFtZSIsInNldEFsdFNjcmVlbkFjdGl2ZSIsImFjdGl2ZSIsIm1vdXNlVHJhY2tpbmciLCJpc0FsdFNjcmVlbkFjdGl2ZSIsInJlYXNzZXJ0VGVybWluYWxNb2RlcyIsImluY2x1ZGVBbHRTY3JlZW4iLCJkZXRhY2hGb3JTaHV0ZG93biIsImlzUmF3Iiwic2V0UmF3TW9kZSIsIm0iLCJkcmFpblN0ZGluIiwiYmxhbmsiLCJjb3B5U2VsZWN0aW9uTm9DbGVhciIsInRleHQiLCJ0aGVuIiwicmF3IiwiY29weVNlbGVjdGlvbiIsIm5vdGlmeVNlbGVjdGlvbkNoYW5nZSIsImNsZWFyVGV4dFNlbGVjdGlvbiIsInNldFNlYXJjaEhpZ2hsaWdodCIsInF1ZXJ5Iiwic2NhbkVsZW1lbnRTdWJ0cmVlIiwiZWwiLCJjZWlsIiwiZ2V0Q29tcHV0ZWRXaWR0aCIsImdldENvbXB1dGVkSGVpZ2h0IiwiZWxMZWZ0IiwiZ2V0Q29tcHV0ZWRMZWZ0IiwiZWxUb3AiLCJnZXRDb21wdXRlZFRvcCIsIm91dHB1dCIsIm9mZnNldFgiLCJvZmZzZXRZIiwicHJldlNjcmVlbiIsInJlbmRlcmVkIiwibWFya0RpcnR5Iiwic2xpY2UiLCJtYXAiLCJwIiwic2V0U2VhcmNoUG9zaXRpb25zIiwic3RhdGUiLCJzZXRTZWxlY3Rpb25CZ0NvbG9yIiwiY29sb3IiLCJ3cmFwcGVkIiwibnVsIiwiaW5kZXhPZiIsInNldFNlbGVjdGlvbkJnIiwiY29kZSIsImVuZENvZGUiLCJmaXJzdFJvdyIsImxhc3RSb3ciLCJzaWRlIiwic2hpZnRTZWxlY3Rpb25Gb3JTY3JvbGwiLCJkUm93IiwibWluUm93IiwibWF4Um93IiwiaGFkU2VsIiwibW92ZVNlbGVjdGlvbkZvY3VzIiwibW92ZSIsIm1heENvbCIsImhhc1RleHRTZWxlY3Rpb24iLCJzdWJzY3JpYmVUb1NlbGVjdGlvbkNoYW5nZSIsImFkZCIsImRlbGV0ZSIsImRpc3BhdGNoS2V5Ym9hcmRFdmVudCIsInBhcnNlZEtleSIsImFjdGl2ZUVsZW1lbnQiLCJkZWZhdWx0UHJldmVudGVkIiwibmFtZSIsImN0cmwiLCJtZXRhIiwic2hpZnQiLCJmb2N1c1ByZXZpb3VzIiwiZm9jdXNOZXh0IiwiZ2V0SHlwZXJsaW5rQXQiLCJjZWxsIiwidXJsIiwiaHlwZXJsaW5rIiwiU3BhY2VyVGFpbCIsIm9uSHlwZXJsaW5rQ2xpY2siLCJvcGVuSHlwZXJsaW5rIiwiaGFuZGxlTXVsdGlDbGljayIsImNvdW50IiwiaGFuZGxlU2VsZWN0aW9uRHJhZyIsInNlbCIsImFuY2hvclNwYW4iLCJzdGRpbkxpc3RlbmVycyIsIkFycmF5IiwibGlzdGVuZXIiLCJhcmdzIiwid2FzUmF3TW9kZSIsInJlYWRhYmxlTGlzdGVuZXJzIiwibGlzdGVuZXJzIiwiZm9yRWFjaCIsInJlbW92ZUxpc3RlbmVyIiwic3RkaW5XaXRoUmF3IiwibW9kZSIsImFkZExpc3RlbmVyIiwid3JpdGVSYXciLCJkYXRhIiwic2V0Q3Vyc29yRGVjbGFyYXRpb24iLCJjbGVhcklmTm9kZSIsInRyZWUiLCJ1cGRhdGVDb250YWluZXJTeW5jIiwiZmx1c2hTeW5jV29yayIsImVycm9yIiwicmVuZGVyUHJldmlvdXNPdXRwdXRfREVQUkVDQVRFRCIsImZyZWUiLCJyZXNvbHZlIiwicmVqZWN0IiwicmVzZXRMaW5lQ291bnQiLCJjb24iLCJjb25zb2xlIiwib3JpZ2luYWxzIiwiUGFydGlhbCIsIlJlY29yZCIsIkNvbnNvbGUiLCJ0b0RlYnVnIiwidG9FcnJvciIsIkNPTlNPTEVfU1RET1VUX01FVEhPRFMiLCJDT05TT0xFX1NUREVSUl9NRVRIT0RTIiwiYXNzZXJ0IiwiY29uZGl0aW9uIiwiYXNzaWduIiwib3JpZ2luYWxXcml0ZSIsInJlZW50ZXJlZCIsImludGVyY2VwdCIsImNodW5rIiwiVWludDhBcnJheSIsImVuY29kaW5nT3JDYiIsIkJ1ZmZlckVuY29kaW5nIiwiZXJyIiwiY2FsbGJhY2siLCJlbmNvZGluZyIsImNhbGwiLCJCdWZmZXIiLCJ0b1N0cmluZyIsInJlYWQiLCJwbGF0Zm9ybSIsInR0eSIsIndhc1JhdyIsImZkIiwiT19SRE9OTFkiLCJPX05PTkJMT0NLIiwiYnVmIiwiYWxsb2MiLCJpIl0sInNvdXJjZXMiOlsiaW5rLnRzeCJdLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgYXV0b0JpbmQgZnJvbSAnYXV0by1iaW5kJ1xuaW1wb3J0IHtcbiAgY2xvc2VTeW5jLFxuICBjb25zdGFudHMgYXMgZnNDb25zdGFudHMsXG4gIG9wZW5TeW5jLFxuICByZWFkU3luYyxcbiAgd3JpdGVTeW5jLFxufSBmcm9tICdmcydcbmltcG9ydCBub29wIGZyb20gJ2xvZGFzaC1lcy9ub29wLmpzJ1xuaW1wb3J0IHRocm90dGxlIGZyb20gJ2xvZGFzaC1lcy90aHJvdHRsZS5qcydcbmltcG9ydCBSZWFjdCwgeyB0eXBlIFJlYWN0Tm9kZSB9IGZyb20gJ3JlYWN0J1xuaW1wb3J0IHR5cGUgeyBGaWJlclJvb3QgfSBmcm9tICdyZWFjdC1yZWNvbmNpbGVyJ1xuaW1wb3J0IHsgQ29uY3VycmVudFJvb3QgfSBmcm9tICdyZWFjdC1yZWNvbmNpbGVyL2NvbnN0YW50cy5qcydcbmltcG9ydCB7IG9uRXhpdCB9IGZyb20gJ3NpZ25hbC1leGl0J1xuaW1wb3J0IHsgZmx1c2hJbnRlcmFjdGlvblRpbWUgfSBmcm9tICdzcmMvYm9vdHN0cmFwL3N0YXRlLmpzJ1xuaW1wb3J0IHsgZ2V0WW9nYUNvdW50ZXJzIH0gZnJvbSAnc3JjL25hdGl2ZS10cy95b2dhLWxheW91dC9pbmRleC5qcydcbmltcG9ydCB7IGxvZ0ZvckRlYnVnZ2luZyB9IGZyb20gJ3NyYy91dGlscy9kZWJ1Zy5qcydcbmltcG9ydCB7IGxvZ0Vycm9yIH0gZnJvbSAnc3JjL3V0aWxzL2xvZy5qcydcbmltcG9ydCB7IGZvcm1hdCB9IGZyb20gJ3V0aWwnXG5pbXBvcnQgeyBjb2xvcml6ZSB9IGZyb20gJy4vY29sb3JpemUuanMnXG5pbXBvcnQgQXBwIGZyb20gJy4vY29tcG9uZW50cy9BcHAuanMnXG5pbXBvcnQgdHlwZSB7XG4gIEN1cnNvckRlY2xhcmF0aW9uLFxuICBDdXJzb3JEZWNsYXJhdGlvblNldHRlcixcbn0gZnJvbSAnLi9jb21wb25lbnRzL0N1cnNvckRlY2xhcmF0aW9uQ29udGV4dC5qcydcbmltcG9ydCB7IEZSQU1FX0lOVEVSVkFMX01TIH0gZnJvbSAnLi9jb25zdGFudHMuanMnXG5pbXBvcnQgKiBhcyBkb20gZnJvbSAnLi9kb20uanMnXG5pbXBvcnQgeyBLZXlib2FyZEV2ZW50IH0gZnJvbSAnLi9ldmVudHMva2V5Ym9hcmQtZXZlbnQuanMnXG5pbXBvcnQgeyBGb2N1c01hbmFnZXIgfSBmcm9tICcuL2ZvY3VzLmpzJ1xuaW1wb3J0IHsgZW1wdHlGcmFtZSwgdHlwZSBGcmFtZSwgdHlwZSBGcmFtZUV2ZW50IH0gZnJvbSAnLi9mcmFtZS5qcydcbmltcG9ydCB7IGRpc3BhdGNoQ2xpY2ssIGRpc3BhdGNoSG92ZXIgfSBmcm9tICcuL2hpdC10ZXN0LmpzJ1xuaW1wb3J0IGluc3RhbmNlcyBmcm9tICcuL2luc3RhbmNlcy5qcydcbmltcG9ydCB7IExvZ1VwZGF0ZSB9IGZyb20gJy4vbG9nLXVwZGF0ZS5qcydcbmltcG9ydCB7IG5vZGVDYWNoZSB9IGZyb20gJy4vbm9kZS1jYWNoZS5qcydcbmltcG9ydCB7IG9wdGltaXplIH0gZnJvbSAnLi9vcHRpbWl6ZXIuanMnXG5pbXBvcnQgT3V0cHV0IGZyb20gJy4vb3V0cHV0LmpzJ1xuaW1wb3J0IHR5cGUgeyBQYXJzZWRLZXkgfSBmcm9tICcuL3BhcnNlLWtleXByZXNzLmpzJ1xuaW1wb3J0IHJlY29uY2lsZXIsIHtcbiAgZGlzcGF0Y2hlcixcbiAgZ2V0TGFzdENvbW1pdE1zLFxuICBnZXRMYXN0WW9nYU1zLFxuICBpc0RlYnVnUmVwYWludHNFbmFibGVkLFxuICByZWNvcmRZb2dhTXMsXG4gIHJlc2V0UHJvZmlsZUNvdW50ZXJzLFxufSBmcm9tICcuL3JlY29uY2lsZXIuanMnXG5pbXBvcnQgcmVuZGVyTm9kZVRvT3V0cHV0LCB7XG4gIGNvbnN1bWVGb2xsb3dTY3JvbGwsXG4gIGRpZExheW91dFNoaWZ0LFxufSBmcm9tICcuL3JlbmRlci1ub2RlLXRvLW91dHB1dC5qcydcbmltcG9ydCB7XG4gIGFwcGx5UG9zaXRpb25lZEhpZ2hsaWdodCxcbiAgdHlwZSBNYXRjaFBvc2l0aW9uLFxuICBzY2FuUG9zaXRpb25zLFxufSBmcm9tICcuL3JlbmRlci10by1zY3JlZW4uanMnXG5pbXBvcnQgY3JlYXRlUmVuZGVyZXIsIHsgdHlwZSBSZW5kZXJlciB9IGZyb20gJy4vcmVuZGVyZXIuanMnXG5pbXBvcnQge1xuICBDZWxsV2lkdGgsXG4gIENoYXJQb29sLFxuICBjZWxsQXQsXG4gIGNyZWF0ZVNjcmVlbixcbiAgSHlwZXJsaW5rUG9vbCxcbiAgaXNFbXB0eUNlbGxBdCxcbiAgbWlncmF0ZVNjcmVlblBvb2xzLFxuICBTdHlsZVBvb2wsXG59IGZyb20gJy4vc2NyZWVuLmpzJ1xuaW1wb3J0IHsgYXBwbHlTZWFyY2hIaWdobGlnaHQgfSBmcm9tICcuL3NlYXJjaEhpZ2hsaWdodC5qcydcbmltcG9ydCB7XG4gIGFwcGx5U2VsZWN0aW9uT3ZlcmxheSxcbiAgY2FwdHVyZVNjcm9sbGVkUm93cyxcbiAgY2xlYXJTZWxlY3Rpb24sXG4gIGNyZWF0ZVNlbGVjdGlvblN0YXRlLFxuICBleHRlbmRTZWxlY3Rpb24sXG4gIHR5cGUgRm9jdXNNb3ZlLFxuICBmaW5kUGxhaW5UZXh0VXJsQXQsXG4gIGdldFNlbGVjdGVkVGV4dCxcbiAgaGFzU2VsZWN0aW9uLFxuICBtb3ZlRm9jdXMsXG4gIHR5cGUgU2VsZWN0aW9uU3RhdGUsXG4gIHNlbGVjdExpbmVBdCxcbiAgc2VsZWN0V29yZEF0LFxuICBzaGlmdEFuY2hvcixcbiAgc2hpZnRTZWxlY3Rpb24sXG4gIHNoaWZ0U2VsZWN0aW9uRm9yRm9sbG93LFxuICBzdGFydFNlbGVjdGlvbixcbiAgdXBkYXRlU2VsZWN0aW9uLFxufSBmcm9tICcuL3NlbGVjdGlvbi5qcydcbmltcG9ydCB7XG4gIFNZTkNfT1VUUFVUX1NVUFBPUlRFRCxcbiAgc3VwcG9ydHNFeHRlbmRlZEtleXMsXG4gIHR5cGUgVGVybWluYWwsXG4gIHdyaXRlRGlmZlRvVGVybWluYWwsXG59IGZyb20gJy4vdGVybWluYWwuanMnXG5pbXBvcnQge1xuICBDVVJTT1JfSE9NRSxcbiAgY3Vyc29yTW92ZSxcbiAgY3Vyc29yUG9zaXRpb24sXG4gIERJU0FCTEVfS0lUVFlfS0VZQk9BUkQsXG4gIERJU0FCTEVfTU9ESUZZX09USEVSX0tFWVMsXG4gIEVOQUJMRV9LSVRUWV9LRVlCT0FSRCxcbiAgRU5BQkxFX01PRElGWV9PVEhFUl9LRVlTLFxuICBFUkFTRV9TQ1JFRU4sXG59IGZyb20gJy4vdGVybWlvL2NzaS5qcydcbmltcG9ydCB7XG4gIERCUCxcbiAgREZFLFxuICBESVNBQkxFX01PVVNFX1RSQUNLSU5HLFxuICBFTkFCTEVfTU9VU0VfVFJBQ0tJTkcsXG4gIEVOVEVSX0FMVF9TQ1JFRU4sXG4gIEVYSVRfQUxUX1NDUkVFTixcbiAgU0hPV19DVVJTT1IsXG59IGZyb20gJy4vdGVybWlvL2RlYy5qcydcbmltcG9ydCB7XG4gIENMRUFSX0lURVJNMl9QUk9HUkVTUyxcbiAgQ0xFQVJfVEFCX1NUQVRVUyxcbiAgc2V0Q2xpcGJvYXJkLFxuICBzdXBwb3J0c1RhYlN0YXR1cyxcbiAgd3JhcEZvck11bHRpcGxleGVyLFxufSBmcm9tICcuL3Rlcm1pby9vc2MuanMnXG5pbXBvcnQgeyBUZXJtaW5hbFdyaXRlUHJvdmlkZXIgfSBmcm9tICcuL3VzZVRlcm1pbmFsTm90aWZpY2F0aW9uLmpzJ1xuXG4vLyBBbHQtc2NyZWVuOiByZW5kZXJlci50cyBzZXRzIGN1cnNvci52aXNpYmxlID0gIWlzVFRZIHx8IHNjcmVlbi5oZWlnaHQ9PT0wLFxuLy8gd2hpY2ggaXMgYWx3YXlzIGZhbHNlIGluIGFsdC1zY3JlZW4gKFRUWSArIGNvbnRlbnQgZmlsbHMgc2NyZWVuKS5cbi8vIFJldXNpbmcgYSBmcm96ZW4gb2JqZWN0IHNhdmVzIDEgYWxsb2NhdGlvbiBwZXIgZnJhbWUuXG5jb25zdCBBTFRfU0NSRUVOX0FOQ0hPUl9DVVJTT1IgPSBPYmplY3QuZnJlZXplKHsgeDogMCwgeTogMCwgdmlzaWJsZTogZmFsc2UgfSlcbmNvbnN0IENVUlNPUl9IT01FX1BBVENIID0gT2JqZWN0LmZyZWV6ZSh7XG4gIHR5cGU6ICdzdGRvdXQnIGFzIGNvbnN0LFxuICBjb250ZW50OiBDVVJTT1JfSE9NRSxcbn0pXG5jb25zdCBFUkFTRV9USEVOX0hPTUVfUEFUQ0ggPSBPYmplY3QuZnJlZXplKHtcbiAgdHlwZTogJ3N0ZG91dCcgYXMgY29uc3QsXG4gIGNvbnRlbnQ6IEVSQVNFX1NDUkVFTiArIENVUlNPUl9IT01FLFxufSlcblxuLy8gQ2FjaGVkIHBlci1JbmstaW5zdGFuY2UsIGludmFsaWRhdGVkIG9uIHJlc2l6ZS4gZnJhbWUuY3Vyc29yLnkgZm9yXG4vLyBhbHQtc2NyZWVuIGlzIGFsd2F5cyB0ZXJtaW5hbFJvd3MgLSAxIChyZW5kZXJlci50cykuXG5mdW5jdGlvbiBtYWtlQWx0U2NyZWVuUGFya1BhdGNoKHRlcm1pbmFsUm93czogbnVtYmVyKSB7XG4gIHJldHVybiBPYmplY3QuZnJlZXplKHtcbiAgICB0eXBlOiAnc3Rkb3V0JyBhcyBjb25zdCxcbiAgICBjb250ZW50OiBjdXJzb3JQb3NpdGlvbih0ZXJtaW5hbFJvd3MsIDEpLFxuICB9KVxufVxuXG5leHBvcnQgdHlwZSBPcHRpb25zID0ge1xuICBzdGRvdXQ6IE5vZGVKUy5Xcml0ZVN0cmVhbVxuICBzdGRpbjogTm9kZUpTLlJlYWRTdHJlYW1cbiAgc3RkZXJyOiBOb2RlSlMuV3JpdGVTdHJlYW1cbiAgZXhpdE9uQ3RybEM6IGJvb2xlYW5cbiAgcGF0Y2hDb25zb2xlOiBib29sZWFuXG4gIHdhaXRVbnRpbEV4aXQ/OiAoKSA9PiBQcm9taXNlPHZvaWQ+XG4gIG9uRnJhbWU/OiAoZXZlbnQ6IEZyYW1lRXZlbnQpID0+IHZvaWRcbn1cblxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgSW5rIHtcbiAgcHJpdmF0ZSByZWFkb25seSBsb2c6IExvZ1VwZGF0ZVxuICBwcml2YXRlIHJlYWRvbmx5IHRlcm1pbmFsOiBUZXJtaW5hbFxuICBwcml2YXRlIHNjaGVkdWxlUmVuZGVyOiAoKCkgPT4gdm9pZCkgJiB7IGNhbmNlbD86ICgpID0+IHZvaWQgfVxuICAvLyBJZ25vcmUgbGFzdCByZW5kZXIgYWZ0ZXIgdW5tb3VudGluZyBhIHRyZWUgdG8gcHJldmVudCBlbXB0eSBvdXRwdXQgYmVmb3JlIGV4aXRcbiAgcHJpdmF0ZSBpc1VubW91bnRlZCA9IGZhbHNlXG4gIHByaXZhdGUgaXNQYXVzZWQgPSBmYWxzZVxuICBwcml2YXRlIHJlYWRvbmx5IGNvbnRhaW5lcjogRmliZXJSb290XG4gIHByaXZhdGUgcm9vdE5vZGU6IGRvbS5ET01FbGVtZW50XG4gIHJlYWRvbmx5IGZvY3VzTWFuYWdlcjogRm9jdXNNYW5hZ2VyXG4gIHByaXZhdGUgcmVuZGVyZXI6IFJlbmRlcmVyXG4gIHByaXZhdGUgcmVhZG9ubHkgc3R5bGVQb29sOiBTdHlsZVBvb2xcbiAgcHJpdmF0ZSBjaGFyUG9vbDogQ2hhclBvb2xcbiAgcHJpdmF0ZSBoeXBlcmxpbmtQb29sOiBIeXBlcmxpbmtQb29sXG4gIHByaXZhdGUgZXhpdFByb21pc2U/OiBQcm9taXNlPHZvaWQ+XG4gIHByaXZhdGUgcmVzdG9yZUNvbnNvbGU/OiAoKSA9PiB2b2lkXG4gIHByaXZhdGUgcmVzdG9yZVN0ZGVycj86ICgpID0+IHZvaWRcbiAgcHJpdmF0ZSByZWFkb25seSB1bnN1YnNjcmliZVRUWUhhbmRsZXJzPzogKCkgPT4gdm9pZFxuICBwcml2YXRlIHRlcm1pbmFsQ29sdW1uczogbnVtYmVyXG4gIHByaXZhdGUgdGVybWluYWxSb3dzOiBudW1iZXJcbiAgcHJpdmF0ZSBjdXJyZW50Tm9kZTogUmVhY3ROb2RlID0gbnVsbFxuICBwcml2YXRlIGZyb250RnJhbWU6IEZyYW1lXG4gIHByaXZhdGUgYmFja0ZyYW1lOiBGcmFtZVxuICBwcml2YXRlIGxhc3RQb29sUmVzZXRUaW1lID0gcGVyZm9ybWFuY2Uubm93KClcbiAgcHJpdmF0ZSBkcmFpblRpbWVyOiBSZXR1cm5UeXBlPHR5cGVvZiBzZXRUaW1lb3V0PiB8IG51bGwgPSBudWxsXG4gIHByaXZhdGUgbGFzdFlvZ2FDb3VudGVyczoge1xuICAgIG1zOiBudW1iZXJcbiAgICB2aXNpdGVkOiBudW1iZXJcbiAgICBtZWFzdXJlZDogbnVtYmVyXG4gICAgY2FjaGVIaXRzOiBudW1iZXJcbiAgICBsaXZlOiBudW1iZXJcbiAgfSA9IHsgbXM6IDAsIHZpc2l0ZWQ6IDAsIG1lYXN1cmVkOiAwLCBjYWNoZUhpdHM6IDAsIGxpdmU6IDAgfVxuICBwcml2YXRlIGFsdFNjcmVlblBhcmtQYXRjaDogUmVhZG9ubHk8eyB0eXBlOiAnc3Rkb3V0JzsgY29udGVudDogc3RyaW5nIH0+XG4gIC8vIFRleHQgc2VsZWN0aW9uIHN0YXRlIChhbHQtc2NyZWVuIG9ubHkpLiBPd25lZCBoZXJlIHNvIHRoZSBvdmVybGF5XG4gIC8vIHBhc3MgaW4gb25SZW5kZXIgY2FuIHJlYWQgaXQgYW5kIEFwcC50c3ggY2FuIHVwZGF0ZSBpdCBmcm9tIG1vdXNlXG4gIC8vIGV2ZW50cy4gUHVibGljIHNvIGluc3RhbmNlcy5nZXQoKSBjYWxsZXJzIGNhbiBhY2Nlc3MuXG4gIHJlYWRvbmx5IHNlbGVjdGlvbjogU2VsZWN0aW9uU3RhdGUgPSBjcmVhdGVTZWxlY3Rpb25TdGF0ZSgpXG4gIC8vIFNlYXJjaCBoaWdobGlnaHQgcXVlcnkgKGFsdC1zY3JlZW4gb25seSkuIFNldHRlciBiZWxvdyB0cmlnZ2Vyc1xuICAvLyBzY2hlZHVsZVJlbmRlcjsgYXBwbHlTZWFyY2hIaWdobGlnaHQgaW4gb25SZW5kZXIgaW52ZXJ0cyBtYXRjaGluZyBjZWxscy5cbiAgcHJpdmF0ZSBzZWFyY2hIaWdobGlnaHRRdWVyeSA9ICcnXG4gIC8vIFBvc2l0aW9uLWJhc2VkIGhpZ2hsaWdodC4gVk1MIHNjYW5zIHBvc2l0aW9ucyBPTkNFICh2aWFcbiAgLy8gc2NhbkVsZW1lbnRTdWJ0cmVlLCB3aGVuIHRoZSB0YXJnZXQgbWVzc2FnZSBpcyBtb3VudGVkKSwgc3RvcmVzIHRoZW1cbiAgLy8gbWVzc2FnZS1yZWxhdGl2ZSwgc2V0cyB0aGlzIGZvciBldmVyeS1mcmFtZSBhcHBseS4gcm93T2Zmc2V0ID1cbiAgLy8gbWVzc2FnZSdzIGN1cnJlbnQgc2NyZWVuLXRvcC4gY3VycmVudElkeCA9IHdoaWNoIHBvc2l0aW9uIGlzXG4gIC8vIFwiY3VycmVudFwiICh5ZWxsb3cpLiBudWxsIGNsZWFycy4gUG9zaXRpb25zIGFyZSBrbm93biB1cGZyb250IOKAlFxuICAvLyBuYXZpZ2F0aW9uIGlzIGluZGV4IGFyaXRobWV0aWMsIG5vIHNjYW4tZmVlZGJhY2sgbG9vcC5cbiAgcHJpdmF0ZSBzZWFyY2hQb3NpdGlvbnM6IHtcbiAgICBwb3NpdGlvbnM6IE1hdGNoUG9zaXRpb25bXVxuICAgIHJvd09mZnNldDogbnVtYmVyXG4gICAgY3VycmVudElkeDogbnVtYmVyXG4gIH0gfCBudWxsID0gbnVsbFxuICAvLyBSZWFjdC1sYW5kIHN1YnNjcmliZXJzIGZvciBzZWxlY3Rpb24gc3RhdGUgY2hhbmdlcyAodXNlSGFzU2VsZWN0aW9uKS5cbiAgLy8gRmlyZWQgYWxvbmdzaWRlIHRoZSB0ZXJtaW5hbCByZXBhaW50IHdoZW5ldmVyIHRoZSBzZWxlY3Rpb24gbXV0YXRlc1xuICAvLyBzbyBVSSAoZS5nLiBmb290ZXIgaGludHMpIGNhbiByZWFjdCB0byBzZWxlY3Rpb24gYXBwZWFyaW5nL2NsZWFyaW5nLlxuICBwcml2YXRlIHJlYWRvbmx5IHNlbGVjdGlvbkxpc3RlbmVycyA9IG5ldyBTZXQ8KCkgPT4gdm9pZD4oKVxuICAvLyBET00gbm9kZXMgY3VycmVudGx5IHVuZGVyIHRoZSBwb2ludGVyIChtb2RlLTEwMDMgbW90aW9uKS4gSGVsZCBoZXJlXG4gIC8vIHNvIEFwcC50c3gncyBoYW5kbGVNb3VzZUV2ZW50IGlzIHN0YXRlbGVzcyDigJQgZGlzcGF0Y2hIb3ZlciBkaWZmc1xuICAvLyBhZ2FpbnN0IHRoaXMgc2V0IGFuZCBtdXRhdGVzIGl0IGluIHBsYWNlLlxuICBwcml2YXRlIHJlYWRvbmx5IGhvdmVyZWROb2RlcyA9IG5ldyBTZXQ8ZG9tLkRPTUVsZW1lbnQ+KClcbiAgLy8gU2V0IGJ5IDxBbHRlcm5hdGVTY3JlZW4+IHZpYSBzZXRBbHRTY3JlZW5BY3RpdmUoKS4gQ29udHJvbHMgdGhlXG4gIC8vIHJlbmRlcmVyJ3MgY3Vyc29yLnkgY2xhbXBpbmcgKGtlZXBzIGN1cnNvciBpbi12aWV3cG9ydCB0byBhdm9pZFxuICAvLyBMRi1pbmR1Y2VkIHNjcm9sbCB3aGVuIHNjcmVlbi5oZWlnaHQgPT09IHRlcm1pbmFsUm93cykgYW5kIGdhdGVzXG4gIC8vIGFsdC1zY3JlZW4tYXdhcmUgU0lHQ09OVC9yZXNpemUvdW5tb3VudCBoYW5kbGluZy5cbiAgcHJpdmF0ZSBhbHRTY3JlZW5BY3RpdmUgPSBmYWxzZVxuICAvLyBTZXQgYWxvbmdzaWRlIGFsdFNjcmVlbkFjdGl2ZSBzbyBTSUdDT05UIHJlc3VtZSBrbm93cyB3aGV0aGVyIHRvXG4gIC8vIHJlLWVuYWJsZSBtb3VzZSB0cmFja2luZyAobm90IGFsbCA8QWx0ZXJuYXRlU2NyZWVuPiB1c2VzIHdhbnQgaXQpLlxuICBwcml2YXRlIGFsdFNjcmVlbk1vdXNlVHJhY2tpbmcgPSBmYWxzZVxuICAvLyBUcnVlIHdoZW4gdGhlIHByZXZpb3VzIGZyYW1lJ3Mgc2NyZWVuIGJ1ZmZlciBjYW5ub3QgYmUgdHJ1c3RlZCBmb3JcbiAgLy8gYmxpdCDigJQgc2VsZWN0aW9uIG92ZXJsYXkgbXV0YXRlZCBpdCwgcmVzZXRGcmFtZXNGb3JBbHRTY3JlZW4oKVxuICAvLyByZXBsYWNlZCBpdCB3aXRoIGJsYW5rcywgb3IgZm9yY2VSZWRyYXcoKSByZXNldCBpdCB0byAww5cwLiBGb3JjZXNcbiAgLy8gb25lIGZ1bGwtcmVuZGVyIGZyYW1lOyBzdGVhZHktc3RhdGUgZnJhbWVzIGFmdGVyIGNsZWFyIGl0IGFuZCByZWdhaW5cbiAgLy8gdGhlIGJsaXQgKyBuYXJyb3ctZGFtYWdlIGZhc3QgcGF0aC5cbiAgcHJpdmF0ZSBwcmV2RnJhbWVDb250YW1pbmF0ZWQgPSBmYWxzZVxuICAvLyBTZXQgYnkgaGFuZGxlUmVzaXplOiBwcmVwZW5kIEVSQVNFX1NDUkVFTiB0byB0aGUgbmV4dCBvblJlbmRlcidzIHBhdGNoZXNcbiAgLy8gSU5TSURFIHRoZSBCU1UvRVNVIGJsb2NrIHNvIGNsZWFyK3BhaW50IGlzIGF0b21pYy4gV3JpdGluZyBFUkFTRV9TQ1JFRU5cbiAgLy8gc3luY2hyb25vdXNseSBpbiBoYW5kbGVSZXNpemUgd291bGQgbGVhdmUgdGhlIHNjcmVlbiBibGFuayBmb3IgdGhlIH44MG1zXG4gIC8vIHJlbmRlcigpIHRha2VzOyBkZWZlcnJpbmcgaW50byB0aGUgYXRvbWljIGJsb2NrIG1lYW5zIG9sZCBjb250ZW50IHN0YXlzXG4gIC8vIHZpc2libGUgdW50aWwgdGhlIG5ldyBmcmFtZSBpcyBmdWxseSByZWFkeS5cbiAgcHJpdmF0ZSBuZWVkc0VyYXNlQmVmb3JlUGFpbnQgPSBmYWxzZVxuICAvLyBOYXRpdmUgY3Vyc29yIHBvc2l0aW9uaW5nOiBhIGNvbXBvbmVudCAodmlhIHVzZURlY2xhcmVkQ3Vyc29yKSBkZWNsYXJlc1xuICAvLyB3aGVyZSB0aGUgdGVybWluYWwgY3Vyc29yIHNob3VsZCBiZSBwYXJrZWQgYWZ0ZXIgZWFjaCBmcmFtZS4gVGVybWluYWxcbiAgLy8gZW11bGF0b3JzIHJlbmRlciBJTUUgcHJlZWRpdCB0ZXh0IGF0IHRoZSBwaHlzaWNhbCBjdXJzb3IgcG9zaXRpb24sIGFuZFxuICAvLyBzY3JlZW4gcmVhZGVycyAvIHNjcmVlbiBtYWduaWZpZXJzIHRyYWNrIGl0IOKAlCBzbyBwYXJraW5nIGF0IHRoZSB0ZXh0XG4gIC8vIGlucHV0J3MgY2FyZXQgbWFrZXMgQ0pLIGlucHV0IGFwcGVhciBpbmxpbmUgYW5kIGxldHMgYTExeSB0b29scyBmb2xsb3cuXG4gIHByaXZhdGUgY3Vyc29yRGVjbGFyYXRpb246IEN1cnNvckRlY2xhcmF0aW9uIHwgbnVsbCA9IG51bGxcbiAgLy8gTWFpbi1zY3JlZW46IHBoeXNpY2FsIGN1cnNvciBwb3NpdGlvbiBhZnRlciB0aGUgZGVjbGFyZWQtY3Vyc29yIG1vdmUsXG4gIC8vIHRyYWNrZWQgc2VwYXJhdGVseSBmcm9tIGZyYW1lLmN1cnNvciAod2hpY2ggbXVzdCBzdGF5IGF0IGNvbnRlbnQtYm90dG9tXG4gIC8vIGZvciBsb2ctdXBkYXRlJ3MgcmVsYXRpdmUtbW92ZSBpbnZhcmlhbnRzKS4gQWx0LXNjcmVlbiBkb2Vzbid0IG5lZWRcbiAgLy8gdGhpcyDigJQgZXZlcnkgZnJhbWUgYmVnaW5zIHdpdGggQ1NJIEguIG51bGwgPSBubyBtb3ZlIGVtaXR0ZWQgbGFzdCBmcmFtZS5cbiAgcHJpdmF0ZSBkaXNwbGF5Q3Vyc29yOiB7IHg6IG51bWJlcjsgeTogbnVtYmVyIH0gfCBudWxsID0gbnVsbFxuXG4gIGNvbnN0cnVjdG9yKHByaXZhdGUgcmVhZG9ubHkgb3B0aW9uczogT3B0aW9ucykge1xuICAgIGF1dG9CaW5kKHRoaXMpXG5cbiAgICBpZiAodGhpcy5vcHRpb25zLnBhdGNoQ29uc29sZSkge1xuICAgICAgdGhpcy5yZXN0b3JlQ29uc29sZSA9IHRoaXMucGF0Y2hDb25zb2xlKClcbiAgICAgIHRoaXMucmVzdG9yZVN0ZGVyciA9IHRoaXMucGF0Y2hTdGRlcnIoKVxuICAgIH1cblxuICAgIHRoaXMudGVybWluYWwgPSB7XG4gICAgICBzdGRvdXQ6IG9wdGlvbnMuc3Rkb3V0LFxuICAgICAgc3RkZXJyOiBvcHRpb25zLnN0ZGVycixcbiAgICB9XG5cbiAgICB0aGlzLnRlcm1pbmFsQ29sdW1ucyA9IG9wdGlvbnMuc3Rkb3V0LmNvbHVtbnMgfHwgODBcbiAgICB0aGlzLnRlcm1pbmFsUm93cyA9IG9wdGlvbnMuc3Rkb3V0LnJvd3MgfHwgMjRcbiAgICB0aGlzLmFsdFNjcmVlblBhcmtQYXRjaCA9IG1ha2VBbHRTY3JlZW5QYXJrUGF0Y2godGhpcy50ZXJtaW5hbFJvd3MpXG4gICAgdGhpcy5zdHlsZVBvb2wgPSBuZXcgU3R5bGVQb29sKClcbiAgICB0aGlzLmNoYXJQb29sID0gbmV3IENoYXJQb29sKClcbiAgICB0aGlzLmh5cGVybGlua1Bvb2wgPSBuZXcgSHlwZXJsaW5rUG9vbCgpXG4gICAgdGhpcy5mcm9udEZyYW1lID0gZW1wdHlGcmFtZShcbiAgICAgIHRoaXMudGVybWluYWxSb3dzLFxuICAgICAgdGhpcy50ZXJtaW5hbENvbHVtbnMsXG4gICAgICB0aGlzLnN0eWxlUG9vbCxcbiAgICAgIHRoaXMuY2hhclBvb2wsXG4gICAgICB0aGlzLmh5cGVybGlua1Bvb2wsXG4gICAgKVxuICAgIHRoaXMuYmFja0ZyYW1lID0gZW1wdHlGcmFtZShcbiAgICAgIHRoaXMudGVybWluYWxSb3dzLFxuICAgICAgdGhpcy50ZXJtaW5hbENvbHVtbnMsXG4gICAgICB0aGlzLnN0eWxlUG9vbCxcbiAgICAgIHRoaXMuY2hhclBvb2wsXG4gICAgICB0aGlzLmh5cGVybGlua1Bvb2wsXG4gICAgKVxuXG4gICAgdGhpcy5sb2cgPSBuZXcgTG9nVXBkYXRlKHtcbiAgICAgIGlzVFRZOiAob3B0aW9ucy5zdGRvdXQuaXNUVFkgYXMgYm9vbGVhbiB8IHVuZGVmaW5lZCkgfHwgZmFsc2UsXG4gICAgICBzdHlsZVBvb2w6IHRoaXMuc3R5bGVQb29sLFxuICAgIH0pXG5cbiAgICAvLyBzY2hlZHVsZVJlbmRlciBpcyBjYWxsZWQgZnJvbSB0aGUgcmVjb25jaWxlcidzIHJlc2V0QWZ0ZXJDb21taXQsIHdoaWNoXG4gICAgLy8gcnVucyBCRUZPUkUgUmVhY3QncyBsYXlvdXQgcGhhc2UgKHJlZiBhdHRhY2ggKyB1c2VMYXlvdXRFZmZlY3QpLiBBbnlcbiAgICAvLyBzdGF0ZSBzZXQgaW4gbGF5b3V0IGVmZmVjdHMg4oCUIG5vdGFibHkgdGhlIGN1cnNvckRlY2xhcmF0aW9uIGZyb21cbiAgICAvLyB1c2VEZWNsYXJlZEN1cnNvciDigJQgd291bGQgbGFnIG9uZSBjb21taXQgYmVoaW5kIGlmIHdlIHJlbmRlcmVkXG4gICAgLy8gc3luY2hyb25vdXNseS4gRGVmZXJyaW5nIHRvIGEgbWljcm90YXNrIHJ1bnMgb25SZW5kZXIgYWZ0ZXIgbGF5b3V0XG4gICAgLy8gZWZmZWN0cyBoYXZlIGNvbW1pdHRlZCwgc28gdGhlIG5hdGl2ZSBjdXJzb3IgdHJhY2tzIHRoZSBjYXJldCB3aXRob3V0XG4gICAgLy8gYSBvbmUta2V5c3Ryb2tlIGxhZy4gU2FtZSBldmVudC1sb29wIHRpY2ssIHNvIHRocm91Z2hwdXQgaXMgdW5jaGFuZ2VkLlxuICAgIC8vIFRlc3QgZW52IHVzZXMgb25JbW1lZGlhdGVSZW5kZXIgKGRpcmVjdCBvblJlbmRlciwgbm8gdGhyb3R0bGUpIHNvXG4gICAgLy8gZXhpc3Rpbmcgc3luY2hyb25vdXMgbGFzdEZyYW1lKCkgdGVzdHMgYXJlIHVuYWZmZWN0ZWQuXG4gICAgY29uc3QgZGVmZXJyZWRSZW5kZXIgPSAoKTogdm9pZCA9PiBxdWV1ZU1pY3JvdGFzayh0aGlzLm9uUmVuZGVyKVxuICAgIHRoaXMuc2NoZWR1bGVSZW5kZXIgPSB0aHJvdHRsZShkZWZlcnJlZFJlbmRlciwgRlJBTUVfSU5URVJWQUxfTVMsIHtcbiAgICAgIGxlYWRpbmc6IHRydWUsXG4gICAgICB0cmFpbGluZzogdHJ1ZSxcbiAgICB9KVxuXG4gICAgLy8gSWdub3JlIGxhc3QgcmVuZGVyIGFmdGVyIHVubW91bnRpbmcgYSB0cmVlIHRvIHByZXZlbnQgZW1wdHkgb3V0cHV0IGJlZm9yZSBleGl0XG4gICAgdGhpcy5pc1VubW91bnRlZCA9IGZhbHNlXG5cbiAgICAvLyBVbm1vdW50IHdoZW4gcHJvY2VzcyBleGl0c1xuICAgIHRoaXMudW5zdWJzY3JpYmVFeGl0ID0gb25FeGl0KHRoaXMudW5tb3VudCwgeyBhbHdheXNMYXN0OiBmYWxzZSB9KVxuXG4gICAgaWYgKG9wdGlvbnMuc3Rkb3V0LmlzVFRZKSB7XG4gICAgICBvcHRpb25zLnN0ZG91dC5vbigncmVzaXplJywgdGhpcy5oYW5kbGVSZXNpemUpXG4gICAgICBwcm9jZXNzLm9uKCdTSUdDT05UJywgdGhpcy5oYW5kbGVSZXN1bWUpXG5cbiAgICAgIHRoaXMudW5zdWJzY3JpYmVUVFlIYW5kbGVycyA9ICgpID0+IHtcbiAgICAgICAgb3B0aW9ucy5zdGRvdXQub2ZmKCdyZXNpemUnLCB0aGlzLmhhbmRsZVJlc2l6ZSlcbiAgICAgICAgcHJvY2Vzcy5vZmYoJ1NJR0NPTlQnLCB0aGlzLmhhbmRsZVJlc3VtZSlcbiAgICAgIH1cbiAgICB9XG5cbiAgICB0aGlzLnJvb3ROb2RlID0gZG9tLmNyZWF0ZU5vZGUoJ2luay1yb290JylcbiAgICB0aGlzLmZvY3VzTWFuYWdlciA9IG5ldyBGb2N1c01hbmFnZXIoKHRhcmdldCwgZXZlbnQpID0+XG4gICAgICBkaXNwYXRjaGVyLmRpc3BhdGNoRGlzY3JldGUodGFyZ2V0LCBldmVudCksXG4gICAgKVxuICAgIHRoaXMucm9vdE5vZGUuZm9jdXNNYW5hZ2VyID0gdGhpcy5mb2N1c01hbmFnZXJcbiAgICB0aGlzLnJlbmRlcmVyID0gY3JlYXRlUmVuZGVyZXIodGhpcy5yb290Tm9kZSwgdGhpcy5zdHlsZVBvb2wpXG4gICAgdGhpcy5yb290Tm9kZS5vblJlbmRlciA9IHRoaXMuc2NoZWR1bGVSZW5kZXJcbiAgICB0aGlzLnJvb3ROb2RlLm9uSW1tZWRpYXRlUmVuZGVyID0gdGhpcy5vblJlbmRlclxuICAgIHRoaXMucm9vdE5vZGUub25Db21wdXRlTGF5b3V0ID0gKCkgPT4ge1xuICAgICAgLy8gQ2FsY3VsYXRlIGxheW91dCBkdXJpbmcgUmVhY3QncyBjb21taXQgcGhhc2Ugc28gdXNlTGF5b3V0RWZmZWN0IGhvb2tzXG4gICAgICAvLyBoYXZlIGFjY2VzcyB0byBmcmVzaCBsYXlvdXQgZGF0YVxuICAgICAgLy8gR3VhcmQgYWdhaW5zdCBhY2Nlc3NpbmcgZnJlZWQgWW9nYSBub2RlcyBhZnRlciB1bm1vdW50XG4gICAgICBpZiAodGhpcy5pc1VubW91bnRlZCkge1xuICAgICAgICByZXR1cm5cbiAgICAgIH1cblxuICAgICAgaWYgKHRoaXMucm9vdE5vZGUueW9nYU5vZGUpIHtcbiAgICAgICAgY29uc3QgdDAgPSBwZXJmb3JtYW5jZS5ub3coKVxuICAgICAgICB0aGlzLnJvb3ROb2RlLnlvZ2FOb2RlLnNldFdpZHRoKHRoaXMudGVybWluYWxDb2x1bW5zKVxuICAgICAgICB0aGlzLnJvb3ROb2RlLnlvZ2FOb2RlLmNhbGN1bGF0ZUxheW91dCh0aGlzLnRlcm1pbmFsQ29sdW1ucylcbiAgICAgICAgY29uc3QgbXMgPSBwZXJmb3JtYW5jZS5ub3coKSAtIHQwXG4gICAgICAgIHJlY29yZFlvZ2FNcyhtcylcbiAgICAgICAgY29uc3QgYyA9IGdldFlvZ2FDb3VudGVycygpXG4gICAgICAgIHRoaXMubGFzdFlvZ2FDb3VudGVycyA9IHsgbXMsIC4uLmMgfVxuICAgICAgfVxuICAgIH1cblxuICAgIC8vIEB0cy1leHBlY3QtZXJyb3IgQHR5cGVzL3JlYWN0LXJlY29uY2lsZXJAMC4zMi4zIGRlY2xhcmVzIDExIGFyZ3Mgd2l0aCB0cmFuc2l0aW9uQ2FsbGJhY2tzLFxuICAgIC8vIGJ1dCByZWFjdC1yZWNvbmNpbGVyIDAuMzMuMCBzb3VyY2Ugb25seSBhY2NlcHRzIDEwIGFyZ3MgKG5vIHRyYW5zaXRpb25DYWxsYmFja3MpXG4gICAgdGhpcy5jb250YWluZXIgPSByZWNvbmNpbGVyLmNyZWF0ZUNvbnRhaW5lcihcbiAgICAgIHRoaXMucm9vdE5vZGUsXG4gICAgICBDb25jdXJyZW50Um9vdCxcbiAgICAgIG51bGwsXG4gICAgICBmYWxzZSxcbiAgICAgIG51bGwsXG4gICAgICAnaWQnLFxuICAgICAgbm9vcCwgLy8gb25VbmNhdWdodEVycm9yXG4gICAgICBub29wLCAvLyBvbkNhdWdodEVycm9yXG4gICAgICBub29wLCAvLyBvblJlY292ZXJhYmxlRXJyb3JcbiAgICAgIG5vb3AsIC8vIG9uRGVmYXVsdFRyYW5zaXRpb25JbmRpY2F0b3JcbiAgICApXG5cbiAgICBpZiAoXCJwcm9kdWN0aW9uXCIgPT09ICdkZXZlbG9wbWVudCcpIHtcbiAgICAgIHJlY29uY2lsZXIuaW5qZWN0SW50b0RldlRvb2xzKHtcbiAgICAgICAgYnVuZGxlVHlwZTogMCxcbiAgICAgICAgLy8gUmVwb3J0aW5nIFJlYWN0IERPTSdzIHZlcnNpb24sIG5vdCBJbmsnc1xuICAgICAgICAvLyBTZWUgaHR0cHM6Ly9naXRodWIuY29tL2ZhY2Vib29rL3JlYWN0L2lzc3Vlcy8xNjY2NiNpc3N1ZWNvbW1lbnQtNTMyNjM5OTA1XG4gICAgICAgIHZlcnNpb246ICcxNi4xMy4xJyxcbiAgICAgICAgcmVuZGVyZXJQYWNrYWdlTmFtZTogJ2luaycsXG4gICAgICB9KVxuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgaGFuZGxlUmVzdW1lID0gKCkgPT4ge1xuICAgIGlmICghdGhpcy5vcHRpb25zLnN0ZG91dC5pc1RUWSkge1xuICAgICAgcmV0dXJuXG4gICAgfVxuXG4gICAgLy8gQWx0IHNjcmVlbjogYWZ0ZXIgU0lHQ09OVCwgY29udGVudCBpcyBzdGFsZSAoc2hlbGwgbWF5IGhhdmUgd3JpdHRlblxuICAgIC8vIHRvIG1haW4gc2NyZWVuLCBzd2l0Y2hpbmcgZm9jdXMgYXdheSkgYW5kIG1vdXNlIHRyYWNraW5nIHdhc1xuICAgIC8vIGRpc2FibGVkIGJ5IGhhbmRsZVN1c3BlbmQuXG4gICAgaWYgKHRoaXMuYWx0U2NyZWVuQWN0aXZlKSB7XG4gICAgICB0aGlzLnJlZW50ZXJBbHRTY3JlZW4oKVxuICAgICAgcmV0dXJuXG4gICAgfVxuXG4gICAgLy8gTWFpbiBzY3JlZW46IHN0YXJ0IGZyZXNoIHRvIHByZXZlbnQgY2xvYmJlcmluZyB0ZXJtaW5hbCBjb250ZW50XG4gICAgdGhpcy5mcm9udEZyYW1lID0gZW1wdHlGcmFtZShcbiAgICAgIHRoaXMuZnJvbnRGcmFtZS52aWV3cG9ydC5oZWlnaHQsXG4gICAgICB0aGlzLmZyb250RnJhbWUudmlld3BvcnQud2lkdGgsXG4gICAgICB0aGlzLnN0eWxlUG9vbCxcbiAgICAgIHRoaXMuY2hhclBvb2wsXG4gICAgICB0aGlzLmh5cGVybGlua1Bvb2wsXG4gICAgKVxuICAgIHRoaXMuYmFja0ZyYW1lID0gZW1wdHlGcmFtZShcbiAgICAgIHRoaXMuYmFja0ZyYW1lLnZpZXdwb3J0LmhlaWdodCxcbiAgICAgIHRoaXMuYmFja0ZyYW1lLnZpZXdwb3J0LndpZHRoLFxuICAgICAgdGhpcy5zdHlsZVBvb2wsXG4gICAgICB0aGlzLmNoYXJQb29sLFxuICAgICAgdGhpcy5oeXBlcmxpbmtQb29sLFxuICAgIClcbiAgICB0aGlzLmxvZy5yZXNldCgpXG4gICAgLy8gUGh5c2ljYWwgY3Vyc29yIHBvc2l0aW9uIGlzIHVua25vd24gYWZ0ZXIgdGhlIHNoZWxsIHRvb2sgb3ZlciBkdXJpbmdcbiAgICAvLyBzdXNwZW5kLiBDbGVhciBkaXNwbGF5Q3Vyc29yIHNvIHRoZSBuZXh0IGZyYW1lJ3MgY3Vyc29yIHByZWFtYmxlXG4gICAgLy8gZG9lc24ndCBlbWl0IGEgcmVsYXRpdmUgbW92ZSBmcm9tIGEgc3RhbGUgcGFyayBwb3NpdGlvbi5cbiAgICB0aGlzLmRpc3BsYXlDdXJzb3IgPSBudWxsXG4gIH1cblxuICAvLyBOT1QgZGVib3VuY2VkLiBBIGRlYm91bmNlIG9wZW5zIGEgd2luZG93IHdoZXJlIHN0ZG91dC5jb2x1bW5zIGlzIE5FV1xuICAvLyBidXQgdGhpcy50ZXJtaW5hbENvbHVtbnMvWW9nYSBhcmUgT0xEIOKAlCBhbnkgc2NoZWR1bGVSZW5kZXIgZHVyaW5nIHRoYXRcbiAgLy8gd2luZG93IChzcGlubmVyLCBjbG9jaykgbWFrZXMgbG9nLXVwZGF0ZSBkZXRlY3QgYSB3aWR0aCBjaGFuZ2UgYW5kXG4gIC8vIGNsZWFyIHRoZSBzY3JlZW4sIHRoZW4gdGhlIGRlYm91bmNlIGZpcmVzIGFuZCBjbGVhcnMgYWdhaW4gKGRvdWJsZVxuICAvLyBibGFua+KGknBhaW50IGZsaWNrZXIpLiB1c2VWaXJ0dWFsU2Nyb2xsJ3MgaGVpZ2h0IHNjYWxpbmcgYWxyZWFkeSBib3VuZHNcbiAgLy8gdGhlIHBlci1yZXNpemUgY29zdDsgc3luY2hyb25vdXMgaGFuZGxpbmcga2VlcHMgZGltZW5zaW9ucyBjb25zaXN0ZW50LlxuICBwcml2YXRlIGhhbmRsZVJlc2l6ZSA9ICgpID0+IHtcbiAgICBjb25zdCBjb2xzID0gdGhpcy5vcHRpb25zLnN0ZG91dC5jb2x1bW5zIHx8IDgwXG4gICAgY29uc3Qgcm93cyA9IHRoaXMub3B0aW9ucy5zdGRvdXQucm93cyB8fCAyNFxuICAgIC8vIFRlcm1pbmFscyBvZnRlbiBlbWl0IDIrIHJlc2l6ZSBldmVudHMgZm9yIG9uZSB1c2VyIGFjdGlvbiAod2luZG93XG4gICAgLy8gc2V0dGxpbmcpLiBTYW1lLWRpbWVuc2lvbiBldmVudHMgYXJlIG5vLW9wczsgc2tpcCB0byBhdm9pZCByZWR1bmRhbnRcbiAgICAvLyBmcmFtZSByZXNldHMgYW5kIHJlbmRlcnMuXG4gICAgaWYgKGNvbHMgPT09IHRoaXMudGVybWluYWxDb2x1bW5zICYmIHJvd3MgPT09IHRoaXMudGVybWluYWxSb3dzKSByZXR1cm5cbiAgICB0aGlzLnRlcm1pbmFsQ29sdW1ucyA9IGNvbHNcbiAgICB0aGlzLnRlcm1pbmFsUm93cyA9IHJvd3NcbiAgICB0aGlzLmFsdFNjcmVlblBhcmtQYXRjaCA9IG1ha2VBbHRTY3JlZW5QYXJrUGF0Y2godGhpcy50ZXJtaW5hbFJvd3MpXG5cbiAgICAvLyBBbHQgc2NyZWVuOiByZXNldCBmcmFtZSBidWZmZXJzIHNvIHRoZSBuZXh0IHJlbmRlciByZXBhaW50cyBmcm9tXG4gICAgLy8gc2NyYXRjaCAocHJldkZyYW1lQ29udGFtaW5hdGVkIOKGkiBldmVyeSBjZWxsIHdyaXR0ZW4sIHdyYXBwZWQgaW5cbiAgICAvLyBCU1UvRVNVIOKAlCBvbGQgY29udGVudCBzdGF5cyB2aXNpYmxlIHVudGlsIHRoZSBuZXcgZnJhbWUgc3dhcHNcbiAgICAvLyBhdG9taWNhbGx5KS4gUmUtYXNzZXJ0IG1vdXNlIHRyYWNraW5nIChzb21lIGVtdWxhdG9ycyByZXNldCBpdCBvblxuICAgIC8vIHJlc2l6ZSkuIERvIE5PVCB3cml0ZSBFTlRFUl9BTFRfU0NSRUVOOiBpVGVybTIgdHJlYXRzID8xMDQ5aCBhcyBhXG4gICAgLy8gYnVmZmVyIGNsZWFyIGV2ZW4gd2hlbiBhbHJlYWR5IGluIGFsdCDigJQgdGhhdCdzIHRoZSBibGFuayBmbGlja2VyLlxuICAgIC8vIFNlbGYtaGVhbGluZyByZS1lbnRyeSAoaWYgc29tZXRoaW5nIGtpY2tlZCB1cyBvdXQgb2YgYWx0KSBpcyBoYW5kbGVkXG4gICAgLy8gYnkgaGFuZGxlUmVzdW1lIChTSUdDT05UKSBhbmQgdGhlIHNsZWVwLXdha2UgZGV0ZWN0b3I7IHJlc2l6ZSBpdHNlbGZcbiAgICAvLyBkb2Vzbid0IGV4aXQgYWx0LXNjcmVlbi4gRG8gTk9UIHdyaXRlIEVSQVNFX1NDUkVFTjogcmVuZGVyKCkgYmVsb3dcbiAgICAvLyBjYW4gdGFrZSB+ODBtczsgZXJhc2luZyBmaXJzdCBsZWF2ZXMgdGhlIHNjcmVlbiBibGFuayB0aGF0IHdob2xlIHRpbWUuXG4gICAgaWYgKHRoaXMuYWx0U2NyZWVuQWN0aXZlICYmICF0aGlzLmlzUGF1c2VkICYmIHRoaXMub3B0aW9ucy5zdGRvdXQuaXNUVFkpIHtcbiAgICAgIGlmICh0aGlzLmFsdFNjcmVlbk1vdXNlVHJhY2tpbmcpIHtcbiAgICAgICAgdGhpcy5vcHRpb25zLnN0ZG91dC53cml0ZShFTkFCTEVfTU9VU0VfVFJBQ0tJTkcpXG4gICAgICB9XG4gICAgICB0aGlzLnJlc2V0RnJhbWVzRm9yQWx0U2NyZWVuKClcbiAgICAgIHRoaXMubmVlZHNFcmFzZUJlZm9yZVBhaW50ID0gdHJ1ZVxuICAgIH1cblxuICAgIC8vIFJlLXJlbmRlciB0aGUgUmVhY3QgdHJlZSB3aXRoIHVwZGF0ZWQgcHJvcHMgc28gdGhlIGNvbnRleHQgdmFsdWUgY2hhbmdlcy5cbiAgICAvLyBSZWFjdCdzIGNvbW1pdCBwaGFzZSB3aWxsIGNhbGwgb25Db21wdXRlTGF5b3V0KCkgdG8gcmVjYWxjdWxhdGUgeW9nYSBsYXlvdXRcbiAgICAvLyB3aXRoIHRoZSBuZXcgZGltZW5zaW9ucywgdGhlbiBjYWxsIG9uUmVuZGVyKCkgdG8gcmVuZGVyIHRoZSB1cGRhdGVkIGZyYW1lLlxuICAgIC8vIFdlIGRvbid0IGNhbGwgc2NoZWR1bGVSZW5kZXIoKSBoZXJlIGJlY2F1c2UgdGhhdCB3b3VsZCByZW5kZXIgYmVmb3JlIHRoZVxuICAgIC8vIGxheW91dCBpcyB1cGRhdGVkLCBjYXVzaW5nIGEgbWlzbWF0Y2ggYmV0d2VlbiB2aWV3cG9ydCBhbmQgY29udGVudCBkaW1lbnNpb25zLlxuICAgIGlmICh0aGlzLmN1cnJlbnROb2RlICE9PSBudWxsKSB7XG4gICAgICB0aGlzLnJlbmRlcih0aGlzLmN1cnJlbnROb2RlKVxuICAgIH1cbiAgfVxuXG4gIHJlc29sdmVFeGl0UHJvbWlzZTogKCkgPT4gdm9pZCA9ICgpID0+IHt9XG4gIHJlamVjdEV4aXRQcm9taXNlOiAocmVhc29uPzogRXJyb3IpID0+IHZvaWQgPSAoKSA9PiB7fVxuICB1bnN1YnNjcmliZUV4aXQ6ICgpID0+IHZvaWQgPSAoKSA9PiB7fVxuXG4gIC8qKlxuICAgKiBQYXVzZSBJbmsgYW5kIGhhbmQgdGhlIHRlcm1pbmFsIG92ZXIgdG8gYW4gZXh0ZXJuYWwgVFVJIChlLmcuIGdpdFxuICAgKiBjb21taXQgZWRpdG9yKS4gSW4gbm9uLWZ1bGxzY3JlZW4gbW9kZSB0aGlzIGVudGVycyB0aGUgYWx0IHNjcmVlbjtcbiAgICogaW4gZnVsbHNjcmVlbiBtb2RlIHdlJ3JlIGFscmVhZHkgaW4gYWx0IHNvIHdlIGp1c3QgY2xlYXIgaXQuXG4gICAqIENhbGwgYGV4aXRBbHRlcm5hdGVTY3JlZW4oKWAgd2hlbiBkb25lIHRvIHJlc3RvcmUgSW5rLlxuICAgKi9cbiAgZW50ZXJBbHRlcm5hdGVTY3JlZW4oKTogdm9pZCB7XG4gICAgdGhpcy5wYXVzZSgpXG4gICAgdGhpcy5zdXNwZW5kU3RkaW4oKVxuICAgIHRoaXMub3B0aW9ucy5zdGRvdXQud3JpdGUoXG4gICAgICAvLyBEaXNhYmxlIGV4dGVuZGVkIGtleSByZXBvcnRpbmcgZmlyc3Qg4oCUIGVkaXRvcnMgdGhhdCBkb24ndCBzcGVha1xuICAgICAgLy8gQ1NJLXUgKGUuZy4gbmFubykgc2hvdyBcIlVua25vd24gc2VxdWVuY2VcIiBmb3IgZXZlcnkgQ3RybC08a2V5PiBpZlxuICAgICAgLy8ga2l0dHkvbW9kaWZ5T3RoZXJLZXlzIHN0YXlzIGFjdGl2ZS4gZXhpdEFsdGVybmF0ZVNjcmVlbiByZS1lbmFibGVzLlxuICAgICAgRElTQUJMRV9LSVRUWV9LRVlCT0FSRCArXG4gICAgICAgIERJU0FCTEVfTU9ESUZZX09USEVSX0tFWVMgK1xuICAgICAgICAodGhpcy5hbHRTY3JlZW5Nb3VzZVRyYWNraW5nID8gRElTQUJMRV9NT1VTRV9UUkFDS0lORyA6ICcnKSArIC8vIGRpc2FibGUgbW91c2UgKG5vLW9wIGlmIG9mZilcbiAgICAgICAgKHRoaXMuYWx0U2NyZWVuQWN0aXZlID8gJycgOiAnXFx4MWJbPzEwNDloJykgKyAvLyBlbnRlciBhbHQgKGFscmVhZHkgaW4gYWx0IGlmIGZ1bGxzY3JlZW4pXG4gICAgICAgICdcXHgxYls/MTAwNGwnICsgLy8gZGlzYWJsZSBmb2N1cyByZXBvcnRpbmdcbiAgICAgICAgJ1xceDFiWzBtJyArIC8vIHJlc2V0IGF0dHJpYnV0ZXNcbiAgICAgICAgJ1xceDFiWz8yNWgnICsgLy8gc2hvdyBjdXJzb3JcbiAgICAgICAgJ1xceDFiWzJKJyArIC8vIGNsZWFyIHNjcmVlblxuICAgICAgICAnXFx4MWJbSCcsIC8vIGN1cnNvciBob21lXG4gICAgKVxuICB9XG5cbiAgLyoqXG4gICAqIFJlc3VtZSBJbmsgYWZ0ZXIgYW4gZXh0ZXJuYWwgVFVJIGhhbmRvZmYgd2l0aCBhIGZ1bGwgcmVwYWludC5cbiAgICogSW4gbm9uLWZ1bGxzY3JlZW4gbW9kZSB0aGlzIGV4aXRzIHRoZSBhbHQgc2NyZWVuIGJhY2sgdG8gbWFpbjtcbiAgICogaW4gZnVsbHNjcmVlbiBtb2RlIHdlIHJlLWVudGVyIGFsdCBhbmQgY2xlYXIgKyByZXBhaW50LlxuICAgKlxuICAgKiBUaGUgcmUtZW50ZXIgbWF0dGVyczogdGVybWluYWwgZWRpdG9ycyAodmltLCBuYW5vLCBsZXNzKSB3cml0ZVxuICAgKiBzbWN1cC9ybWN1cCAoPzEwNDloLz8xMDQ5bCksIHNvIGV2ZW4gdGhvdWdoIHdlIHN0YXJ0ZWQgaW4gYWx0LFxuICAgKiB0aGUgZWRpdG9yJ3Mgcm1jdXAgb24gZXhpdCBkcm9wcyB1cyB0byBtYWluIHNjcmVlbi4gV2l0aG91dFxuICAgKiByZS1lbnRlcmluZywgdGhlIDJKIGJlbG93IHdpcGVzIHRoZSB1c2VyJ3MgbWFpbi1zY3JlZW4gc2Nyb2xsYmFja1xuICAgKiBhbmQgc3Vic2VxdWVudCByZW5kZXJzIGxhbmQgaW4gbWFpbiDigJQgbmF0aXZlIHRlcm1pbmFsIHNjcm9sbFxuICAgKiByZXR1cm5zLCBmdWxsc2NyZWVuIHNjcm9sbCBpcyBkZWFkLlxuICAgKi9cbiAgZXhpdEFsdGVybmF0ZVNjcmVlbigpOiB2b2lkIHtcbiAgICB0aGlzLm9wdGlvbnMuc3Rkb3V0LndyaXRlKFxuICAgICAgKHRoaXMuYWx0U2NyZWVuQWN0aXZlID8gRU5URVJfQUxUX1NDUkVFTiA6ICcnKSArIC8vIHJlLWVudGVyIGFsdCDigJQgdmltJ3Mgcm1jdXAgZHJvcHBlZCB1cyB0byBtYWluXG4gICAgICAgICdcXHgxYlsySicgKyAvLyBjbGVhciBzY3JlZW4gKG5vdyBhbHQgaWYgZnVsbHNjcmVlbilcbiAgICAgICAgJ1xceDFiW0gnICsgLy8gY3Vyc29yIGhvbWVcbiAgICAgICAgKHRoaXMuYWx0U2NyZWVuTW91c2VUcmFja2luZyA/IEVOQUJMRV9NT1VTRV9UUkFDS0lORyA6ICcnKSArIC8vIHJlLWVuYWJsZSBtb3VzZSAoc2tpcCBpZiBDTEFVREVfQ09ERV9ESVNBQkxFX01PVVNFKVxuICAgICAgICAodGhpcy5hbHRTY3JlZW5BY3RpdmUgPyAnJyA6ICdcXHgxYls/MTA0OWwnKSArIC8vIGV4aXQgYWx0IChub24tZnVsbHNjcmVlbiBvbmx5KVxuICAgICAgICAnXFx4MWJbPzI1bCcsIC8vIGhpZGUgY3Vyc29yIChJbmsgbWFuYWdlcylcbiAgICApXG4gICAgdGhpcy5yZXN1bWVTdGRpbigpXG4gICAgaWYgKHRoaXMuYWx0U2NyZWVuQWN0aXZlKSB7XG4gICAgICB0aGlzLnJlc2V0RnJhbWVzRm9yQWx0U2NyZWVuKClcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5yZXBhaW50KClcbiAgICB9XG4gICAgdGhpcy5yZXN1bWUoKVxuICAgIC8vIFJlLWVuYWJsZSBmb2N1cyByZXBvcnRpbmcgYW5kIGV4dGVuZGVkIGtleSByZXBvcnRpbmcg4oCUIHRlcm1pbmFsXG4gICAgLy8gZWRpdG9ycyAodmltLCBuYW5vLCBldGMuKSB3cml0ZSB0aGVpciBvd24gbW9kaWZ5T3RoZXJLZXlzIGxldmVsIG9uXG4gICAgLy8gZW50cnkgYW5kIHJlc2V0IGl0IG9uIGV4aXQsIGxlYXZpbmcgdXMgdW5hYmxlIHRvIGRpc3Rpbmd1aXNoXG4gICAgLy8gY3RybCtzaGlmdCs8bGV0dGVyPiBmcm9tIGN0cmwrPGxldHRlcj4uIFBvcC1iZWZvcmUtcHVzaCBrZWVwcyB0aGVcbiAgICAvLyBLaXR0eSBzdGFjayBiYWxhbmNlZCAoYSB3ZWxsLWJlaGF2ZWQgZWRpdG9yIHJlc3RvcmVzIG91ciBlbnRyeSwgc29cbiAgICAvLyB3aXRob3V0IHRoZSBwb3Agd2UnZCBhY2N1bXVsYXRlIGRlcHRoIG9uIGVhY2ggZWRpdG9yIHJvdW5kLXRyaXApLlxuICAgIHRoaXMub3B0aW9ucy5zdGRvdXQud3JpdGUoXG4gICAgICAnXFx4MWJbPzEwMDRoJyArXG4gICAgICAgIChzdXBwb3J0c0V4dGVuZGVkS2V5cygpXG4gICAgICAgICAgPyBESVNBQkxFX0tJVFRZX0tFWUJPQVJEICtcbiAgICAgICAgICAgIEVOQUJMRV9LSVRUWV9LRVlCT0FSRCArXG4gICAgICAgICAgICBFTkFCTEVfTU9ESUZZX09USEVSX0tFWVNcbiAgICAgICAgICA6ICcnKSxcbiAgICApXG4gIH1cblxuICBvblJlbmRlcigpIHtcbiAgICBpZiAodGhpcy5pc1VubW91bnRlZCB8fCB0aGlzLmlzUGF1c2VkKSB7XG4gICAgICByZXR1cm5cbiAgICB9XG4gICAgLy8gRW50ZXJpbmcgYSByZW5kZXIgY2FuY2VscyBhbnkgcGVuZGluZyBkcmFpbiB0aWNrIOKAlCB0aGlzIHJlbmRlciB3aWxsXG4gICAgLy8gaGFuZGxlIHRoZSBkcmFpbiAoYW5kIHJlLXNjaGVkdWxlIGJlbG93IGlmIG5lZWRlZCkuIFByZXZlbnRzIGFcbiAgICAvLyB3aGVlbC1ldmVudC10cmlnZ2VyZWQgcmVuZGVyIEFORCBhIGRyYWluLXRpbWVyIHJlbmRlciBib3RoIGZpcmluZy5cbiAgICBpZiAodGhpcy5kcmFpblRpbWVyICE9PSBudWxsKSB7XG4gICAgICBjbGVhclRpbWVvdXQodGhpcy5kcmFpblRpbWVyKVxuICAgICAgdGhpcy5kcmFpblRpbWVyID0gbnVsbFxuICAgIH1cblxuICAgIC8vIEZsdXNoIGRlZmVycmVkIGludGVyYWN0aW9uLXRpbWUgdXBkYXRlIGJlZm9yZSByZW5kZXJpbmcgc28gd2UgY2FsbFxuICAgIC8vIERhdGUubm93KCkgYXQgbW9zdCBvbmNlIHBlciBmcmFtZSBpbnN0ZWFkIG9mIG9uY2UgcGVyIGtleXByZXNzLlxuICAgIC8vIERvbmUgYmVmb3JlIHRoZSByZW5kZXIgdG8gYXZvaWQgZGlydHlpbmcgc3RhdGUgdGhhdCB3b3VsZCB0cmlnZ2VyXG4gICAgLy8gYW4gZXh0cmEgUmVhY3QgcmUtcmVuZGVyIGN5Y2xlLlxuICAgIGZsdXNoSW50ZXJhY3Rpb25UaW1lKClcblxuICAgIGNvbnN0IHJlbmRlclN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KClcbiAgICBjb25zdCB0ZXJtaW5hbFdpZHRoID0gdGhpcy5vcHRpb25zLnN0ZG91dC5jb2x1bW5zIHx8IDgwXG4gICAgY29uc3QgdGVybWluYWxSb3dzID0gdGhpcy5vcHRpb25zLnN0ZG91dC5yb3dzIHx8IDI0XG5cbiAgICBjb25zdCBmcmFtZSA9IHRoaXMucmVuZGVyZXIoe1xuICAgICAgZnJvbnRGcmFtZTogdGhpcy5mcm9udEZyYW1lLFxuICAgICAgYmFja0ZyYW1lOiB0aGlzLmJhY2tGcmFtZSxcbiAgICAgIGlzVFRZOiB0aGlzLm9wdGlvbnMuc3Rkb3V0LmlzVFRZLFxuICAgICAgdGVybWluYWxXaWR0aCxcbiAgICAgIHRlcm1pbmFsUm93cyxcbiAgICAgIGFsdFNjcmVlbjogdGhpcy5hbHRTY3JlZW5BY3RpdmUsXG4gICAgICBwcmV2RnJhbWVDb250YW1pbmF0ZWQ6IHRoaXMucHJldkZyYW1lQ29udGFtaW5hdGVkLFxuICAgIH0pXG4gICAgY29uc3QgcmVuZGVyZXJNcyA9IHBlcmZvcm1hbmNlLm5vdygpIC0gcmVuZGVyU3RhcnRcblxuICAgIC8vIFN0aWNreS9hdXRvLWZvbGxvdyBzY3JvbGxlZCB0aGUgU2Nyb2xsQm94IHRoaXMgZnJhbWUuIFRyYW5zbGF0ZSB0aGVcbiAgICAvLyBzZWxlY3Rpb24gYnkgdGhlIHNhbWUgZGVsdGEgc28gdGhlIGhpZ2hsaWdodCBzdGF5cyBhbmNob3JlZCB0byB0aGVcbiAgICAvLyBURVhUIChuYXRpdmUgdGVybWluYWwgYmVoYXZpb3Ig4oCUIHRoZSBzZWxlY3Rpb24gd2Fsa3MgdXAgdGhlIHNjcmVlblxuICAgIC8vIGFzIGNvbnRlbnQgc2Nyb2xscywgZXZlbnR1YWxseSBjbGlwcGluZyBhdCB0aGUgdG9wKS4gZnJvbnRGcmFtZVxuICAgIC8vIHN0aWxsIGhvbGRzIHRoZSBQUkVWSU9VUyBmcmFtZSdzIHNjcmVlbiAoc3dhcCBpcyBhdCB+NTAwIGJlbG93KSwgc29cbiAgICAvLyBjYXB0dXJlU2Nyb2xsZWRSb3dzIHJlYWRzIHRoZSByb3dzIHRoYXQgYXJlIGFib3V0IHRvIHNjcm9sbCBvdXRcbiAgICAvLyBiZWZvcmUgdGhleSdyZSBvdmVyd3JpdHRlbiDigJQgdGhlIHRleHQgc3RheXMgY29weWFibGUgdW50aWwgdGhlXG4gICAgLy8gc2VsZWN0aW9uIHNjcm9sbHMgZW50aXJlbHkgb2ZmLiBEdXJpbmcgZHJhZywgZm9jdXMgdHJhY2tzIHRoZSBtb3VzZVxuICAgIC8vIChzY3JlZW4tbG9jYWwpIHNvIG9ubHkgYW5jaG9yIHNoaWZ0cyDigJQgc2VsZWN0aW9uIGdyb3dzIHRvd2FyZCB0aGVcbiAgICAvLyBtb3VzZSBhcyB0aGUgYW5jaG9yIHdhbGtzIHVwLiBBZnRlciByZWxlYXNlLCBib3RoIGVuZHMgYXJlIHRleHQtXG4gICAgLy8gYW5jaG9yZWQgYW5kIG1vdmUgYXMgYSBibG9jay5cbiAgICBjb25zdCBmb2xsb3cgPSBjb25zdW1lRm9sbG93U2Nyb2xsKClcbiAgICBpZiAoXG4gICAgICBmb2xsb3cgJiZcbiAgICAgIHRoaXMuc2VsZWN0aW9uLmFuY2hvciAmJlxuICAgICAgLy8gT25seSB0cmFuc2xhdGUgaWYgdGhlIHNlbGVjdGlvbiBpcyBPTiBzY3JvbGxib3ggY29udGVudC4gU2VsZWN0aW9uc1xuICAgICAgLy8gaW4gdGhlIGZvb3Rlci9wcm9tcHQvU3RpY2t5UHJvbXB0SGVhZGVyIGFyZSBvbiBzdGF0aWMgdGV4dCDigJQgdGhlXG4gICAgICAvLyBzY3JvbGwgZG9lc24ndCBtb3ZlIHdoYXQncyB1bmRlciB0aGVtLiBXaXRob3V0IHRoaXMgZ3VhcmQsIGFcbiAgICAgIC8vIGZvb3RlciBzZWxlY3Rpb24gd291bGQgYmUgc2hpZnRlZCBieSAtZGVsdGEgdGhlbiBjbGFtcGVkIHRvXG4gICAgICAvLyB2aWV3cG9ydEJvdHRvbSwgdGVsZXBvcnRpbmcgaXQgaW50byB0aGUgc2Nyb2xsYm94LiBNaXJyb3IgdGhlXG4gICAgICAvLyBib3VuZHMgY2hlY2sgdGhlIGRlbGV0ZWQgY2hlY2soKSBpbiBTY3JvbGxLZXliaW5kaW5nSGFuZGxlciBoYWQuXG4gICAgICB0aGlzLnNlbGVjdGlvbi5hbmNob3Iucm93ID49IGZvbGxvdy52aWV3cG9ydFRvcCAmJlxuICAgICAgdGhpcy5zZWxlY3Rpb24uYW5jaG9yLnJvdyA8PSBmb2xsb3cudmlld3BvcnRCb3R0b21cbiAgICApIHtcbiAgICAgIGNvbnN0IHsgZGVsdGEsIHZpZXdwb3J0VG9wLCB2aWV3cG9ydEJvdHRvbSB9ID0gZm9sbG93XG4gICAgICAvLyBjYXB0dXJlU2Nyb2xsZWRSb3dzIGFuZCBzaGlmdCogYXJlIGEgcGFpcjogY2FwdHVyZSBncmFicyByb3dzIGFib3V0XG4gICAgICAvLyB0byBzY3JvbGwgb2ZmLCBzaGlmdCBtb3ZlcyB0aGUgc2VsZWN0aW9uIGVuZHBvaW50IHNvIHRoZSBzYW1lIHJvd3NcbiAgICAgIC8vIHdvbid0IGludGVyc2VjdCBhZ2FpbiBuZXh0IGZyYW1lLiBDYXB0dXJpbmcgd2l0aG91dCBzaGlmdGluZyBsZWF2ZXNcbiAgICAgIC8vIHRoZSBlbmRwb2ludCBpbiBwbGFjZSwgc28gdGhlIFNBTUUgdmlld3BvcnQgcm93cyByZS1pbnRlcnNlY3QgZXZlcnlcbiAgICAgIC8vIGZyYW1lIGFuZCBzY3JvbGxlZE9mZkFib3ZlIGdyb3dzIHdpdGhvdXQgYm91bmQg4oCUIGdldFNlbGVjdGVkVGV4dFxuICAgICAgLy8gdGhlbiByZXR1cm5zIGV2ZXItZ3Jvd2luZyB0ZXh0IG9uIGVhY2ggcmUtY29weS4gS2VlcCBjYXB0dXJlIGluc2lkZVxuICAgICAgLy8gZWFjaCBzaGlmdCBicmFuY2ggc28gdGhlIHBhaXJpbmcgY2FuJ3QgYmUgYnJva2VuIGJ5IGEgbmV3IGd1YXJkLlxuICAgICAgaWYgKHRoaXMuc2VsZWN0aW9uLmlzRHJhZ2dpbmcpIHtcbiAgICAgICAgaWYgKGhhc1NlbGVjdGlvbih0aGlzLnNlbGVjdGlvbikpIHtcbiAgICAgICAgICBjYXB0dXJlU2Nyb2xsZWRSb3dzKFxuICAgICAgICAgICAgdGhpcy5zZWxlY3Rpb24sXG4gICAgICAgICAgICB0aGlzLmZyb250RnJhbWUuc2NyZWVuLFxuICAgICAgICAgICAgdmlld3BvcnRUb3AsXG4gICAgICAgICAgICB2aWV3cG9ydFRvcCArIGRlbHRhIC0gMSxcbiAgICAgICAgICAgICdhYm92ZScsXG4gICAgICAgICAgKVxuICAgICAgICB9XG4gICAgICAgIHNoaWZ0QW5jaG9yKHRoaXMuc2VsZWN0aW9uLCAtZGVsdGEsIHZpZXdwb3J0VG9wLCB2aWV3cG9ydEJvdHRvbSlcbiAgICAgIH0gZWxzZSBpZiAoXG4gICAgICAgIC8vIEZsYWctMyBndWFyZDogdGhlIGFuY2hvciBjaGVjayBhYm92ZSBvbmx5IHByb3ZlcyBPTkUgZW5kcG9pbnQgaXNcbiAgICAgICAgLy8gb24gc2Nyb2xsYm94IGNvbnRlbnQuIEEgZHJhZyBmcm9tIHJvdyAzIChzY3JvbGxib3gpIGludG8gdGhlXG4gICAgICAgIC8vIGZvb3RlciBhdCByb3cgNiwgdGhlbiByZWxlYXNlLCBsZWF2ZXMgZm9jdXMgb3V0c2lkZSB0aGUgdmlld3BvcnRcbiAgICAgICAgLy8g4oCUIHNoaWZ0U2VsZWN0aW9uRm9yRm9sbG93IHdvdWxkIGNsYW1wIGl0IHRvIHZpZXdwb3J0Qm90dG9tLFxuICAgICAgICAvLyB0ZWxlcG9ydGluZyB0aGUgaGlnaGxpZ2h0IGZyb20gc3RhdGljIGZvb3RlciBpbnRvIHRoZSBzY3JvbGxib3guXG4gICAgICAgIC8vIFN5bW1ldHJpYyBjaGVjazogcmVxdWlyZSBCT1RIIGVuZHMgaW5zaWRlIHRvIHRyYW5zbGF0ZS4gQVxuICAgICAgICAvLyBzdHJhZGRsaW5nIHNlbGVjdGlvbiBmYWxscyB0aHJvdWdoIHRvIE5FSVRIRVIgc2hpZnQgTk9SIGNhcHR1cmU6XG4gICAgICAgIC8vIHRoZSBmb290ZXIgZW5kcG9pbnQgcGlucyB0aGUgc2VsZWN0aW9uLCB0ZXh0IHNjcm9sbHMgYXdheSB1bmRlclxuICAgICAgICAvLyB0aGUgaGlnaGxpZ2h0LCBhbmQgZ2V0U2VsZWN0ZWRUZXh0IHJlYWRzIHRoZSBDVVJSRU5UIHNjcmVlblxuICAgICAgICAvLyBjb250ZW50cyDigJQgbm8gYWNjdW11bGF0aW9uLiBEcmFnZ2luZyBicmFuY2ggZG9lc24ndCBuZWVkIHRoaXM6XG4gICAgICAgIC8vIHNoaWZ0QW5jaG9yIGlnbm9yZXMgZm9jdXMsIGFuZCB0aGUgYW5jaG9yIERPRVMgc2hpZnQgKHNvIGNhcHR1cmVcbiAgICAgICAgLy8gaXMgY29ycmVjdCB0aGVyZSBldmVuIHdoZW4gZm9jdXMgaXMgaW4gdGhlIGZvb3RlcikuXG4gICAgICAgICF0aGlzLnNlbGVjdGlvbi5mb2N1cyB8fFxuICAgICAgICAodGhpcy5zZWxlY3Rpb24uZm9jdXMucm93ID49IHZpZXdwb3J0VG9wICYmXG4gICAgICAgICAgdGhpcy5zZWxlY3Rpb24uZm9jdXMucm93IDw9IHZpZXdwb3J0Qm90dG9tKVxuICAgICAgKSB7XG4gICAgICAgIGlmIChoYXNTZWxlY3Rpb24odGhpcy5zZWxlY3Rpb24pKSB7XG4gICAgICAgICAgY2FwdHVyZVNjcm9sbGVkUm93cyhcbiAgICAgICAgICAgIHRoaXMuc2VsZWN0aW9uLFxuICAgICAgICAgICAgdGhpcy5mcm9udEZyYW1lLnNjcmVlbixcbiAgICAgICAgICAgIHZpZXdwb3J0VG9wLFxuICAgICAgICAgICAgdmlld3BvcnRUb3AgKyBkZWx0YSAtIDEsXG4gICAgICAgICAgICAnYWJvdmUnLFxuICAgICAgICAgIClcbiAgICAgICAgfVxuICAgICAgICBjb25zdCBjbGVhcmVkID0gc2hpZnRTZWxlY3Rpb25Gb3JGb2xsb3coXG4gICAgICAgICAgdGhpcy5zZWxlY3Rpb24sXG4gICAgICAgICAgLWRlbHRhLFxuICAgICAgICAgIHZpZXdwb3J0VG9wLFxuICAgICAgICAgIHZpZXdwb3J0Qm90dG9tLFxuICAgICAgICApXG4gICAgICAgIC8vIEF1dG8tY2xlYXIgKGJvdGggZW5kcyBvdmVyc2hvdCBtaW5Sb3cpIG11c3Qgbm90aWZ5IFJlYWN0LWxhbmRcbiAgICAgICAgLy8gc28gdXNlSGFzU2VsZWN0aW9uIHJlLXJlbmRlcnMgYW5kIHRoZSBmb290ZXIgY29weS9lc2NhcGUgaGludFxuICAgICAgICAvLyBkaXNhcHBlYXJzLiBub3RpZnlTZWxlY3Rpb25DaGFuZ2UoKSB3b3VsZCByZWN1cnNlIGludG8gb25SZW5kZXI7XG4gICAgICAgIC8vIGZpcmUgdGhlIGxpc3RlbmVycyBkaXJlY3RseSDigJQgdGhleSBzY2hlZHVsZSBhIFJlYWN0IHVwZGF0ZSBmb3JcbiAgICAgICAgLy8gTEFURVIsIHRoZXkgZG9uJ3QgcmUtZW50ZXIgdGhpcyBmcmFtZS5cbiAgICAgICAgaWYgKGNsZWFyZWQpIGZvciAoY29uc3QgY2Igb2YgdGhpcy5zZWxlY3Rpb25MaXN0ZW5lcnMpIGNiKClcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBTZWxlY3Rpb24gb3ZlcmxheTogaW52ZXJ0IGNlbGwgc3R5bGVzIGluIHRoZSBzY3JlZW4gYnVmZmVyIGl0c2VsZixcbiAgICAvLyBzbyB0aGUgZGlmZiBwaWNrcyB1cCBzZWxlY3Rpb24gYXMgb3JkaW5hcnkgY2VsbCBjaGFuZ2VzIGFuZFxuICAgIC8vIExvZ1VwZGF0ZSByZW1haW5zIGEgcHVyZSBkaWZmIGVuZ2luZS5cbiAgICAvL1xuICAgIC8vIEZ1bGwtc2NyZWVuIGRhbWFnZSAoUFIgIzIwMTIwKSBpcyBhIGNvcnJlY3RuZXNzIGJhY2tzdG9wIGZvciB0aGVcbiAgICAvLyBzaWJsaW5nLXJlc2l6ZSBibGVlZDogd2hlbiBmbGV4Ym94IHNpYmxpbmdzIHJlc2l6ZSBiZXR3ZWVuIGZyYW1lc1xuICAgIC8vIChzcGlubmVyIGFwcGVhcnMg4oaSIGJvdHRvbSBncm93cyDihpIgc2Nyb2xsYm94IHNocmlua3MpLCB0aGVcbiAgICAvLyBjYWNoZWQtY2xlYXIgKyBjbGlwLWFuZC1jdWxsICsgc2V0Q2VsbEF0IGRhbWFnZSB1bmlvbiBjYW4gbWlzc1xuICAgIC8vIHRyYW5zaXRpb24gY2VsbHMgYXQgdGhlIGJvdW5kYXJ5LiBCdXQgdGhhdCBvbmx5IGhhcHBlbnMgd2hlbiBsYXlvdXRcbiAgICAvLyBhY3R1YWxseSBTSElGVFMg4oCUIGRpZExheW91dFNoaWZ0KCkgdHJhY2tzIGV4YWN0bHkgdGhpcyAoYW55IG5vZGUnc1xuICAgIC8vIGNhY2hlZCB5b2dhIHBvc2l0aW9uL3NpemUgZGlmZmVycyBmcm9tIGN1cnJlbnQsIG9yIGEgY2hpbGQgd2FzXG4gICAgLy8gcmVtb3ZlZCkuIFN0ZWFkeS1zdGF0ZSBmcmFtZXMgKHNwaW5uZXIgcm90YXRlLCBjbG9jayB0aWNrLCB0ZXh0XG4gICAgLy8gc3RyZWFtIGludG8gZml4ZWQtaGVpZ2h0IGJveCkgZG9uJ3Qgc2hpZnQgbGF5b3V0LCBzbyBub3JtYWwgZGFtYWdlXG4gICAgLy8gYm91bmRzIGFyZSBjb3JyZWN0IGFuZCBkaWZmRWFjaCBvbmx5IGNvbXBhcmVzIHRoZSBkYW1hZ2VkIHJlZ2lvbi5cbiAgICAvL1xuICAgIC8vIFNlbGVjdGlvbiBhbHNvIHJlcXVpcmVzIGZ1bGwgZGFtYWdlOiBvdmVybGF5IHdyaXRlcyB2aWEgc2V0Q2VsbFN0eWxlSWRcbiAgICAvLyB3aGljaCBkb2Vzbid0IHRyYWNrIGRhbWFnZSwgYW5kIHByZXYtZnJhbWUgb3ZlcmxheSBjZWxscyBuZWVkIHRvIGJlXG4gICAgLy8gY29tcGFyZWQgd2hlbiBzZWxlY3Rpb24gbW92ZXMvY2xlYXJzLiBwcmV2RnJhbWVDb250YW1pbmF0ZWQgY292ZXJzXG4gICAgLy8gdGhlIGZyYW1lLWFmdGVyLXNlbGVjdGlvbi1jbGVhcnMgY2FzZS5cbiAgICBsZXQgc2VsQWN0aXZlID0gZmFsc2VcbiAgICBsZXQgaGxBY3RpdmUgPSBmYWxzZVxuICAgIGlmICh0aGlzLmFsdFNjcmVlbkFjdGl2ZSkge1xuICAgICAgc2VsQWN0aXZlID0gaGFzU2VsZWN0aW9uKHRoaXMuc2VsZWN0aW9uKVxuICAgICAgaWYgKHNlbEFjdGl2ZSkge1xuICAgICAgICBhcHBseVNlbGVjdGlvbk92ZXJsYXkoZnJhbWUuc2NyZWVuLCB0aGlzLnNlbGVjdGlvbiwgdGhpcy5zdHlsZVBvb2wpXG4gICAgICB9XG4gICAgICAvLyBTY2FuLWhpZ2hsaWdodDogaW52ZXJzZSBvbiBBTEwgdmlzaWJsZSBtYXRjaGVzIChsZXNzL3ZpbSBzdHlsZSkuXG4gICAgICAvLyBQb3NpdGlvbi1oaWdobGlnaHQgKGJlbG93KSBvdmVybGF5cyBDVVJSRU5UICh5ZWxsb3cpIG9uIHRvcC5cbiAgICAgIGhsQWN0aXZlID0gYXBwbHlTZWFyY2hIaWdobGlnaHQoXG4gICAgICAgIGZyYW1lLnNjcmVlbixcbiAgICAgICAgdGhpcy5zZWFyY2hIaWdobGlnaHRRdWVyeSxcbiAgICAgICAgdGhpcy5zdHlsZVBvb2wsXG4gICAgICApXG4gICAgICAvLyBQb3NpdGlvbi1iYXNlZCBDVVJSRU5UOiB3cml0ZSB5ZWxsb3cgYXQgcG9zaXRpb25zW2N1cnJlbnRJZHhdICtcbiAgICAgIC8vIHJvd09mZnNldC4gTm8gc2Nhbm5pbmcg4oCUIHBvc2l0aW9ucyBjYW1lIGZyb20gYSBwcmlvciBzY2FuIHdoZW5cbiAgICAgIC8vIHRoZSBtZXNzYWdlIGZpcnN0IG1vdW50ZWQuIE1lc3NhZ2UtcmVsYXRpdmUgKyByb3dPZmZzZXQgPSBzY3JlZW4uXG4gICAgICBpZiAodGhpcy5zZWFyY2hQb3NpdGlvbnMpIHtcbiAgICAgICAgY29uc3Qgc3AgPSB0aGlzLnNlYXJjaFBvc2l0aW9uc1xuICAgICAgICBjb25zdCBwb3NBcHBsaWVkID0gYXBwbHlQb3NpdGlvbmVkSGlnaGxpZ2h0KFxuICAgICAgICAgIGZyYW1lLnNjcmVlbixcbiAgICAgICAgICB0aGlzLnN0eWxlUG9vbCxcbiAgICAgICAgICBzcC5wb3NpdGlvbnMsXG4gICAgICAgICAgc3Aucm93T2Zmc2V0LFxuICAgICAgICAgIHNwLmN1cnJlbnRJZHgsXG4gICAgICAgIClcbiAgICAgICAgaGxBY3RpdmUgPSBobEFjdGl2ZSB8fCBwb3NBcHBsaWVkXG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gRnVsbC1kYW1hZ2UgYmFja3N0b3A6IGFwcGxpZXMgb24gQk9USCBhbHQtc2NyZWVuIGFuZCBtYWluLXNjcmVlbi5cbiAgICAvLyBMYXlvdXQgc2hpZnRzIChzcGlubmVyIGFwcGVhcnMsIHN0YXR1cyBsaW5lIHJlc2l6ZXMpIGNhbiBsZWF2ZSBzdGFsZVxuICAgIC8vIGNlbGxzIGF0IHNpYmxpbmcgYm91bmRhcmllcyB0aGF0IHBlci1ub2RlIGRhbWFnZSB0cmFja2luZyBtaXNzZXMuXG4gICAgLy8gU2VsZWN0aW9uL2hpZ2hsaWdodCBvdmVybGF5cyB3cml0ZSB2aWEgc2V0Q2VsbFN0eWxlSWQgd2hpY2ggZG9lc24ndFxuICAgIC8vIHRyYWNrIGRhbWFnZS4gcHJldkZyYW1lQ29udGFtaW5hdGVkIGNvdmVycyB0aGUgY2xlYW51cCBmcmFtZS5cbiAgICBpZiAoXG4gICAgICBkaWRMYXlvdXRTaGlmdCgpIHx8XG4gICAgICBzZWxBY3RpdmUgfHxcbiAgICAgIGhsQWN0aXZlIHx8XG4gICAgICB0aGlzLnByZXZGcmFtZUNvbnRhbWluYXRlZFxuICAgICkge1xuICAgICAgZnJhbWUuc2NyZWVuLmRhbWFnZSA9IHtcbiAgICAgICAgeDogMCxcbiAgICAgICAgeTogMCxcbiAgICAgICAgd2lkdGg6IGZyYW1lLnNjcmVlbi53aWR0aCxcbiAgICAgICAgaGVpZ2h0OiBmcmFtZS5zY3JlZW4uaGVpZ2h0LFxuICAgICAgfVxuICAgIH1cblxuICAgIC8vIEFsdC1zY3JlZW46IGFuY2hvciB0aGUgcGh5c2ljYWwgY3Vyc29yIHRvICgwLDApIGJlZm9yZSBldmVyeSBkaWZmLlxuICAgIC8vIEFsbCBjdXJzb3IgbW92ZXMgaW4gbG9nLXVwZGF0ZSBhcmUgUkVMQVRJVkUgdG8gcHJldi5jdXJzb3I7IGlmIHRtdXhcbiAgICAvLyAob3IgYW55IGVtdWxhdG9yKSBwZXJ0dXJicyB0aGUgcGh5c2ljYWwgY3Vyc29yIG91dC1vZi1iYW5kIChzdGF0dXNcbiAgICAvLyBiYXIgcmVmcmVzaCwgcGFuZSByZWRyYXcsIENtZCtLIHdpcGUpLCB0aGUgcmVsYXRpdmUgbW92ZXMgZHJpZnQgYW5kXG4gICAgLy8gY29udGVudCBjcmVlcHMgdXAgMSByb3cvZnJhbWUuIENTSSBIIHJlc2V0cyB0aGUgcGh5c2ljYWwgY3Vyc29yO1xuICAgIC8vIHBhc3NpbmcgcHJldi5jdXJzb3I9KDAsMCkgbWFrZXMgdGhlIGRpZmYgY29tcHV0ZSBmcm9tIHRoZSBzYW1lIHNwb3QuXG4gICAgLy8gU2VsZi1oZWFsaW5nIGFnYWluc3QgYW55IGV4dGVybmFsIGN1cnNvciBtYW5pcHVsYXRpb24uIE1haW4tc2NyZWVuXG4gICAgLy8gY2FuJ3QgZG8gdGhpcyDigJQgY3Vyc29yLnkgdHJhY2tzIHNjcm9sbGJhY2sgcm93cyBDU0kgSCBjYW4ndCByZWFjaC5cbiAgICAvLyBUaGUgQ1NJIEggd3JpdGUgaXMgZGVmZXJyZWQgdW50aWwgYWZ0ZXIgdGhlIGRpZmYgaXMgY29tcHV0ZWQgc28gd2VcbiAgICAvLyBjYW4gc2tpcCBpdCBmb3IgZW1wdHkgZGlmZnMgKG5vIHdyaXRlcyDihpIgcGh5c2ljYWwgY3Vyc29yIHVudXNlZCkuXG4gICAgbGV0IHByZXZGcmFtZSA9IHRoaXMuZnJvbnRGcmFtZVxuICAgIGlmICh0aGlzLmFsdFNjcmVlbkFjdGl2ZSkge1xuICAgICAgcHJldkZyYW1lID0geyAuLi50aGlzLmZyb250RnJhbWUsIGN1cnNvcjogQUxUX1NDUkVFTl9BTkNIT1JfQ1VSU09SIH1cbiAgICB9XG5cbiAgICBjb25zdCB0RGlmZiA9IHBlcmZvcm1hbmNlLm5vdygpXG4gICAgY29uc3QgZGlmZiA9IHRoaXMubG9nLnJlbmRlcihcbiAgICAgIHByZXZGcmFtZSxcbiAgICAgIGZyYW1lLFxuICAgICAgdGhpcy5hbHRTY3JlZW5BY3RpdmUsXG4gICAgICAvLyBERUNTVEJNIG5lZWRzIEJTVS9FU1UgYXRvbWljaXR5IOKAlCB3aXRob3V0IGl0IHRoZSBvdXRlciB0ZXJtaW5hbFxuICAgICAgLy8gcmVuZGVycyB0aGUgc2Nyb2xsZWQtYnV0LW5vdC15ZXQtcmVwYWludGVkIGludGVybWVkaWF0ZSBzdGF0ZS5cbiAgICAgIC8vIHRtdXggaXMgdGhlIG1haW4gY2FzZSAocmUtZW1pdHMgREVDU1RCTSB3aXRoIGl0cyBvd24gdGltaW5nIGFuZFxuICAgICAgLy8gZG9lc24ndCBpbXBsZW1lbnQgREVDIDIwMjYsIHNvIFNZTkNfT1VUUFVUX1NVUFBPUlRFRCBpcyBmYWxzZSkuXG4gICAgICBTWU5DX09VVFBVVF9TVVBQT1JURUQsXG4gICAgKVxuICAgIGNvbnN0IGRpZmZNcyA9IHBlcmZvcm1hbmNlLm5vdygpIC0gdERpZmZcbiAgICAvLyBTd2FwIGJ1ZmZlcnNcbiAgICB0aGlzLmJhY2tGcmFtZSA9IHRoaXMuZnJvbnRGcmFtZVxuICAgIHRoaXMuZnJvbnRGcmFtZSA9IGZyYW1lXG5cbiAgICAvLyBQZXJpb2RpY2FsbHkgcmVzZXQgY2hhci9oeXBlcmxpbmsgcG9vbHMgdG8gcHJldmVudCB1bmJvdW5kZWQgZ3Jvd3RoXG4gICAgLy8gZHVyaW5nIGxvbmcgc2Vzc2lvbnMuIDUgbWludXRlcyBpcyBpbmZyZXF1ZW50IGVub3VnaCB0aGF0IHRoZSBPKGNlbGxzKVxuICAgIC8vIG1pZ3JhdGlvbiBjb3N0IGlzIG5lZ2xpZ2libGUuIFJldXNlcyByZW5kZXJTdGFydCB0byBhdm9pZCBleHRyYSBjbG9jayBjYWxsLlxuICAgIGlmIChyZW5kZXJTdGFydCAtIHRoaXMubGFzdFBvb2xSZXNldFRpbWUgPiA1ICogNjAgKiAxMDAwKSB7XG4gICAgICB0aGlzLnJlc2V0UG9vbHMoKVxuICAgICAgdGhpcy5sYXN0UG9vbFJlc2V0VGltZSA9IHJlbmRlclN0YXJ0XG4gICAgfVxuXG4gICAgY29uc3QgZmxpY2tlcnM6IEZyYW1lRXZlbnRbJ2ZsaWNrZXJzJ10gPSBbXVxuICAgIGZvciAoY29uc3QgcGF0Y2ggb2YgZGlmZikge1xuICAgICAgaWYgKHBhdGNoLnR5cGUgPT09ICdjbGVhclRlcm1pbmFsJykge1xuICAgICAgICBmbGlja2Vycy5wdXNoKHtcbiAgICAgICAgICBkZXNpcmVkSGVpZ2h0OiBmcmFtZS5zY3JlZW4uaGVpZ2h0LFxuICAgICAgICAgIGF2YWlsYWJsZUhlaWdodDogZnJhbWUudmlld3BvcnQuaGVpZ2h0LFxuICAgICAgICAgIHJlYXNvbjogcGF0Y2gucmVhc29uLFxuICAgICAgICB9KVxuICAgICAgICBpZiAoaXNEZWJ1Z1JlcGFpbnRzRW5hYmxlZCgpICYmIHBhdGNoLmRlYnVnKSB7XG4gICAgICAgICAgY29uc3QgY2hhaW4gPSBkb20uZmluZE93bmVyQ2hhaW5BdFJvdyhcbiAgICAgICAgICAgIHRoaXMucm9vdE5vZGUsXG4gICAgICAgICAgICBwYXRjaC5kZWJ1Zy50cmlnZ2VyWSxcbiAgICAgICAgICApXG4gICAgICAgICAgbG9nRm9yRGVidWdnaW5nKFxuICAgICAgICAgICAgYFtSRVBBSU5UXSBmdWxsIHJlc2V0IMK3ICR7cGF0Y2gucmVhc29ufSDCtyByb3cgJHtwYXRjaC5kZWJ1Zy50cmlnZ2VyWX1cXG5gICtcbiAgICAgICAgICAgICAgYCAgcHJldjogXCIke3BhdGNoLmRlYnVnLnByZXZMaW5lfVwiXFxuYCArXG4gICAgICAgICAgICAgIGAgIG5leHQ6IFwiJHtwYXRjaC5kZWJ1Zy5uZXh0TGluZX1cIlxcbmAgK1xuICAgICAgICAgICAgICBgICBjdWxwcml0OiAke2NoYWluLmxlbmd0aCA/IGNoYWluLmpvaW4oJyA8ICcpIDogJyhubyBvd25lciBjaGFpbiBjYXB0dXJlZCknfWAsXG4gICAgICAgICAgICB7IGxldmVsOiAnd2FybicgfSxcbiAgICAgICAgICApXG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICBjb25zdCB0T3B0aW1pemUgPSBwZXJmb3JtYW5jZS5ub3coKVxuICAgIGNvbnN0IG9wdGltaXplZCA9IG9wdGltaXplKGRpZmYpXG4gICAgY29uc3Qgb3B0aW1pemVNcyA9IHBlcmZvcm1hbmNlLm5vdygpIC0gdE9wdGltaXplXG4gICAgY29uc3QgaGFzRGlmZiA9IG9wdGltaXplZC5sZW5ndGggPiAwXG4gICAgaWYgKHRoaXMuYWx0U2NyZWVuQWN0aXZlICYmIGhhc0RpZmYpIHtcbiAgICAgIC8vIFByZXBlbmQgQ1NJIEggdG8gYW5jaG9yIHRoZSBwaHlzaWNhbCBjdXJzb3IgdG8gKDAsMCkgc29cbiAgICAgIC8vIGxvZy11cGRhdGUncyByZWxhdGl2ZSBtb3ZlcyBjb21wdXRlIGZyb20gYSBrbm93biBzcG90IChzZWxmLWhlYWxpbmdcbiAgICAgIC8vIGFnYWluc3Qgb3V0LW9mLWJhbmQgY3Vyc29yIGRyaWZ0LCBzZWUgdGhlIEFMVF9TQ1JFRU5fQU5DSE9SX0NVUlNPUlxuICAgICAgLy8gY29tbWVudCBhYm92ZSkuIEFwcGVuZCBDU0kgcm93OzEgSCB0byBwYXJrIHRoZSBjdXJzb3IgYXQgdGhlIGJvdHRvbVxuICAgICAgLy8gcm93ICh3aGVyZSB0aGUgcHJvbXB0IGlucHV0IGlzKSDigJQgd2l0aG91dCB0aGlzLCB0aGUgY3Vyc29yIGVuZHNcbiAgICAgIC8vIHdoZXJldmVyIHRoZSBsYXN0IGRpZmYgd3JpdGUgbGFuZGVkIChhIGRpZmZlcmVudCByb3cgZXZlcnkgZnJhbWUpLFxuICAgICAgLy8gbWFraW5nIGlUZXJtMidzIGN1cnNvciBndWlkZSBmbGlja2VyIGFzIGl0IGNoYXNlcyB0aGUgY3Vyc29yLlxuICAgICAgLy8gQlNVL0VTVSBwcm90ZWN0cyBjb250ZW50IGF0b21pY2l0eSBidXQgaVRlcm0yJ3MgZ3VpZGUgdHJhY2tzIGN1cnNvclxuICAgICAgLy8gcG9zaXRpb24gaW5kZXBlbmRlbnRseS4gUGFya2luZyBhdCBib3R0b20gKG5vdCAwLDApIGtlZXBzIHRoZSBndWlkZVxuICAgICAgLy8gd2hlcmUgdGhlIHVzZXIncyBhdHRlbnRpb24gaXMuXG4gICAgICAvL1xuICAgICAgLy8gQWZ0ZXIgcmVzaXplLCBwcmVwZW5kIEVSQVNFX1NDUkVFTiB0b28uIFRoZSBkaWZmIG9ubHkgd3JpdGVzIGNlbGxzXG4gICAgICAvLyB0aGF0IGNoYW5nZWQ7IGNlbGxzIHdoZXJlIG5ldz1ibGFuayBhbmQgcHJldi1idWZmZXI9YmxhbmsgZ2V0IHNraXBwZWRcbiAgICAgIC8vIOKAlCBidXQgdGhlIHBoeXNpY2FsIHRlcm1pbmFsIHN0aWxsIGhhcyBzdGFsZSBjb250ZW50IHRoZXJlIChzaG9ydGVyXG4gICAgICAvLyBsaW5lcyBhdCBuZXcgd2lkdGggbGVhdmUgb2xkLXdpZHRoIHRleHQgdGFpbHMgdmlzaWJsZSkuIEVSQVNFIGluc2lkZVxuICAgICAgLy8gQlNVL0VTVSBpcyBhdG9taWM6IG9sZCBjb250ZW50IHN0YXlzIHZpc2libGUgdW50aWwgdGhlIHdob2xlXG4gICAgICAvLyBlcmFzZStwYWludCBsYW5kcywgdGhlbiBzd2FwcyBpbiBvbmUgZ28uIFdyaXRpbmcgRVJBU0VfU0NSRUVOXG4gICAgICAvLyBzeW5jaHJvbm91c2x5IGluIGhhbmRsZVJlc2l6ZSB3b3VsZCBibGFuayB0aGUgc2NyZWVuIGZvciB0aGUgfjgwbXNcbiAgICAgIC8vIHJlbmRlcigpIHRha2VzLlxuICAgICAgaWYgKHRoaXMubmVlZHNFcmFzZUJlZm9yZVBhaW50KSB7XG4gICAgICAgIHRoaXMubmVlZHNFcmFzZUJlZm9yZVBhaW50ID0gZmFsc2VcbiAgICAgICAgb3B0aW1pemVkLnVuc2hpZnQoRVJBU0VfVEhFTl9IT01FX1BBVENIKVxuICAgICAgfSBlbHNlIHtcbiAgICAgICAgb3B0aW1pemVkLnVuc2hpZnQoQ1VSU09SX0hPTUVfUEFUQ0gpXG4gICAgICB9XG4gICAgICBvcHRpbWl6ZWQucHVzaCh0aGlzLmFsdFNjcmVlblBhcmtQYXRjaClcbiAgICB9XG5cbiAgICAvLyBOYXRpdmUgY3Vyc29yIHBvc2l0aW9uaW5nOiBwYXJrIHRoZSB0ZXJtaW5hbCBjdXJzb3IgYXQgdGhlIGRlY2xhcmVkXG4gICAgLy8gcG9zaXRpb24gc28gSU1FIHByZWVkaXQgdGV4dCByZW5kZXJzIGlubGluZSBhbmQgc2NyZWVuIHJlYWRlcnMgL1xuICAgIC8vIG1hZ25pZmllcnMgY2FuIGZvbGxvdyB0aGUgaW5wdXQuIG5vZGVDYWNoZSBob2xkcyB0aGUgYWJzb2x1dGUgc2NyZWVuXG4gICAgLy8gcmVjdCBwb3B1bGF0ZWQgYnkgcmVuZGVyTm9kZVRvT3V0cHV0IHRoaXMgZnJhbWUgKGluY2x1ZGluZyBzY3JvbGxUb3BcbiAgICAvLyB0cmFuc2xhdGlvbikg4oCUIGlmIHRoZSBkZWNsYXJlZCBub2RlIGRpZG4ndCByZW5kZXIgKHN0YWxlIGRlY2xhcmF0aW9uXG4gICAgLy8gYWZ0ZXIgcmVtb3VudCwgb3Igc2Nyb2xsZWQgb3V0IG9mIHZpZXcpLCBpdCB3b24ndCBiZSBpbiB0aGUgY2FjaGVcbiAgICAvLyBhbmQgbm8gbW92ZSBpcyBlbWl0dGVkLlxuICAgIGNvbnN0IGRlY2wgPSB0aGlzLmN1cnNvckRlY2xhcmF0aW9uXG4gICAgY29uc3QgcmVjdCA9IGRlY2wgIT09IG51bGwgPyBub2RlQ2FjaGUuZ2V0KGRlY2wubm9kZSkgOiB1bmRlZmluZWRcbiAgICBjb25zdCB0YXJnZXQgPVxuICAgICAgZGVjbCAhPT0gbnVsbCAmJiByZWN0ICE9PSB1bmRlZmluZWRcbiAgICAgICAgPyB7IHg6IHJlY3QueCArIGRlY2wucmVsYXRpdmVYLCB5OiByZWN0LnkgKyBkZWNsLnJlbGF0aXZlWSB9XG4gICAgICAgIDogbnVsbFxuICAgIGNvbnN0IHBhcmtlZCA9IHRoaXMuZGlzcGxheUN1cnNvclxuXG4gICAgLy8gUHJlc2VydmUgdGhlIGVtcHR5LWRpZmYgemVyby13cml0ZSBmYXN0IHBhdGg6IHNraXAgYWxsIGN1cnNvciB3cml0ZXNcbiAgICAvLyB3aGVuIG5vdGhpbmcgcmVuZGVyZWQgQU5EIHRoZSBwYXJrIHRhcmdldCBpcyB1bmNoYW5nZWQuXG4gICAgY29uc3QgdGFyZ2V0TW92ZWQgPVxuICAgICAgdGFyZ2V0ICE9PSBudWxsICYmXG4gICAgICAocGFya2VkID09PSBudWxsIHx8IHBhcmtlZC54ICE9PSB0YXJnZXQueCB8fCBwYXJrZWQueSAhPT0gdGFyZ2V0LnkpXG4gICAgaWYgKGhhc0RpZmYgfHwgdGFyZ2V0TW92ZWQgfHwgKHRhcmdldCA9PT0gbnVsbCAmJiBwYXJrZWQgIT09IG51bGwpKSB7XG4gICAgICAvLyBNYWluLXNjcmVlbiBwcmVhbWJsZTogbG9nLXVwZGF0ZSdzIHJlbGF0aXZlIG1vdmVzIGFzc3VtZSB0aGVcbiAgICAgIC8vIHBoeXNpY2FsIGN1cnNvciBpcyBhdCBwcmV2RnJhbWUuY3Vyc29yLiBJZiBsYXN0IGZyYW1lIHBhcmtlZCBpdFxuICAgICAgLy8gZWxzZXdoZXJlLCBtb3ZlIGJhY2sgYmVmb3JlIHRoZSBkaWZmIHJ1bnMuIEFsdC1zY3JlZW4ncyBDU0kgSFxuICAgICAgLy8gYWxyZWFkeSByZXNldHMgdG8gKDAsMCkgc28gbm8gcHJlYW1ibGUgbmVlZGVkLlxuICAgICAgaWYgKHBhcmtlZCAhPT0gbnVsbCAmJiAhdGhpcy5hbHRTY3JlZW5BY3RpdmUgJiYgaGFzRGlmZikge1xuICAgICAgICBjb25zdCBwZHggPSBwcmV2RnJhbWUuY3Vyc29yLnggLSBwYXJrZWQueFxuICAgICAgICBjb25zdCBwZHkgPSBwcmV2RnJhbWUuY3Vyc29yLnkgLSBwYXJrZWQueVxuICAgICAgICBpZiAocGR4ICE9PSAwIHx8IHBkeSAhPT0gMCkge1xuICAgICAgICAgIG9wdGltaXplZC51bnNoaWZ0KHsgdHlwZTogJ3N0ZG91dCcsIGNvbnRlbnQ6IGN1cnNvck1vdmUocGR4LCBwZHkpIH0pXG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgaWYgKHRhcmdldCAhPT0gbnVsbCkge1xuICAgICAgICBpZiAodGhpcy5hbHRTY3JlZW5BY3RpdmUpIHtcbiAgICAgICAgICAvLyBBYnNvbHV0ZSBDVVAgKDEtaW5kZXhlZCk7IG5leHQgZnJhbWUncyBDU0kgSCByZXNldHMgcmVnYXJkbGVzcy5cbiAgICAgICAgICAvLyBFbWl0dGVkIGFmdGVyIGFsdFNjcmVlblBhcmtQYXRjaCBzbyB0aGUgZGVjbGFyZWQgcG9zaXRpb24gd2lucy5cbiAgICAgICAgICBjb25zdCByb3cgPSBNYXRoLm1pbihNYXRoLm1heCh0YXJnZXQueSArIDEsIDEpLCB0ZXJtaW5hbFJvd3MpXG4gICAgICAgICAgY29uc3QgY29sID0gTWF0aC5taW4oTWF0aC5tYXgodGFyZ2V0LnggKyAxLCAxKSwgdGVybWluYWxXaWR0aClcbiAgICAgICAgICBvcHRpbWl6ZWQucHVzaCh7IHR5cGU6ICdzdGRvdXQnLCBjb250ZW50OiBjdXJzb3JQb3NpdGlvbihyb3csIGNvbCkgfSlcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAvLyBBZnRlciB0aGUgZGlmZiAob3IgcHJlYW1ibGUpLCBjdXJzb3IgaXMgYXQgZnJhbWUuY3Vyc29yLiBJZiBub1xuICAgICAgICAgIC8vIGRpZmYgQU5EIHByZXZpb3VzbHkgcGFya2VkLCBpdCdzIHN0aWxsIGF0IHRoZSBvbGQgcGFyayBwb3NpdGlvblxuICAgICAgICAgIC8vIChsb2ctdXBkYXRlIHdyb3RlIG5vdGhpbmcpLiBPdGhlcndpc2UgaXQncyBhdCBmcmFtZS5jdXJzb3IuXG4gICAgICAgICAgY29uc3QgZnJvbSA9XG4gICAgICAgICAgICAhaGFzRGlmZiAmJiBwYXJrZWQgIT09IG51bGxcbiAgICAgICAgICAgICAgPyBwYXJrZWRcbiAgICAgICAgICAgICAgOiB7IHg6IGZyYW1lLmN1cnNvci54LCB5OiBmcmFtZS5jdXJzb3IueSB9XG4gICAgICAgICAgY29uc3QgZHggPSB0YXJnZXQueCAtIGZyb20ueFxuICAgICAgICAgIGNvbnN0IGR5ID0gdGFyZ2V0LnkgLSBmcm9tLnlcbiAgICAgICAgICBpZiAoZHggIT09IDAgfHwgZHkgIT09IDApIHtcbiAgICAgICAgICAgIG9wdGltaXplZC5wdXNoKHsgdHlwZTogJ3N0ZG91dCcsIGNvbnRlbnQ6IGN1cnNvck1vdmUoZHgsIGR5KSB9KVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICB0aGlzLmRpc3BsYXlDdXJzb3IgPSB0YXJnZXRcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIC8vIERlY2xhcmF0aW9uIGNsZWFyZWQgKGlucHV0IGJsdXIsIHVubW91bnQpLiBSZXN0b3JlIHBoeXNpY2FsIGN1cnNvclxuICAgICAgICAvLyB0byBmcmFtZS5jdXJzb3IgYmVmb3JlIGZvcmdldHRpbmcgdGhlIHBhcmsgcG9zaXRpb24g4oCUIG90aGVyd2lzZVxuICAgICAgICAvLyBkaXNwbGF5Q3Vyc29yPW51bGwgbGllcyBhYm91dCB3aGVyZSB0aGUgY3Vyc29yIGlzLCBhbmQgdGhlIE5FWFRcbiAgICAgICAgLy8gZnJhbWUncyBwcmVhbWJsZSAob3IgbG9nLXVwZGF0ZSdzIHJlbGF0aXZlIG1vdmVzKSBjb21wdXRlcyBmcm9tIGFcbiAgICAgICAgLy8gd3Jvbmcgc3BvdC4gVGhlIHByZWFtYmxlIGFib3ZlIGhhbmRsZXMgaGFzRGlmZjsgdGhpcyBoYW5kbGVzXG4gICAgICAgIC8vICFoYXNEaWZmIChlLmcuIGFjY2Vzc2liaWxpdHkgbW9kZSB3aGVyZSBibHVyIGRvZXNuJ3QgY2hhbmdlXG4gICAgICAgIC8vIHJlbmRlcmVkVmFsdWUgc2luY2UgaW52ZXJ0IGlzIGlkZW50aXR5KS5cbiAgICAgICAgaWYgKHBhcmtlZCAhPT0gbnVsbCAmJiAhdGhpcy5hbHRTY3JlZW5BY3RpdmUgJiYgIWhhc0RpZmYpIHtcbiAgICAgICAgICBjb25zdCByZHggPSBmcmFtZS5jdXJzb3IueCAtIHBhcmtlZC54XG4gICAgICAgICAgY29uc3QgcmR5ID0gZnJhbWUuY3Vyc29yLnkgLSBwYXJrZWQueVxuICAgICAgICAgIGlmIChyZHggIT09IDAgfHwgcmR5ICE9PSAwKSB7XG4gICAgICAgICAgICBvcHRpbWl6ZWQucHVzaCh7IHR5cGU6ICdzdGRvdXQnLCBjb250ZW50OiBjdXJzb3JNb3ZlKHJkeCwgcmR5KSB9KVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICB0aGlzLmRpc3BsYXlDdXJzb3IgPSBudWxsXG4gICAgICB9XG4gICAgfVxuXG4gICAgY29uc3QgdFdyaXRlID0gcGVyZm9ybWFuY2Uubm93KClcbiAgICB3cml0ZURpZmZUb1Rlcm1pbmFsKFxuICAgICAgdGhpcy50ZXJtaW5hbCxcbiAgICAgIG9wdGltaXplZCxcbiAgICAgIHRoaXMuYWx0U2NyZWVuQWN0aXZlICYmICFTWU5DX09VVFBVVF9TVVBQT1JURUQsXG4gICAgKVxuICAgIGNvbnN0IHdyaXRlTXMgPSBwZXJmb3JtYW5jZS5ub3coKSAtIHRXcml0ZVxuXG4gICAgLy8gVXBkYXRlIGJsaXQgc2FmZXR5IGZvciB0aGUgTkVYVCBmcmFtZS4gVGhlIGZyYW1lIGp1c3QgcmVuZGVyZWRcbiAgICAvLyBiZWNvbWVzIGZyb250RnJhbWUgKD0gbmV4dCBmcmFtZSdzIHByZXZTY3JlZW4pLiBJZiB3ZSBhcHBsaWVkIHRoZVxuICAgIC8vIHNlbGVjdGlvbiBvdmVybGF5LCB0aGF0IGJ1ZmZlciBoYXMgaW52ZXJ0ZWQgY2VsbHMuIHNlbEFjdGl2ZS9obEFjdGl2ZVxuICAgIC8vIGFyZSBvbmx5IGV2ZXIgdHJ1ZSBpbiBhbHQtc2NyZWVuOyBpbiBtYWluLXNjcmVlbiB0aGlzIGlzIGZhbHNl4oaSZmFsc2UuXG4gICAgdGhpcy5wcmV2RnJhbWVDb250YW1pbmF0ZWQgPSBzZWxBY3RpdmUgfHwgaGxBY3RpdmVcblxuICAgIC8vIEEgU2Nyb2xsQm94IGhhcyBwZW5kaW5nU2Nyb2xsRGVsdGEgbGVmdCB0byBkcmFpbiDigJQgc2NoZWR1bGUgdGhlIG5leHRcbiAgICAvLyBmcmFtZS4gTVVTVCBOT1QgY2FsbCB0aGlzLnNjaGVkdWxlUmVuZGVyKCkgaGVyZTogd2UncmUgaW5zaWRlIGFcbiAgICAvLyB0cmFpbGluZy1lZGdlIHRocm90dGxlIGludm9jYXRpb24sIHRpbWVySWQgaXMgdW5kZWZpbmVkLCBhbmQgbG9kYXNoJ3NcbiAgICAvLyBkZWJvdW5jZSBzZWVzIHRpbWVTaW5jZUxhc3RDYWxsID49IHdhaXQgKGxhc3QgY2FsbCB3YXMgYXQgdGhlIHN0YXJ0XG4gICAgLy8gb2YgdGhpcyB3aW5kb3cpIOKGkiBsZWFkaW5nRWRnZSBmaXJlcyBJTU1FRElBVEVMWSDihpIgZG91YmxlIHJlbmRlciB+MC4xbXNcbiAgICAvLyBhcGFydCDihpIgamFuay4gVXNlIGEgcGxhaW4gdGltZW91dC4gSWYgYSB3aGVlbCBldmVudCBhcnJpdmVzIGZpcnN0LFxuICAgIC8vIGl0cyBzY2hlZHVsZVJlbmRlciBwYXRoIGZpcmVzIGEgcmVuZGVyIHdoaWNoIGNsZWFycyB0aGlzIHRpbWVyIGF0XG4gICAgLy8gdGhlIHRvcCBvZiBvblJlbmRlciDigJQgbm8gZG91YmxlLlxuICAgIC8vXG4gICAgLy8gRHJhaW4gZnJhbWVzIGFyZSBjaGVhcCAoREVDU1RCTSArIH4xMCBwYXRjaGVzLCB+MjAwIGJ5dGVzKSBzbyBydW4gYXRcbiAgICAvLyBxdWFydGVyIGludGVydmFsICh+MjUwZnBzLCBzZXRUaW1lb3V0IHByYWN0aWNhbCBmbG9vcikgZm9yIG1heCBzY3JvbGxcbiAgICAvLyBzcGVlZC4gUmVndWxhciByZW5kZXJzIHN0YXkgYXQgRlJBTUVfSU5URVJWQUxfTVMgdmlhIHRoZSB0aHJvdHRsZS5cbiAgICBpZiAoZnJhbWUuc2Nyb2xsRHJhaW5QZW5kaW5nKSB7XG4gICAgICB0aGlzLmRyYWluVGltZXIgPSBzZXRUaW1lb3V0KFxuICAgICAgICAoKSA9PiB0aGlzLm9uUmVuZGVyKCksXG4gICAgICAgIEZSQU1FX0lOVEVSVkFMX01TID4+IDIsXG4gICAgICApXG4gICAgfVxuXG4gICAgY29uc3QgeW9nYU1zID0gZ2V0TGFzdFlvZ2FNcygpXG4gICAgY29uc3QgY29tbWl0TXMgPSBnZXRMYXN0Q29tbWl0TXMoKVxuICAgIGNvbnN0IHljID0gdGhpcy5sYXN0WW9nYUNvdW50ZXJzXG4gICAgLy8gUmVzZXQgc28gZHJhaW4tb25seSBmcmFtZXMgKG5vIFJlYWN0IGNvbW1pdCkgZG9uJ3QgcmVwZWF0IHN0YWxlIHZhbHVlcy5cbiAgICByZXNldFByb2ZpbGVDb3VudGVycygpXG4gICAgdGhpcy5sYXN0WW9nYUNvdW50ZXJzID0ge1xuICAgICAgbXM6IDAsXG4gICAgICB2aXNpdGVkOiAwLFxuICAgICAgbWVhc3VyZWQ6IDAsXG4gICAgICBjYWNoZUhpdHM6IDAsXG4gICAgICBsaXZlOiAwLFxuICAgIH1cbiAgICB0aGlzLm9wdGlvbnMub25GcmFtZT8uKHtcbiAgICAgIGR1cmF0aW9uTXM6IHBlcmZvcm1hbmNlLm5vdygpIC0gcmVuZGVyU3RhcnQsXG4gICAgICBwaGFzZXM6IHtcbiAgICAgICAgcmVuZGVyZXI6IHJlbmRlcmVyTXMsXG4gICAgICAgIGRpZmY6IGRpZmZNcyxcbiAgICAgICAgb3B0aW1pemU6IG9wdGltaXplTXMsXG4gICAgICAgIHdyaXRlOiB3cml0ZU1zLFxuICAgICAgICBwYXRjaGVzOiBkaWZmLmxlbmd0aCxcbiAgICAgICAgeW9nYTogeW9nYU1zLFxuICAgICAgICBjb21taXQ6IGNvbW1pdE1zLFxuICAgICAgICB5b2dhVmlzaXRlZDogeWMudmlzaXRlZCxcbiAgICAgICAgeW9nYU1lYXN1cmVkOiB5Yy5tZWFzdXJlZCxcbiAgICAgICAgeW9nYUNhY2hlSGl0czogeWMuY2FjaGVIaXRzLFxuICAgICAgICB5b2dhTGl2ZTogeWMubGl2ZSxcbiAgICAgIH0sXG4gICAgICBmbGlja2VycyxcbiAgICB9KVxuICB9XG5cbiAgcGF1c2UoKTogdm9pZCB7XG4gICAgLy8gRmx1c2ggcGVuZGluZyBSZWFjdCB1cGRhdGVzIGFuZCByZW5kZXIgYmVmb3JlIHBhdXNpbmcuXG4gICAgLy8gQHRzLWV4cGVjdC1lcnJvciBmbHVzaFN5bmNGcm9tUmVjb25jaWxlciBleGlzdHMgaW4gcmVhY3QtcmVjb25jaWxlciAwLjMxIGJ1dCBub3QgaW4gQHR5cGVzL3JlYWN0LXJlY29uY2lsZXJcbiAgICByZWNvbmNpbGVyLmZsdXNoU3luY0Zyb21SZWNvbmNpbGVyKClcbiAgICB0aGlzLm9uUmVuZGVyKClcblxuICAgIHRoaXMuaXNQYXVzZWQgPSB0cnVlXG4gIH1cblxuICByZXN1bWUoKTogdm9pZCB7XG4gICAgdGhpcy5pc1BhdXNlZCA9IGZhbHNlXG4gICAgdGhpcy5vblJlbmRlcigpXG4gIH1cblxuICAvKipcbiAgICogUmVzZXQgZnJhbWUgYnVmZmVycyBzbyB0aGUgbmV4dCByZW5kZXIgd3JpdGVzIHRoZSBmdWxsIHNjcmVlbiBmcm9tIHNjcmF0Y2guXG4gICAqIENhbGwgdGhpcyBiZWZvcmUgcmVzdW1lKCkgd2hlbiB0aGUgdGVybWluYWwgY29udGVudCBoYXMgYmVlbiBjb3JydXB0ZWQgYnlcbiAgICogYW4gZXh0ZXJuYWwgcHJvY2VzcyAoZS5nLiB0bXV4LCBzaGVsbCwgZnVsbC1zY3JlZW4gVFVJKS5cbiAgICovXG4gIHJlcGFpbnQoKTogdm9pZCB7XG4gICAgdGhpcy5mcm9udEZyYW1lID0gZW1wdHlGcmFtZShcbiAgICAgIHRoaXMuZnJvbnRGcmFtZS52aWV3cG9ydC5oZWlnaHQsXG4gICAgICB0aGlzLmZyb250RnJhbWUudmlld3BvcnQud2lkdGgsXG4gICAgICB0aGlzLnN0eWxlUG9vbCxcbiAgICAgIHRoaXMuY2hhclBvb2wsXG4gICAgICB0aGlzLmh5cGVybGlua1Bvb2wsXG4gICAgKVxuICAgIHRoaXMuYmFja0ZyYW1lID0gZW1wdHlGcmFtZShcbiAgICAgIHRoaXMuYmFja0ZyYW1lLnZpZXdwb3J0LmhlaWdodCxcbiAgICAgIHRoaXMuYmFja0ZyYW1lLnZpZXdwb3J0LndpZHRoLFxuICAgICAgdGhpcy5zdHlsZVBvb2wsXG4gICAgICB0aGlzLmNoYXJQb29sLFxuICAgICAgdGhpcy5oeXBlcmxpbmtQb29sLFxuICAgIClcbiAgICB0aGlzLmxvZy5yZXNldCgpXG4gICAgLy8gUGh5c2ljYWwgY3Vyc29yIHBvc2l0aW9uIGlzIHVua25vd24gYWZ0ZXIgZXh0ZXJuYWwgdGVybWluYWwgY29ycnVwdGlvbi5cbiAgICAvLyBDbGVhciBkaXNwbGF5Q3Vyc29yIHNvIHRoZSBjdXJzb3IgcHJlYW1ibGUgZG9lc24ndCBlbWl0IGEgc3RhbGVcbiAgICAvLyByZWxhdGl2ZSBtb3ZlIGZyb20gd2hlcmUgd2UgbGFzdCBwYXJrZWQgaXQuXG4gICAgdGhpcy5kaXNwbGF5Q3Vyc29yID0gbnVsbFxuICB9XG5cbiAgLyoqXG4gICAqIENsZWFyIHRoZSBwaHlzaWNhbCB0ZXJtaW5hbCBhbmQgZm9yY2UgYSBmdWxsIHJlZHJhdy5cbiAgICpcbiAgICogVGhlIHRyYWRpdGlvbmFsIHJlYWRsaW5lIGN0cmwrbCDigJQgY2xlYXJzIHRoZSB2aXNpYmxlIHNjcmVlbiBhbmRcbiAgICogcmVkcmF3cyB0aGUgY3VycmVudCBjb250ZW50LiBBbHNvIHRoZSByZWNvdmVyeSBwYXRoIHdoZW4gdGhlIHRlcm1pbmFsXG4gICAqIHdhcyBjbGVhcmVkIGV4dGVybmFsbHkgKG1hY09TIENtZCtLKSBhbmQgSW5rJ3MgZGlmZiBlbmdpbmUgdGhpbmtzXG4gICAqIHVuY2hhbmdlZCBjZWxscyBkb24ndCBuZWVkIHJlcGFpbnRpbmcuIFNjcm9sbGJhY2sgaXMgcHJlc2VydmVkLlxuICAgKi9cbiAgZm9yY2VSZWRyYXcoKTogdm9pZCB7XG4gICAgaWYgKCF0aGlzLm9wdGlvbnMuc3Rkb3V0LmlzVFRZIHx8IHRoaXMuaXNVbm1vdW50ZWQgfHwgdGhpcy5pc1BhdXNlZCkgcmV0dXJuXG4gICAgdGhpcy5vcHRpb25zLnN0ZG91dC53cml0ZShFUkFTRV9TQ1JFRU4gKyBDVVJTT1JfSE9NRSlcbiAgICBpZiAodGhpcy5hbHRTY3JlZW5BY3RpdmUpIHtcbiAgICAgIHRoaXMucmVzZXRGcmFtZXNGb3JBbHRTY3JlZW4oKVxuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLnJlcGFpbnQoKVxuICAgICAgLy8gcmVwYWludCgpIHJlc2V0cyBmcm9udEZyYW1lIHRvIDDDlzAuIFdpdGhvdXQgdGhpcyBmbGFnIHRoZSBuZXh0XG4gICAgICAvLyBmcmFtZSdzIGJsaXQgb3B0aW1pemF0aW9uIGNvcGllcyBmcm9tIHRoYXQgZW1wdHkgc2NyZWVuIGFuZCB0aGVcbiAgICAgIC8vIGRpZmYgc2VlcyBubyBjb250ZW50LiBvblJlbmRlciByZXNldHMgdGhlIGZsYWcgYXQgZnJhbWUgZW5kLlxuICAgICAgdGhpcy5wcmV2RnJhbWVDb250YW1pbmF0ZWQgPSB0cnVlXG4gICAgfVxuICAgIHRoaXMub25SZW5kZXIoKVxuICB9XG5cbiAgLyoqXG4gICAqIE1hcmsgdGhlIHByZXZpb3VzIGZyYW1lIGFzIHVudHJ1c3R3b3J0aHkgZm9yIGJsaXQsIGZvcmNpbmcgdGhlIG5leHRcbiAgICogcmVuZGVyIHRvIGRvIGEgZnVsbC1kYW1hZ2UgZGlmZiBpbnN0ZWFkIG9mIHRoZSBwZXItbm9kZSBmYXN0IHBhdGguXG4gICAqXG4gICAqIExpZ2h0ZXIgdGhhbiBmb3JjZVJlZHJhdygpIOKAlCBubyBzY3JlZW4gY2xlYXIsIG5vIGV4dHJhIHdyaXRlLiBDYWxsXG4gICAqIGZyb20gYSB1c2VMYXlvdXRFZmZlY3QgY2xlYW51cCB3aGVuIHVubW91bnRpbmcgYSB0YWxsIG92ZXJsYXk6IHRoZVxuICAgKiBibGl0IGZhc3QgcGF0aCBjYW4gY29weSBzdGFsZSBjZWxscyBmcm9tIHRoZSBvdmVybGF5IGZyYW1lIGludG8gcm93c1xuICAgKiB0aGUgc2hydW5rZW4gbGF5b3V0IG5vIGxvbmdlciByZWFjaGVzLCBsZWF2aW5nIGEgZ2hvc3QgdGl0bGUvZGl2aWRlci5cbiAgICogb25SZW5kZXIgcmVzZXRzIHRoZSBmbGFnIGF0IGZyYW1lIGVuZCBzbyBpdCdzIG9uZS1zaG90LlxuICAgKi9cbiAgaW52YWxpZGF0ZVByZXZGcmFtZSgpOiB2b2lkIHtcbiAgICB0aGlzLnByZXZGcmFtZUNvbnRhbWluYXRlZCA9IHRydWVcbiAgfVxuXG4gIC8qKlxuICAgKiBDYWxsZWQgYnkgdGhlIDxBbHRlcm5hdGVTY3JlZW4+IGNvbXBvbmVudCBvbiBtb3VudC91bm1vdW50LlxuICAgKiBDb250cm9scyBjdXJzb3IueSBjbGFtcGluZyBpbiB0aGUgcmVuZGVyZXIgYW5kIGdhdGVzIGFsdC1zY3JlZW4tYXdhcmVcbiAgICogYmVoYXZpb3IgaW4gU0lHQ09OVC9yZXNpemUvdW5tb3VudCBoYW5kbGVycy4gUmVwYWludHMgb24gY2hhbmdlIHNvXG4gICAqIHRoZSBmaXJzdCBhbHQtc2NyZWVuIGZyYW1lIChhbmQgZmlyc3QgbWFpbi1zY3JlZW4gZnJhbWUgb24gZXhpdCkgaXNcbiAgICogYSBmdWxsIHJlZHJhdyB3aXRoIG5vIHN0YWxlIGRpZmYgc3RhdGUuXG4gICAqL1xuICBzZXRBbHRTY3JlZW5BY3RpdmUoYWN0aXZlOiBib29sZWFuLCBtb3VzZVRyYWNraW5nID0gZmFsc2UpOiB2b2lkIHtcbiAgICBpZiAodGhpcy5hbHRTY3JlZW5BY3RpdmUgPT09IGFjdGl2ZSkgcmV0dXJuXG4gICAgdGhpcy5hbHRTY3JlZW5BY3RpdmUgPSBhY3RpdmVcbiAgICB0aGlzLmFsdFNjcmVlbk1vdXNlVHJhY2tpbmcgPSBhY3RpdmUgJiYgbW91c2VUcmFja2luZ1xuICAgIGlmIChhY3RpdmUpIHtcbiAgICAgIHRoaXMucmVzZXRGcmFtZXNGb3JBbHRTY3JlZW4oKVxuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLnJlcGFpbnQoKVxuICAgIH1cbiAgfVxuXG4gIGdldCBpc0FsdFNjcmVlbkFjdGl2ZSgpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdGhpcy5hbHRTY3JlZW5BY3RpdmVcbiAgfVxuXG4gIC8qKlxuICAgKiBSZS1hc3NlcnQgdGVybWluYWwgbW9kZXMgYWZ0ZXIgYSBnYXAgKD41cyBzdGRpbiBzaWxlbmNlIG9yIGV2ZW50LWxvb3BcbiAgICogc3RhbGwpLiBDYXRjaGVzIHRtdXggZGV0YWNo4oaSYXR0YWNoLCBzc2ggcmVjb25uZWN0LCBhbmQgbGFwdG9wXG4gICAqIHNsZWVwL3dha2Ug4oCUIG5vbmUgb2Ygd2hpY2ggc2VuZCBTSUdDT05ULiBUaGUgdGVybWluYWwgbWF5IHJlc2V0IERFQ1xuICAgKiBwcml2YXRlIG1vZGVzIG9uIHJlY29ubmVjdDsgdGhpcyBtZXRob2QgcmVzdG9yZXMgdGhlbS5cbiAgICpcbiAgICogQWx3YXlzIHJlLWFzc2VydHMgZXh0ZW5kZWQga2V5IHJlcG9ydGluZyBhbmQgbW91c2UgdHJhY2tpbmcuIE1vdXNlXG4gICAqIHRyYWNraW5nIGlzIGlkZW1wb3RlbnQgKERFQyBwcml2YXRlIG1vZGUgc2V0LXdoZW4tc2V0IGlzIGEgbm8tb3ApLiBUaGVcbiAgICogS2l0dHkga2V5Ym9hcmQgcHJvdG9jb2wgaXMgTk9UIOKAlCBDU0kgPjF1IGlzIGEgc3RhY2sgcHVzaCwgc28gd2UgcG9wXG4gICAqIGZpcnN0IHRvIGtlZXAgZGVwdGggYmFsYW5jZWQgKHBvcCBvbiBlbXB0eSBzdGFjayBpcyBhIG5vLW9wIHBlciBzcGVjLFxuICAgKiBzbyBhZnRlciBhIHRlcm1pbmFsIHJlc2V0IHRoaXMgc3RpbGwgcmVzdG9yZXMgZGVwdGggMOKGkjEpLiBXaXRob3V0IHRoZVxuICAgKiBwb3AsIGVhY2ggPjVzIGlkbGUgZ2FwIGFkZHMgYSBzdGFjayBlbnRyeSwgYW5kIHRoZSBzaW5nbGUgcG9wIG9uIGV4aXRcbiAgICogb3Igc3VzcGVuZCBjYW4ndCBkcmFpbiB0aGVtIOKAlCB0aGUgc2hlbGwgaXMgbGVmdCBpbiBDU0kgdSBtb2RlIHdoZXJlXG4gICAqIEN0cmwrQy9DdHJsK0QgbGVhayBhcyBlc2NhcGUgc2VxdWVuY2VzLiBUaGUgYWx0LXNjcmVlblxuICAgKiByZS1lbnRyeSAoRVJBU0VfU0NSRUVOICsgZnJhbWUgcmVzZXQpIGlzIE5PVCBpZGVtcG90ZW50IOKAlCBpdCBibGFua3MgdGhlXG4gICAqIHNjcmVlbiDigJQgc28gaXQncyBvcHQtaW4gdmlhIGluY2x1ZGVBbHRTY3JlZW4uIFRoZSBzdGRpbi1nYXAgY2FsbGVyIGZpcmVzXG4gICAqIG9uIG9yZGluYXJ5ID41cyBpZGxlICsga2V5cHJlc3MgYW5kIG11c3Qgbm90IGVyYXNlOyB0aGUgZXZlbnQtbG9vcCBzdGFsbFxuICAgKiBkZXRlY3RvciBmaXJlcyBvbiBnZW51aW5lIHNsZWVwL3dha2UgYW5kIG9wdHMgaW4uIHRtdXggYXR0YWNoIC8gc3NoXG4gICAqIHJlY29ubmVjdCB0eXBpY2FsbHkgc2VuZCBhIHJlc2l6ZSwgd2hpY2ggYWxyZWFkeSBjb3ZlcnMgYWx0LXNjcmVlbiB2aWFcbiAgICogaGFuZGxlUmVzaXplLlxuICAgKi9cbiAgcmVhc3NlcnRUZXJtaW5hbE1vZGVzID0gKGluY2x1ZGVBbHRTY3JlZW4gPSBmYWxzZSk6IHZvaWQgPT4ge1xuICAgIGlmICghdGhpcy5vcHRpb25zLnN0ZG91dC5pc1RUWSkgcmV0dXJuXG4gICAgLy8gRG9uJ3QgdG91Y2ggdGhlIHRlcm1pbmFsIGR1cmluZyBhbiBlZGl0b3IgaGFuZG9mZiDigJQgcmUtZW5hYmxpbmcga2l0dHlcbiAgICAvLyBrZXlib2FyZCBoZXJlIHdvdWxkIHVuZG8gZW50ZXJBbHRlcm5hdGVTY3JlZW4ncyBkaXNhYmxlIGFuZCBuYW5vIHdvdWxkXG4gICAgLy8gc3RhcnQgc2VlaW5nIENTSS11IHNlcXVlbmNlcyBhZ2Fpbi5cbiAgICBpZiAodGhpcy5pc1BhdXNlZCkgcmV0dXJuXG4gICAgLy8gRXh0ZW5kZWQga2V5cyDigJQgcmUtYXNzZXJ0IGlmIGVuYWJsZWQgKEFwcC50c3ggZW5hYmxlcyB0aGVzZSBvblxuICAgIC8vIGFsbG93bGlzdGVkIHRlcm1pbmFscyBhdCByYXctbW9kZSBlbnRyeTsgYSB0ZXJtaW5hbCByZXNldCBjbGVhcnMgdGhlbSkuXG4gICAgLy8gUG9wLWJlZm9yZS1wdXNoIGtlZXBzIEtpdHR5IHN0YWNrIGRlcHRoIGF0IDEgaW5zdGVhZCBvZiBhY2N1bXVsYXRpbmdcbiAgICAvLyBvbiBlYWNoIGNhbGwuXG4gICAgaWYgKHN1cHBvcnRzRXh0ZW5kZWRLZXlzKCkpIHtcbiAgICAgIHRoaXMub3B0aW9ucy5zdGRvdXQud3JpdGUoXG4gICAgICAgIERJU0FCTEVfS0lUVFlfS0VZQk9BUkQgK1xuICAgICAgICAgIEVOQUJMRV9LSVRUWV9LRVlCT0FSRCArXG4gICAgICAgICAgRU5BQkxFX01PRElGWV9PVEhFUl9LRVlTLFxuICAgICAgKVxuICAgIH1cbiAgICBpZiAoIXRoaXMuYWx0U2NyZWVuQWN0aXZlKSByZXR1cm5cbiAgICAvLyBNb3VzZSB0cmFja2luZyDigJQgaWRlbXBvdGVudCwgc2FmZSB0byByZS1hc3NlcnQgb24gZXZlcnkgc3RkaW4gZ2FwLlxuICAgIGlmICh0aGlzLmFsdFNjcmVlbk1vdXNlVHJhY2tpbmcpIHtcbiAgICAgIHRoaXMub3B0aW9ucy5zdGRvdXQud3JpdGUoRU5BQkxFX01PVVNFX1RSQUNLSU5HKVxuICAgIH1cbiAgICAvLyBBbHQtc2NyZWVuIHJlLWVudHJ5IOKAlCBkZXN0cnVjdGl2ZSAoRVJBU0VfU0NSRUVOKS4gT25seSBmb3IgY2FsbGVycyB0aGF0XG4gICAgLy8gaGF2ZSBhIHN0cm9uZyBzaWduYWwgdGhlIHRlcm1pbmFsIGFjdHVhbGx5IGRyb3BwZWQgbW9kZSAxMDQ5LlxuICAgIGlmIChpbmNsdWRlQWx0U2NyZWVuKSB7XG4gICAgICB0aGlzLnJlZW50ZXJBbHRTY3JlZW4oKVxuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBNYXJrIHRoaXMgaW5zdGFuY2UgYXMgdW5tb3VudGVkIHNvIGZ1dHVyZSB1bm1vdW50KCkgY2FsbHMgZWFybHktcmV0dXJuLlxuICAgKiBDYWxsZWQgYnkgZ3JhY2VmdWxTaHV0ZG93bidzIGNsZWFudXBUZXJtaW5hbE1vZGVzKCkgYWZ0ZXIgaXQgaGFzIHNlbnRcbiAgICogRVhJVF9BTFRfU0NSRUVOIGJ1dCBiZWZvcmUgdGhlIHJlbWFpbmluZyB0ZXJtaW5hbC1yZXNldCBzZXF1ZW5jZXMuXG4gICAqIFdpdGhvdXQgdGhpcywgc2lnbmFsLWV4aXQncyBkZWZlcnJlZCBpbmsudW5tb3VudCgpICh0cmlnZ2VyZWQgYnlcbiAgICogcHJvY2Vzcy5leGl0KCkpIHJ1bnMgdGhlIGZ1bGwgdW5tb3VudCBwYXRoOiBvblJlbmRlcigpICsgd3JpdGVTeW5jXG4gICAqIGNsZWFudXAgYmxvY2sgKyB1cGRhdGVDb250YWluZXJTeW5jIOKGkiBBbHRlcm5hdGVTY3JlZW4gdW5tb3VudCBjbGVhbnVwLlxuICAgKiBUaGUgcmVzdWx0IGlzIDItMyByZWR1bmRhbnQgRVhJVF9BTFRfU0NSRUVOIHNlcXVlbmNlcyBsYW5kaW5nIG9uIHRoZVxuICAgKiBtYWluIHNjcmVlbiBBRlRFUiBwcmludFJlc3VtZUhpbnQoKSwgd2hpY2ggdG11eCAoYXQgbGVhc3QpIGludGVycHJldHNcbiAgICogYXMgcmVzdG9yaW5nIHRoZSBzYXZlZCBjdXJzb3IgcG9zaXRpb24g4oCUIGNsb2JiZXJpbmcgdGhlIHJlc3VtZSBoaW50LlxuICAgKi9cbiAgZGV0YWNoRm9yU2h1dGRvd24oKTogdm9pZCB7XG4gICAgdGhpcy5pc1VubW91bnRlZCA9IHRydWVcbiAgICAvLyBDYW5jZWwgYW55IHBlbmRpbmcgdGhyb3R0bGVkIHJlbmRlciBzbyBpdCBkb2Vzbid0IGZpcmUgYmV0d2VlblxuICAgIC8vIGNsZWFudXBUZXJtaW5hbE1vZGVzKCkgYW5kIHByb2Nlc3MuZXhpdCgpIGFuZCB3cml0ZSB0byBtYWluIHNjcmVlbi5cbiAgICB0aGlzLnNjaGVkdWxlUmVuZGVyLmNhbmNlbD8uKClcbiAgICAvLyBSZXN0b3JlIHN0ZGluIGZyb20gcmF3IG1vZGUuIHVubW91bnQoKSB1c2VkIHRvIGRvIHRoaXMgdmlhIFJlYWN0XG4gICAgLy8gdW5tb3VudCAoQXBwLmNvbXBvbmVudFdpbGxVbm1vdW50IOKGkiBoYW5kbGVTZXRSYXdNb2RlKGZhbHNlKSkgYnV0IHdlJ3JlXG4gICAgLy8gc2hvcnQtY2lyY3VpdGluZyB0aGF0IHBhdGguIE11c3QgdXNlIHRoaXMub3B0aW9ucy5zdGRpbiDigJQgTk9UXG4gICAgLy8gcHJvY2Vzcy5zdGRpbiDigJQgYmVjYXVzZSBnZXRTdGRpbk92ZXJyaWRlKCkgbWF5IGhhdmUgb3BlbmVkIC9kZXYvdHR5XG4gICAgLy8gd2hlbiBzdGRpbiBpcyBwaXBlZC5cbiAgICBjb25zdCBzdGRpbiA9IHRoaXMub3B0aW9ucy5zdGRpbiBhcyBOb2RlSlMuUmVhZFN0cmVhbSAmIHtcbiAgICAgIGlzUmF3PzogYm9vbGVhblxuICAgICAgc2V0UmF3TW9kZT86IChtOiBib29sZWFuKSA9PiB2b2lkXG4gICAgfVxuICAgIHRoaXMuZHJhaW5TdGRpbigpXG4gICAgaWYgKHN0ZGluLmlzVFRZICYmIHN0ZGluLmlzUmF3ICYmIHN0ZGluLnNldFJhd01vZGUpIHtcbiAgICAgIHN0ZGluLnNldFJhd01vZGUoZmFsc2UpXG4gICAgfVxuICB9XG5cbiAgLyoqIEBzZWUgZHJhaW5TdGRpbiAqL1xuICBkcmFpblN0ZGluKCk6IHZvaWQge1xuICAgIGRyYWluU3RkaW4odGhpcy5vcHRpb25zLnN0ZGluKVxuICB9XG5cbiAgLyoqXG4gICAqIFJlLWVudGVyIGFsdC1zY3JlZW4sIGNsZWFyLCBob21lLCByZS1lbmFibGUgbW91c2UgdHJhY2tpbmcsIGFuZCByZXNldFxuICAgKiBmcmFtZSBidWZmZXJzIHNvIHRoZSBuZXh0IHJlbmRlciByZXBhaW50cyBmcm9tIHNjcmF0Y2guIFNlbGYtaGVhbCBmb3JcbiAgICogU0lHQ09OVCwgcmVzaXplLCBhbmQgc3RkaW4tZ2FwL2V2ZW50LWxvb3Atc3RhbGwgKHNsZWVwL3dha2UpIOKAlCBhbnkgb2ZcbiAgICogd2hpY2ggY2FuIGxlYXZlIHRoZSB0ZXJtaW5hbCBpbiBtYWluLXNjcmVlbiBtb2RlIHdoaWxlIGFsdFNjcmVlbkFjdGl2ZVxuICAgKiBzdGF5cyB0cnVlLiBFTlRFUl9BTFRfU0NSRUVOIGlzIGEgdGVybWluYWwtc2lkZSBuby1vcCBpZiBhbHJlYWR5IGluIGFsdC5cbiAgICovXG4gIHByaXZhdGUgcmVlbnRlckFsdFNjcmVlbigpOiB2b2lkIHtcbiAgICB0aGlzLm9wdGlvbnMuc3Rkb3V0LndyaXRlKFxuICAgICAgRU5URVJfQUxUX1NDUkVFTiArXG4gICAgICAgIEVSQVNFX1NDUkVFTiArXG4gICAgICAgIENVUlNPUl9IT01FICtcbiAgICAgICAgKHRoaXMuYWx0U2NyZWVuTW91c2VUcmFja2luZyA/IEVOQUJMRV9NT1VTRV9UUkFDS0lORyA6ICcnKSxcbiAgICApXG4gICAgdGhpcy5yZXNldEZyYW1lc0ZvckFsdFNjcmVlbigpXG4gIH1cblxuICAvKipcbiAgICogU2VlZCBwcmV2L2JhY2sgZnJhbWVzIHdpdGggZnVsbC1zaXplIEJMQU5LIHNjcmVlbnMgKHJvd3PDl2NvbHMgb2YgZW1wdHlcbiAgICogY2VsbHMsIG5vdCAww5cwKS4gSW4gYWx0LXNjcmVlbiBtb2RlLCBuZXh0LnNjcmVlbi5oZWlnaHQgaXMgYWx3YXlzXG4gICAqIHRlcm1pbmFsUm93czsgaWYgcHJldi5zY3JlZW4uaGVpZ2h0IGlzIDAgKGVtcHR5RnJhbWUncyBkZWZhdWx0KSxcbiAgICogbG9nLXVwZGF0ZSBzZWVzIGhlaWdodERlbHRhID4gMCAoJ2dyb3dpbmcnKSBhbmQgY2FsbHMgcmVuZGVyRnJhbWVTbGljZSxcbiAgICogd2hvc2UgdHJhaWxpbmcgcGVyLXJvdyBDUitMRiBhdCB0aGUgbGFzdCByb3cgc2Nyb2xscyB0aGUgYWx0IHNjcmVlbixcbiAgICogcGVybWFuZW50bHkgZGVzeW5jaW5nIHRoZSB2aXJ0dWFsIGFuZCBwaHlzaWNhbCBjdXJzb3JzIGJ5IDEgcm93LlxuICAgKlxuICAgKiBXaXRoIGEgcm93c8OXY29scyBibGFuayBwcmV2LCBoZWlnaHREZWx0YSA9PT0gMCDihpIgc3RhbmRhcmQgZGlmZkVhY2hcbiAgICog4oaSIG1vdmVDdXJzb3JUbyAoQ1NJIGN1cnNvck1vdmUsIG5vIExGLCBubyBzY3JvbGwpLlxuICAgKlxuICAgKiB2aWV3cG9ydC5oZWlnaHQgPSByb3dzICsgMSBtYXRjaGVzIHRoZSByZW5kZXJlcidzIGFsdC1zY3JlZW4gb3V0cHV0LFxuICAgKiBwcmV2ZW50aW5nIGEgc3B1cmlvdXMgcmVzaXplIHRyaWdnZXIgb24gdGhlIGZpcnN0IGZyYW1lLiBjdXJzb3IueSA9IDBcbiAgICogbWF0Y2hlcyB0aGUgcGh5c2ljYWwgY3Vyc29yIGFmdGVyIEVOVEVSX0FMVF9TQ1JFRU4gKyBDU0kgSCAoaG9tZSkuXG4gICAqL1xuICBwcml2YXRlIHJlc2V0RnJhbWVzRm9yQWx0U2NyZWVuKCk6IHZvaWQge1xuICAgIGNvbnN0IHJvd3MgPSB0aGlzLnRlcm1pbmFsUm93c1xuICAgIGNvbnN0IGNvbHMgPSB0aGlzLnRlcm1pbmFsQ29sdW1uc1xuICAgIGNvbnN0IGJsYW5rID0gKCk6IEZyYW1lID0+ICh7XG4gICAgICBzY3JlZW46IGNyZWF0ZVNjcmVlbihcbiAgICAgICAgY29scyxcbiAgICAgICAgcm93cyxcbiAgICAgICAgdGhpcy5zdHlsZVBvb2wsXG4gICAgICAgIHRoaXMuY2hhclBvb2wsXG4gICAgICAgIHRoaXMuaHlwZXJsaW5rUG9vbCxcbiAgICAgICksXG4gICAgICB2aWV3cG9ydDogeyB3aWR0aDogY29scywgaGVpZ2h0OiByb3dzICsgMSB9LFxuICAgICAgY3Vyc29yOiB7IHg6IDAsIHk6IDAsIHZpc2libGU6IHRydWUgfSxcbiAgICB9KVxuICAgIHRoaXMuZnJvbnRGcmFtZSA9IGJsYW5rKClcbiAgICB0aGlzLmJhY2tGcmFtZSA9IGJsYW5rKClcbiAgICB0aGlzLmxvZy5yZXNldCgpXG4gICAgLy8gRGVmZW5zZS1pbi1kZXB0aDogYWx0LXNjcmVlbiBza2lwcyB0aGUgY3Vyc29yIHByZWFtYmxlIGFueXdheSAoQ1NJIEhcbiAgICAvLyByZXNldHMpLCBidXQgYSBzdGFsZSBkaXNwbGF5Q3Vyc29yIHdvdWxkIGJlIG1pc2xlYWRpbmcgaWYgd2UgbGF0ZXJcbiAgICAvLyBleGl0IHRvIG1haW4tc2NyZWVuIHdpdGhvdXQgYW4gaW50ZXJ2ZW5pbmcgcmVuZGVyLlxuICAgIHRoaXMuZGlzcGxheUN1cnNvciA9IG51bGxcbiAgICAvLyBGcmVzaCBmcm9udEZyYW1lIGlzIGJsYW5rIHJvd3PDl2NvbHMg4oCUIGJsaXR0aW5nIGZyb20gaXQgd291bGQgY29weVxuICAgIC8vIGJsYW5rcyBvdmVyIGNvbnRlbnQuIE5leHQgYWx0LXNjcmVlbiBmcmFtZSBtdXN0IGZ1bGwtcmVuZGVyLlxuICAgIHRoaXMucHJldkZyYW1lQ29udGFtaW5hdGVkID0gdHJ1ZVxuICB9XG5cbiAgLyoqXG4gICAqIENvcHkgdGhlIGN1cnJlbnQgc2VsZWN0aW9uIHRvIHRoZSBjbGlwYm9hcmQgd2l0aG91dCBjbGVhcmluZyB0aGVcbiAgICogaGlnaGxpZ2h0LiBNYXRjaGVzIGlUZXJtMidzIGNvcHktb24tc2VsZWN0IGJlaGF2aW9yIHdoZXJlIHRoZSBzZWxlY3RlZFxuICAgKiByZWdpb24gc3RheXMgdmlzaWJsZSBhZnRlciB0aGUgYXV0b21hdGljIGNvcHkuXG4gICAqL1xuICBjb3B5U2VsZWN0aW9uTm9DbGVhcigpOiBzdHJpbmcge1xuICAgIGlmICghaGFzU2VsZWN0aW9uKHRoaXMuc2VsZWN0aW9uKSkgcmV0dXJuICcnXG4gICAgY29uc3QgdGV4dCA9IGdldFNlbGVjdGVkVGV4dCh0aGlzLnNlbGVjdGlvbiwgdGhpcy5mcm9udEZyYW1lLnNjcmVlbilcbiAgICBpZiAodGV4dCkge1xuICAgICAgLy8gUmF3IE9TQyA1Miwgb3IgRENTLXBhc3N0aHJvdWdoLXdyYXBwZWQgT1NDIDUyIGluc2lkZSB0bXV4ICh0bXV4XG4gICAgICAvLyBkcm9wcyBpdCBzaWxlbnRseSB1bmxlc3MgYWxsb3ctcGFzc3Rocm91Z2ggaXMgb24g4oCUIG5vIHJlZ3Jlc3Npb24pLlxuICAgICAgdm9pZCBzZXRDbGlwYm9hcmQodGV4dCkudGhlbihyYXcgPT4ge1xuICAgICAgICBpZiAocmF3KSB0aGlzLm9wdGlvbnMuc3Rkb3V0LndyaXRlKHJhdylcbiAgICAgIH0pXG4gICAgfVxuICAgIHJldHVybiB0ZXh0XG4gIH1cblxuICAvKipcbiAgICogQ29weSB0aGUgY3VycmVudCB0ZXh0IHNlbGVjdGlvbiB0byB0aGUgc3lzdGVtIGNsaXBib2FyZCB2aWEgT1NDIDUyXG4gICAqIGFuZCBjbGVhciB0aGUgc2VsZWN0aW9uLiBSZXR1cm5zIHRoZSBjb3BpZWQgdGV4dCAoZW1wdHkgaWYgbm8gc2VsZWN0aW9uKS5cbiAgICovXG4gIGNvcHlTZWxlY3Rpb24oKTogc3RyaW5nIHtcbiAgICBpZiAoIWhhc1NlbGVjdGlvbih0aGlzLnNlbGVjdGlvbikpIHJldHVybiAnJ1xuICAgIGNvbnN0IHRleHQgPSB0aGlzLmNvcHlTZWxlY3Rpb25Ob0NsZWFyKClcbiAgICBjbGVhclNlbGVjdGlvbih0aGlzLnNlbGVjdGlvbilcbiAgICB0aGlzLm5vdGlmeVNlbGVjdGlvbkNoYW5nZSgpXG4gICAgcmV0dXJuIHRleHRcbiAgfVxuXG4gIC8qKiBDbGVhciB0aGUgY3VycmVudCB0ZXh0IHNlbGVjdGlvbiB3aXRob3V0IGNvcHlpbmcuICovXG4gIGNsZWFyVGV4dFNlbGVjdGlvbigpOiB2b2lkIHtcbiAgICBpZiAoIWhhc1NlbGVjdGlvbih0aGlzLnNlbGVjdGlvbikpIHJldHVyblxuICAgIGNsZWFyU2VsZWN0aW9uKHRoaXMuc2VsZWN0aW9uKVxuICAgIHRoaXMubm90aWZ5U2VsZWN0aW9uQ2hhbmdlKClcbiAgfVxuXG4gIC8qKlxuICAgKiBTZXQgdGhlIHNlYXJjaCBoaWdobGlnaHQgcXVlcnkuIE5vbi1lbXB0eSDihpIgYWxsIHZpc2libGUgb2NjdXJyZW5jZXNcbiAgICogYXJlIGludmVydGVkIChTR1IgNykgb24gdGhlIG5leHQgZnJhbWU7IGZpcnN0IG9uZSBhbHNvIHVuZGVybGluZWQuXG4gICAqIEVtcHR5IOKGkiBjbGVhcnMgKHByZXZGcmFtZUNvbnRhbWluYXRlZCBoYW5kbGVzIHRoZSBmcmFtZSBhZnRlcikuIFNhbWVcbiAgICogZGFtYWdlLXRyYWNraW5nIG1hY2hpbmVyeSBhcyBzZWxlY3Rpb24g4oCUIHNldENlbGxTdHlsZUlkIGRvZXNuJ3QgdHJhY2tcbiAgICogZGFtYWdlLCBzbyB0aGUgb3ZlcmxheSBmb3JjZXMgZnVsbC1mcmFtZSBkYW1hZ2Ugd2hpbGUgYWN0aXZlLlxuICAgKi9cbiAgc2V0U2VhcmNoSGlnaGxpZ2h0KHF1ZXJ5OiBzdHJpbmcpOiB2b2lkIHtcbiAgICBpZiAodGhpcy5zZWFyY2hIaWdobGlnaHRRdWVyeSA9PT0gcXVlcnkpIHJldHVyblxuICAgIHRoaXMuc2VhcmNoSGlnaGxpZ2h0UXVlcnkgPSBxdWVyeVxuICAgIHRoaXMuc2NoZWR1bGVSZW5kZXIoKVxuICB9XG5cbiAgLyoqIFBhaW50IGFuIEVYSVNUSU5HIERPTSBzdWJ0cmVlIHRvIGEgZnJlc2ggU2NyZWVuIGF0IGl0cyBuYXR1cmFsXG4gICAqICBoZWlnaHQsIHNjYW4gZm9yIHF1ZXJ5LiBSZXR1cm5zIHBvc2l0aW9ucyByZWxhdGl2ZSB0byB0aGUgZWxlbWVudCdzXG4gICAqICBib3VuZGluZyBib3ggKHJvdyAwID0gZWxlbWVudCB0b3ApLlxuICAgKlxuICAgKiAgVGhlIGVsZW1lbnQgY29tZXMgZnJvbSB0aGUgTUFJTiB0cmVlIOKAlCBidWlsdCB3aXRoIGFsbCByZWFsXG4gICAqICBwcm92aWRlcnMsIHlvZ2EgYWxyZWFkeSBjb21wdXRlZC4gV2UgcGFpbnQgaXQgdG8gYSBmcmVzaCBidWZmZXJcbiAgICogIHdpdGggb2Zmc2V0cyBzbyBpdCBsYW5kcyBhdCAoMCwwKS4gU2FtZSBwYWludCBwYXRoIGFzIHRoZSBtYWluXG4gICAqICByZW5kZXIuIFplcm8gZHJpZnQuIE5vIHNlY29uZCBSZWFjdCByb290LCBubyBjb250ZXh0IGJyaWRnZS5cbiAgICpcbiAgICogIH4xLTJtcyAocGFpbnQgb25seSwgbm8gcmVjb25jaWxlIOKAlCB0aGUgRE9NIGlzIGFscmVhZHkgYnVpbHQpLiAqL1xuICBzY2FuRWxlbWVudFN1YnRyZWUoZWw6IGRvbS5ET01FbGVtZW50KTogTWF0Y2hQb3NpdGlvbltdIHtcbiAgICBpZiAoIXRoaXMuc2VhcmNoSGlnaGxpZ2h0UXVlcnkgfHwgIWVsLnlvZ2FOb2RlKSByZXR1cm4gW11cbiAgICBjb25zdCB3aWR0aCA9IE1hdGguY2VpbChlbC55b2dhTm9kZS5nZXRDb21wdXRlZFdpZHRoKCkpXG4gICAgY29uc3QgaGVpZ2h0ID0gTWF0aC5jZWlsKGVsLnlvZ2FOb2RlLmdldENvbXB1dGVkSGVpZ2h0KCkpXG4gICAgaWYgKHdpZHRoIDw9IDAgfHwgaGVpZ2h0IDw9IDApIHJldHVybiBbXVxuICAgIC8vIHJlbmRlck5vZGVUb091dHB1dCBhZGRzIGVsJ3MgT1dOIGNvbXB1dGVkTGVmdC9Ub3AgdG8gb2Zmc2V0WC9ZLlxuICAgIC8vIFBhc3NpbmcgLWVsTGVmdC8tZWxUb3AgbmV0cyB0byAwIOKGkiBwYWludHMgYXQgKDAsMCkgaW4gb3VyIGJ1ZmZlci5cbiAgICBjb25zdCBlbExlZnQgPSBlbC55b2dhTm9kZS5nZXRDb21wdXRlZExlZnQoKVxuICAgIGNvbnN0IGVsVG9wID0gZWwueW9nYU5vZGUuZ2V0Q29tcHV0ZWRUb3AoKVxuICAgIGNvbnN0IHNjcmVlbiA9IGNyZWF0ZVNjcmVlbihcbiAgICAgIHdpZHRoLFxuICAgICAgaGVpZ2h0LFxuICAgICAgdGhpcy5zdHlsZVBvb2wsXG4gICAgICB0aGlzLmNoYXJQb29sLFxuICAgICAgdGhpcy5oeXBlcmxpbmtQb29sLFxuICAgIClcbiAgICBjb25zdCBvdXRwdXQgPSBuZXcgT3V0cHV0KHtcbiAgICAgIHdpZHRoLFxuICAgICAgaGVpZ2h0LFxuICAgICAgc3R5bGVQb29sOiB0aGlzLnN0eWxlUG9vbCxcbiAgICAgIHNjcmVlbixcbiAgICB9KVxuICAgIHJlbmRlck5vZGVUb091dHB1dChlbCwgb3V0cHV0LCB7XG4gICAgICBvZmZzZXRYOiAtZWxMZWZ0LFxuICAgICAgb2Zmc2V0WTogLWVsVG9wLFxuICAgICAgcHJldlNjcmVlbjogdW5kZWZpbmVkLFxuICAgIH0pXG4gICAgY29uc3QgcmVuZGVyZWQgPSBvdXRwdXQuZ2V0KClcbiAgICAvLyByZW5kZXJOb2RlVG9PdXRwdXQgd3JvdGUgb3VyIG9mZnNldCBwb3NpdGlvbnMgdG8gbm9kZUNhY2hlIOKAlFxuICAgIC8vIGNvcnJ1cHRzIHRoZSBtYWluIHJlbmRlciAoaXQnZCBibGl0IGZyb20gd3JvbmcgY29vcmRzKS4gTWFyayB0aGVcbiAgICAvLyBzdWJ0cmVlIGRpcnR5IHNvIHRoZSBuZXh0IG1haW4gcmVuZGVyIHJlcGFpbnRzICsgcmUtY2FjaGVzXG4gICAgLy8gY29ycmVjdGx5LiBPbmUgZXh0cmEgcGFpbnQgb2YgdGhpcyBtZXNzYWdlLCBidXQgY29ycmVjdCA+IGZhc3QuXG4gICAgZG9tLm1hcmtEaXJ0eShlbClcbiAgICBjb25zdCBwb3NpdGlvbnMgPSBzY2FuUG9zaXRpb25zKHJlbmRlcmVkLCB0aGlzLnNlYXJjaEhpZ2hsaWdodFF1ZXJ5KVxuICAgIGxvZ0ZvckRlYnVnZ2luZyhcbiAgICAgIGBzY2FuRWxlbWVudFN1YnRyZWU6IHE9JyR7dGhpcy5zZWFyY2hIaWdobGlnaHRRdWVyeX0nIGAgK1xuICAgICAgICBgZWw9JHt3aWR0aH14JHtoZWlnaHR9QCgke2VsTGVmdH0sJHtlbFRvcH0pIG49JHtwb3NpdGlvbnMubGVuZ3RofSBgICtcbiAgICAgICAgYFske3Bvc2l0aW9uc1xuICAgICAgICAgIC5zbGljZSgwLCAxMClcbiAgICAgICAgICAubWFwKHAgPT4gYCR7cC5yb3d9OiR7cC5jb2x9YClcbiAgICAgICAgICAuam9pbignLCcpfWAgK1xuICAgICAgICBgJHtwb3NpdGlvbnMubGVuZ3RoID4gMTAgPyAnLOKApicgOiAnJ31dYCxcbiAgICApXG4gICAgcmV0dXJuIHBvc2l0aW9uc1xuICB9XG5cbiAgLyoqIFNldCB0aGUgcG9zaXRpb24tYmFzZWQgaGlnaGxpZ2h0IHN0YXRlLiBFdmVyeSBmcmFtZSwgd3JpdGVzIENVUlJFTlRcbiAgICogIHN0eWxlIGF0IHBvc2l0aW9uc1tjdXJyZW50SWR4XSArIHJvd09mZnNldC4gbnVsbCBjbGVhcnMuIFRoZSBzY2FuLVxuICAgKiAgaGlnaGxpZ2h0IChpbnZlcnNlIG9uIGFsbCBtYXRjaGVzKSBzdGlsbCBydW5zIOKAlCB0aGlzIG92ZXJsYXlzIHllbGxvd1xuICAgKiAgb24gdG9wLiByb3dPZmZzZXQgY2hhbmdlcyBhcyB0aGUgdXNlciBzY3JvbGxzICg9IG1lc3NhZ2UncyBjdXJyZW50XG4gICAqICBzY3JlZW4tdG9wKTsgcG9zaXRpb25zIHN0YXkgc3RhYmxlIChtZXNzYWdlLXJlbGF0aXZlKS4gKi9cbiAgc2V0U2VhcmNoUG9zaXRpb25zKFxuICAgIHN0YXRlOiB7XG4gICAgICBwb3NpdGlvbnM6IE1hdGNoUG9zaXRpb25bXVxuICAgICAgcm93T2Zmc2V0OiBudW1iZXJcbiAgICAgIGN1cnJlbnRJZHg6IG51bWJlclxuICAgIH0gfCBudWxsLFxuICApOiB2b2lkIHtcbiAgICB0aGlzLnNlYXJjaFBvc2l0aW9ucyA9IHN0YXRlXG4gICAgdGhpcy5zY2hlZHVsZVJlbmRlcigpXG4gIH1cblxuICAvKipcbiAgICogU2V0IHRoZSBzZWxlY3Rpb24gaGlnaGxpZ2h0IGJhY2tncm91bmQgY29sb3IuIFJlcGxhY2VzIHRoZSBwZXItY2VsbFxuICAgKiBTR1ItNyBpbnZlcnNlIHdpdGggYSBzb2xpZCB0aGVtZS1hd2FyZSBiZyAobWF0Y2hlcyBuYXRpdmUgdGVybWluYWxcbiAgICogc2VsZWN0aW9uKS4gQWNjZXB0cyB0aGUgc2FtZSBjb2xvciBmb3JtYXRzIGFzIFRleHQgYmFja2dyb3VuZENvbG9yXG4gICAqIChyZ2IoKSwgYW5zaTpuYW1lLCAjaGV4LCBhbnNpMjU2KCkpIOKAlCBjb2xvcml6ZSgpIHJvdXRlcyB0aHJvdWdoXG4gICAqIGNoYWxrIHNvIHRoZSB0bXV4L3h0ZXJtLmpzIGxldmVsIGNsYW1wcyBpbiBjb2xvcml6ZS50cyBhcHBseSBhbmRcbiAgICogdGhlIGVtaXR0ZWQgU0dSIGlzIGNvcnJlY3QgZm9yIHRoZSBjdXJyZW50IHRlcm1pbmFsLlxuICAgKlxuICAgKiBDYWxsZWQgYnkgUmVhY3QtbGFuZCBvbmNlIHRoZW1lIGlzIGtub3duIChTY3JvbGxLZXliaW5kaW5nSGFuZGxlcidzXG4gICAqIHVzZUVmZmVjdCB3YXRjaGluZyB1c2VUaGVtZSkuIEJlZm9yZSB0aGF0IGNhbGwsIHdpdGhTZWxlY3Rpb25CZ1xuICAgKiBmYWxscyBiYWNrIHRvIHdpdGhJbnZlcnNlIHNvIHNlbGVjdGlvbiBzdGlsbCByZW5kZXJzIG9uIHRoZSBmaXJzdFxuICAgKiBmcmFtZTsgdGhlIGVmZmVjdCBmaXJlcyBiZWZvcmUgYW55IG1vdXNlIGlucHV0IHNvIHRoZSBmYWxsYmFjayBpc1xuICAgKiB1bm9ic2VydmFibGUgaW4gcHJhY3RpY2UuXG4gICAqL1xuICBzZXRTZWxlY3Rpb25CZ0NvbG9yKGNvbG9yOiBzdHJpbmcpOiB2b2lkIHtcbiAgICAvLyBXcmFwIGEgTlVMIG1hcmtlciwgdGhlbiBzcGxpdCBvbiBpdCB0byBleHRyYWN0IHRoZSBvcGVuL2Nsb3NlIFNHUi5cbiAgICAvLyBjb2xvcml6ZSByZXR1cm5zIHRoZSBpbnB1dCB1bmNoYW5nZWQgaWYgdGhlIGNvbG9yIHN0cmluZyBpcyBiYWQg4oCUXG4gICAgLy8gbm8gTlVMLXNwbGl0IHRoZW4sIHNvIGZhbGwgdGhyb3VnaCB0byBudWxsIChpbnZlcnNlIGZhbGxiYWNrKS5cbiAgICBjb25zdCB3cmFwcGVkID0gY29sb3JpemUoJ1xcMCcsIGNvbG9yLCAnYmFja2dyb3VuZCcpXG4gICAgY29uc3QgbnVsID0gd3JhcHBlZC5pbmRleE9mKCdcXDAnKVxuICAgIGlmIChudWwgPD0gMCB8fCBudWwgPT09IHdyYXBwZWQubGVuZ3RoIC0gMSkge1xuICAgICAgdGhpcy5zdHlsZVBvb2wuc2V0U2VsZWN0aW9uQmcobnVsbClcbiAgICAgIHJldHVyblxuICAgIH1cbiAgICB0aGlzLnN0eWxlUG9vbC5zZXRTZWxlY3Rpb25CZyh7XG4gICAgICB0eXBlOiAnYW5zaScsXG4gICAgICBjb2RlOiB3cmFwcGVkLnNsaWNlKDAsIG51bCksXG4gICAgICBlbmRDb2RlOiB3cmFwcGVkLnNsaWNlKG51bCArIDEpLCAvLyBhbHdheXMgXFx4MWJbNDltIGZvciBiZ1xuICAgIH0pXG4gICAgLy8gTm8gc2NoZWR1bGVSZW5kZXI6IHRoaXMgaXMgY2FsbGVkIGZyb20gYSBSZWFjdCBlZmZlY3QgdGhhdCBhbHJlYWR5XG4gICAgLy8gcnVucyBpbnNpZGUgdGhlIHJlbmRlciBjeWNsZSwgYW5kIHRoZSBiZyBvbmx5IG1hdHRlcnMgb25jZSBhXG4gICAgLy8gc2VsZWN0aW9uIGV4aXN0cyAod2hpY2ggaXRzZWxmIHRyaWdnZXJzIGEgZnVsbC1kYW1hZ2UgZnJhbWUpLlxuICB9XG5cbiAgLyoqXG4gICAqIENhcHR1cmUgdGV4dCBmcm9tIHJvd3MgYWJvdXQgdG8gc2Nyb2xsIG91dCBvZiB0aGUgdmlld3BvcnQgZHVyaW5nXG4gICAqIGRyYWctdG8tc2Nyb2xsLiBNdXN0IGJlIGNhbGxlZCBCRUZPUkUgdGhlIFNjcm9sbEJveCBzY3JvbGxzIHNvIHRoZVxuICAgKiBzY3JlZW4gYnVmZmVyIHN0aWxsIGhvbGRzIHRoZSBvdXRnb2luZyBjb250ZW50LiBBY2N1bXVsYXRlZCBpbnRvXG4gICAqIHRoZSBzZWxlY3Rpb24gc3RhdGUgYW5kIGpvaW5lZCBiYWNrIGluIGJ5IGdldFNlbGVjdGVkVGV4dC5cbiAgICovXG4gIGNhcHR1cmVTY3JvbGxlZFJvd3MoXG4gICAgZmlyc3RSb3c6IG51bWJlcixcbiAgICBsYXN0Um93OiBudW1iZXIsXG4gICAgc2lkZTogJ2Fib3ZlJyB8ICdiZWxvdycsXG4gICk6IHZvaWQge1xuICAgIGNhcHR1cmVTY3JvbGxlZFJvd3MoXG4gICAgICB0aGlzLnNlbGVjdGlvbixcbiAgICAgIHRoaXMuZnJvbnRGcmFtZS5zY3JlZW4sXG4gICAgICBmaXJzdFJvdyxcbiAgICAgIGxhc3RSb3csXG4gICAgICBzaWRlLFxuICAgIClcbiAgfVxuXG4gIC8qKlxuICAgKiBTaGlmdCBhbmNob3IgQU5EIGZvY3VzIGJ5IGRSb3csIGNsYW1wZWQgdG8gW21pblJvdywgbWF4Um93XS4gVXNlZCBieVxuICAgKiBrZXlib2FyZCBzY3JvbGwgaGFuZGxlcnMgKFBnVXAvUGdEbiBldGMuKSBzbyB0aGUgaGlnaGxpZ2h0IHRyYWNrcyB0aGVcbiAgICogY29udGVudCBpbnN0ZWFkIG9mIGRpc2FwcGVhcmluZy4gVW5saWtlIHNoaWZ0QW5jaG9yIChkcmFnLXRvLXNjcm9sbCksXG4gICAqIHRoaXMgbW92ZXMgQk9USCBlbmRwb2ludHMg4oCUIHRoZSB1c2VyIGlzbid0IGhvbGRpbmcgdGhlIG1vdXNlIGF0IG9uZVxuICAgKiBlZGdlLiBTdXBwbGllcyBzY3JlZW4ud2lkdGggZm9yIHRoZSBjb2wtcmVzZXQtb24tY2xhbXAgYm91bmRhcnkuXG4gICAqL1xuICBzaGlmdFNlbGVjdGlvbkZvclNjcm9sbChkUm93OiBudW1iZXIsIG1pblJvdzogbnVtYmVyLCBtYXhSb3c6IG51bWJlcik6IHZvaWQge1xuICAgIGNvbnN0IGhhZFNlbCA9IGhhc1NlbGVjdGlvbih0aGlzLnNlbGVjdGlvbilcbiAgICBzaGlmdFNlbGVjdGlvbihcbiAgICAgIHRoaXMuc2VsZWN0aW9uLFxuICAgICAgZFJvdyxcbiAgICAgIG1pblJvdyxcbiAgICAgIG1heFJvdyxcbiAgICAgIHRoaXMuZnJvbnRGcmFtZS5zY3JlZW4ud2lkdGgsXG4gICAgKVxuICAgIC8vIHNoaWZ0U2VsZWN0aW9uIGNsZWFycyB3aGVuIGJvdGggZW5kcG9pbnRzIG92ZXJzaG9vdCB0aGUgc2FtZSBlZGdlXG4gICAgLy8gKEhvbWUvZy9FbmQvRyBwYWdlLWp1bXAgcGFzdCB0aGUgc2VsZWN0aW9uKS4gTm90aWZ5IHN1YnNjcmliZXJzIHNvXG4gICAgLy8gdXNlSGFzU2VsZWN0aW9uIHVwZGF0ZXMuIFNhZmUgdG8gY2FsbCBub3RpZnlTZWxlY3Rpb25DaGFuZ2UgaGVyZSDigJRcbiAgICAvLyB0aGlzIHJ1bnMgZnJvbSBrZXlib2FyZCBoYW5kbGVycywgbm90IGluc2lkZSBvblJlbmRlcigpLlxuICAgIGlmIChoYWRTZWwgJiYgIWhhc1NlbGVjdGlvbih0aGlzLnNlbGVjdGlvbikpIHtcbiAgICAgIHRoaXMubm90aWZ5U2VsZWN0aW9uQ2hhbmdlKClcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogS2V5Ym9hcmQgc2VsZWN0aW9uIGV4dGVuc2lvbiAoc2hpZnQrYXJyb3cvaG9tZS9lbmQpLiBNb3ZlcyBmb2N1cztcbiAgICogYW5jaG9yIHN0YXlzIGZpeGVkIHNvIHRoZSBoaWdobGlnaHQgZ3Jvd3Mgb3Igc2hyaW5rcyByZWxhdGl2ZSB0byBpdC5cbiAgICogTGVmdC9yaWdodCB3cmFwIGFjcm9zcyByb3cgYm91bmRhcmllcyDigJQgbmF0aXZlIG1hY09TIHRleHQtZWRpdFxuICAgKiBiZWhhdmlvcjogc2hpZnQrbGVmdCBhdCBjb2wgMCB3cmFwcyB0byBlbmQgb2YgdGhlIHByZXZpb3VzIHJvdy5cbiAgICogVXAvZG93biBjbGFtcCBhdCB2aWV3cG9ydCBlZGdlcyAobm8gc2Nyb2xsLXRvLWV4dGVuZCB5ZXQpLiBEcm9wcyB0b1xuICAgKiBjaGFyIG1vZGUuIE5vLW9wIG91dHNpZGUgYWx0LXNjcmVlbiBvciB3aXRob3V0IGFuIGFjdGl2ZSBzZWxlY3Rpb24uXG4gICAqL1xuICBtb3ZlU2VsZWN0aW9uRm9jdXMobW92ZTogRm9jdXNNb3ZlKTogdm9pZCB7XG4gICAgaWYgKCF0aGlzLmFsdFNjcmVlbkFjdGl2ZSkgcmV0dXJuXG4gICAgY29uc3QgeyBmb2N1cyB9ID0gdGhpcy5zZWxlY3Rpb25cbiAgICBpZiAoIWZvY3VzKSByZXR1cm5cbiAgICBjb25zdCB7IHdpZHRoLCBoZWlnaHQgfSA9IHRoaXMuZnJvbnRGcmFtZS5zY3JlZW5cbiAgICBjb25zdCBtYXhDb2wgPSB3aWR0aCAtIDFcbiAgICBjb25zdCBtYXhSb3cgPSBoZWlnaHQgLSAxXG4gICAgbGV0IHsgY29sLCByb3cgfSA9IGZvY3VzXG4gICAgc3dpdGNoIChtb3ZlKSB7XG4gICAgICBjYXNlICdsZWZ0JzpcbiAgICAgICAgaWYgKGNvbCA+IDApIGNvbC0tXG4gICAgICAgIGVsc2UgaWYgKHJvdyA+IDApIHtcbiAgICAgICAgICBjb2wgPSBtYXhDb2xcbiAgICAgICAgICByb3ctLVxuICAgICAgICB9XG4gICAgICAgIGJyZWFrXG4gICAgICBjYXNlICdyaWdodCc6XG4gICAgICAgIGlmIChjb2wgPCBtYXhDb2wpIGNvbCsrXG4gICAgICAgIGVsc2UgaWYgKHJvdyA8IG1heFJvdykge1xuICAgICAgICAgIGNvbCA9IDBcbiAgICAgICAgICByb3crK1xuICAgICAgICB9XG4gICAgICAgIGJyZWFrXG4gICAgICBjYXNlICd1cCc6XG4gICAgICAgIGlmIChyb3cgPiAwKSByb3ctLVxuICAgICAgICBicmVha1xuICAgICAgY2FzZSAnZG93bic6XG4gICAgICAgIGlmIChyb3cgPCBtYXhSb3cpIHJvdysrXG4gICAgICAgIGJyZWFrXG4gICAgICBjYXNlICdsaW5lU3RhcnQnOlxuICAgICAgICBjb2wgPSAwXG4gICAgICAgIGJyZWFrXG4gICAgICBjYXNlICdsaW5lRW5kJzpcbiAgICAgICAgY29sID0gbWF4Q29sXG4gICAgICAgIGJyZWFrXG4gICAgfVxuICAgIGlmIChjb2wgPT09IGZvY3VzLmNvbCAmJiByb3cgPT09IGZvY3VzLnJvdykgcmV0dXJuXG4gICAgbW92ZUZvY3VzKHRoaXMuc2VsZWN0aW9uLCBjb2wsIHJvdylcbiAgICB0aGlzLm5vdGlmeVNlbGVjdGlvbkNoYW5nZSgpXG4gIH1cblxuICAvKiogV2hldGhlciB0aGVyZSBpcyBhbiBhY3RpdmUgdGV4dCBzZWxlY3Rpb24uICovXG4gIGhhc1RleHRTZWxlY3Rpb24oKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIGhhc1NlbGVjdGlvbih0aGlzLnNlbGVjdGlvbilcbiAgfVxuXG4gIC8qKlxuICAgKiBTdWJzY3JpYmUgdG8gc2VsZWN0aW9uIHN0YXRlIGNoYW5nZXMuIEZpcmVzIHdoZW5ldmVyIHRoZSBzZWxlY3Rpb25cbiAgICogaXMgc3RhcnRlZCwgdXBkYXRlZCwgY2xlYXJlZCwgb3IgY29waWVkLiBSZXR1cm5zIGFuIHVuc3Vic2NyaWJlIGZuLlxuICAgKi9cbiAgc3Vic2NyaWJlVG9TZWxlY3Rpb25DaGFuZ2UoY2I6ICgpID0+IHZvaWQpOiAoKSA9PiB2b2lkIHtcbiAgICB0aGlzLnNlbGVjdGlvbkxpc3RlbmVycy5hZGQoY2IpXG4gICAgcmV0dXJuICgpID0+IHRoaXMuc2VsZWN0aW9uTGlzdGVuZXJzLmRlbGV0ZShjYilcbiAgfVxuXG4gIHByaXZhdGUgbm90aWZ5U2VsZWN0aW9uQ2hhbmdlKCk6IHZvaWQge1xuICAgIHRoaXMub25SZW5kZXIoKVxuICAgIGZvciAoY29uc3QgY2Igb2YgdGhpcy5zZWxlY3Rpb25MaXN0ZW5lcnMpIGNiKClcbiAgfVxuXG4gIC8qKlxuICAgKiBIaXQtdGVzdCB0aGUgcmVuZGVyZWQgRE9NIHRyZWUgYXQgKGNvbCwgcm93KSBhbmQgYnViYmxlIGEgQ2xpY2tFdmVudFxuICAgKiBmcm9tIHRoZSBkZWVwZXN0IGhpdCBub2RlIHVwIHRocm91Z2ggYW5jZXN0b3JzIHdpdGggb25DbGljayBoYW5kbGVycy5cbiAgICogUmV0dXJucyB0cnVlIGlmIGEgRE9NIGhhbmRsZXIgY29uc3VtZWQgdGhlIGNsaWNrLiBHYXRlZCBvblxuICAgKiBhbHRTY3JlZW5BY3RpdmUg4oCUIGNsaWNrcyBvbmx5IG1ha2Ugc2Vuc2Ugd2l0aCBhIGZpeGVkIHZpZXdwb3J0IHdoZXJlXG4gICAqIG5vZGVDYWNoZSByZWN0cyBtYXAgMToxIHRvIHRlcm1pbmFsIGNlbGxzIChubyBzY3JvbGxiYWNrIG9mZnNldCkuXG4gICAqL1xuICBkaXNwYXRjaENsaWNrKGNvbDogbnVtYmVyLCByb3c6IG51bWJlcik6IGJvb2xlYW4ge1xuICAgIGlmICghdGhpcy5hbHRTY3JlZW5BY3RpdmUpIHJldHVybiBmYWxzZVxuICAgIGNvbnN0IGJsYW5rID0gaXNFbXB0eUNlbGxBdCh0aGlzLmZyb250RnJhbWUuc2NyZWVuLCBjb2wsIHJvdylcbiAgICByZXR1cm4gZGlzcGF0Y2hDbGljayh0aGlzLnJvb3ROb2RlLCBjb2wsIHJvdywgYmxhbmspXG4gIH1cblxuICBkaXNwYXRjaEhvdmVyKGNvbDogbnVtYmVyLCByb3c6IG51bWJlcik6IHZvaWQge1xuICAgIGlmICghdGhpcy5hbHRTY3JlZW5BY3RpdmUpIHJldHVyblxuICAgIGRpc3BhdGNoSG92ZXIodGhpcy5yb290Tm9kZSwgY29sLCByb3csIHRoaXMuaG92ZXJlZE5vZGVzKVxuICB9XG5cbiAgZGlzcGF0Y2hLZXlib2FyZEV2ZW50KHBhcnNlZEtleTogUGFyc2VkS2V5KTogdm9pZCB7XG4gICAgY29uc3QgdGFyZ2V0ID0gdGhpcy5mb2N1c01hbmFnZXIuYWN0aXZlRWxlbWVudCA/PyB0aGlzLnJvb3ROb2RlXG4gICAgY29uc3QgZXZlbnQgPSBuZXcgS2V5Ym9hcmRFdmVudChwYXJzZWRLZXkpXG4gICAgZGlzcGF0Y2hlci5kaXNwYXRjaERpc2NyZXRlKHRhcmdldCwgZXZlbnQpXG5cbiAgICAvLyBUYWIgY3ljbGluZyBpcyB0aGUgZGVmYXVsdCBhY3Rpb24g4oCUIG9ubHkgZmlyZXMgaWYgbm8gaGFuZGxlclxuICAgIC8vIGNhbGxlZCBwcmV2ZW50RGVmYXVsdCgpLiBNaXJyb3JzIGJyb3dzZXIgYmVoYXZpb3IuXG4gICAgaWYgKFxuICAgICAgIWV2ZW50LmRlZmF1bHRQcmV2ZW50ZWQgJiZcbiAgICAgIHBhcnNlZEtleS5uYW1lID09PSAndGFiJyAmJlxuICAgICAgIXBhcnNlZEtleS5jdHJsICYmXG4gICAgICAhcGFyc2VkS2V5Lm1ldGFcbiAgICApIHtcbiAgICAgIGlmIChwYXJzZWRLZXkuc2hpZnQpIHtcbiAgICAgICAgdGhpcy5mb2N1c01hbmFnZXIuZm9jdXNQcmV2aW91cyh0aGlzLnJvb3ROb2RlKVxuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy5mb2N1c01hbmFnZXIuZm9jdXNOZXh0KHRoaXMucm9vdE5vZGUpXG4gICAgICB9XG4gICAgfVxuICB9XG4gIC8qKlxuICAgKiBMb29rIHVwIHRoZSBVUkwgYXQgKGNvbCwgcm93KSBpbiB0aGUgY3VycmVudCBmcm9udCBmcmFtZS4gQ2hlY2tzIGZvclxuICAgKiBhbiBPU0MgOCBoeXBlcmxpbmsgZmlyc3QsIHRoZW4gZmFsbHMgYmFjayB0byBzY2FubmluZyB0aGUgcm93IGZvciBhXG4gICAqIHBsYWluLXRleHQgVVJMIChtb3VzZSB0cmFja2luZyBpbnRlcmNlcHRzIHRoZSB0ZXJtaW5hbCdzIG5hdGl2ZVxuICAgKiBDbWQrQ2xpY2sgVVJMIGRldGVjdGlvbiwgc28gd2UgcmVwbGljYXRlIGl0KS4gVGhpcyBpcyBhIHB1cmUgbG9va3VwXG4gICAqIHdpdGggbm8gc2lkZSBlZmZlY3RzIOKAlCBjYWxsIGl0IHN5bmNocm9ub3VzbHkgYXQgY2xpY2sgdGltZSBzbyB0aGVcbiAgICogcmVzdWx0IHJlZmxlY3RzIHRoZSBzY3JlZW4gdGhlIHVzZXIgYWN0dWFsbHkgY2xpY2tlZCBvbiwgdGhlbiBkZWZlclxuICAgKiB0aGUgYnJvd3Nlci1vcGVuIGFjdGlvbiB2aWEgYSB0aW1lci5cbiAgICovXG4gIGdldEh5cGVybGlua0F0KGNvbDogbnVtYmVyLCByb3c6IG51bWJlcik6IHN0cmluZyB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKCF0aGlzLmFsdFNjcmVlbkFjdGl2ZSkgcmV0dXJuIHVuZGVmaW5lZFxuICAgIGNvbnN0IHNjcmVlbiA9IHRoaXMuZnJvbnRGcmFtZS5zY3JlZW5cbiAgICBjb25zdCBjZWxsID0gY2VsbEF0KHNjcmVlbiwgY29sLCByb3cpXG4gICAgbGV0IHVybCA9IGNlbGw/Lmh5cGVybGlua1xuICAgIC8vIFNwYWNlclRhaWwgY2VsbHMgKHJpZ2h0IGhhbGYgb2Ygd2lkZS9DSksvZW1vamkgY2hhcnMpIHN0b3JlIHRoZVxuICAgIC8vIGh5cGVybGluayBvbiB0aGUgaGVhZCBjZWxsIGF0IGNvbC0xLlxuICAgIGlmICghdXJsICYmIGNlbGw/LndpZHRoID09PSBDZWxsV2lkdGguU3BhY2VyVGFpbCAmJiBjb2wgPiAwKSB7XG4gICAgICB1cmwgPSBjZWxsQXQoc2NyZWVuLCBjb2wgLSAxLCByb3cpPy5oeXBlcmxpbmtcbiAgICB9XG4gICAgcmV0dXJuIHVybCA/PyBmaW5kUGxhaW5UZXh0VXJsQXQoc2NyZWVuLCBjb2wsIHJvdylcbiAgfVxuXG4gIC8qKlxuICAgKiBPcHRpb25hbCBjYWxsYmFjayBmaXJlZCB3aGVuIGNsaWNraW5nIGFuIE9TQyA4IGh5cGVybGluayBpbiBmdWxsc2NyZWVuXG4gICAqIG1vZGUuIFNldCBieSBGdWxsc2NyZWVuTGF5b3V0IHZpYSB1c2VMYXlvdXRFZmZlY3QuXG4gICAqL1xuICBvbkh5cGVybGlua0NsaWNrOiAoKHVybDogc3RyaW5nKSA9PiB2b2lkKSB8IHVuZGVmaW5lZFxuXG4gIC8qKlxuICAgKiBTdGFibGUgcHJvdG90eXBlIHdyYXBwZXIgZm9yIG9uSHlwZXJsaW5rQ2xpY2suIFBhc3NlZCB0byA8QXBwPiBhc1xuICAgKiBvbk9wZW5IeXBlcmxpbmsgc28gdGhlIHByb3AgaXMgYSBib3VuZCBtZXRob2QgKGF1dG9CaW5kJ2QpIHRoYXQgcmVhZHNcbiAgICogdGhlIG11dGFibGUgZmllbGQgYXQgY2FsbCB0aW1lIOKAlCBub3QgdGhlIHVuZGVmaW5lZC1hdC1yZW5kZXIgdmFsdWUuXG4gICAqL1xuICBvcGVuSHlwZXJsaW5rKHVybDogc3RyaW5nKTogdm9pZCB7XG4gICAgdGhpcy5vbkh5cGVybGlua0NsaWNrPy4odXJsKVxuICB9XG5cbiAgLyoqXG4gICAqIEhhbmRsZSBhIGRvdWJsZS0gb3IgdHJpcGxlLWNsaWNrIGF0IChjb2wsIHJvdyk6IHNlbGVjdCB0aGUgd29yZCBvclxuICAgKiBsaW5lIHVuZGVyIHRoZSBjdXJzb3IgYnkgcmVhZGluZyB0aGUgY3VycmVudCBzY3JlZW4gYnVmZmVyLiBDYWxsZWQgb25cbiAgICogUFJFU1MgKG5vdCByZWxlYXNlKSBzbyB0aGUgaGlnaGxpZ2h0IGFwcGVhcnMgaW1tZWRpYXRlbHkgYW5kIGRyYWcgY2FuXG4gICAqIGV4dGVuZCB0aGUgc2VsZWN0aW9uIHdvcmQtYnktd29yZCAvIGxpbmUtYnktbGluZS4gRmFsbHMgYmFjayB0b1xuICAgKiBjaGFyLW1vZGUgc3RhcnRTZWxlY3Rpb24gaWYgdGhlIGNsaWNrIGxhbmRzIG9uIGEgbm9TZWxlY3QgY2VsbC5cbiAgICovXG4gIGhhbmRsZU11bHRpQ2xpY2soY29sOiBudW1iZXIsIHJvdzogbnVtYmVyLCBjb3VudDogMiB8IDMpOiB2b2lkIHtcbiAgICBpZiAoIXRoaXMuYWx0U2NyZWVuQWN0aXZlKSByZXR1cm5cbiAgICBjb25zdCBzY3JlZW4gPSB0aGlzLmZyb250RnJhbWUuc2NyZWVuXG4gICAgLy8gc2VsZWN0V29yZEF0L3NlbGVjdExpbmVBdCBuby1vcCBvbiBub1NlbGVjdC9vdXQtb2YtYm91bmRzLiBTZWVkIHdpdGhcbiAgICAvLyBhIGNoYXItbW9kZSBzZWxlY3Rpb24gc28gdGhlIHByZXNzIHN0aWxsIHN0YXJ0cyBhIGRyYWcgZXZlbiBpZiB0aGVcbiAgICAvLyB3b3JkL2xpbmUgc2NhbiBmaW5kcyBub3RoaW5nIHNlbGVjdGFibGUuXG4gICAgc3RhcnRTZWxlY3Rpb24odGhpcy5zZWxlY3Rpb24sIGNvbCwgcm93KVxuICAgIGlmIChjb3VudCA9PT0gMikgc2VsZWN0V29yZEF0KHRoaXMuc2VsZWN0aW9uLCBzY3JlZW4sIGNvbCwgcm93KVxuICAgIGVsc2Ugc2VsZWN0TGluZUF0KHRoaXMuc2VsZWN0aW9uLCBzY3JlZW4sIHJvdylcbiAgICAvLyBFbnN1cmUgaGFzU2VsZWN0aW9uIGlzIHRydWUgc28gcmVsZWFzZSBkb2Vzbid0IHJlLWRpc3BhdGNoIG9uQ2xpY2tBdC5cbiAgICAvLyBzZWxlY3RXb3JkQXQgbm8tb3BzIG9uIG5vU2VsZWN0OyBzZWxlY3RMaW5lQXQgbm8tb3BzIG91dC1vZi1ib3VuZHMuXG4gICAgaWYgKCF0aGlzLnNlbGVjdGlvbi5mb2N1cykgdGhpcy5zZWxlY3Rpb24uZm9jdXMgPSB0aGlzLnNlbGVjdGlvbi5hbmNob3JcbiAgICB0aGlzLm5vdGlmeVNlbGVjdGlvbkNoYW5nZSgpXG4gIH1cblxuICAvKipcbiAgICogSGFuZGxlIGEgZHJhZy1tb3Rpb24gYXQgKGNvbCwgcm93KS4gSW4gY2hhciBtb2RlIHVwZGF0ZXMgZm9jdXMgdG8gdGhlXG4gICAqIGV4YWN0IGNlbGwuIEluIHdvcmQvbGluZSBtb2RlIHNuYXBzIHRvIHdvcmQvbGluZSBib3VuZGFyaWVzIHNvIHRoZVxuICAgKiBzZWxlY3Rpb24gZXh0ZW5kcyBieSB3b3JkL2xpbmUgbGlrZSBuYXRpdmUgbWFjT1MuIEdhdGVkIG9uXG4gICAqIGFsdFNjcmVlbkFjdGl2ZSBmb3IgdGhlIHNhbWUgcmVhc29uIGFzIGRpc3BhdGNoQ2xpY2suXG4gICAqL1xuICBoYW5kbGVTZWxlY3Rpb25EcmFnKGNvbDogbnVtYmVyLCByb3c6IG51bWJlcik6IHZvaWQge1xuICAgIGlmICghdGhpcy5hbHRTY3JlZW5BY3RpdmUpIHJldHVyblxuICAgIGNvbnN0IHNlbCA9IHRoaXMuc2VsZWN0aW9uXG4gICAgaWYgKHNlbC5hbmNob3JTcGFuKSB7XG4gICAgICBleHRlbmRTZWxlY3Rpb24oc2VsLCB0aGlzLmZyb250RnJhbWUuc2NyZWVuLCBjb2wsIHJvdylcbiAgICB9IGVsc2Uge1xuICAgICAgdXBkYXRlU2VsZWN0aW9uKHNlbCwgY29sLCByb3cpXG4gICAgfVxuICAgIHRoaXMubm90aWZ5U2VsZWN0aW9uQ2hhbmdlKClcbiAgfVxuXG4gIC8vIE1ldGhvZHMgdG8gcHJvcGVybHkgc3VzcGVuZCBzdGRpbiBmb3IgZXh0ZXJuYWwgZWRpdG9yIHVzYWdlXG4gIC8vIFRoaXMgaXMgbmVlZGVkIHRvIHByZXZlbnQgSW5rIGZyb20gc3dhbGxvd2luZyBrZXlzdHJva2VzIHdoZW4gYW4gZXh0ZXJuYWwgZWRpdG9yIGlzIGFjdGl2ZVxuICBwcml2YXRlIHN0ZGluTGlzdGVuZXJzOiBBcnJheTx7XG4gICAgZXZlbnQ6IHN0cmluZ1xuICAgIGxpc3RlbmVyOiAoLi4uYXJnczogdW5rbm93bltdKSA9PiB2b2lkXG4gIH0+ID0gW11cbiAgcHJpdmF0ZSB3YXNSYXdNb2RlID0gZmFsc2VcblxuICBzdXNwZW5kU3RkaW4oKTogdm9pZCB7XG4gICAgY29uc3Qgc3RkaW4gPSB0aGlzLm9wdGlvbnMuc3RkaW5cbiAgICBpZiAoIXN0ZGluLmlzVFRZKSB7XG4gICAgICByZXR1cm5cbiAgICB9XG5cbiAgICAvLyBTdG9yZSBhbmQgcmVtb3ZlIGFsbCAncmVhZGFibGUnIGV2ZW50IGxpc3RlbmVycyB0ZW1wb3JhcmlseVxuICAgIC8vIFRoaXMgcHJldmVudHMgSW5rIGZyb20gY29uc3VtaW5nIHN0ZGluIHdoaWxlIHRoZSBlZGl0b3IgaXMgYWN0aXZlXG4gICAgY29uc3QgcmVhZGFibGVMaXN0ZW5lcnMgPSBzdGRpbi5saXN0ZW5lcnMoJ3JlYWRhYmxlJylcbiAgICBsb2dGb3JEZWJ1Z2dpbmcoXG4gICAgICBgW3N0ZGluXSBzdXNwZW5kU3RkaW46IHJlbW92aW5nICR7cmVhZGFibGVMaXN0ZW5lcnMubGVuZ3RofSByZWFkYWJsZSBsaXN0ZW5lcihzKSwgd2FzUmF3TW9kZT0keyhzdGRpbiBhcyBOb2RlSlMuUmVhZFN0cmVhbSAmIHsgaXNSYXc/OiBib29sZWFuIH0pLmlzUmF3ID8/IGZhbHNlfWAsXG4gICAgKVxuICAgIHJlYWRhYmxlTGlzdGVuZXJzLmZvckVhY2gobGlzdGVuZXIgPT4ge1xuICAgICAgdGhpcy5zdGRpbkxpc3RlbmVycy5wdXNoKHtcbiAgICAgICAgZXZlbnQ6ICdyZWFkYWJsZScsXG4gICAgICAgIGxpc3RlbmVyOiBsaXN0ZW5lciBhcyAoLi4uYXJnczogdW5rbm93bltdKSA9PiB2b2lkLFxuICAgICAgfSlcbiAgICAgIHN0ZGluLnJlbW92ZUxpc3RlbmVyKCdyZWFkYWJsZScsIGxpc3RlbmVyIGFzICguLi5hcmdzOiB1bmtub3duW10pID0+IHZvaWQpXG4gICAgfSlcblxuICAgIC8vIElmIHJhdyBtb2RlIGlzIGVuYWJsZWQsIGRpc2FibGUgaXQgdGVtcG9yYXJpbHlcbiAgICBjb25zdCBzdGRpbldpdGhSYXcgPSBzdGRpbiBhcyBOb2RlSlMuUmVhZFN0cmVhbSAmIHtcbiAgICAgIGlzUmF3PzogYm9vbGVhblxuICAgICAgc2V0UmF3TW9kZT86IChtb2RlOiBib29sZWFuKSA9PiB2b2lkXG4gICAgfVxuICAgIGlmIChzdGRpbldpdGhSYXcuaXNSYXcgJiYgc3RkaW5XaXRoUmF3LnNldFJhd01vZGUpIHtcbiAgICAgIHN0ZGluV2l0aFJhdy5zZXRSYXdNb2RlKGZhbHNlKVxuICAgICAgdGhpcy53YXNSYXdNb2RlID0gdHJ1ZVxuICAgIH1cbiAgfVxuXG4gIHJlc3VtZVN0ZGluKCk6IHZvaWQge1xuICAgIGNvbnN0IHN0ZGluID0gdGhpcy5vcHRpb25zLnN0ZGluXG4gICAgaWYgKCFzdGRpbi5pc1RUWSkge1xuICAgICAgcmV0dXJuXG4gICAgfVxuXG4gICAgLy8gUmUtYXR0YWNoIGFsbCB0aGUgc3RvcmVkIGxpc3RlbmVyc1xuICAgIGlmICh0aGlzLnN0ZGluTGlzdGVuZXJzLmxlbmd0aCA9PT0gMCAmJiAhdGhpcy53YXNSYXdNb2RlKSB7XG4gICAgICBsb2dGb3JEZWJ1Z2dpbmcoXG4gICAgICAgICdbc3RkaW5dIHJlc3VtZVN0ZGluOiBjYWxsZWQgd2l0aCBubyBzdG9yZWQgbGlzdGVuZXJzIGFuZCB3YXNSYXdNb2RlPWZhbHNlIChwb3NzaWJsZSBkZXN5bmMpJyxcbiAgICAgICAgeyBsZXZlbDogJ3dhcm4nIH0sXG4gICAgICApXG4gICAgfVxuICAgIGxvZ0ZvckRlYnVnZ2luZyhcbiAgICAgIGBbc3RkaW5dIHJlc3VtZVN0ZGluOiByZS1hdHRhY2hpbmcgJHt0aGlzLnN0ZGluTGlzdGVuZXJzLmxlbmd0aH0gbGlzdGVuZXIocyksIHdhc1Jhd01vZGU9JHt0aGlzLndhc1Jhd01vZGV9YCxcbiAgICApXG4gICAgdGhpcy5zdGRpbkxpc3RlbmVycy5mb3JFYWNoKCh7IGV2ZW50LCBsaXN0ZW5lciB9KSA9PiB7XG4gICAgICBzdGRpbi5hZGRMaXN0ZW5lcihldmVudCwgbGlzdGVuZXIpXG4gICAgfSlcbiAgICB0aGlzLnN0ZGluTGlzdGVuZXJzID0gW11cblxuICAgIC8vIFJlLWVuYWJsZSByYXcgbW9kZSBpZiBpdCB3YXMgZW5hYmxlZCBiZWZvcmVcbiAgICBpZiAodGhpcy53YXNSYXdNb2RlKSB7XG4gICAgICBjb25zdCBzdGRpbldpdGhSYXcgPSBzdGRpbiBhcyBOb2RlSlMuUmVhZFN0cmVhbSAmIHtcbiAgICAgICAgc2V0UmF3TW9kZT86IChtb2RlOiBib29sZWFuKSA9PiB2b2lkXG4gICAgICB9XG4gICAgICBpZiAoc3RkaW5XaXRoUmF3LnNldFJhd01vZGUpIHtcbiAgICAgICAgc3RkaW5XaXRoUmF3LnNldFJhd01vZGUodHJ1ZSlcbiAgICAgIH1cbiAgICAgIHRoaXMud2FzUmF3TW9kZSA9IGZhbHNlXG4gICAgfVxuICB9XG5cbiAgLy8gU3RhYmxlIGlkZW50aXR5IGZvciBUZXJtaW5hbFdyaXRlQ29udGV4dC4gQW4gaW5saW5lIGFycm93IGhlcmUgd291bGRcbiAgLy8gY2hhbmdlIG9uIGV2ZXJ5IHJlbmRlcigpIGNhbGwgKGluaXRpYWwgbW91bnQgKyBlYWNoIHJlc2l6ZSksIHdoaWNoXG4gIC8vIGNhc2NhZGVzIHRocm91Z2ggdXNlQ29udGV4dCDihpIgPEFsdGVybmF0ZVNjcmVlbj4ncyB1c2VMYXlvdXRFZmZlY3QgZGVwXG4gIC8vIGFycmF5IOKGkiBzcHVyaW91cyBleGl0K3JlLWVudGVyIG9mIHRoZSBhbHQgc2NyZWVuIG9uIGV2ZXJ5IFNJR1dJTkNILlxuICBwcml2YXRlIHdyaXRlUmF3KGRhdGE6IHN0cmluZyk6IHZvaWQge1xuICAgIHRoaXMub3B0aW9ucy5zdGRvdXQud3JpdGUoZGF0YSlcbiAgfVxuXG4gIHByaXZhdGUgc2V0Q3Vyc29yRGVjbGFyYXRpb246IEN1cnNvckRlY2xhcmF0aW9uU2V0dGVyID0gKFxuICAgIGRlY2wsXG4gICAgY2xlYXJJZk5vZGUsXG4gICkgPT4ge1xuICAgIGlmIChcbiAgICAgIGRlY2wgPT09IG51bGwgJiZcbiAgICAgIGNsZWFySWZOb2RlICE9PSB1bmRlZmluZWQgJiZcbiAgICAgIHRoaXMuY3Vyc29yRGVjbGFyYXRpb24/Lm5vZGUgIT09IGNsZWFySWZOb2RlXG4gICAgKSB7XG4gICAgICByZXR1cm5cbiAgICB9XG4gICAgdGhpcy5jdXJzb3JEZWNsYXJhdGlvbiA9IGRlY2xcbiAgfVxuXG4gIHJlbmRlcihub2RlOiBSZWFjdE5vZGUpOiB2b2lkIHtcbiAgICB0aGlzLmN1cnJlbnROb2RlID0gbm9kZVxuXG4gICAgY29uc3QgdHJlZSA9IChcbiAgICAgIDxBcHBcbiAgICAgICAgc3RkaW49e3RoaXMub3B0aW9ucy5zdGRpbn1cbiAgICAgICAgc3Rkb3V0PXt0aGlzLm9wdGlvbnMuc3Rkb3V0fVxuICAgICAgICBzdGRlcnI9e3RoaXMub3B0aW9ucy5zdGRlcnJ9XG4gICAgICAgIGV4aXRPbkN0cmxDPXt0aGlzLm9wdGlvbnMuZXhpdE9uQ3RybEN9XG4gICAgICAgIG9uRXhpdD17dGhpcy51bm1vdW50fVxuICAgICAgICB0ZXJtaW5hbENvbHVtbnM9e3RoaXMudGVybWluYWxDb2x1bW5zfVxuICAgICAgICB0ZXJtaW5hbFJvd3M9e3RoaXMudGVybWluYWxSb3dzfVxuICAgICAgICBzZWxlY3Rpb249e3RoaXMuc2VsZWN0aW9ufVxuICAgICAgICBvblNlbGVjdGlvbkNoYW5nZT17dGhpcy5ub3RpZnlTZWxlY3Rpb25DaGFuZ2V9XG4gICAgICAgIG9uQ2xpY2tBdD17dGhpcy5kaXNwYXRjaENsaWNrfVxuICAgICAgICBvbkhvdmVyQXQ9e3RoaXMuZGlzcGF0Y2hIb3Zlcn1cbiAgICAgICAgZ2V0SHlwZXJsaW5rQXQ9e3RoaXMuZ2V0SHlwZXJsaW5rQXR9XG4gICAgICAgIG9uT3Blbkh5cGVybGluaz17dGhpcy5vcGVuSHlwZXJsaW5rfVxuICAgICAgICBvbk11bHRpQ2xpY2s9e3RoaXMuaGFuZGxlTXVsdGlDbGlja31cbiAgICAgICAgb25TZWxlY3Rpb25EcmFnPXt0aGlzLmhhbmRsZVNlbGVjdGlvbkRyYWd9XG4gICAgICAgIG9uU3RkaW5SZXN1bWU9e3RoaXMucmVhc3NlcnRUZXJtaW5hbE1vZGVzfVxuICAgICAgICBvbkN1cnNvckRlY2xhcmF0aW9uPXt0aGlzLnNldEN1cnNvckRlY2xhcmF0aW9ufVxuICAgICAgICBkaXNwYXRjaEtleWJvYXJkRXZlbnQ9e3RoaXMuZGlzcGF0Y2hLZXlib2FyZEV2ZW50fVxuICAgICAgPlxuICAgICAgICA8VGVybWluYWxXcml0ZVByb3ZpZGVyIHZhbHVlPXt0aGlzLndyaXRlUmF3fT5cbiAgICAgICAgICB7bm9kZX1cbiAgICAgICAgPC9UZXJtaW5hbFdyaXRlUHJvdmlkZXI+XG4gICAgICA8L0FwcD5cbiAgICApXG5cbiAgICAvLyBAdHMtZXhwZWN0LWVycm9yIHVwZGF0ZUNvbnRhaW5lclN5bmMgZXhpc3RzIGluIHJlYWN0LXJlY29uY2lsZXIgYnV0IG5vdCBpbiBAdHlwZXMvcmVhY3QtcmVjb25jaWxlclxuICAgIHJlY29uY2lsZXIudXBkYXRlQ29udGFpbmVyU3luYyh0cmVlLCB0aGlzLmNvbnRhaW5lciwgbnVsbCwgbm9vcClcbiAgICAvLyBAdHMtZXhwZWN0LWVycm9yIGZsdXNoU3luY1dvcmsgZXhpc3RzIGluIHJlYWN0LXJlY29uY2lsZXIgYnV0IG5vdCBpbiBAdHlwZXMvcmVhY3QtcmVjb25jaWxlclxuICAgIHJlY29uY2lsZXIuZmx1c2hTeW5jV29yaygpXG4gIH1cblxuICB1bm1vdW50KGVycm9yPzogRXJyb3IgfCBudW1iZXIgfCBudWxsKTogdm9pZCB7XG4gICAgaWYgKHRoaXMuaXNVbm1vdW50ZWQpIHtcbiAgICAgIHJldHVyblxuICAgIH1cblxuICAgIHRoaXMub25SZW5kZXIoKVxuICAgIHRoaXMudW5zdWJzY3JpYmVFeGl0KClcblxuICAgIGlmICh0eXBlb2YgdGhpcy5yZXN0b3JlQ29uc29sZSA9PT0gJ2Z1bmN0aW9uJykge1xuICAgICAgdGhpcy5yZXN0b3JlQ29uc29sZSgpXG4gICAgfVxuICAgIHRoaXMucmVzdG9yZVN0ZGVycj8uKClcblxuICAgIHRoaXMudW5zdWJzY3JpYmVUVFlIYW5kbGVycz8uKClcblxuICAgIC8vIE5vbi1UVFkgZW52aXJvbm1lbnRzIGRvbid0IGhhbmRsZSBlcmFzaW5nIGFuc2kgZXNjYXBlcyB3ZWxsLCBzbyBpdCdzIGJldHRlciB0b1xuICAgIC8vIG9ubHkgcmVuZGVyIGxhc3QgZnJhbWUgb2Ygbm9uLXN0YXRpYyBvdXRwdXRcbiAgICBjb25zdCBkaWZmID0gdGhpcy5sb2cucmVuZGVyUHJldmlvdXNPdXRwdXRfREVQUkVDQVRFRCh0aGlzLmZyb250RnJhbWUpXG4gICAgd3JpdGVEaWZmVG9UZXJtaW5hbCh0aGlzLnRlcm1pbmFsLCBvcHRpbWl6ZShkaWZmKSlcblxuICAgIC8vIENsZWFuIHVwIHRlcm1pbmFsIG1vZGVzIHN5bmNocm9ub3VzbHkgYmVmb3JlIHByb2Nlc3MgZXhpdC5cbiAgICAvLyBSZWFjdCdzIGNvbXBvbmVudFdpbGxVbm1vdW50IHdvbid0IHJ1biBpbiB0aW1lIHdoZW4gcHJvY2Vzcy5leGl0KCkgaXMgY2FsbGVkLFxuICAgIC8vIHNvIHdlIG11c3QgcmVzZXQgdGVybWluYWwgbW9kZXMgaGVyZSB0byBwcmV2ZW50IGVzY2FwZSBzZXF1ZW5jZSBsZWFrYWdlLlxuICAgIC8vIFVzZSB3cml0ZVN5bmMgdG8gc3Rkb3V0IChmZCAxKSB0byBlbnN1cmUgd3JpdGVzIGNvbXBsZXRlIGJlZm9yZSBleGl0LlxuICAgIC8vIFdlIHVuY29uZGl0aW9uYWxseSBzZW5kIGFsbCBkaXNhYmxlIHNlcXVlbmNlcyBiZWNhdXNlIHRlcm1pbmFsIGRldGVjdGlvblxuICAgIC8vIG1heSBub3Qgd29yayBjb3JyZWN0bHkgKGUuZy4sIGluIHRtdXgsIHNjcmVlbikgYW5kIHRoZXNlIGFyZSBuby1vcHMgb25cbiAgICAvLyB0ZXJtaW5hbHMgdGhhdCBkb24ndCBzdXBwb3J0IHRoZW0uXG4gICAgLyogZXNsaW50LWRpc2FibGUgY3VzdG9tLXJ1bGVzL25vLXN5bmMtZnMgLS0gcHJvY2VzcyBleGl0aW5nOyBhc3luYyB3cml0ZXMgd291bGQgYmUgZHJvcHBlZCAqL1xuICAgIGlmICh0aGlzLm9wdGlvbnMuc3Rkb3V0LmlzVFRZKSB7XG4gICAgICBpZiAodGhpcy5hbHRTY3JlZW5BY3RpdmUpIHtcbiAgICAgICAgLy8gPEFsdGVybmF0ZVNjcmVlbj4ncyB1bm1vdW50IGVmZmVjdCB3b24ndCBydW4gZHVyaW5nIHNpZ25hbC1leGl0LlxuICAgICAgICAvLyBFeGl0IGFsdCBzY3JlZW4gRklSU1Qgc28gb3RoZXIgY2xlYW51cCBzZXF1ZW5jZXMgZ28gdG8gdGhlIG1haW4gc2NyZWVuLlxuICAgICAgICB3cml0ZVN5bmMoMSwgRVhJVF9BTFRfU0NSRUVOKVxuICAgICAgfVxuICAgICAgLy8gRGlzYWJsZSBtb3VzZSB0cmFja2luZyDigJQgdW5jb25kaXRpb25hbCBiZWNhdXNlIGFsdFNjcmVlbkFjdGl2ZSBjYW4gYmVcbiAgICAgIC8vIHN0YWxlIGlmIEFsdGVybmF0ZVNjcmVlbidzIHVubW91bnQgKHdoaWNoIGZsaXBzIHRoZSBmbGFnKSByYWNlZCBhXG4gICAgICAvLyBibG9ja2VkIGV2ZW50IGxvb3AgKyBTSUdJTlQuIE5vLW9wIGlmIHRyYWNraW5nIHdhcyBuZXZlciBlbmFibGVkLlxuICAgICAgd3JpdGVTeW5jKDEsIERJU0FCTEVfTU9VU0VfVFJBQ0tJTkcpXG4gICAgICAvLyBEcmFpbiBzdGRpbiBzbyBpbi1mbGlnaHQgbW91c2UgZXZlbnRzIGRvbid0IGxlYWsgdG8gdGhlIHNoZWxsXG4gICAgICB0aGlzLmRyYWluU3RkaW4oKVxuICAgICAgLy8gRGlzYWJsZSBleHRlbmRlZCBrZXkgcmVwb3J0aW5nIChib3RoIGtpdHR5IGFuZCBtb2RpZnlPdGhlcktleXMpXG4gICAgICB3cml0ZVN5bmMoMSwgRElTQUJMRV9NT0RJRllfT1RIRVJfS0VZUylcbiAgICAgIHdyaXRlU3luYygxLCBESVNBQkxFX0tJVFRZX0tFWUJPQVJEKVxuICAgICAgLy8gRGlzYWJsZSBmb2N1cyBldmVudHMgKERFQ1NFVCAxMDA0KVxuICAgICAgd3JpdGVTeW5jKDEsIERGRSlcbiAgICAgIC8vIERpc2FibGUgYnJhY2tldGVkIHBhc3RlIG1vZGVcbiAgICAgIHdyaXRlU3luYygxLCBEQlApXG4gICAgICAvLyBTaG93IGN1cnNvclxuICAgICAgd3JpdGVTeW5jKDEsIFNIT1dfQ1VSU09SKVxuICAgICAgLy8gQ2xlYXIgaVRlcm0yIHByb2dyZXNzIGJhclxuICAgICAgd3JpdGVTeW5jKDEsIENMRUFSX0lURVJNMl9QUk9HUkVTUylcbiAgICAgIC8vIENsZWFyIHRhYiBzdGF0dXMgKE9TQyAyMTMzNykgc28gYSBzdGFsZSBkb3QgZG9lc24ndCBsaW5nZXJcbiAgICAgIGlmIChzdXBwb3J0c1RhYlN0YXR1cygpKVxuICAgICAgICB3cml0ZVN5bmMoMSwgd3JhcEZvck11bHRpcGxleGVyKENMRUFSX1RBQl9TVEFUVVMpKVxuICAgIH1cbiAgICAvKiBlc2xpbnQtZW5hYmxlIGN1c3RvbS1ydWxlcy9uby1zeW5jLWZzICovXG5cbiAgICB0aGlzLmlzVW5tb3VudGVkID0gdHJ1ZVxuXG4gICAgLy8gQ2FuY2VsIGFueSBwZW5kaW5nIHRocm90dGxlZCByZW5kZXJzIHRvIHByZXZlbnQgYWNjZXNzaW5nIGZyZWVkIFlvZ2Egbm9kZXNcbiAgICB0aGlzLnNjaGVkdWxlUmVuZGVyLmNhbmNlbD8uKClcbiAgICBpZiAodGhpcy5kcmFpblRpbWVyICE9PSBudWxsKSB7XG4gICAgICBjbGVhclRpbWVvdXQodGhpcy5kcmFpblRpbWVyKVxuICAgICAgdGhpcy5kcmFpblRpbWVyID0gbnVsbFxuICAgIH1cblxuICAgIC8vIEB0cy1leHBlY3QtZXJyb3IgdXBkYXRlQ29udGFpbmVyU3luYyBleGlzdHMgaW4gcmVhY3QtcmVjb25jaWxlciBidXQgbm90IGluIEB0eXBlcy9yZWFjdC1yZWNvbmNpbGVyXG4gICAgcmVjb25jaWxlci51cGRhdGVDb250YWluZXJTeW5jKG51bGwsIHRoaXMuY29udGFpbmVyLCBudWxsLCBub29wKVxuICAgIC8vIEB0cy1leHBlY3QtZXJyb3IgZmx1c2hTeW5jV29yayBleGlzdHMgaW4gcmVhY3QtcmVjb25jaWxlciBidXQgbm90IGluIEB0eXBlcy9yZWFjdC1yZWNvbmNpbGVyXG4gICAgcmVjb25jaWxlci5mbHVzaFN5bmNXb3JrKClcbiAgICBpbnN0YW5jZXMuZGVsZXRlKHRoaXMub3B0aW9ucy5zdGRvdXQpXG5cbiAgICAvLyBGcmVlIHRoZSByb290IHlvZ2Egbm9kZSwgdGhlbiBjbGVhciBpdHMgcmVmZXJlbmNlLiBDaGlsZHJlbiBhcmUgYWxyZWFkeVxuICAgIC8vIGZyZWVkIGJ5IHRoZSByZWNvbmNpbGVyJ3MgcmVtb3ZlQ2hpbGRGcm9tQ29udGFpbmVyOyB1c2luZyAuZnJlZSgpIChub3RcbiAgICAvLyAuZnJlZVJlY3Vyc2l2ZSgpKSBhdm9pZHMgZG91YmxlLWZyZWVpbmcgdGhlbS5cbiAgICB0aGlzLnJvb3ROb2RlLnlvZ2FOb2RlPy5mcmVlKClcbiAgICB0aGlzLnJvb3ROb2RlLnlvZ2FOb2RlID0gdW5kZWZpbmVkXG5cbiAgICBpZiAoZXJyb3IgaW5zdGFuY2VvZiBFcnJvcikge1xuICAgICAgdGhpcy5yZWplY3RFeGl0UHJvbWlzZShlcnJvcilcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5yZXNvbHZlRXhpdFByb21pc2UoKVxuICAgIH1cbiAgfVxuXG4gIGFzeW5jIHdhaXRVbnRpbEV4aXQoKTogUHJvbWlzZTx2b2lkPiB7XG4gICAgdGhpcy5leGl0UHJvbWlzZSB8fD0gbmV3IFByb21pc2UoKHJlc29sdmUsIHJlamVjdCkgPT4ge1xuICAgICAgdGhpcy5yZXNvbHZlRXhpdFByb21pc2UgPSByZXNvbHZlXG4gICAgICB0aGlzLnJlamVjdEV4aXRQcm9taXNlID0gcmVqZWN0XG4gICAgfSlcblxuICAgIHJldHVybiB0aGlzLmV4aXRQcm9taXNlXG4gIH1cblxuICByZXNldExpbmVDb3VudCgpOiB2b2lkIHtcbiAgICBpZiAodGhpcy5vcHRpb25zLnN0ZG91dC5pc1RUWSkge1xuICAgICAgLy8gU3dhcCBzbyBvbGQgZnJvbnQgYmVjb21lcyBiYWNrIChmb3Igc2NyZWVuIHJldXNlKSwgdGhlbiByZXNldCBmcm9udFxuICAgICAgdGhpcy5iYWNrRnJhbWUgPSB0aGlzLmZyb250RnJhbWVcbiAgICAgIHRoaXMuZnJvbnRGcmFtZSA9IGVtcHR5RnJhbWUoXG4gICAgICAgIHRoaXMuZnJvbnRGcmFtZS52aWV3cG9ydC5oZWlnaHQsXG4gICAgICAgIHRoaXMuZnJvbnRGcmFtZS52aWV3cG9ydC53aWR0aCxcbiAgICAgICAgdGhpcy5zdHlsZVBvb2wsXG4gICAgICAgIHRoaXMuY2hhclBvb2wsXG4gICAgICAgIHRoaXMuaHlwZXJsaW5rUG9vbCxcbiAgICAgIClcbiAgICAgIHRoaXMubG9nLnJlc2V0KClcbiAgICAgIC8vIGZyb250RnJhbWUgaXMgcmVzZXQsIHNvIGZyYW1lLmN1cnNvciBvbiB0aGUgbmV4dCByZW5kZXIgaXMgKDAsMCkuXG4gICAgICAvLyBDbGVhciBkaXNwbGF5Q3Vyc29yIHNvIHRoZSBwcmVhbWJsZSBkb2Vzbid0IGNvbXB1dGUgYSBzdGFsZSBkZWx0YS5cbiAgICAgIHRoaXMuZGlzcGxheUN1cnNvciA9IG51bGxcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogUmVwbGFjZSBjaGFyL2h5cGVybGluayBwb29scyB3aXRoIGZyZXNoIGluc3RhbmNlcyB0byBwcmV2ZW50IHVuYm91bmRlZFxuICAgKiBncm93dGggZHVyaW5nIGxvbmcgc2Vzc2lvbnMuIE1pZ3JhdGVzIHRoZSBmcm9udCBmcmFtZSdzIHNjcmVlbiBJRHMgaW50b1xuICAgKiB0aGUgbmV3IHBvb2xzIHNvIGRpZmZpbmcgcmVtYWlucyBjb3JyZWN0LiBUaGUgYmFjayBmcmFtZSBkb2Vzbid0IG5lZWRcbiAgICogbWlncmF0aW9uIOKAlCByZXNldFNjcmVlbiB6ZXJvcyBpdCBiZWZvcmUgYW55IHJlYWRzLlxuICAgKlxuICAgKiBDYWxsIGJldHdlZW4gY29udmVyc2F0aW9uIHR1cm5zIG9yIHBlcmlvZGljYWxseS5cbiAgICovXG4gIHJlc2V0UG9vbHMoKTogdm9pZCB7XG4gICAgdGhpcy5jaGFyUG9vbCA9IG5ldyBDaGFyUG9vbCgpXG4gICAgdGhpcy5oeXBlcmxpbmtQb29sID0gbmV3IEh5cGVybGlua1Bvb2woKVxuICAgIG1pZ3JhdGVTY3JlZW5Qb29scyhcbiAgICAgIHRoaXMuZnJvbnRGcmFtZS5zY3JlZW4sXG4gICAgICB0aGlzLmNoYXJQb29sLFxuICAgICAgdGhpcy5oeXBlcmxpbmtQb29sLFxuICAgIClcbiAgICAvLyBCYWNrIGZyYW1lJ3MgZGF0YSBpcyB6ZXJvZWQgYnkgcmVzZXRTY3JlZW4gYmVmb3JlIHJlYWRzLCBidXQgaXRzIHBvb2xcbiAgICAvLyByZWZlcmVuY2VzIGFyZSB1c2VkIGJ5IHRoZSByZW5kZXJlciB0byBpbnRlcm4gbmV3IGNoYXJhY3RlcnMuIFBvaW50XG4gICAgLy8gdGhlbSBhdCB0aGUgbmV3IHBvb2xzIHNvIHRoZSBuZXh0IGZyYW1lJ3MgSURzIGFyZSBjb21wYXJhYmxlLlxuICAgIHRoaXMuYmFja0ZyYW1lLnNjcmVlbi5jaGFyUG9vbCA9IHRoaXMuY2hhclBvb2xcbiAgICB0aGlzLmJhY2tGcmFtZS5zY3JlZW4uaHlwZXJsaW5rUG9vbCA9IHRoaXMuaHlwZXJsaW5rUG9vbFxuICB9XG5cbiAgcGF0Y2hDb25zb2xlKCk6ICgpID0+IHZvaWQge1xuICAgIC8vIGJpb21lLWlnbm9yZSBsaW50L3N1c3BpY2lvdXMvbm9Db25zb2xlOiBpbnRlbnRpb25hbGx5IHBhdGNoaW5nIGdsb2JhbCBjb25zb2xlXG4gICAgY29uc3QgY29uID0gY29uc29sZVxuICAgIGNvbnN0IG9yaWdpbmFsczogUGFydGlhbDxSZWNvcmQ8a2V5b2YgQ29uc29sZSwgQ29uc29sZVtrZXlvZiBDb25zb2xlXT4+ID0ge31cbiAgICBjb25zdCB0b0RlYnVnID0gKC4uLmFyZ3M6IHVua25vd25bXSkgPT5cbiAgICAgIGxvZ0ZvckRlYnVnZ2luZyhgY29uc29sZS5sb2c6ICR7Zm9ybWF0KC4uLmFyZ3MpfWApXG4gICAgY29uc3QgdG9FcnJvciA9ICguLi5hcmdzOiB1bmtub3duW10pID0+XG4gICAgICBsb2dFcnJvcihuZXcgRXJyb3IoYGNvbnNvbGUuZXJyb3I6ICR7Zm9ybWF0KC4uLmFyZ3MpfWApKVxuICAgIGZvciAoY29uc3QgbSBvZiBDT05TT0xFX1NURE9VVF9NRVRIT0RTKSB7XG4gICAgICBvcmlnaW5hbHNbbV0gPSBjb25bbV1cbiAgICAgIGNvblttXSA9IHRvRGVidWdcbiAgICB9XG4gICAgZm9yIChjb25zdCBtIG9mIENPTlNPTEVfU1RERVJSX01FVEhPRFMpIHtcbiAgICAgIG9yaWdpbmFsc1ttXSA9IGNvblttXVxuICAgICAgY29uW21dID0gdG9FcnJvclxuICAgIH1cbiAgICBvcmlnaW5hbHMuYXNzZXJ0ID0gY29uLmFzc2VydFxuICAgIGNvbi5hc3NlcnQgPSAoY29uZGl0aW9uOiB1bmtub3duLCAuLi5hcmdzOiB1bmtub3duW10pID0+IHtcbiAgICAgIGlmICghY29uZGl0aW9uKSB0b0Vycm9yKC4uLmFyZ3MpXG4gICAgfVxuICAgIHJldHVybiAoKSA9PiBPYmplY3QuYXNzaWduKGNvbiwgb3JpZ2luYWxzKVxuICB9XG5cbiAgLyoqXG4gICAqIEludGVyY2VwdCBwcm9jZXNzLnN0ZGVyci53cml0ZSBzbyBzdHJheSB3cml0ZXMgKGNvbmZpZy50cywgaG9va3MudHMsXG4gICAqIHRoaXJkLXBhcnR5IGRlcHMpIGRvbid0IGNvcnJ1cHQgdGhlIGFsdC1zY3JlZW4gYnVmZmVyLiBwYXRjaENvbnNvbGUgb25seVxuICAgKiBob29rcyBjb25zb2xlLiogbWV0aG9kcyDigJQgZGlyZWN0IHN0ZGVyciB3cml0ZXMgYnlwYXNzIGl0LCBsYW5kIGF0IHRoZVxuICAgKiBwYXJrZWQgY3Vyc29yLCBzY3JvbGwgdGhlIGFsdC1zY3JlZW4sIGFuZCBkZXN5bmMgZnJvbnRGcmFtZSBmcm9tIHRoZVxuICAgKiBwaHlzaWNhbCB0ZXJtaW5hbC4gTmV4dCBkaWZmIHdyaXRlcyBvbmx5IGNoYW5nZWQtaW4tUmVhY3QgY2VsbHMgYXRcbiAgICogYWJzb2x1dGUgY29vcmRzIOKGkiBpbnRlcmxlYXZlZCBnYXJiYWdlLlxuICAgKlxuICAgKiBTd2FsbG93cyB0aGUgd3JpdGUgKHJvdXRlcyB0ZXh0IHRvIHRoZSBkZWJ1ZyBsb2cpIGFuZCwgaW4gYWx0LXNjcmVlbixcbiAgICogZm9yY2VzIGEgZnVsbC1kYW1hZ2UgcmVwYWludCBhcyBhIGRlZmVuc2l2ZSByZWNvdmVyeS4gTm90IHBhdGNoaW5nXG4gICAqIHByb2Nlc3Muc3Rkb3V0IOKAlCBJbmsgaXRzZWxmIHdyaXRlcyB0aGVyZS5cbiAgICovXG4gIHByaXZhdGUgcGF0Y2hTdGRlcnIoKTogKCkgPT4gdm9pZCB7XG4gICAgY29uc3Qgc3RkZXJyID0gcHJvY2Vzcy5zdGRlcnJcbiAgICBjb25zdCBvcmlnaW5hbFdyaXRlID0gc3RkZXJyLndyaXRlXG4gICAgbGV0IHJlZW50ZXJlZCA9IGZhbHNlXG4gICAgY29uc3QgaW50ZXJjZXB0ID0gKFxuICAgICAgY2h1bms6IFVpbnQ4QXJyYXkgfCBzdHJpbmcsXG4gICAgICBlbmNvZGluZ09yQ2I/OiBCdWZmZXJFbmNvZGluZyB8ICgoZXJyPzogRXJyb3IpID0+IHZvaWQpLFxuICAgICAgY2I/OiAoZXJyPzogRXJyb3IpID0+IHZvaWQsXG4gICAgKTogYm9vbGVhbiA9PiB7XG4gICAgICBjb25zdCBjYWxsYmFjayA9IHR5cGVvZiBlbmNvZGluZ09yQ2IgPT09ICdmdW5jdGlvbicgPyBlbmNvZGluZ09yQ2IgOiBjYlxuICAgICAgLy8gUmVlbnRyYW5jeSBndWFyZDogbG9nRm9yRGVidWdnaW5nIOKGkiB3cml0ZVRvU3RkZXJyIOKGkiBoZXJlLiBQYXNzXG4gICAgICAvLyB0aHJvdWdoIHRvIHRoZSBvcmlnaW5hbCBzbyAtLWRlYnVnLXRvLXN0ZGVyciBzdGlsbCB3b3JrcyBhbmQgd2VcbiAgICAgIC8vIGRvbid0IHN0YWNrLW92ZXJmbG93LlxuICAgICAgaWYgKHJlZW50ZXJlZCkge1xuICAgICAgICBjb25zdCBlbmNvZGluZyA9XG4gICAgICAgICAgdHlwZW9mIGVuY29kaW5nT3JDYiA9PT0gJ3N0cmluZycgPyBlbmNvZGluZ09yQ2IgOiB1bmRlZmluZWRcbiAgICAgICAgcmV0dXJuIG9yaWdpbmFsV3JpdGUuY2FsbChzdGRlcnIsIGNodW5rLCBlbmNvZGluZywgY2FsbGJhY2spXG4gICAgICB9XG4gICAgICByZWVudGVyZWQgPSB0cnVlXG4gICAgICB0cnkge1xuICAgICAgICBjb25zdCB0ZXh0ID1cbiAgICAgICAgICB0eXBlb2YgY2h1bmsgPT09ICdzdHJpbmcnXG4gICAgICAgICAgICA/IGNodW5rXG4gICAgICAgICAgICA6IEJ1ZmZlci5mcm9tKGNodW5rKS50b1N0cmluZygndXRmOCcpXG4gICAgICAgIGxvZ0ZvckRlYnVnZ2luZyhgW3N0ZGVycl0gJHt0ZXh0fWAsIHsgbGV2ZWw6ICd3YXJuJyB9KVxuICAgICAgICBpZiAodGhpcy5hbHRTY3JlZW5BY3RpdmUgJiYgIXRoaXMuaXNVbm1vdW50ZWQgJiYgIXRoaXMuaXNQYXVzZWQpIHtcbiAgICAgICAgICB0aGlzLnByZXZGcmFtZUNvbnRhbWluYXRlZCA9IHRydWVcbiAgICAgICAgICB0aGlzLnNjaGVkdWxlUmVuZGVyKClcbiAgICAgICAgfVxuICAgICAgfSBmaW5hbGx5IHtcbiAgICAgICAgcmVlbnRlcmVkID0gZmFsc2VcbiAgICAgICAgY2FsbGJhY2s/LigpXG4gICAgICB9XG4gICAgICByZXR1cm4gdHJ1ZVxuICAgIH1cbiAgICBzdGRlcnIud3JpdGUgPSBpbnRlcmNlcHRcbiAgICByZXR1cm4gKCkgPT4ge1xuICAgICAgaWYgKHN0ZGVyci53cml0ZSA9PT0gaW50ZXJjZXB0KSB7XG4gICAgICAgIHN0ZGVyci53cml0ZSA9IG9yaWdpbmFsV3JpdGVcbiAgICAgIH1cbiAgICB9XG4gIH1cbn1cblxuLyoqXG4gKiBEaXNjYXJkIHBlbmRpbmcgc3RkaW4gYnl0ZXMgc28gaW4tZmxpZ2h0IGVzY2FwZSBzZXF1ZW5jZXMgKG1vdXNlIHRyYWNraW5nXG4gKiByZXBvcnRzLCBicmFja2V0ZWQtcGFzdGUgbWFya2VycykgZG9uJ3QgbGVhayB0byB0aGUgc2hlbGwgYWZ0ZXIgZXhpdC5cbiAqXG4gKiBUd28gbGF5ZXJzIG9mIHRyaWNraW5lc3M6XG4gKlxuICogMS4gc2V0UmF3TW9kZSBpcyB0ZXJtaW9zLCBub3QgZmNudGwg4oCUIHRoZSBzdGRpbiBmZCBzdGF5cyBibG9ja2luZywgc29cbiAqICAgIHJlYWRTeW5jIG9uIGl0IHdvdWxkIGhhbmcgZm9yZXZlci4gTm9kZSBkb2Vzbid0IGV4cG9zZSBmY250bCwgc28gd2VcbiAqICAgIG9wZW4gL2Rldi90dHkgZnJlc2ggd2l0aCBPX05PTkJMT0NLIChhbGwgZmRzIHRvIHRoZSBjb250cm9sbGluZ1xuICogICAgdGVybWluYWwgc2hhcmUgb25lIGxpbmUtZGlzY2lwbGluZSBpbnB1dCBxdWV1ZSkuXG4gKlxuICogMi4gQnkgdGhlIHRpbWUgZm9yY2VFeGl0IGNhbGxzIHRoaXMsIGRldGFjaEZvclNodXRkb3duIGhhcyBhbHJlYWR5IHB1dFxuICogICAgdGhlIFRUWSBiYWNrIGluIGNvb2tlZCAoY2Fub25pY2FsKSBtb2RlLiBDYW5vbmljYWwgbW9kZSBsaW5lLWJ1ZmZlcnNcbiAqICAgIGlucHV0IHVudGlsIG5ld2xpbmUsIHNvIE9fTk9OQkxPQ0sgcmVhZHMgcmV0dXJuIEVBR0FJTiBldmVuIHdoZW5cbiAqICAgIG1vdXNlIGJ5dGVzIGFyZSBzaXR0aW5nIGluIHRoZSBidWZmZXIuIFdlIGJyaWVmbHkgcmUtZW50ZXIgcmF3IG1vZGVcbiAqICAgIHNvIHJlYWRzIHJldHVybiBhbnkgYXZhaWxhYmxlIGJ5dGVzLCB0aGVuIHJlc3RvcmUgY29va2VkIG1vZGUuXG4gKlxuICogU2FmZSB0byBjYWxsIG11bHRpcGxlIHRpbWVzLiBDYWxsIGFzIExBVEUgYXMgcG9zc2libGUgaW4gdGhlIGV4aXQgcGF0aDpcbiAqIERJU0FCTEVfTU9VU0VfVFJBQ0tJTkcgaGFzIHRlcm1pbmFsIHJvdW5kLXRyaXAgbGF0ZW5jeSwgc28gZXZlbnRzIGNhblxuICogYXJyaXZlIGZvciBhIGZldyBtcyBhZnRlciBpdCdzIHdyaXR0ZW4uXG4gKi9cbi8qIGVzbGludC1kaXNhYmxlIGN1c3RvbS1ydWxlcy9uby1zeW5jLWZzIC0tIG11c3QgYmUgc3luYzsgY2FsbGVkIGZyb20gc2lnbmFsIGhhbmRsZXIgLyB1bm1vdW50ICovXG5leHBvcnQgZnVuY3Rpb24gZHJhaW5TdGRpbihzdGRpbjogTm9kZUpTLlJlYWRTdHJlYW0gPSBwcm9jZXNzLnN0ZGluKTogdm9pZCB7XG4gIGlmICghc3RkaW4uaXNUVFkpIHJldHVyblxuICAvLyBEcmFpbiBOb2RlJ3Mgc3RyZWFtIGJ1ZmZlciAoYnl0ZXMgbGlidXYgYWxyZWFkeSBwdWxsZWQgaW4pLiByZWFkKClcbiAgLy8gcmV0dXJucyBudWxsIHdoZW4gZW1wdHkg4oCUIG5ldmVyIGJsb2Nrcy5cbiAgdHJ5IHtcbiAgICB3aGlsZSAoc3RkaW4ucmVhZCgpICE9PSBudWxsKSB7XG4gICAgICAvKiBkaXNjYXJkICovXG4gICAgfVxuICB9IGNhdGNoIHtcbiAgICAvKiBzdHJlYW0gbWF5IGJlIGRlc3Ryb3llZCAqL1xuICB9XG4gIC8vIE5vIC9kZXYvdHR5IG9uIFdpbmRvd3M7IENPTklOJCBkb2Vzbid0IHN1cHBvcnQgT19OT05CTE9DSyBzZW1hbnRpY3MuXG4gIC8vIFdpbmRvd3MgVGVybWluYWwgYWxzbyBkb2Vzbid0IGJ1ZmZlciBtb3VzZSByZXBvcnRzIHRoZSBzYW1lIHdheS5cbiAgaWYgKHByb2Nlc3MucGxhdGZvcm0gPT09ICd3aW4zMicpIHJldHVyblxuICAvLyB0ZXJtaW9zIGlzIHBlci1kZXZpY2U6IGZsaXAgc3RkaW4gdG8gcmF3IHNvIGNhbm9uaWNhbC1tb2RlIGxpbmVcbiAgLy8gYnVmZmVyaW5nIGRvZXNuJ3QgaGlkZSBwYXJ0aWFsIGlucHV0IGZyb20gdGhlIG5vbi1ibG9ja2luZyByZWFkLlxuICAvLyBSZXN0b3JlZCBpbiB0aGUgZmluYWxseSBibG9jay5cbiAgY29uc3QgdHR5ID0gc3RkaW4gYXMgTm9kZUpTLlJlYWRTdHJlYW0gJiB7XG4gICAgaXNSYXc/OiBib29sZWFuXG4gICAgc2V0UmF3TW9kZT86IChyYXc6IGJvb2xlYW4pID0+IHZvaWRcbiAgfVxuICBjb25zdCB3YXNSYXcgPSB0dHkuaXNSYXcgPT09IHRydWVcbiAgLy8gRHJhaW4gdGhlIGtlcm5lbCBUVFkgYnVmZmVyIHZpYSBhIGZyZXNoIE9fTk9OQkxPQ0sgZmQuIEJvdW5kZWQgYXQgNjRcbiAgLy8gcmVhZHMgKDY0S0IpIOKAlCBhIHJlYWwgbW91c2UgYnVyc3QgaXMgYSBmZXcgaHVuZHJlZCBieXRlczsgdGhlIGNhcFxuICAvLyBndWFyZHMgYWdhaW5zdCBhIHRlcm1pbmFsIHRoYXQgaWdub3JlcyBPX05PTkJMT0NLLlxuICBsZXQgZmQgPSAtMVxuICB0cnkge1xuICAgIC8vIHNldFJhd01vZGUgaW5zaWRlIHRyeTogb24gcmV2b2tlZCBUVFkgKFNJR0hVUC9TU0ggZGlzY29ubmVjdCkgdGhlXG4gICAgLy8gaW9jdGwgdGhyb3dzIEVCQURGIOKAlCBzYW1lIHJlY292ZXJ5IHBhdGggYXMgb3BlblN5bmMvcmVhZFN5bmMgYmVsb3cuXG4gICAgaWYgKCF3YXNSYXcpIHR0eS5zZXRSYXdNb2RlPy4odHJ1ZSlcbiAgICBmZCA9IG9wZW5TeW5jKCcvZGV2L3R0eScsIGZzQ29uc3RhbnRzLk9fUkRPTkxZIHwgZnNDb25zdGFudHMuT19OT05CTE9DSylcbiAgICBjb25zdCBidWYgPSBCdWZmZXIuYWxsb2MoMTAyNClcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IDY0OyBpKyspIHtcbiAgICAgIGlmIChyZWFkU3luYyhmZCwgYnVmLCAwLCBidWYubGVuZ3RoLCBudWxsKSA8PSAwKSBicmVha1xuICAgIH1cbiAgfSBjYXRjaCB7XG4gICAgLy8gRUFHQUlOIChidWZmZXIgZW1wdHkg4oCUIGV4cGVjdGVkKSwgRU5YSU8vRU5PRU5UIChubyBjb250cm9sbGluZyB0dHkpLFxuICAgIC8vIEVCQURGL0VJTyAoVFRZIHJldm9rZWQg4oCUIFNJR0hVUCwgU1NIIGRpc2Nvbm5lY3QpXG4gIH0gZmluYWxseSB7XG4gICAgaWYgKGZkID49IDApIHtcbiAgICAgIHRyeSB7XG4gICAgICAgIGNsb3NlU3luYyhmZClcbiAgICAgIH0gY2F0Y2gge1xuICAgICAgICAvKiBpZ25vcmUgKi9cbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKCF3YXNSYXcpIHtcbiAgICAgIHRyeSB7XG4gICAgICAgIHR0eS5zZXRSYXdNb2RlPy4oZmFsc2UpXG4gICAgICB9IGNhdGNoIHtcbiAgICAgICAgLyogVFRZIG1heSBiZSBnb25lICovXG4gICAgICB9XG4gICAgfVxuICB9XG59XG4vKiBlc2xpbnQtZW5hYmxlIGN1c3RvbS1ydWxlcy9uby1zeW5jLWZzICovXG5cbmNvbnN0IENPTlNPTEVfU1RET1VUX01FVEhPRFMgPSBbXG4gICdsb2cnLFxuICAnaW5mbycsXG4gICdkZWJ1ZycsXG4gICdkaXInLFxuICAnZGlyeG1sJyxcbiAgJ2NvdW50JyxcbiAgJ2NvdW50UmVzZXQnLFxuICAnZ3JvdXAnLFxuICAnZ3JvdXBDb2xsYXBzZWQnLFxuICAnZ3JvdXBFbmQnLFxuICAndGFibGUnLFxuICAndGltZScsXG4gICd0aW1lRW5kJyxcbiAgJ3RpbWVMb2cnLFxuXSBhcyBjb25zdFxuY29uc3QgQ09OU09MRV9TVERFUlJfTUVUSE9EUyA9IFsnd2FybicsICdlcnJvcicsICd0cmFjZSddIGFzIGNvbnN0XG4iXSwibWFwcGluZ3MiOiJBQUFBLE9BQU9BLFFBQVEsTUFBTSxXQUFXO0FBQ2hDLFNBQ0VDLFNBQVMsRUFDVEMsU0FBUyxJQUFJQyxXQUFXLEVBQ3hCQyxRQUFRLEVBQ1JDLFFBQVEsRUFDUkMsU0FBUyxRQUNKLElBQUk7QUFDWCxPQUFPQyxJQUFJLE1BQU0sbUJBQW1CO0FBQ3BDLE9BQU9DLFFBQVEsTUFBTSx1QkFBdUI7QUFDNUMsT0FBT0MsS0FBSyxJQUFJLEtBQUtDLFNBQVMsUUFBUSxPQUFPO0FBQzdDLGNBQWNDLFNBQVMsUUFBUSxrQkFBa0I7QUFDakQsU0FBU0MsY0FBYyxRQUFRLCtCQUErQjtBQUM5RCxTQUFTQyxNQUFNLFFBQVEsYUFBYTtBQUNwQyxTQUFTQyxvQkFBb0IsUUFBUSx3QkFBd0I7QUFDN0QsU0FBU0MsZUFBZSxRQUFRLG9DQUFvQztBQUNwRSxTQUFTQyxlQUFlLFFBQVEsb0JBQW9CO0FBQ3BELFNBQVNDLFFBQVEsUUFBUSxrQkFBa0I7QUFDM0MsU0FBU0MsTUFBTSxRQUFRLE1BQU07QUFDN0Is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QUMsRUFBRSxPQUFPO0lBQ2ZDLFVBQVUsQ0FBQyxFQUFFLENBQUNPLEdBQUcsRUFBRSxPQUFPLEVBQUUsR0FBRyxJQUFJO0VBQ3JDLENBQUM7RUFDRCxNQUFNNkgsTUFBTSxHQUFHRCxHQUFHLENBQUNwSSxLQUFLLEtBQUssSUFBSTtFQUNqQztFQUNBO0VBQ0E7RUFDQSxJQUFJc0ksRUFBRSxHQUFHLENBQUMsQ0FBQztFQUNYLElBQUk7SUFDRjtJQUNBO0lBQ0EsSUFBSSxDQUFDRCxNQUFNLEVBQUVELEdBQUcsQ0FBQ25JLFVBQVUsR0FBRyxJQUFJLENBQUM7SUFDbkNxSSxFQUFFLEdBQUczYyxRQUFRLENBQUMsVUFBVSxFQUFFRCxXQUFXLENBQUM2YyxRQUFRLEdBQUc3YyxXQUFXLENBQUM4YyxVQUFVLENBQUM7SUFDeEUsTUFBTUMsR0FBRyxHQUFHVCxNQUFNLENBQUNVLEtBQUssQ0FBQyxJQUFJLENBQUM7SUFDOUIsS0FBSyxJQUFJQyxDQUFDLEdBQUcsQ0FBQyxFQUFFQSxDQUFDLEdBQUcsRUFBRSxFQUFFQSxDQUFDLEVBQUUsRUFBRTtNQUMzQixJQUFJL2MsUUFBUSxDQUFDMGMsRUFBRSxFQUFFRyxHQUFHLEVBQUUsQ0FBQyxFQUFFQSxHQUFHLENBQUM5TCxNQUFNLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFO0lBQ25EO0VBQ0YsQ0FBQyxDQUFDLE1BQU07SUFDTjtJQUNBO0VBQUEsQ0FDRCxTQUFTO0lBQ1IsSUFBSTJMLEVBQUUsSUFBSSxDQUFDLEVBQUU7TUFDWCxJQUFJO1FBQ0Y5YyxTQUFTLENBQUM4YyxFQUFFLENBQUM7TUFDZixDQUFDLENBQUMsTUFBTTtRQUNOO01BQUE7SUFFSjtJQUNBLElBQUksQ0FBQ0QsTUFBTSxFQUFFO01BQ1gsSUFBSTtRQUNGRCxHQUFHLENBQUNuSSxVQUFVLEdBQUcsS0FBSyxDQUFDO01BQ3pCLENBQUMsQ0FBQyxNQUFNO1FBQ047TUFBQTtJQUVKO0VBQ0Y7QUFDRjtBQUNBOztBQUVBLE1BQU0rRyxzQkFBc0IsR0FBRyxDQUM3QixLQUFLLEVBQ0wsTUFBTSxFQUNOLE9BQU8sRUFDUCxLQUFLLEVBQ0wsUUFBUSxFQUNSLE9BQU8sRUFDUCxZQUFZLEVBQ1osT0FBTyxFQUNQLGdCQUFnQixFQUNoQixVQUFVLEVBQ1YsT0FBTyxFQUNQLE1BQU0sRUFDTixTQUFTLEVBQ1QsU0FBUyxDQUNWLElBQUl4VSxLQUFLO0FBQ1YsTUFBTXlVLHNCQUFzQixHQUFHLENBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxPQUFPLENBQUMsSUFBSXpVLEtBQUsiLCJpZ25vcmVMaXN0IjpbXX0=
