import type { Cursor } from './cursor.js'
import type { Size } from './layout/geometry.js'
import type { ScrollHint } from './render-node-to-output.js'
import { type CharPool, createScreen, type HyperlinkPool, type Screen, type StylePool } from './screen.js'

export type Frame = {
  readonly screen: Screen
  readonly viewport: Size
  readonly cursor: Cursor
  /** DECSTBM scroll optimization hint (alt-screen only, null otherwise). */
  readonly scrollHint?: ScrollHint | null
  /** A ScrollBox has remaining pendingScrollDelta — schedule another frame. */
  readonly scrollDrainPending?: boolean
  /** Absolute overlay moved/resized — schedule corrective frame without prevScreen. */
  readonly absoluteOverlayMoved?: boolean
}

export function emptyFrame(
  rows: number,
  columns: number,
  stylePool: StylePool,
  charPool: CharPool,
  hyperlinkPool: HyperlinkPool
): Frame {
  return {
    screen: createScreen(0, 0, stylePool, charPool, hyperlinkPool),
    viewport: { width: columns, height: rows },
    cursor: { x: 0, y: 0, visible: true }
  }
}

export type FlickerReason = 'resize' | 'offscreen' | 'clear'

export type FrameEvent = {
  durationMs: number
  /** Phase breakdown in ms + patch count. Populated when the ink instance
   *  has frame-timing instrumentation enabled (via onFrame wiring). */
  phases?: {
    /** createRenderer output: DOM → yoga layout → screen buffer */
    renderer: number
    /** LogUpdate.render(): screen diff → Patch[] (the hot path this PR optimizes) */
    diff: number
    /** optimize(): patch merge/dedupe */
    optimize: number
    /** writeDiffToTerminal(): serialize patches → ANSI → stdout */
    write: number
    /** Pre-optimize patch count (proxy for how much changed this frame) */
    patches: number
    /** Post-optimize patch count. */
    optimizedPatches: number
    /** Bytes written to stdout this frame. */
    writeBytes: number
    /** Whether stdout.write returned false. */
    backpressure: boolean
    /** Previous stdout.write callback latency; 0 if drained before next frame. */
    prevFrameDrainMs: number
    /** yoga calculateLayout() time (runs in resetAfterCommit, before onRender) */
    yoga: number
    /** React reconcile time: scrollMutated → resetAfterCommit. 0 if no commit. */
    commit: number
    /** layoutNode() calls this frame (recursive, includes cache-hit returns) */
    yogaVisited: number
    /** measureFunc (text wrap/width) calls — the expensive part */
    yogaMeasured: number
    /** early returns via _hasL single-slot cache */
    yogaCacheHits: number
    /** total yoga Node instances alive (create - free). Growth = leak. */
    yogaLive: number
  }
  flickers: Array<{
    desiredHeight: number
    availableHeight: number
    reason: FlickerReason
  }>
}

export type Patch =
  | { type: 'stdout'; content: string }
  | { type: 'clear'; count: number }
  | {
      type: 'clearTerminal'
      reason: FlickerReason
      // Populated by log-update when a scrollback diff triggers the reset.
      debug?: { triggerY: number; prevLine: string; nextLine: string }
    }
  | { type: 'cursorHide' }
  | { type: 'cursorShow' }
  | { type: 'cursorMove'; x: number; y: number }
  | { type: 'cursorTo'; col: number }
  | { type: 'carriageReturn' }
  | { type: 'hyperlink'; uri: string }
  // Pre-serialized style transition string from StylePool.transition() —
  // cached by (fromId, toId), zero allocations after warmup.
  | { type: 'styleStr'; str: string }

export type Diff = Patch[]

/**
 * Determines whether the screen should be cleared based on the current and previous frame.
 * Returns the reason for clearing, or undefined if no clear is needed.
 *
 * Screen clearing is triggered when:
 * 1. Terminal has been resized (viewport dimensions changed) → 'resize'
 * 2. Current frame screen height exceeds available terminal rows → 'offscreen'
 * 3. Previous frame screen height exceeded available terminal rows → 'offscreen'
 */
export function shouldClearScreen(prevFrame: Frame, frame: Frame): FlickerReason | undefined {
  const didResize =
    frame.viewport.height !== prevFrame.viewport.height || frame.viewport.width !== prevFrame.viewport.width

  if (didResize) {
    return 'resize'
  }

  const currentFrameOverflows = frame.screen.height >= frame.viewport.height

  const previousFrameOverflowed = prevFrame.screen.height >= prevFrame.viewport.height

  if (currentFrameOverflows || previousFrameOverflowed) {
    return 'offscreen'
  }

  return undefined
}
