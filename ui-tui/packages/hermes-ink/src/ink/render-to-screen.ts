import noop from 'lodash-es/noop.js'
import type { ReactElement } from 'react'
import { LegacyRoot } from 'react-reconciler/constants.js'

import { logForDebugging } from '../utils/debug.js'

import { createNode, type DOMElement } from './dom.js'
import { FocusManager } from './focus.js'
import Output from './output.js'
import reconciler from './reconciler.js'
import renderNodeToOutput, { resetLayoutShifted } from './render-node-to-output.js'
import {
  cellAtIndex,
  CellWidth,
  CharPool,
  createScreen,
  HyperlinkPool,
  type Screen,
  setCellStyleId,
  StylePool
} from './screen.js'

/** Position of a match within a rendered message, relative to the message's
 *  own bounding box (row 0 = message top). Stable across scroll — to
 *  highlight on the real screen, add the message's screen-row offset. */
export type MatchPosition = {
  row: number
  col: number
  /** Number of CELLS the match spans (= query.length for ASCII, more
   *  for wide chars in the query). */
  len: number
}

// Shared across calls. Pools accumulate style/char interns — reusing them
// means later calls hit cache more. Root/container reuse saves the
// createContainer cost (~1ms). LegacyRoot: all work sync, no scheduling —
// ConcurrentRoot's scheduler backlog leaks across roots via flushSyncWork.
let root: DOMElement | undefined
let container: ReturnType<typeof reconciler.createContainer> | undefined
let stylePool: StylePool | undefined
let charPool: CharPool | undefined
let hyperlinkPool: HyperlinkPool | undefined
let output: Output | undefined

const timing = { reconcile: 0, yoga: 0, paint: 0, scan: 0, calls: 0 }
const LOG_EVERY = 20

/** Render a React element (wrapped in all contexts the component needs —
 *  caller's job) to an isolated Screen buffer at the given width. Returns
 *  the Screen + natural height (from yoga). Used for search: render ONE
 *  message, scan its Screen for the query, get exact (row, col) positions.
 *
 *  ~1-3ms per call (yoga alloc + calculateLayout + paint). The
 *  flushSyncWork cross-root leak measured ~0.0003ms/call growth — fine
 *  for on-demand single-message rendering, pathological for render-all-
 *  8k-upfront. Cache per (msg, query, width) upstream.
 *
 *  Unmounts between calls. Root/container/pools persist for reuse. */
export function renderToScreen(el: ReactElement, width: number): { screen: Screen; height: number } {
  if (!root) {
    root = createNode('ink-root')
    root.focusManager = new FocusManager(() => false)
    stylePool = new StylePool()
    charPool = new CharPool()
    hyperlinkPool = new HyperlinkPool()
    container = reconciler.createContainer(root, LegacyRoot, null, false, null, 'search-render', noop, noop, noop, noop)
  }

  const t0 = performance.now()
  reconciler.updateContainerSync(el, container, null, noop)
  reconciler.flushSyncWork()
  const t1 = performance.now()

  // Yoga layout. Root might not have a yogaNode if the tree is empty.
  root.yogaNode?.setWidth(width)
  root.yogaNode?.calculateLayout(width)
  const height = Math.ceil(root.yogaNode?.getComputedHeight() ?? 0)
  const t2 = performance.now()

  // Paint to a fresh Screen. Width = given, height = yoga's natural.
  // No alt-screen, no prevScreen (every call is fresh).
  const screen = createScreen(
    width,
    Math.max(1, height), // avoid 0-height Screen (createScreen may choke)
    stylePool!,
    charPool!,
    hyperlinkPool!
  )

  if (!output) {
    output = new Output({ width, height, stylePool: stylePool!, screen })
  } else {
    output.reset(width, height, screen)
  }

  resetLayoutShifted()
  renderNodeToOutput(root, output, { prevScreen: undefined })
  // renderNodeToOutput queues writes into Output; .get() flushes the
  // queue into the Screen's cell arrays. Without this the screen is
  // blank (constructor-zero).
  const rendered = output.get()
  const t3 = performance.now()

  // Unmount so next call gets a fresh tree. Leaves root/container/pools.
  reconciler.updateContainerSync(null, container, null, noop)
  reconciler.flushSyncWork()

  timing.reconcile += t1 - t0
  timing.yoga += t2 - t1
  timing.paint += t3 - t2

  if (++timing.calls % LOG_EVERY === 0) {
    const total = timing.reconcile + timing.yoga + timing.paint + timing.scan
    logForDebugging(
      `renderToScreen: ${timing.calls} calls · ` +
        `reconcile=${timing.reconcile.toFixed(1)}ms yoga=${timing.yoga.toFixed(1)}ms ` +
        `paint=${timing.paint.toFixed(1)}ms scan=${timing.scan.toFixed(1)}ms · ` +
        `total=${total.toFixed(1)}ms · avg ${(total / timing.calls).toFixed(2)}ms/call`
    )
  }

  return { screen: rendered, height }
}

/** Scan a Screen buffer for all occurrences of query. Returns positions
 *  relative to the buffer (row 0 = buffer top). Same cell-skip logic as
 *  applySearchHighlight (SpacerTail/SpacerHead/noSelect) so positions
 *  match what the overlay highlight would find. Case-insensitive.
 *
 *  For the side-render use: this Screen is the FULL message (natural
 *  height, not viewport-clipped). Positions are stable — to highlight
 *  on the real screen, add the message's screen offset (lo). */
export function scanPositions(screen: Screen, query: string): MatchPosition[] {
  const lq = query.toLowerCase()

  if (!lq) {
    return []
  }

  const qlen = lq.length
  const w = screen.width
  const h = screen.height
  const noSelect = screen.noSelect
  const positions: MatchPosition[] = []

  const t0 = performance.now()

  for (let row = 0; row < h; row++) {
    const rowOff = row * w
    // Same text-build as applySearchHighlight. Keep in sync — or extract
    // to a shared helper (TODO once both are stable). codeUnitToCell
    // maps indexOf positions (code units in the LOWERCASED text) to cell
    // indices in colOf — surrogate pairs (emoji) and multi-unit lowercase
    // (Turkish İ → i + U+0307) make text.length > colOf.length.
    let text = ''
    const colOf: number[] = []
    const codeUnitToCell: number[] = []

    for (let col = 0; col < w; col++) {
      const idx = rowOff + col
      const cell = cellAtIndex(screen, idx)

      if (cell.width === CellWidth.SpacerTail || cell.width === CellWidth.SpacerHead || noSelect[idx] === 1) {
        continue
      }

      const lc = cell.char.toLowerCase()
      const cellIdx = colOf.length

      for (let i = 0; i < lc.length; i++) {
        codeUnitToCell.push(cellIdx)
      }

      text += lc
      colOf.push(col)
    }

    // Non-overlapping — same advance as applySearchHighlight.
    let pos = text.indexOf(lq)

    while (pos >= 0) {
      const startCi = codeUnitToCell[pos]!
      const endCi = codeUnitToCell[pos + qlen - 1]!
      const col = colOf[startCi]!
      const endCol = colOf[endCi]! + 1
      positions.push({ row, col, len: endCol - col })
      pos = text.indexOf(lq, pos + qlen)
    }
  }

  timing.scan += performance.now() - t0

  return positions
}

/** Write CURRENT (yellow+bold+underline) at positions[currentIdx] +
 *  rowOffset. OTHER positions are NOT styled here — the scan-highlight
 *  (applySearchHighlight with null hint) does inverse for all visible
 *  matches, including these. Two-layer: scan = 'you could go here',
 *  position = 'you ARE here'. Writing inverse again here would be a
 *  no-op (withInverse idempotent) but wasted work.
 *
 *  Positions are message-relative (row 0 = message top). rowOffset =
 *  message's current screen-top (lo). Clips outside [0, height). */
export function applyPositionedHighlight(
  screen: Screen,
  stylePool: StylePool,
  positions: MatchPosition[],
  rowOffset: number,
  currentIdx: number
): boolean {
  if (currentIdx < 0 || currentIdx >= positions.length) {
    return false
  }

  const p = positions[currentIdx]!
  const row = p.row + rowOffset

  if (row < 0 || row >= screen.height) {
    return false
  }

  const transform = (id: number) => stylePool.withCurrentMatch(id)
  const rowOff = row * screen.width

  for (let col = p.col; col < p.col + p.len; col++) {
    if (col < 0 || col >= screen.width) {
      continue
    }

    const cell = cellAtIndex(screen, rowOff + col)
    setCellStyleId(screen, col, row, transform(cell.styleId))
  }

  return true
}
