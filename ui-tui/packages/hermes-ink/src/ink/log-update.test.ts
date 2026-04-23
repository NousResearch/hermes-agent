import { describe, expect, it } from 'vitest'

import type { Frame } from './frame.js'
import { LogUpdate } from './log-update.js'
import {
  CellWidth,
  CharPool,
  createScreen,
  HyperlinkPool,
  type Screen,
  setCellAt,
  StylePool
} from './screen.js'

/**
 * Contract tests for LogUpdate.render() — the diff-to-ANSI path that owns
 * whether the terminal picks up each React commit correctly.
 *
 * These tests pin down a few load-bearing invariants so that any fix for
 * the "scattered letters after rapid resize" artifact in xterm.js hosts
 * can be grounded against them.
 */

const stylePool = new StylePool()
const charPool = new CharPool()
const hyperlinkPool = new HyperlinkPool()

const mkScreen = (w: number, h: number) => createScreen(w, h, stylePool, charPool, hyperlinkPool)

const paint = (screen: Screen, y: number, text: string) => {
  for (let x = 0; x < text.length; x++) {
    setCellAt(screen, x, y, {
      char: text[x]!,
      styleId: stylePool.none,
      width: CellWidth.Narrow,
      hyperlink: undefined
    })
  }
}

const mkFrame = (screen: Screen, viewportW: number, viewportH: number): Frame => ({
  screen,
  viewport: { width: viewportW, height: viewportH },
  cursor: { x: 0, y: 0, visible: true }
})

const stdoutOnly = (diff: ReturnType<LogUpdate['render']>) =>
  diff
    .filter(p => p.type === 'stdout')
    .map(p => (p as { type: 'stdout'; content: string }).content)
    .join('')

describe('LogUpdate.render diff contract', () => {
  it('emits only changed cells when most rows match', () => {
    const w = 20
    const h = 4
    const prev = mkScreen(w, h)
    paint(prev, 0, 'HELLO')
    paint(prev, 1, 'WORLD')
    paint(prev, 2, 'STAYSHERE')

    const next = mkScreen(w, h)
    paint(next, 0, 'HELLO')
    paint(next, 1, 'CHANGE')
    paint(next, 2, 'STAYSHERE')
    next.damage = { x: 0, y: 0, width: w, height: h }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, w, h), mkFrame(next, w, h), true, false)

    const written = stdoutOnly(diff)
    expect(written).toContain('CHANGE')
    expect(written).not.toContain('HELLO')
    expect(written).not.toContain('STAYSHERE')
  })

  it('width change emits a clearTerminal patch before repainting', () => {
    const prevW = 20
    const nextW = 15
    const h = 3

    const prev = mkScreen(prevW, h)
    paint(prev, 0, 'thiswaswiderrow')

    const next = mkScreen(nextW, h)
    paint(next, 0, 'shorterrownow')
    next.damage = { x: 0, y: 0, width: nextW, height: h }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, prevW, h), mkFrame(next, nextW, h), true, false)

    expect(diff.some(p => p.type === 'clearTerminal')).toBe(true)
    expect(stdoutOnly(diff)).toContain('shorterrownow')
  })

  it('drift repro: if terminal has content that prev.screen does not know about, diff leaves it orphaned', () => {
    // Simulates prev/terminal desync: the physical terminal has STALE
    // content at row 2 from a prior frame that was never reconciled into
    // prev.screen. next.screen is blank at row 2. Diff finds prev==next
    // (both blank at row 2), emits nothing → the stale content survives
    // on the terminal as an artifact.
    //
    // This is the load-bearing theory for the rapid-resize scattered-letter
    // bug: whenever the ink renderer believes prev.screen is authoritative
    // but the physical terminal was mutated out-of-band (resize-induced
    // reflow writing past the prev-frame's tracked cells), those cells
    // drift and artifacts appear at that row on subsequent frames.
    const w = 20
    const h = 3
    const prevAsInk = mkScreen(w, h)
    paint(prevAsInk, 0, 'same')
    // row 2 in prevAsInk is blank — but pretend the terminal has stale
    // characters there. ink has no way to know.
    const terminalReally = mkScreen(w, h)
    paint(terminalReally, 0, 'same')
    paint(terminalReally, 2, 'orphaned')

    const next = mkScreen(w, h)
    paint(next, 0, 'same')
    next.damage = { x: 0, y: 0, width: w, height: h }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prevAsInk, w, h), mkFrame(next, w, h), true, false)

    const written = stdoutOnly(diff)
    expect(written).not.toContain('orphaned')
    expect(diff.some(p => p.type === 'clearTerminal')).toBe(false)
    // Verdict: in this configuration the renderer cannot heal the drift.
    // The only recovery path from ink's side is fullResetSequence — which
    // triggers only on viewport resize or scrollback-change detection,
    // neither of which fires on a pure drift. A fix has to either (a)
    // defensively emit a full repaint on every xterm.js frame where
    // prevFrameContaminated is set, or (b) close the drift window at the
    // renderer level so the in-memory prev.screen cannot diverge.
  })
})
