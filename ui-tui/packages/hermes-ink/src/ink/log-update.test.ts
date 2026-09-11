import { describe, expect, it } from 'vitest'

import type { Frame } from './frame.js'
import { LogUpdate } from './log-update.js'
import { CellWidth, CharPool, createScreen, HyperlinkPool, type Screen, setCellAt, StylePool } from './screen.js'

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

const mkFrame = (screen: Screen, viewportW: number, viewportH: number, cursorY = 0): Frame => ({
  screen,
  viewport: { width: viewportW, height: viewportH },
  cursor: { x: 0, y: cursorY, visible: true }
})

const stdoutOnly = (diff: ReturnType<LogUpdate['render']>) =>
  diff
    .filter(p => p.type === 'stdout')
    .map(p => (p as { type: 'stdout'; content: string }).content)
    .join('')

const ESC = '\u001b'
const hasDecstbm = (text: string) => new RegExp(`${ESC}\\[\\d+;\\d+r`).test(text)

describe('LogUpdate.render diff contract', () => {
  it.each([
    ['adds a combining mark', 'ร', 'ร้', 0],
    ['removes a combining mark', 'ร้', 'ร', 1],
    ['replaces a combining mark', 'ร้', 'ร่', 2]
  ])('clears a narrow grapheme cell before it %s', (_label, before, after, x) => {
    const prev = mkScreen(4, 1)
    const next = mkScreen(4, 1)

    setCellAt(prev, x, 0, {
      char: before,
      styleId: stylePool.none,
      width: CellWidth.Narrow,
      hyperlink: undefined
    })
    setCellAt(next, x, 0, {
      char: after,
      styleId: stylePool.none,
      width: CellWidth.Narrow,
      hyperlink: undefined
    })
    next.damage = { x, y: 0, width: 1, height: 1 }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, 4, 1), mkFrame(next, 4, 1), true, false)
    const writeIndex = diff.findIndex(part => part.type === 'stdout' && part.content === after)

    expect(writeIndex).toBeGreaterThan(0)
    expect(diff.slice(0, writeIndex)).toContainEqual({ type: 'stdout', content: ' ' })
    expect(diff.slice(0, writeIndex)).toContainEqual({ type: 'cursorTo', col: x + 1 })
  })

  it('does not clear before replacing ordinary single-codepoint text', () => {
    const prev = mkScreen(4, 1)
    const next = mkScreen(4, 1)
    paint(prev, 0, 'a')
    paint(next, 0, 'b')
    next.damage = { x: 0, y: 0, width: 1, height: 1 }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, 4, 1), mkFrame(next, 4, 1), true, false)

    expect(stdoutOnly(diff)).toBe('b')
    expect(diff.some(part => part.type === 'cursorTo')).toBe(false)
    expect(diff.some(part => part.type === 'stdout' && part.content === ' ')).toBe(false)
  })

  it('does not clear before painting a grapheme into an empty cell', () => {
    const prev = mkScreen(4, 1)
    const next = mkScreen(4, 1)

    setCellAt(next, 1, 0, {
      char: 'ร้',
      styleId: stylePool.none,
      width: CellWidth.Narrow,
      hyperlink: undefined
    })
    next.damage = { x: 1, y: 0, width: 1, height: 1 }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, 4, 1), mkFrame(next, 4, 1), true, false)

    expect(stdoutOnly(diff)).toBe('ร้')
    expect(diff.some(part => part.type === 'cursorTo')).toBe(false)
  })

  it('does not clear when only a grapheme cell style changes', () => {
    const prev = mkScreen(4, 1)
    const next = mkScreen(4, 1)
    const style = stylePool.intern([{ code: '\u001b[32m', endCode: '\u001b[39m' }])

    setCellAt(prev, 1, 0, { char: 'ร้', styleId: stylePool.none, width: CellWidth.Narrow, hyperlink: undefined })
    setCellAt(next, 1, 0, { char: 'ร้', styleId: style, width: CellWidth.Narrow, hyperlink: undefined })
    next.damage = { x: 1, y: 0, width: 1, height: 1 }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, 4, 1), mkFrame(next, 4, 1), true, false)

    expect(stdoutOnly(diff)).toBe('ร้')
    expect(diff.some(part => part.type === 'cursorTo')).toBe(false)
  })

  it('resets active hyperlink and style before pre-clearing a grapheme cell', () => {
    const prev = mkScreen(4, 1)
    const next = mkScreen(4, 1)
    const oldStyle = stylePool.intern([{ code: '\u001b[31m', endCode: '\u001b[39m' }])
    const newStyle = stylePool.intern([{ code: '\u001b[32m', endCode: '\u001b[39m' }])

    setCellAt(prev, 0, 0, { char: 'x', styleId: stylePool.none, width: CellWidth.Narrow, hyperlink: undefined })
    setCellAt(next, 0, 0, { char: 'y', styleId: oldStyle, width: CellWidth.Narrow, hyperlink: 'https://old.example' })
    setCellAt(prev, 1, 0, { char: 'ร้', styleId: oldStyle, width: CellWidth.Narrow, hyperlink: 'https://old.example' })
    setCellAt(next, 1, 0, { char: 'ร', styleId: newStyle, width: CellWidth.Narrow, hyperlink: undefined })
    next.damage = { x: 0, y: 0, width: 2, height: 1 }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, 4, 1), mkFrame(next, 4, 1), true, false)
    const writeIndex = diff.findIndex(part => part.type === 'stdout' && part.content === 'ร')
    const clearIndex = diff.findIndex(part => part.type === 'stdout' && part.content === ' ')
    const resetStyle = stylePool.transition(oldStyle, stylePool.none)
    const styleResetIndex = diff.findIndex(part => part.type === 'styleStr' && part.str === resetStyle)
    const closeIndex = diff.findIndex(part => part.type === 'hyperlink' && part.uri === '')

    expect(writeIndex).toBeGreaterThan(clearIndex)
    expect(clearIndex).toBeGreaterThan(-1)
    expect(styleResetIndex).toBeGreaterThan(-1)
    expect(styleResetIndex).toBeLessThan(clearIndex)
    expect(closeIndex).toBeGreaterThan(-1)
    expect(closeIndex).toBeLessThan(clearIndex)
    expect(diff.slice(clearIndex, writeIndex)).toContainEqual({ type: 'cursorTo', col: 2 })
  })

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

  it('height growth emits a clearTerminal patch before repainting', () => {
    const w = 20
    const prevH = 3
    const nextH = 6

    const prev = mkScreen(w, prevH)
    paint(prev, 0, 'old rows')

    const next = mkScreen(w, nextH)
    paint(next, 0, 'new rows')
    next.damage = { x: 0, y: 0, width: w, height: nextH }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, w, prevH), mkFrame(next, w, nextH), true, false)

    expect(diff.some(p => p.type === 'clearTerminal')).toBe(true)
    expect(stdoutOnly(diff)).toContain('newrows')
  })

  it('drift repro: identical prev/next emits no heal, even when the physical terminal is stale', () => {
    // Load-bearing theory for the rapid-resize scattered-letter bug: if the
    // physical terminal has stale cells that prev.screen doesn't know about
    // (e.g. resize-induced reflow wrote past ink's tracked range), the
    // renderer has no signal to heal them. LogUpdate.render only sees
    // prev/next — no view of the physical terminal — so when prev==next,
    // it emits nothing and any orphaned glyphs survive.
    //
    // The fix path is upstream of this diff: either (a) defensively
    // full-repaint on xterm.js frames where prevFrameContaminated is set,
    // or (b) close the drift window so prev.screen cannot diverge.
    const w = 20
    const h = 3

    const prev = mkScreen(w, h)
    paint(prev, 0, 'same')

    const next = mkScreen(w, h)
    paint(next, 0, 'same')
    next.damage = { x: 0, y: 0, width: w, height: h }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, w, h), mkFrame(next, w, h), true, false)

    expect(stdoutOnly(diff)).toBe('')
    expect(diff.some(p => p.type === 'clearTerminal')).toBe(false)
  })

  it('ignores main-screen scrollback-only changes instead of resetting repeatedly', () => {
    const w = 20
    const viewportH = 5
    const h = 8

    const prev = mkScreen(w, h)
    paint(prev, 0, 'timer 1s')
    paint(prev, 6, 'visible prompt')

    const next = mkScreen(w, h)
    paint(next, 0, 'timer 2s')
    paint(next, 6, 'visible prompt')
    next.damage = { x: 0, y: 0, width: w, height: h }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, w, viewportH, h), mkFrame(next, w, viewportH, h), false, false)

    expect(diff.some(p => p.type === 'clearTerminal')).toBe(false)
    expect(stdoutOnly(diff)).not.toContain('timer2s')
  })

  it('keeps alt-screen full reset for unreachable scrollback row changes', () => {
    const w = 20
    const viewportH = 5
    const h = 8

    const prev = mkScreen(w, h)
    paint(prev, 0, 'timer 1s')
    paint(prev, 6, 'visible prompt')

    const next = mkScreen(w, h)
    paint(next, 0, 'timer 2s')
    paint(next, 6, 'visible prompt')
    next.damage = { x: 0, y: 0, width: w, height: h }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(mkFrame(prev, w, viewportH, h), mkFrame(next, w, viewportH, h), true, false)

    expect(diff.some(p => p.type === 'clearTerminal')).toBe(true)
    expect(stdoutOnly(diff)).toContain('timer2s')
  })

  it('keeps DECSTBM fast-path when scroll region stays above bottom row', () => {
    const w = 12
    const h = 6
    const prev = mkScreen(w, h)
    const next = mkScreen(w, h)

    paint(prev, 1, 'row one')
    paint(next, 1, 'row one')

    const prevFrame = mkFrame(prev, w, h)

    const nextFrame: Frame = {
      ...mkFrame(next, w, h),
      scrollHint: { top: 1, bottom: 4, delta: 1 }
    }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(prevFrame, nextFrame, true, true)

    expect(hasDecstbm(stdoutOnly(diff))).toBe(true)
  })

  it('skips DECSTBM when scroll region touches the bottom row', () => {
    const w = 12
    const h = 6
    const prev = mkScreen(w, h)
    const next = mkScreen(w, h)

    paint(prev, 1, 'row one')
    paint(next, 1, 'row one')

    const prevFrame = mkFrame(prev, w, h)

    const nextFrame: Frame = {
      ...mkFrame(next, w, h),
      scrollHint: { top: 1, bottom: 5, delta: 1 }
    }

    const log = new LogUpdate({ isTTY: true, stylePool })
    const diff = log.render(prevFrame, nextFrame, true, true)

    expect(hasDecstbm(stdoutOnly(diff))).toBe(false)
  })
})
