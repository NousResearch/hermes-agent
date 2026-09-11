import { EventEmitter } from 'events'

import React from 'react'
import { describe, expect, it } from 'vitest'

import Text from './components/Text.js'
import Ink from './ink.js'
import { CURSOR_HOME, ERASE_SCREEN } from './termio/csi.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 20
  rows = 5
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

const tick = () => new Promise<void>(resolve => queueMicrotask(resolve))

describe('Ink resize healing', () => {
  it('heals same-dimension alt-screen resize events with an erase before repaint', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()

    const ink = new Ink({
      exitOnCtrlC: false,
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    ink.setAltScreenActive(true)
    ink.render(React.createElement(Text, null, 'hello'))
    ink.onRender()
    stdout.chunks = []

    stdout.emit('resize')
    ink.onRender()
    await tick()

    // The heal may also erase scrollback (CSI 3J interposed between 2J and H)
    // depending on which recovery path runs, so assert the invariant — screen
    // erased, then content repainted after — rather than an exact byte run.
    const out = stdout.chunks.join('')
    expect(out).toContain(ERASE_SCREEN)
    expect(out).toContain(CURSOR_HOME)
    expect(out.indexOf(ERASE_SCREEN)).toBeLessThan(out.lastIndexOf('hello'))

    ink.unmount()
  })

  // Regression for issue #18449: dragging the terminal back and forth quickly
  // emits a BURST of resize events (the single-event test above only covers one
  // tick). Each tick resets the frame buffers and arms needsEraseBeforePaint, so
  // the burst must still converge to a clean erase+repaint — a stacked event
  // must never consume the erase and leave the final paint as a partial diff
  // that lets stale glyphs survive.
  it('converges to a clean erased frame after a rapid resize burst', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()

    const ink = new Ink({
      exitOnCtrlC: false,
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    ink.setAltScreenActive(true)
    ink.render(React.createElement(Text, null, 'hello'))
    ink.onRender()
    stdout.chunks = []

    // Wobble the dimensions like a drag — widen, shrink, grow rows — then
    // settle back on the STARTING geometry. Even though the net dimensions are
    // unchanged, a host reflow during the burst can have scattered glyphs, so
    // the renderer must still heal rather than treat the end state as a no-op.
    const wobble: Array<[number, number]> = [
      [30, 5],
      [12, 9],
      [25, 4],
      [20, 5]
    ]

    for (const [columns, rows] of wobble) {
      stdout.columns = columns
      stdout.rows = rows
      stdout.emit('resize')
    }

    ink.onRender()
    await tick()

    // The heal can erase scrollback too (CSI 3J interposed), so assert the
    // semantic invariant rather than an exact byte sequence: the screen was
    // erased and the content was repainted AFTER the erase — i.e. the final
    // frame is a clean repaint, not a partial diff over drifted cells.
    const out = stdout.chunks.join('')
    expect(out).toContain(ERASE_SCREEN)
    expect(out).toContain(CURSOR_HOME)
    expect(out.indexOf(ERASE_SCREEN)).toBeLessThan(out.lastIndexOf('hello'))

    ink.unmount()
  })

  // The burst above ends on a same-dimension event; this isolates that worst
  // case on its own — a resize event whose dims equal the last known geometry
  // (the terminal restored the buffer / reflowed without a net size change)
  // must still arm the erase, because the physical screen may carry drift the
  // diff path cannot see (see log-update "drift repro").
  it('heals a same-dimension resize even when no React commit changes the tree', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()

    const ink = new Ink({
      exitOnCtrlC: false,
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    ink.setAltScreenActive(true)
    ink.render(React.createElement(Text, null, 'hello'))
    ink.onRender()
    stdout.chunks = []

    // Dimensions are identical to the initial render — the tree never changes.
    stdout.emit('resize')
    ink.onRender()
    await tick()

    const out = stdout.chunks.join('')
    expect(out).toContain(ERASE_SCREEN)
    expect(out).toContain(CURSOR_HOME)
    expect(out.indexOf(ERASE_SCREEN)).toBeLessThan(out.lastIndexOf('hello'))

    ink.unmount()
  })
})
