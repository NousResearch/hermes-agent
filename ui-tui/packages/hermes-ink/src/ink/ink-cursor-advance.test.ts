import { EventEmitter } from 'events'

import React from 'react'
import { describe, expect, it } from 'vitest'

import Text from './components/Text.js'
import Ink from './ink.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 40
  rows = 8
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

function makeInk() {
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

  return { ink, stdout, stdin, stderr }
}

// Closes the cursor-drift bug: when TextInput's fast-echo path writes a
// printable character directly to stdout, the hardware cursor advances by
// one cell BUT Ink's `displayCursor` cache (used as the basis for the
// next frame's relative cursor preamble) wasn't being updated. On long
// sessions an unrelated re-render (status bar timer, streaming
// reasoning, etc.) would then park the hardware cursor N cells offset
// from the actual caret — visible as "extra whitespace between my last
// typed character and the cursor block".
describe('Ink.noteExternalCursorAdvance', () => {
  it('bumps an already-tracked displayCursor by the given delta', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()

    // Seed a known parked position directly. In production this is set by
    // the cursor-park branch in onRender when a useDeclaredCursor caller
    // commits a declaration; this test bypasses React for hermeticity.
    ;(ink as unknown as { displayCursor: { x: number; y: number } | null }).displayCursor = { x: 5, y: 0 }

    ink.noteExternalCursorAdvance(3)
    expect(ink.__getDisplayCursorForTest()).toEqual({ x: 8, y: 0 })

    ink.noteExternalCursorAdvance(-1)
    expect(ink.__getDisplayCursorForTest()).toEqual({ x: 7, y: 0 })

    ink.noteExternalCursorAdvance(0, 2)
    expect(ink.__getDisplayCursorForTest()).toEqual({ x: 7, y: 2 })

    ink.unmount()
  })

  it('seeds displayCursor from frontFrame.cursor when nothing was parked', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hello'))
    ink.onRender()

    expect(ink.__getDisplayCursorForTest()).toBeNull()
    const base = ink.__getFrontFrameCursorForTest()

    ink.noteExternalCursorAdvance(4)
    expect(ink.__getDisplayCursorForTest()).toEqual({ x: base.x + 4, y: base.y })

    ink.unmount()
  })

  it('is a no-op when the delta is zero', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()

    ink.noteExternalCursorAdvance(0)
    expect(ink.__getDisplayCursorForTest()).toBeNull()

    ink.noteExternalCursorAdvance(0, 0)
    expect(ink.__getDisplayCursorForTest()).toBeNull()

    ink.unmount()
  })

  it('is a no-op on alt-screen — CSI H resets cursor every frame', () => {
    const { ink } = makeInk()

    ink.setAltScreenActive(true)
    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()
    ;(ink as unknown as { displayCursor: { x: number; y: number } | null }).displayCursor = { x: 5, y: 0 }

    ink.noteExternalCursorAdvance(3)

    expect(ink.__getDisplayCursorForTest()).toEqual({ x: 5, y: 0 })

    ink.unmount()
  })

  // Closes Copilot review feedback on PR #26717: the fast-echo path
  // defers TextInput's React `cur` state for 16ms, so during that
  // window `cursorDeclaration.relativeX` (which `useDeclaredCursor`
  // sets from `cursorLayout(display, cur, columns)`) still points at
  // the PRE-keystroke caret column. If we advanced only `displayCursor`,
  // an unrelated re-render in that window would re-run onRender's
  // cursor-park branch with the stale declaration and visually undo
  // the fast-echo's advance. We must bump BOTH so the cursor stays
  // anchored to the physical caret until React state catches up.
  it('advances the active cursorDeclaration in lock-step with displayCursor', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()

    const fakeNode = {} as unknown as Record<string, unknown>

    const inkAny = ink as unknown as {
      cursorDeclaration: { node: unknown; relativeX: number; relativeY: number } | null
      displayCursor: { x: number; y: number } | null
    }

    inkAny.cursorDeclaration = { node: fakeNode, relativeX: 7, relativeY: 0 }
    inkAny.displayCursor = { x: 12, y: 0 }

    ink.noteExternalCursorAdvance(3)

    expect(ink.__getDisplayCursorForTest()).toEqual({ x: 15, y: 0 })
    expect(ink.__getCursorDeclarationForTest()).toEqual({ node: fakeNode, relativeX: 10, relativeY: 0 })

    ink.noteExternalCursorAdvance(-1)
    expect(ink.__getDisplayCursorForTest()).toEqual({ x: 14, y: 0 })
    expect(ink.__getCursorDeclarationForTest()).toEqual({ node: fakeNode, relativeX: 9, relativeY: 0 })

    ink.unmount()
  })

  it('leaves cursorDeclaration unchanged when no declaration is active', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()

    expect(ink.__getCursorDeclarationForTest()).toBeNull()

    ink.noteExternalCursorAdvance(3)

    expect(ink.__getCursorDeclarationForTest()).toBeNull()

    ink.unmount()
  })
})
