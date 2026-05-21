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

// Cast helper instead of exposing __get*ForTest methods on production Ink —
// these are internal frame/cursor caches we only inspect from tests.
type InkPrivate = {
  displayCursor: { x: number; y: number } | null
  cursorDeclaration: { node: unknown; relativeX: number; relativeY: number } | null
  frontFrame: { cursor: { x: number; y: number } }
}
const peek = (ink: Ink): InkPrivate => ink as unknown as InkPrivate

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
    peek(ink).displayCursor = { x: 5, y: 0 }

    ink.noteExternalCursorAdvance(3)
    expect(peek(ink).displayCursor).toEqual({ x: 8, y: 0 })

    ink.noteExternalCursorAdvance(-1)
    expect(peek(ink).displayCursor).toEqual({ x: 7, y: 0 })

    ink.noteExternalCursorAdvance(0, 2)
    expect(peek(ink).displayCursor).toEqual({ x: 7, y: 2 })

    ink.unmount()
  })

  it('seeds displayCursor from frontFrame.cursor when nothing was parked', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hello'))
    ink.onRender()

    expect(peek(ink).displayCursor).toBeNull()
    const base = { x: peek(ink).frontFrame.cursor.x, y: peek(ink).frontFrame.cursor.y }

    ink.noteExternalCursorAdvance(4)
    expect(peek(ink).displayCursor).toEqual({ x: base.x + 4, y: base.y })

    ink.unmount()
  })

  it('is a no-op when the delta is zero', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()

    ink.noteExternalCursorAdvance(0)
    expect(peek(ink).displayCursor).toBeNull()

    ink.noteExternalCursorAdvance(0, 0)
    expect(peek(ink).displayCursor).toBeNull()

    ink.unmount()
  })

  it('skips displayCursor on alt-screen — CSI H resets every frame', () => {
    const { ink } = makeInk()

    ink.setAltScreenActive(true)
    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()
    peek(ink).displayCursor = { x: 5, y: 0 }

    ink.noteExternalCursorAdvance(3)

    expect(peek(ink).displayCursor).toEqual({ x: 5, y: 0 })

    ink.unmount()
  })

  // Closes Copilot follow-up on PR #26717: the default TUI wraps the
  // composer in <AlternateScreen>, so alt-screen is the production
  // path. CSI H only resets the log-update relative-move basis — the
  // declared cursor target is still consulted by onRender's alt-screen
  // park branch (`cursorPosition(row, col)` using rect + decl). So
  // cursorDeclaration MUST advance on alt-screen too, even though
  // displayCursor doesn't need to.
  it('still advances cursorDeclaration on alt-screen', () => {
    const { ink } = makeInk()

    ink.setAltScreenActive(true)
    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()

    const fakeNode = {} as unknown as Record<string, unknown>

    peek(ink).cursorDeclaration = { node: fakeNode, relativeX: 7, relativeY: 0 }
    peek(ink).displayCursor = { x: 12, y: 0 }

    ink.noteExternalCursorAdvance(3)

    // displayCursor untouched on alt-screen
    expect(peek(ink).displayCursor).toEqual({ x: 12, y: 0 })
    // declaration still advanced — onRender's alt-screen park reads this
    expect(peek(ink).cursorDeclaration).toEqual({ node: fakeNode, relativeX: 10, relativeY: 0 })

    ink.unmount()
  })

  // Closes Copilot review feedback on PR #26717: even after the
  // TextInput-level fix where layout reads `curRef.current` directly,
  // there's still a window where a fast-echo wrote to stdout but the
  // current cursor declaration on Ink (set by an earlier render's
  // useDeclaredCursor commit) points at the PRE-keystroke caret
  // column. If we advanced only `displayCursor`, an unrelated re-render
  // in that window would re-run onRender's cursor-park branch with the
  // stale declaration and visually undo the fast-echo's advance. We
  // must bump BOTH so the cursor stays anchored to the physical caret
  // until the next React commit publishes a fresh declaration
  // (computed from `curRef.current` via the cursorLayout call in
  // textInput.tsx) that supersedes the bump.
  it('advances the active cursorDeclaration in lock-step with displayCursor', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()

    const fakeNode = {} as unknown as Record<string, unknown>

    peek(ink).cursorDeclaration = { node: fakeNode, relativeX: 7, relativeY: 0 }
    peek(ink).displayCursor = { x: 12, y: 0 }

    ink.noteExternalCursorAdvance(3)

    expect(peek(ink).displayCursor).toEqual({ x: 15, y: 0 })
    expect(peek(ink).cursorDeclaration).toEqual({ node: fakeNode, relativeX: 10, relativeY: 0 })

    ink.noteExternalCursorAdvance(-1)
    expect(peek(ink).displayCursor).toEqual({ x: 14, y: 0 })
    expect(peek(ink).cursorDeclaration).toEqual({ node: fakeNode, relativeX: 9, relativeY: 0 })

    ink.unmount()
  })

  // Closes Copilot follow-up on PR #26717: the dy half of the notifier
  // contract was tested for `displayCursor` but not for
  // `cursorDeclaration.relativeY`. Newlines in fast-echoed text never
  // hit the bypass today (canFastAppendShape rejects '\n'), but `dy`
  // is part of the public API and must propagate symmetrically with
  // dx so future callers (e.g. multi-line paste shortcuts) don't get
  // a half-implemented contract.
  it('advances cursorDeclaration.relativeY when dy is non-zero', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()

    const fakeNode = {} as unknown as Record<string, unknown>

    peek(ink).cursorDeclaration = { node: fakeNode, relativeX: 2, relativeY: 1 }
    peek(ink).displayCursor = { x: 4, y: 2 }

    ink.noteExternalCursorAdvance(1, 3)

    expect(peek(ink).displayCursor).toEqual({ x: 5, y: 5 })
    expect(peek(ink).cursorDeclaration).toEqual({ node: fakeNode, relativeX: 3, relativeY: 4 })

    // Negative dy too — cursor moving up across visual rows.
    ink.noteExternalCursorAdvance(0, -2)
    expect(peek(ink).displayCursor).toEqual({ x: 5, y: 3 })
    expect(peek(ink).cursorDeclaration).toEqual({ node: fakeNode, relativeX: 3, relativeY: 2 })

    ink.unmount()
  })

  it('leaves cursorDeclaration unchanged when no declaration is active', () => {
    const { ink } = makeInk()

    ink.render(React.createElement(Text, null, 'hi'))
    ink.onRender()

    expect(peek(ink).cursorDeclaration).toBeNull()

    ink.noteExternalCursorAdvance(3)

    expect(peek(ink).cursorDeclaration).toBeNull()

    ink.unmount()
  })
})
