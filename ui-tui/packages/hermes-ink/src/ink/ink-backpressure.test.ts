import { EventEmitter } from 'events'

import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import Text from './components/Text.js'
import { MAX_COALESCED_BACKPRESSURE_FRAMES } from './constants.js'
import Ink from './ink.js'

// Regression for issue #31486 (stdout-backpressure strand): when the
// previous frame's stdout.write has not drained (the terminal parser is
// overwhelmed — a wide CR+LF burst on a high-context session), the renderer
// must COALESCE rather than pile another write on the backed-up pipe. Piling
// writes keeps the macrotask queue hot and starves the stdin 'readable'
// callback, which is the observed freeze. The coalesce must be bounded: after
// MAX_COALESCED_BACKPRESSURE_FRAMES skipped frames it forces a write through
// so a terminal whose drain callback never fires can't wedge the renderer.

/**
 * A TTY whose write() reports backpressure (returns false) and WITHHOLDS the
 * drain callback until fireDrain() is called — simulating a wedged terminal
 * parser. Each write records its drain callback so the test controls timing.
 */
class WedgedTty extends EventEmitter {
  chunks: string[] = []
  columns = 20
  rows = 5
  isTTY = true
  private pendingDrains: Array<(err?: Error | null) => void> = []

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))

    if (cb) {
      // Hold the callback — do NOT fire it. This leaves the renderer's
      // pendingWriteStart non-null, the backpressure signal it coalesces on.
      this.pendingDrains.push(cb)
    }

    // Report backpressure.
    return false
  }

  /** Fire all withheld drain callbacks, simulating the pipe recovering. */
  fireDrain(): void {
    const drains = this.pendingDrains
    this.pendingDrains = []

    for (const cb of drains) {
      cb()
    }
  }

  get pendingDrainCount(): number {
    return this.pendingDrains.length
  }
}

/** A normal fast TTY: write succeeds and drains synchronously. */
class FastTty extends EventEmitter {
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

const makeInk = (stdout: WedgedTty | FastTty) => {
  const stdin = new EventEmitter() as unknown as NodeJS.ReadStream
  const stderr = new FastTty()

  return new Ink({
    exitOnCtrlC: false,
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin,
    stdout: stdout as unknown as NodeJS.WriteStream
  })
}

describe('Ink stdout backpressure coalescing (issue #31486)', () => {
  it('coalesces frames while the previous write has not drained', () => {
    vi.useFakeTimers()

    try {
      const stdout = new WedgedTty()
      const ink = makeInk(stdout)

      ink.render(React.createElement(Text, null, 'hello'))
      ink.onRender()

      // First frame wrote (and reported backpressure; drain withheld).
      const writesAfterFirst = stdout.chunks.length
      expect(writesAfterFirst).toBeGreaterThan(0)
      expect(stdout.pendingDrainCount).toBe(1)

      // Subsequent renders while the write is still pending must coalesce —
      // no new bytes written, a retry timer scheduled instead.
      ink.render(React.createElement(Text, null, 'world'))
      ink.onRender()
      expect(stdout.chunks.length).toBe(writesAfterFirst)

      ink.onRender()
      expect(stdout.chunks.length).toBe(writesAfterFirst)

      ink.unmount()
    } finally {
      vi.useRealTimers()
    }
  })

  it('resumes writing once the wedged pipe drains', () => {
    vi.useFakeTimers()

    try {
      const stdout = new WedgedTty()
      const ink = makeInk(stdout)

      ink.render(React.createElement(Text, null, 'hello'))
      ink.onRender()
      const writesAfterFirst = stdout.chunks.length

      // Backed up: this render coalesces.
      ink.render(React.createElement(Text, null, 'changed'))
      ink.onRender()
      expect(stdout.chunks.length).toBe(writesAfterFirst)

      // Pipe recovers — drain callback fires, clearing pendingWriteStart.
      stdout.fireDrain()

      // The retry tick now finds the pipe drained and writes the pending frame.
      vi.runAllTimers()
      expect(stdout.chunks.length).toBeGreaterThan(writesAfterFirst)

      ink.unmount()
    } finally {
      vi.useRealTimers()
    }
  })

  it('forces a write through after the coalesce ceiling so it never wedges forever', () => {
    vi.useFakeTimers()

    try {
      const stdout = new WedgedTty()
      const ink = makeInk(stdout)

      ink.render(React.createElement(Text, null, 'hello'))
      ink.onRender()
      const writesAfterFirst = stdout.chunks.length

      // Mark content dirty and drive renders. The drain callback NEVER fires
      // (pendingDrainCount stays > 0). After MAX_COALESCED_BACKPRESSURE_FRAMES
      // coalesced retries, the renderer must force a write through.
      ink.render(React.createElement(Text, null, 'forced'))

      // Drive enough retry ticks to exceed the ceiling.
      for (let i = 0; i <= MAX_COALESCED_BACKPRESSURE_FRAMES + 2; i++) {
        vi.advanceTimersByTime(4)
      }

      // A write was forced through despite the never-firing drain callback.
      expect(stdout.chunks.length).toBeGreaterThan(writesAfterFirst)

      ink.unmount()
    } finally {
      vi.useRealTimers()
    }
  })

  it('never coalesces on a fast terminal that drains synchronously', () => {
    const stdout = new FastTty()
    const ink = makeInk(stdout)

    ink.render(React.createElement(Text, null, 'a'))
    ink.onRender()
    const afterA = stdout.chunks.length
    expect(afterA).toBeGreaterThan(0)

    // Each changed render writes immediately — synchronous drain clears the
    // backpressure signal before the next frame, so nothing is coalesced.
    ink.render(React.createElement(Text, null, 'b'))
    ink.onRender()
    expect(stdout.chunks.length).toBeGreaterThan(afterA)

    ink.unmount()
  })
})
