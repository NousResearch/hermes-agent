import { EventEmitter } from 'node:events'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type ColumnsStream, RESIZE_SETTLE_MS, subscribeColumns } from '../lib/terminalColumns.js'

class FakeStdout extends EventEmitter implements ColumnsStream {
  columns: number | undefined = 80
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('subscribeColumns', () => {
  it('reads the new width on the leading edge of a resize', () => {
    const stdout = new FakeStdout()
    const seen: number[] = []
    const unsubscribe = subscribeColumns(stdout, c => seen.push(c))

    stdout.columns = 120
    stdout.emit('resize')

    expect(seen[0]).toBe(120)
    unsubscribe()
  })

  // Regression for #36666: Ghostty settles `columns` AFTER the final resize
  // event, so a leading-edge-only read latches a stale, larger width and the
  // status bar lays out wider than the terminal (it wraps). The trailing
  // re-read must converge to the settled width.
  it('converges to the settled width when columns updates after the final resize event', () => {
    const stdout = new FakeStdout()
    stdout.columns = 200
    const seen: number[] = []
    const unsubscribe = subscribeColumns(stdout, c => seen.push(c))

    // The final resize event arrives while columns still reports the old width…
    stdout.emit('resize')
    expect(seen.at(-1)).toBe(200)

    // …the terminal then settles narrower, with no further resize event.
    stdout.columns = 100
    vi.advanceTimersByTime(RESIZE_SETTLE_MS)

    expect(seen.at(-1)).toBe(100)
    unsubscribe()
  })

  it('coalesces a burst of resize events into a single trailing read', () => {
    const stdout = new FakeStdout()
    const seen: number[] = []
    const unsubscribe = subscribeColumns(stdout, c => seen.push(c))

    for (const w of [120, 110, 100, 90]) {
      stdout.columns = w
      stdout.emit('resize')
    }

    // One leading read per event so far.
    expect(seen).toEqual([120, 110, 100, 90])

    stdout.columns = 84
    vi.advanceTimersByTime(RESIZE_SETTLE_MS)

    // Exactly one trailing read, picking up the settled width.
    expect(seen).toEqual([120, 110, 100, 90, 84])
    unsubscribe()
  })

  it('stops reading after unsubscribe', () => {
    const stdout = new FakeStdout()
    const seen: number[] = []
    const unsubscribe = subscribeColumns(stdout, c => seen.push(c))

    unsubscribe()
    stdout.columns = 50
    stdout.emit('resize')
    vi.advanceTimersByTime(RESIZE_SETTLE_MS)

    expect(seen).toEqual([])
  })

  it('falls back to 80 columns when the width is unavailable', () => {
    const stdout = new FakeStdout()
    stdout.columns = undefined
    const seen: number[] = []
    const unsubscribe = subscribeColumns(stdout, c => seen.push(c))

    stdout.emit('resize')

    expect(seen[0]).toBe(80)
    unsubscribe()
  })
})
