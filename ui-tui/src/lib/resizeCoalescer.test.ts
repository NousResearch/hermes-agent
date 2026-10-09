import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createResizeCoalescer } from './resizeCoalescer.js'

describe('createResizeCoalescer', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('reflows immediately on the first event (leading edge)', () => {
    const reflow = vi.fn()
    const coalescer = createResizeCoalescer(reflow, 32)

    coalescer.schedule()

    expect(reflow).toHaveBeenCalledTimes(1)
  })

  it('collapses a rapid burst into one leading + one trailing reflow', () => {
    const reflow = vi.fn()
    const coalescer = createResizeCoalescer(reflow, 32)

    // A drag: five events inside one 32ms window.
    coalescer.schedule()

    for (let i = 0; i < 4; i++) {
      vi.advanceTimersByTime(5)
      coalescer.schedule()
    }

    // Only the leading edge has fired mid-burst.
    expect(reflow).toHaveBeenCalledTimes(1)

    // The trailing edge lands the final width once the window settles.
    vi.advanceTimersByTime(32)
    expect(reflow).toHaveBeenCalledTimes(2)
  })

  it('reflows immediately again once the interval has elapsed', () => {
    const reflow = vi.fn()
    const coalescer = createResizeCoalescer(reflow, 32)

    coalescer.schedule()
    expect(reflow).toHaveBeenCalledTimes(1)

    // Gap longer than the window — the next event is a fresh leading edge.
    vi.advanceTimersByTime(40)
    coalescer.schedule()

    expect(reflow).toHaveBeenCalledTimes(2)
  })

  it('cancel() drops a pending trailing reflow', () => {
    const reflow = vi.fn()
    const coalescer = createResizeCoalescer(reflow, 32)

    coalescer.schedule() // leading
    vi.advanceTimersByTime(5)
    coalescer.schedule() // schedules a trailing reflow
    coalescer.cancel()

    vi.advanceTimersByTime(100)
    expect(reflow).toHaveBeenCalledTimes(1)
  })

  it('continuous dragging reflows about once per interval, not per event', () => {
    const reflow = vi.fn()
    const coalescer = createResizeCoalescer(reflow, 30)

    // 30 events over 300ms (one every 10ms) — a sustained drag.
    for (let i = 0; i < 30; i++) {
      coalescer.schedule()
      vi.advanceTimersByTime(10)
    }

    // Flush the final trailing reflow.
    vi.advanceTimersByTime(30)

    // ~300ms / 30ms ≈ 10 reflows, not 30. Bound it loosely to stay robust.
    expect(reflow.mock.calls.length).toBeLessThanOrEqual(12)
    expect(reflow.mock.calls.length).toBeGreaterThanOrEqual(8)
  })
})
