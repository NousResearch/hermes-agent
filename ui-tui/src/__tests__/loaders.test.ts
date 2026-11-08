import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { renderToScreen } from '../../packages/hermes-ink/src/ink/render-to-screen.js'
import { cellAtIndex } from '../../packages/hermes-ink/src/ink/screen.js'
import { ShimmerRows, shimmerSegments, subscribeShimmerClock } from '../components/loaders.js'

describe('ShimmerRows leniency (agent-authored calls)', () => {
  it('accepts a bare row COUNT and derives widths — the generated-code shape', async () => {
    const { createElement } = await import('react')

    const { screen, height } = renderToScreen(
      createElement(ShimmerRows, {
        rows: 3,
        width: 20,
        t: { color: { completionBg: '#1a1a2e', label: '#DAA520', muted: '#B8860B' } }
      }),
      30
    )

    expect(height).toBe(3)
    // Row 0 renders block cells, not a crash.
    expect(cellAtIndex(screen, 0).char).toBe('▁')
  })
})

describe('shimmerSegments', () => {
  it('always partitions the full width', () => {
    for (let phase = -40; phase < 80; phase++) {
      const [pre, band, post] = shimmerSegments(20, phase)

      expect(pre + band + post).toBe(20)
      expect(Math.min(pre, band, post)).toBeGreaterThanOrEqual(0)
    }
  })

  it('sweeps: enters from the left edge, exits off the right, then wraps', () => {
    const bandAt = (phase: number) => shimmerSegments(10, phase, 4)

    expect(bandAt(0)).toEqual([10, 0, 0]) // band fully off-left
    expect(bandAt(1)).toEqual([0, 1, 9]) // entering
    expect(bandAt(7)).toEqual([3, 4, 3]) // mid-sweep
    expect(bandAt(13)).toEqual([9, 1, 0]) // exiting
    expect(bandAt(14)).toEqual([10, 0, 0]) // gone → next cycle re-enters
    expect(bandAt(15)).toEqual([0, 1, 9])
  })

  it('negative phases (row stagger) wrap instead of vanishing', () => {
    const [pre, band, post] = shimmerSegments(10, -3, 4)

    expect(pre + band + post).toBe(10)
  })
})

// Review on #20379 (finding 5): independent 90 ms intervals per shimmer
// composition meant an idle TUI with two lazy sections repainted ~22x/sec
// forever. All compositions now share ONE clock, and the interval exists
// only while subscribers do.
describe('subscribeShimmerClock', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('drives any number of subscribers from a single interval', () => {
    const a: number[] = []
    const b: number[] = []

    const timersBefore = vi.getTimerCount()
    const unsubA = subscribeShimmerClock(p => a.push(p))
    const unsubB = subscribeShimmerClock(p => b.push(p))

    // Two subscribers, ONE new timer.
    expect(vi.getTimerCount()).toBe(timersBefore + 1)

    vi.advanceTimersByTime(300)

    // Same shared phases, in lockstep.
    expect(a.length).toBeGreaterThan(0)
    expect(a).toEqual(b)

    unsubA()
    unsubB()
  })

  it('stops the interval with the last unsubscribe', () => {
    const unsubA = subscribeShimmerClock(() => {})
    const unsubB = subscribeShimmerClock(() => {})
    const timersRunning = vi.getTimerCount()

    unsubA()
    // Still one subscriber — clock keeps running.
    expect(vi.getTimerCount()).toBe(timersRunning)

    unsubB()
    expect(vi.getTimerCount()).toBe(timersRunning - 1)
  })

  it('a late subscriber restarts the clock cleanly', () => {
    const unsubA = subscribeShimmerClock(() => {})

    unsubA()

    const seen: number[] = []
    const unsubB = subscribeShimmerClock(p => seen.push(p))

    vi.advanceTimersByTime(200)
    expect(seen.length).toBeGreaterThan(0)
    unsubB()
  })
})
