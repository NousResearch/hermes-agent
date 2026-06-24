import { afterEach, describe, expect, it, vi } from 'vitest'

import { HEARTBEAT_INTERVAL_MS, startHeartbeat } from '../lib/heartbeat.js'

describe('heartbeat (Stage 3 frozen-detection liveness)', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('is a no-op when no heartbeat file is configured', () => {
    const touch = vi.fn()
    const stop = startHeartbeat(undefined, 1000, touch)
    stop()
    expect(touch).not.toHaveBeenCalled()
  })

  it('is a no-op for an empty/whitespace path', () => {
    const touch = vi.fn()
    startHeartbeat('   ', 1000, touch)()
    expect(touch).not.toHaveBeenCalled()
  })

  it('beats immediately on start (no initial stale window)', () => {
    const touch = vi.fn()
    const stop = startHeartbeat('/tmp/hb', 1000, touch)
    expect(touch).toHaveBeenCalledTimes(1)
    expect(touch).toHaveBeenCalledWith('/tmp/hb')
    stop()
  })

  it('beats on each interval tick', () => {
    vi.useFakeTimers()
    const touch = vi.fn()
    const stop = startHeartbeat('/tmp/hb', 1000, touch)
    expect(touch).toHaveBeenCalledTimes(1) // immediate
    vi.advanceTimersByTime(3000)
    expect(touch).toHaveBeenCalledTimes(4) // immediate + 3 ticks
    stop()
  })

  it('stops beating after stop()', () => {
    vi.useFakeTimers()
    const touch = vi.fn()
    const stop = startHeartbeat('/tmp/hb', 1000, touch)
    stop()
    vi.advanceTimersByTime(5000)
    expect(touch).toHaveBeenCalledTimes(1) // only the immediate one
  })

  it('a throwing touch never escapes the timer (renderer must not crash on a bad fs)', () => {
    vi.useFakeTimers()
    const touch = vi.fn(() => {
      throw new Error('disk gone')
    })
    // start itself beats immediately; must not throw
    const stop = startHeartbeat('/tmp/hb', 1000, touch)
    expect(() => vi.advanceTimersByTime(2000)).not.toThrow()
    stop()
  })

  it('exposes a sane default interval', () => {
    expect(HEARTBEAT_INTERVAL_MS).toBeGreaterThan(0)
    expect(HEARTBEAT_INTERVAL_MS).toBeLessThanOrEqual(30_000)
  })
})
