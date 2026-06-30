import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// memory.js performs real heap dumps / fs work — stub it so the monitor's
// dump path is a no-op in tests.
vi.mock('../lib/memory.js', () => ({
  performHeapDump: vi.fn(async () => null)
}))

// @hermes/ink is dynamically imported only on the dump path; stub the eviction.
vi.mock('@hermes/ink', () => ({ evictInkCaches: vi.fn() }))

vi.mock('node:v8', () => ({
  getHeapStatistics: vi.fn(() => ({ heap_size_limit: 8 * 1024 ** 3 }))
}))

import { startMemoryMonitor } from '../lib/memoryMonitor.js'

const GB = 1024 ** 3
const MB = 1024 ** 2

describe('startMemoryMonitor thresholds (#34095)', () => {
  let stop: (() => void) | undefined

  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    stop?.()
    stop = undefined
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  const withHeap = (heapUsed: number, rss = heapUsed) =>
    vi.spyOn(process, 'memoryUsage').mockReturnValue({
      arrayBuffers: 0,
      external: 0,
      heapTotal: heapUsed,
      heapUsed,
      rss
    } as NodeJS.MemoryUsage)

  it('does NOT fire onCritical at 2.5GB when the heap ceiling is 8GB', async () => {
    // The old hardcoded 2.5GB constant killed the process at ~31% of the real
    // ceiling. With relative thresholds (~88%), 2.5GB is well within normal.
    const onCritical = vi.fn()
    withHeap(2.5 * GB)
    stop = startMemoryMonitor({ intervalMs: 1, onCritical })

    await vi.advanceTimersByTimeAsync(5)

    expect(onCritical).not.toHaveBeenCalled()
  })

  it('fires onCritical only near the configured ceiling', async () => {
    const onCritical = vi.fn()
    // Explicit small ceiling-derived thresholds via override to keep the test
    // independent of the host V8 heap_size_limit.
    withHeap(7.5 * GB)
    stop = startMemoryMonitor({ criticalBytes: 7 * GB, highBytes: 5 * GB, intervalMs: 1, onCritical })

    await vi.advanceTimersByTimeAsync(5)

    expect(onCritical).toHaveBeenCalledTimes(1)
  })

  it('fires onWarn once on fast sub-threshold heap growth, then re-arms', async () => {
    const onWarn = vi.fn()
    // Start low, then jump >150MB across a tick while above the 600MB floor and
    // below `high` — the silent-death regime.
    const spy = withHeap(100 * MB)
    stop = startMemoryMonitor({ highBytes: 2 * GB, intervalMs: 1, onWarn, warnBytes: 600 * MB })

    await vi.advanceTimersByTimeAsync(2) // seed lastHeap at 100MB, below floor
    expect(onWarn).not.toHaveBeenCalled()

    spy.mockReturnValue({ arrayBuffers: 0, external: 0, heapTotal: 800 * MB, heapUsed: 800 * MB, rss: 800 * MB } as NodeJS.MemoryUsage)
    await vi.advanceTimersByTimeAsync(2) // jumped 700MB → above floor + steep
    expect(onWarn).toHaveBeenCalledTimes(1)

    // Stays elevated but not re-firing.
    await vi.advanceTimersByTimeAsync(2)
    expect(onWarn).toHaveBeenCalledTimes(1)

    // Falls back below the floor → re-armed, then climbs again → fires again.
    spy.mockReturnValue({ arrayBuffers: 0, external: 0, heapTotal: 100 * MB, heapUsed: 100 * MB, rss: 100 * MB } as NodeJS.MemoryUsage)
    await vi.advanceTimersByTimeAsync(2)
    spy.mockReturnValue({ arrayBuffers: 0, external: 0, heapTotal: 800 * MB, heapUsed: 800 * MB, rss: 800 * MB } as NodeJS.MemoryUsage)
    await vi.advanceTimersByTimeAsync(2)
    expect(onWarn).toHaveBeenCalledTimes(2)
  })

  it('does not warn on slow growth below the steep-growth step', async () => {
    const onWarn = vi.fn()
    const spy = withHeap(650 * MB)
    stop = startMemoryMonitor({ highBytes: 2 * GB, intervalMs: 1, onWarn, warnBytes: 600 * MB })

    await vi.advanceTimersByTimeAsync(2)
    // +50MB per tick — above the floor but gentle, not a render-tree blowup.
    spy.mockReturnValue({ arrayBuffers: 0, external: 0, heapTotal: 700 * MB, heapUsed: 700 * MB, rss: 700 * MB } as NodeJS.MemoryUsage)
    await vi.advanceTimersByTimeAsync(2)

    expect(onWarn).not.toHaveBeenCalled()
  })
})
