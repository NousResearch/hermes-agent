import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { evictInkCaches, performHeapDump } = vi.hoisted(() => ({
  evictInkCaches: vi.fn(),
  performHeapDump: vi.fn(async () => ({ heapPath: '/tmp/heap.heapsnapshot', success: true }))
}))

vi.mock('@hermes/ink', () => ({ evictInkCaches }))
vi.mock('../lib/memory.js', () => ({ performHeapDump }))

const originalMemoryUsage = process.memoryUsage
let originalAutoHeapDump: string | undefined

function mockMemoryUsage(heapUsed: number, rss = heapUsed) {
  const memoryUsage = Object.assign(
    vi.fn(() => ({
      arrayBuffers: 0,
      external: 0,
      heapTotal: heapUsed,
      heapUsed,
      rss
    })),
    { rss: vi.fn(() => rss) }
  )

  process.memoryUsage = memoryUsage as unknown as typeof process.memoryUsage
}

describe('memory monitor heap dumps', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    originalAutoHeapDump = process.env.HERMES_AUTO_HEAPDUMP
    delete process.env.HERMES_AUTO_HEAPDUMP
    mockMemoryUsage(2)
    performHeapDump.mockClear()
    evictInkCaches.mockClear()
  })

  afterEach(() => {
    vi.useRealTimers()
    process.memoryUsage = originalMemoryUsage

    if (originalAutoHeapDump === undefined) {
      delete process.env.HERMES_AUTO_HEAPDUMP
    } else {
      process.env.HERMES_AUTO_HEAPDUMP = originalAutoHeapDump
    }
  })

  it('does not write automatic heap snapshots unless explicitly enabled', async () => {
    const { startMemoryMonitor } = await import('../lib/memoryMonitor.js')
    const onHigh = vi.fn()
    const stop = startMemoryMonitor({ highBytes: 1, intervalMs: 10, onHigh })

    await vi.advanceTimersByTimeAsync(10)

    expect(performHeapDump).not.toHaveBeenCalled()
    expect(onHigh).toHaveBeenCalledWith(expect.objectContaining({ level: 'high' }), null)

    stop()
  })

  it('writes automatic heap snapshots when HERMES_AUTO_HEAPDUMP is enabled', async () => {
    process.env.HERMES_AUTO_HEAPDUMP = '1'

    const { startMemoryMonitor } = await import('../lib/memoryMonitor.js')
    const onHigh = vi.fn()
    const stop = startMemoryMonitor({ highBytes: 1, intervalMs: 10, onHigh })

    await vi.advanceTimersByTimeAsync(10)

    expect(performHeapDump).toHaveBeenCalledWith('auto-high')
    expect(onHigh).toHaveBeenCalledWith(
      expect.objectContaining({ level: 'high' }),
      expect.objectContaining({ heapPath: '/tmp/heap.heapsnapshot', success: true })
    )

    stop()
  })
})
