import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { EventEmitter } from 'node:events'
import { startParentWatchdog } from '../lib/gracefulExit.js'

/**
 * Create a mock stdin (a Readable-like EventEmitter with once/removeListener).
 */
function mockStdin() {
  const ee = new EventEmitter()
  // Mimic the parts of process.stdin we use
  return ee as unknown as NodeJS.ReadStream
}

describe('startParentWatchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  // ── Stdin-based detection (primary) ──────────────────────────────

  it('calls onOrphaned when stdin emits end', () => {
    const stdin = mockStdin()
    const onOrphaned = vi.fn()

    // Temporarily replace process.stdin
    const origStdin = process.stdin
    Object.defineProperty(process, 'stdin', { value: stdin, configurable: true })

    try {
      const stop = startParentWatchdog(onOrphaned)
      expect(onOrphaned).not.toHaveBeenCalled()

      // Simulate parent closing the pipe
      stdin.emit('end')

      expect(onOrphaned).toHaveBeenCalledTimes(1)
      expect(onOrphaned.mock.calls[0][0]).toContain('stdin pipe closed')

      stop()
    } finally {
      Object.defineProperty(process, 'stdin', { value: origStdin, configurable: true })
    }
  })

  it('calls onOrphaned when stdin emits close', () => {
    const stdin = mockStdin()
    const onOrphaned = vi.fn()
    const origStdin = process.stdin
    Object.defineProperty(process, 'stdin', { value: stdin, configurable: true })

    try {
      const stop = startParentWatchdog(onOrphaned)

      stdin.emit('close')

      expect(onOrphaned).toHaveBeenCalledTimes(1)
      expect(onOrphaned.mock.calls[0][0]).toContain('stdin stream closed')

      stop()
    } finally {
      Object.defineProperty(process, 'stdin', { value: origStdin, configurable: true })
    }
  })

  it('fires only once even if both end and close emit', () => {
    const stdin = mockStdin()
    const onOrphaned = vi.fn()
    const origStdin = process.stdin
    Object.defineProperty(process, 'stdin', { value: stdin, configurable: true })

    try {
      const stop = startParentWatchdog(onOrphaned)

      stdin.emit('end')
      stdin.emit('close')

      expect(onOrphaned).toHaveBeenCalledTimes(1)

      stop()
    } finally {
      Object.defineProperty(process, 'stdin', { value: origStdin, configurable: true })
    }
  })

  // ── Ppid fallback detection ──────────────────────────────────────

  it('calls onOrphaned when ppid changes to 1 (fallback)', () => {
    const stdin = mockStdin()
    const onOrphaned = vi.fn()
    let ppid = 1234
    const originalPpid = process.ppid
    const origStdin = process.stdin

    Object.defineProperty(process, 'ppid', { get: () => ppid, configurable: true })
    Object.defineProperty(process, 'stdin', { value: stdin, configurable: true })

    try {
      const stop = startParentWatchdog(onOrphaned, 1000)
      expect(onOrphaned).not.toHaveBeenCalled()

      // Simulate parent exit → reparented to init
      ppid = 1
      vi.advanceTimersByTime(1000)

      expect(onOrphaned).toHaveBeenCalledTimes(1)
      expect(onOrphaned.mock.calls[0][0]).toContain('ppid changed from 1234 to 1')

      stop()
    } finally {
      Object.defineProperty(process, 'ppid', { get: () => originalPpid, configurable: true })
      Object.defineProperty(process, 'stdin', { value: origStdin, configurable: true })
    }
  })

  it('calls onOrphaned when ppid changes to a different non-init PID', () => {
    const stdin = mockStdin()
    const onOrphaned = vi.fn()
    let ppid = 1234
    const originalPpid = process.ppid
    const origStdin = process.stdin

    Object.defineProperty(process, 'ppid', { get: () => ppid, configurable: true })
    Object.defineProperty(process, 'stdin', { value: stdin, configurable: true })

    try {
      const stop = startParentWatchdog(onOrphaned, 1000)

      // Simulate PID recycling
      ppid = 5678
      vi.advanceTimersByTime(1000)

      expect(onOrphaned).toHaveBeenCalledTimes(1)
      expect(onOrphaned.mock.calls[0][0]).toContain('parent changed (ppid 1234 → 5678)')

      stop()
    } finally {
      Object.defineProperty(process, 'ppid', { get: () => originalPpid, configurable: true })
      Object.defineProperty(process, 'stdin', { value: origStdin, configurable: true })
    }
  })

  it('does not call onOrphaned when ppid stays the same', () => {
    const stdin = mockStdin()
    const onOrphaned = vi.fn()
    const originalPpid = process.ppid
    const fixedPpid = originalPpid > 1 ? originalPpid : 9999
    const origStdin = process.stdin

    Object.defineProperty(process, 'ppid', { get: () => fixedPpid, configurable: true })
    Object.defineProperty(process, 'stdin', { value: stdin, configurable: true })

    try {
      const stop = startParentWatchdog(onOrphaned, 1000)

      // Advance several intervals — parent stays alive
      vi.advanceTimersByTime(5000)

      expect(onOrphaned).not.toHaveBeenCalled()

      stop()
    } finally {
      Object.defineProperty(process, 'ppid', { get: () => originalPpid, configurable: true })
      Object.defineProperty(process, 'stdin', { value: origStdin, configurable: true })
    }
  })

  // ── Lifecycle / edge cases ───────────────────────────────────────

  it('returns a no-op when already orphaned at startup (ppid <= 1)', () => {
    const stdin = mockStdin()
    const onOrphaned = vi.fn()
    const originalPpid = process.ppid
    const origStdin = process.stdin

    Object.defineProperty(process, 'ppid', { get: () => 1, configurable: true })
    Object.defineProperty(process, 'stdin', { value: stdin, configurable: true })

    try {
      const stop = startParentWatchdog(onOrphaned, 1000)

      // Should not fire even after time passes
      vi.advanceTimersByTime(10000)
      stdin.emit('end')

      expect(onOrphaned).not.toHaveBeenCalled()

      // stop() should not throw
      expect(() => stop()).not.toThrow()
    } finally {
      Object.defineProperty(process, 'ppid', { get: () => originalPpid, configurable: true })
      Object.defineProperty(process, 'stdin', { value: origStdin, configurable: true })
    }
  })

  it('stop() prevents further checks and removes stdin listeners', () => {
    const stdin = mockStdin()
    const onOrphaned = vi.fn()
    let ppid = 1234
    const originalPpid = process.ppid
    const origStdin = process.stdin

    Object.defineProperty(process, 'ppid', { get: () => ppid, configurable: true })
    Object.defineProperty(process, 'stdin', { value: stdin, configurable: true })

    try {
      const stop = startParentWatchdog(onOrphaned, 1000)

      stop()

      // Both stdin events and ppid changes should be ignored after stop()
      ppid = 1
      stdin.emit('end')
      vi.advanceTimersByTime(10000)

      expect(onOrphaned).not.toHaveBeenCalled()
    } finally {
      Object.defineProperty(process, 'ppid', { get: () => originalPpid, configurable: true })
      Object.defineProperty(process, 'stdin', { value: origStdin, configurable: true })
    }
  })

  it('stdin detection fires before ppid fallback', () => {
    const stdin = mockStdin()
    const onOrphaned = vi.fn()
    let ppid = 1234
    const originalPpid = process.ppid
    const origStdin = process.stdin

    Object.defineProperty(process, 'ppid', { get: () => ppid, configurable: true })
    Object.defineProperty(process, 'stdin', { value: stdin, configurable: true })

    try {
      const stop = startParentWatchdog(onOrphaned, 1000)

      // stdin end fires immediately
      stdin.emit('end')
      expect(onOrphaned).toHaveBeenCalledTimes(1)
      expect(onOrphaned.mock.calls[0][0]).toContain('stdin')

      // ppid change later should not fire again (already fired)
      ppid = 1
      vi.advanceTimersByTime(1000)
      expect(onOrphaned).toHaveBeenCalledTimes(1)

      stop()
    } finally {
      Object.defineProperty(process, 'ppid', { get: () => originalPpid, configurable: true })
      Object.defineProperty(process, 'stdin', { value: origStdin, configurable: true })
    }
  })
})
