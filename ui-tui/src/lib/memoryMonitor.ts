import { type HeapDumpResult, performHeapDump } from './memory.js'

export type MemoryLevel = 'critical' | 'high' | 'normal'

export interface MemorySnapshot {
  heapUsed: number
  level: MemoryLevel
  rss: number
}

export interface MemoryMonitorOptions {
  criticalBytes?: number
  highBytes?: number
  intervalMs?: number
  onCritical?: (snap: MemorySnapshot, dump: HeapDumpResult | null) => void
  onHigh?: (snap: MemorySnapshot, dump: HeapDumpResult | null) => void
  onSnapshot?: (snap: MemorySnapshot) => void
}

const GB = 1024 ** 3

const DEFAULTS = {
  criticalBytes: 2.5 * GB,
  highBytes: 1.5 * GB,
  intervalMs: 10_000
}

export function startMemoryMonitor({
  criticalBytes = DEFAULTS.criticalBytes,
  highBytes = DEFAULTS.highBytes,
  intervalMs = DEFAULTS.intervalMs,
  onCritical,
  onHigh,
  onSnapshot
}: MemoryMonitorOptions = {}): () => void {
  let dumpedHigh = false
  let dumpedCritical = false

  const tick = async () => {
    const { heapUsed, rss } = process.memoryUsage()
    const level: MemoryLevel = heapUsed >= criticalBytes ? 'critical' : heapUsed >= highBytes ? 'high' : 'normal'
    const snap: MemorySnapshot = { heapUsed, level, rss }

    onSnapshot?.(snap)

    if (level === 'normal') {
      dumpedHigh = false
      dumpedCritical = false

      return
    }

    if (level === 'high' && !dumpedHigh) {
      dumpedHigh = true
      const dump = await performHeapDump('auto-high').catch(() => null)

      onHigh?.(snap, dump)

      return
    }

    if (level === 'critical' && !dumpedCritical) {
      dumpedCritical = true
      const dump = await performHeapDump('auto-critical').catch(() => null)

      onCritical?.(snap, dump)
    }
  }

  const handle = setInterval(() => void tick(), intervalMs)

  handle.unref?.()

  return () => clearInterval(handle)
}
