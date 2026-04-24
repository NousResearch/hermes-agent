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
}

const GB = 1024 ** 3

export function startMemoryMonitor({
  criticalBytes = 2.5 * GB,
  highBytes = 1.5 * GB,
  intervalMs = 10_000,
  onCritical,
  onHigh
}: MemoryMonitorOptions = {}): () => void {
  const dumped = new Set<Exclude<MemoryLevel, 'normal'>>()

  const tick = async () => {
    const { heapUsed, rss } = process.memoryUsage()
    const level: MemoryLevel = heapUsed >= criticalBytes ? 'critical' : heapUsed >= highBytes ? 'high' : 'normal'

    if (level === 'normal') {
      return void dumped.clear()
    }

    if (dumped.has(level)) {
      return
    }

    dumped.add(level)
    const dump = await performHeapDump(level === 'critical' ? 'auto-critical' : 'auto-high').catch(() => null)

    const snap: MemorySnapshot = { heapUsed, level, rss }

    ;(level === 'critical' ? onCritical : onHigh)?.(snap, dump)
  }

  const handle = setInterval(() => void tick(), intervalMs)

  handle.unref?.()

  return () => clearInterval(handle)
}
