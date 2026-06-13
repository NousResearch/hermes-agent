import { createWriteStream } from 'node:fs'
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'
import { pipeline } from 'node:stream/promises'
import { getHeapSnapshot, getHeapSpaceStatistics, getHeapStatistics } from 'node:v8'

export type MemoryTrigger = 'auto-critical' | 'auto-high' | 'manual'

export interface MemoryDiagnostics {
  activeHandles: number
  activeRequests: number
  analysis: {
    potentialLeaks: string[]
    recommendation: string
  }
  memoryGrowthRate: {
    bytesPerSecond: number
    mbPerHour: number
  }
  memoryUsage: {
    arrayBuffers: number
    external: number
    heapTotal: number
    heapUsed: number
    rss: number
  }
  nodeVersion: string
  openFileDescriptors?: number
  platform: string
  resourceUsage: {
    maxRSS: number
    systemCPUTime: number
    userCPUTime: number
  }
  smapsRollup?: string
  timestamp: string
  trigger: MemoryTrigger
  uptimeSeconds: number
  v8HeapSpaces?: { available: number; name: string; size: number; used: number }[]
  v8HeapStats: {
    detachedContexts: number
    heapSizeLimit: number
    mallocedMemory: number
    nativeContexts: number
    peakMallocedMemory: number
  }
}

export interface HeapDumpResult {
  diagPath?: string
  error?: string
  heapPath?: string
  success: boolean
}

export async function captureMemoryDiagnostics(trigger: MemoryTrigger): Promise<MemoryDiagnostics> {
  const usage = process.memoryUsage()
  const heapStats = getHeapStatistics()
  const resourceUsage = process.resourceUsage()
  const uptimeSeconds = process.uptime()

  // Not available on Bun / older Node.
  let heapSpaces: ReturnType<typeof getHeapSpaceStatistics> | undefined

  try {
    heapSpaces = getHeapSpaceStatistics()
  } catch {
    /* noop */
  }

  const internals = process as unknown as {
    _getActiveHandles: () => unknown[]
    _getActiveRequests: () => unknown[]
  }

  const activeHandles = internals._getActiveHandles().length
  const activeRequests = internals._getActiveRequests().length
  const openFileDescriptors = await swallow(async () => (await readdir('/proc/self/fd')).length)
  const smapsRollup = await swallow(() => readFile('/proc/self/smaps_rollup', 'utf8'))

  const nativeMemory = usage.rss - usage.heapUsed
  // Real growth rate since STARTED_AT (captured at module load) — NOT a lifetime
  // average of rss/uptime, which would report phantom "growth" for a stable process.
  const elapsed = Math.max(0, uptimeSeconds - STARTED_AT.uptime)
  const bytesPerSecond = elapsed > 0 ? (usage.rss - STARTED_AT.rss) / elapsed : 0
  const mbPerHour = (bytesPerSecond * 3600) / (1024 * 1024)

  const potentialLeaks = [
    heapStats.number_of_detached_contexts > 0 &&
      `${heapStats.number_of_detached_contexts} detached context(s) — possible component/closure leak`,
    activeHandles > 100 && `${activeHandles} active handles — possible timer/socket leak`,
    nativeMemory > usage.heapUsed && 'Native memory > heap — leak may be in native addons',
    mbPerHour > 100 && `High memory growth rate: ${mbPerHour.toFixed(1)} MB/hour`,
    openFileDescriptors && openFileDescriptors > 500 && `${openFileDescriptors} open FDs — possible file/socket leak`
  ].filter((s): s is string => typeof s === 'string')

  return {
    activeHandles,
    activeRequests,
    analysis: {
      potentialLeaks,
      recommendation: potentialLeaks.length
        ? `WARNING: ${potentialLeaks.length} potential leak indicator(s). See potentialLeaks.`
        : 'No obvious leak indicators. Inspect heap snapshot for retained objects.'
    },
    memoryGrowthRate: { bytesPerSecond, mbPerHour },
    memoryUsage: {
      arrayBuffers: usage.arrayBuffers,
      external: usage.external,
      heapTotal: usage.heapTotal,
      heapUsed: usage.heapUsed,
      rss: usage.rss
    },
    nodeVersion: process.version,
    openFileDescriptors,
    platform: process.platform,
    resourceUsage: {
      maxRSS: resourceUsage.maxRSS * 1024,
      systemCPUTime: resourceUsage.systemCPUTime,
      userCPUTime: resourceUsage.userCPUTime
    },
    smapsRollup,
    timestamp: new Date().toISOString(),
    trigger,
    uptimeSeconds,
    v8HeapSpaces: heapSpaces?.map(s => ({
      available: s.space_available_size,
      name: s.space_name,
      size: s.space_size,
      used: s.space_used_size
    })),
    v8HeapStats: {
      detachedContexts: heapStats.number_of_detached_contexts,
      heapSizeLimit: heapStats.heap_size_limit,
      mallocedMemory: heapStats.malloced_memory,
      nativeContexts: heapStats.number_of_native_contexts,
      peakMallocedMemory: heapStats.peak_malloced_memory
    }
  }
}

export async function performHeapDump(trigger: MemoryTrigger = 'manual'): Promise<HeapDumpResult> {
  try {
    // Diagnostics first — heap-snapshot serialization can crash on very large
    // heaps, and the JSON sidecar is the most actionable artifact if so.
    const diagnostics = await captureMemoryDiagnostics(trigger)
    const dir = process.env.HERMES_HEAPDUMP_DIR?.trim() || join(homedir() || tmpdir(), '.hermes', 'heapdumps')

    await mkdir(dir, { recursive: true })

    const base = `hermes-${new Date().toISOString().replace(/[:.]/g, '-')}-${process.pid}-${trigger}`
    const heapPath = join(dir, `${base}.heapsnapshot`)
    const diagPath = join(dir, `${base}.diagnostics.json`)

    await writeFile(diagPath, JSON.stringify(diagnostics, null, 2), { mode: 0o600 })
    await pipeline(getHeapSnapshot(), createWriteStream(heapPath, { mode: 0o600 }))

    return { diagPath, heapPath, success: true }
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e), success: false }
  }
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0B'
  }

  const exp = Math.min(UNITS.length - 1, Math.floor(Math.log10(bytes) / 3))
  const value = bytes / 1024 ** exp

  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)}${UNITS[exp]}`
}

const UNITS = ['B', 'KB', 'MB', 'GB', 'TB']

const STARTED_AT = { rss: process.memoryUsage().rss, uptime: process.uptime() }

// Returns undefined when the probe isn't available (non-Linux paths, sandboxed FS).
const swallow = async <T>(fn: () => Promise<T>): Promise<T | undefined> => {
  try {
    return await fn()
  } catch {
    return undefined
  }
}
