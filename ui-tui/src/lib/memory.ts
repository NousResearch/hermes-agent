import { createWriteStream } from 'node:fs'
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'
import { pipeline } from 'node:stream/promises'
import { getHeapSnapshot, getHeapSpaceStatistics, getHeapStatistics } from 'node:v8'

export type MemoryTrigger = 'auto-high' | 'auto-critical' | 'manual'

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

const heapDumpRoot = () =>
  process.env.HERMES_HEAPDUMP_DIR?.trim() || join(homedir() || tmpdir(), '.hermes', 'heapdumps')

const processInternals = process as unknown as {
  _getActiveHandles: () => unknown[]
  _getActiveRequests: () => unknown[]
}

export async function captureMemoryDiagnostics(trigger: MemoryTrigger): Promise<MemoryDiagnostics> {
  const usage = process.memoryUsage()
  const heapStats = getHeapStatistics()
  const resourceUsage = process.resourceUsage()
  const uptimeSeconds = process.uptime()

  let heapSpaces: ReturnType<typeof getHeapSpaceStatistics> | undefined

  try {
    heapSpaces = getHeapSpaceStatistics()
  } catch {
    /* Bun / older Node — ignore */
  }

  const activeHandles = processInternals._getActiveHandles().length
  const activeRequests = processInternals._getActiveRequests().length

  let openFileDescriptors: number | undefined

  try {
    openFileDescriptors = (await readdir('/proc/self/fd')).length
  } catch {
    /* non-Linux */
  }

  let smapsRollup: string | undefined

  try {
    smapsRollup = await readFile('/proc/self/smaps_rollup', 'utf8')
  } catch {
    /* non-Linux / no access */
  }

  const nativeMemory = usage.rss - usage.heapUsed
  const bytesPerSecond = uptimeSeconds > 0 ? usage.rss / uptimeSeconds : 0
  const mbPerHour = (bytesPerSecond * 3600) / (1024 * 1024)

  const potentialLeaks: string[] = []

  if (heapStats.number_of_detached_contexts > 0) {
    potentialLeaks.push(
      `${heapStats.number_of_detached_contexts} detached context(s) — possible component/closure leak`
    )
  }

  if (activeHandles > 100) {
    potentialLeaks.push(`${activeHandles} active handles — possible timer/socket leak`)
  }

  if (nativeMemory > usage.heapUsed) {
    potentialLeaks.push('Native memory > heap — leak may be in native addons')
  }

  if (mbPerHour > 100) {
    potentialLeaks.push(`High memory growth rate: ${mbPerHour.toFixed(1)} MB/hour`)
  }

  if (openFileDescriptors && openFileDescriptors > 500) {
    potentialLeaks.push(`${openFileDescriptors} open FDs — possible file/socket leak`)
  }

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
    const diagnostics = await captureMemoryDiagnostics(trigger)
    const dir = heapDumpRoot()

    await mkdir(dir, { recursive: true })

    const stamp = new Date().toISOString().replace(/[:.]/g, '-')
    const base = `hermes-${stamp}-${process.pid}-${trigger}`
    const heapPath = join(dir, `${base}.heapsnapshot`)
    const diagPath = join(dir, `${base}.diagnostics.json`)

    await writeFile(diagPath, JSON.stringify(diagnostics, null, 2), { mode: 0o600 })
    await writeSnapshot(heapPath)

    return { diagPath, heapPath, success: true }
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e), success: false }
  }
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0B'
  }

  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const exp = Math.min(units.length - 1, Math.floor(Math.log10(bytes) / 3))
  const value = bytes / 1024 ** exp

  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)}${units[exp]}`
}

async function writeSnapshot(filepath: string) {
  const stream = createWriteStream(filepath, { mode: 0o600 })

  await pipeline(getHeapSnapshot(), stream)
}
