import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import {
  _setV8LoaderForTesting,
  captureMemoryDiagnostics,
  getV8HeapStatistics,
  performHeapDump,
} from '../memory.js'
import { startMemoryMonitor } from '../memoryMonitor.js'

const originalHeapdumpDir = process.env.HERMES_HEAPDUMP_DIR

afterEach(() => {
  _setV8LoaderForTesting(null)
  if (originalHeapdumpDir === undefined) {
    delete process.env.HERMES_HEAPDUMP_DIR
  } else {
    process.env.HERMES_HEAPDUMP_DIR = originalHeapdumpDir
  }
})

describe('memory diagnostics when node:v8 is absent or incomplete', () => {
  it('returns null heap stats and does not throw in captureMemoryDiagnostics when node:v8 is missing', async () => {
    _setV8LoaderForTesting(() => null)
    expect(getV8HeapStatistics()).toBeNull()
    const diag = await captureMemoryDiagnostics('manual')
    expect(diag.v8HeapStats.heapSizeLimit).toBe(0)
    expect(diag.v8HeapSpaces).toBeUndefined()
    expect(typeof diag.memoryUsage.rss).toBe('number')
  })

  it('degrades gracefully when node:v8 omits getHeapStatistics or throws', async () => {
    _setV8LoaderForTesting(() => ({}))
    expect(getV8HeapStatistics()).toBeNull()
    const emptyDiag = await captureMemoryDiagnostics('manual')
    expect(emptyDiag.v8HeapStats.heapSizeLimit).toBe(0)

    _setV8LoaderForTesting(() => {
      throw new Error('node:v8 unavailable')
    })
    expect(getV8HeapStatistics()).toBeNull()
    const errDiag = await captureMemoryDiagnostics('manual')
    expect(errDiag.v8HeapStats.heapSizeLimit).toBe(0)
  })

  it('suppresses heap snapshot cleanly in performHeapDump when getHeapSnapshot is unavailable', async () => {
    _setV8LoaderForTesting(() => null)
    const dir = mkdtempSync(join(tmpdir(), 'hermes-mem-test-'))
    try {
      process.env.HERMES_HEAPDUMP_DIR = dir
      const res = await performHeapDump('manual')
      expect(res.success).toBe(true)
      expect(res.suppressed).toBe(true)
      expect(res.diagPath).toBeTruthy()
      expect(res.heapPath).toBeUndefined()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('turns startMemoryMonitor into a safe no-op when node:v8 heap stats are unavailable', () => {
    _setV8LoaderForTesting(() => null)
    const stop = startMemoryMonitor()
    expect(typeof stop).toBe('function')
    expect(() => stop()).not.toThrow()
  })
})
