import { mkdir, readdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { afterEach, describe, expect, it } from 'vitest'

import { pruneAutomaticHeapDumps, shouldCaptureHeapDump } from './memory.js'

const ORIGINAL_ENV = { ...process.env }

afterEach(() => {
  process.env = { ...ORIGINAL_ENV }
})

describe('shouldCaptureHeapDump', () => {
  it('keeps manual heap dumps available', () => {
    process.env.HERMES_AUTO_HEAPDUMP = '0'
    process.env.HERMES_AUTO_HEAPDUMP_HIGH = '0'

    expect(shouldCaptureHeapDump('manual')).toBe(true)
  })

  it('disables automatic high-threshold heap dumps by default', () => {
    delete process.env.HERMES_AUTO_HEAPDUMP
    delete process.env.HERMES_AUTO_HEAPDUMP_HIGH

    expect(shouldCaptureHeapDump('auto-high')).toBe(false)
  })

  it('allows explicit opt-in for high-threshold heap dumps', () => {
    process.env.HERMES_AUTO_HEAPDUMP_HIGH = '1'

    expect(shouldCaptureHeapDump('auto-high')).toBe(true)
  })

  it('keeps critical automatic heap dumps unless globally disabled', () => {
    delete process.env.HERMES_AUTO_HEAPDUMP

    expect(shouldCaptureHeapDump('auto-critical')).toBe(true)

    process.env.HERMES_AUTO_HEAPDUMP = '0'
    expect(shouldCaptureHeapDump('auto-critical')).toBe(false)
  })
})

describe('pruneAutomaticHeapDumps', () => {
  it('prunes old automatic dump files while preserving manual dumps', async () => {
    const dir = join(tmpdir(), `hermes-memory-test-${process.pid}-${Date.now()}`)
    await mkdir(dir, { recursive: true })

    await writeFile(join(dir, 'hermes-old-auto-high.heapsnapshot'), 'old-auto')
    await writeFile(join(dir, 'hermes-old-auto-high.diagnostics.json'), 'old-diag')
    await writeFile(join(dir, 'hermes-new-auto-critical.heapsnapshot'), 'new-auto')
    await writeFile(join(dir, 'hermes-new-auto-critical.diagnostics.json'), 'new-diag')
    await writeFile(join(dir, 'hermes-manual.heapsnapshot'), 'manual')

    const removed = await pruneAutomaticHeapDumps(dir, { maxAutomaticBytes: 100, maxAutomaticFiles: 2 })
    const remaining = await readdir(dir)

    expect(removed.length).toBe(2)
    expect(remaining).toContain('hermes-manual.heapsnapshot')
    expect(remaining.filter(name => name.includes('-auto-')).length).toBe(2)
  })

  it('bounds automatic dump bytes even when file count allows more', async () => {
    const dir = join(tmpdir(), `hermes-memory-byte-test-${process.pid}-${Date.now()}`)
    await mkdir(dir, { recursive: true })

    await writeFile(join(dir, 'hermes-a-auto-high.heapsnapshot'), '12345')
    await writeFile(join(dir, 'hermes-b-auto-high.heapsnapshot'), '12345')
    await writeFile(join(dir, 'hermes-c-auto-high.heapsnapshot'), '12345')

    await pruneAutomaticHeapDumps(dir, { maxAutomaticBytes: 10, maxAutomaticFiles: 10 })
    const remaining = await readdir(dir)

    expect(remaining.filter(name => name.includes('-auto-')).length).toBeLessThanOrEqual(2)
  })
})
