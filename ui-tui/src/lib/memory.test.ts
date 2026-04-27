import { mkdtempSync, readdirSync, utimesSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { pruneHeapDumpArtifacts } from './memory.js'

const makeFile = (dir: string, name: string, ageDays = 0) => {
  const path = join(dir, name)
  writeFileSync(path, name)

  if (ageDays > 0) {
    const ts = new Date(Date.now() - ageDays * 86400 * 1000)
    utimesSync(path, ts, ts)
  }

  return path
}

describe('pruneHeapDumpArtifacts', () => {
  it('keeps only the newest three heap snapshots', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'heap-prune-'))

    for (let i = 1; i <= 5; i += 1) {
      makeFile(dir, `test_${i}.heapsnapshot`, 6 - i)
    }

    const result = await pruneHeapDumpArtifacts(dir)
    const snapshots = readdirSync(dir)
      .filter(name => name.endsWith('.heapsnapshot'))
      .sort()

    expect(result.removedSnapshots).toBe(2)
    expect(snapshots).toEqual([
      'test_3.heapsnapshot',
      'test_4.heapsnapshot',
      'test_5.heapsnapshot'
    ])
  })

  it('removes diagnostics older than 60 days and keeps recent ones', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'heap-diag-prune-'))

    makeFile(dir, 'old.diagnostics.json', 61)
    makeFile(dir, 'recent.diagnostics.json', 5)
    makeFile(dir, 'keep.heapsnapshot', 1)

    const result = await pruneHeapDumpArtifacts(dir)
    const files = readdirSync(dir).sort()

    expect(result.removedDiagnostics).toBe(1)
    expect(files).toEqual([
      'keep.heapsnapshot',
      'recent.diagnostics.json'
    ])
  })

  it('tolerates a missing directory', async () => {
    await expect(pruneHeapDumpArtifacts(join(tmpdir(), 'does-not-exist-prune'))).resolves.toEqual({
      removedDiagnostics: 0,
      removedSnapshots: 0
    })
  })
})
