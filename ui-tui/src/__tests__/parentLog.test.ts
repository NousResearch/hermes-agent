import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// parentLog gates itself off under VITEST so unit tests can't pollute a real
// ~/.hermes. To exercise the real persistence path we clear that gate, point
// HERMES_HOME at a temp dir, and re-import the module fresh (path + enabled
// flag are captured at module load).
const loadFresh = async (home: string) => {
  vi.resetModules()
  vi.stubEnv('VITEST', '')
  vi.stubEnv('HERMES_HOME', home)

  return import('../lib/parentLog.js')
}

describe('recordParentLifecycle', () => {
  let home: string

  beforeEach(() => {
    home = mkdtempSync(join(tmpdir(), 'hermes-parentlog-'))
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    rmSync(home, { force: true, recursive: true })
  })

  it('appends a timestamped breadcrumb to logs/tui_gateway_crash.log', async () => {
    const { recordParentLifecycle } = await loadFresh(home)

    recordParentLifecycle('graceful-exit received signal=SIGHUP → killing gateway')

    const contents = readFileSync(join(home, 'logs', 'tui_gateway_crash.log'), 'utf8')

    expect(contents).toContain('[tui-parent]')
    expect(contents).toContain('graceful-exit received signal=SIGHUP → killing gateway')
    expect(contents).toMatch(/\d{4}-\d{2}-\d{2}T/)
  })

  it('collapses embedded newlines so a value stays one breadcrumb', async () => {
    const { recordParentLifecycle } = await loadFresh(home)

    recordParentLifecycle('uncaughtException: boom\n  at foo()\r\n  at bar()')

    const lines = readFileSync(join(home, 'logs', 'tui_gateway_crash.log'), 'utf8').trimEnd().split('\n')

    expect(lines).toHaveLength(1)
    expect(lines[0]).toContain('boom ↵   at foo() ↵   at bar()')
  })

  it('caps an oversized breadcrumb so it cannot bloat the shared crash log', async () => {
    const { recordParentLifecycle } = await loadFresh(home)

    recordParentLifecycle('x'.repeat(10_000))

    const line = readFileSync(join(home, 'logs', 'tui_gateway_crash.log'), 'utf8')

    expect(line).toContain('[truncated 10000 chars]')
    expect(line.length).toBeLessThan(4_500)
  })

  it('is a no-op under VITEST so tests stay hermetic', async () => {
    vi.resetModules()
    vi.stubEnv('VITEST', 'true')
    vi.stubEnv('HERMES_HOME', home)

    const { recordParentLifecycle } = await import('../lib/parentLog.js')

    expect(() => recordParentLifecycle('should not be written')).not.toThrow()
    expect(() => readFileSync(join(home, 'logs', 'tui_gateway_crash.log'), 'utf8')).toThrow()
  })
})
