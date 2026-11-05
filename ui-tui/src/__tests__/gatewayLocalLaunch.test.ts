import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

vi.mock('node:child_process', () => ({
  execFile: vi.fn(),
  spawn: vi.fn(() => {
    throw new Error('independent owner forbidden')
  })
}))
import { GatewayClient } from '../gatewayClient.js'

const home = mkdtempSync(join(tmpdir(), 'ink-bootstrap-test-'))
afterEach(() => {
  vi.unstubAllEnvs()
  rmSync(home, { recursive: true, force: true })
})

it('reports failed canonical bootstrap without creating an independent owner or retrying ensure', async () => {
  vi.stubEnv('HERMES_HOME', home)
  vi.stubEnv('HERMES_TUI_GATEWAY_URL', '')
  const bootstrap = vi.fn().mockRejectedValue(new Error('gateway draining: update_paused'))
  const gw = new GatewayClient(bootstrap)
  const events: any[] = []
  gw.on('event', e => events.push(e))
  gw.start()
  gw.drain()
  await vi.waitFor(() =>
    expect(
      events.some(e => e.type === 'gateway.start_timeout' && e.payload.stderr_tail.includes('update_paused'))
    ).toBe(true)
  )
  await expect(gw.request('session.create')).rejects.toThrow('update_paused')
  expect(bootstrap).toHaveBeenCalledTimes(1)
  gw.kill()
})

it('reports a bootstrap that cannot load Hermes once and stops retrying (#138878)', async () => {
  vi.useFakeTimers()
  vi.stubEnv('HERMES_HOME', home)
  vi.stubEnv('HERMES_TUI_GATEWAY_URL', '')

  const bootstrap = vi
    .fn()
    .mockRejectedValue(
      new Error(
        "Command failed: python3 gateway_bootstrap.py --start\ngateway unavailable: bootstrap_dependencies\nNo module named 'ruamel'"
      )
    )

  const gw = new GatewayClient(bootstrap)
  const events: any[] = []
  gw.on('event', e => events.push(e))

  try {
    gw.start()
    gw.drain()
    await vi.advanceTimersByTimeAsync(5 * 60_000)
    expect(bootstrap).toHaveBeenCalledTimes(1)
    expect(events.filter(e => e.type === 'gateway.start_timeout')).toHaveLength(1)
    expect(events.some(e => e.type === 'gateway.reconnecting')).toBe(false)
  } finally {
    gw.kill()
    vi.useRealTimers()
  }
})
