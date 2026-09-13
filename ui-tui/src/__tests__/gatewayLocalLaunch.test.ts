import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

vi.mock('node:child_process', () => ({ execFile: vi.fn(), spawn: vi.fn(() => { throw new Error('independent owner forbidden') }) }))
import { GatewayClient } from '../gatewayClient.js'

const home = mkdtempSync(join(tmpdir(), 'ink-bootstrap-test-'))
afterEach(() => { vi.unstubAllEnvs(); rmSync(home, { recursive: true, force: true }) })

it('reports failed canonical bootstrap without creating an independent owner or retrying ensure', async () => {
  vi.stubEnv('HERMES_HOME', home)
  vi.stubEnv('HERMES_TUI_GATEWAY_URL', '')
  const bootstrap = vi.fn().mockRejectedValue(new Error('gateway draining: update_paused'))
  const gw = new GatewayClient(bootstrap)
  const events: any[] = []
  gw.on('event', e => events.push(e))
  gw.start()
  gw.drain()
  await vi.waitFor(() => expect(events.some(e => e.type === 'gateway.start_timeout' && e.payload.stderr_tail.includes('update_paused'))).toBe(true))
  await expect(gw.request('session.create')).rejects.toThrow('update_paused')
  expect(bootstrap).toHaveBeenCalledTimes(1)
  gw.kill()
})
