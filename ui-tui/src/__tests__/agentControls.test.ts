import { expect, it } from 'vitest'

import { rosterViewport, sendAgentSteer } from '../components/agentControls.js'
import type { GatewayClient } from '../gatewayClient.js'

it('reports queued acceptance rather than claiming delivery and preserves rejected text', async () => {
  const calls: unknown[] = []

  const gw = {
    request: async (...args: unknown[]) => {
      calls.push(args)

      return { status: 'queued' }
    }
  } as unknown as GatewayClient

  expect(await sendAgentSteer(gw, 'owner', 'child', 'check tests')).toEqual({
    accepted: true,
    message: 'Queued for child — applied at the next tool boundary.'
  })
  expect(calls).toEqual([['subagent.steer', { session_id: 'owner', subagent_id: 'child', text: 'check tests' }]])
  const rejected = { request: async () => ({ status: 'rejected' }) } as unknown as GatewayClient
  expect((await sendAgentSteer(rejected, 'owner', 'child', 'check tests')).accepted).toBe(false)
})

it('keeps every roster selection in the visible viewport including short terminals', () => {
  for (const height of [10, 14, 24, 40]) {
    for (const cursor of [0, 9, 19]) {
      const view = rosterViewport(height, 20, cursor)
      expect(view.start).toBeLessThanOrEqual(cursor)
      expect(view.start + view.rows).toBeGreaterThan(cursor)
      expect(view.rows + (view.timelineRows ? view.timelineRows + 4 : 0) + 7).toBeLessThanOrEqual(height)
    }
  }
})
