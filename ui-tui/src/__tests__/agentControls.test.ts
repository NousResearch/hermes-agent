import { expect, it } from 'vitest'

import { rosterViewport, sendAgentSteer } from '../components/agentControls.js'
import { filterLabel, sortLabel } from '../components/agentsOverlay.js'
import { processSummary } from '../components/agentsPanel.js'
import type { GatewayClient } from '../gatewayClient.js'
import { applyLocale, messages, resetLocale } from '../i18n/runtime.js'

it('reports queued acceptance rather than claiming delivery and preserves rejected text', async () => {
  const calls: unknown[] = []

  const gw = {
    request: async (...args: unknown[]) => {
      calls.push(args)

      return { status: 'queued' }
    }
  } as unknown as GatewayClient

  const queued = await sendAgentSteer(gw, 'owner', 'child', 'check tests')
  expect(queued.accepted).toBe(true)
  expect(queued.message).toBe(messages().hubs.agentControls.queued)
  expect(calls).toEqual([['subagent.steer', { session_id: 'owner', subagent_id: 'child', text: 'check tests' }]])
  const rejected = { request: async () => ({ status: 'rejected' }) } as unknown as GatewayClient
  const declined = await sendAgentSteer(rejected, 'owner', 'child', 'check tests')
  expect(declined.accepted).toBe(false)
  expect(declined.message).toBe(messages().hubs.agentControls.notQueued)
})

it('resolves hub labels against the active language at call time, not import time', () => {
  try {
    expect(sortLabel('tools-desc')).toBe('busiest')
    expect(processSummary({ hidden: 0, rows: [], running: 2, total: 3 })).toBe('2 running · 1 done')

    applyLocale('xx', {
      lang: 'xx',
      surface: 'tui',
      messages: {
        'hubs.agents.sort.toolsDesc': 'Z',
        'hubs.agents.filter.leaf': 'L',
        'hubs.agentsPanel.running': '{0} R',
        'hubs.agentsPanel.done': '{0} D'
      }
    })

    expect(sortLabel('tools-desc')).toBe('Z')
    expect(filterLabel('leaf')).toBe('L')
    expect(sortLabel('status')).toBe('status')
    expect(processSummary({ hidden: 0, rows: [], running: 2, total: 3 })).toBe('2 R · 1 D')
  } finally {
    resetLocale()
  }
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
