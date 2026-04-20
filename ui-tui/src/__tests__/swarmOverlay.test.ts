import { describe, expect, it } from 'vitest'

import { buildSwarmOverlayModel } from '../components/swarmOverlay.js'

describe('buildSwarmOverlayModel', () => {
  it('summarizes empty swarm state', () => {
    expect(buildSwarmOverlayModel([])).toEqual({
      empty: true,
      headline: 'Swarm · idle',
      summary: 'No active subagents yet',
      total: 0
    })
  })

  it('summarizes active and completed subagents for the dedicated swarm surface', () => {
    expect(
      buildSwarmOverlayModel([
        { id: 'sa:0', index: 0, notes: [], status: 'running', summary: '', thinking: [], tools: [] },
        { id: 'sa:1', index: 1, notes: [], status: 'completed', summary: 'done', thinking: [], tools: [] },
        { id: 'sa:2', index: 2, notes: [], status: 'failed', summary: 'boom', thinking: [], tools: [] }
      ])
    ).toEqual({
      empty: false,
      headline: 'Swarm · 3 subagents',
      summary: '1 running · 1 completed · 1 failed',
      total: 3
    })
  })
})
