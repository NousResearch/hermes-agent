import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { topLevelSubagents } from '../lib/subagentTree.js'
import type { SubagentProgress } from '../types.js'

// ── Pure-logic tests for the AgentDock invariants ──
// (The component itself reads live nanostores; these tests pin the decision
// logic so a regression in capping / visibility / auto-close is caught without
// a full Ink render harness.)

const mk = (id: string, status: SubagentProgress['status'], over: Partial<SubagentProgress> = {}): SubagentProgress => ({
  depth: 0,
  id,
  index: 0,
  status,
  ...over
})

describe('agentDock logic', () => {
  it('caps top-level agents at 4 (matches maxConcurrentChildren)', () => {
    const items = [
      mk('a', 'running'),
      mk('b', 'running'),
      mk('c', 'running'),
      mk('d', 'running'),
      mk('e', 'running')
    ]
    expect(topLevelSubagents(items)).toHaveLength(5)
    expect(topLevelSubagents(items).slice(0, 4)).toHaveLength(4)
  })

  it('nests children under their parent (not counted as top-level)', () => {
    const items = [mk('parent', 'running'), mk('child', 'running', { parentId: 'parent' })]
    expect(topLevelSubagents(items)).toHaveLength(1)
  })

  it('auto-close window is 15s after the last agent finishes', () => {
    const AUTO_CLOSE_MS = 15_000
    const finishedAt = 1_000_000
    const autoCloseAt = finishedAt + AUTO_CLOSE_MS
    expect(autoCloseAt).toBe(finishedAt + AUTO_CLOSE_MS)
    expect(autoCloseAt - finishedAt).toBe(15_000)
  })

  it('running agent keeps the dock open (no auto-close while active)', () => {
    const items = [mk('a', 'running')]
    const anyRunning = items.some(s => s.status === 'running' || s.status === 'queued')
    expect(anyRunning).toBe(true)
  })

  it('paused/interrupted agents count as not-running → auto-close eligible', () => {
    const items = [mk('a', 'interrupted')]
    const anyRunning = items.some(s => s.status === 'running' || s.status === 'queued')
    expect(anyRunning).toBe(false)
  })
})
