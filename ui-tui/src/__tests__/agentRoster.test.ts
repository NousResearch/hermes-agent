import { expect, it } from 'vitest'

import { $agentSnapshot, applyAgentSnapshot, mergeAgentRoster } from '../app/agentRoster.js'
import { shouldPassThroughToGlobalHandler } from '../components/textInput.js'
import type { SubagentProgress } from '../types.js'

it('never rolls event progress back when an older live snapshot arrives', () => {
  const event: SubagentProgress = {
    id: 'child',
    goal: 'inspect',
    depth: 0,
    index: 0,
    parentId: null,
    notes: [],
    thinking: [],
    tools: ['read_file'],
    taskCount: 1,
    status: 'running',
    toolCount: 5
  }

  const stale = {
    subagents: [{ subagent_id: event.id, status: 'queued', tool_count: 1 }],
    delegations: []
  }

  for (const status of ['running', 'completed', 'error', 'interrupted'] as const) {
    const latest = { ...event, status }
    const [row] = mergeAgentRoster([latest], stale)
    expect(row?.status).toBe(latest.status)
    expect(row?.toolCount).toBe(latest.toolCount)
  }
})

it('lets the agents shortcut leave the composer without stealing redo', () => {
  const key = { ctrl: true, shift: false, meta: false } as Parameters<typeof shouldPassThroughToGlobalHandler>[1]
  expect(shouldPassThroughToGlobalHandler('t', key)).toBe(true)
  expect(shouldPassThroughToGlobalHandler('y', key)).toBe(false)
})

it('merges one row per actual child and does not notify unchanged snapshots', () => {
  const data = {
    subagents: [{ subagent_id: 'child', delegation_id: 'batch', goal: 'inspect', started_at: 1 }],
    delegations: [{ delegation_id: 'batch-1', status: 'running', subagent_ids: [] }]
  }

  applyAgentSnapshot('session', data)
  const previous = $agentSnapshot.get()
  applyAgentSnapshot('session', structuredClone(data))
  expect($agentSnapshot.get()).toBe(previous)
  expect(mergeAgentRoster([], data).map(s => s.id)).toEqual(['child'])
  applyAgentSnapshot('next')
  expect($agentSnapshot.get().data.subagents).toEqual([])
})
