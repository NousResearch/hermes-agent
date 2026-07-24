import { describe, expect, it } from 'vitest'

import type { AsyncDelegationRecord } from '../gatewayTypes.js'
import { buildAgentRows } from '../lib/agentRows.js'
import type { SubagentProgress } from '../types.js'

const liveSub = (over: Partial<SubagentProgress> = {}): SubagentProgress => ({
  depth: 1,
  goal: 'map auth handshake edge cases',
  id: 's1',
  index: 0,
  notes: [],
  status: 'running',
  taskCount: 0,
  thinking: [],
  toolCount: 3,
  tools: ['read_file'],
  ...over
})

const asyncRec = (over: Partial<AsyncDelegationRecord> = {}): AsyncDelegationRecord => ({
  delegation_id: 'd1',
  dispatched_at: 1000,
  goal: 'patch token-bucket refill race',
  role: 'fixer',
  status: 'running',
  ...over
})

describe('buildAgentRows', () => {
  it('counts running from both live subagents and async delegations', () => {
    const { done, rows, running } = buildAgentRows(
      [liveSub({ id: 'a', status: 'running' }), liveSub({ id: 'b', status: 'completed' })],
      [asyncRec({ delegation_id: 'd1', status: 'running' }), asyncRec({ delegation_id: 'd2', status: 'completed' })],
      2_000_000
    )

    expect(running).toBe(2) // one live + one async
    expect(done).toBe(2) // one live completed + one async completed
    expect(rows).toHaveLength(4)
  })

  it('orders live rows before async rows', () => {
    const { rows } = buildAgentRows([liveSub({ id: 'a' })], [asyncRec()], 2_000_000)

    expect(rows[0].key).toBe('live:a')
    expect(rows[1].key).toBe('async:d1')
  })

  it('marks a completed async delegation as result-ready with a "result ready" detail', () => {
    const { rows } = buildAgentRows([], [asyncRec({ status: 'completed' })], 2_000_000)

    expect(rows[0].resultReady).toBe(true)
    expect(rows[0].detail).toBe('result ready')
  })

  it('surfaces the live subagent last tool as the row detail', () => {
    const { rows } = buildAgentRows([liveSub({ tools: ['read_file', 'bash'] })], [], 2_000_000)

    expect(rows[0].detail).toBe('bash')
    expect(rows[0].resultReady).toBe(false)
  })

  it('clocks a running live subagent from startedAt', () => {
    const now = 100_000
    const { rows } = buildAgentRows([liveSub({ startedAt: now - 12_000, durationSeconds: undefined })], [], now)

    expect(rows[0].elapsedSeconds).toBeCloseTo(12, 0)
  })

  it('freezes async elapsed at completed_at once done', () => {
    const { rows } = buildAgentRows(
      [],
      [asyncRec({ status: 'completed', dispatched_at: 1000, completed_at: 1044 })],
      9_999_999_999
    )

    // 1044 - 1000 = 44s, independent of `now`.
    expect(rows[0].elapsedSeconds).toBeCloseTo(44, 0)
  })

  it('returns an empty result for no agents', () => {
    const { done, rows, running } = buildAgentRows([], [], 1)

    expect(rows).toEqual([])
    expect(running).toBe(0)
    expect(done).toBe(0)
  })

  it('counts a queued live subagent as running', () => {
    const { running } = buildAgentRows([liveSub({ status: 'queued' })], [], 1)
    expect(running).toBe(1)
  })

  it('treats the "done" async status as result-ready (alias of completed)', () => {
    const { done, rows } = buildAgentRows([], [asyncRec({ status: 'done' })], 2_000_000)
    expect(rows[0].resultReady).toBe(true)
    expect(done).toBe(1)
  })

  it('does not count a failed/interrupted agent as running or done', () => {
    const { done, running } = buildAgentRows(
      [liveSub({ status: 'failed' })],
      [asyncRec({ status: 'error' })],
      2_000_000
    )

    expect(running).toBe(0)
    expect(done).toBe(0)
  })

  it('falls back to "agent" when a live subagent has no goal', () => {
    const { rows } = buildAgentRows([liveSub({ goal: '' })], [], 1)
    expect(rows[0].goal).toBe('agent')
  })

  it('uses the async role as the row name, defaulting to "agent"', () => {
    const { rows } = buildAgentRows(
      [],
      [asyncRec({ delegation_id: 'x', role: undefined }), asyncRec({ delegation_id: 'y', role: 'fixer' })],
      1
    )

    expect(rows[0].name).toBe('agent')
    expect(rows[1].name).toBe('fixer')
  })

  it('yields null elapsed for an async record with no dispatched_at', () => {
    const { rows } = buildAgentRows([], [asyncRec({ dispatched_at: undefined })], 1)
    expect(rows[0].elapsedSeconds).toBeNull()
  })

  it('shows the raw status as detail for a still-running async row', () => {
    const { rows } = buildAgentRows([], [asyncRec({ status: 'running' })], 1)
    expect(rows[0].detail).toBe('running')
    expect(rows[0].resultReady).toBe(false)
  })

  it('leaves live elapsed null when neither duration nor startedAt is known', () => {
    const { rows } = buildAgentRows([liveSub({ durationSeconds: undefined, startedAt: undefined })], [], 1)
    expect(rows[0].elapsedSeconds).toBeNull()
  })
})
