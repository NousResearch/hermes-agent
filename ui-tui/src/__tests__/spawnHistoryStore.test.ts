import { beforeEach, describe, expect, it } from 'vitest'

import { clearSpawnHistory, getSpawnHistory, pushDiskSnapshot } from '../app/spawnHistoryStore.js'

describe('spawnHistoryStore status normalization', () => {
  beforeEach(() => {
    clearSpawnHistory()
  })

  it('keeps timeout/error statuses from disk snapshots', () => {
    pushDiskSnapshot(
      {
        finished_at: 1_700_000_001,
        label: 'status test',
        session_id: 'sess-1',
        started_at: 1_700_000_000,
        subagents: [
          { goal: 'timeout child', id: 'sa-timeout', index: 0, status: 'timeout' },
          { goal: 'error child', id: 'sa-error', index: 1, status: 'error' }
        ]
      },
      '/tmp/snap-timeout-error.json'
    )

    const statuses = getSpawnHistory()[0]?.subagents.map(s => s.status)

    expect(statuses).toEqual(['timeout', 'error'])
  })

  it('falls back unknown disk statuses to completed', () => {
    pushDiskSnapshot(
      {
        finished_at: 1_700_000_011,
        label: 'unknown status test',
        session_id: 'sess-2',
        started_at: 1_700_000_010,
        subagents: [{ goal: 'mystery child', id: 'sa-unknown', index: 0, status: 'mystery_status' }]
      },
      '/tmp/snap-unknown.json'
    )

    const status = getSpawnHistory()[0]?.subagents[0]?.status

    expect(status).toBe('completed')
  })

  it('pluralizes the fallback label as singular for exactly one subagent', () => {
    pushDiskSnapshot(
      {
        finished_at: 1_700_000_021,
        session_id: 'sess-3',
        started_at: 1_700_000_020,
        subagents: [{ goal: 'lone child', id: 'sa-solo', index: 0, status: 'completed' }]
      },
      '/tmp/snap-one.json'
    )

    expect(getSpawnHistory()[0]?.label).toBe('1 subagent')
  })

  it('pluralizes the fallback label for multiple subagents', () => {
    pushDiskSnapshot(
      {
        finished_at: 1_700_000_031,
        session_id: 'sess-4',
        started_at: 1_700_000_030,
        subagents: [
          { goal: 'first child', id: 'sa-a', index: 0, status: 'completed' },
          { goal: 'second child', id: 'sa-b', index: 1, status: 'completed' }
        ]
      },
      '/tmp/snap-two.json'
    )

    expect(getSpawnHistory()[0]?.label).toBe('2 subagents')
  })
})
