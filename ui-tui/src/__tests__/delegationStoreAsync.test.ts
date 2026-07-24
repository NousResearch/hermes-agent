import { afterEach, describe, expect, it } from 'vitest'

import { $asyncDelegations, applyAsyncList, resetAsyncDelegations } from '../app/delegationStore.js'
import { buildAgentRows } from '../lib/agentRows.js'

afterEach(() => resetAsyncDelegations())

describe('applyAsyncList', () => {
  it('populates $asyncDelegations from an RPC response', () => {
    applyAsyncList({ delegations: [{ delegation_id: 'd1', goal: 'g', role: 'fixer', status: 'running' }], running: 1 })

    expect($asyncDelegations.get()).toHaveLength(1)
    expect($asyncDelegations.get()[0]!.delegation_id).toBe('d1')
  })

  it('tolerates null / undefined / missing-array payloads by clearing', () => {
    applyAsyncList({ delegations: [{ delegation_id: 'd1' }] })
    applyAsyncList(null)
    expect($asyncDelegations.get()).toEqual([])

    applyAsyncList({ delegations: [{ delegation_id: 'd1' }] })
    applyAsyncList(undefined)
    expect($asyncDelegations.get()).toEqual([])

    applyAsyncList({ delegations: [{ delegation_id: 'd1' }] })
    applyAsyncList({})
    expect($asyncDelegations.get()).toEqual([])
  })

  it('replaces (does not append) on each successive snapshot', () => {
    applyAsyncList({ delegations: [{ delegation_id: 'a' }, { delegation_id: 'b' }] })
    applyAsyncList({ delegations: [{ delegation_id: 'c' }] })

    const ids = $asyncDelegations.get().map(d => d.delegation_id)
    expect(ids).toEqual(['c'])
  })

  it('feeds buildAgentRows end-to-end from the store', () => {
    applyAsyncList({
      delegations: [
        { delegation_id: 'd1', dispatched_at: 1000, goal: 'sweep', role: 'tests', status: 'completed', completed_at: 1044 }
      ],
      running: 0
    })

    const { done, rows } = buildAgentRows([], $asyncDelegations.get(), 9_999_999)
    expect(done).toBe(1)
    expect(rows[0]!.resultReady).toBe(true)
    expect(rows[0]!.elapsedSeconds).toBeCloseTo(44, 0)
  })
})
