import { DRAFT_CONTROL_PREFIX, type DraftAttachRequest, type DraftResult, type DraftState } from '@hermes/shared'
import { expect, it } from 'vitest'

it('exports the reserved control envelope and interoperable identities', () => {
  const state: DraftState = { type: 'draft.state', available: true, identity: {
    pty_instance: 'pty', connection_generation: 1, session_id: 'runtime', draft_id: 'draft'
  } }

  const request: DraftAttachRequest = {
    type: 'draft.attach', request_id: 'request', expected: state.identity, path: '/staged/report.txt'
  }

  const result: DraftResult = {
    type: 'draft.result', request_id: request.request_id, identity: request.expected, status: 'attached'
  }

  expect(DRAFT_CONTROL_PREFIX).toBe('\u0000hermes-draft-v1:')
  expect(JSON.parse((DRAFT_CONTROL_PREFIX + JSON.stringify(request)).slice(DRAFT_CONTROL_PREFIX.length)).expected)
    .toEqual(result.identity)
})
