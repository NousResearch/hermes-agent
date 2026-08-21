import { describe, expect, it } from 'vitest'

import type { SessionActiveItem } from '../gatewayTypes.js'

import { shouldRecoverStaleBusy } from './staleBusyRecovery.js'

const session = (status: SessionActiveItem['status'], id = 'sid-1'): SessionActiveItem => ({
  id,
  status
})

const completion = (awaiting: boolean, epoch = 1) => ({
  awaiting,
  epoch
})

const shouldRecover = ({
  busy = true,
  currentCompletion = completion(true),
  currentSessionId = 'sid-1',
  requestedCompletion = completion(true),
  requestedSessionId = 'sid-1',
  sessions = [session('idle')]
}: Partial<Parameters<typeof shouldRecoverStaleBusy>[0]> = {}) =>
  shouldRecoverStaleBusy({
    busy,
    currentCompletion,
    currentSessionId,
    requestedCompletion,
    requestedSessionId,
    sessions
  })

describe('shouldRecoverStaleBusy', () => {
  it('recovers when the same in-flight turn is busy locally but idle on the backend', () => {
    expect(shouldRecover()).toBe(true)
  })

  it.each(['working', 'waiting', 'starting'] as const)('does not recover while the backend reports %s', status => {
    expect(shouldRecover({ sessions: [session(status)] })).toBe(false)
  })

  it('does not recover when the frontend is already idle', () => {
    expect(shouldRecover({ busy: false })).toBe(false)
  })

  it('does not recover when the current session is missing', () => {
    expect(shouldRecover({ sessions: [session('idle', 'sid-2')] })).toBe(false)
  })

  it('does not recover without a current session id', () => {
    expect(shouldRecover({ currentSessionId: null })).toBe(false)
  })

  it('ignores a stale current flag from an older active-list response', () => {
    expect(
      shouldRecover({
        sessions: [
          { current: true, id: 'sid-old', status: 'idle' },
          { id: 'sid-1', status: 'working' }
        ]
      })
    ).toBe(false)
  })

  it('does not recover busy states that were not waiting for message.complete', () => {
    expect(
      shouldRecover({
        requestedCompletion: completion(false)
      })
    ).toBe(false)
  })

  it('does not recover after the turn has already stopped awaiting completion', () => {
    expect(
      shouldRecover({
        currentCompletion: completion(false)
      })
    ).toBe(false)
  })

  it('ignores an active-list response requested for a different session', () => {
    expect(
      shouldRecover({
        requestedSessionId: 'sid-old'
      })
    ).toBe(false)
  })

  it('ignores a response from an older turn in the same session', () => {
    expect(
      shouldRecover({
        currentCompletion: completion(true, 2),
        requestedCompletion: completion(true, 1)
      })
    ).toBe(false)
  })
})
