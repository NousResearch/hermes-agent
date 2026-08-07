import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getTurnState, patchTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import {
  captureSessionResponseTurnFence,
  hydrateLiveSessionInflight,
  liveSessionInflightMessages,
  reconcileSessionResponseTurn,
  scheduleResumeScrollToBottom,
  signalFreshSessionBoundary,
  writeActiveSessionFile
} from '../app/useSessionLifecycle.js'

describe('fresh session boundary', () => {
  it('signals only when a live session is replaced by a different session', () => {
    const onFreshSessionStarted = vi.fn()

    expect(signalFreshSessionBoundary('old-session', 'new-session', onFreshSessionStarted)).toBe(true)
    expect(signalFreshSessionBoundary(null, 'first-session', onFreshSessionStarted)).toBe(false)
    expect(signalFreshSessionBoundary('same-session', 'same-session', onFreshSessionStarted)).toBe(false)
    expect(signalFreshSessionBoundary('old-session', null, onFreshSessionStarted)).toBe(false)
    expect(signalFreshSessionBoundary('old-session', 'new-session')).toBe(false)
    expect(onFreshSessionStarted).toHaveBeenCalledOnce()
    expect(onFreshSessionStarted).toHaveBeenCalledWith('new-session')
  })
})

describe('writeActiveSessionFile', () => {
  let dir = ''

  afterEach(() => {
    if (dir) {
      rmSync(dir, { force: true, recursive: true })
      dir = ''
    }
  })

  it('writes the actual resumed session id for the shell exit summary', () => {
    dir = mkdtempSync(join(tmpdir(), 'hermes-tui-active-'))
    const path = join(dir, 'active.json')

    writeActiveSessionFile('actual_session', path)

    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({ session_id: 'actual_session' })
  })
})

describe('live session activation in-flight state', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ streaming: true })
  })

  it('keeps the in-flight user prompt in history and hydrates partial assistant text', () => {
    const inflight = { assistant: 'partial answer', streaming: true, user: 'write a long answer' }

    expect(liveSessionInflightMessages(inflight)).toEqual([{ role: 'user', text: 'write a long answer' }])

    hydrateLiveSessionInflight(inflight)

    expect(turnController.bufRef).toBe('partial answer')
    expect(getTurnState().streaming).toBe('partial answer')
  })

  it('ignores empty in-flight payloads', () => {
    expect(liveSessionInflightMessages({ assistant: '', streaming: false, user: '   ' })).toEqual([])

    hydrateLiveSessionInflight({ assistant: '', streaming: false, user: '' })

    expect(turnController.bufRef).toBe('')
    expect(getTurnState().streaming).toBe('')
  })
})
describe.each(['resume', 'activate'] as const)('%s response turn fence', kind => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  const activeResponse = (sessionId: string, revision: number) => ({
    info: {
      model: 'test-model',
      running: true,
      skills: {},
      tools: {},
      turn_generation: 1,
      turn_origin: 'notification' as const,
      turn_state_revision: revision
    },
    messages: [],
    running: true,
    session_id: sessionId,
    status: 'working' as const,
    turn_generation: 1,
    turn_origin: 'notification' as const,
    turn_state_revision: revision,
    ...(kind === 'resume' ? { resumed: sessionId } : {})
  })

  it('adopts the target revision sequence after the source starts and settles', () => {
    patchUiState({ sid: 'session-a' })
    turnController.reconcileTurn(null, 40, false, 50)
    const requestFence = captureSessionResponseTurnFence()

    // Session A keeps running while the request for B is in flight.
    turnController.reconcileTurn('notification', 41, true, 51)
    turnController.reconcileTurn('notification', 41, false, 52)

    const reconciled = reconcileSessionResponseTurn(activeResponse('session-b', 1), requestFence)

    expect(reconciled.running).toBe(true)
    expect(reconciled.turn).toEqual({
      turnGeneration: 1,
      turnOrigin: 'notification',
      turnStateRevision: 1
    })
    expect(reconciled.info).toMatchObject({
      running: true,
      turn_generation: 1,
      turn_origin: 'notification',
      turn_state_revision: 1
    })

    // Match resetSession() + patchTurnState() in the lifecycle callbacks.
    turnController.fullReset()
    patchTurnState(reconciled.turn)
    patchUiState({ busy: reconciled.running, info: reconciled.info, sid: 'session-b' })

    expect(getUiState()).toMatchObject({ busy: true, sid: 'session-b' })
    expect(turnController.reconcileTurn('notification', 1, false, 2)).toBe(true)
    expect(turnController.reconcileTurn('user', 2, true, 3)).toBe(true)
    expect(turnController.reconcileTurn('user', 2, false, 4)).toBe(true)
    expect(turnController.reconcileTurn(null, 2, false, 4)).toBe(true)
    expect(getTurnState()).toMatchObject({ turnGeneration: 2, turnOrigin: null, turnStateRevision: 4 })
    expect(getUiState().busy).toBe(false)
  })

  it('keeps same-session state that settles ahead of a stale response', () => {
    patchUiState({ sid: 'session-a' })
    turnController.reconcileTurn('notification', 6, true, 50)
    const requestFence = captureSessionResponseTurnFence()

    turnController.reconcileTurn(null, 6, false, 51)

    const reconciled = reconcileSessionResponseTurn(activeResponse('session-a', 50), requestFence)

    expect(reconciled.running).toBe(false)
    expect(reconciled.turn).toEqual({
      turnGeneration: 6,
      turnOrigin: null,
      turnStateRevision: 51
    })
    expect(reconciled.info).toMatchObject({
      running: false,
      turn_generation: 6,
      turn_origin: null,
      turn_state_revision: 51
    })
  })
})

describe('resume scroll settle', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('re-snaps while sticky and stops when the user scrolls away', () => {
    vi.useFakeTimers()
    let sticky = true
    let lastManualScrollAt = 0
    const scrollToBottom = vi.fn()

    const cancel = scheduleResumeScrollToBottom(
      {
        current: {
          getLastManualScrollAt: () => lastManualScrollAt,
          isSticky: () => sticky,
          scrollToBottom
        }
      } as any,
      [0, 80, 240]
    )

    vi.advanceTimersByTime(0)
    expect(scrollToBottom).toHaveBeenCalledTimes(1)

    vi.advanceTimersByTime(80)
    expect(scrollToBottom).toHaveBeenCalledTimes(2)

    sticky = false
    lastManualScrollAt = Date.now() + 1
    vi.advanceTimersByTime(160)
    expect(scrollToBottom).toHaveBeenCalledTimes(2)

    cancel()
  })

  it('cancels pending resume snaps', () => {
    vi.useFakeTimers()
    const scrollToBottom = vi.fn()

    const cancel = scheduleResumeScrollToBottom(
      {
        current: {
          getLastManualScrollAt: () => 0,
          isSticky: () => true,
          scrollToBottom
        }
      } as any,
      [20]
    )

    cancel()
    vi.advanceTimersByTime(20)

    expect(scrollToBottom).not.toHaveBeenCalled()
  })

  it('keeps the immediate resume snap even before sticky state settles', () => {
    vi.useFakeTimers()
    let sticky = false
    const scrollToBottom = vi.fn()

    const cancel = scheduleResumeScrollToBottom(
      {
        current: {
          getLastManualScrollAt: () => 0,
          isSticky: () => sticky,
          scrollToBottom
        }
      } as any,
      [0, 80]
    )

    vi.advanceTimersByTime(0)
    expect(scrollToBottom).toHaveBeenCalledTimes(1)

    vi.advanceTimersByTime(80)
    expect(scrollToBottom).toHaveBeenCalledTimes(1)

    sticky = true
    cancel()
  })
})
