import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  acknowledgeDetachedTask,
  registerDetachedTasks,
  resetDetachedTaskTrackingForTests
} from '../app/detachedTasks.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import {
  hydrateLiveSessionInflight,
  liveSessionInflightMessages,
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

describe('detached task registration retries', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    resetUiState()
    resetDetachedTaskTrackingForTests()
  })

  afterEach(() => {
    resetDetachedTaskTrackingForTests()
    vi.useRealTimers()
  })

  it('retries a null ACK and consumes a later terminal snapshot once', async () => {
    const task = {
      source_session_id: 'source-sid',
      status: 'running' as const,
      task_id: 'bg_turn_retry'
    }

    patchUiState({ sid: 'owner-sid' })
    registerDetachedTasks([task])

    let ackCalls = 0

    const rpc = vi.fn(async (method: string) => {
      if (method === 'session.detach_turn_ack') {
        ackCalls += 1

        return ackCalls === 1
          ? null
          : {
              task: {
                ...task,
                status: 'complete' as const,
                text: 'terminal snapshot'
              }
            }
      }

      if (method === 'session.detach_turn_consumed') {
        return { consumed: true }
      }

      return null
    })

    const sys = vi.fn()

    const pending = acknowledgeDetachedTask(task, 'owner-sid', rpc as any, sys)
    await Promise.resolve()
    expect(ackCalls).toBe(1)

    await vi.advanceTimersByTimeAsync(50)
    await pending
    await Promise.resolve()

    expect(ackCalls).toBe(2)
    expect(sys).toHaveBeenCalledTimes(1)
    expect(sys).toHaveBeenCalledWith('[bg bg_turn_retry] terminal snapshot')
    expect(rpc).toHaveBeenCalledWith('session.detach_turn_consumed', {
      session_id: 'owner-sid',
      task_id: 'bg_turn_retry'
    })
    expect(getUiState().bgTasks.has('bg_turn_retry')).toBe(false)
  })

  it('cancels retries silently when the presentation owner switches', async () => {
    const task = {
      source_session_id: 'source-sid',
      status: 'running' as const,
      task_id: 'bg_turn_switched'
    }

    patchUiState({ sid: 'owner-sid' })
    registerDetachedTasks([task])
    const rpc = vi.fn(async () => null)
    const sys = vi.fn()

    const pending = acknowledgeDetachedTask(task, 'owner-sid', rpc as any, sys)
    await Promise.resolve()
    expect(rpc).toHaveBeenCalledTimes(1)

    patchUiState({ sid: 'other-sid' })
    await vi.runAllTimersAsync()
    await pending

    expect(rpc).toHaveBeenCalledTimes(1)
    expect(sys).not.toHaveBeenCalled()
    expect(getUiState().bgTasks.has('bg_turn_switched')).toBe(true)
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
