import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/ink', () => ({ evictInkCaches: vi.fn() }))

import { pendingPromptOverlay } from '../app/pendingPromptOverlay.js'
import { turnController } from '../app/turnController.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import {
  hydrateLiveSessionInflight,
  liveSessionInflightMessages,
  restorePendingPrompt,
  scheduleResumeScrollToBottom,
  signalFreshSessionBoundary,
  writeActiveSessionFile
} from '../app/useSessionLifecycle.js'
import type { SessionPendingPrompt } from '../gatewayTypes.js'

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

describe('pending prompt restoration', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('restores a clarify picker after resuming a live session', () => {
    expect(
      restorePendingPrompt({
        event: 'clarify.request',
        payload: { choices: ['a', 'b', 'c', 'd', 'e'], question: 'Which option?', request_id: 'clarify-1' }
      })
    ).toBe(true)

    expect(getOverlayState().clarify).toEqual({
      choices: ['a', 'b', 'c', 'd', 'e'],
      question: 'Which option?',
      requestId: 'clarify-1'
    })
    expect(getUiState().status).toBe('waiting for input…')
  })

  it.each([
    [
      { event: 'sudo.request', payload: { request_id: 'sudo-1' } },
      { sudo: { requestId: 'sudo-1' } },
      'sudo password needed'
    ],
    [
      {
        event: 'secret.request',
        payload: { env_var: 'API_KEY', prompt: 'Paste key', request_id: 'secret-1' }
      },
      { secret: { envVar: 'API_KEY', prompt: 'Paste key', requestId: 'secret-1' } },
      'secret input needed'
    ]
  ] as const)('restores %s prompts', (prompt, overlay, status) => {
    expect(restorePendingPrompt(prompt)).toBe(true)
    expect(getOverlayState()).toMatchObject(overlay)
    expect(getUiState().status).toBe(status)
  })

  it('does nothing when the resumed session has no pending prompt', () => {
    expect(restorePendingPrompt()).toBe(false)
    expect(getOverlayState().clarify).toBeNull()
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

describe('pending prompt overlay mapping', () => {
  // The live event path and the rehydration path used to carry their own copy
  // of this mapping. A field added to one payload would then be dropped on the
  // rehydration side only — an overlay that renders with a missing question or
  // env var, but exclusively after a session switch. Both now funnel through
  // pendingPromptOverlay; these lock the shape so the split cannot come back
  // unnoticed.
  const PROMPTS: SessionPendingPrompt[] = [
    {
      event: 'clarify.request',
      payload: { choices: ['a', 'b'], question: 'Which option?', request_id: 'clarify-9' }
    },
    { event: 'sudo.request', payload: { request_id: 'sudo-9' } },
    {
      event: 'secret.request',
      payload: { env_var: 'OPENAI_API_KEY', prompt: 'Paste the key', request_id: 'secret-9' }
    }
  ]

  it('maps every rehydratable prompt to an overlay and a status', () => {
    for (const prompt of PROMPTS) {
      const mapped = pendingPromptOverlay(prompt)

      expect(mapped, prompt.event).not.toBeNull()
      expect(Object.keys(mapped!.overlay), prompt.event).toHaveLength(1)
      expect(mapped!.status, prompt.event).toBeTruthy()
    }
  })

  it('carries every payload field into the overlay', () => {
    expect(pendingPromptOverlay(PROMPTS[0])!.overlay).toEqual({
      clarify: { choices: ['a', 'b'], question: 'Which option?', requestId: 'clarify-9' }
    })
    expect(pendingPromptOverlay(PROMPTS[1])!.overlay).toEqual({ sudo: { requestId: 'sudo-9' } })
    expect(pendingPromptOverlay(PROMPTS[2])!.overlay).toEqual({
      secret: { envVar: 'OPENAI_API_KEY', prompt: 'Paste the key', requestId: 'secret-9' }
    })
  })

  it('produces the same overlay whether the prompt arrives live or on resume', () => {
    // The invariant the extraction exists to hold: a prompt replayed by the
    // gateway must leave the UI in the state the original event would have.
    for (const prompt of PROMPTS) {
      resetOverlayState()
      resetUiState()
      expect(restorePendingPrompt(prompt)).toBe(true)
      const afterResume = { overlay: getOverlayState(), status: getUiState().status }

      resetOverlayState()
      resetUiState()
      const mapped = pendingPromptOverlay(prompt)!

      patchOverlayState(mapped.overlay)
      patchUiState({ status: mapped.status })

      expect({ overlay: getOverlayState(), status: getUiState().status }, prompt.event).toEqual(afterResume)
    }
  })

  it('ignores a prompt with no overlay to render', () => {
    expect(pendingPromptOverlay(undefined)).toBeNull()
    expect(pendingPromptOverlay(null)).toBeNull()
  })
})
