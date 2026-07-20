import { describe, expect, it, vi } from 'vitest'

import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { copyLatestAssistantResponse } from '../app/slash/commands/core.js'
import {
  applyVoiceRecordResponse,
  consumeDashboardNativeSubmission,
  dismissSensitivePrompt,
  handleDashboardCopyLastShortcut,
  handleDashboardNativeSubmission,
  handleIdleHotkeyExit,
  rememberNativeSubmitRequest,
  shouldAllowIdleHotkeyExit,
  shouldFallThroughForScroll
} from '../app/useInputHandlers.js'

const baseKey = {
  downArrow: false,
  pageDown: false,
  pageUp: false,
  shift: false,
  upArrow: false,
  wheelDown: false,
  wheelUp: false
}

describe('shouldFallThroughForScroll — keep transcript scrolling alive during prompt overlays', () => {
  it('falls through for wheel scrolls', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, wheelUp: true })).toBe(true)
    expect(shouldFallThroughForScroll({ ...baseKey, wheelDown: true })).toBe(true)
  })

  it('falls through for PageUp / PageDown', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, pageUp: true })).toBe(true)
    expect(shouldFallThroughForScroll({ ...baseKey, pageDown: true })).toBe(true)
  })

  it('falls through for Shift+ArrowUp / Shift+ArrowDown', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, shift: true, upArrow: true })).toBe(true)
    expect(shouldFallThroughForScroll({ ...baseKey, shift: true, downArrow: true })).toBe(true)
  })

  it('does NOT fall through for plain arrows — those drive in-prompt selection', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, upArrow: true })).toBe(false)
    expect(shouldFallThroughForScroll({ ...baseKey, downArrow: true })).toBe(false)
  })

  it('does NOT fall through for plain Shift — without an arrow it is a no-op', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, shift: true })).toBe(false)
  })

  it('does NOT fall through for unrelated state (no scroll keys held)', () => {
    expect(shouldFallThroughForScroll(baseKey)).toBe(false)
  })
})

describe('shouldAllowIdleHotkeyExit', () => {
  it('keeps idle exit hotkeys enabled in normal terminals', () => {
    expect(shouldAllowIdleHotkeyExit(false)).toBe(true)
  })

  it('disables idle exit hotkeys in dashboard chat', () => {
    expect(shouldAllowIdleHotkeyExit(true)).toBe(false)
  })
})

describe('handleDashboardCopyLastShortcut', () => {
  it('copies the latest assistant response without changing draft or selection', async () => {
    const state = { draft: 'keep this draft', selection: 'selected draft text' }
    const write = vi.fn().mockResolvedValue('osc52')
    let copy: Promise<void> | undefined
    const copyLastAssistantResponse = vi.fn(() => {
      expect(state).toEqual({ draft: 'keep this draft', selection: 'selected draft text' })
      copy = copyLatestAssistantResponse(
        [
          { role: 'assistant', text: 'older response' },
          { role: 'assistant', text: 'latest response' }
        ],
        vi.fn(),
        write
      )
    })

    expect(
      handleDashboardCopyLastShortcut(
        { copyLastAssistantResponse },
        { ctrl: true, super: true },
        'c',
        true
      )
    ).toBe(true)
    expect(copyLastAssistantResponse).toHaveBeenCalledOnce()
    await copy
    expect(write).toHaveBeenCalledWith('latest response')
    expect(state).toEqual({ draft: 'keep this draft', selection: 'selected draft text' })
  })

  it('ignores the sequence outside dashboard mode', () => {
    const copyLastAssistantResponse = vi.fn()

    expect(
      handleDashboardCopyLastShortcut(
        { copyLastAssistantResponse },
        { ctrl: true, super: true },
        'c',
        false
      )
    ).toBe(false)
    expect(copyLastAssistantResponse).not.toHaveBeenCalled()
  })
})

describe('handleDashboardNativeSubmission', () => {
  const frame = '\x1b_HERMES_SUBMIT;request-1;7ZWc6riAIOuplOyLnOyngA==\x1b\\'

  it('submits exactly the decoded native draft in dashboard mode', () => {
    const submitDashboardNativeDraft = vi.fn()

    expect(handleDashboardNativeSubmission({ submitDashboardNativeDraft }, frame, true)).toBe(true)
    expect(submitDashboardNativeDraft).toHaveBeenCalledWith('한글 메시지', 'request-1')
  })

  it('ignores the protocol outside dashboard mode, malformed data, or whitespace-only drafts', () => {
    const submitDashboardNativeDraft = vi.fn()

    expect(handleDashboardNativeSubmission({ submitDashboardNativeDraft }, frame, false)).toBe(false)
    expect(
      handleDashboardNativeSubmission(
        { submitDashboardNativeDraft },
        '\x1b_HERMES_SUBMIT;request-1;%%%\x1b\\',
        true
      )
    ).toBe(false)
    expect(
      handleDashboardNativeSubmission(
        { submitDashboardNativeDraft },
        '\x1b_HERMES_SUBMIT;request-1;ICAg\x1b\\',
        true
      )
    ).toBe(false)
    expect(submitDashboardNativeDraft).not.toHaveBeenCalled()
  })
})

describe('consumeDashboardNativeSubmission', () => {
  it('dispatches and stops propagation for a valid frame only', () => {
    const submitDashboardNativeDraft = vi.fn()
    const stopImmediatePropagation = vi.fn()
    const event = {
      keypress: { raw: '\x1b_HERMES_SUBMIT;request-1;ZHJhZnQ=\x1b\\' },
      stopImmediatePropagation
    }

    expect(consumeDashboardNativeSubmission({ submitDashboardNativeDraft }, event, true)).toBe(true)
    expect(submitDashboardNativeDraft).toHaveBeenCalledWith('draft', 'request-1')
    expect(stopImmediatePropagation).toHaveBeenCalledOnce()

    event.keypress.raw = '\x1b_HERMES_SUBMIT;request-2;%%%\x1b\\'
    stopImmediatePropagation.mockClear()
    expect(consumeDashboardNativeSubmission({ submitDashboardNativeDraft }, event, true)).toBe(false)
    expect(stopImmediatePropagation).not.toHaveBeenCalled()
  })
})

describe('rememberNativeSubmitRequest', () => {
  it('accepts a request once, rejects replays, and bounds receiver memory', () => {
    const accepted: string[] = []

    expect(rememberNativeSubmitRequest(accepted, 'request-1', 2)).toBe(true)
    expect(rememberNativeSubmitRequest(accepted, 'request-1', 2)).toBe(false)
    expect(rememberNativeSubmitRequest(accepted, 'request-2', 2)).toBe(true)
    expect(rememberNativeSubmitRequest(accepted, 'request-3', 2)).toBe(true)
    expect(accepted).toEqual(['request-2', 'request-3'])
  })
})

describe('handleIdleHotkeyExit', () => {
  it('exits in normal terminals', () => {
    const actions = { die: vi.fn(), sys: vi.fn() }

    handleIdleHotkeyExit(actions, false)

    expect(actions.die).toHaveBeenCalledTimes(1)
    expect(actions.sys).not.toHaveBeenCalled()
  })

  it('asks the dashboard for a fresh chat instead of leaving a ghost session', () => {
    const actions = { die: vi.fn(), sys: vi.fn() }
    const requestDashboardNewSession = vi.fn()

    handleIdleHotkeyExit(actions, true, requestDashboardNewSession)

    expect(actions.die).not.toHaveBeenCalled()
    expect(requestDashboardNewSession).toHaveBeenCalledTimes(1)
    expect(actions.sys).toHaveBeenCalledWith('starting a fresh dashboard chat...')
  })
})

describe('applyVoiceRecordResponse', () => {
  it('reverts optimistic REC state when the gateway reports voice busy', () => {
    const setProcessing = vi.fn()
    const setRecording = vi.fn()
    const sys = vi.fn()

    applyVoiceRecordResponse({ status: 'busy' }, true, { setProcessing, setRecording }, sys)

    expect(setRecording).toHaveBeenCalledWith(false)
    expect(setProcessing).toHaveBeenCalledWith(true)
    expect(sys).toHaveBeenCalledWith('voice: still transcribing; try again shortly')
  })

  it('keeps optimistic REC state for successful recording starts', () => {
    const setProcessing = vi.fn()
    const setRecording = vi.fn()

    applyVoiceRecordResponse({ status: 'recording' }, true, { setProcessing, setRecording }, vi.fn())

    expect(setRecording).not.toHaveBeenCalled()
    expect(setProcessing).not.toHaveBeenCalled()
  })

  it('reverts optimistic REC state when the gateway returns null', () => {
    const setProcessing = vi.fn()
    const setRecording = vi.fn()

    applyVoiceRecordResponse(null, true, { setProcessing, setRecording }, vi.fn())

    expect(setRecording).toHaveBeenCalledWith(false)
    expect(setProcessing).toHaveBeenCalledWith(false)
  })
})

describe('dismissSensitivePrompt', () => {
  it('clears a sudo overlay before a stale cancel RPC resolves', async () => {
    resetOverlayState()
    patchOverlayState({ sudo: { requestId: 'sudo-1' } })
    const rpc = vi.fn().mockResolvedValue(null)
    const sys = vi.fn()

    const pending = dismissSensitivePrompt(getOverlayState(), rpc, sys)

    expect(getOverlayState().sudo).toBeNull()
    expect(sys).toHaveBeenCalledWith('sudo cancelled')
    expect(rpc).toHaveBeenCalledWith('sudo.respond', { password: '', request_id: 'sudo-1' })
    await pending
  })

  it('clears a secret overlay before a stale cancel RPC resolves', async () => {
    resetOverlayState()
    patchOverlayState({ secret: { envVar: 'API_KEY', prompt: 'Enter API key', requestId: 'secret-1' } })
    const rpc = vi.fn().mockResolvedValue(null)
    const sys = vi.fn()

    const pending = dismissSensitivePrompt(getOverlayState(), rpc, sys)

    expect(getOverlayState().secret).toBeNull()
    expect(sys).toHaveBeenCalledWith('secret entry cancelled')
    expect(rpc).toHaveBeenCalledWith('secret.respond', { request_id: 'secret-1', value: '' })
    await pending
  })
})
