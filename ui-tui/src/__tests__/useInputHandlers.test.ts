import { describe, expect, it, vi } from 'vitest'

import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { rememberServerRequest, resetServerRequestsForTests } from '../app/serverRequestStore.js'
import {
  applyVoiceRecordResponse,
  composerHasDraft,
  dismissSensitivePrompt,
  handleIdleHotkeyExit,
  handleTranscriptBoundaryJump,
  resolveCtrlCComposerAction,
  resolveDoubleEscAction,
  shouldDetachEditedHistoryInput,
  shouldFallThroughForScroll,
  transcriptBoundaryIntent
} from '../app/useInputHandlers.js'

const baseKey = {
  ctrl: false,
  downArrow: false,
  end: false,
  home: false,
  meta: false,
  pageDown: false,
  pageUp: false,
  shift: false,
  super: false,
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

  it('falls through for Ctrl+Home / Ctrl+End', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, ctrl: true, home: true })).toBe(true)
    expect(shouldFallThroughForScroll({ ...baseKey, ctrl: true, end: true })).toBe(true)
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

describe('Ctrl+Home / Ctrl+End transcript boundary jumps (#65308)', () => {
  it('recognizes only the unmodified Ctrl chords', () => {
    expect(transcriptBoundaryIntent({ ...baseKey, ctrl: true, home: true })).toBe('top')
    expect(transcriptBoundaryIntent({ ...baseKey, ctrl: true, end: true })).toBe('bottom')
    expect(transcriptBoundaryIntent({ ...baseKey, home: true })).toBeNull()
    expect(transcriptBoundaryIntent({ ...baseKey, ctrl: true, shift: true, home: true })).toBeNull()
    expect(transcriptBoundaryIntent({ ...baseKey, ctrl: true, meta: true, end: true })).toBeNull()
  })

  it('jumps to the top and clears any transcript selection', () => {
    const scroll = { scrollTo: vi.fn(), scrollToBottom: vi.fn() }
    const selection = { clearSelection: vi.fn() }

    expect(
      handleTranscriptBoundaryJump(
        { ...baseKey, ctrl: true, home: true },
        { scrollRef: { current: scroll as any }, selection: selection as any }
      )
    ).toBe(true)
    expect(selection.clearSelection).toHaveBeenCalledTimes(1)
    expect(scroll.scrollTo).toHaveBeenCalledWith(0)
    expect(scroll.scrollToBottom).not.toHaveBeenCalled()
  })

  it('jumps to the sticky bottom so new output keeps following', () => {
    const scroll = { scrollTo: vi.fn(), scrollToBottom: vi.fn() }
    const selection = { clearSelection: vi.fn() }

    expect(
      handleTranscriptBoundaryJump(
        { ...baseKey, ctrl: true, end: true },
        { scrollRef: { current: scroll as any }, selection: selection as any }
      )
    ).toBe(true)
    expect(selection.clearSelection).toHaveBeenCalledTimes(1)
    expect(scroll.scrollToBottom).toHaveBeenCalledTimes(1)
    expect(scroll.scrollTo).not.toHaveBeenCalled()
  })

  it('leaves plain Home / End to the composer', () => {
    const scroll = { scrollTo: vi.fn(), scrollToBottom: vi.fn() }
    const selection = { clearSelection: vi.fn() }

    expect(
      handleTranscriptBoundaryJump(
        { ...baseKey, home: true },
        { scrollRef: { current: scroll as any }, selection: selection as any }
      )
    ).toBe(false)
    expect(selection.clearSelection).not.toHaveBeenCalled()
    expect(scroll.scrollTo).not.toHaveBeenCalled()
    expect(scroll.scrollToBottom).not.toHaveBeenCalled()
  })
})

describe('composerHasDraft — Ctrl+D exits only from an empty composer (#116443)', () => {
  it('is false for an empty composer and true for text, multi-line buffer or attachments', () => {
    expect(composerHasDraft({ input: '', inputBuf: [], tokens: [] })).toBe(false)
    expect(composerHasDraft({ input: 'hi', inputBuf: [], tokens: [] })).toBe(true)
    expect(composerHasDraft({ input: '', inputBuf: ['line 1'], tokens: [] })).toBe(true)
    expect(composerHasDraft({ input: '', inputBuf: [], tokens: [{ kind: 'image' }] })).toBe(true)
  })
})

describe('shouldDetachEditedHistoryInput', () => {
  const history = ['older message', 'line one\nline two']

  it('detaches a recalled entry as soon as the user edits it', () => {
    expect(shouldDetachEditedHistoryInput(1, history, 'line one edited\nline two')).toBe(true)
  })

  it('keeps unchanged recalled entries in history navigation', () => {
    expect(shouldDetachEditedHistoryInput(1, history, 'line one\nline two')).toBe(false)
  })

  it('does not detach an ordinary current draft', () => {
    expect(shouldDetachEditedHistoryInput(null, history, 'new draft')).toBe(false)
  })
})

describe('resolveCtrlCComposerAction — draft wins over interrupt', () => {
  it('clears a non-empty composer even while the agent is streaming', () => {
    expect(resolveCtrlCComposerAction({ busy: true, hasDraft: true, hasSession: true })).toBe('clear')
  })

  it('interrupts a running turn when the composer is empty', () => {
    expect(resolveCtrlCComposerAction({ busy: true, hasDraft: false, hasSession: true })).toBe('interrupt')
  })

  it('clears an idle composer instead of exiting', () => {
    expect(resolveCtrlCComposerAction({ busy: false, hasDraft: true, hasSession: true })).toBe('clear')
  })

  it('exits when idle with an empty composer', () => {
    expect(resolveCtrlCComposerAction({ busy: false, hasDraft: false, hasSession: true })).toBe('exit')
  })

  it('does not interrupt a busy session that has no sid yet', () => {
    expect(resolveCtrlCComposerAction({ busy: true, hasDraft: false, hasSession: false })).toBe('exit')
  })
})

describe('resolveDoubleEscAction — double-Esc interrupts only a busy, draft-free turn (#62478)', () => {
  it('interrupts the running turn when the composer is empty', () => {
    expect(resolveDoubleEscAction({ busy: true, hasDraft: false, hasSession: true })).toBe('interrupt')
  })

  it('clears a draft instead of interrupting, even mid-stream', () => {
    expect(resolveDoubleEscAction({ busy: true, hasDraft: true, hasSession: true })).toBe('clear')
  })

  it('does nothing while idle with an empty composer (single-Esc conventions unchanged)', () => {
    expect(resolveDoubleEscAction({ busy: false, hasDraft: false, hasSession: true })).toBe('none')
  })

  it('does not interrupt a busy session that has no sid yet', () => {
    expect(resolveDoubleEscAction({ busy: true, hasDraft: false, hasSession: false })).toBe('none')
  })

  it('keeps clearing an idle draft (existing discard convention)', () => {
    expect(resolveDoubleEscAction({ busy: false, hasDraft: true, hasSession: true })).toBe('clear')
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
    expect(actions.sys).toHaveBeenCalled()
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
    expect(sys).toHaveBeenCalled()
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
  const openRequest = (id: string, method: string) => {
    const respond = vi.fn()

    rememberServerRequest({ fail: vi.fn(), id, method, params: {}, respond })

    return respond
  }

  it('clears a sudo overlay and answers the server request with an empty value', () => {
    resetOverlayState()
    resetServerRequestsForTests()
    patchOverlayState({ sudo: { requestId: 'srq-sudo' } })
    const respond = openRequest('srq-sudo', 'sudo')
    const sys = vi.fn()

    dismissSensitivePrompt(getOverlayState(), vi.fn(), sys)

    expect(getOverlayState().sudo).toBeNull()
    expect(sys).toHaveBeenCalled()
    expect(respond).toHaveBeenCalledWith({ value: '' })
  })

  it('clears a secret overlay even when its request already expired (nothing left to answer)', () => {
    resetOverlayState()
    resetServerRequestsForTests()
    patchOverlayState({ secret: { envVar: 'API_KEY', prompt: 'Enter API key', requestId: 'srq-gone' } })
    const sys = vi.fn()

    dismissSensitivePrompt(getOverlayState(), vi.fn(), sys)

    expect(getOverlayState().secret).toBeNull()
    expect(sys).toHaveBeenCalled()
  })
})
