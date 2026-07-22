import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GridTestState } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import {
  applyVoiceRecordResponse,
  dismissSensitivePrompt,
  handleIdleHotkeyExit,
  handleStackedModalInput,
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

// Review on #20379 (finding 3): a dialog stacked over /grid-test was
// visually modal but did not receive input — the grid branch ran first, so
// every advertised close key (Esc/q/Enter) mutated the hidden grid instead
// of closing the visible dialog. Input routing must follow visual stacking.
describe('handleStackedModalInput — dialog over grid-test', () => {
  const baseModalKey = {
    ctrl: false,
    downArrow: false,
    escape: false,
    leftArrow: false,
    return: false,
    rightArrow: false,
    upArrow: false
  }

  const grid: GridTestState = {
    activeCol: 1,
    activeRow: 1,
    areas: false,
    cols: 4,
    gap: null,
    nested: false,
    paddingX: null,
    rows: 3,
    streamFocus: 0,
    streamMain: 0,
    streams: false,
    zoomed: false
  }

  beforeEach(() => {
    resetOverlayState()
    patchOverlayState({ gridTest: { ...grid } })
  })

  const openDialogViaD = () => {
    expect(handleStackedModalInput(getOverlayState(), baseModalKey, 'd')).toBe(true)
    expect(getOverlayState().dialog).not.toBeNull()
    expect(getOverlayState().gridTest).not.toBeNull()
  }

  it.each([
    ['Esc', { ...baseModalKey, escape: true }, ''],
    ['q', baseModalKey, 'q'],
    ['Enter', { ...baseModalKey, return: true }, ''],
    ['Ctrl+C', { ...baseModalKey, ctrl: true }, 'c']
  ])('%s closes only the dialog, leaving the grid untouched', (_label, key, ch) => {
    openDialogViaD()

    const gridBefore = getOverlayState().gridTest

    expect(handleStackedModalInput(getOverlayState(), key, ch)).toBe(true)
    expect(getOverlayState().dialog).toBeNull()
    // The grid must be byte-identical: not closed, not zoomed, not reset.
    expect(getOverlayState().gridTest).toBe(gridBefore)
  })

  it('after the dialog closes, the same keys route to the grid again', () => {
    openDialogViaD()
    handleStackedModalInput(getOverlayState(), { ...baseModalKey, escape: true }, '')
    expect(getOverlayState().dialog).toBeNull()

    // Esc now closes the grid — the dialog no longer shields it.
    expect(handleStackedModalInput(getOverlayState(), { ...baseModalKey, escape: true }, '')).toBe(true)
    expect(getOverlayState().gridTest).toBeNull()
  })

  it('the dialog swallows grid keys entirely while open (no leak-through)', () => {
    openDialogViaD()

    const gridBefore = getOverlayState().gridTest

    // 'a' toggles areas mode when the grid has focus — it must not now.
    expect(handleStackedModalInput(getOverlayState(), baseModalKey, 'a')).toBe(true)
    expect(getOverlayState().gridTest).toBe(gridBefore)
    expect(getOverlayState().dialog).not.toBeNull()
  })

  it('stacking works from streams mode too', () => {
    patchOverlayState({ gridTest: { ...grid, streams: true } })
    openDialogViaD()

    expect(handleStackedModalInput(getOverlayState(), { ...baseModalKey, return: true }, '')).toBe(true)
    expect(getOverlayState().dialog).toBeNull()
    expect(getOverlayState().gridTest?.streams).toBe(true)
  })

  it('reports unconsumed when neither modal is up', () => {
    resetOverlayState()
    expect(handleStackedModalInput(getOverlayState(), baseModalKey, 'x')).toBe(false)
  })
})
