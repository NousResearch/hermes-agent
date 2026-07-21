import { describe, expect, it, vi } from 'vitest'

import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import {
  applyVoiceRecordResponse,
  dismissApprovalPrompt,
  dismissSensitivePrompt,
  handleIdleHotkeyExit,
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

    const pending = dismissSensitivePrompt(getOverlayState(), rpc, sys, 'owner-sid')

    expect(getOverlayState().sudo).toBeNull()
    expect(sys).toHaveBeenCalledWith('sudo cancelled')
    expect(rpc).toHaveBeenCalledWith('sudo.respond', {
      password: '',
      request_id: 'sudo-1',
      session_id: 'owner-sid'
    })
    await pending
  })

  it('clears a secret overlay before a stale cancel RPC resolves', async () => {
    resetOverlayState()
    patchOverlayState({ secret: { envVar: 'API_KEY', prompt: 'Enter API key', requestId: 'secret-1' } })
    const rpc = vi.fn().mockResolvedValue(null)
    const sys = vi.fn()

    const pending = dismissSensitivePrompt(getOverlayState(), rpc, sys, 'owner-sid')

    expect(getOverlayState().secret).toBeNull()
    expect(sys).toHaveBeenCalledWith('secret entry cancelled')
    expect(rpc).toHaveBeenCalledWith('secret.respond', {
      request_id: 'secret-1',
      session_id: 'owner-sid',
      value: ''
    })
    await pending
  })

  it('does not erase a newer sudo prompt B when acting on a stale sudo A snapshot', async () => {
    resetOverlayState()
    patchOverlayState({ sudo: { requestId: 'sudo-A' } })
    const staleOverlay = getOverlayState()

    // Newer sudo prompt B has already taken the slot by the time the Ctrl+C
    // handler (holding a stale render snapshot) fires.
    patchOverlayState({ sudo: { requestId: 'sudo-B' } })

    const rpc = vi.fn().mockResolvedValue(null)
    const sys = vi.fn()

    const pending = dismissSensitivePrompt(staleOverlay, rpc, sys, 'owner-sid')

    expect(getOverlayState().sudo).toEqual({ requestId: 'sudo-B' })
    expect(rpc).toHaveBeenCalledWith('sudo.respond', {
      password: '',
      request_id: 'sudo-A',
      session_id: 'owner-sid'
    })
    expect(sys).toHaveBeenCalledWith('sudo cancelled')
    await pending
  })

  it('does not erase a newer secret prompt B when acting on a stale secret A snapshot', async () => {
    resetOverlayState()
    patchOverlayState({ secret: { envVar: 'API_KEY', prompt: 'Enter API key', requestId: 'secret-A' } })
    const staleOverlay = getOverlayState()

    patchOverlayState({ secret: { envVar: 'API_KEY', prompt: 'Enter API key', requestId: 'secret-B' } })

    const rpc = vi.fn().mockResolvedValue(null)
    const sys = vi.fn()

    const pending = dismissSensitivePrompt(staleOverlay, rpc, sys, 'owner-sid')

    expect(getOverlayState().secret).toEqual({ envVar: 'API_KEY', prompt: 'Enter API key', requestId: 'secret-B' })
    expect(rpc).toHaveBeenCalledWith('secret.respond', {
      request_id: 'secret-A',
      session_id: 'owner-sid',
      value: ''
    })
    expect(sys).toHaveBeenCalledWith('secret entry cancelled')
    await pending
  })
})

describe('dismissApprovalPrompt — Ctrl+C dismissal must not erase a newer approval prompt', () => {
  const deferred = <T>() => {
    let resolve!: (value: T) => void

    const promise = new Promise<T>(r => {
      resolve = r
    })

    return { promise, resolve }
  }

  it("captures A's requestId, denies A only after the RPC resolves, and leaves a newer B installed while A's RPC was in flight intact", async () => {
    resetOverlayState()
    resetTurnState()
    resetUiState()
    patchOverlayState({ approval: { command: 'rm -rf /tmp/x', description: 'delete tmp', requestId: 'req-A' } })
    patchUiState({ status: 'approval needed' })

    const rpcDeferred = deferred<{ ok: true }>()
    const rpc = vi.fn().mockReturnValue(rpcDeferred.promise)
    const staleOverlaySnapshot = getOverlayState()

    const pending = dismissApprovalPrompt(staleOverlaySnapshot, rpc, 'sid-1')

    // Backend FIFO removes A and installs a newer B (same kind) before A's
    // deny RPC resolves.
    patchOverlayState({ approval: { command: 'sudo reboot', description: 'reboot', requestId: 'req-B' } })
    patchUiState({ status: 'approval needed' })

    rpcDeferred.resolve({ ok: true })
    await pending

    expect(rpc).toHaveBeenCalledWith('approval.respond', {
      choice: 'deny',
      request_id: 'req-A',
      session_id: 'sid-1'
    })
    expect(getOverlayState().approval).toEqual({ command: 'sudo reboot', description: 'reboot', requestId: 'req-B' })
    expect(getTurnState().outcome).toBe('denied')
    expect(getUiState().status).toBe('approval needed')
  })

  it('clears its own approval prompt once denied when no newer prompt has superseded it', async () => {
    resetOverlayState()
    resetTurnState()
    patchOverlayState({ approval: { command: 'rm -rf /tmp/x', description: 'delete tmp', requestId: 'req-A' } })

    const rpc = vi.fn().mockResolvedValue({ ok: true })

    await dismissApprovalPrompt(getOverlayState(), rpc, 'sid-1')

    expect(getOverlayState().approval).toBeNull()
    expect(getTurnState().outcome).toBe('denied')
  })

  it('does not deny or touch the overlay when the RPC fails (resolves null)', async () => {
    resetOverlayState()
    resetTurnState()
    patchOverlayState({ approval: { command: 'rm -rf /tmp/x', description: 'delete tmp', requestId: 'req-A' } })

    const rpc = vi.fn().mockResolvedValue(null)

    await dismissApprovalPrompt(getOverlayState(), rpc, 'sid-1')

    expect(getOverlayState().approval).toEqual({ command: 'rm -rf /tmp/x', description: 'delete tmp', requestId: 'req-A' })
    expect(getTurnState().outcome).toBe('')
  })
})
