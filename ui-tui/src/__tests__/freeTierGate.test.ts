import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $freeTierBlocks } from '../app/freeTierGate.js'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import type { GatewayEventHandlerContext } from '../app/interfaces.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { FREE_TIER_LIMIT_KEY, FREE_TIER_LIMIT_TITLE } from '../content/setup.js'
import type { GatewayEvent } from '../gatewayTypes.js'

function context() {
  return {
    composer: { dequeue: vi.fn(), queueEditRef: { current: null }, sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn() }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '', colsRef: { current: 80 }, newSession: vi.fn(),
      resetSession: vi.fn(), resumeById: vi.fn(), setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }
}

const limitNotice: GatewayEvent = {
  type: 'notification.show', session_id: 'focused',
  payload: { key: FREE_TIER_LIMIT_KEY, kind: 'sticky', level: 'info', text: 'Choose a provider to continue.' }
}

describe('free-tier continuation in TUI and embedded web chat', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    $freeTierBlocks.set({})
    resetTurnState()
    turnController.fullReset()
    patchUiState({ sid: 'focused' })
  })

  it('finishes the active reply before presenting existing provider choices without consuming a draft', () => {
    const ctx = context()
    const handle = createGatewayEventHandler(ctx as unknown as GatewayEventHandlerContext)

    handle({ type: 'message.start', session_id: 'focused', payload: {} })
    handle(limitNotice)
    expect(getUiState().busy).toBe(true)
    expect(getOverlayState().modelPicker).toBe(false)
    expect(ctx.transcript.panel).not.toHaveBeenCalled()

    handle({ type: 'message.complete', session_id: 'focused', payload: { text: 'Your work is ready.' } })
    expect(ctx.transcript.appendMessage.mock.calls[0]?.[0]).toMatchObject({ text: 'Your work is ready.' })
    expect(ctx.transcript.panel).toHaveBeenCalledWith(FREE_TIER_LIMIT_TITLE, expect.any(Array))
    expect(getUiState().busy).toBe(false)
    const picker = getOverlayState().modelPicker
    expect(typeof picker).toBe('object')
    if (typeof picker !== 'object') throw new Error('missing continuation picker')
    picker.onSignIn?.()
    picker.onSetup?.()
    expect(ctx.submission.submitRef.current.mock.calls).toEqual([['/login'], ['/setup model']])
    expect(ctx.composer.setInput).not.toHaveBeenCalled()
    handle(limitNotice)
    expect(ctx.transcript.panel).toHaveBeenCalledTimes(1)
    handle({ type: 'notification.show', session_id: 'focused', payload: {
      key: 'free_tier.login', text: 'https://example.com/device\nABCD-EFGH\nWaiting for sign-in.'
    } })
    expect(ctx.system.sys).toHaveBeenCalledWith(expect.stringContaining('https://example.com/device'))
    expect($freeTierBlocks.get().focused).toBeDefined()
    handle({ type: 'notification.clear', session_id: 'focused', payload: { key: FREE_TIER_LIMIT_KEY } })
    expect($freeTierBlocks.get().focused).toBeUndefined()
  })

  it('isolates foreign sessions and reopens recovery on typed refusal without disguising it as a crash', () => {
    const ctx = context()
    const handle = createGatewayEventHandler(ctx as unknown as GatewayEventHandlerContext)
    handle({ ...limitNotice, session_id: 'background' })
    expect(getOverlayState().modelPicker).toBe(false)
    expect(ctx.transcript.panel).not.toHaveBeenCalled()

    handle({ type: 'error', session_id: 'focused', payload: { code: 'free_tier_limit', message: 'Sign in or use your own model.' } })
    expect(getUiState().busy).toBe(false)
    expect(getOverlayState().modelPicker).toMatchObject({ continuationMessage: 'Sign in or use your own model.' })
    expect(ctx.system.sys).not.toHaveBeenCalled()
    expect(ctx.session.newSession).not.toHaveBeenCalled()
  })
})
