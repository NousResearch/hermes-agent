import { beforeEach, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

// T9: a shared-owner session's voice mode is the owner's per-session /voice state, the one
// voice.record gates on; the sidecar's voice.toggle flips only its process-global flag.

beforeEach(() => resetUiState())

it('turns on the shared owner\'s per-session voice mode, not the sidecar flag, before a wake capture', async () => {
  const request = vi.fn(async () => null)
  const rpc = vi.fn(async () => null)

  const ctx = {
    composer: { setInput: vi.fn() },
    gateway: { gw: { isCanonical: true, request }, rpc },
    session: { colsRef: { current: 80 }, newSession: vi.fn() },
    submission: { submitRef: { current: vi.fn() } },
    system: { sys: vi.fn() },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  } as any

  patchUiState({ sid: 'owner-session' })
  createGatewayEventHandler(ctx)({ payload: { start_new_session: false }, type: 'wake.detected' } as any)

  await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('voice.record', { action: 'start', session_id: 'owner-session' }))
  expect(request).toHaveBeenCalledWith('slash.exec', { command: 'voice on', session_id: 'owner-session' })
  expect(rpc).not.toHaveBeenCalledWith('voice.toggle', expect.anything())
})
