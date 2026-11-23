import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getOverlayState, patchOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState } from '../app/uiStore.js'

const ref = <T>(current: T) => ({ current })

const ctx = (sys = vi.fn()) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

describe('a model pick made during a turn that did not apply at the boundary', () => {
  afterEach(() => {
    patchOverlayState({ confirm: null })
  })

  it('names the refusal and shows the committed model again', () => {
    patchUiState({ info: { model: 'picked' } as any, sid: 'sess-1' })
    const sys = vi.fn()

    createGatewayEventHandler(ctx(sys))({
      payload: {
        model: 'frozen',
        operation: 'model',
        reason: 'model_resolution_failed',
        session_id: 'sess-1',
        status: 'refused',
        target_model: 'picked'
      },
      session_id: 'sess-1',
      type: 'session.updated'
    } as any)

    expect(sys).toHaveBeenCalledWith(expect.stringContaining('model_resolution_failed'))
    expect(sys.mock.calls[0][0]).toContain('picked')
    expect(getUiState().info?.model).toBe('frozen')
  })

  it('asks before applying a guarded pick, and the answer re-sends with the owner token', async () => {
    patchUiState({ info: { model: 'frozen' } as any, sid: 'sess-1' })
    const context = ctx()
    context.gateway.gw.request = vi.fn(async (method: string) =>
      method === 'session.resume'
        ? { execution_generation: 2, revision: 4 }
        : { model: 'guarded', session_id: 'sess-1' }
    )

    createGatewayEventHandler(context)({
      payload: {
        confirm: 'owner-token',
        confirm_message: 'Costly',
        operation: 'model',
        payload: { model: 'guarded' },
        session_id: 'sess-1',
        status: 'confirmation_required',
        target_model: 'guarded'
      },
      session_id: 'sess-1',
      type: 'session.updated'
    } as any)

    const ask = getOverlayState().confirm
    expect(ask?.detail).toBe('Costly')
    ask?.onConfirm()
    await vi.waitFor(() => expect(context.gateway.gw.request).toHaveBeenCalledTimes(2))
    expect(context.gateway.gw.request.mock.calls.at(-1)[1]).toMatchObject({
      operation: 'model',
      payload: { confirm: 'owner-token', model: 'guarded' }
    })
  })

  it('a confirmation refused again (a queued turn moved the session) is reported, not dropped', async () => {
    patchUiState({ info: { model: 'frozen' } as any, sid: 'sess-1' })
    const sys = vi.fn()
    const context = ctx(sys)
    context.gateway.gw.request = vi.fn(async (method: string) =>
      method === 'session.resume'
        ? { execution_generation: 2, revision: 4 }
        : {
            confirm: 'fresh-token',
            confirm_message: 'Still costly',
            operation: 'model',
            session_id: 'sess-1',
            status: 'confirmation_required',
            target_model: 'guarded'
          }
    )

    createGatewayEventHandler(context)({
      payload: {
        confirm: 'old-token',
        operation: 'model',
        payload: { model: 'guarded' },
        session_id: 'sess-1',
        status: 'confirmation_required',
        target_model: 'guarded'
      },
      session_id: 'sess-1',
      type: 'session.updated'
    } as any)

    getOverlayState().confirm?.onConfirm()
    await vi.waitFor(() => expect(sys).toHaveBeenCalled())
    expect(sys.mock.calls[0][0]).toContain('run /model again')
    expect(getUiState().info?.model).toBe('frozen')
  })
})
