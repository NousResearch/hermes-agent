import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

const info = { cwd: '/w', model: 'test', skills: {}, tools: {} }

const deferred = <T>() => {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((res, rej) => ((resolve = res), (reject = rej)))

  return { promise, reject, resolve }
}

/** The real lifecycle hook and the real gateway.ready handler sharing one recovery ref,
 * as useMainApp wires them, over an owner whose answers the test controls. */
function mount(answer: (method: string, params: any) => Promise<unknown>) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null
  const recoverSidRef = { current: null as null | string }
  const request = vi.fn(answer)
  const gw = { isCanonical: true, request }

  const rpc = (async (method: string, params: any) => {
    try {
      return await request(method, params)
    } catch {
      return null
    }
  }) as any

  function Probe() {
    const lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() } as any,
      gw: gw as any,
      panel: vi.fn(),
      recoverSidRef,
      rpc,
      scrollRef: { current: null },
      setHistoryItems: vi.fn(),
      setLastUserMsg: vi.fn(),
      setSessionStartedAt: vi.fn(),
      setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(),
      setVoiceRecording: vi.fn(),
      sys: vi.fn()
    })

    useEffect(() => {
      api = lifecycle
    })

    return null
  }

  const stream = () => Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })

  renderSync(React.createElement(Probe), {
    patchConsole: false,
    stderr: stream() as unknown as NodeJS.WriteStream,
    stdin: stream() as unknown as NodeJS.ReadStream,
    stdout: stream() as unknown as NodeJS.WriteStream
  })

  const ready = () =>
    createGatewayEventHandler({
      composer: { setInput: vi.fn() },
      gateway: { gw, rpc },
      session: {
        STARTUP_RESUME_ID: '',
        colsRef: { current: 80 },
        newSession: (...args: any[]) => api!.newSession(...args),
        recoverSidRef,
        resetSession: vi.fn(),
        resumeById: (id: string) => api!.resumeById(id),
        setCatalog: vi.fn()
      },
      submission: { submitLiteralRef: { current: vi.fn() }, submitRef: { current: vi.fn() } },
      system: { bellOnComplete: false, sys: vi.fn() },
      transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
      voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
    } as any)({ payload: {}, type: 'gateway.ready' } as any)

  return { api: () => api!, ready, recoverSidRef, request }
}

const resumed = (session_id: string) => ({ info, messages: [], session_id, session_key: session_id, status: 'idle' })

// Socket closed on session "old": the exit handler armed recovery and marked the view
// disconnected. The user types /new during "reconnecting…"; its frame forces the
// reconnect, and gateway.ready is published after runtime.describe.
describe('a user attach during reconnect supersedes the recovery resume', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ gatewayConnected: false, sid: 'old', storedSid: 'old' })
  })

  it('/new answered after gateway.ready keeps the new session', async () => {
    const create = deferred<unknown>()

    const h = mount(async method =>
      method === 'session.create' ? create.promise : method === 'session.resume' ? resumed('old') : null
    )

    h.recoverSidRef.current = 'old'
    await vi.waitFor(() => expect(h.api()).toBeTruthy())

    const minted = h.api().newSession()
    h.ready()
    create.resolve({ info, session_id: 'new', stored_session_id: 'new' })

    expect(await minted).toBe('new')
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(getUiState().sid).toBe('new')
    expect(h.request.mock.calls.filter(([method]) => method === 'session.resume')).toEqual([])
    expect(h.request.mock.calls.filter(([method]) => method === 'session.create')).toHaveLength(1)
    expect(h.request).not.toHaveBeenCalledWith('session.detach', expect.objectContaining({ session_id: 'new' }))
    expect(h.recoverSidRef.current).toBeNull()
  })

  it('a refused /new hands the target back and recovers the old session', async () => {
    const create = deferred<unknown>()

    const h = mount(async method =>
      method === 'session.create' ? create.promise : method === 'session.resume' ? resumed('old') : null
    )

    h.recoverSidRef.current = 'old'
    await vi.waitFor(() => expect(h.api()).toBeTruthy())

    const minted = h.api().newSession()
    h.ready()
    create.reject(new Error('runtime_draining'))

    expect(await minted).toBeNull()
    await vi.waitFor(() =>
      expect(h.request).toHaveBeenCalledWith('session.resume', expect.objectContaining({ session_id: 'old' }))
    )
    await vi.waitFor(() => expect(getUiState().gatewayConnected).toBe(true))
    expect(getUiState().sid).toBe('old')
    expect(h.recoverSidRef.current).toBeNull()
  })
})
