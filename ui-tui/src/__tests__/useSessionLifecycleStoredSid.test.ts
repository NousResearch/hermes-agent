import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

/** Mount the real hook and hand its API to the test once the first commit is done. */
function mountLifecycle(
  request: (method: string, params: unknown) => Promise<unknown>,
  rpc: (method: string, params: unknown) => Promise<unknown> = async () => null,
  setHistoryItems = vi.fn()
) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null

  function Probe() {
    const lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() } as any,
      gw: { request } as any,
      panel: vi.fn(),
      rpc: rpc as any,
      scrollRef: { current: null },
      setHistoryItems,
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

  return () => api!
}

describe('useSessionLifecycle durable session id', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it('activating an agent-less session records its session_key as the recovery target', async () => {
    const request = vi.fn(async () => ({
      // _fallback_session_info shape: no stored_session_id on the info object.
      info: { cwd: '/tmp/w', lazy: true, model: 'test', skills: {}, tools: {} },
      messages: [],
      running: false,
      session_id: 'runtime-42',
      session_key: 'durable-key-123',
      status: 'idle'
    }))

    const api = mountLifecycle(request)

    await vi.waitFor(() => expect(api()).toBeTruthy())
    api().activateLiveSession('durable-key-123')

    await vi.waitFor(() => expect(getUiState().sid).toBe('runtime-42'))
    expect(request).toHaveBeenCalledWith('session.activate', { session_id: 'durable-key-123' })
    expect(getUiState().storedSid).toBe('durable-key-123')
  })

  it('keeps an early explicit resume when startup creation finishes later', async () => {
    let finishCreate!: (value: unknown) => void
    const created = new Promise<unknown>(resolve => { finishCreate = resolve })
    const rpc = vi.fn(async (method: string) => {
      if (method === 'session.create') return created
      if (method === 'setup.status') return { provider_configured: true }
      return null
    })
    const request = vi.fn(async () => ({
      info: null,
      messages: [{ role: 'user', text: 'saved conversation' }],
      session_id: 'resumed',
      status: 'idle',
      stored_session_id: 'resumed'
    }))
    const setHistoryItems = vi.fn()
    const api = mountLifecycle(request, rpc, setHistoryItems)
    await vi.waitFor(() => expect(api()).toBeTruthy())

    const startup = api().newSession(undefined, undefined, true)
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('session.create', { cols: 80 }))
    await api().resumeById('resumed')
    finishCreate({ session_id: 'startup', stored_session_id: 'startup' })
    await startup

    expect(getUiState().sid).toBe('resumed')
    expect(getUiState().storedSid).toBe('resumed')
    expect(setHistoryItems).toHaveBeenLastCalledWith([{ role: 'user', text: 'saved conversation' }])
    expect(rpc).toHaveBeenCalledWith('session.close', { session_id: 'startup' })
    expect(rpc).not.toHaveBeenCalledWith('session.close', { session_id: 'resumed' })
  })

  it('does not create a startup session after an explicit resume has begun', async () => {
    const rpc = vi.fn(async (method: string) => method === 'setup.status' ? { provider_configured: true } : null)
    const request = vi.fn(async () => ({ messages: [], session_id: 'resumed', status: 'idle' }))
    const api = mountLifecycle(request, rpc)
    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().resumeById('resumed')
    await api().newSession(undefined, undefined, true)
    expect(getUiState().sid).toBe('resumed')
    expect(rpc).not.toHaveBeenCalledWith('session.create', expect.anything())
  })

  it('keeps an explicit resume when an auto-resume response arrives later', async () => {
    let finishRecent!: (value: unknown) => void
    const recent = new Promise<unknown>(resolve => { finishRecent = resolve })
    const request = vi.fn(async (method: string) => method === 'session.resume' ? recent : null)
    const rpc = vi.fn(async () => ({ provider_configured: true }))
    const setHistoryItems = vi.fn()
    const api = mountLifecycle(request, rpc, setHistoryItems)
    await vi.waitFor(() => expect(api()).toBeTruthy())

    const autoResume = api().resumeById('recent', true)
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.resume', { cols: 80, session_id: 'recent' }))
    request.mockImplementationOnce(async () => ({
      messages: [{ role: 'user', text: 'chosen conversation' }],
      session_id: 'chosen',
      status: 'idle'
    }))
    await api().resumeById('chosen')
    finishRecent({ messages: [], session_id: 'recent', status: 'idle' })
    await autoResume

    expect(getUiState().sid).toBe('chosen')
    expect(getUiState().status).toBe('ready')
    expect(setHistoryItems).toHaveBeenLastCalledWith([{ role: 'user', text: 'chosen conversation' }])
    expect(api().canSelectStartupSession()).toBe(false)
  })
})
