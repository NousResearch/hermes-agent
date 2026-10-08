import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

/** Mount the real hook and hand its API to the test once the first commit is done. */
function mountLifecycle(
  request: (method: string, params: unknown) => Promise<unknown>,
  rpc: (method: string, params: unknown) => Promise<unknown> = async () => null
) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null
  const setHistoryItems = vi.fn()

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

  const app = renderSync(React.createElement(Probe), {
    patchConsole: false,
    stderr: stream() as unknown as NodeJS.WriteStream,
    stdin: stream() as unknown as NodeJS.ReadStream,
    stdout: stream() as unknown as NodeJS.WriteStream
  })

  return { api: () => api!, setHistoryItems, unmount: () => app.unmount() }
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

    const mounted = mountLifecycle(request)

    await vi.waitFor(() => expect(mounted.api()).toBeTruthy())
    mounted.api().activateLiveSession('durable-key-123')

    await vi.waitFor(() => expect(getUiState().sid).toBe('runtime-42'))
    expect(request).toHaveBeenCalledWith('session.activate', { session_id: 'durable-key-123' })
    expect(getUiState().storedSid).toBe('durable-key-123')
    mounted.unmount()
  })

  it('refreshes a resumed transcript when another surface persists new messages', async () => {
    let followTick: null | (() => void) = null

    const intervalSpy = vi.spyOn(globalThis, 'setInterval').mockImplementation((handler, timeout) => {
      if (timeout === 2000) {
        followTick = handler as () => void
      }

      return 1 as unknown as ReturnType<typeof setInterval>
    })

    const request = vi.fn(async () => ({
      info: { cwd: '/tmp/w', model: 'test', skills: {}, tools: {} },
      message_count: 1,
      messages: [{ role: 'user', text: 'before' }],
      resumed: 'durable-key-123',
      running: false,
      session_id: 'runtime-42',
      status: 'idle'
    }))

    const rpc = vi.fn(async (method: string) => {
      if (method === 'setup.status') {
        return { provider_configured: true }
      }

      if (method === 'session.history') {
        return {
          count: 2,
          messages: [
            { role: 'user', text: 'before' },
            { role: 'assistant', text: 'arrived elsewhere' }
          ]
        }
      }

      return null
    })

    const mounted = mountLifecycle(request, rpc)

    await vi.waitFor(() => expect(mounted.api()).toBeTruthy())
    mounted.api().resumeById('durable-key-123')
    await vi.waitFor(() => expect(getUiState().sid).toBe('runtime-42'))
    expect(followTick).toBeTypeOf('function')

    followTick!()
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('session.history', { session_id: 'runtime-42' }))

    const update = mounted.setHistoryItems.mock.calls.at(-1)?.[0]
    expect(update([{ info: {}, kind: 'intro', role: 'system', text: '' }])).toEqual([
      { info: {}, kind: 'intro', role: 'system', text: '' },
      { role: 'user', text: 'before' },
      { role: 'assistant', text: 'arrived elsewhere' }
    ])

    mounted.unmount()
    intervalSpy.mockRestore()
  })

  it('keeps a local turn visible when a follow request resolves after the UI becomes busy', async () => {
    let followTick: null | (() => void) = null
    let resolveHistory: null | ((snapshot: { count: number; messages: unknown[] }) => void) = null

    const intervalSpy = vi.spyOn(globalThis, 'setInterval').mockImplementation((handler, timeout) => {
      if (timeout === 2000) {
        followTick = handler as () => void
      }

      return 1 as unknown as ReturnType<typeof setInterval>
    })

    const history = new Promise<{ count: number; messages: unknown[] }>(resolve => {
      resolveHistory = resolve
    })

    const request = vi.fn(async () => ({
      info: { cwd: '/tmp/w', model: 'test', skills: {}, tools: {} },
      message_count: 1,
      messages: [{ role: 'user', text: 'before' }],
      resumed: 'durable-key-123',
      running: false,
      session_id: 'runtime-42',
      status: 'idle'
    }))

    const rpc = vi.fn(async (method: string) => {
      if (method === 'setup.status') {
        return { provider_configured: true }
      }

      return method === 'session.history' ? history : null
    })

    const mounted = mountLifecycle(request, rpc)

    await vi.waitFor(() => expect(mounted.api()).toBeTruthy())
    mounted.api().resumeById('durable-key-123')
    await vi.waitFor(() => expect(getUiState().sid).toBe('runtime-42'))
    mounted.setHistoryItems.mockClear()

    followTick!()
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('session.history', { session_id: 'runtime-42' }))
    patchUiState({ busy: true })
    resolveHistory!({
      count: 2,
      messages: [
        { role: 'user', text: 'before' },
        { role: 'assistant', text: 'stale follower snapshot' }
      ]
    })
    await history
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(mounted.setHistoryItems).not.toHaveBeenCalled()

    mounted.unmount()
    intervalSpy.mockRestore()
  })
})
