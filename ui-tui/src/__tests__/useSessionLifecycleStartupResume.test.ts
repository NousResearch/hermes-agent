import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

const INFO = { cwd: '/tmp/w', lazy: false, model: 'test', skills: {}, tools: {}, version: '1' }

const RESUMED = {
  info: INFO,
  messages: [{ role: 'user', text: 'resumed question' }],
  running: false,
  session_id: 'resumed-runtime',
  status: 'idle',
  stored_session_id: 'resumed-stored'
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => (resolve = done))

  return { promise, resolve }
}

/** Mount the real hook and hand its API and history setter to the test. */
function mountLifecycle(rpc: (method: string, params: unknown) => Promise<unknown>) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null
  const setHistoryItems = vi.fn()
  const request = vi.fn(async (method: string) => (method === 'session.resume' ? RESUMED : null))

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

  return { api: () => api!, setHistoryItems }
}

describe('a /resume typed while the startup session is created (#121456)', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it('keeps the resumed session and closes the startup session created behind it', async () => {
    const create = deferred<unknown>()

    const rpc = vi.fn(async (method: string) => {
      if (method === 'setup.status') {
        return { provider_configured: true }
      }

      if (method === 'session.create') {
        return create.promise
      }

      return null
    })

    const { api, setHistoryItems } = mountLifecycle(rpc)

    await vi.waitFor(() => expect(api()).toBeTruthy())

    const startup = api().newSession()

    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('session.create', expect.anything()))
    await api().resumeById('resumed-stored')
    expect(getUiState().sid).toBe('resumed-runtime')

    create.resolve({ info: INFO, session_id: 'startup-runtime', stored_session_id: 'startup-stored' })

    expect(await startup).toBeNull()
    expect(getUiState().sid).toBe('resumed-runtime')
    expect(getUiState().storedSid).toBe('resumed-stored')
    expect(setHistoryItems).toHaveBeenLastCalledWith([
      expect.objectContaining({ kind: 'intro' }),
      expect.objectContaining({ role: 'user', text: 'resumed question' })
    ])
    expect(rpc).toHaveBeenCalledWith('session.close', { session_id: 'startup-runtime' })
    expect(rpc).not.toHaveBeenCalledWith('session.close', { session_id: 'resumed-runtime' })
  })

  it('does not create a session when the resume lands before the startup create starts', async () => {
    // Only the startup session's setup check is held; the resume's own check answers at once.
    const setup = deferred<unknown>()
    let setupCalls = 0

    const rpc = vi.fn(async (method: string) => {
      if (method !== 'setup.status') {
        return null
      }

      setupCalls += 1

      return setupCalls === 1 ? setup.promise : { provider_configured: true }
    })

    const { api } = mountLifecycle(rpc)

    await vi.waitFor(() => expect(api()).toBeTruthy())

    const startup = api().newSession()

    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('setup.status', {}))
    await api().resumeById('resumed-stored')
    expect(getUiState().sid).toBe('resumed-runtime')

    setup.resolve({ provider_configured: true })

    expect(await startup).toBeNull()
    expect(getUiState().sid).toBe('resumed-runtime')
    expect(rpc).not.toHaveBeenCalledWith('session.create', expect.anything())
    expect(rpc).not.toHaveBeenCalledWith('session.close', expect.anything())
  })
})
