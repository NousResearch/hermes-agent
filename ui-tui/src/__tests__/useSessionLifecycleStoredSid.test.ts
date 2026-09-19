import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

/** Mount the real hook and hand its API to the test once the first commit is done. */
function mountLifecycle(request: (method: string, params: unknown) => Promise<unknown>) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null

  function Probe() {
    const lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() } as any,
      gw: { request } as any,
      panel: vi.fn(),
      rpc: vi.fn(async () => null),
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
})
