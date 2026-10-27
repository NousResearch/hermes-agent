import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

const info = { cwd: '/w', model: 'test', skills: {}, tools: {} }

function mountLifecycle(answer: (method: string) => unknown) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null
  const request = vi.fn(async (method: string) => answer(method))

  function Probe() {
    const lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() } as any,
      gw: { isCanonical: true, request } as any,
      panel: vi.fn(),
      rpc: request as any,
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

// After a socket drop the exit handler sets gatewayConnected:false, which parks every
// plain prompt in the local queue (useSubmission) and stops draining (useQueue). Any
// bind the owner answered over the live socket proves the connection is back.
describe('a successful bind restores gatewayConnected', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ gatewayConnected: false, sid: 'old', storedSid: 'old' })
  })

  it('/new after a failed recovery resume leaves the composer sending again', async () => {
    const api = mountLifecycle(() => ({ info, session_id: 'new', stored_session_id: 'new' }))

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().newSession()

    expect(getUiState().sid).toBe('new')
    expect(getUiState().gatewayConnected).toBe(true)
  })

  it('activating a live session after a failed recovery resume leaves the composer sending again', async () => {
    const api = mountLifecycle(() => ({ info, messages: [], session_id: 'live', session_key: 'live', status: 'idle' }))

    await vi.waitFor(() => expect(api()).toBeTruthy())
    api().activateLiveSession('live')

    await vi.waitFor(() => expect(getUiState().sid).toBe('live'))
    expect(getUiState().gatewayConnected).toBe(true)
  })
})
