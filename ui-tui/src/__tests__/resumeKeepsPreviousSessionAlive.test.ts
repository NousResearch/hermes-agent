// @vitest-environment jsdom
import React, { useEffect } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/ink', () => ({ evictInkCaches: vi.fn() }))

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

let root: null | Root = null

afterEach(() => {
  root?.unmount()
  root = null
})

/** Mount the real hook and hand its API to the test once effects have flushed. */
function mountLifecycle(opts: {
  request: (method: string, params: unknown) => Promise<unknown>
  rpc: (method: string, params: unknown) => Promise<unknown>
}) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null

  function Probe() {
    const lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() } as any,
      gw: { request: opts.request } as any,
      panel: vi.fn(),
      rpc: opts.rpc as any,
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

  root = createRoot(document.createElement('div'))
  root.render(React.createElement(Probe))

  return () => api!
}

// Regression test for #121121: resuming a history session must not close the
// previous session. session.close tears down the server session and hard
// interrupts its running delegate_task subagents. Switching to a live session
// (activateLiveSession) only attaches and never closes; resume must match.
describe('resume keeps the previous session alive (#121121)', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it('resumeById does not issue session.close for the session being left', async () => {
    const rpcCalls: Array<[string, unknown]> = []
    const rpc = vi.fn(async (method: string, params: unknown) => {
      rpcCalls.push([method, params])

      if (method === 'setup.status') {
        return { provider_configured: true }
      }

      return null
    })
    const request = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: { cwd: '/tmp/w', lazy: false, model: 'test', skills: {}, tools: {} },
          messages: [],
          running: false,
          session_id: 'session-B',
          status: 'idle'
        }
      }

      return null
    })

    const api = mountLifecycle({ request, rpc })

    await vi.waitFor(() => expect(api()).toBeTruthy())
    patchUiState({ sid: 'session-A' })

    await api().resumeById('history-B')

    await vi.waitFor(() => expect(getUiState().sid).toBe('session-B'))
    // The implicit close (when present) fires in the same microtask chain as
    // the sid switch; yield so a trailing close would have landed by now.
    await new Promise(resolve => setTimeout(resolve, 50))

    expect(request).toHaveBeenCalledWith('session.resume', expect.objectContaining({ session_id: 'history-B' }))
    expect(rpcCalls.filter(([method]) => method === 'session.close')).toEqual([])
  })
})
