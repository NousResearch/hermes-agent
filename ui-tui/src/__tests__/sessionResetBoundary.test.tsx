import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

const noop = () => {}

it.each(['reset', 'keep', 'initial', 'failed-create'] as const)('legacy new-session boundary: %s', async mode => {
  resetUiState()
  patchUiState({ sid: mode === 'initial' ? null : 'old-runtime', storedSid: 'old-stored' })
  const sys = vi.fn()

  const rpc = vi.fn(async (method: string) => {
    if (method === 'setup.status') {
      return { provider_configured: true }
    }

    if (method === 'session.close') {
      return { closed: true, messages: ['finalized predecessor'] }
    }

    return mode === 'failed-create' ? null : { session_id: 'new-runtime', stored_session_id: 'new-stored', info: null }
  })

  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Probe() {
    lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: noop },
      gw: { isCanonical: false },
      rpc,
      scrollRef: { current: null },
      panel: noop,
      sys,
      setHistoryItems: noop,
      setLastUserMsg: noop,
      setSessionStartedAt: noop,
      setStickyPrompt: noop,
      setVoiceProcessing: noop,
      setVoiceRecording: noop
    } as any)

    return null
  }

  const stream = () => Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false })

  const view = renderSync(<Probe />, {
    stdin: stream() as any,
    stdout: stream() as any,
    stderr: stream() as any,
    patchConsole: false
  })

  try {
    await new Promise(resolve => setImmediate(resolve))

    if (mode === 'keep') {
      await lifecycle.newLiveSession()
    } else {
      await lifecycle.newSession()
    }

    if (mode === 'reset') {
      expect(rpc.mock.calls).toEqual([
        ['setup.status', {}],
        ['session.create', expect.objectContaining({ cols: 80 })],
        ['session.close', { session_id: 'old-runtime', reset_to_session_id: 'new-runtime' }]
      ])
      expect(sys).toHaveBeenCalledWith('finalized predecessor')
    } else {
      expect(rpc.mock.calls.some(([method]) => method === 'session.close')).toBe(false)
    }

    expect(getUiState().sid).toBe(mode === 'failed-create' ? 'old-runtime' : 'new-runtime')
  } finally {
    view.unmount()
    view.cleanup()
    turnController.fullReset()
    resetUiState()
  }
})
