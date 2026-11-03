import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { beforeEach, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

const drain = { reason: 'update' as const, target: 'v0.99.0', waiting_for: { chat: 1 } }
const resumed = { info: { cwd: '/w', model: 'test', skills: {}, tools: {} }, messages: [], session_id: 's1', session_key: 's1', status: 'idle' }

function mountLifecycle(gw: Record<string, unknown>) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null

  function Probe() {
    const lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() } as any,
      gw: gw as any,
      panel: vi.fn(),
      rpc: gw.request as any,
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

beforeEach(() => {
  resetUiState()
  resetTurnState()
  turnController.fullReset()
})

// Reconnecting to an owner that drains for an update resumes the session after the ready that
// announced the update; the resume's reset must not leave the user with a bare "ready".
it('a session resumed on a draining owner still names the update it waits for', async () => {
  const api = mountLifecycle({
    isCanonical: true,
    refreshUpdateDrain: vi.fn(async () => drain),
    request: vi.fn(async () => resumed)
  })

  await vi.waitFor(() => expect(api()).toBeTruthy())
  await api().resumeById('s1')

  await vi.waitFor(() => expect(getUiState().status).toBe('Updating to v0.99.0 · finishing 1 turn'))
  expect(getUiState().sid).toBe('s1')
  expect(getTurnState().activity.map(a => a.text)).toContain(
    'Updating to v0.99.0 · finishing 1 turn · new messages wait for the updated Hermes')
})
