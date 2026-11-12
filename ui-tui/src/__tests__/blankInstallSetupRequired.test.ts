import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { resetTurnState } from '../app/turnStore.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { setupRequiredTitle } from '../content/setup.js'
import { t } from '../i18n/runtime.js'

// `hermes --tui` attaches to the profile's shared gateway, which mints no session for a blank install:
// the TUI asks it whether a provider is configured before creating or resuming one.
function mountCanonicalLifecycle(providerConfigured: boolean) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null
  const panel = vi.fn()

  const request = vi.fn(async (method: string) =>
    method === 'setup.status'
      ? { provider_configured: providerConfigured }
      : { info: { cwd: '/w', model: 'test', skills: {}, tools: {} }, messages: [], session_id: 's1' }
  )

  function Probe() {
    const lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() } as any,
      gw: { isCanonical: true, request } as any,
      panel,
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

  const methods = () => request.mock.calls.map(([method]) => method)

  return { api: () => api!, methods, panel }
}

describe('a blank install on the shared gateway', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
  })

  it('opens Setup Required instead of creating a session', async () => {
    const { api, methods, panel } = mountCanonicalLifecycle(false)

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().newSession()

    expect(panel).toHaveBeenCalledWith(setupRequiredTitle(), expect.any(Array))
    expect(getUiState().status).toBe(t('session.status.setupRequired'))
    expect(methods()).not.toContain('session.create')
  })

  it('opens Setup Required instead of resuming a session', async () => {
    const { api, methods, panel } = mountCanonicalLifecycle(false)

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().resumeById('stored')

    expect(panel).toHaveBeenCalledWith(setupRequiredTitle(), expect.any(Array))
    expect(methods()).not.toContain('session.resume')
  })

  it('creates the session once a provider is configured', async () => {
    const { api, panel } = mountCanonicalLifecycle(true)

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().newSession()

    expect(panel).not.toHaveBeenCalled()
    expect(getUiState().sid).toBe('s1')
  })
})
