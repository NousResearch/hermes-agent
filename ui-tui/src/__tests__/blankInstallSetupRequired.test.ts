import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { resetTurnState } from '../app/turnStore.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { setupRequiredTitle } from '../content/setup.js'
import { t } from '../i18n/runtime.js'

interface Owner {
  providerConfigured: boolean
  resume?: 'fails' | 'resumes'
  runtimeCheck?: { model?: null | string; ok: boolean }
}

const SESSION = { info: { cwd: '/w', model: 'test', skills: {}, tools: {} }, messages: [], session_id: 's1' }
const mounted: ReturnType<typeof renderSync>[] = []

// `hermes --tui` attaches to the profile's shared gateway, which mints no session for a blank install:
// the TUI asks it whether a provider is configured before creating one.
function mountCanonicalLifecycle({ providerConfigured, resume = 'resumes', runtimeCheck }: Owner) {
  let api: null | ReturnType<typeof useSessionLifecycle> = null
  const panel = vi.fn()
  const sys = vi.fn()

  const request = vi.fn(async (method: string, _params?: Record<string, unknown>) => {
    if (method === 'setup.status') {
      return { provider_configured: providerConfigured }
    }

    if (method === 'setup.runtime_check') {
      return runtimeCheck ?? { error: 'No usable credentials found.', ok: false }
    }

    if (method === 'session.resume' && resume === 'fails') {
      throw new Error('storage_unavailable')
    }

    return SESSION
  })

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
      sys
    })

    useEffect(() => {
      api = lifecycle
    })

    return null
  }

  const stream = () => Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })

  mounted.push(
    renderSync(React.createElement(Probe), {
      patchConsole: false,
      stderr: stream() as unknown as NodeJS.WriteStream,
      stdin: stream() as unknown as NodeJS.ReadStream,
      stdout: stream() as unknown as NodeJS.WriteStream
    })
  )

  const methods = () => request.mock.calls.map(([method]) => method)
  const created = () => request.mock.calls.find(([method]) => method === 'session.create')?.[1]

  return { api: () => api!, created, methods, panel, sys }
}

const expectSetupRequired = (panel: ReturnType<typeof vi.fn>) => {
  expect(panel).toHaveBeenCalledWith(setupRequiredTitle(), expect.any(Array))
  expect(getUiState().status).toBe(t('session.status.setupRequired'))
}

beforeEach(() => {
  resetUiState()
  resetTurnState()
})

afterEach(() => {
  vi.unstubAllEnvs()

  for (const view of mounted.splice(0)) {
    view.unmount()
    view.cleanup()
  }
})

describe('a blank install on the shared gateway', () => {
  it('opens Setup Required instead of creating a session', async () => {
    const { api, methods, panel } = mountCanonicalLifecycle({ providerConfigured: false })

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().newSession()

    expectSetupRequired(panel)
    expect(methods()).not.toContain('session.create')
  })

  it('opens Setup Required when resuming fails', async () => {
    const { api, panel, sys } = mountCanonicalLifecycle({ providerConfigured: false, resume: 'fails' })

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().resumeById('stored')

    expect(sys).toHaveBeenCalledWith('error: storage_unavailable')
    expectSetupRequired(panel)
  })

  it('creates the session once a provider is configured', async () => {
    const { api, panel } = mountCanonicalLifecycle({ providerConfigured: true })

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().newSession()

    expect(panel).not.toHaveBeenCalled()
    expect(getUiState().sid).toBe('s1')
  })
})

// A keyless `providers:` endpoint saves no `model:` block, so `setup.status` calls the profile
// unconfigured while `hermes --tui --provider local -m qwen`, and sessions stored on that route, work.
describe('an unconfigured profile with a working route', () => {
  it.each([
    ['the launch', 'qwen', null],
    ['the profile', undefined, 'qwen']
  ])('creates on the launch provider with a model from %s', async (_from, launchModel, profileModel) => {
    vi.stubEnv('HERMES_TUI_PROVIDER', 'local')
    vi.stubEnv('HERMES_MODEL', launchModel)

    const { api, created, panel } = mountCanonicalLifecycle({
      providerConfigured: false,
      runtimeCheck: { model: profileModel, ok: true }
    })

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().newSession()

    expect(panel).not.toHaveBeenCalled()
    expect(created()).toMatchObject({ provider: 'local' })
    expect(getUiState().sid).toBe('s1')
  })

  it.each([
    ['resolves with no model anywhere', 'local', undefined, { model: null, ok: true }],
    ['does not resolve', 'local', 'qwen', { model: null, ok: false }],
    ['is not named', undefined, 'qwen', { model: 'qwen', ok: true }]
  ])('opens Setup Required when the launch provider %s', async (_case, provider, model, runtimeCheck) => {
    vi.stubEnv('HERMES_TUI_PROVIDER', provider)
    vi.stubEnv('HERMES_MODEL', model)

    const { api, methods, panel } = mountCanonicalLifecycle({ providerConfigured: false, runtimeCheck })

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().newSession()

    expectSetupRequired(panel)
    expect(methods()).not.toContain('session.create')
  })

  it('resumes a stored session on its own route', async () => {
    const { api, panel } = mountCanonicalLifecycle({ providerConfigured: false })

    await vi.waitFor(() => expect(api()).toBeTruthy())
    await api().resumeById('stored')

    expect(panel).not.toHaveBeenCalled()
    expect(getUiState().sid).toBe('s1')
  })
})
