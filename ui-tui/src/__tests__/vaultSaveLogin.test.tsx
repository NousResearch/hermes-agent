import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { dismissSensitivePrompt } from '../app/useInputHandlers.js'
import { submitVaultSaveLogin } from '../app/vaultSaveLogin.js'
import { VaultSaveLoginPrompt } from '../components/vaultSaveLoginPrompt.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const ref = <T,>(current: T) => ({ current })

const buildCtx = () =>
  ({
    composer: { dequeue: () => undefined, queueEditRef: ref<null | number>(null), sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn() }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: ref(vi.fn()) },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: {
      appendMessage: (_message: Msg) => undefined,
      panel: vi.fn(),
      setHistoryItems: vi.fn()
    },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

const render = (tree: React.ReactElement) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: true, ref: () => {}, setRawMode: () => {}, unref: () => {} })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(tree, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  return { output: () => output, unmount: () => (instance.unmount(), instance.cleanup()) }
}

describe('vault save-login prompt', () => {
  beforeEach(() => {
    resetOverlayState()
    resetTurnState()
    resetUiState()
    patchUiState({ showReasoning: true })
  })

  afterEach(() => vi.restoreAllMocks())

  it('opens an origin-bound masked form, responds without transcript data, and clears only its expiry', async () => {
    const onEvent = createGatewayEventHandler(buildCtx())
    onEvent({
      payload: { origin: 'https://github.com', request_id: 'login-1', site: 'github.com' },
      type: 'vault.save_login.request'
    } as any)

    expect(getOverlayState().vaultSaveLogin).toEqual({
      origin: 'https://github.com',
      requestId: 'login-1',
      site: 'github.com'
    })

    const view = render(
      <VaultSaveLoginPrompt
        cols={80}
        onSubmit={vi.fn()}
        request={getOverlayState().vaultSaveLogin!}
        t={DEFAULT_THEME}
      />
    )
    expect(view.output()).toContain('github.com')
    expect(view.output()).toContain('https://github.com')
    expect(view.output()).toContain('password · hidden')
    view.unmount()

    const rpc = vi.fn(async () => ({ status: 'ok' }))
    await submitVaultSaveLogin(rpc, 'login-1', 'tek@acme.test', 'fixture-pw')
    expect(rpc).toHaveBeenCalledWith('vault.save_login.respond', {
      login: JSON.stringify({ identifier: 'tek@acme.test', password: 'fixture-pw' }),
      request_id: 'login-1'
    })

    onEvent({ payload: { request_id: 'other-login' }, type: 'vault.save_login.expire' } as any)
    expect(getOverlayState().vaultSaveLogin?.requestId).toBe('login-1')
    onEvent({ payload: { request_id: 'login-1' }, type: 'vault.save_login.expire' } as any)
    expect(getOverlayState().vaultSaveLogin).toBeNull()

    await submitVaultSaveLogin(rpc, 'login-2')
    expect(rpc).toHaveBeenLastCalledWith('vault.save_login.respond', { login: '', request_id: 'login-2' })

    patchOverlayState({ vaultSaveLogin: { origin: 'https://gitlab.com', requestId: 'login-3', site: 'gitlab.com' } } as any)
    await dismissSensitivePrompt(getOverlayState(), rpc, vi.fn())
    expect(getOverlayState().vaultSaveLogin).toBeNull()
    expect(rpc).toHaveBeenLastCalledWith('vault.save_login.respond', { login: '', request_id: 'login-3' })
  })
})
