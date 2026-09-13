import { beforeEach, describe, expect, it } from 'vitest'

import { getOverlayState, patchOverlayState } from '../app/overlayStore.js'
import { restorePendingPrompt } from '../app/pendingPromptOverlay.js'
import type { SessionResumeResponse } from '../gatewayTypes.js'

const base = (): SessionResumeResponse => ({
  messages: [],
  session_id: 'sid-1'
})

describe('restorePendingPrompt — reopen a prompt still blocking a resumed session', () => {
  beforeEach(() => {
    patchOverlayState({ clarify: null, secret: null, sudo: null })
  })

  it('rebuilds the clarify picker with the original question and choices', () => {
    restorePendingPrompt({
      ...base(),
      pending_clarify: {
        choices: ['staging', 'production'],
        question: 'Which deployment target?',
        request_id: 'rid-clarify'
      }
    })

    expect(getOverlayState().clarify).toEqual({
      choices: ['staging', 'production'],
      question: 'Which deployment target?',
      requestId: 'rid-clarify'
    })
  })

  it('carries the request_id so the restored overlay answers the original request', () => {
    restorePendingPrompt({ ...base(), pending_sudo: { request_id: 'rid-sudo' } })
    expect(getOverlayState().sudo).toEqual({ requestId: 'rid-sudo' })
  })

  it('restores a secret prompt with its env var and prompt text', () => {
    restorePendingPrompt({
      ...base(),
      pending_secret: {
        env_var: 'OPENAI_API_KEY',
        prompt: 'Paste your key',
        request_id: 'rid-secret'
      }
    })

    expect(getOverlayState().secret).toEqual({
      envVar: 'OPENAI_API_KEY',
      prompt: 'Paste your key',
      requestId: 'rid-secret'
    })
  })

  it('tolerates a clarify payload with no choices — the picker still opens', () => {
    restorePendingPrompt({
      ...base(),
      pending_clarify: { question: 'Free-form?', request_id: 'rid-open' }
    })

    expect(getOverlayState().clarify).toEqual({
      choices: null,
      question: 'Free-form?',
      requestId: 'rid-open'
    })
  })

  it('leaves every overlay untouched when nothing is pending', () => {
    restorePendingPrompt(base())

    const state = getOverlayState()
    expect(state.clarify).toBeNull()
    expect(state.secret).toBeNull()
    expect(state.sudo).toBeNull()
  })
})
